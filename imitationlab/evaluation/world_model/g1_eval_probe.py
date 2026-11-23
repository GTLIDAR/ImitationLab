# Ported from NDM Gym (https://github.com/Frank-ZY-Dou/ndm-gym @ 05ea9f3,
# ndm_gym/algos/eprobe.py, ndm_gym/sim/rigid/reset_stats.py), which ports it from
# BlinkMimic (https://github.com/fuyichen0725/blinkmimic @ d264658,
# src/blinkmimic/train.py: --eprobe_*), under the MIT License:
#
# Copyright (c) 2026 NDM Gym Contributors
# Copyright (c) 2026 Fuyi Chen
# Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which this software is derived)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Evaluation-guided resets for WMBPTT on ``imitation-g1-v1`` (``wm.eprobe_share``).

``wm.eprobe_max_worlds > 0`` selects a bounded rotating sample instead of the
full sweep below. Its observed-bin failure-rate EMA keeps unvisited history;
it does not estimate whole-bank success. Sampled execution is synchronous
(``eprobe_async=0``) until overlap is qualified. On a host-mapped bank enable
``env.library_host_shared`` so the worker attaches to the trainer's rows.

The port of NDM Gym's ``--eprobe_*`` options. From iteration ``eprobe_start``,
every ``eprobe_every`` iterations, the deployed policy (the ``actor_ema`` average,
deterministic, with the residual setpoint of ``actor_residual``) is swept over
every clip under evaluation conditions: one probe world per ``eprobe_tile``
frames of each clip, started a random lead (one per sweep, in
[``eprobe_lead_min``, ``eprobe_lead_max``]) frames earlier on the reference, run to
the tile's end. The probes' first failures (termination, or a non-finite action)
add to per-bin counts that decay by ``eprobe_decay`` per sweep. Each bin weighs
its own count and the ``eprobe_win`` bins after it in its clip, which is the
window before each failure. ``eprobe_share`` of the env's training resets follow
these weights (``G1ResetSampler.set_focus``).

Isaac Lab holds one ``SimulationContext`` per process, so the sweep runs in a
worker process on the same GPU, as in NDM Gym:
``python -m imitationlab.evaluation.world_model.g1_eval_probe --worker <dir>
<spec.json>``. The worker builds the replica readout's env
(``imitation-g1-v1`` with ``env.reset_noise=false env.observation_noise=false
env.randomize_context=false``, the trainer's ``physics=`` and ``env.data.*``
overrides) with one world per tile, and acts with
``imitationlab.algorithms.common.world_model.learner.deployed_policy``, the policy ``load_policy`` gives an
evaluator. ``eprobe_graph`` runs the same task with
``env.step_mode=graph`` and every lead + tile step of a sweep, as NDM Gym's
``--eprobe_graph``; without it the eager step runs until every probe is done.

The trainer sends the deployed policy at the sweep's start and applies the
failures ``ceil(steps / eprobe_async)`` iterations later (steps = lead + tile, the
first sweep three fewer, BlinkMimic's graph warm-up), before that iteration's
collection, waiting for the worker if it is not done; with ``eprobe_async`` 0, at
once. The checkpoint keeps the counts, the focus and its share; a sweep in flight
is dropped, as in NDM Gym.

Index mapping. NDM Gym numbers frames over the concatenated clips and its 50-frame
bins clip by clip; here a frame is (trajectory rank, local frame) and the bins are
``G1ResetSampler``'s, ``bin_offsets[rank] + frame // 50``, the same partition in
the same order. A failure judged at f + 1 is charged to the pre-step frame f, as
NDM Gym's ``f_now`` and ``reset_fail_share`` do.

Differences from NDM Gym, each deliberate:

- The lead is drawn from the probe's own CPU generator, seeded from the env seed,
  not from torch's default generator, so the trainer's other draws do not move.
- A probe world that fails or reaches its clip's end is reset by the env (NDM Gym
  keeps stepping it with its frame capped); a done world counts no further failure
  either way.
- The worker's policy step runs eagerly; NDM Gym captures it with the env step.
- Off (``eprobe_share`` 0), WMBPTT refuses every other ``eprobe_*`` setting.

On LAFAN1 the training and the evaluation clips are the same 40, so these resets
feed evaluation outcomes on the evaluated clips back into training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

READY, STOP = "ready", "stop"
#: The worker's task; ``eprobe_graph`` adds ``env.step_mode=graph``.
EVAL_TASK = "imitation-g1-v1"
#: The readout's evaluation conditions (``launch/score_local.sh``), less
#: ``pinned_reset``: the probe gives every world its own start. The policy reads
#: the ``wm`` group alone, so the worker computes no other.
EVAL_OVERRIDES = (
    "env.reset_noise=false",
    "env.observation_noise=false",
    "env.randomize_context=false",
    "env.wm_only_observations=true",
)
#: Trainer overrides the worker's env takes over: the physics preset and the bank.
CARRIED_PREFIXES = ("physics=", "presets=", "renderer=", "env.data.")
MEMORY_PREFIXES = ("env.library_", "env.graph_reference_source=")
#: BlinkMimic's graph warm-up: its first sweep applies three steps earlier.
FIRST_SWEEP_WARMUP = 3


# ---------------------------------------------------------------------------
# Sweep geometry and the failure focus (both processes).
# ---------------------------------------------------------------------------


def probe_tiles(lengths: Tensor, tile: int) -> tuple[Tensor, Tensor]:
    """(ranks, local frames) of the probe worlds' tile starts.

    Every ``tile`` frames of each clip, up to two before its end: NDM Gym's
    ``arange(clip_start, clip_end - 1, tile)`` with ``clip_end`` the last frame,
    in local frames ``arange(0, L - 2, tile)``.
    """
    ranks, frames = [], []
    for rank, length in enumerate(lengths.tolist()):
        local = torch.arange(0, int(length) - 2, int(tile), dtype=torch.long)
        ranks.append(torch.full_like(local, rank))
        frames.append(local)
    return torch.cat(ranks).to(lengths.device), torch.cat(frames).to(lengths.device)


def sweep_frames(
    lengths: Tensor, ranks: Tensor, tiles: Tensor, lead: int, tile: int
) -> tuple[Tensor, Tensor]:
    """(start, stop) local frames of a sweep with this lead.

    Each probe starts ``lead`` frames before its tile, not before its clip, and is
    done at the tile's end or the clip's last frame (NDM Gym's ``sweep_frames``).
    """
    last = lengths.to(tiles.device)[ranks] - 1
    start = (tiles - int(lead)).clamp_min(0)
    stop = torch.minimum(tiles + int(tile), last)
    return start, stop


def next_bins(bin_rank: Tensor) -> Tensor:
    """Each bin's next bin in its clip; a clip's last bin maps to itself."""
    n = bin_rank.numel()
    nxt = torch.arange(n, device=bin_rank.device) + 1
    same = (nxt < n) & (bin_rank[nxt.clamp(max=n - 1)] == bin_rank)
    return torch.where(same, nxt, nxt - 1)


class FailureFocus:
    """BlinkMimic's ``EP_FAIL`` and the focus made of it (NDM Gym's ``FailureFocus``)."""

    def __init__(self, bin_rank: Tensor, win: int, decay: float) -> None:
        self.win, self.decay = int(win), float(decay)
        self.counts = torch.zeros(bin_rank.numel(), device=bin_rank.device)
        self.next = next_bins(bin_rank)
        self._own = torch.arange(bin_rank.numel(), device=bin_rank.device)

    def update(
        self, fail_bins: Tensor, observed_bins: Tensor | None = None
    ) -> Tensor | None:
        """Full sweep: decay counts and add one per failure. Sampled sweep:
        update the failure-rate EMA in visited bins only.

        Returns the focus over bins (each bin: the counts of itself and the ``win``
        bins after it in its clip, normalized), or None while there is no failure.
        """
        fb = fail_bins.to(self.counts.device, torch.long)
        if observed_bins is None:
            self.counts.mul_(self.decay)
            self.counts.index_add_(0, fb, torch.ones_like(fb, dtype=self.counts.dtype))
        else:
            # Each world's visited bins count once. Success clears stale failures
            # only where the policy actually ran; partial sweeps do not erase the
            # rest of the bank. Repeated visits estimate a rate, not raw frequency.
            visits = torch.bincount(
                observed_bins.to(self.counts.device), minlength=self.counts.numel()
            )
            hits = torch.bincount(fb, minlength=self.counts.numel())
            seen = visits > 0
            self.counts[seen] = (
                self.decay * self.counts[seen]
                + (1 - self.decay) * hits[seen] / visits[seen]
            )
        if not self.counts.sum() > 0:
            return None
        w, q = self.counts.clone(), self.counts.clone()
        for _ in range(self.win):  # q[b] = the counts over b .. b + win (same clip)
            w = w[self.next] * (self.next != self._own).float()
            q += w
        return q / q.sum()


def apply_iteration(
    iteration: int, lead: int, tile: int, async_steps: int, *, graph: bool, first: bool
) -> int:
    """The iteration whose start applies a sweep started at ``iteration``.

    ``ceil(steps / async_steps)`` later, steps = lead + tile, three fewer for the
    first sweep of a graph worker (BlinkMimic's warm-up); the same iteration
    without ``async_steps``.
    """
    if not async_steps:
        return int(iteration)
    steps = int(lead) + int(tile) - (FIRST_SWEEP_WARMUP if graph and first else 0)
    return int(iteration) + math.ceil(steps / int(async_steps))


def carried_overrides(argv: list[str]) -> list[str]:
    """The trainer's Hydra overrides the worker's env needs: physics and bank."""
    return [
        arg for arg in argv if arg.startswith((*CARRIED_PREFIXES, *MEMORY_PREFIXES))
    ]


class TileSchedule:
    """Bounded, shuffled passes over eligible tiles, with no duplicate in a batch.

    A batch spanning a pass boundary consumes the old tail, then a fresh
    permutation with those tail IDs moved to its end. Thus no tile is omitted
    from either pass and each world in a batch represents a different tile.
    All randomness is on a private CPU generator.
    """

    def __init__(self, total: int, cap: int, seed: int) -> None:
        if total < 1 or cap < 1:
            raise ValueError(
                "sampled probes need at least one eligible tile and one world."
            )
        self.total, self.size = total, min(total, cap)
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(total, generator=self.generator)
        self.cursor = 0

    def draw(self) -> Tensor:
        tail = self.order[self.cursor : self.cursor + self.size]
        if tail.numel() == self.size:
            self.cursor += self.size
            return tail.clone()
        order = torch.randperm(self.total, generator=self.generator)
        overlap = torch.isin(order, tail)
        self.order = torch.cat((order[~overlap], order[overlap]))
        need = self.size - tail.numel()
        self.cursor = need
        return torch.cat((tail, self.order[:need]))

    def state_dict(self) -> dict[str, Any]:
        return {
            "order": self.order.clone(),
            "cursor": self.cursor,
            "rng": self.generator.get_state(),
            "size": self.size,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        order = state["order"].cpu()
        if (
            state["size"] != self.size
            or order.numel() != self.total
            or not torch.equal(order.sort().values, torch.arange(self.total))
            or not 0 <= state["cursor"] <= self.total
        ):
            raise ValueError(
                "the sampled probe checkpoint has incompatible tile geometry."
            )
        self.order, self.cursor = order.clone(), int(state["cursor"])
        self.generator.set_state(state["rng"].cpu())


# ---------------------------------------------------------------------------
# The trainer's side.
# ---------------------------------------------------------------------------


class EvalProbe:
    """The worker process, the sweeps' timing and the focus they set on the env.

    ``launch=False`` builds the trainer's state without a worker (tests answer the
    requests themselves).
    """

    def __init__(
        self, env: Any, wm: Any, run_dir: str | Path, *, launch: bool = True
    ) -> None:
        self.env = env
        self.share = float(wm.eprobe_share)
        self.tile = int(wm.eprobe_tile)
        self.lead_min, self.lead_max = int(wm.eprobe_lead_min), int(wm.eprobe_lead_max)
        self.every, self.start = int(wm.eprobe_every), int(wm.eprobe_start)
        self.graph, self.async_steps = bool(wm.eprobe_graph), int(wm.eprobe_async)
        sampler = env.g1_reset_sampler
        stepper = getattr(env, "_masked_stepper", None)
        if stepper is not None and getattr(stepper, "graphs", None) is not None:
            raise RuntimeError(
                "wm.eprobe_share: the env step was captured before the reset focus existed; "
                "a captured reset would never read it."
            )
        sampler.enable_focus()
        self.sampler = sampler
        self.lengths = sampler.lengths
        self.ranks, self.tiles = probe_tiles(self.lengths, self.tile)
        self.focus = FailureFocus(
            sampler.bin_rank, int(wm.eprobe_win), float(wm.eprobe_decay)
        )
        self.focus_set = False
        seed = int(getattr(getattr(env, "cfg", None), "seed", None) or 0)
        self.generator = torch.Generator().manual_seed(seed)
        cap = int(getattr(wm, "eprobe_max_worlds", 0))
        self.schedule = None
        if cap:
            self.all_ranks, self.all_tiles = self.ranks, self.tiles
            # Excluded training clips must not be reintroduced by probe resets.
            keep = getattr(sampler, "bin_keep", None)
            if keep is not None:
                eligible = keep[sampler.bin_ids(self.all_ranks, self.all_tiles)]
                self.all_ranks, self.all_tiles = (
                    self.all_ranks[eligible],
                    self.all_tiles[eligible],
                )
            self.schedule = TileSchedule(self.all_tiles.numel(), cap, seed + 1)
            self.ranks, self.tiles = (
                self.all_ranks[: self.schedule.size],
                self.all_tiles[: self.schedule.size],
            )
        self.k = 0  # requests sent
        self.pending: dict[str, Any] | None = None
        self.metrics: dict[str, float] = {}
        self.dir: str | None = None
        self.proc: subprocess.Popen | None = None
        self.log = None
        if launch:
            self._launch(Path(run_dir))

    @property
    def num_worlds(self) -> int:
        return int(self.tiles.numel())

    # -- the worker ----------------------------------------------------------
    def worker_spec(self) -> dict[str, Any]:
        """What the worker builds: its task, overrides, worlds and the bank's shape."""
        overrides = carried_overrides(sys.argv[1:]) + list(EVAL_OVERRIDES)
        if self.graph:
            overrides.append("env.step_mode=graph")
        return {
            "task": EVAL_TASK,
            "overrides": overrides,
            "num_envs": self.num_worlds,
            "device": str(self.env.device),
            "seed": int(getattr(self.env.cfg, "seed", None) or 0),
            "tile": self.tile,
            "max_steps": self.lead_max + self.tile,
            "graph": self.graph,
            "sampled": self.schedule is not None,
            "lengths": self.lengths.tolist(),
            "ranks": self.ranks.tolist(),
            "tiles": self.tiles.tolist(),
            "step_dt": float(self.env.step_dt),
            "shared_library": self._shared_library_spec(),
        }

    def _shared_library_spec(self) -> dict[str, Any] | None:
        library = getattr(self.env, "g1_library", None)
        path = getattr(library, "shared_file", "")
        if not path:
            return None
        return {"path": path, "lengths": self.lengths.cpu().tolist()}

    def _launch(self, run_dir: Path) -> None:
        if (
            self.schedule is not None
            and getattr(self.env, "g1_library", None) is not None
        ):
            library = self.env.g1_library
            if library.host_mapped and not library.shared_file:
                raise ValueError(
                    "sampled probes with host-mapped references need env.library_host_shared=true "
                    "to avoid a second full host bank in the worker."
                )
        if self.graph:
            from imitationlab.envs.isaaclab.envs.collect_fastpath import (
                ENV_VAR,
                fastpath_enabled,
            )
            from imitationlab.envs.isaaclab.envs.graph_step.graph import (
                CAPTURE_FAST_PATHS,
            )

            missing = [
                name for name in CAPTURE_FAST_PATHS if not fastpath_enabled(name)
            ]
            if missing:
                raise ValueError(
                    f"wm.eprobe_graph runs the captured env step, which needs {ENV_VAR}=1 ({missing} off)."
                )
        self.dir = tempfile.mkdtemp(prefix="eprobe_", dir=os.environ.get("TMPDIR"))
        spec_path = Path(self.dir) / "spec.json"
        spec_path.write_text(json.dumps(self.worker_spec()))
        run_dir.mkdir(parents=True, exist_ok=True)
        self.log = open(run_dir / "eprobe_worker.log", "a")  # noqa: SIM115 (closed in close)
        cmd = [
            sys.executable,
            "-m",
            "imitationlab.evaluation.world_model.g1_eval_probe",
            "--worker",
            self.dir,
            str(spec_path),
        ]
        self.proc = subprocess.Popen(
            cmd, stdout=self.log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
        )
        print(
            f"[INFO] eprobe: {self.num_worlds} probe worlds (tiles of {self.tile} frames) in worker pid "
            f"{self.proc.pid} ({self.dir}), log {run_dir / 'eprobe_worker.log'}",
            flush=True,
        )

    def _path(self, kind: str, k: int) -> str:
        assert self.dir is not None
        return os.path.join(self.dir, f"{kind}_{k:05d}.pt")

    def _check_worker(self) -> None:
        if self.proc is not None and self.proc.poll() is not None:
            if self.log is not None:
                self.log.flush()
            raise RuntimeError(
                f"the eprobe worker exited ({self.proc.returncode}); see eprobe_worker.log in the run's directory"
            )

    def _send(self, k: int, request: dict[str, Any]) -> None:
        path = self._path("req", k)
        torch.save(request, path + ".tmp")
        os.replace(path + ".tmp", path)

    def _wait(self, path: str) -> dict[str, Any]:
        while not os.path.exists(path):
            self._check_worker()
            time.sleep(0.005)
        return torch.load(path, map_location="cpu", weights_only=False)

    # -- the schedule --------------------------------------------------------
    def tick(self, iteration: int, payload: Callable[[], dict[str, Any]]) -> None:
        """Iteration ``iteration``, before its collection: a running sweep's failures
        applied when due, then a new sweep started on a sweep iteration (NDM Gym's
        ``EvalProbe.tick``, in its order). ``payload`` gives the deployed policy's
        ``policy_state_dict`` and ``policy_meta`` on the CPU."""
        self._check_worker()
        if self.pending is not None and iteration >= self.pending["due"]:
            self._apply(iteration)
        if iteration >= self.start and iteration % self.every == 0:
            if self.pending is not None:
                raise RuntimeError(
                    "wm.eprobe_async too small: a sweep is still pending."
                )
            lead = int(
                torch.randint(
                    self.lead_min, self.lead_max + 1, (1,), generator=self.generator
                )
            )
            geometry = {}
            if self.schedule is not None:
                selected = self.schedule.draw().to(self.all_ranks.device)
                self.ranks, self.tiles = (
                    self.all_ranks[selected],
                    self.all_tiles[selected],
                )
                geometry = {"ranks": self.ranks.cpu(), "tiles": self.tiles.cpu()}
            due = apply_iteration(
                iteration,
                lead,
                self.tile,
                self.async_steps,
                graph=self.graph,
                first=self.k == 0,
            )
            self._send(
                self.k,
                {**payload(), **geometry, "lead": lead, "iteration": int(iteration)},
            )
            self.pending = {
                "k": self.k,
                "iteration": int(iteration),
                "lead": lead,
                "due": due,
                "t0": time.monotonic(),
            }
            self.k += 1
            if not self.async_steps:
                self._apply(iteration)

    def _apply(self, iteration: int) -> None:
        pending = self.pending
        assert pending is not None
        t_wait = time.monotonic()
        result = self._wait(self._path("res", pending["k"]))
        wait_s = time.monotonic() - t_wait
        os.remove(self._path("res", pending["k"]))
        self.apply_result(result["fail_f"], lead=pending["lead"])
        self.pending = None
        failed = result["fail_f"] >= 0
        pairs = sorted(
            {
                (int(r), int(f))
                for r, f in zip(
                    self.ranks.cpu()[failed].tolist(),
                    result["fail_f"][failed].tolist(),
                    strict=True,
                )
            }
        )
        self.metrics.update(
            {
                "eprobe/sweep_s": float(result["sweep_s"]),
                "eprobe/wait_s": wait_s,
                "eprobe/start_to_apply_s": time.monotonic() - pending["t0"],
                "eprobe/applied_lag_iters": float(iteration - pending["iteration"]),
                "eprobe/worker_reserved_mib": float(
                    result.get("reserved_mib", float("nan"))
                ),
            }
        )
        print(
            f"[INFO] eprobe it {pending['iteration']}: lead {pending['lead']} | {int(failed.sum())}/{self.num_worlds} worlds "
            f"failed, at {len(pairs)} (rank, frame) {pairs[:8]} | focus {self.metrics['eprobe/focus_bins']:.0f} bins, "
            f"{self.metrics['eprobe/focus_mass']:.3f} of the resets; 90% of it in "
            f"{self.metrics['eprobe/focus_bins_90']:.0f} bins, {self.metrics['eprobe/focus_mass_90']:.3f} | sweep {result['sweep_s']:.1f} s "
            f"({result.get('steps', 0)} steps), {self.metrics['eprobe/start_to_apply_s']:.1f} s to apply, "
            f"waited {wait_s:.1f} s | applied at it {iteration}",
            flush=True,
        )

    def apply_result(self, fail_f: Tensor, *, lead: int = 0) -> Tensor | None:
        """One sweep's first-failure frames (local, -1 none; one per world) into the
        counts and the sampler's focus. Returns the focus, or None while none."""
        fail_f = fail_f.to(self.ranks.device, torch.long)
        if fail_f.shape != self.ranks.shape:
            raise ValueError(
                "the probe result must have one failure frame per requested world."
            )
        hit = fail_f >= 0
        bins = self.sampler.bin_ids(self.ranks[hit], fail_f[hit])
        observed = None
        if self.schedule is not None:
            start, stop = sweep_frames(
                self.lengths, self.ranks, self.tiles, lead, self.tile
            )
            if bool(((fail_f >= 0) & ((fail_f < start) | (fail_f >= stop))).any()):
                raise ValueError(
                    "a sampled probe failure lies outside its requested interval."
                )
            last = torch.where(hit, fail_f, stop - 1)
            first_bin = self.sampler.bin_ids(self.ranks, start)
            last_bin = self.sampler.bin_ids(self.ranks, last)
            offsets = torch.arange(
                math.ceil((lead + self.tile) / 50) + 1, device=self.ranks.device
            )
            visited = first_bin[:, None] + offsets
            observed = visited[visited <= last_bin[:, None]]
        q = self.focus.update(bins, observed)
        if q is not None:
            self.sampler.set_focus(q, self.share)
            self.focus_set = True
        elif self.schedule is not None and self.focus_set:
            self.sampler.set_focus(self.sampler.focus_q, 0.0)
            self.focus_set = False
        sweeps = self.metrics.get("eprobe/sweeps", 0.0) + 1.0
        self.metrics.update(self._focus_metrics(q))
        self.metrics.update(
            {
                "eprobe/sweeps": sweeps,
                "eprobe/failures": float(hit.sum()),
                "eprobe/worlds": float(self.num_worlds),
            }
        )
        if self.schedule is not None:
            self.metrics["eprobe/eligible_tiles"] = float(self.all_tiles.numel())
        return q

    def _focus_metrics(self, q: Tensor | None) -> dict[str, float]:
        """The focus's bins and the mixed reset distribution's mass on them (its share
        plus the base distribution's mass there), as NDM Gym reports them, and the
        fewest bins holding 90% of the focus with the mixed mass on those: decayed
        counts never reach zero, so after early sweeps the support alone is most bins.
        Read once per sweep."""
        if q is None:
            return {
                "eprobe/focus_bins": 0.0,
                "eprobe/focus_mass": 0.0,
                "eprobe/focus_bins_90": 0.0,
                "eprobe/focus_mass_90": 0.0,
            }
        on = q > 0
        probs = self.sampler.sampling_probabilities()
        mass, order = q.sort(descending=True)
        top = order[: int((mass.cumsum(0) < 0.9).sum()) + 1]
        return {
            "eprobe/focus_bins": float(on.sum()),
            "eprobe/focus_mass": float(probs[on].sum()),
            "eprobe/focus_bins_90": float(top.numel()),
            "eprobe/focus_mass_90": float(probs[top].sum()),
        }

    def log_metrics(self) -> dict[str, float]:
        """The last applied sweep's numbers, host values (none before the first)."""
        return dict(self.metrics)

    # -- checkpoints ---------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """NDM Gym's ``eprobe_fail``, ``eprobe_focus`` and ``eprobe_focus_share``; a sweep
        in flight is not kept."""
        state = {
            "eprobe_fail": self.focus.counts.detach().cpu().clone(),
            "eprobe_focus": self.sampler.focus_q.detach().cpu().clone()
            if self.focus_set
            else None,
            "eprobe_focus_share": float(self.sampler.focus_share)
            if self.focus_set
            else 0.0,
            "num_bins": int(self.sampler.num_bins),
        }
        if self.schedule is not None:
            state.update(
                {
                    "sampled_schedule": self.schedule.state_dict(),
                    "probe_lengths": self.lengths.cpu(),
                    "probe_tile": self.tile,
                    "probe_lead_rng": self.generator.get_state(),
                    "probe_eligible_ranks": self.all_ranks.cpu(),
                    "probe_eligible_tiles": self.all_tiles.cpu(),
                }
            )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if "eprobe_fail" not in state:
            raise ValueError(
                "wm.eprobe_share: the checkpoint has no probe state (eprobe_fail)."
            )
        if int(state.get("num_bins", self.sampler.num_bins)) != self.sampler.num_bins:
            raise ValueError(
                "wm.eprobe_share: the checkpoint's probe state belongs to a bank with other bins."
            )
        if (self.schedule is not None) != ("sampled_schedule" in state):
            raise ValueError(
                "cannot restore probe statistics across full and sampled modes."
            )
        if self.schedule is not None:
            if (
                state["probe_tile"] != self.tile
                or not torch.equal(state["probe_lengths"].cpu(), self.lengths.cpu())
                or not torch.equal(
                    state["probe_eligible_ranks"].cpu(), self.all_ranks.cpu()
                )
                or not torch.equal(
                    state["probe_eligible_tiles"].cpu(), self.all_tiles.cpu()
                )
            ):
                raise ValueError(
                    "the sampled probe checkpoint has incompatible tile geometry."
                )
            self.schedule.load_state_dict(state["sampled_schedule"])
            self.generator.set_state(state["probe_lead_rng"].cpu())
        self.focus.counts.copy_(state["eprobe_fail"].to(self.focus.counts.device))
        if state["eprobe_focus"] is not None:
            self.sampler.set_focus(
                state["eprobe_focus"].to(self.focus.counts.device),
                float(state["eprobe_focus_share"]),
            )
            self.focus_set = True

    def close(self) -> None:
        """Stop the worker (waited for; killed if it does not stop) and remove the
        request directory. A second call does nothing."""
        if self.log is None or self.log.closed:
            return
        if self.proc is not None and self.proc.poll() is None:
            assert self.dir is not None
            Path(self.dir, STOP).touch()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()
        if self.dir is not None:
            shutil.rmtree(self.dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# The worker's side.
# ---------------------------------------------------------------------------


class ProbeSweep:
    """One world per tile in an evaluation-mode G1 env (the worker process)."""

    def __init__(
        self, env: Any, ranks: Tensor, tiles: Tensor, tile: int, *, graph: bool
    ) -> None:
        self.env, self.tile, self.graph = env, int(tile), bool(graph)
        dev = torch.device(env.device)
        self.ranks, self.tiles = ranks.to(dev), tiles.to(dev)
        self.lengths = env.g1_library.length.to(dev)
        n = self.ranks.numel()
        if n != env.num_envs:
            raise ValueError(f"the probe env has {env.num_envs} worlds for {n} tiles.")
        self.fail_f = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.done = torch.zeros(n, dtype=torch.bool, device=dev)
        self.sweeps = 0

    def set_tiles(self, ranks: Tensor, tiles: Tensor) -> None:
        """Reassign existing worker worlds; the simulator and CUDA graph stay fixed."""
        if ranks.shape != self.ranks.shape or tiles.shape != self.tiles.shape:
            raise ValueError("sampled geometry must keep the worker world count fixed.")
        ranks, tiles = ranks.to(self.ranks), tiles.to(self.tiles)
        if bool(((ranks < 0) | (ranks >= self.lengths.numel())).any()):
            raise ValueError("sampled geometry has an invalid clip rank.")
        if bool(
            (
                (tiles < 0)
                | (tiles >= self.lengths[ranks] - 2)
                | (tiles % self.tile != 0)
            ).any()
        ):
            raise ValueError("sampled geometry has an invalid tile frame.")
        self.ranks.copy_(ranks)
        self.tiles.copy_(tiles)

    @torch.no_grad()
    def run(self, policy: Any, lead: int) -> tuple[Tensor, int]:
        """One sweep with this lead: the first-failure local frames (-1: none) on the
        CPU, and the steps run."""
        from tensordict import TensorDict
        from tensordict.nn import InteractionType
        from torchrl.envs.utils import set_exploration_type

        env = self.env
        start, stop = sweep_frames(
            self.lengths, self.ranks, self.tiles, lead, self.tile
        )
        obs, _ = env.reset_to_reference(self.ranks, start)
        self.fail_f.fill_(-1)
        self.done.zero_()
        steps = int(lead) + self.tile
        keys = list(policy.in_keys)
        n = self.ranks.numel()
        ran = 0
        for _ in range(steps):
            frame = env.g1_cursor()[
                1
            ].clone()  # the pre-step frame f; the step judges f + 1
            groups: dict[str, dict[str, Tensor]] = {}
            for group, name in keys:
                groups.setdefault(group, {})[name] = obs[group][name]
            td = TensorDict(
                {g: TensorDict(v, batch_size=[n]) for g, v in groups.items()},
                batch_size=[n],
            )
            with (
                torch.inference_mode(),
                set_exploration_type(InteractionType.DETERMINISTIC),
            ):
                action = policy(td)["action"]
            obs, _, terminated, truncated, _ = env.step(action)
            ran += 1
            alive = ~self.done
            failed = alive & (
                terminated.reshape(-1).bool() | ~torch.isfinite(action).all(-1)
            )
            torch.where(failed, frame, self.fail_f, out=self.fail_f)
            self.done |= (
                terminated.reshape(-1).bool()
                | truncated.reshape(-1).bool()
                | (frame + 1 >= stop)
            )
            # The eager step reads the device every step anyway; the graph worker runs all steps.
            if not self.graph and bool(self.done.all()):
                break
        self.sweeps += 1
        return self.fail_f.cpu(), ran


def _build_env(spec: dict[str, Any]):
    """The worker's simulation and env, as the replica readout configures them."""
    # Match train.py before importing isaaclab.app: its optional isaacsim probe
    # otherwise bootstraps Kit even for Newton in a fresh worker process.

    from imitationlab.runners.runtime_bootstrap import (
        install_kit_import_guard,
        requested_backend,
    )

    if requested_backend(spec["overrides"]) == "newton":
        install_kit_import_guard()
    from imitationlab.runners.launch_compat import add_launcher_args
    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(
        parser, ["--device", spec["device"], *spec["overrides"]]
    )
    args.headless = True
    sys.argv = [sys.argv[0], *hydra_args]
    import imitationlab.envs.isaaclab.tasks  # noqa: F401
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    env_cfg, _ = resolve_task_config(spec["task"], "wmbptt_cfg_entry_point")
    shared = spec.get("shared_library")
    if shared is not None:
        if shared["lengths"] != spec["lengths"]:
            raise ValueError("the shared library descriptor has different clips.")
        env_cfg.library_host_mapped = True
        env_cfg.library_shared_file = shared["path"]
    env_cfg.scene.num_envs = int(spec["num_envs"])
    env_cfg.seed = int(spec["seed"])
    env_cfg.sim.device = spec["device"]
    # A sweep runs up to lead_max + tile steps; the readout extends its episodes too.
    needed = (int(spec["max_steps"]) + 2) * float(spec["step_dt"])
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), needed)
    return env_cfg, args


def worker(qdir: str, spec_path: str) -> None:
    """Serve the trainer's sweep requests in order until it says stop or is gone."""
    spec = json.loads(Path(spec_path).read_text())
    parent = os.getppid()
    env_cfg, args = _build_env(spec)
    import gymnasium as gym
    from isaaclab.app import launch_simulation

    with launch_simulation(env_cfg, args):
        from imitationlab.algorithms.common.world_model.learner import deployed_policy

        env = gym.make(spec["task"], cfg=env_cfg).unwrapped
        lengths = env.g1_library.length.cpu().tolist()
        if lengths != spec["lengths"]:
            raise ValueError(
                "the probe worker's bank has other clips than the trainer's."
            )
        device = torch.device(env.device)
        sweep = ProbeSweep(
            env,
            torch.tensor(spec["ranks"]),
            torch.tensor(spec["tiles"]),
            int(spec["tile"]),
            graph=bool(spec["graph"]),
        )
        Path(qdir, READY).touch()
        print(
            f"eprobe worker ready: {env.num_envs} worlds, task {spec['task']}",
            flush=True,
        )
        policy, meta, k = None, None, 0
        while True:
            req = os.path.join(qdir, f"req_{k:05d}.pt")
            while not os.path.exists(req):
                if os.path.exists(os.path.join(qdir, STOP)):
                    print("eprobe worker stopped", flush=True)
                    return
                if os.getppid() != parent:
                    print("eprobe worker: the trainer is gone, exiting", flush=True)
                    return
                time.sleep(0.005)
            request = torch.load(req, map_location=device, weights_only=False)
            os.remove(req)
            if policy is None:
                policy, meta = (
                    deployed_policy(env, request, device),
                    request["policy_meta"],
                )
            else:
                if request["policy_meta"] != meta:
                    raise ValueError(
                        "the deployed policy's metadata changed during the run."
                    )
                policy.load_state_dict(request["policy_state_dict"], strict=True)
            t0 = time.monotonic()
            if spec.get("sampled", False):
                sweep.set_tiles(request["ranks"], request["tiles"])
            fail_f, steps = sweep.run(policy, int(request["lead"]))
            sweep_s = time.monotonic() - t0
            reserved = (
                torch.cuda.max_memory_reserved(device) / 2**20
                if device.type == "cuda"
                else float("nan")
            )
            res = os.path.join(qdir, f"res_{k:05d}.pt")
            torch.save(
                {
                    "fail_f": fail_f,
                    "sweep_s": sweep_s,
                    "steps": steps,
                    "reserved_mib": reserved,
                },
                res + ".tmp",
            )
            os.replace(res + ".tmp", res)
            print(
                f"eprobe sweep {k} (it {request['iteration']}, lead {request['lead']}): {int((fail_f >= 0).sum())} "
                f"failures, {steps} steps, {sweep_s:.1f} s, torch reserved {reserved:.0f} MiB",
                flush=True,
            )
            k += 1


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit(
            "usage: python -m imitationlab.evaluation.world_model.g1_eval_probe --worker <dir> <spec.json>"
        )
    worker(sys.argv[2], sys.argv[3])
