"""Render measured time-to-tracking curves without publishing motion data."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

from imitationlab.runners.registry import DEFAULT_TASK, INTERFACES, canonical_interface

POLICIES = ("ppo", "supermimic", "svg", "diffsr")
DATASETS = ("lafan1", "bones_seed")


def summarize(paths: list[Path]) -> dict:
    from imitationlab.runners.benchmark import first_reached

    cases = {}
    boards = {}
    sources = set()
    gpus = set()
    for path in paths:
        raw = json.loads(path.read_text())
        key = (raw["dataset"], canonical_interface(raw["interface"]), raw["policy"])
        if key in cases:
            raise ValueError(f"Duplicate benchmark case: {key}")
        if raw["status"] == "completed":
            counts = (raw.get("checkpoint_validation") or {}).get("optimizer_steps", {})
            required = {"ppo"} if raw["policy"] == "ppo" else {"wm", "actor", "critic"}
            if not required.issubset(counts) or any(
                not math.isfinite(float(counts[name])) or float(counts[name]) <= 0
                for name in required
            ):
                raise ValueError(
                    f"Completed case {key} lacks positive policy optimizer update evidence; rerun the benchmark."
                )
        board = raw["board"]
        if (
            raw["dataset"] in boards
            and boards[raw["dataset"]]["sha256"] != board["sha256"]
        ):
            raise ValueError("Different evaluation boards cannot share a curve table.")
        boards[raw["dataset"]] = {
            k: board[k]
            for k in (
                "bank_id",
                "bank_sha256",
                "bank_clips",
                "bank_frames",
                "sha256",
                "steps",
                "control_dt",
                "seed",
                "success",
                "failure_rule",
                "population",
            )
        }
        boards[raw["dataset"]]["evaluation_clips"] = len(board["ranks"])
        sources.add(raw["source_sha256"])
        selection = raw.get("run", {}).get("hardware_selection") or {}
        hardware = selection.get("hardware", {})
        if hardware.get("gpu_name"):
            gpus.add(hardware["gpu_name"])
        points = [
            {
                k: p[k]
                for k in (
                    "train_seconds",
                    "end_to_end_seconds",
                    "frames",
                    "success",
                    "mpjpe_l_mm",
                    "mpjpe_g_mm",
                    "survival_seconds",
                    "valid_transitions",
                    "checkpoint_sha256",
                    "evaluation_seconds",
                )
            }
            for p in raw["points"]
        ]
        cases[key] = dict(
            dataset=key[0],
            interface=key[1],
            policy=key[2],
            status=raw["status"],
            budget_seconds=raw["budget_seconds"],
            milestones=raw["milestones"],
            seed=raw["seed"],
            source_sha256=raw["source_sha256"],
            software=raw.get("runtime", {}).get("packages", {}),
            hardware=hardware,
            profile=selection.get("profile"),
            resolved=selection.get("resolved"),
            points=points,
            pretrain_seconds=raw["pretrain_seconds"],
            startup_seconds=raw.get("startup_seconds"),
            encoder_sha256=(raw.get("run", {}).get("encoder") or {}).get("sha256"),
            encoder_pretraining=raw.get("encoder_pretraining"),
            thresholds=raw.get("thresholds", {}),
            checkpoint_validation=raw.get("checkpoint_validation"),
        )
        if raw["status"] == "completed":
            # Recompute from saved measurements, including records produced before
            # threshold comparisons accounted for float32 success-rate rounding.
            thresholds = {
                str(level): first_reached(points, level) for level in (0.5, 0.9, 0.95)
            }
            thresholds["0.9_and_40mm"] = first_reached(points, 0.9, local_error=40)
            if thresholds != raw.get("thresholds", {}):
                cases[key]["recorded_thresholds"] = raw.get("thresholds", {})
            cases[key]["thresholds"] = thresholds
        # Errors can contain proprietary paths. The report only identifies the failed cell.
    if len(sources) > 1 or len(gpus) > 1:
        raise ValueError(
            "Use separate reports for different source snapshots or GPU models."
        )
    return dict(
        schema_version=1,
        boards=boards,
        sources=sorted(sources),
        gpus=sorted(gpus),
        cases=list(cases.values()),
    )


def sample_at(points: list[dict], seconds: float) -> dict | None:
    return next(
        (
            p
            for p in sorted(points, key=lambda x: x["train_seconds"])
            if p["train_seconds"] >= seconds
        ),
        None,
    )


def render_markdown(summary: dict) -> str:
    cases = {(c["dataset"], c["interface"], c["policy"]): c for c in summary["cases"]}
    complete = sum(c["status"] == "completed" for c in cases.values())
    budgets = sorted({c["budget_seconds"] for c in cases.values()})
    milestones = sorted({t for c in cases.values() for t in c["milestones"]})
    columns = milestones or [60, 120, 300, 600]
    seeds = sorted({c["seed"] for c in cases.values()})
    lines = [
        "# G1 learning speed",
        "",
        f"{complete}/24 cases completed. GPU: {', '.join(summary['gpus']) or 'not measured yet'}.",
        "",
        "Each cell below is **tracking success % / MPJPE-L mm @ actual training minutes**.",
        "A time column selects the first saved checkpoint at or after its requested time.",
        "If one iteration crosses multiple times, the columns repeat that checkpoint; they are not independent measurements.",
        "Missing measurements remain pending. A completed run is a measurement, not a claim of successful tracking.",
        "Final checkpoints are checked for finite tensors and positive learner optimizer step counters, including the policy.",
        "",
    ]
    for dataset in DATASETS:
        lines += [
            f"## {dataset}",
            "",
            "| Learner / command | "
            + " | ".join(f"≥{seconds / 60:g} min" for seconds in columns)
            + " | First ≥90% success |",
            "| " + " | ".join(["---"] * (len(columns) + 2)) + " |",
        ]
        for interface, policy in itertools.product(INTERFACES, POLICIES):
            case = cases.get((dataset, interface, policy))
            cells = []
            for seconds in columns:
                point = sample_at(case["points"], seconds) if case else None
                cells.append(
                    f"{point['success'] * 100:.1f}% / {point['mpjpe_l_mm']:.1f} @ {point['train_seconds'] / 60:.2f}"
                    if point
                    else (
                        "not sampled"
                        if case and case["status"] == "completed"
                        else case["status"]
                        if case
                        else "pending"
                    )
                )
            threshold = case["thresholds"].get("0.9") if case else None
            reached = (
                f"{threshold['seconds'] / 60:.2f} min"
                if threshold
                else (
                    "not reached"
                    if case and case["status"] == "completed"
                    else "pending"
                )
            )
            lines.append(
                f"| {policy} / {interface} | " + " | ".join(cells) + f" | {reached} |"
            )
        lines += [""]
    lines += [
        "## Protocol and limits",
        "",
        f"Recorded training budgets (seconds): {budgets or 'pending'}; seeds: {seeds or 'pending'}; task: `{DEFAULT_TASK}`. Requested checkpoint times (seconds): {milestones or 'pending'}. Save at completed iteration boundaries and use the checkpoint clock, including learner compilation and checkpoint overhead. Startup ends after the first collected batch and is recorded separately. No evaluation runs inside the training clock. Model and execution settings are retained in each run's metadata.",
        "",
        "Evaluate deterministic actions from frame zero without noise or randomization to the recorded horizon, or natural clip completion if shorter. The board sizes and horizons are listed below. These are training motions: this measures acquisition, not held-out generalization or full-bank coverage.",
        "",
        "Success means no root tracking failure: absolute root-height error ≤0.25 m and absolute root R[2,2] error ≤0.8. MPJPE-L and MPJPE-G aggregate valid transitions before termination across the whole board, so failed runs contribute shorter trajectories. Read pose error together with success and survival time; lower error alone does not establish better tracking.",
        "",
        "Threshold times are the first sampled checkpoints meeting the target, without interpolation. Success comparisons allow 1e-7 absolute float32 representation error; displayed measurements are unchanged. “Not reached” means no saved checkpoint met the target within this run; a transient crossing between checkpoints is unknown. One seed does not establish a statistically reliable learner ranking.",
        "",
        "FSQ and spectral each use a dataset-specific frozen encoder; its pretraining record includes the update count and architecture. Encoder process time and policy startup are added separately for end-to-end time; queueing and evaluation are excluded. Reusing an encoder amortizes its cost across policies. All timing is specific to the recorded hardware, software, source snapshot, and compiler cache state. The cluster sweep uses separate cold compiler caches per job.",
        "",
        "Raw BONES-SEED data, per-motion results, and model weights stay outside Git. Only aggregate measurements and hashes are published here.",
        "",
        "## Cost and resolved execution",
        "",
        "| Dataset | Command | Learner | Profile | Envs | Encoder min | Startup min | Last train min | Last end-to-end min | MPJPE-G mm | Mean survival s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in sorted(
        cases.values(), key=lambda c: (c["dataset"], c["interface"], c["policy"])
    ):
        if not case["points"]:
            continue
        p = case["points"][-1]
        lines.append(
            f"| {case['dataset']} | {case['interface']} | {case['policy']} | {case['profile']} | {(case['resolved'] or {}).get('num_envs', 'unknown')} | {case['pretrain_seconds'] / 60:.2f} | {(case['startup_seconds'] or 0) / 60:.2f} | {p['train_seconds'] / 60:.2f} | {p['end_to_end_seconds'] / 60:.2f} | {p['mpjpe_g_mm']:.1f} | {p['survival_seconds']:.2f} |"
        )
    lines += [
        "",
        "## Provenance",
        "",
        "Source fingerprint: " + ", ".join(f"`{x}`" for x in summary["sources"]) + ".",
        "",
    ]
    for name, board in summary["boards"].items():
        lines += [
            f"- {name}: {board['evaluation_clips']} clips, up to {board['steps']} steps ({board['steps'] * board['control_dt']:g} seconds), seed {board['seed']}; `{board['bank_id']}`; manifest SHA-256 `{board['bank_sha256']}`; board SHA-256 `{board['sha256']}`."
        ]
    versions = {
        json.dumps(c.get("software", {}), sort_keys=True)
        for c in cases.values()
        if c.get("software")
    }
    if versions:
        lines += [
            "",
            "Recorded software: " + "; ".join(f"`{v}`" for v in sorted(versions)) + ".",
        ]
    return "\n".join(lines) + "\n"


def plot(summary: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(POLICIES, ["#1874b4", "#ed7d21", "#279b58", "#9950aa"]))
    for metric, label, filename in [
        ("success", "Tracking success (%)", "success"),
        ("mpjpe_l_mm", "MPJPE-L (mm)", "pose-error"),
    ]:
        fig, axes = plt.subplots(
            2, 3, figsize=(12, 6.6), sharex=True, sharey=True, layout="constrained"
        )
        for row, dataset in enumerate(DATASETS):
            for col, interface in enumerate(INTERFACES):
                ax = axes[row, col]
                for case in summary["cases"]:
                    if (
                        case["dataset"] != dataset
                        or case["interface"] != interface
                        or not case["points"]
                    ):
                        continue
                    points = sorted(case["points"], key=lambda p: p["train_seconds"])
                    ax.plot(
                        [p["train_seconds"] / 60 for p in points],
                        [
                            p[metric] * (100 if metric == "success" else 1)
                            for p in points
                        ],
                        marker="o",
                        ms=4,
                        lw=1.5,
                        color=colors[case["policy"]],
                        label=case["policy"],
                    )
                ax.set_title(f"{dataset} · {interface}")
                ax.grid(alpha=0.2)
                ax.set_xlim(left=0)
                if metric == "success":
                    ax.set_ylim(-2, 102)
                if row == 1:
                    ax.set_xlabel("Actual policy training time (min)")
                if col == 0:
                    ax.set_ylabel(label)
        handles = [
            plt.Line2D([0], [0], color=colors[p], marker="o", label=p) for p in POLICIES
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False)
        fig.savefig(output / f"{filename}.svg")
        fig.savefig(output / f"{filename}.png", dpi=160)
        plt.close(fig)
    plot_overview(summary, output)


# Headline figure: one line per learner (median over the dataset x interface
# cases) with its min-max range, in light and dark variants for README use.
OVERVIEW_POLICIES = (
    ("supermimic", "SuperMimic"),
    ("svg", "SVG"),
    ("diffsr", "DiffSR"),
    ("ppo", "PPO"),
)
OVERVIEW_THEMES = {
    "light": {
        "ink": "#1f2328",
        "muted": "#59636e",
        "grid": "#d1d9e0",
        "series": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
    },
    "dark": {
        "ink": "#f0f6fc",
        "muted": "#9198a1",
        "grid": "#3d444d",
        "series": ("#3987e5", "#d95926", "#199e70", "#c98500"),
    },
}


def overview_curves(summary: dict, metric: str) -> dict[str, dict]:
    """Median, min and max of a metric over each learner's cases on a shared grid.

    Each case is linearly interpolated between its measured checkpoints, as the
    per-case plots draw it. The grid starts at the latest first checkpoint so
    every case contributes to every grid point.
    """
    import numpy as np

    scale = 100.0 if metric == "success" else 1.0
    curves = {}
    for policy, _ in OVERVIEW_POLICIES:
        cases = [
            sorted(c["points"], key=lambda p: p["train_seconds"])
            for c in summary["cases"]
            if c["policy"] == policy and c["points"]
        ]
        if not cases:
            continue
        start = max(points[0]["train_seconds"] for points in cases) / 60
        stop = min(points[-1]["train_seconds"] for points in cases) / 60
        grid = np.linspace(start, stop, 200)
        values = np.stack(
            [
                np.interp(
                    grid,
                    [p["train_seconds"] / 60 for p in points],
                    [p[metric] * scale for p in points],
                )
                for points in cases
            ]
        )
        curves[policy] = {
            "minutes": grid,
            "median": np.median(values, axis=0),
            "low": values.min(axis=0),
            "high": values.max(axis=0),
            "cases": len(cases),
        }
    return curves


def plot_overview(summary: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator

    panels = [
        ("success", "Tracking success", "%"),
        ("mpjpe_l_mm", "Local pose error (MPJPE-L)", "mm"),
    ]
    curves = {metric: overview_curves(summary, metric) for metric, *_ in panels}
    if not curves["success"]:
        return
    budget = max(float(c["minutes"][-1]) for c in curves["success"].values())
    style = {
        "font.family": ["Lato", "Noto Sans", "DejaVu Sans"],
        "font.size": 11,
        "svg.hashsalt": "imitationlab-overview",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
    for theme, palette in OVERVIEW_THEMES.items():
        colors = dict(zip((p for p, _ in OVERVIEW_POLICIES), palette["series"]))
        ink, muted, grid = palette["ink"], palette["muted"], palette["grid"]
        with plt.rc_context(style):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
            fig.subplots_adjust(
                left=0.055, right=0.875, top=0.725, bottom=0.125, wspace=0.22
            )
            for ax, (metric, title, unit) in zip(axes, panels):
                for z, (policy, name) in enumerate(reversed(OVERVIEW_POLICIES)):
                    curve = curves[metric].get(policy)
                    if curve is None:
                        continue
                    color = colors[policy]
                    ax.fill_between(
                        curve["minutes"],
                        curve["low"],
                        curve["high"],
                        color=color,
                        alpha=0.14,
                        lw=0,
                        zorder=1 + z,
                    )
                    ax.plot(
                        curve["minutes"],
                        curve["median"],
                        color=color,
                        lw=2.4 if policy == "supermimic" else 1.8,
                        solid_capstyle="round",
                        zorder=10 + z,
                        label=name,
                    )
                    ax.scatter(
                        [curve["minutes"][-1]],
                        [curve["median"][-1]],
                        s=36,
                        color=color,
                        edgecolors="none",
                        zorder=20 + z,
                    )
                ax.set_xlim(0, 31)
                ax.xaxis.set_major_locator(FixedLocator([0, 5, 10, 15, 20, 25, 30]))
                ax.set_xlabel("Policy training time on one H100 (min)", color=muted)
                ax.set_title(
                    f"{title} ({unit})", loc="left", color=ink, fontsize=12.5, pad=10
                )
                if metric == "success":
                    ax.set_ylim(0, 103)
                    ax.yaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
                else:
                    ax.set_yscale("log")
                    ax.set_ylim(10, 200)
                    ticks = [10, 20, 40, 80, 160]
                    ax.yaxis.set_major_locator(FixedLocator(ticks))
                    ax.yaxis.set_minor_locator(NullLocator())
                    ax.set_yticklabels([str(t) for t in ticks])
                    _label_line_ends(ax, curves[metric], colors, ink)
                ax.grid(axis="y", color=grid, lw=0.8, alpha=0.7)
                ax.set_axisbelow(True)
                ax.tick_params(colors=muted, labelcolor=muted, length=0, pad=6)
                ax.spines["left"].set_visible(False)
                ax.spines["bottom"].set_color(grid)
                ax.patch.set_alpha(0)
            handles, labels = axes[0].get_legend_handles_labels()
            order = [
                labels.index(name) for _, name in OVERVIEW_POLICIES if name in labels
            ]
            legend = fig.legend(
                [handles[i] for i in order],
                [labels[i] for i in order],
                loc="lower left",
                bbox_to_anchor=(0.048, 0.79),
                ncol=len(order),
                frameon=False,
                handlelength=1.6,
                columnspacing=1.4,
                fontsize=11,
            )
            for text in legend.get_texts():
                text.set_color(ink)
            fig.text(
                0.055,
                0.915,
                f"Unitree G1 motion tracking over {budget:.0f} minutes of training on one GPU",
                color=ink,
                fontsize=16,
                fontweight="bold",
            )
            fig.savefig(
                output / f"overview-{theme}.svg",
                transparent=True,
                metadata={"Date": None},
            )
            plt.close(fig)


def _label_line_ends(ax, curves: dict, colors: dict, ink: str) -> None:
    """Direct-label each median line at its right end, nudged apart on a log axis."""
    import math

    ends = sorted(
        ((math.log10(curve["median"][-1]), policy) for policy, curve in curves.items()),
    )
    gap = 0.075
    placed: list[float] = []
    for y, _ in ends:
        placed.append(max(y, placed[-1] + gap) if placed else y)
    names = dict(OVERVIEW_POLICIES)
    for (_, policy), y in zip(ends, placed):
        x = curves[policy]["minutes"][-1]
        ax.annotate(
            f"{names[policy]}  {curves[policy]['median'][-1]:.0f} mm",
            xy=(x, 10**y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            color=ink,
            annotation_clip=False,
        )


def main(argv=None, *, prog: str | None = None) -> int:
    p = argparse.ArgumentParser(prog=prog, description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing per-case result.json files.",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)
    summary = summarize(sorted(args.input.rglob("result.json")))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "README.md").write_text(render_markdown(summary))
    if not args.no_plot and any(case["points"] for case in summary["cases"]):
        plot(summary, args.output)
    print(
        f"{sum(c['status'] == 'completed' for c in summary['cases'])}/24 completed; report: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
