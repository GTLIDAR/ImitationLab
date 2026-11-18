"""The environment half of the training-log contract, without a simulator.

A running env is checked by the live smokes; these tests pin what a smoke
cannot keep pinned.
"""

from __future__ import annotations

import inspect
from collections import deque
from types import SimpleNamespace

import torch

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers import TerminationManager

from imitationlab.envs.isaaclab.contracts import (
    REQUIRED_ENV_LOG_ENTRIES,
    tracking_failed_share,
)
from imitationlab.envs.isaaclab.envs import ImitationRLEnv
from imitationlab.envs.isaaclab.envs.g1_env import G1RLEnv
from imitationlab.envs.isaaclab.envs.torchrl_adapter import IsaacLabWrapper
from imitationlab.envs.isaaclab.tasks.manager_based.imitation.config.g1.imitation_g1_env import (
    ImitationG1EnvCfg,
)
from imitationlab.envs.isaaclab.tasks.manager_based.imitation.mdp.commands import (
    reference,
)


def test_the_consumers_read_these_exact_names():
    assert set(REQUIRED_ENV_LOG_ENTRIES) == {
        "Metrics/reference/mpjpe_l_mm",
        "Metrics/reference/mpjpe_g_mm",
        "Episode_Termination/time_out",
        "Episode_Termination/reference_finished",
        "Episode_Termination/tracking_failed",
    }


def test_failure_share_counts_an_env_once_and_skips_success_terms():
    # Terms: time_out, reference_finished, anchor_pos, ee_body_pos.
    failure_terms = torch.tensor([False, False, True, True])
    dones = torch.tensor(
        [
            [True, False, False, False],  # timed out
            [False, True, False, False],  # finished the clip
            [False, False, True, True],  # two failure terms on one step
            [False, False, False, True],  # one failure term
        ]
    )
    assert tracking_failed_share(dones, failure_terms).item() == 0.5


def test_the_task_owns_the_three_termination_entries():
    terminations = ImitationG1EnvCfg().terminations
    assert {name for name, term in vars(terminations).items() if term is not None} == {
        "time_out",
        "reference_finished",
        "tracking_failed",
    }


def test_every_env_declares_the_entries():
    assert ImitationRLEnv.required_log_entries == REQUIRED_ENV_LOG_ENTRIES
    assert G1RLEnv.required_log_entries == REQUIRED_ENV_LOG_ENTRIES


def test_the_reference_term_owns_the_mpjpe_entries():
    source = inspect.getsource(reference.ReferenceCommandTerm.__init__)
    assert 'self.metrics["mpjpe_l_mm"]' in source
    assert 'self.metrics["mpjpe_g_mm"]' in source


def test_the_wrapper_forwards_the_declaration_and_live_scalars():
    wrapper = IsaacLabWrapper.__new__(IsaacLabWrapper)
    live = {"TrainHealth/x": torch.tensor(1.0)}
    base = SimpleNamespace(
        required_log_entries=REQUIRED_ENV_LOG_ENTRIES,
        reference_command=SimpleNamespace(transition_metrics=lambda: live),
    )
    wrapper.__dict__["_env"] = base
    wrapper.__dict__["log_infos"] = deque(maxlen=1)
    assert wrapper.required_log_entries == REQUIRED_ENV_LOG_ENTRIES
    assert wrapper.live_log_metrics() == live


def test_upstream_still_rebuilds_the_log_and_keeps_the_done_buffer():
    """Pin the Isaac Lab internals the failure share reads."""
    assert 'self.extras["log"] = dict()' in inspect.getsource(
        ManagerBasedRLEnv._reset_idx
    )
    assert "_last_episode_dones" in inspect.getsource(TerminationManager.reset)
