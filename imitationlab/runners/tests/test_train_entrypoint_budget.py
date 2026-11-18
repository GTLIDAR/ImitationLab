"""Training entrypoint wall-clock budget and checkpoint dispatch.

`train.py`/`train_impl` import Isaac Lab, so the pieces are read from source
here instead of importing the modules (as test_resolve_checkpoint_path.py does).
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import torch

_RUNNERS = Path(__file__).resolve().parents[1]


def _load_function(source: Path, name: str):
    tree = ast.parse(source.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            namespace: dict = {"torch": torch}
            exec(textwrap.dedent(ast.unparse(node)), namespace)  # noqa: S102
            return namespace[name]
    raise AssertionError(f"{name} not found in {source.name}")


load_checkpoint = _load_function(
    _RUNNERS / "train_impl.py", "_load_training_checkpoint"
)


class _Agent:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def load(self, path: str) -> None:
        self.calls.append(("load", path))

    def load_model(self, path: str) -> None:
        self.calls.append(("load_model", path))


def _ckpt(tmp_path: Path, payload: dict) -> str:
    path = tmp_path / "model_step_8.pt"
    torch.save(payload, path)
    return str(path)


def test_model_checkpoint_uses_load_model(tmp_path):
    path = _ckpt(tmp_path, {"policy_state_dict": {}})
    agent = _Agent()
    load_checkpoint(agent, path)
    assert agent.calls == [("load_model", path)]


def test_full_state_checkpoint_uses_agent_load(tmp_path):
    path = _ckpt(tmp_path, {"actor_critic": {}})
    agent = _Agent()
    load_checkpoint(agent, path)
    assert agent.calls == [("load", path)]


def test_cli_declares_max_train_seconds_as_float():
    tree = ast.parse((_RUNNERS / "train.py").read_text())
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--max_train_seconds"
    ]
    assert len(found) == 1
    kwargs = {kw.arg: kw.value for kw in found[0].keywords}
    assert isinstance(kwargs["type"], ast.Name) and kwargs["type"].id == "float"
    assert (
        isinstance(kwargs["default"], ast.Constant) and kwargs["default"].value is None
    )


def test_entrypoint_maps_the_flag_onto_the_trainer_config():
    source = (_RUNNERS / "train_impl.py").read_text()
    assert "agent_cfg.trainer.max_train_seconds = float(max_train_seconds)" in source
    assert "supports_train_time_budget" in source


def test_signal_handler_ignores_repeated_sigterm():
    """The first TERM interrupts; a forwarded second TERM must not abort the save."""
    import signal

    import pytest

    tree = ast.parse((_RUNNERS / "train_impl.py").read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "cleanup_pbar"
    )
    namespace: dict = {"signal": signal, "_sigint_seen": False}
    exec(textwrap.dedent(ast.unparse(node)), namespace)  # noqa: S102
    handler = namespace["cleanup_pbar"]
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGTERM, None)
        assert namespace["_sigint_seen"] is True
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
        assert signal.getsignal(signal.SIGINT) == signal.default_int_handler
    finally:
        for sig, previous in saved.items():
            signal.signal(sig, previous)


def test_wmbptt_compile_options_come_before_the_env() -> None:
    """``ImitationLearningTools`` helpers compile during env setup, and
    ``wm.elementwise_3x3`` refuses a process that has compiled."""
    source = (_RUNNERS / "train_impl.py").read_text()
    assert source.index("apply_compile_options(agent_cfg.wm)") < source.index(
        "gym.make(args_cli.task, cfg=env_cfg)"
    )
