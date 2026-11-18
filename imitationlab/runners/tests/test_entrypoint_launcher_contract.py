"""Static checks that Isaac entrypoints use the Isaac Lab ``develop`` launcher API.

The checks parse the sources and import nothing from Isaac Lab, so they run in
the default Pixi environment.
"""

from __future__ import annotations

import ast
import re
import shlex
import tomllib
from pathlib import Path

import pytest

from imitationlab.paths import REPO_ROOT

ENTRYPOINT_ROOTS = (
    REPO_ROOT / "scripts",
    REPO_ROOT / "imitationlab",
)
REMOVED_LAUNCHER_NAMES = {
    "isaaclab_tasks.utils": {
        "add_launcher_args",
        "compute_kit_requirements",
        "launch_simulation",
    },
    "isaaclab_tasks.utils.hydra": {"hydra_task_config"},
}
# The compatibility module wraps the removed names on purpose.
EXEMPT = {
    REPO_ROOT / "imitationlab/runners/launch_compat.py",
}


def _python_sources() -> list[Path]:
    paths: list[Path] = []
    for root in ENTRYPOINT_ROOTS:
        paths.extend(
            path
            for path in root.rglob("*.py")
            if "tests" not in path.parts and path not in EXEMPT
        )
    return sorted(paths)


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize(
    "path", _python_sources(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_no_removed_launcher_api(path: Path) -> None:
    problems: list[str] = []
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.ImportFrom) and node.module in REMOVED_LAUNCHER_NAMES:
            removed = REMOVED_LAUNCHER_NAMES[node.module]
            problems += [
                f"line {node.lineno}: from {node.module} import {alias.name}"
                for alias in node.names
                if alias.name in removed
            ]
        elif isinstance(node, ast.Attribute) and _dotted(node).endswith(
            "wrappers.RecordVideo"
        ):
            problems.append(f"line {node.lineno}: gym.wrappers.RecordVideo")
        elif (
            isinstance(node, ast.keyword)
            and node.arg == "render_mode"
            and isinstance(node.value, ast.Constant)
            and node.value.value == "rgb_array"
        ):
            problems.append(f"line {node.lineno}: render_mode='rgb_array'")
    assert not problems, f"{path} uses the removed launcher API:\n" + "\n".join(
        problems
    )


def _algorithm_map_keys(tree: ast.Module) -> set[str]:
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "ALGORITHM_CLASS_MAP"
                for target in node.targets
            )
            and isinstance(node.value, ast.Dict)
        ):
            return {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    return set()


def _algo_choices(tree: ast.Module) -> set[str] | None:
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call) and _dotted(node.func).endswith("add_argument")
        ):
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
        if "--algo" not in flags:
            continue
        for keyword in node.keywords:
            if keyword.arg == "choices" and isinstance(
                keyword.value, (ast.List, ast.Tuple)
            ):
                return {
                    item.value
                    for item in keyword.value.elts
                    if isinstance(item, ast.Constant)
                }
    return None


# Entrypoints exempt from the class-map vs `--algo choices` check.
ALGO_CHOICES_GAP: set[str] = set()


def test_every_algorithm_in_a_class_map_is_selectable_from_the_cli() -> None:
    """A map entry that ``--algo choices=`` rejects is dead code behind a silent argparse error."""
    problems: list[str] = []
    for path in _python_sources():
        if path.name in ALGO_CHOICES_GAP:
            continue
        tree = _tree(path)
        keys = _algorithm_map_keys(tree)
        choices = _algo_choices(tree)
        if keys and choices is not None and keys - choices:
            problems.append(f"{path.relative_to(REPO_ROOT)}: {sorted(keys - choices)}")
    assert not problems, (
        "ALGORITHM_CLASS_MAP keys missing from --algo choices:\n" + "\n".join(problems)
    )


def test_pixi_tasks_point_at_existing_entrypoints() -> None:
    manifest = tomllib.loads((REPO_ROOT / "pixi.toml").read_text(encoding="utf-8"))
    tasks: dict[str, object] = dict(manifest.get("tasks", {}))
    for feature in manifest.get("feature", {}).values():
        tasks.update(feature.get("tasks", {}))
    missing: list[str] = []
    for name, task in tasks.items():
        cmd = task if isinstance(task, str) else (task or {}).get("cmd", "")
        if not isinstance(cmd, str):
            continue
        cwd = REPO_ROOT / (task.get("cwd", "") if isinstance(task, dict) else "")
        tokens = shlex.split(cmd)
        for index, token in enumerate(tokens):
            if index and tokens[index - 1] == "--exclude":
                continue  # an exclude pattern may name a file that is gone
            if (
                re.fullmatch(
                    r"(scripts|experiments|imitationlab|tests)/[\w./-]+\.py", token
                )
                and not (cwd / token).exists()
                and not (REPO_ROOT / token).exists()
            ):
                missing.append(f"{name}: {token}")
    assert not missing, "pixi tasks name missing files:\n" + "\n".join(missing)


def test_every_test_file_runs_in_a_pixi_test_task() -> None:
    """A test file that no ``test-*`` task names never runs in CI."""
    manifest = tomllib.loads((REPO_ROOT / "pixi.toml").read_text(encoding="utf-8"))
    tasks: dict[str, object] = dict(manifest.get("tasks", {}))
    for feature in manifest.get("feature", {}).values():
        tasks.update(feature.get("tasks", {}))
    targets: list[Path] = []
    for name, task in tasks.items():
        cmd = task if isinstance(task, str) else (task or {}).get("cmd", "")
        if not name.startswith("test-") or not isinstance(cmd, str):
            continue
        targets += [REPO_ROOT / t for t in shlex.split(cmd) if (REPO_ROOT / t).exists()]
    roots = (REPO_ROOT / "imitationlab", REPO_ROOT / "scripts")
    uncovered = [
        str(path.relative_to(REPO_ROOT))
        for root in roots
        for path in sorted(root.rglob("test_*.py"))
        if not any(path == t or t in path.parents for t in targets)
    ]
    assert not uncovered, "test files no pixi test task runs:\n" + "\n".join(uncovered)


def test_pixi_tasks_do_not_pass_the_removed_headless_flag() -> None:
    text = (REPO_ROOT / "pixi.toml").read_text(encoding="utf-8")
    assert "--headless" not in text


def _isaaclab_torch_version() -> str:
    """The torch version that the `isaaclab` Pixi environment locks."""
    manifest = tomllib.loads((REPO_ROOT / "pixi.toml").read_text())
    feature = manifest["feature"]["isaaclab"]
    spec = feature["pypi-options"]["dependency-overrides"]["torch"]
    version = spec["version"] if isinstance(spec, dict) else spec
    return version.removeprefix("==")


@pytest.mark.parametrize(
    "relative",
    [
        "imitationlab/runners/runtime_bootstrap.py",
        "scripts/tracking/kit_bootstrap/sitecustomize.py",
    ],
)
def test_cu130_bridge_checks_the_locked_torch_version(relative: str) -> None:
    """The Kit bridge rejects any other torch, so its pin must follow pixi.toml."""
    expected = _isaaclab_torch_version()
    pinned = set(
        re.findall(r'!= "(\d+\.\d+\.\d+)"', (REPO_ROOT / relative).read_text())
    )
    assert pinned == {expected}, (
        f"{relative} checks {pinned}, pixi.toml locks {expected}"
    )
