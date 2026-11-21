"""Every data file tracked under ``imitationlab/`` ships in the wheel.

Pixi installs the package in editable mode, where a file that
``[tool.setuptools.package-data]`` does not name still loads. A wheel would
lack it, and the simulator would fail to find a robot file at run time.
"""

from __future__ import annotations

import fnmatch
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

# Guides read on GitHub and on the documentation site, not at run time.
NOT_SHIPPED = {
    "imitationlab/algorithms/README.md",
    "imitationlab/data_tools/README.md",
}


def _matches(relative: str, pattern: str) -> bool:
    """Match like setuptools: ``*`` stays within one directory."""
    parts, wanted = relative.split("/"), pattern.split("/")
    return len(parts) == len(wanted) and all(
        fnmatch.fnmatchcase(part, want)
        for part, want in zip(parts, wanted, strict=True)
    )


def _shipped(path: str, package_data: dict[str, list[str]]) -> bool:
    for package, patterns in package_data.items():
        root = package.replace(".", "/") + "/"
        if path.startswith(root) and any(
            _matches(path.removeprefix(root), pattern) for pattern in patterns
        ):
            return True
    return False


def test_tracked_data_files_are_package_data() -> None:
    try:
        listing = subprocess.run(
            ["git", "ls-files", "imitationlab"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("needs a Git checkout")
    data_files = [
        path
        for path in listing.stdout.splitlines()
        if not path.endswith(".py") and "/tests/" not in path
    ]
    assert data_files, "git ls-files returned no data files"

    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = project["tool"]["setuptools"]["package-data"]
    missing = sorted(
        path
        for path in data_files
        if path not in NOT_SHIPPED and not _shipped(path, package_data)
    )
    assert not missing, (
        "tracked files that a wheel would leave out; add them to "
        f"[tool.setuptools.package-data] in pyproject.toml: {missing}"
    )


def test_a_file_outside_the_patterns_is_reported() -> None:
    package_data = {"imitationlab.envs.isaaclab": ["assets/robot/*.usd"]}
    assert _shipped("imitationlab/envs/isaaclab/assets/robot/g1.usd", package_data)
    assert not _shipped(
        "imitationlab/envs/isaaclab/assets/robot/configuration/base.usd", package_data
    )
