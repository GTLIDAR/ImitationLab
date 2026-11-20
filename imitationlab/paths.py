"""Location-independent repository paths for the ImitationLab package.

The repository root is found by walking up from this file to its markers
(``pixi.toml`` and ``experiments/``) instead of counting parent directories, so
callers do not depend on how deeply a module is nested.
"""

from __future__ import annotations

from pathlib import Path
import sys
import sysconfig


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root containing ``pixi.toml`` and ``experiments/``."""
    origin = (start or Path(__file__)).resolve()
    for candidate in origin.parents:
        if (candidate / "pixi.toml").is_file() and (candidate / "experiments").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate the repository root above {origin}")


try:
    REPO_ROOT = find_repo_root()
except RuntimeError:
    # A wheel has no repository. Relative user data/output paths use the caller's workspace.
    REPO_ROOT = Path.cwd()


def recipes_root() -> Path:
    checkout = REPO_ROOT / "experiments" / "recipes"
    if checkout.is_dir():
        return checkout
    for prefix in (
        Path(sys.prefix),
        Path(sysconfig.get_path("data")),
        Path(__file__).resolve().parents[1],
    ):
        installed = prefix / "share" / "imitationlab" / "recipes"
        if installed.is_dir():
            return installed
    raise FileNotFoundError(
        "ImitationLab recipes are missing; reinstall the package with its data files."
    )


def module_source_rel(module_name: str) -> str:
    """Repo-relative source path of a package module, for provenance hashing."""
    import importlib.util

    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        raise RuntimeError(f"Cannot resolve source file for {module_name}")
    return Path(spec.origin).resolve().relative_to(REPO_ROOT).as_posix()
