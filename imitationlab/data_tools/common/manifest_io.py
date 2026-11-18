"""Manifest schema access and small file helpers shared by the data tools.

The clip-manifest schema authority is
``imitationlab.envs.isaaclab.tasks.manager_based.imitation.motion_manifest``.
That module is stdlib-only, but importing it by its package name runs
``imitationlab.envs.isaaclab.tasks.__init__``, which imports Isaac Lab to
register the Gym tasks. The data tools must run in the default (CPU) Pixi
environment, so :func:`motion_manifest_schema` loads the same source file
directly. In a process that already imported the task package, the loaded canonical
module is returned instead.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

MOTION_MANIFEST_MODULE = (
    "imitationlab.envs.isaaclab.tasks.manager_based.imitation.motion_manifest"
)
_MOTION_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "envs"
    / "isaaclab"
    / "tasks"
    / "manager_based"
    / "imitation"
    / "motion_manifest.py"
)
# Private name for the copy loaded by path, so it never shadows the canonical
# package module in ``sys.modules``.
_STANDALONE_NAME = "imitationlab.data_tools._motion_manifest"


@functools.cache
def _load_standalone() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        _STANDALONE_NAME, _MOTION_MANIFEST_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load the motion manifest schema: {_MOTION_MANIFEST_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_STANDALONE_NAME] = module
    spec.loader.exec_module(module)
    return module


def motion_manifest_schema() -> ModuleType:
    """Return the motion-manifest schema module without importing Isaac Lab."""
    loaded = sys.modules.get(MOTION_MANIFEST_MODULE)
    if loaded is not None:
        return loaded
    return _load_standalone()


def sanitize_motion_name(value: str | Path) -> str:
    """Map a relative clip path or name to a manifest motion name.

    Path separators, dashes, and other characters outside ``[A-Za-z0-9_]``
    become ``_``; repeated underscores collapse and edge underscores are
    stripped. An empty result becomes ``"motion"``. Existing manifests depend
    on these names, so do not change the mapping.
    """
    text = value.as_posix() if isinstance(value, Path) else str(value)
    name = re.sub(r"[^A-Za-z0-9_\-/]+", "_", text)
    name = name.replace("/", "__").replace("-", "_")
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "motion"


def write_json(path: Path, payload: object) -> None:
    """Write indented JSON atomically, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".tmp")
    staged.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    staged.replace(path)
