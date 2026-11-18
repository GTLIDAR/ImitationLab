"""Write large checkpoints atomically through a staging file with room for them."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

# Zip framing, pickled metadata and alignment on top of the raw tensor bytes.
_OVERHEAD_FACTOR = 1.1
_OVERHEAD_BYTES = 64 << 20


def tensor_nbytes(obj: Any) -> int:
    """Total tensor storage referenced by a nested checkpoint state."""
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, Mapping):
        return sum(tensor_nbytes(value) for value in obj.values())
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return sum(tensor_nbytes(value) for value in obj)
    return 0


def _free_bytes(path: Path) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


def staging_dir(target: Path, nbytes: int) -> Path:
    """Pick where to write the staged file before moving it to ``target``.

    The temporary directory (``$TMPDIR``, else the platform default) comes
    first: on clusters it is node-local disk, and torch's zip writer can fail
    on network filesystems. When it lacks room, as with a small RAM-backed
    ``/tmp``, stage beside the target instead; the move is then a rename.
    """
    need = int(nbytes * _OVERHEAD_FACTOR) + _OVERHEAD_BYTES
    temporary = Path(tempfile.gettempdir())
    if _free_bytes(temporary) >= need:
        return temporary
    return target.parent


def save_staged(state: Any, target: str | Path) -> None:
    """``torch.save`` to ``target`` so readers never see a partial file."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    nbytes = tensor_nbytes(state)
    root = staging_dir(target, nbytes)
    handle, staged = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".pt", dir=root)
    os.close(handle)
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(staged, 0o666 & ~umask)
    try:
        try:
            torch.save(state, staged)
        except RuntimeError as error:
            raise RuntimeError(
                f"Could not write a {nbytes / 2**30:.1f} GiB checkpoint to {root} "
                f"({_free_bytes(Path(root)) / 2**30:.1f} GiB free). Point TMPDIR "
                "at a directory with more space."
            ) from error
        shutil.move(staged, target)
    finally:
        if os.path.exists(staged):
            os.unlink(staged)
