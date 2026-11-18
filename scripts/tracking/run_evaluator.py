"""Deprecated alias of ``scripts/tracking/eval_single_checkpoint.py``.

Use ``eval_single_checkpoint.py``: its name matches the cluster runtime
selection in ``docker/cluster/run_singularity.sh``, and this alias's does not.
"""

from __future__ import annotations

from pathlib import Path
import runpy
import warnings

if __name__ == "__main__":
    warnings.warn(
        "scripts/tracking/run_evaluator.py is deprecated; use "
        "scripts/tracking/eval_single_checkpoint.py",
        DeprecationWarning,
        stacklevel=1,
    )
    runpy.run_path(
        str(Path(__file__).resolve().with_name("eval_single_checkpoint.py")),
        run_name="__main__",
    )
