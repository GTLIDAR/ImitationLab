"""Run ``imitationlab.evaluation.evaluate_checkpoint`` with the CU130 torch bridge.

Every argument goes to the evaluator unchanged. ``eval_checkpoint_tree.py``
does the same for a checkpoint tree. The file name matters on the cluster:
``docker/cluster/run_singularity.sh`` gives ``scripts/tracking/eval*.py`` the
CU130 runtime Python (torch and NCCL). A script outside that pattern runs on
Kit's bundled Python, which has no NCCL.

    python scripts/tracking/eval_single_checkpoint.py --task imitation-g1-v1 ...
"""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "experiments")]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from imitationlab.runners.runtime_bootstrap import (  # noqa: E402
    configure_cu130_bridge,
    verify_cu130_torch,
)

EVALUATOR = "imitationlab.evaluation.evaluate_checkpoint"


def main() -> int:
    site_packages = configure_cu130_bridge(
        required=os.environ.get("ISAACLAB_REQUIRE_CU130_RUNTIME") == "1"
    )
    if site_packages is not None:
        verify_cu130_torch(site_packages)
    sys.argv = [f"{EVALUATOR}.py"] + sys.argv[1:]
    runpy.run_module(EVALUATOR, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
