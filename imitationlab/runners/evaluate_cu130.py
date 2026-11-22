"""Run ``imitationlab.evaluation.evaluate_checkpoint`` with the CU130 torch bridge.

Every argument goes to the evaluator unchanged. On the cluster,
``docker/cluster/run_singularity.sh`` runs evaluation modules on Kit's Python
with the CU130 runtime site-packages (torch and NCCL) injected; this module
configures and checks that bridge before the evaluator imports torch.

    python -m imitationlab.runners.evaluate_cu130 --task imitation-g1-v1 ...
"""

from __future__ import annotations

import os
import runpy
import sys

from imitationlab.runners.runtime_bootstrap import (
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
