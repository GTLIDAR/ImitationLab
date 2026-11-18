"""Kit-first bootstrap for ImitationLab PhysX training in the split CU130 runtime."""

from __future__ import annotations

import argparse
import os
import sys

from imitationlab.runners.runtime_bootstrap import (
    configure_cu130_bridge,
    detect_gpu_names,
    requested_backend,
    validate_gpu_policy,
    verify_cu130_torch,
)


def _requests_kit_window(argv: list[str]) -> bool:
    """Return whether the raw CLI tokens ask for a Kit window or a livestream."""
    for index, token in enumerate(argv):
        flag, has_value, value = token.partition("=")
        if flag not in ("--visualizer", "--viz", "--livestream"):
            continue
        if not has_value and index + 1 < len(argv):
            value = argv[index + 1]
        if flag == "--livestream":
            if value.strip() not in ("", "0", "-1"):
                return True
        elif "kit" in value.split(","):
            return True
    return os.environ.get("LIVESTREAM", "0") not in ("", "0")


def _parse_launcher_args(argv: list[str]):
    """Parse only arguments needed to launch Kit; ImitationLab parses the rest later."""
    # ``parse_known_args`` drops the old ``--headless`` flag here, and
    # ``train.py`` accepts it later through ``imitationlab.runners.launch_compat``.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--video", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args(argv)
    return args_cli


def main(argv: list[str] | None = None) -> int:
    """Verify CU130 Torch, start one SimulationApp, then invoke ImitationLab."""
    if argv is None:
        argv = sys.argv[1:]
    if "--assert-kitless" in argv:
        raise RuntimeError(
            "The PhysX bootstrap cannot be combined with --assert-kitless."
        )
    backend = requested_backend(argv)
    if backend != "physx":
        raise RuntimeError(
            "scripts/tracking/train_physx.py only owns the PhysX/Kit runtime. "
            "Run Newton through scripts/tracking/train.py with --assert-kitless."
        )

    gpu_names = detect_gpu_names()
    if not gpu_names and os.environ.get("ISAACLAB_REQUIRE_GPU_IDENTIFICATION") == "1":
        raise RuntimeError("Could not identify the visible GPU before PhysX startup.")
    experiment_flag = "--experimental-compute-only-physx"
    allow_compute_only_physx = (
        os.environ.get("ISAACLAB_ALLOW_COMPUTE_ONLY_PHYSX") == "1"
        or experiment_flag in argv
    )
    argv = [token for token in argv if token != experiment_flag]
    # Isaac Lab develop runs headless by default. A compute-only GPU cannot
    # render, so refuse only a request for the Kit window or a livestream.
    if allow_compute_only_physx and _requests_kit_window(argv):
        raise RuntimeError(
            "Experimental PhysX on a compute-only GPU must run headless: do not "
            "select the Kit visualizer (--visualizer kit) or a livestream."
        )
    validate_gpu_policy(
        "physx",
        gpu_names,
        allow_compute_only_physx=allow_compute_only_physx,
    )
    if gpu_names:
        print(f"[INFO] PhysX GPU policy accepted: {', '.join(gpu_names)}")
        if allow_compute_only_physx:
            print("[INFO] Experimental headless compute-only PhysX override enabled.")

    require_bridge = os.environ.get("ISAACLAB_REQUIRE_CU130_RUNTIME") == "1"
    site_packages = configure_cu130_bridge(required=require_bridge)
    if site_packages is not None:
        # Import the immutable CU130 Torch stack before Kit extensions can load
        # Isaac Sim's bundled NCCL into the process.  Once loaded, the dynamic
        # linker keeps the matching CU130 NCCL for the subsequent Kit startup.
        verify_cu130_torch(site_packages)

    # AppLauncher must be the first import that can start Isaac Sim/Kit.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(_parse_launcher_args(argv))
    try:
        from train import run

        status = run(argv, require_running_kit=True)
        # The training entry point historically returns None after a clean
        # run; normalize that conventional Python success result for the
        # cluster wrapper and its explicit completion marker.
        if status is None:
            status = 0
        if status == 0:
            success_marker = os.environ.get("ISAACLAB_WORKLOAD_SUCCESS_MARKER")
            if success_marker:
                from pathlib import Path

                Path(success_marker).touch()
            print("[INFO] ImitationLab PhysX workload completed successfully.")
        return status
    finally:
        app_launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
