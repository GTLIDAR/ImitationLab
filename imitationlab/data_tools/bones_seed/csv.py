"""Normalize raw BONES-SEED G1 CSVs for the batched Isaac converter."""

import csv
import importlib.util
from pathlib import Path

import numpy as np


def _load_unitree_g1_joint_order() -> tuple[str, ...]:
    """Load the repo's canonical G1 order without importing Isaac task modules."""
    module_path = (
        Path(__file__).resolve().parents[3]
        / "imitationlab"
        / "envs"
        / "isaaclab"
        / "assets"
        / "robots"
        / "unitree_joint_order.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_imitationlab_unitree_joint_order", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Unitree joint-order module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module.UNITREE_G1_29DOF_SDK_JOINT_NAMES)


UNITREE_G1_29DOF_JOINT_NAMES = _load_unitree_g1_joint_order()

BONES_G1_ROOT_COLUMNS: tuple[str, ...] = (
    "Frame",
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
)
BONES_G1_EXPECTED_COLUMNS: tuple[str, ...] = BONES_G1_ROOT_COLUMNS + tuple(
    f"{joint_name}_dof" for joint_name in UNITREE_G1_29DOF_JOINT_NAMES
)


def _read_csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle), [])


def _validate_bones_g1_columns(path: Path) -> None:
    header = _read_csv_header(path)
    if tuple(header) != BONES_G1_EXPECTED_COLUMNS:
        expected_joint_columns = list(
            BONES_G1_EXPECTED_COLUMNS[len(BONES_G1_ROOT_COLUMNS) :]
        )
        actual_joint_columns = header[len(BONES_G1_ROOT_COLUMNS) :]
        raise ValueError(
            "BONES-SEED G1 CSV joint columns do not match this repo's Unitree "
            f"G1 order for {path}.\n"
            f"Expected: {expected_joint_columns}\n"
            f"Actual:   {actual_joint_columns}"
        )


def normalize_bones_csv(source: Path, target: Path) -> None:
    """Write root metres, XYZW quaternions, and SDK-ordered joint radians."""
    from scipy.spatial.transform import Rotation

    _validate_bones_g1_columns(source)
    data = np.loadtxt(source, delimiter=",", skiprows=1, dtype=np.float64)
    data = np.atleast_2d(data)
    if data.shape[1] != 36 or data.shape[0] < 2 or not np.isfinite(data).all():
        raise ValueError(
            f"{source} needs at least two finite frames with 36 columns including Frame."
        )

    root_pos_m = data[:, 1:4] * 0.01
    # BONES seed-viewer uses Three.js Euler(rx, ry, rz, "ZYX"). In SciPy,
    # lowercase "xyz" with the CSV X/Y/Z columns is the matching extrinsic XYZ
    # convention.
    root_quat_xyzw = Rotation.from_euler("xyz", data[:, 4:7], degrees=True).as_quat()
    joint_pos_rad = np.deg2rad(data[:, 7:])
    output = np.concatenate([root_pos_m, root_quat_xyzw, joint_pos_rad], axis=1)
    temp_target = target.with_suffix(target.suffix + ".tmp")
    np.savetxt(temp_target, output.astype(np.float32), delimiter=",", fmt="%.9g")
    temp_target.replace(target)
