"""Names of the skill encoder's window frames (``env.expert_macro_anchor_mode``).

``robot``
    The first convention: pretraining windows at the reference's full pose,
    live windows at the robot's full pose.
``expert_heading``
    Pretraining and live windows both in the yaw frame of the window's own
    first reference frame, at its xy.
``robot_pos_heading``
    Live windows in the live robot's yaw frame at the robot's xy, so the
    encoder sees the planar tracking error and a deployed policy needs a
    position estimate as well as an IMU. SONIC v1.1 uses the same yaw frame
    for its reference orientation but has no position term. This mode was
    called ``robot_heading`` until the position dependence was made explicit;
    encoders recorded under that name load as this one.
``reference_heading``
    Live windows are the explicit goal's own numbers: the yaw frame and full
    position of the reference frame the robot is on, yaw taken from the body
    x-axis projected onto the ground. Nothing is read from the robot, so a
    deployed policy needs only an IMU (for the proprioception).
"""

from __future__ import annotations

ANCHOR_MODES = ("robot", "expert_heading", "robot_pos_heading", "reference_heading")
_EARLIER_NAMES = {"robot_heading": "robot_pos_heading"}
# What ``reference_heading`` fixes in the encoder record.
REFERENCE_HEADING = {
    "rollout_anchor": "previous_reference",
    "macro_origin": "xyz",
    "macro_heading": "forward",
}


def canonical_anchor_mode(mode: object) -> str:
    """The current name of ``mode``; raises on a name that never existed."""
    name = str(mode).strip()
    name = _EARLIER_NAMES.get(name, name)
    if name not in ANCHOR_MODES:
        msg = f"expert_macro_anchor_mode must be one of {ANCHOR_MODES}, got {mode!r}."
        raise ValueError(msg)
    return name
