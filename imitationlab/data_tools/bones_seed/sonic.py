"""The public SONIC release motion-name filter."""

# Verbatim default --filter-keywords list from
# gear_sonic/data_process/filter_and_copy_bones_data.py in
# NVlabs/GR00T-WholeBodyControl (the public SONIC release).
SONIC_RELEASE_FILTER_KEYWORDS = [
    "bed",
    "bike",
    "chair",
    "climb",
    "com_up_50cm",
    "sitting",
    "step_on",
    "seat",
    "table",
    "_sit_",
    "sit_",
    "ladder",
    "crutch",
    "_bed_",
    "_ride_",
    "scooter",
    "stepdown",
    "acrobatics_",
    "box_HSPU",
    "cartwheel",
    "50cm_box_",
    "on_box",
    "fall_from",
    "handstand_ff_",
    "on_1m",
    "form_box",
    "off_1m",
    "230m",
    "jump_over_obstacle_",
    "lift_crate_come_up_",
    "jump_to_shoulder_roll",
    "kozak_dance",
    "stair",
    "handstand",
    "box_jump",
    "monkey_jump",
    "safety_roll",
    "box_dips",
    "walking_on_edge",
    "push_obstacle",
]


def matched_keywords(name: str, keywords: list[str]) -> list[str]:
    name_lower = name.lower()
    return [kw for kw in keywords if kw.lower() in name_lower]
