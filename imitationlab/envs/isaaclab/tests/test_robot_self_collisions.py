"""``robot_self_collisions`` writes the articulation-root flag before the scene is built."""

from __future__ import annotations

from types import SimpleNamespace

from isaaclab.sim.schemas import ArticulationRootPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from imitationlab.envs.isaaclab.envs.imitation_rl_env import (
    _apply_robot_self_collisions,
)


def _cfg(value, props=None):
    spawn = UsdFileCfg(usd_path="robot.usd", articulation_props=props)
    return SimpleNamespace(
        robot_self_collisions=value,
        scene=SimpleNamespace(robot=SimpleNamespace(spawn=spawn)),
    )


def test_none_keeps_the_asset_setting() -> None:
    cfg = _cfg(None)
    _apply_robot_self_collisions(cfg)
    assert cfg.scene.robot.spawn.articulation_props is None


def test_false_overrides_an_unset_root() -> None:
    cfg = _cfg(False)
    _apply_robot_self_collisions(cfg)
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is False
    assert cfg.scene.robot.spawn.usd_path == "robot.usd"


def test_false_keeps_the_other_root_properties() -> None:
    props = ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8
    )
    cfg = _cfg(False, props)
    _apply_robot_self_collisions(cfg)
    out = cfg.scene.robot.spawn.articulation_props
    assert out.enabled_self_collisions is False
    assert out.solver_position_iteration_count == 8


def test_filtered_pairs_are_authored_between_named_bodies() -> None:
    import pytest
    from pxr import Usd, UsdGeom, UsdPhysics

    from imitationlab.envs.isaaclab.envs.imitation_rl_env import (
        author_filtered_body_pairs,
    )

    stage = Usd.Stage.CreateInMemory()
    robot = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot").GetPrim()
    for name in ("left_wrist_roll_link", "left_wrist_yaw_link", "torso_link"):
        UsdPhysics.RigidBodyAPI.Apply(
            stage.DefinePrim(f"/World/envs/env_0/Robot/{name}", "Xform")
        )
    author_filtered_body_pairs(robot, [("left_wrist_roll_link", "left_wrist_yaw_link")])
    roll = stage.GetPrimAtPath("/World/envs/env_0/Robot/left_wrist_roll_link")
    targets = UsdPhysics.FilteredPairsAPI(roll).GetFilteredPairsRel().GetTargets()
    assert [str(p) for p in targets] == ["/World/envs/env_0/Robot/left_wrist_yaw_link"]
    torso = stage.GetPrimAtPath("/World/envs/env_0/Robot/torso_link")
    assert not torso.HasAPI(UsdPhysics.FilteredPairsAPI)
    with pytest.raises(ValueError, match="no rigid body"):
        author_filtered_body_pairs(robot, [("left_wrist_roll_link", "no_such_link")])


def test_body_pairs_parse_from_cli_strings() -> None:
    import pytest

    from imitationlab.envs.isaaclab.envs.imitation_rl_env import _body_pair

    assert _body_pair("left_wrist_roll_link:left_wrist_yaw_link") == (
        "left_wrist_roll_link",
        "left_wrist_yaw_link",
    )
    assert _body_pair(["a", "b"]) == ("a", "b")
    with pytest.raises(ValueError):
        _body_pair("a:b:c")
