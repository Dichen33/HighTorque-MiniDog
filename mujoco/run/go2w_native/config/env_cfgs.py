from __future__ import annotations

from copy import deepcopy

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.viewer import ViewerConfig
from mjlab.envs.mdp.actions.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.envs.mdp import dr as envs_dr

from minidog_native.config.env_cfgs import (
    flat_env_cfg,
    plum_blossom_env_cfg,
    rough_env_cfg,
)
from minidog_native.mdp.curriculums import terrain_levels_focused_obstacle
from ..mdp.rewards import (
    gap_landing_stability,
    gap_takeoff_progress,
    high_wall_clearance_progress,
    obstacle_feet_air_time,
    wheel_mode_leg_activity_l2,
    wheel_velocity_tracking,
)
from ..mdp.observations import terrain_context
from ..mdp.curriculums import terrain_levels_plum_obstacle_release
from ..robot_cfg import (
    GO2W_FOOT_BODY_NAMES,
    GO2W_JOINT_NAMES,
    GO2W_LEG_BODY_NAMES,
    GO2W_LEG_JOINT_NAMES,
    GO2W_WHEEL_GEOM_NAMES,
    GO2W_WHEEL_JOINT_NAMES,
    LEG_ACTION_SCALE,
    STAND_HEIGHT,
    WHEEL_ACTION_SCALE,
    WHEEL_RADIUS,
    get_robot_cfg,
)

def _foot_sensor() -> ContactSensorCfg:
    return ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="body", pattern=GO2W_FOOT_BODY_NAMES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=3,
        track_air_time=True,
    )


def _body_sensor(name: str, body_names: tuple[str, ...], history_length: int = 4) -> ContactSensorCfg:
    return ContactSensorCfg(
        name=name,
        primary=ContactMatch(mode="body", pattern=body_names, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=history_length,
    )


def _sensors() -> tuple[ContactSensorCfg, ...]:
    return (
        _foot_sensor(),
        _body_sensor("base_ground_contact", ("base",)),
        _body_sensor("self_collision", ("base",)),
        _body_sensor("thigh_ground_touch", ("FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh")),
        _body_sensor("shank_ground_touch", ("FR_calf", "FL_calf", "RR_calf", "RL_calf")),
        _body_sensor("trunk_ground_touch", ("base",)),
        RayCastSensorCfg(
            name="height_scanner",
            frame=ObjRef(type="body", name="base", entity="robot"),
            pattern=GridPatternCfg(resolution=0.08, size=(2.4, 1.0)),
            ray_alignment="yaw",
            max_distance=3.0,
            exclude_parent_body=True,
            include_geom_groups=(0,),
            debug_vis=False,
        ),
    )


def _actions() -> dict:
    actions = {}
    for joint_name in GO2W_LEG_JOINT_NAMES:
        actions[f"position_{joint_name}"] = JointPositionActionCfg(
            entity_name="robot", actuator_names=(joint_name,), scale=LEG_ACTION_SCALE, use_default_offset=True
        )
    for joint_name in GO2W_WHEEL_JOINT_NAMES:
        actions[f"velocity_{joint_name}"] = JointVelocityActionCfg(
            entity_name="robot", actuator_names=(joint_name,), scale=WHEEL_ACTION_SCALE, use_default_offset=True
        )
    return actions


def _adapt(cfg, obstacle_mode: bool = False):
    joint_asset = SceneEntityCfg("robot", joint_names=GO2W_JOINT_NAMES, preserve_order=True)
    leg_asset = SceneEntityCfg("robot", joint_names=GO2W_LEG_JOINT_NAMES, preserve_order=True)
    wheel_asset = SceneEntityCfg("robot", joint_names=GO2W_WHEEL_JOINT_NAMES, preserve_order=True)
    # MJLab's actuator DR terms index the configured actuator *groups* (the
    # robot has one position group for legs and one velocity group for wheels),
    # rather than individual MuJoCo actuator names.
    leg_actuator = SceneEntityCfg("robot", actuator_ids=[0])
    wheel_geom_asset = SceneEntityCfg("robot", geom_names=GO2W_WHEEL_GEOM_NAMES, preserve_order=True)
    foot_asset = SceneEntityCfg("robot", body_names=GO2W_FOOT_BODY_NAMES, preserve_order=True)
    cfg.scene.entities = {"robot": get_robot_cfg()}
    cfg.scene.sensors = _sensors()
    cfg.actions = _actions()
    cfg.viewer = ViewerConfig(body_name="base", distance=1.8, elevation=-18.0, azimuth=45.0)
    # Wheel traction and actuator response are dominant Go2W sim2sim
    # mismatch sources.  Model-field events run on reset so each episode sees
    # a new plausible hardware instance instead of one fixed instance per env.
    cfg.events["wheel_joint_friction"] = EventTermCfg(
        func=envs_dr.joint_friction,
        mode="reset",
        params={
            "asset_cfg": wheel_asset,
            "ranges": (0.002, 0.020),
            "operation": "abs",
            "shared_random": True,
        },
    )
    cfg.events["wheel_geom_friction"] = EventTermCfg(
        func=envs_dr.geom_friction,
        mode="reset",
        params={
            "asset_cfg": wheel_geom_asset,
            "ranges": (0.70, 1.30),
            "operation": "scale",
            "axes": [0],
            "shared_random": True,
        },
    )
    cfg.events["leg_joint_friction"] = EventTermCfg(
        func=envs_dr.joint_friction,
        mode="reset",
        params={
            "asset_cfg": leg_asset,
            "ranges": (0.0005, 0.0040),
            "operation": "abs",
            "shared_random": True,
        },
    )
    cfg.events["leg_joint_armature"] = EventTermCfg(
        func=envs_dr.joint_armature,
        mode="reset",
        params={
            "asset_cfg": leg_asset,
            "ranges": (0.0001, 0.0015),
            "operation": "abs",
            "shared_random": True,
        },
    )
    cfg.events["wheel_joint_armature"] = EventTermCfg(
        func=envs_dr.joint_armature,
        mode="reset",
        params={
            "asset_cfg": wheel_asset,
            "ranges": (0.0003, 0.0030),
            "operation": "abs",
            "shared_random": True,
        },
    )
    cfg.events["leg_joint_damping"] = EventTermCfg(
        func=envs_dr.joint_damping,
        mode="reset",
        params={
            "asset_cfg": leg_asset,
            "ranges": (0.005, 0.030),
            "operation": "abs",
            "shared_random": True,
        },
    )
    cfg.events["wheel_joint_damping"] = EventTermCfg(
        func=envs_dr.joint_damping,
        mode="reset",
        params={
            "asset_cfg": wheel_asset,
            "ranges": (0.005, 0.030),
            "operation": "abs",
            "shared_random": True,
        },
    )
    cfg.events["leg_pd_gains"] = EventTermCfg(
        func=envs_dr.pd_gains,
        mode="reset",
        params={
            "asset_cfg": leg_actuator,
            "kp_range": (0.90, 1.10),
            "kd_range": (0.80, 1.20),
            "operation": "scale",
        },
    )
    cfg.events["leg_effort_limit"] = EventTermCfg(
        func=envs_dr.effort_limits,
        mode="reset",
        params={
            "asset_cfg": leg_actuator,
            "effort_limit_range": (0.90, 1.10),
            "operation": "scale",
        },
    )
    cfg.events["leg_encoder_bias"] = EventTermCfg(
        func=envs_dr.encoder_bias,
        mode="reset",
        params={"asset_cfg": leg_asset, "bias_range": (-0.008, 0.008)},
    )
    cfg.events["wheel_encoder_bias"] = EventTermCfg(
        func=envs_dr.encoder_bias,
        mode="reset",
        params={"asset_cfg": wheel_asset, "bias_range": (-0.012, 0.012)},
    )
    cfg.events["robot_inertia_scale"] = EventTermCfg(
        func=envs_dr.pseudo_inertia,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
            # pseudo_inertia interprets alpha exponentially; +/-0.05 is
            # approximately a +/-10% mass/inertia scale, not (0.95, 1.05).
            "alpha_range": (-0.05, 0.05),
        },
    )
    if "critic" in cfg.observations:
        # Privileged task context reduces value interference across terrain
        # families without leaking terrain labels into the deployed actor.
        cfg.observations["critic"].terms["terrain_context"] = ObservationTermCfg(
            func=terrain_context,
            clip=(0.0, 1.0),
        )

    for group_name in ("actor_proprio", "critic"):
        group = cfg.observations[group_name]
        if "joint_pos" in group.terms:
            group.terms["joint_pos"].params["asset_cfg"] = joint_asset
            # Feed the same calibrated encoder error to the actor that a real
            # controller would expose.  The reset event above samples it per
            # episode; critic remains on the same observation contract.
            group.terms["joint_pos"].params["biased"] = True
        if "joint_vel" in group.terms:
            group.terms["joint_vel"].params["asset_cfg"] = joint_asset
    cfg.events["reset_joints"].params["asset_cfg"] = joint_asset
    cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg("robot", body_names=("base",))
    if "upright" in cfg.rewards:
        cfg.rewards["upright"].params["asset_cfg"] = SceneEntityCfg("robot", body_names=("base",))
    cfg.rewards["base_height"].params["target_height"] = STAND_HEIGHT
    cfg.rewards["stand_still"].params["target_height"] = STAND_HEIGHT

    for name in ("pose", "stand_pose", "leg_mirror", "diagonal_leg_symmetry"):
        if name in cfg.rewards:
            cfg.rewards[name].params["asset_cfg"] = leg_asset
    for name in ("stand_foot_geometry", "diagonal_foot_symmetry", "stance_width", "foot_clearance", "foot_swing_height", "foot_slip"):
        if name in cfg.rewards:
            cfg.rewards[name].params["asset_cfg"] = foot_asset

    # Wheel rolling is the primary flat/rough locomotion mode. Leg swing
    # rewards and foot-slip penalties would incorrectly penalize four wheels
    # while they are rolling in contact with the floor.
    for name in ("feet_air_time", "feet_gait_consistency", "foot_clearance", "foot_swing_height", "foot_slip", "diagonal_foot_symmetry"):
        if name in cfg.rewards:
            cfg.rewards[name].weight = 0.0
    cfg.rewards["wheel_velocity_tracking"] = RewardTermCfg(
        func=wheel_velocity_tracking,
        weight=0.75,
        params={
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_WHEEL_JOINT_NAMES, preserve_order=True),
            "wheel_radius": WHEEL_RADIUS,
            "obstacle_scale": 0.25 if obstacle_mode else 1.0,
        },
    )
    cfg.rewards["wheel_mode_leg_activity"] = RewardTermCfg(
        func=wheel_mode_leg_activity_l2,
        weight=-0.04,
        params={
            "command_name": "twist",
            "asset_cfg": leg_asset,
            "position_weight": 0.25,
            "velocity_weight": 1.0,
        },
    )
    cfg.rewards["gap_takeoff_progress"] = RewardTermCfg(
        func=gap_takeoff_progress,
        weight=0.0,
        params={"command_name": "twist", "gap_start_x": 1.15},
    )
    cfg.rewards["gap_landing_stability"] = RewardTermCfg(
        func=gap_landing_stability,
        weight=0.0,
        params={"command_name": "twist", "gap_end_x": 2.20},
    )
    cfg.rewards["high_wall_clearance_progress"] = RewardTermCfg(
        func=high_wall_clearance_progress,
        weight=0.0,
        params={
            "command_name": "twist",
            "asset_cfg": foot_asset,
            "wall_centers_x": (1.40, 2.95, 4.50),
        },
    )
    if obstacle_mode:
        cfg.rewards["gap_takeoff_progress"].weight = 0.20
        cfg.rewards["gap_landing_stability"].weight = 0.20
        cfg.rewards["high_wall_clearance_progress"].weight = 0.20
    if obstacle_mode and "feet_air_time" in cfg.rewards:
        cfg.rewards["feet_air_time"].func = obstacle_feet_air_time
        cfg.rewards["feet_air_time"].weight = 0.25
        cfg.rewards["feet_air_time"].params["obstacle_difficulty_threshold"] = 0.10
    cfg.rewards["track_lin_vel"].weight = 3.5
    if not obstacle_mode and "pose" in cfg.rewards:
        # Flat and rough pretraining should keep the legs near the standing
        # pose while the wheels learn the primary locomotion mode.
        cfg.rewards["pose"].weight = 0.0
    if "stand_contact" in cfg.rewards:
        cfg.rewards["stand_contact"].weight = -0.35
    if "stand_foot_geometry" in cfg.rewards:
        cfg.rewards["stand_foot_geometry"].weight = -0.30
    if "foot_clearance" in cfg.rewards:
        cfg.rewards["foot_clearance"].weight = 0.0
    cfg.terminations["base_ground_contact"].params["sensor_name"] = "base_ground_contact"
    cfg.terminations["bad_orientation"].params["limit_angle"] = 1.0
    return cfg


def go2w_flat_env_cfg(play: bool = False):
    return _adapt(flat_env_cfg(play=play))


def go2w_rough_env_cfg(play: bool = False):
    return _adapt(rough_env_cfg(play=play))


def go2w_step_env_cfg(play: bool = False):
    """Low-obstacle release stage between wheel pretraining and hard courses."""
    cfg = _adapt(rough_env_cfg(play=play), obstacle_mode=True)
    cfg.rewards["wheel_velocity_tracking"].weight = 0.65
    cfg.rewards["track_lin_vel"].weight = 3.1
    if "pose" in cfg.rewards:
        cfg.rewards["pose"].weight = 0.30
    if "foot_clearance" in cfg.rewards:
        cfg.rewards["foot_clearance"].weight = -0.80
    if "foot_swing_height" in cfg.rewards:
        cfg.rewards["foot_swing_height"].weight = -0.10
    return cfg


def go2w_plum_blossom_env_cfg(play: bool = False):
    cfg = _adapt(plum_blossom_env_cfg(play=play), obstacle_mode=True)
    if not play:
        # Use the Go2W LP-ACRL implementation for the unified task.  The
        # focused gap/wall tasks intentionally keep their interpretable
        # single-obstacle curriculum.
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
            func=terrain_levels_plum_obstacle_release,
            params={
                "command_name": "twist",
                "success_distance": 2.7,
                "gap_success_distance": 3.35,
                "fail_distance_ratio": 0.35,
                "wall_success_distance": 4.80,
                "default_success_distance": 4.0,
            },
        )
    # Keep the first Go2W curriculum stage wheel-friendly. Difficulty is
    # released by the existing measured terrain gates, not at iteration zero.
    cfg.rewards["wheel_velocity_tracking"].weight = 0.55
    cfg.rewards["track_lin_vel"].weight = 2.8
    if "feet_air_time" in cfg.rewards:
        cfg.rewards["feet_air_time"].func = obstacle_feet_air_time
        cfg.rewards["feet_air_time"].weight = 0.35
        cfg.rewards["feet_air_time"].params["obstacle_difficulty_threshold"] = 0.10
    return cfg


def _single_obstacle_env_cfg(
    play: bool,
    terrain_name: str,
    success_distance: float,
    history_terrain_names: tuple[str, ...] = (),
    revisit_proportion: float = 0.10,
):
    """Create a focused obstacle task from the shared medium-to-hard course.

    Keeping one terrain column makes success rates interpretable and prevents
    a difficult obstacle from being hidden by easier terrain columns.  The
    terrain generator still owns all difficulty rows, so the same curriculum
    starts below the XML medium anchor and can progress beyond it.
    """
    if terrain_name not in {"flat_gap", "flat_high_wall"}:
        raise ValueError(f"Unsupported focused Go2W terrain: {terrain_name}")

    cfg = _adapt(plum_blossom_env_cfg(play=play), obstacle_mode=True)
    generator = deepcopy(cfg.scene.terrain.terrain_generator)
    terrain_cfg = deepcopy(generator.sub_terrains[terrain_name])
    history_names = tuple(name for name in history_terrain_names if name in generator.sub_terrains and name != terrain_name)
    history_fraction = min(max(revisit_proportion, 0.0), 0.40)
    if history_names:
        terrain_cfg.proportion = 1.0 - history_fraction
        history_cfgs = {}
        each_fraction = history_fraction / len(history_names)
        for name in history_names:
            history_cfg = deepcopy(generator.sub_terrains[name])
            history_cfg.proportion = each_fraction
            history_cfgs[name] = history_cfg
        generator.sub_terrains = {terrain_name: terrain_cfg, **history_cfgs}
    else:
        terrain_cfg.proportion = 1.0
        generator.sub_terrains = {terrain_name: terrain_cfg}
    generator.num_cols = len(generator.sub_terrains)
    generator.curriculum = not play
    cfg.scene.terrain.terrain_generator = generator

    if not play:
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
            func=terrain_levels_focused_obstacle,
            params={
                "command_name": "twist",
                "success_distance": success_distance,
                "fail_distance_ratio": 0.35,
                "primary_terrain_name": terrain_name,
            },
        )
    cfg.rewards["course_progress"].params["gap_success_distance"] = success_distance
    cfg.rewards["course_progress"].params["wall_success_distance"] = success_distance
    cfg.rewards["course_completion"].params["gap_success_distance"] = success_distance
    cfg.rewards["course_completion"].params["wall_success_distance"] = success_distance
    # The foothold term is meaningful only on stepping stones, so focused gap
    # and wall tasks must not carry a dead reward into their score.
    cfg.rewards["plum_foothold"].weight = 0.0
    if terrain_name == "flat_gap":
        cfg.rewards["gap_crossing"].weight = 5.0
        cfg.rewards["gap_takeoff_progress"].weight = 0.55
        cfg.rewards["gap_landing_stability"].weight = 0.65
    else:
        cfg.rewards["gap_crossing"].weight = 0.0
        cfg.rewards["high_wall_clearance_progress"].weight = 0.70
    return cfg


def go2w_gap_env_cfg(play: bool = False):
    return _single_obstacle_env_cfg(
        play,
        "flat_gap",
        success_distance=3.35,
        history_terrain_names=("flat", "random_rough", "pyramid_stairs"),
        revisit_proportion=0.10,
    )


def go2w_high_wall_env_cfg(play: bool = False):
    # The third wall is at x=6.0 in terrain-local coordinates and the spawn
    # origin is x=1.5; 4.8 m relative displacement clears the final wall
    # while remaining inside the generated 8 m terrain patch.
    return _single_obstacle_env_cfg(
        play,
        "flat_high_wall",
        success_distance=4.80,
        history_terrain_names=("pyramid_stairs", "flat_gap"),
        revisit_proportion=0.10,
    )


def go2w_mixed_env_cfg(play: bool = False):
    """Full adaptive course used only after focused obstacle tasks stabilize."""
    return go2w_plum_blossom_env_cfg(play=play)

