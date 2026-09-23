from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr as envs_dr
from mjlab.envs.mdp.actions.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    BoxRandomGridTerrainCfg,
    HfPerlinNoiseTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    TerrainEntityCfg,
    TerrainGeneratorCfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from ..mdp.curriculums import terrain_levels_velocity
from ..mdp.rewards import (
    base_height_l2,
    contact_fraction_obs,
    contact_fraction_reward,
    diagonal_foot_symmetry,
    diagonal_leg_symmetry,
    feet_gait_consistency,
    foot_clearance,
    foot_swing_height,
    foot_slip,
    flat_orientation_l2,
    joint_deviation_l2,
    leg_mirror,
    lin_vel_z_l2,
    safe_base_lin_vel,
    safe_height_scan,
    stance_width,
    stand_contact,
    stand_foot_geometry,
    stand_pose,
    stand_still,
    terrain_level_bonus,
    track_linear_velocity,
    track_yaw_velocity,
)
from ..mdp.observations import gait_phase
from ..robot_cfg import (
    ACTION_SCALE,
    BASE_GEOM_PATTERNS,
    CALF_GEOM_PATTERNS,
    CRAWL_HEIGHT,
    FOOT_BODY_PATTERNS,
    FOOT_GEOM_PATTERNS,
    LEG_JOINT_PATTERNS,
    STAND_HEIGHT,
    THIGH_GEOM_PATTERNS,
)
from ..robot_cfg import get_robot_cfg, get_robot_crawl_cfg


def _foot_contact_sensor() -> ContactSensorCfg:
    return ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="body", pattern=FOOT_BODY_PATTERNS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=3,
        track_air_time=True,
    )


def _base_contact_sensor() -> ContactSensorCfg:
    return ContactSensorCfg(
        name="base_ground_contact",
        primary=ContactMatch(mode="body", pattern="base_link", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
        history_length=4,
    )


def _self_collision_sensor() -> ContactSensorCfg:
    return ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )


def _thigh_ground_sensor() -> ContactSensorCfg:
    return ContactSensorCfg(
        name="thigh_ground_touch",
        primary=ContactMatch(mode="geom", pattern=THIGH_GEOM_PATTERNS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )


def _shank_ground_sensor() -> ContactSensorCfg:
    return ContactSensorCfg(
        name="shank_ground_touch",
        primary=ContactMatch(mode="geom", pattern=CALF_GEOM_PATTERNS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )


def _trunk_ground_sensor() -> ContactSensorCfg:
    return ContactSensorCfg(
        name="trunk_ground_touch",
        primary=ContactMatch(mode="geom", pattern=BASE_GEOM_PATTERNS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )


def _height_scan_sensor() -> RayCastSensorCfg:
    return RayCastSensorCfg(
        name="height_scanner",
        frame=ObjRef(type="body", name="base_link", entity="robot"),
        pattern=GridPatternCfg(resolution=0.08, size=(1.4, 0.9)),
        ray_alignment="yaw",
        max_distance=3.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )


def _make_base_env_cfg() -> ManagerBasedRlEnvCfg:
    joint_asset = SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERNS)

    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            scale=0.25,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=velocity_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "command": ObservationTermCfg(
            func=velocity_mdp.generated_commands,
            params={"command_name": "twist"},
        ),
        "gait_phase": ObservationTermCfg(
            func=gait_phase,
            params={
                "command_name": "twist",
                "command_threshold": 0.03,
                "stationary_frequency": 0.0,
                "base_frequency": 1.8,
                "max_frequency": 2.8,
                "speed_scale": 1.0,
            },
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={"asset_cfg": joint_asset},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={"asset_cfg": joint_asset},
            scale=0.05,
            noise=Unoise(n_min=-1.0, n_max=1.0),
        ),
        "actions": ObservationTermCfg(func=velocity_mdp.last_action, history_length=1),
    }
    critic_terms = {
        **actor_terms,
        "base_lin_vel": ObservationTermCfg(func=safe_base_lin_vel, scale=2.0),
        "foot_contact": ObservationTermCfg(
            func=contact_fraction_obs,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "height_scan": ObservationTermCfg(
            func=safe_height_scan,
            params={"sensor_name": "height_scanner"},
            clip=(-1.0, 1.0),
        ),
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=LEG_JOINT_PATTERNS,
            scale=ACTION_SCALE,
            use_default_offset=True,
        )
    }

    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.15,
            rel_heading_envs=0.5,
            heading_command=True,
            heading_control_stiffness=0.6,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.35, 0.45),
                lin_vel_y=(-0.18, 0.18),
                ang_vel_z=(-0.8, 0.8),
                heading=(-math.pi, math.pi),
            ),
        )
    }

    events = {
        "reset_scene": EventTermCfg(func=envs_mdp.reset_scene_to_default, mode="reset"),
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"z": (0.22, 0.28), "yaw": (-math.pi, math.pi)},
                "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.1, 0.1)},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "reset_joints": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={"position_range": (-0.03, 0.03), "velocity_range": (0.0, 0.0), "asset_cfg": joint_asset},
        ),
        "push_robot": EventTermCfg(
            func=envs_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(8.0, 12.0),
            params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, "asset_cfg": SceneEntityCfg("robot")},
        ),
        "base_com": EventTermCfg(
            func=envs_dr.body_com_offset,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
                "operation": "add",
                "ranges": {0: (-0.02, 0.02), 1: (-0.02, 0.02), 2: (-0.015, 0.015)},
            },
        ),
        "actuator_stiffness": EventTermCfg(
            func=envs_dr.joint_stiffness,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot"), "ranges": (0.9, 1.1), "operation": "scale"},
        ),
        "actuator_damping": EventTermCfg(
            func=envs_dr.joint_damping,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot"), "ranges": (0.9, 1.1), "operation": "scale"},
        ),
    }

    rewards = {
        "track_lin_vel": RewardTermCfg(func=track_linear_velocity, weight=2.5, params={"std": 0.35, "command_name": "twist"}),
        "track_yaw_vel": RewardTermCfg(func=track_yaw_velocity, weight=1.2, params={"std": 0.45, "command_name": "twist"}),
        "upright": RewardTermCfg(func=velocity_mdp.upright, weight=1.0, params={"std": 0.5, "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))}),
        "base_height": RewardTermCfg(func=base_height_l2, weight=-1.5, params={"target_height": STAND_HEIGHT}),
        "flat_orientation": RewardTermCfg(func=flat_orientation_l2, weight=-0.8),
        "lin_vel_z": RewardTermCfg(func=lin_vel_z_l2, weight=-0.2),
        "joint_torques": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-2.0e-4),
        "joint_acc": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-2.5e-7),
        "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
        "joint_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-1.0),
        "stand_still": RewardTermCfg(func=stand_still, weight=-0.2, params={"command_name": "twist", "command_threshold": 0.08}),
        "feet_contact": RewardTermCfg(func=contact_fraction_reward, weight=0.2, params={"sensor_name": "feet_ground_contact"}),
        "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-50.0),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "bad_orientation": TerminationTermCfg(func=envs_mdp.bad_orientation, params={"limit_angle": 1.0}),
        "base_ground_contact": TerminationTermCfg(func=velocity_mdp.illegal_contact, params={"sensor_name": "base_ground_contact"}),
        "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=2048,
            env_spacing=2.5,
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=TerrainGeneratorCfg(
                    size=(6.0, 6.0),
                    border_width=10.0,
                    num_rows=5,
                    num_cols=5,
                    sub_terrains={"flat": BoxFlatTerrainCfg(proportion=1.0, size=(6.0, 6.0))},
                ),
            ),
            sensors=(
                _foot_contact_sensor(),
                _base_contact_sensor(),
                _height_scan_sensor(),
                _self_collision_sensor(),
                _thigh_ground_sensor(),
                _shank_ground_sensor(),
                _trunk_ground_sensor(),
            ),
        ),
        commands=commands,
        actions=actions,
        observations={
            "actor": ObservationGroupCfg(terms=actor_terms, concatenate_terms=True, enable_corruption=True, history_length=6),
            "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True, enable_corruption=False),
        },
        rewards=rewards,
        terminations=terminations,
        events=events,
        metrics={"mean_action_acc": MetricsTermCfg(func=velocity_mdp.mean_action_acc)},
        curriculum={},
        decimation=10,
        episode_length_s=20.0,
        sim=SimulationCfg(mujoco=MujocoCfg(impratio=100, cone="elliptic")),
        viewer=ViewerConfig(body_name="base_link", distance=1.4, elevation=-20.0, azimuth=45.0),
    )


def flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _make_base_env_cfg()
    cfg.scene.entities = {"robot": get_robot_cfg()}
    if play:
        cfg.scene.num_envs = 1
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
    return cfg


def rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _make_base_env_cfg()
    cfg.scene.entities = {"robot": get_robot_cfg()}
    joint_asset = SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERNS, preserve_order=True)
    foot_asset = SceneEntityCfg("robot", body_names=FOOT_BODY_PATTERNS, preserve_order=True)
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            curriculum=True,
            sub_terrains={
                "flat": BoxFlatTerrainCfg(proportion=0.10, size=(8.0, 8.0)),
                "stairs_up": BoxPyramidStairsTerrainCfg(proportion=0.20, step_height_range=(0.0, 0.18), step_width=0.28, size=(8.0, 8.0)),
                "stairs_down": BoxInvertedPyramidStairsTerrainCfg(proportion=0.15, step_height_range=(0.0, 0.16), step_width=0.28, size=(8.0, 8.0)),
                "blocks": BoxRandomGridTerrainCfg(proportion=0.20, grid_width=0.40, grid_height_range=(0.0, 0.12), size=(8.0, 8.0)),
                "random_rough": HfRandomUniformTerrainCfg(proportion=0.15, noise_range=(0.0, 0.04), noise_step=0.01, horizontal_scale=0.16, downsampled_scale=0.16, border_width=0.25, base_thickness_ratio=100.0, size=(8.0, 8.0)),
                "perlin_noise": HfPerlinNoiseTerrainCfg(proportion=0.10, height_range=(0.0, 0.05), octaves=2, persistence=0.4, lacunarity=2.0, horizontal_scale=0.16, resolution=0.16, border_width=0.25, base_thickness_ratio=100.0, size=(8.0, 8.0)),
                "slope": HfPyramidSlopedTerrainCfg(proportion=0.10, slope_range=(0.03, 0.20), platform_width=2.0, border_width=0.25, base_thickness_ratio=100.0, horizontal_scale=0.16, size=(8.0, 8.0)),
            },
        ),
        max_init_terrain_level=0,
    )
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(func=terrain_levels_velocity, params={"command_name": "twist"})
    cfg.rewards["track_lin_vel"].weight = 3.0
    cfg.rewards["track_yaw_vel"].weight = 1.0
    cfg.rewards["base_height"].weight = -1.0
    cfg.rewards["flat_orientation"].weight = -1.1
    cfg.rewards["lin_vel_z"].weight = -0.35
    cfg.rewards["pose"] = RewardTermCfg(
        func=velocity_mdp.variable_posture,
        weight=0.80,
        params={
            "asset_cfg": joint_asset,
            "command_name": "twist",
            "walking_threshold": 0.05,
            "running_threshold": 1.0,
            "std_standing": {
                ".*_hip_joint": 0.05,
                ".*_thigh_joint": 0.05,
                ".*_calf_joint": 0.08,
            },
            "std_walking": {
                ".*_hip_joint": 0.22,
                ".*_thigh_joint": 0.32,
                ".*_calf_joint": 0.60,
            },
            "std_running": {
                ".*_hip_joint": 0.28,
                ".*_thigh_joint": 0.42,
                ".*_calf_joint": 0.75,
            },
        },
    )
    cfg.rewards["stand_still"].weight = -0.55
    cfg.rewards["stand_still"].params["command_threshold"] = 0.03
    cfg.rewards["stand_still"].params["lin_vel_threshold"] = 0.04
    cfg.rewards["stand_still"].params["yaw_vel_threshold"] = 0.06
    cfg.rewards["stand_still"].params["height_tol"] = 0.02
    cfg.rewards["stand_still"].params["tilt_tol"] = 0.12
    cfg.rewards["stand_still"].params["target_height"] = STAND_HEIGHT
    cfg.rewards["stand_pose"] = RewardTermCfg(
        func=stand_pose,
        weight=-0.45,
        params={
            "command_name": "twist",
            "command_threshold": 0.06,
            "hip_weight": 1.4,
            "thigh_weight": 1.8,
            "calf_weight": 1.8,
            "velocity_weight": 0.04,
            "asset_cfg": joint_asset,
        },
    )
    cfg.rewards["stand_contact"] = RewardTermCfg(
        func=stand_contact,
        weight=-0.80,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist", "command_threshold": 0.06},
    )
    cfg.rewards["stand_foot_geometry"] = RewardTermCfg(
        func=stand_foot_geometry,
        weight=-1.20,
        params={
            "command_name": "twist",
            "command_threshold": 0.06,
            "min_width": 0.18,
            "min_length": 0.18,
            "asset_cfg": foot_asset,
        },
    )
    cfg.rewards["leg_mirror"] = RewardTermCfg(
        func=leg_mirror,
        weight=-0.25,
        params={"command_name": "twist", "command_threshold": 0.03, "active_only": True, "asset_cfg": joint_asset},
    )
    cfg.rewards["diagonal_leg_symmetry"] = RewardTermCfg(
        func=diagonal_leg_symmetry,
        weight=-0.30,
        params={"command_name": "twist", "command_threshold": 0.03, "velocity_weight": 0.02, "asset_cfg": joint_asset},
    )
    cfg.rewards["diagonal_foot_symmetry"] = RewardTermCfg(
        func=diagonal_foot_symmetry,
        weight=-0.18,
        params={"command_name": "twist", "command_threshold": 0.03, "height_scale": 0.06, "velocity_scale": 0.35, "asset_cfg": foot_asset},
    )
    cfg.rewards["stance_width"] = RewardTermCfg(
        func=stance_width,
        weight=-4.0,
        params={"min_width": 0.17, "asset_cfg": foot_asset},
    )
    cfg.rewards["feet_contact"] = RewardTermCfg(func=contact_fraction_reward, weight=0.10, params={"sensor_name": "feet_ground_contact"})
    cfg.rewards["feet_air_time"] = RewardTermCfg(
        func=velocity_mdp.feet_air_time,
        weight=0.0,
        params={
            "sensor_name": "feet_ground_contact",
            "threshold_min": 0.08,
            "threshold_max": 0.45,
            "command_name": "twist",
            "command_threshold": 0.03,
        },
    )
    cfg.rewards["feet_gait_consistency"] = RewardTermCfg(
        func=feet_gait_consistency,
        weight=0.45,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.03,
            "duty_factor": 0.55,
            "transition_width": 0.08,
            "base_frequency": 1.8,
            "max_frequency": 2.8,
            "speed_scale": 1.0,
        },
    )
    cfg.rewards["foot_clearance"] = RewardTermCfg(
        func=foot_clearance,
        weight=-2.2,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "target_height": 0.055,
            "command_threshold": 0.03,
            "asset_cfg": foot_asset,
        },
    )
    cfg.rewards["foot_swing_height"] = RewardTermCfg(
        func=foot_swing_height,
        weight=-0.35,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "target_height": 0.065,
            "command_threshold": 0.03,
            "asset_cfg": foot_asset,
        },
    )
    cfg.rewards["foot_slip"] = RewardTermCfg(
        func=foot_slip,
        weight=-0.35,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist", "command_threshold": 0.03, "asset_cfg": foot_asset},
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=velocity_mdp.self_collision_cost,
        weight=-0.10,
        params={"sensor_name": "self_collision"},
    )
    cfg.rewards["thigh_collision"] = RewardTermCfg(
        func=velocity_mdp.self_collision_cost,
        weight=-0.20,
        params={"sensor_name": "thigh_ground_touch"},
    )
    cfg.rewards["shank_collision"] = RewardTermCfg(
        func=velocity_mdp.self_collision_cost,
        weight=-0.12,
        params={"sensor_name": "shank_ground_touch"},
    )
    cfg.rewards["trunk_collision"] = RewardTermCfg(
        func=velocity_mdp.self_collision_cost,
        weight=-0.30,
        params={"sensor_name": "trunk_ground_touch"},
    )
    cfg.rewards["terrain_level"] = RewardTermCfg(func=terrain_level_bonus, weight=0.15)
    cfg.terminations["bad_orientation"].params["limit_angle"] = math.radians(80.0)
    cfg.terminations["illegal_thigh_contact"] = TerminationTermCfg(
        func=velocity_mdp.illegal_contact,
        params={"sensor_name": "thigh_ground_touch"},
    )
    cfg.terminations["out_of_terrain_bounds"] = TerminationTermCfg(
        func=velocity_mdp.out_of_terrain_bounds,
        time_out=True,
    )
    cfg.events["push_robot"].interval_range_s = (6.0, 10.0)
    cfg.episode_length_s = 30.0
    cfg.sim = SimulationCfg(contact_sensor_maxmatch=128, mujoco=MujocoCfg(impratio=100, cone="elliptic", ccd_iterations=80))
    if play:
        cfg.scene.num_envs = 1
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_rows = 5
            cfg.scene.terrain.terrain_generator.num_cols = 5
    return cfg


def crawl_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = rough_env_cfg(play=play)
    cfg.scene.entities = {"robot": get_robot_crawl_cfg()}
    cfg.commands["twist"].ranges.lin_vel_x = (0.0, 0.25)
    cfg.commands["twist"].ranges.lin_vel_y = (-0.06, 0.06)
    cfg.commands["twist"].ranges.ang_vel_z = (-0.4, 0.4)
    cfg.commands["twist"].heading_command = False
    cfg.rewards["base_height"].params["target_height"] = CRAWL_HEIGHT
    cfg.rewards["base_height"].weight = -0.4
    cfg.rewards["joint_deviation"] = RewardTermCfg(func=joint_deviation_l2, weight=-0.4)
    cfg.terminations["bad_orientation"].params["limit_angle"] = math.radians(80.0)
    cfg.terminations.pop("base_ground_contact", None)
    return cfg
