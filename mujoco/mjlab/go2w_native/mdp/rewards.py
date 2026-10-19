from __future__ import annotations

import torch

from minidog_native.mdp.rewards import *  # noqa: F401,F403
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp


OBSTACLE_TERRAIN_NAMES = (
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "stairs_up",
    "stairs_down",
    "random_grid",
    "blocks",
    "sloped_terrain",
    "slope",
    "flat_gap",
    "rough_plum_blossom",
    "flat_high_wall",
)


def wheel_mode_leg_activity_l2(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.03,
    position_weight: float = 0.25,
    velocity_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize unnecessary leg motion outside generated obstacle terrain."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    active = (
        torch.linalg.vector_norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    ) > command_threshold
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    cost = position_weight * torch.mean(torch.square(q - q0), dim=1)
    cost = cost + velocity_weight * torch.mean(torch.square(qd), dim=1)

    terrain = getattr(env.scene, "terrain", None)
    obstacle = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if terrain is not None and terrain.terrain_types is not None:
        names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        for name in OBSTACLE_TERRAIN_NAMES:
            if name in names:
                obstacle |= terrain.terrain_types == names.index(name)
    return cost * active.float() * (~obstacle).float()


def wheel_velocity_tracking(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float = 0.10,
    command_threshold: float = 0.03,
    std: float = 1.5,
    obstacle_scale: float = 0.25,
) -> torch.Tensor:
    """Track the commanded forward speed with wheel rotation.

    The absolute value is used because left/right wheel encoder signs depend
    on the wheel mesh/joint axis convention. Direction and yaw are still
    constrained by the base velocity tracking term.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    target = torch.abs(command[:, 0]).unsqueeze(1) / max(wheel_radius, 1.0e-3)
    error = torch.mean(torch.square(torch.abs(wheel_vel) - target), dim=1)
    active = (torch.linalg.vector_norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) > command_threshold
    terrain = getattr(env.scene, "terrain", None)
    obstacle = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if terrain is not None and terrain.terrain_types is not None:
        names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        for name in OBSTACLE_TERRAIN_NAMES:
            if name in names:
                obstacle |= terrain.terrain_types == names.index(name)
    obstacle_scale = min(max(float(obstacle_scale), 0.0), 1.0)
    mode_scale = torch.where(obstacle, torch.full_like(error, obstacle_scale), torch.ones_like(error))
    return torch.exp(-error / max(std, 1.0e-3) ** 2) * active.float() * mode_scale


def obstacle_feet_air_time(
    env,
    sensor_name: str,
    threshold_min: float = 0.08,
    threshold_max: float = 0.45,
    command_name: str | None = None,
    command_threshold: float = 0.03,
    obstacle_difficulty_threshold: float = 0.10,
) -> torch.Tensor:
    """Enable swing reward only on a terrain row that contains an obstacle.

    Wheels remain the default locomotion mode.  At low rows a generated
    obstacle may still be effectively flat (for example a zero-height stair),
    so the reward is additionally gated by the current normalized terrain
    difficulty.  The contact sensor still supplies the actual air-time term;
    this function only decides when it is relevant.
    """
    reward = velocity_mdp.feet_air_time(
        env,
        sensor_name=sensor_name,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        command_name=command_name,
        command_threshold=command_threshold,
    )
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or terrain.terrain_types is None or terrain.terrain_origins is None:
        return torch.zeros_like(reward)

    names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    is_obstacle = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in OBSTACLE_TERRAIN_NAMES:
        if name in names:
            is_obstacle |= terrain.terrain_types == names.index(name)
    level_count = max(int(terrain.terrain_origins.shape[0]) - 1, 1)
    difficulty = terrain.terrain_levels.float() / float(level_count)
    active = is_obstacle & (difficulty >= obstacle_difficulty_threshold)
    return reward * active.float()


def gap_takeoff_progress(
    env,
    command_name: str,
    # FlatGapTerrainCfg spawns at local x=1.20; reward coordinates are
    # relative to that origin rather than absolute terrain coordinates.
    gap_start_x: float = 1.15,
    approach_width: float = 0.45,
    command_threshold: float = 0.05,
    vertical_speed_scale: float = 0.45,
) -> torch.Tensor:
    """Shape a controlled takeoff immediately before a ditch."""
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or terrain.terrain_types is None:
        return torch.zeros(env.num_envs, device=env.device)
    names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    if "flat_gap" not in names:
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene["robot"]
    displacement = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    command = env.command_manager.get_command(command_name)
    active = command[:, 0] > command_threshold
    approach = torch.exp(-torch.square(displacement - (gap_start_x - 0.12)) / max(approach_width, 1.0e-3) ** 2)
    upward_speed = torch.clamp(asset.data.root_link_lin_vel_w[:, 2], min=0.0)
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)
    return (terrain.terrain_types == names.index("flat_gap")).float() * active.float() * approach * upright * torch.clamp(
        upward_speed / max(vertical_speed_scale, 1.0e-3), 0.0, 1.0
    )


def gap_landing_stability(
    env,
    command_name: str,
    # The widest generated gap ends near local x=3.25, or 2.05 m relative to
    # the spawn origin. Keep a small landing buffer after the exit edge.
    gap_end_x: float = 2.20,
    landing_width: float = 0.55,
    command_threshold: float = 0.05,
    vertical_speed_scale: float = 0.45,
) -> torch.Tensor:
    """Reward a quiet, upright landing after the ditch exit."""
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or terrain.terrain_types is None:
        return torch.zeros(env.num_envs, device=env.device)
    names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    if "flat_gap" not in names:
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene["robot"]
    displacement = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    command = env.command_manager.get_command(command_name)
    active = command[:, 0] > command_threshold
    landing = torch.sigmoid((displacement - gap_end_x) / max(landing_width, 1.0e-3))
    vertical_stability = torch.exp(-torch.square(asset.data.root_link_lin_vel_w[:, 2]) / max(vertical_speed_scale, 1.0e-3) ** 2)
    forward = torch.clamp(asset.data.root_link_lin_vel_b[:, 0] / torch.clamp(command[:, 0], min=0.08), 0.0, 1.0)
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)
    return (terrain.terrain_types == names.index("flat_gap")).float() * active.float() * landing * vertical_stability * forward * upright


def high_wall_clearance_progress(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.05,
    # FlatHighWallTerrainCfg spawns at local x=1.5. These positions are the
    # three wall centers relative to that spawn origin.
    wall_centers_x: tuple[float, ...] = (1.40, 2.95, 4.50),
    foot_clearance: float = 0.13,
    clearance_range: tuple[float, float] = (0.13, 0.42),
) -> torch.Tensor:
    """Encourage leg clearance and forward motion at each wall front."""
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or terrain.terrain_types is None:
        return torch.zeros(env.num_envs, device=env.device)
    names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    if "flat_high_wall" not in names:
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    displacement = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    command = env.command_manager.get_command(command_name)
    active = command[:, 0] > command_threshold
    nearest = torch.stack([torch.abs(displacement - center) for center in wall_centers_x], dim=1).amin(dim=1)
    near_wall = torch.exp(-torch.square(nearest) / 0.18**2)
    foot_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, None, 2]
    level_count = max(int(terrain.terrain_origins.shape[0]) - 1, 1)
    difficulty = terrain.terrain_levels.float() / float(level_count)
    required_clearance = clearance_range[0] + difficulty * (
        clearance_range[1] - clearance_range[0]
    )
    clearance = torch.sigmoid((torch.amax(foot_z, dim=1) - required_clearance) / 0.025)
    forward = torch.clamp(asset.data.root_link_lin_vel_b[:, 0] / torch.clamp(command[:, 0], min=0.08), 0.0, 1.0)
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)
    return (terrain.terrain_types == names.index("flat_high_wall")).float() * active.float() * near_wall * clearance * forward * upright

