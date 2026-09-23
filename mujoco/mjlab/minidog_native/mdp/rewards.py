from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _command_activity(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    command = env.command_manager.get_command(command_name)
    command_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    return command_norm, command_norm > command_threshold


def _contact_flags(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    found = (sensor.data.found > 0).float()
    while found.ndim > 2:
        found = torch.mean(found, dim=-1)
    if found.ndim == 1:
        found = found.unsqueeze(-1)
    return found


def _foot_body_state(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> tuple[Entity, torch.Tensor, torch.Tensor]:
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    foot_vel = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]
    return asset, foot_pos, foot_vel


def _foot_pos_b(asset: Entity, foot_pos_w: torch.Tensor) -> torch.Tensor:
    rel_w = foot_pos_w - asset.data.root_link_pos_w[:, None, :]
    root_quat = asset.data.root_link_quat_w[:, None, :].expand(-1, rel_w.shape[1], -1)
    return quat_apply_inverse(root_quat, rel_w)


def _selected_joint_names(asset: Entity, asset_cfg: SceneEntityCfg) -> list[str]:
    if isinstance(asset_cfg.joint_ids, slice):
        return list(asset.joint_names[asset_cfg.joint_ids])
    return [asset.joint_names[i] for i in asset_cfg.joint_ids]


def _gait_phase(
    env: ManagerBasedRlEnv,
    command_norm: torch.Tensor,
    active: torch.Tensor,
    base_frequency: float,
    max_frequency: float,
    speed_scale: float,
) -> torch.Tensor:
    moving_frequency = torch.clamp(
        base_frequency + speed_scale * command_norm,
        min=base_frequency,
        max=max_frequency,
    )
    frequency = torch.where(active, moving_frequency, torch.zeros_like(command_norm))
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=command_norm.dtype)
    return (env.episode_length_buf.float() * env.step_dt * frequency + env_ids * 0.17320508) % 1.0


def track_linear_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.sum(torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-error / std**2)


def track_yaw_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
    return torch.exp(-error / std**2)


def base_height_l2(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    origins_z = getattr(env.scene, "env_origins", None)
    if origins_z is None:
        ground_z = torch.zeros(env.num_envs, device=env.device)
    else:
        ground_z = env.scene.env_origins[:, 2]
    return torch.square(asset.data.root_link_pos_w[:, 2] - ground_z - target_height)


def flat_orientation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def lin_vel_z_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def safe_base_lin_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
    from mjlab.envs.mdp.observations import base_lin_vel

    value = base_lin_vel(env)
    return torch.nan_to_num(value, nan=0.0, posinf=100.0, neginf=-100.0)


def safe_height_scan(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    from mjlab.envs.mdp.observations import height_scan

    value = height_scan(env, sensor_name)
    return torch.nan_to_num(value, nan=0.0, posinf=3.0, neginf=-3.0)


def joint_deviation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg(
            "robot",
            joint_names=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"),
        )
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.mean(torch.square(q - q0), dim=1)


def stand_pose(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.03,
    hip_weight: float = 1.2,
    thigh_weight: float = 1.6,
    calf_weight: float = 1.6,
    velocity_weight: float = 0.03,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg(
            "robot",
            joint_names=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"),
        )
    _, active = _command_activity(env, command_name, command_threshold)
    asset: Entity = env.scene[asset_cfg.name]
    joint_names = _selected_joint_names(asset, asset_cfg)
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    weights = []
    for name in joint_names:
        if "_hip_joint" in name:
            weights.append(hip_weight)
        elif "_thigh_joint" in name:
            weights.append(thigh_weight)
        elif "_calf_joint" in name:
            weights.append(calf_weight)
        else:
            weights.append(1.0)
    weight = torch.tensor(weights, device=env.device, dtype=q.dtype).unsqueeze(0)
    pose_cost = torch.mean(torch.square(q - q0) * weight, dim=1)
    vel_cost = torch.mean(torch.square(qd), dim=1)
    return (pose_cost + velocity_weight * vel_cost) * (~active).float()


def stand_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.04,
) -> torch.Tensor:
    contact = _contact_flags(env, sensor_name)
    _, active = _command_activity(env, command_name, command_threshold)
    if contact.shape[1] < 4:
        return torch.zeros(env.num_envs, device=env.device)
    missing_contact = 1.0 - torch.clamp(contact[:, :4], 0.0, 1.0)
    return torch.mean(missing_contact, dim=1) * (~active).float()


def stand_foot_geometry(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.04,
    min_width: float = 0.18,
    min_length: float = 0.18,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", body_names=("FL_foot", "FR_foot", "RL_foot", "RR_foot"), preserve_order=True)
    _, active = _command_activity(env, command_name, command_threshold)
    asset, foot_pos_w, _ = _foot_body_state(env, asset_cfg)
    if foot_pos_w.shape[1] < 4:
        return torch.zeros(env.num_envs, device=env.device)
    foot_pos = _foot_pos_b(asset, foot_pos_w)

    front_width = torch.abs(foot_pos[:, 0, 1] - foot_pos[:, 1, 1])
    rear_width = torch.abs(foot_pos[:, 2, 1] - foot_pos[:, 3, 1])
    left_length = torch.abs(foot_pos[:, 0, 0] - foot_pos[:, 2, 0])
    right_length = torch.abs(foot_pos[:, 1, 0] - foot_pos[:, 3, 0])
    width_cost = 0.5 * (
        torch.square(torch.clamp(min_width - front_width, min=0.0) / max(min_width, 1.0e-6))
        + torch.square(torch.clamp(min_width - rear_width, min=0.0) / max(min_width, 1.0e-6))
    )
    length_cost = 0.5 * (
        torch.square(torch.clamp(min_length - left_length, min=0.0) / max(min_length, 1.0e-6))
        + torch.square(torch.clamp(min_length - right_length, min=0.0) / max(min_length, 1.0e-6))
    )
    pair_symmetry = (
        torch.square((foot_pos[:, 0, 0] - foot_pos[:, 1, 0]) / max(min_length, 1.0e-6))
        + torch.square((foot_pos[:, 2, 0] - foot_pos[:, 3, 0]) / max(min_length, 1.0e-6))
        + torch.square((foot_pos[:, 0, 1] - foot_pos[:, 2, 1]) / max(min_width, 1.0e-6))
        + torch.square((foot_pos[:, 1, 1] - foot_pos[:, 3, 1]) / max(min_width, 1.0e-6))
    )
    return (width_cost + 0.5 * length_cost + 0.15 * pair_symmetry) * (~active).float()


def stand_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.03,
    lin_vel_threshold: float = 0.04,
    yaw_vel_threshold: float = 0.06,
    height_tol: float = 0.02,
    tilt_tol: float = 0.12,
    target_height: float = 0.235,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg(
            "robot",
            joint_names=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"),
        )
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    inactive = (command_norm < command_threshold).float()
    joint_cost = joint_deviation_l2(env, asset_cfg)
    base_lin_vel = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
    yaw_vel = torch.abs(asset.data.root_link_ang_vel_b[:, 2])
    tilt_cost = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)

    if hasattr(env.scene, "env_origins") and env.scene.env_origins is not None:
        ground_z = env.scene.env_origins[:, 2]
    else:
        ground_z = torch.zeros(env.num_envs, device=env.device)
    height_error = torch.abs(asset.data.root_link_pos_w[:, 2] - ground_z - target_height)

    stability_cost = (
        joint_cost
        + torch.relu(base_lin_vel - lin_vel_threshold)
        + 0.5 * torch.relu(yaw_vel - yaw_vel_threshold)
        + 2.0 * torch.relu(height_error - height_tol)
        + 0.8 * torch.relu(tilt_cost - tilt_tol)
    )
    stability_cost *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return stability_cost * inactive


def contact_fraction_reward(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    found = (sensor.data.found > 0).float()
    while found.ndim > 2:
        found = torch.mean(found, dim=-1)
    return torch.mean(found, dim=1)


def contact_fraction_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    return contact_fraction_reward(env, sensor_name).unsqueeze(-1)


def feet_contact_target_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.03,
    target_contact: float = 4.0,
    sigma: float = 0.75,
) -> torch.Tensor:
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    found = (sensor.data.found > 0).float()
    while found.ndim > 2:
        found = torch.mean(found, dim=-1)
    if found.ndim == 1:
        found = found.unsqueeze(-1)
    contact_count = torch.sum(found, dim=1)

    command = env.command_manager.get_command(command_name)
    command_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    inactive = (command_norm < command_threshold).float()

    reward = torch.exp(-torch.square(contact_count - target_contact) / max(sigma, 1.0e-6) ** 2)
    return reward * inactive


def feet_gait_consistency(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.03,
    duty_factor: float = 0.55,
    transition_width: float = 0.08,
    base_frequency: float = 1.8,
    max_frequency: float = 2.8,
    speed_scale: float = 1.0,
) -> torch.Tensor:
    contact = _contact_flags(env, sensor_name)
    if contact.shape[1] < 4:
        return torch.zeros(env.num_envs, device=env.device)

    command_norm, active = _command_activity(env, command_name, command_threshold)
    phase = _gait_phase(env, command_norm, active, base_frequency, max_frequency, speed_scale)
    group_a_phase = phase
    group_b_phase = (phase + 0.5) % 1.0

    def stance_gate(local_phase: torch.Tensor) -> torch.Tensor:
        distance_to_edge = torch.minimum(local_phase, torch.abs(local_phase - duty_factor))
        raw = torch.clamp(distance_to_edge / max(transition_width, 1.0e-6), 0.0, 1.0)
        return torch.where(local_phase < duty_factor, raw, torch.zeros_like(raw))

    stance_a = stance_gate(group_a_phase)
    stance_b = stance_gate(group_b_phase)
    expected = torch.stack((stance_a, stance_b, stance_b, stance_a), dim=1)
    reward = 1.0 - torch.mean(torch.abs(contact[:, :4] - expected), dim=1)
    return torch.clamp(reward, min=0.0) * active.float()


def foot_clearance(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    target_height: float = 0.055,
    command_threshold: float = 0.03,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", body_names=("FL_foot", "FR_foot", "RL_foot", "RR_foot"), preserve_order=True)
    _, active = _command_activity(env, command_name, command_threshold)
    contact = _contact_flags(env, sensor_name)
    _, foot_pos, foot_vel = _foot_body_state(env, asset_cfg)
    if hasattr(env.scene, "env_origins") and env.scene.env_origins is not None:
        ground_z = env.scene.env_origins[:, 2].unsqueeze(1)
    else:
        ground_z = torch.zeros(env.num_envs, 1, device=env.device)
    clearance = foot_pos[:, :, 2] - ground_z
    swing = 1.0 - contact[:, : clearance.shape[1]]
    moving_foot = torch.norm(foot_vel[:, :, :2], dim=2)
    error = torch.square(torch.clamp(target_height - clearance, min=0.0))
    return torch.mean(error * swing * moving_foot, dim=1) * active.float()


class foot_swing_height:
    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset_cfg: SceneEntityCfg = cfg.params.get(
            "asset_cfg",
            SceneEntityCfg("robot", body_names=("FL_foot", "FR_foot", "RL_foot", "RR_foot"), preserve_order=True),
        )
        asset: Entity = env.scene[asset_cfg.name]
        body_ids = asset_cfg.body_ids
        if isinstance(body_ids, slice):
            num_feet = len(asset.body_names[body_ids])
        else:
            num_feet = len(body_ids)
        self.peak_heights = torch.zeros((env.num_envs, num_feet), device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        command_name: str,
        target_height: float = 0.065,
        command_threshold: float = 0.03,
        asset_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        from mjlab.sensor import ContactSensor

        if asset_cfg is None:
            asset_cfg = SceneEntityCfg("robot", body_names=("FL_foot", "FR_foot", "RL_foot", "RR_foot"), preserve_order=True)
        command_norm, active = _command_activity(env, command_name, command_threshold)
        del command_norm

        sensor: ContactSensor = env.scene[sensor_name]
        asset, foot_pos, _ = _foot_body_state(env, asset_cfg)
        del asset

        if hasattr(env.scene, "env_origins") and env.scene.env_origins is not None:
            ground_z = env.scene.env_origins[:, 2].unsqueeze(1)
        else:
            ground_z = torch.zeros(env.num_envs, 1, device=env.device)
        clearance = foot_pos[:, :, 2] - ground_z

        contact = _contact_flags(env, sensor_name)[:, : clearance.shape[1]]
        in_air = contact <= 0.0
        self.peak_heights = torch.where(in_air, torch.maximum(self.peak_heights, clearance), self.peak_heights)

        first_contact = sensor.compute_first_contact(dt=env.step_dt)[:, : clearance.shape[1]]
        normalized_error = self.peak_heights / max(target_height, 1.0e-6) - 1.0
        cost = torch.mean(torch.square(normalized_error) * first_contact.float(), dim=1) * active.float()

        self.peak_heights = torch.where(first_contact, torch.zeros_like(self.peak_heights), self.peak_heights)
        return cost

    def reset(self, env_ids: torch.Tensor) -> None:
        self.peak_heights[env_ids] = 0.0


def foot_slip(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.03,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", body_names=("FL_foot", "FR_foot", "RL_foot", "RR_foot"), preserve_order=True)
    _, active = _command_activity(env, command_name, command_threshold)
    contact = _contact_flags(env, sensor_name)
    _, _, foot_vel = _foot_body_state(env, asset_cfg)
    contact = contact[:, : foot_vel.shape[1]]
    slip_speed = torch.norm(foot_vel[:, :, :2], dim=2)
    return torch.mean(slip_speed * contact, dim=1) * active.float()


def leg_mirror(
    env: ManagerBasedRlEnv,
    command_name: str | None = None,
    command_threshold: float = 0.03,
    active_only: bool = False,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg(
            "robot",
            joint_names=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"),
            preserve_order=True,
        )
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_names = _selected_joint_names(asset, asset_cfg)
    if not joint_names:
        return torch.zeros(env.num_envs, device=env.device)
    name_to_index = {name: i for i, name in enumerate(joint_names)}

    pairs = (
        ("FL_hip_joint", "FR_hip_joint", -1.0),
        ("RL_hip_joint", "RR_hip_joint", -1.0),
        ("FL_thigh_joint", "FR_thigh_joint", 1.0),
        ("RL_thigh_joint", "RR_thigh_joint", 1.0),
        ("FL_calf_joint", "FR_calf_joint", 1.0),
        ("RL_calf_joint", "RR_calf_joint", 1.0),
        ("FL_hip_joint", "RL_hip_joint", 1.0),
        ("FR_hip_joint", "RR_hip_joint", 1.0),
    )
    costs = []
    for left, right, sign in pairs:
        if left in name_to_index and right in name_to_index:
            costs.append(torch.square(q[:, name_to_index[left]] - sign * q[:, name_to_index[right]]))
    if not costs:
        return torch.zeros(env.num_envs, device=env.device)
    reward = torch.mean(torch.stack(costs, dim=1), dim=1)
    if active_only and command_name is not None:
        _, active = _command_activity(env, command_name, command_threshold)
        reward = reward * active.float()
    return reward


def diagonal_leg_symmetry(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.03,
    velocity_weight: float = 0.02,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg(
            "robot",
            joint_names=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"),
            preserve_order=True,
        )
    _, active = _command_activity(env, command_name, command_threshold)
    asset: Entity = env.scene[asset_cfg.name]
    joint_names = _selected_joint_names(asset, asset_cfg)
    if not joint_names:
        return torch.zeros(env.num_envs, device=env.device)
    name_to_index = {name: i for i, name in enumerate(joint_names)}
    q = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]

    pairs = (
        ("FL_hip_joint", "RR_hip_joint", -1.0),
        ("FR_hip_joint", "RL_hip_joint", -1.0),
        ("FL_thigh_joint", "RR_thigh_joint", 1.0),
        ("FR_thigh_joint", "RL_thigh_joint", 1.0),
        ("FL_calf_joint", "RR_calf_joint", 1.0),
        ("FR_calf_joint", "RL_calf_joint", 1.0),
    )
    costs = []
    for a_name, b_name, sign in pairs:
        if a_name in name_to_index and b_name in name_to_index:
            a_idx = name_to_index[a_name]
            b_idx = name_to_index[b_name]
            pos_cost = torch.square(q[:, a_idx] - sign * q[:, b_idx])
            vel_cost = torch.square(qd[:, a_idx] - sign * qd[:, b_idx])
            costs.append(pos_cost + velocity_weight * vel_cost)
    if not costs:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.mean(torch.stack(costs, dim=1), dim=1) * active.float()


def diagonal_foot_symmetry(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.03,
    height_scale: float = 0.06,
    velocity_scale: float = 0.35,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", body_names=("FL_foot", "FR_foot", "RL_foot", "RR_foot"), preserve_order=True)
    _, active = _command_activity(env, command_name, command_threshold)
    _, foot_pos, foot_vel = _foot_body_state(env, asset_cfg)
    if foot_pos.shape[1] < 4:
        return torch.zeros(env.num_envs, device=env.device)
    vel_xy = torch.norm(foot_vel[:, :, :2], dim=2)
    height_cost = (
        torch.square((foot_pos[:, 0, 2] - foot_pos[:, 3, 2]) / max(height_scale, 1.0e-6))
        + torch.square((foot_pos[:, 1, 2] - foot_pos[:, 2, 2]) / max(height_scale, 1.0e-6))
    )
    velocity_cost = (
        torch.square((vel_xy[:, 0] - vel_xy[:, 3]) / max(velocity_scale, 1.0e-6))
        + torch.square((vel_xy[:, 1] - vel_xy[:, 2]) / max(velocity_scale, 1.0e-6))
    )
    return 0.5 * (height_cost + 0.25 * velocity_cost) * active.float()


def stance_width(
    env: ManagerBasedRlEnv,
    min_width: float = 0.13,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", body_names=("FL_foot", "FR_foot", "RL_foot", "RR_foot"), preserve_order=True)
    _, foot_pos, _ = _foot_body_state(env, asset_cfg)
    if foot_pos.shape[1] < 4:
        return torch.zeros(env.num_envs, device=env.device)
    front_width = torch.abs(foot_pos[:, 0, 1] - foot_pos[:, 1, 1])
    rear_width = torch.abs(foot_pos[:, 2, 1] - foot_pos[:, 3, 1])
    return 0.5 * (torch.square(torch.clamp(min_width - front_width, min=0.0)) + torch.square(torch.clamp(min_width - rear_width, min=0.0)))


def terrain_level_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
    terrain = getattr(env.scene, "terrain", None)
    levels = getattr(terrain, "terrain_levels", None)
    if levels is None:
        return torch.zeros(env.num_envs, device=env.device)
    max_level = torch.clamp(torch.max(levels.float()), min=1.0)
    return levels.float() / max_level
