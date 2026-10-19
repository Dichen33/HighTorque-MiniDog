from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import numpy as np

from policy import PolicyRunner
from scene import GAP_END_X, GAP_START_X, compose_scene
from navigation import NavigationController
from sim_config import SimConfig, load_sim_config


HERE = Path(__file__).resolve().parent
MUJOCO_ROOT = HERE.parent
ROBOT_XML = MUJOCO_ROOT / "htdw_4438" / "htdw_4438.xml"
DEFAULT_COURSE = Path("D:/mujoco_test/ROBOCON2027.xml")
DEFAULT_MINIDOG_CHECKPOINT = MUJOCO_ROOT / "mjlab" / "model" / "plum_v2" / "2026-10-11_14-50-33_plum_v2" / "model_1999.pt"
DEFAULT_CONFIG = HERE / "config.yaml"


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    qv = np.asarray((x, y, z), dtype=np.float32)
    t = 2.0 * np.cross(qv, v)
    return v - w * t + np.cross(qv, t)


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _is_descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    """Return whether a body belongs to an articulated body subtree."""
    current = body_id
    # Body 0 is MuJoCo's world body and its parent is itself.
    while current > 0:
        if current == ancestor_id:
            return True
        current = int(model.body_parentid[current])
    return False


class Command:
    def __init__(self, initial: tuple[float, float, float], limits: tuple[float, float, float]):
        self.value = np.asarray(initial, dtype=np.float32)
        self.limits = np.asarray(limits, dtype=np.float32)
        self.value[:] = np.clip(self.value, -self.limits, self.limits)

    def key(self, key) -> None:
        if isinstance(key, int):
            try:
                key = chr(key)
            except ValueError:
                return
        key = str(key).lower()
        if key in ("w", "s", "a", "d", "q", "e", "x"):
            self.value[:] = 0.0
        if key == "w":
            self.value[0] = 0.25
        elif key == "s":
            self.value[0] = -0.20
        elif key == "a":
            self.value[1] = 0.18
        elif key == "d":
            self.value[1] = -0.18
        elif key == "q":
            self.value[2] = 0.45
        elif key == "e":
            self.value[2] = -0.45


class RobotSim:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, runner: PolicyRunner, command: Command, config: SimConfig):
        self.m, self.d, self.runner, self.command = model, data, runner, command
        self.config = config
        self.joint_ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in config.joint_names], dtype=np.int32)
        self.qpos_ids = model.jnt_qposadr[self.joint_ids]
        self.qvel_ids = model.jnt_dofadr[self.joint_ids]
        self.position_joint_names = config.position_joint_names or config.joint_names
        self.velocity_joint_names = config.velocity_joint_names
        self.position_action_indices = np.asarray([config.joint_names.index(name) for name in self.position_joint_names], dtype=np.int32)
        self.velocity_action_indices = np.asarray([config.joint_names.index(name) for name in self.velocity_joint_names], dtype=np.int32)
        self.position_actuator_ids = np.asarray([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}{config.position_actuator_suffix}")
            for name in self.position_joint_names
        ], dtype=np.int32)
        self.velocity_actuator_ids = np.asarray([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}{config.velocity_actuator_suffix}")
            for name in self.velocity_joint_names
        ], dtype=np.int32)
        if np.any(self.joint_ids < 0) or np.any(self.position_actuator_ids < 0) or np.any(self.velocity_actuator_ids < 0):
            raise ValueError("Robot XML joint/actuator mapping does not match config.yaml")
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.base_body_name)
        if self.base_id < 0:
            raise ValueError(f"Robot XML does not contain the required base body '{config.base_body_name}'")
        # Training scans include geom group 0 (the terrain and course). A zero
        # mask means "include nothing" in mj_ray, which silently turns every
        # obstacle into a flat observation.
        self.map_group = np.asarray((1, 0, 0, 0, 0, 0), dtype=np.uint8)
        self.robot_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if _is_descendant(model, int(model.geom_bodyid[geom_id]), self.base_id)
            and int(model.geom_group[geom_id]) != 0
        }
        foot_body_ids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in config.foot_body_names
        }
        self.foot_geom_ids = {
            geom_id for geom_id in self.robot_geom_ids
            if int(model.geom_bodyid[geom_id]) in foot_body_ids
            or (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").endswith("_foot_collision")
        }
        self.plum_block_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith("minidog_plum_block_")
        }
        self.high_wall_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith("minidog_high_wall_")
        }
        self.low_bar_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith("minidog_low_bar_")
        }
        self.gap_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith("minidog_gap_")
        }
        self.plum_blocks_contacted: set[int] = set()
        self._plum_block_hold_steps: dict[int, int] = {}
        self.plum_blocks_stable: set[int] = set()
        self.wall_crossed = np.zeros(3, dtype=bool)
        self.gap_entered = False
        self.gap_crossed = False
        self._previous_x = 0.0
        self.floor_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if model.geom(geom_id).name in {"arena_floor", "floor", "ground"}
        }
        self.steps = 0
        self.collision_count = 0
        self.illegal_collision_seen = False
        self.support_contact = False
        self.stable_exit_steps = 0
        self._collision_pairs: set[tuple[int, int]] = set()
        self.fallen = False
        self.current_spawn = (0.0, 0.0, 0.0, 0.0)
        self._scan_cache: np.ndarray | None = None
        self._scan_age = 0
        self._rng = np.random.default_rng(config.random_seed)
        self._base_geom_friction = model.geom_friction.copy()
        self._base_dof_damping = model.dof_damping.copy()
        self._base_dof_frictionloss = model.dof_frictionloss.copy()
        self._base_actuator_gainprm = model.actuator_gainprm.copy()
        self._base_body_mass = model.body_mass.copy()
        self._base_body_ipos = model.body_ipos.copy()
        self._robot_body_ids = np.asarray(
            [body_id for body_id in range(1, model.nbody) if _is_descendant(model, body_id, self.base_id)],
            dtype=np.int32,
        )
        self._action_queue: list[np.ndarray] = []
        self._executed_action = np.zeros(runner.action_dim, dtype=np.float32)
        self._position_target = np.zeros(len(self.position_actuator_ids), dtype=np.float32)
        self._velocity_target = np.zeros(len(self.velocity_actuator_ids), dtype=np.float32)
        self._motor_response = 1.0
        self._action_delay_steps = 0
        self._control_steps = config.control_decimation
        self._elapsed_time = 0.0

    def _uniform(self, value_range: tuple[float, float]) -> float:
        return float(self._rng.uniform(value_range[0], value_range[1]))

    def _randomize_dynamics(self) -> None:
        """Restore XML parameters, then sample one physically consistent variant."""
        self.m.geom_friction[:] = self._base_geom_friction
        self.m.dof_damping[:] = self._base_dof_damping
        self.m.dof_frictionloss[:] = self._base_dof_frictionloss
        self.m.actuator_gainprm[:] = self._base_actuator_gainprm
        self.m.body_mass[:] = self._base_body_mass
        self.m.body_ipos[:] = self._base_body_ipos
        if not self.config.randomization_enabled:
            self._motor_response = 1.0
            self._action_delay_steps = 0
            self._control_steps = self.config.control_decimation
            return

        ground_scale = self._uniform(self.config.ground_friction_range)
        for geom_id in range(self.m.ngeom):
            if int(self.m.geom_group[geom_id]) == 0 and geom_id not in self.robot_geom_ids:
                self.m.geom_friction[geom_id] = self._base_geom_friction[geom_id] * ground_scale
        wheel_scale = self._uniform(self.config.wheel_friction_range)
        for geom_id in self.foot_geom_ids:
            self.m.geom_friction[geom_id] = self._base_geom_friction[geom_id] * wheel_scale

        damping_scale = self._uniform(self.config.joint_damping_scale_range)
        friction_scale = self._uniform(self.config.joint_friction_scale_range)
        self.m.dof_damping[self.qvel_ids] *= damping_scale
        self.m.dof_frictionloss[self.qvel_ids] *= friction_scale
        gain_scale = self._uniform(self.config.actuator_gain_scale_range)
        all_actuator_ids = np.concatenate((self.position_actuator_ids, self.velocity_actuator_ids))
        self.m.actuator_gainprm[all_actuator_ids] *= gain_scale

        mass_scale = self._uniform(self.config.mass_scale_range)
        self.m.body_mass[self._robot_body_ids] *= mass_scale
        if self.config.com_offset_range > 0.0:
            base_offset = self._rng.uniform(
                -self.config.com_offset_range, self.config.com_offset_range, size=2
            )
            self.m.body_ipos[self.base_id, :2] += base_offset
        self._motor_response = self._uniform(self.config.motor_response_range)
        low_delay, high_delay = self.config.action_delay_steps_range
        self._action_delay_steps = int(self._rng.integers(low_delay, high_delay + 1))
        jitter = self.config.control_decimation_jitter
        self._control_steps = max(
            1,
            self.config.control_decimation
            + (int(self._rng.integers(-jitter, jitter + 1)) if jitter else 0),
        )

    def reset(self, spawn: tuple[float, float, float, float]) -> None:
        mujoco.mj_resetData(self.m, self.d)
        self._randomize_dynamics()
        # Recompute MuJoCo's derived mass/inverse-inertia constants after
        # changing model arrays; otherwise mass randomization would only alter
        # the bookkeeping fields and not the dynamics used by the solver.
        mujoco.mj_setConst(self.m, self.d)
        x, y, z, yaw = spawn
        self.d.qpos[:7] = (x, y, z, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        self.d.qpos[self.qpos_ids] = np.asarray(self.config.default_angles, dtype=np.float32)
        self.d.qvel[:] = 0.0
        self.d.ctrl[self.position_actuator_ids] = np.asarray(self.config.default_angles, dtype=np.float32)[self.position_action_indices]
        self.d.ctrl[self.velocity_actuator_ids] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self.runner.reset()
        self.steps = 0
        self.collision_count = 0
        self._collision_pairs.clear()
        self.fallen = False
        self.current_spawn = tuple(float(value) for value in spawn)
        self.plum_blocks_contacted.clear()
        self._plum_block_hold_steps.clear()
        self.plum_blocks_stable.clear()
        self.wall_crossed[:] = False
        self.gap_entered = False
        self.gap_crossed = False
        self._previous_x = float(x)
        self._scan_cache = None
        self._scan_age = 0
        self._action_queue = [np.zeros(self.runner.action_dim, dtype=np.float32) for _ in range(self._action_delay_steps)]
        self._executed_action.fill(0.0)
        self._position_target[:] = np.asarray(self.config.default_angles, dtype=np.float32)[self.position_action_indices]
        self._velocity_target.fill(0.0)
        self._elapsed_time = 0.0

    def _height_scan(self) -> np.ndarray:
        width, depth = self.runner.scan_size
        # GridPatternCfg uses inclusive endpoints and a fixed 8 cm interval.
        resolution = self.config.scan_resolution
        xs = np.arange(-width / 2.0, width / 2.0 + 1e-6, resolution, dtype=np.float32)
        ys = np.arange(-depth / 2.0, depth / 2.0 + 1e-6, resolution, dtype=np.float32)
        points = np.asarray([(x, y) for x in xs for y in ys], dtype=np.float32)
        if len(points) != self.runner.scan_dim:
            raise ValueError(f"Scan grid {len(points)} points does not match checkpoint {self.runner.scan_dim}")
        yaw = yaw_from_quat(self.d.qpos[3:7])
        cy, sy = math.cos(yaw), math.sin(yaw)
        base = self.d.qpos[:3].copy()
        # A missed ray in MJLab is max_distance (3 m), which is clipped to 1.0
        # by the observation term. Keep the same value here instead of turning
        # unknown space into a false zero-height obstacle.
        nx, ny = len(xs), len(ys)
        stride = self.config.scan_stride
        sample_x = list(range(0, nx, stride))
        sample_y = list(range(0, ny, stride))
        if sample_x[-1] != nx - 1:
            sample_x.append(nx - 1)
        if sample_y[-1] != ny - 1:
            sample_y.append(ny - 1)

        sampled = np.ones((nx, ny), dtype=np.float32)
        for ix in sample_x:
            for iy in sample_y:
                x, y = float(xs[ix]), float(ys[iy])
                origin = base + np.asarray((cy * x - sy * y, sy * x + cy * y, 0.0), dtype=np.float64)
                geomid = np.asarray([-1], dtype=np.int32)
                distance = mujoco.mj_ray(
                    self.m,
                    self.d,
                    origin,
                    np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
                    self.map_group,
                    True,
                    self.base_id,
                    geomid,
                )
                if distance >= 0.0:
                    sampled[ix, iy] = np.clip(float(origin[2] - distance - base[2]), -1.0, 1.0)

        values = np.ones((nx, ny), dtype=np.float32)
        for ix in range(nx):
            nearest_x = min(sample_x, key=lambda value: abs(value - ix))
            for iy in range(ny):
                nearest_y = min(sample_y, key=lambda value: abs(value - iy))
                values[ix, iy] = sampled[nearest_x, nearest_y]
        values = values.reshape(-1)
        if self.config.randomization_enabled:
            if self.config.scan_noise_std > 0.0:
                values = values + self._rng.normal(0.0, self.config.scan_noise_std, size=values.shape).astype(np.float32)
            if self.config.scan_dropout_prob > 0.0:
                dropped = self._rng.random(values.shape) < self.config.scan_dropout_prob
                values[dropped] = 1.0
        values = np.clip(values, -1.0, 1.0)
        return values

    def _cached_height_scan(self) -> np.ndarray:
        if self._scan_cache is None or self._scan_age >= self.config.scan_update_interval:
            self._scan_cache = self._height_scan()
            self._scan_age = 0
        else:
            self._scan_age += 1
        return self._scan_cache

    def observation(self) -> np.ndarray:
        q = self.d.qpos[3:7]
        gravity = quat_rotate_inverse(q, np.asarray((0.0, 0.0, -1.0), dtype=np.float32))
        command = self.command.value.copy()
        command_norm = float(np.linalg.norm(command[:2]) + abs(command[2]))
        active = command_norm > 0.03
        frequency = min(2.8, max(1.8, 1.8 + command_norm)) if active else 0.0
        phase = (self._elapsed_time * frequency) % 1.0
        gait = np.asarray((math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase)), dtype=np.float32)
        default_angles = np.asarray(self.config.default_angles, dtype=np.float32)
        joint_pos = ((self.d.qpos[self.qpos_ids] - default_angles) * self.config.dof_pos_scale).astype(np.float32)
        joint_vel = (self.d.qvel[self.qvel_ids] * self.config.dof_vel_scale).astype(np.float32)
        scan = self._cached_height_scan()
        ang_vel = quat_rotate_inverse(q, self.d.qvel[3:6]).astype(np.float32) * self.config.ang_vel_scale
        if self.config.randomization_enabled:
            ang_vel += self._rng.normal(0.0, self.config.imu_noise_std, size=ang_vel.shape).astype(np.float32)
            gravity += self._rng.normal(0.0, self.config.imu_noise_std, size=gravity.shape).astype(np.float32)
            joint_pos += self._rng.normal(0.0, self.config.joint_pos_noise_std, size=joint_pos.shape).astype(np.float32)
            joint_vel += self._rng.normal(0.0, self.config.joint_vel_noise_std, size=joint_vel.shape).astype(np.float32)
        return np.concatenate((
            ang_vel,
            gravity,
            command * np.asarray(self.config.command_scale, dtype=np.float32),
            gait,
            joint_pos,
            joint_vel,
            self._executed_action,
            scan,
        ))

    def apply_policy(self) -> None:
        action = self.runner.step(self.observation())
        if self._action_delay_steps:
            self._action_queue.append(action.copy())
            delayed_action = self._action_queue.pop(0)
        else:
            delayed_action = action
        self._executed_action = delayed_action.copy()
        default_angles = np.asarray(self.config.default_angles, dtype=np.float32)
        position_scale = self.config.position_action_scale or self.config.action_scale
        velocity_scale = self.config.velocity_action_scale or self.config.action_scale
        position_target = default_angles[self.position_action_indices] + delayed_action[self.position_action_indices] * position_scale
        velocity_target = delayed_action[self.velocity_action_indices] * velocity_scale
        self._position_target += self._motor_response * (position_target - self._position_target)
        self._velocity_target += self._motor_response * (velocity_target - self._velocity_target)
        self.d.ctrl[self.position_actuator_ids] = self._position_target
        self.d.ctrl[self.velocity_actuator_ids] = self._velocity_target

    def step_control(self) -> None:
        self.apply_policy()
        for _ in range(self._control_steps):
            mujoco.mj_step(self.m, self.d)
        self.steps += 1
        self._elapsed_time += self._control_steps * self.m.opt.timestep
        self._update_metrics()

    def _update_metrics(self) -> None:
        tilt = float(np.linalg.norm(quat_rotate_inverse(self.d.qpos[3:7], np.asarray((0.0, 0.0, -1.0)))[:2]))
        self.fallen = bool(self.d.qpos[2] < 0.10 or tilt > 0.85)
        current_x = float(self.d.qpos[0])
        current_y = float(self.d.qpos[1])
        for index, wall_x in enumerate((-1.6, 0.0, 1.6)):
            if (
                not self.wall_crossed[index]
                and self._previous_x <= wall_x < current_x
                and abs(current_y - 0.85) <= 0.55
            ):
                self.wall_crossed[index] = True
        if not self.gap_entered and self._previous_x < GAP_START_X <= current_x:
            self.gap_entered = True
        if self.gap_entered and not self.gap_crossed and self._previous_x < GAP_END_X <= current_x:
            self.gap_crossed = True
        self._previous_x = current_x
        self.support_contact = False
        current_plum_contacts: set[int] = set()
        for i in range(self.d.ncon):
            contact = self.d.contact[i]
            a, b = int(contact.geom1), int(contact.geom2)
            if a in self.robot_geom_ids or b in self.robot_geom_ids:
                other = b if a in self.robot_geom_ids else a
                if int(self.m.geom_group[other]) == 0 and other not in self.floor_geom_ids:
                    pair = (min(a, b), max(a, b))
                    if pair not in self._collision_pairs:
                        self._collision_pairs.add(pair)
                        self.collision_count += 1
                foot_contact = a in self.foot_geom_ids or b in self.foot_geom_ids
                if foot_contact and int(self.m.geom_group[other]) == 0:
                    self.support_contact = True
                elif int(self.m.geom_group[other]) == 0 and other not in self.floor_geom_ids:
                    self.illegal_collision_seen = True
                # The added plum blocks are 0.15 m high. Counting only a foot
                # contact near the top surface prevents a leg-side collision
                # from being reported as a successful foothold.
                top_contact = float(contact.pos[2]) >= 0.13
                if other in self.plum_block_geom_ids and foot_contact and top_contact:
                    current_plum_contacts.add(other)
                    self._plum_block_hold_steps[other] = self._plum_block_hold_steps.get(other, 0) + 1
                    if self._plum_block_hold_steps[other] >= 2:
                        self.plum_blocks_stable.add(other)

        for geom_id in tuple(self._plum_block_hold_steps):
            if geom_id not in current_plum_contacts:
                self._plum_block_hold_steps[geom_id] = 0
        self.plum_blocks_contacted = set(self.plum_blocks_stable)
        stable_now = (
            not self.fallen
            and self.d.qpos[2] > 0.14
            and np.linalg.norm(
                quat_rotate_inverse(self.d.qpos[3:7], np.asarray((0.0, 0.0, -1.0)))[:2]
            ) < 0.70
            and self.d.qvel[0] > 0.02
            and abs(self.d.qvel[2]) < 0.75
            and self.support_contact
            and not self.illegal_collision_seen
        )
        lane = self.obstacle_kind()
        obstacle_exit_reached = (
            (lane == "plum" and len(self.plum_blocks_stable) == len(self.plum_block_geom_ids))
            or (lane == "high_wall" and bool(np.all(self.wall_crossed)))
            or (lane == "gap" and self.gap_crossed)
        )
        self.stable_exit_steps = self.stable_exit_steps + 1 if stable_now and obstacle_exit_reached else 0

    def obstacle_kind(self) -> str:
        """Infer the added validation lane from the active spawn position."""
        y = self.current_spawn[1]
        if abs(y - 3.35) < 0.25:
            return "plum"
        if abs(y - 0.85) < 0.25:
            return "high_wall"
        if abs(y + 0.85) < 0.25:
            return "low_bar"
        if abs(y + 3.35) < 0.25:
            return "gap"
        return "unknown"

    def evaluation(self, pass_distance: float, obstacle: str | None = None) -> dict[str, object]:
        kind = obstacle or self.obstacle_kind()
        distance = float(self.d.qpos[0] - self.current_spawn[0])
        distance_pass = (not self.fallen) and distance >= pass_distance
        obstacle_pass: bool | None
        if kind == "plum":
            obstacle_pass = (
                not self.fallen
                and len(self.plum_blocks_stable) == len(self.plum_block_geom_ids)
                and distance >= 4.5
                and self.stable_exit_steps >= 5
                and not self.illegal_collision_seen
            )
        elif kind == "high_wall":
            obstacle_pass = (
                bool(np.all(self.wall_crossed))
                and self.stable_exit_steps >= 5
                and not self.illegal_collision_seen
            )
        elif kind == "low_bar":
            # The current map has no dedicated low-bar completion geometry; do
            # not label distance through this lane as a successful validation.
            obstacle_pass = None
        elif kind == "gap":
            obstacle_pass = (
                self.gap_crossed
                and distance >= (GAP_END_X - self.current_spawn[0] + 0.8)
                and self.stable_exit_steps >= 5
                and not self.fallen
                and not self.illegal_collision_seen
            )
        else:
            obstacle_pass = None
        return {
            "kind": kind,
            "distance": distance,
            "distance_pass": distance_pass,
            "obstacle_pass": obstacle_pass,
            "plum_blocks_contacted": len(self.plum_blocks_contacted),
            "plum_blocks_total": len(self.plum_block_geom_ids),
            "wall_crossed": int(np.count_nonzero(self.wall_crossed)),
            "wall_total": len(self.wall_crossed),
            "gap_crossed": self.gap_crossed,
            "illegal_collision_seen": self.illegal_collision_seen,
            "stable_exit_steps": self.stable_exit_steps,
        }

    def status(self) -> str:
        return f"t={self._elapsed_time:5.2f}s x={self.d.qpos[0]:6.2f} y={self.d.qpos[1]:6.2f} z={self.d.qpos[2]:.3f} collisions={self.collision_count} fallen={self.fallen}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a native MJLab robot policy on the ROBOCON 2027 MuJoCo obstacle map.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="MuJoCo validation YAML")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Policy checkpoint; required for the default Go2W config")
    parser.add_argument("--course", type=Path, default=None, help="ROBOCON 2027 obstacle-course XML")
    parser.add_argument("--scene", type=Path, default=HERE / "generated" / "minidog_obstacle_course.xml")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--command", nargs=3, type=float, metavar=("VX", "VY", "YAW"))
    parser.add_argument("--spawn", nargs=4, type=float, metavar=("X", "Y", "Z", "YAW"))
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="运行时长（秒）；设置为 0 表示不限制时间",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--no-panel", action="store_true", help="Do not open the pygame button panel")
    parser.add_argument("--scan-size", nargs=2, type=float, metavar=("X", "Y"), help="Override height scan size in metres")
    parser.add_argument("--pass-distance", type=float, default=4.0)
    parser.add_argument(
        "--obstacle",
        choices=("auto", "plum", "gap", "high_wall", "low_bar", "unknown"),
        default="auto",
        help="Obstacle-specific evaluator; auto infers the added lane from --spawn.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_sim_config(args.config)
    duration = config.duration if args.duration is None else args.duration
    if duration < 0:
        raise ValueError("--duration must be non-negative; use 0 for unlimited")
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is false")
    course = (args.course or config.course_xml).expanduser().resolve()
    scene = compose_scene(
        config.robot_xml,
        course,
        args.scene.expanduser().resolve(),
        actuator_kp=config.actuator_kp,
        actuator_kv=config.actuator_kv,
        velocity_actuator_kv=config.velocity_actuator_kv,
        actuator_effort=config.actuator_effort,
        position_joints=config.position_joint_names or config.joint_names,
        velocity_joints=config.velocity_joint_names,
    )
    print(f"[Scene] {scene}")
    if args.build_only:
        model = mujoco.MjModel.from_xml_path(str(scene))
        print(f"[Scene] nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} ngeom={model.ngeom}")
        return

    if args.checkpoint is None:
        if config.base_body_name == "base_link":
            checkpoint = DEFAULT_MINIDOG_CHECKPOINT.expanduser().resolve()
        else:
            raise ValueError("Go2W sim2sim requires --checkpoint path/to/model_<step>.pt")
    else:
        checkpoint = args.checkpoint.expanduser().resolve()

    runner = PolicyRunner(checkpoint, config, args.device, tuple(args.scan_size) if args.scan_size else None)
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = config.physics_dt
    data = mujoco.MjData(model)
    command = Command(
        tuple(args.command) if args.command else config.command,
        config.max_command,
    )
    navigation = NavigationController()
    sim = RobotSim(model, data, runner, command, config)
    spawn = tuple(args.spawn) if args.spawn else config.spawn
    sim.reset(spawn)
    steps = None if duration == 0 else int(round(duration / (model.opt.timestep * config.control_decimation)))
    step_text = "unlimited" if steps is None else str(steps)
    print(f"[Run] control_dt={model.opt.timestep * config.control_decimation:.3f}s steps={step_text} command={command.value.tolist()} spawn={spawn}")

    def key_callback(key):
        command.key(key)
        if str(key).lower() in ("r", "reset"):
            sim.reset(spawn)

    panel = None
    if not args.headless and not args.no_panel:
        try:
            from ui import ControlPanel
            panel = ControlPanel(sim, command, spawn, navigation=navigation, scene_path=scene)
        except Exception as exc:
            print(f"[UI] pygame panel unavailable: {exc}")
            print("[UI] Continue with MuJoCo viewer only; install pygame to enable buttons.")

    if args.headless:
        step = 0
        while steps is None or step < steps:
            sim.step_control()
            if step % 25 == 0:
                print("[Run]", sim.status())
            if sim.fallen:
                break
            step += 1
    else:
        from mujoco import viewer as mujoco_viewer
        with mujoco_viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            viewer.cam.distance = 4.5
            viewer.cam.elevation = -25
            viewer.cam.azimuth = 90
            viewer.cam.lookat[:] = data.qpos[:3]
            deadline = time.perf_counter() + duration if duration > 0 else None
            while viewer.is_running() and (deadline is None or time.perf_counter() < deadline):
                if panel is not None:
                    panel.poll()
                    if panel.closed:
                        viewer.close()
                        break
                    command.value[:] = panel.navigation_command()
                started = time.perf_counter()
                if panel is None or not panel.paused:
                    sim.step_control()
                viewer.cam.lookat[:] = data.qpos[:3]
                viewer.sync()
                if panel is not None:
                    panel.draw()
                rate = panel.sim_speed if panel is not None else 1.0
                remaining = model.opt.timestep * config.control_decimation / rate - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
                if sim.fallen:
                    print("[Run] FALL detected; close viewer or press R")
                    break
    if panel is not None:
        panel.close()
    result = sim.evaluation(
        args.pass_distance,
        None if args.obstacle == "auto" else args.obstacle,
    )
    obstacle_pass = result["obstacle_pass"]
    verdict = "PASS" if obstacle_pass is True else "FAIL" if obstacle_pass is False else "UNVERIFIED"
    print(
        f"[Result] {verdict} obstacle={result['kind']} "
        f"distance_x={result['distance']:.3f} distance_pass={result['distance_pass']} "
        f"plum_blocks={result['plum_blocks_contacted']}/{result['plum_blocks_total']} "
        f"walls={result['wall_crossed']}/{result['wall_total']} gap_crossed={result['gap_crossed']} "
        f"collisions={sim.collision_count} illegal={result['illegal_collision_seen']} "
        f"stable_steps={result['stable_exit_steps']} fallen={sim.fallen} "
        f"time={sim._elapsed_time:.2f}s"
    )


if __name__ == "__main__":
    main()
