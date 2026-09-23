from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "db-data" / "runs"


def curve(step: int, start: float, end: float, tau: float = 2500.0) -> float:
    ratio = 1.0 - math.exp(-step / tau)
    return start + (end - start) * ratio


def make_run(run_id: str, name: str, quality: float, lateral_focus: float) -> dict:
    metrics = []
    for idx in range(61):
        step = idx * 100
        noise = math.sin(idx * 0.7) * (1.0 - quality)
        reward = curve(step, 8, 95 * quality, 2100) + 7 * noise
        terrain = min(9.0, 1.0 + step / 850.0)
        vel_xy = max(0.18, curve(step, 2.0, 0.9 - 0.35 * lateral_focus, 2600) + 0.12 * noise)
        vel_yaw = max(0.12, curve(step, 2.5, 0.8 - 0.20 * quality, 2400) + 0.10 * math.sin(idx * 0.4))
        bad_orientation = max(0, 0.45 * (1 - quality) + 0.08 * math.sin(idx * 0.45))
        base_contact = max(0, 0.30 * (1 - quality) + 0.05 * math.cos(idx * 0.35))
        time_out = max(0, 1.0 - bad_orientation - base_contact - 0.02)
        nan_detection = 0.015 if idx in (18, 43) and quality < 0.75 else 0.0
        metrics.append(
            {
                "step": step,
                "mean_reward": round(reward, 3),
                "mean_episode_length": round(curve(step, 450, 3400 * quality + 400, 1200), 3),
                "policy_mean_std": round(curve(step, 0.9, 2.8 + 1.0 * (1 - quality), 3000), 3),
                "total_fps": round(curve(step, 12500, 16800, 1800) + 300 * math.sin(idx * 0.3), 3),
                "collection_time": round(curve(step, 3.5, 2.7, 2200) + 0.08 * math.sin(idx), 3),
                "learning_time": round(curve(step, 0.25, 0.16, 1900) + 0.015 * math.cos(idx * 0.4), 3),
                "wheel_roll_error_mean": round(max(0.8, curve(step, 5.5, 4.6 - 0.45 * quality, 2600) + 0.25 * math.sin(idx * 0.25)), 3),
                "error_vel_xy": round(vel_xy, 3),
                "error_vel_yaw": round(vel_yaw, 3),
                "loss_value": round(max(0.03, curve(step, 1.8, 0.08 + 0.08 * (1 - quality), 1300)), 3),
                "loss_surrogate": round(-0.03 + 0.015 * math.sin(idx * 0.5), 4),
                "loss_learning_rate": round(max(0.0001, 0.01 * math.exp(-step / 1800)), 6),
                "loss_entropy": round(curve(step, 20, 34 + 8 * (1 - quality), 3300), 3),
                "termination_time_out": round(time_out, 3),
                "termination_nan_detection": round(nan_detection, 3),
                "termination_bad_orientation": round(bad_orientation, 3),
                "termination_base_ground_contact": round(base_contact, 3),
                "reward_wheel_roll_tracking": round(curve(step, 0.05, 1.4 + 0.5 * quality, 1900), 3),
                "reward_track_lin_vel": round(curve(step, 0.1, 1.5 + lateral_focus * 0.6, 1700), 3),
                "reward_wheel_contact_bonus": round(curve(step, 0.4, 0.22 + 0.12 * quality, 2300), 3),
                "reward_leg_motion_penalty": round(-curve(step, 0.02, 0.25 - 0.08 * quality, 2300), 3),
                "reward_joint_torques": round(-curve(step, 0.03, 0.16 - 0.05 * quality, 2600), 3),
                "reward_action_rate": round(-curve(step, 0.1, 0.9 - 0.25 * quality, 2600), 3),
                "reward_joint_pos_limits": round(max(0, 0.04 * math.sin(idx * 0.35) - 0.03 * quality), 3),
                "curriculum_max": round(terrain, 3),
            }
        )
    return {
        "id": run_id,
        "name": name,
        "created_at": "2026-08-18 00:30",
        "status": "已完成",
        "config": {
            "算法": "PPO",
            "控制形式": "IK 名义步态 + 12 关节残差",
            "环境": "HTDW-4438 MuJoCo",
            "命令范围": "vx [-0.12,0.16], vy [-0.09,0.09], yaw [-0.8,0.8]",
        },
        "metrics": metrics,
    }


def main() -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    runs = [
        make_run("run_001", "第一次训练：基础奖励", quality=0.58, lateral_focus=0.15),
        make_run("run_002", "第二次训练：姿态惩罚增强", quality=0.76, lateral_focus=0.35),
        make_run("run_003", "第三次训练：横移跟踪优化", quality=0.88, lateral_focus=0.80),
    ]
    for run in runs:
        path = RUNS_ROOT / f"{run['id']}.json"
        path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
