from __future__ import annotations

import math
import random
import time

from zhdb_client import ZhDBRun


def main() -> None:
    run = ZhDBRun(
        "live_demo",
        name="实时演示：像 W&B 一样刷新曲线",
        config={"算法": "PPO", "用途": "演示实时曲线写入"},
        silent=False,
    )
    for i in range(120):
        step = i * 100
        progress = 1 - math.exp(-i / 28)
        run.log(
            {
                "mean_reward": 12 + 75 * progress + random.uniform(-3, 3),
                "mean_episode_length": 500 + 2900 * progress,
                "error_vel_xy": max(0.2, 1.8 - 1.1 * progress + random.uniform(-0.05, 0.05)),
                "error_vel_yaw": max(0.15, 1.6 - 0.9 * progress + random.uniform(-0.04, 0.04)),
                "loss_value": max(0.04, 1.2 * math.exp(-i / 24)),
                "loss_entropy": 24 + 10 * progress,
                "reward_track_lin_vel": 0.2 + 1.8 * progress,
                "reward_action_rate": -(0.15 + 0.45 * progress),
                "termination_time_out": min(0.96, 0.25 + 0.70 * progress),
                "termination_bad_orientation": max(0.02, 0.35 * (1 - progress)),
                "curriculum_max": min(9.0, 1.0 + step / 900),
                "total_fps": 14500 + random.uniform(-350, 350),
            },
            step=step,
        )
        time.sleep(0.15)
    run.finish()
    print("live_demo 写入完成。")


if __name__ == "__main__":
    main()
