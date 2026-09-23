from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "db-data" / "runs"


TAG_MAP = {
    "Train/mean_reward": "mean_reward",
    "Train/mean_episode_length": "mean_episode_length",
    "Policy/mean_std": "policy_mean_std",
    "Perf/total_fps": "total_fps",
    "Perf/collection_time": "collection_time",
    "Perf/learning_time": "learning_time",
    "Loss/value": "loss_value",
    "Loss/surrogate": "loss_surrogate",
    "Loss/entropy": "loss_entropy",
    "Loss/learning_rate": "loss_learning_rate",
    "Metrics/twist/error_vel_xy": "error_vel_xy",
    "Metrics/twist/error_vel_yaw": "error_vel_yaw",
    "Episode_Metrics/mean_action_acc": "mean_action_acc",
    "Curriculum/terrain_levels/mean": "curriculum_mean",
    "Curriculum/terrain_levels/max": "curriculum_max",
}


def _safe_metric_name(tag: str) -> str:
    if tag in TAG_MAP:
        return TAG_MAP[tag]
    name = tag
    name = name.replace("Episode_Reward/", "reward_")
    name = name.replace("Episode_Termination/", "termination_")
    name = name.replace("Curriculum/terrain_levels/", "curriculum_terrain_")
    name = name.replace("Train/", "train_")
    name = name.replace("Loss/", "loss_")
    name = name.replace("Policy/", "policy_")
    name = name.replace("Perf/", "perf_")
    name = name.replace("Metrics/", "metrics_")
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return name.lower()


def _record_hash(metric: dict[str, Any]) -> str:
    payload = {key: value for key, value in metric.items() if key != "record_hash"}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_event_file(run_dir: Path) -> Path:
    files = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No TensorBoard event file found in {run_dir}")
    return files[0]


def import_run(run_dir: Path, run_id: str, name: str | None = None) -> Path:
    event_file = _latest_event_file(run_dir)
    acc = EventAccumulator(str(run_dir))
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    tags = [tag for tag in tags if not tag.endswith("/time")]

    rows: dict[int, dict[str, Any]] = {}
    wall_times: dict[int, float] = {}
    for tag in tags:
        metric_name = _safe_metric_name(tag)
        for event in acc.Scalars(tag):
            row = rows.setdefault(
                int(event.step),
                {
                    "step": int(event.step),
                    "sample_step": int(event.step),
                    "iteration": int(event.step),
                },
            )
            row[metric_name] = float(event.value)
            wall_times[int(event.step)] = float(event.wall_time)

    metrics = []
    previous_hash = ""
    for step in sorted(rows):
        row = rows[step]
        wall_time = wall_times.get(step)
        if wall_time is not None:
            row["received_at"] = datetime.fromtimestamp(wall_time).strftime("%Y-%m-%d %H:%M:%S")
        row["prev_record_hash"] = previous_hash
        row["record_hash"] = _record_hash(row)
        previous_hash = row["record_hash"]
        metrics.append(row)

    created_at = ""
    if metrics and metrics[0].get("received_at"):
        created_at = str(metrics[0]["received_at"])
    else:
        created_at = datetime.fromtimestamp(event_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    data = {
        "id": run_id,
        "name": name or run_id,
        "created_at": created_at,
        "status": "已完成",
        "config": {
            "算法": "原生 MJLab + RSL-RL PPO",
            "任务": "Robot-Rough-v0" if "rough" in run_id.lower() else "MiniDog",
            "源目录": str(run_dir),
            "源事件文件": str(event_file),
            "源事件 SHA256": _source_sha256(event_file),
            "模型目录": str(run_dir),
        },
        "metrics": metrics,
    }

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    output = RUNS_ROOT / f"{run_id}.json"
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a TensorBoard RSL-RL run into Zh-db.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    output = import_run(args.run_dir.resolve(), args.run_id, args.name)
    print(output)


if __name__ == "__main__":
    main()
