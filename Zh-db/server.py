from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import socket
import socketserver
import struct
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
DB_ROOT = ROOT / "db-data"
RUNS_ROOT = DB_ROOT / "runs"
DOWNLOAD_ROOT = ROOT / "db-download"
GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


METRIC_INFO = {
    "mean_reward": ("平均奖励", "Train", "越高越好，代表策略更符合奖励设计"),
    "mean_episode_length": ("平均回合长度", "Train", "越接近上限越好，代表不易摔倒"),
    "policy_mean_std": ("策略标准差", "Policy", "过高说明探索强，过低可能过早收敛"),
    "total_fps": ("训练 FPS", "Perf", "越高越好，代表仿真/采样效率"),
    "collection_time": ("采样耗时", "Perf", "越低越好，升高可能是环境变慢"),
    "learning_time": ("学习耗时", "Perf", "越低越稳定，突增可能训练端瓶颈"),
    "wheel_roll_error_mean": ("轮子滚动误差均值", "Metrics", "越低越好，表示轮子滚动/目标速度更一致"),
    "error_vel_xy": ("平面速度误差", "Metrics", "越低越好，表示速度跟踪更准确"),
    "error_vel_yaw": ("转向速度误差", "Metrics", "越低越好，表示 yaw 跟踪更准确"),
    "loss_value": ("价值损失", "Loss", "下降并稳定通常较好"),
    "loss_surrogate": ("PPO 策略损失", "Loss", "小幅波动正常，剧烈波动需关注"),
    "loss_learning_rate": ("学习率", "Loss", "逐步衰减通常表示训练进入微调阶段"),
    "loss_entropy": ("策略熵", "Loss", "探索程度，过高可能动作太随机"),
    "termination_time_out": ("超时结束", "Termination", "越高越好，说明大部分回合跑满时长"),
    "termination_nan_detection": ("NaN 数值终止", "Termination", "越低越好，非零时需要检查动作/梯度/仿真稳定性"),
    "termination_bad_orientation": ("姿态失败", "Termination", "越低越好"),
    "termination_base_ground_contact": ("机身触地", "Termination", "越低越好"),
    "reward_wheel_roll_tracking": ("轮子滚动跟踪奖励", "Episode Reward", "越高越好，表示轮子滚动控制更好"),
    "reward_track_lin_vel": ("线速度跟踪奖励", "Episode Reward", "越高越好"),
    "reward_wheel_contact_bonus": ("轮子接触奖励", "Episode Reward", "越高通常表示轮子接触更合理"),
    "reward_leg_motion_penalty": ("腿部动作惩罚", "Episode Reward", "通常为负，绝对值越小表示腿部动作越平滑"),
    "reward_joint_torques": ("关节力矩惩罚", "Episode Reward", "通常为负，绝对值越小表示用力更省"),
    "reward_action_rate": ("动作变化惩罚", "Episode Reward", "一般为负，绝对值越小越平滑"),
    "reward_joint_pos_limits": ("关节限位惩罚", "Episode Reward", "接近 0 较好，说明关节很少打到限位"),
    "curriculum_max": ("课程最大难度", "Curriculum", "逐步上升表示课程正在推进"),
}


@dataclass
class Client:
    sock: socket.socket
    addr: tuple[str, int]


class RunStore:
    def __init__(self, root: Path, download_root: Path = DOWNLOAD_ROOT):
        self.root = root
        self.download_root = download_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.download_root.mkdir(parents=True, exist_ok=True)

    def list_runs(self) -> list[dict[str, Any]]:
        runs = []
        for path in sorted(self.root.glob("*.json")):
            try:
                data = self._load(path)
            except (OSError, json.JSONDecodeError):
                continue
            metrics = data.get("metrics", [])
            last = metrics[-1] if metrics else {}
            last_iteration = self._metric_iteration(data, len(metrics) - 1, last) if metrics else 0
            runs.append(
                {
                    "id": data.get("id", path.stem),
                    "name": data.get("name", path.stem),
                    "created_at": data.get("created_at", ""),
                    "steps": len(metrics),
                    "last_step": last.get("step", 0),
                    "last_iteration": last_iteration,
                    "last_reward": last.get("mean_reward"),
                    "status": data.get("status", "已完成"),
                }
            )
        return sorted(runs, key=lambda item: item.get("created_at", ""))

    def metric_catalog(self) -> dict[str, tuple[str, str, str]]:
        catalog = dict(METRIC_INFO)
        for path in self.root.glob("*.json"):
            try:
                data = self._load(path)
            except (OSError, json.JSONDecodeError):
                continue
            catalog.update(self._discover_metric_info(data))
        return catalog

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        found = self._find_run(run_id)
        if found is not None:
            path, data = found
            self._ensure_iteration_axis(data)
            data["metric_info"] = self._discover_metric_info(data)
            data["source_json"] = str(path)
            data["source_sha256"] = self._sha256(path)
            return data
        return None

    def append_metric(self, run_id: str, metric: dict[str, Any], run_meta: dict[str, Any] | None = None) -> dict[str, Any]:
        path = self.root / f"{run_id}.json"
        if path.exists():
            data = self._load(path)
        else:
            data = {
                "id": run_id,
                "name": (run_meta or {}).get("name", run_id),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": (run_meta or {}).get("status", "训练中"),
                "config": (run_meta or {}).get("config", {}),
                "metrics": [],
            }
        if run_meta:
            if "name" in run_meta:
                data["name"] = run_meta["name"]
            if "config" in run_meta and isinstance(run_meta["config"], dict):
                data["config"] = run_meta["config"]
            if "status" in run_meta:
                data["status"] = run_meta["status"]
        if "iteration" not in metric:
            metric["iteration"] = self._metric_iteration(data, len(data.get("metrics", [])), metric)
        if "sample_step" not in metric and "step" in metric:
            metric["sample_step"] = metric["step"]
        metric.setdefault("received_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        previous = data.get("metrics", [])[-1].get("record_hash", "") if data.get("metrics") else ""
        metric["prev_record_hash"] = previous
        metric["record_hash"] = self._record_hash(metric)
        data.setdefault("metrics", []).append(metric)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        data["metric_info"] = self._discover_metric_info(data)
        return data

    def export_run_csv(self, run_id: str, metric_filter: str = "all") -> dict[str, Any]:
        found = self._find_run(run_id)
        if found is None:
            raise ValueError(f"run not found: {run_id}")
        source_path, run = found
        self._ensure_iteration_axis(run)
        info = self._discover_metric_info(run)
        metrics = run.get("metrics", [])

        base_columns = ["iteration", "sample_step", "step", "received_at", "prev_record_hash", "record_hash"]
        metric_columns = []
        for key in self._numeric_metric_keys(run):
            if metric_filter != "all" and info.get(key, ("", "other", ""))[1] != metric_filter:
                continue
            metric_columns.append(key)
        columns = [key for key in base_columns if any(key in row for row in metrics)] + metric_columns

        safe_id = self._safe_name(run.get("id", run_id))
        safe_filter = self._safe_name(metric_filter)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = self.download_root / f"{safe_id}_{safe_filter}_{stamp}.csv"
        meta_path = csv_path.with_suffix(".meta.json")

        with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in metrics:
                writer.writerow({key: row.get(key, "") for key in columns})

        meta = {
            "run_id": run.get("id", run_id),
            "run_name": run.get("name", run_id),
            "metric_filter": metric_filter,
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_json": str(source_path),
            "source_json_sha256": self._sha256(source_path),
            "csv_path": str(csv_path),
            "csv_sha256": self._sha256(csv_path),
            "last_record_hash": metrics[-1].get("record_hash") if metrics else "",
            "hash_chain_status": self._hash_chain_status(metrics),
            "row_count": len(metrics),
            "columns": columns,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        meta["meta_path"] = str(meta_path)
        return meta

    def update_run_status(self, run_id: str, status: str) -> dict[str, Any]:
        path = self.root / f"{run_id}.json"
        if not path.exists():
            raise ValueError(f"没有找到 run：{run_id}")
        data = self._load(path)
        data["status"] = status
        data["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        data["metric_info"] = self._discover_metric_info(data)
        return data

    def delete_run(self, run_id: str) -> dict[str, Any]:
        found = self._find_run(run_id)
        if found is None:
            raise ValueError(f"run not found: {run_id}")

        path, data = found
        safe_id = self._safe_name(data.get("id", run_id))
        removed_exports: list[str] = []

        for export_path in self.download_root.glob(f"{safe_id}_*"):
            if export_path.is_file():
                export_path.unlink()
                removed_exports.append(str(export_path))

        path.unlink()
        return {
            "run_id": data.get("id", run_id),
            "run_name": data.get("name", run_id),
            "deleted_json": str(path),
            "deleted_exports": removed_exports,
        }

    def _load(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _find_run(self, run_id: str) -> tuple[Path, dict[str, Any]] | None:
        for path in self.root.glob("*.json"):
            try:
                data = self._load(path)
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("id", path.stem) == run_id:
                return path, data
        return None

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _record_hash(self, metric: dict[str, Any]) -> str:
        payload = {key: value for key, value in metric.items() if key != "record_hash"}
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def _hash_chain_status(self, metrics: list[dict[str, Any]]) -> str:
        if not metrics or not any("record_hash" in row for row in metrics):
            return "not_available"
        previous = ""
        for row in metrics:
            if row.get("prev_record_hash", "") != previous:
                return "broken"
            if row.get("record_hash") != self._record_hash(row):
                return "broken"
            previous = str(row.get("record_hash", ""))
        return "valid"

    def _safe_name(self, value: Any) -> str:
        text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
        return text or "run"

    def _ensure_iteration_axis(self, run: dict[str, Any]) -> None:
        metrics = run.get("metrics", [])
        for index, row in enumerate(metrics):
            if isinstance(row, dict):
                if "iteration" not in row:
                    row["iteration"] = self._metric_iteration(run, index, row)
                if "sample_step" not in row and "step" in row:
                    row["sample_step"] = row["step"]

    def _metric_iteration(self, run: dict[str, Any], index: int, metric: dict[str, Any]) -> float:
        value = metric.get("iteration")
        if isinstance(value, (int, float)):
            return float(value)
        config = run.get("config", {})
        rollout_size = config.get("rollout_size")
        if isinstance(rollout_size, (int, float)) and rollout_size > 0 and isinstance(metric.get("step"), (int, float)):
            return round(float(metric["step"]) / float(rollout_size), 6)
        return float(index + 1)

    def _discover_metric_info(self, run: dict[str, Any]) -> dict[str, tuple[str, str, str]]:
        info = dict(METRIC_INFO)
        for key in self._numeric_metric_keys(run):
            if key not in info:
                info[key] = infer_metric_info(key)
        return info

    def _numeric_metric_keys(self, run: dict[str, Any]) -> list[str]:
        keys = []
        for row in run.get("metrics", []):
            for key, value in row.items():
                if key not in {"step", "sample_step", "iteration"} and isinstance(value, (int, float)) and key not in keys:
                    keys.append(key)
        return keys


def infer_metric_info(key: str) -> tuple[str, str, str]:
    label = key.replace("/", "_").replace(".", "_").replace("-", "_")
    words = [part for part in label.split("_") if part]
    zh_name = " ".join(words)
    lower = label.lower()

    if lower.startswith(("reward_", "episode_reward_")):
        module = "Episode Reward"
        desc = "奖励分项，正值通常越高越好，负值通常表示惩罚项。"
    elif "penalty" in lower or lower.startswith("penalty_"):
        module = "Episode Reward"
        desc = "惩罚项，通常越接近 0 越好。"
    elif lower.startswith(("termination_", "episode_termination_")) or "termination" in lower:
        module = "Termination"
        desc = "回合终止原因，失败类终止越低越好，time_out 越高越好。"
    elif lower.startswith("loss_") or "/loss" in lower or "loss" in lower:
        module = "Loss"
        desc = "优化损失曲线，用于判断 PPO/价值函数更新是否稳定。"
    elif lower.startswith("policy_") or "std" in lower or "entropy" in lower:
        module = "Policy"
        desc = "策略探索和动作分布相关指标。"
    elif lower.startswith("curriculum_") or "terrain" in lower or "difficulty" in lower:
        module = "Curriculum"
        desc = "课程学习或地形难度相关指标。"
    elif "fps" in lower or "time" in lower:
        module = "Perf"
        desc = "训练效率或耗时指标。"
    elif "error" in lower or "tracking" in lower or "vel" in lower:
        module = "Metrics"
        desc = "控制误差或跟踪精度指标。"
    else:
        module = "其他"
        desc = "训练脚本记录的自定义数值指标。"

    return zh_name, module, desc


class Analyzer:
    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        self.api_url = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")

    def analyze(self, run: dict[str, Any]) -> dict[str, Any]:
        summary = self._structured_summary(run)
        text = self._summary_text(summary)
        if not self.api_key:
            return {"source": "规则分析", "summary": summary, "text": text}
        try:
            ai_text = self._deepseek(run, text)
            return {"source": "DeepSeek", "summary": summary, "text": ai_text}
        except (OSError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            return {
                "source": "规则分析",
                "summary": summary,
                "text": text + f"\n\nDeepSeek 调用失败，已回退到规则分析：{exc}",
            }

    def _structured_summary(self, run: dict[str, Any]) -> dict[str, Any]:
        metrics = run.get("metrics", [])
        if not metrics:
            return {
                "title": run.get("name", run.get("id", "未知训练")),
                "verdict": "暂无数据",
                "health_score": 0,
                "key_numbers": [],
                "strengths": [],
                "risks": ["当前训练没有可分析的指标。"],
                "suggestions": ["先确认训练脚本是否正在写入 db-data/runs 中的 JSON 指标。"],
            }

        first = metrics[0]
        last = metrics[-1]
        tail = metrics[max(0, len(metrics) - max(5, len(metrics) // 5)) :]
        mean_tail = self._mean_values(tail)
        risks = []
        strengths = []
        suggestions = []
        score = 50

        key_numbers = [
            {"name": "最后步数", "value": self._fmt(last.get("step"))},
            {"name": "尾段平均奖励", "value": self._fmt(mean_tail.get("mean_reward"))},
            {"name": "尾段速度误差", "value": self._fmt(mean_tail.get("error_vel_xy"))},
            {"name": "尾段 yaw 误差", "value": self._fmt(mean_tail.get("error_vel_yaw"))},
            {"name": "尾段超时结束", "value": self._fmt(mean_tail.get("termination_time_out"))},
            {"name": "尾段课程难度", "value": self._fmt(mean_tail.get("curriculum_max"))},
        ]

        reward_delta = self._delta(first, last, "mean_reward")
        if reward_delta is not None:
            if reward_delta > 20:
                strengths.append(f"平均奖励从 {first.get('mean_reward'):.2f} 提升到 {last.get('mean_reward'):.2f}，策略在学习。")
                score += 14
            elif reward_delta < -10:
                risks.append("平均奖励后期下降明显，可能是课程难度提升过快或奖励项冲突。")
                suggestions.append("降低课程难度增长速度，或检查正奖励和惩罚项是否互相拉扯。")
                score -= 16
            else:
                risks.append("平均奖励提升不明显，可以检查速度跟踪奖励、姿态惩罚和动作惩罚权重。")
                suggestions.append("先固定课程难度，单独提高速度跟踪奖励权重观察奖励是否恢复上升。")
                score -= 8

        ep_len = mean_tail.get("mean_episode_length")
        if ep_len is not None:
            if ep_len > 2500:
                strengths.append("平均回合长度接近上限，基础生存能力较好。")
                score += 12
            elif ep_len < 1200:
                risks.append("平均回合长度偏低，机器人可能频繁提前终止，需要优先处理摔倒/姿态问题。")
                suggestions.append("优先处理 termination_bad_orientation 和 base_ground_contact，再扩大速度命令范围。")
                score -= 18

        vel_xy = mean_tail.get("error_vel_xy")
        if vel_xy is not None:
            if vel_xy > 1.0:
                risks.append(f"平面速度误差偏高，尾段均值约 {vel_xy:.2f}，说明 x/y 速度跟踪较差。")
                suggestions.append("提高 track_lin_vel 权重，或把 vx/vy 命令范围做课程式放大。")
                score -= 14
            else:
                strengths.append(f"平面速度误差较低，尾段均值约 {vel_xy:.2f}。")
                score += 10

        vel_yaw = mean_tail.get("error_vel_yaw")
        if vel_yaw is not None:
            if vel_yaw > 1.0:
                risks.append(f"转向速度误差偏高，尾段均值约 {vel_yaw:.2f}，yaw 控制仍需加强。")
                suggestions.append("提高 yaw-rate tracking 奖励，降低过大的随机探索，观察转向误差是否下降。")
                score -= 12
            else:
                strengths.append(f"转向速度误差较稳定，尾段均值约 {vel_yaw:.2f}。")
                score += 8

        bad_orientation = mean_tail.get("termination_bad_orientation")
        base_contact = mean_tail.get("termination_base_ground_contact")
        if bad_orientation and bad_orientation > 0.2:
            risks.append("姿态失败占比偏高，可能需要降低课程难度上升速度或增强姿态稳定奖励。")
            suggestions.append("增加 roll/pitch 稳定奖励，或降低动作残差尺度。")
            score -= 14
        if base_contact and base_contact > 0.2:
            risks.append("机身触地终止偏多，建议检查机身高度、足端清地高度和碰撞体。")
            suggestions.append("检查站立高度、足端 clearance、机身碰撞体和 base_ground_contact 阈值。")
            score -= 14

        entropy = mean_tail.get("loss_entropy")
        std = mean_tail.get("policy_mean_std")
        if entropy and entropy > 35:
            risks.append("策略熵较高，探索很强；若后期仍不收敛，可降低 ent_coef。")
            suggestions.append("若奖励已经停滞，可适当降低 ent_coef，让策略更快收敛。")
            score -= 6
        if std and std > 3.5:
            risks.append("策略动作标准差偏高，动作随机性大，可能导致速度跟踪和姿态波动。")
            suggestions.append("约束 action_std 或增加 action_rate 惩罚，减少动作抖动。")
            score -= 8

        fps = mean_tail.get("total_fps")
        if fps and fps > 10000:
            strengths.append("训练 FPS 较高，仿真采样性能正常。")
            score += 6

        wheel_error = mean_tail.get("wheel_roll_error_mean")
        if wheel_error is not None:
            if wheel_error > 4.5:
                risks.append(f"轮子滚动误差仍偏高，尾段均值约 {wheel_error:.2f}，轮子/足端速度配合可能不够好。")
                suggestions.append("提高 wheel_roll_tracking 权重，或检查轮子滚动误差的计算尺度是否过大。")
                score -= 8
            else:
                strengths.append(f"轮子滚动误差可接受，尾段均值约 {wheel_error:.2f}。")
                score += 4

        timeout = mean_tail.get("termination_time_out")
        nan_detection = mean_tail.get("termination_nan_detection")
        if timeout is not None and timeout > 0.75:
            strengths.append("大部分回合以 timeout 正常结束，生存能力较强。")
            score += 8
        if nan_detection is not None and nan_detection > 0.01:
            risks.append("出现 NaN 数值终止，可能存在动作越界、奖励爆炸或仿真不稳定。")
            suggestions.append("记录 NaN 前后的动作、关节速度和 reward 分项，必要时降低学习率或 action scale。")
            score -= 20

        learning_rate = mean_tail.get("loss_learning_rate")
        if learning_rate is not None and learning_rate < 0.001:
            strengths.append("学习率已衰减到较低水平，训练进入微调阶段。")

        curriculum = mean_tail.get("curriculum_max")
        if curriculum is not None:
            if curriculum >= 8.0:
                strengths.append("课程难度已经推进到较高区间。")
            elif reward_delta is not None and reward_delta < 5:
                suggestions.append("课程难度还不高但奖励提升慢，可以先停住难度，集中优化奖励设计。")

        reward_items = {key: value for key, value in mean_tail.items() if key.startswith("reward_")}
        weak_rewards = []
        heavy_penalties = []
        for key, value in reward_items.items():
            if "penalty" in key or key in {"reward_action_rate", "reward_joint_torques", "reward_leg_motion_penalty"}:
                if value < -0.6:
                    heavy_penalties.append((key, value))
            elif value < 0.35:
                weak_rewards.append((key, value))
        if weak_rewards:
            names = ", ".join(f"{name}={value:.2f}" for name, value in weak_rewards[:3])
            risks.append(f"部分正奖励分项偏低：{names}。")
            suggestions.append("优先检查偏低的正奖励是否尺度太小，或是否被惩罚项抵消。")
            score -= min(10, 3 * len(weak_rewards))
        if heavy_penalties:
            names = ", ".join(f"{name}={value:.2f}" for name, value in heavy_penalties[:3])
            risks.append(f"部分惩罚项绝对值偏大：{names}。")
            suggestions.append("若机器人动作抖动或用力过大，增加动作平滑约束；若学习太保守，则降低对应惩罚权重。")
            score -= min(10, 3 * len(heavy_penalties))

        if not suggestions:
            suggestions.append("当前曲线整体健康，可以延长训练或轻微扩大命令范围继续验证。")

        score = int(max(0, min(100, score)))
        if score >= 78:
            verdict = "健康"
        elif score >= 55:
            verdict = "可继续训练，但需要观察"
        else:
            verdict = "存在明显风险，建议先调参"

        return {
            "title": run.get("name", run.get("id", "未知训练")),
            "verdict": verdict,
            "health_score": score,
            "key_numbers": key_numbers,
            "strengths": strengths,
            "risks": risks,
            "suggestions": suggestions,
        }

    def _summary_text(self, summary: dict[str, Any]) -> str:
        lines = [
            f"训练：{summary['title']}",
            f"总体判断：{summary['verdict']}（健康分 {summary['health_score']}/100）",
        ]
        if summary["key_numbers"]:
            lines.append("\n关键指标：")
            lines.extend([f"- {item['name']}：{item['value']}" for item in summary["key_numbers"]])
        if summary["strengths"]:
            lines.append("\n做得比较好的地方：")
            lines.extend([f"- {item}" for item in summary["strengths"]])
        if summary["risks"]:
            lines.append("\n需要关注的问题：")
            lines.extend([f"- {item}" for item in summary["risks"]])
        lines.append("\n下一步建议：")
        lines.extend([f"- {item}" for item in summary["suggestions"]])
        return "\n".join(lines)

    def _fmt(self, value: Any) -> str:
        if isinstance(value, (int, float)):
            if abs(float(value)) >= 100:
                return f"{float(value):.0f}"
            return f"{float(value):.3f}"
        if value is None:
            return "-"
        else:
            return str(value)

    def _deepseek(self, run: dict[str, Any], rule_summary: str) -> str:
        compact = {
            "name": run.get("name"),
            "config": run.get("config", {}),
            "last_metrics": run.get("metrics", [])[-20:],
            "rule_summary": rule_summary,
        }
        payload = {
            "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            "messages": [
                {
                    "role": "system",
                    "content": "你是四足机器人强化学习训练分析助手。请用中文给出简洁、工程可执行的诊断。",
                },
                {
                    "role": "user",
                    "content": "请根据以下训练曲线数据分析奖励、速度跟踪、姿态、终止原因、探索度和下一步调参建议：\n"
                    + json.dumps(compact, ensure_ascii=False),
                },
            ],
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    def _delta(self, first: dict[str, Any], last: dict[str, Any], key: str) -> float | None:
        if key not in first or key not in last:
            return None
        return float(last[key]) - float(first[key])

    def _mean_values(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        values: dict[str, list[float]] = {}
        for row in rows:
            for key, value in row.items():
                if isinstance(value, (int, float)) and key != "step":
                    values.setdefault(key, []).append(float(value))
        return {key: sum(items) / len(items) for key, items in values.items() if items}


class WebSocketHub:
    def __init__(self, store: RunStore):
        self.store = store
        self.analyzer = Analyzer()
        self.clients: list[Client] = []
        self.lock = threading.Lock()

    def add(self, client: Client) -> None:
        with self.lock:
            self.clients.append(client)

    def remove(self, sock: socket.socket) -> None:
        with self.lock:
            self.clients = [client for client in self.clients if client.sock is not sock]

    def handle_message(self, client: Client, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "list_runs":
            self.send(client.sock, {"type": "runs", "runs": self.store.list_runs(), "metric_info": self.store.metric_catalog()})
        elif kind == "get_run":
            run = self.store.get_run(str(message.get("run_id", "")))
            self.send(client.sock, {"type": "run", "run": run})
        elif kind == "analyze":
            run = self.store.get_run(str(message.get("run_id", "")))
            if run is None:
                self.send(client.sock, {"type": "analysis", "error": "没有找到该训练次数。"})
            else:
                self.send(client.sock, {"type": "analysis", **self.analyzer.analyze(run)})
        elif kind == "export_csv":
            try:
                result = self.store.export_run_csv(
                    str(message.get("run_id", "")),
                    str(message.get("metric_filter", "all")),
                )
                self.send(client.sock, {"type": "export_csv", "ok": True, **result})
            except (OSError, ValueError) as exc:
                self.send(client.sock, {"type": "export_csv", "ok": False, "error": str(exc)})
        elif kind == "delete_run":
            run_id = str(message.get("run_id", ""))
            try:
                result = self.store.delete_run(run_id)
                self.send(client.sock, {"type": "delete_run", "ok": True, **result})
                self.broadcast(
                    {
                        "type": "run_deleted",
                        "run_id": result["run_id"],
                        "runs": self.store.list_runs(),
                        "metric_info": self.store.metric_catalog(),
                    }
                )
            except (OSError, ValueError) as exc:
                self.send(client.sock, {"type": "delete_run", "ok": False, "error": str(exc)})
        elif kind == "append_metric":
            run_id = str(message.get("run_id", "live_run"))
            metric = message.get("metric", {})
            run = self.store.append_metric(run_id, metric)
            self.broadcast({"type": "run_updated", "run": run, "runs": self.store.list_runs()})
        else:
            self.send(client.sock, {"type": "error", "message": f"未知消息类型：{kind}"})

    def broadcast(self, payload: dict[str, Any]) -> None:
        with self.lock:
            clients = list(self.clients)
        for client in clients:
            self.send(client.sock, payload)

    def send(self, sock: socket.socket, payload: dict[str, Any]) -> None:
        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            sock.sendall(self._frame(data))
        except OSError:
            self.remove(sock)

    def _frame(self, data: bytes) -> bytes:
        length = len(data)
        if length < 126:
            return bytes([0x81, length]) + data
        if length < 65536:
            return bytes([0x81, 126]) + struct.pack("!H", length) + data
        return bytes([0x81, 127]) + struct.pack("!Q", length) + data


class RequestHandler(socketserver.BaseRequestHandler):
    hub: WebSocketHub

    def handle(self) -> None:
        request = self.request.recv(65536)
        if not request:
            return
        header_text = request.decode("utf-8", errors="ignore")
        request_line = header_text.splitlines()[0]
        parts = request_line.split()
        if len(parts) < 2:
            return
        method = parts[0].upper()
        path = parts[1].split("?", 1)[0]
        headers = self._parse_headers(header_text)
        if headers.get("upgrade", "").lower() == "websocket":
            self._handle_ws(headers)
        elif method == "POST" and path == "/api/log":
            self._handle_api_log(request, headers)
        elif method == "POST" and path == "/api/finish":
            self._handle_api_finish(request, headers)
        else:
            self._handle_http(path)

    def _handle_ws(self, headers: dict[str, str]) -> None:
        key = headers.get("sec-websocket-key", "")
        accept = base64.b64encode(hashlib.sha1((key + GUID).encode("ascii")).digest()).decode("ascii")
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        )
        self.request.sendall(response.encode("ascii"))
        client = Client(self.request, self.client_address)
        self.hub.add(client)
        self.hub.send(self.request, {"type": "hello", "message": "WebSocket 已连接"})
        try:
            while True:
                payload = self._read_frame(self.request)
                if payload is None:
                    break
                try:
                    message = json.loads(payload.decode("utf-8"))
                except json.JSONDecodeError:
                    self.hub.send(self.request, {"type": "error", "message": "JSON 解析失败"})
                    continue
                self.hub.handle_message(client, message)
        finally:
            self.hub.remove(self.request)

    def _read_frame(self, sock: socket.socket) -> bytes | None:
        head = self._recv_exact(sock, 2)
        if not head:
            return None
        opcode = head[0] & 0x0F
        if opcode == 0x8:
            return None
        masked = head[1] & 0x80
        length = head[1] & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._recv_exact(sock, 2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._recv_exact(sock, 8))[0]
        mask = self._recv_exact(sock, 4) if masked else b""
        data = self._recv_exact(sock, length)
        if not data:
            return None
        if masked:
            data = bytes(byte ^ mask[index % 4] for index, byte in enumerate(data))
        return data

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                return b""
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _handle_http(self, path: str) -> None:
        if path == "/":
            path = "/index.html"
        file_path = (WEB_ROOT / path.lstrip("/")).resolve()
        if not str(file_path).startswith(str(WEB_ROOT.resolve())) or not file_path.exists() or not file_path.is_file():
            self._send_http(404, b"Not Found", "text/plain; charset=utf-8")
            return
        content_type = self._content_type(file_path)
        self._send_http(200, file_path.read_bytes(), content_type)

    def _handle_api_log(self, request: bytes, headers: dict[str, str]) -> None:
        try:
            payload = self._read_json_body(request, headers)
            run_id = str(payload.get("run_id", "live_run"))
            metric = payload.get("metric", {})
            if not isinstance(metric, dict):
                raise ValueError("metric 必须是对象")
            run_meta = payload.get("run", {})
            run = self.hub.store.append_metric(run_id, metric, run_meta if isinstance(run_meta, dict) else None)
            self.hub.broadcast({"type": "run_updated", "run": run, "runs": self.hub.store.list_runs()})
            self._send_json({"ok": True, "run_id": run_id, "points": len(run.get("metrics", []))})
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)

    def _handle_api_finish(self, request: bytes, headers: dict[str, str]) -> None:
        try:
            payload = self._read_json_body(request, headers)
            run_id = str(payload.get("run_id", "live_run"))
            status = str(payload.get("status", "已完成"))
            run = self.hub.store.update_run_status(run_id, status)
            self.hub.broadcast({"type": "run_updated", "run": run, "runs": self.hub.store.list_runs()})
            self._send_json({"ok": True, "run_id": run_id, "status": status})
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)

    def _read_json_body(self, request: bytes, headers: dict[str, str]) -> dict[str, Any]:
        header_end = request.find(b"\r\n\r\n")
        initial = b"" if header_end < 0 else request[header_end + 4 :]
        length = int(headers.get("content-length", "0") or "0")
        body = initial
        while len(body) < length:
            chunk = self.request.recv(length - len(body))
            if not chunk:
                break
            body += chunk
        if not body:
            raise ValueError("请求体为空")
        data = json.loads(body.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return data

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        reason = {200: "OK", 400: "Bad Request"}.get(status, "OK")
        headers = (
            f"HTTP/1.1 {status} {reason}\r\n"
            "Content-Type: application/json; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        )
        self.request.sendall(headers.encode("utf-8") + body)

    def _send_http(self, status: int, body: bytes, content_type: str) -> None:
        reason = {200: "OK", 404: "Not Found"}.get(status, "OK")
        headers = (
            f"HTTP/1.1 {status} {reason}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        )
        self.request.sendall(headers.encode("utf-8") + body)

    def _content_type(self, path: Path) -> str:
        if path.suffix == ".html":
            return "text/html; charset=utf-8"
        if path.suffix == ".css":
            return "text/css; charset=utf-8"
        if path.suffix == ".js":
            return "application/javascript; charset=utf-8"
        if path.suffix == ".json":
            return "application/json; charset=utf-8"
        return "application/octet-stream"

    def _parse_headers(self, text: str) -> dict[str, str]:
        headers = {}
        for line in text.split("\r\n")[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()
        return headers


class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zh-db local WebSocket training dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = RunStore(RUNS_ROOT)
    RequestHandler.hub = WebSocketHub(store)
    with ThreadedServer((args.host, args.port), RequestHandler) as server:
        print(f"Zh-db 已启动：http://{args.host}:{args.port}")
        print(f"WebSocket：ws://{args.host}:{args.port}/ws")
        server.serve_forever()


if __name__ == "__main__":
    main()
