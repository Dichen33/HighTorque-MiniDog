from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ZhDBRun:
    """Small training-side logger for Zh-db.

    Usage:

        run = ZhDBRun("exp_001", name="第一次 PPO 训练", config={"algo": "PPO"})
        run.log({"mean_reward": 1.0, "loss_value": 0.2}, step=0)
        run.finish()
    """

    run_id: str
    name: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    url: str = "http://127.0.0.1:8765"
    enabled: bool = True
    timeout: float = 2.0
    silent: bool = True

    def log(self, metrics: dict[str, Any], step: int | float | None = None) -> bool:
        if not self.enabled:
            return False
        metric = dict(metrics)
        if step is not None:
            metric["step"] = step
        elif "step" not in metric:
            metric["step"] = int(time.time())
        payload = {
            "run_id": self.run_id,
            "run": {
                "name": self.name or self.run_id,
                "config": self.config,
                "status": "训练中",
            },
            "metric": metric,
        }
        return self._post("/api/log", payload)

    def finish(self, status: str = "已完成") -> bool:
        if not self.enabled:
            return False
        return self._post("/api/finish", {"run_id": self.run_id, "status": status})

    def _post(self, path: str, payload: dict[str, Any]) -> bool:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.url.rstrip("/") + path,
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            ok = bool(result.get("ok"))
            if not ok and not self.silent:
                print(f"Zh-db log failed: {result}")
            return ok
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if not self.silent:
                print(f"Zh-db log error: {exc}")
            return False
