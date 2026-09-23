from __future__ import annotations

try:
    from mjlab.tasks.registry import register_mjlab_task
except ModuleNotFoundError as exc:  # pragma: no cover - depends on native mjlab install
    raise ModuleNotFoundError(
        "minidog_native requires the native mjlab package. Install/activate mjlab "
        "before using `uv run train Robot-Flat-v0` or `uv run play Robot-Rough-v0`."
    ) from exc

from .config.env_cfgs import crawl_env_cfg, flat_env_cfg, rough_env_cfg
from .config.rl_cfg import crawl_ppo_runner_cfg, flat_ppo_runner_cfg, rough_ppo_runner_cfg


register_mjlab_task(
    task_id="Robot-Flat-v0",
    env_cfg=flat_env_cfg(),
    play_env_cfg=flat_env_cfg(play=True),
    rl_cfg=flat_ppo_runner_cfg(),
)

register_mjlab_task(
    task_id="Robot-Rough-v0",
    env_cfg=rough_env_cfg(),
    play_env_cfg=rough_env_cfg(play=True),
    rl_cfg=rough_ppo_runner_cfg(),
)

register_mjlab_task(
    task_id="Robot-Crawl-v0",
    env_cfg=crawl_env_cfg(),
    play_env_cfg=crawl_env_cfg(play=True),
    rl_cfg=crawl_ppo_runner_cfg(),
)
