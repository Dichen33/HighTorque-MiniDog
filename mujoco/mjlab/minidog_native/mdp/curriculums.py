from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def terrain_levels_velocity(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    asset: Entity = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    if terrain is None or terrain.cfg.terrain_generator is None:
        return {}

    command = env.command_manager.get_command(command_name)
    distance = torch.norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )
    command_distance = torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s

    terrain_length = terrain.cfg.terrain_generator.size[0]
    move_up = distance > 0.5 * terrain_length
    move_down = (distance < 0.35 * command_distance) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)

    levels = terrain.terrain_levels.float()
    result: dict[str, torch.Tensor] = {
        "mean": torch.mean(levels),
        "max": torch.max(levels),
    }

    terrain_types = getattr(terrain, "terrain_types", None)
    origins = getattr(terrain, "terrain_origins", None)
    if terrain_types is not None and origins is not None:
        sub_names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        if origins.shape[1] == len(sub_names):
            for index, name in enumerate(sub_names):
                mask = terrain_types == index
                if mask.any():
                    result[name] = torch.mean(levels[mask])
    return result
