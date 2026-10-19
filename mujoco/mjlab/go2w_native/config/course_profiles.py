"""Reusable terrain profiles for the MiniDog obstacle curriculum.

The Go2 reference keeps terrain composition in one task configuration.  This
module applies the same rule to the native MJLab task, while leaving the
custom MuJoCo geometries in ``terrain.py`` unchanged.
"""

from __future__ import annotations

from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    BoxRandomGridTerrainCfg,
    HfPerlinNoiseTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    TerrainGeneratorCfg,
)

from ..terrain import FlatGapTerrainCfg, FlatHighWallTerrainCfg, RoughPlumBlossomTerrainCfg


BASE_TERRAIN_NAMES = ("flat", "random_rough", "perlin_noise")
STANDARD_OBSTACLE_NAMES = (
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "random_grid",
    "sloped_terrain",
)
SPECIAL_OBSTACLE_NAMES = ("flat_gap", "rough_plum_blossom", "flat_high_wall")

# ROBOCON2027.xml is the medium checkpoint. The final generated rows go beyond
# it so the policy is not capped at the competition scene's nominal sizes.
XML_MEDIUM_OBSTACLE = {
    "step_height": 0.24,
    "stepping_stone_size": 0.20,
    "gap_width": 0.42,
    "gap_depth": 0.22,
    "plum_block_height": 0.20,
    "high_wall_height": 0.30,
}


def plum_course_terrain_generator() -> TerrainGeneratorCfg:
    """Build the complete grid used by ``Robot-PlumBlossom-v0``.

    Curriculum mode creates one column per named terrain and one difficulty
    row per level.  The curriculum manager decides which columns may be
    sampled; all columns remain compiled so level changes do not rebuild the
    MuJoCo model during training.
    """

    size = (8.0, 8.0)
    return TerrainGeneratorCfg(
        size=size,
        border_width=1.0,
        num_rows=10,
        num_cols=len(BASE_TERRAIN_NAMES + STANDARD_OBSTACLE_NAMES + SPECIAL_OBSTACLE_NAMES),
        curriculum=True,
        sub_terrains={
            "flat": BoxFlatTerrainCfg(proportion=0.10, size=size),
            "pyramid_stairs": BoxPyramidStairsTerrainCfg(
                proportion=0.10, step_height_range=(0.04, 0.40), step_width=0.32, size=size
            ),
            "pyramid_stairs_inv": BoxInvertedPyramidStairsTerrainCfg(
                proportion=0.08, step_height_range=(0.04, 0.36), step_width=0.32, size=size
            ),
            "random_grid": BoxRandomGridTerrainCfg(
                proportion=0.10, grid_width=0.30, grid_height_range=(0.02, 0.30), size=size
            ),
            "random_rough": HfRandomUniformTerrainCfg(
                proportion=0.08,
                noise_range=(0.0, 0.04),
                noise_step=0.01,
                horizontal_scale=0.16,
                downsampled_scale=0.16,
                border_width=0.25,
                base_thickness_ratio=100.0,
                size=size,
            ),
            "perlin_noise": HfPerlinNoiseTerrainCfg(
                proportion=0.07,
                height_range=(0.0, 0.04),
                octaves=2,
                persistence=0.4,
                lacunarity=2.0,
                horizontal_scale=0.16,
                resolution=0.16,
                border_width=0.25,
                base_thickness_ratio=100.0,
                size=size,
            ),
            "sloped_terrain": HfPyramidSlopedTerrainCfg(
                proportion=0.05,
                slope_range=(0.03, 0.18),
                platform_width=2.0,
                border_width=0.25,
                base_thickness_ratio=100.0,
                horizontal_scale=0.16,
                size=size,
            ),
            "flat_gap": FlatGapTerrainCfg(
                proportion=0.12,
                gap_start_x=2.35,
                gap_width_range=(0.10, 0.90),
                gap_depth_range=(0.04, 0.45),
            ),
            "rough_plum_blossom": RoughPlumBlossomTerrainCfg(
                proportion=0.22,
                block_size_range=(0.12, 0.36),
                block_height_range=(0.10, 0.32),
                platform_width=0.80,
                platform_height=0.28,
                rough_height_range=(0.0, 0.10),
                course_start_difficulty=0.15,
            ),
            "flat_high_wall": FlatHighWallTerrainCfg(
                proportion=0.10,
                wall_start_difficulty=0.10,
                wall_height_range=(0.08, 0.50),
                wall_length_frac=1.0,
                wall_thickness=0.05,
                wall_centers_x=(2.9, 4.45, 6.0),
            ),
        },
    )
