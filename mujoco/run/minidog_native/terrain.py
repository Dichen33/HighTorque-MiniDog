from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput


@dataclass(kw_only=True)
class PlumBlossomTerrainCfg(SubTerrainCfg):
    """Two staggered rows of three blocks between two raised platforms."""

    block_size_range: tuple[float, float] = (0.12, 0.36)
    block_height_range: tuple[float, float] = (0.10, 0.32)
    platform_width: float = 0.80
    platform_height: float = 0.28
    floor_thickness: float = 0.10
    position_jitter: float = 0.0
    height_jitter: float = 0.0
    course_start_difficulty: float = 0.15

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        body = spec.body("terrain")
        difficulty = float(np.clip(difficulty, 0.0, 1.0))

        floor = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.size[0] / 2.0, self.size[1] / 2.0, self.floor_thickness / 2.0),
            pos=(self.size[0] / 2.0, self.size[1] / 2.0, -self.floor_thickness / 2.0),
        )
        return self._add_course(
            difficulty=difficulty,
            spec=spec,
            rng=rng,
            base_z=0.0,
            geometries=[TerrainGeometry(geom=floor, color=(0.55, 0.55, 0.55, 1.0))],
        )

    def _add_course(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
        base_z: float,
        geometries: list[TerrainGeometry],
    ) -> TerrainOutput:
        body = spec.body("terrain")

        block_size = self.block_size_range[1] + difficulty * (
            self.block_size_range[0] - self.block_size_range[1]
        )
        block_height = self.block_height_range[0] + difficulty * (
            self.block_height_range[1] - self.block_height_range[0]
        )
        platform_height = self.platform_height * (0.5 + 0.5 * difficulty)

        # Keep the 3.5 m course centered in the terrain patch. The returned
        # origin is the top center of the entrance platform.
        course_x = self.size[0] / 2.0 - 1.75
        course_y = self.size[1] / 2.0
        # Keep the base terrain geometries. The rough-grid subclass passes them
        # in so the course is added on top of rough ground rather than replacing it.

        platform_size = self.platform_width / 2.0
        for x in (course_x + 0.4, course_x + 3.1):
            platform = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(platform_size, platform_size, platform_height / 2.0),
                pos=(x, course_y, base_z + platform_height / 2.0),
            )
            geometries.append(TerrainGeometry(geom=platform, color=(0.35, 0.62, 0.20, 1.0)))

        centers = (
            (1.0, 0.2), (1.6, 0.2), (2.2, 0.2),
            (1.3, -0.2), (1.9, -0.2), (2.5, -0.2),
        )
        for local_x, local_y in centers:
            jitter_x = rng.uniform(-self.position_jitter, self.position_jitter)
            jitter_y = rng.uniform(-self.position_jitter, self.position_jitter)
            height = block_height + rng.uniform(-self.height_jitter, self.height_jitter)
            height = max(0.02, height)
            block = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(block_size / 2.0, block_size / 2.0, height / 2.0),
                pos=(
                    course_x + local_x + jitter_x,
                    course_y + local_y + jitter_y,
                    base_z + height / 2.0,
                ),
            )
            geometries.append(TerrainGeometry(geom=block, color=(0.35, 0.62, 0.20, 1.0)))

        origin = np.array((course_x + 0.4, course_y, base_z + platform_height))
        return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class RoughPlumBlossomTerrainCfg(PlumBlossomTerrainCfg):
    """Rough ground with a staggered plum-blossom stepping course."""

    rough_grid_width: float = 0.40
    rough_height_range: tuple[float, float] = (0.0, 0.10)
    course_corridor_width: float = 1.10
    course_pit_depth: float = 0.35

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        body = spec.body("terrain")
        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        rough_height = self.rough_height_range[0] + difficulty * (
            self.rough_height_range[1] - self.rough_height_range[0]
        )

        # Generate a shallow rough grid over the complete patch. The stepping
        # course is placed on top of the highest rough-grid level so the blocks
        # remain solidly supported even when the surrounding ground is uneven.
        terrain_height = 1.0
        num_x = max(1, int(self.size[0] / self.rough_grid_width))
        num_y = max(1, int(self.size[1] / self.rough_grid_width))
        cell_x = self.size[0] / num_x
        cell_y = self.size[1] / num_y
        heights = rng.uniform(-rough_height, rough_height, (num_x, num_y))
        geometries: list[TerrainGeometry] = []
        for ix in range(num_x):
            for iy in range(num_y):
                height = float(heights[ix, iy])
                # Once the stepping course is active, remove the lateral
                # bypass. The blocks should be the only usable route.
                if difficulty >= self.course_start_difficulty:
                    cell_center_y = (iy + 0.5) * cell_y
                    if abs(cell_center_y - self.size[1] / 2.0) > self.course_corridor_width / 2.0:
                        height = -self.course_pit_depth
                geom = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=(cell_x / 2.0, cell_y / 2.0, (terrain_height + height) / 2.0),
                    pos=(
                        (ix + 0.5) * cell_x,
                        (iy + 0.5) * cell_y,
                        -terrain_height / 2.0 + height / 2.0,
                    ),
                )
                normalized = 0.5 if rough_height == 0.0 else (height + rough_height) / (2.0 * rough_height)
                shade = float(np.clip(normalized, 0.0, 1.0))
                geometries.append(
                    TerrainGeometry(
                        geom=geom,
                        color=(0.42 + 0.10 * shade, 0.42 + 0.10 * shade, 0.42 + 0.10 * shade, 1.0),
                    )
                )

        # Level 0 is a genuine rough-locomotion warm-up. The obstacle course
        # starts from the next curriculum row, so training progresses from
        # walking to stepping stones and finally to the wall.
        if difficulty < self.course_start_difficulty:
            course_x = self.size[0] / 2.0 - 1.75
            origin = np.array((course_x + 0.4, self.size[1] / 2.0, rough_height))
            return TerrainOutput(origin=origin, geometries=geometries)

        return self._add_course(
            difficulty=difficulty,
            spec=spec,
            rng=rng,
            base_z=rough_height,
            geometries=geometries,
        )


@dataclass(kw_only=True)
class FlatHighWallTerrainCfg(SubTerrainCfg):
    """Flat ground with three repeated transverse walls and no stepping stones."""

    wall_start_difficulty: float = 0.10
    wall_height_range: tuple[float, float] = (0.08, 0.50)
    wall_length_frac: float = 1.0
    wall_thickness: float = 0.05
    wall_centers_x: tuple[float, float, float] = (2.9, 4.45, 6.0)
    floor_thickness: float = 0.10

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        floor = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.size[0] / 2.0, self.size[1] / 2.0, self.floor_thickness / 2.0),
            pos=(self.size[0] / 2.0, self.size[1] / 2.0, -self.floor_thickness / 2.0),
        )
        geometries = [TerrainGeometry(geom=floor, color=(0.52, 0.52, 0.52, 1.0))]

        if difficulty >= self.wall_start_difficulty:
            wall_progress = (difficulty - self.wall_start_difficulty) / max(
                1.0 - self.wall_start_difficulty, 1.0e-6
            )
            wall_height = self.wall_height_range[0] + wall_progress * (
                self.wall_height_range[1] - self.wall_height_range[0]
            )
            wall_length = self.wall_length_frac * self.size[1]
            center_y = self.size[1] / 2.0
            for center_x in self.wall_centers_x:
                wall = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=(self.wall_thickness / 2.0, wall_length / 2.0, wall_height / 2.0),
                    pos=(center_x, center_y, wall_height / 2.0),
                )
                geometries.append(TerrainGeometry(geom=wall, color=(0.72, 0.30, 0.16, 1.0)))

        # Spawn before the first wall. The exit is after all three walls.
        origin = np.array((1.5, self.size[1] / 2.0, 0.0))
        return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class FlatGapTerrainCfg(SubTerrainCfg):
    """Flat approach and landing areas separated by a transverse ditch."""

    gap_start_x: float = 2.35
    gap_width_range: tuple[float, float] = (0.10, 0.90)
    gap_depth_range: tuple[float, float] = (0.04, 0.45)
    floor_thickness: float = 0.10
    gap_floor_thickness: float = 0.08

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        body = spec.body("terrain")
        difficulty = float(np.clip(difficulty, 0.0, 1.0))

        # Keep the obstacle transverse to the route. Small random variations
        # prevent the policy from memorizing one exact edge and width.
        gap_width = self.gap_width_range[0] + difficulty * (
            self.gap_width_range[1] - self.gap_width_range[0]
        )
        gap_depth = self.gap_depth_range[0] + difficulty * (
            self.gap_depth_range[1] - self.gap_depth_range[0]
        )
        gap_width = max(0.05, gap_width + rng.uniform(-0.015, 0.015))
        gap_depth = max(0.02, gap_depth + rng.uniform(-0.01, 0.01))
        gap_start = self.gap_start_x
        gap_end = min(self.size[0] - 0.8, gap_start + gap_width)
        gap_width = gap_end - gap_start
        center_y = self.size[1] / 2.0
        floor_z = -self.floor_thickness / 2.0

        geometries: list[TerrainGeometry] = []

        def add_floor(x0: float, x1: float, z: float, color: tuple[float, float, float, float]) -> None:
            if x1 <= x0:
                return
            floor = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=((x1 - x0) / 2.0, self.size[1] / 2.0, self.floor_thickness / 2.0),
                pos=((x0 + x1) / 2.0, center_y, z),
            )
            geometries.append(TerrainGeometry(geom=floor, color=color))

        # There is no hidden full-height floor beneath the gap. The lower box
        # is the visible ditch bottom, so the robot can step down but cannot
        # exploit an infinite fall-through hole.
        add_floor(0.0, gap_start, floor_z, (0.52, 0.52, 0.52, 1.0))
        add_floor(gap_end, self.size[0], floor_z, (0.52, 0.52, 0.52, 1.0))

        gap_bottom = -gap_depth - self.gap_floor_thickness / 2.0
        bottom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(gap_width / 2.0, self.size[1] / 2.0, self.gap_floor_thickness / 2.0),
            pos=((gap_start + gap_end) / 2.0, center_y, gap_bottom),
        )
        geometries.append(TerrainGeometry(geom=bottom, color=(0.28, 0.28, 0.28, 1.0)))

        # The origin is before the ditch and the route ends after it. This
        # makes curriculum progress a meaningful measure of crossing success.
        origin = np.array((1.20, center_y, 0.0))
        return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class LowBarTerrainCfg(SubTerrainCfg):
    """Flat ground with repeated low bars whose clearance decreases by level."""

    clearance_range: tuple[float, float] = (0.24, 0.34)
    bar_radius: float = 0.025
    bar_length_frac: float = 0.85
    num_bars: int = 6
    floor_thickness: float = 0.10

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        floor = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.size[0] / 2.0, self.size[1] / 2.0, self.floor_thickness / 2.0),
            pos=(self.size[0] / 2.0, self.size[1] / 2.0, -self.floor_thickness / 2.0),
        )
        geometries = [TerrainGeometry(geom=floor, color=(0.50, 0.50, 0.50, 1.0))]

        clearance = self.clearance_range[1] - difficulty * (
            self.clearance_range[1] - self.clearance_range[0]
        )
        bar_z = clearance + self.bar_radius
        bar_length = self.bar_length_frac * self.size[1]
        center_y = self.size[1] / 2.0
        bar_color = (0.18, 0.34, 0.72, 1.0)
        positions = np.linspace(1.0, self.size[0] - 1.0, self.num_bars)

        for center_x in positions:
            bar = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=(self.bar_radius, bar_length / 2.0),
                pos=(center_x, center_y, bar_z),
                quat=(np.cos(np.pi / 4.0), np.sin(np.pi / 4.0), 0.0, 0.0),
            )
            geometries.append(TerrainGeometry(geom=bar, color=bar_color))
            post_height = bar_z
            for y_offset in (-bar_length / 2.0, bar_length / 2.0):
                post = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=(self.bar_radius, post_height / 2.0),
                    pos=(center_x, center_y + y_offset, post_height / 2.0),
                )
                geometries.append(TerrainGeometry(geom=post, color=bar_color))

        origin = np.array((1.5, center_y, 0.0))
        return TerrainOutput(origin=origin, geometries=geometries)
