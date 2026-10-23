"""Native 3D volume geometry and missing-aware scalar encoding.

The renderer samples nearest voxels: unsupported cells never interpolate into
observations. Each voxel keeps its scalar separate from its support flag, so
color saturation cannot change a maximum or an isosurface. Support zero marks
missing observations; negative values and genuine zeros remain
observations, and an all-missing ray stays transparent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.environment._fastplotlib import ColorScale


@dataclass(frozen=True)
class UniformGrid:
    """Active cells of a uniform grid and the cell-center to world affine.

    ``matrix`` maps a cell's index ``(i, j[, k])`` to its center in world
    ``(x, y[, z])``, any environment frame transform included.
    """

    active_mask: NDArray[np.bool_]
    matrix: NDArray[np.float64]

    def encode(self, values: NDArray[np.float64]) -> NDArray[np.float32]:
        """Contiguous reversed-axis channels: unclipped scalar and support.

        Color limits never participate in scalar encoding. Keeping support in
        its own channel preserves negative values and true zeros without a
        scalar sentinel that could affect thresholds or maximum selection.
        """
        grid = np.zeros((*self.active_mask.shape, 2), dtype=np.float32)
        finite = np.isfinite(values)
        grid[self.active_mask] = np.column_stack(
            (
                np.where(finite, values, 0.0),
                finite,
            )
        )
        n_dims = self.active_mask.ndim
        return np.ascontiguousarray(grid.transpose(*range(n_dims - 1, -1, -1), n_dims))

    @property
    def middle_plane(self) -> tuple[float, float, float, float]:
        """A world-horizontal plane through a 3-D volume's central voxel."""
        center = (np.asarray(self.active_mask.shape) - 1) / 2
        world = self.matrix @ np.r_[center, 1.0]
        return (0.0, 0.0, 1.0, -float(world[2]))


def uniform_grid(env: Environment) -> UniformGrid | None:
    """The environment's uniform grid cells, transformed grids included.

    None when the bins are not a grid, or a grid with unequal cells.
    """
    from neurospatial.environment._bin_mapping import _lattice

    lattice = _lattice(env.binning)
    if lattice is None:
        return None
    forward, grid = lattice
    widths = np.array([np.diff(edge)[0] for edge in grid.edges])
    if any(
        not np.allclose(np.diff(edge), width, rtol=1e-7, atol=1e-12)
        for edge, width in zip(grid.edges, widths, strict=True)
    ):
        return None
    # Native image/volume coordinate (0, ...) is a pixel CENTER, not an edge.
    n_dims = len(grid.edges)
    native = np.eye(n_dims + 1)
    native[:n_dims, :n_dims] = np.diag(widths)
    native[:n_dims, n_dims] = np.array([edge[0] for edge in grid.edges]) + widths / 2
    return UniformGrid(grid.active_mask, forward @ native)


def resolve_field_rendering(
    env: Environment,
    mode: str,
    *,
    volume_threshold: float | None = None,
    slice_plane: tuple[float, float, float, float] | None = None,
) -> str:
    """Validate explicit requests without importing or allocating a viewer."""
    if mode not in {"auto", "points", "mip", "iso", "slice"}:
        raise ValueError(
            "field_rendering must be 'auto', 'points', 'mip', 'iso', or 'slice'."
        )
    supported = env.n_dims == 3 and uniform_grid(env) is not None
    if mode in {"mip", "iso", "slice"} and not supported:
        raise ValueError(
            f"field_rendering={mode!r} requires a uniform regular 3D grid "
            "(affine transformed grids are supported). Use field_rendering='points' "
            "or 'auto' for this environment."
        )
    resolved = ("mip" if supported else "points") if mode == "auto" else mode
    if volume_threshold is not None:
        if resolved != "iso":
            raise ValueError(
                "volume_threshold is used only with field_rendering='iso'."
            )
        if not np.isfinite(volume_threshold):
            raise ValueError("volume_threshold must be finite, in field data units.")
    if slice_plane is not None:
        if resolved != "slice":
            raise ValueError("slice_plane is used only with field_rendering='slice'.")
        plane = np.asarray(slice_plane, dtype=float)
        if (
            plane.shape != (4,)
            or not np.all(np.isfinite(plane))
            or not np.any(plane[:3])
        ):
            raise ValueError(
                "slice_plane must contain finite (a, b, c, d), with nonzero normal, "
                "for the WORLD XYZ plane a*x + b*y + c*z + d = 0."
            )
    return resolved


class VolumeField:
    """One native ImageVolumeGraphic, updated with one bounded frame at a time."""

    def __init__(
        self,
        subplot: Any,
        env: Environment,
        values: NDArray[np.float64],
        scale: ColorScale,
        *,
        mode: str,
        threshold: float | None,
        plane: tuple[float, float, float, float] | None,
    ) -> None:
        from dataclasses import replace

        from neurospatial.environment._fastplotlib_compat import (
            configure_volume_material,
        )

        grid = uniform_grid(env)
        assert grid is not None
        self.grid: UniformGrid = grid
        self.scale = scale
        chosen = (scale.vmin + scale.vmax) / 2 if threshold is None else threshold
        if not scale.vmin <= chosen <= scale.vmax:
            raise ValueError(
                "volume_threshold must lie between fixed vmin and vmax "
                f"({scale.vmin}, {scale.vmax}), in field data units."
            )
        self.graphic = subplot.add_image_volume(
            self.grid.encode(values),
            mode=mode,
            vmin=scale.vmin,
            vmax=scale.vmax,
            cmap=scale.cmap if isinstance(scale.cmap, str) else "viridis",
            # Both are necessary: native linear filtering would mix unknown cells
            # into observations or create artificial intermediate iso intensities.
            interpolation="nearest",
            cmap_interpolation="nearest",
            threshold=chosen,
            plane=self.grid.middle_plane if plane is None else tuple(plane),
        )
        self.graphic.world_object.local.matrix = self.grid.matrix
        # Public pygfx materials permit an exact lookup table. The missing entry
        # remains transparent regardless of the 2D nan_color preference.
        texture = replace(scale, nan_color=None).texture
        configure_volume_material(self.graphic, texture)

    def update(self, values: NDArray[np.float64]) -> None:
        self.graphic.data[:] = self.grid.encode(values)
