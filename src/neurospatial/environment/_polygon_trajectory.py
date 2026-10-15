"""Partition straight trajectory segments on exact planar cell support."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import LineString
from shapely.strtree import STRtree

from neurospatial.arena import MaskArena
from neurospatial.layout._polygon import PolygonCells, materialize_cells

if TYPE_CHECKING:
    from neurospatial.environment._protocols import TrajectoryBinningConsumer


@dataclass(frozen=True)
class RetainedLinearInterval:
    """A half-open fraction of one observation interval assigned to one bin."""

    sample_index: int
    start_fraction: float
    stop_fraction: float
    bin_index: int


def _crosses_wall(
    segment: LineString, wall_segments: NDArray[np.float64], wall_tree: STRtree
) -> bool:
    """Test incident wall rays on both sides of an interior trajectory contact."""
    origin, stop = (
        tuple(Fraction(float(value)) for value in coordinate)
        for coordinate in (segment.coords[0], segment.coords[-1])
    )
    dx, dy = stop[0] - origin[0], stop[1] - origin[1]
    for index in wall_tree.query(segment, predicate="intersects"):
        contact = segment.intersection(wall_tree.geometries[index])
        if not isinstance(contact, shapely.Point) or segment.boundary.covers(contact):
            continue
        incident = np.union1d(wall_tree.query(contact, predicate="intersects"), [index])
        positive = negative = False
        for endpoint in wall_segments[incident].reshape(-1, 2):
            # Exact signs for stored float coordinates avoid an angle or
            # distance tolerance that could open a narrow physical crossing.
            x, y = (Fraction(float(value)) for value in endpoint)
            side = dx * (y - origin[1]) - dy * (x - origin[0])
            positive |= side > 0
            negative |= side < 0
        if positive and negative:
            return True
    return False


def uses_polygon_allocation(env: TrajectoryBinningConsumer) -> bool:
    """Use exact planar cells, retaining the native allocator for matching N-D masks."""
    polygon_cells = isinstance(env._binning.shapes, PolygonCells)
    planar_cells = env.n_dims == 2 and env._binning.bins.measure_dim == 2
    arena = env.arena
    if not arena.domain_known:
        return polygon_cells or planar_cells
    if isinstance(arena, MaskArena) and not polygon_cells and not planar_cells:
        grid = env._binning.rectilinear
        if (
            grid is not None
            and len(arena.edges) == len(grid.edges)
            and arena.active_mask.shape == grid.active_mask.shape
            and all(
                np.array_equal(left, right)
                for left, right in zip(arena.edges, grid.edges, strict=True)
            )
            and np.all(~grid.active_mask | arena.active_mask)
        ):
            # Every active cell lies in the Arena. This includes subsets that
            # keep their parent's larger mask without changing its axes.
            return False
    if arena.polygonal_domain() is None:
        support = (
            "a restricted MaskArena"
            if isinstance(arena, MaskArena)
            else "this known Arena"
        )
        raise NotImplementedError(
            f"Linear occupancy cannot subdivide {support} without exact planar "
            "polygon support. Use time_allocation='start' or construct a grid "
            "whose active cells lie wholly within a matching Arena mask."
        )
    return True


def retained_linear_intervals(
    env: TrajectoryBinningConsumer,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
) -> tuple[RetainedLinearInterval, ...]:
    """Split observation-valid intervals by exact cells and known Arena support.

    Parameters
    ----------
    env : Environment
        Cartesian environment publishing exact planar cells.
    times : ndarray, shape (n_samples,)
        Validated timestamps, used to check interval-mask length.
    positions : ndarray, shape (n_samples, 2)
        Segment endpoints. Intervals with a non-finite endpoint are excluded.
    valid_mask : ndarray of bool, shape (n_samples - 1,)
        Observation validity (gaps, speed and recording windows), independent
        of whether the start point belongs to a retained bin.

    Returns
    -------
    tuple of RetainedLinearInterval
        Positive parameter spans in sample order. Fractions refer to the full
        original interval; dropped support is never renormalized. Midpoint
        lookup resolves shared-boundary ownership using ``env.bin_at``.

    Raises
    ------
    ValueError
        If an observation-valid segment crosses an explicit physical wall.
        Wall endpoint contact and collinear boundary motion remain permitted.
    """
    if positions.shape != (len(times), 2) or valid_mask.shape != (
        max(len(times) - 1, 0),
    ):
        raise ValueError(
            "polygon allocation needs (n_samples, 2) positions and one validity per interval"
        )
    cells = materialize_cells(env._binning)
    domain = env.arena.polygonal_domain() if env.arena.domain_known else None
    effective_cells = tuple(
        cell if domain is None else cell.intersection(domain) for cell in cells
    )
    tree = STRtree(effective_cells)
    walls = env.arena.walls
    wall_segments = (
        np.concatenate(
            [
                np.stack((coordinates[:-1], coordinates[1:]), axis=1)
                for wall in walls
                for coordinates in (np.asarray(wall.coords),)
            ]
        )
        if walls
        else np.empty((0, 2, 2), dtype=np.float64)
    )
    wall_segments = wall_segments[
        np.any(wall_segments[:, 0] != wall_segments[:, 1], axis=1)
    ]
    wall_tree = STRtree(shapely.linestrings(wall_segments))
    retained = []
    for index in np.flatnonzero(valid_mask):
        start = positions[index]
        stop = positions[index + 1]
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(stop)):
            continue
        delta = stop - start
        if np.all(delta == 0):
            owner = int(env.bin_at(start[None, :])[0])
            if owner >= 0:
                retained.append(RetainedLinearInterval(int(index), 0.0, 1.0, owner))
            continue
        segment = LineString([start, stop])
        if len(wall_segments) and _crosses_wall(segment, wall_segments, wall_tree):
            raise ValueError(
                f"Trajectory interval {index} crosses a physical wall; "
                "provide observations that resolve the route around the wall "
                "or exclude the interval with the observation filters."
            )
        # One varying coordinate gives the segment parameter without squaring
        # its length. Every intersection endpoint is a possible ownership change.
        axis = int(np.argmax(np.abs(delta)))
        breaks = [0.0, 1.0]
        for cell_index in tree.query(segment, predicate="intersects"):
            coordinates = shapely.get_coordinates(
                segment.intersection(effective_cells[cell_index])
            )
            breaks.extend(((coordinates[:, axis] - start[axis]) / delta[axis]).tolist())
        fractions = np.unique(np.clip(breaks, 0.0, 1.0))
        midpoints = (fractions[:-1] + fractions[1:]) / 2
        owners = env.bin_at(start + midpoints[:, None] * delta)
        for left, right, owner in zip(
            fractions[:-1], fractions[1:], owners, strict=True
        ):
            if owner >= 0:
                retained.append(
                    RetainedLinearInterval(
                        int(index), float(left), float(right), int(owner)
                    )
                )
    return tuple(retained)
