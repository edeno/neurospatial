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
    from neurospatial.layout._grid_frame import FramedGrid
    from neurospatial.layout._locators import RectilinearLocator


def _source_grid(
    frame: FramedGrid, positions: NDArray[np.float64]
) -> tuple[RectilinearLocator, NDArray[np.float64]]:
    """Recover source edge planes only within bounded projection roundoff."""
    from neurospatial.layout._locators import RectilinearLocator

    grid = frame.source
    original = frame._project_to_planes(
        positions, grid.edges, operation="Linear allocation"
    )
    mapping = np.full(grid.active_mask.size, -1, dtype=np.intp)
    mapping[grid.active_mask.ravel()] = np.arange(np.count_nonzero(grid.active_mask))
    return RectilinearLocator(
        grid.edges, grid.shape, grid.active_mask, mapping
    ), original


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
    if env._binning.grid_frame is not None:
        return True
    if env.n_dims != 2:
        from neurospatial.layout._composition import composition_view

        if composition_view(env._binning.locator) is not None:
            from neurospatial.layout._combine_geometry import _boxes

            try:
                _boxes(env._binning.locator)
            except ValueError as error:
                raise NotImplementedError(
                    "Linear allocation needs exact component cells; use time_allocation='start'."
                ) from error
            return True
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
    if env.n_dims != 2:
        return _retained_box_intervals(env, times, positions, valid_mask)

    if positions.shape != (len(times), 2) or valid_mask.shape != (
        max(len(times) - 1, 0),
    ):
        raise ValueError(
            "polygon allocation needs (n_samples, 2) positions and one validity per interval"
        )
    cells = materialize_cells(env._binning)
    arena = env.arena
    frame = env._binning.grid_frame
    source_locator = None
    domain = arena.polygonal_domain() if arena.domain_known else None
    walls = arena.walls
    if frame is not None:
        from neurospatial.environment._arena import _wall_source_frame
        from neurospatial.layout._polygon import cells_from_locator

        source_locator, positions = _source_grid(frame, positions)
        cells = cells_from_locator(source_locator)
        source = _wall_source_frame(
            env._binning.locator, arena, env._binning.bins.centers
        )
        if source is not None and all(
            shapely.equals(left, right)
            for left, right in zip(cells, source[0], strict=True)
        ):
            # Cells prove that the shared wall frame uses these source axes.
            # Retain exact stored walls/domain instead of undoing rounded world vertices.
            arena = source[-1][-1].base
            domain = arena.polygonal_domain() if arena.domain_known else None
            walls = arena.walls
        else:
            domain = (
                None if domain is None else shapely.transform(domain, frame.to_source)
            )
            walls = tuple(shapely.transform(wall, frame.to_source) for wall in walls)

    def locate(queries):
        if source_locator is None:
            return env.bin_at(queries)
        owners = source_locator.locate(queries)
        if domain is not None:
            owners[~shapely.covers(domain, shapely.points(queries))] = -1
        return owners

    effective_cells = tuple(
        cell if domain is None else cell.intersection(domain) for cell in cells
    )
    tree = STRtree(effective_cells)
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
            owner = int(locate(start[None, :])[0])
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
        queries = start + midpoints[:, None] * delta
        owners = locate(queries)
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


def _retained_box_intervals(
    env: TrajectoryBinningConsumer,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
) -> tuple[RetainedLinearInterval, ...]:
    """Partition N-D combined box trajectories at cell and domain planes."""
    from neurospatial.arena import UnknownArena, _CombinedArena, _TransformedArena
    from neurospatial.layout._combine_geometry import _arena_pieces, _boxes

    frame = env._binning.grid_frame
    if frame is None:
        boxes = _boxes(env._binning.locator)
    else:
        source_locator, positions = _source_grid(frame, positions)
        boxes = _boxes(source_locator)

    def domain_boxes(arena):
        if isinstance(arena, UnknownArena):
            return []
        if isinstance(arena, _CombinedArena):
            return [box for child in arena.children for box in domain_boxes(child)]
        if isinstance(arena, _TransformedArena):
            matrix = arena.forward
            return [
                box @ matrix[:-1, :-1].T + matrix[:-1, -1]
                for box in domain_boxes(arena.base)
            ]
        pieces = _arena_pieces(arena)
        if any(piece.kind != "box" for piece in pieces):
            raise NotImplementedError(
                "Linear N-D allocation requires rectangular component domains; use start allocation."
            )
        return [piece.data for piece in pieces]

    domains = domain_boxes(env.arena)
    boxes.extend(
        domains if frame is None else [frame.to_source(box) for box in domains]
    )
    normals, offsets = [], []
    for vertices in boxes:
        dims = env.n_dims
        edges = np.column_stack(
            [vertices[1 << (dims - axis - 1)] - vertices[0] for axis in range(dims)]
        )
        inverse = np.linalg.inv(edges)
        normals.extend([*inverse, *inverse])
        origin = inverse @ vertices[0]
        offsets.extend([*origin, *(origin + 1)])
    normal_array = np.asarray(normals, dtype=np.float64).reshape(-1, env.n_dims)
    offset_array = np.asarray(offsets, dtype=np.float64)
    intervals = []
    for sample in np.flatnonzero(valid_mask):
        start, stop = positions[sample], positions[sample + 1]
        if not np.all(np.isfinite([start, stop])):
            continue
        slopes = normal_array @ (stop - start)
        moving = slopes != 0
        fractions = (offset_array[moving] - normal_array[moving] @ start) / slopes[
            moving
        ]
        fractions = np.unique(
            np.concatenate(([0.0, 1.0], fractions[(fractions > 0) & (fractions < 1)]))
        )
        midpoints = start + ((fractions[:-1] + fractions[1:]) / 2)[:, None] * (
            stop - start
        )
        if frame is None:
            bins = env.bin_at(midpoints)
        else:
            bins = source_locator.locate(midpoints)
            if env.arena.domain_known:
                bins[~env.arena.contains(frame.to_world(midpoints))] = -1
        for left, right, index in zip(fractions[:-1], fractions[1:], bins, strict=True):
            if index >= 0 and right > left:
                intervals.append(
                    RetainedLinearInterval(
                        int(sample), float(left), float(right), int(index)
                    )
                )
    return tuple(intervals)
