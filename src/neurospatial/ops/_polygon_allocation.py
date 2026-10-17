"""Partition straight trajectory segments on exact planar cell support."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from typing import TYPE_CHECKING

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import LineString
from shapely.strtree import STRtree

from neurospatial.geometry._arena import (
    Arena,
    MaskArena,
    UnknownArena,
    _CombinedArena,
    _TransformedArena,
)
from neurospatial.geometry._binning import Binning
from neurospatial.geometry._combine_geometry import _arena_pieces, _boxes
from neurospatial.geometry._composition import composition_view
from neurospatial.geometry._locators import (
    CallIndexes,
    LookupIndexes,
    RectilinearLocator,
)
from neurospatial.geometry._lookup import locate
from neurospatial.geometry._polygon import (
    PolygonCells,
    cells_from_locator,
    materialize_cells,
)
from neurospatial.geometry._wall_frame import _wall_source_frame
from neurospatial.ops._linear_allocation import crossing_blocks, ray_grid_pieces

if TYPE_CHECKING:
    from neurospatial.geometry._grid_frame import FramedGrid


def source_grid(
    frame: FramedGrid, positions: NDArray[np.float64]
) -> tuple[RectilinearLocator, NDArray[np.float64]]:
    """Recover source edge planes only within bounded projection roundoff."""
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


@dataclass(frozen=True)
class _WallArrival:
    """Known approach and possible travel sides at the last wall contact."""

    previous: NDArray[np.float64]
    sides: frozenset[int] | None


def _wall_sectors(
    previous: NDArray[np.float64] | None,
    contact: NDArray[np.float64],
    following: NDArray[np.float64],
    sides: frozenset[int] | None,
    wall_segments: NDArray[np.float64],
    wall_tree: STRtree,
) -> frozenset[int] | None:
    """Transport possible left/right travel sides through one exact wall contact.

    A path along a wall may occupy either adjacent open sector until observed
    entry constrains it. Carry that constraint through subsequent wall motion;
    an isolated tip has one open sector and permits rounding the endpoint.
    """
    incident = wall_tree.query(shapely.Point(contact), predicate="intersects")
    center = tuple(Fraction(float(value)) for value in contact)

    def direction(point: NDArray[np.float64]) -> tuple[Fraction, Fraction]:
        x, y = (Fraction(float(value)) for value in point)
        return x - center[0], y - center[1]

    def compare(a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]) -> int:
        half_a = int(a[1] < 0 or (a[1] == 0 and a[0] < 0))
        half_b = int(b[1] < 0 or (b[1] == 0 and b[0] < 0))
        if half_a != half_b:
            return half_a - half_b
        cross = a[0] * b[1] - a[1] * b[0]
        return -1 if cross > 0 else int(cross < 0)

    rays = sorted(
        {
            direction(endpoint)
            for endpoint in wall_segments[incident].reshape(-1, 2)
            if not np.array_equal(endpoint, contact)
        },
        key=cmp_to_key(compare),
    )
    rays = [ray for i, ray in enumerate(rays) if i == 0 or compare(ray, rays[i - 1])]
    if not rays:
        return None

    def sectors(
        point: NDArray[np.float64],
        allowed_sides: frozenset[int] | None = None,
        *,
        incoming: bool = False,
    ) -> set[int]:
        vector = direction(point)
        for index, ray in enumerate(rays):
            order = compare(vector, ray)
            if order == 0:
                if allowed_sides is None:
                    return {(index - 1) % len(rays), index}
                # Incoming direction points backwards along the path, so left
                # of travel is clockwise of that ray. Outgoing is the reverse.
                return {
                    (index - 1) % len(rays) if (side > 0) == incoming else index
                    for side in allowed_sides
                }
            if order < 0:
                return {(index - 1) % len(rays)}
        return {len(rays) - 1}

    possible = (
        set(range(len(rays)))
        if previous is None
        else sectors(previous, sides, incoming=True)
    )
    possible &= sectors(following)
    if not possible:
        return frozenset()  # No continuous route through this wall contact.
    vector = direction(following)
    for index, ray in enumerate(rays):
        if compare(vector, ray) == 0:
            return frozenset(
                side
                for side, sector in ((1, index), (-1, (index - 1) % len(rays)))
                if sector in possible
            )
    return None  # The outgoing path is strictly inside one open sector.


def _advance_wall_contact(
    start: NDArray[np.float64],
    stop: NDArray[np.float64],
    arrival: _WallArrival | None,
    wall_segments: NDArray[np.float64],
    wall_tree: STRtree,
) -> tuple[bool, _WallArrival | None]:
    """Check a moving interval, including unsampled joints along a wall run."""
    segment = LineString([start, stop])
    if _crosses_wall(segment, wall_segments, wall_tree):
        return True, None
    contacts = []
    for index in wall_tree.query(segment, predicate="intersects"):
        contacts.extend(
            shapely.get_coordinates(segment.intersection(wall_tree.geometries[index]))
        )
    if not contacts:
        return False, None
    points = np.unique(np.asarray(contacts), axis=0)
    axis = int(np.argmax(np.abs(stop - start)))
    fractions = (points[:, axis] - start[axis]) / (stop[axis] - start[axis])
    points = points[np.argsort(fractions)]
    previous: NDArray[np.float64] | None = start
    sides: frozenset[int] | None = None
    for contact in points:
        if np.array_equal(contact, start):
            previous = None if arrival is None else arrival.previous
            sides = None if arrival is None else arrival.sides
        if np.array_equal(contact, stop):
            assert previous is not None  # A moving interval has an approach.
            return False, _WallArrival(previous, sides)
        outgoing = _wall_sectors(
            previous, contact, stop, sides, wall_segments, wall_tree
        )
        if outgoing == frozenset():
            return True, None
        previous, sides = contact, outgoing
    return False, None


def uses_polygon_allocation(binning: Binning, arena: Arena) -> bool:
    """Use exact planar cells, retaining the native allocator for matching N-D masks."""
    if binning.grid_frame is not None:
        return True
    n_dims = binning.bins.n_dims
    if n_dims != 2 and composition_view(binning.locator) is not None:
        try:
            _boxes(binning.locator)
        except ValueError as error:
            raise NotImplementedError(
                "Linear allocation needs exact component cells; use time_allocation='start'."
            ) from error
        return True
    polygon_cells = isinstance(binning.shapes, PolygonCells)
    planar_cells = n_dims == 2 and binning.bins.measure_dim == 2
    if not arena.domain_known:
        return polygon_cells or planar_cells
    if isinstance(arena, MaskArena) and not polygon_cells and not planar_cells:
        grid = binning.rectilinear
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
    binning: Binning,
    arena: Arena,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    indexes: LookupIndexes | None = None,
) -> tuple[RetainedLinearInterval, ...]:
    """Split observation-valid intervals by exact cells and known Arena support.

    Parameters
    ----------
    binning : Binning
        Published Cartesian bins with exact planar cells.
    arena : Arena
        Continuous domain and walls of the same environment.
    times : ndarray, shape (n_samples,)
        Validated timestamps, used to check interval-mask length.
    positions : ndarray, shape (n_samples, 2)
        Segment endpoints. Intervals with a non-finite endpoint are excluded.
    valid_mask : ndarray of bool, shape (n_samples - 1,)
        Observation validity (gaps, speed and recording windows), independent
        of whether the start point belongs to a retained bin.
    indexes : LookupIndexes, optional
        Keeps the locator's search structures, usually the environment's
        analysis cache. None keeps them for this call only.

    Returns
    -------
    tuple of RetainedLinearInterval
        Positive parameter spans in sample order. Fractions refer to the full
        original interval; dropped support is never renormalized. Midpoint
        lookup resolves shared-boundary ownership with the environment's bin
        lookup (``Environment.bin_at``).

    Raises
    ------
    ValueError
        If an observation-valid segment crosses an explicit physical wall.
        Adjacent valid intervals are checked together at an on-wall sample;
        duplicate stationary samples and motion along a wall retain the known
        approach side. Joints inside an interval restrict the possible sides;
        filtered intervals break this continuity.
        Wall endpoint contact and collinear boundary motion remain permitted.
    """
    indexes = CallIndexes() if indexes is None else indexes
    if binning.bins.n_dims != 2:
        return _retained_box_intervals(binning, arena, positions, valid_mask, indexes)

    if positions.shape != (len(times), 2) or valid_mask.shape != (
        max(len(times) - 1, 0),
    ):
        raise ValueError(
            "polygon allocation needs (n_samples, 2) positions and one validity per interval"
        )
    cells = materialize_cells(binning)
    frame = binning.grid_frame
    source_locator = None
    domain = arena.polygonal_domain() if arena.domain_known else None
    walls = arena.walls
    if frame is not None:
        source_locator, positions = source_grid(frame, positions)
        cells = cells_from_locator(source_locator)
        source = _wall_source_frame(binning.locator, arena, binning.bins.centers)
        if source is not None and all(
            shapely.equals(left, right)
            for left, right in zip(cells, source[0], strict=True)
        ):
            # Cells prove that the shared wall frame uses these source axes.
            # Retain exact stored walls/domain instead of undoing rounded world vertices.
            source_arena = source[-1][-1].base
            domain = (
                source_arena.polygonal_domain() if source_arena.domain_known else None
            )
            walls = source_arena.walls
        else:
            domain = (
                None if domain is None else shapely.transform(domain, frame.to_source)
            )
            walls = tuple(shapely.transform(wall, frame.to_source) for wall in walls)

    def owner_of(queries):
        if source_locator is None:
            return locate(binning, arena, queries, indexes)
        owners = source_locator.locate(queries, indexes)
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
    arrival = None
    previous_index = -2
    for index in np.flatnonzero(valid_mask):
        if index != previous_index + 1:
            arrival = None
        previous_index = index
        start = positions[index]
        stop = positions[index + 1]
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(stop)):
            arrival = None
            continue
        delta = stop - start
        if np.all(delta == 0):
            owner = int(owner_of(start[None, :])[0])
            if owner >= 0:
                retained.append(RetainedLinearInterval(int(index), 0.0, 1.0, owner))
            continue
        segment = LineString([start, stop])
        if len(wall_segments):
            crosses, arrival = _advance_wall_contact(
                start, stop, arrival, wall_segments, wall_tree
            )
            if crosses:
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
        owners = owner_of(queries)
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


def polygon_linear_occupancy(
    binning: Binning,
    arena: Arena,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    *,
    return_seconds: bool,
    indexes: LookupIndexes | None = None,
) -> tuple[tuple[RetainedLinearInterval, ...], NDArray[np.float64]]:
    """Split valid intervals on exact cells and add up each bin's share.

    Parameters
    ----------
    binning : Binning
        Published Cartesian bins with exact cells, as for
        :func:`retained_linear_intervals`.
    arena : Arena
        Continuous domain and walls of the same environment.
    times : ndarray of float64, shape (n_samples,)
        Validated timestamps in seconds.
    positions : ndarray of float64, shape (n_samples, n_dims)
        Segment endpoints, which may lie outside the environment.
    valid_mask : ndarray of bool, shape (n_samples - 1,)
        Observation validity of each interval, independent of geometry.
    return_seconds : bool
        If True, each piece adds its share of the interval's duration. If
        False, each interval adds one count split by path length. Framed N-D
        grids normalize that count over the full source grid before inactive
        cells drop their share, as native N-D linear allocation does.
    indexes : LookupIndexes, optional
        Keeps the locator's search structures, usually the environment's
        analysis cache. None keeps them for this call only.

    Returns
    -------
    pieces : tuple of RetainedLinearInterval
        The retained pieces, in sample order.
    occupancy : ndarray of float64, shape (n_bins,)
        Seconds (or proportional counts) in each bin. Dropped support is
        never renormalized.
    """
    dt = np.diff(times)
    normalizers = np.ones(len(dt))
    frame = binning.grid_frame
    if frame is not None and binning.bins.n_dims != 2 and not return_seconds:
        _, source_positions = source_grid(frame, positions)
        samples = np.flatnonzero(valid_mask)
        normalizers[samples] = 0.0
        starts, stops = source_positions[samples], source_positions[samples + 1]
        for block in crossing_blocks(starts, stops, frame.source.edges):
            interval, _, fraction = ray_grid_pieces(
                starts[block],
                stops[block],
                frame.source.edges,
                frame.source.shape,
                np.ones(block.stop - block.start),
            )
            np.add.at(normalizers, samples[block][interval], fraction)
    pieces = retained_linear_intervals(
        binning, arena, times, positions, valid_mask, indexes
    )
    occupancy = np.zeros(binning.bins.n_bins, dtype=np.float64)
    for piece in pieces:
        weight = dt[piece.sample_index] if return_seconds else 1.0
        if normalizers[piece.sample_index] <= 0:
            continue
        occupancy[piece.bin_index] += (
            (piece.stop_fraction - piece.start_fraction)
            * weight
            / normalizers[piece.sample_index]
        )
    return pieces, occupancy


def _retained_box_intervals(
    binning: Binning,
    arena: Arena,
    positions: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    indexes: LookupIndexes,
) -> tuple[RetainedLinearInterval, ...]:
    """Partition N-D combined box trajectories at cell and domain planes."""
    frame = binning.grid_frame
    if frame is None:
        boxes = _boxes(binning.locator)
    else:
        source_locator, positions = source_grid(frame, positions)
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

    domains = domain_boxes(arena)
    boxes.extend(
        domains if frame is None else [frame.to_source(box) for box in domains]
    )
    normals, offsets = [], []
    for vertices in boxes:
        dims = binning.bins.n_dims
        edges = np.column_stack(
            [vertices[1 << (dims - axis - 1)] - vertices[0] for axis in range(dims)]
        )
        inverse = np.linalg.inv(edges)
        normals.extend([*inverse, *inverse])
        origin = inverse @ vertices[0]
        offsets.extend([*origin, *(origin + 1)])
    normal_array = np.asarray(normals, dtype=np.float64).reshape(
        -1, binning.bins.n_dims
    )
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
            bins = locate(binning, arena, midpoints, indexes)
        else:
            bins = source_locator.locate(midpoints, indexes)
            if arena.domain_known:
                bins[~arena.contains(frame.to_world(midpoints))] = -1
        for left, right, index in zip(fractions[:-1], fractions[1:], bins, strict=True):
            if index >= 0 and right > left:
                intervals.append(
                    RetainedLinearInterval(
                        int(sample), float(left), float(right), int(index)
                    )
                )
    return tuple(intervals)
