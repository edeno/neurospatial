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
        Adjacent valid intervals are checked together at an on-wall sample;
        duplicate stationary samples and motion along a wall retain the known
        approach side. Joints inside an interval restrict the possible sides;
        filtered intervals break this continuity.
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
            owner = int(env.bin_at(start[None, :])[0])
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
