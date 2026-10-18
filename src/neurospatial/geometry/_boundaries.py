"""Exact planar wall evidence shared by border and visibility consumers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.ops import unary_union

from neurospatial._exceptions import CartesianOnlyError, _format_error
from neurospatial.geometry._arena import (
    _CombinedArena,
    _feature_cap,
    _TransformedArena,
    mapped_geometry,
)
from neurospatial.geometry._composition import CombinedLocator
from neurospatial.geometry._locators import (
    CallIndexes,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.geometry._lookup import locate
from neurospatial.geometry._polygon import materialize_cells
from neurospatial.geometry._polygon_cells import (
    ClippedLocator,
    PolygonCells,
    _polygon_coordinate_errors,
)
from neurospatial.geometry._wall_frame import wall_source_frame

if TYPE_CHECKING:
    from neurospatial.geometry._arena import Arena
    from neurospatial.geometry._binning import Binning
    from neurospatial.geometry._locators import Locator, LookupIndexes

BoundarySource = Literal["physical", "analysis"]
Interior = Callable[[NDArray[np.float64]], NDArray[np.bool_]]
# Names the analysis in a lookup error; its points are 2-D by construction.
_ANALYSIS_WALLS = "the boundary_source='analysis' wall search"


def boundary_geometry(
    binning: Binning, arena: Arena, source: BoundarySource
) -> shapely.Geometry:
    """Return declared physical walls or explicitly selected analysis boundaries.

    Physical mode uses exact planar footprint rings and explicit internal walls.
    It never infers physical walls from bins, sampling masks or unknown domains.
    Analysis mode uses the exact union of retained planar analysis cells.

    Raises
    ------
    CartesianOnlyError
        If the arena holds polar ``(distance, angle)`` coordinates.
    NotImplementedError
        If the arena is not two-dimensional, or its walls or cells are not
        exact planar geometry.
    """
    if source not in ("physical", "analysis"):
        raise ValueError("boundary_source must be 'physical' or 'analysis'.")
    if arena.coordinate_kind != "cartesian":
        raise CartesianOnlyError(
            f"Wall geometry (boundary_source={source!r})",
            why="Walls and cell edges are lines in the Cartesian (x, y) plane.",
            fix="compute wall-based measures on the allocentric environment "
            "the positions were recorded in",
        )
    if arena.n_dims != 2:
        raise NotImplementedError(
            _format_error(
                f"Wall geometry needs a 2-D environment; this one is {arena.n_dims}-D.",
                why="Why: walls and cell edges are lines in a plane.",
                fix="compute wall-based measures on a 2-D environment",
            )
        )
    if source == "analysis":
        return _analysis_boundary(binning, arena)
    return physical_walls(arena)


def _analysis_boundary(binning: Binning, arena: Arena) -> shapely.Geometry:
    """Return retained cell edges restricted by the supplied planar domain."""

    try:
        domain = unary_union(materialize_cells(binning))
        restriction = arena.polygonal_domain() if arena.domain_known else None
        if restriction is not None:
            domain = domain.intersection(restriction)
        return unary_union([domain.boundary, *arena.walls])
    except ValueError as exc:
        raise NotImplementedError(
            "Analysis-boundary geometry needs exact planar cells; supply supported grid or polygon cells."
        ) from exc


def _resolve_wall_contacts(
    locator: Locator,
    arena: Arena,
    cells: Sequence[shapely.Geometry],
    walls: shapely.Geometry,
    hits: NDArray[np.float64],
    origins: NDArray[np.float64],
    bins: NDArray[np.intp],
    indexes: LookupIndexes,
) -> NDArray[np.intp]:
    """Resolve rounded boundary hits using cells that actually touch the wall.

    A single representable step can remain outside a slanted or curved
    polygon, and clipped cell edges only nearly coincide with such a wall. For
    a failed hit, a retained cell reaches the wall there when the hit lies
    within the cell's bounded coordinate uncertainty of both the cell and the
    wall. Intersect the approach segment ending at the hit with that cell, and
    query an interior point of the terminal piece. A missing strip between a
    cell and the wall is wider than that uncertainty and is never bridged by
    an inward search.
    """

    def same_parent(target: NDArray[np.float64], query: NDArray[np.float64]) -> bool:
        parent = locator
        while True:
            if isinstance(parent, RectilinearLocator):
                # Optimized grid subsets retain inactive cells in their grid
                # mask, rather than a SubsetLocator wrapper. Preserve that
                # full-grid owner even when its active-bin index is -1.
                return all(
                    np.searchsorted(edge, target[0, axis], side="right")
                    == np.searchsorted(edge, query[0, axis], side="right")
                    for axis, edge in enumerate(parent.edges)
                )
            original = parent.locate(target, indexes)[0]
            if original >= 0 and parent.locate(query, indexes)[0] != original:
                return False
            if not isinstance(parent, (SubsetLocator, ClippedLocator)):
                return True
            parent = parent.base

    missing = np.flatnonzero((bins < 0) & np.all(np.isfinite(hits), axis=1))
    if not len(missing):
        return bins
    polygons = np.asarray(cells, dtype=object)
    # Distance to the hit, not an overlay of cells with the wall: near-
    # coincident clipped edges intersect a curved wall only in scattered
    # rounded fragments, which can lie far from the hit.
    budgets = np.array(
        [
            np.hypot.reduce(
                np.minimum(_polygon_coordinate_errors(cell), _feature_cap(arena))
            )
            for cell in cells
        ]
    )
    for row in missing:
        point = shapely.Point(hits[row])
        distances = shapely.distance(polygons, point)
        candidates = np.flatnonzero(
            (distances <= budgets) & (walls.distance(point) <= budgets)
        )
        segment = shapely.LineString([origins[row], hits[row]])
        for index in candidates:
            pieces = shapely.get_parts(segment.intersection(cells[index]))
            for piece in pieces:
                if piece.geom_type != "LineString" or piece.length == 0:
                    continue
                # Only the terminal approach interval establishes ownership.
                if piece.distance(point) > budgets[index]:
                    continue
                middle = piece.interpolate(0.5, normalized=True)
                query = np.array([[middle.x, middle.y]])
                if arena.domain_known and not arena.contains(query)[0]:
                    continue
                original = np.nextafter(hits[row : row + 1], origins[row : row + 1])
                if locator.locate(query, indexes)[0] == index and same_parent(
                    original, query
                ):
                    bins[row] = index
                    break
            if bins[row] >= 0:
                break

    return bins


def wall_hit_bins(
    binning: Binning,
    arena: Arena,
    hits: NDArray[np.float64],
    origins: NDArray[np.float64],
    bins: NDArray[np.intp],
    source: BoundarySource,
) -> NDArray[np.intp]:
    """Assign wall hits to observer-side bins using owned geometry.

    ``hits`` and ``origins`` have shape ``(n_hits, 2)``. ``bins`` contains
    the caller's initial lookup of the observer-side limit, shape ``(n_hits,)``.
    Shared affine frames are resolved in their stored cell-owner coordinates.
    """

    def source_locator(
        locator: Locator, arena: Arena, centers: NDArray[np.float64]
    ) -> Locator:
        if isinstance(locator, TransformedLocator):
            if isinstance(arena, _TransformedArena) and np.array_equal(
                locator.inverse, arena.inverse
            ):
                original, _error = arena._project(centers)
                return source_locator(locator.base, arena.base, original)
            if source == "physical":
                raise NotImplementedError(
                    "Physical wall-bin assignment needs a shared source frame; retain the arena's transform with the bins, or explicitly choose boundary_source='analysis'."
                )
        if isinstance(locator, SubsetLocator):
            return SubsetLocator(
                source_locator(locator.base, arena, centers), locator.parent_to_child
            )
        if isinstance(locator, ClippedLocator):
            local = wall_source_frame(locator, arena, centers)
            if local is not None:
                return ClippedLocator(
                    source_locator(locator.base, arena, centers),
                    PolygonCells(local[0]),
                    locator.mapping,
                )
        if isinstance(locator, CombinedLocator) and source == "physical":
            # A child with its own affine frame has no one shared owner frame.
            pending = [child.locator for child in locator.children]
            while pending:
                child = pending.pop()
                if isinstance(child, TransformedLocator):
                    raise NotImplementedError(
                        "Physical wall-bin assignment needs a shared source frame for combined components. Combine before a shared transform, or explicitly choose boundary_source='analysis'."
                    )
                if isinstance(child, (SubsetLocator, ClippedLocator)):
                    pending.append(child.base)
                elif isinstance(child, CombinedLocator):
                    pending.extend(part.locator for part in child.children)
        return locator

    locator = source_locator(binning.locator, arena, binning.bins.centers)
    frame = wall_source_frame(binning.locator, arena, binning.bins.centers)
    indexes = CallIndexes()
    if frame is None:
        if source == "physical":
            walls = physical_walls(arena)
        else:
            walls = _analysis_boundary(binning, arena)
        return _resolve_wall_contacts(
            locator,
            arena,
            materialize_cells(binning),
            walls,
            hits,
            origins,
            bins,
            indexes,
        )
    cells, _walls, _centers, _inverse, frames = frame
    query = np.vstack((hits, origins))
    error = None
    for arena in frames:
        query, error = arena._project(query, error)
    base = frames[-1].base
    if source == "physical":
        walls = physical_walls(base)
    else:
        domain = unary_union(cells)
        restriction = base.polygonal_domain() if base.domain_known else None
        if restriction is not None:
            domain = domain.intersection(restriction)
        walls = unary_union([domain.boundary, *base.walls])
    assert error is not None
    points = shapely.points(query[: len(hits)])
    allowance = np.hypot.reduce(
        np.minimum(error[: len(hits)], _feature_cap(base)), axis=1
    )
    close = shapely.distance(points, walls) <= allowance
    nearest = shapely.get_point(shapely.shortest_line(points[close], walls), -1)
    query[np.flatnonzero(close)] = np.column_stack(
        (shapely.get_x(nearest), shapely.get_y(nearest))
    )
    inside = np.nextafter(query[: len(hits)], query[len(hits) :])
    bins = locator.locate(inside, indexes)
    if base.domain_known:
        bins[~base.contains(inside)] = -1
    return _resolve_wall_contacts(
        locator,
        base,
        cells,
        walls,
        query[: len(hits)],
        query[len(hits) :],
        bins,
        indexes,
    )


def _physical_evidence(
    arena: Arena,
) -> tuple[list[shapely.Geometry], list[shapely.Geometry]]:
    """Explicit wall lines and asserted footprint polygons in the arena frame."""

    if isinstance(arena, _CombinedArena):
        lines: list[shapely.Geometry] = []
        footprints: list[shapely.Geometry] = []
        for child in arena.children:
            child_lines, child_footprints = _physical_evidence(child)
            lines += child_lines
            footprints += child_footprints
        return lines, footprints
    if isinstance(arena, _TransformedArena):
        lines, footprints = _physical_evidence(arena.base)
        return (
            [mapped_geometry(part, arena.forward) for part in lines],
            [mapped_geometry(part, arena.forward) for part in footprints],
        )
    lines = list(arena.walls)
    footprints = []
    if arena.physical_footprint_known:
        domain = arena.polygonal_domain()
        if domain is None:
            raise NotImplementedError(
                "Physical footprint has no exact planar polygon boundary; curved tracks require a separate wall representation. Supply explicit polygonal geometry or choose boundary_source='analysis' when supported."
            )
        footprints.append(domain)
    return lines, footprints


def physical_footprint_interior(arena: Arena) -> Interior | None:
    """Closed asserted footprints, or None when only wall lines are known."""
    footprints = _physical_evidence(arena)[1]
    if not footprints:
        return None
    footprint = unary_union(footprints)
    return lambda points: np.asarray(
        shapely.covers(footprint, shapely.points(points)), dtype=np.bool_
    )


def analysis_interior(
    binning: Binning, arena: Arena, indexes: LookupIndexes | None = None
) -> Interior:
    """Retained analysis support, matching analysis-mode observer validity.

    ``indexes`` keeps the locator's search structures across calls.
    """
    return lambda points: np.asarray(
        locate(binning, arena, points, indexes, call=_ANALYSIS_WALLS) >= 0,
        dtype=np.bool_,
    )


def physical_walls(arena: Arena) -> shapely.Geometry:
    """Read asserted planar wall evidence independently of analysis bins."""
    lines, footprints = _physical_evidence(arena)
    parts = lines + [footprint.boundary for footprint in footprints]
    if not parts:
        raise NotImplementedError(
            "Physical walls are unknown: positions and analysis masks do not establish a measured footprint. "
            "Supply measured geometry with Environment.from_polygon(...) or an asserted OpenArena or "
            "MaskArena(physical_footprint_known=True). Wall-based rate maps also need bin support at the "
            "measured walls; attaching an arena does not extend bins or supply missing observations. "
            "Alternatively, explicitly choose boundary_source='analysis' to use retained bin edges as an approximation."
        )
    return unary_union(parts)


def valid_observers(
    binning: Binning,
    arena: Arena,
    positions: NDArray[np.float64],
    source: BoundarySource,
    indexes: LookupIndexes | None = None,
) -> NDArray[np.bool_]:
    """Require finite observers inside the domain justified by the chosen source.

    ``indexes`` keeps the locator's search structures across calls.
    """
    valid = np.asarray(np.all(np.isfinite(positions), axis=1), dtype=np.bool_)
    if source == "analysis":
        valid[valid] = (
            locate(binning, arena, positions[valid], indexes, call=_ANALYSIS_WALLS) >= 0
        )
    elif arena.physical_footprint_known:
        valid[valid] = arena.contains(positions[valid])
    return valid


def _ray_segments(
    origins: NDArray[np.float64], angles: NDArray[np.float64], max_distance: float
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    """Validate rays and return only finite starts and finite search endpoints."""
    if (
        isinstance(max_distance, (bool, np.bool_))
        or not isinstance(max_distance, (int, float, np.integer, np.floating))
        or not np.isfinite(max_distance)
        or max_distance <= 0
    ):
        raise ValueError("max_distance must be finite and positive.")
    if origins.ndim != 2 or origins.shape[1] != 2 or angles.shape != (len(origins),):
        raise ValueError(
            "Ray positions must have shape (n_time, 2) and directions (n_time,)."
        )
    valid = np.asarray(
        np.all(np.isfinite(origins), axis=1) & np.isfinite(angles), dtype=np.bool_
    )
    selected = origins[valid]
    direction = np.column_stack((np.cos(angles[valid]), np.sin(angles[valid])))
    with np.errstate(over="ignore", invalid="ignore"):
        end = selected + max_distance * direction
    if np.any(~np.isfinite(end)) or np.any(np.all(end == selected, axis=1)):
        raise ValueError(
            "Ray length is not representable at these coordinates; rescale or recenter positions."
        )
    return valid, selected, end


def _start_allowance(starts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Eight ULPs of each start coordinate, as a distance, shape (n_segments,)."""
    return np.asarray(np.hypot.reduce(8 * np.spacing(np.abs(starts)), axis=1))


def _start_departures(
    contacts: shapely.Geometry | NDArray[np.object_],
    starts: NDArray[np.float64],
    ends: NDArray[np.float64],
    interior: Interior | None,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Find segments that start on a wall and leave it into the domain.

    A segment starting on a wall meets it at zero distance. That contact is the
    observer's own position, not a viewed wall, when the segment departs from
    the wall into the domain: every start contact is a point (the segment does
    not run along a wall), and the open interval up to the next contact, or to
    the segment end, lies in ``interior``. Testing that interval's midpoint
    decides the side from the wall arrangement alone, without a length scale.
    Zero-thickness walls inside the domain are therefore left on either side.
    ``interior=None`` means only wall lines are known, so any departure counts.
    Start contacts allow the same eight ULPs of the start coordinates as
    terminal cue contacts allow at the cue.

    Parameters
    ----------
    contacts : shapely geometry array, shape (n_segments,)
        Intersection of each segment with the walls.
    starts, ends : ndarray, shape (n_segments, 2)
        Oriented segment endpoints.
    interior : callable or None
        Maps points of shape (n_points, 2) to closed-domain membership.

    Returns
    -------
    departs : ndarray of bool, shape (n_segments,)
        Whether the start contact should be ignored.
    later : ndarray, shape (n_segments, 2)
        First contact after the start contacts, or NaN when there is none.
    """
    n_segments = len(starts)
    departs = np.zeros(n_segments, dtype=bool)
    later = np.full((n_segments, 2), np.nan)
    parts, owners = shapely.get_parts(contacts, return_index=True)
    if not len(parts):
        return departs, later
    origins = shapely.points(starts)
    distance = shapely.distance(parts, origins[owners])
    at_start = distance <= _start_allowance(starts)[owners]
    touches = np.zeros(n_segments, dtype=bool)
    touches[owners[at_start]] = True
    along = np.zeros(n_segments, dtype=bool)
    along[owners[at_start & (shapely.get_dimensions(parts) > 0)]] = True
    rest = np.flatnonzero(~at_start)
    if len(rest):
        ordered = rest[np.lexsort((distance[rest], owners[rest]))]
        first = ordered[np.r_[True, owners[ordered][1:] != owners[ordered][:-1]]]
        nearest = shapely.get_point(
            shapely.shortest_line(origins[owners[first]], parts[first]), -1
        )
        later[owners[first]] = np.column_stack(
            (shapely.get_x(nearest), shapely.get_y(nearest))
        )
    candidates = np.flatnonzero(touches & ~along)
    if len(candidates):
        probe = np.where(
            np.isfinite(later[candidates]), later[candidates], ends[candidates]
        )
        middle = (starts[candidates] + probe) / 2
        departs[candidates] = True if interior is None else interior(middle)
    return departs, later


def _segment_hits(
    walls: shapely.Geometry,
    starts: NDArray[np.float64],
    ends: NDArray[np.float64],
    interior: Interior | None,
) -> NDArray[np.float64]:
    """First viewed wall contact along each oriented finite segment, or NaN.

    A zero-distance contact counts only when the segment points into or along
    the observer's wall; leaving that wall into the domain views the next wall.
    """
    segments = shapely.linestrings(np.stack((starts, ends), axis=1))
    intersections = shapely.intersection(segments, walls)
    nearest = shapely.get_point(
        shapely.shortest_line(shapely.points(starts), intersections), -1
    )
    hits = np.column_stack((shapely.get_x(nearest), shapely.get_y(nearest)))
    with np.errstate(invalid="ignore"):
        rows = np.flatnonzero(
            np.hypot.reduce(hits - starts, axis=1) <= _start_allowance(starts)
        )
    if len(rows):
        departs, later = _start_departures(
            intersections[rows], starts[rows], ends[rows], interior
        )
        hits[rows[departs]] = later[departs]
    return hits


def ray_hits(
    walls: shapely.Geometry,
    origins: NDArray[np.float64],
    angles: NDArray[np.float64],
    max_distance: float,
    interior: Interior | None = None,
) -> NDArray[np.float64]:
    """First exact wall point on each finite forward ray, or NaN when absent.

    Parameters
    ----------
    walls : shapely.Geometry
        Known planar line geometry, including holes and internal walls.
    origins : ndarray, shape (n_rays, 2)
        Ray origins. Invalid observations produce NaN output rows.
    angles : ndarray, shape (n_rays,)
        Directions in radians, zero east and positive counterclockwise.
    max_distance : float
        Finite positive physical search length. A search endpoint is not a hit.
    interior : callable or None, optional
        Domain membership deciding whether a ray starting on a wall leaves it
        into the domain; None treats both sides of every wall as open.
    """
    origins = np.asarray(origins, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    valid, starts, ends = _ray_segments(origins, angles, max_distance)
    result = np.full(origins.shape, np.nan)
    result[valid] = _segment_hits(walls, starts, ends, interior)
    return result


def _wall_frame(
    arena: Arena, query: NDArray[np.float64]
) -> tuple[Arena, NDArray[np.float64], list[_TransformedArena]]:
    """Use stored projection errors for contact in a shared physical-wall frame."""

    frames = []
    error = None
    while isinstance(arena, _TransformedArena):
        frames.append(arena)
        query, error = arena._project(query, error)
        arena = arena.base
    if error is not None:
        walls = physical_walls(arena)
        finite = np.all(np.isfinite(query), axis=1)
        points = shapely.points(query[finite])
        allowance = np.hypot.reduce(
            np.minimum(error[finite], _feature_cap(arena)), axis=1
        )
        close = shapely.distance(points, walls) <= allowance
        nearest = shapely.get_point(shapely.shortest_line(points[close], walls), -1)
        selected = np.flatnonzero(finite)[close]
        query[selected] = np.column_stack(
            (shapely.get_x(nearest), shapely.get_y(nearest))
        )
    return arena, query, frames


def physical_ray_hits(
    arena: Arena,
    origins: NDArray[np.float64],
    angles: NDArray[np.float64],
    max_distance: float,
) -> NDArray[np.float64]:
    """Preserve first contacts through owned affine frames and bounded rounding."""
    origins = np.asarray(origins, dtype=np.float64)
    valid, starts, ends = _ray_segments(
        origins, np.asarray(angles, dtype=np.float64), max_distance
    )
    base, query, frames = _wall_frame(arena, np.vstack((starts, ends)))
    if not np.all(np.isfinite(query)):
        raise ValueError(
            "Ray projection exceeds coordinate precision; recenter or rescale positions."
        )
    hits = _segment_hits(
        physical_walls(base),
        query[: len(starts)],
        query[len(starts) :],
        physical_footprint_interior(base),
    )
    finite = np.all(np.isfinite(hits), axis=1)
    for frame in reversed(frames):
        hits[finite] = hits[finite] @ frame.forward[:-1, :-1].T + frame.forward[:-1, -1]
    result = np.full(origins.shape, np.nan)
    result[valid] = hits
    return result


def physical_clear_segments(
    arena: Arena, observer: NDArray[np.float64], targets: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Keep on-wall cues visible under shared-frame inverse rounding."""
    base, query, _frames = _wall_frame(arena, np.vstack((observer, targets)))
    return clear_segments(
        physical_walls(base), query[0], query[1:], physical_footprint_interior(base)
    )


def clear_segments(
    walls: shapely.Geometry,
    observer: NDArray[np.float64],
    targets: NDArray[np.float64],
    interior: Interior | None = None,
) -> NDArray[np.bool_]:
    """Keep terminal contact within coordinate rounding; earlier contact blocks.

    An observer standing on a wall does not block its own view when the
    segment leaves that wall into ``interior`` (see ``_start_departures``).
    """
    targets = np.asarray(targets, dtype=np.float64)
    valid = np.all(np.isfinite(targets), axis=1) & np.all(np.isfinite(observer))
    result = np.zeros(len(targets), dtype=bool)
    selected = targets[valid]
    same = np.all(selected == observer, axis=1)
    if np.any(~same):
        lines = shapely.linestrings(
            np.stack(
                (np.broadcast_to(observer, selected[~same].shape), selected[~same]),
                axis=1,
            )
        )
        contact = shapely.intersection(lines, walls)
        # Intersection and interpolation can cancel one endpoint coordinate
        # to zero. Allow eight ULPs in the cue's own coordinates and project
        # that rounding box onto the intersected wall's tangent. A coordinate
        # perpendicular to the wall contributes no cross-axis allowance, and
        # distant endpoints cannot widen gaps. Any positive wall overlap blocks.
        coordinates, owners = shapely.get_coordinates(contact, return_index=True)
        terminal = selected[~same]
        allowance = 8 * np.spacing(np.abs(terminal[owners]))
        if len(coordinates):
            parts = shapely.get_parts(walls)
            vertices, part_index = shapely.get_coordinates(parts, return_index=True)
            adjacent = part_index[:-1] == part_index[1:]
            wall_segments = np.stack((vertices[:-1], vertices[1:]), axis=1)[adjacent]
            wall_segments = wall_segments[
                np.any(wall_segments[:, 0] != wall_segments[:, 1], axis=1)
            ]
            nearest = shapely.STRtree(shapely.linestrings(wall_segments)).nearest(
                shapely.points(coordinates)
            )
            tangent = np.abs(np.diff(wall_segments[nearest], axis=1)[:, 0])
            tangent /= np.hypot.reduce(tangent, axis=1)[:, None]
            allowance += tangent * np.sum(tangent * allowance, axis=1)[:, None]
        at_terminal = np.all(
            np.abs(coordinates - terminal[owners]) <= allowance, axis=1
        )
        own_wall = (
            np.hypot.reduce(coordinates - observer, axis=1)
            <= _start_allowance(observer.reshape(1, 2))[0]
        )
        rows = np.unique(owners[own_wall])
        if len(rows):
            departs, _later = _start_departures(
                contact[rows],
                np.broadcast_to(observer, (len(rows), 2)),
                terminal[rows],
                interior,
            )
            departing = np.zeros(len(terminal), dtype=bool)
            departing[rows[departs]] = True
            at_terminal |= departing[owners] & own_wall
        clear = np.ones(len(terminal), dtype=bool)
        np.logical_and.at(clear, owners, at_terminal)
        clear &= shapely.is_empty(contact) | (shapely.get_dimensions(contact) <= 0)
        same[~same] = clear
    result[valid] = same
    return result


def search_distance(walls: shapely.Geometry, positions: NDArray[np.float64]) -> float:
    """Bound all observer-to-wall distances for automatic finite-domain searches."""
    finite = np.asarray(positions)[np.all(np.isfinite(positions), axis=1)]
    coordinates = np.vstack((np.array(walls.bounds).reshape(2, 2), finite))
    with np.errstate(over="ignore", invalid="ignore"):
        extent = float(
            np.hypot.reduce(coordinates.max(axis=0) - coordinates.min(axis=0))
        )
    if not np.isfinite(extent):
        raise ValueError(
            "Wall search extent exceeds coordinate precision; rescale positions."
        )
    return max(extent * 2, 1.0)


def boundary_view(
    binning: Binning,
    arena: Arena,
    positions: NDArray[np.float64],
    gaze_direction: NDArray[np.float64],
    max_distance: float,
    boundary_source: Literal["physical", "analysis"],
    indexes: LookupIndexes,
) -> NDArray[np.float64]:
    """First wall each gaze ray reaches, shape (n_time, 2); NaN for none.

    ``indexes`` keeps the locator's search structures for the computation.
    """

    walls = boundary_geometry(binning, arena, boundary_source)
    origins = positions.copy()
    origins[~valid_observers(binning, arena, positions, boundary_source, indexes)] = (
        np.nan
    )
    if boundary_source == "physical":
        return physical_ray_hits(arena, origins, gaze_direction, max_distance)
    return ray_hits(
        walls,
        origins,
        gaze_direction,
        max_distance,
        analysis_interior(binning, arena, indexes),
    )


def line_of_sight_clear(
    binning: Binning,
    arena: Arena,
    observer: NDArray[np.float64],
    target: NDArray[np.float64],
    n_samples: int = 20,
    *,
    boundary_source: Literal["physical", "analysis"] = "physical",
    indexes: LookupIndexes | None = None,
) -> bool:
    """Check if line of sight between observer and target is clear.

    Parameters
    ----------
    binning : Binning
        The environment's bins, from ``env.binning``.
    arena : Arena
        The environment's continuous domain, from ``env.arena``.
    observer : NDArray[np.float64], shape (2,)
        Observer position.
    target : NDArray[np.float64], shape (2,)
        Target position.
    n_samples : int
        Retained for compatibility; line-of-sight tests use exact intersections.

    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls, or explicitly requested analysis-cell boundaries.
        Physical mode requires exact supported wall geometry; missing analysis
        bins do not occlude a physical ray. Fixed-distance gaze ignores this.
    indexes : LookupIndexes, optional
        Search structures shared by the calls of one computation, such as one
        line-of-sight check per timestep; None keeps them for this call only.

    Returns
    -------
    bool
        True if line of sight is clear (all points inside environment).
    """
    return bool(
        batch_line_of_sight_clear(
            binning,
            arena,
            observer,
            np.asarray(target).reshape(1, 2),
            n_samples=n_samples,
            boundary_source=boundary_source,
            indexes=indexes,
        )[0]
    )


def batch_line_of_sight_clear(
    binning: Binning,
    arena: Arena,
    observer: NDArray[np.float64],
    targets: NDArray[np.float64],
    n_samples: int = 20,
    *,
    boundary_source: Literal["physical", "analysis"] = "physical",
    indexes: LookupIndexes | None = None,
) -> NDArray[np.bool_]:
    """Check line of sight between observer and multiple targets (vectorized).

    Parameters
    ----------
    binning : Binning
        The environment's bins, from ``env.binning``.
    arena : Arena
        The environment's continuous domain, from ``env.arena``.
    observer : NDArray[np.float64], shape (2,)
        Observer position.
    targets : NDArray[np.float64], shape (n_targets, 2)
        Target positions.
    n_samples : int
        Retained for compatibility; line-of-sight tests use exact intersections.

    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls, or explicitly requested analysis-cell boundaries.
        Physical mode requires exact supported wall geometry; missing analysis
        bins do not occlude a physical ray. Fixed-distance gaze ignores this.
    indexes : LookupIndexes, optional
        Search structures shared by the calls of one computation; None keeps
        them for this call only.

    Returns
    -------
    NDArray[np.bool_], shape (n_targets,)
        True if line of sight is clear for each target.
    """

    walls = boundary_geometry(binning, arena, boundary_source)
    observer = np.asarray(observer, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if observer.shape != (2,) or targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError(
            "Line-of-sight observer must have shape (2,) and targets (n_targets, 2)."
        )
    indexes = CallIndexes() if indexes is None else indexes
    if not valid_observers(
        binning, arena, observer.reshape(1, 2), boundary_source, indexes
    )[0]:
        return np.zeros(len(targets), dtype=bool)
    return (
        physical_clear_segments(arena, observer, targets)
        if boundary_source == "physical"
        else clear_segments(
            walls, observer, targets, analysis_interior(binning, arena, indexes)
        )
    )
