"""Exact planar wall evidence shared by border and visibility consumers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union

from neurospatial._exceptions import CartesianOnlyError, _format_error
from neurospatial._validation import validate_finite_scalar
from neurospatial.geometry._arena import (
    Blocking,
    _CombinedArena,
    _feature_cap,
    _TransformedArena,
    blocks,
    mapped_geometry,
    outline_declared,
)
from neurospatial.geometry._composition import CombinedLocator, iter_locators
from neurospatial.geometry._locators import (
    CallIndexes,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
    coordinate_roundoff,
)
from neurospatial.geometry._lookup import locate, locate_outline_points
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
    binning: Binning, arena: Arena, source: BoundarySource, *, blocking: Blocking
) -> shapely.Geometry:
    """Return declared physical boundaries or explicitly selected analysis ones.

    Physical mode uses exact planar footprint rings and named walls. It never
    infers physical boundaries from bins, sampling masks or unknown domains.
    Analysis mode uses the exact union of retained planar analysis cells, plus
    the named walls.

    ``blocking`` chooses the boundaries: ``"movement"`` keeps every physical
    boundary (walls, transparent walls and drop edges), for analyses of
    where the animal is relative to the edges it cannot cross; ``"gaze"``
    keeps only opaque walls, for analyses of what the animal sees. An arena
    whose physical boundaries are all transparent walls or drop edges has
    no gaze boundary: the result is empty.

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
        return _analysis_boundary(binning, arena, blocking)
    return physical_walls(arena, blocking=blocking)


def _analysis_boundary(
    binning: Binning, arena: Arena, blocking: Blocking
) -> shapely.Geometry:
    """Return retained cell edges restricted by the supplied planar domain."""

    try:
        cells = materialize_cells(binning)
    except ValueError as exc:
        raise NotImplementedError(
            _format_error(
                "Analysis-boundary geometry (boundary_source='analysis') needs "
                "exact planar cells, and these bins have none: "
                f"{str(exc).splitlines()[0]}",
                fix="use a 2-D grid, hexagonal layout, triangular mesh or "
                "polygon cells, or declare the walls in the arena and use "
                "boundary_source='physical'",
            )
        ) from exc
    return _cells_boundary(cells, arena, blocking)


def _cells_boundary(
    cells: Sequence[shapely.Geometry], arena: Arena, blocking: Blocking
) -> shapely.Geometry:
    """Edges of the union of ``cells`` within the arena domain, plus its walls.

    Only the walls that block ``blocking`` are added.
    """
    domain = unary_union(cells)
    restriction = arena.polygonal_domain() if arena.domain_known else None
    if restriction is not None:
        domain = domain.intersection(restriction)
    walls = [
        wall
        for wall, kind in zip(arena.walls, arena.wall_kinds, strict=True)
        if blocks(kind, blocking)
    ]
    return unary_union([domain.boundary, *walls])


def _snap_onto_walls(
    query: NDArray[np.float64],
    error: NDArray[np.float64],
    walls: shapely.Geometry,
    arena: Arena,
) -> None:
    """Move finite ``query`` rows within their bounded error of ``walls`` onto them.

    ``query`` and ``error`` have shape ``(n_points, 2)``; ``query`` is
    updated in place.
    """
    finite = np.all(np.isfinite(query), axis=1)
    points = shapely.points(query[finite])
    allowance = np.hypot.reduce(np.minimum(error[finite], _feature_cap(arena)), axis=1)
    close = shapely.distance(points, walls) <= allowance
    nearest = shapely.get_point(shapely.shortest_line(points[close], walls), -1)
    query[np.flatnonzero(finite)[close]] = np.column_stack(
        (shapely.get_x(nearest), shapely.get_y(nearest))
    )


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
    cap = _feature_cap(arena)
    budgets = np.array(
        [
            np.hypot.reduce(np.minimum(_polygon_coordinate_errors(cell), cap))
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
    """Assign gaze hits on walls to observer-side bins using owned geometry.

    ``hits`` and ``origins`` have shape ``(n_hits, 2)``. ``bins`` contains
    the caller's initial lookup of the observer-side limit, shape ``(n_hits,)``.
    Shared affine frames are resolved in their stored cell-owner coordinates.
    The hits lie on the boundaries that block gaze (see
    :func:`boundary_geometry`).
    """

    def source_locator(
        locator: Locator, arena: Arena, centers: NDArray[np.float64]
    ) -> Locator:
        if isinstance(locator, TransformedLocator):
            if isinstance(arena, _TransformedArena) and locator.affine.same_as(
                arena.affine
            ):
                original, _error = arena.affine.project(centers)
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
        # A child with its own affine frame has no one shared owner frame.
        if (
            isinstance(locator, CombinedLocator)
            and source == "physical"
            and any(
                isinstance(part, TransformedLocator) for part in iter_locators(locator)
            )
        ):
            raise NotImplementedError(
                "Physical wall-bin assignment needs a shared source frame for combined components. Combine before a shared transform, or explicitly choose boundary_source='analysis'."
            )
        return locator

    locator = source_locator(binning.locator, arena, binning.bins.centers)
    frame = wall_source_frame(binning.locator, arena, binning.bins.centers)
    indexes = CallIndexes()
    if frame is None:
        if source == "physical":
            walls = physical_walls(arena, blocking="gaze")
        else:
            walls = _analysis_boundary(binning, arena, "gaze")
        resolved = _resolve_wall_contacts(
            locator,
            arena,
            materialize_cells(binning),
            walls,
            hits,
            origins,
            bins,
            indexes,
        )
        return _sliver_owners(binning, arena, hits, origins, resolved)
    cells, _walls, _centers, _inverse, frames, _errors = frame
    query = np.vstack((hits, origins))
    error = None
    for arena in frames:
        query, error = arena.affine.project(query, error)
    base = frames[-1].base
    walls = (
        physical_walls(base, blocking="gaze")
        if source == "physical"
        else _cells_boundary(cells, base, "gaze")
    )
    assert error is not None
    _snap_onto_walls(query[: len(hits)], error[: len(hits)], walls, base)
    inside = np.nextafter(query[: len(hits)], query[len(hits) :])
    bins = locator.locate(inside, indexes)
    if base.domain_known:
        bins[~base.contains(inside)] = -1
    resolved = _resolve_wall_contacts(
        locator,
        base,
        cells,
        walls,
        query[: len(hits)],
        query[len(hits) :],
        bins,
        indexes,
    )
    return _sliver_owners(binning, arena, hits, origins, resolved)


def _sliver_owners(
    binning: Binning,
    arena: Arena,
    hits: NDArray[np.float64],
    origins: NDArray[np.float64],
    bins: NDArray[np.intp],
) -> NDArray[np.intp]:
    """Give hits in an outline sliver the bin that owns positions there.

    A grid keeps the cells whose centers lie strictly inside the outline, so
    a slanted or curved wall, or a wall through a row of cell centers, runs
    through grid cells that are not bins. No retained cell touches a hit
    there; like a wall piece in a border score, the hit belongs to the bin
    that owns positions in that sliver (see :func:`locate_outline_points`).
    ``hits``, ``origins`` have shape ``(n_hits, 2)``; ``bins`` is updated for
    the hits still without a bin and returned.
    """
    missing = np.flatnonzero((bins < 0) & np.all(np.isfinite(hits), axis=1))
    if len(missing) and arena.domain_known:
        near = np.nextafter(hits[missing], origins[missing])
        bins[missing] = locate_outline_points(binning, arena, near)
    return bins


def _physical_footprints(arena: Arena) -> list[shapely.Geometry]:
    """Asserted footprint polygons in the arena frame, component by component.

    A combined arena gives each component's asserted footprint, even when
    other components assert none.
    """
    if isinstance(arena, _CombinedArena):
        return [
            part for child in arena.children for part in _physical_footprints(child)
        ]
    if isinstance(arena, _TransformedArena):
        return [
            mapped_geometry(part, arena.forward)
            for part in _physical_footprints(arena.base)
        ]
    if arena.outline is None:
        return []
    domain = arena.polygonal_domain()
    if domain is None:
        raise NotImplementedError(
            "Physical footprint has no exact planar polygon boundary. Supply explicit polygonal geometry or choose boundary_source='analysis' when supported."
        )
    return [domain]


def _physical_boundaries(arena: Arena, blocking: Blocking) -> list[shapely.Geometry]:
    """Physical boundary lines that block ``blocking``, component by component.

    The outline of each footprint has its arena's ``outline`` kind, except
    where a named wall lies on it and gives that stretch its own kind.
    """
    if isinstance(arena, _CombinedArena):
        return [
            part
            for child in arena.children
            for part in _physical_boundaries(child, blocking)
        ]
    if isinstance(arena, _TransformedArena):
        return [
            mapped_geometry(part, arena.forward)
            for part in _physical_boundaries(arena.base, blocking)
        ]
    walls = list(zip(arena.walls, arena.wall_kinds, strict=True))
    parts = [wall for wall, kind in walls if blocks(kind, blocking)]
    if arena.outline is not None and blocks(arena.outline, blocking):
        ring = unary_union([domain.boundary for domain in _physical_footprints(arena)])
        open_walls = [wall for wall, kind in walls if not blocks(kind, blocking)]
        if open_walls:
            ring = ring.difference(unary_union(open_walls))
        parts.append(ring)
    return parts


def physical_footprint_interior(arena: Arena) -> Interior | None:
    """Closed asserted footprints, or None when only wall lines are known."""
    footprints = _physical_footprints(arena)
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


def physical_walls(arena: Arena, *, blocking: Blocking) -> shapely.Geometry:
    """Declared physical boundaries that block ``blocking``, independent of bins.

    ``"movement"`` keeps every physical boundary; ``"gaze"`` keeps only opaque
    walls, so the result is empty for an arena bounded by drop edges or
    transparent walls alone.

    Raises
    ------
    NotImplementedError
        If the arena declares no physical boundary at all: its outline is
        not physical and it has no named walls.
    """
    if not arena.walls and not _physical_footprints(arena):
        raise NotImplementedError(
            "Physical boundaries are unknown: this environment's outline was inferred from positions, "
            "bins or a mask, not declared as a wall or drop edge, and it has no named walls. Declare what "
            "the outline is: Environment.from_polygon(outline, bin_size) for a walled arena, or "
            "arena=OpenArena(outline, outline='drop') for an elevated track or platform "
            "(MaskArena(..., outline=...) for a mask). Boundary-based rate maps also need bin support at the "
            "measured boundaries; attaching an arena does not extend bins or supply missing observations. "
            "Alternatively, explicitly choose boundary_source='analysis' to use retained bin edges as an approximation."
        )
    return unary_union(_physical_boundaries(arena, blocking))


def movement_enclosure(
    binning: Binning, arena: Arena
) -> tuple[Polygon | MultiPolygon, list[LineString]] | None:
    """The floor positions lie on and the walls on it that block movement.

    The floor is the closed region that position lookup accepts: the
    arena's polygonal domain when it is known, else the union of the planar
    cells. The walls are the arena's named walls that block movement (every
    physical kind does). A simulated animal moves inside the floor's outline
    and never crosses a wall.

    Returns
    -------
    tuple of (Polygon or MultiPolygon, list of LineString), or None
        None when neither the domain nor the bins are planar 2-D areas, such
        as a track's lengths.
    """
    floor = arena.polygonal_domain() if arena.domain_known else None
    if floor is None:
        bins = binning.bins
        if bins.n_dims != 2 or not bins.has_measure_dim(2):
            return None
        floor = unary_union(list(materialize_cells(binning)))
    walls = [
        wall
        for wall, kind in zip(arena.walls, arena.wall_kinds, strict=True)
        if blocks(kind, "movement")
    ]
    return floor, walls


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
    elif outline_declared(arena):
        valid[valid] = arena.contains(positions[valid])
    return valid


def _ray_segments(
    origins: NDArray[np.float64], angles: NDArray[np.float64], max_distance: float
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    """Validate rays and return only finite starts and finite search endpoints."""
    validate_finite_scalar(max_distance, name="max_distance", positive=True)
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


def _reference(walls: shapely.Geometry) -> float:
    """Largest coordinate magnitude of ``walls``; zero when there are none."""
    bounds = np.asarray(shapely.bounds(walls), dtype=np.float64)
    return float(np.max(np.abs(np.nan_to_num(bounds)), initial=0.0))


def _start_departures(
    contacts: shapely.Geometry | NDArray[np.object_],
    starts: NDArray[np.float64],
    ends: NDArray[np.float64],
    interior: Interior | None,
    reference: float,
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
    A start within :func:`coordinate_roundoff` of a wall, the one
    on-boundary tolerance, is on it.

    Parameters
    ----------
    contacts : shapely geometry array, shape (n_segments,)
        Intersection of each segment with the walls.
    starts, ends : ndarray, shape (n_segments, 2)
        Oriented segment endpoints.
    interior : callable or None
        Maps points of shape (n_points, 2) to closed-domain membership.
    reference : float
        Largest coordinate magnitude of the walls.

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
    at_start = distance <= coordinate_roundoff(starts, reference)[owners]
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
    reference = _reference(walls)
    with np.errstate(invalid="ignore"):
        rows = np.flatnonzero(
            np.hypot.reduce(hits - starts, axis=1)
            <= coordinate_roundoff(starts, reference)
        )
    if len(rows):
        departs, later = _start_departures(
            intersections[rows], starts[rows], ends[rows], interior, reference
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
    arena: Arena,
    query: NDArray[np.float64],
    blocking: Blocking,
    walls: shapely.Geometry | None = None,
) -> tuple[Arena, NDArray[np.float64], list[_TransformedArena]]:
    """Use stored projection errors for contact in a shared physical-wall frame.

    ``walls``, when given, are the base arena's ``physical_walls`` for
    ``blocking``, already built.
    """

    frames = []
    error = None
    while isinstance(arena, _TransformedArena):
        frames.append(arena)
        query, error = arena.affine.project(query, error)
        arena = arena.base
    if error is not None:
        if walls is None:
            walls = physical_walls(arena, blocking=blocking)
        _snap_onto_walls(query, error, walls, arena)
    return arena, query, frames


def physical_ray_hits(
    arena: Arena,
    origins: NDArray[np.float64],
    angles: NDArray[np.float64],
    max_distance: float,
    *,
    blocking: Blocking,
) -> NDArray[np.float64]:
    """Preserve first contacts through owned affine frames and bounded rounding.

    Rays stop at the physical boundaries that block ``blocking`` (see
    :func:`physical_walls`); a ray that meets none is NaN.
    """
    origins = np.asarray(origins, dtype=np.float64)
    valid, starts, ends = _ray_segments(
        origins, np.asarray(angles, dtype=np.float64), max_distance
    )
    base, query, frames = _wall_frame(arena, np.vstack((starts, ends)), blocking)
    if not np.all(np.isfinite(query)):
        raise ValueError(
            "Ray projection exceeds coordinate precision; recenter or rescale positions."
        )
    hits = _segment_hits(
        physical_walls(base, blocking=blocking),
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


def physical_sight_lines(
    arena: Arena,
) -> Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.bool_]]:
    """Clear sight lines past every boundary but opaque walls, walls built once.

    Keeps on-wall cues visible under shared-frame inverse rounding. The
    returned test takes an observer, shape (2,), and targets, shape
    (n_targets, 2), as :func:`physical_clear_segments` does.
    """
    base = arena
    while isinstance(base, _TransformedArena):
        base = base.base
    walls = physical_walls(base, blocking="gaze")
    sight = SightWalls(walls)
    interior = physical_footprint_interior(base)

    def clear(
        observer: NDArray[np.float64], targets: NDArray[np.float64]
    ) -> NDArray[np.bool_]:
        _base, query, _frames = _wall_frame(
            arena, np.vstack((observer, targets)), "gaze", walls
        )
        return sight.clear(query[0], query[1:], interior)

    return clear


def physical_clear_segments(
    arena: Arena, observer: NDArray[np.float64], targets: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Clear sight lines past every boundary but opaque walls.

    Keeps on-wall cues visible under shared-frame inverse rounding.
    """
    return physical_sight_lines(arena)(observer, targets)


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
    return SightWalls(walls).clear(observer, targets, interior)


class SightWalls:
    """The straight segments of ``walls``, indexed once for many sight lines.

    Parameters
    ----------
    walls : shapely.Geometry
        Known planar line geometry, possibly empty.
    """

    def __init__(self, walls: shapely.Geometry) -> None:
        parts = shapely.get_parts(walls)
        vertices, part_index = shapely.get_coordinates(parts, return_index=True)
        adjacent = part_index[:-1] == part_index[1:]
        segments = np.stack((vertices[:-1], vertices[1:]), axis=1)[adjacent]
        #: Wall segments, shape (n_segments, 2, 2).
        self.segments = segments[np.any(segments[:, 0] != segments[:, 1], axis=1)]
        self.tree = shapely.STRtree(shapely.linestrings(self.segments))
        self.reference = float(np.max(np.abs(self.segments), initial=0.0))

    def clear(
        self,
        observer: NDArray[np.float64],
        targets: NDArray[np.float64],
        interior: Interior | None = None,
    ) -> NDArray[np.bool_]:
        """Whether each target is in sight of ``observer``; see `clear_segments`."""
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
            # Intersect each sight line only with the wall segments it touches:
            # most lines touch none, and overlaying a line with the whole wall
            # geometry costs time in the number of wall vertices.
            tree, wall_segments = self.tree, self.segments
            line_index, segment_index = tree.query(lines, predicate="intersects")
            order = np.argsort(line_index, kind="stable")
            line_index, segment_index = line_index[order], segment_index[order]
            pieces = shapely.intersection(
                lines[line_index], tree.geometries[segment_index]
            )
            kept = ~shapely.is_empty(pieces)
            contact = np.full(len(lines), shapely.GeometryCollection(), dtype=object)
            if kept.any():
                shapely.geometrycollections(
                    pieces[kept], indices=line_index[kept], out=contact
                )
            # Intersection and interpolation can cancel one endpoint coordinate
            # to zero. Allow eight ULPs in the cue's own coordinates and project
            # that rounding box onto the intersected wall's tangent. A coordinate
            # perpendicular to the wall contributes no cross-axis allowance, and
            # distant endpoints cannot widen gaps. Any positive wall overlap blocks.
            coordinates, owners = shapely.get_coordinates(contact, return_index=True)
            terminal = selected[~same]
            allowance = 8 * np.spacing(np.abs(terminal[owners]))
            if len(coordinates):
                nearest = tree.nearest(shapely.points(coordinates))
                tangent = np.abs(np.diff(wall_segments[nearest], axis=1)[:, 0])
                tangent /= np.hypot.reduce(tangent, axis=1)[:, None]
                allowance += tangent * np.sum(tangent * allowance, axis=1)[:, None]
            at_terminal = np.all(
                np.abs(coordinates - terminal[owners]) <= allowance, axis=1
            )
            own_wall = (
                np.hypot.reduce(coordinates - observer, axis=1)
                <= coordinate_roundoff(observer.reshape(1, 2), self.reference)[0]
            )
            rows = np.unique(owners[own_wall])
            if len(rows):
                departs, _later = _start_departures(
                    contact[rows],
                    np.broadcast_to(observer, (len(rows), 2)),
                    terminal[rows],
                    interior,
                    self.reference,
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
    """Bound all observer-to-wall distances for automatic finite-domain searches.

    Empty ``walls`` (nothing blocks the search) bound only the positions.
    """
    finite = np.asarray(positions)[np.all(np.isfinite(positions), axis=1)]
    corners = np.empty((0, 2)) if walls.is_empty else np.reshape(walls.bounds, (2, 2))
    coordinates = np.vstack((corners, finite))
    if not len(coordinates):
        return 1.0
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
    boundary_source: BoundarySource,
    indexes: LookupIndexes,
    *,
    blocking: Blocking,
) -> NDArray[np.float64]:
    """First boundary each ray reaches, shape (n_time, 2); NaN for none.

    Parameters
    ----------
    binning : Binning
        The environment's bins, from ``env.binning``.
    arena : Arena
        The environment's continuous domain, from ``env.arena``.
    positions : ndarray, shape (n_time, 2)
        Observer positions. A non-finite observer, or one outside the domain
        the chosen ``boundary_source`` justifies, gives a NaN row.
    gaze_direction : ndarray, shape (n_time,)
        Allocentric gaze direction in radians: 0 = East (+x), +π/2 = North
        (+y). A non-finite direction gives a NaN row.
    max_distance : float
        Finite positive search length; a ray reaching no wall within it gives
        a NaN row.
    boundary_source : {'physical', 'analysis'}
        Declared physical walls, or the boundaries of the analysis cells.
    indexes : LookupIndexes
        Keeps the locator's search structures for the computation.
    blocking : {'movement', 'gaze'}
        Which boundaries stop a ray (see :func:`boundary_geometry`):
        ``"gaze"`` for what the animal sees, so a gaze passes transparent
        walls and drop edges; ``"movement"`` for the distance to every edge
        the animal cannot cross.

    Returns
    -------
    ndarray, shape (n_time, 2)
        The first boundary point on each ray, or NaN.
    """

    walls = boundary_geometry(binning, arena, boundary_source, blocking=blocking)
    origins = positions.copy()
    origins[~valid_observers(binning, arena, positions, boundary_source, indexes)] = (
        np.nan
    )
    if boundary_source == "physical":
        return physical_ray_hits(
            arena, origins, gaze_direction, max_distance, blocking=blocking
        )
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
    *,
    boundary_source: BoundarySource = "physical",
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
        Observer position; must be finite.
    target : NDArray[np.float64], shape (2,)
        Target position; must be finite.
    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls, or explicitly requested analysis-cell boundaries.
        Physical mode requires exact supported wall geometry; missing analysis
        bins do not occlude a physical ray.
    indexes : LookupIndexes, optional
        Search structures shared by the calls of one computation, such as one
        line-of-sight check per timestep; None keeps them for this call only.

    Returns
    -------
    bool
        True if the segment from observer to target crosses no opaque wall
        of the chosen boundary source; transparent walls and drop edges do
        not block a line of sight. False if it does, or if the observer is
        outside the region that source covers.

    Raises
    ------
    CartesianOnlyError
        If the arena holds polar ``(distance, angle)`` coordinates.
    NotImplementedError
        If the environment is not 2-D, or the chosen boundary source has no
        exact planar walls or cells.
    ValueError
        If the observer or target is not finite (a missing observation), or
        has the wrong shape.
    """
    return bool(
        batch_line_of_sight_clear(
            binning,
            arena,
            observer,
            np.asarray(target).reshape(1, 2),
            boundary_source=boundary_source,
            indexes=indexes,
        )[0]
    )


def batch_line_of_sight_clear(
    binning: Binning,
    arena: Arena,
    observer: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    boundary_source: BoundarySource = "physical",
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
        Observer position; must be finite.
    targets : NDArray[np.float64], shape (n_targets, 2)
        Target positions; must be finite.
    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls, or explicitly requested analysis-cell boundaries.
        Physical mode requires exact supported wall geometry; missing analysis
        bins do not occlude a physical ray.
    indexes : LookupIndexes, optional
        Search structures shared by the calls of one computation; None keeps
        them for this call only.

    Returns
    -------
    NDArray[np.bool_], shape (n_targets,)
        True where the segment from the observer to the target crosses no
        opaque wall of the chosen boundary source; transparent walls and
        drop edges do not block a line of sight. All False when the observer
        is outside the region that source covers.

    Raises
    ------
    CartesianOnlyError
        If the arena holds polar ``(distance, angle)`` coordinates.
    NotImplementedError
        If the environment is not 2-D, or the chosen boundary source has no
        exact planar walls or cells.
    ValueError
        If the observer or a target is not finite (a missing observation), or
        the arrays have the wrong shape. A missing position is not evidence
        of occlusion, so callers keep such frames missing instead.
    """

    walls = boundary_geometry(binning, arena, boundary_source, blocking="gaze")
    observer = np.asarray(observer, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if observer.shape != (2,) or targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError(
            "Line-of-sight observer must have shape (2,) and targets (n_targets, 2)."
        )
    missing = []
    if not np.all(np.isfinite(observer)):
        missing.append(f"the observer {observer.tolist()}")
    bad_targets = np.flatnonzero(~np.isfinite(targets).all(axis=1))
    if len(bad_targets):
        missing.append(f"{len(bad_targets)} target(s) (first row {bad_targets[0]})")
    if missing:
        raise ValueError(
            _format_error(
                "Line of sight needs finite positions; NaN or infinite: "
                + " and ".join(missing)
                + ".",
                why="Why: a missing position is not an observation, and calling "
                "its targets blocked would read lost tracking as occlusion.",
                fix="check only frames with a finite observer and targets, and "
                "keep the other frames missing (NaN or masked) in the result",
            )
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
