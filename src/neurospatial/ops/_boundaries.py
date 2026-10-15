"""Exact planar wall evidence shared by border and visibility consumers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.ops import unary_union

if TYPE_CHECKING:
    from neurospatial.arena import Arena, _TransformedArena
    from neurospatial.ops._protocols import BoundaryConsumer, ObserverConsumer

BoundarySource = Literal["physical", "analysis"]
Interior = Callable[[NDArray[np.float64]], NDArray[np.bool_]]


def boundary_geometry(
    env: BoundaryConsumer, source: BoundarySource
) -> shapely.Geometry:
    """Return declared physical walls or explicitly selected analysis boundaries.

    Physical mode uses exact planar footprint rings and explicit internal walls.
    It never infers physical walls from bins, sampling masks or unknown domains.
    Analysis mode uses the exact union of retained planar analysis cells.
    """
    if source not in ("physical", "analysis"):
        raise ValueError("boundary_source must be 'physical' or 'analysis'.")
    if env.arena.coordinate_kind != "cartesian" or env.arena.n_dims != 2:
        raise NotImplementedError(
            "Wall geometry requires planar Cartesian coordinates."
        )
    if source == "analysis":
        from neurospatial.layout._polygon import materialize_cells

        try:
            domain = unary_union(materialize_cells(env._binning))
            restriction = (
                env.arena.polygonal_domain() if env.arena.domain_known else None
            )
            if restriction is not None:
                domain = domain.intersection(restriction)
            return unary_union([domain.boundary, *env.arena.walls])
        except ValueError as exc:
            raise NotImplementedError(
                "Analysis-boundary geometry needs exact planar cells; supply supported grid or polygon cells."
            ) from exc
    return physical_walls(env.arena)


def _physical_evidence(
    arena: Arena,
) -> tuple[list[shapely.Geometry], list[shapely.Geometry]]:
    """Explicit wall lines and asserted footprint polygons in the arena frame."""
    from neurospatial.arena import _CombinedArena, _mapped_geometry, _TransformedArena

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
            [_mapped_geometry(part, arena.forward) for part in lines],
            [_mapped_geometry(part, arena.forward) for part in footprints],
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


def analysis_interior(env: ObserverConsumer) -> Interior:
    """Retained analysis support, matching analysis-mode observer validity."""
    return lambda points: np.asarray(env.bin_at(points) >= 0, dtype=np.bool_)


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
    env: ObserverConsumer, positions: NDArray[np.float64], source: BoundarySource
) -> NDArray[np.bool_]:
    """Require finite observers inside the domain justified by the chosen source."""
    valid = np.asarray(np.all(np.isfinite(positions), axis=1), dtype=np.bool_)
    if source == "analysis":
        valid[valid] = env.bin_at(positions[valid]) >= 0
    elif env.arena.physical_footprint_known:
        valid[valid] = env.arena.contains(positions[valid])
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
    from neurospatial.arena import _feature_cap, _TransformedArena

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
