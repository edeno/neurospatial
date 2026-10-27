"""Wall geometry for border scores: individual walls, wall contact and distances.

Walls come from the arena's geometry in physical units. Each closed outline
(an arena ring, or the retained bins' outline) is split into walls at its
corners, and each declared wall (``OpenArena(walls=...)``) is one wall: on
the outline it replaces the piece of outline it lies on, inside the arena it
is an internal wall.

Corners
-------
Two lengths set where an outline has a corner.

- The resolution ``epsilon``: the median distance from a bin's center to its
  farthest cell vertex (half the diagonal of a square bin). The rate map
  cannot place a wall more finely than its bins, and an outline along cell
  edges (a pixel mask, the retained bins' outline) lies within this distance
  of the boundary it approximates: each outline point is on the edge between
  a cell whose center is inside and one whose center is outside, so the
  boundary crosses within half a cell diagonal of it. A measured outline
  (an arena polygon) can be rougher than the bins, for example a hand trace
  with jitter. Where it is sampled more finely than the window ``w`` below,
  a vertex lies off the line through its neighbors by at most twice the
  outline's error, so half the largest such offset is a lower bound on that
  error, and ``epsilon`` is the larger of the two.
- The arena's own curvature: the radius ``rho`` of the largest circle inside
  the outline. A circular arena's wall turns exactly as fast as this circle.

A vertex of the outline is a corner when it is resolvable: rounding it off
with the largest tangent arc that fits in half of each of its edges moves
the outline by ``min(a, b) / 2 * tan(turn / 4)`` (edges ``a``, ``b``), and a
vertex that moves it by more than ``epsilon`` is not the sample of a smooth
curve. This keeps the user's polygon edges as walls (an octagon has eight
walls) while a densely sampled curve (``Point.buffer``, a hand trace, a pixel
staircase) has no corner vertex.

A rounded or traced corner is found from how much the outline turns between
the chord to the point a length ``w`` back along it and the chord to the
point ``w`` ahead. On the arena's inscribed circle this turn is
``2 asin(w / 2 rho)``, and moving each chord end by up to ``epsilon`` changes
it by at most ``2 atan(2 epsilon / w)``; a turn larger than their sum is
faster than the arena's own curvature by more than the map can resolve.
``w = 2 sqrt(epsilon rho)`` minimizes this threshold (the two terms are
``w / rho`` and ``4 epsilon / w`` for small angles), so it detects the gentlest
corner the map resolves. A stretch turning faster than the inscribed circle
is one corner, cut where it turns most, when somewhere along it the turn
exceeds the threshold; measurement error near the threshold therefore
cannot break one corner into several. A circular arena never turns faster
than its inscribed circle, so it has no corner at any bin size, while a box
with rounded corners has four.

A closed outline without corners (a circle, an ellipse) is one ring-shaped
wall. Høydal et al. (2018) score a circular arena against half of its wall,
since border cells there often cover half of it, so a field along half the
ring covers its whole wall. Here the wall a ring offers a field is its best
half (by length).

References
----------
Høydal, Ø. A., Skytøen, E. R., Moser, M.-B., & Moser, E. I. (2018).
    Object-vector coding in the medial entorhinal cortex. bioRxiv 286286.
    https://doi.org/10.1101/286286 (Methods, "Analysis of border cells").
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.sparse import coo_array, csr_array
from shapely.ops import substring, unary_union

from neurospatial.environment._cache import CacheKey
from neurospatial.geometry import locate_outline_points, materialize_cells

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.geometry._arena import Arena
    from neurospatial.geometry._boundaries import BoundarySource
    from neurospatial.ops._types import DistanceMetric


@dataclass(frozen=True)
class _BorderGeometry:
    """Rate-independent wall geometry of one environment, in its coordinates.

    Each wall is cut into atoms: pieces of wall owned by one set of bins (the
    bins whose cells touch the piece or, in an outline sliver, the bin that
    owns positions there). A field covers an atom when it contains one of the
    atom's owners; a rate map observes it when one of its owners has a finite
    rate.

    Attributes
    ----------
    walls : shapely geometry
        All walls together.
    wall_lengths : ndarray of float64, shape (n_walls,)
        Length of each individual wall.
    closed : ndarray of bool, shape (n_walls,)
        Walls that close on themselves (a ring without corners). A field's
        coverage of such a wall is measured on its best half.
    atom_wall : ndarray of intp, shape (n_atoms,)
        The wall each atom belongs to.
    atom_start : ndarray of float64, shape (n_atoms,)
        Distance along its wall to the start of each atom; atoms of one wall
        are contiguous and in order.
    atom_length : ndarray of float64, shape (n_atoms,)
        Length of each atom.
    atom_owners : scipy.sparse.csr_array, shape (n_atoms, n_bins)
        1 where a bin owns an atom.
    wall_bins : ndarray of bool, shape (n_bins,)
        Bins that own some wall.
    direct : ndarray of float64, shape (n_bins,)
        Straight-line distance from each bin center to the nearest wall.
    geodesic : ndarray of float64, shape (n_bins,)
        Path length from the walls through the navigation graph, seeded at the
        wall-owning bins; ``inf`` where no path exists.
    radius : float
        Radius of the largest circle that fits inside the arena without
        crossing a wall: half the shorter side of a rectangle.
    """

    walls: shapely.Geometry
    wall_lengths: NDArray[np.float64]
    closed: NDArray[np.bool_]
    atom_wall: NDArray[np.intp]
    atom_start: NDArray[np.float64]
    atom_length: NDArray[np.float64]
    atom_owners: csr_array
    wall_bins: NDArray[np.bool_]
    direct: NDArray[np.float64]
    geodesic: NDArray[np.float64]
    radius: float


PHYSICAL_BORDER: CacheKey[_BorderGeometry] = CacheKey("physical_border")
ANALYSIS_BORDER: CacheKey[_BorderGeometry] = CacheKey("analysis_border")


def border_geometry(env: Environment, source: BoundarySource) -> _BorderGeometry:
    """The environment's prepared wall geometry, built once and cached.

    The walls depend only on the environment, so they are the same for every
    rate map of the environment.
    """
    key = PHYSICAL_BORDER if source == "physical" else ANALYSIS_BORDER
    return env._cache.get(key, lambda: _prepare_border(env, source))


def _prepare_border(env: Environment, source: BoundarySource) -> _BorderGeometry:
    """Individual walls, wall ownership and distances, refusing unowned walls.

    Independent of any firing rate, so an environment prepares it once (see
    :func:`border_geometry`).
    """
    from neurospatial.geometry import boundary_geometry

    # Validates the source, coordinates and dimensionality, and raises when
    # the walls are unknown. A border is any edge the animal cannot cross:
    # border cells keep firing along a drop edge once the walls are removed
    # (see border_score).
    boundary_geometry(env.binning, env.arena, source, blocking="movement")
    if source == "physical":
        cells, source_arena, forward = _physical_frame(env)
        source_domain = unary_union(_footprints(source_arena))
        groups = _declared_walls(source_arena)
        domain = unary_union(_footprints(env.arena))
        pixels = _pixel_footprints(source_arena)
    else:
        try:
            cells = list(materialize_cells(env.binning))
        except ValueError as exc:
            raise NotImplementedError(
                "Analysis-boundary border scores need exact planar cells; use a 2-D "
                "grid, hexagonal layout, triangular mesh or polygon cells."
            ) from exc
        forward = None
        domain = unary_union(cells)
        if env.arena.domain_known:
            restriction = env.arena.polygonal_domain()
            if restriction is not None:
                domain = domain.intersection(restriction)
        source_domain = domain
        # The retained bins' outline runs along cell edges.
        pixels = [domain]
        # Only the parts of declared walls among the retained bins bound them.
        groups = [
            [
                part
                for segment in segments
                for part in shapely.get_parts(segment.intersection(domain))
                if part.length > 0
            ]
            for segments in _declared_walls(env.arena)
        ]
    tolerance = 1e-9 * np.sqrt(source_domain.area)
    pixel_edges = unary_union([polygon.boundary for polygon in pixels])
    rings = [
        (ring, not _lies_on(ring, pixel_edges, tolerance))
        for ring in _polygon_rings(source_domain)
    ]
    walls, closed = _split_walls(
        rings,
        [group for group in groups if group],
        resolution=_resolution(cells),
        scale=_inscribed_radius(source_domain, ()),
        tolerance=tolerance,
    )
    atoms = _wall_atoms(env, walls, cells, forward)
    wall_bins = np.asarray(atoms[3].sum(axis=0)).ravel() > 0
    env_walls = walls if forward is None else [_mapped(w, forward) for w in walls]
    wall_union = unary_union(env_walls)
    direct = np.asarray(
        shapely.distance(shapely.points(env.bin_centers), wall_union), dtype=float
    )
    atom_wall, atom_start, atom_length, atom_owners = atoms
    return _BorderGeometry(
        walls=wall_union,
        wall_lengths=np.bincount(atom_wall, atom_length, minlength=len(walls)).astype(
            np.float64
        ),
        closed=np.asarray(closed, dtype=bool),
        atom_wall=atom_wall,
        atom_start=atom_start,
        atom_length=atom_length,
        atom_owners=atom_owners,
        wall_bins=wall_bins,
        direct=direct,
        geodesic=_distance_from_walls(env, direct, wall_bins),
        radius=_inscribed_radius(domain, env.arena.walls),
    )


def _mapped(
    geometry: shapely.Geometry, forward: NDArray[np.float64]
) -> shapely.Geometry:
    from neurospatial.geometry import mapped_geometry

    return mapped_geometry(geometry, forward)


def _physical_frame(
    env: Environment,
) -> tuple[list[shapely.Geometry], Arena, NDArray[np.float64] | None]:
    """Cells and arena in the frame the walls were declared in, and the map out.

    Refuses arenas whose outer walls are unknown and combined cells without a
    shared source frame.
    """
    from neurospatial.geometry import (
        ClippedLocator,
        CombinedLocator,
        SubsetLocator,
        TransformedLocator,
        _CombinedArena,
        _TransformedArena,
        composition_view,
        outline_declared,
        wall_source_frame,
    )

    def independent_frame(locator: object) -> bool:
        if isinstance(locator, TransformedLocator):
            return True
        if isinstance(locator, (SubsetLocator, ClippedLocator)):
            return independent_frame(locator.base)
        if isinstance(locator, CombinedLocator):
            return any(independent_frame(child.locator) for child in locator.children)
        return False

    # Explicit wall lines do not say where the outer walls are. Without a
    # physical outline, a field on an undeclared outer edge would be scored
    # against walls elsewhere: internal walls or other components' walls.
    binning = env.binning
    composition = composition_view(binning.locator)
    arena = env.arena
    while isinstance(arena, _TransformedArena):
        arena = arena.base
    if isinstance(arena, _CombinedArena) and not outline_declared(arena):
        missing = [
            index
            for index, child in enumerate(arena.children)
            if not outline_declared(child)
        ]
        keys = composition[0].keys if composition is not None else ()
        named = (
            ", ".join(repr(keys[index]) for index in missing)
            if len(keys) == len(arena.children)
            else f"{len(missing)} of {len(arena.children)}"
        )
        raise NotImplementedError(
            f"Physical border scores need a physical outline for every combined component; component(s) {named} "
            "have none (their outline was inferred), so fields at their edges would be scored against walls "
            "elsewhere (their own internal walls or walls in other components). Build each component with its "
            "measured outline and say what it is (Environment.from_polygon(...) for walls, "
            "OpenArena(outline, outline='drop') for an elevated track, or MaskArena(..., outline=...)) before "
            "combining, or explicitly use boundary_source='analysis'."
        )
    if not outline_declared(arena):
        raise NotImplementedError(
            "Physical border scores need the arena's outer boundary, but its outline was inferred rather than "
            "declared: only internal walls are declared, so a field on the outer edge would be scored against "
            "those walls alone. If the outline is the measured enclosure, say what it is, for example "
            "Environment.from_polygon(outline, bin_size, arena=OpenArena(outline, walls=...)) for a walled box "
            "or arena=OpenArena(outline, outline='drop') for an elevated platform. Otherwise explicitly "
            "use boundary_source='analysis' to score the retained analysis boundary."
        )
    if composition is not None and any(
        independent_frame(child.locator) for child in composition[0].children
    ):
        raise NotImplementedError(
            "Physical border contact needs a shared source frame for combined cells and walls. Combine components before applying a shared transform, or explicitly use boundary_source='analysis'."
        )

    try:
        cells = materialize_cells(binning)
    except ValueError as exc:
        raise NotImplementedError(
            "Physical border coverage needs exact planar cells; supply supported grid or polygon cells, or choose boundary_source='analysis'."
        ) from exc
    frame = wall_source_frame(binning.locator, env.arena, env.bin_centers)
    if frame is None:
        return list(cells), env.arena, None
    contact_cells, _walls, _centers, inverse, frames, _errors = frame
    forward = np.asarray(np.linalg.inv(inverse), dtype=np.float64)
    return list(contact_cells), frames[-1].base, forward


def _footprints(arena: Arena) -> list[shapely.Geometry]:
    from neurospatial.geometry._boundaries import _physical_footprints

    return _physical_footprints(arena)


def _pixel_footprints(arena: Arena) -> list[shapely.Geometry]:
    """Asserted footprints made of mask cells, in the arena's frame.

    Their outlines run along cell edges, so they are known to within a cell
    rather than as measured lines.
    """
    from neurospatial.geometry import mapped_geometry
    from neurospatial.geometry._arena import (
        MaskArena,
        _CombinedArena,
        _TransformedArena,
    )

    if isinstance(arena, _CombinedArena):
        return [part for child in arena.children for part in _pixel_footprints(child)]
    if isinstance(arena, _TransformedArena):
        return [
            mapped_geometry(part, arena.forward)
            for part in _pixel_footprints(arena.base)
        ]
    if isinstance(arena, MaskArena) and arena.outline is not None:
        domain = arena.polygonal_domain()
        return [] if domain is None else [domain]
    return []


def _polygon_rings(polygonal: shapely.Geometry) -> list[shapely.LineString]:
    """Exterior and hole rings of each polygon part, as closed lines.

    Each ring starts at its lowest vertex (smallest x, then y) and runs in a
    fixed direction. Overlay operations start an outline wherever their
    floating-point path takes them, which differs between platforms; a ring
    starting in the middle of a side keeps that collinear vertex through
    simplification, which shortens both neighboring edges and can hide their
    corners. No straight side passes through the lowest vertex.
    """
    rings = shapely.get_rings(shapely.get_parts(shapely.normalize(polygonal)))
    return [shapely.LineString(ring.coords) for ring in rings if ring.length > 0]


def _declared_walls(arena: Arena) -> list[list[shapely.Geometry]]:
    """Declared walls, as the segments of each name, in the arena's frame.

    Segments that share a name form one wall. Combined arenas name walls per
    component, so equal names in different components stay different walls.
    """
    from neurospatial.geometry import mapped_geometry
    from neurospatial.geometry._arena import _CombinedArena, _TransformedArena

    if isinstance(arena, _CombinedArena):
        return [group for child in arena.children for group in _declared_walls(child)]
    if isinstance(arena, _TransformedArena):
        return [
            [mapped_geometry(segment, arena.forward) for segment in group]
            for group in _declared_walls(arena.base)
        ]
    groups: dict[str, list[shapely.Geometry]] = {}
    for name, wall in zip(arena.wall_names, arena.walls, strict=True):
        groups.setdefault(name, []).append(wall)
    return list(groups.values())


def _resolution(cells: list[shapely.Geometry]) -> float:
    """The map's positional resolution: the median cell circumradius.

    The distance from a cell's centroid to its farthest vertex, half the
    diagonal for a square bin (see the module notes).
    """
    cell_array = np.asarray(cells, dtype=object)
    vertices, owner = shapely.get_coordinates(
        shapely.get_exterior_ring(cell_array), return_index=True
    )
    centers = shapely.get_coordinates(shapely.centroid(cell_array))
    reach = np.zeros(len(cell_array))
    np.maximum.at(reach, owner, np.linalg.norm(vertices - centers[owner], axis=1))
    return float(np.median(reach))


def _split_walls(
    rings: list[tuple[shapely.LineString, bool]],
    groups: list[list[shapely.Geometry]],
    *,
    resolution: float,
    scale: float,
    tolerance: float,
) -> tuple[list[shapely.Geometry], list[bool]]:
    """Walls from closed outlines and declared walls, and which are rings.

    ``rings`` pairs each closed outline with whether it is a measured line
    (an arena polygon), rather than one running along cell edges. A declared
    wall lying on an outline replaces that piece of the outline, and its ends
    cut the outline; other declared walls are internal walls.
    """
    boundary = unary_union([ring for ring, _measured in rings])
    on_outline: list[shapely.Geometry] = []
    internal: list[shapely.Geometry] = []
    for segments in groups:
        on = [s for s in segments if _lies_on(s, boundary, tolerance)]
        off = [s for s in segments if not _lies_on(s, boundary, tolerance)]
        on_outline.extend([shapely.line_merge(unary_union(on))] if on else [])
        internal.extend([shapely.line_merge(unary_union(off))] if off else [])
    declared = unary_union(on_outline) if on_outline else None
    walls: list[shapely.Geometry] = [*internal, *on_outline]
    closed = [False] * len(walls)
    for ring, measured in rings:
        cuts = _corner_positions(ring, resolution, scale, measured=measured)
        if declared is not None:
            ends = shapely.get_coordinates(
                shapely.boundary(shapely.get_parts(declared))
            )
            ends = ends[shapely.dwithin(ring, shapely.points(ends), tolerance)]
            cuts = np.concatenate(
                [cuts, shapely.line_locate_point(ring, shapely.points(ends))]
            )
        cuts = _unique_positions(cuts % ring.length, tolerance)
        pieces = _cut_ring(ring, cuts)
        if declared is not None:
            middles = shapely.line_interpolate_point(pieces, 0.5, normalized=True)
            pieces = [
                piece
                for piece, own in zip(
                    pieces, shapely.dwithin(declared, middles, tolerance), strict=True
                )
                if not own
            ]
        walls.extend(pieces)
        closed.extend([len(cuts) <= 1 and len(pieces) == 1] * len(pieces))
    return walls, closed


def _lies_on(
    segment: shapely.Geometry, boundary: shapely.Geometry, tolerance: float
) -> bool:
    """Whether a declared wall segment runs along an outline."""
    points = shapely.get_coordinates(shapely.segmentize(segment, segment.length / 4))
    return bool(np.all(shapely.dwithin(boundary, shapely.points(points), tolerance)))


def _unique_positions(
    positions: NDArray[np.float64], tolerance: float
) -> NDArray[np.float64]:
    """Sorted positions, dropping any within ``tolerance`` of the previous one."""
    positions = np.sort(positions)
    if positions.size == 0:
        return positions
    keep = np.concatenate([[True], np.diff(positions) > tolerance])
    return np.asarray(positions[keep], dtype=np.float64)


def _cut_ring(
    ring: shapely.LineString, cuts: NDArray[np.float64]
) -> list[shapely.Geometry]:
    """Pieces of a closed outline between consecutive cut positions."""
    length = ring.length
    if len(cuts) == 0:
        return [ring]
    walls = [substring(ring, a, b) for a, b in pairwise(cuts)]
    # The last wall runs from the last cut through the ring's start.
    wrap = [substring(ring, cuts[-1], length), substring(ring, 0.0, cuts[0])]
    walls.append(shapely.line_merge(unary_union([w for w in wrap if w.length > 0])))
    return [wall for wall in walls if wall.length > 0]


def _corner_positions(
    ring: shapely.LineString, resolution: float, scale: float, *, measured: bool
) -> NDArray[np.float64]:
    """Positions along a closed outline of its corners (see the module notes).

    Parameters
    ----------
    ring : shapely.LineString
        Closed outline.
    resolution : float
        The map's resolution ``epsilon``.
    scale : float
        Radius ``rho`` of the largest circle inside the outline's arena.
    measured : bool
        Whether the outline is a measured line, which can be rougher than
        the map's resolution; an outline along cell edges is known to within
        a cell.

    Returns
    -------
    ndarray of float64, shape (n_corners,)
        Distance along ``ring`` to each corner, sorted.
    """
    length = ring.length
    # Dropping collinear vertices keeps the path, so positions along it are
    # positions along ``ring``.
    coordinates = shapely.get_coordinates(shapely.simplify(ring, 0.0))
    steps = np.diff(coordinates, axis=0)
    step_length = np.linalg.norm(steps, axis=1)
    steps, step_length = steps[step_length > 0], step_length[step_length > 0]
    if len(steps) < 3 or resolution <= 0:
        return np.empty(0)
    vertex_position = np.concatenate([[0.0], np.cumsum(step_length)[:-1]])
    incoming, outgoing = np.roll(steps, 1, axis=0), steps
    incoming_length = np.roll(step_length, 1)

    # A measured outline is known no better than its roughness. Where it is
    # sampled more finely than the window (a hand trace, a buffered curve),
    # a vertex lies off the line through its neighbors by at most twice the
    # outline's error, so half the largest such offset is a lower bound on
    # that error. A longer edge is a measured wall, not a sample of one.
    window = min(2 * np.sqrt(resolution * scale), length / 4)
    sampled = np.maximum(incoming_length, step_length) < window
    if measured and sampled.any():
        chord = incoming + outgoing
        offset = np.abs(
            incoming[:, 0] * chord[:, 1] - incoming[:, 1] * chord[:, 0]
        ) / np.linalg.norm(chord, axis=1).clip(min=np.finfo(float).tiny)
        resolution = max(resolution, 0.5 * float(np.max(offset[sampled])))
        window = min(2 * np.sqrt(resolution * scale), length / 4)

    # Vertices that are corners at this resolution.
    vertex_turn = np.arctan2(
        incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0],
        np.sum(incoming * outgoing, axis=1),
    )
    shorter = np.minimum(incoming_length, step_length)
    sharp = 0.5 * shorter * np.tan(np.abs(vertex_turn) / 4) > resolution

    # Stretches turning faster than the inscribed circle, by more than the
    # map resolves, between chords of length ``window``.
    circle = 2 * np.arcsin(min(1.0, window / (2 * scale)))
    threshold = circle + 2 * np.arctan(2 * resolution / window)
    position = np.union1d(
        np.arange(0, length, min(resolution, window) / 4), vertex_position
    )
    position = position[position < length]
    here = shapely.get_coordinates(shapely.line_interpolate_point(ring, position))
    back = shapely.get_coordinates(
        shapely.line_interpolate_point(ring, (position - window) % length)
    )
    ahead = shapely.get_coordinates(
        shapely.line_interpolate_point(ring, (position + window) % length)
    )
    before, after = here - back, ahead - here
    turn = np.arctan2(
        before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0],
        np.sum(before * after, axis=1),
    )
    # A stretch turning faster than the inscribed circle is one corner when
    # somewhere it turns faster by more than the map resolves; measurement
    # error near that threshold then cannot break one corner into several.
    turning = np.where(np.abs(turn) > circle, np.sign(turn), 0.0)
    sharp_sample = np.zeros(len(position), dtype=bool)
    sharp_sample[np.searchsorted(position, vertex_position[sharp])] = True

    # Resolvable vertices are corners. A turning stretch without one (a
    # rounded or traced corner) is one corner, where it turns most.
    cuts = list(position[sharp_sample])
    if turning.any() and not turning.all():
        # Scan from a point that is not turning, so no stretch wraps around
        # the ring's start.
        order = np.roll(np.arange(len(position)), -int(np.flatnonzero(turning == 0)[0]))
        edges = np.flatnonzero(np.diff(turning[order]) != 0) + 1
        for run in np.split(order, edges):
            if (
                turning[run[0]] != 0
                and np.max(np.abs(turn[run])) > threshold
                and not sharp_sample[run].any()
            ):
                cuts.append(position[run[int(np.argmax(np.abs(turn[run])))]])
    return np.sort(np.asarray(cuts, dtype=float))


def _wall_atoms(
    env: Environment,
    walls: list[shapely.Geometry],
    cells: list[shapely.Geometry],
    forward: NDArray[np.float64] | None,
) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64], csr_array]:
    """Cut the walls into atoms owned by bins (see :class:`_BorderGeometry`).

    Refuses walls with a piece that no bin owns.

    Returns
    -------
    atom_wall, atom_start, atom_length : ndarray, shape (n_atoms,)
        Wall of each atom, its start along the wall and its length, in the
        environment's coordinates.
    atom_owners : scipy.sparse.csr_array, shape (n_atoms, n_bins)
    """
    cell_array = np.asarray(cells, dtype=object)
    covered = unary_union(cells)
    piece_length = _segment_length(cells)
    tolerance = 1e-9 * np.sqrt(covered.area)
    atom_wall, atom_start, atom_length, rows, columns = [], [], [], [], []
    unowned: list[shapely.Geometry] = []
    n_atoms = 0
    for index, wall in enumerate(walls):
        pieces = shapely.intersection(cell_array, wall)
        owners = np.flatnonzero(shapely.length(pieces) > 0)
        owned = list(pieces[owners])
        uncovered = wall.difference(covered)
        if uncovered.length > 0:
            segments = _short_segments(uncovered, piece_length)
            midpoints = shapely.get_coordinates(
                shapely.line_interpolate_point(segments, 0.5, normalized=True)
            )
            if forward is not None:
                midpoints = midpoints @ forward[:-1, :-1].T + forward[:-1, -1]
            sliver_owners = locate_outline_points(env.binning, env.arena, midpoints)
            found = sliver_owners >= 0
            owners = np.concatenate([owners, sliver_owners[found]])
            owned.extend(segments[found])
        lines, line_index = shapely.get_parts(
            np.asarray(owned, dtype=object), return_index=True
        )
        is_line = shapely.get_type_id(lines) == 1
        lines, line_owner = lines[is_line], owners[line_index[is_line]]
        ends = shapely.points(
            np.concatenate(
                [
                    shapely.get_coordinates(shapely.get_point(lines, 0)),
                    shapely.get_coordinates(shapely.get_point(lines, -1)),
                ]
            )
        )
        tree = shapely.STRtree(lines)
        offset = 0.0
        for part in shapely.get_parts(wall):
            part_length = part.length
            on_part = shapely.dwithin(part, ends, tolerance)
            breaks = _unique_positions(
                np.concatenate(
                    [[0.0, part_length], shapely.line_locate_point(part, ends[on_part])]
                ),
                tolerance,
            )
            breaks[-1] = part_length
            starts, stops = breaks[:-1], breaks[1:]
            middles = shapely.line_interpolate_point(part, 0.5 * (starts + stops))
            atom, piece = tree.query(middles, predicate="dwithin", distance=tolerance)
            ownerless = np.ones(len(starts), dtype=bool)
            ownerless[atom] = False
            unowned.extend(
                substring(part, a, b)
                for a, b in zip(starts[ownerless], stops[ownerless], strict=True)
            )
            if forward is None:
                lengths = stops - starts
            else:
                lengths = np.array(
                    [
                        _mapped(substring(part, a, b), forward).length
                        for a, b in zip(starts, stops, strict=True)
                    ]
                )
            atom_wall.append(np.full(len(starts), index))
            atom_start.append(offset + np.concatenate([[0.0], np.cumsum(lengths)[:-1]]))
            atom_length.append(lengths)
            rows.append(n_atoms + atom)
            columns.append(line_owner[piece])
            n_atoms += len(starts)
            offset += float(np.sum(lengths))
    if unowned:
        _refuse_unowned_walls(unowned, forward, env.units)
    owners_matrix = coo_array(
        (
            np.ones(sum(len(r) for r in rows)),
            (np.concatenate(rows), np.concatenate(columns)),
        ),
        shape=(n_atoms, env.n_bins),
    ).tocsr()
    owners_matrix.data[:] = 1.0  # a bin owning several pieces of one atom
    return (
        np.concatenate(atom_wall).astype(np.intp),
        np.concatenate(atom_start),
        np.concatenate(atom_length),
        owners_matrix,
    )


def _refuse_unowned_walls(
    pieces: list[shapely.Geometry],
    forward: NDArray[np.float64] | None,
    units: str | None,
) -> None:
    """Refuse walls that no bin owns, saying where and how much."""
    unowned = unary_union(pieces)
    if forward is not None:
        unowned = _mapped(unowned, forward)
    where = shapely.get_coordinates(
        shapely.line_interpolate_point(
            max(shapely.get_parts(unowned), key=lambda g: g.length),
            0.5,
            normalized=True,
        )
    )[0]
    unit = f" {units}" if units else " position units"
    raise NotImplementedError(
        f"Physical border scores need a bin at every measured wall, but {unowned.length:.3g}{unit} of wall "
        f"(for example near ({where[0]:.4g}, {where[1]:.4g})) lies outside every bin and not in a cell next "
        "to one. The bins stop short of the walls there: they were built from positions inside the measured "
        "outline, removed with subset(), or a corridor or notch of the outline is narrower than a bin. Build "
        "the bins from the arena outline, for example Environment.from_polygon(outline, bin_size), with bins "
        "smaller than the narrowest corridor; or explicitly use boundary_source='analysis' to score the "
        "outline of the retained bins."
    )


def _inscribed_radius(
    domain: shapely.Geometry, internal: tuple[shapely.Geometry, ...]
) -> float:
    """Radius of the largest circle inside ``domain`` that crosses no wall."""
    if internal:
        # Internal walls become thin holes, so the circle cannot cross them.
        thickness = 1e-9 * float(np.sqrt(domain.area))
        domain = domain.difference(unary_union(internal).buffer(thickness))
    return float(shapely.maximum_inscribed_circle(domain).length)


def _segment_length(
    cells: tuple[shapely.Geometry, ...] | list[shapely.Geometry],
) -> float:
    """A tenth of the smallest cell's side: sliver wall pieces stay in one cell."""
    return 0.1 * float(np.sqrt(np.min(shapely.area(np.asarray(cells, dtype=object)))))


def _short_segments(lines: shapely.Geometry, length: float) -> NDArray[np.object_]:
    """Split line geometry into two-point segments no longer than ``length``."""
    parts = shapely.get_parts(shapely.segmentize(shapely.line_merge(lines), length))
    segments = []
    for part in parts:
        coordinates = shapely.get_coordinates(part)
        segments.append(
            shapely.linestrings(np.stack([coordinates[:-1], coordinates[1:]], axis=1))
        )
    return np.concatenate(segments) if segments else np.empty(0, dtype=object)


def _distance_from_walls(
    env: Environment, direct: NDArray[np.float64], wall_bins: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Center-to-wall length at the wall bins, then paths through the graph.

    Contact with a wall is a seed, not an invented zero-distance bin.
    Unrepresented routes are unknown, so unreachable bins are ``inf``.
    """
    distances = np.full(env.n_bins, np.inf)
    seeds = np.flatnonzero(wall_bins)
    if seeds.size == 0:
        return distances
    graph = env.connectivity.copy()
    source = object()
    graph.add_weighted_edges_from(
        [(source, int(index), direct[index]) for index in seeds], weight="distance"
    )
    try:
        paths = nx.single_source_dijkstra_path_length(graph, source, weight="distance")
    except (ValueError, nx.NetworkXException) as exc:
        raise RuntimeError(
            "border_score: geodesic wall distance computation failed; check navigation distances or use metric='euclidean'."
        ) from exc
    for node, length in paths.items():
        if node is not source:
            distances[int(node)] = length
    return distances


def observed_lengths(
    geometry: _BorderGeometry, firing_rate: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Length of each atom that the rate map observes, else 0.

    Returns
    -------
    ndarray of float64, shape (n_atoms,)
    """
    finite = np.isfinite(firing_rate).astype(float)
    seen = np.asarray(geometry.atom_owners @ finite) > 0
    return np.asarray(geometry.atom_length * seen, dtype=np.float64)


def _wall_coverage(
    geometry: _BorderGeometry,
    fields: list[NDArray[np.intp]],
    firing_rate: NDArray[np.float64],
) -> float:
    """Largest fraction of one wall's observed length that one field covers.

    NaN when no wall has an observed piece. Unobserved pieces of a wall are
    missing, not uncovered: each wall's coverage is the covered fraction of
    its observed length.
    """
    n_bins = len(firing_rate)
    in_field = np.zeros((n_bins, len(fields)))
    for column, field in enumerate(fields):
        in_field[field, column] = 1.0
    observed = observed_lengths(geometry, firing_rate)
    # A field bin has a finite rate, so covered atoms are observed.
    covered = geometry.atom_length[:, None] * (geometry.atom_owners @ in_field > 0)
    n_walls = len(geometry.wall_lengths)
    best = np.full(n_walls, np.nan)
    for wall in range(n_walls):
        atoms = geometry.atom_wall == wall
        if geometry.closed[wall]:
            best[wall] = _half_ring_coverage(
                geometry.atom_start[atoms],
                geometry.wall_lengths[wall],
                covered[atoms],
                observed[atoms],
            )
            continue
        seen = observed[atoms].sum()
        if seen > 0:
            best[wall] = np.max(covered[atoms].sum(axis=0)) / seen
    return float(np.nanmax(best)) if np.isfinite(best).any() else float("nan")


def _half_ring_coverage(
    start: NDArray[np.float64],
    perimeter: float,
    covered: NDArray[np.float64],
    observed: NDArray[np.float64],
) -> float:
    """Largest covered fraction of the observed length in any half of a ring.

    Parameters
    ----------
    start : ndarray of float64, shape (n_atoms,)
        Contiguous atoms' starts along the ring, increasing from 0.
    perimeter : float
        The ring's length.
    covered : ndarray of float64, shape (n_atoms, n_fields)
        Length of each atom that each field covers.
    observed : ndarray of float64, shape (n_atoms,)
        Length of each atom the map observes.

    Notes
    -----
    Covered and observed lengths in the window ``[t, t + perimeter / 2)`` are
    linear in ``t`` between atom ends and atom ends less half the perimeter,
    so their ratio is monotone between those points and its maximum is at
    one of them.
    """
    half = perimeter / 2
    bounds = np.concatenate([start, [perimeter]])

    def cumulative(values: NDArray[np.float64], at: NDArray[np.float64]) -> NDArray:
        """Integral of ``values`` (per atom) from 0 to ``at``, over two turns."""
        totals = np.concatenate(
            [np.zeros((1, *values.shape[1:])), np.cumsum(values, axis=0)]
        )
        turns, place = np.divmod(at, perimeter)
        atom = np.clip(
            np.searchsorted(bounds, place, side="right") - 1, 0, len(start) - 1
        )
        widths = bounds[atom + 1] - bounds[atom]
        fraction = np.divide(
            place - bounds[atom], widths, out=np.zeros_like(place), where=widths > 0
        )
        if values.ndim == 2:
            fraction, turns = fraction[:, None], turns[:, None]
        integral: NDArray[np.float64] = (
            totals[atom]
            + fraction * (totals[atom + 1] - totals[atom])
            + turns * totals[-1]
        )
        return integral

    windows = np.unique(np.concatenate([start, (start - half) % perimeter]))
    seen = cumulative(observed, windows + half) - cumulative(observed, windows)
    hit = cumulative(covered, windows + half) - cumulative(covered, windows)
    usable = seen > 0
    if not usable.any():
        return float("nan")
    return float(np.max(hit[usable] / seen[usable, None]))


def _border_score(
    fields: list[NDArray[np.intp]],
    firing_rate: NDArray[np.float64],
    metric: DistanceMetric,
    geometry: _BorderGeometry,
) -> float:
    """Solstad border score of the fields, with these walls.

    Coverage is the largest fraction of one wall's observed length that one
    field covers (on a ring, of its best half). Distance is the
    firing-rate-weighted mean distance from all field bins to the nearest
    wall, over the inscribed radius. Geodesic distances include a
    center-to-wall seed length and then paths through the retained navigation
    graph; any unreachable field bin makes the score NaN.
    """
    coverage = _wall_coverage(geometry, fields, firing_rate)
    if np.isnan(coverage):
        return float("nan")
    bins = np.concatenate(fields)
    if metric == "euclidean":
        distances = geometry.direct[bins]
    else:
        distances = geometry.geodesic[bins]
        if not np.all(np.isfinite(distances)):
            return float("nan")
    weights = firing_rate[bins]
    distance = float(np.sum(weights * distances) / np.sum(weights)) / geometry.radius
    denominator = coverage + distance
    if denominator == 0:
        return float("nan")
    return float((coverage - distance) / denominator)
