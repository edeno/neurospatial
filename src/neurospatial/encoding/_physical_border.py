"""Wall geometry for border scores: individual walls, wall contact and distances."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.ops import substring, unary_union

from neurospatial.environment._cache import CacheKey
from neurospatial.geometry import locate_outline_points, materialize_cells

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.geometry._arena import Arena
    from neurospatial.geometry._boundaries import BoundarySource
    from neurospatial.ops._types import DistanceMetric

# An outline turning by more than this within two bin widths has a corner
# there. Square-bin staircases turn by at most 45 degrees over two bin widths
# and regular polygons with eight or more sides by at most 45 degrees per
# vertex, so a circle, an octagon or a pixel-traced curve is one wall, while
# triangles, rectangles, pentagons and hexagons have one wall per side.
CORNER_ANGLE = np.radians(50.0)


@dataclass(frozen=True)
class _BorderGeometry:
    """Rate-independent wall geometry of one environment, in its coordinates.

    Attributes
    ----------
    walls : shapely geometry
        All walls together.
    wall_lengths : ndarray of float64, shape (n_walls,)
        Length of each individual wall.
    contact : tuple of (ndarray of intp, tuple of shapely geometry), length n_walls
        For each wall, the bins that own pieces of it and those pieces: a
        bin's cell contact with the wall, plus wall pieces in outline slivers,
        which belong to the bin that owns positions there.
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
    contact: tuple[tuple[NDArray[np.intp], tuple[shapely.Geometry, ...]], ...]
    wall_bins: NDArray[np.bool_]
    direct: NDArray[np.float64]
    geodesic: NDArray[np.float64]
    radius: float


_COVERAGE_UNAVAILABLE = (
    "Physical-wall coverage is unavailable: some measured wall segments have no retained cells "
    "with finite firing rates. Build bins that reach every measured wall and collect sufficient "
    "observations there. For rotated rectangular arenas, build bins in the arena-aligned frame "
    "and transform the bins, arena and observations together. Adding OpenArena or clipping "
    "existing bins cannot supply missing bins or observations. Alternatively, explicitly use "
    "boundary_source='analysis' to score the retained analysis boundary."
)


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
    from neurospatial.geometry import boundary_geometry, mapped_geometry

    # Validates the source, coordinates and dimensionality, and raises when
    # the walls are unknown.
    boundary_geometry(env.binning, env.arena, source)
    if source == "physical":
        cells, source_arena, forward = _physical_frame(env)
        window = _corner_window(cells)
        named = _named_walls(source_arena)
        rings = _footprint_rings(source_arena)
        domain = unary_union(_footprints(env.arena))
    else:
        try:
            cells = list(materialize_cells(env.binning))
        except ValueError as exc:
            raise NotImplementedError(
                "Analysis-boundary border scores need exact planar cells; use a 2-D "
                "grid, hexagonal layout, triangular mesh or polygon cells."
            ) from exc
        forward = None
        window = _corner_window(cells)
        domain = unary_union(cells)
        if env.arena.domain_known:
            restriction = env.arena.polygonal_domain()
            if restriction is not None:
                domain = domain.intersection(restriction)
        # Only the parts of internal walls among the retained bins bound them.
        clipped = (wall.intersection(domain) for wall in _named_walls(env.arena))
        named = [wall for wall in clipped if wall.length > 0]
        rings = _polygon_rings(domain)
    walls = [
        *named,
        *(wall for ring in rings for wall in _split_at_corners(ring, window)),
    ]

    cell_array = np.asarray(cells, dtype=object)
    covered = unary_union(cells)
    piece_length = _segment_length(cells)
    contact = []
    for wall in walls:
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
            if np.any(sliver_owners < 0):
                raise NotImplementedError(_COVERAGE_UNAVAILABLE)
            owners = np.concatenate([owners, sliver_owners])
            owned.extend(segments)
        if forward is not None:
            owned = [mapped_geometry(piece, forward) for piece in owned]
        contact.append((owners.astype(np.intp), tuple(owned)))
    env_walls = (
        walls if forward is None else [mapped_geometry(w, forward) for w in walls]
    )
    wall_union = unary_union(env_walls)
    wall_bins = np.zeros(env.n_bins, dtype=bool)
    for owners, _pieces in contact:
        wall_bins[owners] = True
    direct = np.asarray(
        shapely.distance(shapely.points(env.bin_centers), wall_union), dtype=float
    )
    return _BorderGeometry(
        walls=wall_union,
        wall_lengths=np.asarray(shapely.length(env_walls), dtype=float),
        contact=tuple(contact),
        wall_bins=wall_bins,
        direct=direct,
        geodesic=_distance_from_walls(env, direct, wall_bins),
        radius=_inscribed_radius(domain, env.arena.walls),
    )


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
    # known footprint, a field on an undeclared outer edge would be scored
    # against walls elsewhere: internal walls or other components' walls.
    binning = env.binning
    composition = composition_view(binning.locator)
    arena = env.arena
    while isinstance(arena, _TransformedArena):
        arena = arena.base
    if isinstance(arena, _CombinedArena) and not arena.physical_footprint_known:
        missing = [
            index
            for index, child in enumerate(arena.children)
            if not child.physical_footprint_known
        ]
        keys = composition[0].keys if composition is not None else ()
        named = (
            ", ".join(repr(keys[index]) for index in missing)
            if len(keys) == len(arena.children)
            else f"{len(missing)} of {len(arena.children)}"
        )
        raise NotImplementedError(
            f"Physical border scores need a measured footprint for every combined component; component(s) {named} "
            "have none, so fields at their edges would be scored against walls elsewhere (their own internal walls "
            "or walls in other components). Build each component with its measured outline "
            "(Environment.from_polygon(...), OpenArena(outline, physical_footprint_known=True) or "
            "MaskArena(physical_footprint_known=True)) before combining, or explicitly use boundary_source='analysis'."
        )
    if not arena.physical_footprint_known:
        raise NotImplementedError(
            "Physical border scores need the arena's outer walls, but its physical footprint is unknown: only "
            "internal walls are declared, so a field on the outer edge would be scored against those walls alone. "
            "If the arena outline is the measured enclosure, declare it with "
            "OpenArena(outline, walls=..., physical_footprint_known=True), for example "
            "Environment.from_polygon(outline, bin_size, arena=OpenArena(outline, walls=...)). Otherwise explicitly "
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


def _footprint_rings(arena: Arena) -> list[shapely.LineString]:
    """The rings of every asserted footprint polygon, in the arena's frame."""
    return [ring for polygon in _footprints(arena) for ring in _polygon_rings(polygon)]


def _polygon_rings(polygonal: shapely.Geometry) -> list[shapely.LineString]:
    """Exterior and hole rings of each polygon part, as closed lines."""
    rings = shapely.get_rings(shapely.get_parts(polygonal))
    return [shapely.LineString(ring.coords) for ring in rings if ring.length > 0]


def _named_walls(arena: Arena) -> list[shapely.Geometry]:
    """Declared internal walls, one geometry per name, in the arena's frame.

    Segments that share a name form one wall. Combined arenas name walls per
    component, so equal names in different components stay different walls.
    """
    from neurospatial.geometry import mapped_geometry
    from neurospatial.geometry._arena import _CombinedArena, _TransformedArena

    if isinstance(arena, _CombinedArena):
        return [wall for child in arena.children for wall in _named_walls(child)]
    if isinstance(arena, _TransformedArena):
        return [
            mapped_geometry(wall, arena.forward) for wall in _named_walls(arena.base)
        ]
    groups: dict[str, list[shapely.Geometry]] = {}
    for name, wall in zip(arena.wall_names, arena.walls, strict=True):
        groups.setdefault(name, []).append(wall)
    return [shapely.line_merge(unary_union(segments)) for segments in groups.values()]


def _corner_window(cells: list[shapely.Geometry]) -> float:
    """Two typical bin widths: the scale over which an outline turn is a corner."""
    return 2.0 * float(
        np.sqrt(np.median(shapely.area(np.asarray(cells, dtype=object))))
    )


def _split_at_corners(
    ring: shapely.LineString, window: float
) -> list[shapely.LineString]:
    """Split a closed outline into walls at its corners.

    At each point of the outline, the turn is the angle between the chord
    from ``window`` back along the outline to the point and the chord from the
    point to ``window`` ahead. Consecutive points turning by more than
    :data:`CORNER_ANGLE` are one corner, placed where the turn is largest.
    Chords, unlike single edges, see through the staircase of square bins.
    """
    length = ring.length
    if window <= 0 or length <= 2 * window:
        return [ring]
    points = shapely.get_coordinates(shapely.segmentize(ring, window / 8))[:-1]
    position = shapely.line_locate_point(ring, shapely.points(points))
    before = shapely.get_coordinates(
        shapely.line_interpolate_point(ring, (position - window) % length)
    )
    after = shapely.get_coordinates(
        shapely.line_interpolate_point(ring, (position + window) % length)
    )
    incoming = points - before
    outgoing = after - points
    turn = np.abs(
        np.arctan2(
            incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0],
            np.sum(incoming * outgoing, axis=1),
        )
    )
    corner = turn > CORNER_ANGLE
    if not corner.any() or corner.all():
        return [ring]
    # Start the scan at a point that is not a corner, so no run of corner
    # points wraps around the ring's start.
    order = np.roll(np.arange(len(points)), -int(np.flatnonzero(~corner)[0]))
    flags = np.concatenate([[False], corner[order], [False]]).astype(np.int8)
    starts = np.flatnonzero(np.diff(flags) == 1)
    stops = np.flatnonzero(np.diff(flags) == -1)
    cuts = np.sort(
        [
            position[order[start + int(np.argmax(turn[order[start:stop]]))]]
            for start, stop in zip(starts, stops, strict=True)
        ]
    )
    walls = [substring(ring, a, b) for a, b in pairwise(cuts)]
    # The last wall runs from the last corner through the ring's start.
    wrap = [substring(ring, cuts[-1], length), substring(ring, 0.0, cuts[0])]
    walls.append(shapely.line_merge(unary_union([w for w in wrap if w.length > 0])))
    return [wall for wall in walls if wall.length > 0]


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


def _require_wall_rates(
    geometry: _BorderGeometry, firing_rate: NDArray[np.float64]
) -> None:
    """Refuse a rate map with no finite rate at some wall-owning bin."""
    if np.any(geometry.wall_bins & ~np.isfinite(firing_rate)):
        raise NotImplementedError(_COVERAGE_UNAVAILABLE)


def _border_score(
    fields: list[NDArray[np.intp]],
    firing_rate: NDArray[np.float64],
    metric: DistanceMetric,
    geometry: _BorderGeometry,
) -> float:
    """Solstad border score of the fields, with these walls.

    Coverage is the largest fraction of one wall that one field covers.
    Distance is the firing-rate-weighted mean distance from all field bins to
    the nearest wall, over the inscribed radius. Geodesic distances include a
    center-to-wall seed length and then paths through the retained navigation
    graph; any unreachable field bin makes the score NaN.
    """
    coverage = 0.0
    for field in fields:
        for (owners, pieces), length in zip(
            geometry.contact, geometry.wall_lengths, strict=True
        ):
            selected = [
                piece
                for piece, inside in zip(pieces, np.isin(owners, field), strict=True)
                if inside
            ]
            if selected:
                coverage = max(coverage, unary_union(selected).length / length)
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
