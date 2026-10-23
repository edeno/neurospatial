"""Border scores from measured wall contact and distance to actual walls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import networkx as nx
import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.ops import unary_union

from neurospatial.environment._cache import CacheKey
from neurospatial.geometry import locate_outline_points, materialize_cells

if TYPE_CHECKING:
    from neurospatial.environment import Environment


@dataclass(frozen=True)
class _PhysicalBorderGeometry:
    """Rate-independent wall geometry of one environment, in its coordinates.

    Attributes
    ----------
    walls : shapely geometry
        The measured walls.
    contact : tuple of tuple of shapely geometry, length n_bins
        Pieces of wall each bin owns: its cell's contact with the walls, plus
        wall pieces in outline slivers, which belong to the bin that owns
        positions there.
    wall_bins : ndarray of bool, shape (n_bins,)
        Bins that own some wall.
    direct : ndarray of float64, shape (n_bins,)
        Straight-line distance from each bin center to the walls.
    geodesic : ndarray of float64, shape (n_bins,)
        Path length from the walls through the navigation graph, seeded at the
        wall-owning bins; ``inf`` where no path exists.
    extent : float
        Largest distance between two points of the walls.
    """

    walls: shapely.Geometry
    contact: tuple[tuple[shapely.Geometry, ...], ...]
    wall_bins: NDArray[np.bool_]
    direct: NDArray[np.float64]
    geodesic: NDArray[np.float64]
    extent: float


_COVERAGE_UNAVAILABLE = (
    "Physical-wall coverage is unavailable: some measured wall segments have no retained cells "
    "with finite firing rates. Build bins that reach every measured wall and collect sufficient "
    "observations there. For rotated rectangular arenas, build bins in the arena-aligned frame "
    "and transform the bins, arena and observations together. Adding OpenArena or clipping "
    "existing bins cannot supply missing bins or observations. Alternatively, explicitly use "
    "boundary_source='analysis' to score the retained analysis boundary."
)


PHYSICAL_BORDER: CacheKey[_PhysicalBorderGeometry] = CacheKey("physical_border")


def physical_border_geometry(
    env: Environment, walls: shapely.Geometry
) -> _PhysicalBorderGeometry:
    """The environment's prepared wall geometry, built once and cached.

    ``walls`` is the environment's own measured boundary, so it is the same
    for every rate map of the environment.
    """
    return env._cache.get(PHYSICAL_BORDER, lambda: _prepare_physical_border(env, walls))


def _prepare_physical_border(
    env: Environment, walls: shapely.Geometry
) -> _PhysicalBorderGeometry:
    """Wall ownership and distances, refusing walls no bin can own.

    Independent of any firing rate, so an environment prepares it once (see
    :func:`physical_border_geometry`).
    """
    from neurospatial.geometry import (
        ClippedLocator,
        CombinedLocator,
        SubsetLocator,
        TransformedLocator,
        _CombinedArena,
        _TransformedArena,
        composition_view,
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
    from neurospatial.geometry import physical_walls, wall_source_frame

    frame = wall_source_frame(binning.locator, env.arena, env.bin_centers)
    contact_cells, contact_walls = cells, walls
    forward = None
    if frame is not None:
        contact_cells, _internal, _centers, inverse, frames, _errors = frame
        contact_walls = physical_walls(frames[-1].base)
        forward = np.asarray(np.linalg.inv(inverse), dtype=np.float64)

    n_bins = env.n_bins
    owned: list[list[shapely.Geometry]] = [[] for _ in range(n_bins)]
    pieces = shapely.intersection(
        np.asarray(contact_cells, dtype=object), contact_walls
    )
    for index in np.flatnonzero(shapely.length(pieces) > 0):
        owned[index].append(pieces[index])
    uncovered = contact_walls.difference(unary_union(contact_cells))
    if uncovered.length > 0:
        segments = _short_segments(uncovered, _segment_length(contact_cells))
        midpoints = shapely.get_coordinates(
            shapely.line_interpolate_point(segments, 0.5, normalized=True)
        )
        if forward is not None:
            midpoints = midpoints @ forward[:-1, :-1].T + forward[:-1, -1]
        owners = locate_outline_points(env.binning, env.arena, midpoints)
        if np.any(owners < 0):
            raise NotImplementedError(_COVERAGE_UNAVAILABLE)
        for segment, owner in zip(segments, owners, strict=True):
            owned[int(owner)].append(segment)
    if forward is not None:
        from neurospatial.geometry import mapped_geometry

        owned = [
            [mapped_geometry(piece, forward) for piece in bin_pieces]
            for bin_pieces in owned
        ]
    wall_bins = np.array([bool(bin_pieces) for bin_pieces in owned])
    direct = np.asarray(
        shapely.distance(shapely.points(env.bin_centers), walls), dtype=float
    )
    geodesic = _distance_from_walls(env, direct, wall_bins)
    # The wall diameter is invariant to rigid frame changes. Work one hull
    # vertex at a time instead of allocating a dense all-pairs distance array.
    hull = shapely.get_coordinates(walls.convex_hull)
    extent = max(float(np.hypot.reduce(hull - point, axis=1).max()) for point in hull)
    return _PhysicalBorderGeometry(
        walls,
        tuple(tuple(bin_pieces) for bin_pieces in owned),
        wall_bins,
        direct,
        geodesic,
        extent,
    )


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
            "border_score: physical-wall geodesic computation failed; check navigation distances or use metric='euclidean'."
        ) from exc
    for node, length in paths.items():
        if node is not source:
            distances[int(node)] = length
    return distances


def _require_wall_rates(
    geometry: _PhysicalBorderGeometry, firing_rate: NDArray[np.float64]
) -> None:
    """Refuse a rate map with no finite rate at some wall-owning bin."""
    if np.any(geometry.wall_bins & ~np.isfinite(firing_rate)):
        raise NotImplementedError(_COVERAGE_UNAVAILABLE)


def _physical_border_score(
    field_bins: NDArray[np.int64],
    metric: Literal["geodesic", "euclidean"],
    geometry: _PhysicalBorderGeometry,
) -> float:
    """Use observed wall-length coverage and mean center-to-wall distance.

    Geodesic distances include a center-to-wall seed length and then paths
    through the retained navigation graph. Unrepresented routes are unknown,
    not zero distance, and any unreachable field bin makes that score NaN.
    """
    contact = unary_union(
        [piece for index in field_bins for piece in geometry.contact[index]]
    )
    coverage = contact.length / geometry.walls.length
    if metric == "euclidean":
        distances = geometry.direct[field_bins]
    else:
        if not geometry.wall_bins.any():
            return float("nan")
        distances = geometry.geodesic[field_bins]
        if not np.all(np.isfinite(distances)):
            return float("nan")
    normalized = float(np.mean(distances)) / geometry.extent
    denominator = coverage + normalized
    if denominator == 0:
        return float("nan")
    return float(np.clip((coverage - normalized) / denominator, -1, 1))
