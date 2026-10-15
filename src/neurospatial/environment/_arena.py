"""Infer continuous domains from supplied geometry without inventing walls."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from neurospatial.arena import (
    Arena,
    EgocentricSurround,
    MaskArena,
    OpenArena,
    TrackArena,
    UnknownArena,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from shapely.geometry import LineString, Polygon

    from neurospatial.layout._binning import Binning, Locator
    from neurospatial.layout.base import LayoutEngine


def infer_arena(binning: Binning, *, polar: bool, layout: LayoutEngine) -> Arena:
    """Use declared shapes or an inferred cell domain, without asserting walls."""
    n_dims = binning.bins.n_dims
    grid = binning.rectilinear
    if binning.bins.n_bins == 0:
        return UnknownArena(n_dims, coordinate_kind="polar" if polar else "cartesian")
    if polar:
        if grid is not None:
            return EgocentricSurround(
                (float(grid.edges[0][0]), float(grid.edges[0][-1])),
                (float(grid.edges[1][0]), float(grid.edges[1][-1])),
                circular=bool(layout._build_params_used.get("circular_angle", False)),
            )
        return UnknownArena(n_dims, coordinate_kind="polar")
    track = binning.linearization
    if track is not None:
        graph = getattr(track, "graph", None)
        if graph is not None:
            segments = np.array(
                [
                    [graph.nodes[u]["pos"], graph.nodes[v]["pos"]]
                    for u, v in graph.edges
                ],
                dtype=np.float64,
            )
            return TrackArena(segments, width=None)
        return UnknownArena(n_dims)
    if binning.outline is not None:
        return OpenArena(binning.outline.geometry)
    if grid is not None and len(grid.shape) == n_dims:
        return MaskArena(grid.edges, grid.active_mask)
    # Exact area cells are sufficient to infer an analysis domain, but not
    # measured physical walls. Keep unsupported custom layouts usable.
    if binning.shapes is not None and n_dims == 2:
        from neurospatial.layout._binning import LayoutLocator

        if isinstance(binning.locator, LayoutLocator):
            return UnknownArena(n_dims)
        from shapely.ops import unary_union

        from neurospatial.layout._polygon import materialize_cells

        return OpenArena(
            unary_union(materialize_cells(binning)), physical_footprint_known=False
        )
    return UnknownArena(n_dims)


def validate_arena(arena: Arena, binning: Binning, *, polar: bool) -> None:
    """Reject a domain whose coordinate semantics disagree with its bins."""
    if arena.n_dims != binning.bins.n_dims:
        raise ValueError("Arena coordinate dimensions must match the binning.")
    kind = "polar" if polar else "cartesian"
    if arena.coordinate_kind != kind:
        raise ValueError(
            f"Arena coordinate_kind must be {kind!r} for this environment."
        )
    if arena.domain_known and not np.all(arena.contains(binning.bins.centers)):
        raise ValueError(
            "Arena excludes a bin center; select bins by center before constructing the environment."
        )
    from neurospatial.layout._polygon import PolygonCells

    if isinstance(binning.shapes, PolygonCells):
        domain = arena.polygonal_domain()
        if domain is not None:
            from neurospatial.layout._binning_layout import _polygon_area_allowance

            domain_clearance = domain.minimum_clearance
            for index, cell in enumerate(binning.shapes.polygons):
                outside = cell.difference(domain).area
                if outside == 0:
                    continue
                allowance = _polygon_area_allowance(
                    cell, reference_clearance=domain_clearance
                )
                if outside > allowance:
                    raise ValueError(
                        f"Polygon cell {index} extends outside the saved arena domain."
                    )


def _wall_source_frame(
    locator: Locator, arena: Arena, centers: NDArray[np.float64]
) -> tuple[tuple[Polygon, ...], tuple[LineString, ...], NDArray[np.float64]] | None:
    """Recover a shared stored frame through retained cell-owner mappings."""
    from neurospatial.arena import _TransformedArena
    from neurospatial.layout._locators import SubsetLocator, TransformedLocator
    from neurospatial.layout._polygon import ClippedLocator, cells_from_locator

    if isinstance(locator, TransformedLocator) and isinstance(arena, _TransformedArena):
        if np.array_equal(locator.inverse, arena.inverse):
            original = (centers - arena.forward[:-1, -1]) @ arena.inverse[:-1, :-1].T
            inner = _wall_source_frame(locator.base, arena.base, original)
            return (
                inner
                if inner is not None
                else (cells_from_locator(locator.base), arena.base.walls, original)
            )
    elif isinstance(locator, (SubsetLocator, ClippedLocator)):
        inner = _wall_source_frame(locator.base, arena, centers)
        if inner is not None:
            cells, walls, original = inner
            if isinstance(locator, SubsetLocator):
                selected = np.flatnonzero(locator.parent_to_child >= 0)
                selected = selected[np.argsort(locator.parent_to_child[selected])]
            else:
                selected = locator.mapping.parent_bin_indices
            return tuple(cells[index] for index in selected), walls, original
    return None


def apply_walls(
    binning: Binning, graph: nx.Graph, arena: Arena
) -> tuple[Binning, nx.Graph]:
    """Remove complete blocked cell interfaces from paths and finite-volume faces."""
    if not arena.walls:
        return binning, graph
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    from neurospatial.layout._binning import KnownFaces, SpatialBins
    from neurospatial.layout._diffusion import FaceGeometry
    from neurospatial.layout._polygon import materialize_cells

    wall_frame = _wall_source_frame(binning.locator, arena, binning.bins.centers)
    if wall_frame is None:
        cells, wall_segments, centers = (
            materialize_cells(binning),
            arena.walls,
            binning.bins.centers,
        )
    else:
        # Shared snapshots avoid independently rounded collinear vertices.
        # Subset/clip descendants inherit the same source-cell wall decisions.
        cells, wall_segments, centers = wall_frame
    walls = unary_union(wall_segments)
    for index, cell in enumerate(cells):
        interior_part = walls.intersection(cell).difference(cell.boundary)
        if interior_part.length > 0:
            raise ValueError(
                f"Physical wall cuts through bin {index}. Refine or rebuild cells so walls follow complete cell faces."
            )
    keep = np.ones(len(binning.bins.pairs), dtype=bool)
    for index, (u, v) in enumerate(binning.bins.pairs):
        contact = cells[u].boundary.intersection(cells[v].boundary)
        blocked = contact.intersection(walls)
        if contact.length > 0 and blocked.length > 0:
            if blocked.length != contact.length:
                raise ValueError(
                    f"Physical wall covers only part of face ({u}, {v}). Align doorway endpoints with cell vertices."
                )
            keep[index] = False
        elif contact.length == 0:
            link = LineString(centers[[u, v]])
            keep[index] = contact.intersection(walls).is_empty and not link.crosses(
                walls
            )
    if np.all(keep):
        return binning, graph
    bins = binning.bins
    updated = SpatialBins(
        bins.centers,
        bins.sizes,
        bins.measure_dim,
        bins.pairs[keep],
        bins.pair_distances[keep],
    )
    faces = binning.faces
    if faces.available:
        source = faces.build()
        faces = KnownFaces(
            FaceGeometry(
                updated.pairs, source.A[keep], source.distance[keep], updated.sizes
            )
        )
    result_graph = graph.copy()
    result_graph.remove_edges_from(bins.pairs[~keep])
    return replace(binning, bins=updated, faces=faces), result_graph
