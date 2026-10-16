"""Infer continuous domains from supplied geometry without inventing walls."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from neurospatial.geometry._arena import (
    Arena,
    EgocentricSurround,
    MaskArena,
    OpenArena,
    TrackArena,
    UnknownArena,
    _CombinedArena,
    encode_arena,
)
from neurospatial.geometry._wall_frame import _wall_source_frame as _wall_source_frame

if TYPE_CHECKING:
    from neurospatial.geometry._binning import Binning


def infer_arena(binning: Binning, *, polar: bool, circular: bool = False) -> Arena:
    """Use declared shapes or an inferred cell domain, without asserting walls."""
    combined = _composition_arena(binning.locator)
    if combined is not None:
        return combined
    n_dims = binning.bins.n_dims
    grid = binning.rectilinear
    if binning.bins.n_bins == 0:
        return UnknownArena(n_dims, coordinate_kind="polar" if polar else "cartesian")
    if polar:
        if grid is not None:
            return EgocentricSurround(
                (float(grid.edges[0][0]), float(grid.edges[0][-1])),
                (float(grid.edges[1][0]), float(grid.edges[1][-1])),
                circular=circular,
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
    if binning.grid_frame is not None:
        frame = binning.grid_frame
        return MaskArena(frame.source.edges, frame.source.active_mask).transformed(
            frame.forward
        )
    if grid is not None and len(grid.shape) == n_dims:
        return MaskArena(grid.edges, grid.active_mask)
    # Exact area cells are sufficient to infer an analysis domain, but not
    # measured physical walls. Keep unsupported custom layouts usable.
    if binning.shapes is not None and n_dims == 2:
        from shapely.ops import unary_union

        from neurospatial.geometry._polygon import materialize_cells

        return OpenArena(
            unary_union(materialize_cells(binning)), physical_footprint_known=False
        )
    return UnknownArena(n_dims)


def _composition_arena(locator: object) -> Arena | None:
    from neurospatial.geometry._composition import CombinedLocator
    from neurospatial.geometry._locators import SubsetLocator, TransformedLocator
    from neurospatial.geometry._polygon import ClippedLocator

    if isinstance(locator, CombinedLocator):
        return _CombinedArena(locator.arenas)
    if isinstance(locator, (SubsetLocator, TransformedLocator, ClippedLocator)):
        arena = _composition_arena(locator.base)
        if arena is not None and isinstance(locator, TransformedLocator):
            arena = arena.transformed(
                np.asarray(np.linalg.inv(locator.inverse), dtype=np.float64)
            )
        return arena
    return None


def validate_arena(arena: Arena, binning: Binning, *, polar: bool) -> None:
    """Reject a domain whose coordinate semantics disagree with its bins."""
    combined_arena = _composition_arena(binning.locator)
    if combined_arena is not None:
        from neurospatial.geometry._arena import _TransformedArena
        from neurospatial.geometry._composition import CombinedLocator
        from neurospatial.geometry._locators import SubsetLocator, TransformedLocator
        from neurospatial.geometry._polygon import ClippedLocator

        def matches(locator: object, candidate: Arena) -> bool:
            if isinstance(locator, (SubsetLocator, ClippedLocator)):
                return matches(locator.base, candidate)
            if isinstance(locator, TransformedLocator):
                return (
                    isinstance(candidate, _TransformedArena)
                    and np.array_equal(locator.inverse, candidate.inverse)
                    and matches(locator.base, candidate.base)
                )
            if not isinstance(locator, CombinedLocator):
                return False
            for child, child_arena in zip(
                locator.children, locator.arenas, strict=True
            ):
                validate_arena(child_arena, child, polar=False)
                if child_arena.walls:
                    graph = nx.Graph()
                    graph.add_nodes_from(range(child.bins.n_bins))
                    graph.add_edges_from((int(u), int(v)) for u, v in child.bins.pairs)
                    checked, _graph = apply_walls(child, graph, child_arena)
                    if not np.array_equal(checked.bins.pairs, child.bins.pairs):
                        raise ValueError(
                            "Component connectivity crosses its declared physical walls."
                        )
            if locator.children[0].bins.n_dims == 2:
                from shapely.geometry import LineString

                centers = locator.spatial_bins().centers
                for pair in locator.bridge_pairs:
                    link = LineString(centers[pair])
                    if any(
                        link.intersects(wall)
                        for domain in locator.arenas
                        for wall in domain.walls
                    ):
                        raise ValueError(
                            "A stored navigation bridge crosses a known physical wall."
                        )
            actual_meta, actual_arrays = encode_arena(candidate)
            expected_meta, expected_arrays = encode_arena(
                _CombinedArena(locator.arenas)
            )
            return (
                actual_meta == expected_meta
                and actual_arrays.keys() == expected_arrays.keys()
                and all(
                    np.array_equal(actual_arrays[key], expected_arrays[key])
                    for key in actual_arrays
                )
            )

        if not matches(binning.locator, arena):
            raise ValueError("Combined arena disagrees with stored component domains.")
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
    from neurospatial.geometry._polygon import PolygonCells

    if isinstance(binning.shapes, PolygonCells) and combined_arena is None:
        domain = arena.polygonal_domain()
        if domain is not None:
            from neurospatial.geometry._validation import _polygon_area_allowance

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


def apply_walls(
    binning: Binning, graph: nx.Graph, arena: Arena
) -> tuple[Binning, nx.Graph]:
    """Remove complete blocked cell interfaces from paths and finite-volume faces."""
    if _composition_arena(binning.locator) is not None:
        # Each owned child already carries its certified wall geometry. Rechecking
        # its independently rounded global polygons loses the native source frame.
        # New bridge crossings are checked when constructing/validating the root.
        return binning, graph
    if not arena.walls:
        return binning, graph
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    from neurospatial.geometry._binning import KnownFaces, SpatialBins
    from neurospatial.geometry._diffusion import FaceGeometry
    from neurospatial.geometry._polygon import materialize_cells

    wall_frame = _wall_source_frame(binning.locator, arena, binning.bins.centers)
    if wall_frame is None:
        cells, wall_segments, centers = (
            materialize_cells(binning),
            arena.walls,
            binning.bins.centers,
        )
    else:
        # Shared snapshots avoid independently rounded collinear vertices.
        # Retained source-frame cells preserve current clipping and ownership.
        cells, wall_segments, centers, _, _ = wall_frame
    walls = unary_union(wall_segments)
    for index, cell in enumerate(cells):
        interior_part = walls.intersection(cell).difference(cell.boundary)
        if interior_part.length > 0:
            raise ValueError(
                f"Physical wall cuts through bin {index}. Refine or rebuild cells so walls follow complete cell faces."
            )
    keep = np.ones(len(binning.bins.pairs), dtype=bool)
    current_cells = None
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
            crosses = link.crosses(walls)
            if wall_frame is not None and crosses:
                if current_cells is None:
                    current_cells = materialize_cells(binning)
                if (
                    current_cells[u]
                    .boundary.intersection(current_cells[v].boundary)
                    .length
                    > 0
                ):
                    raise ValueError(
                        f"Physical wall face alignment for bins ({u}, {v}) is unresolved in the source frame. "
                        "Declare walls before clipping or transforming, or recenter coordinates."
                    )
            keep[index] = contact.intersection(walls).is_empty and not crosses
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
    from neurospatial.geometry._face_normals import FaceNormals

    normals = (
        FaceNormals(binning.face_normals.values[keep])
        if binning.face_normals is not None
        else None
    )
    return replace(
        binning, bins=updated, faces=faces, face_normals=normals
    ), result_graph
