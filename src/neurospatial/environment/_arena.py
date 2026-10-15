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
    _CombinedArena,
    encode_arena,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from shapely.geometry import LineString, MultiPolygon, Polygon

    from neurospatial.arena import _TransformedArena
    from neurospatial.layout._binning import Binning, Locator
    from neurospatial.layout.base import LayoutEngine


def infer_arena(binning: Binning, *, polar: bool, layout: LayoutEngine) -> Arena:
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


def _composition_arena(locator: object) -> Arena | None:
    from neurospatial.layout._composition import CombinedLocator
    from neurospatial.layout._locators import SubsetLocator, TransformedLocator
    from neurospatial.layout._polygon import ClippedLocator

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
        from neurospatial.arena import _TransformedArena
        from neurospatial.layout._composition import CombinedLocator
        from neurospatial.layout._locators import SubsetLocator, TransformedLocator
        from neurospatial.layout._polygon import ClippedLocator

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
    from neurospatial.layout._polygon import PolygonCells

    if isinstance(binning.shapes, PolygonCells) and combined_arena is None:
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
) -> (
    tuple[
        tuple[Polygon | MultiPolygon, ...],
        tuple[LineString, ...],
        NDArray[np.float64],
        NDArray[np.float64],
        tuple[_TransformedArena, ...],
    ]
    | None
):
    """Recover a shared stored frame through retained cell-owner mappings."""
    from neurospatial.arena import _TransformedArena
    from neurospatial.layout._locators import SubsetLocator, TransformedLocator
    from neurospatial.layout._polygon import ClippedLocator, cells_from_locator

    if isinstance(locator, TransformedLocator) and isinstance(arena, _TransformedArena):
        if np.array_equal(locator.inverse, arena.inverse):
            original = (centers - arena.forward[:-1, -1]) @ arena.inverse[:-1, :-1].T
            inner = _wall_source_frame(locator.base, arena.base, original)
            if inner is None:
                return (
                    cells_from_locator(locator.base),
                    arena.base.walls,
                    original,
                    locator.inverse,
                    (arena,),
                )
            cells, walls, original, inverse, frames = inner
            return cells, walls, original, inverse @ locator.inverse, (arena, *frames)
    elif isinstance(locator, (SubsetLocator, ClippedLocator)):
        inner = _wall_source_frame(locator.base, arena, centers)
        if inner is not None:
            cells, walls, original, inverse, frames = inner
            if isinstance(locator, SubsetLocator):
                selected = np.flatnonzero(locator.parent_to_child >= 0)
                selected = selected[np.argsort(locator.parent_to_child[selected])]
                return (
                    tuple(cells[index] for index in selected),
                    walls,
                    original,
                    inverse,
                    frames,
                )
            else:
                import shapely
                from shapely.geometry import MultiPolygon, Polygon

                from neurospatial.layout._binning_layout import (
                    _polygon_coordinate_errors,
                )

                # Retain the *current* clipped support in the exact wall frame.
                # Project shared vertices once through the actual stored frames.
                # Their existing forward-residual bounds include inverse and
                # large-origin arithmetic without expanding physical support.
                polygons = np.asarray(locator.cells.polygons, dtype=object)
                coordinates = shapely.get_coordinates(polygons)
                unique, indices = np.unique(coordinates, axis=0, return_inverse=True)
                mapped = unique
                projection_errors = np.zeros_like(unique)
                for frame in frames:
                    mapped, projection_errors = frame._project(
                        mapped, projection_errors
                    )
                current_parents = cells_from_locator(locator.base)
                current_errors = [
                    _polygon_coordinate_errors(p) for p in current_parents
                ]
                source_errors = [_polygon_coordinate_errors(p) for p in cells]
                projected = shapely.set_coordinates(polygons, mapped[indices])
                clipped: list[Polygon | MultiPolygon] = []
                signed_permutation = all(
                    np.all(np.isin(frame.inverse[:2, :2], (-1.0, 0.0, 1.0)))
                    and np.all(np.count_nonzero(frame.inverse[:2, :2], axis=0) == 1)
                    and np.all(np.count_nonzero(frame.inverse[:2, :2], axis=1) == 1)
                    for frame in frames
                )
                axis_aligned = all(
                    np.all(
                        np.any(
                            np.diff(shapely.get_coordinates(ring), axis=0) == 0, axis=1
                        )
                    )
                    for polygon in (*cells, *current_parents)
                    for ring in shapely.get_rings(shapely.get_parts(polygon))
                )
                # Signed permutations recover the stored axis coordinates
                # exactly on axis-aligned parent geometry. Other inverses or
                # rounded diagonal boundaries may leave a parent-edge gap.
                # Refuse uncertain wall clearance instead of expanding cells.
                exact_axes = signed_permutation and axis_aligned
                projection_error = float(
                    np.max(np.linalg.norm(projection_errors, axis=1))
                )
                projection_error += float(
                    np.max(
                        np.linalg.norm(
                            np.asarray(current_errors) @ np.abs(inverse[:2, :2]).T,
                            axis=1,
                        )
                    )
                )
                for child, (polygon, parent) in enumerate(
                    zip(projected, locator.mapping.parent_bin_indices, strict=True)
                ):
                    retained = polygon.intersection(cells[parent])
                    if (
                        not isinstance(retained, (Polygon, MultiPolygon))
                        or retained.is_empty
                    ):
                        raise ValueError(
                            "Current clipped cells lose their source-frame support."
                        )
                    if not exact_axes:
                        error = projection_error + float(
                            np.linalg.norm(source_errors[parent])
                        )
                        for wall in walls:
                            interior = wall.intersection(cells[parent]).difference(
                                cells[parent].boundary
                            )
                            if (
                                interior.length > 0
                                and interior.intersection(retained)
                                .difference(retained.boundary)
                                .length
                                == 0
                                and interior.distance(retained) <= error
                            ):
                                raise ValueError(
                                    f"Physical wall clearance from clipped bin {child} is unresolved in the source frame. "
                                    "Declare walls before clipping or transforming, or recenter coordinates."
                                )
                    clipped.append(retained)
                return tuple(clipped), walls, original, inverse, frames
    return None


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
    from neurospatial.layout._face_normals import FaceNormals

    normals = (
        FaceNormals(binning.face_normals.values[keep])
        if binning.face_normals is not None
        else None
    )
    return replace(
        binning, bins=updated, faces=faces, face_normals=normals
    ), result_graph
