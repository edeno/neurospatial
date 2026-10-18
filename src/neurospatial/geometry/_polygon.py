"""Owned planar cells, exact clipping and conditional two-point geometry."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
import shapely
import shapely.ops
from numpy.typing import NDArray
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import linemerge, unary_union

from neurospatial.geometry._binning import (
    Binning,
    KnownFaces,
    MissingFaces,
    SpatialBins,
)
from neurospatial.geometry._components import HexagonShapes, TriangleShapes
from neurospatial.geometry._composition import CombinedLocator
from neurospatial.geometry._diffusion import FaceGeometry
from neurospatial.geometry._face_normals import FaceNormals, geometric_normals
from neurospatial.geometry._locators import (
    HexLocator,
    Locator,
    MeshLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.geometry._polygon_cells import (
    ClipMapping,
    ClippedLocator,
    PolygonCells,
    _error,
    _validate_polygon,
    transform_polygon_cells,
)


def cells_from_locator(locator: Locator) -> tuple[Polygon, ...]:
    """Materialize native cell geometry in a stored locator's current frame."""

    if isinstance(locator, CombinedLocator):
        return tuple(
            polygon
            for child in locator.children
            for polygon in materialize_cells(child)
        )
    if isinstance(locator, ClippedLocator):
        return locator.cells.polygons
    if isinstance(locator, SubsetLocator):
        parent_polygons = cells_from_locator(locator.base)
        order = np.flatnonzero(locator.parent_to_child >= 0)
        order = order[np.argsort(locator.parent_to_child[order])]
        return tuple(parent_polygons[index] for index in order)
    if isinstance(locator, TransformedLocator):
        return transform_polygon_cells(
            PolygonCells(cells_from_locator(locator.base)),
            np.asarray(np.linalg.inv(locator.inverse), dtype=np.float64),
        ).polygons
    if isinstance(locator, RectilinearLocator):
        if len(locator.shape) != 2:
            raise _error(
                "Cell materialization requires a Cartesian 2-D locator.",
                "retain planar area-measured geometry",
            )
        mask = (
            np.ones(locator.shape, dtype=bool)
            if locator.active_mask is None
            else locator.active_mask
        )
        mapping = locator.flat_to_active
        if mapping is None:
            mapping = np.full(mask.size, -1, dtype=np.intp)
            mapping[mask.ravel()] = np.arange(np.count_nonzero(mask))
        cells: dict[int, Polygon] = {}
        for flat in np.flatnonzero(mapping >= 0):
            i, j = np.unravel_index(flat, locator.shape)
            cells[int(mapping[flat])] = box(
                locator.edges[0][i],
                locator.edges[1][j],
                locator.edges[0][i + 1],
                locator.edges[1][j + 1],
            )
        return tuple(cells[index] for index in range(len(cells)))
    if isinstance(locator, MeshLocator):
        order = np.flatnonzero(locator.simplex_to_active >= 0)
        order = order[np.argsort(locator.simplex_to_active[order])]
        return tuple(
            Polygon(locator.points[locator.simplices[index]]) for index in order
        )
    if not isinstance(locator, HexLocator):
        raise _error(
            "The locator does not publish exact planar cell geometry.",
            "retain a native grid, hexagonal, mesh or clipped locator and supported wrappers",
        )
    width = np.sqrt(3.0) * locator.radius
    offsets = ((0, 2), (1, 1), (1, -1), (0, -2), (-1, -1), (-1, 1))
    vertices: dict[tuple[int, int], tuple[float, float]] = {}
    polygons: dict[int, Polygon] = {}
    for flat in np.flatnonzero(locator.flat_to_active >= 0):
        row, column = divmod(int(flat), locator.centers_shape[1])
        keys = [(2 * column + row % 2 + dx, 3 * row + dy) for dx, dy in offsets]
        for x, y in keys:
            vertices.setdefault(
                (x, y),
                (
                    locator.offset_x + x * width / 2,
                    locator.offset_y + y * locator.radius / 2,
                ),
            )
        polygons[int(locator.flat_to_active[flat])] = Polygon(
            [vertices[key] for key in keys]
        )
    return tuple(polygons[index] for index in range(len(polygons)))


def materialize_cells(binning: Binning) -> tuple[Polygon, ...]:
    """Read exact Cartesian 2-D cells from published data, without any builder."""
    if binning.bins.n_dims != 2 or binning.bins.measure_dim != 2:
        raise _error(
            "Polygon clipping requires planar area-measured cells.",
            "use a Cartesian 2-D grid, hexagonal layout or triangular mesh",
        )
    shapes = binning.shapes
    if isinstance(shapes, PolygonCells):
        polygons = shapes.polygons
    elif isinstance(shapes, TriangleShapes):
        polygons = tuple(
            Polygon(vertices) for vertices in shapes.points[shapes.triangles]
        )
    elif isinstance(shapes, HexagonShapes):
        polygons = cells_from_locator(binning.locator)
        centers = np.array([polygon.centroid.coords[0] for polygon in polygons])
        local = min(float(shapes.radius), float(np.min(binning.bins.sizes)) ** 0.5)
        allowance = (
            64
            * np.finfo(float).eps
            * np.maximum(np.abs(centers), np.abs(binning.bins.centers))
            + 64 * np.finfo(float).eps * local
        )
        if np.any(allowance >= local / 4) or not np.all(
            np.abs(centers - binning.bins.centers) <= allowance
        ):
            raise _error(
                "Canonical hexagon centers disagree with published bin centers.",
                "use consistent lattice geometry or rescale the coordinate origin",
            )
    elif binning.rectilinear is not None and len(binning.rectilinear.shape) == 2:
        grid = binning.rectilinear
        polygons = tuple(
            box(
                grid.edges[0][i],
                grid.edges[1][j],
                grid.edges[0][i + 1],
                grid.edges[1][j + 1],
            )
            for i, j in np.argwhere(grid.active_mask)
        )
    else:
        polygons = cells_from_locator(binning.locator)
    if len(polygons) != binning.bins.n_bins:
        raise ValueError("cell geometry count does not match bins.n_bins.")
    return PolygonCells(polygons).polygons


def _parts(geometry: shapely.Geometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry] if geometry.area > 0 else []
    if hasattr(geometry, "geoms"):
        return [part for child in geometry.geoms for part in _parts(child)]
    return []


def _boundary_cell_indices(polygons: Sequence[Polygon]) -> NDArray[np.intp]:
    """Find cells with positive-length edges exposed on their union's boundary."""
    boundary = unary_union(polygons).boundary
    return np.flatnonzero(
        [polygon.boundary.intersection(boundary).length > 0 for polygon in polygons]
    ).astype(np.intp, copy=False)


def _key(polygon: Polygon) -> tuple[Any, ...]:
    return (*polygon.bounds, polygon.area, shapely.normalize(polygon).wkb)


def _representative(polygon: Polygon) -> tuple[float, float]:
    point = polygon.centroid
    if not polygon.contains(point):
        point = shapely.normalize(polygon).representative_point()
    if not polygon.contains(point) or not np.all(np.isfinite(point.coords[0])):
        raise _error(
            "A clipped cell has no certifiable finite interior representative.",
            "rescale coordinates or provide resolvable polygon geometry",
        )
    return point.x, point.y


def _overlay(
    source: tuple[Polygon, ...], polygon: Polygon | MultiPolygon
) -> tuple[PolygonCells, ClipMapping]:
    # Each intersection retains its known parent. Polygonizing all boundaries
    # together can create rounding-sized strips whose representative point is
    # assigned to a different parent, losing coverage of an otherwise valid
    # cell. GEOS overlay computes the common intersections directly instead.
    records = [
        (parent, shapely.normalize(piece))
        for parent, cell in enumerate(source)
        for piece in sorted(_parts(cell.intersection(polygon)), key=_key)
    ]
    if not records:
        raise _error(
            "Polygon clip retained no positive-area cells.",
            "choose a polygon overlapping represented active cells",
        )
    cells = PolygonCells(tuple(piece for _, piece in records))
    mapping = ClipMapping(
        np.array([parent for parent, _ in records], dtype=np.intp), len(source)
    )
    for parent in range(len(source)):
        expected = source[parent].intersection(polygon)
        actual = unary_union([piece for index, piece in records if index == parent])
        positive = [piece.area for piece in _parts(expected)]
        positive.extend(piece.area for index, piece in records if index == parent)
        if not positive:
            continue
        scale = max(abs(value) for value in source[parent].bounds) + max(
            source[parent].length, polygon.length
        )
        bound = min(
            min(positive) / 4, 256 * np.finfo(float).eps * scale * source[parent].length
        )
        if actual.symmetric_difference(expected).area > bound:
            raise _error(
                f"Polygon overlay failed coverage for parent bin {parent}.",
                "rescale coordinates or simplify unresolvable geometry explicitly",
            )
    return cells, mapping


def _opening(
    first: Polygon, second: Polygon, walls: shapely.Geometry
) -> shapely.Geometry:
    shared = first.boundary.intersection(second.boundary)
    return shared.difference(walls) if not walls.is_empty else shared


def certify_faces(
    cells: PolygonCells,
    centers: NDArray[np.float64],
    pairs: NDArray[np.intp],
    *,
    walls: Sequence[shapely.Geometry] = (),
    path_only: NDArray[np.bool_] | None = None,
    center_errors: NDArray[np.float64] | None = None,
) -> KnownFaces | MissingFaces:
    """Certify two-point fluxes, optionally bounding recovered-center roundoff.

    ``center_errors`` is used when validating geometry recovered from stored
    affine locators. It propagates source arithmetic through cancelled shifts;
    native/clipping certification keeps its existing current-coordinate bound.
    """
    polygons = cells.polygons
    for index, polygon in enumerate(polygons):
        if polygon.interiors or not polygon.equals(polygon.convex_hull):
            return MissingFaces(
                f"Cell {index} is holed or nonconvex; its two-point diffusion geometry is not certified."
            )
        if not polygon.contains(Point(centers[index])):
            return MissingFaces(
                f"Cell {index} representative is not strictly interior."
            )
    wall = unary_union(walls)
    areas = np.zeros(len(pairs), dtype=np.float64)
    distances = np.hypot.reduce(centers[pairs[:, 1]] - centers[pairs[:, 0]], axis=1)
    path_only = np.zeros(len(pairs), dtype=bool) if path_only is None else path_only
    for index, (u, v) in enumerate(pairs):
        if path_only[index]:
            continue
        opening = _opening(polygons[u], polygons[v], wall)
        if opening.geom_type == "MultiLineString":
            opening = linemerge(opening)
        if (
            not isinstance(opening, LineString)
            or opening.is_empty
            or opening.length <= 0
        ):
            return MissingFaces(
                f"Pair ({u}, {v}) has no single connected straight opening for two-point diffusion."
            )
        vertices = np.asarray(opening.coords)
        tangent = vertices[-1] - vertices[0]
        length = float(np.hypot.reduce(tangent))
        if length == 0 or not np.isfinite(distances[index]) or distances[index] <= 0:
            return MissingFaces(
                f"Pair ({u}, {v}) has unresolved face or representative distance."
            )
        unit = tangent / length
        delta = centers[v] - centers[u]
        feature = min(
            distances[index],
            length,
            polygons[u].minimum_clearance,
            polygons[v].minimum_clearance,
        )
        # Mapped vertices and representatives can round independently. Bound
        # that storage discrepancy by coordinate ULPs; local subtraction,
        # normalization and dot products depend on lengths, not the origin.
        coordinates = np.r_[vertices.ravel(), centers[[u, v]].ravel()]
        arithmetic = 4 * float(np.max(np.spacing(np.abs(coordinates)))) + 64 * np.finfo(
            float
        ).eps * max(length, distances[index])
        if (
            arithmetic >= feature / 4
            or np.max(np.abs((vertices - vertices[0]) @ np.array([-unit[1], unit[0]])))
            > arithmetic
        ):
            return MissingFaces(
                f"Pair ({u}, {v}) opening is not a resolvable straight face."
            )
        orthogonality_error = arithmetic
        if center_errors is not None:
            orthogonality_error += float(
                (center_errors[u] + center_errors[v]) @ np.abs(unit)
            )
            orthogonality_error = min(orthogonality_error, feature / 4)
        if abs(float(delta @ unit)) > orthogonality_error:
            return MissingFaces(
                f"Pair ({u}, {v}) loses orthogonality between its center link and face; two-point diffusion is not certified."
            )
        normal = np.array([-unit[1], unit[0]])
        sides = (centers[[u, v]] - vertices[0]) @ normal
        crossing = LineString(centers[[u, v]]).intersection(opening)
        if (
            sides[0] * sides[1] >= 0
            or not isinstance(crossing, Point)
            or opening.boundary.covers(crossing)
        ):
            return MissingFaces(
                f"Pair ({u}, {v}) representative link does not cross the open face interior."
            )
        areas[index] = opening.length
    volumes = np.array([polygon.area for polygon in polygons])
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        weights = areas / distances
        rates = weights / np.minimum(volumes[pairs[:, 0]], volumes[pairs[:, 1]])
    if np.any(~np.isfinite(weights)) or np.any(~np.isfinite(rates)):
        return MissingFaces(
            "Clipped face conductances exceed finite numerical range; rescale the coordinate units."
        )
    return KnownFaces(FaceGeometry(pairs, areas, distances, volumes))


def clip_binning(
    binning: Binning,
    polygon: Polygon | MultiPolygon,
    *,
    domain: Polygon | MultiPolygon | None = None,
    walls: Sequence[shapely.Geometry] = (),
) -> tuple[Binning, ClipMapping]:
    """Intersect stored cells with a polygon, retaining exact support and ancestry."""
    _validate_polygon(polygon, "clip polygon")
    if domain is not None:
        _validate_polygon(domain, "arena domain")
        effective = polygon.intersection(domain)
    else:
        effective = polygon
    if effective.is_empty or effective.area <= 0:
        raise _error(
            "Clip and arena domain have no positive-area intersection.",
            "choose a clip overlapping the known domain",
        )
    # Polygon intersections can contain line/point-only contacts alongside
    # retained area. Those contacts have no cell measure or boundary authority.
    effective = unary_union(_parts(effective))
    source = materialize_cells(binning)
    cells, mapping = _overlay(source, effective)
    return _pieces_binning(binning, source, cells, mapping, walls), mapping


def split_binning(
    binning: Binning, walls: Sequence[shapely.Geometry]
) -> tuple[Binning, ClipMapping]:
    """Split every cell a wall divides into the pieces on each side of it.

    A cell crossed from edge to edge by a wall becomes one exact polygon cell
    per side; a cell holding only a wall's end stays whole. The pieces of
    one cell are not neighbors, and the walls themselves are applied later
    by :func:`apply_walls`, which removes the paths they block.

    Returns
    -------
    binning : Binning
        Exact polygon cells, with a lookup that finds the piece containing a
        point and a mapping from each piece to the cell it came from.
    mapping : ClipMapping
        Parent cell of each piece.
    """
    source = materialize_cells(binning)
    wall = unary_union(walls)
    records = [
        (parent, shapely.normalize(piece))
        for parent, cell in enumerate(source)
        for piece in sorted(_parts(shapely.ops.split(cell, wall)), key=_key)
    ]
    cells = PolygonCells(tuple(piece for _, piece in records))
    mapping = ClipMapping(
        np.array([parent for parent, _ in records], dtype=np.intp), len(source)
    )
    return _pieces_binning(binning, source, cells, mapping, ()), mapping


def _pieces_binning(
    binning: Binning,
    source: tuple[Polygon, ...],
    cells: PolygonCells,
    mapping: ClipMapping,
    walls: Sequence[shapely.Geometry],
) -> Binning:
    """Bins, paths and faces of cells cut from ``source``, keeping ancestry.

    Pieces of neighboring source cells are neighbors where their shared
    boundary has an opening past ``walls``; path lengths and face roles come
    from the source pairs.
    """
    centers = np.array([_representative(cell) for cell in cells.polygons])
    by_parent: dict[int, list[int]] = defaultdict(list)
    for child, parent in enumerate(mapping.parent_bin_indices):
        by_parent[int(parent)].append(child)
    old_faces = binning.faces.build() if binning.faces.available else None
    wall = unary_union(walls)
    records: list[tuple[int, int, bool, float]] = []
    parent_distances = np.hypot.reduce(
        binning.bins.centers[binning.bins.pairs[:, 1]]
        - binning.bins.centers[binning.bins.pairs[:, 0]],
        axis=1,
    )
    declared_paths = ~np.isclose(
        binning.bins.pair_distances,
        parent_distances,
        rtol=64 * np.finfo(np.float64).eps,
        atol=0.0,
    )
    reason = (
        None
        if old_faces is not None
        else "The parent has no certified face geometry; inherited path-only or partial-face roles cannot be reconstructed."
    )
    for index, (parent_u, parent_v) in enumerate(binning.bins.pairs):
        old_opening = _opening(source[parent_u], source[parent_v], wall)
        path_only = old_faces is not None and old_faces.A[index] == 0
        if old_faces is not None and not path_only:
            full_length = (
                source[parent_u].boundary.intersection(source[parent_v].boundary).length
            )
            coordinates = shapely.get_coordinates((source[parent_u], source[parent_v]))
            feature = max(old_opening.length, full_length, old_faces.A[index])
            tolerance = (
                4 * float(np.max(np.spacing(np.abs(coordinates))))
                + 64 * np.finfo(float).eps * feature
            )
            if (
                tolerance >= feature / 4
                or min(
                    abs(old_faces.A[index] - old_opening.length),
                    abs(old_faces.A[index] - full_length),
                )
                > tolerance
            ):
                reason = f"Parent pair ({parent_u}, {parent_v}) has an unlocated partial or non-geometric face conductance."
        for u in by_parent[int(parent_u)]:
            for v in by_parent[int(parent_v)]:
                opening = _opening(cells.polygons[u], cells.polygons[v], wall)
                if opening.length > 0 or (
                    (path_only or old_faces is None) and not opening.is_empty
                ):
                    if declared_paths[index]:
                        if not np.array_equal(
                            centers[[u, v]], binning.bins.centers[[parent_u, parent_v]]
                        ):
                            raise _error(
                                "Clipping moved an endpoint of declared navigation "
                                f"path ({parent_u}, {parent_v}).",
                                "use subset() to retain whole bins, or construct "
                                "clipped bins with updated navigation path lengths "
                                "using Environment.from_bins()",
                            )
                        distance = float(binning.bins.pair_distances[index])
                    else:
                        distance = float(np.hypot.reduce(centers[v] - centers[u]))
                    records.append((min(u, v), max(u, v), bool(path_only), distance))
    records.sort(key=lambda item: (item[0], item[1]))
    pairs = np.array([(u, v) for u, v, _, _ in records], dtype=np.intp).reshape(-1, 2)
    distances = np.array([distance for _, _, _, distance in records], dtype=np.float64)
    bins = SpatialBins(
        centers, np.array([cell.area for cell in cells.polygons]), 2, pairs, distances
    )
    faces = (
        MissingFaces(reason)
        if reason is not None
        else certify_faces(
            cells,
            centers,
            pairs,
            walls=walls,
            path_only=np.array([path for _, _, path, _ in records], dtype=bool),
        )
    )

    locator = ClippedLocator(binning.locator, cells, mapping)
    normal_values = (
        geometric_normals(locator, bins.centers, bins.pairs)
        if faces.available
        else None
    )
    if normal_values is not None:
        normal_values[faces.build().A == 0] = 0
    return Binning(
        binning.kind,
        bins,
        locator,
        faces,
        face_normals=FaceNormals(normal_values) if normal_values is not None else None,
        shapes=cells,
        grid_spec=binning.grid_spec,
    )
