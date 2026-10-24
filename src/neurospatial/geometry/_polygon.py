"""Owned planar cells, exact clipping and conditional two-point geometry."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import shapely
import shapely.ops
from numpy.typing import NDArray
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import linemerge, unary_union

from neurospatial.geometry._binning import (
    BinConnections,
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
        flat = np.flatnonzero(mapping >= 0)
        i, j = np.unravel_index(flat[np.argsort(mapping[flat])], locator.shape)
        return _boxes(locator.edges, i, j)
    if isinstance(locator, MeshLocator):
        order = np.flatnonzero(locator.simplex_to_active >= 0)
        order = order[np.argsort(locator.simplex_to_active[order])]
        return tuple(shapely.polygons(locator.points[locator.simplices[order]]))
    if not isinstance(locator, HexLocator):
        raise _error(
            "The locator does not publish exact planar cell geometry.",
            "retain a native grid, hexagonal, mesh or clipped locator and supported wrappers",
        )
    # Corners on the lattice of half-widths and half-radii: integer keys, so
    # neighboring cells share exactly the same corner coordinates.
    width = np.sqrt(3.0) * locator.radius
    offsets = np.array([(0, 2), (1, 1), (1, -1), (0, -2), (-1, -1), (-1, 1)])
    flat = np.flatnonzero(locator.flat_to_active >= 0)
    flat = flat[np.argsort(locator.flat_to_active[flat])]
    row, column = np.divmod(flat, locator.centers_shape[1])
    x = (2 * column + row % 2)[:, None] + offsets[:, 0]
    y = (3 * row)[:, None] + offsets[:, 1]
    corners = np.stack(
        [locator.offset_x + x * width / 2, locator.offset_y + y * locator.radius / 2],
        axis=-1,
    )
    return tuple(shapely.polygons(corners))


def _boxes(
    edges: Sequence[NDArray[np.float64]], i: NDArray[np.intp], j: NDArray[np.intp]
) -> tuple[Polygon, ...]:
    """Grid cells ``(i, j)`` as boxes, in the order given."""
    return tuple(
        shapely.box(edges[0][i], edges[1][j], edges[0][i + 1], edges[1][j + 1])
    )


def materialize_cells(binning: Binning) -> tuple[Polygon, ...]:
    """Read exact Cartesian 2-D cells from published data, without any builder.

    The cells of :func:`cell_polygons`, checked to be valid, nonoverlapping
    and area-measured.
    """
    if binning.bins.n_dims != 2 or not binning.bins.has_measure_dim(2):
        raise _error(
            "Polygon clipping requires planar area-measured cells.",
            "use a Cartesian 2-D grid, hexagonal layout or triangular mesh",
        )
    return PolygonCells(cell_polygons(binning)).polygons


def cell_polygons(binning: Binning) -> tuple[Polygon, ...]:
    """Every bin's cell as a polygon, in bin order, from the published data.

    The one description of a planar binning's cells: analyses read it through
    :func:`materialize_cells`, which validates it, and the plots draw it.
    Unlike ``materialize_cells`` it also reads cells whose coordinates are
    not area-measured, such as a polar (distance, angle) grid.

    Parameters
    ----------
    binning : Binning
        Two-dimensional bins with cell geometry.

    Returns
    -------
    tuple of shapely.Polygon, length n_bins

    Raises
    ------
    ValueError
        If the bins are not two-dimensional or carry no cell geometry.
    """
    if binning.bins.n_dims != 2:
        raise _error(
            "Cell polygons require two-dimensional bins.",
            "use a planar grid, hexagonal layout, triangular mesh or clipped cells",
        )
    shapes = binning.shapes
    if isinstance(shapes, PolygonCells):
        polygons = shapes.polygons
    elif isinstance(shapes, TriangleShapes):
        polygons = tuple(shapely.polygons(shapes.points[shapes.triangles]))
    elif isinstance(shapes, HexagonShapes):
        polygons = cells_from_locator(binning.locator)
        centers = shapely.get_coordinates(
            shapely.centroid(np.asarray(polygons, dtype=object))
        )
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
        polygons = _boxes(grid.edges, *np.nonzero(grid.active_mask))
    else:
        polygons = cells_from_locator(binning.locator)
    if len(polygons) != binning.bins.n_bins:
        raise ValueError("cell geometry count does not match bins.n_bins.")
    return polygons


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
    binning: Binning,
    walls: Sequence[shapely.Geometry],
    *,
    domain: Polygon | MultiPolygon | None = None,
) -> tuple[Binning, ClipMapping]:
    """Split every cell a wall divides into the pieces on each side of it.

    A cell crossed from edge to edge by a wall becomes one exact polygon cell
    per side; a cell holding only a wall's end stays whole. The pieces of
    one cell are closed neighbors, and the walls themselves are applied later
    by :func:`apply_walls`, which closes the paths they block. With
    ``domain``, each cell not inside it is first clipped to it, as
    :func:`clip_binning` does, so that every polygon cell lies in the arena.

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
    regions = (
        source
        if domain is None
        else tuple(
            cell if domain.covers(cell) else cell.intersection(domain)
            for cell in source
        )
    )
    records = [
        (parent, shapely.normalize(piece))
        for parent, region in enumerate(regions)
        for piece in sorted(
            (
                piece
                for part in _parts(region)
                for piece in _parts(shapely.ops.split(part, wall))
            ),
            key=_key,
        )
    ]
    cells = PolygonCells(tuple(piece for _, piece in records))
    mapping = ClipMapping(
        np.array([parent for parent, _ in records], dtype=np.intp), len(source)
    )
    # The walls only divide pieces here; apply_walls applies them at
    # publication, narrowing a partly covered face by its open share. Passing
    # them as walls instead makes face certification refuse such faces.
    return _pieces_binning(
        binning, source, cells, mapping, (), dividers=(wall,)
    ), mapping


class _PairRecord(NamedTuple):
    """One neighbor pair of clipped or split pieces, ``u < v``."""

    u: int
    v: int
    is_open: bool
    path_only: bool
    distance: float


def _pieces_binning(
    binning: Binning,
    source: tuple[Polygon, ...],
    cells: PolygonCells,
    mapping: ClipMapping,
    walls: Sequence[shapely.Geometry],
    *,
    dividers: Sequence[shapely.Geometry] | None = None,
) -> Binning:
    """Bins, paths and faces of cells cut from ``source``, keeping ancestry.

    ``dividers`` are the walls that separate pieces into closed pairs; they
    default to ``walls``. Splitting passes its walls here but not as
    ``walls``, which publication applies afterwards.

    Pieces of neighboring source cells are neighbors where their shared
    boundary has an opening past ``walls``; path lengths and face roles come
    from the source pairs. Pieces that meet only along or at a wall, and
    touching pieces of source pairs that were already closed, are closed
    neighbors.
    """
    centers = np.array([_representative(cell) for cell in cells.polygons])
    by_parent: dict[int, list[int]] = defaultdict(list)
    for child, parent in enumerate(mapping.parent_bin_indices):
        by_parent[int(parent)].append(child)
    old_faces = binning.faces.build() if binning.faces.available else None
    wall = unary_union(walls)
    divider = wall if dividers is None else unary_union(dividers)

    def contact(u: int, v: int) -> shapely.Geometry:
        """The boundary two pieces share, possibly a single corner point."""
        return cells.polygons[u].boundary.intersection(cells.polygons[v].boundary)

    def divided(shared: shapely.Geometry) -> bool:
        """Whether pieces meet, possibly only at a corner, and only on a wall."""
        return (
            not divider.is_empty
            and not shared.is_empty
            and shared.difference(divider).is_empty
        )

    records: list[_PairRecord] = []

    links = binning.connections
    parent_distances = np.hypot.reduce(
        binning.bins.centers[links.all_pairs[:, 1]]
        - binning.bins.centers[links.all_pairs[:, 0]],
        axis=1,
    )
    # A declared navigation distance differs from the straight center link.
    declared = ~np.isclose(
        links.all_distances,
        parent_distances,
        rtol=64 * np.finfo(np.float64).eps,
        atol=0.0,
    )
    open_rows = np.flatnonzero(links.is_open)

    def distance(u: int, v: int, parent: int | None) -> float:
        """The parent pair's declared distance, or the new center link."""
        if parent is None or not declared[parent]:
            return float(np.hypot.reduce(centers[v] - centers[u]))
        parent_u, parent_v = links.all_pairs[parent]
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
        return float(links.all_distances[parent])

    def record(
        u: int, v: int, parent: int | None, *, is_open: bool, path_only: bool = False
    ) -> _PairRecord:
        return _PairRecord(
            min(u, v), max(u, v), is_open, path_only, distance(u, v, parent)
        )

    reason = (
        None
        if old_faces is not None
        else "The parent has no certified face geometry; inherited path-only or partial-face roles cannot be reconstructed."
    )
    for index, (parent_u, parent_v) in enumerate(links.pairs):
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
        row = int(open_rows[index])
        for u in by_parent[int(parent_u)]:
            for v in by_parent[int(parent_v)]:
                shared = contact(u, v)
                opening = shared if wall.is_empty else shared.difference(wall)
                if opening.length > 0 or (
                    (path_only or old_faces is None) and not opening.is_empty
                ):
                    records.append(
                        record(u, v, row, is_open=True, path_only=bool(path_only))
                    )
                elif divided(shared):
                    # Contact only where a wall runs: the wall closes the pair.
                    records.append(record(u, v, row, is_open=False))
    for parent in np.flatnonzero(~links.is_open):
        parent_u, parent_v = links.all_pairs[parent]
        for u in by_parent[int(parent_u)]:
            for v in by_parent[int(parent_v)]:
                # Corner contact counts: a diagonal across a wall stays closed.
                if not contact(u, v).is_empty:
                    records.append(record(u, v, int(parent), is_open=False))
    for children in by_parent.values():
        # Pieces of one cell that meet on the wall that divided it.
        for position, u in enumerate(children):
            for v in children[position + 1 :]:
                if divided(contact(u, v)):
                    records.append(record(u, v, None, is_open=False))
    records.sort(key=lambda item: (item.u, item.v))
    all_pairs = np.array([(r.u, r.v) for r in records], dtype=np.intp).reshape(-1, 2)
    is_open = np.array([r.is_open for r in records], dtype=bool)
    bins = SpatialBins.from_arrays(
        centers, np.array([cell.area for cell in cells.polygons]), 2
    )
    # Publication records the wall that closes each closed pair.
    connections = BinConnections.from_arrays(
        all_pairs,
        np.array([r.distance for r in records], dtype=np.float64),
        is_open,
    )
    pairs = connections.pairs
    faces = (
        MissingFaces(reason)
        if reason is not None
        else certify_faces(
            cells,
            centers,
            pairs,
            walls=walls,
            path_only=np.array([r.path_only for r in records if r.is_open], dtype=bool),
        )
    )

    locator = ClippedLocator(binning.locator, cells, mapping)
    normal_values = (
        geometric_normals(locator, bins.centers, pairs) if faces.available else None
    )
    if normal_values is not None:
        normal_values[faces.build().A == 0] = 0
    return Binning(
        binning.kind,
        bins,
        connections,
        locator,
        faces,
        face_normals=FaceNormals(normal_values) if normal_values is not None else None,
        shapes=cells,
        grid_spec=binning.grid_spec,
    )
