"""Owned planar cells, exact clipping and conditional two-point geometry."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shapely
from numpy.typing import ArrayLike, NDArray
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import linemerge, polygonize, unary_union
from shapely.strtree import STRtree

from neurospatial._exceptions import _format_error
from neurospatial.layout._binning import (
    Binning,
    KnownFaces,
    Locator,
    MissingFaces,
    SpatialBins,
)
from neurospatial.layout._components import HexagonShapes, TriangleShapes
from neurospatial.layout._diffusion import FaceGeometry
from neurospatial.layout._locators import (
    HexLocator,
    MeshLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)


def _error(what: str, fix: str) -> ValueError:
    return ValueError(
        _format_error(
            what,
            why="Why: exact cell geometry must remain consistent with point lookup and physical measures.",
            fix=fix,
        )
    )


def _validate_polygon(geometry: Any, name: str) -> None:
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise _error(
            f"{name} must be a Polygon or MultiPolygon.",
            "provide valid finite Cartesian 2-D polygon geometry",
        )
    if (
        geometry.is_empty
        or geometry.has_z
        or not np.all(np.isfinite(shapely.get_coordinates(geometry)))
    ):
        raise _error(
            f"{name} must be nonempty, finite and two-dimensional.",
            "remove non-finite or extra coordinates and provide a nonempty polygon",
        )
    if not geometry.is_valid or not np.isfinite(geometry.area) or geometry.area <= 0:
        raise _error(
            f"{name} is invalid: {shapely.is_valid_reason(geometry)}.",
            "correct the polygon without implicit geometry repair",
        )


@dataclass(frozen=True, eq=False)
class PolygonCells:
    """Ordered immutable connected polygons, including their interior rings."""

    polygons: tuple[Polygon, ...]

    def __post_init__(self) -> None:
        polygons = tuple(self.polygons)
        for index, polygon in enumerate(polygons):
            if not isinstance(polygon, Polygon):
                raise _error(
                    f"cells[{index}] must be a connected Polygon.",
                    "split disconnected positive-area pieces into separate bins",
                )
            _validate_polygon(polygon, f"cells[{index}]")
        _check_disjoint(polygons)
        object.__setattr__(self, "polygons", polygons)


@dataclass(frozen=True, eq=False)
class ClipMapping:
    """Owned child-to-immediate-parent indices, shape ``(n_child_bins,)``."""

    parent_bin_indices: NDArray[np.intp]
    n_parent_bins: int

    def __post_init__(self) -> None:
        raw = np.asarray(self.parent_bin_indices)
        if (
            not isinstance(self.n_parent_bins, (int, np.integer))
            or isinstance(self.n_parent_bins, (bool, np.bool_))
            or self.n_parent_bins < 0
            or self.n_parent_bins > np.iinfo(np.intp).max
        ):
            raise ValueError(
                "n_parent_bins must be a representable nonnegative integer."
            )
        if (
            raw.ndim != 1
            or (raw.size and raw.dtype.kind not in "iu")
            or np.any(raw < 0)
            or np.any(raw >= self.n_parent_bins)
        ):
            raise ValueError(
                "parent_bin_indices must be one-dimensional integer indices in [0, n_parent_bins)."
            )
        indices = np.array(raw, dtype=np.intp, copy=True)
        indices.setflags(write=False)
        object.__setattr__(self, "parent_bin_indices", indices)
        object.__setattr__(self, "n_parent_bins", int(self.n_parent_bins))

    def __setstate__(self, state: dict[str, Any]) -> None:
        state["parent_bin_indices"].setflags(write=False)
        for name, value in state.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, eq=False)
class ClippedLocator:
    """Apply immediate-parent ownership, then closed child-polygon membership."""

    base: Locator
    cells: PolygonCells
    mapping: ClipMapping
    _tree: STRtree | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.cells.polygons) != len(self.mapping.parent_bin_indices):
            raise ValueError("cells and mapping must contain the same number of bins.")

    def __getstate__(self) -> dict[str, Any]:
        return {name: value for name, value in vars(self).items() if name != "_tree"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_tree", None)

    def locate(self, points: NDArray[np.float64]) -> NDArray[np.intp]:
        """Locate ``(n_points, 2)`` coordinates without resurrecting dropped owners."""
        query = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if query.ndim != 2 or query.shape[1] != 2:
            raise ValueError(f"Expected points of shape (M, 2), got {query.shape}.")
        result = np.full(len(query), -1, dtype=np.intp)
        rows = np.flatnonzero(np.all(np.isfinite(query), axis=1))
        if not len(rows) or not self.cells.polygons:
            return result
        parents = self.base.locate(query[rows])
        if np.any(parents >= self.mapping.n_parent_bins):
            raise ValueError("base locator returned a parent outside n_parent_bins.")
        if self._tree is None:
            object.__setattr__(self, "_tree", STRtree(self.cells.polygons))
        assert self._tree is not None
        point_ids, child_ids = self._tree.query(
            shapely.points(query[rows]), predicate="covered_by"
        )
        matching = self.mapping.parent_bin_indices[child_ids] == parents[point_ids]
        local = np.full(len(rows), len(self.cells.polygons), dtype=np.intp)
        np.minimum.at(local, point_ids[matching], child_ids[matching])
        local[local == len(self.cells.polygons)] = -1
        result[rows] = local
        return result


def _check_disjoint(polygons: tuple[Polygon, ...]) -> None:
    tree = STRtree(polygons)
    for index, polygon in enumerate(polygons):
        for other in tree.query(polygon):
            if other > index and polygon.intersection(polygons[other]).area > 0:
                raise _error(
                    f"cells[{index}] and cells[{other}] overlap in their interiors.",
                    "provide a nonoverlapping cell partition",
                )


def subset_polygon_cells(cells: PolygonCells, keep: ArrayLike) -> PolygonCells:
    """Retain polygons in their existing bin order."""
    mask = np.asarray(keep)
    if mask.dtype.kind != "b" or mask.shape != (len(cells.polygons),):
        raise ValueError("keep must be a boolean mask with one entry per polygon.")
    return PolygonCells(
        tuple(
            polygon
            for polygon, selected in zip(cells.polygons, mask, strict=True)
            if selected
        )
    )


def transform_polygon_cells(
    cells: PolygonCells, matrix: NDArray[np.float64]
) -> PolygonCells:
    """Map shared vertex coordinates with one homogeneous Cartesian affine map."""
    from neurospatial.layout._locators import _check_homogeneous

    matrix = np.asarray(matrix, dtype=np.float64)
    _check_homogeneous(matrix)
    if matrix.shape != (3, 3):
        raise ValueError("polygon transform must have shape (3, 3).")
    # Node both sides before mapping: rounding an extra vertex in only one
    # ring can bend a formerly shared straight edge on only that side.
    tree = STRtree(cells.polygons)
    noded: dict[int, Polygon] = {}
    for atom in polygonize(unary_union([p.boundary for p in cells.polygons])):
        owners = tree.query(atom.representative_point(), predicate="within")
        if len(owners):
            owner = int(owners[0])
            if owner in noded or not atom.equals(cells.polygons[owner]):
                raise _error(
                    "Shared cell boundaries could not be noded without changing support.",
                    "provide a resolvable nonoverlapping polygon partition",
                )
            noded[owner] = atom
    if len(noded) != len(cells.polygons):
        raise _error(
            "Noding lost a cell's interior.",
            "rescale coordinates or provide resolvable polygon geometry",
        )
    geometries = np.asarray([noded[i] for i in range(len(noded))], dtype=object)
    coordinates = shapely.get_coordinates(geometries)
    unique, inverse = np.unique(coordinates, axis=0, return_inverse=True)
    with np.errstate(over="ignore", invalid="ignore"):
        mapped = unique @ matrix[:2, :2].T + matrix[:2, 2]
    if not np.all(np.isfinite(mapped)):
        raise _error(
            "Transformed polygon vertices exceed floating-point range.",
            "rescale the coordinate units before transforming",
        )
    return PolygonCells(tuple(shapely.set_coordinates(geometries, mapped[inverse])))


def cells_from_locator(locator: Locator) -> tuple[Polygon, ...]:
    """Materialize native cell geometry in a stored locator's current frame."""
    from neurospatial.layout._composition import CombinedLocator

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
    tree = STRtree(source)
    atoms: dict[int, list[Polygon]] = defaultdict(list)
    linework = unary_union([*(cell.boundary for cell in source), polygon.boundary])
    for atom in polygonize(linework):
        point = atom.representative_point()
        if not polygon.contains(point):
            continue
        owners = tree.query(point, predicate="within")
        if len(owners) > 1:
            raise _error(
                "Polygon overlay found ambiguous parent ownership.",
                "provide cells with disjoint interiors",
            )
        if len(owners):
            atoms[int(owners[0])].append(atom)
    records = [
        (parent, shapely.normalize(piece))
        for parent in sorted(atoms)
        for piece in sorted(_parts(unary_union(atoms[parent])), key=_key)
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
) -> KnownFaces | MissingFaces:
    """Certify conservative isotropic two-point fluxes, keeping geometry on refusal."""
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
        if abs(float(delta @ unit)) > arithmetic:
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
    centers = np.array([_representative(cell) for cell in cells.polygons])
    by_parent: dict[int, list[int]] = defaultdict(list)
    for child, parent in enumerate(mapping.parent_bin_indices):
        by_parent[int(parent)].append(child)
    old_faces = binning.faces.build() if binning.faces.available else None
    wall = unary_union(walls)
    records: list[tuple[int, int, bool]] = []
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
                    records.append((min(u, v), max(u, v), bool(path_only)))
    records.sort(key=lambda item: (item[0], item[1]))
    pairs = np.array([(u, v) for u, v, _ in records], dtype=np.intp).reshape(-1, 2)
    distances = np.hypot.reduce(centers[pairs[:, 1]] - centers[pairs[:, 0]], axis=1)
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
            path_only=np.array([path for _, _, path in records], dtype=bool),
        )
    )
    return Binning(
        "PolygonCells",
        bins,
        ClippedLocator(binning.locator, cells, mapping),
        faces,
        shapes=cells,
        clip_mapping=mapping,
    ), mapping
