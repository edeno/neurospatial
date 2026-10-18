"""Pure consistency checks for owned binning, locators, components and connectivity."""

from __future__ import annotations

import math
from contextlib import suppress
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from shapely.affinity import affine_transform
from shapely.geometry import MultiPolygon, Polygon

from neurospatial.geometry._arrays import get_centers, per_dimension
from neurospatial.geometry._binning import (
    BinConnections,
    Binning,
    KnownFaces,
    MissingFaces,
    SpatialBins,
    TrackLinearization,
)
from neurospatial.geometry._combine_geometry import certify_components
from neurospatial.geometry._components import HexagonShapes, TriangleShapes
from neurospatial.geometry._composition import (
    CombinedLocator,
    combined_normals,
)
from neurospatial.geometry._coordinates import CartesianSystem, CoordinateSystem
from neurospatial.geometry._derive import transform_binning
from neurospatial.geometry._diffusion import FaceGeometry
from neurospatial.geometry._face_normals import (
    FaceNormals,
    _unit_rows,
    geometric_normals,
)
from neurospatial.geometry._grid_frame import (
    FramedGrid,
    _underflow_error,
    frame_from_locator,
)
from neurospatial.geometry._locators import (
    HexLocator,
    Locator,
    MeshLocator,
    NearestBinLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.geometry._polygon import (
    _representative,
    cells_from_locator,
    certify_faces,
)
from neurospatial.geometry._polygon_cells import (
    ClippedLocator,
    PolygonCells,
    _polygon_coordinate_errors,
    transform_polygon_cells,
)
from neurospatial.geometry._rebin_spec import GridBinningSpec


def _equal(actual: Any, expected: Any, name: str) -> None:
    """Name the first inconsistent stored field."""
    if not np.array_equal(actual, expected):
        raise ValueError(f"{name} does not match the binning geometry.")


def _active_indices(
    mapping: NDArray[np.intp], count: int, name: str
) -> NDArray[np.intp]:
    """Validate a bijection and return active full-cell IDs in bin order."""
    if mapping.ndim != 1 or np.any(mapping < -1):
        raise ValueError(f"{name} must be a one-dimensional active-bin mapping.")
    active = np.flatnonzero(mapping >= 0).astype(np.intp, copy=False)
    _equal(np.sort(mapping[active]), np.arange(count, dtype=np.intp), name)
    return active[np.argsort(mapping[active])]


def _gap(points: NDArray[np.float64]) -> float:
    """Smallest active-center gap, rather than a scale set by the origin."""
    if len(points) < 2:
        return math.inf
    distances = cKDTree(points).query(points, k=2)[0][:, 1]
    return float(np.min(distances))


def _polygon_area_allowance(
    polygon: Polygon | MultiPolygon,
    *,
    reference_clearance: float = math.inf,
) -> float:
    """Return a local, coordinate-ULP bounded polygon area comparison budget.

    Parameters
    ----------
    polygon : Polygon or MultiPolygon
        Stored finite polygon whose area is being checked.
    reference_clearance : float, optional
        Cached minimum clearance of a containing reference geometry. It further
        limits the uncertainty without repeatedly traversing that geometry.

    Returns
    -------
    float
        Boundary variation times directional coordinate uncertainty, including
        second-order corner terms. The budget is capped by the actual area and
        by the containing reference's smallest feature.

    Raises
    ------
    ValueError
        If local features cannot be resolved at the stored coordinate origin.
    """
    uncertainty = _polygon_coordinate_errors(polygon)
    coordinates = shapely.get_coordinates(polygon)
    variation = np.sum(np.abs(np.diff(coordinates, axis=0)), axis=0)
    # Varying each shoelace coordinate independently changes area by at most
    # x-error * total y-variation + y-error * total x-variation, plus corner
    # products. Ring-to-ring jumps only enlarge this conservative bound.
    allowance = float(uncertainty @ variation[::-1])
    allowance += 2 * len(coordinates) * float(np.prod(uncertainty))
    return float(
        min(
            allowance,
            polygon.area * 1e-4,
            reference_clearance**2 / 16,
        )
    )


def _computed_equal(
    actual: NDArray[np.float64],
    expected: NDArray[np.float64],
    error: NDArray[np.float64] | float,
    name: str,
    *,
    gap: float = math.inf,
) -> None:
    """Bound computed geometry without relaxing duplicated stored arrays.

    A quarter of the closest active-center/cell gap caps propagated arithmetic
    error. In particular, a large coordinate origin cannot make a neighboring
    bin an acceptable substitute. Shared arrays continue to use ``_equal``.
    """
    bound = np.minimum(error, gap / 4.0)
    if actual.shape != expected.shape or not np.all(np.abs(actual - expected) <= bound):
        raise ValueError(f"{name} does not match the binning geometry.")


def _same_ring(first: Any, second: Any, tolerance: NDArray[np.float64]) -> bool:
    """Compare oriented rings without a rounding-sensitive starting vertex."""
    actual, expected = np.asarray(first.coords)[:-1], np.asarray(second.coords)[:-1]
    if actual.shape != expected.shape:
        return False
    starts = np.flatnonzero(np.all(np.abs(expected - actual[0]) <= tolerance, axis=1))
    return any(
        np.all(np.abs(actual - np.roll(expected, -start, axis=0)) <= tolerance)
        for start in starts
    )


@dataclass(eq=False)
class _LocatorGeometry:
    """Geometry recovered from a locator, including arithmetic error only."""

    leaf: Locator
    centers: NDArray[np.float64]
    center_error: NDArray[np.float64]
    measures: list[tuple[NDArray[np.float64], NDArray[np.float64]]]
    cell_gap: float = math.inf
    points: NDArray[np.float64] | None = None
    point_error: NDArray[np.float64] | None = None
    triangles: NDArray[np.intp] | None = None
    boundary: Polygon | MultiPolygon | None = None
    boundary_error: float = 0.0
    radius: float = 0.0
    orientation: float = 0.0
    radius_error: float = 0.0
    orientation_error: float = 0.0
    computed: bool = False
    shapes_supported: bool = True
    similarity: bool = True
    polygons: tuple[Polygon, ...] | None = None
    # Per-axis arithmetic error of ``polygons`` mapped through recovered maps.
    polygon_error: NDArray[np.float64] | float = 0.0
    grid_indices: NDArray[np.intp] | None = None
    grid_linear: NDArray[np.float64] | None = None
    grid_linear_error: NDArray[np.float64] | None = None
    # Per-bin measure dimensions, when the recovered cells mix them.
    measure_dims: NDArray[np.intp] | None = None


def _grid_width_errors(
    edges: tuple[NDArray[np.float64], ...],
) -> list[NDArray[np.float64]]:
    """Bound subtraction of stored edge planes independently on each axis."""
    return [
        8 * (np.spacing(np.abs(edge[:-1])) + np.spacing(np.abs(edge[1:])))
        for edge in edges
    ]


def _uniform_grid_widths(
    edges: tuple[NDArray[np.float64], ...], widths: list[NDArray[np.float64]]
) -> bool:
    """Use the existing saved-grid uniformity condition for quantized edges."""
    return all(
        np.max(np.abs(axis - axis[0]))
        <= 1e-7 * axis[0] + 4 * np.spacing(np.max(np.abs(edge)))
        for edge, axis in zip(edges, widths, strict=True)
    )


def _stored_cells(locator: Locator) -> tuple[Polygon, ...]:
    """Recover parent cell bounds to validate a clipped locator's containment."""
    return cells_from_locator(locator)


def _rectilinear_geometry(
    locator: RectilinearLocator,
    coordinate_kind: Literal["cartesian", "polar"],
) -> _LocatorGeometry:
    shape, edges = locator.shape, locator.edges
    if (
        len(edges) != len(shape)
        or not shape
        or any(
            not isinstance(size, (int, np.integer))
            or isinstance(size, (bool, np.bool_))
            or size < 1
            for size in shape
        )
    ):
        raise ValueError("locator shape must match the bin coordinate dimensions.")
    for axis, (edge, size) in enumerate(zip(edges, shape, strict=True)):
        if (
            edge.shape != (size + 1,)
            or not np.all(np.isfinite(edge))
            or np.any(np.diff(edge) <= 0)
        ):
            raise ValueError(f"locator edges[{axis}] must increase and match shape.")
    mask = (
        np.ones(shape, dtype=bool)
        if locator.active_mask is None
        else locator.active_mask
    )
    if mask.shape != shape:
        raise ValueError("locator active_mask shape does not match locator shape.")
    active = np.flatnonzero(mask)
    if locator.flat_to_active is not None:
        mapping = np.full(mask.size, -1, dtype=np.intp)
        mapping[active] = np.arange(len(active))
        _equal(locator.flat_to_active, mapping, "locator flat_to_active ordering")
    coordinates = np.unravel_index(active, shape)
    centers = locator.active_centers()
    # Mapped endpoints and mapped centers can differ by a few ULPs even for
    # positive diagonal transforms. Eight endpoint ULPs cover the multiply,
    # shift, subtraction and midpoint operations, including cancellation.
    eps = np.finfo(np.float64).eps
    widths = [np.diff(edge) for edge in edges]
    width_errors = _grid_width_errors(edges)
    center_error = np.column_stack(
        [error[index] for error, index in zip(width_errors, coordinates, strict=True)]
    )
    cell_gap = min(float(np.min(axis)) for axis in widths)
    center_error = np.minimum(center_error, cell_gap / 4)
    measures: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []

    def add_measure(selected: NDArray[np.float64], errors: NDArray[np.float64]) -> None:
        errors = np.minimum(errors, selected / 4)
        value = np.prod(selected, axis=1)
        lower = np.prod(selected - errors, axis=1)
        upper = np.prod(selected + errors, axis=1)
        error = (
            np.maximum(value - lower, upper - value)
            + 8 * eps * (len(edges) + 1) * value
        )
        if coordinate_kind == "polar":
            if len(edges) != 2:
                raise ValueError(
                    "polar cell measures need radial and angular grid edges."
                )
            error = error * np.abs(centers[:, 0]) + (value + error) * center_error[:, 0]
            value = value * centers[:, 0]
        measures.append((value, error))

    add_measure(
        np.column_stack(
            [axis[index] for axis, index in zip(widths, coordinates, strict=True)]
        ),
        np.column_stack(
            [axis[index] for axis, index in zip(width_errors, coordinates, strict=True)]
        ),
    )
    # Native grids use first spacings. Preserve that candidate only under
    # their existing uniformity rule, including quantized large-offset edges.
    uniform = _uniform_grid_widths(edges, widths)
    if uniform:
        add_measure(
            np.tile([axis[0] for axis in widths], (len(active), 1)),
            np.tile([axis[0] for axis in width_errors], (len(active), 1)),
        )
    return _LocatorGeometry(
        locator,
        centers,
        center_error,
        measures,
        cell_gap,
        grid_indices=np.column_stack(coordinates),
        grid_linear=np.eye(len(edges)),
        grid_linear_error=np.zeros((len(edges), len(edges))),
    )


def _primitive_geometry(
    locator: Locator,
    coordinate_kind: Literal["cartesian", "polar"],
) -> _LocatorGeometry:
    if isinstance(locator, CombinedLocator):
        if coordinate_kind != "cartesian":
            raise ValueError("Combined lookup requires Cartesian coordinates.")
        for child in locator.children:
            raw, links = child.bins, child.connections
            SpatialBins.from_arrays(raw.centers, raw.sizes, raw.measure_dims)
            BinConnections.from_arrays(
                links.all_pairs, links.all_distances, links.is_open
            )
            if isinstance(child.faces, KnownFaces):
                KnownFaces(child.faces.faces)
            validate_components(child, CartesianSystem())

        certify_components(locator.children, locator.arenas, locator.keys)
        bins = locator.spatial_bins()
        geometry = _LocatorGeometry(
            locator,
            bins.centers,
            np.zeros_like(bins.centers),
            [(bins.sizes, np.zeros_like(bins.sizes))],
            measure_dims=bins.measure_dims,
        )
        if bins.n_dims == 2 and bins.has_measure_dim(2):
            with suppress(ValueError):
                geometry.polygons = cells_from_locator(locator)
        return geometry
    if isinstance(locator, ClippedLocator):
        if coordinate_kind != "cartesian":
            raise ValueError("polygon cells require Cartesian coordinates.")
        parent = _locator_geometry(locator.base, coordinate_kind, frozenset({2}))
        if locator.mapping.n_parent_bins != len(parent.centers):
            raise ValueError(
                "clipped mapping parent count disagrees with its base locator."
            )
        source = _stored_cells(locator.base)
        polygons = locator.cells.polygons
        parent_clearances = [polygon.minimum_clearance for polygon in source]
        clearances = [polygon.minimum_clearance for polygon in polygons]
        for index, polygon in enumerate(polygons):
            parent_index = locator.mapping.parent_bin_indices[index]
            parent_cell = source[parent_index]
            allowance = _polygon_area_allowance(
                polygon,
                reference_clearance=parent_clearances[parent_index],
            )
            if polygon.difference(parent_cell).area > allowance:
                raise ValueError(
                    f"polygon cell {index} extends outside its source parent."
                )
        centers = np.array([_representative(polygon) for polygon in polygons])
        areas = np.array([polygon.area for polygon in polygons])
        return _LocatorGeometry(
            locator,
            centers,
            np.zeros_like(centers),
            [(areas, np.zeros_like(areas))],
            cell_gap=min(clearances, default=math.inf),
            polygons=polygons,
        )
    if isinstance(locator, RectilinearLocator):
        return _rectilinear_geometry(locator, coordinate_kind)
    if isinstance(locator, HexLocator):
        if (
            len(locator.centers_shape) != 2
            or any(
                not isinstance(size, (int, np.integer))
                or isinstance(size, (bool, np.bool_))
                or size < 1
                for size in locator.centers_shape
            )
            or not np.all(
                np.isfinite([locator.offset_x, locator.offset_y, locator.radius])
            )
            or locator.radius <= 0
        ):
            raise ValueError("hex locator geometry must define a finite 2D lattice.")
        if locator.flat_to_active.shape != (int(np.prod(locator.centers_shape)),):
            raise ValueError(
                "hex locator flat_to_active shape does not match its lattice."
            )
        active = _active_indices(
            locator.flat_to_active,
            int(np.count_nonzero(locator.flat_to_active >= 0)),
            "hex locator flat_to_active ordering",
        )
        rows, columns = np.unravel_index(active, locator.centers_shape)
        x = columns.astype(np.float64)
        x[rows % 2 == 1] += 0.5
        width = np.sqrt(3.0) * locator.radius
        scaled = np.column_stack((x * width, rows * (np.sqrt(3.0) / 2.0 * width)))
        offsets = np.array([locator.offset_x, locator.offset_y])
        error = 8 * np.finfo(np.float64).eps * (np.abs(scaled) + np.abs(offsets))
        area = np.full(len(active), 3.0 * np.sqrt(3.0) / 2.0 * locator.radius**2.0)
        return _LocatorGeometry(
            locator,
            scaled + offsets,
            error,
            [(area, np.zeros_like(area))],
            width,
            radius=locator.radius,
        )
    if isinstance(locator, MeshLocator):
        if (
            locator.points.ndim != 2
            or locator.points.shape[1] != 2
            or not np.all(np.isfinite(locator.points))
            or locator.simplices.ndim != 2
            or locator.simplices.shape[1] != 3
            or np.any(locator.simplices < 0)
            or np.any(locator.simplices >= len(locator.points))
        ):
            raise ValueError(
                "mesh locator points and simplices must define 2D triangles."
            )
        if locator.simplex_to_active.shape != (len(locator.simplices),):
            raise ValueError(
                "mesh locator simplex_to_active shape does not match simplices."
            )
        active = _active_indices(
            locator.simplex_to_active,
            int(np.count_nonzero(locator.simplex_to_active >= 0)),
            "mesh locator simplex_to_active ordering",
        )
        triangles = locator.simplices[active]
        vertices = locator.points[triangles]
        centers = vertices.mean(axis=1)
        areas = np.abs(np.linalg.det(vertices[:, 1:, :] - vertices[:, :1, :])) / 2.0
        return _LocatorGeometry(
            locator,
            centers,
            np.zeros_like(centers),
            [(areas, np.zeros_like(areas))],
            points=locator.points,
            point_error=np.zeros_like(locator.points),
            triangles=triangles,
            boundary=locator.boundary,
        )
    if isinstance(locator, NearestBinLocator):
        if locator.centers.ndim != 2 or not np.all(np.isfinite(locator.centers)):
            raise ValueError(
                "nearest locator centers must be finite and two dimensional."
            )
        assert locator.bin_indices is not None
        order = _active_indices(
            locator.bin_indices,
            len(locator.centers),
            "nearest locator bin_indices ordering",
        )
        centers = locator.centers[order]
        return _LocatorGeometry(locator, centers, np.zeros_like(centers), [])
    raise ValueError("locator needs supported stored geometry for consistency checks.")


def mapped_geometry(
    points: NDArray[np.float64],
    error: NDArray[np.float64],
    matrix: NDArray[np.float64],
    linear_error: NDArray[np.float64],
    shift_error: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    linear, shift = matrix[:-1, :-1], matrix[:-1, -1]
    mapped = points @ linear.T + shift
    # Recovering a forward map from its stored inverse and applying it costs
    # O(n) floating operations. Include input uncertainty and inverse error;
    # the final comparison separately caps this bound by the active-bin gap.
    magnitude = np.abs(points) @ np.abs(linear).T + np.abs(shift)
    bound = (
        error @ (np.abs(linear) + linear_error).T
        + np.abs(points) @ linear_error.T
        + shift_error
        + 16 * (points.shape[1] + 1) * np.finfo(np.float64).eps * magnitude
        # Relative error bounds vanish at zero/subnormal coordinates.
        + 8 * np.spacing(np.abs(mapped))
    )
    return mapped, bound


def _locator_geometry(
    locator: Locator,
    coordinate_kind: Literal["cartesian", "polar"],
    measure_dims: frozenset[int],
) -> _LocatorGeometry:
    if isinstance(locator, SubsetLocator):
        geometry = _locator_geometry(locator.base, coordinate_kind, measure_dims)
        mapping = locator.parent_to_child
        if mapping.shape != (len(geometry.centers),):
            raise ValueError(
                "subset parent_to_child length does not match parent locator bins."
            )
        active = _active_indices(
            mapping,
            int(np.count_nonzero(mapping >= 0)),
            "subset parent_to_child ordering",
        )
        geometry.centers = geometry.centers[active]
        geometry.center_error = geometry.center_error[active]
        geometry.measures = [
            (value[active], error[active]) for value, error in geometry.measures
        ]
        if geometry.grid_indices is not None:
            geometry.grid_indices = geometry.grid_indices[active]
        if geometry.measure_dims is not None:
            geometry.measure_dims = geometry.measure_dims[active]
        if geometry.triangles is not None:
            geometry.triangles = geometry.triangles[active]
        if geometry.polygons is not None:
            geometry.polygons = tuple(geometry.polygons[index] for index in active)
        # Whole-bin subsets do not retain an arena outline.
        geometry.boundary = None
        return geometry
    if not isinstance(locator, TransformedLocator):
        return _primitive_geometry(locator, coordinate_kind)
    geometry = _locator_geometry(locator.base, coordinate_kind, measure_dims)
    n_dims = geometry.centers.shape[1]
    if locator.inverse.shape != (n_dims + 1, n_dims + 1):
        raise ValueError(
            "transformed locator inverse dimensions do not match parent geometry."
        )
    matrix = np.asarray(np.linalg.inv(locator.inverse), dtype=np.float64)
    linear = matrix[:-1, :-1]
    # Componentwise inverse sensitivity avoids inflating independent diagonal
    # axes by their normwise condition number. The homogeneous product also
    # includes cancellation in t = -A @ inverse_t across coordinate axes.
    rounding = 16 * (n_dims + 1) * np.finfo(np.float64).eps
    matrix_error = rounding * (
        np.abs(matrix) @ np.abs(locator.inverse) @ np.abs(matrix)
    ) + _underflow_error(matrix)
    linear_error = matrix_error[:-1, :-1]
    shift_error = matrix_error[:-1, -1]
    # First-order relative determinant sensitivity is tr(A^-1 dA).
    # Bound each term before summing and include determinant arithmetic.
    arithmetic = rounding + float(
        np.sum(np.abs(locator.inverse[:-1, :-1].T) * linear_error)
    )
    sign, logdet = np.linalg.slogdet(linear)
    scale = math.exp(float(logdet) / n_dims)
    normalized = linear / scale
    normalized_error = float(np.max(linear_error)) / scale
    similarity = np.max(np.abs(normalized @ normalized.T - np.eye(n_dims))) <= (
        1e-12 + 8 * normalized_error
    )
    cell_dims = geometry.measure_dims
    lowest = int(cell_dims.min()) if cell_dims is not None and len(cell_dims) else None
    lowest = min(measure_dims, default=n_dims) if lowest is None else lowest
    if lowest < n_dims and not similarity:
        raise ValueError(
            "linearization locator transform must preserve track lengths by a similarity."
        )
    factor: float | NDArray[np.float64]
    if not similarity:
        factor = math.exp(float(logdet))
    elif cell_dims is not None:
        factor = per_dimension(cell_dims, lambda dim: math.exp(math.log(scale) * dim))
    else:
        if geometry.measures and len(measure_dims) != 1:
            raise ValueError(
                "locator cells measured in more than one dimension have no "
                "single transformed measure."
            )
        factor = math.exp(math.log(scale) * min(measure_dims, default=n_dims))
    geometry.centers, geometry.center_error = mapped_geometry(
        geometry.centers, geometry.center_error, matrix, linear_error, shift_error
    )
    if geometry.grid_linear is not None:
        assert geometry.grid_linear_error is not None
        source = geometry.grid_linear
        geometry.grid_linear_error = (
            np.abs(linear) @ geometry.grid_linear_error
            + linear_error @ (np.abs(source) + geometry.grid_linear_error)
            + rounding * (np.abs(linear) @ np.abs(source))
        )
        geometry.grid_linear = linear @ source
    geometry.measures = [
        (value * factor, (error + arithmetic * (value + error)) * factor)
        for value, error in geometry.measures
    ]
    geometry.cell_gap *= float(np.min(np.linalg.svd(linear, compute_uv=False)))
    geometry.similarity &= bool(similarity)
    if (
        geometry.radius > 0
        and geometry.polygons is None
        and not (similarity and sign > 0)
    ):
        # The map does not keep hexagons regular, so derivation stores the
        # parent's exact cells mapped forward; map the same cells here.
        geometry.polygons = cells_from_locator(locator.base)
    if geometry.polygons is not None:
        # Vertices carry the arithmetic error of recovering and applying this
        # forward map, like centers; bound it at the cells' largest coordinates.
        extent = np.max(
            np.abs(shapely.get_coordinates(geometry.polygons)), axis=0, keepdims=True
        )
        _, vertex_error = mapped_geometry(
            extent,
            np.broadcast_to(geometry.polygon_error, extent.shape),
            matrix,
            linear_error,
            shift_error,
        )
        geometry.polygon_error = vertex_error[0]
        geometry.polygons = transform_polygon_cells(
            PolygonCells(geometry.polygons), matrix
        ).polygons
    if geometry.points is not None:
        assert geometry.point_error is not None
        geometry.points, geometry.point_error = mapped_geometry(
            geometry.points, geometry.point_error, matrix, linear_error, shift_error
        )
    if geometry.boundary is not None:
        coordinates = shapely.get_coordinates(geometry.boundary)
        _, errors = mapped_geometry(
            coordinates,
            np.full_like(coordinates, geometry.boundary_error),
            matrix,
            linear_error,
            shift_error,
        )
        geometry.boundary_error = float(np.max(errors))
        geometry.boundary = affine_transform(
            geometry.boundary,
            [
                linear[0, 0],
                linear[0, 1],
                linear[1, 0],
                linear[1, 1],
                matrix[0, -1],
                matrix[1, -1],
            ],
        )
    if geometry.radius > 0:
        geometry.shapes_supported &= bool(similarity and sign > 0)
        geometry.radius *= scale
        geometry.radius_error = (
            geometry.radius_error * scale + arithmetic * geometry.radius
        )
        geometry.orientation += math.atan2(normalized[1, 0], normalized[0, 0])
        geometry.orientation_error += (
            4 * (rounding + normalized_error) * (1 + abs(geometry.orientation))
        )
    geometry.computed = True
    return geometry


def _validate_rectilinear_faces(binning: Binning, geometry: _LocatorGeometry) -> None:
    """Check Cartesian grid face measures separately from navigation weights."""
    if not isinstance(binning.faces, KnownFaces) or not isinstance(
        geometry.leaf, RectilinearLocator
    ):
        return
    locator = geometry.leaf
    indices = geometry.grid_indices
    linear, linear_error = geometry.grid_linear, geometry.grid_linear_error
    assert indices is not None and linear is not None and linear_error is not None
    pairs = binning.connections.pairs
    n_dims = len(locator.edges)
    widths = [np.diff(edge) for edge in locator.edges]
    width_errors = _grid_width_errors(locator.edges)
    offsets = np.abs(indices[pairs[:, 1]] - indices[pairs[:, 0]])
    axes = np.argmax(offsets, axis=1)
    adjacent = (np.sum(offsets > 0, axis=1) == 1) & (np.max(offsets, axis=1) == 1)
    area = np.zeros(len(pairs))
    area_error = np.zeros(len(pairs))
    native_area, native_error = area.copy(), area_error.copy()
    eps = np.finfo(float).eps

    for axis in range(n_dims):
        other = [index for index in range(n_dims) if index != axis]
        basis, basis_error = linear[:, other], linear_error[:, other]
        # The product of singular values is the affine face-area stretch.
        # A matrix perturbation bounds each singular-value perturbation, so
        # source-frame projection errors stay independent of the world origin.
        singular = np.linalg.svd(basis, compute_uv=False)
        perturbation = float(np.hypot.reduce(basis_error.ravel(), initial=0.0))
        stretch = float(np.prod(singular))
        stretch_error = (
            max(
                float(np.prod(singular + perturbation)) - stretch,
                stretch - float(np.prod(np.maximum(singular - perturbation, 0))),
            )
            + 256 * eps * stretch
        )
        selected = np.flatnonzero(adjacent & (axes == axis))
        if len(selected):
            # A two-point flux needs its center link normal to the shared face.
            vector = linear[:, axis]
            normal_error = (
                np.abs(vector) @ basis_error
                + linear_error[:, axis] @ np.abs(basis)
                + linear_error[:, axis] @ basis_error
                + 256 * eps * (np.abs(vector) @ np.abs(basis))
            )
            if np.any(
                np.abs(vector @ basis)
                > normal_error
                + 1e-12 * np.hypot.reduce(vector) * np.hypot.reduce(basis, axis=0)
            ):
                raise ValueError(
                    "grid faces need center links normal to shared cell faces."
                )
            local_widths = (
                np.column_stack(
                    [widths[d][indices[pairs[selected, 0], d]] for d in other]
                )
                if other
                else np.empty((len(selected), 0))
            )
            local_errors = (
                np.column_stack(
                    [width_errors[d][indices[pairs[selected, 0], d]] for d in other]
                )
                if other
                else np.empty((len(selected), 0))
            )
            values = np.prod(local_widths, axis=1) * stretch
            upper = np.prod(local_widths + local_errors, axis=1) * (
                stretch + stretch_error
            )
            lower = np.prod(np.maximum(local_widths - local_errors, 0), axis=1) * max(
                stretch - stretch_error, 0
            )
            area[selected] = values
            area_error[selected] = np.maximum(upper - values, values - lower)
        value = float(np.prod([widths[d][0] for d in other])) * stretch
        upper = float(np.prod([widths[d][0] + width_errors[d][0] for d in other])) * (
            stretch + stretch_error
        )
        lower = float(
            np.prod([max(widths[d][0] - width_errors[d][0], 0) for d in other])
        ) * max(stretch - stretch_error, 0)
        native_area[selected] = value
        native_error[selected] = max(upper - value, value - lower)

    faces = binning.faces.faces
    candidates = [(area, area_error)]
    if _uniform_grid_widths(locator.edges, widths):
        candidates.append((native_area, native_error))
    if not any(
        np.all(np.abs(faces.A - expected) <= np.minimum(error, expected / 4))
        for expected, error in candidates
    ):
        raise ValueError(
            "grid faces.A does not match shared cell-face measures. Supply physical "
            "shared-face areas, using zero only for pairs without a face, or rebuild "
            "the environment from its source geometry. Older tiny-grid records with "
            "incorrect zero-area faces must be regenerated."
        )
    _validate_face_distances(binning, geometry, "grid faces.distance")


def _center_distances(
    geometry: _LocatorGeometry, pairs: NDArray[np.intp]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Center-link lengths ``(n_pairs,)`` and their propagated arithmetic error."""
    vectors = geometry.centers[pairs[:, 1]] - geometry.centers[pairs[:, 0]]
    distances = np.hypot.reduce(vectors, axis=1)
    distance_error = (
        np.hypot.reduce(
            geometry.center_error[pairs[:, 0]] + geometry.center_error[pairs[:, 1]],
            axis=1,
        )
        + 256 * np.finfo(float).eps * distances
    )
    return distances, distance_error


def _validate_face_distances(
    binning: Binning, geometry: _LocatorGeometry, name: str
) -> None:
    """Check stored two-point flux distances against the cell-center links."""
    assert isinstance(binning.faces, KnownFaces)
    distances, distance_error = _center_distances(geometry, binning.connections.pairs)
    _computed_equal(
        binning.faces.faces.distance,
        distances,
        np.minimum(distance_error, distances / 4),
        name,
    )


def _validate_cell_faces(binning: Binning, geometry: _LocatorGeometry) -> None:
    """Check hexagon and triangle face lengths and distances against the cells.

    Each pair's face measure is the length of the cell edge both bins share,
    and zero when the cells share no edge. Regular hexagons share a side of
    length ``radius`` exactly when their centers are ``sqrt(3) * radius``
    apart; triangles share an edge exactly when they share two mesh points.
    """
    if not isinstance(binning.faces, KnownFaces):
        return
    pairs = binning.connections.pairs
    eps = np.finfo(float).eps
    if geometry.radius > 0:
        label = "hexagon faces"
        if not geometry.similarity:
            raise ValueError(
                "hexagon faces need center links normal to shared cell faces, "
                "which an affine map other than a similarity does not keep. "
                "Use MissingFaces for these bins or rebuild the environment "
                "from its source geometry."
            )
        radius = geometry.radius
        distances, distance_error = _center_distances(geometry, pairs)
        # Nearest hexagon centers are sqrt(3) * radius apart; the next-nearest
        # are 3 * radius apart, so a quarter-spacing band separates them.
        spacing = np.sqrt(3.0) * radius
        shares_edge = np.abs(distances - spacing) <= spacing / 4
        expected = np.where(shares_edge, radius, 0.0)
        error = np.where(
            shares_edge,
            geometry.radius_error + distance_error / np.sqrt(3.0) + 256 * eps * radius,
            0.0,
        )
    elif geometry.triangles is not None and geometry.points is not None:
        assert geometry.point_error is not None
        label = "triangle faces"
        first = geometry.triangles[pairs[:, 0]]
        second = geometry.triangles[pairs[:, 1]]
        shared = np.any(first[:, :, None] == second[:, None, :], axis=2)
        shares_edge = np.count_nonzero(shared, axis=1) == 2
        ends = first[shares_edge][shared[shares_edge]].reshape(-1, 2)
        lengths = np.hypot.reduce(
            geometry.points[ends[:, 1]] - geometry.points[ends[:, 0]], axis=1
        )
        expected = np.zeros(len(pairs))
        error = np.zeros(len(pairs))
        expected[shares_edge] = lengths
        error[shares_edge] = (
            np.hypot.reduce(
                geometry.point_error[ends[:, 0]] + geometry.point_error[ends[:, 1]],
                axis=1,
            )
            + 256 * eps * lengths
        )
    else:
        return
    if not np.all(
        np.abs(binning.faces.faces.A - expected) <= np.minimum(error, expected / 4)
    ):
        raise ValueError(
            f"{label}.A does not match shared cell-edge lengths. Supply the "
            "physical length of the edge each pair of cells shares, using zero "
            "only for pairs without a shared edge, or rebuild the environment "
            "from its source geometry."
        )
    _validate_face_distances(binning, geometry, f"{label}.distance")


def validate_components(binning: Binning, coordinates: CoordinateSystem) -> None:
    """Compare shared arrays exactly and recursively check computed geometry."""
    coordinate_kind = coordinates.kind
    bins, links, locator = binning.bins, binning.connections, binning.locator
    links.check_bins(bins.n_bins)
    if binning.grid_spec is not None:
        spec = binning.grid_spec
        if type(spec) is not GridBinningSpec:
            raise TypeError("grid_spec must be an owned GridBinningSpec.")
        GridBinningSpec(spec.origin, spec.forward, spec.connect_diagonal)
        if len(spec.origin) != bins.n_dims or coordinate_kind != "cartesian":
            raise ValueError(
                "Grid specification must match Cartesian coordinate dimensions."
            )
    if isinstance(binning.faces, KnownFaces):
        _equal(binning.faces.faces.edges, links.pairs, "faces.edges")
        _equal(binning.faces.faces.volumes, bins.sizes, "faces.volumes")
    if binning.face_normals is not None and coordinate_kind != "cartesian":
        raise ValueError("Physical face normals require Cartesian coordinates.")
    validate_normals(binning)
    if isinstance(locator, CombinedLocator):
        if binning.face_normals is not None:
            expected_normals = combined_normals(locator)
            if expected_normals is None:
                raise ValueError(
                    "Combined normals require authoritative child normals."
                )
            _equal(
                binning.face_normals.values,
                expected_normals.values,
                "combined face normals",
            )
        expected_links = locator.bin_connections()
        _equal(links.all_pairs, expected_links.all_pairs, "combined neighbor pairs")
        _equal(links.is_open, expected_links.is_open, "combined open pairs")
        _equal(
            links.all_distances,
            expected_links.all_distances,
            "combined path distances",
        )
        if locator.bridge_pairs.size and binning.faces.available:
            raise ValueError(
                "Navigation bridges cannot be published as physical faces."
            )
    if isinstance(locator, CombinedLocator) and isinstance(binning.faces, KnownFaces):
        if not all(child.faces.available for child in locator.children):
            raise ValueError(
                "Combined faces require certified geometry in every child."
            )
        for name in ("A", "distance"):
            _equal(
                getattr(binning.faces.faces, name),
                np.concatenate(
                    [getattr(child.faces.build(), name) for child in locator.children]
                ),
                f"combined faces.{name}",
            )
    grid, shapes, track = binning.rectilinear, binning.shapes, binning.linearization
    geometry = _locator_geometry(
        locator,
        coordinate_kind,
        frozenset(int(dim) for dim in np.unique(bins.measure_dims)),
    )
    gap = min(_gap(geometry.centers), geometry.cell_gap)
    if binning.grid_frame is not None:
        frame = binning.grid_frame
        if coordinate_kind != "cartesian" or binning.rectilinear is not None:
            raise ValueError(
                "FramedGrid requires Cartesian cells without a duplicate world-axis grid."
            )
        checked = FramedGrid(frame.source, frame.forward)
        _equal(frame.inverse, checked.inverse, "grid_frame.inverse")
        expected_frame, error = frame_from_locator(locator)
        _equal(
            frame.source.shape, expected_frame.source.shape, "grid_frame.source.shape"
        )
        _equal(
            frame.source.active_mask,
            expected_frame.source.active_mask,
            "grid_frame.source.active_mask",
        )
        for axis, (actual, expected) in enumerate(
            zip(frame.source.edges, expected_frame.source.edges, strict=True)
        ):
            _equal(actual, expected, f"grid_frame.source.edges[{axis}]")
        reference = expected_frame.forward
        _computed_equal(
            frame.forward,
            reference,
            error,
            "grid_frame.forward and locator frame",
            gap=gap,
        )
        indices = np.unravel_index(
            np.flatnonzero(frame.source.active_mask), frame.source.shape
        )
        source_centers = np.column_stack(
            [
                (edge[:-1] + np.diff(edge) / 2)[index]
                for edge, index in zip(frame.source.edges, indices, strict=True)
            ]
        )
        _computed_equal(
            frame.to_world(source_centers),
            expected_frame.to_world(source_centers),
            np.column_stack((np.abs(source_centers), np.ones(len(source_centers))))
            @ error[:-1].T,
            "grid_frame.forward projected cells and locator frame",
            gap=gap,
        )
    if not isinstance(locator, CombinedLocator):
        composition = expected_composition_binning(locator)
        if composition is not None:
            expected_binning, computed = composition
            if binning.face_normals is not None:
                expected_normals = expected_binning.face_normals
                if expected_normals is None:
                    raise ValueError(
                        "Derived combined normals require authoritative child normals."
                    )
                actual_normals = binning.face_normals.values
                if computed:
                    if not np.allclose(
                        actual_normals, expected_normals.values, rtol=0, atol=1e-12
                    ):
                        raise ValueError(
                            "Derived combined face normals disagree with child geometry."
                        )
                else:
                    _equal(
                        actual_normals,
                        expected_normals.values,
                        "derived combined face normals",
                    )
            expected_links = expected_binning.connections
            _equal(
                links.all_pairs,
                expected_links.all_pairs,
                "derived combined neighbor pairs",
            )
            _equal(links.is_open, expected_links.is_open, "derived combined open pairs")
            if computed:
                u, v = links.all_pairs.T
                error = np.linalg.norm(
                    geometry.center_error[u] + geometry.center_error[v], axis=1
                )
                error += 64 * np.finfo(float).eps * np.abs(expected_links.all_distances)
                _computed_equal(
                    links.all_distances,
                    expected_links.all_distances,
                    error,
                    "derived combined path distances",
                    gap=gap,
                )
            else:
                _equal(
                    links.all_distances,
                    expected_links.all_distances,
                    "derived combined path distances",
                )
            if isinstance(binning.faces, KnownFaces):
                if not expected_binning.faces.available:
                    raise ValueError(
                        "Combined physical faces cannot be inferred from navigation bridges or missing child faces."
                    )
                for name in ("A", "distance"):
                    actual = getattr(binning.faces.faces, name)
                    reference = getattr(expected_binning.faces.build(), name)
                    if computed:
                        _computed_equal(
                            actual,
                            reference,
                            64 * np.finfo(float).eps * np.abs(reference),
                            f"derived combined faces.{name}",
                        )
                    else:
                        _equal(actual, reference, f"derived combined faces.{name}")
    _computed_equal(
        bins.centers,
        geometry.centers,
        geometry.center_error,
        "bins.centers and locator geometry",
        gap=gap,
    )
    if (
        track is None
        and not isinstance(geometry.leaf, (NearestBinLocator, CombinedLocator))
        and not bins.has_measure_dim(bins.n_dims)
    ):
        raise ValueError("bins.measure_dim does not match the cell geometry dimension.")
    if geometry.measures and not any(
        actual.shape == bins.sizes.shape
        and np.all(np.abs(bins.sizes - actual) <= np.minimum(error, actual / 4))
        for actual, error in geometry.measures
    ):
        raise ValueError(
            f"bins.sizes do not match {coordinate_kind} cell measures from locator geometry."
        )
    if coordinate_kind == "cartesian":
        _validate_rectilinear_faces(binning, geometry)
        _validate_cell_faces(binning, geometry)
    if grid is not None:
        if isinstance(locator, RectilinearLocator):
            grid_locator = locator
        else:
            raise ValueError(
                "rectilinear geometry needs a matching rectilinear locator."
            )
        _equal(grid.shape, grid_locator.shape, "rectilinear shape")
        if len(grid.edges) != len(grid_locator.edges):
            raise ValueError("rectilinear edges do not match locator edges.")
        for axis, (actual, expected) in enumerate(
            zip(grid.edges, grid_locator.edges, strict=True)
        ):
            _equal(actual, expected, f"rectilinear edges[{axis}]")
        mask = (
            np.ones(grid_locator.shape, dtype=bool)
            if grid_locator.active_mask is None
            else grid_locator.active_mask
        )
        _equal(grid.active_mask, mask, "rectilinear active_mask")
    if (
        shapes is not None
        and not geometry.shapes_supported
        and not isinstance(shapes, PolygonCells)
    ):
        raise ValueError("shapes do not match the locator affine geometry.")
    if isinstance(shapes, HexagonShapes):
        _equal(shapes.centers, bins.centers, "shapes.centers")
        if geometry.radius <= 0:
            raise ValueError("hexagon shapes need a matching hex locator.")
        _computed_equal(
            np.asarray(shapes.radius),
            np.asarray(geometry.radius),
            geometry.radius_error,
            "shapes.radius and hex locator radius",
            gap=geometry.radius,
        )
        orientation = shapes.orientation - geometry.orientation
        if geometry.computed:
            # Recovering a forward rotation can cross atan2's +/-pi branch.
            orientation = math.atan2(math.sin(orientation), math.cos(orientation))
        _computed_equal(
            np.asarray(orientation),
            np.asarray(0.0),
            geometry.orientation_error,
            "shapes.orientation and hex locator",
            gap=math.pi / 3,
        )
    elif isinstance(shapes, TriangleShapes):
        if (
            geometry.triangles is None
            or geometry.points is None
            or geometry.point_error is None
        ):
            raise ValueError("triangle shapes need a matching mesh locator.")
        _equal(
            shapes.triangles,
            geometry.triangles,
            "shapes.triangles and active simplex ordering",
        )
        _computed_equal(
            shapes.points,
            geometry.points,
            geometry.point_error,
            "shapes.points and mesh locator points",
            gap=_gap(geometry.points),
        )
    elif isinstance(shapes, PolygonCells):
        if geometry.polygons is None or len(shapes.polygons) != len(geometry.polygons):
            raise ValueError("polygon shapes need matching clipped locator cells.")
        clearances = [polygon.minimum_clearance for polygon in shapes.polygons]
        coordinate_errors = [
            _polygon_coordinate_errors(polygon) for polygon in shapes.polygons
        ]
        for index, (actual, expected) in enumerate(
            zip(shapes.polygons, geometry.polygons, strict=True)
        ):
            # Stored-coordinate rounding plus the arithmetic error of the cells
            # recovered from the locator; the gap cap keeps it far below a cell.
            tolerance = (
                np.minimum(
                    coordinate_errors[index],
                    _polygon_coordinate_errors(expected),
                )
                + geometry.polygon_error
            )
            actual_normalized, expected_normalized = (
                shapely.normalize(actual),
                shapely.normalize(expected),
            )
            actual_rings = [actual_normalized.exterior, *actual_normalized.interiors]
            expected_rings = [
                expected_normalized.exterior,
                *expected_normalized.interiors,
            ]
            ring_tolerance = np.minimum(tolerance, gap / 4)
            same = len(actual_rings) == len(expected_rings) and _same_ring(
                actual_rings[0], expected_rings[0], ring_tolerance
            )
            remaining = list(expected_rings[1:])
            for ring in actual_rings[1:]:
                match = next(
                    (
                        i
                        for i, candidate in enumerate(remaining)
                        if _same_ring(ring, candidate, ring_tolerance)
                    ),
                    None,
                )
                if match is None:
                    same = False
                    break
                remaining.pop(match)
            if not same:
                raise ValueError(f"polygon cell {index} disagrees with its locator.")
            if not actual.contains(shapely.Point(bins.centers[index])):
                raise ValueError(
                    f"polygon cell {index} does not contain its bin center."
                )
            allowance = _polygon_area_allowance(actual)
            if abs(actual.area - bins.sizes[index]) > allowance:
                raise ValueError(
                    f"polygon cell {index} area disagrees with bins.sizes."
                )
        if isinstance(binning.faces, KnownFaces) and not isinstance(
            geometry.leaf, CombinedLocator
        ):
            stored = binning.faces.faces
            contacts = [
                shapes.polygons[u].boundary.intersection(shapes.polygons[v].boundary)
                for u, v in binning.connections.pairs
            ]
            certified = certify_faces(
                shapes,
                bins.centers,
                binning.connections.pairs,
                center_errors=geometry.center_error if geometry.computed else None,
                path_only=np.array(
                    [
                        not contact.is_empty and contact.length == 0
                        for contact in contacts
                    ],
                    dtype=bool,
                ),
            )
            if not isinstance(certified, KnownFaces):
                raise ValueError("stored polygon faces fail geometric certification.")
            for name in ("A", "distance"):
                expected = getattr(certified.faces, name)
                actual = getattr(stored, name)
                error = (
                    256
                    * np.finfo(float).eps
                    * np.maximum(np.abs(expected), np.abs(actual))
                )
                for index, (u, v) in enumerate(binning.connections.pairs):
                    # Distances and face lengths subtract independently rounded
                    # mapped endpoints. Bound that discrepancy locally, retaining
                    # exact zero-area path-only interfaces.
                    if name == "A" and expected[index] == 0:
                        continue
                    error[index] += 2 * float(
                        np.linalg.norm(coordinate_errors[u] + coordinate_errors[v])
                    )
                    error[index] = min(
                        error[index], min(clearances[u], clearances[v]) / 4
                    )
                _computed_equal(actual, expected, error, f"polygon faces.{name}")
    elif geometry.polygons is not None:
        raise ValueError(
            "clipped locators require their matching PolygonCells component."
        )
    if isinstance(geometry.leaf, MeshLocator):
        outline = binning.outline
        if (outline is None) != (geometry.boundary is None) or (
            outline is not None
            and geometry.boundary is not None
            and not shapely.equals_exact(
                outline.geometry,
                geometry.boundary,
                tolerance=min(geometry.boundary_error, gap / 4),
            )
        ):
            raise ValueError("outline boundary does not match mesh locator boundary.")
    if track is not None:
        if not isinstance(track, TrackLinearization) or not isinstance(
            geometry.leaf, NearestBinLocator
        ):
            raise ValueError(
                "linearization needs stored track geometry and a nearest-bin locator."
            )
        if len(track.grid_edges) != 1:
            raise ValueError("linearization grid_edges must contain one linear axis.")
        edges = track.grid_edges[0]
        if (
            edges.ndim != 1
            or len(edges) < 2
            or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges) <= 0)
            or track.active_mask.shape != (len(edges) - 1,)
        ):
            raise ValueError("linearization edges and active_mask shape do not agree.")
        if np.count_nonzero(track.active_mask) != bins.n_bins:
            raise ValueError("linearization active_mask count does not match n_bins.")
        projected = track.linear_to_nd(get_centers(edges)[track.active_mask])
        error = geometry.center_error + (
            32 * np.finfo(np.float64).eps * (np.abs(projected) + np.abs(bins.centers))
        )
        _computed_equal(
            projected,
            bins.centers,
            error if geometry.computed else 0.0,
            "linearization grid_edges/graph projected centers",
            gap=gap,
        )
        widths = np.diff(edges)[track.active_mask]
        error = (
            8
            * np.finfo(np.float64).eps
            * (np.abs(edges[:-1]) + np.abs(edges[1:]))[track.active_mask]
        )
        _computed_equal(
            bins.sizes,
            widths,
            error if geometry.computed else 0.0,
            "linearization bin sizes",
            gap=float(np.min(widths)),
        )
        if not bins.has_measure_dim(1):
            raise ValueError("linearization bins.measure_dim must be 1.")
    _validate_center_lookup(binning)


def _validate_center_lookup(binning: Binning) -> None:
    """Check that the locator places every bin center in its own bin."""
    bins = binning.bins
    located = binning.locator.locate(bins.centers)
    if binning.linearization is not None:
        # An embedded track can revisit an identical spatial point at a
        # different linear coordinate. Nearest-point lookup may choose either
        # coincident bin; the validated track component owns path identity.
        if located.shape != (bins.n_bins,) or np.any(
            (located < 0) | (located >= bins.n_bins)
        ):
            raise ValueError("Track locator does not locate its bin centers.")
        _equal(
            bins.centers[located],
            bins.centers,
            "track locator bin-center coordinates",
        )
    else:
        _equal(located, np.arange(bins.n_bins), "locator bin-center lookup")


def validate_normals(binning: Binning) -> None:
    """Check shape, known face activity, orientation and geometric direction.

    MissingFaces does not retain areas: a zero row may intentionally suppress
    a geometric contact as a navigation-only edge. In that case only nonzero
    directions can be cross-checked. Such data never enables numerical faces.
    """
    component = binning.face_normals
    if component is None:
        return
    FaceNormals(component.values)  # Revalidate mutable ndarray headers at ingress.
    bins, pairs = binning.bins, binning.connections.pairs
    values = component.values
    if not bins.has_measure_dim(bins.n_dims) or values.shape != (
        len(pairs),
        bins.n_dims,
    ):
        raise ValueError(
            "face normals must match full-dimensional bins and pair order."
        )
    active = np.any(values != 0, axis=1)
    # A native face recipe is deliberately lazy, including its mesh
    # warning. Explicit/saved records always supply KnownFaces or MissingFaces.
    if isinstance(binning.faces, KnownFaces):
        areas = binning.faces.faces.A
        if not np.array_equal(active, areas > 0):
            raise ValueError(
                "face normals must be unit on positive faces and zero on zero-area edges."
            )
    delta = bins.centers[pairs[:, 1]] - bins.centers[pairs[:, 0]]
    if np.any(np.sum(values[active] * _unit_rows(delta[active]), axis=1) <= 0):
        raise ValueError(
            "face normals must point from each pair's first bin toward its second."
        )
    expected = geometric_normals(binning.locator, bins.centers, pairs)
    if expected is not None and not np.allclose(
        values[active], expected[active], rtol=0, atol=1e-12
    ):
        raise ValueError(
            "face normals disagree with the locator's physical face directions."
        )


def expected_composition_binning(locator: Locator) -> tuple[Binning, bool] | None:
    """Recover physical face authority through subsets and affine wrappers.

    This is validation of stored duplicates, not a layout rebuild or bridge
    inference. The flag distinguishes affine arithmetic from exact selection.
    """

    if isinstance(locator, CombinedLocator):
        bins = locator.spatial_bins()
        connections = locator.bin_connections()
        faces: KnownFaces | MissingFaces = MissingFaces(
            "Navigation bridges have no physical faces."
        )
        if not len(locator.bridge_pairs) and all(
            child.faces.available for child in locator.children
        ):
            parts = [child.faces.build() for child in locator.children]
            faces = KnownFaces(
                FaceGeometry(
                    connections.pairs,
                    np.concatenate([part.A for part in parts]),
                    np.concatenate([part.distance for part in parts]),
                    bins.sizes,
                )
            )
        normals = combined_normals(locator)
        return Binning(
            "combined", bins, connections, locator, faces, face_normals=normals
        ), False
    if isinstance(locator, SubsetLocator):
        result = expected_composition_binning(locator.base)
        if result is None:
            return None
        parent, computed = result
        mapping = locator.parent_to_child
        selected = np.flatnonzero(mapping >= 0)
        selected = selected[np.argsort(mapping[selected])]
        bins = SpatialBins.from_arrays(
            parent.bins.centers[selected],
            parent.bins.sizes[selected],
            parent.bins.measure_dims[selected],
        )
        connections, kept = parent.connections.select(mapping)
        subset_faces = parent.faces
        if subset_faces.available:
            old = subset_faces.build()
            subset_faces = KnownFaces(
                FaceGeometry(
                    connections.pairs, old.A[kept], old.distance[kept], bins.sizes
                )
            )

        normals = (
            FaceNormals(parent.face_normals.values[kept])
            if parent.face_normals is not None
            else None
        )
        return Binning(
            "combined_subset",
            bins,
            connections,
            locator,
            subset_faces,
            face_normals=normals,
        ), computed
    if isinstance(locator, TransformedLocator):
        result = expected_composition_binning(locator.base)
        if result is None:
            return None
        parent, _computed = result
        return transform_binning(
            parent, SimpleNamespace(A=np.linalg.inv(locator.inverse))
        ), True
    return None
