"""A layout-compatible view of saved binning data, without a builder."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx
import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from shapely.affinity import affine_transform
from shapely.geometry import MultiPolygon, Polygon

from neurospatial.layout._binning import (
    Binning,
    KnownFaces,
    LayoutLocator,
    Locator,
    SpatialBins,
    TrackLinearization,
    _outline_of,
    _shapes_of,
)
from neurospatial.layout._components import HexagonShapes, TriangleShapes
from neurospatial.layout._composition import CombinedLocator
from neurospatial.layout._derive import _LegacyTrackLinearization
from neurospatial.layout._locators import (
    HexLocator,
    MeshLocator,
    NearestBinLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.layout._plotting import (
    plot_grid,
    plot_hexagons,
    plot_polygon_cells,
    plot_polygon_grid,
    plot_track,
    plot_track_linear,
    plot_triangles,
)
from neurospatial.layout._polygon import (
    ClipMapping,
    ClippedLocator,
    PolygonCells,
    _representative,
    cells_from_locator,
    certify_faces,
    transform_polygon_cells,
)
from neurospatial.layout.helpers.utils import get_centers

if TYPE_CHECKING:
    from matplotlib.axes import Axes


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


def _validate_graph(binning: Binning, graph: nx.Graph) -> None:
    """Check graph equivalence while keeping bin-pair order authoritative."""
    bins = binning.bins
    nodes = list(graph.nodes)
    if (
        graph.is_directed()
        or graph.is_multigraph()
        or not all(
            isinstance(node, (int, np.integer))
            and not isinstance(node, (bool, np.bool_))
            for node in nodes
        )
        or set(nodes) != set(range(bins.n_bins))
    ):
        raise ValueError("connectivity nodes must be integer labels 0..n_bins-1.")
    for node in range(bins.n_bins):
        _equal(graph.nodes[node].get("pos"), bins.centers[node], f"node {node} pos")
    expected = {frozenset((int(u), int(v))) for u, v in bins.pairs}
    actual = {frozenset((int(u), int(v))) for u, v in graph.edges()}
    if actual != expected:
        raise ValueError("connectivity pairs do not match the binning geometry.")
    for index, (u, v) in enumerate(bins.pairs):
        _equal(
            graph.edges[u, v].get("distance"),
            bins.pair_distances[index],
            f"edge ({u}, {v}) distance",
        )
    if isinstance(binning.faces, KnownFaces):
        _equal(binning.faces.faces.edges, bins.pairs, "faces.edges")
        _equal(binning.faces.faces.volumes, bins.sizes, "faces.volumes")


def _gap(points: NDArray[np.float64]) -> float:
    """Smallest active-center gap, rather than a scale set by the origin."""
    if len(points) < 2:
        return math.inf
    distances = cKDTree(points).query(points, k=2)[0][:, 1]
    return float(np.min(distances))


def _polygon_coordinate_errors(polygon: Polygon | MultiPolygon) -> NDArray[np.float64]:
    """Bound each coordinate axis without hiding a thin, well-resolved axis."""
    coordinates = shapely.get_coordinates(polygon)
    # As for native grid edges, a feature needs at least 10,000 times the
    # coordinate uncertainty. Normalize each axis independently: a long y
    # edge does not make an exactly stored tiny x width unresolvable.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        uncertainty = 4 * np.max(
            np.spacing(np.abs(coordinates)), axis=0
        ) + 64 * np.finfo(float).eps * np.ptp(coordinates, axis=0)
        normalized = (coordinates - coordinates[0]) / uncertainty
    # Overlay can insert consecutive vertices separated by less than the
    # coordinate uncertainty at an otherwise well-resolved corner. Those
    # redundant vertices are not the cell's width. Simplify only this temporary
    # normalized clearance probe; retain the original geometry and every cell.
    if (
        not np.all(np.isfinite(normalized))
        or shapely.set_coordinates(polygon, normalized)
        .simplify(1.0, preserve_topology=True)
        .minimum_clearance
        < 1e4
    ):
        raise ValueError(
            "Polygon features are not resolvable at the stored coordinate precision. "
            "Rescale or recenter coordinates before clipping or transforming."
        )
    return np.asarray(uncertainty, dtype=np.float64)


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


def _current_clip_mapping(locator: Locator) -> ClipMapping | None:
    """Recover the latest clip's mapping in the current active-bin order."""
    if isinstance(locator, ClippedLocator):
        return locator.mapping
    if isinstance(locator, TransformedLocator):
        return _current_clip_mapping(locator.base)
    if isinstance(locator, SubsetLocator):
        mapping = _current_clip_mapping(locator.base)
        if mapping is None:
            return None
        if locator.parent_to_child.shape != mapping.parent_bin_indices.shape:
            raise ValueError("clip_mapping and subset parent count disagree.")
        active = _active_indices(
            locator.parent_to_child,
            int(np.count_nonzero(locator.parent_to_child >= 0)),
            "clip_mapping subset ordering",
        )
        return ClipMapping(mapping.parent_bin_indices[active], mapping.n_parent_bins)
    return None


def _validate_clip_mapping(binning: Binning) -> None:
    """Require the current published map to describe the latest clipped locator."""
    expected = _current_clip_mapping(binning.locator)
    actual = binning.clip_mapping
    if (actual is None) != (expected is None):
        raise ValueError(
            "clip_mapping presence disagrees with the latest clipped locator."
        )
    if actual is not None and expected is not None:
        if actual.n_parent_bins != expected.n_parent_bins:
            raise ValueError(
                "clip_mapping parent count disagrees with the latest clip."
            )
        _equal(
            actual.parent_bin_indices,
            expected.parent_bin_indices,
            "clip_mapping parent_bin_indices and latest clip",
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
    polygons: tuple[Polygon, ...] | None = None
    grid_indices: NDArray[np.intp] | None = None
    grid_linear: NDArray[np.float64] | None = None
    grid_linear_error: NDArray[np.float64] | None = None


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
    centers = np.column_stack(
        [
            get_centers(edge)[index]
            for edge, index in zip(edges, coordinates, strict=True)
        ]
    )
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
            raw = child.bins
            SpatialBins.from_arrays(
                raw.centers, raw.sizes, raw.measure_dim, raw.pairs, raw.pair_distances
            )
            if isinstance(child.faces, KnownFaces):
                KnownFaces(child.faces.faces)
            _validate_components(
                child, child.linearization is not None, coordinate_kind
            )
        from neurospatial.layout._combine_geometry import certify_components

        certify_components(locator.children, locator.arenas, locator.keys)
        bins = locator.spatial_bins()
        geometry = _LocatorGeometry(
            locator,
            bins.centers,
            np.zeros_like(bins.centers),
            [(bins.sizes, np.zeros_like(bins.sizes))],
        )
        if bins.n_dims == 2 and bins.measure_dim == 2:
            with suppress(ValueError):
                geometry.polygons = cells_from_locator(locator)
        return geometry
    if isinstance(locator, ClippedLocator):
        if coordinate_kind != "cartesian":
            raise ValueError("polygon cells require Cartesian coordinates.")
        parent = _locator_geometry(locator.base, coordinate_kind, 2)
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
    if isinstance(locator, LayoutLocator):
        # Existing custom layouts retain their in-memory lookup contract.
        # This exact adapter type has no codec, including under wrappers;
        # saved data cannot select this delegated-containment fallback.
        centers = np.asarray(locator.layout.bin_centers, dtype=np.float64)
        sizes = np.asarray(locator.layout.bin_sizes(), dtype=np.float64)
        geometry = _LocatorGeometry(
            locator,
            centers,
            np.zeros_like(centers),
            [(sizes, np.zeros_like(sizes))],
        )
        shapes = _shapes_of(locator.layout)
        if isinstance(shapes, HexagonShapes):
            _equal(shapes.centers, centers, "shapes.centers")
            if not np.isfinite(shapes.radius) or shapes.radius <= 0:
                raise ValueError("shapes.radius must be finite and positive.")
            geometry.radius = shapes.radius
            geometry.orientation = shapes.orientation
        elif isinstance(shapes, TriangleShapes):
            if shapes.triangles.shape != (len(centers), 3):
                raise ValueError("shapes.triangles count does not match n_bins.")
            geometry.points = shapes.points
            geometry.point_error = np.zeros_like(shapes.points)
            geometry.triangles = shapes.triangles
        outline = _outline_of(locator.layout)
        geometry.boundary = outline.geometry if outline is not None else None
        return geometry
    raise ValueError("locator needs supported stored geometry for consistency checks.")


def _mapped_geometry(
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
    measure_dim: int,
) -> _LocatorGeometry:
    if isinstance(locator, SubsetLocator):
        geometry = _locator_geometry(locator.base, coordinate_kind, measure_dim)
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
        if geometry.triangles is not None:
            geometry.triangles = geometry.triangles[active]
        if geometry.polygons is not None:
            geometry.polygons = tuple(geometry.polygons[index] for index in active)
        # Whole-bin subsets do not retain an arena outline.
        geometry.boundary = None
        return geometry
    if not isinstance(locator, TransformedLocator):
        return _primitive_geometry(locator, coordinate_kind)
    geometry = _locator_geometry(locator.base, coordinate_kind, measure_dim)
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
    )
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
    if measure_dim < n_dims and not similarity:
        raise ValueError(
            "linearization locator transform must preserve track lengths by a similarity."
        )
    factor = (
        math.exp(math.log(scale) * measure_dim)
        if similarity
        else math.exp(float(logdet))
    )
    geometry.centers, geometry.center_error = _mapped_geometry(
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
    if geometry.polygons is not None:
        geometry.polygons = transform_polygon_cells(
            PolygonCells(geometry.polygons), matrix
        ).polygons
    if geometry.points is not None:
        assert geometry.point_error is not None
        geometry.points, geometry.point_error = _mapped_geometry(
            geometry.points, geometry.point_error, matrix, linear_error, shift_error
        )
    if geometry.boundary is not None:
        coordinates = shapely.get_coordinates(geometry.boundary)
        _, errors = _mapped_geometry(
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
    if geometry.radius > 0 and not (similarity and sign > 0):
        geometry.polygons = cells_from_locator(locator)
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
    pairs = binning.bins.pairs
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
    vectors = geometry.centers[pairs[:, 1]] - geometry.centers[pairs[:, 0]]
    distances = np.hypot.reduce(vectors, axis=1)
    distance_error = (
        np.hypot.reduce(
            geometry.center_error[pairs[:, 0]] + geometry.center_error[pairs[:, 1]],
            axis=1,
        )
        + 256 * eps * distances
    )
    _computed_equal(
        faces.distance,
        distances,
        np.minimum(distance_error, distances / 4),
        "grid faces.distance",
    )


def _validate_components(
    binning: Binning,
    is_track: bool,
    coordinate_kind: Literal["cartesian", "polar"],
) -> None:
    """Compare shared arrays exactly and recursively check computed geometry."""
    bins, locator = binning.bins, binning.locator
    if binning.grid_spec is not None:
        from neurospatial.layout._rebin_spec import GridBinningSpec

        spec = binning.grid_spec
        if type(spec) is not GridBinningSpec:
            raise TypeError("grid_spec must be an owned GridBinningSpec.")
        GridBinningSpec(spec.origin, spec.forward, spec.connect_diagonal)
        if len(spec.origin) != bins.n_dims or coordinate_kind != "cartesian":
            raise ValueError(
                "Grid specification must match Cartesian coordinate dimensions."
            )
    if isinstance(binning.faces, KnownFaces):
        _equal(binning.faces.faces.edges, bins.pairs, "faces.edges")
        _equal(binning.faces.faces.volumes, bins.sizes, "faces.volumes")
    from neurospatial.layout._face_normals import validate_normals

    if binning.face_normals is not None and coordinate_kind != "cartesian":
        raise ValueError("Physical face normals require Cartesian coordinates.")
    validate_normals(binning)
    _validate_clip_mapping(binning)
    if isinstance(locator, CombinedLocator):
        if binning.face_normals is not None:
            from neurospatial.layout._composition import combined_normals

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
        expected_bins = locator.spatial_bins()
        _equal(bins.pairs, expected_bins.pairs, "combined neighbor pairs")
        _equal(
            bins.pair_distances, expected_bins.pair_distances, "combined path distances"
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
    geometry = _locator_geometry(locator, coordinate_kind, bins.measure_dim)
    gap = min(_gap(geometry.centers), geometry.cell_gap)
    if binning.grid_frame is not None:
        from neurospatial.layout._grid_frame import FramedGrid, frame_from_locator

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
        from neurospatial.layout._composition import expected_composition_binning

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
            _equal(
                bins.pairs,
                expected_binning.bins.pairs,
                "derived combined neighbor pairs",
            )
            if computed:
                u, v = bins.pairs.T
                error = np.linalg.norm(
                    geometry.center_error[u] + geometry.center_error[v], axis=1
                )
                error += (
                    64
                    * np.finfo(float).eps
                    * np.abs(expected_binning.bins.pair_distances)
                )
                _computed_equal(
                    bins.pair_distances,
                    expected_binning.bins.pair_distances,
                    error,
                    "derived combined path distances",
                    gap=gap,
                )
            else:
                _equal(
                    bins.pair_distances,
                    expected_binning.bins.pair_distances,
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
        and not isinstance(
            geometry.leaf, (NearestBinLocator, LayoutLocator, CombinedLocator)
        )
        and bins.measure_dim != bins.n_dims
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
    if grid is not None:
        if isinstance(locator, RectilinearLocator):
            grid_locator = locator
        elif isinstance(geometry.leaf, LayoutLocator):
            # The custom adapter has no stored edge duplicate. Check the
            # retained component internally as well as delegated lookup.
            grid_locator = RectilinearLocator(
                grid.edges, grid.shape, grid.active_mask, None
            )
            grid_geometry = _rectilinear_geometry(grid_locator, coordinate_kind)
            _computed_equal(
                bins.centers,
                grid_geometry.centers,
                grid_geometry.center_error,
                "rectilinear bins.centers",
                gap=min(_gap(grid_geometry.centers), grid_geometry.cell_gap),
            )
            if not any(
                value.shape == bins.sizes.shape
                and np.all(np.abs(bins.sizes - value) <= np.minimum(error, value / 4))
                for value, error in grid_geometry.measures
            ):
                raise ValueError("rectilinear bins.sizes do not match cell measures.")
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
            tolerance = np.minimum(
                coordinate_errors[index],
                _polygon_coordinate_errors(expected),
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
                for u, v in bins.pairs
            ]
            certified = certify_faces(
                shapes,
                bins.centers,
                bins.pairs,
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
                for index, (u, v) in enumerate(bins.pairs):
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
    if isinstance(geometry.leaf, (MeshLocator, LayoutLocator)):
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
    if is_track != (track is not None):
        raise ValueError(
            "is_linearized_track does not match the linearization component."
        )
    if isinstance(track, _LegacyTrackLinearization):
        # This owned view preserves the conversion-only custom-layout
        # contract. It has no codec and must not replace stored track data.
        if not isinstance(geometry.leaf, LayoutLocator):
            raise ValueError("legacy linearization needs a custom LayoutLocator.")
        if (
            track.active_mask.ndim != 1
            or np.count_nonzero(track.active_mask) != bins.n_bins
        ):
            raise ValueError("linearization active_mask count does not match n_bins.")
        if len(track.grid_edges) > 1:
            raise ValueError("linearization grid_edges must contain one linear axis.")
        if track.grid_edges:
            edges = track.grid_edges[0]
            if (
                edges.ndim != 1
                or len(edges) < 2
                or not np.all(np.isfinite(edges))
                or np.any(np.diff(edges) <= 0)
                or track.active_mask.shape != (len(edges) - 1,)
            ):
                raise ValueError(
                    "linearization edges and active_mask shape do not agree."
                )
        if bins.measure_dim != 1:
            raise ValueError("linearization bins.measure_dim must be 1.")
    elif track is not None:
        if not isinstance(track, TrackLinearization) or not isinstance(
            geometry.leaf, (NearestBinLocator, LayoutLocator)
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
        if bins.measure_dim != 1:
            raise ValueError("linearization bins.measure_dim must be 1.")
    located = locator.locate(bins.centers)
    if track is not None:
        # An embedded track can revisit an identical spatial point at a
        # different linear coordinate. Nearest-point lookup may choose either
        # coincident bin; the validated track component owns path identity.
        if located.shape != (bins.n_bins,) or np.any(
            (located < 0) | (located >= bins.n_bins)
        ):
            raise ValueError("Track locator does not locate its bin centers.")
        _equal(
            bins.centers[located], bins.centers, "track locator bin-center coordinates"
        )
    else:
        _equal(located, np.arange(bins.n_bins), "locator bin-center lookup")


class BinningLayout:
    """Expose stored bins through the layout interface without rebuilding them.

    Parameters
    ----------
    binning : Binning
        Owned bins, point lookup, faces and optional drawing components.
    connectivity : networkx.Graph
        Graph whose positions and ordered edge distances match the bins.
    dimension_ranges : sequence of pairs or None
        Coordinate bounds used when drawing the layout.
    layout_type_tag, layout_type : str
        Descriptive original layout identifier and category.
    build_params : dict
        Descriptive construction parameters. They are never executed.
    is_linearized_track : bool
        Whether the binning carries track coordinates.
    coordinate_kind : {"cartesian", "polar"}, default="cartesian"
        Internal coordinate context for validating physical cell measures.
        Polar grids store annular-sector areas, including the radial factor.

    Raises
    ------
    ValueError
        If graph, locator or component geometry disagrees with the bins.
    """

    dimension_ranges: Sequence[tuple[float, float]] | None

    def __init__(
        self,
        binning: Binning,
        *,
        connectivity: nx.Graph,
        dimension_ranges: Sequence[tuple[float, float]] | None,
        layout_type_tag: str,
        layout_type: str,
        build_params: dict[str, Any],
        is_linearized_track: bool,
        coordinate_kind: Literal["cartesian", "polar"] = "cartesian",
    ) -> None:
        if coordinate_kind not in ("cartesian", "polar"):
            raise ValueError("coordinate_kind must be 'cartesian' or 'polar'.")
        graph = copy.deepcopy(connectivity)
        _validate_graph(binning, graph)
        _validate_components(binning, is_linearized_track, coordinate_kind)
        self._binning = binning
        self.bin_centers = binning.bins.centers
        self.connectivity = graph
        self.dimension_ranges = (
            tuple((float(low), float(high)) for low, high in dimension_ranges)
            if dimension_ranges is not None
            else None
        )
        self._layout_type_tag = layout_type_tag
        self._layout_type = layout_type
        self._build_params_used = copy.deepcopy(build_params)
        self._is_linearized_track = is_linearized_track

    def binning(self) -> Binning:
        """Return the retained binning itself, preserving its data authority."""
        return self._binning

    def build(self, **kwargs: Any) -> None:
        """Keep the stored geometry; descriptive build parameters are not run."""

    def point_to_bin_index(self, points: NDArray[np.float64]) -> NDArray[np.intp]:
        """Locate ``(n_points, n_dims)`` points using the stored locator."""
        return self._binning.locator.locate(points)

    def bin_sizes(self) -> NDArray[np.float64]:
        """Return stored cell measures in active-bin order."""
        return self._binning.bins.sizes

    @property
    def layout_type(self) -> str:
        """Original descriptive layout category."""
        return self._layout_type

    @property
    def is_grid_compatible(self) -> bool:
        """Whether rectangular grid cells are available."""
        return self._binning.rectilinear is not None

    @property
    def is_linearized_track(self) -> bool:
        """Whether the binning carries absolute linear track coordinates."""
        return self._is_linearized_track

    @property
    def grid_edges(self) -> tuple[NDArray[np.float64], ...]:
        """Cell edges derived from the rectangular or linear track component."""
        if self._binning.rectilinear is not None:
            return self._binning.rectilinear.edges
        track = self._binning.linearization
        return (
            track.grid_edges
            if isinstance(track, (TrackLinearization, _LegacyTrackLinearization))
            else ()
        )

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Full grid shape, or active-bin count for non-grid cells."""
        if self._binning.rectilinear is not None:
            return self._binning.rectilinear.shape
        track = self._binning.linearization
        if isinstance(track, (TrackLinearization, _LegacyTrackLinearization)):
            return track.active_mask.shape
        if isinstance(self._binning.locator, HexLocator):
            return self._binning.locator.centers_shape
        locator = self._binning.locator
        if isinstance(locator, MeshLocator):
            return (len(locator.simplices),)
        return (self._binning.bins.n_bins,)

    @property
    def active_mask(self) -> NDArray[np.bool_]:
        """Active grid cells, or all active bins for a non-grid layout."""
        if self._binning.rectilinear is not None:
            return self._binning.rectilinear.active_mask
        track = self._binning.linearization
        if isinstance(track, (TrackLinearization, _LegacyTrackLinearization)):
            return track.active_mask
        if isinstance(self._binning.locator, HexLocator):
            hex_locator = self._binning.locator
            hex_mask: NDArray[np.bool_] = (hex_locator.flat_to_active >= 0).reshape(
                hex_locator.centers_shape
            )
            hex_mask.setflags(write=False)
            return hex_mask
        locator = self._binning.locator
        mask = (
            locator.simplex_to_active >= 0
            if isinstance(locator, MeshLocator)
            else np.ones(self._binning.bins.n_bins, dtype=bool)
        )
        mask.setflags(write=False)
        return mask

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Draw retained track, shape or rectangular geometry by component."""
        data = self._binning
        common: dict[str, Any] = {
            "title": self._layout_type_tag,
            "connectivity": self.connectivity,
            "dimension_ranges": self.dimension_ranges,
        }
        if isinstance(data.linearization, TrackLinearization):
            return plot_track(
                ax,
                title=self._layout_type_tag,
                linearization=data.linearization,
                bin_centers=self.bin_centers,
                connectivity=self.connectivity,
                grid_edges=self.grid_edges,
                **kwargs,
            )
        if isinstance(data.shapes, HexagonShapes):
            return plot_hexagons(ax, shapes=data.shapes, **common, **kwargs)
        if isinstance(data.shapes, TriangleShapes):
            ranges = self.dimension_ranges
            if ranges is None:
                ranges = tuple(
                    (float(axis.min()), float(axis.max()))
                    for axis in self.bin_centers.T
                )
            return plot_triangles(
                ax,
                title=self._layout_type_tag,
                shapes=data.shapes,
                outline=data.outline,
                connectivity=self.connectivity,
                dimension_ranges=ranges,
                bin_centers=self.bin_centers,
                **kwargs,
            )
        if isinstance(data.shapes, PolygonCells):
            return plot_polygon_cells(ax, shapes=data.shapes, **common, **kwargs)
        if data.grid_frame is not None:
            return plot_grid(
                ax,
                rectilinear=data.grid_frame.source,
                frame=data.grid_frame,
                bin_centers=self.bin_centers,
                **common,
                **kwargs,
            )
        if data.rectilinear is not None:
            if data.outline is not None:
                return plot_polygon_grid(
                    ax,
                    rectilinear=data.rectilinear,
                    outline=data.outline,
                    bin_centers=self.bin_centers,
                    **common,
                    **kwargs,
                )
            return plot_grid(
                ax,
                rectilinear=data.rectilinear,
                bin_centers=self.bin_centers,
                **common,
                **kwargs,
            )
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        if self.bin_centers.shape[1] == 2:
            ax.scatter(self.bin_centers[:, 0], self.bin_centers[:, 1], **kwargs)
        return ax

    def plot_linear_layout(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Draw a retained track in its absolute linear coordinate system."""
        track = self._binning.linearization
        if not isinstance(track, TrackLinearization):
            raise NotImplementedError(
                "plot_linear_layout needs stored track linearization."
            )
        return plot_track_linear(
            ax,
            title=self._layout_type_tag,
            linearization=track,
            grid_edges=track.grid_edges,
            **kwargs,
        )
