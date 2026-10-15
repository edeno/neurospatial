"""A layout-compatible view of saved binning data, without a builder."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
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
    TrackLinearization,
    _outline_of,
    _shapes_of,
)
from neurospatial.layout._components import HexagonShapes, TriangleShapes
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
    plot_polygon_grid,
    plot_track,
    plot_track_linear,
    plot_triangles,
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
    """Check graph identities and navigation geometry in stored edge order."""
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
    pairs = np.asarray(list(graph.edges()), dtype=np.intp).reshape(-1, 2)
    _equal(pairs, bins.pairs, "connectivity pairs (edge order and orientation)")
    for index, (u, v) in enumerate(graph.edges()):
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
    width_errors = [
        8 * (np.spacing(np.abs(edge[:-1])) + np.spacing(np.abs(edge[1:])))
        for edge in edges
    ]
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
    uniform = all(
        np.max(np.abs(axis - axis[0]))
        <= 1e-7 * axis[0] + 4 * np.spacing(np.max(np.abs(edge)))
        for edge, axis in zip(edges, widths, strict=True)
    )
    if uniform:
        add_measure(
            np.tile([axis[0] for axis in widths], (len(active), 1)),
            np.tile([axis[0] for axis in width_errors], (len(active), 1)),
        )
    return _LocatorGeometry(locator, centers, center_error, measures, cell_gap)


def _primitive_geometry(
    locator: Locator,
    coordinate_kind: Literal["cartesian", "polar"],
) -> _LocatorGeometry:
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
        if geometry.triangles is not None:
            geometry.triangles = geometry.triangles[active]
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
    geometry.measures = [
        (value * factor, (error + arithmetic * (value + error)) * factor)
        for value, error in geometry.measures
    ]
    geometry.cell_gap *= float(np.min(np.linalg.svd(linear, compute_uv=False)))
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


def _validate_components(
    binning: Binning,
    is_track: bool,
    coordinate_kind: Literal["cartesian", "polar"],
) -> None:
    """Compare shared arrays exactly and recursively check computed geometry."""
    bins, locator = binning.bins, binning.locator
    grid, shapes, track = binning.rectilinear, binning.shapes, binning.linearization
    geometry = _locator_geometry(locator, coordinate_kind, bins.measure_dim)
    gap = min(_gap(geometry.centers), geometry.cell_gap)
    _computed_equal(
        bins.centers,
        geometry.centers,
        geometry.center_error,
        "bins.centers and locator geometry",
        gap=gap,
    )
    if (
        track is None
        and not isinstance(geometry.leaf, (NearestBinLocator, LayoutLocator))
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
    if shapes is not None and not geometry.shapes_supported:
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
