"""A layout-compatible view of saved binning data, without a builder."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.layout._binning import Binning, KnownFaces, TrackLinearization
from neurospatial.layout._components import HexagonShapes, TriangleShapes
from neurospatial.layout._locators import (
    HexLocator,
    MeshLocator,
    NearestBinLocator,
    RectilinearLocator,
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


def _validate_rectilinear_measures(
    binning: Binning,
    edges: tuple[NDArray[np.float64], ...],
    coordinates: tuple[NDArray[np.intp], ...],
    coordinate_kind: Literal["cartesian", "polar"],
) -> None:
    """Check physical measures while retaining native uniform-grid arithmetic."""
    if coordinate_kind == "polar" and len(edges) != 2:
        raise ValueError("polar cell measures need radial and angular grid edges.")
    widths = [np.diff(edge) for edge in edges]
    # Bound rounding in the width products and optional radial product by
    # eight machine epsilons per operation; do not add a unit-dependent floor.
    tolerance = 8 * np.finfo(np.float64).eps * (len(edges) + 1)

    def matches(measures: NDArray[np.float64]) -> bool:
        if coordinate_kind == "polar":
            # _PolarGridLayout.bin_sizes scales its base grid measure by
            # the center radius: r_mean * delta_r * delta_theta.
            measures = measures * binning.bins.centers[:, 0]
        return bool(np.allclose(binning.bins.sizes, measures, rtol=tolerance, atol=0))

    with np.errstate(over="ignore", invalid="ignore"):
        per_cell = np.prod(
            np.column_stack(
                [axis[index] for axis, index in zip(widths, coordinates, strict=True)]
            ),
            axis=1,
        )
        if matches(per_cell):
            return
        # Native grid builders report the product of FIRST axis spacings.
        # MaskedGridLayout accepts widths differing by 1e-7*w0 + 4 edge ULPs;
        # quantized linspace/arange edges need not have identical differences.
        # Retain that native measure only within this existing uniform rule.
        uniform = all(
            np.max(np.abs(axis - axis[0]))
            <= 1e-7 * axis[0] + 4 * np.spacing(np.max(np.abs(edge)))
            for edge, axis in zip(edges, widths, strict=True)
        )
        if uniform and matches(
            np.full(binning.bins.n_bins, np.prod([axis[0] for axis in widths]))
        ):
            return
    raise ValueError(
        f"bins.sizes do not match {coordinate_kind} cell measures from grid edges."
    )


def _validate_rectilinear(
    binning: Binning,
    locator: RectilinearLocator,
    coordinate_kind: Literal["cartesian", "polar"],
) -> None:
    """Check cell edges, active numbering and exact midpoint coordinates."""
    shape, edges = locator.shape, locator.edges
    if (
        len(shape) != binning.bins.n_dims
        or len(edges) != len(shape)
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
    if len(active) != binning.bins.n_bins:
        raise ValueError("locator active_mask count does not match n_bins.")
    if locator.flat_to_active is not None:
        expected_mapping = np.full(mask.size, -1, dtype=np.intp)
        expected_mapping[active] = np.arange(len(active))
        _equal(
            locator.flat_to_active, expected_mapping, "locator flat_to_active ordering"
        )
    grid = binning.rectilinear
    if grid is not None:
        _equal(grid.shape, shape, "rectilinear shape")
        if len(grid.edges) != len(edges):
            raise ValueError("rectilinear edges do not match locator edges.")
        for axis, (actual, expected) in enumerate(zip(grid.edges, edges, strict=True)):
            _equal(actual, expected, f"rectilinear edges[{axis}]")
        _equal(grid.active_mask, mask, "rectilinear active_mask")
    coordinates = np.unravel_index(active, shape)
    expected_centers = np.column_stack(
        [
            get_centers(edge)[indices]
            for edge, indices in zip(edges, coordinates, strict=True)
        ]
    )
    _equal(binning.bins.centers, expected_centers, "bins.centers and rectangular cells")
    _validate_rectilinear_measures(binning, edges, coordinates, coordinate_kind)


def _validate_hexagonal(binning: Binning, locator: HexLocator) -> None:
    """Check lattice geometry, active ordering and regular-hexagon measures."""
    if (
        binning.bins.n_dims != 2
        or len(locator.centers_shape) != 2
        or any(
            not isinstance(size, (int, np.integer))
            or isinstance(size, (bool, np.bool_))
            or size < 1
            for size in locator.centers_shape
        )
        or not np.all(np.isfinite([locator.offset_x, locator.offset_y, locator.radius]))
        or locator.radius <= 0
    ):
        raise ValueError("hex locator geometry must define a finite 2D lattice.")
    if locator.flat_to_active.shape != (int(np.prod(locator.centers_shape)),):
        raise ValueError("hex locator flat_to_active shape does not match its lattice.")
    active = _active_indices(
        locator.flat_to_active,
        binning.bins.n_bins,
        "hex locator flat_to_active ordering",
    )
    rows, columns = np.unravel_index(active, locator.centers_shape)
    column_coordinates = columns.astype(np.float64)
    column_coordinates[rows % 2 == 1] += 0.5
    width = np.sqrt(3.0) * locator.radius
    scaled = np.column_stack(
        (column_coordinates * width, rows * (np.sqrt(3.0) / 2.0 * width))
    )
    offsets = np.array([locator.offset_x, locator.offset_y])
    centers = scaled + offsets
    # Construction stores width / sqrt(3), so reconstructing width may lose
    # an ulp. Shared stored coordinates remain exact; only this inverse
    # arithmetic allows roundoff, capped below a quarter of a lattice step.
    tolerance = np.minimum(
        8 * np.finfo(np.float64).eps * (np.abs(scaled) + np.abs(offsets)),
        width / 4,
    )
    if not np.all(np.abs(binning.bins.centers - centers) <= tolerance):
        raise ValueError("bins.centers do not match hex locator lattice geometry.")
    if isinstance(binning.shapes, HexagonShapes):
        _equal(
            binning.shapes.radius,
            locator.radius,
            "shapes.radius and hex locator radius",
        )
        _equal(binning.shapes.orientation, 0.0, "shapes.orientation and hex locator")
    area = 3.0 * np.sqrt(3.0) / 2.0 * locator.radius**2.0
    _equal(
        binning.bins.sizes, np.full(len(active), area), "bins.sizes and hexagon area"
    )


def _validate_mesh(binning: Binning, locator: MeshLocator) -> None:
    """Check active triangle identities, boundary, centroids and cell areas."""
    if (
        locator.points.ndim != 2
        or locator.points.shape[1] != 2
        or not np.all(np.isfinite(locator.points))
        or locator.simplices.ndim != 2
        or locator.simplices.shape[1] != 3
        or np.any(locator.simplices < 0)
        or np.any(locator.simplices >= len(locator.points))
    ):
        raise ValueError("mesh locator points and simplices must define 2D triangles.")
    if locator.simplex_to_active.shape != (len(locator.simplices),):
        raise ValueError(
            "mesh locator simplex_to_active shape does not match simplices."
        )
    active = _active_indices(
        locator.simplex_to_active,
        binning.bins.n_bins,
        "mesh locator simplex_to_active ordering",
    )
    triangles = locator.simplices[active]
    if isinstance(binning.shapes, TriangleShapes):
        _equal(
            binning.shapes.points,
            locator.points,
            "shapes.points and mesh locator points",
        )
        _equal(
            binning.shapes.triangles,
            triangles,
            "shapes.triangles and active simplex ordering",
        )
    outline = binning.outline
    if (outline is None) != (locator.boundary is None) or (
        outline is not None
        and locator.boundary is not None
        and outline.geometry.wkb != locator.boundary.wkb
    ):
        raise ValueError("outline boundary does not match mesh locator boundary.")
    vertices = locator.points[triangles]
    _equal(
        binning.bins.centers,
        vertices.mean(axis=1),
        "bins.centers and triangle centroids",
    )
    area = np.abs(np.linalg.det(vertices[:, 1:, :] - vertices[:, :1, :])) / 2.0
    _equal(binning.bins.sizes, area, "bins.sizes and triangle areas")


def _validate_components(
    binning: Binning,
    is_track: bool,
    coordinate_kind: Literal["cartesian", "polar"],
) -> None:
    """Compare component data directly, including boundaries between centers."""
    bins, locator = binning.bins, binning.locator
    grid, shapes, track = binning.rectilinear, binning.shapes, binning.linearization
    if (
        track is None
        and isinstance(locator, (RectilinearLocator, HexLocator, MeshLocator))
        and bins.measure_dim != bins.n_dims
    ):
        raise ValueError("bins.measure_dim does not match the cell geometry dimension.")
    if grid is not None:
        if np.count_nonzero(grid.active_mask) != bins.n_bins:
            raise ValueError("rectilinear active_mask count does not match n_bins.")
        if not isinstance(locator, RectilinearLocator):
            raise ValueError(
                "rectilinear geometry needs a matching rectilinear locator."
            )
    if isinstance(shapes, HexagonShapes):
        _equal(shapes.centers, bins.centers, "shapes.centers")
        if not isinstance(locator, HexLocator):
            raise ValueError("hexagon shapes need a matching hex locator.")
    elif isinstance(shapes, TriangleShapes):
        if shapes.triangles.shape != (bins.n_bins, 3):
            raise ValueError("shapes.triangles count does not match n_bins.")
        if not isinstance(locator, MeshLocator):
            raise ValueError("triangle shapes need a matching mesh locator.")
    if is_track != (track is not None):
        raise ValueError(
            "is_linearized_track does not match the linearization component."
        )
    if isinstance(locator, RectilinearLocator):
        _validate_rectilinear(binning, locator, coordinate_kind)
    elif isinstance(locator, HexLocator):
        _validate_hexagonal(binning, locator)
    elif isinstance(locator, MeshLocator):
        _validate_mesh(binning, locator)
    elif isinstance(locator, NearestBinLocator):
        if locator.centers.shape != bins.centers.shape:
            raise ValueError(
                "nearest locator centers shape does not match bins.centers."
            )
        assert locator.bin_indices is not None
        order = _active_indices(
            locator.bin_indices, bins.n_bins, "nearest locator bin_indices ordering"
        )
        _equal(
            locator.centers[order],
            bins.centers,
            "nearest locator bin_indices/centers ordering",
        )
    if track is not None:
        if not isinstance(track, TrackLinearization) or not isinstance(
            locator, NearestBinLocator
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
        _equal(
            track.linear_to_nd(get_centers(edges)[track.active_mask]),
            bins.centers,
            "linearization grid_edges/graph projected centers",
        )
        _equal(np.diff(edges)[track.active_mask], bins.sizes, "linearization bin sizes")
        if bins.measure_dim != 1:
            raise ValueError("linearization bins.measure_dim must be 1.")
    _equal(
        locator.locate(bins.centers),
        np.arange(bins.n_bins),
        "locator bin-center lookup",
    )


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
        return track.grid_edges if isinstance(track, TrackLinearization) else ()

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Full grid shape, or active-bin count for non-grid cells."""
        if self._binning.rectilinear is not None:
            return self._binning.rectilinear.shape
        track = self._binning.linearization
        if isinstance(track, TrackLinearization):
            return track.active_mask.shape
        return (self._binning.bins.n_bins,)

    @property
    def active_mask(self) -> NDArray[np.bool_]:
        """Active grid cells, or all active bins for a non-grid layout."""
        if self._binning.rectilinear is not None:
            return self._binning.rectilinear.active_mask
        track = self._binning.linearization
        if isinstance(track, TrackLinearization):
            return track.active_mask
        mask = np.ones(self._binning.bins.n_bins, dtype=bool)
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
