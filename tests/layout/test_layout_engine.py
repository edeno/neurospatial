from typing import Any

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import Point, box
from shapely.geometry import Polygon as ShapelyPoly

from neurospatial import LayoutNotBuiltError, Regions
from neurospatial.environment import Environment
from neurospatial.geometry import graph_from_binning
from neurospatial.layout import ImageMaskLayout
from neurospatial.layout.base import LayoutEngine
from neurospatial.layout.factories import (
    _ENGINES,
    _size_parameter,
    create_layout,
    get_layout_parameters,
    list_available_layouts,
)


def add_edge_distances(graph: nx.Graph) -> nx.Graph:
    for u, v in graph.edges():
        pos_u = np.array(graph.nodes[u]["pos"])
        pos_v = np.array(graph.nodes[v]["pos"])
        graph.edges[u, v]["distance"] = np.linalg.norm(pos_v - pos_u)
    return graph


def test_list_available_layouts():
    layouts = list_available_layouts()
    assert isinstance(layouts, list)
    assert "RegularGrid" in layouts
    assert "Hexagonal" in layouts
    assert "Graph" in layouts
    assert "MaskedGrid" in layouts
    assert "ImageMask" in layouts


def test_get_layout_parameters_regular_grid():
    params = get_layout_parameters("RegularGrid")
    assert "bin_size" in params
    assert "dimension_ranges" in params
    assert params["bin_size"]["annotation"] is not None


def test_get_layout_parameters_tells_required_from_defaulting_to_none():
    params = get_layout_parameters("RegularGrid")
    # bin_size has no default; dimension_ranges defaults to None.
    assert params["bin_size"]["required"] is True
    assert params["bin_size"]["default"] is None
    assert params["dimension_ranges"]["required"] is False
    assert params["dimension_ranges"]["default"] is None
    assert params["bin_count_threshold"] == {
        "annotation": int,
        "default": 0,
        "required": False,
        "kind": "KEYWORD_ONLY",
    }


def _box_cells(half_widths):
    """Containment oracle for axis-aligned rectangular cells."""

    def contains(points, centers):
        return np.all(np.abs(points[:, None] - centers[None]) < half_widths, axis=2)

    return contains


def _pointy_top_hexagons(width):
    """Containment oracle for pointy-top hexagons ``width`` across the flats."""

    def contains(points, centers):
        dx, dy = np.moveaxis(np.abs(points[:, None] - centers[None]), -1, 0)
        return (dx <= width / 2) & (0.5 * dx + np.sqrt(3) / 2 * dy <= width / 2)

    return contains


def _straight_track():
    """Two 1-unit segments along the x-axis, binned every 0.5."""
    graph = nx.Graph()
    for node in range(3):
        graph.add_node(node, pos=(float(node), 0.0))
    graph.add_edges_from([(0, 1), (1, 2)])
    return create_layout(
        "Graph",
        graph_definition=add_edge_distances(graph),
        edge_order=[(0, 1), (1, 2)],
        edge_spacing=0.0,
        bin_size=0.5,
    )


def _points_in_box(low, high):
    def sample(rng):
        return rng.uniform(low, high, (500, 2))

    return sample


def _points_on_x_axis(rng):
    return np.column_stack((rng.uniform(0.0, 2.0, 200), np.zeros(200)))


# An asymmetric mask with an inactive cell inside its bounding box.
_HOLED_MASK = np.array(
    [[True, True, True], [True, False, True], [False, False, True], [True, True, True]]
)

# layout builder, containment oracle, query points, expected size of every bin
_CELL_CASES = {
    "RegularGrid": (
        lambda: create_layout(
            "RegularGrid",
            bin_size=2.0,
            dimension_ranges=[(0, 10), (0, 10)],
            infer_active_bins=False,
        ),
        _box_cells((1.0, 1.0)),
        _points_in_box(-3, 13),
        2.0**2,
    ),
    "Hexagonal": (
        lambda: create_layout(
            "Hexagonal",
            hexagon_width=2.0,
            dimension_ranges=((0, 10), (0, 10)),
            infer_active_bins=False,
        ),
        _pointy_top_hexagons(2.0),
        _points_in_box(-3, 13),
        np.sqrt(3) / 2 * 2.0**2,
    ),
    "MaskedGrid": (
        lambda: create_layout(
            "MaskedGrid",
            active_mask=_HOLED_MASK,
            grid_edges=(np.arange(5.0), np.arange(0.0, 7.0, 2.0)),
        ),
        _box_cells((0.5, 1.0)),
        _points_in_box((-1, -1), (5, 7)),
        1.0 * 2.0,
    ),
    "ImageMask": (
        lambda: create_layout(
            "ImageMask", image_mask=_HOLED_MASK, pixel_size=(1.0, 2.0)
        ),
        _box_cells((0.5, 1.0)),
        _points_in_box((-1, -1), (4, 9)),
        1.0 * 2.0,
    ),
    "Graph": (
        _straight_track,
        lambda points, centers: np.abs(points[:, None, 0] - centers[None, :, 0]) < 0.25,
        _points_on_x_axis,
        0.5,
    ),
}


@pytest.mark.parametrize("kind", sorted(_CELL_CASES))
def test_engine_locates_each_point_in_the_cell_that_contains_it(kind):
    """A point is assigned to the bin whose cell holds it, or -1 outside all cells."""
    build, contains, sample, _ = _CELL_CASES[kind]
    layout = build()
    points = sample(np.random.default_rng(0))
    inside = contains(points, layout.bin_centers)
    assert inside.sum(axis=1).max() == 1, "cells of the oracle overlap"
    expected = np.where(inside.any(axis=1), inside.argmax(axis=1), -1)
    if kind != "Graph":  # every point on the track is in a bin
        assert np.any(expected == -1)
    np.testing.assert_array_equal(layout.binning().locator.locate(points), expected)


@pytest.mark.parametrize("kind", sorted(_CELL_CASES))
def test_engine_bin_sizes_are_the_cell_measure(kind):
    """Each bin's size is its cell's area (or a track bin's length)."""
    build, _, _, size = _CELL_CASES[kind]
    layout = build()
    np.testing.assert_allclose(layout.bin_sizes(), size, rtol=1e-12)


def test_track_bins_tile_the_track_at_their_bin_size():
    layout = _straight_track()
    assert layout.is_linearized_track
    np.testing.assert_allclose(
        layout.bin_centers, [[0.25, 0], [0.75, 0], [1.25, 0], [1.75, 0]]
    )
    assert layout.bin_sizes().sum() == pytest.approx(2.0)


def test_create_layout_invalid_kind():
    with pytest.raises(ValueError):
        create_layout("NotARealLayout", foo=1)


def test_create_layout_refuses_an_unexpected_keyword():
    """A misspelled build option is a TypeError that lists the accepted ones."""
    with pytest.raises(TypeError) as excinfo:
        create_layout(
            "RegularGrid", bin_sise=2.0, dimension_ranges=[(0.0, 10.0), (0.0, 10.0)]
        )
    message = str(excinfo.value)
    assert "'bin_sise'" in message
    assert "\nWhy: " in message
    assert "\nFix: " in message
    for name in get_layout_parameters("RegularGrid"):
        assert name in message


def test_get_layout_parameters_invalid():
    with pytest.raises(ValueError):
        get_layout_parameters("NotARealLayout")


@pytest.fixture
def grid_env_for_indexing(plus_maze_positions) -> Environment:
    """A 2D RegularGrid environment suitable for index testing."""
    return Environment.from_samples(
        positions=plus_maze_positions,  # Creates a reasonable grid
        bin_size=1.0,
        infer_active_bins=True,
        bin_count_threshold=0,
        name="GridForIndexing",
    )


MINIMAL_BUILD_PARAMS: dict[str, dict[str, Any]] = {
    "RegularGrid": {
        "bin_size": 1.0,
        "dimension_ranges": [(0, 2), (0, 2)],
        "infer_active_bins": False,  # Simplest case: all bins in range are active
    },
    "Hexagonal": {
        "hexagon_width": 1.0,
        "dimension_ranges": ((0, 2), (0, 2)),
        "infer_active_bins": False,
    },
    "Graph": {  # Requires a pre-built graph with 'pos', 'distance', 'edge_id'
        "graph_definition": nx.Graph(),  # Will be populated in test
        "edge_order": [],  # Will be populated in test
        "edge_spacing": 0.0,
        "bin_size": 0.5,
    },
    "MaskedGrid": {
        "active_mask": np.array([[True, False], [False, True]], dtype=bool),
        "grid_edges": (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])),
    },
    "ImageMask": {
        "image_mask": np.array([[True, False], [False, True]], dtype=bool),
        "pixel_size": 1.0,
    },
}

MINIMAL_BUILD_PARAMS["ShapelyPolygon"] = {
    "polygon": ShapelyPoly([(0, 0), (0, 1), (1, 1), (1, 0)]),
    "bin_size": 0.5,
}

MINIMAL_BUILD_PARAMS["TriangularMesh"] = {
    "boundary_polygon": ShapelyPoly([(0, 0), (0, 2), (2, 2), (2, 0)]),
    "point_spacing": 0.5,
}


@pytest.mark.parametrize("layout_kind", list_available_layouts())
def test_layout_engine_protocol_adherence(
    layout_kind: str, simple_graph_for_layout: nx.Graph
):
    """Every built-in layout is a LayoutEngine and offers the shared extras."""
    if layout_kind not in MINIMAL_BUILD_PARAMS:
        pytest.skip(
            f"Minimal build parameters not defined for layout kind: {layout_kind}"
        )

    params = MINIMAL_BUILD_PARAMS[layout_kind].copy()

    if layout_kind == "Graph":
        # Setup specific params for GraphLayout
        params["graph_definition"] = simple_graph_for_layout
        params["edge_order"] = list(simple_graph_for_layout.edges())

    try:
        layout: LayoutEngine = create_layout(layout_kind, **params)
    except Exception as e:
        pytest.fail(f"Failed to create layout {layout_kind} with params {params}: {e}")

    # 1. The protocol (build and binning), then the built-in engines' extras
    assert isinstance(layout, LayoutEngine)
    assert hasattr(layout, "bin_centers"), f"{layout_kind} missing bin_centers"
    assert hasattr(layout, "connectivity"), f"{layout_kind} missing connectivity"
    assert hasattr(layout, "dimension_ranges"), (
        f"{layout_kind} missing dimension_ranges"
    )
    assert hasattr(layout, "grid_edges"), f"{layout_kind} missing grid_edges"
    assert hasattr(layout, "grid_shape"), f"{layout_kind} missing grid_shape"
    assert hasattr(layout, "active_mask"), f"{layout_kind} missing active_mask"
    assert hasattr(layout, "_build_params_used"), (
        f"{layout_kind} missing _build_params_used"
    )
    assert hasattr(layout, "is_linearized_track"), (
        f"{layout_kind} missing is_linearized_track property"
    )

    # 2. Check basic types and consistency
    assert isinstance(layout.bin_centers, np.ndarray), (
        f"{layout_kind}.bin_centers not ndarray"
    )
    if layout.bin_centers.size > 0:  # Only check shape if not empty
        assert layout.bin_centers.ndim == 2, f"{layout_kind}.bin_centers not 2D"

    if (
        layout.connectivity is not None
    ):  # Optional for protocol, but usually present if active bins
        assert isinstance(layout.connectivity, nx.Graph), (
            f"{layout_kind}.connectivity not nx.Graph"
        )
        if (
            layout.bin_centers.shape[0] > 0
        ):  # If there are active bins, graph should not be None
            assert layout.connectivity.number_of_nodes() == layout.bin_centers.shape[0]

    assert isinstance(layout.is_linearized_track, bool), (
        f"{layout_kind}.is_linearized_track not bool"
    )
    assert layout.layout_type == layout_kind, f"{layout_kind}.layout_type mismatch"
    # The record keeps every build choice and no input data (positions,
    # masks, polygons, graphs, nested ranges).
    from neurospatial.layout.base import _choice

    for k, value in params.items():
        try:
            expected = _choice(value)
        except TypeError:
            assert k not in layout._build_params_used, (
                f"{layout_kind}._build_params_used records data argument '{k}'"
            )
        else:
            assert layout._build_params_used[k] == expected, (
                f"{layout_kind}._build_params_used missing choice '{k}'"
            )

    if layout.active_mask is not None:
        assert isinstance(layout.active_mask, np.ndarray), (
            f"{layout_kind}.active_mask not ndarray"
        )
        assert layout.active_mask.dtype == bool, f"{layout_kind}.active_mask not bool"
        # This check needs to be careful. If grid_shape is (N_active,), active_mask is (N_active,)
        # If grid_shape is (R,C), active_mask is (R,C)
        # For point-based layouts, grid_shape might be (N,) and active_mask (N,)
        # For grid-based, active_mask N-D shape matches grid_shape N-D shape
        if (
            layout.grid_shape is not None
            and len(layout.grid_shape) == layout.active_mask.ndim
            and (
                len(layout.grid_shape) > 1
                or (
                    len(layout.grid_shape) == 1
                    and layout.grid_shape[0] != layout.active_mask.shape[0]
                )
            )
            and not (
                len(layout.grid_shape) == 1
                and layout.grid_shape[0] == layout.active_mask.shape[0]
                and layout.active_mask.ndim == 1
            )
        ):
            assert layout.active_mask.shape == layout.grid_shape, (
                f"{layout_kind} active_mask shape vs grid_shape"
            )

    # 3. Connectivity Graph Node/Edge Attributes (if graph exists and has elements)
    if layout.connectivity and layout.connectivity.number_of_nodes() > 0:
        sample_node = next(iter(layout.connectivity.nodes()))
        node_data = layout.connectivity.nodes[sample_node]
        assert "pos" in node_data, f"{layout_kind} node missing 'pos'"
        assert isinstance(node_data["pos"], tuple), (
            f"{layout_kind} node 'pos' not tuple"
        )
        assert "source_grid_flat_index" in node_data, (
            f"{layout_kind} node missing 'source_grid_flat_index'"
        )
        assert isinstance(node_data["source_grid_flat_index"], (int, np.integer)), (
            f"{layout_kind} node 'source_grid_flat_index' not int"
        )
        assert "original_grid_nd_index" in node_data, (
            f"{layout_kind} node missing 'original_grid_nd_index'"
        )
        assert isinstance(node_data["original_grid_nd_index"], tuple), (
            f"{layout_kind} node 'original_grid_nd_index' not tuple"
        )

        if layout.connectivity.number_of_edges() > 0:
            _u, _v, edge_data = next(iter(layout.connectivity.edges(data=True)))
            assert "distance" in edge_data, f"{layout_kind} edge missing 'distance'"
            assert isinstance(edge_data["distance"], (float, np.floating)), (
                f"{layout_kind} edge 'distance' not float"
            )
            assert "edge_id" in edge_data  # Usually present

    # 4. The owned binning snapshots this layout's own bins. Its pair
    # distances are checked against the geometry in
    # test_every_geometry_measures_its_pairs_from_its_bin_centers.
    binning = layout.binning()
    assert binning.kind == layout.layout_type
    np.testing.assert_array_equal(binning.bins.centers, layout.bin_centers)
    np.testing.assert_array_equal(binning.bins.sizes, layout.bin_sizes())
    assert binning.bins.measure_dim == (
        1 if layout.is_linearized_track else layout.bin_centers.shape[1]
    )
    np.testing.assert_array_equal(
        binning.locator.locate(layout.bin_centers), np.arange(len(layout.bin_centers))
    )
    assert binning.faces.available
    np.testing.assert_array_equal(
        binning.build_faces().edges, binning.connections.pairs
    )
    assert (binning.linearization is not None) == layout.is_linearized_track


def _track_segment_of_each_bin(env: Environment) -> list[tuple]:
    """The track-graph edge whose straight segment holds each bin center."""
    graph = env._binning.linearization.graph
    segments = []
    for center in env.bin_centers:
        on = []
        for u, v in graph.edges():
            a, b = np.asarray(graph.nodes[u]["pos"]), np.asarray(graph.nodes[v]["pos"])
            t = np.clip(np.dot(center - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
            if np.linalg.norm(a + t * (b - a) - center) < 1e-9:
                on.append((u, v))
        assert len(on) == 1, f"bin center {center} lies on {len(on)} segments"
        segments.append(on[0])
    return segments


def _expected_pair_distances(env: Environment) -> np.ndarray:
    """Pair lengths computed from the bin centers and the geometry alone.

    Cartesian pairs are the straight line between centers, except track
    pairs on two segments, which are measured through the shared node.
    Polar ``(distance, angle)`` pairs are ``sqrt(Δr² + (r̄·Δθ)²)``, with
    ``Δθ`` the shorter way round, so angular pairs are the arc ``r̄·Δθ``
    and radial pairs ``|Δr|``.
    """
    centers = env.bin_centers
    pairs = env._binning.connections.all_pairs
    first, second = centers[pairs[:, 0]], centers[pairs[:, 1]]
    if env.coordinate_kind == "polar":
        d_radius = second[:, 0] - first[:, 0]
        d_angle = np.angle(np.exp(1j * (second[:, 1] - first[:, 1])))
        mean_radius = 0.5 * (first[:, 0] + second[:, 0])
        return np.hypot(d_radius, mean_radius * d_angle)
    expected = np.linalg.norm(second - first, axis=1)
    if env.layout_type == "Graph":
        graph = env._binning.linearization.graph
        segment = _track_segment_of_each_bin(env)
        for index, (u, v) in enumerate(pairs):
            shared = set(segment[u]) & set(segment[v])
            if segment[u] != segment[v]:
                assert len(shared) == 1, "track pair joins segments with no node"
                node = np.asarray(graph.nodes[shared.pop()]["pos"])
                expected[index] = np.linalg.norm(centers[u] - node) + np.linalg.norm(
                    centers[v] - node
                )
    return expected


def test_every_geometry_measures_its_pairs_from_its_bin_centers(built_environment):
    """Every engine's pairs are ordered, unique, and as long as the geometry says.

    Navigation distances feed geodesic distance, smoothing and gradients, so a
    pair length that is not the physical distance between its bins (a chord
    in place of an arc, or a shortcut across a track junction) biases every
    path-based analysis.
    """
    env = built_environment
    connections = env._binning.connections
    pairs = connections.all_pairs
    assert len(pairs) > 0
    assert np.all(pairs[:, 0] < pairs[:, 1])
    assert len(np.unique(pairs, axis=0)) == len(pairs)
    assert np.all(pairs < env.n_bins)
    np.testing.assert_allclose(
        connections.all_distances, _expected_pair_distances(env), rtol=1e-12
    )
    np.testing.assert_array_equal(env.bin_sizes, env._binning.bins.sizes)
    low, high = np.asarray(env.dimension_ranges).T
    assert np.all((env.bin_centers >= low) & (env.bin_centers <= high))


@pytest.mark.parametrize("layout_kind", list_available_layouts())
def test_size_parameter_sets_the_cell_size(
    layout_kind: str, simple_graph_for_layout: nx.Graph
):
    """The option an engine names as its cell size scales its cells.

    ``from_samples`` passes its ``bin_size`` to this option, and a refused
    ``rebin`` suggests a coarser value of it.
    """
    option = _size_parameter(layout_kind)
    if option is None:
        # A masked grid takes its cell edges, not a size.
        assert layout_kind == "MaskedGrid"
        return
    assert option in get_layout_parameters(layout_kind)
    params = MINIMAL_BUILD_PARAMS[layout_kind].copy()
    if layout_kind == "Graph":
        params["graph_definition"] = simple_graph_for_layout
        params["edge_order"] = list(simple_graph_for_layout.edges())
    coarse = create_layout(layout_kind, **params).binning().bins
    params[option] = params[option] / 2
    fine = create_layout(layout_kind, **params).binning().bins
    # Mesh triangles along the outline keep the ratio from being exact.
    np.testing.assert_allclose(
        np.mean(fine.sizes),
        np.mean(coarse.sizes) / 2**coarse.measure_dim,
        rtol=0.2,
    )


@pytest.mark.parametrize("layout_kind", list_available_layouts())
def test_engine_connectivity_is_derived_from_its_binning(
    layout_kind: str, simple_graph_for_layout: nx.Graph
):
    """``connectivity`` is the binning's graph, built once per build, never fed back."""
    params = MINIMAL_BUILD_PARAMS[layout_kind].copy()
    if layout_kind == "Graph":
        params["graph_definition"] = simple_graph_for_layout
        params["edge_order"] = list(simple_graph_for_layout.edges())
    layout = create_layout(layout_kind, **params)
    unbuilt = type(layout)()
    with pytest.raises(LayoutNotBuiltError):
        _ = unbuilt.connectivity

    graph = layout.connectivity
    expected = graph_from_binning(layout.binning())
    assert type(graph) is nx.Graph
    assert list(graph.nodes(data=True)) == list(expected.nodes(data=True))
    assert list(graph.edges(data=True)) == list(expected.edges(data=True))
    assert layout.connectivity is graph

    # The graph is a convenience copy: editing it leaves the bins unchanged.
    pairs = layout.binning().connections.pairs
    graph.remove_edges_from(list(graph.edges))
    np.testing.assert_array_equal(layout.binning().connections.pairs, pairs)

    # Building again replaces the graph.
    layout.build(**params)
    assert layout.connectivity is not graph
    assert layout.connectivity.number_of_edges() == len(pairs)


def test_located_bin_dtype_is_intp():
    """Layout index arrays are ``np.intp`` (portable), never int32/int_.

    Tripwire for the index-dtype convention documented in
    ``neurospatial.environment._trajectory``: every bin-index array is
    ``np.intp``. On 64-bit ``np.intp`` equals ``int64``, so this catches a
    regression to the narrower ``int32``.
    """
    rng = np.random.default_rng(0)
    data = rng.random((200, 2)) * 10
    points = np.array([[1.0, 1.0], [5.0, 5.0], [1e6, 1e6]])

    # _GridMixin (RegularGrid) and the Hexagonal path.
    for name, kwargs in [
        ("RegularGrid", {"bin_size": 2.0, "add_boundary_bins": False}),
        ("Hexagonal", {"hexagon_width": 2.0}),
    ]:
        layout = create_layout(name, positions=data, infer_active_bins=True, **kwargs)
        assert layout.binning().locator.locate(points).dtype == np.intp, name

    # RegularGrid uses geometric containment: the outside point maps to the -1
    # sentinel -- confirm it survives under the intp dtype.
    grid = create_layout(
        "RegularGrid",
        positions=data,
        infer_active_bins=True,
        bin_size=2.0,
        add_boundary_bins=False,
    )
    assert grid.binning().locator.locate(points)[-1] == -1

    # GraphLayout is the nearest-bin path (the historical int32 source).
    track = Environment.linear_track(endpoints=[(0.0, 0.0), (100.0, 0.0)], bin_size=5.0)
    assert track._binning.locator.locate(np.array([[10.0, 0.0]])).dtype == np.intp


def test_environment_index_surface_dtype_is_intp():
    """Every method that returns bin indices returns ``np.intp``.

    One dtype means a result from any of them indexes a field the same way,
    independent of the layout.
    """
    rng = np.random.default_rng(0)
    env = Environment.open_field(rng.random((300, 2)) * 40, bin_size=4.0)
    pts = np.array([[10.0, 10.0], [1e6, 1e6], [np.nan, 1.0]])
    times = np.arange(len(pts)) / 30.0
    regions = Regions()
    regions.add("corner", polygon=box(0.0, 0.0, 12.0, 12.0))
    env = env.with_regions(regions)
    assert env.bin_at(pts)[-2] == -1  # geometric grid: outside -> -1 sentinel
    index_arrays = {
        "bin_at": env.bin_at(pts),
        "nearest_bin": env.nearest_bin(pts),
        "nearest_bin finite": env.nearest_bin(pts[:1]),
        "nearest_bin with distance": env.nearest_bin(pts, return_dist=True)[0],
        "boundary_bins": env.boundary_bins,
        "bins_in_region": env.bins_in_region("corner"),
        "bin_sequence": env.bin_sequence(times, pts),
        "bin_sequence_with_runs": env.bin_sequence_with_runs(times, pts).bins,
        "components": env.components()[0],
        "rings": env.rings(bins=0, hops=1)[1],
        "edge_pairs": env.edge_pairs,
    }
    for name, indices in index_arrays.items():
        assert indices.dtype == np.intp, name


class TestBuildChoices:
    """capture_build_params records scalar choices and refuses undeclared data."""

    def test_declared_data_is_left_out_and_choices_are_plain_values(self):
        from neurospatial.layout.base import capture_build_params

        class Engine:
            @capture_build_params(data=("positions",))
            def build(self, *, positions, bin_size, per_axis=(1, 2), label=None):
                pass

        engine = Engine()
        engine.build(positions=np.zeros((4, 2)), bin_size=np.float64(2.0))
        assert engine._build_params_used == {
            "bin_size": 2.0,
            "per_axis": (1, 2),
            "label": None,
        }
        assert type(engine._build_params_used["bin_size"]) is float

    @pytest.mark.parametrize("style", ["named", "kwargs"])
    def test_undeclared_data_is_refused_and_the_printed_fix_works(self, style):
        """Both build signatures refuse data, and declaring it as the error says works."""
        from neurospatial.layout.base import capture_build_params

        def engine(decorator):
            if style == "named":

                class Engine:
                    @decorator
                    def build(self, *, positions, bin_size):
                        self.built = positions

            else:

                class Engine:  # the protocol's documented build(**kwargs) form
                    @decorator
                    def build(self, **kwargs):
                        self.built = kwargs["positions"]

            return Engine()

        refused = engine(capture_build_params)
        with pytest.raises(TypeError, match=r"'positions' is not a build choice") as e:
            refused.build(positions=np.zeros((4, 2)), bin_size=2.0)
        assert "@capture_build_params(data=('positions',))" in str(e.value)
        assert not hasattr(refused, "built")

        declared = engine(capture_build_params(data=("positions",)))
        declared.build(positions=np.zeros((4, 2)), bin_size=2.0)
        assert declared._build_params_used == {"bin_size": 2.0}
        assert declared.built.shape == (4, 2)

    def test_unknown_data_name_is_refused_at_decoration(self):
        from neurospatial.layout.base import capture_build_params

        with pytest.raises(TypeError, match="does not take") as error:

            class Engine:
                @capture_build_params(data=("points",))
                def build(self, *, positions):
                    pass

        assert "Fix: list only names" in str(error.value)

    def test_subclass_build_record_wins_over_its_parent_build(self):
        """A subclass that calls super().build() records its own choices."""
        from neurospatial.layout.engines._polar import _PolarGridLayout

        layout = _PolarGridLayout()
        layout.build(
            active_mask=np.ones((2, 4), dtype=bool),
            grid_edges=(np.arange(3.0), np.linspace(-np.pi, np.pi, 5)),
            circular_angle=False,
        )
        assert layout._build_params_used == {
            "connect_diagonal_neighbors": True,
            "circular_angle": False,
        }

    def test_saved_data_is_dropped_on_read_and_choices_kept(self):
        """Files that recorded input data among the parameters still load."""
        from neurospatial import Environment

        env = Environment.from_grid_mask(
            np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
        )
        data = env.to_dict()
        data["layout_parameters"]["grid_edges"] = [[0.0, 1.0, 2.0]] * 2
        restored = Environment.from_dict(data)
        assert restored.layout_parameters == env.layout_parameters
        np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)


_EMPTY_BUILDS = {
    "polygon_with_no_cell_center": lambda: Environment.from_polygon(
        Point(0, 0).buffer(10).difference(Point(0, 0).buffer(9)), bin_size=25.0
    ),
    "all_false_mask": lambda: Environment.from_grid_mask(
        np.zeros((3, 3), dtype=bool), (np.arange(4.0), np.arange(4.0))
    ),
    "hexagons_below_threshold": lambda: Environment.from_samples(
        np.random.default_rng(0).uniform(0, 20, (200, 2)),
        bin_size=2.0,
        layout="Hexagonal",
        bin_count_threshold=10_000,
    ),
}


@pytest.mark.parametrize("build", sorted(_EMPTY_BUILDS))
def test_every_engine_refuses_a_build_with_no_active_bins(build):
    """An environment with zero bins is refused, never published.

    Zero bins would put every position outside the environment, so
    occupancy and rate maps would silently come back empty.
    """
    with pytest.raises(ValueError, match="has no active bins") as error:
        _EMPTY_BUILDS[build]()
    assert "Why:" in str(error.value) and "Fix:" in str(error.value)


def _sized_build_params(layout_kind, graph, size):
    """Minimal build parameters with the engine's cell size set to ``size``."""
    params = MINIMAL_BUILD_PARAMS[layout_kind].copy()
    if layout_kind == "Graph":
        params["graph_definition"] = graph
        params["edge_order"] = list(graph.edges())
    params[_size_parameter(layout_kind)] = size
    return params


# A masked grid takes its cell edges, not a size.
_SIZED_LAYOUTS = [kind for kind in list_available_layouts() if kind != "MaskedGrid"]


@pytest.mark.parametrize("layout_kind", _SIZED_LAYOUTS)
@pytest.mark.parametrize("size", [np.nan, np.inf, 0.0, -1.0, "2"])
def test_every_engine_refuses_a_bad_cell_size_the_same_way(
    layout_kind, size, simple_graph_for_layout
):
    """NaN, infinite, non-positive and non-numeric sizes get one refusal."""
    params = _sized_build_params(layout_kind, simple_graph_for_layout, size)
    error = TypeError if isinstance(size, str) else ValueError
    with pytest.raises(error, match=r"\[E1002\][\s\S]*Fix:"):
        create_layout(layout_kind, **params)


@pytest.mark.parametrize("layout_kind", _SIZED_LAYOUTS)
@pytest.mark.parametrize("scalar", [np.float32(0.5), np.int64(1)])
def test_a_numpy_scalar_cell_size_builds_like_the_python_number(
    layout_kind, scalar, simple_graph_for_layout
):
    expected = create_layout(
        layout_kind,
        **_sized_build_params(layout_kind, simple_graph_for_layout, scalar.item()),
    )
    layout = create_layout(
        layout_kind, **_sized_build_params(layout_kind, simple_graph_for_layout, scalar)
    )
    np.testing.assert_array_equal(layout.bin_centers, expected.bin_centers)


def test_a_built_engine_owns_read_only_copies_of_its_inputs():
    """Editing the caller's mask after build cannot change the engine."""
    mask = np.array([[True, True], [True, False]])
    edges = (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    layout = create_layout("MaskedGrid", active_mask=mask, grid_edges=edges)
    mask[0, 0] = False
    edges[0][0] = -5.0
    assert layout.active_mask[0, 0]
    assert layout.grid_edges[0][0] == 0.0
    for array in (layout.bin_centers, layout.active_mask, *layout.grid_edges):
        assert not array.flags.writeable
    assert layout.binning().bins.n_bins == 3


def test_a_failed_rebuild_leaves_the_engine_unbuilt():
    """A build that raises does not leave old bins mixed with new settings."""
    layout = create_layout(
        "RegularGrid",
        bin_size=10.0,
        dimension_ranges=[(0, 30), (0, 30)],
        infer_active_bins=False,
    )
    outside = np.full((5, 2), 500.0)
    with pytest.raises(ValueError, match="No active bins"):
        layout.build(
            bin_size=5.0,
            dimension_ranges=[(0, 100), (0, 100)],
            positions=outside,
            bin_count_threshold=10,
        )
    with pytest.raises(LayoutNotBuiltError):
        layout.binning()
    with pytest.raises(LayoutNotBuiltError):
        layout.locator()


@pytest.mark.parametrize("layout_kind", list_available_layouts())
def test_every_engine_plots_its_cells_like_environment_plot(
    layout_kind, simple_graph_for_layout
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = MINIMAL_BUILD_PARAMS[layout_kind].copy()
    if layout_kind == "Graph":
        params["graph_definition"] = simple_graph_for_layout
        params["edge_order"] = list(simple_graph_for_layout.edges())
    layout = create_layout(layout_kind, **params)
    ax = layout.plot()
    assert layout_kind in ax.get_title()
    assert ax.collections or ax.lines or ax.patches
    plt.close(ax.figure)


def test_track_plot_honors_figsize_and_refuses_unknown_options(
    simple_graph_for_layout,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layout = create_layout(
        "Graph",
        graph_definition=simple_graph_for_layout,
        edge_order=list(simple_graph_for_layout.edges()),
        edge_spacing=0.0,
        bin_size=0.5,
    )
    ax = layout.plot(figsize=(2, 3))
    np.testing.assert_allclose(ax.figure.get_size_inches(), (2, 3))
    plt.close(ax.figure)
    with pytest.raises(TypeError, match="node_kwargs"):
        layout.plot(node_kwargs={})


_UNBUILT_CALLS = [
    (engine, method)
    for engine in _ENGINES
    for method in (
        "binning",
        "bin_sizes",
        "bin_shapes",
        "outline",
        "locator",
        "linear_point_to_bin_ind",
    )
    if hasattr(engine, method)
    # An unbuilt mesh has no outline yet: outline() returns None.
    and (engine.__name__, method) != ("TriangularMeshLayout", "outline")
]


@pytest.mark.parametrize(
    ("engine", "method"),
    _UNBUILT_CALLS,
    ids=[f"{engine.__name__}.{method}" for engine, method in _UNBUILT_CALLS],
)
def test_an_unbuilt_engine_names_the_method_called(engine, method):
    """Calling an unbuilt engine raises LayoutNotBuiltError naming that method.

    The message names what the caller can act on, not private state such as
    ``_full_delaunay_tri`` or ``hex_radius_``.
    """
    call = getattr(engine(), method)
    args = (np.zeros(1),) if method == "linear_point_to_bin_ind" else ()
    with pytest.raises(LayoutNotBuiltError) as excinfo:
        call(*args)
    assert excinfo.value.layout_name == engine.__name__
    assert excinfo.value.attribute == method


@pytest.mark.parametrize(
    ("layout_kind", "size"),
    [("RegularGrid", "bin_size"), ("Hexagonal", "hexagon_width")],
)
def test_an_infinite_position_row_is_a_missing_sample(layout_kind, size):
    """Engines ignore a row with an infinite coordinate, as from_samples does.

    Built directly, RegularGrid refused it with a message about a non-finite
    range and Hexagonal raised OverflowError.
    """
    positions = np.random.default_rng(1).uniform(0, 20, (200, 2))
    with_inf = np.insert(positions, 7, [np.inf, 3.0], axis=0)
    expected = create_layout(layout_kind, positions=positions, **{size: 2.0})
    layout = create_layout(layout_kind, positions=with_inf, **{size: 2.0})
    np.testing.assert_array_equal(layout.bin_centers, expected.bin_centers)


@pytest.mark.parametrize(
    ("build", "missing"),
    [
        (lambda: create_layout("RegularGrid", positions=np.zeros((3, 2))), "bin_size"),
        (
            lambda: ImageMaskLayout().build(image_mask=np.ones((2, 2), dtype=bool)),
            "pixel_size",
        ),
    ],
    ids=["create_layout", "engine_build"],
)
def test_a_missing_build_option_is_named_with_a_fix(build, missing):
    """Python's bare 'missing a required argument' names neither the engine nor a fix."""
    with pytest.raises(
        TypeError, match=rf"build\(\) is missing[^\n]*{missing}"
    ) as error:
        build()
    assert "Why:" in str(error.value) and "Fix:" in str(error.value)
