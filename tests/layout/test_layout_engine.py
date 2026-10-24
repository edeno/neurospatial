from typing import Any

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import Polygon as ShapelyPoly
from shapely.geometry import box

from neurospatial import LayoutNotBuiltError, Regions
from neurospatial.environment import Environment
from neurospatial.geometry import graph_from_binning
from neurospatial.layout.base import LayoutEngine
from neurospatial.layout.factories import (
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


def test_create_layout_regular_grid():
    rng = np.random.default_rng(42)
    data = rng.random((100, 2)) * 10
    layout = create_layout(
        "RegularGrid",
        bin_size=2.0,
        positions=data,
        infer_active_bins=True,
        add_boundary_bins=False,
    )
    assert hasattr(layout, "bin_centers")
    assert layout.bin_centers.ndim == 2
    assert layout.connectivity is not None
    assert layout.active_mask is not None
    assert layout.grid_edges is not None
    assert not layout.is_linearized_track


def test_regular_grid_locates_points():
    rng = np.random.default_rng(42)
    data = rng.random((100, 2)) * 10
    layout = create_layout(
        "RegularGrid",
        bin_size=2.0,
        positions=data,
        infer_active_bins=True,
        add_boundary_bins=False,
    )
    points = np.array([[1.0, 1.0], [5.0, 5.0], [100.0, 100.0]])
    inds = layout.binning().locator.locate(points)
    assert inds.shape == (3,)
    assert np.any(inds >= 0)


def test_regular_grid_bin_area_volume():
    rng = np.random.default_rng(42)
    data = rng.random((100, 2)) * 10
    layout = create_layout(
        "RegularGrid",
        bin_size=2.0,
        positions=data,
        infer_active_bins=True,
        add_boundary_bins=False,
    )
    areas = layout.bin_sizes()
    assert np.allclose(areas, areas[0])
    assert areas.shape[0] == layout.bin_centers.shape[0]


def test_create_layout_hexagonal():
    rng = np.random.default_rng(42)
    data = rng.random((100, 2)) * 10
    layout = create_layout(
        "Hexagonal",
        hexagon_width=2.0,
        positions=data,
        infer_active_bins=True,
    )
    assert hasattr(layout, "bin_centers")
    assert layout.bin_centers.ndim == 2
    assert layout.connectivity is not None
    assert not layout.is_linearized_track


def test_hexagonal_locates_points():
    rng = np.random.default_rng(42)
    data = rng.random((100, 2)) * 10
    layout = create_layout(
        "Hexagonal",
        hexagon_width=2.0,
        positions=data,
        infer_active_bins=True,
    )
    points = np.array([[1.0, 1.0], [5.0, 5.0], [100.0, 100.0]])
    inds = layout.binning().locator.locate(points)
    assert inds.shape == (3,)
    assert np.any(inds >= 0)


def test_hexagonal_bin_area_volume():
    rng = np.random.default_rng(42)
    data = rng.random((100, 2)) * 10
    layout = create_layout(
        "Hexagonal",
        hexagon_width=2.0,
        positions=data,
        infer_active_bins=True,
    )
    areas = layout.bin_sizes()
    assert np.allclose(areas, areas[0])
    assert areas.shape[0] == layout.bin_centers.shape[0]


def test_create_layout_masked_grid():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    edges = (
        np.arange(6).astype(float),
        np.arange(6).astype(float),
    )  # Ensure float for grid_edges
    layout = create_layout(
        "MaskedGrid",
        active_mask=mask,
        grid_edges=edges,
    )
    assert hasattr(layout, "bin_centers")
    assert layout.bin_centers.ndim == 2
    assert layout.bin_centers.shape[0] == np.sum(mask)  # Should be 9
    assert layout.connectivity is not None
    assert not layout.is_linearized_track


def test_create_layout_image_mask():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    layout = create_layout(
        "ImageMask",
        image_mask=mask,
        pixel_size=1.0,
    )
    assert hasattr(layout, "bin_centers")
    assert layout.bin_centers.ndim == 2
    assert layout.connectivity is not None
    assert not layout.is_linearized_track


def test_image_mask_locates_points():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    layout = create_layout(
        "ImageMask",
        image_mask=mask,
        pixel_size=1.0,
    )
    points = np.array([[2.5, 2.5], [0.5, 0.5]])
    inds = layout.binning().locator.locate(points)
    assert inds.shape == (2,)
    assert inds[0] >= 0


def test_create_layout_graph():
    G = nx.Graph()
    G.add_node(0, pos=(0, 0))
    G.add_node(1, pos=(1, 0))
    G.add_node(2, pos=(2, 0))
    G.add_edges_from([(0, 1), (1, 2)])
    G = add_edge_distances(G)
    edge_order = [(0, 1), (1, 2)]
    layout = create_layout(
        "Graph",
        graph_definition=G,
        edge_order=edge_order,
        edge_spacing=0.0,
        bin_size=0.5,
    )
    assert hasattr(layout, "bin_centers")
    assert layout.bin_centers.ndim == 2
    # Graph: 0 --1m-- 1 --1m-- 2. Total length 2m. Bin size 0.5m. Expected 4 bins.
    assert layout.bin_centers.shape[0] == 4
    assert layout.connectivity is not None
    assert layout.is_linearized_track


def test_graph_locates_points():
    G = nx.Graph()
    G.add_node(0, pos=(0, 0))
    G.add_node(1, pos=(1, 0))
    G.add_node(2, pos=(2, 0))
    G.add_edges_from([(0, 1), (1, 2)])
    G = add_edge_distances(G)
    edge_order = [(0, 1), (1, 2)]
    layout = create_layout(
        "Graph",
        graph_definition=G,
        edge_order=edge_order,
        edge_spacing=0.0,
        bin_size=0.5,
    )
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    inds = layout.binning().locator.locate(points)
    assert inds.shape == (3,)
    assert np.any(inds >= 0)


def test_graph_bin_area_volume():
    G = nx.Graph()
    G.add_node(0, pos=(0, 0))
    G.add_node(1, pos=(1, 0))
    G.add_node(2, pos=(2, 0))
    G.add_edges_from([(0, 1), (1, 2)])
    G = add_edge_distances(G)
    edge_order = [(0, 1), (1, 2)]
    layout = create_layout(
        "Graph",
        graph_definition=G,
        edge_order=edge_order,
        edge_spacing=0.0,
        bin_size=0.5,
    )
    lengths = layout.bin_sizes()
    assert np.allclose(lengths, lengths[0])
    assert lengths.shape[0] == layout.bin_centers.shape[0]


def test_create_layout_invalid_kind():
    with pytest.raises(ValueError):
        create_layout("NotARealLayout", foo=1)


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

    # 4. The owned binning snapshots this layout's own bins and connectivity.
    binning = layout.binning()
    edges = list(layout.connectivity.edges(data=True))
    assert binning.kind == layout.layout_type
    np.testing.assert_array_equal(binning.bins.centers, layout.bin_centers)
    np.testing.assert_array_equal(binning.bins.sizes, layout.bin_sizes())
    np.testing.assert_array_equal(
        binning.connections.pairs,
        np.array([(u, v) for u, v, _ in edges]).reshape(-1, 2),
    )
    np.testing.assert_array_equal(
        binning.connections.pair_distances, [data["distance"] for _, _, data in edges]
    )
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

    # GraphLayout is the _KDTreeMixin path (the historical int32 source).
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
