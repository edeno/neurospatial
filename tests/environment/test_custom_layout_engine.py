"""A layout engine needs only build(...) and binning(); the rest follows from it."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import Polygon

from neurospatial import Environment
from neurospatial.binning import (
    BinConnections,
    Binning,
    FaceGeometry,
    KnownFaces,
    MissingFaces,
    NearestBinLocator,
    RectilinearGrid,
    RectilinearLocator,
    SpatialBins,
)
from neurospatial.layout import create_layout
from neurospatial.layout.base import LayoutEngine

RNG = np.random.default_rng(13)


class _RecordedGrid:
    """A rectangular grid built from scratch from public binning records."""

    def build(self, *, bin_size: float, extent: tuple[float, float]) -> None:
        edges = tuple(
            np.arange(0.0, length + bin_size / 2, bin_size) for length in extent
        )
        shape = tuple(len(edge) - 1 for edge in edges)
        centers = np.stack(
            np.meshgrid(*[(e[:-1] + e[1:]) / 2 for e in edges], indexing="ij"), -1
        ).reshape(-1, 2)
        index = np.arange(np.prod(shape)).reshape(shape)
        # Four-neighbor pairs, listed column pairs first to differ from the
        # built-in engine's pair order.
        pairs = np.concatenate(
            [
                np.column_stack([index[:, :-1].ravel(), index[:, 1:].ravel()]),
                np.column_stack([index[:-1, :].ravel(), index[1:, :].ravel()]),
            ]
        )
        n_pairs = len(pairs)
        mask = np.ones(shape, dtype=bool)
        self._binning = Binning(
            "RecordedGrid",
            SpatialBins.from_arrays(centers, np.full(len(centers), bin_size**2), 2),
            BinConnections.from_arrays(pairs, np.full(n_pairs, bin_size)),
            RectilinearLocator(edges, shape, mask, None),
            KnownFaces(
                FaceGeometry(
                    pairs,
                    np.full(n_pairs, bin_size),
                    np.full(n_pairs, bin_size),
                    np.full(len(centers), bin_size**2),
                )
            ),
            rectilinear=RectilinearGrid(edges, shape, mask),
        )

    def binning(self) -> Binning:
        return self._binning


class _RecordedPoints:
    """Bins at recorded points on a ring, located by the nearest center."""

    def build(self, *, n_points: int, radius: float) -> None:
        angles = np.linspace(0.0, 2 * np.pi, n_points, endpoint=False)
        centers = radius * np.column_stack([np.cos(angles), np.sin(angles)])
        pairs = np.column_stack(
            [np.arange(n_points), (np.arange(n_points) + 1) % n_points]
        )
        distances = np.linalg.norm(centers[pairs[:, 1]] - centers[pairs[:, 0]], axis=1)
        self._binning = Binning(
            "RecordedPoints",
            SpatialBins.from_arrays(centers, np.ones(n_points), 2),
            BinConnections.from_arrays(pairs, distances),
            NearestBinLocator(centers, max_distance=radius / 4),
            MissingFaces("Ring points have no cells."),
            dimension_ranges=((-radius, radius), (-radius, radius)),
        )

    def binning(self) -> Binning:
        return self._binning


class _WrappedEngine:
    """A plain object that publishes a built-in engine's binning, nothing else."""

    def build(self, kind: str, **params: Any) -> None:
        self._binning = create_layout(kind, **params).binning()

    def binning(self) -> Binning:
        return self._binning


def _from_layout(engine: object) -> Environment:
    return Environment.from_layout(cast("LayoutEngine", engine))


def _edge_attributes(graph: nx.Graph, *, drop: frozenset[str] = frozenset()):
    return {
        frozenset((u, v)): {
            key: value for key, value in data.items() if key not in drop
        }
        for u, v, data in graph.edges(data=True)
    }


def _node_attributes(graph: nx.Graph):
    return {node: dict(data) for node, data in graph.nodes(data=True)}


def _assert_same_edge_geometry(derived: nx.Graph, engine: nx.Graph) -> None:
    """Edges match exactly; float attributes match to rounding.

    The engine and the derived graph compute distances and angles separately
    (scalar ``math.atan2`` against vectorized ``np.arctan2``), and NumPy's
    SIMD paths may round the last bit differently on some CPUs.
    """
    derived_edges, engine_edges = _edge_attributes(derived), _edge_attributes(engine)
    assert derived_edges.keys() == engine_edges.keys()
    for edge, attributes in derived_edges.items():
        expected = engine_edges[edge]
        assert attributes.keys() == expected.keys(), edge
        for key, value in attributes.items():
            if isinstance(value, int):
                assert value == expected[key], (edge, key)
            else:
                np.testing.assert_allclose(
                    value,
                    expected[key],
                    rtol=1e-12,
                    atol=1e-12,
                    err_msg=f"{edge} {key}",
                )


def test_engine_has_only_build_and_binning() -> None:
    public = {name for name in dir(_RecordedGrid) if not name.startswith("_")}
    assert public == {"build", "binning"}
    assert _RecordedGrid.__mro__ == (_RecordedGrid, object)


def test_grid_engine_matches_builtin_grid() -> None:
    engine = _RecordedGrid()
    engine.build(bin_size=2.0, extent=(6.0, 8.0))
    env = _from_layout(engine)
    builtin = Environment.from_layout(
        create_layout(
            "RegularGrid",
            bin_size=2.0,
            dimension_ranges=[(0.0, 6.0), (0.0, 8.0)],
            infer_active_bins=False,
            connect_diagonal_neighbors=False,
        )
    )

    np.testing.assert_array_equal(env.bin_centers, builtin.bin_centers)
    assert _node_attributes(env.connectivity) == _node_attributes(builtin.connectivity)
    # Edge IDs number each environment's own pair rows; everything else agrees.
    assert _edge_attributes(
        env.connectivity, drop=frozenset({"edge_id"})
    ) == _edge_attributes(builtin.connectivity, drop=frozenset({"edge_id"}))
    rows = {frozenset(map(int, pair)): i for i, pair in enumerate(env.edge_pairs)}
    assert all(
        data["edge_id"] == rows[frozenset((u, v))]
        for u, v, data in env.connectivity.edges(data=True)
    )
    assert env.grid_shape == builtin.grid_shape == (3, 4)
    for ours, theirs in zip(env.grid_edges, builtin.grid_edges, strict=True):
        np.testing.assert_array_equal(ours, theirs)
    np.testing.assert_array_equal(env.active_mask, builtin.active_mask)
    assert env.dimension_ranges == builtin.dimension_ranges
    points = RNG.uniform(-1.0, 9.0, (500, 2))
    np.testing.assert_array_equal(env.bin_at(points), builtin.bin_at(points))
    assert env.layout_type == "RecordedGrid"
    assert env.layout_parameters == {}
    np.testing.assert_allclose(
        env.compute_kernel(bandwidth=2.0, cache=False, mode="density"),
        builtin.compute_kernel(bandwidth=2.0, cache=False, mode="density"),
    )


def test_point_engine_matches_explicit_bins() -> None:
    engine = _RecordedPoints()
    engine.build(n_points=24, radius=10.0)
    env = _from_layout(engine)
    explicit = Environment.from_bins(engine.binning())

    np.testing.assert_array_equal(env.bin_centers, explicit.bin_centers)
    assert _node_attributes(env.connectivity) == _node_attributes(explicit.connectivity)
    assert _edge_attributes(env.connectivity) == _edge_attributes(explicit.connectivity)
    assert env.grid_shape == explicit.grid_shape == (24,)
    assert env.grid_edges == explicit.grid_edges == ()
    assert env.dimension_ranges == ((-10.0, 10.0), (-10.0, 10.0))
    assert explicit.dimension_ranges == env.dimension_ranges
    points = RNG.uniform(-12.0, 12.0, (500, 2))
    located = env.bin_at(points)
    np.testing.assert_array_equal(located, explicit.bin_at(points))
    assert np.any(located == -1) and np.any(located >= 0)


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("Hexagonal", {"hexagon_width": 4.0, "dimension_ranges": [(0, 30), (0, 20)]}),
        (
            "TriangularMesh",
            {
                "boundary_polygon": Polygon(
                    [(0, 0), (30, 0), (30, 30), (0, 30)],
                    [[(10, 10), (20, 10), (20, 20), (10, 20)]],
                ),
                "point_spacing": 4.0,
            },
        ),
    ],
)
def test_non_grid_engine_matches_builtin_engine(kind, params) -> None:
    """A plain wrapper publishes what the engine's own attributes describe."""
    source = create_layout(kind, **params)
    engine = _WrappedEngine()
    engine.build(kind, **params)
    env = _from_layout(engine)
    builtin = Environment.from_layout(source)

    np.testing.assert_array_equal(env.bin_centers, source.bin_centers)
    assert _node_attributes(env.connectivity) == _node_attributes(source.connectivity)
    _assert_same_edge_geometry(env.connectivity, source.connectivity)
    assert env.grid_shape == source.grid_shape
    np.testing.assert_array_equal(env.active_mask, source.active_mask)
    np.testing.assert_array_equal(
        np.asarray(env.dimension_ranges), np.asarray(source.dimension_ranges)
    )
    points = RNG.uniform(-2.0, 32.0, (500, 2))
    np.testing.assert_array_equal(env.bin_at(points), builtin.bin_at(points))


def _track(spacing: float | list[float]):
    graph = nx.Graph()
    for node, position in enumerate(
        [(0.0, 0.0), (30.0, 0.0), (30.0, 20.0), (60.0, 0.0)]
    ):
        graph.add_node(node, pos=position)
    for u, v, length in [(0, 1, 30.0), (1, 2, 20.0), (1, 3, 30.0)]:
        graph.add_edge(u, v, distance=length)
    return create_layout(
        "Graph",
        graph_definition=graph,
        edge_order=[(0, 1), (1, 2), (1, 3)],
        edge_spacing=spacing,
        bin_size=7.0,
    )


@pytest.mark.parametrize("spacing", [0.0, 5.0, [0.0, 3.0]])
def test_track_graph_follows_binning(spacing) -> None:
    layout = _track(spacing)
    env = Environment.from_layout(layout)
    engine = layout.connectivity
    graph = env.connectivity
    for node in graph.nodes:
        assert graph.nodes[node]["pos_1D"] == engine.nodes[node]["pos_1D"]
        assert (
            graph.nodes[node]["source_edge_id"] == engine.nodes[node]["source_edge_id"]
        )
    # Flat indices name cells of the full linear grid, gaps included.
    flat = [graph.nodes[node]["source_grid_flat_index"] for node in graph.nodes]
    assert env.active_mask[flat].all()
    np.testing.assert_array_equal(flat, np.flatnonzero(env.active_mask))
    # Vectors point from the lower to the higher bin ID, as on every layout.
    centers = env.bin_centers
    for u, v, data in graph.edges(data=True):
        low, high = sorted((u, v))
        np.testing.assert_array_equal(data["vector"], centers[high] - centers[low])
    edges = env.get_edge_attributes()
    vectors = edges[["vector_dim0", "vector_dim1"]].to_numpy()
    np.testing.assert_array_equal(
        vectors, centers[edges["target"]] - centers[edges["source"]]
    )


def test_polar_edge_ids_number_pair_rows() -> None:
    env = Environment.from_polar_egocentric(
        distance_range=(0.0, 20.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=5.0,
        angle_bin_size=np.pi / 4,
    )
    rows = {frozenset(map(int, pair)): i for i, pair in enumerate(env.edge_pairs)}
    assert sorted(rows.values()) == list(range(len(rows)))
    for u, v, data in env.connectivity.edges(data=True):
        assert data["edge_id"] == rows[frozenset((u, v))]


def test_environment_binning_records_its_ranges() -> None:
    env = Environment.from_layout(
        create_layout(
            "Hexagonal", hexagon_width=4.0, dimension_ranges=[(0, 30), (0, 20)]
        )
    )
    assert env.binning.dimension_ranges == env.dimension_ranges
    explicit = replace(env.binning, faces=MissingFaces("Not needed here."))
    assert Environment.from_bins(explicit).dimension_ranges == env.dimension_ranges
    subset = env.subset(mask=np.arange(env.n_bins) < env.n_bins // 2)
    assert subset.binning.dimension_ranges == subset.dimension_ranges


def test_binning_ranges_need_one_pair_per_axis() -> None:
    engine = _RecordedPoints()
    engine.build(n_points=6, radius=1.0)
    binning = engine.binning()
    with pytest.raises(ValueError, match="one \\(low, high\\) pair per axis"):
        Binning(
            binning.kind,
            binning.bins,
            binning.connections,
            binning.locator,
            binning.faces,
            dimension_ranges=((0.0, 1.0),),
        )


def test_binning_must_return_a_binning() -> None:
    class _Broken:
        def build(self) -> None:
            pass

        def binning(self) -> object:
            return {"centers": np.zeros((1, 2))}

    with pytest.raises(TypeError, match=r"_Broken.binning\(\) returned dict"):
        _from_layout(_Broken())


def test_explicit_mesh_and_track_bins_get_the_layout_node_attributes() -> None:
    mesh = Environment.from_layout(
        create_layout(
            "TriangularMesh",
            boundary_polygon=Polygon(
                [(0, 0), (30, 0), (30, 30), (0, 30)],
                [[(10, 10), (20, 10), (20, 20), (10, 20)]],
            ),
            point_spacing=4.0,
        )
    )
    track = Environment.from_layout(_track(5.0))
    for env in (mesh, track):
        explicit = Environment.from_bins(
            replace(env.binning, faces=MissingFaces("Not needed here."))
        )
        assert _node_attributes(explicit.connectivity) == _node_attributes(
            env.connectivity
        )
    # Mesh nodes name their triangle, which differs from the bin ID once a
    # hole removes triangles.
    flat = [
        mesh.connectivity.nodes[n]["source_grid_flat_index"] for n in mesh.connectivity
    ]
    assert flat != list(range(mesh.n_bins))
    assert mesh.active_mask[flat].all()


@pytest.mark.parametrize("engine", [_RecordedPoints(), _WrappedEngine()])
def test_an_engine_with_build_and_binning_is_a_layout_engine(engine):
    assert isinstance(engine, LayoutEngine)
