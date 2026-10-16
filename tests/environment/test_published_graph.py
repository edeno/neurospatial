"""The published connectivity graph is frozen once and shares immutable values."""

from __future__ import annotations

import copy
import pickle

import networkx as nx
import numpy as np
import pytest
from environment_helpers import publish_binning

from neurospatial import Environment


def _annotated_environment():
    """A grid whose graph carries list, dict and array attributes."""
    source = Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool), grid_edges=(np.arange(4.0),) * 2
    )
    graph = source.to_networkx()
    graph.nodes[0]["tags"] = ["start", 1]
    graph.nodes[0]["details"] = {"nested": {"name": "original"}}
    graph.nodes[0]["samples"] = np.array([1.0, 2.0])
    edge = next(iter(graph.edges))
    graph.edges[edge]["notes"] = ["measured"]
    env = publish_binning(
        source._binning,
        connectivity=graph,
        dimension_ranges=source.dimension_ranges,
        layout_type="MaskedGrid",
        category="mask",
    )
    return env, graph, edge


@pytest.fixture(params=["fresh", "deepcopy", "pickle"])
def annotated(request):
    env, graph, edge = _annotated_environment()
    if request.param == "deepcopy":
        env = copy.deepcopy(env)
    elif request.param == "pickle":
        env = pickle.loads(pickle.dumps(env))
    return env, graph, edge


def test_attribute_reads_return_the_stored_immutable_values(annotated):
    env, _, edge = annotated
    node = env.connectivity.nodes[0]
    # Reads do not copy: the same stored object comes back each time.
    assert node["pos"] is env.connectivity.nodes[0]["pos"]
    assert node["tags"] is env.connectivity.nodes[0]["tags"]
    assert node["tags"] == ("start", 1)
    assert env.connectivity.edges[edge]["notes"] == ("measured",)
    assert node["details"]["nested"]["name"] == "original"
    with pytest.raises(TypeError, match="read-only"):
        node["details"]["nested"]["name"] = "changed"
    samples = node["samples"]
    np.testing.assert_array_equal(samples, [1.0, 2.0])
    assert not samples.flags.writeable
    with pytest.raises(ValueError):
        samples.setflags(write=True)


def test_every_edit_of_the_published_graph_is_refused(annotated):
    env, _, edge = annotated
    graph = env.connectivity
    with pytest.raises(TypeError, match="read-only"):
        graph.nodes[0]["pos"] = (9.0, 9.0)
    with pytest.raises(TypeError, match="read-only"):
        graph.edges[edge]["distance"] = 9.0
    with pytest.raises(TypeError, match="read-only"):
        graph.nodes[0].update(pos=(9.0, 9.0))
    with pytest.raises(nx.NetworkXError):
        graph.add_edge(0, 8)
    with pytest.raises(nx.NetworkXError):
        graph.remove_node(0)
    # The undirected edge shares one attribute record in both directions.
    u, v = edge
    assert graph._adj[u][v] is graph._adj[v][u]


def test_publication_leaves_the_supplied_graph_editable():
    env, graph, edge = _annotated_environment()
    graph.nodes[0]["tags"].append("caller")
    graph.edges[edge]["distance"] = 99.0
    graph.add_edge(0, 8)
    assert env.connectivity.nodes[0]["tags"] == ("start", 1)
    assert env.connectivity.edges[edge]["distance"] != 99.0
    assert not env.connectivity.has_edge(0, 8)


def test_to_networkx_returns_an_independent_editable_graph(annotated):
    env, _, edge = annotated
    exported = env.to_networkx()
    assert type(exported) is nx.Graph
    assert list(exported.nodes) == list(env.connectivity.nodes)
    assert list(exported.edges) == list(env.connectivity.edges)
    node = exported.nodes[0]
    # Stored types come back as editable values at every level.
    assert node["tags"] == ["start", 1]
    assert isinstance(node["pos"], tuple)
    node["tags"].append("caller")
    node["details"]["nested"]["name"] = "changed"
    node["samples"][0] = -99.0
    exported.edges[edge]["notes"].append("caller")
    exported.edges[edge]["distance"] = 99.0
    exported.add_edge(0, 8)
    published = env.connectivity
    assert published.nodes[0]["tags"] == ("start", 1)
    assert published.nodes[0]["details"]["nested"]["name"] == "original"
    np.testing.assert_array_equal(published.nodes[0]["samples"], [1.0, 2.0])
    assert published.edges[edge]["notes"] == ("measured",)
    assert published.edges[edge]["distance"] != 99.0
    assert not published.has_edge(0, 8)


def test_shortest_paths_match_center_distances(annotated):
    env, _, _ = annotated
    # Independent oracle: straight-line grid distances between bin centers.
    centers = env.bin_centers
    oracle = nx.Graph()
    for u, v in env.edge_pairs:
        oracle.add_edge(
            int(u), int(v), distance=float(np.linalg.norm(centers[u] - centers[v]))
        )
    expected = nx.single_source_dijkstra_path_length(oracle, 0, weight="distance")
    actual = nx.single_source_dijkstra_path_length(
        env.connectivity, 0, weight="distance"
    )
    assert actual.keys() == expected.keys()
    for node, length in expected.items():
        assert actual[node] == pytest.approx(length, rel=1e-12)


def _edge_id_environments():
    from shapely.geometry import Polygon

    from neurospatial.layout import create_layout

    positions = np.random.default_rng(3).uniform(0, 30, (3000, 2))
    track = nx.Graph()
    for node, pos in enumerate([(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (40.0, 0.0)]):
        track.add_node(node, pos=pos)
    for u, v in [(0, 1), (1, 2), (1, 3)]:
        track.add_edge(u, v, distance=20.0)
    return {
        "grid": lambda: Environment.from_samples(positions, bin_size=3.0),
        "hexagonal": lambda: Environment.from_samples(
            positions, bin_size=3.0, layout="Hexagonal"
        ),
        "mesh": lambda: Environment.from_layout(
            create_layout(
                "TriangularMesh",
                boundary_polygon=Polygon([(0, 0), (30, 0), (30, 30), (0, 30)]),
                point_spacing=4.0,
            )
        ),
        "track": lambda: Environment.from_graph(
            track, [(0, 1), (1, 2), (1, 3)], bin_size=2.0, edge_spacing=3.0
        ),
    }


def _edge_ids_are_pair_rows(env):
    rows = {frozenset(map(int, pair)): i for i, pair in enumerate(env.edge_pairs)}
    assert sorted(rows.values()) == list(range(env.edge_pairs.shape[0]))
    for u, v, data in env.connectivity.edges(data=True):
        assert data["edge_id"] == rows[frozenset((u, v))]


@pytest.mark.parametrize("layout", ["grid", "hexagonal", "mesh", "track"])
def test_subset_numbers_edge_ids_by_kept_pair_row(layout):
    env = _edge_id_environments()[layout]()
    _edge_ids_are_pair_rows(env)
    # Dropping every third bin removes pairs from the middle of the list.
    child = env.subset(bins=np.arange(env.n_bins) % 3 != 0)
    assert child.edge_pairs.shape[0] < env.edge_pairs.shape[0]
    _edge_ids_are_pair_rows(child)


@pytest.mark.parametrize("circular", [True, False])
def test_polar_edge_ids_number_pair_rows_across_the_seam(circular):
    env = Environment.from_polar_egocentric(
        distance_range=(0.0, 30.0),
        angle_range=(-np.pi, np.pi) if circular else (-np.pi / 2, np.pi / 2),
        distance_bin_size=5.0,
        angle_bin_size=np.pi / 6,
        circular_angle=circular,
    )
    _edge_ids_are_pair_rows(env)
    _edge_ids_are_pair_rows(Environment.from_dict(env.to_dict()))
