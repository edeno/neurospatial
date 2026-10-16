"""The published connectivity graph is frozen once and shares immutable values."""

from __future__ import annotations

import copy
import pickle

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment


@pytest.fixture(params=["fresh", "deepcopy", "pickle"])
def published(request):
    env = Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool), grid_edges=(np.arange(4.0),) * 2
    )
    if request.param == "deepcopy":
        env = copy.deepcopy(env)
    elif request.param == "pickle":
        env = pickle.loads(pickle.dumps(env))
    return env


def test_attribute_reads_return_the_stored_immutable_values(published):
    node = published.connectivity.nodes[0]
    # Reads do not copy: the same stored object comes back each time.
    assert node["pos"] is published.connectivity.nodes[0]["pos"]
    assert node["pos"] == tuple(published.bin_centers[0])
    assert isinstance(node["original_grid_nd_index"], tuple)
    for _, _, data in published.connectivity.edges(data=True):
        assert isinstance(data["vector"], tuple)


def test_every_edit_of_the_published_graph_is_refused(published):
    graph = published.connectivity
    edge = next(iter(graph.edges))
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


def test_to_networkx_returns_an_independent_editable_graph(published):
    exported = published.to_networkx()
    edge = next(iter(exported.edges))
    assert type(exported) is nx.Graph
    assert list(exported.nodes) == list(published.connectivity.nodes)
    assert list(exported.edges) == list(published.connectivity.edges)
    assert isinstance(exported.nodes[0]["pos"], tuple)
    exported.nodes[0]["pos"] = (-99.0, -99.0)
    exported.edges[edge]["distance"] = 99.0
    exported.add_edge(0, 8)
    graph = published.connectivity
    assert graph.nodes[0]["pos"] == tuple(published.bin_centers[0])
    assert graph.edges[edge]["distance"] != 99.0
    assert not graph.has_edge(0, 8)


def test_graph_follows_the_binning(published):
    """Nodes are bins and edges are neighbor pairs with their distances."""
    graph = published.connectivity
    assert list(graph.nodes) == list(range(published.n_bins))
    pairs = {frozenset(map(int, pair)) for pair in published.edge_pairs}
    assert {frozenset(edge) for edge in graph.edges} == pairs
    distances = published._binning.bins.pair_distances
    for row, (u, v) in enumerate(published.edge_pairs):
        assert graph.edges[int(u), int(v)]["distance"] == distances[row]


def test_shortest_paths_match_center_distances(published):
    env = published
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
    child = env.subset(mask=np.arange(env.n_bins) % 3 != 0)
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


def test_wall_removed_pairs_renumber_edge_ids_by_kept_pair_row():
    """Walls drop pairs; each remaining edge_id is its row in env.edge_pairs."""
    from shapely.geometry import LineString, box

    from neurospatial.arena import OpenArena

    open_env = Environment.from_polygon(box(0, 0, 10, 10), bin_size=1.0)
    walled = Environment.from_polygon(
        box(0, 0, 10, 10),
        bin_size=1.0,
        arena=OpenArena(box(0, 0, 10, 10), walls=(LineString([(5, 0), (5, 8)]),)),
    )
    assert walled.edge_pairs.shape[0] < open_env.edge_pairs.shape[0]
    _edge_ids_are_pair_rows(walled)
    _edge_ids_are_pair_rows(Environment.from_dict(walled.to_dict()))


def test_combined_environment_graph_order_survives_a_round_trip():
    """A read-back combined environment iterates neighbors in the same order."""
    from shapely.geometry import box

    first = Environment.from_polygon(box(0, 0, 10, 10), bin_size=2.0)
    second = Environment.from_polygon(box(12, 0, 22, 10), bin_size=2.0)
    combined = Environment.combine({"a": first, "b": second})
    restored = Environment.from_dict(combined.to_dict())

    def neighbors(env):
        return [list(env.connectivity.neighbors(node)) for node in env.connectivity]

    assert neighbors(restored) == neighbors(combined)
    # Equally short paths are broken by neighbor order, so they agree too.
    for target in range(0, combined.n_bins, 3):
        assert list(restored.path_between(bins=[0, target])) == list(
            combined.path_between(bins=[0, target])
        )
