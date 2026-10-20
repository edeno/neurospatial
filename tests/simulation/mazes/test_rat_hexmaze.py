"""Tests for the Rat HexMaze: four mouse HexMaze modules joined by five bridges.

The geometry is checked against the published apparatus: each mouse HexMaze
module has 24 nodes and 30 equal corridors meeting at 120° and spans about
2 × 1.9 m (Alonso et al. 2021); the rat HexMaze has 96 nodes, three small and
two large bridges and spans about 9 × 5 m (Wijnen et al. 2024).
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString

from neurospatial.simulation.mazes._base import MazeEnvironments
from neurospatial.simulation.mazes.rat_hexmaze import RatHexmazeDims, make_rat_hexmaze

DIMS = RatHexmazeDims()
BIN_SIZE = 4.0


def _graph(maze: MazeEnvironments) -> nx.Graph:
    return maze.env_track.binning.linearization.graph


def _length(graph: nx.Graph, u: str, v: str) -> float:
    return float(np.hypot(*np.subtract(graph.nodes[u]["pos"], graph.nodes[v]["pos"])))


def _module(node: str) -> str:
    return node.removeprefix("node_")[0]


@pytest.fixture(scope="module")
def maze() -> MazeEnvironments:
    return make_rat_hexmaze(bin_size=BIN_SIZE)


@pytest.fixture(scope="module")
def graph(maze) -> nx.Graph:
    return _graph(maze)


@pytest.fixture(scope="module")
def mouse() -> MazeEnvironments:
    """The published mouse HexMaze: one module, 36.3 cm node spacing."""
    dims = RatHexmazeDims(n_modules=1, node_spacing=36.3, corridor_width=10.0)
    return make_rat_hexmaze(dims, bin_size=2.0)


class TestTopology:
    def test_has_96_choice_points_and_125_corridors(self, graph):
        assert graph.number_of_nodes() == 96
        assert graph.number_of_edges() == 125
        assert nx.is_connected(graph)

    def test_has_58_three_way_and_38_two_way_nodes(self, graph):
        assert Counter(d for _, d in graph.degree()) == {3: 58, 2: 38}

    def test_each_module_is_a_mouse_hexmaze(self, graph):
        for letter in "ABCD":
            module = graph.subgraph(n for n in graph if _module(n) == letter)
            assert module.number_of_nodes() == 24
            assert module.number_of_edges() == 30
            # Six hexagons around a central hexagonal hole: seven 6-cycles.
            cycles = nx.minimum_cycle_basis(module)
            assert sorted(len(c) for c in cycles) == [6] * 7

    def test_three_small_and_two_large_bridges_join_the_modules(self, graph):
        bridges = {
            "".join(sorted({_module(u), _module(v)})): _length(graph, u, v)
            for u, v in graph.edges
            if _module(u) != _module(v)
        }
        s, L = DIMS.node_spacing, DIMS.bridge_length
        assert bridges == pytest.approx(
            {"AB": L, "BC": L, "CD": L, "AC": 3 * s + L, "BD": 3 * s + L}
        )

    def test_module_corridors_are_one_node_spacing_long(self, graph):
        lengths = [
            _length(graph, u, v) for u, v in graph.edges if _module(u) == _module(v)
        ]
        assert len(lengths) == 120
        np.testing.assert_allclose(lengths, DIMS.node_spacing)

    def test_edge_distances_are_the_corridor_lengths(self, graph):
        for u, v, distance in graph.edges(data="distance"):
            assert distance == pytest.approx(_length(graph, u, v))

    def test_corridors_meet_at_120_degrees(self, graph):
        for node in (n for n, d in graph.degree() if d == 3):
            vectors = np.subtract(
                [graph.nodes[n]["pos"] for n in graph.neighbors(node)],
                graph.nodes[node]["pos"],
            )
            angles = np.sort(np.arctan2(vectors[:, 1], vectors[:, 0]))
            gaps = np.diff(np.append(angles, angles[0] + 2 * np.pi))
            np.testing.assert_allclose(gaps, 2 * np.pi / 3, atol=1e-9, err_msg=node)

    def test_corridors_cross_only_at_shared_nodes(self, graph):
        segments = {
            (u, v): LineString([graph.nodes[u]["pos"], graph.nodes[v]["pos"]])
            for u, v in graph.edges
        }
        for (e1, s1), (e2, s2) in combinations(segments.items(), 2):
            if set(e1) & set(e2):
                assert s1.intersection(s2).geom_type == "Point", (e1, e2)
            else:
                assert not s1.intersects(s2), (e1, e2)


class TestGeometry:
    def test_spans_about_9_by_5_m(self, maze, graph):
        s, L, w = DIMS.node_spacing, DIMS.bridge_length, DIMS.corridor_width
        nodes = np.array([graph.nodes[n]["pos"] for n in graph])
        np.testing.assert_allclose(
            np.ptp(nodes, axis=0),
            [17 * s + 1.5 * L, 5 * np.sqrt(3) * s + np.sqrt(3) / 2 * L],
        )
        # Bin centers sit up to half a bin inside the corridor edges.
        extent = np.ptp(maze.env_2d.bin_centers, axis=0) + BIN_SIZE
        np.testing.assert_allclose(extent, np.ptp(nodes, axis=0) + w, atol=BIN_SIZE)
        np.testing.assert_allclose(extent, [900.0, 500.0], rtol=0.05)

    def test_every_corridor_lies_inside_env_2d(self, maze, graph):
        t = np.linspace(0.0, 1.0, 25)[:, np.newaxis]
        for u, v in graph.edges:
            p, q = np.array(graph.nodes[u]["pos"]), np.array(graph.nodes[v]["pos"])
            assert maze.env_2d.contains(p + t * (q - p)).all(), (u, v)

    def test_env_2d_is_one_connected_maze(self, maze):
        assert nx.is_connected(maze.env_2d.connectivity)

    def test_hexagon_interiors_are_not_walkable(self, maze):
        # The center of module A's central hole and of one of its six hexagons.
        s = DIMS.node_spacing
        holes = np.array([[0.0, 0.0], [1.5 * s, np.sqrt(3) / 2 * s]])
        assert not maze.env_2d.contains(holes).any()


class TestRegions:
    def test_modules_and_bridges_are_named_polygons(self, maze):
        regions = maze.env_2d.regions
        names = ["module_A", "module_B", "module_C", "module_D"] + [
            f"bridge_{b}" for b in ("AB", "BC", "CD", "AC", "BD")
        ]
        assert [n for n in regions if not n.startswith("node_")] == names
        assert all(regions[n].kind == "polygon" for n in names)

    def test_modules_and_bridges_do_not_overlap(self, maze):
        polygons = [r.data for r in maze.env_2d.regions.values() if r.kind == "polygon"]
        for a, b in combinations(polygons, 2):
            assert a.intersection(b).area < 1e-6

    def test_every_choice_point_is_a_point_region(self, maze, graph):
        regions = maze.env_2d.regions
        nodes = {n: r for n, r in regions.items() if n.startswith("node_")}
        assert set(nodes) == set(graph.nodes)
        for name, region in nodes.items():
            assert region.kind == "point"
            np.testing.assert_allclose(region.data, graph.nodes[name]["pos"])


class TestMouseHexMaze:
    def test_has_24_nodes_and_30_corridors(self, mouse):
        graph = _graph(mouse)
        assert graph.number_of_nodes() == 24
        assert graph.number_of_edges() == 30
        # "12 two-way and 12 three-way choice points" (Alonso et al. 2021)
        assert Counter(d for _, d in graph.degree()) == {2: 12, 3: 12}
        lengths = [_length(graph, u, v) for u, v in graph.edges]
        np.testing.assert_allclose(lengths, 36.3)

    def test_spans_about_2_by_1_9_m(self, mouse):
        extent = np.ptp(mouse.env_2d.bin_centers, axis=0) + 2.0
        np.testing.assert_allclose(sorted(extent), [190.0, 200.0], rtol=0.05)

    def test_has_one_module_region_and_no_bridges(self, mouse):
        names = [n for n in mouse.env_2d.regions if not n.startswith("node_")]
        assert names == ["module_A"]


class TestDims:
    def test_defaults(self):
        assert (
            RatHexmazeDims(
                node_spacing=48.0, corridor_width=10.0, bridge_length=62.4, n_modules=4
            )
            == DIMS
        )

    @pytest.mark.parametrize(
        ("n_modules", "n_nodes", "n_edges"), [(2, 48, 61), (3, 72, 93)]
    )
    def test_fewer_modules_keep_the_bridges_between_them(
        self, n_modules, n_nodes, n_edges
    ):
        maze = make_rat_hexmaze(RatHexmazeDims(n_modules=n_modules), bin_size=8.0)
        graph = _graph(maze)
        assert (graph.number_of_nodes(), graph.number_of_edges()) == (n_nodes, n_edges)
        assert nx.is_connected(graph)
        assert nx.is_connected(maze.env_2d.connectivity)

    @pytest.mark.parametrize("n_modules", [0, 5])
    def test_rejects_unsupported_module_counts(self, n_modules):
        with pytest.raises(ValueError, match="n_modules must be 1, 2, 3 or 4"):
            make_rat_hexmaze(RatHexmazeDims(n_modules=n_modules))

    def test_corridor_width_sets_the_track_width(self):
        maze = make_rat_hexmaze(RatHexmazeDims(corridor_width=14.0), bin_size=8.0)
        assert maze.env_track.arena.width == pytest.approx(14.0)


class TestEnvironments:
    def test_both_environments_are_in_cm(self, maze):
        assert maze.env_2d.units == "cm"
        assert maze.env_track.units == "cm"
        assert maze.env_track.is_linearized_track

    def test_include_track_false_has_no_track(self):
        assert make_rat_hexmaze(bin_size=8.0, include_track=False).env_track is None

    def test_finer_bins_give_more_bins(self, maze):
        coarse = make_rat_hexmaze(bin_size=2 * BIN_SIZE, include_track=False)
        assert maze.env_2d.n_bins > coarse.env_2d.n_bins

    def test_none_dims_uses_defaults(self, maze):
        explicit = make_rat_hexmaze(dims=DIMS, bin_size=BIN_SIZE, include_track=False)
        assert explicit.env_2d.n_bins == maze.env_2d.n_bins
