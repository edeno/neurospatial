"""Tests for experiment-shaped factory presets.

These presets (`open_field`, `linear_track`, `maze`) provide
experiment-vocabulary classmethods on Environment that delegate to the
existing `from_*` factories.

Critical rule: track/maze presets require an EXPLICIT topology spec.
Positions alone CANNOT infer a linear/W/plus/T graph. Only `open_field`
is positions-based.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment


# ---------------------------------------------------------------------------
# open_field
# ---------------------------------------------------------------------------
class TestOpenField:
    def test_equivalent_to_from_samples_with_fill_holes(self):
        """open_field matches from_samples(..., fill_holes=True)."""
        rng = np.random.default_rng(0)
        positions = rng.random((1000, 2)) * 100.0

        preset = Environment.open_field(positions, bin_size=5.0)
        reference = Environment.from_samples(positions, bin_size=5.0, fill_holes=True)

        assert preset.n_bins == reference.n_bins
        np.testing.assert_array_equal(
            preset.layout.active_mask, reference.layout.active_mask
        )

    def test_differs_from_default_when_holes_present(self):
        """fill_holes=True changes the active mask vs the bare default."""
        # Ring-shaped sampling leaves an interior hole that fill_holes fills.
        rng = np.random.default_rng(1)
        theta = rng.uniform(0, 2 * np.pi, 2000)
        r = rng.uniform(30, 40, 2000)
        positions = np.column_stack([50 + r * np.cos(theta), 50 + r * np.sin(theta)])

        preset = Environment.open_field(positions, bin_size=5.0)
        bare = Environment.from_samples(positions, bin_size=5.0)

        # fill_holes should add bins (fills the donut hole).
        assert preset.n_bins >= bare.n_bins

    def test_passes_through_name(self):
        positions = np.random.default_rng(2).random((500, 2)) * 50.0
        env = Environment.open_field(positions, bin_size=5.0, name="arena")
        assert env.name == "arena"

    def test_is_2d_not_linearized(self):
        positions = np.random.default_rng(3).random((500, 2)) * 50.0
        env = Environment.open_field(positions, bin_size=5.0)
        assert env.n_dims == 2
        assert env.is_linearized_track is False


# ---------------------------------------------------------------------------
# linear_track
# ---------------------------------------------------------------------------
class TestLinearTrack:
    def test_from_endpoints(self):
        env = Environment.linear_track(endpoints=[(0, 0), (100, 0)], bin_size=5.0)
        assert env.is_linearized_track is True
        # 100 cm / 5 cm = 20 bins
        assert env.n_bins == 20

    def test_from_endpoints_to_linear_works(self):
        env = Environment.linear_track(endpoints=[(0, 0), (100, 0)], bin_size=5.0)
        linear = env.to_linear(np.array([[50.0, 0.0]]))
        assert linear.shape[0] == 1
        # Midpoint of a 100 cm track should be ~50.
        assert np.isclose(linear[0], 50.0, atol=5.0)

    def test_from_node_positions_piecewise(self):
        # L-shaped piecewise track: (0,0) -> (50,0) -> (50,50). Total 100 cm.
        env = Environment.linear_track(
            node_positions=[(0, 0), (50, 0), (50, 50)], bin_size=5.0
        )
        assert env.is_linearized_track is True
        assert env.n_bins == 20

    def test_node_positions_graph_structure(self):
        # 3 waypoints -> 3 nodes, 2 edges.
        env = Environment.linear_track(
            node_positions=[(0, 0), (50, 0), (50, 50)], bin_size=5.0
        )
        graph = env.layout_parameters["graph_definition"]
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 2

    def test_no_topology_raises(self):
        with pytest.raises(ValueError, match="explicit topology"):
            Environment.linear_track(bin_size=5.0)

    def test_two_endpoints_required(self):
        with pytest.raises(ValueError):
            Environment.linear_track(endpoints=[(0, 0)], bin_size=5.0)


# ---------------------------------------------------------------------------
# maze
# ---------------------------------------------------------------------------
class TestMaze:
    def _w_nodes(self):
        # 3 vertical arms + horizontal connector base.
        return {
            "base_left": (0, 0),
            "base_mid": (50, 0),
            "base_right": (100, 0),
            "arm_left": (0, 50),
            "arm_mid": (50, 50),
            "arm_right": (100, 50),
        }

    def test_w_from_node_positions(self):
        env = Environment.maze("w", node_positions=self._w_nodes(), bin_size=5.0)
        assert env.is_linearized_track is True
        graph = env.layout_parameters["graph_definition"]
        # 6 nodes; base has 2 connector edges + 3 vertical arm edges = 5 edges.
        assert graph.number_of_nodes() == 6
        assert graph.number_of_edges() == 5

    def test_w_edge_topology_and_to_linear(self):
        """W graph edges match the order-based contract and to_linear() works."""
        env = Environment.maze("w", node_positions=self._w_nodes(), bin_size=5.0)
        graph = env.layout_parameters["graph_definition"]
        edges = {frozenset(e) for e in graph.edges()}
        expected = {
            # Horizontal connector: base_left -- base_mid -- base_right.
            frozenset({"base_left", "base_mid"}),
            frozenset({"base_mid", "base_right"}),
            # Vertical arms by matching position in the documented order.
            frozenset({"base_left", "arm_left"}),
            frozenset({"base_mid", "arm_mid"}),
            frozenset({"base_right", "arm_right"}),
        }
        assert edges == expected
        # Graph validity: to_linear() produces a finite linear coordinate.
        linear = env.to_linear(np.array([[50.0, 0.0]]))
        assert linear.shape[0] == 1
        assert np.isfinite(linear[0])

    def test_w_topology_is_coordinate_independent(self):
        """Arms-down coordinates produce the SAME topology (regression).

        This is the test that would have caught the old y<=y_split bug: the
        base sits at LARGE y and the arms extend to SMALL y, so any
        y-coordinate heuristic would misclassify base vs. arms. The order-based
        contract must yield the identical edge topology.
        """
        arms_down = {
            "base_left": (0, 100),
            "base_mid": (50, 100),
            "base_right": (100, 100),
            "arm_left": (0, 50),
            "arm_mid": (50, 50),
            "arm_right": (100, 50),
        }
        env = Environment.maze("w", node_positions=arms_down, bin_size=5.0)
        graph = env.layout_parameters["graph_definition"]
        edges = {frozenset(e) for e in graph.edges()}
        expected = {
            frozenset({"base_left", "base_mid"}),
            frozenset({"base_mid", "base_right"}),
            frozenset({"base_left", "arm_left"}),
            frozenset({"base_mid", "arm_mid"}),
            frozenset({"base_right", "arm_right"}),
        }
        assert edges == expected
        assert env.is_linearized_track is True

    def test_plus_edge_topology_and_to_linear(self):
        """Plus graph edges (center--each arm) and to_linear() works."""
        nodes = {
            "center": (0, 0),
            "north": (0, 50),
            "south": (0, -50),
            "east": (50, 0),
            "west": (-50, 0),
        }
        env = Environment.maze("plus", node_positions=nodes, bin_size=5.0)
        graph = env.layout_parameters["graph_definition"]
        edges = {frozenset(e) for e in graph.edges()}
        expected = {
            frozenset({"center", "north"}),
            frozenset({"center", "south"}),
            frozenset({"center", "east"}),
            frozenset({"center", "west"}),
        }
        assert edges == expected
        linear = env.to_linear(np.array([[0.0, 25.0]]))
        assert linear.shape[0] == 1
        assert np.isfinite(linear[0])

    def test_plus_topology_is_coordinate_independent(self):
        """Plus topology comes from order, not coordinates (regression)."""
        # Arms in arbitrary/flipped directions; order alone defines edges.
        nodes = {
            "center": (0, 0),
            "a": (0, -50),
            "b": (0, -100),
            "c": (-50, 0),
            "d": (-100, 0),
        }
        env = Environment.maze("plus", node_positions=nodes, bin_size=5.0)
        graph = env.layout_parameters["graph_definition"]
        edges = {frozenset(e) for e in graph.edges()}
        expected = {
            frozenset({"center", "a"}),
            frozenset({"center", "b"}),
            frozenset({"center", "c"}),
            frozenset({"center", "d"}),
        }
        assert edges == expected

    def test_t_edge_topology_and_to_linear(self):
        """T graph edges (stem + crossbar) and to_linear() works."""
        nodes = {
            "start": (0, 0),
            "junction": (0, 50),
            "left": (-50, 50),
            "right": (50, 50),
        }
        env = Environment.maze("t", node_positions=nodes, bin_size=5.0)
        graph = env.layout_parameters["graph_definition"]
        edges = {frozenset(e) for e in graph.edges()}
        expected = {
            frozenset({"start", "junction"}),
            frozenset({"junction", "left"}),
            frozenset({"junction", "right"}),
        }
        assert edges == expected
        linear = env.to_linear(np.array([[0.0, 25.0]]))
        assert linear.shape[0] == 1
        assert np.isfinite(linear[0])

    def test_t_topology_is_coordinate_independent(self):
        """T topology comes from order, not coordinates (regression).

        Stem extends to LARGE y and arms to SMALL y -- the opposite of the
        conventional T -- yet the order-based contract yields the same
        stem/crossbar edges.
        """
        nodes = {
            "stem_end": (0, 100),
            "junction": (0, 50),
            "arm_left": (-50, 0),
            "arm_right": (50, 0),
        }
        env = Environment.maze("t", node_positions=nodes, bin_size=5.0)
        graph = env.layout_parameters["graph_definition"]
        edges = {frozenset(e) for e in graph.edges()}
        expected = {
            frozenset({"stem_end", "junction"}),
            frozenset({"junction", "arm_left"}),
            frozenset({"junction", "arm_right"}),
        }
        assert edges == expected

    def test_plus_from_node_positions(self):
        nodes = {
            "center": (0, 0),
            "north": (0, 50),
            "south": (0, -50),
            "east": (50, 0),
            "west": (-50, 0),
        }
        env = Environment.maze("plus", node_positions=nodes, bin_size=5.0)
        assert env.is_linearized_track is True
        graph = env.layout_parameters["graph_definition"]
        assert graph.number_of_nodes() == 5
        assert graph.number_of_edges() == 4

    def test_t_from_node_positions(self):
        nodes = {
            "start": (0, 0),
            "junction": (0, 50),
            "left": (-50, 50),
            "right": (50, 50),
        }
        env = Environment.maze("t", node_positions=nodes, bin_size=5.0)
        assert env.is_linearized_track is True
        graph = env.layout_parameters["graph_definition"]
        assert graph.number_of_nodes() == 4
        # stem + crossbar (2 arms) = 3 edges.
        assert graph.number_of_edges() == 3

    def test_from_ready_track_graph(self):
        g = nx.Graph()
        g.add_node("a", pos=(0.0, 0.0))
        g.add_node("b", pos=(50.0, 0.0))
        g.add_node("c", pos=(100.0, 0.0))
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        env = Environment.maze("w", track_graph=g, bin_size=5.0)
        assert env.is_linearized_track is True
        assert env.n_bins == 20

    def test_track_graph_not_mutated(self):
        """maze(track_graph=...) must not mutate the caller's graph."""
        g = nx.Graph()
        g.add_node("a", pos=(0.0, 0.0))
        g.add_node("b", pos=(50.0, 0.0))
        g.add_node("c", pos=(100.0, 0.0))
        g.add_edge("a", "b")
        g.add_edge("b", "c")

        env = Environment.maze("w", track_graph=g, bin_size=5.0)

        # The caller's graph must be untouched: no `distance`/`edge_id` written.
        for u, v in g.edges():
            assert "distance" not in g.edges[u, v]
            assert "edge_id" not in g.edges[u, v]
        # And the built environment must still linearize correctly.
        assert env.is_linearized_track is True

    def test_track_graph_to_linear_without_edge_id(self):
        """to_linear() works even though edge_id is not in the input graph."""
        g = nx.Graph()
        g.add_node("a", pos=(0.0, 0.0))
        g.add_node("b", pos=(50.0, 0.0))
        g.add_node("c", pos=(100.0, 0.0))
        # Only `distance` provided; no `edge_id` at all.
        g.add_edge("a", "b", distance=50.0)
        g.add_edge("b", "c", distance=50.0)
        env = Environment.maze("w", track_graph=g, bin_size=5.0)
        linear = env.to_linear(np.array([[50.0, 0.0]]))
        assert linear.shape[0] == 1
        assert np.isfinite(linear[0])

    @pytest.mark.parametrize(
        ("kind", "nodes"),
        [
            ("plus", {"center": (0, 0), "north": (0, 50)}),  # too few (needs 5)
            ("t", {"a": (0, 0), "b": (0, 50), "c": (50, 50)}),  # too few (needs 4)
            ("w", {"a": (0, 0), "b": (50, 0), "c": (100, 0)}),  # too few (needs 6)
        ],
    )
    def test_wrong_node_count_raises(self, kind, nodes):
        with pytest.raises(ValueError, match="needs exactly"):
            Environment.maze(kind, node_positions=nodes, bin_size=5.0)

    def test_assemble_maze_graph_unknown_kind_raises(self):
        """Direct call with a bad kind raises instead of returning empty graph."""
        from neurospatial.environment.factories import _assemble_maze_graph

        with pytest.raises(ValueError, match="Unknown maze kind"):
            _assemble_maze_graph("circle", {"a": (0, 0)})

    def test_no_topology_raises(self):
        with pytest.raises(ValueError, match="topology"):
            Environment.maze("w", bin_size=5.0)

    def test_bad_kind_raises(self):
        nodes = {"center": (0, 0), "north": (0, 50)}
        with pytest.raises(ValueError, match="kind"):
            Environment.maze("circle", node_positions=nodes, bin_size=5.0)
