"""Linear track positions name one place.

Consecutive edges of ``edge_order`` that do not meet (an arm ending at a well,
the next edge starting at a junction) must be separated by a gap; otherwise
the end of one and the start of the next share a linear position and
``linear_to_nd`` cannot return both.
"""

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment


@pytest.fixture
def t_maze_graph():
    graph = nx.Graph()
    graph.add_node("start", pos=(0.0, 0.0))
    graph.add_node("junction", pos=(0.0, 40.0))
    graph.add_node("left", pos=(-30.0, 40.0))
    graph.add_node("right", pos=(30.0, 40.0))
    graph.add_edge("start", "junction")
    graph.add_edge("junction", "left")
    graph.add_edge("junction", "right")
    return graph


T_ORDER = [("start", "junction"), ("junction", "left"), ("junction", "right")]


def test_every_node_round_trips_with_default_spacing(t_maze_graph):
    env = Environment.from_graph(t_maze_graph, T_ORDER, bin_size=2.0)
    for node, attrs in t_maze_graph.nodes(data=True):
        position = np.asarray(attrs["pos"], dtype=float)[np.newaxis]
        back = env.linear_to_nd(env.to_linear(position))
        np.testing.assert_allclose(back[0], position[0], atol=1e-9, err_msg=str(node))


def test_default_gap_only_where_edges_do_not_meet(t_maze_graph):
    env = Environment.from_graph(t_maze_graph, T_ORDER, bin_size=2.0)
    # start->junction continues into junction->left: no gap; left arm ends at
    # the left well and the right arm starts at the junction: a fixed gap.
    assert env._binning.linearization.edge_spacing == (0.0, 15.0)


def test_linear_positions_do_not_depend_on_bin_size(t_maze_graph):
    right_tip = np.array([[30.0, 40.0]])
    linear = [
        Environment.from_graph(t_maze_graph, T_ORDER, bin_size=b).to_linear(right_tip)
        for b in (1.0, 2.0, 4.0)
    ]
    np.testing.assert_allclose(linear, [[40.0 + 30.0 + 15.0 + 30.0]] * 3)


def test_polyline_gets_no_gaps():
    env = Environment.linear_track(
        node_positions=[(0.0, 0.0), (60.0, 0.0), (60.0, 40.0)], bin_size=2.0
    )
    assert env._binning.linearization.edge_spacing == (0.0,)
    np.testing.assert_allclose(env.to_linear(np.array([[60.0, 40.0]])), [100.0])


@pytest.mark.parametrize("edge_spacing", [0.0, [0.0, 0.0], [5.0, 0.0]])
def test_zero_gap_between_edges_that_do_not_meet_is_refused(t_maze_graph, edge_spacing):
    with pytest.raises(ValueError, match=r"'left'.*'junction'") as raised:
        Environment.from_graph(
            t_maze_graph, T_ORDER, bin_size=2.0, edge_spacing=edge_spacing
        )
    assert "edge_spacing" in str(raised.value)


def test_zero_gap_where_edges_meet_is_allowed(t_maze_graph):
    env = Environment.from_graph(
        t_maze_graph, T_ORDER, bin_size=2.0, edge_spacing=[0.0, 10.0]
    )
    assert env._binning.linearization.edge_spacing == (0.0, 10.0)
