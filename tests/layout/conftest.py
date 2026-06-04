"""Shared fixtures for layout-engine tests."""

import networkx as nx
import numpy as np
import pytest


@pytest.fixture
def synthetic_16x16_mask():
    """Deterministic 16x16 boolean mask: a centered disk of radius 6.

    Returns
    -------
    NDArray[np.bool_], shape (16, 16)
        True where the pixel falls inside the disk.
    """
    n = 16
    yy, xx = np.mgrid[0:n, 0:n]
    center = (n - 1) / 2.0
    return (xx - center) ** 2 + (yy - center) ** 2 <= 6.0**2


@pytest.fixture
def l_shaped_polygon():
    """Shapely L-shaped polygon in the unit square scaled to 10 units.

    The L occupies the lower and left strips of a 10x10 square, leaving the
    upper-right quadrant empty.

    Returns
    -------
    shapely.geometry.Polygon
    """
    from shapely.geometry import Polygon

    # L-shape: full bottom strip + left strip, missing the upper-right block.
    exterior = [
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 4.0),
        (4.0, 4.0),
        (4.0, 10.0),
        (0.0, 10.0),
    ]
    return Polygon(exterior)


@pytest.fixture
def polygon_with_hole():
    """Shapely 10x10 square polygon with a 2x2 square hole at its center.

    Returns
    -------
    shapely.geometry.Polygon
    """
    from shapely.geometry import Polygon

    exterior = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    hole = [(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)]
    return Polygon(exterior, [hole])


@pytest.fixture
def simple_y_maze_graph():
    """Networkx Y-maze graph with pos, distance, and edge_id attributes.

    Stem node 0 -> junction node 1, then two arms to nodes 2 and 3. Edge
    distances are the Euclidean lengths between node positions.

    Returns
    -------
    tuple[nx.Graph, list[tuple[int, int]]]
        The graph and a valid edge ordering for linearization.
    """
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(0.0, 5.0))
    graph.add_node(2, pos=(-5.0, 10.0))
    graph.add_node(3, pos=(5.0, 10.0))

    arm_length = float(np.hypot(5.0, 5.0))
    graph.add_edge(0, 1, distance=5.0, edge_id=0)
    graph.add_edge(1, 2, distance=arm_length, edge_id=1)
    graph.add_edge(1, 3, distance=arm_length, edge_id=2)

    edge_order = [(0, 1), (1, 2), (1, 3)]
    return graph, edge_order


@pytest.fixture
def graph_layout_with_gap(simple_y_maze_graph):
    """A built 1D ``GraphLayout`` with non-zero ``edge_spacing`` (gaps).

    Linearizing the Y-maze with ``edge_spacing > 0`` inserts inactive gap
    bins between consecutive edges, so the full (gap-inclusive) bin list is
    longer than the active-bin list. This is the configuration that
    distinguishes active-bin indices from full-grid indices.

    Returns
    -------
    GraphLayout
        A built layout whose ``active_mask`` contains at least one ``False``
        (gap) entry.
    """
    from neurospatial.layout.engines.graph import GraphLayout

    graph, edge_order = simple_y_maze_graph
    layout = GraphLayout()
    layout.build(
        graph_definition=graph,
        edge_order=edge_order,
        edge_spacing=10.0,
        bin_size=2.0,
    )
    return layout
