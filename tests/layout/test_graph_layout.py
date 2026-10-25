"""Behavioral tests for GraphLayout driven directly through the engine."""

import itertools

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.layout.engines.graph import GraphLayout


def _build_graph_layout(simple_y_maze_graph, bin_size=1.0, edge_spacing=0.0):
    graph, edge_order = simple_y_maze_graph
    layout = GraphLayout()
    layout.build(
        graph_definition=graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
        bin_size=bin_size,
    )
    return layout


def test_1d_track_linearization(simple_y_maze_graph):
    """Linearization is monotonic and n_bins matches the connectivity node count.

    The engine exposes no ``n_bins`` attribute; the active count is read from
    ``bin_centers.shape[0]``, which must equal the number of graph nodes.
    """
    layout = _build_graph_layout(simple_y_maze_graph)

    assert layout.bin_centers.shape[0] == layout.connectivity.number_of_nodes()

    # Linearized bin centers increase monotonically along the unrolled track.
    assert layout.linear_bin_centers_ is not None
    assert np.all(np.diff(layout.linear_bin_centers_) > 0)


def test_point_to_bin_on_node(simple_y_maze_graph):
    """A point at a node position maps to a valid bin near that node."""
    layout = _build_graph_layout(simple_y_maze_graph)

    # Node 0 sits at (0, 0); its bin is the first along the linearized track.
    result = layout.binning().locator.locate(np.array([[0.0, 0.0]]))
    assert result[0] == 0


def test_point_off_graph_returns_negative_one(simple_y_maze_graph):
    """A linearized position outside the track range maps to -1.

    Note: the track locator is KDTree-based and snaps every query to its
    nearest active bin, so it never returns -1 for an off-graph point. The -1
    sentinel for off-track positions is produced by the linearized lookup
    ``linear_point_to_bin_ind``, which is asserted here. Both behaviors are
    pinned.
    """
    layout = _build_graph_layout(simple_y_maze_graph)

    # A linear coordinate well beyond the track length is outside all bins.
    track_length = float(layout.grid_edges[0][-1])
    off_track = np.array([track_length + 100.0])
    assert layout.linear_point_to_bin_ind(off_track)[0] == -1

    # Pin the KDTree-snap behavior: a far 2D point still binds to some bin.
    far = layout.binning().locator.locate(np.array([[1000.0, 1000.0]]))
    assert far[0] >= 0


def test_point_to_bin_maps_nonfinite_rows_to_sentinel(simple_y_maze_graph):
    """A NaN/inf query row maps to -1 without collapsing the finite rows.

    Regression: scipy's ``KDTree.query`` raises if ANY row is non-finite, and
    the previous broad ``except`` swallowed that to a warning and returned -1
    for EVERY point -- so a single dropped-frame NaN in a trajectory silently
    emptied the whole occupancy / rate map on graph/linearized environments.
    Only the non-finite rows should become the -1 sentinel; the finite rows
    must still map to their nearest bin (matching the grid layouts).
    """
    layout = _build_graph_layout(simple_y_maze_graph)

    query = np.array(
        [
            [0.0, 0.0],  # node 0 -> bin 0
            [np.nan, np.nan],  # dropped frame -> -1
            [1000.0, 1000.0],  # far but finite -> snaps to some valid bin
            [np.inf, 0.0],  # partially non-finite -> -1
        ]
    )
    result = layout.binning().locator.locate(query)

    assert result[0] == 0
    assert result[1] == -1
    assert result[2] >= 0  # finite rows are NOT collapsed to -1
    assert result[3] == -1


def test_is_linearized_track_true(simple_y_maze_graph):
    """A graph layout reports itself as a linearized 1D track."""
    layout = _build_graph_layout(simple_y_maze_graph)
    assert layout.is_linearized_track is True


def test_graph_linear_point_to_bin_active_indices_with_gap(graph_layout_with_gap):
    """Linear lookups after a gap return active-bin indices, not full-grid ones.

    With ``edge_spacing > 0`` the full (gap-inclusive) bin list is longer than
    the active-bin list, so a position in a segment after a gap has a full-grid
    index that exceeds its active-bin index. The public method must return the
    active-bin index (consistent with ``bin_centers``), and -1 for positions in
    a gap or out of range.
    """
    layout = graph_layout_with_gap
    n_active = layout.bin_centers.shape[0]
    assert not layout.active_mask.all()  # the fixture really has a gap

    # Find the first active bin that follows an inactive (gap) bin and query a
    # linear position at its center.
    mask = layout.active_mask
    full_to_active = np.full(mask.size, -1, dtype=int)
    full_to_active[mask] = np.arange(int(mask.sum()))
    post_gap_full = next(i for i in range(1, mask.size) if mask[i] and not mask[i - 1])
    expected_active = full_to_active[post_gap_full]
    assert expected_active != post_gap_full  # active idx differs from full idx

    pos = float(layout.linear_bin_centers_[expected_active])
    result = layout.linear_point_to_bin_ind(np.array([pos]))
    assert 0 <= result[0] < n_active
    assert result[0] == expected_active
    np.testing.assert_allclose(
        layout.bin_centers[result[0]], layout.bin_centers[expected_active]
    )

    # A position inside the gap and one out of range both return -1.
    gap_full = next(i for i in range(mask.size) if not mask[i])
    gap_pos = 0.5 * (
        layout.grid_edges[0][gap_full] + layout.grid_edges[0][gap_full + 1]
    )
    assert layout.linear_point_to_bin_ind(np.array([gap_pos]))[0] == -1

    track_length = float(layout.grid_edges[0][-1])
    assert layout.linear_point_to_bin_ind(np.array([track_length + 100.0]))[0] == -1


@pytest.mark.parametrize("spacing", [np.float32(1.5), np.float64(1.5), np.int64(2)])
def test_numpy_scalar_edge_spacing_matches_python_number(simple_y_maze_graph, spacing):
    """A NumPy scalar gap builds the same track as the equal Python number."""
    graph, edge_order = simple_y_maze_graph
    expected = Environment.from_graph(
        graph, edge_order, bin_size=1.0, edge_spacing=float(spacing)
    )
    env = Environment.from_graph(graph, edge_order, bin_size=1.0, edge_spacing=spacing)

    np.testing.assert_array_equal(env.bin_centers, expected.bin_centers)
    np.testing.assert_allclose(
        env.to_linear(env.bin_centers), expected.to_linear(expected.bin_centers)
    )
    assert env.layout_parameters == expected.layout_parameters


def _star_graph(arm_ends):
    """Track graph with node 0 at the origin joined to one node per arm end."""
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    for node, end in enumerate(arm_ends, start=1):
        graph.add_node(node, pos=end)
        graph.add_edge(0, node)
    return graph


_STAR_MAZES = {
    "Y": [(0.0, -10.0), (-7.0, 7.0), (7.0, 7.0)],
    "T": [(0.0, -10.0), (-10.0, 0.0), (10.0, 0.0)],
    "plus": [(0.0, 10.0), (0.0, -10.0), (10.0, 0.0), (-10.0, 0.0)],
}


def _junction_bins(layout):
    """Bin nearest the junction (origin) on each arm, keyed by arm node."""
    segments = layout._segment_ids
    edges = list(layout.linearization().graph.edges())
    nearest = {}
    for segment_id in np.unique(segments):
        bins = np.flatnonzero(segments == segment_id)
        arm_node = next(node for node in edges[segment_id] if node != 0)
        distance = np.linalg.norm(layout.bin_centers[bins], axis=1)
        nearest[arm_node] = int(bins[np.argmin(distance)])
    return nearest


@pytest.mark.parametrize("maze", sorted(_STAR_MAZES))
def test_every_segment_end_at_a_junction_is_a_neighbor(maze):
    """All arm ends meeting at a junction are pairwise neighbors.

    A junction is a single point of the track: an animal at the end of one
    arm steps straight into any other arm. Chaining the arm ends in
    ``edge_order`` order instead would route one arm to another through a
    third arm, so maze distances would depend on the edge order.
    """
    arms = _STAR_MAZES[maze]
    graph = _star_graph(arms)
    arm_nodes = list(range(1, len(arms) + 1))
    reference = None
    for order in itertools.permutations(arm_nodes):
        layout = GraphLayout()
        layout.build(
            graph_definition=graph,
            edge_order=[(0, node) for node in order],
            edge_spacing=0.0,
            bin_size=2.5,
        )
        ends = _junction_bins(layout)
        connectivity = layout.connectivity
        for a, b in itertools.combinations(arm_nodes, 2):
            assert connectivity.has_edge(ends[a], ends[b]), (maze, order, a, b)
        distances = {
            (a, b): nx.shortest_path_length(
                connectivity, ends[a], ends[b], weight="distance"
            )
            for a, b in itertools.combinations(arm_nodes, 2)
        }
        if reference is None:
            reference = distances
        assert distances == pytest.approx(reference), (maze, order)


@pytest.mark.parametrize("maze", sorted(_STAR_MAZES))
def test_cross_junction_edges_are_measured_along_the_track(maze):
    """An edge between two arms is the path through the junction node.

    The track is the union of its segments, so an animal moving between the
    arm ends at a junction travels through the node: center to node, then
    node to center. The same length is the flux distance of the track faces,
    so the connectivity graph and smoothing agree.
    """
    graph = _star_graph(_STAR_MAZES[maze])
    layout = GraphLayout()
    layout.build(
        graph_definition=graph,
        edge_order=[(0, node) for node in graph.neighbors(0)],
        edge_spacing=0.0,
        bin_size=2.5,
    )
    ends = list(_junction_bins(layout).values())
    centers = layout.bin_centers
    faces = layout.binning().build_faces()
    face_distance = dict(
        zip(map(tuple, faces.edges.tolist()), faces.distance, strict=True)
    )
    for a, b in itertools.combinations(ends, 2):
        expected = np.linalg.norm(centers[a]) + np.linalg.norm(centers[b])
        assert layout.connectivity.edges[a, b]["distance"] == pytest.approx(expected)
        assert face_distance[min(a, b), max(a, b)] == pytest.approx(expected)


def test_linear_lookup_agrees_with_bin_at_at_the_track_ends(simple_y_maze_graph):
    """The track's start and end (e.g. reward wells) get their end bins.

    Node 0 starts the first segment and node 3 ends the last one; an inner
    segment end shares its linear position with the next segment's start.
    """
    graph, edge_order = simple_y_maze_graph
    env = Environment.from_graph(graph, edge_order, bin_size=1.0, track_width=1.0)
    layout = _build_graph_layout(simple_y_maze_graph)
    ends = np.array([graph.nodes[node]["pos"] for node in (0, 3)], dtype=float)
    linear = layout.to_linear(ends)
    np.testing.assert_array_equal(
        layout.linear_point_to_bin_ind(linear), env.bin_at(ends)
    )


def test_junction_faces_conserve_flux_of_a_balanced_linear_field():
    """At a junction, faces act like each arm meeting the node, not each other.

    On a Y, a field rising linearly from the node along each arm, with
    slopes summing to zero, is continuous at the node and balanced there:
    every bin's net diffusive flux must be zero. Giving each pair of arms a
    unit face, as for two arms, makes the junction 50% too conductive with
    three equal arms.
    """
    graph = _star_graph(_STAR_MAZES["Y"])
    layout = GraphLayout()
    layout.build(
        graph_definition=graph,
        edge_order=[(0, node) for node in (1, 2, 3)],
        edge_spacing=0.0,
        bin_size=2.0,
    )
    segments = layout._segment_ids
    edges = list(layout.linearization().graph.edges())
    slope = {1: 1.0, 2: -0.25, 3: -0.75}
    distance_to_node = np.linalg.norm(layout.bin_centers, axis=1)
    arm = np.array([next(n for n in edges[s] if n != 0) for s in segments])
    field = np.array([slope[a] for a in arm]) * distance_to_node

    faces = layout.binning().build_faces()
    conductance = faces.A / faces.distance
    low, high = faces.edges.T
    flux = conductance * (field[high] - field[low])
    net = np.zeros(len(field))
    np.add.at(net, low, flux)
    np.add.at(net, high, -flux)
    interior = np.flatnonzero(np.bincount(faces.edges.ravel()) > 1)
    np.testing.assert_allclose(net[interior], 0.0, atol=1e-12)
