"""Behavioral tests for GraphLayout driven directly through the engine."""

import numpy as np

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
    result = layout.point_to_bin_index(np.array([[0.0, 0.0]]))
    assert result[0] == 0


def test_point_off_graph_returns_negative_one(simple_y_maze_graph):
    """A linearized position outside the track range maps to -1.

    Note: ``point_to_bin_index`` is KDTree-based and snaps every query to its
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
    far = layout.point_to_bin_index(np.array([[1000.0, 1000.0]]))
    assert far[0] >= 0


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
