"""Tests for generic graph connectivity building helper.

This module tests the _create_connectivity_graph_generic() function which
provides a common implementation for creating connectivity graphs across
different layout types.
"""

import networkx as nx
import numpy as np

from neurospatial.layout.helpers.graph_building import (
    _create_connectivity_graph_generic,
)


def get_2d_orthogonal_neighbors(
    flat_index: int, grid_shape: tuple[int, ...]
) -> list[int]:
    """Helper: Get orthogonal neighbors in a 2D grid.

    Parameters
    ----------
    flat_index : int
        Flat index of the current bin
    grid_shape : tuple[int, ...]
        Shape of the grid (n_rows, n_cols)

    Returns
    -------
    list[int]
        List of flat indices of orthogonal neighbors
    """
    n_rows, n_cols = grid_shape
    row = flat_index // n_cols
    col = flat_index % n_cols

    neighbors = []
    # Up, down, left, right
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < n_rows and 0 <= new_col < n_cols:
            neighbors.append(new_row * n_cols + new_col)

    return neighbors


def get_2d_diagonal_neighbors(
    flat_index: int, grid_shape: tuple[int, ...]
) -> list[int]:
    """Helper: Get all 8 neighbors (orthogonal + diagonal) in a 2D grid.

    Parameters
    ----------
    flat_index : int
        Flat index of the current bin
    grid_shape : tuple[int, ...]
        Shape of the grid (n_rows, n_cols)

    Returns
    -------
    list[int]
        List of flat indices of all neighbors
    """
    n_rows, n_cols = grid_shape
    row = flat_index // n_cols
    col = flat_index % n_cols

    neighbors = []
    # All 8 directions
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < n_rows and 0 <= new_col < n_cols:
                neighbors.append(new_row * n_cols + new_col)

    return neighbors


def test_create_connectivity_graph_generic_empty_bins():
    """Test that empty active bins returns an empty graph."""
    active_indices = np.array([], dtype=int)
    bin_centers = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
    grid_shape = (2, 2)

    graph = _create_connectivity_graph_generic(
        active_indices, bin_centers, grid_shape, get_2d_orthogonal_neighbors
    )

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


def test_create_connectivity_graph_generic_single_bin():
    """Test graph with a single active bin (no edges)."""
    active_indices = np.array([0])
    bin_centers = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
    grid_shape = (2, 2)

    graph = _create_connectivity_graph_generic(
        active_indices, bin_centers, grid_shape, get_2d_orthogonal_neighbors
    )

    assert graph.number_of_nodes() == 1
    assert graph.number_of_edges() == 0

    # Check node attributes
    node_data = graph.nodes[0]
    assert "pos" in node_data
    assert "source_grid_flat_index" in node_data
    assert "original_grid_nd_index" in node_data
    assert node_data["pos"] == (0.5, 0.5)
    assert node_data["source_grid_flat_index"] == 0
    assert node_data["original_grid_nd_index"] == (0, 0)


def test_create_connectivity_graph_generic_2x2_all_active_orthogonal():
    """Test 2x2 grid with all bins active, orthogonal connections."""
    active_indices = np.array([0, 1, 2, 3])
    # Create 2x2 grid centers
    bin_centers = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
    grid_shape = (2, 2)

    graph = _create_connectivity_graph_generic(
        active_indices, bin_centers, grid_shape, get_2d_orthogonal_neighbors
    )

    assert graph.number_of_nodes() == 4
    # In a 2x2 grid with orthogonal connections: 4 edges (2 horizontal + 2 vertical)
    assert graph.number_of_edges() == 4

    # Check all nodes have degree 2
    degrees = [d for n, d in graph.degree()]
    assert all(deg == 2 for deg in degrees)

    # Check edge attributes
    for _u, _v, data in graph.edges(data=True):
        assert "distance" in data
        assert "vector" in data
        assert "edge_id" in data
        assert "angle_2d" in data  # Should exist for 2D grids
        assert data["distance"] > 0
        assert isinstance(data["edge_id"], int)


def test_create_connectivity_graph_generic_2x2_all_active_diagonal():
    """Test 2x2 grid with all bins active, diagonal connections."""
    active_indices = np.array([0, 1, 2, 3])
    bin_centers = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
    grid_shape = (2, 2)

    graph = _create_connectivity_graph_generic(
        active_indices, bin_centers, grid_shape, get_2d_diagonal_neighbors
    )

    assert graph.number_of_nodes() == 4
    # In a 2x2 grid with diagonal connections: 4 orthogonal + 2 diagonal = 6 edges
    assert graph.number_of_edges() == 6

    # Each corner bin should have 3 neighbors (2 orthogonal + 1 diagonal)
    degrees = [d for n, d in graph.degree()]
    assert all(deg == 3 for deg in degrees)


def test_create_connectivity_graph_generic_partial_active():
    """Test graph with only some bins active."""
    # Only activate bins 0 and 3 (opposite corners)
    active_indices = np.array([0, 3])
    bin_centers = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
    grid_shape = (2, 2)

    graph = _create_connectivity_graph_generic(
        active_indices, bin_centers, grid_shape, get_2d_diagonal_neighbors
    )

    assert graph.number_of_nodes() == 2
    # Bins 0 and 3 are diagonal neighbors, so 1 edge
    assert graph.number_of_edges() == 1

    # Check node IDs are remapped to 0 and 1
    assert set(graph.nodes()) == {0, 1}

    # Check that the correct original indices are stored
    assert graph.nodes[0]["source_grid_flat_index"] == 0
    assert graph.nodes[1]["source_grid_flat_index"] == 3


def test_create_connectivity_graph_generic_edge_attributes_correctness():
    """Test that edge attributes (distance, vector, angle) are computed correctly."""
    # 1D horizontal line: 3 bins at x=0.5, 1.5, 2.5
    active_indices = np.array([0, 1, 2])
    bin_centers = np.array([[0.5, 0.0], [1.5, 0.0], [2.5, 0.0]])
    grid_shape = (3, 1)

    def get_1d_horizontal_neighbors(
        flat_index: int, grid_shape: tuple[int, ...]
    ) -> list[int]:
        """Get left and right neighbors in 1D horizontal layout."""
        n_bins = grid_shape[0]
        neighbors = []
        if flat_index > 0:
            neighbors.append(flat_index - 1)
        if flat_index < n_bins - 1:
            neighbors.append(flat_index + 1)
        return neighbors

    graph = _create_connectivity_graph_generic(
        active_indices, bin_centers, grid_shape, get_1d_horizontal_neighbors
    )

    # Should have 2 edges: 0-1 and 1-2
    assert graph.number_of_edges() == 2

    # Check edge 0-1
    edge_01 = graph.edges[0, 1]
    assert np.isclose(edge_01["distance"], 1.0)
    assert np.allclose(edge_01["vector"], (1.0, 0.0))
    assert np.isclose(edge_01["angle_2d"], 0.0)  # Horizontal right = 0 radians

    # Check edge 1-2
    edge_12 = graph.edges[1, 2]
    assert np.isclose(edge_12["distance"], 1.0)
    assert np.allclose(edge_12["vector"], (1.0, 0.0))
    assert np.isclose(edge_12["angle_2d"], 0.0)


def test_create_connectivity_graph_generic_3d():
    """Test that graph building works for 3D grids (no angle_2d)."""
    # 2x2x2 cube, all bins active
    active_indices = np.arange(8)
    # Create 3D grid centers
    centers = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                centers.append([i + 0.5, j + 0.5, k + 0.5])
    bin_centers = np.array(centers)
    grid_shape = (2, 2, 2)

    def get_3d_orthogonal_neighbors(
        flat_index: int, grid_shape: tuple[int, ...]
    ) -> list[int]:
        """Get orthogonal neighbors in a 3D grid."""
        nz, ny, nx = grid_shape
        z = flat_index // (ny * nx)
        y = (flat_index % (ny * nx)) // nx
        x = flat_index % nx

        neighbors = []
        for dz, dy, dx in [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]:
            nz_new, ny_new, nx_new = z + dz, y + dy, x + dx
            if 0 <= nz_new < nz and 0 <= ny_new < ny and 0 <= nx_new < nx:
                neighbors.append(nz_new * ny * nx + ny_new * nx + nx_new)
        return neighbors

    graph = _create_connectivity_graph_generic(
        active_indices, bin_centers, grid_shape, get_3d_orthogonal_neighbors
    )

    assert graph.number_of_nodes() == 8
    # In a 2x2x2 cube with orthogonal connections: 12 edges
    assert graph.number_of_edges() == 12

    # Check that 3D edges do NOT have angle_2d
    for _u, _v, data in graph.edges(data=True):
        assert "distance" in data
        assert "vector" in data
        assert "edge_id" in data
        assert "angle_2d" not in data  # Should NOT exist for 3D grids


def test_create_connectivity_graph_generic_edge_ids_unique():
    """Test that edge IDs are unique and sequential."""
    active_indices = np.array([0, 1, 2, 3])
    bin_centers = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
    grid_shape = (2, 2)

    graph = _create_connectivity_graph_generic(
        active_indices, bin_centers, grid_shape, get_2d_orthogonal_neighbors
    )

    # Collect all edge IDs
    edge_ids = [data["edge_id"] for u, v, data in graph.edges(data=True)]

    # Check they are unique
    assert len(edge_ids) == len(set(edge_ids))

    # Check they are sequential from 0
    assert set(edge_ids) == set(range(len(edge_ids)))


def test_create_connectivity_graph_generic_no_self_loops():
    """Test that the graph contains no self-loops."""
    active_indices = np.array([0, 1, 2, 3])
    bin_centers = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
    grid_shape = (2, 2)

    graph = _create_connectivity_graph_generic(
        active_indices, bin_centers, grid_shape, get_2d_orthogonal_neighbors
    )

    # Check no self-loops
    assert nx.number_of_selfloops(graph) == 0


def test_create_connectivity_graph_generic_node_remapping():
    """Test that nodes are correctly remapped from original flat indices to 0...n-1."""
    # Activate non-consecutive bins: 1, 3, 5
    active_indices = np.array([1, 3, 5])
    bin_centers = np.array(
        [[0.5, 0.5], [1.5, 0.5], [2.5, 0.5], [0.5, 1.5], [1.5, 1.5], [2.5, 1.5]]
    )  # 2x3 grid
    grid_shape = (2, 3)

    graph = _create_connectivity_graph_generic(
        active_indices, bin_centers, grid_shape, get_2d_orthogonal_neighbors
    )

    # Nodes should be 0, 1, 2 (remapped)
    assert set(graph.nodes()) == {0, 1, 2}

    # Check original indices are preserved
    assert graph.nodes[0]["source_grid_flat_index"] == 1
    assert graph.nodes[1]["source_grid_flat_index"] == 3
    assert graph.nodes[2]["source_grid_flat_index"] == 5

    # Check original N-D indices
    assert graph.nodes[0]["original_grid_nd_index"] == (0, 1)
    assert graph.nodes[1]["original_grid_nd_index"] == (1, 0)
    assert graph.nodes[2]["original_grid_nd_index"] == (1, 2)
