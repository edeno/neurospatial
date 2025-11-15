"""Generic graph connectivity building for layout engines.

This module provides a generic implementation for creating NetworkX graphs
that represent the connectivity between active bins in different layout types.
By abstracting the neighbor-finding logic, it eliminates code duplication
between regular grid, hexagonal, and other layout engines.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray


def _create_connectivity_graph_generic(
    active_original_flat_indices: NDArray[np.int_],
    full_grid_bin_centers: NDArray[np.float64],
    grid_shape: tuple[int, ...],
    get_neighbor_flat_indices: Callable[[int, tuple[int, ...]], list[int]],
) -> nx.Graph:
    """Create a connectivity graph for active bins with custom neighbor logic.

    This is a generic helper function that creates a NetworkX graph representing
    the spatial connectivity between active bins in a layout. It delegates the
    layout-specific neighbor-finding logic to a callback function, allowing it
    to work with any grid topology (regular, hexagonal, etc.).

    The resulting graph has nodes indexed from 0 to n_active_bins-1, with each
    node storing its spatial coordinates and mapping back to the original grid.
    Edges connect neighboring active bins and store geometric attributes like
    distance and displacement vectors.

    Parameters
    ----------
    active_original_flat_indices : NDArray[np.int_], shape (n_active_bins,)
        Array of flat indices (row-major order) corresponding to bins that are
        active in the original full grid. These are the bins that will become
        nodes in the graph.
    full_grid_bin_centers : NDArray[np.float64], shape (n_total_bins, n_dims)
        Coordinates of centers of *all* bins in the original full grid, ordered
        by flattened grid index (row-major). Even inactive bins must be included
        to maintain index consistency.
    grid_shape : tuple[int, ...]
        Shape of the original full grid (n_bins_dim0, n_bins_dim1, ...).
        For example, a 2D grid with 3 rows and 4 columns would be (3, 4).
    get_neighbor_flat_indices : Callable[[int, tuple[int, ...]], list[int]]
        Callback function that implements the layout-specific neighbor-finding
        logic. Takes a flat index and the grid shape, and returns a list of flat
        indices of its neighbors in the original grid (including inactive bins).
        The generic helper will filter out inactive neighbors automatically.

    Returns
    -------
    connectivity_graph : nx.Graph
        NetworkX graph with nodes indexed from 0 to n_active_bins-1. Each node
        has the following attributes:

        - 'pos' : tuple[float, ...]
            N-dimensional coordinates of the bin center.
        - 'source_grid_flat_index' : int
            Original flat index in the full grid.
        - 'original_grid_nd_index' : tuple[int, ...]
            Original N-dimensional index tuple in the full grid.

        Each edge has the following attributes:

        - 'distance' : float
            Euclidean distance between bin centers.
        - 'vector' : tuple[float, ...]
            Displacement vector from source to target bin (pos_v - pos_u).
        - 'edge_id' : int
            Unique sequential identifier for the edge (0, 1, 2, ...).
        - 'angle_2d' : float, optional
            Angle in radians from source to target (atan2(dy, dx)).
            Only present for 2D grids (n_dims == 2).

    Notes
    -----
    This function eliminates ~70% code duplication between regular grid and
    hexagonal layout connectivity builders by providing a common implementation.

    The neighbor callback function should return all potential neighbors (active
    or inactive). This function will automatically filter to keep only edges
    between active bins.

    Examples
    --------
    >>> # Define a simple 2D orthogonal neighbor finder
    >>> def get_orthogonal_neighbors(flat_idx, grid_shape):
    ...     n_rows, n_cols = grid_shape
    ...     row, col = flat_idx // n_cols, flat_idx % n_cols
    ...     neighbors = []
    ...     for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    ...         r, c = row + dr, col + dc
    ...         if 0 <= r < n_rows and 0 <= c < n_cols:
    ...             neighbors.append(r * n_cols + c)
    ...     return neighbors
    >>> # Create a 2x2 grid with all bins active
    >>> active_indices = np.array([0, 1, 2, 3])
    >>> bin_centers = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
    >>> graph = _create_connectivity_graph_generic(
    ...     active_indices, bin_centers, (2, 2), get_orthogonal_neighbors
    ... )
    >>> graph.number_of_nodes()
    4
    >>> graph.number_of_edges()
    4
    """
    # Initialize empty graph
    connectivity_graph = nx.Graph()

    # 1. Handle empty case
    n_active_bins = len(active_original_flat_indices)
    if n_active_bins == 0:
        return connectivity_graph

    # 2. Create mapping from original flat index to new node ID (0 to n_active_bins-1)
    original_flat_to_new_node_id_map: dict[int, int] = {
        int(original_idx): new_idx
        for new_idx, original_idx in enumerate(active_original_flat_indices)
    }

    # 3. Add nodes to the graph with new IDs and attributes
    n_dims = full_grid_bin_centers.shape[1]

    for new_node_id, original_flat_idx in enumerate(active_original_flat_indices):
        # Convert flat index to N-D index tuple
        original_nd_idx_tuple = tuple(
            np.unravel_index(int(original_flat_idx), grid_shape)
        )

        # Get bin center coordinates
        pos_coordinates = tuple(full_grid_bin_centers[int(original_flat_idx)])

        # Add node with standard attributes
        connectivity_graph.add_node(
            new_node_id,
            pos=pos_coordinates,
            source_grid_flat_index=int(original_flat_idx),
            original_grid_nd_index=original_nd_idx_tuple,
        )

    # 4. Add edges between active neighbor nodes
    edges_to_add_with_attrs = []

    for new_node_id, original_flat_idx in enumerate(active_original_flat_indices):
        # Get neighbor flat indices using the callback
        neighbor_flat_indices = get_neighbor_flat_indices(
            int(original_flat_idx), grid_shape
        )

        # Check each neighbor
        for neighbor_flat_idx in neighbor_flat_indices:
            # Only add edge if neighbor is also active
            if neighbor_flat_idx in original_flat_to_new_node_id_map:
                neighbor_node_id = original_flat_to_new_node_id_map[neighbor_flat_idx]

                # Add edge only if u < v to avoid duplicates (undirected graph)
                if new_node_id < neighbor_node_id:
                    # Get positions
                    pos_u = np.asarray(connectivity_graph.nodes[new_node_id]["pos"])
                    pos_v = np.asarray(
                        connectivity_graph.nodes[neighbor_node_id]["pos"]
                    )

                    # Compute edge attributes
                    displacement_vector = pos_v - pos_u
                    distance = float(np.linalg.norm(displacement_vector))

                    edge_attrs: dict[str, Any] = {
                        "distance": distance,
                        "vector": tuple(displacement_vector.tolist()),
                    }

                    # Add angle_2d only for 2D grids
                    if n_dims == 2:
                        edge_attrs["angle_2d"] = math.atan2(
                            displacement_vector[1], displacement_vector[0]
                        )

                    edges_to_add_with_attrs.append(
                        (new_node_id, neighbor_node_id, edge_attrs)
                    )

    # Add all edges with their attributes
    connectivity_graph.add_edges_from(edges_to_add_with_attrs)

    # 5. Add unique edge IDs (sequential starting from 0)
    for edge_id_counter, (u, v) in enumerate(connectivity_graph.edges()):
        connectivity_graph.edges[u, v]["edge_id"] = edge_id_counter

    return connectivity_graph
