"""Utility functions for creating and managing hexagonal grid layouts.

This module provides helper functions specifically for environments that use
hexagonal tiling. These functions cover:
- Generation of hexagonal grid coordinates (`_create_hex_grid`).
- Conversion between Cartesian and hexagonal (cube/axial) coordinate systems
  (`cartesian_to_fractional_cube`, `round_fractional_cube_to_integer_axial`,
  `axial_to_offset_bin_indices`).
- Mapping continuous 2D points to discrete hexagonal bin indices
  (`points_to_hex_bin_ind`).
- Inferring active hexagonal bins based on data sample occupancy
  (`_infer_active_bins_from_hex_grid`).
- Determining neighbor relationships in the hexagonal grid
  (`get_hex_grid_neighbor_deltas`).
- Constructing a connectivity graph for active hexagonal bins
  (`_create_hex_connectivity_graph`).

These utilities are primarily used by the `HexagonalLayout` engine.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray

# Numerical stability constants
# Minimum hex_radius to prevent division by zero and numerical instability.
# Values below this threshold are too small for reliable hexagonal coordinate conversion.
from neurospatial.geometry import (
    get_hex_grid_neighbor_deltas,
    points_to_hex_bin_ind,
)


def _create_hex_grid(
    positions: NDArray[np.float64] | None,
    dimension_range: Sequence[tuple[float, float]] | None = None,
    hexagon_width: float = 1.0,
) -> tuple[
    NDArray[np.float64],  # bin_centers
    tuple[int, int],  # centers_shape
    float,  # hex_radius
    float,  # hex_orientation
    float,  # min_x
    float,  # min_y
    Sequence[tuple[float, float]],  # effective_dimension_range
]:
    """Generate a 2D hexagonal grid (pointy-top) that covers either:
    - A user-specified bounding box (`dimension_range`), or
    - The min/max extent of `positions`.

    Parameters
    ----------
    positions : ndarray of shape (n_samples, 2), optional
        2D points to infer bounding box if `dimension_range` is None.
        NaNs are ignored. If None, `dimension_range` must be provided.
    dimension_range : sequence of ((min_x, max_x), (min_y, max_y)), optional
        If provided, must be length 2. Otherwise inferred from `positions`.
    hexagon_width : float, default=1.0
        Distance between parallel sides of each hexagon; must be positive.

    Returns
    -------
    bin_centers : ndarray, shape (n_hex_rows * n_hex_cols, 2)
        (x, y) coordinates of every hexagon center in the full grid.
    centers_shape : (n_hex_rows, n_hex_cols)
        The 2D shape of the conceptual hex-grid, before flattening.
    hex_radius : float
        Distance from center to any vertex (hexagon side length).
    hex_orientation : float
        Orientation in radians (0.0 = point-up).
    min_x : float
        Minimum x coordinate of the bounding box (grid origin).
    min_y : float
        Minimum y coordinate of the bounding box (grid origin).
    effective_dimension_range : [ (min_x, max_x), (min_y, max_y) ]
        The actual range used (either user‐provided or inferred).

    Raises
    ------
    ValueError
        - If `hexagon_width <= 0`.
        - If neither `dimension_range` nor `positions` (with valid shape) is provided.
        - If `dimension_range` is provided but not length 2 or has min > max.
        - If `positions` is not a 2D array of shape (n_samples, 2).

    """
    # 1) Validate hexagon_width
    if hexagon_width <= 0:
        raise ValueError(
            f"`hexagon_width` must be a positive float (got {hexagon_width})."
        )

    hex_orientation = 0.0  # point-up

    # 2) Clean & validate positions
    if positions is not None:
        ds_arr = np.asarray(positions, dtype=float)
        if ds_arr.ndim != 2 or ds_arr.shape[1] != 2:
            raise ValueError("`positions` must be shape (n_samples, 2).")
        # Remove any rows containing NaNs
        ds_arr = ds_arr[~np.isnan(ds_arr).any(axis=1)]
    else:
        ds_arr = np.empty((0, 2), dtype=float)

    # 3) Determine bounding box
    if dimension_range is not None:
        if len(dimension_range) != 2:
            raise ValueError(
                f"`dimension_range` must be ((min_x,max_x),(min_y,max_y)), got length {len(dimension_range)}."
            )
        (ux0, ux1), (uy0, uy1) = dimension_range
        min_x, max_x = sorted((float(ux0), float(ux1)))
        min_y, max_y = sorted((float(uy0), float(uy1)))
        effective_range = [(min_x, max_x), (min_y, max_y)]
    else:
        if ds_arr.shape[0] == 0:
            raise ValueError("`positions` is empty; cannot infer bounding box.")
        min_vals = np.min(ds_arr, axis=0)
        max_vals = np.max(ds_arr, axis=0)
        min_x, max_x = float(min_vals[0]), float(max_vals[0])
        min_y, max_y = float(min_vals[1]), float(max_vals[1])
        effective_range = [(min_x, max_x), (min_y, max_y)]

    # 4) Compute hex geometry constants
    vertical_step = (np.sqrt(3.0) / 2.0) * hexagon_width
    hex_radius = hexagon_width / np.sqrt(3)

    # 5) Compute how many hexes fit in x/y direction (ensure at least one)
    span_x = max_x - min_x
    span_y = max_y - min_y
    n_hex_cols = max(1, int(np.ceil(span_x / hexagon_width)) + 1)
    n_hex_rows = max(1, int(np.ceil(span_y / vertical_step)) + 1)

    # 6) Create a regular grid of (row, col) indices
    col_idx, row_idx = np.meshgrid(
        np.arange(n_hex_cols),
        np.arange(n_hex_rows),
        indexing="xy",
    )
    col_idx = col_idx.astype(float)
    row_idx = row_idx.astype(float)

    # 7) Shift odd rows horizontally by half‐cell
    col_idx[1::2, :] += 0.5

    # 8) Scale to physical coordinates, then translate by (min_x, min_y)
    x_coords = col_idx * hexagon_width + min_x
    y_coords = row_idx * vertical_step + min_y

    bin_centers = np.column_stack((x_coords.ravel(), y_coords.ravel()))
    centers_shape = (n_hex_rows, n_hex_cols)

    return (
        bin_centers,
        centers_shape,
        hex_radius,
        hex_orientation,
        min_x,
        min_y,
        effective_range,
    )


def _infer_active_bins_from_hex_grid(
    positions: NDArray[np.float64],
    centers_shape: tuple[int, int],
    hex_radius: float,
    min_x: float,
    min_y: float,
    bin_count_threshold: int = 0,
) -> NDArray[np.int_]:
    """Infer active bins in a hexagonal grid based on data sample occupancy.

    Maps `positions` to their respective hexagon bins within the full
    conceptual grid. Hexagons are marked active if their occupancy count
    exceeds `bin_count_threshold`.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, 2)
        2D data samples (e.g., positions).
    centers_shape : Tuple[int, int]
        Shape (n_hex_rows, n_hex_cols) of the full conceptual grid of hexagons.
    hex_radius : float
        Radius of the hexagons.
    min_x : float
        Minimum x-coordinate of the grid's effective origin.
    min_y : float
        Minimum y-coordinate of the grid's effective origin.
    bin_count_threshold : int, optional, default=0
        Minimum number of samples a hexagon must contain to be considered active.

    Returns
    -------
    NDArray[np.int_], shape (n_active_hex_bins,)
        An array of original flat indices (relative to the full conceptual grid)
        of the hexagons that were deemed active. Returns an empty array if no
        bins are found to be active.

    """
    bin_ind = points_to_hex_bin_ind(
        points=positions,
        grid_offset_x=min_x,
        grid_offset_y=min_y,
        hex_radius=hex_radius,
        centers_shape=centers_shape,
    )
    # Filter out invalid bin indices
    bin_ind = bin_ind[bin_ind >= 0]

    if len(bin_ind) == 0:
        # No valid bins found, return an empty array
        return np.array([], dtype=int)

    # Count occurrences of each unique bin ind
    bin_ind, bin_count = np.unique(bin_ind, return_counts=True)

    # Filter bins based on the count threshold
    bin_ind = bin_ind[bin_count >= bin_count_threshold]

    return bin_ind.astype(int)


def _create_hex_connectivity_graph(
    active_original_flat_indices: NDArray[np.int_],
    full_grid_bin_centers: NDArray[np.float64],
    centers_shape: tuple[int, int],
) -> nx.Graph:
    """Create a connectivity graph for active bins in a hexagonal grid.

    Nodes in the returned graph are indexed from `0` to `n_active_bins - 1`.
    Each node corresponds to an active hexagon and stores attributes:
    - 'pos': (x, y) coordinates of the active hexagon's center.
    - 'source_grid_flat_index': Original flat index in the full conceptual hex grid.
    - 'original_grid_nd_index': (row, col) index in the full conceptual hex grid.

    Edges connect active hexagons that are immediate neighbors in the lattice.
    Edges store 'distance', 'vector', and 'angle_2d'.

    Parameters
    ----------
    active_original_flat_indices : NDArray[np.int_], shape (n_active_bins,)
        Array of original flat indices (relative to the full conceptual grid)
        for hexagons that are active.
    full_grid_bin_centers : NDArray[np.float64], shape (n_total_hex_bins, 2)
        Center coordinates of *all* potential hexagons in the full grid.
    centers_shape : Tuple[int, int]
        Shape (n_hex_rows, n_hex_cols) of the full conceptual grid of hexagons.

    Returns
    -------
    nx.Graph
        Connectivity graph of active hexagonal bins. Nodes are re-indexed
        `0` to `n_active_bins - 1`.

    Notes
    -----
    This function now delegates to the generic connectivity graph builder,
    providing a hexagonal grid-specific neighbor-finding callback.

    """
    # Import the generic helper (local import to avoid circular dependencies)
    from neurospatial.layout.helpers.graph_building import (
        _create_connectivity_graph_generic,
    )

    # Define neighbor-finding callback for hexagonal grids
    def get_hex_grid_neighbors(
        flat_index: int, grid_shape: tuple[int, ...]
    ) -> list[int]:
        """Get neighbor flat indices for a hexagonal grid bin.

        Parameters
        ----------
        flat_index : int
            Flat index of the current hexagon.
        grid_shape : tuple[int, ...]
            Shape of the hexagonal grid (n_hex_rows, n_hex_cols).

        Returns
        -------
        list[int]
            List of flat indices of neighbors in the hexagonal lattice.
        """
        n_hex_y, n_hex_x = grid_shape

        # Convert flat index to 2D index
        row_idx = flat_index // n_hex_x
        col_idx = flat_index % n_hex_x

        # Determine if the current row is odd or even
        is_odd_row = (row_idx % 2) == 1

        # Get the neighbor deltas based on the row parity
        neighbor_deltas = get_hex_grid_neighbor_deltas(is_odd_row)

        # Find neighbors
        neighbors = []
        for delta_col, delta_row in neighbor_deltas:
            neighbor_row = row_idx + delta_row
            neighbor_col = col_idx + delta_col

            # Check if the neighbor is within bounds
            if 0 <= neighbor_row < n_hex_y and 0 <= neighbor_col < n_hex_x:
                # Calculate the original flat index of the neighbor
                neighbor_flat_index = neighbor_row * n_hex_x + neighbor_col
                neighbors.append(neighbor_flat_index)

        return neighbors

    # Use generic helper to build the graph
    return _create_connectivity_graph_generic(
        active_original_flat_indices,
        full_grid_bin_centers,
        centers_shape,
        get_hex_grid_neighbors,
    )
