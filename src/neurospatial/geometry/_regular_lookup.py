"""Pure point assignment on half-open rectilinear cells."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _points_to_regular_grid_bin_ind(
    points: NDArray[np.float64],
    grid_edges: tuple[NDArray[np.float64], ...],
    grid_shape: tuple[int, ...],
    active_mask: NDArray[np.bool_] | None = None,
    inverse_map: NDArray[np.intp] | None = None,
) -> NDArray[np.int_]:
    """Map N-D points to their corresponding bin indices in a regular grid.

    If `active_mask` is provided, maps to indices relative to active bins
    (0 to `n_active_bins - 1`). Otherwise, maps to flat indices of the
    full conceptual grid.

    Parameters
    ----------
    points : NDArray[np.float64], shape (n_points, n_dims)
        N-dimensional points to map. NaNs are filtered out.
    grid_edges : Tuple[NDArray[np.float64], ...]
        Tuple where each element is a 1D array of bin edge positions for one
        dimension of the full grid.
    grid_shape : Tuple[int, ...]
        N-D shape (number of bins in each dimension) of the full grid.
    active_mask : Optional[NDArray[np.bool_]], shape `grid_shape`, optional
        If provided, an N-D boolean mask indicating active bins in the full
        grid. Output indices will be relative to these active bins.
        If None (default), output indices are flat indices of the full grid.
    inverse_map : Optional[NDArray[np.intp]], shape (active_mask.size,), optional
        Pre-computed inverse mapping from flat grid index to active bin index.
        If provided with active_mask, avoids recomputing the inverse map on
        each call. Should satisfy: inverse_map[flat_idx] = active_bin_idx
        for active bins, -1 otherwise.

    Returns
    -------
    bin_indices : NDArray[np.int_], shape (n_valid_points,)
        Integer indices of the bins corresponding to each valid input point.
        - If `active_mask` is provided: Indices are `0` to `n_active_bins - 1`.
          Points outside active bins or grid boundaries get -1.
        - If `active_mask` is None: Flat indices of the full grid. Points
          outside grid boundaries get -1 or out-of-range indices (depending
          on `np.digitize` behavior for edge cases, typically -1 due to clipping).

    """
    points_atleast_2d = np.atleast_2d(points)
    valid_input_mask = ~np.any(np.isnan(points_atleast_2d), axis=1)

    # Initialize output assuming all points are invalid or unmapped
    # Output shape should match original number of points (before NaN filter)
    output_indices = np.full(points_atleast_2d.shape[0], -1, dtype=int)

    if not np.any(valid_input_mask):  # No valid points after NaN filter
        return output_indices

    valid_points = points_atleast_2d[valid_input_mask]
    if valid_points.shape[0] == 0:  # Should be caught by above, but defensive
        return output_indices

    n_dims = valid_points.shape[1]
    if n_dims != len(grid_edges) or n_dims != len(grid_shape):
        raise ValueError(
            f"Dimensionality mismatch: points have {n_dims} dimension(s), but "
            f"grid_edges has {len(grid_edges)} and grid_shape has "
            f"{len(grid_shape)}. Points must match the grid's dimensionality."
        )

    # Calculate N-D indices for valid_points
    multi_bin_idx_list = []
    point_is_within_grid_bounds = np.ones(valid_points.shape[0], dtype=bool)

    for i in range(n_dims):
        # np.digitize returns indices from 1 to len(bins)+1
        # We subtract 1 to get 0-based bin indices
        dim_indices = np.digitize(valid_points[:, i], grid_edges[i]) - 1

        # Check if indices are within the valid range [0, grid_shape[i]-1]
        point_is_within_grid_bounds &= (dim_indices >= 0) & (
            dim_indices < grid_shape[i]
        )
        multi_bin_idx_list.append(dim_indices)

    # Initialize flat indices for valid_points to -1
    original_bin_flat_idx_for_valid_points = np.full(
        valid_points.shape[0],
        -1,
        dtype=int,
    )

    if np.any(point_is_within_grid_bounds):
        # Filter to only points that are fully within grid bounds for ravel_multi_index
        coords_for_ravel = tuple(
            idx[point_is_within_grid_bounds] for idx in multi_bin_idx_list
        )

        # np.ravel_multi_index requires all coords to be within dimension bounds
        original_bin_flat_idx_for_valid_points[point_is_within_grid_bounds] = (
            np.ravel_multi_index(coords_for_ravel, grid_shape)
        )

    # Place these calculated flat indices (or -1 for out-of-bounds) back into the full output array
    # This mapping depends on whether an active_mask is used for final indexing.

    final_mapped_indices_for_valid_points = np.full(
        valid_points.shape[0],
        -1,
        dtype=int,
    )

    if active_mask is not None:
        # Vectorized mapping from original_full_grid_flat_index to active_bin_id (0 to N-1)
        active_mask_flat = active_mask.ravel()

        # Use provided inverse_map if available, otherwise compute it
        if inverse_map is None:
            # Create inverse lookup array: inverse_map[original_flat_idx] = active_bin_id
            # For inactive bins, inverse_map contains -1
            inverse_map = np.full(active_mask_flat.size, -1, dtype=np.intp)
            active_indices = np.flatnonzero(active_mask_flat)
            inverse_map[active_indices] = np.arange(len(active_indices), dtype=np.intp)

        # Vectorized lookup: find points with valid flat indices that are also in active bins
        valid_flat_mask = original_bin_flat_idx_for_valid_points != -1

        # For valid flat indices, check if they're in active bins and map them
        # Use clip to avoid out-of-bounds indexing (clipped values will have valid_flat_mask=False)
        safe_indices = np.clip(
            original_bin_flat_idx_for_valid_points, 0, active_mask_flat.size - 1
        )
        in_active_mask = valid_flat_mask & active_mask_flat[safe_indices]

        # Map valid active indices using the inverse lookup
        final_mapped_indices_for_valid_points[in_active_mask] = inverse_map[
            original_bin_flat_idx_for_valid_points[in_active_mask]
        ]
    else:
        # No active_mask, so original_bin_flat_idx_for_valid_points are the final indices
        # (where -1 means out of bounds)
        final_mapped_indices_for_valid_points = original_bin_flat_idx_for_valid_points

    output_indices[valid_input_mask] = final_mapped_indices_for_valid_points
    return output_indices
