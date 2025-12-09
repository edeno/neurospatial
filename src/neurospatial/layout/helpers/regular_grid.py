"""Utility functions for creating and managing regular N-dimensional grid layouts.

This module provides a collection of helper functions used primarily by
`RegularGridLayout` and other grid-based layout engines within the
`neurospatial` package. These functions handle tasks such as:

- Defining the structure (bin edges, bin centers, shape) of a regular N-D grid
  based on data samples or specified dimension ranges (`_create_regular_grid`).
- Inferring which bins within this grid are "active" based on the density of
  provided data samples, often involving morphological operations to refine
  the active area (`_infer_active_bins_from_regular_grid`).
- Constructing a `networkx.Graph` that represents the connectivity between
  these active grid bins, allowing for orthogonal and diagonal connections
  (`_create_regular_grid_connectivity_graph`).
- Mapping continuous N-D points to their corresponding discrete bin indices
  within the grid, taking into account active areas (`_points_to_regular_grid_bin_ind`).

The module also includes functions for inferring dimensional properties from
data samples, which might be shared with or used by other utility modules.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from neurospatial.layout.helpers.utils import get_centers, get_n_bins


def _create_regular_grid_connectivity_graph(
    full_grid_bin_centers: NDArray[np.float64],
    active_mask_nd: NDArray[np.bool_],
    grid_shape: tuple[int, ...],
    connect_diagonal: bool = False,
) -> nx.Graph:
    """Create a graph connecting centers of active bins in an N-D grid.

    Nodes in the returned graph are indexed from `0` to `n_active_bins - 1`.
    Each node stores attributes:
    - 'pos': N-D coordinates of the active bin center.
    - 'source_grid_flat_index': Original flat index in the full conceptual grid.
    - 'original_grid_nd_index': Original N-D tuple index in the full grid.

    Edges connect active bins that are adjacent (orthogonally or, optionally,
    diagonally) in the N-D grid. Edges store 'distance', 'vector', and 'angle_2d'
    attributes.

    Parameters
    ----------
    full_grid_bin_centers : NDArray[np.float64], shape (n_total_bins, n_dims)
        Coordinates of centers of *all* bins in the original full grid,
        ordered by flattened grid index (row-major).
    active_mask_nd : NDArray[np.bool_], shape (dim0_size, dim1_size, ...)
        N-dimensional boolean mask indicating active bins in the full grid.
        Must match `grid_shape`.
    grid_shape : Tuple[int, ...]
        The N-D shape (number of bins in each dimension) of the original full grid.
    connect_diagonal : bool, default=False
        If True, connect diagonally adjacent active bins. Otherwise, only
        orthogonally adjacent active bins are connected.

    Returns
    -------
    connectivity_graph : nx.Graph
        Graph of active bins. Nodes are re-indexed `0` to `n_active_bins - 1`.

    Raises
    ------
    ValueError
        If input shapes (`full_grid_bin_centers`, `active_mask_nd`,
        `grid_shape`) are inconsistent.

    Notes
    -----
    This function now delegates to the generic connectivity graph builder,
    providing a regular grid-specific neighbor-finding callback.

    """
    # Import the generic helper (local import to avoid circular dependencies)
    from neurospatial.layout.helpers.graph_building import (
        _create_connectivity_graph_generic,
    )

    # Validate inputs
    if full_grid_bin_centers.shape[0] != np.prod(grid_shape):
        raise ValueError(
            f"Mismatch: full_grid_bin_centers length ({full_grid_bin_centers.shape[0]}) "
            f"and product of grid_shape ({np.prod(grid_shape)}).",
        )
    if active_mask_nd.shape != grid_shape:
        raise ValueError(
            f"Shape of active_mask_nd {active_mask_nd.shape} "
            f"does not match grid_shape {grid_shape}.",
        )

    # Extract active bin indices from mask
    active_original_flat_indices = np.flatnonzero(active_mask_nd)

    # Define neighbor-finding callback for regular grids
    def get_regular_grid_neighbors(
        flat_index: int, grid_shape: tuple[int, ...]
    ) -> list[int]:
        """Get neighbor flat indices for a regular grid bin.

        Parameters
        ----------
        flat_index : int
            Flat index of the current bin.
        grid_shape : tuple[int, ...]
            Shape of the grid.

        Returns
        -------
        list[int]
            List of flat indices of neighbors in the grid.
        """
        n_dims = len(grid_shape)

        # Convert flat index to N-D index
        nd_index = np.array(np.unravel_index(flat_index, grid_shape))

        # Define neighbor offsets based on connectivity mode
        if connect_diagonal:
            # All combinations of -1, 0, 1 across n_dims, excluding (0,0,...)
            neighbor_offsets = [
                offset
                for offset in itertools.product([-1, 0, 1], repeat=n_dims)
                if not all(o == 0 for o in offset)
            ]
        else:  # Orthogonal neighbors only
            neighbor_offsets = []
            for dim_idx in range(n_dims):
                for offset_val in [-1, 1]:
                    offset = [0] * n_dims
                    offset[dim_idx] = offset_val
                    neighbor_offsets.append(tuple(offset))

        # Find neighbors
        neighbors = []
        for offset_tuple in neighbor_offsets:
            neighbor_nd_index = nd_index + np.array(offset_tuple)

            # Check if neighbor is within grid bounds
            if all(0 <= neighbor_nd_index[d] < grid_shape[d] for d in range(n_dims)):
                # Convert back to flat index
                neighbor_flat_idx = int(
                    np.ravel_multi_index(tuple(neighbor_nd_index), grid_shape)
                )
                neighbors.append(neighbor_flat_idx)

        return neighbors

    # Use generic helper to build the graph
    return _create_connectivity_graph_generic(
        active_original_flat_indices,
        full_grid_bin_centers,
        grid_shape,
        get_regular_grid_neighbors,
    )


def _infer_active_bins_from_regular_grid(
    positions: NDArray[np.float64],
    edges: tuple[NDArray[np.float64], ...],
    close_gaps: bool = False,
    fill_holes: bool = False,
    dilate: bool = False,
    bin_count_threshold: int = 0,
    boundary_exists: bool = False,
) -> NDArray[np.bool_]:
    """Infer active bins in a regular grid based on data sample density.

    This function first counts positions in each grid bin defined by `edges`.
    Bins with counts above `bin_count_threshold` are initially marked active.
    Optional morphological operations (closing, filling, dilation) can then be
    applied to refine the active area.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        N-dimensional data samples (e.g., positions). NaNs are ignored.
    edges : Tuple[NDArray[np.float64], ...]
        A tuple where each element is a 1D array of bin edge positions for
        one dimension of the grid.
    close_gaps : bool, default=False
        If True, apply binary closing (dilation then erosion) to close small
        gaps in the active area.
    fill_holes : bool, default=False
        If True, apply binary hole filling to fill holes enclosed by active bins.
    dilate : bool, default=False
        If True, apply binary dilation to expand the boundary of the active area.
    bin_count_threshold : int, default=0
        Minimum number of samples a bin must contain to be initially
        considered active (before morphological operations).
    boundary_exists : bool, default=False
        If True, explicitly mark the outermost layer of bins in each dimension
        as inactive *after* morphological operations. This can be used if
        `add_boundary_bins` was True during grid creation to ensure these
        added boundary bins are not part of the inferred active track.

    Returns
    -------
    active_mask : NDArray[np.bool_], shape (n_bins_dim0, n_bins_dim1, ...)
        N-dimensional boolean mask indicating which bins in the grid are
        considered active or part of the track interior.

    """
    pos_clean = positions[~np.any(np.isnan(positions), axis=1)]

    if pos_clean.shape[0] == 0:
        # Handle case with no valid positions
        grid_shape = tuple(len(edge_array) - 1 for edge_array in edges)
        warnings.warn(
            "infer_active_bins is True, but no positions provided. "
            "Returning all bins inactive (no data to infer from).",
            UserWarning,
        )
        return np.zeros(grid_shape, dtype=bool)

    bin_counts, _ = np.histogramdd(pos_clean, bins=edges)
    active_mask = bin_counts > bin_count_threshold

    n_dims = positions.shape[1]
    if n_dims > 1:
        # Use connectivity=1 for 4-neighbor (2D) or 6-neighbor (3D) etc.
        structure = ndimage.generate_binary_structure(n_dims, connectivity=2)

        if close_gaps:
            # Closing operation first (dilation then erosion) to close small gaps
            active_mask = ndimage.binary_closing(active_mask, structure=structure)

        if fill_holes:
            # Fill larger holes enclosed by occupied bins
            active_mask = ndimage.binary_fill_holes(active_mask, structure=structure)

        if dilate:
            # Expand the occupied area
            active_mask = ndimage.binary_dilation(active_mask, structure=structure)

    if boundary_exists:
        if active_mask.ndim == 1 or (
            active_mask.ndim == 2 and active_mask.shape[1] == 1
        ):
            if active_mask.size > 0:
                active_mask[0] = False
            if active_mask.size > 1:
                active_mask[-1] = False
        elif active_mask.ndim > 1 and active_mask.size > 0:
            for axis_n in range(active_mask.ndim):
                slicer_first: list[slice | int] = [slice(None)] * active_mask.ndim
                slicer_first[axis_n] = 0
                active_mask[tuple(slicer_first)] = False
                slicer_last: list[slice | int] = [slice(None)] * active_mask.ndim
                slicer_last[axis_n] = -1
                active_mask[tuple(slicer_last)] = False

    return active_mask.astype(bool)


def _validate_and_prepare_inputs(
    positions: NDArray[np.float64] | None,
    bin_size: float | Sequence[float],
    dimension_range: Sequence[tuple[float, float]] | None,
) -> tuple[NDArray[np.float64] | None, int, NDArray[np.float64]]:
    """Validate inputs and prepare normalized parameters for grid creation.

    Parameters
    ----------
    positions : NDArray[np.float64] or None
        N-dimensional data samples. If None, dimension_range must be provided.
    bin_size : float or sequence of floats
        Bin size(s) for each dimension.
    dimension_range : sequence of (min, max) tuples or None
        Explicit dimension ranges. If None, positions must be provided.

    Returns
    -------
    samples : NDArray[np.float64] or None
        Cleaned positions array (NaNs removed) or None if positions not provided.
    n_dims : int
        Number of dimensions.
    bin_sizes : NDArray[np.float64], shape (n_dims,)
        Normalized bin sizes for each dimension.

    Raises
    ------
    ValueError
        If both positions and dimension_range are None, or if validation fails.
    TypeError
        If inputs have incorrect types.

    """
    # 1) Determine dimensionality
    if positions is None and dimension_range is None:
        raise ValueError("Either `positions` or `dimension_range` must be provided.")

    if positions is not None:
        # Validate and convert positions with helpful error messages
        try:
            samples = np.asarray(positions, dtype=float)
        except (TypeError, ValueError) as e:
            actual_type = type(positions).__name__
            raise TypeError(
                f"positions must be a numeric array-like object (e.g., numpy array, "
                f"list of lists, pandas DataFrame). Got {actual_type}: {positions!r}"
            ) from e

        if samples.ndim != 2:
            raise ValueError(f"`positions` must be 2D, got shape {samples.shape}.")
        n_dims = samples.shape[1]
        # Remove NaNs
        samples = samples[~np.isnan(samples).any(axis=1)]
        if samples.size == 0 and dimension_range is None:
            raise ValueError(
                "`positions` has no valid points and no `dimension_range` given.",
            )
    else:
        samples = None
        if dimension_range is None:
            raise ValueError("dimension_range must be provided when positions is None")
        n_dims = len(dimension_range)

    # 2) Normalize & validate bin_size
    if isinstance(bin_size, (float, int)):
        try:
            bin_sizes = np.full(n_dims, float(bin_size))
        except (TypeError, ValueError) as e:
            actual_type = type(bin_size).__name__
            raise TypeError(
                f"bin_size must be a numeric value. Got {actual_type}: {bin_size!r}"
            ) from e
    else:
        try:
            bin_sizes = np.asarray(bin_size, dtype=float)
        except (TypeError, ValueError) as e:
            actual_type = type(bin_size).__name__
            raise TypeError(
                f"bin_size must be a numeric value or sequence of numeric values. "
                f"Got {actual_type}: {bin_size!r}"
            ) from e

        if bin_sizes.ndim != 1 or bin_sizes.shape[0] != n_dims:
            raise ValueError(f"`bin_size` length must be {n_dims}, got {bin_sizes}.")

    # Check for NaN or Inf values
    if np.any(np.isnan(bin_sizes)):
        raise ValueError(
            f"[E1002] bin_size contains NaN (Not a Number) values (got {bin_size}). "
            "bin_size must be finite numeric values."
        )
    if np.any(np.isinf(bin_sizes)):
        raise ValueError(
            f"[E1002] bin_size contains infinite values (got {bin_size}). "
            "bin_size must be finite numeric values."
        )

    if np.any(bin_sizes <= 0.0):
        raise ValueError(
            f"[E1002] All elements of `bin_size` must be positive (got {bin_size})."
        )

    return samples, n_dims, bin_sizes


def _compute_dimension_ranges(
    samples: NDArray[np.float64] | None,
    dimension_range: Sequence[tuple[float, float]] | None,
    n_dims: int,
    bin_sizes: NDArray[np.float64],
) -> list[tuple[float, float]]:
    """Compute dimension ranges from explicit specification or data samples.

    Parameters
    ----------
    samples : NDArray[np.float64] or None
        Cleaned data samples (NaNs already removed).
    dimension_range : sequence of (min, max) tuples or None
        Explicit dimension ranges.
    n_dims : int
        Number of dimensions.
    bin_sizes : NDArray[np.float64], shape (n_dims,)
        Bin sizes for each dimension.

    Returns
    -------
    ranges : list of (float, float) tuples
        Computed dimension ranges for each dimension.

    Raises
    ------
    ValueError
        If dimension_range length doesn't match n_dims, or if ranges cannot be inferred.
    TypeError
        If dimension_range contains non-numeric values.

    """
    # 3) Determine dimension ranges
    if dimension_range is not None:
        if len(dimension_range) != n_dims:
            raise ValueError(
                f"`dimension_range` length ({len(dimension_range)}) must match n_dims ({n_dims}).",
            )
        ranges = []
        for i, ((lo, hi), size) in enumerate(
            zip(dimension_range, bin_sizes, strict=False)
        ):
            try:
                lo_f, hi_f = float(min(lo, hi)), float(max(lo, hi))
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"dimension_range must contain numeric tuples of (min, max). "
                    f"Error at dimension {i}: got ({lo!r}, {hi!r})"
                ) from e
            # If user gave a zero-span range (lo == hi), expand by 0.5 * bin_size
            if np.isclose(lo_f, hi_f):
                lo_f -= 0.5 * size
                hi_f += 0.5 * size
            ranges.append((lo_f, hi_f))
    else:
        # Infer from `samples`
        if samples is None:
            raise ValueError("Cannot infer ranges without samples")
        ranges = []
        for dim in range(n_dims):
            dim_vals = samples[:, dim]
            lo_f, hi_f = float(np.nanmin(dim_vals)), float(np.nanmax(dim_vals))
            if np.isclose(lo_f, hi_f):
                # If all data is constant, expand by half a bin
                lo_f -= 0.5 * bin_sizes[dim]
                hi_f += 0.5 * bin_sizes[dim]
            ranges.append((lo_f, hi_f))

    return ranges


def _build_grid_structure(
    samples: NDArray[np.float64] | None,
    n_dims: int,
    bin_sizes: NDArray[np.float64],
    ranges: list[tuple[float, float]],
    add_boundary_bins: bool,
) -> tuple[
    tuple[NDArray[np.float64], ...],
    NDArray[np.float64],
    tuple[int, ...],
]:
    """Build the grid structure (edges, centers, shape).

    Parameters
    ----------
    samples : NDArray[np.float64] or None
        Cleaned data samples (NaNs already removed).
    n_dims : int
        Number of dimensions.
    bin_sizes : NDArray[np.float64], shape (n_dims,)
        Bin sizes for each dimension.
    ranges : list of (float, float) tuples
        Dimension ranges for each dimension.
    add_boundary_bins : bool
        Whether to add boundary bins.

    Returns
    -------
    edges_tuple : tuple of ndarrays
        Each element is a 1D array of bin-edge coordinates for that dimension.
    bin_centers : NDArray[np.float64], shape (n_total_bins, n_dims)
        Cartesian product of centers of each bin (flattened).
    centers_shape : tuple of ints
        Number of bins along each dimension.

    """
    # 4) Compute number of bins in each dimension
    data_for_bins = samples if samples is not None else np.zeros((1, n_dims))
    # Convert bin_sizes to list to match get_n_bins signature
    bin_sizes_list: list[float] = bin_sizes.tolist()
    n_bins = get_n_bins(data_for_bins, bin_sizes_list, ranges)  # ensures at least 1

    # 5) Generate core edges via np.histogramdd
    #    Use data_for_bins which is either samples or a dummy point at the center
    _, core_edges = np.histogramdd(data_for_bins, bins=n_bins, range=ranges)

    # 6) Optionally add boundary bins by extending each edge array
    final_edges = []
    for edges_dim, size in zip(core_edges, bin_sizes, strict=False):
        diff = np.diff(edges_dim)
        step = size if diff.size == 0 else diff[0]
        if add_boundary_bins:
            extended = np.concatenate(
                ([edges_dim[0] - step], edges_dim, [edges_dim[-1] + step]),
            )
        else:
            extended = edges_dim
        final_edges.append(extended.astype(float))

    edges_tuple = tuple(final_edges)

    # 7) Compute centers for each dimension
    centers_per_dim = [get_centers(e) for e in final_edges]
    centers_shape = tuple(len(c) for c in centers_per_dim)

    # 8) Build the full Cartesian product of centers
    mesh = np.meshgrid(*centers_per_dim, indexing="ij")
    bin_centers = np.stack([m.ravel() for m in mesh], axis=-1)

    return edges_tuple, bin_centers, centers_shape


def _create_regular_grid(
    positions: NDArray[np.float64] | None = None,
    bin_size: float | Sequence[float] = 2.0,
    dimension_range: Sequence[tuple[float, float]] | None = None,
    add_boundary_bins: bool = False,
) -> tuple[
    tuple[NDArray[np.float64], ...],  # edges_tuple
    NDArray[np.float64],  # bin_centers
    tuple[int, ...],  # centers_shape
]:
    """Define bin edges and centers for a regular N-D Cartesian grid.

    Parameters
    ----------
    positions : ndarray of shape (n_samples, n_dims), optional
        Used to infer dimension ranges if `dimension_range` is None.
        NaNs are ignored. If None, `dimension_range` must be provided.
    bin_size : float or sequence of floats, default=2.0
        If float, same bin size along every dimension. If sequence, must match `n_dims`.
    dimension_range : sequence of (min, max) tuples, length `n_dims`, optional
        Explicit bounding box for each dimension, used if `positions` is None.
    add_boundary_bins : bool, default=False
        If True, extends each axis by one extra bin on both ends.

    Returns
    -------
    edges_tuple : tuple of ndarrays
        Each element is a 1D array of bin-edge coordinates for that dimension,
        length = n_bins_dim + 1.
    bin_centers : ndarray, shape (∏(n_bins_dim), n_dims)
        Cartesian product of centers of each bin (flattened).
    centers_shape : tuple of ints
        Number of bins along each dimension, e.g. (n_x, n_y, n_z).

    Raises
    ------
    ValueError
        - If both `positions` and `dimension_range` are None.
        - If `positions` is provided but not a 2D array.
        - If `dimension_range` length ≠ inferred `n_dims`.
        - If `bin_size` sequence length ≠ `n_dims`, or any `bin_size` ≤ 0.

    """
    # Validate inputs and prepare normalized parameters
    samples, n_dims, bin_sizes = _validate_and_prepare_inputs(
        positions, bin_size, dimension_range
    )

    # Compute dimension ranges from explicit specification or data
    ranges = _compute_dimension_ranges(samples, dimension_range, n_dims, bin_sizes)

    # Build the grid structure (edges, centers, shape)
    return _build_grid_structure(samples, n_dims, bin_sizes, ranges, add_boundary_bins)


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
        # This case should ideally be caught earlier or raise a more specific error.
        # For now, assume dimensions match if we reach here.
        # If not, returning all -1s is a safe fallback.
        warnings.warn(
            "Dimensionality mismatch between points, grid_edges, or grid_shape.",
            RuntimeWarning,
        )
        return output_indices

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
