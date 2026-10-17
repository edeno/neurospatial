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
import math
import warnings
from collections.abc import Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from neurospatial._exceptions import _format_error
from neurospatial.geometry._regular_lookup import (
    _points_to_regular_grid_bin_ind as _points_to_regular_grid_bin_ind,
)
from neurospatial.layout.helpers.utils import get_centers, get_n_bins

# Hard ceiling on the full (dense) grid's bin-center array, allocated in
# ``_build_grid_structure`` by ``np.histogramdd`` / ``np.meshgrid`` BEFORE any
# active-bin filtering. The Cartesian product of the per-axis bin counts can be
# astronomically large even when every axis is modest -- e.g. a transposed
# ``(2, N)`` trajectory read as N dimensions -- so a per-axis check (as in
# ``get_n_bins``) is not enough; the product must be preflighted or the
# allocation OOMs. 8 GiB is far above any realistic dense spatial grid (whose
# connectivity graph would itself be infeasible well below this) while still
# catching the catastrophic transpose. Raised as ``float`` bytes for comparison.
_MAX_GRID_CENTERS_BYTES = 8 * 1024**3


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
    Visited bins with counts at least `bin_count_threshold` are initially active.
    Thresholds 0 and 1 both retain every visited bin; empty bins remain inactive.
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
        considered active (before morphological operations). Equality passes;
        a bin must always contain at least one sample.
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
            stacklevel=2,
        )
        return np.zeros(grid_shape, dtype=bool)

    bin_counts, _ = np.histogramdd(pos_clean, bins=edges)
    active_mask = (bin_counts > 0) & (bin_counts >= bin_count_threshold)

    n_dims = positions.shape[1]
    if n_dims > 1:
        # Morphological operations use the maximal scipy structuring element
        # (``connectivity=n_dims``): in 2D this is the full 8-neighbor (3x3)
        # element, matching the graph's diagonal connectivity exactly. In 3D,
        # however, ``connectivity=2`` is the 18-neighbor element (faces + edges,
        # NO corners), whereas the connectivity GRAPH built with
        # ``connect_diagonal=True`` uses the full 26-neighbor (3x3x3) stencil
        # including corner diagonals. This mismatch is intentional and
        # conservative: the 18-neighbor element gives slightly tighter
        # closing/dilation of the active mask in 3D while the graph still wires
        # all 26 diagonal neighbors for traversal. (Note: scipy clamps
        # ``connectivity`` to at most ``n_dims``, so ``connectivity=2`` is the
        # full element in 2D but the 18-neighbor element in 3D.)
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
            _format_error(
                f"[E1002] bin_size contains NaN (Not a Number) values (got {bin_size}). bin_size must be finite numeric values.",
                fix="pass bin_size=2.0 (in the same units as positions)",
                why="Why: grid widths must be finite and positive in every spatial dimension.",
            )
        )
    if np.any(np.isinf(bin_sizes)):
        raise ValueError(
            _format_error(
                f"[E1002] bin_size contains infinite values (got {bin_size}). bin_size must be finite numeric values.",
                fix="pass bin_size=2.0 (in the same units as positions)",
                why="Why: grid widths must be finite and positive in every spatial dimension.",
            )
        )

    if np.any(bin_sizes <= 0.0):
        raise ValueError(
            _format_error(
                f"[E1002] All elements of `bin_size` must be positive (got {bin_size}).",
                fix="pass bin_size=2.0 (in the same units as positions)",
                why="Why: grid widths must be finite and positive in every spatial dimension.",
            )
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

    # 4a) Preflight the FULL grid size before allocating anything. get_n_bins
    #     validates each AXIS against the int64 limit, but np.histogramdd and
    #     np.meshgrid below allocate the Cartesian PRODUCT of the per-axis
    #     counts, which can be astronomically large even when every axis is
    #     modest (e.g. a transposed (2, N) trajectory read as N dimensions).
    #     Use exact Python-int arithmetic (math.prod on int()s) so the product
    #     itself cannot overflow, then fail loud instead of OOMing.
    total_bins = math.prod(int(b) for b in n_bins)
    centers_bytes = total_bins * n_dims * 8
    if centers_bytes > _MAX_GRID_CENTERS_BYTES:
        # Python refuses str() of an int with more than 4300 digits, so give
        # huge counts as a power of ten from their logarithm.
        def magnitude(count: int) -> str:
            return f"{count:,}" if count < 10**12 else f"~10^{int(math.log10(count))}"

        raise ValueError(
            _format_error(
                f"A regular grid over these positions would have "
                f"{magnitude(total_bins)} bins in {n_dims} dimension(s), and its "
                f"bin centers would need {magnitude(centers_bytes)} bytes, more "
                f"than the {_MAX_GRID_CENTERS_BYTES // 1024**3} GiB limit.",
                why="Why: each column of positions is one spatial dimension, so "
                "a transposed array of shape (n_dims, n_samples) reads as one "
                "dimension per sample; a bin_size far smaller than the arena "
                "has the same effect.",
                fix="pass positions with shape (n_samples, n_dims), for example "
                "positions.T, and check that bin_size is in the units of "
                "positions",
            )
        )

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
