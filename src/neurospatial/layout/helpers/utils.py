"""Helpers shared by the layout engines.

Bin counts and ranges for regular grids, the grid of active-bin data and the
grid memory check.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._stacklevel import _external_stacklevel
from neurospatial.geometry import get_centers as get_centers
from neurospatial.layout.validation import (
    validate_cell_size,
    validate_dimension_ranges,
)


def get_n_bins(
    positions: NDArray[np.float64],
    *,
    bin_size: float | Sequence[float],
    dimension_range: Sequence[tuple[float, float]] | None = None,
) -> NDArray[np.int_]:
    """Calculate the number of bins needed for each dimension of a dataset.

    Each dimension gets ``ceil(extent / bin_size)`` bins, where the extent is
    the span of `dimension_range` or, without one, of the
    finite rows of `positions`.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        N-dimensional position samples. Used to determine the data extent if
        `dimension_range` is not provided, and ignored when it is. Rows with a
        NaN or infinite coordinate are ignored.
    bin_size : Union[float, Sequence[float]]
        Bin width in the units of the positions. A single number applies to
        every dimension; a sequence gives one width per dimension and must
        have one entry per dimension (a length-1 sequence only fits 1-D
        data). Must be positive and finite.
    dimension_range : Optional[Sequence[Tuple[float, float]]], optional
        Explicit range `[(min_d0, max_d0), ..., (min_dN-1, max_dN-1)]` for
        each dimension. If None (default), the range is calculated from
        the min/max of `positions`.

    Returns
    -------
    NDArray[np.int64], shape (n_dims,)
        Number of bins in each dimension; a zero extent gives 1 bin.

    Raises
    ------
    TypeError
        If `bin_size` is not numeric, or a `dimension_range` pair is not two
        numbers.
    ValueError
        If `bin_size` is not positive and finite, or has the wrong number of
        entries; if a `dimension_range` pair is not finite or has its minimum
        above its maximum; if `positions` has no finite row; or if a bin count
        would exceed the int64 limit.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.layout import get_n_bins
    >>> positions = np.array([[0.0, 0.0], [10.0, 5.0]])
    >>> get_n_bins(positions, bin_size=2.0)
    array([5, 3])
    >>> get_n_bins(positions, bin_size=[2.0, 1.0], dimension_range=[(0, 20), (0, 5)])
    array([10,  5])
    """
    if dimension_range is not None:
        ranges = validate_dimension_ranges(dimension_range, name="dimension_range")
        extent = np.array([hi - lo for lo, hi in ranges])
    else:
        samples = np.asarray(positions, dtype=float)
        samples = samples[np.all(np.isfinite(samples), axis=1)]
        if len(samples) == 0:
            raise ValueError(
                _format_error(
                    "positions has no row without a NaN or infinite coordinate.",
                    why="Why: the extent of the finite positions sets the bin count.",
                    fix="drop or interpolate tracking dropouts, or pass "
                    "dimension_range",
                )
            )
        extent = samples.max(axis=0) - samples.min(axis=0)

    bin_size_arr = validate_cell_size(bin_size, n_dims=len(extent))

    # Calculate number of bins, ensuring at least 1 bin even if extent is 0.
    # Compute in float then cast straight to int64 (never int32) so a large
    # extent / small bin_size cannot silently overflow.
    n_bins_float = np.ceil(np.asarray(extent, dtype=np.float64) / bin_size_arr)
    if np.any(n_bins_float > np.iinfo(np.int64).max):
        raise ValueError(
            f"Requested binning overflows: extent / bin_size implies "
            f"{n_bins_float.max():.3g} bins, exceeding the int64 limit. "
            f"Increase bin_size or reduce the dimension range."
        )
    n_bins = n_bins_float.astype(np.int64)
    n_bins[n_bins == 0] = 1  # Handle zero-extent case
    return np.asarray(n_bins, dtype=np.int64)


def _infer_dimension_ranges_from_samples(
    positions: NDArray[np.float64],
    buffer_around_data: float | Sequence[float] = 0.0,
) -> Sequence[tuple[float, float]]:
    """Infer min/max range for each dimension from position samples, with a buffer.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position points from which to infer ranges. Rows with a NaN or
        infinite coordinate are ignored.
    buffer_around_data : Union[float, Sequence[float]], default=0.0
        Buffer to add to the min and max of the inferred range in each
        dimension. If a float, applied to all dimensions. If a sequence,
        specifies buffer per dimension. Useful for ensuring extent if data
        is collinear or a margin around the data is desired.

    Returns
    -------
    Sequence[Tuple[float, float]]
        Inferred ranges: `[(min_d0, max_d0), ..., (min_dN-1, max_dN-1)]`.

    Raises
    ------
    ValueError
        If every row of `positions` has a NaN, `positions` is empty or not
        2D, or `buffer_around_data` dimensionality mismatches `positions`.

    """
    if positions.ndim != 2:
        raise ValueError(
            f"positions must be a 2D array, shape is {positions.shape}",
        )
    n_dimensions = positions.shape[1]

    clean_samples = positions[np.all(np.isfinite(positions), axis=1)]
    if clean_samples.shape[0] == 0:
        raise ValueError(
            _format_error(
                f"No position has a finite value: all {len(positions)} rows contain "
                "NaN or infinite coordinates, or the array is empty; check for "
                "tracking dropouts.",
                why="Why: finite positions are required to infer the environment's extent.",
                fix="remove tracking-dropout rows or interpolate them before calling Environment.from_samples(positions, bin_size=2.0)",
            )
        )

    min_vals = np.min(clean_samples, axis=0)
    max_vals = np.max(clean_samples, axis=0)

    buffer_array = np.asarray(buffer_around_data, dtype=float)
    if buffer_array.ndim == 0:
        buffer_values = np.full(n_dimensions, float(buffer_array))
    elif buffer_array.shape == (n_dimensions,):
        buffer_values = buffer_array
    else:
        raise ValueError(
            f"buffer_around_data sequence length ({buffer_array.size}) "
            f"must match number of dimensions ({n_dimensions}).",
        )

    inferred_ranges = []
    for dim_idx in range(n_dimensions):
        d_min, d_max = min_vals[dim_idx], max_vals[dim_idx]
        buffer = buffer_values[dim_idx]

        # Exact comparisons: a tolerance relative to the coordinates would
        # read a short extent far from the origin as a point.
        if d_min == d_max and buffer == 0.0:
            warnings.warn(
                f"Dimension {dim_idx} has zero extent and no buffer specified. "
                "Extending it by 0.5 on each side of the point.",
                UserWarning,
                stacklevel=2,
            )
            buffer = 0.5
        d_min -= buffer
        d_max += buffer

        inferred_ranges.append((d_min, d_max))

    return tuple(inferred_ranges)


def map_active_data_to_grid(
    grid_shape: tuple[int, ...],
    active_mask: NDArray[np.bool_],
    active_bin_data: NDArray[np.float64],
    fill_value: float = np.nan,
) -> NDArray[np.float64]:
    """Map a 1D array of data corresponding to active bins onto a full N-D grid.

    This is useful for visualizing data (e.g., place fields, posterior
    probabilities) on grid-based layouts using functions like `pcolormesh`.

    Parameters
    ----------
    grid_shape : tuple of int
        Shape of the full grid, ``(n_0, ..., n_{d-1})``.
    active_mask : NDArray[np.bool_], shape grid_shape
        True for the grid's active cells.
    active_bin_data : NDArray, shape (n_active_bins,)
        One value per active bin, in active-bin order: the order of
        ``active_mask``'s True cells in row-major (C) order, which is the
        order of the layout's ``bin_centers``.
    fill_value : float, optional
        The value to use for bins in the full grid that are not active,
        by default np.nan.

    Returns
    -------
    NDArray, shape grid_shape
        Active cells hold `active_bin_data`, others `fill_value`. The dtype
        is the common type of `active_bin_data` and `fill_value`.

    Raises
    ------
    ValueError
        If `grid_shape` or `active_mask` is None, `active_bin_data` is not a
        1D NumPy array, or its length is not the number of active bins.

    """
    if grid_shape is None or active_mask is None:
        raise ValueError(
            "This method is applicable only to grid-based environments "
            "that have 'grid_shape' and 'active_mask' attributes.",
        )
    if not isinstance(active_bin_data, np.ndarray) or active_bin_data.ndim != 1:
        raise ValueError("active_bin_data must be a 1D NumPy array.")

    n_bins = active_mask.sum()
    if active_bin_data.shape[0] != n_bins:
        raise ValueError(
            f"Length of active_bin_data ({active_bin_data.shape[0]}) "
            f"must match the number of active bins ({n_bins}).",
        )

    # Create an array for the full grid, filled with the fill_value
    # Ensure dtype compatibility, e.g., promote fill_value to active_bin_data.dtype
    # or choose a suitable default like float if active_bin_data can be int.
    dtype = np.result_type(active_bin_data.dtype, type(fill_value))
    full_grid_data = np.full(grid_shape, fill_value, dtype=dtype)

    # Place the active data into the grid using the N-D active_mask
    full_grid_data[active_mask] = active_bin_data
    return full_grid_data


def estimate_grid_memory(grid_shape: tuple[int, ...]) -> float:
    """Estimate memory usage in MB for a grid of given shape.

    A rough, approximate figure for an Environment whose every grid cell is
    a bin: the bin-center array plus the nodes and edges of its NetworkX
    connectivity graph. It counts the full grid, before active bins are
    chosen, and is meant only to flag grids too large to build.

    Parameters
    ----------
    grid_shape : tuple[int, ...]
        Shape of the N-D grid (e.g., (100, 100) for 2D grid); its length is
        the number of dimensions.

    Returns
    -------
    float
        Estimated memory usage in megabytes.

    Notes
    -----
    Memory components estimated:

    - **bin_centers**: n_bins * n_dims * 8 bytes (float64)
    - **NetworkX graph nodes**: ~500 bytes per node (approximate overhead)
    - **NetworkX graph edges**: ~300 bytes per edge (approximate overhead)

    The per-node and per-edge sizes approximate NetworkX's dict-based
    storage of the node attributes (pos, source_grid_flat_index,
    original_grid_nd_index) and edge attributes (distance, vector, edge_id).

    The edge count assumes 2*n_dims neighbors per cell (4 in 2D, 6 in 3D),
    as with face-only connectivity. Grid layouts connect diagonal neighbors
    by default (8 in 2D, 26 in 3D), so the edge term usually undercounts.

    Examples
    --------
    >>> from neurospatial.layout.helpers.utils import estimate_grid_memory
    >>> # Estimate memory for 100×100 2D grid
    >>> mem = estimate_grid_memory((100, 100))
    >>> mem > 8.0 and mem < 20.0  # Approximately 8-20 MB
    True

    >>> # Estimate memory for 100×100×100 3D grid
    >>> mem = estimate_grid_memory((100, 100, 100))
    >>> mem > 1000  # At least 1000 MB
    True

    See Also
    --------
    check_grid_size_safety : Validate grid size and warn if memory usage is high
    """
    n_bins = int(np.prod(grid_shape))
    n_dims = len(grid_shape)

    # bin_centers: n_bins * n_dims * 8 bytes (float64)
    bin_centers_mb = (n_bins * n_dims * 8) / (1024**2)

    # Approximate NetworkX overhead: ~500 bytes per node, ~300 per edge.
    nodes_mb = (n_bins * 500) / (1024**2)

    # Face neighbors only: 2*n_dims per cell.
    avg_edges_per_node = 2 * n_dims
    n_edges = n_bins * avg_edges_per_node / 2  # Divide by 2 (edges counted twice)
    edges_mb = (n_edges * 300) / (1024**2)

    total_mb = bin_centers_mb + nodes_mb + edges_mb
    return total_mb


def check_grid_size_safety(
    grid_shape: tuple[int, ...],
    *,
    warn_threshold_mb: float = 100.0,
) -> None:
    """Check if grid size is large and warn if it exceeds threshold.

    This function provides a safety check to warn users when creating
    very large grids that may have significant memory requirements.
    The grid creation will still proceed, but users are informed about
    the estimated memory usage.

    Parameters
    ----------
    grid_shape : tuple[int, ...]
        Shape of the N-D grid.
    warn_threshold_mb : float, default 100.0
        Memory threshold (MB) for emitting a warning. Set to inf to disable warnings.

    Warnings
    --------
    UserWarning
        If estimated memory exceeds warn_threshold_mb.

    Notes
    -----
    The estimate (see `estimate_grid_memory`) counts every cell of the full
    grid, before active bins are chosen, so only a coarser or smaller grid
    lowers it: increase ``bin_size``, narrow ``dimension_ranges``, or check
    that ``positions`` has shape ``(n_samples, n_dims)`` and that
    ``bin_size`` is in its units.

    The environment factories do not take ``warn_threshold_mb``. To silence
    the warning for a grid you know is large, filter it::

        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Creating large grid")
            env = Environment.from_samples(positions, bin_size=1.0)

    Examples
    --------
    >>> from neurospatial.layout.helpers.utils import check_grid_size_safety
    >>> # Small grid - no warning
    >>> check_grid_size_safety((50, 50))

    >>> # Large grid - warning (but creation proceeds)
    >>> check_grid_size_safety((500, 500))  # doctest: +SKIP
    UserWarning: Creating large grid with shape (500, 500) (250,000 bins). Estimated memory usage: 266.1 MB.
    Why: allocating this grid may consume substantial memory.
    Fix: increase bin_size; this grid has 250,000 bins. For example, use bin_size=2.0 in the same units as positions, and check that positions has shape (n_samples, n_dims)

    See Also
    --------
    estimate_grid_memory : Estimate memory requirements for a grid
    """
    n_bins = int(np.prod(grid_shape))
    estimated_mb = estimate_grid_memory(grid_shape)

    if estimated_mb > warn_threshold_mb:
        warnings.warn(
            _format_error(
                f"Creating large grid with shape {grid_shape} ({n_bins:,} bins). Estimated memory usage: {estimated_mb:.1f} MB.",
                why="Why: allocating this grid may consume substantial memory.",
                fix=f"increase bin_size; this grid has {n_bins:,} bins. For example, use bin_size=2.0 in the same units as positions, and check that positions has shape (n_samples, n_dims)",
            ),
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
