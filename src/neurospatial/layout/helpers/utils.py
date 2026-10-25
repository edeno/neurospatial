"""Helpers shared by the layout engines.

Bin counts and ranges for regular grids, the grid of active-bin data, the
graph plot used by hexagonal layouts, and the grid memory check.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial.geometry import get_centers as get_centers
from neurospatial.layout.validation import validate_cell_size

if TYPE_CHECKING:
    import matplotlib.axes


def get_n_bins(
    positions: NDArray[np.float64],
    *,
    bin_size: float | Sequence[float],
    dimension_range: Sequence[tuple[float, float]] | None = None,
) -> NDArray[np.int_]:
    """Calculate the number of bins needed for each dimension of a dataset.

    The number of bins is determined based on the extent of the data (or a
    specified `dimension_range`) and the desired `bin_size`.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        N-dimensional position samples. Used to determine the data extent if
        `dimension_range` is not provided. NaNs are ignored for range calculation.
    bin_size : Union[float, Sequence[float]]
        The desired size of the bins. If a float, this size is applied to
        all dimensions. If a sequence, it specifies the bin size for each
        dimension and its length must match `n_dims`. Must be positive.
    dimension_range : Optional[Sequence[Tuple[float, float]]], optional
        Explicit range `[(min_d0, max_d0), ..., (min_dN-1, max_dN-1)]` for
        each dimension. If None (default), the range is calculated from
        the min/max of `positions`.

    Returns
    -------
    NDArray[np.int_], shape (n_dims,)
        An array containing the calculated number of bins required for each
        dimension. Each value is at least 1.

    Raises
    ------
    ValueError
        If `bin_size` is not positive or if its length (if a sequence)
        does not match the number of dimensions.
        If `dimension_range` (if provided) does not have two values (min, max)
        per dimension.

    """
    if dimension_range is not None:
        # Ensure dimension_range is numpy array for consistent processing
        pr = np.asarray(dimension_range)
        if pr.shape[1] != 2:
            raise ValueError("dimension_range must be sequence of (min, max) pairs.")
        extent = np.diff(pr, axis=1).squeeze(axis=1)
    else:
        # Ignore NaNs when calculating range from data
        extent = np.nanmax(positions, axis=0) - np.nanmin(positions, axis=0)

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
        Position points from which to infer ranges. NaNs are ignored.
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
        If `positions` are all NaN, empty after NaN removal, not 2D,
        or if `buffer_around_data` dimensionality mismatches `positions`.
    TypeError
        If `buffer_around_data` is not a float or sequence of floats.

    """
    if positions.ndim != 2:
        raise ValueError(
            f"positions must be a 2D array, shape is {positions.shape}",
        )
    n_dimensions = positions.shape[1]

    clean_samples = positions[~np.any(np.isnan(positions), axis=1)]
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

        # Ensure range has some extent if data is point-like or buffer is zero
        if np.isclose(d_min, d_max):
            if np.isclose(buffer, 0.0):  # If buffer is also zero, data is a point
                warnings.warn(
                    f"Dimension {dim_idx} has zero extent and no buffer specified. "
                    "Using a default small buffer of 1.0 around the point.",
                    UserWarning,
                    stacklevel=2,
                )
                d_min -= 0.5  # Default small extent
                d_max += 0.5
            else:  # Buffer will create the extent
                d_min -= buffer
                d_max += buffer
        else:  # Data already has extent
            d_min -= buffer
            d_max += buffer

        inferred_ranges.append((d_min, d_max))

    return tuple(inferred_ranges)


def _generic_graph_plot(
    graph: nx.Graph,
    name: str,
    ax: matplotlib.axes.Axes | None = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Draw a graph whose nodes have 2-D ``pos`` coordinates.

    Parameters
    ----------
    graph : nx.Graph
        The graph to plot. Nodes must have a 2-D ``pos`` attribute.
    name : str
        A name for the graph, used in the plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None (default), a new figure and axes are created.
    **kwargs : Any
        - ``figsize`` (tuple of float): Size of the figure if created.
        - ``node_kwargs`` (dict): Keyword arguments for ``nx.draw_networkx_nodes``.
        - ``edge_kwargs`` (dict): Keyword arguments for ``nx.draw_networkx_edges``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes on which the graph was plotted.

    Raises
    ------
    ValueError
        If `graph` is empty or None, or its nodes have no ``pos``.

    """
    import matplotlib.pyplot as plt

    if graph is None or graph.number_of_nodes() == 0:
        raise ValueError("Graph is empty or None. Cannot plot an empty graph.")

    node_positions = nx.get_node_attributes(graph, "pos")
    if not node_positions:  # No 'pos' attributes
        raise ValueError("Nodes in graph are missing 'pos' attribute for plotting.")

    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 7)))
    node_kwargs = {"node_size": 20, "ax": ax, **kwargs.get("node_kwargs", {})}
    nx.draw_networkx_nodes(graph, pos=node_positions, **node_kwargs)
    edge_kwargs = {
        "alpha": 0.5,
        "edge_color": "gray",
        **kwargs.get("edge_kwargs", {}),
        "ax": ax,
    }
    nx.draw_networkx_edges(graph, pos=node_positions, **edge_kwargs)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{name} Graph")
    ax.set_xlabel("Dim 0")
    ax.set_ylabel("Dim 1")
    return ax


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
    active_bin_data : NDArray[np.float64], shape (n_bins,)
        A 1D array of values, one for each active bin, ordered consistently
        with `self.bin_centers` (i.e., by active bin index 0 to `n_bins - 1`).
    fill_value : float, optional
        The value to use for bins in the full grid that are not active,
        by default np.nan.

    Returns
    -------
    NDArray[np.float64], shape (dim0_size, dim1_size, ...)
        An N-D array with shape `self.grid_shape`. Active bin locations are
        filled with `active_bin_data`, others with `fill_value`.

    Raises
    ------
    ValueError
        If the environment is not grid-based (missing `grid_shape` or
        `active_mask`), or if `active_bin_data` has an incorrect shape
        or incompatible data type.

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

    This function provides a conservative estimate of memory requirements for
    creating an Environment with a grid-based layout. The estimate includes
    bin_centers array, NetworkX graph nodes, and edges.

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

    The graph overhead estimates are conservative approximations based on
    NetworkX's dict-based storage with required attributes (pos,
    source_grid_flat_index, original_grid_nd_index for nodes; distance,
    vector, edge_id for edges).

    Edge count assumes average of 2*n_dims edges per node (e.g., 4 neighbors
    in 2D, 6 in 3D) for grid connectivity.

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

    # NetworkX graph overhead (approximate)
    # Each node: dict with ~5 attributes (~500 bytes conservative estimate)
    # Each edge: dict with ~4 attributes (~300 bytes conservative estimate)
    nodes_mb = (n_bins * 500) / (1024**2)

    # Average edges per node: 2*n_dims for grid connectivity
    # (e.g., 4 in 2D: up, down, left, right; 6 in 3D; etc.)
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
    **Reducing Memory Usage:**

    If you encounter warnings, consider:

    - **Increase bin_size** to reduce grid resolution
    - **Use infer_active_bins=True** to filter out empty bins
    - **Set bin_count_threshold** to exclude sparse bins
    - **Process data in spatial chunks** for very large environments
    - **Use morphological operations** (dilate, fill_holes) to clean up active mask

    **Disabling Warnings:**

    To disable the warning entirely, set warn_threshold_mb to infinity::

        env = Environment.from_samples(
            positions, bin_size=1.0, warn_threshold_mb=float("inf")
        )

    Examples
    --------
    >>> from neurospatial.layout.helpers.utils import check_grid_size_safety
    >>> # Small grid - no warning
    >>> check_grid_size_safety((50, 50))

    >>> # Large grid - warning (but creation proceeds)
    >>> check_grid_size_safety((500, 500))  # doctest: +SKIP
    UserWarning: Creating large grid with shape (500, 500) (250,000 bins).
    Estimated memory usage: 214.8 MB
    Consider increasing bin_size or using infer_active_bins=True.

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
                fix=f"increase bin_size; this grid has {n_bins:,} bins. For example, use bin_size=2.0 in the same units as positions, or infer_active_bins=True",
            ),
            UserWarning,
            stacklevel=4,  # Adjust stacklevel to point to user code
        )
