"""Utility functions for creating and managing regular N-dimensional grid layouts.

This module provides a collection of helper functions used primarily by
`RegularGridLayout` and other grid-based layout engines within the
`neurospatial` package. These functions handle tasks such as:

- Defining the structure (bin edges, bin centers, shape) of a regular N-D grid
  based on data samples or specified dimension ranges (`_create_regular_grid`).
- Inferring which bins within this grid are "active" based on the density of
  provided data samples, often involving morphological operations to refine
  the active area (`_infer_active_bins_from_regular_grid`).
- Pairing neighboring active grid bins, allowing for orthogonal and
  diagonal connections (`regular_grid_pairs`).
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from neurospatial._exceptions import _format_error
from neurospatial.geometry import grid_cell_centers
from neurospatial.layout.helpers.neighbor_pairs import lattice_neighbor_pairs
from neurospatial.layout.helpers.utils import get_n_bins
from neurospatial.layout.validation import (
    validate_cell_size,
    validate_count_threshold,
    validate_dimension_ranges,
    validate_uniform_edges,
)

# Hard ceiling on the full (dense) grid's bin-center array, allocated in
# ``_build_grid_structure`` by ``grid_cell_centers`` BEFORE any active-bin
# filtering. The Cartesian product of the per-axis bin counts can be
# astronomically large even when every axis is modest -- e.g. a transposed
# ``(2, N)`` trajectory read as N dimensions -- so a per-axis check (as in
# ``get_n_bins``) is not enough; the product must be preflighted or the
# allocation OOMs. 8 GiB is far above any realistic dense spatial grid (whose
# connectivity graph would itself be infeasible well below this) while still
# catching the catastrophic transpose.
_MAX_GRID_CENTERS_BYTES = 8 * 1024**3


def regular_grid_pairs(
    active_mask: NDArray[np.bool_], *, connect_diagonal: bool = False
) -> NDArray[np.intp]:
    """Pair active cells of an N-D grid with their adjacent active cells.

    Bins are the active cells numbered in row-major order. Each bin pairs
    with its higher-numbered active neighbors, visited in a fixed step order:
    along each axis in turn when only face neighbors connect, otherwise every
    step in ``{-1, 0, 1}^n_dims`` in lexicographic order.

    Parameters
    ----------
    active_mask : ndarray of bool, shape (n_0, ..., n_{d-1})
        Active cells of the full grid.
    connect_diagonal : bool, default=False
        Also pair cells that share only an edge or a corner.

    Returns
    -------
    ndarray of intp, shape (n_pairs, 2)
        Pairs ``(lower, higher)`` of bin IDs in authoritative order.

    Examples
    --------
    >>> regular_grid_pairs(np.ones((2, 2), dtype=bool)).tolist()
    [[0, 2], [0, 1], [1, 3], [2, 3]]
    >>> len(regular_grid_pairs(np.ones((2, 2), dtype=bool), connect_diagonal=True))
    6
    """
    n_dims = active_mask.ndim
    if connect_diagonal:
        steps = [
            step for step in itertools.product([-1, 0, 1], repeat=n_dims) if any(step)
        ]
    else:
        steps = [
            tuple(value if axis == dim else 0 for axis in range(n_dims))
            for dim in range(n_dims)
            for value in (-1, 1)
        ]
    # A step whose first nonzero component is negative reaches a lower
    # row-major index, so it never proposes a higher-numbered bin.
    forward = [step for step in steps if next(v for v in step if v) > 0]
    offsets = np.array(forward, dtype=np.intp).reshape(len(forward), 1, n_dims)
    return lattice_neighbor_pairs(active_mask, offsets)


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
    Optional morphological operations can then refine the active area. They
    run only for 2-D and higher grids, in the order closing, hole filling,
    dilation; a 1-D grid keeps exactly the visited bins.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        N-dimensional data samples (e.g., positions). Rows with a NaN or
        infinite coordinate are ignored.
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
    bin_count_threshold = validate_count_threshold(bin_count_threshold)
    pos_clean = positions[np.all(np.isfinite(positions), axis=1)]

    if pos_clean.shape[0] == 0:
        grid_shape = tuple(len(edge_array) - 1 for edge_array in edges)
        warnings.warn(
            "infer_active_bins is True, but positions has no row without NaN. "
            "Returning all bins inactive (no data to infer from).",
            UserWarning,
            stacklevel=2,
        )
        return np.zeros(grid_shape, dtype=bool)

    bin_counts, _ = np.histogramdd(pos_clean, bins=edges)
    active_mask = (bin_counts > 0) & (bin_counts >= bin_count_threshold)

    n_dims = positions.shape[1]
    if n_dims > 1:
        # The scipy structuring element with ``connectivity=2``: in 2D the
        # full 8-neighbor (3x3) element, matching the graph's diagonal
        # connectivity. In 3D it is the 18-neighbor element (faces and edges,
        # no corners), while the graph built with ``connect_diagonal=True``
        # links all 26 neighbors, so closing and dilation of the mask are
        # slightly tighter than the graph's connectivity.
        structure = ndimage.generate_binary_structure(n_dims, connectivity=2)

        if close_gaps:
            # Closing (dilation then erosion) closes small gaps. The erosion
            # treats cells past the array as inactive, which would erode
            # visited bins on the grid border; pad by the element's radius so
            # closing only adds bins.
            padded = np.pad(active_mask, 1)
            closed = ndimage.binary_closing(padded, structure=structure)
            active_mask = closed[(slice(1, -1),) * n_dims]

        if fill_holes:
            # Fill larger holes enclosed by occupied bins
            active_mask = ndimage.binary_fill_holes(active_mask, structure=structure)

        if dilate:
            # Expand the occupied area
            active_mask = ndimage.binary_dilation(active_mask, structure=structure)

    if boundary_exists:
        # Clear the outermost cell at both ends of every axis.
        interior = np.zeros_like(active_mask, dtype=bool)
        interior[(slice(1, -1),) * active_mask.ndim] = True
        active_mask = active_mask & interior

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
        The finite rows of positions when they set the ranges (no
        dimension_range), else None.
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
    if positions is None:
        if dimension_range is None:
            raise ValueError(
                "Either `positions` or `dimension_range` must be provided."
            )
        n_dims = len(dimension_range)
        return None, n_dims, validate_cell_size(bin_size, n_dims=n_dims)

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
    finite = None
    if dimension_range is None:
        finite = samples[np.isfinite(samples).all(axis=1)]
        if finite.size == 0:
            raise ValueError(
                "`positions` has no valid points and no `dimension_range` given.",
            )
    return finite, n_dims, validate_cell_size(bin_size, n_dims=n_dims)


def _compute_dimension_ranges(
    samples: NDArray[np.float64] | None,
    dimension_range: Sequence[tuple[float, float]] | None,
    n_dims: int,
) -> list[tuple[float, float]]:
    """Compute dimension ranges from explicit specification or data samples.

    Parameters
    ----------
    samples : NDArray[np.float64] or None
        Cleaned data samples (rows that are not finite removed).
    dimension_range : sequence of (min, max) tuples or None
        Explicit dimension ranges.
    n_dims : int
        Number of dimensions.

    Returns
    -------
    ranges : list of (float, float) tuples
        Computed dimension ranges for each dimension; a zero-span range is
        kept as is (widening to whole cells gives it one cell).

    Raises
    ------
    ValueError
        If dimension_range length doesn't match n_dims, a range is not finite
        or has its minimum above its maximum, or ranges cannot be inferred.
    TypeError
        If dimension_range contains non-numeric values.

    """
    if dimension_range is not None:
        ranges = validate_dimension_ranges(dimension_range, n_dims=n_dims)
    else:
        # Infer from `samples`
        if samples is None:
            raise ValueError("Cannot infer ranges without samples")
        ranges = []
        for dim in range(n_dims):
            dim_vals = samples[:, dim]
            lo_f, hi_f = float(np.nanmin(dim_vals)), float(np.nanmax(dim_vals))
            ranges.append((lo_f, hi_f))

    return ranges


# A range within this fraction of a cell of a whole number of cells is
# taken as whole: float error in extent / bin_size must not add a cell.
_WHOLE_CELL_TOLERANCE = 1e-9


def _whole_cells(lo: float, hi: float, size: float, count: int) -> int:
    """Return the fewest cells of width ``size`` from ``lo`` that reach ``hi``.

    ``count`` is ``ceil((hi - lo) / size)``; one cell fewer is enough when
    it falls short of ``hi`` only by float error.
    """
    if count > 1 and lo + (count - 1) * size >= hi - _WHOLE_CELL_TOLERANCE * size:
        return count - 1
    return count


def _whole_cell_ranges(
    ranges: Sequence[tuple[float, float]], bin_sizes: Sequence[float]
) -> list[tuple[float, float]]:
    """Widen each range evenly on both sides to a whole number of cells.

    Cells keep exactly the requested width, and every cell center stays
    inside the original range.

    Parameters
    ----------
    ranges : sequence of (float, float)
        ``(low, high)`` per axis.
    bin_sizes : sequence of float
        Cell width per axis, in the units of ``ranges``.

    Returns
    -------
    list of (float, float)
        Ranges whose span is a whole number of cells.
    """
    widened = []
    for (lo, hi), size in zip(ranges, bin_sizes, strict=True):
        count = _whole_cells(lo, hi, size, max(int(np.ceil((hi - lo) / size)), 1))
        pad = max(count * size - (hi - lo), 0.0) / 2.0
        widened.append((lo - pad, hi + pad))
    return widened


def _build_grid_structure(
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
    # With ranges given, get_n_bins reads no positions.
    n_bins = get_n_bins(
        np.zeros((1, n_dims)), bin_size=bin_sizes.tolist(), dimension_range=ranges
    )

    # Preflight the full grid size before allocating anything. get_n_bins
    # checks each axis against the int64 limit, but the bin centers are the
    # Cartesian product of the per-axis counts, which can be astronomically
    # large even when every axis is modest. Python ints keep the product
    # itself from overflowing.
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

    # Lay cells of exactly bin_size across each (whole-cell) range.
    final_edges = []
    for (lo, hi), size, count in zip(ranges, bin_sizes, n_bins, strict=True):
        count = _whole_cells(lo, hi, size, int(count))
        edges_dim = lo + np.arange(count + 1) * size
        # Within float error of the high end, end on it exactly, so a
        # position there stays inside the grid.
        edges_dim[-1] = max(edges_dim[-1], hi)
        if add_boundary_bins:
            edges_dim = np.concatenate(
                ([edges_dim[0] - size], edges_dim, [edges_dim[-1] + size]),
            )
        final_edges.append(edges_dim.astype(float))

    # The same check as given edges: cell measures use each axis's first
    # width, which rounding far from the origin can make wrong.
    edges_tuple = validate_uniform_edges(final_edges, label="axis {axis} of the grid")
    centers_shape = tuple(len(edges) - 1 for edges in final_edges)
    return edges_tuple, grid_cell_centers(edges_tuple), centers_shape


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
        Rows with a NaN or infinite coordinate are ignored. If None,
        `dimension_range` must be provided.
    bin_size : float or sequence of floats, default=2.0
        If float, same bin size along every dimension. If sequence, must match `n_dims`.
    dimension_range : sequence of (min, max) tuples, length `n_dims`, optional
        Explicit bounding box for each dimension. When given, it sets the
        grid's extent and `positions` is not used for it; otherwise the
        extent of `positions` is used. A range that is not a whole number of
        cells is widened evenly past both ends.
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
    ranges = _compute_dimension_ranges(samples, dimension_range, n_dims)
    # Cells are exactly bin_size wide; a range that is not a whole number of
    # cells is widened evenly past both ends, so every cell center stays
    # inside the range.
    ranges = _whole_cell_ranges(ranges, bin_sizes.tolist())

    # Build the grid structure (edges, centers, shape)
    return _build_grid_structure(n_dims, bin_sizes, ranges, add_boundary_bins)
