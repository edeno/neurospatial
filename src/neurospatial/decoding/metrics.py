"""Decoding quality metrics for evaluating position decoding accuracy.

This module provides functions for computing error metrics between decoded
and actual positions, measured along the environment's connectivity graph by
default or as straight-line distance.

Functions
---------
decoding_error
    Compute position error for each time bin.
median_decoding_error
    Compute median decoding error (convenience wrapper).
confusion_matrix
    Confusion matrix between decoded and actual bin indices.
decoding_correlation
    Weighted Pearson correlation between decoded and actual positions.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray

from neurospatial.geometry import require_cartesian

if TYPE_CHECKING:
    from neurospatial.environment import Environment


def decoding_error(
    env: Environment,
    decoded_positions: NDArray[np.float64],
    actual_positions: NDArray[np.float64],
    *,
    metric: Literal["euclidean", "geodesic"] = "geodesic",
) -> NDArray[np.float64]:
    """Compute position error for each time bin.

    By default the error is the shortest path between the decoded and actual
    positions along the environment's connectivity graph, so a decode in the
    wrong arm of a maze, or across a wall, is as far off as the animal would
    have to travel. ``metric="euclidean"`` gives the straight-line distance.

    Parameters
    ----------
    env : Environment
        Environment the positions are in. Its connectivity graph gives
        geodesic distances and its ``n_dims`` the number of position columns.
    decoded_positions : NDArray[np.float64], shape (n_time_bins, n_dims)
        Decoded position estimates (e.g., MAP or mean positions).
    actual_positions : NDArray[np.float64], shape (n_time_bins, n_dims)
        Ground truth positions.
    metric : {"geodesic", "euclidean"}, default="geodesic"
        Distance metric to use:

        - "geodesic": Shortest-path distance between the positions' bins
          along the environment's connectivity graph (along the track on a
          linearized track). A position outside every bin, such as a
          posterior mean between two arms, is measured from its nearest bin.
          Bins in disconnected components are ``inf`` apart. On an open grid
          the graph path can exceed the straight line by up to about 8%
          (8-connected neighbors).
        - "euclidean": Straight-line distance between the positions.

    Returns
    -------
    errors : NDArray[np.float64], shape (n_time_bins,)
        Distance between decoded and actual position at each time bin.
        Units match environment units (e.g., cm).

    Raises
    ------
    ValueError
        If the position arrays differ in shape or do not have ``env.n_dims``
        columns, or if ``metric`` is not "euclidean" or "geodesic".
    CartesianOnlyError
        If ``metric="geodesic"`` on an egocentric polar environment.

    Notes
    -----
    NaN values in either input propagate to NaN in output. This allows
    downstream functions (like ``median_decoding_error``) to use
    ``np.nanmedian`` for robust statistics.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding.metrics import decoding_error
    >>> env = Environment.linear_track(
    ...     node_positions=[(0, 0), (50, 0), (50, 50)], bin_size=1.0, track_width=4.0
    ... )

    Along the L-shaped track the end bins are about 99 cm apart, though only
    about 70 cm in a straight line:

    >>> ends = np.array([[0.5, 0.0]]), np.array([[50.0, 49.5]])
    >>> round(float(decoding_error(env, *ends)[0]))
    99
    >>> round(float(decoding_error(env, *ends, metric="euclidean")[0]))
    70

    See Also
    --------
    median_decoding_error : Median error summary statistic.
    DecodingResult.error_against : Error against separately sampled tracking.
    """
    decoded_positions = np.asarray(decoded_positions, dtype=np.float64)
    actual_positions = np.asarray(actual_positions, dtype=np.float64)

    # Validate shapes match
    if decoded_positions.shape != actual_positions.shape:
        raise ValueError(
            f"Shape mismatch: decoded_positions has shape {decoded_positions.shape}, "
            f"but actual_positions has shape {actual_positions.shape}"
        )
    if decoded_positions.ndim != 2 or decoded_positions.shape[1] != env.n_dims:
        raise ValueError(
            f"decoding_error() needs positions with {env.n_dims} columns, one per "
            f"dimension of the environment; got shape {decoded_positions.shape}. "
            "Pass (n_time_bins, n_dims) positions in the environment's coordinates."
        )

    if metric == "euclidean":
        diff = decoded_positions - actual_positions
        return cast("NDArray[np.float64]", np.linalg.norm(diff, axis=1))
    if metric != "geodesic":
        raise ValueError(
            f"Invalid metric '{metric}'. Must be 'euclidean' or 'geodesic'."
        )

    require_cartesian(
        env.coordinate_system,
        "decoding_error(metric='geodesic')",
        why="It locates the decoded and actual positions as allocentric "
        "positions in the environment's bins.",
        fix="decode on the allocentric environment the positions were "
        "recorded in, or use metric='euclidean' on Cartesian positions",
    )
    errors = np.full(len(decoded_positions), np.nan, dtype=np.float64)
    valid = ~(
        np.any(np.isnan(decoded_positions), axis=1)
        | np.any(np.isnan(actual_positions), axis=1)
    )
    if np.any(valid):
        errors[valid] = _graph_distances(
            env,
            _bins_or_nearest(env, decoded_positions[valid]),
            _bins_or_nearest(env, actual_positions[valid]),
        )
    return errors


def _bins_or_nearest(
    env: Environment, positions: NDArray[np.float64]
) -> NDArray[np.intp]:
    """Bin of each finite position, or its nearest bin when outside every bin."""
    bins = np.asarray(env.bin_at(positions), dtype=np.intp)
    outside = bins < 0
    if np.any(outside):
        bins[outside] = env.nearest_bin(positions[outside])
    return bins


# Distances computed per block of source bins: (block, n_bins) float64 rows
# stay near 160 MB however large the environment.
_GRAPH_DISTANCE_BLOCK_VALUES = 20_000_000


def _graph_distances(
    env: Environment, sources: NDArray[np.intp], targets: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Shortest-path length along ``env.connectivity`` for each bin pair.

    Runs Dijkstra only from the distinct source bins, in blocks, instead of
    building the dense ``(n_bins, n_bins)`` all-pairs matrix.

    Parameters
    ----------
    env : Environment
        Environment whose connectivity graph (``distance`` weights) is used.
    sources, targets : NDArray[np.intp], shape (n_pairs,)
        Bin index pairs.

    Returns
    -------
    NDArray[np.float64], shape (n_pairs,)
        Path length per pair; ``inf`` between disconnected components.
    """
    import networkx as nx
    from scipy.sparse.csgraph import dijkstra

    adjacency = nx.to_scipy_sparse_array(
        env.connectivity, nodelist=range(env.n_bins), weight="distance", format="csr"
    )
    unique_sources, source_row = np.unique(sources, return_inverse=True)
    out = np.empty(len(sources), dtype=np.float64)
    block = max(1, _GRAPH_DISTANCE_BLOCK_VALUES // max(env.n_bins, 1))
    for start in range(0, len(unique_sources), block):
        rows = np.arange(start, min(start + block, len(unique_sources)))
        lengths = dijkstra(adjacency, directed=False, indices=unique_sources[rows])
        in_block = (source_row >= rows[0]) & (source_row <= rows[-1])
        out[in_block] = lengths[source_row[in_block] - rows[0], targets[in_block]]
    return out


def median_decoding_error(
    env: Environment,
    decoded_positions: NDArray[np.float64],
    actual_positions: NDArray[np.float64],
    *,
    metric: Literal["euclidean", "geodesic"] = "geodesic",
) -> float:
    """Compute the median decoding error, ignoring NaN.

    Convenience function equivalent to
    ``np.nanmedian(decoding_error(env, decoded, actual, metric=metric))``.

    Parameters
    ----------
    env : Environment
        Environment the positions are in.
    decoded_positions : NDArray[np.float64], shape (n_time_bins, n_dims)
        Decoded position estimates (e.g., MAP or mean positions).
    actual_positions : NDArray[np.float64], shape (n_time_bins, n_dims)
        Ground truth positions.
    metric : {"geodesic", "euclidean"}, default="geodesic"
        Distance metric; see :func:`decoding_error`.

    Returns
    -------
    median_error : float
        Median distance between decoded and actual positions, ignoring
        time bins where either position contains NaN.
        Returns NaN if all time bins contain NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding.metrics import median_decoding_error
    >>> env = Environment.linear_track(
    ...     endpoints=[(0, 0), (10, 0)], bin_size=1.0, track_width=2.0
    ... )
    >>> decoded = np.zeros((5, 2)) + [0.5, 0.0]
    >>> actual = np.column_stack([np.arange(1.5, 6.5), np.zeros(5)])
    >>> median_decoding_error(env, decoded, actual)
    3.0

    See Also
    --------
    decoding_error : Per-time-bin error computation.
    """
    errors = decoding_error(env, decoded_positions, actual_positions, metric=metric)
    return float(np.nanmedian(errors))


def confusion_matrix(
    env: Environment,
    posterior: NDArray[np.float64],
    actual_bins: NDArray[np.int64],
    *,
    method: Literal["map", "expected"] = "map",
) -> NDArray[np.float64]:
    """Confusion matrix between decoded and actual bin indices.

    Computes a confusion matrix summarizing decoding performance across
    all spatial bins. Rows represent actual bins, columns represent decoded bins.

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization (used for n_bins).
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution from decoding.
        Each row should sum to 1.0.
    actual_bins : NDArray[np.int64], shape (n_time_bins,)
        Ground truth bin indices. Values must be in [0, n_bins).
    method : {"map", "expected"}, default="map"
        How to summarize the posterior for each time bin:

        - "map": Use argmax (most likely bin). Returns integer counts.
          Cell (i, j) contains the count of time bins where actual=i and
          decoded=j.
        - "expected": Accumulate full posterior mass. Cell (i, j) contains
          sum of P(decoded=j | actual=i) across all time bins where
          actual=i. Rows sum to the count of actual bin occurrences.

    Returns
    -------
    cm : NDArray[np.float64], shape (n_bins, n_bins)
        Confusion matrix. Rows are actual bins, columns are decoded bins.
        For ``method="map"``, the matrix sums to n_time_bins.
        For ``method="expected"``, each row sums to the count of that bin's
        occurrences in actual_bins.

    Raises
    ------
    ValueError
        If ``method`` is not "map" or "expected".
        If ``actual_bins`` contains values outside [0, n_bins).
        If shapes are inconsistent.

    Notes
    -----
    Time bins whose posterior row contains any NaN/Inf entry are dropped
    from the matrix (a ``UserWarning`` reports how many), so they neither
    inflate the bin-0 column under ``"map"`` nor poison an actual-bin row
    under ``"expected"``.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding.metrics import confusion_matrix

    Create a simple environment and posterior:

    >>> positions = np.array([[0, 0], [5, 0], [0, 5], [5, 5]])
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> n_bins = env.n_bins

    Perfect decoding (posterior peaks at actual position):

    >>> posterior = np.eye(n_bins)  # Delta functions
    >>> actual_bins = np.arange(n_bins)
    >>> cm = confusion_matrix(env, posterior, actual_bins, method="map")
    >>> bool(np.allclose(cm, np.eye(n_bins)))
    True

    See Also
    --------
    decoding_error : Per-time-bin position error.
    decoding_correlation : Weighted correlation between decoded and actual.
    """
    posterior = np.asarray(posterior, dtype=np.float64)
    actual_bins = np.asarray(actual_bins, dtype=np.int64)

    n_bins = env.n_bins
    n_time_bins = posterior.shape[0]

    # Validate method
    if method not in ("map", "expected"):
        raise ValueError(f"Invalid method '{method}'. Must be 'map' or 'expected'.")

    # Validate shapes
    if posterior.ndim != 2:
        raise ValueError(f"posterior must be 2D, got shape {posterior.shape}")
    if posterior.shape[1] != n_bins:
        raise ValueError(
            f"posterior has {posterior.shape[1]} bins but environment has {n_bins} bins"
        )
    if actual_bins.ndim != 1:
        raise ValueError(f"actual_bins must be 1D, got shape {actual_bins.shape}")
    if len(actual_bins) != n_time_bins:
        raise ValueError(
            f"Length mismatch: posterior has {n_time_bins} time bins but "
            f"actual_bins has {len(actual_bins)}"
        )

    # Validate actual_bins range
    if np.any(actual_bins < 0) or np.any(actual_bins >= n_bins):
        min_bin = int(actual_bins.min())
        max_bin = int(actual_bins.max())
        raise ValueError(
            f"actual_bins contains values outside valid range [0, {n_bins}). "
            f"Found range [{min_bin}, {max_bin}]."
        )

    # Initialize confusion matrix
    cm = np.zeros((n_bins, n_bins), dtype=np.float64)

    # Rows with any non-finite posterior entry are undecodable (e.g. a
    # time bin with no spikes against a masked environment). argmax over
    # a NaN row returns bin 0, and accumulating a NaN row into the
    # "expected" matrix poisons that actual-bin's entire row. Drop such
    # rows from both summaries and surface how many were excluded.
    finite_rows = np.isfinite(posterior).all(axis=1)
    n_dropped = int(np.sum(~finite_rows))
    if n_dropped > 0:
        warnings.warn(
            f"confusion_matrix: skipped {n_dropped} of {n_time_bins} time "
            f"bins whose posterior contained NaN/Inf. The returned matrix is "
            f"built from the remaining {n_time_bins - n_dropped} bins.",
            UserWarning,
            stacklevel=2,
        )

    if method == "map":
        # Use argmax to get decoded bin for each (finite) time step.
        decoded_bins = np.argmax(posterior[finite_rows], axis=1)
        # Vectorized counting using np.add.at.
        np.add.at(cm, (actual_bins[finite_rows], decoded_bins), 1.0)

    else:  # method == "expected"
        # Accumulate posterior mass only from finite rows.
        np.add.at(cm, actual_bins[finite_rows], posterior[finite_rows])

    return cm


def decoding_correlation(
    decoded_positions: NDArray[np.float64],
    actual_positions: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> float:
    """Weighted Pearson correlation between decoded and actual positions.

    Computes a (possibly weighted) Pearson correlation coefficient between
    decoded position estimates and ground truth positions. For multi-dimensional
    positions, returns the mean correlation across all dimensions.

    Parameters
    ----------
    decoded_positions : NDArray[np.float64], shape (n_time_bins, n_dims)
        Decoded position estimates (e.g., MAP or mean positions).
    actual_positions : NDArray[np.float64], shape (n_time_bins, n_dims)
        Ground truth positions.
    weights : NDArray[np.float64], shape (n_time_bins,), optional
        Per-time-bin weights. If None, uniform weights (standard correlation).
        Typical use: weight by posterior certainty (1 - normalized_entropy) to
        down-weight uncertain time bins.

    Returns
    -------
    r : float
        Weighted Pearson correlation coefficient.
        For ``n_dims > 1``, returns mean correlation across dimensions.
        Returns NaN if:

        - Fewer than 2 valid (non-NaN, non-zero-weight) time bins remain
        - Either decoded or actual positions has zero variance in the
          valid samples (constant values after excluding NaN/zero-weight bins)
        - All weights are zero or weight sum overflows

    Raises
    ------
    ValueError
        If ``decoded_positions`` and ``actual_positions`` have different shapes.
        If ``weights`` shape doesn't match ``(n_time_bins,)``.

    Notes
    -----
    Time bins with NaN in positions or zero weight are excluded from the
    computation. This allows robust correlation even with missing data.

    The implementation uses a numerically stable centered formula to minimize
    catastrophic cancellation, especially important for long time series with
    large weights:

    .. code-block:: python

        # Normalize weights
        w = weights / weights.sum()

        # Weighted means (stable via np.average)
        mean_x = np.average(x, weights=w)
        mean_y = np.average(y, weights=w)

        # Center the data (reduces cancellation)
        x_centered = x - mean_x
        y_centered = y - mean_y

        # Weighted covariance and variances
        cov_xy = np.sum(w * x_centered * y_centered)
        var_x = np.sum(w * x_centered**2)
        var_y = np.sum(w * y_centered**2)

        # Correlation with zero-variance check
        denom = np.sqrt(var_x * var_y)
        r = cov_xy / denom if denom > 0 else np.nan

    This approach is more stable than the direct formula
    ``(sum(wx*y) - n*mean_x*mean_y) / ...`` which can suffer from
    cancellation when sums are large.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.decoding.metrics import decoding_correlation

    Perfect correlation (same positions):

    >>> decoded = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> actual = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> r = decoding_correlation(decoded, actual)
    >>> bool(abs(r - 1.0) < 1e-10)
    True

    Weighted correlation (down-weight uncertain bins):

    >>> decoded = np.array([[0.0], [1.0], [2.0], [100.0]])  # Outlier
    >>> actual = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> weights = np.array([1.0, 1.0, 1.0, 0.0])  # Zero-weight outlier
    >>> r = decoding_correlation(decoded, actual, weights=weights)
    >>> bool(abs(r - 1.0) < 1e-10)
    True

    See Also
    --------
    decoding_error : Per-time-bin position error.
    median_decoding_error : Median error summary statistic.
    """
    decoded_positions = np.asarray(decoded_positions, dtype=np.float64)
    actual_positions = np.asarray(actual_positions, dtype=np.float64)

    # Validate shapes match
    if decoded_positions.shape != actual_positions.shape:
        raise ValueError(
            f"Shape mismatch: decoded_positions has shape {decoded_positions.shape}, "
            f"but actual_positions has shape {actual_positions.shape}"
        )

    n_time_bins = decoded_positions.shape[0]

    # Handle 1D case: reshape to (n_time_bins, 1)
    if decoded_positions.ndim == 1:
        decoded_positions = decoded_positions.reshape(-1, 1)
        actual_positions = actual_positions.reshape(-1, 1)

    # Setup weights
    if weights is None:
        weights = np.ones(n_time_bins, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (n_time_bins,):
            raise ValueError(
                f"weights must have shape ({n_time_bins},), got {weights.shape}"
            )

    # Create valid mask: exclude NaN in any dimension and zero weights
    valid_mask = np.ones(n_time_bins, dtype=bool)
    valid_mask &= ~np.any(np.isnan(decoded_positions), axis=1)
    valid_mask &= ~np.any(np.isnan(actual_positions), axis=1)
    valid_mask &= weights > 0

    n_valid = valid_mask.sum()

    # Need at least 2 valid samples for correlation
    if n_valid < 2:
        return float(np.nan)

    # Filter to valid samples
    decoded_valid = decoded_positions[valid_mask]
    actual_valid = actual_positions[valid_mask]
    weights_valid = weights[valid_mask]

    # Normalize weights to sum to 1 (with overflow check)
    weight_sum = weights_valid.sum()
    if weight_sum == 0 or not np.isfinite(weight_sum):
        return float(np.nan)
    w = weights_valid / weight_sum

    # Compute correlation for each dimension, then average (vectorized)
    # Weighted means for all dimensions at once: shape (n_dims,)
    mean_decoded = np.average(decoded_valid, axis=0, weights=w)
    mean_actual = np.average(actual_valid, axis=0, weights=w)

    # Center the data: shape (n_valid, n_dims)
    decoded_centered = decoded_valid - mean_decoded
    actual_centered = actual_valid - mean_actual

    # Weighted covariance and variances for all dims: shape (n_dims,)
    # w has shape (n_valid,), so we broadcast: w[:, None] has shape (n_valid, 1)
    cov_xy = np.sum(w[:, np.newaxis] * decoded_centered * actual_centered, axis=0)
    var_decoded = np.sum(w[:, np.newaxis] * decoded_centered**2, axis=0)
    var_actual = np.sum(w[:, np.newaxis] * actual_centered**2, axis=0)

    # Correlation with zero-variance check (vectorized)
    denom = np.sqrt(var_decoded * var_actual)
    correlations = np.where(denom > 0, cov_xy / denom, np.nan)

    # Return mean correlation across dimensions
    # If any dimension is NaN, result is NaN
    if np.any(np.isnan(correlations)):
        return float(np.nan)

    return float(np.mean(correlations))
