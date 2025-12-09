"""Decoding quality metrics for evaluating position decoding accuracy.

This module provides functions for computing error metrics between decoded
and actual positions, including Euclidean and graph-based distance metrics.

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

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial import Environment


def decoding_error(
    decoded_positions: NDArray[np.float64],
    actual_positions: NDArray[np.float64],
    *,
    metric: Literal["euclidean", "graph"] = "euclidean",
    env: Environment | None = None,
) -> NDArray[np.float64]:
    """Compute position error for each time bin.

    Calculates the distance between decoded and actual positions using either
    Euclidean distance (straight-line) or graph distance (shortest path along
    environment connectivity graph).

    Parameters
    ----------
    decoded_positions : NDArray[np.float64], shape (n_time_bins, n_dims)
        Decoded position estimates (e.g., MAP or mean positions).
    actual_positions : NDArray[np.float64], shape (n_time_bins, n_dims)
        Ground truth positions.
    metric : {"euclidean", "graph"}, default="euclidean"
        Distance metric to use:

        - "euclidean": Straight-line Euclidean distance. Fast and simple.
        - "graph": Shortest-path distance along environment graph.
          Useful for mazes where Euclidean distance is misleading.
          Requires ``env`` parameter.
    env : Environment, optional
        Required when ``metric="graph"``. Used to compute graph distances
        via ``env.distance_between()``.

    Returns
    -------
    errors : NDArray[np.float64], shape (n_time_bins,)
        Distance between decoded and actual position at each time bin.
        Units match environment units (e.g., cm).

    Raises
    ------
    ValueError
        If ``metric="graph"`` but ``env`` is None.
        If ``metric`` is not "euclidean" or "graph".

    Notes
    -----
    NaN values in either input propagate to NaN in output. This allows
    downstream functions (like ``median_decoding_error``) to use
    ``np.nanmedian`` for robust statistics.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.decoding.metrics import decoding_error

    Simple 2D Euclidean errors:

    >>> decoded = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    >>> actual = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    >>> errors = decoding_error(decoded, actual)
    >>> bool(np.allclose(errors, [0.0, 3.0, 4.0]))
    True

    See Also
    --------
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

    errors: NDArray[np.float64]

    if metric == "euclidean":
        # Euclidean distance: sqrt(sum((decoded - actual)^2, axis=1))
        diff = decoded_positions - actual_positions
        errors = cast("NDArray[np.float64]", np.linalg.norm(diff, axis=1))

    elif metric == "graph":
        if env is None:
            raise ValueError(
                "env is required when metric='graph'. "
                "Provide an Environment instance for graph-based distance computation."
            )

        from neurospatial.ops.distance import geodesic_distance_matrix

        n_time_bins = decoded_positions.shape[0]

        # Identify NaN positions (vectorized)
        nan_mask = np.any(np.isnan(decoded_positions), axis=1) | np.any(
            np.isnan(actual_positions), axis=1
        )

        # Initialize errors array
        errors = np.empty(n_time_bins, dtype=np.float64)
        errors[nan_mask] = np.nan

        # Process non-NaN positions
        valid_mask = ~nan_mask
        if np.any(valid_mask):
            valid_decoded = decoded_positions[valid_mask]
            valid_actual = actual_positions[valid_mask]

            # Map positions to bins (vectorized)
            decoded_bins = env.bin_at(valid_decoded)
            actual_bins = env.bin_at(valid_actual)

            # Identify positions outside environment (bin_at returns -1)
            outside_mask = (decoded_bins < 0) | (actual_bins < 0)
            inside_mask = ~outside_mask

            # For positions inside environment, use graph distance
            if np.any(inside_mask):
                # Compute geodesic distance matrix (cached per environment)
                dist_matrix = geodesic_distance_matrix(
                    env.connectivity,
                    env.n_bins,
                    weight="distance",
                )
                # Look up graph distances (vectorized)
                inside_decoded = decoded_bins[inside_mask]
                inside_actual = actual_bins[inside_mask]
                graph_errors = dist_matrix[inside_decoded, inside_actual]

                # Assign to valid positions that are inside
                valid_indices = np.where(valid_mask)[0]
                errors[valid_indices[inside_mask]] = graph_errors

            # For positions outside environment, fall back to Euclidean
            if np.any(outside_mask):
                outside_decoded = valid_decoded[outside_mask]
                outside_actual = valid_actual[outside_mask]
                euclidean_errors = np.linalg.norm(
                    outside_decoded - outside_actual, axis=1
                )

                # Assign to valid positions that are outside
                valid_indices = np.where(valid_mask)[0]
                errors[valid_indices[outside_mask]] = euclidean_errors

    else:
        raise ValueError(f"Invalid metric '{metric}'. Must be 'euclidean' or 'graph'.")

    return errors


def median_decoding_error(
    decoded_positions: NDArray[np.float64],
    actual_positions: NDArray[np.float64],
) -> float:
    """Compute median Euclidean decoding error (ignoring NaN).

    Convenience function equivalent to:
    ``np.nanmedian(decoding_error(decoded, actual))``

    Parameters
    ----------
    decoded_positions : NDArray[np.float64], shape (n_time_bins, n_dims)
        Decoded position estimates (e.g., MAP or mean positions).
    actual_positions : NDArray[np.float64], shape (n_time_bins, n_dims)
        Ground truth positions.

    Returns
    -------
    median_error : float
        Median distance between decoded and actual positions, ignoring
        time bins where either position contains NaN.
        Returns NaN if all time bins contain NaN.

    Notes
    -----
    Uses ``np.nanmedian`` internally, so NaN values from position errors
    are automatically excluded from the median calculation.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.decoding.metrics import median_decoding_error

    >>> decoded = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])
    >>> actual = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    >>> median_decoding_error(decoded, actual)
    3.0

    See Also
    --------
    decoding_error : Per-time-bin error computation.
    """
    errors = decoding_error(decoded_positions, actual_positions, metric="euclidean")
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

    if method == "map":
        # Use argmax to get decoded bin for each time step
        decoded_bins = np.argmax(posterior, axis=1)

        # Vectorized counting using np.add.at (much faster for large datasets)
        np.add.at(cm, (actual_bins, decoded_bins), 1.0)

    else:  # method == "expected"
        # Accumulate posterior mass: cm[actual, :] += posterior[t, :]
        # Vectorized using np.add.at with row indices
        np.add.at(cm, actual_bins, posterior)

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
