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

        n_time_bins = decoded_positions.shape[0]
        errors = np.empty(n_time_bins, dtype=np.float64)

        for t in range(n_time_bins):
            decoded_point = decoded_positions[t]
            actual_point = actual_positions[t]

            # Check for NaN - propagate to output
            if np.any(np.isnan(decoded_point)) or np.any(np.isnan(actual_point)):
                errors[t] = np.nan
            else:
                # Use environment's distance_between method
                try:
                    errors[t] = env.distance_between(  # type: ignore[misc]
                        decoded_point, actual_point
                    )
                except (ValueError, KeyError):
                    # If points are outside environment, fall back to Euclidean
                    diff = decoded_point - actual_point
                    errors[t] = float(np.linalg.norm(diff))

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
        for t in range(n_time_bins):
            cm[actual_bins[t], :] += posterior[t, :]

    return cm
