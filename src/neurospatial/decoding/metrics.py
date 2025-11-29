"""Decoding quality metrics for evaluating position decoding accuracy.

This module provides functions for computing error metrics between decoded
and actual positions, including Euclidean and graph-based distance metrics.

Functions
---------
decoding_error
    Compute position error for each time bin.
median_decoding_error
    Compute median decoding error (convenience wrapper).
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
