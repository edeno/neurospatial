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

from neurospatial._exceptions import _format_error
from neurospatial._validation import validate_choice
from neurospatial.decoding.estimates import _egocentric_plane
from neurospatial.geometry import require_cartesian
from neurospatial.ops.distance import _geodesic_pair_distances

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.ops._types import DistanceMetric


def decoding_error(
    env: Environment,
    decoded_positions: NDArray[np.float64],
    actual_positions: NDArray[np.float64],
    *,
    metric: DistanceMetric = "geodesic",
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
          linearized track), measured between bin centers: the error is
          quantized to the bins, and two positions in one bin are 0 apart.
          A position outside every bin but within one ``env.bin_spacing``
          of the environment's extent (its bounding box), such as tracking
          jitter at an edge or a posterior mean between two arms, is
          measured from its nearest bin. A position farther out is not in
          the environment: its error is NaN, with a warning. Bins in
          disconnected components are ``inf`` apart. On an open grid the
          graph path can exceed the straight line by up to about 8%
          (8-connected neighbors).
        - "euclidean": Straight-line distance between the positions. On an
          egocentric polar environment, the distance between the two
          places in the animal's plane (the chord
          ``sqrt(r1**2 + r2**2 - 2 * r1 * r2 * cos(angle1 - angle2))``), so
          angles a turn apart coincide.

    Returns
    -------
    errors : NDArray[np.float64], shape (n_time_bins,)
        Distance between decoded and actual position at each time bin.
        Units match environment units (e.g., cm).

    Raises
    ------
    ValueError
        If the position arrays differ in shape or do not have ``env.n_dims``
        columns, if either contains Inf, or if ``metric`` is not
        "euclidean" or "geodesic".
    CartesianOnlyError
        If ``metric="geodesic"`` on an egocentric polar environment.

    Warns
    -----
    UserWarning
        If ``metric="geodesic"`` and a position lies outside the
        environment (see ``metric``); its error is NaN.

    Notes
    -----
    NaN values in either input propagate to NaN in output: a missing
    position has no error. This allows downstream functions (like
    ``median_decoding_error``) to use ``np.nanmedian`` for robust
    statistics.

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
    decoded_positions, actual_positions = _paired_positions(
        env, decoded_positions, actual_positions, call="decoding_error()"
    )
    validate_choice(
        metric, ("euclidean", "geodesic"), name="metric", owner="decoding_error()"
    )
    for name, positions in (
        ("decoded_positions", decoded_positions),
        ("actual_positions", actual_positions),
    ):
        infinite = np.isinf(positions).any(axis=1)
        if np.any(infinite):
            raise ValueError(
                _format_error(
                    f"decoding_error(): {name} contains Inf in "
                    f"{int(np.sum(infinite))} of {len(positions)} time bins.",
                    why="an infinite position is not a place, so it has no "
                    "distance to another position",
                    fix="mark missing or invalid positions as NaN (their error "
                    "is NaN), e.g. positions[np.isinf(positions).any(axis=1)] "
                    "= np.nan",
                )
            )

    if metric == "euclidean":
        if env.coordinate_kind == "polar":
            # (distance, angle) rows: measure between places in the animal's
            # plane, so angles a turn apart coincide.
            decoded_positions = _egocentric_plane(decoded_positions)
            actual_positions = _egocentric_plane(actual_positions)
        diff = decoded_positions - actual_positions
        return cast("NDArray[np.float64]", np.linalg.norm(diff, axis=1))

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
    decoded_bins = np.full(len(errors), -1, dtype=np.intp)
    actual_bins = np.full(len(errors), -1, dtype=np.intp)
    decoded_bins[valid] = _bins_or_nearest(env, decoded_positions[valid])
    actual_bins[valid] = _bins_or_nearest(env, actual_positions[valid])
    far = valid & ((decoded_bins < 0) | (actual_bins < 0))
    if np.any(far):
        warnings.warn(
            f"decoding_error(): {int(np.sum(far))} of {len(errors)} time bins "
            "have a decoded or actual position outside the environment, more "
            "than one bin spacing beyond its extent, so their geodesic error "
            "is NaN. Check those positions for tracking glitches or a "
            "different coordinate frame, or use metric='euclidean'.",
            UserWarning,
            stacklevel=2,
        )
    measured = valid & ~far
    if np.any(measured):
        errors[measured] = _geodesic_pair_distances(
            env.connectivity,
            decoded_bins[measured],
            actual_bins[measured],
            operation="decoding_error()",
        )
    return errors


def _paired_positions(
    env: Environment,
    decoded_positions: NDArray[np.float64],
    actual_positions: NDArray[np.float64],
    *,
    call: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return both position arrays as float, refusing mismatched shapes."""
    decoded_positions = np.asarray(decoded_positions, dtype=np.float64)
    actual_positions = np.asarray(actual_positions, dtype=np.float64)
    if decoded_positions.shape != actual_positions.shape:
        raise ValueError(
            f"Shape mismatch: decoded_positions has shape {decoded_positions.shape}, "
            f"but actual_positions has shape {actual_positions.shape}"
        )
    if decoded_positions.ndim != 2 or decoded_positions.shape[1] != env.n_dims:
        raise ValueError(
            f"{call} needs positions with {env.n_dims} columns, one per "
            f"dimension of the environment; got shape {decoded_positions.shape}. "
            "Pass (n_time_bins, n_dims) positions in the environment's coordinates."
        )
    return decoded_positions, actual_positions


def _bins_or_nearest(
    env: Environment, positions: NDArray[np.float64]
) -> NDArray[np.intp]:
    """Bin of each finite position, its nearest bin when just outside, else -1.

    A position outside every bin is measured from its nearest bin when it
    lies within one ``env.bin_spacing`` of the environment's extent (its
    domain's bounding box, or the bin centers' when the domain is unknown):
    tracking jitter at an edge, or a posterior mean between two arms of a
    maze. A position farther out is not in the environment and gets -1.
    """
    bins = np.asarray(env.bin_at(positions), dtype=np.intp)
    outside = np.flatnonzero(bins < 0)
    if outside.size:
        bounds = env.arena.domain_bounds()
        if bounds is None:
            centers = env.bin_centers
            bounds = np.stack([centers.min(axis=0), centers.max(axis=0)])
        margin = env.bin_spacing
        near = np.all(
            (positions[outside] >= bounds[0] - margin)
            & (positions[outside] <= bounds[1] + margin),
            axis=1,
        )
        bins[outside[near]] = env.nearest_bin(positions[outside[near]])
    return bins


def median_decoding_error(
    env: Environment,
    decoded_positions: NDArray[np.float64],
    actual_positions: NDArray[np.float64],
    *,
    metric: DistanceMetric = "geodesic",
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

        - "map": Use the MAP bin, the bin of highest posterior density
          (probability divided by bin size). Returns integer counts.
          Cell (i, j) contains the count of time bins where actual=i and
          decoded=j.
        - "expected": Accumulate full posterior mass. Cell (i, j) contains
          sum of P(decoded=j | actual=i) across all time bins where
          actual=i. Rows sum to the count of actual bin occurrences.

    Returns
    -------
    cm : NDArray[np.float64], shape (n_bins, n_bins)
        Confusion matrix. Rows are actual bins, columns are decoded bins.
        For ``method="map"``, the matrix sums to the number of time bins
        with a finite posterior row (n_time_bins when none is dropped; see
        Notes). For ``method="expected"``, each row sums to the count of
        that bin's occurrences in actual_bins among those time bins.

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

    validate_choice(
        method, ("map", "expected"), name="method", owner="confusion_matrix()"
    )

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

    # Rows with any non-finite posterior entry are undecodable (e.g. the
    # all-NaN rows that normalize_to_posterior(handle_degenerate="nan")
    # gives where the likelihood is zero in every bin). argmax over a NaN
    # row returns bin 0, and accumulating a NaN row into the "expected"
    # matrix poisons that actual-bin's entire row. Drop such rows from both
    # summaries and surface how many were excluded.
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

    if n_dropped:
        posterior, actual_bins = posterior[finite_rows], actual_bins[finite_rows]

    if method == "map":
        # The decoded bin is the highest-density bin of each finite row.
        from neurospatial.decoding._support import relative_bin_measures
        from neurospatial.decoding.estimates import _density_mode

        measures = relative_bin_measures(env, context="confusion_matrix")
        decoded_bins = _density_mode(posterior, measures)
        counts = np.bincount(actual_bins * n_bins + decoded_bins, minlength=n_bins**2)
        return counts.reshape(n_bins, n_bins).astype(np.float64)

    # method == "expected": sum each actual bin's posterior rows with a
    # one-hot (actual bin x time bin) product.
    from scipy import sparse

    rows_of_bin = sparse.csr_matrix(
        (np.ones(len(actual_bins)), (actual_bins, np.arange(len(actual_bins)))),
        shape=(n_bins, len(actual_bins)),
    )
    return np.asarray(rows_of_bin @ posterior, dtype=np.float64)


def decoding_correlation(
    env: Environment,
    decoded_positions: NDArray[np.float64],
    actual_positions: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> float:
    """Weighted Pearson correlation between decoded and actual positions.

    On a track the correlation is between the decoded and actual positions
    along the track. In an open arena it is the mean, over spatial
    dimensions, of the per-dimension correlation.

    Parameters
    ----------
    env : Environment
        Environment the positions are in. On a linearized track the
        positions are first projected onto the track
        (:meth:`Environment.to_linear`); its ``n_dims`` is the number of
        position columns.
    decoded_positions : NDArray[np.float64], shape (n_time_bins, n_dims)
        Decoded position estimates (e.g., MAP or mean positions).
    actual_positions : NDArray[np.float64], shape (n_time_bins, n_dims)
        Ground truth positions.
    weights : NDArray[np.float64], shape (n_time_bins,), optional
        Non-negative, finite per-time-bin weights; a time bin of weight
        ``k`` counts as ``k`` copies of it and a zero weight leaves it out.
        If None, uniform weights (standard correlation). Typical use:
        weight by posterior certainty (1 - normalized_entropy) to
        down-weight uncertain time bins.

    Returns
    -------
    r : float
        Weighted Pearson correlation coefficient, in [-1, 1]:

        - On a linearized track or a 1-D environment, the correlation of
          the position along the track.
        - Otherwise, the mean of the per-dimension correlations over the
          dimensions in which both the decoded and actual positions vary.
          A dimension that does not vary (e.g. a horizontal track built as
          a 2-D arena) has no correlation; it is left out with a warning.

        NaN, with a warning when the positions do not vary, if fewer than 2
        time bins have finite positions and a positive weight, or if no
        dimension varies.

    Raises
    ------
    ValueError
        If the position arrays differ in shape or do not have ``env.n_dims``
        columns, or if ``weights`` does not have shape ``(n_time_bins,)`` or
        holds negative, NaN or Inf values.
    CartesianOnlyError
        If ``env`` is an egocentric polar environment: averaging a
        distance correlation with a linear correlation of angles has no
        meaning.

    Notes
    -----
    Time bins with NaN in either position or zero weight are excluded from
    the computation, so missing positions do not count as zero.

    With weights ``w`` normalized to sum to 1 and weighted means
    ``mean_x = sum(w * x)``, the correlation of one dimension is

    .. code-block:: python

        cov_xy = np.sum(w * (x - mean_x) * (y - mean_y))
        var_x = np.sum(w * (x - mean_x) ** 2)
        var_y = np.sum(w * (y - mean_y) ** 2)
        r = cov_xy / np.sqrt(var_x * var_y)

    Centering before multiplying avoids the cancellation of the direct
    formula ``(sum(w * x * y) - mean_x * mean_y) / ...`` when sums are large.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding.metrics import decoding_correlation
    >>> env = Environment.linear_track(
    ...     endpoints=[(0.0, 0.0), (10.0, 0.0)], bin_size=1.0, track_width=2.0
    ... )

    Perfect correlation along the track:

    >>> actual = np.array([[0.5, 0.0], [2.5, 0.0], [4.5, 0.0], [6.5, 0.0]])
    >>> r = decoding_correlation(env, actual, actual)
    >>> bool(abs(r - 1.0) < 1e-10)
    True

    Weighted correlation (down-weight uncertain bins):

    >>> decoded = actual.copy()
    >>> decoded[3, 0] = 0.5  # an outlier
    >>> weights = np.array([1.0, 1.0, 1.0, 0.0])  # zero-weight outlier
    >>> r = decoding_correlation(env, decoded, actual, weights=weights)
    >>> bool(abs(r - 1.0) < 1e-10)
    True

    See Also
    --------
    decoding_error : Per-time-bin position error.
    median_decoding_error : Median error summary statistic.
    """
    require_cartesian(
        env.coordinate_system,
        "decoding_correlation()",
        why="averaging a correlation of distances with a linear correlation "
        "of angles has no meaning, and angles a turn apart are one direction",
        fix="correlate the distance column with np.corrcoef and the angle "
        "column with neurospatial.stats.circular_circular_correlation",
    )
    decoded_positions, actual_positions = _paired_positions(
        env, decoded_positions, actual_positions, call="decoding_correlation()"
    )
    n_time_bins = decoded_positions.shape[0]

    if weights is None:
        weights = np.ones(n_time_bins, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (n_time_bins,):
            raise ValueError(
                f"weights must have shape ({n_time_bins},), got {weights.shape}"
            )
        invalid = ~np.isfinite(weights) | (weights < 0)
        if np.any(invalid):
            raise ValueError(
                _format_error(
                    f"decoding_correlation(): weights has {int(np.sum(invalid))} "
                    f"negative, NaN or Inf values of {n_time_bins}.",
                    why="a weight is how many times a time bin counts, so it "
                    "must be a finite number of at least 0",
                    fix="give undecodable or excluded time bins weight 0, e.g. "
                    "weights = np.nan_to_num(weights, nan=0.0), and check how "
                    "negative weights arose",
                )
            )

    # Exclude NaN positions and zero weights.
    valid = (
        ~np.any(np.isnan(decoded_positions), axis=1)
        & ~np.any(np.isnan(actual_positions), axis=1)
        & (weights > 0)
    )
    if np.sum(valid) < 2:
        return float(np.nan)
    decoded_valid = decoded_positions[valid]
    actual_valid = actual_positions[valid]
    if env.is_linearized_track:
        decoded_valid = env.to_linear(decoded_valid)[:, np.newaxis]
        actual_valid = env.to_linear(actual_valid)[:, np.newaxis]
    w = weights[valid] / weights[valid].sum()

    varies = (np.ptp(decoded_valid, axis=0) > 0) & (np.ptp(actual_valid, axis=0) > 0)
    if not np.all(varies):
        constant = np.flatnonzero(~varies).tolist()
        where = (
            "along the track"
            if env.is_linearized_track
            else f"in dimension(s) {constant} of {env.n_dims}"
        )
        warnings.warn(
            f"decoding_correlation(): the decoded or actual positions do not "
            f"vary {where}, so they have no correlation there"
            + (
                "; it is left out of the mean."
                if np.any(varies)
                else ", and the result is NaN."
            ),
            UserWarning,
            stacklevel=2,
        )
        if not np.any(varies):
            return float(np.nan)
    decoded_centered = decoded_valid[:, varies] - w @ decoded_valid[:, varies]
    actual_centered = actual_valid[:, varies] - w @ actual_valid[:, varies]

    # Weighted covariance and variances per dimension: shape (n_varying,)
    cov_xy = w @ (decoded_centered * actual_centered)
    var_decoded = w @ decoded_centered**2
    var_actual = w @ actual_centered**2
    correlations = cov_xy / np.sqrt(var_decoded * var_actual)
    return float(np.clip(np.mean(correlations), -1.0, 1.0))
