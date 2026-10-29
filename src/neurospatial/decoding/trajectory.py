"""Trajectory analysis for decoded position sequences.

This module provides functions for fitting and detecting trajectories in
posterior probability sequences from Bayesian decoding. These are useful for
analyzing replay events and sequential activation patterns.

Getting replay (ripple) intervals
----------------------------------
The trajectory fits here characterize a posterior *within* a candidate replay
event. They read each bin's position along the track (the linear position of a
linearized track, or the coordinate of a 1-D environment), so slopes are in
environment units per second and do not depend on how bins are numbered. They
refuse 2-D and 3-D arenas, which have no single axis to fit along.

The fits do not themselves detect the sharp-wave-ripple (SWR) events that
delimit replay. neurospatial does not implement ripple detection. Instead,
detect ripple intervals with the external ``ripple_detection`` package
(``pip install ripple_detection``), then decode the spikes within each
returned interval and fit the posterior with the functions in this module.

The intervals ``ripple_detection`` returns (an array of ``(start_time,
end_time)`` pairs in seconds) also feed directly into
:func:`neurospatial.events.peri_event_histogram` — pass the interval start
times as the event times to build a peri-ripple histogram. For example::

    from ripple_detection import Kay_ripple_detector
    from neurospatial.events import peri_event_histogram

    ripple_times = Kay_ripple_detector(time, lfps, speed, sampling_frequency)
    ripple_starts = ripple_times[:, 0]  # (start, end) pairs -> start times
    psth = peri_event_histogram(spike_times, ripple_starts, window=(-0.5, 0.5))

Functions
---------
fit_isotonic_trajectory
    Fit monotonic trajectory using isotonic regression.
fit_linear_trajectory
    Fit linear trajectory with optional Monte Carlo uncertainty.
detect_trajectory_radon
    Detect linear trajectory using Radon transform.

Result Dataclasses
------------------
IsotonicFitResult
    Container for isotonic regression results.
LinearFitResult
    Container for linear regression results.
RadonDetectionResult
    Container for Radon transform detection results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from neurospatial.environment import Environment

# Check for optional scikit-image dependency
_SKIMAGE_AVAILABLE = False
radon: Any
try:
    from skimage.transform import radon

    _SKIMAGE_AVAILABLE = True
except ImportError:
    radon = None


@dataclass(frozen=True, repr=False, eq=False)
class IsotonicFitResult(ReadOnlyArraysMixin, ResultMixin):
    """Container for isotonic regression trajectory fit results.

    Isotonic regression fits a monotonic (either strictly increasing or
    decreasing) function to the decoded position sequence. This is useful
    for detecting replay events that represent sequential traversal of
    spatial locations.

    Attributes
    ----------
    fitted_positions : NDArray[np.float64]
        Fitted monotonic positions along the track, in environment units,
        shape (n_time_bins,). These are constrained to be monotonically
        increasing or decreasing.
    r_squared : float
        Coefficient of determination (R²) measuring goodness of fit.
        Range [0, 1] where 1 indicates perfect fit.
    direction : {"increasing", "decreasing"}
        Direction of the fitted monotonic trajectory.
        "increasing" means position increases with time (forward replay).
        "decreasing" means position decreases with time (reverse replay).
    residuals : NDArray[np.float64]
        Residuals (observed - fitted) for each time bin, in environment
        units, shape (n_time_bins,). Useful for diagnostics and outlier
        detection.

    See Also
    --------
    fit_isotonic_trajectory : Function that produces this result.
    LinearFitResult : For linear (non-monotonic constrained) fits.

    Examples
    --------
    >>> result = fit_isotonic_trajectory(env, posterior, times)  # doctest: +SKIP
    >>> print(
    ...     f"R² = {result.r_squared:.3f}, direction = {result.direction}"
    ... )  # doctest: +SKIP
    >>> replay_speed_cm_s = np.diff(result.fitted_positions) / np.diff(
    ...     times
    ... )  # doctest: +SKIP
    """

    fitted_positions: NDArray[np.float64]
    r_squared: float
    direction: Literal["increasing", "decreasing"]
    residuals: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Own read-only copies of fitted positions and residuals."""
        self._own_arrays()

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {"r_squared": self.r_squared, "direction": self.direction}


@dataclass(frozen=True, repr=False)
class LinearFitResult(ResultMixin):
    """Container for linear trajectory fit results.

    Linear regression fits a straight line to the decoded position sequence.
    This provides slope (replay speed) and intercept estimates, optionally
    with uncertainty quantification via Monte Carlo sampling.

    Attributes
    ----------
    slope : float
        Slope of the fitted line: replay speed along the track in
        environment units per second (cm/s for a track in cm). Positive
        slope means position increases along the track with time.
    intercept : float
        Position along the track, in environment units, where the fitted
        line crosses ``times == 0``. This is time zero of the ``times``
        values passed in, so shift ``times`` to the event start to read the
        intercept as the start position.
    r_squared : float
        Coefficient of determination (R²) measuring goodness of fit.
        Range [0, 1] where 1 indicates perfect linear relationship.
    slope_std : float | None
        Standard deviation of slope estimates from Monte Carlo sampling.
        Measures the spread of sampled slopes across different draws from
        the posterior. Only available when method="sample".
        None when method="map" (deterministic fit).

    See Also
    --------
    fit_linear_trajectory : Function that produces this result.
    IsotonicFitResult : For monotonic (non-linear) fits.

    Examples
    --------
    >>> result = fit_linear_trajectory(
    ...     env, posterior, times, method="sample", rng=42
    ... )  # doctest: +SKIP
    >>> print(
    ...     f"speed = {result.slope:.0f} ± {result.slope_std:.0f} cm/s"
    ... )  # doctest: +SKIP
    """

    slope: float
    intercept: float
    r_squared: float
    slope_std: float | None

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "slope_std": self.slope_std,
        }


@dataclass(frozen=True, repr=False, eq=False)
class RadonDetectionResult(ReadOnlyArraysMixin, ResultMixin):
    """Container for Radon transform trajectory detection results.

    The Radon transform treats the posterior as a 2D image (time × position)
    and finds the line with maximum integrated probability mass. This is
    particularly effective for detecting diagonal stripe patterns in replay
    posteriors.

    Attributes
    ----------
    angle_degrees : float
        Detected trajectory angle in degrees.
        0° = horizontal (constant position, no movement).
        90° = vertical (instantaneous teleportation).
        Positive angles indicate forward replay, negative indicate reverse.
    score : float
        Radon transform value at the detected peak.
        Higher values indicate stronger linear structure in the posterior.
        Useful for comparing replay "quality" across events.
    offset : float
        Perpendicular offset from origin in the Radon parameterization.
        Combined with angle, uniquely identifies the detected line.
    sinogram : NDArray[np.float64]
        Full Radon transform (sinogram) array, shape (n_angles, n_offsets).
        Useful for visualization and secondary peak detection.

    See Also
    --------
    detect_trajectory_radon : Function that produces this result.

    Notes
    -----
    The Radon transform uses ``scikit-image``, which ships as a core
    neurospatial dependency (always installed).

    Examples
    --------
    >>> result = detect_trajectory_radon(env, posterior)  # doctest: +SKIP
    >>> print(
    ...     f"Detected angle: {result.angle_degrees:.1f}°, score: {result.score:.3f}"
    ... )  # doctest: +SKIP
    """

    angle_degrees: float
    score: float
    offset: float
    sinogram: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Own a read-only copy of the Radon sinogram."""
        self._own_arrays()

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "angle_degrees": self.angle_degrees,
            "score": self.score,
            "offset": self.offset,
        }


_TRACK_ALTERNATIVE = (
    "a 2-D arena has no single axis to fit a replay line along, so decode on "
    "a linearized track (Environment.linear_track, Environment.maze or "
    "Environment.from_graph)"
)


def _track_positions(
    env: Environment, posterior: NDArray[np.float64], operation: str
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Each bin's extent along the track, in track order.

    Parameters
    ----------
    env : Environment
        A 1-D environment or a linearized track.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior mass per bin, in bin numbering.
    operation : str
        The calling function, for the refusal message.

    Returns
    -------
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Columns reordered so that bins run along the track.
    starts, stops : NDArray[np.float64], shape (n_bins,)
        Where each reordered bin starts and ends along the track, in
        environment units.

    Raises
    ------
    ValueError
        If ``env`` is neither 1-D nor a linearized track, if ``posterior`` is
        not 2-D, if it does not have one column per bin, or if a row is not a
        probability mass (see :func:`_require_probability_rows`).
    """
    from neurospatial.environment._queries import require_track_extents

    starts, stops = require_track_extents(
        env, operation, alternative=_TRACK_ALTERNATIVE
    )
    posterior = np.asarray(posterior)
    if posterior.ndim != 2:
        raise ValueError(f"posterior must be 2D, got shape {posterior.shape}")
    if posterior.shape[1] != env.n_bins:
        raise ValueError(
            f"posterior has {posterior.shape[1]} columns but the environment "
            f"has {env.n_bins} bins. Pass the posterior decoded on this "
            "environment, with one column per bin."
        )
    posterior = _require_probability_rows(posterior, operation)
    # A stable sort keeps bins that are already in track order in place, so
    # sampling below draws the same bins as it would in bin numbering.
    order = np.argsort(starts, kind="stable")
    return posterior[:, order], starts[order], stops[order]


def _require_probability_rows(
    posterior: NDArray[Any], operation: str
) -> NDArray[np.float64]:
    """Refuse rows that are not a probability mass over the bins.

    Parameters
    ----------
    posterior : NDArray, shape (n_time_bins, n_bins)
        Posterior mass per bin.
    operation : str
        The calling function, for the refusal message.

    Returns
    -------
    NDArray[np.float64], shape (n_time_bins, n_bins)
        The posterior as float64.

    Raises
    ------
    ValueError
        If a row holds NaN or Inf, a negative entry, or does not sum to 1
        within the rounding error of the posterior's precision (the square
        root of its machine epsilon, as for the credible region).
    """
    tolerance = float(np.sqrt(np.finfo(np.result_type(posterior, np.float32)).eps))
    posterior = np.asarray(posterior, dtype=np.float64)
    finite = np.isfinite(posterior).all(axis=1)
    negative = finite & (posterior < 0).any(axis=1)
    sums = posterior.sum(axis=1)
    unnormalized = finite & ~negative & (np.abs(sums - 1.0) > tolerance)
    bad = ~finite | negative | unnormalized
    if not np.any(bad):
        return posterior
    problems = []
    if np.any(~finite):
        problems.append(f"{int(np.sum(~finite))} contain NaN or Inf")
    if np.any(negative):
        problems.append(f"{int(np.sum(negative))} have negative entries")
    if np.any(unnormalized):
        worst = sums[unnormalized][np.argmax(np.abs(sums[unnormalized] - 1.0))]
        problems.append(
            f"{int(np.sum(unnormalized))} do not sum to 1 (worst row sums to "
            f"{worst:.6g})"
        )
    raise ValueError(
        _format_error(
            f"{operation} needs each posterior row to be a probability mass "
            f"over the bins, but {int(np.sum(bad))} of {len(posterior)} rows "
            f"are not: {'; '.join(problems)}.",
            why="a row with NaN has no decoded position but would be read as "
            "the track start (bin 0), and a row that does not sum to 1 weighs "
            "its time bin more or less than the others (a sampled fit draws "
            "the missing mass at the track start)",
            fix="drop undecodable time bins together with their times, "
            "keep = np.isfinite(posterior).all(axis=1); posterior, times = "
            "posterior[keep], times[keep]; renormalize rows that only lack "
            "mass with posterior / posterior.sum(axis=1, keepdims=True)",
        )
    )


def _decoded_positions(
    posterior: NDArray[np.float64],
    centers: NDArray[np.float64],
    method: str,
) -> NDArray[np.float64]:
    """Position along the track per time bin: MAP bin center or posterior mean."""
    if method == "map":
        positions: NDArray[np.float64] = centers[np.argmax(posterior, axis=1)]
        return positions
    # The posterior is a probability mass per bin, so its mean position is
    # the mass-weighted mean of the bin centers.
    return np.asarray(posterior @ centers, dtype=np.float64)


def fit_isotonic_trajectory(
    env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    increasing: bool | None = None,
    method: Literal["map", "expected"] = "expected",
) -> IsotonicFitResult:
    """Fit monotonic trajectory using isotonic regression.

    Isotonic regression fits a monotonic (non-decreasing or non-increasing)
    function to the decoded position along the track. This is useful for
    detecting replay events that represent sequential traversal of spatial
    locations.

    Parameters
    ----------
    env : Environment
        The environment the posterior was decoded on: a 1-D environment or a
        linearized track. Positions are read along the track (the linear
        position of a track, or the coordinate of a 1-D environment), so the
        fit does not depend on how bins are numbered.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution over spatial bins for each time bin.
        Each row should sum to 1.
    times : NDArray[np.float64], shape (n_time_bins,)
        Time values for each time bin. Used as the independent variable (x)
        in the regression.
    increasing : bool | None, optional
        Direction constraint for the isotonic fit:
        - True: Fit increasing (non-decreasing) monotonic function
        - False: Fit decreasing (non-increasing) monotonic function
        - None (default): Try both directions and return the one with higher R²
    method : {"map", "expected"}, optional
        How to extract position from the posterior for fitting:
        - "map": Use the center of the most probable bin
        - "expected" (default): Use the posterior mean position

    Returns
    -------
    IsotonicFitResult
        Container with fitted_positions, r_squared, direction, and residuals,
        with positions in environment units along the track.

    Raises
    ------
    ValueError
        If method is not "map" or "expected", if ``env`` is neither 1-D nor
        a linearized track, or if ``posterior`` does not have one column per
        bin.

    See Also
    --------
    IsotonicFitResult : Container for isotonic fit results.
    fit_linear_trajectory : For unconstrained linear fits.

    Notes
    -----
    This function uses scikit-learn's IsotonicRegression, which implements
    the pool adjacent violators algorithm (PAVA). The R² is computed as:

    .. math::

        R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}

    where :math:`SS_{res}` is the sum of squared residuals and :math:`SS_{tot}`
    is the total sum of squares.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding.trajectory import fit_isotonic_trajectory
    >>>
    >>> # A 100 cm track in 2 cm bins and a posterior that moves along it
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=2.0, units="cm"
    ... )
    >>> n_time_bins = 20
    >>> posterior = np.zeros((n_time_bins, env.n_bins))
    >>> for t in range(n_time_bins):
    ...     posterior[t, t * 2] = 1.0  # Delta posteriors
    >>> times = np.linspace(0, 1, n_time_bins)
    >>>
    >>> result = fit_isotonic_trajectory(env, posterior, times)
    >>> print(f"R² = {result.r_squared:.3f}, direction = {result.direction}")
    R² = 1.000, direction = increasing
    """
    from sklearn.isotonic import IsotonicRegression

    if method not in ("map", "expected"):
        raise ValueError(f"method must be 'map' or 'expected', got {method!r}")

    posterior, starts, stops = _track_positions(
        env, posterior, "fit_isotonic_trajectory()"
    )
    positions = _decoded_positions(posterior, (starts + stops) / 2, method)

    # If direction is specified, fit once
    if increasing is not None:
        iso_reg = IsotonicRegression(increasing=increasing)
        fitted = iso_reg.fit_transform(times, positions)
        direction: Literal["increasing", "decreasing"] = (
            "increasing" if increasing else "decreasing"
        )
        residuals = positions - fitted
        r_squared = _compute_r_squared(positions, fitted)

        return IsotonicFitResult(
            fitted_positions=fitted,
            r_squared=float(r_squared),
            direction=direction,
            residuals=residuals,
        )

    # Try both directions and return the one with better R²
    iso_inc = IsotonicRegression(increasing=True)
    fitted_inc = iso_inc.fit_transform(times, positions)
    r2_inc = _compute_r_squared(positions, fitted_inc)

    iso_dec = IsotonicRegression(increasing=False)
    fitted_dec = iso_dec.fit_transform(times, positions)
    r2_dec = _compute_r_squared(positions, fitted_dec)

    if r2_inc >= r2_dec:
        return IsotonicFitResult(
            fitted_positions=fitted_inc,
            r_squared=float(r2_inc),
            direction="increasing",
            residuals=positions - fitted_inc,
        )
    else:
        return IsotonicFitResult(
            fitted_positions=fitted_dec,
            r_squared=float(r2_dec),
            direction="decreasing",
            residuals=positions - fitted_dec,
        )


def _compute_r_squared(
    observed: NDArray[np.float64], predicted: NDArray[np.float64]
) -> float:
    """Compute R² (coefficient of determination).

    Parameters
    ----------
    observed : NDArray[np.float64]
        Observed values.
    predicted : NDArray[np.float64]
        Predicted (fitted) values.

    Returns
    -------
    float
        R² value in [0, 1]. Returns 0 if total variance is zero.
    """
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return float(1.0 - ss_res / ss_tot)


def fit_linear_trajectory(
    env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    n_samples: int = 1000,
    method: Literal["map", "sample"] = "sample",
    rng: np.random.Generator | int | None = None,
) -> LinearFitResult:
    """Fit linear trajectory to posterior using linear regression.

    Fits a straight line to the decoded position along the track, providing
    slope (replay speed) and intercept estimates. Optionally uses Monte Carlo
    sampling to quantify uncertainty in the slope estimate.

    Parameters
    ----------
    env : Environment
        The environment the posterior was decoded on: a 1-D environment or a
        linearized track. Positions are read along the track (the linear
        position of a track, or the coordinate of a 1-D environment), so the
        slope is a physical speed and does not depend on how bins are
        numbered.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution over spatial bins for each time bin.
        Each row should sum to 1.
    times : NDArray[np.float64], shape (n_time_bins,)
        Time values for each time bin. Used as the independent variable (x)
        in the regression. Need not be uniformly spaced.
    n_samples : int, optional
        Number of Monte Carlo samples from the posterior for uncertainty
        estimation (only used when method="sample"). Default is 1000.
    method : {"map", "sample"}, optional
        How to fit the linear trajectory:
        - "map": Use the center of the most probable bin. Fast but ignores
          uncertainty.
        - "sample" (default): Sample from posterior, fit to each sample,
          average coefficients. Provides uncertainty estimate via slope_std.
    rng : np.random.Generator | int | None, optional
        Random number generator for reproducibility (method="sample" only).
        - If Generator: Use directly
        - If int: Seed for np.random.default_rng()
        - If None: Use default RNG (not reproducible)

    Returns
    -------
    LinearFitResult
        Container with slope (environment units per second), intercept
        (environment units), r_squared, and slope_std. slope_std is None
        when method="map".

    Raises
    ------
    ValueError
        If method is not "map" or "sample".
        If ``env`` is neither 1-D nor a linearized track.
        If posterior is not 2D or does not have one column per bin.
        If times length doesn't match posterior shape.

    See Also
    --------
    LinearFitResult : Container for linear fit results.
    fit_isotonic_trajectory : For monotonic (non-linear) fits.

    Notes
    -----
    **Slope interpretation**: The slope is replay speed along the track, in
    environment units per second (cm/s for a track in cm). Bins of unequal
    length are placed at their centers, so a constant-speed sweep gives a
    straight line. On a linearized track the position includes any
    ``edge_spacing`` gap between segments.

    **Sampling implementation** (for method="sample"):

    Each sample draws one bin per time bin from the posterior, by inverting
    the cumulative posterior with bins in track order, and takes that bin's
    center. A line is fit to each sampled sequence; the result averages the
    coefficients and ``slope_std`` is the spread of the sampled slopes.

    **Edge cases**:

    - **Constant positions**: If all decoded positions are identical, R² = 1.0
      (perfect fit to horizontal line). Check if slope ≈ 0 to detect this case.
    - **Constant times**: If all time values are identical, slope is undefined
      and set to 0.0 with intercept = mean position.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding.trajectory import fit_linear_trajectory
    >>>
    >>> # A 100 cm track in 2 cm bins and a posterior that sweeps along it
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=2.0, units="cm"
    ... )
    >>> n_time_bins = 20
    >>> posterior = np.zeros((n_time_bins, env.n_bins))
    >>> for t in range(n_time_bins):
    ...     posterior[t, t * 2] = 1.0  # 4 cm per time bin
    >>> times = np.linspace(0, 1, n_time_bins)
    >>>
    >>> result = fit_linear_trajectory(env, posterior, times, method="map")
    >>> print(f"speed = {result.slope:.0f} cm/s, R² = {result.r_squared:.3f}")
    speed = 76 cm/s, R² = 1.000
    """
    if method not in ("map", "sample"):
        raise ValueError(f"method must be 'map' or 'sample', got {method!r}")

    posterior, starts, stops = _track_positions(
        env, posterior, "fit_linear_trajectory()"
    )
    centers = (starts + stops) / 2
    n_time_bins = posterior.shape[0]

    if len(times) != n_time_bins:
        raise ValueError(
            f"times length ({len(times)}) must match posterior shape[0] ({n_time_bins})"
        )

    if method == "map":
        positions = _decoded_positions(posterior, centers, "map")

        # Simple linear regression
        slope, intercept = _fit_line(times, positions)
        predicted = slope * times + intercept
        r_squared = _compute_r_squared(positions, predicted)

        return LinearFitResult(
            slope=float(slope),
            intercept=float(intercept),
            r_squared=float(r_squared),
            slope_std=None,
        )

    # method == "sample"
    from neurospatial.stats import _ensure_rng

    rng_gen = _ensure_rng(rng)

    # Invert the cumulative posterior (bins in track order); this is
    # numerically stable for peaky posteriors.
    cumsum = np.cumsum(posterior, axis=1)
    u = rng_gen.random((n_samples, n_time_bins, 1))
    # Shape: (n_samples, n_time_bins)
    sampled_positions = centers[np.argmax(cumsum >= u, axis=-1)]

    # Fit line to each sample
    slopes = np.empty(n_samples)
    intercepts = np.empty(n_samples)

    for i in range(n_samples):
        slopes[i], intercepts[i] = _fit_line(times, sampled_positions[i])

    # Average coefficients
    mean_slope = np.mean(slopes)
    mean_intercept = np.mean(intercepts)
    slope_std = np.std(slopes, ddof=1)

    # Compute R² using mean coefficients on the posterior mean positions
    expected_positions = _decoded_positions(posterior, centers, "expected")
    predicted = mean_slope * times + mean_intercept
    r_squared = _compute_r_squared(expected_positions, predicted)

    return LinearFitResult(
        slope=float(mean_slope),
        intercept=float(mean_intercept),
        r_squared=float(r_squared),
        slope_std=float(slope_std),
    )


def _fit_line(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float]:
    """Fit a line y = slope*x + intercept using least squares.

    Parameters
    ----------
    x : NDArray[np.float64]
        Independent variable.
    y : NDArray[np.float64]
        Dependent variable.

    Returns
    -------
    tuple[float, float]
        (slope, intercept) of the fitted line.
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Compute slope and intercept
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        # All x values are the same - slope is undefined
        return 0.0, float(y_mean)

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    return float(slope), float(intercept)


def _posterior_image(
    posterior: NDArray[np.float64],
    starts: NDArray[np.float64],
    stops: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Posterior mass on equally spaced columns along the track.

    Parameters
    ----------
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior mass per bin, bins in track order.
    starts, stops : NDArray[np.float64], shape (n_bins,)
        Where each bin starts and ends along the track, in track order.

    Returns
    -------
    NDArray[np.float64], shape (n_time_bins, n_columns)
        Posterior mass per column. Contiguous bins of one length are the
        columns themselves. Otherwise columns are as long as the shortest
        bin, shortened so that whole columns span the track, and each holds
        the mass of the overlapping bins in proportion to the overlap: the
        posterior density (mass per unit length) is constant within a bin,
        and a gap between bins holds no mass.
    """
    lengths = stops - starts
    shortest = float(lengths.min())
    tolerance = 1e-9 * shortest
    if np.all(np.abs(lengths - shortest) <= tolerance) and np.all(
        np.abs(starts[1:] - stops[:-1]) <= tolerance
    ):
        return posterior
    span = float(stops[-1] - starts[0])
    n_columns = int(np.ceil(span / shortest - 1e-9))
    edges = np.linspace(starts[0], stops[-1], n_columns + 1)
    overlap = np.clip(
        np.minimum(stops[:, None], edges[None, 1:])
        - np.maximum(starts[:, None], edges[None, :-1]),
        0.0,
        None,
    )  # (n_bins, n_columns)
    image: NDArray[np.float64] = posterior @ (overlap / lengths[:, None])
    return image


def detect_trajectory_radon(
    env: Environment,
    posterior: NDArray[np.float64],
    *,
    theta_range: tuple[float, float] = (-90, 90),
    theta_step: float = 1.0,
) -> RadonDetectionResult:
    """Detect linear trajectory using Radon transform.

    Treats the posterior as a 2D image (time × position along the track) and
    finds the line with maximum integrated probability mass. This is
    particularly useful for detecting diagonal stripe patterns in replay
    posteriors.

    Parameters
    ----------
    env : Environment
        The environment the posterior was decoded on: a 1-D environment or a
        linearized track. Image columns run along the track, so the result
        does not depend on how bins are numbered.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution over spatial bins for each time bin.
        Treated as a 2D image where rows are time bins and columns are
        equally spaced positions along the track.
    theta_range : tuple[float, float], optional
        Range of angles to search in degrees. Default is (-90, 90).
        - 0° corresponds to a horizontal line (constant position).
        - 90° corresponds to a vertical line (instantaneous jump).
        - Positive angles indicate forward replay.
        - Negative angles indicate reverse replay.
    theta_step : float, optional
        Angular resolution in degrees. Default is 1.0.
        Smaller values give finer angle resolution but take longer to compute.

    Returns
    -------
    RadonDetectionResult
        Container with angle_degrees, score, offset, and sinogram.

    Raises
    ------
    ImportError
        If ``scikit-image`` is unavailable. It ships as a core dependency, so
        this indicates a broken install; repair with ``pip install scikit-image``.
    ValueError
        If ``env`` is neither 1-D nor a linearized track, or if ``posterior``
        does not have one column per bin.

    See Also
    --------
    RadonDetectionResult : Container for Radon detection results.
    fit_linear_trajectory : For parametric linear fits.
    fit_isotonic_trajectory : For monotonic trajectory fits.

    Notes
    -----
    The Radon transform computes line integrals of the posterior at all
    specified angles. The angle with the highest integrated mass indicates
    the dominant trajectory direction.

    **Interpretation of angles:**

    The detected angle is in Radon space, where:

    - θ = 0° means the line is perpendicular to the first axis (time axis),
      i.e., a horizontal line in the posterior (constant position).
    - θ = 90° means the line is perpendicular to the second axis (position axis),
      i.e., a vertical line (instantaneous position change).
    - θ ≈ 45° typically indicates forward replay (position increases with time).
    - θ ≈ -45° typically indicates reverse replay (position decreases with time).

    **Position axis:**

    A line in the image is a constant speed only if the columns are equally
    spaced along the track. When the bins are contiguous and of one length,
    each bin is a column, in track order. Otherwise the posterior is
    resampled onto columns as long as the shortest bin (shortened slightly
    so that whole columns span the track): each column receives the mass of
    the bins it overlaps, in proportion to the overlap, which treats the
    posterior density as constant within a bin. The shortest bin sets the
    spacing so that no bin is merged with its neighbor. Gaps between bins,
    such as ``edge_spacing`` between the segments of a track, are columns
    with no mass.

    **Time uniformity assumption:**

    Time bins are assumed to be uniformly spaced for image interpretation.
    For non-uniform times, consider interpolating the posterior to a uniform
    time grid first.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding.trajectory import detect_trajectory_radon
    >>>
    >>> # A 100 cm track in 2 cm bins and a posterior that sweeps along it
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=2.0, units="cm"
    ... )
    >>> posterior = np.eye(env.n_bins)  # one bin per time bin: a diagonal
    >>>
    >>> result = detect_trajectory_radon(env, posterior)
    >>> print(f"Detected angle: {result.angle_degrees:.1f}°")  # doctest: +SKIP
    Detected angle: 45.0°
    """
    if not _SKIMAGE_AVAILABLE:
        raise ImportError(
            "scikit-image is required for Radon transform trajectory detection. "
            "It ships as a core neurospatial dependency, so this usually means a "
            "broken install; repair with: pip install scikit-image"
        )

    posterior, starts, stops = _track_positions(
        env, posterior, "detect_trajectory_radon()"
    )
    image = _posterior_image(posterior, starts, stops)

    # Generate angle array
    theta_start, theta_end = theta_range
    # Use endpoint=False to get consistent behavior with theta_step
    n_angles = int((theta_end - theta_start) / theta_step)
    theta = np.linspace(theta_start, theta_end, n_angles, endpoint=False)

    # Compute Radon transform
    # The posterior is treated as an image with:
    # - rows = time bins (vertical axis in image)
    # - columns = positions along the track (horizontal axis in image)
    sinogram = radon(image, theta=theta, circle=False)

    # Find the peak in the sinogram
    # sinogram has shape (n_offsets, n_angles)
    peak_idx = np.unravel_index(np.argmax(sinogram), sinogram.shape)
    offset_idx, angle_idx = peak_idx

    # Get the detected angle
    angle_degrees = float(theta[angle_idx])

    # Get the score (peak value)
    score = float(sinogram[offset_idx, angle_idx])

    # Compute offset in the original coordinate system
    # The offset index corresponds to the distance from the center
    n_offsets = sinogram.shape[0]
    # Center offset (Radon transform centers the projection)
    center_offset = (n_offsets - 1) / 2
    offset = float(offset_idx - center_offset)

    # Transpose sinogram to match (n_angles, n_offsets) convention
    # (scikit-image's radon() returns (n_offsets, n_angles))
    sinogram_transposed = sinogram.T

    return RadonDetectionResult(
        angle_degrees=angle_degrees,
        score=score,
        offset=offset,
        sinogram=sinogram_transposed,
    )


__all__ = [
    "IsotonicFitResult",
    "LinearFitResult",
    "RadonDetectionResult",
    "detect_trajectory_radon",
    "fit_isotonic_trajectory",
    "fit_linear_trajectory",
]
