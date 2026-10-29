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
        Range [0, 1] where 1 indicates perfect fit. NaN when the decoded
        position never changes (a stationary decode), which has no variance
        to explain.
    direction : {"increasing", "decreasing"} or None
        Monotonic constraint of the reported fit. "increasing" means
        position along the track does not decrease with time, "decreasing"
        that it does not increase. None when the decoded position never
        changes, so the decode has no direction.
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
    direction: Literal["increasing", "decreasing"] | None
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
        Coefficient of determination (R²) measuring goodness of fit, at
        most 1 (a perfect linear relationship). NaN when the decoded
        position never changes (a stationary decode), which has no variance
        to explain.
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
    """The line through a decoded event that holds the most posterior mass.

    :func:`detect_trajectory_radon` treats the posterior as an image, one row
    per time bin and one column per equal length of track, and scores every
    candidate line by the posterior mass it passes through.

    Attributes
    ----------
    speed : float
        Slope of the detected line: replay speed along the track in
        environment units per second (cm/s for a track in cm). Positive when
        position increases along the track with time. Equal to
        ``tan(angle) * column_width / time_bin_size``, so it does not depend
        on the bin size or time bin of the decode.
    start_position : float
        Position along the track, in environment units, of the detected line
        at the event's first time bin, ``times[0]``. It can lie off the track
        when the line enters the track after the event starts.
    score : float
        Mean over time bins of the posterior mass on the detected line, in
        [0, 1]: 1 when every time bin puts all its mass on the line, and
        ``column_width`` divided by the track length for a uniform
        posterior. A time bin contributes the mass in a window one column
        wide centered on the line, which is zero where the line lies off the
        track.
    angle : float
        Angle of the detected line in the image, in radians: 0 is a constant
        position, positive angles a position that increases with time. It
        depends on ``time_bin_size`` and ``column_width``; compare ``speed``
        across decodes instead.
    sinogram : NDArray[np.float64], shape (n_angles, n_positions)
        Score of every candidate line: row ``i`` holds the lines at angle
        ``sinogram_angles[i]``, column ``j`` the line through
        ``sinogram_positions[j]`` at the event's middle time bin.
    sinogram_angles : NDArray[np.float64], shape (n_angles,)
        Angle of each sinogram row, in radians, in the image.
    sinogram_positions : NDArray[np.float64], shape (n_positions,)
        Position along the track, in environment units, of each sinogram
        column's lines at the middle time bin, ``times[(n_time_bins - 1) //
        2]``: the centers of the image columns.
    time_bin_size : float
        Seconds per image row: the median spacing of ``times``.
    column_width : float
        Environment units per image column.

    See Also
    --------
    detect_trajectory_radon : Function that produces this result.
    fit_linear_trajectory : Line fit to the decoded positions.

    Examples
    --------
    >>> result = detect_trajectory_radon(env, posterior, times)  # doctest: +SKIP
    >>> print(
    ...     f"{result.speed:.0f} cm/s from {result.start_position:.0f} cm, "
    ...     f"score {result.score:.2f}"
    ... )  # doctest: +SKIP
    """

    speed: float
    start_position: float
    score: float
    angle: float
    sinogram: NDArray[np.float64]
    sinogram_angles: NDArray[np.float64]
    sinogram_positions: NDArray[np.float64]
    time_bin_size: float
    column_width: float

    def __post_init__(self) -> None:
        """Own read-only copies of the sinogram and its axes."""
        self._own_arrays()

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "speed": self.speed,
            "start_position": self.start_position,
            "score": self.score,
            "angle": self.angle,
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
    starts: NDArray[np.float64],
    stops: NDArray[np.float64],
    method: str,
) -> NDArray[np.float64]:
    """Position along the track per time bin: MAP bin center or posterior mean.

    Parameters
    ----------
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior mass per bin, bins in track order.
    starts, stops : NDArray[np.float64], shape (n_bins,)
        Where each bin starts and ends along the track, in track order.
    method : {"map", "expected"}
        ``"map"`` takes the center of the bin of highest density (mass per
        unit length along the track), as :func:`~neurospatial.decoding.map_bin`
        does, so a long bin does not win only by holding more track.
        ``"expected"`` takes the posterior mean position.

    Returns
    -------
    NDArray[np.float64], shape (n_time_bins,)
        Position along the track, in environment units.
    """
    from neurospatial.decoding.estimates import _density_mode

    centers = (starts + stops) / 2
    if method == "map":
        lengths = stops - starts
        positions: NDArray[np.float64] = centers[
            _density_mode(posterior, lengths / lengths.max())
        ]
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
        - "map": Use the center of the MAP bin, the bin of highest posterior
          density (mass per unit length along the track)
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
        a linearized track, if ``posterior`` does not have one column per
        bin, if a posterior row is not a probability mass (NaN or Inf,
        negative, or not summing to 1), or if ``times`` does not have one
        finite entry per row or holds a single distinct time.

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
    is the total sum of squares. If the decoded position never changes,
    :math:`SS_{tot} = 0`: R² is NaN and ``direction`` is None.

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
    times = _require_times(times, len(posterior), "fit_isotonic_trajectory()")
    positions = _decoded_positions(posterior, starts, stops, method)

    # Fit the requested direction, or both and keep the better fit.
    directions = [True, False] if increasing is None else [increasing]
    fits = [
        IsotonicRegression(increasing=direction).fit_transform(times, positions)
        for direction in directions
    ]
    r_squared = [_compute_r_squared(positions, fitted) for fitted in fits]
    best = 1 if len(fits) == 2 and r_squared[1] > r_squared[0] else 0
    # A decoded position that never changes has no direction.
    direction: Literal["increasing", "decreasing"] | None = None
    if np.ptp(positions) > 0:
        direction = "increasing" if directions[best] else "decreasing"
    return IsotonicFitResult(
        fitted_positions=fits[best],
        r_squared=r_squared[best],
        direction=direction,
        residuals=positions - fits[best],
    )


def _compute_r_squared(
    observed: NDArray[np.float64], predicted: NDArray[np.float64]
) -> float:
    """Compute R² (coefficient of determination), ``1 - SS_res / SS_tot``.

    Parameters
    ----------
    observed : NDArray[np.float64], shape (n_time_bins,)
        Observed values.
    predicted : NDArray[np.float64], shape (n_time_bins,)
        Predicted (fitted) values.

    Returns
    -------
    float
        At most 1, and in [0, 1] when ``predicted`` is the least-squares fit
        of a model that includes constants. NaN when ``observed`` never
        changes: with no variance to explain, R² is undefined, and a
        stationary decode must not pass a replay screen as a perfect fit.
    """
    if np.ptp(observed) == 0:
        return float("nan")
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
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
        estimation (only used when method="sample"), at least 2. Default is
        1000, which estimates slope_std to within about 2% (its relative
        standard error is about ``1 / sqrt(2 * (n_samples - 1))``).
    method : {"map", "sample"}, optional
        How to fit the linear trajectory:
        - "map": Use the center of the MAP bin, the bin of highest posterior
          density (mass per unit length along the track). Fast but ignores
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
        If ``times`` does not have one finite entry per posterior row, or
        holds a single distinct time.
        If ``n_samples`` is less than 2.
        If a posterior row is not a probability mass: NaN or Inf, negative,
        or not summing to 1.

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

    **Stationary decodes**: If the decoded position never changes, the slope
    is 0 and R² is NaN, so a stationary decode does not pass an R² screen.

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

    if n_samples < 2:
        raise ValueError(
            _format_error(
                f"fit_linear_trajectory() needs n_samples of at least 2, got "
                f"{n_samples}.",
                why="slope_std is the spread of the sampled slopes, which "
                "needs at least two samples",
                fix="keep the default n_samples=1000, or use method='map' for "
                "a fit without sampling",
            )
        )
    posterior, starts, stops = _track_positions(
        env, posterior, "fit_linear_trajectory()"
    )
    times = _require_times(times, len(posterior), "fit_linear_trajectory()")

    if method == "map":
        positions = _decoded_positions(posterior, starts, stops, "map")
        slope, intercept = _fit_lines(times, positions)
        r_squared = _compute_r_squared(positions, slope * times + intercept)
        return LinearFitResult(
            slope=float(slope),
            intercept=float(intercept),
            r_squared=r_squared,
            slope_std=None,
        )

    # method == "sample"
    from neurospatial.stats import _ensure_rng

    rng_gen = _ensure_rng(rng)

    # Invert each row's cumulative mass (bins in track order): the drawn bin
    # is the first whose cumulative mass exceeds u times the row's total, so
    # a bin with no mass is never drawn and rounding in the row sum cannot
    # push a draw past the last bin.
    cumsum = np.cumsum(posterior, axis=1)
    u = rng_gen.random((n_samples, len(posterior), 1)) * cumsum[:, -1:]
    centers = (starts + stops) / 2
    # Shape: (n_samples, n_time_bins)
    sampled_positions = centers[np.argmax(cumsum > u, axis=-1)]
    slopes, intercepts = _fit_lines(times, sampled_positions)

    # Average coefficients
    mean_slope = float(np.mean(slopes))
    mean_intercept = float(np.mean(intercepts))

    # Compute R² using mean coefficients on the posterior mean positions
    expected_positions = _decoded_positions(posterior, starts, stops, "expected")
    r_squared = _compute_r_squared(
        expected_positions, mean_slope * times + mean_intercept
    )

    return LinearFitResult(
        slope=mean_slope,
        intercept=mean_intercept,
        r_squared=r_squared,
        slope_std=float(np.std(slopes, ddof=1)),
    )


def _require_times(
    times: NDArray[np.float64], n_time_bins: int, operation: str
) -> NDArray[np.float64]:
    """Check one finite time per posterior row, with at least two distinct.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_time_bins,)
        Time of each posterior row, in seconds.
    n_time_bins : int
        Number of posterior rows.
    operation : str
        The calling function, for the refusal message.

    Returns
    -------
    NDArray[np.float64], shape (n_time_bins,)
        ``times`` as float64.

    Raises
    ------
    ValueError
        If ``times`` is not 1-D with one entry per row, holds NaN or Inf, or
        has a single distinct value.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) != n_time_bins:
        raise ValueError(
            _format_error(
                f"{operation}: times has {times.size} entries in shape "
                f"{times.shape}, but posterior has {n_time_bins} rows.",
                why="each posterior row is the decode at one time",
                fix="pass one time per posterior row, in seconds, with shape "
                "(n_time_bins,); drop the times of any rows you dropped",
            )
        )
    if not np.all(np.isfinite(times)):
        raise ValueError(
            _format_error(
                f"{operation}: times contains NaN or Inf at "
                f"{int(np.sum(~np.isfinite(times)))} of {n_time_bins} rows.",
                why="a fit along time needs the time of every posterior row",
                fix="drop those rows from both posterior and times",
            )
        )
    if np.ptp(times) == 0:
        raise ValueError(
            _format_error(
                f"{operation} needs at least two distinct times, but every "
                "time bin has the same time.",
                why="a line through identical times has no slope",
                fix="pass the center time of each posterior row, in seconds",
            )
        )
    return times


def _fit_lines(
    times: NDArray[np.float64], positions: NDArray[np.float64]
) -> tuple[Any, Any]:
    """Least-squares lines ``positions = slope * times + intercept``.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_time_bins,)
        At least two distinct times.
    positions : NDArray[np.float64], shape (..., n_time_bins)
        One position sequence per leading index.

    Returns
    -------
    slope, intercept : NDArray[np.float64] or float, shape (...)
        Coefficients of the line fitted to each sequence.
    """
    centered = times - times.mean()
    slope = (positions @ centered) / (centered @ centered)
    intercept = positions.mean(axis=-1) - slope * times.mean()
    return slope, intercept


def _posterior_image(
    posterior: NDArray[np.float64],
    starts: NDArray[np.float64],
    stops: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Posterior mass on equally spaced columns along the track.

    Parameters
    ----------
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior mass per bin, bins in track order.
    starts, stops : NDArray[np.float64], shape (n_bins,)
        Where each bin starts and ends along the track, in track order.

    Returns
    -------
    image : NDArray[np.float64], shape (n_time_bins, n_columns)
        Posterior mass per column; the first column starts at ``starts[0]``.
        Contiguous bins of one length are the columns themselves. Otherwise
        columns are as long as the shortest bin, shortened so that whole
        columns span the track, and each holds the mass of the overlapping
        bins in proportion to the overlap: the posterior density (mass per
        unit length) is constant within a bin, and a gap between bins holds
        no mass.
    column_width : float
        Length of each column along the track, in environment units.
    """
    lengths = stops - starts
    shortest = float(lengths.min())
    tolerance = 1e-9 * shortest
    if np.all(np.abs(lengths - shortest) <= tolerance) and np.all(
        np.abs(starts[1:] - stops[:-1]) <= tolerance
    ):
        return posterior, shortest
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
    return image, span / n_columns


def detect_trajectory_radon(
    env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    angle_range: tuple[float, float] = (-np.pi / 2, np.pi / 2),
    angle_step: float = np.pi / 180,
) -> RadonDetectionResult:
    """Detect the line through a decoded event that holds the most posterior mass.

    Treats the posterior as an image, one row per time bin and one column
    per equal length of track, and scores every candidate line by the mean,
    over time bins, of the posterior mass on the line: a discrete Radon
    transform with one sample per time bin. The best line gives the replay
    speed and start position without first reducing each time bin to one
    decoded position, so it uses the whole posterior.

    Parameters
    ----------
    env : Environment
        The environment the posterior was decoded on: a 1-D environment or a
        linearized track. Image columns run along the track, so the result
        does not depend on how bins are numbered.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability mass over spatial bins for each time bin. Each
        row must sum to 1.
    times : NDArray[np.float64], shape (n_time_bins,)
        Center time of each posterior row, in seconds, strictly increasing.
        Times need not be evenly spaced, so time bins dropped as undecodable
        can be left out together with their posterior rows.
    angle_range : tuple of float, default=(-pi/2, pi/2)
        Smallest and largest line angle to search, in radians, within
        [-pi/2, pi/2]. Angle 0 is a constant position and positive angles a
        position that increases with time. A line at exactly +/-pi/2 has no
        position in each time bin (an infinite speed) and is not searched.
        The default searches every direction.
    angle_step : float, default=pi/180
        Spacing of the searched angles, in radians, at most the width of
        ``angle_range``. Angles are ``angle_range[0] + k * angle_step`` up to
        ``angle_range[1]``. A step of one degree resolves the speed of a line
        at angle ``a`` to ``angle_step / (sin(a) * cos(a))`` of its value,
        3.5% at 45 degrees; the resolution is coarser for steep lines, where
        a smaller step or :func:`fit_linear_trajectory` gives a finer speed.

    Returns
    -------
    RadonDetectionResult
        The speed, start position, score and angle of the best line, and the
        score of every candidate line (the sinogram) with its axes.

    Raises
    ------
    ValueError
        If ``env`` is neither 1-D nor a linearized track, if ``posterior``
        does not have one column per bin or a row is not a probability mass
        (NaN or Inf, negative, or not summing to 1), if ``times`` does not
        have one finite, strictly increasing entry per row, or if
        ``angle_range`` or ``angle_step`` leaves no angle to search.

    See Also
    --------
    RadonDetectionResult : Container for Radon detection results.
    fit_linear_trajectory : For parametric linear fits.
    fit_isotonic_trajectory : For monotonic trajectory fits.

    Notes
    -----
    **Score.** Following the line-fitting replay score of Davidson,
    Kloosterman and Wilson (2009), each time bin contributes the posterior
    mass near the line at its center time, and the score is the mean over
    time bins. "Near" is a window one image column wide centered on the
    line: the posterior interpolated linearly between column centers. A
    line therefore scores the same whatever its slope, a position reversed
    along the track scores the same as the original, and the score is
    comparable across events of different durations. A time bin where the
    line lies off the track contributes no mass. Because the window is one
    column wide, a spread-out posterior scores higher on wider columns;
    compare scores between events decoded on the same bins, or against a
    shuffle of the same event (:func:`neurospatial.stats.shuffle_posterior_circular`).

    **Candidate lines.** At each searched angle the candidates are the
    lines through each column center at the middle time bin,
    ``times[(n_time_bins - 1) // 2]``, so every candidate lies on the track
    at mid-event; a line on the track for less than half the event is not
    considered. Ties go to the first line in sinogram order.

    **Image.** Rows are time bins, one ``time_bin_size`` (the median spacing
    of ``times``) apart. A line in the image is a constant speed only if the
    columns are equally spaced along the track. When the bins are
    contiguous and of one length, each bin is a column, in track order.
    Otherwise the posterior is resampled onto columns as long as the
    shortest bin (shortened slightly so that whole columns span the track):
    each column receives the mass of the bins it overlaps, in proportion to
    the overlap, which treats the posterior density as constant within a
    bin. Gaps between bins, such as ``edge_spacing`` between the segments of
    a track, are columns with no mass.

    References
    ----------
    Davidson, T. J., Kloosterman, F., & Wilson, M. A. (2009). Hippocampal
    replay of extended experience. *Neuron*, 63(4), 497-507.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding.trajectory import detect_trajectory_radon
    >>>
    >>> # A 100 cm track in 2 cm bins and a posterior that sweeps along it,
    >>> # one bin per 10 ms time bin
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=2.0, units="cm"
    ... )
    >>> posterior = np.eye(env.n_bins)
    >>> times = np.arange(env.n_bins) * 0.01
    >>>
    >>> result = detect_trajectory_radon(env, posterior, times)
    >>> print(
    ...     f"{result.speed:.0f} cm/s from {result.start_position:.0f} cm, "
    ...     f"score {result.score:.2f}"
    ... )
    200 cm/s from 1 cm, score 1.00
    """
    angles = _search_angles(angle_range, angle_step)
    posterior, starts, stops = _track_positions(
        env, posterior, "detect_trajectory_radon()"
    )
    times = _require_times(times, len(posterior), "detect_trajectory_radon()")
    if np.any(np.diff(times) <= 0):
        raise ValueError(
            _format_error(
                "detect_trajectory_radon() needs strictly increasing times.",
                why="each posterior row is one row of the time-by-position "
                "image, in time order",
                fix="sort the posterior rows and their times together by time, "
                "and drop rows with duplicate times",
            )
        )
    image, column_width = _posterior_image(posterior, starts, stops)
    time_bin_size = float(np.median(np.diff(times)))
    middle = (len(times) - 1) // 2
    sinogram = _line_scores(image, (times - times[middle]) / time_bin_size, angles)

    angle_index, column = np.unravel_index(np.argmax(sinogram), sinogram.shape)
    speed = float(np.tan(angles[angle_index]) * column_width / time_bin_size)
    positions = starts[0] + (np.arange(image.shape[1]) + 0.5) * column_width
    return RadonDetectionResult(
        speed=speed,
        start_position=float(positions[column] + speed * (times[0] - times[middle])),
        score=float(sinogram[angle_index, column]),
        angle=float(angles[angle_index]),
        sinogram=sinogram,
        sinogram_angles=angles,
        sinogram_positions=positions,
        time_bin_size=time_bin_size,
        column_width=column_width,
    )


def _search_angles(
    angle_range: tuple[float, float], angle_step: float
) -> NDArray[np.float64]:
    """Line angles ``low + k * step`` up to ``high``, strictly inside +/-pi/2.

    Raises
    ------
    ValueError
        If ``angle_range`` is not an increasing pair within [-pi/2, pi/2], if
        ``angle_step`` is not positive or exceeds the range, or if no angle
        remains strictly inside (-pi/2, pi/2).
    """
    low, high = (float(value) for value in angle_range)
    step = float(angle_step)
    half_turn = np.pi / 2 + 1e-12
    if not (-half_turn <= low < high <= half_turn):
        raise ValueError(
            _format_error(
                f"detect_trajectory_radon() needs angle_range = (low, high) "
                f"with -pi/2 <= low < high <= pi/2 radians, got ({low:g}, "
                f"{high:g}).",
                why="a line's angle a and a + pi describe the same line, so "
                "every line has one angle in [-pi/2, pi/2]",
                fix="pass the range in radians, e.g. angle_range=(0.0, "
                "np.pi / 2) for lines whose position increases with time",
            )
        )
    if not (np.isfinite(step) and 0 < step <= high - low):
        raise ValueError(
            _format_error(
                f"detect_trajectory_radon() needs 0 < angle_step <= "
                f"{high - low:g} (the width of angle_range), got {step:g}.",
                why="the search takes angles angle_step apart across "
                "angle_range, so a larger step leaves at most one angle",
                fix="pass the step in radians, e.g. angle_step=np.pi / 180 "
                "for one degree",
            )
        )
    n_angles = int(np.floor((high - low) / step + 1e-9)) + 1
    angles = low + step * np.arange(n_angles, dtype=np.float64)
    angles = angles[np.abs(angles) < np.pi / 2 - 1e-12]
    if angles.size == 0:
        raise ValueError(
            _format_error(
                f"detect_trajectory_radon(): angle_range ({low:g}, {high:g}) "
                f"with angle_step {step:g} searches only +/-pi/2.",
                why="a line at +/-pi/2 has no position in each time bin (an "
                "infinite speed), so it is not searched",
                fix="use a smaller angle_step, e.g. np.pi / 180",
            )
        )
    return angles


def _line_scores(
    image: NDArray[np.float64],
    rows: NDArray[np.float64],
    angles: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Mean posterior mass on every candidate line, the sinogram.

    Parameters
    ----------
    image : NDArray[np.float64], shape (n_rows, n_columns)
        Posterior mass per column, one row per time bin.
    rows : NDArray[np.float64], shape (n_rows,)
        Time of each row relative to the middle row, in rows.
    angles : NDArray[np.float64], shape (n_angles,)
        Line angles in the image, in radians, strictly inside +/-pi/2.

    Returns
    -------
    NDArray[np.float64], shape (n_angles, n_columns)
        Entry ``(i, j)`` is the mean over rows of the mass in a window one
        column wide centered on the line at ``angles[i]`` through column
        ``j``'s center at the middle row: the image interpolated linearly
        between column centers, and zero off the track.
    """
    n_rows, n_columns = image.shape
    # One empty column on each side: a line off the track holds no mass.
    padded = np.pad(image, ((0, 0), (1, 1)))
    row_index = np.arange(n_rows)
    columns = np.arange(n_columns, dtype=np.float64)[:, None]
    sinogram = np.empty((len(angles), n_columns))
    # One angle at a time keeps memory at n_columns * n_rows.
    for i, slope in enumerate(np.tan(angles)):
        position = columns + slope * rows  # (n_columns, n_rows), in columns
        left = np.floor(position)
        weight = position - left
        left_column = np.clip(left + 1, 0, n_columns + 1).astype(np.intp)
        right_column = np.clip(left + 2, 0, n_columns + 1).astype(np.intp)
        mass = (1.0 - weight) * padded[row_index, left_column] + weight * padded[
            row_index, right_column
        ]
        sinogram[i] = mass.mean(axis=1)
    return sinogram


__all__ = [
    "IsotonicFitResult",
    "LinearFitResult",
    "RadonDetectionResult",
    "detect_trajectory_radon",
    "fit_isotonic_trajectory",
    "fit_linear_trajectory",
]
