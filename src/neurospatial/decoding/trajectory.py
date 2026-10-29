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

A positive slope or speed, or an "increasing" fit, means that the decoded
position increases along the track (away from its start). Whether that is
forward or reverse replay depends on the direction the animal was running,
which these functions do not know: compare the sign with the animal's running
direction around the event.

The fits do not themselves detect the sharp-wave-ripple (SWR) events that
delimit replay. neurospatial does not implement ripple detection. Instead,
detect ripple intervals with the external ``ripple_detection`` package
(``pip install ripple_detection``), then decode the spikes within each
interval and fit the posterior with the functions in this module.

Its detectors return a pandas DataFrame with one row per event and
``start_time`` and ``end_time`` columns in seconds. The start times feed
directly into :func:`neurospatial.events.peri_event_histogram` as event
times to build a peri-ripple histogram. For example::

    from ripple_detection import Kay_ripple_detector, filter_ripple_band
    from neurospatial.events import peri_event_histogram

    filtered_lfps = filter_ripple_band(lfps, sampling_frequency)
    ripple_times = Kay_ripple_detector(time, filtered_lfps, speed, sampling_frequency)
    ripple_starts = ripple_times["start_time"].to_numpy()
    psth = peri_event_histogram(spike_times, ripple_starts, window=(-0.5, 0.5))

Functions
---------
fit_isotonic_trajectory
    Fit monotonic trajectory using isotonic regression.
fit_linear_trajectory
    Fit linear trajectory with optional Monte Carlo uncertainty.
detect_trajectory_radon
    Fit the replay line holding the most posterior mass (Davidson et al. 2009).
replay_trajectory_significance
    Test a replay line's score against column-cycle, unit-identity and
    pseudo-event shuffles.

Result Dataclasses
------------------
IsotonicFitResult
    Container for isotonic regression results.
LinearFitResult
    Container for linear regression results.
RadonDetectionResult
    Container for Radon transform detection results.
ReplaySignificanceResult
    Container for a replay line and its shuffle tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from numpy.typing import NDArray

    from neurospatial.encoding.spatial import SpatialRatesResult
    from neurospatial.environment import Environment
    from neurospatial.stats.shuffle import ShuffleTestResult


@dataclass(frozen=True, repr=False, eq=False)
class IsotonicFitResult(ReadOnlyArraysMixin, ResultMixin):
    """Container for isotonic regression trajectory fit results.

    Isotonic regression fits a monotonic (non-decreasing or non-increasing)
    function to the decoded position sequence. This is useful
    for detecting replay events that represent sequential traversal of
    spatial locations.

    Attributes
    ----------
    fitted_positions : NDArray[np.float64]
        Fitted monotonic positions along the track, in environment units,
        shape (n_time_bins,). These are constrained to be non-decreasing or
        non-increasing in time.
    r_squared : float
        Coefficient of determination (R²) measuring goodness of fit.
        Range [0, 1] where 1 indicates perfect fit. NaN when the decoded
        position never changes (a stationary decode), which has no variance
        to explain.
    direction : {"increasing", "decreasing"} or None
        Monotonic constraint of the reported fit. "increasing" means
        position along the track does not decrease with time, "decreasing"
        that it does not increase. None when the decoded position never
        changes, so the decode has no direction. Whether a direction is
        forward or reverse replay depends on the animal's running direction.
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
        slope means position increases along the track with time; whether
        that is forward or reverse replay depends on the animal's running
        direction.
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
    """The constant-speed line through a decoded event with the best replay score.

    :func:`detect_trajectory_radon` scores every candidate line, a speed and
    a start position, by the posterior mass within ``band`` of it (the replay
    score of Davidson et al. 2009) and keeps the best.

    Attributes
    ----------
    speed : float
        Replay speed along the track, in environment units per second (cm/s
        for a track in cm). Positive when position increases along the track
        with time.
    start_position : float
        Position along the track, in environment units, of the trajectory at
        the event's first time bin, ``times[0]``. It lies beyond an end of the
        track when the trajectory enters the track after the event starts.
    score : float
        Replay score of Davidson et al. (2009), the best over the searched
        trajectories: the mean over time bins of the posterior mass within
        ``band`` of the trajectory, in [0, 1]. The reported trajectory scores
        within 0.001 of it (see :func:`detect_trajectory_radon`). It is 1 when
        every time bin puts all its mass within ``band`` of the line, and
        about ``2 * band`` divided by the track length for a uniform posterior.
        A time bin where the line lies beyond the ends of the track contributes
        the median of that mass over positions on the track.
    band : float
        Half-width of the band around a line, in environment units.
    sinogram : NDArray[np.float64], shape (n_speeds, n_start_positions)
        Replay score of every candidate trajectory: entry ``(i, j)`` is the
        trajectory of speed ``sinogram_speeds[i]`` through
        ``sinogram_start_positions[j]`` at ``times[0]``; NaN for one that is
        never on the track.
    sinogram_speeds : NDArray[np.float64], shape (n_speeds,)
        Searched speeds, in environment units per second, evenly spaced.
    sinogram_start_positions : NDArray[np.float64], shape (n_start_positions,)
        Searched start positions along the track, in environment units,
        evenly spaced at most ``band / 5`` apart, reaching beyond both ends
        of the track as far as the fastest trajectory travels in the event.

    See Also
    --------
    detect_trajectory_radon : Function that produces this result.
    replay_trajectory_significance : Test the score against shuffles.
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
    band: float
    sinogram: NDArray[np.float64]
    sinogram_speeds: NDArray[np.float64]
    sinogram_start_positions: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Own read-only copies of the sinogram and its axes."""
        self._own_arrays()

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "speed": self.speed,
            "start_position": self.start_position,
            "score": self.score,
            "band": self.band,
        }


@dataclass(frozen=True, repr=False, eq=False)
class ReplaySignificanceResult(ResultMixin):
    """A replay line and its replay score tested against shuffles.

    :func:`replay_trajectory_significance` fits the replay line of Davidson,
    Kloosterman and Wilson (2009) and compares its replay score with the best
    score found in each of up to three kinds of shuffled data.

    Attributes
    ----------
    trajectory : RadonDetectionResult
        The best line through the event: speed, start position and replay
        score.
    shuffles : Mapping[str, ShuffleTestResult]
        One shuffle test per null that was run, keyed ``"column_cycle"``,
        ``"unit_identity"`` and ``"pseudo_event"``. Each holds the observed
        score, the best score of every shuffle and the Monte Carlo p-value
        ``(k + 1) / (n + 1)``.
    p_value : float
        The largest p-value among ``shuffles``. Davidson et al. call an event
        significant replay when this is below 0.01, with all three shuffles
        run.

    See Also
    --------
    replay_trajectory_significance : Function that produces this result.
    detect_trajectory_radon : The line fit alone.
    """

    trajectory: RadonDetectionResult
    shuffles: Mapping[str, ShuffleTestResult]
    p_value: float

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "speed": self.trajectory.speed,
            "start_position": self.trajectory.start_position,
            "score": self.trajectory.score,
            "p_value": self.p_value,
            **{f"{name}_p_value": test.p_value for name, test in self.shuffles.items()},
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
        within the tolerance a ``DecodingResult`` accepts (``1e-6``, which
        covers float32 rounding).
    """
    from neurospatial.decoding._result import _ROW_SUM_TOLERANCE

    posterior = np.asarray(posterior, dtype=np.float64)
    finite = np.isfinite(posterior).all(axis=1)
    negative = finite & (posterior < 0).any(axis=1)
    sums = posterior.sum(axis=1)
    unnormalized = finite & ~negative & (np.abs(sums - 1.0) > _ROW_SUM_TOLERANCE)
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
            f"{worst:.6g}, |row sum - 1| = {abs(worst - 1.0):.2g})"
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


_DAVIDSON_BAND_CM = 15.0
"""Default half-width ``d`` of the replay band, 15 cm.

Davidson, Kloosterman and Wilson (2009, Supplemental Experimental Procedures)
"empirically set [d] to 15 cm to allow for detection of replays with small
local variations in velocity".
"""

_DEFAULT_MAX_SPEED_CM = 3000.0
"""Default largest replay speed searched, 30 m/s, in cm/s.

Davidson et al. (2009, Figure 3E) found replay speeds up to about 20 m/s
(median 8.1 m/s); the search extends half as far again beyond that.
"""

_LINES_PER_BAND = 5
"""Candidate start positions per band half-width.

Start positions are at most ``band / 5`` apart and neighboring speeds end at
most that far apart, so every line is within ``band / 5`` of a searched one,
well inside the band, and the score does not depend on the bin size of the
decode.
"""

_SCORE_DECIMALS = 12
"""Decimals replay scores are rounded to, so that scores equal in exact
arithmetic but summed along different paths compare equal (in particular the
observed score and a shuffle's)."""

_SCORE_TIE = 1e-3
"""Replay scores within this of the best (a tenth of a percent of probability)
count as equally good when choosing the reported trajectory."""


def detect_trajectory_radon(
    env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    band: float | None = None,
    max_speed: float | None = None,
) -> RadonDetectionResult:
    """Detect the constant-speed replay trajectory in a decoded event.

    Implements the replay line search of Davidson, Kloosterman and Wilson
    (2009): every candidate trajectory, a speed and a start position, is
    scored by the mean over time bins of the posterior probability that the
    position lies within ``band`` of it, and the best trajectory gives the
    replay speed, start position and replay score. The search uses the whole
    posterior rather than one decoded position per time bin. Test the score
    against shuffles with :func:`replay_trajectory_significance`.

    Parameters
    ----------
    env : Environment
        The environment the posterior was decoded on: a 1-D environment or a
        linearized track. Positions are read along the track, so the result
        does not depend on how bins are numbered.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability mass over spatial bins for each time bin. Each
        row must sum to 1.
    times : NDArray[np.float64], shape (n_time_bins,)
        Center time of each posterior row, in seconds, strictly increasing.
        Times need not be evenly spaced, so time bins dropped as undecodable
        can be left out together with their posterior rows.
    band : float or None, default=None
        Half-width ``d`` of the band around a trajectory, in environment
        units. None means 15 cm, the value of Davidson et al., in the
        environment's units, which requires metric ``env.units``. A wider
        band tolerates small changes of speed within an event.
    max_speed : float or None, default=None
        Largest speed searched, in environment units per second, in either
        direction. None means 30 m/s in the environment's units: Davidson et
        al. found replay speeds up to about 20 m/s (median 8.1 m/s).

    Returns
    -------
    RadonDetectionResult
        The speed, start position and replay score of the best trajectory,
        and the score of every candidate (the sinogram) with its axes.

    Raises
    ------
    ValueError
        If ``env`` is neither 1-D nor a linearized track, if ``posterior``
        does not have one column per bin or a row is not a probability mass
        (NaN or Inf, negative, or not summing to 1), if ``times`` does not
        have one finite, strictly increasing entry per row, or if ``band`` or
        ``max_speed`` is not positive (or is None on an environment without
        metric units).

    See Also
    --------
    RadonDetectionResult : Container for the detected trajectory.
    replay_trajectory_significance : Test the replay score against shuffles.
    fit_linear_trajectory : For parametric linear fits.
    fit_isotonic_trajectory : For monotonic trajectory fits.

    Notes
    -----
    **Score.** The replay score of Davidson et al. (2009, equation 5),

    .. math:: R(V, \\rho) = \\frac{1}{n} \\sum_{k=0}^{n-1}
              \\Pr\\left(|x - (\\rho + V (t_k - t_0))| \\le d\\right),

    the mean over the ``n`` time bins of the posterior mass within
    ``d = band`` of the trajectory of speed ``V`` starting at ``rho`` at the
    first time bin. Mass within a bin is spread evenly along it, so the mass
    within the band is exact for any bin sizes. For a time bin where the
    trajectory lies beyond the ends of the track, it specifies "any
    location" and the time bin contributes the median, over the searched
    start positions that lie on a bin, of the mass within ``band`` of that
    position, as in Davidson et al. The score is in [0, 1] and equal for a
    sweep reversed in position or time. Davidson et al. used evenly spaced
    20 ms time bins; ``times`` here may be uneven, and ``t_k`` is the time
    of each row.

    **Search.** Davidson et al. "densely sampled" speed and start position.
    Start positions are at most ``band / 5`` apart (3 cm for the default
    band), and speeds from ``-max_speed`` to ``max_speed`` are spaced so that
    neighboring trajectories end at most one start spacing apart. Every
    trajectory that lies on the track in at least one time bin is searched,
    including those that enter the track after the event starts, so the
    score is the same for an event reversed in time. Every trajectory is therefore within ``band / 5`` of
    a searched one at every time bin, well inside the band, and the score
    does not depend on the bin size of the decode. When the posterior is
    sharp, every trajectory within ``band`` of the decoded positions holds
    all of their mass and several share the best score. ``score`` is the
    best score (Davidson et al.'s :math:`R_{max}`); the reported speed and
    start position are those of the trajectory, among those scoring within
    0.001 of the best, whose speed is closest to the least-squares line
    through the posterior mean positions, and then whose start is closest
    to that line's start, so a stationary event has speed 0. Davidson et al.
    do not say how they chose among equally good trajectories. The reported speed is resolved to the speed spacing, at most
    ``band / (5 * (times[-1] - times[0]))``, and the start position to the
    start spacing, at most ``band / 5``.

    **Track geometry.** Positions are read along the track (the linear
    position of a linearized track). Gaps between bins, such as
    ``edge_spacing`` between the segments of a maze, count as track for the
    band but are not positions for the off-track median. Davidson et al.
    used a single linear track.

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
    ...     f"score {result.score:.2f} within {result.band:g} cm"
    ... )
    198 cm/s from 1 cm, score 1.00 within 15 cm
    """
    lines, posterior = _prepare_replay_lines(
        env,
        posterior,
        times,
        band=band,
        max_speed=max_speed,
        operation="detect_trajectory_radon()",
    )
    return _best_line(lines, posterior)


def _prepare_replay_lines(
    env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    band: float | None,
    max_speed: float | None,
    operation: str,
) -> tuple[_ReplayLines, NDArray[np.float64]]:
    """Validate a replay line search and build its scorer.

    Returns
    -------
    lines : _ReplayLines
        Scorer for the event's geometry.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        The validated posterior in bin numbering.
    """
    from neurospatial._units import default_length
    from neurospatial.environment._queries import require_track_extents

    # Validates the environment and the posterior rows.
    _, starts, stops = _track_positions(env, posterior, operation)
    posterior = np.asarray(posterior, dtype=np.float64)
    times = _require_times(times, len(posterior), operation)
    if np.any(np.diff(times) <= 0):
        raise ValueError(
            _format_error(
                f"{operation} needs strictly increasing times.",
                why="a trajectory's position at each time bin depends on the "
                "time since the first",
                fix="sort the posterior rows and their times together by time, "
                "and drop rows with duplicate times",
            )
        )
    if band is None:
        band = default_length(env, _DAVIDSON_BAND_CM, name="band", call=operation)
    if max_speed is None:
        max_speed = default_length(
            env, _DEFAULT_MAX_SPEED_CM, name="max_speed", call=operation
        )
    for name, value, example in (
        ("band", band, "band=15.0 for 15 cm on a track in cm"),
        ("max_speed", max_speed, "max_speed=3000.0 for 30 m/s on a track in cm"),
    ):
        if not (np.isfinite(value) and value > 0):
            raise ValueError(
                _format_error(
                    f"{operation} needs {name} > 0 in the environment's units, "
                    f"got {value:g}.",
                    why="band is the half-width of the band around a trajectory "
                    "and max_speed the largest speed searched",
                    fix=f"pass a positive value, e.g. {example}",
                )
            )
    bin_starts, _ = require_track_extents(
        env, operation, alternative=_TRACK_ALTERNATIVE
    )
    order = np.argsort(bin_starts, kind="stable")  # as in _track_positions
    lines = _ReplayLines(
        starts,
        stops,
        order,
        times - times[0],
        float(band),
        float(max_speed),
        operation,
    )
    return lines, posterior


def _best_line(
    lines: _ReplayLines, posterior: NDArray[np.float64]
) -> RadonDetectionResult:
    """The best-scoring trajectory of one event, ties broken by a line fit."""
    cumulative = lines.cumulative_mass(posterior[None])
    off_track = lines.off_track_mass(cumulative)
    sinogram = lines.sinograms(cumulative, off_track, best_only=False)[0]
    # Trajectories scoring within _SCORE_TIE of the best are equally good.
    # Among them, report the speed closest to the least-squares line through
    # the posterior mean positions, then the start closest to that line's
    # start. The score reported is the best one, as for the shuffles.
    best = float(np.nanmax(sinogram))
    speed_indices, start_indices = np.nonzero(sinogram >= best - _SCORE_TIE)
    mean_position = posterior[:, lines.order] @ lines.bin_centers
    fit_speed, fit_start = _fit_lines(lines.elapsed, mean_position)
    closest = np.lexsort(
        (
            np.abs(lines.start_positions[start_indices] - fit_start),
            np.abs(lines.speeds[speed_indices] - fit_speed),
        )
    )[0]
    speed_index, start_index = speed_indices[closest], start_indices[closest]
    return RadonDetectionResult(
        speed=float(lines.speeds[speed_index]),
        start_position=float(lines.start_positions[start_index]),
        score=best,
        band=lines.band,
        sinogram=sinogram,
        sinogram_speeds=lines.speeds,
        sinogram_start_positions=lines.start_positions,
    )


def replay_trajectory_significance(
    env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    band: float | None = None,
    max_speed: float | None = None,
    spike_counts: NDArray[np.int64] | None = None,
    encoding_models: NDArray[np.float64] | SpatialRatesResult | None = None,
    dt: float | None = None,
    prior: NDArray[np.float64] | None = None,
    pseudo_event_pool: NDArray[np.float64] | None = None,
    n_shuffles: int = 1500,
    rng: np.random.Generator | int | None = None,
) -> ReplaySignificanceResult:
    """Test whether a decoded event replays a constant-speed trajectory.

    Implements the replay test of Davidson, Kloosterman and Wilson (2009):
    the replay score of the best trajectory (:func:`detect_trajectory_radon`)
    is compared with the best score found, by the same search, in each of
    ``n_shuffles`` shuffled versions of the event. Davidson et al. used three
    shuffles, each controlling for a different way a high score can arise
    without replay:

    - ``"column_cycle"`` (always run): each time bin's posterior is shifted
      circularly along the track by a random distance, which keeps each
      decoded distribution but breaks the alignment of positions along a
      trajectory.
    - ``"unit_identity"`` (run when ``spike_counts``, ``encoding_models`` and
      ``dt`` are given): spike trains are reassigned to randomly permuted
      units' rate maps and decoded again, which keeps the spiking dynamics
      (bursts, sequential firing) but breaks the place code.
    - ``"pseudo_event"`` (run when ``pseudo_event_pool`` is given): each time
      bin's posterior is replaced by one drawn at random from other candidate
      events, which keeps the decoder's bias toward particular locations.

    Parameters
    ----------
    env : Environment
        The environment the posterior was decoded on: a 1-D environment or a
        linearized track.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability mass over spatial bins for each time bin of the
        event. Each row must sum to 1.
    times : NDArray[np.float64], shape (n_time_bins,)
        Center time of each posterior row, in seconds, strictly increasing.
    band, max_speed : float or None, default=None
        Half-width of the band around a trajectory and largest speed
        searched, in environment units (per second); None means 15 cm and
        30 m/s. See :func:`detect_trajectory_radon`.
    spike_counts : NDArray[np.int64] or None, shape (n_time_bins, n_neurons)
        Spike counts of the event, one row per posterior row. With
        ``encoding_models`` and ``dt``, runs the unit-identity shuffle.
    encoding_models : array of shape (n_neurons, n_bins), SpatialRatesResult, or None
        The rate maps ``posterior`` was decoded with, as passed to
        :func:`~neurospatial.decoding.decode_position`.
    dt : float or None
        Time-bin duration in seconds that ``posterior`` was decoded with.
    prior : NDArray[np.float64] or None
        The prior ``posterior`` was decoded with, if not the default.
    pseudo_event_pool : NDArray[np.float64] or None, shape (n_pool, n_bins)
        Posterior rows of the session's other candidate events, stacked. Runs
        the pseudo-event shuffle: each pseudo-event draws its time bins at
        random, with replacement, from these rows. Leave out the tested
        event's own rows.
    n_shuffles : int, default=1500
        Shuffles per null, the number of Davidson et al. A p-value is at least
        ``1 / (n_shuffles + 1)``.
    rng : np.random.Generator, int, or None, default=None
        Random number generator or seed; the nulls draw from it in the order
        listed above. Pass a seed for a reproducible result.

    Returns
    -------
    ReplaySignificanceResult
        The detected trajectory, one shuffle test per null run, and the
        largest of their p-values.

    Raises
    ------
    ValueError
        For the inputs :func:`detect_trajectory_radon` refuses; if only some
        of ``spike_counts``, ``encoding_models`` and ``dt`` are given; if
        decoding ``spike_counts`` with ``encoding_models``, ``dt`` and
        ``prior`` does not reproduce ``posterior``; if ``pseudo_event_pool``
        does not hold probability rows with one column per bin; or if
        ``n_shuffles`` is not a positive integer.

    See Also
    --------
    detect_trajectory_radon : The trajectory search alone.
    neurospatial.stats.shuffle_posterior_circular : The column-cycle shuffle.
    neurospatial.stats.shuffle_cell_identity : The unit-identity shuffle.

    Notes
    -----
    Every shuffle is searched over the same trajectories as the observed
    event, so the test accounts for the search. The p-values are
    ``(k + 1) / (n + 1)``, where ``k`` counts shuffles scoring at least the
    observed score (a Monte Carlo p-value; Davison and Hinkley 1997).
    Davidson et al. performed 1500 of each shuffle, called an event
    significant replay when all three p-values were below 0.01, and reported
    the largest; ``p_value`` is the largest among the nulls run here. The
    column-cycle shift is a whole number of bins along the track. The
    unit-identity null decodes the shuffled counts the way ``posterior`` was
    decoded, which is checked by decoding the unshuffled counts first. The
    cost grows with ``n_shuffles``, the number of time bins, the track
    length divided by ``band`` and ``max_speed`` times the event duration
    divided by ``band``.

    References
    ----------
    Davidson, T. J., Kloosterman, F., & Wilson, M. A. (2009). Hippocampal
    replay of extended experience. *Neuron*, 63(4), 497-507.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding import replay_trajectory_significance
    >>>
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=2.0, units="cm"
    ... )
    >>> times = np.arange(25) * 0.02
    >>> posterior = np.zeros((25, env.n_bins))
    >>> posterior[np.arange(25), 2 * np.arange(25)] = 1.0  # 4 cm per 20 ms
    >>> result = replay_trajectory_significance(
    ...     env, posterior, times, n_shuffles=199, rng=0
    ... )
    >>> print(f"{result.trajectory.speed:.0f} cm/s, p = {result.p_value:.3f}")
    202 cm/s, p = 0.005
    """
    from types import MappingProxyType

    from neurospatial.stats import _ensure_rng
    from neurospatial.stats.shuffle import (
        ShuffleTestResult,
        compute_shuffle_pvalue,
        compute_shuffle_zscore,
        shuffle_cell_identity,
        shuffle_posterior_circular,
    )

    operation = "replay_trajectory_significance()"
    if (
        isinstance(n_shuffles, bool)
        or not isinstance(n_shuffles, (int, np.integer))
        or n_shuffles < 1
    ):
        raise ValueError(
            f"{operation} needs n_shuffles to be a positive integer, got "
            f"{n_shuffles!r}. Davidson et al. used 1500 per null."
        )
    lines, posterior = _prepare_replay_lines(
        env,
        posterior,
        times,
        band=band,
        max_speed=max_speed,
        operation=operation,
    )
    unit_inputs = {
        "spike_counts": spike_counts,
        "encoding_models": encoding_models,
        "dt": dt,
    }
    missing = [name for name, value in unit_inputs.items() if value is None]
    if 0 < len(missing) < len(unit_inputs):
        raise ValueError(
            _format_error(
                f"{operation}: the unit-identity shuffle needs spike_counts, "
                f"encoding_models and dt, but {', '.join(missing)} "
                f"{'is' if len(missing) == 1 else 'are'} missing.",
                why="the shuffle decodes the event's spike counts again with "
                "permuted rate maps, as the posterior was decoded",
                fix="pass all three (and prior if the decode used one), or none "
                "to skip this shuffle",
            )
        )
    if pseudo_event_pool is not None:
        pool = np.asarray(pseudo_event_pool, dtype=np.float64)
        if pool.ndim != 2 or pool.shape[1] != env.n_bins or len(pool) == 0:
            raise ValueError(
                _format_error(
                    f"{operation}: pseudo_event_pool has shape {pool.shape}, "
                    f"but needs (n_pool, {env.n_bins}): posterior rows of other "
                    "candidate events on this environment.",
                    why="each pseudo-event draws its time bins from these rows",
                    fix="stack the posteriors of the session's other candidate "
                    "events, e.g. np.vstack([r.posterior for r in others])",
                )
            )
        pool = _require_probability_rows(pool, operation)

    trajectory = _best_line(lines, posterior)
    observed = trajectory.score
    generator = _ensure_rng(rng)
    nulls: dict[str, NDArray[np.float64]] = {}

    shifted = shuffle_posterior_circular(
        env, posterior, n_shuffles=n_shuffles, rng=generator
    )
    nulls["column_cycle"] = lines.best_scores_of(shifted, n_shuffles)

    if spike_counts is not None and encoding_models is not None and dt is not None:
        from neurospatial.decoding.posterior import decode_position

        counts = np.asarray(spike_counts)
        decoded = decode_position(
            env, counts, encoding_models, dt=dt, prior=prior
        ).posterior
        if decoded.shape != posterior.shape or not np.allclose(
            decoded, posterior, rtol=1e-5, atol=1e-7
        ):
            raise ValueError(
                _format_error(
                    f"{operation}: decoding spike_counts with encoding_models, "
                    "dt and prior does not reproduce posterior.",
                    why="the unit-identity null decodes permuted counts; it is "
                    "a valid null only if it decodes them the way the posterior "
                    "was decoded",
                    fix="pass the counts, rate maps (or rate result), dt and "
                    "prior that produced posterior",
                )
            )
        models = getattr(encoding_models, "firing_rates", encoding_models)
        permuted = (
            decode_position(
                env, shuffled, encoding_models, dt=dt, prior=prior
            ).posterior
            for shuffled, _ in shuffle_cell_identity(
                counts, np.asarray(models), n_shuffles=n_shuffles, rng=generator
            )
        )
        nulls["unit_identity"] = lines.best_scores_of(permuted, n_shuffles)

    if pseudo_event_pool is not None:
        draws = generator.integers(len(pool), size=(n_shuffles, len(posterior)))
        nulls["pseudo_event"] = lines.best_scores_of(
            (pool[rows] for rows in draws), n_shuffles
        )

    shuffles = {
        name: ShuffleTestResult(
            observed_score=observed,
            null_scores=scores,
            p_value=compute_shuffle_pvalue(observed, scores, tail="greater"),
            z_score=compute_shuffle_zscore(observed, scores),
            shuffle_type=name,
            n_shuffles=n_shuffles,
        )
        for name, scores in nulls.items()
    }
    return ReplaySignificanceResult(
        trajectory=trajectory,
        shuffles=MappingProxyType(shuffles),
        p_value=max(test.p_value for test in shuffles.values()),
    )


class _ReplayLines:
    """Score candidate replay trajectories through decoded events on one track.

    Built once for an event's geometry (bins, times, band, speeds) and applied
    to the event's posterior and to its shuffles. A trajectory's score is
    Davidson et al.'s (2009) replay score: the mean over time bins of the
    posterior mass within ``band`` of it, with the median of that mass over
    the track for a time bin where it lies beyond the ends of the track. The
    mass within a band is exact for posterior density that is constant within
    each bin. Scores are rounded to ``_SCORE_DECIMALS`` so that equal scores
    computed along different paths compare equal.

    Parameters
    ----------
    starts, stops : NDArray[np.float64], shape (n_bins,)
        Where each bin starts and ends along the track, in track order.
    order : NDArray[np.intp], shape (n_bins,)
        Bin numbers in track order.
    elapsed : NDArray[np.float64], shape (n_time_bins,)
        Time of each time bin since the first, in seconds.
    band : float
        Half-width of the band around a trajectory, in environment units.
    max_speed : float
        Largest speed searched, in environment units per second.
    operation : str
        The calling function, for the refusal message.
    """

    # Memory budgets, in array elements of 8 bytes: the nonzeros of one
    # sparse block of trajectories (building it takes several arrays of this
    # size), the nonzeros kept between batches of shuffles (about 100 MB),
    # and the cumulative masses of the shuffles scored together and the
    # scores of one block for them (about 32 MB each).
    _BLOCK_NONZEROS = 1_000_000
    _CACHED_NONZEROS = 8_000_000
    _BATCH_VALUES = 4_000_000
    _BLOCK_SCORES = 4_000_000
    # Largest search, in trajectories times time bins (about a minute of
    # scoring per 1000 shuffles); far beyond any replay event in seconds.
    _MAX_SEARCH = 1_000_000_000

    def __init__(
        self,
        starts: NDArray[np.float64],
        stops: NDArray[np.float64],
        order: NDArray[np.intp],
        elapsed: NDArray[np.float64],
        band: float,
        max_speed: float,
        operation: str,
    ) -> None:
        self.order = order
        self.elapsed = elapsed
        self.band = band
        self.track_start, self.track_end = float(starts[0]), float(stops[-1])
        length = self.track_end - self.track_start
        n_on_track = int(np.ceil(length * _LINES_PER_BAND / band - 1e-9))
        spacing = length / n_on_track
        # Neighboring speeds end one start spacing apart at the last time bin.
        duration = float(elapsed[-1])
        speed_step = spacing / duration
        n_speeds = int(np.floor(max_speed / speed_step + 1e-9))
        self.speeds: NDArray[np.float64] = speed_step * np.arange(
            -n_speeds, n_speeds + 1, dtype=np.float64
        )
        # Start positions extend beyond the track as far as the fastest
        # trajectory travels, so a trajectory may enter the track as well as
        # leave it; only those on the track at some time bin are scored.
        n_beyond = int(np.ceil(n_speeds * speed_step * duration / spacing - 1e-9))
        self.start_positions: NDArray[np.float64] = (
            self.track_start
            + (np.arange(-n_beyond, n_on_track + n_beyond, dtype=np.float64) + 0.5)
            * spacing
        )
        search = len(self.speeds) * len(self.start_positions) * len(elapsed)
        if search > self._MAX_SEARCH:
            raise ValueError(
                _format_error(
                    f"{operation} would score {len(self.speeds)} speeds x "
                    f"{len(self.start_positions)} start positions over "
                    f"{len(elapsed)} time bins, more than {self._MAX_SEARCH:.0e}: "
                    f"the event spans {duration:g} s and speeds reach "
                    f"{max_speed:g} units/s, in steps of {speed_step:.3g}.",
                    why="neighboring trajectories are spaced band / 5 apart, so "
                    "the search grows with the event's duration times "
                    "max_speed, over band",
                    fix="check that times are in seconds (a replay event spans "
                    "tens to hundreds of milliseconds); otherwise lower "
                    "max_speed or widen band",
                )
            )
        # The cumulative mass is known at each bin's start and stop and grows
        # linearly within a bin; it stays flat across a gap between bins.
        self.knots = np.column_stack([starts, stops]).ravel()
        self.bin_centers = (starts + stops) / 2
        # The off-track median is over the start positions that lie on a bin.
        within = np.searchsorted(stops, self.start_positions, side="left")
        on_bin = (within < len(starts)) & (
            self.start_positions >= starts[np.minimum(within, len(starts) - 1)]
        )
        self.on_bin_positions = self.start_positions[on_bin]
        self._blocks: list[_LineBlock] | None = None

    def cumulative_mass(self, posteriors: NDArray[np.float64]) -> NDArray[np.float64]:
        """Posterior mass from the track start to each bin's start and stop.

        Parameters
        ----------
        posteriors : NDArray[np.float64], shape (n_events, n_time_bins, n_bins)
            Posteriors in bin numbering.

        Returns
        -------
        NDArray[np.float64], shape (n_events, n_time_bins, 2 * n_bins)
            Mass before each of ``self.knots``.
        """
        mass = posteriors[..., self.order]
        after = np.cumsum(mass, axis=-1)
        cumulative: NDArray[np.float64] = np.stack([after - mass, after], axis=-1)
        return cumulative.reshape(*mass.shape[:-1], -1)

    def _locate(
        self, positions: NDArray[np.float64]
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        """Knot interval and fraction along it of each position, held to the track."""
        positions = np.clip(positions, self.track_start, self.track_end)
        left = np.clip(
            np.searchsorted(self.knots, positions, side="right") - 1,
            0,
            len(self.knots) - 2,
        )
        span = self.knots[left + 1] - self.knots[left]
        fraction = np.divide(
            positions - self.knots[left],
            span,
            out=np.zeros_like(positions),
            where=span > 0,
        )
        return left, fraction

    def off_track_mass(self, cumulative: NDArray[np.float64]) -> NDArray[np.float64]:
        """Median over the track of the mass within ``band`` of a position.

        Davidson et al. give a time bin where a trajectory leaves the track the
        median probability over all locations on the track.

        Returns
        -------
        NDArray[np.float64], shape (n_events, n_time_bins)
        """
        mass = np.zeros((*cumulative.shape[:-1], len(self.on_bin_positions)))
        for sign in (1.0, -1.0):
            left, fraction = self._locate(self.on_bin_positions + sign * self.band)
            mass += sign * (
                (1.0 - fraction) * cumulative[..., left]
                + fraction * cumulative[..., left + 1]
            )
        median: NDArray[np.float64] = np.median(mass, axis=-1)
        return median

    def sinograms(
        self,
        cumulative: NDArray[np.float64],
        off_track: NDArray[np.float64],
        *,
        best_only: bool,
    ) -> NDArray[np.float64]:
        """Score every candidate trajectory in every event.

        Parameters
        ----------
        cumulative : NDArray[np.float64], shape (n_events, n_time_bins, 2 * n_bins)
            From :meth:`cumulative_mass`.
        off_track : NDArray[np.float64], shape (n_events, n_time_bins)
            Mass given to a time bin where a trajectory lies off the track.
        best_only : bool
            Return each event's best score instead of every trajectory's.

        Returns
        -------
        NDArray[np.float64]
            Shape ``(n_events, n_speeds, n_start_positions)``, NaN for a
            trajectory that never lies on the track, or ``(n_events,)`` when
            ``best_only``.
        """
        n_events, n_time = off_track.shape
        # One column per event, laid out for the sparse product: cumulative
        # masses, then the off-track mass summed over the time bins before
        # each time bin.
        n_cumulative = cumulative[0].size
        values = np.zeros((n_cumulative + n_time + 1, n_events))
        values[:n_cumulative] = cumulative.reshape(n_events, -1).T
        np.cumsum(off_track.T, axis=0, out=values[n_cumulative + 1 :])
        best = np.full(n_events, -np.inf)
        scores = (
            None
            if best_only
            else np.full(
                (n_events, self.speeds.size * self.start_positions.size), np.nan
            )
        )
        for block in self._line_blocks(n_events):
            block_scores = block.operator @ values  # (n_lines, n_events)
            if scores is None:
                np.maximum(best, block_scores.max(axis=0), out=best)
            else:
                scores[:, block.lines] = block_scores.T
        if scores is None:
            return _round_scores(best)
        return _round_scores(
            scores.reshape(n_events, self.speeds.size, self.start_positions.size)
        )

    def _line_blocks(self, n_events: int) -> Iterator[_LineBlock]:
        """Blocks of trajectories with their sparse operators.

        Blocks keep their nonzeros, and their scores for ``n_events`` events,
        within the memory budgets. They are kept between calls when all of
        them fit ``_CACHED_NONZEROS``.
        """
        if self._blocks is not None:
            yield from self._blocks
            return
        n_starts, n_time = self.start_positions.size, self.elapsed.size
        step = max(
            1,
            min(
                self._BLOCK_NONZEROS // (n_starts * n_time * 4),
                self._BLOCK_SCORES // (n_starts * n_events),
            ),
        )
        built: list[_LineBlock] | None = []
        nonzeros = 0
        for first_speed in range(0, self.speeds.size, step):
            block = self._block(slice(first_speed, first_speed + step))
            if built is not None:
                nonzeros += block.operator.nnz
                if nonzeros <= self._CACHED_NONZEROS:
                    built.append(block)
                else:
                    built = None
            yield block
        if built is not None:
            self._blocks = built

    def _block(self, speeds: slice) -> _LineBlock:
        """The trajectories of ``self.speeds[speeds]`` that touch the track.

        The operator's row ``r`` sums, over time bins on the track, the mass
        within ``band`` of trajectory ``lines[r]``, divided by the number of
        time bins: the cumulative mass at the band's upper end minus that at
        its lower end, each interpolated between bin edges and held flat
        beyond the track.
        """
        from scipy import sparse

        n_starts, n_time = self.start_positions.size, self.elapsed.size
        n_knots = len(self.knots)
        position = (
            self.start_positions[None, :, None]
            + self.speeds[speeds][:, None, None] * self.elapsed[None, None, :]
        )  # (n_speeds_in_block, n_start_positions, n_time)
        on_track = (position >= self.track_start) & (position <= self.track_end)
        position = position.reshape(-1, n_time)
        on_track = on_track.reshape(-1, n_time)
        touches = on_track.any(axis=1)
        position, on_track = position[touches], on_track[touches]
        lines = speeds.start * n_starts + np.flatnonzero(touches)
        # A trajectory is monotonic, so its time bins on the track are a run.
        first = np.argmax(on_track, axis=1)
        last = n_time - 1 - np.argmax(on_track[:, ::-1], axis=1)
        row, time = np.nonzero(on_track)
        row_parts, column_parts, weight_parts = [], [], []
        for sign in (1.0, -1.0):
            left, fraction = self._locate(position[row, time] + sign * self.band)
            base = time * n_knots + left
            for offset, weight in ((0, 1.0 - fraction), (1, fraction)):
                row_parts.append(row)
                column_parts.append(base + offset)
                weight_parts.append(sign * weight / n_time)
        # Off-track time bins are those before the first and after the last
        # on the track: prefix sums of the off-track mass at first, n_time and
        # last + 1.
        prefix = n_time * n_knots
        rows = np.arange(lines.size)
        for prefix_column, sign in ((first, 1.0), (n_time, 1.0), (last + 1, -1.0)):
            row_parts.append(rows)
            column_parts.append(prefix + np.broadcast_to(prefix_column, rows.shape))
            weight_parts.append(np.full(lines.size, sign / n_time))
        operator = sparse.csr_matrix(
            (
                np.concatenate(weight_parts),
                (np.concatenate(row_parts), np.concatenate(column_parts)),
            ),
            shape=(lines.size, prefix + n_time + 1),
        )
        return _LineBlock(lines=lines, operator=operator)

    def best_scores_of(self, posteriors: Any, n_shuffles: int) -> NDArray[np.float64]:
        """Best replay score of each of ``n_shuffles`` posteriors from an iterable.

        Parameters
        ----------
        posteriors : iterable of NDArray[np.float64], each (n_time_bins, n_bins)
            Shuffled posteriors in bin numbering.
        n_shuffles : int
            Number of posteriors the iterable yields.

        Returns
        -------
        NDArray[np.float64], shape (n_shuffles,)
        """
        values_per_event = len(self.elapsed) * (len(self.knots) + 1)
        batch = min(n_shuffles, max(1, self._BATCH_VALUES // values_per_event))
        scores = np.empty(n_shuffles)
        pending = np.empty((batch, len(self.elapsed), len(self.order)))
        filled = 0
        for index, shuffled in enumerate(posteriors):
            pending[index - filled] = shuffled
            if index + 1 - filled == batch:
                scores[filled : index + 1] = self._best(pending)
                filled = index + 1
        if filled < n_shuffles:
            scores[filled:] = self._best(pending[: n_shuffles - filled])
        return scores

    def _best(self, posteriors: NDArray[np.float64]) -> NDArray[np.float64]:
        cumulative = self.cumulative_mass(posteriors)
        return self.sinograms(
            cumulative, self.off_track_mass(cumulative), best_only=True
        )


@dataclass(frozen=True)
class _LineBlock:
    """Trajectories that touch the track, and their scoring operator.

    Attributes
    ----------
    lines : NDArray[np.intp], shape (n_lines,)
        Flat index ``speed * n_start_positions + start`` of each trajectory.
    operator : scipy.sparse.csr_matrix
        Shape ``(n_lines, n_time_bins * n_knots + n_time_bins + 1)``: maps the
        cumulative masses and the prefix sums of the off-track mass to each
        trajectory's replay score.
    """

    lines: NDArray[np.intp]
    operator: Any


def _round_scores(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """Replay scores rounded to ``_SCORE_DECIMALS`` and held to [0, 1]."""
    rounded: NDArray[np.float64] = np.clip(np.round(scores, _SCORE_DECIMALS), 0.0, 1.0)
    return rounded


__all__ = [
    "IsotonicFitResult",
    "LinearFitResult",
    "RadonDetectionResult",
    "ReplaySignificanceResult",
    "detect_trajectory_radon",
    "fit_isotonic_trajectory",
    "fit_linear_trajectory",
    "replay_trajectory_significance",
]
