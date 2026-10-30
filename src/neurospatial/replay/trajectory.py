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
    Weighted isotonic fit of the MAP positions (Michon et al. 2019).
fit_linear_trajectory
    Line fit to positions sampled from the posterior (Karlsson & Frank
    2009) or to the MAP positions (Wu et al. 2017).
detect_trajectory_radon
    Fit the replay line holding the most posterior mass (Davidson et al. 2009).

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
    from collections.abc import Iterator

    from numpy.typing import NDArray

    from neurospatial.environment import Environment


@dataclass(frozen=True, repr=False, eq=False)
class IsotonicFitResult(ReadOnlyArraysMixin, ResultMixin):
    """A monotonic trajectory fit to a decoded event (Michon et al. 2019).

    :func:`fit_isotonic_trajectory` fits a non-decreasing or non-increasing
    position sequence to the decoded positions, weighting each time bin by
    the posterior probability of its MAP position, and scores the fit by
    the mean of those probabilities times the R² of the fit.

    Attributes
    ----------
    fitted_positions : NDArray[np.float64]
        Fitted monotonic positions along the track, in environment units,
        shape (n_time_bins,). These are constrained to be non-decreasing or
        non-increasing in time.
    r_squared : float
        Weighted coefficient of determination (R²) of the fit, with the same
        weights as the fit (the posterior probability of each time bin's MAP
        position). Range [0, 1] where 1 indicates a perfect fit. NaN when the
        decoded position never changes (a stationary decode), which has no
        variance to explain.
    mean_map_probability : float
        Mean over time bins of the posterior probability of each time bin's
        MAP bin, :math:`\\frac{1}{T}\\sum_t P_{MAP,t}`, in (0, 1]. Near 1 when
        every time bin decodes one bin with certainty.
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
    ...     f"score = {result.score:.3f}, direction = {result.direction}"
    ... )  # doctest: +SKIP
    >>> replay_speed_cm_s = np.diff(result.fitted_positions) / np.diff(
    ...     times
    ... )  # doctest: +SKIP
    """

    fitted_positions: NDArray[np.float64]
    r_squared: float
    mean_map_probability: float
    direction: Literal["increasing", "decreasing"] | None
    residuals: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Own read-only copies of fitted positions and residuals."""
        self._own_arrays()

    @property
    def score(self) -> float:
        """Replay score of Michon et al. (2019): ``mean_map_probability * r_squared``.

        :math:`\\frac{1}{T}\\sum_t P_{MAP,t} \\times R^2`, in [0, 1]; NaN for a
        stationary decode.
        """
        return self.mean_map_probability * self.r_squared

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "score": self.score,
            "r_squared": self.r_squared,
            "mean_map_probability": self.mean_map_probability,
            "direction": self.direction,
        }


@dataclass(frozen=True, repr=False)
class LinearFitResult(ResultMixin):
    """A straight-line trajectory fit to a decoded event.

    :func:`fit_linear_trajectory` fits one least-squares line of position on
    time, either to positions sampled from the posterior (Karlsson & Frank
    2009) or to the MAP positions (Wu et al. 2017). With several resamples,
    the slope, intercept and R² are means over resamples.

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
        Coefficient of determination (R²) of the line, in [0, 1] (1 for a
        perfect linear relationship); for ``method="sample"`` the mean over
        resamples of the R² of the line through each resample's sampled
        positions. NaN when the decoded or sampled position never changes
        (a stationary decode), which has no variance to explain.
    slope_std : float | None
        Standard deviation of the slope over resamples, in environment units
        per second: the Monte Carlo spread of the sampled fit, which shrinks
        as ``n_samples`` grows. It is not an uncertainty of the replay speed
        given the posterior. None for ``method="map"`` and for a single
        resample.

    See Also
    --------
    fit_linear_trajectory : Function that produces this result.
    IsotonicFitResult : For monotonic (non-linear) fits.

    Examples
    --------
    >>> result = fit_linear_trajectory(env, posterior, times, rng=42)  # doctest: +SKIP
    >>> print(
    ...     f"speed = {result.slope:.0f} cm/s, R² = {result.r_squared:.2f}"
    ... )  # doctest: +SKIP
    """

    slope: float
    intercept: float
    r_squared: float
    slope_std: float | None

    @property
    def score(self) -> float:
        """The replay score of this fit: its R², as in both papers."""
        return self.r_squared

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
        never on the track or is slower than the search's ``min_speed``.
    sinogram_speeds : NDArray[np.float64], shape (n_speeds,)
        Speeds of the sinogram's rows, in environment units per second,
        evenly spaced; rows slower than ``min_speed`` were not searched.
    sinogram_start_positions : NDArray[np.float64], shape (n_start_positions,)
        Searched start positions along the track, in environment units,
        evenly spaced at most ``band / 5`` apart, reaching beyond both ends
        of the track as far as the fastest trajectory travels in the event.

    See Also
    --------
    detect_trajectory_radon : Function that produces this result.
    neurospatial.replay.replay_significance : Test the score against shuffles.
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
    from neurospatial.environment import _require_track_extents

    starts, stops = _require_track_extents(
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
    from neurospatial.decoding import _ROW_SUM_TOLERANCE

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


def _map_bins(
    posterior: NDArray[np.float64],
    starts: NDArray[np.float64],
    stops: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Bin of highest posterior density (mass per unit length) per time bin.

    As :func:`~neurospatial.decoding.map_bin` does, so a long bin does not win
    only by holding more track.

    Parameters
    ----------
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior mass per bin, bins in track order.
    starts, stops : NDArray[np.float64], shape (n_bins,)
        Where each bin starts and ends along the track, in track order.

    Returns
    -------
    NDArray[np.int64], shape (n_time_bins,)
        Column of ``posterior`` holding each time bin's MAP bin.
    """
    from neurospatial.decoding import _density_mode

    lengths = stops - starts
    return _density_mode(posterior, lengths / lengths.max())


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
        ``"map"`` takes the center of the bin of highest density (see
        :func:`_map_bins`). ``"expected"`` takes the posterior mean position.

    Returns
    -------
    NDArray[np.float64], shape (n_time_bins,)
        Position along the track, in environment units.
    """
    centers = (starts + stops) / 2
    if method == "map":
        positions: NDArray[np.float64] = centers[_map_bins(posterior, starts, stops)]
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
    method: Literal["map", "expected"] = "map",
) -> IsotonicFitResult:
    """Fit a monotonic trajectory to a decoded event (Michon et al. 2019).

    Fits the decoded positions with weighted isotonic regression: the best
    non-decreasing (or non-increasing) position sequence in time, where each
    time bin is weighted by the posterior probability of its MAP position,
    so time bins decoded with confidence count more. The replay score of
    Michon et al. combines how sharp the decode is with how monotonic it is,

    .. math:: \\text{score} = \\frac{1}{T} \\sum_{t=1}^{T} P_{MAP,t}
              \\times R^2,

    where :math:`T` is the number of time bins and :math:`P_{MAP,t}` the
    posterior probability of time bin ``t``'s MAP bin. An increasing and a
    decreasing trajectory are both fit, and the one with the higher score is
    kept. Test the score against shuffles with
    :func:`~neurospatial.replay.replay_significance`.

    Parameters
    ----------
    env : Environment
        The environment the posterior was decoded on: a 1-D environment or a
        linearized track. Positions are read along the track (the linear
        position of a track, or the coordinate of a 1-D environment), so the
        fit does not depend on how bins are numbered.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability mass over spatial bins for each time bin. Each
        row must sum to 1.
    times : NDArray[np.float64], shape (n_time_bins,)
        Center time of each posterior row, in seconds; the independent
        variable of the fit.
    increasing : bool or None, default=None
        Direction of the fit. True fits a non-decreasing trajectory, False a
        non-increasing one, and None (Michon et al.) fits both and keeps the
        one with the higher score.
    method : {"map", "expected"}, default="map"
        The position fit at each time bin. ``"map"`` (Michon et al.) is the
        center of the MAP bin, the bin of highest posterior density (mass
        per unit length along the track). ``"expected"`` is the posterior
        mean position, a variant not in Michon et al.; the weights and the
        score are the same.

    Returns
    -------
    IsotonicFitResult
        The fitted positions, R², mean MAP probability and direction of the
        better fit; ``score`` is the replay score above.

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
    fit_linear_trajectory : For straight-line fits.
    neurospatial.replay.replay_significance : Test the score against shuffles.

    Notes
    -----
    **Fit.** The fit minimizes :math:`\\sum_t w_t (x_t - \\hat{x}_t)^2` over
    monotonic :math:`\\hat{x}` with weights :math:`w_t = P_{MAP,t}`, by the
    pool adjacent violators algorithm of scikit-learn's
    ``IsotonicRegression``.

    **R².** Michon et al. do not say whether R² is weighted. Here it uses the
    same weights as the fit,

    .. math:: R^2 = 1 - \\frac{\\sum_t w_t (x_t - \\hat{x}_t)^2}
              {\\sum_t w_t (x_t - \\bar{x}_w)^2},

    with :math:`\\bar{x}_w` the weighted mean position, so it is the share of
    the weighted variance that the weighted fit explains and lies in
    [0, 1]. If the decoded position never changes, R² is NaN and
    ``direction`` is None, so a stationary decode does not pass a score
    screen. Both directions share :math:`P_{MAP,t}`, so the direction with
    the higher score is the one with the higher R².

    **Time bins.** Michon et al. used 10 ms time bins and do not say how
    time bins without spikes were handled. Rows are fit as given: drop
    undecodable time bins, together with their times, before fitting.

    References
    ----------
    Michon, F., Sun, J.-J., Kim, C. Y., Ciliberti, D., & Kloosterman, F.
    (2019). Post-learning hippocampal replay selectively reinforces spatial
    memory for highly rewarded locations. *Current Biology*, 29(9),
    1436-1444. https://doi.org/10.1016/j.cub.2019.03.048

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import fit_isotonic_trajectory
    >>>
    >>> # A 100 cm track in 2 cm bins and a posterior that moves along it,
    >>> # holding 0.8 of its mass in one bin per 10 ms time bin
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=2.0, units="cm"
    ... )
    >>> n_time_bins = 20
    >>> posterior = np.full((n_time_bins, env.n_bins), 0.2 / (env.n_bins - 1))
    >>> posterior[np.arange(n_time_bins), 2 * np.arange(n_time_bins)] = 0.8
    >>> times = np.arange(n_time_bins) * 0.01
    >>>
    >>> result = fit_isotonic_trajectory(env, posterior, times)
    >>> print(
    ...     f"score = {result.score:.2f} (R² = {result.r_squared:.2f}, "
    ...     f"mean MAP probability = {result.mean_map_probability:.2f}), "
    ...     f"{result.direction}"
    ... )
    score = 0.80 (R² = 1.00, mean MAP probability = 0.80), increasing
    """
    from sklearn.isotonic import IsotonicRegression

    if method not in ("map", "expected"):
        raise ValueError(f"method must be 'map' or 'expected', got {method!r}")

    posterior, starts, stops = _track_positions(
        env, posterior, "fit_isotonic_trajectory()"
    )
    times = _require_times(times, len(posterior), "fit_isotonic_trajectory()")
    positions = _decoded_positions(posterior, starts, stops, method)
    weights = posterior[np.arange(len(posterior)), _map_bins(posterior, starts, stops)]

    # Fit the requested direction, or both and keep the higher score; both
    # share the MAP probabilities, so that is the higher R².
    directions = [True, False] if increasing is None else [increasing]
    fits = [
        IsotonicRegression(increasing=direction).fit_transform(
            times, positions, sample_weight=weights
        )
        for direction in directions
    ]
    r_squared = [_compute_r_squared(positions, fitted, weights) for fitted in fits]
    best = 1 if len(fits) == 2 and r_squared[1] > r_squared[0] else 0
    # A decoded position that never changes has no direction.
    direction: Literal["increasing", "decreasing"] | None = None
    if np.ptp(positions) > 0:
        direction = "increasing" if directions[best] else "decreasing"
    return IsotonicFitResult(
        fitted_positions=fits[best],
        r_squared=r_squared[best],
        mean_map_probability=float(np.mean(weights)),
        direction=direction,
        residuals=positions - fits[best],
    )


def _compute_r_squared(
    observed: NDArray[np.float64],
    predicted: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> float:
    """Compute R² (coefficient of determination), ``1 - SS_res / SS_tot``.

    Parameters
    ----------
    observed : NDArray[np.float64], shape (n_time_bins,)
        Observed values.
    predicted : NDArray[np.float64], shape (n_time_bins,)
        Predicted (fitted) values.
    weights : NDArray[np.float64] or None, shape (n_time_bins,)
        Positive weight of each value in both sums, about the weighted mean
        of ``observed``; None weighs every value the same.

    Returns
    -------
    float
        At most 1, and in [0, 1] when ``predicted`` is the (weighted)
        least-squares fit of a model that includes constants. NaN when
        ``observed`` never changes: with no variance to explain, R² is
        undefined, and a stationary decode must not pass a replay screen as
        a perfect fit.
    """
    if np.ptp(observed) == 0:
        return float("nan")
    if weights is None:
        weights = np.ones_like(observed)
    mean = np.average(observed, weights=weights)
    ss_res = np.sum(weights * (observed - predicted) ** 2)
    ss_tot = np.sum(weights * (observed - mean) ** 2)
    return float(1.0 - ss_res / ss_tot)


def fit_linear_trajectory(
    env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    method: Literal["sample", "map"] = "sample",
    n_samples: int = 10_000,
    n_resamples: int = 1,
    spike_counts: NDArray[np.int64] | None = None,
    allocation: Literal["equal", "spikes"] = "equal",
    rng: np.random.Generator | int | None = None,
) -> LinearFitResult:
    """Fit a straight-line trajectory to a decoded event.

    Fits one least-squares line of position along the track on time. The
    slope is the replay speed and R² the replay score.

    - ``method="sample"`` (Karlsson & Frank 2009): positions are drawn at
      random from each time bin's posterior, and one line is fit to all the
      sampled points, so the R² reflects both how linear the decoded
      sequence is and how sharp each time bin's posterior is.
    - ``method="map"`` (Wu et al. 2017): one line is fit to the MAP position
      of each time bin.

    Test the R² against shuffles with
    :func:`~neurospatial.replay.replay_significance`.

    Parameters
    ----------
    env : Environment
        The environment the posterior was decoded on: a 1-D environment or a
        linearized track. Positions are read along the track (the linear
        position of a track, or the coordinate of a 1-D environment), so the
        slope is a physical speed and does not depend on how bins are
        numbered.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability mass over spatial bins for each time bin. Each
        row must sum to 1.
    times : NDArray[np.float64], shape (n_time_bins,)
        Center time of each posterior row, in seconds; the independent
        variable of the fit. Need not be evenly spaced.
    method : {"sample", "map"}, default="sample"
        ``"sample"`` fits positions sampled from the posterior (Karlsson &
        Frank 2009); ``"map"`` fits the center of each time bin's MAP bin,
        the bin of highest posterior density (mass per unit length along the
        track; Wu et al. 2017).
    n_samples : int, default=10000
        Total number of positions sampled from the posterior in each
        resample (``method="sample"`` only), at least 2 and at least the
        number of time bins sampled. Karlsson & Frank drew 10,000 per decoded
        time bin: pass ``n_samples=10000 * n_decoded_time_bins`` with
        ``allocation="equal"`` to match them. The Jadhav-lab code draws
        10,000 in total, as the default does, with ``allocation="spikes"``.
    n_resamples : int, default=1
        Number of independent resamples (``method="sample"`` only), at least
        1. The slope, intercept and R² are means over resamples and
        ``slope_std`` their slopes' standard deviation. Karlsson & Frank
        used 1; the Jadhav-lab code uses 500.
    spike_counts : NDArray[np.int64] or None, default=None
        Spike counts behind each posterior row, shape ``(n_time_bins,)`` or
        ``(n_time_bins, n_neurons)`` (summed over neurons): non-negative
        whole numbers. Time bins without spikes are left out of the fit, as
        Karlsson & Frank and Wu et al. do, since they cannot be decoded. None
        fits every row.
    allocation : {"equal", "spikes"}, default="equal"
        How ``method="sample"`` splits ``n_samples`` among the time bins it
        samples. ``"equal"``: the same number for each (Karlsson & Frank
        2009). ``"spikes"``: at random in each resample, multinomially in
        proportion to each time bin's spike count, as the Jadhav-lab code
        does; needs ``spike_counts``. The two weight time bins differently,
        so they estimate different R².
    rng : np.random.Generator, int, or None, default=None
        Random number generator or seed for ``method="sample"``. Pass a seed
        for a reproducible result.

    Returns
    -------
    LinearFitResult
        Slope (environment units per second), intercept (environment units),
        R², and ``slope_std``, the spread of slopes over resamples (None for
        ``method="map"`` or a single resample).

    Raises
    ------
    ValueError
        If ``method`` is not "sample" or "map", if ``env`` is neither 1-D
        nor a linearized track, if ``posterior`` is not 2-D with one column
        per bin, if a posterior row is not a probability mass (NaN or Inf,
        negative, or not summing to 1), if ``times`` does not have one
        finite entry per row or holds a single distinct time, if
        ``n_samples`` is less than 2 or ``n_resamples`` less than 1, if
        ``spike_counts`` does not have one row of non-negative whole numbers
        per time bin, or if the fit would have points at fewer than two
        distinct times.

    See Also
    --------
    LinearFitResult : Container for linear fit results.
    fit_isotonic_trajectory : For monotonic fits.
    detect_trajectory_radon : Replay line holding the most posterior mass.
    neurospatial.replay.replay_significance : Test the score against shuffles.

    Notes
    -----
    **Sampled fit.** Karlsson & Frank (2009) "drew 10,000 random samples from
    the P(X|N) distribution for each decoded bin and assigned the sampled
    locations to that bin", then regressed the sampled locations on the
    time-bin number. Each draw here takes a bin at random with probability
    equal to its posterior mass (inverting the cumulative posterior with
    bins in track order) and uses the bin's center as the sampled position.
    Each sample is placed at its time bin's time, and one ordinary
    least-squares line is fit to all the samples of a resample. The papers
    regress on the time-bin number; for evenly spaced time bins that gives
    the same R² and a slope in positions per time bin rather than per
    second.

    **How many samples each time bin gets.** The time bins sampled are all
    of them, or with ``spike_counts`` those with spikes. With
    ``allocation="equal"`` each gets ``n_samples // n_sampled`` and the
    earliest ``n_samples % n_sampled`` one more. With
    ``allocation="spikes"`` the split is random and proportional to spike
    count, as in the Jadhav-lab code (``jds_replay_decoding_CA1_M.m`` in
    JadhavLab/PrefrontalRipples: 10,000 samples per resample, 500
    resamples, and the mean R², slope and intercept over resamples). As the
    number of samples grows, the R² approaches the square of
    :func:`~neurospatial.replay.weighted_correlation` with each time bin
    weighted equally (``"equal"``) or by its spike count (``"spikes"``).

    The fit runs along the track's linear coordinate. On a track with
    several arms that coordinate joins the arms end to end; Michon et al.
    (2019) fit each arm separately, as do the Jadhav-lab analyses for each
    trajectory, so restrict the posterior to one arm first to match them.

    **Monte Carlo spread.** ``slope_std`` is the standard deviation of the
    slope over resamples. It measures how much the fit changes between
    random draws and shrinks as ``n_samples`` grows; it is not the
    uncertainty of the replay speed given the posterior. The R² of a
    sampled fit is below that of the MAP fit when time bins' posteriors are
    wide, because the spread of positions within a time bin is variance the
    line cannot explain. Mass spread along the track also pulls the sampled
    line toward the middle of the track, so its slope can be shallower than
    the MAP fit's. Samples are drawn and fit in blocks, so memory stays at a
    few arrays of 2 million samples however many resamples are asked for.

    **Track geometry.** The slope is replay speed along the track, in
    environment units per second (cm/s for a track in cm). Bins of unequal
    length are placed at their centers, so a constant-speed sweep gives a
    straight line. On a linearized track the position includes any
    ``edge_spacing`` gap between segments.

    **Stationary decodes.** If the decoded (or every sampled) position never
    changes, the slope is 0 and R² is NaN, so a stationary decode does not
    pass an R² screen.

    Karlsson & Frank used 15 ms time bins and 2 cm position bins; Wu et al.
    20 ms time bins stepped by 10 ms, decoding only time bins with at least
    one spike.

    References
    ----------
    Karlsson, M. P., & Frank, L. M. (2009). Awake replay of remote
    experiences in the hippocampus. *Nature Neuroscience*, 12(7), 913-918.
    https://doi.org/10.1038/nn.2344

    Wu, C.-T., Haggerty, D., Kemere, C., & Ji, D. (2017). Hippocampal awake
    replay in fear memory retrieval. *Nature Neuroscience*, 20(4), 571-580.
    https://doi.org/10.1038/nn.4507

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import fit_linear_trajectory
    >>>
    >>> # A 100 cm track in 2 cm bins and a posterior that sweeps 4 cm per
    >>> # 10 ms time bin, holding 0.8 of its mass in one bin
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=2.0, units="cm"
    ... )
    >>> n_time_bins = 20
    >>> posterior = np.full((n_time_bins, env.n_bins), 0.2 / (env.n_bins - 1))
    >>> posterior[np.arange(n_time_bins), 2 * np.arange(n_time_bins)] = 0.8
    >>> times = np.arange(n_time_bins) * 0.01
    >>>
    >>> # Wu et al. (2017): a line through the MAP positions
    >>> map_fit = fit_linear_trajectory(env, posterior, times, method="map")
    >>> print(f"{map_fit.slope:.0f} cm/s, R² = {map_fit.r_squared:.2f}")
    400 cm/s, R² = 1.00
    >>>
    >>> # Karlsson & Frank (2009): 10,000 samples per time bin. The 0.2 of
    >>> # each row spread along the track lowers the R² and pulls the line
    >>> # toward the middle of the track, flattening its slope.
    >>> sampled = fit_linear_trajectory(
    ...     env, posterior, times, n_samples=10_000 * n_time_bins, rng=0
    ... )
    >>> print(f"{sampled.slope:.0f} cm/s, R² = {sampled.r_squared:.2f}")
    319 cm/s, R² = 0.55
    >>>
    >>> # Jadhav-lab code: 10,000 samples split by spike count, 500 resamples
    >>> spikes = np.random.default_rng(1).poisson(2.0, size=n_time_bins)
    >>> jadhav = fit_linear_trajectory(
    ...     env,
    ...     posterior,
    ...     times,
    ...     n_resamples=500,
    ...     spike_counts=spikes,
    ...     allocation="spikes",
    ...     rng=0,
    ... )
    >>> print(f"R² = {jadhav.r_squared:.2f}, slope_std = {jadhav.slope_std:.1f} cm/s")
    R² = 0.55, slope_std = 3.5 cm/s
    """
    if method not in ("sample", "map"):
        raise ValueError(f"method must be 'sample' or 'map', got {method!r}")
    operation = "fit_linear_trajectory()"
    posterior, starts, stops = _track_positions(env, posterior, operation)
    times = _require_times(times, len(posterior), operation)
    spikes = (
        None
        if spike_counts is None
        else _spikes_per_time_bin(spike_counts, len(posterior), times, operation)
    )

    if method == "map":
        positions = _decoded_positions(posterior, starts, stops, "map")
        if spikes is not None:
            positions, times = positions[spikes > 0], times[spikes > 0]
        slope, intercept = _fit_lines(times, positions)
        return LinearFitResult(
            slope=float(slope),
            intercept=float(intercept),
            r_squared=_compute_r_squared(positions, slope * times + intercept),
            slope_std=None,
        )

    for name, value, least in (
        ("n_samples", n_samples, 2),
        ("n_resamples", n_resamples, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(
                f"{operation}: {name} must be an integer, got {value!r} "
                f"({type(value).__name__})."
            )
        if value < least:
            raise ValueError(
                _format_error(
                    f"{operation} needs {name} of at least {least}, got {value}.",
                    why="n_samples is the total number of positions sampled "
                    "from the posterior in each resample, and a line needs two "
                    "points; n_resamples is the number of independent resamples",
                    fix="keep the defaults (n_samples=10000, n_resamples=1), "
                    "or use method='map' for a fit without sampling",
                )
            )

    if allocation not in ("equal", "spikes"):
        raise ValueError(
            f"{operation}: allocation must be 'equal' or 'spikes', got {allocation!r}."
        )
    if allocation == "spikes" and spikes is None:
        raise ValueError(
            _format_error(
                f"{operation}: allocation='spikes' needs spike_counts.",
                why="it splits the samples among time bins in proportion to "
                "their spike counts",
                fix="pass spike_counts, or use allocation='equal'",
            )
        )
    n_sampled = len(posterior) if spikes is None else int(np.sum(spikes > 0))
    if allocation == "equal" and n_samples < n_sampled:
        raise ValueError(
            _format_error(
                f"{operation}: n_samples={n_samples} is fewer than the "
                f"{n_sampled} time bins it samples.",
                why="an equal split gives every sampled time bin at least one "
                "position; fewer samples would leave the event's last time "
                "bins out",
                fix=f"pass n_samples of at least {n_sampled}",
            )
        )

    from neurospatial.stats import _ensure_rng

    generator = _ensure_rng(rng)
    samples_per_bin = _samples_per_time_bin(
        int(n_samples), int(n_resamples), len(posterior), spikes, allocation, generator
    )
    sampled = samples_per_bin > 0
    latest = np.where(sampled, times, -np.inf).max(axis=1)
    earliest = np.where(sampled, times, np.inf).min(axis=1)
    if np.any(latest == earliest):
        raise ValueError(
            _format_error(
                f"{operation}: in {int(np.sum(latest == earliest))} of "
                f"{n_resamples} resamples, all {n_samples} samples fall in "
                "time bins with one time.",
                why="a line along time needs sampled positions at two or "
                "more times; n_samples is the total over all time bins, and "
                "with allocation='spikes' each time bin's share is random",
                fix="raise n_samples (the default is 10000)",
            )
        )
    slopes, intercepts, r_squared = _sampled_lines(
        posterior, (starts + stops) / 2, times, samples_per_bin, generator
    )
    return LinearFitResult(
        slope=float(np.mean(slopes)),
        intercept=float(np.mean(intercepts)),
        r_squared=float(np.mean(r_squared)),
        slope_std=float(np.std(slopes, ddof=1)) if n_resamples > 1 else None,
    )


def _spikes_per_time_bin(
    spike_counts: NDArray[Any],
    n_time_bins: int,
    times: NDArray[np.float64],
    operation: str,
) -> NDArray[np.int64]:
    """Validate spike counts and sum them over neurons.

    Parameters
    ----------
    spike_counts : NDArray, shape (n_time_bins,) or (n_time_bins, n_neurons)
        Spike counts behind each posterior row.
    n_time_bins : int
        Number of posterior rows.
    times : NDArray[np.float64], shape (n_time_bins,)
        Time of each posterior row, in seconds.
    operation : str
        The calling function, for the refusal message.

    Returns
    -------
    NDArray[np.int64], shape (n_time_bins,)
        Spikes in each time bin, summed over neurons.

    Raises
    ------
    ValueError
        If ``spike_counts`` does not have one row per time bin, holds
        anything but non-negative whole numbers, or leaves fewer than two
        distinct times with spikes.
    """
    counts = np.asarray(spike_counts)
    if counts.ndim not in (1, 2) or counts.shape[0] != n_time_bins:
        raise ValueError(
            _format_error(
                f"{operation}: spike_counts has shape {counts.shape}, but "
                f"posterior has {n_time_bins} rows.",
                why="spike_counts gives the spikes behind each posterior row, "
                "which decides which time bins are fit",
                fix="pass the counts the posterior was decoded from, shape "
                "(n_time_bins, n_neurons), or their sum over neurons, shape "
                "(n_time_bins,)",
            )
        )
    whole = np.issubdtype(counts.dtype, np.integer) or (
        np.issubdtype(counts.dtype, np.floating)
        and bool(np.all(np.isfinite(counts)))
        and bool(np.all(counts == np.round(counts)))
    )
    if not whole or np.any(counts < 0):
        raise ValueError(
            _format_error(
                f"{operation} needs spike_counts of non-negative whole numbers "
                f"of spikes, got dtype {counts.dtype} with values from "
                f"{np.min(counts)} to {np.max(counts)}.",
                why="time bins are fit, and sampled, in proportion to how many "
                "spikes they hold",
                fix="pass spike counts per time bin (not rates or "
                "probabilities), e.g. from np.histogram of each neuron's "
                "spike times over the time-bin edges",
            )
        )
    spikes = counts.astype(np.int64)
    if spikes.ndim == 2:
        spikes = spikes.sum(axis=1)
    if np.unique(times[spikes > 0]).size < 2:
        raise ValueError(
            _format_error(
                f"{operation}: only {int(np.sum(spikes > 0))} of {n_time_bins} "
                "time bins have spikes, at fewer than two distinct times.",
                why="time bins without spikes cannot be decoded and are left "
                "out, and a line along time needs two or more times",
                fix="leave this event out of the analysis: it has too few "
                "active time bins to show a trajectory",
            )
        )
    return spikes


def _samples_per_time_bin(
    n_samples: int,
    n_resamples: int,
    n_time_bins: int,
    spikes: NDArray[np.int64] | None,
    allocation: str,
    generator: np.random.Generator,
) -> NDArray[np.int64]:
    """Number of posterior samples each time bin gets in each resample.

    Parameters
    ----------
    n_samples : int
        Total samples per resample.
    n_resamples : int
        Number of resamples.
    n_time_bins : int
        Number of time bins.
    spikes : NDArray[np.int64] or None, shape (n_time_bins,)
        Spikes per time bin; time bins without spikes get no samples. None
        samples every time bin.
    allocation : {"equal", "spikes"}
        ``"equal"`` splits ``n_samples`` equally among the sampled time bins,
        the earliest taking the remainder; ``"spikes"`` splits it in each
        resample multinomially in proportion to spike count.
    generator : np.random.Generator
        Source of the multinomial split.

    Returns
    -------
    NDArray[np.int64], shape (n_resamples, n_time_bins)
        Samples per time bin; each row sums to ``n_samples``.
    """
    sampled = np.arange(n_time_bins) if spikes is None else np.flatnonzero(spikes)
    if allocation == "equal":
        base, extra = divmod(n_samples, len(sampled))
        counts = np.zeros(n_time_bins, dtype=np.int64)
        counts[sampled] = base
        counts[sampled[:extra]] += 1
        return np.broadcast_to(counts, (n_resamples, n_time_bins))
    assert spikes is not None  # allocation="spikes" needs spike_counts
    # Only time bins with spikes are categories, so a spike-free time bin
    # cannot receive a sample through rounding in the multinomial draw.
    active = np.flatnonzero(spikes)
    split = np.zeros((n_resamples, n_time_bins), dtype=np.int64)
    split[:, active] = generator.multinomial(
        n_samples, spikes[active] / spikes[active].sum(), size=n_resamples
    )
    return split


_SAMPLE_BLOCK = 2_000_000
"""Samples drawn and fit together, bounding memory to a few arrays of this
size (16 MB each) however many resamples are requested; a resample larger
than this is drawn and fit alone."""


def _sampled_lines(
    posterior: NDArray[np.float64],
    centers: NDArray[np.float64],
    times: NDArray[np.float64],
    samples_per_bin: NDArray[np.int64],
    generator: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Draw positions from each time bin's posterior and fit a line per resample.

    Each draw inverts the time bin's cumulative posterior (bins in track
    order): the drawn bin is the first whose cumulative mass exceeds ``u``
    times the row's total, for ``u`` uniform on [0, 1), so a bin with no
    mass is never drawn, and the sampled position is the bin's center.

    Parameters
    ----------
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior mass per bin, bins in track order.
    centers : NDArray[np.float64], shape (n_bins,)
        Bin centers along the track, in track order.
    times : NDArray[np.float64], shape (n_time_bins,)
        Time of each time bin, in seconds.
    samples_per_bin : NDArray[np.int64], shape (n_resamples, n_time_bins)
        Draws from each time bin in each resample; each row sums to
        ``n_samples``.
    generator : np.random.Generator
        Source of the uniforms on [0, 1), drawn block by block; consecutive
        resamples use consecutive draws, used in time-bin order within each
        resample.

    Returns
    -------
    slopes, intercepts, r_squared : NDArray[np.float64], shape (n_resamples,)
        The line through each resample's samples (see :func:`_pooled_lines`).
    """
    n_resamples, n_time_bins = samples_per_bin.shape
    n_bins = posterior.shape[1]
    cumulative = np.cumsum(posterior, axis=1)
    totals = cumulative[:, -1]
    # Rows laid end to end, row t shifted by 2 t so that rows never overlap,
    # make one sorted array that a single searchsorted inverts for every
    # sample, without an (n_samples, n_bins) comparison.
    offsets = 2.0 * np.arange(n_time_bins)
    flat = (cumulative + offsets[:, None]).ravel()
    last = n_bins - 1 - np.argmax(posterior[:, ::-1] > 0, axis=1)
    n_samples = int(samples_per_bin[0].sum())
    step = max(1, _SAMPLE_BLOCK // n_samples)
    lines = []
    for first in range(0, n_resamples, step):
        counts = samples_per_bin[first : first + step]
        time_bin = np.repeat(
            np.tile(np.arange(n_time_bins), len(counts)), counts.ravel()
        ).reshape(len(counts), -1)
        uniforms = generator.random((len(counts), n_samples))
        keys = offsets[time_bin] + uniforms * totals[time_bin]
        drawn = np.searchsorted(flat, keys, side="right") - time_bin * n_bins
        # A key rounded up to its row's total lands past the row; that draw
        # is the row's last bin with mass.
        drawn = np.minimum(drawn, last[time_bin])
        lines.append(_pooled_lines(times[time_bin], centers[drawn]))
    slopes, intercepts, r_squared = (
        np.concatenate(parts) for parts in zip(*lines, strict=True)
    )
    return slopes, intercepts, r_squared


def _pooled_lines(
    sample_times: NDArray[np.float64], sample_positions: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Ordinary least-squares line through each resample's samples.

    Parameters
    ----------
    sample_times, sample_positions : NDArray[np.float64], shape (n_resamples, n_samples)
        Samples of each resample, at two or more distinct times.

    Returns
    -------
    slopes, intercepts, r_squared : NDArray[np.float64], shape (n_resamples,)
        Coefficients and R² of each resample's line; R² is NaN for a
        resample whose positions never change.
    """
    mean_time = sample_times.mean(axis=1, keepdims=True)
    mean_position = sample_positions.mean(axis=1, keepdims=True)
    dt = sample_times - mean_time
    dx = sample_positions - mean_position
    sxx = np.sum(dt * dt, axis=1)
    sxy = np.sum(dt * dx, axis=1)
    syy = np.sum(dx * dx, axis=1)
    slopes = sxy / sxx
    intercepts = mean_position[:, 0] - slopes * mean_time[:, 0]
    # With an intercept, R² of the least-squares line is the squared
    # correlation; it is undefined when the positions never change.
    stationary = np.ptp(sample_positions, axis=1) == 0
    r_squared = np.divide(
        sxy**2,
        sxx * syy,
        out=np.full(len(sxx), np.nan),
        where=~stationary,
    )
    return slopes, intercepts, r_squared


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
    min_speed: float = 0.0,
) -> RadonDetectionResult:
    """Detect the constant-speed replay trajectory in a decoded event.

    Implements the replay line search of Davidson, Kloosterman and Wilson
    (2009): every candidate trajectory, a speed and a start position, is
    scored by the mean over time bins of the posterior probability that the
    position lies within ``band`` of it, and the best trajectory gives the
    replay speed, start position and replay score. The search uses the whole
    posterior rather than one decoded position per time bin. Test the score
    against shuffles with :func:`~neurospatial.replay.replay_significance`.

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
    min_speed : float, default=0.0
        Smallest speed searched, in environment units per second, in either
        direction; slower trajectories are not scored, so a stationary
        event does not score as a trajectory (Ólafsdóttir et al. 2017:
        2 m/s). Must be below ``max_speed``.

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
        have one finite, strictly increasing entry per row, if ``band`` or
        ``max_speed`` is not positive (or is None on an environment without
        metric units), or if ``min_speed`` is negative or not below
        ``max_speed``.

    See Also
    --------
    RadonDetectionResult : Container for the detected trajectory.
    neurospatial.replay.replay_significance : Test the replay score against shuffles.
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

    With ``min_speed``, trajectories slower than it are left out of the
    search and have no score in the sinogram (NaN).

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
    >>> from neurospatial.replay.trajectory import detect_trajectory_radon
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
        min_speed=min_speed,
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
    min_speed: float = 0.0,
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
    from neurospatial.environment import _require_track_extents

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
    if not (np.isfinite(min_speed) and 0 <= min_speed < max_speed):
        raise ValueError(
            _format_error(
                f"{operation} needs 0 <= min_speed < max_speed, got min_speed "
                f"{min_speed:g} and max_speed {max_speed:g}.",
                why="trajectories slower than min_speed are not searched, so "
                "some speed must remain",
                fix="pass a min_speed below max_speed, e.g. min_speed=200.0 for "
                "2 m/s on a track in cm",
            )
        )
    bin_starts, _ = _require_track_extents(
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
        min_speed=float(min_speed),
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
        *,
        min_speed: float = 0.0,
    ) -> None:
        self.order = order
        self.min_speed = min_speed
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
        if not np.any(np.abs(self.speeds) >= min_speed - 1e-9):
            raise ValueError(
                _format_error(
                    f"{operation} searches speeds up to "
                    f"{float(np.abs(self.speeds).max()):g} units/s, all below "
                    f"min_speed {min_speed:g}.",
                    why="speeds are spaced so that neighboring trajectories end "
                    "band / 5 apart, so a short event has few, widely spaced "
                    "speeds",
                    fix="lower min_speed, raise max_speed or widen band",
                )
            )

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
            if block.lines.size == 0:  # every speed of the block below min_speed
                continue
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
        fast_enough = np.repeat(
            np.abs(self.speeds[speeds]) >= self.min_speed - 1e-9, n_starts
        )
        touches = on_track.any(axis=1) & fast_enough
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
    "detect_trajectory_radon",
    "fit_isotonic_trajectory",
    "fit_linear_trajectory",
]
