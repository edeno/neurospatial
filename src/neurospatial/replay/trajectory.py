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
regression_line_score
    Posterior mass near the line through the MAP positions (Harvey et al.
    2023; Soula et al. 2023).

Result Dataclasses
------------------
IsotonicFitResult
    Container for isotonic regression results.
LinearFitResult
    Container for linear regression results.
RadonDetectionResult
    Container for Radon transform detection results.
RegressionLineScoreResult
    Container for the regression-line score and its line.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
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


@dataclass(frozen=True, repr=False, eq=False)
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
class RegressionLineScoreResult(ResultMixin):
    """Posterior mass near the line through a decoded event's MAP positions.

    :func:`regression_line_score` fits one least-squares line of the MAP
    position on time and scores the posterior mass within ``band`` of it
    (Harvey et al. 2023; Soula et al. 2023).

    Attributes
    ----------
    score : float
        Mean, over the time bins with posterior mass, of the mass within
        ``band`` of the line, in [0, 1]. Time bins whose posterior is all
        zero (no spikes) are left out of the line and of the mean. NaN with
        fewer than two time bins with mass, which cannot fix a line.
    slope : float
        Slope of the line: replay speed along the track in environment
        units per second (cm/s for a track in cm); positive when position
        increases along the track with time.
    intercept : float
        Position along the track, in environment units, where the line
        crosses ``times == 0``.
    r_squared : float
        R² of the line through the MAP positions, in [0, 1]; NaN when the
        MAP position never changes.
    band : float
        Half-width of the band around the line, in environment units.
    wrap_band : bool
        Whether the band wrapped around the track's ends.

    See Also
    --------
    regression_line_score : Function that produces this result.
    """

    score: float
    slope: float
    intercept: float
    r_squared: float
    band: float
    wrap_band: bool

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "score": self.score,
            "slope": self.slope,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "band": self.band,
            "wrap_band": self.wrap_band,
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
        track when the trajectory enters the track after the event starts
        (only with ``line_ends="any"`` on an open track); on a closed loop
        (``wrap=True``) it lies on the loop.
    score : float
        Replay score of Davidson et al. (2009), the best over the searched
        trajectories: the mean over time bins of the posterior mass within
        ``band`` of the trajectory, in [0, 1]. The reported trajectory scores
        within 0.001 of it (see :func:`detect_trajectory_radon`). It is 1 when
        every time bin puts all its mass within ``band`` of the line, and
        about ``2 * band`` divided by the track length for a uniform posterior.
        A time bin where the line lies beyond the ends of the track contributes
        the median of that mass over positions on the track. With
        ``line_ends="on_track"``, band beyond an end of the track holds the
        time bin's median posterior density instead, and each time bin's
        band mass is capped at 1.
    band : float
        Half-width of the band around a line, in environment units.
    sinogram : NDArray[np.float64], shape (n_speeds, n_start_positions)
        Replay score of every candidate trajectory: entry ``(i, j)`` is the
        trajectory of speed ``sinogram_speeds[i]`` through
        ``sinogram_start_positions[j]`` at ``times[0]``; NaN for one that is
        not searched: never on the track, slower than the search's
        ``min_speed``, or, with ``line_ends="on_track"``, off the track at
        the first or last time bin. With random lines (``n_lines``), shape
        ``(n_lines,)``: entry ``i`` is the line of speed
        ``sinogram_speeds[i]`` through ``sinogram_start_positions[i]``, NaN
        also for one faster than ``max_speed``.
    sinogram_speeds : NDArray[np.float64], shape (n_speeds,)
        Speeds of the sinogram's rows, in environment units per second,
        evenly spaced; rows slower than ``min_speed`` were not searched.
        With ``n_lines``, each line's speed, shape ``(n_lines,)``.
    sinogram_start_positions : NDArray[np.float64], shape (n_start_positions,)
        Searched start positions along the track, in environment units,
        evenly spaced at most ``band / 5`` apart. With ``line_ends="any"``
        on an open track they reach beyond both ends of the track as far as
        the fastest trajectory travels in the event; otherwise they lie on
        the track. With ``n_lines``, each line's position at ``times[0]``,
        shape ``(n_lines,)``.

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
    env: Environment,
    posterior: NDArray[np.float64],
    operation: str,
    *,
    allow_zero_rows: bool = False,
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
    allow_zero_rows : bool, default=False
        Accept rows that are all zero (time bins without spikes).

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
    posterior = _require_probability_rows(
        posterior, operation, allow_zero_rows=allow_zero_rows
    )
    # A stable sort keeps bins that are already in track order in place, so
    # sampling below draws the same bins as it would in bin numbering.
    order = np.argsort(starts, kind="stable")
    return posterior[:, order], starts[order], stops[order]


def _require_probability_rows(
    posterior: NDArray[Any], operation: str, *, allow_zero_rows: bool = False
) -> NDArray[np.float64]:
    """Refuse rows that are not a probability mass over the bins.

    Parameters
    ----------
    posterior : NDArray, shape (n_time_bins, n_bins)
        Posterior mass per bin.
    operation : str
        The calling function, for the refusal message.
    allow_zero_rows : bool, default=False
        Accept rows that are all zero, the time bins without spikes that
        ``empty_time_bins="zero"`` zeroes.

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
    if allow_zero_rows:
        unnormalized &= sums != 0
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


_REGRESSION_BAND_CM = 10.5
"""Default half-width of the band of :func:`regression_line_score`, 10.5 cm.

Harvey et al. (2023) and Soula et al. (2023) scored "the sum probability mass
under the fit line within a bandwidth (21 cm)": read as the band's full width,
the 2w + 1 = 7 position bins of 3 cm that their code (nelpy's
``trajectory_score_array`` with ``w=3``) sums around the line.
"""


def regression_line_score(
    env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    band: float | None = None,
    wrap_band: bool = False,
) -> RegressionLineScoreResult:
    """Posterior mass near the line through a decoded event's MAP positions.

    Fits one least-squares line of the MAP position on time, through the
    time bins with posterior mass, and scores the event by the mean over
    those time bins of the posterior mass within ``band`` of the line
    (Harvey et al. 2023, "alternative approach"; Soula et al. 2023). The
    line is not searched for: unlike :func:`detect_trajectory_radon`, which
    keeps the line holding the most mass, it is the regression line of the
    decoded positions, and unlike :func:`fit_linear_trajectory`, which
    scores that line's R², the score is the mass near it. Test the score
    against shuffles with :func:`~neurospatial.replay.replay_significance`.

    Parameters
    ----------
    env : Environment
        The environment the posterior was decoded on: a 1-D environment or a
        linearized track. Positions are read along the track, so the result
        does not depend on how bins are numbered.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability mass over spatial bins for each time bin. Each
        row sums to 1, or is all zero for a time bin without spikes (as
        ``empty_time_bins="zero"`` of
        :func:`~neurospatial.replay.detect_replay` passes them).
    times : NDArray[np.float64], shape (n_time_bins,)
        Center time of each posterior row, in seconds; the independent
        variable of the line.
    band : float or None, default=None
        Half-width of the band around the line, in environment units. None
        means 10.5 cm in the environment's units, which requires metric
        ``env.units``: Harvey et al. and Soula et al. state a 21 cm
        bandwidth, which their code sums as 7 position bins of 3 cm centered
        on the line.
    wrap_band : bool, default=False
        Where the band reaches past an end of the track, take the posterior
        at the other end, as the authors' code (nelpy) does by circularly
        shifting each time bin's posterior. False counts only the band's
        part on the track, so a line near an end holds less mass.

    Returns
    -------
    RegressionLineScoreResult
        The score, and the slope, intercept and R² of the line.

    Raises
    ------
    TypeError
        If ``wrap_band`` is not a bool.
    ValueError
        If ``env`` is neither 1-D nor a linearized track, if ``posterior``
        is not 2-D with one column per bin, if a row is neither a
        probability mass nor all zero, if ``times`` does not have one finite
        entry per row or holds a single distinct time, or if ``band`` is not
        positive (or is None on an environment without metric units).

    See Also
    --------
    RegressionLineScoreResult : Container for the score and its line.
    detect_trajectory_radon : The line holding the most posterior mass.
    fit_linear_trajectory : R² of a line through the MAP positions.
    neurospatial.replay.replay_significance : Test the score against shuffles.

    Notes
    -----
    **Line.** The MAP position of each time bin with mass is the center of
    its bin of highest posterior density (mass per unit length). The line
    ``x(t) = slope * t + intercept`` is the least-squares fit of these
    positions on the times. The authors' code (nelpy's ``linregress_array``)
    regresses the MAP bin on the time-bin number, which gives the same line
    for evenly spaced time bins and equal bins.

    **Score.** For each time bin with mass, the posterior mass within
    ``band`` of ``x(t)``, with each bin's mass spread evenly along it so
    that the band's mass is exact for any bin sizes and any line position;
    the score is the mean over those time bins. With ``wrap_band=True``,
    positions are read modulo the track's length (from the first bin's
    start to the last bin's end), so band past one end takes the mass at
    the other, and a band longer than the track holds the whole time bin.
    nelpy's ``trajectory_score_array`` (``normalize=True``) rounds the line
    to the nearest bin and sums the ``2w + 1`` bins around it, wrapping
    around the track's ends; for a line through bin centers and ``band =
    (w + 1/2) * bin_size`` the two agree.

    **Time bins without spikes.** nelpy leaves them without a posterior
    (NaN): they are left out of the line and of the mean, and here a row of
    zeros is treated the same way.

    References
    ----------
    Harvey, R. E., Robinson, H. L., Liu, C., Oliva, A., & Fernandez-Ruiz, A.
    (2023). Hippocampo-cortical circuits for selective memory encoding,
    routing, and replay. *Neuron*, 111(13), 2076-2090.

    Soula, M., Maslarova, A., Harvey, R. E., Valero, M., Brandner, S.,
    Hamer, H., Fernández-Ruiz, A., & Buzsáki, G. (2023). Interictal
    epileptiform discharges affect memory in an Alzheimer's disease mouse
    model. *PNAS*, 120(34), e2302676120.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import regression_line_score
    >>>
    >>> # A 150 cm track in 3 cm bins; the posterior sweeps one bin per
    >>> # 20 ms time bin, with 0.7 of its mass in the MAP bin and 0.1 in each
    >>> # of the bins 1, 3 and 6 bins (3, 9 and 18 cm) ahead of it. The
    >>> # band of 10.5 cm around the line holds all but the last.
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [150.0, 0.0]], bin_size=3.0, units="cm"
    ... )
    >>> posterior = np.zeros((10, env.n_bins))
    >>> rows = np.arange(10)
    >>> posterior[rows, 10 + rows] = 0.7
    >>> for ahead in (1, 3, 6):
    ...     posterior[rows, 10 + rows + ahead] = 0.1
    >>> times = np.arange(10) * 0.02
    >>> result = regression_line_score(env, posterior, times)
    >>> print(
    ...     f"{result.slope:.0f} cm/s, score {result.score:.1f} within "
    ...     f"{result.band:g} cm"
    ... )
    150 cm/s, score 0.9 within 10.5 cm
    """
    from neurospatial._units import default_length

    operation = "regression_line_score()"
    if not isinstance(wrap_band, (bool, np.bool_)):
        raise TypeError(
            f"{operation} needs wrap_band=True or wrap_band=False, got "
            f"{wrap_band!r}; True takes band past a track end from the other end."
        )
    posterior, starts, stops = _track_positions(
        env, posterior, operation, allow_zero_rows=True
    )
    times = _require_times(times, len(posterior), operation)
    if band is None:
        band = default_length(env, _REGRESSION_BAND_CM, name="band", call=operation)
    if not (np.isfinite(band) and band > 0):
        raise ValueError(
            _format_error(
                f"{operation} needs band > 0 in the environment's units, got {band:g}.",
                why="band is the half-width of the band around the line",
                fix="pass a positive value, e.g. band=10.5 for the 21 cm band "
                "of Harvey et al. (2023) on a track in cm",
            )
        )
    score, slope, intercept, r_squared = _regression_line_scores(
        posterior[np.newaxis], times, starts, stops, float(band), bool(wrap_band)
    )
    return RegressionLineScoreResult(
        score=float(score[0]),
        slope=float(slope[0]),
        intercept=float(intercept[0]),
        r_squared=float(r_squared[0]),
        band=float(band),
        wrap_band=bool(wrap_band),
    )


def _regression_line_scores(
    posteriors: NDArray[np.float64],
    times: NDArray[np.float64],
    starts: NDArray[np.float64],
    stops: NDArray[np.float64],
    band: float,
    wrap_band: bool,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Regression-line scores of a batch of events with one geometry.

    Parameters
    ----------
    posteriors : NDArray[np.float64], shape (n_events, n_time_bins, n_bins)
        Validated posteriors, bins in track order; rows sum to 1 or 0.
    times : NDArray[np.float64], shape (n_time_bins,)
    starts, stops : NDArray[np.float64], shape (n_bins,)
        Where each bin starts and ends along the track, in track order.
    band : float
        Half-width of the band, in environment units.
    wrap_band : bool
        Read positions modulo the track's length.

    Returns
    -------
    score, slope, intercept, r_squared : NDArray[np.float64], shape (n_events,)
        As in :class:`RegressionLineScoreResult`; NaN for an event with
        fewer than two time bins with mass.
    """
    n_events, n_time, n_bins = posteriors.shape
    has_mass = posteriors.sum(axis=2) > 0  # (n_events, n_time)
    centers = (starts + stops) / 2
    map_bins = _map_bins(posteriors.reshape(-1, n_bins), starts, stops)
    positions = centers[map_bins].reshape(n_events, n_time)

    # Least squares over the time bins with mass, about the middle of the
    # time range so that an offset such as session time adds no round-off.
    weights = has_mass.astype(np.float64)
    n_fit = weights.sum(axis=1)
    fits = n_fit >= 2
    count = np.where(fits, n_fit, 1.0)
    middle = (times.min() + times.max()) / 2.0
    t = times - middle
    mean_t = (weights @ t) / count
    mean_x = (weights * positions).sum(axis=1) / count
    t_dev = t[np.newaxis] - mean_t[:, np.newaxis]
    x_dev = positions - mean_x[:, np.newaxis]
    sxx = (weights * t_dev**2).sum(axis=1)
    fits &= sxx > 0
    slope = np.where(
        fits,
        (weights * t_dev * x_dev).sum(axis=1) / np.where(sxx > 0, sxx, 1.0),
        np.nan,
    )
    line = mean_x[:, np.newaxis] + slope[:, np.newaxis] * t_dev  # (n_events, n_time)
    intercept = np.where(fits, mean_x - slope * (mean_t + middle), np.nan)
    ss_res = (weights * (positions - line) ** 2).sum(axis=1)
    ss_tot = (weights * x_dev**2).sum(axis=1)
    varies = np.where(has_mass, positions, -np.inf).max(axis=1) > np.where(
        has_mass, positions, np.inf
    ).min(axis=1)
    r_squared = np.where(
        fits & varies, 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0), np.nan
    )

    # Mass within the band: F(line + band) - F(line - band), with F the
    # cumulative posterior along the track, linear within each bin.
    cumulative = np.cumsum(posteriors, axis=2) - posteriors  # mass before each bin
    lengths = stops - starts
    origin = float(starts[0])
    loop = float(stops[-1] - starts[0])

    def cumulative_mass(position: NDArray[np.float64]) -> NDArray[np.float64]:
        turns = np.zeros_like(position)
        if wrap_band:
            turns = np.floor((position - origin) / loop)
            position = position - turns * loop
        index = np.searchsorted(starts, position, side="right") - 1
        clipped = np.clip(index, 0, n_bins - 1)[..., np.newaxis]
        before = np.take_along_axis(cumulative, clipped, axis=2)[..., 0]
        mass = np.take_along_axis(posteriors, clipped, axis=2)[..., 0]
        within = np.clip(
            (position - starts[clipped[..., 0]]) / lengths[clipped[..., 0]], 0.0, 1.0
        )
        inside: NDArray[np.float64] = np.where(index >= 0, before + mass * within, 0.0)
        return inside + turns  # each full turn of the loop holds a row's mass of 1

    safe_line = np.where(np.isfinite(line), line, origin)
    band_mass = np.minimum(
        cumulative_mass(safe_line + band) - cumulative_mass(safe_line - band), 1.0
    )
    score = np.where(
        fits,
        (weights * band_mass).sum(axis=1) / count,
        np.nan,
    )
    return _round_scores(score), slope, intercept, r_squared


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

_BOUND_SLACK = 1e-9
"""Slack on the bound that skips trajectories that cannot reach an event's
best score with ``zero_rows="line_median"``, far above rounding error."""

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
    wrap: bool = False,
    line_ends: Literal["any", "on_track"] = "any",
    off_track: Literal["band_median", "bin_median"] = "band_median",
    zero_rows: Literal["refuse", "line_median"] = "refuse",
    n_lines: int | None = None,
    line_seed: int = 0,
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
        al. found replay speeds up to about 20 m/s (median 8.1 m/s). With
        ``line_ends="on_track"``, None means no bound beyond the one the
        track sets: the track's length over the event's duration; with
        ``n_lines``, no bound (lines of every slope are drawn).
    min_speed : float, default=0.0
        Smallest speed searched, in environment units per second, in either
        direction; slower trajectories are not scored, so a stationary
        event does not score as a trajectory (Ólafsdóttir et al. 2017:
        2 m/s). Must be below ``max_speed``.
    wrap : bool, default=False
        Treat the track as a closed loop, such as a circular or square track
        (Muessig et al. 2019). Positions are read along the track modulo the
        loop's length, so a trajectory that runs past the end of the track
        continues from its start, and its band wraps across the seam where
        the end meets the start: a sweep crossing the seam scores as it
        would anywhere else on the loop. Every trajectory lies on the loop,
        so start positions are searched once around it. Needs a closed-loop
        track, built with :meth:`~neurospatial.Environment.from_graph` with
        an ``edge_order`` whose last segment ends at the node where the first
        starts and no ``edge_spacing``, and a ``band`` shorter than half the
        loop. False reads the
        track as open, so on a loop a sweep across the seam scores low.
    line_ends : {"any", "on_track"}, default="any"
        Which trajectories are searched. ``"any"`` (Davidson et al. 2009)
        searches every trajectory that lies on the track in at least one time
        bin, at speeds up to ``max_speed``. ``"on_track"`` (Liu et al. 2023,
        whose code, ``FindReplayScore`` of ayalab1/neurocode, searches every
        line from a start bin to an end bin) searches only trajectories that
        lie on the track at the first and the last time bin, so the speed is
        bounded by the track's length over the event's duration; band that
        extends beyond an end of the track holds the time bin's median
        posterior density over the bins (see Notes). Not available with
        ``wrap=True``: a loop has no ends.
    off_track : {"band_median", "bin_median"}, default="band_median"
        What a time bin where the trajectory lies beyond the ends of the
        track contributes. ``"band_median"`` (Davidson et al. 2009): the
        median, over positions on the track, of the posterior mass within
        ``band`` of the position. ``"bin_median"`` (Maboudi et al. 2018):
        the median, over the track's bins, of the time bin's posterior mass
        in one bin, much smaller than a band's mass. Only with
        ``line_ends="any"`` on an open track, where trajectories can leave
        the track.
    zero_rows : {"refuse", "line_median"}, default="refuse"
        Posterior rows that are all zero: time bins without spikes, as
        ``empty_time_bins="zero"`` of
        :func:`~neurospatial.replay.replay_significance` and
        :func:`~neurospatial.replay.detect_replay` passes them. ``"refuse"``
        raises, as for any row that does not sum to 1. ``"line_median"``
        (Maboudi et al. 2018) scores each such row, for every trajectory,
        by the median of that trajectory's band mass over its time bins
        with spikes on the track (0 when it has none); the score is still
        the mean over all ``n_time_bins``. On a loop (``wrap=True``) such a
        row holds no mass in any lap its band covers. A posterior with no
        mass in any time bin is refused. Dropping the time bins without
        spikes, with their times, instead leaves them out of the mean.
    n_lines : int or None, default=None
        None searches the dense grid of speeds and start positions (see
        Notes). An int searches that many lines drawn at random over every
        slope, as Maboudi et al. (2018) did after Kloosterman (2012), with
        35,000: much faster at a narrow ``band``, but the best line can be
        missed. The same lines are scored in an event and in its shuffles.
    line_seed : int, default=0
        Seed of the random lines with ``n_lines``, so a search is
        reproducible; events with the same numbers of time bins and bins
        draw the same lines.

    Returns
    -------
    RadonDetectionResult
        The speed, start position and replay score of the best trajectory,
        and the score of every candidate (the sinogram) with its axes.

    Raises
    ------
    TypeError
        If ``wrap`` is not a bool.
    ValueError
        If ``env`` is neither 1-D nor a linearized track, if ``posterior``
        does not have one column per bin or a row is not a probability mass
        (NaN or Inf, negative, or not summing to 1), if ``times`` does not
        have one finite, strictly increasing entry per row, if ``band`` or
        ``max_speed`` is not positive (or is None on an environment without
        metric units, unless ``line_ends="on_track"``), if ``min_speed`` is
        negative or not below ``max_speed``, if ``line_ends`` is not one of
        its choices, or if ``wrap=True`` on a track that is not a closed
        loop (or has ``edge_spacing``), with ``line_ends="on_track"``, or
        with a ``band`` of half the loop or more; if ``off_track`` or
        ``zero_rows`` is not one of its choices, if no time bin has mass
        with ``zero_rows="line_median"``, or
        ``off_track="bin_median"`` is given where no trajectory leaves the
        track (``wrap=True`` or ``line_ends="on_track"``); if ``n_lines`` is
        not a positive integer or ``line_seed`` not a non-negative integer,
        or if no random line is both within the speed range and on the
        track; if no speed of the dense grid reaches ``min_speed``.

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
    neighboring trajectories end at most one start spacing apart. With the
    default ``line_ends="any"``, every trajectory that lies on the track in
    at least one time bin is searched,
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

    **Random lines** (``n_lines``). Maboudi et al. (2018) "sampled randomly
    following the approach in (Kloosterman, 2012)" 35,000 lines; their code
    (nelpy's ``score_Davidson_final_bst_fast``) draws them on the image of
    the event, time bins by position bins, with the normal of each line at
    an angle uniform on (-pi/2, pi/2) and its distance from a center point
    uniform within half the image's diagonal, so lines of every slope are
    drawn. The center point is nelpy's, ``(n_time_bins + 1) / 2`` and
    ``(n_bins + 1) / 2`` with bins numbered from 0, one bin past the
    image's center on both axes; lines are drawn about it here too. Here
    the image's units are the mean time step and the mean spacing of the
    bin centers, and positions along a line are exact (the code rounds them
    to bins). The sinogram then holds one score per
    line, in the order drawn, with that line's speed and start position.

    **Track geometry.** Positions are read along the track (the linear
    position of a linearized track). Gaps between bins, such as
    ``edge_spacing`` between the segments of a maze, count as track for the
    band but are not positions for the off-track median. Davidson et al.
    used a single linear track.

    **Closed loops** (``wrap=True``). Muessig et al. (2019) searched lines
    on a square track and, "as the square track has a circular topology",
    wrapped the band around the posterior at its first and last bins. Here
    the start positions cover the loop once, at most ``band / 5`` apart,
    and the reported ``start_position`` lies on the loop. Among equally good
    trajectories the reported one is the closest to the line through the
    circular mean positions, unwrapped around the loop.

    **Trajectories that start and end on the track**
    (``line_ends="on_track"``). Liu et al. (2023) fitted the line of
    Davidson et al. with a band of 22.5 cm; their code searches lines from
    every start bin to every end bin, so a line is on the track at the
    event's first and last time bins (and therefore throughout), and its
    speed runs from 0 to the track's length over the event's duration.
    Here these are the trajectories whose start and end positions are both
    on the track. Where the band extends beyond an end of the track, the
    code fills the missing bins with the time bin's median posterior
    probability over the bins, and caps the band's sum at 1. Here the band
    beyond an end holds the median over the bins of the posterior density
    (mass per unit length), the same for bins of equal size, and each
    time bin's band mass is capped at 1. The code rounds line positions to
    bins; positions here are exact.

    **Off-track and empty time bins after Maboudi et al. (2018).** They
    scored "the mean probability mass under the line, within a bandwidth
    (of 3 cm)": "For time bins during which the sampled line fell outside
    of the extent of the track, the median probability mass of the
    corresponding time bin was used, and for time bins during which no
    spikes were observed, we used the median probability mass across all
    on-track time bins". ``off_track="bin_median"`` and
    ``zero_rows="line_median"`` follow that text. Their code (nelpy's
    ``score_Davidson_final_bst_fast``) scores a time bin off the track by
    its median single-bin mass, as written, but computes the median for
    time bins without spikes and never adds it, so those score 0. Dropping
    the time bins without spikes, with their times, scores every
    trajectory as the code does, multiplied by ``n_time_bins`` over the
    number of time bins with spikes, so it ranks trajectories, and their
    column-cycle shuffles, as the code does.

    References
    ----------
    Davidson, T. J., Kloosterman, F., & Wilson, M. A. (2009). Hippocampal
    replay of extended experience. *Neuron*, 63(4), 497-507.

    Muessig, L., Lasek, M., Varsavsky, I., Cacucci, F., & Wills, T. J.
    (2019). Coordinated emergence of hippocampal replay and theta sequences
    during post-natal development. *Current Biology*, 29(5), 834-840.

    Liu, C., Todorova, R., Tang, W., Oliva, A., & Fernandez-Ruiz, A. (2023).
    Associative and predictive hippocampal codes support memory-guided
    behaviors. *Science*, 382(6668), eadi8237.

    Maboudi, K., Ackermann, E., de Jong, L. W., Pfeiffer, B. E., Foster, D.,
    Diba, K., & Kemere, C. (2018). Uncovering temporal structure in
    hippocampal output patterns. *eLife*, 7, e34467.

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

    On a closed loop, here a square track of 1 m sides, ``wrap=True`` scores
    a sweep that crosses the seam where the end of the track meets its start
    as it would score anywhere else:

    >>> import networkx as nx
    >>> graph = nx.Graph()
    >>> for node, corner in enumerate([(0, 0), (100, 0), (100, 100), (0, 100)]):
    ...     graph.add_node(node, pos=corner)
    >>> edge_order = [(0, 1), (1, 2), (2, 3), (3, 0)]
    >>> graph.add_edges_from(edge_order)
    >>> loop = Environment.from_graph(graph, edge_order, bin_size=2.5, units="cm")
    >>> # One 2.5 cm bin per 10 ms from 350 cm, across the seam at 400 cm
    >>> track_bins = (140 + np.arange(30)) % loop.n_bins
    >>> posterior = np.eye(loop.n_bins)[track_bins]
    >>> times = np.arange(30) * 0.01
    >>> wrapped = detect_trajectory_radon(loop, posterior, times, wrap=True)
    >>> opened = detect_trajectory_radon(loop, posterior, times)
    >>> print(
    ...     f"{wrapped.speed:.0f} cm/s from {wrapped.start_position:.0f} cm: "
    ...     f"score {wrapped.score:.2f} wrapped, {opened.score:.2f} open"
    ... )
    247 cm/s from 351 cm: score 1.00 wrapped, 0.67 open
    """
    lines, posterior = _prepare_replay_lines(
        env,
        posterior,
        times,
        band=band,
        max_speed=max_speed,
        min_speed=min_speed,
        wrap=wrap,
        line_ends=line_ends,
        off_track=off_track,
        zero_rows=zero_rows,
        n_lines=n_lines,
        line_seed=line_seed,
        operation="detect_trajectory_radon()",
    )
    return _best_line(lines, posterior)


_LINE_ENDS = ("any", "on_track")
_OFF_TRACK = ("band_median", "bin_median")
_ZERO_ROWS = ("refuse", "line_median")
# Inside _reusing_replay_lines(): the key and lines of the latest search.
_REUSED_LINES: ContextVar[list[Any] | None] = ContextVar("_REUSED_LINES", default=None)


@contextlib.contextmanager
def _reusing_replay_lines() -> Iterator[None]:
    """Reuse the latest line search for searches of the same geometry.

    Building a search's scoring operators costs far more than scoring one
    posterior with them. A caller that scores many posteriors of one event
    one call at a time, such as :func:`dataset_values` measuring an event's
    shuffles with :func:`detect_replay`, runs those calls inside this
    block: a :func:`detect_trajectory_radon` search with the same bins,
    times and options as the one before it then reuses its lines and
    operators. Only the latest search is kept, and only inside the block.
    """
    token = _REUSED_LINES.set([])
    try:
        yield
    finally:
        _REUSED_LINES.reset(token)


def _require_closed_loop(env: Environment, operation: str) -> None:
    """Refuse ``wrap=True`` unless the track's linear position is a loop.

    The linear position runs along ``edge_order``, each segment from its
    first node to its second, so it is a loop when the last segment ends at
    the node where the first starts, and it is distance around the loop
    only without ``edge_spacing``: a gap between segments would count as
    loop length at every corner but not at the seam.
    """
    linearization = env._binning.linearization
    if linearization is None:
        problem = "it is not a track built from a graph, so it has no loop"
    else:
        first, last = linearization.edge_order[0], linearization.edge_order[-1]
        gaps = np.atleast_1d(np.asarray(linearization.edge_spacing, dtype=np.float64))
        if last[1] != first[0]:
            problem = (
                f"its last segment {tuple(last)!r} ends at node {last[1]!r}, not "
                f"at node {first[0]!r} where its first segment {tuple(first)!r} "
                "starts"
            )
        elif np.any(gaps > 0):
            problem = (
                f"its edge_spacing puts gaps of up to {gaps.max():g} between "
                "segments, which count as loop length at each corner but not "
                "where the last segment meets the first"
            )
        else:
            return
    raise ValueError(
        _format_error(
            f"{operation} got wrap=True, but this environment is not a "
            f"closed-loop track: {problem}.",
            why="wrap=True reads positions modulo the loop's length, so a "
            "trajectory past the end of the track continues from its start; "
            "on an open track the two ends are different places",
            fix="pass wrap=False on an open track; build a circular or square "
            "track with Environment.from_graph(graph, edge_order=[(0, 1), "
            "(1, 2), (2, 3), (3, 0)], bin_size=..., edge_spacing=0.0), so that "
            "the last segment returns to the node where the first starts with "
            "no gaps between segments",
        )
    )


def _prepare_replay_lines(
    env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    band: float | None,
    max_speed: float | None,
    min_speed: float = 0.0,
    wrap: bool = False,
    line_ends: str = "any",
    off_track: str = "band_median",
    zero_rows: str = "refuse",
    n_lines: int | None = None,
    line_seed: int = 0,
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
    from neurospatial._validation import validate_choice
    from neurospatial.environment import _require_track_extents

    if not isinstance(wrap, (bool, np.bool_)):
        raise TypeError(
            f"{operation} needs wrap=True or wrap=False, got {wrap!r}; True "
            "treats the track as a closed loop."
        )
    validate_choice(line_ends, _LINE_ENDS, name="line_ends", owner=operation)
    validate_choice(off_track, _OFF_TRACK, name="off_track", owner=operation)
    validate_choice(zero_rows, _ZERO_ROWS, name="zero_rows", owner=operation)
    for name, count, smallest in (("n_lines", n_lines, 1), ("line_seed", line_seed, 0)):
        if count is not None and (
            isinstance(count, (bool, np.bool_))
            or not isinstance(count, (int, np.integer))
            or count < smallest
        ):
            raise ValueError(
                f"{operation} needs {name} to be an integer >= {smallest}, got "
                f"{count!r}."
            )
    if off_track == "bin_median" and (wrap or line_ends == "on_track"):
        setting = "wrap=True" if wrap else "line_ends='on_track'"
        raise ValueError(
            _format_error(
                f"{operation} got off_track='bin_median' with {setting}.",
                why=f"with {setting} every searched trajectory stays on the "
                "track, so off_track would change nothing",
                fix="leave off_track at its default, or search trajectories "
                "that may leave the track with line_ends='any' and wrap=False",
            )
        )
    if wrap and line_ends == "on_track":
        raise ValueError(
            _format_error(
                f"{operation} got wrap=True with line_ends='on_track'.",
                why="a closed loop has no ends, so every trajectory starts and "
                "ends on the track",
                fix="pass line_ends='any' with wrap=True, and bound the speed "
                "with max_speed",
            )
        )
    # Validates the environment and the posterior rows.
    _, starts, stops = _track_positions(
        env, posterior, operation, allow_zero_rows=zero_rows == "line_median"
    )
    if wrap:
        _require_closed_loop(env, operation)
    posterior = np.asarray(posterior, dtype=np.float64)
    if not posterior.any():
        raise ValueError(
            _format_error(
                f"{operation} got a posterior with no mass in any time bin.",
                why="zero_rows='line_median' scores a time bin without spikes "
                "by the line's other time bins, and here there are none, so "
                "no line has a score",
                fix="leave out events without spikes, or score them as not replay",
            )
        )
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
    # Trajectories that start and end on the track are bounded by it, and
    # random lines are drawn over every slope, so they need no default bound.
    if max_speed is None and line_ends != "on_track" and n_lines is None:
        max_speed = default_length(
            env, _DEFAULT_MAX_SPEED_CM, name="max_speed", call=operation
        )
    for name, value, example in (
        ("band", band, "band=15.0 for 15 cm on a track in cm"),
        ("max_speed", max_speed, "max_speed=3000.0 for 30 m/s on a track in cm"),
    ):
        if value is not None and not (np.isfinite(value) and value > 0):
            raise ValueError(
                _format_error(
                    f"{operation} needs {name} > 0 in the environment's units, "
                    f"got {value:g}.",
                    why="band is the half-width of the band around a trajectory "
                    "and max_speed the largest speed searched",
                    fix=f"pass a positive value, e.g. {example}",
                )
            )
    speed_bound = np.inf if max_speed is None else float(max_speed)
    if not (np.isfinite(min_speed) and 0 <= min_speed < speed_bound):
        raise ValueError(
            _format_error(
                f"{operation} needs 0 <= min_speed < max_speed, got min_speed "
                f"{min_speed:g} and max_speed {speed_bound:g}.",
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
    loop_length = float(stops[-1] - starts[0])  # the track's length on a loop
    if wrap and not 2 * band < loop_length:
        raise ValueError(
            _format_error(
                f"{operation} got wrap=True with band {band:g}, but the loop is "
                f"{loop_length:g} long, at most twice the band.",
                why="a band of half the loop or more covers the whole loop, so "
                "every trajectory would score 1",
                fix=f"pass a band below {loop_length / 2:g}, in the "
                "environment's units",
            )
        )
    elapsed = times - times[0]
    reused = _REUSED_LINES.get()
    key = None
    if reused is not None:
        arrays = (starts, stops, order, elapsed)
        key = (
            *(np.asarray(values).tobytes() for values in arrays),
            *(band, speed_bound, min_speed, wrap, line_ends, off_track),
            *(zero_rows, n_lines, line_seed),
        )
        if reused and reused[0] == key:
            return reused[1], posterior
    lines = _ReplayLines(
        starts,
        stops,
        order,
        elapsed,
        float(band),
        speed_bound,
        operation,
        min_speed=float(min_speed),
        wrap=bool(wrap),
        line_ends=line_ends,
        off_track=off_track,
        zero_rows=zero_rows,
        n_lines=None if n_lines is None else int(n_lines),
        line_seed=int(line_seed),
    )
    if reused is not None:
        reused[:] = [key, lines]
    return lines, posterior


def _best_line(
    lines: _ReplayLines, posterior: NDArray[np.float64]
) -> RadonDetectionResult:
    """The best-scoring trajectory of one event, ties broken by a line fit."""
    sinogram = lines.sinograms(lines.event_values(posterior[None]), best_only=False)[0]
    # Trajectories scoring within _SCORE_TIE of the best are equally good.
    # Among them, report the speed closest to the least-squares line through
    # the posterior mean positions, then the start closest to that line's
    # start. The score reported is the best one, as for the shuffles.
    best = float(np.nanmax(sinogram))
    tied_speeds, tied_starts = lines.line_parameters(
        np.flatnonzero(sinogram >= best - _SCORE_TIE)
    )
    # Rows without mass (time bins without spikes) have no mean position.
    with_mass = posterior.sum(axis=1) > 0
    mass = posterior[with_mass][:, lines.order]
    elapsed = lines.elapsed[with_mass]
    if lines.wrap:
        # On a loop, the circular mean position, unwrapped around the loop.
        angle = 2 * np.pi * (lines.bin_centers - lines.track_start) / lines.length
        mean_angle = np.unwrap(np.angle(mass @ np.exp(1j * angle)))
        mean_position = lines.track_start + mean_angle * lines.length / (2 * np.pi)
    else:
        mean_position = mass @ lines.bin_centers
    if elapsed.size >= 2:
        fit_speed, fit_start = _fit_lines(elapsed, mean_position)
    else:  # one time bin with mass, or none: no speed to fit
        fit_speed = 0.0
        fit_start = (
            float(mean_position[0])
            if elapsed.size
            else (lines.track_start + lines.track_end) / 2
        )
    start_gap = tied_starts - fit_start
    if lines.wrap:
        start_gap = (start_gap + lines.length / 2) % lines.length - lines.length / 2
    closest = np.lexsort((np.abs(start_gap), np.abs(tied_speeds - fit_speed)))[0]
    return RadonDetectionResult(
        speed=float(tied_speeds[closest]),
        start_position=float(tied_starts[closest]),
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

    On a closed loop (``wrap``) positions are read modulo the loop's length.
    With ``line_ends="on_track"`` only trajectories on the track at the first
    and last time bins are scored, band beyond an end of the track holds the
    time bin's median posterior density, and each time bin's band mass is
    capped at 1. With ``off_track="bin_median"`` a time bin off the track
    holds its median mass per bin instead. With ``zero_rows="line_median"``
    each all-zero row of an event adds, for each trajectory, the median of
    its band mass over the time bins with mass on the track, divided by the
    number of time bins (see :meth:`_add_line_medians`).

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
        Largest speed searched, in environment units per second; ``np.inf``
        with ``line_ends="on_track"`` for no bound beyond the track's.
    operation : str
        The calling function, for the refusal message.
    min_speed : float, default=0.0
        Smallest speed searched, in environment units per second.
    wrap : bool, default=False
        Whether the track is a closed loop.
    line_ends : {"any", "on_track"}, default="any"
        Which trajectories are searched (see :func:`detect_trajectory_radon`).
    off_track : {"band_median", "bin_median"}, default="band_median"
        What a time bin off the track holds (see
        :func:`detect_trajectory_radon`).
    zero_rows : {"refuse", "line_median"}, default="refuse"
        How all-zero rows are scored (see :func:`detect_trajectory_radon`).
    n_lines : int or None, default=None
        None for the dense grid; else the number of random lines.
    line_seed : int, default=0
        Seed of the random lines.
    """

    # Memory budgets, in array elements of 8 bytes: the nonzeros of one
    # sparse block of trajectories (building it takes several arrays of this
    # size), the nonzeros kept between batches of shuffles (about 100 MB),
    # and the per-event values of the shuffles scored together and the
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
        wrap: bool = False,
        line_ends: str = "any",
        off_track: str = "band_median",
        zero_rows: str = "refuse",
        n_lines: int | None = None,
        line_seed: int = 0,
    ) -> None:
        self.order = order
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.elapsed = elapsed
        self.band = band
        self.wrap = wrap
        self.line_ends = line_ends
        self.off_track = off_track
        self.zero_rows = zero_rows
        self.random = n_lines is not None
        self.track_start, self.track_end = float(starts[0]), float(stops[-1])
        self.length = self.track_end - self.track_start
        n_on_track = int(np.ceil(self.length * _LINES_PER_BAND / band - 1e-9))
        spacing = self.length / n_on_track
        # The cumulative mass is known at each bin's start and stop and grows
        # linearly within a bin; it stays flat across a gap between bins.
        self.knots = np.column_stack([starts, stops]).ravel()
        self.bin_centers = (starts + stops) / 2
        self.bin_lengths = stops - starts
        # The off-track median is over the positions on the track, at most
        # band / 5 apart, that lie on a bin.
        on_track_positions = (
            self.track_start + (np.arange(n_on_track, dtype=np.float64) + 0.5) * spacing
        )
        within = np.searchsorted(stops, on_track_positions, side="left")
        on_bin = (within < len(starts)) & (
            on_track_positions >= starts[np.minimum(within, len(starts) - 1)]
        )
        self.on_bin_positions = on_track_positions[on_bin]
        self._blocks: list[_LineBlock] | None = None
        # Speeds and start positions: the grid's two axes, or each line's.
        self.speeds, self.start_positions = (
            self._grid(n_on_track, spacing, operation)
            if n_lines is None
            else self._random_lines(n_lines, line_seed)
        )
        if self.random:
            self._require_a_searched_line(operation, line_seed)
        elif not np.any(np.abs(self.speeds) >= min_speed - 1e-9):
            raise ValueError(
                _format_error(
                    f"{operation} searches speeds up to "
                    f"{float(np.abs(self.speeds).max()):g} units/s, all below "
                    f"min_speed {min_speed:g}.",
                    why="speeds are spaced so that neighboring trajectories end "
                    "band / 5 apart, so a short event has few, widely spaced "
                    "speeds; with line_ends='on_track' they reach the track's "
                    "length over the event's duration",
                    fix="lower min_speed, raise max_speed or widen band",
                )
            )

    def _require_a_searched_line(self, operation: str, line_seed: int) -> None:
        """Refuse random lines of which none is searched."""
        _, _, in_speed_range, placed = self._search_mask(
            self.speeds, self.start_positions
        )
        if np.any(in_speed_range & placed):
            return
        speed_range = (
            f"of at least min_speed {self.min_speed:g}"
            if np.isinf(self.max_speed)
            else f"between min_speed {self.min_speed:g} and max_speed "
            f"{self.max_speed:g}"
        )
        where = (
            "at the first and last time bins"
            if self.line_ends == "on_track"
            else "at some time bin"
        )
        raise ValueError(
            _format_error(
                f"{operation} has no line to score: none of the "
                f"{self.speeds.size} random lines (line_seed {line_seed}) is "
                f"both within the speed range and on the track. "
                f"{int(in_speed_range.sum())} have speeds {speed_range} "
                f"units/s, and {int(placed.sum())} lie on the track {where}.",
                why="random lines are drawn over every slope and across the "
                "event's image of time bins by position bins, so few lines, "
                "or a narrow speed range, can leave none to score",
                fix="draw more lines (Maboudi et al. 2018 drew n_lines=35000), "
                "widen the range between min_speed and max_speed, or search "
                "the dense grid with n_lines=None",
            )
        )

    def _grid(
        self, n_on_track: int, spacing: float, operation: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The dense grid's speeds and start positions (its two axes)."""
        elapsed, max_speed = self.elapsed, self.max_speed
        wrap, line_ends = self.wrap, self.line_ends
        # Neighboring speeds end one start spacing apart at the last time bin.
        duration = float(elapsed[-1])
        speed_step = spacing / duration
        # A trajectory from the first start position to the last ends one
        # start spacing short of a track length away.
        bounds = [n_on_track - 1] if line_ends == "on_track" else []
        if np.isfinite(max_speed):
            bounds.append(int(np.floor(max_speed / speed_step + 1e-9)))
        n_speeds = min(bounds)
        speeds = speed_step * np.arange(-n_speeds, n_speeds + 1, dtype=np.float64)
        # With any line ends, start positions extend beyond the track as far
        # as the fastest trajectory travels, so a trajectory may enter the
        # track as well as leave it; only those on the track at some time bin
        # are scored. On a loop, or with line ends on the track, every
        # trajectory starts on the track.
        n_beyond = (
            0
            if wrap or line_ends == "on_track"
            else int(np.ceil(n_speeds * speed_step * duration / spacing - 1e-9))
        )
        start_positions = (
            self.track_start
            + (np.arange(-n_beyond, n_on_track + n_beyond, dtype=np.float64) + 0.5)
            * spacing
        )
        search = len(speeds) * len(start_positions) * len(elapsed)
        if search > self._MAX_SEARCH:
            raise ValueError(
                _format_error(
                    f"{operation} would score {len(speeds)} speeds x "
                    f"{len(start_positions)} start positions over "
                    f"{len(elapsed)} time bins, more than {self._MAX_SEARCH:.0e}: "
                    f"the event spans {duration:g} s and speeds reach "
                    f"{float(speeds[-1]):g} units/s, in steps of "
                    f"{speed_step:.3g}.",
                    why="neighboring trajectories are spaced band / 5 apart, so "
                    "the search grows with the event's duration times "
                    "max_speed, over band",
                    fix="check that times are in seconds (a replay event spans "
                    "tens to hundreds of milliseconds); otherwise lower "
                    "max_speed or widen band",
                )
            )
        return speeds, start_positions

    def _random_lines(
        self, n_lines: int, line_seed: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Speed and start position of ``n_lines`` lines drawn as nelpy draws them.

        As in nelpy's ``score_Davidson_final_bst_fast`` (Maboudi et al.
        2018, after Kloosterman 2012), lines are drawn on the image of the
        event, time bins by position bins: the normal of each line makes an
        angle uniform on (-pi/2, pi/2) with the time axis, and its distance
        from nelpy's center is uniform within half the image's diagonal, so
        every slope is drawn. nelpy's center is ``((n_time_bins + 1) / 2,
        (n_bins + 1) / 2)`` with bins numbered from 0, one bin past the
        image's center on both axes, so some lines through the image's
        first corner (first time bin, first position bin) are never drawn,
        while every line through the opposite corner can be. Image units
        are the mean time step and the mean spacing of the bin centers.

        Parameters
        ----------
        n_lines : int
            Number of lines drawn.
        line_seed : int
            Seed of the draws: angles first, then distances.

        Returns
        -------
        speeds, start_positions : NDArray[np.float64], shape (n_lines,)
            Each line's speed, in environment units per second, and its
            position at the first time bin.
        """
        n_time, n_bins = self.elapsed.size, self.bin_centers.size
        time_step = float(self.elapsed[-1]) / (n_time - 1)
        bin_step = (
            float(self.bin_centers[-1] - self.bin_centers[0]) / (n_bins - 1)
            if n_bins > 1
            else self.length
        )
        diagonal = np.hypot(n_time - 1, n_bins - 1)
        generator = np.random.default_rng(line_seed)
        angle = np.pi * (generator.random(n_lines) - 0.5)
        distance = diagonal * (generator.random(n_lines) - 0.5)
        # nelpy keeps lines off the vertical, whose speed is infinite.
        angle[np.abs(angle) < 1e-4] = 1e-4
        # (c - c_mid) cos(angle) + (r - r_mid) sin(angle) = distance, for time
        # bin c and position bin r numbered from 0, about nelpy's center
        # c_mid = (n_time + 1) / 2, r_mid = (n_bins + 1) / 2, one bin past
        # the image's center on both axes.
        slope = -np.cos(angle) / np.sin(angle)
        start = (distance + (n_time + 1) / 2 * np.cos(angle)) / np.sin(angle)
        start += (n_bins + 1) / 2
        speeds: NDArray[np.float64] = slope * bin_step / time_step
        start_positions: NDArray[np.float64] = self.bin_centers[0] + start * bin_step
        return speeds, start_positions

    def line_parameters(
        self, lines: NDArray[np.intp]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Speed and start position of trajectories by flat index.

        Parameters
        ----------
        lines : NDArray[np.intp], shape (n,)
            Flat index into the sinogram: ``speed * n_start_positions +
            start`` on the grid, the line itself with ``n_lines``.

        Returns
        -------
        speeds, start_positions : NDArray[np.float64], shape (n,)
        """
        if self.random:
            return self.speeds[lines], self.start_positions[lines]
        n_starts = self.start_positions.size
        return self.speeds[lines // n_starts], self.start_positions[lines % n_starts]

    @property
    def n_trajectories(self) -> int:
        """Number of entries of the sinogram."""
        if self.random:
            return self.speeds.size
        return self.speeds.size * self.start_positions.size

    def event_values(self, posteriors: NDArray[np.float64]) -> NDArray[np.float64]:
        """What the scoring operators read from each event, one column each.

        Parameters
        ----------
        posteriors : NDArray[np.float64], shape (n_events, n_time_bins, n_bins)
            Posteriors in bin numbering.

        Returns
        -------
        NDArray[np.float64], shape (n_time_bins * (n_knots + 2) + 1, n_events)
            Per event: the posterior mass before each of ``self.knots`` at
            each time bin; the off-track mass summed over the time bins before
            each time bin (``n_time_bins + 1`` prefix sums, zero unless lines
            may leave the track); and each time bin's median posterior density
            (zero unless ``line_ends="on_track"``). The mass before the last
            knot of a time bin is its row's total.
        """
        mass = posteriors[..., self.order]
        after = np.cumsum(mass, axis=-1)
        cumulative = np.stack([after - mass, after], axis=-1).reshape(
            *mass.shape[:-1], -1
        )
        n_events, n_time = mass.shape[:2]
        n_cumulative = n_time * len(self.knots)
        values = np.zeros((n_cumulative + 2 * n_time + 1, n_events))
        values[:n_cumulative] = cumulative.reshape(n_events, -1).T
        if not self.wrap and self.line_ends == "any":
            off_track = (
                np.median(mass, axis=-1)
                if self.off_track == "bin_median"
                else self._off_track_mass(cumulative)
            )
            np.cumsum(
                off_track.T,
                axis=0,
                out=values[n_cumulative + 1 : n_cumulative + n_time + 1],
            )
        if self.line_ends == "on_track":
            values[n_cumulative + n_time + 1 :] = np.median(
                mass / self.bin_lengths, axis=-1
            ).T
        return values

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

    def _off_track_mass(self, cumulative: NDArray[np.float64]) -> NDArray[np.float64]:
        """Median over the track of the mass within ``band`` of a position.

        Davidson et al. give a time bin where a trajectory leaves the track the
        median probability over all locations on the track.

        Parameters
        ----------
        cumulative : NDArray[np.float64], shape (n_events, n_time_bins, 2 * n_bins)
            Mass before each of ``self.knots``.

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
        self, values: NDArray[np.float64], *, best_only: bool
    ) -> NDArray[np.float64]:
        """Score every candidate trajectory in every event.

        Parameters
        ----------
        values : NDArray[np.float64], shape (n_values, n_events)
            From :meth:`event_values`.
        best_only : bool
            Return each event's best score instead of every trajectory's.

        Returns
        -------
        NDArray[np.float64]
            Shape ``(n_events, n_speeds, n_start_positions)``, NaN for a
            trajectory that is not searched, or ``(n_events,)`` when
            ``best_only``.
        """
        n_events = values.shape[1]
        best = np.full(n_events, -np.inf)
        scores = None if best_only else np.full((n_events, self.n_trajectories), np.nan)
        zero = self._zero_rows_of(values)
        for block in self._line_blocks(n_events):
            if block.lines.size == 0:  # every speed of the block below min_speed
                continue
            block_scores = block.operator @ values  # (n_lines, n_events)
            if block.padded is not None:
                # Band mass above 1 at a time bin, from the padding, is cut.
                excess = np.maximum(block.padded @ values - 1.0, 0.0)
                block_scores -= block.padded_lines @ excess
            if zero is not None:
                self._add_line_medians(
                    block, values, block_scores, zero, best if best_only else None
                )
            if scores is None:
                np.maximum(best, block_scores.max(axis=0), out=best)
            else:
                scores[:, block.lines] = block_scores.T
        if scores is None:
            return _round_scores(best)
        if self.random:
            return _round_scores(scores)
        return _round_scores(
            scores.reshape(n_events, self.speeds.size, self.start_positions.size)
        )

    def _zero_rows_of(self, values: NDArray[np.float64]) -> NDArray[np.bool_] | None:
        """Which rows of each event are all zero, with ``zero_rows="line_median"``.

        Parameters
        ----------
        values : NDArray[np.float64], shape (n_values, n_events)
            From :meth:`event_values`.

        Returns
        -------
        NDArray[np.bool_] or None, shape (n_time_bins, n_events)
            True for a row without mass; None when no event has one or
            ``zero_rows`` is not ``"line_median"`` (validation refuses such
            rows otherwise).
        """
        if self.zero_rows != "line_median":
            return None
        n_knots = len(self.knots)
        totals = values[n_knots - 1 : self.elapsed.size * n_knots : n_knots]
        zero: NDArray[np.bool_] = totals == 0
        return zero if zero.any() else None

    def _add_line_medians(
        self,
        block: _LineBlock,
        values: NDArray[np.float64],
        block_scores: NDArray[np.float64],
        zero: NDArray[np.bool_],
        best: NDArray[np.float64] | None,
    ) -> None:
        """Add the score of each event's all-zero rows to its trajectories.

        Each all-zero row scores, for each trajectory, the median of the
        trajectory's band mass over its time bins on the track with mass,
        divided by the number of time bins (Maboudi et al. 2018); 0 when it
        has none.

        The median is not linear in the posterior, so it is computed per
        trajectory and event. With ``best`` (only the best score of each
        event is wanted), it is computed only for trajectories that could
        reach the best score. A trajectory's score is at least ``s``, its
        other rows' part (``block_scores`` before this call). The median of
        its ``k`` band masses is at most their sum over ``ceil(k / 2)``,
        since at least that many are at or above it; their sum is at most
        ``n_time * s``, and ``k`` is at least its time bins on the track
        less the event's ``n_zero`` all-zero rows. So its score is at most
        ``s * (1 + n_zero / max(ceil((n_on_track - n_zero) / 2), 1))``, and
        a trajectory whose bound is below a score already reached cannot be
        the best. The best score is therefore exact.

        Parameters
        ----------
        block : _LineBlock
            The trajectories, with their ``pairs``.
        values : NDArray[np.float64], shape (n_values, n_events)
            From :meth:`event_values`.
        block_scores : NDArray[np.float64], shape (n_lines, n_events)
            Scores of the block's trajectories without the all-zero rows;
            updated in place.
        zero : NDArray[np.bool_], shape (n_time_bins, n_events)
            From :meth:`_zero_rows_of`.
        best : NDArray[np.float64] or None, shape (n_events,)
            Each event's best score so far, or None to score every
            trajectory.
        """
        pairs = block.pairs
        assert pairs is not None  # built with zero_rows="line_median"
        n_lines, n_events = block_scores.shape
        n_zero = zero.sum(axis=0)
        if best is None:
            with_zero = np.flatnonzero(n_zero)
            line_index, event_index = (
                index.ravel() for index in np.indices((n_lines, with_zero.size))
            )
            event_index = with_zero[event_index]
        else:
            n_on_track = pairs.on_track.sum(axis=1)
            line_parts, event_parts = [], []
            for count in np.unique(n_zero[n_zero > 0]):
                events = np.flatnonzero(n_zero == count)
                other = (
                    block_scores if events.size == n_events else block_scores[:, events]
                )
                factor = 1 + count / np.maximum(np.ceil((n_on_track - count) / 2), 1)
                # A score already reached: the best so far, or the exact
                # score of the trajectory with the largest other-row part.
                top = np.argmax(other, axis=0)
                reached = other[top, np.arange(events.size)] + (
                    count / self.elapsed.size
                ) * self._line_medians(pairs, values, top, events, zero)
                floor = np.maximum(best[events], reached)
                lines, chosen = np.nonzero(
                    other * factor[:, None] >= floor - _BOUND_SLACK
                )
                line_parts.append(lines)
                event_parts.append(events[chosen])
            line_index = np.concatenate(line_parts)
            event_index = np.concatenate(event_parts)
        step = max(1, self._BLOCK_SCORES // pairs.columns[0].size)
        for first in range(0, line_index.size, step):
            lines = line_index[first : first + step]
            events = event_index[first : first + step]
            block_scores[lines, events] += (
                n_zero[events] / self.elapsed.size
            ) * self._line_medians(pairs, values, lines, events, zero)

    @staticmethod
    def _line_medians(
        pairs: _PairTerms,
        values: NDArray[np.float64],
        lines: NDArray[np.intp],
        events: NDArray[np.intp],
        zero: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Median band mass over each trajectory's time bins on the track with mass.

        Parameters
        ----------
        pairs : _PairTerms
            The block's band terms per trajectory and time bin.
        values : NDArray[np.float64], shape (n_values, n_events)
            From :meth:`_ReplayLines.event_values`.
        lines, events : NDArray[np.intp], shape (n_scored,)
            Trajectory in the block, and event, of each median.
        zero : NDArray[np.bool_], shape (n_time_bins, n_events)
            Rows without mass.

        Returns
        -------
        NDArray[np.float64], shape (n_scored,)
            The median, or 0 when the trajectory has no such time bin.
        """
        mass = np.einsum(
            "ntk,ntk->nt",
            pairs.weights[lines],
            values[pairs.columns[lines], events[:, None, None]],
        )
        if pairs.laps is not None:
            assert pairs.total_columns is not None  # set with laps
            # Each whole lap holds the time bin's total mass (0 without mass).
            mass += pairs.laps[lines] * values[pairs.total_columns][:, events].T
        if pairs.padded is not None:
            # As in the score, band mass above 1 from the padding is cut.
            mass = np.where(pairs.padded[lines], np.minimum(mass, 1.0), mass)
        counted = pairs.on_track[lines] & ~zero[:, events].T
        ordered = np.sort(np.where(counted, mass, np.inf), axis=1)
        count = counted.sum(axis=1)
        rows = np.arange(len(lines))
        middle = (
            ordered[rows, np.maximum(count - 1, 0) // 2] + ordered[rows, count // 2]
        ) / 2
        medians: NDArray[np.float64] = np.where(count > 0, middle, 0.0)
        return medians

    def _line_blocks(self, n_events: int) -> Iterator[_LineBlock]:
        """Blocks of trajectories with their sparse operators.

        Blocks keep their nonzeros, and their scores for ``n_events`` events,
        within the memory budgets. They are kept between calls when all of
        them fit ``_CACHED_NONZEROS``.
        """
        if self._blocks is not None:
            yield from self._blocks
            return
        # On the grid a block is whole speeds, with every start position.
        n_starts = 1 if self.random else self.start_positions.size
        n_time = self.elapsed.size
        step = max(
            1,
            min(
                self._BLOCK_NONZEROS // (n_starts * n_time * 4),
                self._BLOCK_SCORES // (n_starts * n_events),
            ),
        )
        built: list[_LineBlock] | None = []
        nonzeros = 0
        for first in range(0, self.n_trajectories, step * n_starts):
            block = self._block(
                np.arange(first, min(first + step * n_starts, self.n_trajectories))
            )
            if built is not None:
                nonzeros += block.operator.nnz
                if block.padded is not None:
                    nonzeros += block.padded.nnz + block.padded_lines.nnz
                if block.pairs is not None:
                    nonzeros += 2 * block.pairs.columns.size
                if nonzeros <= self._CACHED_NONZEROS:
                    built.append(block)
                else:
                    built = None
            yield block
        if built is not None:
            self._blocks = built

    def _band_terms(
        self, positions: NDArray[np.float64], time: NDArray[np.intp]
    ) -> tuple[
        NDArray[np.intp],
        NDArray[np.intp],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
    ]:
        """Band mass around each position, as weights on :meth:`event_values`.

        Each band end is located on the track: its cumulative mass is
        interpolated between bin edges and held flat beyond the track. On a
        loop, the band end is first brought onto the loop and each whole
        lap it went round adds the time bin's total mass (1, or 0 for a
        row without mass; the caller weights the total's value row by
        ``laps``). With line ends on the track, band
        beyond an end holds that length times the time bin's median density.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (n_pairs,)
            Trajectory position at a time bin, in environment units.
        time : NDArray[np.intp], shape (n_pairs,)
            That time bin.

        Returns
        -------
        pair, column : NDArray[np.intp], shape (n_terms,)
            Pair and value row of each weight.
        weight : NDArray[np.float64], shape (n_terms,)
            Weights on the values.
        laps : NDArray[np.float64], shape (n_pairs,)
            Whole laps the band covers, each holding the time bin's total
            mass (zero off a loop).
        padded : NDArray[np.bool_], shape (n_pairs,)
            Whether the pair's band extends beyond an end of the track and
            holds padding.
        """
        n_knots, n_time = len(self.knots), self.elapsed.size
        pairs = np.arange(positions.size)
        pair_parts, column_parts, weight_parts = [], [], []
        laps = np.zeros(positions.size)
        for sign in (1.0, -1.0):
            end = positions + sign * self.band
            if self.wrap:
                turns = np.floor((end - self.track_start) / self.length)
                end = end - turns * self.length
                laps += sign * turns
            left, fraction = self._locate(end)
            base = time * n_knots + left
            for offset, weight in ((0, 1.0 - fraction), (1, fraction)):
                pair_parts.append(pairs)
                column_parts.append(base + offset)
                weight_parts.append(sign * weight)
        padded = np.zeros(positions.size, dtype=bool)
        if self.line_ends == "on_track":
            beyond = np.maximum(positions + self.band - self.track_end, 0.0)
            beyond += np.maximum(self.track_start - (positions - self.band), 0.0)
            padded = beyond > 0
            pair_parts.append(pairs[padded])
            column_parts.append(n_time * (n_knots + 1) + 1 + time[padded])
            weight_parts.append(beyond[padded])
        return (
            np.concatenate(pair_parts),
            np.concatenate(column_parts),
            np.concatenate(weight_parts),
            laps,
            padded,
        )

    def _search_mask(
        self, speeds: NDArray[np.float64], start_positions: NDArray[np.float64]
    ) -> tuple[
        NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]
    ]:
        """Which trajectories are searched, by speed and by where they lie.

        Parameters
        ----------
        speeds, start_positions : NDArray[np.float64], shape (n,)
            Each trajectory's speed and position at the first time bin.

        Returns
        -------
        position : NDArray[np.float64], shape (n, n_time_bins)
            Each trajectory's position at each time bin.
        on_track : NDArray[np.bool_], shape (n, n_time_bins)
            Whether it lies on the track at each time bin.
        in_speed_range : NDArray[np.bool_], shape (n,)
            Whether its speed is at least ``min_speed`` (and, for random
            lines, at most ``max_speed``).
        placed : NDArray[np.bool_], shape (n,)
            Whether it lies on the track where the search needs it: at some
            time bin, or at the first and last with ``line_ends="on_track"``.
        """
        position = start_positions[:, None] + speeds[:, None] * self.elapsed[None, :]
        in_speed_range = np.abs(speeds) >= self.min_speed - 1e-9
        if self.random:
            in_speed_range &= np.abs(speeds) <= self.max_speed
        if self.wrap:
            on_track = np.ones(position.shape, dtype=bool)
            placed = np.ones(speeds.shape, dtype=bool)
        else:
            on_track = (position >= self.track_start) & (position <= self.track_end)
            # A trajectory is monotonic: on the track at both ends means on
            # it throughout.
            placed = np.asarray(
                on_track[:, 0] & on_track[:, -1]
                if self.line_ends == "on_track"
                else on_track.any(axis=1)
            )
        return position, on_track, in_speed_range, placed

    def _block(self, candidates: NDArray[np.intp]) -> _LineBlock:
        """The searched trajectories among ``candidates`` (flat indices).

        The operator's row ``r`` sums, over time bins on the track, the mass
        within ``band`` of trajectory ``lines[r]`` (see :meth:`_band_terms`),
        and with any line ends the off-track mass of the other time bins,
        divided by the number of time bins.
        """
        from scipy import sparse

        n_time = self.elapsed.size
        n_values = n_time * (len(self.knots) + 2) + 1
        speeds, start_positions = self.line_parameters(candidates)
        position, on_track, in_speed_range, placed = self._search_mask(
            speeds, start_positions
        )
        searched = in_speed_range & placed
        position, on_track = position[searched], on_track[searched]
        lines = candidates[searched]
        row, time = np.nonzero(on_track)
        pair, column, weight, laps, padded = self._band_terms(position[row, time], time)
        row_parts, column_parts, weight_parts = [row[pair]], [column], [weight / n_time]
        if self.wrap:
            # Each whole lap a band covers holds the time bin's total mass:
            # 1 for a decoded row, 0 for a row without mass.
            row_parts.append(row)
            column_parts.append(time * len(self.knots) + len(self.knots) - 1)
            weight_parts.append(laps / n_time)
        if not self.wrap and self.line_ends == "any":
            # Off-track time bins are those before the first and after the
            # last on the track (a trajectory is monotonic, so its time bins
            # on the track are a run): prefix sums of the off-track mass at
            # first, n_time and last + 1.
            first = np.argmax(on_track, axis=1)
            last = n_time - 1 - np.argmax(on_track[:, ::-1], axis=1)
            prefix = n_time * len(self.knots)
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
            shape=(lines.size, n_values),
        )
        pairs = (
            self._pair_terms(on_track, row, time, column, weight, laps, padded)
            if self.zero_rows == "line_median"
            else None
        )
        if not np.any(padded):
            return _LineBlock(lines=lines, operator=operator, pairs=pairs)
        # The band mass of each padded pair, and the sum over a line's padded
        # pairs, divided by the number of time bins.
        n_padded = int(np.sum(padded))
        index = np.cumsum(padded) - 1
        keep = padded[pair]
        return _LineBlock(
            lines=lines,
            operator=operator,
            pairs=pairs,
            padded=sparse.csr_matrix(
                (weight[keep], (index[pair[keep]], column[keep])),
                shape=(n_padded, n_values),
            ),
            padded_lines=sparse.csr_matrix(
                (np.full(n_padded, 1.0 / n_time), (row[padded], np.arange(n_padded))),
                shape=(lines.size, n_padded),
            ),
        )

    def _pair_terms(
        self,
        on_track: NDArray[np.bool_],
        row: NDArray[np.intp],
        time: NDArray[np.intp],
        column: NDArray[np.intp],
        weight: NDArray[np.float64],
        laps: NDArray[np.float64],
        padded: NDArray[np.bool_],
    ) -> _PairTerms:
        """Each trajectory's band terms at each time bin, for line medians.

        Parameters
        ----------
        on_track : NDArray[np.bool_], shape (n_lines, n_time_bins)
            Whether each searched trajectory is on the track at each time bin.
        row, time : NDArray[np.intp], shape (n_pairs,)
            Trajectory and time bin of each pair on the track.
        column : NDArray[np.intp], shape (n_terms,)
            From :meth:`_band_terms`: four terms per pair, in four runs of
            ``n_pairs``, then one per padded pair.
        weight : NDArray[np.float64], shape (n_terms,)
            From :meth:`_band_terms`, in the same order.
        laps : NDArray[np.float64], shape (n_pairs,)
            From :meth:`_band_terms`.
        padded : NDArray[np.bool_], shape (n_pairs,)
            From :meth:`_band_terms`.
        """
        n_lines, n_time = on_track.shape
        n_pairs = row.size
        n_terms = 5 if np.any(padded) else 4
        columns = np.zeros((n_lines, n_time, n_terms), dtype=np.intp)
        weights = np.zeros((n_lines, n_time, n_terms))
        columns[row, time, :4] = column[: 4 * n_pairs].reshape(4, n_pairs).T
        weights[row, time, :4] = weight[: 4 * n_pairs].reshape(4, n_pairs).T
        pair_padded = None
        if n_terms == 5:
            columns[row[padded], time[padded], 4] = column[4 * n_pairs :]
            weights[row[padded], time[padded], 4] = weight[4 * n_pairs :]
            pair_padded = np.zeros((n_lines, n_time), dtype=bool)
            pair_padded[row, time] = padded
        pair_laps = total_columns = None
        if self.wrap:
            pair_laps = np.zeros((n_lines, n_time))
            pair_laps[row, time] = laps
            n_knots = len(self.knots)
            total_columns = np.arange(n_time) * n_knots + n_knots - 1
        return _PairTerms(
            columns=columns,
            weights=weights,
            on_track=on_track,
            laps=pair_laps,
            total_columns=total_columns,
            padded=pair_padded,
        )

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
        values_per_event = len(self.elapsed) * (len(self.knots) + 2) + 1
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
        return self.sinograms(self.event_values(posteriors), best_only=True)


@dataclass(frozen=True)
class _LineBlock:
    """Searched trajectories and their scoring operators.

    Attributes
    ----------
    lines : NDArray[np.intp], shape (n_lines,)
        Flat index ``speed * n_start_positions + start`` of each trajectory.
    operator : scipy.sparse.csr_matrix
        Shape ``(n_lines, n_values)``: maps the values of
        :meth:`_ReplayLines.event_values` to each trajectory's replay score.
    padded : scipy.sparse.csr_matrix or None
        Shape ``(n_padded, n_values)``: the band mass of each trajectory and
        time bin whose band holds padding beyond an end of the track.
    padded_lines : scipy.sparse.csr_matrix or None
        Shape ``(n_lines, n_padded)``: sums a trajectory's padded time bins,
        divided by the number of time bins.
    pairs : _PairTerms or None
        Each trajectory's band terms per time bin, with
        ``zero_rows="line_median"``.
    """

    lines: NDArray[np.intp]
    operator: Any
    padded: Any = None
    padded_lines: Any = None
    pairs: _PairTerms | None = None


@dataclass(frozen=True)
class _PairTerms:
    """Band mass of each trajectory at each time bin, as weights on event values.

    Attributes
    ----------
    columns, weights : NDArray, shape (n_lines, n_time_bins, n_terms)
        The band mass at a time bin is ``sum(weights * values[columns])``
        over its terms (zero weights off the track).
    on_track : NDArray[np.bool_], shape (n_lines, n_time_bins)
        Whether the trajectory is on the track at each time bin.
    laps : NDArray[np.float64] or None, shape (n_lines, n_time_bins)
        On a loop, the whole laps each band covers.
    total_columns : NDArray[np.intp] or None, shape (n_time_bins,)
        On a loop, the value row holding each time bin's total mass, which
        each whole lap holds.
    padded : NDArray[np.bool_] or None, shape (n_lines, n_time_bins)
        Whether the band holds padding beyond an end of the track, and so is
        capped at 1.
    """

    columns: NDArray[np.intp]
    weights: NDArray[np.float64]
    on_track: NDArray[np.bool_]
    laps: NDArray[np.float64] | None = None
    total_columns: NDArray[np.intp] | None = None
    padded: NDArray[np.bool_] | None = None


def _round_scores(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """Replay scores rounded to ``_SCORE_DECIMALS`` and held to [0, 1]."""
    rounded: NDArray[np.float64] = np.clip(np.round(scores, _SCORE_DECIMALS), 0.0, 1.0)
    return rounded


__all__ = [
    "IsotonicFitResult",
    "LinearFitResult",
    "RadonDetectionResult",
    "RegressionLineScoreResult",
    "detect_trajectory_radon",
    "fit_isotonic_trajectory",
    "fit_linear_trajectory",
    "regression_line_score",
]
