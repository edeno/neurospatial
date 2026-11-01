"""One definition of velocity and speed for every speed gate in the package.

Differencing raw positions turns tracking jitter into speed: a still animal
tracked with 0.5 cm of camera noise per coordinate reads as a mean speed of
about 13 cm/s at 30 Hz and 220 cm/s at 500 Hz (centered secant). The velocity
at each sample is therefore the slope of a straight line fitted to the
positions around it, weighted by a Gaussian of ``smooth_window`` seconds in
time (spyglass's speed smoothing default is 0.1 s); after the default 0.1 s
the same jitter reads as about 1.4 cm/s at 30 Hz and 0.3 cm/s at 500 Hz.
The intercept of the same fit is the smoothed position of
``smooth_positions``, so velocity, speed and direction of motion agree with
the smoothed path. A local line has no lag in the middle of a run and, unlike
a mean of frame-to-frame velocities, is not dominated by the single noisy
one-sided difference at the end of a run: with 0.5 cm of jitter the speed at
the last sample of a 25 cm/s run is off by about 1 cm/s instead of 3-6 cm/s.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._validation import is_finite_real

#: Gaussian standard deviation (seconds) of the default velocity smoothing.
DEFAULT_SPEED_SMOOTHING = 0.1


def smoothed_velocity(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
    *,
    smooth_window: float,
) -> NDArray[np.float64]:
    """Velocity at each sample, the slope of a local line fit within each run.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Strictly increasing timestamps in seconds.
    positions : NDArray[np.float64], shape (n_samples, n_dims) or (n_samples,)
        Positions in environment units.
    interval_mask : NDArray[np.bool_], shape (n_samples - 1,)
        Usable intervals (gaps, epochs), with missing samples already bridged
        (``ObservationSupport.interval_mask``). Within each run, samples with a
        non-finite coordinate are skipped as if deleted, so velocity across a
        bridged gap uses the elapsed time between the valid samples on either
        side.
    smooth_window : float
        Standard deviation, in seconds, of the Gaussian that weights the
        samples of the run in a local straight-line fit around each sample
        (:func:`time_smoothed_positions`); the velocity is the slope of that
        line. Samples are weighted by their time difference, so the window
        stays ``smooth_window`` seconds wide across irregular sampling and
        bridged gaps. A sample with no other sample within four standard
        deviations, and every sample when ``smooth_window`` is 0, keeps the
        unsmoothed centered-secant velocity.

    Returns
    -------
    NDArray[np.float64], shape (n_samples, n_dims)
        Velocity in position units per second; NaN at missing samples and at
        samples in no run of at least two valid samples.
    """
    from neurospatial._intervals import valid_run_samples

    if not is_finite_real(smooth_window, positive=False):
        raise ValueError(
            _format_error(
                "smooth_window must be a finite number of seconds >= 0; got "
                f"{smooth_window!r}.",
                why="smooth_window is the standard deviation, in seconds, of "
                "the Gaussian that averages the velocity over time.",
                fix="pass smooth_window=0.0 for the unsmoothed velocity, or a "
                "number of seconds such as the default "
                f"smooth_window={DEFAULT_SPEED_SMOOTHING}.",
            )
        )
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim == 1:
        positions = positions[:, np.newaxis]
    finite = np.all(np.isfinite(positions), axis=1)
    mask = np.asarray(interval_mask, dtype=bool)
    velocity = np.full(positions.shape, np.nan, dtype=np.float64)
    # Missing samples inside a run are skipped, exactly like deleted rows.
    for run in valid_run_samples(mask, finite):
        if run.size < 2:
            continue
        run_times, run_positions = times[run], positions[run]
        run_velocity = _secant_velocity(run_times, run_positions)
        if smooth_window > 0:
            _, slope, fitted = _local_line_fits(run_times, run_positions, smooth_window)
            run_velocity[fitted] = slope[fitted]
        velocity[run] = run_velocity
    return velocity


def gaussian_pair_weights(
    times: NDArray[np.float64], sigma: float
) -> Iterator[tuple[int, NDArray[np.float64]]]:
    """Gaussian weights of every pair of samples within four standard deviations.

    Weighting pairs by their time difference, not by how many samples lie
    between them, keeps a smoothing window ``sigma`` seconds wide across
    irregular sampling and bridged gaps of missing samples. A pair exactly
    ``4 * sigma`` apart, as at 30 Hz with ``sigma = 0.1``, is kept whatever
    the rounding of its timestamps, so the window stays symmetric and the
    result does not depend on the clock's offset.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Strictly increasing, in seconds.
    sigma : float
        Positive standard deviation in seconds.

    Yields
    ------
    lag : int
        Sample lag ``k``, from 1 up to the widest lag of any pair at most
        ``4 * sigma`` apart.
    weight : NDArray[np.float64], shape (n_samples - lag,)
        ``exp(-dt**2 / (2 * sigma**2))`` of the pair ``(i, i + lag)``, where
        ``dt = times[i + lag] - times[i]``; 0 for pairs more than
        ``4 * sigma`` apart. It is a view of a buffer the next lag
        overwrites, so it is valid only until the next iteration.
    """
    from neurospatial._intervals import _timestamp_rounding

    reach = 4.0 * sigma + _timestamp_rounding(times[0], times[-1])
    # The widest reach of any sample, forward or backward, bounds the lags.
    index = np.arange(times.size)
    lo = np.searchsorted(times, times - reach, side="left")
    hi = np.searchsorted(times, times + reach, side="right")
    widest = int(max(np.max(index - lo), np.max(hi - 1 - index)))
    weights = np.empty(max(times.size - 1, 0))
    beyond = np.empty(weights.size, dtype=bool)
    for lag in range(1, widest + 1):
        weight = weights[: times.size - lag]
        far = beyond[: weight.size]
        np.subtract(times[lag:], times[:-lag], out=weight)
        np.greater(weight, reach, out=far)
        # In place: exp(-0.5 * (dt / sigma) ** 2), the costly step of smoothing.
        weight /= sigma
        np.square(weight, out=weight)
        weight *= -0.5
        np.exp(weight, out=weight)
        np.putmask(weight, far, 0.0)
        yield lag, weight


def smooth_run_in_time(
    values: NDArray[np.float64], times: NDArray[np.float64], sigma: float
) -> NDArray[np.float64]:
    """Gaussian-weighted mean in time of the finite rows of one run.

    Each finite row becomes the mean of the finite rows, weighted by a
    Gaussian of their time difference (standard deviation ``sigma`` seconds,
    cut at four standard deviations; :func:`gaussian_pair_weights`). Rows
    with a non-finite entry get no weight and stay NaN. Near the ends of the
    run, and next to a bridged gap, the window holds fewer samples and the
    weights are renormalized, so a constant stays constant; this one-sided
    mean lags a changing value by up to ``sigma * sqrt(2 / pi)`` (about
    ``0.8 * sigma``) seconds at the first and last samples.

    Parameters
    ----------
    values : NDArray[np.float64], shape (n_samples,) or (n_samples, n_dims)
    times : NDArray[np.float64], shape (n_samples,)
        Strictly increasing, in seconds.
    sigma : float
        Positive standard deviation in seconds.

    Returns
    -------
    NDArray[np.float64], same shape as ``values``
    """
    table = values.reshape(values.shape[0], -1)
    smoothed = np.full(table.shape, np.nan)
    defined = np.all(np.isfinite(table), axis=1)
    if defined.any():
        # One row per column plus the weight row, contiguous in time.
        weighted = np.vstack([np.where(defined[:, None], table, 0.0).T, defined])
        sums = weighted.copy()
        product = np.empty(table.shape[0])
        # With every row defined the weight row is all ones, so w adds to
        # its sums unscaled.
        ones = bool(defined.all())
        scaled = weighted[:-1] if ones else weighted
        # Each pair (i, i + k) within reach adds to both samples.
        for k, w in gaussian_pair_weights(times, sigma):
            buffer = product[: w.size]
            for row, total in zip(scaled, sums[: len(scaled)], strict=True):
                np.multiply(w, row[k:], out=buffer)
                total[:-k] += buffer
                np.multiply(w, row[:-k], out=buffer)
                total[k:] += buffer
            if ones:
                sums[-1, :-k] += w
                sums[-1, k:] += w
        smoothed[defined] = (sums[:-1, defined] / sums[-1, defined]).T
    result: NDArray[np.float64] = smoothed.reshape(values.shape)
    return result


def _secant_velocity(
    times: NDArray[np.float64], positions: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Centered-secant velocity, one-sided at the ends, shape (n, n_dims).

    ``(p[i+1] - p[i-1]) / (t[i+1] - t[i-1])`` is exactly zero for a constant
    coordinate, unlike ``np.gradient``'s non-uniform-spacing weights, whose
    roundoff gives a still animal a speed of ~1e-16 and due West a heading of
    -pi. ``times`` has at least two strictly increasing entries.
    """
    velocity = np.empty_like(positions)
    velocity[0] = (positions[1] - positions[0]) / (times[1] - times[0])
    velocity[-1] = (positions[-1] - positions[-2]) / (times[-1] - times[-2])
    if len(times) > 2:
        velocity[1:-1] = (positions[2:] - positions[:-2]) / (times[2:] - times[:-2])[
            :, np.newaxis
        ]
    return velocity


def smoothed_speed(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
    *,
    smooth_window: float,
) -> NDArray[np.float64]:
    """Magnitude of :func:`smoothed_velocity`, shape (n_samples,)."""
    velocity = smoothed_velocity(
        times, positions, interval_mask, smooth_window=smooth_window
    )
    speed: NDArray[np.float64] = np.linalg.norm(velocity, axis=1)
    return speed


def time_smoothed_positions(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
    sigma: float = DEFAULT_SPEED_SMOOTHING,
) -> NDArray[np.float64]:
    """Positions smoothed by Gaussian-weighted local linear fits in time.

    Each valid sample is replaced by the value at its own time of a straight
    line fitted, within its observed run, to the samples weighted by a
    Gaussian of their time difference (standard deviation ``sigma``
    seconds, cut at four standard deviations).
    A local linear fit leaves motion at constant velocity unchanged, also at
    the ends of a run, and averages tracking jitter away. Weighting by time
    rather than sample distance means a bridged gap much longer than the
    kernel is not smoothed across. Missing rows stay NaN.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
    positions : ndarray, shape (n_samples, n_dims)
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        ``ObservationSupport.interval_mask``.
    sigma : float, default=DEFAULT_SPEED_SMOOTHING
        Standard deviation of the Gaussian, in seconds; 0 returns a copy of
        ``positions``.

    Returns
    -------
    ndarray, shape (n_samples, n_dims)
    """
    from neurospatial._intervals import valid_run_samples

    finite = np.all(np.isfinite(positions), axis=1)
    smoothed = positions.astype(np.float64, copy=True)
    if sigma == 0:
        return smoothed
    for run in valid_run_samples(np.asarray(interval_mask, dtype=bool), finite):
        if run.size < 2:
            continue
        shift, _, _ = _local_line_fits(times[run], positions[run], sigma)
        smoothed[run] = positions[run] + shift
    return smoothed


def _local_line_fits(
    times: NDArray[np.float64], positions: NDArray[np.float64], sigma: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Gaussian-weighted straight-line fit around each sample of one run.

    Around each sample ``i`` the line ``p_j - p_i ~ a + b * (t_j - t_i)`` is
    fitted by least squares with weights ``exp(-(t_j - t_i)**2 / (2 sigma**2))``
    (:func:`gaussian_pair_weights`). Working in differences from each sample
    keeps a still animal exactly still and its velocity exactly zero.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Strictly increasing, in seconds; at least two samples.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Finite positions.
    sigma : float
        Positive standard deviation in seconds.

    Returns
    -------
    shift : NDArray[np.float64], shape (n_samples, n_dims)
        Intercept ``a``: the fitted position minus the sample's own; 0 where
        not ``fitted``.
    slope : NDArray[np.float64], shape (n_samples, n_dims)
        Slope ``b``, the velocity in position units per second; 0 where not
        ``fitted``.
    fitted : NDArray[np.bool_], shape (n_samples,)
        False for a sample with no other sample within reach, where a line is
        undetermined.
    """
    # Each sample weighs itself by 1; each pair (i, i + k) within reach adds
    # to both samples, with lag and deviation negated for the later.
    s0 = np.ones(times.size)
    s1 = np.zeros(times.size)
    s2 = np.zeros(times.size)
    m0 = np.zeros_like(positions)
    m1 = np.zeros_like(positions)
    for k, weight in gaussian_pair_weights(times, sigma):
        lag = times[k:] - times[:-k]
        weighted_lag = weight * lag
        weighted_square = weighted_lag * lag
        deviation = positions[k:] - positions[:-k]
        weighted_deviation = weight[:, None] * deviation
        lagged_deviation = weighted_lag[:, None] * deviation
        s0[:-k] += weight
        s0[k:] += weight
        s1[:-k] += weighted_lag
        s1[k:] -= weighted_lag
        s2[:-k] += weighted_square
        s2[k:] += weighted_square
        m0[:-k] += weighted_deviation
        m0[k:] -= weighted_deviation
        m1[:-k] += lagged_deviation
        m1[k:] += lagged_deviation
    det = s0 * s2 - s1**2
    fitted = det > 1e-12 * s0 * np.maximum(s2, np.finfo(float).tiny)
    safe_det = np.where(fitted, det, 1.0)[:, None]
    shift = np.where(
        fitted[:, None], (s2[:, None] * m0 - s1[:, None] * m1) / safe_det, 0.0
    )
    slope = np.where(
        fitted[:, None], (s0[:, None] * m1 - s1[:, None] * m0) / safe_det, 0.0
    )
    return shift, slope, fitted


def validated_smooth_window(smooth_window: float, *, call: str) -> float:
    """A finite ``smooth_window`` of seconds >= 0, as a float."""
    if not is_finite_real(smooth_window, positive=False):
        raise ValueError(
            _format_error(
                f"{call}() needs smooth_window to be a finite number of seconds "
                f">= 0; got {smooth_window!r}.",
                why="smooth_window is the standard deviation, in seconds, of the "
                "Gaussian that smooths positions in time before distance is "
                "measured.",
                fix="pass smooth_window=0.0 for raw steps, or a number of "
                f"seconds such as the default smooth_window={DEFAULT_SPEED_SMOOTHING}.",
            )
        )
    return float(smooth_window)
