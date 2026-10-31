"""One definition of velocity and speed for every speed gate in the package.

Differencing raw positions turns tracking jitter into speed: a still animal
tracked with 0.5 cm of camera noise per coordinate reads as a mean speed of
about 13 cm/s at 30 Hz and 220 cm/s at 500 Hz (centered secant). The velocity
vector is therefore smoothed in time with a Gaussian of ``smooth_window``
seconds before its magnitude is taken, as in non_local_detector's
``get_velocity`` and spyglass's position processing (whose speed smoothing
default is 0.1 s); after the default 0.1 s the same jitter reads as about
1.3 cm/s at 30 Hz and 0.3 cm/s at 500 Hz.
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
    """Velocity at each sample, smoothed within each observed run.

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
        side. Zero-duration intervals end a run, so it never spans a
        duplicate timestamp.
    smooth_window : float
        Standard deviation, in seconds, of the Gaussian that averages each
        velocity component over time within the run
        (:func:`smooth_run_in_time`): samples are weighted by their time
        difference, so the window stays ``smooth_window`` seconds wide across
        irregular sampling and bridged gaps. 0 keeps the unsmoothed
        centered-secant velocity.

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
    # A zero-length interval (duplicate timestamps) has no defined velocity.
    mask = np.asarray(interval_mask, dtype=bool) & (np.diff(times) > 0)
    velocity = np.full(positions.shape, np.nan, dtype=np.float64)
    # Missing samples inside a run are skipped, exactly like deleted rows.
    for run in valid_run_samples(mask, finite):
        if run.size < 2:
            continue
        run_times = times[run]
        run_velocity = _secant_velocity(run_times, positions[run])
        if smooth_window > 0:
            run_velocity = smooth_run_in_time(run_velocity, run_times, smooth_window)
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
        ``4 * sigma`` apart.
    """
    from neurospatial._intervals import _timestamp_rounding

    reach = 4.0 * sigma + _timestamp_rounding(times[0], times[-1])
    # The widest reach of any sample, forward or backward, bounds the lags.
    index = np.arange(times.size)
    lo = np.searchsorted(times, times - reach, side="left")
    hi = np.searchsorted(times, times + reach, side="right")
    widest = int(max(np.max(index - lo), np.max(hi - 1 - index)))
    for lag in range(1, widest + 1):
        weight = times[lag:] - times[:-lag]
        far = weight > reach
        # In place: exp(-0.5 * (dt / sigma) ** 2), the costly step of smoothing.
        weight /= sigma
        np.square(weight, out=weight)
        weight *= -0.5
        np.exp(weight, out=weight)
        weight[far] = 0.0
        yield lag, weight


def smooth_in_time(
    values: NDArray[np.float64],
    times: NDArray[np.float64],
    run_bounds: NDArray[np.intp],
    sigma: float,
) -> NDArray[np.float64]:
    """Gaussian-weighted mean in time of the finite rows of each run.

    Runs are smoothed independently (:func:`smooth_run_in_time`).

    Parameters
    ----------
    values : NDArray[np.float64], shape (n_samples,) or (n_samples, n_dims)
    times : NDArray[np.float64], shape (n_samples,)
        Strictly increasing within each run, in seconds.
    run_bounds : NDArray[np.intp], shape (n_runs, 2)
        Inclusive first and last sample of each run.
    sigma : float
        Positive standard deviation in seconds.

    Returns
    -------
    NDArray[np.float64], same shape as ``values``
        NaN at non-finite rows and outside every run.
    """
    smoothed = np.full(values.shape, np.nan)
    for first, last in run_bounds:
        run = slice(first, last + 1)
        smoothed[run] = smooth_run_in_time(values[run], times[run], sigma)
    return smoothed


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
        # Each pair (i, i + k) within reach adds to both samples.
        for k, w in gaussian_pair_weights(times, sigma):
            buffer = product[: w.size]
            for row, total in zip(weighted, sums, strict=True):
                np.multiply(w, row[k:], out=buffer)
                total[:-k] += buffer
                np.multiply(w, row[:-k], out=buffer)
                total[k:] += buffer
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
