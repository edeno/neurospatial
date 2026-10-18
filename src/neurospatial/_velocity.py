"""One definition of velocity and speed for every speed gate in the package.

Differencing raw positions turns tracking jitter into speed: 0.5 cm of camera
noise reads as about 20 cm/s at 30 Hz and 200 cm/s at 500 Hz. The velocity
vector is therefore smoothed with a Gaussian of ``smooth_window`` seconds
before its magnitude is taken, as in non_local_detector's ``get_velocity`` and
spyglass's position processing (whose speed smoothing default is 0.1 s).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

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
        Observed intervals (gaps, epochs). Intervals with a non-finite end or
        zero duration are also excluded, so a run never spans missing tracking
        or a duplicate timestamp.
    smooth_window : float
        Standard deviation, in seconds, of the Gaussian applied to each
        velocity component. 0 keeps the unsmoothed centered-secant velocity.

    Returns
    -------
    NDArray[np.float64], shape (n_samples, n_dims)
        Velocity in position units per second; NaN at samples in no run of
        at least two samples.
    """
    from neurospatial._intervals import run_sample_bounds

    if smooth_window < 0 or not np.isfinite(smooth_window):
        raise ValueError(
            f"smooth_window must be a finite number of seconds >= 0, got {smooth_window}."
        )
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim == 1:
        positions = positions[:, np.newaxis]
    finite = np.all(np.isfinite(positions), axis=1)
    # A zero-length interval (duplicate timestamps) has no defined velocity.
    mask = (
        np.asarray(interval_mask, dtype=bool)
        & finite[:-1]
        & finite[1:]
        & (np.diff(times) > 0)
    )
    velocity = np.full(positions.shape, np.nan, dtype=np.float64)
    for first, last in run_sample_bounds(mask):
        run = slice(int(first), int(last) + 1)
        run_times = times[run]
        run_velocity = _secant_velocity(run_times, positions[run])
        if smooth_window > 0:
            sigma = smooth_window / float(np.median(np.diff(run_times)))
            run_velocity = gaussian_filter1d(
                run_velocity, sigma, axis=0, mode="nearest"
            )
        velocity[run] = run_velocity
    return velocity


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
