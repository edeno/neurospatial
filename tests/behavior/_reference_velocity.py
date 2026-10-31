"""Smoothed velocity of one run, computed independently of the package.

The package sums Gaussian weights pair by pair over sample lags. This
reference instead lays out, for every sample, the window of neighbours within
four standard deviations in time and averages their centered-secant
velocities with weights ``exp(-dt**2 / (2 * sigma**2))``.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def reference_velocity(times, positions, sigma):
    """Velocity of one run of valid samples, shape (n_samples, n_dims).

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Strictly increasing, in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Finite positions.
    sigma : float
        Gaussian standard deviation in seconds; 0 returns the secant.
    """
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    previous = np.r_[0, np.arange(times.size - 1)]
    following = np.r_[np.arange(1, times.size), times.size - 1]
    secant = (positions[following] - positions[previous]) / (
        times[following] - times[previous]
    )[:, None]
    if sigma == 0:
        return secant
    half = int(np.ceil(4 * sigma / np.min(np.diff(times)))) + 1
    padded_t = np.pad(times, half, constant_values=np.nan)
    padded_v = np.pad(secant, ((half, half), (0, 0)))
    window_t = sliding_window_view(padded_t, 2 * half + 1)
    window_v = sliding_window_view(padded_v, 2 * half + 1, axis=0)
    dt = window_t - times[:, None]
    with np.errstate(invalid="ignore"):
        # Pairs exactly 4 sigma apart count, whatever the timestamp rounding;
        # NaN padding compares False.
        inside = np.abs(dt) <= 4 * sigma + 1e-9
    weight = np.where(inside, np.exp(-0.5 * (np.nan_to_num(dt) / sigma) ** 2), 0.0)
    return (window_v * weight[:, None, :]).sum(axis=2) / weight.sum(axis=1)[:, None]


def reference_speed(times, positions, sigma):
    """Magnitude of :func:`reference_velocity`, shape (n_samples,)."""
    return np.linalg.norm(reference_velocity(times, positions, sigma), axis=1)
