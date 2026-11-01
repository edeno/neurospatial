"""Smoothed velocity of one run, computed independently of the package.

The package sums Gaussian weights pair by pair over sample lags. This
reference instead lays out, for every sample, the window of neighbours within
four standard deviations in time and solves the weighted least-squares line
``p_j ~ a + b * (t_j - t_i)`` with weights ``exp(-dt**2 / (2 * sigma**2))``
as a 2 x 2 linear system; the velocity is the slope ``b``. A sample with no
neighbour in its window keeps its centered-secant velocity.
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
    padded_p = np.pad(positions, ((half, half), (0, 0)))
    window_t = sliding_window_view(padded_t, 2 * half + 1)
    window_p = sliding_window_view(padded_p, 2 * half + 1, axis=0)
    dt = window_t - times[:, None]
    with np.errstate(invalid="ignore"):
        # Pairs exactly 4 sigma apart count, whatever the timestamp rounding;
        # NaN padding compares False.
        inside = np.abs(dt) <= 4 * sigma + 1e-9
    dt = np.nan_to_num(dt)
    weight = np.where(inside, np.exp(-0.5 * (dt / sigma) ** 2), 0.0)
    # Normal equations of the weighted line fit around each sample.
    normal = np.empty((times.size, 2, 2))
    normal[:, 0, 0] = weight.sum(axis=1)
    normal[:, 0, 1] = normal[:, 1, 0] = (weight * dt).sum(axis=1)
    normal[:, 1, 1] = (weight * dt**2).sum(axis=1)
    rhs = np.stack(
        [
            (window_p * weight[:, None, :]).sum(axis=2),
            (window_p * (weight * dt)[:, None, :]).sum(axis=2),
        ],
        axis=1,
    )
    alone = inside.sum(axis=1) < 2
    normal[alone] = np.eye(2)
    slope = np.linalg.solve(normal, rhs)[:, 1, :]
    return np.where(alone[:, None], secant, slope)


def reference_speed(times, positions, sigma):
    """Magnitude of :func:`reference_velocity`, shape (n_samples,)."""
    return np.linalg.norm(reference_velocity(times, positions, sigma), axis=1)
