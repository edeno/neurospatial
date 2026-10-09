"""Per-interval kinematics that never span an invalid interval."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def interval_velocity(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Velocity of each interval, NaN where the interval is invalid.

    A zero-length interval (duplicate timestamps) has no defined velocity and
    is NaN, like an unobserved interval, rather than infinite.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
    positions : ndarray, shape (n_samples, n_dims) or (n_samples,)
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        ``observed_interval_mask`` output.

    Returns
    -------
    ndarray, shape (n_samples - 1, n_dims)
    """
    # 1-D positions (a linear track) are one coordinate per sample.
    positions = np.asarray(positions)
    if positions.ndim == 1:
        positions = positions[:, np.newaxis]
    dt = np.diff(times)
    defined = interval_mask & (dt > 0)
    velocity: NDArray[np.float64] = np.full(
        (dt.size, positions.shape[1]), np.nan, dtype=np.float64
    )
    velocity[defined] = np.diff(positions, axis=0)[defined] / dt[defined, np.newaxis]
    return velocity
