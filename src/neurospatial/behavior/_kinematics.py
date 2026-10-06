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

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
    positions : ndarray, shape (n_samples, n_dims)
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        ``observed_interval_mask`` output.

    Returns
    -------
    ndarray, shape (n_samples - 1, n_dims)
    """
    dt = np.diff(times)
    velocity: NDArray[np.float64] = np.diff(positions, axis=0) / dt[:, np.newaxis]
    velocity[~interval_mask] = np.nan
    return velocity
