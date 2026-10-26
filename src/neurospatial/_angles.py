"""Angle wrapping shared by the circular statistics and egocentric frames."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def wrap_angle(angle: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Wrap angle to (-π, π].

    A half-turn in either direction maps to +π, so an egocentric bearing
    directly behind the animal is +π.

    Parameters
    ----------
    angle : NDArray[np.float64]
        Angles in radians.

    Returns
    -------
    NDArray[np.float64]
        Angles wrapped to (-π, π].

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.stats.circular import wrap_angle
    >>> wrap_angle(np.array([3 * np.pi / 2, np.pi, -np.pi]))
    array([-1.57079633,  3.14159265,  3.14159265])
    """
    # x % (-2π) lies in (-2π, 0], so shifting by -π then +π gives (-π, π].
    wrapped = (angle - np.pi) % (-2 * np.pi) + np.pi
    # A remainder just below 0 can round to -2π, which would give -π.
    out: NDArray[np.float64] = np.where(wrapped <= -np.pi, np.pi, wrapped)
    return out
