"""Angle wrapping and vector turns shared across the package."""

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


def cross_2d(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Z component of the cross product ``a x b``; ``a``, ``b`` are ``(n, 2)``."""
    cross: NDArray[np.float64] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return cross


def signed_turns(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Turn from ``a`` to ``b`` in (-π, π], as `wrap_angle`.

    Parameters
    ----------
    a, b : NDArray[np.float64], shape (n, 2)
        Incoming and outgoing planar vectors.

    Returns
    -------
    NDArray[np.float64], shape (n,)
        Counterclockwise turns are positive; a reversal is +π whatever the
        sign of the zero cross product.
    """
    angles: NDArray[np.float64] = np.arctan2(cross_2d(a, b), np.sum(a * b, axis=1))
    angles[angles == -np.pi] = np.pi
    return angles


def unit_vector_angles(
    u: NDArray[np.float64], w: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Unsigned angle between unit vectors, in [0, π].

    Kahan's formula ``2 atan2(|u - w|, |u + w|)`` stays accurate near 0 and
    π, where ``arccos(u . w)`` loses precision.

    Parameters
    ----------
    u, w : NDArray[np.float64], shape (n, n_dims)
        Unit vectors, one per row.

    Returns
    -------
    NDArray[np.float64], shape (n,)
    """
    angles: NDArray[np.float64] = 2.0 * np.arctan2(
        np.linalg.norm(u - w, axis=1), np.linalg.norm(u + w, axis=1)
    )
    return angles
