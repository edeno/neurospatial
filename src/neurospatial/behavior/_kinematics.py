"""Per-interval kinematics that never span an invalid interval."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def interval_velocity(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Velocity of each interval, NaN where the interval is invalid.

    A sample with a non-finite coordinate is missing and is bridged like a
    deleted row: every usable interval inside the gap between two valid
    samples gets that gap's velocity, its displacement over its elapsed
    time. A zero-length gap (duplicate timestamps) has no defined velocity
    and is NaN, like an unobserved interval, rather than infinite.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
    positions : ndarray, shape (n_samples, n_dims) or (n_samples,)
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        Usable intervals with missing samples bridged
        (``ObservationSupport.interval_mask``).

    Returns
    -------
    ndarray, shape (n_samples - 1, n_dims)
    """
    from neurospatial._intervals import gap_endpoints

    # 1-D positions (a linear track) are one coordinate per sample.
    positions = np.asarray(positions)
    if positions.ndim == 1:
        positions = positions[:, np.newaxis]
    start, end, linked = gap_endpoints(
        np.asarray(np.isfinite(positions).all(axis=1), dtype=bool)
    )
    dt = times[end] - times[start]
    defined = interval_mask & linked & (dt > 0)
    velocity: NDArray[np.float64] = np.full(
        (dt.size, positions.shape[1]), np.nan, dtype=np.float64
    )
    velocity[defined] = (positions[end] - positions[start])[defined] / dt[
        defined, np.newaxis
    ]
    return velocity


#: Speed below which an animal counts as stationary, in centimeters per second.
DEFAULT_MIN_SPEED_CM_PER_S = 5.0


def validated_min_speed(min_speed: float, *, call: str) -> float:
    """A finite, non-negative movement speed gate in position units per second.

    Parameters
    ----------
    min_speed : float
        Speed below which the direction of movement is not observed.
    call : str
        Public function name, used in the error.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``min_speed`` is NaN, infinite or negative.
    """
    from neurospatial._exceptions import _format_error

    value = float(min_speed)
    if not (np.isfinite(value) and value >= 0):
        raise ValueError(
            _format_error(
                f"{call}() needs a finite, non-negative min_speed; got {min_speed!r}.",
                why=(
                    "min_speed is the speed below which the direction of "
                    "movement is not observed; a NaN or negative gate would "
                    "silently drop or keep every heading."
                ),
                fix=(
                    "pass min_speed in position units per second, e.g. 5.0 for "
                    "5 cm/s with positions in cm."
                ),
            )
        )
    return value


def resolve_min_speed(env: Any, min_speed: float | None, *, call: str) -> float:
    """``min_speed`` checked, or 5 cm/s expressed in ``env.units`` when None.

    Parameters
    ----------
    env : Environment
        Environment whose ``units`` give the positions' length unit.
    min_speed : float or None
        Threshold in position units per second, or None for the default.
    call : str
        Public function name, used in the error.

    Returns
    -------
    float
        The threshold in position units per second.

    Raises
    ------
    ValueError
        If ``min_speed`` is NaN, infinite or negative, or it is None and
        ``env.units`` is not a metric length.
    """
    from neurospatial._units import default_speed

    if min_speed is not None:
        return validated_min_speed(min_speed, call=call)
    return default_speed(
        env.units,
        DEFAULT_MIN_SPEED_CM_PER_S,
        name="min_speed",
        call=f"{call}()",
        problem="not a known length, so the 5 cm/s default",
        placeholder="threshold",
    )


def _validate_velocity_positions(
    positions: NDArray[np.float64], *, call: str
) -> NDArray[np.float64]:
    """Reject positions without x and y columns, naming the shape as passed.

    Call it on the caller's array, before any reshaping, so the message
    shows the shape the scientist passed.

    Parameters
    ----------
    positions : array-like
        Positions exactly as the scientist passed them.
    call : str
        Public function name, used in the message.
    """
    from neurospatial._exceptions import _format_error

    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError(
            _format_error(
                f"{call}() needs positions with x and y columns; got shape "
                f"{positions.shape}.",
                why=(
                    "Why: a movement heading is the direction of travel in the "
                    "x-y plane, so it needs both x and y; 1-D positions (such as "
                    "a linearized track) have only one coordinate."
                ),
                fix=(
                    "pass positions with shape (n_samples, 2), for example "
                    "np.column_stack([x, y]), with one timestamp per row."
                ),
            )
        )
    return positions


def _movement_heading_and_speed(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
    *,
    smooth_window: float,
    min_speed: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Direction of movement and horizontal speed of the smoothed velocity.

    No validation: the public caller checks the inputs and ``min_speed``.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Strictly increasing timestamps in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Positions in environment units, ``n_dims >= 2``; only x and y are used.
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        Usable intervals.
    smooth_window : float
        Gaussian standard deviation in seconds (see ``smoothed_velocity``).
    min_speed : float
        Speed in position units per second below which the animal is not
        moving.

    Returns
    -------
    heading : ndarray, shape (n_samples,)
        ``arctan2(vy, vx)`` in radians where the animal moves (speed above 0
        and at least ``min_speed``); NaN where it does not, since a still
        animal has no direction of movement (``arctan2(0, 0)`` would read as
        East), and at samples in no run.
    speed : ndarray, shape (n_samples,)
        ``hypot(vx, vy)`` in position units per second; NaN at samples in no
        run.
    """
    from neurospatial._velocity import smoothed_velocity

    velocity = smoothed_velocity(
        times, positions[:, :2], interval_mask, smooth_window=smooth_window
    )
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    moving = (speed > 0) & (speed >= min_speed)
    heading = np.where(moving, np.arctan2(velocity[:, 1], velocity[:, 0]), np.nan)
    return heading, speed
