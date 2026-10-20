"""Per-interval kinematics that never span an invalid interval."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._velocity import DEFAULT_SPEED_SMOOTHING


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


#: Speed below which an animal counts as stationary, in centimeters per second.
DEFAULT_MIN_SPEED_CM_PER_S = 5.0


def resolve_min_speed(env: Any, min_speed: float | None, *, call: str) -> float:
    """``min_speed``, or 5 cm/s expressed in ``env.units`` when it is None.

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
        If ``min_speed`` is None and ``env.units`` is not a metric length.
    """
    from neurospatial._exceptions import _format_error
    from neurospatial._units import units_per_centimeter

    if min_speed is not None:
        return float(min_speed)
    per_cm = units_per_centimeter(env.units)
    if per_cm is None:
        raise ValueError(
            _format_error(
                f"{call} needs min_speed: the environment's units "
                f"({env.units!r}) are not a known length, so the 5 cm/s default "
                "cannot be expressed in them.",
                why="Why: min_speed is in position units per second.",
                fix="pass min_speed=<threshold in position units per second>, "
                "or declare the units with env.with_metadata(units='cm')",
            )
        )
    return DEFAULT_MIN_SPEED_CM_PER_S * per_cm


def _validate_velocity_positions(
    positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Reject positions without x and y columns, naming the shape as passed.

    Call it on the caller's array, before any reshaping, so the message
    shows the shape the scientist passed.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError(
            f"positions must be a 2-D array with at least x/y coordinates; "
            f"got shape {positions.shape}.\n"
            "Why: a movement heading is the direction of travel in the x-y "
            "plane, so it needs both x and y; 1-D positions (such as a "
            "linearized track) have only one coordinate.\n"
            "Fix: pass positions with shape (n_samples, 2), for example "
            "np.column_stack([x, y]), with one timestamp per row."
        )
    return positions


def _validate_velocity_times(
    times: NDArray[np.float64], n_samples: int
) -> NDArray[np.float64]:
    """Validate the timestamp array before building an interval mask."""
    try:
        times = np.asarray(times, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"times must be numeric timestamps; got {times!r}.\n"
            "Why: velocities need seconds between consecutive samples.\n"
            "Fix: pass neurospatial.behavior.heading_from_velocity(times, "
            "positions) with a numeric 1-D timestamp array, one timestamp per "
            "position."
        ) from error
    problems = []
    if times.ndim != 1:
        problems.append(f"times must be 1-D; got shape {times.shape}")
    else:
        if len(times) != n_samples:
            problems.append(
                f"times and positions must have the same length; got "
                f"{len(times)} times and {n_samples} positions"
            )
        finite = np.isfinite(times)
        if not finite.all():
            index = int(np.flatnonzero(~finite)[0])
            problems.append(
                f"times must be finite; got {times[index]!r} at index {index}"
            )
        elif np.any(np.diff(times) <= 0):
            index = int(np.flatnonzero(np.diff(times) <= 0)[0])
            problems.append(
                f"times must be strictly increasing; got "
                f"{times[index]!r}, {times[index + 1]!r} at indices "
                f"{index}, {index + 1}"
            )
    if problems:
        raise ValueError(
            "; ".join(problems) + ".\n"
            "Why: each position needs a finite timestamp and a positive "
            "elapsed interval for velocity.\n"
            "Fix: pass neurospatial.behavior.heading_from_velocity(times, "
            "positions) with a 1-D timestamp array of matching length; remove "
            "non-finite samples and sort/de-duplicate positions and times "
            "together."
        )
    return times


def _velocity_heading_and_speed(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    interval_mask: NDArray[np.bool_],
    smooth_window: float = DEFAULT_SPEED_SMOOTHING,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Movement heading and horizontal speed from the smoothed velocity.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Finite, strictly increasing timestamps in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Position coordinates in environment units; heading and speed use the
        first two (x, y).
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        Observed intervals supplied by the shared mask helper.
    smooth_window : float, default=0.1
        Gaussian standard deviation in seconds applied to each velocity
        component within each run (:func:`neurospatial.behavior.compute_speed`).

    Returns
    -------
    heading, speed : ndarray, shape (n_samples,)
        Radians and position units per second; samples in no run are NaN.
    """
    from neurospatial._validation import validate_finite

    positions = _validate_velocity_positions(positions)
    times = _validate_velocity_times(times, len(positions))
    validate_finite(positions, name="positions")
    if len(positions) < 2:
        raise ValueError(
            f"Cannot compute heading: insufficient trajectory data; need at "
            f"least 2 position samples, got {len(positions)}.\n"
            "Why: heading needs a position change over time.\n"
            "Fix: pass at least two aligned position/timestamp samples or "
            "use neurospatial.behavior.heading_from_body_orientation() for "
            "single-frame pose data."
        )
    return _movement_heading_and_speed(
        times, positions, interval_mask, smooth_window=smooth_window
    )


def _movement_heading_and_speed(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
    *,
    smooth_window: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Direction and horizontal speed of the smoothed velocity; no validation.

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

    Returns
    -------
    heading, speed : ndarray, shape (n_samples,)
        ``arctan2(vy, vx)`` in radians and ``hypot(vx, vy)`` in position units
        per second; NaN at samples in no run.
    """
    from neurospatial._velocity import smoothed_velocity

    velocity = smoothed_velocity(
        times, positions[:, :2], interval_mask, smooth_window=smooth_window
    )
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    heading = np.arctan2(velocity[:, 1], velocity[:, 0])
    return heading, speed
