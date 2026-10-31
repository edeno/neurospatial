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
    time; ``times`` are strictly increasing, so every gap has a duration.

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
    defined = interval_mask & linked
    velocity: NDArray[np.float64] = np.full(
        (dt.size, positions.shape[1]), np.nan, dtype=np.float64
    )
    velocity[defined] = (positions[end] - positions[start])[defined] / dt[
        defined, np.newaxis
    ]
    return velocity


def observed_steps(
    sample_valid: NDArray[np.bool_], interval_mask: NDArray[np.bool_]
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.bool_]]:
    """The observed step of each interval, between valid samples.

    A bridged stretch of missing samples is one step, from the valid sample
    before it to the valid sample after it, counted once: at the stretch's
    last interval, the one ending at a valid sample.

    Parameters
    ----------
    sample_valid : ndarray of bool, shape (n_samples,)
        ``ObservationSupport.sample_valid``.
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        ``ObservationSupport.interval_mask`` of the same samples.

    Returns
    -------
    start, end : ndarray of intp, shape (n_samples - 1,)
        Samples the step of each interval runs between (``gap_endpoints``).
    counted : ndarray of bool, shape (n_samples - 1,)
        True at one interval per usable step.
    """
    from neurospatial._intervals import gap_endpoints

    start, end, _ = gap_endpoints(sample_valid)
    counted: NDArray[np.bool_] = interval_mask & sample_valid[1:]
    return start, end, counted


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


def validated_motion(
    values: Any, *, name: str, n_samples: int, call: str
) -> NDArray[np.float64] | None:
    """A caller's per-sample ``speed`` or ``headings``, as float64, or None.

    Parameters
    ----------
    values : array-like, shape (n_samples,), or None
        Speed in position units per second (finite and >= 0, NaN where not
        observed), or headings in radians (finite, NaN where not observed).
    name : {"speed", "headings"}
    n_samples : int
    call : str
        Public function name, used in the error.

    Raises
    ------
    ValueError
        If ``values`` is not one real number per sample, or holds an
        infinite value, or (for speed) a negative one.
    """
    from neurospatial._exceptions import _format_error

    if values is None:
        return None
    source = (
        "compute_speed(times, positions)"
        if name == "speed"
        else "heading_from_velocity(times, positions)"
    )
    array = np.asarray(values)
    if array.shape != (n_samples,) or not np.issubdtype(array.dtype, np.number):
        raise ValueError(
            _format_error(
                f"{call}() needs {name} to be one number per sample, shape "
                f"({n_samples},); got dtype {array.dtype} and shape {array.shape}.",
                why=f"{name} is read sample by sample alongside times and positions.",
                fix=f"pass one value per sample, e.g. {source}.",
            )
        )
    array = array.astype(np.float64)
    bad = np.isinf(array) | (array < 0 if name == "speed" else False)
    if np.any(bad):
        rule = "finite and >= 0" if name == "speed" else "finite"
        raise ValueError(
            _format_error(
                f"{call}() needs every {name} value to be {rule} or NaN; sample "
                f"{int(np.flatnonzero(bad)[0])} is {array[bad][0]!r}.",
                why="NaN marks a sample where it was not observed; any other "
                "value is used as measured.",
                fix=f"set unobserved samples to NaN, e.g. as {source} does.",
            )
        )
    return array


def _movement_heading_and_speed(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
    *,
    smooth_window: float,
    min_speed: float,
    speed: NDArray[np.float64] | None = None,
    headings: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Direction of movement and the speed that gates it.

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
    speed, headings : ndarray, shape (n_samples,), optional
        The caller's speed and direction of movement (``validated_motion``),
        used instead of the horizontal speed and direction of the smoothed
        velocity. Either is NaN outside the samples the smoothed velocity
        covers, so ``max_gap``, ``epochs`` and breaks apply to it alike.

    Returns
    -------
    heading : ndarray, shape (n_samples,)
        Direction in radians where the animal moves (speed above 0 and at
        least ``min_speed``); NaN where it does not, since a still animal has
        no direction of movement (``arctan2(0, 0)`` would read as East), and
        at samples in no run.
    speed : ndarray, shape (n_samples,)
        Speed in position units per second; NaN at samples in no run.
    """
    from neurospatial._velocity import smoothed_velocity

    if speed is None or headings is None:
        velocity = smoothed_velocity(
            times, positions[:, :2], interval_mask, smooth_window=smooth_window
        )
        in_run = np.isfinite(velocity[:, 0])
    else:
        in_run = observed_run_samples(positions[:, :2], interval_mask)
    if speed is None:
        speed = np.hypot(velocity[:, 0], velocity[:, 1])
    else:
        speed = np.where(in_run, speed, np.nan)
    if headings is None:
        direction = np.arctan2(velocity[:, 1], velocity[:, 0])
    else:
        direction = np.where(in_run, headings, np.nan)
    moving = (speed > 0) & (speed >= min_speed)
    return np.where(moving, direction, np.nan), speed


def run_speed(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
    *,
    speed: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """``compute_speed``'s speed under ``interval_mask``, or the caller's.

    The caller's ``speed`` is NaN outside the samples the smoothed speed
    covers, so ``max_gap``, ``epochs`` and breaks apply to it alike.
    """
    from neurospatial._velocity import DEFAULT_SPEED_SMOOTHING, smoothed_speed

    if speed is None:
        return smoothed_speed(
            times, positions, interval_mask, smooth_window=DEFAULT_SPEED_SMOOTHING
        )
    return np.where(observed_run_samples(positions, interval_mask), speed, np.nan)


def observed_run_samples(
    positions: NDArray[np.float64], interval_mask: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    """True at observed samples in a run of at least two, shape (n_samples,).

    These are the samples where ``smoothed_velocity`` is defined.
    """
    from neurospatial._intervals import valid_run_samples

    finite = np.all(np.isfinite(positions.reshape(len(positions), -1)), axis=1)
    covered = np.zeros(finite.size, dtype=bool)
    for run in valid_run_samples(np.asarray(interval_mask, dtype=bool), finite):
        if run.size >= 2:
            covered[run] = True
    return covered
