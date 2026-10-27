"""Shared input validation helpers for encoding compute functions.

The public ``compute_*_rate(s)`` entry points and cell predicates share:

- ``_validate_times`` — finite, non-decreasing timestamps with at least two
  samples.
- ``validate_trajectory`` — joint length and dimensionality checks on
  ``(times, positions?, headings?)``, plus a warning for headings that look
  like degrees.
- ``warn_if_angles_look_like_degrees`` — the degree check for any angle
  input read as radians.
- ``validate_spike_times`` / ``validate_spike_trains`` — spike timestamps.

The smoothing-method validator lives in
``neurospatial.encoding._smoothing._validate_smoothing_parameters`` and
is intentionally not duplicated here; entry points should import it
directly.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._stacklevel import _external_stacklevel
from neurospatial._validation import (
    format_times_positions_error,
    times_positions_problems,
    validate_finite,
)
from neurospatial.environment import Environment
from neurospatial.geometry import wrong_position_columns

__all__ = [
    "_validate_env_argument",
    "_validate_times",
    "validate_classifier_trajectory",
    "validate_spike_times",
    "validate_spike_trains",
    "validate_trajectory",
    "warn_if_angles_look_like_degrees",
]


def _validate_env_argument(env: object, *, context: str, arguments: str) -> None:
    """Raise ``TypeError`` if ``env`` is not an environment.

    Public ``compute_*_rate(s)`` and ``decode_position`` entry points use
    this to fail at the API boundary rather than letting a non-environment
    surface as a confusing ``AttributeError`` from a deep helper. The
    ``context`` label is the user-facing free-function name (e.g.
    ``"compute_spatial_rate"``) and is forwarded to the free-function
    error message.

    Parameters
    ----------
    env : object
        Object expected to be an ``Environment``.
    context : str
        Name of the calling public free function, used in the message.
    arguments : str
        Required argument names after ``env`` in the corrected public call.

    Raises
    ------
    TypeError
        If ``env`` is not an environment. The message shows how to build and
        pass one.
    """
    if not isinstance(env, Environment):
        description = type(env).__name__
        shape = getattr(env, "shape", None)
        if shape is not None:
            description += f" with shape {shape}"
        raise TypeError(
            _format_error(
                f"{context}() expects an Environment as its first argument, got {description}.",
                why="Why: spatial analysis needs the environment's geometry and bins.",
                fix=f"build one with env = Environment.from_samples(positions, bin_size=2.0), then call {context}(env, {arguments}).",
            )
        )


def _raise_input_problems(context: str, problems: list[tuple[str, str]]) -> None:
    """Report all input problems with their combined concrete fixes."""
    if not problems:
        return
    details = (
        problems[0][0]
        if len(problems) == 1
        else "invalid inputs:\n" + "\n".join(f"- {what}" for what, _ in problems)
    )
    raise ValueError(
        _format_error(
            f"{context}: {details}",
            why="Why: timestamps in seconds and sample-aligned coordinates are needed to assign observations to the correct bins.",
            fix="; ".join(dict.fromkeys(fix for _, fix in problems)),
        )
    )


def _time_problems(times: NDArray[np.float64]) -> list[tuple[str, str]]:
    """Collect timestamp-shape, finite-value and ordering problems."""
    problems = []
    if times.ndim != 1:
        return [
            (
                f"times must be 1D, got shape {times.shape}",
                "pass times as a 1-D array of timestamps in seconds",
            )
        ]
    if len(times) < 2:
        problems.append(
            (
                f"At least 2 samples required, got {len(times)}",
                "pass at least two timestamped position samples",
            )
        )
    if not np.all(np.isfinite(times)):
        n_bad = int(np.sum(~np.isfinite(times)))
        problems.append(
            (
                f"times must be finite; got {n_bad} NaN/inf entries",
                "remove non-finite timestamps and the corresponding sample rows",
            )
        )
    decreasing = np.flatnonzero(np.diff(times) < 0)
    if decreasing.size:
        problems.append(
            (
                f"times must be monotonically non-decreasing (sorted); found {decreasing.size} decreasing interval(s) at indices {decreasing[:5].tolist()}",
                "sort times and all corresponding sample rows with order = np.argsort(times)",
            )
        )
    return problems


def _validate_times(times: NDArray[np.float64], *, context: str) -> None:
    """Check finite, non-decreasing timestamps with at least two samples.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Timestamps in seconds; adjacent equal timestamps are allowed.
    context : str
        Name of the calling function, used in the error message.

    Raises
    ------
    ValueError
        If the shape, sample count, finite values or ordering is invalid.
    """
    _raise_input_problems(context, _time_problems(times))


def validate_spike_times(
    spike_times: NDArray[np.float64], *, context: str, allow_empty: bool = True
) -> None:
    """Check one-dimensional finite, sorted, non-negative spike timestamps.

    Parameters
    ----------
    spike_times : ndarray, shape (n_spikes,)
        Spike timestamps in seconds. Empty trains are allowed by default.
    context : str
        Name of the calling function, used in the error message.
    allow_empty : bool, default True
        Whether a neuron with no spikes is a valid input.

    Raises
    ------
    ValueError
        If the shape, finite values, sign, ordering or emptiness is invalid.
    """
    problems = []
    if spike_times.ndim != 1:
        problems.append(
            (
                f"spike_times must be 1-D, got shape {spike_times.shape}",
                "pass one 1-D spike_times array per unit",
            )
        )
    else:
        if not len(spike_times) and not allow_empty:
            problems.append(
                (
                    "spike_times is empty (no spikes)",
                    "pass at least one spike timestamp in seconds",
                )
            )
        if not np.all(np.isfinite(spike_times)):
            n_bad = int(np.sum(~np.isfinite(spike_times)))
            problems.append(
                (
                    f"spike_times must be finite (seconds); got {n_bad} NaN/inf entries",
                    "remove non-finite spike_times entries",
                )
            )
        if np.any(spike_times < 0):
            n_negative = int(np.sum(spike_times < 0))
            problems.append(
                (
                    f"spike_times must be non-negative (seconds); got {n_negative} negative entries (min: {float(np.nanmin(spike_times)):.6g} s)",
                    "align spike_times to the recording's non-negative time origin",
                )
            )
        decreasing = np.flatnonzero(np.diff(spike_times) < 0)
        if decreasing.size:
            problems.append(
                (
                    f"spike_times must be monotonically non-decreasing (sorted in ascending order); found {decreasing.size} decreasing interval(s) at indices {decreasing[:5].tolist()}",
                    "if spikes were merged from multiple sources, sort the array with np.sort(spike_times)",
                )
            )
    _raise_input_problems(context, problems)


def validate_spike_trains(
    spike_trains: Sequence[NDArray[np.float64]], *, context: str
) -> None:
    """Validate each train, naming the failing unit as ``context (neuron i)``.

    Parameters
    ----------
    spike_trains : sequence of ndarray, each shape (n_spikes,)
        One spike-time array per unit, in seconds.
    context : str
        Name of the calling function, used in the error message.

    Raises
    ------
    ValueError
        From :func:`validate_spike_times` for the first invalid train.
    """
    for i, spike_times in enumerate(spike_trains):
        validate_spike_times(spike_times, context=f"{context} (neuron {i})")


def warn_if_angles_look_like_degrees(
    angles: NDArray[np.float64],
    *,
    name: str,
    context: str,
    angle_unit_keyword: bool = False,
) -> None:
    """Warn when angles read as radians look like degrees.

    Radians beyond ±2π are legitimate (unwrapped angles), so this warns
    instead of raising. It fires when the largest finite magnitude exceeds
    2π, every finite value lies within ±360, and some pair of consecutive
    finite values differs by more than π.

    The last condition separates degrees from unwrapped radians. Unwrapped
    radians are continuous: :func:`numpy.unwrap` never leaves a step larger
    than π, and a sampled heading that turned more than half a circle
    between samples could not be unwrapped at all. Degree headings step by
    more than π (3.14°) whenever the head turns faster than about 94° per
    second at 30 Hz, and wrapped degrees (in [0, 360) or (-180, 180]) jump
    by nearly 360 at every crossing of the seam. Degree data in which every
    step is below 3.14° and the seam is never crossed is not detected.

    Parameters
    ----------
    angles : ndarray
        Angles that will be read as radians, in sample (or event) order.
        Non-finite values are ignored.
    name : str
        Argument name shown in the message, e.g. ``"headings"``.
    context : str
        Name of the calling public function.
    angle_unit_keyword : bool, default False
        Whether the caller accepts ``angle_unit="deg"``, so the message can
        offer it.

    Warns
    -----
    UserWarning
        When the finite angles look like degrees.
    """
    angles = np.asarray(angles, dtype=np.float64).ravel()
    finite = np.isfinite(angles)
    if not finite.any():
        return
    largest = float(np.abs(angles[finite]).max())
    if not (2 * np.pi + 1e-6 < largest <= 360.0):
        return
    both_finite = finite[1:] & finite[:-1]
    steps = np.abs(np.diff(angles))[both_finite]
    if not np.any(steps > np.pi):
        return
    largest_step = float(steps.max())
    fix = f"convert with np.radians({name})"
    if angle_unit_keyword:
        fix += ' or pass angle_unit="deg"'
    warnings.warn(
        f"{context}: {name} look like degrees (largest magnitude {largest:.4g}, "
        f"all within ±360, and consecutive values differ by up to "
        f"{largest_step:.4g}, more than the π a continuous radian angle can "
        f"step) but are read as radians, which wraps them into meaningless "
        f"directions. {fix[0].upper()}{fix[1:]}.",
        UserWarning,
        stacklevel=_external_stacklevel(),
    )


def validate_trajectory(
    times: NDArray[np.float64],
    positions: NDArray[np.float64] | None = None,
    headings: NDArray[np.float64] | None = None,
    *,
    context: str,
    n_dims: int | None = None,
    heading_unit: Literal["rad", "deg"] | None = None,
) -> None:
    """Check trajectory shapes, aligned sample rows and timestamp validity.

    All detected problems are reported together. Non-finite positions and
    headings remain valid missing observations for the binning layer to drop.
    Headings read as radians that look like degrees warn
    (:func:`warn_if_angles_look_like_degrees`).

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Finite non-decreasing timestamps in seconds, with at least two samples.
    positions : ndarray, shape (n_samples,) or (n_samples, n_dims), optional
        Coordinates aligned with times. One-dimensional positions are accepted
        for a one-dimensional environment.
    headings : ndarray, shape (n_samples,), optional
        Head directions aligned with times.
    context : str
        Name of the calling function, used in the error message.
    n_dims : int, optional
        Expected coordinate dimension of the environment.
    heading_unit : {"rad", "deg"} or None, optional
        The caller's ``angle_unit``, or None when the caller only accepts
        radians. Headings are checked for degrees unless it is ``"deg"``.

    Raises
    ------
    ValueError
        If timestamp validity, shapes, coordinate dimensions or lengths disagree.

    Warns
    -----
    UserWarning
        When headings read as radians look like degrees.
    """
    looks_swapped = False
    if positions is None:
        problems = [what for what, _ in _time_problems(times)]
    else:
        problems, looks_swapped = times_positions_problems(times, positions)
        if times.ndim == 1 and len(times) < 2:
            problems.append(f"At least 2 samples required, got {len(times)}.")
        if n_dims is not None and (
            (positions.ndim == 1 and n_dims > 1)
            or (positions.ndim == 2 and positions.shape[1] != n_dims)
        ):
            problems.append(f"Got {wrong_position_columns(positions.shape, n_dims)}.")
    if headings is not None:
        if headings.ndim != 1:
            problems.append(f"headings must be 1D, got shape {headings.shape}.")
        if times.ndim >= 1 and headings.ndim >= 1 and len(headings) != len(times):
            problems.append(
                f"times length ({len(times)}) must match headings length ({len(headings)})."
            )
    if problems:
        raise ValueError(
            format_times_positions_error(problems, looks_swapped, call=context)
        )
    if headings is not None and heading_unit != "deg":
        warn_if_angles_look_like_degrees(
            headings,
            name="headings",
            context=context,
            angle_unit_keyword=heading_unit is not None,
        )


def validate_classifier_trajectory(
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    context: str,
) -> None:
    """Validate ``(spike_times, times, headings)`` for directional classifiers.

    Raises on genuine input errors so they propagate instead of being
    swallowed as a False classification. Does NOT enforce statistical
    significance — that is decided after the (valid) computation.

    ``headings`` are intentionally not passed through :func:`validate_finite`:
    non-finite headings are a legitimate, droppable condition handled by the
    directional binning layer (masked out of occupancy and spike counts), not
    a hard error. The guard is for shape/length/timestamp sanity that should
    surface as errors.

    Parameters
    ----------
    spike_times : ndarray, shape (n_spikes,)
        Spike timestamps in seconds.
    times : ndarray, shape (n_samples,)
        Timestamps of head direction samples in seconds.
    headings : ndarray, shape (n_samples,)
        Head direction at each time point.
    context : str
        Name of the calling function for error messages.

    Raises
    ------
    ValueError
        If lengths disagree, ``times`` is non-finite or not monotonically
        non-decreasing, or ``spike_times`` is malformed.
    """
    times = np.asarray(times, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)
    spike_times = np.asarray(spike_times, dtype=np.float64)
    validate_trajectory(times, headings=headings, context=context)
    validate_finite(times, name="times")
    validate_spike_times(spike_times, context=context)
