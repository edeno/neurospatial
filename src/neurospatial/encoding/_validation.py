"""Shared input validation helpers for encoding compute functions.

The four public ``compute_*_rate(s)`` entry points
(``compute_spatial_rate``, ``compute_directional_rate``,
``compute_egocentric_rate``, ``compute_view_rate`` and their plural
variants) had drifted on input validation: only ``compute_view_rate`` and
``compute_egocentric_rate`` length-checked their trajectory inputs, and
each surfaced length errors with a slightly different message style.

This module centralizes:

- ``validate_times`` — monotonic-non-decreasing + min-length check on a
  timestamp array (previously duplicated byte-for-byte in
  ``_view_binning.py`` and ``_egocentric_binning.py``).
- ``validate_trajectory`` — joint length + dimensionality check on
  ``(times, positions?, headings?)``.

The smoothing-method validator lives in
``neurospatial.encoding._smoothing._validate_smoothing_parameters`` and
is intentionally not duplicated here; entry points should import it
directly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import EnvironmentNotFittedError, _format_error
from neurospatial._validation import validate_finite

__all__ = [
    "validate_classifier_trajectory",
    "validate_env_fitted",
    "validate_spike_times",
    "validate_times",
    "validate_trajectory",
]


def validate_env_fitted(env: object, *, context: str, arguments: str) -> None:
    """Raise ``EnvironmentNotFittedError`` if ``env`` is not fitted.

    Public ``compute_*_rate(s)`` and ``decode_position`` entry points use
    this to fail at the API boundary rather than letting an unfitted env
    surface as a confusing ``AttributeError`` from a deep helper. The
    ``context`` label is the user-facing free-function name (e.g.
    ``"compute_spatial_rate"``) and is forwarded to the free-function
    form of :class:`EnvironmentNotFittedError` so the rendered message
    reads ``compute_spatial_rate()`` rather than the misleading
    ``Environment.compute_spatial_rate()``.

    Parameters
    ----------
    env : object
        Object expected to expose ``_is_fitted`` (typically an
        ``Environment``). Untyped here because import-time importing the
        ``Environment`` symbol would re-introduce the import cycle that
        ``encoding._validation`` exists to avoid.
    context : str
        Name of the calling public free function, used as the
        ``EnvironmentNotFittedError`` function-name argument.
    arguments : str
        Required argument names after ``env`` in the corrected public call.

    Raises
    ------
    TypeError
        If ``env`` has no fitted-state attribute and cannot serve as an
        environment. The message shows how to build and pass one.
    EnvironmentNotFittedError
        If ``env`` exposes ``_is_fitted`` but is not fitted.
    """
    if not hasattr(env, "_is_fitted"):
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
    if not getattr(env, "_is_fitted", False):
        raise EnvironmentNotFittedError(context, is_function=True)


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


def validate_times(times: NDArray[np.float64], *, context: str) -> None:
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


def validate_trajectory(
    times: NDArray[np.float64],
    positions: NDArray[np.float64] | None = None,
    headings: NDArray[np.float64] | None = None,
    *,
    context: str,
    n_dims: int | None = None,
) -> None:
    """Check trajectory shapes, aligned sample rows and timestamp validity.

    All detected problems are reported together. Non-finite positions and
    headings remain valid missing observations for the binning layer to drop.

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

    Raises
    ------
    ValueError
        If timestamp validity, shapes, coordinate dimensions or lengths disagree.
    """
    problems = _time_problems(times)
    if positions is not None:
        if times.ndim == 2 and positions.ndim == 1:
            problems.append(
                (
                    f"times has shape {times.shape} and positions has shape {positions.shape}; did you pass positions before times?",
                    f"call {context}(env, spike_times, times, positions) with times before positions",
                )
            )
        if positions.ndim not in (1, 2):
            problems.append(
                (
                    f"positions must be 1D or 2D, got shape {positions.shape}",
                    "pass positions with shape (n_samples, n_dims)",
                )
            )
        if times.ndim >= 1 and positions.ndim >= 1 and len(positions) != len(times):
            problems.append(
                (
                    f"times length ({len(times)}) must match positions length ({len(positions)})",
                    "align times and positions so each timestamp has one position row",
                )
            )
        if n_dims is not None and (
            (positions.ndim == 1 and n_dims > 1)
            or (positions.ndim == 2 and positions.shape[1] != n_dims)
        ):
            example = (
                "np.column_stack([x, y])"
                if n_dims == 2
                else f"an array with {n_dims} coordinate columns"
            )
            problems.append(
                (
                    f"positions has shape {positions.shape} but env is {n_dims}-D, so positions must have shape (n_samples, {n_dims})",
                    f"pass all coordinates, e.g. {example}; for a 1-D track, build env from 1-D data (positions[:, None]) or Environment.linear_track(...)",
                )
            )
    if headings is not None:
        if headings.ndim != 1:
            problems.append(
                (
                    f"headings must be 1D, got shape {headings.shape}",
                    "pass headings as a 1-D array with one angle per timestamp",
                )
            )
        if times.ndim >= 1 and headings.ndim >= 1 and len(headings) != len(times):
            problems.append(
                (
                    f"times length ({len(times)}) must match headings length ({len(headings)})",
                    "align times and headings so each timestamp has one angle",
                )
            )
    _raise_input_problems(context, problems)


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
