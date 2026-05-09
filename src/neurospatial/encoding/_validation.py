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

__all__ = [
    "validate_env_fitted",
    "validate_spike_times",
    "validate_times",
    "validate_trajectory",
]


def validate_env_fitted(env: object, *, context: str) -> None:
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

    Raises
    ------
    EnvironmentNotFittedError
        If ``env`` does not have ``_is_fitted=True``.
    """
    # Local import to keep this module dependency-free at module load time;
    # the encoding package imports decorators lazily for the same reason.
    from neurospatial.environment.decorators import EnvironmentNotFittedError

    if not getattr(env, "_is_fitted", False):
        raise EnvironmentNotFittedError(context, is_function=True)


def validate_times(times: NDArray[np.float64], context: str = "encoding") -> None:
    """Check that ``times`` has at least 2 samples, is finite, and monotonic.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Timestamps to validate.
    context : str, default "encoding"
        Description of the calling function for error messages.

    Raises
    ------
    ValueError
        If ``times`` has fewer than 2 samples, contains NaN or +/-inf, or
        if any pair of adjacent samples is decreasing (``times`` must be
        sorted; equal-valued adjacent samples are allowed).
    """
    n_samples = len(times)
    if n_samples < 2:
        raise ValueError(f"At least 2 samples required for {context}, got {n_samples}")

    if not np.all(np.isfinite(times)):
        # NaN comparisons are False, so the monotonic check below would
        # silently accept NaN-laced timestamps. Reject explicitly here.
        n_bad = int(np.sum(~np.isfinite(times)))
        raise ValueError(
            f"times must be finite for {context}; got {n_bad} NaN/inf entries"
        )

    time_diffs = np.diff(times)
    if np.any(time_diffs < 0):
        decreasing_indices = np.where(time_diffs < 0)[0]
        raise ValueError(
            "times must be monotonically non-decreasing (sorted). "
            f"Found {len(decreasing_indices)} decreasing interval(s) at "
            f"indices: {decreasing_indices.tolist()[:5]}"
            + (" ..." if len(decreasing_indices) > 5 else "")
        )


def validate_spike_times(
    spike_times: NDArray[np.float64],
    *,
    context: str = "encoding",
    allow_empty: bool = True,
) -> None:
    """Check that ``spike_times`` is 1-D, finite, sorted, and non-negative.

    Internal helpers downstream (``bin_spike_train`` and friends) use
    ``np.searchsorted`` against the spike-time array, so an out-of-order
    spike train silently produces wrong bin assignments. The four public
    ``compute_*_rate(s)`` entry points should call this once on user input.

    Parameters
    ----------
    spike_times : ndarray, shape (n_spikes,)
        Spike timestamps in seconds. Empty arrays are allowed by default
        (a neuron with zero spikes is a valid input).
    context : str, default "encoding"
        Description of the calling function for error messages.
    allow_empty : bool, default True
        If False, also reject zero-length spike trains. Use this when the
        caller cannot meaningfully proceed without at least one spike.

    Raises
    ------
    ValueError
        If ``spike_times`` is not 1-D, contains NaN or +/-inf, contains a
        negative value, has any pair of adjacent samples in decreasing
        order, or (with ``allow_empty=False``) is empty.
    """
    if spike_times.ndim != 1:
        raise ValueError(
            f"spike_times must be 1-D for {context}, got shape {spike_times.shape}"
        )

    n_spikes = len(spike_times)
    if n_spikes == 0:
        if not allow_empty:
            raise ValueError(f"spike_times is empty (no spikes) for {context}")
        return

    if not np.all(np.isfinite(spike_times)):
        n_bad = int(np.sum(~np.isfinite(spike_times)))
        raise ValueError(
            f"spike_times must be finite (seconds) for {context}; "
            f"got {n_bad} NaN/inf entries"
        )

    if np.any(spike_times < 0.0):
        n_negative = int(np.sum(spike_times < 0.0))
        raise ValueError(
            f"spike_times must be non-negative (seconds) for {context}; "
            f"got {n_negative} negative entr{'y' if n_negative == 1 else 'ies'} "
            f"(min: {float(spike_times.min()):.6g} s)"
        )

    diffs = np.diff(spike_times)
    if np.any(diffs < 0):
        decreasing = np.where(diffs < 0)[0]
        sample = decreasing.tolist()[:5]
        more = " ..." if decreasing.size > 5 else ""
        raise ValueError(
            "spike_times must be monotonically non-decreasing (sorted in "
            f"ascending order) for {context}. Found {decreasing.size} "
            f"decreasing interval(s) at indices: {sample}{more}. "
            "If your spikes were merged from multiple sources, sort the "
            "array with `np.sort(spike_times)` before passing it in."
        )


def validate_trajectory(
    times: NDArray[np.float64],
    positions: NDArray[np.float64] | None = None,
    headings: NDArray[np.float64] | None = None,
    *,
    context: str = "encoding",
) -> None:
    """Check that trajectory arrays are 1D-aligned and ``times`` is sane.

    Combines the ndim/length cross-check on ``(times, positions?, headings?)``
    with the timestamp-shape check from :func:`validate_times` (min length 2,
    finite, monotonically non-decreasing). Public ``compute_*`` entry points
    should call this once on their trajectory inputs.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Timestamps. Must be 1D and pass :func:`validate_times`.
    positions : ndarray, shape (n_samples,) or (n_samples, n_dims), optional
        Position coordinates. Must be 1D (linearized) or 2D with first
        axis matching ``times``.
    headings : ndarray, shape (n_samples,), optional
        Head direction values. Must be 1D with length matching ``times``.
    context : str, default "encoding"
        Description of the calling function for error messages.

    Raises
    ------
    ValueError
        If ``times`` is not 1D, fails :func:`validate_times`, if
        ``headings`` is not 1D, if ``positions`` is not 1D or 2D, or if
        any provided array's first axis disagrees with ``len(times)``.
    """
    if times.ndim != 1:
        raise ValueError(f"times must be 1D, got shape {times.shape}")

    validate_times(times, context=context)

    n_samples = len(times)

    if positions is not None:
        if positions.ndim not in (1, 2):
            raise ValueError(
                f"in {context}: positions must be 1D or 2D, got shape {positions.shape}"
            )
        if len(positions) != n_samples:
            raise ValueError(
                f"in {context}: times length ({n_samples}) must match "
                f"positions length ({len(positions)})"
            )

    if headings is not None:
        if headings.ndim != 1:
            raise ValueError(
                f"in {context}: headings must be 1D, got shape {headings.shape}"
            )
        if len(headings) != n_samples:
            raise ValueError(
                f"in {context}: times length ({n_samples}) must match "
                f"headings length ({len(headings)})"
            )
