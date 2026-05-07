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
    "validate_times",
    "validate_trajectory",
]


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
            raise ValueError(f"positions must be 1D or 2D, got shape {positions.shape}")
        if len(positions) != n_samples:
            raise ValueError(
                f"times length ({n_samples}) must match positions length "
                f"({len(positions)})"
            )

    if headings is not None:
        if headings.ndim != 1:
            raise ValueError(f"headings must be 1D, got shape {headings.shape}")
        if len(headings) != n_samples:
            raise ValueError(
                f"times length ({n_samples}) must match headings length "
                f"({len(headings)})"
            )
