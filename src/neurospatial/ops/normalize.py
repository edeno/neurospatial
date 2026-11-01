"""
Field normalization and combination operations.

This module provides common operations on bin-valued fields (probability
distributions, rate maps, occupancy, etc.). All functions operate on 1-D
arrays of shape (n_bins,).

Examples
--------
>>> from neurospatial.ops.normalize import normalize_field, clamp, combine_fields
>>> import numpy as np
>>> field = np.array([1.0, 2.0, 3.0])
>>> normalized = normalize_field(field)
>>> float(normalized.sum())
1.0
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._validation import validate_choice
from neurospatial.ops.diffusion import _HEAT_KERNEL_RANK_TOL

#: Negative entries no larger than this fraction of a field's largest value
#: are numerical noise, not negative weight. ``env.smooth`` keeps heat-kernel
#: modes down to a weight of ``_HEAT_KERNEL_RANK_TOL`` (1e-6), so smoothing a
#: non-negative field rings below zero by up to about that fraction of the
#: field's maximum (6.7e-7 measured on lines, boxes, discs and meshes over
#: every mode and bandwidth 2-30 bins); ten times it leaves a margin.
_SMOOTHING_NOISE_RTOL = 10 * _HEAT_KERNEL_RANK_TOL

__all__ = [
    "clamp",
    "combine_fields",
    "normalize_field",
]


def normalize_field(field: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Normalize field to sum to 1 (probability distribution).

    Parameters
    ----------
    field : NDArray[np.float64], shape (n_bins,)
        One field to normalize. Must be 1-D, finite and non-negative.
        Negative entries no larger than 1e-5 of the largest value, the
        numerical noise of :meth:`Environment.smooth` on a non-negative
        field, count as 0.

    Returns
    -------
    normalized : NDArray[np.float64], shape (n_bins,)
        Normalized field where values sum to 1.0.

    Raises
    ------
    ValueError
        If field is not 1-D (normalize each row of a ``(n_time, n_bins)``
        array with ``field / field.sum(axis=1, keepdims=True)``), or contains
        negative values, NaN or Inf, or is all zeros.

    Examples
    --------
    >>> field = np.array([1.0, 2.0, 3.0])
    >>> normalized = normalize_field(field)
    >>> print(normalized.sum())
    1.0
    >>> np.allclose(normalized, [1 / 6, 1 / 3, 0.5])
    True

    Notes
    -----
    The normalization preserves relative proportions:
    normalized[i] / normalized[j] = field[i] / field[j]
    """
    field = np.asarray(field, dtype=np.float64)

    if field.ndim != 1:
        raise ValueError(
            _format_error(
                f"normalize_field() takes one 1-D field of shape (n_bins,), got "
                f"shape {field.shape}.",
                why="It makes the whole array sum to 1; a (n_time, n_bins) "
                "array holds one distribution per row, and its rows would not "
                "each sum to 1.",
                fix="normalize each row with field / field.sum(axis=1, "
                "keepdims=True), or pass one row at a time",
            )
        )

    # Validate input
    if not np.isfinite(field).all():
        if np.isnan(field).any():
            n_nan = np.isnan(field).sum()
            msg = (
                f"Field contains NaN values ({n_nan} NaN entries). "
                "Remove or impute NaN values before normalizing."
            )
            raise ValueError(msg)
        if np.isinf(field).any():
            n_inf = np.isinf(field).sum()
            msg = (
                f"Field contains Inf values ({n_inf} Inf entries). "
                "Replace Inf values before normalizing."
            )
            raise ValueError(msg)

    field = _zero_smoothing_noise(field)
    if (field < 0).any():
        n_negative = (field < 0).sum()
        msg = (
            f"Field contains negative values ({n_negative} negative entries). "
            "Normalization requires non-negative values."
        )
        raise ValueError(msg)

    total = field.sum()
    if total == 0.0:
        msg = (
            f"Cannot normalize field: all zeros (sum={total:.2e}). "
            "Provide a field with positive values."
        )
        raise ValueError(msg)

    normalized: NDArray[np.float64] = field / total
    return normalized


def _zero_smoothing_noise(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """``values`` with negative smoothing noise set to 0, along the last axis.

    A negative entry no larger than ``_SMOOTHING_NOISE_RTOL`` times the largest
    value of its row is the numerical noise of smoothing a non-negative field
    (:meth:`Environment.smooth`), and becomes 0. Larger negative entries are
    kept, for the caller to refuse.

    Parameters
    ----------
    values : NDArray[np.float64], shape (..., n_bins)
        Finite weights.

    Returns
    -------
    NDArray[np.float64], shape (..., n_bins)
        ``values`` itself when nothing changes, otherwise a copy.
    """
    negative = values < 0
    if not negative.any():
        return values
    floor = -_SMOOTHING_NOISE_RTOL * values.max(axis=-1, keepdims=True)
    noise = negative & (values >= floor)
    return np.where(noise, 0.0, values)


def clamp(
    field: NDArray[np.float64],
    *,
    lo: float = 0.0,
    hi: float = np.inf,
) -> NDArray[np.float64]:
    """
    Clamp field values to [lo, hi] range.

    Parameters
    ----------
    field : NDArray[np.float64], shape (n_bins,)
        Field values to clamp.
    lo : float, default=0.0
        Lower bound. Values below this are set to lo.
    hi : float, default=np.inf
        Upper bound. Values above this are set to hi.

    Returns
    -------
    clamped : NDArray[np.float64], shape (n_bins,)
        Field with values clamped to [lo, hi].

    Raises
    ------
    ValueError
        If lo > hi.

    Examples
    --------
    >>> field = np.array([-1.0, 0.5, 2.0])
    >>> clamped = clamp(field, lo=0.0, hi=1.0)
    >>> print(clamped)
    [0.  0.5 1. ]

    Notes
    -----
    NaN values are preserved (not clamped). This follows NumPy convention
    where NaN propagates through operations.
    """
    field = np.asarray(field, dtype=np.float64)

    if lo > hi:
        msg = (
            f"lo ({lo}) must be less than or equal to hi ({hi}). Provide valid bounds."
        )
        raise ValueError(msg)

    return np.clip(field, lo, hi)


def combine_fields(
    fields: Sequence[NDArray[np.float64]],
    *,
    weights: Sequence[float] | None = None,
    mode: Literal["mean", "max", "min"] = "mean",
) -> NDArray[np.float64]:
    """
    Combine multiple fields using specified aggregation mode.

    Parameters
    ----------
    fields : Sequence[NDArray[np.float64]], each shape (n_bins,)
        Fields to combine. All must have the same shape.
    weights : Sequence[float], optional
        Weights for mode='mean'. Must have same length as fields and sum to 1.
        If None, uniform weights are used.
    mode : {'mean', 'max', 'min'}, default='mean'
        Aggregation mode:

        - 'mean': Weighted average (requires weights or uses uniform).
        - 'max': Element-wise maximum across fields.
        - 'min': Element-wise minimum across fields.

    Returns
    -------
    combined : NDArray[np.float64], shape (n_bins,)
        Combined field.

    Raises
    ------
    ValueError
        If fields is empty, fields have mismatched shapes, weights have wrong
        length, weights don't sum to 1, weights provided with max/min mode,
        or mode is not 'mean', 'max' or 'min'.

    Examples
    --------
    >>> f1 = np.array([1.0, 2.0, 3.0])
    >>> f2 = np.array([3.0, 4.0, 5.0])
    >>> combined = combine_fields([f1, f2], mode="mean")
    >>> print(combined)
    [2. 3. 4.]

    >>> # Weighted mean
    >>> combined = combine_fields([f1, f2], weights=[0.25, 0.75], mode="mean")
    >>> print(combined)
    [2.5 3.5 4.5]

    >>> # Element-wise maximum
    >>> f1 = np.array([1.0, 5.0, 3.0])
    >>> f2 = np.array([3.0, 2.0, 4.0])
    >>> combined = combine_fields([f1, f2], mode="max")
    >>> print(combined)
    [3. 5. 4.]
    """
    if len(fields) == 0:
        msg = "At least one field required for combining."
        raise ValueError(msg)

    # Convert to list of arrays
    fields_list = [np.asarray(f, dtype=np.float64) for f in fields]

    # Validate shapes
    shape = fields_list[0].shape
    for i, f in enumerate(fields_list[1:], start=1):
        if f.shape != shape:
            msg = (
                f"All fields must have the same shape. "
                f"Field 0 has shape {shape}, but field {i} has shape {f.shape}."
            )
            raise ValueError(msg)

    mode = validate_choice(
        mode, ("mean", "max", "min"), name="mode", owner="combine_fields()"
    )

    # Handle weights parameter
    if weights is not None:
        if mode != "mean":
            msg = (
                f"Weights only valid for mode='mean', got mode='{mode}'. "
                "Remove weights parameter or use mode='mean'."
            )
            raise ValueError(msg)

        if len(weights) != len(fields_list):
            msg = (
                f"Number of weights ({len(weights)}) must match "
                f"number of fields ({len(fields_list)})."
            )
            raise ValueError(msg)

        weights_array = np.asarray(weights, dtype=np.float64)
        if not np.isclose(weights_array.sum(), 1.0, atol=1e-6):
            msg = (
                f"Weights must sum to 1, got sum={weights_array.sum():.6f}. "
                "Normalize weights before passing."
            )
            raise ValueError(msg)
    else:
        # Uniform weights for mean mode
        if mode == "mean":
            weights_array = np.ones(len(fields_list)) / len(fields_list)

    # Combine fields
    match mode:
        case "mean":
            # Weighted sum
            combined = np.zeros_like(fields_list[0])
            for field, weight in zip(fields_list, weights_array, strict=True):
                combined += weight * field
            return combined

        case "max":
            max_combined: NDArray[np.float64] = np.maximum.reduce(fields_list)
            return max_combined

        case _:
            min_combined: NDArray[np.float64] = np.minimum.reduce(fields_list)
            return min_combined
