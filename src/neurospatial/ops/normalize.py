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
>>> normalized.sum()
1.0
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "clamp",
    "combine_fields",
    "normalize_field",
]


def normalize_field(
    field: NDArray[np.float64],
    *,
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """
    Normalize field to sum to 1 (probability distribution).

    Parameters
    ----------
    field : NDArray[np.float64], shape (n_bins,)
        Field values to normalize. Must be non-negative.
    eps : float, default=1e-12
        Small constant for numerical stability (unused, kept for API compatibility).

    Returns
    -------
    normalized : NDArray[np.float64], shape (n_bins,)
        Normalized field where values sum to 1.0.

    Raises
    ------
    ValueError
        If field contains negative values, NaN, Inf, is all zeros, or eps is non-positive.

    Examples
    --------
    >>> field = np.array([1.0, 2.0, 3.0])
    >>> normalized = normalize_field(field)
    >>> print(normalized.sum())
    1.0
    >>> print(normalized)
    [0.166... 0.333... 0.5]

    Notes
    -----
    The normalization preserves relative proportions:
    normalized[i] / normalized[j] = field[i] / field[j]
    """
    field = np.asarray(field, dtype=np.float64)

    if eps <= 0:
        raise ValueError(
            f"eps must be positive (got {eps}). "
            "Provide a small positive constant like 1e-12."
        )

    # Validate input
    if not np.isfinite(field).all():
        if np.isnan(field).any():
            n_nan = np.isnan(field).sum()
            raise ValueError(
                f"Field contains NaN values ({n_nan} NaN entries). "
                "Remove or impute NaN values before normalizing."
            )
        if np.isinf(field).any():
            n_inf = np.isinf(field).sum()
            raise ValueError(
                f"Field contains Inf values ({n_inf} Inf entries). "
                "Replace Inf values before normalizing."
            )

    if (field < 0).any():
        n_negative = (field < 0).sum()
        raise ValueError(
            f"Field contains negative values ({n_negative} negative entries). "
            "Normalization requires non-negative values."
        )

    total = field.sum()
    if total == 0.0:
        raise ValueError(
            f"Cannot normalize field: all zeros (sum={total:.2e}). "
            "Provide a field with positive values."
        )

    normalized: NDArray[np.float64] = field / total
    return normalized


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
        raise ValueError(
            f"lo ({lo}) must be less than or equal to hi ({hi}). Provide valid bounds."
        )

    return np.clip(field, lo, hi)


def combine_fields(
    fields: Sequence[NDArray[np.float64]],
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
        length, weights don't sum to 1, or weights provided with max/min mode.

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
        raise ValueError("At least one field required for combining.")

    # Convert to list of arrays
    fields_list = [np.asarray(f, dtype=np.float64) for f in fields]

    # Validate shapes
    shape = fields_list[0].shape
    for i, f in enumerate(fields_list[1:], start=1):
        if f.shape != shape:
            raise ValueError(
                f"All fields must have the same shape. "
                f"Field 0 has shape {shape}, but field {i} has shape {f.shape}."
            )

    # Handle weights parameter
    if weights is not None:
        if mode != "mean":
            raise ValueError(
                f"Weights only valid for mode='mean', got mode='{mode}'. "
                "Remove weights parameter or use mode='mean'."
            )

        if len(weights) != len(fields_list):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of fields ({len(fields_list)})."
            )

        weights_array = np.asarray(weights, dtype=np.float64)
        if not np.isclose(weights_array.sum(), 1.0, atol=1e-6):
            raise ValueError(
                f"Weights must sum to 1, got sum={weights_array.sum():.6f}. "
                "Normalize weights before passing."
            )
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

        case "min":
            min_combined: NDArray[np.float64] = np.minimum.reduce(fields_list)
            return min_combined

        case _:
            raise ValueError(
                f"Unknown mode '{mode}'. Valid modes: 'mean', 'max', 'min'."
            )
