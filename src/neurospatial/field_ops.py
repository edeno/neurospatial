"""
Field math utility functions for spatial analysis.

This module provides common operations on bin-valued fields (probability
distributions, rate maps, occupancy, etc.). All functions operate on 1-D
arrays of shape (n_bins,).
"""

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "clamp",
    "combine_fields",
    "kl_divergence",
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
    if mode == "mean":
        # Weighted sum
        combined = np.zeros_like(fields_list[0])
        for field, weight in zip(fields_list, weights_array, strict=True):
            combined += weight * field
        return combined

    elif mode == "max":
        max_combined: NDArray[np.float64] = np.maximum.reduce(fields_list)
        return max_combined

    elif mode == "min":
        min_combined: NDArray[np.float64] = np.minimum.reduce(fields_list)
        return min_combined

    else:
        raise ValueError(f"Unknown mode '{mode}'. Valid modes: 'mean', 'max', 'min'.")


def kl_divergence(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    *,
    kind: Literal["kl", "js", "cosine"] = "kl",
    eps: float = 1e-12,
) -> float:
    """
    Compute divergence between two probability distributions.

    .. note::
        Renamed from ``divergence()`` in v0.3.0 to avoid confusion with the
        graph signal processing divergence operator. This function computes
        statistical divergences (KL, JS, cosine) between probability distributions.

    Parameters
    ----------
    p : NDArray[np.float64], shape (n_bins,)
        First distribution. Should be non-negative and ideally sum to 1.
    q : NDArray[np.float64], shape (n_bins,)
        Second distribution. Should be non-negative and ideally sum to 1.
    kind : {'kl', 'js', 'cosine'}, default='kl'
        Type of divergence:

        - 'kl': Kullback-Leibler divergence D_KL(p || q).
        - 'js': Jensen-Shannon divergence (symmetric, bounded [0, 1]).
        - 'cosine': Cosine distance (1 - cosine_similarity), bounded [0, 2].

    eps : float, default=1e-12
        Small constant to avoid log(0) and division by zero.

    Returns
    -------
    distance : float
        Divergence or distance value. Non-negative for all kinds.

    Raises
    ------
    ValueError
        If p and q have different shapes, contain negative values, NaN, or Inf,
        or if kind is unknown.

    Warns
    -----
    UserWarning
        If distributions don't sum to 1 (only for kl and js kinds).

    Examples
    --------
    >>> p = np.array([0.5, 0.5])
    >>> q = np.array([0.25, 0.75])
    >>> kl = kl_divergence(p, q, kind="kl")
    >>> print(f"{kl:.3f}")
    0.144

    >>> # JS divergence is symmetric
    >>> js_pq = kl_divergence(p, q, kind="js")
    >>> js_qp = kl_divergence(q, p, kind="js")
    >>> print(np.isclose(js_pq, js_qp))
    True

    >>> # Cosine distance
    >>> p = np.array([1.0, 0.0, 0.0])
    >>> q = np.array([0.0, 1.0, 0.0])
    >>> cos_dist = kl_divergence(p, q, kind="cosine")
    >>> print(cos_dist)
    1.0

    Notes
    -----
    **KL Divergence** (Kullback-Leibler):

    .. math::
        D_{KL}(p \\| q) = \\sum_i p_i \\log(p_i / q_i)

    - Not symmetric: D_KL(p||q) ≠ D_KL(q||p)
    - Unbounded: D_KL ∈ [0, ∞)
    - Undefined when q_i = 0 but p_i > 0 (uses eps to avoid this)

    **JS Divergence** (Jensen-Shannon):

    .. math::
        D_{JS}(p \\| q) = \\frac{1}{2} D_{KL}(p \\| m) + \\frac{1}{2} D_{KL}(q \\| m)

    where m = (p + q) / 2

    - Symmetric: D_JS(p||q) = D_JS(q||p)
    - Bounded: D_JS ∈ [0, 1] (when using log base 2)
    - Square root of JS divergence is a metric

    **Cosine Distance**:

    .. math::
        d_{cos}(p, q) = 1 - \\frac{p \\cdot q}{\\|p\\| \\|q\\|}

    - Symmetric
    - Bounded: d_cos ∈ [0, 2]
    - d_cos = 0 for identical directions, 1 for orthogonal, 2 for opposite
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Validate shapes
    if p.shape != q.shape:
        raise ValueError(
            f"p and q must have the same shape. Got p.shape={p.shape}, q.shape={q.shape}."
        )

    # Validate finite values
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        if np.isnan(p).any() or np.isnan(q).any():
            raise ValueError(
                "Distributions contain NaN values. Remove NaN before computing divergence."
            )
        if np.isinf(p).any() or np.isinf(q).any():
            raise ValueError(
                "Distributions contain Inf values. Replace Inf before computing divergence."
            )

    # Validate non-negative (only for probabilistic divergences)
    if kind in ["kl", "js"] and ((p < 0).any() or (q < 0).any()):
        raise ValueError(
            "Distributions must be non-negative for KL/JS divergence. "
            f"Found {(p < 0).sum()} negative values in p, {(q < 0).sum()} in q."
        )

    # Check normalization (warn if not normalized for probabilistic divergences)
    if kind in ["kl", "js"]:
        p_sum = p.sum()
        q_sum = q.sum()
        if not np.isclose(p_sum, 1.0, atol=1e-6):
            warnings.warn(
                f"Distribution p does not sum to 1 (sum={p_sum:.6f}). "
                "Consider normalizing for valid probability distributions.",
                UserWarning,
                stacklevel=2,
            )
        if not np.isclose(q_sum, 1.0, atol=1e-6):
            warnings.warn(
                f"Distribution q does not sum to 1 (sum={q_sum:.6f}). "
                "Consider normalizing for valid probability distributions.",
                UserWarning,
                stacklevel=2,
            )

    # Validate eps
    if eps <= 0:
        raise ValueError(
            f"eps must be positive (got {eps}). "
            "Provide a small positive constant like 1e-12."
        )

    # Compute divergence
    if kind == "kl":
        # KL divergence: D_KL(p || q) = sum(p * log(p / q))
        # By convention, 0 * log(0/q) = 0
        # Add eps only to q to prevent division by zero
        q_safe = q + eps

        # Only include terms where p > 0
        mask = p > 0
        if not mask.any():
            # All p values are zero
            return 0.0

        div = np.sum(p[mask] * np.log(p[mask] / q_safe[mask]))
        return float(div)

    elif kind == "js":
        # JS divergence: D_JS(p || q) = 0.5 * D_KL(p || m) + 0.5 * D_KL(q || m)
        # where m = (p + q) / 2
        m = (p + q) / 2.0
        m_safe = m + eps

        # Compute D_KL(p || m)
        mask_p = p > 0
        if mask_p.any():
            kl_p_m = np.sum(p[mask_p] * np.log(p[mask_p] / m_safe[mask_p]))
        else:
            kl_p_m = 0.0

        # Compute D_KL(q || m)
        mask_q = q > 0
        if mask_q.any():
            kl_q_m = np.sum(q[mask_q] * np.log(q[mask_q] / m_safe[mask_q]))
        else:
            kl_q_m = 0.0

        div = 0.5 * kl_p_m + 0.5 * kl_q_m
        return float(div)

    elif kind == "cosine":
        # Cosine distance: 1 - cos(p, q) = 1 - (p·q) / (||p|| ||q||)
        dot_product = np.dot(p, q)
        norm_p = np.linalg.norm(p)
        norm_q = np.linalg.norm(q)

        if norm_p < eps or norm_q < eps:
            # If either vector is near-zero, cosine similarity is undefined
            # Return maximum distance
            return 2.0

        cosine_similarity = dot_product / (norm_p * norm_q)
        # Clamp to [-1, 1] to handle numerical errors
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
        cosine_distance = 1.0 - cosine_similarity
        return float(cosine_distance)

    else:
        raise ValueError(
            f"Unknown divergence kind '{kind}'. Valid kinds: 'kl', 'js', 'cosine'."
        )
