"""Keep training support separate from a zero firing-rate prediction."""

from __future__ import annotations

from typing import Any

import numpy as np


def models_on_support(models: Any, support: Any) -> Any:
    """Use harmless finite placeholders only in bins the prior excludes."""
    if support is None:
        return models
    return np.where(np.asarray(support), models, 0.0)


def supported_prior(prior: Any, support: Any, n_bins: int, n_time: int) -> Any:
    """Intersect a user prior with observed training bins, validating first."""
    if support is None:
        return prior
    mask = np.asarray(support)
    if mask.shape != (n_bins,) or mask.dtype != np.bool_:
        raise ValueError(
            f"spatial_support must be a boolean array of shape ({n_bins},)."
        )
    if not mask.any():
        raise ValueError(
            "No spatial bin has supported training data. "
            "Fix: check recording coverage, epochs, min_speed and min_occupancy."
        )
    if prior is None:
        return mask.astype(float)
    values = np.asarray(prior, dtype=float)
    if values.shape not in ((n_bins,), (n_time, n_bins)):
        raise ValueError(
            f"prior must have shape ({n_bins},) or ({n_time}, {n_bins}), "
            f"got {values.shape}."
        )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("prior must contain finite non-negative probabilities.")
    values = values * mask
    if np.any(values.sum(axis=-1) <= 0):
        raise ValueError(
            "prior assigns no probability to supported training bins. "
            "Fix: choose a prior overlapping spatial_support or collect training data there."
        )
    return values
