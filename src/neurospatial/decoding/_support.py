"""Keep training support separate from a zero firing-rate prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error

if TYPE_CHECKING:
    from neurospatial.environment import Environment

# Bin measures computed from bin edges differ by floating-point rounding
# (about 1e-16 relative) even when the bins are geometrically equal. Treat
# differences below this relative size as rounding, so equal bins keep the
# flat-prior and plain-argmax paths exactly.
_EQUAL_MEASURE_RTOL = 1e-12


def relative_bin_measures(env: Environment, *, context: str) -> Any:
    """Each bin's length, area or volume relative to the largest bin.

    Parameters
    ----------
    env : Environment
        Decoding environment.
    context : str
        Name of the calling operation, used in the error message.

    Returns
    -------
    NDArray[np.float64], shape (n_bins,), or None
        ``env.bin_sizes / env.bin_sizes.max()``, or ``None`` when every bin
        has the same measure. Dividing a probability mass by these weights
        gives a value proportional to the probability density over space.

    Raises
    ------
    ValueError
        If the environment mixes bins measured by length and by area, so no
        density over space has a common unit.
    """
    try:
        sizes = np.asarray(env.bin_sizes, dtype=np.float64)
    except ValueError as err:
        raise ValueError(
            _format_error(
                f"{context}: this environment mixes bins measured by length "
                "and by area, so posterior density over space has no common unit.",
                why="the uniform-over-space prior, the MAP position and the "
                "credible region all compare probability per unit length or area.",
                fix="decode each component on its own environment, using "
                "env.bin_sources to select its bins with env.subset(mask=...)",
            )
        ) from err
    if sizes.size == 0:
        return None
    largest = sizes.max()
    if np.all(largest - sizes <= _EQUAL_MEASURE_RTOL * largest):
        return None
    return sizes / largest


def validate_support(support: Any, n_bins: int) -> NDArray[np.bool_]:
    """Check a training-support mask and return it as an array."""
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
    return mask


def models_on_support(models: Any, support: Any) -> Any:
    """Use harmless finite placeholders only in bins the prior excludes."""
    if support is None:
        return models
    return np.where(np.asarray(support), models, 0.0)


def resolve_prior(
    env: Environment, prior: Any, support: Any, n_time: int, *, context: str
) -> Any:
    """Return the decoding prior: a probability mass per bin.

    With no ``prior``, the prior is uniform over space on the supported bins:
    each bin's mass is proportional to its length, area or volume. ``None`` is
    returned when that prior is flat over every bin, because a constant prior
    leaves the posterior unchanged. A user prior is intersected with the
    supported bins after validation.
    """
    mask = None if support is None else validate_support(support, env.n_bins)
    if prior is None:
        weights = relative_bin_measures(env, context=context)
        if mask is None:
            return weights
        return mask.astype(float) if weights is None else weights * mask
    if mask is None:
        return prior
    n_bins = env.n_bins
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
