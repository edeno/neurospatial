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
    """Check a training-support mask and return it as an array.

    Parameters
    ----------
    support : array-like of bool, shape (n_bins,)
        Bins with supported training data.
    n_bins : int
        Number of bins in the decoding environment.

    Returns
    -------
    NDArray[np.bool_], shape (n_bins,)
        The mask as an array.

    Raises
    ------
    ValueError
        If the mask is not boolean with shape ``(n_bins,)``, or selects no bin.
    """
    mask = np.asarray(support)
    if mask.shape != (n_bins,) or mask.dtype != np.bool_:
        raise ValueError(
            _format_error(
                f"spatial_support must be a boolean array of shape ({n_bins},), "
                f"got dtype {mask.dtype} with shape {mask.shape}.",
                why="it marks, for each environment bin, whether training data "
                "supports that bin.",
                fix="pass rates.spatial_support from the rate result, or build "
                "the mask on the decoding environment, e.g. "
                "np.isfinite(rates).all(axis=0).",
            )
        )
    if not mask.any():
        raise ValueError(
            _format_error(
                "No spatial bin has supported training data.",
                why="a posterior needs at least one bin with a rate estimate.",
                fix="check recording coverage, epochs, min_speed and min_occupancy.",
            )
        )
    return mask


def models_on_support(models: Any, support: Any) -> Any:
    """Use harmless finite placeholders only in bins the prior excludes."""
    if support is None:
        return models
    return np.where(np.asarray(support), models, 0.0)


def validate_prior(
    prior: Any, n_time: int, n_bins: int, *, context: str
) -> NDArray[np.float64]:
    """Check a user prior and return it as a float array.

    Parameters
    ----------
    prior : array-like, shape (n_bins,) or (n_time, n_bins)
        Non-negative weights per bin, normalized by the decoder.
    n_time, n_bins : int
        Number of decoded time bins and environment bins.
    context : str
        Name of the calling operation, used in the error message.

    Returns
    -------
    NDArray[np.float64], shape (n_bins,) or (n_time, n_bins)

    Raises
    ------
    ValueError
        If the shape is wrong, an entry is negative or non-finite, or a row
        has no positive weight.
    """
    values = np.asarray(prior, dtype=np.float64)
    if values.shape not in ((n_bins,), (n_time, n_bins)):
        raise ValueError(
            _format_error(
                f"{context}: prior has shape {values.shape}; expected "
                f"({n_bins},) for one prior over the {n_bins} bins, or "
                f"({n_time}, {n_bins}) for one row per decoded time bin.",
                why="the prior gives a weight to each environment bin.",
                fix="build the prior on the decoding environment, e.g. "
                "mask * env.bin_sizes for a prior uniform over space on mask.",
            )
        )
    if not np.isfinite(values).all() or np.any(values < 0):
        n_bad = int(np.sum(~np.isfinite(values) | (values < 0)))
        raise ValueError(
            _format_error(
                f"{context}: prior has {n_bad} negative or non-finite entr"
                f"{'y' if n_bad == 1 else 'ies'}.",
                why="a prior is a non-negative weight per bin; a negative "
                "weight would give negative posterior probability.",
                fix="pass finite non-negative weights; use 0 to exclude a bin.",
            )
        )
    if np.any(values.sum(axis=-1) <= 0):
        raise ValueError(
            _format_error(
                f"{context}: prior has zero total mass"
                f"{' in at least one time bin' if values.ndim == 2 else ''}.",
                why="normalizing an all-zero prior gives no posterior.",
                fix="give at least one bin a positive weight, or pass "
                "prior=None for the prior uniform over space.",
            )
        )
    return values


def resolve_prior(
    env: Environment, prior: Any, support: Any, n_time: int, *, context: str
) -> Any:
    """Return the decoding prior: a probability mass per bin.

    With no ``prior``, the prior is uniform over space on the supported bins:
    each bin's mass is proportional to its length, area or volume. ``None`` is
    returned when that prior is flat over every bin, because a constant prior
    leaves the posterior unchanged. A user prior is validated and then
    intersected with the supported bins.
    """
    mask = None if support is None else validate_support(support, env.n_bins)
    if prior is None:
        weights = relative_bin_measures(env, context=context)
        if mask is None:
            return weights
        return mask.astype(float) if weights is None else weights * mask
    values = validate_prior(prior, n_time, env.n_bins, context=context)
    if mask is None:
        return values
    values = values * mask
    if np.any(values.sum(axis=-1) <= 0):
        raise ValueError(
            _format_error(
                f"{context}: prior assigns no probability to supported training bins.",
                why="bins without training support have zero posterior mass.",
                fix="choose a prior overlapping spatial_support, or collect "
                "training data where the prior has mass.",
            )
        )
    return values


def exclude_bins(
    env: Environment, prior: Any, excluded: NDArray[np.bool_], *, context: str
) -> NDArray[np.float64]:
    """Return ``prior`` with zero mass in the ``excluded`` bins.

    Parameters
    ----------
    env : Environment
        Decoding environment; its bin sizes give the default prior.
    prior : NDArray[np.float64] or None, shape (n_bins,) or (n_time, n_bins)
        Resolved prior; ``None`` is the prior uniform over space.
    excluded : NDArray[np.bool_], shape (n_bins,)
        Bins that get zero posterior mass.
    context : str
        Name of the calling operation, used in the error message.

    Raises
    ------
    ValueError
        If no prior mass remains outside the excluded bins.
    """
    if prior is None:
        weights = relative_bin_measures(env, context=context)
        prior = np.ones(env.n_bins) if weights is None else weights
    values = np.where(excluded, 0.0, np.asarray(prior, dtype=np.float64))
    if np.any(values.sum(axis=-1) <= 0):
        raise ValueError(
            _format_error(
                f"{context}: the prior has no mass in the bins where every "
                "neuron has a finite rate.",
                why="bins with a missing (NaN or Inf) rate for any neuron get "
                "zero posterior mass.",
                fix="choose a prior with mass where the rates are finite, or "
                "recompute rates covering the prior's bins.",
            )
        )
    return values
