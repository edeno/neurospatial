"""JAX core array operations for encoding computations.

This module provides the JAX implementation of core array operations used
by encoding functions. These are pure array operations that work on shapes
like ``(n_neurons, n_bins)`` and ``(n_bins,)``.

Functions
---------
compute_firing_rates_batch
    Convert spike counts and occupancy to firing rates for multiple neurons.
spatial_information_single, spatial_information_batch
    Skaggs spatial information (bits per spike).
sparsity_single, sparsity_batch
    Skaggs sparsity.

Notes
-----
This module provides JAX implementations of core encoding operations.
These functions are designed to be compatible with JAX transformations
like ``jit``, ``vmap``, and ``grad`` for GPU acceleration and automatic
differentiation.

The NumPy equivalent of this module is ``_core_numpy.py``, which provides
the same interface but uses NumPy operations for CPU computation.

This module requires JAX to be installed. It is only supported on
Linux and macOS platforms.

See Also
--------
neurospatial.encoding._backend : Backend selection infrastructure.
neurospatial.encoding._core_numpy : NumPy implementation of these functions.
neurospatial.encoding._metrics : Shared metric implementations.
neurospatial.encoding._smoothing : Rate-map smoothing.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax

# Enable JAX's float64 mode globally. The encoding pipeline computes
# spike-rate maps in float64 throughout (see compute_*_rate's explicit
# `dtype=jnp.float64` casts) and the kernel comparisons are precision-
# sensitive: e.g. a `min_occupancy` threshold differing from `occupancy`
# in sub-float32 bits silently keeps bins under JAX's default x32 mode.
# Doing this at JAX-backend import time is the standard pattern for
# float64-required JAX libraries; it has no effect on processes that
# never import the JAX backend.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  -- must follow the x64 toggle above

if TYPE_CHECKING:
    from jax import Array

__all__ = [
    "compute_firing_rates_batch",
    "sparsity_batch",
    "sparsity_single",
    "spatial_information_batch",
    "spatial_information_single",
]


def compute_firing_rates_batch(
    spike_counts: Array,
    occupancy: Array,
    *,
    min_occupancy: float = 0.0,
) -> Array:
    """Convert spike counts and occupancy to firing rates for multiple neurons.

    Computes each rate as spike_counts / occupancy, NaN where the occupancy
    is zero or below ``min_occupancy``.

    Parameters
    ----------
    spike_counts : jax.Array, shape (n_neurons, n_bins)
        Number of spikes in each spatial bin for each neuron.
    occupancy : jax.Array, shape (n_bins,)
        Time spent in each spatial bin (seconds). Shared across all neurons.
    min_occupancy : float, default=0.0
        Minimum occupancy threshold. Bins with occupancy below this value
        will have NaN firing rate across all neurons.

    Returns
    -------
    jax.Array, shape (n_neurons, n_bins)
        Firing rates in Hz (spikes per second) for each neuron.
        Bins with insufficient occupancy are set to NaN.

    Notes
    -----
    The occupancy array broadcasts across all neurons. The function is
    compatible with JAX transformations like ``jit`` and ``vmap``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from neurospatial.encoding._core_jax import compute_firing_rates_batch
    >>> spike_counts = jnp.array(
    ...     [
    ...         [0, 5, 10, 5, 0],
    ...         [5, 0, 5, 0, 5],
    ...     ],
    ...     dtype=jnp.float64,
    ... )
    >>> occupancy = jnp.array([1.0, 1.0, 2.0, 1.0, 0.0], dtype=jnp.float64)
    >>> rates = compute_firing_rates_batch(spike_counts, occupancy)  # doctest: +SKIP
    >>> rates.shape  # doctest: +SKIP
    (2, 5)
    """
    spike_counts = jnp.asarray(spike_counts)
    occupancy = jnp.asarray(occupancy)
    # Pass min_occupancy at the same dtype as occupancy. Casting to float32
    # would round the threshold and let JAX keep bins NumPy would mask
    # (e.g. occupancy=0.100000001 vs min_occupancy=0.100000005 collapse to
    # equal under float32).
    out: Array = _firing_rates_kernel(
        spike_counts,
        occupancy,
        jnp.asarray(min_occupancy, dtype=occupancy.dtype),
    )
    return out


@jax.jit
def _firing_rates_kernel(
    spike_counts: Array, occupancy: Array, min_occupancy: Array
) -> Array:
    # Single mask covers both safe-divide (occupancy > 0) and the
    # user threshold (occupancy >= min_occupancy). When min_occupancy == 0
    # this collapses to the original "occupancy > 0" semantics.
    valid = (occupancy > 0) & (occupancy >= min_occupancy)
    return jnp.where(valid, spike_counts / occupancy, jnp.nan)


def spatial_information_single(
    firing_rate: Array,
    occupancy: Array,
    *,
    base: float = 2.0,
) -> Array:
    """Compute Skaggs spatial information (bits per spike) for single neuron.

    Spatial information quantifies how much information each spike conveys
    about the animal's spatial location. This is a fundamental metric for
    classifying place cells and other spatially-tuned neurons.

    Parameters
    ----------
    firing_rate : jax.Array, shape (n_bins,)
        Firing rate map in Hz. Bins with non-finite rates are excluded.
    occupancy : jax.Array, shape (n_bins,)
        Time spent in each bin (seconds or any time unit). Will be normalized
        over bins with finite rates and occupancy. Observed zero rates remain
        included.
    base : float, default=2.0
        Logarithm base. Use 2.0 for bits (standard), e for nats.

    Returns
    -------
    jax.Array (scalar)
        Spatial information in bits per spike (if base=2.0).
        NaN for a silent map (mean rate 0), an all-NaN map or zero total
        occupancy: bits per spike are undefined without spikes.

    Notes
    -----
    **Formula (Skaggs et al. 1993)**:

    .. math::

        I = \\sum_i p_i \\frac{r_i}{\\bar{r}} \\log \\left( \\frac{r_i}{\\bar{r}} \\right)

    where :math:`p_i` is occupancy probability, :math:`r_i` is firing rate
    in bin :math:`i`, and :math:`\\bar{r}` is mean firing rate.

    This function is designed to be compatible with JAX transformations
    like ``jit`` and ``vmap`` for GPU acceleration.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from neurospatial.encoding._core_jax import spatial_information_single
    >>> firing_rate = jnp.ones(100) * 3.0  # Uniform
    >>> occupancy = jnp.ones(100)
    >>> info = spatial_information_single(firing_rate, occupancy)  # doctest: +SKIP
    >>> float(info) < 1e-6  # ~0 for uniform  # doctest: +SKIP
    True
    """
    firing_rate = jnp.asarray(firing_rate)
    occupancy = jnp.asarray(occupancy)
    out: Array = _spatial_information_kernel(firing_rate, occupancy, base)
    return out


@partial(jax.jit, static_argnames=("base",))
def _spatial_information_kernel(
    firing_rate: Array, occupancy: Array, base: float
) -> Array:
    valid = jnp.isfinite(firing_rate) & jnp.isfinite(occupancy)
    firing_rate_clean = jnp.where(valid, firing_rate, 0.0)
    occupancy_clean = jnp.where(valid, occupancy, 0.0)

    occ_sum = jnp.sum(occupancy_clean)
    occupancy_prob = jnp.where(
        occ_sum > 0,
        occupancy_clean / jnp.where(occ_sum > 0, occ_sum, 1.0),
        jnp.zeros_like(occupancy_clean),
    )

    mean_rate = jnp.sum(occupancy_prob * firing_rate_clean)
    valid_mask = (occupancy_prob > 0) & (firing_rate_clean > 0)

    ratio = jnp.where(
        (mean_rate > 0) & valid_mask,
        firing_rate_clean / jnp.where(mean_rate > 0, mean_rate, 1.0),
        1.0,
    )
    # Subnormal firing_rate / mean_rate can underflow to 0; clamp so the
    # subsequent log doesn't produce -inf and trigger 0*-inf = NaN.
    ratio = jnp.where(ratio > 0, ratio, 1.0)
    log_ratio = jnp.log(ratio) / jnp.log(base)
    contribution = jnp.where(
        valid_mask & (mean_rate > 0),
        occupancy_prob * ratio * log_ratio,
        0.0,
    )
    information = jnp.maximum(jnp.sum(contribution), 0.0)
    return jnp.where((mean_rate > 0) & (occ_sum > 0), information, jnp.nan)


# Cached vmap of the per-neuron information kernel. Building vmap once at
# module scope (rather than fresh in each batch call) prevents re-tracing.
_spatial_information_vmapped = jax.vmap(
    _spatial_information_kernel, in_axes=(0, None, None)
)


def spatial_information_batch(
    firing_rates: Array,
    occupancy: Array,
    *,
    base: float = 2.0,
) -> Array:
    """Compute Skaggs spatial information for multiple neurons.

    Vectorized version of :func:`spatial_information_single` for efficient
    population analysis using JAX's ``vmap``.

    Parameters
    ----------
    firing_rates : jax.Array, shape (n_neurons, n_bins)
        Firing rate maps for each neuron in Hz.
    occupancy : jax.Array, shape (n_bins,)
        Shared occupancy for all neurons (time spent in each bin), normalized
        separately over each neuron's finite rate/occupancy pairs.
    base : float, default=2.0
        Logarithm base. Use 2.0 for bits (standard), e for nats.

    Returns
    -------
    jax.Array, shape (n_neurons,)
        Spatial information in bits per spike for each neuron.

    Notes
    -----
    This function uses ``vmap`` to vectorize :func:`spatial_information_single`
    over the first axis (neurons) for efficient batch computation.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from neurospatial.encoding._core_jax import spatial_information_batch
    >>> firing_rates = jnp.ones((5, 100)) * 5.0  # 5 neurons, uniform
    >>> occupancy = jnp.ones(100)
    >>> info = spatial_information_batch(firing_rates, occupancy)  # doctest: +SKIP
    >>> info.shape  # doctest: +SKIP
    (5,)
    """
    firing_rates = jnp.asarray(firing_rates)
    occupancy = jnp.asarray(occupancy)

    if firing_rates.shape[0] == 0:
        return jnp.array([], dtype=firing_rates.dtype)

    # Reuse the module-level vmap rather than rebuilding it per call.
    out: Array = _spatial_information_vmapped(firing_rates, occupancy, base)
    return out


def sparsity_single(
    firing_rate: Array,
    occupancy: Array,
) -> Array:
    """Compute sparsity of spatial firing for single neuron.

    Sparsity measures what fraction of the environment elicits significant
    firing. Lower values indicate sparser, more selective place fields.

    Parameters
    ----------
    firing_rate : jax.Array, shape (n_bins,)
        Firing rate map in Hz. Bins with non-finite rates are excluded.
    occupancy : jax.Array, shape (n_bins,)
        Time spent in each bin (seconds or any time unit). Will be normalized
        over bins with finite rates and occupancy. Observed zero rates remain
        included.

    Returns
    -------
    jax.Array (scalar)
        Sparsity value in range [0, 1]. Lower values indicate sparser firing.
        NaN for a silent map, an all-NaN map or zero total occupancy.

    Notes
    -----
    **Formula (Skaggs et al. 1996)**:

    .. math::

        S = \\frac{\\left( \\sum_i p_i r_i \\right)^2}{\\sum_i p_i r_i^2}

    where :math:`p_i` is occupancy probability and :math:`r_i` is firing rate.

    **Interpretation**:

    - Range: [0, 1]
    - Low sparsity (0.1-0.3): Sparse, selective place field
    - High sparsity (~1.0): Uniform firing throughout environment
    - Typical place cells: 0.1-0.3

    This function is designed to be compatible with JAX transformations
    like ``jit`` and ``vmap`` for GPU acceleration.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from neurospatial.encoding._core_jax import sparsity_single
    >>> firing_rate = jnp.ones(100) * 5.0  # Uniform
    >>> occupancy = jnp.ones(100)
    >>> spars = sparsity_single(firing_rate, occupancy)  # doctest: +SKIP
    >>> float(spars) > 0.99  # Close to 1 for uniform  # doctest: +SKIP
    True
    """
    firing_rate = jnp.asarray(firing_rate)
    occupancy = jnp.asarray(occupancy)
    out: Array = _sparsity_kernel(firing_rate, occupancy)
    return out


@jax.jit
def _sparsity_kernel(firing_rate: Array, occupancy: Array) -> Array:
    valid = jnp.isfinite(firing_rate) & jnp.isfinite(occupancy)
    firing_rate_clean = jnp.where(valid, firing_rate, 0.0)
    occupancy_clean = jnp.where(valid, occupancy, 0.0)

    occ_sum = jnp.sum(occupancy_clean)
    occupancy_prob = jnp.where(
        occ_sum > 0,
        occupancy_clean / jnp.where(occ_sum > 0, occ_sum, 1.0),
        jnp.zeros_like(occupancy_clean),
    )

    # Sparsity is unchanged by rescaling rates; dividing by the peak keeps
    # the squared moments from underflowing or overflowing.
    peak = jnp.max(jnp.abs(firing_rate_clean), initial=0.0)
    firing_rate_clean = firing_rate_clean / jnp.where(peak > 0, peak, 1.0)
    numerator = jnp.sum(occupancy_prob * firing_rate_clean) ** 2
    denominator = jnp.sum(occupancy_prob * firing_rate_clean**2)
    sparsity_value = jnp.where(
        denominator > 0,
        numerator / jnp.where(denominator > 0, denominator, 1.0),
        jnp.nan,
    )
    return jnp.clip(sparsity_value, 0.0, 1.0)


# Cached vmap of the per-neuron sparsity kernel.
_sparsity_vmapped = jax.vmap(_sparsity_kernel, in_axes=(0, None))


def sparsity_batch(
    firing_rates: Array,
    occupancy: Array,
) -> Array:
    """Compute sparsity for multiple neurons.

    Vectorized version of :func:`sparsity_single` for efficient population
    analysis using JAX's ``vmap``.

    Parameters
    ----------
    firing_rates : jax.Array, shape (n_neurons, n_bins)
        Firing rate maps for each neuron in Hz.
    occupancy : jax.Array, shape (n_bins,)
        Shared occupancy for all neurons (time spent in each bin), normalized
        separately over each neuron's finite rate/occupancy pairs.

    Returns
    -------
    jax.Array, shape (n_neurons,)
        Sparsity values in range [0, 1] for each neuron.

    Notes
    -----
    This function uses ``vmap`` to vectorize :func:`sparsity_single` over
    the first axis (neurons) for efficient batch computation.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from neurospatial.encoding._core_jax import sparsity_batch
    >>> firing_rates = jnp.ones((5, 100)) * 5.0  # 5 neurons, uniform
    >>> occupancy = jnp.ones(100)
    >>> spars = sparsity_batch(firing_rates, occupancy)  # doctest: +SKIP
    >>> spars.shape  # doctest: +SKIP
    (5,)
    """
    firing_rates = jnp.asarray(firing_rates)
    occupancy = jnp.asarray(occupancy)

    if firing_rates.shape[0] == 0:
        return jnp.array([], dtype=firing_rates.dtype)

    # Reuse the module-level vmap rather than rebuilding it per call.
    out: Array = _sparsity_vmapped(firing_rates, occupancy)
    return out
