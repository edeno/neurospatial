"""JAX core array operations for encoding computations.

This module provides the JAX implementation of core array operations used
by encoding functions. These are pure array operations that work on shapes
like ``(n_neurons, n_bins)`` and ``(n_bins,)``.

Functions
---------
compute_firing_rate_single
    Convert spike counts and occupancy to firing rate for single neuron.
compute_firing_rates_batch
    Convert spike counts and occupancy to firing rates for multiple neurons.
smooth_rate_map_single
    Apply spatial smoothing to a single firing rate map.
smooth_rate_maps_batch
    Apply spatial smoothing to multiple firing rate maps.

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
neurospatial.encoding._smoothing : Detailed smoothing implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array

__all__ = [
    "compute_firing_rate_single",
    "compute_firing_rates_batch",
    "smooth_rate_map_single",
    "smooth_rate_maps_batch",
    "sparsity_batch",
    "sparsity_single",
    "spatial_information_batch",
    "spatial_information_single",
]


def compute_firing_rate_single(
    spike_counts: Array,
    occupancy: Array,
    *,
    min_occupancy: float = 0.0,
) -> Array:
    """Convert spike counts and occupancy to firing rate for single neuron.

    Computes firing rate as spike_counts / occupancy, with handling for
    low-occupancy bins.

    Parameters
    ----------
    spike_counts : jax.Array, shape (n_bins,)
        Number of spikes in each spatial bin.
    occupancy : jax.Array, shape (n_bins,)
        Time spent in each spatial bin (seconds).
    min_occupancy : float, default=0.0
        Minimum occupancy threshold. Bins with occupancy below this value
        will have NaN firing rate. Use 0.0 to mask only zero-occupancy bins.

    Returns
    -------
    jax.Array, shape (n_bins,)
        Firing rate in Hz (spikes per second). Bins with insufficient
        occupancy are set to NaN.

    Notes
    -----
    This function performs the core rate computation:

    .. math::

        r_i = \\frac{n_i}{t_i}

    where :math:`n_i` is the spike count and :math:`t_i` is the occupancy
    time for bin :math:`i`.

    This function is designed to be compatible with JAX transformations
    like ``jit`` and ``vmap``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from neurospatial.encoding._core_jax import compute_firing_rate_single
    >>> spike_counts = jnp.array([0, 5, 10, 5, 0], dtype=jnp.float64)
    >>> occupancy = jnp.array([1.0, 1.0, 2.0, 1.0, 0.0], dtype=jnp.float64)
    >>> rate = compute_firing_rate_single(spike_counts, occupancy)  # doctest: +SKIP
    >>> rate  # doctest: +SKIP
    Array([ 0.,  5.,  5.,  5., nan], dtype=float64)
    """
    spike_counts = jnp.asarray(spike_counts)
    occupancy = jnp.asarray(occupancy)

    # Compute firing rate with safe division using jnp.where
    # JAX's division handles inf/nan differently, so we use where explicitly
    firing_rate = jnp.where(
        occupancy > 0,
        spike_counts / occupancy,
        jnp.nan,
    )

    # Apply min_occupancy threshold
    if min_occupancy > 0:
        firing_rate = jnp.where(occupancy >= min_occupancy, firing_rate, jnp.nan)

    return firing_rate


def compute_firing_rates_batch(
    spike_counts: Array,
    occupancy: Array,
    *,
    min_occupancy: float = 0.0,
) -> Array:
    """Convert spike counts and occupancy to firing rates for multiple neurons.

    Batch version of :func:`compute_firing_rate_single` that operates on
    multiple neurons efficiently using JAX vectorization.

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
    This is the batch equivalent of :func:`compute_firing_rate_single`.
    It broadcasts the occupancy array across all neurons for efficient
    computation. When JAX is used, this function can leverage ``vmap``
    for automatic vectorization.

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

    # Compute firing rate with safe division (broadcasting over neurons)
    firing_rates = jnp.where(
        occupancy > 0,
        spike_counts / occupancy,
        jnp.nan,
    )

    # Apply min_occupancy threshold
    if min_occupancy > 0:
        firing_rates = jnp.where(occupancy >= min_occupancy, firing_rates, jnp.nan)

    return firing_rates


def smooth_rate_map_single(
    firing_rate: Array,
    adjacency: Array,
    *,
    bandwidth: float = 5.0,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
) -> Array:
    """Apply spatial smoothing to a single firing rate map.

    Parameters
    ----------
    firing_rate : jax.Array, shape (n_bins,)
        Unsmoothed firing rate map (Hz).
    adjacency : jax.Array, shape (n_bins, n_bins)
        Adjacency matrix encoding spatial connectivity between bins.
        Used for graph-based smoothing methods. For diffusion_kde and
        gaussian_kde, this should be a row-normalized kernel/weight matrix.
    bandwidth : float, default=5.0
        Smoothing bandwidth in physical units (e.g., cm). The interpretation
        depends on the smoothing method. For "binned" method with bandwidth=0,
        the input is returned unchanged.
    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Smoothing method to apply:

        - ``"diffusion_kde"``: Graph-based boundary-aware kernel density
          estimation using heat diffusion on the connectivity graph.
        - ``"gaussian_kde"``: Standard Gaussian kernel smoothing, treating
          the rate map as a 2D image.
        - ``"binned"``: Returns input unchanged (no smoothing applied).

    Returns
    -------
    jax.Array, shape (n_bins,)
        Smoothed firing rate map.

    Notes
    -----
    The ``"diffusion_kde"`` method is recommended for environments with
    irregular boundaries or obstacles, as it respects the connectivity
    structure and avoids bleeding across barriers.

    This function is designed to be compatible with JAX transformations
    like ``jit`` for efficient computation.

    This function expects the adjacency/kernel matrix to already be computed.
    For higher-level smoothing that computes kernels from Environment objects,
    use :func:`neurospatial.encoding._smoothing.smooth_rate_map`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from neurospatial.encoding._core_jax import smooth_rate_map_single
    >>> firing_rate = jnp.array([0.0, 1.0, 5.0, 1.0, 0.0], dtype=jnp.float64)
    >>> adjacency = jnp.eye(5, dtype=jnp.float64)  # Identity - no smoothing
    >>> smoothed = smooth_rate_map_single(
    ...     firing_rate, adjacency, bandwidth=2.0, method="binned"
    ... )  # doctest: +SKIP
    >>> smoothed  # doctest: +SKIP
    Array([0., 1., 5., 1., 0.], dtype=float64)
    """
    firing_rate = jnp.asarray(firing_rate)
    adjacency = jnp.asarray(adjacency)

    if method == "binned":
        # No smoothing - return input unchanged
        return firing_rate

    # For diffusion_kde and gaussian_kde, apply kernel smoothing
    # The adjacency matrix is the smoothing kernel (row-normalized weights)
    smoothed = adjacency @ firing_rate

    return smoothed


def smooth_rate_maps_batch(
    firing_rates: Array,
    adjacency: Array,
    *,
    bandwidth: float = 5.0,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
) -> Array:
    """Apply spatial smoothing to multiple firing rate maps.

    Batch version of :func:`smooth_rate_map_single` that operates on
    multiple neurons efficiently using JAX vectorization.

    Parameters
    ----------
    firing_rates : jax.Array, shape (n_neurons, n_bins)
        Unsmoothed firing rate maps for each neuron (Hz).
    adjacency : jax.Array, shape (n_bins, n_bins)
        Adjacency matrix encoding spatial connectivity between bins.
        Shared across all neurons. For diffusion_kde and gaussian_kde,
        this should be a row-normalized kernel/weight matrix.
    bandwidth : float, default=5.0
        Smoothing bandwidth in physical units (e.g., cm).
    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Smoothing method to apply. See :func:`smooth_rate_map_single` for
        detailed descriptions of each method.

    Returns
    -------
    jax.Array, shape (n_neurons, n_bins)
        Smoothed firing rate maps for each neuron.

    Notes
    -----
    This function applies the same smoothing kernel to all neurons,
    which is more efficient than calling :func:`smooth_rate_map_single`
    in a loop when the adjacency matrix and bandwidth are shared.

    When using JAX, this function can leverage ``vmap`` for automatic
    vectorization over neurons.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from neurospatial.encoding._core_jax import smooth_rate_maps_batch
    >>> firing_rates = jnp.array(
    ...     [
    ...         [0.0, 1.0, 5.0, 1.0, 0.0],
    ...         [1.0, 3.0, 1.0, 0.0, 0.0],
    ...     ],
    ...     dtype=jnp.float64,
    ... )
    >>> adjacency = jnp.eye(5, dtype=jnp.float64)
    >>> smoothed = smooth_rate_maps_batch(
    ...     firing_rates, adjacency, bandwidth=2.0, method="binned"
    ... )  # doctest: +SKIP
    >>> smoothed.shape  # doctest: +SKIP
    (2, 5)
    """
    firing_rates = jnp.asarray(firing_rates)
    adjacency = jnp.asarray(adjacency)

    if method == "binned":
        # No smoothing - return input unchanged
        return firing_rates

    # For diffusion_kde and gaussian_kde, apply kernel smoothing
    # Batch matrix multiplication: (n_bins, n_bins) @ (n_bins, n_neurons).T
    # Result: (n_neurons, n_bins)
    smoothed = (adjacency @ firing_rates.T).T

    return smoothed


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
        Firing rate map in Hz. Can contain NaN values which are ignored.
    occupancy : jax.Array, shape (n_bins,)
        Time spent in each bin (seconds or any time unit). Will be normalized
        to probability internally. Can contain NaN values which are ignored.
    base : float, default=2.0
        Logarithm base. Use 2.0 for bits (standard), e for nats.

    Returns
    -------
    jax.Array (scalar)
        Spatial information in bits per spike (if base=2.0).
        Returns 0.0 if mean rate is zero or undefined.

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

    # Replace NaN with 0 for computation (will be masked out by occupancy anyway)
    firing_rate_clean = jnp.nan_to_num(firing_rate, nan=0.0)
    occupancy_clean = jnp.nan_to_num(occupancy, nan=0.0)

    # Normalize occupancy to probability
    occ_sum = jnp.sum(occupancy_clean)
    # Handle zero total occupancy
    occupancy_prob = jnp.where(
        occ_sum > 0,
        occupancy_clean / occ_sum,
        jnp.zeros_like(occupancy_clean),
    )

    # Mean firing rate (weighted by occupancy)
    mean_rate = jnp.sum(occupancy_prob * firing_rate_clean)

    # For valid bins: positive occupancy and positive finite firing rate
    # We need valid_mask for computing information only in valid bins
    valid_mask = (occupancy_prob > 0) & (firing_rate_clean > 0)

    # Compute ratio = r_i / mean_rate where valid
    # Use where to avoid division by zero
    ratio = jnp.where(
        (mean_rate > 0) & valid_mask,
        firing_rate_clean / mean_rate,
        1.0,  # Use 1.0 to give log(1) = 0 for invalid bins
    )

    # Compute p * ratio * log(ratio) for each bin
    # log(ratio) where ratio = 1 gives 0, so invalid bins contribute nothing
    log_ratio = jnp.log(ratio) / jnp.log(base)
    contribution = jnp.where(
        valid_mask & (mean_rate > 0),
        occupancy_prob * ratio * log_ratio,
        0.0,
    )

    information = jnp.sum(contribution)

    # Clamp to non-negative (floating point errors can produce tiny negatives)
    information = jnp.maximum(information, 0.0)

    # Return 0 if mean_rate is 0 or occupancy sum is 0
    return jnp.where((mean_rate > 0) & (occ_sum > 0), information, 0.0)


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
        Shared occupancy for all neurons (time spent in each bin).
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
    import jax

    firing_rates = jnp.asarray(firing_rates)
    occupancy = jnp.asarray(occupancy)

    # Handle empty batch
    if firing_rates.shape[0] == 0:
        return jnp.array([], dtype=firing_rates.dtype)

    # vmap over first axis of firing_rates
    vmapped_fn = jax.vmap(
        lambda rate: spatial_information_single(rate, occupancy, base=base)
    )
    return vmapped_fn(firing_rates)


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
        Firing rate map in Hz. Can contain NaN values which are ignored.
    occupancy : jax.Array, shape (n_bins,)
        Time spent in each bin (seconds or any time unit). Will be normalized
        to probability internally. Can contain NaN values which are ignored.

    Returns
    -------
    jax.Array (scalar)
        Sparsity value in range [0, 1]. Lower values indicate sparser firing.
        Returns 0.0 if denominator is zero or undefined.

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

    # Replace NaN with 0 for computation
    firing_rate_clean = jnp.nan_to_num(firing_rate, nan=0.0)
    occupancy_clean = jnp.nan_to_num(occupancy, nan=0.0)

    # Normalize occupancy to probability
    occ_sum = jnp.sum(occupancy_clean)
    occupancy_prob = jnp.where(
        occ_sum > 0,
        occupancy_clean / occ_sum,
        jnp.zeros_like(occupancy_clean),
    )

    # Compute sparsity
    numerator = jnp.sum(occupancy_prob * firing_rate_clean) ** 2
    denominator = jnp.sum(occupancy_prob * firing_rate_clean**2)

    # Safe division
    sparsity_value = jnp.where(
        denominator > 0,
        numerator / denominator,
        0.0,
    )

    # Clamp to [0, 1] to handle floating point precision issues
    return jnp.clip(sparsity_value, 0.0, 1.0)


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
        Shared occupancy for all neurons (time spent in each bin).

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
    import jax

    firing_rates = jnp.asarray(firing_rates)
    occupancy = jnp.asarray(occupancy)

    # Handle empty batch
    if firing_rates.shape[0] == 0:
        return jnp.array([], dtype=firing_rates.dtype)

    # vmap over first axis of firing_rates
    vmapped_fn = jax.vmap(lambda rate: sparsity_single(rate, occupancy))
    return vmapped_fn(firing_rates)
