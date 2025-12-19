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
