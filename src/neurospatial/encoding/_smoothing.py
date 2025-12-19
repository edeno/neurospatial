"""Shared smoothing implementations for encoding rate map computation.

This module provides smoothing functions that are used by the encoding
compute functions (compute_spatial_rate, compute_directional_rate, etc.).

The functions in this module operate on dense arrays:
- Single neuron: spike_counts (n_bins,), occupancy (n_bins,)
- Batch: spike_counts (n_neurons, n_bins), occupancy (n_bins,)

Three smoothing methods are supported:
- **diffusion_kde**: Graph-based boundary-aware KDE (recommended)
- **gaussian_kde**: Standard Euclidean KDE (ignores boundaries)
- **binned**: Legacy method with bin-then-smooth order

The key difference between methods is the order of operations:
- diffusion_kde/gaussian_kde: Smooth counts → Smooth occupancy → Normalize
- binned: Normalize → Smooth result

References
----------
.. [1] Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. (1993).
       An information-theoretic approach to deciphering the hippocampal code.
.. [2] Barry, C., et al. (2006). The boundary vector cell model of
       place cell firing and spatial memory. Reviews in the Neurosciences.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.environment._protocols import EnvironmentProtocol

__all__ = [
    "smooth_rate_map",
    "smooth_rate_maps_batch",
]


def smooth_rate_map(
    env: Environment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy: float = 0.0,
    kernel: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute smoothed firing rate map from spike counts and occupancy.

    This function applies smoothing to spike counts and occupancy to compute
    a firing rate map. The smoothing method determines the order of operations
    and the type of smoothing applied.

    Parameters
    ----------
    env : Environment
        The spatial environment. Used for graph structure and kernel computation.
    spike_counts : ndarray, shape (n_bins,)
        Number of spikes in each spatial bin.
    occupancy : ndarray, shape (n_bins,)
        Time spent in each spatial bin (seconds).
    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Smoothing method:

        - **diffusion_kde** (recommended): Graph-based boundary-aware KDE.
          Respects environment boundaries (walls, obstacles). Uses diffusion
          kernel computed from environment graph. Order: smooth → normalize.
        - **gaussian_kde**: Standard Euclidean KDE. Uses Gaussian kernel based
          on Euclidean distance between bin centers. Ignores boundaries (mass
          can "bleed through" walls). Order: smooth → normalize.
        - **binned**: Legacy method. Computes raw rate first, then smooths.
          Order: normalize → smooth. Can introduce discretization artifacts.

    bandwidth : float, default=5.0
        Smoothing bandwidth in the same units as bin_size. Larger values
        produce more smoothing. For diffusion_kde, this is the kernel
        bandwidth σ. For gaussian_kde, this is the Gaussian σ. For binned,
        this is passed to env.smooth().
    min_occupancy : float, default=0.0
        Minimum occupancy (seconds) for a bin to be included. Bins with
        occupancy below this threshold are set to NaN.
    kernel : ndarray, shape (n_bins, n_bins), optional
        Precomputed diffusion kernel for efficiency when processing multiple
        neurons. Only used with method="diffusion_kde". If None, the kernel
        is computed from the environment.

    Returns
    -------
    ndarray, shape (n_bins,)
        Smoothed firing rate in Hz (spikes/second). Bins with zero or
        low occupancy are NaN.

    Raises
    ------
    ValueError
        If method is not one of the valid options.
        If bandwidth is negative.
        If spike_counts and occupancy have different shapes.
        If spike_counts shape doesn't match env.n_bins.

    Notes
    -----
    **Method Comparison**:

    +--------------+----------------+------------+--------------+
    | Method       | Boundaries     | Speed      | Artifacts    |
    +==============+================+============+==============+
    | diffusion_kde| Respects       | Medium     | None         |
    +--------------+----------------+------------+--------------+
    | gaussian_kde | Ignores        | Slow       | Wall bleed   |
    +--------------+----------------+------------+--------------+
    | binned       | Respects*      | Fast       | Discretization|
    +--------------+----------------+------------+--------------+

    *binned uses graph smoothing but applies it after normalization.

    **Algorithm Details**:

    For diffusion_kde and gaussian_kde (correct KDE order):
    1. Smooth spike counts using kernel
    2. Smooth occupancy using kernel
    3. Compute rate: smoothed_spikes / smoothed_occupancy

    For binned (legacy order):
    1. Compute raw rate: spike_counts / occupancy
    2. Smooth the rate map

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._smoothing import smooth_rate_map

    >>> # Create environment
    >>> positions = np.random.rand(1000, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Simulate spike counts and occupancy
    >>> spike_counts = np.random.poisson(5, env.n_bins).astype(float)
    >>> occupancy = np.ones(env.n_bins) * 1.0  # 1 second per bin

    >>> # Compute smoothed rate map
    >>> rate_map = smooth_rate_map(
    ...     env, spike_counts, occupancy, method="diffusion_kde", bandwidth=10.0
    ... )
    >>> rate_map.shape
    (400,)

    See Also
    --------
    smooth_rate_maps_batch : Batch version for multiple neurons
    Environment.compute_kernel : Compute diffusion kernel
    Environment.smooth : Apply smoothing to a field
    """
    # Input validation
    _validate_smoothing_inputs(env, spike_counts, occupancy, method, bandwidth)

    # Dispatch to appropriate method
    match method:
        case "diffusion_kde":
            return _diffusion_kde(
                env, spike_counts, occupancy, bandwidth, min_occupancy, kernel
            )
        case "gaussian_kde":
            return _gaussian_kde(env, spike_counts, occupancy, bandwidth, min_occupancy)
        case "binned":
            return _binned(env, spike_counts, occupancy, bandwidth, min_occupancy)
        case _:
            # This should never be reached due to validation
            raise ValueError(f"Unknown method: {method}")


def smooth_rate_maps_batch(
    env: Environment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy: float = 0.0,
    kernel: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute smoothed firing rate maps for multiple neurons.

    Batch version of smooth_rate_map that efficiently processes multiple
    neurons sharing the same occupancy. The kernel is computed once and
    reused for all neurons.

    Parameters
    ----------
    env : Environment
        The spatial environment.
    spike_counts : ndarray, shape (n_neurons, n_bins)
        Number of spikes in each spatial bin for each neuron.
    occupancy : ndarray, shape (n_bins,)
        Shared time spent in each spatial bin (seconds).
    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Smoothing method. See smooth_rate_map for details.
    bandwidth : float, default=5.0
        Smoothing bandwidth.
    min_occupancy : float, default=0.0
        Minimum occupancy threshold.
    kernel : ndarray, shape (n_bins, n_bins), optional
        Precomputed diffusion kernel.

    Returns
    -------
    ndarray, shape (n_neurons, n_bins)
        Smoothed firing rates in Hz for each neuron.

    Raises
    ------
    ValueError
        If spike_counts is not 2D.
        If spike_counts.shape[1] != occupancy.shape[0].
        If spike_counts.shape[1] != env.n_bins.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._smoothing import smooth_rate_maps_batch

    >>> # Create environment
    >>> positions = np.random.rand(1000, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Simulate spike counts for 10 neurons
    >>> spike_counts = np.random.poisson(5, (10, env.n_bins)).astype(float)
    >>> occupancy = np.ones(env.n_bins) * 1.0

    >>> # Compute smoothed rate maps
    >>> rate_maps = smooth_rate_maps_batch(
    ...     env, spike_counts, occupancy, method="diffusion_kde", bandwidth=10.0
    ... )
    >>> rate_maps.shape
    (10, 400)

    See Also
    --------
    smooth_rate_map : Single-neuron version
    """
    # Validate batch-specific requirements
    spike_counts = np.asarray(spike_counts)
    occupancy = np.asarray(occupancy)

    if spike_counts.ndim != 2:
        raise ValueError(
            f"spike_counts must be 2D (n_neurons, n_bins), got shape {spike_counts.shape}"
        )

    if spike_counts.shape[1] != occupancy.shape[0]:
        raise ValueError(
            f"spike_counts has {spike_counts.shape[1]} bins but "
            f"occupancy has {occupancy.shape[0]} bins"
        )

    if spike_counts.shape[1] != env.n_bins:
        raise ValueError(
            f"spike_counts has {spike_counts.shape[1]} bins but "
            f"env has {env.n_bins} bins (n_bins mismatch)"
        )

    n_neurons = spike_counts.shape[0]

    # For diffusion_kde, precompute kernel once and reuse
    if method == "diffusion_kde" and kernel is None:
        kernel = cast("EnvironmentProtocol", env).compute_kernel(
            bandwidth, mode="density", cache=True
        )

    # Process each neuron
    # Note: Could be parallelized with joblib, but kept simple for now
    # JAX backend can use vmap for efficient batch processing
    result = np.zeros_like(spike_counts, dtype=np.float64)
    for i in range(n_neurons):
        result[i] = smooth_rate_map(
            env,
            spike_counts[i],
            occupancy,
            method=method,
            bandwidth=bandwidth,
            min_occupancy=min_occupancy,
            kernel=kernel,
        )

    return result


# =============================================================================
# Private Implementation Functions
# =============================================================================


def _validate_smoothing_inputs(
    env: Environment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    method: str,
    bandwidth: float,
) -> None:
    """Validate inputs for smoothing functions."""
    # Validate method
    valid_methods = {"diffusion_kde", "gaussian_kde", "binned"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    # Validate bandwidth (binned allows 0)
    if method == "binned":
        if bandwidth < 0:
            raise ValueError(f"bandwidth must be non-negative, got {bandwidth}")
    else:
        if bandwidth <= 0:
            raise ValueError(f"bandwidth must be positive, got {bandwidth}")

    # Convert to arrays
    spike_counts = np.asarray(spike_counts)
    occupancy = np.asarray(occupancy)

    # Check shapes match
    if spike_counts.shape != occupancy.shape:
        raise ValueError(
            f"spike_counts shape {spike_counts.shape} does not match "
            f"occupancy shape {occupancy.shape}"
        )

    # Check matches environment
    if spike_counts.shape[0] != env.n_bins:
        raise ValueError(
            f"spike_counts has {spike_counts.shape[0]} elements but "
            f"env has {env.n_bins} bins (n_bins mismatch)"
        )


def _diffusion_kde(
    env: Environment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    bandwidth: float,
    min_occupancy: float,
    kernel: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Apply diffusion KDE smoothing.

    Algorithm (correct KDE order):
    1. Spread spike counts using diffusion kernel
    2. Spread occupancy using diffusion kernel
    3. Normalize: spike_density / occupancy_density
    """
    spike_counts = np.asarray(spike_counts, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)

    # Get or compute kernel
    if kernel is None:
        kernel = cast("EnvironmentProtocol", env).compute_kernel(
            bandwidth, mode="density", cache=True
        )

    # Spread spike counts using kernel (kernel @ counts)
    spike_density = kernel @ spike_counts

    # Spread occupancy using kernel
    occupancy_density = kernel @ occupancy

    # Compute firing rate with safe division
    with np.errstate(divide="ignore", invalid="ignore"):
        firing_rate = np.where(
            occupancy_density > 0,
            spike_density / occupancy_density,
            np.nan,
        )

    # Apply min_occupancy threshold
    if min_occupancy > 0:
        firing_rate = np.where(occupancy >= min_occupancy, firing_rate, np.nan)

    return firing_rate.astype(np.float64)


def _gaussian_kde(
    env: Environment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    bandwidth: float,
    min_occupancy: float,
) -> NDArray[np.float64]:
    """Apply Gaussian KDE smoothing (Euclidean distance).

    Algorithm:
    1. For each bin, compute Gaussian-weighted spike density from all bins
    2. For each bin, compute Gaussian-weighted occupancy density
    3. Normalize: spike_density / occupancy_density

    Note: This ignores graph connectivity and uses Euclidean distance.
    Mass can "bleed through" walls.
    """
    spike_counts = np.asarray(spike_counts, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)

    bin_centers = env.bin_centers  # (n_bins, n_dims)
    two_sigma_sq = 2.0 * bandwidth**2

    # Compute pairwise squared distances
    # dist_sq[i, j] = ||bin_i - bin_j||^2
    # Using efficient broadcasting: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    bin_sq_norm = np.sum(bin_centers**2, axis=1, keepdims=True)  # (n_bins, 1)
    dist_sq = bin_sq_norm + bin_sq_norm.T - 2 * (bin_centers @ bin_centers.T)
    dist_sq = np.maximum(dist_sq, 0)  # Numerical stability

    # Gaussian weights: exp(-dist^2 / (2*sigma^2))
    weights = np.exp(-dist_sq / two_sigma_sq)  # (n_bins, n_bins)

    # Compute spike density: sum of Gaussian-weighted spikes
    spike_density = weights @ spike_counts  # (n_bins,)

    # Compute occupancy density: sum of Gaussian-weighted occupancy
    occupancy_density = weights @ occupancy  # (n_bins,)

    # Normalize
    with np.errstate(divide="ignore", invalid="ignore"):
        firing_rate = np.where(
            occupancy_density > 0,
            spike_density / occupancy_density,
            np.nan,
        )

    # Apply min_occupancy threshold
    if min_occupancy > 0:
        firing_rate = np.where(occupancy >= min_occupancy, firing_rate, np.nan)

    return firing_rate.astype(np.float64)


def _binned(
    env: Environment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    bandwidth: float,
    min_occupancy: float,
) -> NDArray[np.float64]:
    """Apply binned smoothing (legacy method).

    Algorithm (legacy order):
    1. Compute raw rate: spike_counts / occupancy
    2. Apply diffusion smoothing to the rate

    This is the legacy order that can introduce discretization artifacts.
    """
    spike_counts = np.asarray(spike_counts, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)

    # Step 1: Compute raw firing rate
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_rate = np.where(occupancy > 0, spike_counts / occupancy, np.nan)

    # Apply min_occupancy threshold before smoothing
    if min_occupancy > 0:
        raw_rate = np.where(occupancy >= min_occupancy, raw_rate, np.nan)

    # Step 2: Smooth the rate map (if bandwidth > 0)
    if bandwidth <= 0:
        return raw_rate.astype(np.float64)

    # Handle NaN values by smoothing with weight normalization
    nan_mask = np.isnan(raw_rate)

    if np.all(nan_mask):
        # All NaN, nothing to smooth
        return raw_rate.astype(np.float64)

    # Fill NaN with 0 for smoothing
    rate_filled = raw_rate.copy()
    rate_filled[nan_mask] = 0.0

    # Create weights (1 where valid, 0 where NaN)
    weights = np.ones_like(raw_rate)
    weights[nan_mask] = 0.0

    # Smooth both rate and weights
    rate_smoothed = cast("EnvironmentProtocol", env).smooth(
        rate_filled, bandwidth=bandwidth
    )
    weights_smoothed = cast("EnvironmentProtocol", env).smooth(
        weights, bandwidth=bandwidth
    )

    # Normalize by smoothed weights
    with np.errstate(divide="ignore", invalid="ignore"):
        firing_rate = np.where(
            weights_smoothed > 0,
            rate_smoothed / weights_smoothed,
            np.nan,
        )

    return firing_rate.astype(np.float64)
