"""Place cell encoding analysis.

This module provides tools for computing place fields from spike trains.

Spike → Rate Map Conversion
---------------------------
compute_place_field : Compute place field from spike train.

For directional place fields, field detection, and spatial metrics,
see the following modules:
- neurospatial.encoding.spatial : DirectionalPlaceFields, detect_place_fields
- neurospatial.encoding._metrics : spatial_information, sparsity, selectivity
- neurospatial.encoding._field_metrics : field_size, field_stability, etc.

Examples
--------
>>> import numpy as np
>>> from neurospatial import Environment
>>> from neurospatial.encoding.place import compute_place_field
>>>
>>> # Create environment and trajectory
>>> positions = np.random.uniform(0, 100, (1000, 2))
>>> times = np.linspace(0, 100, 1000)
>>> env = Environment.from_samples(positions, bin_size=10.0)
>>>
>>> # Compute place field
>>> spike_times = np.random.uniform(0, 100, 50)
>>> firing_rate = compute_place_field(
...     env, spike_times, times, positions, bandwidth=10.0
... )

See Also
--------
neurospatial.encoding.spatial : Spatial rate computation and field detection.
neurospatial.encoding.grid : Grid cell analysis.
neurospatial.encoding.directional : Head direction cell analysis.
neurospatial.encoding.border : Border/boundary cell analysis.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.environment._protocols import EnvironmentProtocol

# Re-export skaggs_information for backward compatibility
# The canonical name is now spatial_information in _metrics.py
from neurospatial.encoding._metrics import spatial_information as skaggs_information

__all__ = [
    "compute_place_field",
    "skaggs_information",
]


def _interpolate_spike_positions(
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate spike positions from trajectory.

    Uses vectorized binary search and linear interpolation to compute
    the spatial position at each spike time.

    Parameters
    ----------
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Timestamps of spike occurrences. Need not be sorted, but should
        fall within the range of ``times``.
    times : NDArray[np.float64], shape (n_timepoints,)
        Timestamps of trajectory samples (seconds). Must be sorted.
    positions : NDArray[np.float64], shape (n_timepoints, n_dims)
        Position trajectory. Must be 2D.

    Returns
    -------
    spike_positions : NDArray[np.float64], shape (n_spikes, n_dims)
        Interpolated position at each spike time.

    Notes
    -----
    Uses searchsorted for O(log n) lookup per spike, then vectorized
    linear interpolation across all dimensions simultaneously.

    If a spike time falls exactly on a trajectory sample, the position
    at that sample is returned. If spike times fall outside the trajectory
    time range, they are clipped to the nearest endpoint.

    Examples
    --------
    >>> import numpy as np
    >>> times = np.array([0.0, 1.0, 2.0, 3.0])
    >>> positions = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    >>> spike_times = np.array([0.5, 1.5, 2.5])
    >>> spike_pos = _interpolate_spike_positions(spike_times, times, positions)
    >>> spike_pos
    array([[0.5, 1. ],
           [1.5, 3. ],
           [2.5, 5. ]])
    """
    n_spikes = len(spike_times)
    n_dims = positions.shape[1]

    # Handle empty spike array
    if n_spikes == 0:
        return np.zeros((0, n_dims), dtype=np.float64)

    # Use searchsorted to find insertion points
    insert_idx = np.searchsorted(times, spike_times, side="right")
    insert_idx = np.clip(insert_idx, 1, len(times) - 1)

    # Get bounding times for interpolation
    t0 = times[insert_idx - 1]
    t1 = times[insert_idx]

    # Compute interpolation weights, avoiding division by zero
    dt_interp = t1 - t0
    dt_interp = np.where(dt_interp == 0, 1.0, dt_interp)
    weights = (spike_times - t0) / dt_interp

    # Interpolate all dimensions at once: (n_spikes, n_dims)
    pos0 = positions[insert_idx - 1]  # Shape: (n_spikes, n_dims)
    pos1 = positions[insert_idx]  # Shape: (n_spikes, n_dims)
    spike_positions = pos0 + weights[:, np.newaxis] * (pos1 - pos0)

    return spike_positions


def _binned_rate_map(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    min_occupancy_seconds: float = 0.0,
) -> NDArray[np.float64]:
    """Convert spike train to occupancy-normalized firing rate field.

    Computes the spatial firing rate map for a spike train by:
    1. Computing occupancy (time spent in each spatial bin)
    2. Interpolating spike positions from trajectory
    3. Counting spikes per bin
    4. Normalizing by occupancy to get firing rate (spikes/second)
    5. Optionally setting bins with insufficient occupancy to NaN

    This is the standard approach for place field analysis in neuroscience.

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Timestamps of spike occurrences (seconds).
    times : NDArray[np.float64], shape (n_timepoints,)
        Timestamps of trajectory samples (seconds).
    positions : NDArray[np.float64], shape (n_timepoints, n_dims) or (n_timepoints,)
        Position trajectory. For 1D, can be shape (n_timepoints,) or (n_timepoints, 1).
    min_occupancy_seconds : float, default=0.0
        Minimum occupancy (seconds) required for reliable firing rate estimate.
        Bins with less occupancy are set to NaN. Set to 0.0 (default) to include
        all bins. For typical place field analysis, 0.5 seconds is recommended
        to exclude bins with unreliable rate estimates.

    Returns
    -------
    field : NDArray[np.float64], shape (n_bins,)
        Firing rate field (spikes/second) for each spatial bin.
        Bins with insufficient occupancy are set to NaN.

    Raises
    ------
    ValueError
        If times and positions have different lengths.

    Warns
    -----
    UserWarning
        If spikes fall outside the time range of the trajectory.
        If interpolated spike positions fall outside the environment bounds.

    Notes
    -----
    The firing rate field is computed as:

    .. math::
        r_i = \\frac{n_i}{T_i}

    where :math:`n_i` is the spike count in bin :math:`i` and :math:`T_i` is
    the occupancy time (seconds) in that bin.

    Bins with occupancy less than `min_occupancy_seconds` are set to NaN.
    Empty spike trains produce a field of zeros (or NaN where occupancy
    is below threshold).
    """
    # Step 0: Validate inputs
    if len(times) != len(positions):
        raise ValueError(
            f"times and positions must have same length, got {len(times)} and {len(positions)}"
        )

    # Normalize positions to 2D array (n_samples, n_dims)
    # This ensures all downstream code can assume 2D shape
    if positions.ndim == 1:
        positions = positions[:, np.newaxis]
    elif positions.ndim != 2:
        raise ValueError(
            f"positions must be 1D or 2D array, got shape {positions.shape}"
        )

    # Validate min_occupancy_seconds is non-negative
    if min_occupancy_seconds < 0:
        raise ValueError(
            f"min_occupancy_seconds must be non-negative, got {min_occupancy_seconds}"
        )

    # Handle empty spikes
    if len(spike_times) == 0:
        # Compute occupancy to determine which bins to set to NaN
        occupancy = cast("EnvironmentProtocol", env).occupancy(
            times, positions, return_seconds=True
        )
        field = np.zeros(env.n_bins, dtype=np.float64)
        field[occupancy < min_occupancy_seconds] = np.nan
        return field

    # Step 1: Filter spikes to valid time range
    time_min, time_max = times[0], times[-1]
    valid_spike_mask = (spike_times >= time_min) & (spike_times <= time_max)

    # Filter out-of-range spikes if any
    if not np.all(valid_spike_mask):
        n_filtered = np.sum(~valid_spike_mask)
        warnings.warn(
            f"{n_filtered} spike(s) out of time range [{time_min}, {time_max}] will be filtered",
            UserWarning,
            stacklevel=2,
        )
        spike_times = spike_times[valid_spike_mask]

    # Guard clause: handle empty spikes after time filtering
    if len(spike_times) == 0:
        occupancy = cast("EnvironmentProtocol", env).occupancy(
            times, positions, return_seconds=True
        )
        field = np.zeros(env.n_bins, dtype=np.float64)
        field[occupancy < min_occupancy_seconds] = np.nan
        return field

    # Step 2: Compute occupancy using return_seconds=True
    occupancy = cast("EnvironmentProtocol", env).occupancy(
        times, positions, return_seconds=True
    )

    # Step 3: Interpolate spike positions
    spike_positions = _interpolate_spike_positions(spike_times, times, positions)

    # Step 4: Assign spikes to bins
    spike_bins = env.bin_at(spike_positions)

    # Step 5: Filter out-of-bounds spikes (bin_at returns -1 for out-of-bounds)
    valid_bins_mask = spike_bins >= 0

    # Filter out-of-bounds spikes if any
    if not np.all(valid_bins_mask):
        n_filtered = np.sum(~valid_bins_mask)
        warnings.warn(
            f"{n_filtered} spike(s) fall outside environment bounds and will be filtered",
            UserWarning,
            stacklevel=2,
        )
        spike_bins = spike_bins[valid_bins_mask]

    # Guard clause: handle empty spikes after spatial filtering
    if len(spike_bins) == 0:
        field = np.zeros(env.n_bins, dtype=np.float64)
        field[occupancy < min_occupancy_seconds] = np.nan
        return field

    # Step 6: Count spikes per bin
    spike_counts = np.bincount(spike_bins, minlength=env.n_bins)

    # Step 7: Normalize by occupancy where valid
    field = np.zeros(env.n_bins, dtype=np.float64)
    valid_occupancy_mask = occupancy >= min_occupancy_seconds

    if not np.any(valid_occupancy_mask):
        # All bins have insufficient occupancy
        warnings.warn(
            f"All bins have occupancy < {min_occupancy_seconds} seconds. "
            "Returning all NaN field.",
            UserWarning,
            stacklevel=2,
        )
        field[:] = np.nan
        return field

    # Compute firing rate for valid bins
    field[valid_occupancy_mask] = (
        spike_counts[valid_occupancy_mask] / occupancy[valid_occupancy_mask]
    )

    # Step 8: Set low-occupancy bins to NaN
    field[~valid_occupancy_mask] = np.nan

    return field


def _diffusion_kde(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    bandwidth: float,
    *,
    position_bins: NDArray[np.int64] | None = None,
    dt: NDArray[np.float64] | None = None,
    occupancy_density: NDArray[np.float64] | None = None,
    kernel: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute place field using graph-based diffusion KDE.

    This method spreads spike and occupancy mass using diffusion kernels
    BEFORE normalization, which is mathematically more principled than
    binning first. Respects environment boundaries via graph connectivity.

    Algorithm:
    1. Count spikes and occupancy per bin
    2. Spread both using diffusion kernel (respects walls/boundaries)
    3. Normalize: firing_rate = spike_density / occupancy_density

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    spike_times : NDArray[np.float64]
        Timestamps of spike occurrences.
    times : NDArray[np.float64]
        Timestamps of trajectory samples.
    positions : NDArray[np.float64]
        Position trajectory.
    bandwidth : float
        Smoothing bandwidth in environment units.
    position_bins : NDArray[np.int64] | None, optional
        Precomputed bin indices for positions.
    dt : NDArray[np.float64] | None, optional
        Precomputed time intervals between position samples.
    occupancy_density : NDArray[np.float64] | None, optional
        Precomputed smoothed occupancy density.
    kernel : NDArray[np.float64] | None, optional
        Precomputed smoothing kernel matrix.
    """
    # Normalize positions to 2D
    if positions.ndim == 1:
        positions = positions[:, np.newaxis]

    # Get diffusion kernel (respects boundaries via graph)
    if kernel is None:
        kernel = cast("EnvironmentProtocol", env).compute_kernel(
            bandwidth, mode="density", cache=True
        )

    # === SPIKE DENSITY ===
    # Filter spikes to valid time range
    time_min, time_max = times[0], times[-1]
    valid_spike_mask = (spike_times >= time_min) & (spike_times <= time_max)
    spike_times_valid = spike_times[valid_spike_mask]

    if len(spike_times_valid) > 0:
        # Interpolate spike positions
        spike_positions = _interpolate_spike_positions(
            spike_times_valid, times, positions
        )

        # Map spikes to bins
        spike_bins = env.bin_at(spike_positions)
        spike_bins = spike_bins[spike_bins >= 0]  # Remove out-of-bounds

        # Count spikes per bin
        spike_counts = np.bincount(spike_bins, minlength=env.n_bins).astype(np.float64)
    else:
        spike_counts = np.zeros(env.n_bins, dtype=np.float64)

    # Spread spikes using diffusion kernel
    spike_density = kernel @ spike_counts

    # === OCCUPANCY DENSITY ===
    if occupancy_density is None:
        # Map trajectory to bins (use precomputed if provided)
        traj_bins = env.bin_at(positions) if position_bins is None else position_bins
        valid_traj_mask = traj_bins >= 0
        traj_bins_valid = traj_bins[valid_traj_mask]

        # Compute time spent per sample (use precomputed if provided)
        dt_computed = np.diff(times, prepend=times[0]) if dt is None else dt
        dt_valid = dt_computed[valid_traj_mask]

        # Count occupancy per bin (weighted by dt)
        occupancy_counts = np.bincount(
            traj_bins_valid, weights=dt_valid, minlength=env.n_bins
        )

        # Spread occupancy using diffusion kernel
        occupancy_density = kernel @ occupancy_counts

    # === NORMALIZE ===
    firing_rate = np.zeros(env.n_bins, dtype=np.float64)
    valid_mask = occupancy_density > 0
    firing_rate[valid_mask] = spike_density[valid_mask] / occupancy_density[valid_mask]
    firing_rate[~valid_mask] = np.nan

    return firing_rate


def _gaussian_kde(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    bandwidth: float,
    *,
    position_bins: NDArray[np.int64] | None = None,
    dt: NDArray[np.float64] | None = None,
    occupancy_density: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute place field using standard Gaussian KDE.

    This method uses Euclidean distance for the kernel, ignoring boundaries.
    Appropriate for open fields, but not for mazes/tracks where spikes can
    "bleed through" walls.

    Algorithm:
    For each bin center:
        spike_density = sum of Gaussian kernels centered at each spike
        occupancy_density = integral of Gaussian over trajectory
        firing_rate = spike_density / occupancy_density

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    spike_times : NDArray[np.float64]
        Timestamps of spike occurrences.
    times : NDArray[np.float64]
        Timestamps of trajectory samples.
    positions : NDArray[np.float64]
        Position trajectory.
    bandwidth : float
        Smoothing bandwidth in environment units.
    position_bins : NDArray[np.int64] | None, optional
        Precomputed bin indices for positions. Not used by this method
        (included for API consistency).
    dt : NDArray[np.float64] | None, optional
        Precomputed time intervals between position samples.
    occupancy_density : NDArray[np.float64] | None, optional
        Precomputed occupancy density for each bin.
    """
    # Silence unused parameter warning - position_bins not used for gaussian_kde
    # as we need actual positions for Euclidean distance calculation
    _ = position_bins

    # Normalize positions to 2D
    if positions.ndim == 1:
        positions = positions[:, np.newaxis]

    # Filter and interpolate spike positions
    time_min, time_max = times[0], times[-1]
    valid_spike_mask = (spike_times >= time_min) & (spike_times <= time_max)
    spike_times_valid = spike_times[valid_spike_mask]

    # Interpolate spike positions using helper function
    spike_positions = _interpolate_spike_positions(spike_times_valid, times, positions)

    # Compute dt for trajectory (use precomputed if provided)
    dt_computed = np.diff(times, prepend=times[0]) if dt is None else dt

    # Vectorized KDE computation using broadcasting for all pairwise distances
    two_sigma_sq = 2 * bandwidth**2
    bin_centers = env.bin_centers  # shape: (n_bins, n_dims)

    # Precompute bin center squared norms once (reused for spikes and trajectory)
    bin_centers_sq_norm = np.sum(bin_centers**2, axis=1)  # shape: (n_bins,)

    # Memory-efficient chunked computation
    n_spikes = len(spike_positions)
    n_positions = len(positions)
    n_bins = env.n_bins

    # Chunk size chosen to keep memory under ~40MB per chunk
    # (n_points_chunk * n_bins * 8 bytes < 40MB)
    chunk_size = max(1, 40_000_000 // (n_bins * 8))

    # Compute spike density using chunked accumulation for memory efficiency
    spike_density = np.zeros(n_bins, dtype=np.float64)
    if n_spikes > 0:
        for start in range(0, n_spikes, chunk_size):
            end = min(start + chunk_size, n_spikes)
            chunk = spike_positions[start:end]
            # Compute squared distances for chunk: (chunk_size, n_bins)
            chunk_sq_norm = np.sum(chunk**2, axis=1, keepdims=True)
            chunk_dists_sq = (
                chunk_sq_norm + bin_centers_sq_norm - 2 * chunk @ bin_centers.T
            )
            chunk_weights = np.exp(-chunk_dists_sq / two_sigma_sq)
            spike_density += np.sum(chunk_weights, axis=0)

    # Compute occupancy density using chunked accumulation
    if occupancy_density is not None:
        occ_dens = occupancy_density
    else:
        occ_dens = np.zeros(n_bins, dtype=np.float64)
        for start in range(0, n_positions, chunk_size):
            end = min(start + chunk_size, n_positions)
            chunk = positions[start:end]
            chunk_dt = dt_computed[start:end]
            # Compute squared distances for chunk: (chunk_size, n_bins)
            chunk_sq_norm = np.sum(chunk**2, axis=1, keepdims=True)
            chunk_dists_sq = (
                chunk_sq_norm + bin_centers_sq_norm - 2 * chunk @ bin_centers.T
            )
            chunk_weights = np.exp(-chunk_dists_sq / two_sigma_sq)
            occ_dens += np.sum(chunk_weights * chunk_dt[:, np.newaxis], axis=0)

    # Compute firing rate with safe division
    firing_rate = np.where(occ_dens > 0, spike_density / occ_dens, np.nan)

    return firing_rate


def _binned(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    bandwidth: float,
    min_occupancy_seconds: float,
) -> NDArray[np.float64]:
    """Compute place field using binned approach (legacy method).

    This method bins spikes, normalizes by occupancy, then smooths.
    Has discretization artifacts but is fast.

    Algorithm:
    1. Bin spikes and trajectory
    2. Normalize: firing_rate = spike_counts / occupancy
    3. Smooth with diffusion kernel
    """
    # Use existing _binned_rate_map function
    field = _binned_rate_map(
        env,
        spike_times,
        times,
        positions,
        min_occupancy_seconds=min_occupancy_seconds,
    )

    # Smooth if bandwidth > 0
    if bandwidth <= 0:
        return field

    # Handle NaN values with proper weight normalization
    nan_mask = np.isnan(field)

    if not np.any(nan_mask):
        # No NaN - smooth directly
        return cast("EnvironmentProtocol", env).smooth(field, bandwidth=bandwidth)

    # Has NaN - use NaN-aware smoothing
    weights = np.ones_like(field)
    weights[nan_mask] = 0.0

    field_filled = field.copy()
    field_filled[nan_mask] = 0.0

    # Smooth both field and weights
    field_smoothed = cast("EnvironmentProtocol", env).smooth(
        field_filled, bandwidth=bandwidth
    )
    weights_smoothed = cast("EnvironmentProtocol", env).smooth(
        weights, bandwidth=bandwidth
    )

    # Normalize
    field_normalized = np.zeros_like(field_smoothed)
    valid_mask = weights_smoothed > 0
    field_normalized[valid_mask] = (
        field_smoothed[valid_mask] / weights_smoothed[valid_mask]
    )
    field_normalized[~valid_mask] = np.nan

    return field_normalized


def compute_place_field(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    smoothing_method: Literal[
        "diffusion_kde", "gaussian_kde", "binned"
    ] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy_seconds: float = 0.0,
    position_bins: NDArray[np.int64] | None = None,
    dt: NDArray[np.float64] | None = None,
    occupancy_density: NDArray[np.float64] | None = None,
    kernel: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute place field from spike train with multiple estimation methods.

    Converts a spike train to a spatial firing rate map using one of three
    kernel density estimation approaches. The default diffusion KDE method
    respects environment boundaries and is mathematically principled.

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Timestamps of spike occurrences (seconds).
    times : NDArray[np.float64], shape (n_timepoints,)
        Timestamps of trajectory samples (seconds).
    positions : NDArray[np.float64], shape (n_timepoints, n_dims) or (n_timepoints,)
        Position trajectory. For 1D, can be shape (n_timepoints,) or (n_timepoints, 1).
    smoothing_method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Estimation method:

        - **"diffusion_kde"** (default, recommended):
          Graph-based diffusion kernel respecting environment boundaries.
          Spreads spike and occupancy mass using graph Laplacian before
          normalization. Most accurate for irregular geometries (mazes,
          tracks, arenas with walls). Uses the correct mathematical order:
          spread → normalize.

        - **"gaussian_kde"**:
          Standard Gaussian kernel density estimation with Euclidean distance.
          Faster than diffusion_kde but ignores boundaries/walls. Good for
          open fields only. Spikes can "bleed through" walls in mazes.

        - **"binned"**:
          Legacy method - bin spikes, normalize by occupancy, then smooth.
          Fastest but has discretization artifacts. Uses the order:
          bin → normalize → smooth. Useful for quick visualization.

    bandwidth : float, default=5.0
        Smoothing bandwidth in environment units (e.g., cm).
        Controls spatial scale of smoothing kernel.
        Typical values: 5-10 cm for small arenas, 20-50 cm for large fields.

    min_occupancy_seconds : float, default=0.0
        Minimum occupancy threshold in seconds (only used for method="binned").
        Bins with less occupancy are set to NaN. For diffusion_kde and
        gaussian_kde, this parameter is ignored as low occupancy is handled
        naturally by the normalization. For binned method, 0.5 seconds is
        typical for place field analysis.
    position_bins : NDArray[np.int64] | None, default=None
        Precomputed bin indices for positions. If provided, skips the
        `env.bin_at(positions)` call. Useful when computing multiple place
        fields from the same trajectory. Shape: (n_timepoints,).
    dt : NDArray[np.float64] | None, default=None
        Precomputed time intervals between position samples. If provided,
        skips the `np.diff(times, prepend=times[0])` call. Shape: (n_timepoints,).
    occupancy_density : NDArray[np.float64] | None, default=None
        Precomputed smoothed occupancy density. If provided, skips occupancy
        computation entirely. Useful when computing multiple place fields
        from the same trajectory. Shape: (n_bins,). Only used with
        method="diffusion_kde" or "gaussian_kde".
    kernel : NDArray[np.float64] | None, default=None
        Precomputed smoothing kernel matrix. If provided, skips the
        `env.compute_kernel()` call. Shape: (n_bins, n_bins). Only used
        with method="diffusion_kde".

    Returns
    -------
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Firing rate field (spikes/second) for each spatial bin.
        Bins with zero occupancy are set to NaN.

    Raises
    ------
    ValueError
        If method is not one of {"diffusion_kde", "gaussian_kde", "binned"}.
    ValueError
        If bandwidth is not positive.
    ValueError
        If times and positions have different lengths.

    See Also
    --------
    _binned_rate_map : Lower-level binning function (used by method="binned").
    Environment.compute_kernel : Compute diffusion kernel.
    Environment.smooth : Smooth spatial fields.

    Notes
    -----
    **Method Comparison:**

    ========================================  ==============  =============  =================
    Method                                    Boundary-Aware  Artifacts      Speed
    ========================================  ==============  =============  =================
    diffusion_kde (recommended)               Yes (graph)     None           Medium
    gaussian_kde                              No              Bleeds walls   Slow (O(n*m))
    binned                                    Yes (graph)     Discretization Fast
    ========================================  ==============  =============  =================

    **Mathematical Difference:**

    - **diffusion_kde** and **gaussian_kde**: Spread → Normalize
      (mathematically correct KDE)
    - **binned**: Bin → Normalize → Smooth
      (has discretization artifacts)

    **When to Use Each:**

    - **diffusion_kde**: Default for all analyses. Best for mazes, tracks,
      irregular geometries. Proper KDE with boundary awareness.
    - **gaussian_kde**: Open fields only. Use when boundaries don't matter
      and you want standard Euclidean KDE.
    - **binned**: Quick visualization, legacy code, or when you need exact
      compatibility with older analyses that used binning.

    Examples
    --------
    Compute place field with default diffusion KDE:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.place import compute_place_field
    >>>
    >>> # Create trajectory
    >>> positions = np.random.uniform(0, 100, (1000, 2))
    >>> times = np.linspace(0, 100, 1000)
    >>> spike_times = np.random.uniform(0, 100, 50)
    >>>
    >>> # Create environment
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>>
    >>> # Compute place field (default: diffusion_kde)
    >>> firing_rate = compute_place_field(
    ...     env, spike_times, times, positions, bandwidth=8.0
    ... )
    >>> firing_rate.shape == (env.n_bins,)
    True

    Compare all three methods:

    >>> rate_diffusion = compute_place_field(
    ...     env,
    ...     spike_times,
    ...     times,
    ...     positions,
    ...     smoothing_method="diffusion_kde",
    ...     bandwidth=8.0,
    ... )
    >>> rate_gaussian = compute_place_field(
    ...     env,
    ...     spike_times,
    ...     times,
    ...     positions,
    ...     smoothing_method="gaussian_kde",
    ...     bandwidth=8.0,
    ... )
    >>> rate_binned = compute_place_field(
    ...     env, spike_times, times, positions, smoothing_method="binned", bandwidth=8.0
    ... )
    """
    # Validate smoothing_method
    valid_methods = {"diffusion_kde", "gaussian_kde", "binned"}
    if smoothing_method not in valid_methods:
        raise ValueError(
            f"smoothing_method must be one of {valid_methods}, got '{smoothing_method}'"
        )

    # Validate bandwidth
    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")

    # Validate inputs (times and positions length)
    if len(times) != len(positions):
        raise ValueError(
            f"times and positions must have same length, got {len(times)} and {len(positions)}"
        )

    # Validate precomputed parameter shapes
    if position_bins is not None and len(position_bins) != len(positions):
        raise ValueError(
            f"position_bins must have same length as positions, "
            f"got {len(position_bins)} and {len(positions)}"
        )

    if dt is not None and len(dt) != len(times):
        raise ValueError(
            f"dt must have same length as times, got {len(dt)} and {len(times)}"
        )

    if occupancy_density is not None and len(occupancy_density) != env.n_bins:
        raise ValueError(
            f"occupancy_density must have length n_bins={env.n_bins}, "
            f"got {len(occupancy_density)}"
        )

    if kernel is not None and kernel.shape != (env.n_bins, env.n_bins):
        raise ValueError(
            f"kernel must have shape (n_bins, n_bins)=({env.n_bins}, {env.n_bins}), "
            f"got {kernel.shape}"
        )

    # Dispatch to appropriate backend
    match smoothing_method:
        case "diffusion_kde":
            return _diffusion_kde(
                env,
                spike_times,
                times,
                positions,
                bandwidth,
                position_bins=position_bins,
                dt=dt,
                occupancy_density=occupancy_density,
                kernel=kernel,
            )
        case "gaussian_kde":
            return _gaussian_kde(
                env,
                spike_times,
                times,
                positions,
                bandwidth,
                position_bins=position_bins,
                dt=dt,
                occupancy_density=occupancy_density,
            )
        case "binned":
            return _binned(
                env, spike_times, times, positions, bandwidth, min_occupancy_seconds
            )
        case _:
            # Should never reach here due to validation above
            raise ValueError(f"Unknown smoothing_method: {smoothing_method}")
