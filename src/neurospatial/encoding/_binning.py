"""Binning layer for spatial encoding.

This module converts spike trains and trajectories into discrete spike counts
and occupancy arrays that can be processed by the smoothing layer.

The functions in this module handle:
1. Spike interpolation from continuous spike times to bin positions
2. Occupancy computation from continuous trajectory to time-per-bin
3. Batch processing of multiple neurons with joblib parallelization

Output shapes:
- Spike counts (single neuron): (n_bins,)
- Spike counts (batch): (n_neurons, n_bins)
- Occupancy: (n_bins,) - always shared across neurons

The binning layer is intentionally separated from smoothing to allow:
- Reusing occupancy across multiple neurons
- Precomputing position bins for efficiency
- Future JAX implementations to leverage different parallelization strategies
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.environment._protocols import EnvironmentProtocol

__all__ = [
    "bin_spike_train",
    "bin_spike_trains",
    "compute_occupancy",
]


_DROP_WARN_THRESHOLD = 0.5  # warn when fraction dropped exceeds this


def bin_spike_train(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    context: str = "bin_spike_train",
    warn_on_drop: bool = True,
) -> NDArray[np.float64]:
    """Bin spike train into spatial bins.

    Converts continuous spike times to spike counts per spatial bin by
    interpolating spike positions from the trajectory and counting spikes
    in each bin.

    Parameters
    ----------
    env : Environment
        The spatial environment defining bin structure.
    spike_times : ndarray, shape (n_spikes,)
        Times of spike events in seconds.
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Position coordinates at each time sample.
    context : str, optional
        Label used in error messages to identify the calling function.
    warn_on_drop : bool, default=True
        If ``True`` (the default), emit a ``UserWarning`` when a large
        fraction of spikes are silently dropped — either because they
        fall outside the position time window or because they map to
        inactive/out-of-environment bins.  A warning is always emitted
        when **all** spikes are dropped (regardless of threshold).
        Set to ``False`` to suppress all drop-related warnings (e.g.
        when the calling batch function will issue its own aggregate
        warning).

    Returns
    -------
    ndarray, shape (n_bins,)
        Number of spikes in each spatial bin (float64 for compatibility
        with smoothing operations).

    Raises
    ------
    ValueError
        If times and positions have different lengths.

    Notes
    -----
    Spikes are interpolated to positions using linear interpolation of
    spike times onto the trajectory. Spikes outside the trajectory time
    range are excluded. Spikes that fall in invalid bins (outside the
    environment) are excluded from counts.

    The default ``warn_on_drop=True`` guards against common unit
    mismatches (e.g. spike_times in milliseconds while times is in
    seconds), which would silently produce a near-empty firing field.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._binning import bin_spike_train

    >>> # Create environment
    >>> positions = np.random.rand(1000, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=10.0)

    >>> # Create trajectory and spikes
    >>> times = np.linspace(0, 10, 1000)
    >>> trajectory = np.random.rand(1000, 2) * 100
    >>> spike_times = np.array([1.0, 2.5, 4.0, 7.5])

    >>> # Bin spikes
    >>> spike_counts = bin_spike_train(env, spike_times, times, trajectory)
    >>> spike_counts.shape[0] == env.n_bins
    True
    """
    spike_times = np.asarray(spike_times, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    if len(times) != len(positions):
        raise ValueError(
            f"in {context}: times length ({len(times)}) must match "
            f"positions length ({len(positions)})"
        )

    n_bins = env.n_bins
    spike_counts = np.zeros(n_bins, dtype=np.float64)

    if len(spike_times) == 0:
        return spike_counts

    n_total = len(spike_times)
    t_min, t_max = times.min(), times.max()
    valid_time_mask = (spike_times >= t_min) & (spike_times <= t_max)
    spike_times_valid = spike_times[valid_time_mask]
    n_time_dropped = n_total - len(spike_times_valid)

    # Warn on time-window drops at this boundary (before returning early)
    if warn_on_drop and n_total > 0 and n_time_dropped > 0:
        frac_dropped = n_time_dropped / n_total
        if frac_dropped > _DROP_WARN_THRESHOLD or n_time_dropped == n_total:
            warnings.warn(
                f"{n_time_dropped}/{n_total} spike_times "
                f"({100 * frac_dropped:.0f}%) fell outside the position time "
                f"window [{t_min:.6g}, {t_max:.6g}]; "
                f"spike_times.min()={spike_times.min():.6g} "
                f"spike_times.max()={spike_times.max():.6g}. "
                f"Check that spike_times and times share units (both seconds). "
                f"Dropped spikes do not contribute.",
                UserWarning,
                stacklevel=2,
            )

    if len(spike_times_valid) == 0:
        return spike_counts

    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)

    # Linearly interpolate the spike position from the trajectory in each
    # dimension, then map to a bin. Using the most recent trajectory frame
    # (snapshot) instead of interp would shift spikes that fall between
    # samples to the previous bin under fast movement / sparse sampling.
    n_dims = positions.shape[1]
    spike_positions = np.empty((len(spike_times_valid), n_dims), dtype=np.float64)
    for d in range(n_dims):
        spike_positions[:, d] = np.interp(spike_times_valid, times, positions[:, d])

    spike_bins = env.bin_at(spike_positions)
    valid_bins = spike_bins[spike_bins >= 0]
    n_bin_dropped = len(spike_times_valid) - len(valid_bins)

    # Warn on inactive-bin drops
    if warn_on_drop and len(spike_times_valid) > 0 and n_bin_dropped > 0:
        n_after_time = len(spike_times_valid)
        frac_bin_dropped = n_bin_dropped / n_after_time
        if frac_bin_dropped > _DROP_WARN_THRESHOLD or n_bin_dropped == n_after_time:
            warnings.warn(
                f"{n_bin_dropped}/{n_after_time} spike_times "
                f"({100 * frac_bin_dropped:.0f}%) mapped to positions outside "
                f"the active environment bins (bin index -1). "
                f"Check that spike_times and times share units (both seconds) "
                f"and that positions are in the same coordinate frame as the "
                f"environment. Dropped spikes do not contribute.",
                UserWarning,
                stacklevel=2,
            )

    if len(valid_bins) == 0:
        return spike_counts

    return np.bincount(valid_bins, minlength=n_bins).astype(np.float64)


def compute_occupancy(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    context: str = "compute_occupancy",
) -> NDArray[np.float64]:
    """Compute occupancy (time spent in each bin).

    Computes the total time spent in each spatial bin by accumulating
    time intervals from the trajectory.

    Parameters
    ----------
    env : Environment
        The spatial environment defining bin structure.
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Position coordinates at each time sample.

    Returns
    -------
    ndarray, shape (n_bins,)
        Time in seconds spent in each spatial bin.

    Raises
    ------
    ValueError
        If times and positions have different lengths.
        If positions have wrong number of dimensions.

    Notes
    -----
    Delegates to Environment.occupancy() which handles:
    - Time interval allocation to bins
    - Speed filtering (if configured)
    - Gap handling
    - Kernel smoothing (if configured)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._binning import compute_occupancy

    >>> # Create environment
    >>> positions = np.random.rand(1000, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=10.0)

    >>> # Create trajectory
    >>> times = np.linspace(0, 10, 1000)
    >>> trajectory = np.random.rand(1000, 2) * 100

    >>> # Compute occupancy
    >>> occupancy = compute_occupancy(env, times, trajectory)
    >>> occupancy.shape[0] == env.n_bins
    True
    """
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    # Validate input shapes
    if len(times) != len(positions):
        raise ValueError(
            f"in {context}: times length ({len(times)}) must match "
            f"positions length ({len(positions)})"
        )

    # Handle 1D positions
    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)

    # Check position dimensions match environment
    if positions.shape[1] != env.n_dims:
        raise ValueError(
            f"in {context}: positions have {positions.shape[1]} dimensions "
            f"but environment has {env.n_dims} dimensions"
        )

    # Delegate to Environment.occupancy() which handles all the complexity
    occupancy = cast("EnvironmentProtocol", env).occupancy(
        times, positions, return_seconds=True
    )

    return occupancy.astype(np.float64)


def bin_spike_trains(
    env: Environment,
    spike_times: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    n_jobs: int = 1,
    warn_on_drop: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bin multiple spike trains into spatial bins.

    Batch version of bin_spike_train that efficiently processes multiple
    neurons. Precomputes shared quantities (position bins, occupancy) and
    optionally parallelizes spike counting with joblib.

    Parameters
    ----------
    env : Environment
        The spatial environment defining bin structure.
    spike_times : sequence of arrays or 2D array
        Spike times for each neuron. Can be:
        - List/tuple of 1D arrays (one per neuron)
        - 2D array shape (n_neurons, max_spikes) with NaN padding
        Input is normalized via normalize_spike_times().
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Position coordinates at each time sample.
    n_jobs : int, default=1
        Number of parallel jobs for spike counting. Use -1 for all CPUs.
        1 means sequential processing (no parallelization overhead).
    warn_on_drop : bool, default=True
        If ``True`` (the default), emit a single ``UserWarning`` (per drop
        cause) when a large fraction of spikes are silently dropped across
        all neurons.  The warning is computed in the main process from
        aggregate statistics, so it fires exactly once even when
        ``n_jobs != 1`` (joblib worker warnings are commonly swallowed).
        Set to ``False`` to suppress all drop-related warnings.

    Returns
    -------
    spike_counts : ndarray, shape (n_neurons, n_bins)
        Number of spikes in each spatial bin for each neuron.
    occupancy : ndarray, shape (n_bins,)
        Time in seconds spent in each spatial bin (shared across neurons).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._binning import bin_spike_trains

    >>> # Create environment
    >>> positions = np.random.rand(1000, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=10.0)

    >>> # Create trajectory and spikes for 3 neurons
    >>> times = np.linspace(0, 10, 1000)
    >>> trajectory = np.random.rand(1000, 2) * 100
    >>> spike_times = [
    ...     np.array([1.0, 2.5, 4.0]),  # Neuron 0
    ...     np.array([0.5, 1.5, 2.5, 3.5]),  # Neuron 1
    ...     np.array([5.0]),  # Neuron 2
    ... ]

    >>> # Bin spikes
    >>> spike_counts, occupancy = bin_spike_trains(
    ...     env, spike_times, times, trajectory, n_jobs=2
    ... )
    >>> spike_counts.shape[0] == 3  # 3 neurons
    True
    >>> spike_counts.shape[1] == env.n_bins
    True
    >>> occupancy.shape[0] == env.n_bins
    True

    See Also
    --------
    bin_spike_train : Single-neuron version
    compute_occupancy : Compute occupancy only
    normalize_spike_times : Input format normalization
    """
    from neurospatial.encoding._spikes import normalize_spike_times

    # Normalize spike times to canonical list-of-arrays format
    spike_times_list = normalize_spike_times(spike_times)
    n_neurons = len(spike_times_list)

    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)

    # Occupancy is independent of which neuron we're binning, so compute once.
    # Spike binning itself depends on per-neuron spike_times (interpolated to
    # spike positions), so it stays inside the per-neuron loop.
    occupancy = compute_occupancy(env, times, positions)

    # Always call leaf with warn_on_drop=False: we aggregate and warn ONCE
    # here in the main process (see below) instead of once per neuron.
    # This is critical for n_jobs != 1 where joblib worker warnings are
    # swallowed, but we keep it consistent for n_jobs == 1 too (no duplicates).
    if n_jobs == 1:
        spike_counts = np.zeros((n_neurons, env.n_bins), dtype=np.float64)
        for i, spikes in enumerate(spike_times_list):
            spike_counts[i] = bin_spike_train(
                env, spikes, times, positions, warn_on_drop=False
            )
    else:
        from joblib import Parallel, delayed

        def _process_neuron(spikes: NDArray[np.float64]) -> NDArray[np.float64]:
            # warn_on_drop=False: do NOT warn from workers (warnings swallowed)
            return bin_spike_train(env, spikes, times, positions, warn_on_drop=False)

        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_neuron)(spikes) for spikes in spike_times_list
        )
        spike_counts = np.array(results, dtype=np.float64)

    # Aggregate drop statistics across all neurons and warn ONCE in main process.
    if warn_on_drop and n_neurons > 0:
        t_min, t_max = times.min(), times.max()

        total_spikes = 0
        total_time_dropped = 0
        total_after_time = 0
        total_bin_dropped = 0

        for spikes in spike_times_list:
            spikes_arr = np.asarray(spikes, dtype=np.float64)
            n = len(spikes_arr)
            if n == 0:
                continue
            total_spikes += n
            mask = (spikes_arr >= t_min) & (spikes_arr <= t_max)
            n_valid_time = int(mask.sum())
            n_time_drop = n - n_valid_time
            total_time_dropped += n_time_drop
            # For bin-drop stats we need the spike positions
            if n_valid_time > 0:
                total_after_time += n_valid_time
                spikes_valid = spikes_arr[mask]
                pos2d = positions.reshape(-1, 1) if positions.ndim == 1 else positions
                n_dims = pos2d.shape[1]
                spike_positions = np.empty((n_valid_time, n_dims), dtype=np.float64)
                for d in range(n_dims):
                    spike_positions[:, d] = np.interp(spikes_valid, times, pos2d[:, d])
                spike_bins = env.bin_at(spike_positions)
                n_bin_drop = int((spike_bins < 0).sum())
                total_bin_dropped += n_bin_drop

        # Warn for time-window drops
        if total_spikes > 0 and total_time_dropped > 0:
            frac = total_time_dropped / total_spikes
            if frac > _DROP_WARN_THRESHOLD or total_time_dropped == total_spikes:
                all_spikes_cat = np.concatenate(
                    [
                        np.asarray(s, dtype=np.float64)
                        for s in spike_times_list
                        if len(s) > 0
                    ]
                )
                warnings.warn(
                    f"{total_time_dropped}/{total_spikes} spike_times "
                    f"({100 * frac:.0f}%) across all neurons fell outside the "
                    f"position time window [{t_min:.6g}, {t_max:.6g}]; "
                    f"spike_times.min()={all_spikes_cat.min():.6g} "
                    f"spike_times.max()={all_spikes_cat.max():.6g}. "
                    f"Check that spike_times and times share units (both "
                    f"seconds). Dropped spikes do not contribute.",
                    UserWarning,
                    stacklevel=2,
                )

        # Warn for inactive-bin drops
        if total_after_time > 0 and total_bin_dropped > 0:
            frac_bin = total_bin_dropped / total_after_time
            if frac_bin > _DROP_WARN_THRESHOLD or total_bin_dropped == total_after_time:
                warnings.warn(
                    f"{total_bin_dropped}/{total_after_time} spike_times "
                    f"({100 * frac_bin:.0f}%) across all neurons mapped to "
                    f"positions outside the active environment bins (bin index "
                    f"-1). Check that spike_times and times share units (both "
                    f"seconds) and that positions are in the same coordinate "
                    f"frame as the environment. Dropped spikes do not contribute.",
                    UserWarning,
                    stacklevel=2,
                )

    return spike_counts, occupancy
