"""Binning layer for view encoding (spatial view cells).

This module converts spike trains and gaze data into discrete spike counts
and view occupancy arrays that can be processed by the smoothing layer.

The key difference from spatial binning (_binning.py) is that:
- Position binning: bins spikes by *where the animal was*
- View binning: bins spikes by *where the animal was looking*

The functions in this module handle:
1. Computation of viewed locations from positions, headings, and gaze model
2. View occupancy computation (time spent *viewing* each bin)
3. Spike binning based on viewed location at spike time
4. Batch processing of multiple neurons with joblib parallelization

Output shapes:
- Spike counts (single neuron): (n_bins,)
- Spike counts (batch): (n_neurons, n_bins)
- View occupancy: (n_bins,) - always shared across neurons

Gaze models supported:
- "fixed_distance": Point at fixed distance in gaze direction (default)
- "ray_cast": Intersection with environment boundary
- "boundary": Nearest boundary point in gaze direction
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial.ops.visibility import compute_viewed_location

if TYPE_CHECKING:
    from neurospatial.environment import Environment

__all__ = [
    "bin_view_spike_train",
    "bin_view_spike_trains",
    "compute_view_occupancy",
]


def compute_view_occupancy(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    gaze_model: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance",
    view_distance: float = 10.0,
    gaze_offsets: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute view occupancy (time spent viewing each bin).

    Computes the total time spent *viewing* each spatial bin by computing
    viewed locations from positions and headings, then accumulating time
    intervals per viewed bin.

    This differs from position occupancy which measures time spent *at* each bin.

    Parameters
    ----------
    env : Environment
        The spatial environment defining bin structure.
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, 2)
        Position coordinates at each time sample.
    headings : ndarray, shape (n_samples,)
        Head direction at each time sample (radians, 0=East).
    gaze_model : {"fixed_distance", "ray_cast", "boundary"}, default="fixed_distance"
        Method for computing viewed location:
        - "fixed_distance": Point at fixed distance in gaze direction
        - "ray_cast": Intersection with environment boundary
        - "boundary": Nearest boundary point in gaze direction
    view_distance : float, default=10.0
        Distance for fixed_distance gaze model (environment units).
    gaze_offsets : ndarray, shape (n_samples,), optional
        Offset from heading to gaze direction (e.g., from eye tracking).
        If None, gaze is aligned with heading.

    Returns
    -------
    ndarray, shape (n_bins,)
        Time in seconds spent viewing each spatial bin.

    Raises
    ------
    ValueError
        If input arrays have mismatched lengths.
        If gaze_model is invalid.
        If fewer than 3 samples provided.

    Notes
    -----
    View occupancy is computed by:
    1. Computing viewed locations for each time sample using the gaze model
    2. Mapping viewed locations to spatial bins
    3. Accumulating time intervals (dt) per bin using np.add.at

    Viewed locations outside the environment (invalid bins) do not contribute
    to occupancy. This means total view occupancy may be less than recording
    duration if many views fall outside the environment.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._view_binning import compute_view_occupancy

    >>> # Create environment
    >>> positions = np.random.rand(100, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=10.0)

    >>> # Create trajectory with positions and headings
    >>> times = np.linspace(0, 10, 1000)
    >>> trajectory = np.column_stack(
    ...     [
    ...         50 + np.cumsum(np.random.randn(1000) * 0.5),  # x
    ...         50 + np.cumsum(np.random.randn(1000) * 0.5),  # y
    ...     ]
    ... )
    >>> trajectory = np.clip(trajectory, 20, 80)
    >>> headings = np.random.uniform(0, 2 * np.pi, 1000)

    >>> # Compute view occupancy
    >>> view_occ = compute_view_occupancy(env, times, trajectory, headings)
    >>> view_occ.shape[0] == env.n_bins
    True
    """
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)

    # Validate input shapes
    n_samples = len(times)
    if len(positions) != n_samples:
        raise ValueError(
            f"times length ({n_samples}) must match positions length ({len(positions)})"
        )
    if len(headings) != n_samples:
        raise ValueError(
            f"times length ({n_samples}) must match headings length ({len(headings)})"
        )

    # Validate minimum samples
    if n_samples < 2:
        raise ValueError(
            f"At least 2 samples required for view occupancy computation, got {n_samples}"
        )

    # Validate times are monotonically non-decreasing
    if n_samples > 1:
        time_diffs = np.diff(times)
        if np.any(time_diffs < 0):
            decreasing_indices = np.where(time_diffs < 0)[0]
            raise ValueError(
                "times must be monotonically non-decreasing (sorted). "
                f"Found {len(decreasing_indices)} decreasing interval(s) at "
                f"indices: {decreasing_indices.tolist()[:5]}"
                + (" ..." if len(decreasing_indices) > 5 else "")
            )

    # Validate gaze_model
    valid_gaze_models = {"fixed_distance", "ray_cast", "boundary"}
    if gaze_model not in valid_gaze_models:
        raise ValueError(
            f"Invalid gaze_model: '{gaze_model}'. "
            f"Must be one of {sorted(valid_gaze_models)}"
        )

    # Validate gaze_offsets if provided
    if gaze_offsets is not None:
        gaze_offsets = np.asarray(gaze_offsets, dtype=np.float64)
        if len(gaze_offsets) != n_samples:
            raise ValueError(
                f"gaze_offsets length ({len(gaze_offsets)}) must match "
                f"times length ({n_samples})"
            )

    # Compute per-sample time deltas (n-1 intervals for n samples)
    # Each interval[i] represents the time from sample[i] to sample[i+1]
    dt = np.diff(times)

    # Compute viewed locations for all timepoints
    viewed_locations = compute_viewed_location(
        positions,
        headings,
        method=gaze_model,
        view_distance=view_distance,
        gaze_offsets=gaze_offsets,
        env=env if gaze_model in ("ray_cast", "boundary") else None,
    )

    # Identify valid viewed locations (finite and inside environment)
    valid_view_mask = np.all(np.isfinite(viewed_locations), axis=1)

    # Map viewed locations to bins
    view_bins = np.full(n_samples, -1, dtype=np.intp)
    if np.any(valid_view_mask):
        valid_viewed = viewed_locations[valid_view_mask]
        valid_bins = env.bin_at(valid_viewed)
        view_bins[valid_view_mask] = valid_bins
        # Update valid mask to only include views inside environment
        valid_view_mask = view_bins >= 0

    # Compute view occupancy (time spent viewing each bin)
    # We have n_samples positions and n_samples-1 intervals.
    # Each interval[i] is assigned to the bin at position[i] (start of interval).
    # This matches Environment.occupancy() behavior (time_allocation='start').
    view_occupancy = np.zeros(env.n_bins, dtype=np.float64)

    # Only consider positions that start valid intervals (all except last)
    # interval_bins[i] is the bin viewed during interval[i]
    interval_bins = view_bins[:-1]  # Exclude last position (no following interval)
    valid_interval_mask = interval_bins >= 0

    # Accumulate time per bin
    valid_bins = interval_bins[valid_interval_mask]
    valid_dt = dt[valid_interval_mask]
    np.add.at(view_occupancy, valid_bins, valid_dt)

    return view_occupancy


def bin_view_spike_train(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    gaze_model: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance",
    view_distance: float = 10.0,
    gaze_offsets: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Bin spike train by viewed location.

    Converts continuous spike times to spike counts per spatial bin based on
    where the animal was *looking* at each spike time, not where the animal
    was located.

    Parameters
    ----------
    env : Environment
        The spatial environment defining bin structure.
    spike_times : ndarray, shape (n_spikes,)
        Times of spike events in seconds.
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, 2)
        Position coordinates at each time sample.
    headings : ndarray, shape (n_samples,)
        Head direction at each time sample (radians, 0=East).
    gaze_model : {"fixed_distance", "ray_cast", "boundary"}, default="fixed_distance"
        Method for computing viewed location.
    view_distance : float, default=10.0
        Distance for fixed_distance gaze model (environment units).
    gaze_offsets : ndarray, shape (n_samples,), optional
        Offset from heading to gaze direction.

    Returns
    -------
    ndarray, shape (n_bins,)
        Number of spikes in each spatial bin based on viewed location
        (float64 for compatibility with smoothing operations).

    Notes
    -----
    Spikes are assigned to bins based on the viewed location at the nearest
    behavioral frame to each spike time. This uses nearest-neighbor lookup
    (via np.searchsorted) rather than interpolation because gaze direction
    cannot be meaningfully interpolated.

    Spikes where the viewed location is outside the environment or invalid
    are excluded from counts.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._view_binning import bin_view_spike_train

    >>> # Create environment
    >>> positions = np.random.rand(100, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=10.0)

    >>> # Create trajectory
    >>> times = np.linspace(0, 10, 1000)
    >>> trajectory = np.column_stack(
    ...     [
    ...         50 + np.cumsum(np.random.randn(1000) * 0.5),
    ...         50 + np.cumsum(np.random.randn(1000) * 0.5),
    ...     ]
    ... )
    >>> trajectory = np.clip(trajectory, 20, 80)
    >>> headings = np.random.uniform(0, 2 * np.pi, 1000)
    >>> spike_times = np.array([1.0, 2.5, 4.0, 7.5])

    >>> # Bin spikes by viewed location
    >>> spike_counts = bin_view_spike_train(
    ...     env, spike_times, times, trajectory, headings
    ... )
    >>> spike_counts.shape[0] == env.n_bins
    True
    """
    spike_times = np.asarray(spike_times, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)

    n_samples = len(times)
    n_bins = env.n_bins
    spike_counts = np.zeros(n_bins, dtype=np.float64)

    # Handle empty spike train
    if len(spike_times) == 0:
        return spike_counts

    # Filter spikes to valid time range
    t_min, t_max = times.min(), times.max()
    valid_time_mask = (spike_times >= t_min) & (spike_times <= t_max)
    spike_times_valid = spike_times[valid_time_mask]

    if len(spike_times_valid) == 0:
        return spike_counts

    # Compute viewed locations for all timepoints
    viewed_locations = compute_viewed_location(
        positions,
        headings,
        method=gaze_model,
        view_distance=view_distance,
        gaze_offsets=gaze_offsets,
        env=env if gaze_model in ("ray_cast", "boundary") else None,
    )

    # Map viewed locations to bins for all frames
    view_bins = np.full(n_samples, -1, dtype=np.intp)
    valid_view_mask = np.all(np.isfinite(viewed_locations), axis=1)
    if np.any(valid_view_mask):
        valid_viewed = viewed_locations[valid_view_mask]
        valid_bins = env.bin_at(valid_viewed)
        view_bins[valid_view_mask] = valid_bins

    # Find nearest behavioral frame for each spike
    spike_frame_idx = np.searchsorted(times, spike_times_valid, side="right") - 1
    spike_frame_idx = np.clip(spike_frame_idx, 0, n_samples - 1)

    # Get viewed bin at each spike time
    spike_view_bins = view_bins[spike_frame_idx]

    # Only count spikes where view was valid (inside environment)
    valid_spike_views = spike_view_bins >= 0
    valid_spike_view_bins = spike_view_bins[valid_spike_views]

    if len(valid_spike_view_bins) == 0:
        return spike_counts

    # Count spikes per viewed bin
    np.add.at(spike_counts, valid_spike_view_bins, 1.0)

    return spike_counts


def bin_view_spike_trains(
    env: Environment,
    spike_times: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    gaze_model: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance",
    view_distance: float = 10.0,
    gaze_offsets: NDArray[np.float64] | None = None,
    n_jobs: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bin multiple spike trains by viewed location.

    Batch version of bin_view_spike_train that efficiently processes multiple
    neurons. Precomputes shared quantities (viewed locations, view bins,
    view occupancy) and optionally parallelizes spike counting with joblib.

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
    positions : ndarray, shape (n_samples, 2)
        Position coordinates at each time sample.
    headings : ndarray, shape (n_samples,)
        Head direction at each time sample (radians, 0=East).
    gaze_model : {"fixed_distance", "ray_cast", "boundary"}, default="fixed_distance"
        Method for computing viewed location.
    view_distance : float, default=10.0
        Distance for fixed_distance gaze model (environment units).
    gaze_offsets : ndarray, shape (n_samples,), optional
        Offset from heading to gaze direction.
    n_jobs : int, default=1
        Number of parallel jobs for spike counting. Use -1 for all CPUs.
        1 means sequential processing (no parallelization overhead).

    Returns
    -------
    spike_counts : ndarray, shape (n_neurons, n_bins)
        Number of spikes in each spatial bin for each neuron,
        based on viewed location at spike time.
    view_occupancy : ndarray, shape (n_bins,)
        Time in seconds spent viewing each spatial bin (shared across neurons).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._view_binning import bin_view_spike_trains

    >>> # Create environment
    >>> positions = np.random.rand(100, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=10.0)

    >>> # Create trajectory
    >>> times = np.linspace(0, 10, 1000)
    >>> trajectory = np.column_stack(
    ...     [
    ...         50 + np.cumsum(np.random.randn(1000) * 0.5),
    ...         50 + np.cumsum(np.random.randn(1000) * 0.5),
    ...     ]
    ... )
    >>> trajectory = np.clip(trajectory, 20, 80)
    >>> headings = np.random.uniform(0, 2 * np.pi, 1000)
    >>> spike_times = [
    ...     np.array([1.0, 2.5, 4.0]),  # Neuron 0
    ...     np.array([0.5, 1.5, 2.5, 3.5]),  # Neuron 1
    ...     np.array([5.0]),  # Neuron 2
    ... ]

    >>> # Bin spikes by viewed location
    >>> spike_counts, view_occ = bin_view_spike_trains(
    ...     env, spike_times, times, trajectory, headings, n_jobs=2
    ... )
    >>> spike_counts.shape == (3, env.n_bins)
    True
    >>> view_occ.shape == (env.n_bins,)
    True

    See Also
    --------
    bin_view_spike_train : Single-neuron version
    compute_view_occupancy : Compute view occupancy only
    """
    from neurospatial.encoding._spikes import normalize_spike_times

    # Normalize spike times to canonical list-of-arrays format
    spike_times_list = normalize_spike_times(spike_times)
    n_neurons = len(spike_times_list)

    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)

    # Compute view occupancy once (shared across all neurons)
    view_occupancy = compute_view_occupancy(
        env,
        times,
        positions,
        headings,
        gaze_model=gaze_model,
        view_distance=view_distance,
        gaze_offsets=gaze_offsets,
    )

    # Process neurons
    if n_jobs == 1:
        # Sequential processing
        spike_counts = np.zeros((n_neurons, env.n_bins), dtype=np.float64)
        for i, spikes in enumerate(spike_times_list):
            spike_counts[i] = bin_view_spike_train(
                env,
                spikes,
                times,
                positions,
                headings,
                gaze_model=gaze_model,
                view_distance=view_distance,
                gaze_offsets=gaze_offsets,
            )
    else:
        # Parallel processing with joblib
        from joblib import Parallel, delayed

        def _process_neuron(spikes: NDArray[np.float64]) -> NDArray[np.float64]:
            return bin_view_spike_train(
                env,
                spikes,
                times,
                positions,
                headings,
                gaze_model=gaze_model,
                view_distance=view_distance,
                gaze_offsets=gaze_offsets,
            )

        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_neuron)(spikes) for spikes in spike_times_list
        )
        spike_counts = np.array(results, dtype=np.float64)

    return spike_counts, view_occupancy
