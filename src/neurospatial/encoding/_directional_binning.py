"""Binning layer for directional encoding.

This module converts spike trains and head direction data into discrete spike
counts and occupancy arrays for head direction cell analysis.

The functions in this module handle:
1. Circular binning of head directions into angular bins (0 to 2π)
2. Occupancy computation from continuous head direction time series
3. Spike counting by interpolating head direction at spike times
4. Batch processing of multiple neurons with joblib parallelization

Output shapes:
- Spike counts (single neuron): (n_bins,)
- Spike counts (batch): (n_neurons, n_bins)
- Occupancy: (n_bins,) - always shared across neurons
- Bin centers: (n_bins,) - angles in radians [0, 2π)

The binning layer is intentionally separated from smoothing to allow:
- Reusing occupancy across multiple neurons
- Precomputing bin centers for efficiency
- Future JAX implementations with different parallelization strategies

Notes
-----
Unlike spatial binning, directional binning does not require an Environment.
Head direction is a 1D circular variable independent of spatial position.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "bin_directional_spike_train",
    "bin_directional_spike_trains",
    "compute_directional_occupancy",
]


def compute_directional_occupancy(
    times: NDArray[np.float64],
    headings: NDArray[np.float64],
    bin_size: float,
    *,
    angle_unit: Literal["rad", "deg"] = "rad",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute occupancy (time spent at each direction) and bin centers.

    Computes the total time spent facing each direction by accumulating
    time intervals from the head direction time series.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Timestamps of head direction samples in seconds.
        Must be strictly monotonically increasing.
    headings : ndarray, shape (n_samples,)
        Head direction at each time point. Units determined by ``angle_unit``.
    bin_size : float
        Width of angular bins. Units match ``angle_unit``.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of ``headings`` and ``bin_size``.
        - 'rad': headings in radians, bin_size in radians
        - 'deg': headings in degrees, bin_size in degrees

    Returns
    -------
    occupancy : ndarray, shape (n_bins,)
        Time in seconds spent at each direction.
    bin_centers : ndarray, shape (n_bins,)
        Center of each angular bin in radians [0, 2π).

    Raises
    ------
    ValueError
        If times and headings have different lengths.
        If times are not strictly monotonically increasing.
        If fewer than 3 samples provided.
        If angle_unit is not 'rad' or 'deg'.

    Notes
    -----
    **Occupancy calculation**: Uses actual time deltas between frames
    (``np.diff(times)``) rather than assuming uniform sampling.
    This correctly handles dropped frames and variable sampling rates.
    The last frame is excluded from occupancy since we don't know how
    long the animal stayed at that direction.

    **Circular binning**: Headings are wrapped to [0, 2π) before binning.
    Bins are evenly spaced from 0 to 2π.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._directional_binning import (
    ...     compute_directional_occupancy,
    ... )

    >>> # Create trajectory
    >>> times = np.linspace(0, 10.0, 100)
    >>> headings = np.random.uniform(0, 2 * np.pi, 100)

    >>> # Compute occupancy
    >>> occupancy, bin_centers = compute_directional_occupancy(
    ...     times, headings, bin_size=np.pi / 30
    ... )
    >>> occupancy.shape[0] == 60  # 2π / (π/30) = 60 bins
    True
    """
    # Validate angle_unit
    if angle_unit not in ("rad", "deg"):
        raise ValueError(f"angle_unit must be 'rad' or 'deg', got '{angle_unit}'")

    # Convert inputs to arrays
    times = np.asarray(times, dtype=np.float64).ravel()
    headings = np.asarray(headings, dtype=np.float64).ravel()

    # Validate inputs
    if len(headings) != len(times):
        raise ValueError(
            f"headings and times must have the same length. "
            f"Got headings: {len(headings)}, times: {len(times)}.\n"
            f"Fix: Ensure both arrays represent the same time series."
        )

    if len(times) < 3:
        raise ValueError(
            f"Need at least 3 samples to compute occupancy. "
            f"Got {len(times)} samples.\n"
            f"Fix: Provide more data points."
        )

    # Check strict monotonicity (no duplicates, no decreasing)
    time_diffs = np.diff(times)
    if np.any(time_diffs <= 0):
        n_problems = np.sum(time_diffs <= 0)
        raise ValueError(
            f"times must be strictly monotonically increasing (no duplicates). "
            f"Found {n_problems} non-increasing time steps.\n"
            f"Fix: Remove duplicate timestamps or check for timestamp errors."
        )

    # Validate bin_size
    if bin_size <= 0:
        raise ValueError(
            f"bin_size must be positive, got {bin_size}.\n"
            f"Fix: Use a positive bin size (e.g., np.pi/30 radians or 6 degrees)."
        )

    # Convert to radians if needed
    if angle_unit == "deg":
        headings_rad = np.radians(headings)
        bin_size_rad = np.radians(bin_size)
    else:
        headings_rad = headings
        bin_size_rad = bin_size

    # Validate bin_size produces valid number of bins
    n_bins = int(np.round(2 * np.pi / bin_size_rad))
    if n_bins < 1:
        raise ValueError(
            f"bin_size is too large: {bin_size} ({angle_unit}). "
            f"Results in {n_bins} bins (need at least 1).\n"
            f"Fix: Use a smaller bin_size (max ~2π radians or 360 degrees)."
        )
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Wrap headings to [0, 2*pi)
    headings_wrapped = headings_rad % (2 * np.pi)

    # Compute occupancy using actual time deltas
    # Each frame i contributes the time until frame i+1
    # The last frame is excluded (we don't know how long the animal stayed there)
    time_deltas = np.diff(times)

    # Assign each frame (except last) to a bin
    # headings_wrapped[:-1] has n-1 elements, matching time_deltas
    frame_bins = np.digitize(headings_wrapped[:-1], bin_edges) - 1
    # Handle edge case: value exactly at 2*pi goes to bin n_bins, wrap to 0
    frame_bins[frame_bins >= n_bins] = 0

    # Compute occupancy per bin using vectorized bincount
    occupancy = np.bincount(frame_bins, weights=time_deltas, minlength=n_bins).astype(
        np.float64
    )

    return occupancy, bin_centers


def bin_directional_spike_train(
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    headings: NDArray[np.float64],
    bin_size: float,
    *,
    angle_unit: Literal["rad", "deg"] = "rad",
) -> NDArray[np.float64]:
    """Bin spike train into directional bins.

    Converts continuous spike times to spike counts per angular bin by
    looking up the head direction at each spike time and counting spikes
    in each bin.

    Parameters
    ----------
    spike_times : ndarray, shape (n_spikes,)
        Times of spike events in seconds.
    times : ndarray, shape (n_samples,)
        Timestamps of head direction samples in seconds.
    headings : ndarray, shape (n_samples,)
        Head direction at each time point. Units determined by ``angle_unit``.
    bin_size : float
        Width of angular bins. Units match ``angle_unit``.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of ``headings`` and ``bin_size``.

    Returns
    -------
    ndarray, shape (n_bins,)
        Number of spikes in each angular bin (float64 for compatibility
        with smoothing operations).

    Notes
    -----
    **Spike assignment**: Spikes are assigned to bins using nearest-neighbor
    lookup (not interpolation) to correctly handle circular discontinuities.
    Linear interpolation would give wrong results when head direction crosses
    the 0°/360° boundary (e.g., 350° to 10° would incorrectly interpolate to
    180°). Spikes outside the recording window are excluded.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._directional_binning import (
    ...     bin_directional_spike_train,
    ... )

    >>> # Create trajectory and spikes
    >>> times = np.linspace(0, 10, 100)
    >>> headings = np.random.uniform(0, 2 * np.pi, 100)
    >>> spike_times = np.array([1.0, 2.5, 4.0, 7.5])

    >>> # Bin spikes
    >>> spike_counts = bin_directional_spike_train(
    ...     spike_times, times, headings, bin_size=np.pi / 30
    ... )
    >>> spike_counts.shape[0] == 60
    True

    See Also
    --------
    compute_directional_occupancy : Compute occupancy
    bin_directional_spike_trains : Batch version for multiple neurons
    """
    # Validate angle_unit
    if angle_unit not in ("rad", "deg"):
        raise ValueError(f"angle_unit must be 'rad' or 'deg', got '{angle_unit}'")

    # Validate bin_size
    if bin_size <= 0:
        raise ValueError(
            f"bin_size must be positive, got {bin_size}.\n"
            f"Fix: Use a positive bin size (e.g., np.pi/30 radians or 6 degrees)."
        )

    spike_times = np.asarray(spike_times, dtype=np.float64).ravel()
    times = np.asarray(times, dtype=np.float64).ravel()
    headings = np.asarray(headings, dtype=np.float64).ravel()

    # Convert to radians if needed
    if angle_unit == "deg":
        headings_rad = np.radians(headings)
        bin_size_rad = np.radians(bin_size)
    else:
        headings_rad = headings
        bin_size_rad = bin_size

    # Validate bin_size produces valid number of bins
    n_bins = int(np.round(2 * np.pi / bin_size_rad))
    if n_bins < 1:
        raise ValueError(
            f"bin_size is too large: {bin_size} ({angle_unit}). "
            f"Results in {n_bins} bins (need at least 1).\n"
            f"Fix: Use a smaller bin_size (max ~2π radians or 360 degrees)."
        )

    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)

    # Initialize spike counts
    spike_counts = np.zeros(n_bins, dtype=np.float64)

    # Handle empty spike train
    if len(spike_times) == 0:
        return spike_counts

    # Filter spikes to valid time range
    valid_mask = (spike_times >= times[0]) & (spike_times <= times[-1])
    valid_spike_times = spike_times[valid_mask]

    if len(valid_spike_times) == 0:
        return spike_counts

    # Wrap headings to [0, 2*pi)
    headings_wrapped = headings_rad % (2 * np.pi)

    # Use nearest-neighbor assignment to avoid circular interpolation issues
    # np.interp would give wrong results when head direction crosses 0/2pi
    # (e.g., 350° to 10° would interpolate to 180° instead of ~0°)
    spike_indices = np.searchsorted(times, valid_spike_times, side="right") - 1
    spike_indices = np.clip(spike_indices, 0, len(headings_wrapped) - 1)
    spike_hd = headings_wrapped[spike_indices]

    # Assign spikes to bins
    spike_bins = np.digitize(spike_hd, bin_edges) - 1
    spike_bins[spike_bins >= n_bins] = 0

    # Count spikes per bin using vectorized bincount
    spike_counts = np.bincount(spike_bins, minlength=n_bins).astype(np.float64)

    return spike_counts


def bin_directional_spike_trains(
    spike_times: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    times: NDArray[np.float64],
    headings: NDArray[np.float64],
    bin_size: float,
    *,
    angle_unit: Literal["rad", "deg"] = "rad",
    n_jobs: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Bin multiple spike trains into directional bins.

    Batch version of bin_directional_spike_train that efficiently processes
    multiple neurons. Precomputes shared quantities (occupancy, bin centers)
    and optionally parallelizes spike counting with joblib.

    Parameters
    ----------
    spike_times : sequence of arrays or 2D array
        Spike times for each neuron. Can be:
        - List/tuple of 1D arrays (one per neuron)
        - 2D array shape (n_neurons, max_spikes) with NaN padding
        Input is normalized via normalize_spike_times().
    times : ndarray, shape (n_samples,)
        Timestamps of head direction samples in seconds.
    headings : ndarray, shape (n_samples,)
        Head direction at each time point. Units determined by ``angle_unit``.
    bin_size : float
        Width of angular bins. Units match ``angle_unit``.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of ``headings`` and ``bin_size``.
    n_jobs : int, default=1
        Number of parallel jobs for spike counting. Use -1 for all CPUs.
        1 means sequential processing (no parallelization overhead).

    Returns
    -------
    spike_counts : ndarray, shape (n_neurons, n_bins)
        Number of spikes in each angular bin for each neuron.
    occupancy : ndarray, shape (n_bins,)
        Time in seconds spent at each direction (shared across neurons).
    bin_centers : ndarray, shape (n_bins,)
        Center of each angular bin in radians [0, 2π).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._directional_binning import (
    ...     bin_directional_spike_trains,
    ... )

    >>> # Create trajectory and spikes for 3 neurons
    >>> times = np.linspace(0, 10, 100)
    >>> headings = np.random.uniform(0, 2 * np.pi, 100)
    >>> spike_times = [
    ...     np.array([1.0, 2.5, 4.0]),  # Neuron 0
    ...     np.array([0.5, 1.5, 2.5, 3.5]),  # Neuron 1
    ...     np.array([5.0]),  # Neuron 2
    ... ]

    >>> # Bin spikes
    >>> spike_counts, occupancy, bin_centers = bin_directional_spike_trains(
    ...     spike_times, times, headings, bin_size=np.pi / 30, n_jobs=2
    ... )
    >>> spike_counts.shape[0] == 3  # 3 neurons
    True
    >>> spike_counts.shape[1] == 60  # 60 bins
    True
    >>> occupancy.shape[0] == 60
    True

    See Also
    --------
    bin_directional_spike_train : Single-neuron version
    compute_directional_occupancy : Compute occupancy only
    normalize_spike_times : Input format normalization
    """
    from neurospatial.encoding._spikes import normalize_spike_times

    # Normalize spike times to canonical list-of-arrays format
    spike_times_list = normalize_spike_times(spike_times)
    n_neurons = len(spike_times_list)

    times = np.asarray(times, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)

    # Compute occupancy and bin centers once (shared across all neurons)
    occupancy, bin_centers = compute_directional_occupancy(
        times, headings, bin_size, angle_unit=angle_unit
    )
    n_bins = len(occupancy)

    # Process neurons
    if n_jobs == 1:
        # Sequential processing
        spike_counts = np.zeros((n_neurons, n_bins), dtype=np.float64)
        for i, spikes in enumerate(spike_times_list):
            spike_counts[i] = bin_directional_spike_train(
                spikes, times, headings, bin_size, angle_unit=angle_unit
            )
    else:
        # Parallel processing with joblib
        from joblib import Parallel, delayed

        def _process_neuron(spikes: NDArray[np.float64]) -> NDArray[np.float64]:
            return bin_directional_spike_train(
                spikes, times, headings, bin_size, angle_unit=angle_unit
            )

        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_neuron)(spikes) for spikes in spike_times_list
        )
        spike_counts = np.array(results, dtype=np.float64)

    return spike_counts, occupancy, bin_centers
