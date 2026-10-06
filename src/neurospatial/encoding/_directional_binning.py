"""Binning layer for directional encoding.

This module converts spike trains and head direction data into discrete spike
counts and occupancy arrays for head direction cell analysis.

The functions in this module handle:
1. Circular binning of head directions into angular bins (0 to 2π)
2. Occupancy computation from continuous head direction time series
3. Spike counting from the most recent frame, using shared interval validity
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

from neurospatial.encoding._binning import count_spikes_by_frame
from neurospatial.environment.trajectory import (
    interval_valid_mask,
    start_allocated_occupancy,
)

__all__ = [
    "bin_directional_spike_train",
    "bin_directional_spike_trains",
    "compute_directional_occupancy",
]


def _validate_directional_samples(
    times: NDArray[np.float64], headings: NDArray[np.float64]
) -> None:
    """Validate aligned directional samples and positive sampling intervals."""
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


def compute_directional_occupancy(
    times: NDArray[np.float64],
    headings: NDArray[np.float64],
    bin_size: float,
    *,
    angle_unit: Literal["rad", "deg"] = "rad",
    max_gap: float | None = 0.5,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
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

    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : ndarray of shape (n, 2), or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    spike_window : ndarray of shape (n, 2), or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording.
        The calling public encoder records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
        Windows must already be normalized by ``resolve_time_windows``;
        public encoders accept and normalize the other supported input forms.

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
    times = np.asarray(times, dtype=np.float64).ravel()
    headings = np.asarray(headings, dtype=np.float64).ravel()
    _validate_directional_samples(times, headings)
    frame_bins, bin_centers = directional_frame_bins(
        headings, bin_size, angle_unit=angle_unit
    )
    n_bins = len(bin_centers)
    mask = interval_valid_mask(
        times,
        start_bin=frame_bins,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
    )
    occupancy = start_allocated_occupancy(frame_bins, np.diff(times), mask, n_bins)
    return occupancy, bin_centers


def directional_frame_bins(
    headings: NDArray[np.float64],
    bin_size: float,
    *,
    angle_unit: Literal["rad", "deg"] = "rad",
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Map finite headings to circular bins; non-finite headings map to -1."""
    if angle_unit not in ("rad", "deg"):
        raise ValueError(f"angle_unit must be 'rad' or 'deg', got '{angle_unit}'")

    if bin_size <= 0:
        raise ValueError(
            f"bin_size must be positive, got {bin_size}.\n"
            f"Fix: Use a positive bin size (e.g., np.pi/30 radians or 6 degrees)."
        )

    headings_rad = np.radians(headings) if angle_unit == "deg" else headings
    bin_size_rad = np.radians(bin_size) if angle_unit == "deg" else bin_size

    n_bins = int(np.round(2 * np.pi / bin_size_rad))
    if n_bins < 1:
        raise ValueError(
            f"bin_size is too large: {bin_size} ({angle_unit}). "
            f"Results in {n_bins} bins (need at least 1).\n"
            f"Fix: Use a smaller bin_size (max ~2π radians or 360 degrees)."
        )

    finite = np.isfinite(headings_rad)
    wrapped = headings_rad[finite] % (2 * np.pi)
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    frame_bins = np.full(headings.shape, -1, dtype=np.intp)
    bins = np.digitize(wrapped, bin_edges) - 1
    bins[bins >= n_bins] = 0
    frame_bins[finite] = bins
    return frame_bins, bin_centers


def bin_directional_spike_train(
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    headings: NDArray[np.float64],
    bin_size: float,
    *,
    angle_unit: Literal["rad", "deg"] = "rad",
    max_gap: float | None = 0.5,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
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

    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : ndarray of shape (n, 2), or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    spike_window : ndarray of shape (n, 2), or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording.
        The calling public encoder records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
        Windows must already be normalized by ``resolve_time_windows``;
        public encoders accept and normalize the other supported input forms.

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
    spike_times = np.asarray(spike_times, dtype=np.float64).ravel()
    times = np.asarray(times, dtype=np.float64).ravel()
    headings = np.asarray(headings, dtype=np.float64).ravel()
    frame_bins, bin_centers = directional_frame_bins(
        headings, bin_size, angle_unit=angle_unit
    )
    n_bins = len(bin_centers)
    mask = interval_valid_mask(
        times,
        start_bin=frame_bins,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
    )
    return count_spikes_by_frame(spike_times, times, frame_bins, mask, n_bins)


def bin_directional_spike_trains(
    spike_times: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    times: NDArray[np.float64],
    headings: NDArray[np.float64],
    bin_size: float,
    *,
    angle_unit: Literal["rad", "deg"] = "rad",
    max_gap: float | None = 0.5,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
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
        Input is coerced to per-neuron spike trains via as_spike_trains().
    times : ndarray, shape (n_samples,)
        Timestamps of head direction samples in seconds.
    headings : ndarray, shape (n_samples,)
        Head direction at each time point. Units determined by ``angle_unit``.
    bin_size : float
        Width of angular bins. Units match ``angle_unit``.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of ``headings`` and ``bin_size``.
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : ndarray of shape (n, 2), or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    spike_window : ndarray of shape (n, 2), or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording.
        The calling public encoder records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
        Windows must already be normalized by ``resolve_time_windows``;
        public encoders accept and normalize the other supported input forms.
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
    as_spike_trains : Coerce input to canonical per-neuron spike trains
    """
    from neurospatial.encoding._spikes import as_spike_trains

    spike_times_list = as_spike_trains(spike_times)
    n_neurons = len(spike_times_list)
    times = np.asarray(times, dtype=np.float64).ravel()
    headings = np.asarray(headings, dtype=np.float64).ravel()
    _validate_directional_samples(times, headings)
    frame_bins, bin_centers = directional_frame_bins(
        headings, bin_size, angle_unit=angle_unit
    )
    n_bins = len(bin_centers)
    mask = interval_valid_mask(
        times,
        start_bin=frame_bins,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
    )
    occupancy = start_allocated_occupancy(frame_bins, np.diff(times), mask, n_bins)
    spike_counts = np.zeros((n_neurons, n_bins), dtype=np.float64)
    if n_neurons and n_jobs != 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(
            delayed(count_spikes_by_frame)(spikes, times, frame_bins, mask, n_bins)
            for spikes in spike_times_list
        )
        spike_counts = np.asarray(results, dtype=np.float64)
    else:
        for i, spikes in enumerate(spike_times_list):
            spike_counts[i] = count_spikes_by_frame(
                spikes, times, frame_bins, mask, n_bins
            )
    return spike_counts, occupancy, bin_centers
