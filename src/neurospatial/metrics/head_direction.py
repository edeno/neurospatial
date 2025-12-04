"""
Head direction cell analysis module.

This module provides tools for analyzing head direction (HD) cells, neurons
that fire preferentially when an animal faces a particular direction.
HD cells are found in various brain regions including the postsubiculum,
anterodorsal thalamus, and lateral mammillary nucleus.

Which Function Should I Use?
----------------------------
**Computing tuning curve from raw data?**
    Use ``head_direction_tuning_curve()`` to compute firing rate as a function
    of head direction from spike times and head direction time series.

**Analyzing tuning curve properties?**
    Use ``head_direction_metrics()`` to compute preferred direction, mean
    vector length, tuning width, and HD cell classification.

**Screening many neurons (100s-1000s)?**
    Use ``is_head_direction_cell()`` for fast boolean filtering without
    manually computing tuning curves and metrics.

**Visualizing HD tuning?**
    Use ``plot_head_direction_tuning()`` for standard polar or linear plots.

Typical Workflow
----------------
1. Compute tuning curve from spike times and head directions::

    >>> bin_centers, firing_rates = head_direction_tuning_curve(
    ...     head_directions, spike_times, position_times,
    ...     bin_size=6.0, angle_unit='deg'
    ... )  # doctest: +SKIP

2. Compute metrics and classify::

    >>> metrics = head_direction_metrics(bin_centers, firing_rates)
    >>> print(metrics)  # Human-readable interpretation  # doctest: +SKIP
    >>> if metrics.is_hd_cell:
    ...     print(f"Preferred direction: {metrics.preferred_direction_deg:.1f}°")
    ...     # doctest: +SKIP

3. Visualize::

    >>> plot_head_direction_tuning(bin_centers, firing_rates, metrics)
    ...     # doctest: +SKIP

Common Parameters
-----------------
Most functions accept ``angle_unit`` parameter: ``'rad'`` (default) or ``'deg'``.
HD research commonly uses degrees, but we default to radians for scipy
compatibility. Use ``angle_unit='deg'`` if your data is in degrees.

References
----------
Taube, J.S., Muller, R.U., & Ranck, J.B. (1990). Head-direction cells recorded
    from the postsubiculum in freely moving rats. I. Description and
    quantitative analysis. Journal of Neuroscience, 10(2), 420-435.
Sargolini, F. et al. (2006). Conjunctive representation of position, direction,
    and velocity in entorhinal cortex. Science, 312(5774), 758-762.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from neurospatial.metrics.circular import _to_radians

# Mark that circular imports are available for testing
_has_circular_imports = True

__all__: list[str] = [
    "head_direction_tuning_curve",
]


def head_direction_tuning_curve(
    head_directions: NDArray[np.float64],
    spike_times: NDArray[np.float64],
    position_times: NDArray[np.float64],
    *,
    bin_size: float = 6.0,
    angle_unit: Literal["rad", "deg"] = "deg",
    smoothing_window: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute head direction tuning curve from spike times and head direction data.

    This function computes firing rate as a function of head direction by:
    1. Binning head directions into angular bins
    2. Computing occupancy (time spent in each bin) using actual time deltas
    3. Counting spikes in each bin by interpolating head direction at spike times
    4. Computing firing rate = spike count / occupancy
    5. Optionally applying Gaussian smoothing with circular boundary handling

    Parameters
    ----------
    head_directions : array, shape (n_frames,)
        Head direction at each time point. Units determined by ``angle_unit``.
    spike_times : array, shape (n_spikes,)
        Times of spikes in the same units as ``position_times``.
    position_times : array, shape (n_frames,)
        Timestamps corresponding to each head direction sample.
        Must be monotonically increasing.
    bin_size : float, default=6.0
        Width of angular bins. Units match ``angle_unit`` (degrees by default).
    angle_unit : {'rad', 'deg'}, default='deg'
        Unit of ``head_directions`` and ``bin_size``.
        Note: HD research commonly uses degrees, hence the default differs
        from other circular statistics functions.
    smoothing_window : int, default=5
        Standard deviation of Gaussian smoothing kernel in bins.
        Set to 0 to disable smoothing.

    Returns
    -------
    bin_centers : array, shape (n_bins,)
        Center of each angular bin in radians.
    firing_rates : array, shape (n_bins,)
        Firing rate (Hz) in each bin.

    Raises
    ------
    ValueError
        If head_directions and position_times have different lengths.
        If position_times are not monotonically increasing.
        If fewer than 3 samples provided.

    See Also
    --------
    head_direction_metrics : Compute tuning curve properties.
    is_head_direction_cell : Fast screening for HD cells.

    Notes
    -----
    **Occupancy calculation**: Uses actual time deltas between frames
    (``np.diff(position_times)``) rather than assuming uniform sampling.
    This correctly handles dropped frames and variable sampling rates.
    The last frame is excluded from occupancy since we don't know how
    long the animal stayed at that position.

    **Spike assignment**: Spikes are assigned to bins using nearest-neighbor
    lookup (not interpolation) to correctly handle circular discontinuities.
    Linear interpolation would give wrong results when head direction crosses
    the 0°/360° boundary (e.g., 350° to 10° would incorrectly interpolate to
    180°). Spikes outside the recording window are excluded.

    **Bin edge handling**: Head directions are assigned using ``np.digitize``,
    which assigns values in range ``[bin_edges[i], bin_edges[i+1])`` to bin ``i``.
    Values exactly at 2π are wrapped to bin 0.

    **Circular smoothing**: Gaussian smoothing uses ``mode='wrap'`` to
    correctly handle the circular boundary (0° neighbors 360°).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics import head_direction_tuning_curve
    >>> # Create sample data: 10 seconds at 30 Hz
    >>> position_times = np.linspace(0, 10, 300)
    >>> head_directions = np.random.default_rng(42).uniform(0, 360, 300)
    >>> spike_times = np.random.default_rng(42).uniform(0, 10, 50)
    >>> bin_centers, firing_rates = head_direction_tuning_curve(
    ...     head_directions,
    ...     spike_times,
    ...     position_times,
    ...     bin_size=30.0,
    ...     angle_unit="deg",
    ... )
    >>> len(bin_centers)  # 360 / 30 = 12 bins
    12
    >>> firing_rates.shape
    (12,)
    """
    # Convert inputs to arrays
    head_directions = np.asarray(head_directions, dtype=np.float64).ravel()
    spike_times = np.asarray(spike_times, dtype=np.float64).ravel()
    position_times = np.asarray(position_times, dtype=np.float64).ravel()

    # Validate inputs
    if len(head_directions) != len(position_times):
        raise ValueError(
            f"head_directions and position_times must have the same length. "
            f"Got head_directions: {len(head_directions)}, "
            f"position_times: {len(position_times)}.\n"
            f"Fix: Ensure both arrays represent the same time series."
        )

    if len(position_times) < 3:
        raise ValueError(
            f"Need at least 3 samples to compute tuning curve. "
            f"Got {len(position_times)} samples.\n"
            f"Fix: Provide more data points."
        )

    # Check strict monotonicity (no duplicates, no decreasing)
    time_diffs = np.diff(position_times)
    if np.any(time_diffs <= 0):
        n_problems = np.sum(time_diffs <= 0)
        raise ValueError(
            f"position_times must be strictly increasing (no duplicates). "
            f"Found {n_problems} non-increasing time steps.\n"
            f"Fix: Remove duplicate timestamps or check for timestamp errors."
        )

    # Convert to radians if needed
    head_directions_rad = _to_radians(head_directions, angle_unit)
    bin_size_rad = np.radians(bin_size) if angle_unit == "deg" else bin_size

    # Compute bin edges and centers
    n_bins = int(np.round(2 * np.pi / bin_size_rad))
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Wrap head directions to [0, 2*pi)
    head_directions_wrapped = head_directions_rad % (2 * np.pi)

    # Compute occupancy using actual time deltas
    # Each frame i contributes the time until frame i+1
    # The last frame is excluded (we don't know how long the animal stayed there)
    time_deltas = np.diff(position_times)

    # Assign each frame (except last) to a bin
    # head_directions_wrapped[:-1] has n-1 elements, matching time_deltas
    frame_bins = np.digitize(head_directions_wrapped[:-1], bin_edges) - 1
    # Handle edge case: value exactly at 2*pi goes to bin n_bins, wrap to 0
    frame_bins[frame_bins >= n_bins] = 0

    # Compute occupancy per bin using vectorized bincount (much faster than loop)
    occupancy = np.bincount(frame_bins, weights=time_deltas, minlength=n_bins).astype(
        np.float64
    )

    # Count spikes per bin
    spike_counts = np.zeros(n_bins, dtype=np.float64)

    if len(spike_times) > 0:
        # Filter spikes to valid time range
        valid_mask = (spike_times >= position_times[0]) & (
            spike_times <= position_times[-1]
        )
        valid_spike_times = spike_times[valid_mask]

        if len(valid_spike_times) > 0:
            # Use nearest-neighbor assignment to avoid circular interpolation issues
            # np.interp would give wrong results when head direction crosses 0/2pi
            # (e.g., 350° to 10° would interpolate to 180° instead of ~0°)
            spike_indices = (
                np.searchsorted(position_times, valid_spike_times, side="right") - 1
            )
            spike_indices = np.clip(spike_indices, 0, len(head_directions_wrapped) - 1)
            spike_hd = head_directions_wrapped[spike_indices]

            # Assign spikes to bins
            spike_bins = np.digitize(spike_hd, bin_edges) - 1
            spike_bins[spike_bins >= n_bins] = 0

            # Count spikes per bin using vectorized bincount
            spike_counts = np.bincount(spike_bins, minlength=n_bins).astype(np.float64)

    # Compute firing rates (Hz) = spike count / occupancy
    # Handle division by zero (bins with no occupancy get zero rate)
    with np.errstate(divide="ignore", invalid="ignore"):
        firing_rates = np.where(occupancy > 0, spike_counts / occupancy, 0.0)

    # Apply Gaussian smoothing with circular boundary
    if smoothing_window > 0:
        firing_rates = gaussian_filter1d(firing_rates, smoothing_window, mode="wrap")

    return bin_centers, firing_rates
