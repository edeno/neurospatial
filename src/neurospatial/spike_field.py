"""Spike train to spatial field conversion primitives.

This module provides foundational functions for converting spike data
into occupancy-normalized spatial fields (firing rate maps).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.environment._protocols import EnvironmentProtocol


def spikes_to_field(
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

    See Also
    --------
    compute_place_field : Convenience function combining spike conversion and smoothing.
    Environment.occupancy : Compute time spent in each bin.
    Environment.smooth : Smooth spatial fields.

    Notes
    -----
    The firing rate field is computed as:

    .. math::
        r_i = \\frac{n_i}{T_i}

    where :math:`n_i` is the spike count in bin :math:`i` and :math:`T_i` is
    the occupancy time (seconds) in that bin.

    Bins with occupancy less than `min_occupancy_seconds` are set to NaN.
    Setting `min_occupancy_seconds > 0` (e.g., 0.5 seconds) is standard
    practice in place field analysis to avoid spurious high rates from
    brief visits. The default (0.0) includes all bins regardless of occupancy.

    Empty spike trains (no spikes) produce a field of zeros (or NaN where
    occupancy is less than `min_occupancy_seconds`).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.spike_field import spikes_to_field
    >>>
    >>> # Create trajectory
    >>> positions = np.column_stack(
    ...     [
    ...         np.linspace(0, 100, 1000),
    ...         np.linspace(0, 100, 1000),
    ...     ]
    ... )
    >>> times = np.linspace(0, 10, 1000)  # 10 seconds
    >>>
    >>> # Create environment
    >>> env = Environment.from_samples(positions, bin_size=10.0)
    >>>
    >>> # Create spike train (50 spikes over 10 seconds = 5 Hz mean rate)
    >>> spike_times = np.linspace(0, 10, 50)
    >>>
    >>> # Compute firing rate field
    >>> field = spikes_to_field(env, spike_times, times, positions)
    >>> field.shape == (env.n_bins,)
    True
    >>> np.nanmean(field)  # Should be close to 5 Hz
    5.0...
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
    # positions is now guaranteed to be 2D (n_timepoints, n_dims)
    if positions.shape[1] == 1:
        # 1D case: positions is shape (n_timepoints, 1)
        spike_x = np.interp(spike_times, times, positions[:, 0])
        spike_positions = spike_x[:, np.newaxis]
    else:
        # Multi-D case: positions is shape (n_timepoints, n_dims)
        spike_positions = np.column_stack(
            [
                np.interp(spike_times, times, positions[:, dim])
                for dim in range(positions.shape[1])
            ]
        )

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


def compute_place_field(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    min_occupancy_seconds: float = 0.0,
    smoothing_bandwidth: float | None = None,
) -> NDArray[np.float64]:
    """Compute smoothed place field from spike train (convenience function).

    This function combines `spikes_to_field()` and `cast("EnvironmentProtocol", env).smooth()` into a
    single convenient call for typical place field analysis workflows.

    Equivalent to::

        field = spikes_to_field(
            env,
            spike_times,
            times,
            positions,
            min_occupancy_seconds=min_occupancy_seconds,
        )
        if smoothing_bandwidth is not None:
            field = cast("EnvironmentProtocol", env).smooth(
                field, bandwidth=smoothing_bandwidth
            )

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Timestamps of spike occurrences (seconds).
    times : NDArray[np.float64], shape (n_timepoints,)
        Timestamps of trajectory samples (seconds).
    positions : NDArray[np.float64], shape (n_timepoints, n_dims) or (n_timepoints,)
        Position trajectory.
    min_occupancy_seconds : float, default=0.0
        Minimum occupancy (seconds) required for reliable firing rate estimate.
        Bins with less occupancy are set to NaN. For typical place field
        analysis, 0.5 seconds is recommended.
    smoothing_bandwidth : float or None, default=None
        Bandwidth for Gaussian smoothing (same units as environment).
        If None, no smoothing is applied.

    Returns
    -------
    field : NDArray[np.float64], shape (n_bins,)
        Smoothed firing rate field (spikes/second) for each spatial bin.
        Bins with insufficient occupancy are set to NaN.

    See Also
    --------
    spikes_to_field : Lower-level function for spike train to field conversion.
    Environment.smooth : Gaussian smoothing of spatial fields.

    Notes
    -----
    This is a convenience wrapper for the common workflow of computing a
    firing rate map and then smoothing it. For more control over the
    smoothing parameters or to skip smoothing entirely, use
    `spikes_to_field()` directly followed by `cast("EnvironmentProtocol", env).smooth()`.

    Typical smoothing bandwidths for place field analysis are 5-10 cm
    for small environments or 20-50 cm for large open fields.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.spike_field import compute_place_field
    >>>
    >>> # Create trajectory and spike train
    >>> positions = np.column_stack(
    ...     [
    ...         np.linspace(0, 100, 1000),
    ...         np.linspace(0, 100, 1000),
    ...     ]
    ... )
    >>> times = np.linspace(0, 10, 1000)
    >>> spike_times = np.linspace(0, 10, 50)
    >>>
    >>> # Create environment
    >>> env = Environment.from_samples(positions, bin_size=10.0)
    >>>
    >>> # Compute smoothed place field (one-liner)
    >>> field = compute_place_field(
    ...     env, spike_times, times, positions, smoothing_bandwidth=5.0
    ... )
    >>> field.shape == (env.n_bins,)
    True
    """
    # Compute raw firing rate field
    field = spikes_to_field(
        env,
        spike_times,
        times,
        positions,
        min_occupancy_seconds=min_occupancy_seconds,
    )

    # Apply smoothing if requested
    if smoothing_bandwidth is None:
        return field

    # Handle NaN values: cast("EnvironmentProtocol", env).smooth() doesn't accept NaN
    # Standard approach: fill NaN with 0, smooth, then restore NaN
    nan_mask = np.isnan(field)

    # No NaN values - smooth directly
    if not np.any(nan_mask):
        return cast("EnvironmentProtocol", env).smooth(
            field, bandwidth=smoothing_bandwidth
        )

    # Has NaN values - fill, smooth, then restore NaN
    field_filled = field.copy()
    field_filled[nan_mask] = 0.0

    # Smooth the filled field
    field_smoothed = cast("EnvironmentProtocol", env).smooth(
        field_filled, bandwidth=smoothing_bandwidth
    )

    # Restore NaN in original low-occupancy bins
    field_smoothed[nan_mask] = np.nan
    return field_smoothed
