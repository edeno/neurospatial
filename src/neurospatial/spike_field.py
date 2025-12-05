"""Spike train to spatial field conversion primitives.

This module provides foundational functions for converting spike data
into occupancy-normalized spatial fields (firing rate maps).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.environment._protocols import EnvironmentProtocol


@dataclass(frozen=True)
class DirectionalPlaceFields:
    """Container for direction-conditioned place field results.

    Stores firing rate maps computed separately for different movement
    directions or trial types. This enables analysis of directional
    tuning in place cells.

    Attributes
    ----------
    fields : Mapping[str, NDArray[np.float64]]
        Dictionary mapping direction labels (e.g., "A→B", "forward") to
        firing rate arrays. Each array has shape (n_bins,) matching the
        environment's bin structure.
    labels : tuple[str, ...]
        Tuple of direction labels in iteration order. Preserves the order
        in which directions were processed, enabling reproducible iteration.

    Examples
    --------
    >>> import numpy as np
    >>> fields = {
    ...     "home→goal": np.array([1.0, 2.0, 3.0]),
    ...     "goal→home": np.array([3.0, 2.0, 1.0]),
    ... }
    >>> result = DirectionalPlaceFields(
    ...     fields=fields,
    ...     labels=("home→goal", "goal→home"),
    ... )
    >>> result.fields["home→goal"]
    array([1., 2., 3.])
    >>> result.labels
    ('home→goal', 'goal→home')

    See Also
    --------
    compute_directional_place_fields : Compute directional place fields from spike data.
    """

    fields: Mapping[str, NDArray[np.float64]]
    labels: tuple[str, ...]


def _subset_spikes_by_time_mask(
    times: NDArray[np.float64],
    spike_times: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Subset spike times by a boolean mask over trajectory times.

    Extracts spikes that fall within the time ranges defined by contiguous
    True segments in the mask. Uses binary search (searchsorted) for
    efficient O(log n) spike slicing per segment.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_timepoints,)
        Timestamps of trajectory samples (seconds). Must be sorted.
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Timestamps of spike occurrences (seconds). Must be sorted.
    mask : NDArray[np.bool_], shape (n_timepoints,)
        Boolean mask indicating which timepoints to include.
        Contiguous True segments define time ranges for spike inclusion.

    Returns
    -------
    times_sub : NDArray[np.float64]
        Subset of times where mask is True. Same as ``times[mask]``.
    spike_times_sub : NDArray[np.float64]
        Spikes that fall within the time ranges of contiguous True segments.
        Boundaries are inclusive: spikes at segment start/end are included.

    Notes
    -----
    For each contiguous segment of True values in mask:
    - ``t_start = times[segment_first_index]``
    - ``t_end = times[segment_last_index]``
    - Spikes in ``[t_start, t_end]`` (inclusive) are selected

    This function is designed for conditioning place field analysis on
    subsets of the trajectory (e.g., by movement direction, trial type).

    Examples
    --------
    >>> import numpy as np
    >>> times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> spike_times = np.array([0.5, 1.5, 2.5, 3.5])
    >>> mask = np.array([False, True, True, False, False])
    >>> times_sub, spikes_sub = _subset_spikes_by_time_mask(times, spike_times, mask)
    >>> times_sub
    array([1., 2.])
    >>> spikes_sub
    array([1.5])
    """
    # Fast path: empty mask
    if not np.any(mask):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Get indices where mask is True
    true_indices = np.where(mask)[0]

    # Find contiguous segments by looking for gaps > 1
    # diff > 1 indicates a break in contiguity
    if len(true_indices) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Find segment boundaries: where consecutive indices are not adjacent
    breaks = np.where(np.diff(true_indices) > 1)[0] + 1
    segment_starts = np.concatenate([[0], breaks])
    segment_ends = np.concatenate([breaks, [len(true_indices)]])

    # Fast path: empty spike train
    if len(spike_times) == 0:
        return times[mask], np.array([], dtype=np.float64)

    # Collect spikes from each segment
    spike_slices = []

    for seg_start_idx, seg_end_idx in zip(segment_starts, segment_ends, strict=True):
        # Get the actual time indices for this segment
        first_time_idx = true_indices[seg_start_idx]
        last_time_idx = true_indices[seg_end_idx - 1]

        # Get time boundaries
        t_start = times[first_time_idx]
        t_end = times[last_time_idx]

        # Use searchsorted for O(log n) spike slicing
        # side="left" for t_start: include spikes at exactly t_start
        # side="right" for t_end: include spikes at exactly t_end
        spike_start = np.searchsorted(spike_times, t_start, side="left")
        spike_end = np.searchsorted(spike_times, t_end, side="right")

        if spike_start < spike_end:
            spike_slices.append(spike_times[spike_start:spike_end])

    # Concatenate all spike slices
    if spike_slices:
        spike_times_sub = np.concatenate(spike_slices)
    else:
        spike_times_sub = np.array([], dtype=np.float64)

    return times[mask], spike_times_sub


def compute_directional_place_fields(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    direction_labels: NDArray[np.object_],
    *,
    smoothing_method: Literal[
        "diffusion_kde", "gaussian_kde", "binned"
    ] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy_seconds: float = 0.0,
) -> DirectionalPlaceFields:
    """Compute place fields conditioned on movement direction or trial type.

    Separates trajectory data by direction labels and computes independent
    place fields for each direction. This enables analysis of directional
    tuning in place cells, where firing rates differ based on which way
    the animal is moving through a location.

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Timestamps of spike occurrences (seconds).
    times : NDArray[np.float64], shape (n_timepoints,)
        Timestamps of trajectory samples (seconds). Must be sorted.
    positions : NDArray[np.float64], shape (n_timepoints, n_dims) or (n_timepoints,)
        Position trajectory. For 1D, can be shape (n_timepoints,) or (n_timepoints, 1).
    direction_labels : NDArray[object], shape (n_timepoints,)
        Direction label for each timepoint. Each label is a hashable string
        (e.g., "A→B", "forward", "CW"). The special label "other" is excluded
        from results, allowing unlabeled periods to be ignored.
    smoothing_method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Estimation method passed to ``compute_place_field``. See that function
        for detailed descriptions of each method.
    bandwidth : float, default=5.0
        Smoothing bandwidth in environment units (e.g., cm).
    min_occupancy_seconds : float, default=0.0
        Minimum occupancy threshold (only used with method="binned").

    Returns
    -------
    DirectionalPlaceFields
        Container with:
        - ``fields``: Mapping from direction label to firing rate array (n_bins,)
        - ``labels``: Tuple of direction labels in iteration order

    Raises
    ------
    ValueError
        If ``direction_labels`` length doesn't match ``times`` length.
    ValueError
        If ``bandwidth`` is not positive (passed through to ``compute_place_field``).

    See Also
    --------
    compute_place_field : Compute single (non-directional) place field.
    goal_pair_direction_labels : Generate labels from trialized tasks.
    heading_direction_labels : Generate labels from movement heading.

    Notes
    -----
    The "other" label is reserved for timepoints that should be excluded from
    analysis (e.g., inter-trial intervals, stationary periods). Any timepoints
    with label "other" are ignored when computing fields.

    For each unique non-"other" label, this function:
    1. Creates a boolean mask for timepoints with that label
    2. Extracts the trajectory and spikes within those masked periods
    3. Calls ``compute_place_field`` on the subset
    4. Stores the resulting field in the output mapping

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.spike_field import compute_directional_place_fields
    >>>
    >>> # Create environment and trajectory
    >>> positions = np.random.uniform(0, 100, (1000, 2))
    >>> times = np.linspace(0, 100, 1000)
    >>> env = Environment.from_samples(positions, bin_size=10.0)
    >>>
    >>> # Create directional labels (first half forward, second half backward)
    >>> labels = np.array(["forward"] * 500 + ["backward"] * 500, dtype=object)
    >>> spike_times = np.random.uniform(0, 100, 50)
    >>>
    >>> # Compute directional place fields
    >>> result = compute_directional_place_fields(
    ...     env, spike_times, times, positions, labels, bandwidth=10.0
    ... )
    >>> "forward" in result.fields
    True
    >>> "backward" in result.fields
    True
    """
    # Validate direction_labels length matches times
    if len(direction_labels) != len(times):
        raise ValueError(
            f"direction_labels must have same length as times, "
            f"got {len(direction_labels)} and {len(times)}"
        )

    # Convert labels to array
    labels_arr = np.asarray(direction_labels, dtype=object)

    # Get unique labels, excluding "other"
    unique_labels = [label for label in np.unique(labels_arr) if label != "other"]

    # Sort labels for reproducibility
    unique_labels = sorted(unique_labels, key=str)

    # Compute place field for each direction
    fields_dict: dict[str, NDArray[np.float64]] = {}

    for label in unique_labels:
        # Build mask for this direction
        mask = labels_arr == label

        # Get subsets using our helper
        times_sub, spike_times_sub = _subset_spikes_by_time_mask(
            times, spike_times, mask
        )
        positions_sub = positions[mask]

        # Compute place field for this direction
        field = compute_place_field(
            env,
            spike_times_sub,
            times_sub,
            positions_sub,
            smoothing_method=smoothing_method,
            bandwidth=bandwidth,
            min_occupancy_seconds=min_occupancy_seconds,
        )

        fields_dict[label] = field

    return DirectionalPlaceFields(fields=fields_dict, labels=tuple(unique_labels))


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


def _diffusion_kde(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    bandwidth: float,
    *,
    trajectory_bins: NDArray[np.int64] | None = None,
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
    trajectory_bins : NDArray[np.int64] | None, optional
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
        if positions.shape[1] == 1:
            spike_x = np.interp(spike_times_valid, times, positions[:, 0])
            spike_positions = spike_x[:, np.newaxis]
        else:
            spike_positions = np.column_stack(
                [
                    np.interp(spike_times_valid, times, positions[:, dim])
                    for dim in range(positions.shape[1])
                ]
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
        if trajectory_bins is None:
            traj_bins = env.bin_at(positions)
        else:
            traj_bins = trajectory_bins
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
    trajectory_bins: NDArray[np.int64] | None = None,
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
    trajectory_bins : NDArray[np.int64] | None, optional
        Precomputed bin indices for positions. Not used by this method
        (included for API consistency).
    dt : NDArray[np.float64] | None, optional
        Precomputed time intervals between position samples.
    occupancy_density : NDArray[np.float64] | None, optional
        Precomputed occupancy density for each bin.
    """
    # Silence unused parameter warning - trajectory_bins not used for gaussian_kde
    # as we need actual positions for Euclidean distance calculation
    _ = trajectory_bins

    # Normalize positions to 2D
    if positions.ndim == 1:
        positions = positions[:, np.newaxis]

    # Filter and interpolate spike positions
    time_min, time_max = times[0], times[-1]
    valid_spike_mask = (spike_times >= time_min) & (spike_times <= time_max)
    spike_times_valid = spike_times[valid_spike_mask]

    if len(spike_times_valid) > 0:
        if positions.shape[1] == 1:
            spike_x = np.interp(spike_times_valid, times, positions[:, 0])
            spike_positions = spike_x[:, np.newaxis]
        else:
            spike_positions = np.column_stack(
                [
                    np.interp(spike_times_valid, times, positions[:, dim])
                    for dim in range(positions.shape[1])
                ]
            )
    else:
        spike_positions = np.zeros((0, positions.shape[1]))

    # Compute dt for trajectory (use precomputed if provided)
    dt_computed = np.diff(times, prepend=times[0]) if dt is None else dt

    # Vectorized KDE computation using broadcasting for all pairwise distances
    two_sigma_sq = 2 * bandwidth**2
    bin_centers = env.bin_centers  # shape: (n_bins, n_dims)

    # Memory warning for large computations
    # spike_dists_sq and traj_dists_sq are (n_points, n_bins) float64 arrays
    n_spikes = len(spike_positions)
    n_positions = len(positions)
    n_bins = env.n_bins
    max_elements = max(n_spikes * n_bins, n_positions * n_bins)
    if max_elements > 50_000_000:  # ~400 MB threshold
        warnings.warn(
            f"Large gaussian_kde computation: {n_spikes} spikes x {n_bins} bins "
            f"and {n_positions} positions x {n_bins} bins will allocate "
            f"~{max_elements * 8 / 1e6:.0f} MB. Consider using "
            f"method='diffusion_kde' or 'binned' for large datasets.",
            UserWarning,
            stacklevel=3,  # Point to compute_place_field call
        )

    # Compute spike density for all bins at once
    if len(spike_positions) > 0:
        # Compute squared distances: (n_spikes, n_bins)
        spike_dists_sq = (
            np.sum(spike_positions**2, axis=1, keepdims=True)
            + np.sum(bin_centers**2, axis=1)
            - 2 * spike_positions @ bin_centers.T
        )
        spike_weights = np.exp(-spike_dists_sq / two_sigma_sq)
        spike_density = np.sum(spike_weights, axis=0)  # shape: (n_bins,)
    else:
        spike_density = np.zeros(env.n_bins, dtype=np.float64)

    # Compute occupancy density for all bins at once
    if occupancy_density is not None:
        occ_dens = occupancy_density
    else:
        # Compute squared distances: (n_positions, n_bins)
        traj_dists_sq = (
            np.sum(positions**2, axis=1, keepdims=True)
            + np.sum(bin_centers**2, axis=1)
            - 2 * positions @ bin_centers.T
        )
        traj_weights = np.exp(-traj_dists_sq / two_sigma_sq)
        occ_dens = np.sum(traj_weights * dt_computed[:, np.newaxis], axis=0)

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
    # Use existing spikes_to_field function
    field = spikes_to_field(
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
    trajectory_bins: NDArray[np.int64] | None = None,
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
    trajectory_bins : NDArray[np.int64] | None, default=None
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
    spikes_to_field : Lower-level binning function (used by method="binned").
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
    >>> from neurospatial import compute_place_field
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
    if trajectory_bins is not None and len(trajectory_bins) != len(positions):
        raise ValueError(
            f"trajectory_bins must have same length as positions, "
            f"got {len(trajectory_bins)} and {len(positions)}"
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
                trajectory_bins=trajectory_bins,
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
                trajectory_bins=trajectory_bins,
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
