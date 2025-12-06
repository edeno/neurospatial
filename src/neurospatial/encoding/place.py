"""Place cell encoding analysis.

This module provides tools for analyzing place cell representations,
including place field computation, detection, and spatial metrics.

Spike → Field Conversion
------------------------
compute_place_field : Compute place field from spike train.
compute_directional_place_fields : Compute direction-conditioned place fields.
spikes_to_field : Convert spike train to spatial firing rate field.
DirectionalPlaceFields : Container for directional place field results.

Field Detection
---------------
detect_place_fields : Detect place fields using iterative peak-based approach.

Information-Theoretic Metrics
-----------------------------
skaggs_information : Compute Skaggs spatial information (bits/spike).
information_per_second : Compute spatial information in bits per second.
mutual_information : Compute mutual information between position and firing rate.

Sparsity/Selectivity Metrics
----------------------------
sparsity : Compute sparsity of spatial firing.
selectivity : Compute spatial selectivity (peak/mean rate).
spatial_coverage_single_cell : Compute fraction of environment covered by cell.

Field Geometry Metrics
----------------------
field_centroid : Compute firing-rate-weighted centroid.
field_size : Compute field size (area) in physical units.
field_shape_metrics : Compute geometric shape metrics for a place field.

Field Comparison Metrics
------------------------
field_stability : Compute stability between two firing rate maps.
field_shift_distance : Compute distance between field centroids.
compute_field_emd : Compute Earth Mover's Distance between rate distributions.
in_out_field_ratio : Compute ratio of in-field to out-of-field firing rate.
rate_map_coherence : Compute spatial coherence of a firing rate map.

Examples
--------
>>> import numpy as np
>>> from neurospatial import Environment
>>> from neurospatial.encoding.place import compute_place_field, detect_place_fields
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
>>>
>>> # Detect place fields
>>> fields = detect_place_fields(firing_rate, env)

See Also
--------
neurospatial.encoding.grid : Grid cell analysis.
neurospatial.encoding.head_direction : Head direction cell analysis.
neurospatial.encoding.border : Border/boundary cell analysis.
"""

from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats

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
    >>> from neurospatial.encoding.place import compute_directional_place_fields
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
    >>> from neurospatial.encoding.place import spikes_to_field
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
    >>> np.nanmean(field)  # Should be close to 5 Hz  # doctest: +ELLIPSIS
    np.float64(5.0...)
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


# =============================================================================
# Place Field Detection and Metrics
# =============================================================================
# The following functions were consolidated from metrics/place_fields.py


def detect_place_fields(
    firing_rate: NDArray[np.float64],
    env: Environment,
    *,
    threshold: float = 0.2,
    min_size: int | None = None,
    max_mean_rate: float = 10.0,
    detect_subfields: bool = True,
) -> list[NDArray[np.int64]]:
    """
    Detect place fields using iterative peak-based approach (neurocode method).

    This implements the field-standard algorithm used by neurocode (AyA Lab)
    with support for subfield discrimination and interneuron exclusion.

    Parameters
    ----------
    firing_rate : array, shape (n_bins,)
        Firing rate map (Hz) from neuron.
    env : Environment
        Spatial environment for binning.
    threshold : float, default=0.2
        Fraction of peak rate for field boundary detection (0-1).
        Standard value is 0.2 (20% of peak).
    min_size : int, optional
        Minimum number of bins for a valid field. If None, defaults to 9 bins.
    max_mean_rate : float, default=10.0
        Maximum mean firing rate (Hz). Neurons exceeding this are excluded
        as putative interneurons (vandermeerlab convention).
    detect_subfields : bool, default=True
        If True, recursively detect subfields within large fields using
        higher thresholds. This discriminates coalescent place fields.

    Returns
    -------
    fields : list of arrays
        List of place fields, where each field is a 1D array of bin indices
        (integers) belonging to that field. Empty list if no fields detected.

    Notes
    -----
    **Algorithm (neurocode approach)**:

    1. **Interneuron exclusion**: If mean rate > max_mean_rate, return no fields
    2. **Peak detection**: Find global maximum in firing rate map
    3. **Field segmentation**: Threshold at fraction of peak to define boundary
    4. **Connected component**: Extract bins above threshold connected to peak
    5. **Size filtering**: Discard fields smaller than min_size
    6. **Subfield recursion**: If detect_subfields=True, recursively apply
       higher thresholds (0.5, 0.7) to discriminate coalescent fields
    7. **Iteration**: Remove detected field bins and repeat until no peaks remain

    **Interneuron exclusion**: Following vandermeerlab convention, neurons with
    mean firing rate > 10 Hz are excluded as putative interneurons. Pyramidal
    cells (place cells) typically fire at 0.5-5 Hz.

    **Subfield detection**: When two place fields are close together, they may
    appear as a single broad field at low thresholds. Recursive thresholding
    at 0.5× and 0.7× peak discriminates true subfields.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.place import detect_place_fields
    >>> # Create synthetic place cell
    >>> positions = np.random.randn(5000, 2) * 20
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> firing_rate = np.zeros(env.n_bins)
    >>> # Add Gaussian place field at center
    >>> for i in range(env.n_bins):
    ...     dist = np.linalg.norm(env.bin_centers[i])
    ...     firing_rate[i] = 8.0 * np.exp(-(dist**2) / (2 * 3.0**2))
    >>> fields = detect_place_fields(firing_rate, env)
    >>> len(fields)  # doctest: +SKIP
    1

    See Also
    --------
    field_size : Compute area of place field
    field_centroid : Compute weighted center of mass
    skaggs_information : Spatial information content

    References
    ----------
    .. [1] neurocode repository (AyA Lab, Cornell): FindPlaceFields.m
    .. [2] Wilson & McNaughton (1993). Dynamics of hippocampal ensemble code
           for space. Science 261(5124).

    """
    # Validate inputs
    if firing_rate.shape[0] != env.n_bins:
        raise ValueError(
            f"firing_rate shape {firing_rate.shape} does not match "
            f"env.n_bins ({env.n_bins})"
        )

    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    # Set default min_size
    if min_size is None:
        min_size = 9  # Standard minimum (3×3 bins for 2D)

    # Interneuron exclusion
    mean_rate = np.nanmean(firing_rate)
    if mean_rate > max_mean_rate:
        return []  # Putative interneuron

    # Make a copy to modify during iteration
    rate_map = firing_rate.copy()
    fields = []

    # Iteratively find fields
    while True:
        # Handle all-NaN case
        if not np.any(np.isfinite(rate_map)):
            break  # No valid values remaining

        # Find peak
        peak_idx = int(np.nanargmax(rate_map))
        peak_rate = rate_map[peak_idx]

        # Check if peak is meaningful
        if peak_rate <= 0 or not np.isfinite(peak_rate):
            break

        # Threshold at fraction of peak
        threshold_rate = peak_rate * threshold

        # Find bins above threshold
        above_threshold = rate_map >= threshold_rate

        # Extract connected component containing peak
        field_bins = _extract_connected_component(peak_idx, above_threshold, env)

        # Check minimum size
        if len(field_bins) < min_size:
            # Remove this small field and continue
            rate_map[field_bins] = 0
            continue

        # Check for subfields (recursive thresholding)
        if detect_subfields and len(field_bins) > min_size * 2:
            # Try higher thresholds to discriminate subfields
            subfields = _detect_subfields(
                firing_rate[field_bins], field_bins, peak_rate, env, min_size
            )
            if len(subfields) > 1:
                # Found subfields - add them separately
                fields.extend(subfields)
            else:
                # No subfields - add as single field
                fields.append(field_bins)
        else:
            # Add field
            fields.append(field_bins)

        # Remove field bins from rate map
        rate_map[field_bins] = 0

        # Check if any meaningful peaks remain
        if np.nanmax(rate_map) < threshold_rate:
            break

    return fields


def _extract_connected_component_scipy(
    seed_idx: int,
    mask: NDArray[np.bool_],
    env: Environment,
) -> NDArray[np.int64]:
    """
    Extract connected component using scipy.ndimage.label (fast path for grids).

    This is the optimized path for grid-based environments, providing ~6× speedup
    over graph-based flood-fill by leveraging scipy's optimized N-D labeling.

    Parameters
    ----------
    seed_idx : int
        Starting bin index in active bin indexing.
    mask : array, shape (n_bins,)
        Boolean mask of candidate bins (active bin indexing).
    env : Environment
        Spatial environment (must be grid-based with grid_shape and active_mask).

    Returns
    -------
    component : array
        Bin indices in connected component (active bin indexing, sorted).

    Raises
    ------
    ValueError
        If environment does not have grid_shape or active_mask attributes.

    Notes
    -----
    This function only works for grid-based environments (RegularGridLayout,
    MaskedGridLayout, etc.). For non-grid environments (1D tracks, irregular
    graphs), use _extract_connected_component_graph() instead.

    The algorithm:
    1. Reshape flat mask to N-D grid using grid_shape
    2. Apply scipy.ndimage.label to find connected components
    3. Identify which component contains the seed
    4. Convert back to flat active bin indices

    """
    from scipy import ndimage

    # Validate environment has required attributes
    if env.grid_shape is None or env.active_mask is None:
        raise ValueError("scipy path requires grid_shape and active_mask")

    # Reshape flat mask (active bin indexing) to N-D grid (original grid indexing)
    grid_mask = np.zeros(env.grid_shape, dtype=bool)
    grid_mask[env.active_mask] = mask

    # Determine connectivity structure to match graph connectivity
    # Check if environment uses diagonal neighbors
    n_dims = len(env.grid_shape)
    if hasattr(env.layout, "_build_params_used"):
        params = env.layout._build_params_used
        connect_diagonal = params.get("connect_diagonal_neighbors", False)
    else:
        # Default: no diagonal connections (4-connected in 2D, 6-connected in 3D)
        connect_diagonal = False

    # Create connectivity structure for scipy
    if connect_diagonal:
        # Full connectivity (includes diagonals): connectivity = n_dims
        structure = ndimage.generate_binary_structure(n_dims, n_dims)
    else:
        # Axial connectivity only (no diagonals): connectivity = 1
        structure = ndimage.generate_binary_structure(n_dims, 1)

    # Label connected components in N-D grid
    labeled, _n_components = ndimage.label(grid_mask, structure=structure)

    # Convert seed from active bin index to grid coordinates
    # active_mask.ravel() gives flat indices of active bins in original grid
    active_flat_indices = np.where(env.active_mask.ravel())[0]
    seed_grid_flat_idx = active_flat_indices[seed_idx]
    seed_grid_coords = np.unravel_index(seed_grid_flat_idx, env.grid_shape)

    # Get label of component containing seed
    seed_label = labeled[seed_grid_coords]

    if seed_label == 0:
        # Seed not in any component (shouldn't happen if mask[seed_idx] is True)
        return np.array([seed_idx], dtype=np.int64)

    # Extract all grid positions in this component
    component_grid_mask = labeled == seed_label

    # Convert back to flat active bin indices
    # Find which active bins correspond to this component
    component_in_active_bins = component_grid_mask.ravel() & env.active_mask.ravel()
    component_grid_flat_indices = np.where(component_in_active_bins)[0]

    # Map from original grid flat indices to active bin indices
    component_bins = np.searchsorted(active_flat_indices, component_grid_flat_indices)

    return np.array(sorted(component_bins), dtype=np.int64)


def _extract_connected_component_graph(
    seed_idx: int,
    mask: NDArray[np.bool_],
    env: Environment,
) -> NDArray[np.int64]:
    """
    Extract connected component using graph-based flood-fill (fallback path).

    This is the fallback path for non-grid environments (1D tracks, irregular
    graphs) and works for any graph structure. It uses breadth-first search
    with direct graph.neighbors() queries.

    Parameters
    ----------
    seed_idx : int
        Starting bin index.
    mask : array, shape (n_bins,)
        Boolean mask of candidate bins.
    env : Environment
        Spatial environment for connectivity.

    Returns
    -------
    component : array
        Bin indices in connected component (sorted).

    Notes
    -----
    This is the original implementation, proven to be already optimal for
    sparse connected components on arbitrary graphs. Benchmarking showed
    this is faster than NetworkX's connected_components() due to avoiding
    subgraph creation overhead.

    """
    # Flood fill using graph connectivity (BFS)
    component_set = {seed_idx}
    frontier = deque([seed_idx])

    while frontier:
        current = frontier.popleft()
        # Get neighbors from graph
        neighbors = list(env.connectivity.neighbors(current))
        for neighbor in neighbors:
            if mask[neighbor] and neighbor not in component_set:
                component_set.add(neighbor)
                frontier.append(neighbor)

    return np.array(sorted(component_set), dtype=np.int64)


def _extract_connected_component(
    seed_idx: int,
    mask: NDArray[np.bool_],
    env: Environment,
) -> NDArray[np.int64]:
    """
    Extract connected component of bins from seed (routes to optimal method).

    Automatically selects the optimal algorithm based on environment type:
    - Grid environments (2D/3D): Uses scipy.ndimage.label (~6× faster)
    - Non-grid environments: Uses graph-based flood-fill

    Parameters
    ----------
    seed_idx : int
        Starting bin index.
    mask : array, shape (n_bins,)
        Boolean mask of candidate bins.
    env : Environment
        Spatial environment for connectivity.

    Returns
    -------
    component : array
        Bin indices in connected component (sorted).

    Notes
    -----
    The routing logic checks for grid-based environments using:
    - env.grid_shape is not None
    - len(env.grid_shape) >= 2 (2D or 3D grids)
    - env.active_mask is not None

    For grid environments, uses scipy.ndimage.label for ~6× speedup.
    For non-grid environments, uses graph-based flood-fill (already optimal).

    """
    # Check if scipy fast path is applicable
    if (
        env.grid_shape is not None
        and len(env.grid_shape) >= 2
        and env.active_mask is not None
    ):
        # Fast path: scipy.ndimage.label for grid environments
        return _extract_connected_component_scipy(seed_idx, mask, env)
    else:
        # Fallback path: graph-based flood-fill for non-grid environments
        return _extract_connected_component_graph(seed_idx, mask, env)


def _detect_subfields(
    field_rates: NDArray[np.float64],
    field_bins: NDArray[np.int64],
    peak_rate: float,
    env: Environment,
    min_size: int,
) -> list[NDArray[np.int64]]:
    """
    Recursively detect subfields using higher thresholds.

    Parameters
    ----------
    field_rates : array
        Firing rates within field bins.
    field_bins : array
        Bin indices of field.
    peak_rate : float
        Peak firing rate in field.
    env : Environment
        Spatial environment.
    min_size : int
        Minimum field size.

    Returns
    -------
    subfields : list of arrays
        List of subfield bin indices. If only one subfield found,
        returns list with original field.

    """
    # Try thresholds: 0.5 and 0.7 of peak
    subfield_thresholds = [0.5, 0.7]

    for thresh in subfield_thresholds:
        threshold_rate = peak_rate * thresh
        above_threshold = field_rates >= threshold_rate

        # Find connected components
        subfields = []
        remaining_mask = above_threshold.copy()

        while remaining_mask.any():
            # Find a seed
            seed_local_idx = np.where(remaining_mask)[0][0]
            seed_global_idx = field_bins[seed_local_idx]

            # Build mask in global coordinates
            global_mask = np.zeros(env.n_bins, dtype=bool)
            global_mask[field_bins[above_threshold]] = True

            # Extract component
            component_global = _extract_connected_component(
                seed_global_idx, global_mask, env
            )

            if len(component_global) >= min_size:
                subfields.append(component_global)

            # Remove from remaining mask
            for bin_idx in component_global:
                # Find local index
                local_indices = np.where(field_bins == bin_idx)[0]
                if len(local_indices) > 0:
                    remaining_mask[local_indices[0]] = False

        # If found multiple subfields, return them
        if len(subfields) > 1:
            return subfields

    # No subfields found
    return [field_bins]


def field_size(
    field_bins: NDArray[np.int64],
    env: EnvironmentProtocol,
) -> float:
    """
    Compute field size (area) in physical units.

    Parameters
    ----------
    field_bins : array
        Bin indices comprising the field.
    env : EnvironmentProtocol
        Spatial environment.

    Returns
    -------
    size : float
        Field area in squared physical units (e.g., cm²).

    Notes
    -----
    Size is computed as the sum of individual bin areas. For regular grids,
    each bin has area ≈ bin_size². For irregular graphs, area is estimated
    from Voronoi cell volumes.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.place import field_size
    >>> positions = np.random.randn(1000, 2) * 10
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> field_bins = np.array([0, 1, 2, 3, 4])
    >>> size = field_size(field_bins, env)
    >>> size > 0
    True

    """
    # Get bin sizes (property, not method)
    bin_sizes = env.bin_sizes

    # Sum areas of field bins
    total_size = np.sum(bin_sizes[field_bins])

    return float(total_size)


def field_centroid(
    firing_rate: NDArray[np.float64],
    field_bins: NDArray[np.int64],
    env: Environment,
    *,
    method: Literal["euclidean", "graph"] = "euclidean",
) -> NDArray[np.float64]:
    """
    Compute firing-rate-weighted centroid of place field.

    Parameters
    ----------
    firing_rate : array, shape (n_bins,)
        Firing rate map (Hz).
    field_bins : array
        Bin indices comprising the field.
    env : EnvironmentProtocol
        Spatial environment.
    method : {"euclidean", "graph"}, default "euclidean"
        Method for computing centroid:

        - ``"euclidean"``: Weighted mean in Euclidean space. Fast but may
          place centroid off-track for irregular geometries.
        - ``"graph"``: Weighted medoid using graph distances. Finds the bin
          within the field that minimizes weighted graph distance to all
          other field bins. Always on-track and respects maze geometry.

    Returns
    -------
    centroid : array, shape (n_dims,)
        Weighted center of mass in physical coordinates.

    Notes
    -----
    For ``method="euclidean"``, centroid is computed as the firing-rate-weighted
    mean position:

    .. math::

        \\mathbf{c} = \\frac{\\sum_i r_i \\mathbf{p}_i}{\\sum_i r_i}

    where :math:`r_i` is firing rate and :math:`\\mathbf{p}_i` is position
    of bin :math:`i`.

    For ``method="graph"``, the centroid is the bin that minimizes the
    weighted sum of graph distances:

    .. math::

        c = \\arg\\min_j \\sum_i r_i \\cdot d_G(i, j)

    where :math:`d_G(i, j)` is the shortest path distance in the environment
    graph between bins :math:`i` and :math:`j`. This approach is preferred
    for mazes and complex geometries where Euclidean distances can cross
    walls.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.place import field_centroid
    >>> positions = np.random.randn(1000, 2) * 10
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> firing_rate = np.random.rand(env.n_bins) * 5
    >>> field_bins = np.array([0, 1, 2, 3, 4])
    >>> centroid = field_centroid(firing_rate, field_bins, env)
    >>> centroid.shape
    (2,)

    Use graph-based centroid for maze environments:

    >>> centroid_graph = field_centroid(firing_rate, field_bins, env, method="graph")

    """
    import networkx as nx

    # Get positions and rates for field bins
    field_positions = env.bin_centers[field_bins]
    field_rates = firing_rate[field_bins]

    centroid: NDArray[np.float64]

    if method == "euclidean":
        # Compute weighted centroid in Euclidean space
        total_rate = np.sum(field_rates)
        if total_rate == 0:
            # Unweighted centroid if no firing
            centroid = field_positions.mean(axis=0)
        else:
            centroid = (
                np.sum(field_positions * field_rates[:, None], axis=0) / total_rate
            )

    elif method == "graph":
        # Compute weighted medoid using graph distances
        # Find bin that minimizes sum of (rate * graph_distance) to all other bins

        if len(field_bins) == 1:
            # Single bin - return its position
            centroid = field_positions[0]
        else:
            # Get graph and compute distances between field bins
            graph = env.connectivity

            # Compute weighted cost for each candidate bin
            min_cost = np.inf
            best_bin_idx = 0

            for j, candidate_bin in enumerate(field_bins):
                cost = 0.0
                for i, source_bin in enumerate(field_bins):
                    if i != j:
                        # Get graph distance
                        try:
                            dist = nx.shortest_path_length(
                                graph, source_bin, candidate_bin, weight="distance"
                            )
                        except nx.NetworkXNoPath:
                            # No path - use large penalty
                            dist = np.inf
                        cost += field_rates[i] * dist

                if cost < min_cost:
                    min_cost = cost
                    best_bin_idx = j

            centroid = field_positions[best_bin_idx]

    else:  # pragma: no cover
        # Runtime safety check (type system enforces at compile time)
        msg = f"Unknown method: {method}. Must be 'euclidean' or 'graph'."  # type: ignore[unreachable]
        raise ValueError(msg)

    return centroid


def skaggs_information(
    firing_rate: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    base: float = 2.0,
) -> float:
    """
    Compute Skaggs spatial information (bits per spike).

    Spatial information quantifies how much information each spike conveys
    about the animal's spatial location.

    Parameters
    ----------
    firing_rate : array, shape (n_bins,)
        Firing rate map (Hz).
    occupancy : array, shape (n_bins,)
        Occupancy probability (normalized to sum to 1).
    base : float, default=2.0
        Logarithm base. Use 2.0 for bits, np.e for nats.

    Returns
    -------
    information : float
        Spatial information in bits per spike (if base=2.0).
        Returns 0.0 if mean rate is zero.

    Notes
    -----
    **Formula (Skaggs et al. 1993)**:

    .. math::

        I = \\sum_i p_i \\frac{r_i}{\\bar{r}} \\log \\left( \\frac{r_i}{\\bar{r}} \\right)

    where :math:`p_i` is occupancy probability, :math:`r_i` is firing rate
    in bin :math:`i`, and :math:`\\bar{r}` is mean firing rate.

    **Interpretation**:
    - Place cells typically have 1-3 bits/spike
    - Higher values indicate more spatially selective firing
    - Zero information means uniform firing (no spatial information)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.place import skaggs_information
    >>> # Uniform firing → zero information
    >>> firing_rate = np.ones(100) * 3.0
    >>> occupancy = np.ones(100) / 100
    >>> info = skaggs_information(firing_rate, occupancy)
    >>> bool(np.abs(info) < 1e-6)  # Should be ~0
    True

    References
    ----------
    .. [1] Skaggs et al. (1993). An information-theoretic approach to
           deciphering the hippocampal code. NIPS.

    """
    # Normalize occupancy to probability
    occupancy_prob = occupancy / np.sum(occupancy)

    # Mean firing rate (use nansum to ignore NaN bins)
    mean_rate = np.nansum(occupancy_prob * firing_rate)

    if mean_rate == 0 or np.isnan(mean_rate):
        return 0.0

    # Compute information (suppress expected warnings from edge cases)
    # The np.log() can produce warnings for edge cases that are handled by the if-condition
    information = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(len(firing_rate)):
            # Skip NaN bins
            if (
                occupancy_prob[i] > 0
                and firing_rate[i] > 0
                and not np.isnan(firing_rate[i])
            ):
                ratio = firing_rate[i] / mean_rate
                information += occupancy_prob[i] * ratio * np.log(ratio) / np.log(base)

    # Ensure non-negative result (floating point errors can produce tiny negative values)
    # Mathematically, information is always >= 0, but uniform firing with floating
    # point arithmetic can produce values like -1e-16
    return float(max(0.0, information))


def sparsity(
    firing_rate: NDArray[np.float64],
    occupancy: NDArray[np.float64],
) -> float:
    """
    Compute sparsity of spatial firing (Skaggs et al. 1996).

    Sparsity measures what fraction of the environment elicits significant
    firing. Lower values indicate sparser, more selective place fields.

    Parameters
    ----------
    firing_rate : array, shape (n_bins,)
        Firing rate map (Hz).
    occupancy : array, shape (n_bins,)
        Occupancy probability (normalized to sum to 1).

    Returns
    -------
    sparsity : float
        Sparsity value in range [0, 1]. Lower values indicate sparser firing.

    Notes
    -----
    **Formula (Skaggs et al. 1996)**:

    .. math::

        S = \\frac{\\left( \\sum_i p_i r_i \\right)^2}{\\sum_i p_i r_i^2}

    where :math:`p_i` is occupancy probability and :math:`r_i` is firing rate.

    **Interpretation**:
    - Range: [0, 1]
    - Low sparsity (0.1-0.3): Sparse, selective place field
    - High sparsity (~1.0): Uniform firing throughout environment
    - Typical place cells: 0.1-0.3

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.place import sparsity
    >>> # Uniform firing → high sparsity
    >>> firing_rate = np.ones(100) * 5.0
    >>> occupancy = np.ones(100) / 100
    >>> spars = sparsity(firing_rate, occupancy)
    >>> spars > 0.9  # Close to 1
    True

    References
    ----------
    .. [1] Skaggs et al. (1996). Theta phase precession in hippocampal
           neuronal populations. Hippocampus 6(2).

    """
    # Normalize occupancy to probability
    occupancy_prob = occupancy / np.sum(occupancy)

    # Compute sparsity (use nansum to ignore NaN bins)
    numerator = np.nansum(occupancy_prob * firing_rate) ** 2
    denominator = np.nansum(occupancy_prob * firing_rate**2)

    if denominator == 0 or np.isnan(denominator):
        return 0.0

    sparsity_value = numerator / denominator

    # Clamp to [0, 1] to handle floating point precision issues
    # Mathematically, sparsity is always in [0, 1], but floating point
    # arithmetic can produce values like 1.0000000000000002
    return float(np.clip(sparsity_value, 0.0, 1.0))


def field_stability(
    rate_map_1: NDArray[np.float64],
    rate_map_2: NDArray[np.float64],
    *,
    method: Literal["pearson", "spearman"] = "pearson",
) -> float:
    """
    Compute stability between two firing rate maps (correlation).

    Stability quantifies how consistent spatial firing is across sessions
    or trial halves. Used to assess place field reliability and memory.

    Parameters
    ----------
    rate_map_1 : array, shape (n_bins,)
        First firing rate map (Hz).
    rate_map_2 : array, shape (n_bins,)
        Second firing rate map (Hz).
    method : {'pearson', 'spearman'}, default='pearson'
        Correlation method. Pearson for linear correlation, Spearman
        for rank-based correlation.

    Returns
    -------
    stability : float
        Correlation coefficient in range [-1, 1]. Higher values indicate
        more stable place fields.

    Notes
    -----
    **Interpretation**:
    - High stability (r > 0.7): Reliable, stable place field
    - Medium stability (0.3 < r < 0.7): Moderately stable
    - Low stability (r < 0.3): Unstable or remapped field

    **Edge Cases**:
    - Constant arrays (zero variance) return NaN (correlation undefined)
    - Arrays with <2 valid bins return 0.0

    **Pearson vs Spearman**: Pearson measures linear correlation of firing
    rates. Spearman measures rank correlation and is robust to outliers.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.place import field_stability
    >>> # Identical maps → perfect correlation
    >>> rate_map = np.random.rand(100) * 5
    >>> stability = field_stability(rate_map, rate_map, method="pearson")
    >>> bool(np.abs(stability - 1.0) < 1e-6)
    True

    References
    ----------
    .. [1] Wilson & McNaughton (1993). Dynamics of hippocampal ensemble code.
           Science 261(5124).

    """
    # Remove NaN values
    valid_mask = np.isfinite(rate_map_1) & np.isfinite(rate_map_2)
    map1_clean = rate_map_1[valid_mask]
    map2_clean = rate_map_2[valid_mask]

    if len(map1_clean) < 2:
        return 0.0

    # Check for constant arrays (zero variance) - correlation undefined
    if np.std(map1_clean) == 0 or np.std(map2_clean) == 0:
        return np.nan

    # Compute correlation
    if method == "pearson":
        correlation, _ = stats.pearsonr(map1_clean, map2_clean)
    elif method == "spearman":
        correlation, _ = stats.spearmanr(map1_clean, map2_clean)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pearson' or 'spearman'.")

    return float(correlation)


def rate_map_coherence(
    firing_rate: NDArray[np.float64],
    env: Environment,
    *,
    method: Literal["pearson", "spearman"] = "pearson",
) -> float:
    """
    Compute spatial coherence of a firing rate map.

    Spatial coherence measures the smoothness of spatial firing patterns by
    computing the correlation between each bin's firing rate and the mean rate
    of its spatial neighbors. High coherence indicates smooth, spatially
    structured firing. Low coherence indicates noisy or scattered firing.

    This metric was introduced by Muller & Kubie (1989) to assess the quality
    of place field representations.

    Parameters
    ----------
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Spatial firing rate map (Hz or spikes/second).
    env : EnvironmentProtocol
        Spatial environment containing bin centers and connectivity.
    method : {'pearson', 'spearman'}, optional
        Correlation method. Default is 'pearson'.
        - 'pearson': Pearson correlation (linear relationship)
        - 'spearman': Spearman correlation (monotonic relationship)

    Returns
    -------
    float
        Spatial coherence in range [-1, 1]. Returns NaN if:
        - All firing rates are zero or constant (no variance)
        - All rates are NaN
        - Insufficient valid bins after NaN removal

    Notes
    -----
    **Algorithm**:

    1. For each bin i, compute mean firing rate of neighbors: m_i = mean(r_j) for j in neighbors(i)
    2. Compute correlation between bin rates r_i and neighbor means m_i
    3. Coherence = corr(r, m)

    **Interpretation**:

    - **High coherence (> 0.7)**: Smooth, spatially structured firing (good place field)
    - **Medium coherence (0.3-0.7)**: Some spatial structure but with noise
    - **Low coherence (< 0.3)**: Noisy, poorly defined spatial firing

    **Graph-based approach**:

    This implementation uses `env.connectivity` to determine spatial neighbors,
    making it applicable to irregular environments and graphs with obstacles.
    For regular grids, results should match Muller & Kubie (1989) approach.

    References
    ----------
    Muller, R. U., & Kubie, J. L. (1989). The firing of hippocampal place cells
        predicts the future position of freely moving rats. Journal of Neuroscience,
        9(12), 4101-4110.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.place import rate_map_coherence
    >>>
    >>> # Create environment
    >>> positions = np.random.randn(5000, 2) * 20
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>>
    >>> # Smooth Gaussian field (high coherence)
    >>> firing_rate_smooth = np.zeros(env.n_bins)
    >>> for i in range(env.n_bins):
    ...     center = env.bin_centers[i]
    ...     dist = np.linalg.norm(center - np.array([0, 0]))
    ...     firing_rate_smooth[i] = 5.0 * np.exp(-(dist**2) / (2 * 8**2))
    >>>
    >>> coherence_smooth = rate_map_coherence(firing_rate_smooth, env)
    >>> print(f"Smooth field coherence: {coherence_smooth:.3f}")  # doctest: +SKIP
    Smooth field coherence: 0.850
    >>>
    >>> # Random noise (low coherence)
    >>> firing_rate_noisy = np.random.rand(env.n_bins) * 5.0
    >>> coherence_noisy = rate_map_coherence(firing_rate_noisy, env)
    >>> print(f"Noisy field coherence: {coherence_noisy:.3f}")  # doctest: +SKIP
    Noisy field coherence: 0.120

    See Also
    --------
    skaggs_information : Spatial information (bits/spike)
    sparsity : Spatial sparsity
    field_stability : Temporal stability of firing rate maps
    """
    # Validate inputs
    if firing_rate.shape != (env.n_bins,):
        raise ValueError(
            f"firing_rate.shape must be ({env.n_bins},), got {firing_rate.shape}"
        )

    # Remove NaN values for computing neighbor means
    # But track which bins are valid for final correlation
    valid_bins = np.isfinite(firing_rate)

    if not np.any(valid_bins):
        # All NaN
        return np.nan

    if np.all(firing_rate[valid_bins] == firing_rate[valid_bins][0]):
        # All values are identical (constant map, no variance)
        return np.nan

    # Compute mean of neighbors for each bin using neighbor_reduce
    # Use nanmean to handle NaN values in neighbors
    from neurospatial.ops.graph import neighbor_reduce

    neighbor_means = neighbor_reduce(
        env,
        firing_rate,
        op="mean",
        include_self=False,
    )

    # Now compute correlation between bin rates and their neighbor means
    # Only use bins where both the bin and its neighbor mean are valid
    valid_for_corr = valid_bins & np.isfinite(neighbor_means)

    if np.sum(valid_for_corr) < 2:
        # Need at least 2 points for correlation
        return np.nan

    bin_rates = firing_rate[valid_for_corr]
    neighbor_rate_means = neighbor_means[valid_for_corr]

    # Check for zero variance (would cause correlation to fail)
    if np.std(bin_rates) == 0 or np.std(neighbor_rate_means) == 0:
        return np.nan

    # Compute correlation
    if method == "pearson":
        coherence, _ = stats.pearsonr(bin_rates, neighbor_rate_means)
    elif method == "spearman":
        coherence, _ = stats.spearmanr(bin_rates, neighbor_rate_means)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pearson' or 'spearman'.")

    return float(coherence)


def selectivity(
    firing_rate: NDArray[np.float64],
    occupancy: NDArray[np.float64],
) -> float:
    """
    Compute spatial selectivity (peak rate / mean rate).

    Selectivity measures how spatially selective a cell's firing is. Higher
    values indicate the cell fires strongly in a small region and weakly
    elsewhere. A value of 1.0 indicates uniform firing throughout the
    environment.

    This metric is used in opexebo and provides a simple, interpretable measure
    of place field quality.

    Parameters
    ----------
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Firing rate map (Hz or spikes/second).
    occupancy : NDArray[np.float64], shape (n_bins,)
        Occupancy probability (normalized to sum to 1).

    Returns
    -------
    float
        Selectivity value, always >= 1.0. Returns NaN if:
        - Mean rate is zero (division by zero)
        - All firing rates are NaN
        Returns infinity if peak rate is positive but mean rate is zero.

    Notes
    -----
    **Formula**:

    .. math::

        S = \\frac{r_{\\text{peak}}}{\\bar{r}}

    where :math:`r_{\\text{peak}}` is the maximum firing rate and
    :math:`\\bar{r}` is the occupancy-weighted mean firing rate.

    **Interpretation**:

    - **Selectivity = 1.0**: Uniform firing (peak equals mean)
    - **Selectivity = 2-5**: Moderately selective place field
    - **Selectivity > 10**: Highly selective place field (fires in small region)

    **NaN handling**: NaN values in firing_rate are excluded from peak and mean
    calculations. Occupancy is renormalized to valid bins.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.place import selectivity
    >>>
    >>> # Uniform firing → selectivity = 1.0
    >>> firing_rate = np.ones(100) * 5.0
    >>> occupancy = np.ones(100) / 100
    >>> select = selectivity(firing_rate, occupancy)
    >>> print(f"Uniform: {select:.2f}")  # doctest: +SKIP
    Uniform: 1.00
    >>>
    >>> # Highly selective cell (fires in one bin)
    >>> firing_rate_selective = np.zeros(100)
    >>> firing_rate_selective[50] = 100.0
    >>> select_high = selectivity(firing_rate_selective, occupancy)
    >>> print(f"Selective: {select_high:.1f}")  # doctest: +SKIP
    Selective: 100.0

    See Also
    --------
    skaggs_information : Spatial information (bits/spike)
    sparsity : Spatial sparsity
    rate_map_coherence : Spatial coherence

    References
    ----------
    .. [1] opexebo package (Moser Lab):
           https://github.com/kavli-ntnu/opexebo
    """
    # Handle NaN values
    valid_mask = np.isfinite(firing_rate) & np.isfinite(occupancy)

    if not np.any(valid_mask):
        # All NaN
        return np.nan

    # Get valid values
    firing_rate_valid = firing_rate[valid_mask]
    occupancy_valid = occupancy[valid_mask]

    # Normalize occupancy to probability
    occupancy_prob = occupancy_valid / np.sum(occupancy_valid)

    # Peak firing rate
    peak_rate = np.max(firing_rate_valid)

    # Mean firing rate (occupancy-weighted)
    mean_rate = np.sum(occupancy_prob * firing_rate_valid)

    # Compute selectivity
    if mean_rate == 0:
        # Division by zero
        if peak_rate > 0:
            return np.inf
        else:
            return np.nan

    selectivity_value = peak_rate / mean_rate

    return float(selectivity_value)


def in_out_field_ratio(
    firing_rate: NDArray[np.float64],
    field_bins: NDArray[np.int64],
) -> float:
    """
    Compute ratio of in-field to out-of-field mean firing rate.

    This metric quantifies how much stronger firing is inside the place field
    compared to outside. Higher values indicate a more distinct place field with
    strong spatial selectivity.

    Parameters
    ----------
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Firing rate map (Hz or spikes/second).
    field_bins : NDArray[np.int64], shape (n_field_bins,)
        Indices of bins belonging to the place field.

    Returns
    -------
    float
        Ratio of in-field to out-of-field mean firing rate. Returns:
        - NaN if field is empty or covers all bins
        - NaN if out-of-field rate is zero and in-field rate is also zero
        - inf if out-of-field rate is zero but in-field rate is positive

    Notes
    -----
    **Formula**:

    .. math::

        R = \\frac{\\bar{r}_{\\text{in}}}{\\bar{r}_{\\text{out}}}

    where :math:`\\bar{r}_{\\text{in}}` is the mean firing rate inside the
    field and :math:`\\bar{r}_{\\text{out}}` is the mean rate outside.

    **Interpretation**:

    - **Ratio = 1.0**: No spatial selectivity (same firing inside and out)
    - **Ratio = 2-5**: Moderate place field (2-5× stronger inside)
    - **Ratio > 10**: Strong place field (10× or more stronger inside)

    **NaN handling**: NaN values in firing_rate are excluded from both
    in-field and out-of-field calculations.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.place import in_out_field_ratio
    >>>
    >>> # Strong place field (10× ratio)
    >>> firing_rate = np.ones(100) * 1.0
    >>> firing_rate[40:50] = 10.0  # Field bins have 10 Hz
    >>> field_bins = np.arange(40, 50)
    >>> ratio = in_out_field_ratio(firing_rate, field_bins)
    >>> print(f"Ratio: {ratio:.1f}")  # doctest: +SKIP
    Ratio: 10.0

    See Also
    --------
    selectivity : Peak rate / mean rate
    detect_place_fields : Detect place field bins

    References
    ----------
    .. [1] Jung et al. (1994). Comparison of spatial firing characteristics of
           units in dorsal and ventral hippocampus of the rat. J Neurosci 14(12).
    """
    # Validate field_bins
    if len(field_bins) == 0:
        return np.nan

    if len(field_bins) >= len(firing_rate):
        # Field covers entire environment
        return np.nan

    # Create mask for in-field and out-of-field bins
    in_field_mask = np.zeros(len(firing_rate), dtype=bool)
    in_field_mask[field_bins] = True

    # Handle NaN values
    valid_mask = np.isfinite(firing_rate)

    # In-field: bins in field AND valid
    in_valid = in_field_mask & valid_mask
    # Out-field: bins NOT in field AND valid
    out_valid = (~in_field_mask) & valid_mask

    if not np.any(in_valid) or not np.any(out_valid):
        return np.nan

    # Compute mean rates
    in_field_rate = np.mean(firing_rate[in_valid])
    out_field_rate = np.mean(firing_rate[out_valid])

    # Handle division by zero
    if out_field_rate == 0:
        if in_field_rate > 0:
            return np.inf
        else:
            return np.nan

    ratio = in_field_rate / out_field_rate

    return float(ratio)


def information_per_second(
    firing_rate: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    base: float = 2.0,
) -> float:
    """
    Compute spatial information in bits per second.

    This metric combines spatial information content (bits/spike) with the
    cell's firing rate to give information transmission rate. It measures
    how many bits of spatial information the cell conveys per second.

    Parameters
    ----------
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Firing rate map (Hz or spikes/second).
    occupancy : NDArray[np.float64], shape (n_bins,)
        Occupancy probability (normalized to sum to 1).
    base : float, default=2.0
        Logarithm base for information calculation. Use 2.0 for bits,
        np.e for nats.

    Returns
    -------
    float
        Information rate in bits/second (or nats/second if base=e).
        Returns NaN if firing rate or occupancy are all NaN.

    Notes
    -----
    **Formula**:

    .. math::

        I_{\\text{rate}} = I_{\\text{content}} \\times \\bar{r}

    where :math:`I_{\\text{content}}` is the Skaggs spatial information
    (bits/spike) and :math:`\\bar{r}` is the mean firing rate (spikes/second).

    **Interpretation**:

    - Combines "how much info per spike" with "how many spikes per second"
    - A cell can have high bits/spike but low bits/second if it fires rarely
    - Conversely, a cell with low selectivity but high rate can have high bits/second

    **Use case**: This metric favors cells that both fire frequently AND are
    spatially selective, making it useful for identifying the most informative
    place cells for population decoding.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.place import information_per_second
    >>>
    >>> # Highly selective but rare firing
    >>> firing_rate = np.zeros(100)
    >>> firing_rate[50] = 10.0  # 10 Hz in one bin, 0.1 Hz mean
    >>> occupancy = np.ones(100) / 100
    >>> info_rate = information_per_second(firing_rate, occupancy)
    >>> print(f"Info rate: {info_rate:.3f} bits/s")  # doctest: +SKIP
    Info rate: 0.664 bits/s

    See Also
    --------
    skaggs_information : Spatial information (bits/spike)
    mutual_information : Mutual information between position and firing

    References
    ----------
    .. [1] Markus et al. (1994). Interactions between location and task affect
           the spatial and directional firing of hippocampal neurons. J Neurosci 14(11).
    """
    # Compute Skaggs information (bits/spike)
    info_content = skaggs_information(firing_rate, occupancy, base=base)

    # Handle NaN values for mean rate calculation
    valid_mask = np.isfinite(firing_rate) & np.isfinite(occupancy)

    if not np.any(valid_mask):
        return np.nan

    firing_rate_valid = firing_rate[valid_mask]
    occupancy_valid = occupancy[valid_mask]

    # Normalize occupancy
    occupancy_prob = occupancy_valid / np.sum(occupancy_valid)

    # Mean firing rate (occupancy-weighted)
    mean_rate = np.sum(occupancy_prob * firing_rate_valid)

    # Information rate = bits/spike × spikes/second = bits/second
    info_rate = info_content * mean_rate

    return float(info_rate)


def mutual_information(
    firing_rate: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    base: float = 2.0,
) -> float:
    """
    Compute mutual information between position and firing rate.

    Mutual information quantifies how much knowing the animal's position
    reduces uncertainty about the neuron's firing rate. This is a fundamental
    information-theoretic measure of spatial coding.

    Parameters
    ----------
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Firing rate map (Hz or spikes/second).
    occupancy : NDArray[np.float64], shape (n_bins,)
        Occupancy probability (normalized to sum to 1).
    base : float, default=2.0
        Logarithm base for information calculation. Use 2.0 for bits,
        np.e for nats.

    Returns
    -------
    float
        Mutual information in bits (or nats if base=e). Returns NaN if
        firing rate or occupancy are all NaN or if mean rate is zero.

    Notes
    -----
    **Formula**:

    .. math::

        MI(X; R) = \\sum_x p(x) \\frac{r(x)}{\\bar{r}} \\log_2 \\frac{r(x)}{\\bar{r}}

    where :math:`p(x)` is occupancy probability, :math:`r(x)` is firing rate
    at position :math:`x`, and :math:`\\bar{r}` is mean firing rate.

    This is equivalent to:

    .. math::

        MI = I_{\\text{content}} \\times \\bar{r}

    where :math:`I_{\\text{content}}` is Skaggs information (bits/spike).

    **Relationship to other metrics**:

    - ``mutual_information`` = ``skaggs_information`` × ``mean_rate``
    - ``mutual_information`` = ``information_per_second``
    - MI is symmetric: MI(position; firing) = MI(firing; position)

    **Interpretation**:

    - **MI = 0**: Position and firing are independent (no place field)
    - **MI > 0**: Position provides information about firing
    - Higher MI indicates stronger spatial coding

    **Difference from Skaggs information**: Skaggs info is bits per spike,
    MI is total bits. A cell with high Skaggs but low firing rate will have
    lower MI than a moderately selective cell that fires frequently.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.place import mutual_information
    >>>
    >>> # Strong place field
    >>> firing_rate = np.ones(100) * 0.5
    >>> firing_rate[40:50] = 10.0
    >>> occupancy = np.ones(100) / 100
    >>> mi = mutual_information(firing_rate, occupancy)
    >>> print(f"MI: {mi:.3f} bits")  # doctest: +SKIP
    MI: 1.234 bits

    See Also
    --------
    skaggs_information : Spatial information (bits/spike)
    information_per_second : Information rate (equivalent to MI)
    sparsity : Sparsity measure

    References
    ----------
    .. [1] Skaggs et al. (1993). An information-theoretic approach to deciphering
           the hippocampal code. NIPS.
    .. [2] Markus et al. (1994). Interactions between location and task affect
           the spatial and directional firing of hippocampal neurons. J Neurosci 14(11).
    """
    # MI is mathematically equivalent to information_per_second
    # Just calling it with a different name for clarity
    return information_per_second(firing_rate, occupancy, base=base)


def spatial_coverage_single_cell(
    firing_rate: NDArray[np.float64],
    *,
    threshold: float = 0.1,
) -> float:
    """
    Compute fraction of environment where cell fires above threshold.

    This metric quantifies how much of the spatial environment a single cell
    covers with its firing. Lower values indicate more spatially selective
    place fields.

    Parameters
    ----------
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Firing rate map (Hz or spikes/second).
    threshold : float, default=0.1
        Minimum firing rate (Hz) to consider a bin as "covered".
        Standard values: 0.1 Hz (minimal activity) or 1.0 Hz (clear activity).

    Returns
    -------
    float
        Fraction of bins with firing rate > threshold, in range [0, 1].
        Returns NaN if all firing rates are NaN.

    Notes
    -----
    **Formula**:

    .. math::

        C = \\frac{\\sum_i \\mathbb{1}[r_i > \\theta]}{N}

    where :math:`r_i` is firing rate in bin :math:`i`, :math:`\\theta` is
    the threshold, and :math:`N` is the total number of bins.

    **Interpretation**:

    - **Coverage = 0.0**: Cell fires nowhere (no place field)
    - **Coverage = 0.1**: Cell fires in 10% of environment (highly selective)
    - **Coverage = 0.5**: Cell fires in half the environment (broad field)
    - **Coverage = 1.0**: Cell fires everywhere (no spatial selectivity)

    **Relationship to other metrics**:

    - Inverse of selectivity: high coverage → low selectivity
    - Complementary to sparsity: both measure spatial specificity
    - Unlike population_coverage, this is for a single cell

    **NaN handling**: NaN values in firing_rate are treated as bins with
    zero firing (below threshold).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.place import spatial_coverage_single_cell
    >>>
    >>> # Highly selective cell (fires in 10% of bins)
    >>> firing_rate = np.zeros(100)
    >>> firing_rate[40:50] = 5.0
    >>> coverage = spatial_coverage_single_cell(firing_rate, threshold=0.1)
    >>> print(f"Coverage: {coverage:.2f}")  # doctest: +SKIP
    Coverage: 0.10

    See Also
    --------
    sparsity : Sparsity measure (inverse of coverage)
    population_coverage : Fraction of environment covered by population
    selectivity : Peak / mean rate ratio

    References
    ----------
    .. [1] Muller et al. (1987). The effects of changes in the environment on
           the spatial firing of hippocampal complex-spike cells. J Neurosci 7(7).
    """
    # Handle NaN values (treat as below threshold)
    valid_mask = np.isfinite(firing_rate)

    if not np.any(valid_mask):
        return np.nan

    # Count bins above threshold
    n_above = np.sum(firing_rate[valid_mask] > threshold)

    # Total number of bins (including NaN bins as zeros)
    n_total = len(firing_rate)

    coverage = n_above / n_total

    return float(coverage)


def field_shape_metrics(
    firing_rate: NDArray[np.float64],
    field_bins: NDArray[np.int64],
    env: Environment,
) -> dict[str, float]:
    """
    Compute geometric shape metrics for a place field.

    Analyzes the spatial geometry of a place field, including eccentricity,
    orientation, and extent along principal axes. Useful for characterizing
    field morphology and detecting elongated or circular fields.

    Parameters
    ----------
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Firing rate map (Hz or spikes/second).
    field_bins : NDArray[np.int64], shape (n_field_bins,)
        Indices of bins belonging to the place field.
    env : Environment
        Spatial environment for bin positions. Must be 2D (n_dims == 2).

    Returns
    -------
    dict[str, float]
        Dictionary with shape metrics:
        - 'eccentricity': float in [0, 1], where 0 = circular, 1 = linear
        - 'major_axis_length': float, extent along major axis (same units as env)
        - 'minor_axis_length': float, extent along minor axis (same units as env)
        - 'orientation': float, angle of major axis in radians [-π/2, π/2]
        - 'area': float, spatial extent of field (number of bins)

        Returns dict with NaN values if field is empty or environment is not 2D.

    Notes
    -----
    **Eccentricity**: Computed from eigenvalues of the spatial covariance matrix
    of rate-weighted bin positions:

    .. math::

        e = \\sqrt{1 - \\frac{\\lambda_{\\text{min}}}{\\lambda_{\\text{max}}}}

    where :math:`\\lambda_{\\text{min}}` and :math:`\\lambda_{\\text{max}}` are
    the smallest and largest eigenvalues.

    **Interpretation**:

    - **Eccentricity = 0**: Circular field (equal extent in all directions)
    - **Eccentricity = 0.5**: Moderately elongated field
    - **Eccentricity → 1**: Highly elongated, linear field

    **Orientation**: Angle of the major axis (eigenvector corresponding to
    largest eigenvalue) relative to the first spatial dimension. Useful for
    detecting field alignment with environmental features.

    **2D only**: This implementation currently supports only 2D environments.
    3D shape analysis would require different metrics (e.g., sphericity).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.place import field_shape_metrics
    >>>
    >>> # Create 2D environment
    >>> data = np.random.randn(1000, 2) * 20
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>>
    >>> # Create elongated field along x-axis
    >>> firing_rate = np.zeros(env.n_bins)
    >>> # Find bins near y=0, x in [0, 20]
    >>> centers = env.bin_centers
    >>> elongated_mask = (
    ...     (np.abs(centers[:, 1]) < 2) & (centers[:, 0] > 0) & (centers[:, 0] < 20)
    ... )
    >>> field_bins = np.where(elongated_mask)[0]
    >>> firing_rate[field_bins] = 10.0
    >>>
    >>> # Compute shape metrics
    >>> shape = field_shape_metrics(firing_rate, field_bins, env)
    >>> print(f"Eccentricity: {shape['eccentricity']:.2f}")  # doctest: +SKIP
    Eccentricity: 0.87

    See Also
    --------
    field_centroid : Compute field center of mass
    field_size : Compute field area

    References
    ----------
    .. [1] Muller & Kubie (1989). The effects of changes in the environment on
           the spatial firing of hippocampal complex-spike cells. J Neurosci 9(1).
    .. [2] Knierim et al. (1995). Place cells, head direction cells, and the
           learning of landmark stability. J Neurosci 15(3).
    """
    # Initialize with NaN defaults
    result = {
        "eccentricity": np.nan,
        "major_axis_length": np.nan,
        "minor_axis_length": np.nan,
        "orientation": np.nan,
        "area": np.nan,
    }

    # Validate inputs
    if len(field_bins) == 0:
        return result

    if env.n_dims != 2:
        warnings.warn(
            f"field_shape_metrics currently only supports 2D environments, got {env.n_dims}D. "
            "Returning NaN values.",
            UserWarning,
            stacklevel=2,
        )
        return result

    # Get bin positions for field
    positions = env.bin_centers[field_bins]  # shape (n_field_bins, 2)

    # Get firing rates for weighting
    rates = firing_rate[field_bins]

    # Handle NaN values
    valid_mask = np.isfinite(rates)
    if not np.any(valid_mask):
        return result

    positions_valid = positions[valid_mask]
    rates_valid = rates[valid_mask]

    # Normalize rates to use as weights
    rate_weights = rates_valid / np.sum(rates_valid)

    # Compute rate-weighted centroid
    centroid = np.sum(rate_weights[:, np.newaxis] * positions_valid, axis=0)

    # Compute rate-weighted covariance matrix
    centered = positions_valid - centroid  # shape (n_valid, 2)
    cov = np.zeros((2, 2))
    for i in range(len(centered)):
        cov += rate_weights[i] * np.outer(centered[i], centered[i])

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (largest first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Extract metrics
    lambda_max = eigenvalues[0]
    lambda_min = eigenvalues[1]

    # Eccentricity
    eccentricity = np.sqrt(1 - lambda_min / lambda_max) if lambda_max > 0 else 0.0

    # Axis lengths (2 standard deviations = ~95% of data)
    major_axis = 2 * np.sqrt(lambda_max)
    minor_axis = 2 * np.sqrt(lambda_min)

    # Orientation (angle of major axis)
    # eigenvectors[:, 0] is the major axis direction
    orientation = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    # Area (number of bins)
    area = float(len(field_bins))

    result.update(
        {
            "eccentricity": float(eccentricity),
            "major_axis_length": float(major_axis),
            "minor_axis_length": float(minor_axis),
            "orientation": float(orientation),
            "area": area,
        }
    )

    return result


def field_shift_distance(
    firing_rate_1: NDArray[np.float64],
    field_bins_1: NDArray[np.int64],
    env_1: Environment,
    firing_rate_2: NDArray[np.float64],
    field_bins_2: NDArray[np.int64],
    env_2: Environment,
    *,
    use_geodesic: bool = False,
) -> float:
    """
    Compute distance between field centroids across sessions/environments.

    This metric quantifies how much a place field has shifted in position between
    two recording sessions or environments. Useful for detecting remapping,
    field stability, and spatial representation changes.

    Parameters
    ----------
    firing_rate_1 : NDArray[np.float64], shape (n_bins_1,)
        Firing rate map from first session (Hz or spikes/second).
    field_bins_1 : NDArray[np.int64], shape (n_field_bins_1,)
        Indices of bins belonging to place field in first session.
    env_1 : Environment
        Spatial environment for first session.
    firing_rate_2 : NDArray[np.float64], shape (n_bins_2,)
        Firing rate map from second session (Hz or spikes/second).
    field_bins_2 : NDArray[np.int64], shape (n_field_bins_2,)
        Indices of bins belonging to place field in second session.
    env_2 : Environment
        Spatial environment for second session.
    use_geodesic : bool, default=False
        If True, compute geodesic distance (shortest path along connectivity graph)
        instead of Euclidean distance. Requires env_1 and env_2 to be the same
        environment or aligned environments with compatible connectivity.
        Geodesic distance respects barriers and boundaries in the environment.

    Returns
    -------
    float
        Distance between field centroids in spatial units (same units as environment).
        Returns NaN if either field is empty or centroid calculation fails.

    Notes
    -----
    **Euclidean distance** (use_geodesic=False):

    Computes straight-line distance between rate-weighted field centroids:

    .. math::

        d = \\|c_1 - c_2\\|_2

    where :math:`c_1` and :math:`c_2` are the centroids in continuous space.

    **Geodesic distance** (use_geodesic=True):

    Computes shortest path distance along environment connectivity graph,
    respecting barriers and boundaries:

    .. math::

        d_{\\text{geo}} = \\min_{\\text{path}} \\sum_{\\text{edges}} w_e

    This is more appropriate for complex environments with barriers (e.g., mazes,
    multi-room environments) where straight-line distance is misleading.

    **Cross-session alignment**:

    For comparing across sessions, environments should be aligned (e.g., using
    estimate_transform and apply_transform_to_environment) to account for
    camera shifts, rotation, or scaling. If environments are not aligned,
    distance will include alignment error.

    **Use cases**:

    - **Rate remapping**: Same field location (distance ~ 0), different rates
    - **Global remapping**: Different field location (distance > 0)
    - **Field stability**: Measure distance across repeated sessions

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.place import (
    ...     field_shift_distance,
    ...     detect_place_fields,
    ... )
    >>>
    >>> # Create two environments (same layout)
    >>> data = np.random.randn(1000, 2) * 20
    >>> env1 = Environment.from_samples(data, bin_size=2.0, name="session1")
    >>> env2 = Environment.from_samples(data, bin_size=2.0, name="session2")
    >>>
    >>> # Create place field in env1
    >>> firing_rate_1 = np.zeros(env1.n_bins)
    >>> centers1 = env1.bin_centers
    >>> field_mask_1 = np.linalg.norm(centers1 - [10, 10], axis=1) < 5
    >>> field_bins_1 = np.where(field_mask_1)[0]
    >>> firing_rate_1[field_bins_1] = 10.0
    >>>
    >>> # Create shifted field in env2 (shifted by ~7 units)
    >>> firing_rate_2 = np.zeros(env2.n_bins)
    >>> centers2 = env2.bin_centers
    >>> field_mask_2 = np.linalg.norm(centers2 - [15, 15], axis=1) < 5
    >>> field_bins_2 = np.where(field_mask_2)[0]
    >>> firing_rate_2[field_bins_2] = 10.0
    >>>
    >>> # Compute shift distance
    >>> shift = field_shift_distance(
    ...     firing_rate_1,
    ...     field_bins_1,
    ...     env1,
    ...     firing_rate_2,
    ...     field_bins_2,
    ...     env2,
    ... )
    >>> print(f"Field shifted by: {shift:.1f} units")  # doctest: +SKIP
    Field shifted by: 7.1 units

    See Also
    --------
    field_centroid : Compute field center of mass
    field_stability : Correlation-based stability measure
    Environment.distance_between : Geodesic distance between bins

    References
    ----------
    .. [1] Leutgeb et al. (2005). Independent codes for spatial and episodic memory
           in hippocampal neuronal ensembles. Science 309(5734).
    .. [2] Colgin et al. (2008). Understanding memory through hippocampal remapping.
           Trends Neurosci 31(9).
    """
    # Compute centroids for both fields
    centroid_1 = field_centroid(firing_rate_1, field_bins_1, env_1)
    centroid_2 = field_centroid(firing_rate_2, field_bins_2, env_2)

    # Check for NaN centroids
    if np.any(np.isnan(centroid_1)) or np.any(np.isnan(centroid_2)):
        return np.nan

    if use_geodesic:
        # Geodesic distance using environment connectivity
        # Validate that centroids fall within environment bounds
        bin_1 = env_1.bin_at(centroid_1.reshape(1, -1))[0]
        bin_2 = env_2.bin_at(centroid_2.reshape(1, -1))[0]

        # Check if bins are valid (centroids in bounds)
        if bin_1 < 0 or bin_2 < 0:
            warnings.warn(
                "One or both centroids fall outside environment bounds. "
                "Cannot compute geodesic distance. Returning NaN.",
                UserWarning,
                stacklevel=2,
            )
            return np.nan

        # Check if environments are compatible for geodesic distance
        if env_1 is not env_2 and env_1.n_bins != env_2.n_bins:
            # Different environment objects - check if they have same bins
            warnings.warn(
                f"Environments have different number of bins ({env_1.n_bins} vs {env_2.n_bins}). "
                "Geodesic distance requires compatible environments. Falling back to Euclidean distance.",
                UserWarning,
                stacklevel=2,
            )
            # Fall back to Euclidean
            distance = float(np.linalg.norm(centroid_1 - centroid_2))
            return distance

        # Compute geodesic distance using centroids (coordinates), not bin indices
        try:
            geodesic_dist = cast("EnvironmentProtocol", env_1).distance_between(
                centroid_1, centroid_2
            )
            return float(geodesic_dist)
        except Exception as e:
            warnings.warn(
                f"Failed to compute geodesic distance: {e}. Falling back to Euclidean distance.",
                UserWarning,
                stacklevel=2,
            )
            # Fall back to Euclidean
            distance = float(np.linalg.norm(centroid_1 - centroid_2))
            return distance
    else:
        # Euclidean distance between centroids
        distance = float(np.linalg.norm(centroid_1 - centroid_2))
        return distance


def compute_field_emd(
    firing_rate_1: NDArray[np.float64],
    firing_rate_2: NDArray[np.float64],
    env: Environment,
    *,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    normalize: bool = True,
) -> float:
    """Compute Earth Mover's Distance (EMD) between two firing rate distributions.

    The Earth Mover's Distance (also known as Wasserstein distance or optimal
    transport distance) measures the minimum cost to transform one distribution
    into another. Unlike simple measures like correlation or mean squared error,
    EMD respects the spatial structure of the environment.

    This implementation supports both Euclidean distance (straight-line) and
    geodesic distance (shortest path through the environment's connectivity graph).
    Geodesic EMD is particularly useful for complex environments with barriers,
    mazes, or non-convex layouts where Euclidean distance is misleading.

    Parameters
    ----------
    firing_rate_1 : NDArray[np.float64], shape (n_bins,)
        First firing rate distribution across spatial bins.
    firing_rate_2 : NDArray[np.float64], shape (n_bins,)
        Second firing rate distribution across spatial bins.
    env : Environment
        Spatial environment defining bin positions and connectivity.
    metric : str, default="euclidean"
        Distance metric to use. Options:
        - "euclidean": Straight-line distance between bin centers
        - "geodesic": Shortest path distance through connectivity graph
    normalize : bool, default=True
        If True, normalize distributions to sum to 1.0 before computing EMD.
        If False, use raw firing rates (distributions must already sum to equal values).

    Returns
    -------
    emd : float
        Earth Mover's Distance between the two distributions.
        - For normalized distributions: unitless distance in [0, ∞)
        - For unnormalized distributions: cost in units of (rate × distance)
        - Returns NaN if distributions cannot be computed (e.g., all zeros)

    Raises
    ------
    ValueError
        If firing_rate arrays have different lengths.
        If firing_rate arrays don't match env.n_bins.
        If metric is not "euclidean" or "geodesic".
        If normalize=False and distributions have different total mass.

    Warns
    -----
    UserWarning
        If distributions contain NaN values (they will be set to zero).
        If normalized distributions have no mass (all zeros or NaN).

    See Also
    --------
    field_shift_distance : Distance between field centroids.
    population_vector_correlation : Correlation between rate distributions.

    Notes
    -----
    The Earth Mover's Distance is computed by solving the optimal transport problem:

    .. math::
        EMD(P, Q) = \\min_{T} \\sum_{i,j} T_{ij} \\cdot d_{ij}

    where:
    - :math:`P` and :math:`Q` are the two distributions
    - :math:`T_{ij}` is the amount of mass transported from bin :math:`i` to bin :math:`j`
    - :math:`d_{ij}` is the distance between bins :math:`i` and :math:`j`

    The optimization is subject to constraints ensuring mass is conserved.

    **Metric choice:**

    - **Euclidean**: Fast, works for any environment, but ignores barriers and walls.
      Use for simple open fields or when computational speed is critical.

    - **Geodesic**: Slower, respects environment structure (barriers, walls, connectivity).
      Use for complex environments like mazes, multi-room layouts, or non-convex arenas.

    **Interpretation:**

    - EMD = 0: Identical distributions
    - Small EMD: Distributions are similar and nearby in space
    - Large EMD: Distributions are different or far apart in space

    EMD is particularly useful for:
    - Quantifying remapping between sessions
    - Measuring population drift over time
    - Comparing spatial representations across environments
    - Assessing stability of place field populations

    **Computational complexity:**

    - Euclidean: O(n²) for distance matrix + O(n³) for optimization
    - Geodesic: O(n² log n) for all-pairs shortest paths + O(n³) for optimization

    For large environments (n_bins > 1000), consider subsampling or using
    approximate methods.

    References
    ----------
    .. [1] Rubner, Y., Tomasi, C., & Guibas, L. J. (2000). The Earth Mover's
           Distance as a Metric for Image Retrieval. International Journal of
           Computer Vision, 40(2), 99-121.
    .. [2] Villani, C. (2009). Optimal Transport: Old and New. Springer.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.place import compute_field_emd
    >>>
    >>> # Create environment
    >>> data = np.random.randn(1000, 2) * 20
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>>
    >>> # Create two similar firing rate distributions
    >>> field1 = np.exp(-0.1 * np.linalg.norm(env.bin_centers - [0, 0], axis=1) ** 2)
    >>> field2 = np.exp(-0.1 * np.linalg.norm(env.bin_centers - [5, 0], axis=1) ** 2)
    >>>
    >>> # Compute EMD with Euclidean distance
    >>> emd_euclidean = compute_field_emd(field1, field2, env, metric="euclidean")
    >>> print(f"Euclidean EMD: {emd_euclidean:.3f}")  # doctest: +SKIP
    >>>
    >>> # Compute EMD with geodesic distance (respects environment structure)
    >>> emd_geodesic = compute_field_emd(field1, field2, env, metric="geodesic")
    >>> print(f"Geodesic EMD: {emd_geodesic:.3f}")  # doctest: +SKIP
    >>>
    >>> # For open fields, Euclidean and geodesic should be similar
    >>> # For mazes or complex environments, geodesic can be much larger
    """
    from scipy.optimize import linprog

    # Validate inputs
    if len(firing_rate_1) != len(firing_rate_2):
        raise ValueError(
            f"firing_rate arrays must have same length, got {len(firing_rate_1)} and {len(firing_rate_2)}"
        )

    if len(firing_rate_1) != env.n_bins:
        raise ValueError(
            f"firing_rate arrays must match env.n_bins ({env.n_bins}), got {len(firing_rate_1)}"
        )

    if metric not in ("euclidean", "geodesic"):
        raise ValueError(f"metric must be 'euclidean' or 'geodesic', got '{metric}'")

    # Handle NaN values
    firing_rate_1 = firing_rate_1.copy()
    firing_rate_2 = firing_rate_2.copy()

    if np.any(~np.isfinite(firing_rate_1)) or np.any(~np.isfinite(firing_rate_2)):
        warnings.warn(
            "Firing rate distributions contain NaN values. Setting to zero.",
            UserWarning,
            stacklevel=2,
        )
        firing_rate_1[~np.isfinite(firing_rate_1)] = 0.0
        firing_rate_2[~np.isfinite(firing_rate_2)] = 0.0

    # Normalize distributions if requested
    if normalize:
        sum1 = np.sum(firing_rate_1)
        sum2 = np.sum(firing_rate_2)

        if sum1 == 0 or sum2 == 0:
            warnings.warn(
                "One or both distributions have zero total mass. Returning NaN.",
                UserWarning,
                stacklevel=2,
            )
            return np.nan

        firing_rate_1 = firing_rate_1 / sum1
        firing_rate_2 = firing_rate_2 / sum2
    else:
        # Check that unnormalized distributions have equal total mass
        sum1 = np.sum(firing_rate_1)
        sum2 = np.sum(firing_rate_2)
        if not np.isclose(sum1, sum2, rtol=1e-6):
            raise ValueError(
                f"Unnormalized distributions must have equal total mass. "
                f"Got {sum1:.6f} and {sum2:.6f}. Set normalize=True to auto-normalize."
            )

    # Filter to bins with non-zero mass in either distribution
    # This reduces problem size significantly for sparse fields
    nonzero_mask = (firing_rate_1 > 0) | (firing_rate_2 > 0)
    n_active = np.sum(nonzero_mask)

    if n_active == 0:
        # Both distributions are all zeros
        return 0.0

    if n_active == 1:
        # Only one bin has mass - EMD is zero if same bin, undefined otherwise
        # Since distributions are normalized, must be same bin
        return 0.0

    # Get active bins and their distributions
    active_bins = np.where(nonzero_mask)[0]
    p = firing_rate_1[nonzero_mask]
    q = firing_rate_2[nonzero_mask]

    # Compute distance matrix between active bins
    if metric == "euclidean":
        # Euclidean distance between bin centers
        positions = env.bin_centers[active_bins]
        n = len(active_bins)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d: float = float(np.linalg.norm(positions[i] - positions[j]))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

    else:  # metric == "geodesic"
        # Geodesic distance using environment's connectivity graph
        n = len(active_bins)
        dist_matrix = np.zeros((n, n))

        # Count disconnected pairs for aggregated warning
        disconnected_count = 0

        for i in range(n):
            for j in range(i + 1, n):
                try:
                    # Use bin centers (coordinates), not bin indices
                    d = float(
                        cast("EnvironmentProtocol", env).distance_between(
                            env.bin_centers[active_bins[i]],
                            env.bin_centers[active_bins[j]],
                        )
                    )
                    if np.isnan(d) or np.isinf(d):
                        # No path exists - fall back to Euclidean
                        d = float(
                            np.linalg.norm(
                                env.bin_centers[active_bins[i]]
                                - env.bin_centers[active_bins[j]]
                            )
                        )
                        disconnected_count += 1
                except Exception:
                    # Fallback to Euclidean on any error
                    d = float(
                        np.linalg.norm(
                            env.bin_centers[active_bins[i]]
                            - env.bin_centers[active_bins[j]]
                        )
                    )
                    disconnected_count += 1

                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Warn once about disconnected pairs instead of spamming
        if disconnected_count > 0:
            warnings.warn(
                f"Found {disconnected_count} disconnected bin pairs out of {n * (n - 1) // 2} total pairs. "
                f"Using Euclidean distance for disconnected pairs.",
                UserWarning,
                stacklevel=2,
            )

    # Solve the optimal transport problem using linear programming
    # Variables: T[i,j] = mass transported from bin i (source) to bin j (target)
    # Objective: minimize sum(T[i,j] * dist[i,j])
    # Constraints:
    #   - sum_j T[i,j] = p[i]  (all mass from source i is transported)
    #   - sum_i T[i,j] = q[j]  (all mass to target j is received)
    #   - T[i,j] >= 0

    n = len(p)

    # Flatten distance matrix for objective function
    c = dist_matrix.flatten()

    # Equality constraints: Ax = b
    # Row constraints: sum over j of T[i,j] = p[i]
    # Column constraints: sum over i of T[i,j] = q[j]
    a_eq = np.zeros((2 * n, n * n))

    # Row constraints (source)
    for i in range(n):
        for j in range(n):
            a_eq[i, i * n + j] = 1.0

    # Column constraints (target)
    for j in range(n):
        for i in range(n):
            a_eq[n + j, i * n + j] = 1.0

    b_eq = np.concatenate([p, q])

    # Solve linear program
    result = linprog(
        c,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0, None),
        method="highs",
    )

    if not result.success:
        warnings.warn(
            f"Optimal transport optimization failed: {result.message}. Returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return np.nan

    # EMD is the optimal cost
    emd = float(result.fun)
    return emd


# ruff: noqa: RUF022  - Intentionally organized into groups with comments
__all__ = [
    # Classes
    "DirectionalPlaceFields",
    # Spike → field conversion
    "compute_directional_place_fields",
    "compute_place_field",
    "spikes_to_field",
    # Field detection
    "detect_place_fields",
    # Information-theoretic metrics
    "skaggs_information",
    "information_per_second",
    "mutual_information",
    # Sparsity/selectivity metrics
    "sparsity",
    "selectivity",
    "spatial_coverage_single_cell",
    # Field geometry metrics
    "field_centroid",
    "field_size",
    "field_shape_metrics",
    # Field comparison metrics
    "field_stability",
    "field_shift_distance",
    "compute_field_emd",
    "in_out_field_ratio",
    "rate_map_coherence",
]
