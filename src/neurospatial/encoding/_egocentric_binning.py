"""Binning layer for egocentric encoding (object-vector cells).

This module converts spike trains and trajectory data into discrete spike counts
and occupancy arrays in egocentric polar coordinates (distance, direction to
nearest object).

The key difference from spatial binning (_binning.py) is that:
- Spatial binning: bins by *where the animal was*
- Egocentric binning: bins by *distance/direction to nearest object*

The functions in this module handle:
1. Computation of egocentric coordinates (distance, bearing to nearest object)
2. Egocentric occupancy computation (time spent at each distance/direction bin)
3. Spike binning based on egocentric coordinates at spike time
4. Batch processing of multiple neurons with joblib parallelization

Output shapes:
- Spike counts (single neuron): (n_bins,)
- Spike counts (batch): (n_neurons, n_bins)
- Occupancy: (n_bins,) - always shared across neurons
- env: Environment in polar coordinates

The binning layer is separated from smoothing to allow:
- Reusing occupancy across multiple neurons
- Precomputing egocentric coordinates for efficiency
- Future JAX implementations with different parallelization strategies

Coordinate Conventions
----------------------
**Egocentric direction** (0=ahead, pi/2=left, -pi/2=right, +/-pi=behind):
- Uses animal-centered reference frame
- Matches the convention in ``neurospatial.ops.egocentric``
- Direction bins span [-pi, pi] (full circle)

**Distance**:
- Euclidean (default): straight-line distance to nearest object
- Geodesic: path distance respecting environment boundaries (requires env)

Notes
-----
Unlike spatial binning, egocentric binning creates a *new* Environment
(``env``) in polar coordinates. This environment has bins arranged
in a (distance, direction) grid that is flattened to 1D for consistency
with other encoding modules.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial.encoding._validation import validate_times as _validate_times
from neurospatial.ops.egocentric import compute_egocentric_bearing

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.environment.polar import EgocentricPolarEnvironment

__all__ = [
    "bin_egocentric_spike_train",
    "bin_egocentric_spike_trains",
    "compute_egocentric_occupancy",
    "normalize_object_positions",
]


def normalize_object_positions(
    object_positions: NDArray[np.float64] | Sequence[float],
) -> NDArray[np.float64]:
    """Normalize object positions to canonical (n_objects, 2) format.

    Converts common input formats to a consistent 2D array representation
    for egocentric encoding functions.

    Parameters
    ----------
    object_positions : array-like
        Object positions in one of these formats:

        - 1D array of length 2 (single object) → reshaped to (1, 2)
        - 2D array of shape (n_objects, 2) (canonical format) → returned as-is
        - List/tuple of length 2 (single object, e.g., ``[x, y]``) → converted
          to (1, 2) array

    Returns
    -------
    ndarray, shape (n_objects, 2)
        Object positions as 2D float64 array. Always at least shape (1, 2).

    Raises
    ------
    ValueError
        If input has unexpected shape or dimensions.

    Examples
    --------
    Single object (common user input):

    >>> import numpy as np
    >>> from neurospatial.encoding._egocentric_binning import normalize_object_positions
    >>> obj = [50.0, 50.0]  # Plain list
    >>> normalized = normalize_object_positions(obj)
    >>> normalized.shape
    (1, 2)
    >>> normalized
    array([[50., 50.]])

    Single object as 1D array:

    >>> obj = np.array([50.0, 50.0])
    >>> normalize_object_positions(obj).shape
    (1, 2)

    Multiple objects (already canonical):

    >>> objs = np.array([[50.0, 50.0], [25.0, 75.0]])
    >>> normalize_object_positions(objs).shape
    (2, 2)

    Notes
    -----
    This normalization happens at the entry point of egocentric encoding
    functions, ensuring consistent internal handling regardless of how the
    user provides object data. Single-object inputs like ``[x, y]`` are a
    common pattern in neuroscience experiments with a single landmark.
    """
    arr = np.asarray(object_positions, dtype=np.float64)

    # 1D array of length 2: single object [x, y]
    if arr.ndim == 1:
        if len(arr) != 2:
            raise ValueError(
                f"1D object_positions must have length 2 (single object [x, y]), "
                f"got length {len(arr)}.\n"
                f"For multiple objects, pass a 2D array with shape (n_objects, 2)."
            )
        return arr.reshape(1, 2)

    # 2D array: validate shape
    if arr.ndim == 2:
        if arr.shape[1] != 2:
            raise ValueError(
                f"object_positions must have shape (n_objects, 2), "
                f"got shape {arr.shape}.\n"
                f"Each row should be [x, y] coordinates."
            )
        if arr.shape[0] == 0:
            raise ValueError(
                "object_positions cannot be empty. "
                "Provide at least one object position."
            )
        return arr

    raise ValueError(
        f"object_positions must be 1D (single object) or 2D (multiple objects), "
        f"got {arr.ndim}D array with shape {arr.shape}."
    )


def _compute_egocentric_coords(
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    object_positions: NDArray[np.float64],
    *,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    env: Environment | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute egocentric coordinates to nearest object at each timepoint.

    Parameters
    ----------
    positions : ndarray, shape (n_time, 2)
        Animal positions in allocentric coordinates.
    headings : ndarray, shape (n_time,)
        Animal heading at each time (radians, 0=East in allocentric frame).
    object_positions : ndarray, shape (n_objects, 2)
        Object positions in allocentric coordinates.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing distance to objects.
    env : Environment, optional
        Required when metric="geodesic".

    Returns
    -------
    distances : ndarray, shape (n_time, 1)
        Distance to nearest object at each timepoint.
    bearings : ndarray, shape (n_time, 1)
        Egocentric bearing to nearest object at each timepoint.
        Convention: 0=ahead, pi/2=left, -pi/2=right.

    Notes
    -----
    Returns arrays with shape (n_time, 1) for consistency with the pattern
    where the second dimension indexes objects/targets. Since we select
    the nearest object, the second dimension is always 1.
    """
    n_time = len(positions)
    n_objects = len(object_positions)

    # Compute distances to all objects
    if metric == "euclidean":
        # distances: (n_time, n_objects)
        distances_all = np.linalg.norm(
            positions[:, np.newaxis, :] - object_positions[np.newaxis, :, :],
            axis=2,
        )
    else:  # geodesic
        from neurospatial.ops.distance import distance_field as compute_distance_field

        # env validated by caller, this is for type narrowing
        if env is None:
            raise ValueError(
                "env is required when metric='geodesic'. "
                "This is a programming error if you see this message."
            )
        distances_all = np.full((n_time, n_objects), np.nan, dtype=np.float64)

        # Vectorized: compute all position bins at once
        pos_bins = env.bin_at(positions)
        valid_pos_mask = (pos_bins >= 0) & (pos_bins < env.n_bins)

        for i, obj_pos in enumerate(object_positions):
            # Find bin containing object
            obj_bins = env.bin_at(obj_pos.reshape(1, -1))
            obj_bin = int(obj_bins[0])

            if obj_bin < 0:
                # Object outside environment - leave distances as NaN
                # (filtered later in _coords_to_flat_bin_idx)
                continue

            # Get distance field from this object
            dist_field = compute_distance_field(env.connectivity, sources=[obj_bin])

            # Vectorized lookup
            valid_bins = pos_bins[valid_pos_mask]
            distances_all[valid_pos_mask, i] = dist_field[valid_bins]

    # Compute bearings to all objects (egocentric)
    # bearings_all: (n_time, n_objects)
    bearings_all = compute_egocentric_bearing(positions, headings, object_positions)

    # Find nearest object at each timepoint
    # Handle NaN distances (objects/positions outside environment with geodesic metric):
    # 1. Identify rows where all distances are NaN (no reachable objects)
    # 2. Replace NaN with inf for argmin (so finite distances are preferred)
    # 3. Use regular argmin on the masked array
    # 4. Restore NaN for all-NaN rows

    all_nan_mask = np.all(np.isnan(distances_all), axis=1)

    # Replace NaN with inf so argmin prefers finite values
    # np.nanargmin raises ValueError on all-NaN slices, so we use this approach
    distances_for_argmin = np.where(np.isnan(distances_all), np.inf, distances_all)
    nearest_obj_idx = np.argmin(distances_for_argmin, axis=1)

    nearest_distances = distances_all[np.arange(n_time), nearest_obj_idx]
    nearest_bearings = bearings_all[np.arange(n_time), nearest_obj_idx]

    # For timepoints where all objects had NaN distances, ensure both distance
    # and bearing are NaN. (argmin on all-inf row returns 0, which may have been
    # NaN in original). Bearing is also NaN because there's no valid nearest object.
    nearest_distances[all_nan_mask] = np.nan
    nearest_bearings[all_nan_mask] = np.nan

    # Return as (n_time, 1) for consistency
    return nearest_distances.reshape(-1, 1), nearest_bearings.reshape(-1, 1)


def _create_egocentric_environment(
    distance_range: tuple[float, float],
    n_distance_bins: int,
    n_direction_bins: int,
) -> EgocentricPolarEnvironment:
    """Create egocentric polar coordinate environment.

    Parameters
    ----------
    distance_range : tuple of float
        (min_distance, max_distance) for binning.
    n_distance_bins : int
        Number of distance bins.
    n_direction_bins : int
        Number of direction bins (covers full circle).

    Returns
    -------
    Environment
        Egocentric polar environment with n_distance_bins * n_direction_bins bins.
    """
    from neurospatial import Environment

    return Environment.from_polar_egocentric(
        distance_range=distance_range,
        angle_range=(-np.pi, np.pi),
        distance_bin_size=(distance_range[1] - distance_range[0]) / n_distance_bins,
        angle_bin_size=2 * np.pi / n_direction_bins,
        circular_angle=True,
    )


def _coords_to_flat_bin_idx(
    distances: NDArray[np.float64],
    bearings: NDArray[np.float64],
    distance_range: tuple[float, float],
    n_distance_bins: int,
    n_direction_bins: int,
) -> NDArray[np.intp]:
    """Convert egocentric coordinates to flat bin indices.

    Parameters
    ----------
    distances : ndarray, shape (n_samples,)
        Distances to nearest object.
    bearings : ndarray, shape (n_samples,)
        Egocentric bearings to nearest object.
    distance_range : tuple of float
        (min_distance, max_distance) for binning.
    n_distance_bins : int
        Number of distance bins.
    n_direction_bins : int
        Number of direction bins.

    Returns
    -------
    flat_bin_idx : ndarray, shape (n_samples,), dtype=intp
        Flat bin index for each sample. -1 for invalid (outside range).
    """
    min_dist, max_dist = distance_range
    dist_bin_size = (max_dist - min_dist) / n_distance_bins
    angle_bin_size = 2 * np.pi / n_direction_bins

    n_samples = len(distances)
    flat_bin_idx = np.full(n_samples, -1, dtype=np.intp)

    # Valid mask: finite and within distance range
    valid_mask = (
        np.isfinite(distances)
        & (distances >= min_dist)
        & (distances < max_dist)
        & np.isfinite(bearings)
    )

    if not np.any(valid_mask):
        return flat_bin_idx

    valid_distances = distances[valid_mask]
    valid_bearings = bearings[valid_mask]

    # Distance bin index
    dist_bin_idx = np.floor((valid_distances - min_dist) / dist_bin_size).astype(
        np.intp
    )
    dist_bin_idx = np.clip(dist_bin_idx, 0, n_distance_bins - 1)

    # Direction bin index: shift from [-pi, pi] to [0, 2*pi), then divide.
    # Wrap modulo 2*pi *before* flooring so that a bearing of exactly +pi
    # (which shifts to 2*pi) wraps to 0 and lands in the same direction bin
    # as -pi -- both name "directly behind". Without the wrap, +pi would
    # floor to n_direction_bins and clip to the last bin, creating a spurious
    # discontinuity at the +/-pi seam.
    angle_shifted = (valid_bearings + np.pi) % (2 * np.pi)  # Now [0, 2*pi)
    angle_bin_idx = np.floor(angle_shifted / angle_bin_size).astype(np.intp)
    angle_bin_idx = np.clip(angle_bin_idx, 0, n_direction_bins - 1)

    # Flat index: distance varies slow, angle varies fast
    flat_bin_idx[valid_mask] = dist_bin_idx * n_direction_bins + angle_bin_idx

    return flat_bin_idx


def compute_egocentric_occupancy(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    object_positions: NDArray[np.float64],
    *,
    distance_range: tuple[float, float] = (0.0, 50.0),
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    env: Environment | None = None,
) -> tuple[NDArray[np.float64], EgocentricPolarEnvironment]:
    """Compute egocentric occupancy (time at each distance/direction bin).

    Computes the total time spent at each egocentric bin by computing
    the distance and direction to the nearest object at each timepoint,
    then accumulating time intervals per bin.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, 2)
        Animal positions in allocentric coordinates.
    headings : ndarray, shape (n_samples,)
        Head direction at each time sample (radians, 0=East).
    object_positions : ndarray, shape (n_objects, 2)
        Object positions in allocentric coordinates.
    distance_range : tuple of float, default=(0.0, 50.0)
        (min_distance, max_distance) for binning.
    n_distance_bins : int, default=10
        Number of distance bins.
    n_direction_bins : int, default=12
        Number of direction bins (covers full circle -pi to pi).
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric:
        - "euclidean": Straight-line distance
        - "geodesic": Path distance respecting environment boundaries
    env : Environment, optional
        Required when metric="geodesic". The allocentric environment
        used to compute geodesic distances.

    Returns
    -------
    occupancy : ndarray, shape (n_bins,)
        Time in seconds spent at each egocentric bin.
        n_bins = n_distance_bins * n_direction_bins.
    env : Environment
        Egocentric polar coordinate environment.

    Raises
    ------
    ValueError
        If input arrays have mismatched lengths.
        If fewer than 2 samples provided.
        If times are not monotonically non-decreasing.
        If metric="geodesic" but env is None.
        If metric is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._egocentric_binning import (
    ...     compute_egocentric_occupancy,
    ... )

    >>> # Create trajectory
    >>> rng = np.random.default_rng(42)
    >>> times = np.linspace(0, 100, 1000)
    >>> positions = rng.uniform(10, 90, (1000, 2))
    >>> headings = rng.uniform(-np.pi, np.pi, 1000)
    >>> object_positions = np.array([[50.0, 50.0]])

    >>> # Compute occupancy
    >>> occupancy, env = compute_egocentric_occupancy(
    ...     times, positions, headings, object_positions
    ... )
    >>> occupancy.shape == (10 * 12,)  # n_distance * n_direction
    True
    """
    # Convert inputs to arrays
    times = np.asarray(times, dtype=np.float64).ravel()
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64).ravel()
    object_positions = np.asarray(object_positions, dtype=np.float64)

    n_samples = len(times)

    # Validate input shapes
    if len(positions) != n_samples:
        raise ValueError(
            f"times length ({n_samples}) must match positions length ({len(positions)})"
        )
    if len(headings) != n_samples:
        raise ValueError(
            f"times length ({n_samples}) must match headings length ({len(headings)})"
        )

    # Validate times
    _validate_times(times, context="egocentric occupancy computation")

    # Validate metric
    if metric not in ("euclidean", "geodesic"):
        raise ValueError(
            f"Invalid metric: '{metric}'. Must be 'euclidean' or 'geodesic'."
        )

    # Validate env requirement for geodesic
    if metric == "geodesic" and env is None:
        raise ValueError(
            "metric='geodesic' requires env parameter.\n"
            "Pass the allocentric environment to compute geodesic distances."
        )

    # Create egocentric environment
    polar_env = _create_egocentric_environment(
        distance_range, n_distance_bins, n_direction_bins
    )
    n_bins = polar_env.n_bins

    # Compute egocentric coordinates
    nearest_distances, nearest_bearings = _compute_egocentric_coords(
        positions,
        headings,
        object_positions,
        metric=metric,
        env=env,
    )

    # Flatten from (n_time, 1) to (n_time,)
    nearest_distances = nearest_distances.ravel()
    nearest_bearings = nearest_bearings.ravel()

    # Convert to flat bin indices
    bin_indices = _coords_to_flat_bin_idx(
        nearest_distances,
        nearest_bearings,
        distance_range,
        n_distance_bins,
        n_direction_bins,
    )

    # Compute per-sample time deltas
    dt = np.diff(times)

    # Compute occupancy
    occupancy = np.zeros(n_bins, dtype=np.float64)

    # Each interval[i] (from time[i] to time[i+1]) is assigned to bin at time[i]
    interval_bins = bin_indices[:-1]  # Exclude last position (no following interval)
    valid_interval_mask = interval_bins >= 0

    valid_bins = interval_bins[valid_interval_mask]
    valid_dt = dt[valid_interval_mask]
    np.add.at(occupancy, valid_bins, valid_dt)

    return occupancy, polar_env


def bin_egocentric_spike_train(
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    object_positions: NDArray[np.float64],
    *,
    distance_range: tuple[float, float] = (0.0, 50.0),
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    env: Environment | None = None,
) -> tuple[NDArray[np.float64], EgocentricPolarEnvironment]:
    """Bin spike train by egocentric coordinates.

    Converts continuous spike times to spike counts per egocentric bin based on
    the distance and direction to the nearest object at each spike time.

    Parameters
    ----------
    spike_times : ndarray, shape (n_spikes,)
        Times of spike events in seconds.
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, 2)
        Animal positions in allocentric coordinates.
    headings : ndarray, shape (n_samples,)
        Head direction at each time sample (radians, 0=East).
    object_positions : ndarray, shape (n_objects, 2)
        Object positions in allocentric coordinates.
    distance_range : tuple of float, default=(0.0, 50.0)
        (min_distance, max_distance) for binning.
    n_distance_bins : int, default=10
        Number of distance bins.
    n_direction_bins : int, default=12
        Number of direction bins.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric.
    env : Environment, optional
        Required when metric="geodesic".

    Returns
    -------
    spike_counts : ndarray, shape (n_bins,)
        Number of spikes in each egocentric bin (float64 for smoothing).
    env : Environment
        Egocentric polar coordinate environment.

    Raises
    ------
    ValueError
        If metric="geodesic" but env is None.
        If fewer than 2 trajectory samples provided.
        If times are not monotonically non-decreasing.

    Notes
    -----
    Spikes are assigned to bins using nearest-neighbor lookup to the behavioral
    frame closest in time. Spikes outside the trajectory time range are excluded.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._egocentric_binning import (
    ...     bin_egocentric_spike_train,
    ... )

    >>> # Create trajectory
    >>> rng = np.random.default_rng(42)
    >>> times = np.linspace(0, 100, 1000)
    >>> positions = rng.uniform(10, 90, (1000, 2))
    >>> headings = rng.uniform(-np.pi, np.pi, 1000)
    >>> object_positions = np.array([[50.0, 50.0]])
    >>> spike_times = np.sort(rng.uniform(0, 100, 100))

    >>> # Bin spikes
    >>> spike_counts, env = bin_egocentric_spike_train(
    ...     spike_times, times, positions, headings, object_positions
    ... )
    >>> spike_counts.shape == (env.n_bins,)
    True
    """
    # Convert inputs
    spike_times = np.asarray(spike_times, dtype=np.float64).ravel()
    times = np.asarray(times, dtype=np.float64).ravel()
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64).ravel()
    object_positions = np.asarray(object_positions, dtype=np.float64)

    # Validate times (minimum samples and monotonicity)
    _validate_times(times, context="egocentric spike binning")

    n_samples = len(times)

    # Validate metric and env
    if metric not in ("euclidean", "geodesic"):
        raise ValueError(
            f"Invalid metric: '{metric}'. Must be 'euclidean' or 'geodesic'."
        )

    if metric == "geodesic" and env is None:
        raise ValueError(
            "metric='geodesic' requires env parameter.\n"
            "Pass the allocentric environment to compute geodesic distances."
        )

    # Create egocentric environment
    polar_env = _create_egocentric_environment(
        distance_range, n_distance_bins, n_direction_bins
    )
    n_bins = polar_env.n_bins

    spike_counts = np.zeros(n_bins, dtype=np.float64)

    # Handle empty spike train
    if len(spike_times) == 0:
        return spike_counts, polar_env

    # Filter spikes to valid time range
    t_min, t_max = times[0], times[-1]
    valid_time_mask = (spike_times >= t_min) & (spike_times <= t_max)
    spike_times_valid = spike_times[valid_time_mask]

    if len(spike_times_valid) == 0:
        return spike_counts, polar_env

    # Compute egocentric coordinates for all behavioral frames
    nearest_distances, nearest_bearings = _compute_egocentric_coords(
        positions,
        headings,
        object_positions,
        metric=metric,
        env=env,
    )

    # Flatten
    nearest_distances = nearest_distances.ravel()
    nearest_bearings = nearest_bearings.ravel()

    # Find nearest behavioral frame for each spike
    spike_frame_idx = np.searchsorted(times, spike_times_valid, side="right") - 1
    spike_frame_idx = np.clip(spike_frame_idx, 0, n_samples - 1)

    # Get egocentric coordinates at spike times
    spike_distances = nearest_distances[spike_frame_idx]
    spike_bearings = nearest_bearings[spike_frame_idx]

    # Convert to flat bin indices
    spike_bin_indices = _coords_to_flat_bin_idx(
        spike_distances,
        spike_bearings,
        distance_range,
        n_distance_bins,
        n_direction_bins,
    )

    # Count valid spikes per bin
    valid_spike_mask = spike_bin_indices >= 0
    valid_spike_bins = spike_bin_indices[valid_spike_mask]

    if len(valid_spike_bins) > 0:
        np.add.at(spike_counts, valid_spike_bins, 1.0)

    return spike_counts, polar_env


def bin_egocentric_spike_trains(
    spike_times: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    object_positions: NDArray[np.float64],
    *,
    distance_range: tuple[float, float] = (0.0, 50.0),
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    env: Environment | None = None,
    n_jobs: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], EgocentricPolarEnvironment]:
    """Bin multiple spike trains by egocentric coordinates.

    Batch version of bin_egocentric_spike_train that efficiently processes
    multiple neurons. Precomputes shared quantities (egocentric coordinates,
    occupancy) and optionally parallelizes spike counting with joblib.

    Parameters
    ----------
    spike_times : sequence of arrays or 2D array
        Spike times for each neuron. Can be:
        - List/tuple of 1D arrays (one per neuron)
        - 2D array shape (n_neurons, max_spikes) with NaN padding
        Input is normalized via normalize_spike_times().
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, 2)
        Animal positions in allocentric coordinates.
    headings : ndarray, shape (n_samples,)
        Head direction at each time sample (radians, 0=East).
    object_positions : ndarray, shape (n_objects, 2)
        Object positions in allocentric coordinates.
    distance_range : tuple of float, default=(0.0, 50.0)
        (min_distance, max_distance) for binning.
    n_distance_bins : int, default=10
        Number of distance bins.
    n_direction_bins : int, default=12
        Number of direction bins.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric.
    env : Environment, optional
        Required when metric="geodesic".
    n_jobs : int, default=1
        Number of parallel jobs for spike counting. Use -1 for all CPUs.

    Returns
    -------
    spike_counts : ndarray, shape (n_neurons, n_bins)
        Number of spikes in each egocentric bin for each neuron.
    occupancy : ndarray, shape (n_bins,)
        Time in seconds spent at each egocentric bin (shared across neurons).
    env : Environment
        Egocentric polar coordinate environment.

    Raises
    ------
    ValueError
        If metric="geodesic" but env is None.
        If times are not monotonically non-decreasing.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._egocentric_binning import (
    ...     bin_egocentric_spike_trains,
    ... )

    >>> # Create trajectory and spikes for 3 neurons
    >>> rng = np.random.default_rng(42)
    >>> times = np.linspace(0, 100, 1000)
    >>> positions = rng.uniform(10, 90, (1000, 2))
    >>> headings = rng.uniform(-np.pi, np.pi, 1000)
    >>> object_positions = np.array([[50.0, 50.0]])
    >>> spike_times = [
    ...     np.sort(rng.uniform(0, 100, 100)),  # Neuron 0
    ...     np.sort(rng.uniform(0, 100, 150)),  # Neuron 1
    ...     np.sort(rng.uniform(0, 100, 50)),  # Neuron 2
    ... ]

    >>> # Bin spikes
    >>> spike_counts, occupancy, env = bin_egocentric_spike_trains(
    ...     spike_times, times, positions, headings, object_positions
    ... )
    >>> spike_counts.shape == (3, env.n_bins)
    True

    See Also
    --------
    bin_egocentric_spike_train : Single-neuron version
    compute_egocentric_occupancy : Compute occupancy only
    """
    from neurospatial.encoding._spikes import normalize_spike_times

    # Normalize spike times to canonical list-of-arrays format
    spike_times_list = normalize_spike_times(spike_times)
    n_neurons = len(spike_times_list)

    times = np.asarray(times, dtype=np.float64).ravel()
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64).ravel()
    object_positions = np.asarray(object_positions, dtype=np.float64)

    n_samples = len(times)

    # Validate metric and env
    if metric not in ("euclidean", "geodesic"):
        raise ValueError(
            f"Invalid metric: '{metric}'. Must be 'euclidean' or 'geodesic'."
        )

    if metric == "geodesic" and env is None:
        raise ValueError(
            "metric='geodesic' requires env parameter.\n"
            "Pass the allocentric environment to compute geodesic distances."
        )

    # Validate times
    _validate_times(times, context="spike binning")

    # Create egocentric environment
    polar_env = _create_egocentric_environment(
        distance_range, n_distance_bins, n_direction_bins
    )
    n_bins = polar_env.n_bins

    # Compute egocentric coordinates ONCE (shared across all neurons)
    nearest_distances, nearest_bearings = _compute_egocentric_coords(
        positions,
        headings,
        object_positions,
        metric=metric,
        env=env,
    )

    # Flatten
    nearest_distances = nearest_distances.ravel()
    nearest_bearings = nearest_bearings.ravel()

    # Precompute bin indices for all behavioral frames
    bin_indices = _coords_to_flat_bin_idx(
        nearest_distances,
        nearest_bearings,
        distance_range,
        n_distance_bins,
        n_direction_bins,
    )

    # Compute occupancy from precomputed bin indices
    dt = np.diff(times)
    occupancy = np.zeros(n_bins, dtype=np.float64)
    interval_bins = bin_indices[:-1]
    valid_interval_mask = interval_bins >= 0
    valid_bins = interval_bins[valid_interval_mask]
    valid_dt = dt[valid_interval_mask]
    np.add.at(occupancy, valid_bins, valid_dt)

    # Handle empty neuron list
    if n_neurons == 0:
        spike_counts = np.zeros((0, n_bins), dtype=np.float64)
        return spike_counts, occupancy, polar_env

    # Helper to bin a single neuron using precomputed bin_indices
    def _bin_single_neuron(
        neuron_spike_times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        counts = np.zeros(n_bins, dtype=np.float64)

        if len(neuron_spike_times) == 0:
            return counts

        # Filter spikes to valid time range
        t_min, t_max = times[0], times[-1]
        valid_mask = (neuron_spike_times >= t_min) & (neuron_spike_times <= t_max)
        valid_spike_times = neuron_spike_times[valid_mask]

        if len(valid_spike_times) == 0:
            return counts

        # Find nearest behavioral frame for each spike
        spike_frame_idx = np.searchsorted(times, valid_spike_times, side="right") - 1
        spike_frame_idx = np.clip(spike_frame_idx, 0, n_samples - 1)

        # Get precomputed bin index at each spike time
        spike_bins = bin_indices[spike_frame_idx]

        # Count valid spikes per bin
        valid_spike_mask = spike_bins >= 0
        valid_spike_bins = spike_bins[valid_spike_mask]

        if len(valid_spike_bins) > 0:
            np.add.at(counts, valid_spike_bins, 1.0)

        return counts

    # Process neurons
    if n_jobs == 1:
        # Sequential processing
        spike_counts = np.zeros((n_neurons, n_bins), dtype=np.float64)
        for i, spikes in enumerate(spike_times_list):
            spike_counts[i] = _bin_single_neuron(spikes)
    else:
        # Parallel processing with joblib
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(
            delayed(_bin_single_neuron)(spikes) for spikes in spike_times_list
        )
        spike_counts = np.array(results, dtype=np.float64)

    return spike_counts, occupancy, polar_env
