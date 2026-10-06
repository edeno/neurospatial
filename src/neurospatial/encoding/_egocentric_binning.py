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

from neurospatial.encoding._binning import count_spikes_by_frame
from neurospatial.encoding._validation import validate_times as _validate_times
from neurospatial.environment.trajectory import (
    interval_valid_mask,
    start_allocated_occupancy,
)
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
    max_gap: float | None = 0.5,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
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

    mask = interval_valid_mask(
        times,
        start_bin=bin_indices,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
    )
    occupancy = start_allocated_occupancy(bin_indices, np.diff(times), mask, n_bins)
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
    max_gap: float | None = 0.5,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
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
    frame at or before each spike. Spikes outside the trajectory time range are excluded.

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

    nearest_distances, nearest_bearings = _compute_egocentric_coords(
        positions,
        headings,
        object_positions,
        metric=metric,
        env=env,
    )
    bin_indices = _coords_to_flat_bin_idx(
        nearest_distances.ravel(),
        nearest_bearings.ravel(),
        distance_range,
        n_distance_bins,
        n_direction_bins,
    )
    mask = interval_valid_mask(
        times,
        start_bin=bin_indices,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
    )
    counts = count_spikes_by_frame(spike_times, times, bin_indices, mask, n_bins)
    return counts, polar_env


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
    max_gap: float | None = 0.5,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
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
        Input is coerced to per-neuron spike trains via as_spike_trains().
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
    from neurospatial.encoding._spikes import as_spike_trains

    # Normalize spike times to canonical list-of-arrays format
    spike_times_list = as_spike_trains(spike_times)
    n_neurons = len(spike_times_list)

    times = np.asarray(times, dtype=np.float64).ravel()
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64).ravel()
    object_positions = np.asarray(object_positions, dtype=np.float64)

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

    mask = interval_valid_mask(
        times,
        start_bin=bin_indices,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
    )
    occupancy = start_allocated_occupancy(bin_indices, np.diff(times), mask, n_bins)
    spike_counts = np.zeros((n_neurons, n_bins), dtype=np.float64)
    if n_neurons and n_jobs != 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(
            delayed(count_spikes_by_frame)(spikes, times, bin_indices, mask, n_bins)
            for spikes in spike_times_list
        )
        spike_counts = np.asarray(results, dtype=np.float64)
    else:
        for i, spikes in enumerate(spike_times_list):
            spike_counts[i] = count_spikes_by_frame(
                spikes, times, bin_indices, mask, n_bins
            )
    return spike_counts, occupancy, polar_env
