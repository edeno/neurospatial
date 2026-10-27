"""Binning layer for egocentric encoding (object-vector cells).

This module converts spike trains and trajectory data into discrete spike counts
and occupancy arrays in egocentric polar coordinates (distance, direction to
nearest object).

The key difference from spatial binning (``_spatial_counts.py``) is that:
- Spatial binning: bins by *where the animal was*
- Egocentric binning: bins by *distance/direction to nearest object*

``prepare_egocentric_binning`` handles:
1. Computation of egocentric coordinates (distance, bearing to nearest object)
2. The polar bin of each sample and the interval support shared by occupancy
   and spike counts

Its ``FrameBinning.count`` then counts each neuron's spikes from the most
recent frame and accumulates occupancy from the same intervals.

Output shapes:
- Spike counts: (n_neurons, n_bins)
- Occupancy: (n_bins,) - always shared across neurons
- env: Environment in polar coordinates

The binning layer is separated from smoothing to allow:
- Reusing occupancy across multiple neurons
- Precomputing egocentric coordinates for efficiency

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
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._angles import wrap_angle
from neurospatial._exceptions import _format_error
from neurospatial._intervals import (
    DEFAULT_MAX_GAP,
    hold_last_valid,
    interval_valid_mask,
)
from neurospatial.encoding._spatial_counts import FrameBinning, prepare_frame_binning
from neurospatial.encoding._validation import _validate_times
from neurospatial.ops.egocentric import compute_egocentric_bearing

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.geometry import EgocentricPolarSystem
    from neurospatial.ops._types import DistanceMetric

    PolarEnvironment = Environment[EgocentricPolarSystem]

__all__ = [
    "normalize_object_positions",
    "prepare_egocentric_binning",
    "resolve_object_positions",
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
    >>> from neurospatial.encoding._egocentric_counts import normalize_object_positions
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


def _landmark_names(object_positions: object) -> list[str] | None:
    """The landmark names ``object_positions`` gives, or None for coordinates."""
    if isinstance(object_positions, str):
        return [object_positions]
    if (
        isinstance(object_positions, (list, tuple))
        and object_positions
        and all(isinstance(item, str) for item in object_positions)
    ):
        return list(object_positions)
    return None


def resolve_object_positions(
    object_positions: Any,
    *,
    env: Environment | None,
    positions: NDArray[np.float64],
    metric: DistanceMetric,
) -> tuple[NDArray[np.float64], list[str] | None]:
    """Object coordinates for the egocentric encoders, and their landmark names.

    Parameters
    ----------
    object_positions : array-like, str or sequence of str
        Fixed positions ``(n_objects, 2)`` (or one ``[x, y]``), positions per
        sample ``(n_time, n_objects, 2)``, or names of ``env.landmarks``.
    env : Environment or None
        Holds the landmarks; required for names.
    positions : ndarray, shape (n_time, 2)
        The animal's positions, for the nearest point of each landmark.
    metric : {"euclidean", "geodesic"}
        Geodesic distance needs fixed object points.

    Returns
    -------
    objects : ndarray, shape (n_objects, 2) or (n_time, n_objects, 2)
        Fixed points when every object is a point; otherwise each object's
        nearest point at each sample.
    names : list of str or None
        The landmark names, in order, or None for coordinates.
    """
    names = _landmark_names(object_positions)
    if names is None:
        array = np.asarray(object_positions, dtype=np.float64)
        if array.ndim != 3:
            return normalize_object_positions(array), None
        if (
            array.shape[0] != len(positions)
            or array.shape[1] == 0
            or array.shape[2] != 2
        ):
            raise ValueError(
                _format_error(
                    f"object_positions per sample must have shape (n_time, "
                    f"n_objects, 2) with n_time={len(positions)}, got {array.shape}.",
                    fix="give each object's position at every sample, or one "
                    "fixed position per object as shape (n_objects, 2)",
                )
            )
        if metric == "geodesic":
            raise ValueError(_moving_geodesic_error())
        return array, None
    if env is None:
        raise ValueError(
            _format_error(
                f"object_positions names landmarks {names} but no environment "
                "was given.",
                fix="pass the environment that holds the landmarks, or give the "
                "objects' coordinates",
            )
        )
    landmarks = [env.landmarks[name] for name in names]
    if any(landmark.n_dims != 2 for landmark in landmarks):
        raise ValueError("Egocentric encoding needs two-dimensional landmarks.")
    if all(landmark.kind == "point" for landmark in landmarks):
        return np.array([landmark.position for landmark in landmarks]), names
    if metric == "geodesic":
        raise ValueError(_moving_geodesic_error())
    return env.landmarks.nearest_points(positions, names), names


def _validate_object_vector_metric(
    env: Environment | None, metric: str, *, context: str
) -> None:
    """Validate public metric choices before coordinate construction."""
    if metric not in ("euclidean", "geodesic"):
        raise ValueError(
            _format_error(
                f"{context}: Invalid metric: {metric!r}; expected 'euclidean' or 'geodesic'.",
                why="Why: nearest-object distance uses a straight line or the environment graph",
                fix="pass metric='euclidean', or metric='geodesic' with an env",
            )
        )
    if metric == "geodesic" and env is None:
        raise ValueError(
            _format_error(
                f"{context}: metric='geodesic' requires env parameter.",
                why="Why: geodesic distance needs the environment's connectivity graph",
                fix="pass an env, or use metric='euclidean' with env=None",
            )
        )


def _moving_geodesic_error() -> str:
    return _format_error(
        "metric='geodesic' needs a fixed position for each object, but these "
        "objects' nearest points move with the animal (a line or polygon "
        "landmark, or positions per sample).",
        why="Why: geodesic distance is computed outward from each object's bin.",
        fix="use metric='euclidean', or name point landmarks",
    )


def _compute_object_coords(
    positions: NDArray[np.float64],
    headings: NDArray[np.float64] | None,
    object_positions: NDArray[np.float64],
    *,
    metric: DistanceMetric = "euclidean",
    env: Environment | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute polar coordinates to the nearest object in the selected frame.

    Parameters
    ----------
    positions : ndarray, shape (n_time, 2)
        Animal positions in allocentric coordinates.
    headings : ndarray, shape (n_time,)
        Heading at each time (radians, 0=East), or None for allocentric bearing.
    object_positions : ndarray, shape (n_objects, 2) or (n_time, n_objects, 2)
        Object positions in allocentric coordinates, fixed or per sample.
        Per-sample positions need the Euclidean metric.
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
    n_objects = object_positions.shape[-2]
    # Objects at every sample, shape (n_time or 1, n_objects, 2).
    targets = object_positions if object_positions.ndim == 3 else object_positions[None]

    # Compute distances to all objects
    if metric == "euclidean":
        # distances: (n_time, n_objects)
        distances_all = np.linalg.norm(targets - positions[:, np.newaxis, :], axis=2)
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
    if headings is None:
        delta = targets - positions[:, None, :]
        bearings_all = wrap_angle(np.arctan2(delta[:, :, 1], delta[:, :, 0]))
    else:
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
) -> PolarEnvironment:
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
    from neurospatial.environment import Environment

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


BEYOND_RANGE = -2
"""Frame bin of a sample whose nearest object is beyond the polar range."""


def observed_frames(
    positions: NDArray[np.float64],
    headings: NDArray[np.float64] | None,
    object_positions: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """True for samples whose position, heading and objects were tracked.

    Any other sample is missing: it is skipped as if its row were deleted,
    and the gap around it is bridged when it is no longer than ``max_gap``.

    Parameters
    ----------
    positions : ndarray, shape (n_time, 2)
    headings : ndarray, shape (n_time,), or None
    object_positions : ndarray, shape (n_objects, 2) or (n_time, n_objects, 2)

    Returns
    -------
    ndarray of bool, shape (n_time,)
    """
    positions = np.asarray(positions, dtype=np.float64)
    observed = np.asarray(
        np.isfinite(positions.reshape(len(positions), -1)).all(axis=1), dtype=bool
    )
    if headings is not None:
        observed &= np.isfinite(np.asarray(headings, dtype=np.float64)).ravel()
    objects = np.asarray(object_positions, dtype=np.float64)
    if objects.ndim == 3:
        observed &= np.isfinite(objects).all(axis=(1, 2))
    return observed


def object_frame_bins(
    positions: NDArray[np.float64],
    headings: NDArray[np.float64] | None,
    object_positions: NDArray[np.float64],
    *,
    distance_range: tuple[float, float],
    n_distance_bins: int,
    n_direction_bins: int,
    metric: DistanceMetric,
    env: Environment | None,
) -> NDArray[np.intp]:
    """Polar bin of the nearest object at each sample, or -1.

    Parameters
    ----------
    positions : ndarray, shape (n_time, 2)
        Animal positions in allocentric coordinates.
    headings : ndarray, shape (n_time,) or None
        Heading in radians, or None for allocentric bearing.
    object_positions : ndarray, shape (n_objects, 2) or (n_time, n_objects, 2)
        Object positions, fixed or per sample.
    distance_range, n_distance_bins, n_direction_bins
        The polar grid.
    metric : {"euclidean", "geodesic"}
        Distance to the objects.
    env : Environment or None
        The environment the positions were recorded in, or None.

    Returns
    -------
    ndarray of intp, shape (n_time,)
        Flat polar bin; :data:`BEYOND_RANGE` for a tracked sample whose
        nearest object is at or beyond ``distance_range[1]``, and -1 for a
        missing sample, a distance below ``distance_range[0]`` and, when
        ``env`` is given, a position outside every environment bin. The
        last is the rule place maps use: a sample tracked outside the
        environment is not an observation of it. Both codes are negative,
        so every consumer that keeps bins ``>= 0`` drops them.
    """
    distances, bearings = _compute_object_coords(
        positions, headings, object_positions, metric=metric, env=env
    )
    distances, bearings = distances.ravel(), bearings.ravel()
    bins = _coords_to_flat_bin_idx(
        distances,
        bearings,
        distance_range,
        n_distance_bins,
        n_direction_bins,
    )
    with np.errstate(invalid="ignore"):
        beyond = np.isfinite(bearings) & (distances >= distance_range[1])
    bins[beyond] = BEYOND_RANGE
    if env is not None:
        bins[env.bin_at(positions) < 0] = -1
    return bins


def prepare_egocentric_binning(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64] | None,
    object_positions: NDArray[np.float64],
    *,
    distance_range: tuple[float, float],
    n_distance_bins: int,
    n_direction_bins: int,
    metric: DistanceMetric = "euclidean",
    env: Environment | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
    frame_bins: NDArray[np.int64] | None = None,
    polar_env: PolarEnvironment | None = None,
    break_before: NDArray[np.bool_] | None = None,
) -> tuple[FrameBinning, PolarEnvironment, float]:
    """Prepare geometry and exact temporal support before counting spikes.

    ``break_before`` is already validated; the interval ending at each
    declared recording break is dropped from counts and occupancy.

    Returns
    -------
    prepared : FrameBinning
        Frame bins and the exact interval support of the map.
    polar_env : Environment
        The polar grid.
    time_beyond : float
        Seconds, within the same temporal gates, during which the nearest
        object was at or beyond ``distance_range[1]``: the time the map
        would have covered had its range been unbounded.
    """
    times = np.asarray(times, dtype=np.float64).ravel()
    positions = np.asarray(positions, dtype=np.float64)
    headings = (
        None if headings is None else np.asarray(headings, dtype=np.float64).ravel()
    )
    object_positions = np.asarray(object_positions, dtype=np.float64)

    _validate_object_vector_metric(env, metric, context="prepare_egocentric_binning")

    # Validate times
    _validate_times(times, context="prepare_egocentric_binning")

    # Create egocentric environment
    if polar_env is None:
        polar_env = _create_egocentric_environment(
            distance_range, n_distance_bins, n_direction_bins
        )
    n_bins = polar_env.n_bins

    if frame_bins is None:
        # Polar bins ONCE (shared across all neurons)
        frame_bins = object_frame_bins(
            positions,
            headings,
            object_positions,
            distance_range=distance_range,
            n_distance_bins=n_distance_bins,
            n_direction_bins=n_direction_bins,
            metric=metric,
            env=env,
        )

    # One temporal gate for the map and for the time its range left out: a
    # sample beyond the range passes the bin check here and is split off below.
    observed = observed_frames(positions, headings, object_positions)
    gated = interval_valid_mask(
        times,
        start_bin=np.where(frame_bins == BEYOND_RANGE, 0, frame_bins),
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
        break_before=break_before,
        sample_valid=observed,
    )
    starts = hold_last_valid(frame_bins, observed, fill=-1)[:-1]
    time_beyond = float(np.diff(times)[gated & (starts == BEYOND_RANGE)].sum())
    prepared = prepare_frame_binning(
        times,
        frame_bins,
        n_bins,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
        interval_mask=gated & (starts >= 0),
        break_before=break_before,
        sample_valid=observed,
    )
    return prepared, polar_env, time_beyond
