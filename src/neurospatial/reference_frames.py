"""Coordinate reference frame transformations.

Supports conversions between:
- Allocentric: World-centered, fixed axes (standard spatial analysis)
- Egocentric: Animal-centered, axes rotate with heading

Common Use Cases
----------------
- Object-vector cell analysis (egocentric distance/direction to objects)
- Spatial view cell analysis (what location is being viewed)
- GLM regressors in egocentric coordinates
- Behavioral analysis of approach/avoidance

Coordinate Conventions
----------------------
**Allocentric (world-centered)**:
- 0 radians = East (+x direction)
- pi/2 radians = North (+y direction)
- Standard mathematical convention

**Egocentric (animal-centered)**:
- Origin at animal's position
- +x axis = forward (heading direction)
- +y axis = left (90 degrees counterclockwise from heading)
- Angles: 0=ahead, pi/2=left, -pi/2=right, +/-pi=behind

Examples
--------
Transform landmark positions to egocentric coordinates:

>>> from neurospatial.reference_frames import allocentric_to_egocentric
>>> import numpy as np
>>> landmarks = np.array([[10.0, 0.0], [0.0, 10.0]])  # 2 landmarks
>>> positions = np.array([[0.0, 0.0]])  # Animal at origin
>>> headings = np.array([0.0])  # Facing East
>>> ego = allocentric_to_egocentric(landmarks, positions, headings)
>>> ego.shape
(1, 2, 2)

References
----------
.. [1] Hoydal, O. A., et al. (2019). Object-vector coding in the medial
       entorhinal cortex. Nature, 568(7752), 400-404.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

if TYPE_CHECKING:
    from neurospatial import Environment

__all__ = [
    "EgocentricFrame",
    "allocentric_to_egocentric",
    "compute_egocentric_bearing",
    "compute_egocentric_distance",
    "egocentric_to_allocentric",
    "heading_from_body_orientation",
    "heading_from_velocity",
]


@dataclass(frozen=True)
class EgocentricFrame:
    """Egocentric reference frame at a single timepoint.

    Attributes
    ----------
    position : NDArray, shape (2,)
        Animal position in allocentric coordinates.
    heading : float
        Animal heading in radians (0=East, pi/2=North in allocentric frame).

    Coordinate Conventions
    ----------------------
    **Allocentric (world-centered)**:
    - 0 radians = East (+x direction)
    - pi/2 radians = North (+y direction)
    - Standard mathematical convention

    **Egocentric (animal-centered)**:
    - Origin at animal's position
    - +x axis = forward (heading direction)
    - +y axis = left (90 degrees counterclockwise from heading)
    - Angles: 0=ahead, pi/2=left, -pi/2=right, +/-pi=behind

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.reference_frames import EgocentricFrame

    Animal at origin facing East (heading=0):

    >>> frame = EgocentricFrame(position=np.array([0.0, 0.0]), heading=0.0)
    >>> # Point 10 units East is 10 units ahead
    >>> frame.to_egocentric(np.array([[10.0, 0.0]]))
    array([[10.,  0.]])

    Animal at origin facing North (heading=pi/2):

    >>> frame = EgocentricFrame(position=np.array([0.0, 0.0]), heading=np.pi / 2)
    >>> # Point 10 units East is now 10 units to the right
    >>> result = frame.to_egocentric(np.array([[10.0, 0.0]]))
    >>> np.allclose(result, [[0.0, -10.0]])
    True

    Round-trip preserves coordinates:

    >>> allocentric = np.array([[5.0, 3.0]])
    >>> egocentric = frame.to_egocentric(allocentric)
    >>> recovered = frame.to_allocentric(egocentric)
    >>> np.allclose(allocentric, recovered)
    True
    """

    position: NDArray[np.float64]
    heading: float

    def to_egocentric(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform allocentric points to egocentric coordinates.

        Parameters
        ----------
        points : NDArray, shape (n_points, 2)
            Points in allocentric coordinates.

        Returns
        -------
        NDArray, shape (n_points, 2)
            Points in egocentric coordinates.
        """
        centered = points - self.position
        cos_h, sin_h = np.cos(-self.heading), np.sin(-self.heading)
        rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        result: NDArray[np.float64] = centered @ rotation.T
        return result

    def to_allocentric(self, ego_points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform egocentric points to allocentric coordinates.

        Parameters
        ----------
        ego_points : NDArray, shape (n_points, 2)
            Points in egocentric coordinates.

        Returns
        -------
        NDArray, shape (n_points, 2)
            Points in allocentric coordinates.
        """
        cos_h, sin_h = np.cos(self.heading), np.sin(self.heading)
        rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        result: NDArray[np.float64] = (ego_points @ rotation.T) + self.position
        return result


def allocentric_to_egocentric(
    points: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Batch transform allocentric points to egocentric coordinates.

    Parameters
    ----------
    points : NDArray, shape (n_points, 2) or (n_time, n_points, 2)
        Points to transform. If 2D, same points transformed at each time.
    positions : NDArray, shape (n_time, 2)
        Animal position at each time.
    headings : NDArray, shape (n_time,)
        Animal heading at each time (radians).

    Returns
    -------
    NDArray, shape (n_time, n_points, 2)
        Points in egocentric coordinates at each timepoint.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.reference_frames import allocentric_to_egocentric

    Transform landmarks at multiple timepoints:

    >>> landmarks = np.array([[10.0, 0.0], [0.0, 10.0]])  # 2 landmarks
    >>> positions = np.array([[0.0, 0.0], [0.0, 0.0]])  # Animal at origin
    >>> headings = np.array([0.0, np.pi / 2])  # Facing East, then North
    >>> ego = allocentric_to_egocentric(landmarks, positions, headings)
    >>> ego.shape
    (2, 2, 2)

    At t=0 (facing East), landmark (10, 0) is ahead:

    >>> np.allclose(ego[0, 0], [10.0, 0.0])
    True

    At t=1 (facing North), landmark (10, 0) is to the right:

    >>> np.allclose(ego[1, 0], [0.0, -10.0])
    True

    Raises
    ------
    ValueError
        If points has wrong shape or positions/headings length mismatch.
    """
    points = np.asarray(points, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)

    # Validate shapes
    if points.ndim < 2:
        raise ValueError(
            f"Cannot transform points: invalid shape {points.shape}.\n\n"
            f"WHAT: points must be 2D array with shape (n_points, 2)\n"
            f"WHY: Each point needs (x, y) coordinates for transformation\n\n"
            f"HOW to fix:\n"
            f"1. Reshape your array: points.reshape(-1, 2)\n"
            f"2. Check data loading - array may have been squeezed\n"
            f"3. Verify you're passing an array of points, not a single point"
        )

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            f"Cannot transform: invalid positions shape {positions.shape}.\n\n"
            f"WHAT: positions must have shape (n_time, 2)\n"
            f"WHY: Need (x, y) position at each timepoint for the transform origin\n\n"
            f"HOW to fix:\n"
            f"1. Reshape: positions.reshape(-1, 2)\n"
            f"2. For single position: positions.reshape(1, 2)"
        )

    if headings.ndim != 1 or len(headings) != len(positions):
        raise ValueError(
            f"Headings/positions length mismatch.\n\n"
            f"WHAT: headings shape {headings.shape} != positions length {len(positions)}\n"
            f"WHY: Need one heading per timepoint for coordinate rotation\n\n"
            f"HOW to fix:\n"
            f"1. Ensure headings and positions are aligned to same timepoints\n"
            f"2. Check for off-by-one errors in slicing\n"
            f"3. Interpolate headings to match positions if sampled differently"
        )

    n_time = len(positions)

    # Expand points to 3D if needed
    if points.ndim == 2:
        points = np.broadcast_to(points, (n_time, points.shape[0], 2)).copy()
    elif points.ndim != 3:
        raise ValueError(
            f"Invalid points dimensionality: got {points.ndim}D array.\n\n"
            f"WHAT: points must be 2D (n_points, 2) or 3D (n_time, n_points, 2)\n"
            f"WHY: 2D broadcasts same points to all timepoints; 3D allows time-varying\n\n"
            f"HOW to fix:\n"
            f"1. Static points: use shape (n_points, 2)\n"
            f"2. Time-varying: use shape (n_time, n_points, 2)\n"
            f"Got shape: {points.shape}"
        )

    # Center points around animal position
    # positions: (n_time, 2) -> (n_time, 1, 2) for broadcasting
    centered = points - positions[:, np.newaxis, :]

    # Build rotation matrices for each timepoint
    # Rotate by -heading to transform from allocentric to egocentric
    cos_h = np.cos(-headings)
    sin_h = np.sin(-headings)

    # Rotation matrices: (n_time, 2, 2)
    rot = np.zeros((n_time, 2, 2), dtype=np.float64)
    rot[:, 0, 0] = cos_h
    rot[:, 0, 1] = -sin_h
    rot[:, 1, 0] = sin_h
    rot[:, 1, 1] = cos_h

    # Apply rotation: (n_time, n_points, 2) @ (n_time, 2, 2).T
    # Use einsum for vectorized rotation
    result: NDArray[np.float64] = np.einsum("tij,tpj->tpi", rot, centered)

    return result


def egocentric_to_allocentric(
    ego_points: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Batch transform egocentric points to allocentric coordinates.

    This is the inverse of allocentric_to_egocentric.

    Parameters
    ----------
    ego_points : NDArray, shape (n_time, n_points, 2)
        Points in egocentric coordinates.
    positions : NDArray, shape (n_time, 2)
        Animal position at each time in allocentric coordinates.
    headings : NDArray, shape (n_time,)
        Animal heading at each time (radians).

    Returns
    -------
    NDArray, shape (n_time, n_points, 2)
        Points in allocentric coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.reference_frames import (
    ...     allocentric_to_egocentric,
    ...     egocentric_to_allocentric,
    ... )

    Round-trip transformation preserves coordinates:

    >>> landmarks = np.array([[10.0, 0.0], [0.0, 10.0]])
    >>> positions = np.array([[5.0, 5.0]])
    >>> headings = np.array([np.pi / 4])
    >>> ego = allocentric_to_egocentric(landmarks, positions, headings)
    >>> recovered = egocentric_to_allocentric(ego, positions, headings)
    >>> np.allclose(recovered[0], landmarks)
    True
    """
    ego_points = np.asarray(ego_points, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)

    n_time = len(positions)

    # Build rotation matrices for each timepoint
    # Rotate by +heading (inverse of -heading used in allocentric_to_egocentric)
    cos_h = np.cos(headings)
    sin_h = np.sin(headings)

    rot = np.zeros((n_time, 2, 2), dtype=np.float64)
    rot[:, 0, 0] = cos_h
    rot[:, 0, 1] = -sin_h
    rot[:, 1, 0] = sin_h
    rot[:, 1, 1] = cos_h

    # Apply rotation
    rotated: NDArray[np.float64] = np.einsum("tij,tpj->tpi", rot, ego_points)

    # Add animal position
    result: NDArray[np.float64] = rotated + positions[:, np.newaxis, :]

    return result


def compute_egocentric_bearing(
    targets: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute bearing angle to targets in egocentric coordinates.

    The bearing is the angle to the target relative to the animal's heading,
    where 0=ahead, pi/2=left, -pi/2=right, +/-pi=behind.

    Parameters
    ----------
    targets : NDArray, shape (n_targets, 2) or (n_time, n_targets, 2)
        Target positions in allocentric coordinates.
    positions : NDArray, shape (n_time, 2)
        Animal position at each time.
    headings : NDArray, shape (n_time,)
        Animal heading at each time (radians).

    Returns
    -------
    NDArray, shape (n_time, n_targets)
        Bearing to each target at each timepoint in radians.
        Range: (-pi, pi], where 0=ahead, pi/2=left, -pi/2=right.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.reference_frames import compute_egocentric_bearing

    Target directly ahead has bearing 0:

    >>> target = np.array([[10.0, 0.0]])
    >>> position = np.array([[0.0, 0.0]])
    >>> heading = np.array([0.0])  # Facing East
    >>> bearing = compute_egocentric_bearing(target, position, heading)
    >>> np.allclose(bearing, [[0.0]])
    True

    Target to the left has bearing pi/2:

    >>> target = np.array([[0.0, 10.0]])
    >>> bearing = compute_egocentric_bearing(target, position, heading)
    >>> np.allclose(bearing, [[np.pi / 2]])
    True
    """
    # Transform targets to egocentric coordinates
    ego = allocentric_to_egocentric(targets, positions, headings)

    # Compute bearing using arctan2
    bearing = np.arctan2(ego[..., 1], ego[..., 0])

    # Wrap to (-pi, pi]
    bearing = _wrap_angle(bearing)

    return bearing


def _wrap_angle(angle: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wrap angles to (-pi, pi] range.

    Parameters
    ----------
    angle : NDArray
        Angles in radians.

    Returns
    -------
    NDArray
        Angles wrapped to (-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def compute_egocentric_distance(
    targets: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    metric: str = "euclidean",
    env: Environment | None = None,
) -> NDArray[np.float64]:
    """Compute distance to targets from animal position.

    Parameters
    ----------
    targets : NDArray, shape (n_targets, 2) or (n_time, n_targets, 2)
        Target positions in allocentric coordinates.
    positions : NDArray, shape (n_time, 2)
        Animal position at each time.
    headings : NDArray, shape (n_time,)
        Animal heading at each time (radians). Not used for distance
        calculation but included for API consistency.
    metric : str, default "euclidean"
        Distance metric. Options:
        - "euclidean": Straight-line distance
        - "geodesic": Path distance respecting environment boundaries
    env : Environment, optional
        Required when metric="geodesic". The environment to compute
        geodesic distances within.

    Returns
    -------
    NDArray, shape (n_time, n_targets)
        Distance to each target at each timepoint.

    Raises
    ------
    ValueError
        If metric is invalid or metric="geodesic" without an environment.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.reference_frames import compute_egocentric_distance

    Euclidean distance:

    >>> targets = np.array([[10.0, 0.0], [0.0, 10.0]])
    >>> position = np.array([[0.0, 0.0]])
    >>> heading = np.array([0.0])
    >>> distances = compute_egocentric_distance(
    ...     targets, position, heading, metric="euclidean"
    ... )
    >>> np.allclose(distances, [[10.0, 10.0]])
    True
    """
    targets = np.asarray(targets, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    if metric not in ("euclidean", "geodesic"):
        raise ValueError(
            f"Invalid distance metric: '{metric}'.\n\n"
            f"WHAT: metric must be 'euclidean' or 'geodesic'\n"
            f"WHY: These are the supported distance algorithms\n\n"
            f"HOW to fix:\n"
            f"1. Use 'euclidean' for straight-line distances (default, faster)\n"
            f"2. Use 'geodesic' for boundary-respecting distances (requires env)"
        )

    if metric == "geodesic" and env is None:
        raise ValueError(
            "Cannot compute geodesic distances: missing environment.\n\n"
            "WHAT: metric='geodesic' requires env parameter\n"
            "WHY: Geodesic distances follow paths that respect environment boundaries\n\n"
            "HOW to fix:\n"
            "1. Pass the environment: compute_egocentric_distance(..., env=env)\n"
            "2. Or use 'euclidean' for straight-line distances:\n"
            "   compute_egocentric_distance(..., metric='euclidean')"
        )

    n_time = len(positions)

    # Expand targets to 3D if needed
    if targets.ndim == 2:
        targets_3d = np.broadcast_to(targets, (n_time, targets.shape[0], 2))
    else:
        targets_3d = targets

    n_targets = targets_3d.shape[1]

    distances: NDArray[np.float64]

    if metric == "euclidean":
        # Compute Euclidean distances
        # targets_3d: (n_time, n_targets, 2)
        # positions: (n_time, 2) -> (n_time, 1, 2)
        diff = targets_3d - positions[:, np.newaxis, :]
        distances = np.sqrt(np.sum(diff**2, axis=-1))

    else:  # geodesic
        from neurospatial.distance import distance_field as compute_distance_field

        # env is guaranteed non-None here (validated at line 471-472)
        assert env is not None

        # Use geodesic distance field for graph-based distances
        distances = np.zeros((n_time, n_targets), dtype=np.float64)

        for i, target in enumerate(targets_3d[0]):  # Assume same targets over time
            # Find the bin containing the target (bin_at expects 2D input)
            target_bins = env.bin_at(target.reshape(1, -1))
            target_bin = int(target_bins[0])
            if target_bin < 0:
                # Target outside environment
                distances[:, i] = np.nan
                continue

            # Get distance field from this target bin
            dist_field = compute_distance_field(env.connectivity, sources=[target_bin])

            for t in range(n_time):
                # Find distance at animal's position
                pos_bins = env.bin_at(positions[t].reshape(1, -1))
                bin_idx = int(pos_bins[0])
                if bin_idx >= 0 and bin_idx < len(dist_field):
                    distances[t, i] = float(dist_field[bin_idx])
                else:
                    distances[t, i] = np.nan

    return distances


def heading_from_velocity(
    positions: NDArray[np.float64],
    dt: float,
    min_speed: float = 0.0,
    smoothing_sigma: float = 0.0,
) -> NDArray[np.float64]:
    """Compute heading from position timeseries using velocity direction.

    Parameters
    ----------
    positions : NDArray, shape (n_time, 2)
        Animal positions over time.
    dt : float
        Time step between samples in seconds.
    min_speed : float, default 0.0
        Minimum speed threshold. Samples with speed below this are
        interpolated from surrounding valid samples.
    smoothing_sigma : float, default 0.0
        Gaussian smoothing sigma in samples. Applied to velocity before
        computing heading. Set to 0 to disable smoothing.

    Returns
    -------
    NDArray, shape (n_time,)
        Heading in radians at each timepoint. If all speeds are below
        threshold, returns NaN array.

    Raises
    ------
    ValueError
        If positions has fewer than 2 samples.

    Warns
    -----
    UserWarning
        If all speeds are below min_speed threshold.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.reference_frames import heading_from_velocity

    Trajectory moving East:

    >>> t = np.linspace(0, 10, 100)
    >>> positions = np.column_stack([t * 10, np.zeros_like(t)])
    >>> headings = heading_from_velocity(positions, dt=t[1] - t[0])
    >>> np.allclose(headings[10:-10], 0.0, atol=0.1)
    True

    Trajectory moving North:

    >>> positions = np.column_stack([np.zeros_like(t), t * 10])
    >>> headings = heading_from_velocity(positions, dt=t[1] - t[0])
    >>> np.allclose(headings[10:-10], np.pi / 2, atol=0.1)
    True
    """
    positions = np.asarray(positions, dtype=np.float64)

    if len(positions) < 2:
        raise ValueError(
            f"Cannot compute heading: insufficient trajectory data.\n\n"
            f"WHAT: Need at least 2 position samples, got {len(positions)}\n"
            f"WHY: Heading is computed from velocity (position change over time)\n\n"
            f"HOW to fix:\n"
            f"1. Check data filtering - may have removed too many frames\n"
            f"2. Verify trajectory isn't empty after quality control\n"
            f"3. For short events, use heading_from_body_orientation() instead"
        )

    # Compute velocity via finite differences
    velocity = np.diff(positions, axis=0) / dt

    # Pad velocity to match positions length
    velocity = np.vstack([velocity, velocity[-1:]])

    # Apply Gaussian smoothing if requested
    if smoothing_sigma > 0:
        velocity[:, 0] = gaussian_filter1d(velocity[:, 0], smoothing_sigma)
        velocity[:, 1] = gaussian_filter1d(velocity[:, 1], smoothing_sigma)

    # Compute speed
    speed = np.sqrt(velocity[:, 0] ** 2 + velocity[:, 1] ** 2)

    # Compute heading
    heading = np.arctan2(velocity[:, 1], velocity[:, 0])

    # Mask low-speed periods
    low_speed_mask = speed < min_speed

    if np.all(low_speed_mask):
        warnings.warn(
            f"All speeds ({speed.max():.4f}) are below min_speed threshold "
            f"({min_speed}). Returning NaN array.",
            UserWarning,
            stacklevel=2,
        )
        return np.full(len(positions), np.nan)

    if np.any(low_speed_mask):
        # Interpolate heading for low-speed periods using circular interpolation
        heading = _interpolate_heading_circular(heading, low_speed_mask)

    return heading


def _interpolate_heading_circular(
    heading: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Interpolate heading values using circular (unit vector) interpolation.

    This avoids discontinuities at the +/-pi boundary.

    Parameters
    ----------
    heading : NDArray, shape (n_time,)
        Heading values in radians.
    mask : NDArray, shape (n_time,)
        Boolean mask where True indicates values to interpolate.

    Returns
    -------
    NDArray, shape (n_time,)
        Heading with masked values interpolated.
    """
    if not np.any(mask):
        return heading

    # Convert to unit vectors
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)

    # Get indices
    valid_indices = np.where(~mask)[0]
    invalid_indices = np.where(mask)[0]

    if len(valid_indices) == 0:
        return heading

    # Interpolate unit vector components
    cos_interp = np.interp(invalid_indices, valid_indices, cos_h[valid_indices])
    sin_interp = np.interp(invalid_indices, valid_indices, sin_h[valid_indices])

    # Convert back to angle
    result: NDArray[np.float64] = heading.copy()
    result[mask] = np.arctan2(sin_interp, cos_interp)

    return result


def heading_from_body_orientation(
    nose: NDArray[np.float64],
    tail: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute heading from nose and tail keypoints.

    Parameters
    ----------
    nose : NDArray, shape (n_time, 2)
        Nose keypoint positions. May contain NaN values.
    tail : NDArray, shape (n_time, 2)
        Tail keypoint positions. May contain NaN values.

    Returns
    -------
    NDArray, shape (n_time,)
        Heading in radians at each timepoint. NaN keypoints are
        interpolated using circular interpolation.

    Raises
    ------
    ValueError
        If all keypoints are NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.reference_frames import heading_from_body_orientation

    Heading from nose/tail pointing East:

    >>> n = 50
    >>> nose = np.tile([10.0, 0.0], (n, 1))
    >>> tail = np.tile([0.0, 0.0], (n, 1))
    >>> headings = heading_from_body_orientation(nose, tail)
    >>> np.allclose(headings, 0.0)
    True
    """
    nose = np.asarray(nose, dtype=np.float64)
    tail = np.asarray(tail, dtype=np.float64)

    # Compute body vector: nose - tail
    body_vector = nose - tail

    # Identify NaN samples
    nan_mask = np.any(np.isnan(body_vector), axis=1)

    if np.all(nan_mask):
        raise ValueError(
            "Cannot compute heading: all keypoints are NaN.\n\n"
            "WHAT: Both nose and tail positions are NaN at all timepoints\n"
            "WHY: Need at least one valid (nose, tail) pair for body orientation\n\n"
            "HOW to fix:\n"
            "1. Check pose estimation output for tracking failures\n"
            "2. Verify keypoint extraction completed successfully\n"
            "3. Consider using heading_from_velocity() if pose data unavailable"
        )

    # Compute heading where valid
    heading = np.arctan2(body_vector[:, 1], body_vector[:, 0])

    # Interpolate NaN samples using circular interpolation
    if np.any(nan_mask):
        heading = _interpolate_heading_circular(heading, nan_mask)

    return heading
