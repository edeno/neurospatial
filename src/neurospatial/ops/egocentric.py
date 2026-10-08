"""Egocentric coordinate reference frame transformations.

Import paths
------------
::

    from neurospatial.ops.egocentric import allocentric_to_egocentric
    from neurospatial.ops import egocentric

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
::

    Allocentric (world):              Egocentric (animal-centered):
          North                              Left
           π/2                                π/2
            |                                  |
    West----+----East                 Back----+----Ahead
      π     |     0                    ±π     |      0
            |                                  |
          South                              Right
          -π/2                               -π/2

**Allocentric (world-centered)**:
- 0 radians = East (+x direction)
- pi/2 radians = North (+y direction)
- Standard mathematical convention

**Egocentric (animal-centered)**:
- Origin at animal's position
- +x axis = forward (heading direction)
- +y axis = left (90 degrees counterclockwise from heading)
- Angles: 0=ahead, pi/2=left, -pi/2=right, +/-pi=behind

**Example**: Animal at (0,0) facing East (heading=0), object at (10, 10):
- Allocentric bearing to object: π/4 (45° from East toward North)
- Egocentric bearing to object: π/4 (45° left of ahead)

Examples
--------
Transform landmark positions to egocentric coordinates:

>>> from neurospatial.ops.egocentric import allocentric_to_egocentric
>>> import numpy as np
>>> landmarks = np.array([[10.0, 0.0], [0.0, 10.0]])  # 2 landmarks
>>> positions = np.array([[0.0, 0.0]])  # Animal at origin
>>> headings = np.array([0.0])  # Facing East
>>> ego = allocentric_to_egocentric(positions, headings, landmarks)
>>> ego.shape
(1, 2, 2)

References
----------
.. [1] Wang, C., et al. (2018). Egocentric coding of external items in the
       lateral entorhinal cortex. Science, 362, 945-949.
       https://doi.org/10.1126/science.aau4940
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
    >>> from neurospatial.ops.egocentric import EgocentricFrame

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
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Batch transform allocentric points to egocentric coordinates.

    Parameters
    ----------
    positions : NDArray, shape (n_time, 2)
        Animal position at each time.
    headings : NDArray, shape (n_time,)
        Animal heading at each time (radians).
    points : NDArray, shape (n_points, 2) or (n_time, n_points, 2)
        Points to transform. If 2D, same points transformed at each time.

    Returns
    -------
    NDArray, shape (n_time, n_points, 2)
        Points in egocentric coordinates at each timepoint.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.ops.egocentric import allocentric_to_egocentric

    Transform landmarks at multiple timepoints:

    >>> landmarks = np.array([[10.0, 0.0], [0.0, 10.0]])  # 2 landmarks
    >>> positions = np.array([[0.0, 0.0], [0.0, 0.0]])  # Animal at origin
    >>> headings = np.array([0.0, np.pi / 2])  # Facing East, then North
    >>> ego = allocentric_to_egocentric(positions, headings, landmarks)
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
            f"Fix:\n"
            f"1. Reshape your array: points.reshape(-1, 2)\n"
            f"2. Check data loading - array may have been squeezed\n"
            f"3. Verify you're passing an array of points, not a single point"
        )

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            f"Cannot transform: invalid positions shape {positions.shape}.\n\n"
            f"WHAT: positions must have shape (n_time, 2)\n"
            f"WHY: Need (x, y) position at each timepoint for the transform origin\n\n"
            f"Fix:\n"
            f"1. Reshape: positions.reshape(-1, 2)\n"
            f"2. For single position: positions.reshape(1, 2)"
        )

    if headings.ndim != 1 or len(headings) != len(positions):
        raise ValueError(
            f"Headings/positions length mismatch.\n\n"
            f"WHAT: headings shape {headings.shape} != positions length {len(positions)}\n"
            f"WHY: Need one heading per timepoint for coordinate rotation\n\n"
            f"Fix:\n"
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
            f"Fix:\n"
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
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    ego_points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Batch transform egocentric points to allocentric coordinates.

    This is the inverse of allocentric_to_egocentric.

    Parameters
    ----------
    positions : NDArray, shape (n_time, 2)
        Animal position at each time in allocentric coordinates.
    headings : NDArray, shape (n_time,)
        Animal heading at each time (radians).
    ego_points : NDArray, shape (n_time, n_points, 2)
        Points in egocentric coordinates.

    Returns
    -------
    NDArray, shape (n_time, n_points, 2)
        Points in allocentric coordinates.

    Raises
    ------
    ValueError
        If ``ego_points`` has the wrong shape, ``positions`` is not
        ``(n_time, 2)``, or ``headings`` length does not match ``positions``.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.ops.egocentric import (
    ...     allocentric_to_egocentric,
    ...     egocentric_to_allocentric,
    ... )

    Round-trip transformation preserves coordinates:

    >>> landmarks = np.array([[10.0, 0.0], [0.0, 10.0]])
    >>> positions = np.array([[5.0, 5.0]])
    >>> headings = np.array([np.pi / 4])
    >>> ego = allocentric_to_egocentric(positions, headings, landmarks)
    >>> recovered = egocentric_to_allocentric(positions, headings, ego)
    >>> np.allclose(recovered[0], landmarks)
    True
    """
    ego_points = np.asarray(ego_points, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)

    # Validate shapes (mirrors allocentric_to_egocentric).
    if ego_points.ndim != 3 or ego_points.shape[2] != 2:
        raise ValueError(
            f"Cannot transform ego_points: invalid shape {ego_points.shape}.\n\n"
            f"WHAT: ego_points must be 3D with shape (n_time, n_points, 2)\n"
            f"WHY: Each point needs (x, y) egocentric coordinates per timepoint\n\n"
            f"Fix:\n"
            f"1. Reshape your array to (n_time, n_points, 2)\n"
            f"2. This is the shape returned by allocentric_to_egocentric\n"
            f"Got shape: {ego_points.shape}"
        )

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            f"Cannot transform: invalid positions shape {positions.shape}.\n\n"
            f"WHAT: positions must have shape (n_time, 2)\n"
            f"WHY: Need (x, y) position at each timepoint for the transform origin\n\n"
            f"Fix:\n"
            f"1. Reshape: positions.reshape(-1, 2)\n"
            f"2. For single position: positions.reshape(1, 2)"
        )

    if headings.ndim != 1 or len(headings) != len(positions):
        raise ValueError(
            f"Headings/positions length mismatch.\n\n"
            f"WHAT: headings shape {headings.shape} != positions length {len(positions)}\n"
            f"WHY: Need one heading per timepoint for coordinate rotation\n\n"
            f"Fix:\n"
            f"1. Ensure headings and positions are aligned to same timepoints\n"
            f"2. Check for off-by-one errors in slicing\n"
            f"3. Interpolate headings to match positions if sampled differently"
        )

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
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute bearing angle to targets in egocentric coordinates.

    The bearing is the angle to the target relative to the animal's heading,
    where 0=ahead, pi/2=left, -pi/2=right, +/-pi=behind.

    Parameters
    ----------
    positions : NDArray, shape (n_time, 2)
        Animal position at each time.
    headings : NDArray, shape (n_time,)
        Animal heading at each time (radians).
    targets : NDArray, shape (n_targets, 2) or (n_time, n_targets, 2)
        Target positions in allocentric coordinates.

    Returns
    -------
    NDArray, shape (n_time, n_targets)
        Bearing to each target at each timepoint in radians.
        Range: (-pi, pi], where 0=ahead, pi/2=left, -pi/2=right.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.ops.egocentric import compute_egocentric_bearing

    Target directly ahead has bearing 0:

    >>> target = np.array([[10.0, 0.0]])
    >>> position = np.array([[0.0, 0.0]])
    >>> heading = np.array([0.0])  # Facing East
    >>> bearing = compute_egocentric_bearing(position, heading, target)
    >>> np.allclose(bearing, [[0.0]])
    True

    Target to the left has bearing pi/2:

    >>> target = np.array([[0.0, 10.0]])
    >>> bearing = compute_egocentric_bearing(position, heading, target)
    >>> np.allclose(bearing, [[np.pi / 2]])
    True
    """
    # Transform targets to egocentric coordinates
    ego = allocentric_to_egocentric(positions, headings, targets)

    # Compute bearing using arctan2
    bearing = np.arctan2(ego[..., 1], ego[..., 0])

    # Wrap to (-pi, pi]
    bearing = _wrap_angle(bearing)

    return bearing


def _wrap_angle(angle: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wrap angles to (-pi, pi].

    Parameters
    ----------
    angle : NDArray
        Angles in radians.

    Returns
    -------
    NDArray
        Angles wrapped to the half-open interval (-pi, pi] (a target directly
        behind the animal returns +pi, never -pi).
    """
    # ((angle - pi) % (2*pi)) - (-pi)  shifts so the open end is at -pi and the
    # closed end at +pi, giving the documented (-pi, pi] half-open interval.
    wrapped = (angle - np.pi) % (-2 * np.pi) + np.pi
    return wrapped


def compute_egocentric_distance(
    positions: NDArray[np.float64],
    headings: NDArray[np.float64] | None,
    targets: NDArray[np.float64],
    *,
    metric: str = "euclidean",
    env: Environment | None = None,
) -> NDArray[np.float64]:
    """Compute distance to targets from animal position.

    Distance is symmetric (does not depend on heading), but headings parameter
    is included as a positional parameter to maintain API consistency with other
    egocentric functions like compute_egocentric_bearing().

    Parameters
    ----------
    positions : NDArray, shape (n_time, 2)
        Animal position at each time.
    headings : NDArray, shape (n_time,), optional
        Animal heading at each time (radians). Not used for distance
        calculation but included for API consistency with canonical argument
        order (positions, headings, targets).
    targets : NDArray, shape (n_targets, 2) or (n_time, n_targets, 2)
        Target positions in allocentric coordinates.
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
    >>> from neurospatial.ops.egocentric import compute_egocentric_distance

    Euclidean distance:

    >>> targets = np.array([[10.0, 0.0], [0.0, 10.0]])
    >>> position = np.array([[0.0, 0.0]])
    >>> heading = np.array([0.0])
    >>> distances = compute_egocentric_distance(
    ...     position, heading, targets, metric="euclidean"
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
            f"Fix:\n"
            f"1. Use 'euclidean' for straight-line distances (default, faster)\n"
            f"2. Use 'geodesic' for boundary-respecting distances (requires env)"
        )

    if metric == "geodesic" and env is None:
        raise ValueError(
            "Cannot compute geodesic distances: missing environment.\n\n"
            "WHAT: metric='geodesic' requires env parameter\n"
            "WHY: Geodesic distances follow paths that respect environment boundaries\n\n"
            "Fix:\n"
            "1. Pass the environment: compute_egocentric_distance(..., env=env)\n"
            "2. Or use 'euclidean' for straight-line distances:\n"
            "   compute_egocentric_distance(..., metric='euclidean')"
        )

    n_time = len(positions)

    # Expand targets to 3D if needed
    if targets.ndim == 2:
        targets_3d = np.broadcast_to(targets, (n_time, targets.shape[0], 2))
    elif targets.ndim == 3:
        if targets.shape[0] != n_time:
            raise ValueError(
                f"targets time axis {targets.shape[0]} does not match positions "
                f"length {n_time}.\n\n"
                f"WHAT: a 3D targets array must have shape (n_time, n_targets, 2)\n"
                f"WHY: each timepoint's distance is computed against that "
                f"timepoint's targets\n\n"
                f"Fix:\n"
                f"1. Pass static targets as a 2D (n_targets, 2) array, or\n"
                f"2. Make targets.shape[0] equal len(positions)"
            )
        targets_3d = targets
    else:
        raise ValueError(
            f"targets must be 2D (n_targets, 2) or 3D (n_time, n_targets, 2), "
            f"got shape {targets.shape}"
        )

    n_targets = targets_3d.shape[1]

    distances: NDArray[np.float64]

    if metric == "euclidean":
        # Compute Euclidean distances
        # targets_3d: (n_time, n_targets, 2)
        # positions: (n_time, 2) -> (n_time, 1, 2)
        diff = targets_3d - positions[:, np.newaxis, :]
        distances = np.sqrt(np.sum(diff**2, axis=-1))

    else:  # geodesic
        from neurospatial.ops.distance import distance_field as compute_distance_field

        # env is guaranteed non-None here (validated above)
        assert env is not None

        # Use geodesic distance field for graph-based distances
        distances = np.full((n_time, n_targets), np.nan, dtype=np.float64)

        # Vectorized: get bin indices for all positions at once
        pos_bins = env.bin_at(positions)

        # Hoist target-bin computation out of the per-target inner loop: map
        # every target to its bin in a single batched ``bin_at`` call. For
        # static targets (broadcast from 2D) this collapses ``n_time * n_targets``
        # redundant lookups into one. Shape: (n_time, n_targets).
        target_bins = env.bin_at(targets_3d.reshape(-1, 2)).reshape(n_time, n_targets)

        # Cache distance fields by target bin: static targets recompute once;
        # time-varying targets pay only for unique target bins.
        distance_field_cache: dict[int, NDArray[np.float64]] = {}

        for t in range(n_time):
            pos_bin = int(pos_bins[t])
            if pos_bin < 0:
                # Position outside environment - row stays NaN
                continue
            for i in range(n_targets):
                target_bin = int(target_bins[t, i])
                if target_bin < 0:
                    # Target outside environment - cell stays NaN
                    continue
                if target_bin not in distance_field_cache:
                    distance_field_cache[target_bin] = compute_distance_field(
                        env.connectivity, sources=[target_bin]
                    )
                distances[t, i] = distance_field_cache[target_bin][pos_bin]

    return distances


def _validate_velocity_positions(
    positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Reject shapes that cannot supply aligned x/y velocity components."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError(
            f"positions must be a 2-D array with at least x/y coordinates; "
            f"got shape {positions.shape}.\n"
            "Why: per-interval timestamps must align with position rows; "
            "a 1-D array would broadcast into a square velocity matrix.\n"
            "Fix: pass positions with shape (n_samples, 2), for example "
            "np.column_stack([x, y]), with one timestamp per row."
        )
    return positions


def _validate_velocity_times(
    times: NDArray[np.float64], n_samples: int
) -> NDArray[np.float64]:
    """Validate the timestamp array before building an interval mask."""
    try:
        times = np.asarray(times, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"times must be numeric timestamps; got {times!r}.\n"
            "Why: velocities need seconds between consecutive samples.\n"
            "Fix: pass heading_from_velocity(positions, times) with a numeric "
            "1-D timestamp array, one timestamp per position."
        ) from error
    problems = []
    if times.ndim != 1:
        problems.append(f"times must be 1-D; got shape {times.shape}")
    else:
        if len(times) != n_samples:
            problems.append(
                f"times and positions must have the same length; got "
                f"{len(times)} times and {n_samples} positions"
            )
        finite = np.isfinite(times)
        if not finite.all():
            index = int(np.flatnonzero(~finite)[0])
            problems.append(
                f"times must be finite; got {times[index]!r} at index {index}"
            )
        elif np.any(np.diff(times) <= 0):
            index = int(np.flatnonzero(np.diff(times) <= 0)[0])
            problems.append(
                f"times must be strictly increasing; got "
                f"{times[index]!r}, {times[index + 1]!r} at indices "
                f"{index}, {index + 1}"
            )
    if problems:
        raise ValueError(
            "; ".join(problems) + ".\n"
            "Why: each position needs a finite timestamp and a positive "
            "elapsed interval for velocity.\n"
            "Fix: pass heading_from_velocity(positions, times) with a 1-D "
            "timestamp array of matching length; remove non-finite samples "
            "and sort/de-duplicate positions and times together."
        )
    return times


def _velocity_heading_and_speed(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    interval_mask: NDArray[np.bool_],
    bandwidth: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute forward velocity headings and speeds separately per recording.

    Parameters
    ----------
    positions : ndarray, shape (n_samples, 2)
        Position coordinates in environment units.
    times : ndarray, shape (n_samples,)
        Finite, strictly increasing timestamps in seconds.
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        Observed intervals supplied by the shared mask helper.
    bandwidth : float, default=0.0
        Gaussian smoothing sigma in samples, applied separately per run.

    Returns
    -------
    heading, speed : ndarray, shape (n_samples,)
        Radians and position units per second. The final interval's velocity
        is repeated at each run's last sample; samples in no run are NaN.
    """
    from neurospatial._intervals import run_sample_bounds
    from neurospatial._validation import validate_finite

    positions = _validate_velocity_positions(positions)
    times = _validate_velocity_times(times, len(positions))
    validate_finite(positions, name="positions")
    if len(positions) < 2:
        raise ValueError(
            f"Cannot compute heading: insufficient trajectory data; need at "
            f"least 2 position samples, got {len(positions)}.\n"
            "Why: heading needs a position change over time.\n"
            "Fix: pass at least two aligned position/timestamp samples or "
            "use heading_from_body_orientation() for single-frame pose data."
        )
    headings: NDArray[np.float64] = np.full(len(positions), np.nan)
    speeds: NDArray[np.float64] = np.full(len(positions), np.nan)
    for first, last in run_sample_bounds(interval_mask):
        run = slice(first, last + 1)
        velocity: NDArray[np.float64] = (
            np.diff(positions[run], axis=0) / np.diff(times[run])[:, None]
        )
        velocity = np.vstack([velocity, velocity[-1:]])
        if bandwidth > 0:
            velocity[:, 0] = gaussian_filter1d(velocity[:, 0], bandwidth)
            velocity[:, 1] = gaussian_filter1d(velocity[:, 1], bandwidth)
        speeds[run] = np.sqrt(velocity[:, 0] ** 2 + velocity[:, 1] ** 2)
        headings[run] = np.arctan2(velocity[:, 1], velocity[:, 0])
    return headings, speeds


def heading_from_velocity(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    min_speed: float = 0.0,
    bandwidth: float = 0.0,
    allow_all_nan: bool = False,
) -> NDArray[np.float64]:
    """Compute heading from position timeseries using velocity direction.

    Parameters
    ----------
    positions : NDArray, shape (n_time, 2)
        Animal positions over time, in environment units (e.g. cm).
    times : array-like, shape (n_time,)
        Finite, strictly increasing timestamps in seconds, one per position.
    min_speed : float, default 0.0
        Minimum speed threshold in **the same units per second as
        ``positions``** (e.g. cm/s if positions are in cm). Samples
        with speed below this are interpolated along the shorter arc,
        linearly in angle between surrounding valid samples.
    bandwidth : float, default 0.0
        Gaussian smoothing sigma in samples. Applied to velocity before
        computing heading. Set to 0 to disable smoothing.
    allow_all_nan : bool, default False
        Controls the degenerate case where **every observed** sample is below
        ``min_speed`` (heading undefined everywhere). ``False`` (the default)
        raises ``ValueError`` so the failure is loud; ``True`` returns an
        all-NaN array with a ``UserWarning`` instead, for batch pipelines that
        handle NaN explicitly.
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from the analysis. ``None`` disables the gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.

    Returns
    -------
    NDArray, shape (n_time,)
        Heading in radians at each timepoint, in the **allocentric
        world-frame convention** (0 = East, π/2 = North, π = West,
        -π/2 = South), wrapped to ``[-π, π]`` per ``numpy.arctan2``
        (so westward motion returns +π, not -π). Samples below ``min_speed``
        are interpolated along the shorter arc from surrounding valid samples.

    Raises
    ------
    ValueError
        If positions has fewer than 2 samples, contains non-finite values,
        or is not 2-D with at least x/y coordinates,
        if times is not 1-D, finite, strictly increasing and aligned, or if
        every observed sample is
        below ``min_speed`` and ``allow_all_nan`` is ``False`` (the default).

    Warns
    -----
    UserWarning
        If every observed sample is below ``min_speed`` and ``allow_all_nan=True``
        (an all-NaN heading array is returned).

    Notes
    -----
    Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``)
    is analyzed as a separate recording; no velocity or heading spans a pause.

    Velocity uses each interval's actual elapsed time. Smoothing bandwidth
    remains Gaussian sigma in samples and is applied separately per run.
    Low-speed interpolation uses moving anchors from that run only. A run
    with no moving anchors and samples touching no valid interval have NaN
    headings. If no samples belong to any run, the result is all NaN.

    Heading is computed from the forward finite difference of position, which
    yields ``n_time - 1`` velocity samples for ``n_time`` positions. To return
    an array aligned to ``positions`` (length ``n_time``), each run's last sample's
    heading is forward-padded: ``heading[-1]`` is a copy of ``heading[-2]``
    rather than an independently measured value. For long trajectories this
    edge effect is negligible; for very short trajectories treat the final
    sample's heading as approximate.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.ops.egocentric import heading_from_velocity

    Trajectory moving East:

    >>> t = np.linspace(0, 10, 100)
    >>> positions = np.column_stack([t * 10, np.zeros_like(t)])
    >>> headings = heading_from_velocity(positions, t)
    >>> np.allclose(headings[10:-10], 0.0, atol=0.1)
    True

    Trajectory moving North:

    >>> positions = np.column_stack([np.zeros_like(t), t * 10])
    >>> headings = heading_from_velocity(positions, t)
    >>> np.allclose(headings[10:-10], np.pi / 2, atol=0.1)
    True
    """
    from neurospatial._intervals import run_sample_bounds
    from neurospatial.environment.trajectory import observed_interval_mask

    positions = _validate_velocity_positions(positions)
    times = _validate_velocity_times(times, len(positions))
    interval_mask = observed_interval_mask(times, max_gap=max_gap, epochs=epochs)
    heading, speed = _velocity_heading_and_speed(
        positions, times, interval_mask=interval_mask, bandwidth=bandwidth
    )
    in_run = np.r_[interval_mask, False] | np.r_[False, interval_mask]
    low_speed_mask = speed < min_speed
    if np.any(in_run) and np.all(low_speed_mask[in_run]):
        fastest = float(np.max(speed[in_run]))
        if not allow_all_nan:
            raise ValueError(
                f"Cannot compute heading: every observed sample's speed is "
                f"below min_speed={min_speed} (the fastest is {fastest:.4g}).\n"
                "Why: velocity direction is undefined for an all-stationary "
                "or too-slow trajectory.\n"
                "Fix: lower min_speed in position-units per second, or pass "
                "allow_all_nan=True for a batch pipeline that handles NaN."
            )
        warnings.warn(
            f"All observed speeds (max {fastest:.4g}) are below min_speed "
            f"threshold ({min_speed}); returning an all-NaN heading array "
            "because allow_all_nan=True.",
            UserWarning,
            stacklevel=2,
        )
        return np.full(len(positions), np.nan)

    heading[low_speed_mask] = np.nan
    for first, last in run_sample_bounds(interval_mask):
        run = slice(first, last + 1)
        heading[run] = _interpolate_heading_circular(heading[run], low_speed_mask[run])
    return heading


def _interpolate_heading_circular(
    heading: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Interpolate masked headings along the shorter arc, linearly in angle.

    Unwrap consecutive finite anchors, interpolate angles, then wrap to
    (-pi, pi]. For an exactly antipodal pair, the sign of the stored
    difference chooses the turn. Samples beyond the anchors keep the nearest
    valid heading. Unmasked non-finite values are not interpolation anchors.

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

    valid = ~mask & np.isfinite(heading)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return heading
    unwrapped = np.unwrap(heading[valid_idx])
    filled = np.interp(np.flatnonzero(mask), valid_idx, unwrapped)
    result: NDArray[np.float64] = heading.copy()
    result[mask] = np.pi - np.mod(np.pi - filled, 2.0 * np.pi)
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
        interpolated along the shorter arc, linearly in angle.

    Raises
    ------
    ValueError
        If all keypoints are NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.ops.egocentric import heading_from_body_orientation

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
            "Fix:\n"
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
