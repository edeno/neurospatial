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
from numpy.typing import ArrayLike, NDArray

from neurospatial._velocity import DEFAULT_SPEED_SMOOTHING
from neurospatial.geometry import CallIndexes, locate, require_cartesian

if TYPE_CHECKING:
    from neurospatial.ops._protocols import GraphLookupConsumer

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
    bearing = _wrap_bearing(bearing)

    return bearing


def _wrap_bearing(angle: NDArray[np.float64]) -> NDArray[np.float64]:
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
    env: GraphLookupConsumer | None = None,
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
    CartesianOnlyError
        If metric="geodesic" and ``env`` is a polar environment.

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

        require_cartesian(
            env.coordinate_system,
            "compute_egocentric_distance(metric='geodesic')",
            why="positions and targets are allocentric Cartesian positions, located "
            "in the environment's bins.",
            fix="use an allocentric environment, or metric='euclidean' for "
            "straight-line distances",
        )
        binning, arena, graph = env.binning, env.arena, env.connectivity

        # Use geodesic distance field for graph-based distances
        distances = np.full((n_time, n_targets), np.nan, dtype=np.float64)

        # Vectorized: get bin indices for all positions at once
        indexes = CallIndexes()
        call = "compute_egocentric_distance(metric='geodesic')"
        pos_bins = locate(binning, arena, positions, indexes, call=call)

        # Hoist target-bin computation out of the per-target inner loop: map
        # every target to its bin in a single batched ``bin_at`` call. For
        # static targets (broadcast from 2D) this collapses ``n_time * n_targets``
        # redundant lookups into one. Shape: (n_time, n_targets).
        target_bins = locate(
            binning, arena, targets_3d.reshape(-1, 2), indexes, call=call
        ).reshape(n_time, n_targets)

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
                        graph, sources=[target_bin]
                    )
                distances[t, i] = distance_field_cache[target_bin][pos_bin]

    return distances


def _validate_velocity_positions(
    positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Reject positions without x and y columns, naming the shape as passed.

    Call it on the caller's array, before any reshaping, so the message
    shows the shape the scientist passed.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError(
            f"positions must be a 2-D array with at least x/y coordinates; "
            f"got shape {positions.shape}.\n"
            "Why: a movement heading is the direction of travel in the x-y "
            "plane, so it needs both x and y; 1-D positions (such as a "
            "linearized track) have only one coordinate.\n"
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
            "Fix: pass heading_from_velocity(times, positions) with a numeric "
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
            "Fix: pass heading_from_velocity(times, positions) with a 1-D "
            "timestamp array of matching length; remove non-finite samples "
            "and sort/de-duplicate positions and times together."
        )
    return times


def _velocity_heading_and_speed(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    interval_mask: NDArray[np.bool_],
    smooth_window: float = DEFAULT_SPEED_SMOOTHING,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Movement heading and horizontal speed from the smoothed velocity.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Finite, strictly increasing timestamps in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Position coordinates in environment units; heading and speed use the
        first two (x, y).
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        Observed intervals supplied by the shared mask helper.
    smooth_window : float, default=0.1
        Gaussian standard deviation in seconds applied to each velocity
        component within each run (:func:`neurospatial.behavior.compute_speed`).

    Returns
    -------
    heading, speed : ndarray, shape (n_samples,)
        Radians and position units per second; samples in no run are NaN.
    """
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
    return _movement_heading_and_speed(
        times, positions, interval_mask, smooth_window=smooth_window
    )


def _movement_heading_and_speed(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
    *,
    smooth_window: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Direction and horizontal speed of the smoothed velocity; no validation.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Strictly increasing timestamps in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Positions in environment units, ``n_dims >= 2``; only x and y are used.
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        Usable intervals.
    smooth_window : float
        Gaussian standard deviation in seconds (see ``smoothed_velocity``).

    Returns
    -------
    heading, speed : ndarray, shape (n_samples,)
        ``arctan2(vy, vx)`` in radians and ``hypot(vx, vy)`` in position units
        per second; NaN at samples in no run.
    """
    from neurospatial._velocity import smoothed_velocity

    velocity = smoothed_velocity(
        times, positions[:, :2], interval_mask, smooth_window=smooth_window
    )
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    heading = np.arctan2(velocity[:, 1], velocity[:, 0])
    return heading, speed


def heading_from_velocity(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    min_speed: float = 0.0,
    smooth_window: float = DEFAULT_SPEED_SMOOTHING,
    allow_all_nan: bool = False,
) -> NDArray[np.float64]:
    """Compute heading from position timeseries using velocity direction.

    Parameters
    ----------
    times : array-like, shape (n_time,)
        Finite, strictly increasing timestamps in seconds, one per position.
    positions : NDArray, shape (n_time, 2)
        Animal positions over time, in environment units (e.g. cm). A row with
        a non-finite coordinate (NaN, for a lost frame) is a missing sample:
        it has no heading, and no velocity is computed across it.
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from the analysis. ``None`` disables the gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted. Windows that
        touch merge into one, so to keep recordings joined on one clock apart
        use ``break_before``.
    break_before : ArrayLike of bool, shape (n_time,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. No velocity or smoothing spans a declared
        break.
    min_speed : float, default 0.0
        Minimum speed threshold in **the same units per second as
        ``positions``** (e.g. cm/s if positions are in cm). Samples slower
        than this have no heading (NaN). Samples with zero speed have no
        heading at any threshold. The velocity is smoothed, so at the default
        of 0 most of a short stop still has a heading; set ``min_speed`` (for
        example 2 cm/s) to mark stops as having no heading.
    smooth_window : float, default 0.1
        Standard deviation, in seconds, of the Gaussian that smooths the
        velocity before its direction and speed are taken, the same
        smoothing as :func:`neurospatial.behavior.compute_speed`; without
        it, tracking jitter at rest gives random headings that pass
        ``min_speed``. Set to 0 to disable smoothing.
    allow_all_nan : bool, default False
        Controls the degenerate case where **every observed** sample is below
        ``min_speed`` or still (heading undefined everywhere). ``False`` (the
        default) raises ``ValueError`` so the failure is loud; ``True`` returns
        an all-NaN array with a ``UserWarning`` instead, for batch pipelines
        that handle NaN explicitly.

    Returns
    -------
    NDArray, shape (n_time,)
        Heading in radians at each timepoint, in the **allocentric
        world-frame convention** (0 = East, π/2 = North, π = West,
        -π/2 = South), wrapped to ``[-π, π]`` per ``numpy.arctan2``
        (so westward motion returns +π, not -π). NaN at missing samples, at
        samples in no run (an isolated sample, or a single-sample input), and
        at samples slower than ``min_speed`` or with zero speed.

    Raises
    ------
    ValueError
        If there are no samples, if positions is not 2-D with at least x/y
        coordinates, if times is not 1-D, finite, strictly increasing (no
        repeated timestamp) and aligned with positions, if ``break_before`` is
        not a Boolean array with one entry per sample, or if every observed
        sample is below ``min_speed`` or still and ``allow_all_nan`` is
        ``False`` (the default).

    Warns
    -----
    UserWarning
        If every observed sample is below ``min_speed`` or still and
        ``allow_all_nan=True`` (an all-NaN heading array is returned).

    Notes
    -----
    Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``
    and not crossing a ``break_before``) is analyzed as a separate recording;
    no velocity or heading spans a pause or a declared break.

    Velocity is the centered secant of position over each sample's actual
    elapsed time (one-sided at a run's ends), smoothed separately within each
    run. Samples touching no valid interval have NaN headings. If no samples
    belong to any run, the result is all NaN.

    Movement heading is the direction of the velocity vector, so it is not
    observed while the animal is slower than ``min_speed`` or still, and the
    function returns NaN there rather than an interpolated direction the
    animal may never have faced. For a heading while the animal is still, use
    :func:`heading_from_body_orientation` with pose keypoints.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.ops.egocentric import heading_from_velocity

    Trajectory moving East:

    >>> t = np.linspace(0, 10, 100)
    >>> positions = np.column_stack([t * 10, np.zeros_like(t)])
    >>> headings = heading_from_velocity(t, positions)
    >>> np.allclose(headings[10:-10], 0.0, atol=0.1)
    True

    Trajectory moving North:

    >>> positions = np.column_stack([np.zeros_like(t), t * 10])
    >>> headings = heading_from_velocity(t, positions)
    >>> np.allclose(headings[10:-10], np.pi / 2, atol=0.1)
    True

    Two recordings joined on one clock: the animal runs East, then is put
    back 40 cm to the west and runs North. Undeclared, the jump reads as
    running West; declared with ``break_before``, it does not.

    >>> times = np.arange(8) / 4  # s
    >>> x = np.array([0.0, 1.0, 2.0, 3.0, -37.0, -37.0, -37.0, -37.0])  # cm
    >>> y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0])  # cm
    >>> positions = np.column_stack([x, y])
    >>> new_recording = times == 1.0
    >>> np.round(heading_from_velocity(times, positions, smooth_window=0.0), 2).tolist()
    [0.0, 0.0, 0.0, 3.14, 3.12, 1.57, 1.57, 1.57]
    >>> np.round(
    ...     heading_from_velocity(
    ...         times, positions, smooth_window=0.0, break_before=new_recording
    ...     ),
    ...     2,
    ... ).tolist()
    [0.0, 0.0, 0.0, 0.0, 1.57, 1.57, 1.57, 1.57]

    A lost frame is a NaN row. It has no heading, and no velocity is
    computed across it:

    >>> times = np.arange(8) * 0.1  # s
    >>> positions = np.column_stack([np.arange(8.0), np.zeros(8)])  # cm
    >>> positions[3] = np.nan
    >>> heading_from_velocity(times, positions, smooth_window=0.0).tolist()
    [0.0, 0.0, 0.0, nan, 0.0, 0.0, 0.0, 0.0]

    The animal runs East, stops for 0.3 s and runs back West. While it is
    slower than ``min_speed`` its direction of movement is not observed:

    >>> times = np.arange(7) * 0.1  # s
    >>> positions = np.column_stack(
    ...     [[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0], np.zeros(7)]
    ... )  # cm
    >>> np.round(heading_from_velocity(times, positions, min_speed=2.0), 2).tolist()
    [0.0, 0.0, nan, nan, 3.14, 3.14, 3.14]
    """
    from neurospatial._observation import validated_observations

    # Validate first so a swapped (positions, times) call gets that message.
    raw_positions = positions
    times, positions, support = validated_observations(
        times,
        raw_positions,
        call="heading_from_velocity",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="every heading is NaN",
    )
    _validate_velocity_positions(raw_positions)
    heading, speed = _movement_heading_and_speed(
        times, positions, support.interval_mask, smooth_window=smooth_window
    )
    # The velocity direction is observed only where the animal moves.
    moving = (speed > 0) & (speed >= min_speed)
    heading[~moving] = np.nan
    in_run = support.run_ids >= 0
    if in_run.any() and not moving.any():
        fastest = float(np.max(speed[in_run]))
        if not allow_all_nan:
            fix = (
                "lower min_speed in position-units per second, or pass "
                "allow_all_nan=True for a batch pipeline that handles NaN."
                if min_speed > 0
                else "the animal never moves, so pass allow_all_nan=True for a "
                "batch pipeline that handles NaN, or use "
                "heading_from_body_orientation() with pose keypoints."
            )
            raise ValueError(
                f"Cannot compute heading: every observed sample's speed is "
                f"below min_speed={min_speed} or zero (the fastest is "
                f"{fastest:.4g}).\n"
                "Why: velocity direction is undefined for an all-stationary "
                f"or too-slow trajectory.\nFix: {fix}"
            )
        warnings.warn(
            f"Every observed sample's speed (max {fastest:.4g}) is below "
            f"min_speed={min_speed} or zero; returning an all-NaN heading "
            "array because allow_all_nan=True.",
            UserWarning,
            stacklevel=2,
        )
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

    Notes
    -----
    NaN keypoints are interpolated by sample index across any gap, including a
    gap that spans a recording break; this function has no ``times`` argument
    and cannot see breaks. When several recordings are concatenated, call it
    once per recording segment, splitting where a recording-id array (or a
    ``break_before`` array) changes, so that no fill bridges two recordings::

        starts = np.flatnonzero(break_before)  # break_before[0] is ignored
        segments = np.split(np.arange(len(nose)), starts[starts > 0])
        headings = np.concatenate(
            [heading_from_body_orientation(nose[s], tail[s]) for s in segments]
        )

    Do not split at every missing frame (for example at the runs of
    ``compute_cumulative_distance``): that would leave the NaN frames unfilled.
    If a gap within a recording is too long for the filled heading to be
    trusted, set the samples inside it back to NaN afterwards.

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
