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

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurospatial._angles import wrap_angle
from neurospatial._exceptions import _format_error
from neurospatial._pynapple import reads_pynapple_samples
from neurospatial.geometry import CallIndexes, locate, require_cartesian

if TYPE_CHECKING:
    from neurospatial.ops._protocols import GraphLookupConsumer

__all__ = [
    "allocentric_to_egocentric",
    "compute_egocentric_bearing",
    "compute_egocentric_distance",
    "egocentric_to_allocentric",
]


@reads_pynapple_samples
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
        Animal heading at each time, in allocentric radians (0 = East,
        +π/2 = North).
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
        If ``points`` is not ``(n_points, 2)`` or ``(n_time, n_points, 2)``,
        including a 3D ``points`` whose time axis differs from
        ``len(positions)``; if ``positions`` is not ``(n_time, 2)``; or if
        ``headings`` length does not match ``positions``.
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

    _check_pose(
        "allocentric_to_egocentric()",
        positions,
        headings,
        why="each time's points are rotated about its position by its heading.",
    )
    n_time = len(positions)

    if points.ndim > 3:
        raise ValueError(
            f"Invalid points dimensionality: got {points.ndim}D array.\n\n"
            f"WHAT: points must be 2D (n_points, 2) or 3D (n_time, n_points, 2)\n"
            f"WHY: 2D broadcasts same points to all timepoints; 3D allows time-varying\n\n"
            f"Fix:\n"
            f"1. Static points: use shape (n_points, 2)\n"
            f"2. Time-varying: use shape (n_time, n_points, 2)\n"
            f"Got shape: {points.shape}"
        )
    if points.shape[-1] != 2:
        raise ValueError(
            _format_error(
                f"points must have 2 columns (x, y); got shape {points.shape}.",
                why="egocentric coordinates are planar: each point is rotated "
                "about the animal's position by its heading.",
                fix="pass points of shape (n_points, 2), or (n_time, n_points, 2) "
                "for points that move",
            )
        )
    if points.ndim == 3 and points.shape[0] != n_time:
        raise ValueError(
            _format_error(
                f"points time axis {points.shape[0]} does not match positions "
                f"length {n_time}.",
                why="a 3D points array holds the points at each time, shape "
                "(n_time, n_points, 2), and is transformed with that time's "
                "position and heading.",
                fix="pass points that do not move as a 2D (n_points, 2) array, "
                "or give one row of points per position",
            )
        )
    if points.ndim == 2:
        points = np.broadcast_to(points, (n_time, points.shape[0], 2))

    # Center points on the animal, then rotate by -heading into its frame.
    centered = points - positions[:, np.newaxis, :]
    result: NDArray[np.float64] = np.einsum(
        "tij,tpj->tpi", _rotations(-headings), centered
    )
    return result


@reads_pynapple_samples
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
        Animal heading at each time, in allocentric radians (0 = East,
        +π/2 = North).
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

    _check_pose(
        "egocentric_to_allocentric()",
        positions,
        headings,
        why="each time's points are rotated by its heading about its position.",
    )
    # Rotate by +heading (the inverse of allocentric_to_egocentric), then
    # move the origin back to the animal.
    rotated = np.einsum("tij,tpj->tpi", _rotations(headings), ego_points)
    result: NDArray[np.float64] = rotated + positions[:, np.newaxis, :]
    return result


def _check_pose(
    call: str,
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    why: str,
) -> None:
    """Require positions ``(n_time, 2)`` and one heading per position.

    ``call`` names the public function and ``why`` says what each heading
    is used for, in the error messages.
    """
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            _format_error(
                f"{call} needs positions of shape (n_time, 2); got {positions.shape}.",
                fix="pass one (x, y) row per sample",
            )
        )
    if headings.shape != (len(positions),):
        raise ValueError(
            _format_error(
                f"{call} needs one heading per position, shape "
                f"({len(positions)},); got {headings.shape}.",
                why=why,
                fix="pass headings with the same length as positions (NaN marks "
                "a missing heading)",
            )
        )


def _rotations(angles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Counterclockwise rotation matrix by each angle, shape (n, 2, 2)."""
    cos, sin = np.cos(angles), np.sin(angles)
    return np.stack(
        [np.stack([cos, -sin], axis=-1), np.stack([sin, cos], axis=-1)], axis=-2
    )


@reads_pynapple_samples
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
        Animal heading at each time, in allocentric radians (0 = East,
        +π/2 = North).
    targets : NDArray, shape (n_targets, 2) or (n_time, n_targets, 2)
        Target positions in allocentric coordinates.

    Returns
    -------
    NDArray, shape (n_time, n_targets)
        Bearing to each target at each timepoint in radians.
        Range: (-pi, pi], where 0=ahead, pi/2=left, -pi/2=right.

    Raises
    ------
    ValueError
        If the shapes are invalid, as for :func:`allocentric_to_egocentric`.

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
    bearing = wrap_angle(bearing)

    return bearing


@reads_pynapple_samples
def compute_egocentric_distance(
    positions: NDArray[np.float64],
    headings: NDArray[np.float64] | None,
    targets: NDArray[np.float64],
    *,
    metric: str = "euclidean",
    env: GraphLookupConsumer | None = None,
) -> NDArray[np.float64]:
    """Compute distance to targets from animal position.

    Distance does not depend on heading, but headings parameter is included
    as a positional parameter to maintain API consistency with other
    egocentric functions like compute_egocentric_bearing().

    Parameters
    ----------
    positions : NDArray, shape (n_time, 2)
        Animal position at each time.
    headings : NDArray, shape (n_time,), optional
        Animal heading at each time, in allocentric radians (0 = East,
        +π/2 = North), or None. Not used for distance
        calculation but included for API consistency with canonical argument
        order (positions, headings, targets).
    targets : NDArray, shape (n_targets, 2) or (n_time, n_targets, 2)
        Target positions in allocentric coordinates.
    metric : str, default "euclidean"
        Distance metric. Options:

        - "euclidean": Straight-line distance.
        - "geodesic": Shortest-path distance along the environment's bin
          graph, from the center of the animal's bin to the center of the
          target's bin, so it goes around walls and holes. A position outside
          the environment, or a NaN position or target (a missing
          observation), gives NaN.
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
    ValueError
        If metric="geodesic" and a finite target lies outside every bin of
        ``env``; the message names the target and its distance to the
        nearest bin center.
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
        from neurospatial.ops.distance import _geodesic_pair_distances

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

        # Map every target to its bin in one batched ``locate`` call rather
        # than one call per target and time. Shape: (n_time, n_targets).
        target_bins = locate(
            binning, arena, targets_3d.reshape(-1, 2), indexes, call=call
        ).reshape(n_time, n_targets)
        _require_targets_in_bins(
            targets_3d,
            target_bins,
            binning.bins.centers,
            time_varying=targets.ndim == 3,
        )

        # A position outside the environment or a missing (NaN) target
        # leaves its cell NaN. Dijkstra runs once per distinct target bin.
        position_bins = np.broadcast_to(pos_bins[:, np.newaxis], target_bins.shape)
        measured = (position_bins >= 0) & (target_bins >= 0)
        distances[measured] = _geodesic_pair_distances(
            graph, target_bins[measured], position_bins[measured], operation=call
        )

    return distances


def _require_targets_in_bins(
    targets: NDArray[np.float64],
    target_bins: NDArray[np.intp],
    bin_centers: NDArray[np.float64],
    *,
    time_varying: bool,
) -> None:
    """Refuse a finite target that lies outside every bin.

    Parameters
    ----------
    targets : ndarray of float64, shape (n_time, n_targets, 2)
        Target positions at each time.
    target_bins : ndarray of intp, shape (n_time, n_targets)
        Bin of each target, ``-1`` outside the environment or at NaN.
    bin_centers : ndarray of float64, shape (n_bins, 2)
        Bin centers, to report how far outside the target is.
    time_varying : bool
        Whether the caller gave targets per time, so the error names the time.

    Raises
    ------
    ValueError
        If a finite target lies outside every bin.
    """
    outside = (target_bins < 0) & np.all(np.isfinite(targets), axis=-1)
    if not outside.any():
        return
    time, target = np.argwhere(outside)[0]
    point = targets[time, target]
    coords = ", ".join(f"{value:g}" for value in point)
    gap = float(np.min(np.linalg.norm(bin_centers - point, axis=1)))
    where = f"at time index {time} " if time_varying else ""
    raise ValueError(
        _format_error(
            f"Target {target} {where}at ({coords}) lies outside every bin, so "
            "compute_egocentric_distance(metric='geodesic') has no path to it.",
            why=f"geodesic distance runs along the environment's bin graph, "
            f"from the animal's bin to the target's bin; the nearest bin center "
            f"is {gap:.3g} away. A target on the arena's edge, such as an "
            "object against a wall, can lie outside.",
            fix="move the target inside the environment, or use "
            "metric='euclidean' for straight-line distances",
        )
    )
