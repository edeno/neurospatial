"""Distance and direction from the animal to the arena's walls.

Boundary-vector cells fire when a wall lies at a preferred distance in a
preferred allocentric direction (Hartley et al., 2000; Lever et al., 2009), and
egocentric boundary cells when it lies at a preferred distance and direction
relative to the animal's heading (Hinman et al., 2019; Alexander et al.,
2020). Both are measured from the walls themselves, not from the edge of the
visited bins: an unvisited corner is not a wall.

``compute_boundary_distances`` casts one ray per direction from each sample
and returns the distance to the first wall along it, the measurement behind
boundary-vector and egocentric boundary rate maps. ``nearest_boundary`` gives
the distance and allocentric direction to the closest wall point.

Walls come from ``boundary_source``: ``"physical"`` (the default) uses the
arena's declared outline and walls, such as ``Environment.from_polygon`` or
an ``OpenArena``; ``"analysis"`` uses the edges of the environment's bins.

References
----------
Hartley, T., et al. (2000). Modeling place fields in terms of the cortical
    inputs to the hippocampus. Hippocampus, 10, 369-379.
Lever, C., et al. (2009). Boundary vector cells in the subiculum of the
    hippocampal formation. Journal of Neuroscience, 29, 9771-9777.
Hinman, J. R., et al. (2019). Neuronal representation of environmental
    boundaries in egocentric coordinates. Nature Communications, 10, 2772.
Alexander, A. S., et al. (2020). Egocentric boundary vector tuning of the
    retrosplenial cortex. Science Advances, 6, eaaz2322.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import shapely
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial.geometry import (
    CallIndexes,
    boundary_geometry,
    boundary_view,
    require_position_columns,
    search_distance,
    valid_observers,
)

if TYPE_CHECKING:
    from neurospatial.ops._protocols import BinningConsumer

__all__ = ["boundary_directions", "compute_boundary_distances", "nearest_boundary"]


def boundary_directions(n_directions: int) -> NDArray[np.float64]:
    """Ray directions, the centers of ``n_directions`` equal sectors of a turn.

    Parameters
    ----------
    n_directions : int
        Number of directions, at least 1.

    Returns
    -------
    NDArray[np.float64], shape (n_directions,)
        Directions in radians in ``(-pi, pi)``, increasing counterclockwise.
        They are the centers of the direction bins of
        ``Environment.from_polar_egocentric(angle_range=(-np.pi, np.pi), ...)``
        with ``angle_bin_size=2 * np.pi / n_directions``.
    """
    if (
        isinstance(n_directions, (bool, np.bool_))
        or not isinstance(n_directions, (int, np.integer))
        or n_directions < 1
    ):
        raise ValueError(
            _format_error(
                f"n_directions must be a positive integer, got {n_directions!r}.",
                why="Why: it is the number of rays cast around each sample.",
                fix="pass an integer such as n_directions=36 (10-degree steps)",
            )
        )
    width = 2 * np.pi / n_directions
    return np.asarray(-np.pi + width * (np.arange(n_directions) + 0.5))


def compute_boundary_distances(
    env: BinningConsumer,
    positions: NDArray[np.float64],
    headings: NDArray[np.float64] | None = None,
    *,
    direction_frame: Literal["allocentric", "egocentric"],
    n_directions: int = 36,
    max_distance: float | None = None,
    boundary_source: Literal["physical", "analysis"] = "physical",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Distance to the first wall along rays in evenly spaced directions.

    Parameters
    ----------
    env : Environment
        Allocentric 2-D environment whose arena holds the walls.
    positions : NDArray[np.float64], shape (n_time, 2)
        Animal positions in the environment's coordinates.
    headings : NDArray[np.float64], shape (n_time,), optional
        Head direction in radians (0 = East, +pi/2 = North). Required with
        ``direction_frame="egocentric"`` and refused with
        ``"allocentric"``, so a frame is never chosen by accident.
    direction_frame : {"allocentric", "egocentric"}
        Frame of the ray directions. Allocentric: 0 = East, +pi/2 = North,
        for boundary-vector cells. Egocentric: 0 = ahead, +pi/2 = left,
        -pi/2 = right, for egocentric boundary cells.
    n_directions : int, default=36
        Number of rays per sample, at the directions
        :func:`boundary_directions` returns (10-degree sectors by default).
    max_distance : float, optional
        Longest distance searched along each ray, in the environment's
        units. By default, far enough to reach every wall.
    boundary_source : {"physical", "analysis"}, default="physical"
        ``"physical"`` uses the arena's declared outline and walls;
        ``"analysis"`` uses the edges of the environment's bins, so an
        unvisited area becomes a boundary.

    Returns
    -------
    distances : NDArray[np.float64], shape (n_time, n_directions)
        Distance to the first wall along each ray, in the environment's
        units. NaN for a sample with no observation (a NaN position or
        heading, or a position outside the walls); ``inf`` where no wall lies
        within ``max_distance``.
    directions : NDArray[np.float64], shape (n_directions,)
        Ray direction of each column, in radians in ``direction_frame``.

    Raises
    ------
    ValueError
        If ``headings`` does not match ``direction_frame``, or the shapes do
        not match.
    NotImplementedError
        If the arena has no known walls and ``boundary_source="physical"``.
    CartesianOnlyError
        If ``env`` is an egocentric polar environment.

    See Also
    --------
    nearest_boundary : Distance and direction to the closest wall point.
    compute_viewed_location : First wall along the gaze direction.

    Examples
    --------
    >>> import numpy as np
    >>> from shapely.geometry import box
    >>> from neurospatial import Environment
    >>> from neurospatial.ops import compute_boundary_distances
    >>> env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=5.0)
    >>> positions = np.array([[20.0, 50.0]])  # cm
    >>> distances, directions = compute_boundary_distances(
    ...     env, positions, direction_frame="allocentric", n_directions=4
    ... )
    >>> np.degrees(directions).round().tolist()
    [-135.0, -45.0, 45.0, 135.0]
    >>> distances.round(1).tolist()  # to the west wall at x = 0 for 135 degrees
    [[28.3, 70.7, 70.7, 28.3]]
    """
    positions, headings = _check_samples(positions, headings, direction_frame)
    directions = boundary_directions(n_directions)
    binning, arena = env.binning, env.arena
    walls = boundary_geometry(binning, arena, boundary_source)
    indexes = CallIndexes()
    observed = valid_observers(binning, arena, positions, boundary_source, indexes)
    if headings is not None:
        observed &= np.isfinite(headings)
    distances = np.full((len(positions), n_directions), np.nan)
    if not observed.any():
        return distances, directions

    origins = positions[observed]
    if max_distance is None:
        max_distance = search_distance(walls, origins)
    turn = headings[observed] if headings is not None else np.zeros(len(origins))
    angles = (turn[:, None] + directions[None, :]).ravel()
    starts = np.repeat(origins, n_directions, axis=0)
    hits = boundary_view(
        binning, arena, starts, angles, max_distance, boundary_source, indexes
    )
    reached = np.hypot.reduce(hits - starts, axis=1).reshape(-1, n_directions)
    distances[observed] = np.where(np.isnan(reached), np.inf, reached)
    return distances, directions


def nearest_boundary(
    env: BinningConsumer,
    positions: NDArray[np.float64],
    *,
    boundary_source: Literal["physical", "analysis"] = "physical",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Distance and allocentric direction from each sample to the closest wall.

    Parameters
    ----------
    env : Environment
        Allocentric 2-D environment whose arena holds the walls.
    positions : NDArray[np.float64], shape (n_time, 2)
        Animal positions in the environment's coordinates.
    boundary_source : {"physical", "analysis"}, default="physical"
        ``"physical"`` uses the arena's declared outline and walls;
        ``"analysis"`` uses the edges of the environment's bins.

    Returns
    -------
    distances : NDArray[np.float64], shape (n_time,)
        Straight-line distance to the closest wall point, in the
        environment's units. NaN for a NaN position or one outside the walls.
    directions : NDArray[np.float64], shape (n_time,)
        Allocentric direction from the animal to that point in radians
        (0 = East, +pi/2 = North). Subtract the heading and wrap to get the
        egocentric direction (0 = ahead, +pi/2 = left). NaN where the
        distance is NaN; 0 for a sample on a wall.

    Raises
    ------
    NotImplementedError
        If the arena has no known walls and ``boundary_source="physical"``.
    CartesianOnlyError
        If ``env`` is an egocentric polar environment.

    See Also
    --------
    compute_boundary_distances : Distance to the first wall along each ray.

    Examples
    --------
    >>> import numpy as np
    >>> from shapely.geometry import box
    >>> from neurospatial import Environment
    >>> from neurospatial.ops import nearest_boundary
    >>> env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=5.0)
    >>> distances, directions = nearest_boundary(env, np.array([[20.0, 50.0]]))
    >>> float(distances[0]), float(np.degrees(directions[0]))  # west wall
    (20.0, 180.0)
    """
    positions, _ = _check_samples(positions, None, "allocentric")
    binning, arena = env.binning, env.arena
    walls = boundary_geometry(binning, arena, boundary_source)
    observed = valid_observers(
        binning, arena, positions, boundary_source, CallIndexes()
    )
    distances = np.full(len(positions), np.nan)
    directions = np.full(len(positions), np.nan)
    if observed.any():
        lines = shapely.shortest_line(shapely.points(positions[observed]), walls)
        ends = shapely.get_coordinates(shapely.get_point(lines, -1))
        offsets = ends - positions[observed]
        distances[observed] = np.hypot(offsets[:, 0], offsets[:, 1])
        directions[observed] = np.arctan2(offsets[:, 1], offsets[:, 0])
    return distances, directions


def _check_samples(
    positions: NDArray[np.float64],
    headings: NDArray[np.float64] | None,
    direction_frame: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Validate positions and the heading that the direction frame needs."""
    if direction_frame not in ("allocentric", "egocentric"):
        raise ValueError(
            f"direction_frame must be 'allocentric' or 'egocentric', got "
            f"{direction_frame!r}."
        )
    positions = np.asarray(positions, dtype=np.float64)
    require_position_columns(positions, 2, call="compute_boundary_distances()")
    if direction_frame == "allocentric" and headings is not None:
        raise ValueError(
            _format_error(
                "direction_frame='allocentric' does not use headings.",
                why="Why: allocentric directions are fixed to the room (0 = "
                "East); a heading would turn them with the animal.",
                fix="drop headings for boundary-vector directions, or pass "
                "direction_frame='egocentric' to measure them from the heading",
            )
        )
    if direction_frame == "egocentric":
        if headings is None:
            raise ValueError(
                _format_error(
                    "direction_frame='egocentric' needs headings.",
                    why="Why: egocentric directions are measured from where "
                    "the animal faces (0 = ahead, +pi/2 = left).",
                    fix="pass headings in radians (0 = East), for example "
                    "from neurospatial.behavior.heading_from_velocity",
                )
            )
        headings = np.asarray(headings, dtype=np.float64).ravel()
        if headings.shape != (len(positions),):
            raise ValueError(
                f"headings must have one value per position: got "
                f"{headings.shape[0]} headings for {len(positions)} positions."
            )
    return positions, headings
