"""Goals of the behavior functions: a region name or a point, one per trial.

Every function that measures behavior toward a goal reads ``goal`` the same
way: a region name, a point in environment coordinates, or, with intervals
such as trials, one of those per interval (``None`` for an unknown goal).
A region is its area: distances and directions run to its edge, and the
distance is 0 inside it. A point is its exact coordinates. Each interval is measured from its own
samples, ``[start, stop)``, as if it were the whole recording.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._interval_table import _missing_as_none

if TYPE_CHECKING:
    import pandas as pd

    from neurospatial._interval_table import Intervals
    from neurospatial._observation import ObservationSupport
    from neurospatial.environment import Environment
    from neurospatial.ops import _EnvironmentPaths


@dataclass(frozen=True, eq=False)
class Goal:
    """One resolved goal: a region (its area and bins) or a point's coordinates.

    A region goal is reached at its edge: the distance to it is the distance
    to the nearest point of the region, 0 inside it, and its direction points
    along the shortest path to that point (see :meth:`distances`).

    Attributes
    ----------
    name : str or None
        Region name; None for a point given by coordinates.
    bins : ndarray of intp, shape (n_goal_bins,), or None
        The region's bins (polygon regions); None for a point. The
        ``cost_map`` distance of :func:`distance_to_goal` grows from them.
    point : ndarray, shape (n_dims,), or None
        The point (coordinates, or a point region's location); None for a
        polygon region.
    area : shapely.Polygon or None
        The region's polygon; None for a point.
    """

    name: str | None
    bins: NDArray[np.intp] | None
    point: NDArray[np.float64] | None
    area: Any = None

    @property
    def label(self) -> str | NDArray[np.float64]:
        """The goal as given: the region name, else the point."""
        if self.name is not None:
            return self.name
        assert self.point is not None
        return self.point

    def distances(
        self,
        env: Environment,
        paths: _EnvironmentPaths | None,
        positions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Distance from each position to the goal, shape (n_samples,).

        To the point itself for a point goal. For a region, to its edge: 0
        inside the region, else the distance to the region's nearest point.
        Straight (``paths`` None): the distance to the polygon. Geodesic: the
        shortest path inside the environment to a point of the region's
        edge. On a track that edge is where the centerline crosses the
        region's boundary, and positions are measured from their projection
        onto the centerline, as every track distance is. In an open
        environment it is the straight line to the polygon's nearest point
        when that line stays inside the environment (the geodesic cannot be
        shorter), else the path through the region's field
        (:meth:`_EnvironmentPaths.to_area`): round walls through bin centres,
        then straight from the last of them to the edge. NaN at a missing
        position or one outside the environment; ``inf`` (geodesic) where the
        goal cannot be reached.
        """
        if self.area is not None:
            assert self.bins is not None
            return _region_paths(self.area, self.bins, env, paths, positions)[0]
        assert self.point is not None
        if paths is not None:
            return paths.to_point(positions, self.point)
        straight: NDArray[np.float64] = np.linalg.norm(positions - self.point, axis=1)
        straight[env.bin_at(positions) < 0] = np.nan
        return straight

    def directions(
        self,
        env: Environment,
        paths: _EnvironmentPaths | None,
        positions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Unit vector toward the goal at each position, shape (n_samples, n_dims).

        The direction in which the shortest path to the goal (geodesic, or
        the straight line), as :meth:`distances` measures it, leaves the
        position. NaN at the goal (inside a region, or exactly at a point),
        at a missing position or one outside the environment, and where the
        goal cannot be reached.
        """
        if self.area is not None:
            assert self.bins is not None
            _, heads, along_track = _region_paths(
                self.area, self.bins, env, paths, positions
            )
            unit = _unit(heads - positions)
            if paths is not None and along_track.any():
                # Along a track, the direction the track runs toward the edge.
                bent = np.flatnonzero(along_track)
                targets, group = np.unique(heads[bent], axis=0, return_inverse=True)
                for k, target in enumerate(targets):
                    rows = bent[group.reshape(-1) == k]
                    unit[rows] = paths.directions(positions[rows], target)
            return unit
        assert self.point is not None
        if paths is not None:
            return paths.directions(positions, self.point)
        vector = np.full(positions.shape, np.nan)
        inside = env.bin_at(positions) >= 0
        vector[inside] = self.point - positions[inside]
        return _unit(vector)

    def contains(
        self, env: Environment, positions: NDArray[np.float64]
    ) -> NDArray[np.bool_]:
        """Whether each position lies in the goal region; all False for a point.

        Inside means inside the region's polygon or on its edge (on a track,
        the position's projection onto the centerline), where the distance
        to the region is 0.
        """
        if self.area is None:
            return np.zeros(len(positions), dtype=bool)
        located = env.bin_at(positions) >= 0
        return located & _covers(self.area, _track_projection(env, positions))


def _unit(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rows scaled to unit length; NaN for zero or missing rows."""
    norm = np.linalg.norm(vector, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit: NDArray[np.float64] = np.where(
            (norm > 0)[:, None], vector / norm[:, None], np.nan
        )
    return unit


def _covers(area: Any, points: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Whether the polygon covers each point (inside or on its edge); False for NaN."""
    import shapely

    finite = np.isfinite(points).all(axis=1)
    out = np.zeros(len(points), dtype=bool)
    out[finite] = shapely.covers(area, shapely.points(points[finite]))
    return out


def _track_projection(
    env: Environment, positions: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Positions on a track's centerline (their projection); unchanged elsewhere."""
    if not env.is_linearized_track:
        return positions
    out = np.full(positions.shape, np.nan)
    finite = np.isfinite(positions).all(axis=1)
    if finite.any():
        out[finite] = env.linear_to_nd(env.to_linear(positions[finite]))
    return out


def _track_edge_points(
    area: Any, env: Environment, paths: _EnvironmentPaths
) -> NDArray[np.float64]:
    """Where a track's centerline crosses a region's boundary, shape (k, 2)."""
    import shapely

    linearization = env.binning.linearization
    assert linearization is not None
    graph = linearization.graph
    segments = shapely.linestrings(
        [[graph.nodes[u]["pos"], graph.nodes[v]["pos"]] for u, v in graph.edges]
    )
    points = shapely.get_coordinates(shapely.intersection(area.boundary, segments))
    points = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    return points[paths.bins(points) >= 0]


def _signed_edge_distance(
    area: Any, points: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Straight distance to a polygon's edge: positive outside, negative inside."""
    import shapely

    shapes = shapely.points(points)
    signed = shapely.distance(area, shapes)
    inside = shapely.covers(area, shapes)
    signed[inside] = -shapely.distance(area.boundary, shapes[inside])
    return np.asarray(signed, dtype=np.float64)


def _region_paths(
    area: Any,
    bins: NDArray[np.intp],
    env: Environment,
    paths: _EnvironmentPaths | None,
    positions: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Distance from each position to a region's edge, and where its path heads.

    Returns
    -------
    distances : ndarray, shape (n_samples,)
        As :meth:`Goal.distances` describes; 0 inside the region.
    heads : ndarray, shape (n_samples, n_dims)
        Where each shortest path goes first: the edge point it ends at for a
        straight path, its first corner otherwise, or (on a track) the edge
        point it ends at; NaN inside the region, outside the environment or
        where it cannot be reached.
    along_track : ndarray of bool, shape (n_samples,)
        Paths along a track to ``heads``, whose direction the track gives.
    """
    import shapely

    n = len(positions)
    distances = np.full(n, np.nan)
    heads = np.full(positions.shape, np.nan)
    along_track = np.zeros(n, dtype=bool)
    located = env.bin_at(positions) >= 0
    inside = located & _covers(area, _track_projection(env, positions))
    distances[inside] = 0.0
    rows = np.flatnonzero(located & ~inside)
    if rows.size == 0:
        return distances, heads, along_track
    points = positions[rows]
    if paths is not None and env.is_linearized_track:
        # Along the centerline to where it crosses the region's boundary.
        best = np.full(len(points), np.inf)
        best_end = np.full(points.shape, np.nan)
        for target in _track_edge_points(area, env, paths):
            length = paths.to_point(points, target)
            better = length < best
            best[better] = length[better]
            best_end[better] = target
        distances[rows], heads[rows] = best, best_end
        along_track[rows] = np.isfinite(best)
        return distances, heads, along_track
    # The polygon's nearest point, by the straight line.
    nearest = shapely.get_coordinates(
        shapely.shortest_line(shapely.points(points), area)
    ).reshape(-1, 2, 2)[:, 1]
    euclidean = np.linalg.norm(nearest - points, axis=1)
    distances[rows], heads[rows] = euclidean, nearest
    if paths is None:
        return distances, heads, along_track
    # No path is shorter than the straight line, so a visible nearest point
    # is the answer; the others go round through the field of the region.
    hidden = np.flatnonzero(~paths.sees(points, nearest))
    if hidden.size == 0:
        return distances, heads, along_track
    centers = np.asarray(env.bin_centers, dtype=np.float64)

    def entry(
        starts: NDArray[np.float64], corners: NDArray[np.intp]
    ) -> NDArray[np.float64]:
        lines = shapely.linestrings(np.stack([starts, centers[corners]], axis=1))
        inside_part = shapely.intersection(lines, area)
        return np.asarray(
            shapely.distance(shapely.points(starts), inside_part), dtype=np.float64
        )

    total, corner = paths.to_area(
        bins,
        _signed_edge_distance(area, centers),
        entry,
        points[hidden],
        paths.bins(points[hidden]),
    )
    found = corner >= 0
    target = rows[hidden]
    distances[target] = np.where(found, total, np.inf)
    heads[target] = np.nan
    heads[target[found]] = centers[corner[found]]
    return distances, heads, along_track


def resolve_goal(
    env: Environment,
    target: Any,
    *,
    paths: _EnvironmentPaths | None,
    call: str,
    name: str = "goal",
) -> Goal:
    """One goal from a region name or a point.

    Parameters
    ----------
    env : Environment
    target : str or array-like of shape (n_dims,)
        A region name in ``env.regions`` (a polygon region is its area and
        bins, a point region its location), or a point in environment coordinates.
    paths : _EnvironmentPaths or None
        The geodesic engine, or None for straight lines; a geodesic goal
        point must lie inside the environment.
    call : str
        Public function name, used in messages.
    name : str
        Argument name, used in messages.

    Raises
    ------
    RegionNotFoundError
        If a region name is not in ``env.regions``.
    ValueError
        If a region covers no bin, a point has the wrong number of
        coordinates or a non-finite one, or a geodesic point lies outside
        the environment.
    """
    if isinstance(target, str):
        from neurospatial.ops.mapping import _require_region_mask

        _require_region_mask(env, target, operation=f"{call}()", argument=name)
        region = env.regions[target]
        if region.kind == "point":
            point = np.asarray(region.data, dtype=np.float64).reshape(-1)
            return Goal(target, None, _checked_point(env, point, paths, name, call))
        bins = np.unique(np.asarray(env.bins_in_region(target), dtype=np.intp))
        if bins.size == 0:
            raise ValueError(
                _format_error(
                    f"{call}(): {name}= region {target!r} covers no bin.",
                    fix="enlarge the region, or use smaller bins.",
                )
            )
        return Goal(target, bins, None, region.data)
    point = np.asarray(target, dtype=np.float64)
    if point.shape != (env.n_dims,):
        raise ValueError(
            _format_error(
                f"{call}(): {name}= has shape {point.shape}; a point needs one "
                f"coordinate per environment dimension ({env.n_dims}).",
                why="the goal is a region name or a point in the environment's "
                "coordinates.",
                fix=f"pass a region name from env.regions, or a point of shape "
                f"({env.n_dims},) in the units of the positions.",
            )
        )
    return Goal(None, None, _checked_point(env, point, paths, name, call))


def _checked_point(
    env: Environment,
    point: NDArray[np.float64],
    paths: _EnvironmentPaths | None,
    name: str,
    call: str,
) -> NDArray[np.float64]:
    if not np.isfinite(point).all():
        raise ValueError(
            _format_error(
                f"{call}(): {name}= has a non-finite coordinate: {point.tolist()}.",
                why="no distance or direction to a missing goal can be measured.",
                fix=f"pass the {name}'s finite coordinates, or a region name.",
            )
        )
    if paths is not None and env.bin_at(point[None, :])[0] < 0:
        raise ValueError(
            _format_error(
                f"{call}(): {name}= point {point.tolist()} is outside the environment.",
                why="a geodesic distance is a path inside the environment, so "
                "both of its ends must lie in a bin.",
                fix=f"pass a {name} inside the environment (check env.contains), "
                "a region name, or metric='euclidean'.",
            )
        )
    return point


def as_point(target: object) -> NDArray[np.float64] | None:
    """``target`` as one point (real numbers, 0-D or 1-D), or None if it is not one."""
    if isinstance(target, str) or target is None:
        return None
    try:
        raw = np.asarray(target)
    except ValueError:  # ragged: points mixed with None, one per interval
        return None
    if raw.dtype.kind not in "iuf" or raw.ndim > 1:
        return None  # names, None entries or one point per interval
    return raw.astype(np.float64).reshape(-1)


def interval_targets(
    intervals: Intervals[Any] | None,
    target: Any,
    *,
    name: str,
    call: str,
    n_dims: int,
) -> tuple[list[Any], bool]:
    """The target of each interval, and whether one target was given for all.

    Without ``intervals`` ``target`` is one target (a list of one). With
    them it is one target for every row or one per row: a sequence of
    region names, points or ``None`` in row order, or a ``pandas.Series``
    indexed by ``interval_id``. In an environment of more than one
    dimension a number per interval (a well number) is refused: a field (a
    Series) of numbers, or integers with one per interval, cannot be a point
    there.
    """
    # A Series is one value per interval_id, whatever its values are.
    by_id = hasattr(target, "reindex") and hasattr(target, "index")
    point = None if by_id else as_point(target)
    single = isinstance(target, str) or point is not None
    one = target if point is None else point
    if intervals is None:
        if not single:
            raise ValueError(
                _format_error(
                    f"{call}: {name}= has one value per interval, but no "
                    "intervals were given.",
                    why="per-interval targets need the intervals they belong to.",
                    fix=f"pass intervals=trials, or one {name} for every sample "
                    "(a region name or a point).",
                )
            )
        return [one], True
    if (
        point is not None
        and n_dims > 1
        and np.asarray(target).dtype.kind in "iu"
        and point.size == len(intervals)
    ):
        raise TypeError(
            _format_error(
                f"{call}: {name}={list(np.asarray(target).tolist())} has one "
                f"integer per interval ({len(intervals)}), like well numbers.",
                why="well numbers are not region names, and integers read as one "
                "point would silently move the target.",
                fix=f"map the numbers to the environment's region names, e.g. "
                f"{name}=trials.goal_well.map({{1: 'well_1', 2: 'well_2'}}); for "
                "one point, give float coordinates.",
            )
        )
    if single:
        return [one] * len(intervals), True
    values = list(intervals._per_row(target, name=name)) if by_id else list(target)
    # A missing entry in a pandas column (NaN, pd.NA; pandas 3 stores a
    # missing string as NaN) is an unknown target, like None.
    values = [_missing_as_none(value) for value in values]
    if len(values) != len(intervals):
        msg = (
            f"{call}: {name}= has {len(values)} entries; give one per interval "
            f"({len(intervals)}), in row order, or one {name} for all."
        )
        raise ValueError(msg)
    targets: list[Any] = []
    for interval_id, value in zip(intervals.interval_ids, values, strict=True):
        if n_dims > 1 and (
            isinstance(value, numbers.Number)
            or (isinstance(value, np.ndarray) and value.ndim == 0)
        ):
            raise TypeError(
                _format_error(
                    f"{call}: {name}= for interval_id {interval_id} is the number "
                    f"{value}, not a region name or a point.",
                    why="a number per interval, such as a well number, is not a "
                    "region name, and a point is a sequence of coordinates.",
                    fix=f"map the numbers to the environment's region names, e.g. "
                    f"{name}=trials.goal_well.map({{1: 'well_1', 2: 'well_2'}}).",
                )
            )
        entry = as_point(value)
        if entry is None and value is not None and not isinstance(value, str):
            msg = (
                f"{call}: {name}= entries must be region names, points or None; "
                f"got {value!r}."
            )
            raise TypeError(msg)
        targets.append(value if entry is None else entry)
    return targets, False


@dataclass(frozen=True)
class GoalRow:
    """The samples of one interval and its goal.

    Attributes
    ----------
    interval_id : int or None
        None without intervals (the whole recording).
    samples : slice
        The interval's samples, ``[start, stop)``.
    goal : Goal or None
        None for an unknown goal.
    """

    interval_id: int | None
    samples: slice
    goal: Goal | None


def goal_rows(
    env: Environment,
    times: NDArray[np.float64],
    intervals: Intervals[Any] | None,
    goal: Any,
    *,
    paths: _EnvironmentPaths | None,
    call: str,
    disjoint: bool,
) -> list[GoalRow]:
    """Each interval's samples and resolved goal, in row order.

    Without ``intervals``, one row: every sample and the one goal. With
    them, the samples in ``[start, stop)`` of each row; ``disjoint`` refuses
    overlapping rows (a per-sample result has one value per sample).
    """
    targets, _ = interval_targets(
        intervals, goal, name="goal", call=f"{call}()", n_dims=env.n_dims
    )
    resolved: dict[tuple[Any, ...], Goal] = {}

    def lookup(target: Any) -> Goal | None:
        if target is None:
            return None
        key = (
            ("region", target)
            if isinstance(target, str)
            else ("point", tuple(np.asarray(target, dtype=float).tolist()))
        )
        if key not in resolved:
            resolved[key] = resolve_goal(env, target, paths=paths, call=call)
        return resolved[key]

    if intervals is None:
        return [GoalRow(None, slice(0, times.size), lookup(targets[0]))]
    if disjoint:
        intervals._refuse_overlap(f"{call}()")
    first = np.searchsorted(times, intervals.start, side="left")
    last = np.searchsorted(times, intervals.stop, side="left")
    return [
        GoalRow(int(interval_id), slice(int(a), int(b)), lookup(target))
        for interval_id, a, b, target in zip(
            intervals.interval_ids, first, last, targets, strict=True
        )
    ]


def row_support(
    times: NDArray[np.float64], support: ObservationSupport, samples: slice
) -> ObservationSupport:
    """The support of one interval's samples, measured as a recording of its own.

    Gaps, ``epochs`` and declared breaks are gated as for the whole
    recording; nothing is bridged across the interval's edges.
    """
    from neurospatial._observation import resolve_sample_support

    breaks = support.break_before
    return resolve_sample_support(
        times[samples],
        support.sample_valid[samples],
        max_gap=support.max_gap,
        epochs=support.epochs,
        break_before=None if breaks is None else breaks[samples],
        result=None,
    )


def by_interval(rows: list[GoalRow], values: list[Any], *, name: str) -> pd.Series:
    """One value per interval, indexed by ``interval_id``."""
    import pandas as pd

    return pd.Series(
        values,
        index=pd.Index([row.interval_id for row in rows], name="interval_id"),
        name=name,
    )
