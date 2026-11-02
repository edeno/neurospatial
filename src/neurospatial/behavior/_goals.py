"""Goals of the behavior functions: a region name or a point, one per trial.

Every function that measures behavior toward a goal reads ``goal`` the same
way: a region name, a point in environment coordinates, or, with intervals
such as trials, one of those per interval (``None`` for an unknown goal).
A region is its bins; distances and directions run to the nearest of them.
A point is its exact coordinates. Each interval is measured from its own
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
    from neurospatial.behavior._environment_paths import EnvironmentPaths
    from neurospatial.environment import Environment


@dataclass(frozen=True, eq=False)
class Goal:
    """One resolved goal: a region's bins or a point's coordinates.

    Attributes
    ----------
    name : str or None
        Region name; None for a point given by coordinates.
    bins : ndarray of intp, shape (n_goal_bins,), or None
        The region's bins (polygon regions); None for a point.
    point : ndarray, shape (n_dims,), or None
        The point (coordinates, or a point region's location); None for a
        polygon region.
    """

    name: str | None
    bins: NDArray[np.intp] | None
    point: NDArray[np.float64] | None

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
        paths: EnvironmentPaths | None,
        positions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Distance from each position to the goal, shape (n_samples,).

        To the nearest goal bin centre for a region, to the point itself
        otherwise; geodesic (``paths``) or straight. NaN at a missing
        position or one outside the environment; ``inf`` (geodesic) where
        the goal cannot be reached.
        """
        if self.bins is not None:
            if paths is not None:
                return paths.from_bins_at(self.bins, positions)
            from scipy.spatial import cKDTree

            out = np.full(len(positions), np.nan)
            inside = env.bin_at(positions) >= 0
            out[inside] = cKDTree(env.bin_centers[self.bins]).query(positions[inside])[
                0
            ]
            return out
        assert self.point is not None
        if paths is not None:
            return paths.to_point(positions, self.point)
        out = np.linalg.norm(positions - self.point, axis=1)
        out[env.bin_at(positions) < 0] = np.nan
        return out

    def directions(
        self,
        env: Environment,
        paths: EnvironmentPaths | None,
        positions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Unit vector toward the goal at each position, shape (n_samples, n_dims).

        The direction in which the shortest path to the goal (geodesic, or
        the straight line) leaves the position. NaN at the goal (inside a
        goal bin, or exactly at a point), at a missing position or one
        outside the environment, and where the goal cannot be reached.
        """
        if self.bins is not None and paths is not None:
            return paths.directions_to_bins(self.bins, positions)
        if self.bins is None and paths is not None:
            assert self.point is not None
            return paths.directions(positions, self.point)
        bins = env.bin_at(positions)
        vector = np.full(positions.shape, np.nan)
        inside = bins >= 0
        if self.bins is not None:
            from scipy.spatial import cKDTree

            centers = env.bin_centers[self.bins]
            nearest = cKDTree(centers).query(positions[inside])[1]
            vector[inside] = centers[nearest] - positions[inside]
            vector[np.isin(bins, self.bins)] = np.nan
        else:
            assert self.point is not None
            vector[inside] = self.point - positions[inside]
        norm = np.linalg.norm(vector, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            unit: NDArray[np.float64] = np.where(
                (norm > 0)[:, None], vector / norm[:, None], np.nan
            )
        return unit

    def contains(
        self, env: Environment, positions: NDArray[np.float64]
    ) -> NDArray[np.bool_]:
        """Whether each position lies in a goal bin; all False for a point."""
        if self.bins is None:
            return np.zeros(len(positions), dtype=bool)
        return np.isin(env.bin_at(positions), self.bins)


def resolve_goal(
    env: Environment,
    target: Any,
    *,
    paths: EnvironmentPaths | None,
    call: str,
    name: str = "goal",
) -> Goal:
    """One goal from a region name or a point.

    Parameters
    ----------
    env : Environment
    target : str or array-like of shape (n_dims,)
        A region name in ``env.regions`` (a polygon region is its bins, a
        point region its location), or a point in environment coordinates.
    paths : EnvironmentPaths or None
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
        return Goal(target, bins, None)
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
    paths: EnvironmentPaths | None,
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
    paths: EnvironmentPaths | None,
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
