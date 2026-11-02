"""Navigation and path efficiency metrics for spatial behavior analysis.

This module provides comprehensive tools for analyzing goal-directed navigation
and path efficiency in spatial environments. It combines:

- **Path progress and distance metrics**: Compute progress toward goals, distances
  (or travel costs) to goals, boundaries and rewards
- **Path efficiency**: Compare actual vs. optimal paths (geodesic or Euclidean)
- **Goal-directed metrics**: Measure how directly animals navigate toward goals
- **Direction labeling**: Generate per-timepoint direction labels for directional
  place field analysis

Imports
-------
All navigation functions are importable from ``behavior.navigation``::

    from neurospatial.behavior.navigation import (
        # Path progress and distance
        path_progress,
        distance_to_goal,
        distance_to_boundary,
        distance_to_reward,
        time_to_goal,
        graph_turn_sequence,
        heading_direction_labels,
        headings_to_direction_labels,
        # Path efficiency
        PathEfficiencyResult,
        SubgoalEfficiencyResult,
        traveled_path_length,
        shortest_path_length,
        time_efficiency,
        angular_efficiency,
        subgoal_efficiency,
        compute_path_efficiency,
        # Goal-directed
        GoalDirectedMetrics,
        goal_vector,
        goal_direction,
        instantaneous_goal_alignment,
        goal_bias,
        approach_rate,
        compute_goal_directed_metrics,
    )

Or from the behavior module::

    from neurospatial.behavior import path_progress, compute_path_efficiency, goal_bias

Typical Workflows
-----------------
**Multi-trial analysis** (T-maze, Y-maze, spatial bandit):
    1. Find the trials between named regions with ``detect_trials()``,
       or read them from a task log or NWB
    2. Compute per-sample metrics with the trials as ``intervals``:
       ``path_progress()``, ``distance_to_goal()``, ``time_to_goal()``

**Path efficiency analysis**:
    1. Compute efficiency: ``result = compute_path_efficiency(env, times, positions, goal="reward")``
    2. Check efficiency: ``if result.is_efficient(threshold=0.8): ...``
    3. Print summary: ``print(result)``
    4. One row per trial: ``compute_path_efficiency(env, times, positions, trials, goal=trials.end_region)``

**Goal-directed analysis**:
    1. Compute metrics: ``result = compute_goal_directed_metrics(env, times, positions, goal="reward")``
    2. Check goal-directed: ``if result.is_goal_directed(): ...``

Example
-------
Complete analysis pipeline for a spatial navigation task::

    from neurospatial.behavior import (
        detect_trials,
        path_progress,
        compute_path_efficiency,
        compute_goal_directed_metrics,
    )

    # 1. Trials between the home box and the goal wells
    trials = detect_trials(
        env, times, positions, regions=["home", "goal_left", "goal_right"]
    )
    reached = trials[trials.outcome == "reached"]

    # 2. Per-sample progress toward each trial's destination
    progress = path_progress(
        env,
        times,
        positions,
        reached,
        start=reached.start_region,
        goal=reached.end_region,
    )

    # 3. Efficiency of each trial toward its own destination
    efficiency = compute_path_efficiency(
        env, times, positions, reached, goal=reached.end_region
    )
    print(efficiency[["efficiency", "angular_efficiency"]])

    # 4. Goal-directed metrics of each trial
    goal_directed = compute_goal_directed_metrics(
        env, times, positions, reached, goal=reached.end_region
    )
    print(goal_directed["goal_bias"].mean())

References
----------
.. [1] Batschelet, E. (1981). Circular Statistics in Biology. Academic Press.
.. [2] Johnson, A., & Redish, A. D. (2007). Neural ensembles in CA3 transiently
       encode paths forward of the animal at a decision point. J Neurosci.
       DOI: 10.1523/JNEUROSCI.3761-07.2007
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import (
    DEFAULT_MAX_GAP,
    break_sample_times,
    recording_index,
    times_in_intervals,
)
from neurospatial._pynapple import reads_pynapple_samples
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin
from neurospatial._stacklevel import _external_stacklevel
from neurospatial._validation import validate_finite_scalar
from neurospatial.behavior._metric import metric_paths, require_metric
from neurospatial.geometry import require_cartesian

if TYPE_CHECKING:
    import pandas as pd

    from neurospatial._interval_table import Intervals
    from neurospatial._observation import ObservationSupport
    from neurospatial.behavior._environment_paths import EnvironmentPaths
    from neurospatial.behavior._goals import Goal, GoalRow
    from neurospatial.environment import Environment
    from neurospatial.geometry import CartesianSystem
    from neurospatial.ops._types import DistanceMetric


__all__ = [  # noqa: RUF022
    # Path progress and distance
    "distance_to_boundary",
    "distance_to_goal",
    "distance_to_reward",
    "graph_turn_sequence",
    "heading_direction_labels",
    "headings_to_direction_labels",
    "path_progress",
    "time_to_goal",
    # Path efficiency
    "PathEfficiencyResult",
    "SubgoalEfficiencyResult",
    "angular_efficiency",
    "compute_path_efficiency",
    "shortest_path_length",
    "subgoal_efficiency",
    "time_efficiency",
    "traveled_path_length",
    # Goal-directed
    "GoalDirectedMetrics",
    "approach_rate",
    "compute_goal_directed_metrics",
    "goal_bias",
    "goal_direction",
    "goal_vector",
    "instantaneous_goal_alignment",
]


# =============================================================================
# Path Efficiency Dataclasses
# =============================================================================


def _result_error(result: str, what: str, *, why: str, fix: str) -> ValueError:
    """A ValueError for an invalid result field, naming the result class."""
    return ValueError(_format_error(f"{result}: {what}", why=why, fix=fix))


def _result_goal(
    result: str, goal: object, *, n_dims: int | None
) -> str | NDArray[np.float64]:
    """A result's goal: a region name, or a point of shape ``(n_dims,)``."""
    if isinstance(goal, str):
        return goal
    point = np.asarray(goal, dtype=float)
    if point.ndim != 1 or (n_dims is not None and point.shape != (n_dims,)):
        raise _result_error(
            result,
            f"goal has shape {point.shape}.",
            why="the goal is a region name or one position.",
            fix="pass a region name, or an array of shape (n_dims,).",
        )
    return point


def _require_length(result: str, name: str, value: float) -> float:
    """Return ``value`` as a float; raise if it is negative (NaN and inf pass)."""
    value = float(value)
    if value < 0:
        raise _result_error(
            result,
            f"{name}={value!r} is negative.",
            why="a path length, distance or duration is never negative.",
            fix=f"pass {name} >= 0, or NaN when it was not measured.",
        )
    return value


@dataclass(frozen=True, repr=False, eq=False)
class PathEfficiencyResult(ReadOnlyArraysMixin, ResultMixin):
    """Path efficiency metrics for a trajectory segment.

    Attributes
    ----------
    traveled_length : float
        Distance traveled over the observed steps, in environment units. NaN
        when no step was observed.
    shortest_length : float
        Length of the shortest path from the start to the goal, in
        environment units. Is ``inf`` when the goal is unreachable under the
        geodesic metric (the goal lies on a disconnected component); it is
        not clamped to ``0.0``.
    efficiency : float
        Derived: ``shortest_length / traveled_length`` when the path is
        ``fully_observed``. Is ``(0, 1]`` when the path ends at the goal, and
        can exceed 1 when it does not. Is 0.0 when a geodesic step is
        infinite (it crosses between disconnected parts of the environment).
        NaN when the path is not fully observed, ``traveled_length`` is 0 or
        NaN, or ``shortest_length`` is not finite.
    time_efficiency : float or None
        Ratio T_optimal / T_actual if reference_speed provided, else None.
    angular_efficiency : float
        ``1 - mean(angle) / pi``, where ``angle`` is the angle between each
        observed step and the direction from the step's start to the goal
        (see :func:`angular_efficiency`). Range [0, 1]: 1.0 for a run
        straight toward the goal, 0.5 for steps perpendicular to it, 0.0 for
        a run straight away. NaN when no step has a defined direction (no
        movement, or fewer than two observed samples).
    start_position : NDArray[np.float64]
        Start position coordinates, shape (n_dims,).
    goal : str or NDArray[np.float64]
        The goal as given: a region name, or a point of shape (n_dims,).
    metric : {"euclidean", "geodesic"}
        Distance metric used.
    fully_observed : bool
        True when every step from the first to the last observed sample is
        usable: missing samples are bridged within ``max_gap`` and there is
        no longer gap, epoch edge or declared break. Only then does
        ``traveled_length`` cover the whole route, so only then is
        ``efficiency`` defined.
    """

    traveled_length: float
    shortest_length: float
    efficiency: float = field(init=False)
    time_efficiency: float | None
    angular_efficiency: float
    start_position: NDArray[np.float64]
    goal: str | NDArray[np.float64]
    metric: Literal["euclidean", "geodesic"]
    fully_observed: bool

    def __post_init__(self) -> None:
        """Validate the fields, derive the efficiency and own the positions."""
        name = "PathEfficiencyResult"
        require_metric(self.metric, where=name)
        traveled = _require_length(name, "traveled_length", self.traveled_length)
        shortest = _require_length(name, "shortest_length", self.shortest_length)
        start = np.asarray(self.start_position, dtype=float)
        goal = _result_goal(name, self.goal, n_dims=start.shape[-1])
        if start.ndim != 1:
            raise _result_error(
                name,
                f"start_position has shape {start.shape}.",
                why="the start is one position.",
                fix="pass an array of shape (n_dims,).",
            )
        object.__setattr__(self, "traveled_length", traveled)
        object.__setattr__(self, "shortest_length", shortest)
        object.__setattr__(self, "angular_efficiency", float(self.angular_efficiency))
        if self.time_efficiency is not None:
            object.__setattr__(self, "time_efficiency", float(self.time_efficiency))
        object.__setattr__(self, "fully_observed", bool(self.fully_observed))
        object.__setattr__(self, "start_position", start)
        object.__setattr__(self, "goal", goal)
        defined = self.fully_observed and traveled > 0 and np.isfinite(shortest)
        object.__setattr__(
            self, "efficiency", shortest / traveled if defined else float("nan")
        )
        self._own_arrays()

    def is_efficient(self, threshold: float = 0.8) -> bool:
        """Return True if the path efficiency is at least ``threshold``.

        Parameters
        ----------
        threshold : float, default=0.8
            Efficiency threshold (0 to 1). Default 0.8 means 80% efficient.

        Returns
        -------
        bool
            True if ``efficiency >= threshold``; False when the efficiency
            is NaN (not measured).
        """
        return bool(self.efficiency >= threshold)

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar path-efficiency metrics.

        For a human-readable string use ``str(result)``.
        """
        return {
            "traveled_length": self.traveled_length,
            "shortest_length": self.shortest_length,
            "efficiency": self.efficiency,
        }

    def __str__(self) -> str:
        if np.isnan(self.efficiency):
            return (
                f"Path: {self.traveled_length:.1f} traveled, "
                f"{self.shortest_length:.1f} optimal (efficiency: N/A)"
            )
        return (
            f"Path: {self.traveled_length:.1f} traveled, "
            f"{self.shortest_length:.1f} optimal "
            f"(efficiency: {self.efficiency:.1%})"
        )


@dataclass(frozen=True, repr=False, eq=False)
class SubgoalEfficiencyResult(ReadOnlyArraysMixin, ResultMixin):
    """Path efficiency with subgoal decomposition.

    Use subgoal decomposition when:
    - Animal must navigate through intermediate waypoints (e.g., T-maze stem)
    - Comparing strategies across complex paths (room-to-room navigation)
    - Detecting hierarchical planning behavior

    Example: Home -> Center -> Goal
    Instead of measuring direct Home->Goal efficiency, measure:
    - Segment 1: Home -> Center efficiency
    - Segment 2: Center -> Goal efficiency

    Attributes
    ----------
    segment_results : tuple of PathEfficiencyResult
        Efficiency result for each segment, one per subgoal.
    subgoal_positions : NDArray[np.float64]
        Positions of subgoals, shape (n_subgoals, n_dims).
    mean_efficiency : float
        Derived: unweighted mean of the segment efficiencies, over the
        segments whose efficiency is defined. NaN when none is.
    weighted_efficiency : float
        Derived: mean of the defined segment efficiencies weighted by each
        segment's ``shortest_length``, so longer segments count more. NaN
        when no efficiency is defined or their shortest lengths sum to 0.
    """

    segment_results: tuple[PathEfficiencyResult, ...]
    subgoal_positions: NDArray[np.float64]
    mean_efficiency: float = field(init=False)
    weighted_efficiency: float = field(init=False)

    def __post_init__(self) -> None:
        """Validate the segments, derive the means and own the subgoals."""
        segments = tuple(self.segment_results)
        subgoals = np.asarray(self.subgoal_positions, dtype=float)
        if subgoals.ndim != 2 or len(segments) != len(subgoals):
            msg = "SubgoalEfficiencyResult"
            raise _result_error(
                msg,
                f"{len(segments)} segment results for subgoal_positions of "
                f"shape {subgoals.shape}.",
                why="each subgoal ends one segment.",
                fix="pass one segment result per row of subgoal_positions.",
            )
        object.__setattr__(self, "segment_results", segments)
        object.__setattr__(self, "subgoal_positions", subgoals)
        efficiency = np.array([seg.efficiency for seg in segments], dtype=float)
        weights = np.array([seg.shortest_length for seg in segments], dtype=float)
        defined = np.isfinite(efficiency)
        mean = float(np.mean(efficiency[defined])) if defined.any() else np.nan
        total = float(np.sum(weights[defined]))
        weighted = (
            float(np.sum(efficiency[defined] * weights[defined]) / total)
            if total > 0
            else np.nan
        )
        object.__setattr__(self, "mean_efficiency", mean)
        object.__setattr__(self, "weighted_efficiency", weighted)
        self._own_arrays()

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar subgoal-efficiency metrics.

        For a human-readable per-segment string use ``str(result)``.
        """
        return {
            "n_segments": len(self.segment_results),
            "mean_efficiency": self.mean_efficiency,
            "weighted_efficiency": self.weighted_efficiency,
        }

    def __str__(self) -> str:
        lines = [f"Subgoal path efficiency ({len(self.segment_results)} segments):"]
        for i, seg in enumerate(self.segment_results):
            lines.append(f"  Segment {i + 1}: {seg.efficiency:.1%}")
        lines.append(f"  Mean: {self.mean_efficiency:.1%}")
        lines.append(f"  Weighted: {self.weighted_efficiency:.1%}")
        return "\n".join(lines)


# =============================================================================
# Goal-Directed Metrics Dataclass
# =============================================================================


@dataclass(frozen=True, repr=False, eq=False)
class GoalDirectedMetrics(ReadOnlyArraysMixin, ResultMixin):
    """Goal-directed navigation metrics for a trajectory.

    Attributes
    ----------
    goal_bias : float
        Mean instantaneous goal alignment, range [-1, 1].
        Interpretation:
        - > 0.5: Strong goal-directed navigation
        - 0 to 0.5: Weak goal-directed with some wandering
        - < 0: Net movement away from goal
    mean_approach_rate : float
        Mean rate of distance change in environment units per second,
        averaged over the observed time: the change of distance over the
        usable intervals divided by their total duration, so it does not
        depend on the sampling rate. Negative values indicate approaching
        the goal. Interpretation: -10 cm/s means closing 10 cm per second on
        average.
    time_to_reach_goal : float or None
        Seconds from the first measured sample to arrival: the first sample
        in a goal bin for a region goal, or within ``goal_radius`` of a point
        goal. None for a point goal without ``goal_radius`` (arrival not
        tested); NaN when the goal was never reached. Not to be confused with
        :func:`time_to_goal`, which returns a per-sample array.
    min_distance_to_goal : float
        Closest distance to the goal (under ``metric``) over the measured
        samples, in environment units. NaN when no sample is measured.
    goal_distance_at_start : float
        Distance to the goal at the first measured sample, in environment
        units. NaN when no sample is measured.
    goal_distance_at_end : float
        Distance to the goal at the last measured sample, in environment
        units. NaN when no sample is measured.
    goal : str or NDArray[np.float64]
        The goal as given: a region name, or a point of shape (n_dims,).
    metric : {"euclidean", "geodesic"}
        Distance metric used.

    Examples
    --------
    A run East toward a goal at x = 100 cm whose first frame was lost: the
    start distance is measured at the first observed sample.

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior import compute_goal_directed_metrics
    >>> track = np.column_stack([np.linspace(0, 100, 101), np.zeros(101)])
    >>> env = Environment.from_samples(track, bin_size=5.0, units="cm")
    >>> times = np.arange(21) / 10  # s
    >>> positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
    >>> positions[0] = np.nan  # a lost frame
    >>> metrics = compute_goal_directed_metrics(
    ...     env, times, positions, goal=[100.0, 0.0]
    ... )
    >>> metrics.goal_distance_at_start, metrics.goal_distance_at_end
    (95.0, 0.0)
    """

    goal_bias: float
    mean_approach_rate: float
    time_to_reach_goal: float | None
    min_distance_to_goal: float
    goal_distance_at_start: float
    goal_distance_at_end: float
    goal: str | NDArray[np.float64]
    metric: Literal["euclidean", "geodesic"]

    def __post_init__(self) -> None:
        """Validate the fields and own a read-only copy of the goal position."""
        name = "GoalDirectedMetrics"
        require_metric(self.metric, where=name)
        for length in (
            "min_distance_to_goal",
            "goal_distance_at_start",
            "goal_distance_at_end",
        ):
            value = _require_length(name, length, getattr(self, length))
            object.__setattr__(self, length, value)
        if self.time_to_reach_goal is not None:
            object.__setattr__(
                self,
                "time_to_reach_goal",
                _require_length(name, "time_to_reach_goal", self.time_to_reach_goal),
            )
        object.__setattr__(self, "goal_bias", float(self.goal_bias))
        object.__setattr__(self, "mean_approach_rate", float(self.mean_approach_rate))
        object.__setattr__(self, "goal", _result_goal(name, self.goal, n_dims=None))
        self._own_arrays()

    def is_goal_directed(self, threshold: float = 0.3) -> bool:
        """Return True if the goal bias is at least ``threshold``.

        Parameters
        ----------
        threshold : float, default=0.3
            Goal bias threshold. Default 0.3 is a moderate threshold.

        Returns
        -------
        bool
            True if ``goal_bias >= threshold``; False when the goal bias is
            NaN (the animal never moved).
        """
        return bool(self.goal_bias >= threshold)

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar goal-directed metrics.

        For a human-readable multi-line string use ``str(result)``.
        """
        return {
            "goal_bias": self.goal_bias,
            "mean_approach_rate": self.mean_approach_rate,
            "goal_distance_at_start": self.goal_distance_at_start,
            "goal_distance_at_end": self.goal_distance_at_end,
            "min_distance_to_goal": self.min_distance_to_goal,
            "time_to_reach_goal": self.time_to_reach_goal,
        }

    def __str__(self) -> str:
        lines = [
            "Goal-directed metrics:",
            f"  Goal bias: {self.goal_bias:.2f} (range [-1, 1])",
            f"  Approach rate: {self.mean_approach_rate:.1f} units/s",
            f"  Distance: {self.goal_distance_at_start:.1f} -> "
            f"{self.goal_distance_at_end:.1f} units",
            f"  Closest approach: {self.min_distance_to_goal:.1f} units",
        ]
        if self.time_to_reach_goal is None:
            lines.append("  Time to reach goal: not tested (no goal_radius)")
        elif np.isnan(self.time_to_reach_goal):
            lines.append("  Time to reach goal: not reached")
        else:
            lines.append(f"  Time to reach goal: {self.time_to_reach_goal:.2f} s")
        return "\n".join(lines)


# =============================================================================
# Trial-Based Navigation Functions
# =============================================================================


def _harmonic_progress(
    conductances: Any, start_bins: NDArray[np.intp], goal_bins: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Harmonic (committor) coordinate between start and goal bins.

    ``u`` is 0 on every start bin, 1 on every goal bin, and on every other bin
    the conductance-weighted mean of its neighbours: ``(Deg - W) u = 0``.

    Parameters
    ----------
    conductances : scipy sparse matrix, shape (n_bins, n_bins)
        Symmetric non-negative conductances ``W``
        (``env.get_diffusion_geometry().W``).
    start_bins, goal_bins : ndarray of intp
        Disjoint, non-empty sets of bins.

    Returns
    -------
    ndarray of float64, shape (n_bins,)
        In ``[0, 1]``; NaN on bins in a component (by positive conductance)
        that does not touch both a start and a goal bin.
    """
    from scipy import sparse
    from scipy.sparse import csgraph
    from scipy.sparse.linalg import spsolve

    u = np.full(conductances.shape[0], np.nan)
    _, component = csgraph.connected_components(conductances, directed=False)
    shared = np.intersect1d(component[start_bins], component[goal_bins])
    usable = np.isin(component, shared)
    u[start_bins] = 0.0
    u[goal_bins] = 1.0
    fixed = np.union1d(start_bins, goal_bins)
    free = np.setdiff1d(np.flatnonzero(usable), fixed)
    if free.size:
        degree = sparse.diags(np.asarray(conductances.sum(axis=1)).ravel())
        laplacian = (degree - conductances).tocsr()
        rhs = -(laplacian[free][:, fixed] @ u[fixed])
        u[free] = spsolve(laplacian[free][:, free].tocsc(), rhs)
    # Start and goal bins in a component without the other stay undefined.
    u[~usable] = np.nan
    # The solve stays within [0, 1] up to rounding (maximum principle).
    np.clip(u, 0.0, 1.0, out=u)
    return u


def _samples_by_row(rows: NDArray[np.intp]) -> list[tuple[int, NDArray[np.intp]]]:
    """Each row with samples, in row order, and its sample indices in time order.

    Parameters
    ----------
    rows : ndarray of intp, shape (n_samples,)
        Row of each sample, ``-1`` outside every row (:func:`_rows_and_targets`).

    Returns
    -------
    list of (int, ndarray of intp)
    """
    inside = np.flatnonzero(rows >= 0)
    if not inside.size:
        return []
    order = inside[np.argsort(rows[inside], kind="stable")]
    unique, first = np.unique(rows[order], return_index=True)
    return list(zip(unique.tolist(), np.split(order, first[1:]), strict=True))


def _rows_and_targets(
    intervals: Intervals[Any] | None,
    times: NDArray[np.float64],
    target: Any,
    *,
    name: str,
    call: str,
    n_dims: int,
    rows: NDArray[np.intp] | None = None,
) -> tuple[NDArray[np.intp], list[Any], bool]:
    """Row of each sample, the target of each row, and whether it is one target.

    Without ``intervals`` every sample is row 0 and ``target`` is one target.
    With them, each sample's row comes from the intervals (``-1`` outside
    every row; overlapping rows are refused); the targets are read by
    :func:`~neurospatial.behavior._goals.interval_targets`.
    """
    from neurospatial.behavior._goals import interval_targets

    targets, single = interval_targets(
        intervals, target, name=name, call=call, n_dims=n_dims
    )
    if intervals is None:
        return np.zeros(times.size, dtype=np.intp), targets, single
    if rows is None:
        rows = intervals._sample_positions(times, call)
    return rows, targets, single


def _target_key(target: Any) -> tuple[Any, ...]:
    """A hashable key for one target (region name or point)."""
    if isinstance(target, str):
        return ("region", target)
    return ("point", tuple(np.asarray(target, dtype=np.float64).tolist()))


def _point_target_bins(
    env: Environment, point: Any, *, name: str, call: str
) -> NDArray[np.intp]:
    """The bin containing a point target; refuse a point outside the environment."""
    coordinates = np.asarray(point, dtype=np.float64).reshape(1, -1)
    if coordinates.shape[1] != env.n_dims:
        msg = (
            f"{call}: {name}= point {coordinates[0].tolist()} has "
            f"{coordinates.shape[1]} coordinates; the environment has {env.n_dims}."
        )
        raise ValueError(msg)
    bins = env.bin_at(coordinates)
    if bins[0] < 0:
        raise ValueError(
            _format_error(
                f"{call}: {name}= point {coordinates[0].tolist()} is outside the "
                "environment.",
                fix="pass a point inside the environment, in the coordinates and "
                "units of env.bin_centers, or a region name.",
            )
        )
    return np.asarray(bins, dtype=np.intp)


def _target_bins(
    env: Environment,
    target: Any,
    *,
    name: str,
    call: str,
    point_radius: float | None,
    sized: bool,
) -> NDArray[np.intp]:
    """The bins of one start or goal target.

    A polygon region gives the bins whose centres lie in it. A point region
    or a point gives the bin containing it, or, when ``sized`` (path progress
    off a linearized track), the bins whose centres lie within
    ``point_radius`` of it.
    """
    point: NDArray[np.float64] | None
    if isinstance(target, str):
        from neurospatial.ops.mapping import _require_region_mask

        _require_region_mask(env, target, operation=call, argument=name)
        region = env.regions[target]
        point = (
            np.asarray(region.data, dtype=np.float64)
            if region.kind == "point"
            else None
        )
        if point is None or not sized:
            bins = np.asarray(env.bins_in_region(target), dtype=np.intp)
            if bins.size == 0:
                raise ValueError(
                    _format_error(
                        f"{call}: {name}= region {target!r} covers no bin.",
                        fix="enlarge the region, or use smaller bins.",
                    )
                )
            return bins
    else:
        point = np.asarray(target, dtype=np.float64)
        if not sized:
            return _point_target_bins(env, point, name=name, call=call)
    # A point start or goal off a linearized track needs a physical size.
    described = repr(target) if isinstance(target, str) else str(point.tolist())
    why = (
        "harmonic progress toward a single bin changes with the bin size and "
        "does not settle as bins shrink, so the start and goal need a physical "
        "size"
    )
    if point_radius is None:
        raise ValueError(
            _format_error(
                f"{call}: {name}={described} is a point in a "
                f"{env.n_dims}-D environment, but point_radius was not given.",
                why=why,
                fix="give the region a size when you define it, "
                "regions.add(name, point=..., radius=...), or pass point_radius= "
                "the size of the reward site or start zone (environment units).",
            )
        )
    _point_target_bins(env, point, name=name, call=call)
    distances = np.linalg.norm(env.bin_centers - point, axis=1)
    bins = np.flatnonzero(distances <= point_radius).astype(np.intp)
    if bins.size < 2:
        raise ValueError(
            _format_error(
                f"{call}: {name}={described} with point_radius={point_radius} "
                f"covers {bins.size} bin(s).",
                why=why,
                fix="increase point_radius to the size of the site, or use "
                "smaller bins.",
            )
        )
    return bins


def _warn_if_mostly_nan(
    values: NDArray[np.float64],
    analyzed: NDArray[np.bool_],
    causes: dict[str, NDArray[np.bool_]],
    *,
    call: str,
) -> None:
    """Warn once when at least half of the analyzed samples are NaN."""
    n_analyzed = int(np.count_nonzero(analyzed))
    missing = analyzed & np.isnan(values)
    n_missing = int(np.count_nonzero(missing))
    if n_analyzed == 0 or 2 * n_missing < n_analyzed:
        return
    counts = {
        cause: int(np.count_nonzero(mask & missing)) for cause, mask in causes.items()
    }
    main = max(counts, key=counts.__getitem__)
    warnings.warn(
        f"{call}: {n_missing} of {n_analyzed} analyzed samples "
        f"({n_missing / n_analyzed:.0%}) are NaN, mostly because {main} "
        f"({counts[main]} samples), so the result is probably misleading. Fix: "
        "check the start and goal, the environment and the positions' "
        "coordinates.",
        UserWarning,
        stacklevel=_external_stacklevel(),
    )


def _validated_route_samples(
    env: Environment,
    times: ArrayLike,
    positions: ArrayLike,
    intervals: Intervals[Any] | None,
    *,
    call: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], Intervals[Any] | None]:
    from neurospatial._interval_table import require_intervals
    from neurospatial._validation import validate_times_positions

    require_cartesian(
        env.coordinate_system,
        f"{call}()",
        why="It reads positions as the animal's allocentric positions, located "
        "in the environment's bins.",
        fix="use the allocentric environment the positions were recorded in",
    )
    if intervals is not None:
        intervals = require_intervals(intervals, name="intervals", owner=f"{call}()")
    t, p = validate_times_positions(times, positions, call=call)
    if t.size == 0:
        from neurospatial._observation import empty_samples_error

        raise empty_samples_error(call, ("times", "positions"))
    p = p.reshape(len(p), -1)
    if p.shape[1] != env.n_dims:
        msg = (
            f"{call}(): positions have {p.shape[1]} coordinate(s) per sample; "
            f"the environment has {env.n_dims}."
        )
        raise ValueError(msg)
    return t, p, intervals


@reads_pynapple_samples
def path_progress(
    env: Environment[CartesianSystem],
    times: ArrayLike,
    positions: ArrayLike,
    intervals: Intervals[Any] | None = None,
    *,
    start: Any,
    goal: Any,
    point_radius: float | None = None,
) -> NDArray[np.float64]:
    """Progress from start to goal at each sample, between 0 and 1.

    Progress at a place is the probability that an animal moving at random
    from there reaches the goal before the start: 0 on the start, 1 on the
    goal. Equivalently, each bin's progress is the average of its
    neighbours' progress, weighted by how strongly they connect (the
    conductances ``env.smooth`` uses: the shared face area over the distance
    between bin centres). On a straight track from start to goal it is the
    fraction of the track covered: 0, 0.25, 0.5, 0.75, 1 at the start, a
    quarter, half and three quarters of the way, and the goal. This is the
    harmonic, or committor, coordinate of start and goal. It works in
    1-D, 2-D and 3-D, uses the environment's own connectivity and walls, and
    has no threshold to choose.

    Parameters
    ----------
    env : Environment
        The environment the positions were recorded in. A linearized track
        (``Environment.from_graph`` or ``linear_track`` with ``track_width``)
        gives the route fraction on corridor mazes; see Notes.
    times : array-like, shape (n_samples,)
        Sample times in seconds, finite and strictly increasing.
    positions : array-like, shape (n_samples, n_dims)
        Positions in the environment's coordinates. A NaN row or a position
        outside the environment gives NaN.
    intervals : Intervals or IntervalSet, optional
        Trials or other intervals. Samples outside every interval are NaN;
        overlapping intervals are refused. Without ``intervals`` every
        sample uses the one ``start`` and ``goal`` (to plot a trial that
        lands on the goal, see Notes). A pynapple ``IntervalSet`` is read as
        :func:`~neurospatial.io.from_pynapple` reads it: metadata as fields,
        its inclusive ``end`` as the half-open stop.
    start, goal : str, array-like of shape (n_dims,), or per-interval sequence
        A region name (the region's bins: those whose centres lie in a
        polygon region; for a point region, see ``point_radius``), a point
        in environment coordinates, or, with ``intervals``, one region name
        or point per interval in row order (a ``pandas.Series`` indexed by
        ``interval_id`` is aligned by id), for example
        ``goal=trials.end_region``. ``None`` marks an unknown start or goal:
        that interval's samples are NaN. Well or port numbers are not region
        names: map them first (``trials.goal_well.map({1: "well_1", ...})``).
        In 2-D and 3-D a field of numbers or integers with one per interval
        are refused; in 1-D a number is a position, so a well number there
        would be read as that many units along the track without an error.
    point_radius : float, optional
        Size, in environment units, of point starts and goals (point regions
        and points) off a linearized track: each becomes the bins whose
        centres lie within ``point_radius`` of the point, for example the
        radius of a reward well. Required for point targets in 2-D and 3-D
        environments that are not linearized tracks (see Notes); ignored in
        1-D and on a linearized track. For a well used across analyses, give
        the region its size once instead: ``regions.add(name, point=...,
        radius=...)``.

    Returns
    -------
    ndarray of float64, shape (n_samples,)
        Progress in ``[0, 1]``: 0 on start bins, 1 on goal bins. NaN for a
        sample outside the environment or outside every interval, in a part
        of the environment that does not connect to both start and goal, or
        in an interval whose start or goal is unknown (``None``) or whose
        start and goal share a bin.

    Raises
    ------
    ValueError
        If a single ``start`` and ``goal`` share a bin; if a point start or
        goal off a linearized track has no ``point_radius`` or one that
        covers fewer than two bins; if a point is outside the environment, a
        region covers no bin, ``times`` or ``positions`` are invalid, or
        intervals overlap.
    TypeError
        If ``intervals`` is not an Intervals or a pynapple IntervalSet.
    RegionNotFoundError
        If a region name is not in ``env.regions``.
    CartesianOnlyError
        If ``env`` is an egocentric polar environment.
    MissingCellGeometryError
        If the environment has no cell faces, from which the conductances
        come.

    Warns
    -----
    UserWarning
        If at least half of the analyzed samples are NaN, with the fraction
        and the main cause.

    Notes
    -----
    **Guarantees.**

    - Values lie in ``[0, 1]``, and no bin other than the start and goal bins
      is higher or lower than all its neighbours: progress has no false peaks
      or dips.
    - In a dead end (a side branch that rejoins nothing) no value exceeds the
      highest value at its entrance. On a linearized track the whole dead end
      has exactly the junction's value, so running into a wrong arm never
      raises progress. In 2-D it is close to flat but not exact.
    - On a linearized track whose start-to-goal connection is a single path
      with side branches (a tree-shaped maze), progress along the route is
      exactly the fraction of the route's length covered.
    - Across the width of a uniform corridor it is constant.

    **Caveats.**

    - *Alternative routes.* When start and goal are joined by more than one
      path (a loop), progress depends on the whole layout, not only on the
      path the animal took: opening or closing a barrier elsewhere changes
      progress along an unchanged path, and progress is not the route
      fraction (0.10 to 0.18 off it on the crossword, Hamlet, honeycomb,
      Hampton Court and rat HexMaze tracks). Compare sessions only with the
      same maze configuration.
    - *Commitment, not distance.* Where the width of the corridor or open
      space changes along the way, a wide section takes a smaller share of
      progress than its length. For a fraction proportional to distance on
      a corridor maze, pass its linearized track (for example
      ``MazeEnvironments.env_track``). For centimetres, use
      :func:`distance_to_goal`. The effect can be large: on the Sungod
      maze, whose home port sits in an open 40 x 57 cm box, 2-D progress
      from the home port to an arm port is up to 0.33 off the route
      fraction (0.01 on its track).
    - *Start and goal size, bin size.* A larger goal raises values. On 2-D
      grids, how many bins wide a narrow corridor is changes values (up to
      0.08 between 2 and 1 cm bins on a 7 cm wide W-maze; the track is
      exact). In 2-D and 3-D a start or goal of a single bin gives values
      that change with bin size and do not settle, so point targets need
      ``point_radius``; with a physical size they converge (5 cm discs on a
      T-maze: within 0.001 between 0.5 and 0.25 cm bins). Compare conditions
      with the same regions and environment.
    - *Environment coverage.* Values depend on the whole environment. An
      environment built from tracked samples (``Environment.from_samples``)
      has holes where the animal never went, and they shift progress near
      them (a 30 cm unvisited patch in a 1 m box: median change 0.007, up to
      0.16 at the patch). Use an environment built from the apparatus (maze
      factories, polygons, tracks) to compare sessions.
    - *Corners.* Bins that touch only at a corner share no face, so they are
      not connected here: progress can be NaN where
      :func:`distance_to_goal`, which follows the bin graph, is finite.
    - *Pauses.* Progress stays constant while the animal stops; exclude
      pauses with a speed threshold when you relate firing to progress.
    - *Retrospective goals.* ``goal=trials.end_region`` is where the animal
      went, known only after the trial. For a predictive model, use the
      instructed goal from the task log (a field read with
      ``Intervals.from_dataframe``).
    - *Cost.* One sparse direct solve per distinct start and goal pair,
      reused within a call. Its time grows about linearly with the number of
      bins on tracks and in 2-D (about 5 ms for 15,000 bins) and faster in
      3-D, where the matrix fills in more (20 ms for 6,500 bins, 2 s for
      33,000).

    **Plotting a trial that lands on the goal.** Intervals are half-open, so a
    trial's arrival sample is just outside it. Compute progress without
    ``intervals`` for one start and goal, and plot
    ``progress[s.start : s.stop + 1]`` for each ``s`` in
    ``trials.sample_slices(times)``.

    Examples
    --------
    Progress from the start of a T-maze to its left well, on the linearized
    track (95 cm of stem, then 45 cm of arm):

    >>> import numpy as np
    >>> from neurospatial.behavior import path_progress
    >>> from neurospatial.simulation.mazes import make_t_maze
    >>> track = make_t_maze().env_track
    >>> positions = np.array(
    ...     [
    ...         track.regions["start"].metadata["center"],
    ...         [0.0, 2.5],  # halfway up the stem
    ...         track.regions["well_left"].metadata["center"],
    ...         track.regions["well_right"].metadata["center"],  # the wrong arm's end
    ...     ]
    ... )
    >>> progress = path_progress(
    ...     track, np.arange(4.0), positions, start="start", goal="well_left"
    ... )
    >>> progress.round(2).tolist()
    [0.0, 0.33, 1.0, 0.69]

    The wrong arm keeps the junction's value, 90 / 130 = 0.69 (90 cm of stem
    between the 5 cm start disc and the junction, out of the 130 cm route to
    the left well's disc): running into it is no progress.

    See Also
    --------
    distance_to_goal : Remaining distance to the goal, in environment units.
    time_to_goal : Time until the goal is reached.
    """
    call = "path_progress"
    times, positions, intervals = _validated_route_samples(
        env, times, positions, intervals, call=call
    )
    radius = validate_finite_scalar(
        point_radius, name="point_radius", positive=True, none_allowed=True
    )
    # One bin converges in 1-D, so only 2-D and 3-D point targets need a size.
    sized = env.n_dims > 1 and not env.is_linearized_track
    rows, starts, single_start = _rows_and_targets(
        intervals, times, start, name="start", call=f"{call}()", n_dims=env.n_dims
    )
    _, goals, single_goal = _rows_and_targets(
        intervals,
        times,
        goal,
        name="goal",
        call=f"{call}()",
        n_dims=env.n_dims,
        rows=rows,
    )
    bins = env.bin_at(positions)
    from neurospatial.geometry import MissingCellGeometryError

    try:
        conductances = env.get_diffusion_geometry().W
    except MissingCellGeometryError as exc:
        raise MissingCellGeometryError(
            _format_error(
                f"{call}() needs the cell faces of the environment's bins: "
                "progress is computed from the conductances between bins.",
                why=str(exc).splitlines()[0],
                fix="build the environment with a layout that has cell geometry "
                "(a grid, polygon, mesh or track), or use distance_to_goal.",
            )
        ) from exc

    progress = np.full(times.size, np.nan)
    unknown = np.zeros(times.size, dtype=bool)
    shared_bin = np.zeros(times.size, dtype=bool)
    fields: dict[tuple[Any, ...], NDArray[np.float64] | None] = {}
    target_bins: dict[tuple[Any, ...], NDArray[np.intp]] = {}

    def bins_of(target: Any, name: str) -> NDArray[np.intp]:
        key = (name, *_target_key(target))
        if key not in target_bins:
            target_bins[key] = _target_bins(
                env,
                target,
                name=name,
                call=f"{call}()",
                point_radius=radius,
                sized=sized,
            )
        return target_bins[key]

    for row, samples in _samples_by_row(rows):
        st, gl = starts[row], goals[row]
        if st is None or gl is None:
            unknown[samples] = True
            continue
        key = (*_target_key(st), *_target_key(gl))
        if key not in fields:
            start_bins, goal_bins = bins_of(st, "start"), bins_of(gl, "goal")
            overlap = np.intersect1d(start_bins, goal_bins)
            if overlap.size and single_start and single_goal:
                raise ValueError(
                    _format_error(
                        f"{call}(): start and goal share {overlap.size} bin(s), so "
                        "progress between them is undefined.",
                        fix="pass a start and goal that do not overlap.",
                    )
                )
            fields[key] = (
                None
                if overlap.size
                else _harmonic_progress(conductances, start_bins, goal_bins)
            )
        field_values = fields[key]
        if field_values is None:
            shared_bin[samples] = True
            continue
        located = samples[bins[samples] >= 0]
        progress[located] = field_values[bins[located]]

    analyzed = rows >= 0
    _warn_if_mostly_nan(
        progress,
        analyzed,
        {
            "the position is outside the environment": analyzed & (bins < 0),
            "the start or goal is unknown (None)": unknown,
            "the start and goal share a bin": shared_bin,
            "the position does not connect to both start and goal": analyzed
            & (bins >= 0)
            & ~unknown
            & ~shared_bin,
        },
        call=f"{call}()",
    )
    return progress


@reads_pynapple_samples
def distance_to_goal(
    env: Environment[CartesianSystem],
    times: ArrayLike,
    positions: ArrayLike,
    intervals: Intervals[Any] | None = None,
    *,
    goal: Any,
    metric: DistanceMetric = "geodesic",
    cost_map: ArrayLike | None = None,
    terrain_difficulty: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Distance from each sample's position to the goal.

    Answers "how far is the goal from here": the length of the shortest path
    inside the environment (``metric="geodesic"``, around walls) or the
    straight-line distance, in environment units, from the position to the
    goal. It needs no start. With ``cost_map`` or ``terrain_difficulty`` it
    is the cost of the cheapest path instead, the spatial equivalent of a
    value function in reinforcement learning. For progress as a fraction of
    the route, use :func:`path_progress`; for distance to the reward that
    was actually delivered, chosen by time, :func:`distance_to_reward`.

    Parameters
    ----------
    env : Environment
        The environment the positions were recorded in.
    times : array-like, shape (n_samples,)
        Sample times in seconds, finite and strictly increasing.
    positions : array-like, shape (n_samples, n_dims)
        Positions in the environment's coordinates. A NaN row or a position
        outside the environment gives NaN.
    intervals : Intervals or IntervalSet, optional
        Trials or other intervals. Samples outside every interval are NaN;
        overlapping intervals are refused. A pynapple ``IntervalSet`` is read as
        :func:`~neurospatial.io.from_pynapple` reads it: metadata as fields,
        its inclusive ``end`` as the half-open stop.
    goal : str, array-like of shape (n_dims,), or per-interval sequence
        A region name (a polygon region is its bins, measured to the nearest
        bin centre; a point region is its location), a point in environment
        coordinates, or, with ``intervals``, one per interval in row order
        (a ``pandas.Series`` indexed by ``interval_id`` is aligned by id),
        for example ``goal=trials.end_region``. ``None`` marks an unknown
        goal: that interval's samples are NaN. Well or port numbers are not
        region names: map them first
        (``trials.goal_well.map({1: "well_1", ...})``). In 2-D and 3-D a
        field of numbers or integers with one per interval are refused; in
        1-D a number is a position, so a well number there would be read as
        that many units along the track without an error.
    metric : {"geodesic", "euclidean"}, default="geodesic"
        ``"geodesic"`` is the length of the shortest path that stays inside
        the environment, measured from the position itself; ``"euclidean"``
        is the straight line. A geodesic goal point must lie inside the
        environment.
    cost_map : array-like, shape (n_bins,), optional
        Extra cost per unit length of travel through each bin, dimensionless
        and ``>= 0``: ``0`` is plain travel and ``9`` makes moving through
        the bin cost ten times its length (learned avoidance, punishment
        zones). Default: no extra cost. Needs ``metric="geodesic"``.
    terrain_difficulty : array-like, shape (n_bins,), optional
        Movement difficulty multiplier per bin, dimensionless and ``> 0``.
        Default: 1 everywhere. Needs ``metric="geodesic"``.

    Returns
    -------
    ndarray of float64, shape (n_samples,)
        Distance in environment units (or travel cost, see Notes); 0 at the
        goal point or a goal bin centre. NaN outside the environment, outside
        every interval, or with an unknown goal; ``inf`` (geodesic) where
        the goal cannot be reached.

    Raises
    ------
    ValueError
        If a geodesic goal point is outside the environment, a region covers
        no bin, ``metric`` is not recognized, a cost is given with
        ``metric="euclidean"``, ``cost_map`` or ``terrain_difficulty`` does
        not have one finite value per bin in range, ``times`` or
        ``positions`` are invalid, or intervals overlap.
    TypeError
        If ``intervals`` is not an Intervals or a pynapple IntervalSet.
    RegionNotFoundError
        If a region name is not in ``env.regions``.
    CartesianOnlyError
        If ``env`` is an egocentric polar environment.

    Warns
    -----
    UserWarning
        If at least half of the analyzed samples are NaN, with the fraction
        and the main cause.

    Notes
    -----
    ``goal=trials.end_region`` is where the animal went, known only after the
    trial; for a predictive model use the instructed goal from the task log.

    **Travel cost.** Each bin has a cost rate per unit length,
    ``rate = terrain_difficulty * (1 + cost_map)``. The cost of a path is the
    integral of the rate along it, each stretch taking the rate of the bin it
    runs through, and the result is the cheapest path inside the environment
    to the nearest goal bin (the bin holding a goal point). Paths are the same
    any-angle paths as the geodesic distance, so with a rate of 1 everywhere
    the cost is the geodesic distance between bin centres. Because cost is per
    unit length it does not depend on the bin size: on a 100 cm corridor with
    ``cost_map = np.ones(n_bins)``, a bin 80 cm from the goal costs 160 at
    every bin size. The cost is measured from the centre of the position's
    bin, so it changes in steps of one bin, while the plain distance is
    measured from the position itself.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment, Regions
    >>> from neurospatial.behavior import distance_to_goal
    >>> line = np.column_stack([np.linspace(0, 100, 101), np.zeros(101)])
    >>> env = Environment.from_samples(line, bin_size=10.0)
    >>> regions = Regions()
    >>> _ = regions.add("reward", point=(100.0, 0.0))
    >>> env = env.with_regions(regions)
    >>> positions = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]])
    >>> distance_to_goal(env, np.arange(3.0), positions, goal="reward").tolist()
    [100.0, 50.0, 0.0]

    Crossing a punished stretch (the bins centred at 60 and 70, 20 cm in
    all) costs ten times its length, 180 more than the distance:

    >>> cost_map = np.zeros(env.n_bins)
    >>> cost_map[env.bin_at(np.array([[60.0, 0.0], [70.0, 0.0]]))] = 9.0
    >>> distance_to_goal(
    ...     env, np.arange(3.0), positions, goal="reward", cost_map=cost_map
    ... ).round(6).tolist()
    [280.0, 230.0, 0.0]

    See Also
    --------
    path_progress : Progress from start to goal, between 0 and 1.
    distance_to_reward : Distance to the reward location chosen by time.
    """
    from neurospatial.behavior._goals import goal_rows

    call = "distance_to_goal"
    paths = metric_paths(metric, env, call=call)
    times, positions, intervals = _validated_route_samples(
        env, times, positions, intervals, call=call
    )
    rate = None
    if cost_map is not None or terrain_difficulty is not None:
        if paths is None:
            raise ValueError(
                _format_error(
                    f"{call}(): cost_map and terrain_difficulty need "
                    "metric='geodesic'.",
                    why="a cost is integrated along a path inside the environment; "
                    "a straight line has no bins to weight.",
                    fix="pass metric='geodesic' (the default), or drop the costs.",
                )
            )
        extra = _validated_bin_values(
            cost_map, env, name="cost_map", default=0.0, positive=False, call=call
        )
        difficulty = _validated_bin_values(
            terrain_difficulty,
            env,
            name="terrain_difficulty",
            default=1.0,
            positive=True,
            call=call,
        )
        rate = difficulty * (1.0 + extra)
    rows = goal_rows(env, times, intervals, goal, paths=paths, call=call, disjoint=True)
    distances = np.full(times.size, np.nan)
    analyzed = np.zeros(times.size, dtype=bool)
    unknown = np.zeros(times.size, dtype=bool)
    # One query per distinct goal: each sample's distance depends only on
    # its own position.
    samples_of: dict[int, list[NDArray[np.intp]]] = {}
    goals = {}
    for row in rows:
        analyzed[row.samples] = True
        if row.goal is None:
            unknown[row.samples] = True
            continue
        goals[id(row.goal)] = row.goal
        samples_of.setdefault(id(row.goal), []).append(
            np.arange(times.size)[row.samples]
        )
    for key, parts in samples_of.items():
        samples = np.concatenate(parts)
        target = goals[key]
        if rate is None:
            distances[samples] = target.distances(env, paths, positions[samples])
            continue
        assert paths is not None
        source = (
            target.bins
            if target.bins is not None
            else env.bin_at(np.asarray(target.point)[None, :]).astype(np.intp)
        )
        field = paths.costs_from_bins(source, rate)
        bins = env.bin_at(positions[samples])
        located = bins >= 0
        distances[samples[located]] = field[bins[located]]

    _warn_if_mostly_nan(
        distances,
        analyzed,
        {
            "the position is outside the environment": analyzed & ~unknown,
            "the goal is unknown (None)": unknown,
        },
        call=f"{call}()",
    )
    return distances


@reads_pynapple_samples
def distance_to_reward(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    reward_times: NDArray[np.float64],
    reward_positions: NDArray[np.float64] | None = None,
    *,
    metric: DistanceMetric = "geodesic",
    mode: Literal["nearest", "last", "next"] = "nearest",
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Compute distance to reward location at each sample.

    For each sample, computes the distance from current position to the
    reward location. The reward location is determined by the ``mode``:

    - "nearest": Uses the closest reward in time (before or after)
    - "last": Uses the most recent reward before current time
    - "next": Uses the next upcoming reward after current time

    Parameters
    ----------
    env : Environment
        Spatial environment.
    times : NDArray[np.float64], shape (n_samples,)
        Sample timestamps (seconds).
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates at each sample time. Must match environment
        dimensionality (e.g., 2D for grid environments). A row with a NaN
        coordinate is a missing sample (a lost tracking frame): its distance
        is NaN.
    reward_times : NDArray[np.float64], shape (n_events,), or pynapple Ts/Tsd
        Reward event timestamps (seconds). A pynapple ``Ts`` or ``Tsd`` is read by its timestamps.
    reward_positions : NDArray[np.float64], shape (n_events, n_dims), optional
        Reward locations in environment coordinates. If None, each reward
        location is read from the tracked positions at its reward time (see
        Notes); a reward whose location cannot be read that way has an
        unknown location. A NaN row is an unknown location.
    metric : {'geodesic', 'euclidean'}, default='geodesic'
        ``'euclidean'`` is the straight-line distance between the position and
        the reward location. ``'geodesic'`` is the length of the shortest path
        between two positions that stays inside the environment: it goes around
        walls and never leaves the environment, and on a track it runs along
        the track's centerline. Both are measured from the position itself, not
        its bin center.
    mode : {'nearest', 'last', 'next'}, default='nearest'
        Which reward to use for distance computation:

        - 'nearest': Distance to temporally closest reward (before or after).
        - 'last': Distance to most recent reward. NaN before first reward.
        - 'next': Distance to next upcoming reward. NaN after last reward.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). Samples outside every window get NaN.
        ``None`` (default) means unrestricted. Epochs do not change which
        reward a sample uses or where an inferred reward is: rewards are often
        delivered while the animal is still, outside running epochs, and their
        locations are still read from the tracked positions.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. A sample only uses rewards from its own
        recording, and no reward location is interpolated across a break.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Distance from current position to the reward location at each
        sample. Units match environment coordinates (e.g., cm).
        Returns NaN when:

        - No relevant reward exists (e.g., mode="last" before first reward,
          or no reward in the sample's own recording)
        - Position is missing (NaN) or outside environment (invalid bin)
        - The sample lies outside ``epochs``
        - Reward position is unknown or outside environment

    Raises
    ------
    ValueError
        If positions and times have different lengths.
        If reward_positions provided but doesn't match reward_times length.
        If positions dimensionality doesn't match environment.
        If ``times`` or ``reward_times`` contain non-finite values, or
        ``positions`` or ``reward_positions`` contains an infinite value.
        If ``times`` and ``positions`` are empty, or ``times`` is not
        strictly increasing.
        If ``max_gap`` is not positive, or ``epochs`` or ``break_before`` is
        malformed.
        If ``metric`` is not ``'geodesic'`` or ``'euclidean'``, or ``mode``
        is not ``'nearest'``, ``'last'`` or ``'next'``.

    Warns
    -----
    UserWarning
        When some reward locations cannot be inferred from ``positions``,
        naming how many and why (outside the recorded span, in a gap longer
        than ``max_gap``, across a declared break or at missing positions).
        When explicit ``reward_times`` fall outside the recorded span of
        ``times`` (often a different clock, such as milliseconds), or a
        known reward location lies outside the environment.

    Notes
    -----
    **Inferred reward locations**:

    When ``reward_positions`` is not provided, a reward exactly at an
    observed sample time uses that sample's position. Otherwise the position
    is linearly interpolated between the nearest observed samples before and
    after the reward time. A missing sample (a NaN row) between them is
    bridged like a deleted row when those observed samples are no more than
    ``max_gap`` apart; only a longer gap, a declared break between them, or a
    reward before the first or after the last observed position leaves the
    location unknown. ``epochs`` play no part here. The samples that select a
    reward of unknown location get NaN; they never fall back to another
    reward or to the nearest recorded position. Provide explicit
    ``reward_positions`` (for example the reward port's coordinates) when the
    tracked position at the reward time is not reliable.

    **Reward selection**:

    Without ``break_before`` every reward is eligible for every sample.
    With it, each recording spans from the first sample after a declared
    break to the next break (``-inf``/``+inf`` at the ends), and a sample only
    selects rewards whose times fall in its own recording's span. Gaps longer
    than ``max_gap`` do not restrict selection: a tracking dropout does not
    change the task.

    **Performance**:

    The geodesic distance is grown once per bin holding a reward location
    and kept on the environment, so rewards delivered at the same ports
    cost one field per port.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior import distance_to_reward
    >>> x, y = np.meshgrid(np.arange(0.0, 10.0), np.arange(0.0, 10.0))
    >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=1.0)
    >>> times = np.arange(6) / 10  # seconds
    >>> positions = np.c_[np.arange(6.0), np.full(6, 5.0)]  # running along y = 5
    >>> reward_times = np.array([0.3])  # the reward is at sample 3, x = 3
    >>> distance_to_reward(env, times, positions, reward_times, metric="euclidean")
    array([3., 2., 1., 0., 1., 2.])

    A lost tracking frame has no distance:

    >>> positions[4] = np.nan
    >>> distance_to_reward(env, times, positions, reward_times, metric="euclidean")
    array([ 3.,  2.,  1.,  0., nan,  2.])

    If a second recording starts at sample 4, its samples have no "last"
    reward:

    >>> new_recording = np.array([False, False, False, False, True, False])
    >>> distance_to_reward(
    ...     env,
    ...     times,
    ...     positions,
    ...     reward_times,
    ...     metric="euclidean",
    ...     mode="last",
    ...     break_before=new_recording,
    ... )
    array([nan, nan, nan,  0., nan, nan])

    As a GLM regressor next to the time to the nearest reward:

    >>> from neurospatial.events import time_to_nearest_event
    >>> X = np.column_stack(  # doctest: +SKIP
    ...     [
    ...         time_to_nearest_event(times, reward_times, max_time=5.0),
    ...         distance_to_reward(env, times, positions, reward_times),
    ...     ]
    ... )

    See Also
    --------
    neurospatial.events.time_to_nearest_event : Temporal distance to events
    distance_to_boundary : Distance to environment boundaries
    distance_to_goal : Distance from each position to a goal
    neurospatial.ops.nearest_boundary : Exact distance and direction to walls
    """
    from neurospatial._intervals import as_intervals, validate_max_gap
    from neurospatial._observation import empty_samples_error, validate_break_before
    from neurospatial._validation import (
        timestamps_of,
        validate_finite,
        validate_times_positions,
    )

    call = "distance_to_reward"
    require_cartesian(
        env.coordinate_system,
        "distance_to_reward()",
        why="It reads positions as the animal's allocentric positions, located "
        "in the environment's bins.",
        fix="use the allocentric environment the positions were recorded in",
    )
    paths = metric_paths(metric, env, call=call)
    if mode not in ("nearest", "last", "next"):
        msg = f"mode must be 'nearest', 'last', or 'next', got '{mode}'."
        raise ValueError(msg)

    # Reward locations are interpolated between neighbouring samples and
    # rewards are assigned to recordings in time order, so times must be
    # strictly increasing. A NaN position is a missing sample and gives NaN
    # distance; an infinite one is a corrupt value.
    times, positions = validate_times_positions(times, positions, call=call)
    if times.size == 0:
        raise empty_samples_error(call, ("times", "positions"))
    positions = positions.reshape(times.size, -1)
    reward_times = validate_finite(
        timestamps_of(reward_times, name="reward_times"), name="reward_times"
    )
    if reward_times.ndim != 1:
        msg = f"reward_times must be 1-D (n_events,), got shape {reward_times.shape}."
        raise ValueError(msg)
    n_samples = times.size
    validate_max_gap(max_gap)
    breaks = validate_break_before(break_before, n_samples, call=call)
    resolved_epochs = as_intervals(epochs, name="epochs")

    if len(reward_times) == 0:
        return np.full(n_samples, np.nan, dtype=np.float64)

    # Get or interpolate reward positions
    if reward_positions is not None:
        reward_positions = validate_finite(
            np.asarray(reward_positions, dtype=np.float64),
            name="reward_positions",
            allow_nan=True,
        )
        if reward_positions.ndim == 1 and positions.shape[1] == 1:
            reward_positions = reward_positions.reshape(-1, 1)
        if len(reward_positions) != len(reward_times):
            msg = (
                f"reward_positions and reward_times must have same length. "
                f"Got reward_positions: {len(reward_positions)}, "
                f"reward_times: {len(reward_times)}."
            )
            raise ValueError(msg)
        _warn_rewards_outside_recording(times, reward_times)
    else:
        reward_positions = _infer_reward_positions(
            times,
            positions,
            reward_times,
            max_gap=max_gap,
            break_before=breaks,
        )

    # Map positions and reward locations to bins (NaN rows are -1)
    position_bins = env.bin_at(positions)
    reward_bins = env.bin_at(reward_positions)
    _warn_rewards_outside_environment(reward_positions, reward_bins)

    # Sort rewards by time for efficient lookup
    sorted_idx = np.argsort(reward_times)
    sorted_reward_times = reward_times[sorted_idx]
    sorted_reward_bins = reward_bins[sorted_idx]
    sorted_reward_positions = reward_positions[sorted_idx]
    n_rewards = len(sorted_reward_times)

    # Recording of each sample (by declared breaks) and of each reward (by
    # time). Without declared breaks every reward is eligible.
    recordings = None
    if breaks is not None:
        recordings = (
            recording_index(breaks),
            np.searchsorted(
                break_sample_times(times, breaks), sorted_reward_times, side="right"
            ),
        )

    # Find the relevant reward of each sample (an index into the time-sorted
    # rewards, possibly out of range when there is none).
    if mode == "nearest":
        idx_right = np.searchsorted(sorted_reward_times, times, side="left")
        idx_left = idx_right - 1
        has_left = idx_left >= 0
        has_right = idx_right < n_rewards
        if recordings is not None:
            has_left &= _same_recording(idx_left, *recordings)
            has_right &= _same_recording(idx_right, *recordings)

        # Time to the left and right rewards; the nearer one wins, the left
        # one on a tie.
        dist_left = np.full(n_samples, np.inf)
        dist_right = np.full(n_samples, np.inf)
        dist_left[has_left] = times[has_left] - sorted_reward_times[idx_left[has_left]]
        dist_right[has_right] = (
            sorted_reward_times[idx_right[has_right]] - times[has_right]
        )
        relevant_reward_idx = np.where(dist_left <= dist_right, idx_left, idx_right)
        no_reward = ~has_left & ~has_right
    else:
        # "last": searchsorted gives the first reward > time, so idx - 1 is the
        # last reward <= time. "next": the first reward >= time.
        if mode == "last":
            relevant_reward_idx = (
                np.searchsorted(sorted_reward_times, times, side="right") - 1
            )
        else:
            relevant_reward_idx = np.searchsorted(
                sorted_reward_times, times, side="left"
            )
        no_reward = (relevant_reward_idx < 0) | (relevant_reward_idx >= n_rewards)
        if recordings is not None:
            no_reward |= ~_same_recording(relevant_reward_idx, *recordings)

    relevant_reward_idx = np.clip(relevant_reward_idx, 0, n_rewards - 1)
    target_bins = sorted_reward_bins[relevant_reward_idx]

    # NaN without a relevant reward, at a missing or outside position, for an
    # unknown or outside reward location, and outside epochs (as samples
    # outside epochs have no speed in compute_speed; each sample is tested as
    # the instant [t, t + ulp)).
    usable = ~no_reward & (target_bins >= 0) & (position_bins >= 0)
    if resolved_epochs is not None:
        usable &= times_in_intervals(resolved_epochs, times)
    starts = positions[usable]
    ends = sorted_reward_positions[relevant_reward_idx[usable]]

    distances = np.full(n_samples, np.nan, dtype=np.float64)
    if paths is None:
        # The exact straight-line distance between the position and the
        # reward location, not between their bin centers.
        distances[usable] = np.linalg.norm(starts - ends, axis=1)
    else:
        # The shortest path inside the environment from each position to its
        # reward's location, with every reward bin's field grown at once.
        end_bins = target_bins[usable]
        paths.prepare(end_bins)
        distances[usable] = paths.between(
            starts, ends, start_bins=position_bins[usable], end_bins=end_bins
        )
    return distances


def _warn_rewards_outside_recording(
    times: NDArray[np.float64], reward_times: NDArray[np.float64]
) -> None:
    """Warn when explicit reward times fall outside the recorded span.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Finite sample times in seconds.
    reward_times : ndarray, shape (n_rewards,)
        Finite reward times in seconds.
    """
    import warnings

    from neurospatial._stacklevel import _external_stacklevel

    first, last = float(times.min()), float(times.max())
    outside = (reward_times < first) | (reward_times > last)
    if outside.any():
        warnings.warn(
            f"{int(outside.sum())} of {len(reward_times)} reward_times fall "
            f"outside the recorded span [{first:g}, {last:g}] s. Samples still "
            "select the temporally nearest, last or next reward, so a reward "
            "time on another clock (for example milliseconds against seconds) "
            "gives plausible but wrong distances. Fix: check that reward_times "
            "are on the same clock as times; ignore this warning if rewards "
            "were truly delivered before or after tracking.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )


def _warn_rewards_outside_environment(
    reward_positions: NDArray[np.float64], reward_bins: NDArray[np.intp]
) -> None:
    """Warn when known reward locations lie in no bin of the environment.

    Parameters
    ----------
    reward_positions : ndarray, shape (n_rewards, n_dims)
        Reward locations; NaN rows are unknown (already reported).
    reward_bins : ndarray of intp, shape (n_rewards,)
        Bin of each location, -1 when it is outside or unknown.
    """
    import warnings

    from neurospatial._stacklevel import _external_stacklevel

    unlocated = (reward_bins < 0) & np.isfinite(reward_positions).all(axis=1)
    if unlocated.any():
        warnings.warn(
            f"{int(unlocated.sum())} of {len(reward_bins)} reward locations lie "
            "outside the environment, so samples whose selected reward is one "
            "of them get NaN distance. Fix: check that reward_positions are in "
            "the environment's coordinates and units, or build the environment "
            "so it covers the reward sites.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )


def _same_recording(
    reward_idx: NDArray[np.intp],
    sample_recording: NDArray[np.intp],
    reward_recording: NDArray[np.intp],
) -> NDArray[np.bool_]:
    """Whether each sample's candidate reward exists and is in its recording.

    Parameters
    ----------
    reward_idx : ndarray of intp, shape (n_samples,)
        Candidate index into the time-sorted rewards; may be out of range.
    sample_recording : ndarray of intp, shape (n_samples,)
        Recording of each sample.
    reward_recording : ndarray of intp, shape (n_rewards,)
        Recording of each time-sorted reward.
    """
    n_rewards = len(reward_recording)
    in_range = (reward_idx >= 0) & (reward_idx < n_rewards)
    clipped = np.clip(reward_idx, 0, n_rewards - 1)
    same: NDArray[np.bool_] = in_range & (reward_recording[clipped] == sample_recording)
    return same


def _infer_reward_positions(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    reward_times: NDArray[np.float64],
    *,
    max_gap: float | None,
    break_before: NDArray[np.bool_] | None,
) -> NDArray[np.float64]:
    """Read each reward location from the tracked positions at its time.

    A reward exactly at an observed sample time uses that sample. Otherwise
    it is interpolated between the observed samples on either side, only when
    the gap between them passes ``max_gap`` and ``break_before`` (missing
    samples in between are skipped as if deleted); ``epochs`` restrict the
    analyzed samples, not where a reward was. Any other reward has an
    unknown (NaN) location, and one ``UserWarning`` says how many and why.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Finite, strictly increasing sample times in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Positions; NaN rows are missing samples.
    reward_times : ndarray, shape (n_rewards,)
        Finite reward times in seconds.
    max_gap, break_before
        The validated gap threshold and declared breaks.

    Returns
    -------
    ndarray, shape (n_rewards, n_dims)
        Reward locations; NaN rows are unknown.
    """
    import warnings

    from neurospatial._intervals import gap_endpoints
    from neurospatial._observation import resolve_observation_support, sample_feature
    from neurospatial._stacklevel import _external_stacklevel

    support = resolve_observation_support(
        times,
        positions,
        max_gap=max_gap,
        epochs=None,
        break_before=break_before,
        result=None,  # the unknown rewards are reported below, with reasons
    )
    columns = []
    for coordinate in positions.T:
        column, known = sample_feature(
            times,
            coordinate,
            support,
            reward_times,
            interpolation="linear",
            kind="linear",
            epochs=None,
        )
        columns.append(column)
    located = np.column_stack(columns)

    n_unknown = int(np.count_nonzero(~known))
    if n_unknown:
        # Why each unknown reward inside the span has no location: the gap
        # holding it (between the observed samples on either side) is too
        # long, crosses a declared break, or has no observed sample on one side.
        in_span = (reward_times >= times[0]) & (reward_times <= times[-1])
        blocked = in_span & ~known
        reasons = {"outside the recorded span": ~in_span}
        n_samples = times.size
        if n_samples > 1:
            k = np.searchsorted(times, reward_times, side="right") - 1
            j = np.clip(k, 0, n_samples - 2)
            start, end, linked = gap_endpoints(support.sample_valid)
            crossed = np.zeros(j.size, dtype=bool)
            if break_before is not None:
                recording = recording_index(break_before)
                crossed = recording[end[j]] != recording[start[j]]
            long_gap = (
                np.zeros(j.size, dtype=bool)
                if max_gap is None
                else times[end[j]] - times[start[j]] > max_gap
            )
            reasons["across a declared break"] = blocked & linked[j] & crossed
            reasons[f"in a gap longer than max_gap={max_gap} s"] = (
                blocked & linked[j] & ~crossed & long_gap
            )
            reasons["before the first or after the last observed position"] = (
                blocked & ~linked[j]
            )
        else:
            reasons["at a missing (NaN) position"] = blocked
        counts = ", ".join(
            f"{int(mask.sum())} {reason}"
            for reason, mask in reasons.items()
            if mask.any()
        )
        warnings.warn(
            f"{n_unknown} of {len(reward_times)} reward locations could not be "
            f"inferred from positions ({counts}; recorded span "
            f"[{times[0]:g}, {times[-1]:g}] s), so samples whose selected "
            "reward is one of them get NaN distance. A reward location is "
            "read from the sample at the reward time, or interpolated only "
            "between observed samples no more than max_gap apart and not "
            "separated by a declared break. Fix: pass reward_positions explicitly (for example the "
            "reward port's coordinates); check that reward_times are on the "
            "same clock as times; if the long intervals are continuous "
            "tracking, raise max_gap.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
    return located


@reads_pynapple_samples
def distance_to_boundary(
    env: Environment,
    positions: NDArray[np.float64],
    *,
    boundary_type: Literal["edge", "region"] = "edge",
    region_name: str | None = None,
    metric: DistanceMetric = "geodesic",
) -> NDArray[np.float64]:
    """
    Compute distance to environment boundary at each position.

    Measures the instantaneous distance from trajectory positions to the
    edge of the environment or to the boundary of a named region.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates at each sample. Must match environment
        dimensionality.
    boundary_type : {'edge', 'region'}, default='edge'
        Type of boundary to compute distance to:

        - 'edge': Distance to the environment's boundary bins,
          :attr:`Environment.boundary_bins`: on a grid, bins next to
          unoccupied space (the outer rim and the rim of every hole); on a
          track, the dead ends of its arms. It is measured to the boundary
          bins' centers, about half a bin inside the walls, and is not the
          exact distance to the arena's walls; for that use
          :func:`neurospatial.ops.nearest_boundary`.
        - 'region': Distance to boundary of a specific named region.
          Requires ``region_name`` parameter.
    region_name : str, optional
        Name of region for boundary_type='region'. Required when
        boundary_type='region', ignored otherwise.
    metric : {'geodesic', 'euclidean'}, default='geodesic'
        ``'euclidean'`` is the straight-line distance. ``'geodesic'`` is the
        length of the shortest path between two positions that stays inside the
        environment: it goes around walls and never leaves the environment, and
        on a track it runs along the track's centerline.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Distance from each position (not its bin center) to the center of
        the nearest boundary bin, in environment units (e.g., cm). NaN for
        positions outside the environment or missing (NaN rows); ``inf``
        (geodesic) where no boundary bin can be reached.

    Raises
    ------
    ValueError
        If boundary_type='region' but region_name is not provided.
        If region_name is provided but region doesn't exist.
        If no boundary bins found for the specified boundary_type.

    Notes
    -----
    **Boundary detection**:

    - **edge**: :attr:`Environment.boundary_bins`, the layout's own rule
      (an inactive or missing lattice neighbor on grids and hexagons, the
      outline of the union on meshes and polygon cells, dead ends on
      tracks).
    - **region**: Finds bins at the boundary of the named region. Boundary
      bins are region bins adjacent to non-region bins.

    **Performance**:

    The geodesic field from the boundary bins is grown once and kept on the
    environment, so repeated calls with the same boundary are lookups.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import distance_to_boundary
    >>> # Distance to the environment's boundary bins (quantized to bins)
    >>> dist_edge = distance_to_boundary(
    ...     env, positions, boundary_type="edge"
    ... )  # doctest: +SKIP
    >>>
    >>> # Distance to goal region boundary
    >>> dist_goal = distance_to_boundary(  # doctest: +SKIP
    ...     env, positions, boundary_type="region", region_name="goal"
    ... )
    >>>
    >>> # GLM design matrix with multiple boundary features
    >>> X = np.column_stack(  # doctest: +SKIP
    ...     [
    ...         distance_to_boundary(env, positions, boundary_type="edge"),
    ...         distance_to_boundary(
    ...             env, positions, boundary_type="region", region_name="goal"
    ...         ),
    ...     ]
    ... )

    See Also
    --------
    distance_to_reward : Distance to reward locations
    distance_to_goal : Distance from each position to a goal
    neurospatial.events.time_to_nearest_event : Temporal distance to events
    neurospatial.ops.nearest_boundary : Exact distance and direction to walls
    neurospatial.ops.compute_boundary_distances : Distance to walls along
        each direction
    """
    from neurospatial.ops.mapping import _require_region_mask

    require_cartesian(
        env.coordinate_system,
        "distance_to_boundary()",
        why="It reads positions as the animal's allocentric positions, located "
        "in the environment's bins.",
        fix="use the allocentric environment the positions were recorded in",
    )

    # Input validation
    from neurospatial._validation import validate_choice

    boundary_type = validate_choice(
        boundary_type,
        ("edge", "region"),
        name="boundary_type",
        owner="distance_to_boundary()",
    )
    paths = metric_paths(metric, env, call="distance_to_boundary")
    positions = np.asarray(positions, dtype=np.float64)
    n_samples = len(positions)

    if n_samples == 0:
        return np.array([], dtype=np.float64)

    if boundary_type == "region" and region_name is None:
        msg = (
            "region_name is required when boundary_type='region'. "
            "Provide the name of the region to compute distance to."
        )
        raise ValueError(msg)

    # Find boundary bins based on boundary_type
    if boundary_type == "edge":
        boundary_bins = env.boundary_bins

    else:
        # Region boundary bins: boundary of the named region
        # region_name is already validated as not None by the earlier check
        assert region_name is not None  # for type checker
        _require_region_mask(
            env,
            region_name,
            operation="distance_to_boundary()",
            argument="region_name",
        )
        boundary_bins = np.asarray(
            _find_region_boundary_bins(env, region_name), dtype=np.intp
        )

    if len(boundary_bins) == 0:
        raise ValueError(
            f"No boundary bins found for boundary_type='{boundary_type}'"
            + (f", region_name='{region_name}'" if region_name else "")
            + ". Check that the environment has valid boundaries."
        )

    boundary_bins = np.unique(np.asarray(boundary_bins, dtype=np.intp))
    return _distance_to_bins(env, paths, boundary_bins, positions)


def _distance_to_bins(
    env: Environment,
    paths: EnvironmentPaths | None,
    bins: NDArray[np.intp],
    positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Distance from each position to the nearest center of ``bins``.

    Parameters
    ----------
    env : Environment
    paths : EnvironmentPaths or None
        Shortest-path engine (:func:`metric_paths`); None for straight lines.
    bins : ndarray of intp, shape (n_targets,)
        Distinct, valid bin indices.
    positions : ndarray, shape (n_samples, n_dims)

    Returns
    -------
    ndarray of float64, shape (n_samples,)
        NaN at positions outside the environment or NaN.
    """
    if paths is not None:
        return paths.from_bins_at(bins, positions)
    from scipy.spatial import cKDTree

    distances = np.full(len(positions), np.nan, dtype=np.float64)
    inside = env.bin_at(positions) >= 0
    distances[inside] = cKDTree(env.bin_centers[bins]).query(positions[inside])[0]
    return distances


def _find_region_boundary_bins(env: Environment, region_name: str) -> list[int]:
    """Find bins at the boundary of a named region.

    A bin is a region boundary bin if it's in the region AND has at least
    one neighbor outside the region.

    Parameters
    ----------
    env : Environment
        Environment.
    region_name : str
        Name of the region.

    Returns
    -------
    list[int]
        Bin indices at the region boundary.
    """
    graph = env.connectivity
    region_bins = set(env.bins_in_region(region_name))

    if not region_bins:
        return []

    boundary_bins = []
    for bin_idx in region_bins:
        neighbors = set(graph.neighbors(bin_idx))
        # Check if any neighbor is outside the region
        if not neighbors.issubset(region_bins):
            boundary_bins.append(bin_idx)

    return boundary_bins


def _validated_bin_values(
    values: ArrayLike | None,
    env: Environment,
    *,
    name: str,
    default: float,
    positive: bool,
    call: str,
) -> NDArray[np.float64]:
    """Check a per-bin cost array: shape ``(n_bins,)``, finite, and in range.

    Parameters
    ----------
    values : array-like, shape (n_bins,), or None
        Per-bin values; None means ``default`` everywhere.
    env : Environment
        Environment whose bins the values describe.
    name : str
        Argument name shown in errors.
    default : float
        Value of every bin when ``values`` is None.
    positive : bool
        Require ``> 0`` (a multiplier) instead of ``>= 0`` (an extra cost).
    call : str
        Public function name, used in messages.

    Returns
    -------
    ndarray, shape (n_bins,)
    """
    if values is None:
        return np.full(env.n_bins, default)
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (env.n_bins,):
        raise ValueError(
            _format_error(
                f"{call}() got {name} with shape {array.shape}, but the "
                f"environment has {env.n_bins} bins.",
                why=f"{name} holds one value per bin.",
                fix=f"pass an array of shape ({env.n_bins},), for example "
                f"np.full(env.n_bins, {default})",
            )
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(
            _format_error(
                f"{call}() got {np.count_nonzero(~np.isfinite(array))} "
                f"non-finite value(s) in {name}; every value must be finite.",
                why="a NaN or infinite edge weight makes every cost infinite "
                "or undefined.",
                fix=f"fill unknown bins with {default} (no change) or remove "
                "impassable bins from the environment",
            )
        )
    bad = array <= 0 if positive else array < 0
    if np.any(bad):
        rule = "are not positive" if positive else "are negative"
        raise ValueError(
            _format_error(
                f"{call}() got {np.count_nonzero(bad)} value(s) in {name} "
                f"that {rule}; the smallest is {array.min():g}.",
                why="shortest paths need positive edge weights: a zero "
                "difficulty makes travel free and a negative cost makes a "
                "loop endlessly cheaper.",
                fix=f"use {name} > 0 as a multiplier"
                if positive
                else f"use {name} >= 0, with 0 for plain travel",
            )
        )
    return array


def _require_trial_outcomes(trials: Intervals[Any]) -> None:
    """Raise unless every row's ``outcome`` is a trial outcome."""
    from neurospatial.behavior.segmentation import _OUTCOMES

    fix = (
        "map your labels to trial outcomes first, for example "
        "trials.with_fields(outcome=trials.<label>.map({'correct': 'reached', "
        "'error': 'returned'})), using 'reached' (arrived at the goal), "
        "'returned' (came back without arriving) or 'censored' (observation "
        "stopped first)"
    )
    why = (
        "time_to_goal counts down only in trials whose outcome is 'reached'; "
        "any other label would leave those trials NaN without notice."
    )
    if "outcome" not in trials.fields:
        raise ValueError(
            _format_error(
                f"time_to_goal() got trials with no 'outcome' field; its "
                f"fields are {list(trials.fields)}.",
                why=why,
                fix=fix,
            )
        )
    seen = list(dict.fromkeys(trials.outcome.tolist()))
    unknown = [value for value in seen if value not in _OUTCOMES]
    if unknown:
        raise ValueError(
            _format_error(
                f"time_to_goal() got trial outcomes {unknown} that are not "
                f"{list(_OUTCOMES)}; the outcomes seen are {seen}.",
                why=why,
                fix=fix,
            )
        )


@reads_pynapple_samples
def time_to_goal(
    times: NDArray[np.float64],
    trials: Intervals[Any],
) -> NDArray[np.float64]:
    """Compute time remaining until goal arrival for each trial.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps for entire session, strictly increasing (seconds).
    trials : Intervals or IntervalSet
        Trials from detect_trials(), read_trials() or a task log, with an
        ``outcome`` field holding trial outcomes: ``"reached"``,
        ``"returned"`` or ``"censored"``. Only reached trials are included;
        returned and censored trials, whose arrival was never observed, are
        not. The arrival is each trial's ``stop``. Map a task log's own
        labels first, for example
        ``trials.with_fields(outcome=trials.result.map({"correct":
        "reached", "error": "returned"}))``. A pynapple ``IntervalSet`` is read as
        :func:`~neurospatial.io.from_pynapple` reads it: metadata as fields,
        its inclusive ``end`` as the half-open stop.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        ``stop - t`` in seconds for samples in a reached trial's
        ``[start, stop)``; NaN outside them. The arrival sample, at ``stop``,
        is outside its trial.

    Raises
    ------
    ValueError
        If ``times`` is empty, not 1-D, holds a non-finite value or is not
        strictly increasing, if two reached trials overlap, or if
        ``trials`` has no ``outcome`` field or holds an outcome other than
        ``"reached"``, ``"returned"`` or ``"censored"``.
    TypeError
        If ``trials`` is not an Intervals or a pynapple IntervalSet.

    Warns
    -----
    UserWarning
        If reached trials contain no sample of ``times`` (usually trials on a
        different clock).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Intervals
    >>> from neurospatial.behavior import time_to_goal
    >>> trials = Intervals.from_arrays(
    ...     [1.0, 5.0], [4.0, 8.0], outcome=["reached", "censored"]
    ... )
    >>> time_to_goal(np.arange(10.0), trials).tolist()
    [nan, 3.0, 2.0, 1.0, nan, nan, nan, nan, nan, nan]

    See Also
    --------
    detect_trials : Trials between visits to named regions
    path_progress : Normalized progress along path
    """
    from neurospatial._interval_table import require_intervals
    from neurospatial._observation import _validated_signal_times

    trials = require_intervals(trials, name="trials", owner="time_to_goal()")
    _require_trial_outcomes(trials)
    times = _validated_signal_times(times, call="time_to_goal")
    # Only an observed arrival has a time to goal: timed-out, returned and
    # censored trials (cut off by a gap, an epoch edge, a break or the end of
    # the recording) never reached it.
    reached = trials[trials.outcome == "reached"]
    arrival: NDArray[np.float64] = reached.sample_values(times, reached.stop)
    return arrival - times


# A heading change wider than this is a reversal (within 30 degrees of
# straight back), which is neither a left nor a right turn.
_MAX_TURN_ANGLE = 5 * np.pi / 6


@reads_pynapple_samples
def graph_turn_sequence(
    env: Environment[CartesianSystem],
    times: ArrayLike,
    positions: ArrayLike,
    intervals: Intervals[Any] | None = None,
    *,
    tolerance: float | None = None,
) -> str | pd.Series:
    """Extract discrete turn sequence from trajectory.

    Reduces the sequence of visited bins to the straight legs of the route
    and labels the turn at each corner between legs. Works for any Cartesian
    layout type: a track graph (T-maze, multiple-T maze, plus maze) or an
    open field.

    Left and right are the animal's own (egocentric) sides: a turn is "left"
    when the new direction is counterclockwise from the old one, so heading
    North and then West is a left turn. This assumes the environment's
    coordinates are y-up (x East, y North), the package convention. Positions
    in image pixels, where y points down, mirror every turn; flip y first,
    for example ``positions[:, 1] = height - positions[:, 1]``, before
    building the environment.

    Parameters
    ----------
    env : Environment
        Cartesian spatial environment (any layout type).
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing. Their order
        sets the direction of travel.
    positions : ArrayLike, shape (n_samples, n_dims)
        Positions in the environment's coordinates. A NaN row or a position
        outside the environment is skipped as if its sample were deleted.
    intervals : Intervals or IntervalSet, optional
        Trials or other intervals; each interval's route is its own samples,
        ``[start, stop)``, and the result has one turn sequence per interval.
        A pynapple ``IntervalSet`` is read as
        :func:`~neurospatial.io.from_pynapple` reads it.
    tolerance : float, optional
        How far, in the environment's units, the route may stray from a
        straight leg before the departure counts as a corner. Default: one
        bin diagonal, ``sqrt(2) * env.bin_spacing``, the largest straying
        that bin quantization alone can produce (see Notes). Raise it to
        ignore wider wobbles; a corner whose legs are shorter than about
        ``tolerance`` cannot be told apart from quantization and is not
        reported.

    Returns
    -------
    str or pandas.Series
        Turn sequence like "left-right-left", or empty string "" if no
        turns detected; with ``intervals``, one per interval indexed by
        ``interval_id``. Each turn is separated by "-". Only the first two
        coordinates set the turn direction. A heading change of more than
        150 degrees is a reversal, neither left nor right, and is left out.

    Raises
    ------
    CartesianOnlyError
        If ``env`` is polar: left and right come from cross products of
        Cartesian bin-center displacements.
    TypeError
        If ``intervals`` is not an Intervals or a pynapple IntervalSet.
    ValueError
        If ``tolerance`` is not finite and non-negative, or ``times`` or
        ``positions`` are invalid.

    Notes
    -----
    The route is the visited bin centers in the x-y plane with repeats
    removed, so the answer depends on where the animal went, not on the
    frame rate or how long it dwelt in a bin. It is simplified with the
    Ramer-Douglas-Peucker algorithm [1]_: a leg is split at the bin farthest
    from it whenever that bin strays more than ``tolerance``. Each bin center
    lies within half a bin diagonal of the true position, so a straight path,
    and the chord between its quantized ends, both stay within half a
    diagonal of the true line; its bin centers therefore stray at most one
    diagonal from the chord, and the default never splits a straight run
    (such as the staircase of bins along an oblique line) or flicker between
    neighboring bins. The sign of the cross product of consecutive legs gives
    the turn direction.

    References
    ----------
    .. [1] Douglas, D. H., & Peucker, T. K. (1973). Algorithms for the
       reduction of the number of points required to represent a digitized
       line or its caricature. Cartographica 10(2), 112-122.

    Examples
    --------
    An L-shaped run, north then west, is one left turn:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior import graph_turn_sequence
    >>> x, y = np.meshgrid(np.arange(0.0, 50.0), np.arange(0.0, 50.0))
    >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
    >>> north = np.c_[np.full(20, 40.0), np.linspace(5, 40, 20)]
    >>> west = np.c_[np.linspace(38, 5, 20), np.full(20, 40.0)]
    >>> positions = np.r_[north, west]
    >>> graph_turn_sequence(env, np.arange(40) / 10, positions)
    'left'

    One sequence per trial: ``graph_turn_sequence(env, times, positions,
    trials)``.

    See Also
    --------
    compute_planar_turn_angles : Signed turn at each tracking sample, from
        continuous positions.
    compute_planar_curvature : Signed geometric curvature of the path.
    """
    require_cartesian(
        env.coordinate_system,
        "graph_turn_sequence()",
        why=(
            "A turn's direction is the sign of the cross product of successive "
            "Cartesian steps between bin centers."
        ),
        fix=(
            "classify turns on an allocentric environment built from the animal's "
            "positions, for example Environment.from_samples(positions, bin_size=...)"
        ),
    )
    times, positions, intervals = _validated_route_samples(
        env, times, positions, intervals, call="graph_turn_sequence"
    )
    bins = env.bin_at(positions)
    if tolerance is None:
        tolerance = float(np.sqrt(2.0) * env.bin_spacing)
    elif not (np.isfinite(tolerance) and tolerance >= 0):
        raise ValueError(
            _format_error(
                f"graph_turn_sequence() got tolerance={tolerance!r}.",
                why="tolerance is a distance in the environment's units.",
                fix="pass a finite tolerance >= 0, or None for one bin diagonal",
            )
        )
    if intervals is None:
        return _turn_sequence(env, bins, tolerance)
    import pandas as pd

    first = np.searchsorted(times, intervals.start, side="left")
    last = np.searchsorted(times, intervals.stop, side="left")
    return pd.Series(
        [
            _turn_sequence(env, bins[a:b], tolerance)
            for a, b in zip(first, last, strict=True)
        ],
        index=pd.Index(intervals.interval_ids, name="interval_id"),
        name="turn_sequence",
    )


def _turn_sequence(env: Environment, bins: NDArray[np.intp], tolerance: float) -> str:
    """:func:`graph_turn_sequence` of one route, from its samples' bins."""
    if env.bin_centers.shape[1] < 2:
        return ""
    points = env.bin_centers[bins[bins >= 0], :2]
    if len(points) == 0:
        return ""
    # One point per visit, whatever the frame rate or dwell time.
    route = points[np.r_[True, np.any(np.diff(points, axis=0) != 0, axis=1)]]
    corners = route[_simplified_polyline(route, tolerance)]
    if len(corners) < 3:
        return ""

    from neurospatial._angles import signed_turns

    legs = np.diff(corners, axis=0)
    angle = signed_turns(legs[:-1], legs[1:])
    # Counterclockwise (angle > 0) in a y-up frame is the animal's left.
    turns = angle[(angle != 0) & (np.abs(angle) <= _MAX_TURN_ANGLE)]
    return "-".join("left" if turn > 0 else "right" for turn in turns)


def _simplified_polyline(
    points: NDArray[np.float64], tolerance: float
) -> NDArray[np.intp]:
    """Indices of the Ramer-Douglas-Peucker simplification of a polyline.

    Parameters
    ----------
    points : ndarray, shape (n_points, 2)
        Polyline vertices in order.
    tolerance : float
        A segment is split at its farthest intermediate vertex when that
        vertex is more than ``tolerance`` from the segment.

    Returns
    -------
    ndarray of intp
        Sorted indices of the kept vertices, always including both ends.
    """
    n = len(points)
    keep = np.zeros(n, dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        start, chord = points[i], points[j] - points[i]
        offsets = points[i + 1 : j] - start
        length_sq = float(chord @ chord)
        # Distance to the segment (not the infinite line), so a reversal
        # past the segment's end is a corner too.
        along = (
            np.zeros(len(offsets))
            if length_sq == 0
            else np.clip(offsets @ chord / length_sq, 0.0, 1.0)
        )
        distance = np.linalg.norm(offsets - along[:, None] * chord, axis=1)
        k = int(np.argmax(distance))
        if distance[k] > tolerance:
            split = i + 1 + k
            keep[split] = True
            stack.extend([(i, split), (split, j)])
    return np.flatnonzero(keep)


@reads_pynapple_samples
def heading_direction_labels(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    min_speed: float,
    speed: ArrayLike | None = None,
    headings: ArrayLike | None = None,
    n_directions: int = 8,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.object_]:
    """Label each sample by the direction the animal moves in.

    Bins the direction of movement into sectors centred on evenly spaced
    allocentric directions (with the default 8: ``"0°"`` East, ``"45°"``,
    ``"90°"`` North, ..., ``"180°"`` West, ``"-135°"``, ``"-90°"`` South,
    ``"-45°"``), labels an observed animal that is not moving "stationary",
    and leaves samples with no observed movement unlabeled (``None``).

    To label headings you already have, such as head direction from
    :func:`heading_from_body_orientation`, use
    :func:`headings_to_direction_labels`.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps (seconds), finite and strictly increasing.
    positions : NDArray[np.float64], shape (n_samples, 2)
        2D position coordinates over time. A row with a non-finite coordinate
        is a missing position.
    min_speed : float
        Required, finite and ``>= 0``, in position units per second (e.g.
        ``5.0`` for 5 cm/s with positions in cm): this function sees no
        environment, so it cannot know the positions' unit. Observed samples
        slower than this, or not moving at all, are labeled "stationary".
    n_directions : int, default=8
        Number of direction sectors, a positive integer. Default 8 creates
        45° sectors centred on East, North-East, North, and so on.
    speed : ArrayLike, shape (n_samples,), optional
        Your own speed of each sample in position units per second (NaN
        where not observed), used instead of computing it from
        ``positions``: for example speed from your own position processing,
        or ``compute_speed(times, positions, smooth_window=0)`` for positions
        you already smoothed. Samples excluded by ``max_gap``, ``epochs`` or
        ``break_before`` stay excluded. Default: the speed of
        :func:`compute_speed` (from x and y, the horizontal speed).
    headings : ArrayLike, shape (n_samples,), optional
        Your own direction of each sample in radians (allocentric: 0 = East,
        +π/2 = North; NaN where not observed), used instead of the direction
        of movement computed from ``positions``: for example the direction of
        your own smoothed velocity, or head direction from a head-mounted LED
        or sensor. Samples slower than ``min_speed`` still have none. Default:
        the direction of movement of :func:`heading_from_velocity`.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    NDArray[np.object_], shape (n_samples,)
        One label per timepoint, of three kinds: a direction sector named by
        its centre direction, such as ``"90°"``, where the animal moves (speed
        above zero and at least ``min_speed``); ``"stationary"`` where it was
        observed but not moving; ``None`` where it was not observed.

    Raises
    ------
    ValueError
        If ``min_speed`` is not finite and non-negative, ``n_directions`` is
        not a positive integer, ``times`` and ``positions`` are empty or have
        different lengths, ``times`` is not finite and strictly increasing,
        ``positions`` has no x and y columns, or ``break_before`` is not a
        Boolean array with one entry per sample.

    See Also
    --------
    headings_to_direction_labels : The same sectors for headings you computed.
    neurospatial.Intervals.sample_values : Labels from trials or laps.
    labels_to_epochs : Turn the labels into ``[start, stop)`` windows.

    Notes
    -----
    Sector ``k`` is centred on ``k * 360 / n_directions`` degrees
    (allocentric: 0° = East, 90° = North) and spans half a sector width on
    either side, half-open: its clockwise edge belongs to it. Centring the
    sectors on the cardinal directions keeps a run along a cardinal
    direction in one label despite tracking jitter; edges on 0° would split
    an eastward run between two labels. Labels are the centre angles in
    ``(-180, 180]``, rounded to 0.1°.

    The speed and heading are computed along each run of observed samples. A
    NaN row is bridged like a deleted row when valid neighbours are within
    ``max_gap``; only a longer gap, an ``epochs`` edge or a declared
    ``break_before`` splits a run, and no velocity or heading spans such a
    split.

    Speed and heading come from the Gaussian-smoothed velocity of
    :func:`compute_speed`, so tracking jitter at rest is "stationary". The
    direction of movement exists only where the animal moves, the rule of
    :func:`heading_from_velocity`. A sample in no observed run (a missing
    position, an isolated sample, an excluded interval) is ``None``, "not
    observed", which :func:`labels_to_epochs` leaves out.

    Passing these labels straight to
    :func:`~neurospatial.encoding.compute_directional_place_fields` gives a
    "stationary" map as well, because ``"stationary"`` is an ordinary label
    there; ``missing_label="stationary"`` drops it. ``None`` samples are
    always left out.

    Examples
    --------
    Label one second of running north-east (about 63 deg, 22 cm/s), then one
    second resting in place. One frame was lost:

    >>> import numpy as np
    >>> from neurospatial.behavior import heading_direction_labels
    >>> times = np.arange(20) / 10
    >>> step = np.r_[np.arange(10.0), np.full(10, 9.0)]
    >>> positions = np.column_stack([step, 2 * step])
    >>> positions[5] = np.nan  # a lost frame
    >>> labels = heading_direction_labels(times, positions, min_speed=5.0)
    >>> labels[0], labels[5], labels[-1]
    ('45°', None, 'stationary')

    ``"stationary"`` is an ordinary label for the place-field helper, which
    drops it only when told to:

    >>> from neurospatial import Environment
    >>> from neurospatial.encoding import compute_directional_place_fields
    >>> env = Environment.from_samples(positions, bin_size=4.0)
    >>> fields = compute_directional_place_fields(
    ...     env,
    ...     np.array([0.5, 1.5]),
    ...     times,
    ...     positions,
    ...     labels,
    ...     method="binned",
    ...     max_gap=None,
    ... )
    >>> sorted(fields.labels)
    ['45°', 'stationary']
    >>> fields = compute_directional_place_fields(
    ...     env,
    ...     np.array([0.5, 1.5]),
    ...     times,
    ...     positions,
    ...     labels,
    ...     method="binned",
    ...     max_gap=None,
    ...     missing_label="stationary",
    ... )
    >>> sorted(fields.labels)
    ['45°']
    """
    from neurospatial._observation import validated_observations
    from neurospatial._velocity import DEFAULT_SPEED_SMOOTHING
    from neurospatial.behavior._kinematics import (
        _movement_heading_and_speed,
        validated_min_speed,
        validated_motion,
    )

    call = "heading_direction_labels"
    min_speed = validated_min_speed(min_speed, call=call)
    n_directions = _validate_n_directions(n_directions, call=call)
    times_arr, positions_arr, support = validated_observations(
        times,
        positions,
        call=call,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="every label is None",
        columns="xy",
    )
    gated, speeds = _movement_heading_and_speed(
        times_arr,
        positions_arr,
        support.interval_mask,
        smooth_window=DEFAULT_SPEED_SMOOTHING,
        min_speed=min_speed,
        speed=validated_motion(
            speed, name="speed", n_samples=len(times_arr), call=call
        ),
        headings=validated_motion(
            headings, name="headings", n_samples=len(times_arr), call=call
        ),
    )
    return _direction_sector_labels(
        gated,
        speeds,
        support.run_ids >= 0,
        min_speed=min_speed,
        n_directions=n_directions,
    )


@reads_pynapple_samples
def headings_to_direction_labels(
    headings: ArrayLike,
    speeds: ArrayLike,
    *,
    min_speed: float,
    n_directions: int = 8,
) -> NDArray[np.object_]:
    """Label each sample by a heading you already computed.

    Bins each heading into the sectors of :func:`heading_direction_labels`
    (with the default 8: ``"0°"`` East, ``"45°"``, ``"90°"`` North, ...,
    ``"180°"`` West, ``"-135°"``, ``"-90°"`` South, ``"-45°"``) wherever the
    animal moves at ``min_speed`` or faster, labels an observed animal that is
    slower "stationary", and leaves samples whose speed was not observed
    unlabeled (``None``). Use it to label head direction from
    :func:`heading_from_body_orientation` or a head-direction sensor by running
    speed, or a heading and speed you computed with your own smoothing.

    Parameters
    ----------
    headings : ArrayLike, shape (n_samples,)
        Allocentric heading of each sample in radians (0 = East, +π/2 =
        North); any real value, wrapped or unwrapped. NaN where the heading is
        unknown.
    speeds : ArrayLike, shape (n_samples,)
        Speed of each sample, finite and ``>= 0``, in the units of
        ``min_speed``. NaN marks a sample that was not observed.
    min_speed : float
        Required, finite and ``>= 0``, in the units of ``speeds`` (e.g.
        ``5.0`` for 5 cm/s). Observed samples slower than this, or not moving
        at all, are labeled "stationary".
    n_directions : int, default=8
        Number of direction sectors, a positive integer. Default 8 creates
        45° sectors centred on East, North-East, North, and so on.

    Returns
    -------
    NDArray[np.object_], shape (n_samples,)
        One label per sample, of three kinds: a direction sector named by its
        centre direction, such as ``"90°"``, where the animal moves (speed
        above zero and at least ``min_speed``) with a known heading;
        ``"stationary"`` where it was observed but not moving; ``None`` where
        the speed is NaN (not observed), or the animal moves with a NaN
        heading.

    Raises
    ------
    ValueError
        If ``min_speed`` is not finite and non-negative, ``n_directions`` is
        not a positive integer, ``headings`` or ``speeds`` is not 1-D, they
        differ in length, a heading is infinite, or a speed is negative or
        infinite.

    Warns
    -----
    UserWarning
        If ``headings`` look like degrees instead of radians.

    See Also
    --------
    heading_direction_labels : Labels from the direction of movement, computed
        from times and positions with time support.
    heading_from_body_orientation : Head direction from two tracked body parts.
    heading_from_velocity : Direction of movement from positions.
    compute_speed : Smoothed speed from positions.

    Notes
    -----
    The sectors are those of :func:`heading_direction_labels`: sector ``k`` is
    centred on ``k * 360 / n_directions`` degrees and half-open, with its
    clockwise edge belonging to it.

    The headings and speeds are labeled as given, sample by sample: there is
    no time support (``max_gap``, ``epochs``, ``break_before``) and no
    smoothing. Compute them with the time support you need, for example
    ``heading_from_body_orientation(times, nose, tail, max_gap=...)`` and
    ``compute_speed(times, positions, max_gap=...)``.

    Examples
    --------
    A NaN speed was not observed; a slow sample is "stationary" whatever its
    heading:

    >>> import numpy as np
    >>> from neurospatial.behavior import headings_to_direction_labels
    >>> headings = np.array([0.3, 0.3, np.pi / 2, 0.3, np.nan])  # radians
    >>> speeds = np.array([10.0, 1.0, 12.0, np.nan, 10.0])  # cm/s
    >>> headings_to_direction_labels(headings, speeds, min_speed=5.0).tolist()
    ['0°', 'stationary', '90°', None, None]

    Label where the head points while the animal runs. It runs East at
    10 cm/s facing North:

    >>> from neurospatial.behavior import (
    ...     compute_speed,
    ...     heading_from_body_orientation,
    ... )
    >>> times = np.arange(20) / 10
    >>> tail = np.column_stack([np.arange(20.0), np.zeros(20)])
    >>> nose = tail + [0.0, 3.0]
    >>> labels = headings_to_direction_labels(
    ...     heading_from_body_orientation(times, nose, tail),
    ...     compute_speed(times, tail),
    ...     min_speed=5.0,
    ... )
    >>> sorted(set(labels.tolist()))
    ['90°']
    """
    from neurospatial._validation import refuse_infinite
    from neurospatial.behavior._kinematics import validated_min_speed
    from neurospatial.encoding._validation import warn_if_angles_look_like_degrees

    call = "headings_to_direction_labels"
    min_speed = validated_min_speed(min_speed, call=call)
    n_directions = _validate_n_directions(n_directions, call=call)
    headings_arr = np.asarray(headings, dtype=np.float64)
    speeds_arr = np.asarray(speeds, dtype=np.float64)
    for name, values in (("headings", headings_arr), ("speeds", speeds_arr)):
        if values.ndim != 1:
            raise ValueError(
                _format_error(
                    f"{call}() needs 1-D {name}, one value per sample; got "
                    f"shape {values.shape}.",
                    fix=f"pass {name} with shape (n_samples,), e.g. "
                    f"np.ravel({name}) for a column vector",
                )
            )
    if len(headings_arr) != len(speeds_arr):
        raise ValueError(
            _format_error(
                f"{call}() got {len(headings_arr)} headings and "
                f"{len(speeds_arr)} speeds.",
                why="Each sample needs a heading and a speed.",
                fix="pass headings and speeds for the same samples",
            )
        )
    refuse_infinite(headings_arr, name="headings", call=call)
    with np.errstate(invalid="ignore"):
        bad_speed = np.isinf(speeds_arr) | (speeds_arr < 0)
    if bad_speed.any():
        first = int(np.flatnonzero(bad_speed)[0])
        raise ValueError(
            _format_error(
                f"{call}() got speeds[{first}] = {float(speeds_arr[first])}.",
                why="A speed is a finite, non-negative magnitude; NaN marks a "
                "sample that was not observed.",
                fix="pass speeds such as compute_speed(times, positions), with "
                "NaN where the animal was not observed; the order is "
                f"{call}(headings, speeds, min_speed=...)",
            )
        )
    warn_if_angles_look_like_degrees(headings_arr, name="headings", context=call)
    return _direction_sector_labels(
        headings_arr,
        speeds_arr,
        np.isfinite(speeds_arr),
        min_speed=min_speed,
        n_directions=n_directions,
    )


def _validate_n_directions(n_directions: object, *, call: str) -> int:
    """Return ``n_directions`` as an int, raising unless it is a positive integer."""
    from neurospatial._validation import validate_integer_at_least

    return validate_integer_at_least(
        n_directions,
        1,
        name="n_directions",
        owner=f"{call}()",
        hint="it is the number of direction sectors, e.g. 8 for 45° sectors",
    )


def _direction_sector_labels(
    headings: NDArray[np.float64],
    speeds: NDArray[np.float64],
    observed: NDArray[np.bool_],
    *,
    min_speed: float,
    n_directions: int,
) -> NDArray[np.object_]:
    """Sector, "stationary" or None label for each sample.

    Parameters
    ----------
    headings : ndarray, shape (n_samples,)
        Radians; NaN where the direction is unknown.
    speeds : ndarray, shape (n_samples,)
        Speed of each sample; read only where ``observed``.
    observed : ndarray of bool, shape (n_samples,)
        Samples whose movement was observed; the others are None.
    min_speed : float
        Validated movement gate.
    n_directions : int
        Validated number of sectors.

    Returns
    -------
    ndarray of object, shape (n_samples,)
    """
    # Sector k is centred on k * width and spans [centre - width/2,
    # centre + width/2), so a run along a cardinal direction stays in one
    # sector despite jitter.
    width = 2 * np.pi / n_directions
    centres_deg = np.degrees(np.arange(n_directions) * width)
    centres_deg = np.where(centres_deg > 180.0, centres_deg - 360.0, centres_deg)
    bin_labels_arr = np.array(
        [f"{round(float(c), 1):g}°" for c in centres_deg], dtype=object
    )
    with np.errstate(invalid="ignore"):
        sector = np.floor(headings / width + 0.5)
    bin_indices = np.mod(np.nan_to_num(sector), n_directions).astype(np.intp)

    # The direction of movement exists only where the animal moves. An observed
    # animal that does not move is "stationary"; an unobserved sample, or a
    # moving one whose direction is unknown, stays None.
    moving = observed & (speeds > 0) & (speeds >= min_speed)
    direction_known = moving & np.isfinite(headings)
    labels = np.full(len(speeds), None, dtype=object)
    labels[observed & ~moving] = "stationary"
    labels[direction_known] = bin_labels_arr[bin_indices[direction_known]]
    return labels


# =============================================================================
# Path Efficiency Functions
# =============================================================================


def _observed_path_length(
    positions: NDArray[np.float64],
    sample_valid: NDArray[np.bool_],
    interval_mask: NDArray[np.bool_],
    *,
    paths: EnvironmentPaths | None,
) -> tuple[float, bool]:
    """Path length over usable intervals, and whether every interval counted.

    Parameters
    ----------
    positions : ndarray, shape (n_samples, n_dims)
    sample_valid : ndarray of bool, shape (n_samples,)
        ``ObservationSupport.sample_valid`` of the same samples.
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        ``ObservationSupport.interval_mask`` of the same samples.
    paths : EnvironmentPaths or None
        Geodesic engine (``metric_paths``), or None for straight steps.

    Returns
    -------
    length : float
        Sum of the counted step lengths between consecutive valid samples
        (missing samples are skipped as if deleted); ``nan`` when none
        counted.
    complete : bool
        True when there is at least one interval between the first and last
        valid samples, every such interval is usable and every step counted.
        Missing samples before the first or after the last valid sample are
        left out, as if their rows were deleted.
    """
    # One step per usable gap, from the valid sample before it to the one
    # ending it; only usable steps are measured.
    starts, ends = _observed_steps(positions, sample_valid, interval_mask)
    steps = (
        np.linalg.norm(ends - starts, axis=1)
        if paths is None
        else paths.between(starts, ends)
    )
    counted = ~np.isnan(steps)
    observed = np.flatnonzero(sample_valid)
    span = (
        interval_mask[observed[0] : observed[-1]]
        if observed.size
        else interval_mask[:0]
    )
    complete = bool(span.size and span.all() and counted.all())
    if not counted.any():
        return float("nan"), complete
    return float(steps[counted].sum()), complete


@reads_pynapple_samples
def traveled_path_length(
    times: ArrayLike,
    positions: ArrayLike,
    *,
    metric: DistanceMetric = "euclidean",
    env: Environment | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> float:
    """Compute the distance traveled over observed intervals.

    Distance is added between consecutive valid samples (finite
    coordinates) when the time between them is at most ``max_gap``, they fall
    inside one of ``epochs`` and no break is declared between them; missing
    samples in between are skipped as if their rows were deleted. Distance
    between recordings is never counted, and a longer gap splits the path
    instead of turning the total into NaN. A break must be declared by a time gap above
    ``max_gap``, by ``break_before`` or by ``epochs``; an undeclared jump on a
    continuous clock is counted as movement.

    Parameters
    ----------
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples, n_dims) or (n_samples,)
        Position coordinates in allocentric frame. A 1-D array is one
        coordinate (for example a linearized track). Rows with a non-finite
        coordinate are missing.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        ``"euclidean"`` sums the straight steps between samples. ``"geodesic"``
        sums, between consecutive samples, the length of the shortest path
        between two positions that stays inside the environment: it goes around
        walls and never leaves the environment, and on a track it runs along
        the track's centerline. It is the straight step whenever the step stays
        inside, and the route around the walls when it does not (a bridged gap
        whose straight line crosses a wall). See Notes.
    env : Environment, optional
        Required when metric="geodesic". An interval with an endpoint outside
        the environment has no geodesic length and is left out; an interval
        between disconnected parts of the environment contributes ``inf``.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : None, (start, stop) or array-like of shape (n_epochs, 2), optional
        Half-open ``[start, stop)`` windows in seconds, on the clock of
        ``times``. An interval must lie inside one window.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    float
        Total path length in the units of ``positions``. ``nan`` when no
        interval was observed (including a single sample); ``0.0`` only for
        observed stationary travel.

    Raises
    ------
    ValueError
        If ``metric`` is not recognized, if metric="geodesic" but env is None,
        or if the inputs are empty or malformed.

    Notes
    -----
    The traveled length is the arc length of the observed path: the animal
    moves continuously, so between two samples it covers at least the
    shortest path between them inside the environment. With the geodesic
    metric that is the straight step when the step stays inside the
    environment (between nearby samples it almost always does), so the
    geodesic and Euclidean lengths agree on an open field and do not depend
    on the bin size. It is not measured bin center to bin center along the
    bin graph: that staircase overestimated an oblique straight run by up to
    a third, by an amount that changed with the bin size and sampling rate.
    On a track the geodesic step runs along the centerline between the two
    positions' projections onto it.

    Like any arc length of sampled positions, tracking jitter adds length;
    smooth noisy positions first.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import traveled_path_length
    >>> times = np.array([0.0, 0.1, 0.2, 0.3])
    >>> positions = np.array([[0.0, 0.0], [1.0, 0.0], [50.0, 0.0], [51.0, 0.0]])
    >>> traveled_path_length(
    ...     times, positions, break_before=np.array([False, False, True, False])
    ... )
    2.0
    >>> traveled_path_length(np.array([0.0, 0.1, 100.0, 100.1]), positions)
    2.0
    """
    from neurospatial._observation import validated_observations

    paths = metric_paths(metric, env, call="traveled_path_length")

    _, positions_arr, support = validated_observations(
        times,
        positions,
        call="traveled_path_length",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="no distance was observed",
    )
    length, _ = _observed_path_length(
        positions_arr,
        support.sample_valid,
        support.interval_mask,
        paths=paths,
    )
    return length


def shortest_path_length(
    env: Environment,
    start: NDArray[np.float64],
    goal: NDArray[np.float64],
    *,
    metric: DistanceMetric = "geodesic",
) -> float:
    """Compute shortest path distance from start to goal.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    start : NDArray[np.float64], shape (n_dims,)
        Start position in allocentric frame.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position in allocentric frame.
    metric : {"geodesic", "euclidean"}, default="geodesic"
        ``"euclidean"`` is the straight-line distance. ``"geodesic"`` is the
        length of the shortest path between two positions that stays inside the
        environment: it goes around walls and never leaves the environment, and
        on a track it runs along the track's centerline (see Notes).

    Returns
    -------
    float
        Shortest path length in environment units; ``inf`` if no path exists
        (start and goal lie in disconnected parts of the environment).

    Raises
    ------
    ValueError
        If ``metric`` is not recognized, ``start`` is not one position, or
        ``goal`` does not have the same shape as ``start`` or has a non-finite
        coordinate. For ``metric="geodesic"``, also if ``start`` or ``goal``
        is outside the environment or ``start`` has NaN coordinates.
    CartesianOnlyError
        For ``metric="geodesic"`` on an egocentric polar environment, whose
        bins are (distance, angle), not allocentric positions.

    Notes
    -----
    The geodesic length is measured in continuous space, not bin center to
    bin center. When the centers of the start's and the goal's bins see each
    other (a straight line between them stays inside the environment), it is
    the straight line from start to goal, so on an open field
    ``metric="geodesic"`` and ``metric="euclidean"`` agree. Otherwise the
    path goes around the walls through waypoints at bin centers, so it keeps
    about half a bin from each corner and can exceed the true shortest path
    by up to about one bin size per corner (the end of a thin wall is two
    corners). On a track it is the distance along the centerline between the
    two positions' projections, whatever the track's width or bin size. The
    bin-graph distance (``env.distance_between``) instead runs along grid
    steps and overestimates an oblique straight line by up to ~8% on an
    8-connected grid.

    Examples
    --------
    An open field: the straight line from start to goal stays inside it.

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior import shortest_path_length
    >>> x, y = np.meshgrid(np.arange(0.5, 100.0), np.arange(0.5, 100.0))
    >>> arena = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
    >>> shortest_path_length(arena, np.array([10.0, 10.0]), np.array([40.0, 50.0]))
    50.0
    """
    paths = metric_paths(metric, env, call="shortest_path_length")
    start = np.asarray(start, dtype=float)
    if start.ndim != 1:
        raise ValueError(
            _format_error(
                f"shortest_path_length() needs start with shape (n_dims,); got "
                f"shape {start.shape}.",
                why="start is one position.",
                fix="pass a single position, e.g. positions[0].",
            )
        )
    goal = _validated_goal(goal, start.size, call="shortest_path_length")

    if paths is None:
        return float(np.linalg.norm(goal - start))

    _require_inside(paths, {"start": start, "goal": goal}, call="shortest_path_length")
    return float(paths.to_point(start[None, :], goal)[0])


def _require_inside(
    paths: Any, points: dict[str, NDArray[np.float64]], *, call: str
) -> None:
    """Raise unless every named point lies inside the environment.

    Parameters
    ----------
    paths : EnvironmentPaths
    points : dict of str to ndarray, shape (n_dims,)
        Named positions, such as ``{"goal": goal}``.
    call : str
        Public function name, used in the message.
    """
    for name, point in points.items():
        if paths.bins(np.asarray(point, dtype=float)[None, :])[0] < 0:
            raise ValueError(
                _format_error(
                    f"{call}() got a {name} outside the environment (or with a "
                    f"NaN coordinate): {point}.",
                    why=(
                        "a geodesic distance is a path inside the environment, "
                        "so both of its ends must lie in a bin."
                    ),
                    fix=(
                        f"pass a {name} inside the environment (check "
                        "env.contains), or use metric='euclidean'."
                    ),
                )
            )


@reads_pynapple_samples
def time_efficiency(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    reference_speed: float,
    optimal_distance: float,
) -> float:
    """Compute time efficiency: ratio of optimal to actual travel time.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Trajectory positions.
    reference_speed : float
        Reference speed in position units per second; finite and greater
        than zero.
    optimal_distance : float
        Shortest-path distance from start to goal under the caller's chosen
        metric (Euclidean or geodesic). The optimal time is
        ``optimal_distance / reference_speed``. Pass ``np.inf`` for an
        unreachable goal; the result is then ``nan``.

    Returns
    -------
    float
        Time efficiency ratio ``T_optimal / T_actual``. ``nan`` if the
        trajectory has < 2 samples, ``actual_time <= 0``, or
        ``optimal_distance`` is non-finite.

    Raises
    ------
    ValueError
        If ``reference_speed`` is not a finite speed greater than zero, or
        the inputs are malformed.

    Examples
    --------
    >>> eff = time_efficiency(
    ...     times, positions, reference_speed=20.0, optimal_distance=50.0
    ... )  # doctest: +SKIP
    >>> print(f"Time efficiency: {eff:.1%}")  # doctest: +SKIP
    """
    from neurospatial._validation import validate_times_positions

    _validate_reference_speed(reference_speed, call="time_efficiency")
    times, positions = validate_times_positions(
        times, positions, call="time_efficiency"
    )
    if len(positions) < 2:
        return np.nan

    if not np.isfinite(optimal_distance):
        return np.nan

    actual_time = times[-1] - times[0]
    if actual_time <= 0:
        return np.nan

    optimal_time = optimal_distance / reference_speed

    return float(optimal_time / actual_time)


def _goal_observations(
    env: Environment,
    times: ArrayLike,
    positions: ArrayLike,
    intervals: Any,
    *,
    call: str,
    max_gap: float | None,
    epochs: Any,
    break_before: ArrayLike | None,
    result: str,
    columns: Literal["xy"] | None = None,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], Intervals[Any] | None, ObservationSupport
]:
    """Validate a goal function's samples and resolve their observation support.

    Tracking gaps, ``epochs`` and declared breaks are resolved once, on the
    whole recording, so a warning that nothing was observed is given once.
    """
    from neurospatial._interval_table import require_intervals
    from neurospatial._observation import validated_observations

    require_cartesian(
        env.coordinate_system,
        f"{call}()",
        why="It reads positions as the animal's allocentric positions, located "
        "in the environment's bins.",
        fix="use the allocentric environment the positions were recorded in",
    )
    if intervals is not None:
        intervals = require_intervals(intervals, name="intervals", owner=f"{call}()")
    t, p, support = validated_observations(
        times,
        positions,
        call=call,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result=result,
        columns=columns,
    )
    if p.shape[1] != env.n_dims:
        msg = (
            f"{call}(): positions have {p.shape[1]} coordinate(s) per sample; "
            f"the environment has {env.n_dims}."
        )
        raise ValueError(msg)
    return t, p, intervals, support


def _support_of(
    times: NDArray[np.float64],
    support: ObservationSupport,
    samples: slice,
    intervals: Intervals[Any] | None,
) -> ObservationSupport:
    """The support of one goal row: the whole recording's, or one interval's."""
    if intervals is None:
        return support
    from neurospatial.behavior._goals import row_support

    return row_support(times, support, samples)


@reads_pynapple_samples
def angular_efficiency(
    env: Environment[CartesianSystem],
    times: ArrayLike,
    positions: ArrayLike,
    intervals: Intervals[Any] | None = None,
    *,
    goal: Any,
    metric: DistanceMetric = "geodesic",
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> float | pd.Series:
    """Compute how directly the path heads toward the goal.

    For every observed step, the angle between the step's direction and the
    direction in which the shortest path from the step's start to the goal
    leaves it is measured; the result is
    ``1 - sum(length * angle) / (pi * sum(length))``, the mean angle per unit
    of distance traveled. A run straight toward the goal scores 1, a run
    straight away scores 0, and a run perpendicular to the goal direction
    scores 0.5.

    Parameters
    ----------
    env : Environment
        The environment the positions were recorded in.
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples, n_dims)
        Positions in the environment's coordinates. A NaN row is a missing
        sample; a step starting outside the environment has no goal
        direction and is left out.
    intervals : Intervals or IntervalSet, optional
        Trials or other intervals. Each is measured from its own samples,
        ``[start, stop)``, as if it were the whole recording, and the result
        has one value per interval. A pynapple ``IntervalSet`` is read as
        :func:`~neurospatial.io.from_pynapple` reads it.
    goal : str, array-like of shape (n_dims,), or per-interval sequence
        A region name, a point in environment coordinates, or, with
        ``intervals``, one per interval (a ``pandas.Series`` indexed by
        ``interval_id`` such as ``trials.end_region``; ``None`` for an unknown
        goal), read as in :func:`distance_to_goal`. A polygon region is
        approached at its nearest bin: the goal direction is the direction of
        the shortest path to the nearest bin of the region, the direction in
        which the distance to the goal falls fastest. For a round well
        (``regions.add(name, point=..., radius=...)``) that is the direction
        to its centre. A point region or a point is aimed at itself.
    metric : {"geodesic", "euclidean"}, default="geodesic"
        ``"geodesic"``: the shortest path that stays inside the environment
        (around walls, along a track's centerline), measured from each
        position; the goal direction is the direction in which that path
        leaves the position, toward its first corner. ``"euclidean"``: the
        straight line. In an arena without walls between the animal and the
        goal they agree. A geodesic goal point must lie inside the
        environment.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    float or pandas.Series
        Angular efficiency in ``[0, 1]``; with ``intervals``, one per interval
        indexed by ``interval_id``. NaN when no step has a defined direction
        toward the goal: fewer than two observed samples, the animal never
        moves, every step starts at the goal, or the goal is unknown.

    Raises
    ------
    ValueError
        If the inputs are empty or malformed (``times`` not finite and
        strictly increasing, positions without one coordinate per environment
        dimension, ``break_before`` not a Boolean array with one entry per
        sample), ``metric`` is not recognized, a region covers no bin, the
        goal point is malformed or (geodesic) outside the environment, or
        per-sample intervals overlap.
    TypeError
        If ``intervals`` is not an Intervals or a pynapple IntervalSet, or a
        per-interval goal is a number (a well number) instead of a region
        name or a point.
    RegionNotFoundError
        If a region name is not in ``env.regions``.
    CartesianOnlyError
        If ``env`` is an egocentric polar environment.

    Notes
    -----
    A step joins two consecutive valid samples. A NaN row is bridged like a
    deleted row when its valid neighbours are no more than ``max_gap`` apart;
    only a longer gap, an ``epochs`` edge or a declared break leaves the step
    out. A step of zero length has no direction and is left out, as is a
    step that starts at the goal (a goal bin, or the goal point).

    Steps are measured on positions smoothed by a Gaussian in time (0.1 s
    standard deviation, the smoothing of :func:`compute_speed`, weighted by
    time difference so it never reaches across a bridged gap much longer
    than that), and each step's angle counts in proportion to its smoothed
    length. The result does not depend on the sampling rate along a smooth
    path, and tracking jitter while the animal pauses, which smoothing
    averages to almost no movement, weighs almost nothing: with raw steps,
    4 s of 0.3 cm jitter after a 60 cm run toward the goal added as much
    path as the run and pulled the result from 1 to 0.75. Smoothing rounds
    sharp corners over about 0.1 s of travel.

    The angle between step ``s`` and goal direction ``g`` is computed as
    ``2 * arctan2(|u - v|, |u + v|)`` with unit vectors ``u = s / |s|`` and
    ``v = g / |g|``. In 2-D it equals ``|wrap(step_heading - goal_bearing)|``,
    and it is defined the same way for 1-D and 3-D positions.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior import angular_efficiency
    >>> corridor = np.column_stack([np.linspace(-20, 20, 81), np.zeros(81)])
    >>> env = Environment.from_samples(corridor, bin_size=1.0)
    >>> times = np.array([0.0, 0.1, 0.2])
    >>> positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    >>> angular_efficiency(env, times, positions, goal=[10.0, 0.0])  # toward
    1.0
    >>> angular_efficiency(env, times, positions, goal=[-10.0, 0.0])  # away
    0.0
    """
    from neurospatial.behavior._goals import by_interval, goal_rows

    call = "angular_efficiency"
    paths = metric_paths(metric, env, call=call)
    times_arr, positions_arr, intervals, support = _goal_observations(
        env,
        times,
        positions,
        intervals,
        call=call,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="the angular efficiency is NaN",
    )
    rows = goal_rows(
        env, times_arr, intervals, goal, paths=paths, call=call, disjoint=False
    )
    values = []
    for row in rows:
        if row.goal is None:
            values.append(np.nan)
            continue
        sl = row.samples
        rs = _support_of(times_arr, support, sl, intervals)
        starts, steps = _smoothed_steps(
            times_arr[sl], positions_arr[sl], rs.sample_valid, rs.interval_mask
        )
        values.append(_angular_efficiency(env, starts, steps, row.goal, paths=paths))
    if intervals is None:
        return values[0]
    return by_interval(rows, values, name="angular_efficiency")


def _smoothed_steps(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    sample_valid: NDArray[np.bool_],
    interval_mask: NDArray[np.bool_],
    smoothed: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Start and displacement of each usable step of the smoothed path.

    A step joins consecutive valid samples (a bridged stretch of missing
    samples is one step), measured on positions smoothed over time by
    :func:`~neurospatial._velocity.time_smoothed_positions`.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
    positions : ndarray, shape (n_samples, n_dims)
    sample_valid, interval_mask : ndarray of bool
        ``ObservationSupport.sample_valid`` (n_samples,) and
        ``ObservationSupport.interval_mask`` (n_samples - 1,).
    smoothed : ndarray, shape (n_samples, n_dims), optional
        Smoothed positions, when the caller already has them for a longer
        recording that these samples are a slice of.

    Returns
    -------
    starts, steps : ndarray, shape (n_steps, n_dims)
    """
    if smoothed is None:
        from neurospatial._velocity import time_smoothed_positions

        smoothed = time_smoothed_positions(times, positions, interval_mask)
    starts, ends = _observed_steps(smoothed, sample_valid, interval_mask)
    return starts, ends - starts


def _observed_steps(
    positions: NDArray[np.float64],
    sample_valid: NDArray[np.bool_],
    interval_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Start and end positions of each usable step between valid samples.

    A bridged stretch of missing samples is one step, from the valid sample
    before it to the valid sample after it.

    Parameters
    ----------
    positions : ndarray, shape (n_samples, n_dims)
    sample_valid, interval_mask : ndarray of bool
        ``ObservationSupport.sample_valid`` (n_samples,) and
        ``ObservationSupport.interval_mask`` (n_samples - 1,).

    Returns
    -------
    starts, ends : ndarray, shape (n_steps, n_dims)
    """
    from neurospatial.behavior._kinematics import observed_steps

    first, last, counted = observed_steps(sample_valid, interval_mask)
    return positions[first[counted]], positions[last[counted]]


def _angular_efficiency(
    env: Environment,
    starts: NDArray[np.float64],
    step: NDArray[np.float64],
    goal: Goal,
    *,
    paths: EnvironmentPaths | None,
) -> float:
    """``1 - sum(length * angle) / (pi * sum(length))`` over steps with an angle.

    Parameters
    ----------
    env : Environment
    starts, step : ndarray, shape (n_steps, n_dims)
        Step start positions and (smoothed) displacements.
    goal : Goal
        The resolved goal; its direction at each step start is the
        direction in which the shortest path to it leaves the start.
    paths : EnvironmentPaths or None
        Geodesic engine (``metric_paths``), or None for straight lines.
    """
    to_goal = goal.directions(env, paths, starts)
    step_norm = np.linalg.norm(step, axis=1)
    goal_norm = np.linalg.norm(to_goal, axis=1)
    with np.errstate(invalid="ignore"):
        defined = (step_norm > 0) & (goal_norm > 0)
    if not defined.any():
        return np.nan
    from neurospatial._angles import unit_vector_angles

    angle = unit_vector_angles(
        step[defined] / step_norm[defined, None],
        to_goal[defined] / goal_norm[defined, None],
    )
    # Mean angle per unit of distance traveled.
    weight = step_norm[defined]
    return float(1.0 - np.sum(weight * angle) / (np.pi * np.sum(weight)))


@reads_pynapple_samples
def subgoal_efficiency(
    env: Environment,
    times: ArrayLike,
    positions: ArrayLike,
    subgoals: ArrayLike,
    *,
    metric: DistanceMetric = "geodesic",
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> SubgoalEfficiencyResult:
    """Compute path efficiency decomposed by subgoals.

    The path is cut at the closest approach to each subgoal in turn, and each
    piece is scored like :func:`compute_path_efficiency`: segment 1 runs from
    the first measured sample to the closest approach to subgoal 1, segment
    2 from there to the closest approach to subgoal 2 after it, and so on.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples, n_dims) or (n_samples,)
        Position coordinates in allocentric frame. Rows with a non-finite
        coordinate are missing.
    subgoals : ArrayLike, shape (n_subgoals, n_dims) or (n_dims,)
        Ordered sequence of subgoals, in the coordinates and units of
        ``positions``. A single row is one subgoal.
    metric : {"geodesic", "euclidean"}, default="geodesic"
        Distance metric for path lengths, the closest approach and the goal
        direction, as in :func:`compute_path_efficiency`. ``"euclidean"`` is
        the straight-line distance. ``"geodesic"`` is the length of the
        shortest path between two positions that stays inside the environment:
        it goes around walls and never leaves the environment, and on a track
        it runs along the track's centerline.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : None, (start, stop) or array-like of shape (n_epochs, 2), optional
        Half-open ``[start, stop)`` windows in seconds, on the clock of
        ``times``. Only samples inside a window are searched for the closest
        approach, and a step must lie inside one window.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    SubgoalEfficiencyResult
        One :class:`PathEfficiencyResult` per subgoal and the derived mean
        and weighted efficiencies. A segment's efficiency is NaN unless it is
        fully observed; a segment of a single sample (the subgoal was never
        approached) has NaN traveled length, efficiency and angular
        efficiency, and its ``shortest_length`` is still the distance from
        its start to the subgoal.

    Raises
    ------
    ValueError
        If the inputs are empty or malformed, ``metric`` is not recognized,
        ``subgoals`` does not have one finite coordinate per position
        dimension, or (geodesic) a subgoal or the first measured position is
        outside the environment.

    Warns
    -----
    UserWarning
        When a subgoal is never approached: the closest measured sample after
        the previous subgoal is the segment's own start (and is not at the
        subgoal), so the animal never got closer to it.

    Notes
    -----
    A NaN row is bridged like a deleted row when its valid neighbours are no
    more than ``max_gap`` apart; only a longer gap, an ``epochs`` edge or a
    declared break splits the path, and a segment containing such a split is
    not fully observed. The closest approach is searched among measured
    samples only: finite coordinates, inside ``epochs`` and (geodesic) inside
    the environment.

    Examples
    --------
    A straight run East through a subgoal at x = 5 cm to the goal at 10 cm:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior import subgoal_efficiency
    >>> positions = np.c_[np.arange(11.0), np.zeros(11)]
    >>> times = np.arange(11) / 10
    >>> env = Environment.from_samples(positions, bin_size=1.0)
    >>> subgoals = np.array([[5.0, 0.0], [10.0, 0.0]])
    >>> result = subgoal_efficiency(env, times, positions, subgoals, metric="euclidean")
    >>> [seg.traveled_length for seg in result.segment_results]
    [5.0, 5.0]
    >>> result.mean_efficiency
    1.0
    """
    import warnings

    from neurospatial._observation import validated_observations
    from neurospatial._stacklevel import _external_stacklevel

    call = "subgoal_efficiency"
    paths = metric_paths(metric, env, call=call)
    times_arr, positions_arr, support = validated_observations(
        times,
        positions,
        call=call,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="no path was observed",
    )
    from neurospatial._velocity import time_smoothed_positions
    from neurospatial.behavior._goals import resolve_goal

    smoothed = time_smoothed_positions(times_arr, positions_arr, support.interval_mask)
    n_dims = positions_arr.shape[1]
    subgoal_arr = np.asarray(subgoals, dtype=float)
    if subgoal_arr.ndim == 1:
        subgoal_arr = subgoal_arr.reshape(1, -1)
    if subgoal_arr.ndim != 2 or subgoal_arr.shape[0] == 0:
        raise ValueError(
            _format_error(
                f"{call}() needs subgoals of shape (n_subgoals, {n_dims}); got "
                f"shape {np.shape(subgoals)}.",
                why="each row is one subgoal position, visited in order.",
                fix=f"pass an array with one row of {n_dims} coordinates per subgoal.",
            )
        )
    for subgoal in subgoal_arr:
        _validated_goal(subgoal, n_dims, call=call)

    measured = support.sample_valid & support.sample_in_epochs
    segment_results = []
    start: int | None = None
    for index, subgoal in enumerate(subgoal_arr):
        target = resolve_goal(env, subgoal, paths=paths, call=call)
        distances = target.distances(env, paths, positions_arr)
        rows = np.flatnonzero(measured & ~np.isnan(distances))
        if start is None:
            if not rows.size:
                raise ValueError(
                    _format_error(
                        f"{call}() found no measured position to start from.",
                        why=(
                            "every sample is missing, outside epochs or (geodesic) "
                            "outside the environment."
                        ),
                        fix="check positions, epochs and the environment.",
                    )
                )
            start = int(rows[0])
        rows = rows[rows >= start]
        arrival = int(rows[np.argmin(distances[rows])]) if rows.size else start
        if arrival == start and distances[start] > 0:
            warnings.warn(
                _format_error(
                    f"{call}(): subgoal {index} at {subgoal.tolist()} is never "
                    "approached; the closest measured sample after the previous "
                    "subgoal is the segment's start.",
                    why=(
                        "the animal never got closer to this subgoal, so its "
                        "segment has no path and its efficiency is NaN."
                    ),
                    fix=(
                        "check the subgoal order and coordinates, or the trial "
                        "window passed as times and positions."
                    ),
                ),
                UserWarning,
                stacklevel=_external_stacklevel(),
            )
        segment = slice(start, arrival + 1)
        traveled, complete = _observed_path_length(
            positions_arr[segment],
            support.sample_valid[segment],
            support.interval_mask[start:arrival],
            paths=paths,
        )
        segment_results.append(
            PathEfficiencyResult(
                traveled_length=traveled,
                shortest_length=float(distances[start]),
                time_efficiency=None,
                angular_efficiency=_angular_efficiency(
                    env,
                    *_smoothed_steps(
                        times_arr[segment],
                        positions_arr[segment],
                        support.sample_valid[segment],
                        support.interval_mask[start:arrival],
                        smoothed=smoothed[segment],
                    ),
                    target,
                    paths=paths,
                ),
                start_position=positions_arr[start],
                goal=subgoal,
                metric=metric,
                fully_observed=complete,
            )
        )
        start = arrival

    return SubgoalEfficiencyResult(
        segment_results=tuple(segment_results), subgoal_positions=subgoal_arr
    )


@reads_pynapple_samples
def compute_path_efficiency(
    env: Environment[CartesianSystem],
    times: ArrayLike,
    positions: ArrayLike,
    intervals: Intervals[Any] | None = None,
    *,
    goal: Any,
    metric: DistanceMetric = "geodesic",
    reference_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> PathEfficiencyResult | pd.DataFrame:
    """Compute comprehensive path efficiency metrics.

    Parameters
    ----------
    env : Environment
        The environment the positions were recorded in.
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples, n_dims)
        Positions in the environment's coordinates. Rows with a non-finite
        coordinate are missing.
    intervals : Intervals or IntervalSet, optional
        Trials or other intervals. Each is measured from its own samples,
        ``[start, stop)``, as if it were the whole recording, and the result
        has one value per interval. A pynapple ``IntervalSet`` is read as
        :func:`~neurospatial.io.from_pynapple` reads it.
    goal : str, array-like of shape (n_dims,), or per-interval sequence
        A region name, a point in environment coordinates, or, with
        ``intervals``, one per interval (a ``pandas.Series`` indexed by
        ``interval_id`` such as ``trials.end_region``; ``None`` for an unknown
        goal), read as in :func:`distance_to_goal`. A polygon region is
        approached at its nearest bin: the goal direction is the direction of
        the shortest path to the nearest bin of the region, the direction in
        which the distance to the goal falls fastest. For a round well
        (``regions.add(name, point=..., radius=...)``) that is the direction
        to its centre. A point region or a point is aimed at itself.
    metric : {"geodesic", "euclidean"}, default="geodesic"
        ``"geodesic"``: the shortest path that stays inside the environment
        (around walls, along a track's centerline), measured from each
        position; the goal direction is the direction in which that path
        leaves the position, toward its first corner. ``"euclidean"``: the
        straight line. In an arena without walls between the animal and the
        goal they agree. A geodesic goal point must lie inside the
        environment.
    reference_speed : float, optional
        Reference speed for time efficiency, in position units per second;
        finite and greater than zero. None leaves ``time_efficiency`` None.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    PathEfficiencyResult or pandas.DataFrame
        All path efficiency metrics, measured from the first observed sample
        (the first row with finite coordinates) to ``goal``; missing rows
        before the first or after the last observed sample are treated as
        deleted, also for ``time_efficiency``. With ``intervals``, a table
        with one row per interval indexed by ``interval_id`` and the columns
        ``traveled_length``, ``shortest_length``, ``efficiency``,
        ``time_efficiency``, ``angular_efficiency``, ``fully_observed`` and
        ``goal``; an interval with an unknown goal has NaN metrics. When the
        goal is unreachable under ``metric="geodesic"`` (it lies on a
        disconnected component), ``shortest_length`` is ``inf`` and the
        derived ``efficiency`` and ``time_efficiency`` are ``nan``; when the
        first observed position is outside the environment,
        ``shortest_length`` is NaN.

    Raises
    ------
    ValueError
        If the inputs are empty or malformed (``times`` not finite and
        strictly increasing, positions without one coordinate per environment
        dimension, ``break_before`` not a Boolean array with one entry per
        sample), ``metric`` is not recognized, a region covers no bin, the
        goal point is malformed or (geodesic) outside the environment, or
        per-sample intervals overlap.
    TypeError
        If ``intervals`` is not an Intervals or a pynapple IntervalSet, or a
        per-interval goal is a number (a well number) instead of a region
        name or a point.
    RegionNotFoundError
        If a region name is not in ``env.regions``.
    CartesianOnlyError
        If ``env`` is an egocentric polar environment.
    ValueError
        If ``reference_speed`` is not a finite speed greater than zero.

    Notes
    -----
    ``traveled_length`` is the distance over usable intervals, as returned by
    ``traveled_path_length`` with the same arguments: a lost frame is bridged
    by the straight step between its neighbours when they are no more than
    ``max_gap`` apart, while a longer gap, a declared break or an interval
    outside ``epochs`` is left out, and it is ``nan`` when nothing was
    observed (including a single sample). ``efficiency`` compares the whole
    route with the shortest one, so it is ``nan`` unless every interval from
    the first to the last observed sample is usable; with an unbridged
    stretch the observed length underestimates the route and the ratio would
    be biased upward. ``shortest_length`` is the distance from the start to
    the goal, as :func:`distance_to_goal` measures it (to the nearest goal
    bin centre for a region). ``time_efficiency`` is wall-clock. When a long
    tracking loss makes a trial's efficiency ``nan``, report its
    ``traveled_length`` instead, or restrict the analysis to completely
    observed trials.

    ``angular_efficiency`` is a mean over the observed steps, as returned by
    :func:`angular_efficiency` with the same arguments, so a tracking loss
    removes steps from the mean but does not bias it.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior import compute_path_efficiency
    >>> positions = np.column_stack([np.arange(11.0), np.zeros(11)])  # cm
    >>> times = np.arange(11) / 10  # seconds
    >>> env = Environment.from_samples(positions, bin_size=1.0)
    >>> goal = positions[-1]
    >>> result = compute_path_efficiency(env, times, positions, goal=goal)
    >>> result.traveled_length, result.efficiency
    (10.0, 1.0)
    >>> positions[5] = np.nan  # one lost frame, bridged
    >>> result = compute_path_efficiency(env, times, positions, goal=goal)
    >>> result.traveled_length, result.efficiency
    (10.0, 1.0)
    >>> result = compute_path_efficiency(env, times, positions, goal=goal, max_gap=0.15)
    >>> result.traveled_length  # the 0.2 s gap is a tracking loss here
    8.0
    >>> bool(np.isnan(result.efficiency))  # the route is incomplete
    True
    """
    from neurospatial.behavior._goals import goal_rows

    call = "compute_path_efficiency"
    paths = metric_paths(metric, env, call=call)
    times_arr, positions_arr, intervals, support = _goal_observations(
        env,
        times,
        positions,
        intervals,
        call=call,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="no path was observed",
    )
    if reference_speed is not None:
        _validate_reference_speed(reference_speed, call=call)
    rows = goal_rows(
        env, times_arr, intervals, goal, paths=paths, call=call, disjoint=False
    )
    results = [
        None
        if row.goal is None
        else _path_efficiency(
            env,
            times_arr[row.samples],
            positions_arr[row.samples],
            _support_of(times_arr, support, row.samples, intervals),
            row.goal,
            paths=paths,
            metric=metric,
            reference_speed=reference_speed,
        )
        for row in rows
    ]
    if intervals is None:
        assert results[0] is not None  # one goal without intervals
        return results[0]
    return _result_table(
        rows,
        results,
        (
            "traveled_length",
            "shortest_length",
            "efficiency",
            "time_efficiency",
            "angular_efficiency",
            "fully_observed",
        ),
    )


def _path_efficiency(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    support: ObservationSupport,
    goal: Goal,
    *,
    paths: EnvironmentPaths | None,
    metric: DistanceMetric,
    reference_speed: float | None,
) -> PathEfficiencyResult:
    """:func:`compute_path_efficiency` of one recording or interval."""
    traveled, complete = _observed_path_length(
        positions, support.sample_valid, support.interval_mask, paths=paths
    )
    ang_eff = _angular_efficiency(
        env,
        *_smoothed_steps(times, positions, support.sample_valid, support.interval_mask),
        goal,
        paths=paths,
    )
    # Missing rows before the first or after the last observed sample are
    # deleted rows: the path starts and ends where the animal was observed.
    observed = np.flatnonzero(support.sample_valid)
    if observed.size:
        times = times[observed[0] : observed[-1] + 1]
        positions = positions[observed[0] : observed[-1] + 1]
    shortest = (
        float(goal.distances(env, paths, positions[:1])[0])
        if len(positions)
        else np.nan
    )
    time_eff = None
    if reference_speed is not None and len(positions) >= 2:
        time_eff = time_efficiency(
            times,
            positions,
            reference_speed=reference_speed,
            optimal_distance=shortest,
        )
    start = positions[0] if len(positions) else np.full(env.n_dims, np.nan)
    return PathEfficiencyResult(
        traveled_length=traveled,
        shortest_length=shortest,  # inf if the goal is unreachable
        time_efficiency=time_eff,
        angular_efficiency=ang_eff,
        start_position=start,
        goal=goal.label,
        metric=metric,
        fully_observed=complete,
    )


def _result_table(
    rows: list[GoalRow], results: list[Any], columns: tuple[str, ...]
) -> pd.DataFrame:
    """One row per interval of the results' ``columns`` and goal.

    A row whose goal is unknown (no result) has NaN metrics and goal None.
    """
    import pandas as pd

    records = [
        {
            **{
                name: (np.nan if result is None else getattr(result, name))
                for name in columns
            },
            "goal": None if row.goal is None else row.goal.label,
        }
        for row, result in zip(rows, results, strict=True)
    ]
    table = pd.DataFrame.from_records(records, columns=[*columns, "goal"])
    table.index = pd.Index([row.interval_id for row in rows], name="interval_id")
    return table


# =============================================================================
# Goal-Directed Navigation Functions
# =============================================================================


@reads_pynapple_samples
def goal_vector(
    positions: NDArray[np.float64],
    goal: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute vector from each position to goal.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates in allocentric frame.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position in allocentric frame.

    Returns
    -------
    NDArray[np.float64], shape (n_samples, n_dims)
        Vector from each position to goal.

    Raises
    ------
    ValueError
        If goal dimensions don't match positions dimensions.

    Examples
    --------
    >>> positions = np.array([[0.0, 0.0], [10.0, 0.0]])
    >>> goal = np.array([50.0, 0.0])
    >>> goal_vector(positions, goal)
    array([[50.,  0.],
           [40.,  0.]])
    """
    return _goal_vector(positions, goal, call="goal_vector")


def _goal_vector(
    positions: NDArray[np.float64], goal: ArrayLike, *, call: str
) -> NDArray[np.float64]:
    """``goal - positions``, naming ``call`` when the dimensions differ."""
    positions = np.asarray(positions)
    goal_arr = _validated_goal(goal, positions.shape[1], call=call)
    vector: NDArray[np.float64] = goal_arr[np.newaxis, :] - positions
    return vector


def _validated_goal(goal: ArrayLike, n_dims: int, *, call: str) -> NDArray[np.float64]:
    """Return ``goal`` as a finite float array of shape ``(n_dims,)``.

    Parameters
    ----------
    goal : array-like
        Goal position.
    n_dims : int
        Number of position dimensions.
    call : str
        Public function name, used in messages.

    Raises
    ------
    ValueError
        If ``goal`` does not have shape ``(n_dims,)`` or has a non-finite
        coordinate. Without this check NumPy would broadcast a one-coordinate
        goal against every position dimension.
    """
    goal_arr = np.asarray(goal, dtype=float)
    if goal_arr.shape != (n_dims,):
        n_goal = goal_arr.shape[0] if goal_arr.ndim == 1 else None
        got = (
            f"goal has {n_goal} coordinates but positions have {n_dims}"
            if n_goal is not None
            else f"goal has shape {goal_arr.shape}"
        )
        raise ValueError(
            _format_error(
                f"{call}() needs a goal with one coordinate per position "
                f"dimension; {got}.",
                why=(
                    "the goal is a point in the positions' coordinate frame; a "
                    "goal of another shape would be broadcast against every "
                    "position dimension."
                ),
                fix=(
                    f"pass goal with shape ({n_dims},), in the same coordinates "
                    "and units as positions."
                ),
            )
        )
    if not np.isfinite(goal_arr).all():
        raise ValueError(
            _format_error(
                f"{call}() got a goal with a non-finite coordinate: {goal_arr}.",
                why="no distance or direction to a missing goal can be measured.",
                fix="pass the goal's finite coordinates.",
            )
        )
    return goal_arr


def _validate_reference_speed(reference_speed: float, *, call: str) -> None:
    """Raise unless ``reference_speed`` is a finite positive speed."""
    if not (np.isfinite(reference_speed) and reference_speed > 0):
        raise ValueError(
            _format_error(
                f"{call}() got reference_speed={reference_speed!r}; it must be a "
                "finite speed greater than zero.",
                why=(
                    "the optimal travel time is the shortest distance divided by "
                    "reference_speed; a zero, negative or non-finite speed gives "
                    "no travel time."
                ),
                fix=(
                    "pass the animal's typical running speed in position units per "
                    "second, e.g. reference_speed=20.0 for 20 cm/s with positions "
                    "in cm."
                ),
            )
        )


@reads_pynapple_samples
def goal_direction(
    positions: NDArray[np.float64],
    goal: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute direction (angle) from each position to goal.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates in allocentric frame.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position in allocentric frame.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Angle in radians from each position to goal.
        Uses allocentric convention: 0=East, pi/2=North.

    Examples
    --------
    >>> positions = np.array([[0.0, 0.0]])
    >>> goal = np.array([1.0, 0.0])  # East
    >>> goal_direction(positions, goal)
    array([0.])
    """
    vec = _goal_vector(positions, goal, call="goal_direction")
    return np.arctan2(vec[:, 1], vec[:, 0])


@reads_pynapple_samples
def instantaneous_goal_alignment(
    env: Environment[CartesianSystem],
    times: ArrayLike,
    positions: ArrayLike,
    intervals: Intervals[Any] | None = None,
    *,
    goal: Any,
    metric: DistanceMetric = "geodesic",
    min_speed: float | None = None,
    speed: ArrayLike | None = None,
    headings: ArrayLike | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute instantaneous alignment between movement and goal direction.

    Parameters
    ----------
    env : Environment
        The environment the positions were recorded in.
    times : ArrayLike, shape (n_samples,)
        Timestamps in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples, n_dims)
        Positions in the environment's coordinates, with x and y columns.
    intervals : Intervals or IntervalSet, optional
        Trials or other intervals. Each is measured from its own samples,
        ``[start, stop)``, as if it were the whole recording, so nothing spans
        an interval's edge. Samples outside every interval are NaN;
        overlapping intervals are refused. A pynapple ``IntervalSet`` is read
        as :func:`~neurospatial.io.from_pynapple` reads it.
    goal : str, array-like of shape (n_dims,), or per-interval sequence
        A region name, a point in environment coordinates, or, with
        ``intervals``, one per interval (a ``pandas.Series`` indexed by
        ``interval_id`` such as ``trials.end_region``; ``None`` for an unknown
        goal), read as in :func:`distance_to_goal`. A polygon region is
        approached at its nearest bin: the goal direction is the direction of
        the shortest path to the nearest bin of the region, the direction in
        which the distance to the goal falls fastest. For a round well
        (``regions.add(name, point=..., radius=...)``) that is the direction
        to its centre. A point region or a point is aimed at itself.
    metric : {"geodesic", "euclidean"}, default="geodesic"
        ``"geodesic"``: the shortest path that stays inside the environment
        (around walls, along a track's centerline), measured from each
        position; the goal direction is the direction in which that path
        leaves the position, toward its first corner. ``"euclidean"``: the
        straight line. In an arena without walls between the animal and the
        goal they agree. A geodesic goal point must lie inside the
        environment.
    min_speed : float or None, default=None
        Minimum speed for a direction of movement, in position units per
        second. None is 5 cm/s expressed in ``env.units`` (``0.05`` for
        meters, ``50.0`` for millimeters); with units that are not a metric
        length, pass it explicitly.
    speed : ArrayLike, shape (n_samples,), optional
        Your own speed of each sample in position units per second (NaN
        where not observed), used instead of computing it from
        ``positions``: for example speed from your own position processing,
        or ``compute_speed(times, positions, smooth_window=0)`` for positions
        you already smoothed. Samples excluded by ``max_gap``, ``epochs`` or
        ``break_before`` stay excluded. Default: the speed of
        :func:`compute_speed` (from x and y, the horizontal speed).
    headings : ArrayLike, shape (n_samples,), optional
        Your own direction of each sample in radians (allocentric: 0 = East,
        +π/2 = North; NaN where not observed), used instead of the direction
        of movement computed from ``positions``: for example the direction of
        your own smoothed velocity, or head direction from a head-mounted LED
        or sensor. Samples slower than ``min_speed`` still have none. Default:
        the direction of movement of :func:`heading_from_velocity`.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Cosine of the angle between the direction of movement and the goal
        direction in the x-y plane, range [-1, 1]. NaN where the animal is
        not moving (speed zero or below ``min_speed``), not observed (a
        missing position, one outside the environment, or a sample in no
        observed run), at the goal (in a goal bin, or exactly at the goal
        point), outside every interval, or with an unknown goal.

    Raises
    ------
    ValueError
        If the inputs are empty or malformed (``times`` not finite and
        strictly increasing, positions without one coordinate per environment
        dimension, ``break_before`` not a Boolean array with one entry per
        sample), ``metric`` is not recognized, a region covers no bin, the
        goal point is malformed or (geodesic) outside the environment, or
        per-sample intervals overlap.
    TypeError
        If ``intervals`` is not an Intervals or a pynapple IntervalSet, or a
        per-interval goal is a number (a well number) instead of a region
        name or a point.
    RegionNotFoundError
        If a region name is not in ``env.regions``.
    CartesianOnlyError
        If ``env`` is an egocentric polar environment.
    ValueError
        If ``min_speed`` is not finite and non-negative, or is None and
        ``env.units`` is not a metric length.

    Notes
    -----
    A NaN row is bridged like a deleted row when its valid neighbours are no
    more than ``max_gap`` apart; only a longer gap, an ``epochs`` edge or a
    declared ``break_before`` splits the recording into runs, and no velocity
    or heading spans a split.

    The direction of movement exists only where the animal moves, the rule of
    :func:`heading_from_velocity`.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior import instantaneous_goal_alignment
    >>> positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
    >>> env = Environment.from_samples(positions, bin_size=5.0, units="cm")
    >>> times = np.linspace(0, 10, 21)
    >>> alignment = instantaneous_goal_alignment(
    ...     env, times, positions, goal=[100.0, 0.0], min_speed=0.0
    ... )
    >>> bool(np.nanmean(alignment) > 0.9)
    True

    A lost frame (NaN row) has no alignment:

    >>> positions[10] = np.nan
    >>> alignment = instantaneous_goal_alignment(
    ...     env, times, positions, goal=[100.0, 0.0], min_speed=1.0
    ... )
    >>> bool(np.isnan(alignment[10])), bool(np.all(alignment[:10] > 0.99))
    (True, True)
    """
    call = "instantaneous_goal_alignment"
    return _alignment_rows(
        env,
        times,
        positions,
        intervals,
        goal=goal,
        metric=metric,
        min_speed=min_speed,
        speed=speed,
        headings=headings,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        call=call,
        result="every alignment is NaN",
        disjoint=True,
    ).alignment


def _alignment_rows(
    env: Environment,
    times: ArrayLike,
    positions: ArrayLike,
    intervals: Any,
    *,
    goal: Any,
    metric: DistanceMetric,
    min_speed: float | None,
    speed: ArrayLike | None,
    headings: ArrayLike | None,
    max_gap: float | None,
    epochs: Any,
    break_before: ArrayLike | None,
    call: str,
    result: str,
    disjoint: bool,
) -> _Alignment:
    """Goal alignment of every sample, and the mean of each goal row."""
    from neurospatial.behavior._goals import goal_rows
    from neurospatial.behavior._kinematics import resolve_min_speed, validated_motion

    paths = metric_paths(metric, env, call=call)
    times_arr, positions_arr, intervals, support = _goal_observations(
        env,
        times,
        positions,
        intervals,
        call=call,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result=result,
        columns="xy",
    )
    min_speed = resolve_min_speed(env, min_speed, call=call)
    n = len(times_arr)
    speed_arr = validated_motion(speed, name="speed", n_samples=n, call=call)
    headings_arr = validated_motion(headings, name="headings", n_samples=n, call=call)
    rows = goal_rows(
        env, times_arr, intervals, goal, paths=paths, call=call, disjoint=disjoint
    )
    alignment = np.full(n, np.nan)
    means = []
    for row in rows:
        if row.goal is None:
            means.append(np.nan)
            continue
        sl = row.samples
        values = _goal_alignment(
            times_arr[sl],
            positions_arr[sl],
            row.goal.directions(env, paths, positions_arr[sl]),
            _support_of(times_arr, support, sl, intervals),
            min_speed=min_speed,
            speed=None if speed_arr is None else speed_arr[sl],
            headings=None if headings_arr is None else headings_arr[sl],
        )
        alignment[sl] = values
        means.append(_mean_alignment(values))
    return _Alignment(
        alignment, rows, means, intervals, times_arr, positions_arr, support
    )


@dataclass(frozen=True)
class _Alignment:
    """:func:`_alignment_rows` output.

    Attributes
    ----------
    alignment : ndarray, shape (n_samples,)
        Per-sample alignment; NaN outside every row (with overlapping rows,
        the last row's).
    rows : list of GoalRow
    means : list of float
        Mean alignment of each row.
    intervals : Intervals or None
        The validated intervals.
    times, positions, support
        The validated samples and their observation support.
    """

    alignment: NDArray[np.float64]
    rows: list[GoalRow]
    means: list[float]
    intervals: Intervals[Any] | None
    times: NDArray[np.float64]
    positions: NDArray[np.float64]
    support: ObservationSupport


def _goal_alignment(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    directions: NDArray[np.float64],
    support: ObservationSupport,
    *,
    min_speed: float,
    speed: NDArray[np.float64] | None,
    headings: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Cosine between movement and goal direction; NaN where not moving.

    No validation or warning: ``support`` is already resolved.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Validated, strictly increasing timestamps in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Validated positions, ``n_dims >= 2``; NaN rows are missing.
    directions : ndarray, shape (n_samples, n_dims)
        Unit goal direction at each sample; NaN where there is none (at the
        goal, outside the environment).
    support : ObservationSupport
        Usable intervals of ``(times, positions)``.
    min_speed : float
        Position units per second.

    Returns
    -------
    ndarray, shape (n_samples,)
        ``cos(heading - goal_direction)`` in the x-y plane where the animal
        moves and the goal direction is defined, else NaN.
    """
    from neurospatial._velocity import DEFAULT_SPEED_SMOOTHING
    from neurospatial.behavior._kinematics import _movement_heading_and_speed

    goal_heading = np.arctan2(directions[:, 1], directions[:, 0])
    # NaN where the animal is not moving: no direction of movement.
    heading, _ = _movement_heading_and_speed(
        times,
        positions,
        support.interval_mask,
        smooth_window=DEFAULT_SPEED_SMOOTHING,
        min_speed=min_speed,
        speed=speed,
        headings=headings,
    )
    # At the goal there is no goal direction (NaN), and arctan2 of a NaN
    # direction stays NaN, so an arrival is never scored.
    alignment: NDArray[np.float64] = np.cos(heading - goal_heading)
    return alignment


@reads_pynapple_samples
def goal_bias(
    env: Environment[CartesianSystem],
    times: ArrayLike,
    positions: ArrayLike,
    intervals: Intervals[Any] | None = None,
    *,
    goal: Any,
    metric: DistanceMetric = "geodesic",
    min_speed: float | None = None,
    speed: ArrayLike | None = None,
    headings: ArrayLike | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> float | pd.Series:
    """Compute mean alignment toward goal over trajectory.

    The mean of :func:`instantaneous_goal_alignment` over the samples where
    the animal moves.

    Parameters
    ----------
    env : Environment
        The environment the positions were recorded in.
    times : ArrayLike, shape (n_samples,)
        Timestamps in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples, n_dims)
        Positions in the environment's coordinates, with x and y columns.
    intervals : Intervals or IntervalSet, optional
        Trials or other intervals. Each is measured from its own samples,
        ``[start, stop)``, as if it were the whole recording, and the result
        has one value per interval. A pynapple ``IntervalSet`` is read as
        :func:`~neurospatial.io.from_pynapple` reads it.
    goal : str, array-like of shape (n_dims,), or per-interval sequence
        A region name, a point in environment coordinates, or, with
        ``intervals``, one per interval (a ``pandas.Series`` indexed by
        ``interval_id`` such as ``trials.end_region``; ``None`` for an unknown
        goal), read as in :func:`distance_to_goal`. A polygon region is
        approached at its nearest bin: the goal direction is the direction of
        the shortest path to the nearest bin of the region, the direction in
        which the distance to the goal falls fastest. For a round well
        (``regions.add(name, point=..., radius=...)``) that is the direction
        to its centre. A point region or a point is aimed at itself.
    metric : {"geodesic", "euclidean"}, default="geodesic"
        ``"geodesic"``: the shortest path that stays inside the environment
        (around walls, along a track's centerline), measured from each
        position; the goal direction is the direction in which that path
        leaves the position, toward its first corner. ``"euclidean"``: the
        straight line. In an arena without walls between the animal and the
        goal they agree. A geodesic goal point must lie inside the
        environment.
    min_speed : float or None, default=None
        Minimum speed for a direction of movement, in position units per
        second. None is 5 cm/s expressed in ``env.units`` (``0.05`` for
        meters, ``50.0`` for millimeters); with units that are not a metric
        length, pass it explicitly.
    speed : ArrayLike, shape (n_samples,), optional
        Your own speed of each sample in position units per second (NaN
        where not observed), used instead of computing it from
        ``positions``: for example speed from your own position processing,
        or ``compute_speed(times, positions, smooth_window=0)`` for positions
        you already smoothed. Samples excluded by ``max_gap``, ``epochs`` or
        ``break_before`` stay excluded. Default: the speed of
        :func:`compute_speed` (from x and y, the horizontal speed).
    headings : ArrayLike, shape (n_samples,), optional
        Your own direction of each sample in radians (allocentric: 0 = East,
        +π/2 = North; NaN where not observed), used instead of the direction
        of movement computed from ``positions``: for example the direction of
        your own smoothed velocity, or head direction from a head-mounted LED
        or sensor. Samples slower than ``min_speed`` still have none. Default:
        the direction of movement of :func:`heading_from_velocity`.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    float or pandas.Series
        Goal bias in [-1, 1]; with ``intervals``, one per interval indexed by
        ``interval_id``. Interpretation:

        - > 0.5: strong goal-directed movement
        - 0 to 0.5: weak goal direction, with wandering
        - < 0: net movement away from the goal

        NaN when the animal never moves at an observed sample away from the
        goal, or the goal is unknown.

    Raises
    ------
    ValueError
        If the inputs are empty or malformed (``times`` not finite and
        strictly increasing, positions without one coordinate per environment
        dimension, ``break_before`` not a Boolean array with one entry per
        sample), ``metric`` is not recognized, a region covers no bin, the
        goal point is malformed or (geodesic) outside the environment, or
        per-sample intervals overlap.
    TypeError
        If ``intervals`` is not an Intervals or a pynapple IntervalSet, or a
        per-interval goal is a number (a well number) instead of a region
        name or a point.
    RegionNotFoundError
        If a region name is not in ``env.regions``.
    CartesianOnlyError
        If ``env`` is an egocentric polar environment.
    ValueError
        If ``min_speed`` is not finite and non-negative, or is None and
        ``env.units`` is not a metric length.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior import goal_bias
    >>> positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
    >>> env = Environment.from_samples(positions, bin_size=5.0, units="cm")
    >>> times = np.linspace(0, 10, 21)
    >>> goal_bias(env, times, positions, goal=[100.0, 0.0], min_speed=0.0)
    1.0

    An animal that never moves has no direction of movement, so no bias:

    >>> still = np.zeros((21, 2))
    >>> goal_bias(env, times, still, goal=[100.0, 0.0], min_speed=0.0)
    nan
    """
    aligned = _alignment_rows(
        env,
        times,
        positions,
        intervals,
        goal=goal,
        metric=metric,
        min_speed=min_speed,
        speed=speed,
        headings=headings,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        call="goal_bias",
        result="the goal bias is NaN",
        disjoint=False,
    )
    if aligned.intervals is None:
        return aligned.means[0]
    from neurospatial.behavior._goals import by_interval

    return by_interval(aligned.rows, aligned.means, name="goal_bias")


def _mean_alignment(alignment: NDArray[np.float64]) -> float:
    """Mean of the finite alignments, or NaN when there is none."""
    moving = alignment[np.isfinite(alignment)]
    return float(np.mean(moving)) if moving.size else np.nan


@reads_pynapple_samples
def approach_rate(
    env: Environment[CartesianSystem],
    times: ArrayLike,
    positions: ArrayLike,
    intervals: Intervals[Any] | None = None,
    *,
    goal: Any,
    metric: DistanceMetric = "geodesic",
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute rate of distance change toward goal.

    The rate at a valid sample ``k`` is the change in the distance to the goal
    (as :func:`distance_to_goal` measures it) since the previous valid
    sample, divided by the time between them. Missing samples (non-finite
    coordinates, or positions outside the environment) are skipped as if
    their rows were deleted, and are NaN. The rate is measured only when the
    time between the two valid samples is at most ``max_gap``, they fall
    inside one of ``epochs`` and no break is declared between them;
    otherwise it is NaN, so no rate is measured across a tracking loss or
    between recordings.

    Parameters
    ----------
    env : Environment
        The environment the positions were recorded in.
    times : ArrayLike, shape (n_samples,)
        Timestamps in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples, n_dims)
        Positions in the environment's coordinates.
    intervals : Intervals or IntervalSet, optional
        Trials or other intervals. Each is measured from its own samples,
        ``[start, stop)``, as if it were the whole recording, so nothing spans
        an interval's edge. Samples outside every interval are NaN;
        overlapping intervals are refused. A pynapple ``IntervalSet`` is read
        as :func:`~neurospatial.io.from_pynapple` reads it.
    goal : str, array-like of shape (n_dims,), or per-interval sequence
        A region name, a point in environment coordinates, or, with
        ``intervals``, one per interval (a ``pandas.Series`` indexed by
        ``interval_id`` such as ``trials.end_region``; ``None`` for an unknown
        goal), read as in :func:`distance_to_goal`. A polygon region is
        approached at its nearest bin: the goal direction is the direction of
        the shortest path to the nearest bin of the region, the direction in
        which the distance to the goal falls fastest. For a round well
        (``regions.add(name, point=..., radius=...)``) that is the direction
        to its centre. A point region or a point is aimed at itself.
    metric : {"geodesic", "euclidean"}, default="geodesic"
        ``"geodesic"``: the shortest path that stays inside the environment
        (around walls, along a track's centerline), measured from each
        position; the goal direction is the direction in which that path
        leaves the position, toward its first corner. ``"euclidean"``: the
        straight line. In an arena without walls between the animal and the
        goal they agree. A geodesic goal point must lie inside the
        environment.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Rate of distance change (d(distance)/dt) in units per second.
        Negative values indicate approaching the goal. The first sample of
        the recording or of each interval, and every sample whose backward
        interval is not usable, is NaN; a single sample gives ``[nan]``.

    Raises
    ------
    ValueError
        If the inputs are empty or malformed (``times`` not finite and
        strictly increasing, positions without one coordinate per environment
        dimension, ``break_before`` not a Boolean array with one entry per
        sample), ``metric`` is not recognized, a region covers no bin, the
        goal point is malformed or (geodesic) outside the environment, or
        per-sample intervals overlap.
    TypeError
        If ``intervals`` is not an Intervals or a pynapple IntervalSet, or a
        per-interval goal is a number (a well number) instead of a region
        name or a point.
    RegionNotFoundError
        If a region name is not in ``env.regions``.
    CartesianOnlyError
        If ``env`` is an egocentric polar environment.

    Examples
    --------
    Run toward a goal at x = 100 at 8 Hz, then jump 48 units ahead:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior import approach_rate
    >>> corridor = np.column_stack([np.linspace(0, 100, 201), np.zeros(201)])
    >>> env = Environment.from_samples(corridor, bin_size=1.0)
    >>> times = np.arange(6) / 8
    >>> positions = np.column_stack([[0.0, 1, 2, 50, 51, 52], np.zeros(6)])
    >>> approach_rate(env, times, positions, goal=[100.0, 0.0]).tolist()
    [nan, -8.0, -8.0, -384.0, -8.0, -8.0]

    On one continuous clock the jump reads as a fast approach. If a new
    recording starts at sample 3, declare the break:

    >>> new_recording = np.array([False, False, False, True, False, False])
    >>> approach_rate(
    ...     env, times, positions, goal=[100.0, 0.0], break_before=new_recording
    ... ).tolist()
    [nan, -8.0, -8.0, nan, -8.0, -8.0]

    A missing position has no rate; the rate after it spans the bridged gap:

    >>> positions[1] = np.nan
    >>> approach_rate(env, times, positions, goal=[100.0, 0.0]).tolist()
    [nan, nan, -8.0, -384.0, -8.0, -8.0]
    """
    from neurospatial.behavior._goals import goal_rows

    call = "approach_rate"
    paths = metric_paths(metric, env, call=call)
    times_arr, positions_arr, intervals, support = _goal_observations(
        env,
        times,
        positions,
        intervals,
        call=call,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="every approach rate is NaN",
    )
    rows = goal_rows(
        env, times_arr, intervals, goal, paths=paths, call=call, disjoint=True
    )
    rates = np.full(len(times_arr), np.nan)
    for row in rows:
        if row.goal is None:
            continue
        sl = row.samples
        rates[sl] = _distance_rates(
            times_arr[sl],
            row.goal.distances(env, paths, positions_arr[sl]),
            _support_of(times_arr, support, sl, intervals),
        )
    return rates


def _interval_distance_rates(
    times: NDArray[np.float64],
    distances: NDArray[np.float64],
    support: ObservationSupport,
) -> NDArray[np.float64]:
    """Change of ``distances`` per second over each interval, shape (n - 1,).

    NaN for an unusable interval. Every interval inside a bridged gap of
    missing distances carries that gap's rate.
    """
    from neurospatial.behavior._kinematics import interval_velocity

    support = support.restricted(times, np.isfinite(distances))
    return interval_velocity(times, distances[:, None], support.interval_mask)[:, 0]


def _distance_rates(
    times: NDArray[np.float64],
    distances: NDArray[np.float64],
    support: ObservationSupport,
) -> NDArray[np.float64]:
    """Change of ``distances`` per second, at the sample ending each interval.

    A sample whose distance is not finite (a position outside the
    environment, or disconnected from the goal) is missing for the rate, so
    its gap is gated again like a deleted row's.
    """
    rates = np.full(len(distances), np.nan)
    rates[1:] = _interval_distance_rates(times, distances, support)
    # A bridged gap's rate belongs to the valid sample that ends it.
    rates[~np.isfinite(distances)] = np.nan

    return rates


@reads_pynapple_samples
def compute_goal_directed_metrics(
    env: Environment[CartesianSystem],
    times: ArrayLike,
    positions: ArrayLike,
    intervals: Intervals[Any] | None = None,
    *,
    goal: Any,
    metric: DistanceMetric = "geodesic",
    min_speed: float | None = None,
    speed: ArrayLike | None = None,
    headings: ArrayLike | None = None,
    goal_radius: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> GoalDirectedMetrics | pd.DataFrame:
    """Compute comprehensive goal-directed navigation metrics.

    Parameters
    ----------
    env : Environment
        The environment the positions were recorded in.
    times : ArrayLike, shape (n_samples,)
        Timestamps in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples, n_dims)
        Positions in the environment's coordinates, with x and y columns.
    intervals : Intervals or IntervalSet, optional
        Trials or other intervals. Each is measured from its own samples,
        ``[start, stop)``, as if it were the whole recording, and the result
        has one value per interval. A pynapple ``IntervalSet`` is read as
        :func:`~neurospatial.io.from_pynapple` reads it.
    goal : str, array-like of shape (n_dims,), or per-interval sequence
        A region name, a point in environment coordinates, or, with
        ``intervals``, one per interval (a ``pandas.Series`` indexed by
        ``interval_id`` such as ``trials.end_region``; ``None`` for an unknown
        goal), read as in :func:`distance_to_goal`. A polygon region is
        approached at its nearest bin: the goal direction is the direction of
        the shortest path to the nearest bin of the region, the direction in
        which the distance to the goal falls fastest. For a round well
        (``regions.add(name, point=..., radius=...)``) that is the direction
        to its centre. A point region or a point is aimed at itself.
    metric : {"geodesic", "euclidean"}, default="geodesic"
        ``"geodesic"``: the shortest path that stays inside the environment
        (around walls, along a track's centerline), measured from each
        position; the goal direction is the direction in which that path
        leaves the position, toward its first corner. ``"euclidean"``: the
        straight line. In an arena without walls between the animal and the
        goal they agree. A geodesic goal point must lie inside the
        environment.
    min_speed : float or None, default=None
        Minimum speed for a direction of movement, in position units per
        second. None is 5 cm/s expressed in ``env.units`` (``0.05`` for
        meters, ``50.0`` for millimeters); with units that are not a metric
        length, pass it explicitly.
    speed : ArrayLike, shape (n_samples,), optional
        Your own speed of each sample in position units per second (NaN
        where not observed), used instead of computing it from
        ``positions``: for example speed from your own position processing,
        or ``compute_speed(times, positions, smooth_window=0)`` for positions
        you already smoothed. Samples excluded by ``max_gap``, ``epochs`` or
        ``break_before`` stay excluded. Default: the speed of
        :func:`compute_speed` (from x and y, the horizontal speed).
    headings : ArrayLike, shape (n_samples,), optional
        Your own direction of each sample in radians (allocentric: 0 = East,
        +π/2 = North; NaN where not observed), used instead of the direction
        of movement computed from ``positions``: for example the direction of
        your own smoothed velocity, or head direction from a head-mounted LED
        or sensor. Samples slower than ``min_speed`` still have none. Default:
        the direction of movement of :func:`heading_from_velocity`.
    goal_radius : float, optional
        Arrival radius for a point goal, in the units of ``positions``: the
        goal is reached at the first measured sample whose distance to the
        goal (under ``metric``) is at most ``goal_radius``. Finite and >= 0.
        None skips arrival detection for a point goal. A region goal is
        reached at the first measured sample in one of its bins, so
        ``goal_radius`` is refused with one: the region is the arrival zone.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    GoalDirectedMetrics or pandas.DataFrame
        All goal-directed metrics. With ``intervals``, a table with one row
        per interval indexed by ``interval_id`` and the columns of
        :meth:`GoalDirectedMetrics.summary` plus ``goal``; an interval with
        an unknown goal has NaN metrics. ``goal_bias`` is NaN if the animal
        never moves at an observed sample. The distance summaries use the
        measured samples: finite coordinates inside the environment, inside
        ``epochs`` when given. ``min_distance_to_goal`` is the minimum
        distance under ``metric`` over them; ``goal_distance_at_start`` and
        ``goal_distance_at_end`` are the distances at the first and last of
        them. All three are NaN when no sample is measured.
        ``time_to_reach_goal`` is the seconds from the first measured sample
        to arrival, NaN when the goal is never reached, and None for a point
        goal without ``goal_radius``.

    Raises
    ------
    ValueError
        If the inputs are empty or malformed (``times`` not finite and
        strictly increasing, positions without one coordinate per environment
        dimension, ``break_before`` not a Boolean array with one entry per
        sample), ``metric`` is not recognized, a region covers no bin, the
        goal point is malformed or (geodesic) outside the environment, or
        per-sample intervals overlap.
    TypeError
        If ``intervals`` is not an Intervals or a pynapple IntervalSet, or a
        per-interval goal is a number (a well number) instead of a region
        name or a point.
    RegionNotFoundError
        If a region name is not in ``env.regions``.
    CartesianOnlyError
        If ``env`` is an egocentric polar environment.
    ValueError
        If ``goal_radius`` is negative or not finite, or is given with a
        region goal; or ``min_speed`` is not finite and non-negative, or is
        None and ``env.units`` is not a metric length.

    Notes
    -----
    A NaN row is bridged like a deleted row when its valid neighbours are no
    more than ``max_gap`` apart; only a longer gap, an ``epochs`` edge or a
    declared ``break_before`` splits the recording into runs, and no
    velocity or heading spans a split.

    Heading alignment and approach-rate averages use observed intervals; the
    direction of movement exists only where the animal moves, the rule of
    :func:`heading_from_velocity`. Arrival is searched only among measured
    samples (inside ``epochs``), and ``time_to_reach_goal`` is wall-clock
    time from the first of them, which is known across a pause.

    Examples
    --------
    A straight run East toward a goal at x = 100 cm, with one lost frame:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior import compute_goal_directed_metrics
    >>> track = np.column_stack([np.linspace(0, 100, 101), np.zeros(101)])
    >>> env = Environment.from_samples(track, bin_size=5.0, units="cm")
    >>> times = np.arange(21) / 10  # s
    >>> positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
    >>> positions[10] = np.nan  # a lost frame
    >>> result = compute_goal_directed_metrics(env, times, positions, goal=[100.0, 0.0])
    >>> bool(result.goal_bias > 0.9)
    True
    >>> result.min_distance_to_goal
    0.0
    """
    call = "compute_goal_directed_metrics"
    if goal_radius is not None and not (np.isfinite(goal_radius) and goal_radius >= 0):
        raise ValueError(
            _format_error(
                f"{call}() got goal_radius={goal_radius!r}; it must be a finite "
                "distance >= 0.",
                why=(
                    "the goal is reached at the first sample whose distance to "
                    "the goal is at most goal_radius."
                ),
                fix=(
                    "pass the reward zone's radius in the units of positions, "
                    "e.g. goal_radius=5.0 for 5 cm, or None to skip arrival."
                ),
            )
        )
    paths = metric_paths(metric, env, call=call)
    aligned = _alignment_rows(
        env,
        times,
        positions,
        intervals,
        goal=goal,
        metric=metric,
        min_speed=min_speed,
        speed=speed,
        headings=headings,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        call=call,
        result="the goal-directed metrics are NaN",
        disjoint=False,
    )
    rows, intervals = aligned.rows, aligned.intervals
    times_arr, positions_arr, support = (
        aligned.times,
        aligned.positions,
        aligned.support,
    )
    if goal_radius is not None:
        regions = sorted(
            {
                str(row.goal.name)
                for row in rows
                if row.goal is not None and row.goal.bins is not None
            }
        )
        if regions:
            raise ValueError(
                _format_error(
                    f"{call}() got goal_radius={goal_radius!r} with the region "
                    f"goal(s) {regions}.",
                    why="a region goal is reached when the animal enters one of "
                    "its bins: the region is the arrival zone.",
                    fix="drop goal_radius, or give the region the size of the "
                    "arrival zone (regions.add(name, point=..., radius=...)).",
                )
            )
    results = [
        None
        if row.goal is None
        else _goal_directed(
            env,
            times_arr[row.samples],
            positions_arr[row.samples],
            _support_of(times_arr, support, row.samples, intervals),
            row.goal,
            bias=bias,
            paths=paths,
            metric=metric,
            goal_radius=goal_radius,
        )
        for row, bias in zip(rows, aligned.means, strict=True)
    ]
    if intervals is None:
        assert results[0] is not None  # one goal without intervals
        return results[0]
    return _result_table(
        rows,
        results,
        (
            "goal_bias",
            "mean_approach_rate",
            "goal_distance_at_start",
            "goal_distance_at_end",
            "min_distance_to_goal",
            "time_to_reach_goal",
        ),
    )


def _goal_directed(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    support: ObservationSupport,
    goal: Goal,
    *,
    bias: float,
    paths: EnvironmentPaths | None,
    metric: DistanceMetric,
    goal_radius: float | None,
) -> GoalDirectedMetrics:
    """:func:`compute_goal_directed_metrics` of one recording or interval."""
    distances = goal.distances(env, paths, positions)
    # Weighting each interval's rate by its duration averages over observed
    # time: the net change of distance divided by the usable time.
    rates = _interval_distance_rates(times, distances, support)
    usable = ~np.isnan(rates)
    durations = np.diff(times)[usable]
    mean_rate = (
        float(rates[usable] @ durations / durations.sum()) if usable.any() else np.nan
    )
    # Samples measured for the distance summaries: observed, inside the
    # environment and inside ``epochs``.
    measured = support.sample_valid & support.sample_in_epochs & ~np.isnan(distances)
    rows = np.flatnonzero(measured)
    if rows.size:
        min_dist = float(np.min(distances[rows]))
        start_dist = float(distances[rows[0]])
        end_dist = float(distances[rows[-1]])
    else:
        min_dist = start_dist = end_dist = np.nan
    time_to_reach: float | None = None
    if goal.bins is not None or goal_radius is not None:
        if goal.bins is not None:
            arrived = rows[goal.contains(env, positions[rows])]
        else:
            assert goal_radius is not None  # a point goal is tested only with one
            arrived = rows[distances[rows] <= goal_radius]
        time_to_reach = (
            float(times[arrived[0]] - times[rows[0]]) if arrived.size else np.nan
        )
    return GoalDirectedMetrics(
        goal_bias=bias,
        mean_approach_rate=mean_rate,
        time_to_reach_goal=time_to_reach,
        min_distance_to_goal=min_dist,
        goal_distance_at_start=start_dist,
        goal_distance_at_end=end_dist,
        goal=goal.label,
        metric=metric,
    )
