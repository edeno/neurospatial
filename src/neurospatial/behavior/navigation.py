"""Navigation and path efficiency metrics for spatial behavior analysis.

This module provides comprehensive tools for analyzing goal-directed navigation
and path efficiency in spatial environments. It combines:

- **Path progress and distance metrics**: Compute progress toward goals, distances
  to regions, boundaries and rewards, and cost-aware navigation
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
        distance_to_region,
        distance_to_boundary,
        distance_to_reward,
        cost_to_goal,
        time_to_goal,
        trials_to_region_arrays,
        graph_turn_sequence,
        goal_pair_direction_labels,
        heading_direction_labels,
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
    1. Segment trajectory into trials with ``segment_trials()``
    2. Convert trials to bin arrays with ``trials_to_region_arrays()``
    3. Compute metrics: ``path_progress()``, ``distance_to_region()``, etc.

**Path efficiency analysis**:
    1. Compute efficiency: ``result = compute_path_efficiency(env, times, positions, goal)``
    2. Check efficiency: ``if result.is_efficient(threshold=0.8): ...``
    3. Print summary: ``print(result)``

**Goal-directed analysis**:
    1. Compute metrics: ``result = compute_goal_directed_metrics(env, times, positions, goal)``
    2. Check goal-directed: ``if result.is_goal_directed(): ...``

Example
-------
Complete analysis pipeline for a spatial navigation task::

    from neurospatial.behavior import (
        segment_trials,
        trials_to_region_arrays,
        path_progress,
        compute_path_efficiency,
        compute_goal_directed_metrics,
    )

    # 1. Segment trajectory into trials
    trials = segment_trials(
        position_bins,
        times,
        env,
        start_region="home",
        end_regions=["goal_left", "goal_right"],
    )

    # 2. Extract trial-based regressors
    start_bins, goal_bins = trials_to_region_arrays(trials, times, env)
    progress = path_progress(
        position_bins, env, start_bins=start_bins, goal_bins=goal_bins
    )

    # 3. Compute efficiency
    result = compute_path_efficiency(env, times, positions, goal)
    print(result)

    # 4. Compute goal-directed metrics
    gd_result = compute_goal_directed_metrics(env, times, positions, goal)
    print(f"Goal bias: {gd_result.goal_bias:.2f}")

References
----------
.. [1] Batschelet, E. (1981). Circular Statistics in Biology. Academic Press.
.. [2] Johnson, A., & Redish, A. D. (2007). Neural ensembles in CA3 transiently
       encode paths forward of the animal at a decision point. J Neurosci.
       DOI: 10.1523/JNEUROSCI.3761-07.2007
"""

from __future__ import annotations

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
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin
from neurospatial.behavior._region_bins import region_representative_bin
from neurospatial.geometry import require_cartesian

if TYPE_CHECKING:
    from neurospatial._observation import ObservationSupport
    from neurospatial.behavior.segmentation import Trial
    from neurospatial.environment import Environment
    from neurospatial.geometry import CartesianSystem
    from neurospatial.ops._types import DistanceMetric


__all__ = [  # noqa: RUF022
    # Path progress and distance
    "cost_to_goal",
    "distance_to_boundary",
    "distance_to_region",
    "distance_to_reward",
    "goal_pair_direction_labels",
    "graph_turn_sequence",
    "heading_direction_labels",
    "path_progress",
    "time_to_goal",
    "trials_to_region_arrays",
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


def _require_result_metric(result: str, metric: object) -> None:
    """Raise unless ``metric`` is ``"euclidean"`` or ``"geodesic"``."""
    if metric not in ("euclidean", "geodesic"):
        raise _result_error(
            result,
            f"metric={metric!r} is not 'euclidean' or 'geodesic'.",
            why="the result records which distance its lengths were measured with.",
            fix="pass metric='euclidean' or metric='geodesic'.",
        )


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
    goal_position : NDArray[np.float64]
        Goal position coordinates, shape (n_dims,).
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
    goal_position: NDArray[np.float64]
    metric: Literal["euclidean", "geodesic"]
    fully_observed: bool

    def __post_init__(self) -> None:
        """Validate the fields, derive the efficiency and own the positions."""
        name = "PathEfficiencyResult"
        _require_result_metric(name, self.metric)
        traveled = _require_length(name, "traveled_length", self.traveled_length)
        shortest = _require_length(name, "shortest_length", self.shortest_length)
        start = np.asarray(self.start_position, dtype=float)
        goal = np.asarray(self.goal_position, dtype=float)
        if start.ndim != 1 or start.shape != goal.shape:
            raise _result_error(
                name,
                f"start_position has shape {start.shape} and goal_position "
                f"{goal.shape}.",
                why="both are one position in the same coordinates.",
                fix="pass two arrays of shape (n_dims,).",
            )
        object.__setattr__(self, "traveled_length", traveled)
        object.__setattr__(self, "shortest_length", shortest)
        object.__setattr__(self, "angular_efficiency", float(self.angular_efficiency))
        if self.time_efficiency is not None:
            object.__setattr__(self, "time_efficiency", float(self.time_efficiency))
        object.__setattr__(self, "fully_observed", bool(self.fully_observed))
        object.__setattr__(self, "start_position", start)
        object.__setattr__(self, "goal_position", goal)
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
            raise _result_error(
                "SubgoalEfficiencyResult",
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
        Mean rate of distance change in environment units per second.
        Negative values indicate approaching the goal.
        Interpretation: -10 cm/s means closing 10 cm per second on average.
    time_to_reach_goal : float or None
        Seconds from the first measured sample to the first sample within
        ``goal_radius`` of the goal. None when no ``goal_radius`` was given
        (arrival not tested); NaN when the goal was never reached. Not to be
        confused with :func:`time_to_goal`, which returns a per-sample array.
    min_distance_to_goal : float
        Closest distance to the goal (under ``metric``) over the measured
        samples, in environment units. NaN when no sample is measured.
    goal_distance_at_start : float
        Distance to the goal at the first measured sample, in environment
        units. NaN when no sample is measured.
    goal_distance_at_end : float
        Distance to the goal at the last measured sample, in environment
        units. NaN when no sample is measured.
    goal_position : NDArray[np.float64]
        Goal position used for computation, shape (n_dims,).
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
    ...     env, times, positions, np.array([100.0, 0.0])
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
    goal_position: NDArray[np.float64]
    metric: Literal["euclidean", "geodesic"]

    def __post_init__(self) -> None:
        """Validate the fields and own a read-only copy of the goal position."""
        name = "GoalDirectedMetrics"
        _require_result_metric(name, self.metric)
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
        goal = np.asarray(self.goal_position, dtype=float)
        if goal.ndim != 1:
            raise _result_error(
                name,
                f"goal_position has shape {goal.shape}.",
                why="the goal is one position.",
                fix="pass an array of shape (n_dims,).",
            )
        object.__setattr__(self, "goal_position", goal)
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


def _warn_if_no_trial_overlaps(
    times: NDArray[np.float64],
    trials: list[Trial],
    *,
    call: str,
    result: str,
) -> None:
    """Warn when trials exist but none of them contains a sample of ``times``.

    That almost always means the trials and ``times`` are on different
    clocks (for example milliseconds against seconds), which would otherwise
    give a silent all-missing result.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Sample times in seconds.
    trials : list of Trial
        Trials with ``start_time`` and ``end_time`` in seconds.
    call : str
        Public function name for the message.
    result : str
        What the caller returns everywhere in that case, e.g. ``"NaN"``.
    """
    import warnings

    from neurospatial._stacklevel import _external_stacklevel

    if len(trials) == 0 or len(times) == 0:
        return
    starts = np.array([trial.start_time for trial in trials], dtype=np.float64)
    ends = np.array([trial.end_time for trial in trials], dtype=np.float64)
    # Each trial's first sample at or after its start; it is inside the trial
    # when it is not after the trial's end.
    ordered = np.sort(times)
    first = np.searchsorted(ordered, starts, side="left")
    inside = (first < len(ordered)) & (
        ordered[np.minimum(first, len(ordered) - 1)] <= ends
    )
    if inside.any():
        return
    warnings.warn(
        f"{call}: none of the {len(trials)} trials (spanning "
        f"[{starts.min():g}, {ends.max():g}] s) contains a sample of times "
        f"(spanning [{ordered[0]:g}, {ordered[-1]:g}] s), so every output is "
        f"{result}. Fix: check that the trials and times are on the same "
        "clock, in seconds (for example trials from segment_trials() on the "
        "same recording).",
        UserWarning,
        stacklevel=_external_stacklevel(),
    )


def trials_to_region_arrays(
    trials: list[Trial],
    times: NDArray[np.float64],
    env: Environment,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Convert trial list to per-timepoint start/goal bin arrays.

    Helper function to construct arrays for path_progress() and similar
    functions. Encapsulates the trial-to-timepoint mapping logic.

    Parameters
    ----------
    trials : list[Trial]
        Trial segmentation from segment_trials().
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps for entire session.
    env : Environment
        Spatial environment (for region lookups).

    Returns
    -------
    start_bins : NDArray[np.int_], shape (n_samples,)
        Start bin index at each timepoint (-1 outside trials).
    goal_bins : NDArray[np.int_], shape (n_samples,)
        Goal bin index at each timepoint (-1 outside trials).

    Warns
    -----
    UserWarning
        If there are trials but none contains a sample of ``times`` (usually
        trials on a different clock).

    Notes
    -----
    This function has a small loop over trials (typically 10-100), but
    avoids looping over timepoints (typically 100k+). The returned arrays
    can be passed to vectorized functions like path_progress().

    **Region Handling**:

    - If a region has no bins (e.g., polygon doesn't overlap environment), that
      trial's bins remain -1
    - If a region has multiple bins (polygon region), the region is represented
      by its centroid bin: the region bin whose center is closest to the
      centroid of the region's bin centers. This representative always lies
      inside the region (robust for non-convex regions); among equally near
      bins the lowest index is chosen.
    - Failed trials (`trial.end_region is None`) have goal_bins = -1
    - Timepoints outside all trials have both arrays = -1

    Examples
    --------
    >>> trials = segment_trials(position_bins, times, env, ...)  # doctest: +SKIP
    >>> start_bins, goal_bins = trials_to_region_arrays(
    ...     trials, times, env
    ... )  # doctest: +SKIP
    >>> progress = path_progress(
    ...     position_bins, env, start_bins=start_bins, goal_bins=goal_bins
    ... )  # doctest: +SKIP
    >>> dist = distance_to_region(
    ...     position_bins, env, target_bins=goal_bins
    ... )  # doctest: +SKIP

    See Also
    --------
    path_progress : Compute normalized path progress
    distance_to_region : Distance to target region
    """
    times = np.asarray(times, dtype=np.float64)
    _warn_if_no_trial_overlaps(
        times, trials, call="trials_to_region_arrays()", result="-1"
    )
    # Initialize arrays with -1 (invalid bin)
    start_bins = np.full(len(times), -1, dtype=np.int_)
    goal_bins = np.full(len(times), -1, dtype=np.int_)

    # Loop over trials (small - typically 10-100)
    for trial in trials:
        # Create mask for timepoints within this trial
        mask = (times >= trial.start_time) & (times <= trial.end_time)

        # Represent each region by its centroid bin (the region bin nearest the
        # centroid of the region's bins), not an arbitrary first bin.
        start_bin = region_representative_bin(env, trial.start_region)
        if start_bin is not None:
            start_bins[mask] = start_bin

        # Get bins for end region (handle None for failed trials)
        if trial.end_region is not None:
            goal_bin = region_representative_bin(env, trial.end_region)
            if goal_bin is not None:
                goal_bins[mask] = goal_bin
        # For failed trials (end_region=None), goal_bins remains -1

    return start_bins, goal_bins


def _validated_position_bins(
    position_bins: ArrayLike, env: Environment, *, operation: str
) -> NDArray[np.intp]:
    """Check per-sample bin indices: integers in ``[0, n_bins)`` or ``-1``.

    ``-1`` is ``env.bin_at``'s marker for a position outside the environment
    and becomes NaN downstream. Any other negative index would silently
    count from the end of the bin arrays, and a float bin would be truncated
    to an unrelated bin, so both are refused.

    Parameters
    ----------
    position_bins : array-like of int, shape (n_samples,)
        Bin index at each sample.
    env : Environment
        Environment the bins belong to.
    operation : str
        The call named in errors, e.g. ``"path_progress()"``.

    Returns
    -------
    ndarray of intp, shape (n_samples,)
    """
    from neurospatial.environment._targets import _validate_target_bins

    if np.ndim(position_bins) != 1:
        raise ValueError(
            _format_error(
                f"{operation} got position_bins with shape {np.shape(position_bins)}.",
                why="position_bins holds one bin index per sample.",
                fix="pass a 1-D integer array, for example env.bin_at(positions)",
            )
        )
    return _validate_target_bins(
        position_bins,
        n_bins=env.n_bins,
        coordinates=env.coordinate_system,
        operation=operation,
        argument="position_bins",
        allow_missing=True,
    )


def path_progress(
    position_bins: NDArray[np.int_],
    env: Environment,
    *,
    start_bins: NDArray[np.int_],
    goal_bins: NDArray[np.int_],
    metric: DistanceMetric = "geodesic",
) -> NDArray[np.float64]:
    """Compute normalized path progress from start to goal (0 -> 1).

    Fully vectorized computation over entire session. For each timepoint t:
        progress[t] = distance(start_bins[t], position_bins[t]) /
                      distance(start_bins[t], goal_bins[t])

    Parameters
    ----------
    position_bins : NDArray[np.int_], shape (n_samples,)
        Current bin index at each timepoint, as from ``env.bin_at``; ``-1``
        marks a sample outside the environment (NaN).
    env : Environment
        Spatial environment.
    start_bins : int or NDArray[np.int_], shape (n_samples,)
        Start bin for each timepoint, or one start bin for every timepoint.
        In a per-sample array ``-1`` marks a sample with no start (NaN); a
        single start must be a bin.
    goal_bins : int or NDArray[np.int_], shape (n_samples,)
        Goal bin for each timepoint, or one goal bin for every timepoint,
        with the same ``-1`` rule.
    metric : {'geodesic', 'euclidean'}, default='geodesic'
        Distance metric.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Path progress from 0 to 1 at each timepoint, clipped to ``[0, 1]``.
        NaN for invalid bins (-1), when no path exists between bins, or when
        start and goal are the same bin and the sample is elsewhere.

    Raises
    ------
    TypeError
        If ``position_bins`` (or a start or goal) is not integer bin indices.
    BinIndexOutOfRangeError
        If a bin index is outside ``[0, n_bins)`` and is not ``-1``.
    ValueError
        If ``start_bins`` or ``goal_bins`` has neither one nor ``n_samples``
        entries.
    CartesianOnlyError
        If ``metric="euclidean"`` and ``env`` is polar. Use
        ``metric="geodesic"``.

    Notes
    -----
    This function is fully vectorized - no loops over timepoints. The only
    loop required is constructing start_bins/goal_bins arrays from trial
    information (see examples).

    **Performance strategy (automatic)**:

    - **Small environments** (n_bins < 5000): Precomputes O(n_bins^2) distance matrix
      (~200 MB for 5000 bins). Fast vectorized lookups.
    - **Large environments** (>=5000 bins): Computes per-pair distance fields.
      Memory-efficient but slower for many unique pairs.

    **What progress measures**: the distance from the start, as a fraction of
    the start-to-goal distance. It is not the distance left to the goal. On a
    branched maze a sample in a wrong arm as far from the start as the goal
    also reads 1.0, so progress alone does not establish arrival; pair it
    with :func:`distance_to_region` to the goal.

    **Edge cases**:

    - start_bin == goal_bin: 1.0 on the goal bin; NaN elsewhere, since the
      ratio is 0/0 and there is no path to make progress along
    - No path exists (graph disconnected): Returns NaN (distance = inf)
    - Invalid bins (-1): Returns NaN
    - Farther from the start than the goal is (overshoot or detour): clipped
      to 1.0. The result is a fraction of the start-to-goal path, so values
      above 1 would not be "more progress"; use ``distance_to_region`` from
      the start bin for the unclipped distance.

    Examples
    --------
    >>> # Single trial - constant start/goal
    >>> progress = path_progress(  # doctest: +SKIP
    ...     position_bins,
    ...     env,
    ...     start_bins=np.full(len(position_bins), 10),
    ...     goal_bins=np.full(len(position_bins), 50),
    ... )

    >>> # Multiple trials - construct arrays once, compute once
    >>> trials = segment_trials(position_bins, times, env, ...)  # doctest: +SKIP
    >>> start_bins, goal_bins = trials_to_region_arrays(
    ...     trials, times, env
    ... )  # doctest: +SKIP
    >>> progress = path_progress(
    ...     position_bins, env, start_bins=start_bins, goal_bins=goal_bins
    ... )  # doctest: +SKIP

    See Also
    --------
    trials_to_region_arrays : Helper to construct start/goal arrays from trials
    distance_to_region : Distance to target region over time
    """
    if metric == "euclidean":
        require_cartesian(
            env.coordinate_system,
            "path_progress(metric='euclidean')",
            why=(
                "The straight-line distance between bin centers treats them as "
                "Cartesian positions."
            ),
            fix=(
                "use metric='geodesic', which follows the connectivity graph and its "
                "physical edge lengths"
            ),
        )

    from neurospatial.environment import _validate_sample_targets

    position_bins = _validated_position_bins(
        position_bins, env, operation="path_progress()"
    )
    n_samples = len(position_bins)
    # A single start or goal applies to every sample.
    start_bins, goal_bins = (
        np.broadcast_to(
            _validate_sample_targets(
                bins,
                n_samples=n_samples,
                n_bins=env.n_bins,
                coordinates=env.coordinate_system,
                operation="path_progress()",
                argument=argument,
            )[0],
            (n_samples,),
        )
        for bins, argument in ((start_bins, "start_bins"), (goal_bins, "goal_bins"))
    )

    # Import distance functions
    from neurospatial.ops.distance import (
        euclidean_distance_matrix,
        geodesic_distance_matrix,
    )

    # Choose strategy based on environment size
    if env.n_bins < 5000:
        # Small environment - precompute full distance matrix
        if metric == "geodesic":
            dist_matrix = geodesic_distance_matrix(env.connectivity, weight="distance")
        else:  # euclidean
            dist_matrix = euclidean_distance_matrix(env.bin_centers)

        # Vectorized lookup: distance from start to current position
        distances_from_start = dist_matrix[start_bins, position_bins]

        # Vectorized lookup: total distance from start to goal
        total_distances = dist_matrix[start_bins, goal_bins]

    else:
        # Large environment - compute per-unique-pair distance fields
        from neurospatial.ops.distance import distance_field

        # Initialize arrays
        distances_from_start = np.full(n_samples, np.nan)
        total_distances = np.full(n_samples, np.nan)

        # Find unique (start, goal) pairs
        valid_mask = (start_bins >= 0) & (goal_bins >= 0)
        valid_pairs = np.column_stack([start_bins[valid_mask], goal_bins[valid_mask]])
        unique_pairs = np.unique(valid_pairs, axis=0)

        # Compute distance field for each unique pair
        for start_bin, goal_bin in unique_pairs:
            pair_mask = (start_bins == start_bin) & (goal_bins == goal_bin)

            start_dist_field = distance_field(
                env.connectivity,
                [start_bin],
                metric=metric,
                bin_centers=env.bin_centers if metric == "euclidean" else None,
            )

            distances_from_start[pair_mask] = start_dist_field[position_bins[pair_mask]]
            total_distances[pair_mask] = start_dist_field[goal_bin]

    # Compute progress
    with np.errstate(divide="ignore", invalid="ignore"):
        progress = distances_from_start / total_distances

    # With start == goal the ratio is 0/0. A sample on the goal has arrived
    # (1.0); any other sample has no defined progress (NaN), not "arrived".
    same_start_goal = start_bins == goal_bins
    progress[same_start_goal] = np.where(
        position_bins[same_start_goal] == goal_bins[same_start_goal], 1.0, np.nan
    )

    disconnected = ~np.isfinite(total_distances) | ~np.isfinite(distances_from_start)
    progress[disconnected] = np.nan

    invalid_bins = (start_bins == -1) | (goal_bins == -1) | (position_bins == -1)
    progress[invalid_bins] = np.nan

    progress = np.clip(progress, 0.0, 1.0)

    return progress


def distance_to_region(
    position_bins: NDArray[np.int_],
    env: Environment,
    *,
    target_bins: NDArray[np.int_] | int,
    metric: DistanceMetric = "geodesic",
) -> NDArray[np.float64]:
    """Compute distance from each trajectory point to target region.

    Measures instantaneous distance to a goal region over time. Useful
    for analyzing approach behavior, goal-directed navigation, and
    spatial value functions.

    Parameters
    ----------
    position_bins : NDArray[np.int_], shape (n_samples,)
        Bin indices over time, as from ``env.bin_at``; ``-1`` marks a sample
        outside the environment (NaN).
    env : Environment
        Spatial environment.
    target_bins : NDArray[np.int_] or int
        Target bin specification:
        - int, or a one-element array such as ``env.bin_at([[x, y]])``: one
          target bin for every sample, in ``[0, n_bins)``; ``-1`` (the outside
          marker of ``bin_at``) is refused.
        - NDArray[np.int_], shape (n_samples,): Target bin at each timepoint;
          ``-1`` marks a sample with no target, which gives NaN.
          This allows dynamic goals.
    metric : {'geodesic', 'euclidean'}, default='geodesic'
        Distance metric.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Distance from current position to target at each timepoint.
        Units match environment coordinates (e.g., cm). NaN where the
        position or the target is ``-1``.

    Raises
    ------
    TypeError
        If ``position_bins`` or ``target_bins`` is not integer bin indices.
    BinIndexOutOfRangeError
        If a bin index is outside ``[0, n_bins)`` and is not ``-1``.
    CartesianOnlyError
        If ``metric="euclidean"`` and ``env`` is polar. Use
        ``metric="geodesic"``.

    Notes
    -----
    **Performance strategy (automatic)**:

    - **Scalar targets**: Delegates to ``env.distance_to()`` for efficiency.
    - **Array targets (small env)**: Precomputes distance matrix.
    - **Array targets (large env)**: Computes per-unique-target distance fields.

    Examples
    --------
    >>> goal_bins = env.bins_in_region("reward_zone")  # doctest: +SKIP
    >>> dist = distance_to_region(
    ...     position_bins, env, target_bins=goal_bins[0]
    ... )  # doctest: +SKIP

    See Also
    --------
    path_progress : Normalized progress along path
    cost_to_goal : Distance with terrain/learned cost
    distance_to_boundary : Distance to the environment's edge or a region's
        boundary
    distance_to_reward : Distance to the reward location at each sample
    neurospatial.ops.nearest_boundary : Exact distance and direction to walls
    """
    if metric == "euclidean":
        require_cartesian(
            env.coordinate_system,
            "distance_to_region(metric='euclidean')",
            why=(
                "The straight-line distance between bin centers treats them as "
                "Cartesian positions."
            ),
            fix=(
                "use metric='geodesic', which follows the connectivity graph and its "
                "physical edge lengths"
            ),
        )

    from neurospatial.environment import _validate_sample_targets

    position_bins = _validated_position_bins(
        position_bins, env, operation="distance_to_region()"
    )
    targets, per_sample = _validate_sample_targets(
        target_bins,
        n_samples=len(position_bins),
        n_bins=env.n_bins,
        coordinates=env.coordinate_system,
        operation="distance_to_region()",
        argument="target_bins",
    )

    if not per_sample:
        (target,) = targets
        dist_field = env.distance_to(bins=target, metric=metric)
        distances = dist_field[position_bins].astype(np.float64)

        invalid_mask = position_bins == -1
        distances[invalid_mask] = np.nan
        return distances

    else:
        n_samples = len(position_bins)

        if env.n_bins < 5000:
            from neurospatial.ops.distance import (
                euclidean_distance_matrix,
                geodesic_distance_matrix,
            )

            if metric == "geodesic":
                dist_matrix = geodesic_distance_matrix(
                    env.connectivity, weight="distance"
                )
            else:
                dist_matrix = euclidean_distance_matrix(env.bin_centers)

            distances = dist_matrix[position_bins, targets]
        else:
            unique_targets = np.unique(targets)
            unique_targets = unique_targets[unique_targets != -1]

            distances = np.full(n_samples, np.nan, dtype=np.float64)

            for target in unique_targets:
                mask = targets == target
                if np.any(mask):
                    dist_field = env.distance_to(bins=target, metric=metric)
                    distances[mask] = dist_field[position_bins[mask]]

        invalid_mask = (position_bins == -1) | (targets == -1)
        distances[invalid_mask] = np.nan

        return distances


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
    reward_times : NDArray[np.float64], shape (n_events,)
        Reward event timestamps (seconds).
    reward_positions : NDArray[np.float64], shape (n_events, n_dims), optional
        Reward locations in environment coordinates. If None, each reward
        location is read from the tracked positions at its reward time (see
        Notes); a reward whose location cannot be read that way has an
        unknown location. A NaN row is an unknown location.
    metric : {'geodesic', 'euclidean'}, default='geodesic'
        Distance metric:

        - 'geodesic': Shortest-path distance through the connectivity graph,
          respecting walls and holes, between the center of the position's
          bin and the center of the reward's bin. It is quantized to bins:
          within one bin it is constant and it steps by about one bin
          spacing between neighboring bins.
        - 'euclidean': Exact straight-line distance between the position
          and the reward location. Ignores walls.
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
        If ``times`` is not ascending when reward positions are inferred
        (no ``reward_positions`` given) or ``break_before`` is given.
        If ``max_gap`` is not positive, or ``epochs`` or ``break_before`` is
        malformed.

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

    This function computes distance fields for each unique reward location,
    which is O(V log V + E) per location using Dijkstra's algorithm. For
    many unique reward locations, consider caching or using Euclidean
    distances.

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
    distance_to_region : Distance to a target bin or region
    neurospatial.ops.nearest_boundary : Exact distance and direction to walls
    """
    from neurospatial._intervals import as_intervals, validate_max_gap
    from neurospatial._observation import validate_break_before
    from neurospatial._validation import validate_finite
    from neurospatial.ops.distance import distance_field

    require_cartesian(
        env.coordinate_system,
        "distance_to_reward()",
        why="It reads positions as the animal's allocentric positions, located "
        "in the environment's bins.",
        fix="use the allocentric environment the positions were recorded in",
    )

    # A non-finite time silently corrupts interpolation and searchsorted
    # ordering; a NaN position is a missing sample and gives NaN distance.
    times = validate_finite(times, name="times")
    reward_times = validate_finite(reward_times, name="reward_times")
    if reward_times.ndim != 1:
        raise ValueError(
            f"reward_times must be 1-D (n_events,), got shape {reward_times.shape}."
        )
    positions = validate_finite(
        np.asarray(positions, dtype=np.float64), name="positions", allow_nan=True
    )
    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)

    n_samples = len(times)

    if len(positions) != n_samples:
        raise ValueError(
            f"positions and times must have same length. "
            f"Got positions: {len(positions)}, times: {n_samples}."
        )
    validate_max_gap(max_gap)
    breaks = validate_break_before(break_before, n_samples, call="distance_to_reward")
    resolved_epochs = as_intervals(epochs, name="epochs")

    # Handle empty inputs
    if n_samples == 0:
        return np.array([], dtype=np.float64)

    if len(reward_times) == 0:
        return np.full(n_samples, np.nan, dtype=np.float64)

    if (reward_positions is None or breaks is not None) and np.any(np.diff(times) < 0):
        raise ValueError(
            "times must be sorted in ascending order to interpolate "
            "reward positions or to assign rewards to recordings.\n"
            "  WHY: reward locations are interpolated between neighbouring "
            "samples and break_before splits the samples in time order; "
            "unsorted times yield wrong reward positions and a corrupted "
            "distance regressor.\n"
            "  Fix: sort (times, positions) by time before calling, or "
            "pass explicit reward_positions (without break_before) to skip "
            "interpolation."
        )

    # Samples outside epochs have no distance, as they have no speed in
    # compute_speed. Each sample is tested as the instant [t, t + ulp).
    if resolved_epochs is None:
        sample_in_epochs = np.ones(n_samples, dtype=bool)
    else:
        sample_in_epochs = times_in_intervals(resolved_epochs, times)

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
            raise ValueError(
                f"reward_positions and reward_times must have same length. "
                f"Got reward_positions: {len(reward_positions)}, "
                f"reward_times: {len(reward_times)}."
            )
        _warn_rewards_outside_recording(times, reward_times)
    else:
        reward_positions = _infer_reward_positions(
            times,
            positions,
            reward_times,
            max_gap=max_gap,
            break_before=breaks,
        )

    # Map positions to bins (NaN rows are -1)
    position_bins = env.bin_at(positions)

    # Map reward positions to bins (unknown, NaN locations are -1)
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

    # Find relevant reward for each sample based on mode
    if mode == "nearest":
        # Find nearest reward in time (before or after)
        # Use searchsorted to find insertion points
        idx_right = np.searchsorted(sorted_reward_times, times, side="left")
        idx_left = idx_right - 1

        has_left = idx_left >= 0
        has_right = idx_right < n_rewards
        if recordings is not None:
            has_left &= _same_recording(idx_left, *recordings)
            has_right &= _same_recording(idx_right, *recordings)

        # Compute time distances to left and right rewards
        dist_left = np.full(n_samples, np.inf)
        dist_right = np.full(n_samples, np.inf)
        dist_left[has_left] = times[has_left] - sorted_reward_times[idx_left[has_left]]
        dist_right[has_right] = (
            sorted_reward_times[idx_right[has_right]] - times[has_right]
        )

        # Select nearest
        use_left = dist_left <= dist_right
        relevant_reward_idx = np.where(use_left, idx_left, idx_right)

        # No eligible reward on either side
        no_reward = ~has_left & ~has_right
        relevant_reward_idx = np.clip(relevant_reward_idx, 0, n_rewards - 1)

    elif mode == "last":
        # Find most recent reward before each sample
        # searchsorted gives index of first reward > time, so idx-1 is last reward <= time
        idx = np.searchsorted(sorted_reward_times, times, side="right") - 1
        relevant_reward_idx = idx
        no_reward = idx < 0
        if recordings is not None:
            no_reward |= ~_same_recording(idx, *recordings)

    elif mode == "next":
        # Find next upcoming reward after each sample
        # searchsorted gives index of first reward >= time
        idx = np.searchsorted(sorted_reward_times, times, side="left")
        relevant_reward_idx = idx
        no_reward = idx >= n_rewards
        if recordings is not None:
            no_reward |= ~_same_recording(idx, *recordings)

    else:
        raise ValueError(f"mode must be 'nearest', 'last', or 'next', got '{mode}'.")

    # Clip indices to valid range for array access
    relevant_reward_idx = np.clip(relevant_reward_idx, 0, len(sorted_reward_bins) - 1)

    # Get reward bins for each sample
    target_bins = sorted_reward_bins[relevant_reward_idx]

    distances = np.full(n_samples, np.nan, dtype=np.float64)
    if metric == "euclidean":
        # The exact straight-line distance between the position and the
        # reward location, not between their bin centers.
        distances = np.linalg.norm(
            positions - sorted_reward_positions[relevant_reward_idx], axis=1
        )
    else:
        # Geodesic distance between bin centers, one distance field per
        # distinct reward bin.
        unique_targets = np.unique(target_bins[~no_reward])
        for target_bin in unique_targets[unique_targets >= 0]:
            mask = (target_bins == target_bin) & ~no_reward
            dist_field = distance_field(env.connectivity, [int(target_bin)])
            sample_bins = position_bins[mask]
            distances[mask] = np.where(
                sample_bins >= 0,
                dist_field[np.clip(sample_bins, 0, len(dist_field) - 1)],
                np.nan,
            )

    # Set NaN for samples with no relevant reward
    distances[no_reward] = np.nan

    # Set NaN for invalid trajectory bins
    distances[position_bins < 0] = np.nan

    # Set NaN for invalid or unknown reward bins
    invalid_reward_mask = target_bins < 0
    distances[invalid_reward_mask & ~no_reward] = np.nan

    # Set NaN for samples outside epochs
    distances[~sample_in_epochs] = np.nan

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
    analyzed samples, not where a reward was. Any other reward has an unknown (NaN) location, and one
    ``UserWarning`` says how many and why.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Finite, non-decreasing sample times in seconds.
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

    from neurospatial._intervals import gap_endpoints, interval_valid_mask
    from neurospatial._stacklevel import _external_stacklevel

    n_samples, n_dims = positions.shape
    observed: NDArray[np.bool_] = np.asarray(np.isfinite(positions).all(axis=1))
    in_span = (reward_times >= times[0]) & (reward_times <= times[-1])
    # Last sample at or before each reward; the interval it starts holds the
    # reward when the reward is not on an observed sample.
    k = np.clip(np.searchsorted(times, reward_times, side="right") - 1, 0, None)
    on_sample = in_span & (times[k] == reward_times) & observed[k]
    between = in_span & ~on_sample
    known = on_sample.copy()
    reasons = {"outside the recorded span": ~in_span}
    if n_samples > 1:
        usable = interval_valid_mask(
            times, max_gap=max_gap, break_before=break_before, sample_valid=observed
        )
        start, end, linked = gap_endpoints(observed)
        j = np.clip(k, 0, n_samples - 2)
        known |= between & usable[j]
        n_breaks = np.cumsum(
            np.r_[False, np.zeros(n_samples - 1, dtype=bool)]
            if break_before is None
            else np.r_[False, break_before[1:]]
        )
        crossed = n_breaks[end[j]] != n_breaks[start[j]]
        long_gap = (
            np.zeros(len(j), dtype=bool)
            if max_gap is None
            else times[end[j]] - times[start[j]] > max_gap
        )
        blocked = between & ~usable[j]
        reasons["across a declared break"] = blocked & linked[j] & crossed
        reasons[f"in a gap longer than max_gap={max_gap} s"] = (
            blocked & linked[j] & ~crossed & long_gap
        )
        reasons["before the first or after the last observed position"] = (
            blocked & ~linked[j]
        )
    else:
        reasons["at a missing (NaN) position"] = between

    located = np.full((len(reward_times), n_dims), np.nan)
    located[known] = np.column_stack(
        [
            np.interp(reward_times[known], times[observed], positions[observed, dim])
            for dim in range(n_dims)
        ]
    )
    located[on_sample] = positions[k[on_sample]]

    n_unknown = int(np.count_nonzero(~known))
    if n_unknown:
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
          track, the dead ends of its arms. This is quantized to bin centers
          and is not the exact distance to the arena's walls; for that use
          :func:`neurospatial.ops.nearest_boundary`.
        - 'region': Distance to boundary of a specific named region.
          Requires ``region_name`` parameter.
    region_name : str, optional
        Name of region for boundary_type='region'. Required when
        boundary_type='region', ignored otherwise.
    metric : {'geodesic', 'euclidean'}, default='geodesic'
        Distance metric:

        - 'geodesic': Shortest-path distance respecting environment geometry.
        - 'euclidean': Straight-line L2 distance.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Distance from each position to the center of the nearest boundary
        bin, so it is quantized to bin centers.
        Units match environment coordinates (e.g., cm).
        Returns NaN for positions outside the environment.

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

    The distance field is computed once and cached for the duration of the
    call. For repeated queries with different positions but the same
    boundary, consider caching the distance field manually using
    ``distance_field(env.connectivity, boundary_bins)``.

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
    distance_to_region : Distance to a target bin or region
    neurospatial.events.time_to_nearest_event : Temporal distance to events
    neurospatial.ops.nearest_boundary : Exact distance and direction to walls
    neurospatial.ops.compute_boundary_distances : Distance to walls along
        each direction
    """
    from neurospatial.ops.distance import distance_field
    from neurospatial.ops.mapping import _require_region_mask

    require_cartesian(
        env.coordinate_system,
        "distance_to_boundary()",
        why="It reads positions as the animal's allocentric positions, located "
        "in the environment's bins.",
        fix="use the allocentric environment the positions were recorded in",
    )

    # Input validation
    positions = np.asarray(positions, dtype=np.float64)
    n_samples = len(positions)

    if n_samples == 0:
        return np.array([], dtype=np.float64)

    if boundary_type == "region" and region_name is None:
        raise ValueError(
            "region_name is required when boundary_type='region'. "
            "Provide the name of the region to compute distance to."
        )

    # Find boundary bins based on boundary_type
    if boundary_type == "edge":
        boundary_bins = env.boundary_bins

    elif boundary_type == "region":
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

    else:
        raise ValueError(
            f"boundary_type must be 'edge' or 'region', got '{boundary_type}'."
        )

    if len(boundary_bins) == 0:
        raise ValueError(
            f"No boundary bins found for boundary_type='{boundary_type}'"
            + (f", region_name='{region_name}'" if region_name else "")
            + ". Check that the environment has valid boundaries."
        )

    # Compute distance field from boundary bins
    dist_field = distance_field(
        env.connectivity,
        list(boundary_bins),
        metric=metric,
        bin_centers=env.bin_centers if metric == "euclidean" else None,
    )

    # Map positions to bins
    position_bins = env.bin_at(positions)

    # Look up distances
    distances = np.full(n_samples, np.nan, dtype=np.float64)
    valid_bins = position_bins >= 0
    distances[valid_bins] = dist_field[position_bins[valid_bins]]

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


def cost_to_goal(
    position_bins: NDArray[np.int_],
    env: Environment,
    *,
    goal_bins: NDArray[np.int_] | int,
    cost_map: NDArray[np.float64] | None = None,
    terrain_difficulty: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute cost-to-goal incorporating terrain difficulty and learned costs.

    Unlike simple geometric distance, cost-to-goal accounts for:

    - Movement difficulty (e.g., barriers, narrow passages)
    - Learned avoidance (e.g., punishment zones, risky areas)

    This is the spatial equivalent of a value function in RL.

    Parameters
    ----------
    position_bins : NDArray[np.int_], shape (n_samples,)
        Bin indices over time, as from ``env.bin_at``; ``-1`` marks a sample
        outside the environment (NaN).
    env : Environment
        Spatial environment.
    goal_bins : NDArray[np.int_] or int
        Target bin(s). One bin for every sample (an int or a one-element
        array, in ``[0, n_bins)``), or an array of shape ``(n_samples,)``
        (dynamic goal, where ``-1`` marks a sample with no goal).
    cost_map : NDArray[np.float64], shape (n_bins,), optional
        Extra cost per unit length of travel through each bin, dimensionless
        and ``>= 0``: ``0`` is plain travel and ``9`` makes moving through
        the bin cost ten times its length. Default: no extra cost.
    terrain_difficulty : NDArray[np.float64], shape (n_bins,), optional
        Movement difficulty multiplier per bin, dimensionless and ``> 0``.
        Default: 1.0 (uniform).

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Cost of the cheapest path from the current bin to the goal bin, in
        the environment's length units (e.g. cm) weighted by the cost rate.
        With neither ``cost_map`` nor ``terrain_difficulty`` it is the
        geodesic distance. Lower cost = preferred path. NaN where the
        position or the goal is ``-1``; ``inf`` where no path exists.

    Raises
    ------
    ValueError
        If ``cost_map`` or ``terrain_difficulty`` does not have shape
        ``(n_bins,)`` or holds a non-finite value, a ``cost_map`` entry is
        negative, or a ``terrain_difficulty`` entry is not positive.
    TypeError
        If ``position_bins`` or ``goal_bins`` is not integer bin indices.
    BinIndexOutOfRangeError
        If a bin index is outside ``[0, n_bins)`` and is not ``-1``.

    Notes
    -----
    Each bin has a cost rate per unit length,
    ``rate = terrain_difficulty * (1 + cost_map)``. Travel cost is the
    integral of the rate along the path, with the rate varying linearly
    between bin centers, so a graph edge of length ``d`` between bins ``u``
    and ``v`` costs ``d * (rate[u] + rate[v]) / 2``. Because cost is per unit
    length, the result does not depend on the bin size: on a 100 cm corridor
    with ``cost_map = np.ones(n_bins)``, a bin 80 cm from the goal costs 160
    at every bin size, and a 20 cm stretch with ``cost_map = 9`` adds
    ``9 * 20 = 180``. A uniform ``cost_map`` only rescales every cost, so it
    never changes which path is cheapest.

    Examples
    --------
    >>> # Simple: uniform cost (equivalent to geodesic distance)
    >>> cost = cost_to_goal(position_bins, env, goal_bins=goal_bin)  # doctest: +SKIP

    >>> # Learned avoidance: crossing a punished bin costs ten times its length
    >>> cost_map = np.zeros(env.n_bins)  # doctest: +SKIP
    >>> cost_map[punishment_bins] = 9.0  # doctest: +SKIP
    >>> cost = cost_to_goal(
    ...     position_bins, env, goal_bins=goal_bin, cost_map=cost_map
    ... )  # doctest: +SKIP

    See Also
    --------
    distance_to_region : Simple geometric distance
    """
    from neurospatial.environment import _validate_sample_targets
    from neurospatial.ops.distance import distance_field

    position_bins = _validated_position_bins(
        position_bins, env, operation="cost_to_goal()"
    )
    goals, per_sample = _validate_sample_targets(
        goal_bins,
        n_samples=len(position_bins),
        n_bins=env.n_bins,
        coordinates=env.coordinate_system,
        operation="cost_to_goal()",
        argument="goal_bins",
    )

    # Case 1: No cost modifications
    if cost_map is None and terrain_difficulty is None:
        return distance_to_region(
            position_bins,
            env,
            target_bins=goals if per_sample else goals[0],
            metric="geodesic",
        )

    # Case 2: per-bin cost rate per unit length, integrated along each edge
    # with the trapezoidal rule.
    extra = _validated_bin_values(
        cost_map, env, name="cost_map", default=0.0, positive=False
    )
    difficulty = _validated_bin_values(
        terrain_difficulty, env, name="terrain_difficulty", default=1.0, positive=True
    )
    rate = difficulty * (1.0 + extra)
    g_weighted = env.connectivity.copy()
    for u, v, data in g_weighted.edges(data=True):
        data["weight"] = data["distance"] * (rate[u] + rate[v]) / 2.0

    if not per_sample:
        dist_field = distance_field(g_weighted, goals, weight="weight")
        costs = dist_field[position_bins]

        costs[position_bins == -1] = np.nan

        return costs
    else:
        unique_goals = np.unique(goals)
        unique_goals = unique_goals[unique_goals != -1]

        costs = np.full(len(position_bins), np.nan, dtype=np.float64)

        for goal in unique_goals:
            mask = goals == goal
            if np.any(mask):
                dist_field = distance_field(g_weighted, [goal], weight="weight")
                costs[mask] = dist_field[position_bins[mask]]

        invalid_mask = (position_bins == -1) | (goals == -1)
        costs[invalid_mask] = np.nan

        return costs


def _validated_bin_values(
    values: ArrayLike | None,
    env: Environment,
    *,
    name: str,
    default: float,
    positive: bool,
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
                f"cost_to_goal() got {name} with shape {array.shape}, but the "
                f"environment has {env.n_bins} bins.",
                why=f"{name} holds one value per bin.",
                fix=f"pass an array of shape ({env.n_bins},), for example "
                f"np.full(env.n_bins, {default})",
            )
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(
            _format_error(
                f"cost_to_goal() got {np.count_nonzero(~np.isfinite(array))} "
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
                f"cost_to_goal() got {np.count_nonzero(bad)} value(s) in {name} "
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


def time_to_goal(
    times: NDArray[np.float64],
    trials: list[Trial],
) -> NDArray[np.float64]:
    """Compute time remaining until goal arrival for each trial.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps for entire session.
    trials : list[Trial]
        Trial segmentation from segment_trials(). Only trials that reached
        their goal (``trial.success``) are included; timed-out and censored
        trials, whose arrival was never observed, are not.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Time in seconds until goal arrival, for samples with
        ``start_time <= t <= end_time`` of a successful trial. NaN outside
        successful trials.

    Warns
    -----
    UserWarning
        If there are trials but none contains a sample of ``times`` (usually
        trials on a different clock).

    Examples
    --------
    >>> trials = segment_trials(position_bins, times, env, ...)  # doctest: +SKIP
    >>> ttg = time_to_goal(times, trials)  # doctest: +SKIP
    >>> approach_mask = (ttg > 0) & (ttg <= 2.0)  # Last 2 seconds  # doctest: +SKIP

    See Also
    --------
    segment_trials : Segment trajectory into trials
    path_progress : Normalized progress along path
    """
    times = np.asarray(times, dtype=np.float64)
    _warn_if_no_trial_overlaps(times, trials, call="time_to_goal()", result="NaN")
    ttg = np.full(len(times), np.nan, dtype=np.float64)

    for trial in trials:
        # Only an observed arrival has a time to goal: timed-out and censored
        # trials (cut off by a gap, an epoch edge, a break or the end of the
        # recording) never reached it.
        if not trial.success:
            continue

        mask = (times >= trial.start_time) & (times <= trial.end_time)
        trial_times = times[mask]

        ttg[mask] = trial.end_time - trial_times
        ttg[mask] = np.maximum(ttg[mask], 0.0)

    return ttg


# A heading change wider than this is a reversal (within 30 degrees of
# straight back), which is neither a left nor a right turn.
_MAX_TURN_ANGLE = 5 * np.pi / 6


def graph_turn_sequence(
    env: Environment[CartesianSystem],
    position_bins: NDArray[np.int_],
    *,
    tolerance: float | None = None,
) -> str:
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
    position_bins : NDArray[np.int_], shape (n_samples,)
        Bin indices along trajectory (single trial or segment), in time
        order, as from ``env.bin_at``. The order of the samples sets the
        direction of travel. ``-1`` (outside the environment) is skipped as
        if its sample were deleted.
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
    str
        Turn sequence like "left-right-left", or empty string "" if no
        turns detected. Each turn is separated by "-". Only the first two
        coordinates set the turn direction. A heading change of more than
        150 degrees is a reversal, neither left nor right, and is left out.

    Raises
    ------
    CartesianOnlyError
        If ``env`` is polar: left and right come from cross products of
        Cartesian bin-center displacements.
    TypeError
        If ``position_bins`` is not integer bin indices.
    BinIndexOutOfRangeError
        If a bin index is outside ``[0, n_bins)`` and is not ``-1``.
    ValueError
        If ``tolerance`` is not finite and non-negative.

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
    >>> for trial in trials:  # doctest: +SKIP
    ...     mask = (times >= trial.start_time) & (times <= trial.end_time)
    ...     turn_seq = graph_turn_sequence(env, position_bins[mask])
    ...     print(f"Trial: {turn_seq}")  # "left" or "right"

    See Also
    --------
    compute_trajectory_curvature : Continuous curvature for any trajectory
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
    bins = _validated_position_bins(
        position_bins, env, operation="graph_turn_sequence()"
    )
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

    legs = np.diff(corners, axis=0)
    cross = legs[:-1, 0] * legs[1:, 1] - legs[:-1, 1] * legs[1:, 0]
    dot = np.sum(legs[:-1] * legs[1:], axis=1)
    angle = np.arctan2(cross, dot)
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


def goal_pair_direction_labels(
    times: NDArray[np.float64],
    trials: list[Trial],
) -> NDArray[np.object_]:
    """Generate per-timepoint direction labels from trial data.

    Creates labels like ``'home→goal_left'`` or ``'goal_left→home'`` from the
    trial (if any) each timepoint belongs to. Samples outside every completed
    trial are ``None``, which :func:`compute_directional_place_fields` and
    :func:`labels_to_epochs` leave out.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps for the entire session (seconds).
    trials : list[Trial]
        Trial segmentation from ``segment_trials()``. Each completed trial
        labels ``[start_time, end_time)``; trials with ``end_region=None``
        label nothing.

    Returns
    -------
    NDArray[np.object_], shape (n_samples,)
        Direction label for each timepoint using arrow notation, dtype
        ``object``; ``None`` outside every completed trial.

    Warns
    -----
    UserWarning
        If there are trials but none contains a sample of ``times`` (usually
        trials on a different clock).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import Trial, goal_pair_direction_labels
    >>> times = np.arange(6.0)
    >>> trials = [Trial(1.0, 3.0, "home", "goal_left", True)]
    >>> goal_pair_direction_labels(times, trials).tolist()
    [None, 'home→goal_left', 'home→goal_left', None, None, None]

    See Also
    --------
    segment_trials : Segment trajectory into behavioral trials
    heading_direction_labels : Direction labels based on heading angle
    laps_to_direction_labels : Direction labels from detected laps
    runs_to_direction_labels : Direction labels from detected runs
    """
    from neurospatial.behavior.segmentation import _interval_labels

    times = np.asarray(times, dtype=np.float64)
    _warn_if_no_trial_overlaps(
        times, trials, call="goal_pair_direction_labels()", result="None"
    )
    intervals = [
        (
            trial.start_time,
            trial.end_time,
            f"{trial.start_region}\u2192{trial.end_region}",
        )
        for trial in trials
        if trial.end_region is not None
    ]
    return _interval_labels(times, intervals)


def heading_direction_labels(
    times: NDArray[np.float64] | None = None,
    positions: NDArray[np.float64] | None = None,
    *,
    speed: NDArray[np.float64] | None = None,
    heading: NDArray[np.float64] | None = None,
    n_directions: int = 8,
    min_speed: float,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.object_]:
    """Generate per-timepoint direction labels from heading angle.

    Bins the direction of movement into sectors centred on evenly spaced
    allocentric directions (with the default 8: ``"0°"`` East, ``"45°"``,
    ``"90°"`` North, ..., ``"180°"`` West, ``"-135°"``, ``"-90°"`` South,
    ``"-45°"``), labels an observed animal that is not moving "stationary",
    and leaves samples with no observed movement unlabeled (``None``).

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,), optional
        Timestamps (seconds). Required if positions provided.
    positions : NDArray[np.float64], shape (n_samples, 2), optional
        2D position coordinates over time. Required if speed/heading not
        provided. A row with a non-finite coordinate is a missing position.
    speed : NDArray[np.float64], shape (n_samples,), optional
        Precomputed speed at each timepoint; NaN marks a sample that was not
        observed.
    heading : NDArray[np.float64], shape (n_samples,), optional
        Precomputed heading angle in radians; NaN where the direction of
        movement is unknown.
    n_directions : int, default=8
        Number of direction sectors, a positive integer. Default 8 creates
        45° sectors centred on East, North-East, North, and so on.
    min_speed : float
        Required, finite and ``>= 0``, in position units per second (e.g.
        ``5.0`` for 5 cm/s with positions in cm): this function sees no
        environment, so it cannot know the positions' unit. Observed samples
        slower than this, or not moving at all, are labeled "stationary".
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
        Trajectory mode only.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted. Trajectory mode only.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. Trajectory mode only.

    Returns
    -------
    NDArray[np.object_], shape (n_samples,)
        One label per timepoint, of three kinds: a direction sector named by
        its centre direction, such as ``"90°"``, where the animal moves (speed
        above zero and at least ``min_speed``); ``"stationary"`` where it was
        observed but not moving; ``None`` where it was not observed, or
        (precomputed input) moving in an unknown direction.

    Raises
    ------
    ValueError
        If ``min_speed`` is not finite and non-negative, or ``n_directions``
        is not a positive integer. In trajectory mode, if ``times`` and
        ``positions`` are empty, have different lengths, ``times`` is not
        finite and strictly increasing, ``positions`` has no x and y columns,
        or ``break_before`` is not a Boolean array with one entry per sample.
        With precomputed input, if ``speed`` and ``heading`` differ in length,
        or ``epochs``, ``break_before`` or a non-default ``max_gap`` is given.

    Warns
    -----
    UserWarning
        If a precomputed ``heading`` looks like degrees instead of radians.

    Notes
    -----
    Sector ``k`` is centred on ``k * 360 / n_directions`` degrees
    (allocentric: 0° = East, 90° = North) and spans half a sector width on
    either side, half-open: its clockwise edge belongs to it. Centring the
    sectors on the cardinal directions keeps a run along a cardinal
    direction in one label despite tracking jitter; edges on 0° would split
    an eastward run between two labels. Labels are the centre angles in
    ``(-180, 180]``, rounded to 0.1°.

    In trajectory mode the speed and heading are computed along each run of
    observed samples. A NaN row is bridged like a deleted row when valid
    neighbours are within ``max_gap``; only a longer gap, an ``epochs`` edge
    or a declared ``break_before`` splits a run, and no velocity or heading
    spans such a split.

    In trajectory mode, speed and heading come from the Gaussian-smoothed
    velocity of :func:`compute_speed`, so tracking jitter at rest is
    "stationary". The direction of movement exists only where the animal
    moves, the rule of :func:`heading_from_velocity`. A sample in no observed
    run (a missing position, an isolated sample, an excluded interval) is
    ``None``, "not observed", which :func:`labels_to_epochs` leaves out.
    Precomputed speed/heading inputs have no time gate and refuse
    ``break_before``; a NaN speed is not observed.

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

    Precomputed speed and heading (radians, 0 = East) are binned directly; a
    NaN speed was not observed:

    >>> heading_direction_labels(
    ...     speed=np.array([10.0, 1.0, np.nan]),
    ...     heading=np.array([0.3, 0.3, 0.3]),
    ...     min_speed=5.0,
    ... ).tolist()
    ['0°', 'stationary', None]

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

    See Also
    --------
    goal_pair_direction_labels : Direction labels for trialized tasks
    """
    if not (np.isfinite(min_speed) and min_speed >= 0):
        raise ValueError(
            _format_error(
                f"heading_direction_labels() got min_speed={min_speed!r}.",
                why="min_speed is a speed in position units per second; NaN "
                "would label every sample 'stationary'.",
                fix="pass a finite min_speed >= 0, e.g. 5.0 for 5 cm/s with "
                "positions in cm",
            )
        )
    if (
        isinstance(n_directions, bool)
        or not isinstance(n_directions, (int, np.integer))
        or n_directions <= 0
    ):
        raise ValueError(
            _format_error(
                f"heading_direction_labels() got n_directions={n_directions!r}.",
                why="n_directions is the number of direction sectors.",
                fix="pass a positive integer, e.g. 8 for 45° sectors",
            )
        )
    has_precomputed = speed is not None or heading is not None
    has_trajectory = times is not None or positions is not None

    if has_precomputed:
        if speed is None or heading is None:
            raise ValueError(
                "If providing precomputed kinematics, both speed and heading "
                "must be provided."
            )
        speed_arr = np.asarray(speed, dtype=np.float64)
        heading_arr = np.asarray(heading, dtype=np.float64)

        if len(speed_arr) != len(heading_arr):
            raise ValueError(
                f"Speed and heading arrays must have the same length. "
                f"Got speed: {len(speed_arr)}, heading: {len(heading_arr)}."
            )
        given = [
            name
            for name, used in (
                ("epochs", epochs is not None),
                ("max_gap", max_gap != DEFAULT_MAX_GAP),
                ("break_before", break_before is not None),
            )
            if used
        ]
        if given:
            names = ", ".join(given)
            raise ValueError(
                _format_error(
                    f"heading_direction_labels() got {names} with "
                    "precomputed speed and heading.",
                    why="Time support (epochs, max_gap, break_before) applies "
                    "to the velocity computed from times and positions; "
                    "precomputed speed and heading are labeled as given, so "
                    "it would be silently ignored.",
                    fix=f"pass times and positions with {names} instead of "
                    "speed and heading, or compute them with it "
                    f"(e.g. compute_speed(times, positions, {given[0]}=...)).",
                )
            )
        from neurospatial.encoding._validation import (
            warn_if_angles_look_like_degrees,
        )

        warn_if_angles_look_like_degrees(
            heading_arr, name="heading", context="heading_direction_labels"
        )

        observed = np.isfinite(speed_arr)

    elif has_trajectory:
        if times is None or positions is None:
            raise ValueError(
                "If providing trajectory data, both times and positions "
                "must be provided."
            )

        from neurospatial._observation import validated_observations
        from neurospatial._velocity import DEFAULT_SPEED_SMOOTHING
        from neurospatial.behavior._kinematics import (
            _movement_heading_and_speed,
            _validate_velocity_positions,
        )

        raw = positions
        times_arr, positions_arr, support = validated_observations(
            times,
            raw,
            call="heading_direction_labels",
            max_gap=max_gap,
            epochs=epochs,
            break_before=break_before,
            result="every label is None",
        )
        _validate_velocity_positions(raw, call="heading_direction_labels")
        heading_arr, speed_arr = _movement_heading_and_speed(
            times_arr,
            positions_arr,
            support.interval_mask,
            smooth_window=DEFAULT_SPEED_SMOOTHING,
        )
        observed = support.run_ids >= 0

    else:
        raise ValueError(
            "Must provide either (times, positions) or (speed, heading). "
            "Neither was provided."
        )

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
        sector = np.floor(heading_arr / width + 0.5)
    bin_indices = np.mod(np.nan_to_num(sector), n_directions).astype(np.intp)

    # The direction of movement exists only where the animal moves. An observed
    # animal that does not move is "stationary"; an unobserved sample, or a
    # moving one whose direction is unknown, stays None.
    moving = observed & (speed_arr > 0) & (speed_arr >= min_speed)
    direction_known = moving & np.isfinite(heading_arr)
    labels = np.full(len(speed_arr), None, dtype=object)
    labels[observed & ~moving] = "stationary"
    labels[direction_known] = bin_labels_arr[bin_indices[direction_known]]

    return labels


# =============================================================================
# Path Efficiency Functions
# =============================================================================


def _require_path_metric(metric: str, *, call: str) -> None:
    """Raise unless ``metric`` is a path-length metric.

    Parameters
    ----------
    metric : str
    call : str
        Name of the public function, for the error message.
    """
    if metric not in ("euclidean", "geodesic"):
        raise ValueError(
            _format_error(
                f"{call}() got metric={metric!r}; it must be "
                "'euclidean' or 'geodesic'.",
                why=(
                    "path length is summed either along straight lines or along "
                    "the environment's connectivity graph."
                ),
                fix=(
                    "use metric='euclidean' for straight-line distances, or "
                    "metric='geodesic' with env= to respect barriers."
                ),
            )
        )


def _observed_path_length(
    positions: NDArray[np.float64],
    sample_valid: NDArray[np.bool_],
    interval_mask: NDArray[np.bool_],
    *,
    metric: DistanceMetric,
    env: Environment | None,
    call: str,
) -> tuple[float, bool]:
    """Path length over usable intervals, and whether every interval counted.

    Parameters
    ----------
    positions : ndarray, shape (n_samples, n_dims)
    sample_valid : ndarray of bool, shape (n_samples,)
        ``ObservationSupport.sample_valid`` of the same samples.
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        ``ObservationSupport.interval_mask`` of the same samples.
    metric : {"euclidean", "geodesic"}
    env : Environment or None
        Required for ``metric="geodesic"``.
    call : str
        Public function name, used in messages.

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
    if metric == "euclidean":
        steps = np.linalg.norm(ends - starts, axis=1)
    else:
        if env is None:
            raise ValueError(
                "env parameter is required when metric='geodesic'. "
                "Provide the Environment instance, or use metric='euclidean' "
                "for straight-line distances."
            )
        from neurospatial.behavior._environment_paths import EnvironmentPaths

        require_cartesian(
            env.coordinate_system,
            f"{call}(metric='geodesic')",
            why=(
                "positions are allocentric Cartesian positions, located in the "
                "environment's bins."
            ),
            fix="use an allocentric environment, or metric='euclidean'",
        )
        steps = EnvironmentPaths(env).between(starts, ends)
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
        ``"euclidean"`` sums the straight steps between samples.
        ``"geodesic"`` sums the shortest path inside the environment between
        consecutive samples: the same straight step whenever it stays inside
        the environment, and the route around the walls when it does not (a
        bridged gap whose straight line crosses a wall). See Notes.
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

    _require_path_metric(metric, call="traveled_path_length")
    if metric == "geodesic" and env is None:
        raise ValueError(
            "env parameter is required when metric='geodesic'. "
            "Provide the Environment instance, or use metric='euclidean' "
            "for straight-line distances."
        )

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
        metric=metric,
        env=env,
        call="traveled_path_length",
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
        Spatial environment with connectivity graph.
    start : NDArray[np.float64], shape (n_dims,)
        Start position in allocentric frame.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position in allocentric frame.
    metric : {"geodesic", "euclidean"}, default="geodesic"
        ``"euclidean"`` is the straight-line distance. ``"geodesic"`` is the
        length of the shortest path that stays inside the environment: the
        straight line when it stays inside, otherwise the route around the
        walls (see Notes).

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
    bin center. When the straight segment from start to goal stays inside the
    environment (probed at a quarter of the shortest bin-graph edge), it is
    the Euclidean distance, so on an open field ``metric="geodesic"`` and
    ``metric="euclidean"`` agree. Otherwise the bin-graph shortest path is
    pulled taut into straight segments that stay inside the environment; its
    waypoints are bin centers, so it rounds each corner about half a bin
    inside the wall and can exceed the true shortest path by up to about one
    bin size per corner. The bin-graph distance itself
    (``env.distance_between``) runs along grid steps and overestimates an
    oblique straight line by up to ~8% on an 8-connected grid.

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
    _require_path_metric(metric, call="shortest_path_length")
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

    if metric == "euclidean":
        return float(np.linalg.norm(goal - start))

    require_cartesian(
        env.coordinate_system,
        "shortest_path_length(metric='geodesic')",
        why=(
            "start and goal are allocentric Cartesian positions, located in the "
            "environment's bins."
        ),
        fix=(
            "use an allocentric environment, or metric='euclidean' for the "
            "straight-line distance from start to goal"
        ),
    )
    from neurospatial.behavior._environment_paths import EnvironmentPaths

    paths = EnvironmentPaths(env)
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


def angular_efficiency(
    times: ArrayLike,
    positions: ArrayLike,
    goal: ArrayLike,
    *,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> float:
    """Compute how directly each step heads toward the goal.

    For every observed step, the angle between the step's direction and the
    direction from the step's start to the goal is measured; the result is
    ``1 - mean(angle) / pi``. A run straight toward the goal scores 1, a run
    straight away scores 0, and steps perpendicular to the goal direction
    score 0.5.

    Parameters
    ----------
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples, n_dims) or (n_samples,)
        Position coordinates in allocentric frame. Rows with a non-finite
        coordinate are missing.
    goal : ArrayLike, shape (n_dims,)
        Goal position, in the coordinates and units of ``positions``.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : None, (start, stop) or array-like of shape (n_epochs, 2), optional
        Half-open ``[start, stop)`` windows in seconds, on the clock of
        ``times``. A step must lie inside one window.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    float
        Angular efficiency in ``[0, 1]``. NaN when no step has a defined
        direction toward the goal: fewer than two observed samples, the
        animal never moves, or every step starts at the goal.

    Raises
    ------
    ValueError
        If the inputs are empty or malformed, or ``goal`` does not have one
        coordinate per position dimension.

    Notes
    -----
    A step joins two consecutive valid samples. A NaN row is bridged like a
    deleted row when its valid neighbours are no more than ``max_gap`` apart;
    only a longer gap, an ``epochs`` edge or a declared break leaves the step
    out. A step of zero length has no direction and is left out, as is a
    step that starts at the goal (no goal direction).

    The angle between step ``s`` and goal direction ``g`` is computed as
    ``2 * arctan2(|u - v|, |u + v|)`` with unit vectors ``u = s / |s|`` and
    ``v = g / |g|``. In 2-D it equals ``|wrap(step_heading - goal_bearing)|``,
    and it is defined the same way for 1-D and 3-D positions.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import angular_efficiency
    >>> times = np.array([0.0, 0.1, 0.2])
    >>> positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    >>> angular_efficiency(times, positions, np.array([10.0, 0.0]))  # toward
    1.0
    >>> angular_efficiency(times, positions, np.array([-10.0, 0.0]))  # away
    0.0
    """
    from neurospatial._observation import validated_observations

    _, positions_arr, support = validated_observations(
        times,
        positions,
        call="angular_efficiency",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="the angular efficiency is NaN",
    )
    starts, ends = _observed_steps(
        positions_arr, support.sample_valid, support.interval_mask
    )
    return _angular_efficiency(starts, ends, goal, call="angular_efficiency")


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
    from neurospatial._intervals import gap_endpoints

    first, last, _ = gap_endpoints(sample_valid)
    counted = interval_mask & sample_valid[1:]
    return positions[first[counted]], positions[last[counted]]


def _angular_efficiency(
    starts: NDArray[np.float64],
    ends: NDArray[np.float64],
    goal: ArrayLike,
    *,
    call: str,
) -> float:
    """``1 - mean(angle) / pi`` over steps with a defined goal-relative angle.

    Parameters
    ----------
    starts, ends : ndarray, shape (n_steps, n_dims)
        Step endpoints.
    goal : array-like, shape (n_dims,)
    call : str
        Public function name, used in the goal-dimension error.
    """
    to_goal = _goal_vector(starts, np.asarray(goal, dtype=float), call=call)
    step = ends - starts
    step_norm = np.linalg.norm(step, axis=1)
    goal_norm = np.linalg.norm(to_goal, axis=1)
    defined = (step_norm > 0) & (goal_norm > 0)
    if not defined.any():
        return np.nan
    u = step[defined] / step_norm[defined, None]
    v = to_goal[defined] / goal_norm[defined, None]
    angle = 2.0 * np.arctan2(
        np.linalg.norm(u - v, axis=1), np.linalg.norm(u + v, axis=1)
    )
    return float(1.0 - np.mean(angle / np.pi))


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
        Distance metric for path lengths and for finding the closest
        approach, as in :func:`compute_path_efficiency`.
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
    _require_path_metric(metric, call=call)
    times_arr, positions_arr, support = validated_observations(
        times,
        positions,
        call=call,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="no path was observed",
    )
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

    measured = support.sample_valid.copy()
    if support.epochs is not None:
        measured &= times_in_intervals(support.epochs, times_arr)
    segment_results = []
    start: int | None = None
    for index, subgoal in enumerate(subgoal_arr):
        distances = _goal_distances(
            positions_arr, subgoal, metric=metric, env=env, call=call
        )
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
            metric=metric,
            env=env,
            call=call,
        )
        segment_results.append(
            PathEfficiencyResult(
                traveled_length=traveled,
                shortest_length=float(distances[start]),
                time_efficiency=None,
                angular_efficiency=_angular_efficiency(
                    *_observed_steps(
                        positions_arr[segment],
                        support.sample_valid[segment],
                        support.interval_mask[start:arrival],
                    ),
                    subgoal,
                    call=call,
                ),
                start_position=positions_arr[start],
                goal_position=subgoal,
                metric=metric,
                fully_observed=complete,
            )
        )
        start = arrival

    return SubgoalEfficiencyResult(
        segment_results=tuple(segment_results), subgoal_positions=subgoal_arr
    )


def compute_path_efficiency(
    env: Environment,
    times: ArrayLike,
    positions: ArrayLike,
    goal: ArrayLike,
    *,
    metric: DistanceMetric = "geodesic",
    reference_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> PathEfficiencyResult:
    """Compute comprehensive path efficiency metrics.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    positions : ArrayLike, shape (n_samples, n_dims) or (n_samples,)
        Position coordinates in allocentric frame. A 1-D array is one
        coordinate (for example a linearized track). Rows with a non-finite
        coordinate are missing.
    goal : ArrayLike, shape (n_dims,)
        Goal position.
    metric : {"geodesic", "euclidean"}, default="geodesic"
        Distance metric for both lengths. ``"geodesic"`` measures shortest
        paths inside the environment: ``traveled_length`` as in
        :func:`traveled_path_length` and ``shortest_length`` as in
        :func:`shortest_path_length`, so a straight run on an open field is
        perfectly efficient at any bin size. An interval with an endpoint
        outside the environment has no geodesic length and is left out, and
        an interval between disconnected parts contributes ``inf`` (so a
        complete path with such a step has ``traveled_length=inf`` and
        ``efficiency=0.0``).
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
    epochs : None, (start, stop) or array-like of shape (n_epochs, 2), optional
        Half-open ``[start, stop)`` windows in seconds, on the clock of
        ``times``. An interval must lie inside one window.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    PathEfficiencyResult
        All path efficiency metrics combined, measured from the first
        observed sample (the first row with finite coordinates) to ``goal``;
        missing rows before the first or after the last observed sample are
        treated as deleted, also for ``time_efficiency``. When the goal is
        unreachable under
        ``metric="geodesic"`` (it lies on a disconnected component),
        ``shortest_length`` is ``inf`` and the derived ``efficiency`` and
        ``time_efficiency`` fields are ``nan`` (not ``0.0``).

    Raises
    ------
    ValueError
        If ``metric`` is not recognized, or if the inputs are empty or
        malformed (for example ``times`` not strictly increasing, or
        ``break_before`` not a Boolean array with one entry per sample),
        ``goal`` does not have shape ``(n_dims,)`` or has a non-finite
        coordinate, or ``reference_speed`` is not a finite speed greater than
        zero. With ``metric="geodesic"``, also if the first observed position or
        ``goal`` is outside the environment, or no position is observed.
        With ``metric="euclidean"`` a
        recording with no observed position gives ``shortest_length=nan``.

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
    be biased upward. ``shortest_length`` and the wall-clock
    ``time_efficiency`` keep their meanings. When a long tracking loss makes
    a trial's efficiency ``nan``, report its ``traveled_length`` instead, or
    restrict the analysis to completely observed trials.

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
    >>> result = compute_path_efficiency(
    ...     env, times, positions, goal, metric="euclidean"
    ... )
    >>> result.traveled_length, result.efficiency
    (10.0, 1.0)
    >>> positions[5] = np.nan  # one lost frame, bridged
    >>> result = compute_path_efficiency(
    ...     env, times, positions, goal, metric="euclidean"
    ... )
    >>> result.traveled_length, result.efficiency
    (10.0, 1.0)
    >>> result = compute_path_efficiency(
    ...     env, times, positions, goal, metric="euclidean", max_gap=0.15
    ... )
    >>> result.traveled_length  # the 0.2 s gap is a tracking loss here
    8.0
    >>> bool(np.isnan(result.efficiency))  # the route is incomplete
    True
    """
    from neurospatial._observation import validated_observations

    _require_path_metric(metric, call="compute_path_efficiency")
    times_arr, positions_arr, support = validated_observations(
        times,
        positions,
        call="compute_path_efficiency",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="no path was observed",
    )
    goal = _validated_goal(goal, positions_arr.shape[1], call="compute_path_efficiency")
    if reference_speed is not None:
        _validate_reference_speed(reference_speed, call="compute_path_efficiency")

    traveled, complete = _observed_path_length(
        positions_arr,
        support.sample_valid,
        support.interval_mask,
        metric=metric,
        env=env,
        call="compute_path_efficiency",
    )
    ang_eff = _angular_efficiency(
        *_observed_steps(positions_arr, support.sample_valid, support.interval_mask),
        goal,
        call="compute_path_efficiency",
    )
    # Missing rows before the first or after the last observed sample are
    # deleted rows: the path starts and ends where the animal was observed.
    observed = np.flatnonzero(support.sample_valid)
    if observed.size:
        times_arr = times_arr[observed[0] : observed[-1] + 1]
        positions_arr = positions_arr[observed[0] : observed[-1] + 1]
    shortest = shortest_path_length(env, positions_arr[0], goal, metric=metric)

    time_eff = None
    if reference_speed is not None and len(positions_arr) >= 2:
        time_eff = time_efficiency(
            times_arr,
            positions_arr,
            reference_speed=reference_speed,
            optimal_distance=shortest,
        )

    return PathEfficiencyResult(
        traveled_length=traveled,
        shortest_length=shortest,  # may be inf if the goal is unreachable
        time_efficiency=time_eff,
        angular_efficiency=ang_eff,
        start_position=positions_arr[0],
        goal_position=goal,
        metric=metric,
        fully_observed=complete,
    )


# =============================================================================
# Goal-Directed Navigation Functions
# =============================================================================


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


def instantaneous_goal_alignment(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    goal: NDArray[np.float64],
    *,
    min_speed: float,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute instantaneous alignment between movement and goal direction.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position.
    min_speed : float
        Required, in position units per second (e.g. ``5.0`` for 5 cm/s
        with positions in cm): this function sees no environment, so it
        cannot know the positions' unit. Samples slower than this have no
        direction of movement and no alignment.
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
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Cosine of angle between velocity and goal direction, range [-1, 1].
        NaN where the animal is not moving (speed zero or below
        ``min_speed``), not observed (a missing position or a sample in no
        observed run), or exactly at the goal, where there is no direction
        to the goal.

    Raises
    ------
    ValueError
        If the inputs are empty, have different lengths, ``times`` is not
        finite and strictly increasing, ``positions`` has no x and y columns,
        ``break_before`` is not a Boolean array with one entry per sample, or
        ``goal`` has a different number of dimensions than ``positions``.

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
    >>> positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
    >>> times = np.linspace(0, 10, 21)
    >>> goal = np.array([100.0, 0.0])
    >>> alignment = instantaneous_goal_alignment(times, positions, goal, min_speed=0.0)
    >>> bool(np.nanmean(alignment) > 0.9)
    True

    A lost frame (NaN row) has no alignment:

    >>> positions[10] = np.nan
    >>> alignment = instantaneous_goal_alignment(times, positions, goal, min_speed=1.0)
    >>> bool(np.isnan(alignment[10])), bool(np.all(alignment[:10] > 0.99))
    (True, True)
    """
    from neurospatial._observation import validated_observations
    from neurospatial.behavior._kinematics import _validate_velocity_positions

    raw = positions
    times, positions, support = validated_observations(
        times,
        raw,
        call="instantaneous_goal_alignment",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="every alignment is NaN",
    )
    _validate_velocity_positions(raw, call="instantaneous_goal_alignment")
    return _goal_alignment(
        times,
        positions,
        goal,
        support,
        min_speed=min_speed,
        call="instantaneous_goal_alignment",
    )


def _goal_alignment(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    goal: NDArray[np.float64],
    support: ObservationSupport,
    *,
    min_speed: float,
    call: str,
) -> NDArray[np.float64]:
    """Cosine between movement and goal direction; NaN where not moving.

    No validation or warning: ``support`` is already resolved.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Validated, strictly increasing timestamps in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Validated positions, ``n_dims >= 2``; NaN rows are missing.
    goal : ndarray, shape (n_dims,)
        Goal position.
    support : ObservationSupport
        Usable intervals of ``(times, positions)``.
    min_speed : float
        Position units per second.
    call : str
        Public function name, used in the goal-dimension error.

    Returns
    -------
    ndarray, shape (n_samples,)
        ``cos(heading - goal_direction)`` where the animal moves, else NaN;
        also NaN at a sample exactly at the goal (no goal direction).
    """
    from neurospatial._velocity import DEFAULT_SPEED_SMOOTHING
    from neurospatial.behavior._kinematics import _movement_heading_and_speed

    # Checks the goal's dimensions before the velocity is computed.
    vec = _goal_vector(positions, goal, call=call)
    goal_heading = np.arctan2(vec[:, 1], vec[:, 0])
    heading, speed = _movement_heading_and_speed(
        times, positions, support.interval_mask, smooth_window=DEFAULT_SPEED_SMOOTHING
    )
    moving = (speed > 0) & (speed >= min_speed)
    # At the goal itself there is no direction to the goal (arctan2(0, 0)
    # would read as East and score an arrival from the East as -1).
    away_from_goal = np.any(vec != 0, axis=1)
    alignment: NDArray[np.float64] = np.where(
        moving & away_from_goal, np.cos(heading - goal_heading), np.nan
    )
    return alignment


def goal_bias(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    goal: NDArray[np.float64],
    *,
    min_speed: float,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> float:
    """Compute mean alignment toward goal over trajectory.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position.
    min_speed : float
        Required, in position units per second (e.g. ``5.0`` for 5 cm/s
        with positions in cm): this function sees no environment, so it
        cannot know the positions' unit. Samples slower than this are
        excluded from the mean.
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
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    float
        Mean goal alignment over the samples where the animal moves and is
        not exactly at the goal, range [-1, 1]. NaN if the animal is never
        moving (speed zero or below ``min_speed``) at an observed sample.

    Raises
    ------
    ValueError
        If the inputs are empty, have different lengths, ``times`` is not
        finite and strictly increasing, ``positions`` has no x and y columns,
        ``break_before`` is not a Boolean array with one entry per sample, or
        ``goal`` has a different number of dimensions than ``positions``.

    Notes
    -----
    A NaN row is bridged like a deleted row when its valid neighbours are no
    more than ``max_gap`` apart; only a longer gap, an ``epochs`` edge or a
    declared ``break_before`` splits the recording into runs, and no velocity
    or heading spans a split.

    Interpretation:
    - > 0.5: Strong goal-directed navigation
    - 0 to 0.5: Weak goal-directed with some wandering
    - < 0: Net movement away from goal

    Examples
    --------
    >>> positions = np.column_stack([np.linspace(0, 100, 101), np.zeros(101)])
    >>> times = np.linspace(0, 10, 101)
    >>> goal = np.array([100.0, 0.0])
    >>> float(goal_bias(times, positions, goal, min_speed=0.0))
    1.0

    An animal that never moves has no direction of movement, so no bias:

    >>> still = np.zeros((10, 2))
    >>> goal_bias(np.arange(10) / 10, still, goal, min_speed=0.0)
    nan
    """
    from neurospatial._observation import validated_observations
    from neurospatial.behavior._kinematics import _validate_velocity_positions

    raw = positions
    times, positions, support = validated_observations(
        times,
        raw,
        call="goal_bias",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="the goal bias is NaN",
    )
    _validate_velocity_positions(raw, call="goal_bias")
    alignment = _goal_alignment(
        times, positions, goal, support, min_speed=min_speed, call="goal_bias"
    )
    return _mean_alignment(alignment)


def _mean_alignment(alignment: NDArray[np.float64]) -> float:
    """Mean of the finite alignments, or NaN when there is none."""
    moving = alignment[np.isfinite(alignment)]
    return float(np.mean(moving)) if moving.size else np.nan


def approach_rate(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    goal: NDArray[np.float64],
    *,
    metric: DistanceMetric = "euclidean",
    env: Environment | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute rate of distance change toward goal.

    The rate at a valid sample ``k`` is the change in distance to the goal
    since the previous valid sample, divided by the time between them.
    Missing samples (non-finite coordinates) are skipped as if their rows
    were deleted, and are NaN. The rate is measured only when the time
    between the two valid samples is at most ``max_gap``, they fall inside
    one of ``epochs`` and no break is declared between them; otherwise it is
    NaN, so no rate is measured across a tracking loss or between
    recordings.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds; finite and strictly increasing.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates. Rows with a non-finite coordinate are missing.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        ``"euclidean"`` uses the straight-line distance to the goal.
        ``"geodesic"`` uses the length of the shortest path inside the
        environment from each position (not its bin center) to the goal, as
        in :func:`shortest_path_length`, so a constant-speed approach gives a
        constant rate instead of a rate quantized by the bins. Requires
        ``env``.
    env : Environment, optional
        Required when metric="geodesic". A position outside the environment
        has no geodesic distance and is treated as missing.
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
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Rate of distance change (d(distance)/dt) in units per second.
        Negative values indicate approaching the goal. The first sample, and
        every sample whose backward interval is not usable, is NaN; a single
        sample gives ``[nan]``.

    Raises
    ------
    ValueError
        If ``metric`` is not 'euclidean' or 'geodesic', ``goal`` does not have
        shape ``(n_dims,)`` or has a non-finite coordinate, or
        metric="geodesic" but env is None or the goal is outside the
        environment.
    ValueError
        If the inputs are empty, have different lengths, or ``times`` is not
        finite and strictly increasing.
    CartesianOnlyError
        For ``metric="geodesic"`` on an egocentric polar environment.

    Examples
    --------
    Run toward a goal at x = 100 at 8 Hz, then jump 48 units ahead:

    >>> import numpy as np
    >>> from neurospatial.behavior import approach_rate
    >>> times = np.arange(6) / 8
    >>> positions = np.column_stack([[0.0, 1, 2, 50, 51, 52], np.zeros(6)])
    >>> goal = np.array([100.0, 0.0])
    >>> approach_rate(times, positions, goal).tolist()
    [nan, -8.0, -8.0, -384.0, -8.0, -8.0]

    On one continuous clock the jump reads as a fast approach. If a new
    recording starts at sample 3, declare the break:

    >>> new_recording = np.array([False, False, False, True, False, False])
    >>> approach_rate(times, positions, goal, break_before=new_recording).tolist()
    [nan, -8.0, -8.0, nan, -8.0, -8.0]

    A missing position has no rate; the rate after it spans the bridged gap:

    >>> positions[1] = np.nan
    >>> approach_rate(times, positions, goal).tolist()
    [nan, nan, -8.0, -384.0, -8.0, -8.0]
    """
    from neurospatial._observation import validated_observations

    times, positions, support = validated_observations(
        times,
        positions,
        call="approach_rate",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="every approach rate is NaN",
    )
    return _approach_rate(
        times,
        positions,
        goal,
        support,
        metric=metric,
        env=env,
        call="approach_rate",
    )


def _approach_rate(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    goal: NDArray[np.float64],
    support: ObservationSupport,
    *,
    metric: DistanceMetric,
    env: Environment | None,
    call: str,
) -> NDArray[np.float64]:
    """Rate of distance change toward the goal over each usable interval.

    No observation validation or warning: ``support`` is already resolved.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Validated, strictly increasing timestamps in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Validated positions; NaN rows are missing.
    goal : ndarray, shape (n_dims,)
        Goal position.
    support : ObservationSupport
        Usable intervals of ``(times, positions)``.
    metric : {"euclidean", "geodesic"}
        Distance metric; geodesic needs ``env``.
    env : Environment or None
        Required when ``metric="geodesic"``.
    call : str
        Public function name, used in messages.

    Returns
    -------
    ndarray, shape (n_samples,)
        Distance change per second over the interval ending at each sample;
        NaN at the first sample and where that interval is not usable.
    """
    distances = _goal_distances(positions, goal, metric=metric, env=env, call=call)
    return _distance_rates(times, distances, support)


def _distance_rates(
    times: NDArray[np.float64],
    distances: NDArray[np.float64],
    support: ObservationSupport,
) -> NDArray[np.float64]:
    """Change of ``distances`` per second over each usable interval.

    A sample whose distance is not finite (a position outside the
    environment, or disconnected from the goal) is missing for the rate, so
    its gap is gated again like a deleted row's.
    """
    from neurospatial.behavior._kinematics import interval_velocity

    support = support.restricted(times, np.isfinite(distances))
    velocity = interval_velocity(times, distances[:, None], support.interval_mask)
    rates = np.full(len(distances), np.nan)
    rates[1:] = velocity[:, 0]
    # A bridged gap's rate belongs to the valid sample that ends it.
    rates[~np.isfinite(distances)] = np.nan

    return rates


def _goal_distances(
    positions: NDArray[np.float64],
    goal: ArrayLike,
    *,
    metric: str,
    env: Environment | None,
    call: str,
) -> NDArray[np.float64]:
    """Distance from each position to the goal under ``metric``.

    Parameters
    ----------
    positions : ndarray, shape (n_samples, n_dims)
        NaN rows are missing.
    goal : array-like, shape (n_dims,)
    metric : {"euclidean", "geodesic"}
        ``"geodesic"`` is the length of the shortest path inside the
        environment from the position itself (not its bin center), as in
        :func:`shortest_path_length`, so it changes continuously as the
        animal moves.
    env : Environment or None
        Required when ``metric="geodesic"``.
    call : str
        Public function name, used in messages.

    Returns
    -------
    ndarray, shape (n_samples,)
        NaN for missing positions and, under the geodesic metric, positions
        outside the environment; ``inf`` for positions disconnected from the
        goal.

    Raises
    ------
    ValueError
        If ``metric`` is not recognized, ``goal`` has the wrong shape or a
        non-finite coordinate, or (geodesic) ``env`` is None or the goal is
        outside the environment.
    CartesianOnlyError
        For ``metric="geodesic"`` on an egocentric polar environment.
    """
    _require_path_metric(metric, call=call)
    goal_arr = _validated_goal(goal, positions.shape[1], call=call)
    if metric == "euclidean":
        distances: NDArray[np.float64] = np.linalg.norm(goal_arr - positions, axis=1)
        return distances
    if env is None:
        raise ValueError(
            _format_error(
                f"{call}(metric='geodesic') needs env=.",
                why="the geodesic distance is a path inside the environment.",
                fix=(
                    "pass the Environment the positions were recorded in, or use "
                    "metric='euclidean' for straight-line distances."
                ),
            )
        )
    require_cartesian(
        env.coordinate_system,
        f"{call}(metric='geodesic')",
        why=(
            "positions and goal are allocentric Cartesian positions, located in "
            "the environment's bins."
        ),
        fix=(
            "use an allocentric environment, or metric='euclidean' for the "
            "straight-line distance to the goal"
        ),
    )
    from neurospatial.behavior._environment_paths import EnvironmentPaths

    paths = EnvironmentPaths(env)
    _require_inside(paths, {"goal": goal_arr}, call=call)
    return paths.to_point(positions, goal_arr)


def compute_goal_directed_metrics(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    goal: NDArray[np.float64],
    *,
    metric: DistanceMetric = "euclidean",
    min_speed: float | None = None,
    goal_radius: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> GoalDirectedMetrics:
    """Compute comprehensive goal-directed navigation metrics.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position.
    metric : {"geodesic", "euclidean"}, default="euclidean"
        Distance metric for the approach rate, the goal distances and goal
        arrival. ``"geodesic"`` is the shortest path inside the environment
        from each position to the goal, as in :func:`approach_rate`; a
        position outside the environment then has no distance and is left
        out of the distance summaries.
    min_speed : float or None, default=None
        Minimum speed for a defined heading, in position units per second.
        None is 5 cm/s expressed in ``env.units`` (``0.05`` for meters,
        ``50.0`` for millimeters); with units that are not a metric length,
        pass it explicitly.
    goal_radius : float, optional
        Arrival radius, in the units of ``positions``: the goal is reached at
        the first measured sample whose distance to the goal (under
        ``metric``) is at most ``goal_radius``. Finite and >= 0. None skips
        arrival detection.
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
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    GoalDirectedMetrics
        Dataclass containing all goal-directed metrics. ``goal_bias`` is NaN
        if the animal never moves at an observed sample. The distance
        summaries use the measured samples: finite coordinates, inside
        ``epochs`` when given, and (geodesic) inside the environment.
        ``min_distance_to_goal`` is the minimum distance under ``metric``
        over them; ``goal_distance_at_start`` and ``goal_distance_at_end``
        are the distances at the first and last of them. All three are NaN
        when no sample is measured. ``time_to_reach_goal`` is None without
        ``goal_radius``, the seconds from the first measured sample to the
        first measured sample within ``goal_radius``, or NaN when the goal
        is never reached.

    Raises
    ------
    ValueError
        If the inputs are empty, have different lengths, ``times`` is not
        finite and strictly increasing, ``positions`` has no x and y columns,
        ``break_before`` is not a Boolean array with one entry per sample,
        ``goal`` does not have one finite coordinate per position dimension,
        ``metric`` is not recognized, ``goal_radius`` is negative or not
        finite, or (geodesic) the goal is outside the environment.
    ValueError
        If ``min_speed`` is None and ``env.units`` is not a metric length.

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
    >>> result = compute_goal_directed_metrics(
    ...     env, times, positions, np.array([100.0, 0.0])
    ... )
    >>> bool(result.goal_bias > 0.9)
    True
    >>> result.min_distance_to_goal
    0.0
    """
    from neurospatial._observation import validated_observations
    from neurospatial.behavior._kinematics import (
        _validate_velocity_positions,
        resolve_min_speed,
    )

    raw = positions
    times, positions, support = validated_observations(
        times,
        raw,
        call="compute_goal_directed_metrics",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="the goal-directed metrics are NaN",
    )
    call = "compute_goal_directed_metrics"
    goal = _validated_goal(goal, positions.shape[1], call=call)
    min_speed = resolve_min_speed(
        env, min_speed, call="compute_goal_directed_metrics()"
    )
    _validate_velocity_positions(raw, call=call)
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
    bias = _mean_alignment(
        _goal_alignment(times, positions, goal, support, min_speed=min_speed, call=call)
    )

    distances = _goal_distances(positions, goal, metric=metric, env=env, call=call)
    rates = _distance_rates(times, distances, support)
    mean_rate = float(np.nanmean(rates)) if np.any(~np.isnan(rates)) else np.nan

    # Samples measured for the distance summaries: observed (and, with the
    # geodesic metric, inside the environment) and inside ``epochs``.
    measured = support.sample_valid & ~np.isnan(distances)
    if support.epochs is not None:
        measured &= times_in_intervals(support.epochs, times)
    rows = np.flatnonzero(measured)
    if rows.size:
        min_dist = float(np.min(distances[rows]))
        start_dist = float(distances[rows[0]])
        end_dist = float(distances[rows[-1]])
    else:
        min_dist = start_dist = end_dist = np.nan

    time_to_reach: float | None = None
    if goal_radius is not None:
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
        goal_position=goal,
        metric=metric,
    )
