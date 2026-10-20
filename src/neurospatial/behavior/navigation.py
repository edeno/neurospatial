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
        # Path progress and distance (from behavioral.py)
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
        # Path efficiency (from metrics/path_efficiency.py)
        PathEfficiencyResult,
        SubgoalEfficiencyResult,
        traveled_path_length,
        shortest_path_length,
        time_efficiency,
        angular_efficiency,
        subgoal_efficiency,
        compute_path_efficiency,
        # Goal-directed (from metrics/goal_directed.py)
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

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin
from neurospatial.behavior._region_bins import region_representative_bin
from neurospatial.geometry import require_cartesian

if TYPE_CHECKING:
    from neurospatial._observation import ObservationSupport
    from neurospatial.behavior.segmentation import Trial
    from neurospatial.environment import Environment
    from neurospatial.geometry import CartesianSystem


__all__ = [  # noqa: RUF022
    # Path progress and distance (from behavioral.py)
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
    # Path efficiency (from metrics/path_efficiency.py)
    "PathEfficiencyResult",
    "SubgoalEfficiencyResult",
    "angular_efficiency",
    "compute_path_efficiency",
    "shortest_path_length",
    "subgoal_efficiency",
    "time_efficiency",
    "traveled_path_length",
    # Goal-directed (from metrics/goal_directed.py)
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


@dataclass(frozen=True, repr=False)
class PathEfficiencyResult(ReadOnlyArraysMixin, ResultMixin):
    """Path efficiency metrics for a trajectory segment.

    Attributes
    ----------
    traveled_length : float
        Distance traveled (sum of step lengths), in environment units. From
        compute_path_efficiency(): the distance over observed intervals, NaN
        when no interval was observed. From subgoal_efficiency(): the length of
        the segment as one continuous polyline, 0.0 for a segment of fewer than
        two samples.
    shortest_length : float
        Geodesic or Euclidean distance from start to goal, in environment units.
        Is ``inf`` when the goal is unreachable under the geodesic metric (the
        goal lies on a disconnected component); it is not clamped to ``0.0``.
        From subgoal_efficiency(): 0.0 for a segment of fewer than two samples
        (no start-to-goal distance is measured).
    efficiency : float
        Ratio shortest_length / traveled_length. Is ``(0, 1]`` when the path
        ends at the goal, and can exceed 1 when it does not. Is 0.0 when a
        geodesic step is infinite. Value of 1.0 indicates optimal path taken.
        NaN if traveled_length is 0 or shortest_length is ``inf`` (unreachable
        goal); from compute_path_efficiency() also NaN if the path is not
        completely observed, and from subgoal_efficiency() for a segment of
        fewer than two samples.
    time_efficiency : float or None
        Ratio T_optimal / T_actual if reference_speed provided, else None.
    angular_efficiency : float
        1 - mean(|delta_theta|) / pi. Range [0, 1].
        Value of 1.0 indicates heading directly toward goal at all times.
        From compute_path_efficiency(): NaN unless the path is completely
        observed; 1.0 for a complete path of two samples (no turns possible).
        From subgoal_efficiency(): 1.0 for a segment of fewer than three samples.
    start_position : NDArray[np.float64]
        Start position coordinates, shape (n_dims,).
    goal_position : NDArray[np.float64]
        Goal position coordinates, shape (n_dims,).
    metric : str
        Distance metric used ("geodesic" or "euclidean").
    """

    traveled_length: float
    shortest_length: float
    efficiency: float
    time_efficiency: float | None
    angular_efficiency: float
    start_position: NDArray[np.float64]
    goal_position: NDArray[np.float64]
    metric: str

    def __post_init__(self) -> None:
        """Own read-only copies of the start and goal positions."""
        self._own_arrays()

    def is_efficient(self, threshold: float = 0.8) -> bool:
        """Return True if path efficiency exceeds threshold.

        Parameters
        ----------
        threshold : float, default=0.8
            Efficiency threshold (0 to 1). Default 0.8 means 80% efficient.

        Returns
        -------
        bool
            True if efficiency > threshold and efficiency is not NaN.
        """
        if np.isnan(self.efficiency):
            return False
        return self.efficiency > threshold

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


@dataclass(frozen=True, repr=False)
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
    segment_results : list[PathEfficiencyResult]
        Efficiency result for each segment between subgoals.
    mean_efficiency : float
        Unweighted mean efficiency across segments.
    weighted_efficiency : float
        Efficiency weighted by segment shortest_length.
        Gives more weight to longer segments.
    subgoal_positions : NDArray[np.float64]
        Positions of subgoals, shape (n_subgoals, n_dims).
    """

    segment_results: list[PathEfficiencyResult]
    mean_efficiency: float
    weighted_efficiency: float
    subgoal_positions: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Own a read-only copy of the subgoal positions."""
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


@dataclass(frozen=True, repr=False)
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
    time_to_goal : float or None
        Time until goal region entered (seconds). None if goal not reached.
    min_distance_to_goal : float
        Closest approach to goal during trajectory, in environment units.
    goal_distance_at_start : float
        Distance to goal at trajectory start, in environment units.
    goal_distance_at_end : float
        Distance to goal at trajectory end, in environment units.
    goal_position : NDArray[np.float64]
        Goal position used for computation, shape (n_dims,).
    metric : str
        Distance metric used ("geodesic" or "euclidean").
    """

    goal_bias: float
    mean_approach_rate: float
    time_to_goal: float | None
    min_distance_to_goal: float
    goal_distance_at_start: float
    goal_distance_at_end: float
    goal_position: NDArray[np.float64]
    metric: str

    def __post_init__(self) -> None:
        """Own a read-only copy of the goal position."""
        self._own_arrays()

    def is_goal_directed(self, threshold: float = 0.3) -> bool:
        """Return True if goal bias exceeds threshold.

        Parameters
        ----------
        threshold : float, default=0.3
            Goal bias threshold. Default 0.3 is a moderate threshold.

        Returns
        -------
        bool
            True if goal_bias > threshold.
        """
        return self.goal_bias > threshold

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
            "time_to_goal": self.time_to_goal,
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
        if self.time_to_goal is not None:
            lines.append(f"  Time to goal: {self.time_to_goal:.2f} s")
        else:
            lines.append("  Time to goal: not reached")
        return "\n".join(lines)


# =============================================================================
# Trial-Based Navigation Functions (from behavioral.py)
# =============================================================================


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
    ...     env, position_bins, start_bins, goal_bins
    ... )  # doctest: +SKIP
    >>> dist = distance_to_region(
    ...     position_bins, env, target_bins=goal_bins
    ... )  # doctest: +SKIP

    See Also
    --------
    path_progress : Compute normalized path progress
    distance_to_region : Distance to target region
    """
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


def path_progress(
    position_bins: NDArray[np.int_],
    env: Environment,
    *,
    start_bins: NDArray[np.int_],
    goal_bins: NDArray[np.int_],
    metric: Literal["geodesic", "euclidean"] = "geodesic",
) -> NDArray[np.float64]:
    """Compute normalized path progress from start to goal (0 -> 1).

    Fully vectorized computation over entire session. For each timepoint t:
        progress[t] = distance(start_bins[t], position_bins[t]) /
                      distance(start_bins[t], goal_bins[t])

    Parameters
    ----------
    position_bins : NDArray[np.int_], shape (n_samples,)
        Current bin index at each timepoint.
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
        Path progress from 0 to 1 at each timepoint. NaN for invalid bins (-1)
        or when no path exists between bins.

    Raises
    ------
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

    **Edge cases**:

    - start_bin == goal_bin: Returns 1.0 (already at goal)
    - No path exists (graph disconnected): Returns NaN (distance = inf)
    - Invalid bins (-1): Returns NaN
    - Detours (progress > 1): Clipped to 1.0

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
            dist_matrix = geodesic_distance_matrix(
                env.connectivity, env.n_bins, weight="distance"
            )
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

    # Handle edge cases
    same_start_goal = start_bins == goal_bins
    progress[same_start_goal] = 1.0

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
    metric: Literal["geodesic", "euclidean"] = "geodesic",
) -> NDArray[np.float64]:
    """Compute distance from each trajectory point to target region.

    Measures instantaneous distance to a goal region over time. Useful
    for analyzing approach behavior, goal-directed navigation, and
    spatial value functions.

    Parameters
    ----------
    position_bins : NDArray[np.int_], shape (n_samples,)
        Bin indices over time.
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
        Units match environment coordinates (e.g., cm).

    Raises
    ------
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
    distance_to_boundary : Distance to the environment's edge, obstacles or a
        region's boundary
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
                    env.connectivity, env.n_bins, weight="distance"
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
    metric: Literal["geodesic", "euclidean"] = "geodesic",
    mode: Literal["nearest", "last", "next"] = "nearest",
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
        dimensionality (e.g., 2D for grid environments).
    reward_times : NDArray[np.float64], shape (n_events,)
        Reward event timestamps (seconds).
    reward_positions : NDArray[np.float64], shape (n_events, n_dims), optional
        Reward locations in environment coordinates. If None, reward
        locations are inferred by interpolating ``positions`` at
        ``reward_times``.
    metric : {'geodesic', 'euclidean'}, default='geodesic'
        Distance metric:

        - 'geodesic': Shortest-path distance respecting environment geometry
          (walls, obstacles). Uses graph-based distances.
        - 'euclidean': Straight-line L2 distance. Ignores obstacles.
    mode : {'nearest', 'last', 'next'}, default='nearest'
        Which reward to use for distance computation:

        - 'nearest': Distance to temporally closest reward (before or after).
        - 'last': Distance to most recent reward. NaN before first reward.
        - 'next': Distance to next upcoming reward. NaN after last reward.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Distance from current position to the reward location at each
        sample. Units match environment coordinates (e.g., cm).
        Returns NaN when:

        - No relevant reward exists (e.g., mode="last" before first reward)
        - Position is outside environment (invalid bin)
        - Reward position is outside environment

    Raises
    ------
    ValueError
        If positions and times have different lengths.
        If reward_positions provided but doesn't match reward_times length.
        If positions dimensionality doesn't match environment.
        If ``times``, ``positions``, or ``reward_times`` contain non-finite
        values.
        If ``times`` is not ascending when reward positions are inferred
        (no ``reward_positions`` given).

    Notes
    -----
    **Performance**:

    This function computes distance fields for each unique reward location,
    which is O(V log V + E) per location using Dijkstra's algorithm. For
    many unique reward locations, consider caching or using Euclidean
    distances.

    **Interpolation**:

    When ``reward_positions`` is not provided, positions are interpolated
    using linear interpolation. This assumes smooth movement between
    samples. For teleporting animals or discontinuous tracking, provide
    explicit reward positions.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import distance_to_reward
    >>> from neurospatial.events import time_to_nearest_event
    >>> # Distance to nearest reward (default)
    >>> dist = distance_to_reward(  # doctest: +SKIP
    ...     env,
    ...     times,
    ...     positions,
    ...     reward_times,
    ...     metric="geodesic",
    ...     mode="nearest",
    ... )
    >>>
    >>> # Distance to most recent reward
    >>> dist_last = distance_to_reward(  # doctest: +SKIP
    ...     env,
    ...     times,
    ...     positions,
    ...     reward_times,
    ...     mode="last",
    ... )
    >>>
    >>> # Distance to next upcoming reward
    >>> dist_next = distance_to_reward(  # doctest: +SKIP
    ...     env,
    ...     times,
    ...     positions,
    ...     reward_times,
    ...     mode="next",
    ... )
    >>>
    >>> # GLM design matrix
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
    import warnings

    from neurospatial._validation import validate_finite
    from neurospatial.ops.distance import distance_field

    require_cartesian(
        env.coordinate_system,
        "distance_to_reward()",
        why="It reads positions as the animal's allocentric positions, located "
        "in the environment's bins.",
        fix="use the allocentric environment the positions were recorded in",
    )

    # Input validation: reject NaN/Inf in the spatial + temporal inputs.
    # A non-finite time or position silently corrupts interpolation,
    # searchsorted ordering, and the resulting distance-to-reward regressor.
    times = validate_finite(times, name="times")
    reward_times = validate_finite(reward_times, name="reward_times")
    positions = validate_finite(
        np.asarray(positions, dtype=np.float64), name="positions"
    )
    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)

    n_samples = len(times)

    if len(positions) != n_samples:
        raise ValueError(
            f"positions and times must have same length. "
            f"Got positions: {len(positions)}, times: {n_samples}."
        )

    # Handle empty inputs
    if n_samples == 0:
        return np.array([], dtype=np.float64)

    if len(reward_times) == 0:
        return np.full(n_samples, np.nan, dtype=np.float64)

    # Get or interpolate reward positions
    if reward_positions is not None:
        reward_positions = np.asarray(reward_positions, dtype=np.float64)
        if len(reward_positions) != len(reward_times):
            raise ValueError(
                f"reward_positions and reward_times must have same length. "
                f"Got reward_positions: {len(reward_positions)}, "
                f"reward_times: {len(reward_times)}."
            )
    else:
        # Interpolate positions at reward times.
        # Reward times outside the recorded session window cannot be
        # interpolated, so they are clipped to the nearest session boundary
        # (np.interp would otherwise just hold the endpoint anyway). Warn so
        # the caller knows the reward position is the session edge, not a
        # true interpolated location -- a silently clipped reward can bias a
        # distance-to-reward regressor near session boundaries.
        #
        # np.interp requires its sample points (times) to be ascending;
        # unsorted times produce silently-wrong interpolated reward positions.
        if np.any(np.diff(times) < 0):
            raise ValueError(
                "times must be sorted in ascending order to interpolate "
                "reward positions.\n"
                "  WHY: linear interpolation of reward locations assumes "
                "monotonically increasing timestamps; unsorted times yield "
                "wrong reward positions and a corrupted distance regressor.\n"
                "  Fix: sort (times, positions) by time before calling, or "
                "pass explicit reward_positions to skip interpolation."
            )
        t_min = times.min()
        t_max = times.max()
        n_out_of_range = int(np.sum((reward_times < t_min) | (reward_times > t_max)))
        if n_out_of_range > 0:
            warnings.warn(
                f"{n_out_of_range} reward time"
                f"{'' if n_out_of_range == 1 else 's'} fall outside the "
                f"session window [{t_min:g}, {t_max:g}] and were clipped to "
                "the nearest boundary before interpolating reward position(s). "
                "Pass explicit reward_positions to avoid clipping.",
                UserWarning,
                stacklevel=2,
            )
        clipped_reward_times = np.clip(reward_times, t_min, t_max)
        reward_positions = np.column_stack(
            [
                np.interp(clipped_reward_times, times, positions[:, dim])
                for dim in range(positions.shape[1])
            ]
        )

    # Map positions to bins
    position_bins = env.bin_at(positions)

    # Map reward positions to bins
    reward_bins = env.bin_at(reward_positions)

    # Sort rewards by time for efficient lookup
    sorted_idx = np.argsort(reward_times)
    sorted_reward_times = reward_times[sorted_idx]
    sorted_reward_bins = reward_bins[sorted_idx]

    # Find relevant reward for each sample based on mode
    if mode == "nearest":
        # Find nearest reward in time (before or after)
        # Use searchsorted to find insertion points
        idx_right = np.searchsorted(sorted_reward_times, times, side="left")
        idx_left = idx_right - 1

        # Compute time distances to left and right rewards
        n_rewards = len(sorted_reward_times)
        dist_left = np.full(n_samples, np.inf)
        dist_right = np.full(n_samples, np.inf)

        has_left = idx_left >= 0
        dist_left[has_left] = times[has_left] - sorted_reward_times[idx_left[has_left]]

        has_right = idx_right < n_rewards
        dist_right[has_right] = (
            sorted_reward_times[idx_right[has_right]] - times[has_right]
        )

        # Select nearest
        use_left = dist_left <= dist_right
        relevant_reward_idx = np.where(use_left, idx_left, idx_right)

        # Handle edge case: no rewards exist
        no_reward = (idx_left < 0) & (idx_right >= n_rewards)
        relevant_reward_idx = np.clip(relevant_reward_idx, 0, n_rewards - 1)

    elif mode == "last":
        # Find most recent reward before each sample
        # searchsorted gives index of first reward > time, so idx-1 is last reward <= time
        idx = np.searchsorted(sorted_reward_times, times, side="right") - 1
        relevant_reward_idx = idx
        no_reward = idx < 0

    elif mode == "next":
        # Find next upcoming reward after each sample
        # searchsorted gives index of first reward >= time
        idx = np.searchsorted(sorted_reward_times, times, side="left")
        relevant_reward_idx = idx
        no_reward = idx >= len(sorted_reward_times)

    else:
        raise ValueError(f"mode must be 'nearest', 'last', or 'next', got '{mode}'.")

    # Clip indices to valid range for array access
    relevant_reward_idx = np.clip(relevant_reward_idx, 0, len(sorted_reward_bins) - 1)

    # Get reward bins for each sample
    target_bins = sorted_reward_bins[relevant_reward_idx]

    # Compute distances
    # Find unique target bins to minimize distance field computations
    unique_targets = np.unique(target_bins[~no_reward])
    unique_targets = unique_targets[unique_targets >= 0]  # Filter invalid bins

    # Initialize distances
    distances = np.full(n_samples, np.nan, dtype=np.float64)

    # Compute distance field for each unique target
    for target_bin in unique_targets:
        # Find samples with this target
        mask = (target_bins == target_bin) & ~no_reward

        if not np.any(mask):
            continue

        # Compute distance field from this reward location
        dist_field = distance_field(
            env.connectivity,
            [int(target_bin)],
            metric=metric,
            bin_centers=env.bin_centers if metric == "euclidean" else None,
        )

        # Look up distances for trajectory bins
        sample_bins = position_bins[mask]
        valid_bins = sample_bins >= 0
        distances[mask] = np.where(
            valid_bins,
            dist_field[np.clip(sample_bins, 0, len(dist_field) - 1)],
            np.nan,
        )

    # Set NaN for samples with no relevant reward
    distances[no_reward] = np.nan

    # Set NaN for invalid trajectory bins
    distances[position_bins < 0] = np.nan

    # Set NaN for invalid reward bins
    invalid_reward_mask = target_bins < 0
    distances[invalid_reward_mask & ~no_reward] = np.nan

    return distances


def distance_to_boundary(
    env: Environment,
    positions: NDArray[np.float64],
    *,
    boundary_type: Literal["edge", "obstacle", "region"] = "edge",
    region_name: str | None = None,
    metric: Literal["geodesic", "euclidean"] = "geodesic",
) -> NDArray[np.float64]:
    """
    Compute distance to environment boundary at each position.

    Measures the instantaneous distance from trajectory positions to various
    types of boundaries: environment edges, obstacles, or named regions.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates at each sample. Must match environment
        dimensionality.
    boundary_type : {'edge', 'obstacle', 'region'}, default='edge'
        Type of boundary to compute distance to:

        - 'edge': Distance to the edge of the environment's bins: bins
          next to unoccupied space, including unvisited areas and holes.
          This is not the distance to the arena's walls; for that use
          :func:`neurospatial.ops.nearest_boundary`.
        - 'obstacle': Distance to obstacle regions. Requires environment
          to have regions with ``obstacle=True`` attribute.
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

    - **edge**: Finds bins adjacent to unoccupied space. A bin is an edge
      bin if any of its graph neighbors are not in the environment.
    - **obstacle**: Finds bins adjacent to obstacle regions. Obstacles
      are regions marked with ``obstacle=True`` during annotation.
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
    >>> # Distance to environment walls
    >>> dist_wall = distance_to_boundary(
    ...     env, positions, boundary_type="edge"
    ... )  # doctest: +SKIP
    >>>
    >>> # Distance to obstacle
    >>> dist_obstacle = distance_to_boundary(
    ...     env, positions, boundary_type="obstacle"
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
        # Edge bins: bins adjacent to unoccupied space
        # A bin is an edge bin if it has fewer neighbors than expected
        # for a fully-surrounded bin
        boundary_bins = _find_edge_bins(env)

    elif boundary_type == "obstacle":
        # Obstacle bins: bins adjacent to obstacle regions
        boundary_bins = _find_obstacle_boundary_bins(env)

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
        boundary_bins = _find_region_boundary_bins(env, region_name)

    else:
        raise ValueError(
            f"boundary_type must be 'edge', 'obstacle', or 'region', "
            f"got '{boundary_type}'."
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


def _find_edge_bins(env: Environment) -> list[int]:
    """Find bins at the edge of the environment.

    A bin is an edge bin if it has fewer than the maximum number of
    neighbors, i.e. it is adjacent to unoccupied space (the outer rim, the
    rim of an interior hole, etc.).

    The notion of "edge" is layout-dependent and falls out naturally from
    the connectivity graph:

    - **1D linearized track** (a path graph): interior bins have degree 2
      and the two end bins have degree 1, so the edge bins are exactly the
      two ends of the track. This is the meaningful boundary of a 1D track.
    - **Masked / holed 2D grids**: bins flanking a masked-out region have
      fewer neighbors than a fully surrounded interior bin and are flagged,
      so both the outer rim and any inner-hole rim are returned.

    Parameters
    ----------
    env : Environment
        Environment.

    Returns
    -------
    list[int]
        Bin (node) indices at the environment boundary. Empty if the graph
        has no nodes.

    Notes
    -----
    The degree scan is vectorized over nodes: degrees are pulled from the
    NetworkX ``DegreeView`` into arrays and the comparison is done with
    NumPy rather than a per-node Python loop.
    """
    graph = env.connectivity
    n_nodes = graph.number_of_nodes()
    if n_nodes == 0:
        return []

    # Pull (node, degree) pairs once and compare vectorized. Node ids are
    # arbitrary hashables in general, but Environment connectivity uses
    # integer bin indices, so an int64 array is safe here.
    degree_view = graph.degree()
    nodes = np.fromiter((n for n, _ in degree_view), dtype=np.int64, count=n_nodes)
    degrees = np.fromiter((d for _, d in graph.degree()), dtype=np.int64, count=n_nodes)

    max_degree = int(degrees.max())

    # A bin is an edge bin if it has fewer neighbors than a fully-surrounded
    # bin in this layout (i.e., it borders unoccupied space).
    edge_nodes = nodes[degrees < max_degree]
    edge_bins: list[int] = edge_nodes.tolist()
    return edge_bins


def _find_obstacle_boundary_bins(env: Environment) -> list[int]:
    """Find bins adjacent to obstacle regions.

    Parameters
    ----------
    env : Environment
        Environment with obstacle regions.

    Returns
    -------
    list[int]
        Bin indices at obstacle boundaries.
    """
    # Find all obstacle regions
    obstacle_regions = [
        name
        for name, region in env.regions.items()
        if hasattr(region, "obstacle") and region.obstacle
    ]

    if not obstacle_regions:
        # No obstacle regions defined - return empty
        return []

    # For environments with obstacle regions, bins adjacent to obstacles
    # are typically edge bins (they have fewer neighbors because the
    # obstacle removes connectivity). This is currently a simplification.
    # A more complete implementation would require the environment to
    # explicitly track which bins were removed due to obstacles.
    return _find_edge_bins(env)


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
        Bin indices over time.
    env : Environment
        Spatial environment.
    goal_bins : NDArray[np.int_] or int
        Target bin(s). One bin for every sample (an int or a one-element
        array, in ``[0, n_bins)``), or an array of shape ``(n_samples,)``
        (dynamic goal, where ``-1`` marks a sample with no goal).
    cost_map : NDArray[np.float64], shape (n_bins,), optional
        Per-bin traversal cost. Default: uniform cost (geodesic distance).
    terrain_difficulty : NDArray[np.float64], shape (n_bins,), optional
        Movement difficulty multiplier per bin. Default: 1.0 (uniform).

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Cost-to-goal from current position. Lower cost = preferred path.

    Examples
    --------
    >>> # Simple: uniform cost (equivalent to geodesic distance)
    >>> cost = cost_to_goal(position_bins, env, goal_bins=goal_bin)  # doctest: +SKIP

    >>> # Learned avoidance: avoid punishment zone
    >>> cost_map = np.ones(env.n_bins)  # doctest: +SKIP
    >>> cost_map[punishment_bins] = 10.0  # doctest: +SKIP
    >>> cost = cost_to_goal(
    ...     position_bins, env, goal_bins=goal_bin, cost_map=cost_map
    ... )  # doctest: +SKIP

    See Also
    --------
    distance_to_region : Simple geometric distance
    """
    from neurospatial.environment import _validate_sample_targets
    from neurospatial.ops.distance import distance_field

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

    # Case 2: Cost modifications - build weighted graph
    g_weighted = env.connectivity.copy()

    for u, v, data in g_weighted.edges(data=True):
        base_dist = data["distance"]

        if terrain_difficulty is not None:
            difficulty = (terrain_difficulty[u] + terrain_difficulty[v]) / 2.0
            base_dist *= difficulty

        if cost_map is not None:
            cost = (cost_map[u] + cost_map[v]) / 2.0
            base_dist += cost

        g_weighted[u][v]["weight"] = base_dist

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
        Trial segmentation from segment_trials(). Only successful trials
        (trial.success=True) are included.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Time in seconds until goal arrival. NaN outside successful trials.

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
    ttg = np.full(len(times), np.nan, dtype=np.float64)

    for trial in trials:
        if not trial.success:
            continue

        mask = (times >= trial.start_time) & (times <= trial.end_time)
        trial_times = times[mask]

        ttg[mask] = trial.end_time - trial_times
        ttg[mask] = np.maximum(ttg[mask], 0.0)

    return ttg


def graph_turn_sequence(
    env: Environment[CartesianSystem],
    position_bins: NDArray[np.int_],
    *,
    min_samples_per_edge: int = 50,
) -> str:
    """Extract discrete turn sequence from trajectory.

    Analyzes which bins were traversed and classifies turns based on
    changes in movement direction. Works for any Cartesian layout type.

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
        order. The order of the samples sets the direction of travel.
    min_samples_per_edge : int, default=50
        Minimum trajectory samples required on a transition to count.

    Returns
    -------
    str
        Turn sequence like "left-right-left", or empty string "" if no
        turns detected. Each turn is separated by "-". Only the first two
        coordinates set the turn direction.

    Raises
    ------
    CartesianOnlyError
        If ``env`` is polar: left and right come from cross products of
        Cartesian bin-center displacements.

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
    consecutive_bins = np.column_stack([position_bins[:-1], position_bins[1:]])
    transition_counts = Counter(map(tuple, consecutive_bins))

    valid_transitions = [
        trans
        for trans, count in transition_counts.items()
        if count >= min_samples_per_edge
    ]

    if len(valid_transitions) == 0:
        return ""

    # Get unique bins preserving order (first occurrence)
    _, unique_indices = np.unique(position_bins, return_index=True)
    unique_bins_ordered = position_bins[np.sort(unique_indices)]

    # Build set of valid bins from transitions
    valid_bins = set()
    for u, v in valid_transitions:
        valid_bins.add(u)
        valid_bins.add(v)

    # Filter to valid bins
    path_bins = np.array([b for b in unique_bins_ordered if b in valid_bins])

    if len(path_bins) < 3:
        return ""

    bin_centers = env.bin_centers
    n_dims = bin_centers.shape[1]

    # Vectorized turn angle computation
    # Get positions for all path bins
    positions = bin_centers[path_bins]  # Shape: (n_path, n_dims)

    # Compute consecutive vectors: vec1[i] = pos[i+1] - pos[i]
    vec1 = np.diff(positions, axis=0)  # Shape: (n_path-1, n_dims)
    vec2 = vec1[1:]  # vec2[i] = pos[i+2] - pos[i+1], Shape: (n_path-2, n_dims)
    vec1 = vec1[:-1]  # Align: vec1[i] = pos[i+1] - pos[i], Shape: (n_path-2, n_dims)

    # Compute norms
    vec1_norms = np.linalg.norm(vec1, axis=1)
    vec2_norms = np.linalg.norm(vec2, axis=1)

    # Mask for valid vectors (non-zero length)
    valid_mask = (vec1_norms >= 1e-10) & (vec2_norms >= 1e-10)

    if not np.any(valid_mask):
        return ""

    # Normalize vectors
    vec1_norm_safe = np.where(
        vec1_norms[:, np.newaxis] > 0, vec1_norms[:, np.newaxis], 1.0
    )
    vec2_norm_safe = np.where(
        vec2_norms[:, np.newaxis] > 0, vec2_norms[:, np.newaxis], 1.0
    )
    vec1_normalized = vec1 / vec1_norm_safe
    vec2_normalized = vec2 / vec2_norm_safe

    # Compute 2D cross product (z-component of 3D cross)
    # cross = vec1_x * vec2_y - vec1_y * vec2_x
    if n_dims >= 2:
        cross = (
            vec1_normalized[:, 0] * vec2_normalized[:, 1]
            - vec1_normalized[:, 1] * vec2_normalized[:, 0]
        )
    else:
        return ""

    # Filter to significant turns (|cross| > 0.1) and valid vectors
    significant_mask = valid_mask & (np.abs(cross) > 0.1)

    # Build turns list
    turns = []
    for i in np.where(significant_mask)[0]:
        # Counterclockwise (cross > 0) in a y-up frame is the animal's left.
        if cross[i] > 0:
            turns.append("left")
        else:
            turns.append("right")

    return "-".join(turns)


def goal_pair_direction_labels(
    times: NDArray[np.float64],
    trials: list[Trial],
) -> NDArray[np.object_]:
    """Generate per-timepoint direction labels from trial data.

    Creates labels like 'home->goal_left', 'goal_left->home', or 'other' based on
    which trial (if any) each timepoint belongs to.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps for the entire session (seconds).
    trials : list[Trial]
        Trial segmentation from ``segment_trials()``.

    Returns
    -------
    NDArray[np.object_], shape (n_samples,)
        Direction label for each timepoint using arrow notation.

    Examples
    --------
    >>> trials = segment_trials(position_bins, times, env, ...)  # doctest: +SKIP
    >>> labels = goal_pair_direction_labels(times, trials)  # doctest: +SKIP
    >>> result = compute_directional_place_fields(  # doctest: +SKIP
    ...     env, spike_times, times, positions, labels
    ... )

    See Also
    --------
    segment_trials : Segment trajectory into behavioral trials
    heading_direction_labels : Direction labels based on heading angle
    laps_to_direction_labels : Direction labels from detected laps
    runs_to_direction_labels : Direction labels from detected runs
    """
    from neurospatial.behavior.segmentation import _interval_labels

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
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.object_]:
    """Generate per-timepoint direction labels from heading angle.

    Bins heading angles into sectors (e.g., "0-45 deg", "45-90 deg", ...) and labels
    slow-moving periods as "stationary".

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,), optional
        Timestamps (seconds). Required if positions provided.
    positions : NDArray[np.float64], shape (n_samples, 2), optional
        2D position coordinates over time. Required if speed/heading not provided.
    speed : NDArray[np.float64], shape (n_samples,), optional
        Precomputed speed at each timepoint.
    heading : NDArray[np.float64], shape (n_samples,), optional
        Precomputed heading angle in radians.
    n_directions : int, default=8
        Number of direction bins. Default 8 creates 45 deg bins.
    min_speed : float
        Required, in position units per second (e.g. ``5.0`` for 5 cm/s
        with positions in cm): this function sees no environment, so it
        cannot know the positions' unit. Samples slower than this are labeled
        "stationary".
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from the analysis. ``None`` disables the gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    NDArray[np.object_], shape (n_samples,)
        Direction label for each timepoint.

    Notes
    -----
    Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``,
    not crossing a ``break_before``) is analyzed as a separate recording;
    no velocity or heading spans a pause.

    In trajectory mode, speed and heading come from the Gaussian-smoothed
    velocity of :func:`compute_speed`, so tracking jitter at rest is
    "stationary"; a sample in no observed run is "stationary" too.
    Precomputed speed/heading inputs retain their existing behavior without a
    time gate, and refuse ``break_before``.

    Examples
    --------
    Label one second of running north-east (about 63 deg, 22 cm/s), then one
    second resting in place:

    >>> import numpy as np
    >>> from neurospatial.behavior import heading_direction_labels
    >>> times = np.arange(20) / 10
    >>> step = np.r_[np.arange(10.0), np.full(10, 9.0)]
    >>> positions = np.column_stack([step, 2 * step])
    >>> labels = heading_direction_labels(times, positions, min_speed=5.0)
    >>> labels[0], labels[-1]
    ('45–90°', 'stationary')

    Precomputed speed and heading (radians, 0 = East) are binned directly:

    >>> heading_direction_labels(
    ...     speed=np.array([10.0, 1.0]), heading=np.array([0.3, 0.3]), min_speed=5.0
    ... ).tolist()
    ['0–45°', 'stationary']

    See Also
    --------
    goal_pair_direction_labels : Direction labels for trialized tasks
    """
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
        if break_before is not None:
            raise ValueError(
                _format_error(
                    "heading_direction_labels() got break_before with "
                    "precomputed speed and heading.",
                    why="Recording breaks apply to the velocity computed from "
                    "times and positions; precomputed speed and heading are "
                    "labeled as given.",
                    fix="pass times and positions with break_before instead of "
                    "speed and heading, or compute them with break_before "
                    "(e.g. compute_speed(times, positions, break_before=...)).",
                )
            )

        n_samples = len(speed_arr)

    elif has_trajectory:
        if times is None or positions is None:
            raise ValueError(
                "If providing trajectory data, both times and positions "
                "must be provided."
            )

        from neurospatial._validation import validate_times_positions

        times_arr, positions_arr = validate_times_positions(
            times, positions, call="heading_direction_labels"
        )
        n_samples = len(times_arr)

        from neurospatial._intervals import observed_interval_mask
        from neurospatial._observation import validate_break_before
        from neurospatial.ops import _velocity_heading_and_speed

        breaks = validate_break_before(
            break_before, n_samples, call="heading_direction_labels"
        )
        interval_mask = observed_interval_mask(
            times_arr, max_gap=max_gap, epochs=epochs, break_before=breaks
        )
        if n_samples == 0:
            return np.array([], dtype=object)
        if n_samples == 1:
            return np.array(["stationary"], dtype=object)

        # Samples in no observed run have NaN heading and are "stationary".
        heading_arr, speed_arr = _velocity_heading_and_speed(
            times_arr, positions_arr, interval_mask=interval_mask
        )

    else:
        raise ValueError(
            "Must provide either (times, positions) or (speed, heading). "
            "Neither was provided."
        )

    bin_edges_rad = np.linspace(-np.pi, np.pi, n_directions + 1)
    bin_edges_deg = np.linspace(-180.0, 180.0, n_directions + 1)

    # Build bin labels
    bin_labels = [
        f"{round(bin_edges_deg[i]):.0f}\u2013{round(bin_edges_deg[i + 1]):.0f}\u00b0"
        for i in range(n_directions)
    ]
    bin_labels_arr = np.array(bin_labels, dtype=object)

    # Vectorized label assignment
    # Normalize headings to [-π, π]
    heading_normalized = np.arctan2(np.sin(heading_arr), np.cos(heading_arr))

    # Digitize all headings at once
    bin_indices = np.digitize(heading_normalized, bin_edges_rad[1:], right=False)
    bin_indices = np.clip(bin_indices, 0, n_directions - 1)

    # Assign labels: stationary where speed < min_speed OR heading is undefined
    # (NaN, e.g. the padded first sample), else use the direction bin label.
    stationary_mask = (speed_arr < min_speed) | np.isnan(heading_arr)
    labels = np.where(stationary_mask, "stationary", bin_labels_arr[bin_indices])

    return labels


# =============================================================================
# Path Efficiency Functions (from metrics/path_efficiency.py)
# =============================================================================


def _polyline_length(
    positions: Any,
    *,
    metric: Literal["euclidean", "geodesic"],
    env: Environment | None,
) -> float:
    """Length of one continuous polyline, with no time support.

    Parameters
    ----------
    positions : array-like, shape (n_samples, n_dims)
        Positions of a path already known to be continuously observed.
    metric : {"euclidean", "geodesic"}
    env : Environment, optional
        Required when ``metric="geodesic"``.
    """
    from neurospatial.behavior.trajectory import compute_step_lengths

    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2:
        raise ValueError(
            f"positions must be 2D array (n_samples, n_dims), got {positions.ndim}D"
        )
    return float(np.sum(compute_step_lengths(positions, metric=metric, env=env)))


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
    support: ObservationSupport,
    *,
    metric: Literal["euclidean", "geodesic"],
    env: Environment | None,
) -> tuple[float, bool]:
    """Path length over usable intervals, and whether every interval counted.

    Parameters
    ----------
    positions : ndarray, shape (n_samples, n_dims)
    support : ObservationSupport
        Resolved from the same samples.
    metric : {"euclidean", "geodesic"}
    env : Environment or None
        Required for ``metric="geodesic"``.

    Returns
    -------
    length : float
        Sum of the counted interval lengths; ``nan`` when none counted.
    complete : bool
        True when there is at least one interval and every interval counted.
    """
    from neurospatial.behavior.trajectory import (
        _observed_cumulative,
        compute_step_lengths,
    )

    if metric == "euclidean":
        steps, _ = _observed_cumulative(positions, support)
        counted = support.interval_mask
    else:
        with np.errstate(invalid="ignore"):
            steps = compute_step_lengths(positions, metric="geodesic", env=env)
        counted = support.interval_mask & ~np.isnan(steps)
    complete = bool(counted.size and counted.all())
    if not counted.any():
        return float("nan"), complete
    return float(steps[counted].sum()), complete


def traveled_path_length(
    times: ArrayLike,
    positions: ArrayLike,
    *,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    env: Environment | None = None,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> float:
    """Compute the distance traveled over observed intervals.

    Distance is added only over usable intervals: both samples have finite
    coordinates, the time between them is at most ``max_gap``, they fall inside
    one of ``epochs`` and no break is declared between them. Distance between
    recordings is never counted, and a missing sample splits the path instead
    of turning the total into NaN. A break must be declared by a time gap above
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
        Distance metric. Use "geodesic" for environments with barriers.
    env : Environment, optional
        Required when metric="geodesic". An interval with an endpoint outside
        the environment has no geodesic length and is left out; an interval
        between disconnected bins contributes ``inf``.
    max_gap : float or None, default=0.5
        Longest interval, in seconds, that counts as continuous observation.
        ``None`` keeps every interval.
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
    length, _ = _observed_path_length(positions_arr, support, metric=metric, env=env)
    return length


def shortest_path_length(
    env: Environment,
    start: NDArray[np.float64],
    goal: NDArray[np.float64],
    *,
    metric: Literal["geodesic", "euclidean"] = "geodesic",
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
        Distance metric.

    Returns
    -------
    float
        Shortest path distance in environment units.
        Returns inf if no path exists (disconnected graph).

    Raises
    ------
    ValueError
        For ``metric="geodesic"``, if ``start`` or ``goal`` is outside the
        environment or has NaN coordinates; the geodesic path starts and ends
        at the bins that contain them, as in ``env.distance_between``.
    CartesianOnlyError
        For ``metric="geodesic"`` on an egocentric polar environment, whose
        bins are (distance, angle), not allocentric positions.

    Examples
    --------
    >>> dist = shortest_path_length(env, start_pos, goal_pos)  # doctest: +SKIP
    >>> print(f"Shortest path: {dist:.1f} cm")  # doctest: +SKIP
    """
    start = np.asarray(start)
    goal = np.asarray(goal)

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
    return env.distance_between(points=np.vstack([start, goal]))


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
        Reference speed in environment units per second.
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

    Examples
    --------
    >>> eff = time_efficiency(
    ...     times, positions, reference_speed=20.0, optimal_distance=50.0
    ... )  # doctest: +SKIP
    >>> print(f"Time efficiency: {eff:.1%}")  # doctest: +SKIP
    """
    from neurospatial._validation import validate_times_positions

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
    positions: NDArray[np.float64],
    goal: NDArray[np.float64],
) -> float:
    """Compute angular path efficiency toward goal.

    Measures how directly the animal heads toward the goal throughout
    the trajectory.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates in allocentric frame.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position in allocentric frame.

    Returns
    -------
    float
        Angular efficiency in range [0, 1].
        Returns 1.0 for paths with < 3 positions.
        Returns NaN if all positions are identical.

    Notes
    -----
    Computed as: 1 - mean(|delta_theta|) / pi

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior import angular_efficiency
    >>> positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    >>> angular_efficiency(positions, goal=np.array([2.0, 0.0]))
    1.0
    """
    from neurospatial.behavior.trajectory import compute_turn_angles

    if len(positions) < 3:
        return 1.0

    if np.ptp(positions, axis=0).max() < 1e-10:
        return np.nan

    angles = compute_turn_angles(positions)

    if len(angles) == 0:
        return 1.0

    mean_abs_angle = np.mean(np.abs(angles))
    return float(1.0 - mean_abs_angle / np.pi)


def subgoal_efficiency(
    env: Environment,
    positions: NDArray[np.float64],
    subgoals: NDArray[np.float64],
    *,
    metric: Literal["geodesic", "euclidean"] = "geodesic",
) -> SubgoalEfficiencyResult:
    """Compute path efficiency decomposed by subgoals.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Trajectory positions.
    subgoals : NDArray[np.float64], shape (n_subgoals, n_dims)
        Ordered sequence of subgoals.
    metric : {"geodesic", "euclidean"}, default="geodesic"
        Distance metric.

    Returns
    -------
    SubgoalEfficiencyResult
        Per-segment efficiency and aggregated metrics.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.behavior import subgoal_efficiency
    >>> positions = np.c_[np.arange(11.0), np.zeros(11)]
    >>> env = Environment.from_samples(positions, bin_size=1.0)
    >>> subgoals = np.array([[5.0, 0.0], [10.0, 0.0]])
    >>> result = subgoal_efficiency(env, positions, subgoals, metric="euclidean")
    """
    subgoals = np.asarray(subgoals)
    if subgoals.ndim == 1:
        subgoals = subgoals.reshape(1, -1)

    n_subgoals = len(subgoals)
    segment_results = []

    current_start_idx = 0

    for i in range(n_subgoals):
        subgoal = subgoals[i]

        distances_to_subgoal = np.linalg.norm(positions - subgoal, axis=1)
        arrival_idx = (
            np.argmin(distances_to_subgoal[current_start_idx:]) + current_start_idx
        )

        segment_positions = positions[current_start_idx : arrival_idx + 1]

        if len(segment_positions) >= 2:
            start_pos = segment_positions[0]

            traveled = _polyline_length(segment_positions, metric=metric, env=env)
            shortest = shortest_path_length(env, start_pos, subgoal, metric=metric)

            if traveled > 0 and np.isfinite(shortest):
                eff = shortest / traveled
            else:
                eff = np.nan

            ang_eff = angular_efficiency(segment_positions, subgoal)

            segment_results.append(
                PathEfficiencyResult(
                    traveled_length=traveled,
                    shortest_length=shortest,
                    efficiency=eff,
                    time_efficiency=None,
                    angular_efficiency=ang_eff,
                    start_position=start_pos,
                    goal_position=subgoal,
                    metric=metric,
                )
            )
        else:
            segment_results.append(
                PathEfficiencyResult(
                    traveled_length=0.0,
                    shortest_length=0.0,
                    efficiency=np.nan,
                    time_efficiency=None,
                    angular_efficiency=1.0,
                    start_position=positions[current_start_idx],
                    goal_position=subgoal,
                    metric=metric,
                )
            )

        current_start_idx = arrival_idx

    efficiencies = [r.efficiency for r in segment_results if not np.isnan(r.efficiency)]
    shortest_lengths = [
        r.shortest_length for r in segment_results if not np.isnan(r.efficiency)
    ]

    if efficiencies:
        mean_eff = float(np.mean(efficiencies))
        weights = np.array(shortest_lengths)
        if weights.sum() > 0:
            weighted_eff = float(np.average(efficiencies, weights=weights))
        else:
            weighted_eff = mean_eff
    else:
        mean_eff = np.nan
        weighted_eff = np.nan

    return SubgoalEfficiencyResult(
        segment_results=segment_results,
        mean_efficiency=mean_eff,
        weighted_efficiency=weighted_eff,
        subgoal_positions=subgoals,
    )


def compute_path_efficiency(
    env: Environment,
    times: ArrayLike,
    positions: ArrayLike,
    goal: ArrayLike,
    *,
    metric: Literal["geodesic", "euclidean"] = "geodesic",
    reference_speed: float | None = None,
    max_gap: float | None = 0.5,
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
        Distance metric. With "geodesic", an interval with an endpoint outside
        the environment has no geodesic length and is left out, and an interval
        between disconnected bins contributes ``inf`` (so a complete path with
        such a step has ``traveled_length=inf`` and ``efficiency=0.0``).
    reference_speed : float, optional
        Reference speed for time efficiency.
    max_gap : float or None, default=0.5
        Longest interval, in seconds, that counts as continuous observation.
        ``None`` keeps every interval.
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
        All path efficiency metrics combined, measured from the first sample
        ``positions[0]`` to ``goal``. When the goal is unreachable under
        ``metric="geodesic"`` (it lies on a disconnected component),
        ``shortest_length`` is ``inf`` and the derived ``efficiency`` and
        ``time_efficiency`` fields are ``nan`` (not ``0.0``).

    Raises
    ------
    ValueError
        If ``metric`` is not recognized, or if the inputs are empty or
        malformed (for example ``times`` not strictly increasing, or
        ``break_before`` not a Boolean array with one entry per sample).
        With ``metric="geodesic"``, also if ``positions[0]`` or ``goal`` is
        NaN or outside the environment (raised by ``distance_between``). With
        ``metric="euclidean"`` a NaN first row gives ``shortest_length=nan``.

    Notes
    -----
    ``traveled_length`` is the distance over usable intervals, as returned by
    ``traveled_path_length`` with the same arguments: a lost frame, a gap
    longer than ``max_gap``, a declared break or an interval outside ``epochs``
    is left out, and it is ``nan`` when nothing was observed (including a
    single sample). ``efficiency`` and ``angular_efficiency`` compare the whole
    route with the shortest one, so they are ``nan`` unless every interval from
    the first to the last sample was observed; with a missing stretch the
    observed length underestimates the route and the ratio would be biased
    upward. ``shortest_length`` and the wall-clock ``time_efficiency`` keep
    their meanings. When one low-confidence frame makes a trial's efficiency
    ``nan``, report its ``traveled_length`` instead, or restrict the analysis
    to completely observed trials.

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
    >>> positions[5] = np.nan  # one lost frame
    >>> result = compute_path_efficiency(
    ...     env, times, positions, goal, metric="euclidean"
    ... )
    >>> result.traveled_length  # the observed travel
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
    goal = np.asarray(goal)

    traveled, complete = _observed_path_length(
        positions_arr, support, metric=metric, env=env
    )
    shortest = shortest_path_length(env, positions_arr[0], goal, metric=metric)

    eff = (
        shortest / traveled
        if complete and traveled > 0 and np.isfinite(shortest)
        else np.nan
    )

    time_eff = None
    if reference_speed is not None and len(positions_arr) >= 2:
        time_eff = time_efficiency(
            times_arr,
            positions_arr,
            reference_speed=reference_speed,
            optimal_distance=shortest,
        )

    ang_eff = angular_efficiency(positions_arr, goal) if complete else np.nan

    return PathEfficiencyResult(
        traveled_length=traveled,
        shortest_length=shortest,  # may be inf if the goal is unreachable
        efficiency=eff,  # already nan when shortest is non-finite
        time_efficiency=time_eff,
        angular_efficiency=ang_eff,
        start_position=positions_arr[0],
        goal_position=goal,
        metric=metric,
    )


# =============================================================================
# Goal-Directed Navigation Functions (from metrics/goal_directed.py)
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
    goal = np.asarray(goal)
    positions = np.asarray(positions)

    if goal.shape[0] != positions.shape[1]:
        raise ValueError(
            f"Goal has {goal.shape[0]} dimensions but positions have "
            f"{positions.shape[1]} dimensions. Both must match."
        )

    return goal[np.newaxis, :] - positions


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
    vec = goal_vector(positions, goal)
    return np.arctan2(vec[:, 1], vec[:, 0])


def instantaneous_goal_alignment(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    goal: NDArray[np.float64],
    *,
    min_speed: float,
    max_gap: float | None = 0.5,
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
        cannot know the positions' unit.
        Minimum speed threshold in environment units per second.
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from the analysis. ``None`` disables the gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Cosine of angle between velocity and goal direction.
        Range [-1, 1]. NaN where speed < ``min_speed`` or no run is observed.

    Notes
    -----
    Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``,
    not crossing a ``break_before``) is analyzed as a separate recording;
    no velocity or heading spans a pause.

    Samples in no observed run and samples below ``min_speed`` have NaN
    alignment. Each run's last sample repeats its previous interval's
    heading, as at the end of a complete recording.

    Examples
    --------
    >>> positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
    >>> times = np.linspace(0, 10, 21)
    >>> goal = np.array([100.0, 0.0])
    >>> alignment = instantaneous_goal_alignment(times, positions, goal, min_speed=0.0)
    >>> bool(np.nanmean(alignment) > 0.9)
    True
    """
    from neurospatial._validation import validate_times_positions

    times, positions = validate_times_positions(
        times, positions, call="instantaneous_goal_alignment"
    )
    from neurospatial.ops import _velocity_heading_and_speed

    goal = np.asarray(goal)

    from neurospatial._intervals import observed_interval_mask
    from neurospatial._observation import validate_break_before

    breaks = validate_break_before(
        break_before, len(times), call="instantaneous_goal_alignment"
    )
    interval_mask = observed_interval_mask(
        times, max_gap=max_gap, epochs=epochs, break_before=breaks
    )
    if len(positions) < 2:
        return np.full(len(positions), np.nan)

    velocity_heading, speed = _velocity_heading_and_speed(
        times,
        positions,
        interval_mask=interval_mask,
    )
    velocity_heading[speed < min_speed] = np.nan
    goal_heading = goal_direction(positions, goal)

    angle_diff = velocity_heading - goal_heading
    alignment = np.cos(angle_diff)

    return alignment


def goal_bias(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    goal: NDArray[np.float64],
    *,
    min_speed: float,
    max_gap: float | None = 0.5,
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
        cannot know the positions' unit.
        Speed threshold in position units per second. Samples slower than this
        are excluded from the mean.
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from the analysis. ``None`` disables the gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    float
        Mean goal alignment, range [-1, 1].
        Returns NaN if all samples are stationary.

    Notes
    -----
    Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``,
    not crossing a ``break_before``) is analyzed as a separate recording;
    no velocity or heading spans a pause.

    Interpretation:
    - > 0.5: Strong goal-directed navigation
    - 0 to 0.5: Weak goal-directed with some wandering
    - < 0: Net movement away from goal

    Examples
    --------
    >>> positions = np.column_stack([np.linspace(0, 100, 101), np.zeros(101)])
    >>> times = np.linspace(0, 10, 101)
    >>> goal = np.array([100.0, 0.0])
    >>> goal_bias(times, positions, goal, min_speed=0.0) > 0.8
    True
    """
    from neurospatial._observation import validate_break_before
    from neurospatial._validation import validate_times_positions

    times, positions = validate_times_positions(times, positions, call="goal_bias")
    breaks = validate_break_before(break_before, len(times), call="goal_bias")
    alignment = instantaneous_goal_alignment(
        times,
        positions,
        goal,
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=epochs,
        break_before=breaks,
    )

    valid_alignment = alignment[~np.isnan(alignment)]

    if len(valid_alignment) == 0:
        return np.nan

    return float(np.mean(valid_alignment))


def approach_rate(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    goal: NDArray[np.float64],
    *,
    metric: Literal["geodesic", "euclidean"] = "euclidean",
    env: Environment | None = None,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute rate of distance change toward goal.

    The rate at sample ``k`` is the change in distance to the goal over the
    interval from sample ``k - 1`` to ``k``, divided by its duration. It is
    measured only over usable intervals: both samples have finite coordinates,
    the time between them is at most ``max_gap``, they fall inside one of
    ``epochs`` and no break is declared between them. Every other sample is
    NaN, so no rate is measured across a missing position or between
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
        Distance metric. Geodesic requires env parameter.
    env : Environment, optional
        Required when metric="geodesic".
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from the analysis. ``None`` disables the gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
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
        If metric="geodesic" but env is None, or the goal is outside the
        environment or has NaN coordinates.
    ValueError
        If the inputs are empty, have different lengths, or ``times`` is not
        finite and strictly increasing.

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

    A missing position gives no rate into or out of it:

    >>> positions[1] = np.nan
    >>> approach_rate(times, positions, goal).tolist()
    [nan, nan, nan, -384.0, -8.0, -8.0]
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
    goal = np.asarray(goal)

    if metric == "geodesic" and env is None:
        raise ValueError(
            "env parameter is required when metric='geodesic'. "
            "Provide the Environment instance, or use metric='euclidean' "
            "for straight-line distances."
        )

    if metric == "euclidean":
        goal_vec = goal_vector(positions, goal)
        distances = np.linalg.norm(goal_vec, axis=1)
    else:
        assert env is not None
        require_cartesian(
            env.coordinate_system,
            "approach_rate(metric='geodesic')",
            why=(
                "positions and goal are allocentric Cartesian positions, located in "
                "the environment's bins."
            ),
            fix=(
                "use an allocentric environment, or metric='euclidean' for the "
                "straight-line distance to the goal"
            ),
        )
        position_bins = env.bin_at(positions)
        # The goal is an explicit point: one outside the environment raises.
        to_goal = env.distance_to(points=goal.reshape(1, -1), metric="geodesic")
        distances = to_goal[position_bins]
        distances[position_bins == -1] = np.nan

    from neurospatial.behavior._kinematics import interval_velocity

    velocity = interval_velocity(times, distances[:, None], support.interval_mask)
    rates = np.full(len(positions), np.nan)
    rates[1:] = velocity[:, 0]

    return rates


def compute_goal_directed_metrics(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    goal: NDArray[np.float64],
    *,
    metric: Literal["geodesic", "euclidean"] = "euclidean",
    min_speed: float | None = None,
    goal_radius: float | None = None,
    max_gap: float | None = 0.5,
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
        Distance metric for approach rate and distance computations.
    min_speed : float or None, default=None
        In position units per second. None is 5 cm/s expressed in
        ``env.units`` (``0.05`` for meters, ``50.0`` for millimeters); with
        units that are not a metric length, pass it explicitly.
        Minimum speed for valid heading (units/s).
    goal_radius : float, optional
        Radius for goal arrival detection.
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from the analysis. ``None`` disables the gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    GoalDirectedMetrics
        Dataclass containing all goal-directed metrics.

    Raises
    ------
    ValueError
        If positions and times have different lengths.
    ValueError
        If ``break_before`` is not a Boolean array with one entry per sample.

    Notes
    -----
    Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``,
    not crossing a ``break_before``) is analyzed as a separate recording;
    no velocity or heading spans a pause.

    Heading alignment and approach-rate averages use observed intervals.
    Time-to-goal remains wall-clock time, which is known across a pause.

    Examples
    --------
    >>> result = compute_goal_directed_metrics(
    ...     env, times, positions, goal
    ... )  # doctest: +SKIP
    >>> print(result)  # doctest: +SKIP
    """
    from neurospatial._observation import validate_break_before
    from neurospatial._validation import validate_times_positions

    times, positions = validate_times_positions(
        times, positions, call="compute_goal_directed_metrics"
    )
    breaks = validate_break_before(
        break_before, len(times), call="compute_goal_directed_metrics"
    )
    goal = np.asarray(goal)

    from neurospatial.behavior._kinematics import resolve_min_speed

    min_speed = resolve_min_speed(
        env, min_speed, call="compute_goal_directed_metrics()"
    )
    bias = goal_bias(
        times,
        positions,
        goal,
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=epochs,
        break_before=breaks,
    )

    rates = approach_rate(
        times,
        positions,
        goal,
        metric=metric,
        env=env,
        max_gap=max_gap,
        epochs=epochs,
        break_before=breaks,
    )
    mean_rate = float(np.nanmean(rates)) if np.any(~np.isnan(rates)) else np.nan

    goal_vec = goal_vector(positions, goal)
    distances = np.linalg.norm(goal_vec, axis=1)

    min_dist = float(np.min(distances))
    start_dist = float(distances[0])
    end_dist = float(distances[-1])

    time_to_goal_val: float | None = None
    if goal_radius is not None:
        arrival_mask = distances <= goal_radius
        if np.any(arrival_mask):
            arrival_idx = np.argmax(arrival_mask)
            time_to_goal_val = float(times[arrival_idx] - times[0])

    return GoalDirectedMetrics(
        goal_bias=bias,
        mean_approach_rate=mean_rate,
        time_to_goal=time_to_goal_val,
        min_distance_to_goal=min_dist,
        goal_distance_at_start=start_dist,
        goal_distance_at_end=end_dist,
        goal_position=goal.copy(),
        metric=metric,
    )
