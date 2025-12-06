"""Navigation and path efficiency metrics for spatial behavior analysis.

This module provides comprehensive tools for analyzing goal-directed navigation
and path efficiency in spatial environments. It combines:

- **Path progress and distance metrics**: Compute progress toward goals, distances
  to regions, and cost-aware navigation
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
        path_efficiency,
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

    from neurospatial.behavior import path_progress, path_efficiency, goal_bias

Typical Workflows
-----------------
**Multi-trial analysis** (T-maze, Y-maze, spatial bandit):
    1. Segment trajectory into trials with ``segment_trials()``
    2. Convert trials to bin arrays with ``trials_to_region_arrays()``
    3. Compute metrics: ``path_progress()``, ``distance_to_region()``, etc.

**Path efficiency analysis**:
    1. Compute efficiency: ``result = compute_path_efficiency(env, positions, times, goal)``
    2. Check efficiency: ``if result.is_efficient(threshold=0.8): ...``
    3. Print summary: ``print(result.summary())``

**Goal-directed analysis**:
    1. Compute metrics: ``result = compute_goal_directed_metrics(env, positions, times, goal)``
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
        trajectory_bins,
        times,
        env,
        start_region="home",
        end_regions=["goal_left", "goal_right"],
    )

    # 2. Extract trial-based regressors
    start_bins, goal_bins = trials_to_region_arrays(trials, times, env)
    progress = path_progress(env, trajectory_bins, start_bins, goal_bins)

    # 3. Compute efficiency
    result = compute_path_efficiency(env, positions, times, goal)
    print(result.summary())

    # 4. Compute goal-directed metrics
    gd_result = compute_goal_directed_metrics(env, positions, times, goal)
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
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.behavior.segmentation import Trial


__all__ = [  # noqa: RUF022
    # Path progress and distance (from behavioral.py)
    "cost_to_goal",
    "distance_to_region",
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
    "path_efficiency",
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


@dataclass(frozen=True)
class PathEfficiencyResult:
    """Path efficiency metrics for a trajectory segment.

    Attributes
    ----------
    traveled_length : float
        Total path length traveled (sum of step lengths), in environment units.
    shortest_length : float
        Geodesic or Euclidean distance from start to goal, in environment units.
    efficiency : float
        Ratio shortest_length / traveled_length. Range (0, 1].
        Value of 1.0 indicates optimal path taken.
        Returns NaN if traveled_length is 0 or path has < 2 points.
    time_efficiency : float or None
        Ratio T_optimal / T_actual if reference_speed provided, else None.
    angular_efficiency : float
        1 - mean(|delta_theta|) / pi. Range [0, 1].
        Value of 1.0 indicates heading directly toward goal at all times.
        Returns 1.0 for paths with < 3 positions (no turns possible).
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

    def summary(self) -> str:
        """Human-readable summary for printing.

        Returns
        -------
        str
            Formatted string with path lengths and efficiency percentage.
        """
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


@dataclass(frozen=True)
class SubgoalEfficiencyResult:
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

    def summary(self) -> str:
        """Human-readable summary for printing."""
        lines = [f"Subgoal path efficiency ({len(self.segment_results)} segments):"]
        for i, seg in enumerate(self.segment_results):
            lines.append(f"  Segment {i + 1}: {seg.efficiency:.1%}")
        lines.append(f"  Mean: {self.mean_efficiency:.1%}")
        lines.append(f"  Weighted: {self.weighted_efficiency:.1%}")
        return "\n".join(lines)


# =============================================================================
# Goal-Directed Metrics Dataclass
# =============================================================================


@dataclass(frozen=True)
class GoalDirectedMetrics:
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

    def summary(self) -> str:
        """Human-readable summary for printing.

        Returns
        -------
        str
            Formatted string with goal-directed metrics.
        """
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
    - If a region has multiple bins (polygon region), uses the first bin (index 0)
    - Failed trials (`trial.end_region is None`) have goal_bins = -1
    - Timepoints outside all trials have both arrays = -1

    Examples
    --------
    >>> trials = segment_trials(trajectory_bins, times, env, ...)  # doctest: +SKIP
    >>> start_bins, goal_bins = trials_to_region_arrays(
    ...     trials, times, env
    ... )  # doctest: +SKIP
    >>> progress = path_progress(
    ...     env, trajectory_bins, start_bins, goal_bins
    ... )  # doctest: +SKIP
    >>> dist = distance_to_region(env, trajectory_bins, goal_bins)  # doctest: +SKIP

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

        # Get bins for start region
        start_region_bins = env.bins_in_region(trial.start_region)
        if len(start_region_bins) > 0:
            # Use first bin if multiple bins in region
            start_bins[mask] = start_region_bins[0]

        # Get bins for end region (handle None for failed trials)
        if trial.end_region is not None:
            end_region_bins = env.bins_in_region(trial.end_region)
            if len(end_region_bins) > 0:
                # Use first bin if multiple bins in region
                goal_bins[mask] = end_region_bins[0]
        # For failed trials (end_region=None), goal_bins remains -1

    return start_bins, goal_bins


def path_progress(
    env: Environment,
    trajectory_bins: NDArray[np.int_],
    start_bins: NDArray[np.int_],
    goal_bins: NDArray[np.int_],
    *,
    metric: Literal["geodesic", "euclidean"] = "geodesic",
) -> NDArray[np.float64]:
    """Compute normalized path progress from start to goal (0 -> 1).

    Fully vectorized computation over entire session. For each timepoint t:
        progress[t] = distance(start_bins[t], trajectory_bins[t]) /
                      distance(start_bins[t], goal_bins[t])

    Parameters
    ----------
    env : Environment
        Spatial environment.
    trajectory_bins : NDArray[np.int_], shape (n_samples,)
        Current bin index at each timepoint.
    start_bins : NDArray[np.int_], shape (n_samples,)
        Start bin for each timepoint. Can be constant (single trial) or
        vary per-timepoint (multiple trials).
    goal_bins : NDArray[np.int_], shape (n_samples,)
        Goal bin for each timepoint. Can be constant or vary per-timepoint.
    metric : {'geodesic', 'euclidean'}, default='geodesic'
        Distance metric.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Path progress from 0 to 1 at each timepoint. NaN for invalid bins (-1)
        or when no path exists between bins.

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
    ...     env,
    ...     trajectory_bins,
    ...     start_bins=np.full(len(trajectory_bins), 10),
    ...     goal_bins=np.full(len(trajectory_bins), 50),
    ... )

    >>> # Multiple trials - construct arrays once, compute once
    >>> trials = segment_trials(trajectory_bins, times, env, ...)  # doctest: +SKIP
    >>> start_bins, goal_bins = trials_to_region_arrays(
    ...     trials, times, env
    ... )  # doctest: +SKIP
    >>> progress = path_progress(
    ...     env, trajectory_bins, start_bins, goal_bins
    ... )  # doctest: +SKIP

    See Also
    --------
    trials_to_region_arrays : Helper to construct start/goal arrays from trials
    distance_to_region : Distance to target region over time
    """
    # Check fitted state
    if not env._is_fitted:
        from neurospatial import EnvironmentNotFittedError

        raise EnvironmentNotFittedError("Environment", "path_progress")

    # Validate array lengths
    n_samples = len(trajectory_bins)
    if len(start_bins) != n_samples or len(goal_bins) != n_samples:
        raise ValueError(
            f"Array length mismatch: trajectory_bins has {n_samples} samples, "
            f"but start_bins has {len(start_bins)} and goal_bins has {len(goal_bins)}. "
            f"All arrays must have the same length."
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
        distances_from_start = dist_matrix[start_bins, trajectory_bins]

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
                [int(start_bin)],
                metric=metric,
                bin_centers=env.bin_centers if metric == "euclidean" else None,
            )

            distances_from_start[pair_mask] = start_dist_field[
                trajectory_bins[pair_mask]
            ]
            total_distances[pair_mask] = start_dist_field[goal_bin]

    # Compute progress
    with np.errstate(divide="ignore", invalid="ignore"):
        progress = distances_from_start / total_distances

    # Handle edge cases
    same_start_goal = start_bins == goal_bins
    progress[same_start_goal] = 1.0

    disconnected = np.isinf(total_distances)
    progress[disconnected] = np.nan

    invalid_bins = (start_bins == -1) | (goal_bins == -1) | (trajectory_bins == -1)
    progress[invalid_bins] = np.nan

    progress = np.clip(progress, 0.0, 1.0)

    return progress


def distance_to_region(
    env: Environment,
    trajectory_bins: NDArray[np.int_],
    target_bins: NDArray[np.int_] | int,
    *,
    metric: Literal["geodesic", "euclidean"] = "geodesic",
) -> NDArray[np.float64]:
    """Compute distance from each trajectory point to target region.

    Measures instantaneous distance to a goal region over time. Useful
    for analyzing approach behavior, goal-directed navigation, and
    spatial value functions.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    trajectory_bins : NDArray[np.int_], shape (n_samples,)
        Bin indices over time.
    target_bins : NDArray[np.int_] or int
        Target bin specification:
        - int: Single target bin (constant over time)
        - NDArray[np.int_], shape (n_samples,): Target bin at each timepoint
          (allows dynamic goals)
    metric : {'geodesic', 'euclidean'}, default='geodesic'
        Distance metric.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Distance from current position to target at each timepoint.
        Units match environment coordinates (e.g., cm).

    Notes
    -----
    **Performance strategy (automatic)**:

    - **Scalar targets**: Delegates to ``env.distance_to()`` for efficiency.
    - **Array targets (small env)**: Precomputes distance matrix.
    - **Array targets (large env)**: Computes per-unique-target distance fields.

    Examples
    --------
    >>> goal_bins = env.bins_in_region("reward_zone")  # doctest: +SKIP
    >>> dist = distance_to_region(env, trajectory_bins, goal_bins[0])  # doctest: +SKIP

    See Also
    --------
    path_progress : Normalized progress along path
    cost_to_goal : Distance with terrain/learned cost
    """
    # Check fitted state
    if not env._is_fitted:
        from neurospatial import EnvironmentNotFittedError

        raise EnvironmentNotFittedError("Environment", "distance_to_region")

    is_scalar_target = np.isscalar(target_bins)

    if is_scalar_target:
        n_samples = len(trajectory_bins)
        target_int = int(target_bins)

        if target_int == -1:
            return np.full(n_samples, np.nan, dtype=np.float64)

        dist_field = env.distance_to([target_int], metric=metric)  # type: ignore[misc]
        distances = dist_field[trajectory_bins].astype(np.float64)

        invalid_mask = trajectory_bins == -1
        distances[invalid_mask] = np.nan
        return distances

    else:
        n_samples = len(trajectory_bins)

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

            distances = dist_matrix[trajectory_bins, target_bins]
        else:
            unique_targets = np.unique(target_bins)
            unique_targets = unique_targets[unique_targets != -1]

            distances = np.full(n_samples, np.nan, dtype=np.float64)

            for target in unique_targets:
                mask = target_bins == target
                if np.any(mask):
                    dist_field = env.distance_to([int(target)], metric=metric)  # type: ignore[misc]
                    distances[mask] = dist_field[trajectory_bins[mask]]

        invalid_mask = (trajectory_bins == -1) | (target_bins == -1)
        distances[invalid_mask] = np.nan

        return distances


def cost_to_goal(
    env: Environment,
    trajectory_bins: NDArray[np.int_],
    goal_bins: NDArray[np.int_] | int,
    *,
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
    env : Environment
        Spatial environment.
    trajectory_bins : NDArray[np.int_], shape (n_samples,)
        Bin indices over time.
    goal_bins : NDArray[np.int_] or int
        Target bin(s). Can be scalar (constant goal) or array (dynamic goal).
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
    >>> cost = cost_to_goal(env, trajectory_bins, goal_bin)  # doctest: +SKIP

    >>> # Learned avoidance: avoid punishment zone
    >>> cost_map = np.ones(env.n_bins)  # doctest: +SKIP
    >>> cost_map[punishment_bins] = 10.0  # doctest: +SKIP
    >>> cost = cost_to_goal(
    ...     env, trajectory_bins, goal_bin, cost_map=cost_map
    ... )  # doctest: +SKIP

    See Also
    --------
    distance_to_region : Simple geometric distance
    """
    from neurospatial.ops.distance import distance_field

    # Case 1: No cost modifications
    if cost_map is None and terrain_difficulty is None:
        return distance_to_region(env, trajectory_bins, goal_bins, metric="geodesic")

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

    if isinstance(goal_bins, (int, np.integer)):
        dist_field = distance_field(g_weighted, [int(goal_bins)], weight="weight")
        costs = dist_field[trajectory_bins]

        invalid_mask = (trajectory_bins == -1) | (goal_bins == -1)
        costs[invalid_mask] = np.nan

        return costs
    else:
        unique_goals = np.unique(goal_bins)
        unique_goals = unique_goals[unique_goals != -1]

        costs = np.full(len(trajectory_bins), np.nan, dtype=np.float64)

        for goal in unique_goals:
            mask = goal_bins == goal
            if np.any(mask):
                dist_field = distance_field(g_weighted, [int(goal)], weight="weight")
                costs[mask] = dist_field[trajectory_bins[mask]]

        invalid_mask = (trajectory_bins == -1) | (goal_bins == -1)
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
    >>> trials = segment_trials(trajectory_bins, times, env, ...)  # doctest: +SKIP
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
    env: Environment,
    trajectory_bins: NDArray[np.int_],
    start_bin: int,
    end_bin: int,
    *,
    min_samples_per_edge: int = 50,
) -> str:
    """Extract discrete turn sequence from trajectory.

    Analyzes which bins were traversed and classifies turns based on
    changes in movement direction. Works for any environment type.

    Parameters
    ----------
    env : Environment
        Spatial environment (any layout type).
    trajectory_bins : NDArray[np.int_], shape (n_samples,)
        Bin indices along trajectory (single trial or segment).
    start_bin, end_bin : int
        Start and end bin indices to orient path direction.
    min_samples_per_edge : int, default=50
        Minimum trajectory samples required on a transition to count.

    Returns
    -------
    str
        Turn sequence like "left-right-left", or empty string "" if no
        turns detected. Each turn is separated by "-".

    Examples
    --------
    >>> for trial in trials:  # doctest: +SKIP
    ...     mask = (times >= trial.start_time) & (times <= trial.end_time)
    ...     turn_seq = graph_turn_sequence(
    ...         env, trajectory_bins[mask], start_bin, end_bin
    ...     )
    ...     print(f"Trial: {turn_seq}")  # "left" or "right"

    See Also
    --------
    compute_trajectory_curvature : Continuous curvature for any trajectory
    """
    if not env._is_fitted:
        from neurospatial import EnvironmentNotFittedError

        raise EnvironmentNotFittedError("Environment", "graph_turn_sequence")

    consecutive_bins = np.column_stack([trajectory_bins[:-1], trajectory_bins[1:]])
    transition_counts = Counter(map(tuple, consecutive_bins))

    valid_transitions = [
        trans
        for trans, count in transition_counts.items()
        if count >= min_samples_per_edge
    ]

    if len(valid_transitions) == 0:
        return ""

    unique_bins_ordered = []
    for bin_idx in trajectory_bins:
        if bin_idx not in unique_bins_ordered:
            unique_bins_ordered.append(bin_idx)

    valid_bins = set()
    for u, v in valid_transitions:
        valid_bins.add(u)
        valid_bins.add(v)

    path_bins = [b for b in unique_bins_ordered if b in valid_bins]

    if len(path_bins) < 3:
        return ""

    bin_centers = env.bin_centers
    n_dims = bin_centers.shape[1]

    turns = []

    for i in range(len(path_bins) - 2):
        bin_a = path_bins[i]
        bin_b = path_bins[i + 1]
        bin_c = path_bins[i + 2]

        pos_a = bin_centers[bin_a]
        pos_b = bin_centers[bin_b]
        pos_c = bin_centers[bin_c]

        vec1 = pos_b - pos_a
        vec2 = pos_c - pos_b

        vec1_norm = np.linalg.norm(vec1)
        vec2_norm = np.linalg.norm(vec2)

        if vec1_norm < 1e-10 or vec2_norm < 1e-10:
            continue

        vec1 = vec1 / vec1_norm
        vec2 = vec2 / vec2_norm

        if n_dims == 2:
            cross = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        elif n_dims >= 3:
            vec1_2d = vec1[:2]
            vec2_2d = vec2[:2]
            cross = vec1_2d[0] * vec2_2d[1] - vec1_2d[1] * vec2_2d[0]
        else:
            continue

        if abs(cross) > 0.1:
            if cross < 0:
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
    >>> trials = segment_trials(trajectory_bins, times, env, ...)  # doctest: +SKIP
    >>> labels = goal_pair_direction_labels(times, trials)  # doctest: +SKIP
    >>> result = compute_directional_place_fields(  # doctest: +SKIP
    ...     env, spike_times, times, positions, labels
    ... )

    See Also
    --------
    segment_trials : Segment trajectory into behavioral trials
    heading_direction_labels : Direction labels based on heading angle
    """
    labels = np.full(len(times), "other", dtype=object)

    for trial in trials:
        if trial.end_region is None:
            continue

        label = f"{trial.start_region}\u2192{trial.end_region}"
        mask = (times >= trial.start_time) & (times <= trial.end_time)
        labels[mask] = label

    return labels


def heading_direction_labels(
    positions: NDArray[np.float64] | None = None,
    times: NDArray[np.float64] | None = None,
    *,
    speed: NDArray[np.float64] | None = None,
    heading: NDArray[np.float64] | None = None,
    n_directions: int = 8,
    min_speed: float = 5.0,
) -> NDArray[np.object_]:
    """Generate per-timepoint direction labels from heading angle.

    Bins heading angles into sectors (e.g., "0-45 deg", "45-90 deg", ...) and labels
    slow-moving periods as "stationary".

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, 2), optional
        2D position coordinates over time. Required if speed/heading not provided.
    times : NDArray[np.float64], shape (n_samples,), optional
        Timestamps (seconds). Required if positions provided.
    speed : NDArray[np.float64], shape (n_samples,), optional
        Precomputed speed at each timepoint.
    heading : NDArray[np.float64], shape (n_samples,), optional
        Precomputed heading angle in radians.
    n_directions : int, default=8
        Number of direction bins. Default 8 creates 45 deg bins.
    min_speed : float, default=5.0
        Minimum speed threshold.

    Returns
    -------
    NDArray[np.object_], shape (n_samples,)
        Direction label for each timepoint.

    Examples
    --------
    >>> labels = heading_direction_labels(
    ...     positions=positions, times=times
    ... )  # doctest: +SKIP
    >>> labels = heading_direction_labels(
    ...     speed=speed, heading=heading
    ... )  # doctest: +SKIP

    See Also
    --------
    goal_pair_direction_labels : Direction labels for trialized tasks
    """
    has_precomputed = speed is not None or heading is not None
    has_trajectory = positions is not None or times is not None

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

        n_samples = len(speed_arr)

    elif has_trajectory:
        if positions is None or times is None:
            raise ValueError(
                "If providing trajectory data, both positions and times "
                "must be provided."
            )

        positions_arr = np.asarray(positions, dtype=np.float64)
        times_arr = np.asarray(times, dtype=np.float64)
        n_samples = len(times_arr)

        if n_samples == 0:
            return np.array([], dtype=object)
        if n_samples == 1:
            return np.array(["stationary"], dtype=object)

        dt = np.diff(times_arr)
        velocity = np.diff(positions_arr, axis=0) / dt[:, np.newaxis]

        speed_computed = np.linalg.norm(velocity, axis=1)
        heading_computed = np.arctan2(velocity[:, 1], velocity[:, 0])

        speed_arr = np.concatenate([[0.0], speed_computed])
        heading_arr = np.concatenate([[0.0], heading_computed])

    else:
        raise ValueError(
            "Must provide either (positions, times) or (speed, heading). "
            "Neither was provided."
        )

    labels = np.empty(n_samples, dtype=object)

    bin_edges_rad = np.linspace(-np.pi, np.pi, n_directions + 1)
    bin_edges_deg = np.linspace(-180.0, 180.0, n_directions + 1)

    bin_labels = []
    for i in range(n_directions):
        start_deg = round(bin_edges_deg[i])
        end_deg = round(bin_edges_deg[i + 1])
        label = f"{start_deg:.0f}\u2013{end_deg:.0f}\u00b0"
        bin_labels.append(label)

    for i in range(n_samples):
        if speed_arr[i] < min_speed:
            labels[i] = "stationary"
        else:
            h = heading_arr[i]
            h = np.arctan2(np.sin(h), np.cos(h))

            bin_idx = np.digitize(h, bin_edges_rad[1:], right=False)
            bin_idx = min(bin_idx, n_directions - 1)

            labels[i] = bin_labels[bin_idx]

    return labels


# =============================================================================
# Path Efficiency Functions (from metrics/path_efficiency.py)
# =============================================================================


def traveled_path_length(
    positions: NDArray[np.float64],
    *,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    env: Environment | None = None,
) -> float:
    """Compute total distance traveled along trajectory.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates in allocentric frame.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric. Use "geodesic" for environments with barriers.
    env : Environment, optional
        Required when metric="geodesic".

    Returns
    -------
    float
        Total path length in environment units. Returns 0.0 for single-point
        trajectories.

    Raises
    ------
    ValueError
        If metric="geodesic" but env is None.

    Examples
    --------
    >>> length = traveled_path_length(positions)  # doctest: +SKIP
    >>> print(f"Animal traveled {length:.1f} cm")  # doctest: +SKIP
    """
    from neurospatial.behavior.trajectory import compute_step_lengths

    if positions.ndim != 2:
        raise ValueError(
            f"positions must be 2D array (n_samples, n_dims), got {positions.ndim}D"
        )

    if len(positions) == 0:
        raise ValueError(
            "Cannot compute path length: positions array is empty. "
            "Provide at least 2 positions to compute path length."
        )

    if len(positions) < 2:
        return 0.0

    if metric == "geodesic" and env is None:
        raise ValueError(
            "env parameter is required when metric='geodesic'. "
            "Provide the Environment instance, or use metric='euclidean' "
            "for straight-line distances."
        )

    distance_type: Literal["euclidean", "geodesic"] = (
        "geodesic" if metric == "geodesic" else "euclidean"
    )
    step_lengths = compute_step_lengths(positions, distance_type=distance_type, env=env)
    return float(np.sum(step_lengths))


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

    Examples
    --------
    >>> dist = shortest_path_length(env, start_pos, goal_pos)  # doctest: +SKIP
    >>> print(f"Shortest path: {dist:.1f} cm")  # doctest: +SKIP
    """
    start = np.asarray(start)
    goal = np.asarray(goal)

    if metric == "euclidean":
        return float(np.linalg.norm(goal - start))

    from neurospatial.ops.distance import distance_field

    start_bin = env.bin_at(start.reshape(1, -1))[0]
    goal_bin = env.bin_at(goal.reshape(1, -1))[0]

    distances = distance_field(env.connectivity, [int(goal_bin)], metric="geodesic")

    return float(distances[start_bin])


def path_efficiency(
    env: Environment,
    positions: NDArray[np.float64],
    goal: NDArray[np.float64],
    *,
    metric: Literal["geodesic", "euclidean"] = "geodesic",
) -> float:
    """Compute path efficiency: ratio of shortest to traveled distance.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Trajectory positions.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position.
    metric : {"geodesic", "euclidean"}, default="geodesic"
        Distance metric for both traveled and shortest path.

    Returns
    -------
    float
        Efficiency ratio in range (0, 1]. Returns NaN if:
        - Trajectory has < 2 positions
        - Traveled length is 0 (stationary)

    Examples
    --------
    >>> eff = path_efficiency(env, positions, goal)  # doctest: +SKIP
    >>> print(f"Efficiency: {eff:.1%}")  # doctest: +SKIP
    """
    if len(positions) < 2:
        return np.nan

    traveled = traveled_path_length(positions, metric=metric, env=env)

    if traveled == 0.0 or np.isnan(traveled):
        return np.nan

    start = positions[0]
    shortest = shortest_path_length(env, start, goal, metric=metric)

    if np.isinf(shortest):
        return np.nan

    return float(shortest / traveled)


def time_efficiency(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    goal: NDArray[np.float64],
    *,
    reference_speed: float,
) -> float:
    """Compute time efficiency: ratio of optimal to actual travel time.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Trajectory positions.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position.
    reference_speed : float
        Reference speed in environment units per second.

    Returns
    -------
    float
        Time efficiency ratio T_optimal / T_actual.

    Examples
    --------
    >>> eff = time_efficiency(
    ...     positions, times, goal, reference_speed=20.0
    ... )  # doctest: +SKIP
    >>> print(f"Time efficiency: {eff:.1%}")  # doctest: +SKIP
    """
    if len(positions) < 2:
        return np.nan

    if len(positions) != len(times):
        raise ValueError(
            f"positions and times must have same length. "
            f"Got positions: {len(positions)}, times: {len(times)}. "
            f"Check that both arrays cover the same time period."
        )

    actual_time = times[-1] - times[0]
    if actual_time <= 0:
        return np.nan

    start = positions[0]
    goal = np.asarray(goal)
    shortest_dist = float(np.linalg.norm(goal - start))

    optimal_time = shortest_dist / reference_speed

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

            traveled = traveled_path_length(segment_positions, metric=metric, env=env)
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
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    goal: NDArray[np.float64],
    *,
    metric: Literal["geodesic", "euclidean"] = "geodesic",
    reference_speed: float | None = None,
) -> PathEfficiencyResult:
    """Compute comprehensive path efficiency metrics.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Trajectory positions.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position.
    metric : {"geodesic", "euclidean"}, default="geodesic"
        Distance metric.
    reference_speed : float, optional
        Reference speed for time efficiency.

    Returns
    -------
    PathEfficiencyResult
        All path efficiency metrics combined.

    Examples
    --------
    >>> result = compute_path_efficiency(env, positions, times, goal)  # doctest: +SKIP
    >>> print(result.summary())  # doctest: +SKIP
    """
    if len(positions) != len(times):
        raise ValueError(
            f"positions and times must have same length. "
            f"Got positions: {len(positions)}, times: {len(times)}. "
            f"Check that both arrays cover the same time period."
        )

    goal = np.asarray(goal)

    if len(positions) >= 2:
        traveled = traveled_path_length(positions, metric=metric, env=env)
    else:
        traveled = 0.0

    if len(positions) >= 1:
        start = positions[0]
        shortest = shortest_path_length(env, start, goal, metric=metric)
    else:
        shortest = np.nan

    eff = shortest / traveled if traveled > 0 and np.isfinite(shortest) else np.nan

    time_eff = None
    if reference_speed is not None and len(positions) >= 2:
        time_eff = time_efficiency(
            positions, times, goal, reference_speed=reference_speed
        )

    ang_eff = angular_efficiency(positions, goal)

    return PathEfficiencyResult(
        traveled_length=traveled,
        shortest_length=shortest if np.isfinite(shortest) else 0.0,
        efficiency=eff,
        time_efficiency=time_eff,
        angular_efficiency=ang_eff,
        start_position=positions[0] if len(positions) > 0 else np.array([]),
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
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    goal: NDArray[np.float64],
    *,
    min_speed: float = 5.0,
) -> NDArray[np.float64]:
    """Compute instantaneous alignment between movement and goal direction.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position.
    min_speed : float, default=5.0
        Minimum speed threshold in environment units per second.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Cosine of angle between velocity and goal direction.
        Range [-1, 1]. NaN for stationary periods.

    Examples
    --------
    >>> positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
    >>> times = np.linspace(0, 10, 21)
    >>> goal = np.array([100.0, 0.0])
    >>> alignment = instantaneous_goal_alignment(positions, times, goal, min_speed=0.0)
    >>> bool(np.nanmean(alignment) > 0.9)
    True
    """
    from neurospatial.ops.egocentric import heading_from_velocity

    positions = np.asarray(positions)
    times = np.asarray(times)
    goal = np.asarray(goal)

    if len(positions) < 2:
        return np.full(len(positions), np.nan)

    dt = float(np.median(np.diff(times)))

    velocity_heading = heading_from_velocity(positions, dt, min_speed=min_speed)
    goal_heading = goal_direction(positions, goal)

    angle_diff = velocity_heading - goal_heading
    alignment = np.cos(angle_diff)

    return alignment


def goal_bias(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    goal: NDArray[np.float64],
    *,
    min_speed: float = 5.0,
) -> float:
    """Compute mean alignment toward goal over trajectory.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position.
    min_speed : float, default=5.0
        Minimum speed threshold. Stationary periods excluded.

    Returns
    -------
    float
        Mean goal alignment, range [-1, 1].
        Returns NaN if all samples are stationary.

    Notes
    -----
    Interpretation:
    - > 0.5: Strong goal-directed navigation
    - 0 to 0.5: Weak goal-directed with some wandering
    - < 0: Net movement away from goal

    Examples
    --------
    >>> positions = np.column_stack([np.linspace(0, 100, 101), np.zeros(101)])
    >>> times = np.linspace(0, 10, 101)
    >>> goal = np.array([100.0, 0.0])
    >>> goal_bias(positions, times, goal, min_speed=0.0) > 0.8
    True
    """
    alignment = instantaneous_goal_alignment(
        positions, times, goal, min_speed=min_speed
    )

    valid_alignment = alignment[~np.isnan(alignment)]

    if len(valid_alignment) == 0:
        return np.nan

    return float(np.mean(valid_alignment))


def approach_rate(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    goal: NDArray[np.float64],
    *,
    metric: Literal["geodesic", "euclidean"] = "euclidean",
    env: Environment | None = None,
) -> NDArray[np.float64]:
    """Compute rate of distance change toward goal.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric. Geodesic requires env parameter.
    env : Environment, optional
        Required when metric="geodesic".

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Rate of distance change (d(distance)/dt) in units per second.
        Negative values indicate approaching the goal.
        First value is NaN.

    Raises
    ------
    ValueError
        If metric="geodesic" but env is None.

    Examples
    --------
    >>> positions = np.column_stack([np.linspace(0, 50, 11), np.zeros(11)])
    >>> times = np.linspace(0, 5, 11)
    >>> goal = np.array([100.0, 0.0])
    >>> rates = approach_rate(positions, times, goal)
    >>> bool(np.nanmean(rates) < 0)  # Negative = approaching
    True
    """
    positions = np.asarray(positions)
    times = np.asarray(times)
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
        trajectory_bins = env.bin_at(positions)
        goal_bin = env.bin_at(goal)
        distances = distance_to_region(
            env, trajectory_bins, goal_bin, metric="geodesic"
        )

    dt = np.diff(times)
    d_distance = np.diff(distances)

    rates = np.full(len(positions), np.nan)
    rates[1:] = d_distance / dt

    return rates


def compute_goal_directed_metrics(
    env: Environment,
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    goal: NDArray[np.float64],
    *,
    metric: Literal["geodesic", "euclidean"] = "euclidean",
    min_speed: float = 5.0,
    goal_radius: float | None = None,
) -> GoalDirectedMetrics:
    """Compute comprehensive goal-directed navigation metrics.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position.
    metric : {"geodesic", "euclidean"}, default="euclidean"
        Distance metric for approach rate and distance computations.
    min_speed : float, default=5.0
        Minimum speed for valid heading (units/s).
    goal_radius : float, optional
        Radius for goal arrival detection.

    Returns
    -------
    GoalDirectedMetrics
        Dataclass containing all goal-directed metrics.

    Raises
    ------
    ValueError
        If positions and times have different lengths.

    Examples
    --------
    >>> result = compute_goal_directed_metrics(
    ...     env, positions, times, goal
    ... )  # doctest: +SKIP
    >>> print(result.summary())  # doctest: +SKIP
    """
    positions = np.asarray(positions)
    times = np.asarray(times)
    goal = np.asarray(goal)

    if len(positions) != len(times):
        raise ValueError(
            f"positions and times must have same length. "
            f"Got positions: {len(positions)}, times: {len(times)}. "
            f"Check that both arrays cover the same time period."
        )

    bias = goal_bias(positions, times, goal, min_speed=min_speed)

    rates = approach_rate(positions, times, goal, metric=metric, env=env)
    mean_rate = float(np.nanmean(rates))

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
