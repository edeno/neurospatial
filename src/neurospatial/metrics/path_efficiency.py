"""Path efficiency metrics for spatial navigation analysis.

This module computes how efficiently an animal navigates from start to goal,
comparing the actual path taken against the optimal (shortest) path.

Metrics include:
- **Path efficiency**: Ratio of shortest to actual path length (1.0 = perfect)
- **Time efficiency**: Accounting for travel duration
- **Angular efficiency**: How directly the animal heads toward the goal
- **Subgoal efficiency**: Decomposed efficiency for multi-waypoint paths

When to Use Each Metric
-----------------------
- **metric="geodesic"**: Environments with walls, barriers, or constrained paths
  (T-maze, plus maze, open field with obstacles). Respects environment topology.
- **metric="euclidean"**: Continuous open spaces without obstacles, or when you
  want "as-the-crow-flies" distance comparison.

Example
-------
>>> from neurospatial.metrics import compute_path_efficiency
>>> result = compute_path_efficiency(env, positions, times, goal_position)
>>> print(result.summary())
Path: 45.2 cm traveled, 32.1 cm optimal (efficiency: 71.0%)
>>> if result.is_efficient(threshold=0.8):
...     print("Efficient navigation")

References
----------
.. [1] Batschelet, E. (1981). Circular Statistics in Biology. Academic Press.
.. [2] Johnson, A., & Redish, A. D. (2007). Neural ensembles in CA3 transiently
       encode paths forward of the animal at a decision point. J Neurosci.
       DOI: 10.1523/JNEUROSCI.3761-07.2007
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


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
        If positions is empty.
        If positions is not 2D array.

    Examples
    --------
    >>> length = traveled_path_length(positions)
    >>> print(f"Animal traveled {length:.1f} cm")
    """
    from neurospatial.behavior.trajectory import compute_step_lengths

    # Input validation
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

    # Map metric parameter to distance_type for compute_step_lengths.
    # Note: compute_step_lengths uses the legacy parameter name "distance_type"
    # while this module standardizes on "metric" for consistency with behavioral.py.
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
        Distance metric:
        - "geodesic": Shortest path on environment graph
        - "euclidean": Straight-line distance

    Returns
    -------
    float
        Shortest path distance in environment units.
        Returns inf if no path exists (disconnected graph).

    Examples
    --------
    >>> dist = shortest_path_length(env, start_pos, goal_pos)
    >>> print(f"Shortest path: {dist:.1f} cm")
    """
    start = np.asarray(start)
    goal = np.asarray(goal)

    if metric == "euclidean":
        # Straight-line distance
        return float(np.linalg.norm(goal - start))

    # Geodesic distance using distance_field
    from neurospatial.ops.distance import distance_field

    # Map positions to bins
    start_bin = env.bin_at(start.reshape(1, -1))[0]
    goal_bin = env.bin_at(goal.reshape(1, -1))[0]

    # Compute distance field from goal
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
    >>> eff = path_efficiency(env, positions, goal)
    >>> print(f"Efficiency: {eff:.1%}")
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
        Optimal time = shortest_distance / reference_speed.

    Returns
    -------
    float
        Time efficiency ratio T_optimal / T_actual.

    Examples
    --------
    >>> eff = time_efficiency(positions, times, goal, reference_speed=20.0)
    >>> print(f"Time efficiency: {eff:.1%}")
    """
    if len(positions) < 2:
        return np.nan

    if len(positions) != len(times):
        raise ValueError(
            f"positions and times must have same length. "
            f"Got positions: {len(positions)}, times: {len(times)}. "
            f"Check that both arrays cover the same time period."
        )

    # Actual travel time
    actual_time = times[-1] - times[0]
    if actual_time <= 0:
        return np.nan

    # Shortest distance (Euclidean)
    start = positions[0]
    goal = np.asarray(goal)
    shortest_dist = float(np.linalg.norm(goal - start))

    # Optimal time at reference speed
    optimal_time = shortest_dist / reference_speed

    return float(optimal_time / actual_time)


def angular_efficiency(
    positions: NDArray[np.float64],
    goal: NDArray[np.float64],
) -> float:
    """Compute angular path efficiency toward goal.

    Measures how directly the animal heads toward the goal throughout
    the trajectory. A value of 1.0 means always heading directly toward
    the goal; lower values indicate wandering or indirect paths.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates in allocentric frame.
    goal : NDArray[np.float64], shape (n_dims,)
        Goal position in allocentric frame (same coordinate system).

    Returns
    -------
    float
        Angular efficiency in range [0, 1].
        Returns 1.0 for paths with < 3 positions (no turns measurable).
        Returns NaN if all positions are identical.

    Notes
    -----
    Computed as: 1 - mean(|delta_theta|) / pi

    where delta_theta is the angular deviation between movement direction
    and goal direction at each step.
    """
    from neurospatial.behavior.trajectory import compute_turn_angles

    if len(positions) < 3:
        return 1.0  # No turns possible

    # Check for degenerate case (all positions identical or near-stationary)
    # Use peak-to-peak range to check if all positions are within tolerance
    if np.ptp(positions, axis=0).max() < 1e-10:
        return np.nan

    # Compute turn angles
    angles = compute_turn_angles(positions)

    if len(angles) == 0:
        return 1.0  # No valid turns (all stationary)

    # Angular efficiency: 1 - mean(|delta_theta|) / pi
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
        Ordered sequence of subgoals. First subgoal is intermediate waypoint,
        last subgoal is final goal.
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

    # First segment: start -> first subgoal
    current_start_idx = 0

    for i in range(n_subgoals):
        subgoal = subgoals[i]

        # Find index when we reach this subgoal (closest approach)
        distances_to_subgoal = np.linalg.norm(positions - subgoal, axis=1)
        arrival_idx = (
            np.argmin(distances_to_subgoal[current_start_idx:]) + current_start_idx
        )

        # Extract segment
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
            # Degenerate segment
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

    # Compute aggregated metrics
    efficiencies = [r.efficiency for r in segment_results if not np.isnan(r.efficiency)]
    shortest_lengths = [
        r.shortest_length for r in segment_results if not np.isnan(r.efficiency)
    ]

    if efficiencies:
        mean_eff = float(np.mean(efficiencies))
        # Weighted by shortest length
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
        Reference speed for time efficiency. If None, time_efficiency is None.

    Returns
    -------
    PathEfficiencyResult
        All path efficiency metrics combined.

        - efficiency: NaN if < 2 positions or traveled_length is 0
        - time_efficiency: None if reference_speed not provided; NaN if < 2 positions
        - angular_efficiency: 1.0 if < 3 positions; NaN if all positions identical

    Raises
    ------
    ValueError
        If positions and times have different lengths.

    Examples
    --------
    >>> result = compute_path_efficiency(env, positions, times, goal)
    >>> print(result.summary())
    >>> if result.is_efficient():
    ...     print("Efficient path!")
    """
    if len(positions) != len(times):
        raise ValueError(
            f"positions and times must have same length. "
            f"Got positions: {len(positions)}, times: {len(times)}. "
            f"Check that both arrays cover the same time period."
        )

    goal = np.asarray(goal)

    # Compute traveled length
    if len(positions) >= 2:
        traveled = traveled_path_length(positions, metric=metric, env=env)
    else:
        traveled = 0.0

    # Compute shortest path
    if len(positions) >= 1:
        start = positions[0]
        shortest = shortest_path_length(env, start, goal, metric=metric)
    else:
        shortest = np.nan

    # Compute efficiency
    eff = shortest / traveled if traveled > 0 and np.isfinite(shortest) else np.nan

    # Compute time efficiency if reference speed provided
    time_eff = None
    if reference_speed is not None and len(positions) >= 2:
        time_eff = time_efficiency(
            positions, times, goal, reference_speed=reference_speed
        )

    # Compute angular efficiency
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
