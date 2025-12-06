"""Spatial decision analysis at choice points.

This module analyzes behavior at decision points (T-junctions, Y-mazes, etc.),
including pre-decision kinematics and goal selection dynamics.

Key Concepts
------------
- **Decision region**: Spatial zone where animal must choose (e.g., junction)
- **Pre-decision window**: Time window before entering decision region
- **Decision boundary**: Geodesic Voronoi boundary between potential goals
- **Boundary crossing**: When animal's nearest goal changes

When to Use
-----------
Use this module when analyzing:
- T-maze or Y-maze choice behavior
- Multi-goal navigation (spatial bandit tasks)
- Decision dynamics and commitment points
- VTE (vicarious trial and error) detection (see also: vte module)

Example
-------
>>> from neurospatial.metrics import compute_decision_analysis
>>> result = compute_decision_analysis(
...     env,
...     positions,
...     times,
...     decision_region="center",
...     goal_regions=["left", "right"],
...     pre_window=1.0,  # 1 second before decision
... )
>>> print(f"Heading variance: {result.pre_decision.heading_circular_variance:.2f}")
>>> if result.pre_decision.heading_circular_variance > 0.5:
...     print("High heading variability - possible deliberation")

References
----------
.. [1] Johnson, A., & Redish, A. D. (2007). Neural ensembles in CA3 transiently
       encode paths forward of the animal at a decision point. J Neurosci.
.. [2] Papale, A. E., et al. (2012). Interplay between hippocampal sharp-wave
       ripple events and vicarious trial and error behaviors. Neuron.
       DOI: 10.1016/j.neuron.2012.10.018
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment import Environment

__all__ = [
    "DecisionAnalysisResult",
    "DecisionBoundaryMetrics",
    "PreDecisionMetrics",
    "compute_decision_analysis",
    "compute_pre_decision_metrics",
    "decision_region_entry_time",
    "detect_boundary_crossings",
    "distance_to_decision_boundary",
    "extract_pre_decision_window",
    "geodesic_voronoi_labels",
    "pre_decision_heading_stats",
    "pre_decision_speed_stats",
]


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class PreDecisionMetrics:
    """Metrics from the pre-decision window.

    The pre-decision window is the time period immediately before the animal
    enters a decision region (e.g., 1 second before entering a T-junction).

    Attributes
    ----------
    mean_speed : float
        Mean speed in pre-decision window (environment units/s).
        Low values may indicate pausing/deliberation.
    min_speed : float
        Minimum speed in window (units/s).
        Near-zero indicates a pause, common during VTE.
    heading_mean_direction : float
        Circular mean of heading direction (radians, -pi to pi).
    heading_circular_variance : float
        Circular variance of heading, range [0, 1].
        High values (> 0.5) indicate variable heading - possible scanning.
        Low values indicate stable, consistent heading.
    heading_mean_resultant_length : float
        Concentration of heading distribution, range [0, 1].
        Inverse of variance: high = concentrated, low = dispersed.
    window_duration : float
        Actual duration of pre-decision window (seconds).
        May be shorter than requested if trajectory starts late.
    n_samples : int
        Number of samples in window.
    """

    mean_speed: float
    min_speed: float
    heading_mean_direction: float
    heading_circular_variance: float
    heading_mean_resultant_length: float
    window_duration: float
    n_samples: int

    def suggests_deliberation(
        self,
        variance_threshold: float = 0.5,
        speed_threshold: float = 10.0,
    ) -> bool:
        """Check if metrics suggest deliberative behavior.

        Parameters
        ----------
        variance_threshold : float, default=0.5
            Circular variance above this suggests head scanning.
        speed_threshold : float, default=10.0
            Mean speed below this (units/s) suggests slowing.

        Returns
        -------
        bool
            True if high heading variance AND low speed.
        """
        return (
            self.heading_circular_variance > variance_threshold
            and self.mean_speed < speed_threshold
        )


@dataclass(frozen=True)
class DecisionBoundaryMetrics:
    """Metrics related to decision boundaries between goals.

    Decision boundaries are the geodesic Voronoi edges between goal regions.
    An animal at the boundary is equidistant from multiple goals.

    Visualization (2-goal example)::

             Goal A          Goal B
               *               *
               |               |
               |    Boundary   |
               |       |       |
               |       |       |
               +-------+-------+
                       ^
                Decision point

    Attributes
    ----------
    goal_labels : NDArray[np.int_]
        Per-timepoint label of nearest goal (Voronoi region), shape (n_samples,).
        Values are indices into the goal_bins list.
    distance_to_boundary : NDArray[np.float64]
        Distance to nearest decision boundary at each timepoint, shape (n_samples,).
        Units match environment. Small values = near boundary = uncommitted.
    crossing_times : list[float]
        Times when trajectory crossed a decision boundary.
    crossing_directions : list[tuple[int, int]]
        (from_goal_idx, to_goal_idx) for each crossing.
        Indicates which goal regions the animal moved between.
    """

    goal_labels: NDArray[np.int_]
    distance_to_boundary: NDArray[np.float64]
    crossing_times: list[float]
    crossing_directions: list[tuple[int, int]]

    @property
    def n_crossings(self) -> int:
        """Number of decision boundary crossings."""
        return len(self.crossing_times)

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Decision boundary: {self.n_crossings} crossing"
            f"{'s' if self.n_crossings != 1 else ''}, "
            f"mean distance to boundary: {np.nanmean(self.distance_to_boundary):.1f}"
        )


@dataclass(frozen=True)
class DecisionAnalysisResult:
    """Complete decision analysis for a trial.

    Attributes
    ----------
    entry_time : float
        Time of decision region entry (seconds).
    pre_decision : PreDecisionMetrics
        Metrics from pre-decision window.
    boundary : DecisionBoundaryMetrics or None
        Boundary metrics. None if only one goal (no boundary defined).
    chosen_goal : int or None
        Index of goal reached. None if trial timed out or goal not reached.
    """

    entry_time: float
    pre_decision: PreDecisionMetrics
    boundary: DecisionBoundaryMetrics | None
    chosen_goal: int | None

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "Decision analysis:",
            f"  Entry time: {self.entry_time:.2f} s",
            f"  Pre-decision: speed={self.pre_decision.mean_speed:.1f}, "
            f"heading_var={self.pre_decision.heading_circular_variance:.2f}",
        ]
        if self.boundary is not None:
            lines.append(f"  {self.boundary.summary()}")
        if self.chosen_goal is not None:
            lines.append(f"  Chosen goal: {self.chosen_goal}")
        else:
            lines.append("  Chosen goal: none (timeout)")
        return "\n".join(lines)


# =============================================================================
# Pre-Decision Window Functions
# =============================================================================


def decision_region_entry_time(
    trajectory_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    region: str,
) -> float:
    """Find time of first entry to a decision region.

    Parameters
    ----------
    trajectory_bins : NDArray[np.int64], shape (n_samples,)
        Sequence of bin indices representing the trajectory.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps corresponding to trajectory bins (seconds).
    env : Environment
        Environment with region definitions.
    region : str
        Name of decision region in env.regions.

    Returns
    -------
    float
        Time of first entry to the decision region (seconds).

    Raises
    ------
    ValueError
        If region not found in env.regions.
    ValueError
        If trajectory never enters the region.

    Examples
    --------
    >>> entry_time = decision_region_entry_time(trajectory_bins, times, env, "center")
    >>> print(f"Entered decision region at t={entry_time:.2f}s")
    """
    if region not in env.regions:
        available = list(env.regions.keys())
        raise ValueError(
            f"Region '{region}' not found in environment. "
            f"Available regions: {available}. "
            f"Add the region using env.regions.add_region()."
        )

    # Get bins in the region
    region_bins = env.bins_in_region(region)
    region_bin_set = set(region_bins)

    # Find first entry
    for i, bin_idx in enumerate(trajectory_bins):
        if bin_idx in region_bin_set:
            return float(times[i])

    raise ValueError(
        f"Trajectory never enters region '{region}'. "
        f"Check that the trajectory passes through the decision region, "
        f"or verify region bounds with env.regions['{region}']."
    )


def extract_pre_decision_window(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    entry_time: float,
    window_duration: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract trajectory segment before decision region entry.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Full trajectory positions.
    times : NDArray[np.float64], shape (n_samples,)
        Full trajectory timestamps (seconds).
    entry_time : float
        Time of decision region entry (seconds).
    window_duration : float
        Duration of pre-decision window to extract (seconds).

    Returns
    -------
    window_positions : NDArray[np.float64]
        Positions within the pre-decision window.
    window_times : NDArray[np.float64]
        Times within the pre-decision window.

    Notes
    -----
    If the requested window extends before the trajectory start,
    the returned window will be shorter than requested.

    Examples
    --------
    >>> window_pos, window_times = extract_pre_decision_window(
    ...     positions, times, entry_time=5.0, window_duration=2.0
    ... )
    >>> # Returns data from t=3.0 to t<5.0
    """
    positions = np.asarray(positions)
    times = np.asarray(times)

    window_start = entry_time - window_duration
    window_start = max(window_start, times[0])  # Clamp to trajectory start

    # Select samples in window (before entry)
    mask = (times >= window_start) & (times < entry_time)

    return positions[mask], times[mask]


def pre_decision_heading_stats(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    min_speed: float = 5.0,
) -> tuple[float, float, float]:
    """Compute circular statistics on heading in a trajectory window.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps (seconds).
    min_speed : float, default=5.0
        Minimum speed for valid heading (units/s).
        Stationary periods are excluded from statistics.

    Returns
    -------
    mean_direction : float
        Circular mean heading in radians (-pi to pi).
    circular_variance : float
        Circular variance, range [0, 1].
        0 = all headings identical, 1 = uniform distribution.
    mean_resultant_length : float
        Mean resultant length, range [0, 1].
        1 = all headings identical, 0 = uniform distribution.

    Notes
    -----
    Circular statistics are computed directly:

    - mean_resultant_length = sqrt(mean(cos(theta))^2 + mean(sin(theta))^2)
    - circular_variance = 1 - mean_resultant_length
    - mean_direction = atan2(mean(sin(theta)), mean(cos(theta)))

    If all samples are stationary (below min_speed), returns:
    mean_direction=0, circular_variance=1, mean_resultant_length=0.

    Examples
    --------
    >>> mean_dir, circ_var, mrl = pre_decision_heading_stats(
    ...     positions, times, min_speed=5.0
    ... )
    >>> if circ_var > 0.5:
    ...     print("High heading variability")
    """
    from neurospatial.ops.egocentric import heading_from_velocity

    positions = np.asarray(positions)
    times = np.asarray(times)

    if len(positions) < 2:
        return 0.0, 1.0, 0.0

    # Compute dt from times (heading_from_velocity expects scalar dt)
    # Use median dt to handle irregular sampling
    dt = float(np.median(np.diff(times)))

    # Get headings
    headings = heading_from_velocity(positions, dt, min_speed=min_speed)

    # Filter out NaN (stationary periods)
    valid_headings = headings[~np.isnan(headings)]

    if len(valid_headings) == 0:
        # No valid headings: undefined direction, max variance
        return 0.0, 1.0, 0.0

    # Compute circular statistics directly
    cos_headings = np.cos(valid_headings)
    sin_headings = np.sin(valid_headings)
    mean_cos = float(np.mean(cos_headings))
    mean_sin = float(np.mean(sin_headings))

    mean_resultant_length = float(np.sqrt(mean_cos**2 + mean_sin**2))
    circular_variance = 1.0 - mean_resultant_length
    mean_direction = float(np.arctan2(mean_sin, mean_cos))

    return mean_direction, circular_variance, mean_resultant_length


def pre_decision_speed_stats(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
) -> tuple[float, float]:
    """Compute speed statistics for a trajectory window.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps (seconds).

    Returns
    -------
    mean_speed : float
        Mean instantaneous speed (units/s).
    min_speed : float
        Minimum instantaneous speed (units/s).

    Examples
    --------
    >>> mean_speed, min_speed = pre_decision_speed_stats(positions, times)
    >>> if min_speed < 1.0:
    ...     print("Animal paused during pre-decision window")
    """
    positions = np.asarray(positions)
    times = np.asarray(times)

    if len(positions) < 2:
        return 0.0, 0.0

    # Compute velocities
    dt = np.diff(times)
    displacement = np.diff(positions, axis=0)
    velocity = displacement / dt[:, np.newaxis]
    speeds = np.linalg.norm(velocity, axis=1)

    return float(np.mean(speeds)), float(np.min(speeds))


def compute_pre_decision_metrics(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    entry_time: float,
    window_duration: float,
    *,
    min_speed: float = 5.0,
) -> PreDecisionMetrics:
    """Compute all pre-decision window metrics.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Full trajectory positions.
    times : NDArray[np.float64], shape (n_samples,)
        Full trajectory timestamps (seconds).
    entry_time : float
        Time of decision region entry (seconds).
    window_duration : float
        Duration of pre-decision window to analyze (seconds).
    min_speed : float, default=5.0
        Minimum speed for valid heading (units/s).

    Returns
    -------
    PreDecisionMetrics
        Dataclass containing all pre-decision metrics.

    Examples
    --------
    >>> metrics = compute_pre_decision_metrics(
    ...     positions, times, entry_time=5.0, window_duration=2.0
    ... )
    >>> if metrics.suggests_deliberation():
    ...     print("Possible VTE behavior")
    """
    # Extract window
    window_pos, window_times = extract_pre_decision_window(
        positions, times, entry_time, window_duration
    )

    # Handle edge case of empty or too-short window
    if len(window_pos) < 2:
        return PreDecisionMetrics(
            mean_speed=0.0,
            min_speed=0.0,
            heading_mean_direction=0.0,
            heading_circular_variance=1.0,
            heading_mean_resultant_length=0.0,
            window_duration=0.0,
            n_samples=len(window_pos),
        )

    # Compute heading stats
    mean_dir, circ_var, mrl = pre_decision_heading_stats(
        window_pos, window_times, min_speed=min_speed
    )

    # Compute speed stats
    mean_speed, min_speed_val = pre_decision_speed_stats(window_pos, window_times)

    # Actual window duration
    actual_duration = (
        float(window_times[-1] - window_times[0]) if len(window_times) > 1 else 0.0
    )

    return PreDecisionMetrics(
        mean_speed=mean_speed,
        min_speed=min_speed_val,
        heading_mean_direction=mean_dir,
        heading_circular_variance=circ_var,
        heading_mean_resultant_length=mrl,
        window_duration=actual_duration,
        n_samples=len(window_pos),
    )


# =============================================================================
# Decision Boundary Functions
# =============================================================================


def geodesic_voronoi_labels(
    env: Environment,
    goal_bins: list[int] | NDArray[np.int_],
) -> NDArray[np.int_]:
    """Label each bin by its nearest goal using geodesic distance.

    Creates a Voronoi-like partition of the environment where each bin
    is assigned to the goal with the shortest geodesic path.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    goal_bins : list[int] or NDArray[np.int_]
        Bin indices of goal locations.

    Returns
    -------
    NDArray[np.int_], shape (n_bins,)
        Index of nearest goal for each bin (indices into goal_bins).
        Bins unreachable from all goals have label -1.

    Notes
    -----
    Performance: O(n_goals * n_bins * log(n_bins)) using Dijkstra's algorithm.
    For large environments (n_bins > 5000) with many goals (> 10), this may
    take several seconds. Consider caching results if calling repeatedly.

    Examples
    --------
    >>> left_bin = env.bin_at([10, 55])
    >>> right_bin = env.bin_at([90, 55])
    >>> labels = geodesic_voronoi_labels(env, [left_bin, right_bin])
    >>> # labels[i] == 0 means bin i is closer to left goal
    >>> # labels[i] == 1 means bin i is closer to right goal
    """
    from neurospatial.ops.distance import distance_field

    goal_bins_arr = np.asarray(goal_bins)
    n_goals = len(goal_bins_arr)
    n_bins = env.n_bins

    # Compute distance field from each goal
    distances = np.full((n_goals, n_bins), np.inf)
    for i, goal_bin in enumerate(goal_bins_arr):
        # Ensure goal_bin is a Python int, not numpy scalar
        goal_bin_int = (
            int(goal_bin.item()) if hasattr(goal_bin, "item") else int(goal_bin)
        )
        distances[i] = distance_field(
            env.connectivity, [goal_bin_int], metric="geodesic"
        )

    # Label by nearest goal
    labels = np.argmin(distances, axis=0)

    # Mark unreachable bins
    min_distances = np.min(distances, axis=0)
    labels_int: NDArray[np.int_] = labels.astype(np.int_)
    labels_int[np.isinf(min_distances)] = -1

    return labels_int


def distance_to_decision_boundary(
    env: Environment,
    trajectory_bins: NDArray[np.int_],
    goal_bins: list[int] | NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute distance to nearest decision boundary for each trajectory point.

    The decision boundary is the Voronoi edge between goal regions - the set
    of points equidistant (geodesically) from multiple goals.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    trajectory_bins : NDArray[np.int_], shape (n_samples,)
        Bin indices along the trajectory.
    goal_bins : list[int] or NDArray[np.int_]
        Bin indices of goal locations.

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Distance to nearest decision boundary at each trajectory point.
        Units match environment. Small values = near boundary = uncommitted.

    Notes
    -----
    Distance to boundary is computed as the absolute difference between
    distances to the two nearest goals. At the boundary, this difference is 0.

    Examples
    --------
    >>> distances = distance_to_decision_boundary(env, trajectory_bins, goal_bins)
    >>> commitment_mask = distances > 20.0  # Committed to a goal
    """
    from neurospatial.ops.distance import distance_field

    goal_bins_arr = np.asarray(goal_bins)
    n_goals = len(goal_bins_arr)

    if n_goals < 2:
        # With single goal, there's no boundary - return infinity
        return np.full(len(trajectory_bins), np.inf)

    # Compute distance from each goal to all bins
    all_distances = np.zeros((n_goals, env.n_bins))
    for i, goal_bin in enumerate(goal_bins_arr):
        # Ensure goal_bin is a Python int, not numpy scalar
        goal_bin_int = (
            int(goal_bin.item()) if hasattr(goal_bin, "item") else int(goal_bin)
        )
        all_distances[i] = distance_field(
            env.connectivity, [goal_bin_int], metric="geodesic"
        )

    # For each trajectory bin, compute distance to boundary
    # Boundary distance = |d1 - d2| where d1, d2 are distances to two nearest goals
    boundary_distances = np.zeros(len(trajectory_bins))

    for i, bin_idx in enumerate(trajectory_bins):
        if bin_idx < 0 or bin_idx >= env.n_bins:
            boundary_distances[i] = np.nan
            continue

        # Get distances to all goals from this bin
        dists = all_distances[:, bin_idx]

        # Sort to find two nearest
        sorted_dists = np.sort(dists)

        # Boundary distance is difference between two nearest goals
        if np.isinf(sorted_dists[0]):
            boundary_distances[i] = np.inf
        else:
            # Extract scalars properly to avoid numpy deprecation warning
            d1 = float(
                sorted_dists[0].item()
                if hasattr(sorted_dists[0], "item")
                else sorted_dists[0]
            )
            d2 = float(
                sorted_dists[1].item()
                if hasattr(sorted_dists[1], "item")
                else sorted_dists[1]
            )
            boundary_distances[i] = d2 - d1

    return boundary_distances


def detect_boundary_crossings(
    trajectory_bins: NDArray[np.int_],
    voronoi_labels: NDArray[np.int_],
    times: NDArray[np.float64],
) -> tuple[list[float], list[tuple[int, int]]]:
    """Detect when trajectory crosses decision boundaries.

    Parameters
    ----------
    trajectory_bins : NDArray[np.int_], shape (n_samples,)
        Bin indices along the trajectory.
    voronoi_labels : NDArray[np.int_], shape (n_bins,)
        Voronoi label for each bin (from geodesic_voronoi_labels).
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps (seconds).

    Returns
    -------
    crossing_times : list[float]
        Times when trajectory crossed a decision boundary.
    crossing_directions : list[tuple[int, int]]
        (from_goal_idx, to_goal_idx) for each crossing.

    Examples
    --------
    >>> crossing_times, directions = detect_boundary_crossings(
    ...     trajectory_bins, voronoi_labels, times
    ... )
    >>> print(f"Animal crossed boundary {len(crossing_times)} times")
    """
    trajectory_bins = np.asarray(trajectory_bins)
    times = np.asarray(times)

    # Get label for each trajectory point
    trajectory_labels = voronoi_labels[trajectory_bins]

    crossing_times: list[float] = []
    crossing_directions: list[tuple[int, int]] = []

    for i in range(1, len(trajectory_labels)):
        prev_label = trajectory_labels[i - 1]
        curr_label = trajectory_labels[i]

        # Skip if either is unreachable (-1)
        if prev_label == -1 or curr_label == -1:
            continue

        # Check for crossing
        if prev_label != curr_label:
            # Interpolate crossing time (assume it happened at midpoint)
            crossing_time = (times[i - 1] + times[i]) / 2
            crossing_times.append(float(crossing_time))
            crossing_directions.append((int(prev_label), int(curr_label)))

    return crossing_times, crossing_directions


# =============================================================================
# Composite Analysis Function
# =============================================================================


def compute_decision_analysis(
    env: Environment,
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    decision_region: str,
    goal_regions: list[str],
    *,
    pre_window: float = 1.0,
    min_speed: float = 5.0,
) -> DecisionAnalysisResult:
    """Compute complete decision analysis for a trajectory.

    Parameters
    ----------
    env : Environment
        Spatial environment with region definitions.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps (seconds).
    decision_region : str
        Name of decision region in env.regions (e.g., "center" for T-maze).
    goal_regions : list[str]
        Names of goal regions (e.g., ["left", "right"]).
    pre_window : float, default=1.0
        Duration of pre-decision window to analyze (seconds).
    min_speed : float, default=5.0
        Minimum speed for valid heading (units/s).

    Returns
    -------
    DecisionAnalysisResult
        Complete decision analysis including pre-decision metrics,
        boundary metrics, and chosen goal.

    Raises
    ------
    ValueError
        If decision_region or any goal_region not found in env.regions.
    ValueError
        If positions and times have different lengths.

    Examples
    --------
    >>> result = compute_decision_analysis(
    ...     env,
    ...     positions,
    ...     times,
    ...     decision_region="center",
    ...     goal_regions=["left", "right"],
    ... )
    >>> print(result.summary())
    """
    positions = np.asarray(positions)
    times = np.asarray(times)

    # Validate inputs
    if len(positions) != len(times):
        raise ValueError(
            f"positions and times must have same length. "
            f"Got positions: {len(positions)}, times: {len(times)}. "
            f"Check that both arrays cover the same time period."
        )

    # Validate regions exist
    if decision_region not in env.regions:
        available = list(env.regions.keys())
        raise ValueError(
            f"Decision region '{decision_region}' not found in environment. "
            f"Available regions: {available}."
        )

    for goal_region in goal_regions:
        if goal_region not in env.regions:
            available = list(env.regions.keys())
            raise ValueError(
                f"Goal region '{goal_region}' not found in environment. "
                f"Available regions: {available}."
            )

    # Get trajectory bins
    trajectory_bins = env.bin_at(positions)

    # Find entry time to decision region
    entry_time = decision_region_entry_time(
        trajectory_bins, times, env, decision_region
    )

    # Compute pre-decision metrics
    pre_decision = compute_pre_decision_metrics(
        positions, times, entry_time, pre_window, min_speed=min_speed
    )

    # Compute boundary metrics
    # Get the representative bin for each goal region (first bin in region)
    goal_bins = []
    for r in goal_regions:
        bins = env.bins_in_region(r)
        if len(bins) > 0:
            goal_bins.append(int(bins[0]))
        else:
            raise ValueError(
                f"Goal region '{r}' contains no bins. "
                f"Check that the region is within the environment bounds."
            )
    voronoi_labels = geodesic_voronoi_labels(env, goal_bins)

    trajectory_labels = voronoi_labels[trajectory_bins]
    boundary_distances = distance_to_decision_boundary(env, trajectory_bins, goal_bins)
    crossing_times, crossing_directions = detect_boundary_crossings(
        trajectory_bins, voronoi_labels, times
    )

    boundary = DecisionBoundaryMetrics(
        goal_labels=trajectory_labels,
        distance_to_boundary=boundary_distances,
        crossing_times=crossing_times,
        crossing_directions=crossing_directions,
    )

    # Determine chosen goal (which goal region was reached)
    chosen_goal: int | None = None
    for i, goal_region in enumerate(goal_regions):
        goal_region_bins = set(env.bins_in_region(goal_region))
        # Check if trajectory ends in this goal region
        for bin_idx in reversed(trajectory_bins):
            if bin_idx in goal_region_bins:
                chosen_goal = i
                break
        if chosen_goal is not None:
            break

    return DecisionAnalysisResult(
        entry_time=entry_time,
        pre_decision=pre_decision,
        boundary=boundary,
        chosen_goal=chosen_goal,
    )
