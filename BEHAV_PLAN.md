# Implementation Plan: Behavioral Trajectory Metrics

**Created**: 2025-12-05
**Revised**: 2025-12-05 (post second code-review)
**Status**: Ready for implementation
**Scope**: Path efficiency, goal-directed metrics, spatial decision analysis, VTE

---

## Overview

This plan implements behavioral trajectory analysis metrics from the mathematical framework document. The implementation maximizes code reuse from existing neurospatial modules.

**Key Principle**: DRY - approximately 60% of the mathematical framework is already implemented. New code wraps and extends existing functionality.

---

## Design Decisions (Post-Review)

### Parameter Naming Convention

**Decision**: Standardize on `metric` parameter (not `distance_type`).

**Rationale**: `behavioral.py` uses `metric` and is the more recent module being extended. This is consistent with `path_progress()` and `distance_to_region()`.

```python
# Correct:
metric: Literal["geodesic", "euclidean"] = "geodesic"

# Not:
distance_type: Literal["euclidean", "geodesic"] = "euclidean"
```

### Function Naming for Clarity

**Decision**: Use descriptive names that communicate meaning to neuroscientists.

| Original | Renamed | Rationale |
|----------|---------|-----------|
| `actual_path_length()` | `traveled_path_length()` | Clearer: distance animal actually traveled |
| `optimal_path_length()` | `shortest_path_length()` | Clearer: geodesic/straight-line distance |
| `integrated_absolute_rotation()` | `head_sweep_magnitude()` | Plain language, with alias to IdPhi |

### Import Paths

**Correct imports** (not re-exported at package level):

```python
from neurospatial.distance import distance_field, geodesic_distance_matrix
from neurospatial.reference_frames import heading_from_velocity
from neurospatial.metrics.circular import rayleigh_test
```

### Circular Statistics

**Decision**: Do NOT use internal `_mean_resultant_length()`. Compute directly:

```python
# Instead of importing private function:
cos_headings = np.cos(headings)
sin_headings = np.sin(headings)
mean_cos = np.mean(cos_headings)
mean_sin = np.mean(sin_headings)
mean_resultant_length = np.sqrt(mean_cos**2 + mean_sin**2)
circular_variance = 1.0 - mean_resultant_length
```

---

## Existing Code Inventory

### Already Implemented (DO NOT Reimplement)

| Math Section | Existing Function | Location |
|-------------|-------------------|----------|
| 1.1 Velocity/Speed | `compute_trajectory_curvature()` | `behavioral.py:699-820` |
| 1.2 Geodesic Distance | `geodesic_distance_matrix()`, `distance_field()` | `distance.py:32-365` |
| 2.1 Actual Path Length | `compute_step_lengths()` | `metrics/trajectory.py:167-311` |
| 2.2 Optimal Path (geodesic) | `geodesic_distance_matrix()` | `distance.py:32-80` |
| 3.1 Goal Distance | `distance_to_region()` | `behavioral.py:359-493` |
| 3.2 Heading Direction | `compute_turn_angles()`, `heading_direction_labels()` | `metrics/trajectory.py:30-164` |
| 3.4 Path Progress | `path_progress()` | `behavioral.py:182-356` |
| 3.5 Time-to-Goal | `time_to_goal()` | `behavioral.py:624-696` |
| 4.1 Decision Region | `segment_trials()`, `Trial` | `segmentation/trials.py` |
| Circular Stats | `rayleigh_test()` | `metrics/circular.py` |
| Egocentric Frames | `compute_egocentric_bearing()`, `heading_from_velocity()` | `reference_frames.py` |
| MSD | `mean_square_displacement()` | `metrics/trajectory.py:406-648` |

---

## New Modules

### Module 1: Path Efficiency

**File**: `src/neurospatial/metrics/path_efficiency.py`

#### Module Docstring

```python
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
```

#### Data Structures

```python
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
```

```python
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
            lines.append(f"  Segment {i+1}: {seg.efficiency:.1%}")
        lines.append(f"  Mean: {self.mean_efficiency:.1%}")
        lines.append(f"  Weighted: {self.weighted_efficiency:.1%}")
        return "\n".join(lines)
```

#### Functions

| Function | Signature | Description | Reuses |
|----------|-----------|-------------|--------|
| `traveled_path_length` | `(positions, *, metric="euclidean", env=None) -> float` | Sum of step lengths | `compute_step_lengths()` |
| `shortest_path_length` | `(env, start, goal, *, metric="geodesic") -> float` | Geodesic/Euclidean distance | `distance_field()` |
| `path_efficiency` | `(env, positions, goal, *, metric="geodesic") -> float` | shortest / traveled ratio | Both above |
| `time_efficiency` | `(positions, times, goal, *, reference_speed) -> float` | T_optimal / T_actual | `shortest_path_length()` |
| `angular_efficiency` | `(positions, goal) -> float` | 1 - mean(\|delta_theta\|) / pi | `compute_turn_angles()` |
| `subgoal_efficiency` | `(env, positions, subgoals, *, metric="geodesic") -> SubgoalEfficiencyResult` | Per-segment efficiency | `path_efficiency()` |
| `compute_path_efficiency` | `(env, positions, times, goal, ...) -> PathEfficiencyResult` | All metrics combined | All above |

#### Implementation Notes

```python
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
        If positions is not 2D array.

    Examples
    --------
    >>> length = traveled_path_length(positions)
    >>> print(f"Animal traveled {length:.1f} cm")
    """
    from neurospatial.metrics.trajectory import compute_step_lengths

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
    distance_type = "geodesic" if metric == "geodesic" else "euclidean"
    step_lengths = compute_step_lengths(
        positions, distance_type=distance_type, env=env
    )
    return float(np.sum(step_lengths))
```

```python
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
    from neurospatial.metrics.trajectory import compute_turn_angles

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
```

#### Error Messages

```python
# Common error patterns with helpful messages:

# 1. Mismatched array lengths
if len(positions) != len(times):
    raise ValueError(
        f"positions and times must have same length. "
        f"Got positions: {len(positions)}, times: {len(times)}. "
        f"Check that both arrays cover the same time period."
    )

# 2. Goal outside environment
if not env.contains(goal):
    bounds = env.dimension_ranges
    raise ValueError(
        f"Goal position {goal} is outside environment bounds "
        f"({bounds[0]:.1f}-{bounds[1]:.1f}, {bounds[2]:.1f}-{bounds[3]:.1f}). "
        f"\nTo fix:\n"
        f"- Verify goal coordinates match your environment units\n"
        f"- Check if environment was created with correct bin_size\n"
        f"- Use env.plot() to visualize environment extent"
    )

# 3. Empty trajectory
if len(positions) == 0:
    raise ValueError(
        "Cannot compute path efficiency: positions array is empty. "
        "Provide at least 2 positions to compute path length."
    )

# 4. Disconnected graph (geodesic)
if np.isinf(shortest_length):
    raise ValueError(
        f"No path exists between start {start} and goal {goal}. "
        f"The environment graph may be disconnected. "
        f"Check that both positions are in connected regions of the environment."
    )
```

---

### Module 2: Goal-Directed Metrics

**File**: `src/neurospatial/metrics/goal_directed.py`

#### Module Docstring

```python
"""Goal-directed navigation metrics.

This module measures how directly an animal navigates toward a goal,
including instantaneous alignment, overall bias, and approach dynamics.

Key Concepts
------------
- **Goal alignment**: Cosine similarity between movement and goal direction.
  +1 = moving toward goal, -1 = moving away, 0 = orthogonal.
- **Goal bias**: Average alignment over trajectory. Positive = net approach.
- **Approach rate**: Rate of distance decrease toward goal (cm/s).

Example
-------
>>> from neurospatial.metrics import compute_goal_directed_metrics
>>> result = compute_goal_directed_metrics(env, positions, times, goal)
>>> print(f"Goal bias: {result.goal_bias:.2f}")  # Range [-1, 1]
>>> if result.goal_bias > 0.5:
...     print("Strong goal-directed navigation")

References
----------
.. [1] Johnson, A., & Redish, A. D. (2007). Neural ensembles in CA3 transiently
       encode paths forward of the animal at a decision point. J Neurosci.
       DOI: 10.1523/JNEUROSCI.3761-07.2007
"""
```

#### Data Structures

```python
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
        """Human-readable summary for printing."""
        lines = [
            f"Goal-directed metrics:",
            f"  Goal bias: {self.goal_bias:.2f} (range [-1, 1])",
            f"  Approach rate: {self.mean_approach_rate:.1f} units/s",
            f"  Distance: {self.goal_distance_at_start:.1f} -> {self.goal_distance_at_end:.1f} units",
            f"  Closest approach: {self.min_distance_to_goal:.1f} units",
        ]
        if self.time_to_goal is not None:
            lines.append(f"  Time to goal: {self.time_to_goal:.2f} s")
        else:
            lines.append("  Time to goal: not reached")
        return "\n".join(lines)
```

#### Functions

| Function | Signature | Description | Reuses |
|----------|-----------|-------------|--------|
| `goal_vector` | `(positions, goal) -> NDArray` | g - x(t) for each timepoint | Vectorized numpy |
| `goal_direction` | `(positions, goal) -> NDArray` | atan2 of goal vector | `goal_vector()` |
| `instantaneous_goal_alignment` | `(positions, times, goal, *, min_speed=5.0) -> NDArray` | cos(angle between v and goal_vector) | `heading_from_velocity()` |
| `goal_bias` | `(positions, times, goal, *, min_speed=5.0) -> float` | Mean alignment over trajectory | `instantaneous_goal_alignment()` |
| `approach_rate` | `(positions, times, goal, *, metric="geodesic", env=None) -> NDArray` | d/dt of distance_to_goal | `distance_to_region()` |
| `compute_goal_directed_metrics` | `(env, positions, times, goal, ...) -> GoalDirectedMetrics` | All metrics combined | All above |

#### Implementation Notes

```python
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
        Goal position in allocentric frame (same coordinate system).
        Must have same number of dimensions as positions.

    Returns
    -------
    NDArray[np.float64], shape (n_samples, n_dims)
        Vector from each position to goal.

    Raises
    ------
    ValueError
        If goal dimensions don't match positions dimensions.
    """
    goal = np.asarray(goal)
    if goal.shape[0] != positions.shape[1]:
        raise ValueError(
            f"Goal has {goal.shape[0]} dimensions but positions have "
            f"{positions.shape[1]} dimensions. Both must match."
        )
    return goal[np.newaxis, :] - positions


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
        Samples below this speed are masked as NaN (stationary periods
        have undefined movement direction).

    Returns
    -------
    NDArray[np.float64], shape (n_samples,)
        Cosine of angle between velocity and goal direction.
        Range [-1, 1]. NaN for stationary periods (speed < min_speed).
    """
    from neurospatial.reference_frames import heading_from_velocity

    # Compute dt from times (heading_from_velocity expects scalar dt)
    # Use median dt to handle irregular sampling
    dt = float(np.median(np.diff(times)))

    # Compute velocity heading
    velocity_heading = heading_from_velocity(positions, dt, min_speed=min_speed)

    # Compute goal direction at each position
    goal_vec = goal_vector(positions, goal)
    goal_heading = np.arctan2(goal_vec[:, 1], goal_vec[:, 0])

    # Compute alignment as cos(velocity_heading - goal_heading)
    angle_diff = velocity_heading - goal_heading
    alignment = np.cos(angle_diff)

    return alignment
```

---

### Module 3: Spatial Decision Analysis

**File**: `src/neurospatial/metrics/decision_analysis.py`

#### Module Docstring

```python
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
...     env, positions, times,
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
```

#### Data Structures

```python
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

    Visualization (2-goal example):
    ```
         Goal A          Goal B
           *               *
           |               |
           |    Boundary   |
           |       |       |
           |       |       |
           +-------+-------+
                   ^
            Decision point
    ```

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
            f"Decision boundary: {self.n_crossings} crossings, "
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
            f"Decision analysis:",
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
```

#### Functions

| Function | Signature | Description | Reuses |
|----------|-----------|-------------|--------|
| `decision_region_entry_time` | `(trajectory_bins, times, env, region) -> float` | First entry time | `segment_trials()` logic |
| `extract_pre_decision_window` | `(positions, times, entry_time, window_duration) -> tuple[NDArray, NDArray]` | Slice trajectory | Array slicing |
| `pre_decision_heading_stats` | `(positions, times, *, min_speed=5.0) -> tuple[float, float, float]` | Circular stats | Direct computation |
| `pre_decision_speed_stats` | `(positions, times) -> tuple[float, float]` | Mean, min speed | `np.linalg.norm()` |
| `compute_pre_decision_metrics` | `(positions, times, entry_time, window_duration, ...) -> PreDecisionMetrics` | Combined | All above |
| `geodesic_voronoi_labels` | `(env, goal_bins) -> NDArray[np.int_]` | Label bins by nearest goal | `distance_field()` |
| `distance_to_decision_boundary` | `(env, trajectory_bins, goal_bins) -> NDArray[np.float64]` | Distance to Voronoi edges | `geodesic_voronoi_labels()` |
| `detect_boundary_crossings` | `(trajectory_bins, voronoi_labels, times) -> tuple[list, list]` | Crossing events | Label change detection |
| `compute_decision_analysis` | `(env, positions, times, decision_region, goal_regions, ...) -> DecisionAnalysisResult` | Complete analysis | All above |

#### Implementation Notes

```python
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
        Timestamps.
    min_speed : float, default=5.0
        Minimum speed for valid heading (units/s).
        Stationary periods are excluded from statistics.

    Returns
    -------
    mean_direction : float
        Circular mean heading in radians.
    circular_variance : float
        Circular variance, range [0, 1].
    mean_resultant_length : float
        Mean resultant length, range [0, 1].

    Notes
    -----
    Circular statistics are computed directly (not using internal functions):

    mean_resultant_length = sqrt(mean(cos(theta))^2 + mean(sin(theta))^2)
    circular_variance = 1 - mean_resultant_length
    mean_direction = atan2(mean(sin(theta)), mean(cos(theta)))
    """
    from neurospatial.reference_frames import heading_from_velocity

    # Compute dt from times (heading_from_velocity expects scalar dt)
    # Use median dt to handle irregular sampling
    dt = float(np.median(np.diff(times)))

    # Get headings
    headings = heading_from_velocity(positions, dt, min_speed=min_speed)

    # Filter out NaN (stationary periods)
    valid_headings = headings[~np.isnan(headings)]

    if len(valid_headings) == 0:
        return 0.0, 1.0, 0.0  # No valid headings: undefined direction, max variance

    # Compute circular statistics directly
    cos_headings = np.cos(valid_headings)
    sin_headings = np.sin(valid_headings)
    mean_cos = np.mean(cos_headings)
    mean_sin = np.mean(sin_headings)

    mean_resultant_length = np.sqrt(mean_cos**2 + mean_sin**2)
    circular_variance = 1.0 - mean_resultant_length
    mean_direction = np.arctan2(mean_sin, mean_cos)

    return float(mean_direction), float(circular_variance), float(mean_resultant_length)


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
    """
    from neurospatial.distance import distance_field

    goal_bins = np.asarray(goal_bins)
    n_goals = len(goal_bins)
    n_bins = env.n_bins

    # Compute distance field from each goal
    distances = np.full((n_goals, n_bins), np.inf)
    for i, goal_bin in enumerate(goal_bins):
        distances[i] = distance_field(
            env.connectivity, [int(goal_bin)], metric="geodesic"
        )

    # Label by nearest goal
    labels = np.argmin(distances, axis=0)

    # Mark unreachable bins
    min_distances = np.min(distances, axis=0)
    labels[np.isinf(min_distances)] = -1

    return labels.astype(np.int_)
```

---

### Module 4: VTE Metrics

**File**: `src/neurospatial/metrics/vte.py`

#### Module Docstring

```python
"""Vicarious Trial and Error (VTE) detection and analysis.

VTE refers to hesitation behavior at decision points, characterized by:
- **Head sweeping**: Looking back and forth between options (high IdPhi)
- **Pausing**: Slowing down or stopping at the choice point (low speed)

These behaviors are thought to reflect deliberative decision-making,
as opposed to habitual or reflexive choices.

Terminology
-----------
- **IdPhi** (Integrated absolute head rotation): Sum of absolute heading changes
  in a time window. High values indicate "scanning" behavior.
- **zIdPhi**: Z-scored IdPhi relative to session baseline. Standardizes across
  animals and sessions for comparison.
- **VTE index**: Combined measure of head sweeping and slowing.

Example
-------
>>> from neurospatial.metrics import compute_vte_session
>>> result = compute_vte_session(
...     positions, times, trials,
...     decision_region="center",
...     window_duration=1.0,
... )
>>> print(f"VTE trials: {result.n_vte_trials}/{len(result.trial_results)}")
>>> for trial in result.trial_results:
...     if trial.is_vte:
...         print(f"  Trial at {trial.window_end:.1f}s: IdPhi={trial.head_sweep_magnitude:.2f}")

References
----------
.. [1] Redish, A. D. (2016). Vicarious trial and error. Nat Rev Neurosci.
       DOI: 10.1038/nrn.2015.30
.. [2] Papale, A. E., et al. (2012). Interplay between hippocampal sharp-wave
       ripple events and vicarious trial and error behaviors. Neuron.
       DOI: 10.1016/j.neuron.2012.10.018
.. [3] Muenzinger, K. F. (1938). Vicarious trial and error at a point of choice.
       J Genet Psychol.
"""
```

#### Data Structures

```python
@dataclass(frozen=True)
class VTETrialResult:
    """VTE metrics for a single trial.

    Attributes
    ----------
    head_sweep_magnitude : float
        Sum of |delta_theta| in pre-decision window (radians).
        Also known as IdPhi (Integrated absolute Phi).
        High values indicate looking back and forth between options.
    z_head_sweep : float or None
        Z-scored head sweep magnitude relative to session baseline.
        None if session statistics not computed (single trial analysis).
    mean_speed : float
        Mean speed in pre-decision window (environment units/s).
    min_speed : float
        Minimum speed in pre-decision window (units/s).
        Near-zero indicates a pause.
    z_speed_inverse : float or None
        Z-scored inverse speed (higher = slower, more VTE-like).
        None if session statistics not computed.
    vte_index : float or None
        Combined VTE index: alpha * z_head_sweep + (1-alpha) * z_speed_inverse.
        None if session statistics not computed.
    is_vte : bool or None
        True if vte_index > threshold. None if not classified.
    window_start : float
        Start time of analysis window (seconds).
    window_end : float
        End time of window (decision region entry time, seconds).
    """

    head_sweep_magnitude: float
    z_head_sweep: float | None
    mean_speed: float
    min_speed: float
    z_speed_inverse: float | None
    vte_index: float | None
    is_vte: bool | None
    window_start: float
    window_end: float

    # Aliases for common terminology
    @property
    def idphi(self) -> float:
        """Alias for head_sweep_magnitude (IdPhi terminology)."""
        return self.head_sweep_magnitude

    @property
    def z_idphi(self) -> float | None:
        """Alias for z_head_sweep (zIdPhi terminology)."""
        return self.z_head_sweep

    def summary(self) -> str:
        """Human-readable summary."""
        vte_str = "VTE" if self.is_vte else "non-VTE" if self.is_vte is not None else "unclassified"
        return (
            f"Trial [{self.window_start:.1f}-{self.window_end:.1f}s]: "
            f"IdPhi={self.head_sweep_magnitude:.2f} rad, "
            f"speed={self.mean_speed:.1f}, {vte_str}"
        )


@dataclass(frozen=True)
class VTESessionResult:
    """VTE analysis for an entire session.

    Attributes
    ----------
    trial_results : list[VTETrialResult]
        Per-trial VTE metrics with z-scores computed.
    mean_head_sweep : float
        Session mean of head sweep magnitude (for z-scoring).
    std_head_sweep : float
        Session std of head sweep magnitude.
    mean_speed : float
        Session mean of pre-decision mean speed.
    std_speed : float
        Session std of pre-decision mean speed.
    n_vte_trials : int
        Number of trials classified as VTE.
    vte_fraction : float
        Fraction of trials classified as VTE.
    """

    trial_results: list[VTETrialResult]
    mean_head_sweep: float
    std_head_sweep: float
    mean_speed: float
    std_speed: float
    n_vte_trials: int
    vte_fraction: float

    # Aliases for common terminology
    @property
    def mean_idphi(self) -> float:
        """Alias for mean_head_sweep (IdPhi terminology)."""
        return self.mean_head_sweep

    @property
    def std_idphi(self) -> float:
        """Alias for std_head_sweep (IdPhi terminology)."""
        return self.std_head_sweep

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"VTE session: {self.n_vte_trials}/{len(self.trial_results)} "
            f"VTE trials ({self.vte_fraction:.1%})\n"
            f"  Head sweep: mean={self.mean_head_sweep:.2f}, std={self.std_head_sweep:.2f}\n"
            f"  Speed: mean={self.mean_speed:.1f}, std={self.std_speed:.1f}"
        )

    def get_vte_trials(self) -> list[VTETrialResult]:
        """Return only trials classified as VTE."""
        return [t for t in self.trial_results if t.is_vte]
```

#### Functions

| Function | Signature | Description | Reuses |
|----------|-----------|-------------|--------|
| `wrap_angle` | `(angle) -> NDArray[np.float64]` | Wrap to (-pi, pi] | Vectorized modulo |
| `head_sweep_magnitude` | `(headings) -> float` | Sum of \|delta_theta\| (alias: integrated_absolute_rotation) | `wrap_angle()` |
| `head_sweep_from_positions` | `(positions, times, *, min_speed=5.0) -> float` | IdPhi from trajectory | `heading_from_velocity()` |
| `normalize_vte_scores` | `(head_sweeps, speeds) -> tuple[NDArray, NDArray]` | Z-score across trials | Vectorized stats |
| `compute_vte_index` | `(z_head_sweep, z_speed_inv, *, alpha=0.5) -> float` | Combined index | Weighted sum |
| `classify_vte` | `(vte_index, *, threshold=0.5) -> bool` | VTE if index > threshold | Comparison |
| `compute_vte_trial` | `(positions, times, entry_time, window_duration, ...) -> VTETrialResult` | Single trial | Multiple above |
| `compute_vte_session` | `(positions, times, trials, decision_region, ...) -> VTESessionResult` | Full session | All above |

#### Implementation Notes

```python
def wrap_angle(angle: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wrap angle to (-pi, pi].

    Parameters
    ----------
    angle : NDArray[np.float64]
        Angles in radians.

    Returns
    -------
    NDArray[np.float64]
        Angles wrapped to (-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def head_sweep_magnitude(headings: NDArray[np.float64]) -> float:
    """Compute integrated absolute head rotation (IdPhi).

    Sums the absolute value of heading changes across a trajectory window.
    High values indicate back-and-forth head movements ("scanning").

    Parameters
    ----------
    headings : NDArray[np.float64], shape (n_samples,)
        Heading angles in radians.

    Returns
    -------
    float
        Sum of absolute heading changes in radians.
        Returns 0.0 if fewer than 2 valid samples.

    Notes
    -----
    This is the core VTE metric from Papale et al. (2012) and Redish (2016).
    Also known as "IdPhi" (integrated absolute dphi/dt * dt).
    """
    # Filter NaN values
    valid_headings = headings[~np.isnan(headings)]

    if len(valid_headings) < 2:
        return 0.0

    delta = wrap_angle(np.diff(valid_headings))
    return float(np.sum(np.abs(delta)))


# Alias for backward compatibility and paper terminology
integrated_absolute_rotation = head_sweep_magnitude


def head_sweep_from_positions(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    min_speed: float = 5.0,
) -> float:
    """Compute head sweep magnitude (IdPhi) from position trajectory.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
    min_speed : float, default=5.0
        Minimum speed for valid heading (units/s).
        Stationary periods are excluded.

    Returns
    -------
    float
        Sum of absolute heading changes (radians).
        Returns 0.0 if fewer than 2 valid heading samples.
    """
    from neurospatial.reference_frames import heading_from_velocity

    # Compute dt from times (heading_from_velocity expects scalar dt)
    # Use median dt to handle irregular sampling
    dt = float(np.median(np.diff(times)))

    # Get headings
    headings = heading_from_velocity(positions, dt, min_speed=min_speed)

    return head_sweep_magnitude(headings)


def normalize_vte_scores(
    head_sweeps: NDArray[np.float64],
    speeds: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Z-score VTE metrics across trials.

    Parameters
    ----------
    head_sweeps : NDArray[np.float64], shape (n_trials,)
        Head sweep magnitude for each trial.
    speeds : NDArray[np.float64], shape (n_trials,)
        Mean speed for each trial.

    Returns
    -------
    z_head_sweeps : NDArray[np.float64], shape (n_trials,)
        Z-scored head sweeps.
    z_speed_inverse : NDArray[np.float64], shape (n_trials,)
        Z-scored inverse speed (higher = slower = more VTE-like).

    Raises
    ------
    ValueError
        If arrays have different lengths.

    Warns
    -----
    UserWarning
        If std is zero (no variation across trials), z-scores will be NaN.
    """
    import warnings

    if len(head_sweeps) != len(speeds):
        raise ValueError(
            f"head_sweeps and speeds must have same length. "
            f"Got {len(head_sweeps)} and {len(speeds)}."
        )

    # Z-score head sweeps
    mean_hs = np.mean(head_sweeps)
    std_hs = np.std(head_sweeps)
    if std_hs < 1e-10:
        warnings.warn(
            "No variation in head sweep magnitude across trials (std=0). "
            "All trials have identical head sweep behavior. "
            "Z-scores will be 0, and VTE classification may not be meaningful. "
            "Consider adjusting window_duration or min_speed parameters.",
            stacklevel=2,
        )
        z_head_sweeps = np.zeros_like(head_sweeps)
    else:
        z_head_sweeps = (head_sweeps - mean_hs) / std_hs

    # Z-score inverse speed (invert so slower = higher score)
    mean_spd = np.mean(speeds)
    std_spd = np.std(speeds)
    if std_spd < 1e-10:
        warnings.warn(
            "No variation in speed across trials (std=0). "
            "All trials have identical speed behavior. "
            "Z-scores will be 0, and VTE classification may not be meaningful. "
            "Consider adjusting window_duration or min_speed parameters.",
            stacklevel=2,
        )
        z_speed_inverse = np.zeros_like(speeds)
    else:
        # Invert: higher speed -> lower z, lower speed -> higher z
        z_speed_inverse = -(speeds - mean_spd) / std_spd

    return z_head_sweeps, z_speed_inverse


def compute_vte_session(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    trials: list[Trial],
    decision_region: str,
    env: Environment,
    *,
    window_duration: float = 1.0,
    min_speed: float = 5.0,
    alpha: float = 0.5,
    vte_threshold: float = 0.5,
) -> VTESessionResult:
    """Compute VTE metrics for all trials in a session.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates for entire session.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps for entire session.
    trials : list[Trial]
        Trial segmentation from segment_trials().
    decision_region : str
        Name of decision region in env.regions.
    env : Environment
        Environment with region definitions.
    window_duration : float, default=1.0
        Duration of pre-decision window in seconds.
        Typical values: 0.5-2.0s depending on maze size and task.
    min_speed : float, default=5.0
        Minimum speed for valid heading (units/s).
    alpha : float, default=0.5
        Weight for head sweep in VTE index (1-alpha for speed).
        Default 0.5 weights both equally.
    vte_threshold : float, default=0.5
        Threshold for VTE classification.
        Trial is VTE if vte_index > threshold.

    Returns
    -------
    VTESessionResult
        Session-level VTE analysis with per-trial metrics.
    """
    from neurospatial.metrics.decision_analysis import (
        decision_region_entry_time,
        extract_pre_decision_window,
    )

    # First pass: compute raw metrics for all trials
    raw_head_sweeps = []
    raw_speeds = []
    trial_windows = []  # Store (window_start, window_end) for each trial

    for trial in trials:
        # Get entry time to decision region
        mask = (times >= trial.start_time) & (times <= trial.end_time)
        trial_bins = env.bin_at(positions[mask])
        trial_times = times[mask]

        try:
            entry_time = decision_region_entry_time(
                trial_bins, trial_times, env, decision_region
            )
        except ValueError:
            # Trial never enters decision region - skip
            continue

        # Extract pre-decision window
        window_positions, window_times = extract_pre_decision_window(
            positions, times, entry_time, window_duration
        )

        if len(window_positions) < 3:
            # Not enough samples for heading analysis - skip
            continue

        # Compute head sweep magnitude
        head_sweep = head_sweep_from_positions(
            window_positions, window_times, min_speed=min_speed
        )

        # Compute mean speed
        dt = np.diff(window_times)
        velocity = np.diff(window_positions, axis=0) / dt[:, np.newaxis]
        speeds = np.linalg.norm(velocity, axis=1)
        mean_speed = float(np.mean(speeds))

        raw_head_sweeps.append(head_sweep)
        raw_speeds.append(mean_speed)
        trial_windows.append((entry_time - window_duration, entry_time))

    # Convert to arrays
    head_sweeps_arr = np.array(raw_head_sweeps)
    speeds_arr = np.array(raw_speeds)

    if len(head_sweeps_arr) == 0:
        # No valid trials
        return VTESessionResult(
            trial_results=[],
            mean_head_sweep=0.0,
            std_head_sweep=0.0,
            mean_speed=0.0,
            std_speed=0.0,
            n_vte_trials=0,
            vte_fraction=0.0,
        )

    # Compute session statistics
    mean_hs = float(np.mean(head_sweeps_arr))
    std_hs = float(np.std(head_sweeps_arr))
    mean_spd = float(np.mean(speeds_arr))
    std_spd = float(np.std(speeds_arr))

    # Second pass: compute z-scores and classify
    z_head_sweeps, z_speed_inv = normalize_vte_scores(head_sweeps_arr, speeds_arr)

    trial_results = []
    n_vte = 0

    for i in range(len(head_sweeps_arr)):
        # Compute VTE index
        vte_idx = alpha * z_head_sweeps[i] + (1 - alpha) * z_speed_inv[i]

        # Classify
        is_vte = vte_idx > vte_threshold
        if is_vte:
            n_vte += 1

        # Compute min speed for this trial
        window_start, window_end = trial_windows[i]
        mask = (times >= window_start) & (times <= window_end)
        window_positions = positions[mask]
        window_times = times[mask]
        dt = np.diff(window_times)
        velocity = np.diff(window_positions, axis=0) / dt[:, np.newaxis]
        min_speed_val = float(np.min(np.linalg.norm(velocity, axis=1)))

        trial_results.append(
            VTETrialResult(
                head_sweep_magnitude=head_sweeps_arr[i],
                z_head_sweep=float(z_head_sweeps[i]),
                mean_speed=speeds_arr[i],
                min_speed=min_speed_val,
                z_speed_inverse=float(z_speed_inv[i]),
                vte_index=float(vte_idx),
                is_vte=is_vte,
                window_start=window_start,
                window_end=window_end,
            )
        )

    return VTESessionResult(
        trial_results=trial_results,
        mean_head_sweep=mean_hs,
        std_head_sweep=std_hs,
        mean_speed=mean_spd,
        std_speed=std_spd,
        n_vte_trials=n_vte,
        vte_fraction=n_vte / len(trial_results) if trial_results else 0.0,
    )
```

---

## File Structure

```
src/neurospatial/metrics/
├── __init__.py                    # Update exports
├── trajectory.py                  # EXISTING (no changes)
├── circular.py                    # EXISTING (no changes)
├── path_efficiency.py             # NEW
├── goal_directed.py               # NEW
├── decision_analysis.py           # NEW
└── vte.py                         # NEW

tests/metrics/
├── test_path_efficiency.py        # NEW
├── test_goal_directed.py          # NEW
├── test_decision_analysis.py      # NEW
└── test_vte.py                    # NEW
```

---

## Dependencies Between Modules

```
path_efficiency.py
    └── uses: compute_step_lengths() from metrics/trajectory
    └── uses: distance_field() from neurospatial.distance
    └── uses: compute_turn_angles() from metrics/trajectory

goal_directed.py
    └── uses: distance_to_region() from behavioral
    └── uses: heading_from_velocity() from reference_frames

decision_analysis.py
    └── uses: segment_trials() from segmentation
    └── uses: distance_field() from neurospatial.distance
    └── uses: heading_from_velocity() from reference_frames
    └── uses: rayleigh_test() from metrics/circular (for validation only)

vte.py
    └── uses: heading_from_velocity() from reference_frames
    └── uses: decision_region_entry_time() from decision_analysis
    └── uses: extract_pre_decision_window() from decision_analysis
```

---

## Implementation Order

### Phase 1: Path Efficiency (Foundation)

- [ ] Create `metrics/path_efficiency.py` with module docstring
- [ ] Implement `PathEfficiencyResult` dataclass with helper methods
- [ ] Implement `traveled_path_length()` wrapping `compute_step_lengths()`
- [ ] Implement `shortest_path_length()` using `distance_field()`
- [ ] Implement `path_efficiency()` with edge case handling
- [ ] Implement `time_efficiency()`
- [ ] Implement `angular_efficiency()` with < 3 position handling
- [ ] Implement `SubgoalEfficiencyResult` and `subgoal_efficiency()`
- [ ] Implement `compute_path_efficiency()` combining all metrics
- [ ] Add comprehensive error messages for common mistakes
- [ ] Write tests in `tests/metrics/test_path_efficiency.py`
- [ ] Add exports to `metrics/__init__.py`

### Phase 2: Goal-Directed Metrics

- [ ] Create `metrics/goal_directed.py` with module docstring
- [ ] Implement `GoalDirectedMetrics` dataclass with helper methods
- [ ] Implement `goal_vector()` with dimension validation
- [ ] Implement `goal_direction()`
- [ ] Implement `instantaneous_goal_alignment()` with min_speed filtering
- [ ] Implement `goal_bias()`
- [ ] Implement `approach_rate()`
- [ ] Implement `compute_goal_directed_metrics()`
- [ ] Add comprehensive error messages
- [ ] Write tests in `tests/metrics/test_goal_directed.py`
- [ ] Add exports to `metrics/__init__.py`

### Phase 3: Decision Analysis

- [ ] Create `metrics/decision_analysis.py` with module docstring
- [ ] Implement `PreDecisionMetrics` dataclass with `suggests_deliberation()`
- [ ] Implement `decision_region_entry_time()`
- [ ] Implement `extract_pre_decision_window()`
- [ ] Implement `pre_decision_heading_stats()` with direct circular computation
- [ ] Implement `pre_decision_speed_stats()`
- [ ] Implement `compute_pre_decision_metrics()`
- [ ] Implement `DecisionBoundaryMetrics` dataclass
- [ ] Implement `geodesic_voronoi_labels()` with performance note
- [ ] Implement `distance_to_decision_boundary()`
- [ ] Implement `detect_boundary_crossings()`
- [ ] Implement `DecisionAnalysisResult` and `compute_decision_analysis()`
- [ ] Write tests in `tests/metrics/test_decision_analysis.py`
- [ ] Add exports to `metrics/__init__.py`

### Phase 4: VTE Metrics

- [ ] Create `metrics/vte.py` with module docstring and terminology glossary
- [ ] Implement `wrap_angle()` utility with return type
- [ ] Implement `head_sweep_magnitude()` with empty array handling
- [ ] Add `integrated_absolute_rotation` alias
- [ ] Implement `head_sweep_from_positions()`
- [ ] Implement `VTETrialResult` dataclass with aliases
- [ ] Implement `normalize_vte_scores()` with std=0 warning
- [ ] Implement `compute_vte_index()`
- [ ] Implement `classify_vte()`
- [ ] Implement `compute_vte_trial()`
- [ ] Implement `VTESessionResult` dataclass with aliases
- [ ] Implement `compute_vte_session()`
- [ ] Write tests in `tests/metrics/test_vte.py`
- [ ] Add exports to `metrics/__init__.py`

### Phase 5: Integration

- [ ] Update `.claude/QUICKSTART.md` with behavioral metrics examples
- [ ] Update `.claude/API_REFERENCE.md` with new imports
- [ ] Add workflow template to module docstrings
- [ ] Run full test suite: `uv run pytest tests/metrics/`
- [ ] Run type checker: `uv run mypy src/neurospatial/metrics/`
- [ ] Run linter: `uv run ruff check src/neurospatial/metrics/`

---

## Testing Strategy

### Unit Tests

Each function tested independently with:
- Basic functionality with known inputs/outputs
- Edge cases (empty arrays, single points, NaN handling)
- Parameter validation (error messages are helpful)

### Integration Tests

- Round-trip consistency: simulated trajectory -> metrics -> verify expected values
- Cross-module consistency: VTE uses decision_analysis correctly
- Comparison with existing functions: path_efficiency consistent with path_progress

### Regression Tests

**Path Efficiency**:
- Straight path: efficiency should be 1.0
- Random walk: efficiency < 0.5
- U-turn path: efficiency ~ 0.5 (doubles length)
- < 2 positions: returns NaN efficiency

**Goal-Directed**:
- Straight approach to goal: goal_bias > 0.8
- Moving away from goal: goal_bias < -0.5
- Circular path around goal: goal_bias ~ 0.0
- Stationary: goal_bias is NaN

**VTE Detection**:
- High head sweep + low speed: classified as VTE
- Low head sweep + high speed: not classified as VTE
- All identical trials: z-scores are 0, warning issued
- Single trial: z-scores are None

---

## API Design Principles

1. **Consistent Signatures**: All main functions take `(env, positions, times, ...)` as first args
2. **Keyword-Only Options**: All optional parameters are keyword-only after positional
3. **Metric Parameter**: Functions supporting geodesic/Euclidean use `metric=` (not `distance_type`)
4. **Return Dataclasses**: Complex results return frozen dataclasses with helper methods
5. **NumPy Docstrings**: All functions have complete NumPy-format docstrings
6. **Vectorized**: No loops over timepoints; all operations vectorized
7. **Helpful Errors**: Error messages explain what's wrong AND how to fix it
8. **Plain Language**: Function names and docstrings use neuroscience terminology

---

## Estimated Effort

| Module | New LOC | Reused Functions | Test LOC |
|--------|---------|------------------|----------|
| `path_efficiency.py` | ~350 | 4 | ~250 |
| `goal_directed.py` | ~400 | 5 | ~300 |
| `decision_analysis.py` | ~500 | 5 | ~400 |
| `vte.py` | ~450 | 4 | ~350 |
| **Total** | **~1,700** | **18** | **~1,300** |

---

## References

### Mathematical Framework

- **Path efficiency**: Batschelet, E. (1981). Circular Statistics in Biology. Academic Press.
  - Used in: `path_efficiency()`, `angular_efficiency()`

- **Goal-directed**: Johnson, A., & Redish, A. D. (2007). Neural ensembles in CA3.
  - DOI: 10.1523/JNEUROSCI.3761-07.2007
  - Used in: `goal_bias()`, `approach_rate()`

- **VTE (IdPhi)**: Redish, A. D. (2016). Vicarious trial and error. Nat Rev Neurosci.
  - DOI: 10.1038/nrn.2015.30
  - Used in: `head_sweep_magnitude()`, `compute_vte_session()`

- **VTE behavior**: Papale, A. E., et al. (2012). Hippocampal ripples and VTE. Neuron.
  - DOI: 10.1016/j.neuron.2012.10.018
  - Used in: VTE classification approach

- **Circular statistics**: Mardia, K. V., & Jupp, P. E. (2000). Directional Statistics.
  - Used in: `pre_decision_heading_stats()`

### Existing Code

- `neurospatial.metrics.trajectory`: Step lengths, turn angles, MSD
- `neurospatial.behavioral`: Path progress, distance to region
- `neurospatial.distance`: Geodesic distance functions
- `neurospatial.metrics.circular`: Circular statistics (public API only)
- `neurospatial.reference_frames`: Heading computation
