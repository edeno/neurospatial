# Implementation Plan: Behavioral Trajectory Metrics

**Created**: 2025-12-05
**Status**: Ready for implementation
**Scope**: Path efficiency, goal-directed metrics, spatial decision analysis, VTE

---

## Overview

This plan implements behavioral trajectory analysis metrics from the mathematical framework document. The implementation maximizes code reuse from existing neurospatial modules.

**Key Principle**: DRY - approximately 60% of the mathematical framework is already implemented. New code wraps and extends existing functionality.

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
| Circular Stats | `rayleigh_test()`, `_mean_resultant_length()` | `metrics/circular.py` |
| Egocentric Frames | `compute_egocentric_bearing()`, `heading_from_velocity()` | `reference_frames.py` |
| MSD | `mean_square_displacement()` | `metrics/trajectory.py:406-648` |

---

## New Modules

### Module 1: Path Efficiency

**File**: `src/neurospatial/metrics/path_efficiency.py`

#### Data Structures

```python
@dataclass(frozen=True)
class PathEfficiencyResult:
    """Path efficiency metrics for a trajectory segment.

    Attributes
    ----------
    actual_length : float
        Total path length traveled (sum of step lengths).
    optimal_length : float
        Geodesic distance from start to goal.
    efficiency : float
        Ratio optimal_length / actual_length. Range (0, 1].
    time_efficiency : float or None
        Ratio T_optimal / T_actual if reference_speed provided.
    angular_efficiency : float
        1 - mean(|delta_theta|) / pi. Range [0, 1].
    start_position : NDArray[np.float64]
        Start position coordinates.
    goal_position : NDArray[np.float64]
        Goal position coordinates.
    """
```

```python
@dataclass(frozen=True)
class SubgoalEfficiencyResult:
    """Path efficiency with subgoal decomposition.

    Attributes
    ----------
    segment_efficiencies : list[PathEfficiencyResult]
        Efficiency for each segment between subgoals.
    mean_efficiency : float
        Mean efficiency across segments.
    weighted_efficiency : float
        Efficiency weighted by segment optimal length.
    subgoal_positions : NDArray[np.float64]
        Positions of subgoals, shape (n_subgoals, n_dims).
    """
```

#### Functions

| Function | Description | Reuses |
|----------|-------------|--------|
| `actual_path_length(positions, *, metric, env)` | Sum of step lengths | `compute_step_lengths()` |
| `optimal_path_length(env, start, goal, *, metric)` | Geodesic/Euclidean distance | `geodesic_distance_matrix()` |
| `path_efficiency(env, positions, goal, *, metric)` | optimal / actual ratio | Both above |
| `time_normalized_efficiency(positions, times, goal, *, reference_speed)` | T_optimal / T_actual | `optimal_path_length()` |
| `angular_path_efficiency(positions, goal)` | 1 - mean(\|delta_theta\|) / pi | `compute_turn_angles()` |
| `subgoal_path_efficiency(env, positions, subgoals, *, metric)` | Per-segment efficiency | `path_efficiency()` |
| `compute_path_efficiency(env, positions, times, goal, ...)` | All metrics combined | All above |

#### Implementation Notes

- `actual_path_length()`: Direct wrapper around `compute_step_lengths().sum()`
- `optimal_path_length()`: For small envs (<5000 bins), use cached distance matrix; for large envs, use `distance_field()`
- `angular_path_efficiency()`: Compute goal direction once, then vectorized angle difference

---

### Module 2: Goal-Directed Metrics

**File**: `src/neurospatial/metrics/goal_directed.py`

#### Data Structures

```python
@dataclass(frozen=True)
class GoalDirectedMetrics:
    """Goal-directed navigation metrics for a trajectory.

    Attributes
    ----------
    goal_bias : float
        Mean instantaneous goal alignment, range [-1, 1].
        Positive = moving toward goal, negative = away.
    mean_approach_rate : float
        Mean rate of distance change (negative = approaching).
    time_to_goal : float or None
        Time until goal region entered (None if not reached).
    min_distance_to_goal : float
        Closest approach to goal during trajectory.
    goal_distance_at_start : float
        Distance to goal at trajectory start.
    goal_distance_at_end : float
        Distance to goal at trajectory end.
    """
```

#### Functions

| Function | Description | Reuses |
|----------|-------------|--------|
| `goal_vector(positions, goal)` | g - x(t) for each timepoint | Vectorized numpy |
| `goal_direction(positions, goal)` | atan2 of goal vector | `goal_vector()` |
| `instantaneous_goal_alignment(positions, times, goal, *, min_speed)` | cos(angle between v and goal_vector) | `heading_from_velocity()` |
| `goal_bias(positions, times, goal, *, min_speed)` | Mean alignment over trajectory | `instantaneous_goal_alignment()` |
| `approach_rate(positions, times, goal, *, metric, env)` | d/dt of distance_to_goal | `distance_to_region()` |
| `compute_goal_directed_metrics(env, positions, times, goal, ...)` | All metrics combined | All above |

#### Implementation Notes

- `goal_vector()`: Pure vectorized: `goal[np.newaxis, :] - positions`
- `instantaneous_goal_alignment()`:
  1. Compute velocity direction via `heading_from_velocity()`
  2. Compute goal direction via `goal_direction()`
  3. Return `cos(velocity_heading - goal_direction)`
- `approach_rate()`: Use `np.gradient()` on distance time series

---

### Module 3: Spatial Decision Analysis

**File**: `src/neurospatial/metrics/decision_analysis.py`

#### Data Structures

```python
@dataclass(frozen=True)
class PreDecisionMetrics:
    """Metrics from the pre-decision window.

    Attributes
    ----------
    mean_speed : float
        Mean speed in pre-decision window.
    min_speed : float
        Minimum speed (pause detection).
    heading_mean_direction : float
        Circular mean of heading, radians.
    heading_circular_variance : float
        Circular variance of heading, range [0, 1].
    heading_mean_resultant_length : float
        Concentration of heading distribution, range [0, 1].
    window_duration : float
        Actual duration of pre-decision window (seconds).
    n_samples : int
        Number of samples in window.
    """
```

```python
@dataclass(frozen=True)
class DecisionBoundaryMetrics:
    """Metrics related to decision boundaries between goals.

    Attributes
    ----------
    goal_labels : NDArray[np.int_]
        Per-timepoint label of nearest goal (Voronoi region).
    distance_to_boundary : NDArray[np.float64]
        Distance to nearest decision boundary at each timepoint.
    crossing_times : list[float]
        Times when trajectory crossed a decision boundary.
    crossing_directions : list[tuple[int, int]]
        (from_goal, to_goal) for each crossing.
    """
```

```python
@dataclass(frozen=True)
class DecisionAnalysisResult:
    """Complete decision analysis for a trial.

    Attributes
    ----------
    entry_time : float
        Time of decision region entry.
    pre_decision : PreDecisionMetrics
        Metrics from pre-decision window.
    boundary : DecisionBoundaryMetrics or None
        Boundary metrics (None if single goal).
    chosen_goal : int or None
        Index of goal reached (None if timeout).
    """
```

#### Functions

| Function | Description | Reuses |
|----------|-------------|--------|
| `decision_region_entry_time(trajectory_bins, times, env, region)` | First entry time to region | `segment_trials()` logic |
| `extract_pre_decision_window(positions, times, entry_time, window_duration)` | Slice trajectory before entry | Array slicing |
| `pre_decision_heading_stats(positions, times, *, min_speed)` | Circular stats on heading | `rayleigh_test()`, `_mean_resultant_length()` |
| `pre_decision_speed_stats(positions, times)` | Mean, min speed | `np.linalg.norm()` |
| `compute_pre_decision_metrics(positions, times, entry_time, window_duration, ...)` | Combined pre-decision | All above |
| `geodesic_voronoi_labels(env, goal_bins)` | Label bins by nearest goal | `distance_field()` per goal |
| `distance_to_decision_boundary(env, trajectory_bins, goal_bins)` | Distance to Voronoi edges | `geodesic_voronoi_labels()` |
| `detect_boundary_crossings(trajectory_bins, voronoi_labels, times)` | Crossing times and directions | Label change detection |
| `compute_decision_analysis(env, positions, times, decision_region, goals, ...)` | Complete analysis | All above |

#### Implementation Notes

- `geodesic_voronoi_labels()`:
  1. Compute `distance_field()` for each goal
  2. Stack into (n_goals, n_bins) array
  3. Return `argmin(axis=0)` for labels
- `distance_to_decision_boundary()`:
  1. For each bin, find distances to all goals
  2. Sort distances, boundary distance = (d_second - d_first) / 2
- Circular statistics reuse internal `_mean_resultant_length()` from `metrics/circular.py`

---

### Module 4: VTE Metrics

**File**: `src/neurospatial/metrics/vte.py`

#### Data Structures

```python
@dataclass(frozen=True)
class VTETrialResult:
    """VTE metrics for a single trial.

    Attributes
    ----------
    integrated_absolute_rotation : float
        Sum of |delta_theta| in pre-decision window (IdPhi).
    z_idphi : float or None
        Z-scored IdPhi (None if session stats not provided).
    mean_speed : float
        Mean speed in pre-decision window.
    min_speed : float
        Minimum speed in pre-decision window.
    z_speed_inv : float or None
        Z-scored inverted speed (None if session stats not provided).
    vte_index : float or None
        Combined VTE index (None if z-scores not available).
    is_vte : bool or None
        Classification result (None if threshold not applied).
    window_start : float
        Start time of analysis window.
    window_end : float
        End time of analysis window (decision region entry).
    """
```

```python
@dataclass(frozen=True)
class VTESessionResult:
    """VTE analysis for an entire session.

    Attributes
    ----------
    trial_results : list[VTETrialResult]
        Per-trial VTE metrics.
    mean_idphi : float
        Session mean of IdPhi (for z-scoring).
    std_idphi : float
        Session std of IdPhi.
    mean_speed : float
        Session mean of pre-decision speed.
    std_speed : float
        Session std of pre-decision speed.
    n_vte_trials : int
        Number of trials classified as VTE.
    vte_fraction : float
        Fraction of trials classified as VTE.
    """
```

#### Functions

| Function | Description | Reuses |
|----------|-------------|--------|
| `wrap_angle(angle)` | Wrap to (-pi, pi] | Vectorized modulo |
| `integrated_absolute_rotation(headings)` | Sum of \|delta_theta\| | `wrap_angle()` |
| `integrated_absolute_rotation_from_positions(positions, times, *, min_speed)` | IdPhi from trajectory | `heading_from_velocity()` |
| `normalize_vte_scores(idphi_values, speed_values)` | Z-score across trials | Vectorized stats |
| `behavioral_vte_index(z_idphi, z_speed_inv, *, alpha)` | alpha * z_idphi + (1-alpha) * z_speed_inv | Weighted sum |
| `classify_vte_trial(vte_index, *, threshold)` | VTE if index > threshold | Comparison |
| `compute_vte_trial(positions, times, entry_time, window_duration, ...)` | Single trial VTE | Multiple above |
| `compute_vte_session(env, positions, times, trials, decision_region, ...)` | Full session analysis | `compute_vte_trial()` per trial |

#### Implementation Notes

- `integrated_absolute_rotation()`:
  ```python
  delta = wrap_angle(np.diff(headings))
  return np.sum(np.abs(delta))
  ```
- Two-pass algorithm for session:
  1. First pass: compute raw IdPhi and speed for all trials
  2. Compute session statistics (mean, std)
  3. Second pass: compute z-scores and VTE indices
- Default `alpha=0.5` weights head sweeps and speed equally

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
    └── uses: compute_step_lengths(), geodesic_distance_matrix(), compute_turn_angles()

goal_directed.py
    └── uses: distance_to_region(), heading_from_velocity()

decision_analysis.py
    ├── uses: segment_trials(), distance_field(), rayleigh_test()
    └── uses: _mean_resultant_length() (internal from circular.py)

vte.py
    ├── uses: heading_from_velocity()
    └── uses: decision_analysis.py functions
```

---

## Implementation Order

### Phase 1: Path Efficiency (Foundation)

- [ ] Create `metrics/path_efficiency.py`
- [ ] Implement `PathEfficiencyResult` dataclass
- [ ] Implement `actual_path_length()` wrapping `compute_step_lengths()`
- [ ] Implement `optimal_path_length()` using distance functions
- [ ] Implement `path_efficiency()`
- [ ] Implement `time_normalized_efficiency()`
- [ ] Implement `angular_path_efficiency()`
- [ ] Implement `SubgoalEfficiencyResult` and `subgoal_path_efficiency()`
- [ ] Implement `compute_path_efficiency()` combining all metrics
- [ ] Write tests in `tests/metrics/test_path_efficiency.py`
- [ ] Add exports to `metrics/__init__.py`

### Phase 2: Goal-Directed Metrics

- [ ] Create `metrics/goal_directed.py`
- [ ] Implement `GoalDirectedMetrics` dataclass
- [ ] Implement `goal_vector()` and `goal_direction()`
- [ ] Implement `instantaneous_goal_alignment()`
- [ ] Implement `goal_bias()`
- [ ] Implement `approach_rate()`
- [ ] Implement `compute_goal_directed_metrics()`
- [ ] Write tests in `tests/metrics/test_goal_directed.py`
- [ ] Add exports to `metrics/__init__.py`

### Phase 3: Decision Analysis

- [ ] Create `metrics/decision_analysis.py`
- [ ] Implement `PreDecisionMetrics` dataclass
- [ ] Implement `decision_region_entry_time()`
- [ ] Implement `extract_pre_decision_window()`
- [ ] Implement `pre_decision_heading_stats()` using circular functions
- [ ] Implement `pre_decision_speed_stats()`
- [ ] Implement `compute_pre_decision_metrics()`
- [ ] Implement `DecisionBoundaryMetrics` dataclass
- [ ] Implement `geodesic_voronoi_labels()`
- [ ] Implement `distance_to_decision_boundary()`
- [ ] Implement `detect_boundary_crossings()`
- [ ] Implement `DecisionAnalysisResult` and `compute_decision_analysis()`
- [ ] Write tests in `tests/metrics/test_decision_analysis.py`
- [ ] Add exports to `metrics/__init__.py`

### Phase 4: VTE Metrics

- [ ] Create `metrics/vte.py`
- [ ] Implement `wrap_angle()` utility
- [ ] Implement `integrated_absolute_rotation()`
- [ ] Implement `integrated_absolute_rotation_from_positions()`
- [ ] Implement `VTETrialResult` dataclass
- [ ] Implement `normalize_vte_scores()`
- [ ] Implement `behavioral_vte_index()`
- [ ] Implement `classify_vte_trial()`
- [ ] Implement `compute_vte_trial()`
- [ ] Implement `VTESessionResult` dataclass
- [ ] Implement `compute_vte_session()`
- [ ] Write tests in `tests/metrics/test_vte.py`
- [ ] Add exports to `metrics/__init__.py`

### Phase 5: Integration

- [ ] Update `.claude/QUICKSTART.md` with behavioral metrics examples
- [ ] Update `.claude/API_REFERENCE.md` with new imports
- [ ] Run full test suite: `uv run pytest tests/metrics/`
- [ ] Run type checker: `uv run mypy src/neurospatial/metrics/`
- [ ] Run linter: `uv run ruff check src/neurospatial/metrics/`

---

## Estimated Effort

| Module | New LOC | Reused Functions | Test LOC |
|--------|---------|------------------|----------|
| `path_efficiency.py` | ~250 | 4 | ~200 |
| `goal_directed.py` | ~300 | 5 | ~250 |
| `decision_analysis.py` | ~400 | 6 | ~350 |
| `vte.py` | ~300 | 4 | ~300 |
| **Total** | **~1,250** | **19** | **~1,100** |

---

## Testing Strategy

### Unit Tests

Each function tested independently with:
- Basic functionality with known inputs/outputs
- Edge cases (empty arrays, single points, NaN handling)
- Parameter validation (error messages)

### Integration Tests

- Round-trip consistency: simulated trajectory → metrics → verify expected values
- Cross-module consistency: VTE uses decision_analysis correctly
- Comparison with existing functions: path_efficiency consistent with path_progress

### Regression Tests

- Simulated cells with known ground truth parameters
- Verify metric recovery within tolerance

---

## API Design Principles

1. **Consistent Signatures**: All main functions take `(env, positions, times, ...)` as first args
2. **Keyword-Only Options**: All optional parameters are keyword-only after positional
3. **Metric Parameter**: Functions supporting both geodesic/Euclidean have `metric=` parameter
4. **Return Dataclasses**: Complex results return frozen dataclasses, not tuples
5. **NumPy Docstrings**: All functions have complete NumPy-format docstrings
6. **Vectorized**: No loops over timepoints; all operations vectorized

---

## References

### Mathematical Framework

- Path efficiency: Batschelet (1981), ecology MCP literature
- Goal-directed: Johnson & Redish (2007), Papale et al. (2012)
- VTE (IdPhi): Redish (2016), Muenzinger (1938)
- Circular statistics: Mardia & Jupp (2000)

### Existing Code

- `neurospatial.metrics.trajectory`: Step lengths, turn angles, MSD
- `neurospatial.behavioral`: Path progress, distance to region
- `neurospatial.distance`: Geodesic distance functions
- `neurospatial.metrics.circular`: Circular statistics
- `neurospatial.reference_frames`: Heading computation
