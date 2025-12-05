# Behavioral Trajectory Metrics: Implementation Tasks

**Source**: [BEHAV_PLAN.md](BEHAV_PLAN.md)
**Created**: 2025-12-05
**Status**: Ready for implementation

---

## Overview

Implementation of behavioral trajectory analysis metrics for spatial navigation research.

**Modules to implement**:

1. Path Efficiency (`path_efficiency.py`)
2. Goal-Directed Metrics (`goal_directed.py`)
3. Decision Analysis (`decision_analysis.py`)
4. VTE Metrics (`vte.py`)

**Estimated LOC**: ~1,700 new + ~1,300 tests

---

## Dependencies

```
path_efficiency.py (no internal dependencies)
    │
    └──► goal_directed.py (no dependencies on path_efficiency)
              │
              └──► decision_analysis.py (no dependencies on goal_directed)
                        │
                        └──► vte.py (depends on decision_analysis)
```

**External dependencies** (already exist):

- `neurospatial.metrics.trajectory`: `compute_step_lengths()`, `compute_turn_angles()`
- `neurospatial.distance`: `distance_field()`, `geodesic_distance_matrix()`
- `neurospatial.behavioral`: `distance_to_region()`, `path_progress()`
- `neurospatial.reference_frames`: `heading_from_velocity()`
- `neurospatial.segmentation.trials`: `segment_trials()`, `Trial`

---

## Milestone 1: Path Efficiency

**Goal**: Compute how efficiently an animal navigates from start to goal.

**File**: `src/neurospatial/metrics/path_efficiency.py`

**Priority**: HIGH (foundation for other modules)

### M1.1: Module Setup

- [x] Create `path_efficiency.py` with module docstring
- [x] Add imports: numpy, typing, dataclasses
- [x] Add internal imports from `neurospatial.metrics.trajectory`, `neurospatial.distance`

### M1.2: Data Structures

- [x] Implement `PathEfficiencyResult` frozen dataclass
  - [x] Fields: `traveled_length`, `shortest_length`, `efficiency`, `time_efficiency`, `angular_efficiency`, `start_position`, `goal_position`, `metric`
  - [x] Method: `is_efficient(threshold=0.8) -> bool`
  - [x] Method: `summary() -> str`
- [x] Implement `SubgoalEfficiencyResult` frozen dataclass
  - [x] Fields: `segment_results`, `mean_efficiency`, `weighted_efficiency`, `subgoal_positions`
  - [x] Method: `summary() -> str`

### M1.3: Core Functions

- [x] Implement `traveled_path_length(positions, *, metric="euclidean", env=None) -> float`
  - [x] Wrap `compute_step_lengths()` with `metric` -> `distance_type` mapping
  - [x] Handle < 2 positions (return 0.0)
  - [x] Validate env required for geodesic
- [x] Implement `shortest_path_length(env, start, goal, *, metric="geodesic") -> float`
  - [x] Use `distance_field()` for geodesic
  - [x] Use euclidean distance for euclidean metric
  - [x] Handle disconnected graph (return inf)
- [x] Implement `path_efficiency(env, positions, goal, *, metric="geodesic") -> float`
  - [x] Compute ratio: `shortest_length / traveled_length`
  - [x] Return NaN for < 2 positions or traveled_length == 0
- [x] Implement `time_efficiency(positions, times, goal, *, reference_speed) -> float`
  - [x] Compute `T_optimal / T_actual`
- [x] Implement `angular_efficiency(positions, goal) -> float`
  - [x] Use `compute_turn_angles()` for heading changes
  - [x] Return 1.0 for < 3 positions
  - [x] Return NaN if all positions identical (`np.ptp() < 1e-10`)
  - [x] Formula: `1 - mean(|delta_theta|) / pi`

### M1.4: Composite Functions

- [x] Implement `subgoal_efficiency(env, positions, subgoals, *, metric="geodesic") -> SubgoalEfficiencyResult`
  - [x] Segment trajectory by subgoal arrivals
  - [x] Compute per-segment efficiency
  - [x] Compute weighted mean by segment length
- [x] Implement `compute_path_efficiency(env, positions, times, goal, ...) -> PathEfficiencyResult`
  - [x] Combine all metrics into single result
  - [x] Optional `reference_speed` for time efficiency

### M1.5: Error Handling

- [x] Add helpful error messages:
  - [x] Mismatched array lengths
  - [x] Goal outside environment bounds
  - [x] Empty trajectory
  - [x] Disconnected graph (infinite distance)

### M1.6: Tests

**File**: `tests/metrics/test_path_efficiency.py`

- [x] Test `traveled_path_length()` with euclidean/geodesic
- [x] Test `shortest_path_length()` euclidean vs geodesic
- [x] Test `path_efficiency()` returns 1.0 for straight path
- [x] Test `path_efficiency()` returns < 0.5 for random walk
- [x] Test `angular_efficiency()` edge cases (< 3 positions, identical positions)
- [x] Test `subgoal_efficiency()` with T-maze trajectory
- [x] Test error messages are helpful

### M1.7: Exports

- [x] Add to `src/neurospatial/metrics/__init__.py`:
  - [x] `PathEfficiencyResult`
  - [x] `SubgoalEfficiencyResult`
  - [x] `traveled_path_length`
  - [x] `shortest_path_length`
  - [x] `path_efficiency`
  - [x] `time_efficiency`
  - [x] `angular_efficiency`
  - [x] `subgoal_efficiency`
  - [x] `compute_path_efficiency`

**Success criteria**:

- [x] Straight-line path to goal returns efficiency = 1.0
- [x] U-turn path returns efficiency ~ 0.5
- [x] All tests pass: `uv run pytest tests/metrics/test_path_efficiency.py -v`

---

## Milestone 2: Goal-Directed Metrics

**Goal**: Measure how directly an animal navigates toward a goal.

**File**: `src/neurospatial/metrics/goal_directed.py`

**Priority**: HIGH

### M2.1: Module Setup

- [x] Create `goal_directed.py` with module docstring
- [x] Add imports from `neurospatial.reference_frames`, `neurospatial.behavioral`

### M2.2: Data Structures

- [x] Implement `GoalDirectedMetrics` frozen dataclass
  - [x] Fields: `goal_bias`, `mean_approach_rate`, `time_to_goal`, `min_distance_to_goal`, `goal_distance_at_start`, `goal_distance_at_end`, `goal_position`, `metric`
  - [x] Method: `is_goal_directed(threshold=0.3) -> bool`
  - [x] Method: `summary() -> str`

### M2.3: Core Functions

- [x] Implement `goal_vector(positions, goal) -> NDArray`
  - [x] Compute `goal - positions` for each timepoint
  - [x] Validate dimension match
- [x] Implement `goal_direction(positions, goal) -> NDArray`
  - [x] Compute `atan2` of goal vector
- [x] Implement `instantaneous_goal_alignment(positions, times, goal, *, min_speed=5.0) -> NDArray`
  - [x] Compute `dt = median(diff(times))` for `heading_from_velocity()`
  - [x] Compute cosine of angle between velocity and goal direction
  - [x] NaN for stationary periods (handled by `heading_from_velocity`)
- [x] Implement `goal_bias(positions, times, goal, *, min_speed=5.0) -> float`
  - [x] Mean of `instantaneous_goal_alignment`, ignoring NaN
- [x] Implement `approach_rate(positions, times, goal, *, metric="geodesic", env=None) -> NDArray`
  - [x] Compute `d/dt` of distance to goal
  - [x] Negative = approaching, positive = retreating

### M2.4: Composite Function

- [x] Implement `compute_goal_directed_metrics(env, positions, times, goal, ...) -> GoalDirectedMetrics`
  - [x] Combine all metrics
  - [x] Compute `time_to_goal` if goal region reached

### M2.5: Tests

**File**: `tests/metrics/test_goal_directed.py`

- [x] Test `goal_vector()` dimension validation
- [x] Test `instantaneous_goal_alignment()` for direct approach (should be ~1.0)
- [x] Test `goal_bias()` for moving away (should be negative)
- [x] Test `goal_bias()` for circular path around goal (should be ~0)
- [x] Test `approach_rate()` sign convention
- [x] Test edge cases (stationary animal, single position)

### M2.6: Exports

- [x] Add to `src/neurospatial/metrics/__init__.py`:
  - [x] `GoalDirectedMetrics`
  - [x] `goal_vector`
  - [x] `goal_direction`
  - [x] `instantaneous_goal_alignment`
  - [x] `goal_bias`
  - [x] `approach_rate`
  - [x] `compute_goal_directed_metrics`

**Success criteria**:

- [x] Direct approach to goal: `goal_bias > 0.8`
- [x] Moving away from goal: `goal_bias < -0.5`
- [x] All tests pass: `uv run pytest tests/metrics/test_goal_directed.py -v`

---

## Milestone 3: Decision Analysis

**Goal**: Analyze behavior at decision points (T-junctions, Y-mazes).

**File**: `src/neurospatial/metrics/decision_analysis.py`

**Priority**: MEDIUM (depends on VTE but can be developed in parallel)

### M3.1: Module Setup

- [x] Create `decision_analysis.py` with module docstring
- [x] Add imports from `neurospatial.distance`, `neurospatial.reference_frames`

### M3.2: Data Structures

- [x] Implement `PreDecisionMetrics` frozen dataclass
  - [x] Fields: `mean_speed`, `min_speed`, `heading_mean_direction`, `heading_circular_variance`, `heading_mean_resultant_length`, `window_duration`, `n_samples`
  - [x] Method: `suggests_deliberation(variance_threshold=0.5, speed_threshold=10.0) -> bool`
- [x] Implement `DecisionBoundaryMetrics` frozen dataclass
  - [x] Fields: `goal_labels`, `distance_to_boundary`, `crossing_times`, `crossing_directions`
  - [x] Property: `n_crossings`
  - [x] Method: `summary() -> str`
- [x] Implement `DecisionAnalysisResult` frozen dataclass
  - [x] Fields: `entry_time`, `pre_decision`, `boundary`, `chosen_goal`
  - [x] Method: `summary() -> str`

### M3.3: Pre-Decision Window Functions

- [x] Implement `decision_region_entry_time(trajectory_bins, times, env, region) -> float`
  - [x] Find first entry to region
  - [x] Raise ValueError if never enters
- [x] Implement `extract_pre_decision_window(positions, times, entry_time, window_duration) -> tuple[NDArray, NDArray]`
  - [x] Slice trajectory before entry
  - [x] Handle case where window starts before trajectory
- [x] Implement `pre_decision_heading_stats(positions, times, *, min_speed=5.0) -> tuple[float, float, float]`
  - [x] Compute `dt = median(diff(times))` for `heading_from_velocity()`
  - [x] Compute circular mean, variance, mean resultant length directly
  - [x] Do NOT use private `_mean_resultant_length()`
- [x] Implement `pre_decision_speed_stats(positions, times) -> tuple[float, float]`
  - [x] Return mean and min speed
- [x] Implement `compute_pre_decision_metrics(...) -> PreDecisionMetrics`
  - [x] Combine window extraction and stats

### M3.4: Decision Boundary Functions

- [x] Implement `geodesic_voronoi_labels(env, goal_bins) -> NDArray[np.int_]`
  - [x] Compute distance field from each goal
  - [x] Label each bin by nearest goal
  - [x] Mark unreachable bins as -1
  - [x] Add performance note: O(n_goals *n_bins* log(n_bins))
- [x] Implement `distance_to_decision_boundary(env, trajectory_bins, goal_bins) -> NDArray[np.float64]`
  - [x] For each position, compute distance to nearest boundary
- [x] Implement `detect_boundary_crossings(trajectory_bins, voronoi_labels, times) -> tuple[list, list]`
  - [x] Find times and directions of label changes

### M3.5: Composite Function

- [x] Implement `compute_decision_analysis(env, positions, times, decision_region, goal_regions, ...) -> DecisionAnalysisResult`
  - [x] Combine pre-decision metrics and boundary metrics
  - [x] Determine chosen goal

### M3.6: Tests

**File**: `tests/metrics/test_decision_analysis.py`

- [x] Test `decision_region_entry_time()` finds correct entry
- [x] Test `extract_pre_decision_window()` slice correctness
- [x] Test `pre_decision_heading_stats()` circular variance computation
- [x] Test `geodesic_voronoi_labels()` with T-maze (2 goals)
- [x] Test `detect_boundary_crossings()` counts correctly
- [x] Test `suggests_deliberation()` method

### M3.7: Exports

- [x] Add to `src/neurospatial/metrics/__init__.py`:
  - [x] `PreDecisionMetrics`
  - [x] `DecisionBoundaryMetrics`
  - [x] `DecisionAnalysisResult`
  - [x] `decision_region_entry_time`
  - [x] `extract_pre_decision_window`
  - [x] `pre_decision_heading_stats`
  - [x] `pre_decision_speed_stats`
  - [x] `compute_pre_decision_metrics`
  - [x] `geodesic_voronoi_labels`
  - [x] `distance_to_decision_boundary`
  - [x] `detect_boundary_crossings`
  - [x] `compute_decision_analysis`

**Success criteria**:

- [x] High heading variance + low speed → `suggests_deliberation() == True`
- [x] T-maze labels bins correctly to left/right goals
- [x] All tests pass: `uv run pytest tests/metrics/test_decision_analysis.py -v`

---

## Milestone 4: VTE Metrics

**Goal**: Detect and quantify Vicarious Trial and Error (VTE) behavior.

**File**: `src/neurospatial/metrics/vte.py`

**Priority**: MEDIUM (depends on M3 decision_analysis)

**Dependency**: Requires `decision_region_entry_time()` and `extract_pre_decision_window()` from M3

### M4.1: Module Setup

- [x] Create `vte.py` with module docstring
- [x] Add terminology glossary (IdPhi, zIdPhi, VTE index)
- [x] Add imports from `neurospatial.reference_frames`, `neurospatial.metrics.decision_analysis`

### M4.2: Data Structures

- [x] Implement `VTETrialResult` frozen dataclass
  - [x] Fields: `head_sweep_magnitude`, `z_head_sweep`, `mean_speed`, `min_speed`, `z_speed_inverse`, `vte_index`, `is_vte`, `window_start`, `window_end`
  - [x] Property aliases: `idphi`, `z_idphi`
  - [x] Method: `summary() -> str`
- [x] Implement `VTESessionResult` frozen dataclass
  - [x] Fields: `trial_results`, `mean_head_sweep`, `std_head_sweep`, `mean_speed`, `std_speed`, `n_vte_trials`, `vte_fraction`
  - [x] Property aliases: `mean_idphi`, `std_idphi`
  - [x] Method: `summary() -> str`
  - [x] Method: `get_vte_trials() -> list[VTETrialResult]`

### M4.3: Core Functions

- [x] Implement `wrap_angle(angle) -> NDArray[np.float64]`
  - [x] Wrap to (-pi, pi]
  - [x] Formula: `(angle + pi) % (2 * pi) - pi`
- [x] Implement `head_sweep_magnitude(headings) -> float`
  - [x] Sum of `|delta_theta|` with angle wrapping
  - [x] Return 0.0 for < 2 valid samples
  - [x] Filter NaN values
- [x] Add `integrated_absolute_rotation = head_sweep_magnitude` alias
- [x] Implement `head_sweep_from_positions(positions, times, *, min_speed=5.0) -> float`
  - [x] Compute `dt = median(diff(times))` for `heading_from_velocity()`
  - [x] Get headings and compute magnitude

### M4.4: Z-Scoring Functions

- [x] Implement `normalize_vte_scores(head_sweeps, speeds) -> tuple[NDArray, NDArray]`
  - [x] Z-score head sweeps
  - [x] Z-score inverse speed (higher = slower = more VTE-like)
  - [x] Warn if std=0 with helpful message about adjusting parameters
  - [x] Return zeros (not NaN) when std=0

### M4.5: Classification Functions

- [x] Implement `compute_vte_index(z_head_sweep, z_speed_inv, *, alpha=0.5) -> float`
  - [x] Formula: `alpha * z_head_sweep + (1 - alpha) * z_speed_inverse`
- [x] Implement `classify_vte(vte_index, *, threshold=0.5) -> bool`
  - [x] Return `vte_index > threshold`

### M4.6: Composite Functions

- [x] Implement `compute_vte_trial(positions, times, entry_time, window_duration, ...) -> VTETrialResult`
  - [x] Single trial analysis (z-scores are None)
- [x] Implement `compute_vte_session(positions, times, trials, decision_region, env, ...) -> VTESessionResult`
  - [x] Loop over trials
  - [x] Compute session statistics
  - [x] Z-score and classify each trial

### M4.7: Tests

**File**: `tests/metrics/test_vte.py`

- [x] Test `wrap_angle()` wraps correctly around boundaries
- [x] Test `head_sweep_magnitude()` for stationary vs scanning trajectory
- [x] Test `head_sweep_from_positions()` with known trajectory
- [x] Test `normalize_vte_scores()` z-scoring correctness
- [x] Test `normalize_vte_scores()` warning when std=0
- [x] Test `compute_vte_index()` weighting
- [x] Test `compute_vte_session()` classifies correctly
- [x] Test high head sweep + low speed → VTE
- [x] Test low head sweep + high speed → non-VTE

### M4.8: Exports

- [x] Add to `src/neurospatial/metrics/__init__.py`:
  - [x] `VTETrialResult`
  - [x] `VTESessionResult`
  - [x] `wrap_angle`
  - [x] `head_sweep_magnitude`
  - [x] `integrated_absolute_rotation`
  - [x] `head_sweep_from_positions`
  - [x] `normalize_vte_scores`
  - [x] `compute_vte_index`
  - [x] `classify_vte`
  - [x] `compute_vte_trial`
  - [x] `compute_vte_session`

**Success criteria**:

- [x] High head sweep + low speed → classified as VTE
- [x] Low head sweep + high speed → not classified as VTE
- [x] std=0 triggers warning, not error
- [x] All tests pass: `uv run pytest tests/metrics/test_vte.py -v`

---

## Milestone 5: Integration and Documentation

**Goal**: Integrate all modules and update documentation.

**Priority**: LOW (after M1-M4 complete)

### M5.1: Full Test Suite

- [x] Run all tests: `uv run pytest tests/metrics/ -v`
- [x] Run type checker: `uv run mypy src/neurospatial/metrics/`
- [x] Run linter: `uv run ruff check src/neurospatial/metrics/`
- [x] Fix any issues

### M5.2: Integration Tests

- [x] Test cross-module consistency (VTE uses decision_analysis functions correctly)
- [x] Test round-trip: simulated VTE trial → compute_vte_session → correct classification
- [x] Test path_efficiency consistent with existing path_progress

### M5.3: Documentation Updates

- [x] Update `.claude/QUICKSTART.md` with behavioral metrics examples
  - [x] Path efficiency example
  - [x] Goal-directed metrics example
  - [x] VTE detection example
- [x] Update `.claude/API_REFERENCE.md` with new imports
  - [x] List all new functions and dataclasses
  - [x] Group by module

### M5.4: Final Validation

- [x] All tests pass
- [x] No type errors
- [x] No linting errors
- [x] Examples in docstrings work

**Success criteria**:

- [x] `uv run pytest tests/metrics/ -v` all pass
- [x] `uv run mypy src/neurospatial/metrics/` no errors
- [x] `uv run ruff check src/neurospatial/metrics/` no errors

---

## Implementation Notes

### Critical Design Decisions

1. **Parameter naming**: Use `metric` (not `distance_type`) for consistency with `behavioral.py`

2. **`heading_from_velocity()` signature**: Takes `dt: float`, not `times` array

   ```python
   dt = float(np.median(np.diff(times)))
   headings = heading_from_velocity(positions, dt, min_speed=min_speed)
   ```

3. **Circular statistics**: Compute directly, not via private functions

   ```python
   cos_headings = np.cos(headings)
   sin_headings = np.sin(headings)
   mean_resultant_length = np.sqrt(np.mean(cos_headings)**2 + np.mean(sin_headings)**2)
   circular_variance = 1.0 - mean_resultant_length
   ```

4. **Edge case handling**:
   - `< 2 positions`: return 0.0 for path length, NaN for efficiency
   - `< 3 positions`: return 1.0 for angular efficiency
   - All identical positions: use `np.ptp(positions, axis=0).max() < 1e-10`
   - std=0 in z-scoring: warn and return zeros

### Import Patterns

```python
# Correct internal imports
from neurospatial.distance import distance_field, geodesic_distance_matrix
from neurospatial.reference_frames import heading_from_velocity
from neurospatial.metrics.trajectory import compute_step_lengths, compute_turn_angles
from neurospatial.metrics.circular import rayleigh_test
from neurospatial.behavioral import distance_to_region
```

### Error Message Template

```python
raise ValueError(
    f"<What's wrong>. "
    f"Got <actual values>. "
    f"\nTo fix:\n"
    f"- <Suggestion 1>\n"
    f"- <Suggestion 2>"
)
```

---

## Execution Order

| Order | Milestone | Dependencies | Est. Time |
|-------|-----------|--------------|-----------|
| 1 | M1: Path Efficiency | None | First |
| 2 | M2: Goal-Directed | None (parallel with M1) | Second |
| 3 | M3: Decision Analysis | None (parallel with M1/M2) | Third |
| 4 | M4: VTE Metrics | M3.3 (pre-decision functions) | Fourth |
| 5 | M5: Integration | M1, M2, M3, M4 | Last |

**Recommended approach**: Implement M1, M2, M3 in parallel, then M4, then M5.

---

## References

- **BEHAV_PLAN.md**: Full implementation details, dataclass definitions, function signatures
- **Mathematical framework**: Batschelet (1981), Johnson & Redish (2007), Redish (2016), Papale et al. (2012)
- **Existing code**: `neurospatial.behavioral`, `neurospatial.metrics.trajectory`, `neurospatial.distance`
