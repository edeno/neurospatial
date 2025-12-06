# SCRATCHPAD - Package Reorganization

**Started**: 2025-12-05
**Current Status**: Milestone 6 COMPLETE - All tasks 6.1-6.9 done

---

## Session Log

### 2025-12-06 (Session 30)

**Starting Point**: Milestone 6 - Move encoding/ Module (Task 6.9: Update Internal Imports)

**Completed**: Update internal imports to use encoding modules

**Work Done**:
1. Updated `src/neurospatial/__init__.py` to import from encoding modules:
   - `neurospatial.encoding` for metrics (detect_place_fields, skaggs_information, border_score, etc.)
   - `neurospatial.encoding.object_vector` for ObjectVectorFieldResult, compute_object_vector_field
   - `neurospatial.encoding.spatial_view` for SpatialViewFieldResult, compute_spatial_view_field
   - `neurospatial.encoding.place` for DirectionalPlaceFields, compute_place_field, etc.
2. Avoided circular imports by keeping `metrics/__init__.py` importing from source modules (not encoding)
3. Ran ruff check (2 auto-fixes for import sorting) and mypy - no issues
4. All encoding tests pass (216 passed)

**Architecture Note**:
The import flow is now:
- Source modules (`spike_field.py`, `metrics/place_fields.py`, etc.) contain implementations
- `encoding/` modules re-export from source modules
- `metrics/__init__.py` imports from source modules (to avoid circular imports)
- `neurospatial/__init__.py` imports from encoding modules (new canonical paths)

**Files Modified**:
- `src/neurospatial/__init__.py` (updated imports to use encoding modules)

**Milestone 6 Status**: COMPLETE
All tasks 6.1-6.9 are done.

**Next Task**: Milestone 7 - Reorganize decoding/ Module

---

### 2025-12-06 (Session 29)

**Starting Point**: Milestone 6 - Move encoding/ Module (Task 6.8: Create encoding/population.py)

**Completed**: Create encoding/population.py with re-exports from metrics/population.py

**Work Done**:
1. Created test file `tests/encoding/test_encoding_population.py` following TDD (RED phase)
   - 33 tests total: 7 import tests from encoding.population, 7 import tests from encoding, 3 module structure tests, 7 re-export identity tests, 9 functionality tests
   - Tests for imports of all 7 symbols: `PopulationCoverageResult`, `population_coverage`, `plot_population_coverage`, `field_density_map`, `count_place_cells`, `field_overlap`, `population_vector_correlation`
2. Verified tests FAIL before implementation (import error expected)
3. Created `encoding/population.py` as a re-export module:
   - From metrics/population.py: `PopulationCoverageResult`, `population_coverage`, `plot_population_coverage`, `field_density_map`, `count_place_cells`, `field_overlap`, `population_vector_correlation`
4. Updated `encoding/__init__.py` to export all 7 population symbols
5. All tests pass: 33 passed (new) + 54 passed (existing metrics/test_population.py)
6. Ran ruff check/format and mypy - no issues
7. Code review APPROVED

**Files Created**:
- `src/neurospatial/encoding/population.py` (new - re-exports from metrics.population)
- `tests/encoding/test_encoding_population.py` (new)

**Files Modified**:
- `src/neurospatial/encoding/__init__.py` (added 7 population exports)

**Milestone 6 Status**: Tasks 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8 COMPLETE
Remaining: Task 6.9

**Next Task**: Milestone 6, Task 6.9 - Update Internal Imports

---

### 2025-12-06 (Session 28)

**Starting Point**: Milestone 6 - Move encoding/ Module (Task 6.7: Create encoding/phase_precession.py)

**Completed**: Create encoding/phase_precession.py with re-exports from metrics/phase_precession.py

**Work Done**:
1. Created test file `tests/encoding/test_encoding_phase_precession.py` following TDD (RED phase)
   - 22 tests total: 4 import tests from encoding.phase_precession, 4 import tests from encoding, 3 module structure tests (using importlib to avoid function shadowing), 4 re-export identity tests, 7 functionality tests
   - Tests for imports of all 4 symbols: `PhasePrecessionResult`, `phase_precession`, `has_phase_precession`, `plot_phase_precession`
2. Verified tests FAIL before implementation (import error expected)
3. Created `encoding/phase_precession.py` as a re-export module:
   - From metrics/phase_precession.py: `PhasePrecessionResult`, `phase_precession`, `has_phase_precession`, `plot_phase_precession`
4. Updated `encoding/__init__.py` to export all 4 phase_precession symbols
5. All tests pass: 22 passed (new) + 38 passed (existing phase_precession tests)
6. Ran ruff check/format and mypy - no issues
7. Code review APPROVED

**Files Created**:
- `src/neurospatial/encoding/phase_precession.py` (new - re-exports from metrics.phase_precession)
- `tests/encoding/test_encoding_phase_precession.py` (new)

**Files Modified**:
- `src/neurospatial/encoding/__init__.py` (added 4 phase_precession exports)

**Milestone 6 Status**: Tasks 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7 COMPLETE
Remaining: Tasks 6.8-6.9

**Next Task**: Milestone 6, Task 6.8 - Create encoding/population.py

---

### 2025-12-06 (Session 27)

**Starting Point**: Milestone 6 - Move encoding/ Module (Task 6.6: Create encoding/spatial_view.py)

**Completed**: Create encoding/spatial_view.py with re-exports from spatial_view_field.py, metrics/spatial_view_cells.py, and ops/visibility.py

**Work Done**:
1. Created test file `tests/encoding/test_encoding_spatial_view.py` following TDD (RED phase)
   - 37 tests total: 9 import tests from encoding.spatial_view, 9 import tests from encoding, 3 module structure tests, 9 re-export identity tests, 7 functionality tests
   - Tests for imports of all 9 symbols: `SpatialViewFieldResult`, `compute_spatial_view_field`, `SpatialViewMetrics`, `spatial_view_cell_metrics`, `is_spatial_view_cell`, `compute_viewed_location`, `compute_viewshed`, `visibility_occupancy`, `FieldOfView`
2. Verified tests FAIL before implementation (import error expected)
3. Created `encoding/spatial_view.py` as a re-export module:
   - From spatial_view_field.py: `SpatialViewFieldResult`, `compute_spatial_view_field`
   - From metrics/spatial_view_cells.py: `SpatialViewMetrics`, `spatial_view_cell_metrics`, `is_spatial_view_cell`
   - From ops/visibility.py (re-exports for convenience): `compute_viewed_location`, `compute_viewshed`, `visibility_occupancy`, `FieldOfView`
4. Updated `encoding/__init__.py` to export all 9 spatial_view symbols
5. All tests pass: 37 passed (new) + 53 passed (existing spatial_view tests)
6. Ran ruff check/format and mypy - no issues
7. Code review APPROVED

**Files Created**:
- `src/neurospatial/encoding/spatial_view.py` (new - re-exports from spatial_view_field.py, metrics.spatial_view_cells, and ops.visibility)
- `tests/encoding/test_encoding_spatial_view.py` (new)

**Files Modified**:
- `src/neurospatial/encoding/__init__.py` (added 9 spatial_view exports)

**Milestone 6 Status**: Tasks 6.1, 6.2, 6.3, 6.4, 6.5, 6.6 COMPLETE
Remaining: Tasks 6.7-6.9

**Next Task**: Milestone 6, Task 6.7 - Create encoding/phase_precession.py

---

### 2025-12-06 (Session 26)

**Starting Point**: Milestone 6 - Move encoding/ Module (Task 6.5: Create encoding/object_vector.py)

**Completed**: Create encoding/object_vector.py with re-exports from object_vector_field.py and metrics/object_vector_cells.py

**Work Done**:
1. Created test file `tests/encoding/test_encoding_object_vector.py` following TDD (RED phase)
   - 32 tests total: 7 import tests from encoding.object_vector, 7 import tests from encoding, 3 module structure tests, 7 re-export tests, 8 functionality tests
   - Tests for imports of all 7 symbols: `ObjectVectorFieldResult`, `compute_object_vector_field`, `ObjectVectorMetrics`, `compute_object_vector_tuning`, `object_vector_score`, `is_object_vector_cell`, `plot_object_vector_tuning`
2. Verified tests FAIL before implementation (import error expected)
3. Created `encoding/object_vector.py` as a re-export module:
   - From object_vector_field.py: `ObjectVectorFieldResult`, `compute_object_vector_field`
   - From metrics/object_vector_cells.py: `ObjectVectorMetrics`, `compute_object_vector_tuning`, `object_vector_score`, `is_object_vector_cell`, `plot_object_vector_tuning`
4. Updated `encoding/__init__.py` to export all 7 object_vector symbols
5. All tests pass: 32 passed (new) + 80 passed (existing object_vector tests)
6. Ran ruff check/format and mypy - no issues
7. Code review APPROVED

**Files Created**:
- `src/neurospatial/encoding/object_vector.py` (new - re-exports from object_vector_field.py and metrics.object_vector_cells)
- `tests/encoding/test_encoding_object_vector.py` (new)

**Files Modified**:
- `src/neurospatial/encoding/__init__.py` (added 7 object_vector exports)

**Milestone 6 Status**: Tasks 6.1, 6.2, 6.3, 6.4, 6.5 COMPLETE
Remaining: Tasks 6.6-6.9

**Next Task**: Milestone 6, Task 6.6 - Create encoding/spatial_view.py

---

### 2025-12-06 (Session 25)

**Starting Point**: Milestone 6 - Move encoding/ Module (Task 6.4: Create encoding/border.py)

**Completed**: Create encoding/border.py with re-exports from metrics/boundary_cells.py

**Work Done**:
1. Created test file `tests/encoding/test_encoding_border.py` following TDD (RED phase)
   - 15 tests total: 2 import tests from encoding.border, 2 import tests from encoding, 3 module structure tests, 2 re-export tests, 6 functionality tests
   - Tests for imports of all 2 symbols: `border_score`, `compute_region_coverage`
2. Verified tests FAIL before implementation (import error expected)
3. Created `encoding/border.py` as a re-export module:
   - From metrics/boundary_cells.py: `border_score`, `compute_region_coverage`
4. Updated `encoding/__init__.py` to export all 2 border symbols
5. All tests pass: 15 passed (new) + 28 passed (existing metrics/test_boundary_cells.py)
6. Ran ruff check/format and mypy - no issues
7. Code review APPROVED

**Files Created**:
- `src/neurospatial/encoding/border.py` (new - re-exports from metrics.boundary_cells)
- `tests/encoding/test_encoding_border.py` (new)

**Files Modified**:
- `src/neurospatial/encoding/__init__.py` (added 2 border exports)

**Milestone 6 Status**: Tasks 6.1, 6.2, 6.3, 6.4 COMPLETE
Remaining: Tasks 6.5-6.9

**Next Task**: Milestone 6, Task 6.5 - Create encoding/object_vector.py

---

### 2025-12-06 (Session 24)

**Starting Point**: Milestone 6 - Move encoding/ Module (Task 6.3: Create encoding/head_direction.py)

**Completed**: Create encoding/head_direction.py with re-exports from metrics/head_direction.py and stats/circular.py

**Work Done**:
1. Created test file `tests/encoding/test_encoding_head_direction.py` following TDD (RED phase)
   - 24 tests total: 5 import tests from encoding.head_direction, 3 re-export tests from stats.circular, 8 import tests from encoding, 2 module structure tests, 6 functionality tests
   - Tests for imports of all 8 symbols: HeadDirectionMetrics, head_direction_metrics, head_direction_tuning_curve, is_head_direction_cell, plot_head_direction_tuning, rayleigh_test, mean_resultant_length, circular_mean
2. Verified tests FAIL before implementation (import error expected)
3. Created `encoding/head_direction.py` as a re-export module:
   - From metrics/head_direction.py: `HeadDirectionMetrics`, `head_direction_metrics`, `head_direction_tuning_curve`, `is_head_direction_cell`, `plot_head_direction_tuning`
   - Re-exports from stats/circular.py: `rayleigh_test`, `mean_resultant_length`, `circular_mean`
4. Updated `encoding/__init__.py` to export all 8 head_direction symbols
5. All tests pass: 24 passed (new) + 76 passed (existing metrics/test_head_direction.py)
6. Ran ruff check/format and mypy - no issues
7. Code review APPROVED

**Files Created**:
- `src/neurospatial/encoding/head_direction.py` (new - re-exports from metrics.head_direction and stats.circular)
- `tests/encoding/test_encoding_head_direction.py` (new)

**Files Modified**:
- `src/neurospatial/encoding/__init__.py` (added 8 head_direction exports)

**Milestone 6 Status**: Tasks 6.1, 6.2, 6.3 COMPLETE
Remaining: Tasks 6.4-6.9

**Next Task**: Milestone 6, Task 6.4 - Create encoding/border.py

---

### 2025-12-06 (Session 23)

**Starting Point**: Milestone 6 - Move encoding/ Module (Task 6.2: Create encoding/grid.py)

**Completed**: Create encoding/grid.py with re-exports from metrics/grid_cells.py

**Work Done**:
1. Created test file `tests/encoding/test_encoding_grid.py` following TDD (RED phase)
   - 19 tests total: 7 import tests from encoding.grid, 4 import tests from encoding, 8 functionality tests
   - Tests for imports of all 7 symbols: GridProperties, grid_score, spatial_autocorrelation, grid_scale, grid_orientation, grid_properties, periodicity_score
2. Verified tests FAIL before implementation (import error expected)
3. Created `encoding/grid.py` as a re-export module:
   - From metrics/grid_cells.py: `GridProperties`, `grid_score`, `spatial_autocorrelation`, `grid_scale`, `grid_orientation`, `grid_properties`, `periodicity_score`
4. Updated `encoding/__init__.py` to export all 7 grid symbols
5. All tests pass: 19 passed (new) + 47 passed (existing metrics/test_grid_cells.py)
6. Ran ruff check/format and mypy - no issues
7. Code review APPROVED

**Files Created**:
- `src/neurospatial/encoding/grid.py` (new - re-exports from metrics.grid_cells)
- `tests/encoding/test_encoding_grid.py` (new)

**Files Modified**:
- `src/neurospatial/encoding/__init__.py` (added 7 grid exports)

**Milestone 6 Status**: Tasks 6.1, 6.2 COMPLETE
Remaining: Tasks 6.3-6.9

**Next Task**: Milestone 6, Task 6.3 - Create encoding/head_direction.py

---

### 2025-12-06 (Session 22)

**Starting Point**: Milestone 6 - Move encoding/ Module (Task 6.1: Create encoding/place.py)

**Completed**: Create encoding/place.py with re-exports from spike_field.py and metrics/place_fields.py

**Work Done**:
1. Created test file `tests/encoding/__init__.py` (test package init)
2. Created test file `tests/encoding/test_encoding_place.py` following TDD (RED phase)
   - 34 tests total: import tests for 19 functions + 1 class, functionality tests for 12 functions
   - Tests for imports from both `encoding.place` and `encoding/__init__.py`
3. Verified tests FAIL before implementation (import error expected)
4. Created `encoding/place.py` as a re-export module:
   - From spike_field.py: `DirectionalPlaceFields`, `compute_place_field`, `compute_directional_place_fields`, `spikes_to_field`
   - From metrics/place_fields.py: `detect_place_fields`, `skaggs_information`, `sparsity`, `selectivity`, `field_centroid`, `field_size`, `field_stability`, `field_shape_metrics`, `field_shift_distance`, `in_out_field_ratio`, `information_per_second`, `mutual_information`, `rate_map_coherence`, `spatial_coverage_single_cell`, `compute_field_emd`
5. Updated `encoding/__init__.py` to export all 20 symbols
6. All tests pass: 34 passed
7. Ran ruff check/format and mypy - no issues
8. Code review APPROVED

**Files Created**:
- `src/neurospatial/encoding/place.py` (new - re-exports from spike_field and metrics.place_fields)
- `tests/encoding/__init__.py` (new)
- `tests/encoding/test_encoding_place.py` (new)

**Files Modified**:
- `src/neurospatial/encoding/__init__.py` (added 20 exports from place.py)

**Milestone 6 Status**: Task 6.1 COMPLETE
Remaining: Tasks 6.2-6.9

**Next Task**: Milestone 6, Task 6.2 - Create encoding/grid.py

---

### 2025-12-06 (Session 21)

**Starting Point**: Milestone 5 - Move behavior/ Module (Task 5.5: Create behavior/reward.py)

**Completed**: Move reward.py → behavior/reward.py

**Work Done**:
1. Created test file `tests/behavior/test_behavior_reward.py` following TDD (RED phase)
   - 14 tests for reward functions: import tests and functionality tests
2. Verified tests FAIL before implementation (import error expected)
3. Moved `reward.py` → `behavior/reward.py` using `git mv` to preserve history
4. Updated module docstring with new import paths (Imports section at top)
5. Updated docstring examples (2 locations) from old import path to new
6. Added `__all__` export list with 2 exports: `goal_reward_field`, `region_reward_field`
7. Updated `behavior/__init__.py` to export 2 reward symbols
8. Updated internal imports (2 files):
   - `src/neurospatial/__init__.py` (updated import path)
   - `tests/test_reward.py` (updated import path)
9. All tests pass:
   - `tests/behavior/test_behavior_reward.py`: 14 passed
   - `tests/test_reward.py`: 15 passed
   - Total behavior tests: 162 passed
10. Ran ruff check/format and mypy - no issues

**Files Modified**:
- `src/neurospatial/behavior/reward.py` (moved from reward.py)
- `src/neurospatial/behavior/__init__.py` (added 2 reward exports)
- `src/neurospatial/__init__.py` (updated import path)
- `tests/behavior/test_behavior_reward.py` (new file)
- `tests/test_reward.py` (updated imports)

**Milestone 5 Status**: COMPLETE
All tasks (5.1-5.5) are done:
- 5.1: behavior/trajectory.py
- 5.2: behavior/segmentation.py
- 5.3: behavior/navigation.py
- 5.4: behavior/decisions.py
- 5.5: behavior/reward.py

**Next Task**: Milestone 6 - Move encoding/ Module

---

### 2025-12-06 (Session 20)

**Starting Point**: Milestone 5 - Move behavior/ Module (Task 5.4: Create behavior/decisions.py)

**Completed**: Create behavior/decisions.py by combining decision_analysis.py and vte.py

**Work Done**:
1. Created test file `tests/behavior/test_behavior_decisions.py` following TDD (RED phase)
   - 42 tests for all decision analysis and VTE functions, dataclasses, and re-exports
2. Verified tests FAIL before implementation (import error expected)
3. Created `behavior/decisions.py` by combining two source modules:
   - From metrics/decision_analysis.py: `PreDecisionMetrics`, `DecisionBoundaryMetrics`, `DecisionAnalysisResult`,
     `compute_decision_analysis`, `compute_pre_decision_metrics`, `decision_region_entry_time`,
     `detect_boundary_crossings`, `distance_to_decision_boundary`, `extract_pre_decision_window`,
     `geodesic_voronoi_labels`, `pre_decision_heading_stats`, `pre_decision_speed_stats`
   - From metrics/vte.py: `VTETrialResult`, `VTESessionResult`, `compute_vte_index`, `compute_vte_trial`,
     `compute_vte_session`, `classify_vte`, `head_sweep_from_positions`, `head_sweep_magnitude`,
     `integrated_absolute_rotation`, `normalize_vte_scores`, `wrap_angle`
4. Updated `behavior/__init__.py` to export all 23 decision/VTE symbols
5. Created re-export wrappers for backward compatibility:
   - `src/neurospatial/metrics/decision_analysis.py` (re-exports from behavior.decisions)
   - `src/neurospatial/metrics/vte.py` (re-exports from behavior.decisions)
6. All tests pass:
   - `tests/behavior/test_behavior_decisions.py`: 42 passed
   - `tests/metrics/test_decision_analysis.py`: 37 passed
   - `tests/metrics/test_vte.py`: 34 passed
   - Total decision/VTE related: 113 passed
7. Ran ruff check/format and mypy - no issues

**Files Created**:
- `src/neurospatial/behavior/decisions.py` (new - combined decision_analysis.py and vte.py)
- `tests/behavior/test_behavior_decisions.py` (new)

**Files Modified**:
- `src/neurospatial/behavior/__init__.py` (added 23 decision/VTE exports)
- `src/neurospatial/metrics/decision_analysis.py` (now re-export wrapper)
- `src/neurospatial/metrics/vte.py` (now re-export wrapper)

**Milestone 5 Status**: Tasks 5.1, 5.2, 5.3, 5.4 COMPLETE
Remaining: Tasks 5.5-5.6

**Next Task**: Milestone 5, Task 5.5 - Create behavior/reward.py

---

### 2025-12-06 (Session 19)

**Starting Point**: Milestone 5 - Move behavior/ Module (Task 5.3: Create behavior/navigation.py)

**Completed**: Move navigation functions from behavioral.py, path_efficiency.py, goal_directed.py → behavior/navigation.py

**Work Done**:
1. Created test file `tests/behavior/test_behavior_navigation.py` following TDD (RED phase)
   - 48 tests for all navigation functions, dataclasses, and re-exports
2. Verified tests FAIL before implementation (import error expected)
3. Created `behavior/navigation.py` by combining three source modules:
   - From behavioral.py: `path_progress`, `distance_to_region`, `cost_to_goal`, `time_to_goal`,
     `trials_to_region_arrays`, `graph_turn_sequence`, `goal_pair_direction_labels`, `heading_direction_labels`
   - From metrics/path_efficiency.py: `PathEfficiencyResult`, `SubgoalEfficiencyResult`,
     `traveled_path_length`, `shortest_path_length`, `path_efficiency`, `time_efficiency`,
     `angular_efficiency`, `subgoal_efficiency`, `compute_path_efficiency`
   - From metrics/goal_directed.py: `GoalDirectedMetrics`, `goal_vector`, `goal_direction`,
     `instantaneous_goal_alignment`, `goal_bias`, `approach_rate`, `compute_goal_directed_metrics`
4. Updated `behavior/__init__.py` to export all 24 navigation symbols
5. Updated internal imports in `src/neurospatial/__init__.py` and `src/neurospatial/metrics/__init__.py`
6. Created re-export wrappers for backward compatibility:
   - `src/neurospatial/behavioral.py` (re-exports from behavior.navigation)
   - `src/neurospatial/metrics/path_efficiency.py` (re-exports from behavior.navigation)
   - `src/neurospatial/metrics/goal_directed.py` (re-exports from behavior.navigation)
7. All tests pass:
   - `tests/behavior/test_behavior_navigation.py`: 48 passed
   - `tests/test_behavioral.py`: 51 passed
   - `tests/metrics/test_path_efficiency.py`: 22 passed
   - `tests/metrics/test_goal_directed.py`: 24 passed
   - Total navigation-related: 145 passed
8. Ran ruff check/format and mypy - no issues

**Files Created**:
- `src/neurospatial/behavior/navigation.py` (new - combined all navigation modules)
- `tests/behavior/test_behavior_navigation.py` (new)

**Files Modified**:
- `src/neurospatial/behavior/__init__.py` (added 24 navigation exports)
- `src/neurospatial/__init__.py` (updated import paths)
- `src/neurospatial/metrics/__init__.py` (re-export from behavior.navigation)
- `src/neurospatial/behavioral.py` (now re-export wrapper)
- `src/neurospatial/metrics/path_efficiency.py` (now re-export wrapper)
- `src/neurospatial/metrics/goal_directed.py` (now re-export wrapper)

**Milestone 5 Status**: Tasks 5.1, 5.2, 5.3 COMPLETE
Remaining: Tasks 5.4-5.6

**Next Task**: Milestone 5, Task 5.4 - Create behavior/decisions.py

---

### 2025-12-06 (Session 18)

**Starting Point**: Milestone 5 - Move behavior/ Module (Task 5.2: Create behavior/segmentation.py)

**Completed**: Move all segmentation/ functions to behavior/segmentation.py (NO backward compat)

**Work Done**:
1. Created test file `tests/behavior/test_behavior_segmentation.py` following TDD (RED phase)
   - 40 tests for all segmentation functions and dataclasses
   - Tests for imports from both `behavior.segmentation` and `behavior/__init__.py`
2. Verified tests FAIL before implementation (import error expected)
3. Created `behavior/segmentation.py` by combining all 4 segmentation submodules:
   - Dataclasses: `Crossing`, `Lap`, `Run`, `Trial`
   - Functions: `detect_region_crossings`, `detect_runs_between_regions`, `segment_by_velocity`,
     `detect_laps`, `segment_trials`, `trajectory_similarity`, `detect_goal_directed_runs`
4. Updated `behavior/__init__.py` to export all 11 segmentation symbols
5. NO backward-compatibility shim created per user request
6. Updated internal imports in 3 src/ files:
   - `src/neurospatial/__init__.py` (import from behavior.segmentation)
   - `src/neurospatial/behavioral.py` (Trial import)
   - `src/neurospatial/metrics/vte.py` (Trial import)
   - `src/neurospatial/io/nwb/_events.py` (docstring example)
7. Updated test imports in 10 test files:
   - `tests/test_direction_labels.py`
   - `tests/test_behavioral.py` (6 occurrences)
   - `tests/segmentation/test_similarity.py`
   - `tests/segmentation/test_integration.py`
   - `tests/segmentation/test_laps.py` (13 occurrences)
   - `tests/segmentation/test_regions.py` (16 occurrences)
   - `tests/segmentation/test_trials.py` (9 occurrences)
   - `tests/metrics/test_behavioral_integration.py`
   - `tests/metrics/test_vte.py` (3 occurrences)
   - `tests/nwb/test_trials.py` (13 occurrences)
8. Deleted old `src/neurospatial/segmentation/` directory entirely
9. All tests pass:
   - `tests/behavior/test_behavior_segmentation.py`: 40 passed
   - `tests/segmentation/`: 57 passed
   - Quick import verification passed
10. Ran ruff check/format and mypy - no issues

**Files Created**:
- `src/neurospatial/behavior/segmentation.py` (new - combined all segmentation modules)
- `tests/behavior/test_behavior_segmentation.py` (new)

**Files Modified**:
- `src/neurospatial/behavior/__init__.py` (added 11 segmentation exports)
- `src/neurospatial/__init__.py` (updated import path)
- `src/neurospatial/behavioral.py` (updated Trial import)
- `src/neurospatial/metrics/vte.py` (updated Trial import)
- `src/neurospatial/io/nwb/_events.py` (updated docstring example)
- 10 test files (updated imports)

**Files Deleted**:
- `src/neurospatial/segmentation/` (entire directory - no backward compat)

**Milestone 5 Status**: Tasks 5.1, 5.2 COMPLETE
Task 5.1 (Create behavior/trajectory.py) is complete.
Task 5.2 (Create behavior/segmentation.py) is complete.
Remaining: Tasks 5.3-5.6

**Next Task**: Milestone 5, Task 5.3 - Create behavior/navigation.py

---

### 2025-12-06 (Session 17)

**Starting Point**: Milestone 5 - Move behavior/ Module (Task 5.1: Create behavior/trajectory.py)

**Completed**: Move trajectory functions to behavior/trajectory.py

**Work Done**:
1. Created test file `tests/behavior/test_behavior_trajectory.py` following TDD (RED phase)
   - 18 tests for all trajectory functions importable from new location
   - Tests for imports from both `behavior.trajectory` and `behavior/__init__.py`
   - Tests for `compute_trajectory_curvature()` (moved from behavioral.py)
2. Verified tests FAIL before implementation (import error expected)
3. Moved `metrics/trajectory.py` → `behavior/trajectory.py` using `git mv` to preserve history
4. Added `compute_trajectory_curvature()` from `behavioral.py` to `behavior/trajectory.py`
5. Updated `behavior/__init__.py` to export all 5 trajectory functions:
   - `compute_turn_angles`, `compute_step_lengths`, `compute_home_range`,
   - `mean_square_displacement`, `compute_trajectory_curvature`
6. Updated `metrics/__init__.py` to re-export from new location for backward compatibility
7. Updated internal imports (6 files):
   - `src/neurospatial/behavioral.py` (import compute_turn_angles from behavior.trajectory)
   - `src/neurospatial/metrics/path_efficiency.py` (2 imports updated)
   - `tests/metrics/test_trajectory.py` (import updated)
   - `tests/environment/test_trajectory_metrics.py` (import updated)
   - `tests/segmentation/test_integration.py` (import updated)
8. Updated documentation:
   - `docs/user-guide/trajectory-and-behavioral-analysis.md` (4 import paths + migration note)
   - `src/neurospatial/behavioral.py` docstring (fixed reference to old path)
9. All tests pass:
   - `tests/behavior/test_behavior_trajectory.py`: 18 passed (new)
   - `tests/metrics/test_trajectory.py`: 19 passed (existing)
   - `tests/environment/test_trajectory_metrics.py`: 11 passed
   - `tests/segmentation/test_integration.py`: 4 passed
   - `tests/test_behavioral.py`: 51 passed
   - Total trajectory-related: 103+ passed
10. Ran ruff check/format (2 fixes) and mypy - no issues
11. Code review APPROVED

**Files Modified**:
- `src/neurospatial/behavior/trajectory.py` (moved from metrics/trajectory.py, added curvature function)
- `src/neurospatial/behavior/__init__.py` (added 5 trajectory exports)
- `src/neurospatial/metrics/__init__.py` (re-export from new location for backward compat)
- `src/neurospatial/behavioral.py` (updated import and docstring)
- `src/neurospatial/metrics/path_efficiency.py` (updated 2 imports)
- `tests/behavior/__init__.py` (new file)
- `tests/behavior/test_behavior_trajectory.py` (new file)
- `tests/metrics/test_trajectory.py` (updated imports)
- `tests/environment/test_trajectory_metrics.py` (updated imports)
- `tests/segmentation/test_integration.py` (updated imports)
- `docs/user-guide/trajectory-and-behavioral-analysis.md` (updated import paths)

**Milestone 5 Status**: Task 5.1 COMPLETE
Task 5.1 (Create behavior/trajectory.py) is complete.
Remaining: Task 5.2-5.6

**Next Task**: Milestone 5, Task 5.2 - Create behavior/segmentation.py

---

### 2025-12-06 (Session 16)

**Starting Point**: Milestone 4 - Move stats/ Module (Task 4.3: Create stats/surrogates.py)

**Completed**: Create stats/surrogates.py with extracted surrogate functions + NEW generate_jittered_spikes()

**Work Done**:
1. Created test file `tests/stats/test_stats_surrogates.py` following TDD (RED phase)
   - 21 tests for all surrogate functions importable from new location
   - Tests for new function: `generate_jittered_spikes()`
2. Verified tests FAIL before implementation (import error expected)
3. Created `stats/surrogates.py` with extracted functions from `stats/shuffle.py`:
   - `generate_poisson_surrogates()` - homogeneous Poisson surrogate spike trains
   - `generate_inhomogeneous_poisson_surrogates()` - time-varying rate surrogates
4. Added new function per PLAN.md:
   - `generate_jittered_spikes()` - temporal jitter surrogates for testing spike timing
5. Updated `stats/__init__.py` to export all 3 surrogate symbols from surrogates.py
6. Updated `stats/shuffle.py`:
   - Removed duplicate function implementations
   - Added re-export from `stats.surrogates` with `# noqa: F401` for backward compatibility
   - Updated module docstring to note surrogates are in separate module
7. All tests pass:
   - `tests/stats/test_stats_surrogates.py`: 21 passed
   - `tests/stats/`: 84 passed (1 skipped)
   - `tests/decoding/test_shuffle.py`: 163 passed
   - Total stats/shuffle related: 248 passed
8. Ran ruff check/format and mypy - no issues

**Files Modified**:
- `src/neurospatial/stats/surrogates.py` (NEW file - extracted from shuffle.py + new jitter function)
- `src/neurospatial/stats/__init__.py` (added 3 surrogate exports)
- `src/neurospatial/stats/shuffle.py` (removed duplicate functions, added re-export)
- `tests/stats/test_stats_surrogates.py` (NEW file)

**Milestone 4 Status**: COMPLETE
Task 4.1 (Create stats/circular.py) is complete.
Task 4.2 (Create stats/shuffle.py) is complete.
Task 4.3 (Create stats/surrogates.py) is complete.

**Next Task**: Milestone 5 - Move behavior/ Module

---

### 2025-12-06 (Session 15)

**Starting Point**: Milestone 4 - Move stats/ Module (Task 4.2: Create stats/shuffle.py)

**Completed**: Move shuffle functions to stats/shuffle.py (NO backward compatibility per user request)

**Work Done**:
1. Created test file `tests/stats/test_stats_shuffle.py` following TDD (RED phase)
   - Tests for all existing shuffle functions importable from new location
   - Tests for new functions: `shuffle_trials()`, `shuffle_spikes_isi()`
2. Verified tests FAIL before implementation (import error expected)
3. Moved `decoding/shuffle.py` → `stats/shuffle.py` using `git mv` to preserve history
4. Updated module docstring with new import paths
5. Added two new functions per PLAN.md:
   - `shuffle_trials()` - shuffles trial labels for testing trial identity significance
   - `shuffle_spikes_isi()` - shuffles inter-spike intervals for testing ISI ordering
6. Updated `stats/__init__.py` to export all 14 shuffle symbols:
   - ShuffleTestResult, shuffle_time_bins, shuffle_time_bins_coherent, shuffle_cell_identity
   - shuffle_place_fields_circular, shuffle_place_fields_circular_2d
   - shuffle_posterior_circular, shuffle_posterior_weighted_circular
   - generate_poisson_surrogates, generate_inhomogeneous_poisson_surrogates
   - compute_shuffle_pvalue, compute_shuffle_zscore
   - shuffle_trials, shuffle_spikes_isi (NEW)
7. NO backward-compatibility wrapper created (user explicitly said not needed)
8. Updated `decoding/__init__.py`:
   - Removed all shuffle function imports
   - Updated `__all__` to remove shuffle exports
   - Updated module docstring to remove shuffle documentation
9. Updated `decoding/trajectory.py` to import `_ensure_rng` from `stats.shuffle`
10. Updated all test imports:
    - `tests/decoding/test_shuffle.py` (batch sed update)
    - `tests/decoding/test_imports.py` (removed shuffle tests, updated expected exports)
11. All tests pass:
    - `tests/stats/test_stats_shuffle.py`: 33 passed
    - `tests/decoding/test_shuffle.py`: 163 passed
    - `tests/decoding/`: 568 passed total
12. Ran ruff check/format and mypy - no issues

**Files Modified**:
- `src/neurospatial/stats/shuffle.py` (moved from decoding/shuffle.py, added 2 new functions)
- `src/neurospatial/stats/__init__.py` (added 14 shuffle exports)
- `src/neurospatial/decoding/__init__.py` (removed shuffle exports and documentation)
- `src/neurospatial/decoding/trajectory.py` (updated _ensure_rng import)
- `tests/stats/test_stats_shuffle.py` (existing file updated with shuffle tests)
- `tests/decoding/test_shuffle.py` (updated imports)
- `tests/decoding/test_imports.py` (removed shuffle tests, updated expected exports)

**Milestone 4 Status**: Task 4.2 COMPLETE
Task 4.1 (Create stats/circular.py) is complete.
Task 4.2 (Create stats/shuffle.py) is complete.
Remaining: Task 4.3 (surrogates.py)

**Next Task**: Milestone 4, Task 4.3 - Create stats/surrogates.py

---

### 2025-12-06 (Session 14)

**Starting Point**: Milestone 4 - Move stats/ Module (Task 4.1: Create stats/circular.py)

**Completed**: Move circular statistics to stats/circular.py

**Work Done**:
1. Created test file `tests/stats/test_stats_circular.py` following TDD (RED phase)
   - 31 tests covering imports and basic functionality
2. Verified tests FAIL before implementation (import error expected)
3. Moved `metrics/circular.py` → `stats/circular.py` using `git mv` to preserve history
4. Updated module docstring with new import paths (Imports section at top)
5. Made private functions public:
   - `_circular_mean` → `circular_mean` (with full docstring and angle_unit support)
   - `_circular_variance` → `circular_variance` (with full docstring and angle_unit support)
   - `_mean_resultant_length` → `mean_resultant_length` (with full docstring and angle_unit support)
6. Added `wrap_angle()` function (moved from metrics/vte.py)
7. Added GLM circular basis functions (from metrics/circular_basis.py):
   - `CircularBasisResult` dataclass
   - `circular_basis()`, `circular_basis_metrics()`, `is_modulated()`
   - `plot_circular_basis_tuning()`
   - `_wald_test_magnitude()` (internal helper)
8. Updated `stats/__init__.py` to export all 15 symbols:
   - Core: rayleigh_test, circular_linear_correlation, circular_circular_correlation, phase_position_correlation
   - Public stats: circular_mean, circular_variance, mean_resultant_length
   - Utilities: wrap_angle
   - GLM basis: CircularBasisResult, circular_basis, circular_basis_metrics, is_modulated, plot_circular_basis_tuning
9. Created backward-compatibility wrapper `metrics/circular_basis.py`:
   - Re-exports all symbols from stats.circular
10. Updated internal imports (7 files):
    - `src/neurospatial/metrics/__init__.py` (consolidated circular imports from stats.circular)
    - `src/neurospatial/metrics/vte.py` (import wrap_angle from stats.circular)
    - `src/neurospatial/metrics/phase_precession.py` (import from stats.circular)
    - `src/neurospatial/metrics/head_direction.py` (import from stats.circular)
    - `src/neurospatial/metrics/object_vector_cells.py` (import from stats.circular)
    - `tests/metrics/test_circular.py` (batch sed update for 45+ imports)
11. All tests pass:
    - `tests/stats/test_stats_circular.py`: 31 passed
    - `tests/metrics/test_circular.py`: 133 passed
    - Total circular-related: 164 passed
12. Ran ruff check/format and mypy - no issues

**Files Modified**:
- `src/neurospatial/stats/circular.py` (moved from metrics/circular.py, extended with public functions and circular_basis content)
- `src/neurospatial/stats/__init__.py` (added 15 exports)
- `src/neurospatial/metrics/__init__.py` (import from stats.circular for backward compatibility)
- `src/neurospatial/metrics/circular_basis.py` (new backward-compat wrapper)
- `src/neurospatial/metrics/vte.py` (updated wrap_angle import)
- `src/neurospatial/metrics/phase_precession.py` (updated imports)
- `src/neurospatial/metrics/head_direction.py` (updated imports)
- `src/neurospatial/metrics/object_vector_cells.py` (updated import)
- `tests/stats/__init__.py` (new file)
- `tests/stats/test_stats_circular.py` (new file)
- `tests/metrics/test_circular.py` (updated imports)

**Milestone 4 Status**: Task 4.1 COMPLETE
Task 4.1 (Create stats/circular.py) is complete.
Remaining: Task 4.2 (shuffle.py), Task 4.3 (surrogates.py)

**Next Task**: Milestone 4, Task 4.2 - Create stats/shuffle.py

---

### 2025-12-06 (Session 13)

**Starting Point**: Continue Milestone 3 - Move nwb/ to io/nwb/

**Completed**: Move `nwb/` → `io/nwb/`

**Work Done**:
1. Created test file `tests/test_io_nwb_imports.py` following TDD (RED phase)
2. Verified tests FAIL before moving (import error expected)
3. Moved `nwb/` → `io/nwb/` using `git mv` to preserve history
4. Updated `io/nwb/__init__.py`:
   - Updated all docstring examples (5 locations) from `neurospatial.nwb` to `neurospatial.io.nwb`
   - Updated `_LAZY_IMPORTS` dictionary (22 entries) to use new paths
5. Updated internal imports in all nwb submodules:
   - `_behavior.py`, `_pose.py`, `_fields.py`, `_events.py`, `_environment.py`, `_overlays.py`
6. Updated external references (2 files):
   - `src/neurospatial/environment/serialization.py` (docstring + import)
   - `src/neurospatial/environment/factories.py` (docstrings + import)
7. Updated test file imports (10 test files) using batch sed replace
8. Moved `tests/nwb/` back from `tests/io/nwb/` to avoid conflict with Python's built-in `io` module
9. Renamed import test file to `tests/test_io_nwb_imports.py`
10. All tests pass:
    - `tests/test_io_nwb_imports.py`: 13 passed (new import verification)
    - `tests/nwb/`: 408 passed
    - `tests/test_events_nwb.py`: 23 passed
    - `tests/test_io.py`: 26 passed
11. Ran ruff check/format (3 fixes) and mypy - no issues

**Files Modified**:
- `src/neurospatial/io/nwb/__init__.py` (moved, updated paths)
- `src/neurospatial/io/nwb/_*.py` (7 files - updated imports)
- `src/neurospatial/environment/serialization.py` (updated import + docstring)
- `src/neurospatial/environment/factories.py` (updated imports + docstrings)
- `tests/test_io_nwb_imports.py` (new file)
- `tests/nwb/*.py` (10 files - updated imports)
- `tests/test_events_nwb.py` (updated imports)

**Milestone 3 Status**: COMPLETE
Both io/ components have been moved:
- `io.py` → `io/files.py` (done in M1)
- `nwb/` → `io/nwb/`

**Next Task**: Milestone 4 - Move stats/ Module

### 2025-12-05 (Session 12)

**Starting Point**: Continue Milestone 2 - Move ops/ Modules

**Completed**: Move `basis.py` → `ops/basis.py` (FINAL ops/ module)

**Work Done**:
1. Created new test file `tests/ops/test_ops_basis.py` following TDD (RED phase)
2. Verified tests FAIL before moving (import error expected)
3. Moved `basis.py` → `ops/basis.py` using `git mv` to preserve history
4. Updated module docstring with new import paths (4 locations)
5. `__all__` export list already present (6 exports):
   - `chebyshev_filter_basis`, `geodesic_rbf_basis`, `heat_kernel_wavelet_basis`,
   - `plot_basis_functions`, `select_basis_centers`, `spatial_basis`
6. Updated `ops/__init__.py` to export all basis functions (6 new exports)
7. Updated internal imports (2 files in src/):
   - `src/neurospatial/__init__.py`
   - `tests/test_basis.py`
8. All tests pass:
   - `tests/ops/test_ops_basis.py`: 12 passed
   - `tests/test_basis.py`: 38 passed
   - Total basis-related: 50 passed
9. Ran ruff check/format and mypy - no issues

**Files Modified**:
- `src/neurospatial/ops/basis.py` (moved from basis.py)
- `src/neurospatial/ops/__init__.py` (added 6 basis exports)
- `src/neurospatial/__init__.py` (updated import path)
- `tests/ops/test_ops_basis.py` (new file)
- `tests/test_basis.py` (updated imports)

**Milestone 2 Status**: COMPLETE
All 12 ops/ modules have been moved and are working:
- binning.py, distance.py, normalize.py, smoothing.py, graph.py, calculus.py
- transforms.py (merged calibration.py), alignment.py, egocentric.py
- visibility.py, basis.py

**Next Task**: Milestone 3 - Move io/ Module (partially done - io.py → io/files.py)

### 2025-12-05 (Session 11)

**Starting Point**: Continue Milestone 2 - Move ops/ Modules

**Completed**: Move `visibility.py` → `ops/visibility.py`

**Work Done**:
1. Created new test file `tests/ops/test_ops_visibility.py` following TDD (RED phase)
2. Verified tests FAIL before moving (import error expected)
3. Moved `visibility.py` → `ops/visibility.py` using `git mv` to preserve history
4. Updated module docstring in visibility.py to document new import paths
5. Updated all docstring examples (9 locations) from old import path to new
6. Fixed circular import by changing `from neurospatial import Environment` to `from neurospatial.environment import Environment`
7. Updated `ops/__init__.py` to export all visibility functions (8 exports):
   - `FieldOfView`, `ViewshedResult`, `compute_view_field`, `compute_viewed_location`,
   - `compute_viewshed`, `compute_viewshed_trajectory`, `visibility_occupancy`, `visible_cues`
8. Updated internal imports (4 files in src/):
   - `src/neurospatial/__init__.py`
   - `src/neurospatial/spatial_view_field.py`
   - `src/neurospatial/simulation/models/spatial_view_cells.py` (import + docstring)
9. Updated test imports:
   - `tests/test_visibility.py` (42+ import updates)
   - `tests/simulation/models/test_spatial_view_cells.py` (3 import updates)
10. All tests pass:
    - `tests/ops/test_ops_visibility.py`: 24 passed
    - `tests/test_visibility.py`: 45 passed
    - `tests/simulation/models/test_spatial_view_cells.py`: 30 passed
    - Total visibility-related: 99 passed
11. Ran ruff check/format and mypy - no issues

**Files Modified**:

- `src/neurospatial/ops/visibility.py` (moved from visibility.py)
- `src/neurospatial/ops/__init__.py` (added 8 visibility exports)
- `src/neurospatial/__init__.py` (updated import path)
- `src/neurospatial/spatial_view_field.py` (updated import)
- `src/neurospatial/simulation/models/spatial_view_cells.py` (updated imports)
- `tests/ops/test_ops_visibility.py` (new file)
- `tests/test_visibility.py` (updated imports)
- `tests/simulation/models/test_spatial_view_cells.py` (updated imports)

**Next Task**: Move `basis.py` → `ops/basis.py`

### 2025-12-05 (Session 10)

**Starting Point**: Continue Milestone 2 - Move ops/ Modules

**Completed**: Move `reference_frames.py` → `ops/egocentric.py`

**Work Done**:
1. Created new test file `tests/ops/test_ops_egocentric.py` following TDD (RED phase)
2. Verified tests FAIL before moving (import error expected)
3. Moved `reference_frames.py` → `ops/egocentric.py` using `git mv` to preserve history
4. Updated module docstring in egocentric.py to document new import paths
5. Updated all docstring examples (7 locations) from old import path to new
6. Updated `ops/__init__.py` to export all egocentric functions (7 exports):
   - `EgocentricFrame`, `allocentric_to_egocentric`, `egocentric_to_allocentric`,
   - `compute_egocentric_bearing`, `compute_egocentric_distance`,
   - `heading_from_velocity`, `heading_from_body_orientation`
7. Did NOT create backward-compatibility shim per PLAN.md ("No backward compatibility shims — clean break")
8. Updated internal imports (10+ files in src/):
   - `src/neurospatial/__init__.py`
   - `src/neurospatial/object_vector_field.py`
   - `src/neurospatial/visibility.py` (2 locations)
   - `src/neurospatial/metrics/object_vector_cells.py`
   - `src/neurospatial/metrics/vte.py`
   - `src/neurospatial/metrics/decision_analysis.py`
   - `src/neurospatial/metrics/goal_directed.py`
   - `src/neurospatial/simulation/models/spatial_view_cells.py` (2 locations)
   - `src/neurospatial/simulation/models/object_vector_cells.py`
9. Updated test imports in `tests/test_reference_frames.py` (35 import updates)
10. All tests pass:
    - `tests/ops/test_ops_egocentric.py`: 20 passed
    - `tests/test_reference_frames.py`: 35 passed
    - Total egocentric-related: 55 passed
11. Ran ruff check/format and mypy - no issues

**Files Modified**:
- `src/neurospatial/ops/egocentric.py` (moved from reference_frames.py)
- `src/neurospatial/ops/__init__.py` (added 7 egocentric exports)
- `src/neurospatial/__init__.py` (updated import path)
- `src/neurospatial/object_vector_field.py` (updated import)
- `src/neurospatial/visibility.py` (updated imports)
- `src/neurospatial/metrics/object_vector_cells.py` (updated import)
- `src/neurospatial/metrics/vte.py` (updated import)
- `src/neurospatial/metrics/decision_analysis.py` (updated import)
- `src/neurospatial/metrics/goal_directed.py` (updated import)
- `src/neurospatial/simulation/models/spatial_view_cells.py` (updated imports)
- `src/neurospatial/simulation/models/object_vector_cells.py` (updated import)
- `tests/ops/test_ops_egocentric.py` (new file)
- `tests/test_reference_frames.py` (updated imports)

**Next Task**: Move `visibility.py` → `ops/visibility.py`

### 2025-12-05 (Session 9)

**Starting Point**: Continue Milestone 2 - Move ops/ Modules

**Completed**: Move `alignment.py` → `ops/alignment.py`

**Work Done**:
1. Created new test file `tests/ops/test_ops_alignment.py` following TDD (RED phase)
2. Verified tests FAIL before moving (import error expected)
3. Moved `alignment.py` → `ops/alignment.py` using `git mv` to preserve history
4. Updated module docstring in alignment.py to document new import paths
5. Added `__all__` export list to alignment.py with 4 exports:
   - `ProbabilityMappingParams`, `get_2d_rotation_matrix`, `apply_similarity_transform`, `map_probabilities`
6. Updated `ops/__init__.py` to export all alignment functions (4 new exports)
7. Created backward-compatibility shim at `src/neurospatial/alignment.py`
8. Updated internal imports (3 files):
   - `src/neurospatial/__init__.py`
   - `tests/test_alignment.py`
   - `tests/test_properties.py`
9. Updated docstring examples in alignment.py (3 locations)
10. All tests pass:
    - `tests/ops/test_ops_alignment.py`: 20 passed
    - `tests/test_alignment.py`: 17 passed
    - Total alignment-related: 37 passed
11. Ran ruff check/format and mypy - no issues

**Files Modified**:
- `src/neurospatial/ops/alignment.py` (moved from alignment.py)
- `src/neurospatial/ops/__init__.py` (added 4 alignment exports)
- `src/neurospatial/alignment.py` (new backward-compat shim)
- `src/neurospatial/__init__.py` (updated import path)
- `tests/ops/test_ops_alignment.py` (new file)
- `tests/test_alignment.py` (updated imports)
- `tests/test_properties.py` (updated import)

**Next Task**: Move `reference_frames.py` → `ops/egocentric.py`

### 2025-12-05 (Session 8)

**Starting Point**: Continue Milestone 2 - Move ops/ Modules

**Completed**: Move `transforms.py` → `ops/transforms.py` (merged `calibration.py`)

**Work Done**:
1. Created new test file `tests/ops/test_ops_transforms.py` following TDD (RED phase)
2. Verified tests FAIL before moving (import error expected)
3. Moved `transforms.py` → `ops/transforms.py` using `git mv` to preserve history
4. Merged `simple_scale()` function from `calibration.py` into `ops/transforms.py`
5. Updated module docstring in transforms.py to document new import paths
6. Added `__all__` export list to transforms.py with 27 exports organized by category
7. Updated `ops/__init__.py` to export all transform functions (27 new exports)
8. Updated internal imports in 17+ source files:
   - `src/neurospatial/__init__.py`
   - `src/neurospatial/animation/__init__.py`
   - `src/neurospatial/animation/calibration.py`
   - `src/neurospatial/animation/overlays.py`
   - `src/neurospatial/annotation/*.py` (6 files)
   - `src/neurospatial/calibration.py`
   - `src/neurospatial/environment/transforms.py`
   - `src/neurospatial/regions/*.py` (3 files)
9. Created backward-compatibility shim at `src/neurospatial/transforms.py`:
   - Re-exports all symbols from `ops/transforms.py`
   - Allows old `from neurospatial.transforms import ...` to continue working
10. Updated test file imports (40+ test files):
    - Used `find ... -exec sed` to batch update
11. Added noqa directives for intentional style choices:
    - `RUF022` for organized `__all__` with category comments
    - `N806` for mathematical variable names (A, R, X for matrices)
12. All tests pass:
    - `tests/ops/test_ops_transforms.py`: 23 passed
    - `tests/test_transforms.py`: 19 passed
    - `tests/test_transforms_3d.py`: 23 passed
    - `tests/test_calibration.py`: 20 passed
    - `tests/environment/test_transforms.py`: 66 passed
    - Total transform-related: 201 passed
13. Ran ruff check/format and mypy - no issues

**Files Modified**:
- `src/neurospatial/ops/transforms.py` (moved from transforms.py)
- `src/neurospatial/ops/__init__.py` (added 27 transform exports)
- `src/neurospatial/transforms.py` (new backward-compat shim)
- `src/neurospatial/calibration.py` (updated import)
- `src/neurospatial/__init__.py` (updated import)
- `src/neurospatial/animation/__init__.py` (updated import)
- `src/neurospatial/animation/calibration.py` (updated import)
- `src/neurospatial/animation/overlays.py` (updated import)
- `src/neurospatial/annotation/*.py` (6 files updated)
- `src/neurospatial/environment/transforms.py` (updated imports)
- `src/neurospatial/regions/*.py` (3 files updated)
- `tests/ops/test_ops_transforms.py` (new file)
- `tests/test_transforms.py`, `tests/test_transforms_3d.py`, `tests/test_calibration.py` (updated imports)
- 40+ other test files (updated imports)

**Next Task**: Move `alignment.py` → `ops/alignment.py`

### 2025-12-05 (Session 7)

**Starting Point**: Continue Milestone 2 - Move ops/ Modules

**Completed**: Move `differential.py` → `ops/calculus.py`

**Work Done**:
1. Created new test file `tests/ops/test_ops_calculus.py` following TDD (RED phase)
2. Verified tests FAIL before moving (import error expected)
3. Moved `differential.py` → `ops/calculus.py` using `git mv` to preserve history
4. Updated module docstring in calculus.py to document new import paths
5. Added `__all__` export list to calculus.py:
   - `compute_differential_operator`, `gradient`, `divergence`
6. Updated `ops/__init__.py` to export public API
7. Updated internal imports (3 files):
   - `src/neurospatial/__init__.py`
   - `src/neurospatial/environment/core.py`
   - `tests/test_differential.py`
8. Updated docstring examples in calculus.py (3 locations)
9. Fixed `__future__` import order (must be at beginning of file)
10. All tests pass:
    - `tests/ops/test_ops_calculus.py`: 12 passed
    - `tests/test_differential.py`: 21 passed
    - Total calculus-related: 33 passed
11. Ran ruff check/format and mypy - no issues

**Files Modified**:
- `src/neurospatial/ops/calculus.py` (moved from differential.py)
- `src/neurospatial/ops/__init__.py` (added calculus exports)
- `src/neurospatial/__init__.py` (updated import path)
- `src/neurospatial/environment/core.py` (updated import)
- `tests/ops/test_ops_calculus.py` (new file)
- `tests/test_differential.py` (updated imports)

**Next Task**: Move `transforms.py` → `ops/transforms.py`

### 2025-12-05 (Session 6)

**Starting Point**: Continue Milestone 2 - Move ops/ Modules

**Completed**: Move `primitives.py` → `ops/graph.py`

**Work Done**:
1. Moved `primitives.py` → `ops/graph.py` using `git mv` to preserve history
2. Added module docstring in graph.py to document new import paths
3. Added `__all__` export list to graph.py:
   - `neighbor_reduce`, `convolve`
4. Updated `ops/__init__.py` to export public API
5. Created new test file `tests/ops/test_ops_graph.py` following TDD
6. Updated internal imports (3 files):
   - `src/neurospatial/__init__.py`
   - `src/neurospatial/metrics/place_fields.py`
   - `tests/test_primitives.py`
7. Updated docstring examples in graph.py (2 locations)
8. All tests pass:
   - `tests/ops/test_ops_graph.py`: 12 passed
   - `tests/test_primitives.py`: 15 passed
   - Total graph-related: 27 passed
9. Ran ruff check/format and mypy - no issues

**Files Modified**:
- `src/neurospatial/ops/graph.py` (moved from primitives.py)
- `src/neurospatial/ops/__init__.py` (added graph exports)
- `src/neurospatial/__init__.py` (updated import path)
- `src/neurospatial/metrics/place_fields.py` (updated import)
- `tests/ops/test_ops_graph.py` (new file)
- `tests/test_primitives.py` (updated imports)

**Next Task**: Move `differential.py` → `ops/calculus.py`

### 2025-12-05 (Session 5)

**Starting Point**: Continue Milestone 2 - Move ops/ Modules

**Completed**: Move `kernels.py` → `ops/smoothing.py`

**Work Done**:
1. Moved `kernels.py` → `ops/smoothing.py` using `git mv` to preserve history
2. Added module docstring in smoothing.py to document new import paths
3. Added `__all__` export list to smoothing.py:
   - `compute_diffusion_kernels`, `apply_kernel`
4. Updated `ops/__init__.py` to export public API
5. Created new test file `tests/ops/test_ops_smoothing.py` following TDD
6. Updated internal imports (6 files):
   - `src/neurospatial/__init__.py`
   - `src/neurospatial/environment/fields.py`
   - `src/neurospatial/ops/binning.py`
   - `tests/test_kernels.py`
   - `tests/environment/test_apply_kernel.py`
   - `tests/benchmarks/test_performance.py`
7. All tests pass:
   - `tests/ops/test_ops_smoothing.py`: 15 passed
   - `tests/test_kernels.py`: 37 passed
   - `tests/environment/test_apply_kernel.py`: 16 passed
   - Total smoothing-related: 103 passed
8. Ran ruff check/format and mypy - no issues

**Files Modified**:
- `src/neurospatial/ops/smoothing.py` (moved from kernels.py)
- `src/neurospatial/ops/__init__.py` (added smoothing exports)
- `src/neurospatial/__init__.py` (updated import path)
- `src/neurospatial/environment/fields.py` (updated import)
- `src/neurospatial/ops/binning.py` (updated import)
- `tests/ops/test_ops_smoothing.py` (new file)
- `tests/test_kernels.py` (updated imports)
- `tests/environment/test_apply_kernel.py` (updated import)
- `tests/benchmarks/test_performance.py` (updated import)

**Next Task**: Move `primitives.py` → `ops/graph.py`

### 2025-12-05 (Session 4)

**Starting Point**: Continue Milestone 2 - Move ops/ Modules

**Completed**: Move `field_ops.py` → `ops/normalize.py`

**Work Done**:
1. Moved `field_ops.py` → `ops/normalize.py` using `git mv` to preserve history
2. Updated module docstring in normalize.py to reflect new import paths
3. Updated `ops/__init__.py` to export public API:
   - `normalize_field`, `clamp`, `combine_fields`
4. Created new test file `tests/ops/test_ops_normalize.py` following TDD
5. Updated internal imports:
   - `src/neurospatial/__init__.py`
   - `tests/test_field_ops.py`
6. All tests pass:
   - `tests/ops/test_ops_normalize.py`: 15 passed
   - `tests/test_field_ops.py`: 31 passed
   - Total normalize-related: 46 passed
7. Ran ruff check/format and mypy - no issues

**Files Modified**:
- `src/neurospatial/ops/normalize.py` (moved from field_ops.py)
- `src/neurospatial/ops/__init__.py` (added normalize exports)
- `src/neurospatial/__init__.py` (updated import path)
- `tests/ops/test_ops_normalize.py` (new file)
- `tests/test_field_ops.py` (updated import)

**Next Task**: Move `kernels.py` → `ops/smoothing.py`

### 2025-12-05 (Session 3)

**Starting Point**: Continue Milestone 2 - Move ops/ Modules

**Completed**: Move `distance.py` → `ops/distance.py`

**Work Done**:
1. Moved `distance.py` → `ops/distance.py` using `git mv` to preserve history
2. Updated docstrings in distance.py to reflect new import paths
3. Updated all internal imports (18 files across src/ and tests/)
4. Updated `ops/__init__.py` to export public API:
   - `distance_field`, `euclidean_distance_matrix`, `geodesic_distance_matrix`,
   - `geodesic_distance_between_points`, `neighbors_within`, `pairwise_distances`
5. Created new test file `tests/ops/test_ops_distance.py` following TDD
6. All tests pass:
   - `tests/ops/test_ops_distance.py`: 13 passed
   - `tests/test_distance.py`: 30 passed
   - `tests/test_distance_new.py`: 40 passed
   - `tests/test_distance_field_extended.py`: 30 passed
   - `tests/test_neighbors_within.py`: 27 passed
   - Total distance-related: 158 passed
7. Ran ruff check and mypy - no issues

**Files Modified**:
- `src/neurospatial/ops/distance.py` (moved from distance.py)
- `src/neurospatial/ops/__init__.py` (added distance exports)
- `src/neurospatial/__init__.py` (updated import)
- `src/neurospatial/reward.py` (updated import)
- `src/neurospatial/object_vector_field.py` (updated import)
- `src/neurospatial/reference_frames.py` (updated import)
- `src/neurospatial/behavioral.py` (updated 4 imports)
- `src/neurospatial/primitives.py` (updated import)
- `src/neurospatial/events/regressors.py` (updated 2 imports)
- `src/neurospatial/environment/queries.py` (updated import)
- `src/neurospatial/metrics/decision_analysis.py` (updated 2 imports)
- `src/neurospatial/metrics/path_efficiency.py` (updated import)
- `src/neurospatial/metrics/trajectory.py` (updated 2 imports)
- `src/neurospatial/simulation/models/boundary_cells.py` (updated import)
- `src/neurospatial/simulation/models/place_cells.py` (updated import)
- `src/neurospatial/simulation/models/object_vector_cells.py` (updated import)
- `tests/ops/test_ops_distance.py` (new file)
- `tests/test_distance.py` (updated import)
- `tests/test_distance_new.py` (updated import)
- `tests/test_distance_field_extended.py` (updated import)
- `tests/test_distance_utilities.py` (updated import)
- `tests/test_neighbors_within.py` (updated import)
- `tests/test_reward.py` (updated 4 imports)

**Next Task**: Move `field_ops.py` → `ops/normalize.py`

### 2025-12-05 (Session 2)

**Starting Point**: Begin Milestone 2 - Move ops/ Modules

**Completed**: Move `spatial.py` → `ops/binning.py`

**Work Done**:
1. Moved `spatial.py` → `ops/binning.py` using `git mv` to preserve history
2. Updated docstrings in binning.py to reflect new import paths
3. Updated all internal imports (17 files across src/ and tests/)
4. Updated `ops/__init__.py` to export public API:
   - `TieBreakStrategy`, `map_points_to_bins`, `regions_to_mask`, `resample_field`, `clear_kdtree_cache`
5. Created new test file `tests/ops/test_binning.py` following TDD
6. All tests pass:
   - `tests/ops/test_binning.py`: 7 passed
   - `tests/test_spatial.py`: 30 passed
   - `tests/test_resample_field.py`: 15 passed
   - `tests/test_deterministic_kdtree.py`: 18 passed
   - `tests/environment/test_core.py`: 53 passed
7. Ran ruff check and mypy - no issues

**Files Modified**:
- `src/neurospatial/ops/binning.py` (moved from spatial.py)
- `src/neurospatial/ops/__init__.py` (added exports)
- `src/neurospatial/__init__.py` (updated import)
- `src/neurospatial/reward.py` (updated import)
- `src/neurospatial/environment/regions.py` (updated import)
- `src/neurospatial/environment/trajectory.py` (updated import)
- `src/neurospatial/environment/fields.py` (updated import)
- `src/neurospatial/segmentation/regions.py` (updated 2 imports)
- `src/neurospatial/segmentation/trials.py` (updated import)
- `src/neurospatial/segmentation/similarity.py` (updated import)
- `tests/ops/test_binning.py` (new file)
- `tests/test_spatial.py` (updated import)
- `tests/test_resample_field.py` (updated import)
- `tests/test_deterministic_kdtree.py` (updated import)
- `tests/test_copy.py` (updated 2 imports)
- `tests/environment/test_region_mask_method.py` (updated import)
- `tests/environment/test_regions_to_mask.py` (updated import)
- `tests/environment/test_core.py` (updated 2 imports)

**Next Task**: Move `distance.py` → `ops/distance.py`

### 2025-12-05 (Session 1)

**Starting Point**: Fresh start on package reorganization per PLAN.md and TASKS.md

**Completed**: Milestone 1 - Create Directory Structure

**Work Done**:
1. Created 5 new directories with `__init__.py` files:
   - `src/neurospatial/encoding/`
   - `src/neurospatial/behavior/`
   - `src/neurospatial/io/`
   - `src/neurospatial/ops/`
   - `src/neurospatial/stats/`
2. Moved `io.py` → `io/files.py` early (from M3) to avoid import conflict
   - The new `io/` directory shadowed the old `io.py` file
   - Updated `io/__init__.py` to re-export `to_file`, `from_file`, `to_dict`, `from_dict`
3. All core tests pass (io tests: 26 passed, environment tests: 869 passed)

**Pre-existing Test Failures Noted** (not related to my changes):
- `test_repeated_mazes.py` - RepeatedTDims missing `n_t_junctions` attribute
- Some flaky tests in boundary_cells and properties
- These were confirmed pre-existing by checking git stash behavior

---

## Decisions Made

1. **Moved io.py → io/files.py early**: When creating `io/` directory, Python found it before `io.py`, causing import errors. Rather than a hacky workaround, moved the file early per PLAN.md structure.

---

## Blockers

*(none)*

---

## Questions for User

*(none)*

---

## Pre-existing Issues

The following tests fail but are NOT related to the reorganization:
- `tests/simulation/mazes/test_repeated_mazes.py` - API mismatch (`n_t_junctions`)
- `tests/metrics/test_boundary_cells.py::TestBorderScore::test_border_score_all_nan` - flaky
- `tests/test_properties.py::TestSparsityProperties::test_single_peak_low_sparsity`
