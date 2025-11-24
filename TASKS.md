# TASKS: Goal-Directed Metrics and Behavioral Analysis (v0.8.0)

**Version**: v0.8.0
**Status**: Not Started
**Last Updated**: 2025-11-24

> **Context**: See [PLAN.md](PLAN.md) for detailed specifications and design rationale.
> **Workflow**: See [freshstart.md](.claude/commands/freshstart.md) for TDD workflow requirements.

---

## Progress Overview

- **Milestone 1**: Public API Fixes - 1/1 complete ✅
- **Milestone 2**: Foundation Functions - 25/25 complete ✅ (M2.1 ✅, M2.2 ✅, M2.3 ✅, M2.4 ✅)
- **Milestone 3**: Time and Curvature - 15/15 complete ✅ (M3.1 ✅, M3.2 ✅)
- **Milestone 4**: Cost and Turn Analysis - 0/19 complete
- **Milestone 5**: Documentation - 0/5 complete

**Total**: 41/70 tasks complete

---

## Milestone 1: Public API Fixes (Phase 1)

**Goal**: Export hidden segmentation functions to public API
**Estimated Time**: 30 minutes
**Dependencies**: None

### M1.1: Add segmentation exports to **init**.py ✅ COMPLETE

- [x] **M1.1.1**: Read `src/neurospatial/segmentation/` to verify function names
  - Files: `detect_goal_directed_runs.py`, `detect_runs_between_regions.py`, `segment_by_velocity.py`
  - Confirm function signatures and existence

- [x] **M1.1.2**: Update `src/neurospatial/__init__.py`
  - Add imports for: `detect_goal_directed_runs`, `detect_runs_between_regions`, `segment_by_velocity`
  - Update `__all__` list with new exports
  - Verify no circular imports

- [x] **M1.1.3**: Write import tests in `tests/test_segmentation.py`
  - **TDD**: Create test file first (or update existing)
  - Test 1: `test_detect_goal_directed_runs_exported()`
  - Test 2: `test_detect_runs_between_regions_exported()`
  - Test 3: `test_segment_by_velocity_exported()`
  - **RUN**: `uv run pytest tests/test_segmentation.py -v` (should FAIL initially)

- [x] **M1.1.4**: Verify tests pass
  - **RUN**: `uv run pytest tests/test_segmentation.py -v`
  - All import tests should PASS

- [x] **M1.1.5**: Run code quality checks
  - **RUN**: `uv run ruff check . && uv run ruff format .`
  - **RUN**: `uv run mypy src/neurospatial/__init__.py`
  - Fix any issues

- [x] **M1.1.6**: Commit changes
  - **COMMIT**: `feat(api): export segmentation functions to public API` (commit a738b5e)

---

## Milestone 2: Foundation Functions (Phase 2)

**Goal**: Implement `trials_to_region_arrays()`, `path_progress()`, `distance_to_region()`
**Estimated Time**: 3-4 hours
**Dependencies**: Milestone 1 complete

### M2.1: Create behavioral.py module

- [x] **M2.1.1**: Create `src/neurospatial/behavioral.py`
  - Add module docstring explaining purpose (behavioral/RL metrics)
  - Add imports: `numpy`, `networkx`, `typing`, `Environment`
  - Add skeleton for 7 functions (empty implementations with docstrings)

- [x] **M2.1.2**: Create test file `tests/test_behavioral.py`
  - **TDD**: Create test file BEFORE implementing functions
  - Add imports and test class structure
  - Create placeholder test functions (all marked `@pytest.mark.skip("not implemented")`)

### M2.2: Implement trials_to_region_arrays() helper ✅ COMPLETE

- [x] **M2.2.1**: Write tests FIRST (TDD)
  - **FILE**: `tests/test_behavioral.py`
  - Test 1: `test_trials_to_region_arrays_single_trial()`
  - Test 2: `test_trials_to_region_arrays_multiple_trials()`
  - Test 3: `test_trials_to_region_arrays_failed_trial()` (end_region=None)
  - Test 4: `test_trials_to_region_arrays_polygon_regions()` (multi-bin)
  - **RUN**: `uv run pytest tests/test_behavioral.py::test_trials_to_region_arrays -v` (should FAIL)

- [x] **M2.2.2**: Implement `trials_to_region_arrays()`
  - **FILE**: `src/neurospatial/behavioral.py`
  - Follow spec from PLAN.md lines 327-396
  - Use `env.bins_in_region()` for region-to-bin mapping
  - Handle failed trials (end_region=None → goal_bins=-1)

- [x] **M2.2.3**: Run tests until PASS
  - **RUN**: `uv run pytest tests/test_behavioral.py::test_trials_to_region_arrays -v`
  - Debug failures, iterate until all tests PASS

- [x] **M2.2.4**: Code review and refactor
  - **REVIEW**: Check for edge cases, type safety, docstring completeness
  - Refactor for clarity

- [x] **M2.2.5**: Run quality checks
  - **RUN**: `uv run mypy src/neurospatial/behavioral.py`
  - **RUN**: `uv run ruff check src/neurospatial/behavioral.py`
  - Fix any issues

- [x] **M2.2.6**: Commit
  - **COMMIT**: `feat(behavioral): implement trials_to_region_arrays helper` (commit 0de2c87)

### M2.3: Implement path_progress() ✅ COMPLETE

- [x] **M2.3.1**: Write tests FIRST (TDD)
  - Test 1: `test_path_progress_single_trial_geodesic()`
  - Test 2: `test_path_progress_multiple_trials()`
  - Test 3: `test_path_progress_euclidean()`
  - Test 4: `test_path_progress_edge_case_same_start_goal()` (should return 1.0)
  - Test 5: `test_path_progress_edge_case_disconnected()` (should return NaN)
  - Test 6: `test_path_progress_edge_case_invalid_bins()` (should return NaN)
  - Test 7: `test_path_progress_large_environment()` (n_bins > 5000, test fallback)
  - **RUN**: `uv run pytest tests/test_behavioral.py::test_path_progress -v` (should FAIL)

- [x] **M2.3.2**: Implement `path_progress()` - geodesic metric
  - **FILE**: `src/neurospatial/behavioral.py`
  - Follow spec from PLAN.md lines 188-320
  - Implement small environment strategy (n_bins < 5000, precompute matrix)
  - Add explicit fitted check (not `@check_fitted`)
  - Handle all edge cases documented in spec

- [x] **M2.3.3**: Run geodesic tests until PASS
  - **RUN**: `uv run pytest tests/test_behavioral.py::test_path_progress -k geodesic -v`
  - Debug and iterate

- [x] **M2.3.4**: Implement euclidean metric support
  - Add `euclidean_distance_matrix()` branch
  - Use `env.bin_centers` for euclidean calculation

- [x] **M2.3.5**: Run all tests until PASS
  - **RUN**: `uv run pytest tests/test_behavioral.py::test_path_progress -v`
  - All tests should PASS

- [x] **M2.3.6**: Implement large environment fallback
  - Add strategy for n_bins >= 5000 (per-pair distance fields)
  - Test memory efficiency

- [x] **M2.3.7**: Code review and refactor
  - **REVIEW**: Vectorization correctness, memory safety, performance

- [x] **M2.3.8**: Run quality checks
  - **RUN**: `uv run mypy src/neurospatial/behavioral.py`
  - **RUN**: `uv run ruff check src/neurospatial/behavioral.py`

- [x] **M2.3.9**: Commit
  - **COMMIT**: `feat(behavioral): implement path_progress with geodesic/euclidean metrics` (commit e3ab9ea)

### M2.4: Implement distance_to_region() ✅ COMPLETE

- [x] **M2.4.1**: Write tests FIRST (TDD)
  - Test 1: `test_distance_to_region_scalar_target()`
  - Test 2: `test_distance_to_region_dynamic_target()` (array of targets)
  - Test 3: `test_distance_to_region_invalid_bins()` (should return NaN)
  - Test 4: `test_distance_to_region_multiple_goal_bins()` (distance to nearest)
  - Test 5: `test_distance_to_region_large_environment()` (memory fallback)
  - **RUN**: `uv run pytest tests/test_behavioral.py::test_distance_to_region -v` (should FAIL)

- [x] **M2.4.2**: Implement `distance_to_region()` - scalar target
  - **FILE**: `src/neurospatial/behavioral.py`
  - Follow spec from PLAN.md lines 407-507
  - Use `env.distance_to()` for scalar targets (already exists)

- [x] **M2.4.3**: Run scalar target tests until PASS
  - **RUN**: `uv run pytest tests/test_behavioral.py::test_distance_to_region_scalar -v`

- [x] **M2.4.4**: Implement dynamic target support (array)
  - Add distance matrix precomputation for small envs
  - Add per-target distance field for large envs

- [x] **M2.4.5**: Run all tests until PASS
  - **RUN**: `uv run pytest tests/test_behavioral.py::test_distance_to_region -v`

- [x] **M2.4.6**: Code review and refactor
  - **REVIEW**: Delegation to `env.distance_to()`, memory efficiency

- [x] **M2.4.7**: Run quality checks
  - **RUN**: `uv run mypy src/neurospatial/behavioral.py`
  - **RUN**: `uv run ruff check src/neurospatial/behavioral.py`

- [x] **M2.4.8**: Commit
  - **COMMIT**: `feat(behavioral): implement distance_to_region for scalar and dynamic targets` (commit 69a0289)

---

## Milestone 3: Time and Curvature (Phase 3)

**Goal**: Implement `time_to_goal()` and `compute_trajectory_curvature()`
**Estimated Time**: 2-3 hours
**Dependencies**: Milestone 2 complete

### M3.1: Implement time_to_goal() ✅ COMPLETE

- [x] **M3.1.1**: Write tests FIRST (TDD)
  - Test 1: `test_time_to_goal_successful_trials()`
  - Test 2: `test_time_to_goal_failed_trials()` (should be NaN)
  - Test 3: `test_time_to_goal_outside_trials()` (should be NaN)
  - Test 4: `test_time_to_goal_countdown()` (verify countdown is correct)
  - Test 5: `test_time_to_goal_after_goal_reached()` (should be 0.0)
  - **RUN**: `uv run pytest tests/test_behavioral.py::test_time_to_goal -v` (should FAIL)

- [x] **M3.1.2**: Implement `time_to_goal()`
  - **FILE**: `src/neurospatial/behavioral.py`
  - Follow spec from PLAN.md lines 633-705
  - Handle all edge cases (failed trials, outside trials)

- [x] **M3.1.3**: Run tests until PASS
  - **RUN**: `uv run pytest tests/test_behavioral.py::test_time_to_goal -v`

- [x] **M3.1.4**: Code review and refactor
  - **REVIEW**: Edge case handling, vectorization

- [x] **M3.1.5**: Run quality checks
  - **RUN**: `uv run mypy src/neurospatial/behavioral.py`
  - **RUN**: `uv run ruff check src/neurospatial/behavioral.py`

- [x] **M3.1.6**: Commit
  - **COMMIT**: `feat(behavioral): implement time_to_goal` (commit d6a27b7)

### M3.2: Implement compute_trajectory_curvature() ✅ COMPLETE

- [x] **M3.2.1**: Read existing `compute_turn_angles()` implementation
  - **FILE**: `src/neurospatial/metrics/trajectory.py` (lines 31-165)
  - Understand input/output format, stationary filtering, angle calculation

- [x] **M3.2.2**: Write tests FIRST (TDD)
  - Test 1: `test_compute_trajectory_curvature_2d_straight()` (should be ~0)
  - Test 2: `test_compute_trajectory_curvature_2d_left_turn()` (positive)
  - Test 3: `test_compute_trajectory_curvature_2d_right_turn()` (negative)
  - Test 4: `test_compute_trajectory_curvature_3d()` (uses first 2 dims)
  - Test 5: `test_compute_trajectory_curvature_smoothing()` (with temporal smoothing)
  - Test 6: `test_compute_trajectory_curvature_output_length()` (n_samples, not n_samples-2)
  - **RUN**: `uv run pytest tests/test_behavioral.py::test_compute_trajectory_curvature -v` (should FAIL)

- [x] **M3.2.3**: Implement `compute_trajectory_curvature()` - basic version
  - **FILE**: `src/neurospatial/behavioral.py`
  - Follow spec from PLAN.md lines 714-813
  - Call `compute_turn_angles()` internally
  - Pad result to n_samples length

- [x] **M3.2.4**: Run basic tests until PASS
  - **RUN**: `uv run pytest tests/test_behavioral.py::test_compute_trajectory_curvature -k "not smoothing" -v`

- [x] **M3.2.5**: Add temporal smoothing support
  - Import `gaussian_filter1d` from scipy (inside function)
  - Add smoothing logic with sigma calculation

- [x] **M3.2.6**: Run all tests until PASS
  - **RUN**: `uv run pytest tests/test_behavioral.py::test_compute_trajectory_curvature -v`

- [x] **M3.2.7**: Code review and refactor
  - **REVIEW**: Reuse of `compute_turn_angles()`, padding strategy, smoothing correctness

- [x] **M3.2.8**: Run quality checks
  - **RUN**: `uv run mypy src/neurospatial/behavioral.py`
  - **RUN**: `uv run ruff check src/neurospatial/behavioral.py`

- [x] **M3.2.9**: Commit
  - **COMMIT**: `feat(behavioral): implement compute_trajectory_curvature with smoothing` (commit 1624063)

---

## Milestone 4: Cost and Turn Analysis (Phase 4)

**Goal**: Implement `cost_to_goal()` and `graph_turn_sequence()`
**Estimated Time**: 3-4 hours
**Dependencies**: Milestone 3 complete

### M4.1: Implement cost_to_goal()

- [ ] **M4.1.1**: Write tests FIRST (TDD)
  - Test 1: `test_cost_to_goal_uniform()` (equivalent to geodesic distance)
  - Test 2: `test_cost_to_goal_with_cost_map()` (punishment zones)
  - Test 3: `test_cost_to_goal_terrain_difficulty()` (narrow passages)
  - Test 4: `test_cost_to_goal_combined()` (cost map + terrain)
  - Test 5: `test_cost_to_goal_dynamic_goal()` (array of goal bins)
  - Test 6: `test_cost_to_goal_invalid_bins()` (should handle gracefully)
  - **RUN**: `uv run pytest tests/test_behavioral.py::test_cost_to_goal -v` (should FAIL)

- [ ] **M4.1.2**: Implement `cost_to_goal()` - uniform cost (baseline)
  - **FILE**: `src/neurospatial/behavioral.py`
  - Follow spec from PLAN.md lines 516-624
  - For uniform cost, delegate to `distance_to_region()`

- [ ] **M4.1.3**: Run uniform cost tests until PASS
  - **RUN**: `uv run pytest tests/test_behavioral.py::test_cost_to_goal_uniform -v`

- [ ] **M4.1.4**: Add cost map support
  - Create weighted graph with modified edge weights
  - Use `distance_field()` with custom weights

- [ ] **M4.1.5**: Run cost map tests until PASS
  - **RUN**: `uv run pytest tests/test_behavioral.py::test_cost_to_goal -k "cost_map" -v`

- [ ] **M4.1.6**: Add terrain difficulty support
  - Multiply base distance by terrain difficulty
  - Combine with cost map if both present

- [ ] **M4.1.7**: Run all tests until PASS
  - **RUN**: `uv run pytest tests/test_behavioral.py::test_cost_to_goal -v`

- [ ] **M4.1.8**: Code review and refactor
  - **REVIEW**: Graph modification correctness, weight calculation, edge case handling

- [ ] **M4.1.9**: Run quality checks
  - **RUN**: `uv run mypy src/neurospatial/behavioral.py`
  - **RUN**: `uv run ruff check src/neurospatial/behavioral.py`

- [ ] **M4.1.10**: Commit
  - **COMMIT**: `feat(behavioral): implement cost_to_goal with cost maps and terrain difficulty`

### M4.2: Implement graph_turn_sequence()

- [ ] **M4.2.1**: Create test fixtures for Y-maze and T-maze
  - **FILE**: `tests/conftest.py`
  - Add `ymaze_environment()` fixture
  - Add `tmaze_trajectory()` fixture
  - Fixtures should include known turn sequences

- [ ] **M4.2.2**: Write tests FIRST (TDD)
  - Test 1: `test_graph_turn_sequence_ymaze_left()` (single left turn)
  - Test 2: `test_graph_turn_sequence_ymaze_right()` (single right turn)
  - Test 3: `test_graph_turn_sequence_grid_multiple()` (multiple turns on grid)
  - Test 4: `test_graph_turn_sequence_straight()` (no turns, empty string)
  - Test 5: `test_graph_turn_sequence_min_samples_filter()` (filters brief crossings)
  - Test 6: `test_graph_turn_sequence_3d()` (3D environment)
  - **RUN**: `uv run pytest tests/test_behavioral.py::test_graph_turn_sequence -v` (should FAIL)

- [ ] **M4.2.3**: Implement `graph_turn_sequence()` - 2D version
  - **FILE**: `src/neurospatial/behavioral.py`
  - Follow spec from PLAN.md lines 828-902
  - Infer transitions from consecutive bin pairs
  - Filter by `min_samples_per_edge`
  - Compute turn directions using cross product

- [ ] **M4.2.4**: Run 2D tests until PASS
  - **RUN**: `uv run pytest tests/test_behavioral.py::test_graph_turn_sequence -k "not 3d" -v`

- [ ] **M4.2.5**: Add N-D support (projection to primary movement plane)
  - Import PCA from sklearn (inside function)
  - Project to first 2 principal components for N-D

- [ ] **M4.2.6**: Run all tests until PASS
  - **RUN**: `uv run pytest tests/test_behavioral.py::test_graph_turn_sequence -v`

- [ ] **M4.2.7**: Code review and refactor
  - **REVIEW**: Turn classification correctness, edge case handling

- [ ] **M4.2.8**: Run quality checks
  - **RUN**: `uv run mypy src/neurospatial/behavioral.py`
  - **RUN**: `uv run ruff check src/neurospatial/behavioral.py`

- [ ] **M4.2.9**: Commit
  - **COMMIT**: `feat(behavioral): implement graph_turn_sequence for turn classification`

---

## Milestone 5: Documentation and Integration (Phase 5)

**Goal**: Update public API, documentation, and validate integration
**Estimated Time**: 2 hours
**Dependencies**: Milestone 4 complete

### M5.1: Update public API exports

- [ ] **M5.1.1**: Update `src/neurospatial/__init__.py`
  - Add imports for all 7 behavioral functions
  - Update `__all__` list
  - Verify alphabetical ordering

- [ ] **M5.1.2**: Write import tests
  - **FILE**: `tests/test_behavioral.py`
  - Test: `test_all_functions_exported()` (verify all 7 functions importable)

- [ ] **M5.1.3**: Run import tests
  - **RUN**: `uv run pytest tests/test_behavioral.py::test_all_functions_exported -v`

- [ ] **M5.1.4**: Commit
  - **COMMIT**: `feat(api): export behavioral analysis functions to public API`

### M5.2: Update CLAUDE.md documentation

- [ ] **M5.2.1**: Update Quick Reference section
  - Add "Behavioral & Goal-Directed Metrics (v0.8.0+)" section
  - Add usage examples for all 7 functions
  - Add common patterns and best practices

- [ ] **M5.2.2**: Update Table of Contents
  - Add link to new section

- [ ] **M5.2.3**: Commit
  - **COMMIT**: `docs(behavioral): update CLAUDE.md with v0.8.0 features`

### M5.3: Run comprehensive test suite

- [ ] **M5.3.1**: Run all behavioral tests
  - **RUN**: `uv run pytest tests/test_behavioral.py -v`
  - All tests should PASS

- [ ] **M5.3.2**: Run full test suite
  - **RUN**: `uv run pytest`
  - Ensure no regressions in other modules

- [ ] **M5.3.3**: Run with coverage
  - **RUN**: `uv run pytest --cov=src/neurospatial tests/test_behavioral.py`
  - Target: >95% coverage for behavioral.py

### M5.4: Run code quality checks

- [ ] **M5.4.1**: Run mypy on entire codebase
  - **RUN**: `uv run mypy src/neurospatial/`
  - Fix any new type errors

- [ ] **M5.4.2**: Run ruff checks
  - **RUN**: `uv run ruff check .`
  - Fix any linting issues

- [ ] **M5.4.3**: Run ruff format
  - **RUN**: `uv run ruff format .`
  - Auto-format code

- [ ] **M5.4.4**: Commit
  - **COMMIT**: `chore(behavioral): fix linting and type errors`

### M5.5: Final validation and release

- [ ] **M5.5.1**: Run verification-before-completion
  - Use skill: `verification-before-completion`
  - Verify all tests pass, no errors

- [ ] **M5.5.2**: Update TODO.md
  - Mark completed items (Section 2.4, Section 3.3 regressors, Section 5.1)
  - Document any deferred features

- [ ] **M5.5.3**: Create integration example
  - **FILE**: `examples/behavioral_analysis.py` (optional)
  - Show end-to-end workflow with all functions

- [ ] **M5.5.4**: Final commit
  - **COMMIT**: `feat(v0.8.0): complete goal-directed metrics and behavioral analysis`

- [ ] **M5.5.5**: Update TASKS.md
  - Mark all tasks complete
  - Update progress overview
  - Set status to "Complete"

---

## Testing Checklist

Before marking any milestone complete, verify:

- [ ] All tests for that milestone PASS
- [ ] No existing tests are broken (run `uv run pytest`)
- [ ] Mypy passes with no new errors
- [ ] Ruff check and format applied
- [ ] Docstrings complete with NumPy format
- [ ] Type annotations present on all functions
- [ ] TASKS.md updated with checkmarks

---

## Notes

- **TDD is mandatory**: Write tests BEFORE implementation for every function
- **Run tests frequently**: After each implementation step, verify tests pass
- **Commit frequently**: Small, focused commits with conventional commit messages
- **Document blockers**: Update SCRATCHPAD.md if you encounter issues
- **Ask for clarification**: Don't proceed with assumptions on unclear requirements

---

## Quick Commands Reference

```bash
# Run specific milestone tests
uv run pytest tests/test_behavioral.py::test_path_progress -v

# Run all behavioral tests
uv run pytest tests/test_behavioral.py -v

# Run with coverage
uv run pytest --cov=src/neurospatial tests/test_behavioral.py

# Check types
uv run mypy src/neurospatial/behavioral.py

# Lint and format
uv run ruff check . && uv run ruff format .

# Run full test suite
uv run pytest
```

---

## Success Criteria

- [x] All 30 subtasks completed
- [ ] All tests pass: `uv run pytest tests/test_behavioral.py -v`
- [ ] No regressions: `uv run pytest`
- [ ] >95% test coverage for behavioral.py
- [ ] Mypy passes with no errors
- [ ] All 7 functions fully documented with NumPy docstrings
- [ ] CLAUDE.md updated with examples
- [ ] Public API exports verified

---

**Last Updated**: 2025-11-24
