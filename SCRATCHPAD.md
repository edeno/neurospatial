# SCRATCHPAD - Package Reorganization

**Started**: 2025-12-05
**Current Status**: Milestone 2 in progress - transforms.py → ops/transforms.py DONE

---

## Session Log

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
