# Test Suite Refactoring Tasks

> **Reference**: See [PLAN.md](PLAN.md) for detailed context and implementation patterns.
>
> **Goal**: Reduce `Environment.from_samples()` calls from 1,011 → ~100 and improve test reliability.

---

## Milestone 1: Complete Fixture-Based Test Refactoring ✅ DONE

Phase 1 fixtures are complete. These tasks are for reference only.

- [x] Add `dense_rectangular_grid_env` fixture to `tests/conftest.py`
- [x] Add `dense_40x40_grid_env` fixture to `tests/conftest.py`
- [x] Add `spike_field_env_100` fixture to `tests/conftest.py`
- [x] Add `spike_field_trajectory` fixture to `tests/conftest.py`
- [x] Add `spike_field_env_random` fixture to `tests/conftest.py`
- [x] Add `minimal_2d_grid_env` fixture to `tests/conftest.py`
- [x] Add `minimal_20x20_grid_env` fixture to `tests/conftest.py`
- [x] Add `linear_track_1d_env` fixture to `tests/conftest.py`

---

## Milestone 2: Refactor Remaining High-Impact Test Files

### 2.1 Refactor `test_spike_field.py` ✅ DONE

- [x] Replace inline environment creations with `spike_field_env_100` fixture
- [x] Parametrize method variations (diffusion_kde, gaussian_kde, binned)
- [x] Reduce from 39 → 19 `from_samples()` calls (18 fixture references added)

### 2.2 Complete `test_boundary_cells.py` Refactoring ✅ DONE

**Before**: 21 inline calls, 5 fixture references
**After**: 7 inline calls, 19 fixture references (67% reduction)

- [x] Audit remaining `Environment.from_samples()` calls in `tests/metrics/test_boundary_cells.py`
- [x] Replace validation tests with `small_2d_env` fixture (9 tests)
- [x] Replace spatial tests with `dense_rectangular_grid_env` fixture (5 tests)
- [x] Refactor loop test to use single env with multiple random firing rates
- [x] Document why region coverage tests need inline environments (add regions → conflict in shared fixtures)
- [x] Verify tests pass: `uv run pytest tests/metrics/test_boundary_cells.py -v`

**Note**: 7 tests in TestComputeRegionCoverage remain inline because they add regions to environments.

### 2.3 Refactor `test_occupancy.py` ← **NEXT PRIORITY**

**Current**: 23 calls, 0 fixture usage
**Target**: ~5 calls

- [ ] Read `tests/environment/test_occupancy.py` and identify all `from_samples()` calls
- [ ] Replace standard 10x10 grids with `minimal_2d_grid_env` fixture
- [ ] Replace 20x20 grids with `minimal_20x20_grid_env` fixture
- [ ] Document any tests that legitimately need custom environments (add comment explaining why)
- [ ] Parametrize tests with similar bin_size variations
- [ ] Verify tests pass: `uv run pytest tests/environment/test_occupancy.py -v`

### 2.4 Refactor `test_transitions.py`

**Current**: 40 calls, 0 fixture usage
**Target**: ~5 calls

- [ ] Read `tests/environment/test_transitions.py` and identify all `from_samples()` calls
- [ ] Replace 1D track tests with `linear_track_1d_env` fixture
- [ ] Replace 2D grid tests with `minimal_2d_grid_env` or `minimal_20x20_grid_env`
- [ ] Parametrize bin sequence variations into combined tests
- [ ] Document any tests requiring custom environments
- [ ] Verify tests pass: `uv run pytest tests/environment/test_transitions.py -v`

---

## Milestone 3: Migrate Global RNG to Local RNG

**Problem**: 84 tests use `np.random.seed()` causing flaky parallel execution.
**Target**: 0 global RNG usages

### 3.1 High-Priority Files (Most Likely to Cause Flaky Tests)

- [ ] Migrate `tests/metrics/test_place_fields.py` (20 occurrences)
  - Replace `np.random.seed(42)` → `rng = np.random.default_rng(42)`
  - Replace `np.random.randn(n, m)` → `rng.standard_normal((n, m))`
  - Replace `np.random.rand(n, m)` → `rng.random((n, m))`
  - Verify: `uv run pytest tests/metrics/test_place_fields.py -v`

- [ ] Migrate `tests/environment/test_occupancy.py` (8 occurrences)
  - Apply same RNG migration pattern
  - Verify: `uv run pytest tests/environment/test_occupancy.py -v`

- [ ] Migrate `tests/test_transforms_3d.py` (2 occurrences)
  - Apply same RNG migration pattern
  - Verify: `uv run pytest tests/test_transforms_3d.py -v`

### 3.2 Medium-Priority Files

- [ ] Migrate `tests/test_validation_new.py` (4 occurrences)
- [ ] Migrate `tests/environment/test_interpolate.py` (2 occurrences)
- [ ] Migrate `tests/environment/test_trajectory_metrics.py` (2 occurrences)
- [ ] Migrate `tests/test_io.py` (4 occurrences)

### 3.3 Low-Priority Files

- [ ] Migrate `tests/test_differential.py` (1 occurrence)
- [ ] Migrate `tests/test_transforms_new.py` (2 occurrences)
- [ ] Migrate `tests/test_behavioral.py` (2 occurrences)
- [ ] Migrate `tests/environment/test_transitions.py` (3 occurrences)
- [ ] Migrate `tests/metrics/test_grid_cells.py` (4 occurrences)
- [ ] Migrate `tests/metrics/test_trajectory.py` (3 occurrences)
- [ ] Migrate `tests/metrics/test_population.py` (2 occurrences)
- [ ] Migrate `tests/segmentation/*.py` (10 occurrences total)
- [ ] Migrate `tests/animation/*.py` (5 occurrences total)

### 3.4 Verification

- [ ] Run parallel test suite 5 times to verify no flaky tests:

  ```bash
  for i in {1..5}; do uv run pytest -n auto -q; done
  ```

---

## Milestone 4: Organize Animation Fixtures

### 4.1 Create Animation Conftest

- [ ] Create `tests/animation/conftest.py` file
- [ ] Move `sample_video` fixture from `tests/conftest.py` (lines 512-551)
- [ ] Move `sample_video_array` fixture from `tests/conftest.py` (lines 555-573)
- [ ] Move `sample_calibration` fixture from `tests/conftest.py` (lines 577-607)
- [ ] Move `linearized_env` fixture from `tests/conftest.py` (lines 616-648)
- [ ] Move `polygon_env` fixture from `tests/conftest.py` (lines 652-667)
- [ ] Move `masked_env` fixture from `tests/conftest.py` (lines 671-698)

### 4.2 Update Main Conftest

- [ ] Remove moved fixtures from `tests/conftest.py` (lines 507-699)
- [ ] Verify animation tests still pass: `uv run pytest tests/animation/ -v`

---

## Milestone 5: Fixture Deduplication

### 5.1 Remove Duplicate Fixtures

- [ ] Delete `simple_graph_for_layout` from `tests/environment/test_core.py:691` (keep conftest.py version)
- [ ] Audit `tests/simulation/conftest.py` for duplicates
- [ ] Remove `simple_2d_env` from `tests/simulation/conftest.py` if identical to main conftest
- [ ] Remove `rng` from `tests/simulation/conftest.py` if identical to main conftest

### 5.2 Centralize Reusable Fixtures

- [ ] Review local fixtures in `tests/environment/test_core.py` (17+ fixtures at lines 350-900)
- [ ] Move any general-purpose fixtures to `tests/conftest.py`
- [ ] Keep test-specific fixtures local (document decision)

---

## Milestone 6: Slow Test Audit

### 6.1 Run Timing Analysis

- [ ] Run `uv run pytest --durations=50 -v 2>&1 | head -100` and save output
- [ ] Identify tests taking >1 second that are NOT marked slow

### 6.2 Mark Unmarked Slow Tests

Look for these patterns and add `@pytest.mark.slow`:

- [ ] Tests with very large arrays (>100K elements)
- [ ] Tests with many repeated `Environment.from_samples()` calls
- [ ] Tests with nested loops creating positions
- [ ] Tests using `large_2d_env` or similar fixtures
- [ ] Animation tests rendering many frames
- [ ] Tests with explicit `time.sleep()` or long timeouts

### 6.3 Verify Marker Configuration

- [ ] Confirm `pyproject.toml` has slow marker defined:

  ```toml
  [tool.pytest.ini_options]
  markers = ["slow: marks tests as slow (deselect with '-m \"not slow\"')"]
  ```

---

## Milestone 7: Standardize Fixture Naming (Optional)

### 7.1 Audit Current Names

- [ ] List all fixtures in `tests/conftest.py`
- [ ] Identify fixtures not following naming convention

### 7.2 Apply Naming Convention

Pattern: `{size}_{dims}d_{connectivity}_{layout}_env`

- [ ] Rename fixtures to follow pattern (update all usages)
- [ ] Document naming convention in conftest.py header comment

---

## Validation Checklist

Run after each milestone:

- [ ] Full test suite passes: `uv run pytest`
- [ ] No regressions: all previously passing tests still pass
- [ ] Coverage maintained: `uv run pytest --cov=src/neurospatial`
- [ ] Linting passes: `uv run ruff check . && uv run ruff format .`

---

## Success Metrics

| Metric | Original | Current | Target |
|--------|----------|---------|--------|
| `Environment.from_samples()` calls | 1,013 | 1,011 | ~100 |
| Parametrized test groups | 6 | 37 | 50+ |
| Global RNG (`np.random.seed`) | 74 | 84 | 0 |
| Duplicated fixtures | 2+ | 2+ | 0 |
| Test execution time | baseline | TBD | -40-60% |
