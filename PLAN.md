# Test Suite Refactoring Plan

## Current Progress (Updated: 2025-11-25)

| Phase | Status | Details |
|-------|--------|---------|
| 1.1 Dense grid fixtures | ✅ DONE | `dense_rectangular_grid_env`, `dense_40x40_grid_env` added |
| 1.2 Spike field fixtures | ✅ DONE | `spike_field_env_100`, `spike_field_trajectory`, `spike_field_env_random` added |
| 1.3 Minimal grid fixtures | ✅ DONE | `minimal_2d_grid_env`, `minimal_20x20_grid_env`, `linear_track_1d_env` added |
| 2.1 test_spike_field.py | ✅ DONE | Reduced from 39 → 19 calls, using 18 fixture references |
| 2.2 test_boundary_cells.py | ✅ DONE | Reduced from 21 → 7 calls (67% reduction) |
| 2.3 test_occupancy.py | ✅ DONE | Reduced from 22 → 4 calls (82% reduction), RNG migrated |
| 2.4 test_transitions.py | ✅ DONE | Reduced from 44 → 9 calls (80% reduction) |
| 3. Parametrization | 🔶 IN PROGRESS | 37 parametrized groups (up from 6) |
| 4. Animation conftest | ❌ NOT STARTED | `tests/animation/conftest.py` not created |
| 5. Slow test markers | ✅ BASELINE | 71 tests marked (unchanged) |
| 6. Fixture naming | ❌ NOT STARTED | |
| 7. RNG migration | ❌ NOT STARTED | 84 occurrences of `np.random.seed` (was 74) |
| 8. Fixture deduplication | ❌ NOT STARTED | `simple_graph_for_layout` duplicated in conftest.py + test_core.py |
| 9. Documentation tests | ✅ OK | Flexible phrase matching approach is reasonable |
| 10. Slow test audit | ❌ NOT STARTED | Need to audit for unmarked slow patterns |

**Current Metrics:**

| Metric | Original | Current | Target |
|--------|----------|---------|--------|
| `Environment.from_samples()` calls | 1,013 | 993 | ~100 |
| Parametrized test groups | 6 | 37 | 50+ |
| Global RNG (`np.random.seed`) | 74 | 78 | 0 |
| Duplicated fixtures | 2+ | 2+ | 0 |

---

## Executive Summary

The neurospatial test suite has significant performance and maintainability issues:

| Metric | Current | Target |
|--------|---------|--------|
| `Environment.from_samples()` calls | **1,013** | ~100 |
| Tests marked `@pytest.mark.slow` | 71 | All slow tests marked |
| Parametrized test groups | 6 | 50+ |
| Fixture utilization | Low | High |

This plan will reduce test execution time by 40-60% and improve maintainability.

---

## Phase 1: Create New Shared Fixtures (Priority: HIGH)

### 1.1 Add Dense Grid Fixture for Boundary/Metrics Tests

**Problem**: `tests/metrics/test_boundary_cells.py` and `tests/metrics/test_place_fields.py` use nested loops to create dense grids:

```python
# Current pattern (250,000 iterations!)
positions_list = []
for x in np.linspace(0, 50, 500):
    for y in np.linspace(0, 50, 500):
        positions_list.append([x, y])
```

**Solution**: Add to `tests/conftest.py`:

```python
@pytest.fixture(scope="session")
def dense_rectangular_grid_env() -> Environment:
    """Dense 50x50 rectangular grid for boundary/place field tests.

    Uses vectorized meshgrid instead of nested loops for 100x faster creation.
    """
    x = np.linspace(0, 50, 500)
    y = np.linspace(0, 50, 500)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    return Environment.from_samples(positions, bin_size=4.0)


@pytest.fixture(scope="session")
def dense_40x40_grid_env() -> Environment:
    """Dense 40x40 rectangular grid for corner/edge tests."""
    x = np.linspace(0, 40, 400)
    y = np.linspace(0, 40, 400)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    return Environment.from_samples(positions, bin_size=4.0)
```

**Files to update**:
- `tests/metrics/test_boundary_cells.py` - Replace 8 instances of nested loop grid creation
- `tests/metrics/test_place_fields.py` - Replace similar patterns

### 1.2 Add Spike Field Test Fixtures

**Problem**: `tests/test_spike_field.py` creates nearly identical environments 39 times.

**Solution**: Add fixtures to `tests/conftest.py`:

```python
@pytest.fixture(scope="session")
def spike_field_env_100() -> Environment:
    """100x100 environment with bin_size=10 for spike field tests."""
    positions = np.column_stack([
        np.linspace(0, 100, 1000),
        np.linspace(0, 100, 1000)
    ])
    return Environment.from_samples(positions, bin_size=10.0)


@pytest.fixture(scope="session")
def spike_field_trajectory() -> tuple[np.ndarray, np.ndarray]:
    """Standard trajectory for spike field tests (1000 points, 10 seconds)."""
    positions = np.column_stack([
        np.linspace(0, 100, 1000),
        np.linspace(0, 100, 1000)
    ])
    times = np.linspace(0, 10, 1000)
    return times, positions
```

**Files to update**:
- `tests/test_spike_field.py` - Replace 39 inline environment creations

### 1.3 Add Simple Grid Fixtures for Occupancy/Transitions

**Problem**: `tests/environment/test_occupancy.py` (23 times) and `tests/environment/test_transitions.py` (40 times) create tiny environments repeatedly.

**Solution**: Add to `tests/conftest.py`:

```python
@pytest.fixture(scope="session")
def minimal_2d_grid_env() -> Environment:
    """Minimal 10x10 grid for quick validation tests."""
    data = np.array([[0, 0], [10, 10]])
    return Environment.from_samples(data, bin_size=2.0)


@pytest.fixture(scope="session")
def minimal_20x20_grid_env() -> Environment:
    """Small 20x20 grid for trajectory tests."""
    data = np.array([[0, 0], [20, 20]])
    return Environment.from_samples(data, bin_size=5.0)


@pytest.fixture(scope="session")
def linear_track_1d_env() -> Environment:
    """Simple 1D linear track (10 positions, bin_size=2.5)."""
    return Environment.from_samples(
        np.array([[i] for i in range(0, 11, 2)], dtype=float),
        bin_size=2.5,
    )
```

**Files to update**:
- `tests/environment/test_occupancy.py` - Replace 23 inline creations
- `tests/environment/test_transitions.py` - Replace 40 inline creations

---

## Phase 2: Refactor High-Impact Test Files (Priority: HIGH)

### 2.1 Refactor `tests/test_spike_field.py`

**Current state**: 39 `Environment.from_samples()` calls, most identical.

**Refactoring approach**:

1. **Group tests by environment requirements**:
   - Tests needing 100x100 environment → use `spike_field_env_100`
   - Tests needing custom environments → keep inline but document why

2. **Parametrize similar tests**:
   ```python
   # Before: 3 separate test methods
   def test_empty_spikes_diffusion(self): ...
   def test_empty_spikes_gaussian(self): ...
   def test_empty_spikes_binned(self): ...

   # After: 1 parametrized test
   @pytest.mark.parametrize("method", ["diffusion_kde", "gaussian_kde", "binned"])
   def test_empty_spikes(self, method, spike_field_env_100, spike_field_trajectory): ...
   ```

3. **Extract shared setup to fixtures**:
   ```python
   @pytest.fixture
   def spike_field_test_data(spike_field_env_100, spike_field_trajectory):
       """Common test data for spike field tests."""
       times, positions = spike_field_trajectory
       rng = np.random.default_rng(42)
       spike_times = rng.uniform(0, 10, 25)
       return spike_field_env_100, spike_times, times, positions
   ```

**Expected reduction**: 39 → ~5 environment creations

### 2.2 Refactor `tests/metrics/test_boundary_cells.py`

**Current state**: 26 `Environment.from_samples()` calls with nested loops.

**Refactoring approach**:

1. **Use dense grid fixtures** for all tests in `TestBorderScore`:
   ```python
   class TestBorderScore:
       def test_border_score_perfect_border_cell(self, dense_rectangular_grid_env):
           env = dense_rectangular_grid_env
           # ... rest of test
   ```

2. **Create field generation helpers** (not fixtures, just functions):
   ```python
   def create_border_field(env: Environment, wall: str = "left", threshold: float = 10.0):
       """Create firing rate field along specified wall."""
       firing_rate = np.zeros(env.n_bins)
       for i in range(env.n_bins):
           center = env.bin_centers[i]
           if wall == "left" and center[0] < threshold:
               firing_rate[i] = 5.0
           # ... other walls
       return firing_rate
   ```

3. **Parametrize wall direction tests**:
   ```python
   @pytest.mark.parametrize("wall,expected_min_score", [
       ("left", 0.5),
       ("right", 0.5),
       ("top", 0.5),
       ("bottom", 0.5),
   ])
   def test_border_score_walls(self, dense_rectangular_grid_env, wall, expected_min_score): ...
   ```

**Expected reduction**: 26 → ~3 environment creations

### 2.3 Refactor `tests/environment/test_occupancy.py`

**Current state**: 23 `Environment.from_samples()` calls.

**Refactoring approach**:

1. **Use shared fixtures** for standard environments
2. **Keep inline only when test requires specific setup** (document why)

**Pattern to apply**:
```python
# Before
def test_occupancy_simple_stationary(self):
    data = np.array([[0, 0], [10, 10]])
    env = Environment.from_samples(data, bin_size=5.0)
    # ...

# After
def test_occupancy_simple_stationary(self, minimal_2d_grid_env):
    env = minimal_2d_grid_env  # Or create specific fixture if needed
    # ...
```

**Expected reduction**: 23 → ~5 environment creations

### 2.4 Refactor `tests/environment/test_transitions.py`

**Current state**: 40 `Environment.from_samples()` calls.

**Refactoring approach**:

1. **Use `linear_track_1d_env`** for most 1D tests
2. **Create 2D grid fixture** for 2D transition tests
3. **Parametrize bin sequence variations** instead of separate tests

**Expected reduction**: 40 → ~5 environment creations

---

## Phase 3: Parametrize Test Variations (Priority: MEDIUM)

### 3.1 Identify Parametrization Candidates

Tests that differ only in input values should be parametrized:

| File | Pattern | Parametrize By |
|------|---------|----------------|
| `test_spike_field.py` | Same test, different methods | `method` |
| `test_boundary_cells.py` | Same test, different walls | `wall` |
| `test_occupancy.py` | Same test, different bin_sizes | `bin_size` |
| `test_transitions.py` | Same test, different sequence lengths | `sequence` |
| `test_place_fields.py` | Same test, different thresholds | `threshold` |

### 3.2 Template for Parametrization

```python
# Pattern 1: Simple value parametrization
@pytest.mark.parametrize("bin_size", [2.0, 5.0, 10.0])
def test_occupancy_bin_sizes(self, bin_size, minimal_2d_grid_env):
    # Test logic shared across all bin sizes
    ...

# Pattern 2: Multiple parameter combinations
@pytest.mark.parametrize("method,bandwidth", [
    ("diffusion_kde", 5.0),
    ("gaussian_kde", 8.0),
    ("binned", 5.0),
])
def test_place_field_methods(self, method, bandwidth, spike_field_env_100):
    ...

# Pattern 3: ID-based parametrization for clarity
@pytest.mark.parametrize("wall,expected", [
    pytest.param("left", 0.5, id="left_wall"),
    pytest.param("right", 0.5, id="right_wall"),
    pytest.param("center", 0.1, id="central_field"),
])
def test_border_score_locations(self, wall, expected, dense_rectangular_grid_env):
    ...
```

---

## Phase 4: Organize Animation Fixtures (Priority: MEDIUM)

### 4.1 Move Video Fixtures to Animation Conftest

**Current state**: Video/animation fixtures mixed in main `tests/conftest.py` (lines 507-699).

**Solution**: Create `tests/animation/conftest.py`:

```python
"""Shared fixtures for animation tests."""

import numpy as np
import pytest
from pathlib import Path

from neurospatial import Environment


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Create a sample test video (16x16 pixels, 10 frames)."""
    # Move from tests/conftest.py lines 512-551
    ...


@pytest.fixture
def sample_video_array() -> np.ndarray:
    """Create a sample video as numpy array."""
    # Move from tests/conftest.py lines 555-573
    ...


@pytest.fixture
def sample_calibration():
    """Create sample VideoCalibration for testing."""
    # Move from tests/conftest.py lines 577-607
    ...


@pytest.fixture(scope="session")
def linearized_env() -> Environment:
    """1D linearized track for video overlay rejection tests."""
    # Move from tests/conftest.py lines 616-648
    ...


@pytest.fixture(scope="session")
def polygon_env() -> Environment:
    """2D polygon environment for video overlay fallback tests."""
    # Move from tests/conftest.py lines 652-667
    ...


@pytest.fixture(scope="session")
def masked_env() -> Environment:
    """2D masked grid for full video overlay support tests."""
    # Move from tests/conftest.py lines 671-698
    ...
```

**Update main conftest.py**: Remove lines 507-699 after moving.

---

## Phase 5: Mark Slow Tests (Priority: MEDIUM)

### 5.1 Audit Slow Tests

Run timing analysis to identify slow tests:
```bash
uv run pytest --durations=50 -v 2>&1 | head -100
```

### 5.2 Mark Tests Using Large Environments

Any test using `large_2d_env` or similar should be marked:

```python
@pytest.mark.slow
def test_performance_large_environment(self, large_2d_env):
    ...
```

### 5.3 Update pytest.ini or pyproject.toml

Ensure marker configuration exists:
```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

---

## Phase 6: Standardize Fixture Naming (Priority: LOW)

### 6.1 Naming Convention

Adopt consistent naming pattern:
```
{size}_{dims}d_{connectivity}_{layout}_env
```

Examples:
- `small_2d_ortho_grid_env` - Small 2D grid with orthogonal connectivity
- `medium_2d_diag_grid_env` - Medium 2D grid with diagonal connectivity
- `large_1d_linear_track_env` - Large 1D linear track

### 6.2 Fixture Categories

| Category | Naming Pattern | Scope |
|----------|----------------|-------|
| Size-based grids | `{small,medium,large}_{n}d_env` | session |
| Layout-specific | `{graph,hex,polygon}_env` | session |
| Test-specific | `{test_module}_env` | function |
| Trajectory data | `{name}_trajectory` | session |

---

## Implementation Order

### Week 1: High-Impact Fixtures
- [x] Add dense grid fixtures to `tests/conftest.py`
- [x] Add spike field fixtures to `tests/conftest.py`
- [x] Add minimal grid fixtures to `tests/conftest.py`
- [x] Update `tests/metrics/test_boundary_cells.py` to use new fixtures (partial - 4 usages)
- [x] Run tests to verify no regressions

### Week 2: Major Refactors

- [x] Refactor `tests/test_spike_field.py` to use fixtures (39 → 19 calls)
- [ ] Refactor `tests/environment/test_occupancy.py` to use fixtures ← **NEXT**
- [ ] Refactor `tests/environment/test_transitions.py` to use fixtures
- [x] Add parametrization to consolidated tests (6 → 37 groups)
- [ ] Run full test suite and benchmark

### Week 3: Organization
- [ ] Create `tests/animation/conftest.py`
- [ ] Move video fixtures from main conftest
- [ ] Audit and mark slow tests
- [ ] Standardize fixture naming
- [ ] Update documentation

---

## Expected Outcomes

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| `Environment.from_samples()` calls | 1,013 | ~100 | -90% |
| Test execution time | X min | ~0.4-0.6X min | -40-60% |
| Parametrized test groups | 6 | 50+ | +800% |
| Test maintainability | Low | High | Significant |

---

## Validation Steps

After each phase:

1. **Run full test suite**: `uv run pytest`
2. **Check for regressions**: All tests should pass
3. **Benchmark performance**: `uv run pytest --durations=0`
4. **Verify coverage**: `uv run pytest --cov=src/neurospatial`

---

## Notes

### Tests That Should Keep Inline Environments

Some tests legitimately need custom environments:

- Tests validating specific bin_size behavior
- Tests validating error handling for edge cases
- Tests that modify environment state (rare)

Document these exceptions with comments:
```python
def test_specific_bin_size_behavior(self):
    # Requires specific bin_size=3.7 for this edge case test
    env = Environment.from_samples(positions, bin_size=3.7)
    ...
```

### Simulation Conftest Duplication

Note: `tests/simulation/conftest.py` duplicates some fixtures from main conftest (`simple_2d_env`, `rng`). Consider removing duplicates and importing from main conftest.

### Redundant Tests to Consider Removing

During refactoring, look for truly redundant tests that test the same behavior:
- Multiple tests asserting the same property with different data
- Tests that duplicate existing parametrized tests

---

## Phase 7: Migrate Global RNG to Local RNG (Priority: HIGH for Reliability)

### 7.1 Problem: Flaky Tests Due to Global Random State

**Root Cause**: 74 tests use `np.random.seed(42)` (global random state), which causes race conditions during parallel execution with pytest-xdist.

**Symptom**: Intermittent test failures like `test_apply_transform_affine_nd_3d` that pass when run individually but occasionally fail in parallel.

**Evidence**:
```bash
# Found 74 instances of global RNG usage
$ grep -r "np.random.seed" tests/ | wc -l
74
```

### 7.2 Files Requiring Migration

| File | Count | Priority |
|------|-------|----------|
| `tests/metrics/test_place_fields.py` | 20 | HIGH |
| `tests/environment/test_occupancy.py` | 8 | HIGH |
| `tests/test_transforms_3d.py` | 2 | HIGH |
| `tests/test_validation_new.py` | 4 | MEDIUM |
| `tests/environment/test_interpolate.py` | 2 | MEDIUM |
| `tests/environment/test_trajectory_metrics.py` | 2 | MEDIUM |
| `tests/test_io.py` | 4 | MEDIUM |
| `tests/test_differential.py` | 1 | LOW |
| `tests/test_transforms_new.py` | 2 | LOW |
| `tests/test_behavioral.py` | 2 | LOW |
| `tests/environment/test_transitions.py` | 3 | LOW |
| `tests/metrics/test_grid_cells.py` | 4 | LOW |
| `tests/metrics/test_trajectory.py` | 3 | LOW |
| `tests/metrics/test_population.py` | 2 | LOW |
| `tests/segmentation/*.py` | 10 | LOW |
| `tests/animation/*.py` | 5 | LOW |
| Others | Various | LOW |

### 7.3 Migration Pattern

**Before (unsafe for parallel execution)**:
```python
def test_something(self):
    np.random.seed(42)  # GLOBAL state - race condition risk!
    data = np.random.randn(100, 2)
    # ...
```

**After (safe for parallel execution)**:
```python
def test_something(self):
    rng = np.random.default_rng(42)  # LOCAL state - isolated per test
    data = rng.standard_normal((100, 2))  # Note: different method name
    # ...
```

### 7.4 Method Name Changes

| Global (`np.random.*`) | Local (`rng.*`) |
|------------------------|-----------------|
| `np.random.randn(n, m)` | `rng.standard_normal((n, m))` |
| `np.random.rand(n, m)` | `rng.random((n, m))` |
| `np.random.choice(...)` | `rng.choice(...)` |
| `np.random.shuffle(x)` | `rng.shuffle(x)` |
| `np.random.permutation(n)` | `rng.permutation(n)` |
| `np.random.uniform(a, b, n)` | `rng.uniform(a, b, n)` |
| `np.random.normal(...)` | `rng.normal(...)` |

### 7.5 Implementation Steps

1. **Start with high-priority files** (most likely to cause flaky tests):
   - `tests/metrics/test_place_fields.py`
   - `tests/environment/test_occupancy.py`
   - `tests/test_transforms_3d.py`

2. **For each file**:
   ```bash
   # Find all occurrences
   grep -n "np.random.seed" tests/path/to/file.py

   # Replace pattern
   # np.random.seed(42) → rng = np.random.default_rng(42)
   # np.random.randn(...) → rng.standard_normal(...)
   # etc.
   ```

3. **Verify no regressions**:
   ```bash
   # Run file multiple times in parallel
   for i in {1..5}; do uv run pytest tests/path/to/file.py -n 10 -q; done
   ```

### 7.6 Fixture-Based RNG (Recommended for New Tests)

Add to `tests/conftest.py`:
```python
@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests.

    Function-scoped (recreated for each test) to ensure isolation.
    """
    return np.random.default_rng(42)
```

Usage:
```python
def test_something(self, rng):
    data = rng.standard_normal((100, 2))
    # ...
```

---

## Summary of Key Issues Found

1. **1,013 direct `Environment.from_samples()` calls** - Massive duplication
2. **Nested-loop grid generation** (O(n²)) - 8 tests in boundary_cells.py alone
3. **6 parametrized test groups** - Far too few given similar test patterns
4. **Animation fixtures in main conftest** - Poor organization
5. **71 slow-marked tests** - May be incomplete coverage
6. **Simulation conftest duplication** - Creates unnecessary environments
7. **74 tests using global RNG** - Causes flaky tests in parallel execution

---

## Phase 8: Fixture Deduplication (Priority: MEDIUM)

### 8.1 Problem: Duplicated Fixtures Across Files

Fixtures are defined in multiple locations, increasing maintenance cost and risk of drift:

| Fixture | Location 1 | Location 2 |
|---------|------------|------------|
| `simple_graph_for_layout` | `tests/conftest.py:162` | `tests/environment/test_core.py:691` |
| `simple_2d_env` | `tests/conftest.py` (implied) | `tests/simulation/conftest.py:10` |

### 8.2 Additional Local Fixtures in test_core.py

`tests/environment/test_core.py` defines **17+ local fixtures** (lines 350-900):

- `data_for_morpho_ops`
- `env_hexagonal`
- `env_with_disconnected_regions`
- `env_no_active_bins`
- `simple_graph_for_layout` (DUPLICATE)
- `simple_hex_env`
- `simple_graph_env`
- `env_all_active_2x2`
- `env_center_hole_3x3`
- `env_hollow_square_4x4`
- `env_line_1x3_in_3x3_grid`
- `env_single_active_cell_3x3`
- `env_no_active_cells_nd_mask`
- `env_1d_grid_3bins`
- `env_path_graph_3nodes`

### 8.3 Solution

1. **Remove duplicates**: Delete `simple_graph_for_layout` from `test_core.py` (use conftest.py version)
2. **Audit simulation/conftest.py**: Remove `simple_2d_env` if identical to main conftest
3. **Centralize reusable fixtures**: Move general-purpose fixtures to `tests/conftest.py`
4. **Keep test-specific fixtures local**: Only fixtures used by a single test file should stay local

---

## Phase 9: Documentation Tests Review (Priority: LOW) ✅ DONE

### 9.1 Resolution

`tests/test_common_pitfalls.py` has been **DELETED** as part of Milestone 8.

**Rationale**: Docstring-based tests are better handled by:

- Documentation linters (sphinx, pydocstyle)
- Manual code review
- Static analysis tools

These provide better coverage without adding fragile runtime test dependencies on documentation content.

### 9.2 Documentation Quality

Documentation quality is now ensured through:

- NumPy docstring format enforcement via ruff
- Comprehensive tests/README.md guidelines
- Code review process

---

## Phase 10: Slow Test Audit (Priority: MEDIUM)

### 10.1 Current Coverage

71 tests marked with `@pytest.mark.slow` across 18 files.

### 10.2 Audit Checklist

Look for unmarked slow patterns:

- [ ] Tests with very large arrays (>100K elements)
- [ ] Tests with many repeated `Environment.from_samples()` calls
- [ ] Tests with nested loops creating positions
- [ ] Tests using `large_2d_env` or similar fixtures
- [ ] Animation tests rendering many frames
- [ ] Tests with explicit `time.sleep()` or long timeouts

### 10.3 Command to Find Candidates

```bash
# Find tests with large array creation
uv run pytest --collect-only -q 2>/dev/null | xargs -I {} sh -c 'grep -l "linspace.*1000\|random.*1000\|zeros.*1000" {} 2>/dev/null'

# Run timing analysis
uv run pytest --durations=50 -v 2>&1 | head -100
```
