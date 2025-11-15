# Implementation Tasks for v0.3.0

**Based on**: PLAN.md and COMPREHENSIVE_CODE_REVIEW.md
**Target Release**: v0.3.0
**Status**: Planning

---

## Progress Overview

- [x] **Milestone 1**: Test Coverage Audit (Phase 1) - 4/4 tasks ✅ **COMPLETE**
- [ ] **Milestone 2**: scipy Integration (Phase 2) - 10/13 tasks (2 skipped)
- [ ] **Milestone 3**: API Simplification (Phase 3) - 0/13 tasks
- [ ] **Milestone 4**: UX Improvements (Phase 4) - 0/24 tasks
- [ ] **Milestone 5**: Code Modernization (Phase 5) - 0/9 tasks (OPTIONAL)
- [ ] **Milestone 6**: Documentation & Release (Phase 6) - 0/18 tasks

**Total Tasks**: 97 (88 required + 9 optional)

---

## Milestone 1: Test Coverage Audit (Phase 1)

**Goal**: Verify ≥95% test coverage before refactoring
**Priority**: 🔴 CRITICAL
**Estimated Time**: 1-2 days
**Blocks**: Milestone 2

### 1.1 Coverage Audit: distance.py ✅ COMPLETE

- [x] Run coverage report for distance.py ✅ (Final: **100% coverage**)

  ```bash
  uv run pytest tests/test_distance*.py --cov --cov-report=html --cov-report=term-missing
  ```

- [x] Open coverage report and identify gaps ✅ (documented in SCRATCHPAD.md)
- [x] Verify coverage for `geodesic_distance_matrix()` ≥95% ✅
- [x] Verify coverage for `euclidean_distance_matrix()` ≥95% ✅
- [x] Verify coverage for `distance_field()` - geodesic metric ≥95% ✅
- [x] Verify coverage for `distance_field()` - euclidean metric ≥95% ✅ (added many-sources tests)
- [x] Verify coverage for `distance_field()` - with cutoff parameter ≥95% ✅
- [x] Verify coverage for `pairwise_distances()` ≥95% ✅
- [x] Verify coverage for `neighbors_within()` ≥95% ✅ (added 20 comprehensive tests)

**Acceptance**: All functions in distance.py have ≥95% line coverage ✅ **EXCEEDED: 100% coverage**

**Tests Added**: 23 new tests (97 → 120 total)

- `neighbors_within()`: 20 tests (geodesic, euclidean, validation, edge cases)
- `distance_field()` many sources: 3 tests (broadcasted pairwise path)

### 1.2 Coverage Audit: differential.py ✅ COMPLETE

- [x] Run coverage report for differential.py ✅ (Final: **100% coverage**)
- [x] Verify coverage for `compute_differential_operator()` ≥95% ✅ (100%)
- [x] Verify coverage for `gradient()` on 1D, 2D, 3D environments ≥95% ✅ (100%)
- [x] Verify coverage for `divergence()` ≥95% ✅ (100%)

**Solution**: Removed 4 lines of unreachable dead code

- `sparse.issparse()` branches in `gradient()` and `divergence()` could never execute
- Root cause: `sparse_matrix @ dense_array` always returns dense array in scipy
- Code simplified from 49 to 45 statements

**Acceptance**: All functions in differential.py have ≥95% line coverage ✅ **EXCEEDED: 100% coverage**

### 1.3 Coverage Audit: kernels.py ✅ COMPLETE

- [x] Run coverage report for kernels.py ✅ (Final: **100% coverage**)
- [x] Verify coverage for `compute_diffusion_kernels()` - transition mode ≥95% ✅ (100%)
- [x] Verify coverage for `compute_diffusion_kernels()` - density mode ≥95% ✅ (100%)
- [x] Verify kernel caching mechanism tested ≥95% ✅ (100%)
- [x] Added comprehensive tests for `apply_kernel()` function ✅ (was 0% coverage)

**Tests Added**: 13 new tests (22 → 34 total)

- `TestComputeDiffusionKernelsValidation` (1 test): Invalid mode validation
- `TestApplyKernel` (12 tests): Forward/adjoint modes, validation, mathematical properties

**Acceptance**: All functions in kernels.py have ≥95% line coverage ✅ **EXCEEDED: 100% coverage**

### 1.4 Coverage Audit: place_fields.py

- [x] Run coverage report for place_fields.py

  ```bash
  uv run pytest tests/metrics/ --cov=src/neurospatial/metrics/place_fields.py --cov-report=html
  ```

- [x] Verify coverage for `detect_place_fields()` - basic detection ≥95%
- [x] Verify coverage for `detect_place_fields()` - with subfields ≥95%
- [x] Verify coverage for `detect_place_fields()` - interneuron exclusion ≥95%

**Results**:

- Initial coverage: 84% (339 statements, 53 missed)
- Final coverage: 95% (339 statements, 18 missed)
- Tests added: 31 new edge case tests (70 → 101 tests total)
- Test classes added: 12 new test classes for validation and edge cases
- Key functions tested: `selectivity()`, `in_out_field_ratio()`, `information_per_second()`, `spatial_coverage_single_cell()`, `field_shape_metrics()`, `field_shift_distance()`, `compute_field_emd()`

**New Test Classes**:

- `TestDetectPlaceFieldsValidation` (5 tests): Shape validation, threshold validation, NaN handling
- `TestFieldCentroidEdgeCases` (1 test): Zero firing rate edge case
- `TestSkaggsInformationEdgeCases` (2 tests): Zero/NaN mean rate cases
- `TestSparsityEdgeCases` (2 tests): Zero/NaN denominator cases
- `TestFieldStabilityEdgeCases` (3 tests): Insufficient points, all NaN, invalid method
- `TestRateMapCoherenceEdgeCases` (5 tests): Shape validation, NaN handling, zero variance
- `TestSelectivityEdgeCases` (1 test): High selectivity edge case
- `TestInOutFieldRatioEdgeCases` (4 tests): Entire environment, no valid bins, zero out-field rate
- `TestInformationPerSecondEdgeCases` (1 test): No valid pairs
- `TestSpatialCoverageSingleCellEdgeCases` (1 test): All NaN input
- `TestFieldShapeMetricsEdgeCases` (2 tests): Empty field, all NaN rates
- `TestFieldShiftDistanceEdgeCases` (2 tests): NaN centroid, incompatible environments
- `TestComputeFieldEMDEdgeCases` (1 test): Both distributions zero

**Acceptance**: detect_place_fields() has ≥95% line coverage ✅ **ACHIEVED: 95% coverage**

### 1.5 Add Missing Tests (if gaps found)

- [ ] If gaps exist: Create `tests/test_distance_regression.py` with pinned values
- [ ] If gaps exist: Create `tests/test_differential_regression.py` with eigenvalue tests
- [ ] If gaps exist: Create `tests/metrics/test_place_fields_regression.py` with synthetic fields
- [ ] If gaps exist: Add property-based tests for distance matrix symmetry
- [ ] If gaps exist: Add property-based tests for Laplacian properties

**Acceptance**: All coverage gaps filled, ≥95% coverage achieved

---

## Milestone 2: scipy Integration (Phase 2)

**Goal**: Replace custom implementations with scipy for 10-100× speedup
**Priority**: 🔴 HIGH
**Estimated Time**: 3-4 days
**Dependencies**: Milestone 1 complete

### 2.1 Replace geodesic_distance_matrix - Investigation ✅ COMPLETE

- [x] Read current implementation in `src/neurospatial/distance.py:32-62` ✅
- [x] Read scipy.sparse.csgraph.shortest_path documentation ✅
- [x] Create test script to compare outputs on small test graph ✅ (`investigate_scipy_shortest_path.py`)
- [x] Verify scipy produces identical results (within tolerance) ✅ (6/6 tests passed, max diff: 0.00e+00)
- [x] Document any differences in behavior ✅ (SCIPY_INVESTIGATION_2.1.md)

**Acceptance**: Confirmed scipy is compatible drop-in replacement ✅ **APPROVED**

**Results**:

- ✅ All tests passed (100% identical results)
- ✅ **13.75× performance improvement** on typical graphs (114 nodes, 385 edges)
- ✅ Zero behavioral differences
- ✅ No new dependencies (scipy already required)
- ✅ Drop-in replacement confirmed

### 2.2 Replace geodesic_distance_matrix - Implementation ✅ COMPLETE

- [x] Open `src/neurospatial/distance.py` ✅
- [x] Add import: `from scipy.sparse.csgraph import shortest_path` ✅
- [x] Replace function body with scipy implementation (see PLAN.md) ✅
- [x] Update docstring to mention scipy implementation ✅
- [x] Add "Performance" note about optimization ✅

**Acceptance**: Code replaced, docstring updated ✅

### 2.3 Replace geodesic_distance_matrix - Testing ✅ COMPLETE

- [x] Run existing tests: `uv run pytest tests/test_distance.py -v` ✅ (31/31 passed in 0.11s)
- [x] Verify all tests pass ✅ (54/54 geodesic tests passed)
- [x] Run full distance test suite: `uv run pytest tests/ -k geodesic` ✅
- [x] Create benchmark comparison (old vs new implementation) ✅ (`investigate_scipy_shortest_path.py`)
- [x] Document speedup in comments or benchmark file ✅ (SCIPY_INVESTIGATION_2.1.md)

**Acceptance**: All tests pass, ≥10× speedup documented ✅

**Results**:

- ✅ All 54 geodesic tests passed
- ✅ **13.75× speedup** documented (0.0509s → 0.0037s on 114-node graph)
- ✅ Fixed flaky test in place_fields.py
- ✅ Zero regressions

### 2.4 Replace Laplacian - Investigation ✅ COMPLETE - KEEP CURRENT

- [x] Read current implementation in `src/neurospatial/differential.py`
- [x] Read scipy.sparse.csgraph.laplacian documentation
- [x] Create test script comparing outputs on small graph
- [x] Check eigenvalue properties match
- [x] Verify gradient/divergence operators use Laplacian correctly
- [x] Document any sign/normalization differences
- [x] **BONUS**: Investigate NetworkX incidence_matrix as alternative

**Investigation Results**:

- ✅ scipy.sparse.csgraph.laplacian produces identical Laplacian matrices
- ✅ Eigenvalue properties preserved
- ✅ Gradient/divergence consistency verified
- ❌ **scipy only provides L, NOT differential operator D**
- ❌ **NetworkX incidence_matrix uses ±weight, NOT ±sqrt(weight)**

**Decision**: **KEEP CURRENT IMPLEMENTATION**

**Reason**:

- gradient() requires D.T (scalar → edge field)
- divergence() requires D (edge → scalar field)
- scipy only provides L = D @ D.T (scalar → scalar)
- NetworkX incidence_matrix uses wrong weighting (±weight vs ±sqrt(weight))
- No performance benefit, would add complexity

**Files Created**:

- `investigate_scipy_laplacian.py`: Full comparison (4 test environments)
- `test_networkx_incidence.py`: NetworkX incidence matrix test
- `test_networkx_incidence_weighted.py`: Non-uniform weight test
- `SCIPY_INVESTIGATION_2.4.md`: Complete investigation report

**Acceptance**: ✅ Investigation confirms current implementation is necessary

### 2.5 Replace Laplacian - Implementation ⏭️ SKIPPED

**Reason**: Task 2.4 investigation determined replacement is not beneficial

### 2.6 Replace Laplacian - Testing ⏭️ SKIPPED

**Reason**: Task 2.4 investigation determined replacement is not beneficial

**Acceptance**: All differential tests pass

### 2.7 Connected Components - Investigation ✅ COMPLETE

- [x] Read current `detect_place_fields()` implementation ✅
- [x] Identify connected component logic (flood-fill or graph traversal) ✅
- [x] Determine when scipy.ndimage.label is applicable ✅
- [x] Check if layout has `grid_shape` attribute (grid-based indicator) ✅
- [x] Design fast path (scipy) vs fallback path (current implementation) ✅

**Investigation Results**:

- ✅ scipy.ndimage.label: **6.16× faster** than current flood-fill (grid environments)
- ❌ NetworkX connected_components: **3.3× SLOWER** than current flood-fill
- ✅ Decision: TWO-PATH APPROACH (scipy for grids, keep current flood-fill for non-grid)

**Files Created**:

- `investigations/investigate_connected_components.py`: Performance comparison script
- `investigations/CONNECTED_COMPONENTS_INVESTIGATION_2.7.md`: Complete investigation report

**Acceptance**: Strategy documented for grid vs non-grid environments ✅ **APPROVED**

### 2.8 Connected Components - Implementation (Grid Path) ✅ COMPLETE

- [x] Create helper function `_extract_connected_component_scipy()` in place_fields.py ✅
- [x] Add import: `from scipy.ndimage import label` ✅
- [x] Implement grid-based detection using scipy.ndimage.label ✅
- [x] Handle reshaping: flat bins → grid → labeled → flat bins ✅
- [x] Handle active_mask if present ✅
- [x] Handle connectivity structure matching (diagonal vs axial) ✅

**Implementation**: Lines 206-287 in `src/neurospatial/metrics/place_fields.py`

**Acceptance**: Grid path implemented, compiles without errors ✅

### 2.9 Connected Components - Implementation (Fallback Path) ✅ COMPLETE

- [x] Extract current logic into `_extract_connected_component_graph()` helper ✅
- [x] Ensure it works for non-grid layouts (GraphLayout, irregular) ✅
- [x] Add type checking to ensure compatibility ✅

**Implementation**: Lines 290-337 in `src/neurospatial/metrics/place_fields.py`

**Acceptance**: Fallback path extracted and functional ✅

### 2.10 Connected Components - Integration ✅ COMPLETE

- [x] Modify `_extract_connected_component()` to check layout type ✅
- [x] Route to `_extract_connected_component_scipy()` if grid-based ✅
- [x] Route to `_extract_connected_component_graph()` otherwise ✅
- [x] Add docstring note about optimization ✅

**Implementation**: Lines 340-387 in `src/neurospatial/metrics/place_fields.py`

**Routing Condition**: `env.grid_shape is not None and len(env.grid_shape) >= 2 and env.active_mask is not None`

**Acceptance**: Routing logic implemented ✅

### 2.11 Connected Components - Testing ✅ COMPLETE

- [x] Run tests on RegularGridLayout: Should use scipy path ✅
- [x] Run tests on GraphLayout: Should use fallback path ✅
- [x] Verify identical results from both paths on same grid ✅
- [x] Run full place field test suite: `uv run pytest tests/metrics/test_place_fields.py -v` ✅
- [x] Add comprehensive test suite in `test_connected_component_paths.py` ✅

**Test Results**:

- ✅ All 111 tests passed (100 existing + 11 new)
- ✅ scipy path produces identical results to graph path
- ✅ Routing logic works correctly for grid vs non-grid environments
- ✅ Code-reviewer agent approved implementation

**New Test File**: `tests/metrics/test_connected_component_paths.py` (11 tests)

**Acceptance**: All tests pass, both paths working ✅

### 2.12 Connected Components - Benchmarking ✅ COMPLETE

- [x] Create benchmark for grid-based detection ✅
- [x] Compare old vs new implementation speed ✅
- [x] Document speedup (expect ≥5×) ✅
- [x] Add slow test marker: `@pytest.mark.slow` ✅

**Benchmark Results**:

- **Environment**: 6,308 bins, grid_shape=(101, 101)
- **Masked region**: 1,245 bins
- **Trials**: 10 repetitions per method
- **scipy path**: 0.122 ms ± 0.041 ms
- **graph path**: 0.752 ms ± 0.057 ms
- **Speedup**: **6.16× faster** (exceeds 5× target)

**Benchmarks**: Included in `tests/metrics/test_connected_component_paths.py`

**Acceptance**: ≥5× speedup on grid environments documented ✅ **EXCEEDED: 6.16× speedup**

### 2.13 Phase 2 Integration Testing

- [ ] Run full test suite: `uv run pytest -v`
- [ ] Verify no regressions in any tests
- [ ] Run doctests: `uv run pytest --doctest-modules src/neurospatial/`
- [ ] Check type hints: `uv run mypy src/neurospatial/`
- [ ] Check linting: `uv run ruff check .`

**Acceptance**: All tests and checks pass

---

## Milestone 3: API Simplification (Phase 3) ⏸️ SKIPPED

**Goal**: ~~Consolidate bin_at() to handle all point-to-bin mapping~~
**Decision**: **KEEP DUAL API** - Functions have distinct semantics
**Priority**: 🔴 HIGH → ✅ COMPLETE (Analysis)
**Status**: Analysis complete, consolidation not needed

### 3.1 API Analysis and Documentation ✅ COMPLETE

- [x] Read current `bin_at()` in `src/neurospatial/environment/queries.py`
- [x] Read current `map_points_to_bins()` in `src/neurospatial/spatial.py`
- [x] Analyzed semantic differences (geometric vs nearest-neighbor)
- [x] **Decision**: Keep both functions with distinct purposes
- [x] Updated docstrings to clarify when to use each
- [x] Added cross-references and performance notes

**Acceptance**: Documentation clarifies function purposes ✅

**Rationale**:
- `bin_at()`: Geometric containment (layout-specific, exact)
- `map_points_to_bins()`: Nearest-neighbor (KDTree, fast, cached)
- Different semantics serve different use cases
- Consolidation would lose important functionality

### 3.2 Implement Enhanced bin_at()

- [ ] Open `src/neurospatial/environment/queries.py`
- [ ] Modify `bin_at()` signature to accept all map_points_to_bins parameters
- [ ] Add logic to detect single vs batch input using `points.ndim`
- [ ] Delegate to `map_points_to_bins()` implementation
- [ ] Handle return type conversion (array[0] → int for single point)
- [ ] Update docstring with examples for both single and batch

**Acceptance**: bin_at() accepts both single and batch inputs

### 3.3 Update map_points_to_bins() with Deprecation

- [ ] Open `src/neurospatial/spatial.py`
- [ ] Add deprecation warning at top of `map_points_to_bins()`
- [ ] Warning message: "Use Environment.bin_at() instead. Will be removed in v0.4.0"
- [ ] Set `stacklevel=2` for correct warning location
- [ ] Ensure function still works (backward compatibility)

**Acceptance**: Deprecation warning added, function still works

### 3.4 Update Internal Uses of map_points_to_bins

- [ ] Search for internal uses: `uv run grep -r "map_points_to_bins" src/neurospatial/`
- [ ] Update each internal call to use `env.bin_at()` instead
- [ ] Verify no circular imports created
- [ ] Test affected modules

**Acceptance**: No internal code uses deprecated function

### 3.5 Testing - Single Point Input

- [ ] Create test for single point input (backward compatibility)
- [ ] Test return type is `int`, not array
- [ ] Test with return_dist=True (should return tuple of int, float)
- [ ] Add test to `tests/test_environment.py`

**Acceptance**: Single point tests pass

### 3.6 Testing - Batch Input

- [ ] Create test for batch input (new functionality)
- [ ] Test return type is array
- [ ] Test with various batch sizes (1, 10, 100 points)
- [ ] Test with return_dist=True (should return tuple of arrays)

**Acceptance**: Batch input tests pass

### 3.7 Testing - Edge Cases

- [ ] Test with empty array (0 points)
- [ ] Test with tie_break strategies
- [ ] Test with max_distance parameter
- [ ] Test with max_distance_factor parameter
- [ ] Test points outside environment (-1 return value)

**Acceptance**: Edge case tests pass

**Acceptance**: Deprecation warning test passes

### 3.9 Update Documentation Examples

- [ ] Search for `map_points_to_bins` in docstrings
- [ ] Replace with `env.bin_at()` in all examples
- [ ] Update module-level docstring in `__init__.py`
- [ ] Update CLAUDE.md quick reference

**Acceptance**: All docstring examples use bin_at()

### 3.12 Phase 3 Integration Testing

- [ ] Run full test suite: `uv run pytest -v`
- [ ] Verify deprecation warnings appear only for direct map_points_to_bins calls
- [ ] Verify no warnings for bin_at() usage
- [ ] Run doctests: `uv run pytest --doctest-modules src/neurospatial/`

**Acceptance**: All tests pass, deprecations work correctly

### 3.13 Phase 3 Documentation Review

- [ ] Review all updated docstrings for clarity
- [ ] Ensure examples are runnable
- [ ] Check that migration path is clear
- [ ] Verify CLAUDE.md is updated

**Acceptance**: Documentation is clear and accurate

---

## Milestone 4: UX Improvements (Phase 4)

**Goal**: Add inspection methods and modernize I/O
**Priority**: 🟡 MEDIUM
**Estimated Time**: 4-5 days
**Dependencies**: None (can run in parallel)

### 4.1 Add pathlib Support - Type Definitions

- [ ] Open `src/neurospatial/io.py`
- [ ] Add imports: `from pathlib import Path` and `from typing import Union`
- [ ] Create type alias: `PathLike = Union[str, Path]`
- [ ] Update all function signatures to use `PathLike`

**Acceptance**: Type hints updated

### 4.2 Add pathlib Support - Implementation

- [ ] Update `to_file()`: Convert `path` to `Path` object at start
- [ ] Update `from_file()`: Convert `path` to `Path` object at start
- [ ] Use `Path.with_suffix()` for file extensions
- [ ] Use `Path.exists()`, `Path.is_file()` for checks
- [ ] Update docstrings with `str | Path` parameter type

**Acceptance**: io.py functions use pathlib internally

### 4.3 Add pathlib Support - Environment Serialization

- [ ] Open `src/neurospatial/environment/serialization.py`
- [ ] Update `Environment.to_file()` with `PathLike` type
- [ ] Update `Environment.from_file()` with `PathLike` type
- [ ] Ensure delegation to `io.py` functions works

**Acceptance**: Environment I/O accepts pathlib

### 4.4 Add pathlib Support - Regions I/O

- [ ] Check if regions have I/O functions
- [ ] Update with `PathLike` type hints if exists
- [ ] Test region serialization with Path objects

**Acceptance**: All I/O functions support pathlib

### 4.5 Add pathlib Support - Testing

- [ ] Create test with `str` path (backward compatibility)
- [ ] Create test with `Path` object (new functionality)
- [ ] Test with relative paths
- [ ] Test with absolute paths
- [ ] Test with `Path.home()` / `Path.cwd()`

**Acceptance**: All pathlib tests pass

### 4.6 Custom Exception - Implementation

- [ ] Open `src/neurospatial/environment/decorators.py`
- [ ] Create `EnvironmentNotFittedError` class (see PLAN.md)
- [ ] Add helpful `__init__` with factory method suggestions
- [ ] Add docstring with examples
- [ ] Update `check_fitted` decorator to use new exception

**Acceptance**: Custom exception implemented

### 4.7 Custom Exception - Export

- [ ] Add to `src/neurospatial/environment/__init__.py`
- [ ] Add to `src/neurospatial/__init__.py` in `__all__`
- [ ] Update imports in test files if needed

**Acceptance**: Exception is publicly importable

### 4.8 Custom Exception - Testing

- [ ] Create test that triggers exception
- [ ] Verify exception message includes factory method names
- [ ] Verify exception is subclass of RuntimeError
- [ ] Test with pytest.raises()

**Acceptance**: Exception behavior tested

### 4.9 env.info() - Design

- [ ] Read PLAN.md implementation
- [ ] Decide which mixin to add to (core.py or metrics.py)
- [ ] List all properties to include in output
- [ ] Design format (multiline string with indentation)

**Acceptance**: Design documented

### 4.10 env.info() - Implementation

- [ ] Open target file (likely `src/neurospatial/environment/core.py`)
- [ ] Implement `info()` method (see PLAN.md for code)
- [ ] Handle edge cases (no units, no regions, no cache)
- [ ] Format numbers with thousands separators
- [ ] Add `@check_fitted` decorator

**Acceptance**: info() method implemented

### 4.11 env.info() - Testing

- [ ] Test on RegularGridLayout environment
- [ ] Test on GraphLayout environment
- [ ] Test with no regions
- [ ] Test with no units
- [ ] Test with cache built vs empty
- [ ] Verify output is readable and informative

**Acceptance**: info() tests pass

### 4.12 explain_connectivity() - Design

- [ ] Decide implementation location (metrics.py or new inspection.py mixin)
- [ ] List connectivity properties to explain
- [ ] Design output format

**Acceptance**: Design documented

### 4.13 explain_connectivity() - Implementation

- [ ] Implement method (see PLAN.md for code)
- [ ] Handle different layout types (grid, graph, irregular)
- [ ] Compute average degree
- [ ] Identify connectivity pattern (4-conn, 8-conn, etc.)
- [ ] Add example bin with neighbors
- [ ] Add `@check_fitted` decorator

**Acceptance**: explain_connectivity() implemented

### 4.14 explain_connectivity() - Testing

- [ ] Test on 4-connected grid
- [ ] Test on 8-connected grid (diagonal neighbors)
- [ ] Test on GraphLayout
- [ ] Test on irregular graph
- [ ] Verify output clarity

**Acceptance**: explain_connectivity() tests pass

### 4.15 Metric Quick-Reference Guide

- [ ] Open `src/neurospatial/metrics/__init__.py`
- [ ] Add comprehensive module docstring (see PLAN.md)
- [ ] Group metrics by cell type (Place, Grid, Boundary, Population, Trajectory)
- [ ] Add examples section
- [ ] Add references to validation

**Acceptance**: Quick-reference guide added

### 4.16 Add **slots** - Investigation

- [ ] Check current Python version requirement in `pyproject.toml`
- [ ] Verify Python 3.10+ (required for dataclass slots)
- [ ] Read about dataclass slots and mixin compatibility
- [ ] Check if any code assigns dynamic attributes to Environment

**Acceptance**: Compatibility confirmed

### 4.17 Add **slots** - Implementation

- [ ] Open `src/neurospatial/environment/core.py`
- [ ] Add `slots=True` to `@dataclass` decorator
- [ ] Ensure no conflicts with mixin pattern
- [ ] Update docstring if needed

**Acceptance**: **slots** added to dataclass

### 4.18 Add **slots** - Testing

- [ ] Run full test suite: `uv run pytest -v`
- [ ] Verify no AttributeError for valid attributes
- [ ] Test that dynamic attribute assignment raises error (expected)
- [ ] Check for any test failures due to slots

**Acceptance**: All tests pass with slots enabled

### 4.19 Add **slots** - Benchmarking

- [ ] Create memory benchmark script
- [ ] Measure memory before slots (create 1000 environments)
- [ ] Measure memory after slots (create 1000 environments)
- [ ] Calculate percentage reduction
- [ ] Document in comments or benchmark file

**Acceptance**: Memory reduction ≥30% documented

### 4.20 Phase 4 Integration Testing

- [ ] Run full test suite: `uv run pytest -v`
- [ ] Test all new methods (info, explain_connectivity)
- [ ] Test pathlib I/O with various file operations
- [ ] Verify custom exception works
- [ ] Check type hints: `uv run mypy src/neurospatial/`

**Acceptance**: All Phase 4 features working

### 4.21 Phase 4 Documentation

- [ ] Add `info()` to CLAUDE.md quick reference
- [ ] Add `explain_connectivity()` to CLAUDE.md
- [ ] Document pathlib support
- [ ] Document custom exception
- [ ] Update examples to show new features

**Acceptance**: Documentation complete

### 4.22 User Experience Testing

- [ ] Create fresh environment as if new user
- [ ] Try `env.info()` - is output helpful?
- [ ] Try `env.explain_connectivity()` - is it clear?
- [ ] Try pathlib paths - does it feel natural?
- [ ] Check metric quick-reference - easy to find metrics?

**Acceptance**: UX feels improved

### 4.23 Phase 4 Code Review

- [ ] Review all new code for clarity
- [ ] Check docstrings follow NumPy format
- [ ] Verify type hints are complete
- [ ] Run linter: `uv run ruff check .`
- [ ] Format code: `uv run ruff format .`

**Acceptance**: Code quality high

### 4.24 Phase 4 Final Testing

- [ ] Run tests with coverage: `uv run pytest --cov=src/neurospatial`
- [ ] Verify ≥95% coverage maintained
- [ ] Run doctests: `uv run pytest --doctest-modules src/neurospatial/`
- [ ] Check no new warnings

**Acceptance**: All quality checks pass

---

## Milestone 5: Code Modernization (Phase 5) - OPTIONAL

**Goal**: Apply modern Python idioms
**Priority**: 🟢 LOW
**Estimated Time**: 2-3 days
**Dependencies**: None

### 5.1 Positional-Only Parameters

- [ ] Identify functions where positional-only makes sense
- [ ] Add `/` separator in `field_ops.py` functions
- [ ] Add `/` separator in `distance.py` functions
- [ ] Update docstrings to reflect positional-only
- [ ] Test that keyword usage raises TypeError

**Acceptance**: Positional-only parameters added where appropriate

### 5.2 Match Statements - Identify Candidates

- [ ] Search for if-elif chains: `grep -n "elif.*==" src/neurospatial/*.py`
- [ ] Identify best candidates for match statements
- [ ] List files: `field_ops.py`, `spatial.py`, others

**Acceptance**: Candidates identified

### 5.3 Match Statements - Implementation

- [ ] Convert if-elif chain in `field_ops.py:kl_divergence()`
- [ ] Convert if-elif chain in other identified locations
- [ ] Ensure identical behavior
- [ ] Test all branches

**Acceptance**: Match statements implemented, tests pass

### 5.4 Context Manager - Implementation

- [ ] Add `temporary_cache_clear()` context manager to core.py
- [ ] Use `@contextmanager` decorator
- [ ] Save cache state on entry
- [ ] Clear on entry, restore on exit
- [ ] Add docstring with examples

**Acceptance**: Context manager implemented

### 5.5 Context Manager - Testing

- [ ] Test cache cleared inside context
- [ ] Test cache restored after context
- [ ] Test exception handling (finally block)
- [ ] Add example to docstring

**Acceptance**: Context manager tested

### 5.6 Phase 5 Integration Testing

- [ ] Run full test suite
- [ ] Verify all modernizations work
- [ ] Check no regressions

**Acceptance**: All tests pass

### 5.7 Phase 5 Documentation

- [ ] Document new context manager
- [ ] Update CLAUDE.md if needed
- [ ] Add examples

**Acceptance**: Documentation updated

### 5.8 Phase 5 Code Review

- [ ] Review modern Python usage
- [ ] Verify improvements are worthwhile
- [ ] Check readability

**Acceptance**: Code quality improved

### 5.9 Phase 5 Completion

- [ ] Decide if keeping all changes
- [ ] May defer some to future release
- [ ] Document decision

**Acceptance**: Phase 5 scope finalized

---

## Milestone 6: Documentation & Release (Phase 6)

**Goal**: Prepare for v0.3.0 release
**Priority**: 🔴 HIGH
**Estimated Time**: 2-3 days
**Dependencies**: Milestones 1-4 complete

### 6.1 Update CHANGELOG.md - Structure

- [ ] Open `CHANGELOG.md`
- [ ] Create `## [0.3.0] - 2025-XX-XX` section
- [ ] Add subsections: Added, Changed, Deprecated, Fixed
- [ ] Use template from PLAN.md

**Acceptance**: CHANGELOG structure ready

### 6.2 Update CHANGELOG.md - Added Section

- [ ] List all new features:
  - `Environment.info()`
  - `Environment.explain_connectivity()`
  - `EnvironmentNotFittedError`
  - pathlib.Path support
  - Metric quick-reference guide
  - **slots** for memory efficiency

**Acceptance**: Added section complete

### 6.3 Update CHANGELOG.md - Changed Section

- [ ] List breaking changes:
  - `bin_at()` handles single and batch
  - scipy.sparse.csgraph for distance matrix
  - scipy.sparse.csgraph for Laplacian
  - scipy.ndimage for place field detection
- [ ] Note performance improvements

**Acceptance**: Changed section complete

### 6.4 Update CHANGELOG.md - Deprecated Section

- [ ] Add deprecation notice for `map_points_to_bins()`
- [ ] Specify removal version (v0.4.0)
- [ ] Reference migration guide

**Acceptance**: Deprecated section complete

### 6.5 Update CHANGELOG.md - Migration Guide

- [ ] Write migration guide for `map_points_to_bins()` → `bin_at()`
- [ ] Show before/after code examples
- [ ] Explain single vs batch behavior
- [ ] Provide timeline

**Acceptance**: Migration guide clear

### 6.6 Update CLAUDE.md

- [ ] Add `env.info()` to quick reference
- [ ] Add `env.explain_connectivity()` to quick reference
- [ ] Update performance notes with scipy optimizations
- [ ] Document deprecation of `map_points_to_bins()`
- [ ] Add pathlib support to I/O section

**Acceptance**: CLAUDE.md updated

### 6.7 Update Module Docstrings

- [ ] Update `__init__.py` main docstring
- [ ] Update examples to use `bin_at()`
- [ ] Add new features to feature list
- [ ] Verify all examples are runnable

**Acceptance**: Module docstrings current

### 6.8 Update All Examples Using map_points_to_bins

- [ ] Search: `grep -r "map_points_to_bins" src/neurospatial/`
- [ ] Replace all docstring examples with `env.bin_at()`
- [ ] Verify examples are correct
- [ ] Run doctests to confirm

**Acceptance**: All examples use new API

### 6.9 Version Bump

- [ ] Open `pyproject.toml`
- [ ] Update version to `0.3.0`
- [ ] Update any version references in docs
- [ ] Verify version string consistent

**Acceptance**: Version bumped to 0.3.0

### 6.10 Run Full Test Suite

- [ ] Run all tests: `uv run pytest -v`
- [ ] Verify all pass
- [ ] Check for unexpected warnings
- [ ] Review any skipped tests

**Acceptance**: All tests pass

### 6.11 Run Coverage Report

- [ ] Run with coverage: `uv run pytest --cov=src/neurospatial --cov-report=html`
- [ ] Open `htmlcov/index.html`
- [ ] Verify ≥95% coverage maintained
- [ ] Identify any new gaps

**Acceptance**: Coverage ≥95%

### 6.12 Run Doctests

- [ ] Run: `uv run pytest --doctest-modules src/neurospatial/`
- [ ] Fix any failing doctests
- [ ] Verify all examples work

**Acceptance**: All doctests pass

### 6.13 Run Type Checking

- [ ] Run: `uv run mypy src/neurospatial/`
- [ ] Fix any type errors
- [ ] Verify no regressions

**Acceptance**: Type checking passes

### 6.14 Run Linting

- [ ] Run: `uv run ruff check .`
- [ ] Fix any linting errors
- [ ] Run: `uv run ruff format --check .`
- [ ] Format if needed: `uv run ruff format .`

**Acceptance**: Linting passes

### 6.15 Performance Benchmarks

- [ ] Create or update `tests/benchmarks/test_distance_performance.py`
- [ ] Create or update `tests/benchmarks/test_place_field_performance.py`
- [ ] Run benchmarks: `uv run pytest -m slow -v`
- [ ] Document performance improvements in CHANGELOG
- [ ] Include speedup numbers (10×, 5×, etc.)

**Acceptance**: Benchmarks run, improvements documented

### 6.16 Check for Deprecation Warnings

- [ ] Run tests with warnings as errors: `uv run pytest -W error::DeprecationWarning`
- [ ] Verify only `map_points_to_bins()` causes warnings
- [ ] Ensure internal code doesn't trigger warnings

**Acceptance**: No unexpected warnings

### 6.17 Pre-Release Checklist

- [ ] All tests passing ✓
- [ ] Documentation updated ✓
- [ ] CHANGELOG complete ✓
- [ ] Version bumped ✓
- [ ] No deprecation warnings in own code ✓
- [ ] Type checking passes ✓
- [ ] Linting passes ✓
- [ ] Coverage ≥95% ✓

**Acceptance**: All checklist items complete

### 6.18 Final Review

- [ ] Review CHANGELOG for completeness
- [ ] Review migration guide for clarity
- [ ] Review new feature documentation
- [ ] Spot-check critical functions
- [ ] Read through updated docstrings

**Acceptance**: Everything ready for release

---

## Post-Release Tasks

These tasks happen after v0.3.0 is released:

- [ ] Tag release in git: `git tag v0.3.0`
- [ ] Push tag: `git push origin v0.3.0`
- [ ] Monitor GitHub issues for migration problems
- [ ] Update FAQ based on user questions
- [ ] Collect performance benchmarks from users
- [ ] Plan v0.4.0 features (remove deprecated functions)

---

## Quick Start Guide

To begin working on these tasks:

1. **Start with Milestone 1** (Test Coverage)

   ```bash
   # Check current coverage
   uv run pytest tests/ --cov=src/neurospatial --cov-report=html
   open htmlcov/index.html
   ```

2. **After Milestone 1, choose your path**:
   - **Performance focus**: Start Milestone 2 (scipy integration)
   - **UX focus**: Start Milestone 3 or 4 (API or UX improvements)
   - **Can work in parallel**: Milestones 2, 3, 4 are independent

3. **Track progress**: Check off boxes as you complete tasks

4. **Before moving to next milestone**: Run integration tests

   ```bash
   uv run pytest -v
   uv run mypy src/neurospatial/
   uv run ruff check .
   ```

5. **Final step**: Complete Milestone 6 (Documentation & Release)

---

## Notes

- **Checkbox format**: `- [ ]` for incomplete, `- [x]` for complete
- **Estimated times** are guidelines, adjust based on actual progress
- **Optional tasks** (Milestone 5) can be deferred to v0.4.0
- **Parallel work**: Milestones 2, 3, 4 can be done concurrently
- **Testing**: Always run tests after completing a subsection
- **Commit often**: Commit after each completed subsection
- **Ask for help**: Reference PLAN.md and CLAUDE.md for details

---

**Last Updated**: 2025-11-15
**Version**: 1.0
