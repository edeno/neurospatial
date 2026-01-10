# Encoding Module Refactor - Scratchpad

## Current Status

**Date**: 2026-01-10
**Last Completed**: Task 8.2 (Enforce monotonic time validation in single-neuron binning helpers)
**Next Task**: Task 8.3 (Decide and codify JAX metric behavior)

### Task 8.2 Completion Notes (2026-01-10)

**Problem**: The single-neuron binning functions `bin_view_spike_train()` and
`bin_egocentric_spike_train()` did not validate that times were monotonically
increasing, while their batch counterparts (`bin_view_spike_trains()` and
`bin_egocentric_spike_trains()`) did. This inconsistency could lead to silent
incorrect results when unsorted times were passed to single-neuron functions.

**Solution**: Added `_validate_times()` calls to both single-neuron functions,
matching the behavior of their batch counterparts.

**Files Modified**:

- `src/neurospatial/encoding/_view_binning.py` (line 448: added validation call, lines 403-406: added Raises section to docstring)
- `src/neurospatial/encoding/_egocentric_binning.py` (line 653: added validation call, lines 617-619: added to Raises section)
- `tests/encoding/test_encoding_view_binning.py` (added `TestBinViewSpikeTrainValidation` class with 2 tests)
- `tests/encoding/test_encoding_egocentric_binning.py` (added `TestBinEgocentricSpikeTrainValidation` class with 2 tests)

**Tests Added**:

- `TestBinViewSpikeTrainValidation.test_unsorted_times_raises_error` - Unsorted times raise ValueError
- `TestBinViewSpikeTrainValidation.test_insufficient_samples_raises_error` - <2 samples raise ValueError
- `TestBinEgocentricSpikeTrainValidation.test_unsorted_times_raises_error` - Unsorted times raise ValueError
- `TestBinEgocentricSpikeTrainValidation.test_insufficient_samples_raises_error` - <2 samples raise ValueError

**Validation**:

- All 102 tests in view/egocentric binning test files pass
- mypy: no issues in 2 source files
- ruff: all checks passed, files already formatted

### Task 8.1 Completion Notes (2026-01-10)

**Problem**: When using geodesic distance metric with objects outside the environment,
`np.argmin` doesn't handle NaN values properly, returning arbitrary indices.

**Solution**: Replace `np.argmin` with NaN-aware approach:
1. Identify rows where all distances are NaN (no reachable objects)
2. Replace NaN with infinity so argmin prefers finite distances
3. Use regular argmin on the masked array
4. Restore NaN for all-NaN rows (both distance and bearing)

**Files Modified**:
- `src/neurospatial/encoding/_egocentric_binning.py` (lines 282-306)
- `tests/encoding/test_encoding_egocentric_binning.py` (added `TestEgocentricNaNHandling` class with 7 tests)

**Tests Added**:
- `test_object_outside_env_geodesic` - Object outside environment returns NaN
- `test_mixed_nan_finite_distances` - Mixed NaN/finite selects nearest valid
- `test_all_objects_outside_env_geodesic` - All objects outside returns NaN
- `test_position_outside_env_geodesic` - Position outside env returns NaN
- `test_mixed_valid_invalid_positions_geodesic` - Mixed positions handled correctly
- `test_occupancy_with_nan_distances` - Occupancy handles NaN gracefully
- `test_spike_binning_with_nan_distances` - Spike binning handles NaN gracefully

**Validation**:
- All 48 tests in `test_encoding_egocentric_binning.py` pass
- All 1986 encoding tests pass
- mypy: no issues
- ruff: all checks passed

### Milestone 7 Completion Notes (2026-01-08)

**Task 7.4**: Updated CLAUDE.md documentation
- Updated "Most Common Patterns" section with new API (compute_spatial_rate, etc.)
- Updated API_REFERENCE.md with new/legacy API sections for all encoding types
- Added "Backend Parameter" section documenting numpy/jax/auto options
- Added "Result Class Methods" section documenting all result class APIs

**Task 7.5**: Updated QUICKSTART.md documentation
- Updated Neural Analysis section with new API patterns
- Added batch processing examples (compute_spatial_rates, n_jobs, to_dataframe)
- Updated Object-Vector Cells section with compute_egocentric_rate(s)
- Updated Spatial View Cells section with compute_view_rate(s)

**Task 7.6**: Validation complete
- pytest: 9052 passed, 19 skipped
- mypy: 25 source files, no issues
- ruff: 580 files formatted, all checks passed

### Task 7.3 Completion Notes

Updated the following notebooks to use new encoding API:

- `08_spike_field_basics.ipynb`: `compute_place_field` → `compute_spatial_rate`
- `10_signal_processing_primitives.ipynb`: `compute_place_field` → `compute_spatial_rate`
- `11_place_field_analysis.ipynb`: `compute_place_field` → `compute_spatial_rate`
- `19_real_data_bandit_task.ipynb`: `compute_place_field` → `compute_spatial_rate`
- `20_bayesian_decoding.ipynb`: `compute_place_field` → `compute_spatial_rate`
- `22_spatial_view_cells.ipynb`: `compute_spatial_view_field` → `compute_view_rate`, `compute_place_field` → `compute_spatial_rate`

Note: `21_directional_place_fields.ipynb` does not exist in the repository.

All notebooks synced to `docs/examples/` directory.

## Milestone 7: Cleanup and Documentation - Migration Plan

### Overview

User requested **full migration** of utilities from old files to new locations, then removal of old files (no deprecation period).

### Files to Remove

1. `encoding/place.py` → merged into `spatial.py` and `_field_metrics.py`
2. `encoding/head_direction.py` → merged into `directional.py`
3. `encoding/spatial_view.py` → merged into `view.py`
4. `encoding/object_vector.py` → merged into `egocentric.py`

### Files to Keep

- `encoding/border.py` - standalone border score utilities
- `encoding/grid.py` - standalone grid score utilities
- `encoding/population.py` - population-level analysis
- `encoding/phase_precession.py` - theta phase precession

### Migration Details

#### From place.py:

| Function | Destination | Notes |
|----------|-------------|-------|
| `compute_place_field` | Remove | Replaced by `compute_spatial_rate` |
| `DirectionalPlaceFields` | `spatial.py` | Dataclass for directional fields |
| `compute_directional_place_fields` | `spatial.py` | Computes direction-conditioned fields |
| `detect_place_fields` | `spatial.py` | Field detection with helpers |
| `skaggs_information` | Re-export from `_metrics.py` | Already exists as `spatial_information` |
| `sparsity` | Re-export from `_metrics.py` | Already exists |
| `field_size` | `_field_metrics.py` | Field geometry metric |
| `rate_map_centroid` | `_field_metrics.py` | Field geometry metric |
| `field_shape_metrics` | `_field_metrics.py` | Field geometry metric |
| `field_stability` | `_field_metrics.py` | Field comparison metric |
| `field_shift_distance` | `_field_metrics.py` | Field comparison metric |
| `compute_field_emd` | `_field_metrics.py` | Field comparison metric |
| `in_out_field_ratio` | `_field_metrics.py` | Field comparison metric |
| `rate_map_coherence` | `_field_metrics.py` | Field comparison metric |
| `information_per_second` | `_metrics.py` | Information metric |
| `mutual_information` | `_metrics.py` | Information metric |
| `selectivity` | `_metrics.py` | Selectivity metric |
| `spatial_coverage_single_cell` | `_metrics.py` | Coverage metric |

#### From head_direction.py:

| Function | Destination | Notes |
|----------|-------------|-------|
| `compute_head_direction_tuning_curve` | Remove | Replaced by `compute_directional_rate` |
| `HeadDirectionMetrics` | Remove | Replaced by `DirectionalRateResult` methods |
| `head_direction_metrics` | Remove | Use `DirectionalRateResult` instead |
| `is_head_direction_cell` | `directional.py` | Keep for convenience |
| `plot_head_direction_tuning` | `directional.py` | Keep for convenience |
| `circular_mean` | Re-export from `stats.circular` | Already exists |
| `mean_resultant_length` | Re-export from `stats.circular` | Already exists |
| `rayleigh_test` | Re-export from `stats.circular` | Already exists |

#### From spatial_view.py:

| Function | Destination | Notes |
|----------|-------------|-------|
| `compute_spatial_view_field` | Remove | Replaced by `compute_view_rate` |
| `SpatialViewFieldResult` | Remove | Replaced by `ViewRateResult` |
| `SpatialViewMetrics` | Remove | Replaced by `ViewRateResult` methods |
| `spatial_view_cell_metrics` | Remove | Use `ViewRateResult` instead |
| `is_spatial_view_cell` | `view.py` | Keep for convenience |
| `compute_viewed_location` | `view.py` | Core gaze computation |
| `compute_viewshed` | `view.py` | Core visibility computation |
| `visibility_occupancy` | `view.py` | Core occupancy computation |
| `FieldOfView` | `view.py` | Dataclass |

#### From object_vector.py:

| Function | Destination | Notes |
|----------|-------------|-------|
| `compute_object_vector_field` | Remove | Replaced by `compute_egocentric_rate` |
| `ObjectVectorFieldResult` | Remove | Replaced by `EgocentricRateResult` |
| `ObjectVectorMetrics` | Remove | Replaced by `EgocentricRateResult` methods |
| `compute_object_vector_tuning` | `egocentric.py` | Keep for tuning analysis |
| `object_vector_score` | `egocentric.py` | Keep for OVC metric |
| `is_object_vector_cell` | `egocentric.py` | Keep for convenience |
| `plot_object_vector_tuning` | `egocentric.py` | Keep for visualization |

---

## Session Notes

### Task 7.2: Full Migration - In Progress (2025-12-20)

**Completed**:

1. **Created `_field_metrics.py`** with field geometry and comparison utilities:
   - `field_size`, `rate_map_centroid`, `field_shape_metrics`
   - `field_stability`, `field_shift_distance`, `compute_field_emd`
   - `in_out_field_ratio`, `rate_map_coherence`

2. **Extended `_metrics.py`** with additional metrics:
   - `selectivity` (peak/mean ratio)
   - `information_per_second` (bits/second)
   - `mutual_information` (= information_per_second)
   - `spatial_coverage_single_cell` (fraction with firing)

**Remaining**:

- Move `detect_place_fields` and `DirectionalPlaceFields` to `spatial.py`
- Move `is_head_direction_cell`, `plot_head_direction_tuning` to `directional.py`
- Move `is_spatial_view_cell`, gaze utilities to `view.py`
- Move OVC utilities to `egocentric.py`
- Update `__init__.py` exports
- Update all external imports (tests, examples, docs)
- Remove old files

---

### Task 6.7: Add performance benchmarks [COMPLETED]

**Goal**: Create a benchmark script comparing NumPy vs JAX backends for spatial rate computation.

**Implementation**:

Created `benchmarks/bench_encoding_backends.py` with:
1. `EncodingBenchmarkResult` dataclass for storing results
2. `create_benchmark_data()` for generating test data with configurable population sizes
3. `run_single_benchmark()` for running individual benchmarks with warmup
4. `run_encoding_benchmark()` for full benchmark suite
5. `results_to_dataframe()` and `print_summary_table()` for output formatting
6. Command-line interface with options for population sizes, iterations, smoothing method, etc.

**Key Design Decisions**:
- Warmup iterations excluded from timing (critical for JAX JIT compilation)
- Triple garbage collection before timing for clean baselines
- Multiple iterations averaged (default n=3) for statistical validity
- Speedup calculated relative to NumPy baseline
- Supports both "binned" and "diffusion_kde" smoothing methods

**Performance Results**:

| Smoothing | n_neurons | NumPy (ms) | JAX (ms) | Speedup |
|-----------|-----------|------------|----------|---------|
| binned    | 10        | 4.6        | 3.8      | 1.23x   |
| binned    | 100       | 13.2       | 13.6     | 0.97x   |
| binned    | 1000      | 112.5      | 115.7    | 0.97x   |
| diffusion | 10        | 3.1        | 3.6      | 0.86x   |
| diffusion | 100       | 12.9       | 12.0     | 1.08x   |
| diffusion | 1000      | 149.3      | 92.0     | **1.62x** |

**Key Findings**:
- JAX shows meaningful speedup for large populations (1.62x at 1000 neurons with diffusion_kde)
- Speedup comes from accelerated matrix operations in the smoothing kernel
- For small populations (<100), JAX overhead negates benefits
- Binning layer always runs on NumPy; JAX only accelerates smoothing step

**Files Created**:
- `benchmarks/bench_encoding_backends.py`: Main benchmark script
- `tests/benchmarks/test_encoding_backends.py`: Tests for benchmark components
- `benchmarks/BASELINE.md`: Updated with encoding backend performance results

**TDD Process**:
1. Created test file first with expected structure
2. Ran tests to verify failure (red)
3. Implemented benchmark script
4. Ran tests to verify success (green)
5. Applied code review, addressed feedback
6. Added JAX x64 fixture to tests

---

### Task 6.6 Follow-up #2: Wire up real JAX compute path [COMPLETED]

**Goal**: Make `backend="jax"` use actual JAX operations for smoothing/rate computation,
not just convert outputs to JAX arrays.

**Issue Found** (via second code review):

The previous fix only converted outputs to JAX arrays at the end. The core math
(smoothing, rate computation) was still NumPy/SciPy. This meant:

1. No GPU acceleration
2. No JAX tracing compatibility (can't use with `jit`, `grad`, `vmap`)
3. `_core_jax.py` functions were still orphaned

**Implementation**:

Added backend-aware smoothing in `_smoothing.py`:

1. `smooth_rate_map(..., backend="jax")` now:
   - Converts inputs to JAX arrays with explicit `dtype=jnp.float64`
   - For diffusion_kde/gaussian_kde: Uses JAX for kernel smoothing (`kernel_j @ spike_counts_j`)
   - For binned: Uses `_core_jax.compute_firing_rate_single()` for rate computation
   - Returns JAX arrays directly

2. `smooth_rate_maps_batch(..., backend="jax")` similarly uses JAX operations

3. `compute_spatial_rate(s)` passes `backend=resolved_backend` to smoothing functions

**JAX Operations Used**:

```python
# In _smooth_rate_map_jax:
kernel_j = jnp.asarray(kernel, dtype=jnp.float64)  # NumPy kernel → JAX
spike_density = kernel_j @ spike_counts_j  # JAX matrix multiply
occupancy_density = kernel_j @ occupancy_j  # JAX matrix multiply
firing_rate = jnp.where(occupancy_density > 0, spike_density / occupancy_density, jnp.nan)
```

**Note on dtype**: JAX defaults to float32 unless `jax_enable_x64` is set. The code
requests float64 but JAX will downcast to float32 with a warning if x64 is disabled.
With `JAX_ENABLE_X64=1`, results match NumPy to machine precision (5e-15).

**Files Modified**:

- `src/neurospatial/encoding/_smoothing.py`: Added `backend` parameter and JAX implementations
- `src/neurospatial/encoding/spatial.py`: Pass backend to smoothing functions

**Verification**:

- All 34 backend/JAX tests pass
- JAX arrays are returned from compute functions
- With `JAX_ENABLE_X64=1`, results match NumPy to ~5e-15

**Remaining Work**:

- Metrics (`_metrics.py`) are still NumPy-only
- Other compute functions (directional, view, egocentric) need similar updates
- Docstrings need updating to reflect actual JAX behavior

---

### Task 6.6 Follow-up: Wire up JAX backend dispatch [COMPLETED]

**Goal**: Fix the JAX backend so `backend="jax"` actually works instead of raising `NotImplementedError`.

**Issue Found** (via code review):

The compute functions (`compute_spatial_rate`, etc.) were raising `NotImplementedError` for
`backend="jax"` because the JAX pipeline was never connected. The code review revealed:

1. **Critical**: JAX backend unreachable - compute functions short-circuit to NumPy
2. **Critical**: `backend="jax"` raises `NotImplementedError` instead of working
3. **Major**: Metrics are NumPy-only and force `_to_numpy` conversion
4. **Major**: `_core_jax.py` is orphaned (not used in compute pipeline)

**Design Decision**: Core math only (not binning)

Following the PLAN.md two-layer architecture:

- **Binning layer**: Always NumPy (CPU/joblib for parallelization)
- **Core rate/metrics layer**: Converts to JAX arrays at the end when JAX backend selected

This approach maximizes compatibility - binning uses joblib parallelization which works great
on CPU, and JAX acceleration applies to the final result arrays which can be used in
downstream JAX computations.

**Implementation**:

Modified all compute functions to:

1. Use `get_backend_name(backend)` to resolve "auto" to actual backend
2. Keep binning on NumPy (CPU/joblib) - this is intentional per PLAN.md
3. Convert results to JAX arrays at the end if JAX backend selected

**Pattern Applied**:

```python
from neurospatial.encoding._backend import (
    SUPPORTED_BACKENDS,
    get_backend_name,
    is_jax_available,
)

# Validate backend
if backend not in SUPPORTED_BACKENDS:
    raise ValueError(...)

resolved_backend = get_backend_name(backend)

# ... binning stays NumPy ...

# Convert to JAX arrays if JAX backend is selected
if resolved_backend == "jax" and is_jax_available():
    import jax.numpy as jnp
    firing_rate = jnp.asarray(firing_rate)
    occupancy = jnp.asarray(occupancy)
```

**Files Modified**:

- `src/neurospatial/encoding/spatial.py`: `compute_spatial_rate`, `compute_spatial_rates`
- `src/neurospatial/encoding/directional.py`: `compute_directional_rate`, `compute_directional_rates`
- `src/neurospatial/encoding/view.py`: `compute_view_rate`, `compute_view_rates`
- `src/neurospatial/encoding/egocentric.py`: `compute_egocentric_rate`, `compute_egocentric_rates`

**Tests Created**:

- `tests/encoding/test_jax_compute_dispatch.py` (8 tests)
  - Tests that `backend="jax"` does not raise
  - Tests that JAX backend produces same results as NumPy
  - Tests across all compute function types

**Tests Removed** (obsolete stubs):

- `tests/encoding/test_encoding_core_numpy.py`: Removed `test_raises_not_implemented_error` tests
- `tests/encoding/test_encoding_core_jax.py`: Removed `test_raises_not_implemented_error` tests

These stub tests expected `NotImplementedError` from core functions that are now implemented.

**Verification**:

- All 1978 encoding tests pass
- All mypy checks pass (added `# type: ignore[assignment]` for JAX array reassignments)
- All ruff checks pass

---

### Task 6.6: Write JAX-specific tests [COMPLETED]

**Goal**: Verify JAX backend behavior through comprehensive tests.

**Analysis**:

Upon review, the tests required for Task 6.6 were already implemented across multiple test files:

1. **JAX backend produces same results as NumPy**:
   - `test_core_jax_metrics.py::TestNumpyEquivalence` (4 tests)
   - `test_jax_backend_awareness.py::TestNumpyJaxConsistency` (2 tests)

2. **`"auto"` behavior with JAX installed**:
   - `test_encoding_backend.py::TestGetBackend::test_auto_backend_returns_numpy_or_jax`
   - All `test_accepts_auto_backend` tests in `test_backend_dispatch.py`

3. **`"auto"` behavior on Windows (forces NumPy)**:
   - `test_encoding_backend.py::TestGetBackend::test_auto_backend_returns_numpy_on_windows`
   - `test_encoding_backend.py::TestIsJaxAvailable::test_returns_false_on_windows`

4. **`"jax"` raises clear error when unavailable**:
   - `test_encoding_backend.py::TestGetBackend::test_jax_backend_raises_when_unavailable`

**Test Coverage Summary**:

- `test_encoding_backend.py`: 24 tests for backend infrastructure
- `test_backend_dispatch.py`: 26 tests for compute function backend parameter
- `test_jax_backend_awareness.py`: 23 tests for result classes with JAX arrays
- `test_core_jax_metrics.py`: 49 tests for JAX metric implementations

**Total**: 122 tests pass, 1 skipped (JAX unavailable path)

**Note**: The compute functions (`compute_spatial_rate`, etc.) with `backend="jax"`
raise `NotImplementedError` because the full JAX pipeline is not yet implemented.
This is intentional - Task 6.1 only implemented the core smoothing operations in
`_core_jax.py`, not the full compute pipeline.

**Files Reviewed**: No new files created; existing tests verified as comprehensive.

---

### Task 6.5: Update result class methods for backend awareness [COMPLETED]

**Goal**: Ensure result class methods correctly handle JAX arrays.

**Issue Found**:

The existing `_get_array_module()` function used `hasattr(arr, "__jax_array__")` to detect JAX arrays. However, modern JAX arrays (`jaxlib._jax.ArrayImpl`) no longer have this attribute. The test `test_jax_array_returns_jax_numpy` was failing.

**Root Cause**:

- JAX changed their array implementation; `__jax_array__` no longer exists on JAX arrays
- The correct way to detect JAX arrays is via `isinstance(arr, jax.Array)`

**Fix Applied** (`_base.py`):

```python
def _get_array_module(arr: ArrayLike) -> Any:
    # Check if JAX is available and if arr is a JAX array
    try:
        import jax
        import jax.numpy as jnp

        if isinstance(arr, jax.Array):
            return jnp
    except ImportError:
        pass
    return np
```

**Tests Created**:

- `tests/encoding/test_jax_backend_awareness.py` (23 tests)
  - `TestToNumpyWithJax`: 3 tests for JAX → NumPy conversion
  - `TestGetArrayModuleWithJax`: 2 tests for JAX array detection
  - `TestSpatialResultMixinWithJax`: 4 tests for mixin methods with JAX
  - `TestSpatialRateResultWithJax`: 3 tests for single result with JAX
  - `TestSpatialRatesResultWithJax`: 3 tests for batch result with JAX
  - `TestDirectionalRateResultWithJax`: 3 tests for directional with JAX
  - `TestViewRateResultWithJax`: 1 test for view with JAX
  - `TestEgocentricRateResultWithJax`: 2 tests for egocentric with JAX
  - `TestNumpyJaxConsistency`: 2 tests verifying NumPy/JAX produce same results

**Files Modified**:

- `src/neurospatial/encoding/_base.py`: Fixed `_get_array_module()` to use `isinstance(arr, jax.Array)`

**Files Created**:

- `tests/encoding/test_jax_backend_awareness.py` (23 tests)

**Verification**:

All plot methods already correctly use `_to_numpy()` or `np.asarray()` which handle JAX arrays transparently:

- `SpatialRateResult.plot()` - uses `_to_numpy()`
- `SpatialRatesResult.plot()` - uses `np.asarray()` + `_to_numpy()`
- `EgocentricRateResult.plot()` - uses `_to_numpy()`
- `EgocentricRatesResult.plot()` - uses `_to_numpy()`
- `DirectionalRateResult.plot()` - uses `np.asarray()`
- `DirectionalRatesResult.plot()` - delegates via `self[idx].plot()`
- `ViewRateResult.plot()` - uses `_to_numpy()`
- `ViewRatesResult.plot()` - uses `_to_numpy()`

**Test Results**: 47/47 tests pass (24 base tests + 23 JAX tests)
**Quality checks**: All ruff and mypy checks pass

---

### Task 6.3: Implement JAX grid/border score computations [SKIPPED - N/A]

**Goal**: Port grid score and border score inner loops to JAX with vmap for batch operations.

**Analysis**:

After reviewing the existing implementations in `grid.py` and `border.py`, I determined that porting these functions to JAX is not feasible:

**Grid Score Dependencies** (cannot port to JAX):

- `scipy.fft.fft2` / `scipy.fft.ifft2` - FFT-based 2D autocorrelation
- `scipy.ndimage.rotate` - Image rotation for 30°, 60°, 90°, 120°, 150° angles
- `scipy.stats.pearsonr` - Correlation computation
- `skimage.feature.peak_local_max` - Peak detection for scale/orientation

**Border Score Dependencies** (cannot port to JAX):

- `networkx.multi_source_dijkstra_path_length` - Geodesic distance computation
- `networkx.adjacency_matrix` - Graph connectivity
- `scipy.sparse.csgraph.dijkstra` - Alternative Dijkstra implementation
- Environment object methods (`boundary_bins`, `bin_centers`) - Not JAX-traceable

**Key Insight**: The "inner loops" identified in the task description are actually:

1. Simple vectorized NumPy operations (min/max, correlation)
2. Already efficient for CPU computation
3. Not the bottleneck in these algorithms

The expensive operations (FFT autocorrelation, image rotation, graph traversal) are fundamentally incompatible with JAX's programming model and would require complete algorithm redesign.

**Decision**: Skip Task 6.3 entirely. The batch functions `batch_grid_scores()` and `batch_border_scores()` in `_metrics.py` will remain NumPy-only. This is documented as an intentional limitation of the JAX backend.

**Rationale**:

1. Grid/border score computations are typically done once per recording session (not per-frame)
2. The NumPy implementations are already efficient with vectorized operations
3. JAX's benefit is in GPU-accelerated batch processing, which doesn't apply here
4. Reimplementing the algorithms in pure JAX would be complex and error-prone

**Files Modified**: None

**Decision made by**: User approval (2025-12-19)

---

### Task 6.4: Add backend dispatch to compute functions [COMPLETED]

**Goal**: Add `backend` parameter to all compute functions for API consistency.

**Implementation**:

Added `backend` parameter (default="numpy") to:

- `compute_directional_rate()` and `compute_directional_rates()` in `directional.py`
- `compute_view_rate()` and `compute_view_rates()` in `view.py`
- `compute_egocentric_rate()` and `compute_egocentric_rates()` in `egocentric.py`

Note: `compute_spatial_rate(s)` already had the backend parameter.

Each function:

1. Validates backend is in `SUPPORTED_BACKENDS` (numpy, jax, auto)
2. Raises `NotImplementedError` for `backend="jax"` (JAX implementation not yet available)
3. Uses NumPy for `backend="auto"` (fallback until JAX implementations exist)

**Tests Created**: `tests/encoding/test_backend_dispatch.py` with 26 tests covering:

- All compute functions accept `backend="numpy"`
- All compute functions accept `backend="auto"`
- All compute functions reject invalid backend
- Consistency tests verifying all functions have `backend` parameter with `"numpy"` default

**Files Modified**:

- `src/neurospatial/encoding/directional.py`
- `src/neurospatial/encoding/view.py`
- `src/neurospatial/encoding/egocentric.py`
- `tests/encoding/test_backend_dispatch.py` (new)

---

### Task 6.2: Implement JAX metric computations in `_core_jax.py` [COMPLETED]

**Goal**: Implement JAX versions of spatial_information and sparsity metrics with vmap for batch operations.

**Implementation**:

- Added four functions to `src/neurospatial/encoding/_core_jax.py`:
  - `spatial_information_single()`: Compute Skaggs spatial information (bits/spike) for single neuron
  - `spatial_information_batch()`: Batch version using `jax.vmap` for multiple neurons
  - `sparsity_single()`: Compute sparsity measure (0-1) for single neuron
  - `sparsity_batch()`: Batch version using `jax.vmap` for multiple neurons

**Key Design Decisions**:

- JAX functions use `jnp.nan_to_num()` to handle NaN inputs gracefully
- Use `jnp.where()` for safe division and conditional computation
- Batch functions use `jax.vmap` to vectorize over neurons efficiently
- All functions compatible with `jit`, `vmap`, and `grad` transformations
- Handle edge cases: zero firing rate, zero occupancy, single bin, empty batch

**Algorithm Details**:

Spatial Information (Skaggs et al. 1993):
```
I = Σᵢ pᵢ × (rᵢ/r̄) × log₂(rᵢ/r̄)
```
where pᵢ is occupancy probability, rᵢ is firing rate, r̄ is mean rate.

Sparsity (Skaggs et al. 1996):
```
S = (Σᵢ pᵢ × rᵢ)² / (Σᵢ pᵢ × rᵢ²)
```

**Files Modified**:

- `src/neurospatial/encoding/_core_jax.py`: Added 4 new functions, updated __all__

**Files Created**:

- `tests/encoding/test_core_jax_metrics.py` (49 tests)

**TDD Process**:
1. Wrote 49 tests covering:
   - Import tests (all functions importable)
   - Return type tests (JAX array outputs)
   - Correctness tests (uniform → 0 info, selective → high info)
   - Edge case handling (zero occupancy, NaN, single bin)
   - NumPy equivalence tests
   - JAX-specific features (jit, vmap)
   - Consistency tests (batch matches single)
2. Ran tests - all failed (expected - stubs raised NotImplementedError)
3. Implemented all 4 functions
4. Ran tests - 48 passed, 1 failed (floating point tolerance near zero)
5. Fixed test with appropriate atol for near-zero comparisons
6. All 49 tests pass

**Test Results**: 49/49 tests pass
**Quality checks**: All ruff and mypy checks pass

---

### Task 6.1: Implement JAX smoothing operations in `_core_jax.py` [COMPLETED]

**Goal**: Implement JAX versions of core array operations for encoding computations, enabling GPU acceleration for batch neural encoding.

**Implementation**:

- Added JAX as optional dependency to `pyproject.toml` (jax>=0.4.30, jaxlib>=0.4.30)
- Implemented four functions in `src/neurospatial/encoding/_core_jax.py`:
  - `compute_firing_rate_single()`: Convert spike counts and occupancy to firing rate
  - `compute_firing_rates_batch()`: Batch version for multiple neurons
  - `smooth_rate_map_single()`: Apply kernel smoothing to firing rate map
  - `smooth_rate_maps_batch()`: Batch version for multiple neurons
- Also implemented the corresponding NumPy functions in `src/neurospatial/encoding/_core_numpy.py`

**Key Design Decisions**:

- JAX functions use `jnp.where()` for safe division (NaN handling)
- Smoothing functions apply precomputed kernel via matrix multiplication (bandwidth not used dynamically)
- All functions work with JAX arrays and are compatible with `jit`, `vmap`, and `grad`
- Enable `jax_enable_x64` in tests for numerical equivalence with NumPy

**Files Modified**:

- `pyproject.toml`: Added JAX optional dependency and mypy overrides
- `src/neurospatial/encoding/_core_numpy.py`: Implemented all 4 stub functions
- `src/neurospatial/encoding/_core_jax.py`: Implemented all 4 stub functions

**Files Created**:

- `tests/encoding/test_core_jax.py` (41 tests)

**TDD Process**:
1. Wrote 41 tests covering:
   - Import tests (all functions importable)
   - Return type tests (JAX array outputs)
   - Shape tests (output matches input)
   - Correctness tests (rate = counts / occupancy)
   - Zero/NaN occupancy handling
   - min_occupancy threshold
   - Kernel smoothing behavior
   - Batch consistency with single calls
   - Numerical equivalence with NumPy
   - JAX-specific features (jit, vmap)
   - Edge cases (all zeros, single bin, all NaN)
2. Ran tests - all failed (expected - stubs raised NotImplementedError)
3. Implemented NumPy functions first
4. Implemented JAX functions
5. Ran tests - all 41 passed

**Test Results**: 41/41 tests pass
**Quality checks**: All ruff and mypy checks pass

---

### Task 5.9: Write comprehensive tests for egocentric encoding [COMPLETED]

**Goal**: Consolidate and verify comprehensive test coverage for the egocentric encoding module.

**Summary**:
This task was a validation/consolidation task. During TDD for Tasks 5.5-5.8, comprehensive tests were already written:

| Test File | Test Count | Coverage |
|-----------|------------|----------|
| `test_encoding_egocentric.py` | 97 tests | Result classes, methods |
| `test_compute_egocentric_rate.py` | 40 tests | Single neuron computation |
| `test_compute_egocentric_rates.py` | 40 tests | Batch computation |
| `test_encoding_egocentric_binning.py` | 41 tests | Binning layer |
| **Total** | **218 tests** | **All requirements** |

**Coverage Verification**:
- [x] Single neuron computation - 40 tests in `test_compute_egocentric_rate.py`
- [x] Batch computation - 40 tests in `test_compute_egocentric_rates.py`
- [x] All result class methods - 97 tests in `test_encoding_egocentric.py`
- [x] Euclidean vs geodesic distance - Tests in all compute test files
- [x] Geodesic without env error - Tests in `TestComputeEgocentricRateDistanceMetric`, `TestComputeEgocentricRatesInputValidation`
- [x] `to_dataframe()` output format - 18 tests in `TestEgocentricRatesResultToDataframe`

**Test Results**: 218/218 tests pass
**Quality checks**: All ruff and mypy checks pass

---

### Task 5.8: Implement `compute_egocentric_rates()` function [COMPLETED]

**Goal**: Implement the batch version of egocentric rate computation for processing multiple neurons efficiently.

**Implementation**:
- Added `compute_egocentric_rates()` function to `src/neurospatial/encoding/egocentric.py` (lines 1185-1471)
- Function signature follows canonical argument order (spike_times, times, positions, headings, object_positions)
- Keyword-only parameters: env, distance_range, n_distance_bins, n_direction_bins, distance_metric, smoothing_method, bandwidth, min_occupancy, n_jobs
- Supports all spike time formats via `normalize_spike_times()`:
  - List/tuple of 1D arrays (canonical)
  - 2D array with NaN padding
  - Single 1D array (wrapped in list)
- Delegates to:
  - `_egocentric_binning.bin_egocentric_spike_trains()` for batch spike counting with shared occupancy
  - `_smoothing.smooth_rate_maps_batch()` for batch smoothing
- Returns `EgocentricRatesResult` with iteration support

**Key Design Decisions**:
- Follows `compute_view_rates()` pattern exactly for consistency
- Precomputes egocentric coordinates and occupancy once (shared across all neurons)
- Uses `n_jobs` parameter for joblib parallelization of spike binning
- Handles edge case of zero neurons gracefully (still computes occupancy)
- Comprehensive input validation (distance_metric, env requirement, array lengths)

**Files Modified**:
- `src/neurospatial/encoding/egocentric.py`: Added compute_egocentric_rates function, updated __all__
- `src/neurospatial/encoding/__init__.py`: Added compute_egocentric_rates to exports and __all__

**Files Created**:
- `tests/encoding/test_compute_egocentric_rates.py` (40 tests)

**TDD Process**:
1. Wrote 40 tests covering:
   - Import tests (from egocentric module, encoding package, in __all__)
   - Return type tests (correct type, shapes for firing_rates/occupancy/ego_env)
   - Spike time format tests (list of arrays, 2D NaN-padded, single 1D array)
   - Parameter storage tests (distance_range, n_distance_bins, n_direction_bins)
   - Parameter handling tests (distance_metric, smoothing_method, n_jobs)
   - Iteration tests (len, getitem, iteration)
   - Edge cases (empty list, single neuron, empty spike trains)
   - Input validation tests (invalid distance_metric, geodesic without env, mismatched lengths)
   - Consistency tests (single neuron matches compute_egocentric_rate, multiple neurons consistent)
   - Signature tests (argument order, keyword-only parameters)
   - Correctness tests (non-negative rates, non-negative occupancy, total occupancy approximates duration)
   - Result method integration tests (plot, preferred_distances, preferred_directions, detect_ovcs, to_dataframe)
2. Ran tests - all 40 failed (expected - function didn't exist)
3. Implemented the function
4. Ran tests - all 40 passed

**Code Review**: APPROVED (no critical issues)
- Excellent consistency with existing batch patterns
- Comprehensive documentation quality
- Complete type annotations (mypy passes)
- Clean code structure with proper delegation

**Test Results**: 40/40 tests pass
**Quality checks**: All ruff and mypy checks pass

---

### Task 5.7: Implement `compute_egocentric_rate()` function [COMPLETED]

**Goal**: Implement the main compute function for egocentric (object-vector) rate maps.

**Implementation**:
- Added `compute_egocentric_rate()` function to `src/neurospatial/encoding/egocentric.py` (lines 942-1181)
- Function signature follows canonical argument order (spike_times, times, positions, headings, object_positions)
- Optional parameters (keyword-only): env, distance_range, n_distance_bins, n_direction_bins, distance_metric, smoothing_method, bandwidth, min_occupancy
- Delegates to:
  - `_egocentric_binning.bin_egocentric_spike_train()` for spike counts
  - `_egocentric_binning.compute_egocentric_occupancy()` for occupancy
  - `_smoothing.smooth_rate_map()` for smoothing
- Returns `EgocentricRateResult` with all required fields

**Key Design Decisions**:
- `env` is optional (keyword-only) - required only for geodesic distance metric
- Default `smoothing_method="binned"` (not diffusion_kde) as boundary-aware smoothing may not apply to egocentric polar grids
- Default `distance_range=(0.0, 50.0)`, `n_distance_bins=10`, `n_direction_bins=12`
- Input validation for array lengths and distance_metric values
- Comprehensive NumPy docstring with examples and scientific references

**Files Modified**:
- `src/neurospatial/encoding/egocentric.py`: Added compute_egocentric_rate function, updated __all__
- `src/neurospatial/encoding/__init__.py`: Added compute_egocentric_rate to exports

**Files Created**:
- `tests/encoding/test_compute_egocentric_rate.py` (900 lines, 40 tests)

**TDD Process**:
1. Wrote 40 tests covering:
   - Import tests (from egocentric module and encoding package)
   - Return type tests (correct type, shapes for firing_rate/occupancy/ego_env)
   - Distance range and bin count parameter tests
   - Distance metric parameter tests (euclidean default, geodesic requires env)
   - Smoothing parameter tests (all three methods, bandwidth)
   - Empty spike train behavior
   - Correctness tests (non-negative rates, total occupancy approximates duration)
   - Result methods integration tests (plot, preferred_distance, preferred_direction, etc.)
   - min_occupancy threshold tests
   - Input validation tests (mismatched lengths)
   - Signature tests (argument order, keyword-only parameters)
2. Ran tests - all 40 failed (expected - function didn't exist)
3. Implemented the function
4. Ran tests - all 40 passed

**Code Review**: APPROVED
- Excellent quality, matches patterns from compute_view_rate()
- Complete type annotations (mypy passes)
- Outstanding NumPy docstring
- Clean code structure with proper delegation

**Test Results**: 40/40 tests pass
**Quality checks**: All ruff and mypy checks pass

---

### Task 5.6: Implement binning layer for egocentric encoding [COMPLETED]

**Goal**: Create the binning layer for egocentric (object-vector) encoding that converts spike trains and trajectory data into spike counts and occupancy in egocentric polar coordinates.

**Implementation**:
- Created `src/neurospatial/encoding/_egocentric_binning.py` with three main functions:
  - `compute_egocentric_occupancy(times, positions, headings, object_positions, ...)`: Computes time spent at each distance/direction bin relative to nearest object
  - `bin_egocentric_spike_train(spike_times, times, positions, headings, object_positions, ...)`: Bins single neuron by egocentric coordinates
  - `bin_egocentric_spike_trains(spike_times, times, positions, headings, object_positions, ...)`: Batch version with joblib parallelization

**Key Design Decisions**:
- Returns `ego_env` (egocentric polar Environment) along with occupancy/spike counts for consistency with other encoding modules
- Uses `_compute_egocentric_coords()` helper that:
  - Computes distance and bearing to ALL objects at each timepoint
  - Finds NEAREST object at each timepoint
  - Supports both "euclidean" and "geodesic" distance metrics
- Uses `_coords_to_flat_bin_idx()` helper for converting (distance, bearing) to flat bin indices
- Precomputes egocentric coordinates once for batch processing (shared across all neurons)
- Follows patterns from `_view_binning.py` and `_directional_binning.py` exactly

**Files Created**:
- `src/neurospatial/encoding/_egocentric_binning.py` (820 lines)
- `tests/encoding/test_encoding_egocentric_binning.py` (618 lines, 41 tests)

**TDD Process**:
1. Wrote 41 tests covering:
   - Import tests
   - Basic functionality (returns correct types/shapes)
   - Occupancy computation (total approximates duration, non-negative, correct dtype)
   - Spike binning (non-negative counts, total <= input spikes)
   - Distance metric handling (euclidean default, geodesic requires env)
   - Input validation (mismatched lengths, insufficient samples, unsorted times)
   - Coordinate correctness (bearing 0 ahead, pi/2 left, -pi/2 right)
   - Nearest object selection
   - Batch consistency with single neuron results
   - n_jobs parallelization
   - 2D NaN-padded array input format
2. Ran tests - all 41 failed (expected)
3. Implemented the module
4. Ran tests - all 41 passed

**Code Review**: APPROVED (9.7/10)
- Perfect consistency with existing patterns (_view_binning.py, _directional_binning.py)
- Excellent documentation (comprehensive NumPy-style docstrings)
- Complete type annotations
- Robust edge case handling
- Minor improvements made:
  - Replaced assert with explicit ValueError for type narrowing
  - Added clarifying comment for NaN handling in geodesic distance

**Test Results**: 41/41 tests pass
**Quality checks**: All ruff and mypy checks pass

---

### Task 5.5: Implement `EgocentricRatesResult.to_dataframe()` [COMPLETED]

**Goal**: Add `to_dataframe()` method to `EgocentricRatesResult` for exporting metrics to pandas DataFrame.

**Implementation**:
- Added `to_dataframe()` method to `EgocentricRatesResult` in `src/neurospatial/encoding/egocentric.py` (lines 840-938)
- Added required imports: `Sequence` from collections.abc, `pd` for TYPE_CHECKING
- DataFrame columns: neuron_id, preferred_distance, preferred_direction, preferred_direction_deg, peak_rate, is_ovc
- Follows exactly the pattern from `ViewRatesResult.to_dataframe()` and `DirectionalRatesResult.to_dataframe()`
- Validates neuron_ids length matches number of neurons
- Handles empty results (0 neurons) gracefully
- Comprehensive NumPy-style docstring with Parameters, Returns, Raises, Notes, Examples, See Also sections

**TDD Process**:
1. Wrote 18 new tests in `tests/encoding/test_encoding_egocentric.py`:
   - 2 tests for basic functionality (returns DataFrame, correct row count)
   - 6 tests for column existence (neuron_id, preferred_distance, preferred_direction, preferred_direction_deg, peak_rate, is_ovc)
   - 3 tests for neuron ID handling (default indices, custom IDs, length mismatch error)
   - 5 tests for correctness validation (each column matches batch method output)
   - 2 tests for edge cases (empty result, single neuron)
2. Ran tests - all 18 failed (expected)
3. Implemented the method
4. Ran tests - all 18 passed

**Code Review**: APPROVED
- Perfect consistency with existing to_dataframe() implementations
- Comprehensive documentation quality
- Robust error handling
- Type-safe implementation
- No critical or quality issues

**Test Results**: 97/97 tests pass (79 original + 18 new)
**Quality checks**: All ruff and mypy checks pass

---

### Task 5.4: Implement `EgocentricRatesResult` batch methods [COMPLETED]

**Goal**: Add batch methods to `EgocentricRatesResult` for processing multiple neurons efficiently.

**Implementation**:
- Added six methods to `EgocentricRatesResult` in `src/neurospatial/encoding/egocentric.py`:
  - `plot(idx, ax, **kwargs)`: Plot a specific neuron's egocentric rate map (lines 608-657)
  - `preferred_distances()`: Returns (n_neurons,) array of preferred distances (lines 659-696)
  - `preferred_directions()`: Returns (n_neurons,) array of preferred directions (lines 698-740)
  - `egocentric_spatial_information()`: Returns (n_neurons,) array of spatial info (lines 742-774)
  - `detect_ovcs(min_info=0.3)`: Returns (n_neurons,) boolean array for OVC classification (lines 776-813)
  - `peak_firing_rates()`: Returns (n_neurons,) array of peak firing rates (lines 815-837)
- All methods follow `ViewRatesResult` pattern exactly for consistency
- Uses `batch_spatial_information()` from `_metrics.py` for efficient vectorized computation
- Added comprehensive NumPy-style docstrings with examples and cross-references

**TDD Process**:
1. Wrote 24 new tests in `tests/encoding/test_encoding_egocentric.py`:
   - 4 tests for `plot()` (returns axes, requires idx, accepts ax/kwargs)
   - 4 tests for `preferred_distances()` (shape, type, non-negative, consistency with single-neuron)
   - 3 tests for `preferred_directions()` (shape, type, consistency with single-neuron)
   - 6 tests for `detect_ovcs()` (shape, dtype, min_info param, default threshold, consistency)
   - 4 tests for `egocentric_spatial_information()` (shape, type, non-negative, consistency)
   - 3 tests for `peak_firing_rates()` (shape, type, correct values)
2. Ran tests - all 24 failed (expected)
3. Implemented the methods
4. Ran tests - all 24 passed

**Code Review**: APPROVED
- Perfect consistency with ViewRatesResult pattern
- Excellent documentation quality
- One suggestion: `to_dataframe()` method could be added in Task 5.5 (non-blocking)
- Minor performance optimization suggestion for vectorizing loops (very low priority)

**Test Results**: 79/79 tests pass (55 original + 24 new)
**Quality checks**: All ruff and mypy checks pass

---

### Task 5.3: Implement `EgocentricRateResult` classification [COMPLETED]

**Goal**: Add classification method (`is_ovc()`) to `EgocentricRateResult` for determining if a neuron is an object-vector cell.

**Implementation**:
- Added two methods to `EgocentricRateResult` in `src/neurospatial/encoding/egocentric.py`:
  - `egocentric_spatial_information()`: Computes spatial information using egocentric occupancy
    - Delegates to `spatial_information()` from `_metrics.py`
    - Uses `_to_numpy()` for JAX compatibility
    - Returns bits/spike (0 for uniform firing)
  - `is_ovc(min_info=0.3)`: Classifies neuron as OVC based on egocentric spatial information threshold
    - Default threshold 0.3 (lower than view cells at 0.5 due to sparser egocentric sampling)
    - Returns `True` if `egocentric_spatial_information() > min_info`
- Both methods follow the pattern from `ViewRateResult.is_view_cell()` for consistency
- Added comprehensive NumPy-style docstrings with scientific references (Hoydal et al., 2019)

**TDD Process**:
1. Wrote 11 new tests in `tests/encoding/test_encoding_egocentric.py`:
   - 6 tests for `is_ovc()` (returns bool, accepts min_info, default threshold, true for high info, false for uniform, respects threshold)
   - 5 tests for `egocentric_spatial_information()` (returns float, non-negative, zero for uniform, high for selective, uses occupancy)
2. Ran tests - all 11 failed (expected)
3. Implemented the methods
4. Ran tests - all 11 passed

**Code Review**: APPROVED
- No critical issues
- One low-priority suggestion: Minor docstring clarification (non-blocking)
- Perfect consistency with ViewRateResult pattern
- Outstanding documentation quality noted

**Test Results**: 55/55 tests pass (44 original + 11 new)
**Quality checks**: All ruff and mypy checks pass

---

### Task 5.2: Implement `EgocentricRateResult` convenience methods [COMPLETED]

**Goal**: Add three convenience methods to `EgocentricRateResult`: plot, preferred_distance, and preferred_direction.

**Implementation**:
- Added three methods to `EgocentricRateResult` in `src/neurospatial/encoding/egocentric.py`:
  - `plot(ax=None, **kwargs)`: Delegates to `ego_env.plot_field()` for visualization
  - `preferred_distance()`: Returns distance component (index 0) of peak bin
  - `preferred_direction()`: Returns direction component (index 1) of peak bin (0=ahead)
- All methods use `_to_numpy()` helper for JAX compatibility
- Added NumPy-style docstrings with coordinate convention documentation

**TDD Process**:
1. Wrote 13 new tests in `tests/encoding/test_encoding_egocentric.py`:
   - 3 tests for `plot()` (returns axes, accepts ax, accepts kwargs)
   - 4 tests for `preferred_distance()` (returns float, non-negative, within range, corresponds to peak)
   - 4 tests for `preferred_direction()` (returns float, valid range, corresponds to peak, 0=ahead)
   - 2 tests for NaN handling (some NaN values)
2. Ran tests - all 13 failed (expected)
3. Implemented the methods
4. Ran tests - all 13 passed

**Code Review**: APPROVED
- Optional suggestion: Add all-NaN edge case handling (matches existing behavior in view.py single-neuron methods)
- This is non-blocking and can be addressed in follow-up if desired

**Test Results**: 44/44 tests pass (31 original + 13 new)
**Quality checks**: All ruff and mypy checks pass

---

### Task 5.1: Create `encoding/egocentric.py` with result class definitions [COMPLETED]

**Goal**: Define the result dataclasses for egocentric (object vector) cell encoding.

**Implementation**:
- Created `src/neurospatial/encoding/egocentric.py` with:
  - `EgocentricRateResult` dataclass (frozen=True) with fields:
    - `firing_rate`: ArrayLike (n_bins,)
    - `occupancy`: ArrayLike (n_bins,)
    - `ego_env`: Environment (egocentric polar environment)
    - `distance_range`: tuple[float, float]
    - `n_distance_bins`: int
    - `n_direction_bins`: int
  - `EgocentricRatesResult` dataclass (frozen=True) with:
    - Same fields but `firing_rates` shape (n_neurons, n_bins)
    - `__len__`, `__getitem__`, `__iter__` methods for iteration
- Updated `src/neurospatial/encoding/__init__.py` to export the new classes
- Created comprehensive test suite: `tests/encoding/test_encoding_egocentric.py` (31 tests)

**TDD Process**:
1. Wrote 31 tests covering imports, definitions, creation, and iteration
2. Ran tests - all 31 failed (expected)
3. Implemented the dataclasses
4. Ran tests - all 31 passed

**Code Review**: APPROVED with no critical or quality issues

**Test Results**: 31/31 tests pass

---


### All Code Review Bugfixes (2025-12-19) [COMPLETED]

**Context**: Code review identified 6 issues in the encoding module. All have been resolved.

**Fixes Applied (First Session)**:

1. **HIGH: view_occupancy used constant dt instead of per-sample deltas**
   - **Bug**: `compute_view_occupancy()` used `dt = np.median(np.diff(times))` which gives incorrect results for non-uniform sampling
   - **Fix**: Changed to `dt = np.diff(times)` and properly accumulate per-interval time
   - **Also fixed**: Last frame was incorrectly contributing to occupancy (n samples should give n-1 intervals)
   - **Files**: `src/neurospatial/encoding/_view_binning.py`
   - **Tests**: Added 4 new tests in `TestViewOccupancyNonUniformSampling` and `TestTimesValidation`

2. **LOW: No time monotonicity validation in binning functions**
   - **Bug**: `compute_view_occupancy()` uses `searchsorted` which assumes sorted input; unsorted times silently mis-bin
   - **Fix**: Added validation that times are monotonically non-decreasing
   - **Files**: `src/neurospatial/encoding/_view_binning.py`
   - **Tests**: Added `test_unsorted_times_raises_error` and `test_duplicate_times_allowed`

3. **MEDIUM: compute_spatial_rates returns zero occupancy for empty neurons**
   - **Bug**: When `spike_times=[]`, occupancy was returned as all zeros instead of actual trajectory occupancy
   - **Fix**: Compute occupancy from trajectory even when no neurons provided
   - **Files**: `src/neurospatial/encoding/spatial.py`
   - **Tests**: Added `test_empty_neurons_list_has_valid_occupancy`

**Fixes Applied (Second Session)**:

4. **MEDIUM: View binning recomputes viewed locations redundantly**
   - **Bug**: `bin_view_spike_trains()` was computing viewed locations separately for each neuron
   - **Fix**: Created `_precompute_view_bins()` and `_bin_spikes_with_precomputed_view_bins()` helpers
   - Refactored `bin_view_spike_trains()` to precompute view bins once and reuse for all neurons
   - Refactored `compute_view_occupancy()` to also use the precomputation helper
   - **Files**: `src/neurospatial/encoding/_view_binning.py`
   - **Tests**: Added `TestBinViewSpikeTrainsPrecomputation` class (2 tests)

5. **MEDIUM: ViewRatesResult.to_dataframe() assumes 2D environment**
   - **Bug**: `to_dataframe()` accessed `peak_locs[:, 1]` without checking if 1D
   - **Fix**: Added `n_dims` check before accessing y-coordinate, fill with NaN for 1D
   - **Files**: `src/neurospatial/encoding/view.py`
   - **Pattern**: Mirrors `SpatialRatesResult.to_dataframe()` 1D handling

6. **MEDIUM: Backend docs vs implementation mismatch**
   - **Bug**: Module docstring said "to be implemented in Tasks 4.7-4.8" for compute functions that were already implemented
   - **Fix**: Updated module docstring to remove outdated task references
   - **Files**: `src/neurospatial/encoding/view.py`

**All tests pass**: 183 total tests (77 view encoding, 48 view binning, 58 compute_view_rate)

---

### Task 4.9: Write Comprehensive Tests for View Encoding [COMPLETED]

**Goal**: Verify comprehensive test coverage for all view encoding functionality.

**Approach**: Reviewed existing test coverage against Task 4.9 requirements.

**Result**:

All requirements already covered by tests written during TDD in Tasks 4.1-4.8:

- **Single neuron computation**: 34 tests (TestComputeViewRateFunction and related classes)
- **Batch computation**: 24 tests (TestComputeViewRatesFunction and related classes)
- **All result class methods**: 77 tests covering ViewRateResult and ViewRatesResult methods
  - `plot()`: 6 tests (returns axes, accepts ax, accepts kwargs, requires idx for batch)
  - `peak_view_location(s)()`: 9 tests (shape, correctness, NaN handling, all-NaN edge case)
  - `view_spatial_information()`: 10 tests (return type, shape, matches single, non-negative, uniform=0)
  - `is_view_cell()` / `detect_view_cells()`: 11 tests (return type, shape, thresholds, matches single)
  - `to_dataframe()`: 18 tests (all columns, neuron IDs, edge cases, metric accuracy)
- **Different gaze models**: 15+ tests across binning and compute functions
  - `fixed_distance`: default and explicit tests
  - `ray_cast`: explicit tests
  - `boundary`: explicit tests
  - Verification that different models produce different results
- **`to_dataframe()` output format**: 18 tests (TestViewRatesResultToDataframe class)

**Total test count**: 176 tests across 3 files:

- `tests/encoding/test_encoding_view.py`: 77 tests (result class definitions and methods)
- `tests/encoding/test_compute_view_rate.py`: 58 tests (compute functions)
- `tests/encoding/test_encoding_view_binning.py`: 41 tests (binning layer)

**All tests pass**: 176 passed in 13.00s
**Code quality**: All ruff and mypy checks pass

**Milestone 4 Status**: COMPLETE! All 9 tasks finished (4.1-4.9).

---

### Task 4.8: Implement `compute_view_rates()` function [COMPLETED]

**Goal**: Create the batch version of view rate computation that efficiently processes multiple neurons with shared trajectory data.

**Approach**: TDD - wrote 24 tests first, then implemented.

**Result**:

- Added `compute_view_rates()` to `src/neurospatial/encoding/view.py`
- Added 24 tests to `tests/encoding/test_compute_view_rate.py` (total now 58 tests, all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `compute_view_rates(env, spike_times, times, positions, headings, *, gaze_model, view_distance, smoothing_method, bandwidth, min_occupancy, n_jobs)`:
  - Parameters follow canonical argument order from CLAUDE.md
  - Uses `normalize_spike_times()` to accept list of arrays, 2D NaN-padded array, or single 1D array
  - Uses `bin_view_spike_trains()` for efficient batch binning (precomputes view occupancy once)
  - Uses `smooth_rate_maps_batch()` for efficient batch smoothing (precomputes diffusion kernel once)
  - Returns `ViewRatesResult` with iteration support
  - Validates gaze_model parameter
  - Validates input array lengths
  - Handles zero-neuron edge case gracefully

**Signature**:

```python
def compute_view_rates(
    env: Environment,
    spike_times: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    gaze_model: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance",
    view_distance: float = 10.0,
    smoothing_method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy: float = 0.0,
    n_jobs: int = 1,
) -> ViewRatesResult:
```

**Test coverage (24 new tests in 10 classes)**:

- `TestComputeViewRatesImport`: 2 tests (import from module, in __all__)
- `TestComputeViewRatesReturnsResult`: 3 tests (return type, firing_rates shape, view_occupancy shape)
- `TestComputeViewRatesSpikeTimeFormats`: 2 tests (list of arrays, 2D NaN-padded)
- `TestComputeViewRatesParameters`: 5 tests (gaze_model, view_distance, smoothing_method, bandwidth, n_jobs)
- `TestComputeViewRatesNeuronIteration`: 3 tests (len, getitem, iteration)
- `TestComputeViewRatesEdgeCases`: 3 tests (empty list, single neuron, neuron with no spikes)
- `TestComputeViewRatesInputValidation`: 3 tests (invalid gaze_model, mismatched times/positions, mismatched times/headings)
- `TestComputeViewRatesConsistencyWithSingle`: 1 test (single neuron matches compute_view_rate)
- `TestComputeViewRatesSignature`: 2 tests (canonical argument order, keyword-only params)

**Efficiency advantages over calling `compute_view_rate()` in a loop**:

1. View occupancy computed once and shared across all neurons
2. Diffusion kernel computed once for batch smoothing
3. Viewed-bin mapping done once
4. Spike binning can be parallelized with joblib (n_jobs parameter)

**Milestone 4 Progress**: Tasks 4.1-4.8 complete, 1 task remaining (4.9).

---

### Task 4.7: Implement `compute_view_rate()` function [COMPLETED]

**Goal**: Create the single-neuron view rate computation function that computes spatial view fields from spike trains and trajectory data.

**Approach**: TDD - wrote 34 tests first, then implemented.

**Result**:

- Added `compute_view_rate()` to `src/neurospatial/encoding/view.py`
- Created `tests/encoding/test_compute_view_rate.py` with 34 tests (all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `compute_view_rate(env, spike_times, times, positions, headings, *, gaze_model, view_distance, smoothing_method, bandwidth, min_occupancy)`:
  - Parameters follow canonical argument order from CLAUDE.md (headings required for egocentric)
  - Uses view binning layer (`_view_binning.py`) to bin spikes by *viewed* location
  - Uses smoothing layer (`_smoothing.py`) to compute smoothed firing rate
  - Returns `ViewRateResult` with all required fields
  - Validates gaze_model parameter (fixed_distance, ray_cast, boundary)
  - Validates input array lengths (times, positions, headings must match)

**Signature**:

```python
def compute_view_rate(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    gaze_model: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance",
    view_distance: float = 10.0,
    smoothing_method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy: float = 0.0,
) -> ViewRateResult:
```

**Test coverage (34 tests in 11 classes)**:

- `TestComputeViewRateImport`: 2 tests (import from module and package)
- `TestComputeViewRateReturnsResult`: 4 tests (return type, shapes, env)
- `TestComputeViewRateGazeModel`: 4 tests (valid models, default, invalid raises)
- `TestComputeViewRateViewDistance`: 2 tests (storage, default)
- `TestComputeViewRateSmoothing`: 5 tests (methods, default, bandwidth storage)
- `TestComputeViewRateEmptySpikes`: 2 tests (zero rate, positive occupancy)
- `TestComputeViewRateCorrectness`: 3 tests (non-negative rate, non-negative occupancy, view vs spatial occupancy)
- `TestComputeViewRateResultMethods`: 4 tests (plot, peak_view_location, view_spatial_information, is_view_cell)
- `TestComputeViewRateMinOccupancy`: 2 tests (threshold masking, default)
- `TestComputeViewRateInputValidation`: 2 tests (mismatched times/positions, mismatched times/headings)
- `TestComputeViewRateSignature`: 2 tests (canonical argument order, keyword-only params)

**Key difference from `compute_spatial_rate()`**:

- Requires `headings` parameter for gaze computation
- Uses view binning (where animal looked) instead of spatial binning (where animal was)
- Uses `view_occupancy` instead of `occupancy`
- Additional parameters: `gaze_model`, `view_distance`

**Code review feedback addressed**:

1. Added input length validation for better error messages
2. Fixed docstring example to use `positions` instead of `trajectory` for consistency with CLAUDE.md

**Documentation**:

- Complete NumPy-style docstring with:
  - Extended description explaining view fields vs place fields
  - Full parameter documentation with gaze model explanations
  - Notes section on algorithm and place vs view cell differences
  - Working examples
  - Scientific references (Rolls et al., 1997)

**Milestone 4 Progress**: Tasks 4.1-4.7 complete, 2 tasks remaining (4.8-4.9).

---

### Task 4.6: Implement Binning Layer for View Encoding [COMPLETED]

**Goal**: Create binning layer that computes view occupancy (time viewing each spatial bin) and bins spikes by viewed location.

**Approach**: TDD - wrote 41 tests first, then implemented.

**Result**:

- Created `src/neurospatial/encoding/_view_binning.py` with 3 functions
- Created `tests/encoding/test_encoding_view_binning.py` with 41 tests (all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `compute_view_occupancy(env, times, positions, headings, gaze_model, view_distance)`: Compute view occupancy
  - Returns `(n_bins,)` float64 array of time spent *viewing* each bin
  - Delegates to `compute_viewed_location()` from `ops/visibility.py` for gaze computation
  - Accumulates time per viewed bin using `np.add.at()`
  - Views outside environment don't contribute to occupancy

- `bin_view_spike_train(env, spike_times, times, positions, headings, gaze_model, view_distance)`: Single neuron
  - Returns `(n_bins,)` float64 array of spike counts by viewed location
  - Uses nearest-neighbor lookup (`np.searchsorted`) for spike→frame mapping
  - Does NOT use interpolation for gaze direction (scientifically correct)
  - Empty spike trains return zeros

- `bin_view_spike_trains(env, spike_times, times, positions, headings, gaze_model, view_distance, n_jobs)`: Batch
  - Returns tuple: `(spike_counts, view_occupancy)` where:
    - spike_counts: `(n_neurons, n_bins)` float64
    - view_occupancy: `(n_bins,)` float64 (shared across neurons)
  - Precomputes view_occupancy once (shared across all neurons)
  - Supports joblib parallelization via `n_jobs` parameter
  - Normalizes input via `normalize_spike_times()` for flexible formats

**Gaze models supported**:

- `"fixed_distance"` (default): Point at fixed distance in gaze direction
- `"ray_cast"`: Intersection with environment boundary
- `"boundary"`: Nearest boundary point in gaze direction

**Test coverage (41 tests in 9 classes)**:

- `TestComputeViewOccupancy`: 6 tests (basic functionality, shape, non-negative, duration, differs from position)
- `TestComputeViewOccupancyGazeModels`: 6 tests (all gaze models, view_distance, different results)
- `TestBinViewSpikeTrain`: 7 tests (basic, shape, non-negative, count, empty, outside range)
- `TestBinViewSpikeTrainGazeModels`: 3 tests (all gaze models)
- `TestBinViewSpikeTrains`: 10 tests (tuple return, shapes, dtypes, empty neuron, consistency, n_jobs, normalization)
- `TestViewBinningEdgeCases`: 3 tests (constant heading, view outside env, all spikes same location)
- `TestViewBinningInputValidation`: 4 tests (mismatched lengths, invalid gaze_model, insufficient samples)
- `TestConsistencyWithSpatialView`: 1 critical test (view occupancy matches existing `compute_spatial_view_field`)

**Consistency verification**:

- `test_view_occupancy_matches_spatial_view_implementation()` proves new binning layer produces identical view_occupancy to existing implementation in `spatial_view.py` using `np.testing.assert_array_almost_equal`, decimal=10

**Code review feedback**: APPROVE - Production-ready code

- Scientifically correct (nearest-neighbor lookup, not interpolation)
- Well-documented (NumPy-style docstrings throughout)
- Thoroughly tested (41 tests including critical consistency test)
- Consistent with existing patterns (`_binning.py`, `_directional_binning.py`)
- Type-safe (mypy passes)
- Performant (batch optimization with shared view_occupancy)

**Milestone 4 Progress**: Tasks 4.1-4.6 complete, 3 tasks remaining (4.7-4.9).

---

### Task 4.5: Implement `ViewRatesResult.to_dataframe()` [COMPLETED]

**Goal**: Add `to_dataframe()` method to `ViewRatesResult` for exporting metrics to pandas DataFrame.

**Approach**: TDD - wrote 18 tests first, then implemented.

**Result**:

- Added 1 method to `ViewRatesResult` in `src/neurospatial/encoding/view.py`
- Added 18 new tests in `tests/encoding/test_encoding_view.py` (total now 77, all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `to_dataframe(neuron_ids=None)`: Export view cell metrics to DataFrame
  - Returns pandas DataFrame with columns:
    - `neuron_id`: User-provided or auto-generated integer indices
    - `peak_view_x`: X-coordinate of peak view location
    - `peak_view_y`: Y-coordinate of peak view location
    - `peak_rate`: Maximum firing rate (Hz)
    - `view_spatial_info`: View spatial information (bits/spike)
    - `is_view_cell`: Classification based on default threshold (0.5 bits/spike)
  - Validates `neuron_ids` length matches number of neurons
  - Handles empty results (0 neurons) gracefully
  - Follows pattern from `DirectionalRatesResult.to_dataframe()` and `SpatialRatesResult.to_dataframe()`

**Code review recommendations addressed**:

1. Used consistent error message format matching other `to_dataframe()` implementations
2. Added explicit type annotation `neuron_ids_list: list[str | int]` for clarity
3. Added explicit type annotation `data: dict[str, Any]` for the data dictionary
4. Simplified empty array handling (removed redundant conditionals for `peak_view_x/y`)

**Test coverage (18 tests in 1 class)**:

- Column existence tests (6): neuron_id, peak_view_x, peak_view_y, peak_rate, view_spatial_info, is_view_cell
- Correctness validation (5): peak_view_x, peak_view_y, peak_rate, view_spatial_info, is_view_cell match batch methods
- Neuron ID handling (3): default indices, custom IDs, length mismatch error
- Edge cases (2): empty result (0 neurons), single neuron
- Basic functionality (2): returns DataFrame, correct row count

**Documentation**:

- Complete NumPy-style docstring with Parameters, Returns, Raises, Notes, Examples, See Also sections
- Common pandas workflows documented (filter, sort, top-N)
- Cross-references to related batch methods

**Milestone 4 Progress**: Tasks 4.1-4.5 complete, 4 tasks remaining (4.6-4.9).

---

### Task 4.4: Implement `ViewRatesResult` Batch Methods [COMPLETED]

**Goal**: Add batch methods to `ViewRatesResult` for efficient population analysis of spatial view cells.

**Approach**: TDD - wrote 19 tests first, then implemented.

**Result**:

- Added 4 batch methods to `ViewRatesResult` in `src/neurospatial/encoding/view.py`
- Added 19 new tests in `tests/encoding/test_encoding_view.py` (total now 59, all pass)
- All mypy and ruff checks pass
- Code review passed after addressing recommendations

**Key Implementation Details**:

- `plot(idx, ax, **kwargs)`: Plot a specific neuron's view field
  - Requires `idx` parameter (0-indexed)
  - Delegates to `env.plot_field()` with the neuron's firing rate
  - Accepts `ax` and kwargs passed through

- `peak_view_locations()`: Peak view locations for all neurons
  - Returns `(n_neurons, n_dims)` array of coordinates
  - Uses `np.nanargmax()` to handle NaN values
  - Returns NaN coordinates for neurons with all-NaN firing rates (edge case)

- `view_spatial_information()`: View spatial information for all neurons
  - Returns `(n_neurons,)` array in bits/spike
  - Delegates to `batch_spatial_information()` from `_metrics.py`
  - Uses `view_occupancy` (not standard occupancy)

- `detect_view_cells(min_info=0.5)`: Classify neurons as spatial view cells
  - Returns `(n_neurons,)` boolean array
  - Uses vectorized comparison: `info > min_info`
  - Default threshold 0.5 bits/spike from Rolls et al. (1997)

**Code review recommendations addressed**:

1. Added NaN validation to `peak_view_locations()` - returns NaN coordinates for all-NaN neurons
2. Vectorized `detect_view_cells()` for performance with large populations

**Test coverage (19 tests in 5 classes)**:

- `TestViewRatesResultPlot`: 4 tests (returns axes, requires idx, accepts ax, accepts kwargs)
- `TestViewRatesResultPeakViewLocations`: 5 tests (returns ndarray, shape, matches single, handles NaN, all-NaN edge case)
- `TestViewRatesResultViewSpatialInformation`: 5 tests (returns ndarray, shape, matches single, non-negative, uniform=0)
- `TestViewRatesResultDetectViewCells`: 5 tests (returns ndarray, shape, matches single, respects min_info, default threshold)

**Documentation**:

- Complete NumPy-style docstrings for all 4 methods
- Examples showing typical usage
- See Also cross-references to single-neuron methods
- Notes sections explaining view occupancy semantics

**Milestone 4 Progress**: Tasks 4.1-4.4 complete, 5 tasks remaining (4.5-4.9).

---

### Task 4.3: Implement `ViewRateResult` Classification [COMPLETED]

**Goal**: Add classification method to `ViewRateResult` for determining if a neuron is a spatial view cell.

**Approach**: TDD - wrote 6 tests first, then implemented.

**Result**:

- Added 1 classification method to `ViewRateResult` in `src/neurospatial/encoding/view.py`
- Added 6 new tests in `tests/encoding/test_encoding_view.py` (total now 40, all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `is_view_cell(min_info=0.5)`: Boolean classification
  - Returns True if view_spatial_information() > min_info
  - Default threshold 0.5 bits/spike from Rolls et al. (1997)
  - Single criterion (spatial info only) - unlike HD cells which use MVL + Rayleigh
  - Simpler than HD classification because view fields lack circular structure

**Test coverage (6 tests in 1 class)**:

- `TestViewRateResultIsViewCell`: 6 tests
  - Returns bool
  - True for high spatial info neurons
  - False for uniform firing
  - Respects min_info parameter
  - Default threshold (0.5)
  - Uses view_spatial_information() for classification

**Documentation**:

- Complete NumPy-style docstring with:
  - Parameter documentation with threshold rationale
  - "How was 0.5 chosen?" section with Rolls et al. (1997) reference
  - "When to adjust" guidance for different use cases
  - Notes comparing to HD cell classification
  - References to original papers
  - Examples with expected output
  - See Also cross-references

**Code review feedback**: APPROVE - Ready to merge

- Excellent documentation quality with educational value
- Clean delegation pattern to view_spatial_information()
- Comprehensive test coverage
- Scientifically correct threshold comparison (>)

**Milestone 4 Progress**: Tasks 4.1-4.3 complete, 6 tasks remaining (4.4-4.9).

---

### Task 4.2: Implement `ViewRateResult` Convenience Methods [COMPLETED]

**Goal**: Add convenience methods to `ViewRateResult` for plotting and basic metrics.

**Approach**: TDD - wrote 12 tests first, then implemented.

**Result**:

- Added 3 convenience methods to `ViewRateResult` in `src/neurospatial/encoding/view.py`
- Added 12 new tests in `tests/encoding/test_encoding_view.py` (total now 34, all pass)
- All mypy and ruff checks pass
- Code review passed after fixing type annotation issue

**Key Implementation Details**:

- `plot(ax, **kwargs)`: Visualize view field
  - Delegates to `env.plot_field()` for consistent visualization
  - Uses `_to_numpy()` for JAX compatibility
  - Accepts optional `ax` argument and passes through kwargs

- `peak_view_location()`: Location of peak view response
  - Returns `(n_dims,)` coordinates of bin with maximum firing rate
  - Uses `np.nanargmax()` to handle NaN values
  - Explicitly typed result for mypy compatibility

- `view_spatial_information()`: Skaggs spatial info based on view occupancy
  - Delegates to `spatial_information()` from `_metrics.py`
  - Uses `view_occupancy` instead of standard `occupancy` (key difference for view cells)
  - Returns float, always non-negative

**Design differences from SpatialRateResult**:

- Method names include "view" prefix (e.g., `peak_view_location` vs `peak_location`)
- `view_spatial_information()` uses `view_occupancy` (time *viewing* each bin) instead of `occupancy` (time *at* each bin)
- No `SpatialResultMixin` inheritance (view cells have different semantics)

**Test coverage (12 tests in 3 classes)**:

- `TestViewRateResultPlot`: 3 tests (returns axes, accepts ax, passes kwargs)
- `TestViewRateResultPeakViewLocation`: 4 tests (returns ndarray, shape, correctness, NaN handling)
- `TestViewRateResultViewSpatialInformation`: 5 tests (returns float, non-negative, uses view_occupancy, uniform=0, peaked>0)

**Code review feedback addressed**:

1. Fixed type annotation in `peak_view_location()` - added explicit `result: NDArray[np.float64]` annotation for mypy

**Milestone 4 Progress**: Tasks 4.1-4.2 complete, 7 tasks remaining (4.3-4.9).

---

### Task 4.1: Create `encoding/view.py` with Result Class Definitions [COMPLETED]

**Goal**: Create the result class definitions for view rate computation (spatial view cells).

**Approach**: TDD - wrote 22 tests first (`test_encoding_view.py`), then implemented.

**Result**:

- Created `src/neurospatial/encoding/view.py` with 2 dataclasses
- Created `tests/encoding/test_encoding_view.py` with 22 tests (all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `ViewRateResult`: Single-neuron result (frozen dataclass)
  - Fields: `firing_rate`, `view_occupancy`, `env`, `gaze_model`, `view_distance`, `smoothing_method`, `bandwidth`
  - Uses `view_occupancy` instead of `occupancy` (time *viewing* each bin, not time *at* each bin)
  - No Environment mixin inheritance (different semantics from spatial results)

- `ViewRatesResult`: Batch result (frozen dataclass)
  - Fields: `firing_rates` (plural), `view_occupancy`, `env`, `gaze_model`, `view_distance`, `smoothing_method`, `bandwidth`
  - Implements `__len__`, `__getitem__`, `__iter__` for iteration
  - `__getitem__(idx)` returns `ViewRateResult` for single neuron

**Design differences from SpatialRateResult**:

- Uses `view_occupancy` instead of `occupancy` (semantic difference)
- No `SpatialResultMixin` inheritance (will add `peak_view_location()` in Task 4.2)
- Additional fields: `gaze_model`, `view_distance`

**Test coverage (22 tests in 9 classes)**:

- `TestViewRateResultImport`: 2 tests (import from module and package)
- `TestViewRateResultCreation`: 2 tests (basic, frozen)
- `TestViewRateResultFields`: 3 tests (shapes, required fields)
- `TestViewRatesResultImport`: 2 tests (import from module and package)
- `TestViewRatesResultCreation`: 2 tests (basic, frozen)
- `TestViewRatesResultFields`: 3 tests (shapes, required fields)
- `TestViewRatesResultIteration`: 6 tests (len, getitem, iter, metadata, order)
- `TestViewRatesResultEdgeCases`: 2 tests (single neuron, empty)

**Exports added**:

- `ViewRateResult` and `ViewRatesResult` exported from `encoding/__init__.py`
- Also added missing `SpatialRateResult` and `SpatialRatesResult` exports

**Code review feedback**:

- Reviewer suggested adding `SpatialResultMixin` inheritance
- Decided NOT to add it because:
  - View results use `view_occupancy` semantics (different from `occupancy`)
  - Will add `peak_view_location()` method (not `peak_location()`) in Task 4.2
  - Naming convention intentionally different to distinguish view vs position

**Milestone 4 Progress**: Task 4.1 complete, 8 tasks remaining (4.2-4.9).

---

### Task 3.11: Write Comprehensive Tests for Directional Encoding [COMPLETED]

**Goal**: Verify comprehensive test coverage for all directional encoding functionality, including metric equivalence with legacy `HeadDirectionMetrics`.

**Approach**: TDD - verified existing coverage, added 10 new tests for metric comparison.

**Result**:

- Added `TestDirectionalRateResultMatchesHeadDirectionMetrics` class with 10 tests
- Total test count: 187 tests in `tests/encoding/test_encoding_directional.py` (all pass)
- All ruff checks pass
- No new mypy errors introduced (pre-existing errors in other tests)

**Key Implementation Details**:

- **No deprecated shim needed**: User confirmed backwards compatibility not required
- All single neuron computation tests: Already covered from TDD in Tasks 3.1-3.4
- All batch computation tests: Already covered from TDD in Tasks 3.5-3.9
- All result class method tests: Already covered
- `angle_unit` conversion tests: Already covered

**New Test Coverage (10 tests)**:

- `test_preferred_direction_matches`: Verifies new API matches legacy
- `test_mean_vector_length_matches`: Verifies MVL matches exactly (rtol=1e-10)
- `test_peak_firing_rate_matches`: Verifies peak rate matches exactly
- `test_rayleigh_pvalue_matches`: Verifies p-value matches exactly
- `test_is_hd_cell_matches`: Verifies HD classification matches for sharply tuned
- `test_is_hd_cell_matches_uniform`: Verifies HD classification matches for non-HD
- `test_tuning_width_same_order_of_magnitude`: Verifies tuning width within 50%
- `test_preferred_direction_deg_matches`: Verifies degree conversion matches
- `test_interpretation_contains_key_info_for_hd_cell`: Verifies output format
- `test_interpretation_contains_key_info_for_non_hd_cell`: Verifies output format

**Milestone 3 Status**: COMPLETE! All 11 tasks finished (3.1-3.9, 3.11).

---

### Task 3.9: Implement `compute_directional_rates()` function [COMPLETED]

**Goal**: Create the batch directional rate computation function for multiple neurons.

**Approach**: TDD - wrote 36 tests first, then implemented.

**Result**:

- Added `compute_directional_rates()` to `src/neurospatial/encoding/directional.py`
- Added 36 tests to `tests/encoding/test_encoding_directional.py` (all 181 tests pass)
- All mypy and ruff checks pass
- Code review passed after addressing type annotation issues

**Key Implementation Details**:

- `compute_directional_rates(spike_times, times, headings, bin_size, smoothing_sigma, angle_unit, n_jobs)`:
  - Accepts multiple input formats: list of arrays, tuple, 2D array with NaN padding, single 1D array
  - Normalizes input via `normalize_spike_times()` from `_spikes.py`
  - Precomputes shared quantities (occupancy, bin_centers) once
  - Uses nested `_process_neuron()` helper for per-neuron processing
  - Supports parallelization via joblib `n_jobs` parameter
  - Returns `DirectionalRatesResult` with all required fields

**Efficiency advantages over calling `compute_directional_rate()` in a loop**:

1. Occupancy is computed once and shared across all neurons
2. Bin centers are computed once
3. Spike binning can be parallelized with joblib

**Test Coverage (36 tests in 13 classes)**:

- `TestComputeDirectionalRatesImport`: 2 tests (import from module and package)
- `TestComputeDirectionalRatesBasic`: 7 tests (return type, shapes, metadata)
- `TestComputeDirectionalRatesInputFormats`: 4 tests (list, tuple, 2D array, 1D array)
- `TestComputeDirectionalRatesNJobs`: 3 tests (parallelization and consistency)
- `TestComputeDirectionalRatesAngleUnit`: 3 tests (rad, deg, equivalence)
- `TestComputeDirectionalRatesSmoothing`: 2 tests (storage, smoother curves)
- `TestComputeDirectionalRatesConsistency`: 2 tests (**CRITICAL** - batch matches single-neuron, shared occupancy)
- `TestComputeDirectionalRatesEdgeCases`: 4 tests (empty spikes, single neuron, empty list)
- `TestComputeDirectionalRatesResultMethods`: 4 tests (preferred_directions, mean_vector_lengths, detect_hd_cells, to_dataframe)
- `TestComputeDirectionalRatesInputValidation`: 1 test (invalid angle_unit)

**Consistency verification**:

- `test_batch_matches_single_neuron_results()` proves batch produces identical results to processing each neuron individually with `compute_directional_rate()` using `decimal=10`

**Code Review Feedback Addressed**:

1. Fixed type annotation for `firing_rate` variable in `_process_neuron()` helper
2. Added explicit type annotations for `spike_counts_smooth`, `occupancy_smooth`
3. Added explicit type annotation for `spike_times_list` after normalization
4. Changed condition to use `smoothing_sigma_rad` consistently instead of `smoothing_sigma`

**Documentation**:

- Complete NumPy-style docstring with:
  - Extended description explaining efficiency advantages
  - Full parameter documentation
  - Notes section on when to use batch vs single
  - Working examples with iteration and parallelization

---

### Task 3.8: Implement `compute_directional_rate()` function [COMPLETED]

**Goal**: Create the single-neuron directional rate computation function.

**Approach**: TDD - wrote 27 tests first, then implemented.

**Result**:

- Added `compute_directional_rate()` to `src/neurospatial/encoding/directional.py`
- Added 27 tests to `tests/encoding/test_encoding_directional.py` (all 145 tests pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `compute_directional_rate(spike_times, times, headings, bin_size, smoothing_sigma, angle_unit)`:
  - Uses binning layer to compute spike counts and occupancy
  - Optionally applies circular Gaussian smoothing via `scipy.ndimage.gaussian_filter1d` with `mode='wrap'`
  - Smoothing is applied to spike counts and occupancy **separately** before division (scientifically correct)
  - Returns `DirectionalRateResult` with firing_rate, occupancy, bin_centers, bin_size, smoothing_sigma
  - All angular values stored in radians internally (even when input is degrees)

**Test Coverage**:

- Basic functionality (return type, shapes, metadata)
- Parameter variations (bin_size, angle_unit, smoothing_sigma)
- Edge cases (empty spikes, single spike, spikes outside time range, constant heading)
- Result methods (preferred_direction, mean_vector_length, is_hd_cell, plot)
- Input validation (invalid angle_unit)

**Design Pattern**: Follows `compute_spatial_rate()` pattern closely:

- Similar docstring structure
- Similar smoothing approach
- Returns immutable frozen dataclass

---

### Task 3.7: Implement Binning Layer for Directional Encoding [COMPLETED]

**Goal**: Create helper functions to convert (spike_times, times, headings, bin_size) → (spike_counts, occupancy) for directional encoding.

**Approach**: TDD - wrote 36 tests first, then implemented.

**Result**:

- Created `src/neurospatial/encoding/_directional_binning.py` with 3 functions
- Created `tests/encoding/test_encoding_directional_binning.py` with 36 tests (all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `bin_directional_spike_train(spike_times, times, headings, bin_size, angle_unit)`: Single neuron spike binning
  - Assigns spikes to angular bins using nearest-neighbor lookup (not interpolation)
  - Handles circular 0/2π boundary correctly
  - Returns `(n_bins,)` float64 array of spike counts
  - Supports `angle_unit` parameter ('rad' or 'deg')

- `compute_directional_occupancy(times, headings, bin_size, angle_unit)`: Compute time spent at each direction
  - Uses actual time deltas between frames (handles variable sampling)
  - Returns tuple: `(occupancy, bin_centers)`
    - occupancy: `(n_bins,)` float64 in seconds
    - bin_centers: `(n_bins,)` float64 in radians [0, 2π)
  - Validates input shapes, monotonicity, and minimum sample count

- `bin_directional_spike_trains(spike_times, times, headings, bin_size, angle_unit, n_jobs)`: Batch version
  - Normalizes spike times via `normalize_spike_times()` for flexible input formats
  - Returns tuple: `(spike_counts, occupancy, bin_centers)`
    - spike_counts: `(n_neurons, n_bins)` float64
    - occupancy: `(n_bins,)` float64 (shared across neurons)
    - bin_centers: `(n_bins,)` float64 in radians
  - Supports parallelization via joblib `n_jobs` parameter

**Design differences from spatial binning**:

- No `env` parameter (head direction is 1D circular, not spatial)
- Returns `bin_centers` in radians (not coordinate bins)
- Uses nearest-neighbor spike assignment (not interpolation) to handle circular discontinuity
- Input headings can be in radians or degrees (via `angle_unit` parameter)

**Test coverage (36 tests in 8 classes)**:

- `TestBinDirectionalSpikeTrain`: 7 tests (basic, shape, non-negative, count, empty, outside range)
- `TestBinDirectionalSpikeTrainAngleUnit`: 3 tests (rad default, deg, equivalence)
- `TestComputeDirectionalOccupancy`: 8 tests (tuple return, shapes, range, dtype, non-negative, duration, angle_unit)
- `TestBinDirectionalSpikeTrains`: 10 tests (tuple return, shapes, dtype, empty neuron, consistency, n_jobs, single neuron, normalization)
- `TestDirectionalBinningEdgeCases`: 3 tests (all spikes at one direction, uniform heading, wrap-around)
- `TestDirectionalBinningInputValidation`: 4 tests (mismatched lengths, insufficient samples, non-monotonic times, invalid angle_unit)

**Documentation**:

- Complete NumPy-style docstrings with parameter descriptions
- Module docstring explains output shapes and layer purpose
- Examples show typical usage patterns
- Notes explain circular binning and spike assignment algorithms

**Code review feedback addressed**:

1. Added validation for `angle_unit` in `bin_directional_spike_train()` (was only in `compute_directional_occupancy()`)
2. Added test `test_invalid_angle_unit_spike_train` to cover this validation

**Circular binning correctness**:

- Headings wrapped to [0, 2π) before binning
- Nearest-neighbor lookup for spikes avoids circular interpolation issues (e.g., 350° to 10° would incorrectly interpolate to 180° with linear interpolation)
- Edge case: value exactly at 2π is wrapped to bin 0

---

### Task 3.6: Implement `DirectionalRatesResult.to_dataframe()` [COMPLETED]

**Goal**: Add `to_dataframe()` method to export directional metrics to pandas DataFrame.

**Approach**: TDD - wrote 22 tests first, then implemented.

**Result**:

- Added `to_dataframe()` method to `DirectionalRatesResult` in `src/neurospatial/encoding/directional.py`
- Added 22 new tests in `tests/encoding/test_encoding_directional.py` (total now 118, all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `to_dataframe(neuron_ids)`: Export to pandas DataFrame
  - Parameters:
    - `neuron_ids`: Optional custom identifiers (default: integer indices)
  - Returns DataFrame with columns:
    - neuron_id: identifier for each neuron
    - preferred_direction: preferred direction in radians [-π, π]
    - preferred_direction_deg: preferred direction in degrees [-180, 180]
    - mean_vector_length: mean vector length [0, 1]
    - tuning_width: tuning width (HWHM) in radians (0, π]
    - tuning_width_deg: tuning width (HWHM) in degrees (0, 180]
    - peak_rate: maximum firing rate (Hz)
    - is_hd_cell: whether classified as HD cell (using default thresholds)

- Delegates to existing batch methods:
  - `preferred_directions()` for preferred_direction
  - `mean_vector_lengths()` for mean_vector_length
  - `tuning_widths()` for tuning_width
  - `peak_firing_rates()` for peak_rate
  - `detect_hd_cells()` for is_hd_cell
  - `np.degrees()` for degree conversions

**Test coverage (22 tests in 1 class)**:

- `TestDirectionalRatesResultToDataframe`: 22 tests
  - Returns DataFrame type
  - Correct row count
  - All column presence tests (8 columns)
  - Default integer neuron IDs
  - Custom neuron IDs
  - Length mismatch validation
  - Metric accuracy (matches batch methods)
  - Degree conversions correct
  - Empty result edge case
  - Single neuron edge case

**Documentation**:

- Complete NumPy-style docstring
- Includes common pandas workflows (filter, sort, top-N)
- Includes practical examples
- Cross-references to related methods

**Code review feedback**: APPROVE - Ready to merge

- Perfect alignment with reference pattern (`SpatialRatesResult.to_dataframe()`)
- Complete type safety with `Sequence[str | int]` type annotation
- Comprehensive test coverage (22 tests)
- Proper pandas import pattern (inside function)

**Type annotation fix**:

Changed `list[str | int]` to `Sequence[str | int]` per code review suggestion for consistency with `SpatialRatesResult.to_dataframe()`.

---

### Task 3.5: Implement `DirectionalRatesResult` Batch Methods [COMPLETED]

**Goal**: Add batch methods to `DirectionalRatesResult` for efficient population analysis.

**Approach**: TDD - wrote 28 tests first, then implemented.

**Result**:

- Added 6 batch methods to `DirectionalRatesResult` in `src/neurospatial/encoding/directional.py`
- Added 28 new tests in `tests/encoding/test_encoding_directional.py` (total now 96, all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `plot(idx, ax, polar, **kwargs)`: Plot a specific neuron's tuning curve
  - Requires `idx` parameter (0-indexed)
  - Delegates to `self[idx].plot()` for the actual plot
  - Accepts `ax`, `polar`, and kwargs passed through

- `preferred_directions()`: Preferred directions for all neurons
  - Returns `(n_neurons,)` array in radians [-π, π]
  - Delegates to `circular_mean()` from `neurospatial.stats.circular`
  - Uses explicit loop (vectorization possible in future)

- `mean_vector_lengths()`: Mean vector lengths for all neurons
  - Returns `(n_neurons,)` array in [0, 1]
  - Delegates to `mean_resultant_length()` from `neurospatial.stats.circular`
  - Uses explicit loop (vectorization possible in future)

- `tuning_widths()`: Tuning widths (HWHM) for all neurons
  - Returns `(n_neurons,)` array in radians (0, π], or NaN
  - Delegates to `self[i].tuning_width()` for each neuron

- `peak_firing_rates()`: Peak firing rates for all neurons
  - Returns `(n_neurons,)` array in Hz
  - Uses vectorized `np.nanmax(rates, axis=1)` for efficiency

- `detect_hd_cells(min_mvl, alpha)`: HD cell classification for all neurons
  - Returns `(n_neurons,)` boolean array
  - Parameters: min_mvl (default 0.4), alpha (default 0.05)
  - Delegates to `self[i].is_hd_cell()` for each neuron

**Test coverage (28 tests in 6 classes)**:

- `TestDirectionalRatesResultPlot`: 6 tests (returns axes, requires idx, polar default, cartesian, existing axes, kwargs)
- `TestDirectionalRatesResultPreferredDirections`: 4 tests (returns array, shape, matches single, valid range)
- `TestDirectionalRatesResultMeanVectorLengths`: 4 tests (returns array, shape, matches single, valid range)
- `TestDirectionalRatesResultTuningWidths`: 4 tests (returns array, shape, matches single, valid range)
- `TestDirectionalRatesResultDetectHdCells`: 6 tests (returns array, shape, matches single, respects min_mvl, respects alpha, default thresholds)
- `TestDirectionalRatesResultPeakFiringRates`: 4 tests (returns array, shape, matches single, non-negative)

**Documentation**:

- Complete NumPy-style docstrings for all methods
- Examples showing typical usage
- See Also cross-references to single-neuron methods

**Code review feedback**: APPROVE - Ready to merge

- Excellent code quality with consistent implementation pattern
- Complete type annotations (mypy passes)
- Comprehensive test coverage (28 tests)
- One smart vectorization (`peak_firing_rates`)
- Performance optimization opportunity noted for future: 4 methods use loops instead of vectorized operations

---

### Task 3.4: Implement `DirectionalRateResult` Classification [COMPLETED]

**Goal**: Add classification methods to `DirectionalRateResult` for determining if a neuron is a head direction cell and providing human-readable interpretation.

**Approach**: TDD - wrote 13 tests first, then implemented.

**Result**:

- Added 2 classification methods to `DirectionalRateResult` in `src/neurospatial/encoding/directional.py`
- Added 13 new tests in `tests/encoding/test_encoding_directional.py` (total now 68, all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `is_hd_cell(min_mvl=0.4, alpha=0.05)`: Boolean classification
  - Returns True if MVL > min_mvl AND rayleigh_pvalue < alpha
  - Default thresholds from Taube et al. (1990)
  - Both criteria must be met (AND logic)
  - Reuses existing `mean_vector_length()` and `rayleigh_pvalue()` methods

- `interpretation(min_mvl=0.4)`: Human-readable string
  - For HD cells: Shows header, preferred direction, MVL (with threshold), peak rate, tuning width, Rayleigh p-value
  - For non-HD cells: Explains which criteria failed with educational guidance
  - Includes threshold selection rationale from Taube et al. (1990)
  - Format matches existing `HeadDirectionMetrics.interpretation()` in `head_direction.py`

**Test coverage (13 tests in 2 classes)**:

- `TestDirectionalRateResultIsHdCell`: 7 tests
  - Returns bool
  - True for sharply tuned neurons
  - False for uniform firing
  - Respects min_mvl parameter
  - Respects alpha parameter
  - Default thresholds (0.4, 0.05)
  - Requires both criteria

- `TestDirectionalRateResultInterpretation`: 6 tests
  - Returns string
  - HD cell format (header, metrics)
  - Non-HD cell format (explains why)
  - Explains low MVL
  - Respects min_mvl parameter
  - Includes threshold value

**Documentation**:

- Complete NumPy-style docstrings with:
  - Parameter descriptions with threshold rationale
  - "How was 0.4 chosen?" section with Taube et al. (1990) reference
  - "When to adjust" guidance for different brain regions/species
  - Examples with expected output
  - See Also cross-references

**API Consistency**:

- Method signatures match existing `HeadDirectionMetrics` API
- Output format identical to `HeadDirectionMetrics.interpretation()`
- Same educational messaging about threshold selection

**Code review feedback**: APPROVE - No critical issues

- Perfect API consistency with legacy HeadDirectionMetrics
- Outstanding documentation with biological context
- Robust implementation with proper boolean logic
- Comprehensive test coverage

---

### Task 3.3: Implement `DirectionalRateResult` Tuning Metrics [COMPLETED]

**Goal**: Add tuning metric methods to `DirectionalRateResult` for quantifying head direction cell tuning.

**Approach**: TDD - wrote 16 tests first, then implemented.

**Result**:

- Added 4 tuning metric methods to `DirectionalRateResult` in `src/neurospatial/encoding/directional.py`
- Added 16 new tests in `tests/encoding/test_encoding_directional.py` (total now 55, all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `mean_vector_length()`: Rayleigh mean vector length
  - Returns float in [0, 1], higher = sharper tuning
  - Delegates to `mean_resultant_length()` from `neurospatial.stats.circular`
  - Uses firing rates as weights for bin centers
  - Interpretation: MVL < 0.2 weak, 0.2-0.4 moderate, > 0.4 strong (Taube et al., 1990)

- `tuning_width()`: Half-width at half-maximum (HWHM) in radians
  - Returns float in (0, π], or NaN for flat tuning curves
  - Custom algorithm: finds peak, searches both directions for half-max crossing
  - Uses linear interpolation for precise crossing location
  - Circular indexing for wraparound handling

- `tuning_width_deg()`: Convenience wrapper returning degrees
  - Returns float in (0, 180], or NaN
  - Simple conversion: `np.degrees(self.tuning_width())`

- `rayleigh_pvalue()`: Rayleigh test p-value for non-uniformity
  - Returns float in [0, 1], lower = more significant tuning
  - Delegates to `rayleigh_test()` from `neurospatial.stats.circular`
  - Uses firing rates as weights
  - Interpretation: p < 0.05 = significant directional tuning

**Test coverage (16 tests in 4 classes)**:

- `TestDirectionalRateResultMeanVectorLength`: 5 tests (returns float, valid range, tuned vs uniform, uniform low, sharply tuned high)
- `TestDirectionalRateResultTuningWidth`: 4 tests (returns float, valid range, sharp < broad, uniform NaN)
- `TestDirectionalRateResultTuningWidthDeg`: 3 tests (returns float, conversion, valid range)
- `TestDirectionalRateResultRayleighPvalue`: 4 tests (returns float, valid range, tuned low, uniform high)

**Documentation**:

- All methods have complete NumPy-style docstrings
- Include algorithm descriptions and formulas
- Include interpretation guidelines with specific thresholds
- Include See Also cross-references to related methods and stats.circular

**Code review feedback**: APPROVE

- Excellent integration with `stats.circular` module
- Comprehensive docstrings with biological context
- Edge case handling (NaN for flat curves)
- Correct circular indexing for HWHM computation

---

### Task 3.2: Implement `DirectionalRateResult` Convenience Methods [COMPLETED]

**Goal**: Add convenience methods to `DirectionalRateResult` for plotting and basic metrics.

**Approach**: TDD - wrote 16 tests first, then implemented.

**Result**:

- Added 4 convenience methods to `DirectionalRateResult` in `src/neurospatial/encoding/directional.py`
- Added 16 new tests in `tests/encoding/test_encoding_directional.py` (total now 39, all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `plot(ax, polar, **kwargs)`: Plot directional tuning curve
  - Default: polar plot (circular representation)
  - `polar=False`: Cartesian plot (direction on x-axis, rate on y-axis)
  - Closes the curve by appending first point to end
  - Accepts optional `ax` argument and passes through all kwargs
  - Uses lazy import for matplotlib

- `preferred_direction()`: Circular mean weighted by firing rate
  - Returns float in radians, range [-π, π]
  - Delegates to `circular_mean()` from `neurospatial.stats.circular`
  - Handles uniform firing gracefully (returns value, though not meaningful)

- `preferred_direction_deg()`: Convenience wrapper returning degrees
  - Returns float in degrees, range [-180, 180]
  - Simple conversion: `np.degrees(self.preferred_direction())`

- `peak_firing_rate()`: Maximum firing rate
  - Returns float (maximum value in firing_rate array)
  - Uses `np.nanmax()` to handle NaN values

**Test coverage (16 tests in 5 classes)**:

- `TestDirectionalRateResultPlot`: 5 tests (returns axes, polar default, cartesian, existing axes, kwargs)
- `TestDirectionalRateResultPreferredDirection`: 4 tests (returns float, valid range, near expected, uniform firing)
- `TestDirectionalRateResultPreferredDirectionDeg`: 3 tests (returns float, conversion, valid range)
- `TestDirectionalRateResultPeakFiringRate`: 4 tests (returns float, correct value, handles NaN, non-negative)

**Documentation**:

- All methods have complete NumPy-style docstrings
- Include algorithm descriptions and formulas
- Include interpretation guidelines
- Include See Also cross-references

**Code review feedback**: APPROVE - Ready to merge

- All 16 tests pass
- Zero mypy errors
- Clean separation of concerns
- Comprehensive documentation
- Proper use of `circular_mean()` from `stats.circular`
- Correct curve closing logic for polar plots

---

### Task 3.1: Create `encoding/directional.py` with Result Class Definitions [COMPLETED]

**Goal**: Create the result class definitions for directional rate computation (head direction cells).

**Approach**: TDD - wrote 23 tests first (`test_encoding_directional.py`), then implemented.

**Result**:

- Created `src/neurospatial/encoding/directional.py` with 2 dataclasses
- Created `tests/encoding/test_encoding_directional.py` with 23 tests (all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `DirectionalRateResult`: Single-neuron result (frozen dataclass)
  - Fields: `firing_rate`, `occupancy`, `bin_centers`, `bin_size`, `smoothing_sigma`
  - No Environment dependency (head direction is 1D circular, not spatial)
  - `bin_centers` stores angular coordinates in radians [0, 2π)

- `DirectionalRatesResult`: Batch result (frozen dataclass)
  - Fields: `firing_rates` (plural), `occupancy`, `bin_centers`, `bin_size`, `smoothing_sigma`
  - Implements `__len__`, `__getitem__`, `__iter__` for iteration
  - `__getitem__(idx)` returns `DirectionalRateResult` for single neuron

**Design differences from SpatialRateResult**:

- No `env` field (replaced by `bin_centers`)
- No `smoothing_method` field (replaced by `smoothing_sigma`)
- `bin_size` in radians (angular resolution)
- `smoothing_sigma` can be None (unsmoothed)

**Test coverage (23 tests in 9 classes)**:

- `TestDirectionalRateResultImport`: 2 tests (import from module and package)
- `TestDirectionalRateResultCreation`: 3 tests (basic, with smoothing, frozen)
- `TestDirectionalRateResultFields`: 4 tests (shapes, value ranges)
- `TestDirectionalRatesResultImport`: 2 tests (import from module and package)
- `TestDirectionalRatesResultCreation`: 2 tests (basic, frozen)
- `TestDirectionalRatesResultFields`: 2 tests (shapes)
- `TestDirectionalRatesResultIteration`: 6 tests (len, getitem, iter, metadata, order)
- `TestDirectionalRatesResultEdgeCases`: 2 tests (single neuron, empty)

**Exports added**:

- `DirectionalRateResult` and `DirectionalRatesResult` exported from `encoding/__init__.py`

**Code review feedback**: APPROVE - Ready to merge

- All 23 tests pass
- Zero mypy errors
- Follows established patterns from spatial.py
- Comprehensive documentation with NumPy-style docstrings

---

### Task 2.10: Write Comprehensive Tests for Spatial Encoding [COMPLETED]

**Goal**: Verify comprehensive test coverage for all spatial encoding functionality.

**Approach**: Reviewed existing test coverage against Task 2.10 requirements.

**Result**:

All requirements already covered by tests written during TDD in Tasks 2.1-2.9:

- **Single neuron computation**: 24 tests (TestComputeSpatialRateFunction and related)
- **Batch computation**: 47 tests (TestComputeSpatialRatesFunction and related)
- **All result class methods**: Complete coverage of SpatialRateResult and SpatialRatesResult methods
- **Edge cases**: 4 dedicated edge case tests + additional coverage throughout
- **to_dataframe() output format**: 25 tests (TestSpatialRatesResultToDataframe)

**Total test count**: 200 tests in `tests/encoding/test_encoding_spatial.py` - all passing

**Milestone 2 Status**: COMPLETE! All 10 tasks finished.

---

### Task 2.9: Implement `compute_spatial_rates()` Function [COMPLETED]

**Goal**: Implement the batch spatial rate computation function for multiple neurons.

**Approach**: TDD - wrote 44 tests first, then implemented.

**Result**:

- Added `compute_spatial_rates()` function to `src/neurospatial/encoding/spatial.py`
- Added 44 tests to `tests/encoding/test_encoding_spatial.py` (total now 203+, all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `compute_spatial_rates(env, spike_times, times, positions, *, smoothing_method, bandwidth, min_occupancy, n_jobs, backend)`:
  - Parameters follow canonical argument order from CLAUDE.md
  - Accepts multiple input formats: list of arrays, tuple, 2D array with NaN padding, single 1D array
  - Normalizes input via `normalize_spike_times()`
  - Uses `bin_spike_trains()` for efficient batch binning with parallelization
  - Uses `smooth_rate_maps_batch()` for batch smoothing
  - Returns `SpatialRatesResult` with all required fields

**Signature**:

```python
def compute_spatial_rates(
    env: Environment,
    spike_times: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    smoothing_method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy: float = 0.0,
    n_jobs: int = 1,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
) -> SpatialRatesResult:
```

**Efficiency advantages over calling `compute_spatial_rate()` in a loop**:

1. Occupancy computed once and shared across all neurons
2. Diffusion kernel computed once (in `smooth_rate_maps_batch()`)
3. Position-to-bin mapping done once (in `bin_spike_trains()`)
4. Spike binning parallelizable with joblib (`n_jobs` parameter)

**Test coverage (44 tests in 11 classes)**:

- `TestComputeSpatialRatesFunction`: 13 tests (basic functionality, shapes, metadata, defaults)
- `TestComputeSpatialRatesInputFormats`: 4 tests (list, tuple, 2D array, 1D array)
- `TestComputeSpatialRatesNJobs`: 3 tests (parallelization parameter and consistency)
- `TestComputeSpatialRatesSmoothingMethods`: 3 tests (diffusion_kde, gaussian_kde, binned)
- `TestComputeSpatialRatesMinOccupancy`: 2 tests (parameter and masking)
- `TestComputeSpatialRatesBackendParameter`: 3 tests (numpy, auto)
- `TestComputeSpatialRatesConsistency`: 2 tests (**CRITICAL** - batch matches single-neuron)
- `TestComputeSpatialRatesEdgeCases`: 4 tests (empty spikes, single neuron, empty list)
- `TestComputeSpatialRatesResultMethods`: 10 tests (all batch methods)

**Consistency verification**:

- `test_batch_matches_single_neuron_results()` proves batch produces identical results to processing each neuron individually with `compute_spatial_rate()` using `rtol=1e-10`

**Documentation**:

- Complete NumPy-style docstring with:
  - Extended description explaining efficiency advantages
  - Full parameter documentation
  - Notes section on when to use batch vs single
  - Working examples with iteration and DataFrame export

**Code review feedback**: APPROVE - Ready to merge

- All 44 tests pass
- Zero mypy errors
- Clean separation of concerns
- Comprehensive documentation
- Follows all project patterns from CLAUDE.md

---

### Task 2.8: Implement `compute_spatial_rate()` Function [COMPLETED]

**Goal**: Implement the single-neuron spatial rate computation function.

**Approach**: TDD - wrote 27 tests first, then implemented.

**Result**:

- Added `compute_spatial_rate()` function to `src/neurospatial/encoding/spatial.py`
- Added 27 tests to `tests/encoding/test_encoding_spatial.py` (total now 160, all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `compute_spatial_rate(env, spike_times, times, positions, *, smoothing_method, bandwidth, min_occupancy, backend)`:
  - Parameters follow canonical argument order from CLAUDE.md
  - Uses binning layer (`_binning.py`) to convert spikes to counts and compute occupancy
  - Uses smoothing layer (`_smoothing.py`) to compute smoothed firing rate
  - Returns `SpatialRateResult` with all required fields

**Signature**:

```python
def compute_spatial_rate(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    smoothing_method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy: float = 0.0,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
) -> SpatialRateResult:
```

**Backend handling**:

- `"numpy"`: Default, works everywhere
- `"jax"`: Raises `NotImplementedError` (not yet implemented)
- `"auto"`: Falls back to NumPy (JAX not yet implemented)

**Test coverage**:

- Core functionality: 14 tests (importability, return type, shapes, metadata, defaults, edge cases)
- Smoothing methods: 4 tests (diffusion_kde, gaussian_kde, binned, different results)
- Parameters: 6 tests (min_occupancy, backend options)
- Result methods: 4 tests (plot, peak_location, spatial_information, peak near expected)

**Documentation**:

- Complete NumPy-style docstring with:
  - One-line summary and extended description
  - Full parameter documentation with types and constraints
  - Returns section with structure details
  - See Also cross-references
  - Notes section explaining algorithm
  - Working examples

**Code review feedback**: APPROVE - Ready to merge

- All 27 tests pass
- Zero mypy errors
- Clean separation of concerns (binning → smoothing → result)
- Comprehensive documentation

---

### Task 2.7: Binning Layer for Spatial Encoding [COMPLETED]

**Goal**: Create helper functions to convert (env, spike_times, times, positions) → (spike_counts, occupancy) for spatial encoding.

**Approach**: TDD - wrote 27 tests first, then implemented.

**Result**:

- Created `src/neurospatial/encoding/_binning.py` with 3 functions
- Created `tests/encoding/test_encoding_binning.py` with 27 tests (all pass)
- All mypy and ruff checks pass
- Code review passed with REQUEST_CHANGES, then APPROVE after fixes

**Key Implementation Details**:

- `bin_spike_train(env, spike_times, times, positions)`: Single neuron spike binning
  - Interpolates spike positions from trajectory using linear interpolation
  - Returns `(n_bins,)` float64 array of spike counts
  - Handles empty spike trains, spikes outside time range, and invalid bins

- `compute_occupancy(env, times, positions)`: Compute time spent in each bin
  - Delegates to `Environment.occupancy()` for robust computation
  - Returns `(n_bins,)` float64 array of occupancy in seconds
  - Validates input shapes and dimensions

- `bin_spike_trains(env, spike_times, times, positions, n_jobs=1)`: Batch version
  - Normalizes spike times via `normalize_spike_times()` for flexible input formats
  - Returns tuple: `(spike_counts, occupancy)`
    - spike_counts: `(n_neurons, n_bins)` float64
    - occupancy: `(n_bins,)` float64 (shared across neurons)
  - Supports parallelization via joblib `n_jobs` parameter

**Code review feedback addressed**:

1. Removed unused `position_bins` parameter from `bin_spike_train()`
   - Spike positions are interpolated, so precomputed trajectory bins are not useful
2. Removed unused `position_bins` parameter from `compute_occupancy()`
   - `Environment.occupancy()` doesn't accept this parameter
3. Fixed docstring examples to use dynamic shape assertions instead of hardcoded values
4. Added input validation for times/positions length mismatch in `bin_spike_train()`

**Documentation**:

- Complete NumPy-style docstrings with parameter descriptions
- Module docstring explains output shapes and layer purpose
- Examples show typical usage patterns

**Separation of concerns**:

The binning layer is intentionally separated from smoothing to allow:
- Reusing occupancy across multiple neurons (computed once per batch)
- Future JAX implementations with different parallelization strategies
- Clear API boundary between discrete binning and continuous smoothing

---

### Task 2.6: `SpatialRatesResult.to_dataframe()` [COMPLETED]

**Goal**: Implement the `to_dataframe()` method for exporting metrics to pandas DataFrame.

**Approach**: TDD - wrote 24 tests first, then implemented.

**Result**:

- Added `to_dataframe()` method to `SpatialRatesResult` in `src/neurospatial/encoding/spatial.py`
- Added 24 new tests in `tests/encoding/test_encoding_spatial.py` (total now 133, all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `to_dataframe(neuron_ids, include_classification)`: Export to pandas DataFrame
  - Parameters:
    - `neuron_ids`: Optional custom identifiers (default: integer indices)
    - `include_classification`: Whether to include cell_type column (default: True)
  - Returns DataFrame with columns:
    - neuron_id, peak_x, peak_y, peak_rate
    - spatial_info, sparsity
    - grid_score, border_score
    - cell_type (optional)

- Delegates to existing batch methods:
  - `peak_locations()` for peak_x, peak_y
  - `peak_firing_rates()` for peak_rate
  - `spatial_information()` for spatial_info
  - `sparsity()` for sparsity
  - `grid_scores()` for grid_score
  - `border_scores()` for border_score
  - `classify()` for cell_type

- Handles 1D environments by setting peak_y to NaN

**Documentation**:

- Complete NumPy-style docstring
- Includes common pandas workflows (filter, sort, top-N)
- Includes practical examples
- Cross-references to related methods

**Code review feedback**: APPROVE - Ready to merge

- No critical issues identified
- Minor documentation clarity improvement applied
- All 24 tests pass
- Proper delegation pattern followed

---

### Task 2.5: `SpatialRatesResult` Batch Metrics [COMPLETED]

**Goal**: Implement batch metrics methods for the SpatialRatesResult class.

**Approach**: TDD - wrote 22 tests first, then implemented.

**Result**:

- Added 3 batch metrics methods to `SpatialRatesResult` in `src/neurospatial/encoding/spatial.py`
- Added 22 new tests in `tests/encoding/test_encoding_spatial.py` (total now 109, all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `grid_scores()`: Batch grid score computation → returns `(n_neurons,)` array
  - Delegates to `batch_grid_scores()` from `_metrics.py`
  - Returns values in range [-2, 2] or NaN
  - NaN returned when grid score cannot be computed (non-2D, constant firing)

- `border_scores(threshold, min_area, distance_metric)`: Batch border score → returns `(n_neurons,)` array
  - Delegates to `batch_border_scores()` from `_metrics.py`
  - Returns values in range [-1, 1] or NaN
  - Accepts same parameters as single-neuron `border_score()` (threshold, min_area, distance_metric)

- `classify(min_spatial_info, min_grid_score, min_border_score)`: Cell type classification → returns `(n_neurons,)` string array
  - Computes all three metrics internally (spatial_info, grid_scores, border_scores)
  - Returns labels: "grid", "border", "place", or "unclassified"
  - Classification priority: grid > border > place > unclassified
  - Default thresholds from literature: spatial_info=0.5, grid_score=0.4, border_score=0.5

**Documentation**:

- All methods have complete NumPy-style docstrings
- Include algorithm descriptions (Sargolini et al. 2006, Solstad et al. 2008, Skaggs et al. 1993)
- Include interpretation guidelines and threshold recommendations
- Include See Also cross-references

**Code review feedback**: APPROVE - Ready to merge

- Excellent adherence to project patterns
- Complete and accurate NumPy-style documentation
- Comprehensive test coverage (22 tests including min_area parameter test)
- Proper delegation to existing utilities

**Test notes**:

- Added test for `min_area` parameter per code review feedback
- Fixed `test_classify_unclassified_low_spatial_info` to use unreachable thresholds for grid/border
  (uniform firing has high border score, so need to disable border classification to test spatial info threshold)

---

### Task 2.4: `SpatialRatesResult` Batch Methods [COMPLETED]

**Goal**: Implement batch convenience methods for the SpatialRatesResult class.

**Approach**: TDD - wrote 18 tests first, then implemented.

**Result**:

- Added 3 batch methods to `SpatialRatesResult` in `src/neurospatial/encoding/spatial.py`
- Added 18 new tests in `tests/encoding/test_encoding_spatial.py` (total now 87, all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `plot(idx, ax, **kwargs)`: Plot rate map for a specific neuron
  - Requires `idx` parameter (0-indexed)
  - Delegates to `env.plot_field()` for visualization
  - Uses `_to_numpy()` for JAX compatibility
  - Accepts optional `ax` argument and passes through all kwargs

- `spatial_information()`: Batch spatial information → returns `(n_neurons,)` array
  - Delegates to `batch_spatial_information()` from `_metrics.py`
  - Uses `_to_numpy()` for JAX compatibility
  - All values are non-negative

- `sparsity()`: Batch sparsity → returns `(n_neurons,)` array
  - Delegates to `batch_sparsity()` from `_metrics.py`
  - Uses `_to_numpy()` for JAX compatibility
  - All values in range [0, 1]

**Documentation**:

- All methods have complete NumPy-style docstrings
- Include algorithm formulas (Skaggs et al. 1993, 1996)
- Include interpretation guidelines
- Include practical examples

**Code review feedback**: APPROVE - Ready to merge

- Excellent adherence to project patterns
- Complete and accurate NumPy-style documentation
- Comprehensive test coverage (18 tests)
- Proper delegation to existing utilities

---

### Task 2.3: `SpatialRateResult` Cell Type Metrics [COMPLETED]

**Goal**: Implement cell type metric convenience methods for the SpatialRateResult class.

**Approach**: TDD - wrote 24 tests first, then implemented.

**Result**:

- Added 4 cell type metric methods to `SpatialRateResult` in `src/neurospatial/encoding/spatial.py`
- Added 24 new tests in `tests/encoding/test_encoding_spatial.py` (total now 69, all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `grid_score()`: Grid cell hexagonal periodicity score
  - Delegates to `grid.spatial_autocorrelation()` and `grid.grid_score()`
  - Returns float in range [-2, 2] or NaN
  - Uses FFT-based autocorrelation for 2D regular grids
  - Added runtime check for tuple return from FFT (should never happen)

- `grid_properties()`: Full grid cell metrics
  - Delegates to `grid.spatial_autocorrelation()` and `grid.grid_properties()`
  - Returns `GridProperties` dataclass with score, scale, orientation, etc.
  - Uses `float(np.min(env.bin_sizes))` for bin_size parameter (handles non-uniform grids)

- `border_score(threshold, min_area, distance_metric)`: Border cell boundary tuning score
  - Delegates to `border.border_score()`
  - Returns float in range [-1, 1] or NaN
  - Added `min_area` parameter (from code review feedback) for filtering small fields
  - Uses `cast("EnvironmentProtocol", self.env)` for mypy compatibility

- `region_coverage(threshold, regions)`: Coverage of spatial regions by firing field
  - Delegates to `border.compute_region_coverage()`
  - Returns dict mapping region names to coverage fractions [0, 1]
  - Handles zero/NaN peak rate by returning zero coverage for all regions

**Code review feedback addressed**:

1. Added `min_area` parameter to `border_score()` (Medium priority - API completeness)
2. Added edge case tests for zero firing and min_area filtering
3. Used type casts for EnvironmentProtocol compatibility

**Documentation**:

- All methods have complete NumPy-style docstrings
- Include algorithm descriptions (Sargolini et al. 2006, Solstad et al. 2008)
- Include interpretation guidelines and threshold recommendations
- Include See Also cross-references

---

### Task 2.2: `SpatialRateResult` Convenience Methods [COMPLETED]

**Goal**: Implement convenience methods for the SpatialRateResult class.

**Approach**: TDD - wrote 17 tests first, then implemented.

**Result**:

- Added 4 convenience methods to `SpatialRateResult` in `src/neurospatial/encoding/spatial.py`
- Added 17 new tests in `tests/encoding/test_encoding_spatial.py` (total now 47, all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `plot(ax, **kwargs)`: Delegates to `env.plot_field()` for visualization
  - Accepts optional `ax` argument
  - Passes through all kwargs
  - Uses `_to_numpy()` for JAX compatibility

- `peak_location()`: Alias for `peak_locations()` from mixin
  - Returns `(n_dims,)` coordinates
  - Convenience method for single-neuron results

- `spatial_information()`: Skaggs spatial information (bits/spike)
  - Delegates to `_metrics.spatial_information()`
  - Returns float, always non-negative
  - Uniform firing returns 0.0

- `sparsity()`: Skaggs sparsity measure
  - Delegates to `_metrics.sparsity()`
  - Returns float in range [0, 1]
  - Uniform firing returns 1.0, selective firing < 0.1

**Documentation**:

- All methods have complete NumPy-style docstrings
- Include formulas in Notes section
- Include Skaggs et al. references
- Include interpretation guidelines

**Code review feedback**: APPROVE with no issues. Implementation follows all project patterns:

- Proper delegation to existing utilities
- `_to_numpy()` for host-only operations
- TYPE_CHECKING guard for matplotlib Axes import
- Method-level imports to avoid circular dependencies

---

### Task 2.1: `encoding/spatial.py` - Result Class Definitions [COMPLETED]

**Goal**: Create spatial.py with SpatialRateResult and SpatialRatesResult dataclasses.

**Approach**: TDD - wrote 30 tests first (`test_encoding_spatial.py`), then implemented.

**Result**:

- Created `src/neurospatial/encoding/spatial.py` with 2 dataclasses
- Created `tests/encoding/test_encoding_spatial.py` with 30 tests (all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `SpatialRateResult`: Single-neuron result (frozen dataclass)
  - Fields: `firing_rate`, `occupancy`, `env`, `smoothing_method`, `bandwidth`
  - Inherits from `SpatialResultMixin` for `peak_locations()`, `peak_firing_rates()`

- `SpatialRatesResult`: Batch result (frozen dataclass)
  - Fields: `firing_rates` (plural), `occupancy`, `env`, `smoothing_method`, `bandwidth`
  - Inherits from `SpatialResultMixin`
  - Implements `__len__`, `__getitem__`, `__iter__` for iteration
  - `__getitem__(idx)` returns `SpatialRateResult` for single neuron

**Iteration interface**:

- `len(result)`: Returns number of neurons
- `result[i]`: Returns `SpatialRateResult` for neuron i
- `for r in result`: Iterates over single-neuron results

**Code review feedback**: APPROVE with no critical issues. Suggestions for future enhancements (TypeVar for JAX support, custom __repr__) are low priority.

---

### Task 1.4: `encoding/_metrics.py` - batch_border_scores [COMPLETED]

**Goal**: Add `batch_border_scores()` function to compute border scores for multiple neurons.

**Approach**: TDD - wrote tests first (`test_encoding_metrics.py`), then implemented.

**Result**:

- Added `batch_border_scores(env, firing_rates, ...)` to `src/neurospatial/encoding/_metrics.py`
- Added 14 new tests in `tests/encoding/test_encoding_metrics.py` (all pass, total now 61)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `batch_border_scores(env, firing_rates, ...)`: Batch border score computation for populations
- Delegates to `border.border_score()` for each neuron
- Returns `(n_neurons,)` array of border scores in range [-1, 1] or NaN

**Parameters**:

- `env`: Environment (spatial context with bin centers and connectivity)
- `firing_rates`: Shape `(n_neurons, n_bins)`
- `threshold`: Default 0.3 (passed to border_score)
- `min_area`: Default 0.0 (passed to border_score, filters small fields)
- `distance_metric`: Default "geodesic" (or "euclidean")

**Edge case handling**:

- Zero firing rate: Returns NaN
- NaN values in firing rate: Handled gracefully
- All-NaN firing: Returns NaN
- Wrong input shape: Raises ValueError
- Computation errors: Caught and returns NaN for that neuron

**Code review notes**:

- Follows same pattern as `batch_grid_scores()` and `batch_spatial_information()`
- Graceful error handling prevents batch failure from single problematic neuron
- Complete NumPy docstring with interpretation guidelines and Solstad et al. reference
- Used `cast("EnvironmentProtocol", env)` pattern for method calls to satisfy mypy

---

### Task 1.3: `encoding/_metrics.py` - batch_grid_scores [COMPLETED]

**Goal**: Add `batch_grid_scores()` function to compute grid scores for multiple neurons.

**Approach**: TDD - wrote tests first (`test_encoding_metrics.py`), then implemented.

**Result**:

- Added `batch_grid_scores(env, firing_rates)` to `src/neurospatial/encoding/_metrics.py`
- Added 9 new tests in `tests/encoding/test_encoding_metrics.py` (all pass, total now 47)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `batch_grid_scores(env, firing_rates, ...)`: Batch grid score computation for populations
- Delegates to `grid.spatial_autocorrelation()` and `grid.grid_score()` for each neuron
- Returns `(n_neurons,)` array of grid scores in range [-2, 2] or NaN

**Parameters**:

- `env`: Environment (must be regular 2D grid for FFT-based autocorrelation)
- `firing_rates`: Shape `(n_neurons, n_bins)`
- `inner_radius_fraction`: Default 0.2 (passed to grid_score)
- `outer_radius_fraction`: Default 0.5 (passed to grid_score)

**Edge case handling**:

- Constant firing rate: Returns NaN (zero variance)
- NaN values in firing rate: Handled gracefully
- Graph-based method (non-FFT): Returns NaN (grid_score requires 2D autocorrelogram)
- Wrong input shape: Raises ValueError
- Computation errors: Caught and returns NaN for that neuron

**Code review notes**:

- Follows same pattern as `batch_spatial_information()` and `batch_sparsity()`
- Graceful error handling prevents batch failure from single problematic neuron
- Complete NumPy docstring with interpretation guidelines

---

### Task 1.2: `encoding/_smoothing.py` [COMPLETED]

**Goal**: Create shared smoothing implementations for rate map computation with three methods.

**Approach**: TDD - wrote tests first (`test_encoding_smoothing.py`), then implemented.

**Result**:

- Created `src/neurospatial/encoding/_smoothing.py` with 2 public functions + 4 private helpers
- Created `tests/encoding/test_encoding_smoothing.py` with 34 tests (all pass)
- All mypy and ruff checks pass
- Code review passed after fixing type annotation issues

**Key Implementation Details**:

- `smooth_rate_map(env, spike_counts, occupancy, method, bandwidth, ...)`: Single neuron smoothing
- `smooth_rate_maps_batch(env, spike_counts, occupancy, ...)`: Batch version for populations

**Three smoothing methods**:

1. **diffusion_kde** (recommended): Graph-based boundary-aware KDE
   - Respects environment boundaries via graph connectivity
   - Order: smooth counts → smooth occupancy → normalize
   - Uses `env.compute_kernel(bandwidth, mode="density")`

2. **gaussian_kde**: Standard Euclidean KDE
   - Ignores boundaries (mass can bleed through walls)
   - Uses pairwise Gaussian weights between bin centers
   - Order: smooth counts → smooth occupancy → normalize

3. **binned** (legacy): Bin-then-smooth order
   - Order: normalize → smooth result
   - Can introduce discretization artifacts
   - Uses `env.smooth()` for diffusion smoothing of rate map

**Parameters**:

- `bandwidth`: Smoothing bandwidth (same units as bin_size)
- `min_occupancy`: Threshold for masking low-occupancy bins
- `kernel`: Optional precomputed kernel for efficiency

**Edge case handling**:

- Zero occupancy bins: Produce NaN in rate map
- Zero spike counts: Return zero rate (not NaN)
- Invalid method: Raises `ValueError`
- Negative bandwidth: Raises `ValueError`
- Shape mismatches: Raises `ValueError`

**Type annotation fix**:

Used `cast("EnvironmentProtocol", env)` pattern for method calls to satisfy mypy,
following the same pattern used in `place.py`, `border.py`, and other encoding modules.

**Code review feedback addressed**:

- Fixed 4 mypy errors by using `cast("EnvironmentProtocol", env)` for method calls
- Documented min_occupancy behavior (uses raw occupancy for threshold, not smoothed)
- Kept bandwidth validation method-specific (binned allows 0, KDE methods require > 0)

---

### Task 1.1: `encoding/_metrics.py` [COMPLETED]

**Goal**: Create shared spatial information and sparsity metric implementations for encoding result classes.

**Approach**: TDD - wrote tests first (`test_encoding_metrics.py`), then implemented.

**Result**:

- Created `src/neurospatial/encoding/_metrics.py` with 4 functions
- Created `tests/encoding/test_encoding_metrics.py` with 38 tests (all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `spatial_information(firing_rate, occupancy, base=2.0)`: Skaggs spatial information (bits/spike)
- `batch_spatial_information(firing_rates, occupancy, base=2.0)`: Batch version for populations
- `sparsity(firing_rate, occupancy)`: Skaggs sparsity measure (0-1)
- `batch_sparsity(firing_rates, occupancy)`: Batch version for populations

**Formula implementations**:

Spatial information (Skaggs et al. 1993):
```
I = sum_i(p_i * (r_i/r_mean) * log2(r_i/r_mean))
```

Sparsity (Skaggs et al. 1996):
```
S = (sum_i(p_i * r_i))^2 / sum_i(p_i * r_i^2)
```

**Edge case handling**:

- NaN values: Ignored via `np.nansum`
- Zero mean rate: Returns 0.0
- Empty arrays: Raises `ValueError`
- Mismatched shapes: Raises `ValueError`
- Inf values: Handled gracefully

**Backward compatibility**:

- Tests verify exact match with legacy `place.py` implementations
- Both `test_matches_place_skaggs_information` and `test_matches_place_sparsity` pass

**Code review notes**:

- Batch functions use list comprehension (intentionally simple, JAX can use vmap in Phase 6)
- Occupancy is normalized internally (more defensive than legacy which expects pre-normalized)
- All functions return Python float (single) or NDArray[float64] (batch)

---

### Task 0.5: `encoding/_core_jax.py` [COMPLETED]

**Goal**: Create JAX core array operations stubs matching `_core_numpy.py` interface.

**Approach**: TDD - wrote tests first (`test_encoding_core_jax.py`), then implemented.

**Result**:

- Created `src/neurospatial/encoding/_core_jax.py` with 4 stub functions
- Created `tests/encoding/test_encoding_core_jax.py` with 23 tests (all skip when JAX unavailable)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `compute_firing_rate_single(spike_counts, occupancy)`: Single neuron rate computation (JAX)
- `compute_firing_rates_batch(spike_counts, occupancy)`: Batch rate computation (JAX)
- `smooth_rate_map_single(firing_rate, adjacency, bandwidth, method)`: Single neuron smoothing (JAX)
- `smooth_rate_maps_batch(firing_rates, adjacency, bandwidth, method)`: Batch smoothing (JAX)

**Stub behavior**:

All functions raise `NotImplementedError` with messages indicating this is Phase 0, and JAX implementations will be added in Phase 6.

**Type annotations**:

- Uses `TYPE_CHECKING` guard to import `jax.Array` only during static type checking
- Module can be imported without JAX installed
- Tests are skipped when JAX is not available via `pytestmark = pytest.mark.skipif`

**Interface consistency**:

- All function signatures match `_core_numpy.py` exactly
- Same parameter names, types, and defaults
- Same docstring structure with JAX-specific notes added

---

### Task 0.4: `encoding/_core_numpy.py` [COMPLETED]

**Goal**: Create NumPy core array operations stubs for rate computation and smoothing.

**Approach**: TDD - wrote tests first (`test_encoding_core_numpy.py`), then implemented.

**Result**:

- Created `src/neurospatial/encoding/_core_numpy.py` with 4 stub functions
- Created `tests/encoding/test_encoding_core_numpy.py` with 22 tests (all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `compute_firing_rate_single(spike_counts, occupancy)`: Single neuron rate computation
- `compute_firing_rates_batch(spike_counts, occupancy)`: Batch rate computation for (n_neurons, n_bins)
- `smooth_rate_map_single(firing_rate, adjacency, bandwidth, method)`: Single neuron smoothing
- `smooth_rate_maps_batch(firing_rates, adjacency, bandwidth, method)`: Batch smoothing

**Stub behavior**:

All functions raise `NotImplementedError` with helpful messages explaining this is Phase 0 of the refactor. Actual implementations will be added in Phase 1 (Milestone 1).

**Function signatures established**:

- Single neuron arrays: `(n_bins,)`
- Batch arrays: `(n_neurons, n_bins)` for firing data, `(n_bins,)` for shared occupancy
- Smoothing methods: `Literal["diffusion_kde", "gaussian_kde", "binned"]`
- Default bandwidth: 5.0, default min_occupancy: 0.0

---

### Task 0.3: `encoding/_backend.py` [COMPLETED]

**Goal**: Create backend selection infrastructure for NumPy/JAX computation backends.

**Approach**: TDD - wrote tests first (`test_encoding_backend.py`), then implemented.

**Result**:

- Created `src/neurospatial/encoding/_backend.py` with backend selection functions
- Created `tests/encoding/test_encoding_backend.py` with 26 tests (21 passed, 5 skipped for JAX)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:

- `SUPPORTED_BACKENDS`: Tuple of valid backend names ("numpy", "jax", "auto")
- `is_jax_available()`: Checks platform (Windows → False) and JAX installation
- `get_backend(name)`: Returns numpy or jax.numpy module based on selection
- `get_backend_name(name)`: Resolves "auto" to actual backend name used
- Error messages include platform-specific guidance (e.g., "JAX is not supported on Windows")

**Backend behavior**:

1. `"numpy"` - Always returns numpy module, works everywhere
2. `"jax"` - Returns jax.numpy module, raises ImportError if unavailable
3. `"auto"` - Uses JAX if available, falls back to NumPy silently

---

### Task 0.2: `encoding/_spikes.py` [COMPLETED]

**Goal**: Create spike format normalization helper that converts various input formats to the canonical list-of-arrays format.

**Approach**: TDD - wrote tests first (`test_encoding_spikes.py`), then implemented.

**Result**:
- Created `src/neurospatial/encoding/_spikes.py` with `normalize_spike_times()` function
- Created `tests/encoding/test_encoding_spikes.py` with 27 tests (all pass)
- All mypy and ruff checks pass
- Code review passed with APPROVE

**Key Implementation Details**:
- Function accepts 1D array (single neuron), 2D array (NaN-padded), or list/tuple of 1D arrays
- Uses `isinstance(spike_times, (list, tuple))` instead of `isinstance(spike_times, Sequence)` to avoid mypy unreachable code warning
- Proper error handling for ragged object arrays, 3D arrays, and non-1D elements in lists
- All output arrays are converted to `float64` dtype

**Input formats handled**:
1. 1D array (single neuron) → wrapped in list
2. 2D array (n_neurons, max_spikes) with NaN padding → split, NaNs removed
3. list/tuple of 1D arrays (canonical) → each element converted to float64 array

---

### Task 0.1: `encoding/_base.py` [COMPLETED]

**Goal**: Create shared infrastructure for result classes including:
- `_to_numpy(arr)` - convert JAX arrays to NumPy for host-only operations
- `_get_array_module(arr)` - detect array backend (numpy vs jax.numpy)
- `HasOccupancy` protocol
- `HasEnvironment` protocol
- `SpatialResultMixin` with `peak_locations()` and `peak_firing_rates()`

**Approach**: TDD - wrote tests first (`test_encoding_base.py`), then implemented.

**Result**:
- Created `src/neurospatial/encoding/_base.py` with all required components
- Created `tests/encoding/test_encoding_base.py` with 24 tests (22 passed, 2 skipped for JAX)
- All mypy and ruff checks pass

**Key Implementation Details**:
- `_to_numpy()` uses `np.asarray()` which handles both NumPy and JAX arrays
- `_get_array_module()` detects JAX via `__jax_array__` attribute
- `SpatialResultMixin._get_rates()` handles both single (`firing_rate`) and batch (`firing_rates`) attributes
- Return type annotations use `Any` in some places to satisfy mypy with mixin pattern

---

## Decisions Made

1. **Used `Any` return type for `_get_array_module()`**: The `ModuleType` annotation was causing mypy issues with jax.numpy. Using `Any` is pragmatic for module-level dispatch.

2. **Used `Any` for mixin attribute access**: Mixins access `self.env`, `self.firing_rate`, etc. that are defined in subclasses. Using `type: ignore[attr-defined]` and `Any` types is the cleanest solution.

3. **1D environment fixture simplified**: True 1D environments via `from_graph` require complex edge setup. Used a "narrow 2D" environment instead for testing mixin behavior.

---

## Blockers

(None currently)

---

## Questions for User

(None currently)
