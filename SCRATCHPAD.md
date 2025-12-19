# Encoding Module Refactor - Scratchpad

## Current Status

**Date**: 2025-12-18
**Last Completed**: Task 2.1 - Create `encoding/spatial.py` with result class definitions
**Next Task**: Task 2.2 - Implement `SpatialRateResult` convenience methods

## Session Notes

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
