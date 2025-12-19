# Encoding Module Refactor - Scratchpad

## Current Status

**Date**: 2025-12-18
**Last Completed**: Task 0.4 - Create `encoding/_core_numpy.py` with stubs
**Next Task**: Task 0.5 - Create `encoding/_core_jax.py` with stubs

## Session Notes

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
