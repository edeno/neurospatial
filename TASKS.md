# Animation Optimization Tasks

**Project Goal**: Enable visualization of 1–2 hour decoding sessions at 250–500 Hz in napari without temporal downsampling.

**Reference**: See [PLAN.md](PLAN.md) for detailed implementation specifications.

---

## Milestone 1: Core Array Preservation (Foundation) ✅ COMPLETE

**Priority**: HIGH - All other milestones depend on this.

### Task 1.1: Preserve Array Format in animate_fields ✅

**File**: `src/neurospatial/animation/core.py`

**What to do**:

1. Add `fields_is_array` flag to detect 2D numpy arrays (line ~205)
2. Compute `n_frames` from array shape or list length
3. Only convert arrays to list for non-napari backends
4. Keep arrays intact when `backend="napari"`

**Success criteria**:

- `isinstance(fields, np.ndarray)` remains True after passing through animate_fields when backend="napari"
- `isinstance(fields, list)` is True for video/html/widget backends
- Existing tests pass: `uv run pytest tests/animation/test_core.py -v`

**Dependencies**: None

---

### Task 1.2: Update Field Validation for Arrays ✅

**File**: `src/neurospatial/animation/core.py`

**What to do**:

1. Modify field shape validation (around line 227-233)
2. For arrays: validate `fields.shape[1] == env.n_bins`
3. For lists: keep existing per-element validation
4. Update error messages to reflect both input types

**Success criteria**:

- Array with wrong shape raises ValueError with helpful message
- List with wrong element shape raises ValueError
- `uv run pytest tests/animation/test_core.py -v` passes

**Dependencies**: Task 1.1

---

### Task 1.3: Ensure Multi-Field Mode Compatibility ✅

**File**: `src/neurospatial/animation/core.py`

**Important**: The array preservation logic must NOT interfere with existing multi-field mode.

**Multi-field mode** accepts a **list of sequences** (each a 2D array) for side-by-side comparison (napari only). This is detected by checking if the input is a list of arrays rather than a single 2D array.

**What to do**:

1. Verify `is_multi_field` detection runs BEFORE `fields_is_array` detection
2. Ensure `fields_is_array` is False for list inputs (multi-field mode)
3. Add test that multi-field mode still works after array preservation changes

**Key check** (pseudo-code):

```python
# Multi-field detection (existing logic, must remain unchanged)
is_multi_field = isinstance(fields, list) and ...  # Nested sequence check

# Array detection (new logic, only for single-field mode)
fields_is_array = isinstance(fields, np.ndarray) and fields.ndim >= 2
# Note: isinstance(fields, np.ndarray) is False for lists, so this is safe
```

**Success criteria**:

- Multi-field mode still works: `env.animate_fields([fields1, fields2], backend="napari")`
- Single-field array mode works: `env.animate_fields(fields_2d, backend="napari")`
- `uv run pytest tests/animation/ -v -k "multi_field"` passes

**Dependencies**: Task 1.1

---

## Milestone 2: Napari Backend Array Support

**Priority**: HIGH - Enables memmap passthrough.

**Key insight**: Napari natively supports dask arrays for lazy loading. Consider using `dask.array.from_array()` + `da.map_blocks()` as an alternative to custom `LazyFieldRenderer`. See: <https://napari.org/dev/tutorials/processing/dask.html>

### Task 2.1: Update render_napari Type Hints ✅

**File**: `src/neurospatial/animation/backends/napari_backend.py`

**What to do**:

1. Update `render_napari` signature to accept `Sequence[NDArray] | NDArray`
2. Add `fields_is_array` detection at function start
3. Compute `n_frames` from `fields.shape[0]` for arrays
4. Only call `_validate_field_types_consistent` for list inputs

**Success criteria**:

- `render_napari` accepts both 2D arrays and lists
- Type hints pass mypy: `uv run mypy src/neurospatial/animation/backends/napari_backend.py`

**Dependencies**: Milestone 1

**Completed**: 2025-11-28

**Implementation Notes**:

- Added `NDArray[np.float64]` to type union in signature
- Added `fields_is_array` detection: `isinstance(fields, np.ndarray) and fields.ndim == 2`
- Skip `_validate_field_types_consistent` and `_is_multi_field_input` for arrays
- Updated docstring to document array mode as recommended for large datasets
- Added 7 comprehensive tests in `TestRenderNapariArrayInput` class
- All existing tests pass (39 passed, 1 skipped)

---

### Task 2.2: Update _create_lazy_field_renderer ✅

**File**: `src/neurospatial/animation/backends/napari_backend.py`

**What to do**:

1. Update function signature to accept `list[NDArray] | NDArray`
2. Change `n_frames = len(fields)` to handle both types
3. Pass input type information to renderer classes

**Success criteria**:

- Function accepts both input types without error
- Correct renderer class selected based on n_frames threshold

**Dependencies**: Task 2.1

**Completed**: 2025-11-28

---

### Task 2.3: Update LazyFieldRenderer for Array Input ✅

**File**: `src/neurospatial/animation/backends/napari_backend.py`

**What to do**:

1. Update `__init__` to accept `list[NDArray] | NDArray`
2. Add `self._fields_is_array` flag
3. Update `__len__` to use `fields.shape[0]` for arrays
4. Update `_get_frame` - field access `self.fields[idx]` works for both types

**Success criteria**:

- `len(renderer)` returns correct frame count for arrays
- `renderer[i]` returns valid RGB frame for array input
- Memmap access uses efficient slicing (no full load)

**Dependencies**: Task 2.2

**Completed**: 2025-11-28

---

### Task 2.4: Update ChunkedLazyFieldRenderer for Array Input ✅

**File**: `src/neurospatial/animation/backends/napari_backend.py`

**What to do**:

1. Same pattern as Task 2.3: update `__init__`, `__len__`
2. Update `_render_chunk` to use slice `self.fields[start:end]` for arrays
3. Ensure slice returns view (not copy) for memmaps

**Success criteria**:

- Chunked renderer works with array input
- `self.fields[start:end]` creates view for memmaps (verify with `np.shares_memory`)

**Dependencies**: Task 2.3

**Completed**: 2025-11-28

---

### Task 2.5: Implement Dask-Based Alternative Renderer ✅

**File**: `src/neurospatial/animation/backends/napari_backend.py`

**What to do**:

1. Add `_create_dask_field_renderer()` function as alternative to `_create_lazy_field_renderer()`
2. Use `dask.array.from_array()` to wrap input fields with chunking
3. Use `dask.array.map_blocks()` to apply `field_to_rgb_for_napari` lazily
4. Return dask array that napari can consume directly
5. Add `use_dask: bool = False` parameter to `render_napari`

**Completed**: 2025-11-28

**Implementation Notes**:

- Added `_create_dask_field_renderer()` with comprehensive input validation:
  - Requires numpy array input (not list)
  - Requires 2D shape (n_frames, n_bins)
  - Requires at least 1 frame
  - Validates chunk_size > 0
- Added `use_dask: bool = False` parameter to `render_napari`
- Added 14 comprehensive tests in `TestDaskFieldRenderer` class:
  - Core functionality (returns dask array, correct shape, dtype)
  - Lazy evaluation verification
  - Memmap support
  - Equivalence with LazyFieldRenderer
  - Edge case validation (empty array, wrong shape, list input, invalid chunk_size)
  - Integration with render_napari
- All existing tests pass (53 passed, 1 skipped)

**Implementation sketch**:

```python
def _create_dask_field_renderer(
    env: Environment,
    fields: NDArray[np.float64],  # Must be array for dask
    cmap_lookup: NDArray[np.uint8],
    vmin: float,
    vmax: float,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> da.Array:
    """Create dask-based lazy renderer leveraging napari's native dask support."""
    import dask.array as da

    # Wrap fields in dask array with chunking along time axis
    fields_dask = da.from_array(fields, chunks=(chunk_size, -1))

    # Define per-chunk conversion function
    def convert_chunk(block, block_info=None):
        # block is shape (chunk_size, n_bins)
        result = np.stack([
            field_to_rgb_for_napari(env, field, cmap_lookup, vmin, vmax)
            for field in block
        ])
        return result  # shape (chunk_size, height, width, 3)

    # Get output shape from sample render
    sample_rgb = field_to_rgb_for_napari(env, fields[0], cmap_lookup, vmin, vmax)
    height, width = sample_rgb.shape[:2]

    # Apply conversion lazily
    rgb_dask = da.map_blocks(
        convert_chunk,
        fields_dask,
        dtype=np.uint8,
        drop_axis=1,  # Remove n_bins axis
        new_axis=[1, 2, 3],  # Add height, width, channels
        chunks=(chunk_size, height, width, 3),
    )

    return rgb_dask
```

**Success criteria**:

- Dask array created without loading all data
- `viewer.add_image(rgb_dask)` works correctly
- Memory usage stays bounded for large datasets

**Dependencies**: Task 2.1

---

### Task 2.6: Benchmark LazyFieldRenderer vs Dask ✅

**File**: `benchmarks/bench_lazy_renderers.py`

**Completed**: 2025-11-28

**Summary**: LazyFieldRenderer significantly outperforms Dask in all metrics (20-45,000x faster creation, 20-220x faster access). Recommendation: Use LazyFieldRenderer as default, offer `use_dask=True` as option.

**Original File**: `benchmarks/benchmark_lazy_renderers.py` (create)

**What to do**:

1. Create benchmark script comparing both approaches
2. Test metrics: memory usage, frame access time, scrubbing responsiveness
3. Test with different dataset sizes: 10K, 100K, 500K frames
4. Test with both in-memory arrays and memmaps

**Benchmark template**:

```python
"""Benchmark LazyFieldRenderer vs Dask-based renderer."""

import tempfile
import time
from pathlib import Path

import numpy as np
import psutil

from neurospatial import Environment
from neurospatial.animation.backends.napari_backend import (
    _create_lazy_field_renderer,
    _create_dask_field_renderer,
)


def measure_memory():
    """Return current process memory in MB."""
    return psutil.Process().memory_info().rss / 1024 / 1024


def benchmark_renderer(renderer, n_accesses=100, random_access=True):
    """Benchmark frame access times."""
    n_frames = len(renderer) if hasattr(renderer, '__len__') else renderer.shape[0]

    if random_access:
        indices = np.random.randint(0, n_frames, n_accesses)
    else:
        indices = np.arange(n_accesses) % n_frames

    times = []
    for idx in indices:
        start = time.perf_counter()
        _ = renderer[idx]
        times.append(time.perf_counter() - start)

    return {
        "mean_ms": np.mean(times) * 1000,
        "median_ms": np.median(times) * 1000,
        "p99_ms": np.percentile(times, 99) * 1000,
    }


def run_benchmark(n_frames: int, n_bins: int = 500, use_memmap: bool = False):
    """Run benchmark for given dataset size."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {n_frames:,} frames, {n_bins} bins, memmap={use_memmap}")
    print('='*60)

    # Create environment
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=5.0)

    # Create fields
    if use_memmap:
        tmpdir = Path(tempfile.mkdtemp())
        mmap_path = tmpdir / "fields.dat"
        fields = np.memmap(str(mmap_path), dtype="float64", mode="w+", shape=(n_frames, n_bins))
        fields[:] = np.random.rand(n_frames, n_bins)
        fields.flush()
    else:
        fields = np.random.rand(n_frames, n_bins).astype(np.float64)

    # Setup colormap
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")
    cmap_lookup = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    vmin, vmax = 0.0, 1.0

    # Benchmark LazyFieldRenderer
    print("\n--- LazyFieldRenderer ---")
    mem_before = measure_memory()
    lazy_renderer = _create_lazy_field_renderer(env, fields, cmap_lookup, vmin, vmax)
    mem_after = measure_memory()
    print(f"Memory overhead: {mem_after - mem_before:.1f} MB")

    results_lazy = benchmark_renderer(lazy_renderer)
    print(f"Random access: mean={results_lazy['mean_ms']:.2f}ms, p99={results_lazy['p99_ms']:.2f}ms")

    # Benchmark Dask renderer (if array input)
    if isinstance(fields, np.ndarray):
        print("\n--- Dask Renderer ---")
        mem_before = measure_memory()
        dask_renderer = _create_dask_field_renderer(env, fields, cmap_lookup, vmin, vmax)
        mem_after = measure_memory()
        print(f"Memory overhead: {mem_after - mem_before:.1f} MB")

        results_dask = benchmark_renderer(dask_renderer)
        print(f"Random access: mean={results_dask['mean_ms']:.2f}ms, p99={results_dask['p99_ms']:.2f}ms")

    # Cleanup
    if use_memmap:
        del fields
        mmap_path.unlink()
        tmpdir.rmdir()


if __name__ == "__main__":
    for n_frames in [10_000, 100_000]:
        for use_memmap in [False, True]:
            run_benchmark(n_frames, use_memmap=use_memmap)
```

**Success criteria**:

- Benchmark runs without errors
- Clear comparison data for both approaches
- Results inform which approach to use by default

**Dependencies**: Tasks 2.4, 2.5

---

## Milestone 3: Scalable Colormap Range Computation ✅ COMPLETE

**Priority**: HIGH - Prevents OOM on large datasets.

**CRITICAL**: Per napari docs, omitting `contrast_limits` (vmin/vmax) causes napari to compute min/max over the **entire dataset** - "extremely long processing times". Also set `multiscale=False` to avoid unnecessary pyramid computation.

### Task 3.1: Add Streaming Path to compute_global_colormap_range ✅

**File**: `src/neurospatial/animation/rendering.py`

**Completed**: 2025-11-28

**Implementation**:

- Added `max_frames_for_exact=50_000` parameter
- Added `sample_stride=None` parameter
- For arrays > threshold: uses chunked streaming (10K frames at a time)
- For sample_stride: subsamples with `fields[::sample_stride]`
- Added 7 comprehensive tests in `TestColormapRangeStreaming` class

---

### Task 3.2: Update render_napari to Use Streaming ✅

**File**: `src/neurospatial/animation/backends/napari_backend.py`

**Completed**: 2025-11-28

**Implementation**:

- Added `large_dataset_threshold = 200_000` for detecting large datasets
- For n_frames > threshold without explicit vmin/vmax:
  - Compute `sample_stride = max(1, n_frames // 50_000)` to sample ~50K frames
  - Emit `UserWarning` with message explaining sampled range estimation
  - Pass `sample_stride` to `compute_global_colormap_range`
- Added 3 tests in `TestLargeDatasetColormapRange` class

---

### Task 3.3: Always Set multiscale=False ✅

**File**: `src/neurospatial/animation/backends/napari_backend.py`

**Completed**: 2025-11-28

**Implementation**:

- Added `multiscale=False` to `viewer.add_image()` in `render_napari()` (line 1727)
- Added `multiscale=False` to `viewer.add_image()` in `render_napari_multi_field()` (line 1995)
- Added `test_render_napari_sets_multiscale_false` test

---

## Milestone 4: Helper Functions ✅ COMPLETE

**Priority**: MEDIUM - Improves ergonomics for large sessions.

### Task 4.1: Add estimate_colormap_range_from_subset ✅

**Completed**: 2025-11-28

**File**: `src/neurospatial/animation/core.py`

**Implementation**:

- Function signature: `(fields, n_samples=10_000, percentile=(1.0, 99.0), seed=None) -> tuple[float, float]`
- Samples random frames (reproducible with fixed seed)
- Computes percentile-based range on samples
- Handles both array and list inputs
- Works efficiently with memory-mapped arrays
- Completes in <0.5s for 1M+ frame datasets

---

### Task 4.2: Add large_session_napari_config ✅

**Completed**: 2025-11-28

**File**: `src/neurospatial/animation/core.py`

**Implementation**:

- Function signature: `(n_frames, sample_rate_hz=None) -> dict[str, Any]`
- Returns recommended fps, chunk_size, max_chunks
- Scales recommendations based on n_frames thresholds

---

### Task 4.3: Export Helpers from Animation Module ✅

**Completed**: 2025-11-28

**File**: `src/neurospatial/animation/__init__.py`

**Implementation**:

```python
from neurospatial.animation import estimate_colormap_range_from_subset, large_session_napari_config
```

**Dependencies**: Tasks 4.1, 4.2

---

## Milestone 5: Overlay Subsampling (DEFERRED)

**Priority**: DEFERRED - Not required for core large-session support. May revisit if overlay memory becomes a bottleneck.

### Task 5.1: Add Overlay Subsampling Parameters

**File**: `src/neurospatial/animation/core.py`

**What to do**:

1. Add parameters to `animate_fields`: `overlay_subsample=None`, `max_overlay_frames=None`
2. Update docstring with parameter descriptions
3. Pass parameters through to `_convert_overlays_to_data`

**Success criteria**:

- Parameters accepted without error
- Parameters passed to overlay conversion

**Dependencies**: Milestone 1

---

### Task 5.2: Implement Overlay Subsampling Logic

**File**: `src/neurospatial/animation/overlays.py`

**What to do**:

1. Add parameters to `_convert_overlays_to_data`
2. Compute effective_frame_times based on subsampling
3. Use effective_frame_times for overlay alignment

**Success criteria**:

- `overlay_subsample=2` results in half the overlay frames
- `max_overlay_frames=1000` caps output regardless of input size

**Dependencies**: Task 5.1

---

## Milestone 6: Example Scripts & Documentation ✅ COMPLETE

**Priority**: MEDIUM - Demonstrates patterns, ensures usability.

### Task 6.1: Update view_bandit_napari.py ✅

**Completed**: 2025-11-28

**File**: `data/view_bandit_napari.py`

**Implementation**:

1. Added imports for `estimate_colormap_range_from_subset` and `large_session_napari_config`
2. Added `frame_times` array for temporal alignment
3. Pre-computed `vmin`, `vmax` using `estimate_colormap_range_from_subset`
4. Added `napari_config` using `large_session_napari_config`
5. Updated docstring with best practices documentation

---

### Task 6.2: Update large_session_napari_example.py ✅

**Completed**: 2025-11-28

**File**: `data/large_session_napari_example.py`

**Implementation**:

1. Increased N_FRAMES from 1,000 to 100,000 (100K frames)
2. Added imports for helper functions
3. Used `estimate_colormap_range_from_subset` for vmin/vmax
4. Used `large_session_napari_config` for recommended settings
5. Added comprehensive memory usage comments in docstring

---

### Task 6.3: Update Animation Documentation ✅

**Completed**: 2025-11-28

**File**: `docs/user-guide/animation.md`

**Implementation**:

1. Added "Large Session Helper Functions" section with:
   - `estimate_colormap_range_from_subset` documentation
   - `large_session_napari_config` documentation
   - Recommended settings table by session size
2. Added "Complete Large-Session Example" section
3. Updated API Reference to include new functions

---

### Task 6.4: Update CLAUDE.md Quick Reference ✅

**Completed**: 2025-11-28

**File**: `CLAUDE.md`

**Implementation**:

1. Added "Large session helpers" section with:
   - Import examples
   - `estimate_colormap_range_from_subset` usage
   - `large_session_napari_config` usage
   - Combined workflow example

---

## Milestone 7: Testing ✅ COMPLETE

**Priority**: HIGH - Ensures correctness and prevents regressions.

### Task 7.1: Test Array Input to Lazy Renderers ✅

**Completed**: 2025-11-28 (as part of Milestone 2)

**File**: `tests/animation/test_napari_backend.py`

**Tests Added** (in `TestRenderNapariArrayInput` class):

- `test_render_napari_accepts_2d_array`
- `test_render_napari_array_produces_correct_frame_count`
- `test_create_lazy_field_renderer_accepts_array`
- `test_create_lazy_field_renderer_array_frame_access`
- `test_create_lazy_field_renderer_array_vs_list_equivalence`
- `test_array_input_skips_validate_field_types`
- `test_memmap_input_accepted`

**Verification**: All 7 tests pass

---

### Task 7.2: Test Streaming Colormap Range ✅

**Completed**: 2025-11-28 (as part of Milestone 3)

**File**: `tests/animation/test_rendering.py`

**Tests Added** (in `TestColormapRangeStreaming` class):

- `test_small_array_uses_exact_computation`
- `test_large_array_uses_streaming`
- `test_sample_stride_reduces_computation`
- `test_streaming_with_nan_values`
- `test_streaming_with_manual_vmin_vmax`
- `test_default_threshold_is_50000`
- `test_memmap_streaming`

**Verification**: All 7 tests pass

---

### Task 7.3: Test Overlay Subsampling (DEFERRED)

**Status**: DEFERRED - Depends on Milestone 5 which is deferred.

**File**: `tests/animation/test_overlays.py` (extend)

**What to do**:

1. Add `test_overlay_subsample`
2. Test `overlay_subsample` parameter reduces output frames
3. Test `max_overlay_frames` caps output

**Success criteria**:

- Tests pass: `uv run pytest tests/animation/test_overlays.py -v -k subsample`

**Dependencies**: Milestone 5 (DEFERRED)

---

### Task 7.4: Run Full Test Suite ✅

**Completed**: 2025-11-28

**Results**:

1. Animation tests: 868 passed, 1 skipped
2. Full test suite: All passing (no regressions)
3. Mypy: Success - no issues found in 16 source files

---

## Implementation Order

Execute milestones in this order:

```
Milestone 1 (Core Array Preservation)
    ↓
Milestone 2 (Napari Backend Array Support)
    ↓
Milestone 3 (Scalable Colormap Range)
    ↓
Milestone 4 (Helper Functions) ←── Can start after Milestone 1
    ↓
Milestone 6 (Examples & Docs)
    ↓
Milestone 7 (Testing) ←── Throughout, finalize at end

~~Milestone 5 (Overlay Subsampling)~~ ←── DEFERRED
```

---

## Quick Commands

```bash
# Run animation tests
uv run pytest tests/animation/ -v

# Run specific test
uv run pytest tests/animation/test_napari_backend.py::TestNapariArrayInput -v

# Type check
uv run mypy src/neurospatial/animation/

# Lint
uv run ruff check src/neurospatial/animation/

# Test large session example
uv run python data/large_session_napari_example.py
```

---

## Verification Checklist ✅ COMPLETE

All items verified on 2025-11-28:

- [x] `uv run pytest tests/animation/` passes (868 passed, 1 skipped)
- [x] `uv run pytest tests/` passes (no regressions)
- [x] `uv run mypy src/neurospatial/animation/` passes (Success: no issues found in 16 source files)
- [x] `data/large_session_napari_example.py` updated with 100K+ frame support and helper functions
- [x] `data/view_bandit_napari.py` updated with best practices
- [x] Memory profiling confirmed memmaps not converted to list (Milestone 2 benchmarks)
- [x] Documentation updated (`docs/user-guide/animation.md`, `CLAUDE.md`)
- [x] Benchmark run: LazyFieldRenderer outperforms Dask by 20-45,000x
- [x] Decision documented: LazyFieldRenderer is the default (dask renderer removed)
