# Animation Optimization Scratchpad

**Started**: 2025-11-28

---

## Completed: Milestone 1 - Core Array Preservation

**Status**: COMPLETED

### Tasks Completed

- [x] **Task 1.1**: Preserve Array Format in animate_fields
- [x] **Task 1.2**: Update Field Validation for Arrays
- [x] **Task 1.3**: Ensure Multi-Field Mode Compatibility

### Implementation Summary

**File: `src/neurospatial/animation/core.py`**

1. Added `fields_is_array` flag to detect 2D+ numpy arrays early (line 206)
2. Modified field normalization to NOT convert arrays to list for napari backend
3. Added array-specific validation for `fields.shape[1] == env.n_bins`
4. Added conversion to list for non-napari backends after backend selection
5. Added type assertions to help mypy understand type narrowing
6. Added `# type: ignore[arg-type]` comments for backend calls (signatures updated in Milestone 2)

**File: `tests/animation/test_core.py`**

Added `TestArrayPreservation` class with 7 tests:

- `test_napari_backend_receives_array_not_list`
- `test_html_backend_receives_list_from_array`
- `test_video_backend_receives_list_from_array`
- `test_widget_backend_receives_list_from_array`
- `test_list_input_stays_list_for_napari`
- `test_list_input_stays_list_for_html`
- `test_memmap_preserved_for_napari`

### Code Review Summary

**Reviewer**: code-reviewer agent (2025-11-28)
**Result**: APPROVED

**Key findings:**

- Logic is sound and handles edge cases
- Tests are comprehensive and use proper mocking
- Backward compatibility fully maintained
- Code is readable with clear comments
- Type safety considered (mypy compliant)

**Minor suggestions (low priority, for future):**

- Add test for 3D array edge case
- Document memmap preservation in docstring
- Consider extracting array validation to helper function

---

## In Progress: Milestone 2 - Napari Backend Array Support

### Task 2.1: Update render_napari Type Hints ✅ COMPLETED

**Status**: COMPLETED (2025-11-28)

**File**: `src/neurospatial/animation/backends/napari_backend.py`

**Implementation**:

1. Updated `render_napari` signature to accept `NDArray[np.float64]` in type union
2. Added `fields_is_array = isinstance(fields, np.ndarray) and fields.ndim == 2`
3. Skip `_validate_field_types_consistent` for array inputs (no list iteration needed)
4. Skip `_is_multi_field_input` for array inputs (arrays can't be multi-field)
5. Updated docstring to document array mode as "recommended for large datasets"

**Tests Added** (`tests/animation/test_napari_backend.py`):

- `test_render_napari_accepts_2d_array`
- `test_render_napari_array_produces_correct_frame_count`
- `test_create_lazy_field_renderer_accepts_array`
- `test_create_lazy_field_renderer_array_frame_access`
- `test_create_lazy_field_renderer_array_vs_list_equivalence`
- `test_array_input_skips_validate_field_types`
- `test_memmap_input_accepted`

**Code Review**: APPROVED

- Implementation is minimal and surgical
- Duck typing handles array indexing correctly in internal functions
- Tests comprehensive (7 new tests, all passing)
- Backward compatible

### Task 2.2: Update _create_lazy_field_renderer ✅ COMPLETED

**Status**: COMPLETED (2025-11-28)

**File**: `src/neurospatial/animation/backends/napari_backend.py`

**Implementation**:

1. Updated function signature to accept `list[NDArray[np.float64]] | NDArray[np.float64]`
2. Changed `n_frames = len(fields)` to `fields.shape[0] if isinstance(fields, np.ndarray) else len(fields)`
3. Updated docstring to document array mode

### Task 2.3: Update LazyFieldRenderer for Array Input ✅ COMPLETED

**Status**: COMPLETED (2025-11-28)

**File**: `src/neurospatial/animation/backends/napari_backend.py`

**Implementation**:

1. Updated `__init__` signature to accept `list[NDArray[np.float64]] | NDArray[np.float64]`
2. Added `_fields_is_array` flag to track input type
3. Updated `__len__` with isinstance check for mypy type narrowing
4. Updated all uses of `len(self.fields)` to `len(self)` in `_getitem_locked`, `_get_frame`, `shape`
5. Updated docstring to document array mode

### Task 2.4: Update ChunkedLazyFieldRenderer for Array Input ✅ COMPLETED

**Status**: COMPLETED (2025-11-28)

**File**: `src/neurospatial/animation/backends/napari_backend.py`

**Implementation**:

1. Same pattern as LazyFieldRenderer
2. Updated `__init__` signature and added `_fields_is_array` flag
3. Updated `__len__` with isinstance check for mypy type narrowing
4. Updated all uses of `len(self.fields)` to `len(self)` in `_render_chunk`, `_getitem_locked`, `_get_frame`, `shape`
5. Updated docstring to document array mode

**Tests**: All 39 napari backend tests pass, all 42 core animation tests pass

**Verification**:

- `uv run pytest tests/animation/test_napari_backend.py -v` → 39 passed, 1 skipped
- `uv run pytest tests/animation/test_core.py -v` → 42 passed
- `uv run ruff check src/neurospatial/animation/backends/napari_backend.py` → All checks passed
- `uv run mypy src/neurospatial/animation/backends/napari_backend.py` → Success: no issues found

### Task 2.5: Implement Dask-Based Alternative Renderer ✅ COMPLETED

**Status**: COMPLETED (2025-11-28)

**File**: `src/neurospatial/animation/backends/napari_backend.py`

**Implementation**:

1. Added `_create_dask_field_renderer()` function with comprehensive validation:
   - Validates input is numpy array (not list)
   - Validates 2D shape (n_frames, n_bins)
   - Validates at least 1 frame
   - Validates chunk_size > 0
2. Uses `dask.array.from_array()` with chunking along time axis
3. Uses `dask.array.map_blocks()` to lazily apply `field_to_rgb_for_napari`
4. Returns dask array that napari consumes directly
5. Added `use_dask: bool = False` parameter to `render_napari`

**Tests Added** (`tests/animation/test_napari_backend.py`):

- `test_create_dask_field_renderer_returns_dask_array`
- `test_dask_renderer_has_correct_shape`
- `test_dask_renderer_has_correct_dtype`
- `test_dask_renderer_frame_access`
- `test_dask_renderer_lazy_until_compute`
- `test_dask_renderer_with_memmap`
- `test_dask_renderer_vs_lazy_renderer_equivalence`
- `test_dask_renderer_chunk_size_parameter`
- `test_dask_renderer_requires_array_input`
- `test_dask_renderer_empty_array_raises_error`
- `test_dask_renderer_wrong_shape_raises_error`
- `test_dask_renderer_invalid_chunk_size_raises_error`
- `test_render_napari_use_dask_parameter`
- `test_render_napari_use_dask_requires_array`

**Code Review**: APPROVED with enhancements

- Initial review identified critical edge cases (empty array, wrong shape)
- Added comprehensive input validation after review
- All 14 tests pass, all 53 napari backend tests pass

**Verification**:

- `uv run pytest tests/animation/test_napari_backend.py -v` → 53 passed, 1 skipped
- `uv run ruff check src/neurospatial/animation/backends/napari_backend.py` → All checks passed
- `uv run mypy src/neurospatial/animation/backends/napari_backend.py` → Success: no issues found

### Task 2.6: Benchmark LazyFieldRenderer vs Dask ✅ COMPLETED

**Status**: COMPLETED (2025-11-28)

**File**: `benchmarks/bench_lazy_renderers.py`

**Implementation**:

1. Created comprehensive benchmark script comparing:
   - Memory overhead during renderer creation
   - Random access frame retrieval time
   - Sequential access frame retrieval time
   - Scrubbing simulation (timeline navigation)

2. Tested configurations:
   - Small (1K frames), Medium (10K frames), Large (100K frames)
   - Both in-memory arrays and memory-mapped arrays

**Benchmark Results Summary**:

| Config | Metric | LazyFieldRenderer | Dask | Winner |
|--------|--------|-------------------|------|--------|
| small (1K frames) | Creation | 0.01 ms | 13.75 ms | **Lazy** (1375x faster) |
| small | Random Access | 0.02 ms | 1.08 ms | **Lazy** (54x faster) |
| medium (10K frames) | Creation | 0.01 ms | 42.45 ms | **Lazy** (4245x faster) |
| medium | Random Access | 0.02 ms | 1.26 ms | **Lazy** (63x faster) |
| large (100K frames) | Creation | 0.01 ms | 450 ms | **Lazy** (45000x faster) |
| large | Random Access | 0.20 ms | 4.27 ms | **Lazy** (21x faster) |
| large | Sequential | 0.02 ms | 4.40 ms | **Lazy** (220x faster) |
| large_memmap | Creation | 0.01 ms | 313 ms | **Lazy** (31300x faster) |
| large_memmap | Random Access | 0.21 ms | 5.91 ms | **Lazy** (28x faster) |

**Key Findings**:

1. **LazyFieldRenderer is significantly faster** in all metrics
2. LazyFieldRenderer has **zero memory overhead** during creation
3. Dask has higher overhead due to graph construction and per-chunk scheduling
4. The performance gap **widens with larger datasets**
5. Memory-mapped arrays show similar patterns to in-memory arrays

**Recommendation**:

**Use LazyFieldRenderer as the default.** It is:

- 20-45,000x faster for creation
- 20-220x faster for frame access
- Zero memory overhead
- No additional dependency
- Proven in production

### Decision: Remove Dask Renderer (2025-11-28)

Based on the benchmark results showing LazyFieldRenderer dramatically outperforming dask (20-45,000x faster creation, 20-220x faster access), the dask renderer has been **removed entirely** from the codebase.

**Rationale**:

1. LazyFieldRenderer is strictly superior in all measured metrics
2. No compelling use case for the dask option
3. Simplifies codebase by removing unused code path
4. Removes dask as an optional dependency

**Removed**:

- `_create_dask_field_renderer()` function from `napari_backend.py`
- `use_dask: bool = False` parameter from `render_napari()`
- 14 dask-related tests from `test_napari_backend.py`
- Dask benchmarking code from `bench_lazy_renderers.py`

**Verification**:

- All 81 animation tests pass (40 napari + 41 core)
- Linting and mypy pass
- Benchmark file simplified to focus on LazyFieldRenderer performance profiling

---

## Completed: Milestone 2 - Napari Backend Array Support

**Status**: COMPLETED

All 6 tasks completed:

- [x] Task 2.1: Update render_napari Type Hints
- [x] Task 2.2: Update _create_lazy_field_renderer
- [x] Task 2.3: Update LazyFieldRenderer for Array Input
- [x] Task 2.4: Update ChunkedLazyFieldRenderer for Array Input
- [x] Task 2.5: Implement Dask-Based Alternative Renderer
- [x] Task 2.6: Benchmark LazyFieldRenderer vs Dask

---

## Next: Milestone 3 - Scalable Colormap Range Computation

### Upcoming Tasks

- [ ] Task 3.1: Add Streaming Path to compute_global_colormap_range
- [ ] Task 3.2: Update render_napari to Use Streaming
- [ ] Task 3.3: Always Set multiscale=False

---

## Blockers

None currently.

---

## Decisions Made

1. **Array preservation strategy**: Use `fields_is_array` boolean flag to track input format, defer list conversion until after backend selection
2. **Type narrowing approach**: Use `assert isinstance()` with `# nosec` comment to help mypy understand type narrowing
3. **Backend type ignores**: Use `# type: ignore[arg-type]` for backend calls since updating signatures is Milestone 2
4. **isinstance in __len__**: Use `isinstance(self.fields, np.ndarray)` directly in `__len__` for mypy type narrowing, even though we have `_fields_is_array` flag
5. **int() cast for shape[0]**: Cast `self.fields.shape[0]` to `int()` to satisfy mypy's `no-any-return` check
6. **Dask renderer validation**: Added comprehensive input validation (array type, 2D shape, n_frames > 0, chunk_size > 0) after code review identified edge case bugs
7. **Remove dask renderer**: After benchmarking showed LazyFieldRenderer is 20-45,000x faster for creation and 20-220x faster for access, removed dask renderer entirely (no `use_dask` parameter)

---

## Questions

(None currently)
