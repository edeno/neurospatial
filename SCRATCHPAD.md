# SCRATCHPAD.md

## Current Task: Milestone 1.1 - Create `_build_skeleton_vectors`

### Understanding the Problem

**Root cause of napari playback "stuck" issue:**
- Skeleton callback's `layer.data` assignment takes **5.38ms per frame** (99% of callback time)
- This blocks the Qt event loop during playback

**Solution:** Replace per-frame Shapes layer updates with precomputed Vectors layer

### Key Insights from Code Review

1. **Current implementation:**
   - `_create_skeleton_frame_data()` computes skeleton for single frame
   - `_setup_skeleton_update_callback()` registers callback to update shapes layer on frame change
   - Uses Shapes layer with line shape_type

2. **New approach:**
   - Precompute ALL skeleton vectors at initialization (not per-frame)
   - Use napari Vectors layer which natively handles time slicing
   - Format: `(n_segments, 2, 3)` where each segment is `[[t, y0, x0], [t, y1, x1]]`

3. **Key coordinate transforms:**
   - `_transform_coords_for_napari()` transforms (x, y) env coords → (row, col) napari coords
   - Uses `_EnvScale` for cached scale factors

### Test Strategy

Write tests for `_build_skeleton_vectors` that verify:
1. Returns correct shape `(n_frames * n_edges, 2, 3)`
2. Time stamps are correct
3. Coordinate transforms are applied
4. Empty/None skeleton returns empty array
5. Features dict contains edge names

### Progress

- [x] Read existing implementation
- [x] Read existing tests
- [x] Write tests for `_build_skeleton_vectors` (15 tests in `tests/animation/test_skeleton_vectors.py`)
- [x] Implement `_build_skeleton_vectors`
- [x] Update TASKS.md

### Implementation Notes

**Function signature:**
```python
def _build_skeleton_vectors(
    bodypart_data: BodypartData,
    env: Environment,
    *,
    dtype: type[np.floating] = np.float32,
) -> tuple[NDArray[np.floating], dict[str, NDArray[np.object_]]]:
```

**Key implementation decisions:**
1. Uses float32 by default for memory efficiency (70k segments = ~1.68 MB)
2. Pre-transforms all bodypart coords once per bodypart (avoiding per-frame transform)
3. Handles NaN endpoints by excluding those segments
4. Handles missing bodyparts in skeleton edges gracefully
5. Returns `(vectors, features)` where features has `edge_name` array

**Test coverage:**
- Shape validation (n_frames * n_edges, 2, 3)
- Time stamp correctness
- Coordinate transformation
- Empty/None skeleton handling
- NaN handling
- Missing bodypart handling
- Large dataset performance (10k, 100k frames)

## Completed: Milestone 2.1 - Update `_render_bodypart_overlay` to Use Vectors

### Changes Made

1. **Updated `_render_bodypart_overlay`** (lines 819-837 in napari_backend.py):
   - Removed: `_create_skeleton_frame_data()` call for frame 0
   - Removed: `viewer.add_shapes()` for skeleton
   - Removed: `_setup_skeleton_update_callback()` registration
   - Added: `_build_skeleton_vectors()` call to precompute all vectors
   - Added: `viewer.add_vectors()` with features for skeleton

2. **Updated tests** (test_napari_overlays.py):
   - `test_bodypart_overlay_creates_layers` - expects `add_vectors` instead of `add_shapes`
   - `test_bodypart_overlay_skeleton_as_precomputed_vectors` - new test for vectors shape
   - `test_bodypart_overlay_skeleton_color_and_width` - checks vectors kwargs
   - `test_bodypart_overlay_without_skeleton` - checks `add_vectors.call_count == 0`
   - `test_mixed_overlay_types` - expects `add_vectors.call_count >= 2`
   - `test_bodypart_skeleton_all_nan_no_vectors_layer` - new edge case test

### Key Implementation Details

```python
# In _render_bodypart_overlay (lines 819-837)
if bodypart_data.skeleton is not None:
    # Precompute all skeleton vectors at initialization
    vectors_data, vector_features = _build_skeleton_vectors(bodypart_data, env)

    # Only add layer if there are valid skeleton segments
    if vectors_data.size > 0:
        skeleton_layer = viewer.add_vectors(
            vectors_data,
            name=f"Skeleton{name_suffix}",
            edge_color=bodypart_data.skeleton_color,
            edge_width=bodypart_data.skeleton_width,
            features=vector_features,
        )
        layers.append(skeleton_layer)
```

### Test Results

- All 44 napari overlay tests pass
- All 15 skeleton vectors tests pass
- Code quality: ruff and mypy pass

### Notes

- The old functions `_create_skeleton_frame_data` and `_setup_skeleton_update_callback` still exist but are no longer called by `_render_bodypart_overlay`
- These will be removed in Milestone 4 (Cleanup and Removal of Old Code Path)
- The napari Vectors layer handles time slicing natively via dims

## Completed: Milestone 4 - Cleanup and Removal of Old Code Path

### Changes Made

1. **Deleted dead code**:
   - Removed `_create_skeleton_frame_data` function (~60 lines)
   - Removed `_setup_skeleton_update_callback` function (~95 lines)
   - Removed unused `weakref` import (auto-fixed by ruff)

2. **Updated docstrings**:
   - Removed "See Also" reference to deprecated `_create_skeleton_frame_data` in `_build_skeleton_vectors`

### Notes

- Tests in Milestone 2 already updated to expect vectors instead of shapes for skeleton
- All 462 animation tests pass
- Mypy passes with no issues

## Completed: Phase 0.1 - Add Timing Instrumentation

### What Was Done

1. **Napari backend timing** (via perfmon_config.json):
   - Added `_build_skeleton_vectors` to callable tracing
   - Added `_render_bodypart_overlay` to callable tracing
   - Added `_render_head_direction_overlay` to callable tracing
   - Added `_render_position_overlay` to callable tracing
   - Use: `NAPARI_PERFMON=scripts/perfmon_config.json uv run python script.py`

2. **Created `_timing.py` module** for video/widget backends:
   - `timing(name)` context manager for timing code blocks
   - `timed(func)` decorator for timing function calls
   - Enabled via `NEUROSPATIAL_TIMING=1` environment variable
   - No-op with minimal overhead when disabled

3. **Added timing to video backend**:
   - Wrapped `render_field_to_rgb` in `rendering.py`

4. **Added timing to widget backend**:
   - Wrapped `render_field_to_png_bytes_with_overlays`
   - Wrapped `PersistentFigureRenderer.render_savefig` (PNG save path)

### How to Use

```bash
# Napari backend profiling (full tracing)
NAPARI_PERFMON=scripts/perfmon_config.json uv run python scripts/test_napari_perfmon.py

# Video/Widget backend timing
NEUROSPATIAL_TIMING=1 uv run python your_script.py
# Output: [TIMING] render_field_to_rgb: 123.45 ms
```

### Notes

- Napari uses its own perfmon infrastructure (callable tracing via JSON config)
- Video/widget use custom `_timing` module (simpler, environment-variable controlled)
- Both approaches have minimal overhead when disabled

## Completed: Phase 0.2 - Create Benchmark Datasets

### What Was Done

1. **Created `scripts/benchmark_datasets/` package**:
   - `scripts/benchmark_datasets/__init__.py` - Package exports
   - `scripts/benchmark_datasets/datasets.py` - Dataset generators
   - Moved to scripts/ to avoid polluting pytest pythonpath config

2. **Implemented BenchmarkConfig dataclass** with pre-defined configs:
   - `SMALL_CONFIG`: 100 frames, 40x40 grid, position overlay only
   - `MEDIUM_CONFIG`: 5k frames, 100x100 grid, all overlays
   - `LARGE_CONFIG`: 100k frames, 100x100 grid, all overlays (7 bodyparts)

3. **Generator functions**:
   - `create_benchmark_env(config, seed)` - Creates Environment
   - `create_benchmark_fields(env, config, seed, memmap_path)` - Creates drifting Gaussian blobs
   - `create_benchmark_overlays(env, config, seed)` - Creates position, bodypart, head direction

4. **Tests**: 22 tests in `tests/animation/test_benchmark_datasets.py`
   - Config value validation
   - Shape/type/reproducibility tests
   - Edge cases (head-direction-only, no overlays)
   - Uses localized sys.path manipulation to import from scripts/

### Implementation Notes

- Uses `np.random.default_rng(seed)` for reproducibility
- Supports memory-mapped arrays for large datasets
- Smooth trajectory generation with boundary reflection
- Head direction angles wrapped to [-pi, pi]

### Key Files

- `scripts/benchmark_datasets/__init__.py` - Package init
- `scripts/benchmark_datasets/datasets.py` - All generators
- `tests/animation/test_benchmark_datasets.py` - 22 tests

## Completed: Phase 0.3 - Record Baseline Metrics

### What Was Done

1. **Created benchmark scripts** in `benchmarks/` directory:
   - `benchmarks/__init__.py` - Package docs
   - `benchmarks/utils.py` - Shared timing/memory utilities
   - `benchmarks/bench_napari.py` - Napari backend benchmark
   - `benchmarks/bench_video.py` - Video backend benchmark
   - `benchmarks/bench_widget.py` - Widget backend benchmark

2. **Recorded baseline metrics** for all backends:
   - Small config: 100 frames, 40x40 grid, position overlay
   - Medium config: 5k frames (truncated to 500 for benchmarks)
   - Large config: 100k frames (truncated to 500 for benchmarks)

3. **Documented results** in `benchmarks/BASELINE.md`

### Key Baseline Metrics

| Backend | Init Time | Frame Time | Notes |
|---------|-----------|------------|-------|
| Napari (small) | 2,585 ms | 4.38 ms/seek | Position overlay |
| Napari (medium) | 2,974 ms | 15.80 ms/seek | All overlays |
| Video (small) | N/A | 29.95 ms/frame | Position overlay |
| Video (500 frames) | N/A | 15.77 ms/frame | 1.99x parallel speedup |
| Widget (small) | 15.39 ms | 8.93 ms/frame | Position overlay |
| Widget (500 frames) | 7.44 ms | 10.00 ms/frame | No overlays |

### Notes

- Overlay truncation is complex due to skeleton relationships - benchmarks skip overlays when truncating
- Napari viewer init is ~2.5-3s regardless of frame count
- Widget PersistentFigureRenderer reuse is effective (~9-10ms/frame)
- Parallel video rendering gives ~2x speedup for 500+ frames

## Completed: Phase 1.1 - Centralize Coordinate Transforms

### What Was Done

1. **Created `neurospatial/animation/transforms.py`** with shared coordinate transforms:
   - `EnvScale` class - cached scale factors for coordinate transformation
   - `make_env_scale(env)` - convenience factory function
   - `transform_coords_for_napari(coords, env_or_scale)` - transforms (x, y) → (row, col)
   - `transform_direction_for_napari(direction, env_or_scale)` - transforms direction vectors
   - `reset_transform_warning()` - resets fallback warning flag (for testing)

2. **Added comprehensive tests** in `tests/animation/test_transforms.py`:
   - 19 tests covering EnvScale, coordinate transforms, direction transforms
   - Tests for consistency between env and pre-computed scale
   - Tests for fallback behavior when env lacks required attributes

3. **Updated napari backend** to use shared functions:
   - Added imports: `EnvScale as _EnvScale`, `make_env_scale as _make_env_scale`,
     `transform_coords_for_napari as _transform_coords_for_napari`,
     `transform_direction_for_napari as _transform_direction_for_napari`
   - Removed ~230 lines of duplicate code from napari_backend.py

4. **Updated existing test**:
   - Fixed `test_env_scale_repr` to expect "EnvScale" in repr (not "_EnvScale")

### Key Design Decisions

1. **Public API**: The transforms module is a public API (`EnvScale`, not `_EnvScale`)
2. **Backward compatibility**: napari_backend re-exports with underscore prefix for internal use
3. **Single warning flag**: Module-level flag `_TRANSFORM_FALLBACK_WARNED` for once-per-session warning
4. **Memory efficient**: Uses `__slots__` for EnvScale to minimize memory overhead

### Test Results

- All 19 new transform tests pass
- All 67 napari tests pass (1 skipped)
- All 462 animation tests pass
- Mypy passes with no issues

## Completed: Phase 1.2 - Normalize Layout Metadata

### What Was Done

1. **Added `layout_type` and `is_grid_compatible` properties to LayoutEngine protocol** (`src/neurospatial/layout/base.py`):
   - `layout_type`: Returns categorical string identifying the layout type
   - `is_grid_compatible`: Returns True if layout can be rendered as 2D image

2. **Implemented properties in all 7 layout engines**:
   | Layout | layout_type | is_grid_compatible |
   |--------|-------------|-------------------|
   | RegularGridLayout | "grid" | True |
   | MaskedGridLayout | "mask" | True |
   | ImageMaskLayout | "mask" | True |
   | ShapelyPolygonLayout | "polygon" | True |
   | HexagonalLayout | "hexagonal" | False |
   | TriangularMeshLayout | "mesh" | False |
   | GraphLayout | "graph" | False |

3. **Updated widget backend** (`_field_to_image_data`):
   - Replaced hardcoded `_layout_type_tag` check against list
   - Now uses `is_grid_compatible` property

4. **Updated napari rendering** (`field_to_rgb_for_napari`):
   - Uses `is_grid_compatible` for grid detection
   - Maintains explicit 2D grid_shape check for safety

5. **Added comprehensive tests** (`tests/layout/test_layout_type.py`):
   - 32 tests covering all layout engines
   - Tests work both before and after `build()` is called
   - Integration tests for widget and rendering backends

### Key Design Decisions

1. **Two separate properties**: `layout_type` for semantic categorization, `is_grid_compatible` for rendering decisions
2. **Properties work before `build()`**: Static classification doesn't require layout to be built
3. **Backwards compatible**: Uses `getattr(layout, "is_grid_compatible", False)` for graceful degradation

### Test Results

- 32 layout_type tests pass
- 220 layout tests pass
- All rendering and widget tests pass
- Code review: APPROVED

## Completed: Phase 1.3 - Verify No Regressions

### Benchmark Results (Post Phase 1.2)

**Widget Benchmark (small):**
| Metric | Baseline | Current | Status |
|--------|----------|---------|--------|
| First render | 15.39 ms | 18.71 ms | Within noise |
| Avg render | 8.93 ms | 8.94 ms | Same |

**Video Benchmark (small):**
| Metric | Baseline | Current | Status |
|--------|----------|---------|--------|
| Export serial | 2,995 ms | 2,983.80 ms | Same |
| Time/frame | 29.95 ms | 29.84 ms | Same |
| Parallel speedup | 1.09x | 1.09x | Same |

**Conclusion:** No performance regressions detected after Phase 1.2 changes.

### Visual Verification

Visual alignment verification requires interactive testing in a graphical environment.
This should be done manually before merging major changes.

---

## Phase 1 Complete!

All Phase 1 (Shared Infrastructure Cleanup) tasks are now complete:
- 1.1 Centralize Coordinate Transforms ✅
- 1.2 Normalize Layout Metadata ✅
- 1.3 Verify No Regressions ✅

## Completed: Phase 2.1 - Vectorize `_build_skeleton_vectors`

### What Was Done

1. **Profiled baseline performance** on medium (5k frames) and large (100k frames) datasets
2. **Replaced Python loops with vectorized NumPy operations**:
   - Changed inner `for frame_idx in range(n_frames)` loop to vectorized boolean mask
   - Used `np.where(valid_mask)` to get valid frame indices in one operation
   - Used array slicing to extract valid coordinates all at once
   - Built vectors array using array assignment instead of element-by-element
   - Used `np.concatenate` to combine all edge results at the end
3. **Removed unused `n_frames` variable** caught by ruff

### Performance Results

| Config | Before (ms) | After (ms) | Speedup |
|--------|-------------|------------|---------|
| Medium (5k frames, 4 edges) | 98.75 | 2.11 | **46.8x** |
| Large (100k frames, 6 edges) | 2628.52 | 62.35 | **42.2x** |

**Result**: Achieved 42-47x speedup, far exceeding the target of 5-20x!

### Test Results

- All 15 existing skeleton vectors tests pass
- All 77 napari backend tests pass
- ruff and mypy pass with no issues

### Key Code Changes

**Before** (slow nested loops):
```python
for frame_idx in range(n_frames):
    start_point = start_coords[frame_idx]
    end_point = end_coords[frame_idx]
    if np.any(np.isnan(start_point)) or np.any(np.isnan(end_point)):
        continue
    valid_segments.append((frame_idx, ...))
```

**After** (vectorized):
```python
valid_mask = ~np.isnan(start_coords).any(axis=1) & ~np.isnan(end_coords).any(axis=1)
valid_frame_indices = np.where(valid_mask)[0]
valid_start = start_coords[valid_frame_indices]
valid_end = end_coords[valid_frame_indices]
# Build vectors using array operations
```

## Completed: Phase 2.2 - Fix Transform Fallback Warning State

### What Was Done

1. **Added `suppress_warning` parameter** to transform functions in `transforms.py`:
   - `transform_coords_for_napari(coords, env_or_scale, *, suppress_warning=False)`
   - `transform_direction_for_napari(direction, env_or_scale, *, suppress_warning=False)`
   - Updated `_warn_fallback(suppress=False)` to accept suppression

2. **Added per-viewer warning tracking** in `napari_backend.py`:
   - `_TRANSFORM_WARNED_KEY` - metadata key for tracking
   - `_check_viewer_warned(viewer)` - check if viewer has warned
   - `_mark_viewer_warned(viewer)` - mark viewer as having warned
   - `_transform_coords_with_viewer(coords, env, viewer)` - wrapper with tracking
   - `_transform_direction_with_viewer(direction, env, viewer)` - wrapper with tracking

3. **Added tests** in `tests/animation/test_transform_per_viewer_warning.py`:
   - 15 new tests covering all scenarios
   - Tests for suppress_warning parameter
   - Tests for per-viewer state management
   - Integration tests for viewer tracking wrappers
   - Module-level fallback behavior tests

### Key Design Decisions

1. **Dual-layer approach**:
   - Module-level `_TRANSFORM_FALLBACK_WARNED` flag provides session-level safety net
   - Per-viewer tracking via `viewer.metadata` allows each viewer to warn once

2. **Backward compatible**:
   - `suppress_warning` is keyword-only with default `False`
   - Existing code works unchanged

3. **Wrapper functions** instead of modifying all call sites:
   - `_transform_coords_with_viewer` and `_transform_direction_with_viewer`
   - Can be used in `_render_*` functions when per-viewer tracking is needed

### Test Results

- All 15 new tests pass
- All 63 related tests pass (transforms + napari overlays)
- ruff and mypy pass

## Completed: Phase 2.3 - Tracks Color Handling Investigation

### Investigation Summary

**Goal**: Use `features` + `color_by="color"` at layer creation instead of post-creation workaround.

**Finding**: The current workaround is CORRECT and still necessary in napari 0.5.6.

### Technical Details

1. **Verified napari 0.5.6 behavior**:
   - `color_by` IS a constructor parameter for Tracks layer
   - BUT passing it at init time triggers warning:
     `UserWarning: Previous color_by key 'color' not present in features. Falling back to track_id`

2. **Root cause** (documented in existing code comment):
   - During `__init__`, napari's data setter resets features to `{}` before our features are applied
   - If `color_by` is passed at init time, the check runs against empty features and warns

3. **Current workaround is correct**:
   ```python
   # Create layer WITHOUT color_by
   layer = viewer.add_tracks(track_data, features=features, colormaps_dict=colormaps_dict)
   layer.color_by = "color"  # Set AFTER creation - no warning!
   ```

### Test Added

Added `test_position_overlay_trail_color_by_workaround` to document and lock in this behavior:
- Verifies `color_by` NOT passed as kwarg at creation
- Verifies `features` and `colormaps_dict` ARE passed at creation
- Verifies `layer.color_by` set to "color" via post-creation assignment

### Test Results

- New test passes
- All 30 napari overlay tests pass
- ruff and mypy pass

## Completed: Phase 2.4 - Playback Widget Throttling Fix

### Problem

The `update_frame_info` function in `_add_speed_control_widget` was throttling updates
even when scrubbing (not playing). This caused unresponsive feedback when manually
navigating through frames at high FPS settings.

### Root Cause

The throttling logic always applied:
```python
# Old code (always throttled)
if (
    current_frame % update_interval != 0
    and current_frame != playback_state["last_frame"]
):
    return  # Skipped update
```

### Fix

Added check for `is_playing` to only throttle during playback:
```python
# New code (only throttle during playback)
if (
    playback_state["is_playing"]
    and current_frame % update_interval != 0
    and current_frame != playback_state["last_frame"]
):
    return  # Only skip during playback
```

### Test Added

Added `test_playback_widget_scrubbing_updates_immediately`:
- Mocks viewer and magicgui to capture the update callback
- Simulates frame changes when not playing (scrubbing)
- Verifies ALL frame changes result in updates (no throttling)

### Test Results

- New test passes
- All 32 napari backend tests pass (1 skipped)
- ruff and mypy pass

## Completed: Phase 2.5 - Re-profile Napari

### Benchmark Results

**Skeleton Vector Generation** (via `bench_skeleton_vectors.py`):

| Config | Baseline (ms) | Current (ms) | Speedup |
|--------|---------------|--------------|---------|
| Medium (5k frames) | 98.75 | 1.99 | **49.6x** |
| Large (100k frames) | 2628.52 | 61.82 | **42.5x** |

**Napari Viewer** (via `bench_napari.py`):

| Config | Init (baseline) | Init (current) | Seek (baseline) | Seek (current) |
|--------|-----------------|----------------|-----------------|----------------|
| Small | 2,585 ms | 2,607 ms | 4.38 ms | 4.38 ms |
| Medium | 2,974 ms | 2,978 ms | 15.80 ms | 15.81 ms |

### Analysis

1. **Skeleton vectors 42-50x faster**: The vectorized implementation delivers massive speedup
2. **Viewer init unchanged**: Skeleton generation is small part of total init time
3. **Random seek unchanged**: Expected - seek performance wasn't the optimization target
4. **Main benefit**: Eliminated per-frame Shapes layer callback (5.38ms/frame → 0ms via native Vectors layer)

### Phase 2 Summary

All Phase 2 tasks complete:

- 2.1: Vectorized `_build_skeleton_vectors` (42-50x speedup)
- 2.2: Per-viewer transform fallback warning tracking
- 2.3: Verified tracks `color_by` workaround is correct
- 2.4: Fixed playback widget throttling (scrubbing now immediate)
- 2.5: Re-profiled and documented improvements

## Completed: Phase 3.1 - Verify Interpolation Vectorization (2025-11-21)

### Investigation Result

**Finding**: Interpolation was ALREADY vectorized in the existing implementation!

The PLAN.md/TASKS.md description mentioned "per-point loops" but the actual code already uses:
- `np.interp` for 1D and per-dimension interpolation
- Boolean masking for extrapolation handling
- The only loops iterate over dimensions (2-3), not frames (100k+)

### Performance Verification (100k frames)

| Function | Time |
|----------|------|
| `_interp_linear` 1D | 0.89 ms |
| `_interp_linear` 2D | 2.10 ms |
| `_interp_nearest` 2D | 54.29 ms |
| Full conversion (3 overlays) | 12.47 ms |

## Completed: Phase 3.2 - Verify Validation Vectorization (2025-11-21)

### Investigation Result

**Finding**: Validation functions were ALREADY vectorized!

- `_validate_finite_values`: uses `np.isfinite`, `np.sum`, `np.argmax`
- `_validate_bounds`: loops over dimensions (2-3), vectorized across points
- `_validate_monotonic_time`: uses `np.diff`, `np.where`

### Performance Verification (100k points)

| Function | Time |
|----------|------|
| `_validate_finite_values` | 0.114 ms |
| `_validate_shape` | 0.002 ms |
| `_validate_bounds` | 0.232 ms |
| `_validate_monotonic_time` | 0.142 ms |

## Phase 3.3 - Skipped (Premature Optimization)

**Decision 2025-11-21**: Conversion takes only 12ms for 100k frames. Caching would add
complexity without meaningful benefit.

## Completed: Phase 3.4 - Harden Multi-Field Detection (2025-11-21)

### Problem

The old implementation only checked `fields[0]` to determine single vs multi-field:

```python
is_multi_field = len(fields) > 0 and isinstance(fields[0], (list, tuple))
```

This would incorrectly process mixed inputs like `[list, array, list]` and fail with
confusing error messages ("same length" instead of "mixed types").

### Solution

Added `_validate_field_types_consistent()` that checks ALL elements:

```python
def _validate_field_types_consistent(fields: list) -> None:
    """Validate all elements are either arrays or lists/tuples (not mixed)."""
    ...
    if mismatched_indices:
        raise ValueError(
            "WHAT: Inconsistent field types detected - mixed arrays and sequences.\n"
            ...
        )
```

### Test Results

- 14 new tests in `tests/animation/test_multi_field_detection.py`
- 17 existing tests in `tests/animation/test_napari_multi_field.py`
- All 31 tests pass

## Completed: Phase 4.1 - Clarify Fallback in PersistentFigureRenderer (2025-11-21)

### Problem

The `PersistentFigureRenderer` was designed to use `set_data()` optimization but it never worked:
- `plot_field` uses `pcolormesh` which creates `QuadMesh` in `ax.collections`
- The code was looking for `AxesImage` in `ax.images` (used by `imshow`)
- Result: EVERY render was a full re-render, even for grid layouts

### Solution

1. **Fixed optimization to work with pcolormesh**:
   - Now looks for `QuadMesh` in `ax.collections` instead of `AxesImage` in `ax.images`
   - Uses `QuadMesh.set_array()` instead of `AxesImage.set_data()`
   - Added `_field_to_mesh_array()` method for QuadMesh-compatible flat array

2. **Added fallback logging**:
   - Non-grid layouts trigger DEBUG log explaining why fallback is required
   - Message includes layout type and reason

3. **Added debug flag**:
   - `raise_on_fallback=True` parameter raises RuntimeError instead of silent fallback
   - Useful for debugging performance issues

### Key Changes

```python
# Before: Looking for wrong type, never found anything
if self._ax.images:
    self._image = self._ax.images[0]  # Always empty for pcolormesh!

# After: Find QuadMesh from pcolormesh
for collection in self._ax.collections:
    if isinstance(collection, self._QuadMesh):
        self._mesh = collection
        break
```

### Test Results

- 13 new tests in `tests/animation/test_widget_fallback.py`
- All 18 widget backend tests pass
- All 540 animation tests pass

## Completed: Phase 4.2 - Stabilize Overlay Artist Lifecycle (2025-11-21)

### Investigation Result

**Finding**: The existing `OverlayArtistManager` implementation was mostly correct, but tests exposed a critical bug!

### Bug Found and Fixed

**Root cause**: In `_initialize_bodypart_overlay`, `_update_bodypart_skeleton` was called with an index BEFORE the skeleton was appended to the list, causing IndexError.

**Before** (buggy):
```python
# Line 596-598: Call update with index == len() (doesn't exist yet!)
self._update_bodypart_skeleton(len(self._bodypart_skeletons), bodypart_data, frame_idx)
# Line 599: Append happens AFTER the call!
self._bodypart_skeletons.append(skeleton_lc)
```

**After** (fixed):
```python
# Line 597: Append FIRST
self._bodypart_skeletons.append(skeleton_lc)
# Line 599-601: Now update with valid index (len() - 1)
if skeleton_lc is not None:
    self._update_bodypart_skeleton(len(self._bodypart_skeletons) - 1, bodypart_data, frame_idx)
```

### New Tests

Created `tests/animation/test_overlay_artist_manager.py` with 30 tests:

| Test Class | Count | Coverage |
|------------|-------|----------|
| TestOverlayArtistManagerInitialization | 9 | Initialization, idempotency, artist creation |
| TestOverlayArtistManagerUpdate | 6 | set_offsets, set_segments, quiver recreation |
| TestOverlayArtistManagerClear | 5 | Artist removal, flag reset, reinitialize |
| TestOverlayArtistManagerEdgeCases | 7 | NaN data, empty lists, multi-animal, no skeleton |
| TestOverlayArtistManagerIntegration | 3 | PersistentFigureRenderer integration |

### Key Verified Behaviors

1. **Manager created once per figure**: Verified with integration tests
2. **Uses set_data/set_offsets**: All overlay types use efficient updates, not recreation
3. **Fallback path correct**: New manager created after ax.clear() (necessary, as old artists are destroyed)
4. **Clear method works**: Properly removes artists and resets `_initialized` flag

### Test Results

- All 30 new tests pass
- All 600 animation tests pass
- ruff and mypy pass

## Completed: Phase 4.3 - Optional JPEG Support (2025-11-22)

### What Was Done

Added optional JPEG image format support to the widget backend, allowing users to choose between PNG (lossless, larger) and JPEG (lossy, smaller) for cached frames.

### Implementation

1. **Added `image_format` parameter to three functions/classes**:
   - `render_field_to_png_bytes_with_overlays()` - main rendering function
   - `PersistentFigureRenderer` - persistent renderer class
   - `render_widget()` - main widget entry point

2. **JPEG rendering implementation**:
   - Uses PIL (Pillow) for JPEG compression
   - Quality: 85 (good balance of quality and size)
   - Optimize: True (smaller file size)
   - Graceful ImportError with clear installation instructions

3. **Validation**:
   - Case-insensitive format comparison (accepts "JPEG", "jpeg", "PNG", "png")
   - ValueError for invalid formats with clear error message

### Key Code Changes

```python
# render_field_to_png_bytes_with_overlays (lines 127-200)
image_format = image_format.lower()
if image_format not in ("png", "jpeg"):
    raise ValueError(f"image_format must be 'png' or 'jpeg', got '{image_format}'")

if image_format == "jpeg":
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[:, :, :3]  # Drop alpha channel
    from PIL import Image
    img = Image.fromarray(rgb)
    img.save(buf, format="JPEG", quality=85, optimize=True)
else:
    fig.savefig(buf, format="png")
```

### New Tests

Created `tests/animation/test_widget_image_format.py` with 14 tests:

| Test Class | Count | Coverage |
|------------|-------|----------|
| TestRenderFieldWithOverlaysImageFormat | 7 | PNG/JPEG output, validation, case-insensitivity |
| TestPersistentFigureRendererImageFormat | 4 | PNG/JPEG in persistent renderer, invalid format |
| TestRenderWidgetImageFormat | 2 | render_widget accepts and passes image_format |
| TestJPEGRequirements | 1 | Pillow availability verification |

### Code Review Fixes

1. Fixed mypy `type: ignore` placement (single-line format)
2. Simplified redundant format normalization in Image widget creation
3. Added test for PersistentFigureRenderer invalid format validation

### Benchmark Results

Tested with 385 bins environment and position overlay (trail_length=5):

| DPI | PNG Size | JPEG Size | PNG Time | JPEG Time |
|-----|----------|-----------|----------|-----------|
| 100 | 15.4 KB | 40.4 KB | 11.25 ms | 12.85 ms |
| 150 | 24.0 KB | 62.3 KB | 18.19 ms | 22.12 ms |
| 200 | 32.5 KB | 76.7 KB | 27.71 ms | 34.91 ms |

**Key Finding**: PNG outperforms JPEG in both size AND speed for scientific visualization!

**Why?**
- Scientific plots have uniform color regions and sharp edges
- PNG excels at run-length encoding for uniform regions
- JPEG introduces block artifacts and doesn't compress well for this content type

**Recommendation**: Keep PNG as default. JPEG option available for users who need it.

### Test Results

- All 14 new JPEG tests pass
- All 615 animation tests pass
- ruff and mypy pass

## Completed: Phase 4.4 - Re-profile Widget (2025-11-22)

### Performance Improvements (Post Phase 4.1-4.3)

The `QuadMesh.set_array()` optimization from Phase 4.1 delivers significant speedups:

**Comparison vs Baseline:**

| Config | Metric | Baseline | Current | Speedup |
|--------|--------|----------|---------|---------|
| small | Avg render | 8.93 ms | 4.20 ms | **2.1x** |
| small | Avg scrub | 8.92 ms | 4.12 ms | **2.2x** |
| medium | Avg render | 10.00 ms | 5.53 ms | **1.8x** |
| medium | Avg scrub | 9.81 ms | 5.49 ms | **1.8x** |
| large | Avg render | 9.95 ms | 5.67 ms | **1.8x** |
| large | Avg scrub | 10.02 ms | 5.71 ms | **1.8x** |

### PNG vs JPEG Comparison

| Format | Avg Time | Size | Conclusion |
|--------|----------|------|------------|
| PNG | 6.79 ms | 7.8 KB | **1.5x faster, 4.1x smaller** |
| JPEG | 10.48 ms | 32.2 KB | Larger files, slower for sci viz |

**Recommendation**: Use PNG (default) for scientific visualization.

### Scrubbing Responsiveness for Large Frame Counts

| Frame Count | Avg Scrub | P50 | P95 | P99 |
|-------------|-----------|-----|-----|-----|
| 10,000 | 5.71 ms | 5.68 ms | 5.93 ms | 6.07 ms |
| 50,000 | 5.71 ms | 5.69 ms | 5.97 ms | 6.06 ms |

**Key Finding**: Scrubbing performance is **O(1)** - constant regardless of frame count!
This confirms the `set_array()` optimization is working correctly.

### Summary

Phase 4 widget improvements deliver:

1. **1.8-2.2x faster rendering** via `QuadMesh.set_array()` optimization
2. **Fixed skeleton initialization bug** (IndexError in `_initialize_bodypart_overlay`)
3. **Optional JPEG support** (PNG recommended for scientific viz)
4. **Constant-time scrubbing** regardless of frame count

## Phase 4 Complete!

All Phase 4 (Widget Backend Performance) tasks are now complete:
- 4.1 Fixed `set_array` optimization ✅
- 4.2 Fixed overlay artist lifecycle bug ✅
- 4.3 Optional JPEG support ✅
- 4.4 Re-profiled widget backend ✅

## Completed: Phase 5.1 - Sanitize Frame Naming Pattern (2025-11-22)

### Investigation Result

**Finding**: The current implementation already uses zero-padded filenames correctly!

### Implementation Details

1. **Digit calculation** (in `parallel_render_frames`):
   ```python
   digits = max(5, len(str(max(0, n_frames - 1))))
   ```
   - Minimum 5 digits (supports up to 99,999 frames)
   - Dynamically expands for larger frame counts

2. **Pattern format**: `frame_%0{digits}d.png`
   - Example: `frame_%05d.png` for <100k frames
   - Example: `frame_%06d.png` for ≥100k frames

3. **Consistency verified**:
   - Pattern passed to workers via `task["digits"]`
   - Workers use same digit count when saving: `f"frame_{frame_number:0{digits}d}.png"`
   - ffmpeg pattern matches saved files exactly

### New Tests

Created `tests/animation/test_frame_naming.py` with 16 tests:

| Test Class | Count | Coverage |
|------------|-------|----------|
| TestFrameNamingPattern | 4 | Zero-padding, minimum digits, file matching, multi-worker |
| TestFrameNamingDigitCalculation | 5 | Small/medium/large/very large/extreme frame counts |
| TestFrameNamingIntegration | 3 | ffmpeg pattern, non-zero start, digits propagation |
| TestFrameNamingEdgeCases | 4 | Single frame, boundary cases (99999, 100000, 100001) |

### Test Results

- All 16 tests pass
- ruff and mypy pass with no issues

## Completed: Phase 5.2 - Control ffmpeg I/O (2025-11-22)

### Problem

The original `capture_output=True` routes both stdout and stderr to PIPE buffers. For long ffmpeg runs with verbose output, the stdout buffer can fill up, potentially causing deadlock.

### Solution

Changed from `capture_output=True` to explicit I/O routing:
- `stdout=subprocess.DEVNULL` - discard ffmpeg progress output
- `stderr=subprocess.PIPE` - capture errors for reporting

### New Tests

Created `tests/animation/test_ffmpeg_io.py` with 6 tests:

| Test | Coverage |
|------|----------|
| `test_ffmpeg_stdout_is_devnull` | stdout routing |
| `test_ffmpeg_stderr_is_captured` | stderr routing |
| `test_ffmpeg_error_includes_stderr_message` | Error messages include stderr |
| `test_ffmpeg_does_not_use_capture_output` | Explicit I/O not capture_output |
| `test_ffmpeg_called_with_correct_arguments` | Basic ffmpeg args |
| `test_ffmpeg_uses_text_mode` | text=True for string stderr |

### Test Results

- All 6 tests pass
- ruff and mypy pass with no issues

## Completed: Phase 5.3 - DPI and Size Guard (2025-11-22)

### Problem

High DPI values can cause:
- Very large video files
- Slow rendering times
- High memory usage

Users should be warned before this happens.

### Solution

Added `UserWarning` when `dpi > 150`:
- Shows estimated resolution (e.g., "1600x1200 pixel frames")
- Suggests using `dpi=100` or `dpi=150`
- Warning only (doesn't restrict users who need high DPI)

### New Tests

Created `tests/animation/test_dpi_guard.py` with 11 tests:

| Test Class | Count | Coverage |
|------------|-------|----------|
| TestDPIWarning | 5 | Warning behavior for various DPI values |
| TestDryRunEstimates | 5 | Dry-run output verification |
| TestDPIEstimatedResolution | 1 | Resolution scaling |

### Test Results

- All 11 tests pass
- ruff and mypy pass with no issues

## Completed: Phase 5.4 - Re-profile Video Export (2025-11-22)

### Benchmark Results

**Video Backend Performance** (measured via `bench_video.py`):

| Config | Frames | Serial (ms) | ms/frame | Parallel (ms) | Speedup |
|--------|--------|-------------|----------|---------------|---------|
| small | 100 | 2,988 | 29.88 | 2,681 | 1.11x |
| medium | 500 | 7,860 | 15.72 | 4,125 | 1.91x |
| large | 500 | 8,041 | 16.08 | 4,084 | 1.97x |

**Comparison vs Baseline:**

| Config | Baseline Serial | Current Serial | Change |
|--------|-----------------|----------------|--------|
| small | 2,995 ms | 2,988 ms | -0.2% |
| medium | 7,885 ms | 7,860 ms | -0.3% |
| large | 8,444 ms | 8,041 ms | -4.8% |

| Config | Baseline Parallel | Current Parallel | Change |
|--------|-------------------|------------------|--------|
| small | 2,757 ms | 2,681 ms | -2.8% |
| medium | 3,967 ms | 4,125 ms | +4.0% |
| large | 4,010 ms | 4,084 ms | +1.8% |

**File Sizes:**

| Config | File Size |
|--------|-----------|
| small (100 frames) | 0.18 MB |
| medium (500 frames) | 0.09 MB |
| large (500 frames) | 0.09 MB |

### Analysis

1. **Performance unchanged**: Phase 5.1-5.3 were robustness improvements (frame naming verification, ffmpeg I/O control, DPI warning), not performance optimizations. Results are within measurement noise.

2. **Time per frame**: ~15-30ms depending on grid size and overlay complexity
   - small (40x40 grid with overlays): 29.88 ms/frame
   - medium/large (100x100 grid, no overlays): 15.72-16.08 ms/frame

3. **Parallel speedup**: ~1.9-2.0x for 500 frames with 4 workers
   - Small frame counts (<200) have limited benefit due to worker startup overhead

4. **Memory usage**: Dominated by field data
   - small: ~230 MB
   - medium: ~250 MB
   - large: ~660 MB (100x100x100k fields in memory)

### Phase 5 Summary

All Phase 5 (Video Backend Robustness) tasks are now complete:

| Task | Status | Notes |
|------|--------|-------|
| 5.1 Frame naming | Complete | Already correct, added 16 tests |
| 5.2 ffmpeg I/O | Complete | stdout→DEVNULL, stderr→PIPE, 6 tests |
| 5.3 DPI guard | Complete | Warning for dpi>150, 11 tests |
| 5.4 Re-profile | Complete | No performance regression |

## Next Task: Phase 6.1 - Normalize Edges in Skeleton

See TASKS.md for details.
