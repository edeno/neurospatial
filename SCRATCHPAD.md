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

## Next Task: Phase 2.3 - Tracks Color Handling Cleanup

Use `features` + `color_by="color"` at layer creation instead of post-creation workaround.
