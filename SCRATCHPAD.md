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

1. **Created `benchmark_datasets/` package** (named to avoid conflict with `tests/benchmarks`):
   - `benchmark_datasets/__init__.py` - Package exports
   - `benchmark_datasets/datasets.py` - Dataset generators

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

### Implementation Notes

- Uses `np.random.default_rng(seed)` for reproducibility
- Supports memory-mapped arrays for large datasets
- Smooth trajectory generation with boundary reflection
- Head direction angles wrapped to [-pi, pi]

### Key Files

- `benchmark_datasets/__init__.py` - Package init
- `benchmark_datasets/datasets.py` - All generators
- `tests/animation/test_benchmark_datasets.py` - 22 tests

## Next Task: Phase 0.3 - Record Baseline Metrics

- Napari: initialization time, random seek time
- Widget: initialization time, scrubbing responsiveness
- Video: time per frame, total export time
- Peak memory usage for each benchmark
- Document results in `benchmarks/BASELINE.md`
