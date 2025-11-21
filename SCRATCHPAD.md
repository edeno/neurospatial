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

## Next Task: Milestone 2.2 - Align with existing bodypart Points layer

Verify that Points layer creation remains unchanged and coordinate transforms are not duplicated.
