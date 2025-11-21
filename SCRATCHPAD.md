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

## Next Task: Milestone 2 - Replace Shapes Skeleton With Vectors Layer

The next task is to update `_render_bodypart_overlay` to use the new vectors layer.
