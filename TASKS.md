# TASKS.md - Animation Performance Implementation Checklist

**Goal**: Optimize animation backends (napari, video, widget) for performance and correctness.

**Approach**: Measure first, optimize second. See PLAN.md for detailed rationale and REVIEW.md for issue analysis.

---

## Phase 0: Profiling & Baseline

### 0.1 Add Timing Instrumentation

- [x] Napari backend: wrap `_build_skeleton_vectors` with timing (via perfmon_config.json)
- [x] Napari backend: wrap `_render_bodypart_overlay`, `_render_head_direction_overlay` (via perfmon_config.json)
- [x] Video backend: wrap `render_field_to_rgb` and parallel worker function (via _timing module)
- [x] Widget backend: wrap `PersistentFigureRenderer.render` and `render_field_to_png_bytes_with_overlays` (via _timing module)

### 0.2 Create Benchmark Datasets

- [x] Create `scripts/benchmark_datasets/` directory (moved from root to scripts/)
- [x] Small benchmark: 100 frames, 40x40 grid (SMALL_CONFIG)
- [x] Medium benchmark: 5k frames, typical spatial grid (MEDIUM_CONFIG)
- [x] Large benchmark: 100k frames with skeleton + head direction overlays (LARGE_CONFIG)
- [x] Save reproducible scripts/notebooks (scripts/benchmark_datasets/datasets.py)

### 0.3 Record Baseline Metrics

- [x] Napari: initialization time, random seek time
- [x] Widget: initialization time, scrubbing responsiveness
- [x] Video: time per frame, total export time
- [x] Peak memory usage for each benchmark
- [x] Document results in `benchmarks/BASELINE.md`

---

## Phase 1: Shared Infrastructure Cleanup

### 1.1 Centralize Coordinate Transforms

- [x] Create `neurospatial/animation/transforms.py` (or add to `rendering.py`)
- [x] Move `_transform_coords_for_napari` to shared module
- [x] Move `_transform_direction_for_napari` to shared module
- [x] Update napari backend to use shared functions
- [x] Add tests for coordinate transforms (19 tests)

### 1.2 Normalize Layout Metadata

- [x] Add `layout_type` property to layouts: `"grid" | "mask" | "polygon" | "hexagonal" | "mesh" | "graph" | "other"`
- [x] Add `is_grid_compatible` property to layouts (True for grid/mask/polygon, False for others)
- [x] Update widget `_field_to_image_data` to use `is_grid_compatible`
- [x] Update `field_to_rgb_for_napari` to use `is_grid_compatible`
- [x] Add tests for layout_type and is_grid_compatible (32 tests in test_layout_type.py)

### 1.3 Verify No Regressions

- [x] Re-run benchmarks after changes (no regressions detected)
- [ ] Visual verification of alignment (napari + widget) - requires interactive testing

---

## Phase 2: Napari Backend Performance

### 2.1 Vectorize `_build_skeleton_vectors` (HIGH IMPACT) - COMPLETE

- [x] Profile current implementation on medium/large dataset
- [x] Replace nested Python loops with boolean masks
- [x] Stack all frames into arrays for bulk processing
- [x] Build vectors in bulk rather than frame-by-frame
- [x] Profile after: target 5-20x speedup (achieved **42-47x** speedup!)
- [x] Add unit tests comparing output to original implementation (15 existing tests all pass)

**Performance Results:**

| Config | Before (ms) | After (ms) | Speedup |
|--------|-------------|------------|---------|
| Medium (5k frames, 4 edges) | 98.75 | 2.11 | **46.8x** |
| Large (100k frames, 6 edges) | 2628.52 | 62.35 | **42.2x** |

### 2.2 Fix Transform Fallback Warning State - COMPLETE

- [x] Replace `_TRANSFORM_FALLBACK_WARNED` global behavior with per-viewer tracking
- [x] Add `suppress_warning` parameter to `transform_coords_for_napari` and `transform_direction_for_napari`
- [x] Add helper functions `_check_viewer_warned`, `_mark_viewer_warned` in napari backend
- [x] Add wrapper functions `_transform_coords_with_viewer`, `_transform_direction_with_viewer`
- [x] Test: one warning per viewer/env combination (15 new tests)
- [x] Test: multiple envs warn once each

### 2.3 Tracks Color Handling Cleanup - VERIFIED CURRENT APPROACH CORRECT

**Investigation Result**: napari 0.5.6 still exhibits the issue where passing `color_by` at
layer creation time triggers a warning because napari's internal data setter resets features
to `{}` before our features are applied. The current workaround (setting `color_by` AFTER
creation) is the correct approach.

- [x] Verified `color_by` is a keyword argument in napari 0.5.6 Tracks layer
- [x] Tested: passing `color_by` at init triggers warning "color_by key 'color' not present in features"
- [x] Verified: post-creation `layer.color_by = "color"` avoids warning
- [x] Added test `test_position_overlay_trail_color_by_workaround` to document/lock behavior
- [N/A] Remove post-creation workaround - NOT POSSIBLE (napari bug still present)

### 2.4 Playback Widget Throttling Fix - COMPLETE

- [x] In `_add_speed_control_widget`, fix `update_frame_info`
- [x] Always update when `playback_state["is_playing"]` is `False`
- [x] Added test `test_playback_widget_scrubbing_updates_immediately`
- [ ] Test smooth scrubbing at high FPS - requires interactive testing
- [ ] Test no UI stalls on large datasets - requires interactive testing

### 2.5 Re-profile Napari - COMPLETE

**Skeleton Vector Generation** (measured via `bench_skeleton_vectors.py`):

| Config | Baseline (ms) | Current (ms) | Speedup |
|--------|---------------|--------------|---------|
| Medium (5k frames) | 98.75 | 1.99 | **49.6x** |
| Large (100k frames) | 2628.52 | 61.82 | **42.5x** |

**Napari Viewer** (measured via `bench_napari.py`):

| Config | Init (baseline) | Init (current) | Seek (baseline) | Seek (current) |
|--------|-----------------|----------------|-----------------|----------------|
| Small | 2,585 ms | 2,607 ms | 4.38 ms | 4.38 ms |
| Medium | 2,974 ms | 2,978 ms | 15.80 ms | 15.81 ms |

**Key findings**:

- [x] Skeleton vector generation 42-50x faster
- [x] Viewer init: unchanged (skeleton is small part of total init)
- [x] Random seek: unchanged (seek wasn't target; playback smoothness was)
- [N/A] Playback smoothness: requires interactive testing with perfmon

**Note**: The main benefit of Phase 2.1-2.4 is eliminating the per-frame Shapes layer
callback that blocked Qt event loop (5.38ms/frame → 0ms via native Vectors layer time slicing).

---

## Phase 3: Overlay Conversion & Core Orchestration

### 3.1 Vectorize Interpolation in `overlays.py` - ALREADY COMPLETE

**Verified 2025-11-21**: Interpolation was already vectorized in the existing implementation.

- [x] Replace Python loops with `np.interp` for 1D series (line 1144 uses `np.interp`)
- [x] Vectorize 2D arrays across frames per dimension (line 1147-1148 loops over dims, not frames)
- [x] Update position overlay interpolation (uses `_interp_linear`/`_interp_nearest`)
- [x] Update bodypart overlay interpolation (uses `_interp_linear`/`_interp_nearest`)
- [x] Update head direction overlay interpolation (uses `_interp_linear`/`_interp_nearest`)

**Performance Results (100k frames)**:

| Function | Time | Notes |
|----------|------|-------|
| `_interp_linear` 1D | 0.89 ms | Fully vectorized |
| `_interp_linear` 2D | 2.10 ms | Loop over 2 dims (fast) |
| `_interp_nearest` 2D | 54.29 ms | Uses distance matrix |
| Full conversion (3 overlays) | 12.47 ms | All overlays |

### 3.2 Optimize Validation Functions - ALREADY COMPLETE

**Verified 2025-11-21**: Validation functions were already vectorized.

- [x] `_validate_bounds`: uses vectorized `np.sum` and boolean operations (loops over dims, not points)
- [x] Keep WHAT/WHY/HOW messaging (excellent UX) - already present
- [x] Profile validation on large datasets

**Performance Results (100k points)**:

| Function | Time |
|----------|------|
| `_validate_finite_values` | 0.114 ms |
| `_validate_shape` | 0.002 ms |
| `_validate_bounds` | 0.232 ms |
| `_validate_monotonic_time` | 0.142 ms |

### 3.3 Add Overlay Conversion Caching - SKIPPED (Premature Optimization)

**Decision 2025-11-21**: Skip caching due to fast conversion times.

- [N/A] In `_convert_overlays_to_data`, add cache keyed by overlay id, frame times, env hash
- [N/A] Test cache hit on repeated `animate_fields` calls

**Rationale**: Conversion takes only 12ms for 100k frames with 3 overlays. This is 200x
faster than napari viewer init (2,600ms). The complexity of cache key computation and
memory management would outweigh the 12ms benefit. If profiling shows conversion becomes
a bottleneck in the future, caching can be added then.

### 3.4 Harden Multi-Field Detection - COMPLETE

**Completed 2025-11-21**: Added validation for consistent field types.

- [x] Replace `fields[0]` single-element check - added `_validate_field_types_consistent()`
- [x] Implement: "all top-level elements are Sequence-like and not ndarray" - validates all elements
- [x] Add tests: single-field list of arrays - `TestMultiFieldDetectionEdgeCases`
- [x] Add tests: multi-field list of lists - `TestMultiFieldDetectionEdgeCases`
- [x] Add tests: mismatched shapes raise correct errors - `TestMismatchedShapeErrors`

**New tests in `tests/animation/test_multi_field_detection.py`**:

- 6 edge case tests for detection logic
- 2 robustness tests for mixed types (key improvement)
- 3 shape mismatch error tests
- 3 ndarray vs sequence distinction tests

**Implementation**:

- Added `_validate_field_types_consistent()` function with WHAT/WHY/HOW error messages
- Called before `_is_multi_field_input()` in `render_napari()` to catch mixed types early
- All 31 multi-field tests pass (14 new + 17 existing)

### 3.5 Re-profile Conversion Time - COMPLETE

**Completed 2025-11-21**: Profiling data collected during Phase 3.1/3.2 verification.

- [x] Measure overlay → `OverlayData` conversion time
- [x] Compare with baseline

**Results (100k frames, 3 overlays)**:

| Operation | Time |
|-----------|------|
| Full conversion pipeline | 12.47 ms |
| `_interp_linear` 1D | 0.89 ms |
| `_interp_linear` 2D | 2.10 ms |
| `_interp_nearest` 2D | 54.29 ms |

**Conclusion**: Conversion is fast (12ms for 100k frames). No optimization needed.

---

## Phase 4: Widget Backend Performance

### 4.1 Clarify Fallback in `PersistentFigureRenderer`

- [ ] In `render()`: log when `_field_to_image_data` returns `None`
- [ ] Consider debug flag to raise instead of silent fallback
- [ ] Document performance implications

### 4.2 Stabilize Overlay Artist Lifecycle

- [ ] Ensure `OverlayArtistManager` created once per figure
- [ ] Use `set_data`, `set_offsets` instead of recreate-on-each-frame
- [ ] In fallback path: reinitialize overlay manager exactly once
- [ ] Profile redraw overhead

### 4.3 Optional JPEG Support

- [ ] Add config flag: `image_format="png" | "jpeg"`
- [ ] Use `render_field_to_image_bytes` internally
- [ ] Benchmark PNG vs JPEG: file size
- [ ] Benchmark PNG vs JPEG: time per render

### 4.4 Re-profile Widget

- [ ] First render time
- [ ] Cache miss render time (PNG/JPEG comparison)
- [ ] Scrubbing responsiveness for 10k and 50k frames

---

## Phase 5: Video Backend Robustness

### 5.1 Sanitize Frame Naming Pattern

- [ ] Ensure `frame_pattern` uses zero-padded filenames: `frame_%06d.png`
- [ ] Verify `parallel_render_frames` uses same pattern

### 5.2 Control ffmpeg I/O

- [ ] Route `stdout=subprocess.DEVNULL`
- [ ] Route `stderr=subprocess.STDOUT`
- [ ] Test with large frame counts

### 5.3 DPI and Size Guard

- [ ] Add warning when `dpi > 150`
- [ ] Show estimated resolution in warning
- [ ] Use dry-run code to show estimated size/time
- [ ] Consider clamping to reasonable upper bound

### 5.4 Re-profile Video Export

- [ ] Time per frame in worker function
- [ ] Total runtime for typical dataset
- [ ] Total runtime for large dataset
- [ ] File sizes comparison

---

## Phase 6: Skeleton Module Enhancements

### 6.1 Normalize Edges in `Skeleton`

- [ ] In `__post_init__` or factory: canonicalize edges to sorted `(min(node), max(node))`
- [ ] Optionally deduplicate edges
- [ ] Test: reversed duplicates handled gracefully

### 6.2 Precompute Adjacency

- [ ] Add cached property: `adjacency: dict[str, list[str]]`
- [ ] Useful for graph traversal and topology-based styling
- [ ] Add tests for adjacency computation

---

## Phase 7: Tests, Docs, and Examples

### 7.1 Unit Tests

- [ ] Napari: `_build_skeleton_vectors` shape/content for synthetic skeleton
- [ ] Overlays: temporal alignment correctness (known angle trajectory)
- [ ] Widget: `_field_to_image_data` returns arrays of expected shape
- [ ] Video: `compute_global_colormap_range` degeneracy tests
- [ ] Video: ffmpeg call constructed with expected arguments

### 7.2 Benchmark Scripts

- [x] `benchmarks/bench_napari.py` - CLI tool with timing table
- [x] `benchmarks/bench_video.py` - CLI tool with timing table
- [x] `benchmarks/bench_widget.py` - CLI tool with timing table
- [x] Each prints compact before/after comparison

### 7.3 Documentation Updates

- [ ] `animate_fields` docstring: overlays, backends, large-dataset recommendations
- [ ] Overlay dataclasses: performance considerations (NaN cleaning, expected shapes)
- [ ] Backend descriptions: auto-selection criteria, strengths per use case
- [ ] Update CLAUDE.md with any new patterns

---

## Progress Summary

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0: Profiling | Complete | Timing instrumentation, datasets, baseline metrics all done |
| Phase 1: Infrastructure | Complete | 1.1-1.3 done (visual verification pending interactive test) |
| Phase 2: Napari | Complete | 2.1-2.5 COMPLETE (42-50x skeleton speedup!) |
| Phase 3: Overlays | Complete | 3.1-3.2 already vectorized, 3.3 skipped, 3.4-3.5 done |
| Phase 4: Widget | Not started | |
| Phase 5: Video | Not started | |
| Phase 6: Skeleton | Not started | |
| Phase 7: Tests/Docs | In Progress | 7.2 benchmark scripts done |

---

## Quick Reference: High-Impact Fixes

From REVIEW.md, prioritize these for maximum benefit:

| Fix | Expected Benefit |
|-----|------------------|
| Vectorize skeleton vector generation | 5-20x faster loading |
| Centralize duplicated transforms | Reduces maintenance & bugs |
| Add napari per-viewer warning state | Robust in multi-viewer workflows |
| Improve widget fallback behavior | Smoother notebook performance |
| Shared artist system for overlays | Reduces redraw overhead |
