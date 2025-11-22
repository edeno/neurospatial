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

### 2.3 Tracks Color Handling Cleanup

- [ ] Use `features` + `color_by="color"` at layer creation (verify it is a keyword argument)
- [ ] Remove post-creation `layer.color_by = "color"` workaround
- [ ] Verify no warnings on layer init
- [ ] Verify trails still uniformly colored

### 2.4 Playback Widget Throttling Fix

- [ ] In `_add_speed_control_widget`, fix `update_frame_info`
- [ ] Always update when `playback_state["is_playing"]` is `False`
- [ ] Test smooth scrubbing at high FPS
- [ ] Test no UI stalls on large datasets

### 2.5 Re-profile Napari

- [ ] Measure skeleton overlay initialization time
- [ ] Measure playback smoothness (perfmon timestamps)
- [ ] Document improvements vs baseline

---

## Phase 3: Overlay Conversion & Core Orchestration

### 3.1 Vectorize Interpolation in `overlays.py`

- [ ] Replace Python loops with `np.interp` for 1D series
- [ ] Vectorize 2D arrays across frames per dimension
- [ ] Update position overlay interpolation
- [ ] Update bodypart overlay interpolation
- [ ] Update head direction overlay interpolation

### 3.2 Optimize Validation Functions

- [ ] `_validate_bounds`: use global min/max instead of per-point loops
- [ ] Keep WHAT/WHY/HOW messaging (excellent UX)
- [ ] Profile validation on large datasets

### 3.3 Add Overlay Conversion Caching

- [ ] In `_convert_overlays_to_data`, add cache keyed by:
  - Overlay id
  - Frame times signature
  - Env dimension_ranges/layout hash
- [ ] Test cache hit on repeated `animate_fields` calls

### 3.4 Harden Multi-Field Detection

- [ ] Replace `fields[0]` single-element check
- [ ] Implement: "all top-level elements are Sequence-like and not ndarray"
- [ ] Add tests: single-field list of arrays
- [ ] Add tests: multi-field list of lists
- [ ] Add tests: mismatched shapes raise correct errors

### 3.5 Re-profile Conversion Time

- [ ] Measure overlay → `OverlayData` conversion time
- [ ] Compare with baseline

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
| Phase 2: Napari | In Progress | 2.1 COMPLETE (42-47x speedup!), 2.2 COMPLETE, 2.3-2.5 pending |
| Phase 3: Overlays | Not started | |
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
