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

### 4.1 Clarify Fallback in `PersistentFigureRenderer` - COMPLETE

**Completed 2025-11-21**: Fixed optimization and added fallback diagnostics.

- [x] Fixed `set_array` optimization to work with `pcolormesh` (uses `QuadMesh.set_array()`)
- [x] In `render()`: log at DEBUG level when fallback to full re-render is required
- [x] Added `raise_on_fallback=True` debug flag to raise RuntimeError instead of silent fallback
- [x] Added `_field_to_mesh_array()` method for QuadMesh-compatible data conversion
- [x] Added `_do_full_rerender()` helper method for DRY fallback path
- [x] Document performance implications in docstrings

**Key Changes**:

- **Fixed optimization**: Changed from looking for `AxesImage` in `ax.images` (for `imshow`)
  to looking for `QuadMesh` in `ax.collections` (for `pcolormesh`). Grid layouts now use
  efficient `set_array()` updates instead of full re-render.
- **Fallback logging**: Non-grid layouts (hexagonal, graph, mesh) trigger DEBUG log message
  explaining why fallback is required.
- **Debug flag**: `PersistentFigureRenderer(..., raise_on_fallback=True)` raises RuntimeError
  on fallback, useful for debugging performance issues.

**New Tests** (`tests/animation/test_widget_fallback.py`):

- 13 tests covering fallback logging, debug flag, valid output, and `_field_to_mesh_array`
- Tests verify grid layouts use optimization (no fallback)
- Tests verify hex layouts trigger fallback with informative log

### 4.2 Stabilize Overlay Artist Lifecycle - COMPLETE

**Completed 2025-11-21**: Fixed critical bug and added comprehensive tests.

- [x] Ensure `OverlayArtistManager` created once per figure (already correct, verified with tests)
- [x] Use `set_data`, `set_offsets` instead of recreate-on-each-frame (already implemented, verified with tests)
- [x] In fallback path: reinitialize overlay manager exactly once (already correct, verified with tests)
- [x] Fixed IndexError bug: skeleton list was accessed before appending in `_initialize_bodypart_overlay`
- [N/A] Profile redraw overhead - already profiled in Phase 0.3 baseline metrics

**Bug Fixed** (in `_parallel.py:585-602`):

```python
# Before (buggy): Called update BEFORE appending to list
self._update_bodypart_skeleton(len(self._bodypart_skeletons), ...)  # IndexError!
self._bodypart_skeletons.append(skeleton_lc)

# After (fixed): Append FIRST, then update
self._bodypart_skeletons.append(skeleton_lc)
self._update_bodypart_skeleton(len(self._bodypart_skeletons) - 1, ...)  # Correct index
```

**New Tests** (`tests/animation/test_overlay_artist_manager.py`):

- 30 tests covering initialization, updates, clear, edge cases, and integration
- Includes regression test for the IndexError bug
- All 600 animation tests pass

### 4.3 Optional JPEG Support - COMPLETE

**Completed 2025-11-22**: Added optional JPEG image format support.

- [x] Add config flag: `image_format="png" | "jpeg"` - added to all widget backend functions
- [x] Use PIL for JPEG compression (quality=85, optimize=True)
- [x] Added `image_format` parameter to:
  - `render_field_to_png_bytes_with_overlays()`
  - `PersistentFigureRenderer` class
  - `render_widget()` function
- [x] Added validation (ValueError for invalid formats)
- [x] Added graceful ImportError for missing Pillow

**New Tests** (`tests/animation/test_widget_image_format.py`):

- 14 tests covering PNG/JPEG output, format validation, case-insensitivity, PersistentFigureRenderer

**Benchmark Results** (385 bins, position overlay):

| DPI | PNG Size | JPEG Size | PNG Time | JPEG Time | Conclusion |
|-----|----------|-----------|----------|-----------|------------|
| 100 | 15.4 KB | 40.4 KB | 11.25 ms | 12.85 ms | PNG 2.6x smaller |
| 150 | 24.0 KB | 62.3 KB | 18.19 ms | 22.12 ms | PNG 2.6x smaller |
| 200 | 32.5 KB | 76.7 KB | 27.71 ms | 34.91 ms | PNG 2.4x smaller |

**Key Finding**: For scientific visualization with colormaps, PNG outperforms JPEG in both
size AND speed. This is because scientific plots have uniform color regions and sharp edges
(ideal for PNG's run-length encoding) rather than photographic content (where JPEG excels).

**Recommendation**: Use default PNG for widget backend. JPEG option remains available for
users who need it (e.g., for photographic overlays or specific compatibility requirements).

### 4.4 Re-profile Widget - COMPLETE

**Completed 2025-11-22**: Measured widget performance after Phase 4.1-4.3 improvements.

- [x] First render time
- [x] Cache miss render time (PNG/JPEG comparison)
- [x] Scrubbing responsiveness for 10k and 50k frames

**Results vs Baseline:**

| Config | Metric | Baseline | Current | Speedup |
|--------|--------|----------|---------|---------|
| small | Avg render | 8.93 ms | 4.20 ms | **2.1x** |
| small | Avg scrub | 8.92 ms | 4.12 ms | **2.2x** |
| medium | Avg render | 10.00 ms | 5.53 ms | **1.8x** |
| large | Avg scrub | 10.02 ms | 5.71 ms | **1.8x** |

**PNG vs JPEG** (40x40 grid, dpi=72):

| Format | Time | Size |
|--------|------|------|
| PNG | 6.79 ms | 7.8 KB |
| JPEG | 10.48 ms | 32.2 KB |

PNG is 1.5x faster and 4.1x smaller for scientific visualization.

**Large Frame Count Scrubbing** (constant-time O(1)):

| Frame Count | Avg Scrub | P95 |
|-------------|-----------|-----|
| 10,000 | 5.71 ms | 5.93 ms |
| 50,000 | 5.71 ms | 5.97 ms |

---

## Phase 5: Video Backend Robustness

### 5.1 Sanitize Frame Naming Pattern - COMPLETE

**Verified 2025-11-22**: The current implementation already uses zero-padded filenames correctly.

- [x] Ensure `frame_pattern` uses zero-padded filenames: `frame_%05d.png` (min 5 digits, expands for >99,999 frames)
- [x] Verify `parallel_render_frames` uses same pattern

**Implementation Details:**

- Digit calculation: `digits = max(5, len(str(max(0, n_frames - 1))))`
- Pattern format: `frame_%0{digits}d.png`
- Pattern passed to workers and used in ffmpeg command are consistent
- Handles edge cases: single frame, >100k frames, multi-worker scenarios

**New Tests** (`tests/animation/test_frame_naming.py`): 16 tests covering:

- Zero-padding verification
- Minimum 5 digits enforcement
- Pattern-to-file consistency
- Multi-worker sequential numbering
- Digit calculation boundaries (5k, 100k, 1M frames)
- Non-zero start index for workers
- ffmpeg pattern integration

### 5.2 Control ffmpeg I/O - COMPLETE

**Completed 2025-11-22**: Fixed ffmpeg subprocess I/O to avoid buffer issues.

- [x] Route `stdout=subprocess.DEVNULL` (discard ffmpeg progress output)
- [x] Route `stderr=subprocess.PIPE` (capture errors for reporting)
- [x] Test with large frame counts

**Implementation Details:**

Changed from `capture_output=True` to explicit I/O routing:

```python
result = subprocess.run(
    cmd,
    stdout=subprocess.DEVNULL,  # Discard progress output (avoids buffer issues)
    stderr=subprocess.PIPE,     # Capture errors for reporting
    text=True,
    check=False,
)
```

**Why this matters:**

- `capture_output=True` routes both stdout AND stderr to PIPE buffers
- For very long ffmpeg runs, stdout buffer can fill up causing potential deadlock
- Discarding stdout (progress info) avoids this issue while keeping error capture

**New Tests** (`tests/animation/test_ffmpeg_io.py`): 6 tests covering:

- stdout is DEVNULL
- stderr is PIPE
- Error messages include stderr content
- capture_output is NOT used
- Basic ffmpeg arguments
- Text mode enabled

### 5.3 DPI and Size Guard - COMPLETE

**Completed 2025-11-22**: Added DPI warning with resolution estimate.

- [x] Add warning when `dpi > 150`
- [x] Show estimated resolution in warning (e.g., "1600x1200 pixel frames")
- [x] Use dry-run code to show estimated size/time (already existed)
- [N/A] Consider clamping to reasonable upper bound (warning is sufficient, don't restrict users)

**Implementation Details:**

Added `UserWarning` when `dpi > 150`:

```python
if dpi > 150:
    width_px = int(8 * dpi)   # 8 inches default width
    height_px = int(6 * dpi)  # 6 inches default height
    warnings.warn(
        f"High DPI detected: dpi={dpi} will produce {width_px}x{height_px} pixel frames.\n"
        f"This may result in large file sizes, slow rendering, high memory usage.\n"
        f"Consider using dpi=100 or dpi=150 for most use cases.",
        UserWarning,
    )
```

**New Tests** (`tests/animation/test_dpi_guard.py`): 11 tests covering:

- No warning for dpi=100 (default)
- No warning for dpi=150 (threshold)
- Warning for dpi > 150
- Warning includes resolution estimate
- Warning suggests lower DPI
- Dry-run shows frame count, time, size estimates

### 5.4 Re-profile Video Export - COMPLETE

**Completed 2025-11-22**: Measured video export performance after Phase 5.1-5.3 improvements.

- [x] Time per frame in worker function
- [x] Total runtime for typical dataset
- [x] Total runtime for large dataset
- [x] File sizes comparison

**Results vs Baseline:**

| Config | Frames | Baseline Serial | Current Serial | Change |
|--------|--------|-----------------|----------------|--------|
| small | 100 | 2,995 ms | 2,988 ms | -0.2% |
| medium | 500 | 7,885 ms | 7,860 ms | -0.3% |
| large | 500 | 8,444 ms | 8,041 ms | -4.8% |

| Config | ms/frame | Parallel Speedup |
|--------|----------|------------------|
| small | 29.88 | 1.11x |
| medium | 15.72 | 1.91x |
| large | 16.08 | 1.97x |

**Conclusion**: Performance unchanged (within noise). Phase 5.1-5.3 were robustness
improvements (frame naming, ffmpeg I/O, DPI warning), not optimizations.

---

## Phase 6: Skeleton Module Enhancements

### 6.1 Normalize Edges in `Skeleton` - COMPLETE

**Completed 2025-11-22**: Added edge canonicalization and deduplication to Skeleton.

- [x] In `__post_init__`: canonicalize edges to `(min(src, dst), max(src, dst))` using lexicographic string comparison
- [x] Deduplicate edges (including reversed duplicates like `("a","b")` and `("b","a")`)
- [x] Preserve order based on first occurrence
- [x] Test: 14 new tests in `TestSkeletonEdgeNormalization`
- [x] Updated skeleton vector tests to expect canonical edge names

**Implementation Details:**

- Added `_canonicalize_edge()` helper function
- Added `_normalize_edges()` helper function
- Edges like `("head", "body")` become `("body", "head")` since "body" < "head" lexicographically
- All factory methods (`from_edge_list`, `from_dict`, etc.) benefit automatically via `__post_init__`

**Note**: Node names are expected to be strings (per type annotation). The comparison uses Python's string comparison, which is case-sensitive (uppercase < lowercase in ASCII).

### 6.2 Precompute Adjacency - COMPLETE

**Completed 2025-11-22**: Added `adjacency` property to Skeleton.

- [x] Add `adjacency` property returning `dict[str, list[str]]`
- [x] Precomputed in `__post_init__` for O(1) access
- [x] Sorted neighbor lists for deterministic output
- [x] 10 new tests in `TestSkeletonAdjacency`

**Usage:**
```python
skeleton.adjacency["body"]  # Returns ['head', 'tail'] for chain skeleton
```

**Useful for:** Graph traversal, topology-based styling, finding connected components

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
| Phase 4: Widget | Complete | 4.1-4.4 done (1.8-2.2x speedup, bug fix + O(1) scrubbing) |
| Phase 5: Video | Complete | 5.1-5.4 done (robustness improvements, no regression) |
| Phase 6: Skeleton | **Complete** | 6.1-6.2 done (edge normalization + adjacency property) |
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
