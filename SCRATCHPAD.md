# VideoOverlay Implementation Scratchpad

**Started**: 2025-11-22
**Current Phase**: Milestone 3 (Video I/O) - Task 3.1

---

## Dependency Analysis

**Issue Found**: Task I.1 (Update Type Signatures) requires `VideoOverlay` to exist, but `VideoOverlay` is created in Task 2.1. This is a dependency ordering issue in TASKS.md.

**Resolution**: Execute tasks in dependency order:
1. I.3, I.4, I.5, I.6 - Verification and prep tasks (no VideoOverlay needed)
2. I.2 - Fix artist reuse (no VideoOverlay needed)
3. 1.1, 1.2 - Calibration infrastructure (needed by VideoCalibration)
4. 2.1 - Create VideoOverlay dataclass
5. I.1 - Update type signatures (NOW VideoOverlay exists)
6. Continue with 2.2, 3.x, etc.

---

## Session Log

### 2025-11-22

- Read PLAN.md and TASKS.md
- Identified dependency issue: I.1 needs VideoOverlay but it's created in 2.1
- **Completed M0**: I.5, I.3, I.4, I.6, I.2 (integration pre-requisites)
- **Completed M1**: 1.1, 1.2 (calibration infrastructure)
  - `calibrate_from_scale_bar()` - scale bar calibration
  - `calibrate_from_landmarks()` - landmark-based calibration
  - `VideoCalibration` - dataclass with serialization
  - 19 unit tests passing
- **Completed M2 (partial)**: 2.1 - VideoOverlay dataclass
  - All fields implemented: source, calibration, times, alpha, z_order, crop, downsample, interp
  - `__post_init__` validation: alpha bounds, downsample, array shape/dtype/channels
  - Comprehensive NumPy-style docstring with examples
  - Added to `__all__` in main package and animation module
  - 22 unit tests passing
- **Completed I.1**: Update type signatures
  - Updated overlays parameter types in core.py and visualization.py
  - Added VideoOverlay and VideoCalibration exports to animation/__init__.py
- **Completed M2**: 2.2 - VideoData internal container
  - All fields: frame_indices, reader, transform_to_env, env_bounds, alpha, z_order
  - `get_frame()` method returns frame or None for out-of-range
  - Pickle-safety verified for parallel rendering
  - 6 unit tests passing
- **Completed M3 (partial)**: 3.1 - VideoReader class
  - Lazy-loading video reader with LRU caching
  - OpenCV backend for video reading
  - Crop and downsample support
  - Pickle-safe (cache dropped on serialization)
  - 18 unit tests passing
- **Completed M3 (partial)**: 3.2 - Pipeline integration
  - `_find_nearest_indices()` helper for index mapping
  - `_validate_video_env()` with WHAT/WHY/HOW error messages
  - VideoOverlay handling in `_convert_overlays_to_data()`
  - Added `videos` field to OverlayData
  - 11 additional tests passing
- **Completed M3**: 3.3 - Update OverlayData
  - Backend imports updated (ruff cleaned unused VideoData imports - will be used in M4)
  - Fixed `_find_nearest_indices()` edge case for single-point sources
  - Added defensive check for empty source arrays
  - 743 tests passing (741 animation + 2 new edge case tests)
  - Code review approved

- **Completed M4**: 4.1 - Napari Backend
  - Added `build_env_to_napari_matrix()` to transforms.py
    - 3x3 homogeneous matrix encoding env→napari transform
    - Matches `transform_coords_for_napari()` exactly (6 unit tests)
  - Implemented `_add_video_layer()` in napari_backend.py
    - Single-frame initialization with napari Image layer
    - Affine transform for video positioning
    - z_order handling via layer reordering
    - Opacity/alpha support
    - Frame metadata storage for callbacks
  - Implemented `_build_video_napari_affine()`
    - Chains video→env→napari transforms
    - Falls back to identity for non-grid environments
  - Implemented `_make_video_frame_callback()`
    - Connects to viewer.dims.events.current_step
    - Updates all video layers when animation frame changes
  - Integrated video layer rendering in render_napari() and _render_multi_field_napari()
  - 9 new tests for video overlays, all passing
  - 758 animation tests passing total
  - **Bug fixes during manual testing**:
    - Fixed `VideoData.get_frame()` to use VideoReader subscript access
    - Fixed `_build_video_napari_affine()` to get actual VideoReader dimensions
    - Added `_make_video_frame_callback()` to update video on scrubbing
  - Manual test created: `test_video_alignment.py`
  - **Manual test PASSED**: video fills environment, corners aligned, scrubbing works, playback works

---

## Current Task

**Working on**: Milestone 4 - Backend Rendering (Task 4.2 - Video Export Backend)

---

## Blockers

None currently.

---

## Decisions Made

1. Reordering tasks to respect dependencies while following TDD
