# VideoOverlay Implementation Scratchpad

**Started**: 2025-11-22
**Current Phase**: Milestone 5 (Environment Integration) - Task 5.1

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

- **Completed M4**: 4.2 - Video Export Backend
  - Implemented `_render_video_background()` in _parallel.py
    - Fresh imshow artist per frame (parallel-safe)
    - Handles -1 frame indices (skip rendering)
    - Correct z-order: -1 for "below", +1 for "above"
  - Implemented `VideoFrameRenderer` class in _parallel.py
    - Reuses imshow artist via set_data() for sequential rendering
    - `_compute_extent()` from transform or env_bounds
    - Hides artist for invalid frames (-1 indices)
  - Updated `_render_all_overlays()` to handle video overlays
    - Renders videos before regions for z_order="below"
    - Renders videos after head direction for z_order="above"
  - 14 new tests in test_video_overlays.py:
    - TestVideoOverlayExportRendering (7 tests)
    - TestRenderAllOverlaysWithVideo (2 tests)
    - TestVideoFrameRenderer (5 tests)
  - All 36 video overlay tests passing
  - All 772 animation tests passing
  - Code review approved

- **Updated VideoOverlay defaults** (post-review fix):
  - Changed default `z_order` from `"below"` to `"above"` (video on top of field)
  - Changed default `alpha` from `0.7` to `0.5` (balanced 50/50 blend)
  - Updated docstrings with usage guidance
  - Added VideoOverlay best practices to CLAUDE.md
  - Rationale: `z_order="below"` hides video behind opaque fields; `"above"` with alpha blending works universally

- **Completed M4**: 4.3 - Widget and HTML Backends
  - **Widget Backend**: Already supports video overlays via `_render_all_overlays()` from Task 4.2
    - `render_field_to_png_bytes_with_overlays()` calls `_render_all_overlays()` which handles video
    - No additional changes needed (artist reuse already handled)
  - **HTML Backend**: Added video overlay skip warning with WHAT/WHY/HOW format
    - Warns users that video overlays cannot render in standalone HTML
    - Suggests video or napari backends as alternatives
    - Other overlays (positions, regions) continue to render
  - 4 new tests:
    - `TestWidgetBackendVideoOverlay::test_widget_backend_renders_video_overlay`
    - `TestHTMLBackendVideoOverlay::test_html_backend_warns_on_video_overlay`
    - `TestHTMLBackendVideoOverlay::test_html_backend_still_renders_with_video_overlay_present`
    - `TestHTMLBackendVideoOverlay::test_html_backend_renders_other_overlays_with_video_present`
  - All 776 animation tests passing
  - Code review approved

---

## Current Task

**Completed**: Milestone 5 - Task 5.1 (calibrate_video function)

---

## Session Log (continued)

### 2025-11-22 (continued)

- **Completed M5**: 5.1 - calibrate_video() convenience function
  - Created `src/neurospatial/animation/calibration.py` with `calibrate_video()` function
  - Supports three calibration methods:
    - `scale_bar`: Two pixel endpoints + known length in cm
    - `landmarks_px/landmarks_env`: Corresponding point pairs
    - `cm_per_px`: Direct scale factor
  - Returns `VideoCalibration` object
  - Validates bounds coverage with WHAT/WHY/HOW warning if env exceeds video
  - Uses existing calibration functions from transforms.py (DRY)
  - Added `cm_per_px > 0` validation (from code review)
  - 17 unit tests passing
  - Code review approved
  - Exported from `neurospatial.animation` module

- **Completed M6 (partial)**: 6.1 - Calibration Edge Case Tests
  - Added `test_rejects_ill_conditioned_landmarks` - tests collinear landmarks rejection
  - Added `test_rejects_nearly_collinear_landmarks` - tests nearly collinear landmarks
  - Added `test_warns_bounds_mismatch` - tests warning when env exceeds video
  - Added `test_no_warning_when_video_covers_env` - tests no warning case
  - Implemented ill-conditioned landmark check in `calibrate_from_landmarks()`
    - Uses SVD to detect collinearity (singular value ratio check)
    - Threshold: `s[1] < 1e-6 * s[0]` indicates collinear points
    - Checks both pixel and environment landmarks
  - All 23 transform tests passing
  - All 816 animation tests passing

- **Completed M6 (partial)**: 6.2 - Video I/O Tests
  - Task 3.1 tests already exist (18 tests for VideoReader)
  - Added shared fixtures to `tests/conftest.py`:
    - `sample_video` - 16x16 video file, 10 frames, 10 fps
    - `sample_video_array` - numpy array (10, 16, 16, 3)
    - `sample_calibration` - VideoCalibration with 1:1 px→cm mapping
  - All 18 video I/O tests passing
  - All 40 video overlay tests passing

- **Completed M6 (partial)**: 6.3 - Backend Integration Tests
  - Created `tests/animation/test_video_overlay.py` with 6 tests:
    - Napari tests (slow): `test_video_layer_added`, `test_video_spatial_alignment`, `test_video_temporal_sync`
    - Export tests: `test_video_composited_in_output`, `test_video_parallel_rendering`, `test_video_zorder_affects_compositing`
  - All 3 export tests passing
  - Napari tests properly marked with `@pytest.mark.slow` and `@pytest.mark.xdist_group`

- **Completed M6 (partial)**: 6.4 - Validation Tests
  - Created `tests/animation/test_video_validation.py` with 8 tests:
    - Environment validation: `test_rejects_1d_environment`, `test_accepts_2d_grid_environment`, `test_accepts_2d_polygon_environment`
    - Non-grid support: `test_non_grid_2d_environment_works_with_warning`, `test_non_grid_extent_uses_dimension_ranges`
    - Napari-free operation: `test_video_export_without_napari`, `test_import_without_napari`, `test_video_overlay_creation_without_napari`
  - All 8 tests passing
  - Verified `_validate_video_env()` correctly rejects 1D envs and accepts 2D envs

- **Completed M6**: 6.5 - Add Test Fixtures
  - Added shared fixtures to `tests/conftest.py`:
    - `linearized_env` - 1D track environment for rejection tests
    - `polygon_env` - non-grid 2D polygon environment for fallback tests
    - `masked_env` - grid 2D environment for full support tests
  - Updated `test_video_validation.py` to use shared fixtures
  - pytest markers already configured in pytest.ini (`slow`, `gui`, `napari`)
  - All 801 animation tests passing

---

## Current Task

**Completed**: Milestone 8 - Task 8.3 (Centralize Pickle Validation Messages)
**Next**: Milestone 8 - Task 8.4 (Document Coordinate Conventions)

---

## Blockers

None currently.

---

## Decisions Made

1. Reordering tasks to respect dependencies while following TDD
2. Ill-conditioned landmark detection uses SVD of centered points, not condition number of output transform
   - Output transform can have good condition even with collinear inputs
   - Singular value ratio of centered source points reliably detects collinearity

---

## Session Log (2025-11-22 continued)

- **Completed M7 (partial)**: 7.1 - Update CLAUDE.md
  - Updated version from v0.4.0 to v0.5.0
  - Added `calibrate_video()` convenience function documentation with all 3 methods:
    - Scale bar calibration
    - Landmark correspondences
    - Direct cm_per_px scale factor
  - Updated import line to include `calibrate_video` from `neurospatial.animation`
  - Added version annotation `(v0.5.0+)` to video overlay section
  - Updated backend capabilities comment to explicitly mention VideoOverlay
  - Updated backend capability matrix in gotcha #12 to include VideoOverlay
  - Added common gotcha #14: VideoOverlay requires 2D environments
    - Includes support matrix (2D grid ✓, 2D polygon ⚠️, 1D ✗)
  - Verified all imports work via Python
  - All 804 animation tests passing

- **Completed M7 (partial)**: 7.2 - Create Example Notebook
  - Created `examples/18_video_overlay.ipynb` with all required sections:
    - Loading and inspecting video metadata (VideoReader)
    - Calibrating with scale bar method (calibrate_from_scale_bar)
    - Calibrating with landmark correspondences (calibrate_from_landmarks)
    - Creating VideoOverlay with various options (alpha, z_order, crop, downsample)
    - Animating fields with video background (Napari backend)
    - Exporting synchronized video (video backend, widget backend)
    - Performance tips for large videos (cache size, downsampling, subsampling)
  - Fixed trajectory to stay within environment bounds (clipping)
  - Fixed widget backend demo with proper temporal alignment (times parameter)
  - **Bug fix**: Changed `video_data.transform_to_env.matrix` to `.A` in napari_backend.py
    - `Affine2D` uses `.A` attribute, not `.matrix`
  - Notebook executes without errors
  - All 804 animation tests passing

- **Completed M7**: 7.3 - Update Animation Guide
  - Updated `docs/animation_overlays.md`:
    - Added VideoOverlay section with full parameter documentation
    - Added Backend Capabilities matrix with Video column
    - Added "Video Overlay Coordinate Systems" section with:
      - Coordinate spaces table (Video Pixel, Environment, Napari)
      - Three calibration methods: scale bar, landmarks, cm_per_px
      - Calibration comparison table
      - Troubleshooting section for spatial misalignment
    - Added VideoOverlay and VideoCalibration API references
    - Updated animate_fields() signature to include VideoOverlay
  - Updated `docs/user-guide/animation.md`:
    - Added "Overlay Support by Backend" matrix with Video column
    - Added link to animation_overlays.md for VideoOverlay docs
  - All 804 animation tests passing
  - ruff and mypy pass

- **Completed M8**: 8.1 - Fix VideoOverlay.interp Not Honored
  - **Decision**: Chose Option B - restrict to "nearest" only with warning
    - Linear interpolation for video would require blending adjacent RGB frames
    - Computationally expensive and rarely needed in practice
  - Added `__post_init__` validation that emits UserWarning when `interp="linear"`
    - WHAT/WHY/HOW format warning message
  - Updated `VideoOverlay.interp` docstring to clarify only "nearest" is implemented
    - Parameter docs clearly state "linear" is not yet implemented
    - Attributes docs note current limitation
  - Added tests:
    - `test_interp_linear_emits_warning` - verifies warning contains "linear" and "nearest"
    - `test_interp_nearest_no_warning` - verifies no warning for default value
  - All 806 animation tests passing
  - ruff and mypy pass

- **Completed M8**: 8.2 - Normalize Regions Representation
  - Changed `OverlayData.regions` type from `list[str] | dict[int, list[str]] | None`
    to `dict[int, list[str]] | None` (simpler, consistent format)
  - Added `show_regions` parameter to `_convert_overlays_to_data()`:
    - `show_regions=["a", "b"]` → `{0: ["a", "b"]}` (key 0 = all frames)
    - `show_regions=True` → `{0: list(env.regions.keys())}`
    - `show_regions=False` → `None`
  - Updated HTML backend to prefer `overlay_data.regions` if available:
    - Both `_estimate_overlay_json_size()` and `_serialize_overlay_data()` updated
    - Falls back to `show_regions` parameter for backwards compatibility
  - Added `TestRegionsNormalization` test class with 4 tests:
    - `test_regions_normalization_list_to_dict`
    - `test_regions_normalization_bool_true_to_dict`
    - `test_regions_normalization_bool_false_to_none`
    - `test_regions_default_to_none`
  - All 810 animation tests passing
  - ruff and mypy pass

- **Completed M8**: 8.3 - Centralize Pickle Validation Messages
  - Created `src/neurospatial/animation/_utils.py` with `_pickling_guidance()` helper
    - Takes optional `n_workers` parameter for customized examples
    - Returns consistent HOW guidance with 3 numbered options:
      1. Clear caches before rendering (`env.clear_cache()`)
      2. Use serial rendering (`n_workers=1`)
      3. Use non-pickling backend (`backend='html'`)
  - Refactored `_validate_pickle_ability()` in overlays.py:
    - Now uses `_pickling_guidance(n_workers=n_workers)` for HOW section
    - Maintains WHAT/WHY/HOW format
  - Refactored `_validate_env_pickleable()` in core.py:
    - Updated to WHAT/WHY/HOW format (previously different format)
    - Now uses `_pickling_guidance()` for consistent HOW section
  - Updated test regex in `test_route_to_video_backend_pickle_validation`
  - 8 new tests in `tests/animation/test_utils.py`
  - All 818 animation tests passing
  - ruff and mypy pass
