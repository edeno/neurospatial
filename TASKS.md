# VideoOverlay Implementation Tasks

**Feature**: Video Overlay Integration (v0.5.0)

**Reference**: See [PLAN.md](PLAN.md) for detailed specifications and code examples.

**Status Legend**: `[ ]` = pending, `[x]` = complete, `[~]` = in progress, `[-]` = blocked

---

## Milestone 0: Integration Pre-requisites

> **Goal**: Prepare codebase for VideoOverlay without breaking existing functionality.
>
> **Verification**: `uv run pytest` passes after each task.

### I.1: Update Type Signatures

- [x] Update `overlays` parameter type in [src/neurospatial/animation/core.py:67-68](src/neurospatial/animation/core.py#L67-L68)
- [x] Update `overlays` parameter type in [src/neurospatial/environment/visualization.py](src/neurospatial/environment/visualization.py)
- [x] Add `VideoOverlay` to `__all__` in [src/neurospatial/**init**.py](src/neurospatial/__init__.py)
- [x] Add exports to [src/neurospatial/animation/\_\_init\_\_.py](src/neurospatial/animation/__init__.py):
  - [x] `from .overlays import VideoOverlay`
  - [x] `from ..transforms import VideoCalibration`
  - [x] Update `__all__` list
- [x] Run tests: `uv run pytest tests/animation/ -v`

### I.2: Fix Artist Reuse Logic

> **Pre-requisite**: Read `_render_worker()` in `_parallel.py` first.

- [x] Review current `ax.images[0]` usage in [src/neurospatial/animation/_parallel.py](src/neurospatial/animation/_parallel.py)
- [x] Create `FrameArtists` dataclass or dict to track artists by purpose
- [x] Update `_render_worker()` to track field artist explicitly
- [x] Update [src/neurospatial/animation/backends/widget_backend.py](src/neurospatial/animation/backends/widget_backend.py) if needed
- [x] Run tests: `uv run pytest tests/animation/ -v`

**Fix Applied**: Changed `ax.images[0]` to `ax.images[-1]` in `_parallel.py:1233`.
Field is always added last by `env.plot_field()`, so `[-1]` is correct even when
video layers are rendered first (z_order="below"). Widget backend was clean.

### I.3: Verify DRY Transform Reuse

- [x] Confirm `flip_y()` exists at [src/neurospatial/transforms.py:459-476](src/neurospatial/transforms.py#L459-L476)
- [x] Confirm `simple_scale()` exists at [src/neurospatial/calibration.py:7-68](src/neurospatial/calibration.py#L7-L68)
- [x] Document plan to wrap these in new calibration helpers (no code changes yet)

**DRY Plan**: Task 1.1 will use existing functions:

- `flip_y(frame_height_px)` - Y-axis flip for video origin conversion
- `scale_2d(sx, sy)` - uniform/anisotropic scaling (transforms.py:396)
- Pattern: `flip_y(height) @ scale_2d(cm_per_px, cm_per_px)` for px→cm with Y-flip

### I.4: Verify Napari Field Orientation

- [x] Review `field_to_rgb_for_napari()` in [src/neurospatial/animation/rendering.py](src/neurospatial/animation/rendering.py)
- [x] Document the transpose + flip pattern for video affine alignment
- [x] Create test plan for alignment verification (Task 6.3)

**Napari Orientation Pattern** (rendering.py:440-450):

1. **Transpose**: `(n_x, n_y, 3)` → `(n_y, n_x, 3)` for napari (row, col) convention
2. **Y-Flip**: `np.flip(transposed, axis=0)` because:
   - Environment: Y increases upward (row 0 = min Y = bottom)
   - Napari: Y increases downward (row 0 = top)

**Video Affine Alignment**: Task 4.1 will implement `build_env_to_napari_matrix()` that encodes this same transformation in matrix form for the video layer affine parameter.

### I.5: Add imageio Dependency

- [x] Add `imageio>=2.35.0` to dev dependencies in [pyproject.toml](pyproject.toml)
- [x] Add `imageio-ffmpeg>=0.5.1` to optional `[video]` extras
- [x] Run `uv sync` to update lockfile
- [x] Verify: `uv run python -c "import imageio; print(imageio.__version__)"`

### I.6: Verify dimension_ranges Validation

- [x] Review `EnvScale.from_env()` at [src/neurospatial/animation/transforms.py:123-130](src/neurospatial/animation/transforms.py#L123-L130)
- [x] Plan `_validate_video_env()` function for Task 3.2
- [x] Document fallback behavior for non-grid 2D environments

**Current Behavior** (EnvScale.from_env):
Returns `None` if env lacks `dimension_ranges` OR `layout.grid_shape`.
When `None`, `_warn_fallback()` emits WHAT/WHY/HOW warning.

**Task 3.2 Validation Plan** (`_validate_video_env()`):

1. **2D Required**: `env.n_dims == 2` or raise `ValueError` (1D/3D not supported)
2. **dimension_ranges Required**: Must exist and be finite (for video bounding box)
3. **grid_shape Optional**: Non-grid 2D envs work with fallback warning

**Fallback for Non-Grid 2D**:

- Use `env.dimension_ranges` for video extent (bounding box)
- Emit existing `_warn_fallback()` warning (alignment may be approximate)
- Video still renders, covers full bounding box

**M0 Checkpoint**: `uv run pytest` passes, no regressions

---

## Milestone 1: Calibration Infrastructure

> **Goal**: Pixel↔cm transforms working with unit tests.
>
> **Files**: `src/neurospatial/transforms.py`, `tests/test_transforms.py`

### Task 1.1: Add Calibration Helpers

- [x] Implement `calibrate_from_scale_bar()` in [src/neurospatial/transforms.py](src/neurospatial/transforms.py)
  - [x] Compute scale from pixel distance between endpoints
  - [x] Compose with `flip_y()` for Y-axis correction
  - [x] Return `Affine2D` with working inverse
- [x] Implement `calibrate_from_landmarks()` in [src/neurospatial/transforms.py](src/neurospatial/transforms.py)
  - [x] Use existing `estimate_transform()` internally
  - [x] Compose with `flip_y()` for Y-axis correction
  - [x] Support `kind` parameter: "rigid", "similarity", "affine"
- [x] Write unit tests in [tests/test_transforms.py](tests/test_transforms.py):
  - [x] `test_scale_bar_calibration`
  - [x] `test_landmark_calibration_rigid`
  - [x] `test_landmark_calibration_similarity`
  - [x] `test_roundtrip_px_cm_px` with tolerance `max(1e-4, 1e-6 * extent)`
- [x] Verify: `uv run pytest tests/test_transforms.py -v -k calibrat`

### Task 1.2: Add VideoCalibration Dataclass

- [x] Implement `VideoCalibration` dataclass in [src/neurospatial/transforms.py](src/neurospatial/transforms.py)
  - [x] `transform_px_to_cm: Affine2D` attribute
  - [x] `frame_size_px: tuple[int, int]` attribute (width, height)
  - [x] `transform_cm_to_px` property (cached inverse)
  - [x] `cm_per_px` property (approximate scale)
- [x] Add `to_dict()` / `from_dict()` for JSON serialization
- [x] Write unit tests:
  - [x] `test_calibration_inverse`
  - [x] `test_calibration_serialization`
- [x] Verify: `uv run pytest tests/test_transforms.py -v -k VideoCalibration`

**M1 Checkpoint**: All calibration tests pass

---

## Milestone 2: Data Model

> **Goal**: `VideoOverlay` and `VideoData` dataclasses defined.
>
> **Files**: `src/neurospatial/animation/overlays.py`

### Task 2.1: Create VideoOverlay Dataclass

- [x] Add `VideoOverlay` dataclass to [src/neurospatial/animation/overlays.py](src/neurospatial/animation/overlays.py)
  - [x] `source: str | Path | NDArray[np.uint8]`
  - [x] `calibration: VideoCalibration | None`
  - [x] `times: NDArray[np.float64] | None`
  - [x] `alpha: float = 0.7`
  - [x] `z_order: Literal["below", "above"] = "below"`
  - [x] `crop: tuple[int, int, int, int] | None = None`
  - [x] `downsample: int = 1`
  - [x] `interp: Literal["linear", "nearest"] = "nearest"`
- [x] Implement `__post_init__` validation:
  - [x] File path existence check (deferred to VideoReader for lazy loading)
  - [x] Array shape/dtype validation
  - [x] Alpha bounds check (0.0-1.0)
- [x] Add comprehensive docstring with examples
- [x] Add to `__all__` (in neurospatial/**init**.py)

### Task 2.2: Create VideoData Internal Container

- [x] Add `VideoData` dataclass to [src/neurospatial/animation/overlays.py](src/neurospatial/animation/overlays.py)
  - [x] `frame_indices: NDArray[np.int_]`
  - [x] `reader: VideoReader | NDArray[np.uint8]`
  - [x] `transform_to_env: Affine2D | None`
  - [x] `env_bounds: tuple[float, float, float, float]`
  - [x] `alpha: float`
  - [x] `z_order: Literal["below", "above"]`
- [x] Implement `get_frame(anim_frame_idx)` method
  - [x] Return `None` for index -1 (out of range)
  - [x] Return RGB frame for valid indices
- [x] Verify pickle-safety for parallel rendering

---

## Milestone 3: Video I/O

> **Goal**: `VideoReader` loads frames with LRU caching.
>
> **Files**: `src/neurospatial/animation/_video_io.py` (NEW)

### Task 3.1: Create VideoReader Class

- [x] Create new file [src/neurospatial/animation/_video_io.py](src/neurospatial/animation/_video_io.py)
- [x] Implement `VideoReader` class:
  - [x] `__init__(path, cache_size=100, downsample=1, crop=None)`
  - [x] `__getitem__(frame_idx)` with LRU caching
  - [x] `get_timestamps()` method (ffprobe or fps fallback)
  - [x] `__reduce__()` for pickle support (drop cache, keep path)
- [x] Attributes:
  - [x] `n_frames: int`
  - [x] `fps: float`
  - [x] `frame_size_px: tuple[int, int]` (after crop/downsample)
  - [x] `original_size_px: tuple[int, int]` (before processing)
  - [x] `crop_offset_px: tuple[int, int]`
  - [x] `duration: float`
- [x] Backend selection: OpenCV (preferred) → imageio (fallback)
- [x] Write unit tests in [tests/animation/test_video_io.py](tests/animation/test_video_io.py) (NEW):
  - [x] `test_reader_loads_metadata`
  - [x] `test_reader_lazy_loading`
  - [x] `test_reader_lru_cache`
  - [x] `test_reader_pickle_roundtrip`
  - [x] `test_reader_timestamps`
- [x] Verify: `uv run pytest tests/animation/test_video_io.py -v`

**M2 Checkpoint**: VideoReader tests pass

### Task 3.2: Pipeline Integration

- [x] Create `_find_nearest_indices()` helper in [src/neurospatial/animation/overlays.py](src/neurospatial/animation/overlays.py)
  - [x] Factor out index-finding from `_interp_nearest()`
  - [x] Return -1 for out-of-range queries
- [x] Refactor `_interp_nearest()` to use `_find_nearest_indices()`
- [x] Add `_validate_video_env()` function:
  - [x] Check `env.n_dims == 2`
  - [x] Check `dimension_ranges` exists and is finite
  - [x] WHAT/WHY/HOW error messages
- [x] Update `_convert_overlays_to_data()` to handle `VideoOverlay`:
  - [x] Create VideoReader or wrap array
  - [x] Compute frame index mapping
  - [x] Compose crop offset into transform
  - [x] Warn if no calibration provided
- [x] Write tests for temporal alignment edge cases

### Task 3.3: Update OverlayData

- [x] Add `videos: list[VideoData] = field(default_factory=list)` to `OverlayData`
- [x] Update all backend imports/type hints
- [x] Verify no `AttributeError` when accessing `overlay_data.videos`
- [x] Run full test suite: `uv run pytest tests/animation/ -v`

**M3 Checkpoint**: 743 tests pass (741 animation + 2 new edge case tests)

---

## Milestone 4: Backend Rendering

> **Goal**: Video renders correctly in all backends.

### Task 4.1: Napari Backend

> **Note**: Requires manual testing with napari viewer.

- [x] Add `build_env_to_napari_matrix()` to [src/neurospatial/animation/transforms.py](src/neurospatial/animation/transforms.py)
- [x] Implement `_add_video_layer()` in [src/neurospatial/animation/backends/napari_backend.py](src/neurospatial/animation/backends/napari_backend.py):
  - [x] Single-frame initialization (NOT 4D array)
  - [x] Compute affine from VideoData transform + EnvScale
  - [x] Register dims callback for frame updates
  - [x] In-place update: `layer.data[...] = frame`
  - [x] Handle z_order via layer positioning
- [x] Implement `_build_video_napari_affine()`:
  - [x] Use `build_env_to_napari_matrix()` (DRY)
  - [x] Handle fallback for non-grid environments
- [x] Implement fallback transform for non-grid envs (returns identity matrix)
- [x] Manual test: verify video aligns with field at known corners

**M4 Checkpoint**: Napari displays video with correct spatial alignment ✓

### Task 4.2: Video Export Backend

- [x] Implement `VideoFrameRenderer` class in [src/neurospatial/animation/_parallel.py](src/neurospatial/animation/_parallel.py):
  - [x] `__init__(ax, video_data, env)`
  - [x] `render(ax, frame_idx)` with artist reuse
  - [x] `_compute_extent()` from transform or env_bounds
- [x] Implement `_render_video_background()` for parallel rendering:
  - [x] Fresh artist per frame (parallel-safe)
  - [x] Handle -1 frame indices (skip rendering)
  - [x] Correct z-order via `zorder` parameter
- [x] Update render pipeline to call video rendering:
  - [x] Before field for `z_order="below"`
  - [x] After field for `z_order="above"`
- [x] Write integration tests in [tests/animation/test_video_overlays.py](tests/animation/test_video_overlays.py):
  - [x] TestVideoOverlayExportRendering (7 tests)
  - [x] TestRenderAllOverlaysWithVideo (2 tests)
  - [x] TestVideoFrameRenderer (5 tests)
- [x] Verify: `uv run pytest tests/animation/test_video_overlays.py -v` (36 tests pass)

**M3 Checkpoint**: Video export produces correct spatial composite ✓

### Task 4.3: Widget and HTML Backends

- [x] Update [src/neurospatial/animation/backends/widget_backend.py](src/neurospatial/animation/backends/widget_backend.py):
  - [x] Widget backend already supports video via `_render_all_overlays()` (Task 4.2)
  - [x] Uses existing `render_field_to_png_bytes_with_overlays()` which handles video
  - [x] No additional changes needed (artist reuse handled by existing code path)
- [x] Update [src/neurospatial/animation/backends/html_backend.py](src/neurospatial/animation/backends/html_backend.py):
  - [x] Add video skip with WHAT/WHY/HOW warning
  - [x] Video stripped automatically (HTML serializer ignores video data)
  - [x] Continue rendering field + other overlays (positions, regions)
- [x] Write tests:
  - [x] `test_widget_backend_renders_video_overlay`
  - [x] `test_html_backend_warns_on_video_overlay`
  - [x] `test_html_backend_still_renders_with_video_overlay_present`
  - [x] `test_html_backend_renders_other_overlays_with_video_present`

**M4 Checkpoint**: Widget and HTML backends handle video overlays correctly ✓

---

## Milestone 5: Environment Integration

> **Goal**: Convenience function for video calibration.

### Task 5.1: Create calibrate_video() Function

- [x] Create new file [src/neurospatial/animation/calibration.py](src/neurospatial/animation/calibration.py)
- [x] Implement `calibrate_video()`:
  - [x] Accept `video_path`, `env`, and calibration params
  - [x] Support scale_bar, landmarks_px/landmarks_env, cm_per_px methods
  - [x] Return `VideoCalibration` object
  - [x] Validate bounds coverage, warn if mismatch
- [x] Add to animation package exports
- [x] Write tests:
  - [x] `test_calibrate_video_scale_bar`
  - [x] `test_calibrate_video_landmarks`
  - [x] `test_calibrate_video_bounds_warning`

---

## Milestone 6: Testing

> **Goal**: Comprehensive test coverage.

### Task 6.1: Calibration Tests

- [x] Complete all tests from Task 1.1 and 1.2
- [x] Add edge case tests:
  - [x] `test_rejects_ill_conditioned_landmarks`
  - [x] `test_warns_bounds_mismatch`

### Task 6.2: Video I/O Tests

- [x] Complete all tests from Task 3.1
- [x] Add fixtures to [tests/conftest.py](tests/conftest.py):
  - [x] `sample_video` (16x16, 10 frames)
  - [x] `sample_video_array`
  - [x] `sample_calibration`

### Task 6.3: Backend Integration Tests

- [x] Create [tests/animation/test_video_overlay.py](tests/animation/test_video_overlay.py)
- [x] Napari tests (mark with `@pytest.mark.slow`):
  - [x] `test_video_layer_added`
  - [x] `test_video_spatial_alignment`
  - [x] `test_video_temporal_sync`
- [x] Export tests:
  - [x] `test_video_composited_in_output`
  - [x] `test_video_parallel_rendering`

### Task 6.4: Validation Tests

- [x] Create [tests/animation/test_video_validation.py](tests/animation/test_video_validation.py)
- [x] Environment validation:
  - [x] `test_rejects_1d_environment`
  - [x] `test_non_grid_2d_environment_works_with_warning`
  - [x] `test_non_grid_extent_uses_dimension_ranges`
- [x] Napari-free operation:
  - [x] `test_video_export_without_napari`
  - [x] `test_import_without_napari`

### Task 6.5: Add Test Fixtures

- [x] Add environment fixtures to [tests/conftest.py](tests/conftest.py):
  - [x] `linearized_env` (1D, for rejection tests)
  - [x] `polygon_env` (non-grid 2D, for fallback tests)
  - [x] `masked_env` (grid 2D, for full support tests)
- [x] Add pytest marker configuration to [pyproject.toml](pyproject.toml):
  - Already configured in pytest.ini: `slow`, `gui`, `napari` markers

**M6 Checkpoint**: `uv run pytest` all tests pass ✓

---

## Milestone 7: Documentation

> **Goal**: Users can discover and use VideoOverlay.

### Task 7.1: Update CLAUDE.md

- [x] Add VideoOverlay to Quick Reference section
- [x] Add code example with calibration
- [x] Add to "Animation overlays (v0.4.0+)" section
- [x] Update backend capabilities matrix
- [x] Add common gotcha for video coordinates

### Task 7.2: Create Example Notebook

- [x] Create [examples/18_video_overlay.ipynb](examples/18_video_overlay.ipynb)
- [x] Sections:
  - [x] Loading and inspecting video metadata
  - [x] Calibrating with scale bar method
  - [x] Calibrating with landmark correspondences
  - [x] Creating VideoOverlay with various options
  - [x] Animating fields with video background
  - [x] Exporting synchronized video
  - [x] Performance tips for large videos
- [x] Test notebook runs without errors

### Task 7.3: Update Animation Guide

- [x] Update `docs/guides/animation.md` (or equivalent)
- [x] Add video integration section:
  - [x] Coordinate systems explanation
  - [x] Calibration methods comparison
  - [x] Backend capabilities (update matrix)
  - [x] Troubleshooting spatial misalignment

---

## Milestone 8: Post-Review Refinements

> **Goal**: Address code review findings for code quality, maintainability, and correctness.
>
> **Reference**: External code review of VideoOverlay implementation (2025-11-22)

### Task 8.1: Fix VideoOverlay.interp Not Honored (BUG)

> **Priority**: High - Documented behavior doesn't match implementation

- [x] Update `_convert_overlays_to_data()` in [src/neurospatial/animation/overlays.py](src/neurospatial/animation/overlays.py)
  - [x] For VideoOverlay with `interp="nearest"`: Use current `_find_nearest_indices()` (no change)
  - [x] For VideoOverlay with `interp="linear"`: Implement frame blending between adjacent video frames
    - Option A: True linear blend (expensive: blend two RGB frames per animation frame)
    - **Option B chosen**: Restrict `interp` to `"nearest"` only and update docstring/validation
  - [x] Decision: Document trade-off and choose approach (prefer Option B for v0.5.0)
- [x] If Option B: Update `VideoOverlay.interp` docstring to clarify only "nearest" is currently supported
- [x] If Option B: Add validation in `__post_init__` to warn if "linear" is requested
- [x] Write test: `test_video_overlay_interp_warning`

### Task 8.2: Normalize Regions Representation

> **Priority**: Medium - Simplifies backend code

- [ ] Update `OverlayData.regions` to always be `dict[int, list[str]] | None`
- [ ] Add normalization in `_convert_overlays_to_data()`:
  - [ ] Convert `list[str]` → `{0: list_of_regions}` (apply to all frames)
  - [ ] Pass through `dict[int, list[str]]` unchanged
- [ ] Update HTML backend to expect normalized format (remove special-casing)
- [ ] Write test: `test_regions_normalization_list_to_dict`

### Task 8.3: Centralize Pickle Validation Messages

> **Priority**: Low - Improves consistency

- [ ] Create `_pickling_guidance()` helper in [src/neurospatial/animation/_utils.py](src/neurospatial/animation/_utils.py) (NEW file)
- [ ] Refactor `_validate_pickle_ability()` in overlays.py to use helper
- [ ] Refactor `_validate_env_pickleable()` in core.py to use helper
- [ ] Ensure both error messages have identical "HOW" guidance

### Task 8.4: Document Coordinate Conventions

> **Priority**: Medium - Reduces debugging time for contributors

- [ ] Create [src/neurospatial/animation/COORDINATES.md](src/neurospatial/animation/COORDINATES.md) (internal dev doc)
  - [ ] Define three coordinate spaces: Video pixel, Environment cm, Napari world
  - [ ] Document Y-flip policy: "flip happens ONCE in VideoCalibration"
  - [ ] Include ASCII diagrams showing axis orientations
- [ ] Add comments in `_add_video_layer()` explaining frame mapping contract
- [ ] Add comments in `calibrate_from_scale_bar()` explaining coordinate flow

### Task 8.5: Add VideoReaderProtocol for Type Safety

> **Priority**: Low - Improves IDE support and reduces `Any`

- [ ] Create `VideoReaderProtocol` in [src/neurospatial/animation/_video_io.py](src/neurospatial/animation/_video_io.py):

  ```python
  class VideoReaderProtocol(Protocol):
      n_frames: int
      frame_size_px: tuple[int, int]
      def __getitem__(self, idx: int) -> NDArray[np.uint8]: ...
  ```

- [ ] Update `VideoData.reader` type: `NDArray[np.uint8] | VideoReaderProtocol`
- [ ] Remove `Any` type annotations where VideoReader is used

### Task 8.6: Document Overlay JSON Schema for HTML

> **Priority**: Low - Improves maintainability

- [ ] Add schema documentation to [src/neurospatial/animation/backends/html_backend.py](src/neurospatial/animation/backends/html_backend.py):

  ```python
  # Overlay JSON Schema:
  # {
  #   "positions": {"name": {"x": [...], "y": [...]}},
  #   "regions": {"name": {"polygon": [[x, y], ...], "color": "..."}},
  #   ...
  # }
  ```

- [ ] Add type stub or TypedDict for overlay JSON structure

### Task 8.7: Improve VideoReader Performance (FUTURE)

> **Priority**: Low - Performance optimization for future release
> **Note**: LRU cache already mitigates this for most use cases

- [ ] Investigate persistent handle approach:
  - [ ] Keep `cv2.VideoCapture` open per instance (not per-frame)
  - [ ] Handle re-opening in `__setstate__` for pickle roundtrip
- [ ] Benchmark sequential read performance with/without persistent handle
- [ ] Decision: Only implement if benchmark shows >2x improvement
- [ ] If implemented, add `prefer_persistent_handle: bool = True` parameter

### Task 8.8: Improve animate_fields Docstring

> **Priority**: Low - Documentation improvement

- [ ] Add explicit "multi-field mode" description to `fields` parameter in [core.py:85-86](src/neurospatial/animation/core.py#L85-L86):

  ```text
  Multi-field mode: For napari backend only, fields can be a list of sequences
  where each sequence contains multiple spatial fields per frame. Example:
  [[field1_frame0, field2_frame0], [field1_frame1, field2_frame1], ...]
  ```

**M8 Checkpoint**: `uv run pytest` passes, improved code quality

---

## Final Verification

- [ ] Full test suite: `uv run pytest`
- [ ] Linting: `uv run ruff check . && uv run ruff format .`
- [ ] Type checking: `uv run mypy src/neurospatial/`
- [ ] Example notebook runs: `uv run jupyter nbconvert --execute examples/18_video_overlay.ipynb`
- [ ] CLAUDE.md examples are copy-pasteable and work

---

## Notes

- **Coordinate convention reminder**: `frame_size_px` is `(width, height)`, NumPy arrays are `(height, width, 3)`
- **Y-flip happens ONCE**: In `VideoCalibration.transform_px_to_cm`, then use `origin="lower"` everywhere
- **Memory**: Never preload all video frames; use streaming with LRU cache
- **Testing**: Use tiny 16x16 videos for fast CI; mark napari tests as `@pytest.mark.slow`
