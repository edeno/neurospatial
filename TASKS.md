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

- [ ] Update `overlays` parameter type in [src/neurospatial/animation/core.py:67-68](src/neurospatial/animation/core.py#L67-L68)
- [ ] Update `overlays` parameter type in [src/neurospatial/environment/visualization.py](src/neurospatial/environment/visualization.py)
- [ ] Add `VideoOverlay` to `__all__` in [src/neurospatial/animation/overlays.py](src/neurospatial/animation/overlays.py)
- [ ] Add exports to [src/neurospatial/animation/\_\_init\_\_.py](src/neurospatial/animation/__init__.py):
  - [ ] `from .overlays import VideoOverlay`
  - [ ] `from ..transforms import VideoCalibration`
  - [ ] Update `__all__` list
- [ ] Run tests: `uv run pytest tests/animation/ -v`

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

- [ ] Implement `VideoCalibration` dataclass in [src/neurospatial/transforms.py](src/neurospatial/transforms.py)
  - [ ] `transform_px_to_cm: Affine2D` attribute
  - [ ] `frame_size_px: tuple[int, int]` attribute (width, height)
  - [ ] `transform_cm_to_px` property (cached inverse)
  - [ ] `cm_per_px` property (approximate scale)
- [ ] Add `to_dict()` / `from_dict()` for JSON serialization
- [ ] Write unit tests:
  - [ ] `test_calibration_inverse`
  - [ ] `test_calibration_serialization`
- [ ] Verify: `uv run pytest tests/test_transforms.py -v -k VideoCalibration`

**M1 Checkpoint**: All calibration tests pass

---

## Milestone 2: Data Model

> **Goal**: `VideoOverlay` and `VideoData` dataclasses defined.
>
> **Files**: `src/neurospatial/animation/overlays.py`

### Task 2.1: Create VideoOverlay Dataclass

- [ ] Add `VideoOverlay` dataclass to [src/neurospatial/animation/overlays.py](src/neurospatial/animation/overlays.py)
  - [ ] `source: str | Path | NDArray[np.uint8]`
  - [ ] `calibration: VideoCalibration | None`
  - [ ] `times: NDArray[np.float64] | None`
  - [ ] `alpha: float = 0.7`
  - [ ] `z_order: Literal["below", "above"] = "below"`
  - [ ] `crop: tuple[int, int, int, int] | None = None`
  - [ ] `downsample: int = 1`
  - [ ] `interp: Literal["linear", "nearest"] = "nearest"`
- [ ] Implement `__post_init__` validation:
  - [ ] File path existence check
  - [ ] Array shape/dtype validation
  - [ ] Alpha bounds check (0.0-1.0)
- [ ] Add comprehensive docstring with examples
- [ ] Add to `__all__`

### Task 2.2: Create VideoData Internal Container

- [ ] Add `VideoData` dataclass to [src/neurospatial/animation/overlays.py](src/neurospatial/animation/overlays.py)
  - [ ] `frame_indices: NDArray[np.int_]`
  - [ ] `reader: VideoReader | NDArray[np.uint8]`
  - [ ] `transform_to_env: Affine2D | None`
  - [ ] `env_bounds: tuple[float, float, float, float]`
  - [ ] `alpha: float`
  - [ ] `z_order: Literal["below", "above"]`
- [ ] Implement `get_frame(anim_frame_idx)` method
  - [ ] Return `None` for index -1 (out of range)
  - [ ] Return RGB frame for valid indices
- [ ] Verify pickle-safety for parallel rendering

---

## Milestone 3: Video I/O

> **Goal**: `VideoReader` loads frames with LRU caching.
>
> **Files**: `src/neurospatial/animation/_video_io.py` (NEW)

### Task 3.1: Create VideoReader Class

- [ ] Create new file [src/neurospatial/animation/_video_io.py](src/neurospatial/animation/_video_io.py)
- [ ] Implement `VideoReader` class:
  - [ ] `__init__(path, cache_size=100, downsample=1, crop=None)`
  - [ ] `__getitem__(frame_idx)` with LRU caching
  - [ ] `get_timestamps()` method (ffprobe or fps fallback)
  - [ ] `__reduce__()` for pickle support (drop cache, keep path)
- [ ] Attributes:
  - [ ] `n_frames: int`
  - [ ] `fps: float`
  - [ ] `frame_size_px: tuple[int, int]` (after crop/downsample)
  - [ ] `original_size_px: tuple[int, int]` (before processing)
  - [ ] `crop_offset_px: tuple[int, int]`
  - [ ] `duration: float`
- [ ] Backend selection: OpenCV (preferred) → imageio (fallback)
- [ ] Write unit tests in [tests/animation/test_video_io.py](tests/animation/test_video_io.py) (NEW):
  - [ ] `test_reader_loads_metadata`
  - [ ] `test_reader_lazy_loading`
  - [ ] `test_reader_lru_cache`
  - [ ] `test_reader_pickle_roundtrip`
  - [ ] `test_reader_timestamps`
- [ ] Verify: `uv run pytest tests/animation/test_video_io.py -v`

**M2 Checkpoint**: VideoReader tests pass

### Task 3.2: Pipeline Integration

- [ ] Create `_find_nearest_indices()` helper in [src/neurospatial/animation/overlays.py](src/neurospatial/animation/overlays.py)
  - [ ] Factor out index-finding from `_interp_nearest()`
  - [ ] Return -1 for out-of-range queries
- [ ] Refactor `_interp_nearest()` to use `_find_nearest_indices()`
- [ ] Add `_validate_video_env()` function:
  - [ ] Check `env.n_dims == 2`
  - [ ] Check `dimension_ranges` exists and is finite
  - [ ] WHAT/WHY/HOW error messages
- [ ] Update `_convert_overlays_to_data()` to handle `VideoOverlay`:
  - [ ] Create VideoReader or wrap array
  - [ ] Compute frame index mapping
  - [ ] Compose crop offset into transform
  - [ ] Warn if no calibration provided
- [ ] Write tests for temporal alignment edge cases

### Task 3.3: Update OverlayData

- [ ] Add `videos: list[VideoData] = field(default_factory=list)` to `OverlayData`
- [ ] Update all backend imports/type hints
- [ ] Verify no `AttributeError` when accessing `overlay_data.videos`
- [ ] Run full test suite: `uv run pytest tests/animation/ -v`

---

## Milestone 4: Backend Rendering

> **Goal**: Video renders correctly in all backends.

### Task 4.1: Napari Backend

> **Note**: Requires manual testing with napari viewer.

- [ ] Add `build_env_to_napari_matrix()` to [src/neurospatial/animation/transforms.py](src/neurospatial/animation/transforms.py)
- [ ] Implement `_add_video_layer()` in [src/neurospatial/animation/backends/napari_backend.py](src/neurospatial/animation/backends/napari_backend.py):
  - [ ] Single-frame initialization (NOT 4D array)
  - [ ] Compute affine from VideoData transform + EnvScale
  - [ ] Register dims callback for frame updates
  - [ ] In-place update: `layer.data[...] = frame`
  - [ ] Handle z_order via layer positioning
- [ ] Implement `_build_video_napari_affine()`:
  - [ ] Use `build_env_to_napari_matrix()` (DRY)
  - [ ] Handle fallback for non-grid environments
- [ ] Implement `_apply_video_fallback_transform()` for non-grid envs
- [ ] Manual test: verify video aligns with field at known corners

**M4 Checkpoint**: Napari displays video with correct spatial alignment

### Task 4.2: Video Export Backend

- [ ] Implement `VideoFrameRenderer` class in [src/neurospatial/animation/backends/video_backend.py](src/neurospatial/animation/backends/video_backend.py):
  - [ ] `__init__(ax, video_data, env)`
  - [ ] `render(ax, frame_idx)` with artist reuse
  - [ ] `_compute_extent()` from transform or env_bounds
- [ ] Implement `_render_video_background()` for parallel rendering:
  - [ ] Fresh artist per frame (parallel-safe)
  - [ ] Handle -1 frame indices (skip rendering)
  - [ ] Correct z-order via `zorder` parameter
- [ ] Update render pipeline to call video rendering:
  - [ ] Before field for `z_order="below"`
  - [ ] After field for `z_order="above"`
- [ ] Write integration tests in [tests/animation/test_video_overlay.py](tests/animation/test_video_overlay.py):
  - [ ] `test_video_composited_in_output`
  - [ ] `test_video_parallel_rendering`
- [ ] Verify: `uv run pytest tests/animation/test_video_overlay.py -v`

**M3 Checkpoint**: Video export produces correct spatial composite

### Task 4.3: Widget and HTML Backends

- [ ] Update [src/neurospatial/animation/backends/widget_backend.py](src/neurospatial/animation/backends/widget_backend.py):
  - [ ] Use `VideoFrameRenderer` for artist reuse
  - [ ] Initialize renderers in `__init__`
  - [ ] Call `renderer.render()` in frame loop
- [ ] Update [src/neurospatial/animation/backends/html_backend.py](src/neurospatial/animation/backends/html_backend.py):
  - [ ] Add video skip with WHAT/WHY/HOW warning
  - [ ] Strip videos from overlay_data before rendering
  - [ ] Continue rendering field + other overlays
- [ ] Write tests:
  - [ ] `test_widget_video_support`
  - [ ] `test_html_skips_video_with_warning`

---

## Milestone 5: Environment Integration

> **Goal**: Convenience function for video calibration.

### Task 5.1: Create calibrate_video() Function

- [ ] Create new file [src/neurospatial/animation/calibration.py](src/neurospatial/animation/calibration.py)
- [ ] Implement `calibrate_video()`:
  - [ ] Accept `video_path`, `env`, and calibration params
  - [ ] Support scale_bar, landmarks_px/landmarks_env, cm_per_px methods
  - [ ] Return `VideoCalibration` object
  - [ ] Validate bounds coverage, warn if mismatch
- [ ] Add to animation package exports
- [ ] Write tests:
  - [ ] `test_calibrate_video_scale_bar`
  - [ ] `test_calibrate_video_landmarks`
  - [ ] `test_calibrate_video_bounds_warning`

---

## Milestone 6: Testing

> **Goal**: Comprehensive test coverage.

### Task 6.1: Calibration Tests

- [ ] Complete all tests from Task 1.1 and 1.2
- [ ] Add edge case tests:
  - [ ] `test_rejects_ill_conditioned_landmarks`
  - [ ] `test_warns_bounds_mismatch`

### Task 6.2: Video I/O Tests

- [ ] Complete all tests from Task 3.1
- [ ] Add fixtures to [tests/conftest.py](tests/conftest.py):
  - [ ] `sample_video` (16x16, 10 frames)
  - [ ] `sample_video_array`
  - [ ] `sample_calibration`

### Task 6.3: Backend Integration Tests

- [ ] Create [tests/animation/test_video_overlay.py](tests/animation/test_video_overlay.py)
- [ ] Napari tests (mark with `@pytest.mark.slow`):
  - [ ] `test_video_layer_added`
  - [ ] `test_video_spatial_alignment`
  - [ ] `test_video_temporal_sync`
- [ ] Export tests:
  - [ ] `test_video_composited_in_output`
  - [ ] `test_video_parallel_rendering`

### Task 6.4: Validation Tests

- [ ] Create [tests/animation/test_video_validation.py](tests/animation/test_video_validation.py)
- [ ] Environment validation:
  - [ ] `test_rejects_1d_environment`
  - [ ] `test_non_grid_2d_environment_works_with_warning`
  - [ ] `test_non_grid_extent_uses_dimension_ranges`
- [ ] Napari-free operation:
  - [ ] `test_video_export_without_napari`
  - [ ] `test_import_without_napari`

### Task 6.5: Add Test Fixtures

- [ ] Add environment fixtures to [tests/conftest.py](tests/conftest.py):
  - [ ] `linearized_env` (1D, for rejection tests)
  - [ ] `polygon_env` (non-grid 2D, for fallback tests)
  - [ ] `masked_env` (grid 2D, for full support tests)
- [ ] Add pytest marker configuration to [pyproject.toml](pyproject.toml):

  ```toml
  [tool.pytest.ini_options]
  markers = ["slow: marks tests as slow"]
  ```

**M5 Checkpoint**: `uv run pytest` all tests pass

---

## Milestone 7: Documentation

> **Goal**: Users can discover and use VideoOverlay.

### Task 7.1: Update CLAUDE.md

- [ ] Add VideoOverlay to Quick Reference section
- [ ] Add code example with calibration
- [ ] Add to "Animation overlays (v0.4.0+)" section
- [ ] Update backend capabilities matrix
- [ ] Add common gotcha for video coordinates

### Task 7.2: Create Example Notebook

- [ ] Create [examples/18_video_overlay.ipynb](examples/18_video_overlay.ipynb)
- [ ] Sections:
  - [ ] Loading and inspecting video metadata
  - [ ] Calibrating with scale bar method
  - [ ] Calibrating with landmark correspondences
  - [ ] Creating VideoOverlay with various options
  - [ ] Animating fields with video background
  - [ ] Exporting synchronized video
  - [ ] Performance tips for large videos
- [ ] Test notebook runs without errors

### Task 7.3: Update Animation Guide

- [ ] Update `docs/guides/animation.md` (or equivalent)
- [ ] Add video integration section:
  - [ ] Coordinate systems explanation
  - [ ] Calibration methods comparison
  - [ ] Backend capabilities (update matrix)
  - [ ] Troubleshooting spatial misalignment

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
