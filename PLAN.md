# Video Overlay Integration Plan

**Feature**: Integrate raw video frames into the animation system with proper pixel↔cm coordinate transformations

**Version Target**: v0.5.0

**Status**: Planning

---

## Files Modified (Summary)

**New Files**:

- `src/neurospatial/animation/_video_io.py` - VideoReader class with LRU caching
- `src/neurospatial/animation/calibration.py` - `calibrate_video()` convenience function
- `tests/animation/test_video_io.py` - VideoReader unit tests
- `tests/animation/test_video_overlay.py` - Backend integration tests
- `tests/animation/test_video_validation.py` - Validation edge case tests
- `examples/18_video_overlay.ipynb` - Tutorial notebook

**Modified Files**:

- `src/neurospatial/transforms.py` - Add `calibrate_from_scale_bar()`, `calibrate_from_landmarks()`, `VideoCalibration`
- `src/neurospatial/animation/overlays.py` - Add `VideoOverlay`, `VideoData`, update `OverlayData`
- `src/neurospatial/animation/transforms.py` - Add `build_env_to_napari_matrix()`
- `src/neurospatial/animation/core.py` - Update type signatures
- `src/neurospatial/animation/__init__.py` - Export new classes
- `src/neurospatial/animation/backends/napari_backend.py` - Video layer rendering
- `src/neurospatial/animation/backends/video_backend.py` - Video compositing
- `src/neurospatial/animation/backends/widget_backend.py` - Video artist reuse
- `src/neurospatial/animation/backends/html_backend.py` - Skip video with warning
- `src/neurospatial/animation/_parallel.py` - Fix artist reuse for multiple images
- `src/neurospatial/environment/visualization.py` - Update type signatures
- `tests/conftest.py` - Add video fixtures
- `pyproject.toml` - Add imageio dependency
- `CLAUDE.md` - Document VideoOverlay API

---

## Common Mistakes to Avoid

Before implementing, be aware of these pitfalls:

1. **DO NOT apply Y-flip twice** - Y-flip happens ONCE in `VideoCalibration.transform_px_to_cm`. Downstream rendering uses `origin="lower"` to preserve this. Double-flip causes upside-down video.

2. **DO NOT preload all video frames** - Behavioral videos can be 50GB+. Use `VideoReader` with streaming access and LRU cache. Never create a 4D array of all frames.

3. **DO NOT use `ax.images[0]` after video is added** - With video layer added first, `ax.images[0]` is the video, not the field. Track artists explicitly by purpose.

4. **DO NOT forget crop offset in calibration** - Calibration is done in ORIGINAL pixel coordinates. If video is cropped, compose crop translation into the transform.

5. **DO NOT mix coordinate conventions** - See "Coordinate Naming Consistency" section. `frame_size_px` is `(width, height)`, NumPy arrays are `(height, width, 3)`.

---

## Overview

This plan adds a `VideoOverlay` type that enables compositing raw behavioral video frames as backgrounds beneath spatial field visualizations. The implementation reuses existing coordinate transformation infrastructure and follows established overlay patterns.

### Goals

1. Display behavioral video synchronized with spatial field animations
2. Support pixel-to-cm calibration via scale bars or landmark correspondences
3. Integrate with all animation backends (napari, video export, widget, html)
4. Handle temporal alignment for videos at different frame rates than fields
5. Lazy-load frames for memory efficiency with large videos

### Non-Goals (Out of Scope)

- Real-time video processing or filtering
- Audio track handling
- Video editing or transcoding
- Multi-camera synchronization

### Constraints

**2D ENVIRONMENTS REQUIRED** (1D and 3D not supported):

Video overlay requires:

- `env.n_dims == 2` (2D spatial coordinates)
- `env.dimension_ranges` exists (for bounding box extent)

**NOT supported** (will raise `ValueError` with WHAT/WHY/HOW message):

- 1D linearized tracks (`env.is_1d == True`)
- 3D environments (`env.n_dims == 3`)

**Supported with fallback** (works but with approximate alignment warning):

- Non-grid 2D layouts (polygon masks, graph layouts, masked grids)
- Custom 2D layouts without `grid_shape`

**Validation placement**: Check in `_convert_overlays_to_data()` before creating VideoData:

```python
def _validate_video_env(env: Environment) -> None:
    """Validate environment supports video overlay."""
    if env.n_dims != 2:
        raise ValueError(
            f"WHAT: VideoOverlay requires 2D environment (got {env.n_dims}D).\n"
            f"WHY: Video frames are 2D images that map to 2D spatial coordinates.\n"
            f"HOW: Use VideoOverlay only with 2D environments."
        )
    # Note: grid_shape is PREFERRED but not REQUIRED
    # Non-grid 2D envs use dimension_ranges fallback with warning
```

### Fallback Behavior for Non-Grid 2D Environments

When `make_env_scale(env)` returns `None` (no `grid_shape`):

1. **Extents**: Use `env.dimension_ranges` bounding box instead of pixel-precise grid alignment
2. **Napari affine**: Use identity/axis-swap transform with translate/scale from dimension_ranges
3. **Warning**: Emit same "fallback transform" warning used by existing overlay system
4. **Behavior**: Video covers full bounding box; may extend beyond occupied bins

```python
# In napari backend - fallback when EnvScale unavailable
if scale is None:
    # Fallback: use dimension_ranges for approximate alignment
    _warn_fallback(suppress=suppress_warning)
    (x_min, x_max), (y_min, y_max) = env.dimension_ranges
    # Set layer translate/scale directly instead of affine matrix
    layer.translate = [y_min, x_min]  # napari (row, col) order
    layer.scale = [
        (y_max - y_min) / frame_height,
        (x_max - x_min) / frame_width,
    ]
    return  # No affine matrix needed

# In video export - fallback extent
if video_data.transform_to_env is None:
    # Use dimension_ranges as extent (bounding box)
    extent = [
        env.dimension_ranges[0][0], env.dimension_ranges[0][1],  # x
        env.dimension_ranges[1][0], env.dimension_ranges[1][1],  # y
    ]
```

### DRY Principle: Reusing Existing Code

**Temporal Alignment**: The existing `_interp_nearest()` in `overlays.py` interpolates **values** with NaN extrapolation. Video needs **indices** with -1 for out-of-range. Solution:

1. Extract `_find_nearest_indices(src_times, query_times) -> NDArray[np.int_]` helper
2. Update `_interp_nearest()` to call this helper, then index into source values
3. Video code calls `_find_nearest_indices()` directly for frame index mapping

**Napari Coordinate Transform**: The existing `animation/transforms.py` has `EnvScale` and `transform_coords_for_napari()`. Video napari affine must reuse this:

1. Add `build_env_to_napari_matrix(scale: EnvScale) -> NDArray[np.float64]` to `animation/transforms.py`
2. Returns the 3x3 homogeneous matrix that transforms env cm → napari pixels
3. Task 4.1 calls this instead of re-deriving the math inline

**Do NOT duplicate**:

- Temporal validation logic (use existing `_validate_temporal_alignment()`)
- Env→napari coordinate math (use new `build_env_to_napari_matrix()`)
- Affine2D composition (use existing `@` operator from `transforms.py`)

---

## Integration Tasks (Pre-requisites)

**CRITICAL**: These integration points must be addressed for VideoOverlay to work with the existing codebase. Complete these before Phase 1.

### I.1: Update Type Signatures End-to-End

The current API only accepts specific overlay types. VideoOverlay must be added to all signatures:

**Files to update**:

- [src/neurospatial/animation/core.py](src/neurospatial/animation/core.py) (line 67-68):

  ```python
  # BEFORE:
  overlays: list[PositionOverlay | BodypartOverlay | HeadDirectionOverlay] | None = None

  # AFTER:
  overlays: list[PositionOverlay | BodypartOverlay | HeadDirectionOverlay | VideoOverlay] | None = None
  ```

- [src/neurospatial/environment/visualization.py](src/neurospatial/environment/visualization.py) (animate_fields method):
  Same type signature update

- [src/neurospatial/animation/overlays.py](src/neurospatial/animation/overlays.py):
  - Add `VideoOverlay` to imports and `__all__`
  - Update `_convert_overlays_to_data()` to handle `isinstance(overlay, VideoOverlay)`
  - Update pickle-safety checks if any

- [src/neurospatial/animation/\_\_init\_\_.py](src/neurospatial/animation/__init__.py):
  - Add `VideoOverlay` and `VideoCalibration` to the `__all__` list
  - Add imports: `from .overlays import VideoOverlay`
  - Add imports: `from ..transforms import VideoCalibration` (or from calibration.py)

### I.2: Fix Artist Reuse Logic for Multiple Images

**Pre-requisite**: Review `_render_worker()` in [src/neurospatial/animation/_parallel.py](src/neurospatial/animation/_parallel.py) to understand artist lifecycle before implementing.

**Problem**: The video backend's fast path assumes `ax.images[0]` is the field image:

```python
# src/neurospatial/animation/_parallel.py (lines 1229-1232)
primary_im: AxesImage | None = ax.images[0] if ax.images else None
```

With a video layer added first (`z_order="below"`), `ax.images[0]` will be the video, breaking field updates.

**Solution**: Track artists explicitly by name/purpose:

```python
# In _parallel.py - replace ax.images[0] lookup with explicit tracking
class FrameArtists:
    """Track named artists for explicit reuse."""
    field_image: AxesImage | None = None
    video_images: list[AxesImage] = field(default_factory=list)

# After env.plot_field():
artists.field_image = ax.images[-1]  # Field is added last

# After _render_video_background():
artists.video_images.append(ax.images[-1])  # Track video images separately

# In update loop:
artists.field_image.set_data(rgb_field)  # Update correct artist
```

**Files to update**:

- [src/neurospatial/animation/_parallel.py](src/neurospatial/animation/_parallel.py): `_render_worker()` function
- [src/neurospatial/animation/backends/widget_backend.py](src/neurospatial/animation/backends/widget_backend.py): If it uses similar pattern
- [src/neurospatial/animation/_overlay_artist_manager.py](src/neurospatial/animation/_overlay_artist_manager.py): May need video artist tracking

### I.3: Reuse Existing Transform Functions (DRY)

**Problem**: transforms.py and calibration.py already have pixel↔cm helpers. New code must wrap, not duplicate.

**Existing functions to reuse**:

- `flip_y(frame_height_px)` in [src/neurospatial/transforms.py:459-476](src/neurospatial/transforms.py#L459-L476) - Y-flip for video origin
- `simple_scale(px_per_cm, offset_px)` in [src/neurospatial/calibration.py:7-68](src/neurospatial/calibration.py#L7-L68) - Scale + offset
- `Affine2D` composition via `@` operator

**Implementation**:

```python
# In transforms.py - calibrate_from_scale_bar wraps existing helpers
def calibrate_from_scale_bar(...) -> Affine2D:
    # Compute scale from endpoints
    px_distance = np.linalg.norm(np.array(p2_px) - np.array(p1_px))
    cm_per_px = known_length_cm / px_distance

    # Compose: Y-flip @ scale (reuse existing flip_y)
    _, height = frame_size_px
    return flip_y(height) @ scale_2d(cm_per_px, cm_per_px)
```

### I.4: Match Napari Field Orientation

**Problem**: `field_to_rgb_for_napari()` applies transpose + flip to field images:

```python
# src/neurospatial/animation/rendering.py (lines 440-444, 447-450)
transposed = np.transpose(full_rgb, (1, 0, 2))  # (n_x, n_y, 3) → (n_y, n_x, 3)
return np.flip(transposed, axis=0)  # Flip vertically for napari
```

Video affine must produce equivalent orientation. The `build_env_to_napari_matrix()` must encode the same Y-flip.

**Validation test**:

```python
def test_video_field_alignment():
    """Video and field layers align at known landmarks."""
    # Create env with known corners
    # Render field with corner bins highlighted
    # Place video landmarks at same corners
    # Assert napari layer pixels overlap at corners
```

### I.5: Add imageio Dependency

**Problem**: pyproject.toml has opencv as optional but no imageio. Test fixtures need a codec stack.

**Solution**: Add imageio to dev dependencies (or core if required for VideoReader fallback):

```toml
# pyproject.toml
[project.optional-dependencies]
video = [
    "opencv-python>=4.11.0.86",
    "imageio>=2.35.0",
    "imageio-ffmpeg>=0.5.1",  # For ffprobe fallback
]

# Or in dev dependencies for test fixtures only:
dev = [
    ...
    "imageio>=2.35.0",
]
```

### I.6: Harden dimension_ranges Validation

**Problem**: `EnvScale.from_env()` returns None if `layout.grid_shape` is missing, even when `dimension_ranges` exists. Some environments may not populate `dimension_ranges` at all.

**Current code** ([animation/transforms.py:123-130](src/neurospatial/animation/transforms.py#L123-L130)):

```python
if (
    env is None
    or not hasattr(env, "dimension_ranges")
    or not hasattr(env, "layout")
    or not hasattr(env.layout, "grid_shape")
):
    return None
```

**Solution**: Add early validation in `_convert_overlays_to_data()` for video overlays:

```python
def _validate_video_env(env: Environment) -> None:
    """Validate environment supports video overlay."""
    if env.n_dims != 2:
        raise ValueError(...)

    # Ensure dimension_ranges exists and is valid
    if not hasattr(env, "dimension_ranges") or env.dimension_ranges is None:
        raise ValueError(
            f"WHAT: VideoOverlay requires environment with dimension_ranges.\n"
            f"WHY: Dimension ranges define the spatial extent for video alignment.\n"
            f"HOW: Use Environment.from_samples() or ensure dimension_ranges is set."
        )

    # Validate ranges are finite
    for i, (lo, hi) in enumerate(env.dimension_ranges):
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(
                f"WHAT: dimension_ranges[{i}] has non-finite values ({lo}, {hi}).\n"
                f"WHY: Video alignment requires finite spatial bounds.\n"
                f"HOW: Check environment creation parameters."
            )
```

---

## Architecture Summary

```
User API                    Internal Pipeline                Backend Rendering
──────────                  ─────────────────                ─────────────────
VideoOverlay    ──────>    VideoData (aligned)    ──────>    Napari: viewer.add_image()
  • path/frames              • RGB frames                     Video: ax.imshow(extent=...)
  • pixel_scale_cm           • transform matrix               Widget: same as video
  • transform                                                 HTML: inline base64 frames
  • times
```

---

## Implementation Guidelines

> **IMPORTANT**: Before implementing any task involving coordinates, review the [Coordinate Naming Consistency](#coordinate-naming-consistency) section below. Mixing `(width, height)` vs `(height, width)` is a common source of bugs.

### Napari Best Practices

**Layer update strategy** (avoid per-frame `layer.data = ...` assignment):

```python
# BAD: Slow - reallocates array every frame
def _update_video_frame(event):
    layer.data = video_data.reader[frame_idx]  # Creates new array

# GOOD: Fast - update in place with pre-allocated buffer
def _update_video_frame(event):
    frame = video_data.reader[frame_idx]
    layer.data[...] = frame  # In-place update, no reallocation
```

**Immutable layer shape**: Initialize with a frame matching ALL subsequent shapes. VideoReader must ensure consistent `frame_size_px` (width, height) after crop/downsample.

**Affine conventions**:

- Napari expects row-major, float64 arrays
- Document flip conventions explicitly in docstrings
- Test affine with known corner points before full integration

**Thread safety**: Video decoding may happen off main thread. Ensure UI updates (layer.data assignment) occur on Qt event loop. Consider `viewer.window._qt_window.run_idle()` for thread-safe updates.

### Video Backend Optimization

**Artist reuse** (avoid re-creating imshow per frame):

```python
# In _render_frame_init (called once):
self._video_artist = ax.imshow(
    initial_frame, extent=extent, origin="lower", alpha=alpha
)

# In _render_frame_update (called per frame):
frame = video_data.reader[video_frame_idx]
self._video_artist.set_data(frame)  # Fast - reuses artist
```

### HTML Backend Limitations

**Explicit constraints**:

- Base64 embedding bloats HTML file size (~1.3x video size)
- Max recommended: 100 frames at 320x240 (~10MB HTML)
- For larger videos: emit warning and auto-skip video layer

```python
# In html_backend.py
if overlay_data.videos:
    total_frames = sum(len(v.frame_indices) for v in overlay_data.videos)
    if total_frames > 100:
        warnings.warn(
            f"HTML backend skipping VideoOverlay ({total_frames} frames). "
            "Base64 embedding would create oversized HTML. "
            "Use video or napari backend for video support.",
            UserWarning,
        )
        overlay_data = dataclasses.replace(overlay_data, videos=[])
```

### Validation Requirements

**Bounds coverage check**: Warn when environment bounds fall outside calibrated video extent:

```python
def _validate_calibration_coverage(
    calibration: VideoCalibration,
    env: Environment,
) -> None:
    """Warn if env bounds exceed calibrated video coverage."""
    # Transform video corners to env coords
    w, h = calibration.frame_size_px
    corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    corners_cm = calibration.transform_px_to_cm(corners_px)

    video_bounds = (
        corners_cm[:, 0].min(), corners_cm[:, 0].max(),
        corners_cm[:, 1].min(), corners_cm[:, 1].max(),
    )
    env_bounds = (
        env.dimension_ranges[0][0], env.dimension_ranges[0][1],
        env.dimension_ranges[1][0], env.dimension_ranges[1][1],
    )

    if (env_bounds[0] < video_bounds[0] or env_bounds[1] > video_bounds[1] or
        env_bounds[2] < video_bounds[2] or env_bounds[3] > video_bounds[3]):
        warnings.warn(
            f"Environment bounds {env_bounds} extend beyond calibrated video "
            f"coverage {video_bounds}. Regions outside video will show blank.",
            UserWarning,
        )
```

**Landmark conditioning**: Reject ill-conditioned transforms:

```python
# In calibrate_from_landmarks()
transform = estimate_transform(landmarks_px, landmarks_cm, kind=kind)
cond = np.linalg.cond(transform.A[:2, :2])
if cond > 1e6:
    raise ValueError(
        f"WHAT: Landmark calibration produced ill-conditioned transform (cond={cond:.1e}).\n"
        f"WHY: Landmarks may be collinear or too close together.\n"
        f"HOW: Use landmarks that span the full video frame with good spread."
    )
```

### Naming Conventions

**frame_size ordering**: Always `(width, height)` in pixels, documented explicitly:

- `VideoReader.frame_size_px: tuple[int, int]` - `(width, height)` after crop/downsample
- `VideoReader.original_size_px: tuple[int, int]` - `(width, height)` before processing
- `VideoCalibration.frame_size_px: tuple[int, int]` - `(width, height)` for calibration

### Coordinate Naming Consistency

**CRITICAL**: Video code mixes multiple coordinate conventions. To prevent confusion and transpose bugs, follow these conventions EXACTLY and document them in every function that handles coordinates:

| Context | Convention | Example |
|---------|------------|---------|
| `frame_size_px` | `(width, height)` | `(640, 480)` = 640 columns × 480 rows |
| NumPy arrays from VideoReader | `(height, width, 3)` | Shape `(480, 640, 3)` for RGB |
| Calibration landmarks | `(x_px, y_px)` = `(column, row)` | `[[100, 50], [200, 50]]` |
| Environment coordinates | `(x_cm, y_cm)` | Same order as landmarks |

**Why this matters**: It's easy to accidentally swap width/height or x/y. The `(width, height)` tuple convention for sizes matches PIL/OpenCV, while NumPy arrays are always `(height, width, channels)`.

**Required docstring text**: Every function handling pixel coordinates MUST include this clarification:

```text
Notes
-----
Pixel coordinates use (x_px, y_px) = (column, row) ordering in image space.
NumPy arrays from VideoReader have shape (height, width, 3) in RGB format.
The frame_size_px tuple uses (width, height) ordering to match video metadata.
```

**Example of correct documentation**:

```python
def calibrate_from_landmarks(
    landmarks_px: NDArray[np.float64],  # Shape (n_points, 2) as (x_px, y_px)
    landmarks_cm: NDArray[np.float64],  # Shape (n_points, 2) as (x_cm, y_cm)
    frame_size_px: tuple[int, int],     # (width, height) in pixels
) -> Affine2D:
    """Build px→cm transform from corresponding landmark pairs.

    Parameters
    ----------
    landmarks_px : ndarray of shape (n_points, 2)
        Landmark coordinates in video pixels as (x_px, y_px) = (column, row).
    landmarks_cm : ndarray of shape (n_points, 2)
        Corresponding coordinates in environment space as (x_cm, y_cm).
    frame_size_px : tuple[int, int]
        Video frame size as (width, height) in pixels.

    Notes
    -----
    Pixel coordinates use (x_px, y_px) = (column, row) ordering in image space.
    NumPy arrays from VideoReader have shape (height, width, 3) in RGB format.
    The frame_size_px tuple uses (width, height) ordering to match video metadata.
    """
```

---

## Phase 1: Calibration and Transform Infrastructure

**Objective**: Extend transform utilities to support pixel↔cm conversion with affine calibration.

### Task 1.1: Add Calibration Helpers to `transforms.py`

**File**: [src/neurospatial/transforms.py](src/neurospatial/transforms.py)

Add functions for building pixel↔cm transforms from common calibration methods:

```python
def calibrate_from_scale_bar(
    p1_px: tuple[float, float],
    p2_px: tuple[float, float],
    known_length_cm: float,
    frame_size_px: tuple[int, int],
) -> Affine2D:
    """Build px→cm transform from a scale bar of known length.

    Parameters
    ----------
    p1_px, p2_px : tuple[float, float]
        Two endpoints of the scale bar in pixel coordinates.
    known_length_cm : float
        Real-world length of the scale bar in centimeters.
    frame_size_px : tuple[int, int]
        Video frame size as (width, height) in pixels.

    Returns
    -------
    Affine2D
        Transform that converts pixel coords to cm coords with Y-flip.
    """

def calibrate_from_landmarks(
    landmarks_px: NDArray[np.float64],
    landmarks_cm: NDArray[np.float64],
    frame_size_px: tuple[int, int],
    kind: str = "similarity",
) -> Affine2D:
    """Build px→cm transform from corresponding landmark pairs.

    Uses estimate_transform internally with Y-flip composition.
    """
```

**Acceptance Criteria**:

- [ ] `calibrate_from_scale_bar()` computes correct scale from pixel distance
- [ ] `calibrate_from_landmarks()` uses `estimate_transform()` with Y-flip
- [ ] Both return inverse-capable `Affine2D` for cm→px conversion
- [ ] Unit tests verify round-trip accuracy with scale-dependent tolerance:
  `atol = max(1e-4, 1e-6 * arena_extent_cm)` where `arena_extent_cm = max(x_range, y_range)`

---

### Task 1.2: Add Frame-Aware Coordinate Conversion

**File**: [src/neurospatial/transforms.py](src/neurospatial/transforms.py)

Add a dataclass for bundling video calibration metadata:

```python
@dataclass(frozen=True)
class VideoCalibration:
    """Stores video→environment coordinate calibration.

    The transform_px_to_cm handles BOTH:
    1. Y-axis flip (video origin top-left → env origin bottom-left)
    2. Scaling from pixels to centimeters

    This is the SINGLE location for Y-flip. Downstream rendering uses
    origin="lower" in matplotlib imshow to preserve this convention.

    Attributes
    ----------
    transform_px_to_cm : Affine2D
        Pixel → centimeter transform (includes Y-flip via flip_y composition).
    frame_size_px : tuple[int, int]
        Video frame size (width, height) in pixels.
    """
    transform_px_to_cm: Affine2D
    frame_size_px: tuple[int, int]

    @property
    def transform_cm_to_px(self) -> Affine2D:
        """Inverse transform from cm → pixel coordinates."""
        return self.transform_px_to_cm.inverse()

    @property
    def cm_per_px(self) -> float:
        """Approximate cm-per-pixel scale (assumes uniform scaling)."""
        # Extract from transform matrix diagonal
```

**Y-Flip Convention (IMPORTANT)**:

- Y-flip happens ONCE in `transform_px_to_cm` (via `flip_y @ scale_2d`)
- All downstream code (imshow, napari) uses `origin="lower"` to preserve this
- Do NOT double-flip by also using `origin="upper"`

**Acceptance Criteria**:

- [ ] `VideoCalibration` stores transform + metadata
- [ ] Y-flip happens exactly once (in calibration transform)
- [ ] Inverse transform computed lazily/cached
- [ ] Serializable to dict for JSON persistence

---

## Phase 2: Video Data Model

**Objective**: Define `VideoOverlay` dataclass and internal `VideoData` container.

### Task 2.1: Create `VideoOverlay` Public Dataclass

**File**: [src/neurospatial/animation/overlays.py](src/neurospatial/animation/overlays.py)

```python
@dataclass
class VideoOverlay:
    """Background video layer for animation.

    Displays raw behavioral video frames beneath spatial field visualizations.
    Supports temporal alignment, coordinate calibration, and lazy loading.

    Parameters
    ----------
    source : str | Path | NDArray[np.uint8]
        Video source: path to video file, or pre-loaded array of shape
        (n_frames, height, width, 3) as RGB uint8.
    calibration : VideoCalibration | None
        Pixel↔cm coordinate calibration. If None, video is displayed
        without spatial alignment (covers full environment bounds).
    times : NDArray[np.float64] | None
        Timestamps for each video frame in seconds. If None with file source,
        synthesized from video fps. Required for pre-loaded arrays.
    alpha : float
        Opacity of video layer (0.0 = transparent, 1.0 = opaque). Default 0.7.
    z_order : Literal["below", "above"]
        Draw order relative to field layer. Default "below".
    crop : tuple[int, int, int, int] | None
        Crop region as (x, y, width, height) in pixels. Applied before transform.
    downsample : int
        Spatial downsampling factor. 2 = half resolution. Default 1 (no downsampling).

    Examples
    --------
    >>> from neurospatial.animation import VideoOverlay
    >>> from neurospatial.transforms import calibrate_from_scale_bar
    >>>
    >>> # Calibrate from known scale bar
    >>> calibration = calibrate_from_scale_bar(
    ...     p1_px=(100, 200), p2_px=(300, 200),
    ...     known_length_cm=50.0,
    ...     frame_size_px=(640, 480),
    ... )
    >>>
    >>> # Create video overlay
    >>> video = VideoOverlay(
    ...     source="session.mp4",
    ...     calibration=calibration,
    ...     alpha=0.5,
    ... )
    >>>
    >>> env.animate_fields(fields, overlays=[video, position_overlay])
    """
    source: str | Path | NDArray[np.uint8]
    calibration: VideoCalibration | None = None
    times: NDArray[np.float64] | None = None
    alpha: float = 0.7
    z_order: Literal["below", "above"] = "below"
    crop: tuple[int, int, int, int] | None = None
    downsample: int = 1
    interp: Literal["linear", "nearest"] = "nearest"  # Frame selection
```

**Acceptance Criteria**:

- [ ] Dataclass validates parameters in `__post_init__`
- [ ] File paths checked for existence
- [ ] Array sources validated for shape/dtype
- [ ] Docstring includes complete examples

---

### Task 2.2: Create `VideoData` Internal Container

**File**: [src/neurospatial/animation/overlays.py](src/neurospatial/animation/overlays.py)

```python
@dataclass
class VideoData:
    """Internal container for video frames aligned to animation timeline.

    Created by conversion pipeline from VideoOverlay. Holds frame indices
    or lazy reader for memory efficiency.

    Parameters
    ----------
    frame_indices : NDArray[np.int_]
        Video frame indices aligned to animation frames. Shape (n_anim_frames,).
        Index -1 indicates no video frame available (extrapolation).
    reader : VideoReader | NDArray[np.uint8]
        Lazy reader or pre-loaded array for frame access.
    transform_to_env : Affine2D | None
        Pixel → environment cm transform.
    env_bounds : tuple[float, float, float, float]
        Environment bounds (xmin, xmax, ymin, ymax) for extent calculation.
    alpha : float
        Layer opacity.
    z_order : str
        Draw order ("below" or "above").
    """
    frame_indices: NDArray[np.int_]
    reader: VideoReader | NDArray[np.uint8]
    transform_to_env: Affine2D | None
    env_bounds: tuple[float, float, float, float]
    alpha: float
    z_order: Literal["below", "above"]

    def get_frame(self, anim_frame_idx: int) -> NDArray[np.uint8] | None:
        """Get RGB frame for animation frame index, or None if unavailable."""
```

**Acceptance Criteria**:

- [ ] Frame access via index mapping (no temporal recomputation)
- [ ] Returns None for extrapolated frames (index -1)
- [ ] Pickle-safe for parallel video export

---

## Phase 3: Video I/O and Caching

**Objective**: Implement lazy video reading with LRU caching.

### Task 3.1: Create `VideoReader` Class

**File**: [src/neurospatial/animation/_video_io.py](src/neurospatial/animation/_video_io.py) (new file)

```python
class VideoReader:
    """Lazy video frame reader with LRU caching.

    Supports OpenCV (cv2) and imageio backends with automatic fallback.
    Caches recently accessed frames to avoid re-decoding.

    Parameters
    ----------
    path : str | Path
        Path to video file.
    cache_size : int
        Maximum number of frames to cache. Default 100.
    downsample : int
        Spatial downsampling factor. Default 1.
    crop : tuple[int, int, int, int] | None
        Crop region (x, y, width, height). Default None.

    Attributes
    ----------
    n_frames : int
        Total frame count.
    fps : float
        Video frame rate.
    frame_size_px : tuple[int, int]
        Frame size (width, height) in pixels AFTER crop/downsample.
    original_size_px : tuple[int, int]
        Original frame size (width, height) BEFORE crop/downsample.
    crop_offset_px : tuple[int, int]
        Crop origin (x, y) in original pixel coordinates. (0, 0) if no crop.
    duration : float
        Video duration in seconds.
    """

    def __init__(
        self,
        path: str | Path,
        cache_size: int = 100,
        downsample: int = 1,
        crop: tuple[int, int, int, int] | None = None,
    ) -> None: ...

    def __getitem__(self, frame_idx: int) -> NDArray[np.uint8]:
        """Get frame by index with caching. Returns RGB uint8 array."""

    def get_timestamps(self) -> NDArray[np.float64]:
        """Get frame timestamps from video metadata or synthesize from fps.

        Uses ffprobe if available for accurate timestamps.
        Falls back to np.arange(n_frames) / fps with warning for long videos.
        """

    def __reduce__(self) -> tuple:
        """Make pickle-able by storing path + params (drops cache)."""
```

**Timestamp Fallback (Design Decision #3)**:

- If `ffprobe` is available: use it for high-quality per-frame timestamps
- If not available: synthesize from `np.arange(n_frames) / fps`
- Emit warning for videos >10min without ffprobe (potential drift)

**Crop Handling (IMPORTANT)**:

Calibration is done in ORIGINAL pixel coordinates (before crop). When crop is applied:

1. `VideoReader.crop_offset_px` stores the crop origin `(crop_x, crop_y)`
2. The calibration transform must be composed with a translation:

   ```python
   # In VideoData construction:
   if reader.crop_offset_px != (0, 0):
       crop_x, crop_y = reader.crop_offset_px
       # Translate cropped coords back to original coords before calibration
       crop_translation = translate(crop_x, crop_y)
       effective_transform = calibration.transform_px_to_cm @ crop_translation
   else:
       effective_transform = calibration.transform_px_to_cm
   ```

3. This ensures pixel (0,0) in the cropped frame maps to the correct world point

**Acceptance Criteria**:

- [ ] Lazy loading: frames decoded on access only
- [ ] LRU cache with configurable size
- [ ] Automatic backend selection (cv2 preferred, imageio fallback)
- [ ] Pickle-able (reconstitutes reader, drops cache)
- [ ] `ffprobe` used for accurate timestamps when available
- [ ] `crop_offset_px` exposed for calibration composition
- [ ] `frame_size_px` and `original_size_px` clearly distinguished

---

### Task 3.2: Add Video Conversion to Pipeline

**File**: [src/neurospatial/animation/overlays.py](src/neurospatial/animation/overlays.py)

Extend `_convert_overlays_to_data()` to handle `VideoOverlay`:

```python
def _convert_overlays_to_data(...) -> OverlayData:
    # ... existing code ...

    video_data_list: list[VideoData] = []

    for overlay in overlays:
        if isinstance(overlay, VideoOverlay):
            # 1. Create or wrap reader
            if isinstance(overlay.source, (str, Path)):
                reader = VideoReader(
                    overlay.source,
                    downsample=overlay.downsample,
                    crop=overlay.crop,
                )
                video_times = overlay.times or reader.get_timestamps()
            else:
                reader = overlay.source  # Pre-loaded array
                if overlay.times is None:
                    raise ValueError("times required for pre-loaded video arrays")
                video_times = overlay.times

            # 2. Validate temporal coverage
            _validate_temporal_alignment(video_times, frame_times, name="VideoOverlay")

            # 3. Compute frame index mapping using SHARED helper (see DRY section)
            # Returns -1 for out-of-range times (no extrapolation)
            frame_indices = _find_nearest_indices(video_times, frame_times)

            # 4. Build transform WITH CROP OFFSET (see Task 3.1 for rationale)
            if overlay.calibration is not None:
                base_transform = overlay.calibration.transform_px_to_cm
                # Compose with crop offset translation if video was cropped
                if isinstance(reader, VideoReader) and reader.crop_offset_px != (0, 0):
                    crop_x, crop_y = reader.crop_offset_px
                    # cropped_px → original_px → env_cm
                    crop_translation = translate(crop_x, crop_y)
                    transform = base_transform @ crop_translation
                else:
                    transform = base_transform
            else:
                # No calibration: stretch to fill env bounds (Design Decision #1)
                transform = None
                warnings.warn(
                    "VideoOverlay has no calibration. Video will be stretched to fill "
                    "environment bounds, which may cause spatial misalignment. "
                    "Use calibrate_video() or VideoCalibration for accurate positioning.",
                    UserWarning,
                    stacklevel=2,
                )

            env_bounds = (
                env.dimension_ranges[0][0], env.dimension_ranges[0][1],
                env.dimension_ranges[1][0], env.dimension_ranges[1][1],
            )

            video_data = VideoData(
                frame_indices=frame_indices,
                reader=reader,
                transform_to_env=transform,  # Now includes crop offset!
                env_bounds=env_bounds,
                alpha=overlay.alpha,
                z_order=overlay.z_order,
            )
            video_data_list.append(video_data)

    return OverlayData(
        ...,
        videos=video_data_list,  # New field
    )
```

**Temporal Alignment Specification (DRY)**:

The video pipeline uses `_find_nearest_indices()`, a **NEW helper created in this task** by factoring out the index-finding logic from `_interp_nearest()`. This helper does not exist yet - it is created here:

```python
def _find_nearest_indices(
    src_times: NDArray[np.float64],
    query_times: NDArray[np.float64],
) -> NDArray[np.int_]:
    """Find nearest source index for each query time.

    This is the SHARED helper used by both `_interp_nearest()` and video
    frame alignment. Returns -1 for out-of-range queries (no extrapolation).

    Parameters
    ----------
    src_times : array of shape (n_src,)
        Source timestamps (must be monotonically increasing).
    query_times : array of shape (n_query,)
        Query timestamps to map.

    Returns
    -------
    indices : array of shape (n_query,)
        Index into src_times for each query time.
        Returns -1 for queries outside [src_times.min(), src_times.max()].

    Behavior
    --------
    - IN RANGE: Returns index of nearest source time
    - BEFORE start: Returns -1 (out-of-range)
    - AFTER end: Returns -1 (out-of-range)
    - NaN in query_times: Returns -1
    """
    # Implementation: np.searchsorted + boundary checks
    # See existing _interp_nearest for similar logic
```

**Refactoring requirement**: Update `_interp_nearest()` to call `_find_nearest_indices()` internally:

```python
def _interp_nearest(t_src, x_src, t_frame):
    """Nearest-neighbor interpolation (existing function, refactored)."""
    indices = _find_nearest_indices(t_src, t_frame)
    result = np.where(indices >= 0, x_src[np.clip(indices, 0, len(x_src)-1)], np.nan)
    return result
```

**Gap handling rules**:

- Animation frames before `video_times.min()` → index -1 (no video)
- Animation frames after `video_times.max()` → index -1 (no video)
- Backends render nothing for index -1 (transparent/skip)
- No clamping to edge frames (would cause frozen video at boundaries)

**Acceptance Criteria**:

- [ ] `_find_nearest_indices()` factored out as shared helper (DRY)
- [ ] `_interp_nearest()` refactored to call `_find_nearest_indices()` internally
- [ ] Video frames aligned to animation timeline via index mapping
- [ ] Temporal validation with WHAT/WHY/HOW errors
- [ ] Both file paths and pre-loaded arrays supported
- [ ] Out-of-range animation times return -1 (no extrapolation)
- [ ] Backends handle -1 gracefully (skip rendering)
- [ ] Crop offset composed into `VideoData.transform_to_env`

---

### Task 3.3: Update `OverlayData` and Downstream Consumers

**Files to update**:

- [src/neurospatial/animation/overlays.py](src/neurospatial/animation/overlays.py) - Add `videos` field to `OverlayData`
- [src/neurospatial/animation/backends/napari_backend.py](src/neurospatial/animation/backends/napari_backend.py) - Handle `overlay_data.videos`
- [src/neurospatial/animation/backends/video_backend.py](src/neurospatial/animation/backends/video_backend.py) - Handle `overlay_data.videos`
- [src/neurospatial/animation/backends/widget_backend.py](src/neurospatial/animation/backends/widget_backend.py) - Handle `overlay_data.videos`
- [src/neurospatial/animation/backends/html_backend.py](src/neurospatial/animation/backends/html_backend.py) - Handle `overlay_data.videos` (with warning)

```python
# In overlays.py - update OverlayData dataclass
@dataclass
class OverlayData:
    positions: list[PositionData] = field(default_factory=list)
    bodypart_sets: list[BodypartData] = field(default_factory=list)
    head_directions: list[HeadDirectionData] = field(default_factory=list)
    regions: list[str] | dict[int, list[str]] | None = None
    videos: list[VideoData] = field(default_factory=list)  # NEW FIELD
```

**Backend consumer pattern**:

```python
# Each backend must check for videos and handle appropriately
def _render_frame(overlay_data: OverlayData, ...):
    # Render video FIRST (z_order="below") or LAST (z_order="above")
    for video_data in overlay_data.videos:
        if video_data.z_order == "below":
            _render_video_background(...)

    # ... render field and other overlays ...

    for video_data in overlay_data.videos:
        if video_data.z_order == "above":
            _render_video_background(...)
```

**Acceptance Criteria**:

- [ ] `OverlayData.videos` field added with proper type hints
- [ ] All 4 backends updated to iterate over `overlay_data.videos`
- [ ] No AttributeError when `videos` field is accessed
- [ ] Existing tests pass (videos defaults to empty list)

---

## Phase 4: Backend Rendering

**Objective**: Implement video rendering for each animation backend.

### Task 4.1: Napari Backend

**File**: [src/neurospatial/animation/backends/napari_backend.py](src/neurospatial/animation/backends/napari_backend.py)

Add video as a napari `Image` layer with **streaming frame access** (NOT preloaded 4D array):

```python
def _add_video_layer(
    viewer: napari.Viewer,
    video_data: VideoData,
    env: Environment,
) -> napari.layers.Image:
    """Add video frames as streaming Image layer.

    IMPORTANT: Do NOT preload all frames into a 4D array - this causes RAM
    blow-up for typical behavioral videos (e.g., 30min @ 30fps @ 640x480 = 50GB).

    Instead, use the existing per-frame caching infrastructure:
    - VideoReader provides LRU-cached frame access
    - Hook into napari's dims slider callback to fetch frames on demand
    - Reuse the existing overlay frame update pattern
    """
    # 1. Get initial frame for layer creation
    initial_frame_idx = video_data.frame_indices[0]
    if initial_frame_idx >= 0:
        initial_frame = video_data.reader[initial_frame_idx]
    else:
        # Create blank frame matching video dimensions
        w, h = video_data.reader.frame_size_px  # (width, height) from VideoReader
        initial_frame = np.zeros((h, w, 3), dtype=np.uint8)

    # 2. Compute affine transform for spatial alignment
    # See "Y-Flip Composition" section below for detailed derivation
    affine = _build_video_napari_affine(video_data, env)

    # 3. Add layer with single frame (will update on playback)
    layer = viewer.add_image(
        data=initial_frame,
        name="video",
        opacity=video_data.alpha,
        rgb=True,
        interpolation="linear",
        affine=affine,
    )

    # 4. Set z-order by moving layer position
    if video_data.z_order == "below":
        viewer.layers.move(viewer.layers.index(layer), 0)

    # 5. Register callback to update frame on slider change
    # Uses IN-PLACE update for performance (see Implementation Guidelines)
    def _update_video_frame(event):
        frame_idx = viewer.dims.current_step[0]
        video_frame_idx = video_data.frame_indices[frame_idx]
        if video_frame_idx >= 0:
            frame = video_data.reader[video_frame_idx]
            layer.data[...] = frame  # In-place update, avoids reallocation
        # For -1 (no video), leave previous frame or clear to black

    viewer.dims.events.current_step.connect(_update_video_frame)

    return layer
```

**Performance notes**:

- `layer.data[...] = frame` is faster than `layer.data = frame` (no reallocation)
- Initial frame shape must match all subsequent frames (VideoReader ensures this)
- Consider throttling callback for very high frame rates (>60 fps)

**Why streaming over preloading**:

- Behavioral videos are typically 30+ minutes at 30fps = 50,000+ frames
- At 640×480×3 bytes per frame = ~1MB each → 50GB for full video
- VideoReader's LRU cache (default 100 frames) keeps memory bounded
- Sequential playback has excellent cache hit rate

**Y-Flip Composition for Napari (CRITICAL)**:

The full transform chain is: `video_px → env_cm → napari_px`

```text
┌─────────────┐     transform_px_to_cm      ┌─────────────┐     _transform_coords_for_napari     ┌─────────────┐
│  video_px   │  ─────────────────────────> │   env_cm    │  ─────────────────────────────────> │  napari_px  │
│ (row, col)  │    (includes Y-flip)        │   (x, y)    │        (includes Y-flip)            │ (row, col)  │
└─────────────┘                             └─────────────┘                                     └─────────────┘
```

**Where Y-flips occur**:

1. `transform_px_to_cm` (VideoCalibration): flips video Y (top-left origin → bottom-left)
2. `_transform_coords_for_napari` (EnvScale): flips env Y for napari display

**Affine construction (DRY)**:

First, add helper to `animation/transforms.py`:

```python
# In src/neurospatial/animation/transforms.py

def build_env_to_napari_matrix(scale: EnvScale) -> NDArray[np.float64]:
    """Build 3x3 homogeneous matrix for env_cm → napari_px transform.

    This matrix encodes the same transformation as transform_coords_for_napari()
    but in matrix form for use with napari's affine parameter.

    Parameters
    ----------
    scale : EnvScale
        Pre-computed scale factors from environment.

    Returns
    -------
    T : ndarray of shape (3, 3)
        Homogeneous transformation matrix.

    Notes
    -----
    From EnvScale: col = (x - x_min) * x_scale
                   row = (n_y - 1) - (y - y_min) * y_scale

    Matrix form: [row]   [0, -y_scale, (n_y-1) + y_min*y_scale] [x]
                 [col] = [x_scale, 0,  -x_min*x_scale         ] [y]
                 [1  ]   [0,       0,   1                     ] [1]
    """
    return np.array([
        [0, -scale.y_scale, (scale.n_y - 1) + scale.y_min * scale.y_scale],
        [scale.x_scale, 0, -scale.x_min * scale.x_scale],
        [0, 0, 1],
    ])
```

Then use in napari backend:

```python
def _build_video_napari_affine(
    video_data: VideoData,
    env: Environment,
    suppress_warning: bool = False,
) -> NDArray[np.float64] | None:
    """Build napari affine transform for video layer.

    Composes: video_px → env_cm → napari_px
    Uses shared helper from animation/transforms.py (DRY).

    For non-grid 2D environments (no grid_shape), returns None and caller
    should use layer.translate/scale fallback with dimension_ranges.
    """
    if video_data.transform_to_env is None:
        return None  # No calibration, use environment bounds stretching

    scale = make_env_scale(env)
    if scale is None:
        # Non-grid 2D environment: use fallback path with warning
        _warn_fallback(suppress=suppress_warning)
        return None  # Caller uses layer.translate/scale instead

    # Step 1: video_px → env_cm (from VideoData, includes Y-flip + crop offset)
    T_video_to_env = video_data.transform_to_env.A  # 3x3 matrix

    # Step 2: env_cm → napari_px (REUSE existing helper)
    T_env_to_napari = build_env_to_napari_matrix(scale)

    # Compose: napari_px = T_env_to_napari @ T_video_to_env @ video_px
    return T_env_to_napari @ T_video_to_env


def _apply_video_fallback_transform(
    layer: napari.layers.Image,
    video_data: VideoData,
    env: Environment,
) -> None:
    """Apply fallback transform for non-grid 2D environments.

    Uses layer.translate/scale with dimension_ranges instead of affine matrix.
    Called when make_env_scale(env) returns None.
    """
    (x_min, x_max), (y_min, y_max) = env.dimension_ranges
    h, w = video_data.reader.frame_size_px[1], video_data.reader.frame_size_px[0]

    # Map video to fill environment bounding box
    # napari uses (row, col) = (y, x) ordering
    layer.translate = [y_min, x_min]
    layer.scale = [
        (y_max - y_min) / h,  # row scale
        (x_max - x_min) / w,  # col scale
    ]
```

**Guard against double-flip**:

- Do NOT apply additional Y-flip in napari layer settings
- Do NOT use `origin="upper"` anywhere in the video path
- The two Y-flips (calibration + napari) compose correctly to show video right-side-up

**Acceptance Criteria**:

- [ ] `build_env_to_napari_matrix()` added to `animation/transforms.py` (DRY)
- [ ] Video displayed as napari Image layer (single frame, updated on playback)
- [ ] Correct spatial alignment with field via affine transform
- [ ] z_order respected (below = index 0)
- [ ] Playback synchronized with field animation via dims callback
- [ ] Memory bounded by VideoReader cache size (NOT by video length)

---

### Task 4.2: Video Export Backend

**File**: [src/neurospatial/animation/backends/video_backend.py](src/neurospatial/animation/backends/video_backend.py)

Add video compositing to matplotlib figure renderer with **artist reuse** for performance:

```python
# For sequential rendering (widget backend), use artist reuse pattern:
class VideoFrameRenderer:
    """Manages video artist for efficient sequential frame updates."""

    def __init__(self, ax: plt.Axes, video_data: VideoData, env: Environment):
        self.video_data = video_data
        self._artist: AxesImage | None = None
        self._extent = self._compute_extent(video_data, env)

    def _compute_extent(self, video_data: VideoData, env: Environment) -> list[float]:
        """Compute extent once (assumes constant frame size)."""
        if video_data.transform_to_env is not None:
            w, h = video_data.reader.frame_size_px
            corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]])
            corners_cm = video_data.transform_to_env(corners_px)
            return [
                corners_cm[:, 0].min(), corners_cm[:, 0].max(),
                corners_cm[:, 1].min(), corners_cm[:, 1].max(),
            ]
        return list(video_data.env_bounds)

    def render(self, ax: plt.Axes, frame_idx: int) -> None:
        """Render video frame, reusing artist if possible."""
        video_frame_idx = self.video_data.frame_indices[frame_idx]
        if video_frame_idx < 0:
            if self._artist is not None:
                self._artist.set_visible(False)
            return

        frame_rgb = self.video_data.reader[video_frame_idx]

        if self._artist is None:
            # First frame: create artist
            self._artist = ax.imshow(
                frame_rgb,
                extent=self._extent,
                aspect="auto",
                origin="lower",
                alpha=self.video_data.alpha,
                zorder=-1 if self.video_data.z_order == "below" else 1,
            )
        else:
            # Subsequent frames: reuse artist with set_data
            self._artist.set_data(frame_rgb)
            self._artist.set_visible(True)


# For parallel rendering (video export), each worker creates fresh artists:
def _render_video_background(
    ax: plt.Axes,
    video_data: VideoData,
    frame_idx: int,
    env: Environment,
) -> None:
    """Render video frame as background image (parallel-safe).

    Note: video_data.transform_to_env already includes crop offset
    composition (see Task 3.2), so corner transform works correctly
    for both cropped and uncropped videos.
    """
    video_frame_idx = video_data.frame_indices[frame_idx]
    if video_frame_idx < 0:
        return  # No video for this frame

    frame_rgb = video_data.reader[video_frame_idx]

    # Calculate extent in environment cm coordinates
    if video_data.transform_to_env is not None:
        h, w = frame_rgb.shape[:2]
        corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        corners_cm = video_data.transform_to_env(corners_px)
        extent = [
            corners_cm[:, 0].min(), corners_cm[:, 0].max(),
            corners_cm[:, 1].min(), corners_cm[:, 1].max(),
        ]
    else:
        extent = list(video_data.env_bounds)

    ax.imshow(
        frame_rgb,
        extent=extent,
        aspect="auto",
        origin="lower",
        alpha=video_data.alpha,
        zorder=-1 if video_data.z_order == "below" else 1,
    )
```

**Performance notes**:

- Widget backend: Use `VideoFrameRenderer` class for artist reuse (`set_data()`)
- Video export with `n_workers > 1`: Each worker creates fresh artists (parallel-safe)
- Video export with `n_workers = 1`: Can use artist reuse for ~2x speedup

**Acceptance Criteria**:

- [ ] Video composited beneath field heatmap
- [ ] Spatial extent computed from transform
- [ ] Works with parallel rendering (`n_workers > 1`)
- [ ] Handles missing frames (video_frame_idx = -1)

---

### Task 4.3: Widget and HTML Backends

**File**: [src/neurospatial/animation/backends/widget_backend.py](src/neurospatial/animation/backends/widget_backend.py)
**File**: [src/neurospatial/animation/backends/html_backend.py](src/neurospatial/animation/backends/html_backend.py)

**Widget backend**: Uses `VideoFrameRenderer` for artist reuse (see Task 4.2):

```python
# widget_backend.py - use artist reuse for sequential rendering
class WidgetAnimator:
    def __init__(self, ...):
        self._video_renderers: list[VideoFrameRenderer] = []
        for video_data in overlay_data.videos:
            self._video_renderers.append(
                VideoFrameRenderer(self.ax, video_data, env)
            )

    def _render_frame(self, frame_idx: int) -> None:
        # Render videos first (z_order="below")
        for renderer in self._video_renderers:
            if renderer.video_data.z_order == "below":
                renderer.render(self.ax, frame_idx)
        # ... render field and other overlays ...
```

**HTML backend**: Auto-skip video with explicit warning (see Implementation Guidelines):

```python
# html_backend.py - VIDEO NOT SUPPORTED, explicit skip
_MAX_HTML_VIDEO_FRAMES = 100  # Threshold for auto-skip

def _render_html_animation(...):
    if overlay_data.videos:
        total_frames = sum(len(v.frame_indices) for v in overlay_data.videos)
        warnings.warn(
            f"WHAT: HTML backend skipping VideoOverlay ({total_frames} frames).\n"
            f"WHY: Base64 embedding would create oversized HTML file "
            f"(~{total_frames * 0.1:.1f}MB estimated).\n"
            f"HOW: Use 'video' or 'napari' backend for video support:\n"
            f"     env.animate_fields(fields, overlays=[video], backend='video')",
            UserWarning,
        )
        # Strip videos from overlay_data for HTML rendering
        overlay_data = dataclasses.replace(overlay_data, videos=[])

    # Continue with field + non-video overlay rendering...
```

**Design rationale for HTML**:

- Base64 embedding adds ~33% overhead (4/3 ratio)
- Even 100 frames at 640x480x3 = ~92MB before encoding → ~123MB in HTML
- Auto-skip is safer than partial/broken support
- Clear WHAT/WHY/HOW message guides users to alternatives

**Acceptance Criteria**:

- [ ] Widget backend: full video support via `VideoFrameRenderer`
- [ ] HTML backend: auto-skip videos with WHAT/WHY/HOW warning
- [ ] HTML continues rendering field + other overlays (positions, regions)
- [ ] No silent partial video support in HTML

---

## Phase 5: Environment Integration

**Objective**: Add convenience methods to Environment for video registration.

### Task 5.1: Add `register_video()` Method

**File**: [src/neurospatial/animation/calibration.py](src/neurospatial/animation/calibration.py) (NEW MODULE)

**Why a new module instead of environment mixin**:

- Avoids circular imports (environment → transforms → alignment → environment)
- Keeps video-specific logic in the animation package
- `VideoCalibration` is already in `transforms.py`, calibration helpers belong nearby
- User imports from `neurospatial.animation` for video-related functionality anyway

**Alternative considered**: `environment/alignment.py` - rejected because it would add
dependency on animation package from core environment module.

```python
def calibrate_video(
    video_path: str | Path,
    env: Environment,
    *,
    scale_bar: tuple[tuple[float, float], tuple[float, float], float] | None = None,
    landmarks_px: NDArray[np.float64] | None = None,
    landmarks_env: NDArray[np.float64] | None = None,
    cm_per_px: float | None = None,
) -> VideoCalibration:
    """Calibrate a video to an environment's coordinate system.

    Computes and returns calibration for pixel↔cm conversion. The calibration
    can be passed to VideoOverlay for spatial alignment.

    Parameters
    ----------
    video_path : str | Path
        Path to video file for extracting frame size.
    env : Environment
        Environment to calibrate against (used for bounds validation).
    scale_bar : tuple, optional
        Scale bar calibration as ((x1, y1), (x2, y2), length_cm).
    landmarks_px : ndarray, optional
        Landmark points in video pixel coordinates.
    landmarks_env : ndarray, optional
        Corresponding landmark points in environment cm coordinates.
    cm_per_px : float, optional
        Direct scale factor if known (with standard Y-flip).

    Returns
    -------
    VideoCalibration
        Calibration object for VideoOverlay.

    Raises
    ------
    ValueError
        If no calibration method specified or incompatible parameters.

    Examples
    --------
    >>> from neurospatial.animation import calibrate_video, VideoOverlay
    >>>
    >>> # Calibrate from arena corners
    >>> corners_px = np.array([[50, 50], [590, 50], [590, 430], [50, 430]])
    >>> corners_env = np.array([[0, 0], [100, 0], [100, 80], [0, 80]])
    >>> calibration = calibrate_video(
    ...     "session.mp4",
    ...     env,
    ...     landmarks_px=corners_px,
    ...     landmarks_env=corners_env,
    ... )
    >>> video = VideoOverlay(source="session.mp4", calibration=calibration)
    """
```

**Acceptance Criteria**:

- [ ] Supports scale bar, landmark, and direct cm_per_px methods
- [ ] Validates that landmarks match environment bounds (warning if not)
- [ ] Returns reusable `VideoCalibration` object

---

## Phase 6: Testing

**Objective**: Comprehensive test coverage for video integration.

### Task 6.1: Unit Tests for Calibration

**File**: `tests/test_transforms.py` (extend existing file with new test class)

Note: Calibration functions are in `src/neurospatial/transforms.py`, so tests belong with other transform tests.

```python
class TestVideoCalibration:
    def test_scale_bar_calibration(self):
        """Scale bar produces correct cm_per_px."""

    def test_landmark_calibration_rigid(self):
        """Landmark calibration with rigid transform."""

    def test_landmark_calibration_similarity(self):
        """Landmark calibration with similarity transform."""

    def test_roundtrip_px_cm_px(self):
        """Pixel → cm → pixel roundtrip within tolerance."""

    def test_calibration_serialization(self):
        """VideoCalibration can be serialized to JSON."""
```

### Task 6.2: Unit Tests for Video I/O

**File**: `tests/animation/test_video_io.py` (new file)

```python
class TestVideoReader:
    def test_reader_loads_metadata(self, sample_video):
        """Reader extracts n_frames, fps, frame_size."""

    def test_reader_lazy_loading(self, sample_video):
        """Frames not loaded until accessed."""

    def test_reader_lru_cache(self, sample_video):
        """Cache evicts least-recently-used frames."""

    def test_reader_pickle_roundtrip(self, sample_video):
        """Reader can be pickled and reconstituted."""

    def test_reader_timestamps(self, sample_video):
        """Timestamps synthesized from fps or extracted from metadata."""
```

### Task 6.3: Integration Tests for Backends

**File**: `tests/animation/test_video_overlay.py` (new file)

**Test markers**:

- `@pytest.mark.slow` - Tests that spin up real napari viewers (skip with `pytest -m "not slow"`)
- Tests without napari can run fast in CI

```python
import pytest


@pytest.mark.slow
class TestVideoOverlayNapari:
    """Napari integration tests - marked slow due to viewer overhead.

    Run with: uv run pytest -m slow tests/animation/test_video_overlay.py
    Skip with: uv run pytest -m "not slow"
    """

    def test_video_layer_added(self, env, sample_video):
        """Video overlay creates Image layer in napari."""

    def test_video_spatial_alignment(self, env, sample_video, calibration):
        """Video aligned to environment coordinates."""

    def test_video_temporal_sync(self, env, sample_video):
        """Video frames synchronized with field animation."""


class TestVideoOverlayExport:
    """Video export tests - fast, no viewer required."""

    def test_video_composited_in_output(self, env, sample_video, tmp_path):
        """Exported video contains background frames."""

    def test_video_parallel_rendering(self, env, sample_video, tmp_path):
        """Parallel export works with video overlay."""
```

**pytest configuration** (add to `pyproject.toml` if not present):

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

### Task 6.4: Test Fixtures

**File**: `tests/conftest.py`

**IMPORTANT**: Use tiny videos (16×16 pixels, 10 frames) for fast CI. Real-world video sizes are unnecessary for testing coordinate transforms and pipeline logic.

```python
# Test video dimensions - INTENTIONALLY TINY for fast CI
_TEST_VIDEO_WIDTH = 16   # pixels
_TEST_VIDEO_HEIGHT = 16  # pixels
_TEST_VIDEO_FRAMES = 10  # frames
_TEST_VIDEO_FPS = 10.0   # fps (1 second duration)


@pytest.fixture
def sample_video(tmp_path):
    """Create a tiny test video file (16×16, 10 frames).

    Uses minimal dimensions to keep CI fast while still testing all
    coordinate transform and pipeline logic.
    """
    import imageio

    video_path = tmp_path / "test_video.mp4"
    frames = []

    for i in range(_TEST_VIDEO_FRAMES):
        # Create frame with identifiable pattern (gradient changes per frame)
        frame = np.zeros((_TEST_VIDEO_HEIGHT, _TEST_VIDEO_WIDTH, 3), dtype=np.uint8)
        # Color gradient that changes each frame for easy visual verification
        frame[:, :, 0] = i * 25  # Red channel varies by frame
        frame[:, :, 1] = np.arange(_TEST_VIDEO_WIDTH) * 16  # Green gradient
        frame[:, :, 2] = np.arange(_TEST_VIDEO_HEIGHT)[:, None] * 16  # Blue gradient
        frames.append(frame)

    imageio.mimwrite(video_path, frames, fps=_TEST_VIDEO_FPS)
    return video_path


@pytest.fixture
def sample_video_array():
    """Pre-loaded video array (16×16, 10 frames) for non-file tests."""
    frames = np.zeros(
        (_TEST_VIDEO_FRAMES, _TEST_VIDEO_HEIGHT, _TEST_VIDEO_WIDTH, 3),
        dtype=np.uint8,
    )
    for i in range(_TEST_VIDEO_FRAMES):
        frames[i, :, :, 0] = i * 25
    return frames


@pytest.fixture
def sample_calibration():
    """VideoCalibration for 16×16 video with 1.0 cm/px.

    Maps the tiny test video to a 16×16 cm environment.
    """
    from neurospatial.transforms import VideoCalibration, scale_2d, flip_y

    # 1 cm per pixel, Y-flip for video→env conversion
    transform = flip_y @ scale_2d(1.0, 1.0)
    return VideoCalibration(
        transform_px_to_cm=transform,
        frame_size_px=(_TEST_VIDEO_WIDTH, _TEST_VIDEO_HEIGHT),  # (width, height)
    )
```

**Why tiny videos**:

- 16×16×3×10 frames = 7.5 KB (vs 640×480×3×30 = 27.6 MB for typical test)
- Coordinate transform math is the same regardless of size
- CI runs in seconds instead of minutes
- Disk I/O minimal

### Task 6.5: Validation and Edge Case Tests

**File**: `tests/animation/test_video_validation.py` (new file)

```python
class TestVideoEnvValidation:
    """Test 2D environment requirement and non-grid fallback."""

    def test_rejects_1d_environment(self, linearized_env, sample_video):
        """VideoOverlay raises ValueError for 1D track environments."""
        video = VideoOverlay(source=sample_video, calibration=None)
        with pytest.raises(ValueError, match="2D environment"):
            linearized_env.animate_fields(fields, overlays=[video])

    def test_non_grid_2d_environment_works_with_warning(self, polygon_env, sample_video):
        """VideoOverlay works on non-grid 2D envs with fallback warning."""
        video = VideoOverlay(source=sample_video, calibration=None)
        with pytest.warns(UserWarning, match="fallback"):
            # Should NOT raise - uses dimension_ranges fallback
            polygon_env.animate_fields(
                fields, overlays=[video], backend="video",
                save_path=tmp_path / "output.mp4"
            )
        assert (tmp_path / "output.mp4").exists()

    def test_non_grid_extent_uses_dimension_ranges(self, polygon_env, sample_video):
        """Non-grid env uses dimension_ranges for video extent."""
        video = VideoOverlay(source=sample_video, calibration=calibration)
        # Video should cover env.dimension_ranges bounding box
        video_data = _convert_overlays_to_data([video], polygon_env, frame_times).videos[0]
        assert video_data.env_bounds == (
            polygon_env.dimension_ranges[0][0], polygon_env.dimension_ranges[0][1],
            polygon_env.dimension_ranges[1][0], polygon_env.dimension_ranges[1][1],
        )

    @pytest.mark.skipif(not HAS_3D_SUPPORT, reason="3D env test")
    def test_rejects_3d_environment(self, env_3d, sample_video):
        """VideoOverlay raises ValueError for 3D environments."""
        video = VideoOverlay(source=sample_video, calibration=None)
        with pytest.raises(ValueError, match="2D environment"):
            env_3d.animate_fields(fields, overlays=[video])


class TestVideoCalibrationValidation:
    """Test calibration edge cases."""

    def test_warns_uncalibrated_video(self, env, sample_video):
        """Uncalibrated video emits UserWarning."""
        video = VideoOverlay(source=sample_video, calibration=None)
        with pytest.warns(UserWarning, match="no calibration"):
            _convert_overlays_to_data([video], env, frame_times)

    def test_warns_bounds_mismatch(self, env, sample_video, small_calibration):
        """Warns when env bounds exceed video coverage."""
        video = VideoOverlay(source=sample_video, calibration=small_calibration)
        with pytest.warns(UserWarning, match="extend beyond"):
            _convert_overlays_to_data([video], env, frame_times)

    def test_rejects_ill_conditioned_landmarks(self):
        """Collinear landmarks raise ValueError."""
        collinear_px = np.array([[0, 0], [10, 0], [20, 0]])  # All on x-axis
        collinear_cm = np.array([[0, 0], [1, 0], [2, 0]])
        with pytest.raises(ValueError, match="ill-conditioned"):
            calibrate_from_landmarks(collinear_px, collinear_cm, (640, 480))


class TestNapariFreeOperation:
    """Ensure video overlay works without napari installed."""

    def test_video_export_without_napari(self, env, sample_video, tmp_path, monkeypatch):
        """Video export works when napari is not installed."""
        # Mock napari import to fail
        monkeypatch.setattr("builtins.__import__", _mock_napari_import_error)
        video = VideoOverlay(source=sample_video, calibration=calibration)
        # Should work without napari
        env.animate_fields(
            fields, overlays=[video], backend="video",
            save_path=tmp_path / "output.mp4"
        )
        assert (tmp_path / "output.mp4").exists()

    def test_import_without_napari(self, monkeypatch):
        """neurospatial.animation imports without napari."""
        monkeypatch.setattr("builtins.__import__", _mock_napari_import_error)
        # Should not raise ImportError
        from neurospatial.animation import VideoOverlay
```

**Test fixtures for various environment types**:

```python
# In tests/conftest.py

@pytest.fixture
def linearized_env():
    """1D linearized track environment (NOT supported for video)."""
    # Create GraphLayout-based environment with is_1d=True
    return Environment.from_graph(track_graph, ...)

@pytest.fixture
def polygon_env():
    """2D non-grid environment (supported with fallback)."""
    # Create ShapelyPolygonLayout - has dimension_ranges but no grid_shape
    return Environment.from_polygon(polygon, bin_size=5.0)

@pytest.fixture
def masked_env():
    """2D masked grid environment (fully supported)."""
    # Create MaskedGridLayout - has grid_shape
    return Environment.from_mask(mask, bin_size=5.0)
```

**Acceptance Criteria**:

- [ ] All new functions have unit tests
- [ ] Integration tests cover napari + video export backends
- [ ] Test fixtures create minimal valid video files
- [ ] CI passes with `uv run pytest`
- [ ] 1D environments rejected with clear error messages
- [ ] Non-grid 2D environments work with fallback warning
- [ ] Tests run without napari installed (no import-time failures)
- [ ] Calibration edge cases (ill-conditioned, bounds mismatch) tested

---

## Phase 7: Documentation

**Objective**: Update documentation with video integration guide.

### Task 7.1: Update CLAUDE.md Quick Reference

**File**: [CLAUDE.md](CLAUDE.md)

Add video overlay examples to Quick Reference section:

```python
# Video overlay with calibration (v0.5.0+)
from neurospatial.animation import VideoOverlay, calibrate_video

# Calibrate from arena corners (video pixels → environment cm)
calibration = calibrate_video(
    "session.mp4",
    env,
    landmarks_px=corner_pixels,
    landmarks_env=corner_env_coords,
)

# Create video overlay
video = VideoOverlay(
    source="session.mp4",
    calibration=calibration,
    alpha=0.5,
)

# Animate with video background
env.animate_fields(
    fields,
    overlays=[video, position_overlay],
    backend="napari",
)
```

### Task 7.2: Add Video Integration Example Notebook

**File**: `examples/18_video_overlay.ipynb` (new file)

Contents:

1. Loading and inspecting video metadata
2. Calibrating with scale bar method
3. Calibrating with landmark correspondences
4. Creating VideoOverlay with various options
5. Animating fields with video background
6. Exporting synchronized video
7. Performance tips for large videos

### Task 7.3: Update Animation Guide

**File**: `docs/guides/animation.md` (or equivalent)

Add section on video integration covering:

- Coordinate systems (video pixel vs environment cm)
- Calibration methods comparison
- Backend capabilities matrix (update for video support)
- Troubleshooting spatial misalignment

**Acceptance Criteria**:

- [ ] CLAUDE.md has working video overlay examples
- [ ] Example notebook runs without errors
- [ ] Animation guide updated with video section

---

## Implementation Order

Recommended sequence to minimize integration risk:

```
Phase I (Integration)    Phase 1 (Foundation)     Phase 2-3 (Data Model)
─────────────────────    ─────────────────────    ─────────────────────────
I.1 Type signatures      1.1 Calibration helpers  2.1 VideoOverlay
I.2 Artist reuse fix     1.2 VideoCalibration     2.2 VideoData
I.3 DRY transforms               │                3.1 VideoReader
I.4 Napari orientation           │                3.2 Pipeline integration
I.5 Add imageio dep              │                3.3 OverlayData + consumers
I.6 dimension_ranges             │                        │
        │                        │                        │
        └────────────────────────┴────────────────────────┘
                                 │
                          Phase 4 (Backends)         Phase 5-7 (Polish)
                          ─────────────────          ─────────────────
                          4.2 Video export           5.1 calibrate_video()
                          4.1 Napari                 6.x Testing
                          4.3 Widget/HTML            7.x Documentation
```

**Linear Task Order** (for sequential execution):

1. **I.1-I.6**: Integration pre-requisites (can run existing tests after each)
2. **1.1-1.2**: Calibration infrastructure (write unit tests as you go)
3. **2.1-2.2**: Data model classes (VideoOverlay, VideoData)
4. **3.1**: VideoReader implementation
5. **3.2-3.3**: Pipeline integration (connect VideoOverlay → VideoData → OverlayData)
6. **4.2**: Video export backend (easiest to test without GUI)
7. **4.1**: Napari backend (requires manual testing)
8. **4.3**: Widget and HTML backends
9. **5.1**: Convenience function `calibrate_video()`
10. **6.x**: Complete test coverage
11. **7.x**: Documentation

**Milestone Checkpoints**:

1. **M0**: Integration tasks complete - existing tests still pass (Tasks I.1-I.6)
2. **M1**: Calibration functions pass unit tests (Tasks 1.1-1.2)
3. **M2**: VideoReader loads frames and timestamps (Task 3.1)
4. **M3**: Video export backend composites video beneath fields (Task 4.2)
5. **M4**: Napari backend displays video with spatial alignment (Task 4.1)
6. **M5**: All tests pass, documentation complete (Phases 6-7)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **Integration Risks** | |
| Artist reuse breaks existing animations | Track field artist explicitly by name, not by index; run existing tests after refactor |
| Type signature changes break downstream | Union types are additive; existing code accepting old types still works |
| Transform DRY violations | Wrap existing `flip_y()`, `scale_2d()` functions; verify with unit tests |
| **Runtime Risks** | |
| Large videos cause OOM | VideoReader with LRU cache, configurable cache_size |
| cv2 not installed | Graceful fallback to imageio with ImportError handling |
| Coordinate misalignment | Validation warnings when calibration doesn't cover env bounds |
| Slow video decoding | Downsample option, chunked frame access |
| Pickle failures in parallel export | `VideoReader.__reduce__` drops cache, reconstitutes from path |
| HTML backend limitations | Early warning, graceful degradation to no-video |
| dimension_ranges missing | Early validation with WHAT/WHY/HOW error message |

---

## Design Decisions (Resolved)

1. **Default calibration behavior**: **(a) Stretch to fill environment bounds** with explicit warning that video is uncalibrated. This matches the behavior of `imshow(extent=env_bounds)` when no transform is provided.

2. **Multiple video overlays**: **Yes, supported**. Render in z-order ("below" videos first, then field, then "above" videos). The `videos: list[VideoData]` in `OverlayData` already supports this.

3. **ffmpeg dependency**: **Optional**. If `ffprobe` available, use for high-quality timestamps. Otherwise, fall back to `n_frames / fps` with warning about potential drift for long videos.

4. **Memory budget**: **LRU cache + downsample is sufficient**. The `VideoReader.cache_size` and `VideoOverlay.downsample` parameters provide adequate control. A higher-level memory warning can be added later if real users hit limits.

---

## Dependencies

**Required**:

- numpy (existing)
- imageio or opencv-python (video reading)

**Optional**:

- ffmpeg (accurate timestamps via ffprobe)
- napari (napari backend)

**New dev dependencies**:

- None (use imageio for test video generation)

---

## Success Criteria

Feature is complete when:

**Integration (Pre-requisites)**:

- [ ] Type signatures updated: `VideoOverlay` accepted in `animate_fields()` (core.py, visualization.py)
- [ ] Artist reuse fixed: Field artist tracked explicitly (not `ax.images[0]`)
- [ ] DRY transforms: `calibrate_from_scale_bar()` wraps existing `flip_y()` and `scale_2d()`
- [ ] Napari orientation: Video affine matches `field_to_rgb_for_napari()` transpose+flip
- [ ] Dependencies: `imageio>=2.35.0` added to dev or video extras
- [ ] Validation: `dimension_ranges` checked for existence and finite values
- [ ] Existing tests still pass after integration changes

**Core Functionality**:

- [ ] `VideoOverlay` and `calibrate_video` documented in CLAUDE.md with working examples
- [ ] Calibration methods tested with scale-dependent tolerance: `max(1e-4, 1e-6 * extent)`
- [ ] Napari backend displays video synchronized with fields (streaming, not preloaded)
- [ ] Video export produces correct spatial composite (Y-flip in calibration only)
- [ ] All tests pass: `uv run pytest tests/animation/test_video*`
- [ ] Example notebook runs without errors

**DRY and Architecture**:

- [ ] `_find_nearest_indices()` used by both `_interp_nearest()` and video pipeline
- [ ] `build_env_to_napari_matrix()` exists in `animation/transforms.py`
- [ ] Crop offset correctly composed into `VideoData.transform_to_env`
- [ ] `OverlayData.videos` field accessible in all backends without AttributeError

**Validation and Error Handling**:

- [ ] 2D validation: `VideoOverlay` raises `ValueError` for 1D and 3D environments
- [ ] Non-grid 2D fallback: Works with fallback warning (uses dimension_ranges)
- [ ] Uncalibrated video warning emitted when `calibration=None`
- [ ] Bounds coverage warning when env extends beyond calibrated video
- [ ] Ill-conditioned landmark calibration rejected with actionable error

**Backend Behavior**:

- [ ] Napari: in-place layer updates (`layer.data[...] = frame`)
- [ ] Widget: artist reuse via `VideoFrameRenderer.set_data()`
- [ ] HTML: auto-skip video with WHAT/WHY/HOW warning
- [ ] No regression in existing overlay tests

**Testing Requirements**:

- [ ] Tests for 1D environments (expect `ValueError`)
- [ ] Tests for non-grid 2D environments (expect warning, successful output)
- [ ] Tests run without napari installed (no import-time failures)
- [ ] Parallel export pickle-safe (VideoReader reconstitutes from path)
- [ ] Test fixtures use tiny videos (16×16, 10 frames) for fast CI
- [ ] Napari integration tests marked with `@pytest.mark.slow`
- [ ] `pytest -m "not slow"` runs fast (~seconds) without napari tests
