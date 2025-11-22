# Video Overlay Coordinate Systems

**Internal Developer Documentation**

This document describes the coordinate systems used in the video overlay feature
and the transformations between them. Understanding these is critical for debugging
spatial alignment issues.

---

## Three Coordinate Spaces

### 1. Video Pixel Space

```
Origin: Top-left of video frame
Axes:
  - x_px (column): Increases rightward (0 to width-1)
  - y_px (row): Increases downward (0 to height-1)

    (0,0) ─────────────────► x_px (column)
      │
      │    ┌──────────────────────┐
      │    │                      │
      │    │   VIDEO FRAME        │
      │    │                      │
      ▼    │                      │
    y_px   └──────────────────────┘
    (row)                     (width-1, height-1)
```

**Array Convention**: NumPy arrays from VideoReader have shape `(height, width, 3)`.
The tuple `frame_size_px` uses `(width, height)` to match video metadata convention.

### 2. Environment Space (cm)

```
Origin: Bottom-left of environment
Axes:
  - x_cm: Increases rightward
  - y_cm: Increases upward (opposite of video!)

    y_cm
      ▲
      │                (x_max, y_max)
      │    ┌──────────────────────┐
      │    │                      │
      │    │   ENVIRONMENT        │
      │    │                      │
      │    └──────────────────────┘
      │  (x_min, y_min)
      └────────────────────────────► x_cm
```

**Key Difference**: Y-axis is inverted relative to video coordinates.
This is the standard scientific convention (y increases upward).

### 3. Napari World Space (pixels)

```
Origin: Top-left (like video)
Axes:
  - col: Increases rightward (maps to environment x)
  - row: Increases downward (maps to inverted environment y)

    (0,0) ─────────────────► col
      │
      │    ┌──────────────────────┐
      │    │                      │
      │    │   NAPARI DISPLAY     │
      │    │   (row 0 = max y)    │
      │    │                      │
      ▼    └──────────────────────┘
    row                      (n_x-1, n_y-1)
```

**Note**: Napari uses (row, col) ordering for coordinates, while environment
uses (x, y). The transformation swaps these axes AND inverts y.

---

## Transform Chain

The full transform chain for video overlay is:

```
video_px ──► env_cm ──► napari_px
         │          │
         │          └── build_env_to_napari_matrix()
         │              [from animation/transforms.py]
         │
         └── VideoCalibration.transform_px_to_cm
             [from neurospatial/transforms.py]
```

### Step 1: Video Pixel → Environment (cm)

**Where**: `VideoCalibration.transform_px_to_cm` (Affine2D)

**What it does**:
1. **Y-flip**: Converts top-left origin to bottom-left origin
2. **Scale**: Converts pixels to centimeters
3. **Optional translate/rotate**: For landmark-based calibration

```python
# Typical construction for scale bar calibration:
transform_px_to_cm = flip_y(frame_height) @ scale_2d(cm_per_px, cm_per_px)
```

**Y-FLIP POLICY**: The Y-flip happens ONCE, here in the calibration transform.
All downstream code uses `origin="lower"` in matplotlib to preserve this.
Do NOT add another Y-flip elsewhere.

### Step 2: Environment (cm) → Napari (pixels)

**Where**: `build_env_to_napari_matrix()` in `animation/transforms.py`

**What it does**:
1. **Swap axes**: (x, y) → (row, col)
2. **Invert Y**: row 0 = max y (napari top = environment top after field flip)
3. **Scale**: cm → pixels based on grid shape

```python
# From the matrix form:
col = (x - x_min) * x_scale           # X maps to column
row = (n_y - 1) - (y - y_min) * y_scale  # Y maps to row (inverted)
```

### Combined Transform (for napari affine parameter)

```python
# In _build_video_napari_affine():
video_to_env = video_data.transform_to_env.A    # 3x3 matrix
env_to_napari = build_env_to_napari_matrix(scale)  # 3x3 matrix
combined = env_to_napari @ video_to_env          # napari affine
```

---

## Y-Flip Summary

| Location | Y-Flip? | Reason |
|----------|---------|--------|
| `calibrate_from_scale_bar()` | YES | Video origin (top-left) → env origin (bottom-left) |
| `calibrate_from_landmarks()` | YES | Same - composed into calibration transform |
| `VideoCalibration.transform_px_to_cm` | YES | This IS where the flip happens |
| `_build_video_napari_affine()` | YES | Env y-up → napari y-down (built into env_to_napari matrix) |
| matplotlib `imshow()` | NO | Uses `origin="lower"` to preserve env orientation |
| napari `add_image()` | NO | Uses affine parameter that includes the flip |

**Total Y-flips**: 2 (one in video→env, one in env→napari)
These compose correctly: video y-down → env y-up → napari y-down (displayed correctly)

---

## Common Bugs and How to Avoid Them

### Bug: Video appears upside-down

**Cause**: Double Y-flip (e.g., using `origin="upper"` after calibration already flipped)

**Fix**: Always use `origin="lower"` with calibrated video. The calibration transform
already handles the video→env Y-flip.

### Bug: Video appears mirrored left-right

**Cause**: Incorrect axis swap in env→napari transform

**Fix**: Check that `build_env_to_napari_matrix()` correctly maps x→col and y→row.
The matrix should have x_scale in position [1,0] and -y_scale in position [0,1].

### Bug: Video offset from expected position

**Cause**: Missing crop offset in transform chain

**Fix**: When video is cropped, compose crop translation BEFORE calibration:
```python
effective_transform = calibration.transform_px_to_cm @ translate(crop_x, crop_y)
```

### Bug: Video and field don't align at corners

**Cause**: Mismatch between calibration landmarks and actual video corners

**Fix**: Verify calibration landmarks are in ORIGINAL pixel coordinates (before crop/downsample).
The VideoReader's `crop_offset_px` must be composed into the transform.

---

## Testing Coordinate Transforms

### Unit Test Strategy

```python
def test_video_field_corner_alignment():
    """Video corners align with environment corners after transform."""
    # 1. Create environment with known bounds
    env = Environment.from_samples(positions, bin_size=5.0)

    # 2. Create calibration that maps video corners to env corners
    video_corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    env_corners = np.array([
        [env.dimension_ranges[0][0], env.dimension_ranges[1][1]],  # top-left
        [env.dimension_ranges[0][1], env.dimension_ranges[1][1]],  # top-right
        [env.dimension_ranges[0][1], env.dimension_ranges[1][0]],  # bottom-right
        [env.dimension_ranges[0][0], env.dimension_ranges[1][0]],  # bottom-left
    ])

    # 3. Transform video corners through full chain
    calibration = calibrate_from_landmarks(video_corners_px, env_corners, (w, h))
    env_coords = calibration.transform_px_to_cm(video_corners_px)
    napari_coords = transform_coords_for_napari(env_coords, env)

    # 4. Verify napari corners match expected grid positions
    assert napari_coords[0] == [0, 0]  # top-left → row 0, col 0
    assert napari_coords[1] == [0, n_x-1]  # top-right → row 0, col n_x-1
    # etc.
```

---

## References

- `src/neurospatial/transforms.py`: `calibrate_from_scale_bar()`, `calibrate_from_landmarks()`, `VideoCalibration`
- `src/neurospatial/animation/transforms.py`: `build_env_to_napari_matrix()`, `transform_coords_for_napari()`
- `src/neurospatial/animation/backends/napari_backend.py`: `_build_video_napari_affine()`, `_add_video_layer()`
