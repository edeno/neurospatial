# Animation Overlays

Visualize animal behavior alongside spatial field dynamics using flexible overlay types for positions, poses, and head direction.

## Quick Start

```python
from neurospatial import Environment
from neurospatial.animation import PositionOverlay, BodypartOverlay, HeadDirectionOverlay
import numpy as np

# Create environment and fields
env = Environment.from_samples(positions, bin_size=2.5)
fields = [compute_place_field(env, spikes[i], times, positions) for i in range(100)]

# Position overlay with trail
position_overlay = PositionOverlay(
    data=trajectory,  # shape: (n_samples, 2)
    times=timestamps,  # Optional: align to fields
    color="red",
    size=10.0,
    trail_length=5  # Show last 5 frames
)

# Animate with overlay
env.animate_fields(
    fields,
    overlays=[position_overlay],
    backend="napari",  # or "video", "html", "widget"
)
```

## Overview

The overlay system enables rich behavioral visualizations by combining spatial fields with:

- **Position tracking**: Trajectory with trails
- **Pose estimation**: Multi-keypoint tracking with skeleton
- **Head direction**: Orientation arrows
- **Multi-animal**: Multiple overlays of the same type
- **Regions**: Named spatial zones

**Key features:**

- Temporal alignment: Overlays automatically sync to field frames
- Mixed sampling rates: Interpolation handles different data frequencies
- Backend flexibility: Full support in Napari/Video/Widget, partial in HTML
- Scientifically correct: NaN extrapolation, validation, actionable errors

**Related resources:**

- [Example Notebook](examples/17_animation_with_overlays.ipynb) - Hands-on tutorial with 7 comprehensive examples
- [Field Animation Guide](examples/16_field_animation.ipynb) - Core animation features without overlays

## Overlay Types

### Position Overlay

Track a single trajectory with optional trail visualization.

```python
from neurospatial.animation import PositionOverlay

# Basic position overlay
overlay = PositionOverlay(
    data=trajectory,        # (n_samples, n_dims) - x, y coordinates
    times=None,             # Optional timestamps for alignment
    color="red",            # Marker color
    size=10.0,              # Marker size
    trail_length=None       # Frames of trail (None = no trail)
)

# With temporal alignment
overlay = PositionOverlay(
    data=trajectory,        # shape: (1000, 2) at 250 Hz
    times=timestamps,       # shape: (1000,) in seconds
    color="blue",
    size=12.0,
    trail_length=10         # Show last 10 frames
)
```

**Parameters:**

- `data`: NDArray of shape `(n_samples, n_dims)` with coordinates
- `times`: Optional timestamps for temporal alignment (default: assumes 1:1 with frames)
- `color`: Matplotlib/Napari color name (default: `"red"`)
- `size`: Marker size in points (default: `10.0`)
- `trail_length`: Number of frames to show as trail (default: `None` = no trail)

### Bodypart Overlay

Multi-keypoint pose tracking with optional skeleton visualization.

```python
from neurospatial.animation import BodypartOverlay

# Pose with skeleton
overlay = BodypartOverlay(
    data={
        "nose": nose_coords,        # (n_samples, 2)
        "ear_left": ear_l_coords,   # (n_samples, 2)
        "ear_right": ear_r_coords,  # (n_samples, 2)
        "tail_base": tail_coords,   # (n_samples, 2)
    },
    times=timestamps,               # Optional alignment
    skeleton=[                      # Connect keypoints
        ("nose", "ear_left"),
        ("nose", "ear_right"),
        ("nose", "tail_base"),
    ],
    colors={                        # Per-keypoint colors
        "nose": "red",
        "ear_left": "blue",
        "ear_right": "blue",
        "tail_base": "green",
    },
    skeleton_color="white",         # Skeleton edge color
    skeleton_width=2.0              # Edge width
)
```

**Parameters:**

- `data`: Dict mapping keypoint names to coordinate arrays `(n_samples, n_dims)`
- `times`: Optional timestamps for temporal alignment
- `skeleton`: List of `(part1, part2)` tuples defining edges (default: `None`)
- `colors`: Dict mapping keypoint names to colors (default: `None` = auto-color)
- `skeleton_color`: Color for skeleton edges (default: `"white"`)
- `skeleton_width`: Width of skeleton edges (default: `2.0`)

**Important:** Each keypoint can have **independent timestamps** for per-keypoint interpolation:

```python
# Per-keypoint temporal alignment (advanced)
from neurospatial.animation.overlays import _convert_overlays_to_data

# Use conversion funnel for per-keypoint interpolation
# (Handled automatically by animate_fields)
```

### Head Direction Overlay

Visualize orientation as arrows using angles or unit vectors.

```python
from neurospatial.animation import HeadDirectionOverlay
import numpy as np

# From angles (radians)
overlay = HeadDirectionOverlay(
    data=angles,            # (n_samples,) in radians
    times=timestamps,       # Optional alignment
    color="yellow",
    length=20.0             # Arrow length in env units (e.g., cm)
)

# From unit vectors
directions = np.column_stack([np.cos(angles), np.sin(angles)])  # (n_samples, 2)
overlay = HeadDirectionOverlay(
    data=directions,
    color="orange",
    length=15.0
)
```

**Parameters:**

- `data`: NDArray of shape `(n_samples,)` (angles in radians) or `(n_samples, n_dims)` (unit vectors)
- `times`: Optional timestamps for temporal alignment
- `color`: Arrow color (default: `"yellow"`)
- `length`: Arrow length in environment units (default: `20.0`)

### Video Overlay

Display raw behavioral video frames as a background or foreground layer with spatial calibration.

```python
from neurospatial.animation import VideoOverlay, calibrate_video

# Calibrate video to environment coordinates
calibration = calibrate_video(
    "session.mp4",
    env,
    landmarks_px=arena_corners_px,    # Video pixel coordinates
    landmarks_env=arena_corners_env,  # Environment cm coordinates
)

# Create video overlay
overlay = VideoOverlay(
    source="session.mp4",       # Path or pre-loaded array
    calibration=calibration,    # Pixel↔cm transform
    alpha=0.5,                  # Blend opacity (0-1)
    z_order="above",            # "above" or "below" field
)

# Animate with video background
env.animate_fields(
    fields,
    overlays=[overlay],
    backend="napari"  # or "video", "widget"
)
```

**Parameters:**

- `source`: Path to video file or pre-loaded array `(n_frames, height, width, 3)`
- `calibration`: `VideoCalibration` object for spatial alignment (optional but recommended)
- `times`: Video frame timestamps for temporal alignment (default: synthesized from fps)
- `alpha`: Opacity of video layer, 0.0=transparent, 1.0=opaque (default: `0.5`)
- `z_order`: Draw order relative to field - `"above"` (default) or `"below"`
- `crop`: Crop region `(x, y, width, height)` in pixels (default: `None`)
- `downsample`: Spatial downsampling factor, e.g., 2 = half resolution (default: `1`)

**Best practices:**

| Goal | Settings | Result |
|------|----------|--------|
| Balanced view | `alpha=0.5, z_order="above"` | Equal video/field visibility (default) |
| Field dominant | `alpha=0.3, z_order="above"` | Field shows through video |
| Video dominant | `alpha=0.7, z_order="above"` | Video shows through field |
| Video as background | `z_order="below"` | Only works if field has transparent regions |

**Requirements:**

- 2D environments only (`env.n_dims == 2`)
- Not supported for 1D linearized tracks or 3D environments
- Non-grid 2D environments work with approximate alignment (warning emitted)

## Multi-Animal Support

Track multiple animals by passing **multiple overlays of the same type**:

```python
# Two animals with different colors
animal1_pos = PositionOverlay(
    data=trajectory_1,
    color="red",
    size=12.0,
    trail_length=10
)

animal2_pos = PositionOverlay(
    data=trajectory_2,
    color="blue",
    size=12.0,
    trail_length=10
)

env.animate_fields(
    fields,
    overlays=[animal1_pos, animal2_pos],  # Both positions rendered
    backend="napari"
)
```

**Automatic naming:** Backends assign suffixes for disambiguation (e.g., "Position 1", "Position 2").

## Temporal Alignment

Overlays automatically sync to field frames using interpolation.

### Without Timestamps (1:1 Alignment)

```python
# Assume overlay has same number of samples as fields
overlay = PositionOverlay(
    data=trajectory,  # shape: (100, 2) - matches 100 fields
    times=None        # No interpolation
)
```

**Requirement:** `len(overlay.data)` must equal `len(fields)`.

### With Timestamps (Interpolation)

```python
# Overlay sampled at 250 Hz, fields at 30 fps
overlay = PositionOverlay(
    data=trajectory,     # shape: (1000, 2) at 250 Hz
    times=timestamps,    # shape: (1000,) in seconds
)

# Provide frame times for alignment
frame_times = np.linspace(0, 10, 100)  # 100 frames over 10 seconds

env.animate_fields(
    fields,
    overlays=[overlay],
    frame_times=frame_times,  # Interpolation target
)
```

**Interpolation behavior:**

- **Linear interpolation** for smooth trajectories
- **NaN extrapolation** outside overlay time range (scientifically correct)
- **Warnings** for partial temporal overlap (<50%)
- **Errors** for zero temporal overlap

### Synthesized Frame Times

If `frame_times` not provided, synthesized from `fps`:

```python
env.animate_fields(
    fields,
    overlays=[overlay],
    fps=30,  # Creates frame_times: [0, 1/30, 2/30, ..., (n_frames-1)/30]
)
```

## Regions Overlay

Display named spatial regions (from `env.regions`) during animation.

```python
from neurospatial.regions import Region

# Add regions to environment
env.regions.add("goal", point=np.array([50.0, 50.0]))
env.regions.add("start", polygon=start_polygon)

# Show all regions
env.animate_fields(
    fields,
    show_regions=True,
    region_alpha=0.3  # Transparency
)

# Show specific regions only
env.animate_fields(
    fields,
    show_regions=["goal", "start"],  # List of names
    region_alpha=0.5
)
```

**Parameters:**

- `show_regions`: `bool` (all regions) or `list[str]` (specific names) (default: `False`)
- `region_alpha`: Transparency for region rendering (default: `0.3`)

## Backend Capabilities

Not all backends support all overlay types.

| Backend | Position | Bodypart | HeadDirection | Video | Regions |
|---------|----------|----------|---------------|-------|---------|
| **Napari** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Video** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **HTML** | ✓ | ✗ | ✗ | ✗ | ✓ |
| **Widget** | ✓ | ✓ | ✓ | ✓ | ✓ |

**HTML backend limitations:**

- Supports **positions** and **regions** only
- **Bodypart, head direction, and video overlays are skipped with warnings**
- Emits warnings for unsupported overlays before rendering
- Animation proceeds with only supported overlays rendered
- Suggestion: Use video or napari backend for full overlay features

```python
# HTML backend warns for unsupported overlays
env.animate_fields(
    fields,
    overlays=[bodypart_overlay, position_overlay],  # Mixed overlays
    backend="html"
)
# UserWarning: HTML backend supports positions and regions only.
#              Bodypart and head direction overlays will not be rendered.
#              Consider using 'video' or 'napari' backend for full overlay support.
# → Animation proceeds with only position_overlay rendered
```

## Video Overlay Coordinate Systems

VideoOverlay requires calibration to align video pixels to environment coordinates.

### Coordinate Spaces

Three coordinate systems are involved:

| Space | Origin | X-axis | Y-axis | Units |
|-------|--------|--------|--------|-------|
| **Video Pixel** | Top-left | Right (columns) | Down (rows) | Pixels |
| **Environment** | Bottom-left | Right | Up | cm (or your units) |
| **Napari Display** | Top-left | Right (columns) | Down (rows) | Pixels |

The calibration transform handles the coordinate conversion, including:

1. **Y-axis flip**: Video origin (top-left) → Environment origin (bottom-left)
2. **Scale**: Pixels → centimeters
3. **Rotation/Translation**: If camera and arena are not perfectly aligned

### Calibration Methods

Choose the method that matches your available reference data:

#### Method 1: Scale Bar (Simplest)

Use when you have two points of known distance in the video.

```python
from neurospatial.animation import calibrate_video

# Two endpoints of a scale bar (e.g., ruler in video)
calibration = calibrate_video(
    "session.mp4",
    env,
    scale_bar=((100, 200), (300, 200), 50.0),  # px1, px2, length_cm
)
```

**Limitations:** Assumes uniform scaling and standard camera orientation (no rotation).

#### Method 2: Landmark Correspondences (Most Flexible)

Use when you can identify the same points in both video and environment.

```python
import numpy as np
from neurospatial.animation import calibrate_video

# Arena corners in video pixels (x, y)
corners_px = np.array([
    [50, 50],    # Top-left in video
    [590, 50],   # Top-right
    [590, 430],  # Bottom-right
    [50, 430],   # Bottom-left
])

# Same corners in environment coordinates (x_cm, y_cm)
corners_env = np.array([
    [0, 80],     # Environment top-left (high Y)
    [100, 80],   # Environment top-right
    [100, 0],    # Environment bottom-right (low Y)
    [0, 0],      # Environment bottom-left
])

calibration = calibrate_video(
    "session.mp4",
    env,
    landmarks_px=corners_px,
    landmarks_env=corners_env,
)
```

**Best for:** Cameras at angles, non-uniform scaling, or complex arena geometries.

#### Method 3: Direct Scale Factor

Use when you know the exact cm-per-pixel ratio.

```python
from neurospatial.animation import calibrate_video

# 4 pixels = 1 cm
calibration = calibrate_video(
    "session.mp4",
    env,
    cm_per_px=0.25,
)
```

**Limitations:** Assumes uniform scaling, no rotation, and video origin at environment origin.

### Calibration Comparison

| Method | Accuracy | Ease of Use | Best For |
|--------|----------|-------------|----------|
| **Scale bar** | Medium | Easiest | Quick setup, overhead cameras |
| **Landmarks** | Highest | More work | Angled cameras, precise alignment |
| **cm_per_px** | Low-Medium | Easy | Known camera setup |

### Troubleshooting Spatial Misalignment

#### Video and field don't align

1. **Check coordinate order**: Video landmarks use `(x_px, y_px)` = `(column, row)`
2. **Verify Y-axis direction**: Environment Y increases upward, video Y increases downward
3. **Use more landmarks**: 4+ points give better transforms than 3

#### Video appears rotated

Use landmark calibration instead of scale bar—it can handle rotation.

#### Warning about bounds mismatch

```text
UserWarning: Environment bounds extend beyond calibrated video coverage.
```

The environment is larger than what the video shows. Either:

- Expand video frame to cover more area
- Accept that regions outside video will show no video pixels

#### 1D environment error

```text
ValueError: VideoOverlay requires 2D environment.
```

Video overlay only works with 2D environments. Linearized tracks (1D) have no meaningful 2D extent for video alignment.

## Complete Example

Combine all overlay types for comprehensive behavioral visualization:

```python
from neurospatial import Environment
from neurospatial.animation import (
    PositionOverlay,
    BodypartOverlay,
    HeadDirectionOverlay,
)
import numpy as np

# Load data
positions = np.load("positions.npy")  # Animal position data
pose_data = np.load("pose.npy")  # DLC/SLEAP output
head_angles = np.load("angles.npy")  # Computed head direction
timestamps = np.load("times.npy")  # Timestamps in seconds

# Create environment
env = Environment.from_samples(positions, bin_size=2.5, units="cm")
env.regions.add("goal", point=np.array([80.0, 80.0]))

# Compute fields (e.g., place fields over trials)
fields = [
    compute_place_field(env, spike_times[i], timestamps, positions)
    for i in range(50)
]

# Create overlays
position_overlay = PositionOverlay(
    data=positions,
    times=timestamps,
    color="red",
    size=15.0,
    trail_length=8
)

pose_overlay = BodypartOverlay(
    data={
        "nose": pose_data[:, 0, :],
        "ear_left": pose_data[:, 1, :],
        "ear_right": pose_data[:, 2, :],
        "tail_base": pose_data[:, 3, :],
    },
    times=timestamps,
    skeleton=[
        ("nose", "ear_left"),
        ("nose", "ear_right"),
        ("nose", "tail_base"),
    ],
    colors={
        "nose": "yellow",
        "ear_left": "cyan",
        "ear_right": "cyan",
        "tail_base": "magenta",
    },
    skeleton_color="white",
)

direction_overlay = HeadDirectionOverlay(
    data=head_angles,
    times=timestamps,
    color="orange",
    length=10.0
)

# Frame times (50 frames at 2 Hz)
frame_times = np.linspace(timestamps[0], timestamps[-1], 50)

# Animate with all overlays
env.animate_fields(
    fields,
    overlays=[position_overlay, pose_overlay, direction_overlay],
    frame_times=frame_times,
    show_regions=["goal"],
    region_alpha=0.4,
    backend="video",
    save_path="behavior_analysis.mp4",
    fps=30,
    title="Place Field Dynamics with Behavior",
)
```

## Common Errors and Solutions

The overlay system provides **actionable error messages** following WHAT/WHY/HOW format.

### Non-Monotonic Times

**Error:**
```
ValueError: Overlay timestamps are not monotonically increasing.
  Found 5 non-monotonic points at indices: [10, 25, 47, ...].
  First violation: times[10] = 1.523 >= times[11] = 1.521

  WHY: Interpolation requires strictly increasing timestamps.

  HOW: Choose one of these solutions:
    1. Sort timestamps and corresponding data: idx = np.argsort(times); times = times[idx]; data = data[idx]
    2. Remove duplicate or reversed timestamps
    3. Verify timestamp source for recording errors
```

**Solution:**
```python
# Sort timestamps
idx = np.argsort(overlay_times)
sorted_times = overlay_times[idx]
sorted_data = overlay_data[idx]

overlay = PositionOverlay(data=sorted_data, times=sorted_times, ...)
```

### NaN/Inf Values

**Error:**
```
ValueError: Found 12 NaN/Inf values in overlay data at indices: [5, 8, 23, ...].
  First invalid value at index 5: [nan, 45.2]

  WHY: Rendering cannot place markers at invalid coordinates.

  HOW: Choose one of these solutions:
    1. Remove invalid samples: mask = np.isfinite(data).all(axis=1); data = data[mask]; times = times[mask]
    2. Interpolate over gaps: from scipy.interpolate import interp1d; ...
    3. Mask invalid values and skip rendering: data[~np.isfinite(data)] = np.nan
```

**Solution:**
```python
# Remove invalid samples
mask = np.isfinite(trajectory).all(axis=1)
clean_trajectory = trajectory[mask]
clean_times = timestamps[mask]

overlay = PositionOverlay(data=clean_trajectory, times=clean_times, ...)
```

### Shape Mismatch

**Error:**
```
ValueError: Overlay data has incorrect shape.
  Expected: (n_samples, 2) for 2D environment
  Got: (500, 3)

  WHY: Coordinate dimensionality must match environment.n_dims = 2.

  HOW: Choose one of these solutions:
    1. Project 3D data to 2D: data_2d = data[:, :2]
    2. Use correct dimensionality for environment
    3. Recreate environment with matching dimensions
```

**Solution:**
```python
# Project to 2D
trajectory_2d = trajectory_3d[:, :2]

overlay = PositionOverlay(data=trajectory_2d, ...)
```

### No Temporal Overlap

**Error:**
```
ValueError: No temporal overlap between overlay and frames.
  Overlay time range: [10.5, 25.3] seconds
  Frame time range: [0.0, 8.7] seconds
  Overlap: 0.0%

  WHY: Interpolation cannot generate values outside source data range.

  HOW: Choose one of these solutions:
    1. Adjust frame_times to overlap with overlay times
    2. Provide overlay data that covers frame time range
    3. Verify timestamp alignment (check units: seconds vs milliseconds)
```

**Solution:**
```python
# Verify timestamp units
print(f"Overlay times: {overlay.times.min():.2f} - {overlay.times.max():.2f}")
print(f"Frame times: {frame_times.min():.2f} - {frame_times.max():.2f}")

# Adjust frame_times to overlap
frame_times = np.linspace(overlay.times.min(), overlay.times.max(), n_frames)
```

### Partial Temporal Overlap (Warning)

**Warning:**
```
UserWarning: Limited temporal overlap between overlay and frames.
  Overlay time range: [0.0, 15.0] seconds
  Frame time range: [10.0, 25.0] seconds
  Overlap: 33.3% (5.0 / 15.0 seconds)

  Some frames will show NaN (no overlay data).
  Consider providing overlay data spanning the full frame time range.
```

**Solution:**
```python
# Extend overlay data to cover full range, or accept NaN extrapolation
# NaN extrapolation is scientifically correct for missing data
```

### Skeleton Missing Parts

**Error:**
```
ValueError: Skeleton references bodyparts not in data.
  Missing: ['ear_left', 'tail_tip']
  Available: ['nose', 'ear_right', 'tail_base']

  WHY: Cannot draw edges without both endpoint keypoints.

  HOW: Choose one of these solutions:
    1. Fix bodypart names in skeleton (check for typos)
    2. Add missing bodyparts to data dict
    3. Remove edges with missing parts from skeleton

  Suggestions (fuzzy match):
    'ear_left' -> Did you mean 'ear_right'?
    'tail_tip' -> Did you mean 'tail_base'?
```

**Solution:**
```python
# Fix skeleton typo
skeleton = [
    ("nose", "ear_right"),   # Fixed: was "ear_left"
    ("nose", "tail_base"),   # Fixed: was "tail_tip"
]
```

### Out-of-Bounds Coordinates (Warning)

**Warning:**
```
UserWarning: 23.4% of overlay points (234 / 1000) are outside environment bounds.
  Environment x range: [0.0, 100.0] cm
  Environment y range: [0.0, 100.0] cm
  Overlay x range: [-5.2, 105.3] cm (min/max)
  Overlay y range: [-2.1, 102.7] cm (min/max)

  Out-of-bounds points will still be rendered (may appear outside field).
  Verify coordinate system and units match.
```

**Solution:**
```python
# Verify coordinate systems match
print(f"Environment range: {env.dimension_ranges}")
print(f"Overlay range: x=[{overlay.data[:, 0].min()}, {overlay.data[:, 0].max()}]")

# If units mismatch (e.g., meters vs cm), convert:
trajectory_cm = trajectory_meters * 100
```

### Pickle-ability Error (Parallel Rendering)

**Error:**
```
ValueError: overlay_data is not pickle-able for parallel rendering.
  WHY: Parallel rendering (n_workers=4) requires serializing
       overlay_data to send to worker processes.

  HOW: Choose one of these solutions:
    1. Remove unpickleable objects (lambdas, closures, local functions)
    2. Ensure overlay_data uses only standard types (numpy arrays, strings, numbers)
    3. Use n_workers=1 for serial rendering (no pickling required)
    4. Call env.clear_cache() before parallel rendering
```

**Solution:**
```python
# Clear environment caches before parallel rendering
env.clear_cache()

env.animate_fields(
    fields,
    overlays=[overlay],
    backend="video",
    n_workers=4,  # Now works
    save_path="output.mp4"
)
```

## Troubleshooting Guide

### Problem: Overlays don't appear in animation

**Possible causes:**

1. **Backend doesn't support overlay type**
   - Check [Backend Capabilities](#backend-capabilities) table
   - HTML backend only supports positions and regions

2. **Temporal misalignment**
   - Overlay time range doesn't overlap with frame times
   - Check for timestamp unit mismatches (seconds vs milliseconds)

3. **NaN values in overlay data**
   - Interpolation may produce all-NaN frames if outside data range
   - Check error messages for NaN detection

**Debug:**
```python
# Check temporal alignment
print(f"Overlay times: {overlay.times.min():.2f} - {overlay.times.max():.2f}")
print(f"Frame times: {frame_times.min():.2f} - {frame_times.max():.2f}")

# Check for NaN
print(f"NaN count: {np.isnan(overlay.data).sum()}")
```

### Problem: Animation runs slowly

**Possible causes:**

1. **Too many overlays or large datasets**
   - Use Napari backend for >10K frames (lazy loading)
   - Reduce trail_length for position overlays

2. **Serial rendering (n_workers=1)**
   - Enable parallel rendering for video backend
   - Ensure environment is pickle-able

3. **High-resolution rendering**
   - Reduce dpi for video backend (default: 100)
   - Use lower fps (default: 30)

**Optimize:**
```python
# Parallel video rendering
env.clear_cache()  # Ensure pickle-ability
env.animate_fields(
    fields,
    overlays=[overlay],
    backend="video",
    n_workers=4,  # Use 4 cores
    dpi=80,  # Lower resolution
    fps=24,  # Lower frame rate
    save_path="output.mp4"
)

# Or use Napari for interactive exploration
env.animate_fields(fields, overlays=[overlay], backend="napari")
```

### Problem: Skeleton doesn't connect properly

**Possible causes:**

1. **Bodypart name typo in skeleton**
   - Error message provides fuzzy match suggestions
   - Check for case sensitivity

2. **Missing keypoints in data**
   - Ensure all skeleton endpoints exist in `data` dict

3. **NaN values in keypoint coordinates**
   - Skeleton won't render if either endpoint is NaN

**Fix:**
```python
# Verify all skeleton parts exist
bodypart_names = set(pose_overlay.data.keys())
skeleton_parts = set(sum(pose_overlay.skeleton, ()))  # Flatten tuples
missing = skeleton_parts - bodypart_names
if missing:
    print(f"Missing bodyparts: {missing}")
```

### Problem: Warning about partial temporal overlap

This is expected when overlay data doesn't span the full frame time range.

**Options:**

1. **Accept NaN extrapolation** (scientifically correct)
   - Frames outside overlay range will show no overlay
   - This is the correct behavior for missing data

2. **Adjust frame_times** to match overlay coverage
   ```python
   frame_times = np.linspace(
       overlay.times.min(),
       overlay.times.max(),
       n_frames
   )
   ```

3. **Extend overlay data** to cover full range
   - Collect more data or adjust time window

## Advanced Usage

### Subsampling for Video Export

Reduce frame count for faster video rendering:

```python
from neurospatial.animation import subsample_frames

# Subsample from 250 Hz to 30 fps
subsampled_fields = subsample_frames(fields, source_fps=250, target_fps=30)

env.animate_fields(
    subsampled_fields,
    overlays=[overlay],  # Interpolation handles rate mismatch
    backend="video",
    fps=30,
    save_path="output.mp4"
)
```

### Custom Frame Times

Precisely control temporal alignment:

```python
# Non-uniform frame times (e.g., event-triggered)
frame_times = np.array([0.0, 0.5, 1.2, 2.8, 3.1, ...])

env.animate_fields(
    fields,
    overlays=[overlay],
    frame_times=frame_times,  # Custom timing
    backend="napari"
)
```

### Clearing Caches for Parallel Rendering

Environment caching can prevent pickling. Clear before parallel operations:

```python
# Clear all caches
env.clear_cache()

# Or selectively clear
env.clear_cache(
    kdtree=True,          # Clear KDTree cache
    kernels=False,        # Keep kernel cache
    cached_properties=True  # Clear @cached_property values
)

# Now parallel rendering works
env.animate_fields(
    fields,
    overlays=[overlay],
    backend="video",
    n_workers=4,
    save_path="output.mp4"
)
```

## API Reference

### PositionOverlay

```python
@dataclass
class PositionOverlay:
    """Single trajectory with optional trail."""
    data: NDArray[np.float64]           # (n_samples, n_dims)
    times: NDArray[np.float64] | None   # (n_samples,) or None
    color: str = "red"
    size: float = 10.0
    trail_length: int | None = None
```

### BodypartOverlay

```python
@dataclass
class BodypartOverlay:
    """Multi-keypoint pose with optional skeleton."""
    data: dict[str, NDArray[np.float64]]    # {name: (n_samples, n_dims)}
    times: NDArray[np.float64] | None       # (n_samples,) or None
    skeleton: list[tuple[str, str]] | None = None
    colors: dict[str, str] | None = None
    skeleton_color: str = "white"
    skeleton_width: float = 2.0
```

### HeadDirectionOverlay

```python
@dataclass
class HeadDirectionOverlay:
    """Heading as angles (rad) or unit vectors."""
    data: NDArray[np.float64]           # (n_samples,) or (n_samples, n_dims)
    times: NDArray[np.float64] | None   # (n_samples,) or None
    color: str = "yellow"
    length: float = 20.0
```

### VideoOverlay

```python
@dataclass
class VideoOverlay:
    """Background video layer with spatial calibration."""
    source: str | Path | NDArray[np.uint8]  # Path or (n_frames, H, W, 3)
    calibration: VideoCalibration | None = None
    times: NDArray[np.float64] | None = None
    alpha: float = 0.5
    z_order: Literal["below", "above"] = "above"
    crop: tuple[int, int, int, int] | None = None  # (x, y, w, h)
    downsample: int = 1
```

### VideoCalibration

```python
@dataclass(frozen=True)
class VideoCalibration:
    """Video→environment coordinate calibration."""
    transform_px_to_cm: Affine2D          # Pixel → cm transform
    frame_size_px: tuple[int, int]        # (width, height) in pixels

    @property
    def transform_cm_to_px(self) -> Affine2D:
        """Inverse transform (cm → pixel)."""

    @property
    def cm_per_px(self) -> float:
        """Approximate scale factor."""
```

### animate_fields() Signature

```python
def animate_fields(
    self,
    fields: NDArray[np.float64],
    *,
    backend: Literal["auto", "napari", "video", "html", "widget"] = "auto",
    save_path: str | None = None,
    fps: int = 30,
    overlays: list[PositionOverlay | BodypartOverlay | HeadDirectionOverlay | VideoOverlay] | None = None,
    frame_times: NDArray[np.float64] | None = None,
    show_regions: bool | list[str] = False,
    region_alpha: float = 0.3,
    # ... other parameters (cmap, vmin, vmax, title, etc.)
) -> Any:
    """Animate spatial fields with optional behavior overlays."""
```

**New parameters:**

- `overlays`: List of overlay instances to render
- `frame_times`: Explicit frame timestamps for temporal alignment (optional)
- `show_regions`: Display named regions (bool for all, list for specific)
- `region_alpha`: Transparency for region rendering (0-1)

## See Also

- [Field Animation](examples/16_field_animation.ipynb) - Core animation features
- [Regions](examples/04_regions_of_interest.ipynb) - Defining spatial zones
