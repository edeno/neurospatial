# Video Annotation

Define spatial environments and regions by annotating video frames directly using an interactive napari interface.

## Overview

The video annotation workflow allows you to:

1. **Draw environment boundaries** from video frames (e.g., arena walls)
2. **Define holes/obstacles** within the environment (excluded areas)
3. **Create named regions** for behavioral analysis (e.g., reward zones, nest areas)
4. **Apply calibration** to convert pixel coordinates to real-world units (cm)

This is particularly useful when you have tracking data and want to define the spatial structure from the same video used for tracking.

## Quick Start

```python
from neurospatial.annotation import annotate_video

# Launch interactive annotation
result = annotate_video("experiment.mp4", bin_size=2.0)

# Access results
env = result.environment  # Discretized Environment
regions = result.regions   # Named Regions
```

## Complete Workflow

### Step 1: Launch Annotation

```python
from neurospatial.annotation import annotate_video

# Basic annotation (pixel coordinates)
result = annotate_video(
    "experiment.mp4",
    frame_index=0,        # Which frame to display
    bin_size=2.0,         # Grid resolution for environment
    mode="both",          # Annotate boundary + regions
)
```

### Step 2: Draw Annotations

When the napari viewer opens:

1. **Draw the environment boundary** (cyan polygon):
   - Click points to define vertices
   - Press `Enter` to complete the polygon
   - This defines the spatial extent of your environment

2. **Press `M` to cycle to hole mode** (red polygon):
   - Draw any obstacles or excluded areas inside the boundary
   - Holes are subtracted from the environment

3. **Press `M` again for region mode** (yellow polygon):
   - Enter a name for each region before drawing
   - Draw polygons for reward zones, nest areas, etc.

4. **Press `Escape` or click "Save and Close"** to return results

### Step 3: Use the Results

```python
# The environment is ready to use
print(f"Environment has {env.n_bins} bins")

# Access regions
for name, region in result.regions.items():
    print(f"Region '{name}': {region.kind}")

# Use with tracking data
bin_indices = env.bin_sequence(positions)
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `M` | Cycle annotation mode (environment → hole → region) |
| `3` | Move shape mode |
| `4` | Edit vertices mode |
| `Delete` | Remove selected shape |
| `Escape` | Save and close viewer |
| `Ctrl+Z` | Undo last action |

## Adding Calibration

Convert pixel coordinates to real-world units (e.g., centimeters):

### Using Scale Bar

```python
from neurospatial.transforms import VideoCalibration, calibrate_from_scale_bar

# Two points on a known length in the video
point1_px = (100, 200)  # Start of scale bar (pixels)
point2_px = (300, 200)  # End of scale bar (pixels)
known_length_cm = 50.0  # Known length in cm
frame_size = (640, 480) # Video dimensions

transform = calibrate_from_scale_bar(
    point1_px, point2_px, known_length_cm, frame_size
)
calibration = VideoCalibration(transform, frame_size)

# Annotate with calibration
result = annotate_video(
    "experiment.mp4",
    calibration=calibration,
    bin_size=2.0,  # Now in cm!
)
```

### Using Landmark Correspondences

```python
from neurospatial.transforms import calibrate_from_landmarks, VideoCalibration
import numpy as np

# Known correspondences: video pixels → environment cm
landmarks_px = np.array([
    [50, 50],    # Top-left corner in pixels
    [590, 50],   # Top-right corner in pixels
    [590, 430],  # Bottom-right corner in pixels
    [50, 430],   # Bottom-left corner in pixels
])
landmarks_cm = np.array([
    [0, 0],      # Top-left in cm
    [100, 0],    # Top-right in cm
    [100, 80],   # Bottom-right in cm
    [0, 80],     # Bottom-left in cm
])

transform = calibrate_from_landmarks(
    landmarks_px, landmarks_cm, frame_size_px=(640, 480)
)
calibration = VideoCalibration(transform, (640, 480))

result = annotate_video(
    "experiment.mp4",
    calibration=calibration,
    bin_size=2.0,
)
```

## Simplifying Hand-Drawn Polygons

Hand-drawn polygons often have jagged edges. Use `simplify_tolerance` to smooth them:

```python
result = annotate_video(
    "experiment.mp4",
    bin_size=2.0,
    simplify_tolerance=1.0,  # Douglas-Peucker tolerance in output units
)
```

Higher values produce smoother polygons but may lose detail.

## Annotation Modes

Control what to annotate with the `mode` parameter:

```python
# Both boundary and regions (default)
result = annotate_video("video.mp4", mode="both", bin_size=2.0)

# Only environment boundary
result = annotate_video("video.mp4", mode="environment", bin_size=2.0)

# Only regions (no environment created)
result = annotate_video("video.mp4", mode="regions")
# Note: bin_size not required for regions-only mode
```

## Handling Multiple Boundaries

If you accidentally draw multiple environment boundaries, control the behavior:

```python
# Use the last drawn boundary (default)
result = annotate_video(
    "video.mp4",
    bin_size=2.0,
    multiple_boundaries="last",
)

# Use the first drawn boundary
result = annotate_video(
    "video.mp4",
    bin_size=2.0,
    multiple_boundaries="first",
)

# Raise error if multiple boundaries (strict mode)
result = annotate_video(
    "video.mp4",
    bin_size=2.0,
    multiple_boundaries="error",
)
```

## Editing Existing Regions

Provide existing regions to edit them:

```python
from neurospatial.regions import Regions, Region
from shapely.geometry import Polygon

# Create initial regions
initial = Regions([
    Region("reward_zone", "polygon", Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])),
])

# Edit them interactively
result = annotate_video(
    "video.mp4",
    initial_regions=initial,
    bin_size=2.0,
)
```

## Importing from External Tools

Import annotations created in other tools:

### LabelMe

```python
from neurospatial.annotation import regions_from_labelme

# Without calibration (pixel coordinates)
regions = regions_from_labelme("annotations.json")

# With calibration (cm coordinates)
regions = regions_from_labelme("annotations.json", calibration=calibration)
```

### CVAT

```python
from neurospatial.annotation import regions_from_cvat

regions = regions_from_cvat("cvat_export.xml", calibration=calibration)
```

## Role Types

Shapes are assigned roles that determine their purpose:

| Role | Color | Purpose |
|------|-------|---------|
| `environment` | Cyan | Primary boundary defining spatial extent |
| `hole` | Red | Excluded areas within the boundary |
| `region` | Yellow | Named regions of interest |

Access the role type in your code:

```python
from neurospatial.annotation import Role

# Type hint for role parameters
def process_annotation(role: Role) -> None:
    if role == "environment":
        # Handle boundary
        ...
```

## API Reference

### annotate_video

::: neurospatial.annotation.annotate_video
    options:
      show_root_heading: false
      heading_level: 4

### AnnotationResult

::: neurospatial.annotation.AnnotationResult
    options:
      show_root_heading: false
      heading_level: 4

### regions_from_labelme

::: neurospatial.annotation.regions_from_labelme
    options:
      show_root_heading: false
      heading_level: 4

### regions_from_cvat

::: neurospatial.annotation.regions_from_cvat
    options:
      show_root_heading: false
      heading_level: 4

## Common Issues

### "napari is not installed"

Install napari with:

```bash
pip install napari[all]
```

### Shapes don't appear

Ensure you're in polygon drawing mode (default). Press `3` to move shapes or `4` to edit vertices.

### Calibration appears wrong

Check that:

- Y-axis convention matches your data (scientific data typically uses Y-up)
- Scale factor is correct (verify by measuring known distances)
- Landmark correspondences are accurate

### Viewer closes unexpectedly

The viewer blocks until closed. If it closes without saving, check for Python errors in the console.

## See Also

- [Regions](regions.md): Managing regions programmatically
- [Environments](environments.md): Creating environments from other sources
- [Alignment & Transforms](alignment.md): Coordinate transformations
