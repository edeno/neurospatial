# Video Annotation

Define spatial environments and regions by annotating video frames directly using an interactive napari interface.

## Overview

The video annotation workflow allows you to:

1. **Draw environment boundaries** from video frames (e.g., arena walls)
2. **Define holes/obstacles** within the environment (excluded areas)
3. **Create named regions** for behavioral analysis (e.g., reward zones, nest areas)
4. **Draw named walls** that block movement, such as a divider or a barrier
5. **Place named landmarks**: reward ports, cue cards, monitors and objects
6. **Apply calibration** to convert pixel coordinates to real-world units (cm)

This is particularly useful when you have tracking data and want to define the spatial structure from the same video used for tracking.

## Quick Start

```python
from neurospatial.annotation import annotate_video

# Launch interactive annotation
result = annotate_video("experiment.mp4", bin_size=2.0)

# Access results
env = result.environment  # Discretized Environment, holding all of the below
regions = result.regions   # Named Regions
walls = result.walls       # Named walls, {name: LineString}
landmarks = result.landmarks  # Named Landmarks
env.plot()                 # Check the result against the video frame
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

4. **Press `M` again for wall mode** (magenta path):
   - The tool switches to paths: click along the wall, press `Enter` to finish
   - Name each wall, such as `divider`; analyses that report blocked
     connections name the wall that closes them
   - A wall is clipped to the boundary, so it may start on or just past the
     boundary's edge

5. **Press `M` again for landmark mode** (orange):
   - Name the landmark, then choose the napari tool by what it is: an
     **ellipse** around a reward port or cue marks a point at its center, a
     **path** along a monitor or a card marks a line, and a **polygon** or
     **rectangle** around an object marks its area
   - Landmarks may lie outside the boundary, as a cue on the room wall does

6. **Press `Escape` or click "Save and Close"** to return results

Saving validates the complete result before closing the viewer. If a boundary,
region, wall or landmark is invalid, the viewer stays open and explains what
to correct. Incomplete shapes are kept for correction as well. Edit the affected
vertices, then save again.

### Step 3: Use the Results

<!-- docs-test: run setup=docs_video_annotation_use_results -->
```python
# The environment is ready to use
print(f"Environment has {env.n_bins} bins")

# Access regions
for name, region in result.regions.items():
    print(f"Region '{name}': {region.kind}")

# Walls and landmarks are already part of the environment
print(env.arena.wall_names, list(env.landmarks))

# Use with tracking data
bin_indices = env.bin_sequence(times, positions)
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `M` | Cycle annotation mode (environment → hole → region → wall → landmark) |
| `3` | Move shape mode |
| `4` | Edit vertices mode |
| `Delete` | Remove selected shape |
| `Escape` | Save and close viewer |

## Track Graph Annotation

Use `annotate_track_graph()` to place nodes and connect them into a track.
Native edge deletion and node renaming participate in undo and redo. To change
an edge's geometry, move its endpoint nodes.

Preview 1D uses the same node coordinates, edge order and gaps as the returned
`TrackGraphResult`. Gaps are in centimeters with calibration and pixels without
it; the spacing control displays the unit. Automatic gaps follow the selected
edge order, and disconnected components are included in the result.

Saving or closing either editor validates the draft before the editor closes.
If it cannot be saved, a dialog says why: choose "Keep Editing" to fix it, or
"Discard and Close" to close without a result, which raises a `RuntimeError`.
Warnings about the result, such as an ignored boundary, print after the editor
closes.

## Adding Calibration

Convert pixel coordinates to real-world units (e.g., centimeters):

The resulting environment records `units="cm"` when calibration is supplied,
or `units="pixels"` otherwise. `bin_size` and `simplify_tolerance` use those
same units.

### Using Scale Bar

```python
from neurospatial.ops import VideoCalibration, calibrate_from_scale_bar

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
import numpy as np

from neurospatial.ops import VideoCalibration, calibrate_from_landmarks

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

Simplification does not repair self-intersections. Correct intersecting edges
in the viewer before saving so that the intended area is unambiguous.

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

Interior holes are retained when reopening either regions or an initial
environment boundary. Each hole remains associated with its original polygon;
a hole in a region does not remove that area from the environment.

Edit these rings directly, or draw new shapes for additional regions.
Copy/paste of reopened polygon rings is rejected because their parent
associations would otherwise be ambiguous.

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

Both import adapters retain coordinate units in each region's
`metadata["coord_system"]`: `"cm"` with calibration and `"pixels"` without it.
This metadata is preserved when constructing an environment from the boundary.

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

## Region Types

Shapes are assigned region types that determine their purpose:

| Region type | Color | Purpose |
|-------------|-------|---------|
| `environment` | Cyan | Primary boundary defining spatial extent |
| `hole` | Red | Excluded areas within the boundary |
| `region` | Yellow | Named regions of interest |
| `wall` | Magenta | Named physical walls, drawn as paths |
| `landmark` | Orange | Named cues, monitors and objects: ellipse = point, path = line, polygon = area |

Access the region type in your code:

```python
from neurospatial.annotation import RegionType

# Type hint for region-type parameters
def process_annotation(role: RegionType) -> None:
    if role == "environment":
        # Handle boundary
        ...
```

## API Reference

- [`annotate_video()`](../api/neurospatial/annotation/index.md#neurospatial.annotation.annotate_video)
- [`AnnotationResult`](../api/neurospatial/annotation/index.md#neurospatial.annotation.AnnotationResult)
- [`regions_from_labelme()`](../api/neurospatial/annotation/index.md#neurospatial.annotation.regions_from_labelme)
- [`regions_from_cvat()`](../api/neurospatial/annotation/index.md#neurospatial.annotation.regions_from_cvat)

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

The viewer blocks until closed, including in IPython with `%gui qt`. If it
closes without saving, check for Python errors in the console.

## See Also

- [Regions](regions.md): Managing regions programmatically
- [Environments](environments.md): Creating environments from other sources
- [Alignment & Transforms](alignment.md): Coordinate transformations
