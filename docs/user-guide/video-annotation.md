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
7. **Save an editable project** and resume from its embedded reference frame
8. **Suggest an outline** from inside/outside clicks, then correct its preview

This is particularly useful when you have tracking data and want to define the spatial structure from the same video used for tracking.

## Quick Start

<!-- docs-test: skip interactive viewer and user video -->
```python
from neurospatial.annotation import annotate_video

# Launch interactive annotation
result = annotate_video(
    "experiment.mp4",
    bin_size=2.0,
    project_path="arena.annotation.npz",
)

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

6. **Click "Done"** to validate the annotations and return results

Done validates the complete result before closing the viewer. If a boundary,
region, wall or landmark is invalid, the viewer stays open and explains what
to correct. Incomplete shapes are kept for correction as well. Edit the affected
vertices, then click Done again. `Escape` keeps napari's normal finish/cancel
behavior while drawing; it does not finish the annotation session.

**Save Project** saves work in progress without requiring a valid final
boundary. Use it before taking a break, even if a polygon is unfinished.

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
| `Escape` | Finish/cancel the current napari drawing operation |
| `Ctrl`/`Cmd` + `S` | Save Project |
| `Ctrl`/`Cmd` + `Z` | Undo |
| Platform redo shortcut | Redo (typically `Ctrl` + `Y` or `Cmd` + `Shift` + `Z`) |

The dock also has Undo and Redo buttons. Accepted assisted outlines and
calibration changes participate in the same history as hand-drawn shapes.

## Save and Resume a Project

Pass `project_path=` to keep an editable draft alongside the exported analysis
result. A project contains the RGB reference frame, pixel coordinates and names
of shapes, their roles and hole associations, calibration, and output settings.
It can retain incomplete or invalid geometry for later correction.

<!-- docs-test: skip interactive viewer and user video -->
```python
from neurospatial.annotation import annotate_project, annotate_video

result = annotate_video(
    "experiment.mp4",
    bin_size=2.0,
    project_path="arena.annotation.npz",
    autosave_interval=30.0,
)

# Resume later, using the reference frame stored in the project.
result = annotate_project(
    "arena.annotation.npz",
    project_path="arena.annotation.npz",
)
```

Use **Save Project** or `Ctrl`/`Cmd` + `S` for an immediate draft save. If no
path has been chosen, Save Project asks for one. Autosave runs every 30 seconds
once a destination is available; set `autosave_interval=None` to disable it.
Draft files are written atomically, so a failed save does not replace the
previous file. A resumed project uses its embedded image even if the original
video has moved. Click **Done** when the draft is ready to export an
`AnnotationResult`.

Done and the window close button both require valid output. Saving an invalid
draft keeps the editor open; it does not export a partial analysis result.

Start `annotate_video` with a new project path; if the file already exists, use
`annotate_project` to resume it. When resuming a path, draft saves default to
that same file; set `project_path=` to save to another destination.

You can also inspect or prepare drafts without launching a viewer. The example
below uses a synthetic frame; replace it with your reference RGB image and
pixel vertices in your own work.

<!-- docs-test: run -->
```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from neurospatial.annotation import AnnotationProject, AnnotationShape

frame = np.zeros((24, 32, 3), dtype=np.uint8)
boundary = AnnotationShape(
    np.array([[2, 2], [29, 2], [29, 21], [2, 21]], dtype=float),
    role="environment",
    name="arena",
)
draft = AnnotationProject(frame, shapes=[boundary], bin_size=2.0)

with TemporaryDirectory() as directory:
    path = Path(directory) / "arena.annotation.npz"
    draft.save(path)
    restored = AnnotationProject.load(path)
    assert np.array_equal(restored.frame, frame)
    assert restored.shapes[0].identifier == boundary.identifier
```

`annotate_project` also accepts an `AnnotationProject` object directly. Project
coordinates stay in pixels; calibration is applied when producing the final
environment, regions, walls and landmarks.

## Assisted Outlines

Open **Project and Assistance** and use **Assisted outline** when the arena
surface has a distinct color:

1. Click **Mark inside points** and place at least one point inside the desired
   area. Add a point in each disconnected area that should be included.
2. Optionally click **Mark outside points** and mark colors to exclude.
3. Click **Suggest outline**. The default CPU provider keeps areas connected to
   inside points whose colors fall within **Color tolerance**. Increase the
   tolerance to include a wider range of colors.
4. Select the preview labels layer and use napari's paint or erase tools to
   correct the mask. Painting foreground includes pixels; erasing excludes them.
5. Click **Accept outline** to add editable polygon geometry. Its holes and
   disconnected components are retained. Use **Discard outline** to remove a
   preview, or **Clear prompts** before trying a different selection.

Choose environment, hole, or region and a name in **Annotation Controls** before
accepting. An environment boundary must be connected; if the preview has several
components, refine it or accept those areas as named regions. Draw walls and
landmarks directly with their existing drawing tools. Replacing an existing
environment outline asks for confirmation.

Review the result against the arena walls and obstacles before clicking Done.
Color similarity can include shadows or nearby objects, so the editable preview
is part of the workflow. Changing prompts or tolerance invalidates an old
preview; request another suggestion before accepting it.

The same segmentation tools work without napari. Prompts use pixel `(x, y)`
coordinates, while masks use `[row, column]` indexing. Polygon boundaries follow
pixel edges: a pixel centered at `(x, y)` covers a unit square extending half a
pixel in each direction. The conversion preserves exact mask area, holes and
every component when `simplify_tolerance=0`.

<!-- docs-test: run -->
```python
import numpy as np

from neurospatial.annotation import (
    RegionGrowProvider,
    SegmentationPrompt,
    mask_to_polygons,
)

image = np.zeros((12, 18, 3), dtype=np.uint8)
image[2:10, 5:15] = [180, 170, 140]  # Arena surface
image[4:7, 8:11] = 0                # Obstacle inside it
prompt = SegmentationPrompt(
    points=np.array([[6.0, 3.0]]),
    labels=np.array([1]),  # 1 = inside, 0 = outside
)
proposal = RegionGrowProvider(tolerance=0.1).predict(image, prompt)
polygons = mask_to_polygons(proposal.mask)
assert len(polygons) == 1
assert len(polygons[0].interiors) == 1
assert polygons[0].area == proposal.mask.sum()
```

`SegmentationProposal` owns a read-only boolean mask, an optional score, and
JSON provenance. A custom `SegmentationProvider` implements
`predict(image, prompt)` and returns a proposal. Pass it as
`segmentation_provider=` to `annotate_video` or `annotate_project`.

### Optional SAM 2 Provider

`SegmentAnythingProvider` can use Meta's SAM 2 image predictor for scenes where
color alone gives a poor outline. Install PyTorch and SAM 2 separately by
following [Meta's installation instructions](https://github.com/facebookresearch/sam2#installation),
and obtain a matching model checkpoint from that repository. These model
dependencies and weights are optional; the default CPU provider needs neither.

Load the provider **before** opening the editor so dependency, config or
checkpoint errors are reported first. Supply an existing local checkpoint;
this factory does not download weights. The default device is `"cpu"`.

<!-- docs-test: skip optional SAM 2 installation, local checkpoint and interactive viewer -->
```python
from neurospatial.annotation import SegmentAnythingProvider, annotate_video

provider = SegmentAnythingProvider.from_sam2(
    "configs/sam2.1/sam2.1_hiera_t.yaml",
    "/path/to/sam2.1_hiera_tiny.pt",
    device="cpu",
)
result = annotate_video(
    "experiment.mp4",
    bin_size=2.0,
    project_path="arena.annotation.npz",
    segmentation_provider=provider,
)
```

You can instead wrap an already constructed SAM2-style image predictor with
`SegmentAnythingProvider(predictor)`. Give the adapter exclusive use of that
predictor: it caches embeddings for unchanged image content and selects the
highest-scoring mask. The integration is tested with mocked predictors;
running real SAM 2 weights requires your separately installed model setup.

## Track Graph Annotation

Use `annotate_track_graph()` to place nodes and connect them into a track.
Native edge deletion and node renaming participate in undo and redo. To change
an edge's geometry, move its endpoint nodes.

Preview 1D uses the same node coordinates, edge order and gaps as the returned
`TrackGraphResult`. Gaps are in centimeters with calibration and pixels without
it; the spacing control displays the unit. Automatic gaps follow the selected
edge order, and disconnected components are included in the result. Saving or
closing validates the draft before the editor closes.

## Adding Calibration

Convert pixel coordinates to real-world units (e.g., centimeters):

The resulting environment records `units="cm"` when calibration is supplied,
or `units="pixels"` otherwise. `bin_size` and `simplify_tolerance` use those
same units.

### Calibrate in the Editor

Under **Calibration** in **Project and Assistance**, click **Mark two scale endpoints** and mark the
ends of an object of known length. Enter that length in centimeters, then click
**Apply calibration**. Check **Output bin size** after applying the scale: its
unit changes to cm, and its numeric value remains your explicit choice. Shapes
stay aligned with the reference frame because editable geometry remains in
pixels. **Clear calibration** returns output units to pixels; these changes
can be undone and are saved in the project.

### Using Scale Bar

```python
from neurospatial.ops import VideoCalibration, calibrate_from_scale_bar

# Two points on a known length in the video
point1_px = (100, 200)  # Start of scale bar (pixels)
point2_px = (300, 200)  # End of scale bar (pixels)
known_length_cm = 50.0  # Known length in cm
frame_size = (640, 480) # Video dimensions

transform = calibrate_from_scale_bar(
    point1_px, point2_px, known_length_cm, frame_size_px=frame_size
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
- [`annotate_project()`](../api/neurospatial/annotation/index.md#neurospatial.annotation.annotate_project)
- [`AnnotationProject`](../api/neurospatial/annotation/project.md#neurospatial.annotation.project.AnnotationProject)
- [`AnnotationShape`](../api/neurospatial/annotation/project.md#neurospatial.annotation.project.AnnotationShape)
- [`RegionGrowProvider`](../api/neurospatial/annotation/segmentation.md#neurospatial.annotation.segmentation.RegionGrowProvider)
- [`SegmentAnythingProvider`](../api/neurospatial/annotation/segmentation.md#neurospatial.annotation.segmentation.SegmentAnythingProvider)
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

The viewer blocks until closed. If it closes without saving, check for Python errors in the console.

## See Also

- [Regions](regions.md): Managing regions programmatically
- [Environments](environments.md): Creating environments from other sources
- [Alignment & Transforms](alignment.md): Coordinate transformations
