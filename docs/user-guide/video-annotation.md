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

### Try the Sample Arena

From a checkout with the animation dependencies installed, this launcher opens
a synthetic arena with an obstacle and reward area. Run from the checkout
containing `examples/try_annotation.py`; if using a separate worktree, change
into that worktree first:

```bash
uv run --no-sync python examples/try_annotation.py --provider cpu
```

Click **Mark points**, then left-click the tan floor. An outline appears
automatically; add inside/outside clicks to refine it, then **Accept outline**.
Use **Paint / erase preview** for manual corrections. To compare providers on the
same frame, launch again with `--provider sam2` after the optional setup below.
Both providers run on the CPU by default; `--provider cpu` selects color region
growing, while `--provider sam2` selects the neural segmentation model.

Add `--image /path/to/frame.png` or `--video /path/to/movie.mp4 --frame-index 100`
to use your own recording. `--save-project arena.annotation.npz` enables saves
and autosave to a new project file. Add `--check` with either provider to run a
sample prediction without opening a viewer.

### Select and Correct an Outline

Open **Project and Assistance** and use **Assisted outline** to suggest a
boundary or region from clicks. The CPU provider works best with a distinctly
colored surface; SAM 2 can handle more complex scenes:

1. Click **Mark points**, then left-click inside the desired area. Add an inside
   point in each disconnected area that should be included.
2. Right-click to add outside points. Alt-click near a point to remove it, or use
   **Undo last point**. To move or select several points, use napari's point
   selection tool; **Delete selected** removes selected prompts. Click **Mark
   points** again to resume left/right clicking. Dragging still pans the image.
3. The outline updates automatically after each change. Keep clicking to refine
   it; the point tool stays active. The default CPU provider keeps areas connected
   to inside points whose colors fall within **Color tolerance**. Increase the
   tolerance to include a wider range of colors; this also updates the preview.
4. **Polygon detail** starts at **1.4 pixels** to reduce jagged edges and the
   number of vertices. Set it to **0** for the exact pixel outline; increasing
   the tolerance simplifies it further. The cyan paths show the exact polygons
   that will be accepted, including every hole. This
   updates immediately without rerunning inference. The vertex count helps you
   balance detail against editing complexity.
5. Optionally click **Paint / erase preview** to correct the mask with napari's
   painting tools. Painting foreground includes pixels; erasing excludes them.
   The polygon paths update after each stroke. Changing polygon detail preserves
   painted edits; further prompt or fragment-cleanup changes replace them.
6. Click **Accept outline** to add editable polygon geometry. Its holes and
   disconnected components are retained. Use **Discard outline** to remove a
   preview, or **Clear prompts** before trying a different selection. Accepting
   an outline clears its prompts so the next object starts with a fresh set.

**Remove stray fragments** is enabled by default. It keeps every connected
foreground region containing an inside point, including holes within those
regions, and removes disconnected pieces you have not marked. Add an inside
point to any separate region you want to retain. The status shows how many
fragments were removed. Turn the option off to see the original prediction;
switching it updates the preview without rerunning the model. If none of your
inside points fall within the predicted foreground, the preview is left intact
and the status asks you to refine it. Cleanup does not fill holes or bridge gaps.
Manual paint/erase corrections made afterward are retained when you accept.

The detail slider preserves holes and disconnected components; it does not
fill gaps or join fragments. Returning it to zero restores the exact current
mask outline. The accepted polygons and their pixel tolerance are retained by
project save/load and Undo/Redo. Any separately configured export simplification
still applies when producing the final result.

For a small arena within a larger image, expand **Region of interest**, click
**Draw region of interest**, and drag a rectangle around the inference area.
SAM then uses that crop for its image embedding; no image resizing is introduced
by the annotation tool. Keep every inside and outside point within the crop.
Move/remove outlying points or enlarge the crop if prompted. Use **Use full image**
to return to the full frame. An incomplete or deleted ROI requires a new rectangle
or an explicit return to the full image. Changing the ROI resets SAM refinement
and invalidates pending predictions. Accepted polygons always use original-frame
coordinates, so cropping does not change calibration or committed annotations.

Choose environment, hole, or region and a name in **Annotation Controls** before
accepting. An environment boundary must be connected; if the preview has several
components, refine it or accept those areas as named regions. Draw walls and
landmarks directly with their existing drawing tools. Replacing an existing
environment outline asks for confirmation.

Review the result against the arena walls and obstacles before clicking Done.
Color similarity can include shadows or nearby objects, so the editable preview
is part of the workflow. While a new prediction runs, the previous outline is
dimmed and cannot be accepted. Rapid clicks are combined into the latest request;
inference runs in the background. The first SAM 2 preview may take longer while
the model embeds the image. For manual updates, turn off **Update outline as I
click**, then click **Update outline** when ready. This works with both providers.

With SAM 2, adding an inside or outside click refines the last displayed model
prediction using its previous low-resolution logits. The first prediction offers
multiple candidates internally and selects the highest-scoring mask; later clicks
request a single refined mask. Moving or removing an existing point starts a fresh
prediction from the remaining prompts. **Clear prompts**, **Discard outline**,
accepting an outline, and inference errors also reset refinement. Repeating
**Update outline** with unchanged prompts reuses the same model result.

Refinement uses the model's raw prediction before fragment cleanup or manual
painting. Further clicks therefore replace manual preview edits. Temporary model
state is not saved in projects; resumed sessions start fresh. Accepted annotations
record the number of refinement steps in their metadata. Direct
`SegmentAnythingProvider.predict(image, prompt)` calls remain independent of
earlier prompts and only reuse image embeddings.

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

To prepare the sample launcher on a Mac, install into the current checkout's
virtual environment and download the default **SAM 2.1 Large** checkpoint
(about 900 MB) once. These commands use a pinned revision of Meta's source and
skip its optional CUDA extension:

```bash
SAM2_BUILD_CUDA=0 uv pip install --python .venv/bin/python 'sam-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4'
mkdir -p data/annotation_demo
curl --fail --location --output data/annotation_demo/sam2.1_hiera_large.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
uv run --no-sync python examples/try_annotation.py --provider sam2
```

The `data/` checkpoint is ignored by Git. Use `--checkpoint /path/to/model.pt`
and `--model-config configs/sam2.1/sam2.1_hiera_l.yaml` for a matching Large
checkpoint/config pair. Keep `--no-sync` when launching so uv retains these
optional packages, which are not part of the core dependency lockfile.

With `--provider sam2`, the launcher defaults to **SAM 2.1 Large**. For a shorter
initial load and lower memory use, download **Tiny** and choose it explicitly:

```bash
curl --fail --location --output data/annotation_demo/sam2.1_hiera_tiny.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
uv run --no-sync python examples/try_annotation.py --provider sam2 --sam2-model tiny
```

Add `--video /path/to/movie.mp4 --frame-index 100` to compare on your own recording
using the same frame and clicks. `--sam2-model` also accepts `small` and
`base-plus`, and selects the matching config and local checkpoint filename.
The launcher prints the config and checkpoint it actually loads. Image embeddings
are reused between clicks, so a larger model's initial image processing can take
longer than subsequent outline updates. `--checkpoint` and `--model-config` still
override the selected defaults; weights are never downloaded by the launcher.
Without `--provider sam2`, the launcher uses CPU color region growing and needs
no SAM checkpoint. Both providers start the polygon preview at **1.4 pixels**
of simplification; set **Polygon detail** to **0** for an exact mask outline.

Load the provider **before** opening the editor so dependency, config or
checkpoint errors are reported first. Supply an existing local checkpoint;
this factory does not download weights. The default device is `"cpu"`.

<!-- docs-test: skip optional SAM 2 installation, local checkpoint and interactive viewer -->
```python
from neurospatial.annotation import SegmentAnythingProvider, annotate_video

provider = SegmentAnythingProvider.from_sam2(
    "configs/sam2.1/sam2.1_hiera_l.yaml",
    "/path/to/sam2.1_hiera_large.pt",
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

### Design Credit and References

We credit [CVAT](https://github.com/cvat-ai/cvat), CVAT.ai, and its contributors
for the interaction design that inspired our assisted outlining: inside/outside
clicks, live mask-to-polygon previews with adjustable detail, and an optional
inference region of interest. The following links pin the public source reviewed
for this integration so it remains traceable as upstream develops:

- [Prompting and ROI workflow](https://github.com/cvat-ai/cvat/blob/172ea51a2ffc11adb08b1c9fb757282d142cd810/site/content/en/docs/annotation/auto-annotation/ai-tools.md): user interactions and polygon-detail controls.
- [Annotation interaction controls](https://github.com/cvat-ai/cvat/blob/172ea51a2ffc11adb08b1c9fb757282d142cd810/cvat-ui/src/components/annotation-page/standard-workspace/controls-side-bar/tools-control.tsx): prompt requests, live previews, ROI submission, and contour selection.
- [Polygon conversion](https://github.com/cvat-ai/cvat/blob/172ea51a2ffc11adb08b1c9fb757282d142cd810/cvat-core/src/opencv/opencv-interface.ts) and [default settings](https://github.com/cvat-ai/cvat/blob/172ea51a2ffc11adb08b1c9fb757282d142cd810/cvat-ui/src/reducers/settings-reducer.ts): contour extraction and the accuracy-to-pixel-tolerance mapping.
- [SAM 2 interactor](https://github.com/cvat-ai/cvat-models/blob/d2ebbc7f37d9c002d74fa8d392e9f3f8cb3aa419/interactor/sam2/func.py) in [cvat-models](https://github.com/cvat-ai/cvat-models): model prompting and image-embedding reuse.

CVAT's referenced default accuracy of 9 maps to `10 / 7` pixels (about 1.43);
our **1.4 px** default rounds that to the slider's 0.1 px steps. Neurospatial's
polygon conversion preserves holes and every retained component, without adding
dilation. Our subsequent SAM clicks also use previous-logit refinement. These
choices mean the implementations can produce different outlines from the same
prompts; the references do not establish CVAT Online's deployed model settings.

## Track Graph Annotation

Use `annotate_track_graph()` to place nodes and connect them into a track.
Native edge deletion and node renaming participate in undo and redo. To change
an edge's geometry, move its endpoint nodes.

Preview 1D uses the same node coordinates, edge order and gaps as the returned
`TrackGraphResult`. Gaps are in centimeters with calibration and pixels without
it; the spacing control displays the unit. Automatic gaps follow the selected
edge order, and disconnected components are included in the result. Adding a
node keeps a manual edge order and gaps; adding or deleting edges, or deleting
nodes, returns to the automatic layout.

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

### Calibrate in the Editor

Expand **Calibration and output settings** below the outlining controls in
**Project and Assistance**. Calibration is optional for drawing outlines; use it
when you need output in centimeters. Click **Mark two scale endpoints** and mark the
ends of an object of known length. Enter that length in centimeters, then click
**Apply calibration**. **Output bin size** is converted to cm, so the grid keeps
its size in the image: an 8 pixel bin at 0.3 cm per pixel becomes 2.4 cm. Enter
a new value if you want a different grid. Shapes stay aligned with the
reference frame because editable geometry remains in pixels. **Clear
calibration** returns output units to pixels and converts the bin size back;
these changes can be undone and are saved in the project.

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

Edit these rings directly, or draw new shapes for additional regions. A pasted
copy of a polygon keeps its own holes and is independent of the original. A hole
pasted without its polygon becomes a separate polygon. Polygons that are not
valid, such as a self-intersecting outline from a LabelMe file, open for repair;
saving reports the problem until it is fixed.

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

The viewer blocks until closed, including in IPython with `%gui qt`. If it
closes without saving, check for Python errors in the console.

## See Also

- [Regions](regions.md): Managing regions programmatically
- [Environments](environments.md): Creating environments from other sources
- [Alignment & Transforms](alignment.md): Coordinate transformations
