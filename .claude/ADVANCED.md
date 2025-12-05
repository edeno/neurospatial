# Advanced Topics

Advanced features and integrations.

---

## NWB Integration (v0.7.0+)

Full reference for NeurodataWithoutBorders format support.

### Installation

```bash
# Basic NWB support
pip install neurospatial[nwb]

# Full support (pose estimation, events)
pip install neurospatial[nwb-full]

# Or with uv
uv add neurospatial[nwb-full]
```

### Reading from NWB

```python
from pynwb import NWBHDF5IO
from neurospatial.nwb import (
    read_position,
    read_head_direction,
    read_pose,
    read_events,
    read_trials,
    read_environment,
)

with NWBHDF5IO("session.nwb", "r") as io:
    nwbfile = io.read()

    # Position data
    positions, timestamps = read_position(nwbfile)

    # Head direction
    angles, timestamps = read_head_direction(nwbfile)

    # Pose estimation (requires ndx-pose)
    bodyparts, timestamps, skeleton = read_pose(nwbfile, pose_estimation_name="DLC")

    # Events (requires ndx-events)
    events_df = read_events(nwbfile, events_name="licks")

    # Trials
    trials_df = read_trials(nwbfile)

    # Environment (from scratch space)
    env = read_environment(nwbfile, name="linear_track")
```

### Writing to NWB

```python
from neurospatial.nwb import (
    write_place_field,
    write_occupancy,
    write_trials,
    write_laps,
    write_region_crossings,
    write_environment,
)

with NWBHDF5IO("session.nwb", "r+") as io:
    nwbfile = io.read()

    # Place field (to analysis/)
    write_place_field(nwbfile, env, place_field, name="cell_001")

    # Occupancy (to analysis/)
    write_occupancy(nwbfile, env, occupancy, unit="seconds")

    # Trials (to intervals/trials/)
    write_trials(nwbfile, trials, overwrite=True)

    # Laps (to processing/behavior/)
    write_laps(nwbfile, lap_times, lap_types=lap_directions)

    # Region crossings (to processing/behavior/)
    write_region_crossings(
        nwbfile, crossing_times,
        region_names=["goal", "start"],
        event_types=["enter", "exit"]
    )

    # Environment (to scratch/)
    write_environment(nwbfile, env, name="my_environment")

    io.write(nwbfile)
```

### Factory Functions

```python
from neurospatial.nwb import (
    environment_from_position,
    position_overlay_from_nwb,
    bodypart_overlay_from_nwb,
    head_direction_overlay_from_nwb,
)

with NWBHDF5IO("session.nwb", "r") as io:
    nwbfile = io.read()

    # Create environment from position data
    env = environment_from_position(nwbfile, bin_size=2.0, units="cm")

    # Create overlays for animation
    position_overlay = position_overlay_from_nwb(nwbfile, color="red", trail_length=10)
    bodypart_overlay = bodypart_overlay_from_nwb(nwbfile, pose_estimation_name="DLC")
    head_direction = head_direction_overlay_from_nwb(nwbfile, color="yellow")
```

### NWB Data Locations

| Data Type | NWB Location | Function |
|-----------|--------------|----------|
| Place fields | `analysis/` | `write_place_field()` |
| Occupancy | `analysis/` | `write_occupancy()` |
| Trials | `intervals/trials/` | `write_trials()` |
| Lap events | `processing/behavior/` | `write_laps()` |
| Region crossings | `processing/behavior/` | `write_region_crossings()` |
| Environment | `scratch/` | `write_environment()` |

### NWB Dependencies

| Extra | Packages | Use Case |
|-------|----------|----------|
| `nwb` | pynwb, hdmf | Basic NWB support |
| `nwb-pose` | pynwb, ndx-pose | Pose estimation data |
| `nwb-events` | pynwb, ndx-events | EventsTable support |
| `nwb-full` | All above | Full NWB support |

### Environment Round-Trip

**What's preserved:**

- ✓ `bin_centers` (exact)
- ✓ `connectivity` graph structure and edge weights
- ✓ `dimension_ranges`
- ✓ `units` and `frame` metadata
- ✓ Regions (points and polygons)

**Not preserved:**

- ✗ Original layout engine type (reconstructed as generic layout)
- ✗ Grid-specific metadata (`grid_shape`, `grid_edges`, `active_mask`)
- ✗ Layout engine's `is_1d` property (always `False` for reconstructed)

Spatial queries still work identically after round-trip.

---

## Video Overlay (v0.5.0+)

Display recorded video behind or above spatial fields.

### Video Calibration

```python
from neurospatial.animation import calibrate_video, VideoOverlay
from neurospatial.transforms import VideoCalibration, calibrate_from_landmarks

# Method 1: Scale bar calibration (recommended)
calibration = calibrate_video(
    "session.mp4",
    env,
    scale_bar=((100, 200), (300, 200), 50.0),  # Two points + known length (cm)
)

# Method 2: Landmark correspondences
corners_px = np.array([[50, 50], [590, 50], [590, 430], [50, 430]])  # Video pixels
corners_env = np.array([[0, 0], [100, 0], [100, 80], [0, 80]])       # Environment cm
calibration = calibrate_video(
    "session.mp4",
    env,
    landmarks_px=corners_px,
    landmarks_env=corners_env,
)

# Method 3: Direct scale factor (most common)
calibration = calibrate_video("session.mp4", env, cm_per_px=0.25)

# If overlay appears inverted, toggle flip_y
calibration = calibrate_video("session.mp4", env, cm_per_px=0.25, flip_y=False)
```

### Video Overlay Usage

```python
# Create video overlay
video_overlay = VideoOverlay(
    source="experiment.mp4",     # Or pre-loaded array (n_frames, H, W, 3)
    calibration=calib,           # Pixel → cm transform
    alpha=0.5,                   # 50% blend (default)
    z_order="above",             # Render on top of field (default)
)
env.animate_fields(fields, frame_times=frame_times, overlays=[video_overlay], backend="napari")
```

### Alpha Blending Control

Adjust `alpha` to control video/field balance:

| Goal | Settings | Result |
|------|----------|--------|
| Balanced view | `alpha=0.5, z_order="above"` | Equal video/field visibility (default) |
| Field dominant | `alpha=0.3, z_order="above"` | Field shows through video |
| Video dominant | `alpha=0.7, z_order="above"` | Video shows through field |
| Video as background | `z_order="below"` | Only works if field has transparent regions |

### Coordinate Conventions

| Environment Units | `flip_y` | When to Use |
|-------------------|----------|-------------|
| cm (most common) | `True` (default) | Scientific tracking (DeepLabCut, SLEAP) |
| meters | `True` (default) | Same as cm, convert: `cm_per_px = m_per_px * 100` |
| pixels | `False` | Environment already in image coordinates |

The video itself validates calibration - if inverted, toggle `flip_y`.

---

## Video Annotation (v0.6.0+)

Interactive annotation of environment boundaries and regions.

### Basic Annotation

```python
from neurospatial import annotate_video
from neurospatial.annotation import boundary_from_positions, BoundaryConfig

# Interactive annotation - draw environment boundary and regions
result = annotate_video("experiment.mp4", bin_size=2.0)
env = result.environment  # Environment from boundary polygon
regions = result.regions   # Named regions

# With calibration (pixel -> cm coordinates)
from neurospatial.transforms import VideoCalibration, calibrate_from_scale_bar
transform = calibrate_from_scale_bar((0, 0), (200, 0), 100.0, (640, 480))
calib = VideoCalibration(transform, (640, 480))
result = annotate_video("experiment.mp4", calibration=calib, bin_size=2.0)
```

### Annotation Modes

```python
# Default: boundary + regions
result = annotate_video("experiment.mp4", mode="both", bin_size=2.0)

# Only boundary
result = annotate_video("experiment.mp4", mode="environment", bin_size=2.0)

# Only regions (no bin_size needed)
result = annotate_video("experiment.mp4", mode="regions")
```

### Boundary Seeding from Position Data

```python
# Pre-draw boundary from trajectory - edit rather than draw from scratch
result = annotate_video(
    "experiment.mp4",
    bin_size=2.0,
    initial_boundary=positions,  # Auto-infer with alpha_shape
    show_positions=True,         # Semi-transparent cyan Points layer
)

# With config for fine-tuning
config = BoundaryConfig(method="convex_hull", buffer_fraction=0.05)
result = annotate_video(
    "experiment.mp4",
    bin_size=2.0,
    initial_boundary=positions,
    boundary_config=config,
)

# Composable: create boundary explicitly
boundary = boundary_from_positions(
    positions,
    method="alpha_shape",  # or "convex_hull"
    alpha=0.05,
    buffer_fraction=0.03,
)
result = annotate_video("experiment.mp4", bin_size=2.0, initial_boundary=boundary)
```

**Boundary inference methods:**

- `"alpha_shape"` (default): Concave hull, tighter fit (requires `pip install alphashape`)
- `"convex_hull"`: Fast, always convex, no extra dependencies

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `E` | Set mode to draw environment boundary (cyan) |
| `R` | Set mode to draw named region (yellow) |
| `3` | Move shape mode |
| `4` | Edit vertices mode |
| `Delete` | Remove selected shape |

### Import from External Tools

```python
from neurospatial import regions_from_labelme, regions_from_cvat

# Import from LabelMe
regions = regions_from_labelme("labelme_export.json", calibration=calib)

# Import from CVAT
regions = regions_from_cvat("cvat_export.xml", calibration=calib)
```

---

## Track Graph Annotation (v0.9.0+)

Interactive annotation for 1D linearized environments (T-maze, linear track, etc.).

### Basic Usage

```python
from neurospatial.annotation import annotate_track_graph

# From video file - opens napari for interactive annotation
result = annotate_track_graph("maze.mp4")

# From static image
import matplotlib.pyplot as plt
img = plt.imread("track_photo.png")
result = annotate_track_graph(image=img)

# With calibration (convert pixels to cm)
from neurospatial.transforms import VideoCalibration, calibrate_from_scale_bar
transform = calibrate_from_scale_bar((0, 0), (200, 0), 100.0, (640, 480))
calib = VideoCalibration(transform, (640, 480))
result = annotate_track_graph("maze.mp4", calibration=calib)
```

### Result Contents

```python
# Result contains everything needed for Environment.from_graph()
print(result.track_graph)      # NetworkX Graph with 'pos', 'distance', 'edge_id'
print(result.node_positions)   # List of (x, y) tuples (cm if calibrated)
print(result.edges)            # Edge list as (node_i, node_j) tuples
print(result.edge_order)       # Ordered edges for linearization
print(result.edge_spacing)     # Spacing between edges
print(result.pixel_positions)  # Original pixel coordinates (preserved)

# Create 1D linearized environment from result
env = result.to_environment(bin_size=2.0)
print(env.is_1d)  # True - ready for linearization

# Override edge spacing if needed
env = result.to_environment(bin_size=2.0, edge_spacing=[0.0, 5.0, 0.0])
```

### Editing Existing Track Graphs

```python
result = annotate_track_graph(
    "maze.mp4",
    initial_nodes=np.array([[100, 200], [300, 200], [300, 400]]),
    initial_edges=[(0, 1), (1, 2)],
    initial_node_labels=["start", "junction", "goal"],
)
```

### Workflow Example

```python
# Annotate → create environment → analyze
result = annotate_track_graph("linear_track.mp4", calibration=calib)
env = result.to_environment(bin_size=2.0, name="linear_track")
env.units = "cm"

# Add regions from node labels
for i, label in enumerate(result.node_labels):
    if label:  # Non-empty label
        env.regions.add(label, point=result.node_positions[i])
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `E` | Switch to Add Edge mode (click two nodes to connect) |
| `Shift+S` | Set selected node as start node (select node first with `3`) |
| `Ctrl+Z` | Undo last action |
| `Ctrl+Shift+Z` | Redo |
| `Escape` | Cancel edge creation in progress |

**Note**: Use napari's native layer shortcuts for node manipulation:

- Press `2` for Add mode (click to add nodes)
- Press `3` for Select mode (click to select nodes)

### UI Components

- **Mode Selector**: Radio buttons for Add Node / Add Edge / Delete modes
- **Node List**: Shows all nodes with labels and start marker (★)
- **Edge List**: Shows all edges as (node_i, node_j)
- **Edge Order**: Reorderable list with Move Up/Down/Reset buttons
- **Edge Spacing**: Input field to set gap between edges
- **Preview Button**: Shows 1D linearization preview (matplotlib)
- **Validation Status**: Green (valid), Orange (warnings), Red (errors)
- **Save Button**: Validates and closes the viewer

---

## Gaze Analysis and Visibility (v0.19.0+)

Advanced analysis of what the animal can see and where it's looking.

### Field of View and Visibility

```python
from neurospatial import (
    FieldOfView,
    compute_viewshed,
    compute_view_field,
    visible_cues,
    visibility_occupancy,
)

# Species-specific field of view presets
fov_rat = FieldOfView.rat()        # ~320° total, ~40° binocular
fov_mouse = FieldOfView.mouse()    # ~310° total
fov_primate = FieldOfView.primate()  # ~180° total, ~120° binocular

# Custom FOV
fov_custom = FieldOfView.symmetric(total_angle=270.0)  # Degrees

# FOV properties
print(f"Total angle: {fov_rat.total_angle_degrees:.0f}°")
print(f"Binocular region: {fov_rat.binocular_half_angle * 2 * 180 / np.pi:.0f}°")
print(f"Blind spot behind: {fov_rat.blind_spot_behind * 180 / np.pi:.0f}°")

# Check if direction is in FOV
in_fov = fov_rat.contains_angle(bearing)  # bearing relative to heading
```

### Viewshed Analysis (What's Visible)

```python
# Compute visible bins from a position
viewshed = compute_viewshed(
    env,
    position=np.array([50.0, 50.0]),
    heading=0.0,  # Facing East
    fov=fov_rat,  # Species-specific FOV
    n_rays=360,   # Angular resolution
)

# Access results
print(f"Visible bins: {viewshed.n_visible_bins}")
print(f"Visibility fraction: {viewshed.visibility_fraction:.1%}")
visible_centers = viewshed.visible_bin_centers(env)

# Occlusion map (which bins block line of sight)
occlusion = viewshed.occlusion_map  # Shape: (n_bins,)
```

### Cue/Landmark Visibility

```python
# Check which cues/landmarks are visible from position
cue_positions = np.array([[80, 50], [20, 80], [50, 90]])

viewshed = compute_viewshed(
    env, position=np.array([50, 50]),
    heading=np.pi/4,  # Facing NE
    fov=fov_rat,
    cue_positions=cue_positions,
)

# Filter visible cues
visible_cues_result = viewshed.filter_cues(visible_only=True)
print(f"Visible cues: {viewshed.n_visible_cues} / {len(cue_positions)}")

# Get distances and bearings to visible cues
distances = viewshed.cue_distances  # NaN if not visible
bearings = viewshed.cue_bearings    # Egocentric angles
```

### Visibility Along Trajectory

```python
# Compute viewshed at each timepoint
viewsheds = compute_viewshed_trajectory(
    env,
    positions=trajectory,    # (n_time, 2)
    headings=headings,       # (n_time,)
    fov=fov_rat,
    n_rays=180,
)

# Accumulate time each bin was visible
view_occupancy = visibility_occupancy(
    env, viewsheds, times=timestamps  # Weighted by time spent
)
```

### Gaze Computation

```python
from neurospatial import compute_viewed_location

# Compute where animal is looking at each timepoint
viewed_locations = compute_viewed_location(
    positions=trajectory,
    headings=headings,
    view_distance=15.0,  # Fixed viewing distance (cm)
    method="fixed_distance",
)
# Shape: (n_time, 2) - NaN where viewing outside environment

# Methods for compute_viewed_location:
# - "fixed_distance": Look at point `view_distance` ahead (default)
# - "ray_cast": Cast ray until hitting boundary (requires env)
# - "boundary": Look at nearest boundary point (requires env)

# With ray casting (finds first obstacle)
viewed_locations = compute_viewed_location(
    positions=trajectory,
    headings=headings,
    method="ray_cast",
    env=env,  # Required for ray_cast and boundary
)
```

### Spatial View Cell Analysis

```python
from neurospatial import (
    compute_spatial_view_field,
    spatial_view_cell_metrics,
    is_spatial_view_cell,
    SpatialViewCellModel,
)

# Compute view field (firing rate by *viewed location*)
result = compute_spatial_view_field(
    env, spike_times, times, positions, headings,
    view_distance=15.0,
    gaze_model="fixed_distance",  # or "ray_cast", "boundary"
    method="diffusion_kde",
)

# Key insight: view field differs from place field for true SVCs
# - Place field: binned by animal position
# - View field: binned by viewed location

# Compare view vs place field
metrics = spatial_view_cell_metrics(
    env, spike_times, times, positions, headings,
    view_distance=15.0,
    info_ratio=1.5,        # View info must be 1.5x place info
    max_correlation=0.7,   # Fields must be dissimilar
)

print(metrics.interpretation())
# Output:
# *** SPATIAL VIEW CELL ***
# View field info: 1.85 bits/spike
# Place field info: 0.62 bits/spike
# View/Place info ratio: 2.98
# View-place correlation: 0.32
```

### Simulate Spatial View Cells

```python
# Model that fires when looking at specific location
svc = SpatialViewCellModel(
    env=env,
    preferred_view_location=np.array([50.0, 50.0]),
    view_field_width=10.0,   # Gaussian tuning width
    view_distance=15.0,
    gaze_model="fixed_distance",
    max_rate=20.0,
    baseline_rate=0.5,
    require_visibility=True,  # Only fire if view is unobstructed
    fov=fov_rat,             # Apply FOV constraint
)

# Generate firing rates
rates = svc.firing_rate(positions, times=times, headings=headings)

# Ground truth parameters
print(svc.ground_truth)
```

### Classification Criteria

A neuron is classified as a spatial view cell if both criteria are met:

1. **Information ratio**: `view_info > ratio × place_info` (default ratio=1.5)
2. **Field dissimilarity**: `correlation < max_corr` (default max_corr=0.7)

This ensures the cell encodes *viewed location*, not just *position*.

---

## Large Session Workflow

Optimize napari for datasets with hundreds of thousands of frames.

### Pre-compute Colormap Range

```python
from neurospatial.animation import (
    estimate_colormap_range_from_subset,
    large_session_napari_config,
)

# Pre-compute colormap range from subset (~10K frames) instead of scanning all data
vmin, vmax = estimate_colormap_range_from_subset(fields, seed=42)

# Get recommended napari settings based on session size
napari_config = large_session_napari_config(n_frames=500_000, sample_rate_hz=250)
# Returns: {'speed': 1.0, 'chunk_size': 1000, 'max_chunks': 50}

# Combine for large session workflow
session_times = np.arange(500_000) / 250.0  # 250 Hz, ~33 minutes
env.animate_fields(
    fields,
    frame_times=session_times,
    backend="napari",
    vmin=vmin,       # Pre-computed from subset (fast)
    vmax=vmax,       # Pre-computed from subset (fast)
    **napari_config, # Optimized speed, chunk_size, max_chunks
)
```

---

## Scale Bars on Visualizations (v0.11.0+)

Add visual scale bars to plots and animations.

```python
from neurospatial import ScaleBarConfig

# Static plots with scale bar
ax = env.plot_field(field, scale_bar=True)  # Auto-sized based on extent
ax = env.plot(scale_bar=True)  # Works with plot() too

# Custom scale bar configuration
config = ScaleBarConfig(length=20, position="lower left", color="white")
ax = env.plot_field(field, scale_bar=config)

# Scale bars in animations
env.animate_fields(fields, frame_times=frame_times, scale_bar=True, backend="napari")
env.animate_fields(fields, frame_times=frame_times, scale_bar=True, save_path="video.mp4")
```

**Note**: This `scale_bar` is different from `calibrate_video(scale_bar=...)` which calibrates video coordinates. This adds visual scale bars to plots.

---

## Coordinate System Details

Environment and napari use different coordinate systems.

| System | X-axis | Y-axis | Origin |
|--------|--------|--------|--------|
| Environment | Horizontal (columns) | Vertical (rows), up is positive | Bottom-left |
| Napari | Column index | Row index, down is positive | Top-left |

**For overlays:**

- Use **environment coordinates** (same as your position data)
- Animation system automatically transforms to napari pixel space
- Transformations include: (x,y) to (row,col) swap and Y-axis inversion

**Example:**

```python
# Your data is in environment coordinates (x, y)
positions = np.array([[10.0, 20.0], [15.0, 25.0]])  # (x, y) format

# Pass directly - transformation happens internally
overlay = PositionOverlay(data=positions)
env.animate_fields(fields, frame_times=frame_times, overlays=[overlay])
# Napari displays correctly with Y-axis increasing upward
```

---

## Animation Playback Control (v0.15.0+)

The animation system uses a speed-based API that separates data sample rate from playback speed.

### Core Concepts

- **`frame_times`** (required): Timestamps for each frame in seconds
- **`speed`** (default 1.0): Playback speed multiplier relative to real-time
  - `speed=1.0`: Real-time (1s data = 1s viewing)
  - `speed=0.1`: Slow motion (good for replay analysis)
  - `speed=2.0`: Fast forward

### How Playback FPS is Computed

```python
sample_rate_hz = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
playback_fps = min(sample_rate_hz * speed, MAX_PLAYBACK_FPS)  # Capped at 60 fps
```

### Use Case Examples

| Analysis Type | Data Rate | Speed | Playback fps | Notes |
|---------------|-----------|-------|--------------|-------|
| Replay decoding | 500 Hz | 0.1 | 50 fps | See trajectory unfold |
| Theta sequences | 30 Hz | 1.0 | 30 fps | Natural dynamics |
| Place fields | 30 Hz | 2.0 | 60 fps | Quick preview |
| High-speed replay | 500 Hz | 1.0 | 60 fps (capped) | Warning emitted |

### Speed Capping Warning

When requested speed exceeds display limits, a `UserWarning` is emitted:

```python
# 500 Hz data at real-time would need 500 fps (impossible)
env.animate_fields(posterior_fields, frame_times=decode_times, speed=1.0)
# UserWarning: Requested speed=1.00x would require 500 fps.
#              Capped to 60 fps (effective speed=0.12x).
```

### Override Max FPS

For high-refresh displays (120/144 Hz):

```python
env.animate_fields(
    fields,
    frame_times=frame_times,
    speed=2.0,
    max_playback_fps=120  # For 120 Hz displays
)
```
