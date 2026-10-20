# Field Animation

Animate spatial fields on their recording clock. Use `Animation` to inspect
identified events and export the same panels; `Environment.animate_fields`
provides the existing direct backend interface.

## Quick Start

<!-- docs-test: run setup=docs_animation_quick_start -->
```python
import numpy as np

from neurospatial import Environment
# A simulated moving spatial field, one row per recording time.
# Rate maps from compute_spatial_rates have one row per UNIT, not per time.
# To animate real decoding, use Animation.from_decoding(result) below.
env = Environment.from_samples(positions, bin_size=2.5)
frame_times = times
distance = np.linalg.norm(
    positions[:, None, :] - env.bin_centers[None, :, :], axis=-1
)
fields = np.exp(-0.5 * (distance / 2.5) ** 2)

# Interactive viewer (best for exploration)
env.animate_fields(fields, frame_times=frame_times, backend="napari")

# Video export (best for presentations)
env.animate_fields(fields, frame_times=frame_times, save_path="animation.mp4")

# HTML player (best for sharing)
env.animate_fields(fields, frame_times=frame_times, save_path="animation.html")

# Jupyter widget (best for notebooks)
env.animate_fields(fields, frame_times=frame_times, backend="widget")
```

## Inspect replay events and export a selected window

Install notebook and export dependencies with
`pip install 'neurospatial[animation,video]'`.

Keep the posterior's environment and clock with
`Animation.from_decoding(result)`. A scene can combine named spatial streams,
tracking, spikes, LFP, epochs and calibrated video. Its notebook view and MP4/GIF
export use the same renderer, panel arrangement and color limits.

The example below uses a simulated posterior to demonstrate the complete
workflow. Replace `result` with your decoding result and the event records with
your detected ripples or trials. Overlapping events retain their separate IDs.

<!-- docs-test: run -->
```python
import numpy as np
import pandas as pd

from neurospatial import Environment
from neurospatial.animation import Animation, PositionOverlay, SpikeRasterOverlay
from neurospatial.decoding import DecodingResult

times = 100.0 + np.arange(21) * 0.02
positions = np.column_stack([np.linspace(0, 10, 21), np.ones(21)])
env = Environment.from_samples(positions, bin_size=1.0)
posterior = np.exp(-0.5 * np.sum(
    (env.bin_centers[None, :, :] - positions[:, None, :]) ** 2, axis=-1
))
posterior /= posterior.sum(axis=1, keepdims=True)
result = DecodingResult(posterior=posterior, env=env, times=times)

scene = Animation.from_decoding(
    result,
    overlays=[
        PositionOverlay(positions, times=times),
        SpikeRasterOverlay(
            [np.array([100.04, 100.12]), np.array([100.08, 100.20])],
            unit_ids=["CA1-7", "CA1-19"],
        ),
    ],
)
events = pd.DataFrame({
    "event_id": ["ripple-17", "ripple-28"],
    "start_time": [100.02, 100.18],
    "stop_time": [100.16, 100.36],
    "trajectory_score": [0.9, 0.4],
})
inspector = scene.inspect_events(events, padding=0.02)
inspector.filter_events(inspector.events_table()["trajectory_score"] > 0.5)
inspector.select("ripple-17")
inspector.annotate("ripple-17", review="accepted", note="Forward replay")

# A complete frame can also be inspected without a GUI.
frame = scene.render_frame(100.08, dpi=60)
assert frame.shape[-1] == 3
```

In a notebook, keep the scene open while working:

```python
inspector.show()                       # event picker, review notes, all panels
inspector.export("ripple-17.mp4", fps=30, speed=0.1)
inspector.export_events("review.json")  # IDs, intervals, scores, annotations
# Save only the selected row:
inspector.export_events("selected.csv", selected_only=True)
inspector.close()                      # controls; the scene is still reusable
scene.close()                          # all views, controls, figures and owned readers
```

`events_table()` preserves column dtypes, categorical levels and `DataFrame.attrs`,
and returns an independent copy, including nested metadata. JSON uses pandas'
table schema, read with `pd.read_json("review.json", orient="table")`. Export
checks that IDs survive that round trip unchanged before creating or replacing
the file. Composite IDs such as tuples remain valid for inspection and CSV
export; encode them explicitly as scalar strings before JSON export. JSON does
not retain `DataFrame.attrs`.
Annotations use an `annotation_` prefix, so reviewing an event cannot overwrite
an analysis score. Export refuses existing files unless `overwrite=True`.

A scene view also provides `seek(time)`, `select_time(start, stop)`, `export(path)`
and `close()`. Times are recording seconds and selected windows are half-open.
`view.native` exposes the underlying viewer. Choose
`scene.show(backend="fastplotlib")` for GPU viewing or
`scene.show(backend="napari")` for napari layers and volume viewing. Each field
keeps its own recording clock, colormap, range and label in all three views.
Napari uses a recording-time control to synchronize layers with signal,
raster, epoch and projected-track panels.

`view.export()` captures the current scientific view, including camera edits,
layer styling and custom graphics. It excludes application controls and restores
the recording time and playback state when it finishes, including after an
encoding failure. Resize the native viewer before exporting to set the output
size. Use `view.export(..., render="scene", dpi=150)` for the standard scene
layout, or `scene.export(...)` without opening a viewer.

<!-- docs-test: skip Requires a live GPU viewer and writes a video -->
```python
view = scene.show(backend="fastplotlib")
view.seek(100.08)
view.select_time(100.04, 100.12)
# Camera and custom-graphic edits made through view.native appear in this clip.
view.export("inspection.mp4", fps=30, speed=0.1)
view.close()  # scene remains available
```

`scene.close()` also closes its views and event-inspection controls, stops
playback and waits for pending frame reads before releasing owned readers.
Use `with Animation(...) as scene:` to give these resources a shared lifetime.
Closing one view leaves the scene available for another view or export. A
caller-supplied napari viewer keeps its unrelated layers and controls.

Scenes support 1-D, 2-D and 3-D environments; napari scenes require 2-D or 3-D.
Regular 3-D cells retain their physical geometry in scene exports and become
volumes in napari; layouts without cell geometry show values at bin centers.
Three-dimensional headings require vectors, not planar angles, and point
regions keep their full coordinates. Planar video, polygon regions and scale
bars require a 2-D scene. In napari, named fields use
separate layers; `layout="horizontal"` separates fields when no overlays or
regions are present. Omit `layout` to keep them registered with the fields.
Scenes reject regions, headings and video in 1-D environments. Notebook and
scene-export overlays place 1-D positions, pose and object vectors on the
spatial axis at zero field amplitude. Fastplotlib scenes also reject scale
bars and, in 1-D, pose and object-vector overlays.

For streams on different clocks, use
`Animation(env, {"posterior": AnimationField(posterior, decode_times),
"likelihood": AnimationField(likelihood, likelihood_times)})`.
Field sources remain lazy and caller-owned. Do not modify a source while a
scene is using it; close any caller-opened file after the scene.
Field metadata is immutable. To change a field's clock, support or color
settings, create a new `AnimationField`; scenes snapshot this metadata without
copying the spatial values. Track panels use each field's own observed
intervals even when `frame_times` selects a coarser scene clock.
Both scene and native-view exports correct floating-point timing roundoff
against field and overlay clocks. An export frame landing within four float64
steps of an observation can use its exact timestamp, preserving isolated
samples beside recording gaps. Selected export windows remain half-open.
Use `AnimationField(..., epochs=recording_windows)` to declare half-open
recording windows. Exact samples outside those windows are missing, and a
held interval must lie entirely within one window. `Animation.from_decoding`
preserves the result's `spike_window`; optional `epochs` further restrict that
support. Neither a wider requested window nor `max_gap=None` fills a known
recording gap.

Position, pose, heading and object-vector overlays preserve NaN observations.
When an overlay omits `times`, its samples inherit `frame_times` and retain
the same gap and missing-sample rules as an explicit clock.
Their default `max_gap=0.5` seconds and optional `break_before` match analysis
support: interpolation stays inside observed intervals, with no extrapolation.
Set `max_gap=None` deliberately for sparsely sampled continuous observations.
Trajectory-based `EventOverlay` locations use the same rules. An event inside
a tracking gap keeps its time but has an unknown position, so no spatial marker
is drawn for it. Explicit fixed event locations are unaffected.
`PositionOverlay(units=..., frame=...)` and `BodypartOverlay(units=..., frame=...)`
validate supplied coordinate metadata against the environment and ask for an
explicit transform when it differs. NWB position and pose adapters retain
declared units and reference frames; keyword overrides cannot erase or replace
known declarations. Equivalent unit spellings are accepted. Pose adapters also
require matching bodypart clocks and spatial metadata. Changing metadata labels
does not transform data.
Zero heading vectors, including those produced halfway between opposite vector
observations, have no defined direction and draw no arrow in any backend.
`SpikeRasterOverlay(rates=rate_result)`
validates the environment and aligns maps by `unit_ids`.
Mixed unit identifiers, such as integer `1` and string `"1"`, remain distinct.
Normalizing a constant `TimeSeriesOverlay` maps finite samples to zero while
preserving NaNs. Object-vector rate opacity uses the finite peak over the
prepared recording, keeping its scale stable when a view selects a window.

Custom `OverlayProtocol` adapters may supply sample and interval validity masks.
Preparation combines them with finite coordinate support, so rejected samples
remain missing in markers, trails and track panels. Omitted masks are inferred
from the observations; omitted bodypart mask entries use that same rule.

For `track=...`, the environment and track must use the same units and
Cartesian coordinate frame. Unit spelling aliases are accepted; convert or
transform coordinates explicitly before projection.
Both may omit metadata when the caller has established their correspondence;
a declaration on only one side is insufficient.

The scene owns video readers it opens. `scene.overlay_data()` returns borrowed
views of those readers, including when requesting another clock within the
scene's bounds. Keep the scene open while using them; `scene.close()` releases
the readers after closing its views. Failed scene preparation also releases
readers already acquired. The legacy `env.animate_fields` entry point releases
owned readers after export or failure; interactive views retain them until
their window or widget is closed.
Custom overlay adapters are converted once. Their prepared data retains its
native clock and source-frame mapping; adapters may be frozen objects.

`VideoCalibration` stores a pixel-to-centimeter transform. Calibrated overlays
convert those coordinates into the environment's metric units, including after
crop or downsampling. `calibrate_video(..., landmarks_env=...)` takes landmarks
in environment units; `cm_per_px` and scale-bar lengths remain in centimeters.
An environment without units uses the historical centimeter convention.
Calibrated video requires metric units and rejects unknown units or pixels;
uncalibrated video can still be fitted to environment bounds.

## Overwriting existing files

Video and HTML exports default to `overwrite=False`. Reusing an existing
`save_path` raises `FileExistsError` before rendering, including a video dry
run. Pass `overwrite=True` to replace an output deliberately:

```python
env.animate_fields(
    fields, frame_times=frame_times, backend="html",
    save_path="animation.html", overwrite=True,
)
```

HTML's default `animation.html` is protected too. With `embed=False`, a
nonempty `frames_dir` is also protected; an empty frames directory is usable.
A refused call creates no new frames directory. The option applies to video
and HTML exports; interactive napari/widget viewing writes no files, so those
backends ignore `overwrite=` with a warning.
Environment saves use the same rule for their paired `.json` and `.npz` files.

## Overview

The `animate_fields()` method visualizes how spatial fields evolve over time. Common use cases include:

- Place field dynamics during learning or context changes
- Theta sequences and replay events
- Population activity patterns
- Remapping phenomena
- Spatial coding changes across behavioral states

**What gets animated:** Any sequence of spatial fields (firing rates, probability distributions, value functions, etc.) mapped to the environment's bin structure.

## Backend Comparison

Choose the right backend for your workflow:

| Backend | Best For | Max Frames | Dependencies | Output |
|---------|----------|-----------|--------------|--------|
| **Napari** | Large datasets (100K+ frames), interactive exploration | Unlimited* | `napari[all]` | Live viewer |
| **Video** | Presentations, publications, archival | Unlimited | `ffmpeg` (system) | .mp4, .webm, .avi |
| **HTML** | Sharing standalone files, web embedding | 500 | None | .html |
| **Widget** | Jupyter notebooks, interactive analysis | ~1000** | `ipywidgets` | Interactive widget |
| **Fastplotlib** | Fast scrubbing in Jupyter, also on a remote GPU server; 3-D environments | Unlimited | `neurospatial[fastplotlib]`, a GPU | Interactive figure |

\* Limited only by disk space (lazy loading with LRU cache)
\** Pre-renders first 500 frames, rest on-demand

### Overlay Support by Backend

| Interface | Position / pose / heading / events | Video | Time series | Spikes / epochs / object vectors | Multiple fields |
|-----------|-----------------------------------|-------|-------------|----------------------------------|-----------------|
| `Animation` notebook / export | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Animation` napari | ✓ | ✓ | ✓ | ✓ | ✓ |
| Direct napari | ✓ | ✓ | ✓ | Warns and skips | ✓ |
| Direct video / widget | ✓ | ✓ | ✓ | Warns and skips | — |
| Direct HTML | ✓ | Warns and skips | Warns and skips | Warns and skips | — |
| Fastplotlib | ✓ | ✓ | ✓ | ✓ | ✓ |

All interfaces draw planar regions. Napari scenes reject polygon regions with
holes; use the notebook view or standard scene export for those regions.
Scene preflight rejects an unsupported overlay instead of producing an
incomplete view. The direct dispatcher retains its warning-and-skip behavior.

See [Animation Overlays](../animation_overlays.md) for detailed overlay documentation including VideoOverlay calibration.

### Backend Auto-Selection

When `backend="auto"` (default), the system automatically selects based on context:

1. **File extension detection**: `.mp4` → video, `.html` → HTML
2. **Large datasets**: >10,000 frames → Napari (lazy loading)
3. **Jupyter environment**: Interactive Python → widget
4. **Fallback**: Napari if available, else HTML

```python
# Auto-selects video backend
env.animate_fields(fields, frame_times=frame_times, save_path="output.mp4")

# Auto-selects HTML backend
env.animate_fields(fields, frame_times=frame_times, save_path="output.html")

# Auto-selects based on environment and dataset size
env.animate_fields(fields, frame_times=frame_times)  # Napari in terminal, widget in Jupyter
```

## Napari Backend (Interactive Viewer)

GPU-accelerated viewer with lazy loading for large-scale sessions.

### Basic Usage

```python
env.animate_fields(
    fields,
    frame_times=frame_times,
    backend="napari",
    cmap="viridis",
    frame_labels=["Trial 1", "Trial 2", ...],  # Optional
)
```

### Trajectory Overlays

Draw the animal's path with a `PositionOverlay`; it works on every backend
(see [Animation overlays](../animation_overlays.md)):

```python
from neurospatial.animation import PositionOverlay

# positions: (n_timepoints, n_dims) in the environment's units; times in seconds
path = PositionOverlay(positions=positions, times=times, trail_length=30)

env.animate_fields(
    fields,
    frame_times=frame_times,
    backend="napari",
    overlays=[path],
)
```

### Performance

- **Lazy loading**: Only loads frames when viewed (never loads full dataset into memory)
- **LRU cache**: Keeps 1000 most recently viewed frames (~30 MB)
- **Seek time**: <100ms for 100K+ frame datasets
- **Memory usage**: ~30 MB regardless of dataset size

### Controls

- **Spacebar**: Play/pause animation
- **Slider**: Scrub through frames
- **Speed control**: Adjust playback speed (widget in lower-left)
- **Arrow keys**: Step forward/backward one frame

## Video Backend (MP4/WebM Export)

Parallel rendering with ffmpeg for high-quality video export.

### Basic Usage

```python
env.animate_fields(
    fields,
    frame_times=frame_times,
    backend="video",
    save_path="animation.mp4",
    codec="h264",  # or "h265", "vp9", "mpeg4"
    crf=18,  # constant quality: lower is better, 18 is the default
    dpi=100,
)
```

By default the video encoder uses constant quality (`crf=18`), which keeps the
image visually near-lossless and lets the bitrate follow the content. Pass
`bitrate=` (kbps) only when you need a fixed bitrate; it replaces `crf`, and
passing both warns that `crf` is ignored.

### Parallel Rendering

Speed up rendering using multiple CPU cores:

```python
env.animate_fields(
    fields,
    frame_times=frame_times,
    backend="video",
    save_path="output.mp4",
    n_workers=4,  # Use 4 CPU cores
)
```

Parallel rendering uses Python's `multiprocessing`, which pickles the
environment and overlays to send them to worker processes. The environment's
cached analyses are never pickled, so there is no need to clear them first.

### Dry Run Mode

Estimate time and file size before rendering:

```python
env.animate_fields(
    fields,
    frame_times=frame_times,
    backend="video",
    save_path="output.mp4",
    dry_run=True,  # Just estimate, don't render
)
# Printed to stderr, for example:
# Video Export Dry Run Estimate:
#   Frames:          30
#   Workers:         4
#   Frame time:      17.3 ms
#   Est. total time: 0.0 minutes
#   Est. file size:  1 MB
#   Output path:     output.mp4
```

`dry_run=True` works with every backend and returns `None`. It checks the
inputs and the output path, then reports what it would do without writing a
file: HTML reports the frame count and estimated file size, napari opens no
viewer and the widget displays nothing.

### Codec Selection

| Codec | Quality | File Size | Browser Support | Notes |
|-------|---------|-----------|-----------------|-------|
| `h264` | High | Medium | Excellent | Best compatibility (default) |
| `h265` | Very High | Small | Good | Modern browsers, smaller files |
| `vp9` | High | Small | Excellent | WebM format, open source |
| `mpeg4` | Medium | Large | Excellent | Fallback for old systems |

### ffmpeg Installation

Video backend requires ffmpeg installed on your system:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (via chocolatey)
choco install ffmpeg

# Conda
conda install -c conda-forge ffmpeg
```

Check installation:

```python
from neurospatial.animation.backends.video_backend import check_ffmpeg_available
check_ffmpeg_available()  # Returns True if available
```

## HTML Backend (Standalone Player)

Self-contained HTML file with embedded frames and JavaScript controls.

### Basic Usage

```python
env.animate_fields(
    fields,
    frame_times=frame_times,
    backend="html",
    save_path="animation.html",
    title="Place Field Remapping",
    dpi=100,
)
```

### File Size Limits

HTML backend embeds frames as base64-encoded PNG images:

- **Hard limit**: 500 frames (configurable via `max_html_frames` parameter)
- **Warning threshold**: >50 MB estimated file size
- **Formula**: File size ≈ `n_frames × 0.1 MB × (dpi/100)²`

```python
# Override default limit (use with caution)
env.animate_fields(
    fields,
    frame_times=frame_times,
    backend="html",
    save_path="large.html",
    max_html_frames=1000,  # Allow up to 1000 frames
)
```

### Controls

The HTML player includes:

- **Play/pause button**: Start/stop animation
- **Range slider**: Scrub through frames
- **Speed control**: 0.25×, 0.5×, 1×, 2×, 4× playback
- **Frame counter**: Current frame / total frames
- **Keyboard shortcuts**:
    - `Space`: Play/pause
    - `←/→`: Previous/next frame

### Browser Compatibility

Works in all modern browsers (Chrome, Firefox, Safari, Edge). No server required—just open the HTML file.

### Numeric scales and timestamp labels

HTML frames do not include a colorbar. Share a labeled static plot with the
same colormap and fixed limits to communicate the values, and supply real
timestamp frame labels.
This complete example uses synthetic rate values in Hz:

<!-- docs-test: run -->
```python
import matplotlib.pyplot as plt
import numpy as np
from neurospatial import Environment

x, y = np.meshgrid([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0, units="cm")
fields = np.random.default_rng(0).uniform(0.0, 10.0, (5, env.n_bins))
frame_times = 10.05 + np.arange(5) * 0.1

ax = env.plot_field(fields[0], cmap="viridis", vmin=0.0, vmax=10.0,
                    colorbar_label="Firing rate (Hz)")
ax.set_title(f"Rate map at {frame_times[0]:.2f} s")
plt.show()
env.animate_fields(
    fields, frame_times=frame_times, backend="html", save_path="scaled_rates.html",
    frame_labels=[f"{time:.2f} s" for time in frame_times],
    cmap="viridis", vmin=0.0, vmax=10.0, dpi=50, n_workers=1,
)
```

## Widget Backend (Jupyter Notebooks)

Interactive widget with play/pause controls for Jupyter environments.

### Basic Usage

```python
env.animate_fields(
    fields,
    frame_times=frame_times,
    backend="widget",
    cmap="viridis",
)
```

### Caching Strategy

- **Pre-renders**: First 500 frames for instant scrubbing
- **On-demand**: Frames beyond 500 rendered when accessed
- **Memory**: ~50-100 MB for pre-rendered frames

### Requirements

```bash
# Install ipywidgets
uv add ipywidgets

# Enable in JupyterLab
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

## Fastplotlib Backend (GPU, local or remote)

Draws each frame on the GPU when it is shown, so a long recording starts at
once and scrubbing stays fast: a frame of a 2,600-bin arena with position,
pose, heading and event overlays updates in about 5 ms. It draws each bin's
own cell (grids, hexagons, triangles, clipped polygons) and 3-D environments
as points whose opacity follows the value, so the high values show through the
volume. `backend="auto"` never chooses it.

```bash
pip install "neurospatial[fastplotlib]"   # Python 3.11 or newer
```

```python
animation = env.animate_fields(
    fields,
    frame_times=frame_times,
    backend="fastplotlib",
    overlays=[
        PositionOverlay(positions=positions, times=times, trail_length=30),
        VideoOverlay(source="session.mp4", times=video_times, calibration=calibration),
        TimeSeriesOverlay(data=speed, times=times, label="speed (cm/s)"),
    ],
)
animation  # in Jupyter: the plot, a time slider and playback controls
```

The animation is a fastplotlib `NDWidget`: a slider in seconds, buttons to
play, step and loop, and a frame rate box. Each frame is shown from its time
until the next frame's. A line above the panels gives the time and the epochs
it is in. Every time shown (the slider, the time axes, that line and the
tooltips) counts seconds from the first frame, so session timestamps such as
1.7e9 s keep their precision; the line gives the first frame's own time once
("t = 4.50 s from 1700000000.00 s"). Hovering over a bin
shows its position and value.

Position, pose, head direction, object vectors and video each move at their
own sampling rate when given their own `times`, so a field shown once a
second (a rate map per lap, say) does not freeze 30 Hz tracking or the camera
between frames. Each sample is shown from its time until the next, or for one
typical interval across a gap (an interval more than twice the typical one;
timestamp jitter and a dropped sample are not gaps), and nothing is drawn
before an overlay's first
sample or after its last. A trail spans the time of its last `trail_length`
animation frames. Overlays without `times` follow the field's frames, as in
the other backends.

Time series are drawn beside the field on the same clock, with every sample
in the window drawn (up to 100,000), so a ripple-band LFP keeps its shape.
Event times from an `EventOverlay` are marked on every series panel in the
events' colors. Drag the cursor in a series to move every panel.

Every panel drawn against time (a track's fields, spike rasters, time series)
shows the same `window_seconds` (default 2 s) around the current time, named
in its title, so a spike, a posterior column and an LFP peak at one x are at
one time. `[` and `]` halve or double it for all of them. A
`TimeSeriesOverlay`'s own `window_seconds` is used by the other backends.

On a track (`Environment.linear_track`, `from_graph`, or any 1-D environment)
the fields are also drawn against time (a 1-D environment has no other
panel for them: the track panel takes the field panel's place): position along the track on the y axis,
time on the x axis, over `window_seconds` (default 2 s) around the current time.
This is the usual way to read a decoded posterior on a linear track:

```python
result = decode_session(env, spike_times, times, positions, dt=0.025)
# A nearly flat posterior has no decoded position worth drawing: its peak
# lands anywhere. Keep the bins whose entropy is below 90% of a flat one's.
confident = result.posterior_entropy < 0.9 * np.log2(env.n_bins)
env.animate_fields(
    result.posterior,
    frame_times=result.times,
    backend="fastplotlib",
    window_seconds=4.0,
    colorbar_label="probability",
    overlays=[
        PositionOverlay(positions=positions, times=times, color="magenta"),
        PositionOverlay(
            positions=result.map_position[confident],
            times=result.times[confident],
            color="limegreen",
            interp="nearest",
        ),
    ],
)
```

True position is magenta and decoded position green, as in the lab's decoding
figures; magenta stays visible on every viridis color and apart from green in
color-blind vision, where white disappears on the posterior's yellow and red
and green look alike. The decoded position is drawn only where the posterior
is not nearly flat: there its peak is arbitrary. On a simulated 200 cm track
decoded in 25 ms bins, 48% of the bins were nearly flat, every one put the
peak at 180 cm, and it was a median 68 cm from the animal (1 cm in the other
bins); drawn with the others, those bins look like confident jumps. The 90% threshold is a
display choice; tighten or loosen it for your data. Each overlay keeps its
own samples, so nothing is drawn between the confident ones.

Each position overlay is drawn as a line along the track. A column starts at
its frame's time and lasts until the next frame, as the field panel holds it,
so the cursor always sits in the column the field panel shows; across a
recording gap (an interval more than twice the typical one) the column ends
after one typical frame interval and the gap stays empty. On a track built from several segments, the spacing between them
(`edge_spacing`) stays empty too, lines mark where segments meet, and each
segment is named by its nodes ("center–left") at the right of the panel, so a
W-track or T-maze posterior can be read by arm. Hovering over a cell shows its time,
position and value; drag the cursor to move every panel.

A 2-D decode can be read the same way: pass a linearized track of the same
maze as `track=`, and the 2-D posterior is also drawn against position along
it and time, with the positions as lines, as non_local_detector's "projected
1D" view does. Each 2-D bin's probability is added to the track bin nearest its
center (as `map_probabilities` does), so pass probabilities, and build the
track with bins no finer than the 2-D ones (a warning says so otherwise); a
track bin no 2-D bin maps to is missing. The panel has its own color scale,
since probability summed along the track is larger than per 2-D bin:

```python
maze = Environment.from_graph(graph, edge_order=edge_order, bin_size=4.0)
env_2d.animate_fields(
    result.posterior,  # decoded on env_2d, 4 cm bins
    frame_times=result.times,
    backend="fastplotlib",
    track=maze,
    overlays=[VideoOverlay(video, calibration=calibration, times=video_times, z_order="below")],
)
```

A `SpikeRasterOverlay` draws each unit's spikes as a row of ticks beside the
field on the same clock; pass `order=np.argsort(env.to_linear(rates.peak_locations()))`
to sort the rows by place-field peak along the track, so a replay or theta
sequence reads as a diagonal under the posterior. With `rates=`, clicking a
unit's row shows its rate map under the field. See [Spike Raster
Overlay](../animation_overlays.md#spike-raster-overlay).

Opacities look as they do in the other backends, though the GPU blends
colors in linear light. Walls, the cursor over a track's fields and the lines
where maze segments meet have a contrasting edge, labels are outlined, and
regions are sky blue, so they stay visible on any colormap color and apart in
color-blind vision.

Several fields can be shown side by side on one color scale, as with napari:
pass a list of lists of frames, such as `[list(posterior),
list(likelihood)]`, with `layer_names=["posterior", "likelihood"]` and
optionally `layout="horizontal"`, `"vertical"` or `"grid"` (side by side for
up to three by default). The overlays are drawn over each, and on a track each
gets its own position-by-time panel.

An `EpochOverlay` marks intervals such as ripples on every time panel and
names the current one above the panels; see [Epoch
Overlay](../animation_overlays.md#epoch-overlay).

Keyboard shortcuts, named in a line above the panels; click the figure first
so it receives the keys (in Jupyter, Tab cannot reach it yet, and the
fastplotlib controls themselves need a mouse):

| Key | Action |
|-----|--------|
| Space | Play or pause |
| Left / Right arrow | Previous or next frame, however unevenly spaced |
| N / Shift+N | Start of the next or previous epoch or event |
| `[` / `]` | Halve or double the window of every panel drawn against time |
| Up / Down arrow | The unit one row up or down in a spike raster with rate maps |
| Escape | No unit |

Playback can show many frames a second: a posterior that jumps between
replay bins can flash over a large area. Lower the frame rate in the controls,
or step with the arrow keys, if flashing is a concern.

Play runs at `speed` (seconds of recording a second) and visits every field
frame; when that needs more than `max_playback_fps` steps a second (fastplotlib
allows up to 100), it plays slower and warns. Recording gaps do not change the
speed. Play and the step buttons also reach every sample of an overlay with
its own `times` between slow field frames, as finely as the speed allows. They move by a fixed time, short enough to reach every
frame when frames are evenly spaced (camera jitter included). For frames with
long gaps or very uneven spacing, such as decoded replay bins from separate
ripples, a warning says so: playback holds a frame through a gap. Step frame
by frame with the arrow keys, jump between ripples with N (given an
`EpochOverlay`), or set `animation.frame` from code.

The video is placed in the environment through its calibration, which maps
camera pixels to environment coordinates (see [Calibration
Methods](../animation_overlays.md#calibration-methods)), so a camera that is rotated, mirrored or in
pixels still lines up with the bins. With `z_order="above"` (the default) the
video is blended over the field at `alpha`. With `"below"` the field is
drawn over the video and is see-through where its values are low: each bin's
opacity rises from clear at the bottom of the color scale to opaque at a
typical frame's peak (the median of the frames' maxima), so a 2-D decoded
posterior sits over the camera's view of the animal. Missing bins keep their
gray. For a posterior, a `vmax` near that typical peak (non_local_detector's
2-D movies use 0.07) spreads most frames over the whole colormap; sharper
frames then show the colorbar's top triangle. Walls, landmarks, regions and the moving
overlays are always drawn on top, so the arena you defined can be checked
against the recording frame by frame.

In Jupyter the frames are rendered on the machine running the kernel and
streamed to the browser as images, so the GPU does not have to be on your
laptop; see [Remote Server Workflow](#remote-server-workflow). Outside Jupyter,
`animation.show()` opens a window (in a script, follow it with
`fastplotlib.loop.run()`). `animation.frame = 120` moves the slider to a frame
from code; `animation.widget` is the `NDWidget` and `animation.figure` its
figure, for adding your own graphics to `animation.figure["field"]`.
`animation.show(sidecar=True)` opens it beside the notebook in JupyterLab.

`env.plot(backend="fastplotlib")` and `env.plot_field(field,
backend="fastplotlib")` draw a single picture the same way, including 3-D
environments, which Matplotlib cannot draw; call `.show()` on the returned
figure.

## Common Parameters

Every backend uses these parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fields` | list/ndarray | Required | Sequence of spatial fields |
| `frame_times` | ndarray | Required | Timestamp for each frame, in seconds |
| `backend` | str | `"auto"` | Backend selection |
| `speed` | float | 1.0 | Playback speed relative to real time |
| `max_playback_fps` | int | 60 | Upper bound for displayed playback frame rate |
| `cmap` | str | `"viridis"` | Matplotlib colormap |
| `vmin` | float | None | Minimum value for colormap; if None, the 1st percentile of the frames' minima |
| `vmax` | float | None | Maximum value for colormap; if None, the 99th percentile of the frames' maxima |
| `frame_labels` | list[str] | None | Custom labels for each frame |
| `overlays` | list | None | Position, pose, heading and other overlays |
| `show_regions` | bool/list[str] | False | Draw all regions, or the named ones |
| `dry_run` | bool | False | Report what would be rendered without rendering |

The default color limits ignore the most extreme 1% of frames at each end:
about 1% of frames reach past the top and show its color, so a single noisy
frame cannot dim the rest. They are percentiles of each frame's extremes, not
of all values: a sharp posterior covers few bins, and a percentile of all
values would put the top below its peak in every frame. Pass `vmin` and `vmax`
to compare animations on one scale. A bottom between zero and a thousandth of
the top, such as a posterior's 1e-56, is zero. In the fastplotlib backend a
triangle at an end of the colorbar shows that values pass it, a swatch below
shows the missing-value color, and `colorbar_label=` names what the colors
measure.

Some parameters apply only to some backends:

| Parameter | Default | Backends |
|-----------|---------|----------|
| `save_path`, `overwrite` | None, False | video, html |
| `dpi` | 100 | video, html, widget |
| `scale_bar` | False | napari, video, html |
| `title` | "Spatial Field Animation" | napari, html |
| `n_workers` | None | video, html (`embed=False`) |
| `codec`, `crf`, `preset`, `bitrate` | "h264", 18, "medium", None | video |
| `image_format` | "png" | html, widget |
| `max_html_frames`, `embed`, `frames_dir` | see HTML backend | html |
| `cache_size`, `chunk_size`, `max_chunks`, `layout`, `layer_names` | see napari backend | napari |
| `initial_cache_size`, `cache_limit` | see widget backend | widget |

A misspelled name that no backend accepts raises `TypeError` with the closest
option (`frame_label=` suggests `frame_labels=`), so a typo cannot silently
change nothing. An option that another backend uses is ignored with one
`UserWarning`, for example `crf=15` with `backend="html"`: "crf is used by the
video backend; the html backend ignores it", followed by the call that would
use it. With `backend="auto"` the warning also says which backend was chosen
and why, for example that there is no video or `.html` `save_path` and this is
not a Jupyter session. Options you leave out use each backend's default, so
one set of settings works when you switch `backend=`.

## Working with Large-Scale Data

### Memory-Mapped Arrays

For sessions with 100K+ frames (e.g., 1-hour recording at 250 Hz), use memory-mapped arrays:

```python
import numpy as np
from neurospatial.animation import subsample_frames

# Create memory-mapped array (doesn't load into RAM)
n_frames = 900_000  # 1 hour at 250 Hz
n_bins = env.n_bins
frame_times = np.arange(n_frames, dtype=float) / 250.0

fields = np.memmap(
    'fields.dat',
    dtype='float32',
    mode='w+',
    shape=(n_frames, n_bins)
)

# `frame_fields` is your (n_frames, n_bins) per-frame field source —
# e.g. a decoded posterior over time: decode_session(...).posterior
# Write it to disk in chunks so the full array never lives in RAM at once.
chunk = 10_000
for start in range(0, n_frames, chunk):
    fields[start:start + chunk] = frame_fields[start:start + chunk]

# Option 1: Interactive exploration (Napari lazy loads from disk)
env.animate_fields(fields, frame_times=frame_times, backend="napari")

# Option 2: Subsample for video (250 Hz → 30 fps)
subsampled = subsample_frames(fields, source_fps=250, target_fps=30)
subsampled_times = subsample_frames(frame_times, source_fps=250, target_fps=30)
env.animate_fields(
    subsampled,
    frame_times=subsampled_times,
    backend="video",
    save_path="replay.mp4",
    n_workers=4,
)
```

### Subsampling Utility

The `subsample_frames()` function efficiently downsamples frame sequences:

```python
from neurospatial.animation import subsample_frames

# Subsample 250 Hz neural data to 30 fps video
subsampled = subsample_frames(
    frames=fields,        # ndarray or list
    source_fps=250,       # Original sampling rate
    target_fps=30,        # Target frame rate
)

# Works with memory-mapped arrays (no data loading)
# Returns same type as input (ndarray → ndarray, list → list)
```

**Formula**: Keeps every `source_fps // target_fps` frames (e.g., every 8th frame for 250→30).

### Performance Tips

**For large datasets (>10K frames):**

1. **Use Napari for exploration** - Lazy loading handles any dataset size
2. **Subsample before video export** - Reduces render time and file size
3. **Use dry_run mode** - Estimate video rendering time before committing
4. **Parallel rendering** - Use `n_workers` to speed up video export
5. **Memory-mapped arrays** - Avoid loading full dataset into RAM

**Typical workflow:**

```python
# 1. Explore with Napari (no data loading)
env.animate_fields(large_fields, frame_times=large_frame_times, backend="napari")

# 2. Identify interesting time window
interesting_subset = large_fields[1000:2000]
subset_times = large_frame_times[1000:2000]

# 3. Export subset as video
env.animate_fields(
    interesting_subset,
    frame_times=subset_times,
    save_path="subset.mp4",
    n_workers=4,
)
```

### Large Session Helper Functions (v0.x.x+)

Two utility functions make large-session workflows ergonomic:

#### estimate_colormap_range_from_subset

Pre-compute colormap range without scanning all frames:

```python
from neurospatial.animation import estimate_colormap_range_from_subset

# Samples ~10K random frames (fast: <0.5s for 1M+ frames)
vmin, vmax = estimate_colormap_range_from_subset(fields, seed=42)

# Use pre-computed range (avoids napari scanning all data)
env.animate_fields(
    fields,
    frame_times=frame_times,
    vmin=vmin,
    vmax=vmax,
    backend="napari",
)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fields` | Required | Array shape (n_frames, n_bins) or list of fields |
| `n_samples` | 10,000 | Max frames to sample |
| `percentile` | (1.0, 99.0) | Percentiles of the sampled frames' minima and maxima, as `animate_fields` uses by default |
| `seed` | None | Random seed for reproducibility |

**Why this matters**: Without explicit `vmin/vmax`, napari computes min/max over the **entire dataset**, causing "extremely long processing times" per napari docs. This function samples a representative subset instead.

#### large_session_napari_config

Get recommended napari settings based on session size:

```python
from neurospatial.animation import large_session_napari_config

# Returns recommended playback/cache settings for napari
config = large_session_napari_config(n_frames=500_000)
# {'chunk_size': 1000, 'max_chunks': 50}

env.animate_fields(fields, frame_times=frame_times, backend="napari", **config)
```

**Recommended Settings by Session Size:**

| Session Duration | Frames (at 250 Hz) | chunk_size | max_chunks | Notes |
|------------------|-------------------|------------|------------|-------|
| <3 min | <50K | 100 | 10 | Default settings |
| 3-13 min | 50K-200K | 500 | 20 | Medium sessions |
| 13-66 min | 200K-1M | 1000 | 50 | Large sessions |
| >66 min | >1M | 1000 | 100 | Very large sessions |

### Complete Large-Session Example

Full workflow combining both helper functions:

```python
import numpy as np
from neurospatial import Environment
from neurospatial.animation import (
    estimate_colormap_range_from_subset,
    large_session_napari_config,
)

# Create or load environment
env = Environment.from_samples(positions, bin_size=2.5)

# Create memory-mapped fields (100K+ frames)
n_frames = 500_000  # ~33 minutes at 250 Hz
frame_times = np.arange(n_frames, dtype=float) / 250.0
fields = np.memmap(
    "large_session.dat",
    dtype="float32",
    mode="w+",
    shape=(n_frames, env.n_bins),
)

# ... populate fields (writes to disk) ...

# STEP 1: Pre-compute colormap range from subset (~10K frames)
vmin, vmax = estimate_colormap_range_from_subset(fields, seed=42)
print(f"Colormap range: [{vmin:.2f}, {vmax:.2f}]")

# STEP 2: Get recommended napari settings
config = large_session_napari_config(n_frames)
print(f"Config: {config}")

# STEP 3: Launch napari with optimized settings
env.animate_fields(
    fields,
    frame_times=frame_times,
    backend="napari",
    vmin=vmin,       # Pre-computed (avoids full scan)
    vmax=vmax,       # Pre-computed (avoids full scan)
    **config,        # Recommended playback/cache settings
)
```

**Memory usage**: ~0 MB RAM regardless of session length (memmap + lazy loading).

## Remote Server Workflow

When working on a remote server without display:

### Option 0: Jupyter on the GPU server, with fastplotlib

Run Jupyter on the server, reach it through an SSH tunnel, and use
`backend="fastplotlib"`. The figure renders on the server's GPU and streams to
your browser, with the slider and playback controls working as on a local machine.

```bash
# On the server
jupyter lab --no-browser --port 8888
# On your laptop, then open http://localhost:8888
ssh -N -L 8888:localhost:8888 user@server
```

VS Code's remote Jupyter works the same way. A Jupyter server started from
an SSH shell inherits `SSH_CLIENT`, and rendercanvas then tries its terminal
canvas first and warns "could not load some backends" before falling back to
the notebook canvas; start Jupyter with `RENDERCANVAS_BACKEND=anywidget` to go
straight to it. To save a picture without a
browser, set `RENDERCANVAS_FORCE_OFFSCREEN=1` before importing fastplotlib and
read `figure.canvas.draw()` as an RGBA array.

### Option 1: HTML Export

Generate HTML on server, download and view locally:

```python
# On remote server (no display needed)
env.animate_fields(fields, frame_times=frame_times, backend="html", save_path="animation.html")

# Download file via scp/rsync
# $ scp user@server:animation.html .

# Open in local browser
```

### Option 2: Video Export

Render video on server, download and view locally:

```python
# On remote server (requires ffmpeg)
env.animate_fields(
    fields,
    frame_times=frame_times,
    backend="video",
    save_path="animation.mp4",
    n_workers=8,  # Use server's CPU cores
)

# Download file
# $ scp user@server:animation.mp4 .
```

### Option 3: X11 Forwarding for Napari

Use X11 forwarding to display Napari viewer locally:

```bash
# Connect with X11 forwarding
ssh -X user@server

# Run Python with Napari backend
python script.py  # env.animate_fields(..., backend="napari")
```

**Note**: X11 forwarding can be slow over high-latency connections. HTML/video export is usually faster.

## Layout Support

Animation works with all layout types:

| Layout Type | Visualization | Notes |
|-------------|---------------|-------|
| **Grid** (RegularGrid) | Rectangular patches | Default, most common |
| **Hexagonal** | Hexagonal patches | Better for isotropic spaces |
| **1D** (GraphLayout) | Line plot | Track linearization |
| **Triangular** | Triangular patches | Mesh-based environments |
| **Masked Grid** | Rectangular patches (active bins only) | Arbitrary boundaries |

All backends support all layouts via the `env.plot_field()` method.

## Troubleshooting

### "Napari not available"

**Error**: `ImportError: Napari backend requires napari to be installed`

**Solution**:

```bash
uv add "napari[all]>=0.5,<0.10"
```

### "ffmpeg not found"

**Error**: `RuntimeError: Video backend requires ffmpeg to be installed`

**Solution**: Install ffmpeg on your system (see [ffmpeg Installation](#ffmpeg-installation))

**Check availability**:

```python
from neurospatial.animation.backends.video_backend import check_ffmpeg_available
check_ffmpeg_available()
```

### "Environment cannot be constructed directly"

**Error**: `ValueError: [E1006] Environment cannot be constructed directly — use a factory method.`

**Solution**: Build the environment with a factory method:

```python
# Wrong - the bare constructor has no bins to animate
env = Environment()  # Error!

# Correct - a factory builds the bins from your data
env = Environment.from_samples(positions, bin_size=2.5)
env.animate_fields(fields, frame_times=frame_times)  # Works!
```

### "Field shape mismatch"

**Error**: `ValueError: All fields must have shape (n_bins,) matching environment`

**Solution**: Ensure fields match environment structure:

```python
env = Environment.from_samples(positions, bin_size=2.5)
print(f"Environment has {env.n_bins} bins")

# Each field must be a 1D array of length n_bins
field = np.random.rand(env.n_bins)  # Correct shape
fields = [field for _ in range(30)]
frame_times = np.arange(len(fields), dtype=float) / 30.0

env.animate_fields(fields, frame_times=frame_times)
```

### "Pickle error during parallel rendering"

**Error**: a `ValueError` saying that the environment, or the overlay data, is
not pickle-able, so frames cannot be rendered in parallel.

**Why?** Parallel rendering uses `multiprocessing`, which pickles the
environment and overlays to send them to worker processes. Something you
supplied cannot be pickled: a lambda or local function, an open file or video
reader, or a lock, stored in region metadata, graph attributes, an attribute
you set on the environment, or an overlay. The environment's cached analyses
are never pickled, so clearing them does not help.

**Solution**: Find the object, then replace it with plain data (arrays,
numbers, strings), or render serially:

```python
import pickle

pickle.dumps(env)  # the traceback names the object that cannot be pickled

env.animate_fields(
    fields,
    frame_times=frame_times,
    backend="video",
    n_workers=1,  # serial rendering pickles nothing
    save_path="output.mp4",
)
```

### "HTML file too large"

**Warning**: `ResourceWarning: Large file size estimated (>50 MB)`

**Solutions**:

1. **Reduce frame count**: Use fewer frames or subsample
2. **Lower DPI**: Reduce `dpi` parameter (default 100)
3. **Use video instead**: Better compression for large animations
4. **Increase limit** (if intentional):

```python
env.animate_fields(
    fields,
    frame_times=frame_times,
    backend="html",
    save_path="large.html",
    max_html_frames=1000,  # Override 500-frame limit
)
```

### Napari viewer shows blank/incorrect colors

**Cause**: The color limits do not match the field values, or the fields are
all NaN.

**Solution**:

1. Check `vmin/vmax` parameters are reasonable
2. Verify fields contain valid data (not all NaN)
3. Try different colormap: `cmap="plasma"` or `cmap="inferno"`

### "ipywidgets not found"

**Error**: `ImportError: Widget backend requires ipywidgets`

**Solution**:

```bash
uv add ipywidgets

# If using JupyterLab, enable extension
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

### Performance issues with Napari

**Symptoms**: Slow seeking, high memory usage

**Solutions**:

1. **Close other Napari layers**: Remove unused layers from viewer
2. **Reduce cache size**: Currently fixed at 1000 frames (~30 MB)
3. **Lower DPI**: Use `dpi=75` instead of default 100
4. **Check field computation**: Slow rendering may indicate slow `env.plot_field()`

## Examples

See [`examples/08_visualization.ipynb`](../examples/08_visualization.ipynb) for complete working examples:

- **Example 1**: Interactive Napari viewer with circular arena
- **Example 2**: Video export with place field remapping
- **Example 3**: HTML standalone player
- **Example 4**: Jupyter widget integration
- **Example 5**: Large-scale session (memory-mapped arrays, subsampling)

## API Reference

For complete parameter documentation, see:

- `Environment.animate_fields()` - Main animation method
- `subsample_frames()` - Frame subsampling utility
- `estimate_colormap_range_from_subset()` - Pre-compute colormap range from subset
- `large_session_napari_config()` - Get recommended napari settings for large sessions
- Animation backend modules in `neurospatial.animation.backends`
