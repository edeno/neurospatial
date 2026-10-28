# Field Animation

A rate map or a decoded posterior summarizes a session in one picture.
Some questions need the time course instead: does the decoded position follow
the animal while it runs, and jump ahead of it during a sharp-wave ripple?
Does the posterior agree with the likelihood from each spike? Does a place
field change between the first and last laps? This guide shows how to play
spatial fields on the recording clock with the animal's position, pose,
spikes, LFP and video drawn over them, how to compare fields side by side, and
how to share a clip.

There are two entry points:

- `env.animate_fields(fields, frame_times=...)` plays one recording in one
  call. It opens a viewer and returns an `AnimationView`, or writes a file and
  returns its `Path`.
- `Animation(env, fields, ...)` builds a reusable scene. Use it to review
  identified events, open several views of the same data, or export selected
  time windows. `animate_fields` builds one of these for you.

Install what you need:

```bash
pip install "neurospatial[animation]"    # napari viewer and notebook widget
pip install "neurospatial[video]"        # .mp4, .webm, .mov, .avi and .gif files
pip install "neurospatial[fastplotlib]"  # GPU viewer (Python 3.11 or newer)
```

HTML players need no extra packages. Video files are encoded with the ffmpeg
binary that `imageio-ffmpeg` (part of the `video` extra) installs, so no
system ffmpeg is needed.

## Watch a decoded posterior with the animal's position

Each row of `fields` holds one value per environment bin, and `frame_times`
gives the time of each row in seconds. A decoded posterior has this shape: one
row per time bin. A rate map from `compute_spatial_rates` does not: it has one
row per unit, so animating it would play units, not time.

The example builds a stand-in for a decoded posterior in 50 ms bins, centered
on the animal. `env`, `positions` and `times` are a 30 Hz tracking recording
in centimeters; with your own data, use `result.posterior` and `result.times`
from `decode_session` instead. The animal's position is a
`PositionOverlay` on its own 30 Hz clock, drawn in magenta with a short trail.

<!-- docs-test: run -->
```python
import numpy as np

from neurospatial.animation import PositionOverlay

# Stand-in for decode_session(...).posterior and .times: 50 ms bins.
decode_times = np.arange(10.0, 12.0, 0.05)
decoded_xy = np.column_stack(
    [np.interp(decode_times, times, positions[:, dim]) for dim in (0, 1)]
)
squared_distance = np.sum(
    (env.bin_centers[None, :, :] - decoded_xy[:, None, :]) ** 2, axis=-1
)
posterior = np.exp(-squared_distance / (2 * 8.0**2))
posterior /= posterior.sum(axis=1, keepdims=True)

animal = PositionOverlay(positions, times=times, color="magenta", trail_length=15)
```

Write a two-second HTML player:

<!-- docs-test: run setup=docs_animation_quick_start -->
```python
path = env.animate_fields(
    posterior,
    frame_times=decode_times,
    overlays=[animal],
    colorbar_label="Probability",
    save_path="decoding.html",
    dpi=60,
)
print(path)  # decoding.html
```

Open the file in a browser. Each frame shows the posterior over the arena,
with its colorbar, the animal's position at that moment and its recent path,
and the recording time above the panel. During running the bright bins should
sit on or just ahead of the magenta marker; a posterior that consistently lags
or sits on the wrong arm points to a problem with the encoding models, the
time alignment or the coordinate frame. Gray cells are missing, not zero:
they are not bins of the environment, or the field has no value there at that
time.

The same call with `backend="widget"` plays the scene in a notebook, and
`backend="napari"` opens a desktop viewer. Close a view when you are done with
it; that also releases the video files it opened.

<!-- docs-test: run setup=docs_animation_quick_start -->
```python
view = env.animate_fields(
    posterior, frame_times=decode_times, overlays=[animal], backend="widget"
)
view.seek(11.0)  # show the field at 11.0 s of the recording
print(view.time)
view.close()
```

## Choose where to watch it

| `backend=` | Use it to | Returns | Needs |
| --- | --- | --- | --- |
| `"napari"` | Explore long recordings on a desktop, 2-D and 3-D environments | `AnimationView` | `animation` extra, a display |
| `"widget"` | Step through a short window in a notebook | `AnimationView` | `animation` extra |
| `"fastplotlib"` | Play long recordings smoothly in Jupyter, also from a remote GPU server; tracks and 3-D volumes | `AnimationView` | `fastplotlib` extra, a GPU |
| `"video"` | Present or archive a clip: `.mp4`, `.webm`, `.mov`, `.avi`, `.gif` | `Path` | `video` extra |
| `"html"` | Send a self-contained player that opens in any browser | `Path` | nothing |

With `backend="auto"` (the default), a `save_path` ending in a video suffix or
`.html` writes that file. Otherwise, in Jupyter, scenes of up to 10,000 frames
open in the notebook widget, and everything else opens in napari. `"auto"`
never chooses fastplotlib; name it.

The notebook widget draws each frame with Matplotlib in the kernel, so it can
fall behind real time on large environments or many panels. Napari and
fastplotlib update their layers on the GPU and keep up with longer recordings.

An `AnimationView` keeps the backend's own viewer in `view.native`: the napari
`Viewer`, the ipywidgets container or the `FastplotlibAnimation`. In Jupyter
the view displays itself. `view.seek(time)`, `view.select_time(start, stop)`
and `view.export(path)` work the same way in every backend (see
[Views](#views-seek-select-and-export)).

## Time: what each frame shows

Fields, tracking, spikes and video each keep their own clock, in seconds. The
viewer and the exported file step through recording time, not row numbers, so
irregular or interrupted clocks are shown at their true times.

**`max_gap` hides fields across gaps.** A row of `fields` is shown from its
time until the next row, as long as the two are at most `max_gap` seconds
apart (default 1.0 s). Across a longer interval, such as a tracking dropout,
a pause in the recording or the gap between two decoded ripples, the field is
shown as missing instead of holding a stale posterior on the screen. Rows
farther apart than `max_gap` are shown only at their exact times, and a
warning says so when that is true of every row. Pass `max_gap=None` when
every row should stay up until the next one, for example one rate map per lap
or per trial. Position, pose and
heading overlays have their own `max_gap` (default 1.0 s): a missing sample
inside a shorter gap is drawn interpolated between the samples around it, as
the analyses bridge it, and nothing is drawn across a longer gap.

**`speed` sets recording seconds per playback second.** `speed=1` plays in
real time; `speed=0.1` plays a 150 ms replay event over 1.5 s. The frame rate
of a file is the sampling rate of `frame_times` times `speed`, so every row is
shown once: 2 ms replay bins at `speed=0.1` give a 50 frame/s file. When that
rate would exceed `max_playback_fps` (default 60), the file is written at
`max_playback_fps` with the same `speed`, skipping rows between output frames,
and a warning says so. Raise `max_playback_fps` or lower `speed` to keep every
row.

Output frame *k* is at `start + k * speed / fps`. An export of the whole
recording runs from the first to the last time in `frame_times`, so the last
row is shown. A selected window, `time_range=(start, stop)` or
`view.select_time(start, stop)`, is half-open, `[start, stop)`: clips of
adjacent windows never repeat a frame.

The viewers step in seconds too. The notebook widget has a "Time (s)" slider
and a play button that advances `speed / max_playback_fps` seconds per tick.
Napari has a "Recording time" box in seconds and a Play button; playback
follows the wall clock times `speed` and skips frames when reading them is
slower than real time. Fastplotlib has a slider in seconds and its own
playback controls (see [below](#fastplotlib-backend-gpu-local-or-remote)).

## Color limits

By default each field gets robust color limits from up to 100 of its rows: the
bottom is the 1st percentile of the rows' minima and the top the 99th
percentile of their maxima. A few noisy frames then cannot dim the rest, and a
sharp posterior that covers few bins still reaches the top color. A bottom
between zero and a thousandth of the top, such as a posterior's 1e-56, is set
to zero.

Pass `vmin` and `vmax` to show two animations, or two sessions, on the same
scale. Without them, the same color means different values in each.
`colorbar_label` names what the colors measure. Colors clip values outside
the limits; in every view, a triangle at the colorbar's end marks a limit that
the field's values pass. For a very long recording,
`estimate_colormap_range_from_subset(fields, seed=0)` applies the same rule to
a larger random sample of rows and returns `(vmin, vmax)` to pass in.

The colorbar, the timestamp label and the fixed limits are drawn into every
frame of a file:

<!-- docs-test: run -->
```python
x, y = np.meshgrid([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
small = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0, units="cm")
rates = np.random.default_rng(0).uniform(0.0, 10.0, (5, small.n_bins))
rate_times = 10.05 + np.arange(5) * 0.1

small.animate_fields(
    rates,
    frame_times=rate_times,
    save_path="scaled_rates.html",
    cmap="viridis",
    vmin=0.0,
    vmax=10.0,
    colorbar_label="Firing rate (Hz)",
    dpi=50,
)
```

## Compare fields side by side

Pass a mapping from name to field to show several fields side by side, such as
the posterior and the likelihood of the same decode, or the field before and
after a manipulation. A list of fields named by `layer_names` works too.
Each field keeps its own colorbar, and every panel gets the same overlays, so
the animal's position can be read against each one.

<!-- docs-test: run setup=docs_animation_quick_start -->
```python
likelihood = np.roll(posterior, 4, axis=0)  # stand-in for a second field
path = env.animate_fields(
    {"posterior": posterior, "likelihood": likelihood},
    frame_times=decode_times,
    overlays=[animal],
    save_path="compare.gif",
    dpi=40,
)
```

In napari, each field is a separate image layer. `layout="horizontal"`,
`"vertical"` or `"grid"` places each field in its own panel with its own copy
of every overlay and region. Without `layout`, the fields are stacked in the
same coordinates.

## Share a clip

The suffix of `save_path` chooses the file: `.mp4`, `.webm`, `.mov` or `.avi`
video, an animated `.gif`, or an `.html` player. Options that do not apply to
that file raise `TypeError` naming the option.

- **Video quality.** `codec` is `"h264"` (default; `"vp9"` for `.webm`),
  `"h265"`, `"vp9"`, `"mpeg4"` or another ffmpeg encoder. The default constant
  quality `crf=18` is visually lossless; lower values give larger files.
  `preset` trades encoding time for file size (`"ultrafast"` to `"veryslow"`,
  default `"medium"`). `bitrate=` (kbit/s) sets a fixed bitrate instead of
  `crf`.
- **Size.** `dpi` (default 100) and `figsize` (inches) set the frame size.
- **HTML players.** The page has play, pause and step buttons, a slider, a
  speed menu and keyboard shortcuts (Space, the arrow keys, Home and End).
  Frames are embedded in one file, up to `max_html_frames=500`, because a
  browser holds them all in memory. `image_format="jpeg"` gives a smaller file
  than the default lossless `"png"`. `embed=False` writes the frames to a
  directory beside the page instead (`player.html` writes `player/`, or set
  `frames_dir`), with no frame limit; share the page and the directory
  together.
- **Safety.** A file appears only when it is complete, and an existing file
  is replaced only with `overwrite=True`, so a failed export never leaves a
  truncated video under the name of a good one.

For a long export, `n_workers` renders frames in several processes and
`n_workers=None` uses every CPU. The frames are identical to a serial export.
Each worker rebuilds the scene: a memory-mapped field is reopened from its
file rather than copied into every process, and anything that cannot be sent
to another process, such as a lazy reader holding an open file handle, raises
an error that says to use `n_workers=1`. On macOS and Windows, a script that
uses workers needs an `if __name__ == "__main__":` guard:

<!-- docs-test: skip Starts worker processes from a script with a main guard -->
```python
if __name__ == "__main__":
    env.animate_fields(
        posterior,
        frame_times=decode_times,
        overlays=[animal],
        save_path="session.mp4",
        speed=0.5,
        n_workers=None,
    )
```

## Use your own recording

A decoded posterior for a whole session can be larger than memory: an hour at
2 ms bins is 1.8 million rows. Keep it on disk as a memory-mapped array; rows
are read only when a frame needs them. To review a stretch of the session,
build an `Animation` on the full recording and export a window with
`time_range`. The color limits and every overlay then come from the whole
session, so clips from different times share one scale.

<!-- docs-test: run setup=docs_animation_quick_start -->
```python
from neurospatial.animation import Animation, AnimationField

# Save once, e.g. np.save("posterior.npy", result.posterior), then reopen
# without reading it into memory.
np.save("posterior.npy", posterior)
on_disk = np.load("posterior.npy", mmap_mode="r")

with Animation(
    env,
    {"posterior": AnimationField(on_disk, decode_times, label="Probability")},
    overlays=[animal],
) as scene:
    clip = scene.export(
        "run_10-11s.gif", time_range=(10.0, 11.0), fps=10, speed=1.0, dpi=40
    )
```

`env.animate_fields(on_disk, frame_times=decode_times, backend="napari")`
reads the same file lazily for interactive exploration. To show a long
recording at a lower frame rate, `subsample_frames(fields, source_fps=500,
target_fps=30)` keeps every 16th or 17th row; subsample `frame_times` the same
way.

## Inspect replay events and export a selected window

Keep the posterior's environment and clock with
`Animation.from_decoding(result)`. A scene can combine named spatial streams,
tracking, spikes, LFP, epochs and calibrated video. Its notebook view and
file exports use the same renderer, panel arrangement and color limits.

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

## Views: seek, select and export

A view provides `seek(time)`, `select_time(start, stop)`, `export(path)` and
`close()`. Times are recording seconds and selected windows are half-open.
`view.native` exposes the underlying viewer. Choose
`scene.show(backend="fastplotlib")` for GPU viewing or
`scene.show(backend="napari")` for napari layers and volume viewing;
`scene.show()` opens the notebook widget. Each field keeps its own recording
clock, colormap, range and label in all three views.

`view.export()` captures the current scientific view, including camera edits,
layer styling and custom graphics. It excludes application controls and restores
the recording time and playback state when it finishes, including after an
encoding failure. Resize the native viewer before exporting to set the output
size. Use `view.export(..., render="scene", dpi=150)` for the standard scene
layout, which also accepts `n_workers`, or `scene.export(...)` without opening a
viewer. Both take the file options described in [Share a clip](#share-a-clip).

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
Closing a native window also closes and unregisters its `AnimationView`.
`env.animate_fields` builds a scene for each call: it closes the scene after
writing a file, and closes it with the view it returns.

Position and event `size`, and object-vector `object_size`, specify marker
diameters in logical screen pixels. They stay independent of environment units
and camera zoom. Export renderers convert this diameter to their native sizing
units; HiDPI screens may use several physical pixels per logical pixel.

## Napari view

`backend="napari"` (or `scene.show(backend="napari")`) opens a napari viewer
for 2-D and 3-D environments. Each field is an image layer with its own
colormap and limits, drawn in environment coordinates and units, and each
overlay is a points, shapes, vectors or image layer that can be styled or
hidden from the layer list. Two docks sit below the canvas:

- **Recording time**: a Play/Pause button and the current time in seconds.
  Type a time or use the arrows to step by `speed / fps` seconds, where `fps`
  is the frame rate described in [Time](#time-what-each-frame-shows).
- **Scientific panels**: time series, spike rasters, epochs and fields along a
  track (`track=`), each showing `window_seconds` (default 2 s) around the
  current time, with a cursor at that time.

Options: `layout=` for separate panels (see
[Compare fields side by side](#compare-fields-side-by-side)), `ndisplay=2` or
`3`, `viewer=` to add the scene to an existing viewer, `show=False` to build
it without a window, and `title=`. In a script, call `napari.run()` after
creating the view.

<!-- docs-test: skip Opens a napari window -->
```python
import napari

view = env.animate_fields(
    posterior, frame_times=decode_times, overlays=[animal], backend="napari"
)
view.native.layers  # the napari layers
napari.run()
```

Napari prepares lazy fields, video and scientific windows on a worker thread
and applies layer updates on the GUI thread, following napari's
[threading guidance](https://napari.org/stable/guides/threading.html).
Playback follows elapsed recording time and can skip frames when reads are
slow. Scrubbing retains the latest pending request instead of building a queue.
Explicit `view.seek(time)` and export remain synchronous, so subsequent work
sees the requested time. Views and scene exports serialize access to shared
readers; closing the scene waits for pending reads. Held field rows and
overlapping track windows use bounded caches.

Uniform 3-D grids become native napari volumes, including grids with affine
coordinate transforms; other 3-D layouts show values at bin centers.

## Fastplotlib backend (GPU, local or remote)

`backend="fastplotlib"` draws each frame on the GPU when it is shown and reads
fields lazily. It draws each 2-D bin's own cell (grids, hexagons, triangles,
clipped polygons), uniform 3-D grids as native volumes, and other 3-D
environments as bin-center points. Playback performance depends on field size,
overlays, source latency and the GPU; measure your workload with the benchmark
in [Measure viewer performance](#measure-viewer-performance).

<!-- docs-test: skip Requires a GPU viewer and a video file -->
```python
view = env.animate_fields(
    fields,
    frame_times=frame_times,
    backend="fastplotlib",
    overlays=[
        PositionOverlay(positions=positions, times=times, trail_length=30),
        VideoOverlay(source="session.mp4", times=video_times, calibration=calibration),
        TimeSeriesOverlay(data=speed, times=times, label="speed (cm/s)"),
    ],
)
view  # in Jupyter: the plot, a time slider and playback controls
```

`view.native` is a `FastplotlibAnimation`, built on fastplotlib's `NDWidget`:
a slider in seconds, buttons to play, step and loop, and a frame rate box.
Each frame is shown from its time until the next frame's. A line above the
panels gives the time and the epochs it is in. Every time shown (the slider,
the time axes, that line and the tooltips) counts seconds from the first
frame, so session timestamps such as 1.7e9 s keep their precision; the line
gives the first frame's own time once ("t = 4.50 s from 1700000000.00 s").
Hovering over a bin shows its position and value. `frame_labels=` gives each
frame your own label instead of its time.

Time series are drawn beside the field on the same clock, with every sample
in the window drawn, so a ripple-band LFP keeps its shape.
Event times from an `EventOverlay` are marked on every series panel in the
events' colors. Drag the cursor in a series to move every panel. Every panel
drawn against time (a track's fields, spike rasters, time series) shows the
same `window_seconds` (default 2 s) around the current time, named in its
title, so a spike, a posterior column and an LFP peak at one x are at one time.
`[` and `]` halve or double it for all of them.

On a track (`Environment.linear_track`, `from_graph`, or any 1-D environment)
the fields are also drawn against time: position along the track on the y
axis, time on the x axis, over `window_seconds` around the current time. A
1-D environment has no other panel for them. This is the usual way to read a
decoded posterior on a linear track:

<!-- docs-test: skip Requires a GPU viewer and a decoding session -->
```python
result = decode_session(env, spike_times, times, positions, dt=0.025)
# A nearly flat posterior has no decoded position worth drawing: its peak
# lands anywhere. Keep the bins whose entropy is below 90% of a flat one's.
confident = result.posterior_entropy < 0.9 * np.log2(env.n_bins)
view = env.animate_fields(
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
bins); drawn with the others, those bins look like confident jumps. The 90%
threshold is a display choice; tighten or loosen it for your data. Each overlay
keeps its own samples, so nothing is drawn between the confident ones.

Each position overlay is drawn as a line along the track. A column starts at
its frame's time and lasts until the next frame, as the field panel holds it,
so the cursor always sits in the column the field panel shows; across a
recording gap the column ends and the gap stays empty. On a track built from
several segments, the spacing between them (`edge_spacing`) stays empty too,
lines mark where segments meet, and each segment is named by its nodes
("center–left") at the right of the panel, so a W-track or T-maze posterior
can be read by arm. Hovering over a cell shows its time, position and value;
drag the cursor to move every panel.

A 2-D decode can be read the same way: pass a linearized track of the same
maze as `track=`, and the 2-D posterior is also drawn against position along
it and time, with the positions as lines, as non_local_detector's "projected
1D" view does. Each 2-D bin's probability is added to the track bin nearest its
center (as `map_probabilities` does), so pass probabilities, and build the
track with bins no finer than the 2-D ones (a warning says so otherwise); a
track bin no 2-D bin maps to is missing. The panel has its own color scale,
since probability summed along the track is larger than per 2-D bin. `track=`
works in every backend.

<!-- docs-test: skip Requires a GPU viewer, a maze graph and a video -->
```python
maze = Environment.from_graph(graph, edge_order=edge_order, bin_size=4.0)
view = env_2d.animate_fields(
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
sequence reads as a diagonal under the posterior. In fastplotlib, with
`rates=`, clicking a unit's row shows its rate map under the field. See
[Spike Raster Overlay](../animation_overlays.md#spike-raster-overlay). An
`EpochOverlay` marks intervals such as ripples on every time panel and names
the current one above the panels; see
[Epoch Overlay](../animation_overlays.md#epoch-overlay).

Opacities look as they do in the other backends, though the GPU blends
colors in linear light. Walls, the cursor over a track's fields and the lines
where maze segments meet have a contrasting edge, labels are outlined, and
regions are sky blue, so they stay visible on any colormap color and apart in
color-blind vision.

Keyboard shortcuts, named in a line above the panels; click the figure first
so it receives the keys (in Jupyter, Tab cannot reach it yet, and the
fastplotlib controls themselves need a mouse):

| Key | Action |
| --- | --- |
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
its own `times` between slow field frames, as finely as the speed allows. For
frames with long gaps or very uneven spacing, such as decoded replay bins from
separate ripples, a warning says so: playback holds a frame through a gap. Step
frame by frame with the arrow keys, jump between ripples with N (given an
`EpochOverlay`), or set `view.native.frame` from code.

The video is placed in the environment through its calibration, which maps
camera pixels to environment coordinates (see [Calibration
Methods](../animation_overlays.md#calibration-methods)), so a camera that is
rotated, mirrored or in pixels still lines up with the bins. With
`z_order="above"` (the default) the video is blended over the field at
`alpha`. With `"below"` the field is drawn over the video and is see-through
where its values are low: each bin's opacity rises from clear at the bottom of
the color scale to opaque at a typical frame's peak (the median of the frames'
maxima), so a 2-D decoded posterior sits over the camera's view of the animal.
Missing bins keep their gray. For a posterior, a `vmax` near that typical peak
(non_local_detector's 2-D movies use 0.07) spreads most frames over the whole
colormap; sharper frames then show the colorbar's top triangle. Walls,
landmarks, regions and the moving overlays are always drawn on top, so the
arena you defined can be checked against the recording frame by frame.

For 3-D environments, `field_rendering="auto"` (the default) draws a
maximum-intensity projection (`"mip"`) for uniform 3-D grids and cells or
points for other environments. Choose `"points"` to keep bin-center points in
3-D, `"iso"` for an isosurface, or `"slice"` for a plane. Explicit volume modes
reject unsupported geometry before allocating a viewer.

<!-- docs-test: skip Requires a live GPU viewer and a 3-D scene -->
```python
view = scene.show(
    backend="fastplotlib", field_rendering="iso", volume_threshold=0.4
)
# Alternatively, the WORLD-coordinate plane z = 20:
slice_view = scene.show(
    backend="fastplotlib", field_rendering="slice", slice_plane=(0, 0, 1, -20)
)
```

`volume_threshold` uses the original field's units and must lie within its
fixed `vmin`/`vmax`; it defaults to their midpoint. Values outside the color
range saturate in color while retaining their scalar order for maximum
selection and comparison with an explicit threshold. `slice_plane=(a, b, c, d)`
means `a*x + b*y + c*z + d = 0` in environment coordinates and defaults to a
horizontal plane through the volume center. Missing and inactive voxels remain
transparent, distinct from finite zero or negative values. Nearest voxel
sampling prevents interpolation across missing cells. These options also work
with `env.animate_fields(..., backend="fastplotlib")`; static
`env.plot_field(..., backend="fastplotlib")` retains its existing bin geometry.

In Jupyter the frames are rendered on the machine running the kernel and
streamed to the browser as images, so the GPU does not have to be on your
laptop; see [Work on a remote server](#work-on-a-remote-server). Outside
Jupyter the view opens a window (in a script, follow it with
`fastplotlib.loop.run()`). `view.native.frame = 120` moves the slider to a
frame from code; `view.native.widget` is the `NDWidget` and
`view.native.figure` its figure, for adding your own graphics to
`view.native.figure["field"]`.

`env.plot(backend="fastplotlib")` and `env.plot_field(field,
backend="fastplotlib")` draw a single picture the same way, including 3-D
environments, which Matplotlib cannot draw; call `.show()` on the returned
figure.

## What each environment can show

Every backend draws every overlay type: position, pose, head direction,
video, events, time series, spike rasters, epochs and object vectors, as well
as regions and scale bars. A request a backend or environment cannot draw is
refused before any viewer opens or any frame is written, with a message that
names the alternative, instead of producing an incomplete animation.

- **1-D environments (tracks).** Fields are drawn against position (and, in
  fastplotlib, against time). Positions, pose and object vectors sit on the
  track axis at zero field amplitude. No view draws regions, head direction
  or video on a track, so `Animation` refuses them when the scene is built;
  show a heading as a `TimeSeriesOverlay` instead. Napari needs 2-D or 3-D
  coordinates, and fastplotlib also refuses pose and object vectors on a
  track.
- **3-D environments.** Headings must be 3-D unit vectors, not planar angles.
  Only point regions are drawn. Video (a planar image) and scale bars are
  refused; the 3-D axes show physical coordinates and units. Regular 3-D cells
  keep their physical geometry in exported files.
- **Fastplotlib** does not draw scale bars.

See [Animation Overlays](../animation_overlays.md) for each overlay's options,
including video calibration.

## Clocks, missing samples and coordinates

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
Their default `max_gap` of 1.0 s and optional `break_before` match analysis
support: a gap that the analyses bridge is drawn bridged, nothing is drawn
across a longer one, and nothing is extrapolated.
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

Custom `OverlayProtocol` adapters are called once, with the scene's clock, and
return sampled data (`PositionData`, `BodypartData`, `HeadDirectionData`,
`ObjectVectorData`, `VideoData`) with its own sample `times`, which views sample
at the times they show. Adapters may supply sample and interval validity masks.
The data combines them with finite coordinate support when it is made, so
rejected samples remain missing in markers, trails and track panels. Omitted
masks are inferred from the observations; omitted bodypart mask entries use that
same rule. Bodyparts without a color take the skeleton's node color, else red,
in every view.

For `track=...`, the environment and track must use the same units and
Cartesian coordinate frame. Unit spelling aliases are accepted; convert or
transform coordinates explicitly before projection.
Both may omit metadata when the caller has established their correspondence;
a declaration on only one side is insufficient.

The scene owns the video readers its `VideoOverlay`s open. `scene.overlay_data()`
samples every overlay on the scene's clock, or on another clock within the
scene's bounds, and returns borrowed views of those readers. Keep the scene open
while using them; `scene.close()` releases the readers after closing its views.
Failed scene preparation also releases readers already acquired. A reader a
custom adapter returns stays the adapter's to close.
Custom overlay adapters are converted once. Their prepared data retains its
native clock and source-frame mapping; adapters may be frozen objects.

`VideoCalibration` stores a pixel-to-centimeter transform. Calibrated overlays
convert those coordinates into the environment's metric units, including after
crop or downsampling. `calibrate_video(..., landmarks_env=...)` takes landmarks
in environment units; `cm_per_px` and scale-bar lengths remain in centimeters.
An environment without units uses the historical centimeter convention.
Calibrated video requires metric units and rejects unknown units or pixels;
uncalibrated video can still be fitted to environment bounds.

## Work on a remote server

**Jupyter on the GPU server, with fastplotlib.** Run Jupyter on the server,
reach it through an SSH tunnel, and use `backend="fastplotlib"`. The figure
renders on the server's GPU and streams to your browser, with the slider and
playback controls working as on a local machine.

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
straight to it. To save a picture without a browser, set
`RENDERCANVAS_FORCE_OFFSCREEN=1` before importing fastplotlib and read
`figure.canvas.draw()` as an RGBA array.

**Export a file.** Video, GIF and HTML exports need no display. Write the file
on the server, using its CPUs with `n_workers=None`, and copy it to your
computer (`scp user@server:replay.mp4 .`).

**Napari over X11.** `ssh -X user@server` forwards the napari window to your
screen. This is slow over high-latency connections; a file or fastplotlib
usually works better.

## Measure viewer performance

`scripts/benchmark_animation_scenes.py` measures napari and fastplotlib scene
views on a reproducible lazy-source workload:

```bash
uv run python scripts/benchmark_animation_scenes.py \
  --backend fastplotlib --dimensions 3 --grid-size 16 \
  --frames 48 --samples 24 --warmup 4 --panels --output /tmp/animation.json
```

Use `--backend napari` for Qt/OpenGL, or `--smoke` for a small rendering check.
With a visible GUI, `--delay-ms 80 --playback-seconds 1 --burst-seeks 12`
exercises slow-source playback and rapid time changes. Fastplotlib also accepts
`--field-rendering points` to compare its 3-D representations. The JSON records
versions, workload, actual capture resolution, source reads, resident memory
and warmed median/p95/p99 latency. GUI playback and event-loop responsiveness
are reported separately from synchronous seek plus GPU readback; capture timing
is not playback FPS. Offscreen runs explicitly omit GUI playback measurements.
Use matching workloads and actual resolutions when comparing runs.

## Troubleshooting

**The field is gray for long stretches.** Rows more than `max_gap` seconds
apart are not held (see [Time](#time-what-each-frame-shows)). Check that
`frame_times` are in seconds, not samples or milliseconds. For one field per
lap or trial, pass `max_gap=None`.

**`fields` and the environment do not match.** Each row needs one value per
bin of `env` (`env.n_bins`), and `frame_times` needs one time per row.
Compute fields on the same environment you animate.

**`TypeError` about an option.** A misspelled option raises with the closest
valid name, and an option that only another backend uses is ignored with a
warning that names that backend; for example `frame_labels=` is used only by
fastplotlib. The frame rate comes from `frame_times` and `speed`, so `fps=` is
refused.

**The file already exists.** Pass `overwrite=True`, or choose another
`save_path`.

**A parallel export cannot send the scene to workers.** Workers receive the
environment, the fields and the overlay observations the scene prepared, not
the overlay objects. The error names the environment, field or prepared
overlay (for example a custom video reader) that cannot be pickled. Pass
`n_workers=1`, or replace the open reader, lambda or lock with plain data such
as arrays and file paths.

**The embedded HTML player has too many frames.** Export a shorter window
with `Animation.export(time_range=...)`, lower the frame rate, pass
`embed=False`, or write a video instead.

**Napari is not installed, or there is no display.** `backend="auto"` outside
Jupyter opens napari. Install the `animation` extra, choose
`backend="widget"` or `"fastplotlib"`, or pass a `save_path` to write a file.

**Video export cannot find imageio or ffmpeg.** Install the `video` extra,
which includes `imageio-ffmpeg` and its ffmpeg binary.

## Examples and API reference

[Inspect maps, behavior and video](../examples/08_visualization.ipynb) plays
maps with tracking, pose, head direction, events and calibrated video, and
writes an HTML player.

- `Environment.animate_fields()`: one call to view or export.
- `Animation`, `AnimationField`, `AnimationView`, `EventInspector`: reusable
  scenes, views and event review.
- `subsample_frames()` and `estimate_colormap_range_from_subset()`: long
  recordings.
- [Animation Overlays](../animation_overlays.md): every overlay type.
