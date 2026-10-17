# Field Animation

Animate spatial fields over time using four different backends optimized for different use cases.

## Quick Start

<!-- docs-test: run setup=docs_animation_quick_start -->
```python
import numpy as np

from neurospatial import Environment
from neurospatial.encoding import compute_spatial_rates

# Create environment and batch-encode all cells at once.
# spikes is a list of per-cell spike-time arrays (length 30 here).
env = Environment.from_samples(positions, bin_size=2.5)
fields = compute_spatial_rates(env, spikes, times, positions).firing_rates  # (30, n_bins)
frame_times = np.arange(len(fields), dtype=float) / 30.0  # one timestamp per field

# Interactive viewer (best for exploration)
env.animate_fields(fields, frame_times=frame_times, backend="napari")

# Video export (best for presentations)
env.animate_fields(fields, frame_times=frame_times, save_path="animation.mp4")

# HTML player (best for sharing)
env.animate_fields(fields, frame_times=frame_times, save_path="animation.html")

# Jupyter widget (best for notebooks)
env.animate_fields(fields, frame_times=frame_times, backend="widget")
```

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

\* Limited only by disk space (lazy loading with LRU cache)
\** Pre-renders first 500 frames, rest on-demand

### Overlay Support by Backend

| Backend | Position | Bodypart | HeadDirection | Video | Regions |
|---------|----------|----------|---------------|-------|---------|
| **Napari** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Video** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **HTML** | ✓ | ✗ | ✗ | ✗ | ✓ |
| **Widget** | ✓ | ✓ | ✓ | ✓ | ✓ |

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
| `vmin` | float | None | Minimum value for colormap (auto if None) |
| `vmax` | float | None | Maximum value for colormap (auto if None) |
| `frame_labels` | list[str] | None | Custom labels for each frame |
| `overlays` | list | None | Position, pose, heading and other overlays |
| `show_regions` | bool/list[str] | False | Draw all regions, or the named ones |
| `dry_run` | bool | False | Report what would be rendered without rendering |

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
| `percentile` | (1.0, 99.0) | Percentile range (excludes outliers) |
| `seed` | None | Random seed for reproducibility |

**Why this matters**: Without explicit `vmin/vmax`, napari computes min/max over the **entire dataset**, causing "extremely long processing times" per napari docs. This function samples a representative subset instead.

#### large_session_napari_config

Get recommended napari settings based on session size:

```python
from neurospatial.animation import large_session_napari_config

# Returns recommended playback/cache settings for napari
config = large_session_napari_config(n_frames=500_000, sample_rate_hz=250)
# {'fps': 30, 'chunk_size': 1000, 'max_chunks': 50}

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
config = large_session_napari_config(n_frames, sample_rate_hz=250)
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
uv add "napari[all]>=0.4.18,<0.6"
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
