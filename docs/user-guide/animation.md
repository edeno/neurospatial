# Field Animation

Animate spatial fields over time using four different backends optimized for different use cases.

## Quick Start

```python
from neurospatial import Environment

# Create environment and compute fields over time
env = Environment.from_samples(positions, bin_size=2.5)
fields = [compute_place_field(env, spikes[i], times, positions) for i in range(30)]

# Interactive viewer (best for exploration)
env.animate_fields(fields, backend="napari")

# Video export (best for presentations)
env.animate_fields(fields, save_path="animation.mp4", fps=30)

# HTML player (best for sharing)
env.animate_fields(fields, save_path="animation.html")

# Jupyter widget (best for notebooks)
env.animate_fields(fields, backend="widget")
```

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

### Backend Auto-Selection

When `backend="auto"` (default), the system automatically selects based on context:

1. **File extension detection**: `.mp4` → video, `.html` → HTML
2. **Large datasets**: >10,000 frames → Napari (lazy loading)
3. **Jupyter environment**: Interactive Python → widget
4. **Fallback**: Napari if available, else HTML

```python
# Auto-selects video backend
env.animate_fields(fields, save_path="output.mp4")

# Auto-selects HTML backend
env.animate_fields(fields, save_path="output.html")

# Auto-selects based on environment and dataset size
env.animate_fields(fields)  # Napari in terminal, widget in Jupyter
```

## Napari Backend (Interactive Viewer)

GPU-accelerated viewer with lazy loading for large-scale sessions.

### Basic Usage

```python
env.animate_fields(
    fields,
    backend="napari",
    fps=30,
    cmap="viridis",
    frame_labels=["Trial 1", "Trial 2", ...],  # Optional
)
```

### Trajectory Overlays

Overlay animal trajectory on spatial fields:

```python
# 2D trajectory (displays as napari "tracks" layer)
trajectory_2d = np.array([[x1, y1], [x2, y2], ...])  # shape: (n_timepoints, 2)

env.animate_fields(
    fields,
    backend="napari",
    overlay_trajectory=trajectory_2d
)

# Higher-dimensional positions (displays as napari "points" layer)
trajectory_nd = np.array([[x, y, z], ...])  # shape: (n_timepoints, n_dims)
env.animate_fields(fields, backend="napari", overlay_trajectory=trajectory_nd)
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
    backend="video",
    save_path="animation.mp4",
    fps=30,
    codec="h264",  # or "h265", "vp9", "mpeg4"
    bitrate=5000,  # kbps
    dpi=100,
)
```

### Parallel Rendering

Speed up rendering using multiple CPU cores:

```python
# IMPORTANT: Clear caches before parallel rendering
env.clear_cache()  # Makes environment pickle-able

env.animate_fields(
    fields,
    backend="video",
    save_path="output.mp4",
    n_workers=4,  # Use 4 CPU cores
)
```

**Why clear caches?** Parallel rendering uses Python's `multiprocessing`, which requires pickle-able objects. Cached properties (KDTrees, kernels) contain non-pickle-able objects.

### Dry Run Mode

Estimate time and file size before rendering:

```python
env.animate_fields(
    fields,
    backend="video",
    save_path="output.mp4",
    dry_run=True,  # Just estimate, don't render
)
# Output:
# Dry run: Rendering 1 test frame...
# Estimated time: 2.5 minutes (30 frames, 5s per frame)
# Estimated file size: 15.2 MB
```

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
    backend="html",
    save_path="animation.html",
    fps=30,
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

## Widget Backend (Jupyter Notebooks)

Interactive widget with play/pause controls for Jupyter environments.

### Basic Usage

```python
env.animate_fields(
    fields,
    backend="widget",
    fps=30,
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

All backends accept these common parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fields` | list/ndarray | Required | Sequence of spatial fields |
| `backend` | str | `"auto"` | Backend selection |
| `save_path` | str/Path | None | Output file path |
| `fps` | int | 30 | Frames per second |
| `cmap` | str | `"viridis"` | Matplotlib colormap |
| `vmin` | float | None | Minimum value for colormap (auto if None) |
| `vmax` | float | None | Maximum value for colormap (auto if None) |
| `frame_labels` | list[str] | None | Custom labels for each frame |
| `dpi` | int | 100 | Resolution (dots per inch) |

## Working with Large-Scale Data

### Memory-Mapped Arrays

For sessions with 100K+ frames (e.g., 1-hour recording at 250 Hz), use memory-mapped arrays:

```python
import numpy as np
from neurospatial.animation import subsample_frames

# Create memory-mapped array (doesn't load into RAM)
n_frames = 900_000  # 1 hour at 250 Hz
n_bins = env.n_bins

fields = np.memmap(
    'fields.dat',
    dtype='float32',
    mode='w+',
    shape=(n_frames, n_bins)
)

# Compute fields (writes directly to disk)
for i, frame in enumerate(frames):
    fields[i] = compute_place_field(env, spikes, times, positions[i])

# Option 1: Interactive exploration (Napari lazy loads from disk)
env.animate_fields(fields, backend="napari")

# Option 2: Subsample for video (250 Hz → 30 fps)
subsampled = subsample_frames(fields, source_fps=250, target_fps=30)
env.clear_cache()
env.animate_fields(subsampled, backend="video", save_path="replay.mp4", n_workers=4)
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
env.animate_fields(large_fields, backend="napari")

# 2. Identify interesting time window
interesting_subset = large_fields[1000:2000]

# 3. Export subset as video
env.clear_cache()
env.animate_fields(interesting_subset, save_path="subset.mp4", n_workers=4)
```

## Remote Server Workflow

When working on a remote server without display:

### Option 1: HTML Export

Generate HTML on server, download and view locally:

```python
# On remote server (no display needed)
env.animate_fields(fields, backend="html", save_path="animation.html")

# Download file via scp/rsync
# $ scp user@server:animation.html .

# Open in local browser
```

### Option 2: Video Export

Render video on server, download and view locally:

```python
# On remote server (requires ffmpeg)
env.clear_cache()
env.animate_fields(
    fields,
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

### "Environment must be fitted"

**Error**: `RuntimeError: Environment must be fitted before calling this method`

**Solution**: Use factory methods to create environments:

```python
# Wrong - bare constructor doesn't fit environment
env = Environment()
env.animate_fields(fields)  # Error!

# Correct - factory methods automatically fit
env = Environment.from_samples(positions, bin_size=2.5)
env.animate_fields(fields)  # Works!
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

env.animate_fields(fields)
```

### "Pickle error during parallel rendering"

**Error**: `RuntimeError: Environment must be pickle-able for parallel rendering`

**Solution**: Clear caches before parallel video export:

```python
env.clear_cache()  # Remove non-pickle-able cached properties
env.animate_fields(fields, backend="video", n_workers=4, save_path="output.mp4")
```

**Why?** Parallel rendering uses `multiprocessing`, which requires pickle-able objects. Cached KDTrees and kernels are not pickle-able.

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
    backend="html",
    save_path="large.html",
    max_html_frames=1000,  # Override 500-frame limit
)
```

### Napari viewer shows blank/incorrect colors

**Cause**: Previously, `contrast_limits` parameter caused issues (fixed in v0.3.0)

**Solution**: Update to neurospatial v0.3.0+. If issues persist:

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

See [`examples/16_field_animation.ipynb`](../../examples/16_field_animation.ipynb) for complete working examples:

- **Example 1**: Interactive Napari viewer with circular arena
- **Example 2**: Video export with place field remapping
- **Example 3**: HTML standalone player
- **Example 4**: Jupyter widget integration
- **Example 5**: Large-scale session (memory-mapped arrays, subsampling)

## API Reference

For complete parameter documentation, see:

- [`Environment.animate_fields()`](../api/environment.md#animate_fields)
- [`subsample_frames()`](../api/animation.md#subsample_frames)
- [Animation backends](../api/animation.md)
