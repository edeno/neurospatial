# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Field Animation Examples
#
# This notebook demonstrates the four animation backends for visualizing spatial fields over time:
#
# 1. **Napari** - GPU-accelerated interactive viewer (large-scale exploration)
# 2. **Video** - Parallel MP4 export (publications, presentations)
# 3. **HTML** - Standalone interactive files (sharing, remote viewing)
# 4. **Jupyter Widget** - Notebook integration (quick exploration)
#
# **Estimated time**: 15-20 minutes
#
# ## Learning Objectives
#
# By the end of this notebook, you will be able to:
#
# - Animate spatial fields over time using the `animate_fields()` method
# - Choose the appropriate backend for different use cases
# - Export videos for publications with parallel rendering
# - Create shareable HTML players with instant scrubbing
# - Handle large-scale datasets (900K+ frames) with memory-mapped arrays
# - Subsample high-frequency neural data for video export
#
# ## Prerequisites
#
# **Optional dependencies** (install as needed):
#
# ```bash
# # For Napari backend
# pip install 'napari[all]>=0.4.18'
#
# # For Jupyter widget backend
# pip install 'ipywidgets>=8.0'
#
# # For video backend (system dependency)
# # macOS: brew install ffmpeg
# # Ubuntu: sudo apt install ffmpeg
# # Windows: https://ffmpeg.org/download.html
# ```
#
# Note: HTML backend requires no additional dependencies.

# %%
import tempfile
from pathlib import Path

import numpy as np
from shapely.geometry import Point

from neurospatial import Environment
from neurospatial.animation import subsample_frames
from neurospatial.animation.backends.video_backend import check_ffmpeg_available

# Set random seed for reproducibility
np.random.seed(42)

# Determine output directory (works whether running as script or notebook)
output_dir = Path.cwd()
print(f"Output directory: {output_dir}")

# %% [markdown]
# ## Setup: Create Environment and Simulate Remapping
#
# We'll simulate place field remapping across 30 trials, where the field:
# - Starts with activity at location A (trials 1-15)
# - Undergoes remapping to location B (trials 16-30)
# - Demonstrates context-dependent spatial coding
#
# This models real phenomena like:
# - Environmental context changes
# - Learning new reward locations
# - Hippocampal remapping events

# %%
print("Creating circular arena environment...")

# Create a circular arena (50 cm radius, 100 cm diameter)
# This is a common neuroscience experimental setup
center = Point(50, 50)
radius = 50.0
circle = center.buffer(radius)

env = Environment.from_polygon(polygon=circle, bin_size=2.5, name="CircularArena")
env.units = "cm"
env.frame = "open_field"

print(f"Environment: Circular arena (radius={radius:.0f} cm)")
print(f"  {env.n_bins} bins, {env.n_dims}D")

# %%
# Simulate place field remapping across trials
print("\nSimulating place field remapping...")

n_trials = 30
remap_trial = 15  # Field remaps halfway through

# Location A: Upper-right quadrant (60, 65) cm
location_a = np.array([60.0, 65.0])
bin_a = env.bin_at(location_a.reshape(1, -1))[0]

# Location B: Lower-left quadrant (40, 35) cm
location_b = np.array([40.0, 35.0])
bin_b = env.bin_at(location_b.reshape(1, -1))[0]

print(
    f"Location A (trials 1-{remap_trial}): bin {bin_a} at [{location_a[0]:.1f}, {location_a[1]:.1f}] cm"
)
print(
    f"Location B (trials {remap_trial + 1}-{n_trials}): bin {bin_b} at [{location_b[0]:.1f}, {location_b[1]:.1f}] cm"
)

fields = []
for trial in range(n_trials):
    # Determine which location is active
    if trial < remap_trial:
        # Before remapping: field at location A
        active_bin = bin_a
        field_strength = 1.0  # Full strength at A
    else:
        # After remapping: field at location B
        active_bin = bin_b
        # Gradual emergence at new location
        field_strength = min(1.0, (trial - remap_trial + 1) / 5)

    # Compute distances from active location
    distances = env.distance_to([active_bin])

    # Gaussian place field with consistent width
    sigma = 8.0  # cm (typical place field size)
    field = field_strength * np.exp(-(distances**2) / (2 * sigma**2))

    # Add realistic noise
    noise = np.random.randn(env.n_bins) * 0.15
    field = field + noise
    field = np.maximum(field, 0)  # Non-negative firing rates

    fields.append(field)

print(f"Generated {len(fields)} trial fields (remapping at trial {remap_trial})")

# %% [markdown]
# ## Example 1: Interactive Napari Viewer
#
# **Best for**: Large datasets, exploration, real-time interaction
#
# **Features**:
# - GPU-accelerated rendering
# - Instant seeking through frames
# - Memory-efficient lazy loading
# - Suitable for 100K+ frames
#
# **Installation**: `pip install 'napari[all]>=0.4.18'`

# %%
try:
    import napari
    from IPython import get_ipython

    print("Launching Napari viewer...")
    print("PLAYBACK CONTROLS (bottom-left corner):")
    print("  ▶ Play button - Start/stop animation")
    print("  ━ Time slider - Scrub through frames")
    print("  ← → Arrow keys - Step through frames")
    print("")
    print("SPEED CONTROL (left sidebar):")
    print("  📊 'Playback Speed' widget - Drag slider to adjust FPS (1-120)")
    print("  Updates in real-time as you drag")
    print("")

    viewer = env.animate_fields(
        fields,
        backend="napari",
        fps=10,
        frame_labels=[f"Trial {i + 1}" for i in range(n_trials)],
        title="Place Field Remapping",
    )

    print("✓ Napari viewer opened")

    # Only call napari.run() when running as a script (not in Jupyter)
    # In Jupyter, the viewer stays open without blocking execution
    if get_ipython() is None:
        print("  (Running as script - window will block until closed)")
        napari.run()
    else:
        print("  (Running in Jupyter - window stays open, execution continues)")

except ImportError:
    print("⊗ Napari not available. Install with: pip install 'napari[all]>=0.4.18'")

# %% [markdown]
# ## Example 2: Video Export (MP4)
#
# **Best for**: Publications, presentations, high-quality renders
#
# **Features**:
# - Parallel rendering for speed
# - High-quality output
# - Multiple codec options (h264, h265, vp9, mpeg4)
# - Dry-run mode for time/size estimation
#
# **Installation**: System dependency (ffmpeg)
# - macOS: `brew install ffmpeg`
# - Ubuntu: `sudo apt install ffmpeg`
# - Windows: Download from https://ffmpeg.org/download.html

# %%
if check_ffmpeg_available():
    print("Exporting video with parallel rendering...")

    output_path = env.animate_fields(
        fields,
        backend="video",
        save_path=output_dir / "16_place_field_remapping.mp4",
        fps=5,
        cmap="hot",
        frame_labels=[f"Trial {i + 1}" for i in range(n_trials)],
        n_workers=4,  # Parallel rendering
        dpi=100,
    )
    print(f"✓ Video saved to {output_path}")

else:
    print("⊗ ffmpeg not available. Video export skipped.")
    print("  Install: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)")

# %% [markdown]
# ## Example 3: Standalone HTML Player
#
# **Best for**: Sharing, remote viewing, no dependencies
#
# **Features**:
# - Single self-contained file
# - Works offline in any browser
# - Instant scrubbing with slider
# - Play/pause controls
# - Keyboard shortcuts (space, arrows)
#
# **Installation**: No dependencies required

# %%
print("Generating HTML player...")

html_path = env.animate_fields(
    fields,
    backend="html",
    save_path=output_dir / "16_place_field_remapping.html",
    fps=10,
    cmap="viridis",
    frame_labels=[f"Trial {i + 1}" for i in range(n_trials)],
)

print(f"✓ HTML player saved to {html_path}")
print("  - Open in any web browser")
print("  - Instant scrubbing with slider")
print("  - Shareable (single file)")
print("  - Keyboard shortcuts: space = play/pause, arrows = step")

# %% [markdown]
# ## Example 4: Jupyter Widget
#
# **Best for**: Quick checks in notebooks, interactive exploration
#
# **Features**:
# - Integrated controls in notebook
# - Play/pause button
# - Slider for frame selection
# - Automatic display in output cell
#
# **Installation**: `pip install 'ipywidgets>=8.0'`

# %%
try:
    from IPython import get_ipython

    if get_ipython() is not None:
        print("Creating Jupyter widget...")

        widget = env.animate_fields(
            fields,
            backend="widget",
            fps=10,
            frame_labels=[f"Trial {i + 1}" for i in range(n_trials)],
        )

        print("✓ Widget created (displayed above)")
    else:
        print("⊗ Not in Jupyter notebook - widget skipped")
except ImportError:
    print("⊗ IPython not available - widget skipped")

# %% [markdown]
# ## Example 5: Large-Scale Session Pattern
#
# **Best for**: Hour-long recordings at high sampling rates (e.g., 250 Hz)
#
# **Key techniques**:
# - Memory-mapped arrays (don't load all data into RAM)
# - Napari for interactive exploration (lazy loading)
# - Frame subsampling for video export
# - Dry-run estimation before rendering
#
# **This example demonstrates the pattern** for handling large sessions:
# - Real sessions: 60K-900K frames (4 min - 1 hour at 250 Hz)
# - Real file sizes: 300 MB - 4.5 GB
# - Demo version: 1000 frames (~5 MB) to avoid filling your disk
#
# The techniques shown here scale to arbitrarily large datasets!

# %%
print("=" * 80)
print("Example 5: Large-Scale Session Pattern")
print("=" * 80)

print("\nDemonstrating techniques for large datasets (60K-900K frames):")
print("  - Use memory-mapped data (don't load into RAM)")
print("  - Use Napari for exploration (lazy loading)")
print("  - Subsample for video export")
print("\nNote: Using 1000 frames (~5 MB) for demo; scales to hours of data")

# %% [markdown]
# ### Step 1: Create Memory-Mapped Data File
#
# In practice, this would be your neural recording data. We'll simulate it here for demonstration.

# %%
# Create memory-mapped data file (simulating neural recording)
print("\nCreating memory-mapped data file...")
# For demo purposes, use a small file (1000 frames ~5 MB)
# In practice, this would be 60K-900K frames for real sessions
n_frames_large = 1000  # Demo size (real: 60K-900K frames)

# Use temporary directory for demo (in practice, use your data directory)
tmpdir = Path(tempfile.mkdtemp(prefix="neurospatial_demo_"))
mmap_path = tmpdir / "large_session.dat"

fields_mmap = np.memmap(
    str(mmap_path),
    dtype="float32",
    mode="w+",  # Create new file
    shape=(n_frames_large, env.n_bins),
)

print("Populating with sample data (in practice, this is your recording)...")
print("  (Writing in chunks to avoid memory issues)")

# Populate with simulated data (in practice, this is your neural recording)
# For this example, we'll simulate a slowly drifting place field
initial_bin = env.n_bins // 2  # Start at center of environment

chunk_size = 10000
for i in range(0, n_frames_large, chunk_size):
    # Simulate place field that drifts slowly over time
    chunk_end = min(i + chunk_size, n_frames_large)
    chunk_len = chunk_end - i

    # Slowly drifting center (drifts 20 bins over the full session)
    drift = int((i / n_frames_large) * 20)
    center_bin = initial_bin + drift
    if center_bin >= env.n_bins:
        center_bin = env.n_bins - 1

    distances = env.distance_to([center_bin])
    for j in range(chunk_len):
        fields_mmap[i + j] = np.exp(-distances / 15) + np.random.randn(env.n_bins) * 0.1

fields_mmap.flush()

print(f"\n✓ Created memory-mapped dataset: {n_frames_large:,} frames")
print(f"  File size: {n_frames_large * env.n_bins * 4 / 1e9:.2f} GB")
print("  RAM usage: ~0 MB (memory-mapped, not loaded)")

# %% [markdown]
# ### Step 2: Interactive Exploration with Napari
#
# Napari loads frames on-demand, making it efficient for exploring large datasets.

# %%
print("\nOption 1: Interactive exploration (Napari)")
print("  Napari loads frames on-demand - would handle 900K frames efficiently")

try:
    # Import napari only if attempting to use it
    import napari
    from IPython import get_ipython

    print("PLAYBACK CONTROLS:")
    print("  Bottom-left: ▶ Play button, time slider, arrow keys")
    print("  Left sidebar: 📊 'Playback Speed' widget (drag slider for 1-120 FPS)")

    viewer = env.animate_fields(
        fields_mmap,
        backend="napari",
        fps=250,  # Match recording rate
        title="Large Session Demo (1000 frames)",
    )
    print("✓ Napari viewer opened!")
    print("  (Same technique works for 60K-900K frame sessions)")

    # Only call napari.run() when running as a script (not in Jupyter)
    if get_ipython() is None:
        print("  (Running as script - window will block until closed)")
        napari.run()
    else:
        print("  (Running in Jupyter - window stays open, execution continues)")

except ImportError:
    print("⊗ Napari not available (install: pip install 'napari[all]>=0.4.18')")

# %% [markdown]
# ### Step 3: Export Subsampled Video
#
# For video export, we need to subsample the high-frequency data to a manageable frame rate.

# %%
print("\nOption 2: Export subsampled video")
print("  For large sessions: 250 Hz → 30 fps requires subsampling")

# Subsample 250 Hz → 30 fps
# For 900K frames, this would produce 108K subsampled frames (1 hour video)
# For our 1000 frame demo, this produces ~120 frames
fields_subsampled = subsample_frames(fields_mmap, target_fps=30, source_fps=250)
print(f"  Subsampled: {len(fields_subsampled):,} frames (every {250 // 30}th frame)")
print(f"  Video duration: {len(fields_subsampled) / 30:.1f} seconds")
print("  (For 900K frames, would produce ~1 hour video)")

# %% [markdown]
# ### Step 4: Dry Run to Estimate Render Time
#
# Before committing to a long render, use dry-run mode to estimate time and file size.

# %%
if check_ffmpeg_available():
    print("\nDry run estimation:")
    env.animate_fields(
        fields_subsampled,
        backend="video",
        save_path=output_dir / "16_large_session_summary.mp4",
        fps=30,
        n_workers=8,
        dry_run=True,  # Estimate first
    )
    print("\n  To render, run with dry_run=False")
else:
    print("  ⊗ ffmpeg not available for video export")

# %% [markdown]
# ### Cleanup

# %%
# Clean up temporary files
print("\nCleaning up temporary files...")
if mmap_path.exists():
    mmap_path.unlink()
    tmpdir.rmdir()
    print("✓ Temporary files removed")

# %% [markdown]
# ## Key Takeaways
#
# ### Backend Selection Guide
#
# | Use Case | Backend | Installation | Best For |
# |----------|---------|--------------|----------|
# | **Exploration** | Napari | `pip install napari[all]` | Large datasets (100K+ frames), interactive |
# | **Publication** | Video | `brew install ffmpeg` | High-quality renders, parallel speed |
# | **Sharing** | HTML | No dependencies | Remote viewing, single file |
# | **Quick check** | Widget | `pip install ipywidgets` | Notebook integration |
#
# ### Performance Tips
#
# - **Large datasets (>10K frames)**: Use Napari for exploration, subsample for video
# - **Memory constraints**: Use memory-mapped arrays (`np.memmap`)
# - **Parallel rendering**: Increase `n_workers` for faster video export
# - **File size**: Use `image_format='jpeg'` for HTML to reduce size
#
# ### Common Patterns
#
# ```python
# # Auto backend selection
# env.animate_fields(fields, backend='auto')
#
# # Quick Napari check
# env.animate_fields(fields, backend='napari')
#
# # Publication video
# env.animate_fields(fields, save_path='video.mp4', fps=5, n_workers=8)
#
# # Shareable HTML
# env.animate_fields(fields, save_path='animation.html')
#
# # Subsample high-frequency data
# from neurospatial.animation import subsample_frames
# fields_30fps = subsample_frames(fields_250hz, target_fps=30, source_fps=250)
# ```

# %% [markdown]
# ## Next Steps
#
# - Try animating your own neural data
# - Experiment with different colormaps (`cmap` parameter)
# - Add trajectory overlays (`overlay_trajectory` parameter)
# - Compare place field evolution across sessions
# - Visualize replay events or value function learning
#
# For more details, see the neurospatial documentation on animation backends.
