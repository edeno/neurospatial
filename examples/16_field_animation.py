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

from neurospatial import Environment
from neurospatial.animation import subsample_frames
from neurospatial.animation.backends.video_backend import check_ffmpeg_available

# Set random seed for reproducibility
np.random.seed(42)

# %% [markdown]
# ## Setup: Create Environment and Simulate Learning
#
# We'll simulate place field formation over 30 trials, where the field:
# - Gradually sharpens (decreasing spatial width)
# - Becomes more reliable (decreasing noise)
# - Centers on a goal location

# %%
print("Creating environment...")

# Create a 100x100 cm open field arena with full coverage
# This ensures no wasted space and realistic spatial structure
arena_size = 100.0  # cm
n_grid = 50

x = np.linspace(0, arena_size, n_grid)
y = np.linspace(0, arena_size, n_grid)
xx, yy = np.meshgrid(x, y)
arena_data = np.column_stack([xx.ravel(), yy.ravel()])

env = Environment.from_samples(
    arena_data,
    bin_size=5.0,
    bin_count_threshold=1,
)
env.units = "cm"
env.frame = "open_field"

print(f"Environment: {arena_size:.0f}x{arena_size:.0f} cm open field")
print(f"  {env.n_bins} bins, {env.n_dims}D")

# %%
# Simulate place field formation over trials
print("\nSimulating place field learning...")

n_trials = 30
# Place goal in upper-right quadrant of the arena (60, 70) cm
goal_position = np.array([60.0, 70.0])
goal_bin = env.bin_at(goal_position.reshape(1, -1))[0]
print(
    f"Goal bin: {goal_bin} at position [{goal_position[0]:.1f}, {goal_position[1]:.1f}] cm"
)

fields = []
for trial in range(n_trials):
    # Field gradually sharpens
    sigma = 30 - trial * 0.5  # Decreasing width
    distances = env.distance_to([goal_bin])

    # Add noise that decreases over trials
    noise_level = 0.3 * (1 - trial / n_trials)
    noise = np.random.randn(env.n_bins) * noise_level

    field = np.exp(-(distances**2) / (2 * sigma**2)) + noise
    field = np.maximum(field, 0)  # Non-negative

    fields.append(field)

print(f"Generated {len(fields)} trial fields")

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

    print("Launching Napari viewer...")
    print("  - Use slider to scrub through trials")
    print("  - Instant seeking through all frames")
    print("  - GPU accelerated")

    viewer = env.animate_fields(
        fields,
        backend="napari",
        fps=10,
        frame_labels=[f"Trial {i + 1}" for i in range(n_trials)],
        title="Place Field Learning",
    )

    print("✓ Napari viewer launched")
    print("  (Close the viewer window to continue)")

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
        save_path="examples/16_place_field_learning.mp4",
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
    save_path="examples/16_place_field_learning.html",
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
# ## Example 5: Large-Scale Session (900K frames)
#
# **Best for**: Hour-long recordings at high sampling rates (e.g., 250 Hz)
#
# **Key techniques**:
# - Memory-mapped arrays (don't load all data into RAM)
# - Napari for interactive exploration (lazy loading)
# - Frame subsampling for video export
# - Dry-run estimation before rendering
#
# This example demonstrates handling a realistic neuroscience session:
# - 1 hour of recording
# - 250 Hz sampling rate
# - 900,000 total frames
# - ~3.6 GB of data (float32)

# %%
print("=" * 80)
print("Example 5: Large-Scale Session (900K frames)")
print("=" * 80)

print("\nFor hour-long sessions with 900K frames:")
print("  - Use memory-mapped data (don't load into RAM)")
print("  - Use Napari for exploration (lazy loading)")
print("  - Subsample for video export")

# %% [markdown]
# ### Step 1: Create Memory-Mapped Data File
#
# In practice, this would be your neural recording data. We'll simulate it here for demonstration.

# %%
# Create memory-mapped data file (simulating neural recording)
print("\nCreating memory-mapped data file...")
n_frames_large = 900_000  # 1 hour at 250 Hz

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
chunk_size = 10000
for i in range(0, n_frames_large, chunk_size):
    # Simulate place field that drifts slowly over time
    chunk_end = min(i + chunk_size, n_frames_large)
    chunk_len = chunk_end - i

    # Slowly drifting center
    drift = (i / n_frames_large) * 20  # Drifts 20 bins over session
    center_bin = goal_bin + int(drift)
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
# Napari loads frames on-demand, making it efficient for exploring 900K+ frames.

# %%
print("\nOption 1: Interactive exploration (Napari)")
print("  Napari loads frames on-demand - handles 900K frames efficiently")

try:
    # Import napari only if attempting to use it
    import napari  # noqa: F401

    viewer = env.animate_fields(
        fields_mmap,
        backend="napari",
        fps=250,  # Match recording rate
        title="Hour-Long Session (900K frames)",
    )
    print("✓ Napari viewer launched - scrub through 900K frames instantly!")
    print("  (Close the viewer window to continue)")

except ImportError:
    print("⊗ Napari not available (install: pip install 'napari[all]>=0.4.18')")

# %% [markdown]
# ### Step 3: Export Subsampled Video
#
# For video export, we need to subsample the high-frequency data to a manageable frame rate.

# %%
print("\nOption 2: Export subsampled video")
print("  900K frames → 30 fps video requires subsampling")

# Subsample 250 Hz → 30 fps
fields_subsampled = subsample_frames(fields_mmap, target_fps=30, source_fps=250)
print(f"  Subsampled: {len(fields_subsampled):,} frames (every {250 // 30}th frame)")
print(f"  Video duration: {len(fields_subsampled) / 30:.1f} seconds")

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
        save_path="examples/16_large_session_summary.mp4",
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
