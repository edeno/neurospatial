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
#     display_name: neurospatial
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Animation Overlays
#
# This notebook demonstrates the new overlay system for visualizing animal behavior alongside spatial fields:
#
# 1. **Position Overlays** - Trajectories with decaying trails
# 2. **Bodypart Overlays** - Pose tracking with skeleton rendering
# 3. **Head Direction Overlays** - Orientation arrows
# 4. **Multi-Animal Support** - Track multiple animals simultaneously
# 5. **Regions** - Highlight spatial regions of interest
# 6. **Temporal Alignment** - Sync overlays at different sampling rates
# 7. **Backend Comparison** - Same data across all backends
#
# **Estimated time**: 20-25 minutes
#
# **Prerequisites**: [16_field_animation.ipynb](16_field_animation.ipynb)
#
# ## Learning Objectives
#
# By the end of this notebook, you will be able to:
#
# - Overlay trajectories on animated spatial fields
# - Visualize pose tracking data with skeletons
# - Display head direction as dynamic arrows
# - Track multiple animals in the same animation
# - Highlight regions of interest with transparency
# - Align overlays at different sampling rates using `frame_times`
# - Choose the right backend for overlay visualization
#
# ## Prerequisites
#
# **Optional dependencies** (install as needed):
#
# ```bash
# # For Napari backend (recommended for overlays)
# pip install 'napari[all]>=0.4.18'
#
# # For video export
# # macOS: brew install ffmpeg
# # Ubuntu: sudo apt install ffmpeg
# ```
#
# **Note**: HTML backend supports position and region overlays only (no pose or head direction).
#
# The default run creates static previews and temporary demo artifacts. Desktop
# viewers and expensive final exports require an explicit opt-in below, even
# when their packages are installed. Re-run cells after changing a flag.
#
# %%
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point

from neurospatial import Environment
from neurospatial.animation import (
    BodypartOverlay,
    HeadDirectionOverlay,
    PositionOverlay,
    Skeleton,
)
from neurospatial.animation.backends.video_backend import check_ffmpeg_available

RUN_NAPARI = False
RUN_VIDEO_EXPORT = False
RUN_WIDGET = False  # optional notebook controls, also disabled for batch execution

# Set random seed for reproducibility
np.random.seed(42)

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (_here / "_style.py").exists() and (_here / "examples" / "_style.py").exists():
    _here = _here / "examples"
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
from _style import apply_style

apply_style(figsize=(12, 8))

# Output directory
output_dir = Path(tempfile.mkdtemp(prefix="neurospatial-example-17-"))
print(f"Output directory: {output_dir}")

# %% [markdown]
# ## Setup: Create Environment and Simulate Data
#
# We'll create a circular arena and simulate:
# - A place field that tracks with the animal
# - Animal trajectory exploring the arena
# - Head direction as the animal moves
# - Pose data (nose, body center, tail base) for skeleton visualization

# %%
print("Creating circular arena environment...")

# Circular arena (50 cm radius)
center = Point(50, 50)
radius = 50.0
circle = center.buffer(radius)

env = Environment.from_polygon(polygon=circle, bin_size=2.5, name="CircularArena")
env.units = "cm"
env.frame = "open_field"

# Add region of interest (reward zone in upper-right quadrant)
reward_zone = Point(65, 65)
env.regions.add("reward", point=reward_zone)

print(f"Environment: {env.n_bins} bins, {env.n_dims}D")
print(f"Regions: {list(env.regions.keys())}")

# %%
print("\nSimulating animal trajectory...")

n_frames = 50  # 50 time points
t = np.linspace(0, 4 * np.pi, n_frames)  # 2 revolutions

# Spiral trajectory from center outward
r = np.linspace(5, 40, n_frames)  # Radius increases
theta = t + np.random.randn(n_frames) * 0.1  # Angle with noise

# Convert to Cartesian (center at 50, 50)
trajectory = np.column_stack([50 + r * np.cos(theta), 50 + r * np.sin(theta)])

# Head direction (tangent to spiral)
velocity = np.gradient(trajectory, axis=0)
head_angles = np.arctan2(velocity[:, 1], velocity[:, 0])  # direction of sampled motion

print(f"Trajectory: {n_frames} frames")
print(f"  Position range: [{trajectory.min():.1f}, {trajectory.max():.1f}] cm")

# %%
print("\nSimulating pose data (nose, body, tail)...")

# Pose: 3 keypoints with skeleton
body_length = 10.0  # cm

# Nose: ahead of body center
nose_offset = body_length * 0.5
nose_x = trajectory[:, 0] + nose_offset * np.cos(head_angles)
nose_y = trajectory[:, 1] + nose_offset * np.sin(head_angles)

# Body center: trajectory position
body_x = trajectory[:, 0]
body_y = trajectory[:, 1]

# Tail: behind body center
tail_offset = body_length * 0.5
tail_x = trajectory[:, 0] - tail_offset * np.cos(head_angles)
tail_y = trajectory[:, 1] - tail_offset * np.sin(head_angles)

# Pose dictionary
pose_data = {
    "nose": np.column_stack([nose_x, nose_y]),
    "body": np.column_stack([body_x, body_y]),
    "tail": np.column_stack([tail_x, tail_y]),
}

# Skeleton: defines bodypart nodes and edge connections
skeleton = Skeleton(
    name="mouse_simple",
    nodes=("nose", "body", "tail"),
    edges=(("tail", "body"), ("body", "nose")),
    edge_color="white",
    edge_width=2.0,
)

print("Pose: 3 keypoints (nose, body, tail)")
print(f"Skeleton: {skeleton.n_edges} edges")
print("  Node colors and edge styling defined in Skeleton object")

# %%
print("\nSimulating synthetic moving activity bump...")

# Synthetic activity centered on position (a.u.), not an estimated place-cell field
fields = []
for i in range(n_frames):
    # Find bin closest to animal position
    pos = trajectory[i : i + 1]  # Shape (1, 2)
    center_bin = env.bin_at(pos)[0]

    # Gaussian field around animal
    distances = env.distance_to([center_bin])
    sigma = 12.0  # cm
    field = np.exp(-(distances**2) / (2 * sigma**2))

    # Add noise
    field = field + np.random.randn(env.n_bins) * 0.1
    field = np.maximum(field, 0)

    fields.append(field)

fields = np.array(fields)
print(f"Fields: {fields.shape} (frames x bins)")

# Create frame_times (required for animate_fields)
# Using 10 Hz frame rate (100ms per frame)
frame_times = np.arange(n_frames) / 10.0  # seconds
print(f"Frame times: {frame_times[0]:.2f}s to {frame_times[-1]:.2f}s (10 Hz)")

# %% [markdown]
# Static snapshots make the synthetic field/trajectory/pose relationship visible
# without a GUI. Angles are radians in Cartesian coordinates (0=east, +π/2=north).
# The moving bump is a visualization fixture in arbitrary units, not an encoding model.

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
for ax, index in zip(axes, [0, n_frames // 2, n_frames - 1], strict=True):
    env.plot_field(
        fields[index],
        ax=ax,
        vmin=0,
        vmax=float(fields.max()),
        colorbar_label="Synthetic activity (a.u.)",
    )
    ax.plot(trajectory[: index + 1, 0], trajectory[: index + 1, 1], "r-", lw=1)
    pose = np.array([pose_data[name][index] for name in ["tail", "body", "nose"]])
    ax.plot(pose[:, 0], pose[:, 1], "wo-", markeredgecolor="black", lw=2)
    ax.set_title(f"t={frame_times[index]:.1f} s")
plt.show()

# %% [markdown]
# ## Example 1: Position Overlay with Trail
#
# Overlay the animal's trajectory on the animated field with a decaying trail showing recent positions.
#
# **Key features**:
# - `trail_length=10` shows last 10 frames
# - Trail fades from current (opaque) to past (transparent)
# - Current position rendered as a marker

# %%
# Create position overlay with trail
position_overlay = PositionOverlay(
    positions=trajectory,
    times=frame_times,
    color="red",
    size=12.0,
    trail_length=10,  # Show last 10 frames as trail
)

print("Example 1: Position Overlay with Trail")
print(f"  Trajectory: {trajectory.shape[0]} frames")
print("  Trail length: 10 frames (decaying alpha)")
print("  Color: red, Size: 12.0")

if RUN_NAPARI:
    try:
        import napari
        from IPython import get_ipython

        print("\nLaunching Napari viewer...")
        viewer = env.animate_fields(
            fields,
            overlays=[position_overlay],
            frame_times=frame_times,
            backend="napari",
            speed=1.0,
            title="Position Overlay with Trail",
        )

        print("✓ Napari viewer opened")
        print("  Watch the red trail follow the animal")

        if get_ipython() is None:
            napari.run()

    except ImportError:
        print("⊗ Napari not available. Install with: pip install 'napari[all]>=0.4.18'")
else:
    print("Optional viewer skipped; set RUN_NAPARI=True to enable it.")

# %% [markdown]
# ## Example 2: Pose Tracking with Skeleton
#
# Overlay full pose data (nose, body, tail) with skeleton connecting the keypoints.
#
# **Key features**:
# - `data` is a dict mapping bodypart names to trajectories
# - `skeleton` defines edges between bodyparts
# - `colors` can customize per-bodypart colors
# - Skeleton rendered with specified color and width

# %%
# Create bodypart overlay with skeleton
# Note: Skeleton styling (edge_color, edge_width) comes from the Skeleton object
bodypart_overlay = BodypartOverlay(
    data=pose_data,
    times=frame_times,
    skeleton=skeleton,  # Skeleton object with nodes, edges, and styling
    colors={"nose": "yellow", "body": "red", "tail": "blue"},
)

print("Example 2: Pose Tracking with Skeleton")
print(f"  Bodyparts: {list(pose_data.keys())}")
print(f"  Skeleton edges: {skeleton.edges}")
print(
    f"  Skeleton styling: edge_color={skeleton.edge_color}, edge_width={skeleton.edge_width}"
)
print("  Node colors: nose=yellow, body=red, tail=blue")

if RUN_NAPARI:
    try:
        import napari
        from IPython import get_ipython

        print("\nLaunching Napari viewer...")
        viewer = env.animate_fields(
            fields,
            overlays=[bodypart_overlay],
            frame_times=frame_times,
            backend="napari",
            speed=1.0,
            title="Pose Tracking with Skeleton",
        )

        print("✓ Napari viewer opened")
        print("  Watch the skeleton follow the animal pose")

        if get_ipython() is None:
            napari.run()

    except ImportError:
        print("⊗ Napari not available. Install with: pip install 'napari[all]>=0.4.18'")
else:
    print("Optional viewer skipped; set RUN_NAPARI=True to enable it.")

# %% [markdown]
# ## Example 3: Head Direction Visualization
#
# Overlay head direction as dynamic arrows pointing in the direction of travel.
#
# **Key features**:
# - `data` can be angles (radians) or unit vectors
# - Arrows rendered with specified color and length
# - Arrow origin is at the animal's position

# %%
# Create head direction overlay (angles in radians)
head_direction_overlay = HeadDirectionOverlay(
    headings=head_angles,
    times=frame_times,
    color="yellow",
    length=15.0,  # Arrow length in cm
)

print("Example 3: Head Direction Visualization")
print(f"  Head angles: {head_angles.shape[0]} frames")
print("  Arrow color: yellow, Length: 15.0 cm")

if RUN_NAPARI:
    try:
        import napari
        from IPython import get_ipython

        print("\nLaunching Napari viewer...")

        # Combine position + head direction overlays
        viewer = env.animate_fields(
            fields,
            overlays=[position_overlay, head_direction_overlay],
            frame_times=frame_times,
            backend="napari",
            speed=1.0,
            title="Position + Head Direction",
        )

        print("✓ Napari viewer opened")
        print("  Watch the yellow arrow show heading direction")

        if get_ipython() is None:
            napari.run()

    except ImportError:
        print("⊗ Napari not available. Install with: pip install 'napari[all]>=0.4.18'")
else:
    print("Optional viewer skipped; set RUN_NAPARI=True to enable it.")

# %% [markdown]
# ## Example 4: Multi-Animal Tracking
#
# Track multiple animals simultaneously by providing multiple overlay instances.
#
# **Key features**:
# - Pass a list of overlays for each animal
# - Each overlay automatically gets a suffix (e.g., "Position_1", "Position_2")
# - All animals rendered in the same animation with different colors

# %%
print("Example 4: Multi-Animal Tracking")
print("\nSimulating second animal...")

# A mirrored synthetic trajectory stays inside the same circular arena.
# A rectangular clip would not guarantee circular-arena membership.
trajectory_2 = np.column_stack([100.0 - trajectory[:, 0], trajectory[:, 1]])

# Create overlays for both animals
animal1_overlay = PositionOverlay(
    positions=trajectory, color="red", size=12.0, trail_length=8
)

animal2_overlay = PositionOverlay(
    positions=trajectory_2, color="blue", size=12.0, trail_length=8
)

print("  Animal 1: red")
print("  Animal 2: blue (mirrored trajectory)")

if RUN_NAPARI:
    try:
        import napari
        from IPython import get_ipython

        print("\nLaunching Napari viewer...")
        viewer = env.animate_fields(
            fields,
            overlays=[animal1_overlay, animal2_overlay],  # Multiple overlays
            frame_times=frame_times,
            backend="napari",
            speed=1.0,
            title="Multi-Animal Tracking",
        )

        print("✓ Napari viewer opened")
        print("  Watch both animals explore simultaneously")

        if get_ipython() is None:
            napari.run()

    except ImportError:
        print("⊗ Napari not available. Install with: pip install 'napari[all]>=0.4.18'")
else:
    print("Optional viewer skipped; set RUN_NAPARI=True to enable it.")

# %% [markdown]
# ## Example 5: Regions Overlay with Spatial Fields
#
# Highlight spatial regions of interest (e.g., reward zones) alongside overlays.
#
# **Key features**:
# - `show_regions=True` displays all defined regions
# - `show_regions=["reward"]` displays specific regions only
# - `region_alpha=0.3` controls transparency
# - Regions rendered as colored polygons/points

# %%
print("Example 5: Regions Overlay")
print(f"  Showing region: {list(env.regions.keys())}")
print("  Region alpha: 0.3 (30% transparent)")

if RUN_NAPARI:
    try:
        import napari
        from IPython import get_ipython

        print("\nLaunching Napari viewer...")
        viewer = env.animate_fields(
            fields,
            overlays=[position_overlay],
            show_regions=True,  # Show all regions
            region_alpha=0.3,  # 30% transparent
            frame_times=frame_times,
            backend="napari",
            speed=1.0,
            title="Position + Reward Region",
        )

        print("✓ Napari viewer opened")
        print("  Watch the animal approach the reward region")

        if get_ipython() is None:
            napari.run()

    except ImportError:
        print("⊗ Napari not available. Install with: pip install 'napari[all]>=0.4.18'")
else:
    print("Optional viewer skipped; set RUN_NAPARI=True to enable it.")

# %% [markdown]
# ## Example 6: Mixed-Rate Temporal Alignment
#
# Align overlays sampled at different rates using temporal timestamps.
#
# **Key features**:
# - Overlay `times` parameter specifies timestamps for each frame
# - `frame_times` parameter specifies field frame timestamps
# - Interpolation resamples known timestamps; it does not synchronize separate clocks
# - Use a shared clock; split observation gaps before interpolation
#
# **Example**: Position tracked at 120 Hz, fields computed at 10 Hz

# %%
print("Example 6: Mixed-Rate Temporal Alignment")
print("\nSimulating high-frequency position tracking...")

# High-frequency position tracking (120 Hz)
duration = 5.0  # seconds
fps_high = 120  # Hz
n_samples_high = int(duration * fps_high)  # 600 samples

# Generate high-frequency trajectory
t_high = np.arange(n_samples_high) / fps_high
theta_high = t_high * 2 * np.pi + np.random.randn(n_samples_high) * 0.05
r_high = 20 + 15 * np.sin(t_high * 3)

trajectory_high_freq = np.column_stack(
    [50 + r_high * np.cos(theta_high), 50 + r_high * np.sin(theta_high)]
)
timestamps_high = t_high

print(f"  Position tracking: {n_samples_high} samples at {fps_high} Hz")

# Low-frequency fields (10 Hz)
fps_low = 10  # Hz
n_frames_low = int(duration * fps_low)  # 50 frames
mixed_frame_times = np.arange(n_frames_low) / fps_low

print(f"  Field computation: {n_frames_low} frames at {fps_low} Hz")

# Compute fields at low frequency
fields_low_freq = []
for t in mixed_frame_times:
    # Find closest high-freq position
    idx = np.argmin(np.abs(timestamps_high - t))
    pos = trajectory_high_freq[idx : idx + 1]
    center_bin = env.bin_at(pos)[0]

    distances = env.distance_to([center_bin])
    field = np.exp(-(distances**2) / (2 * 12.0**2))
    field = field + np.random.randn(env.n_bins) * 0.1
    field = np.maximum(field, 0)
    fields_low_freq.append(field)

fields_low_freq = np.array(fields_low_freq)

# Create overlay with timestamps
position_overlay_timed = PositionOverlay(
    positions=trajectory_high_freq,
    times=timestamps_high,  # 120 Hz timestamps
    color="red",
    size=10.0,
    trail_length=15,
)

print("\n✓ Overlay will be interpolated to match field frame times")
print("  (Linear interpolation: 120 Hz → 10 Hz)")

if RUN_NAPARI:
    try:
        import napari
        from IPython import get_ipython

        print("\nLaunching Napari viewer...")
        viewer = env.animate_fields(
            fields_low_freq,
            overlays=[position_overlay_timed],
            frame_times=mixed_frame_times,  # independent from the original example clock
            backend="napari",
            speed=1.0,
            title="Mixed-Rate Alignment (120 Hz → 10 Hz)",
        )

        print("✓ Napari viewer opened")
        print("  Position automatically aligned to field frames")

        if get_ipython() is None:
            napari.run()

    except ImportError:
        print("⊗ Napari not available. Install with: pip install 'napari[all]>=0.4.18'")
else:
    print("Optional viewer skipped; set RUN_NAPARI=True to enable it.")

# %% [markdown]
# ## Example 7: Backend Comparison
#
# Compare overlay rendering across all backends with the same data.
#
# **Backend capabilities**:
#
# | Backend | Position | Bodypart | HeadDirection | Regions |
# |---------|----------|----------|---------------|--------|
# | Napari  | ✓ | ✓ | ✓ | ✓ |
# | Video   | ✓ | ✓ | ✓ | ✓ |
# | HTML    | ✓ | ✗ | ✗ | ✓ |
# | Widget  | ✓ | ✓ | ✓ | ✓ |
#
# **Note**: HTML backend warns when given unsupported overlay types.

# %%
print("Example 7a: Napari Backend (Full Support)")

if RUN_NAPARI:
    try:
        import napari
        from IPython import get_ipython

        # All overlay types supported
        viewer = env.animate_fields(
            fields,
            overlays=[position_overlay, bodypart_overlay, head_direction_overlay],
            show_regions=True,
            frame_times=frame_times,
            backend="napari",
            speed=1.0,
            title="Napari: All Overlays",
        )

        print("✓ Napari: Position + Pose + Head Direction + Regions")

        if get_ipython() is None:
            napari.run()

    except ImportError:
        print("⊗ Napari not available")
else:
    print("Optional viewer skipped; set RUN_NAPARI=True to enable it.")

# %%
print("Example 7b: Video Backend (Full Support)")

if RUN_VIDEO_EXPORT and check_ffmpeg_available():
    # All overlay types supported
    output_path = env.animate_fields(
        fields,
        overlays=[position_overlay, bodypart_overlay, head_direction_overlay],
        show_regions=True,
        frame_times=frame_times,
        backend="video",
        save_path=output_dir / "17_all_overlays.mp4",
        speed=1.0,
        n_workers=1,
    )
    print(f"✓ Video: Saved to {output_path}")
else:
    print(
        "⊗ Video export disabled by default or ffmpeg unavailable (RUN_VIDEO_EXPORT=True)"
    )

# %%
print("Example 7c: HTML Backend (Position + Regions Only)")
print("  WARNING: HTML backend does NOT support bodypart or head direction overlays")
print("  (Warnings will be emitted if provided)\n")

# HTML: Only position and regions supported
html_path = env.animate_fields(
    fields,
    overlays=[position_overlay],  # Only position overlay
    show_regions=True,
    frame_times=frame_times,
    backend="html",
    save_path=output_dir / "17_position_only.html",
    speed=1.0,
)

print(f"✓ HTML: Saved to {html_path}")
print("  (Position + Regions rendered; pose/head direction not supported)")

# %%
print("Example 7d: Widget Backend (Full Support)")

if RUN_WIDGET:
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            # All overlay types supported
            widget = env.animate_fields(
                fields,
                overlays=[position_overlay, bodypart_overlay, head_direction_overlay],
                show_regions=True,
                frame_times=frame_times,
                backend="widget",
                speed=1.0,
            )
            print("✓ Widget: Position + Pose + Head Direction + Regions")
        else:
            print("⊗ Not in Jupyter notebook")

    except ImportError:
        print("⊗ IPython/ipywidgets not available")
else:
    print("Optional viewer skipped; set RUN_WIDGET=True to enable it.")

# %% [markdown]
# ## Key Takeaways
#
# ### Overlay Types
#
# 1. **PositionOverlay**: Trajectories with decaying trails
#    - `data`: (n_frames, n_dims) array
#    - `trail_length`: Number of past frames to show
#    - `color`, `size`: Marker appearance
#
# 2. **BodypartOverlay**: Pose tracking with skeletons
#    - `data`: Dict mapping bodypart names to (n_frames, n_dims) arrays
#    - `skeleton`: Skeleton object defining nodes, edges, and styling
#    - `colors`: Per-bodypart colors (or use skeleton.node_colors)
#
# 3. **HeadDirectionOverlay**: Orientation arrows
#    - `data`: (n_frames,) angles in radians OR (n_frames, n_dims) unit vectors
#    - `color`, `length`: Arrow appearance
#
# ### Temporal Alignment
#
# - Add `times` parameter to overlay for timestamps
# - Add `frame_times` parameter to `animate_fields()` for field timestamps
# - Interpolation resamples known timestamps; it does not synchronize separate clocks
# - Use a shared clock; split observation gaps before interpolation
#
# ### Backend Capabilities
#
# - **Napari**: Full support (all overlay types + regions)
# - **Video**: Full support (all overlay types + regions)
# - **HTML**: Partial support (position + regions only, warns for others)
# - **Widget**: Full support (all overlay types + regions)
#
# ### Multi-Animal Support
#
# - Pass multiple overlay instances in a list
# - Each overlay automatically gets a suffix (e.g., "Position_1", "Position_2")
# - Use different colors to distinguish animals
#
# ### Common Patterns
#
# ```python
# # Simple trajectory overlay
# from neurospatial.animation import PositionOverlay
# overlay = PositionOverlay(positions=trajectory, color="red", trail_length=10)
# env.animate_fields(fields, frame_times=frame_times, overlays=[overlay], backend="napari")
#
# # Pose with skeleton
# from neurospatial.animation import BodypartOverlay
# from neurospatial.animation import Skeleton
# skeleton = Skeleton(
#     name="mouse",
#     nodes=("nose", "body", "tail"),
#     edges=(("tail", "body"), ("body", "nose")),
#     edge_color="white",
#     edge_width=2.0,
# )
# overlay = BodypartOverlay(
#     data={"nose": nose_traj, "body": body_traj, "tail": tail_traj},
#     skeleton=skeleton,
# )
# env.animate_fields(fields, frame_times=frame_times, overlays=[overlay], backend="napari")
#
# # Mixed-rate alignment
# overlay = PositionOverlay(positions=trajectory_120hz, times=times_120hz)
# env.animate_fields(
#     fields_10hz,
#     overlays=[overlay],
#     frame_times=times_10hz,  # Automatic interpolation
#     backend="napari"
# )
#
# # Multi-animal
# env.animate_fields(
#     fields,
#     frame_times=frame_times,
#     overlays=[overlay_animal1, overlay_animal2],
#     backend="napari"
# )
#
# # Show regions
# env.animate_fields(
#     fields,
#     frame_times=frame_times,
#     overlays=[overlay],
#     show_regions=True,
#     region_alpha=0.3,
#     backend="napari"
# )
# ```
#
# ### Performance Tips
#
# - **Video export**: Use `n_workers > 1` for parallel rendering
# - **Large datasets**: Use Napari for exploration, subsample for video
# - **HTML file size**: Limit frames (default max 500) or use video backend
# - **Parallel rendering**: Call `env.clear_cache()` before video export with `n_workers > 1`
#
# ## Next Steps
#
# - Apply overlays to your own behavioral tracking data
# - Combine multiple overlay types for rich visualizations
# - Export publication-quality videos with overlays
# - Use temporal alignment for multi-modal data (tracking + neural recordings)
#
# For more details, see:
# - [18: video overlays](18_video_overlay.ipynb) — calibration and synchronized video
# - [16: field animation](16_field_animation.ipynb) — backends without overlays
