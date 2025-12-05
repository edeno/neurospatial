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

# %%
from pathlib import Path

import numpy as np
from shapely.geometry import Point

from neurospatial import (
    BodypartOverlay,
    Environment,
    HeadDirectionOverlay,
    PositionOverlay,
)
from neurospatial.animation import Skeleton
from neurospatial.animation.backends.video_backend import check_ffmpeg_available

# Set random seed for reproducibility
np.random.seed(42)

# Output directory
output_dir = Path.cwd()
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
head_angles = theta + np.pi / 2  # Perpendicular to radius

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
)

print("Pose: 3 keypoints (nose, body, tail)")
print(f"Skeleton: {skeleton.n_edges} edges")
print("  Node colors and edge styling defined in Skeleton object")

# %%
print("\nSimulating place field that tracks with animal...")

# Place field centered on animal position at each frame
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
    data=trajectory,
    color="red",
    size=12.0,
    trail_length=10,  # Show last 10 frames as trail
)

print("Example 1: Position Overlay with Trail")
print(f"  Trajectory: {trajectory.shape[0]} frames")
print("  Trail length: 10 frames (decaying alpha)")
print("  Color: red, Size: 12.0")

try:
    import napari
    from IPython import get_ipython

    print("\nLaunching Napari viewer...")
    viewer = env.animate_fields(
        fields,
        overlays=[position_overlay],
        backend="napari",
        fps=10,
        title="Position Overlay with Trail",
    )

    print("✓ Napari viewer opened")
    print("  Watch the red trail follow the animal")

    if get_ipython() is None:
        napari.run()

except ImportError:
    print("⊗ Napari not available. Install with: pip install 'napari[all]>=0.4.18'")

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

try:
    import napari
    from IPython import get_ipython

    print("\nLaunching Napari viewer...")
    viewer = env.animate_fields(
        fields,
        overlays=[bodypart_overlay],
        backend="napari",
        fps=10,
        title="Pose Tracking with Skeleton",
    )

    print("✓ Napari viewer opened")
    print("  Watch the skeleton follow the animal pose")

    if get_ipython() is None:
        napari.run()

except ImportError:
    print("⊗ Napari not available. Install with: pip install 'napari[all]>=0.4.18'")

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
    data=head_angles,
    color="yellow",
    length=15.0,  # Arrow length in cm
)

print("Example 3: Head Direction Visualization")
print(f"  Head angles: {head_angles.shape[0]} frames")
print("  Arrow color: yellow, Length: 15.0 cm")

try:
    import napari
    from IPython import get_ipython

    print("\nLaunching Napari viewer...")

    # Combine position + head direction overlays
    viewer = env.animate_fields(
        fields,
        overlays=[position_overlay, head_direction_overlay],
        backend="napari",
        fps=10,
        title="Position + Head Direction",
    )

    print("✓ Napari viewer opened")
    print("  Watch the yellow arrow show heading direction")

    if get_ipython() is None:
        napari.run()

except ImportError:
    print("⊗ Napari not available. Install with: pip install 'napari[all]>=0.4.18'")

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

# Second animal with offset trajectory
trajectory_2 = trajectory + np.array([10, -10])  # Offset spatially
trajectory_2 = np.clip(trajectory_2, 5, 95)  # Keep in bounds

# Create overlays for both animals
animal1_overlay = PositionOverlay(
    data=trajectory, color="red", size=12.0, trail_length=8
)

animal2_overlay = PositionOverlay(
    data=trajectory_2, color="blue", size=12.0, trail_length=8
)

print("  Animal 1: red")
print("  Animal 2: blue (offset trajectory)")

try:
    import napari
    from IPython import get_ipython

    print("\nLaunching Napari viewer...")
    viewer = env.animate_fields(
        fields,
        overlays=[animal1_overlay, animal2_overlay],  # Multiple overlays
        backend="napari",
        fps=10,
        title="Multi-Animal Tracking",
    )

    print("✓ Napari viewer opened")
    print("  Watch both animals explore simultaneously")

    if get_ipython() is None:
        napari.run()

except ImportError:
    print("⊗ Napari not available. Install with: pip install 'napari[all]>=0.4.18'")

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

try:
    import napari
    from IPython import get_ipython

    print("\nLaunching Napari viewer...")
    viewer = env.animate_fields(
        fields,
        overlays=[position_overlay],
        show_regions=True,  # Show all regions
        region_alpha=0.3,  # 30% transparent
        backend="napari",
        fps=10,
        title="Position + Reward Region",
    )

    print("✓ Napari viewer opened")
    print("  Watch the animal approach the reward region")

    if get_ipython() is None:
        napari.run()

except ImportError:
    print("⊗ Napari not available. Install with: pip install 'napari[all]>=0.4.18'")

# %% [markdown]
# ## Example 6: Mixed-Rate Temporal Alignment
#
# Align overlays sampled at different rates using temporal timestamps.
#
# **Key features**:
# - Overlay `times` parameter specifies timestamps for each frame
# - `frame_times` parameter specifies field frame timestamps
# - Linear interpolation automatically aligns overlay to field frames
# - Works even when overlay and fields have different sampling rates
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
t_high = np.linspace(0, duration, n_samples_high)
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
frame_times = np.linspace(0, duration, n_frames_low)

print(f"  Field computation: {n_frames_low} frames at {fps_low} Hz")

# Compute fields at low frequency
fields_low_freq = []
for t in frame_times:
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
    data=trajectory_high_freq,
    times=timestamps_high,  # 120 Hz timestamps
    color="red",
    size=10.0,
    trail_length=15,
)

print("\n✓ Overlay will be interpolated to match field frame times")
print("  (Linear interpolation: 120 Hz → 10 Hz)")

try:
    import napari
    from IPython import get_ipython

    print("\nLaunching Napari viewer...")
    viewer = env.animate_fields(
        fields_low_freq,
        overlays=[position_overlay_timed],
        frame_times=frame_times,  # Explicit field timestamps
        backend="napari",
        fps=10,
        title="Mixed-Rate Alignment (120 Hz → 10 Hz)",
    )

    print("✓ Napari viewer opened")
    print("  Position automatically aligned to field frames")

    if get_ipython() is None:
        napari.run()

except ImportError:
    print("⊗ Napari not available. Install with: pip install 'napari[all]>=0.4.18'")

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

try:
    import napari
    from IPython import get_ipython

    # All overlay types supported
    viewer = env.animate_fields(
        fields,
        overlays=[position_overlay, bodypart_overlay, head_direction_overlay],
        show_regions=True,
        backend="napari",
        fps=10,
        title="Napari: All Overlays",
    )

    print("✓ Napari: Position + Pose + Head Direction + Regions")

    if get_ipython() is None:
        napari.run()

except ImportError:
    print("⊗ Napari not available")

# %%
print("Example 7b: Video Backend (Full Support)")

if check_ffmpeg_available():
    # All overlay types supported
    output_path = env.animate_fields(
        fields,
        overlays=[position_overlay, bodypart_overlay, head_direction_overlay],
        show_regions=True,
        backend="video",
        save_path=output_dir / "17_all_overlays.mp4",
        fps=10,
        n_workers=4,
    )
    print(f"✓ Video: Saved to {output_path}")
else:
    print("⊗ ffmpeg not available for video export")

# %%
print("Example 7c: HTML Backend (Position + Regions Only)")
print("  WARNING: HTML backend does NOT support bodypart or head direction overlays")
print("  (Warnings will be emitted if provided)\n")

# HTML: Only position and regions supported
html_path = env.animate_fields(
    fields,
    overlays=[position_overlay],  # Only position overlay
    show_regions=True,
    backend="html",
    save_path=output_dir / "17_position_only.html",
    fps=10,
)

print(f"✓ HTML: Saved to {html_path}")
print("  (Position + Regions rendered; pose/head direction not supported)")

# %%
print("Example 7d: Widget Backend (Full Support)")

try:
    from IPython import get_ipython

    if get_ipython() is not None:
        # All overlay types supported
        widget = env.animate_fields(
            fields,
            overlays=[position_overlay, bodypart_overlay, head_direction_overlay],
            show_regions=True,
            backend="widget",
            fps=10,
        )
        print("✓ Widget: Position + Pose + Head Direction + Regions")
    else:
        print("⊗ Not in Jupyter notebook")

except ImportError:
    print("⊗ IPython/ipywidgets not available")

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
# - Linear interpolation automatically aligns overlay to field frames
# - Works even when overlay and fields have different sampling rates
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
# from neurospatial import PositionOverlay
# overlay = PositionOverlay(data=trajectory, color="red", trail_length=10)
# env.animate_fields(fields, overlays=[overlay], backend="napari")
#
# # Pose with skeleton
# from neurospatial import BodypartOverlay
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
# env.animate_fields(fields, overlays=[overlay], backend="napari")
#
# # Mixed-rate alignment
# overlay = PositionOverlay(data=trajectory_120hz, times=times_120hz)
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
#     overlays=[overlay_animal1, overlay_animal2],
#     backend="napari"
# )
#
# # Show regions
# env.animate_fields(
#     fields,
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
# - `docs/animation_overlays.md` - Complete overlay documentation
# - `examples/16_field_animation.ipynb` - Animation backends without overlays
