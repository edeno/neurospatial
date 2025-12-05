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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Video Overlay
#
# This notebook demonstrates the VideoOverlay feature for compositing raw behavioral video frames with spatial field animations:
#
# 1. **Loading Video Metadata** - Inspect video dimensions, frame rate, and duration
# 2. **Scale Bar Calibration** - Map pixels to cm using a known-length scale bar
# 3. **Landmark Calibration** - Map pixels to cm using corresponding arena corners
# 4. **VideoOverlay Options** - Control alpha, z-order, crop, and downsample
# 5. **Animation with Video Background** - Composite video beneath spatial fields
# 6. **Exporting Synchronized Video** - Export video with overlays
# 7. **Performance Tips** - Handle large videos efficiently
#
# **Estimated time**: 15-20 minutes
#
# ## Learning Objectives
#
# By the end of this notebook, you will be able to:
#
# - Create video calibrations using scale bars or landmark correspondences
# - Overlay behavioral video on animated spatial fields
# - Control video appearance with alpha blending and z-order
# - Export synchronized video files
# - Handle memory-efficient streaming for large videos
#
# ## Prerequisites
#
# **Required dependencies**:
#
# ```bash
# # OpenCV for video reading
# pip install opencv-python>=4.11.0
#
# # imageio for video creation in examples
# pip install imageio>=2.35.0 imageio-ffmpeg>=0.5.1
# ```
#
# **Optional dependencies**:
#
# ```bash
# # For Napari backend (recommended for interactive viewing)
# pip install 'napari[all]>=0.4.18'
#
# # For video export
# # macOS: brew install ffmpeg
# # Ubuntu: sudo apt install ffmpeg
# ```
#
# **Note**: VideoOverlay requires 2D environments (not supported on 1D linearized tracks).

# %%
from pathlib import Path

import numpy as np

from neurospatial import Environment, PositionOverlay
from neurospatial.animation import VideoOverlay, calibrate_video
from neurospatial.animation.backends.video_backend import check_ffmpeg_available
from neurospatial.transforms import (
    VideoCalibration,
    calibrate_from_landmarks,
    calibrate_from_scale_bar,
)

# Set random seed for reproducibility
np.random.seed(42)

# Output directory
output_dir = Path.cwd()
print(f"Output directory: {output_dir}")

# %% [markdown]
# ## Setup: Create Environment and Synthetic Video
#
# We'll create:
# 1. A square arena environment (100 x 100 cm)
# 2. A synthetic video with a moving gradient pattern
# 3. A simulated place field and trajectory

# %%
print("Creating square arena environment...")

# Square arena (100 x 100 cm)
positions = np.array(
    [
        [5, 5],
        [95, 5],
        [95, 95],
        [5, 95],  # Corners
        [50, 50],  # Center
    ]
)

env = Environment.from_samples(
    positions=np.random.uniform(0, 100, (1000, 2)),  # Fill arena
    bin_size=2.5,
    name="SquareArena",
)
env.units = "cm"
env.frame = "behavioral_box"

print(f"Environment: {env.n_bins} bins, {env.n_dims}D")
print(f"Spatial extent: {env.dimension_ranges}")

# %%
print("\nCreating synthetic video...")

try:
    import imageio.v3 as iio
except ImportError:
    import imageio as iio

# Video parameters
video_width, video_height = 320, 240  # pixels
n_video_frames = 100
video_fps = 30.0
video_path = output_dir / "synthetic_video.mp4"

# Create frames with moving gradient pattern
frames = []
for i in range(n_video_frames):
    # Create gradient that shifts over time
    x = np.linspace(0, 1, video_width)
    y = np.linspace(0, 1, video_height)
    X, Y = np.meshgrid(x, y)

    # Phase shifts create moving pattern
    phase = 2 * np.pi * i / n_video_frames

    # RGB channels with different patterns
    R = (128 + 127 * np.sin(4 * np.pi * X + phase)).astype(np.uint8)
    G = (128 + 127 * np.sin(4 * np.pi * Y + phase)).astype(np.uint8)
    B = (128 + 127 * np.cos(4 * np.pi * (X + Y) + phase)).astype(np.uint8)

    frame = np.stack([R, G, B], axis=-1)
    frames.append(frame)

# Write video file
frames_array = np.array(frames)
iio.imwrite(video_path, frames_array, fps=video_fps)

print(f"Video created: {video_path}")
print(f"  Size: {video_width}x{video_height} pixels")
print(f"  Frames: {n_video_frames} at {video_fps} fps")
print(f"  Duration: {n_video_frames / video_fps:.1f} seconds")

# %%
print("\nSimulating trajectory and place field...")

# Number of animation frames (match video)
n_frames = n_video_frames

# Trajectory: spiral through arena (ensure within bounds)
t = np.linspace(0, 4 * np.pi, n_frames)
r = np.linspace(10, 40, n_frames)  # Radius increases (stay within 100x100 arena)
trajectory = np.column_stack(
    [
        50 + r * np.cos(t),  # Center at (50, 50), max radius 40 -> range [10, 90]
        50 + r * np.sin(t),
    ]
)

# Clip trajectory to environment bounds (with small margin)
trajectory = np.clip(trajectory, 5, 95)

# Place field following trajectory
fields = []
for i in range(n_frames):
    pos = trajectory[i : i + 1]
    center_bin = env.bin_at(pos)[0]
    if center_bin >= 0:  # Valid bin
        distances = env.distance_to([center_bin])
        field = np.exp(-(distances**2) / (2 * 10.0**2))
    else:
        # Position outside environment - use uniform low field
        field = np.ones(env.n_bins) * 0.1
    field = np.clip(field + np.random.randn(env.n_bins) * 0.05, 0, 1)
    fields.append(field)

fields = np.array(fields)

# Create frame_times (required for animate_fields)
frame_times = np.arange(n_frames) / video_fps  # seconds

print(f"Trajectory: {n_frames} frames")
print(
    f"Trajectory range: x=[{trajectory[:, 0].min():.1f}, {trajectory[:, 0].max():.1f}], y=[{trajectory[:, 1].min():.1f}, {trajectory[:, 1].max():.1f}]"
)
print(f"Fields: {fields.shape}")

# %% [markdown]
# ## Example 1: Loading and Inspecting Video Metadata
#
# Before calibration, inspect the video properties using the `VideoReader` class.
#
# **Key properties**:
# - `n_frames`: Total frame count
# - `fps`: Frame rate
# - `frame_size_px`: (width, height) in pixels
# - `duration`: Video length in seconds

# %%
from neurospatial.animation._video_io import VideoReader  # noqa: E402

print("Example 1: Inspecting Video Metadata")
print("=" * 50)

# Create reader to inspect video
reader = VideoReader(video_path)

print(f"Video file: {video_path.name}")
print(f"  Frame size: {reader.frame_size_px} (width, height)")
print(f"  Frame count: {reader.n_frames}")
print(f"  Frame rate: {reader.fps} fps")
print(f"  Duration: {reader.duration:.2f} seconds")

# Access a single frame
frame_0 = reader[0]
print(f"\nFrame 0 shape: {frame_0.shape} (height, width, channels)")
print(f"Frame 0 dtype: {frame_0.dtype}")

# %% [markdown]
# ## Example 2: Calibrating with Scale Bar Method
#
# The scale bar method uses two known points in the video and their real-world distance.
#
# **Use case**: When you have a ruler or known-length object visible in the video.
#
# **Parameters**:
# - `p1_px`, `p2_px`: Endpoints of scale bar in pixels (x, y)
# - `known_length_cm`: Real-world length in cm

# %%
print("Example 2: Scale Bar Calibration")
print("=" * 50)

# Assume we have a scale bar spanning the video width
# Video is 320 pixels wide, representing 100 cm
p1_px = (10, 120)  # Left end of scale bar (x, y) in pixels
p2_px = (310, 120)  # Right end of scale bar
known_length_cm = 100.0  # 300 pixels = 100 cm

# Method 1: Using calibrate_from_scale_bar directly
transform = calibrate_from_scale_bar(
    p1_px=p1_px,
    p2_px=p2_px,
    known_length_cm=known_length_cm,
    frame_size_px=(video_width, video_height),
)

calibration_scalebar = VideoCalibration(
    transform_px_to_cm=transform,
    frame_size_px=(video_width, video_height),
)

print(f"Scale bar: {p1_px} to {p2_px}")
print(f"Known length: {known_length_cm} cm")
print(f"Computed scale: {calibration_scalebar.cm_per_px:.4f} cm/px")

# Verify calibration by transforming test points
test_px = np.array([[160, 120]])  # Center of video
test_cm = calibration_scalebar.transform_px_to_cm(test_px)
print(f"\nCenter pixel {test_px[0]} maps to {test_cm[0]} cm")

# %%
print("\nMethod 2: Using calibrate_video() convenience function")
print("-" * 50)

# The convenience function combines calibration and validation
calibration_easy = calibrate_video(
    video_path,
    env,
    scale_bar=(p1_px, p2_px, known_length_cm),
)

print("Calibration created successfully!")
print(f"  cm_per_px: {calibration_easy.cm_per_px:.4f}")
print(f"  frame_size_px: {calibration_easy.frame_size_px}")

# %% [markdown]
# ## Example 3: Calibrating with Landmark Correspondences
#
# The landmark method uses multiple corresponding points between video and environment.
#
# **Use case**: When you know the pixel locations of arena corners or markers.
#
# **Parameters**:
# - `landmarks_px`: Points in video pixels (n_points, 2)
# - `landmarks_env`: Corresponding points in environment cm (n_points, 2)
#
# **Note**: Use at least 3 non-collinear points for reliable calibration.

# %%
print("Example 3: Landmark Calibration")
print("=" * 50)

# Arena corners in video pixels
# Assuming video shows arena with some margin
landmarks_px = np.array(
    [
        [10, 10],  # Top-left corner (pixel coords, origin top-left)
        [310, 10],  # Top-right corner
        [310, 230],  # Bottom-right corner
        [10, 230],  # Bottom-left corner
    ]
)

# Corresponding arena corners in environment coordinates
# Environment origin is bottom-left, Y increases upward
landmarks_env = np.array(
    [
        [0, 100],  # Top-left (x=0, y=max)
        [100, 100],  # Top-right (x=max, y=max)
        [100, 0],  # Bottom-right (x=max, y=0)
        [0, 0],  # Bottom-left (x=0, y=0)
    ]
)

print("Landmark correspondences:")
for i, (px, cm) in enumerate(zip(landmarks_px, landmarks_env, strict=True)):
    print(f"  Point {i + 1}: pixel {px} -> env {cm} cm")

# Method 1: Using calibrate_from_landmarks directly
transform_lm = calibrate_from_landmarks(
    landmarks_px=landmarks_px,
    landmarks_cm=landmarks_env,
    frame_size_px=(video_width, video_height),
    kind="similarity",  # or "rigid", "affine"
)

calibration_landmarks = VideoCalibration(
    transform_px_to_cm=transform_lm,
    frame_size_px=(video_width, video_height),
)

print("\nCalibration created (similarity transform)")
print(f"  cm_per_px: {calibration_landmarks.cm_per_px:.4f}")

# %%
print("\nMethod 2: Using calibrate_video() convenience function")
print("-" * 50)

calibration_lm_easy = calibrate_video(
    video_path,
    env,
    landmarks_px=landmarks_px,
    landmarks_env=landmarks_env,
)

print("Calibration created successfully!")
print(f"  cm_per_px: {calibration_lm_easy.cm_per_px:.4f}")

# Verify round-trip accuracy
transformed = calibration_lm_easy.transform_px_to_cm(landmarks_px)
error = np.abs(transformed - landmarks_env).max()
print(f"\nRound-trip error: {error:.6f} cm (should be ~0)")

# %% [markdown]
# ## Example 4: Creating VideoOverlay with Various Options
#
# VideoOverlay supports several options for controlling appearance:
#
# | Option | Description | Default |
# |--------|-------------|----------|
# | `alpha` | Opacity (0.0-1.0) | 0.5 |
# | `z_order` | "above" or "below" field | "above" |
# | `crop` | (x, y, width, height) | None |
# | `downsample` | Spatial downsampling factor | 1 |
# | `times` | Video frame timestamps | Auto from fps |

# %%
print("Example 4a: Basic VideoOverlay")
print("=" * 50)

# Basic video overlay with calibration
video_overlay = VideoOverlay(
    source=video_path,
    calibration=calibration_landmarks,
)

print("VideoOverlay created:")
print(f"  source: {video_path.name}")
print(f"  alpha: {video_overlay.alpha} (default)")
print(f"  z_order: {video_overlay.z_order} (default)")

# %%
print("\nExample 4b: Alpha Blending Options")
print("-" * 50)

# Field dominant (low alpha)
overlay_field_dominant = VideoOverlay(
    source=video_path,
    calibration=calibration_landmarks,
    alpha=0.3,  # 30% video, 70% field
)

# Balanced (default)
overlay_balanced = VideoOverlay(
    source=video_path,
    calibration=calibration_landmarks,
    alpha=0.5,  # 50% video, 50% field
)

# Video dominant (high alpha)
overlay_video_dominant = VideoOverlay(
    source=video_path,
    calibration=calibration_landmarks,
    alpha=0.7,  # 70% video, 30% field
)

print("Alpha options:")
print("  alpha=0.3: Field shows through video (field dominant)")
print("  alpha=0.5: Equal visibility (balanced, default)")
print("  alpha=0.7: Video shows through field (video dominant)")

# %%
print("\nExample 4c: Z-Order Options")
print("-" * 50)

# Video above field (default)
overlay_above = VideoOverlay(
    source=video_path,
    calibration=calibration_landmarks,
    z_order="above",  # Video on top of field
)

# Video below field (only visible if field has transparency)
overlay_below = VideoOverlay(
    source=video_path,
    calibration=calibration_landmarks,
    z_order="below",  # Video behind field
)

print("Z-order options:")
print("  z_order='above': Video on top (default, works with opaque fields)")
print("  z_order='below': Video behind (only visible if field has NaN/transparent)")

# %%
print("\nExample 4d: Crop and Downsample")
print("-" * 50)

# Crop to region of interest
overlay_cropped = VideoOverlay(
    source=video_path,
    calibration=calibration_landmarks,
    crop=(50, 30, 200, 150),  # (x, y, width, height) in pixels
)

# Downsample for faster rendering
overlay_downsampled = VideoOverlay(
    source=video_path,
    calibration=calibration_landmarks,
    downsample=2,  # Half resolution (160x120)
)

print("Advanced options:")
print("  crop=(50, 30, 200, 150): Crop to 200x150 region starting at (50, 30)")
print("  downsample=2: Reduce resolution by factor of 2")

# %% [markdown]
# ## Example 5: Animating Fields with Video Background
#
# Combine the spatial field animation with the video overlay using `animate_fields()`.
#
# **Supported backends**:
# - `napari`: Best for interactive exploration
# - `video`: Best for exporting MP4 files
# - `widget`: For Jupyter notebook playback
# - `html`: NOT supported for VideoOverlay (warning emitted)

# %%
print("Example 5a: Napari Backend (Interactive)")
print("=" * 50)

# Create video overlay with position tracking
video_overlay = VideoOverlay(
    source=video_path,
    calibration=calibration_landmarks,
    alpha=0.5,
)

position_overlay = PositionOverlay(
    data=trajectory,
    color="red",
    size=10.0,
    trail_length=10,
)

try:
    import napari
    from IPython import get_ipython

    print("Launching Napari viewer...")

    # IMPORTANT: Clear cache before parallel rendering
    env.clear_cache()

    viewer = env.animate_fields(
        fields,
        overlays=[video_overlay, position_overlay],
        frame_times=frame_times,
        backend="napari",
        fps=video_fps,
        title="Video + Field Animation",
    )

    print("\nNapari viewer opened:")
    print("  - Video layer with field overlay")
    print("  - Position tracking with trail")
    print("  - Use slider or play button to animate")

    if get_ipython() is None:
        napari.run()

except ImportError:
    print("Napari not available. Install with: pip install 'napari[all]>=0.4.18'")

# %% [markdown]
# ## Example 6: Exporting Synchronized Video
#
# Export the animation with video overlay to an MP4 file.
#
# **Key parameters**:
# - `n_workers`: Number of parallel workers (speeds up rendering)
# - `fps`: Output frame rate
# - `dpi`: Output resolution
#
# **Important**: Call `env.clear_cache()` before parallel rendering!

# %%
print("Example 6: Exporting Video with Overlay")
print("=" * 50)

if check_ffmpeg_available():
    output_path = output_dir / "18_video_overlay_export.mp4"

    # IMPORTANT: Clear cache before parallel rendering
    env.clear_cache()

    result = env.animate_fields(
        fields,
        overlays=[video_overlay, position_overlay],
        frame_times=frame_times,
        backend="video",
        save_path=output_path,
        fps=video_fps,
        n_workers=4,  # Parallel rendering
        dpi=100,
    )

    print(f"\nVideo exported to: {result}")
    print(f"  Frames: {n_frames}")
    print(f"  FPS: {video_fps}")
    print("  Workers: 4 (parallel)")
else:
    print("ffmpeg not available for video export.")
    print("Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)")

# %%
print("Example 6b: Widget Backend (Jupyter Playback)")
print("-" * 50)

try:
    from IPython import get_ipython

    if get_ipython() is not None:
        # Clear cache before rendering
        env.clear_cache()

        # Use only first 30 frames for quick demo
        n_demo = 30

        # Create video timestamps for the full video
        video_times = np.arange(n_video_frames) / video_fps
        # Create field timestamps for the demo subset
        field_times = video_times[:n_demo]

        widget = env.animate_fields(
            fields[:n_demo],
            overlays=[
                VideoOverlay(
                    source=video_path,
                    calibration=calibration_landmarks,
                    times=video_times,  # Full video timestamps
                    alpha=0.5,
                ),
                PositionOverlay(
                    data=trajectory[:n_demo],
                    color="red",
                    trail_length=10,
                ),
            ],
            frame_times=field_times,  # Field timestamps for alignment
            backend="widget",
            fps=10,
        )
        print("Widget created - use slider to navigate")
        display(widget)  # noqa: F821
    else:
        print("Not in Jupyter notebook environment")

except ImportError:
    print("IPython/ipywidgets not available")

# %% [markdown]
# ## Example 7: Performance Tips for Large Videos
#
# Behavioral videos can be 30+ minutes at 30fps = 50,000+ frames. Here's how to handle them efficiently:
#
# ### Memory Management
#
# 1. **LRU Caching**: VideoReader caches recently accessed frames (default: 100 frames)
# 2. **Streaming**: Frames are loaded on-demand, never all at once
# 3. **Downsampling**: Reduce resolution for faster rendering
#
# ### Rendering Optimization
#
# 1. **Parallel Export**: Use `n_workers > 1` for video export
# 2. **Subsampling**: Reduce frame count for preview
# 3. **Clear Cache**: Always call `env.clear_cache()` before parallel rendering

# %%
from neurospatial.animation import subsample_frames  # noqa: E402

print("Example 7: Performance Optimization")
print("=" * 50)

# Tip 1: Adjust cache size for your workflow
reader_small_cache = VideoReader(video_path, cache_size=50)
reader_large_cache = VideoReader(video_path, cache_size=200)

print("Tip 1: Adjust cache size")
print("  cache_size=50: Low memory, good for random access")
print("  cache_size=200: Higher memory, better for sequential playback")

# Tip 2: Downsample for faster rendering
print("\nTip 2: Downsample for speed")
overlay_fast = VideoOverlay(
    source=video_path,
    calibration=calibration_landmarks,
    downsample=2,  # Half resolution
)
print(f"  Original: {video_width}x{video_height} pixels")
print(f"  Downsampled: {video_width // 2}x{video_height // 2} pixels")

# Tip 3: Subsample frames for preview
print("\nTip 3: Subsample frames for quick preview")
# Subsample from 30fps to 10fps
fields_subsampled = subsample_frames(fields, source_fps=30, target_fps=10)
print(f"  Original: {len(fields)} frames at 30fps")
print(f"  Subsampled: {len(fields_subsampled)} frames at 10fps")

# %%
print("\nTip 4: CRITICAL - Clear cache before parallel rendering")
print("-" * 50)

# This is REQUIRED for parallel video export
env.clear_cache()

print("Always call env.clear_cache() before animate_fields() with n_workers > 1")
print("")
print("Why? The Environment object must be pickle-able for multiprocessing.")
print("Cached KDTree and kernel matrices cannot be pickled.")
print("")
print("Example:")
print("  env.clear_cache()  # Make environment pickle-able")
print("  env.animate_fields(fields, overlays=[video], n_workers=4, ...)")

# %% [markdown]
# ## Cleanup
#
# Remove the synthetic video file created for this example.

# %%
# Cleanup synthetic video
if video_path.exists():
    video_path.unlink()
    print(f"Removed: {video_path}")

# Cleanup exported video
export_path = output_dir / "18_video_overlay_export.mp4"
if export_path.exists():
    export_path.unlink()
    print(f"Removed: {export_path}")

print("\nCleanup complete!")

# %% [markdown]
# ## Key Takeaways
#
# ### Calibration Methods
#
# 1. **Scale Bar**: Use when you have a known-length reference in the video
#    ```python
#    calibration = calibrate_video(
#        "video.mp4", env,
#        scale_bar=((x1, y1), (x2, y2), length_cm)
#    )
#    ```
#
# 2. **Landmarks**: Use when you know pixel locations of arena corners
#    ```python
#    calibration = calibrate_video(
#        "video.mp4", env,
#        landmarks_px=corners_px,
#        landmarks_env=corners_cm
#    )
#    ```
#
# 3. **Direct Scale**: Use when you know the exact cm/pixel ratio
#    ```python
#    calibration = calibrate_video(
#        "video.mp4", env,
#        cm_per_px=0.25
#    )
#    ```
#
# ### VideoOverlay Best Practices
#
# | Goal | Settings |
# |------|----------|
# | Balanced view | `alpha=0.5, z_order="above"` (default) |
# | Field dominant | `alpha=0.3, z_order="above"` |
# | Video dominant | `alpha=0.7, z_order="above"` |
# | Video background | `z_order="below"` (needs transparent field) |
#
# ### Backend Support
#
# | Backend | VideoOverlay Support |
# |---------|---------------------|
# | Napari | Full support |
# | Video | Full support |
# | Widget | Full support |
# | HTML | NOT supported (warning emitted) |
#
# ### Performance Checklist
#
# - [ ] Call `env.clear_cache()` before parallel rendering
# - [ ] Use `downsample=2` or higher for faster preview
# - [ ] Use `subsample_frames()` to reduce frame count
# - [ ] Adjust `cache_size` in VideoReader based on workflow
#
# ## Next Steps
#
# - Apply VideoOverlay to your own behavioral recordings
# - Combine with other overlays (Position, Bodypart, HeadDirection)
# - Export publication-quality videos with synchronized behavior
#
# For more details, see:
# - `examples/17_animation_with_overlays.ipynb` - Other overlay types
# - `examples/16_field_animation.ipynb` - Animation backends
