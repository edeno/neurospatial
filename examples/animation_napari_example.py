"""Example: Napari interactive animation backend.

This example demonstrates the napari backend for GPU-accelerated interactive
viewing of spatial field animations. Napari provides:
- Interactive 3D viewer with built-in controls
- GPU-accelerated rendering (handles 100K+ frames)
- Lazy loading via LRU cache (500 frames cached by default)
- Trajectory overlay support
- High-performance seeking (<100ms for any frame)

Requirements:
    pip install "napari[all]>=0.4.18,<0.6"
    # or: uv add "napari[all]>=0.4.18,<0.6"
"""

import napari
import numpy as np

from neurospatial import Environment
from neurospatial.animation.backends.napari_backend import render_napari

# ============================================================================
# Example 1: Basic Napari Animation
# ============================================================================

print("Example 1: Basic Napari Animation")
print("=" * 60)

# Create a 2D environment
positions = np.random.randn(100, 2) * 50
env = Environment.from_samples(positions, bin_size=10.0)
print(f"Environment: {env.n_bins} bins, 2D")

# Create animated fields (e.g., place field activity over time)
n_frames = 100
fields = []
for i in range(n_frames):
    # Simulate a moving "bump" of activity
    center = np.array([np.sin(i / 10) * 40, np.cos(i / 10) * 40])
    field = np.exp(-np.sum((env.bin_centers - center) ** 2, axis=1) / 200)
    fields.append(field)

print(f"Fields: {n_frames} frames")
print("\nLaunching napari viewer...")
print("Controls:")
print("  - Play/pause: Spacebar")
print("  - Seek: Slider at bottom")
print("  - Zoom: Mouse wheel")
print("  - Pan: Click and drag")
print("\nClose the napari window to continue.")

# Launch napari viewer (interactive GUI window)
viewer = render_napari(env, fields, title="Basic Napari Animation", fps=30)

# IMPORTANT: Call napari.run() to block and keep window open
napari.run()

print("\nViewer closed.")


# ============================================================================
# Example 2: Napari with Trajectory Overlay
# ============================================================================

print("\n\nExample 2: Napari with Trajectory Overlay")
print("=" * 60)

# Create trajectory (e.g., animal path through environment)
n_frames = 200
trajectory = np.zeros((n_frames, 2))
for i in range(n_frames):
    t = i / 20
    trajectory[i] = [np.sin(t) * 40, np.cos(t) * 40]

# Create fields with corresponding trajectory
fields_with_traj = []
for i in range(n_frames):
    center = trajectory[i]
    field = np.exp(-np.sum((env.bin_centers - center) ** 2, axis=1) / 200)
    fields_with_traj.append(field)

print(f"Trajectory: {n_frames} time points")
print("\nLaunching napari viewer with trajectory overlay...")
print("Note: Blue line shows the animal's path through the environment")

# Launch with trajectory
viewer = render_napari(
    env,
    fields_with_traj,
    trajectory=trajectory,
    title="Napari with Trajectory",
    fps=30,
    cmap="hot",
)

napari.run()

print("\nViewer closed.")


# ============================================================================
# Example 3: Large-Scale Animation (Lazy Loading Demo)
# ============================================================================

print("\n\nExample 3: Large-Scale Animation (Lazy Loading Demo)")
print("=" * 60)

# Create a large dataset (1000 frames - napari handles this efficiently)
n_frames_large = 1000
print(f"Creating {n_frames_large} frames...")

# For memory efficiency, we can use a generator or list
# Napari's LazyFieldRenderer will cache the first 500 frames
# and render others on-demand
fields_large = []
for i in range(n_frames_large):
    t = i / 100
    center = np.array([np.sin(t) * 40, np.cos(t) * 40])
    field = np.exp(-np.sum((env.bin_centers - center) ** 2, axis=1) / 300)
    fields_large.append(field)

print(f"Fields created: {len(fields_large)} frames")
print("\nLaunching napari with lazy loading...")
print("Note: First 500 frames are cached, remaining frames render on-demand")
print("Try seeking to different frames - performance should be consistent!")

viewer = render_napari(
    env,
    fields_large,
    title=f"Large Dataset ({n_frames_large} frames)",
    fps=60,  # Higher FPS for smooth playback
    cmap="viridis",
)

napari.run()

print("\nViewer closed.")


# ============================================================================
# Example 4: Custom Colormap and Normalization
# ============================================================================

print("\n\nExample 4: Custom Colormap and Normalization")
print("=" * 60)

# Create fields with different scales
fields_mixed = []
for i in range(50):
    scale = 1.0 + np.sin(i / 5) * 0.5  # Varying intensity
    center = np.array([0, 0])
    field = np.exp(-np.sum((env.bin_centers - center) ** 2, axis=1) / 200) * scale
    fields_mixed.append(field)

# Override color scale for consistent normalization
vmin, vmax = 0.0, 1.5
print(f"Color scale: [{vmin}, {vmax}]")
print("Colormap: plasma")
print("\nLaunching napari with custom color scale...")

viewer = render_napari(
    env,
    fields_mixed,
    title="Custom Color Scale",
    fps=20,
    cmap="plasma",
    vmin=vmin,
    vmax=vmax,
)

napari.run()

print("\nViewer closed.")


# ============================================================================
# Performance Notes
# ============================================================================

print("\n\nPerformance Notes")
print("=" * 60)
print("Napari Backend Features:")
print("  ✓ GPU-accelerated rendering (fast for large datasets)")
print("  ✓ Lazy loading with LRU cache (500 frames cached by default)")
print("  ✓ Interactive controls (play, pause, seek, zoom, pan)")
print("  ✓ Trajectory overlay support (2D tracks)")
print("  ✓ Memory efficient (doesn't load all frames at once)")
print("  ✓ Suitable for 100K+ frame datasets (hour-long sessions at 250 Hz)")
print("\nUse Cases:")
print("  - Interactive exploration of long recordings")
print("  - GPU-accelerated playback on powerful workstations")
print("  - Real-time analysis during experiments")
print("  - Comparing multiple sessions side-by-side")
print("\nWhen to use:")
print("  - Need interactive controls (zoom, pan, seek)")
print("  - Working with >1000 frames")
print("  - Have GPU available")
print("  - Want to overlay trajectories")
print("\nWhen NOT to use:")
print("  - Remote server (no display)")
print("  - Need to share animations (use video or HTML backend)")
print("  - Simple quick visualizations (use HTML backend)")
