"""Minimal test to isolate napari playback issue."""

import napari
import numpy as np
from napari.settings import get_settings

# Create simple test data
n_frames = 1000
height, width = 100, 100
data = np.random.rand(n_frames, height, width).astype(np.float32)

print(f"Test data: {data.shape} ({data.nbytes / 1e6:.1f} MB)")

# Create viewer
viewer = napari.Viewer()
viewer.add_image(data, name="Test")

# Configure playback

settings = get_settings()
print(f"Default playback_fps: {settings.application.playback_fps}")

# Set to 30 fps (original default)
settings.application.playback_fps = 30
print(f"Set playback_fps to: {settings.application.playback_fps}")

print("\nInstructions:")
print("  1. Press Space to start playback")
print("  2. Observe if playback is smooth or gets stuck")
print("  3. Close viewer to exit")

napari.run()
