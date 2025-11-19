"""Demo of HTML animation backend with moving Gaussian field."""

import numpy as np

from neurospatial import Environment
from neurospatial.animation.backends.html_backend import render_html

# Create a 2D environment
np.random.seed(42)
positions = np.random.randn(500, 2) * 20
env = Environment.from_samples(positions, bin_size=2.0)

print(f"Environment: {env.n_bins} bins")

# Create animated fields: moving Gaussian bump
n_frames = 30
fields = []

for i in range(n_frames):
    # Moving center position
    t = i / n_frames
    center_x = -15 + 30 * t  # Move from left to right
    center_y = -10 + 20 * np.sin(2 * np.pi * t)  # Oscillate up/down

    # Compute Gaussian field
    field = np.zeros(env.n_bins)
    for bin_idx in range(env.n_bins):
        x, y = env.bin_centers[bin_idx]
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        field[bin_idx] = np.exp(-(dist**2) / (2 * 5**2))

    fields.append(field)

# Export to HTML
output_path = render_html(
    env,
    fields,
    save_path="animation_demo.html",
    fps=10,
    cmap="viridis",
    title="Moving Gaussian Field Demo",
    dpi=100,
)

print(f"\n✓ Demo saved to: {output_path}")
print("  Open in browser to see animated field with:")
print("  - Play/pause controls")
print("  - Frame scrubbing slider")
print("  - Speed control (0.25x to 4x)")
print("  - Keyboard shortcuts (space, arrows)")
