"""Spatial View Cell Analysis.

Spatial view cells fire when the animal is *looking at* a specific location,
not when the animal is *at* that location. This example demonstrates:

1. Simulating spatial view cells
2. Computing view fields (binned by viewed location)
3. Comparing view fields to place fields
4. Classifying spatial view cells

Key difference from place cells:
- Place cell: fires when animal is AT a location
- Spatial view cell: fires when animal is LOOKING AT a location
"""

import matplotlib.pyplot as plt
import numpy as np

from neurospatial import (
    Environment,
    FieldOfView,
    SpatialViewCellModel,
    compute_place_field,
    compute_spatial_view_field,
    compute_viewed_location,
    heading_from_velocity,
    spatial_view_cell_metrics,
)
from neurospatial.simulation import PlaceCellModel, generate_poisson_spikes

# %% Create environment
rng = np.random.default_rng(42)
samples = rng.uniform(0, 100, (2000, 2))
env = Environment.from_samples(samples, bin_size=5.0)
env.units = "cm"
print(f"Environment: {env.n_bins} bins")

# %% Generate trajectory with realistic heading
n_time = 10000
dt = 0.02  # 50 Hz
times = np.arange(n_time) * dt

# Smooth random walk
velocities = np.cumsum(rng.normal(0, 0.5, (n_time, 2)), axis=0)
velocities = np.clip(velocities, -20, 20)  # Limit speed
positions = 50 + np.cumsum(velocities * dt, axis=0)
positions = np.clip(positions, 10, 90)  # Stay in bounds

# Compute heading from velocity
headings = heading_from_velocity(positions, dt, min_speed=2.0, smoothing_sigma=3.0)

print(f"Trajectory: {n_time} samples, {times[-1]:.1f}s duration")

# %% Compute viewed locations
# This is where the animal is looking, not where it is
viewed_locations = compute_viewed_location(
    positions, headings, view_distance=15.0, method="fixed_distance"
)
valid_views = np.all(np.isfinite(viewed_locations), axis=1)
print(f"Valid view samples: {np.sum(valid_views)} / {n_time}")

# %% Simulate a spatial view cell
# This cell fires when looking at (70, 50), regardless of position
preferred_view = np.array([70.0, 50.0])

svc_model = SpatialViewCellModel(
    env=env,
    preferred_view_location=preferred_view,
    view_field_width=12.0,
    view_distance=15.0,
    gaze_model="fixed_distance",
    max_rate=25.0,
    baseline_rate=0.5,
)

# Generate firing rates and spikes
svc_rates = svc_model.firing_rate(positions, times=times, headings=headings)
svc_spikes = generate_poisson_spikes(svc_rates, times, seed=42)
print(f"Spatial view cell: {len(svc_spikes)} spikes")

# %% Simulate a place cell for comparison
# This cell fires when AT (70, 50), not when looking at it
place_model = PlaceCellModel(
    env=env,
    center=np.array([70.0, 50.0]),
    width=12.0,
    max_rate=25.0,
    baseline_rate=0.5,
)

pc_rates = place_model.firing_rate(positions)
pc_spikes = generate_poisson_spikes(pc_rates, times, seed=43)
print(f"Place cell: {len(pc_spikes)} spikes")

# %% Compute view fields and place fields
# View field: binned by VIEWED location
svc_view_result = compute_spatial_view_field(
    env,
    svc_spikes,
    times,
    positions,
    headings,
    view_distance=15.0,
    method="diffusion_kde",
    bandwidth=8.0,
)
svc_view_field = svc_view_result.field

# Place field: binned by POSITION
svc_place_field = compute_place_field(
    env, svc_spikes, times, positions, method="diffusion_kde", bandwidth=8.0
)

# Same for place cell
pc_view_result = compute_spatial_view_field(
    env,
    pc_spikes,
    times,
    positions,
    headings,
    view_distance=15.0,
    method="diffusion_kde",
    bandwidth=8.0,
)
pc_view_field = pc_view_result.field
pc_place_field = compute_place_field(
    env, pc_spikes, times, positions, method="diffusion_kde", bandwidth=8.0
)

# %% Classify cells
svc_metrics = spatial_view_cell_metrics(
    env, svc_spikes, times, positions, headings, view_distance=15.0
)

pc_metrics = spatial_view_cell_metrics(
    env, pc_spikes, times, positions, headings, view_distance=15.0
)

print("\n=== Spatial View Cell Metrics ===")
print(svc_metrics.interpretation())

print("\n=== Place Cell Metrics ===")
print(pc_metrics.interpretation())

# Quick classification
print(f"\nSVC classified as spatial view cell: {svc_metrics.is_spatial_view_cell}")
print(f"Place cell classified as spatial view cell: {pc_metrics.is_spatial_view_cell}")

# %% Visualize fields
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Row 1: Spatial view cell
ax = axes[0, 0]
env.plot_field(
    svc_view_field, ax=ax, title="SVC View Field\n(binned by viewed location)"
)
ax.scatter(
    [preferred_view[0]], [preferred_view[1]], c="red", s=100, marker="*", zorder=5
)
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")

ax = axes[0, 1]
env.plot_field(svc_place_field, ax=ax, title="SVC Place Field\n(binned by position)")
ax.scatter(
    [preferred_view[0]], [preferred_view[1]], c="red", s=100, marker="*", zorder=5
)
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")

# Row 2: Place cell
ax = axes[1, 0]
env.plot_field(
    pc_view_field, ax=ax, title="Place Cell View Field\n(binned by viewed location)"
)
ax.scatter([70], [50], c="blue", s=100, marker="*", zorder=5)
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")

ax = axes[1, 1]
env.plot_field(
    pc_place_field, ax=ax, title="Place Cell Place Field\n(binned by position)"
)
ax.scatter([70], [50], c="blue", s=100, marker="*", zorder=5)
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")

fig.suptitle(
    "Spatial View Cell vs Place Cell\n"
    "Red star: preferred location | "
    "SVC view field should peak at preferred location",
    fontsize=12,
)
plt.tight_layout()
plt.savefig("spatial_view_cells_comparison.png", dpi=150)
plt.show()

# %% Field of view analysis
print("\n=== Field of View Analysis ===")
fov_rat = FieldOfView.rat()
fov_primate = FieldOfView.primate()

print(f"Rat FOV: {fov_rat.total_angle_degrees:.0f}° total")
print(f"Primate FOV: {fov_primate.total_angle_degrees:.0f}° total")

# Check if specific directions are in FOV
test_bearings = np.array([0, np.pi / 2, np.pi, -np.pi / 2])
labels = ["ahead (0°)", "left (90°)", "behind (180°)", "right (-90°)"]

print("\nRat can see:")
for bearing, label in zip(test_bearings, labels, strict=True):
    visible = fov_rat.contains_angle(bearing)
    print(f"  {label}: {'yes' if visible else 'no'}")
