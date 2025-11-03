# Complete Workflows

This page demonstrates end-to-end analysis workflows that integrate multiple neurospatial features.

## Workflow 1: Place Field Analysis

A complete workflow for analyzing spatial firing patterns of neurons during navigation.

### Overview

**Goal**: Compute spatial firing rate maps from position tracking and spike data

**Steps**: Load data → Create environment → Compute occupancy → Process spikes → Calculate firing rates → Visualize

### Complete Example

```python
import numpy as np
import matplotlib.pyplot as plt
from neurospatial import Environment

# Step 1: Load experimental data
# Position data: (n_timepoints, 2) array of x, y coordinates in cm
# Spike times: timestamps when neuron fired
position_data = load_position_data()  # Your data loading function
spike_times = load_spike_times()      # Your spike loading function
sampling_rate = 30.0  # Hz

# Step 2: Create environment with appropriate parameters
env = Environment.from_samples(
    data_samples=position_data,
    bin_size=2.5,  # 2.5 cm bins for 100x100 cm arena
    infer_active_bins=True,
    bin_count_threshold=5,  # Require at least 5 samples per bin
    dilate=True,  # Expand active region slightly
    fill_holes=True,  # Fill interior gaps
    name="OpenFieldSession1"
)

print(f"Created environment with {env.n_bins} active bins")
print(f"Spatial extent: {env.dimension_ranges}")

# Step 3: Compute occupancy (time spent in each bin)
position_bin_indices = env.bin_at(position_data)
occupancy_counts, _ = np.histogram(
    position_bin_indices,
    bins=np.arange(env.n_bins + 1)
)
occupancy_time = occupancy_counts / sampling_rate  # Convert to seconds

# Sanity check
total_time = len(position_data) / sampling_rate
print(f"Total session time: {total_time:.1f} seconds")
print(f"Time accounted for: {occupancy_time.sum():.1f} seconds")
assert np.isclose(occupancy_time.sum(), total_time, rtol=0.01)

# Step 4: Map spikes to bins
# Find position at each spike time (requires interpolation in real data)
spike_positions = interpolate_position(position_data, spike_times)
spike_bin_indices = env.bin_at(spike_positions)

# Count spikes per bin
spike_counts, _ = np.histogram(
    spike_bin_indices,
    bins=np.arange(env.n_bins + 1)
)

# Step 5: Calculate firing rate map
# Only compute firing rate for bins with sufficient occupancy
min_occupancy = 0.1  # seconds
firing_rate = np.full(env.n_bins, np.nan)
valid_bins = occupancy_time >= min_occupancy

firing_rate[valid_bins] = (
    spike_counts[valid_bins] / occupancy_time[valid_bins]
)

print(f"Peak firing rate: {np.nanmax(firing_rate):.2f} Hz")
print(f"Mean firing rate: {np.nanmean(firing_rate):.2f} Hz")
print(f"Bins with valid firing rate: {np.sum(valid_bins)}/{env.n_bins}")

# Step 6: Smooth firing rate map (optional)
from scipy.ndimage import gaussian_filter

# Reshape to 2D grid for smoothing
if hasattr(env.layout, 'grid_shape'):
    grid_shape = env.layout.grid_shape
    active_mask = env.layout.active_mask

    # Create full grid with NaN for inactive bins
    firing_rate_grid = np.full(grid_shape, np.nan)
    firing_rate_grid[active_mask] = firing_rate

    # Smooth (only affects active regions)
    smoothed_grid = gaussian_filter(
        np.nan_to_num(firing_rate_grid),
        sigma=1.0
    )
    firing_rate_smoothed = smoothed_grid[active_mask]
else:
    firing_rate_smoothed = firing_rate  # Can't smooth non-grid layouts

# Step 7: Visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Trajectory with environment
ax1 = axes[0]
env.plot(ax=ax1)
ax1.plot(position_data[:, 0], position_data[:, 1],
         'r-', alpha=0.3, linewidth=0.5)
ax1.set_title('Trajectory')

# Plot 2: Occupancy map
ax2 = axes[1]
scatter = ax2.scatter(
    env.bin_centers[:, 0],
    env.bin_centers[:, 1],
    c=occupancy_time,
    s=50,
    cmap='viridis'
)
plt.colorbar(scatter, ax=ax2, label='Time (s)')
ax2.set_title('Occupancy')
ax2.set_xlabel('X (cm)')
ax2.set_ylabel('Y (cm)')

# Plot 3: Firing rate map
ax3 = axes[2]
scatter = ax3.scatter(
    env.bin_centers[:, 0],
    env.bin_centers[:, 1],
    c=firing_rate_smoothed,
    s=50,
    cmap='hot',
    vmin=0
)
plt.colorbar(scatter, ax=ax3, label='Firing Rate (Hz)')
ax3.set_title('Place Field')
ax3.set_xlabel('X (cm)')
ax3.set_ylabel('Y (cm)')

plt.tight_layout()
plt.savefig('place_field_analysis.png', dpi=300)
plt.show()

# Step 8: Export results
results = {
    'environment': env,
    'occupancy_time': occupancy_time,
    'spike_counts': spike_counts,
    'firing_rate': firing_rate,
    'firing_rate_smoothed': firing_rate_smoothed,
    'bin_centers': env.bin_centers,
}

# Save for later analysis
np.savez('place_field_results.npz', **results)
```

### Key Considerations

**Bin Size Selection**:
- Too large: Lose spatial resolution
- Too small: Insufficient occupancy, noisy firing rates
- Rule of thumb: 2-5 cm for rat open field (100x100 cm arena)

**Occupancy Threshold**:
- Exclude bins with low occupancy to avoid division by zero
- Typical: 0.1-1.0 seconds minimum
- Trade-off: Coverage vs. reliability

**Smoothing**:
- Reduces noise but blurs fine spatial structure
- Gaussian filter with sigma=1-2 bins is typical
- Consider not smoothing if bin size already large

## Workflow 2: Region-Based Analysis

Analyzing behavior across experimentally-defined spatial zones.

### Overview

**Goal**: Compare neural activity and behavior across different regions of the environment

**Steps**: Define regions → Compute metrics per region → Statistical comparison

### Complete Example

```python
from neurospatial import Environment
from shapely.geometry import Point
import numpy as np

# Create environment from position data
env = Environment.from_samples(position_data, bin_size=3.0)

# Define experimental regions
# Center zone (15 cm radius circle)
center_point = Point(50.0, 50.0)  # Arena center
env.regions.add("Center", polygon=center_point.buffer(15.0))

# Corner zones (10x10 cm squares)
corners = {
    "TopLeft": [(0, 90), (10, 90), (10, 100), (0, 100)],
    "TopRight": [(90, 90), (100, 90), (100, 100), (90, 100)],
    "BottomLeft": [(0, 0), (10, 0), (10, 10), (0, 10)],
    "BottomRight": [(90, 0), (100, 0), (100, 10), (90, 10)],
}

for name, coords in corners.items():
    from shapely.geometry import Polygon
    env.regions.add(name, polygon=Polygon(coords))

# Find which bins belong to each region
region_bins = {}
for region_name in env.regions.list_names():
    region_polygon = env.regions[region_name].polygon
    bins_in_region = []

    for bin_idx in range(env.n_bins):
        bin_point = Point(env.bin_centers[bin_idx])
        if region_polygon.contains(bin_point):
            bins_in_region.append(bin_idx)

    region_bins[region_name] = np.array(bins_in_region)
    print(f"{region_name}: {len(bins_in_region)} bins")

# Compute occupancy per region
position_bins = env.bin_at(position_data)
sampling_rate = 30.0  # Hz

region_occupancy = {}
for region_name, bins in region_bins.items():
    time_in_region = np.sum(np.isin(position_bins, bins)) / sampling_rate
    region_occupancy[region_name] = time_in_region
    print(f"Time in {region_name}: {time_in_region:.2f} seconds")

# Compute firing rate per region
spike_positions = interpolate_position(position_data, spike_times)
spike_bins = env.bin_at(spike_positions)

region_firing_rates = {}
for region_name, bins in region_bins.items():
    spikes_in_region = np.sum(np.isin(spike_bins, bins))
    time_in_region = region_occupancy[region_name]

    if time_in_region > 0.5:  # Require 0.5s minimum
        firing_rate = spikes_in_region / time_in_region
        region_firing_rates[region_name] = firing_rate
    else:
        region_firing_rates[region_name] = np.nan

    print(f"{region_name} firing rate: {firing_rate:.2f} Hz")

# Statistical comparison
# Example: Is firing rate higher in center vs. corners?
center_rate = region_firing_rates["Center"]
corner_rates = [region_firing_rates[name] for name in corners.keys()]
corner_rates = [r for r in corner_rates if not np.isnan(r)]

print(f"\nCenter: {center_rate:.2f} Hz")
print(f"Corners: {np.mean(corner_rates):.2f} ± {np.std(corner_rates):.2f} Hz")

# Visualize regions
fig, ax = plt.subplots(figsize=(8, 8))
env.plot(ax=ax)

# Color-code regions
colors = plt.cm.Set3(np.linspace(0, 1, len(env.regions)))
for idx, region_name in enumerate(env.regions.list_names()):
    region = env.regions[region_name]
    if region.polygon:
        x, y = region.polygon.exterior.xy
        ax.fill(x, y, alpha=0.3, color=colors[idx], label=region_name)

ax.legend()
ax.set_title('Experimental Regions')
plt.show()
```

## Workflow 3: Multi-Session Alignment

Comparing environments across recording sessions.

### Overview

**Goal**: Align spatial representations from different sessions to track stability

**Steps**: Create environments for each session → Align using transforms → Compare firing patterns

### Complete Example

```python
from neurospatial import Environment
from neurospatial.alignment import map_probabilities_to_nearest_target_bin
import numpy as np

# Session 1 (reference)
env1 = Environment.from_samples(
    session1_position,
    bin_size=2.5,
    name="Session1"
)
firing_rate1 = compute_firing_rate(env1, session1_position, session1_spikes)

# Session 2 (may have slight camera shift or animal positioning differences)
env2 = Environment.from_samples(
    session2_position,
    bin_size=2.5,
    name="Session2"
)
firing_rate2 = compute_firing_rate(env2, session2_position, session2_spikes)

# Align session 2 to session 1 coordinate frame
firing_rate2_aligned = map_probabilities_to_nearest_target_bin(
    source_env=env2,
    target_env=env1,
    source_probabilities=firing_rate2
)

# Compute spatial correlation
valid_bins = ~np.isnan(firing_rate1) & ~np.isnan(firing_rate2_aligned)
correlation = np.corrcoef(
    firing_rate1[valid_bins],
    firing_rate2_aligned[valid_bins]
)[0, 1]

print(f"Spatial correlation: {correlation:.3f}")

# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Session 1
axes[0].scatter(env1.bin_centers[:, 0], env1.bin_centers[:, 1],
                c=firing_rate1, s=50, cmap='hot')
axes[0].set_title('Session 1')

# Session 2 (aligned)
axes[1].scatter(env1.bin_centers[:, 0], env1.bin_centers[:, 1],
                c=firing_rate2_aligned, s=50, cmap='hot')
axes[1].set_title('Session 2 (aligned)')

# Difference
difference = firing_rate2_aligned - firing_rate1
axes[2].scatter(env1.bin_centers[:, 0], env1.bin_centers[:, 1],
                c=difference, s=50, cmap='RdBu_r',
                vmin=-np.nanmax(np.abs(difference)),
                vmax=np.nanmax(np.abs(difference)))
axes[2].set_title(f'Difference (r={correlation:.3f})')

plt.tight_layout()
plt.show()
```

## Workflow 4: Track Linearization

Analyzing maze experiments with branching structures.

### Overview

**Goal**: Convert 2D maze positions to 1D linearized coordinates for sequential analysis

**Steps**: Define track graph → Create 1D environment → Map positions → Analyze

See the complete example in [examples/05_track_linearization.ipynb](../examples/05_track_linearization.ipynb).

## Common Patterns

### Pattern: Handling Edge Cases

```python
# Always check for valid bins
bin_indices = env.bin_at(positions)
valid = bin_indices != -1  # -1 indicates point outside environment

# Use only valid data
valid_positions = positions[valid]
valid_bins = bin_indices[valid]

# Or handle invalid gracefully
firing_rate = np.full(env.n_bins, np.nan)
valid_occupancy = occupancy_time > min_threshold
firing_rate[valid_occupancy] = spike_counts[valid_occupancy] / occupancy_time[valid_occupancy]
```

### Pattern: Batch Processing

```python
# Process multiple neurons efficiently
neurons = load_all_neurons()
firing_rate_maps = []

for neuron_id, spike_times in neurons.items():
    spike_positions = interpolate_position(position_data, spike_times)
    spike_bins = env.bin_at(spike_positions)
    spike_counts, _ = np.histogram(spike_bins, bins=np.arange(env.n_bins + 1))

    firing_rate = spike_counts / occupancy_time
    firing_rate[occupancy_time < min_occupancy] = np.nan

    firing_rate_maps.append(firing_rate)

firing_rate_maps = np.array(firing_rate_maps)  # Shape: (n_neurons, n_bins)
```

### Pattern: Progressive Refinement

```python
# Start with coarse binning for quick overview
env_coarse = Environment.from_samples(positions, bin_size=10.0)
# ... analyze ...

# Refine in regions of interest
env_fine = Environment.from_samples(
    positions,
    bin_size=2.0,
    infer_active_bins=True,
    dilate=True
)
# ... detailed analysis ...
```

## See Also

- [Environment API](../api/neurospatial/environment.md): Complete method documentation
- [Regions Guide](regions.md): Working with ROIs
- [Example Notebooks](../examples/index.md): Interactive tutorials
