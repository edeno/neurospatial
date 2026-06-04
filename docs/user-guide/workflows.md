# Complete Workflows

This page demonstrates end-to-end analysis workflows that integrate multiple neurospatial features.

## Workflow 1: Place Field Analysis

A complete workflow for analyzing spatial firing patterns of neurons during navigation.

### Overview

**Goal**: Compute spatial firing rate maps from position tracking and spike data

**Steps**: Simulate trajectory → Create environment → Generate spikes → Compute place field → Visualize

### Complete Example

```python
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.encoding import compute_spatial_rate
from neurospatial.simulation import (
    PlaceCellModel,
    generate_poisson_spikes,
    simulate_trajectory_ou,
)

# Step 1: Simulate an open-field session in a 100x100 cm arena.
# We seed the trajectory simulator with a polygon environment so the animal
# explores the full arena from the start (avoids a degenerate seed from
# sparse random points). Duration=300 s gives ~90% arena coverage at 15 cm/s
# while keeping CI runtime well under 60 s.
env_seed = Environment.from_polygon(box(0, 0, 100, 100), bin_size=5.0)
env_seed.units = "cm"  # Required by simulate_trajectory_ou
positions, times = simulate_trajectory_ou(
    env_seed,
    duration=300.0,   # 5-minute session — full arena coverage
    dt=1 / 30.0,      # 30 Hz tracking
    speed_mean=15.0,  # cm/s
    seed=0,
    speed_units="cm",
)

# Step 2: Create a finer environment from the recorded trajectory
# (2.5 cm bins → ~1 600 active bins for a 100×100 cm open field)
env = Environment.from_samples(
    positions,
    bin_size=2.5,             # 2.5 cm bins for a 100×100 cm arena
    bin_count_threshold=5,
    dilate=True,
    fill_holes=True,
    name="OpenFieldSession1",
)
env.units = "cm"

print(f"Created environment with {env.n_bins} active bins")
print(f"Spatial extent: {env.dimension_ranges}")
assert env.bin_at([50.0, 50.0]) != -1, "Place cell center must be inside env!"

# Step 3: Generate spike train for a simulated place cell
# (In real experiments, load your spike timestamps here.)
cell = PlaceCellModel(env, center=np.array([50.0, 50.0]), width=12.0, max_rate=20.0)
rates = cell.firing_rate(positions, times)
spike_times = generate_poisson_spikes(rates, times, seed=1)

print(f"Total spikes: {len(spike_times)}")

# Step 4: Compute the place field with the canonical one-liner
result = compute_spatial_rate(
    env,
    spike_times,
    times,
    positions,
    smoothing_method="diffusion_kde",  # boundary-aware graph-based KDE
    bandwidth=5.0,                     # smoothing bandwidth in cm
    # min_occupancy threshold is applied to the *smoothed* occupancy density,
    # not raw seconds. Low-coverage bins are excluded by bin_count_threshold
    # when creating the environment; leave min_occupancy at its default (0.0)
    # unless you have a specific density threshold in mind.
)
firing_rate = result.firing_rate

print(f"Peak firing rate: {np.nanmax(firing_rate):.2f} Hz")
print(f"Mean firing rate: {np.nanmean(firing_rate):.2f} Hz")

# Step 5: Visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Trajectory overlaid on environment layout
ax1 = axes[0]
env.plot(ax=ax1)
ax1.plot(positions[:, 0], positions[:, 1], "r-", alpha=0.3, linewidth=0.5)
ax1.set_title("Trajectory")

# Plot 2: Occupancy map (seconds per bin)
ax2 = axes[1]
env.plot_field(result.occupancy, ax=ax2, cmap="viridis")
ax2.set_title("Occupancy (s)")

# Plot 3: Place field (smoothed firing rate)
ax3 = axes[2]
env.plot_field(firing_rate, ax=ax3, cmap="hot")
ax3.set_title("Place Field (Hz)")

plt.tight_layout()
plt.show()

print(f"Spatial information: {result.spatial_information():.3f} bits/spike")
```

### Key Considerations

**Bin Size Selection**:
- Too large: Lose spatial resolution
- Too small: Insufficient occupancy, noisy firing rates
- Rule of thumb: 2-5 cm for rat open field (100x100 cm arena)

**Occupancy Threshold** (`min_occupancy`):
- For the default KDE methods (`diffusion_kde`/`gaussian_kde`), `min_occupancy`
  is a threshold on the *smoothed* occupancy density (the firing-rate
  denominator), not raw seconds — leave it at the default `0.0` unless you have
  a specific density threshold in mind. Low-coverage bins are already excluded
  at environment creation via `bin_count_threshold`.
- Only the legacy `smoothing_method="binned"` thresholds raw per-bin occupancy
  in seconds.
- Bins below the threshold are set to `NaN` in `result.firing_rate`.

**Smoothing**:
- `"diffusion_kde"` (default) is boundary-aware and works on any graph layout
- `"gaussian_kde"` gives comparable results in the interior of regular
  rectangular grids (`diffusion_kde` is boundary-aware, so they differ near
  boundaries)
- Increase `bandwidth` for noisier data or coarser bins

## Workflow 2: Bayesian Decoding (one call)

Reconstruct an animal's position from population spike trains in a single call.

### Overview

**Goal**: Decode position over time from a population of place cells

**Steps**: Simulate or load `(env, spike_times, times, positions)` → `decode_session(...)` → inspect `result.map_position` / plot

`decode_session` is the one-call golden path: it builds the encoding models
(place fields), bins the spikes onto a regular time grid, and runs the Bayesian
decoder for you. The whole encode → bin → decode pipeline fits in one line.

### Complete Example

```python
import numpy as np
import matplotlib.pyplot as plt

from neurospatial import Environment
from neurospatial.decoding import decode_session, decoding_error
from neurospatial.simulation import (
    PlaceCellModel,
    generate_population_spikes,
    simulate_trajectory_ou,
)

# Step 1: Simulate a population of place cells on a 100 cm linear track.
# (In a real analysis, load your env, spike_times, times, and positions here.)
env = Environment.from_samples(
    np.linspace(0.0, 100.0, 51).reshape(-1, 1), bin_size=2.0
)
env.units = "cm"
positions, times = simulate_trajectory_ou(
    env, duration=600.0, dt=0.02, speed_mean=15.0, seed=0, speed_units="cm"
)
cells = [
    PlaceCellModel(env, center=np.array([c]), width=10.0, max_rate=20.0, seed=i)
    for i, c in enumerate(np.linspace(5.0, 95.0, 25))
]
spike_times = generate_population_spikes(
    cells, positions, times, seed=0, show_progress=False
)

# Step 2: Decode position in a single call (encode -> bin -> decode).
result = decode_session(env, spike_times, times, positions, dt=0.1)

print(f"Decoded {result.posterior.shape[0]} time bins over {env.n_bins} bins")

# Step 3: Evaluate — the decoded MAP position should track the trajectory.
actual = np.interp(result.times, times, positions[:, 0]).reshape(-1, 1)
median_err = np.nanmedian(decoding_error(result.map_position, actual))
print(f"Median decoding error: {median_err:.1f} cm")

# Step 4: Plot decoded vs. actual position.
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(result.times, actual[:, 0], label="Actual", linewidth=1)
ax.plot(result.times, result.map_position[:, 0], label="Decoded (MAP)", linewidth=1)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position (cm)")
ax.set_title("Bayesian decoding with decode_session")
ax.legend()
plt.tight_layout()
plt.show()
```

`result` is a `DecodingResult` with `.posterior` (n_time, n_bins),
`.map_position` (n_time, n_dims), `.mean_position`, `.posterior_entropy`, and
`.times`. See [`decoding_error`](../api/neurospatial/decoding/index.md) and
related helpers for accuracy metrics.

### When to reach for the manual path

`decode_session` covers the common case. When you need custom control — passing
your own `encoding_models`, reusing fitted place fields across sessions, or
inspecting the binned spike counts — use the manual three-call path
(`compute_spatial_rates` → `bin_spikes_in_time` → `decode_position`). That
walk-through, plus trajectory analysis and shuffle-based significance testing
for replay detection, is covered in
[example 20](../examples/20_bayesian_decoding.ipynb).

## Workflow 3: Region-Based Analysis

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

## Workflow 4: Multi-Session Alignment

Comparing environments across recording sessions.

### Overview

**Goal**: Align spatial representations from different sessions to track stability

**Steps**: Create environments for each session → Align using transforms → Compare firing patterns

### Complete Example

```python
import numpy as np
import matplotlib.pyplot as plt

from neurospatial import Environment
from neurospatial.encoding import compute_spatial_rate
from neurospatial.ops import map_probabilities

# Session 1 (reference)
env1 = Environment.from_samples(
    session1_position,
    bin_size=2.5,
    name="Session1",
)
firing_rate1 = compute_spatial_rate(
    env1, session1_spikes, session1_times, session1_position,
    smoothing_method="diffusion_kde", bandwidth=5.0, min_occupancy=0.5,
).firing_rate

# Session 2 (may have slight camera shift or animal positioning differences)
env2 = Environment.from_samples(
    session2_position,
    bin_size=2.5,
    name="Session2",
)
firing_rate2 = compute_spatial_rate(
    env2, session2_spikes, session2_times, session2_position,
    smoothing_method="diffusion_kde", bandwidth=5.0, min_occupancy=0.5,
).firing_rate

# Align session 2 to session 1 coordinate frame
firing_rate2_aligned = map_probabilities(
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

## Workflow 5: Track Linearization

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
# Process multiple units efficiently
spike_trains_by_unit_id = load_all_neurons()
unit_ids = []
firing_rate_maps = []

for unit_id, spike_times in spike_trains_by_unit_id.items():
    unit_ids.append(unit_id)
    spike_positions = interpolate_position(position_data, spike_times)
    spike_bins = env.bin_at(spike_positions)
    spike_counts, _ = np.histogram(spike_bins, bins=np.arange(env.n_bins + 1))

    firing_rate = spike_counts / occupancy_time
    firing_rate[occupancy_time < min_occupancy] = np.nan

    firing_rate_maps.append(firing_rate)

firing_rate_maps = np.array(firing_rate_maps)  # Shape: (n_units, n_bins)
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

- [Environment API](../api/neurospatial/environment/index.md): Complete method documentation
- [Regions Guide](regions.md): Working with ROIs
- [Example Notebooks](../examples/index.md): Interactive tutorials
