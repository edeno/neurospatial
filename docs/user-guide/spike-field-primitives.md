# Spike Train to Spatial Field Conversion

This guide covers the primitives for converting spike trains into occupancy-normalized spatial fields (firing rate maps), a foundational operation in place field analysis and spatial neuroscience.

## Overview

The neurospatial library provides two functions for spike-to-field conversion:

- **`spikes_to_field()`** - Core function for converting spike trains to firing rate fields
- **`compute_place_field()`** - Convenience wrapper that combines spike conversion with optional smoothing

Both functions follow the neurospatial API convention: **environment comes first** in the parameter order, matching the existing pattern used throughout the library.

## Why Occupancy Normalization Matters

Firing rate fields must be normalized by the time spent in each spatial bin (occupancy) to produce meaningful estimates of spatial tuning:

$$
\text{firing rate}_i = \frac{\text{spike count}_i}{\text{occupancy}_i}
$$

Without occupancy normalization, bins visited more frequently would appear to have higher firing rates simply due to increased sampling, not actual spatial preference. This normalization is **standard practice** in place field analysis and is used universally in neuroscience literature (O'Keefe & Dostrovsky, 1971; Muller et al., 1987).

### What is Occupancy?

Occupancy is the **time-weighted** measure of how long an animal spent in each spatial bin. Neurospatial computes occupancy by:

1. Binning the position trajectory into spatial bins
2. Computing the duration of each time interval (difference between consecutive timestamps)
3. Allocating these durations to the bins visited

By default, `env.occupancy()` returns occupancy in **seconds** (time-weighted). You can optionally get interval counts (unweighted) using `return_seconds=False`, but for firing rate computation, **always use the default** `return_seconds=True`.

## Basic Usage

### Converting Spike Trains to Firing Rate Fields

The `spikes_to_field()` function converts spike train data into spatial firing rate maps:

```python
import numpy as np
from neurospatial import Environment
from neurospatial import spikes_to_field

# Create environment from trajectory data
env = Environment.from_samples(trajectory_positions, bin_size=2.5)

# Spike times (seconds)
spike_times = np.array([1.2, 3.5, 5.1, 7.8, ...])

# Trajectory timestamps and positions
times = np.array([0.0, 0.033, 0.067, ...])  # 30 Hz sampling
positions = np.array([[x1, y1], [x2, y2], ...])  # Shape (n_timepoints, 2)

# Compute firing rate field
firing_rate = spikes_to_field(
    env,
    spike_times,
    times,
    positions,
    min_occupancy_seconds=0.5  # Exclude bins with < 0.5 seconds occupancy
)

# Result: firing_rate.shape = (env.n_bins,)
# Units: spikes/second
# Bins with insufficient occupancy are set to NaN
```

**Key points:**

- Parameter order: **environment comes first**, followed by spike data and trajectory data
- Returns firing rate in **spikes/second**
- Bins with occupancy less than `min_occupancy_seconds` are set to `NaN`
- Default `min_occupancy_seconds=0.0` includes all bins

### One-Liner with Smoothing: `compute_place_field()`

For typical place field analysis workflows, use the convenience function that combines spike conversion and smoothing:

```python
from neurospatial import compute_place_field

# Compute place field with Gaussian smoothing
place_field = compute_place_field(
    env,
    spike_times,
    times,
    positions,
    min_occupancy_seconds=0.5,
    smoothing_bandwidth=5.0  # Gaussian kernel bandwidth (cm)
)
```

This is equivalent to:

```python
# Two-step workflow (manual)
firing_rate = spikes_to_field(env, spike_times, times, positions, min_occupancy_seconds=0.5)
place_field = env.smooth(firing_rate, bandwidth=5.0)
```

If `smoothing_bandwidth=None`, `compute_place_field()` behaves identically to `spikes_to_field()`.

## Parameter Order: Environment First

**Important:** All neurospatial functions follow a consistent API pattern where the **environment comes first**:

```python
# Correct parameter order
spikes_to_field(env, spike_times, times, positions, ...)
compute_place_field(env, spike_times, times, positions, ...)

# This matches the existing neurospatial API:
env.occupancy(times, positions)
env.smooth(field, bandwidth)
map_points_to_bins(points, env)
distance_field(env.connectivity, sources)
```

This design decision ensures:

- **Consistency** across the entire neurospatial API
- **Readability** - the environment context is always explicit and comes first
- **Composability** - functions can be easily chained and combined

## Min Occupancy Threshold: Best Practices

The `min_occupancy_seconds` parameter controls reliability filtering:

```python
# Include all bins (default)
firing_rate = spikes_to_field(env, spike_times, times, positions, min_occupancy_seconds=0.0)

# Exclude bins with < 0.5 seconds occupancy (typical for place fields)
firing_rate = spikes_to_field(env, spike_times, times, positions, min_occupancy_seconds=0.5)

# More conservative (1 second threshold)
firing_rate = spikes_to_field(env, spike_times, times, positions, min_occupancy_seconds=1.0)
```

**Recommendations:**

- **Default (0.0)**: Use when you need complete spatial coverage or will apply additional filtering
- **0.5 seconds**: Standard for place field analysis - excludes bins with too few samples for reliable rate estimates
- **1.0+ seconds**: Conservative threshold for high-confidence place field detection

Bins with occupancy below the threshold are set to **NaN** (not zero), which:

- Allows you to distinguish between "silent" bins (visited but no spikes) and "unsampled" bins
- Prevents these bins from contaminating spatial statistics
- Works correctly with `env.smooth()` which handles NaN values appropriately

### Why Not Use Default Filtering?

The default `min_occupancy_seconds=0.0` (no filtering) is intentional:

1. **Flexibility**: Users can inspect all bins and make informed filtering decisions
2. **Transparency**: All data is preserved, no hidden thresholding
3. **Composability**: You can apply custom filters after computing firing rates

For typical place field analysis, **explicitly set `min_occupancy_seconds=0.5`** to match standard neuroscience practices.

## Edge Cases and Validation

### Empty Spike Trains

Empty spike arrays (no spikes) produce fields of zeros:

```python
spike_times = np.array([])  # No spikes

firing_rate = spikes_to_field(env, spike_times, times, positions)
# Result: all zeros (or NaN where occupancy < min_occupancy_seconds)
```

### Out-of-Bounds Spikes

Spikes occurring outside the trajectory time range are filtered with a warning:

```python
spike_times = np.array([1.0, 100.0, 200.0])  # Some spikes beyond times.max()
times = np.array([0.0, 0.033, ..., 50.0])  # Max time = 50.0

firing_rate = spikes_to_field(env, spike_times, times, positions)
# UserWarning: "X spikes fall outside trajectory time range [0.0, 50.0] and were excluded"
```

Similarly, spikes whose interpolated positions fall outside the environment bounds are filtered with a warning.

### 1D Trajectories

The function handles 1D trajectories (linear tracks, circular tracks) correctly:

```python
# 1D positions - either shape (n_timepoints, 1) or (n_timepoints,)
positions_1d = np.array([0.0, 2.5, 5.0, 7.5, ...])  # Shape (n_timepoints,)

firing_rate = spikes_to_field(env, spike_times, times, positions_1d)
# Automatically handles 1D interpolation
```

### Input Validation

The function validates inputs and raises informative errors:

```python
# Mismatched lengths
times = np.array([0.0, 0.033, 0.067])
positions = np.array([[0, 0], [1, 1]])  # Different length!
# Raises ValueError: "times and positions must have the same length"

# Negative min_occupancy
spikes_to_field(env, spike_times, times, positions, min_occupancy_seconds=-1.0)
# Raises ValueError: "min_occupancy_seconds must be non-negative (got -1.0)"
```

## NaN Handling in Smoothing

When using `compute_place_field()` with smoothing, NaN values (from low-occupancy bins) are handled automatically:

1. NaN bins are temporarily filled with 0
2. Gaussian smoothing is applied
3. NaN values are restored to their original locations

**Important caveat:** This approach can artificially **reduce firing rates near unvisited regions** due to the zero-filling. For scientific applications requiring high precision near boundaries, use the two-step workflow:

```python
# Manual workflow with custom NaN handling
firing_rate = spikes_to_field(env, spike_times, times, positions, min_occupancy_seconds=0.5)

# Option 1: Smooth only valid bins (NaN stays NaN)
smoothed = env.smooth(firing_rate, bandwidth=5.0)

# Option 2: Inpaint NaN before smoothing (advanced)
from scipy.ndimage import distance_transform_edt
# ... custom NaN handling ...
```

For most use cases, the default NaN handling in `compute_place_field()` is appropriate and matches common neuroscience practice.

## Visualizing Results

Typical visualization workflow:

```python
import matplotlib.pyplot as plt

# Compute occupancy and firing rate
occupancy = env.occupancy(times, positions, return_seconds=True)
firing_rate = spikes_to_field(env, spike_times, times, positions, min_occupancy_seconds=0.5)
place_field = compute_place_field(env, spike_times, times, positions,
                                   min_occupancy_seconds=0.5, smoothing_bandwidth=5.0)

# Visualize using environment plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

env.plot(occupancy, ax=axes[0], cmap='viridis')
axes[0].set_title('Occupancy (seconds)')

env.plot(firing_rate, ax=axes[1], cmap='hot')
axes[1].set_title('Firing Rate (raw)')

env.plot(place_field, ax=axes[2], cmap='hot')
axes[2].set_title('Place Field (smoothed)')

plt.tight_layout()
plt.show()
```

**Tips:**

- Use `cmap='viridis'` for occupancy (sequential)
- Use `cmap='hot'` or `cmap='inferno'` for firing rates (perceptually uniform, peaks visible)
- The `env.plot()` method automatically handles NaN values (shown as white/transparent)

## Complete Example

Here's a complete workflow from trajectory data to smoothed place field:

```python
import numpy as np
from neurospatial import Environment, compute_place_field

# 1. Load or generate trajectory data
times = np.linspace(0, 60, 1800)  # 1 minute at 30 Hz
positions = np.column_stack([
    10 * np.sin(0.5 * times),  # X coordinate
    10 * np.cos(0.5 * times)   # Y coordinate
])  # Circular trajectory

# 2. Generate spike train (simulated place cell)
preferred_location = np.array([5.0, 5.0])
distances = np.linalg.norm(positions - preferred_location, axis=1)
firing_prob = 0.5 * np.exp(-distances**2 / (2 * 3**2))  # Gaussian tuning
spike_times = times[np.random.rand(len(times)) < firing_prob * 0.033]

# 3. Create environment
env = Environment.from_samples(positions, bin_size=2.5)
env.units = "cm"

# 4. Compute place field
place_field = compute_place_field(
    env,
    spike_times,
    times,
    positions,
    min_occupancy_seconds=0.5,
    smoothing_bandwidth=5.0
)

# 5. Visualize
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6))
env.plot(place_field, ax=ax, cmap='hot')
ax.set_title('Place Field')
plt.show()

print(f"Environment: {env.n_bins} bins")
print(f"Peak firing rate: {np.nanmax(place_field):.2f} Hz")
print(f"Spatial information: {np.sum(place_field > 0.5 * np.nanmax(place_field))} bins > 50% peak")
```

## API Reference

For complete parameter descriptions and examples, see:

- [`spikes_to_field()`](../api/neurospatial/spike_field.md#spikes_to_field) - Core spike conversion function
- [`compute_place_field()`](../api/neurospatial/spike_field.md#compute_place_field) - Convenience function with smoothing
- [`Environment.occupancy()`](../api/neurospatial/environment/index.md#occupancy) - Occupancy computation
- [`Environment.smooth()`](../api/neurospatial/environment/index.md#smooth) - Spatial field smoothing

## Related Topics

- [Spatial Analysis](spatial-analysis.md) - Broader spatial analysis workflows
- [RL Primitives](rl-primitives.md) - Reward field generation for reinforcement learning
- [Workflows](workflows.md) - Complete analysis pipelines
