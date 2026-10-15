# Spike Train to Spatial Field Conversion

This guide covers the primitives for converting spike trains into occupancy-normalized spatial fields (firing rate maps), a foundational operation in place field analysis and spatial neuroscience.

## Overview

The neurospatial encoding API provides one canonical entry point for
spike-to-field conversion:

- **`compute_spatial_rate()`** - Converts spike trains to occupancy-normalized spatial firing-rate maps and returns a `SpatialRateResult`

The function follows the neurospatial API convention: **environment comes first** in the parameter order.

## Why Occupancy Normalization Matters

Firing rate fields must be normalized by the time spent in each spatial bin (occupancy) to produce meaningful estimates of spatial tuning:

$$
\text{firing rate}_i = \frac{\text{spike count}_i}{\text{occupancy}_i}
$$

Without occupancy normalization, more frequently visited bins can accumulate
more spikes simply because they were observed longer. The equation above is
the unsmoothed binned estimator. KDE methods smooth spike counts and occupancy
before dividing; smoothing the already divided rate gives a different estimator.

### What is Occupancy?

Occupancy is the **time-weighted** measure of how long an animal spent in each spatial bin. Neurospatial computes occupancy by:

1. Binning the position trajectory into spatial bins
2. Computing the duration of each time interval (difference between consecutive timestamps)
3. Allocating these durations to the bins visited

By default, `env.occupancy()` returns occupancy in **seconds** (time-weighted). You can optionally get interval counts (unweighted) using `return_seconds=False`, but for firing rate computation, **always use the default** `return_seconds=True`.

## Basic Usage

### Converting Spike Trains to Firing Rate Fields

The `compute_spatial_rate()` function converts spike train data into spatial firing rate maps:

```python
import numpy as np
from neurospatial import Environment
from neurospatial.encoding import compute_spatial_rate

# Create environment from trajectory data
env = Environment.from_samples(positions, bin_size=2.5)

# Spike times (seconds)
spike_times = np.array([1.2, 3.5, 5.1, 7.8, ...])

# Trajectory timestamps and positions
times = np.array([0.0, 0.033, 0.067, ...])  # 30 Hz sampling
positions = np.array([[x1, y1], [x2, y2], ...])  # Shape (n_timepoints, 2)

# Compute firing rate field
result = compute_spatial_rate(
    env,
    spike_times,
    times,
    positions,
    method="binned",
    bandwidth=0.0,  # No smoothing: spike count / occupancy
    min_occupancy=0.5,  # Exclude bins with < 0.5 seconds occupancy
)
firing_rate = result.firing_rate

# Result: firing_rate.shape = (env.n_bins,)
# Units: spikes/second
# Bins with insufficient occupancy are set to NaN
```

**Key points:**

- Parameter order: **environment comes first**, followed by spike data and trajectory data
- Returns firing rate in **spikes/second**
- Bins with occupancy less than `min_occupancy` are set to `NaN`
- Default `min_occupancy=0.0` disables occupancy masking; bins without
  sufficient estimator support can still be `NaN`

### One-Liner with Smoothing: `compute_spatial_rate()`

For typical place field analysis workflows, use `compute_spatial_rate()` with a smoothing method:

```python
from neurospatial.encoding import compute_spatial_rate

# Compute place field with Gaussian smoothing
result = compute_spatial_rate(
    env,
    spike_times,
    times,
    positions,
    min_occupancy=0.5,
    method="gaussian_kde",
    bandwidth=5.0,  # Gaussian kernel bandwidth (cm)
)
place_field = result.firing_rate
```

Use `method="binned", bandwidth=0.0` for an unsmoothed count/occupancy map.
With positive bandwidth, `binned` smooths the rate after division; KDE methods
smooth the numerator and denominator separately.

## Parameter Order: Environment First

Spatial encoding functions take the **environment first**. Other domains have
their own signatures; for example, directional encoding does not need an environment.

```python
# Correct parameter order
compute_spatial_rate(env, spike_times, times, positions, ...)

# This matches the existing neurospatial API:
env.occupancy(times, positions)
env.smooth(field, bandwidth)
map_points_to_bins(env, points)
distance_field(env.connectivity, sources)
```

This design decision ensures:

- **Consistency** across the entire neurospatial API
- **Readability** - the environment context is always explicit and comes first
- **Composability** - functions can be easily chained and combined

## Min Occupancy Threshold: Best Practices

For `diffusion_kde`, `gaussian_kde` and `binned`, `min_occupancy` thresholds
**raw occupancy in seconds**, after recording windows, tracking gaps and any
speed filter have been applied. It does not threshold smoothed density.
`method="glm"` rejects this option and uses its own penalty/rank controls.

```python
# Disable occupancy masking (default); unsupported bins can still be NaN
firing_rate = compute_spatial_rate(
    env, spike_times, times, positions, min_occupancy=0.0
).firing_rate

# Illustrative threshold: exclude bins with < 0.5 seconds occupancy
firing_rate = compute_spatial_rate(
    env, spike_times, times, positions, min_occupancy=0.5
).firing_rate

# Stricter occupancy mask (1 second); not a confidence threshold
firing_rate = compute_spatial_rate(
    env, spike_times, times, positions, min_occupancy=1.0
).firing_rate
```

Choose the threshold together with bin size, recording duration and the
scientific question. The values above illustrate the API; no fixed occupancy
threshold establishes reliable tuning or a confidence level. Inspect the
occupancy map and sensitivity to parameter choices.

Sub-threshold bins are `NaN` unless `fill_value` is supplied. A finite KDE rate
in an unvisited bin reflects smoothing from observed bins; it is not evidence
that the animal visited that bin. Keep `result.occupancy` when using
`fill_value=0.0`, since zero-filled missing support and observed silence can
otherwise look identical. Check the NaN policy of each downstream statistic.

### Checking the Occupancy Mask

This small example has 0.6 seconds in the first bin, 0.4 in the second and
none in the third. All three ratio estimators mask the same two bins at a
0.5-second threshold, regardless of their different smoothing rules.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment, compute_spatial_rate

env = Environment.from_grid_mask(
    np.ones(3, dtype=bool), grid_edges=(np.arange(4.0),)
)
times = np.arange(11) / 10
positions = np.array([0.5] * 6 + [1.5] * 5).reshape(-1, 1)
spike_times = np.array([0.05, 0.15, 0.25, 0.65, 0.75])
for method in ("binned", "gaussian_kde", "diffusion_kde"):
    result = compute_spatial_rate(
        env, spike_times, times, positions,
        method=method, bandwidth=0.8, min_occupancy=0.5,
    )
    np.testing.assert_allclose(result.occupancy, [0.6, 0.4, 0.0])
    np.testing.assert_array_equal(
        np.isnan(result.firing_rate), result.occupancy < 0.5
    )
    assert np.isfinite(result.firing_rate[0])
```

## Edge Cases and Validation

### Empty Spike Trains

Empty spike arrays produce zero rates where the estimator has occupancy
support. Masked or unsupported bins can remain `NaN`:

```python
spike_times = np.array([])  # No spikes

firing_rate = compute_spatial_rate(env, spike_times, times, positions).firing_rate
# Result: zeros where supported; NaN in masked/unsupported bins
```

### Out-of-Bounds Spikes

Spikes occurring outside the trajectory time range are filtered with a warning:

```python
spike_times = np.array([1.0, 100.0, 200.0])  # Some spikes beyond times.max()
times = np.array([0.0, 0.033, ..., 50.0])  # Max time = 50.0

firing_rate = compute_spatial_rate(env, spike_times, times, positions).firing_rate
# UserWarning: "X spikes fall outside trajectory time range [0.0, 50.0] and were excluded"
```

An interval between two tracking samples counts only when both samples lie inside the environment. An interval that ends in a tracking dropout (NaN) or outside the environment is left out of both the spike counts and the occupancy, rather than guessing where the animal was; fill short dropouts in your tracking preprocessing if you want that time back. Spikes whose interpolated position still falls outside the environment bins (for example, across a hole between two tracked samples) are filtered with a warning.

### 1D Trajectories

The function handles 1D trajectories (linear tracks, circular tracks) correctly:

```python
# 1D positions - either shape (n_timepoints, 1) or (n_timepoints,)
positions_1d = np.array([0.0, 2.5, 5.0, 7.5, ...])  # Shape (n_timepoints,)

firing_rate = compute_spatial_rate(env, spike_times, times, positions_1d).firing_rate
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
compute_spatial_rate(env, spike_times, times, positions, min_occupancy=-1.0)
# Raises ValueError: "min_occupancy must be non-negative (got -1.0)"
```

## Smoothing and Missing Support

`diffusion_kde` and `gaussian_kde` smooth counts and occupancy separately,
divide where the smoothed denominator has support, then apply the raw-seconds
occupancy mask. `diffusion_kde` respects the environment's connectivity;
`gaussian_kde` uses Euclidean proximity and can smooth across nearby walls.

`binned` divides counts by occupancy first. With positive bandwidth it uses
a masked average that excludes undefined and sub-threshold input rates, then
reapplies the output occupancy mask. This is not zero-filling missing bins
before averaging. Set `bandwidth=0.0` to inspect the unsmoothed estimate.

Smoothing a returned rate map again with `env.smooth()` changes the estimator
and is not equivalent to KDE. Likewise, replacing `NaN` with zero changes the
meaning of missing support; retain the original occupancy when doing so.

## Visualizing Results

Typical visualization workflow:

```python
import matplotlib.pyplot as plt

# Compute occupancy and firing rate
occupancy = env.occupancy(times, positions, return_seconds=True)
firing_rate = compute_spatial_rate(
    env,
    spike_times,
    times,
    positions,
    method="binned",
    bandwidth=0.0,
    min_occupancy=0.5,
).firing_rate
place_field = compute_spatial_rate(
    env,
    spike_times,
    times,
    positions,
    min_occupancy=0.5,
    method="gaussian_kde",
    bandwidth=5.0,
).firing_rate

# Visualize using environment plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

env.plot_field(occupancy, ax=axes[0], cmap='viridis')
axes[0].set_title('Occupancy (seconds)')

env.plot_field(firing_rate, ax=axes[1], cmap='hot')
axes[1].set_title('Firing Rate (raw)')

env.plot_field(place_field, ax=axes[2], cmap='hot')
axes[2].set_title('Place Field (smoothed)')

plt.tight_layout()
plt.show()
```

**Tips:**

- Use `cmap='viridis'` for occupancy (sequential)
- Use `cmap='hot'` or `cmap='inferno'` for firing rates (perceptually uniform, peaks visible)
- The `env.plot_field()` method automatically handles NaN values (shown as white/transparent)

## Complete Example

Here's a complete workflow from trajectory data to smoothed place field:

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment
from neurospatial.encoding import compute_spatial_rate

# 1. Load or generate trajectory data
times = np.linspace(0, 60, 1800)  # 1 minute at 30 Hz
positions = np.column_stack([
    10 * np.sin(0.5 * times),  # X coordinate
    10 * np.cos(0.5 * times)   # Y coordinate
])  # Circular trajectory

# 2. Generate spike train from a position-dependent rate (Hz)
random = np.random.default_rng(42)
preferred_location = np.array([5.0, 5.0])
distances = np.linalg.norm(positions - preferred_location, axis=1)
rate_hz = 15.0 * np.exp(-distances**2 / (2 * 3**2))
dt = np.diff(times)
# Bernoulli approximation on short intervals: at most one spike per interval.
spike_times = times[:-1][random.random(len(dt)) < 1 - np.exp(-rate_hz[:-1] * dt)]

# 3. Create environment
env = Environment.from_samples(positions, bin_size=2.5)
env = env.with_metadata(units="cm")

# 4. Compute place field
result = compute_spatial_rate(
    env,
    spike_times,
    times,
    positions,
    min_occupancy=0.5,
    method="gaussian_kde",
    bandwidth=5.0,
)
place_field = result.firing_rate

assert spike_times.size > 0
assert np.nanmax(place_field) > 0
np.testing.assert_array_equal(
    np.isnan(place_field), result.occupancy < 0.5
)

# 5. Visualize
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6))
env.plot_field(place_field, ax=ax, cmap='hot')
ax.set_title('Place Field')
plt.show()

print(f"Environment: {env.n_bins} bins")
print(f"Peak firing rate: {np.nanmax(place_field):.2f} Hz")
print(f"Spatial information: {np.sum(place_field > 0.5 * np.nanmax(place_field))} bins > 50% peak")
```

## API Reference

For complete parameter descriptions and examples, see:

- [`compute_spatial_rate()`](../api/neurospatial/encoding/spatial.md#neurospatial.encoding.spatial.compute_spatial_rate) - Place field computation
- [`bin_spike_train()`](../api/neurospatial/encoding/_binning.md#neurospatial.encoding._binning.bin_spike_train) - Spike binning helper
- [`Environment.occupancy()`](../api/neurospatial/environment/cartesian.md#neurospatial.environment.cartesian.CartesianCoordinates.occupancy) - Occupancy computation
- [`Environment.smooth()`](../api/neurospatial/environment/fields.md#neurospatial.environment.fields.EnvironmentFields.smooth) - Spatial field smoothing

## Related Topics

- [Spatial Analysis](spatial-analysis.md) - Broader spatial analysis workflows
- [RL Primitives](rl-primitives.md) - Reward field generation for reinforcement learning
- [Workflows](workflows.md) - Complete analysis pipelines
