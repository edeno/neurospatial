# Quickstart

In a few lines, neurospatial turns spike times and a trajectory into a **place
field** — a smoothed map of where a neuron fires in space. This page walks
through that end-to-end analysis on simulated data, then explains the pieces.

## Install

```bash
pip install neurospatial
```

See the [installation guide](installation.md) for optional extras (napari
animation, NWB, JAX, xarray) and `uv`/conda instructions.

## Your first place field

The block below is complete and runnable top to bottom. It creates an
environment, simulates a foraging trajectory and one place cell's spikes with
neurospatial's built-in simulators, and estimates the cell's firing-rate map.

```python
import matplotlib.pyplot as plt
import numpy as np

from neurospatial import Environment
from neurospatial.simulation import (
    PlaceCellModel,
    generate_poisson_spikes,
    simulate_trajectory_ou,
)
from neurospatial.encoding import compute_spatial_rate

rng = np.random.default_rng(1)

# 1. Create an Environment: a 100 x 100 cm open-field arena binned at 2 cm.
arena_samples = rng.uniform(0, 100, size=(2000, 2))
env = Environment.from_samples(arena_samples, bin_size=2.0, units="cm")

# 2. Simulate 5 minutes of foraging, plus one place cell tuned to arena center.
positions, times = simulate_trajectory_ou(
    env, duration=300.0, speed_units="cm", seed=1
)
place_cell = PlaceCellModel(
    env, center=np.array([50.0, 50.0]), width=15.0, max_rate=30.0, seed=1
)
firing_rate_hz = place_cell.firing_rate(positions, times)
spike_times = generate_poisson_spikes(firing_rate_hz, times, seed=1)

# 3. Estimate the place field: a boundary-aware, smoothed firing-rate map.
result = compute_spatial_rate(env, spike_times, times, positions, bandwidth=8.0)
```

!!! note "Bringing your own data"
    Replace the simulation in step 2 with your own arrays: `spike_times`
    (shape `(n_spikes,)`, seconds), `times` (shape `(n_samples,)`, seconds),
    and `positions` (shape `(n_samples, 2)`, same units as `bin_size`). Build
    `env` from your recorded trajectory instead of `arena_samples`. Nothing
    else changes.

## Inspect and plot the result

`compute_spatial_rate` returns a `SpatialRateResult`. Ask it for headline
numbers, then plot the map:

```python
# Scalar headline metrics as a plain dict.
print(result.summary())
# -> a dict of scalars, e.g. ~1378 bins, peak firing rate ~9.3 Hz,
#    ~300 s total occupancy (exact values depend on the simulation).

# Where the cell fires most, and how spatially informative it is.
print("Peak firing location (cm):", result.peak_location())
print("Spatial information (bits/spike):", result.spatial_information())

# Plot the firing-rate map; returns a Matplotlib Axes you can further style.
ax = result.plot()
ax.set_title("Simulated place field")
plt.show()
```

The peak sits near `(50, 50)` cm — right where the place cell was tuned — and
the map is a smooth bump over the arena. That is the core loop: **environment +
spikes + trajectory → firing-rate map you can measure and plot.**

## What just happened

Three concepts carried that analysis. They are worth a minute now; the
[Core Concepts](core-concepts.md) page goes deeper.

**Environment.** `Environment.from_samples(...)` discretized the continuous
arena into a grid of **bins** (small square tiles), keeping only the bins your
samples actually covered. `env.n_bins` reports how many there are, and every
later step — occupancy, smoothing, plotting — is defined over those bins:

```python
print(f"{env.n_bins} active bins, {env.n_dims}D, units = {env.units!r}")
```

**Bins.** Each bin has an integer index and a center coordinate. A position is
mapped to the bin whose tile contains it, so `positions` (continuous cm) become
per-bin occupancy (seconds spent in each tile), and `spike_times` become
per-bin spike counts. Firing rate is spikes ÷ occupancy, smoothed.

**Connectivity graph.** The environment also carries a graph linking
neighboring bins. That is what makes the smoothing *boundary-aware*: probability
mass flows between adjacent, reachable bins instead of leaking across walls or
gaps — the difference between a physically meaningful map and a blurry one. The
same graph powers geodesic distances and shortest paths:

```python
center_bin = env.bin_at([[50.0, 50.0]])[0]
print("Neighbors of the center bin:", env.neighbors(center_bin))
```

## Next steps

- **[Core Concepts](core-concepts.md)** — bins, connectivity graphs, layout
  engines, and 1D linearized tracks, in depth.
- **[Place Field Analysis example](../examples/11_place_field_analysis.ipynb)** —
  the same workflow on a fuller dataset, with a whole population of cells.
- **[User Guide](../user-guide/index.md)** — decoding, trajectory and behavioral
  analysis, egocentric frames, animation, and more.
- **[API Reference](../api/index.md)** — every function and result class.
