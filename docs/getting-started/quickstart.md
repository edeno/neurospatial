# Quickstart

In a few lines, neurospatial turns spike times and a trajectory into a
**firing-rate map**: a neuron's firing rate in each small area of the arena.
A **place field** is a thresholded region of that map, a contiguous patch of
high firing rate, not the whole map. This page walks through the end-to-end
analysis on simulated data, then explains the pieces.

## Install

```bash
pip install neurospatial
```

See the [installation guide](installation.md) for optional extras (napari
animation, NWB, JAX, xarray) and `uv`/conda instructions.

## Your first rate map

The block below is complete and runnable top to bottom. It creates an
environment, simulates a foraging trajectory and one place cell's spikes with
neurospatial's built-in simulators, and estimates the cell's firing-rate map.

<!-- docs-test: run -->
```python
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.simulation import (
    PlaceCellModel,
    generate_poisson_spikes,
    simulate_trajectory_ou,
)
from neurospatial import compute_spatial_rate

# 1. Create an Environment: a 100 x 100 cm open-field arena binned at 2 cm.
env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")

# 2. Simulate 30 minutes of foraging, plus one place cell tuned to arena center.
positions, times = simulate_trajectory_ou(
    env, duration=1800.0, speed_units="cm", rng=1
)
place_cell = PlaceCellModel(
    env, center=np.array([50.0, 50.0]), width=15.0, max_rate=30.0
)
firing_rate_hz = place_cell.firing_rate(times, positions)
spike_times = generate_poisson_spikes(firing_rate_hz, times, rng=2)

# 3. Estimate the rate map: a boundary-aware, smoothed firing-rate map.
result = compute_spatial_rate(env, spike_times, times, positions, bandwidth=8.0)

# Check that the map covers the arena: most bins should have a finite rate.
assert np.isfinite(result.firing_rate).mean() > 0.8
```

!!! note "Bringing your own data"
    Replace the simulation in step 2 with your own arrays: `spike_times`
    (shape `(n_spikes,)`, seconds), `times` (shape `(n_samples,)`, seconds),
    and `positions` (shape `(n_samples, 2)`, same units as `bin_size`). Build
    `env` from your arena's outline with `Environment.from_polygon(...)`:
    `from_samples` on a trajectory leaves gaps wherever the animal did not
    go. Nothing else changes.

Required tracking pairs always put timestamps before positions:
`compute_spatial_rate(env, spike_times, times, positions)`, and behavior uses
`(env, times, positions)` or `(times, positions)`. For pynapple tracking, pass
`tsd.t, tsd.values` explicitly. Spatial rate functions require the positions
argument; a tracking object cannot replace the timestamp array.

## Inspect and plot the result

`compute_spatial_rate` returns a `SpatialRateResult`. Ask it for headline
numbers, then plot the map:

```python
# Scalar headline metrics include peak_firing_rate, spatial_information and sparsity.
print(result.summary())
# -> a dict of scalars, e.g. 2500 bins, peak firing rate ~22 Hz,
#    spatial_information ~0.95 bits/spike and ~1800 s of shared occupancy (exact values vary).

# Where the cell fires most, and how spatially informative it is.
print("Peak firing location (cm):", result.peak_location())
print("Spatial information (bits/spike):", result.spatial_information())
print("Place candidate (information screen):", result.is_place_cell())
print("Has a detected field:", result.has_place_field())

# One-row table has the same columns as a population rate table.
table = result.summary_table()
print(table)
print("Metric units:", table.attrs["units"])
print("Heuristic thresholds:", table.attrs["classification_thresholds"])

# Plot the firing-rate map; returns a Matplotlib Axes you can further style.
ax = result.plot()
ax.set_title("Rate map of a simulated place cell")
plt.show()
```

The peak sits near `(50, 50)` cm — right where the place cell was tuned — and
the map is a smooth bump with a finite rate in almost every bin of the arena
(bins the animal never visited stay NaN). The assert in step 3 checks this
coverage; with your own recording, a low fraction means the animal did not
sample much of the arena. That is the core loop: **environment +
spikes + trajectory → firing-rate map you can measure and plot.**

`result.is_place_cell()` is a fast screen: it compares spatial information,
minus what an untuned cell shows by chance, with a cutoff. For a verdict, call
the free function `is_place_cell(env, spike_times, times, positions, rng=0)`
with the raw arrays. By default it also requires the information to be
significant against 1000 circularly shifted spike trains, which costs about
1000 map recomputes; `rng=0` makes it reproducible. Results offer screens and
field detection; they keep no raw arrays.

### Native 1D fields

Native 1D grids and graph tracks both support rate plots. A native grid plots
firing rate in Hz over physical bin coordinates. Pass simulator arrays and
labels explicitly, then choose a population row or its singular result:

```python
from neurospatial import compute_spatial_rates
from neurospatial.simulation import linear_track_session

sim = linear_track_session(duration=60, n_place_cells=3, rng=0)
track_rates = compute_spatial_rates(
    sim.env, sim.spike_times, sim.times, sim.positions, unit_ids=sim.unit_ids,
)
ax = track_rates.plot(idx=0)  # Same field as track_rates[0].plot().
ax.set_title(f"Unit {track_rates.unit_ids[0]}: native 1D firing rate")
plt.show()
```

## What just happened

Four concepts carried that analysis. They are worth a minute now; the
[Core Concepts](core-concepts.md) page goes deeper.

**Environment.** `Environment.from_polygon(box(0, 0, 100, 100), ...)`
discretized the continuous arena into a grid of **bins** (small square tiles),
keeping every bin inside the arena outline: 50 x 50 = 2500 bins of 2 cm,
visited or not. `Environment.from_samples(positions, ...)` instead keeps only
the bins your samples covered. Presets cover common apparatus:
`Environment.open_field` builds the arena from a trajectory and fills the
unvisited holes inside the explored area, and
`Environment.linear_track` and `Environment.maze` build linearized tracks (see
[presets](../user-guide/environments.md#presets)). `env.n_bins` reports how
many bins there are, and every later step — occupancy, smoothing, plotting —
is defined over those bins:

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
print("Neighbors of the center bin:", env.neighbors(bins=center_bin))
```

**Recording gaps and time windows.** A missing sample, whether a dropped
frame or a NaN row, is skipped. A gap between tracked samples longer than
`max_gap` (1.0 s by default, calibrated for typical video and DeepLabCut
tracking) is excluded as a tracking loss; shorter gaps are bridged. Use `epochs=` to select
analysis windows and `spike_window=` when ephys started late or stopped early.
Without it, spikes are assumed recorded whenever position was;
`result.spike_window_assumed` makes that assumption visible.

## Next steps

- **[Core Concepts](core-concepts.md)** — bins, connectivity graphs, layout
  engines, and 1D linearized tracks, in depth.
- **[Place Field Analysis example](../examples/02_spatial_rate_maps.ipynb)** —
  the same workflow on a fuller dataset, with a whole population of cells.
- **[User Guide](../user-guide/index.md)** — decoding, trajectory and behavioral
  analysis, egocentric frames, animation, and more.
- **[API Reference](../api/index.md)** — every function and result class.
