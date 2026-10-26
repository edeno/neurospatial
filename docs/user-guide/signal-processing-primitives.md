# Signal Processing Primitives

Use graph operations to describe a spatial field's neighborhood or build a custom filter. For standard firing-rate estimation, start with [`compute_spatial_rate()`](spike-field-primitives.md): it handles spike counts, occupancy, and observation support together.

| Operation | Use | Scope |
|---|---|---|
| `env.smooth(field, bandwidth=..., mode="average")` | Smooth an intensive field, such as Hz | Finite-volume heat diffusion |
| `neighbor_reduce(env, field, op="mean")` | Aggregate immediate graph neighbors | One graph step |
| `graph_convolve(env, field, kernel, normalize=True)` | Apply custom distance weights | Graph shortest-path distances |

Import the last two functions from `neurospatial.ops.graph`. Graph connectivity represents permitted movement; an unvisited bin inferred from tracking is not evidence of a physical wall. Define known geometry before interpreting a boundary-aware filter.

## A working field with observation support

This small synthetic recording is sufficient to run the recipes below in order. Coordinates are centimeters, timestamps are seconds, and rates are Hz. The known arena includes unvisited bins.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment
from neurospatial.encoding import compute_spatial_rate
from neurospatial.ops.graph import neighbor_reduce
from neurospatial.simulation import generate_poisson_spikes

edges = np.arange(0.0, 100.0 + 10.0, 10.0)
env = Environment.from_grid_mask(
    active_mask=np.ones((10, 10), dtype=bool),
    grid_edges=(edges, edges),
)
env = env.with_metadata(units="cm")
times = np.arange(2400, dtype=float) / 20.0
positions = 50.0 + 35.0 * np.column_stack(
    [np.sin(times / 4.0), np.cos(times / 7.0)]
)
instantaneous_rate = 0.2 + 12.0 * np.exp(
    -np.sum((positions - [65.0, 35.0]) ** 2, axis=1) / (2 * 12.0**2)
)
spike_times = generate_poisson_spikes(instantaneous_rate, times, seed=707)
result = compute_spatial_rate(
    env, spike_times, times, positions,
    bandwidth=5.0, min_occupancy=0.1, max_gap=0.1,
)
field = np.where(result.spatial_support, result.firing_rate, np.nan)
neighbor_mean = neighbor_reduce(env, field, op="mean")
assert field.shape == neighbor_mean.shape == (env.n_bins,)
```

`neighbor_reduce` propagates a missing neighbor's NaN. It does not silently interpret it as zero or ignore it. A field of zero Hz at an observed bin and a missing rate estimate carry different information. Keep the original support mask when producing derived summaries.

## Local aggregation and weights

The five operations are `sum`, `mean`, `max`, `min`, and `std`. By default they exclude the bin itself; `include_self=True` includes it. Empty neighborhoods return NaN. Boundary nodes use their actual neighbors, whose number depends on the chosen connectivity.

The following analytic field has no missing observations, so every graph neighbor can contribute:

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial.ops.graph import neighbor_reduce

field = np.sin(env.bin_centers[:, 0] / 10.0)
local_mean = neighbor_reduce(env, field, op="mean")
local_std = neighbor_reduce(env, field, op="std")
local_max = neighbor_reduce(env, field, op="max", include_self=True)
assert local_mean.shape == local_std.shape == local_max.shape == (env.n_bins,)
```

Only `sum` and `mean` accept per-source-bin `weights`. To average observed rates while excluding missing source bins, supply an explicit zero weight for missing sources and a finite placeholder for their values. The placeholder does not create observation support:

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial.encoding import compute_spatial_rate
from neurospatial.ops.graph import neighbor_reduce

result = compute_spatial_rate(
    env, spike_times, times, positions, bandwidth=5.0, min_occupancy=0.1
)
observed = result.spatial_support & np.isfinite(result.firing_rate)
values = np.where(observed, result.firing_rate, 0.0)
weights = np.where(observed, result.occupancy, 0.0)
local_mean = neighbor_reduce(env, values, op="mean", weights=weights)
local_mean[~observed] = np.nan
```

These weights give more influence to neighbors with greater dwell time. Using `observed.astype(float)` instead gives equal weight to observed neighbor bins. Neither operation replaces estimating a rate from jointly smoothed counts and time.

### Descriptive spatial coherence

Correlate each observed bin's rate with its observed-neighbor average. Restrict both sides of the correlation to valid support and check that they vary:

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial.encoding import compute_spatial_rate
from neurospatial.ops.graph import neighbor_reduce

result = compute_spatial_rate(
    env, spike_times, times, positions, bandwidth=5.0, min_occupancy=0.1
)
observed = result.spatial_support & np.isfinite(result.firing_rate)
values = np.where(observed, result.firing_rate, 0.0)
neighbor_mean = neighbor_reduce(
    env, values, op="mean", weights=observed.astype(float)
)
valid = observed & np.isfinite(neighbor_mean)
coherence = np.nan
if valid.sum() >= 3:
    x, y = values[valid], neighbor_mean[valid]
    if np.ptp(x) > 0 and np.ptp(y) > 0:
        coherence = float(np.corrcoef(x, y)[0, 1])
print(f"Descriptive coherence: {coherence:.3f}")
```

Coherence depends on binning, support, smoothing, and neighborhood definition. Smoothing a map before measuring coherence can raise it by construction. There is no universal coherence cutoff that establishes a place cell; use a stated analysis and validation procedure.

## Custom graph-distance filtering

`graph_convolve(env, field, kernel)` accepts either a callable applied to the full pairwise graph-distance matrix or a precomputed `(n_bins, n_bins)` weight matrix. Entry `[i, j]` weights source bin `j` at destination bin `i`.

For positive kernels, `normalize=True` divides each destination's weights by the total over non-NaN source values. It preserves a constant field wherever a source contributes. A finite prediction at an unobserved destination does not extend measured support.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial.ops.graph import graph_convolve

field = np.sin(env.bin_centers[:, 0] / 10.0)
field[0] = np.nan

def box_kernel(distances):
    return (distances <= 10.0).astype(float)  # Radius in cm

local_average = graph_convolve(env, field, box_kernel, normalize=True)
constant = graph_convolve(env, np.full(env.n_bins, 5.0), box_kernel)
assert np.allclose(constant, 5.0)
```

Other positive distance profiles, such as Gaussian or power-law weights, use the same call. A one-observation field can predict values wherever its kernel reaches; destinations with no contributing source remain NaN under normalized convolution. A compact box kernel and an unbounded Gaussian therefore have different support behavior.

### Balanced center-surround contrast

Subtract a broad local average from a narrow average. Normalize the two averages separately over observed source bins. This removes constant fields even near boundaries and missing observations. Subtracting two equal-height Gaussians directly would give zero weight at distance zero and negative weights elsewhere.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial.encoding import compute_spatial_rate
from neurospatial.ops.graph import graph_convolve

result = compute_spatial_rate(
    env, spike_times, times, positions, bandwidth=5.0, min_occupancy=0.1
)
observed = result.spatial_support & np.isfinite(result.firing_rate)
field = np.where(observed, result.firing_rate, np.nan)

def narrow(distances):
    return np.exp(-(distances**2) / (2 * 5.0**2))

def broad(distances):
    return np.exp(-(distances**2) / (2 * 15.0**2))

contrast = graph_convolve(env, field, narrow) - graph_convolve(env, field, broad)
contrast[~observed] = np.nan
constant = np.where(observed, 5.0, np.nan)
constant_contrast = (
    graph_convolve(env, constant, narrow) - graph_convolve(env, constant, broad)
)
assert np.allclose(constant_contrast[observed], 0.0, atol=1e-12)
```

Positive contrast indicates greater narrow-scale than broad-scale activity; negative contrast indicates the reverse. It is not a field boundary, a field center, or a significance test by itself. Thresholding the top 5% will select bins even for an unsuitable map.

Use `normalize=False` only when your supplied weights already have the intended scale or signed balance. A signed center-surround matrix should have zero row sums to remove constants; re-normalizing it by its signed sum destroys that property. Without normalization, a destination is NaN wherever a nonzero weight falls on a NaN source, because the weighted sum is unknown; with a full-support kernel such as a Gaussian, that is every connected bin. Set those weights to exactly zero in a precomputed matrix, or fill the field explicitly, to decide how missing sources count.

### Reusing a precomputed matrix

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial.ops.distance import geodesic_distance_matrix
from neurospatial.ops.graph import graph_convolve

distances = geodesic_distance_matrix(env.connectivity, weight="distance")
kernel_matrix = np.exp(-(distances**2) / (2 * 5.0**2))
field = np.sin(env.bin_centers[:, 0] / 10.0)
filtered = graph_convolve(env, field, kernel_matrix)
assert filtered.shape == (env.n_bins,)
```

A callable distance kernel also builds dense distance and weight matrices. Both approaches require quadratic memory in the number of bins. For repeated standard diffusion smoothing, prefer the cached matrix-free `env.smooth()` path. `mode="average"` preserves intensive constants; `mode="density"` and `mode="transition"` have different density and mass interpretations. See [spatial analysis](spatial-analysis.md).

## Mathematical interpretation

Graph filtering is a weighted sum over source bins:

$$
y_i = \sum_j K_{ij} f_j.
$$

For a positive normalized kernel and observed source set $O$:

$$
y_i = \frac{\sum_{j \in O} K_{ij} f_j}{\sum_{j \in O} K_{ij}}.
$$

This is a bin-weighted average. On irregular layouts with unequal cell areas, it is not automatically an area-weighted continuum convolution. The finite-volume diffusion used by `env.smooth()` accounts for the published cell geometry.

For multi-scale analysis, repeat a filter with explicit physical scales and compare the supported results. For anisotropic filters, construct a weight matrix with a declared orientation convention and retain the permitted graph connections. A distance-only callable cannot encode heading dependence on its own.

Graph derivatives have physical length, face, and volume factors. A difference of Gaussian averages is a scale-dependent contrast operator, not the gradient magnitude. Use the [differential-operator guide](differential-operators.md) for `gradient`, `divergence`, and finite-volume Laplacians.

## Performance and interpretation

- Neighborhood sums and means use sparse adjacency operations; nonlinear reductions operate over the actual local neighbors.
- Dense graph convolution is flexible but becomes expensive as the number of bins grows. Avoid unsupported fixed speedup claims when choosing an implementation.
- A smooth, coherent, or high-contrast map is a description of the supplied field. Keep occupancy, recording gaps, and the original support available for validation.

See [neuroscience metrics](neuroscience-metrics.md) for spatial metrics and [spike-field primitives](spike-field-primitives.md) for rate estimation. For the graph signal-processing framework, see Shuman et al. (2013), *The emerging field of signal processing on graphs*, IEEE Signal Processing Magazine 30(3), 83–98.
