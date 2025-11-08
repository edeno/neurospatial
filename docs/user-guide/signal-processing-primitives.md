# Signal Processing Primitives

This guide covers the spatial signal processing primitives for graph-based operations, enabling custom filtering, local aggregation, and advanced spatial analyses on irregular environments.

## Overview

Neurospatial provides two fundamental primitives for spatial signal processing:

- **`neighbor_reduce(field, env, *, op='mean', weights=None, include_self=False)`** - Aggregate field values over spatial neighborhoods
- **`convolve(field, kernel, env, *, normalize=True)`** - Apply custom convolution kernels to spatial fields

These primitives operate on **graph connectivity**, making them suitable for irregular spatial structures, mazes, and track-based environments where Euclidean operations don't apply.

## Why These Primitives Matter

Signal processing primitives provide flexible tools for custom spatial analyses that go beyond standard Gaussian smoothing:

### Neuroscience Applications

1. **Spatial Coherence** - Measure correlation between firing rate and neighbor average (Muller & Kubie 1989)
2. **Edge Detection** - Apply Mexican hat (DoG) filters to detect place field boundaries
3. **Occupancy Thresholding** - Use box filters to smooth binary occupancy masks
4. **Local Statistics** - Compute variability, extrema, or consistency across neighborhoods
5. **Custom Smoothing** - Design problem-specific kernels (e.g., directional smoothing for head direction cells)

### Comparison with `env.smooth()`

| Feature | `env.smooth()` | `neighbor_reduce()` | `convolve()` |
|---------|---------------|---------------------|--------------|
| **Purpose** | Gaussian smoothing | Local aggregation | Custom kernels |
| **Kernel** | Fixed (Gaussian) | Implicit (neighbors) | Arbitrary |
| **Operations** | Weighted average | sum/mean/max/min/std | Weighted sum |
| **Weights** | Distance-based | Optional | Distance or custom |
| **Speed** | Fast (precomputed) | Fast (sparse) | Slower (flexible) |
| **When to use** | Standard smoothing | Simple statistics | Complex filters |

## neighbor_reduce: Local Aggregation

The `neighbor_reduce()` function applies reduction operations (sum, mean, max, min, std) to the values of neighboring bins in the spatial graph.

### Basic Usage

```python
from neurospatial import Environment
from neurospatial.primitives import neighbor_reduce
import numpy as np

# Create environment
env = Environment.from_samples(trajectory_data, bin_size=2.5)

# Compute firing rate field
firing_rate = env.occupancy(spike_times, spike_positions)

# Compute neighbor mean (local spatial average)
neighbor_mean = neighbor_reduce(firing_rate, env, op='mean')

# Compute local variability
neighbor_std = neighbor_reduce(firing_rate, env, op='std')
```

### Available Operations

The `op` parameter supports five aggregation operations:

```python
# Sum: Total of neighbor values
total = neighbor_reduce(field, env, op='sum')

# Mean: Average of neighbors (default)
average = neighbor_reduce(field, env, op='mean')

# Max: Maximum neighbor value
maximum = neighbor_reduce(field, env, op='max')

# Min: Minimum neighbor value
minimum = neighbor_reduce(field, env, op='min')

# Std: Standard deviation of neighbors
variability = neighbor_reduce(field, env, op='std')
```

### Including Self in Neighborhood

By default, `neighbor_reduce()` excludes the bin itself from the neighborhood. Use `include_self=True` to include it:

```python
# Exclude self (default): average of neighbors only
neighbor_only = neighbor_reduce(field, env, op='mean', include_self=False)

# Include self: average of self + neighbors
with_self = neighbor_reduce(field, env, op='mean', include_self=True)
```

### Weighted Aggregation

For `sum` and `mean` operations, you can provide custom weights:

```python
# Distance-based weights (closer neighbors weighted more)
distances_to_goal = env.distance_to([goal_bin])
weights = np.exp(-distances_to_goal / scale)

# Weighted mean (emphasizes neighbors closer to goal)
weighted_mean = neighbor_reduce(field, env, op='mean', weights=weights)
```

### Spatial Coherence Example

Compute spatial coherence, a measure of how well firing rate predicts neighbor firing rate (Muller & Kubie 1989):

```python
import numpy as np
from neurospatial import Environment
from neurospatial.primitives import neighbor_reduce

# Create firing rate field
firing_rate = spikes_to_field(env, spike_times, times, positions)

# Compute neighbor average
neighbor_avg = neighbor_reduce(firing_rate, env, op='mean', include_self=False)

# Spatial coherence: correlation between firing rate and neighbor average
valid_mask = ~np.isnan(firing_rate) & ~np.isnan(neighbor_avg)
coherence = np.corrcoef(
    firing_rate[valid_mask],
    neighbor_avg[valid_mask]
)[0, 1]

print(f"Spatial coherence: {coherence:.3f}")
# High coherence (> 0.6): smooth place field
# Low coherence (< 0.3): fragmented or noisy field
```

### Edge Cases

**Isolated nodes**: Bins with no neighbors return `NaN`:

```python
result = neighbor_reduce(field, env, op='mean')
# result[isolated_bin] == np.nan
```

**Boundary bins**: Correctly handle fewer neighbors at environment boundaries:

```python
# Corner bins have 3 neighbors (8-connected grid)
# Edge bins have 5 neighbors
# Center bins have 8 neighbors
# Each bin's result uses only its actual neighbors
```

## convolve: Custom Filtering

The `convolve()` function applies custom spatial convolution using either a callable kernel function (distance → weight) or a precomputed kernel matrix.

### Basic Usage with Callable Kernels

```python
from neurospatial import Environment
from neurospatial.primitives import convolve
import numpy as np

# Create environment
env = Environment.from_samples(trajectory_data, bin_size=2.5)

# Define custom kernel (distance → weight)
def gaussian_kernel(distances):
    bandwidth = 5.0  # cm
    return np.exp(-(distances**2) / (2 * bandwidth**2))

# Apply convolution
smoothed = convolve(field, gaussian_kernel, env, normalize=True)
```

### Kernel Types

**Box kernel**: Uniform weights within distance threshold

```python
def box_kernel(distances):
    """Uniform weight within radius, zero outside."""
    radius = 10.0  # cm
    return np.where(distances <= radius, 1.0, 0.0)

# Box filter smoothing (uniform local average)
result = convolve(field, box_kernel, env, normalize=True)
```

**Mexican hat (DoG)**: Difference of Gaussians for edge detection

```python
def mexican_hat(distances):
    """Edge detection kernel (center-surround)."""
    sigma_center = 5.0   # cm (narrow positive peak)
    sigma_surround = 15.0  # cm (wide negative surround)

    center = np.exp(-(distances**2) / (2 * sigma_center**2))
    surround = np.exp(-(distances**2) / (2 * sigma_surround**2))

    return center - surround

# Edge detection (DON'T normalize - breaks edge detection)
edges = convolve(field, mexican_hat, env, normalize=False)

# Positive values: local maxima (place field centers)
# Negative values: local minima (place field boundaries)
```

**Custom distance functions**: Any distance-based weighting

```python
def power_law_kernel(distances):
    """Power-law decay (slower than Gaussian)."""
    scale = 10.0
    power = 1.5
    return scale / (1 + distances)**power

result = convolve(field, power_law_kernel, env, normalize=True)
```

### Precomputed Kernel Matrix

For repeated operations or complex kernels, precompute the kernel matrix:

```python
import numpy as np

# Compute kernel matrix (n_bins × n_bins)
kernel_matrix = np.zeros((env.n_bins, env.n_bins))

for i in range(env.n_bins):
    for j in range(env.n_bins):
        # Compute distance from bin i to bin j
        if i == j:
            kernel_matrix[i, j] = 1.0  # Self weight
        else:
            dist = env.distance_between(env.bin_centers[i], env.bin_centers[j])
            kernel_matrix[i, j] = np.exp(-(dist**2) / (2 * bandwidth**2))

# Apply precomputed kernel (fast for repeated use)
smoothed = convolve(field, kernel_matrix, env, normalize=True)
```

### Normalization

**Normalized convolution** (`normalize=True`, default): Weights sum to 1 per bin

- Preserves constant fields (if input is constant, output is constant)
- Appropriate for smoothing operations
- Handles NaN by renormalizing over valid neighbors

```python
# Normalized: preserves field scale
smoothed = convolve(field, gaussian_kernel, env, normalize=True)

# For constant field, output equals input
constant_field = np.ones(env.n_bins) * 5.0
result = convolve(constant_field, gaussian_kernel, env, normalize=True)
assert np.allclose(result, 5.0)  # Preserved
```

**Unnormalized convolution** (`normalize=False`): Use raw kernel weights

- Required for edge detection kernels (Mexican hat, LoG)
- Sum of kernel may not be 1 (e.g., Mexican hat sums to ~0)
- Output scale depends on kernel amplitude

```python
# Unnormalized: preserves kernel structure
edges = convolve(field, mexican_hat, env, normalize=False)

# Mexican hat kernel sums to ~0 (center-surround cancellation)
# Normalization would destroy the edge detection property
```

### NaN Handling

`convolve()` handles NaN values gracefully by excluding them from convolution:

```python
# Field with NaN (unvisited bins)
field_with_nan = firing_rate.copy()
field_with_nan[unvisited_bins] = np.nan

# NaN values are skipped in convolution
# Weights are renormalized over valid neighbors
smoothed = convolve(field_with_nan, gaussian_kernel, env, normalize=True)

# NaN doesn't propagate to neighbors
# Each bin uses only non-NaN neighbor values
```

**Important**: Bins surrounded entirely by NaN will remain NaN:

```python
# Isolated valid bin surrounded by NaN
field = np.full(env.n_bins, np.nan)
field[center_bin] = 1.0

result = convolve(field, gaussian_kernel, env, normalize=True)
# result[center_bin] will be valid (self-convolution)
# Neighbors may be NaN (no valid neighbors to smooth with)
```

## Practical Examples

### Example 1: Occupancy Thresholding with Box Filter

Smooth binary occupancy to reduce noise from single-visit bins:

```python
from neurospatial import Environment
from neurospatial.primitives import convolve
import numpy as np

# Compute binary occupancy (visited vs unvisited)
occupancy = env.occupancy(times, positions)
binary_occupancy = (occupancy > 0).astype(float)

# Box filter: smooth over 10cm radius
def box_kernel(distances):
    return np.where(distances <= 10.0, 1.0, 0.0)

# Smooth binary occupancy
smoothed_occupancy = convolve(binary_occupancy, box_kernel, env, normalize=True)

# Threshold: keep bins with >50% neighbors visited
reliable_visited = smoothed_occupancy > 0.5
```

### Example 2: Place Field Boundary Detection

Use Mexican hat filter to detect place field boundaries:

```python
from neurospatial import Environment
from neurospatial.primitives import convolve
import numpy as np
import matplotlib.pyplot as plt

# Compute smoothed firing rate
firing_rate = compute_place_field(
    env, spike_times, times, positions,
    smoothing_bandwidth=5.0
)

# Mexican hat for edge detection
def mexican_hat(distances):
    sigma_center = 5.0
    sigma_surround = 15.0
    center = np.exp(-(distances**2) / (2 * sigma_center**2))
    surround = np.exp(-(distances**2) / (2 * sigma_surround**2))
    return center - surround

# Detect edges (DO NOT normalize)
edges = convolve(firing_rate, mexican_hat, env, normalize=False)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original field
axes[0].scatter(env.bin_centers[:, 0], env.bin_centers[:, 1], c=firing_rate, cmap='hot')
axes[0].set_title('Firing Rate')

# Edges (positive = local maxima)
axes[1].scatter(env.bin_centers[:, 0], env.bin_centers[:, 1], c=edges, cmap='RdBu_r')
axes[1].set_title('Edges (Mexican Hat)')

# Detect place field centers (positive peaks)
field_centers = edges > np.percentile(edges, 95)
axes[2].scatter(env.bin_centers[:, 0], env.bin_centers[:, 1], c=field_centers, cmap='gray')
axes[2].set_title('Detected Field Centers')

plt.tight_layout()
plt.show()
```

### Example 3: Spatial Coherence Analysis

Compute spatial coherence as a measure of place field quality:

```python
from neurospatial import Environment
from neurospatial.primitives import neighbor_reduce
import numpy as np

# Compute firing rate
firing_rate = spikes_to_field(env, spike_times, times, positions)

# Compute neighbor average (exclude self)
neighbor_avg = neighbor_reduce(firing_rate, env, op='mean', include_self=False)

# Spatial coherence: Pearson correlation
valid = ~np.isnan(firing_rate) & ~np.isnan(neighbor_avg)
correlation = np.corrcoef(firing_rate[valid], neighbor_avg[valid])[0, 1]

print(f"Spatial coherence: {correlation:.3f}")

# Interpretation (Muller & Kubie 1989):
# > 0.6: High coherence (good place field)
# 0.3-0.6: Moderate coherence
# < 0.3: Low coherence (noisy or fragmented field)
```

### Example 4: Local Field Variability

Measure local consistency of firing rates:

```python
from neurospatial.primitives import neighbor_reduce

# Compute firing rate
firing_rate = spikes_to_field(env, spike_times, times, positions)

# Local variability (standard deviation of neighbors)
local_std = neighbor_reduce(firing_rate, env, op='std', include_self=False)

# Normalize by local mean
local_mean = neighbor_reduce(firing_rate, env, op='mean', include_self=False)
coefficient_of_variation = local_std / (local_mean + 1e-6)

# High CV: variable firing (place field edge)
# Low CV: consistent firing (place field center or background)
```

## When to Use Which Tool

### Use `env.smooth()` when:
- You need standard Gaussian smoothing
- Performance is critical (precomputed kernels)
- You want automatic bandwidth selection
- You're following standard neuroscience practices

### Use `neighbor_reduce()` when:
- You need simple aggregation (sum, max, min, std)
- You want unweighted or custom-weighted averages
- You need local statistics (variability, extrema)
- You're computing spatial coherence

### Use `convolve()` when:
- You need custom kernel shapes (box, Mexican hat)
- You're doing edge detection or feature extraction
- You need control over normalization
- You want problem-specific filtering (directional, anisotropic)

## Mathematical Background

### Convolution on Graphs

Classical convolution in Euclidean space:

$$(f * g)(x) = \int f(y) \, g(x - y) \, dy$$

Graph convolution replaces integration with summation over graph nodes:

$$(f * g)_i = \sum_j g_{ij} \, f_j$$

where $g_{ij}$ is the kernel weight from node $j$ to node $i$.

### Normalized Convolution

Normalized convolution ensures constant fields remain constant:

$$(f * g)_i = \frac{\sum_j g_{ij} \, f_j}{\sum_j g_{ij}}$$

This is equivalent to normalizing kernel weights to sum to 1 per bin.

### Relationship to Differential Operators

Convolution can implement discrete derivatives:

- **Gradient magnitude**: Mexican hat (center-surround) approximates Laplacian operator
- **Smoothness**: Gaussian convolution approximates heat diffusion
- **Edge detection**: DoG kernel approximates Laplacian of Gaussian (LoG)

For exact differential operators, see [Differential Operators](differential-operators.md).

## Performance Notes

### neighbor_reduce()

- **Time complexity**: O(n_bins × avg_degree)
- **Space complexity**: O(n_bins)
- **Optimization**: Uses NetworkX neighbor iteration (sparse graph)
- **Typical performance**: ~3 µs per bin for 8-connected grids

### convolve()

**Callable kernels**:
- **Time complexity**: O(n_bins²) for distance computation
- **Space complexity**: O(n_bins²) for kernel matrix
- **Optimization**: Precompute kernel matrix for repeated use

**Precomputed kernels**:
- **Time complexity**: O(n_bins²) for matrix multiplication
- **Space complexity**: O(n_bins²) for kernel storage
- **Optimization**: Use sparse matrices if kernel is sparse

**Comparison with `env.smooth()`**:
- `env.smooth()`: ~50x faster (precomputed kernels + caching)
- `convolve()`: More flexible but slower
- For repeated Gaussian smoothing, use `env.smooth()`

## Advanced Topics

### Directional Smoothing

Create anisotropic kernels for directional place fields (e.g., head direction cells):

```python
def directional_kernel(distances, angles, preferred_direction):
    """Smooth more along preferred direction."""
    distance_weight = np.exp(-(distances**2) / (2 * bandwidth**2))

    # Angular difference from preferred direction
    angular_diff = np.abs(angles - preferred_direction)
    angular_diff = np.minimum(angular_diff, 2*np.pi - angular_diff)

    # Angular weight (wider tolerance)
    angular_weight = np.exp(-(angular_diff**2) / (2 * (np.pi/4)**2))

    return distance_weight * angular_weight

# Apply directional smoothing
# Note: Requires computing angles between bins (use env.connectivity edge attributes)
```

### Multi-Scale Analysis

Apply convolution at multiple scales to detect features of different sizes:

```python
# Multi-scale Gaussian kernels
scales = [5.0, 10.0, 20.0]  # cm

results = []
for scale in scales:
    def kernel(distances):
        return np.exp(-(distances**2) / (2 * scale**2))

    smoothed = convolve(field, kernel, env, normalize=True)
    results.append(smoothed)

# Detect features by comparing scales
# Large features: similar across scales
# Small features: strong at small scales, weak at large scales
```

### Kernel Design Principles

**For smoothing** (use `normalize=True`):
- Positive weights (all kernel values ≥ 0)
- Weights sum to 1 (or close to 1)
- Monotonically decreasing with distance
- Examples: Gaussian, box, power-law

**For edge detection** (use `normalize=False`):
- Mixed signs (positive center, negative surround)
- Kernel sums to ~0 (no DC response)
- Zero-crossing at feature scale
- Examples: Mexican hat, LoG, Laplacian

## See Also

- [Differential Operators](differential-operators.md) - Gradient, divergence, Laplacian
- [Spike Field Primitives](spike-field-primitives.md) - Converting spikes to fields
- [Neuroscience Metrics](neuroscience-metrics.md) - Standard spatial metrics

## References

1. **Muller & Kubie (1989)**. The firing of hippocampal place cells predicts the future position of freely moving rats. *Journal of Neuroscience*, 9(12), 4101-4110.
   - Original spatial coherence metric

2. **Shuman et al. (2013)**. The emerging field of signal processing on graphs. *IEEE Signal Processing Magazine*, 30(3), 83-98.
   - Mathematical foundation for graph signal processing

3. **Marr & Hildreth (1980)**. Theory of edge detection. *Proceedings of the Royal Society B*, 207(1167), 187-217.
   - Laplacian of Gaussian (LoG) for edge detection

4. **Diehl et al. (2017)**. Grid and nongrid cells in medial entorhinal cortex represent spatial location and environmental features with complementary coding schemes. *Neuron*, 94(1), 83-92.
   - Applications of spatial filtering in grid cell analysis
