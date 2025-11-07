# Neuroscience Metrics for Spatial Analysis

This guide covers standard neuroscience metrics implemented in neurospatial for analyzing spatial cells, including place cells, boundary cells, and population-level spatial representations. These metrics are validated against field-standard packages (opexebo, neurocode, buzcode) and implement algorithms from peer-reviewed publications.

## Overview

The `neurospatial.metrics` module provides three categories of metrics:

1. **Place Field Metrics** - Single-cell spatial tuning analysis
2. **Population Metrics** - Multi-cell spatial representation analysis
3. **Boundary Cell Metrics** - Border cell detection and analysis

All metrics follow the neurospatial API convention: **field/data comes first, then environment** in the parameter order.

## Place Field Detection and Metrics

### Detecting Place Fields

Place fields are localized regions of elevated firing rate that characterize spatial cells. The `detect_place_fields()` function implements an iterative peak-based detection algorithm adapted from neurocode (AyA Lab):

```python
import numpy as np
from neurospatial import Environment
from neurospatial.metrics import detect_place_fields

# Create environment and compute firing rate map
env = Environment.from_samples(positions, bin_size=2.5)
firing_rate = spikes_to_field(env, spike_times, times, positions)

# Detect place fields
fields = detect_place_fields(
    firing_rate,
    env,
    threshold=0.2,          # 20% of peak rate
    min_size=None,          # Auto-compute from bin size
    max_mean_rate=10.0,     # Exclude interneurons (>10 Hz)
    detect_subfields=True   # Discriminate coalescent fields
)

# Result: list of arrays
# fields[0] = np.array([10, 11, 12, 23, 24, 25])  # Bin indices for field 1
# fields[1] = np.array([45, 46, 56, 57])          # Bin indices for field 2
```

**Algorithm details:**

1. **Iterative peak detection**: Identifies local maxima above threshold
2. **Connected component extraction**: Groups neighboring bins into fields
3. **Interneuron exclusion**: Filters cells with mean rate > 10 Hz (hippocampus standard)
4. **Subfield discrimination**: Recursively re-thresholds at 50% and 70% to separate coalescent fields

**Key parameters:**

- `threshold`: Fraction of peak rate (default 0.2 = 20%, following Muller & Kubie, 1989)
- `min_size`: Minimum field size in bins (defaults to reasonable value based on bin size)
- `max_mean_rate`: Maximum mean firing rate (10 Hz standard for pyramidal cells)
- `detect_subfields`: Enable subfield discrimination (default True, following neurocode)

### Field Size

Compute the physical area of a place field:

```python
from neurospatial.metrics import field_size

# Get size of first detected field
area = field_size(fields[0], env)
# Returns: area in squared physical units (e.g., cm²)
```

Field size is computed as the sum of individual bin areas. For regular grids, each bin has area ≈ `bin_size²`. For irregular graphs, areas are estimated from Voronoi cell volumes.

### Field Centroid

Compute the firing-rate-weighted center of mass of a place field:

```python
from neurospatial.metrics import field_centroid

# Compute centroid of first field
center = field_centroid(firing_rate, fields[0], env)
# Returns: array of shape (n_dims,) with N-D coordinates
```

The centroid is the weighted average position where weights are the firing rates:

$$
\text{centroid} = \frac{\sum_i \text{rate}_i \cdot \text{position}_i}{\sum_i \text{rate}_i}
$$

### Skaggs Spatial Information

Skaggs spatial information quantifies how much information (in bits) a cell's firing conveys about the animal's spatial location (Skaggs et al., 1996):

```python
from neurospatial.metrics import skaggs_information

# Compute occupancy
occupancy = env.occupancy(times, positions, return_seconds=True)

# Compute spatial information
info = skaggs_information(firing_rate, occupancy, base=2.0)
# Returns: bits per spike
```

**Formula:**

$$
I = \sum_i p_i \frac{r_i}{\bar{r}} \log_2\left(\frac{r_i}{\bar{r}}\right)
$$

where:
- $p_i$ = occupancy probability (normalized occupancy)
- $r_i$ = firing rate in bin $i$
- $\bar{r}$ = mean firing rate across all bins

**Interpretation:**
- **High values (>1 bit/spike)**: Strong spatial selectivity
- **Low values (<0.5 bit/spike)**: Weak or no spatial tuning
- **Units**: bits per spike (information per action potential)

### Sparsity

Sparsity measures how selectively a cell fires across space (Skaggs et al., 1996):

```python
from neurospatial.metrics import sparsity

sparseness = sparsity(firing_rate, occupancy)
# Returns: value in [0, 1]
```

**Formula:**

$$
\text{sparsity} = \frac{\left(\sum_i p_i r_i\right)^2}{\sum_i p_i r_i^2}
$$

**Interpretation:**
- **0**: Cell fires uniformly across entire environment
- **1**: Cell fires in only one location (maximally sparse)
- Typical place cells: 0.2-0.5 (fire in 20-50% of environment)

### Field Stability

Measure the stability of place fields across recording sessions:

```python
from neurospatial.metrics import field_stability

# Compute firing rate maps from two sessions
rate_map_session1 = spikes_to_field(env, spikes1, times1, positions1)
rate_map_session2 = spikes_to_field(env, spikes2, times2, positions2)

# Compute stability
stability = field_stability(
    rate_map_session1,
    rate_map_session2,
    method='pearson'  # or 'spearman'
)
# Returns: correlation coefficient [-1, 1]
```

**Methods:**
- `'pearson'`: Pearson correlation (linear relationship, parametric)
- `'spearman'`: Spearman rank correlation (monotonic relationship, non-parametric)

**Interpretation:**
- **>0.7**: Highly stable field
- **0.3-0.7**: Moderately stable
- **<0.3**: Unstable or remapped field

## Population-Level Metrics

### Population Coverage

Fraction of environment covered by place fields across a population:

```python
from neurospatial.metrics import population_coverage

# Detect fields for all cells
all_place_fields = [
    detect_place_fields(rate_map_cell1, env),
    detect_place_fields(rate_map_cell2, env),
    detect_place_fields(rate_map_cell3, env),
]

# Compute coverage
coverage = population_coverage(all_place_fields, env.n_bins)
# Returns: fraction in [0, 1]
```

Coverage is the fraction of bins contained in at least one place field. High coverage (>0.8) indicates the population represents most of the environment.

### Field Density Map

Count how many place fields overlap at each location:

```python
from neurospatial.metrics import field_density_map

density = field_density_map(all_place_fields, env.n_bins)
# Returns: array of shape (n_bins,) with overlap counts
```

Useful for identifying "hotspots" where many cells have overlapping fields, which may indicate salient locations (e.g., reward zones, decision points).

### Field Overlap

Measure the spatial overlap between two fields using the Jaccard index:

```python
from neurospatial.metrics import field_overlap

# Compare fields from two cells
overlap = field_overlap(field_bins_cell1, field_bins_cell2)
# Returns: Jaccard coefficient in [0, 1]
```

**Formula:**

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

**Interpretation:**
- **0**: No overlap (fields completely disjoint)
- **1**: Perfect overlap (identical fields)
- **~0.5**: Moderate overlap (typical for neighboring place cells)

### Count Place Cells

Count cells exceeding a spatial information threshold:

```python
from neurospatial.metrics import count_place_cells

# Compute spatial information for all cells
spatial_info = [
    skaggs_information(rate_map1, occupancy),
    skaggs_information(rate_map2, occupancy),
    skaggs_information(rate_map3, occupancy),
]

# Count place cells
n_place_cells = count_place_cells(
    spatial_info,
    threshold=0.5  # bits/spike
)
# Returns: integer count
```

Standard threshold: **0.5 bits/spike** (Skaggs et al., 1996; Thompson & Best, 1989).

### Population Vector Correlation

Compute pairwise correlations between firing rate maps:

```python
from neurospatial.metrics import population_vector_correlation

# Stack rate maps into matrix (n_cells × n_bins)
population_matrix = np.array([
    rate_map_cell1,
    rate_map_cell2,
    rate_map_cell3,
])

# Compute correlation matrix
corr_matrix = population_vector_correlation(population_matrix)
# Returns: (n_cells, n_cells) correlation matrix
```

Useful for identifying cell assemblies and functional clustering in spatial representations.

## Boundary Cell Metrics

### Border Score

The border score quantifies how strongly a cell's firing field is aligned with environmental boundaries (walls). Implements the algorithm from Solstad et al. (2008), adapted for arbitrary graph-based environments:

```python
from neurospatial.metrics import border_score

# Compute firing rate map
firing_rate = spikes_to_field(env, spike_times, times, positions)

# Compute border score
score = border_score(
    firing_rate,
    env,
    threshold=0.3,    # 30% of peak (Solstad et al. standard)
    min_area=200.0    # Minimum field area (cm²)
)
# Returns: score in [-1, 1]
```

**Algorithm (adapted for irregular graphs):**

1. Segment field at threshold: bins where `firing_rate >= threshold × peak_rate`
2. Compute boundary coverage (cM): fraction of boundary bins in field
3. Compute normalized mean distance (d): mean distance from field bins to nearest boundary
4. Border score: `(cM - d) / (cM + d)`

**Interpretation:**
- **+1**: Perfect border cell (field covers boundary, far from center)
- **0**: No boundary preference (uniform or mixed)
- **-1**: Anti-border (field in center, far from boundaries)
- **>0.5**: Strong boundary cell (Solstad et al. criterion)

**Key differences from original algorithm:**

The original Solstad et al. (2008) paper used rectangular arenas with 4 discrete walls. This implementation generalizes to arbitrary graph-based environments:

- **Boundary definition**: Uses `env.boundary_bins` (graph-based boundary detection) instead of discrete walls
- **Distance metric**: Uses graph geodesic distances instead of Euclidean distances
- **Coverage**: Computed over all boundary bins (not per-wall)
- **Normalization**: Distance normalized by environment extent (bounding box diagonal)

This adaptation is appropriate for irregular layouts, mazes, and complex environments where the concept of "walls" doesn't apply cleanly.

**Parameters:**

- `threshold`: Fraction of peak for field segmentation (default 0.3 = 30%, following Solstad et al.)
- `min_area`: Minimum field area to compute score (default 0.0, Solstad et al. used 200 cm² for rats)

**Example: Validating a border cell**

```python
# Expected behavior for a true border cell:
# - High firing along one or more walls
# - Low firing in center

# Generate synthetic border cell
firing_rate = np.zeros(env.n_bins)
boundary_bins = env.boundary_bins
firing_rate[boundary_bins] = 5.0  # High firing at boundaries

score = border_score(firing_rate, env)
print(f"Border score: {score:.3f}")  # Should be > 0.5

# Visualize
import matplotlib.pyplot as plt
env.plot_field(firing_rate, cmap='hot')
plt.title(f"Border Cell (score = {score:.2f})")
plt.show()
```

## References and Cross-References

### Scientific References

**Place Field Analysis:**
- O'Keefe, J., & Dostrovsky, J. (1971). The hippocampus as a spatial map. Brain Research, 34(1), 171-175.
- Muller, R. U., & Kubie, J. L. (1989). The effects of changes in the environment on the spatial firing of hippocampal complex-spike cells. Journal of Neuroscience, 9(1), 137-154.
- Skaggs, W. E., McNaughton, B. L., Wilson, M. A., & Barnes, C. A. (1996). Theta phase precession in hippocampal neuronal populations and the compression of temporal sequences. Hippocampus, 6(2), 149-172.
- Thompson, L. T., & Best, P. J. (1989). Place cells and silent cells in the hippocampus of freely-behaving rats. Journal of Neuroscience, 9(7), 2382-2390.

**Boundary Cell Analysis:**
- Solstad, T., Boccara, C. N., Kropff, E., Moser, M. B., & Moser, E. I. (2008). Representation of geometric borders in the entorhinal cortex. Science, 322(5909), 1865-1868.

### Package Cross-References

**opexebo** - Open-source spatial analysis toolbox:
- Place field detection: Similar iterative approach
- Spatial information: Matches Skaggs formula exactly
- Sparsity: Compatible definition (validated)
- Border score: Different implementation (rectangular arenas only)

**neurocode** - AyA Lab analysis tools (MATLAB):
- Place field detection: This implementation follows neurocode's `FindPlaceFields.m` algorithm
- Subfield discrimination: Matches neurocode's recursive thresholding approach
- Parameters: Default thresholds match neurocode standards

**buzcode** - Buzsaki Lab analysis suite (MATLAB):
- Spatial information: Compatible with buzcode's `bz_spatialInfo.m`
- Interneuron exclusion: 10 Hz threshold matches buzcode standards

### Validation Notes

The neurospatial metrics have been **validated** to match reference implementations where applicable:

1. **Place field detection**: Algorithm validated against neurocode (iterative peak-based approach with subfield discrimination)
2. **Spatial information**: Formula matches opexebo, neurocode, and buzcode exactly
3. **Sparsity**: Definition compatible with all three reference packages
4. **Border score**: Intentionally different adaptation for irregular graphs (see algorithm notes above)

**Key advantages of neurospatial metrics:**

- ✅ **Graph-based environments**: Works on irregular layouts, not just rectangular grids
- ✅ **Consistent API**: All metrics follow neurospatial's parameter order convention
- ✅ **Type-safe**: Full mypy type checking with zero errors
- ✅ **Well-documented**: Comprehensive NumPy-style docstrings with examples
- ✅ **Scientific validation**: Algorithms from peer-reviewed publications

## Common Workflows

### Complete Place Cell Analysis Pipeline

```python
import numpy as np
from neurospatial import Environment
from neurospatial import spikes_to_field
from neurospatial.metrics import (
    detect_place_fields,
    field_size,
    field_centroid,
    skaggs_information,
    sparsity,
)

# 1. Create environment and compute firing rate
env = Environment.from_samples(positions, bin_size=2.5)
firing_rate = spikes_to_field(env, spike_times, times, positions, min_occupancy_seconds=0.5)

# 2. Detect place fields
fields = detect_place_fields(firing_rate, env, detect_subfields=True)
print(f"Detected {len(fields)} place fields")

# 3. Compute field properties
for i, field in enumerate(fields):
    area = field_size(field, env)
    center = field_centroid(firing_rate, field, env)
    print(f"Field {i+1}: area={area:.1f} cm², center={center}")

# 4. Compute single-cell metrics
occupancy = env.occupancy(times, positions, return_seconds=True)
info = skaggs_information(firing_rate, occupancy)
sparse = sparsity(firing_rate, occupancy)
print(f"Spatial information: {info:.3f} bits/spike")
print(f"Sparsity: {sparse:.3f}")

# 5. Classify as place cell
is_place_cell = info > 0.5  # Standard threshold
print(f"Place cell: {is_place_cell}")
```

### Population-Level Analysis

```python
from neurospatial.metrics import (
    population_coverage,
    field_density_map,
    count_place_cells,
)

# Analyze multiple cells
all_fields = []
spatial_info = []

for cell_spikes in all_spike_trains:
    # Compute firing rate
    rate_map = spikes_to_field(env, cell_spikes, times, positions)

    # Detect fields
    fields = detect_place_fields(rate_map, env)
    all_fields.append(fields)

    # Compute metrics
    info = skaggs_information(rate_map, occupancy)
    spatial_info.append(info)

# Population metrics
coverage = population_coverage(all_fields, env.n_bins)
density = field_density_map(all_fields, env.n_bins)
n_place_cells = count_place_cells(spatial_info, threshold=0.5)

print(f"Population coverage: {coverage:.1%}")
print(f"Place cells: {n_place_cells}/{len(all_spike_trains)}")

# Visualize density
import matplotlib.pyplot as plt
env.plot_field(density, cmap='viridis')
plt.title("Place Field Density")
plt.colorbar(label="Number of overlapping fields")
plt.show()
```

### Border Cell Detection

```python
from neurospatial.metrics import border_score

# Compute firing rate
firing_rate = spikes_to_field(env, spike_times, times, positions)

# Compute border score
score = border_score(firing_rate, env, threshold=0.3, min_area=200.0)

# Classify as border cell
is_border_cell = score > 0.5  # Solstad et al. criterion
print(f"Border score: {score:.3f}")
print(f"Border cell: {is_border_cell}")

# Visualize with boundaries highlighted
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Firing rate map
env.plot_field(firing_rate, ax=ax1, cmap='hot')
ax1.set_title(f"Firing Rate (Border Score = {score:.2f})")

# Highlight boundaries
boundary_field = np.zeros(env.n_bins)
boundary_field[env.boundary_bins] = 1.0
env.plot_field(boundary_field, ax=ax2, cmap='binary', alpha=0.5)
ax2.set_title("Boundary Bins")

plt.tight_layout()
plt.show()
```

## See Also

- [Spike Train to Spatial Field Conversion](spike-field-primitives.md) - Computing firing rate maps
- [Signal Processing Primitives](signal-processing-primitives.md) - Smoothing and filtering fields
- [Spatial Analysis](spatial-analysis.md) - Core spatial operations
