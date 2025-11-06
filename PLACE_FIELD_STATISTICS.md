# Place Field Statistics: Comprehensive Analysis

## Overview

You're asking about the **full suite of place field metrics** used to characterize spatial representations. This document catalogs:

1. **Individual place field properties** (per neuron)
2. **Population-level metrics** (across neurons)
3. **Stability and remapping metrics** (across sessions/conditions)
4. **What primitives each requires**
5. **Architectural recommendation**

## Current State

Neurospatial provides **primitives** but **not** these domain-specific metrics. Users currently implement them manually (see `examples/08_complete_workflow.ipynb`).

## Individual Place Field Properties

These characterize individual neurons' spatial tuning:

### 1. Place Field Detection

**What it is**: Identify contiguous regions where firing rate exceeds threshold

**Algorithm**:
```python
def detect_place_fields(firing_rate, threshold=0.2):
    """
    Detect place fields as connected components above threshold.

    Parameters
    ----------
    firing_rate : array, shape (n_bins,)
        Firing rate map (Hz)
    threshold : float
        Threshold as fraction of peak rate (0-1)

    Returns
    -------
    fields : list of arrays
        Each array contains bin indices for one place field
    """
    peak_rate = firing_rate.max()
    active_mask = firing_rate >= (threshold * peak_rate)

    # Find connected components
    # Need: graph-based connected components on active bins
    fields = find_connected_components(active_mask, env.connectivity)

    return fields
```

**Primitives needed**:
- ✅ `firing_rate` - User computes from spike counts
- ✅ Connected components on graph - NetworkX or custom
- ✅ Threshold operation - NumPy

**Status**: All primitives exist (NetworkX provides connected components)

---

### 2. Place Field Size (Area)

**What it is**: Spatial extent of each place field

**Formula**: `area = Σ bin_sizes[i]` for bins in field

**Algorithm**:
```python
def field_size(field_bins, env):
    """
    Compute place field area.

    Parameters
    ----------
    field_bins : array of int
        Bin indices comprising the field
    env : Environment
        Spatial environment

    Returns
    -------
    size : float
        Field area in physical units (e.g., cm²)
    """
    bin_sizes = env.bin_sizes()  # Already exists!
    return bin_sizes[field_bins].sum()
```

**Primitives needed**:
- ✅ `env.bin_sizes()` - **EXISTS** (returns area of each bin)
- ✅ Array indexing and sum - NumPy

**Status**: All primitives exist ✅

---

### 3. Peak Firing Rate

**What it is**: Maximum firing rate within the field

**Formula**: `peak_rate = max(firing_rate[field_bins])`

**Primitives needed**:
- ✅ Firing rate map - User computes
- ✅ Max operation - NumPy

**Status**: Trivial, all primitives exist ✅

---

### 4. Spatial Information Content (Skaggs)

**What it is**: How much information each spike carries about position

**Formula**: `I = Σ p(x) * (r(x) / r_mean) * log2(r(x) / r_mean)`

**Status**: Already discussed in DOMAIN_SPECIFIC_METRICS.md - all primitives exist ✅

---

### 5. Place Field Centroid

**What it is**: Center of mass of the field

**Formula**: `centroid = Σ (r_i * pos_i) / Σ r_i`

**Algorithm**:
```python
def field_centroid(firing_rate, field_bins, env):
    """
    Compute place field center of mass.

    Parameters
    ----------
    firing_rate : array, shape (n_bins,)
        Firing rate map
    field_bins : array of int
        Bin indices in field
    env : Environment
        Spatial environment

    Returns
    -------
    centroid : array, shape (n_dims,)
        Weighted center of mass in physical coordinates
    """
    rates = firing_rate[field_bins]
    positions = env.bin_centers[field_bins]

    # Weighted average
    centroid = (rates[:, np.newaxis] * positions).sum(axis=0) / rates.sum()
    return centroid
```

**Primitives needed**:
- ✅ Firing rate map - User computes
- ✅ `env.bin_centers` - EXISTS
- ✅ Weighted average - NumPy

**Status**: All primitives exist ✅

---

### 6. In-Field vs Out-of-Field Firing

**What it is**: Ratio of mean firing rate inside vs outside field

**Formula**: `ratio = mean(r_in) / mean(r_out)`

**Primitives needed**:
- ✅ Firing rate map - User computes
- ✅ Boolean indexing - NumPy
- ✅ Mean calculation - NumPy

**Status**: All primitives exist ✅

---

### 7. Place Field Stability

**What it is**: Consistency of firing across time periods

**Formula**: `stability = corr(rate_map_1, rate_map_2)`

**Algorithm**:
```python
def field_stability(rate_map_1, rate_map_2):
    """
    Compute spatial correlation between two rate maps.

    Parameters
    ----------
    rate_map_1, rate_map_2 : arrays, shape (n_bins,)
        Firing rate maps from different time periods

    Returns
    -------
    correlation : float
        Pearson correlation coefficient
    """
    valid = (rate_map_1 > 0) | (rate_map_2 > 0)
    return np.corrcoef(rate_map_1[valid], rate_map_2[valid])[0, 1]
```

**Primitives needed**:
- ✅ Two rate maps - User computes
- ✅ Correlation - NumPy

**Status**: All primitives exist ✅

---

## Population-Level Metrics

These characterize the entire population's spatial representation:

### 8. Percentage of Environment Covered

**What it is**: Fraction of space represented by at least one place field

**Formula**: `coverage = |union(all_fields)| / n_bins`

**Algorithm**:
```python
def population_coverage(all_place_fields, n_bins):
    """
    Compute fraction of environment covered by place fields.

    Parameters
    ----------
    all_place_fields : list of lists of arrays
        For each neuron, list of place fields (each field = array of bins)
    n_bins : int
        Total number of bins in environment

    Returns
    -------
    coverage : float
        Fraction of bins covered (0-1)
    """
    covered_bins = set()
    for neuron_fields in all_place_fields:
        for field_bins in neuron_fields:
            covered_bins.update(field_bins)

    return len(covered_bins) / n_bins
```

**Primitives needed**:
- ✅ Place field detection (already covered)
- ✅ Set operations - Python sets
- ✅ Division - NumPy

**Status**: All primitives exist ✅

---

### 9. Place Field Density

**What it is**: Number of place fields per unit area

**Formula**: `density = n_fields / total_area`

Or as spatial map: `density(x) = count(fields containing x)`

**Algorithm**:
```python
def field_density_map(all_place_fields, n_bins):
    """
    Compute how many fields overlap at each location.

    Parameters
    ----------
    all_place_fields : list of lists of arrays
        Place fields for all neurons
    n_bins : int
        Total bins

    Returns
    -------
    density : array, shape (n_bins,)
        Number of place fields at each bin
    """
    density = np.zeros(n_bins)

    for neuron_fields in all_place_fields:
        for field_bins in neuron_fields:
            density[field_bins] += 1

    return density
```

**Primitives needed**:
- ✅ Place field detection
- ✅ Array indexing and increment - NumPy

**Status**: All primitives exist ✅

---

### 10. Number of Active Cells/Fields

**What it is**: Count of neurons with significant spatial tuning

**Algorithm**:
```python
def count_place_cells(spatial_information, threshold=0.5):
    """
    Count place cells based on Skaggs information threshold.

    Parameters
    ----------
    spatial_information : dict or array
        Skaggs information per neuron
    threshold : float
        Minimum information content (bits/spike)

    Returns
    -------
    n_place_cells : int
        Number of neurons exceeding threshold
    """
    return np.sum(np.array(list(spatial_information.values())) > threshold)
```

**Primitives needed**:
- ✅ Skaggs information (already covered)
- ✅ Threshold and count - NumPy

**Status**: All primitives exist ✅

---

### 11. Place Field Overlap

**What it is**: Degree to which fields from different neurons overlap

**Metrics**:
- **Pairwise overlap coefficient**: `|field_i ∩ field_j| / min(|field_i|, |field_j|)`
- **Population overlap distribution**: Distribution of overlap values

**Algorithm**:
```python
def field_overlap(field_bins_i, field_bins_j):
    """
    Compute overlap between two place fields.

    Parameters
    ----------
    field_bins_i, field_bins_j : arrays
        Bin indices for two fields

    Returns
    -------
    overlap : float
        Overlap coefficient (0-1)
    """
    set_i = set(field_bins_i)
    set_j = set(field_bins_j)

    intersection = len(set_i & set_j)
    smaller_field = min(len(set_i), len(set_j))

    return intersection / smaller_field if smaller_field > 0 else 0.0

def population_overlap_distribution(all_place_fields):
    """
    Compute distribution of overlap coefficients.

    Returns
    -------
    overlaps : array
        All pairwise overlap values
    """
    overlaps = []

    for i in range(len(all_place_fields)):
        for j in range(i + 1, len(all_place_fields)):
            for field_i in all_place_fields[i]:
                for field_j in all_place_fields[j]:
                    overlap = field_overlap(field_i, field_j)
                    overlaps.append(overlap)

    return np.array(overlaps)
```

**Primitives needed**:
- ✅ Place field detection
- ✅ Set operations - Python sets
- ✅ Min/division - NumPy

**Status**: All primitives exist ✅

---

## Remapping and Stability Metrics

These characterize how representations change across conditions:

### 12. Spatial Correlation Between Rate Maps

**What it is**: Correlation between firing rate maps across sessions/conditions

**Use cases**:
- Rate remapping (correlation high, fields shift slightly)
- Global remapping (correlation low, completely different fields)

**Algorithm**:
```python
def rate_map_correlation(rate_map_1, rate_map_2, *, method='pearson'):
    """
    Compute correlation between two rate maps.

    Parameters
    ----------
    rate_map_1, rate_map_2 : arrays, shape (n_bins,)
        Firing rate maps from different sessions
    method : {'pearson', 'spearman'}
        Correlation method

    Returns
    -------
    correlation : float
        Correlation coefficient (-1 to 1)
    """
    # Only bins active in at least one session
    valid = (rate_map_1 > 0) | (rate_map_2 > 0)

    if method == 'pearson':
        return np.corrcoef(rate_map_1[valid], rate_map_2[valid])[0, 1]
    elif method == 'spearman':
        from scipy.stats import spearmanr
        return spearmanr(rate_map_1[valid], rate_map_2[valid])[0]
```

**Primitives needed**:
- ✅ Rate maps from both sessions - User computes
- ✅ Correlation - NumPy/SciPy

**Status**: All primitives exist ✅

---

### 13. Distance Between Field Centroids

**What it is**: How far place field centers shift between conditions

**Use cases**:
- Quantify rate remapping (same field, shifted position)
- Detect global remapping (large centroid shifts)

**Algorithm**:
```python
def centroid_shift(rate_map_1, rate_map_2, field_bins, env):
    """
    Compute distance between place field centroids across sessions.

    Parameters
    ----------
    rate_map_1, rate_map_2 : arrays, shape (n_bins,)
        Rate maps from sessions 1 and 2
    field_bins : array
        Bin indices defining the field region
    env : Environment
        Spatial environment

    Returns
    -------
    shift_distance : float
        Euclidean distance between centroids (physical units)
    """
    centroid_1 = field_centroid(rate_map_1, field_bins, env)
    centroid_2 = field_centroid(rate_map_2, field_bins, env)

    return np.linalg.norm(centroid_1 - centroid_2)
```

**Primitives needed**:
- ✅ Field centroid calculation (already covered)
- ✅ Euclidean distance - NumPy

**Status**: All primitives exist ✅

---

### 14. Population Vector Correlation

**What it is**: Correlation between population activity patterns at different locations

**Use case**: How similar is the population code at location A vs location B?

**Algorithm** (from examples/08_complete_workflow.ipynb):
```python
def population_vector_correlation(population_matrix):
    """
    Compute correlation between population vectors at all bin pairs.

    Parameters
    ----------
    population_matrix : array, shape (n_neurons, n_bins)
        Firing rates for all neurons at all bins

    Returns
    -------
    correlation_matrix : array, shape (n_bins, n_bins)
        Correlation between population vectors
    """
    n_bins = population_matrix.shape[1]
    correlation_matrix = np.zeros((n_bins, n_bins))

    for i in range(n_bins):
        for j in range(i, n_bins):
            pop_vec_i = population_matrix[:, i]
            pop_vec_j = population_matrix[:, j]

            # Only compute if both have activity
            if (pop_vec_i > 0).any() and (pop_vec_j > 0).any():
                corr = np.corrcoef(pop_vec_i, pop_vec_j)[0, 1]
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr

    return correlation_matrix
```

**Primitives needed**:
- ✅ Population matrix (n_neurons × n_bins) - User computes
- ✅ Pairwise correlation - NumPy

**Status**: All primitives exist ✅

---

## Summary Table

| Metric | Category | Primitives Needed | All Exist? | Complexity |
|--------|----------|-------------------|------------|------------|
| Place field detection | Individual | Connected components | ✅ Yes | Medium |
| Field size | Individual | bin_sizes, sum | ✅ Yes | Trivial |
| Peak rate | Individual | max | ✅ Yes | Trivial |
| Skaggs information | Individual | occupancy, log | ✅ Yes | Easy |
| Field centroid | Individual | bin_centers, weighted avg | ✅ Yes | Easy |
| In-field/out-field ratio | Individual | boolean indexing, mean | ✅ Yes | Trivial |
| Field stability | Individual | correlation | ✅ Yes | Trivial |
| Population coverage | Population | set union | ✅ Yes | Easy |
| Field density | Population | counting | ✅ Yes | Easy |
| Active cells count | Population | threshold, sum | ✅ Yes | Trivial |
| Field overlap | Population | set intersection | ✅ Yes | Easy |
| Rate map correlation | Remapping | correlation | ✅ Yes | Trivial |
| Centroid shift | Remapping | norm | ✅ Yes | Easy |
| Population vector corr | Remapping | correlation | ✅ Yes | Medium |

**Key finding**: **ALL primitives exist!** These are pure convenience wrappers.

---

## Architectural Recommendation

### These Should Go in `neurospatial.metrics.place_fields` Module

**Structure**:
```python
neurospatial/
  metrics/
    __init__.py
    place_fields.py      # Individual field properties
    population.py        # Population-level metrics
    remapping.py         # Stability and remapping
    grid_cells.py        # Grid score, spatial autocorrelation (needs primitives!)
    boundary_cells.py    # Border score, head direction
```

**Example API**:
```python
from neurospatial.metrics.place_fields import (
    detect_place_fields,
    field_size,
    field_centroid,
    skaggs_information,
    field_stability,
)

from neurospatial.metrics.population import (
    population_coverage,
    field_density_map,
    count_place_cells,
    field_overlap,
    population_vector_correlation,
)

from neurospatial.metrics.remapping import (
    rate_map_correlation,
    centroid_shift,
)

# Usage
firing_rate = spike_counts / env.occupancy(times, positions)
fields = detect_place_fields(firing_rate, env, threshold=0.2)

for field_bins in fields:
    size = field_size(field_bins, env)
    centroid = field_centroid(firing_rate, field_bins, env)
    print(f"Field: {len(field_bins)} bins, {size:.1f} cm², center={centroid}")
```

---

## Contrast with Grid Cell Metrics

**Key difference**: Place field metrics use **existing primitives**, but grid cell metrics need **new primitives**:

| Metric | Needs New Primitives? |
|--------|-----------------------|
| Place field size, centroid, peak | ❌ No - use NumPy/existing |
| Skaggs information | ❌ No - use occupancy + NumPy |
| Population coverage/overlap | ❌ No - use set operations |
| **Grid score** | ✅ **YES** - needs spatial_autocorrelation |
| **Coherence** | ✅ **YES** - needs neighbor_reduce |

This validates the architecture:
1. **Core package**: Implement missing primitives (spatial_autocorrelation, neighbor_reduce, gradient, etc.)
2. **Metrics module**: After primitives exist, add convenience wrappers for standard analyses

---

## Implementation Priority

### Phase 1: Core Primitives (HIGH PRIORITY)

These **enable** metrics that don't currently exist:
1. `spatial_autocorrelation` → Enables grid score
2. `neighbor_reduce` → Enables coherence
3. `gradient`, `divergence` → Enables general differential operators

### Phase 2: Metrics Module (MEDIUM PRIORITY)

After primitives exist, add convenience wrappers:
1. `place_fields.py` - Individual field properties (all primitives exist)
2. `population.py` - Population metrics (all primitives exist)
3. `grid_cells.py` - Grid score (needs spatial_autocorrelation from Phase 1)
4. `remapping.py` - Stability metrics (all primitives exist)

---

## Documentation Standards

Each metric function should include:

**1. Clear scientific citation**:
```python
"""
References
----------
.. [1] O'Keefe, J., & Dostrovsky, J. (1971). The hippocampus as a spatial map.
       Brain Research, 34(1), 171-175.
.. [2] Muller, R. U., & Kubie, J. L. (1989). The firing of hippocampal place
       cells predicts the future position of freely moving rats.
       Journal of Neuroscience, 9(12), 4101-4110.
"""
```

**2. Parameter assumptions**:
```python
"""
Parameters
----------
firing_rate : array, shape (n_bins,)
    Firing rate map in Hz. Should be smoothed (recommended: 5 cm bandwidth)
    and occupancy-normalized before calling.
threshold : float, default=0.2
    Detection threshold as fraction of peak rate (0-1).
    Typical values: 0.1-0.3 depending on data quality.
"""
```

**3. Expected value ranges**:
```python
"""
Returns
-------
size : float
    Place field area in physical units (e.g., cm²).
    Typical rodent place fields: 200-800 cm² in open field.
"""
```

---

## Comparison: Place Fields vs Grid Cells

| Analysis | Place Fields | Grid Cells |
|----------|--------------|------------|
| **Metrics** | Size, peak rate, Skaggs info, centroid | Grid score, spatial periodicity |
| **Primitives needed** | All exist ✅ | **Spatial autocorrelation missing** ❌ |
| **Implementation** | Straightforward | Requires new primitives first |
| **Complexity** | Low-medium | High (requires graph signal processing) |
| **Priority** | High (standard analyses) | High (Nobel Prize discovery) |

**Key takeaway**: Place field metrics are **ready to implement now**. Grid cell metrics **require primitives first**.

---

## References

Standard papers for place field analyses:

1. **O'Keefe & Dostrovsky (1971)** - Original place cell discovery
2. **Muller et al. (1987)** - Place field properties in open field
3. **Skaggs et al. (1993, 1996)** - Spatial information and sparsity
4. **Wilson & McNaughton (1993)** - Population dynamics
5. **Leutgeb et al. (2005)** - Rate vs global remapping
6. **Muller & Kubie (1989)** - Spatial coherence
7. **Hafting et al. (2005)** - Grid cells (need autocorrelation!)

---

## Conclusion

**All place field metrics can be implemented using existing primitives** - they're pure convenience wrappers over NumPy, set operations, and neurospatial's existing API.

**Recommendation**: Create `neurospatial.metrics` module with these standard analyses to:
- Lower barrier to entry for new users
- Standardize implementations across labs
- Reduce code duplication
- Document canonical formulas with citations

But **implement core primitives first** (spatial_autocorrelation, neighbor_reduce, etc.) because those enable metrics that currently **cannot** be computed (grid score, coherence).
