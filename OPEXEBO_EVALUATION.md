# Evaluation: opexebo Package Analysis

## Overview

**opexebo** (https://github.com/simon-ball/opexebo) is a Python electrophysiology analysis library developed by the **Moser group at the Kavli Institute** (Nobel Prize 2014 for grid cell discovery). It provides 19 analysis modules for spatial neuroscience, combining translated MATLAB code from the Behavioural Neurology Toolbox with new Python implementations.

**Key finding**: This is THE authoritative implementation source for grid cell and spatial analysis metrics.

---

## What opexebo Provides

### Spatial Analysis Metrics (19 modules)

| Category | Metrics | Status |
|----------|---------|--------|
| **Place cells** | Rate maps, place field detection, coherence, spatial info | ✅ Complete |
| **Grid cells** | Grid score, spatial autocorrelation | ✅ Complete |
| **Boundary cells** | Border score, border coverage | ✅ Complete |
| **Speed cells** | Speed score, speed occupancy | ✅ Complete |
| **Head direction** | Tuning curves, tuning stats | ✅ Complete |
| **Population** | Population vector correlation | ✅ Complete |
| **Basic** | Occupancy, rate maps, angular occupancy | ✅ Complete |
| **Theta modulation** | Theta modulation index | ✅ Complete |

---

## Detailed Analysis of Key Implementations

### 1. Spatial Autocorrelation

**File**: `opexebo/analysis/autocorrelation.py`

**Algorithm**:
```python
def autocorrelation(firing_map):
    # Replace NaNs with zeros
    firing_map = np.nan_to_num(firing_map)

    # Normalized cross-correlation (via FFT)
    acorr = normxcorr2_general(firing_map)

    # Crop edges (overlap_amount=0.8)
    # Reduces boundary artifacts
    return cropped_acorr
```

**Approach**:
- Uses normalized 2D cross-correlation via FFT
- Assumes **regular rectangular grids only**
- Replaces NaNs with zeros
- Crops 20% of edges to reduce boundary artifacts

**Strengths**:
- ✅ Fast (FFT-based)
- ✅ Standard approach in the field
- ✅ From Nobel Prize-winning lab

**Limitations**:
- ❌ **Only works on regular grids** (not irregular graphs)
- ❌ NaN handling is crude (zeros may bias correlation)
- ❌ Edge cropping loses information

**Relevance to neurospatial**:
- This is exactly "Option A: Interpolation to Regular Grid" from our implementation plan
- **Gap remains**: Irregular graph support is unique to neurospatial

---

### 2. Grid Score

**File**: `opexebo/analysis/grid_score.py`

**Algorithm** (Sargolini et al. 2006):
```python
def grid_score(autocorr_map):
    # Automatically detect central field radius

    # For expanding radii from center:
    for radius in range(min_radius, max_radius):
        # Extract annular ring (donut shape)
        ring_pixels = get_annulus(autocorr_map, radius, width=1)

        # Rotate autocorr at specific angles
        for angle in [30, 60, 90, 120, 150]:
            rotated = rotate(autocorr_map, angle)
            rotated_ring = get_annulus(rotated, radius, width=1)

            # Compute Pearson correlation between rings
            corr[angle] = pearson(ring_pixels, rotated_ring)

        # Grid score for this radius
        gs[radius] = min(corr[60], corr[120]) - max(corr[30], corr[90], corr[150])

    # Smooth with sliding window (3 radii)
    # Return maximum grid score
    return max(smooth(gs, window=3))
```

**Key features**:
- Uses annular rings (not full map) to reduce noise
- Sliding window smoothing for robustness
- Automatic field radius detection
- Returns values in range [-2, 2], typical good grids ~1.3

**Strengths**:
- ✅ Sophisticated (annular rings, sliding window)
- ✅ Automatic parameter detection
- ✅ Matches published methods exactly
- ✅ From the lab that discovered grid cells!

**Limitations**:
- ❌ Requires rectangular autocorrelation map
- ❌ No support for irregular grids

**Relevance to neurospatial**:
- We should adopt this exact algorithm
- Use their automatic radius detection
- Use annular rings + sliding window approach

---

### 3. Rate Map Coherence

**File**: `opexebo/analysis/rate_map_coherence.py`

**Algorithm** (Muller & Kubie 1989):
```python
def rate_map_coherence(rate_map):
    # 3x3 convolution kernel (mean of 8 neighbors, exclude center)
    kernel = np.array([
        [0.125, 0.125, 0.125],
        [0.125, 0,     0.125],
        [0.125, 0.125, 0.125]
    ])

    # Convolve to get neighbor averages
    neighbor_avg = convolve2d(rate_map, kernel, mode='fill', fillvalue=0)

    # Replace NaNs with zeros
    neighbor_avg = np.nan_to_num(neighbor_avg)
    rate_map = np.nan_to_num(rate_map)

    # Pearson correlation
    coherence = np.corrcoef(
        rate_map.flatten(),
        neighbor_avg.flatten()
    )[0, 1]

    return coherence
```

**Approach**:
- Hardcoded 3x3 kernel (8 neighbors, equal weights)
- Zero-padding at boundaries
- Simple Pearson correlation

**Strengths**:
- ✅ Simple, fast
- ✅ Matches published method

**Limitations**:
- ❌ **Hardcoded for regular grids** (4 or 8 connectivity)
- ❌ Zero-padding may bias boundary bins
- ❌ Not extensible to irregular graphs

**Relevance to neurospatial**:
- This is exactly what `neighbor_reduce` generalizes!
- **Gap**: Our `neighbor_reduce` works on **any graph**, not just regular grids
- **Gap**: We support weighted aggregation, arbitrary operations (mean/sum/max/std)

---

### 4. Place Field Detection

**File**: `opexebo/analysis/place_field.py`

**Algorithm**:
```python
def place_field_detection(rate_map):
    # Iterative adaptive thresholding
    for threshold_pct in np.arange(0.96, 0.2, -0.02):
        threshold_val = peak_rate * threshold_pct

        # Label connected components (4-connectivity)
        labeled = morphology.label(rate_map > threshold_val, connectivity=1)

        # Extract region properties
        regions = regionprops(labeled)

        # Check stability (area doesn't change too much)
        # Check no holes (via morphological filling)
        # Check doesn't encompass other peaks

        if stable:
            break

    # Compute properties for each field
    for region in regions:
        field_props = {
            'area': region.area,
            'centroid': region.centroid,
            'bbox': region.bbox,
            'mean_rate': np.nanmean(rate_map[region.coords]),
            'peak_rate': np.nanmax(rate_map[region.coords]),
        }

    return field_props
```

**Key features**:
- Adaptive thresholding with iterative refinement
- 4-connectivity (orthogonal only, not diagonal)
- Validation: no holes, stable boundaries
- Uses scikit-image's `morphology.label` and `regionprops`

**Strengths**:
- ✅ Sophisticated (adaptive, validated)
- ✅ Leverages scikit-image infrastructure
- ✅ Robust to noise

**Limitations**:
- ❌ **Regular grids only** (relies on image processing)
- ❌ 4-connectivity assumption may not match environment connectivity

**Relevance to neurospatial**:
- We can use NetworkX's `connected_components` on graph
- Should adopt adaptive thresholding + validation approach
- **Gap**: Graph-based detection works on **any connectivity**

---

### 5. Rate Map Statistics

**File**: `opexebo/analysis/rate_map_stats.py`

**Metrics computed**:
```python
def rate_map_stats(rate_map, occupancy_map):
    mean_rate = np.nanmean(rate_map)
    peak_rate = np.nanmax(rate_map)

    # Sparsity (Skaggs et al.)
    sparsity = mean_rate**2 / np.nanmean(rate_map**2)

    # Selectivity
    selectivity = peak_rate / mean_rate

    # Spatial information rate (bits/sec)
    position_pdf = occupancy_map / occupancy_map.sum()
    valid = (rate_map / mean_rate) >= 1
    inf_rate = np.sum(
        position_pdf[valid] * rate_map[valid] * np.log2(rate_map[valid] / mean_rate)
    )

    # Spatial information content (bits/spike)
    inf_content = inf_rate / mean_rate

    return {
        'spatial_information_rate': inf_rate,
        'spatial_information_content': inf_content,
        'sparsity': sparsity,
        'selectivity': selectivity,
        'peak_rate': peak_rate,
        'mean_rate': mean_rate,
    }
```

**Strengths**:
- ✅ Matches Skaggs formulas exactly
- ✅ Comprehensive set of standard metrics
- ✅ Clean API

**Limitations**:
- ❌ Assumes regular grid (but this is less critical)

**Relevance to neurospatial**:
- We should provide **identical metrics** in `neurospatial.metrics.place_fields`
- Formula compatibility is essential for reproducibility

---

### 6. Border Score

**File**: `opexebo/analysis/border_score.py`

**Algorithm** (Solstad et al. 2008):
```python
def border_score(rate_map, arena_shape):
    # Find largest field with wall coverage
    field = find_field_with_max_wall_coverage(rate_map)

    # Coverage: fraction of single wall touched by field
    coverage = compute_wall_coverage(field, arena_shape)

    # Weighted firing distance from border
    # Uses taxicab distance, normalized by arena size
    firing_dist = compute_taxicab_distance_from_edge(rate_map, field)
    wfd = np.average(firing_dist, weights=rate_map[field])

    # Border score
    score = (coverage - wfd) / (coverage + wfd)

    return score
```

**Key features**:
- Different handling for circular vs rectangular arenas
- Uses taxicab (Manhattan) distance
- Only evaluates single field with greatest wall coverage
- Theoretical max ~0.91 (not 1.0 due to binning)

**Strengths**:
- ✅ Matches published method
- ✅ Handles common arena shapes

**Limitations**:
- ❌ Hardcoded for rectangular/circular arenas
- ❌ Doesn't generalize to arbitrary polygon environments

**Relevance to neurospatial**:
- We have `env.boundary_bins()` already ✅
- Can generalize to arbitrary polygon environments
- **Advantage**: neurospatial supports any shape via Shapely

---

## Comparison: opexebo vs neurospatial

### What opexebo Does Well

| Feature | opexebo | neurospatial (current) |
|---------|---------|------------------------|
| **Place field metrics** | ✅ Complete | ❌ Missing |
| **Grid score** | ✅ Complete | ❌ Missing |
| **Border score** | ✅ Complete | ✅ Have primitives (`boundary_bins`) |
| **Spatial autocorrelation** | ✅ Regular grids | ❌ Missing |
| **Rate map coherence** | ✅ Regular grids | ❌ Missing |
| **Authority** | ✅ Nobel Prize lab | N/A |

### What neurospatial Does Well

| Feature | opexebo | neurospatial |
|---------|---------|--------------|
| **Irregular graphs** | ❌ No support | ✅ Core feature |
| **Arbitrary connectivity** | ❌ Grid-only | ✅ Any graph |
| **Custom layouts** | ❌ Grid-only | ✅ 7+ layout engines |
| **Differential operators** | ❌ Not provided | 🔶 Proposed |
| **Graph signal processing** | ❌ Not provided | 🔶 Proposed |
| **RL/replay primitives** | ❌ Not provided | 🔶 Proposed |
| **Flexible primitives** | ❌ Hardcoded | 🔶 `neighbor_reduce`, etc. |
| **Shapely integration** | ❌ No | ✅ Full support |

**Legend**: ✅ Exists, ❌ Missing, 🔶 Proposed in implementation plan

---

## Critical Insights

### 1. Regular Grids Only

**opexebo assumes rectangular grids throughout**:
- Autocorrelation: 2D FFT requires rectangular array
- Coherence: Hardcoded 3×3 convolution kernel
- Place field detection: Uses image morphology
- Grid score: Rotates rectangular arrays

**Neurospatial's unique value**: Support for **irregular graphs**

### 2. They Implement What We Proposed

All the metrics in our `DOMAIN_SPECIFIC_METRICS.md` exist in opexebo:
- ✅ Skaggs information (`rate_map_stats`)
- ✅ Sparsity (`rate_map_stats`)
- ✅ Coherence (`rate_map_coherence`)
- ✅ Grid score (`grid_score`)
- ✅ Border score (`border_score`)
- ✅ Place field detection (`place_field`)

**This validates our proposed `neurospatial.metrics` module!**

### 3. Our Primitives Enable Generalization

| opexebo Implementation | Hardcoded For | neurospatial Primitive | Works On |
|------------------------|---------------|------------------------|----------|
| 3×3 convolution (coherence) | Regular grids | `neighbor_reduce` | Any graph |
| 2D FFT autocorrelation | Rectangular arrays | `spatial_autocorrelation` | Irregular graphs (via interpolation) |
| Image morphology (place fields) | Regular grids | Graph connected components | Any graph |

**Our primitives are more general** - they work on any connectivity structure.

### 4. Algorithm Sophistication

opexebo's grid score is **more sophisticated than I initially proposed**:
- Uses annular rings (not full map)
- Sliding window smoothing (3 radii)
- Automatic radius detection

**We should adopt these refinements!**

---

## Gaps in opexebo That neurospatial Fills

### Missing from opexebo:

1. **Irregular graph support** ← Biggest gap
2. **Differential operators** (gradient, divergence, Laplacian)
3. **RL/replay primitives** (value iteration, successor representation)
4. **Flexible spatial primitives** (neighbor_reduce, propagate, accumulate)
5. **Custom layout engines** (hexagonal, 1D tracks, masked grids, etc.)
6. **Arbitrary polygon environments** (Shapely integration)
7. **Graph signal processing** (weighted differential operator)
8. **Distance fields** (neurospatial has this, opexebo doesn't)

### Redundant with opexebo:

**None!** neurospatial focuses on:
- Environment discretization
- Graph construction
- Spatial queries
- Trajectory operations

opexebo focuses on:
- Neuroscience metrics (place fields, grid cells, etc.)
- Rate map analysis

**They are complementary, not competitive.**

---

## Recommendations for neurospatial Implementation

### 1. Adopt opexebo's Algorithms

For metrics that overlap, **use opexebo's exact algorithms**:

✅ **Grid score**: Annular rings + sliding window
✅ **Rate map stats**: Exact Skaggs formulas
✅ **Place field detection**: Adaptive thresholding + validation
✅ **Border score**: Coverage vs weighted firing distance

**Rationale**: Authority (Nobel Prize lab), field-tested, matches publications

### 2. Extend to Irregular Graphs

**Key innovation**: Make these algorithms work on **any connectivity structure**

Example: Coherence
```python
# opexebo (hardcoded 3x3)
kernel = [[0.125, 0.125, 0.125],
          [0.125, 0,     0.125],
          [0.125, 0.125, 0.125]]
neighbor_avg = convolve2d(rate_map, kernel)

# neurospatial (any graph)
def coherence(firing_rate, env):
    neighbor_avg = neighbor_reduce(firing_rate, env, op='mean')
    return np.corrcoef(firing_rate, neighbor_avg)[0, 1]
```

### 3. Spatial Autocorrelation Strategy

opexebo's approach (2D FFT) is **fast and standard**, but **limited to regular grids**.

**Proposed approach for neurospatial**:

```python
def spatial_autocorrelation(field, env, *, method='auto'):
    """
    Compute spatial autocorrelation.

    Parameters
    ----------
    method : {'auto', 'fft', 'graph'}
        - 'auto': Use FFT for regular grids, graph-based for irregular
        - 'fft': Interpolate to regular grid, use FFT (opexebo approach)
        - 'graph': Graph-based correlation (slower but exact)
    """
    if method == 'auto':
        if env.layout._layout_type_tag == 'RegularGridLayout':
            method = 'fft'
        else:
            method = 'graph'

    if method == 'fft':
        # Interpolate to regular grid
        # Use opexebo's FFT approach
        return fft_autocorrelation(field, env)

    elif method == 'graph':
        # Graph-based approach
        # Compute correlation at different distances
        return graph_autocorrelation(field, env)
```

**Benefit**: Fast for regular grids (FFT), works for irregular graphs (graph-based)

### 4. Documentation Strategy

**Cross-reference opexebo**:
```python
def skaggs_information(firing_rate, occupancy):
    """
    Compute Skaggs spatial information content (bits/spike).

    This implementation matches the algorithm used by opexebo
    (Moser group, Kavli Institute), which is the standard in the field.

    References
    ----------
    .. [1] Skaggs et al. (1993). An Information-Theoretic Approach to
           Deciphering the Hippocampal Code. NIPS.
    .. [2] opexebo.analysis.rate_map_stats:
           https://github.com/simon-ball/opexebo

    See Also
    --------
    opexebo.analysis.rate_map_stats : Reference implementation
    """
```

### 5. Testing Strategy

**Validate against opexebo outputs**:
```python
def test_grid_score_matches_opexebo():
    """Grid score should match opexebo for regular grids."""
    # Create synthetic grid cell pattern
    rate_map = create_hexagonal_pattern()

    # Compute with neurospatial
    env = Environment.from_samples(positions, bin_size=2.5)
    autocorr = spatial_autocorrelation(rate_map, env)
    gs_neurospatial = grid_score(autocorr, env)

    # Compute with opexebo (if installed)
    try:
        import opexebo
        gs_opexebo = opexebo.analysis.grid_score(rate_map)
        np.testing.assert_allclose(gs_neurospatial, gs_opexebo, rtol=0.01)
    except ImportError:
        pytest.skip("opexebo not installed")
```

### 6. Integration Path

Users can combine neurospatial + opexebo:

```python
import numpy as np
from neurospatial import Environment
import opexebo

# neurospatial: Environment discretization
env = Environment.from_samples(positions, bin_size=2.5)

# neurospatial: Occupancy and bin sequence
occupancy = env.occupancy(times, positions)
bin_sequence = env.bin_sequence(times, positions)

# Map to rate map (regular grid required for opexebo)
if env.layout._layout_type_tag == 'RegularGridLayout':
    # Reshape to 2D grid
    rate_map_2d = firing_rate.reshape(env.layout.grid_shape)

    # opexebo: Compute metrics
    stats = opexebo.analysis.rate_map_stats(rate_map_2d, occupancy)
    grid_score = opexebo.analysis.grid_score(rate_map_2d)
else:
    # Use neurospatial metrics (work on irregular graphs)
    from neurospatial.metrics import skaggs_information, coherence
    spatial_info = skaggs_information(firing_rate, occupancy)
    coh = coherence(firing_rate, env)
```

---

## Updated Implementation Plan Priorities

### Changes Based on opexebo Analysis

#### Phase 2.2: spatial_autocorrelation (UPDATED)

**Original plan**: 4 weeks, high risk

**Updated strategy**:
1. **Week 5-6**: Implement FFT-based approach (opexebo method)
   - For regular grids: reshape to 2D, use FFT
   - Fast, validated, matches field standard
   - **Risk: LOW** (known algorithm)

2. **Week 7-8**: Extend to irregular graphs (optional)
   - Graph-based correlation approach
   - Slower but works on any connectivity
   - **Risk: MEDIUM** (novel algorithm)

**Reduced risk**: FFT approach is well-understood and tested

#### Phase 4.3: Grid Cell Metrics (UPDATED)

**Use opexebo's exact algorithm**:
- Annular rings for correlation
- Sliding window smoothing (3 radii)
- Automatic radius detection
- Return same value ranges

**Effort**: Reduced from 5 days → **3 days** (adopt known algorithm)

#### New Phase 4.5: Algorithm Validation

**Test against opexebo outputs** for regular grids:
- Grid score should match within 1%
- Coherence should match exactly
- Spatial information should match exactly

**Effort**: 2 days

---

## Strategic Positioning

### neurospatial vs opexebo

**They are complementary, not competitive:**

| Package | Focus | Strength |
|---------|-------|----------|
| **opexebo** | Neuroscience metrics | Authority (Moser lab), validated algorithms |
| **neurospatial** | Spatial discretization & graphs | Irregular graphs, flexible primitives |

**Ideal workflow**:
1. **neurospatial**: Environment discretization, occupancy, trajectories
2. **opexebo**: Metrics for regular grids (when applicable)
3. **neurospatial.metrics**: Metrics for irregular graphs (when needed)

### Unique Value Proposition

**neurospatial fills gaps that opexebo cannot**:
1. ✅ Irregular graph support (complex mazes, 1D tracks)
2. ✅ Custom connectivity structures
3. ✅ Differential operators on graphs
4. ✅ RL/replay primitives
5. ✅ Flexible spatial primitives (neighbor_reduce, etc.)

**opexebo provides validated neuroscience metrics**:
- Authority from Nobel Prize-winning lab
- Field-tested algorithms
- Standard implementations

**Our strategy**:
- **Core primitives**: Work on any graph (unique to neurospatial)
- **Metrics module**: Match opexebo algorithms for regular grids
- **Documentation**: Cross-reference opexebo as gold standard
- **Testing**: Validate against opexebo outputs

---

## Conclusion

### Key Takeaways

1. **opexebo is authoritative** - Moser lab (Nobel Prize for grid cells)
2. **Our proposal is validated** - They implement all metrics we identified
3. **Unique value: irregular graphs** - opexebo is grid-only
4. **Adopt their algorithms** - Use exact formulas, annular rings, adaptive thresholding
5. **Reduced risk for autocorrelation** - FFT approach is well-tested
6. **Complementary packages** - Not competitive, can be used together

### Updated Risk Assessment

| Component | Original Risk | Updated Risk | Reason |
|-----------|---------------|--------------|--------|
| spatial_autocorrelation | HIGH | **MEDIUM** | Adopt opexebo's FFT approach |
| grid_score | Medium | **LOW** | Use exact opexebo algorithm |
| place_field_detection | Low | **LOW** | Validated approach available |
| neighbor_reduce | Low | **LOW** | Generalizes opexebo's convolution |

### Recommendation

**Proceed with implementation plan**, with these updates:

1. ✅ **Adopt opexebo algorithms** for overlapping metrics
2. ✅ **Use FFT autocorrelation** for regular grids (fast, validated)
3. ✅ **Extend to irregular graphs** (unique neurospatial capability)
4. ✅ **Cross-reference opexebo** in documentation
5. ✅ **Validate against opexebo** in test suite
6. ✅ **Reduced timeline**: 14 weeks instead of 16 (lower autocorrelation risk)

**Strategic positioning**: neurospatial + opexebo are complementary tools that together provide complete spatial neuroscience analysis for both regular and irregular environments.

---

## References

- **opexebo**: https://github.com/simon-ball/opexebo
- **Moser group**: Kavli Institute, Trondheim (Nobel Prize 2014)
- **Sargolini et al. (2006)**: Grid score algorithm
- **Muller & Kubie (1989)**: Spatial coherence
- **Solstad et al. (2008)**: Border score
- **Skaggs et al. (1993, 1996)**: Spatial information and sparsity
