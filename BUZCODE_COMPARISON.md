# Buzcode Comparison Analysis

## Overview

**buzcode** (https://github.com/buzsakilab/buzcode) is a MATLAB-based analysis framework from the **Buzsáki Lab** for electrophysiology and spatial navigation research. This document compares buzcode's implementations with opexebo (Python, Moser lab) and our proposed neurospatial primitives.

**Key finding**: buzcode provides **basic spatial analysis** but **lacks comprehensive metrics** that opexebo and neurospatial aim to provide.

---

## What buzcode Provides

### Package Overview

| Property | Details |
|----------|---------|
| **Language** | MATLAB (66.2%), Jupyter Notebooks (27.6%), C (3.7%) |
| **Lab** | Buzsáki Lab (NYU) - pioneering hippocampal research |
| **License** | GPL-3.0 |
| **Focus** | Internal lab tools, preprocessing pipelines, data structures |
| **Contributors** | 26 contributors, 2,175 commits |

### Analysis Modules

```
buzcode/analysis/
├── placeFields/              # Basic place field detection
├── positionDecoding/         # Bayesian decoding
├── spikeLFPcoupling/        # Phase-locking analysis
├── SharpWaveRipples/        # SWR detection
├── assemblies/              # Population dynamics
├── cellTypeClassification/  # Cell type ID
└── CrossFrequencyCoupling/  # CFC analysis
```

---

## Detailed Analysis: placeFields Module

### Available Functions (7 total)

| Function | Purpose |
|----------|---------|
| `bz_firingMap1D.m` | Compute 1D firing rate maps |
| `bz_findPlaceFields1D.m` | Detect place fields (1D) |
| `bz_getPlaceFields1D.m` | Extract place field properties |
| `bz_firingMapAvg.m` | Average firing maps across trials |
| `KalmanVel.m` | Kalman filter for velocity |
| `bz_findPlaceFieldsTemplate.m` | Template for custom detectors |
| `bz_getPlaceFields.m` | General place field retrieval |

**Key limitation**: Focus on **1D tracks** (linear mazes), limited 2D support

---

## Algorithm Analysis

### 1. Firing Map Computation (`bz_firingMap1D.m`)

**Algorithm**:
```matlab
% 1. Bin spikes by position
for each spike:
    find nearest position timestamp
    assign to closest spatial bin
    increment spike count

% 2. Compute occupancy
for each position sample:
    assign to nearest spatial bin
    increment occupancy count

% 3. Smooth
smoothed_spikes = smooth(spike_counts, tau)  % Box or Gaussian
smoothed_occupancy = medfilt1(occupancy, 5)  % Median filter

% 4. Compute rate
firing_rate = smoothed_spikes / (smoothed_occupancy * dt)
```

**Strengths**:
- ✅ Simple, understandable
- ✅ Handles multiple trials
- ✅ Provides both smoothed and unsmoothed versions

**Limitations**:
- ❌ **1D only** (no 2D or irregular environments)
- ❌ Median filter for occupancy (crude compared to Gaussian)
- ❌ No adaptive binning
- ❌ No morphological operations for sparse data

---

### 2. Place Field Detection (`bz_findPlaceFields1D.m`)

**Algorithm**:
```matlab
while fields remain:
    % Find peak
    [peak_rate, peak_loc] = max(firing_map)

    % Check threshold
    if peak_rate < minPeak:
        break

    % Define field boundaries
    field_bins = (firing_map >= threshold * peak_rate)

    % Validate
    if size_ok and not_at_edge:
        save_field(field_bins, peak_rate, peak_loc)

    % Remove detected field
    firing_map(field_bins) = NaN

    % Check for secondary fields
    if peak_rate < minPeak2nd * global_max:
        break
```

**Thresholds**:
- `threshold = 0.2` (20% of peak)
- `minPeak = 2.0` Hz (first field)
- `minPeak2nd = 0.60` (60% of global max for secondary fields)
- `minSize = 0.05` (5% of track)
- `maxSize = 0.60` (60% of track)

**Field Properties Computed**:
- `start`, `stop` - Boundaries
- `width` - Field size
- `peakFR` - Peak firing rate
- `peakLoc` - Peak location
- `COM` - Center of mass

**Strengths**:
- ✅ Iterative detection (finds multiple fields)
- ✅ Adaptive thresholds (different for 1st vs 2nd field)
- ✅ Size validation

**Limitations**:
- ❌ **No validation against opexebo/published methods**
- ❌ Hardcoded thresholds (no data-driven optimization)
- ❌ Simple peak-finding (not connected-component based)
- ❌ **Missing standard metrics**: spatial information, sparsity, coherence

---

## Comparison: buzcode vs opexebo vs neurospatial

### Spatial Metrics Coverage

| Metric | buzcode | opexebo | neurospatial (proposed) |
|--------|---------|---------|-------------------------|
| **Place field detection** | ✅ 1D only | ✅ 2D grids | ✅ Any graph |
| **Firing rate maps** | ✅ 1D only | ✅ 2D grids | ✅ Any graph |
| **Peak firing rate** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Field size** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Field centroid** | ✅ COM | ✅ Center of mass | ✅ Center of mass |
| **Skaggs information** | ❌ No | ✅ Yes | ✅ Yes |
| **Sparsity** | ❌ No | ✅ Yes | ✅ Yes |
| **Coherence** | ❌ No | ✅ Yes | ✅ Yes (generalizes to any graph) |
| **Grid score** | ❌ No | ✅ Yes | ✅ Yes |
| **Spatial autocorrelation** | ❌ No | ✅ Yes (2D grids) | ✅ Yes (any graph) |
| **Border score** | ❌ No | ✅ Yes | ✅ Yes |

**Summary**:
- **buzcode**: Basic place field detection, minimal metrics
- **opexebo**: Comprehensive metrics, 2D grids only
- **neurospatial**: Comprehensive metrics, any graph structure

---

### Environment Support

| Feature | buzcode | opexebo | neurospatial |
|---------|---------|---------|--------------|
| **1D tracks** | ✅ Primary focus | ⚠️ Via regular grids | ✅ GraphLayout |
| **2D open field** | ⚠️ Limited | ✅ Regular grids | ✅ Multiple layouts |
| **Circular arenas** | ❌ No | ⚠️ Hardcoded | ✅ Shapely polygons |
| **Complex mazes** | ❌ No | ❌ No | ✅ Any connectivity |
| **Irregular graphs** | ❌ No | ❌ No | ✅ **Core feature** |
| **Custom layouts** | ❌ No | ❌ No | ✅ 7+ engines |

---

### Bayesian Decoding

| Feature | buzcode | opexebo | neurospatial |
|---------|---------|---------|--------------|
| **Position decoding** | ✅ Multiple methods | ❌ No | 🔶 Future (via metrics) |
| **Rate-based** | ✅ Yes | N/A | N/A |
| **Phase-based** | ✅ Yes | N/A | N/A |
| **GLM-based** | ✅ Yes | N/A | N/A |
| **Max correlation** | ✅ Yes | N/A | N/A |

**Note**: buzcode excels at position decoding, which is outside opexebo's and neurospatial's scope.

---

## Key Differences

### 1. Dimensionality

**buzcode**:
- **Primary focus: 1D tracks** (linear mazes, W-tracks)
- Limited 2D support
- No irregular environments

**opexebo**:
- **Primary focus: 2D regular grids** (open field arenas)
- Assumes rectangular discretization
- Grid cells, place cells in open field

**neurospatial**:
- **Supports any dimensionality** (1D, 2D, N-D)
- **Any connectivity structure** (graphs, irregular meshes)
- Flexible layout engines

### 2. Language & Ecosystem

**buzcode**:
- MATLAB (66%)
- Internal lab tool
- MATLAB ecosystem (limited portability)

**opexebo**:
- Python (100%)
- Public research tool
- NumPy/SciPy ecosystem

**neurospatial**:
- Python (100%)
- Open-source library
- Scientific Python stack

### 3. Scope

**buzcode**:
- **Broad**: Preprocessing, LFP, spikes, decoding, assemblies, SWR detection
- **Shallow spatial metrics**: Basic place field detection
- Focus on **preprocessing pipelines** and **data structures**

**opexebo**:
- **Narrow focus**: Spatial neuroscience metrics
- **Deep spatial metrics**: Comprehensive place/grid/border cell analyses
- Authority from Nobel Prize-winning lab

**neurospatial**:
- **Focus**: Spatial discretization and graph operations
- **Provides primitives** for building metrics
- **Complements opexebo** (different level of abstraction)

---

## What buzcode Does Well

### 1. Bayesian Position Decoding ⭐

**Multiple decoding methods**:
- Rate-based Bayesian
- Phase-based (theta sequences)
- GLM-based
- Maximum correlation

**This is buzcode's strength** - not replicated in opexebo or neurospatial.

### 2. Comprehensive Electrophysiology Pipeline

**Full preprocessing stack**:
- Spike sorting integration
- LFP processing
- Artifact detection
- Data structure standardization

### 3. 1D Track Analysis

**Well-suited for**:
- Linear tracks
- T-mazes, W-mazes
- Theta sequences
- Place field detection on tracks

**Limitation**: Not generalizable to complex 2D environments

---

## What buzcode Lacks

### 1. Comprehensive Spatial Metrics ❌

**Missing from buzcode** (available in opexebo):
- Skaggs spatial information
- Sparsity
- Coherence
- Grid score
- Spatial autocorrelation
- Border score
- Selectivity
- Stability metrics

**Impact**: Users must implement these themselves or use other packages.

### 2. 2D Spatial Support ❌

**Limitations**:
- 1D-focused API
- Limited 2D open field support
- No support for circular arenas
- No polygon-bounded environments

**Impact**: Less useful for open field experiments (common in grid cell research).

### 3. Modern Spatial Primitives ❌

**Missing** (proposed for neurospatial):
- Differential operators (gradient, divergence)
- Spatial autocorrelation on graphs
- Graph-based neighbor operations
- RL/replay primitives
- Flexible spatial kernels

---

## Comparison Table: All Three Packages

| Feature | buzcode | opexebo | neurospatial (proposed) |
|---------|---------|---------|-------------------------|
| **Language** | MATLAB | Python | Python |
| **Authority** | Buzsáki Lab | Moser Lab (Nobel 2014) | N/A |
| **Primary Focus** | Preprocessing + 1D tracks | Spatial metrics | Spatial primitives |
| **Place fields** | ✅ 1D detection | ✅ 2D detection | ✅ Any graph |
| **Grid cells** | ❌ No | ✅ Yes | ✅ Yes |
| **Spatial info** | ❌ No | ✅ Yes | ✅ Yes |
| **Coherence** | ❌ No | ✅ Yes | ✅ Yes (generalizes) |
| **Border cells** | ❌ No | ✅ Yes | ✅ Yes |
| **Bayesian decoding** | ✅ Multiple methods | ❌ No | 🔶 Future |
| **1D tracks** | ✅ **Excellent** | ⚠️ Via grids | ✅ GraphLayout |
| **2D open field** | ⚠️ Limited | ✅ **Excellent** | ✅ Multiple layouts |
| **Irregular graphs** | ❌ No | ❌ No | ✅ **Core feature** |
| **Differential ops** | ❌ No | ❌ No | ✅ Yes |
| **LFP analysis** | ✅ Extensive | ❌ No | ❌ No |
| **SWR detection** | ✅ Yes | ❌ No | ❌ No |
| **Open source** | ✅ GPL-3.0 | ✅ Open | ✅ Open |

**Legend**: ✅ Complete, ⚠️ Limited, ❌ Missing, 🔶 Planned

---

## Strategic Positioning

### Three Complementary Tools

**buzcode** - Preprocessing & 1D Analysis:
- Full electrophysiology pipeline
- 1D track analysis (linear mazes)
- Bayesian position decoding
- LFP, SWR, assemblies

**opexebo** - 2D Spatial Metrics:
- Comprehensive place/grid/border metrics
- 2D regular grid environments
- Authority from Nobel lab
- Grid score, spatial info, coherence

**neurospatial** - Spatial Primitives & Graphs:
- Spatial discretization (any structure)
- Graph-based operations
- Differential operators
- Works with irregular environments

### Ideal Workflow

**1. Preprocessing**: buzcode
- Load data, spike sorting, LFP filtering
- Data structure standardization

**2. Spatial Discretization**: neurospatial
- Create environment (1D, 2D, irregular)
- Compute occupancy, trajectories
- Graph-based operations

**3. Spatial Metrics**:
- **Regular grids**: opexebo (authoritative)
- **Irregular graphs**: neurospatial.metrics
- **1D tracks**: buzcode (if MATLAB) or neurospatial

**4. Decoding**: buzcode (MATLAB) or custom Python

---

## Key Insights for neurospatial

### 1. buzcode Validates Need for Comprehensive Metrics

**buzcode's limitation**: Minimal spatial metrics

Users working with buzcode **need**:
- Skaggs information
- Sparsity
- Coherence
- Grid score

**This confirms demand** for neurospatial.metrics module.

### 2. 1D Track Support is Important

**buzcode shows**: 1D track analysis is common

**neurospatial advantage**: GraphLayout supports 1D tracks natively
- Better than buzcode (not MATLAB-locked)
- More flexible than opexebo (irregular connectivity)

### 3. Bayesian Decoding Gap

**Neither opexebo nor neurospatial** provide position decoding.

**Opportunity**: Future extension (low priority)
- Could leverage neurospatial's flexible spatial representation
- Would complement buzcode (Python vs MATLAB)

### 4. No Competition with buzcode

**buzcode focuses on**:
- Preprocessing pipelines
- LFP analysis
- Sharp-wave ripple detection
- MATLAB ecosystem

**neurospatial focuses on**:
- Spatial discretization
- Graph operations
- Spatial primitives
- Python ecosystem

**Complementary, not competitive.**

---

## Recommendations

### 1. Do Not Replicate buzcode's Preprocessing

**buzcode's strength**: Full electrophysiology pipeline

**neurospatial should focus**: Spatial analysis only

**Rationale**: Preprocessing is well-served by existing tools.

### 2. Support 1D Tracks Well

**buzcode shows demand** for 1D track analysis.

**neurospatial already does**: GraphLayout
- Track linearization (via track-linearization package)
- 1D environments with custom connectivity

**Action**: Emphasize 1D support in documentation and examples.

### 3. Provide Missing Metrics

**buzcode lacks**: Comprehensive spatial metrics

**neurospatial.metrics should provide**:
- All metrics in opexebo
- Extend to irregular graphs
- 1D track support

**Action**: Implement metrics module (Phase 4 of plan).

### 4. Python Bridge for buzcode Users

**Many buzcode users** may want Python tools.

**neurospatial advantage**: Python ecosystem
- NumPy, SciPy, pandas
- Jupyter notebooks
- Modern ML frameworks

**Action**: Highlight Python advantages in documentation.

---

## Updated Implementation Plan Impact

### No Major Changes Needed

**buzcode analysis confirms**:
1. ✅ Need for comprehensive metrics (validates Phase 4)
2. ✅ 1D track support (neurospatial already has)
3. ✅ Python ecosystem value (no MATLAB dependency)
4. ✅ Gap in existing tools (buzcode minimal metrics)

**No changes to plan**: buzcode fills different niche (preprocessing, MATLAB).

### Documentation Updates

**Add section**: "Comparison with buzcode"
- Emphasize complementary nature
- Show workflow: buzcode preprocessing → neurospatial analysis
- MATLAB vs Python comparison

**Add examples**:
- 1D track analysis (GraphLayout)
- Converting buzcode output to neurospatial
- Python equivalents for buzcode functions

---

## Algorithm Comparison: Place Field Detection

### buzcode Approach

```matlab
% Iterative peak detection
while peaks_remain:
    [peak, loc] = max(rate_map)
    if peak < threshold:
        break
    field = (rate_map >= 0.2 * peak)
    save_field(field)
    rate_map(field) = NaN
```

**Pros**: Simple, fast
**Cons**: Not connected-component based, hardcoded thresholds

### opexebo Approach

```python
# Adaptive thresholding with validation
for threshold_pct in np.arange(0.96, 0.2, -0.02):
    threshold = peak * threshold_pct
    labeled = morphology.label(rate_map > threshold, connectivity=1)
    regions = regionprops(labeled)

    # Validate: no holes, stable boundaries
    if stable and no_holes:
        return regions
```

**Pros**: Sophisticated (adaptive, validated)
**Cons**: More complex, slower

### neurospatial Approach (Proposed)

```python
# Graph-based connected components
def detect_place_fields(firing_rate, env, threshold=0.2):
    peak = firing_rate.max()
    active_bins = firing_rate >= (threshold * peak)

    # Use NetworkX connected components on environment graph
    subgraph = env.connectivity.subgraph(np.where(active_bins)[0])
    fields = list(nx.connected_components(subgraph))

    return fields
```

**Pros**: Works on **any graph** (not just grids)
**Cons**: Need to implement adaptive thresholding from opexebo

**Recommendation**: Adopt opexebo's adaptive thresholding, apply to graph-based detection.

---

## Conclusion

### Key Findings

1. **buzcode provides basic spatial analysis** (1D tracks, minimal metrics)
2. **opexebo provides comprehensive metrics** (2D grids, Nobel lab authority)
3. **neurospatial fills unique gap** (irregular graphs, spatial primitives)

### Three Complementary Tools

- **buzcode**: Preprocessing, 1D tracks, Bayesian decoding, MATLAB
- **opexebo**: Comprehensive metrics, 2D grids, Python, Nobel lab
- **neurospatial**: Spatial primitives, any graph, Python, flexible

### No Competition

**Each tool serves different needs**:
- buzcode: MATLAB users, full pipeline, 1D focus
- opexebo: Python users, 2D metrics, grid cells
- neurospatial: Python users, irregular graphs, primitives

**Users can combine**:
- buzcode → preprocessing
- neurospatial → discretization
- opexebo → metrics (regular grids)
- neurospatial.metrics → metrics (irregular graphs)

### Implementation Plan: No Changes

**buzcode analysis validates plan**:
- ✅ Need for metrics module
- ✅ 1D track support (already have)
- ✅ Python ecosystem value
- ✅ Complementary positioning

**Action items**:
- Document buzcode comparison
- Add examples for 1D tracks
- Show workflow integration
- Emphasize Python advantages

---

## References

- **buzcode**: https://github.com/buzsakilab/buzcode
- **Buzsáki Lab**: https://buzsakilab.com/
- **opexebo**: https://github.com/simon-ball/opexebo
- **Moser Lab**: Kavli Institute, Trondheim
- Key buzcode functions analyzed:
  - `bz_firingMap1D.m`
  - `bz_findPlaceFields1D.m`
  - `bz_getPlaceFields1D.m`
  - `bz_positionDecodingBayesian.m`
