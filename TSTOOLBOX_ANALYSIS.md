# TSToolbox_Utils Analysis: Legacy MATLAB Utilities

**Package**: [PeyracheLab/TSToolbox_Utils](https://github.com/PeyracheLab/TSToolbox_Utils)
**Lab**: Peyrache Lab (now at Simons Foundation Center for Computational Neuroscience)
**Language**: MATLAB (87.3%)
**Type**: LEGACY utility collection (MATLAB predecessor to pynapple)
**Status**: Minimal activity (7 commits, 2 stars, 1 fork)

## 1. Critical Context: TSToolbox → pynapple Evolution

**TSToolbox_Utils is part of the LEGACY MATLAB ecosystem that evolved into pynapple.**

### Historical Timeline

1. **TSToolbox** (MATLAB) - Original package by Francesco P. Battaglia
   - Repository: https://github.com/PeyracheLab/TStoolbox
   - MATLAB-based time-series analysis

2. **neuroseries** (Python) - Core infrastructure extracted from TSToolbox
   - Constituted the foundation of modern pynapple

3. **TSToolbox_Utils** (MATLAB) - Utility scripts for TSToolbox
   - Repository: https://github.com/PeyracheLab/TSToolbox_Utils
   - "Sometimes depending on the TSToolbox"
   - **THIS PACKAGE** (minimal activity, legacy status)

4. **pynapple** (Python) - Modern successor (2021 fork from neuroseries)
   - Developed by Guillaume Viejo, Adrien Peyrache, and team
   - Now maintained at Simons Foundation
   - **CURRENT ACTIVE PACKAGE**

**Conclusion**: TSToolbox_Utils is the MATLAB legacy code that was REPLACED by pynapple (Python).

## 2. Package Overview

TSToolbox_Utils provides "scripts for neuronal and behavioral data analysis" in MATLAB.

**Core capabilities**:
- Spike analysis (burst detection, clustering, waveform features)
- EEG processing
- Position tracking
- Cross-correlation analysis
- Information measures (Fisher info, border score)
- Navigation analysis (place fields, head direction)
- Statistical analysis
- Visualization

## 3. Module Structure

```
TSToolbox_Utils/
├── Compute/              # Computational analysis functions
├── CrossCorr/            # Cross-correlation analysis
├── DatFiles/             # Data file handling
├── EEG/                  # EEG signal processing
├── ExternalScripts/      # Third-party scripts
├── Graphics/             # Visualization
├── InfoMeasures/         # Information-theoretic metrics (46 files)
│   ├── Compute_BorderScore.m
│   ├── FischerInfo.m
│   ├── SpkTrainGenerator.m
│   ├── SpkTrainLogLikelihood.m
│   ├── SpkTrainSpatialValuation.m
│   └── CrossValidation*.m
├── Log/                  # Logging utilities
├── Navigation/           # Spatial navigation analysis (17 files)
│   ├── SpatialInfo.m
│   ├── PlaceFields.m (4 variants)
│   ├── HeadDirectionField.m (6 variants)
│   ├── FindHDCells.m
│   ├── HDIndex.m, HDScore.m
│   ├── BayesReconstruction_1D.m
│   └── AngularAcceleration.m
├── PositionTracking/     # Position tracking
├── Spikes/               # Spike analysis (11 files)
│   ├── BurstIndex.m, BurstSpikes.m
│   ├── ClusterSpkTrains.m
│   ├── FindSynapse.m
│   ├── Make_WaveFormFeatures.m
│   └── Make_MonoSynConnection.m
├── Stats/                # Statistical analysis
└── Wrappers/             # Integrated workflows
```

## 4. Spatial Analysis Capabilities

### 4.1 Spatial Information (SpatialInfo.m)

**Algorithm**: Skaggs information metric

```matlab
% Formula:
SpatBits = SUM { p(x) * lambda(x) * log2(lambda(x) / lambda_mean) }

% where:
% p(x) = occupancy probability at position x
% lambda(x) = firing rate at position x
% lambda_mean = mean firing rate across all positions

% Implementation:
occ = occ / sum(occ)              % Normalize occupancy
f = sum(occ .* pf)                % Mean firing rate
pf = pf / f                       % Normalize place field
SpatBits = sum(occ(pf ~= 0) .* pf(pf ~= 0) .* log2(pf(pf ~= 0)))
```

**Output**: Bits per spike

**Comparison**:
- TSToolbox_Utils: MATLAB implementation
- opexebo: Python implementation (identical formula)
- neurocode: MATLAB implementation (identical formula)
- **ALL USE IDENTICAL SKAGGS FORMULA** ✅

### 4.2 Place Fields (PlaceFields.m)

**Algorithm**:

```matlab
% 1. Compute 2D histogram of spike locations
pfH = hist2(spikePos, bins_x, bins_y)

% 2. Compute 2D histogram of occupancy
occH = hist2(animalPos, bins_x, bins_y)

% 3. Calculate firing rate (Hz)
dt = 1 / median(diff(sampleTimes))  % Sampling interval
pf = dt * pfH ./ occH               % Spikes per second

% 4. Handle invalid values
occH(isnan(occH)) = 0
occH(isinf(occH)) = 0

% 5. Apply Gaussian smoothing
db = round(bins / 5)               % Kernel size
pf = gaussFilt(pf, db/5)           % Final smoothing
```

**Features**:
- 2D spatial binning (similar to vandermeerlab, neurocode)
- Gaussian smoothing (standard approach)
- Flexible occupancy handling (3 usage modes)

**Variants**:
- `PlaceFields.m` - Standard with smoothing
- `PlaceFieldsCont.m` - Continuous tracking
- `PlaceFields_NoSmoothing.m` - Raw firing rate map
- `PlaceFieldsStability.m` - Stability analysis

### 4.3 Border Score (Compute_BorderScore.m)

**Algorithm**:

```matlab
% Formula:
border_score = (cM - d) / (cM + d)

% where:
% cM = maximum wall contact ratio
% d = normalized distance from peak to wall

% Implementation steps:

% 1. Compute distance to all walls
dist_matrix = min(dist_to_wall_east, dist_to_wall_north,
                  dist_to_wall_west, dist_to_wall_south)

% 2. Segment place field (30% peak threshold)
CC, nc = bwlabel(double(pf > 0.3 * peak))

% 3. Compute wall contact ratio
for each_wall:
    wIx(wall) = sum(pixels_touching_wall) / total_wall_pixels

cM = max(wIx)  % Maximum contact ratio

% 4. Compute distance metric (weighted by firing rate)
d = sum(dist_matrix .* pf) / (max_dist * sum(pf))

% 5. Compute border score
b = (cM - d) / (cM + d)  % Range: [-1, +1]
```

**Filtering**:
- Only fields with area > 200 pixels
- Must contact at least one wall

**Comparison**:
- TSToolbox_Utils: Wall contact ratio + distance metric
- opexebo: Similar approach (border score in opexebo.analysis)

### 4.4 Bayesian Decoding (BayesReconstruction_1D.m)

**Algorithm**: Poisson Bayesian decoder (1D only)

```matlab
% Posterior formula:
% P(x|spikes) ∝ P(spikes|x) * P(x)

% Log-likelihood computation:
log_pf = log(pf)                   % Log tuning curves
log_pf(log_pf < -16) = -16         % Prevent underflow

% Prior term:
const_term = log(Px) - tau * sum(pf, 2)

% Posterior (log-space):
log_Pxn = exp(log_pf * Q + const_term)

% Normalization:
log_Pxn = log_Pxn / sum(log_Pxn)

% MAP estimate:
decoded_position = theta_vec(argmax(log_Pxn))
```

**Features**:
- Log-domain computation (numerical stability)
- Chunked processing (100 time bins per chunk)
- Assumes Poisson-distributed spike counts
- Assumes independence across neurons

**Comparison**:
- TSToolbox_Utils: 1D only
- vandermeerlab: 1D + 2D (DecodeZ.m)
- neurocode: 1D + 2D (ReconstructPosition.m)
- buzcode: 1D + 2D + phase + GLM
- pynapple: N-D (decode_2d)

**ALL USE IDENTICAL POISSON LIKELIHOOD MODEL** ✅

### 4.5 Head Direction Analysis

**Functions**:
- `FindHDCells.m` - Detect head direction cells
- `HDIndex.m` - Compute directional tuning strength
- `HDScore.m` - Score head direction selectivity
- `HeadDirectionField.m` (6 variants) - Compute tuning curves

**Variants**:
- Raw firing rates
- Normalized by speed
- Motive-specific processing
- Different smoothing approaches

**Comparison**:
- TSToolbox_Utils: 6 variants (flexibility)
- neurocode: von Mises fitting (circular statistics)
- neurospatial: Circular statistics planned (Phase 4.4)

## 5. What TSToolbox_Utils Provides

### 5.1 Spatial Metrics

| Metric | TSToolbox_Utils | opexebo | neurocode | neurospatial |
|--------|-----------------|---------|-----------|--------------|
| Skaggs info | ✅ | ✅ | ✅ | ❌ Planned |
| Place fields | ✅ 2D | ✅ 2D | ✅ 1D+2D | ❌ Planned |
| Border score | ✅ | ✅ | ❌ | ❌ Planned |
| HD tuning | ✅ | ❌ | ❌ | ❌ Planned |
| HD index | ✅ | ❌ | ❌ | ❌ Planned |
| Grid score | ❌ | ✅ | ❌ | ❌ Planned |
| Coherence | ❌ | ✅ | ❌ | ❌ Planned |
| Sparsity | ❌ | ✅ | ❌ | ❌ Planned |

### 5.2 Decoding and Prediction

**Bayesian decoding**:
- `BayesReconstruction_1D.m` - 1D position decoding

**Cross-validation**:
- `CrossValidationGLM_Cells.m` - GLM-based prediction
- `CrossValidationHDCheck_Cells.m` - Head direction validation
- `ComputePeerPrediction.m` - Peer prediction
- `ComputePeerHDPrediction.m` - HD peer prediction

### 5.3 Spike Train Analysis

**Information measures**:
- `FischerInfo.m` - Fisher information
- `SpkTrainLogLikelihood.m` - Log-likelihood evaluation
- `SpkTrainSpatialValuation.m` - Spatial coding quality
- `SpkTrainHDValuation.m` - HD coding quality

**Burst detection**:
- `BurstIndex.m` - Burst index calculation
- `BurstSpikes.m` - Burst identification

**Synaptic connections**:
- `FindSynapse.m` - Synapse detection
- `Make_MonoSynConnection.m` - Monosynaptic connections

## 6. What TSToolbox_Utils Does NOT Provide

**Missing spatial metrics**:
1. ❌ **Grid score** (use opexebo instead)
2. ❌ **Spatial autocorrelation** (use opexebo instead)
3. ❌ **Coherence** (use opexebo instead)
4. ❌ **Sparsity** (use opexebo instead)

**Missing capabilities**:
1. ❌ **2D Bayesian decoding** (only 1D)
2. ❌ **Replay detection** (use neurocode, buzcode, or vandermeerlab)
3. ❌ **Time-series infrastructure** (use pynapple instead)
4. ❌ **Modern API** (use pynapple instead)

**Platform limitations**:
1. ❌ **MATLAB-only** (legacy code, not actively developed)
2. ❌ **No package manager** (manual installation)
3. ❌ **Minimal documentation** (7 commits, no README)
4. ❌ **No tests** (no test suite)

## 7. Comparison with pynapple (Modern Successor)

### 7.1 Evolution: TSToolbox → pynapple

| Aspect | TSToolbox_Utils | pynapple |
|--------|-----------------|----------|
| **Language** | MATLAB | Python |
| **Status** | Legacy (7 commits) | Active (1000+ commits) |
| **Maintenance** | Minimal | Active (Simons Foundation) |
| **Documentation** | Minimal | Excellent |
| **Tests** | None | Comprehensive |
| **Community** | Small (2 stars) | Large (300+ stars) |
| **Package manager** | None | pip, conda |

### 7.2 Functionality Comparison

| Feature | TSToolbox_Utils | pynapple | neurospatial |
|---------|-----------------|----------|--------------|
| **Time-series** | ❌ (TSToolbox dependency) | ✅ Excellent | ❌ |
| **Spatial info** | ✅ Skaggs | ✅ Skaggs | ❌ Planned |
| **Place fields** | ✅ 2D | ❌ | ❌ Planned |
| **Bayesian decode** | ✅ 1D | ✅ N-D | ❌ |
| **Tuning curves** | ✅ | ✅ | ❌ Planned |
| **Border score** | ✅ | ❌ | ❌ Planned |
| **HD cells** | ✅ 6 variants | ❌ | ❌ Planned |
| **Cross-correlation** | ✅ | ✅ | ❌ |
| **Spike sorting** | ✅ Basic | ❌ | ❌ |
| **Graph discretization** | ❌ | ❌ | ✅ |
| **Differential ops** | ❌ | ❌ | ✅ Planned |

### 7.3 Code Comparison

**Spatial information**:

```matlab
% TSToolbox_Utils (MATLAB)
occ = occ / sum(occ)
f = sum(occ .* pf)
pf = pf / f
SpatBits = sum(occ(pf~=0) .* pf(pf~=0) .* log2(pf(pf~=0)))
```

```python
# pynapple (Python)
occupancy = occupancy / np.sum(occupancy)
mean_rate = np.sum(occupancy * rate_map)
rate_map_norm = rate_map / mean_rate
info = np.sum(occupancy[rate_map > 0] * rate_map[rate_map > 0] *
              np.log2(rate_map[rate_map > 0]))
```

**Identical algorithm, different language!**

## 8. Strategic Positioning

### 8.1 Legacy Status

**TSToolbox_Utils is LEGACY CODE that has been SUPERSEDED by pynapple.**

| Package | Status | Recommendation |
|---------|--------|----------------|
| **TSToolbox_Utils** | LEGACY | ❌ Do not use for new projects |
| **pynapple** | ACTIVE | ✅ Use for time-series analysis |
| **neurospatial** | ACTIVE | ✅ Use for spatial discretization |
| **opexebo** | ACTIVE | ✅ Use for spatial metrics |

### 8.2 When to Use TSToolbox_Utils

**Use TSToolbox_Utils if**:
- ✅ You have existing MATLAB workflows (already using TSToolbox)
- ✅ You need head direction analysis (6 variants not in other packages)
- ✅ You need border score (simpler than implementing from scratch)

**Do NOT use TSToolbox_Utils if**:
- ❌ You're starting a new project (use pynapple instead)
- ❌ You work in Python (use pynapple, opexebo, neurospatial)
- ❌ You need modern API/documentation (minimal in TSToolbox_Utils)
- ❌ You need active maintenance (7 commits, no recent activity)

### 8.3 Modern Python Stack (Recommendation)

**For NEW projects, use this stack**:

```python
# 1. Time-series infrastructure (pynapple)
import pynapple as nap

spikes = nap.Ts(t=spike_times)
position = nap.Tsd(t=time, d=coords)
epochs = nap.IntervalSet(start=starts, end=ends)

# Tuning curves (pynapple provides this)
tc = nap.compute_2d_tuning_curves(spikes, position, nb_bins=50)

# Skaggs information (pynapple provides this)
info = nap.compute_1d_mutual_info(tc, position)

# 2. Spatial discretization (neurospatial)
from neurospatial import Environment

env = Environment.from_samples(position.values, bin_size=2.0)
env.units = 'cm'

# 3. Spatial metrics (neurospatial - once implemented)
from neurospatial.metrics import (
    detect_place_fields,     # Not in pynapple
    coherence,               # Not in pynapple
    grid_score,              # Not in pynapple
)

fields = detect_place_fields(tc, env)
coh = coherence(tc, env)

# 4. Validation (opexebo - for comparison)
import opexebo

autocorr = opexebo.analysis.spatial_autocorrelation(rate_map)
grid = opexebo.analysis.grid_score(autocorr)
```

**This provides EVERYTHING TSToolbox_Utils provided, plus more, in modern Python.**

## 9. Impact on neurospatial Implementation Plan

### 9.1 What TSToolbox_Utils Validates

✅ **Skaggs information formula** - Confirmed across TSToolbox_Utils, opexebo, neurocode, pynapple

✅ **Poisson Bayesian decoding** - Confirmed as field standard

✅ **Border score algorithm** - Wall contact ratio + distance metric

✅ **Head direction analysis** - Multiple variants show importance of flexibility

### 9.2 What TSToolbox_Utils Adds

**Border score** (NEW):
- TSToolbox_Utils provides `Compute_BorderScore.m`
- opexebo also provides border score
- **Recommendation**: Add to Phase 4.3 (grid cell metrics)

**Head direction index** (NEW):
- TSToolbox_Utils provides `HDIndex.m`, `HDScore.m`
- 6 variants show multiple analysis approaches
- **Recommendation**: Add to Phase 4.4 (circular statistics)

### 9.3 No Changes to Core Plan

The implementation plan remains unchanged:

| Phase | Timeline | Risk | Status |
|-------|----------|------|--------|
| Phase 1: Differential operators | 3 weeks | Low | Validated |
| Phase 2: Signal processing | 6 weeks | Medium | Validated |
| Phase 3: Path operations | 1 week | Low | Validated |
| Phase 4: Metrics module | 2 weeks | Low | Validated ✅ |
| Phase 5: Polish | 2 weeks | Low | Validated |

**Timeline**: 15 weeks (no change)

**Minor addition** (Phase 4.3 - Grid Cell Metrics):

Add **border_score** function:
```python
def border_score(
    firing_rate: NDArray,
    env: Environment,
    *,
    threshold: float = 0.3,
    min_area: int = 200,
) -> float:
    """
    Compute border score (Solstad et al. 2008).

    Formula: b = (cM - d) / (cM + d)

    where:
    - cM = maximum wall contact ratio
    - d = normalized distance from peak to wall

    Follows TSToolbox_Utils and opexebo implementations.

    Parameters
    ----------
    firing_rate : array
        Spatial firing rate map
    env : Environment
        Spatial environment
    threshold : float, default=0.3
        Fraction of peak for field segmentation (30%)
    min_area : int, default=200
        Minimum field area (pixels) for evaluation

    Returns
    -------
    score : float
        Border score [-1, +1]. Higher values indicate stronger
        boundary tuning.

    References
    ----------
    .. [1] Solstad et al. (2008). Neuron 58(6).
    .. [2] TSToolbox_Utils Compute_BorderScore.m
    .. [3] opexebo.analysis.border_score
    """
    # Implementation follows TSToolbox_Utils approach
    pass
```

**Effort**: <1 hour (trivial addition, well-documented algorithm)

## 10. Key Takeaways

### 10.1 What TSToolbox_Utils Is

✅ **LEGACY MATLAB utilities** for neuronal and behavioral data analysis
✅ **Predecessor to pynapple** (historical context)
✅ **Provides spatial metrics** (Skaggs info, place fields, border score, HD analysis)
✅ **Poisson Bayesian decoding** (1D only)
✅ **Spike train analysis** (burst detection, waveform features, synaptic connections)

### 10.2 What TSToolbox_Utils Is NOT

❌ **Active project** (7 commits, minimal activity)
❌ **Modern API** (MATLAB, no documentation, no tests)
❌ **Time-series infrastructure** (depends on TSToolbox)
❌ **Python** (MATLAB-only, superseded by pynapple)

### 10.3 Strategic Recommendation

**For neurospatial users**:
- ❌ **Do NOT use TSToolbox_Utils** for new projects
- ✅ **Use pynapple** for time-series infrastructure (modern Python replacement)
- ✅ **Use neurospatial** for spatial discretization and graph-based analysis
- ✅ **Use opexebo** for spatial metric validation
- ✅ **Reference TSToolbox_Utils algorithms** (border score, HD index) but implement in Python

**For the ecosystem**:
- TSToolbox_Utils provides **algorithm validation** (Skaggs info, border score)
- pynapple is the **modern successor** (Python, active development)
- neurospatial fills **unique gap** (graph-based spatial primitives not in pynapple or TSToolbox_Utils)

### 10.4 Implementation Plan Status

**Validated**: Border score algorithm (add to Phase 4.3)
**Timeline**: 15 weeks (no change)
**Risk**: MEDIUM (unchanged)
**Impact**: HIGH (TSToolbox_Utils confirms our approach)

---

**Package comparison summary** (8 packages analyzed):

| Package | Type | Language | Status | Unique Value |
|---------|------|----------|--------|--------------|
| **TSToolbox_Utils** | ANALYSIS | MATLAB | LEGACY | Border score, HD variants |
| **pynapple** | ANALYSIS | Python | ACTIVE | Time-series (TSToolbox successor) |
| RatInABox | SIMULATION | Python | ACTIVE | Synthetic data generation |
| neurospatial | ANALYSIS | Python | ACTIVE | Any graph, differential ops |
| opexebo | ANALYSIS | Python | ACTIVE | Metrics validation |
| vandermeerlab | ANALYSIS | MATLAB | ACTIVE | Task workflows |
| neurocode | ANALYSIS | MATLAB | ACTIVE | Comprehensive pipeline |
| buzcode | ANALYSIS | MATLAB | ACTIVE | Preprocessing |

**pynapple (Python) has REPLACED TSToolbox_Utils (MATLAB)** ✅

**neurospatial fills the gap for graph-based spatial analysis** ✅
