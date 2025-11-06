# neurocode Analysis: Spatial Primitives & Metrics

**Package**: [ayalab1/neurocode](https://github.com/ayalab1/neurocode)
**Lab**: AyA Lab (Brain Computation and Behavior Lab), Cornell University
**Language**: MATLAB (58.7%), Jupyter Notebooks (30.8%)
**Version**: v1.0.0 (April 2023)
**Status**: Actively maintained (1,874 commits, 14 contributors)

## 1. Package Overview

neurocode is a comprehensive MATLAB toolkit for analyzing neural and behavioral data, developed by Adrien Peyrache's lab at Cornell. It provides end-to-end analysis pipelines from raw data loading through advanced spatial and temporal analyses.

**Core capabilities**:
- Electrophysiology analysis (spike sorting, spike-LFP interactions)
- Place cell analysis and spatial navigation
- Sharp-wave ripple detection
- Reactivation and replay analysis
- Brain state classification (sleep scoring)
- Behavioral tracking and integration
- Spectral analysis and signal processing

## 2. Module Structure

```
neurocode/
├── PlaceCells/          # Spatial neuroscience analysis (18 files)
├── behavior/            # Behavioral tracking and metrics (35+ files)
├── reactivation/        # Replay and reactivation detection
├── SharpWaveRipples/    # HFO detection
├── brainStates/         # Sleep scoring
├── spikes/              # Spike train analysis
├── lfp/                 # LFP analysis
├── core/                # Data structures (IntervalArray, SpikeArray)
├── utilities/           # Spatial primitives (Bin, Accumulate, Smooth)
├── SpectralAnalyses/    # Temporal frequency analysis (Chronux)
└── tutorials/           # Examples and documentation
```

## 3. Spatial Analysis Capabilities

### 3.1 Place Cell Analysis (PlaceCells/)

**Core functions** (18 total):

1. **FiringMap.m** - Generate spatial firing rate maps (1D/2D)
2. **MapStats.m** - Compute place field statistics
3. **findPlaceFieldsAvg1D.m** - Detect place fields in 1D environments
4. **findPlaceFieldsTemplate.m** - Template-based field detection
5. **firingMapAvg.m** - Smoothed firing map generation
6. **firingMapLaps.m** - Lap-based firing analysis
7. **PhasePrecession.m** - Phase precession analysis
8. **ReconstructPosition.m** - Bayesian position decoding
9. **ReconstructPosition2Dto1D.m** - 2D→1D position conversion
10. **TestRemapping.m** - Context remapping detection

### 3.2 Spatial Metrics Computed

**MapStats.m outputs**:

| Metric | Description | Units |
|--------|-------------|-------|
| `peak` | Maximum in-field firing rate | Hz |
| `mean` | Average firing rate within field | Hz |
| `size` | Field extent | bins |
| `field` | Binary mask (1 = in field) | boolean |
| `fieldX, fieldY` | Rectangular boundaries | bins |
| `specificity` | Skaggs spatial information | bits |
| `x, y` | Location of peak | bin coords |

**Circular statistics** (1D circular tracks):
- `m` - Mean angle
- `mode` - Distribution mode
- `r` - Mean resultant length
- `k` - Von Mises concentration parameter

### 3.3 Spatial Primitives (utilities/)

**Key functions used for spatial analysis**:

```matlab
% Binning and accumulation
Bin.m              % Discretize continuous coordinates
Accumulate.m       % Accumulate values into spatial bins
NormToInt.m        % Normalize to integer indices

% Smoothing and filtering
Smooth.m           % Basic smoothing
gausskernel.m      % Gaussian kernel generation
nansmooth.m        % Smoothing with NaN handling

% Statistical operations
CumSum.m           % Cumulative summation
Diff.m             % Spatial differentiation
nancorr.m          % Correlation with NaN support
```

## 4. Algorithm Analysis

### 4.1 Firing Map Generation (Map.m)

**Algorithm**:
```matlab
% 1. Normalize coordinates to [0, 1]
x = (x - min(x)) / (max(x) - min(x))
y = (y - min(y)) / (max(y) - min(y))

% 2. Discretize into bins (default 50x50)
x_binned = Bin(x, [0 1], nBinsX)
y_binned = Bin(y, [0 1], nBinsY)

% 3. Accumulate spike counts and occupancy time
map.count = Accumulate([x y], n, 'size', nBins)
map.time = Accumulate([x y], dt, 'size', nBins)

% 4. Compute firing rate (Hz)
map.z = map.count / (map.time + eps)  % eps prevents division by zero

% 5. Apply Gaussian smoothing (default: 2 bins)
map.z = Smooth(map.z, smoothing_size)
```

**Occupancy handling**:
- Minimum time threshold (`minTime`, default 0s)
- Gap clipping (`maxGap`, default 0.1s)
- Interpolation or discard for missing bins

### 4.2 Place Field Detection (MapStats.m, findPlaceFieldsAvg1D.m)

**Iterative peak-based algorithm**:
```matlab
% 1. Compute Skaggs information
specificity = SUM { p(i) * λ(i)/λ * log2(λ(i)/λ) }
% where p(i) = occupancy probability, λ(i) = firing rate

% 2. Iteratively find peaks
while peaks_remain:
    peak = max(rate_map)

    % 3. Define field as connected region > threshold*peak
    field = (rate_map >= threshold * peak)  % default threshold = 0.2

    % 4. Check for coalescent subfields
    subfields = detect_subfields(field)
    if subfield_size < 0.5 * parent_size:
        retain_subfields()

    % 5. Filter by size and peak
    if size >= minSize and peak >= minPeak:
        save_field()

    % 6. Remove detected field and continue
    rate_map(field) = NaN

% Default thresholds:
% - threshold = 0.2 (20% of peak)
% - minSize = 100 bins (2D) or 10 bins (1D)
% - minPeak = 1 Hz
% - sepEdge = 0% (distance from maze edge)
```

**Detection criteria**:
- **Primary fields**: peak ≥ 1 Hz, size 5-50% of maze
- **Secondary fields**: peak ≥ 60% of max rate
- **Spatial**: separated from edges by `sepEdge`
- **Quality**: coalescent subfield discrimination

### 4.3 Bayesian Position Decoding (ReconstructPosition.m)

**Poisson likelihood model**:
```matlab
% For each spatial bin x and time window t:
P(x|spikes) ∝ P(spikes|x) * P(x)

% Poisson likelihood for each neuron i:
P(spikes_i|x) = (λ_i(x) * dt)^n_i * exp(-λ_i(x) * dt) / n_i!

% Combined likelihood across neurons:
P(spikes|x) = PRODUCT_i P(spikes_i|x)

% Posterior (uniform prior):
posterior = P(spikes|x) / SUM_x P(spikes|x)
```

## 5. Behavioral Analysis (behavior/)

**Spatial processing capabilities** (35+ files):

**Velocity and movement**:
- `LinearVelocity.m`, `AngularVelocity.m`, `KalmanVel.m`
- `MovementPeriods.m`, `QuietPeriods.m` - activity state detection

**Spatial regions**:
- `DefineZone.m`, `IsInZone.m` - ROI mapping
- `behavioral_ROI_locator.m` - region identification

**Trajectory processing**:
- `linearization_pipeline.py` - track linearization (Python!)
- `NSMAFindGoodLaps.m`, `FindLapsNSMAadapted.m` - lap detection

**Tracking integration**:
- `LED2Tracking.m`, `detectLED.m` - LED marker detection
- `load_dlc_csv.m`, `process_and_sync_dlc.m` - DeepLabCut integration

**Behavioral metrics**:
- `objPlaceScore.m`, `preferenceScore.m` - behavioral scoring

## 6. Reactivation Analysis (reactivation/)

**Methods provided**:

1. **ReplayScore.m** - Quantify replay strength
2. **FindReplayScore.m** - Detect replay events
3. **ReactivationStrength.m** - Measure reactivation magnitude
4. **ExplainedVariance.m** - Variance explained by replay
5. **ActivityTemplates.m** - Generate reference patterns
6. **NMF/** - Non-negative matrix factorization for assemblies

**Typical workflow**:
```matlab
% 1. Generate activity templates from behavior
templates = ActivityTemplates(spikes, position, fields)

% 2. Compute reactivation during rest
reactivation = ReactivationStrength(rest_spikes, templates)

% 3. Score replay quality
score = ReplayScore(rest_activity, templates)

% 4. Explained variance
variance = ExplainedVariance(activity, templates)
```

## 7. Data Structures (core/)

**Foundation classes**:

```matlab
% Time intervals (similar to pynapple.IntervalSet)
IntervalArray.m

% Spike trains (similar to pynapple.Ts)
SpikeArray.m

% Continuous signals (similar to pynapple.Tsd)
analogSignalArray.m
```

These provide time-series infrastructure similar to pynapple, but in MATLAB.

## 8. Comparison with Other Packages

### 8.1 Spatial Metrics Coverage

| Metric | neurocode | opexebo | buzcode | pynapple | neurospatial |
|--------|-----------|---------|---------|----------|--------------|
| **Place Fields** |
| Skaggs info | ✅ MapStats | ✅ | ✅ | ✅ | ❌ Planned |
| Peak rate | ✅ | ✅ | ✅ | - | - |
| Field size | ✅ | ✅ | ✅ | - | - |
| Sparsity | ❌ | ✅ | ❌ | - | ❌ Planned |
| Coherence | ❌ | ✅ | ❌ | - | ❌ Planned |
| Field detection | ✅ Advanced | ✅ Adaptive | ✅ Simple | - | ❌ Planned |
| **Grid Cells** |
| Grid score | ❌ | ✅ | ❌ | - | ❌ Planned |
| Spatial autocorr | ❌ | ✅ FFT | ❌ | - | ❌ Planned |
| **Decoding** |
| Bayesian | ✅ Poisson | - | ✅ Multiple | ✅ | - |
| Position recon | ✅ | - | ✅ | - | - |
| **Replay** |
| Replay score | ✅ | - | ✅ | - | - |
| Reactivation | ✅ NMF | - | ✅ | - | - |

### 8.2 Implementation Comparison

**Place field detection algorithms**:

```matlab
% neurocode (iterative peak-based)
while peaks_remain:
    peak = max(rate_map)
    field = (rate_map >= 0.2 * peak)
    check_subfields()
    save_if_valid()
    rate_map(field) = NaN

% opexebo (adaptive thresholding)
for threshold in arange(0.96, 0.2, -0.02):
    labeled = morphology.label(rate_map > threshold)
    if stable and no_holes:
        return regions

% buzcode (simple iterative)
while peaks_remain:
    peak = max(rate_map)
    field = (rate_map >= 0.2 * peak)
    save_field(field)
    rate_map(field) = NaN
```

**Comparison**:
- neurocode: Most sophisticated (subfield discrimination, circular stats)
- opexebo: Most robust (adaptive thresholding, morphological ops)
- buzcode: Simplest (basic iterative)

### 8.3 Language and Platform

| Package | Language | Platform | Time-series | Spatial Discretization |
|---------|----------|----------|-------------|------------------------|
| neurocode | MATLAB | MATLAB only | ✅ IntervalArray | Regular grids only |
| opexebo | Python | Cross-platform | ❌ | Regular grids only |
| buzcode | MATLAB | MATLAB only | ✅ | 1D/2D regular grids |
| pynapple | Python | Cross-platform | ✅ Excellent | ❌ |
| neurospatial | Python | Cross-platform | ❌ | ✅ Any graph |

## 9. What neurocode DOES NOT Provide

**Missing spatial metrics**:
1. ❌ **Sparsity** - fraction of environment where cell fires
2. ❌ **Coherence** - correlation with neighbor average
3. ❌ **Grid score** - hexagonal periodicity measure
4. ❌ **Spatial autocorrelation** - 2D autocorrelation maps
5. ❌ **Border score** - proximity to boundaries
6. ❌ **Head direction** - circular tuning metrics
7. ❌ **Speed score** - speed modulation

**Missing spatial primitives**:
1. ❌ **Graph-based operations** - only regular grids supported
2. ❌ **Irregular spatial binning** - no arbitrary graph support
3. ❌ **Geodesic distances** - only Euclidean
4. ❌ **Differential operators** - no gradient/divergence
5. ❌ **Spatial autocorrelation on graphs** - FFT only for grids

**Platform limitations**:
1. ❌ **MATLAB-only** - not accessible to Python-only users
2. ❌ **Regular grids only** - cannot handle irregular environments
3. ❌ **No graph abstraction** - limited to Cartesian binning

## 10. What neurocode DOES Uniquely Provide

**Strengths**:

1. ✅ **Comprehensive pipeline** - data loading → preprocessing → analysis → visualization
2. ✅ **Replay analysis** - sophisticated reactivation detection (NMF, assemblies)
3. ✅ **Sharp-wave ripples** - dedicated HFO detection module
4. ✅ **Brain states** - sleep scoring and state classification
5. ✅ **Behavioral integration** - DeepLabCut, LED tracking, ROI mapping
6. ✅ **Circular statistics** - von Mises fitting for 1D circular tracks
7. ✅ **Coalescent subfield detection** - advanced place field discrimination
8. ✅ **Multi-dimensional decoding** - 2D→1D conversion
9. ✅ **Lap-based analysis** - lap detection and lap-wise firing maps
10. ✅ **Production-ready** - actively maintained, DOI-registered, v1.0.0 release

## 11. Strategic Positioning

### 11.1 Complementary Ecosystem

neurocode fits into the broader ecosystem as:

| Package | Role | Unique Value | Limitation |
|---------|------|--------------|------------|
| **neurocode** | Full MATLAB pipeline | Comprehensive end-to-end analysis | MATLAB-only, regular grids |
| **opexebo** | Python spatial metrics | Nobel Prize validation, comprehensive | Regular grids only |
| **buzcode** | MATLAB preprocessing | Full pipeline, Bayesian decoding | MATLAB-only, minimal metrics |
| **pynapple** | Python time-series | Excellent time infrastructure | No spatial discretization |
| **neurospatial** | Python spatial primitives | Any graph topology, differential ops | No time-series (use pynapple) |

### 11.2 Workflow Integration

**Ideal workflow combining tools**:

```python
# 1. Preprocessing and spike sorting (neurocode MATLAB or buzcode)
# 2. Convert to Python
# 3. Time-series analysis (pynapple)
from pynapple import Ts, Tsd, IntervalSet

spikes = Ts(t=spike_times)  # Spike trains
position = Tsd(t=time, d=coords)  # Position tracking
epochs = IntervalSet(start=starts, end=ends)  # Task epochs

# 4. Spatial discretization (neurospatial)
from neurospatial import Environment

env = Environment.from_samples(position.values, bin_size=2.0)
env.units = "cm"

# 5. Tuning curves (pynapple)
tuning_curves = nap.compute_2d_tuning_curves(
    spikes, position, nb_bins=env.n_bins
)

# 6. Spatial metrics (neurospatial - once implemented)
from neurospatial.metrics import (
    skaggs_information, sparsity, coherence, grid_score
)

info = skaggs_information(tuning_curves, env)
sparse = sparsity(tuning_curves, env)
coh = coherence(tuning_curves, env)

# 7. Validate against opexebo (regular grids only)
import opexebo
autocorr = opexebo.analysis.spatial_autocorrelation(rate_map)
grid = opexebo.analysis.grid_score(autocorr)

# 8. Advanced replay analysis (neurocode MATLAB)
% Back to MATLAB for replay detection
templates = ActivityTemplates(spikes, position, fields);
replay = ReplayScore(rest_activity, templates);
```

### 11.3 User Personas

**Choose neurocode if**:
- ✅ You work in MATLAB
- ✅ You need complete pipeline (loading → analysis → visualization)
- ✅ You analyze replay/reactivation
- ✅ You need sharp-wave ripple detection
- ✅ You analyze sleep states
- ✅ You work with 1D circular tracks (circular statistics)
- ✅ You need production-ready, validated tools

**Choose neurospatial if**:
- ✅ You work in Python
- ✅ You have irregular environments (complex mazes, open fields with obstacles)
- ✅ You need graph-based spatial primitives
- ✅ You want differential operators (gradient, divergence, Laplacian)
- ✅ You need spatial autocorrelation on irregular graphs
- ✅ You integrate with pynapple for time-series

**Use both if**:
- ✅ You preprocess in neurocode (MATLAB)
- ✅ Then switch to neurospatial+pynapple (Python) for analysis
- ✅ Validate spatial metrics against opexebo

## 12. Impact on neurospatial Implementation Plan

### 12.1 Algorithm Validation

neurocode provides additional validation for our implementation plan:

**✅ Skaggs information formula confirmed**:
```matlab
% neurocode (MapStats.m)
specificity = SUM { p(i) * λ(i)/λ * log2(λ(i)/λ) }
```

This matches opexebo and our proposal.

**✅ Place field detection approach validated**:
- Iterative peak-based detection (confirmed across neurocode, opexebo, buzcode)
- Threshold at 20% of peak (confirmed)
- Subfield discrimination (neurocode most sophisticated)

**✅ Bayesian decoding formula confirmed**:
- Poisson likelihood model (confirmed)
- Matches buzcode and pynapple implementations

### 12.2 New Insights

**Circular statistics**:
- neurocode provides **von Mises fitting** for 1D circular tracks
- This is MISSING from our implementation plan
- **Recommendation**: Add circular statistics module (Phase 4.3)

**Coalescent subfield detection**:
- neurocode has sophisticated subfield discrimination
- This is MORE ADVANCED than opexebo's adaptive thresholding
- **Recommendation**: Adopt neurocode's recursive threshold approach

**Lap-based analysis**:
- neurocode provides lap detection and lap-wise firing maps
- This is useful for within-session plasticity studies
- **Recommendation**: Consider lap-based trajectory analysis (future)

### 12.3 No Changes to Core Plan

The implementation plan remains largely unchanged:

| Phase | Timeline | Risk | Status |
|-------|----------|------|--------|
| Phase 1: Differential operators | 3 weeks | Low | Validated (NetworkX confirms gaps) |
| Phase 2: Signal processing | 6 weeks | Medium | Validated (opexebo algorithms) |
| Phase 3: Path operations | 1 week | Low | Validated (buzcode, neurocode) |
| Phase 4: Metrics module | 2 weeks | Low | Validated (opexebo, neurocode, buzcode) |
| Phase 5: Polish | 2 weeks | Low | Validated |

**Minor additions**:
- Phase 4.3: Circular statistics (von Mises fitting, 1 week)
- Phase 4.4: Coalescent subfield detection (recursive threshold, included in field detection)

**Updated timeline**: 14 weeks → 15 weeks (adding circular statistics)

## 13. Key Takeaways

### 13.1 What neurocode Validates

✅ **Algorithms**: Skaggs info, place field detection, Bayesian decoding
✅ **Thresholds**: 20% peak, 1 Hz minimum, 5-50% field size
✅ **Approach**: Iterative peak-based detection is standard
✅ **Need**: Spatial primitives (Bin, Accumulate, Smooth) are fundamental

### 13.2 What neurocode Lacks

❌ **Python implementation**: MATLAB-only limits accessibility
❌ **Irregular graphs**: Regular grids only
❌ **Grid cell metrics**: No grid score, spatial autocorrelation
❌ **Differential operators**: No gradient, divergence, Laplacian
❌ **Coherence and sparsity**: Missing key place field metrics

### 13.3 What neurospatial Should Adopt

1. ✅ **Coalescent subfield detection**: Recursive threshold approach (neurocode)
2. ✅ **Circular statistics**: Von Mises fitting for circular tracks (neurocode)
3. ✅ **Spatial primitives**: Bin, Accumulate, Smooth as foundation (ALREADY have neighbor_reduce!)
4. ✅ **Validation**: Cross-check against neurocode outputs (MATLAB bridge)

### 13.4 Strategic Positioning Confirmed

**neurocode confirms that neurospatial fills a unique gap**:

1. ✅ **Python implementation** (neurocode is MATLAB-only)
2. ✅ **Irregular graph support** (neurocode is regular grids only)
3. ✅ **Spatial autocorrelation on graphs** (neurocode lacks this)
4. ✅ **Differential operators** (neurocode lacks this)
5. ✅ **Grid cell metrics** (neurocode lacks this)
6. ✅ **Integration with pynapple** (neurocode is MATLAB ecosystem)

**The combination of neurospatial + pynapple provides a Python alternative to the neurocode MATLAB pipeline, with additional capabilities for irregular environments and graph-based spatial analysis.**

## 14. Conclusion

neurocode is an excellent, production-ready MATLAB toolkit with:
- ✅ Comprehensive end-to-end pipeline
- ✅ Advanced place field analysis (coalescent subfields, circular stats)
- ✅ Replay/reactivation detection
- ✅ Sharp-wave ripple detection
- ✅ Sleep scoring and brain states
- ✅ Behavioral integration (DeepLabCut, LED tracking)

However, it is **limited to MATLAB and regular grids**, leaving a gap for:
- ❌ Python-native implementation
- ❌ Irregular spatial graphs
- ❌ Grid cell metrics
- ❌ Differential operators

**neurospatial + pynapple together provide a Python alternative to neurocode with unique capabilities for graph-based spatial analysis.**

The implementation plan is validated and strengthened by neurocode's algorithms. No major changes needed, minor additions for circular statistics and subfield detection.

---

**Next steps**:
1. Commit this analysis
2. Update IMPLEMENTATION_PLAN.md with circular statistics (Phase 4.3, +1 week)
3. Proceed with Phase 1 implementation (differential operators)
