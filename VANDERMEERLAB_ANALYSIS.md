# vandermeerlab Analysis: Spatial Primitives & Metrics

**Package**: [vandermeerlab/vandermeerlab](https://github.com/vandermeerlab/vandermeerlab)
**Lab**: Vandermeer Lab, Dartmouth College
**Language**: MATLAB (78.3%), Jupyter Notebooks (15.3%)
**Version**: 2 releases (latest 2016)
**Status**: Actively maintained (736 commits, 10 contributors)

## 1. Package Overview

vandermeerlab is a comprehensive MATLAB toolkit for analyzing neural data from behavioral neuroscience experiments, particularly focused on spatial navigation and hippocampal recordings. The package emphasizes practical workflows and task-specific analyses.

**Core capabilities**:
- Place cell detection and analysis (1D environments)
- 2D spatial tuning curves
- Bayesian position decoding (Poisson likelihood model)
- Replay detection and scoring (template matching)
- Track linearization (2D → 1D)
- Sharp-wave ripple detection
- LFP analysis and filtering
- Multi-unit activity extraction
- Cell classification

## 2. Module Structure

```
vandermeerlab/
├── code-matlab/
│   ├── shared/
│   │   ├── datatypes/      # Data structures (ts, tsd, iv, tc)
│   │   ├── io/             # Data loaders
│   │   ├── proc/           # Core analysis functions
│   │   │   ├── TuningCurves.m
│   │   │   ├── DetectPlaceCells1D.m
│   │   │   ├── Decoding/  # Bayesian decoding
│   │   │   └── CellClassification/
│   │   ├── linearize/      # Track linearization
│   │   ├── stats/          # Statistical functions
│   │   ├── viz/            # Visualization
│   │   └── util/           # Utilities
│   ├── tasks/              # Task-specific analyses
│   │   ├── Alyssa_Tmaze/
│   │   ├── Julien_linear_track/
│   │   ├── Eric_square_maze/
│   │   └── Replay_Analysis/
│   └── example_workflows/  # Tutorial workflows
│       ├── PlaceCellDecoding_2D.m
│       ├── replay_scoring.m
│       └── Loading_and_Linearizing_Position_Data.m
└── toolboxes/              # Third-party toolboxes
```

## 3. Spatial Analysis Capabilities

### 3.1 Tuning Curves (TuningCurves.m)

**Algorithm**:
```matlab
% 1. Bin tuning variable(s) into occupancy bins
occupancy = histc(tuning_var, bin_edges)

% 2. Convert occupancy to time
occupancy_time = occupancy * cfg.occ_dt  % default 1/30 s per sample

% 3. For each cell, bin spikes
spike_counts = histc(spike_positions, bin_edges)

% 4. Optional smoothing (before division)
if cfg.smooth
    spike_counts = conv(spike_counts, gausskernel, 'same')
    occupancy = conv(occupancy, gausskernel, 'same')
end

% 5. Compute tuning curve (firing rate)
tc(cell, bin) = spike_counts(cell, bin) / occupancy_time(bin)

% 6. Set low-occupancy bins to NaN
tc(occupancy < cfg.minOcc) = NaN
```

**Key features**:
- Supports 1D and 2D tuning variables
- Configurable bin edges (irregular binning supported)
- Gaussian smoothing via convolution
- Minimum occupancy threshold (excludes sparse bins)
- Sample-based occupancy converted to time

**Comparison with neurospatial**:
- vandermeerlab: Irregular bin edges supported ✅
- neurospatial: Regular grids only (currently) ❌
- Both: Gaussian smoothing, occupancy normalization

### 3.2 Place Cell Detection (DetectPlaceCells1D.m)

**Detection criteria**:
```matlab
% 1. Peak firing rate threshold
peak_rate >= cfg.thr  % default 5 Hz

% 2. Mean firing rate constraint (exclude interneurons)
mean_rate <= cfg.max_meanfr  % default 10 Hz

% 3. Field size constraint
field_size >= cfg.minSize  % default 4 bins

% 4. Optional spike count requirement
n_spikes_in_field >= cfg.nSpikesInField  % if specified
```

**Field detection**:
```matlab
% Detect contiguous regions above threshold
fields = rate_map >= cfg.thr

% Find connected components
field_regions = find_contiguous_regions(fields)

% Filter by minimum size
valid_fields = field_regions(field_size >= cfg.minSize)

% Extract peak locations
peak_loc = find_peak_within_field(rate_map, valid_fields)
```

**Output statistics**:
- `template_idx` - Indices of detected place cells
- `peak_idx` - Location of maximum firing rate per cell
- `peak_loc` - All firing field peaks per cell (multiple fields allowed)
- `field_loc` - Unified list of field peaks, sorted by location
- `field_template_idx` - Cell identity for each field

**Comparison with other packages**:

| Feature | vandermeerlab | neurocode | opexebo | buzcode |
|---------|---------------|-----------|---------|---------|
| Dimension | 1D only | 1D + 2D | 2D only | 1D + 2D |
| Threshold | 5 Hz peak | 1 Hz peak | 20% of peak | 1 Hz peak |
| Min size | 4 bins | 5% of maze | 100 bins (2D) | 10 bins (1D) |
| Interneuron filter | ✅ max_meanfr | ❌ | ❌ | ❌ |
| Subfield detection | ❌ | ✅ Sophisticated | ✅ Adaptive | ❌ |
| Multiple fields | ✅ | ✅ | ✅ | ✅ |

**Unique features**:
- ✅ **Interneuron exclusion** via maximum mean firing rate
- ❌ **No subfield discrimination** (simpler than neurocode/opexebo)
- ✅ **Optional spike count validation** within field boundaries

### 3.3 Bayesian Decoding (DecodeZ.m)

**Algorithm**: Poisson likelihood Bayesian decoder

```matlab
% For each spatial bin x and time window t:
% P(x|spikes) ∝ P(spikes|x) * P(x)

% Poisson likelihood for each neuron i:
% P(spikes_i|x) = (λ_i(x) * dt)^n_i * exp(-λ_i(x) * dt) / n_i!

% Log-likelihood computation:
log_likelihood = sum_i(Q_i * log(tc_i(x)))  % spike counts * log(rate)

% Exponential decay correction:
decay = exp(-binsize * sum(tc(:,x)))

% Posterior (log-space, then exponentiate):
log_posterior = log_likelihood + log(decay) + log(prior)
posterior = exp(log_posterior)

% Normalize:
posterior = posterior / sum(posterior)

% Decode (maximum a posteriori):
decoded_position = argmax(posterior)
```

**Key features**:
- **Poisson likelihood** (standard for spike count data)
- **Uniform prior** (can be replaced with occupancy-based prior)
- **Log-space computation** (numerical stability)
- **Bin-by-bin normalization** (each time window independently decoded)

**Comparison with other decoders**:

| Package | Method | Likelihood | Prior | Implementation |
|---------|--------|------------|-------|----------------|
| vandermeerlab | Bayesian | Poisson | Uniform | MATLAB |
| neurocode | Bayesian | Poisson | Uniform | MATLAB |
| buzcode | Bayesian | Poisson, Phase, GLM | Uniform/Occupancy | MATLAB |
| pynapple | Bayesian | Poisson | Uniform | Python |
| neurospatial | - | - | - | Not implemented |

**All packages use identical Poisson Bayesian approach** - this is the field standard.

### 3.4 Track Linearization (LinearizePos.m)

**Algorithm**: Nearest-neighbor projection

```matlab
% Given: 2D position (x, y) and coordinate path Coord_in
% Goal: Project to 1D linear position

% 1. Find nearest coordinate point for each position sample
linpos_idx = griddata(
    Coord_in.coord(1,:),  % x coords of path
    Coord_in.coord(2,:),  % y coords of path
    coord_vals,           % index values (1, 2, 3, ...)
    x,                    % position x samples
    y,                    % position y samples
    'nearest'             % nearest-neighbor interpolation
)

% 2. Convert index to distance (optional)
if output_mode == 'dist':
    linpos = linspace(0, run_dist, length(Coord_in.coord))
    linpos_dist = linpos(linpos_idx)
end

% 3. Compute perpendicular distance (optional)
z_dist = distance_to_nearest_point(x, y, Coord_in.coord)
```

**Approach**: **Nearest-neighbor projection** (NOT orthogonal projection or shortest path)

**Outputs**:
- `'idx'` - Index of nearest coordinate point
- `'dist'` - Distance along linearized path
- `'z_dist'` - Perpendicular distance from track (debug mode)

**Comparison with linearization methods**:

| Method | vandermeerlab | track-linearization | neurospatial |
|--------|---------------|---------------------|--------------|
| Approach | Nearest neighbor | Shortest path graph | GraphLayout |
| Handles loops | ❌ | ✅ | ✅ |
| Handles shortcuts | ❌ | ✅ | ✅ |
| Orthogonal projection | ❌ | ✅ | ✅ |
| Multiple segments | ✅ Manual | ✅ Automatic | ✅ |

**Limitation**: Nearest-neighbor can fail at junctions and loops (no graph-based routing).

### 3.5 Replay Analysis (replay_scoring.m)

**Workflow**:

**Step 1: Detect candidate events** (Sharp-wave ripples)
```matlab
% 1. Filter LFP in ripple band (140-220 Hz)
lfp_ripple = FilterLFP(lfp, [140 220])

% 2. Compute differential power (SWR channel - control)
power_diff = power(swr_channel) - power(control_channel)

% 3. Z-score power
power_z = zscore(power_diff)

% 4. Threshold detection
candidates = (power_z > 3) & (duration > 25ms) & (max_power_z > 5)
```

**Step 2: Score replay events** (Template matching)
```matlab
% 1. Build place field templates (separately for left/right trials)
templates_left = TuningCurves(spikes_left, position_left)
templates_right = TuningCurves(spikes_right, position_right)

% 2. For each candidate event, extract spike sequence
spike_seq = spikes_during_event(candidate)

% 3. Score sequence against templates
score = scoreCandSeq(
    spike_seq,
    templates,
    method='exact',      % exact sequence matching
    nShuffles=100        % statistical comparison
)

% 4. Attach scores to candidate events
candidate.score_left = score_left
candidate.score_right = score_right
```

**Method**: **Template matching with sequence correlation**

**Key metric**: Correlation between template order and spike time order

**Comparison with replay analysis methods**:

| Package | Detection | Scoring | Method |
|---------|-----------|---------|--------|
| vandermeerlab | SWR power threshold | Template matching | Sequence correlation |
| neurocode | SWR detection | NMF, assemblies | Dimensionality reduction |
| buzcode | SWR detection | Bayesian decoding | Posterior probability |

**Unique features**:
- ✅ Differential power (SWR - control) reduces noise
- ✅ Separate left/right templates (directional replay)
- ✅ Shuffle-based statistics (null distribution)

### 3.6 Data Structures (datatypes/)

vandermeerlab defines custom data types similar to pynapple:

| vandermeerlab | pynapple | Description |
|---------------|----------|-------------|
| `ts` | `Ts` | Spike times (time series) |
| `tsd` | `Tsd` | Time series with data |
| `iv` | `IntervalSet` | Time intervals/epochs |
| `tc` | - | Time cells (spike trains) |

**Example**:
```matlab
% Load spike times
S = LoadSpikes(cfg)  % returns ts structure

% Load position
pos = LoadPos(cfg)   % returns tsd structure

% Define intervals
run_epochs = iv([start_times], [end_times])

% Restrict data to intervals
S_run = restrict(S, run_epochs)
pos_run = restrict(pos, run_epochs)
```

**Comparison**:
- vandermeerlab: MATLAB OOP with operator overloading
- pynapple: Python with NumPy/pandas backend
- neurospatial: No time-series (use pynapple) ✅

## 4. What vandermeerlab DOES NOT Provide

**Missing spatial metrics**:
1. ❌ **Skaggs information** - spatial information content
2. ❌ **Sparsity** - fraction of environment where cell fires
3. ❌ **Coherence** - correlation with neighbor average
4. ❌ **Grid score** - hexagonal periodicity measure
5. ❌ **Spatial autocorrelation** - 2D autocorrelation maps
6. ❌ **Border score** - proximity to boundaries
7. ❌ **Head direction** - circular tuning metrics

**Missing spatial primitives**:
1. ❌ **Graph-based operations** - only regular grids + linearization
2. ❌ **Irregular spatial binning** - predefined bins only
3. ❌ **Geodesic distances** - only Euclidean
4. ❌ **Differential operators** - no gradient/divergence
5. ❌ **Spatial autocorrelation on graphs**

**Missing 2D place field analysis**:
1. ❌ **2D place field detection** - only 1D (DetectPlaceCells1D.m)
2. ❌ **Subfield discrimination** - simpler than neurocode/opexebo
3. ❌ **Adaptive thresholding** - fixed threshold only

**Platform limitations**:
1. ❌ **MATLAB-only** - not accessible to Python-only users
2. ❌ **Limited documentation** - wiki is access-restricted
3. ❌ **No package manager** - manual installation

## 5. What vandermeerlab DOES Uniquely Provide

**Strengths**:

1. ✅ **Task-specific workflows** - 8 task directories with complete pipelines
2. ✅ **Practical examples** - 12 example workflows including full analysis scripts
3. ✅ **Interneuron exclusion** - maximum mean firing rate filter (unique!)
4. ✅ **Differential SWR power** - SWR channel minus control (reduces noise)
5. ✅ **Directional replay** - separate left/right templates
6. ✅ **Flexible binning** - supports irregular bin edges (unlike most packages)
7. ✅ **Simple linearization** - nearest-neighbor (fast, easy to understand)
8. ✅ **Integrated workflows** - loading → preprocessing → analysis → visualization
9. ✅ **Cell classification** - dedicated module for cell type identification
10. ✅ **Production-ready** - used in published research

## 6. Comparison with Other Packages

### 6.1 Spatial Metrics Coverage

| Metric | vandermeerlab | neurocode | opexebo | buzcode | pynapple | neurospatial |
|--------|---------------|-----------|---------|---------|----------|--------------|
| **Place Fields** |
| Tuning curves | ✅ 1D+2D | ✅ 1D+2D | ✅ 2D | ✅ 1D+2D | ✅ N-D | ❌ Planned |
| Field detection | ✅ 1D only | ✅ 1D+2D | ✅ 2D | ✅ 1D+2D | ❌ | ❌ Planned |
| Skaggs info | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ Planned |
| Peak rate | ✅ | ✅ | ✅ | ✅ | - | - |
| Field size | ✅ | ✅ | ✅ | ✅ | - | - |
| Sparsity | ❌ | ❌ | ✅ | ❌ | - | ❌ Planned |
| Coherence | ❌ | ❌ | ✅ | ❌ | - | ❌ Planned |
| Interneuron filter | ✅ Unique | ❌ | ❌ | ❌ | - | - |
| **Grid Cells** |
| Grid score | ❌ | ❌ | ✅ | ❌ | - | ❌ Planned |
| Spatial autocorr | ❌ | ❌ | ✅ FFT | ❌ | - | ❌ Planned |
| **Linearization** |
| Track linear | ✅ Nearest | ✅ Python | - | ✅ | - | ✅ GraphLayout |
| **Decoding** |
| Bayesian | ✅ Poisson | ✅ Poisson | - | ✅ Multiple | ✅ | - |
| Position recon | ✅ | ✅ | - | ✅ | - | - |
| **Replay** |
| Replay score | ✅ Template | ✅ NMF | - | ✅ | - | - |
| SWR detection | ✅ Differential | ✅ | - | ✅ | - | - |

### 6.2 Implementation Comparison

**Tuning curves algorithm**:

```matlab
% vandermeerlab (flexible binning)
tc = spike_counts / (occupancy * occ_dt)
tc(occupancy < minOcc) = NaN

% neurocode (Map.m)
map.z = map.count / (map.time + eps)

% pynapple (compute_2d_tuning_curves)
rate = spike_counts / (occupancy * dt)

% All use identical approach!
```

**Place field detection**:

```matlab
% vandermeerlab (simple threshold, 1D only)
fields = (rate_map >= threshold) & (size >= minSize) & (mean_rate <= max_meanfr)

% neurocode (iterative peak-based, 2D)
while peaks_remain:
    peak = max(rate_map)
    field = (rate_map >= 0.2 * peak)
    check_subfields()  # Sophisticated

% opexebo (adaptive thresholding, 2D)
for threshold in arange(0.96, 0.2, -0.02):
    labeled = morphology.label(rate_map > threshold)
    if stable and no_holes:
        return regions
```

**Comparison**:
- vandermeerlab: Simplest (fixed threshold, no subfield detection)
- neurocode: Most sophisticated (subfield discrimination)
- opexebo: Most robust (adaptive thresholding)

**Bayesian decoding**:

```matlab
% vandermeerlab (DecodeZ.m)
log_p = sum(Q .* log(tc)) - binsize * sum(tc)
p = exp(log_p)
p = p / sum(p)

% neurocode (ReconstructPosition.m)
P_spikes_given_x = product_i((λ_i(x) * dt)^n_i * exp(-λ_i(x) * dt) / n_i!)
P_x_given_spikes = P_spikes_given_x / sum(P_spikes_given_x)

% pynapple (decode_2d)
likelihood = poisson.pmf(spikes, rates * dt)
posterior = likelihood / sum(likelihood)

% Identical Poisson Bayesian approach across all packages!
```

### 6.3 Language and Platform

| Package | Language | Platform | Time-series | Spatial Discretization | Documentation |
|---------|----------|----------|-------------|------------------------|---------------|
| vandermeerlab | MATLAB | MATLAB only | ✅ ts, tsd, iv | Regular grids + 1D linearization | Wiki (restricted) |
| neurocode | MATLAB | MATLAB only | ✅ IntervalArray | Regular grids only | Minimal |
| opexebo | Python | Cross-platform | ❌ | Regular grids only | Excellent |
| buzcode | MATLAB | MATLAB only | ✅ | 1D/2D regular grids | Good |
| pynapple | Python | Cross-platform | ✅ Excellent | ❌ | Excellent |
| neurospatial | Python | Cross-platform | ❌ | ✅ Any graph | Good |

## 7. Strategic Positioning

### 7.1 Complementary Ecosystem

vandermeerlab fits into the broader ecosystem as:

| Package | Role | Unique Value | Limitation |
|---------|------|--------------|------------|
| **vandermeerlab** | Practical MATLAB workflows | Task-specific pipelines, flexible binning, interneuron filter | MATLAB-only, 1D place fields, no metrics |
| **neurocode** | Full MATLAB pipeline | Comprehensive, subfield detection, circular stats | MATLAB-only, regular grids |
| **opexebo** | Python spatial metrics | Nobel Prize validation, comprehensive | Regular grids only |
| **buzcode** | MATLAB preprocessing | Full pipeline, multiple decoding | MATLAB-only, minimal metrics |
| **pynapple** | Python time-series | Excellent time infrastructure | No spatial discretization |
| **neurospatial** | Python spatial primitives | Any graph topology, differential ops | No time-series (use pynapple) |

### 7.2 User Personas

**Choose vandermeerlab if**:
- ✅ You work in MATLAB
- ✅ You need task-specific workflows (T-maze, linear track, square maze)
- ✅ You need practical, working examples (12 workflows)
- ✅ You want flexible bin edges (irregular binning)
- ✅ You need to exclude interneurons (max mean firing rate)
- ✅ You analyze directional replay (left/right templates)
- ✅ You prefer simple linearization (nearest-neighbor)

**Choose neurospatial if**:
- ✅ You work in Python
- ✅ You have irregular environments (complex mazes, obstacles)
- ✅ You need graph-based spatial primitives
- ✅ You need 2D place field detection
- ✅ You need spatial metrics (Skaggs info, sparsity, coherence)
- ✅ You need grid cell analysis (grid score, autocorrelation)

**Use both if**:
- ✅ Prototype in vandermeerlab (fast MATLAB workflows)
- ✅ Then switch to neurospatial+pynapple for advanced analysis
- ✅ Validate decoding against vandermeerlab (identical Poisson approach)

### 7.3 Ideal Workflow

```python
# 1. Prototype and explore data (vandermeerlab MATLAB)
% Quick tuning curves and visualization
tc = TuningCurves(S, pos, cfg);
PlotFields(tc);

% Detect place cells (1D)
[template_idx, peak_idx] = DetectPlaceCells1D(tc, cfg);

% Bayesian decoding
posterior = DecodeZ(Q, tc, cfg);

# 2. Export to Python for advanced analysis
# 3. Load in pynapple for time-series
import pynapple as nap

spikes = nap.Ts(t=spike_times)
position = nap.Tsd(t=time, d=coords)
epochs = nap.IntervalSet(start=starts, end=ends)

# 4. Spatial discretization (neurospatial)
from neurospatial import Environment

env = Environment.from_samples(position.values, bin_size=2.0)
env.units = "cm"

# 5. Advanced spatial metrics (neurospatial - once implemented)
from neurospatial.metrics import (
    detect_place_fields,  # 2D detection (not in vandermeerlab)
    skaggs_information,    # Not in vandermeerlab
    sparsity,              # Not in vandermeerlab
    coherence,             # Not in vandermeerlab
    grid_score,            # Not in vandermeerlab
)

fields_2d = detect_place_fields(firing_rate, env)
info = skaggs_information(firing_rate, occupancy)
sparse = sparsity(firing_rate, occupancy)
coh = coherence(firing_rate, env)

# 6. Validate decoding against vandermeerlab
# (identical Poisson approach, should match exactly)
```

## 8. Impact on neurospatial Implementation Plan

### 8.1 Algorithm Validation

vandermeerlab provides additional validation:

**✅ Tuning curves confirmed**:
```matlab
tc = spike_counts / (occupancy * occ_dt)
```
Identical to neurocode, pynapple, and standard approach.

**✅ Bayesian decoding confirmed**:
- Poisson likelihood model (field standard)
- Log-space computation (numerical stability)
- Matches neurocode, buzcode, pynapple implementations

**✅ Linearization approach confirmed**:
- Nearest-neighbor is valid but limited (no graph routing)
- GraphLayout (neurospatial) is more sophisticated ✅

### 8.2 New Insights

**Interneuron exclusion**:
- vandermeerlab uses **maximum mean firing rate** filter
- This is MISSING from all other packages
- **Recommendation**: Add to Phase 4.1 (place field detection)

**Flexible binning**:
- vandermeerlab supports **irregular bin edges**
- Most packages require regular grids
- neurospatial ALREADY supports irregular graphs ✅

**Differential SWR power**:
- vandermeerlab uses **SWR channel - control channel**
- Reduces noise from muscle artifacts
- **Recommendation**: Document this approach (not core to neurospatial)

### 8.3 No Changes to Core Plan

The implementation plan remains unchanged:

| Phase | Timeline | Risk | Status |
|-------|----------|------|--------|
| Phase 1: Differential operators | 3 weeks | Low | Validated |
| Phase 2: Signal processing | 6 weeks | Medium | Validated |
| Phase 3: Path operations | 1 week | Low | Validated |
| Phase 4: Metrics module | 2 weeks | Low | Validated |
| Phase 5: Polish | 2 weeks | Low | Validated |

**Minor addition**:
- Phase 4.1: Add `max_meanfr` parameter to `detect_place_fields()` (interneuron exclusion)

**No timeline change**: This is a trivial addition (<1 hour)

## 9. Key Takeaways

### 9.1 What vandermeerlab Validates

✅ **Algorithms**: Tuning curves, Bayesian decoding (Poisson), replay template matching
✅ **Data structures**: ts, tsd, iv are essential (matches pynapple)
✅ **Workflows**: Task-specific analyses are valuable for users
✅ **Flexibility**: Irregular bin edges are important (neurospatial already supports via graphs)

### 9.2 What vandermeerlab Lacks

❌ **Python implementation**: MATLAB-only limits accessibility
❌ **2D place field detection**: Only 1D (DetectPlaceCells1D.m)
❌ **Spatial metrics**: No Skaggs info, sparsity, coherence
❌ **Grid cell metrics**: No grid score, spatial autocorrelation
❌ **Subfield detection**: Simpler than neurocode/opexebo
❌ **Documentation**: Wiki is access-restricted

### 9.3 What neurospatial Should Adopt

1. ✅ **Interneuron exclusion**: Add `max_meanfr` parameter (vandermeerlab)
2. ✅ **Flexible binning**: ALREADY HAVE via irregular graphs
3. ✅ **Time-series integration**: Use pynapple (matches vandermeerlab's ts/tsd/iv)
4. ✅ **Task-specific examples**: Add example workflows (Phase 5)

### 9.4 Strategic Positioning Confirmed

**vandermeerlab confirms that neurospatial fills a unique gap**:

1. ✅ **Python implementation** (vandermeerlab is MATLAB-only)
2. ✅ **2D place field detection** (vandermeerlab is 1D only)
3. ✅ **Spatial metrics** (vandermeerlab lacks Skaggs info, sparsity, coherence)
4. ✅ **Grid cell metrics** (vandermeerlab lacks this)
5. ✅ **Graph-based spatial analysis** (vandermeerlab uses nearest-neighbor linearization)
6. ✅ **Integration with pynapple** (vandermeerlab is MATLAB ecosystem)

**The combination of neurospatial + pynapple provides a Python alternative to vandermeerlab with additional capabilities for 2D spatial analysis, spatial metrics, and graph-based primitives.**

## 10. Unique Contributions from vandermeerlab

### 10.1 Interneuron Exclusion ⭐

**Algorithm**:
```matlab
% Exclude cells with high mean firing rate (likely interneurons)
mean_rate = total_spikes / total_time
is_place_cell = (peak_rate >= 5) & (mean_rate <= 10)
```

**Why important**:
- Pyramidal cells (place cells): 0.5-5 Hz mean firing rate
- Interneurons: 10-50 Hz mean firing rate
- This filter is MISSING from all other packages ✅

**Recommendation**: Add to neurospatial `detect_place_fields()`:
```python
def detect_place_fields(
    firing_rate: NDArray,
    env: Environment,
    *,
    threshold: float = 0.2,
    min_size: float | None = None,
    max_mean_rate: float | None = 10.0,  # NEW from vandermeerlab
    detect_subfields: bool = True,
) -> list[NDArray[np.int64]]:
    """
    Detect place fields with optional interneuron exclusion.

    Parameters
    ----------
    max_mean_rate : float, optional
        Maximum mean firing rate (Hz). Cells exceeding this are excluded
        as likely interneurons. Default 10 Hz (vandermeerlab).
        Set to None to disable.
    """
    mean_rate = np.mean(firing_rate)
    if max_mean_rate is not None and mean_rate > max_mean_rate:
        return []  # Exclude as interneuron

    # ... rest of detection
```

### 10.2 Flexible Binning ⭐

**vandermeerlab supports irregular bin edges**:
```matlab
cfg.binEdges = [0, 2, 5, 10, 20, 50, 100];  % Irregular spacing
tc = TuningCurves(S, pos, cfg);
```

**Most packages require regular grids**:
- opexebo: Regular 2D grids only
- neurocode: Regular grids only
- buzcode: Regular grids only

**neurospatial ALREADY supports this** via:
1. Irregular graphs (arbitrary connectivity)
2. Any bin shape/size
3. Variable resolution

✅ **No action needed** - neurospatial is already more flexible!

### 10.3 Differential SWR Power ⭐

**Algorithm**:
```matlab
% Compute differential power (reduces muscle artifacts)
power_swr = power(swr_channel)
power_control = power(control_channel)
power_diff = power_swr - power_control

% Z-score differential power
power_diff_z = zscore(power_diff)

% Detect events
events = (power_diff_z > 3) & (duration > 25ms)
```

**Why important**:
- SWR channel picks up ripples + noise
- Control channel picks up noise only
- Difference isolates ripples (reduces false positives)

**Recommendation**: Document this approach in examples (not core to neurospatial).

## 11. Conclusion

vandermeerlab is a practical MATLAB toolkit with:
- ✅ Task-specific workflows (8 tasks, 12 examples)
- ✅ Flexible binning (irregular bin edges)
- ✅ Interneuron exclusion (unique!)
- ✅ Differential SWR detection (noise reduction)
- ✅ Simple linearization (nearest-neighbor)
- ✅ Production-ready code (used in publications)

However, it is **limited to MATLAB, 1D place fields, and lacks spatial metrics**, leaving a gap for:
- ❌ Python-native implementation
- ❌ 2D place field detection
- ❌ Spatial metrics (Skaggs info, sparsity, coherence)
- ❌ Grid cell metrics (grid score, autocorrelation)
- ❌ Subfield detection (simpler than neurocode/opexebo)

**neurospatial + pynapple together provide a Python alternative to vandermeerlab with unique capabilities for 2D spatial analysis, spatial metrics, and graph-based primitives.**

The implementation plan is validated. Minor addition: `max_mean_rate` parameter for interneuron exclusion (<1 hour, no timeline change).

---

**Package comparison summary** (6 packages analyzed):

| Package | Language | Platform | Unique Value | Limitation |
|---------|----------|----------|--------------|------------|
| **vandermeerlab** | MATLAB | MATLAB | Task workflows, interneuron filter | 1D only, no metrics |
| **neurocode** | MATLAB | MATLAB | Subfield detection, circular stats | Regular grids |
| **opexebo** | Python | Cross | Nobel validation, comprehensive | Regular grids |
| **buzcode** | MATLAB | MATLAB | Full pipeline, multiple decoding | Minimal metrics |
| **pynapple** | Python | Cross | Time-series excellence | No spatial discretization |
| **neurospatial** | Python | Cross | Any graph, differential ops | No time-series (use pynapple) |

**neurospatial fills the unique gap for Python-based, graph-general spatial analysis.**
