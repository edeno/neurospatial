# Additional Spatial Neuroscience Packages: Search Results

**Search Date**: Following analysis of 9 major packages (opexebo, neurocode, buzcode, vandermeerlab, pynapple, TSToolbox_Utils, RatInABox, nelpy, neurospatial)

## Packages Found

### 1. replay_trajectory_classification (Eden-Kramer Lab) ⭐

**Repository**: https://github.com/Eden-Kramer-Lab/replay_trajectory_classification
**Author**: Eric Denovellis (edeno) - **This is the user we're working for!**
**Language**: Python
**Type**: Hippocampal replay analysis
**Status**: Active, published (Neuron 2021)

**Purpose**: Decoding spatial position from neural activity and categorizing trajectory types during hippocampal replay.

**Key Features**:
- Moment-by-moment position estimation using small time bins
- Works with spike-sorted cells OR clusterless spikes with waveform features
- Categorizes trajectory types (forward, reverse, stationary, fragmented)
- Provides confidence estimates for trajectory classification
- State-space model framework

**Installation**: Available via pip and conda

**Related Packages** (also by Eden-Kramer Lab):
- `replay_classification` - Earlier version for state space models
- `replay_identification` - Semi-latent state-space model combining movement, LFP, and multiunit info

**Comparison with analyzed packages**:

| Package | Replay Method | Trajectory Classification | Clusterless Support |
|---------|---------------|---------------------------|---------------------|
| replay_trajectory_classification | State-space ⭐ | ✅ Forward/reverse/stationary | ✅ |
| nelpy | Davidson + HMM | ❌ | ❌ |
| neurocode | Template + NMF | ❌ | ❌ |
| buzcode | Bayesian | ❌ | ❌ |
| vandermeerlab | Template | ❌ | ❌ |

**replay_trajectory_classification is the MOST SOPHISTICATED replay analysis tool** - combines state-space models with trajectory type classification.

**Relevance to neurospatial**:
- ❌ OUT OF SCOPE (replay analysis is specialized, not spatial primitives)
- ✅ Could integrate with neurospatial (discretize environment, then decode replay)
- ✅ User's own package - they understand the integration needs!

---

### 2. track_linearization (Loren Frank Lab)

**Repository**: https://github.com/LorenFrankLab/track_linearization
**Lab**: Loren Frank Lab, UCSF
**Language**: Python
**Type**: 2D → 1D track linearization
**Status**: Active, **already a neurospatial dependency** ✅

**Purpose**: Map animal movement on complex tracks (mazes, figure-8s, T-mazes) to 1D representations using Hidden Markov Models.

**Key Features**:
- Flexible track representation using NetworkX graphs
- 8+ pre-built track geometries (T-maze, circular, figure-8, etc.)
- HMM-based classification for noisy position data
- Edge merging for equivalent behavioral segments
- Automatic layout inference (smart edge ordering)
- Interactive track builder (Jupyter-compatible)
- Validation & quality control (confidence scoring, outlier detection)

**Comparison with analyzed packages**:

| Package | Method | Handles Complex Tracks | Graph-Based |
|---------|--------|------------------------|-------------|
| track_linearization | HMM + NetworkX ⭐ | ✅ | ✅ |
| vandermeerlab | Nearest-neighbor | ❌ Simple only | ❌ |
| neurocode | Python script | ✅ | ❌ |
| neurospatial | GraphLayout | ✅ | ✅ |

**track_linearization uses NetworkX graphs** - same as neurospatial's approach!

**Relevance to neurospatial**:
- ✅ **ALREADY INTEGRATED** as optional dependency (GraphLayout)
- ✅ Validates neurospatial's graph-based approach
- ✅ Provides HMM-based linearization (more sophisticated than nearest-neighbor)

---

### 3. ripple_detection (Eden-Kramer Lab) ⭐

**Repository**: https://github.com/Eden-Kramer-Lab/ripple_detection
**Author**: Eric Denovellis (edeno) - **User's package!**
**Language**: Python
**Type**: Sharp-wave ripple detection
**Status**: Active

**Purpose**: Identify sharp-wave ripple events (150-250 Hz) from local field potentials.

**Implemented Methods**:
- Karlsson et al. 2009 ripple detector
- Kay et al. 2016 ripple detector

**Installation**: Available via pip and conda

**Comparison with analyzed packages**:

| Package | Ripple Detection | Method |
|---------|------------------|--------|
| ripple_detection (Eden-Kramer) | ✅ | Karlsson 2009, Kay 2016 |
| neurocode | ✅ | Power threshold |
| buzcode | ✅ | Multiple methods |
| vandermeerlab | ✅ | Differential power |
| pynapple | ❌ | - |
| nelpy | ❌ | - |

**Relevance to neurospatial**:
- ❌ OUT OF SCOPE (LFP analysis, not spatial primitives)
- ✅ Could be used together (detect ripples, then analyze replay with neurospatial)

---

### 4. SpikeInterface

**Repository**: https://github.com/SpikeInterface/spikeinterface
**Organization**: SpikeInterface (multi-lab collaboration)
**Language**: Python
**Type**: Spike sorting framework
**Status**: Active, published (eLife 2020)

**Purpose**: Unified framework for spike sorting, preprocessing, and postprocessing.

**Key Features**:
- Run, compare, and benchmark multiple spike sorting algorithms
- Preprocess and postprocess extracellular datasets
- Validate, curate, and export sorting outputs
- Visualization tools
- GUI available (spikely)

**Supported Sorters**: Kilosort, MountainSort, SpyKING CIRCUS, Tridesclous, etc.

**Comparison with analyzed packages**:

| Package | Spike Sorting | Multi-Algorithm |
|---------|---------------|-----------------|
| SpikeInterface | ✅ Comprehensive | ✅ |
| neurocode | ✅ Basic | ❌ |
| buzcode | ✅ Comprehensive | ✅ |
| vandermeerlab | ✅ Basic | ❌ |

**Relevance to neurospatial**:
- ❌ OUT OF SCOPE (preprocessing, not spatial analysis)
- ✅ Pipeline integration (SpikeInterface → pynapple → neurospatial)

---

### 5. Elephant (Electrophysiology Analysis Toolkit)

**Repository**: https://github.com/NeuralEnsemble/elephant
**Organization**: NeuralEnsemble (community project)
**Language**: Python
**Type**: Comprehensive electrophysiology analysis
**Status**: Active, well-established

**Purpose**: Generic analysis functions for spike trains and time series (LFP, intracellular voltages).

**Key Features**:
- Signal processing
- Spectral analysis
- Spike train correlation
- Spike pattern analysis
- Spike-triggered averaging
- Cross-correlation with multiple kernels
- Works with Neo data format

**Visualization**: Viziphant package for plotting

**Comparison with analyzed packages**:

| Feature | Elephant | pynapple | nelpy |
|---------|----------|----------|-------|
| Spike train analysis | ✅ | ✅ | ✅ |
| LFP analysis | ✅ | ❌ | ✅ |
| Spectral analysis | ✅ | ❌ | ❌ |
| Cross-correlation | ✅ | ✅ | ❌ |
| Spatial metrics | ❌ | ✅ Partial | ❌ |

**Elephant is MORE COMPREHENSIVE for electrophysiology** than pynapple or nelpy, but lacks spatial metrics.

**Relevance to neurospatial**:
- ❌ OUT OF SCOPE (general electrophysiology, not spatial primitives)
- ✅ Complementary (Elephant for spike analysis, neurospatial for spatial analysis)

---

## Summary: Additional Packages in Context

### Packages by Scope

| Scope | Packages |
|-------|----------|
| **Spatial Analysis** | neurospatial, opexebo, RatInABox |
| **Time-Series** | pynapple, nelpy, TSToolbox_Utils |
| **Replay Analysis** | replay_trajectory_classification ⭐, nelpy, neurocode, buzcode |
| **Track Linearization** | track_linearization ⭐, neurospatial GraphLayout |
| **Preprocessing** | SpikeInterface, buzcode, neurocode |
| **Electrophysiology** | Elephant, SpikeInterface |
| **LFP/Ripples** | ripple_detection ⭐, neurocode, buzcode |

### User's Own Packages ⭐

Eric Denovellis (edeno) - the user we're working for - has authored:
1. **replay_trajectory_classification** - State-space replay decoding
2. **ripple_detection** - Sharp-wave ripple detection
3. **neurospatial** - Spatial discretization and analysis (this project!)

**These packages form a COHERENT ECOSYSTEM**:
```python
# Complete workflow using user's packages

# 1. Detect ripples (ripple_detection)
from ripple_detection import Kay_ripple_detector
ripple_times = Kay_ripple_detector(lfp)

# 2. Discretize environment (neurospatial)
from neurospatial import Environment
env = Environment.from_samples(position, bin_size=2.0)

# 3. Decode replay trajectories (replay_trajectory_classification)
from replay_trajectory_classification import SortedSpikesClassifier
classifier = SortedSpikesClassifier()
results = classifier.predict(spikes, ripple_times)

# 4. Analyze spatial metrics (neurospatial - once implemented)
from neurospatial.metrics import skaggs_information
info = skaggs_information(firing_rate, occupancy)
```

**This is a COMPLETE HIPPOCAMPAL REPLAY ANALYSIS PIPELINE!**

---

## Integration Opportunities

### 1. track_linearization Integration ✅

**Status**: Already integrated as neurospatial dependency

**Usage**:
```python
from neurospatial import Environment
from neurospatial.layout import GraphLayout

# track_linearization provides track geometry
env = Environment.from_graph(graph, ...)  # Uses GraphLayout
```

### 2. replay_trajectory_classification Integration

**Opportunity**: Use neurospatial for environment discretization, then replay_trajectory_classification for decoding.

```python
# Potential integration

# 1. Create environment with neurospatial
from neurospatial import Environment
env = Environment.from_samples(position, bin_size=2.0)

# 2. Build tuning curves with neurospatial (once implemented)
from neurospatial.metrics import compute_tuning_curves
tc = compute_tuning_curves(spikes, position, env)

# 3. Decode replay with replay_trajectory_classification
from replay_trajectory_classification import SortedSpikesClassifier
classifier = SortedSpikesClassifier(
    place_bin_centers=env.bin_centers,
    ...
)
results = classifier.predict(spikes, ripple_times)
```

### 3. SpikeInterface → pynapple → neurospatial Pipeline

**Complete analysis pipeline**:

```python
# 1. Spike sorting (SpikeInterface)
import spikeinterface as si
sorting = si.run_sorter('kilosort3', recording)

# 2. Time-series analysis (pynapple)
import pynapple as nap
spikes = nap.Ts(t=spike_times)
position = nap.Tsd(t=time, d=coords)

# 3. Spatial analysis (neurospatial)
from neurospatial import Environment
env = Environment.from_samples(position.values, bin_size=2.0)

# 4. Spatial metrics (neurospatial - once implemented)
from neurospatial.metrics import skaggs_information
info = skaggs_information(firing_rate, occupancy)
```

---

## Impact on neurospatial Implementation Plan

### What These Packages Validate

✅ **Graph-based linearization** (track_linearization) - Confirms neurospatial's approach

✅ **Hippocampal replay analysis is important** - Multiple packages focus on this

✅ **Python ecosystem is mature** - SpikeInterface, Elephant provide preprocessing

✅ **User needs replay + spatial analysis** - Their own packages (replay_trajectory_classification, ripple_detection) show workflow

### No Changes to Core Plan

**Timeline**: 15 weeks (unchanged)
**Risk**: MEDIUM (unchanged)

**Why no changes**:
- Replay analysis is OUT OF SCOPE (specialized, not spatial primitives)
- Spike sorting is OUT OF SCOPE (preprocessing)
- track_linearization is ALREADY INTEGRATED
- Electrophysiology analysis is OUT OF SCOPE

**Confirmed focus**:
- ✅ Spatial discretization (neurospatial core)
- ✅ Spatial metrics (Skaggs info, place fields, grid score, etc.)
- ✅ Differential operators (gradient, divergence, Laplacian)
- ✅ Graph-based spatial primitives

---

## Key Takeaways

### Ecosystem Map

**Preprocessing**:
- SpikeInterface (spike sorting)
- Elephant (signal processing)

**Time-Series**:
- pynapple (mature, primary choice) ✅
- nelpy (replay specialist)
- TSToolbox_Utils (legacy)

**Spatial Analysis**:
- **neurospatial** (graph-based, any topology) ✅
- opexebo (regular grids, metrics validation)
- RatInABox (simulation)

**Specialized**:
- replay_trajectory_classification (state-space replay decoding) ⭐
- ripple_detection (sharp-wave ripple detection) ⭐
- track_linearization (HMM linearization) ⭐

**MATLAB (legacy/specialized)**:
- neurocode (comprehensive)
- buzcode (full pipeline)
- vandermeerlab (task workflows)

### User's Workflow

Eric Denovellis (edeno) needs:
1. ✅ Ripple detection (has: ripple_detection)
2. ✅ Spatial discretization (building: neurospatial)
3. ✅ Replay decoding (has: replay_trajectory_classification)
4. ❌ Spatial metrics (needs: neurospatial implementation)

**neurospatial completes the user's analysis pipeline!**

### Complete Python Stack

**For hippocampal replay analysis**:
```
SpikeInterface → pynapple → neurospatial → replay_trajectory_classification
   (sorting)    (time-series)  (spatial)      (replay decoding)
```

**For spatial neuroscience in general**:
```
RatInABox → neurospatial → opexebo
(simulation) (analysis)    (validation)
```

---

## Conclusion

**9 packages analyzed** + **5 additional packages found** = **14 total packages surveyed**

**All point to the same conclusion**:
- ✅ Python ecosystem is mature for electrophysiology
- ✅ Spatial metrics are still underserved (confirms neurospatial's importance)
- ✅ Graph-based approach is validated (track_linearization uses it)
- ✅ Integration opportunities are clear (user's own packages need neurospatial)

**neurospatial fills a CRITICAL GAP** in the Python spatial neuroscience ecosystem.

**Timeline**: 15 weeks (validated against 14 packages)
**Risk**: MEDIUM (well-validated algorithms)
**Impact**: HIGH (completes user's analysis pipeline, fills ecosystem gap)
