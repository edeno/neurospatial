# nelpy Analysis: Neuroelectrophysiology Data Analysis

**Package**: [nelpy/nelpy](https://github.com/nelpy/nelpy)
**Lab**: Independent project (inspired by van der Meer lab's python-vdmlab → nept)
**Language**: Python (77.1% Jupyter Notebook, 22.9% Python)
**Type**: ANALYSIS package for neuroelectrophysiology data
**Status**: Active development (1,041 commits, 50 stars, 99 open issues)
**Initial Release**: December 2016

## 1. Critical Assessment: Mature vs Aspirational

**nelpy is PARTIALLY IMPLEMENTED** - many features are **design templates** rather than working code.

### What Actually Works ✅

1. **Data structures** (core infrastructure)
   - `EventArray` (spike trains)
   - `AnalogSignalArray` (continuous signals)
   - `IntervalArray` (epochs/intervals)
   - `ValEventArray` (valued events)
   - `Coordinates` (spatial positions)

2. **Bayesian decoding** (decoding.py)
   - `decode1D()` - 1D position reconstruction
   - `decode2D()` - 2D position reconstruction
   - `BayesianDecoder` - sklearn-compatible wrapper
   - Cross-validation utilities

3. **Replay analysis** (analysis/replay.py)
   - Davidson et al. 2009 trajectory scoring
   - HMM-based replay detection
   - Multiple shuffle methods for null distributions
   - Time-resolved and cumulative scoring

4. **HMM utilities** (hmmutils.py)
   - Hidden Markov Model analysis
   - Sparsity-constrained HMMs

5. **Visualization** (plotting/)
   - Spike rasters
   - Analog signal plots
   - Support plots (epoch display)

### What Does NOT Work ❌

1. **estimators.py** - Mostly `NotImplementedError`:
   - `RateMap.fit()` - NOT implemented
   - `RateMap.predict()` - NOT implemented
   - `NDRateMap` methods - NOT implemented
   - GLM, GVM, GP firing rate estimation - NOT implemented

2. **Spatial metrics** - NOT found:
   - No place field detection
   - No spatial information (Skaggs)
   - No grid score
   - No coherence, sparsity
   - No border score

3. **Analysis directory** - MINIMAL:
   - Only 4 files (ergodic.py, hmm_sparsity.py, replay.py, __init__.py)
   - Most spatial analyses missing

**Conclusion**: nelpy is a **WORK-IN-PROGRESS** with strong time-series infrastructure and replay analysis, but **limited spatial metrics**.

## 2. Package Overview

nelpy (**N**euro**el**ectro**p**hysiolog**y**) provides:

**Core philosophy**:
- "Simpler and less comprehensive than neo" (neuralensemble.org)
- Emphasizes **binned spike trains** as first-class objects
- Incorporates **support** (functional domain of definition) as core feature
- Prioritizes **ease of use** over comprehensive coverage

**Inspiration**:
- python-vdmlab (later renamed nept) from Dartmouth's van der Meer lab
- neuralensemble.org NEO project

## 3. Module Structure

```
nelpy/
├── core/
│   ├── _analogsignalarray.py    # AnalogSignalArray (continuous signals)
│   ├── _eventarray.py            # EventArray (spike trains)
│   ├── _intervalarray.py         # IntervalArray (epochs)
│   ├── _valeventarray.py         # ValEventArray (valued events)
│   └── _coordinates.py           # Coordinates (spatial positions)
├── analysis/
│   ├── replay.py                 # Replay analysis ✅
│   ├── hmm_sparsity.py           # HMM with sparsity
│   └── ergodic.py                # Ergodic analysis
├── decoding.py                   # Bayesian decoding ✅
├── estimators.py                 # Rate maps (MOSTLY NOT IMPLEMENTED ❌)
├── hmmutils.py                   # HMM utilities ✅
├── filtering.py                  # Filtering methods
├── preprocessing.py              # Data preparation
├── scoring.py                    # Evaluation metrics
├── plotting/                     # Visualization ✅
├── io/                           # Data I/O
├── synthesis/                    # Data synthesis
└── utils_/                       # Utilities
```

## 4. Data Structures (Core Infrastructure)

### 4.1 Time-Series Objects

**nelpy vs pynapple comparison**:

| Data Type | nelpy | pynapple | Equivalent |
|-----------|-------|----------|------------|
| Spike times | `EventArray` | `Ts` | ✅ Same concept |
| Time series with data | `AnalogSignalArray` | `Tsd` | ✅ Same concept |
| Time intervals | `IntervalArray` | `IntervalSet` | ✅ Same concept |
| Valued events | `ValEventArray` | - | ⭐ Unique to nelpy |
| Coordinates | `Coordinates` | - | ⭐ Unique to nelpy |

**Unique features in nelpy**:

1. **Support** (functional domain):
   ```python
   # nelpy emphasizes "support" - the temporal domain of validity
   st = nel.EventArray(spike_times, support=epoch)
   st.support  # Returns IntervalArray defining valid time range
   ```

2. **ValEventArray** - Events with values:
   ```python
   # Spikes with amplitudes, LFP events with phase, etc.
   va = nel.ValEventArray(times, values)
   ```

3. **Coordinates** - Explicit spatial position object:
   ```python
   # Dedicated object for position tracking
   coords = nel.Coordinates(x, y, t)
   ```

### 4.2 Comparison with pynapple

**Similarities**:
- Both provide spike trains (`EventArray` / `Ts`)
- Both provide time series with data (`AnalogSignalArray` / `Tsd`)
- Both provide time intervals (`IntervalArray` / `IntervalSet`)
- Both emphasize **time as primary index**

**Differences**:

| Feature | nelpy | pynapple |
|---------|-------|----------|
| **Maturity** | In-development (99 issues) | Mature (production-ready) |
| **Documentation** | "Bare-bones", work-in-progress | Excellent, comprehensive |
| **Spatial metrics** | Minimal (mostly unimplemented) | ✅ Skaggs info, tuning curves |
| **Community** | Small (50 stars) | Large (300+ stars) |
| **Maintenance** | Independent | Simons Foundation |
| **Unique features** | Support, ValEventArray, Coordinates | pandas backend, xarray tuning curves |

**Conclusion**: **pynapple is MORE MATURE** than nelpy for production use.

## 5. Bayesian Decoding (WORKING ✅)

### 5.1 Algorithm

```python
# nelpy implements standard Poisson Bayesian decoding

# Posterior computation:
posterior = exp(spike_counts * log(ratemap) + evidence_term - logsumexp)

# where:
# spike_counts: observed spikes in time bin
# ratemap: tuning curves (firing rates in Hz)
# evidence_term: -ratemap.sum(axis=0) * bin_duration * weight
# logsumexp: normalization factor (log-sum-exp trick)

# Position estimates:
# Mode: argmax(posterior)  # Maximum likelihood
# Mean: sum(posterior * bin_centers)  # Expected value
```

**Features**:
- Log-domain computation (numerical stability)
- Supports 1D and 2D decoding
- sklearn-compatible `BayesianDecoder` class
- Returns mode path, mean path, and full posterior

**Comparison**:

| Package | Decoding | Dimension | Likelihood |
|---------|----------|-----------|------------|
| nelpy | ✅ Bayesian | 1D + 2D | Poisson |
| pynapple | ✅ Bayesian | N-D | Poisson |
| vandermeerlab | ✅ Bayesian | 1D + 2D | Poisson |
| neurocode | ✅ Bayesian | 1D + 2D | Poisson |
| buzcode | ✅ Bayesian | 1D + 2D | Poisson + Phase + GLM |
| TSToolbox_Utils | ✅ Bayesian | 1D only | Poisson |

**ALL USE IDENTICAL POISSON LIKELIHOOD** ✅ (field standard confirmed again)

### 5.2 Cross-Validation

```python
# nelpy provides k-fold cross-validation utilities

from nelpy.decoding import k_fold_cross_validation

# Generate train/test splits
train_idx, test_idx = k_fold_cross_validation(n_samples, k=5)

# Evaluate decoding error
from nelpy.decoding import cumulative_dist_decoding_error_using_xval

error = cumulative_dist_decoding_error_using_xval(
    bst, position, k=5, method='1d'
)
```

**This is more sophisticated than most packages** - explicit cross-validation support for decoding evaluation.

## 6. Replay Analysis (WORKING ✅)

### 6.1 Davidson et al. 2009 Method

**Trajectory scoring**:

```python
# Score how well decoded activity follows a spatial trajectory

from nelpy.analysis.replay import score_Davidson_final_bst

score = score_Davidson_final_bst(
    posterior,        # Decoded posterior (n_bins, n_time)
    w=0.05,          # Trajectory band width (meters)
    n_samples=1000   # Number of random lines to sample
)

# Algorithm:
# 1. Sample random lines (parameterized by angle phi, distance rho)
# 2. For each line, compute fraction of posterior mass within band w
# 3. Return score for best-fitting line
```

**Linear regression**:

```python
# Fit line through decoded position

from nelpy.analysis.replay import linregress_bst

slope, intercept, r_value, p_value = linregress_bst(posterior)
```

### 6.2 HMM-Based Replay Detection

**Cross-validated HMM scoring**:

```python
from nelpy.analysis.replay import score_hmm_events

score = score_hmm_events(
    bst,                    # BinnedSpikeTrain
    position,               # Position data
    n_states=20,            # Number of HMM states
    n_shuffles=100,         # Shuffle iterations
    shuffle_type='time_swap'  # Shuffle method
)

# Shuffle methods:
# - 'time_swap': Randomize time bins
# - 'pooled_time_swap': Shuffle across entire dataset
# - 'column_cycle': Rotate posterior columns
# - 'transmat': Shuffle transition matrix
# - 'unit_id': Randomize unit assignments
```

**Time-resolved scoring**:

```python
from nelpy.analysis.replay import score_hmm_time_resolved

score_per_bin = score_hmm_time_resolved(
    bst, position, n_states=20
)
```

### 6.3 Significance Testing

**Monte Carlo p-values**:

```python
from nelpy.analysis.replay import get_significant_events

significant_events = get_significant_events(
    score_real,         # Real event scores
    score_shuffled,     # Shuffled null distribution
    q=95,               # Percentile threshold
    min_consecutive=3   # Minimum consecutive significant bins
)
```

**Comparison with other packages**:

| Package | Replay Method | Significance |
|---------|---------------|--------------|
| nelpy | Davidson + HMM ✅ | Monte Carlo shuffle |
| neurocode | Template matching + NMF | Shuffle-based |
| buzcode | Bayesian decode | Template correlation |
| vandermeerlab | Template matching | Shuffle-based |

**nelpy has MOST SOPHISTICATED replay analysis** ⭐ (Davidson + HMM + multiple shuffle methods)

## 7. What nelpy Does NOT Provide

### 7.1 Spatial Metrics (NOT IMPLEMENTED ❌)

```python
# estimators.py has skeleton code but NotImplementedError

from nelpy.estimators import RateMap

rm = RateMap()
rm.fit(bst, position)  # NotImplementedError ❌

# The following are MISSING:
# - Place field detection
# - Spatial information (Skaggs)
# - Grid score
# - Coherence
# - Sparsity
# - Border score
# - Head direction score
```

**Why this matters**: Users expect spatial metrics in a neuroelectrophysiology package, but nelpy doesn't provide them.

### 7.2 Comparison with pynapple

| Feature | nelpy | pynapple |
|---------|-------|----------|
| Skaggs information | ❌ | ✅ |
| Tuning curves | ❌ | ✅ |
| 2D tuning curves | ❌ | ✅ |
| Cross-correlograms | ❌ | ✅ |
| Perievent | ❌ | ✅ |
| Ripple detection | ❌ | ❌ |
| Place field detection | ❌ | ❌ |

**pynapple provides MORE spatial analysis tools** than nelpy.

## 8. Strategic Positioning

### 8.1 nelpy vs pynapple

**Historical context**:
- nelpy: Inspired by python-vdmlab/nept (van der Meer lab)
- pynapple: Fork from neuroseries (Peyrache lab)
- **NO DIRECT RELATIONSHIP** (independent projects)

**Current status**:

| Aspect | nelpy | pynapple |
|--------|-------|----------|
| **Maturity** | In-development | Production-ready |
| **Documentation** | Bare-bones | Excellent |
| **Community** | 50 stars | 300+ stars |
| **Maintenance** | Independent | Simons Foundation |
| **Unique strength** | Replay analysis | Time-series + tuning curves |

**Recommendation**: **Use pynapple** for general neuroelectrophysiology analysis.

**Use nelpy ONLY for**:
- ✅ Replay analysis (Davidson method, HMM-based)
- ✅ Support-based time-series operations
- ✅ ValEventArray (events with values)

### 8.2 nelpy in the Ecosystem

| Package | Type | Language | Status | Unique Value |
|---------|------|----------|--------|--------------|
| **nelpy** | ANALYSIS | Python | IN-DEV | Replay analysis, support objects |
| **pynapple** | ANALYSIS | Python | ACTIVE | Time-series (mature) ✅ |
| TSToolbox_Utils | ANALYSIS | MATLAB | LEGACY | Border score, HD variants |
| RatInABox | SIMULATION | Python | ACTIVE | Synthetic data |
| neurospatial | ANALYSIS | Python | ACTIVE | Any graph, differential ops |
| opexebo | ANALYSIS | Python | ACTIVE | Metrics validation |
| vandermeerlab | ANALYSIS | MATLAB | ACTIVE | Task workflows |
| neurocode | ANALYSIS | MATLAB | ACTIVE | Comprehensive pipeline |
| buzcode | ANALYSIS | MATLAB | ACTIVE | Preprocessing |

**nelpy occupies a NICHE position**: Replay analysis specialists might use it, but most users should choose pynapple.

## 9. Impact on neurospatial Implementation Plan

### 9.1 What nelpy Validates

✅ **Bayesian decoding** - Confirms Poisson likelihood is universal (9th package confirming this!)

✅ **Replay analysis is important** - Davidson method + HMM scoring shows demand for replay tools

✅ **Cross-validation utilities** - Shows need for evaluation frameworks

❌ **Spatial metrics gap** - nelpy planned but didn't implement spatial metrics (validates neurospatial's importance)

### 9.2 What nelpy Adds

**Replay analysis sophistication** ⭐:

nelpy's replay analysis is MORE SOPHISTICATED than neurocode, buzcode, or vandermeerlab:

| Method | nelpy | neurocode | buzcode | vandermeerlab |
|--------|-------|-----------|---------|---------------|
| Davidson 2009 | ✅ | ❌ | ❌ | ❌ |
| HMM-based | ✅ | ✅ NMF | ❌ | ❌ |
| Time-resolved | ✅ | ❌ | ❌ | ❌ |
| Multiple shuffles | ✅ 5 types | ✅ | ✅ | ✅ |
| Trajectory scoring | ✅ | ❌ | ❌ | ❌ |

**Recommendation**: Consider adding replay analysis to neurospatial (future, low priority).

### 9.3 No Changes to Core Plan

The implementation plan remains unchanged:

| Phase | Timeline | Risk | Status |
|-------|----------|------|--------|
| Phase 1: Differential operators | 3 weeks | Low | Validated |
| Phase 2: Signal processing | 6 weeks | Medium | Validated |
| Phase 3: Path operations | 1 week | Low | Validated |
| Phase 4: Metrics module | 3 weeks | Low | Validated ✅ |
| Phase 5: Polish | 2 weeks | Low | Validated |

**Timeline**: 15 weeks (no change)

**Why no changes**:
- nelpy validates Bayesian decoding (already in plan)
- nelpy's replay analysis is OUT OF SCOPE for neurospatial (focus on spatial primitives, not replay)
- Spatial metrics gap in nelpy CONFIRMS neurospatial's importance

## 10. Key Takeaways

### 10.1 What nelpy Is

✅ **Time-series infrastructure** - EventArray, AnalogSignalArray, IntervalArray (similar to pynapple)
✅ **Replay analysis** - Most sophisticated implementation (Davidson + HMM)
✅ **Bayesian decoding** - Standard Poisson likelihood (working)
✅ **Support-based operations** - Unique emphasis on functional domain

### 10.2 What nelpy Is NOT

❌ **Mature package** - Work-in-progress (99 open issues, many NotImplementedError)
❌ **Spatial metrics provider** - estimators.py mostly unimplemented
❌ **Comprehensive analysis toolkit** - "Simpler and less comprehensive than neo"
❌ **Production-ready** - Documentation is "bare-bones", tutorials are "work-in-progress"

### 10.3 Strategic Recommendation

**For general neuroelectrophysiology**:
- ✅ Use **pynapple** (mature, well-documented, actively maintained)

**For replay analysis**:
- ✅ Use **nelpy** (most sophisticated replay methods)
- ✅ Use **neurocode** (NMF-based assemblies)
- ✅ Use **buzcode** (multiple decoding approaches)

**For spatial analysis**:
- ✅ Use **neurospatial** (once implemented - fills gap nelpy didn't fill)
- ✅ Use **opexebo** (for metrics validation)

**For simulation + analysis**:
- ✅ Use **RatInABox + neurospatial + pynapple** (complete Python stack)

### 10.4 nelpy's Legacy

**What nelpy contributed to the ecosystem**:
1. ✅ **Support** concept - Functional domain as first-class citizen
2. ✅ **ValEventArray** - Events with associated values
3. ✅ **Sophisticated replay analysis** - Davidson + HMM methods
4. ✅ **Cross-validation framework** - Evaluation utilities for decoding

**What nelpy didn't deliver**:
1. ❌ **Spatial metrics** - Planned but not implemented
2. ❌ **Production maturity** - Still in-development after 8 years (since 2016)
3. ❌ **Comprehensive documentation** - "Bare-bones" tutorials

**Conclusion**: nelpy is a **NICHE PACKAGE** for replay specialists, not a general-purpose tool like pynapple.

---

**Package comparison summary** (9 packages analyzed):

| Package | Type | Language | Status | Unique Value | Recommendation |
|---------|------|----------|--------|--------------|----------------|
| **nelpy** | ANALYSIS | Python | IN-DEV | Replay analysis | ⚠️ Niche use only |
| **pynapple** | ANALYSIS | Python | ACTIVE | Time-series (mature) | ✅ Primary choice |
| TSToolbox_Utils | ANALYSIS | MATLAB | LEGACY | Border score | ❌ Use pynapple |
| RatInABox | SIMULATION | Python | ACTIVE | Synthetic data | ✅ Simulation |
| neurospatial | ANALYSIS | Python | ACTIVE | Any graph | ✅ Spatial discretization |
| opexebo | ANALYSIS | Python | ACTIVE | Metrics validation | ✅ Cross-validation |
| vandermeerlab | ANALYSIS | MATLAB | ACTIVE | Task workflows | ⚠️ MATLAB users |
| neurocode | ANALYSIS | MATLAB | ACTIVE | Comprehensive | ⚠️ MATLAB users |
| buzcode | ANALYSIS | MATLAB | ACTIVE | Preprocessing | ⚠️ MATLAB users |

**Python stack for spatial neuroscience: pynapple + neurospatial + RatInABox** ✅

**Timeline**: 15 weeks (validated against 9 packages)
**Risk**: MEDIUM (well-validated algorithms)
**Impact**: HIGH (fills unique gap - all packages confirm neurospatial's importance)
