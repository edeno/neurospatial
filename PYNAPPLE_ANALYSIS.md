# Pynapple Process Module Analysis

## Overview

**pynapple** (https://github.com/pynapple-org/pynapple) is a modern Python library for neurophysiological data analysis. It provides a lightweight, time-series-focused API for spike trains, behavioral data, and neural decoding. This document analyzes pynapple's spatial processing capabilities and compares with opexebo, buzcode, and neurospatial.

**Key finding**: pynapple provides **excellent time-series infrastructure** and **basic spatial analysis**, but **lacks comprehensive spatial metrics** and **spatial discretization primitives**.

---

## What pynapple Provides

### Package Overview

| Property | Details |
|----------|---------|
| **Language** | Python 100% |
| **License** | MIT |
| **Focus** | Time series (spike times, events, intervals) |
| **Design** | Lightweight, pandas-like API |
| **Core Classes** | Ts, Tsd, TsGroup, IntervalSet |
| **Integration** | xarray, NWB format, numba/jax backends |

### Process Module Functions (27 total)

```python
pynapple.process:
├── tuning_curves:
│   ├── compute_tuning_curves()              # N-dimensional
│   ├── compute_1d_tuning_curves()          # Convenience wrapper
│   ├── compute_2d_tuning_curves()          # Place fields
│   ├── compute_1d/2d_tuning_curves_continuous()  # Calcium imaging
│   ├── compute_mutual_information()        # Skaggs information
│   ├── compute_1d/2d_mutual_info()        # Convenience wrappers
│   ├── compute_discrete_tuning_curves()    # Categorical features
│   └── compute_response_per_epoch()        # Trial-averaged
├── decoding:
│   ├── decode_bayes()                      # Bayesian (Poisson)
│   ├── decode_template()                   # Template matching
│   ├── decode_1d()                         # 1D convenience
│   └── decode_2d()                         # 2D convenience
├── correlograms:
│   ├── compute_autocorrelogram()
│   ├── compute_crosscorrelogram()
│   ├── compute_eventcorrelogram()
│   └── compute_isi_distribution()
├── perievent:
│   ├── compute_perievent()
│   ├── compute_perievent_continuous()
│   └── compute_event_trigger_average()
├── filtering:
│   ├── apply_bandpass/bandstop/highpass/lowpass_filter()
│   └── get_filter_frequency_response()
├── spectrum:
│   ├── compute_fft()
│   ├── compute_power_spectral_density()
│   └── compute_mean_power_spectral_density()
├── randomize:
│   ├── jitter_timestamps()
│   ├── resample_timestamps()
│   ├── shift_timestamps()
│   └── shuffle_ts_intervals()
├── wavelets:
│   ├── compute_wavelet_transform()
│   └── generate_morlet_filterbank()
└── warping:
    ├── build_tensor()
    └── warp_tensor()
```

---

## Detailed Analysis: Spatial Processing

### 1. Tuning Curves (`tuning_curves.py`)

**Core function**: `compute_tuning_curves()`

**Algorithm**:
```python
def compute_tuning_curves(data, features, bins, epochs=None):
    """
    Compute n-dimensional tuning curves.

    Parameters
    ----------
    data : TsGroup, Tsd, TsdFrame
        Neural activity (spike times or continuous signals)
    features : Tsd or TsdFrame
        Behavioral variables (position, head direction, speed, etc.)
    bins : int, sequence, or dict
        Binning specification for each feature dimension
    epochs : IntervalSet, optional
        Time intervals to restrict analysis

    Returns
    -------
    xarray.DataArray
        N-dimensional array with:
        - Dims: feature_1, feature_2, ..., neuron
        - Coords: bin centers for each feature
        - Attrs: occupancy, bin_edges

    Notes
    -----
    Uses np.histogramdd() for binning.
    Normalizes spike counts by occupancy (time in bin).
    """
    # 1. Bin features and count occupancy
    occupancy, edges = np.histogramdd(features, bins=bins)

    # 2. Bin spikes into feature space
    spike_counts = np.histogramdd(
        features[spike_indices],
        bins=edges
    )

    # 3. Normalize by occupancy
    firing_rate = spike_counts / (occupancy * dt)

    # 4. Return as xarray with labeled dimensions
    return xr.DataArray(firing_rate, dims=dims, coords=coords)
```

**Strengths**:
- ✅ **N-dimensional**: Handles arbitrary feature combinations
- ✅ **xarray output**: Labeled arrays with metadata
- ✅ **Occupancy normalization**: Proper rate computation
- ✅ **Flexible binning**: Int (auto), sequence (per-dim), or explicit edges
- ✅ **Epoch support**: Restrict analysis to specific time intervals
- ✅ **Continuous signals**: Separate functions for calcium imaging

**Limitations**:
- ❌ **No spatial discretization**: Users provide pre-binned features
- ❌ **No environment representation**: No graph, connectivity, or layout
- ❌ **No morphological operations**: No dilate, fill_holes, etc.
- ❌ **Assumes regular grids**: np.histogramdd requires rectangular bins
- ❌ **No place field detection**: Just computes rate maps, no field finding

**Comparison**:
- **Similar to neurospatial**: Occupancy normalization, flexible binning
- **Different from neurospatial**: No environment abstraction, no graph operations
- **Similar to opexebo**: Rate map computation
- **Different from opexebo**: No metrics (sparsity, coherence, grid score)

---

### 2. Mutual Information (`tuning_curves.py`)

**Core function**: `compute_mutual_information()`

**Algorithm** (Skaggs et al.):
```python
def compute_mutual_information(tuning_curves, occupancy):
    """
    Compute spatial information (Skaggs et al.).

    Formula:
        I = Σ p(x) * (r(x) / r_mean) * log2(r(x) / r_mean)

    where:
        p(x) = occupancy probability
        r(x) = firing rate at position x
        r_mean = mean firing rate

    Returns
    -------
    dict with keys:
        - 'SI': Spatial information rate (bits/s)
        - 'SI_bits_spike': Spatial information content (bits/spike)
    """
    p_x = occupancy / occupancy.sum()
    r_mean = (tuning_curves * p_x).sum()

    # Only bins where rate > 0
    valid = tuning_curves > 0
    info = (p_x[valid] *
            tuning_curves[valid] / r_mean *
            np.log2(tuning_curves[valid] / r_mean)).sum()

    return {
        'SI': info,  # bits/s
        'SI_bits_spike': info / r_mean  # bits/spike
    }
```

**Strengths**:
- ✅ **Standard formula**: Matches Skaggs et al. (1993)
- ✅ **Both versions**: Bits/s and bits/spike
- ✅ **Works with N-D tuning curves**

**Limitations**:
- ❌ **Only mutual information**: No sparsity, selectivity, etc.
- ❌ **Requires pre-computed tuning curves**: Not integrated workflow

**Comparison**:
- **Matches opexebo**: Same Skaggs formula
- **Matches neurospatial proposed**: Same implementation
- **More than buzcode**: buzcode doesn't provide this

---

### 3. Bayesian Decoding (`decoding.py`)

**Core function**: `decode_bayes()`

**Algorithm** (Poisson likelihood):
```python
def decode_bayes(tuning_curves, spike_data, bin_size, uniform_prior=True):
    """
    Bayesian position decoding.

    Uses Bayes rule: P(x|n) ∝ P(n|x) * P(x)

    Likelihood: Poisson model
        P(n|x) = Π_i (λ_i(x)^n_i * exp(-λ_i(x))) / n_i!

    Log-likelihood (for numerical stability):
        log P(n|x) = Σ_i (n_i * log(λ_i(x)) - bin_size * λ_i(x))

    Parameters
    ----------
    tuning_curves : xarray.DataArray
        Rate maps for each neuron (features × neurons)
    spike_data : TsdFrame
        Spike counts binned at target resolution
    bin_size : float
        Time bin size (seconds)
    uniform_prior : bool
        If True, use uniform prior. If False, use occupancy.

    Returns
    -------
    xarray.DataArray
        Posterior probability P(x|n) over time
    """
    # Log-likelihood
    log_likelihood = (
        spike_data @ np.log(tuning_curves + EPS) -
        bin_size * tuning_curves.sum(axis=-1)
    )

    # Log-prior
    if uniform_prior:
        log_prior = 0
    else:
        log_prior = np.log(occupancy / occupancy.sum())

    # Log-posterior
    log_posterior = log_likelihood + log_prior

    # Exponentiate and normalize
    posterior = np.exp(log_posterior - log_posterior.max(axis=1, keepdims=True))
    posterior /= posterior.sum(axis=1, keepdims=True)

    return posterior
```

**Strengths**:
- ✅ **Standard Bayesian decoding**: Poisson likelihood model
- ✅ **Numerically stable**: Log-space computation
- ✅ **Optional prior**: Uniform or occupancy-based
- ✅ **Returns posterior distribution**: Full probability over space
- ✅ **Works with N-D features**

**Limitations**:
- ❌ **Only Poisson model**: No other likelihood models
- ❌ **No place field integration**: Separate from field detection
- ❌ **No cross-validation**: No built-in train/test splits

**Comparison**:
- **Similar to buzcode**: Both provide Bayesian decoding
- **Simpler than buzcode**: buzcode has multiple decoding methods (rate, phase, GLM)
- **More advanced than opexebo**: opexebo doesn't provide decoding
- **Different from neurospatial**: neurospatial focuses on spatial primitives, not decoding

---

### 4. Template Matching Decoding (`decoding.py`)

**Core function**: `decode_template()`

**Algorithm**:
```python
def decode_template(tuning_curves, spike_data, distance='correlation'):
    """
    Template matching decoding.

    For each time bin, finds feature bin with most similar
    neural population response to observed spike pattern.

    Parameters
    ----------
    distance : str
        Distance metric: 'correlation', 'euclidean', etc.
        (passed to scipy.spatial.distance.cdist)

    Returns
    -------
    decoded_position : array
        Index of best-matching feature bin at each time
    """
    # Compute pairwise distances between:
    # - spike_data (time × neurons)
    # - tuning_curves (features × neurons)

    distances = cdist(spike_data, tuning_curves.T, metric=distance)

    # Argmin over feature dimension
    decoded = distances.argmin(axis=1)

    return decoded
```

**Strengths**:
- ✅ **Simple and fast**: No likelihood computation
- ✅ **Flexible distance metrics**: Correlation, Euclidean, etc.
- ✅ **Works with any features**

**Limitations**:
- ❌ **Point estimate only**: No posterior distribution
- ❌ **Less principled than Bayesian**: No probabilistic interpretation

---

## Comparison: pynapple vs Others

### Spatial Metrics Coverage

| Metric | buzcode | opexebo | pynapple | neurospatial (proposed) |
|--------|---------|---------|----------|-------------------------|
| **Tuning curves** | ⚠️ 1D only | ✅ 2D grids | ✅ N-D | ✅ Any graph |
| **Skaggs information** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Sparsity** | ❌ No | ✅ Yes | ❌ No | ✅ Yes |
| **Coherence** | ❌ No | ✅ Yes | ❌ No | ✅ Yes (any graph) |
| **Grid score** | ❌ No | ✅ Yes | ❌ No | ✅ Yes |
| **Spatial autocorrelation** | ❌ No | ✅ Yes | ❌ No | ✅ Yes |
| **Border score** | ❌ No | ✅ Yes | ❌ No | ✅ Yes |
| **Place field detection** | ✅ 1D | ✅ 2D | ❌ No | ✅ Any graph |
| **Bayesian decoding** | ✅ Multiple | ❌ No | ✅ Poisson | 🔶 Future |
| **Template decoding** | ✅ MaxCorr | ❌ No | ✅ Yes | 🔶 Future |

**Summary**:
- **pynapple**: Excellent time-series + basic spatial (tuning, Skaggs, decoding)
- **opexebo**: Comprehensive spatial metrics (place, grid, border)
- **buzcode**: 1D tracks + preprocessing + decoding (MATLAB)
- **neurospatial**: Spatial primitives + any graph structure

---

### Environment Support

| Feature | buzcode | opexebo | pynapple | neurospatial |
|---------|---------|---------|----------|--------------|
| **Time series** | ✅ Good | ⚠️ Basic | ✅ **Excellent** | ⚠️ Basic |
| **Intervals/Epochs** | ✅ Good | ❌ No | ✅ **Excellent** | ⚠️ Via trajectories |
| **Spatial discretization** | ⚠️ 1D bins | ✅ 2D grids | ❌ **User provides** | ✅ **Core feature** |
| **Environment graph** | ❌ No | ❌ No | ❌ No | ✅ **Core feature** |
| **Irregular graphs** | ❌ No | ❌ No | ❌ No | ✅ **Core feature** |
| **N-D features** | ❌ 1D | ⚠️ 2D | ✅ **N-D** | ✅ N-D |
| **Custom layouts** | ❌ No | ❌ No | ❌ No | ✅ 7+ engines |

---

### Design Philosophy

| Aspect | pynapple | opexebo | buzcode | neurospatial |
|--------|----------|---------|---------|--------------|
| **Abstraction level** | Time series | Metrics | Full pipeline | Spatial primitives |
| **User provides** | Binned features | Rate maps | Raw data | Position data |
| **Package provides** | Tuning curves | Metrics | Everything | Discretization + metrics |
| **Focus** | Temporal analysis | Spatial metrics | Preprocessing | Graph operations |
| **Core data type** | Ts, Tsd, IntervalSet | NumPy arrays | MATLAB structs | Environment, Graph |

---

## What pynapple Does Well

### 1. Time Series Infrastructure ⭐⭐⭐

**Excellent time-series API**:
```python
# Clean, intuitive API
spikes = nap.Ts(spike_times)  # Spike train
position = nap.Tsd(t=times, d=positions)  # Time series
epochs = nap.IntervalSet(start=[0, 100], end=[50, 150])  # Intervals

# Restrict to epochs
spikes_task = spikes.restrict(epochs)
position_task = position.restrict(epochs)

# Compute tuning curves
tc = nap.compute_2d_tuning_curves(
    spikes,
    position,
    bins=20
)
```

**Strengths**:
- ✅ **Pandas-like API**: Familiar, intuitive
- ✅ **IntervalSet**: First-class support for epochs, trials, states
- ✅ **Restrict operations**: Easy temporal filtering
- ✅ **Metadata support**: Label neurons, conditions, etc.

**Comparison**:
- **Better than opexebo**: opexebo has no time-series abstraction
- **Better than buzcode**: Cleaner API than MATLAB structs
- **Different from neurospatial**: neurospatial focuses on spatial, not temporal

### 2. N-Dimensional Tuning Curves ⭐⭐

**Flexible feature combinations**:
```python
# 1D: Head direction
tc_hd = nap.compute_1d_tuning_curves(spikes, head_direction, bins=36)

# 2D: Position (place fields)
tc_pos = nap.compute_2d_tuning_curves(spikes, position, bins=20)

# 3D: Position + head direction
tc_3d = nap.compute_tuning_curves(
    spikes,
    features={'x': position[:, 0], 'y': position[:, 1], 'hd': head_direction},
    bins={'x': 20, 'y': 20, 'hd': 36}
)

# Returns xarray with labeled dimensions
tc_3d.dims  # ('x', 'y', 'hd', 'neuron')
```

**Strengths**:
- ✅ **N-dimensional**: Any feature combination
- ✅ **xarray output**: Labeled, self-documenting
- ✅ **Flexible binning**: Per-dimension control

**Comparison**:
- **More flexible than opexebo**: opexebo is 2D-focused
- **More flexible than buzcode**: buzcode is 1D-focused
- **Different from neurospatial**: neurospatial provides spatial graph, not feature binning

### 3. Bayesian Decoding ⭐

**Clean decoding API**:
```python
# Compute tuning curves
tc = nap.compute_2d_tuning_curves(spikes, position_train, bins=20)

# Decode test data
decoded = nap.decode_bayes(
    tc,
    spikes_test,
    bin_size=0.1,  # 100 ms bins
    uniform_prior=False  # Use occupancy prior
)

# Returns posterior distribution over space
decoded.shape  # (n_time_bins, n_spatial_bins_x, n_spatial_bins_y)
```

**Strengths**:
- ✅ **Standard Bayesian method**: Poisson likelihood
- ✅ **Numerically stable**: Log-space computation
- ✅ **Returns full posterior**: Not just point estimate

**Comparison**:
- **Similar to buzcode**: Both provide Bayesian decoding
- **Simpler than buzcode**: buzcode has more methods (phase, GLM)
- **More than opexebo**: opexebo doesn't provide decoding
- **Different from neurospatial**: neurospatial focuses on primitives

---

## What pynapple Lacks

### 1. Spatial Discretization ❌

**Users must provide pre-binned features**:
```python
# User must discretize position themselves
position_2d = load_position()  # (n_samples, 2)

# pynapple just bins whatever you give it
tc = nap.compute_2d_tuning_curves(spikes, position_2d, bins=20)
# Uses np.histogramdd - assumes regular rectangular grid
```

**Missing**:
- ❌ No Environment abstraction
- ❌ No spatial graph / connectivity
- ❌ No layout engines (hexagonal, irregular, etc.)
- ❌ No morphological operations (dilate, fill_holes)
- ❌ No spatial queries (neighbors, distance, paths)

**Impact**: pynapple is **complementary** to neurospatial
- neurospatial: Discretize space → Environment
- pynapple: Compute tuning curves on discretized features

### 2. Comprehensive Spatial Metrics ❌

**Only provides Skaggs information**:
```python
# Provided
tc = nap.compute_2d_tuning_curves(spikes, position, bins=20)
info = nap.compute_mutual_information(tc, occupancy)
# Returns: {'SI': bits/s, 'SI_bits_spike': bits/spike}

# Missing (available in opexebo)
sparsity = ???  # Not provided
coherence = ???  # Not provided
grid_score = ???  # Not provided
border_score = ???  # Not provided
```

**Missing metrics** (from opexebo):
- ❌ Sparsity
- ❌ Selectivity
- ❌ Coherence
- ❌ Grid score
- ❌ Spatial autocorrelation
- ❌ Border score
- ❌ Head direction tuning (Rayleigh, MVL)

**Impact**: Users need opexebo or neurospatial.metrics for comprehensive analysis.

### 3. Place Field Detection ❌

**No field detection algorithms**:
```python
# pynapple provides
tc = nap.compute_2d_tuning_curves(spikes, position, bins=20)

# But does NOT provide
fields = ???  # No place field detection
# Must implement yourself or use opexebo
```

**Missing** (from opexebo):
- ❌ Place field detection
- ❌ Field properties (size, centroid, peak rate)
- ❌ Adaptive thresholding
- ❌ Field validation (no holes, stable boundaries)

**Impact**: pynapple is for **computing rate maps**, not **analyzing place fields**.

### 4. Spatial Primitives ❌

**No graph-based operations**:
```python
# Missing from pynapple (proposed for neurospatial)
gradient = ???  # No differential operators
divergence = ???  # No differential operators
neighbor_reduce = ???  # No graph neighbor operations
spatial_autocorrelation = ???  # No graph autocorrelation
distance_field = ???  # No graph distances
```

**Missing primitives**:
- ❌ Differential operators (gradient, divergence, Laplacian)
- ❌ Neighbor operations (reduce, aggregate)
- ❌ Path operations (accumulate, propagate)
- ❌ Distance fields
- ❌ Spatial autocorrelation on graphs

**Impact**: pynapple focuses on time-series, not spatial graph operations.

---

## Strategic Positioning

### Four Complementary Tools

**buzcode** - Preprocessing & 1D (MATLAB):
- Full electrophysiology pipeline
- 1D track analysis
- Bayesian decoding (multiple methods)
- LFP, SWR, assemblies

**opexebo** - 2D Spatial Metrics (Python):
- Comprehensive place/grid/border metrics
- 2D regular grid environments
- Authority from Nobel lab
- Grid score, spatial autocorrelation

**pynapple** - Time Series & Decoding (Python):
- Excellent time-series infrastructure
- N-D tuning curves
- Skaggs information
- Bayesian decoding (Poisson)
- Intervals, epochs, trials

**neurospatial** - Spatial Primitives (Python):
- Spatial discretization (any structure)
- Graph-based operations
- Differential operators
- Irregular environments
- Spatial metrics (future)

---

## Ideal Workflow

```python
# 1. Time series: pynapple
import pynapple as nap

spikes = nap.load_file('spikes.nwb')
position_raw = nap.load_file('position.nwb')
trials = nap.IntervalSet(start=trial_starts, end=trial_ends)

# Restrict to task periods
spikes_task = spikes.restrict(trials)
position_task = position_raw.restrict(trials)

# 2. Spatial discretization: neurospatial
from neurospatial import Environment

env = Environment.from_samples(
    position_task.values,
    bin_size=2.5,
    units='cm'
)

# Get occupancy and bin sequence
occupancy = env.occupancy(
    position_task.index.values,
    position_task.values
)

# Map continuous position to discrete bins
binned_position = env.bin_at(position_task.values)

# 3. Tuning curves: pynapple
# Convert discrete bins back to continuous for pynapple
position_binned_continuous = env.bin_centers[binned_position]

tc = nap.compute_2d_tuning_curves(
    spikes_task,
    nap.Tsd(t=position_task.index, d=position_binned_continuous),
    bins=env.layout.grid_shape  # Use neurospatial's binning
)

# 4. Spatial metrics: opexebo (regular grid) or neurospatial.metrics (any graph)
if env.layout._layout_type_tag == 'RegularGridLayout':
    # Use opexebo (authoritative for regular grids)
    import opexebo
    rate_map_2d = tc[neuron_id].values.reshape(env.layout.grid_shape)
    info = opexebo.analysis.rate_map_stats(rate_map_2d, occupancy.reshape(...))
    grid_score = opexebo.analysis.grid_score(rate_map_2d)
else:
    # Use neurospatial.metrics (works on irregular graphs)
    from neurospatial.metrics import skaggs_information, coherence
    info = skaggs_information(tc[neuron_id].values, occupancy)
    coh = coherence(tc[neuron_id].values, env)

# 5. Decoding: pynapple
decoded = nap.decode_bayes(tc, spikes_test, bin_size=0.1)
```

**Summary**: Each tool does one thing well, combine for complete workflow.

---

## Key Insights for neurospatial

### 1. pynapple Validates Time-Series Gap

**pynapple's strength**: Time-series infrastructure (Ts, Tsd, IntervalSet)

**neurospatial's gap**: No time-series abstractions
- Currently: Users pass raw NumPy arrays
- No IntervalSet equivalent
- No restrict-by-epoch operations

**Recommendation**: **Don't replicate pynapple**
- pynapple handles time-series excellently
- neurospatial should focus on spatial
- Users can combine both packages

**Action**: Document integration with pynapple

### 2. Spatial Discretization is Unique Value

**pynapple requires**: Users provide pre-binned features
- Uses `np.histogramdd` (rectangular grids only)
- No environment abstraction
- No spatial graph

**neurospatial provides**: Spatial discretization
- Environment class with graph
- Flexible layouts (irregular, hexagonal, etc.)
- Morphological operations
- Spatial queries

**This confirms**: neurospatial's unique value proposition

### 3. Metrics Module Still Needed

**pynapple provides**: Skaggs information only

**pynapple missing**:
- Sparsity
- Coherence
- Grid score
- Border score
- Place field detection

**neurospatial.metrics needed**: Fill this gap
- Extend opexebo metrics to irregular graphs
- Integrate with neurospatial Environment
- Provide comprehensive metrics

### 4. Decoding is Lower Priority

**Both buzcode and pynapple** provide Bayesian decoding.

**neurospatial should**: Focus on spatial primitives first
- Decoding can be future extension (Phase 6+)
- Users can use pynapple for decoding in meantime
- Not critical path

---

## Recommendations

### 1. Document Integration with pynapple

**Add example workflow**:
```python
# pynapple → neurospatial → pynapple workflow
```

**Show**:
- How to convert between Tsd and NumPy arrays
- How to use neurospatial Environment with pynapple tuning curves
- How to restrict analysis to IntervalSets

**Benefit**: Users get best of both worlds.

### 2. Don't Replicate Time-Series

**pynapple excels at**: Ts, Tsd, IntervalSet, epochs, trials

**neurospatial should**: Focus on spatial
- Environment discretization
- Graph operations
- Spatial primitives
- Spatial metrics

**Rationale**: Complementary tools, not competitive.

### 3. Emphasize Spatial Discretization Value

**Marketing message**: "neurospatial handles spatial discretization so you can use pynapple for tuning curves"

**Workflow**:
1. neurospatial: Discretize space
2. pynapple: Compute tuning curves
3. opexebo/neurospatial: Compute metrics

**Value**: Each tool does one thing well.

### 4. Provide Missing Metrics

**pynapple provides**: Skaggs information

**neurospatial should provide** (Phase 4):
- Sparsity
- Coherence
- Grid score
- Border score
- Place field detection
- All other opexebo metrics

**Action**: Implement metrics module (already in plan).

---

## Implementation Plan Impact

### No Major Changes Needed

**pynapple analysis confirms**:
1. ✅ Spatial discretization is unique value (neurospatial's core)
2. ✅ Metrics module needed (pynapple only has Skaggs info)
3. ✅ Time-series is separate concern (use pynapple)
4. ✅ Decoding is lower priority (pynapple/buzcode provide)

**Validation**: pynapple fills different niche (time-series), validates our spatial focus.

### Documentation Updates

**Add sections**:
1. "Integration with pynapple"
2. "Converting between pynapple and neurospatial"
3. "Complete workflow example" (combining both)

**Example notebooks**:
- `12_pynapple_integration.ipynb`
- Show Ts → NumPy → Environment → tuning curves workflow

---

## Algorithm Comparison: Skaggs Information

### pynapple Implementation

```python
def compute_mutual_information(tc, occupancy):
    p_x = occupancy / occupancy.sum()
    r_mean = (tc * p_x).sum()
    valid = tc > 0
    info = (p_x[valid] * tc[valid] / r_mean *
            np.log2(tc[valid] / r_mean)).sum()
    return {'SI': info, 'SI_bits_spike': info / r_mean}
```

### opexebo Implementation

```python
def rate_map_stats(rate_map, occupancy_map):
    position_pdf = occupancy_map / occupancy_map.sum()
    mean_rate = np.nanmean(rate_map)
    valid = (rate_map / mean_rate) >= 1
    inf_rate = np.sum(
        position_pdf[valid] * rate_map[valid] *
        np.log2(rate_map[valid] / mean_rate)
    )
    inf_content = inf_rate / mean_rate
    return {'spatial_information_rate': inf_rate,
            'spatial_information_content': inf_content}
```

### neurospatial (Proposed)

```python
def skaggs_information(firing_rate, occupancy, *, base=2):
    """Same as opexebo, with reference to both opexebo and pynapple."""
    p_x = occupancy / occupancy.sum()
    r_mean = np.nanmean(firing_rate)
    valid = (firing_rate > 0) & (p_x > 0)

    info_rate = np.sum(
        p_x[valid] * firing_rate[valid] / r_mean *
        np.log2(firing_rate[valid] / r_mean)
    )
    info_content = info_rate / r_mean

    return {
        'spatial_information_rate': info_rate,
        'spatial_information_content': info_content
    }
```

**All three are identical** - confirms standard formula.

---

## Conclusion

### Key Findings

1. **pynapple provides excellent time-series infrastructure**
2. **pynapple has basic spatial analysis** (tuning curves, Skaggs info, decoding)
3. **pynapple lacks spatial discretization** (no Environment, no graph)
4. **pynapple lacks comprehensive metrics** (only Skaggs info)
5. **pynapple is complementary to neurospatial**

### Four Complementary Tools Summary

| Package | Strength | Gap | Use For |
|---------|----------|-----|---------|
| **buzcode** | Preprocessing, 1D, MATLAB | Minimal metrics, MATLAB-only | Full pipeline, MATLAB users |
| **opexebo** | Comprehensive metrics, Nobel lab | 2D grids only | Spatial metrics (2D) |
| **pynapple** | Time-series, N-D tuning | No spatial discretization | Spike trains, epochs, trials |
| **neurospatial** | Spatial primitives, any graph | Time-series (use pynapple) | Spatial discretization, graphs |

### Strategic Position Validated

**neurospatial is complementary to all three**:
- Use **pynapple** for time-series
- Use **neurospatial** for spatial discretization
- Use **opexebo** for metrics (regular grids)
- Use **neurospatial.metrics** for metrics (irregular graphs)

**No competition** - Each tool solves different problems.

### Implementation Plan: Validated

**pynapple analysis confirms**:
- ✅ Spatial discretization is unique value
- ✅ Metrics module needed
- ✅ Time-series is separate (use pynapple)
- ✅ Focus on spatial primitives

**Action items**:
- Document pynapple integration
- Add conversion examples
- Show combined workflow
- Emphasize complementary positioning

---

## References

- **pynapple**: https://github.com/pynapple-org/pynapple
- **pynapple docs**: https://pynapple.org/
- **Key modules analyzed**:
  - `tuning_curves.py` - N-D tuning curves, Skaggs information
  - `decoding.py` - Bayesian decoding (Poisson), template matching
- **Skaggs et al. (1993)**: Spatial information metric
- **Comparison packages**: opexebo, buzcode, NetworkX
