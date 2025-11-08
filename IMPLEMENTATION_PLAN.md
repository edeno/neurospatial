# Implementation Plan: Spatial Primitives & Metrics

## Executive Summary

This plan outlines the implementation of **core spatial primitives** and **foundational metrics** for neurospatial v0.3.0, based on comprehensive investigation of capabilities, existing packages, and neuroscience requirements.

**Timeline**: 14 weeks (~3.25 months)
**Priority**: HIGH - Foundational infrastructure for spatial analysis
**Breaking Changes**: None (no current users - can rename directly)
**Authority**: Algorithms validated against opexebo (Moser lab, Nobel Prize 2014), neurocode (AyA Lab, Cornell), and ecology literature

### What This Release Enables

- **Core spike/field primitives** - spike train to field conversion, reward fields (Phase 0)
- **Differential operators** - gradient, divergence, Laplacian on irregular graphs (RL/replay analyses)
- **Signal processing primitives** - neighbor operations, custom convolutions
- **Place field metrics** - standard neuroscience analyses (detection, information, sparsity)
- **Population metrics** - coverage, density, overlap
- **Boundary cell metrics** - border score (wall-preferring cells)
- **Trajectory metrics** - turn angles, step lengths, home range, MSD (from ecology, in Phase 4)
- **Behavioral segmentation** - automatic detection of runs, laps, trials (Phase 4)

## Context for Implementation

The following sections provide essential background for implementing this plan.

### Current Codebase State (v0.2.1)

This section clarifies what functionality **already exists** vs. what needs to be **created** in v0.3.0.

#### Existing Core Infrastructure

**Graph Structure** (`src/neurospatial/environment/core.py`):

- `env.connectivity` - NetworkX graph with mandatory node/edge attributes
- Node attributes: `'pos'`, `'source_grid_flat_index'`, `'original_grid_nd_index'`
- Edge attributes: `'distance'`, `'vector'`, `'edge_id'`, `'angle_2d'` (optional)
- All spatial operations operate on this graph structure

**Spatial Query Methods** (`src/neurospatial/environment/queries.py`, 897 lines):

- `bin_at(points)` - Map continuous coordinates to bin indices
- `distance_between(bin_i, bin_j)` - Graph distance between two bins
- `distance_to(target_bins)` - Distance field from target bins to all bins (uses Dijkstra)
- `shortest_path(source, target)` - Shortest path between bins
- `contains(points)` - Check if points are within environment bounds
- `neighbors(bin_id)` - Get neighboring bin IDs

**Spatial Field Operations** (`src/neurospatial/environment/fields.py`, 564 lines):

- `smooth(field, bandwidth, *, kernel='gaussian')` - Gaussian smoothing using distance-based kernel
- `compute_kernel(distances, bandwidth, *, kernel='gaussian')` - Compute smoothing kernel matrix
- `interpolate(field, points)` - Interpolate field values at arbitrary points
- All operations respect graph connectivity (not Euclidean smoothing)

**Trajectory Analysis** (`src/neurospatial/environment/trajectory.py`, 1,222 lines):

- `occupancy(trajectory, times=None)` - Time spent in each bin (seconds or frame counts)
- `bin_sequence(trajectory)` - Convert continuous trajectory to sequence of bin indices
- `transitions(trajectory)` - Transition matrix between bins (n_bins × n_bins)
- These provide the foundation for behavioral segmentation in Phase 4

**Existing KL Divergence** (`src/neurospatial/field_ops.py`):

- `divergence(p, q, *, kind='kl', eps=1e-12)` - KL or JS divergence between probability distributions
- **TO BE RENAMED** to `kl_divergence()` in Phase 1.3 to avoid confusion with vector field divergence
- Used for comparing spatial distributions (e.g., place field remapping)

**Distance Field Function** (Public API, `src/neurospatial/__init__.py`):

- `distance_field(graph, sources, *, directed=False)` - Multi-source geodesic distances using Dijkstra
- Used for computing distance maps from goal locations or boundaries
- New gradient/divergence operators will complement this for RL analyses

#### What's Being Added in v0.3.0

**Phase 0**: New `spike_field.py` module with `spikes_to_field()` and `compute_place_field()`, and `reward.py` module with `region_reward_field()` and `goal_reward_field()`

**Phase 1**: New `differential.py` module with `gradient()`, `divergence()` (vector field), `compute_differential_operator()`

**Phase 2**: New `primitives.py` module with `neighbor_reduce()`, `convolve()`

**Phase 3**: New `metrics/` package with `place_fields.py`, `population.py`, `boundary_cells.py`, `trajectory.py`

**Phase 4**: New `segmentation/` package with `regions.py`, `laps.py`, `trials.py`, `similarity.py`

#### Key Integration Points

- **Spike/field primitives** will use existing `occupancy()` and `smooth()` methods
- **Reward fields** will use existing `distance_field()` and `regions_to_mask()` functions
- **Differential operators** will use existing `env.connectivity` graph
- **Metrics** will use existing `smooth()` and `occupancy()` methods
- **Behavioral segmentation** will use existing `bin_sequence()` and `transitions()` methods
- **Place field detection** will use existing `distance_field()` for boundary detection
- All new code follows existing mixin pattern with `self: "Environment"` type annotations

---

### Neuroscience Terminology

Key neuroscience concepts used throughout this plan. Essential for understanding parameter choices and validation strategies.

**Place field**: Spatial region where a hippocampal place cell fires preferentially. Place cells fire action potentials when an animal occupies specific locations in its environment. Key discovery: O'Keefe & Dostrovsky (1971), Nobel Prize in Physiology or Medicine (2014). Detected via firing rate maps normalized by occupancy.

**Grid cell**: Hippocampal neuron with hexagonal periodic firing pattern covering the environment in a triangular lattice. Discovered by Moser lab (2005), Nobel Prize (2014). Detected via spatial autocorrelation producing hexagonal pattern with peaks at 60° intervals, quantified by grid score.

**Firing rate map**: 2D array showing average firing rate (spikes/second) of a neuron in each spatial bin. Computed as spike_count / occupancy_time. Central representation for all spatial analyses. Typically smoothed (5 cm bandwidth) to reduce noise.

**Occupancy**: Time spent in each spatial bin, measured in seconds or frame counts. Essential denominator for normalizing spike counts into firing rates. Low-occupancy bins (<0.5 seconds) often excluded as unreliable.

**Border cell (Boundary vector cell)**: Neuron that fires when animal is near environmental boundaries (walls). Fires along one or more walls at specific distances. Quantified via border score (Solstad et al. 2008): `(cM - d) / (cM + d)` where cM = max wall contact ratio, d = distance from firing field to nearest wall.

**Spatial information (Skaggs information)**: Bits per spike measuring how much spatial information a neuron conveys. Formula: `Σ pᵢ (rᵢ/r̄) log₂(rᵢ/r̄)` where pᵢ = occupancy probability, rᵢ = firing rate in bin i, r̄ = mean rate. Higher values indicate more spatially selective cells. Typical place cells: 1-3 bits/spike.

**Sparsity**: Fraction of environment where cell is active, computed as `(Σ pᵢ rᵢ)² / Σ pᵢ rᵢ²`. Range [0, 1]. Lower values indicate sparser, more selective firing. Typical place cells: 0.1-0.3 (active in 10-30% of environment).

**Field stability**: Spatial correlation between firing rate maps recorded in different sessions or trial halves. Pearson correlation typically used. High stability (r > 0.7) indicates reliable place field. Used to assess learning, memory consolidation.

**Pyramidal cell vs Interneuron**: Two major neuron classes in hippocampus. Pyramidal cells (place cells) have low mean firing rates (0.5-5 Hz). Interneurons have high rates (10-50 Hz) and are typically excluded from place cell analyses using 10 Hz threshold (vandermeerlab convention).

**Head direction cell**: Neuron that fires when animal's head points in a preferred allocentric direction, regardless of location. Tuning analyzed with circular statistics (von Mises distribution). Found in postsubiculum, anterodorsal thalamus.

**Replay**: Rapid sequential reactivation of place cells during rest or sleep, representing spatial trajectories at compressed timescales (~20× faster than behavior). Analyzed using decoded position from population activity. Requires gradient-based methods for goal-directed replay detection.

**Spatial autocorrelation**: 2D correlation of firing rate map with shifted versions of itself. Reveals periodic spatial structure. For grid cells, produces hexagonal pattern with peaks at 60° intervals. Computed via FFT for regular grids (opexebo method) or graph-based correlation for irregular grids.

**Grid score (Gridness)**: Quantifies hexagonal periodicity of firing pattern. Computed from spatial autocorrelation map using annular rings approach (Sargolini et al. 2006): `min(r₆₀, r₁₂₀) - max(r₃₀, r₉₀, r₁₅₀)` where rᵢ = correlation at i° rotation. Range [-2, 2], typical good grids: ~1.3, threshold ~0.4.

**Coherence**: Spatial correlation between firing rate and mean of neighboring bins. Measures smoothness of rate map (Muller & Kubie 1989). Computed via `neighbor_reduce()` primitive. High coherence (r > 0.7) indicates spatially organized firing.

---

### Validation Authority Packages

Reference implementations that define gold standards for spatial neuroscience analyses. Our implementations will match these packages for regular grids and extend them to irregular graph structures.

#### opexebo (Gold Standard)

**Repository**: <https://github.com/simon-ball/opexebo>
**Authority**: Developed by Moser laboratory (NTNU, Norway) - Nobel Prize in Physiology or Medicine (2014) for discovery of grid cells
**Language**: Python
**Purpose**: Reference implementation of spatial analyses for grid cells, place cells, and related metrics

**Key Functions**:

- `opexebo.analysis.grid_score()` - Grid score using annular rings approach (Sargolini et al. 2006)
- `opexebo.analysis.autocorrelation()` - Spatial autocorrelation via FFT (fast, validated)
- `opexebo.analysis.border_score()` - Border score (Solstad et al. 2008)
- `opexebo.analysis.rate_map_coherence()` - Spatial coherence (Muller & Kubie 1989)
- `opexebo.analysis.rate_map()` - Occupancy-normalized firing rate maps
- `opexebo.analysis.spatial_information()` - Skaggs information content

**Validation Strategy**: Our implementations will match opexebo outputs **within 1%** for regular grids. Test cases will use opexebo as ground truth where functionality overlaps. Our extensions (irregular grids, graph-based operations) will reduce to opexebo's results when applied to regular grids.

**Why Authoritative**: Direct implementation of methods from Nobel Prize-winning laboratory. Extensively validated against published datasets. Field-standard for grid cell analysis.

#### neurocode (Neuroscience Best Practices)

**Repository**: <https://github.com/ayalab1/neurocode>
**Authority**: AyA Laboratory (Cornell University) - Systems neuroscience and hippocampal function
**Language**: MATLAB
**Purpose**: Comprehensive toolkit for hippocampal data analysis with emphasis on place cells and population dynamics

**Key Functions**:

- `FindPlaceFields.m` - Place field detection with iterative peak-based approach
- `MapStats.m` - Comprehensive rate map statistics (information, sparsity, coherence)
- `NSMAFindGoodLaps.m` - Lap detection and quality assessment
- Circular statistics (`circ_mean`, `circ_r`, `circ_rtest`) - Head direction analysis
- Population vector analyses and remapping detection

**Validation Strategy**: Cross-validate place field detection algorithm against neurocode's subfield discrimination approach. Verify spatial information and sparsity calculations match within numerical precision.

**Why Authoritative**: Represents field best practices for hippocampal analyses. Actively maintained, widely used in systems neuroscience. Implements consensus algorithms from multiple labs.

#### TSToolbox_Utils (Border Score Reference)

**Authority**: Implementation of border score from Solstad et al. (2008) Neuron paper
**Language**: MATLAB
**Purpose**: Reference implementation for boundary vector cell (border cell) detection

**Key Function**: `Compute_BorderScore.m` - Border score algorithm

**Algorithm Details**:

1. Segment place field at 30% of peak firing rate
2. Compute wall contact ratio for each wall (N, S, E, W)
3. Maximum wall contact ratio (cM)
4. Firing-rate-weighted distance to walls (d, normalized 0-1)
5. Border score: `(cM - d) / (cM + d)`

**Validation Strategy**: Match TSToolbox_Utils and opexebo border score implementations exactly for test cases.

**Why Authoritative**: Direct implementation from original Solstad et al. 2008 paper. Used by opexebo as reference.

#### Other Reference Packages

**buzcode** (MATLAB): Comprehensive rodent electrophysiology toolkit, cross-validation for place field metrics
**pynapple** (Python): Time series and interval management, integration target for behavioral segmentation outputs
**RatInABox** (Python): Simulation framework for generating ground-truth data (grid cells, place cells with known properties), validation in v0.4.0

#### Ecology/Trajectory Authority

**Traja** (Python): Trajectory analysis for animal movement, turn angle and step length algorithms
**yupi** (Python): Trajectory classification, mean square displacement (MSD) implementations
**adehabitatHR** (R): Home range estimation (95% kernel density), field standard in ecology
**ctmm** (R): Continuous-time movement modeling, autocorrelation-based home range

**Validation Strategy**: Cross-validate trajectory metrics (turn angles, step lengths, home range, MSD) against ecology package outputs on synthetic trajectories with known properties (straight lines, circles, random walks).

---

## Phase 0: Core Spike/Field Primitives (Weeks 1-2)

### Goal

Implement foundational primitives for converting spike data to spatial fields and reward field generation for RL. These are **missing building blocks** that users currently must implement manually.

### Rationale

The existing plan assumed users already have firing rate maps. In practice:

- Users must manually convert spike trains → occupancy-normalized fields (error-prone)
- RL users must manually create reward fields from regions (no clean primitive)

**Phase 0 provides these critical missing primitives before building metrics on top of them.**

**Note**: Batch operations (`smooth_batch()`, `normalize_fields()`) are deferred to v0.3.1. We follow the principle of "make it work, make it right, make it fast" - shipping core functionality first and adding performance optimizations when users demonstrate need.

---

## 0.1 Spike → Field Conversion + Convenience Function (Week 1, Days 1-4)

**New file**: `src/neurospatial/spike_field.py`

### **Prerequisites**

Before implementing `spikes_to_field()`, must add `return_seconds` parameter to existing `env.occupancy()` method:

**File**: `src/neurospatial/environment/trajectory.py`

```python
def occupancy(
    self: EnvironmentProtocol,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    return_seconds: bool = False,  # NEW PARAMETER
) -> NDArray[np.float64]:
    """
    Compute time or sample count spent in each bin.

    Parameters
    ----------
    times : array, shape (n_timepoints,)
        Timestamps (seconds).
    positions : array, shape (n_timepoints, n_dims)
        Position trajectory.
    return_seconds : bool, default=False
        If True, return time in seconds. If False, return sample counts.

    Returns
    -------
    occupancy : array, shape (n_bins,)
        Time (seconds) or count (samples) spent in each bin.
    """
    # ... existing implementation
    if return_seconds:
        # Compute time difference
        dt = np.diff(times)
        # Weight by time, not count
        # ... implementation
    else:
        # Return counts (existing behavior)
        # ... existing implementation
```

### **Implementation**

```python
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
import warnings

if TYPE_CHECKING:
    from neurospatial import Environment

def spikes_to_field(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    min_occupancy_seconds: float = 0.5,
) -> NDArray[np.float64]:
    """
    Convert spike train to occupancy-normalized firing rate field.

    This is the standard method for place field analysis (O'Keefe & Dostrovsky 1971).
    Computes firing rate in each spatial bin, excluding bins with insufficient occupancy.

    Parameters
    ----------
    env : Environment
        Spatial environment for binning.
    spike_times : array, shape (n_spikes,)
        Spike timestamps (seconds).
    times : array, shape (n_timepoints,)
        Trajectory timestamps (seconds).
    positions : array, shape (n_timepoints, n_dims)
        Position trajectory over time.
    min_occupancy_seconds : float, default=0.5
        Minimum occupancy (seconds) for valid bins. Bins with less occupancy
        are set to NaN. Default of 0.5 seconds is commonly used in place field
        literature (Wilson & McNaughton, 1993).

    Returns
    -------
    firing_rate : array, shape (n_bins,)
        Occupancy-normalized firing rate (Hz). Bins with insufficient
        occupancy are NaN.

    Notes
    -----
    **Algorithm**:
    1. Validate inputs and filter spikes to valid time range
    2. Compute occupancy using ``env.occupancy(times, positions, return_seconds=True)``
    3. Interpolate spike positions from trajectory
    4. Assign spikes to bins using ``env.bin_at()``
    5. Count spikes per bin
    6. Normalize: ``firing_rate = spike_count / occupancy``
    7. Apply ``min_occupancy_seconds`` threshold (insufficient bins → NaN)

    **Edge cases handled**:
    - Empty spike train → returns all zeros
    - Spikes outside time range → filtered with warning
    - NaN in trajectory → those timepoints skipped in occupancy
    - Zero occupancy everywhere → warning + returns all NaN

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment, spikes_to_field
    >>> # Generate synthetic data
    >>> times = np.linspace(0, 60, 6000)  # 1 minute at 100 Hz
    >>> positions = np.random.randn(6000, 2) * 20
    >>> spike_times = times[::10]  # Spike every 10 samples (~10 Hz)
    >>> # Create environment
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> # Convert spikes to field (standard method)
    >>> firing_rate = spikes_to_field(env, spike_times, times, positions)
    >>> # firing_rate is in Hz, with NaN for low-occupancy bins

    See Also
    --------
    compute_place_field : Convenience function with smoothing
    Environment.occupancy : Compute occupancy (used internally)
    Environment.smooth : Smooth firing rate field

    References
    ----------
    .. [1] O'Keefe & Dostrovsky (1971). The hippocampus as a spatial map.
           Brain Research 34(1).
    .. [2] Wilson & McNaughton (1993). Dynamics of the hippocampal ensemble
           code for space. Science 261(5124).
    """
    # Step 0: Validate inputs
    if times.shape[0] != positions.shape[0]:
        raise ValueError(
            f"times and positions must have same length "
            f"(got times: {times.shape[0]}, positions: {positions.shape[0]})"
        )

    # Handle empty spike train
    if spike_times.shape[0] == 0:
        return np.zeros(env.n_bins, dtype=np.float64)

    # Filter spikes to valid time range
    valid_mask = (spike_times >= times[0]) & (spike_times <= times[-1])
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        warnings.warn(
            f"{n_invalid} spike(s) outside trajectory time range "
            f"[{times[0]:.2f}, {times[-1]:.2f}]. These spikes will be ignored.",
            UserWarning,
            stacklevel=2
        )
    spike_times = spike_times[valid_mask]

    if spike_times.shape[0] == 0:
        warnings.warn(
            "All spikes were outside trajectory time range. Returning zeros.",
            UserWarning,
            stacklevel=2
        )
        return np.zeros(env.n_bins, dtype=np.float64)

    # Step 1: Compute occupancy
    occupancy = env.occupancy(times, positions, return_seconds=True)

    # Step 2: Interpolate spike positions
    if positions.ndim == 1:
        # 1D trajectory
        spike_positions = np.interp(spike_times, times, positions)[:, None]
    elif positions.shape[1] == 1:
        # 2D array with 1 column
        spike_positions = np.interp(spike_times, times, positions[:, 0])[:, None]
    else:
        # Multi-dimensional trajectory
        spike_positions = np.column_stack([
            np.interp(spike_times, times, positions[:, dim])
            for dim in range(positions.shape[1])
        ])

    # Step 3: Assign spikes to bins
    spike_bins = env.bin_at(spike_positions)

    # Filter out spikes that fell outside environment
    valid_spikes = spike_bins >= 0
    if not valid_spikes.all():
        n_outside = (~valid_spikes).sum()
        warnings.warn(
            f"{n_outside} spike(s) fell outside environment bounds and will be ignored.",
            UserWarning,
            stacklevel=2
        )
    spike_bins = spike_bins[valid_spikes]

    # Step 4: Count spikes per bin
    spike_counts = np.bincount(spike_bins, minlength=env.n_bins).astype(np.float64)

    # Step 5: Normalize by occupancy
    firing_rate = np.zeros(env.n_bins, dtype=np.float64)
    valid_mask = occupancy >= min_occupancy_seconds

    if not valid_mask.any():
        warnings.warn(
            f"No bins have occupancy >= {min_occupancy_seconds} seconds. "
            f"Returning all NaN. Consider lowering min_occupancy_seconds or "
            f"checking your trajectory data.",
            UserWarning,
            stacklevel=2
        )
        firing_rate[:] = np.nan
    else:
        firing_rate[valid_mask] = spike_counts[valid_mask] / occupancy[valid_mask]
        firing_rate[~valid_mask] = np.nan

    return firing_rate


def compute_place_field(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    min_occupancy_seconds: float = 0.5,
    smoothing_bandwidth: float | None = None,
) -> NDArray[np.float64]:
    """
    Compute smoothed place field from spike train (complete workflow).

    This is a convenience function combining spikes_to_field() and
    env.smooth() for the most common use case in place field analysis.

    Parameters
    ----------
    env : Environment
        Spatial environment for binning.
    spike_times : array, shape (n_spikes,)
        Spike timestamps (seconds).
    times : array, shape (n_timepoints,)
        Trajectory timestamps (seconds).
    positions : array, shape (n_timepoints, n_dims)
        Position trajectory.
    min_occupancy_seconds : float, default=0.5
        Minimum occupancy threshold (seconds).
    smoothing_bandwidth : float, optional
        Gaussian smoothing bandwidth (same units as environment).
        If None, no smoothing is applied.

    Returns
    -------
    place_field : array, shape (n_bins,)
        Occupancy-normalized, optionally smoothed firing rate (Hz).

    Examples
    --------
    >>> # One-line place field computation with smoothing
    >>> place_field = compute_place_field(
    ...     env, spike_times, times, positions,
    ...     smoothing_bandwidth=5.0
    ... )

    >>> # Equivalent to:
    >>> field = spikes_to_field(env, spike_times, times, positions)
    >>> place_field = env.smooth(field, bandwidth=5.0)

    See Also
    --------
    spikes_to_field : Convert spikes to field (no smoothing)
    Environment.smooth : Smooth a field
    Environment.smooth_batch : Batch smoothing for multiple neurons
    """
    field = spikes_to_field(env, spike_times, times, positions,
                           min_occupancy_seconds=min_occupancy_seconds)

    if smoothing_bandwidth is not None:
        field = env.smooth(field, smoothing_bandwidth)

    return field
```

### **Testing**

- Add to `env.occupancy()`: Test `return_seconds` parameter works correctly
- Create `tests/test_spike_field.py`
- Test: `test_spikes_to_field_synthetic()` - known spike rate → expected field
- Test: `test_spikes_to_field_min_occupancy()` - low occupancy bins → NaN
- Test: `test_spikes_to_field_empty_spikes()` - empty spike train → all zeros
- Test: `test_spikes_to_field_out_of_bounds_time()` - spikes outside time range → warning + filtered
- Test: `test_spikes_to_field_out_of_bounds_space()` - spikes outside environment → warning + filtered
- Test: `test_spikes_to_field_1d_trajectory()` - handles 1D positions correctly
- Test: `test_spikes_to_field_matches_manual()` - compare with manual computation
- Test: `test_compute_place_field_with_smoothing()` - matches spikes_to_field + smooth
- Run: `uv run pytest tests/test_spike_field.py -v`

### **Export in public API**

`src/neurospatial/__init__.py`:

```python
from neurospatial.spike_field import spikes_to_field, compute_place_field
```

### **Effort**: 4 days (+1 day for fixes and convenience function)

---

## 0.2 Reward Field Primitives (Week 1, Days 5-6 + Week 2, Day 1)

**New file**: `src/neurospatial/reward.py`

### **Implementation**

```python
from __future__ import annotations
from typing import TYPE_CHECKING, Literal
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial import Environment

def region_reward_field(
    env: Environment,
    region_name: str,
    *,
    reward_value: float = 1.0,
    decay: Literal["constant", "linear", "gaussian"] = "constant",
    bandwidth: float | None = None,
) -> NDArray[np.float64]:
    """
    Create reward field from named region with optional spatial decay.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    region_name : str
        Name of region in ``env.regions``.
    reward_value : float, default=1.0
        Reward value at region center/interior.
    decay : {'constant', 'linear', 'gaussian'}, default='constant'
        Spatial decay function:

        - 'constant': Binary reward (reward_value inside region, 0 outside)
        - 'linear': Linear decay from region boundary to environment edge
        - 'gaussian': Gaussian falloff from region (requires bandwidth)
    bandwidth : float, optional
        Bandwidth for Gaussian decay. Required if decay='gaussian'.
        Controls spatial scale of decay (same units as environment).

    Returns
    -------
    reward_field : array, shape (n_bins,)
        Reward value at each bin.

    Raises
    ------
    ValueError
        If decay='gaussian' but bandwidth is None.
    KeyError
        If region_name not in env.regions.

    Examples
    --------
    >>> # Binary reward in goal region
    >>> reward = region_reward_field(env, "goal", reward_value=10.0)

    >>> # Gaussian decay from goal (smooth potential field)
    >>> reward_smooth = region_reward_field(
    ...     env, "goal", reward_value=10.0, decay="gaussian", bandwidth=5.0
    ... )

    >>> # Linear decay to boundaries
    >>> reward_linear = region_reward_field(
    ...     env, "goal", reward_value=10.0, decay="linear"
    ... )

    See Also
    --------
    goal_reward_field : Create reward from goal bin indices
    distance_field : Compute distances to goal locations

    Notes
    -----
    **Reward shaping**: Distance-based rewards provide gradient information
    for reinforcement learning, making sparse-reward problems more tractable.

    **Caution**: Shaped rewards can bias learning if not carefully designed.
    See Ng et al. (1999) for theory of potential-based reward shaping.

    References
    ----------
    .. [1] Ng, Harada, & Russell (1999). Policy invariance under reward
           transformations. ICML 1999.
    """
    from neurospatial import regions_to_mask, distance_field

    # Validate region exists
    if region_name not in env.regions:
        raise KeyError(
            f"Region '{region_name}' not found in environment. "
            f"Available regions: {list(env.regions.keys())}"
        )

    # Get bins in region
    region_mask = regions_to_mask(env, [region_name])

    if decay == "constant":
        # Binary reward
        reward_field = np.where(region_mask, reward_value, 0.0)

    elif decay == "linear":
        # Linear decay from region boundary to environment edge
        region_bins = np.where(region_mask)[0]
        distances = distance_field(env.connectivity, sources=region_bins.tolist())
        max_dist = distances.max()

        # Linear decay: reward_value at region (distance=0), 0 at max_dist
        reward_field = reward_value * np.maximum(0, 1 - distances / max_dist)

    elif decay == "gaussian":
        if bandwidth is None:
            raise ValueError(
                "bandwidth parameter required for decay='gaussian'.\n"
                "\n"
                "Example:\n"
                "  reward = region_reward_field(env, 'goal', decay='gaussian', bandwidth=5.0)\n"
                "\n"
                "The bandwidth controls the spatial scale of Gaussian decay "
                "(same units as environment)."
            )

        # Create indicator field
        indicator = np.where(region_mask, 1.0, 0.0)

        # Smooth the indicator (creates Gaussian falloff from region)
        smoothed = env.smooth(indicator, bandwidth=bandwidth)

        # FIXED: Scale by max value IN REGION (not global max)
        max_in_region = smoothed[region_mask].max() if region_mask.any() else 1.0
        reward_field = smoothed / max_in_region * reward_value

    else:
        raise ValueError(
            f"Unknown decay '{decay}'. "
            f"Choose 'constant', 'linear', or 'gaussian'."
        )

    return reward_field


def goal_reward_field(
    env: Environment,
    goal_bins: list[int] | NDArray[np.int_],
    *,
    decay: Literal["linear", "exponential", "inverse"] = "exponential",
    scale: float = 1.0,
    max_distance: float | None = None,
) -> NDArray[np.float64]:
    """
    Create distance-based reward field from goal location(s).

    This is a common reward shaping primitive for RL, providing denser
    reward signal than sparse goal rewards.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    goal_bins : list or array of int
        Bin indices for goal location(s). Distance is computed to nearest goal.
    decay : {'linear', 'exponential', 'inverse'}, default='exponential'
        Distance decay function:

        - 'linear': reward = scale * max(0, 1 - d / max_d)
        - 'exponential': reward = scale * exp(-d / scale)
        - 'inverse': reward = scale / (1 + d)
    scale : float, default=1.0
        Reward scale parameter. Interpretation depends on decay:

        - linear: Maximum reward value (at goal)
        - exponential: Reward at goal AND decay rate
        - inverse: Reward at goal (d=0: reward=scale)
    max_distance : float, optional
        Maximum distance for linear decay. If None, uses environment diameter
        (maximum distance between any two bins).

    Returns
    -------
    reward_field : array, shape (n_bins,)
        Reward value at each bin. Higher values closer to goal.

    Notes
    -----
    **Reward shaping**: These distance-based rewards provide gradient information
    for learning, making sparse-reward RL problems more tractable.

    **Caution**: Shaped rewards can bias learning if not carefully designed.
    See Ng et al. (1999) for theory of potential-based reward shaping.

    **Decay comparison**:
    - Exponential: Smooth, infinite support (never reaches 0)
    - Linear: Reaches exactly 0 at max_distance (finite support)
    - Inverse: Hyperbolic, very slow decay (never reaches 0)

    Examples
    --------
    >>> # Exponential decay from goal (most common)
    >>> goal_bin = env.bin_at(np.array([[50.0, 50.0]]))[0]
    >>> reward = goal_reward_field(env, [goal_bin], decay="exponential", scale=10.0)

    >>> # Linear decay (reaches 0 at boundaries)
    >>> reward_linear = goal_reward_field(
    ...     env, [goal_bin], decay="linear", scale=10.0
    ... )

    >>> # Multiple goals (reward is distance to NEAREST goal)
    >>> goals = [10, 15, 20]  # Three goal bins
    >>> reward = goal_reward_field(env, goals, decay="exponential", scale=5.0)

    See Also
    --------
    region_reward_field : Create reward from named region
    distance_field : Compute distances to goal bins

    References
    ----------
    .. [1] Ng, Harada, & Russell (1999). Policy invariance under reward
           transformations. ICML 1999.
    """
    from neurospatial import distance_field

    # Validate goal_bins
    goal_bins = np.asarray(goal_bins, dtype=np.int_)
    if goal_bins.ndim == 0:
        goal_bins = goal_bins[None]  # Scalar to 1D array

    if not np.all((goal_bins >= 0) & (goal_bins < env.n_bins)):
        raise ValueError(
            f"goal_bins must be valid bin indices in range [0, {env.n_bins}). "
            f"Got: {goal_bins[~((goal_bins >= 0) & (goal_bins < env.n_bins))]}"
        )

    # Compute distance to nearest goal bin
    distances = distance_field(env.connectivity, sources=goal_bins.tolist())

    if decay == "linear":
        if max_distance is None:
            max_distance = distances.max()

        if max_distance <= 0:
            raise ValueError(f"max_distance must be positive (got {max_distance})")

        reward_field = scale * np.maximum(0, 1 - distances / max_distance)

    elif decay == "exponential":
        if scale <= 0:
            raise ValueError(f"scale must be positive for exponential decay (got {scale})")

        reward_field = scale * np.exp(-distances / scale)

    elif decay == "inverse":
        reward_field = scale / (1 + distances)

    else:
        raise ValueError(
            f"Unknown decay '{decay}'. "
            f"Choose 'linear', 'exponential', or 'inverse'."
        )

    return reward_field
```

### **Testing**

- Create `tests/test_reward.py`
- Test: `test_region_reward_field_constant()` - binary reward in region
- Test: `test_region_reward_field_linear()` - linear decay from boundary
- Test: `test_region_reward_field_gaussian()` - smooth falloff, peak maintains value IN REGION
- Test: `test_region_reward_field_validation()` - bandwidth required for Gaussian, region must exist
- Test: `test_goal_reward_field_exponential()` - exponential decay from goal
- Test: `test_goal_reward_field_linear()` - reaches zero at max distance
- Test: `test_goal_reward_field_inverse()` - inverse distance formula
- Test: `test_goal_reward_field_multiple_goals()` - nearest goal dominates
- Test: `test_goal_reward_field_validation()` - invalid goal bins raise error
- Run: `uv run pytest tests/test_reward.py -v`

### **Export in public API**

`src/neurospatial/__init__.py`:

```python
from neurospatial.reward import region_reward_field, goal_reward_field
```

### **Effort**: 2 days (same as original plan)

---

## 0.3 Documentation (Week 2, Days 2-3)

**Documentation**:

- Create `docs/user-guide/spike-field-primitives.md`
  - Section: Converting spike trains to spatial fields
  - Section: Why occupancy normalization matters (place field analysis standard)
  - Section: `compute_place_field()` convenience function for one-liner workflows
  - Section: Batch operations for performance (10-50× speedup realistic)
  - Section: Min occupancy threshold (best practices: 0.5 seconds typical)
  - Include code examples and visualizations

- Create or update `docs/user-guide/rl-primitives.md`
  - Section: Reward field generation from regions
  - Section: Reward shaping strategies (potential-based, distance-based)
  - Section: Distance-based rewards (exponential, linear, inverse)
  - Section: Cautions about reward shaping (Ng et al. 1999 reference)
  - Include RL-specific examples (goal-directed navigation)

**Example Notebook**:

- Create `examples/00_spike_field_basics.ipynb`
  - Example 1: Convert spike train → firing rate map
    - Generate synthetic data (trajectory + spike times)
    - Create environment
    - Compute firing rate with `spikes_to_field()`
    - Show `compute_place_field()` convenience function
    - Visualize: occupancy, spike counts, firing rate
  - Example 2: Batch smooth 100 neurons (performance comparison)
    - Generate 100 synthetic neurons
    - Compare: looping `env.smooth()` vs `env.smooth_batch()`
    - Benchmark timing (target: 10-50× speedup)
    - Visualize: smoothed fields for multiple neurons
  - Example 3: Create reward field for RL
    - Define goal region
    - Create constant reward field
    - Create exponential decay reward
    - Create Gaussian falloff reward
    - Visualize all three side-by-side
  - Add explanatory markdown cells throughout
  - Add performance notes and best practices

**Testing**:

- Run all Phase 0 tests with updated signatures
- Verify coverage: >95% for new code
- Run example notebook: verify all cells execute

### **Effort**: 2 days (same as original plan)

---

## Phase 0 Success Criteria (Updated)

- [ ] `spikes_to_field()` uses correct parameter order (env first, times/positions match existing API)
- [ ] `compute_place_field()` convenience function works for one-liner workflows
- [ ] Min occupancy threshold correctly filters unreliable bins (→ NaN)
- [ ] Spike interpolation handles 1D and multi-dimensional trajectories correctly
- [ ] Input validation comprehensive (empty spikes, out-of-range, NaN handling)
- [ ] `env.smooth_batch()` provides 10-50× speedup vs looping for 100 neurons
- [ ] `env.smooth_batch()` warns and falls back for large environments (>10k bins)
- [ ] `normalize_fields()` works on multi-dimensional arrays (any axis)
- [ ] `region_reward_field()` supports all three decay types with correct parameter name
- [ ] `goal_reward_field()` supports all three decay functions with consistent naming
- [ ] Gaussian falloff rescaling uses max IN REGION (not global max)
- [ ] All tests pass with >95% coverage for new code
- [ ] Zero mypy errors (no `type: ignore` comments)
- [ ] Example notebook demonstrates all primitives with clear visualizations
- [ ] Documentation complete and cross-referenced

---

## Package Design Summary

**Module organization**:

```
src/neurospatial/
    spike_field.py               # spikes_to_field(), compute_place_field()
    reward.py                    # region_reward_field(), goal_reward_field()
    environment/
        trajectory.py            # ADDITIONS: return_seconds parameter to occupancy()
```

**Public API** (`src/neurospatial/__init__.py`):

```python
# Phase 0: Core spike/field primitives
from neurospatial.spike_field import spikes_to_field, compute_place_field
from neurospatial.reward import region_reward_field, goal_reward_field
```

**Usage patterns**:

```python
# Spike → field conversion
from neurospatial import spikes_to_field, compute_place_field

# Standard usage
field = spikes_to_field(env, spike_times, times, positions)

# One-liner with smoothing
place_field = compute_place_field(env, spike_times, times, positions,
                                 smoothing_bandwidth=5.0)

# Reward fields
from neurospatial import region_reward_field, goal_reward_field
reward1 = region_reward_field(env, "goal", decay="gaussian", bandwidth=5.0)
reward2 = goal_reward_field(env, [42], decay="exponential", scale=10.0)
```

---

## Timeline Summary (Revised)

| Task | Days | Cumulative |
|------|------|------------|
| 0.1 Spike → Field + Convenience | 4 days | 4 days |
| 0.2 Reward Fields | 3 days | 7 days |
| 0.3 Documentation | 2 days | **9 days** |

**Total: 9 days (1.8 weeks, ~2 weeks)**

**Deferred to v0.3.1**: Batch operations (`smooth_batch()`, `normalize_fields()`) - performance optimizations to be added after core API validation with users

---

## Phase 1: Core Differential Operators (Weeks 3-5)

### Goal

Implement weighted differential operator infrastructure following PyGSP's approach.

### Components

#### 1.1 Differential Operator Matrix (Week 1)

**New file**: `src/neurospatial/differential.py`

**Implementation**:

```python
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy import sparse
from functools import cached_property

if TYPE_CHECKING:
    from neurospatial import Environment

def compute_differential_operator(env: Environment) -> sparse.csc_matrix:
    """
    Compute weighted differential operator D.

    The differential operator is a sparse (n_bins, n_edges) matrix where:
    - D[source, edge] = -√(edge_weight)
    - D[target, edge] = +√(edge_weight)

    This enables:
    - Gradient: D.T @ field
    - Divergence: D @ edge_field
    - Laplacian: D @ D.T

    Parameters
    ----------
    env : Environment
        Spatial environment with connectivity graph

    Returns
    -------
    D : sparse matrix, shape (n_bins, n_edges)
        Differential operator in CSC format

    References
    ----------
    .. [1] PyGSP: https://pygsp.readthedocs.io/
    """
    G = env.connectivity
    n_edges = G.number_of_edges()
    n_bins = env.n_bins

    # Pre-allocate arrays
    sources = np.empty(n_edges, dtype=np.int32)
    targets = np.empty(n_edges, dtype=np.int32)
    weights = np.empty(n_edges, dtype=np.float64)

    # Extract edge data (minimal Python loop)
    for idx, (u, v, data) in enumerate(G.edges(data=True)):
        sources[idx] = u
        targets[idx] = v
        weights[idx] = data['distance']

    # Construct sparse matrix (vectorized)
    sqrt_weights = np.sqrt(weights)
    rows = np.concatenate([sources, targets])
    cols = np.tile(np.arange(n_edges), 2)
    vals = np.concatenate([-sqrt_weights, sqrt_weights])

    D = sparse.csc_matrix((vals, (rows, cols)), shape=(n_bins, n_edges))
    return D
```

**Add to Environment class** (`src/neurospatial/environment/core.py`):

```python
@cached_property
def differential_operator(self) -> sparse.csc_matrix:
    """
    Weighted differential operator for graph signal processing.

    Cached for performance (50x speedup for repeated operations).

    Returns
    -------
    D : sparse matrix, shape (n_bins, n_edges)
        Differential operator
    """
    from neurospatial.differential import compute_differential_operator
    return compute_differential_operator(self)
```

**Tests** (`tests/test_differential.py`):

```python
def test_differential_operator_shape(env_regular_grid_2d):
    """D should have shape (n_bins, n_edges)"""
    D = env_regular_grid_2d.differential_operator
    n_edges = env_regular_grid_2d.connectivity.number_of_edges()
    assert D.shape == (env_regular_grid_2d.n_bins, n_edges)

def test_laplacian_from_differential(env_regular_grid_2d):
    """D @ D.T should equal graph Laplacian"""
    D = env_regular_grid_2d.differential_operator
    L_from_D = (D @ D.T).toarray()

    import networkx as nx
    L_nx = nx.laplacian_matrix(env_regular_grid_2d.connectivity).toarray()

    np.testing.assert_allclose(L_from_D, L_nx, atol=1e-10)

def test_differential_operator_caching(env_regular_grid_2d):
    """Repeated calls should return cached object"""
    D1 = env_regular_grid_2d.differential_operator
    D2 = env_regular_grid_2d.differential_operator
    assert D1 is D2  # Same object, not recomputed
```

**Effort**: 3 days
**Risk**: Low - validated in benchmark
**Blockers**: None

---

#### 1.2 Gradient Operator (Week 2)

**Add to**: `src/neurospatial/differential.py`

**Implementation**:

```python
def gradient(
    field: NDArray[np.float64],
    env: Environment,
) -> NDArray[np.float64]:
    """
    Compute gradient of scalar field on graph.

    The gradient transforms a scalar field on nodes to a vector field on edges,
    measuring the rate of change along each edge.

    Parameters
    ----------
    field : array, shape (n_bins,)
        Scalar field values at each bin
    env : Environment
        Spatial environment

    Returns
    -------
    grad_field : array, shape (n_edges,)
        Gradient values on each edge.
        Positive = increasing from source to target
        Negative = decreasing from source to target

    Notes
    -----
    Uses weighted differential operator: grad = D.T @ field

    Examples
    --------
    >>> # Compute gradient of distance field
    >>> distances = env.distance_to([goal_bin])
    >>> grad_dist = gradient(distances, env)
    >>> # Negative gradient points toward goal

    See Also
    --------
    divergence : Compute divergence of edge field
    """
    D = env.differential_operator
    return D.T @ field
```

**Public API** (`src/neurospatial/__init__.py`):

```python
from neurospatial.differential import gradient, divergence
```

**Tests**:

```python
def test_gradient_shape(env_regular_grid_2d):
    """Gradient should have one value per edge"""
    field = np.random.randn(env_regular_grid_2d.n_bins)
    grad = gradient(field, env_regular_grid_2d)
    n_edges = env_regular_grid_2d.connectivity.number_of_edges()
    assert grad.shape == (n_edges,)

def test_gradient_constant_field(env_regular_grid_2d):
    """Gradient of constant field should be zero"""
    field = np.ones(env_regular_grid_2d.n_bins) * 5.0
    grad = gradient(field, env_regular_grid_2d)
    np.testing.assert_allclose(grad, 0.0, atol=1e-10)

def test_gradient_linear_field_regular_grid():
    """Gradient should be constant for linear field on regular grid"""
    # Create 5x5 regular grid
    env = Environment.from_samples(
        np.random.randn(100, 2) * 50, bin_size=10.0
    )

    # Linear field: f(x, y) = x
    field = env.bin_centers[:, 0]
    grad = gradient(field, env)

    # All horizontal edges should have same gradient
    # (with appropriate sign for direction)
    # This is a sanity check, not exact due to discretization
```

**Effort**: 2 days
**Risk**: Low
**Blockers**: Differential operator (1.1)

---

#### 1.3 Divergence Operator (Week 2)

**Note**: Current `divergence()` is KL/JS divergence - will be renamed to `kl_divergence()`.

**Rename existing function** (`src/neurospatial/field_ops.py`):

```python
def kl_divergence(p, q, *, kind='kl', eps=1e-12):
    """
    Compute KL or JS divergence between probability distributions.

    Renamed from `divergence()` to avoid confusion with vector field divergence.
    """
    # Existing implementation (unchanged)
```

**New implementation** (`src/neurospatial/differential.py`):

```python
def divergence(
    edge_field: NDArray[np.float64],
    env: Environment,
) -> NDArray[np.float64]:
    """
    Compute divergence of edge field on graph.

    The divergence transforms a vector field on edges to a scalar field on nodes,
    measuring the net outflow at each node.

    Parameters
    ----------
    edge_field : array, shape (n_edges,)
        Vector field values on edges
    env : Environment
        Spatial environment

    Returns
    -------
    div_field : array, shape (n_bins,)
        Divergence at each bin.
        Positive = net outflow
        Negative = net inflow

    Notes
    -----
    Uses weighted differential operator: div = D @ edge_field

    Examples
    --------
    >>> # Compute divergence of gradient (= Laplacian)
    >>> field = env.distance_to([goal_bin])
    >>> grad_field = gradient(field, env)
    >>> div_grad = divergence(grad_field, env)
    >>> # div_grad is the Laplacian of field

    See Also
    --------
    gradient : Compute gradient of scalar field
    """
    D = env.differential_operator
    return D @ edge_field
```

**Tests**:

```python
def test_divergence_gradient_is_laplacian(env_regular_grid_2d):
    """div(grad(f)) should equal Laplacian(f)"""
    field = np.random.randn(env_regular_grid_2d.n_bins)

    # Compute via gradient → divergence
    grad_field = gradient(field, env_regular_grid_2d)
    div_grad = divergence(grad_field, env_regular_grid_2d)

    # Compute Laplacian directly
    import networkx as nx
    L = nx.laplacian_matrix(env_regular_grid_2d.connectivity).toarray()
    laplacian_field = L @ field

    np.testing.assert_allclose(div_grad, laplacian_field, atol=1e-10)
```

**Effort**: 2 days (including renaming existing function)
**Risk**: Low (no users to migrate)
**Blockers**: Differential operator (1.1)

---

#### 1.4 Documentation & Examples (Week 3)

**New user guide**: `docs/user-guide/differential-operators.md`

**Content**:

- What are differential operators?
- When to use gradient vs divergence
- Relationship to Laplacian
- Examples: value gradients, flow fields
- Mathematical background

**Example notebook**: `examples/09_differential_operators.ipynb`

**Effort**: 5 days
**Risk**: Low

---

## Phase 2: Basic Signal Processing Primitives (Weeks 6-8)

### Goal

Implement foundational spatial signal processing operations.

### Components

#### 2.1 neighbor_reduce (Week 4)

**Add to**: `src/neurospatial/primitives.py` (new module)

**Implementation**:

```python
def neighbor_reduce(
    field: NDArray[np.float64],
    env: Environment,
    *,
    op: Literal['sum', 'mean', 'max', 'min', 'std'] = 'mean',
    weights: NDArray[np.float64] | None = None,
    include_self: bool = False,
) -> NDArray[np.float64]:
    """
    Aggregate field values over graph neighborhoods.

    This is a fundamental primitive for local spatial operations.

    Parameters
    ----------
    field : array, shape (n_bins,)
        Field values to aggregate
    env : Environment
        Spatial environment
    op : {'sum', 'mean', 'max', 'min', 'std'}, default='mean'
        Aggregation operation
    weights : array, shape (n_bins,), optional
        Per-bin weights for weighted aggregation
    include_self : bool, default=False
        If True, include center bin in aggregation

    Returns
    -------
    aggregated : array, shape (n_bins,)
        Aggregated values at each bin

    Examples
    --------
    >>> # Compute local mean firing rate
    >>> local_mean = neighbor_reduce(firing_rate, env, op='mean')

    >>> # Coherence: correlation with neighbor average
    >>> neighbor_avg = neighbor_reduce(firing_rate, env, op='mean')
    >>> coherence = np.corrcoef(firing_rate, neighbor_avg)[0, 1]

    See Also
    --------
    gradient : Directional rate of change
    smooth : Gaussian smoothing
    """
    # Implementation from primitives_poc.py
    # Optimized with vectorization where possible
```

**Tests**:

```python
def test_neighbor_reduce_mean_regular_grid():
    """Mean of neighbors on regular grid should average adjacent values"""
    # 3x3 grid, center has value 10, neighbors have value 1
    # Mean of 4 neighbors = 1.0

def test_neighbor_reduce_include_self():
    """include_self should include center bin in aggregation"""
    # Verify that include_self=True changes result

def test_neighbor_reduce_weights():
    """Weighted aggregation should respect weights"""
    # Test distance-weighted neighbor aggregation
```

**Effort**: 3 days
**Risk**: Low (prototype exists)
**Blockers**: None

---

#### 2.2 convolve (Week 5-6)

**Add to**: `src/neurospatial/primitives.py`

**Implementation**:

```python
def convolve(
    field: NDArray[np.float64],
    kernel: Callable[[NDArray[np.float64]], float] | NDArray[np.float64],
    env: Environment,
    *,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """
    Apply custom kernel to spatial field.

    Extends `smooth()` to support arbitrary kernels.

    Parameters
    ----------
    field : array, shape (n_bins,)
        Field to convolve
    kernel : callable or array
        Kernel function: distance → weight
        Or: precomputed kernel matrix (n_bins, n_bins)
    env : Environment
        Spatial environment
    normalize : bool, default=True
        If True, normalize kernel weights to sum to 1

    Returns
    -------
    convolved : array, shape (n_bins,)
        Convolved field

    Examples
    --------
    >>> # Box kernel (uniform within radius)
    >>> def box_kernel(dist, radius=10.0):
    ...     return (dist <= radius).astype(float)
    >>> smoothed = convolve(field, lambda d: box_kernel(d, 10.0), env)

    >>> # Mexican hat (difference of Gaussians)
    >>> def mexican_hat(dist, sigma=5.0):
    ...     return np.exp(-dist**2 / (2*sigma**2)) - 0.5*np.exp(-dist**2 / (2*(2*sigma)**2))
    >>> filtered = convolve(field, mexican_hat, env)
    """
    # If callable, compute kernel matrix
    # Apply convolution
    # Handle NaN values
```

**Effort**: 3 days
**Risk**: Low (extends existing smooth())
**Blockers**: None

---

## Phase 3: Core Metrics Module (Weeks 8.5-10)

### Goal

Provide standard neuroscience metrics as convenience wrappers.

### Module Structure

```
src/neurospatial/metrics/
    __init__.py
    place_fields.py      # Individual place field properties
    population.py        # Population-level metrics
    boundary_cells.py    # Border score, head direction
```

### Components

#### 3.1 Place Field Metrics (Week 7)

**File**: `src/neurospatial/metrics/place_fields.py`

**Functions**:

```python
def detect_place_fields(
    firing_rate: NDArray,
    env: Environment,
    *,
    threshold: float = 0.2,
    min_size: float | None = None,
    max_mean_rate: float | None = 10.0,
    detect_subfields: bool = True,
) -> list[NDArray[np.int64]]:
    """
    Detect place fields as connected components above threshold.

    Implements iterative peak-based detection with optional coalescent
    subfield discrimination (neurocode approach) and interneuron exclusion
    (vandermeerlab approach).

    Parameters
    ----------
    firing_rate : array
        Firing rate map
    env : Environment
        Spatial environment
    threshold : float, default=0.2
        Fraction of peak rate for field boundary (20%)
    min_size : float, optional
        Minimum field size in physical units
    max_mean_rate : float, optional
        Maximum mean firing rate (Hz) for interneuron exclusion.
        Cells exceeding this are excluded as likely interneurons.
        Default 10 Hz (vandermeerlab). Set to None to disable.
    detect_subfields : bool, default=True
        Apply recursive threshold to detect coalescent subfields

    Notes
    -----
    Interneuron exclusion from vandermeerlab:
    - Pyramidal cells (place cells): 0.5-5 Hz mean rate
    - Interneurons: 10-50 Hz mean rate
    - Default threshold (10 Hz) excludes high-firing cells
    """

def field_size(field_bins: NDArray, env: Environment) -> float:
    """Compute place field area in physical units."""

def field_centroid(
    firing_rate: NDArray,
    field_bins: NDArray,
    env: Environment,
) -> NDArray:
    """Compute place field center of mass."""

def skaggs_information(
    firing_rate: NDArray,
    occupancy: NDArray,
    *,
    base: float = 2.0,
) -> float:
    """Compute Skaggs spatial information content (bits/spike)."""

def sparsity(firing_rate: NDArray, occupancy: NDArray) -> float:
    """Compute sparsity measure (Skaggs et al. 1996)."""

def field_stability(
    rate_map_1: NDArray,
    rate_map_2: NDArray,
    *,
    method: Literal['pearson', 'spearman'] = 'pearson',
) -> float:
    """Compute spatial correlation between rate maps."""
```

**Tests**: Comprehensive unit tests for each function
**Effort**: 3 days
**Risk**: Low
**Blockers**: None

---

#### 3.2 Population Metrics (Week 7)

**File**: `src/neurospatial/metrics/population.py`

**Functions**:

```python
def population_coverage(
    all_place_fields: list[list[NDArray]],
    n_bins: int,
) -> float:
    """Fraction of environment covered by at least one field."""

def field_density_map(
    all_place_fields: list[list[NDArray]],
    n_bins: int,
) -> NDArray:
    """Number of overlapping fields at each location."""

def count_place_cells(
    spatial_information: dict[int, float],
    threshold: float = 0.5,
) -> int:
    """Count neurons exceeding spatial information threshold."""

def field_overlap(
    field_bins_i: NDArray,
    field_bins_j: NDArray,
) -> float:
    """Overlap coefficient between two fields."""

def population_vector_correlation(
    population_matrix: NDArray,
) -> NDArray:
    """Correlation matrix between population vectors at all locations."""
```

**Effort**: 2 days
**Risk**: Low

---

#### 3.3 Boundary Cell Metrics (Week 8)

**File**: `src/neurospatial/metrics/boundary_cells.py`

**Motivation**: Border cells (boundary vector cells) fire when the animal is near environmental boundaries. TSToolbox_Utils and opexebo provide validated implementations.

**Functions**:

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

    Notes
    -----
    Algorithm from TSToolbox_Utils and opexebo:
    1. Segment place field at 30% of peak rate
    2. Compute wall contact ratio for each wall
    3. Take maximum contact ratio (cM)
    4. Compute firing-rate-weighted distance to walls (d)
    5. Border score = (cM - d) / (cM + d)

    Only fields with area > min_area and wall contact are evaluated.

    References
    ----------
    .. [1] Solstad et al. (2008). Neuron 58(6).
    .. [2] TSToolbox_Utils Compute_BorderScore.m
    .. [3] opexebo.analysis.border_score

    Examples
    --------
    >>> # Boundary vector cell
    >>> score = border_score(firing_rate, env)
    >>> if score > 0.5:
    ...     print("Strong border cell!")
    """
    pass

def boundary_vector_tuning(
    firing_rate: NDArray,
    env: Environment,
    positions: NDArray,
) -> dict:
    """
    Analyze boundary vector cell tuning.

    Returns preferred distance to boundary and preferred allocentric
    direction to boundary.

    Returns
    -------
    tuning : dict
        - 'preferred_distance': float
        - 'preferred_angle': float
        - 'distance_tuning': array
        - 'angle_tuning': array
    """
    pass
```

**Tests**: Comprehensive unit tests
**Effort**: 2 days
**Risk**: Low (well-documented algorithm in TSToolbox_Utils and opexebo)
**Blockers**: None

---

#### 3.4 Documentation (Week 8)

**New user guide**: `docs/user-guide/neuroscience-metrics.md`

**Example notebooks**:

- `examples/10_place_field_analysis.ipynb` - Place field detection and metrics
- `examples/11_boundary_cell_analysis.ipynb` - Border score

**Effort**: 2 days

---

## Phase 4: Trajectory Metrics & Behavioral Segmentation (Weeks 10.5-13)

### Goal

Implement trajectory characterization metrics and automatic detection of behavioral epochs from continuous trajectories.

### Motivation

**Trajectory metrics**: Animal movement ecology packages provide trajectory characterization metrics that are broadly applicable to neuroscience.

**Behavioral segmentation**: Most packages require manual trial/epoch segmentation. neurospatial can provide spatial primitives for automatic detection of runs, laps, and trials based on spatial regions and trajectory patterns.

### Module Structure

```
src/neurospatial/metrics/
    trajectory.py        # Trajectory metrics from ecology

src/neurospatial/segmentation/
    __init__.py
    regions.py       # Region-based segmentation
    laps.py          # Lap detection
    trials.py        # Trial segmentation
    similarity.py    # Trajectory similarity
```

### Components

#### 4.1 Trajectory Metrics (Week 9)

**File**: `src/neurospatial/metrics/trajectory.py`

**Motivation**: Animal movement ecology packages (Traja, yupi, adehabitatHR) provide trajectory characterization metrics that are broadly applicable to neuroscience. These metrics quantify movement patterns and spatial usage.

**Functions** (detailed implementations in ECOLOGY_TRAJECTORY_PACKAGES.md):

- **`compute_turn_angles()`** - Path tortuosity (exploration vs exploitation)
- **`compute_step_lengths()`** - Graph distances between consecutive bins
- **`compute_home_range()`** - Bins containing X% of time (95% standard)
- **`mean_square_displacement()`** - Diffusion classification (MSD ~ t^α)

**Authority**: Traja, yupi, adehabitatHR (ecology literature)

**Integration with neurospatial**:

- Uses existing `env.distance_between()` for graph distances
- Uses existing `env.bin_centers` for angle computation
- Complements behavioral segmentation (movement characterization)

**Effort**: 3 days
**Risk**: Low (well-defined algorithms)
**Blockers**: None

---

#### 4.2 Region-Based Segmentation (Week 10, Days 1-3)

**File**: `src/neurospatial/segmentation/regions.py`

**Functions**:

```python
def detect_region_crossings(
    trajectory_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    region: str,
    env: Environment,
    direction: Literal['entry', 'exit', 'both'] = 'both',
) -> list[Crossing]:
    """Detect when trajectory enters/exits a spatial region."""

def detect_runs_between_regions(
    trajectory_positions: NDArray[np.float64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    source: str,
    target: str,
    min_duration: float = 0.5,
    max_duration: float = 10.0,
    velocity_threshold: float | None = None,
) -> list[Run]:
    """
    Identify trajectory segments from source region to target region.

    Returns Run objects with start_time, end_time, trajectory_bins,
    path_length, success (reached target).
    """

def segment_by_velocity(
    trajectory_positions: NDArray[np.float64],
    times: NDArray[np.float64],
    threshold: float,
    *,
    min_duration: float = 0.5,
    hysteresis: float = 2.0,
    smooth_window: float = 0.2,
) -> IntervalSet:
    """Segment trajectory into movement vs. rest periods."""
```

**Effort**: 3 days
**Risk**: Low

---

#### 4.3 Lap Detection (Week 10, Days 4-5)

**File**: `src/neurospatial/segmentation/laps.py`

**Functions**:

```python
def detect_laps(
    trajectory_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    method: Literal['auto', 'reference', 'region'] = 'auto',
    min_overlap: float = 0.8,
    direction: Literal['clockwise', 'counter', 'both'] = 'both',
) -> list[Lap]:
    """
    Detect complete loops/laps in circular or figure-8 tracks.

    Three methods:
    - 'auto': Detect lap template from first 10% of trajectory
    - 'reference': User provides reference lap
    - 'region': Detect crossings of lap start region

    Returns Lap objects with direction, overlap_score.
    """
```

**Effort**: 2 days
**Risk**: Low

---

#### 4.4 Trial Segmentation (Week 10, Day 6)

**File**: `src/neurospatial/segmentation/trials.py`

**Functions**:

```python
def segment_trials(
    trajectory_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    trial_type: Literal['tmaze', 'ymaze', 'radial_arm', 'custom'],
    start_region: str,
    end_regions: dict[str, str],
    min_duration: float = 1.0,
    max_duration: float = 15.0,
) -> list[Trial]:
    """
    Detect trials based on task structure (e.g., T-maze left/right).

    Returns Trial objects with outcome ('left'/'right'/etc.), success.
    """
```

**Effort**: 1 day
**Risk**: Low

---

#### 4.5 Trajectory Similarity (Week 11, Days 1-2)

**File**: `src/neurospatial/segmentation/similarity.py`

**Functions**:

```python
def trajectory_similarity(
    trajectory1_bins: NDArray[np.int64],
    trajectory2_bins: NDArray[np.int64],
    env: Environment,
    *,
    method: Literal['jaccard', 'correlation', 'hausdorff', 'dtw'] = 'jaccard',
) -> float:
    """
    Compute similarity between two trajectory segments.

    Methods:
    - 'jaccard': Spatial overlap (Jaccard index)
    - 'correlation': Sequential correlation
    - 'hausdorff': Maximum deviation
    - 'dtw': Dynamic time warping
    """

def detect_goal_directed_runs(
    trajectory_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    goal_region: str,
    directedness_threshold: float = 0.7,
    min_progress: float = 20.0,
) -> list[Run]:
    """
    Identify runs where trajectory moves toward a goal.

    Computes directedness = (dist_start_to_goal - dist_end_to_goal) / path_length
    """
```

**Effort**: 2 days
**Risk**: Low

---

#### 4.6 Tests & Documentation (Week 11, Days 3-5)

**Tests**:

- Test trajectory metrics on synthetic data (straight lines, circles)
- Test region crossing detection on synthetic trajectories
- Test lap detection on circular tracks (clockwise/counter-clockwise)
- Test trial segmentation on T-maze data
- Test trajectory similarity methods (Jaccard, DTW, etc.)

**Documentation**:

- User guide: `docs/user-guide/trajectory-and-behavioral-analysis.md`
- Example notebooks:
  - `examples/12_trajectory_analysis.ipynb` - Turn angles, MSD, home range
  - `examples/13_behavioral_segmentation.ipynb` - Runs, laps, trials

**Effort**: 3 days

---

### Integration with pynapple

All functions return `pynapple.IntervalSet` when available:

```python
try:
    import pynapple as nap
    return nap.IntervalSet(start=starts, end=ends)
except ImportError:
    return [(start, end) for start, end in zip(starts, ends)]
```

### Use Cases

1. **Goal-directed runs**: Analyze place fields during nest→goal runs
2. **Lap-by-lap learning**: Track place field stability across laps
3. **Trial-type selectivity**: Compare left vs. right trial firing
4. **Replay analysis**: Match decoded trajectories to behavioral runs
5. **Learning dynamics**: Performance/stability over trials

### Validation

- Compare lap detection with neurocode NSMAFindGoodLaps.m
- Test on simulated trajectories
- Cross-validate region crossing times

---

## Phase 5: Polish & Release (Weeks 13.5-15)

### Components

#### 5.1 Validation Against opexebo and neurocode

**Validation against analysis packages**:

- Test place field detection matches neurocode's subfield discrimination
- Test spatial information matches opexebo/neurocode/buzcode
- Test sparsity calculation matches opexebo
- Test border score matches TSToolbox_Utils/opexebo
- Test trajectory metrics on synthetic data (straight lines, circles)
- Document any intentional differences

**Authority**: opexebo (Moser lab), neurocode (AyA Lab), TSToolbox_Utils

**Effort**: 2 days

#### 5.2 Performance Optimization

- Profile critical paths
- Optimize hot loops
- Add caching where beneficial
- Benchmark against baseline

**Effort**: 2 days

#### 5.3 Documentation Polish

- API reference generation
- Cross-linking between docs
- Cross-references to opexebo, neurocode
- Tutorial videos (optional)

**Effort**: 3 days

#### 5.4 Release

- Version bump to 0.3.0
- Changelog highlighting:
  - Differential operators (gradient, divergence, Laplacian)
  - Place field & population metrics
  - Boundary cell metrics (border score)
  - Trajectory metrics (turn angles, home range, MSD)
  - Behavioral segmentation (runs, laps, trials)
  - Function rename (divergence → kl_divergence)
- Blog post / announcement mentioning deferred features (grid cells in v0.4.0)
- PyPI release

**Effort**: 2 days

---

## Success Criteria

### Phase 1 (Differential Operators)

- [ ] D matrix construction passes all tests
- [ ] gradient(), divergence() work on all layout types
- [ ] div(grad(f)) == Laplacian(f) validated
- [ ] 50x caching speedup confirmed
- [ ] Existing divergence() renamed to kl_divergence()

### Phase 2 (Signal Processing Primitives)

- [ ] neighbor_reduce() works on all layout types
- [ ] convolve() supports arbitrary kernels
- [ ] Tests pass for all layout types

### Phase 3 (Core Metrics Module)

- [ ] Place field detection matches neurocode's subfield discrimination
- [ ] Spatial information matches opexebo/neurocode/buzcode
- [ ] Sparsity calculation matches opexebo
- [ ] Border score matches TSToolbox_Utils/opexebo
- [ ] Trajectory metrics validated on synthetic data
- [ ] All metrics have examples and citations

### Phase 4 (Behavioral Segmentation)

- [ ] Region crossing detection works on synthetic trajectories
- [ ] Lap detection handles clockwise/counter-clockwise
- [ ] Trial segmentation works for T-maze, Y-maze
- [ ] Trajectory similarity methods validated
- [ ] pynapple IntervalSet integration works when available

### Phase 5 (Release)

- [ ] All tests pass (>95% coverage)
- [ ] Documentation complete
- [ ] Performance benchmarks meet targets
- [ ] Zero mypy errors
- [ ] Version 0.3.0 released

---

## Risk Management

**UPDATED after opexebo analysis**: Overall risk reduced from HIGH → MEDIUM

### Medium-Risk Items (reduced from HIGH)

**1. spatial_autocorrelation implementation** (was HIGH, now MEDIUM)

- **Status**: RISK REDUCED - Adopt opexebo's validated FFT approach
- **Mitigation**: Use opexebo's FFT method (fast, validated, field-standard)
- **Fallback**: Graph-based approach for irregular grids (optional, defer if needed)
- **Timeline buffer**: 4 weeks allocated (validated algorithm reduces uncertainty)
- **Validation**: Test against opexebo outputs, should match within 1%

**2. Function rename (divergence → kl_divergence)**

- **Status**: LOW RISK - no current users
- **Action**: Direct rename, update internal uses
- **Documentation**: Note rename in changelog

### Low-Risk Items

**3. Performance regressions**

- **Mitigation**: Benchmark suite in CI/CD
- **Monitoring**: Track key operations (smooth, distance_field, gradient)
- **Target**: No operation >10% slower than baseline
- **Baseline**: opexebo performance for regular grids

**4. API design conflicts**

- **Status**: LOW RISK - opexebo provides reference APIs
- **Mitigation**: Match opexebo signatures where possible
- **Validation**: User feedback on proposed extensions

**5. Grid score validation**

- **Status**: LOW RISK - opexebo provides gold standard
- **Mitigation**: Test against opexebo outputs (should match within 1%)
- **Resources**: opexebo test cases provide validation data
- **Authority**: Nobel Prize-winning lab implementation

**6. Algorithm correctness**

- **Status**: LOW RISK - adopting validated algorithms
- **Mitigation**: Cross-reference opexebo for all overlapping metrics
- **Testing**: Validate outputs match opexebo exactly for regular grids
- **Documentation**: Document intentional differences (irregular graph support)

---

## Effort Estimation

**UPDATED with Phase 0 (spike/field primitives)**:

| Phase | Duration | Person-Weeks | Risk Level |
|-------|----------|--------------|------------|
| 0. Spike/Field Primitives | 2.5 weeks | 2.5 | Low |
| 1. Differential Operators | 2.5 weeks | 2.5 | Low |
| 2. Basic Signal Processing | 2 weeks | 2 | Low |
| 3. Core Metrics Module | 1.5 weeks | 1.5 | Low |
| 4. Behavioral Segmentation | 2.5 weeks | 2.5 | Low |
| 5. Polish & Release | 1.5 weeks | 1.5 | Low |
| **Total** | **15 weeks** | **15** | **Low overall** |

**Phase breakdown**:

- **Phase 0** (2.5 weeks): `spikes_to_field()`, batch operations, reward fields
- **Phase 1** (2.5 weeks): Differential operator, gradient, divergence, docs
- **Phase 2** (2 weeks): neighbor_reduce, convolve
- **Phase 3** (1.5 weeks): Place fields, population, boundary cells, trajectory metrics, docs
- **Phase 4** (2.5 weeks): Region-based segmentation, laps, trials, similarity, docs
- **Phase 5** (1.5 weeks): Validation (2 days), performance (2 days), docs (3 days), release (2 days)

**Assumptions**:

- One full-time developer
- No major blockers
- Deferred complex features to v0.4.0 (spatial_autocorrelation, grid cells, circular stats)
- Speed/acceleration/angular velocity out of scope (computed outside package)

**Changes from original v17 plan**:

- **Timeline**: 17 weeks → 15 weeks with Phase 0 added (**12% reduction** while adding critical primitives)
- **Added Phase 0**: spikes_to_field, batch operations, reward fields (2 weeks)
- **Deferred to v0.4.0**: spatial_autocorrelation, grid cells, coherence, circular statistics
- **Deferred examples**: RatInABox integration
- **Out of scope**: Speed/acceleration/angular velocity (kinematic primitives computed externally)
- **Risk**: LOW overall

**Optimistic**: 14 weeks (if all implementations straightforward)
**Pessimistic**: 17 weeks (if Phase 0 spike interpolation needs iteration)

---

## Dependencies & Blockers

```
Phase 0: Spike/Field Primitives
├── 0.1 spikes_to_field (no blockers) ───┐
├── 0.2 batch operations (no blockers) ───┤
├── 0.3 reward fields (no blockers) ──────┤
└── 0.4 docs (needs 0.1-0.3) ─────────────┘

Phase 1: Differential Operators
├── 1.1 D matrix (no blockers) ───┐
├── 1.2 gradient (needs 1.1) ─────┤
├── 1.3 divergence (needs 1.1) ───┤
└── 1.4 docs (needs 1.2, 1.3) ────┘

Phase 2: Basic Signal Processing
├── 2.1 neighbor_reduce (no blockers)
└── 2.2 convolve (no blockers)

Phase 3: Core Metrics Module
├── 3.1 place_fields (no blockers)
├── 3.2 population (no blockers)
├── 3.3 boundary_cells (no blockers)
├── 3.4 trajectory_metrics (no blockers)
└── 3.5 docs (needs 3.1-3.4)

Phase 4: Behavioral Segmentation
├── 4.1 region_segmentation (no blockers)
├── 4.2 lap_detection (no blockers)
├── 4.3 trial_segmentation (no blockers)
├── 4.4 trajectory_similarity (no blockers)
└── 4.5 tests & docs (needs 4.1-4.4)

Phase 5: Polish & Release
├── 5.1 package_validation (no blockers)
├── 5.2 performance (no blockers)
├── 5.3 documentation (no blockers)
└── 5.4 release (needs all above)
```

**Critical path**: None - all phases are independent

**Parallelization opportunities**:

- Phase 1, 2, 3 have no dependencies and can run in parallel
- Phase 4 can start as soon as Phase 3 complete (for test data)
- Phase 5 requires all phases complete

---

## Testing Strategy

### Unit Tests

- Test each primitive independently
- Edge cases (empty graphs, single node, disconnected)
- NaN handling
- Input validation

### Integration Tests

- Composed operations (div∘grad, smooth∘gradient)
- Cross-layout validation (regular, hex, irregular)
- Performance regression tests

### Validation Tests

- Compare with NetworkX (Laplacian)
- Compare with PyGSP (gradient, divergence)
- Compare with opexebo/neurocode (place fields, spatial info, border score)
- Validate on synthetic data with known properties

### Benchmark Suite

```python
# benchmarks/bench_differential.py
def bench_differential_operator_construction(env):
    """Measure D matrix construction time"""

def bench_gradient_computation(env, field):
    """Measure gradient computation time"""

def bench_cached_vs_uncached(env):
    """Verify caching provides 50x speedup"""
```

---

## Documentation Requirements

### API Documentation

- NumPy-style docstrings for all functions
- Type hints for all parameters
- Examples in every docstring
- Cross-references to related functions

### User Guides

- `differential-operators.md` - Theory and usage
- `neuroscience-metrics.md` - Standard analyses
- `advanced-primitives.md` - RL/replay operations

### Example Notebooks

- `09_differential_operators.ipynb` - Gradient, divergence, flow fields
- `10_place_field_analysis.ipynb` - Complete place cell workflow
- `11_grid_cell_detection.ipynb` - Grid score computation

### Migration Guide

- Breaking changes (divergence → kl_divergence)
- New functionality overview
- Code migration examples

---

## Version Strategy

**Version 0.3.0** (Major feature release)

**Breaking changes**:

- `divergence()` → `kl_divergence()` (alias provided in 0.3.x)

**New features**:

- Differential operators (gradient, divergence)
- Spatial primitives (neighbor_reduce, spatial_autocorrelation, convolve)
- Metrics module (place_fields, grid_cells, population, remapping)

**Deprecations**:

- `divergence()` as KL divergence (use `kl_divergence()`)

**Future** (0.4.0):

- Remove deprecated aliases
- Additional primitives based on user feedback

---

## Open Questions

1. **Should `propagate` be included?**
   - Decision: Defer - seems redundant with distance_field
   - Action: Add if users request it

2. **Should metrics be separate package?**
   - Decision: No - include in core, but as optional import
   - Rationale: Lowers barrier, maintains cohesion

3. **Graph-based vs interpolation-based autocorrelation?**
   - Decision: Start with interpolation (faster, simpler)
   - Action: Add graph-based if users need it for irregular grids

4. **How to handle divergence rename?**
   - Decision: Alias in 0.3.x, remove in 0.4.0
   - Action: Announce early, provide migration guide

---

## Next Steps (Immediate)

### Week 1 (Immediate)

1. Review this plan with maintainers
2. Get feedback on API design
3. Decide on breaking change strategy (divergence rename)
4. Set up project board / issue tracking

### Week 2-3 (Phase 1 Start)

1. Implement differential operator (1.1)
2. Implement gradient (1.2)
3. Implement divergence with migration (1.3)
4. Write tests and documentation

### Communication

- Announce plan in GitHub discussion
- Request feedback on API design
- Warn users about breaking changes
- Ask for grid cell datasets for validation

---

---

## Future Implementations (v0.4.0+)

The following features are **deferred to future releases** but have detailed implementation plans ready.

---

### v0.4.0: Grid Cell Analysis

#### Spatial Autocorrelation

**Priority**: HIGH (enables grid cell detection)
**Complexity**: MEDIUM-HIGH (FFT for regular grids, graph-based for irregular)
**Estimated Effort**: 4 weeks

**Implementation**:

```python
def spatial_autocorrelation(
    field: NDArray[np.float64],
    env: Environment,
    *,
    method: Literal['auto', 'fft', 'graph'] = 'auto',
    overlap_amount: float = 0.8,
) -> NDArray[np.float64]:
    """
    Compute 2D spatial autocorrelation map.

    Parameters
    ----------
    field : array, shape (n_bins,)
        Spatial field (e.g., firing rate map)
    env : Environment
        Spatial environment
    method : {'auto', 'fft', 'graph'}, default='auto'
        - 'auto': Use FFT for regular grids, graph-based for irregular
        - 'fft': FFT-based (opexebo method, fast, regular grids only)
        - 'graph': Graph-based correlation (slower, works on any connectivity)
    overlap_amount : float, default=0.8
        Fraction of map to retain after edge cropping (reduces boundary noise)

    Returns
    -------
    autocorr_map : array, shape (height, width)
        2D autocorrelation map (for grid score computation)

    Notes
    -----
    **FFT method** (from opexebo, Moser lab):
    1. Replace NaNs with zeros
    2. Compute normalized cross-correlation via FFT
    3. Crop edges (default: keep central 80%)

    **Graph method** (neurospatial extension for irregular grids):
    1. Interpolate to regular grid
    2. Apply FFT method
    3. Or: compute pairwise correlations at each distance (slower but exact)

    References
    ----------
    .. [1] opexebo: https://github.com/simon-ball/opexebo
    .. [2] Sargolini et al. (2006). Science 312(5774).

    Examples
    --------
    >>> firing_rate_smooth = env.smooth(firing_rate, bandwidth=5.0)
    >>> autocorr_map = spatial_autocorrelation(firing_rate_smooth, env)
    >>> # Use for grid score computation
    >>> gs = grid_score(autocorr_map, env)

    See Also
    --------
    grid_score : Compute grid score from autocorrelation map
    opexebo.analysis.autocorrelation : Reference implementation
    """
    if method == 'auto':
        # Choose based on layout type
        if env.layout._layout_type_tag == 'RegularGridLayout':
            method = 'fft'
        else:
            method = 'fft'  # Interpolate irregular → regular grid

    if method == 'fft':
        # Adopt opexebo's FFT approach
        # Step 1: Reshape to 2D if regular grid (or interpolate if irregular)
        # Step 2: Replace NaNs with zeros
        # Step 3: normxcorr2_general() via FFT
        # Step 4: Crop edges
        pass  # Implementation details

    elif method == 'graph':
        # Graph-based approach for irregular grids
        # More principled but slower
        pass  # Implementation details
```

**Validation**:

- Test on synthetic hexagonal grid data
- Compare with opexebo outputs (should match within 1%)
- Validate rotation sensitivity

**Tests**:

```python
def test_autocorr_matches_opexebo():
    """Autocorrelation should match opexebo for regular grids"""
    env = Environment.from_samples(positions, bin_size=2.5)
    firing_rate = create_hexagonal_pattern()

    autocorr_ns = spatial_autocorrelation(firing_rate, env)

    try:
        import opexebo
        rate_map_2d = firing_rate.reshape(env.layout.grid_shape)
        autocorr_opexebo = opexebo.analysis.autocorrelation(rate_map_2d)
        np.testing.assert_allclose(autocorr_ns, autocorr_opexebo, rtol=0.01)
    except ImportError:
        pytest.skip("opexebo not installed")

def test_autocorr_hexagonal_field():
    """Hexagonal field should show peaks at 60° multiples"""
    # Create synthetic grid cell firing pattern
    # Compute autocorrelation
    # Verify peaks at correct angles

def test_autocorr_constant_field():
    """Constant field should have autocorr = 1 everywhere"""
    field = np.ones(env.n_bins)
    autocorr = spatial_autocorrelation(field, env)
    np.testing.assert_allclose(autocorr.max(), 1.0)
```

**Risk**: MEDIUM

- FFT approach validated (opexebo)
- Graph-based approach needs research (if irregular grids critical)

---

#### Grid Score

**File**: `src/neurospatial/metrics/grid_cells.py`

**Implementation**:

```python
def grid_score(
    firing_rate: NDArray,
    env: Environment,
    *,
    method: Literal['sargolini', 'langston'] = 'sargolini',
    num_gridness_radii: int = 3,
) -> float:
    """
    Compute grid score (gridness) using annular rings approach.

    This implementation matches opexebo's algorithm (Moser lab, Nobel Prize 2014):
    1. Compute spatial autocorrelation map
    2. Automatically detect central field radius
    3. For expanding radii, extract annular rings (donut shapes)
    4. Rotate autocorr at 30°, 60°, 90°, 120°, 150°
    5. Compute Pearson correlation between rings
    6. Grid score = min(corr[60°, 120°]) - max(corr[30°, 90°, 150°])
    7. Apply sliding window smoothing (3 radii default)
    8. Return maximum grid score

    Parameters
    ----------
    firing_rate : array, shape (n_bins,)
        Firing rate map (should be smoothed, 5 cm bandwidth recommended)
    env : Environment
        Spatial environment
    method : {'sargolini', 'langston'}, default='sargolini'
        Grid score formula variant
    num_gridness_radii : int, default=3
        Sliding window width for smoothing

    Returns
    -------
    grid_score : float
        Grid score. Range: [-2, 2]. Typical good grids: ~1.3

    References
    ----------
    .. [1] Sargolini et al. (2006). Science 312(5774).
    .. [2] opexebo.analysis.grid_score: Reference implementation

    See Also
    --------
    spatial_autocorrelation : Compute autocorrelation map
    opexebo.analysis.grid_score : Reference implementation
    """
    # Step 1: Compute autocorrelation
    autocorr_map = spatial_autocorrelation(firing_rate, env)

    # Step 2: Detect central field radius (automatic)
    # Step 3: For expanding radii, compute correlations with rotations
    # Step 4: Apply sliding window smoothing
    # Step 5: Return maximum
    pass  # Implementation follows opexebo exactly

def grid_spacing(autocorr_map: NDArray, env: Environment) -> float:
    """Estimate grid spacing from autocorrelation map."""
    pass

def grid_orientation(autocorr_map: NDArray, env: Environment) -> float:
    """Estimate grid orientation (degrees)."""
    pass

def coherence(
    firing_rate: NDArray,
    env: Environment,
    *,
    op: str = 'mean',
) -> float:
    """
    Spatial coherence (Muller & Kubie 1989).

    Correlation between firing rate and mean of neighbors.
    Uses neighbor_reduce primitive (generalizes opexebo's 3x3 convolution).

    References
    ----------
    .. [1] Muller & Kubie (1989). J Neurosci 9(12).
    .. [2] opexebo.analysis.rate_map_coherence: Reference implementation
    """
    neighbor_avg = neighbor_reduce(firing_rate, env, op=op)
    return np.corrcoef(firing_rate, neighbor_avg)[0, 1]
```

**Validation**:

- Test grid score matches opexebo within 1%
- Test coherence matches opexebo exactly
- Test on known grid cell data

**Authority**: opexebo (Moser lab, Nobel Prize 2014)

---

### v0.4.0+: Circular Statistics

**Priority**: MEDIUM (specialized for head direction cells)
**Complexity**: LOW (well-established algorithms)
**Estimated Effort**: 1 week

**File**: `src/neurospatial/metrics/circular.py`

**Functions**:

```python
def circular_mean(
    angles: NDArray,
    weights: NDArray | None = None,
) -> float:
    """
    Compute circular mean angle.

    Parameters
    ----------
    angles : array
        Angles in radians
    weights : array, optional
        Weights for each angle (e.g., occupancy)

    Returns
    -------
    mean_angle : float
        Circular mean in radians [-π, π]
    """
    if weights is None:
        weights = np.ones_like(angles)

    # Compute mean resultant vector
    C = np.sum(weights * np.cos(angles))
    S = np.sum(weights * np.sin(angles))

    return np.arctan2(S, C)

def circular_variance(
    angles: NDArray,
    weights: NDArray | None = None,
) -> float:
    """
    Compute circular variance (1 - r).

    Parameters
    ----------
    angles : array
        Angles in radians
    weights : array, optional
        Weights for each angle

    Returns
    -------
    variance : float
        Circular variance [0, 1]
    """
    r = resultant_vector_length(angles, weights)
    return 1 - r

def resultant_vector_length(
    angles: NDArray,
    weights: NDArray | None = None,
) -> float:
    """
    Compute mean resultant length (concentration measure).

    Parameters
    ----------
    angles : array
        Angles in radians
    weights : array, optional
        Weights for each angle

    Returns
    -------
    r : float
        Mean resultant length [0, 1]
        - r ≈ 0: Uniform distribution
        - r ≈ 1: Highly concentrated
    """
    if weights is None:
        weights = np.ones_like(angles)

    C = np.sum(weights * np.cos(angles))
    S = np.sum(weights * np.sin(angles))
    R = np.sqrt(C**2 + S**2)

    return R / np.sum(weights)

def fit_von_mises(
    angles: NDArray,
    weights: NDArray | None = None,
) -> tuple[float, float]:
    """
    Fit von Mises distribution to circular data.

    The von Mises distribution is the circular analog of the Gaussian.

    Parameters
    ----------
    angles : array
        Angles in radians
    weights : array, optional
        Weights for each angle

    Returns
    -------
    mu : float
        Mean direction (radians)
    kappa : float
        Concentration parameter
        - kappa ≈ 0: Uniform distribution
        - kappa >> 1: Highly concentrated

    References
    ----------
    .. [1] neurocode MapStats.m (circular statistics)
    .. [2] Fisher (1993). Statistical Analysis of Circular Data.

    Examples
    --------
    >>> # Head direction cell
    >>> angles = np.linspace(0, 2*np.pi, 100)
    >>> firing_rate = np.exp(kappa * np.cos(angles - preferred_direction))
    >>> mu, kappa = fit_von_mises(angles, weights=firing_rate)
    """
    # Circular mean
    mu = circular_mean(angles, weights)

    # Mean resultant length
    r = resultant_vector_length(angles, weights)

    # Estimate kappa from r (approximate MLE)
    if r < 0.53:
        kappa = 2 * r + r**3 + 5 * r**5 / 6
    elif r < 0.85:
        kappa = -0.4 + 1.39 * r + 0.43 / (1 - r)
    else:
        kappa = 1 / (2 * (1 - r))

    return mu, kappa

def rayleigh_test(
    angles: NDArray,
    weights: NDArray | None = None,
) -> tuple[float, float]:
    """
    Rayleigh test for uniformity of circular data.

    Tests null hypothesis: angles are uniformly distributed.

    Parameters
    ----------
    angles : array
        Angles in radians
    weights : array, optional
        Weights for each angle

    Returns
    -------
    z : float
        Rayleigh Z statistic
    p_value : float
        P-value for uniformity test

    Notes
    -----
    Small p-value (< 0.05) indicates non-uniform distribution
    (e.g., significant head direction tuning).
    """
    n = len(angles)
    r = resultant_vector_length(angles, weights)

    # Rayleigh Z statistic
    z = n * r**2

    # Approximate p-value (valid for n > 10)
    p_value = np.exp(-z) * (1 + (2*z - z**2) / (4*n) -
                             (24*z - 132*z**2 + 76*z**3 - 9*z**4) / (288*n**2))

    return z, p_value
```

**Validation**:

- Test against neurocode MapStats.m
- Test on synthetic von Mises distributions
- Test Rayleigh test p-values

**Authority**: neurocode (AyA Lab, Cornell), Fisher (1993)

---

### Integration Examples (Later)

#### RatInABox Validation

**Priority**: MEDIUM (nice-to-have for validation)
**Estimated Effort**: 1 week

**Example**: `examples/XX_ratinabox_integration.ipynb`

```python
# 1. Generate synthetic data with RatInABox
from ratinabox import Environment, Agent
from ratinabox.Neurons import PlaceCells, GridCells

Env = Environment(params={'scale': 1.0})
Ag = Agent(Env)
# ... simulate 5 minutes ...

PCs = PlaceCells(Ag, params={'n': 50})
GCs = GridCells(Ag, params={'n': 20, 'gridscale': 0.3})

# 2. Discretize with neurospatial
from neurospatial import Environment as NSEnv

env = NSEnv.from_samples(Ag.history['pos'], bin_size=0.05)

# 3. Analyze with neurospatial metrics
from neurospatial.metrics import grid_score, detect_place_fields

# Compute grid score
score = grid_score(GCs.history['firingrate'], env)

# 4. Validate against ground truth
true_spacing = 0.3  # Known from simulation!
estimated_spacing = estimate_grid_spacing(autocorr)
print(f"Error: {abs(estimated_spacing - true_spacing):.3f} m")
```

**Validation tests**:

- Grid score on synthetic GridCells with known spacing
- Place field detection on PlaceCells with known centers
- Spatial autocorrelation accuracy (correlation > 0.95)

---

## Summary

This implementation plan delivers **foundational spatial infrastructure** for neurospatial v0.3.0:

### What's Included (v0.3.0)

1. **Differential operators** - gradient, divergence, Laplacian on irregular graphs (RL/replay)
2. **Signal processing primitives** - neighbor_reduce, custom convolutions
3. **Place field metrics** - detection, information, sparsity, stability
4. **Population metrics** - coverage, density, overlap
5. **Boundary cell metrics** - border score (wall-preferring cells)
6. **Trajectory metrics** - turn angles, step lengths, home range, MSD (from ecology)
7. **Behavioral segmentation** - automatic detection of runs, laps, trials

### Deferred to v0.4.0+

**Grid Cell Analysis** (complex - FFT for regular, graph-based for irregular):

- `spatial_autocorrelation()` - 2D correlation for periodic pattern detection
- `grid_score()` - Grid cell detection (Nobel Prize 2014)
- `grid_spacing()`, `grid_orientation()`, `coherence()`

**Circular Statistics** (specialized for head direction cells):

- `circular_mean()`, `circular_variance()`, `fit_von_mises()`, `rayleigh_test()`

**Integration Examples**:

- RatInABox simulation examples
- pynapple IntervalSet integration (basic integration included in v0.3.0)

**Rationale**: Deferring spatial autocorrelation (most complex feature) reduces scope by 24% while delivering core functionality. Grid cell analysis requires it, so both deferred together.

---

**Timeline**: 13 weeks (~3 months) - **24% reduction from 17 weeks**
**Risk**: LOW (removed most complex feature)
**Impact**: HIGH - Delivers foundational infrastructure for spatial analysis

**Validation strategy**:

- ✅ Cross-validate against opexebo, neurocode, TSToolbox_Utils
- ✅ Test on synthetic trajectories
- ❌ RatInABox validation deferred to v0.4.0

**Ecosystem context** (24 packages analyzed):

- **Neuroscience** (16): pynapple, SpikeInterface, opexebo, neurocode, movement
- **Ecology/Trajectory** (8): Traja, yupi, PyRAT, adehabitatHR, ctmm, moveHMM
- **Authority**: opexebo (Moser lab), neurocode (AyA Lab), ecology literature

**Module structure** (v0.3.0):

```
src/neurospatial/
    differential.py        # Phase 1: gradient, divergence, differential_operator
    primitives.py          # Phase 2: neighbor_reduce, convolve
    metrics/               # Phase 3: core metrics
        place_fields.py    #   detect_place_fields, skaggs_information, sparsity
        population.py      #   population_coverage, field_density_map
        boundary_cells.py  #   border_score (Solstad et al. 2008)
        trajectory.py      #   turn_angles, step_lengths, home_range, MSD
    segmentation/          # Phase 4: behavioral epoch segmentation
        regions.py         #   detect_runs_between_regions, detect_region_crossings
        laps.py            #   detect_laps (3 methods)
        trials.py          #   segment_trials (task-specific)
        similarity.py      #   trajectory_similarity, detect_goal_directed_runs
```

**Function rename** (no users affected):

- `divergence()` → `kl_divergence()` (direct rename, no migration needed)

**Next steps**:

1. Review with maintainers
2. Get approval for scope (v0.3.0 focused)
3. Proceed with Phase 1 implementation

### Deferred to Future Releases

**v0.3.1 - Batch Operations (Performance Optimization):**

- `env.smooth_batch()` - Batch smoothing for multiple fields (10-50× speedup)
- `normalize_fields()` - Batch normalization utility
- **Rationale**: Ship core primitives first, add performance optimizations when users report need. Follow "make it work, make it right, make it fast" principle (Raymond Hettinger).

**v0.4.0 - Grid Cell Analysis:**

- `spatial_autocorrelation()` - 2D correlation for periodic pattern detection
- `grid_score()` - Grid cell detection (Nobel Prize 2014)
- `grid_spacing()`, `grid_orientation()`
- `coherence()` - Spatial smoothness metric

**v0.4.0+ - Circular Statistics:**

- `circular_mean()`, `circular_variance()`, `resultant_vector_length()`
- `fit_von_mises()` - Head direction cell analysis
- `rayleigh_test()` - Uniformity testing

**Later - Integration Examples:**

- RatInABox simulation examples
- pynapple IntervalSet integration examples

**Rationale**: Spatial autocorrelation is complex (FFT for regular grids, graph-based for irregular), grid cells require it, circular stats are specialized. Batch operations are performance optimizations best added after validating core API with users. Deferring these reduces scope by ~35% while delivering core functionality.

---
