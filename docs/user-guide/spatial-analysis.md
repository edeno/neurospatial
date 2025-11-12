# Spatial Analysis Operations

This guide provides a comprehensive reference for spatial analysis operations in neurospatial. These operations enable you to analyze trajectories, compute movement patterns, manipulate spatial fields, and perform common spatial calculations.

!!! info "New in v0.1.0"
    All operations documented on this page were introduced in version 0.1.0 as part of the core spatial analysis API.

## Quick Reference

| Operation | Category | Purpose | Returns |
|-----------|----------|---------|---------|
| [`occupancy()`](#occupancy) | Trajectory | Time spent in each bin | `NDArray[np.float64]` |
| [`bin_sequence()`](#bin-sequence) | Trajectory | Sequence of visited bins | `NDArray[np.int64]` |
| [`transitions()`](#transitions) | Movement | Bin-to-bin transition counts | `scipy.sparse.csr_matrix` |
| [`components()`](#components) | Movement | Connected subgraphs | `List[Set[int]]` |
| [`reachable_from()`](#reachable-from) | Movement | Accessible bins from source | `Set[int]` |
| [`smooth()`](#smooth) | Fields | Gaussian smoothing of values | `NDArray[np.float64]` |
| [`interpolate()`](#interpolate) | Fields | Fill missing values | `NDArray[np.float64]` |
| [`rebin()`](#rebin) | Fields | Change spatial resolution | `Tuple[Environment, NDArray]` |
| [`subset()`](#subset) | Fields | Extract sub-environment | `Environment` |
| [`distance_to()`](#distance-to) | Utilities | Geodesic distances to point | `NDArray[np.float64]` |
| [`rings()`](#rings) | Utilities | Bins by graph hop distance (BFS layers) | `list[NDArray[np.int32]]` |
| [`region_membership()`](#region-membership) | Utilities | Which region contains each bin | `NDArray[np.int64]` |
| [`copy()`](#copy) | Utilities | Deep copy environment | `Environment` |

## Trajectory Analysis

### Occupancy

Compute the time spent in each spatial bin based on position tracking data.

```python
def occupancy(
    self,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = 0.5,
    kernel_bandwidth: float | None = None,
    time_allocation: Literal["start", "linear"] = "start",
) -> NDArray[np.float64]:
    """
    Compute occupancy (time spent in each bin).

    Parameters
    ----------
    times : array, shape (n_samples,)
        Timestamps in seconds. Must be monotonically increasing.
    positions : array, shape (n_samples, n_dims)
        Position coordinates matching environment dimensions.
    speed : array, shape (n_samples,), optional
        Instantaneous speed at each sample. If provided with min_speed,
        samples below threshold are excluded from occupancy.
    min_speed : float, optional
        Minimum speed threshold (physical units/second). Requires speed
        parameter. Samples with speed < min_speed are excluded.
    max_gap : float, optional
        Maximum time gap in seconds. Intervals with Δt > max_gap are not
        counted toward occupancy. Default: 0.5 seconds. Set to None to
        count all intervals.
    kernel_bandwidth : float, optional
        If provided, apply diffusion kernel smoothing with this bandwidth
        (in physical units). Preserves total occupancy time.
    time_allocation : {'start', 'linear'}, default='start'
        Method for allocating time intervals:
        - 'start': Assign entire Δt to starting bin (fast, all layouts)
        - 'linear': Split Δt across bins traversed (accurate, grids only)

    Returns
    -------
    occupancy : array, shape (n_bins,)
        Time in seconds spent in each bin.
    """
```

**Example:**

```python
import numpy as np
from neurospatial import Environment

# Load your tracking data
times = np.arange(0, 600, 0.033)  # 30 Hz for 10 minutes
positions = load_position_data()  # shape: (n_samples, 2)

# Create environment
env = Environment.from_samples(positions, bin_size=2.5)

# Basic occupancy
occupancy_time = env.occupancy(times, positions)

print(f"Total time: {occupancy_time.sum():.1f} seconds")
print(f"Bins visited: {np.sum(occupancy_time > 0)} / {env.n_bins}")
print(f"Mean time per visited bin: {occupancy_time[occupancy_time > 0].mean():.2f} s")

# Filter bins with sufficient sampling
well_sampled = occupancy_time >= 1.0  # At least 1 second
print(f"Well-sampled bins: {well_sampled.sum()}")
```

**Advanced Usage:**

```python
# Speed-filtered occupancy (exclude immobility periods)
speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)
speed = np.concatenate([[0], speed])  # Pad to match length
occupancy_moving = env.occupancy(times, positions, speed=speed, min_speed=2.0)

# Smoothed occupancy (preserves total time)
occupancy_smooth = env.occupancy(times, positions, kernel_bandwidth=5.0)
assert np.isclose(occupancy_smooth.sum(), occupancy_time.sum())

# Linear allocation (more accurate for grids)
occupancy_linear = env.occupancy(times, positions, time_allocation='linear')
```

**Use Cases:**

- Normalizing firing rates for place field analysis
- Identifying under-sampled regions
- Computing exploration metrics
- Weighting spatial statistics by sampling
- Speed-filtering to exclude resting periods

**Common Patterns:**

```python
# Occupancy-normalized firing rate
firing_rate = spike_counts / env.occupancy(times, positions)

# Exclude low-occupancy bins
min_occupancy = 0.5  # seconds
valid_mask = env.occupancy(times, positions) >= min_occupancy
firing_rate[~valid_mask] = np.nan
```

---

### Bin Sequence

Extract the temporal sequence of bins visited during a trajectory.

```python
def bin_sequence(
    self,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    dedup: bool = True,
    return_runs: bool = False,
    outside_value: int | None = -1,
) -> (
    NDArray[np.int32]
    | tuple[NDArray[np.int32], NDArray[np.int64], NDArray[np.int64]]
):
    """
    Get sequence of bins visited over time.

    Parameters
    ----------
    times : array, shape (n_samples,)
        Timestamps for each position sample.
    positions : array, shape (n_samples, n_dims)
        Position coordinates over time.
    dedup : bool, optional
        If True, remove consecutive duplicate bins (compress runs).
        Default is True.
    return_runs : bool, optional
        If True, return (bins, start_indices, end_indices) tuple
        for run analysis. Default is False.
    outside_value : int or None, optional
        Value to use for positions outside active bins.
        If None, raises error on out-of-bounds positions.
        Default is -1.

    Returns
    -------
    sequence : array, shape (n_samples,) or (n_runs,)
        Bin indices in temporal order. If dedup=True, consecutive
        duplicates are removed.
    start_indices : array, shape (n_runs,), optional
        Start indices of each run (if return_runs=True).
    end_indices : array, shape (n_runs,), optional
        End indices of each run (if return_runs=True).
    """
```

**Example:**

```python
# Get bin sequence (deduplicated by default)
sequence = env.bin_sequence(times, positions)

# Get full sequence with runs
bins, starts, ends = env.bin_sequence(times, positions, return_runs=True)

# Analyze run durations
run_durations = ends - starts
print(f"Total runs: {len(bins)}")
print(f"Mean run length: {run_durations.mean():.1f} samples")

# Find longest run
longest_idx = np.argmax(run_durations)
print(f"Longest run: {run_durations[longest_idx]} samples in bin {bins[longest_idx]}")

# Get run times (requires timestamps)
run_times = times[ends] - times[starts]
print(f"Mean dwell time: {run_times.mean():.3f} seconds")
```

**Advanced Usage:**

```python
# Keep duplicates for fixed-rate analysis
full_sequence = env.bin_sequence(times, positions, dedup=False)

# Handle out-of-bounds strictly
try:
    sequence = env.bin_sequence(times, positions, outside_value=None)
except ValueError as e:
    print(f"Out of bounds detected: {e}")

# Analyze time outside environment
sequence_with_invalid = env.bin_sequence(times, positions, outside_value=-1)
time_outside = np.sum(sequence_with_invalid == -1) / len(times) * (times[-1] - times[0])
print(f"Time outside environment: {time_outside:.1f} seconds")
```

**Use Cases:**

- Run detection and dwell time analysis
- State change detection
- Temporal ordering of spatial visits
- Trajectory segmentation
- Detecting out-of-bounds movements

**Common Patterns:**

```python
# Detect spatial teleportation (large jumps between consecutive bins)
bins, starts, _ = env.bin_sequence(times, positions, return_runs=True)
spatial_jumps = []
for i in range(len(bins) - 1):
    dist = env.distance_between_bins(bins[i], bins[i+1])
    if dist > 20.0:  # cm
        jump_time = times[starts[i+1]]
        spatial_jumps.append((jump_time, dist))

# Compute dwell time per bin
bins, starts, ends = env.bin_sequence(times, positions, return_runs=True)
dwell_times = np.zeros(env.n_bins)
for bin_id, start, end in zip(bins, starts, ends):
    if bin_id >= 0:  # Skip out-of-bounds
        dwell_times[bin_id] += times[end] - times[start]
```

---

## Movement Patterns

### Transitions

Compute the transition matrix between bins, either from empirical data or theoretical models.

!!! tip "Why This Matters"
    Transition matrices reveal behavioral patterns invisible in raw trajectories. They help answer:

    - **Behavioral stereotypy**: "Does the animal follow stereotyped paths or explore randomly?"
    - **Path preferences**: "Are certain routes preferred over others with equal distance?"
    - **Decision points**: "Where does behavior become predictable vs exploratory?"
    - **Neural prediction**: "Do neural patterns predict upcoming movements?"

    Essential for Markov models, behavioral state detection, decision-making analysis, and comparing empirical behavior to random walk null models.

```python
def transitions(
    self,
    bins: NDArray[np.int32] | None = None,
    *,
    times: NDArray[np.float64] | None = None,
    positions: NDArray[np.float64] | None = None,
    # Empirical parameters
    lag: int = 1,
    allow_teleports: bool = False,
    # Model-based parameters
    method: Literal["diffusion", "random_walk"] | None = None,
    bandwidth: float | None = None,
    # Common
    normalize: bool = True,
) -> scipy.sparse.csr_matrix:
    """
    Compute bin-to-bin transition matrix.

    Two modes of operation:

    1. **Empirical Mode** (provide bins OR times+positions):
       Count actual transitions from trajectory data.

    2. **Model-Based Mode** (provide method):
       Generate theoretical transition matrix from diffusion or
       random walk models based on environment connectivity.

    Parameters
    ----------
    bins : array, shape (n_samples,), optional
        Pre-computed bin sequence. If None, computed from times+positions.
    times : array, shape (n_samples,), optional
        Timestamps (required if bins is None).
    positions : array, shape (n_samples, n_dims), optional
        Position trajectory (required if bins is None).
    lag : int, optional
        Number of steps between transitions (empirical mode).
        Default is 1 (consecutive samples).
    allow_teleports : bool, optional
        If True, allow transitions between non-adjacent bins (empirical mode).
        If False, only count transitions along edges. Default is False.
    method : {"diffusion", "random_walk"}, optional
        Theoretical model to use (model-based mode):
        - "diffusion": Continuous diffusion on graph
        - "random_walk": Discrete random walk with uniform neighbor transitions
    bandwidth : float, optional
        Spatial bandwidth for diffusion model (model-based mode).
    normalize : bool, optional
        If True, normalize rows to sum to 1 (probabilities).
        If False, return raw counts. Default is True.

    Returns
    -------
    transitions : sparse matrix, shape (n_bins, n_bins)
        Element [i, j] is count/probability of transitions from bin i to bin j.
        Uses CSR sparse format for efficiency.
    """
```

**Empirical Mode Examples:**

```python
# From pre-computed bin sequence
bins = env.bin_sequence(times, positions)
T = env.transitions(bins=bins, normalize=True)

# From trajectory data directly
T = env.transitions(times=times, positions=positions, normalize=True)

# Analyze transition patterns
print(f"Total transitions: {T.nnz}")
print(f"Sparsity: {100 * (1 - T.nnz / (env.n_bins ** 2)):.1f}%")

# Find most common transitions
T_dense = T.toarray()
max_prob = np.max(T_dense)
max_idx = np.unravel_index(np.argmax(T_dense), T_dense.shape)
print(f"Most common: bin {max_idx[0]} → bin {max_idx[1]} ({max_prob:.3f})")

# Multi-step transitions (lag > 1)
T_lag2 = env.transitions(bins=bins, lag=2, normalize=True)

# Allow non-adjacent transitions (teleportation)
T_all = env.transitions(bins=bins, allow_teleports=True, normalize=True)
```

**Model-Based Mode Examples:**

```python
# Random walk model (uniform transitions to neighbors)
T_rw = env.transitions(method="random_walk", normalize=True)

# Diffusion model with spatial bandwidth
T_diff = env.transitions(method="diffusion", bandwidth=10.0, normalize=True)

# Compare empirical vs theoretical
T_empirical = env.transitions(bins=bins, normalize=True)
T_model = env.transitions(method="random_walk", normalize=True)

diff = (T_empirical - T_model).toarray()
deviation = np.linalg.norm(diff, 'fro')
print(f"Frobenius norm difference: {deviation:.3f}")
```

**Visualization:**

```python
import matplotlib.pyplot as plt

# Visualize transition matrix
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(T.toarray(), cmap='hot', interpolation='nearest')
ax.set_xlabel('To Bin')
ax.set_ylabel('From Bin')
ax.set_title('Transition Probability Matrix')
plt.colorbar(im, label='Probability')
plt.show()
```

**Use Cases:**

- Movement pattern analysis and behavioral stereotypy
- Behavioral state transitions
- Markov model estimation for trajectory prediction
- Detecting preferred paths or place preferences
- Comparing empirical behavior to theoretical null models

**Common Patterns:**

```python
# Compute entropy of transitions (movement randomness)
T_norm = env.transitions(bins=bins, normalize=True)
T_dense = T_norm.toarray()

entropy = np.zeros(env.n_bins)
for i in range(env.n_bins):
    row = T_dense[i]
    # Only non-zero probabilities
    p = row[row > 0]
    if len(p) > 0:
        entropy[i] = -np.sum(p * np.log2(p))

print(f"Mean transition entropy: {entropy.mean():.2f} bits")

# Detect directional bias
T = env.transitions(bins=bins, normalize=False)
directional_bias = T - T.T  # Asymmetry
max_bias_idx = np.unravel_index(np.argmax(np.abs(directional_bias.toarray())),
                                 directional_bias.shape)
print(f"Strongest directional bias: bin {max_bias_idx[0]} ↔ bin {max_bias_idx[1]}")
```

---

### Components

Find connected components (subgraphs) within the environment's connectivity graph.

```python
def components(self) -> List[Set[int]]:
    """
    Find connected components in environment.

    Returns
    -------
    components : list of sets
        Each set contains bin indices in one connected component.
        Sorted by size (largest first).
    """
```

**Example:**

```python
# Find components
comps = env.components()

print(f"Number of components: {len(comps)}")
for i, comp in enumerate(comps):
    print(f"  Component {i}: {len(comp)} bins")

# Check if environment is fully connected
if len(comps) == 1:
    print("Environment is fully connected")
else:
    print(f"Environment has {len(comps)} disconnected regions")

# Visualize components
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10))
colors = plt.cm.tab10(np.arange(len(comps)))

for i, comp in enumerate(comps):
    comp_bins = list(comp)
    ax.scatter(
        env.bin_centers[comp_bins, 0],
        env.bin_centers[comp_bins, 1],
        c=[colors[i]],
        s=100,
        label=f'Component {i} ({len(comp)} bins)'
    )

ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_title('Connected Components')
ax.legend()
ax.set_aspect('equal')
plt.show()
```

**Use Cases:**

- Detecting disconnected maze regions
- Identifying isolated bins (component size = 1)
- Validating environment connectivity
- Multi-compartment analysis

---

### Reachable From

Find all bins reachable from a source bin via the connectivity graph.

```python
def reachable_from(
    self,
    source: int,
    *,
    max_distance: Optional[float] = None,
) -> Set[int]:
    """
    Find bins reachable from source.

    Parameters
    ----------
    source : int
        Starting bin index.
    max_distance : float, optional
        Maximum geodesic distance to traverse.
        If None, returns all reachable bins. Default is None.

    Returns
    -------
    reachable : set of int
        Bin indices reachable from source.
    """
```

**Example:**

```python
# Find all bins reachable from center
center_bin = env.bin_at(np.array([[50.0, 50.0]]))[0]
reachable = env.reachable_from(center_bin)

print(f"Bins reachable from center: {len(reachable)} / {env.n_bins}")

# Find bins within 20 cm geodesic distance
nearby = env.reachable_from(center_bin, max_distance=20.0)
print(f"Bins within 20 cm: {len(nearby)}")

# Detect unreachable regions
unreachable = set(range(env.n_bins)) - reachable
if unreachable:
    print(f"Warning: {len(unreachable)} bins unreachable from center!")
```

**Use Cases:**

- Accessibility analysis
- Detecting barriers or walls
- Goal reachability queries
- Spatial subregion extraction

---

## Field Operations

### Smooth

Apply Gaussian smoothing to spatial fields (e.g., firing rate maps).

!!! tip "Why This Matters"
    Smoothing is critical for accurate neuroscience analysis:

    - **Place field visualization**: Reduces noise from limited sampling without losing spatial structure
    - **Bayesian decoding**: Smooth spatial priors improve position inference accuracy
    - **Field detection**: Enhances signal-to-noise for automated place field algorithms
    - **Scientific conclusions**: Over-smoothing merges distinct fields and inflates field size; under-smoothing creates spurious peaks

    **Rule of thumb**: Use bandwidth ≈ expected place field size / 2. Test multiple bandwidths to ensure conclusions are robust.

```python
def smooth(
    self,
    field: NDArray[np.float64],
    bandwidth: float,
    *,
    mode: Literal["transition", "density"] = "density",
) -> NDArray[np.float64]:
    """
    Apply Gaussian smoothing to field values.

    Parameters
    ----------
    field : array, shape (n_bins,)
        Field values to smooth.
    bandwidth : float
        Spatial bandwidth of Gaussian kernel in physical units
        (matching environment coordinates, e.g., cm, meters).
        REQUIRED parameter - no default.
    mode : {"transition", "density"}, optional
        Smoothing mode:
        - "density": Standard Gaussian smoothing for density fields
        - "transition": Graph-based smoothing following connectivity
        Default is "density".

    Returns
    -------
    smoothed : array, shape (n_bins,)
        Smoothed field values.
    """
```

!!! warning "Bandwidth Units"
    The `bandwidth` parameter is in **physical units** (cm, meters, etc.),
    matching your environment's coordinate system. For bandwidth in bins,
    multiply by bin_size: `bandwidth = bandwidth_bins * bin_size`.

    Example: If bin_size=2.5 cm and you want 2-bin smoothing:
    `bandwidth = 2 * 2.5 = 5.0  # cm`

**Example:**

```python
# Compute raw firing rate
spike_counts = compute_spike_counts(env, spike_times, positions)
occupancy = env.occupancy(times, positions)
firing_rate_raw = np.divide(
    spike_counts,
    occupancy,
    where=occupancy > 0.1,
    out=np.full(env.n_bins, np.nan)
)

# Smooth the firing rate map (bandwidth in physical units)
firing_rate_smooth = env.smooth(firing_rate_raw, bandwidth=5.0)  # 5 cm

# Compare raw vs smoothed
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, data, title in zip(
    axes,
    [firing_rate_raw, firing_rate_smooth],
    ['Raw Firing Rate', 'Smoothed (5 cm bandwidth)']
):
    scatter = ax.scatter(
        env.bin_centers[:, 0],
        env.bin_centers[:, 1],
        c=data,
        s=100,
        cmap='hot',
        vmin=0
    )
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Rate (Hz)')

plt.show()
```

**Use Cases:**

- Noise reduction in firing rate maps
- Place field visualization
- Spatial field preprocessing
- Feature extraction for decoding

**Best Practices:**

```python
# Specify bandwidth in physical units (not bins!)
bandwidth_cm = 5.0  # 5 cm smoothing
firing_rate_smooth = env.smooth(firing_rate_raw, bandwidth=bandwidth_cm)

# Or compute from bin sizes
bin_size_cm = 2.5
bandwidth_bins = 2  # Smooth over 2 bins
bandwidth_cm = bandwidth_bins * bin_size_cm  # 5.0 cm
firing_rate_smooth = env.smooth(firing_rate_raw, bandwidth=bandwidth_cm)

# Graph-based smoothing (follows connectivity, respects barriers)
firing_rate_smooth_graph = env.smooth(firing_rate_raw, bandwidth=5.0, mode="transition")

# NaN values are handled automatically - they don't contribute to smoothing
# and smoothed values at NaN locations remain NaN
assert np.isnan(firing_rate_smooth)[np.isnan(firing_rate_raw)].all()
```

---

### Interpolate

Evaluate bin field values at arbitrary continuous spatial points.

```python
def interpolate(
    self,
    field: NDArray[np.float64],
    points: NDArray[np.float64],
    *,
    mode: Literal["nearest", "linear"] = "nearest",
) -> NDArray[np.float64]:
    """
    Evaluate field at query points in continuous space.

    Parameters
    ----------
    field : array, shape (n_bins,)
        Field values defined at bin centers.
    points : array, shape (n_points, n_dims)
        Query points in continuous spatial coordinates.
    mode : {"nearest", "linear"}, optional
        Interpolation mode:
        - "nearest": Use value from nearest bin center
        - "linear": Linear interpolation between bin centers
        Default is "nearest".

    Returns
    -------
    values : array, shape (n_points,)
        Field values evaluated at query points.
    """
```

**Example:**

```python
# Compute firing rate field on bins
spike_counts = compute_spike_counts(env, spike_times, bin_positions)
occupancy = env.occupancy(times, positions)
firing_rate_field = spike_counts / occupancy  # shape: (n_bins,)

# Evaluate field at specific locations
query_points = np.array([
    [25.0, 30.0],  # Location A
    [40.0, 15.0],  # Location B
    [10.0, 45.0],  # Location C
])

# Get firing rates at these specific locations
rates_at_points = env.interpolate(firing_rate_field, query_points, mode='nearest')
print(f"Rates at query points: {rates_at_points}")

# Linear interpolation for smoother estimates
rates_smooth = env.interpolate(firing_rate_field, query_points, mode='linear')

# Evaluate along a trajectory
trajectory_points = generate_trajectory(start, end, n_samples=100)
rates_along_path = env.interpolate(firing_rate_field, trajectory_points, mode='linear')
```

**Use Cases:**

- Evaluating fields at arbitrary spatial locations
- Computing values along continuous trajectories
- Extracting field values at specific points of interest
- Comparing field values at landmark positions
- Generating continuous field representations from discrete bins

**Common Patterns:**

```python
# Decode position by finding location with maximum firing rate
query_grid = create_dense_grid(env.bounds, resolution=0.5)  # 0.5 cm resolution
decoded_rates = env.interpolate(firing_rate_field, query_grid, mode='linear')
max_idx = np.argmax(decoded_rates)
decoded_position = query_grid[max_idx]

# Evaluate field along a 1D slice
x_slice = np.linspace(0, 100, 200)
y_fixed = 50.0
slice_points = np.column_stack([x_slice, np.full_like(x_slice, y_fixed)])
values_along_slice = env.interpolate(field, slice_points, mode='linear')

# Compare field values at region centers
region_centers = np.array([env.regions.region_center(name) for name in env.regions.keys()])
values_at_regions = env.interpolate(field, region_centers, mode='nearest')
```

---

### Rebin

Coarsen the environment grid by integer subsampling (geometry-only operation).

```python
def rebin(
    self,
    factor: int | tuple[int, ...],
) -> Environment:
    """
    Create coarser environment by integer subsampling.

    This is a geometry-only operation that reduces spatial resolution
    by subsampling the grid at integer factors. It does NOT remap or
    aggregate field values - use separate field processing for that.

    Parameters
    ----------
    factor : int or tuple of int
        Integer subsampling factor(s):
        - int: Apply same factor to all dimensions
        - tuple: Per-dimension factors (length must match n_dims)

        Examples:
        - factor=2: Keep every 2nd bin (halve resolution)
        - factor=3: Keep every 3rd bin (third resolution)
        - factor=(2, 3): Different factors per dimension

    Returns
    -------
    coarse_env : Environment
        New environment with reduced spatial resolution.
        Number of bins reduced by product of factors.
    """
```

!!! warning "Geometry Only"
    `rebin()` creates a coarser grid but does NOT remap field values.
    To aggregate field values to the new grid, you must handle that
    separately using your own aggregation logic.

**Example:**

```python
# Create environment with 2.0 cm bins
env_fine = Environment.from_samples(positions, bin_size=2.0)
print(f"Fine environment: {env_fine.n_bins} bins, bin_size={2.0} cm")

# Coarsen by factor of 2 (effective bin_size = 4.0 cm)
env_coarse = env_fine.rebin(factor=2)
print(f"Coarse environment: {env_coarse.n_bins} bins, bin_size≈{4.0} cm")
print(f"Reduction: {env_fine.n_bins / env_coarse.n_bins:.1f}x")

# Different factors per dimension (2D)
env_anisotropic = env_fine.rebin(factor=(2, 3))
print(f"Anisotropic coarsening: {env_anisotropic.n_bins} bins")

# Coarsen by factor of 4 for very coarse analysis
env_very_coarse = env_fine.rebin(factor=4)
```

**Field Value Aggregation (Manual):**

Since `rebin()` doesn't aggregate field values, here's how to do it manually:

```python
# Get bin correspondence between fine and coarse
fine_bins = np.arange(env_fine.n_bins)
fine_centers = env_fine.bin_centers
coarse_bins = env_coarse.bin_at(fine_centers)

# Aggregate field values to coarse grid
firing_rate_fine = compute_firing_rate(env_fine, spikes, positions)
firing_rate_coarse = np.zeros(env_coarse.n_bins)

for coarse_bin in range(env_coarse.n_bins):
    # Find all fine bins that map to this coarse bin
    mask = (coarse_bins == coarse_bin)
    if mask.any():
        # Aggregate (mean, sum, max, etc.)
        firing_rate_coarse[coarse_bin] = np.mean(firing_rate_fine[mask])

# Or use pandas for convenience
import pandas as pd
df = pd.DataFrame({
    'coarse_bin': coarse_bins,
    'firing_rate': firing_rate_fine
})
firing_rate_coarse_alt = df.groupby('coarse_bin')['firing_rate'].mean().values
```

**Use Cases:**

- Multi-scale spatial analysis
- Reducing computational cost for large environments
- Creating visualization at different resolutions
- Testing analysis robustness across spatial scales

---

### Subset

Extract a sub-environment containing only specified bins.

```python
def subset(
    self,
    bin_indices: Union[List[int], NDArray[np.int64], Set[int]],
    *,
    name: Optional[str] = None,
) -> Environment:
    """
    Create sub-environment from bin subset.

    Parameters
    ----------
    bin_indices : list, array, or set of int
        Bins to include in subset.
    name : str, optional
        Name for new environment.
        If None, uses f"{original_name}_subset".

    Returns
    -------
    subset_env : Environment
        New environment containing only specified bins.
        Bin indices are remapped (starting from 0).
    """
```

**Example:**

```python
# Extract region of interest
# Get bins in north arm
north_center = np.array([50.0, 75.0])
distances = np.linalg.norm(env.bin_centers - north_center, axis=1)
north_bins = np.where(distances < 15.0)[0]

# Create subset environment
env_north = env.subset(north_bins, name='NorthArm')

print(f"Original: {env.n_bins} bins")
print(f"North arm: {env_north.n_bins} bins")

# Analyze just this region
occupancy_north = env_north.occupancy(timestamps, positions)
```

**Use Cases:**

- Region-specific analysis
- Excluding walls or barriers
- Multi-compartment experiments
- Focused decoding

---

## Spatial Utilities

### Distance To

Compute geodesic or Euclidean distance from each bin to target bins or regions.

!!! tip "Why This Matters"
    Distance fields are fundamental for spatial navigation analysis:

    - **Goal-directed behavior**: Quantify how activity relates to distance from rewards/goals
    - **Value functions**: Build spatial value maps for reinforcement learning models
    - **Path planning**: Understand navigation strategies and route selection
    - **Multi-goal tasks**: Analyze behavior when multiple targets compete for attention

    **Geodesic vs Euclidean**: Geodesic distance (also called "path distance" or "through-environment distance") follows the connectivity graph and respects barriers, while Euclidean measures straight-line distance ignoring structure.

```python
def distance_to(
    self,
    targets: Sequence[int] | str,
    *,
    metric: Literal["euclidean", "geodesic"] = "geodesic",
) -> NDArray[np.float64]:
    """
    Compute distances to target bins or region.

    Parameters
    ----------
    targets : sequence of int, or str
        Target specification:
        - Sequence of bin indices: Distance to closest target bin
        - String: Region name - distance to closest bin in region
    metric : {"euclidean", "geodesic"}, optional
        Distance metric:
        - "geodesic": Follows connectivity graph (path distance, respects barriers)
        - "euclidean": Straight-line distance
        Default is "geodesic".

    Returns
    -------
    distances : array, shape (n_bins,)
        Distance from each bin to nearest target.
        For geodesic: Unreachable bins have np.inf distance.
    """
```

**Example:**

```python
# Distance to specific bin indices
goal_bins = [42, 87, 103]
dist_to_goals = env.distance_to(goal_bins, metric='geodesic')

# Distance to named region (requires regions defined)
env.regions.add('goal_zone', polygon=goal_polygon)
dist_to_goal_region = env.distance_to('goal_zone', metric='geodesic')

# Euclidean vs geodesic comparison
dist_euclidean = env.distance_to([goal_bin], metric='euclidean')
dist_geodesic = env.distance_to([goal_bin], metric='geodesic')

ratio = dist_geodesic / dist_euclidean
print(f"Mean geodesic/euclidean ratio: {ratio[np.isfinite(ratio)].mean():.2f}")

# Multiple target bins (distance to nearest)
reward_bins = [10, 50, 90]
dist_to_nearest_reward = env.distance_to(reward_bins, metric='geodesic')

# Visualize distance field
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, dist, title in zip(
    axes,
    [dist_euclidean, dist_geodesic],
    ['Euclidean Distance', 'Geodesic Distance']
):
    scatter = ax.scatter(
        env.bin_centers[:, 0],
        env.bin_centers[:, 1],
        c=dist,
        s=100,
        cmap='viridis'
    )
    goal_center = env.bin_centers[goal_bin]
    ax.scatter(*goal_center, c='red', s=300, marker='*',
               edgecolors='white', linewidth=2, label='Goal')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Distance (cm)')

plt.show()
```

**Use Cases:**

- Goal-directed behavior analysis
- Distance-to-reward computations for reinforcement learning
- Spatial gradients and value functions
- Path planning and navigation analysis
- Multi-goal tasks (distance to nearest goal)

---

### Rings

Get bins organized by graph distance (hop count) from a center bin using BFS layers.

!!! tip "Why This Matters"
    BFS layers reveal how spatial information propagates through connectivity:

    - **Local neighborhood analysis**: Study how neural activity decays with graph distance from landmarks
    - **Connectivity structure**: Understand environment topology independent of Euclidean geometry
    - **Reachability testing**: Verify which bins are accessible within N steps
    - **Spatial spread**: Analyze how behavioral or neural patterns diffuse through the environment

    Unlike `distance_to()` which uses physical distance, `rings()` uses hop count - two adjacent bins = 1 hop regardless of physical separation.

```python
def rings(
    self,
    center_bin: int,
    *,
    hops: int,
) -> list[NDArray[np.int32]]:
    """
    Get bins organized by graph hop distance from center.

    Uses breadth-first search (BFS) to partition bins into layers
    based on graph distance (hop count) from the center bin.

    Parameters
    ----------
    center_bin : int
        Starting bin index.
    hops : int
        Maximum number of hops (BFS layers) to compute.

    Returns
    -------
    layers : list of arrays
        Length hops + 1 (includes center at layer 0).
        layers[k] contains bin indices at exactly k hops from center.
        - layers[0]: [center_bin] (the center itself)
        - layers[1]: Direct neighbors of center
        - layers[2]: Bins 2 hops away
        - etc.
    """
```

!!! note "Hop Count vs Distance"
    This operates on **graph hops** (connectivity), not spatial distance.
    Two adjacent bins = 1 hop, regardless of their physical distance.
    For distance-based analysis, use `distance_to()` instead.

**Example:**

```python
# Get bins by hop distance from center
center_bin = 42
hop_layers = env.rings(center_bin, hops=5)

print(f"Center bin: {center_bin}")
for k, layer_bins in enumerate(hop_layers):
    print(f"  {k} hops: {len(layer_bins)} bins")

# Analyze neural activity by hop distance
mean_rates_by_hop = []
for k, layer_bins in enumerate(hop_layers):
    if len(layer_bins) > 0:
        mean_rate = firing_rate[layer_bins].mean()
        mean_rates_by_hop.append(mean_rate)
        print(f"Hop {k}: mean rate = {mean_rate:.2f} Hz")

# Visualize BFS layers
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 10))
colors = plt.cm.viridis(np.linspace(0, 1, len(hop_layers)))

for k, layer_bins in enumerate(hop_layers):
    ax.scatter(
        env.bin_centers[layer_bins, 0],
        env.bin_centers[layer_bins, 1],
        c=[colors[k]],
        s=150,
        label=f'{k} hops',
        edgecolors='white',
        linewidth=0.5
    )

# Highlight center
ax.scatter(
    env.bin_centers[center_bin, 0],
    env.bin_centers[center_bin, 1],
    c='red',
    s=400,
    marker='*',
    edgecolors='white',
    linewidth=2,
    label='Center',
    zorder=10
)

ax.set_title('BFS Layers (Hop Distance)')
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

**Use Cases:**

- Breadth-first search from landmarks
- Analyzing spatial spread through connectivity
- Local neighborhood analysis at different scales
- Graph-based spatial partitioning
- Testing connectivity and reachability

**Common Patterns:**

```python
# Count total bins within k hops
center_bin = 50
layers = env.rings(center_bin, hops=3)
bins_within_3_hops = np.concatenate(layers[:4])  # layers 0-3
print(f"Bins within 3 hops: {len(bins_within_3_hops)}")

# Analyze decay of neural activity with hop distance
hop_rates = []
for k, layer_bins in enumerate(layers):
    if len(layer_bins) > 0:
        hop_rates.append((k, firing_rate[layer_bins].mean()))

import matplotlib.pyplot as plt
hops_x, rates_y = zip(*hop_rates)
plt.plot(hops_x, rates_y, 'o-')
plt.xlabel('Hop Distance')
plt.ylabel('Mean Firing Rate (Hz)')
plt.title('Decay of Activity with Graph Distance')
plt.show()

# Get immediate neighbors (1-hop layer)
neighbors_1hop = layers[1] if len(layers) > 1 else np.array([], dtype=np.int32)
```

---

### Region Membership

Determine which bins belong to which regions (supports multiple membership).

```python
def region_membership(
    self,
    regions: Regions | None = None,
    *,
    include_boundary: bool = True,
) -> NDArray[np.bool_]:
    """
    Get boolean membership matrix for bins in regions.

    Parameters
    ----------
    regions : Regions, optional
        Regions to test membership for. If None, uses self.regions.
        Default is None.
    include_boundary : bool, optional
        If True, bins on region boundaries are considered members.
        If False, only bins strictly inside regions. Default is True.

    Returns
    -------
    membership : array, shape (n_bins, n_regions)
        Boolean matrix where membership[i, j] is True if bin i
        belongs to region j. Bins can belong to multiple regions
        (columns are independent).
    """
```

!!! note "Boolean Matrix"
    Returns a 2D boolean array `(n_bins, n_regions)`, NOT a 1D region index array.
    This allows bins to belong to multiple overlapping regions.

**Example:**

```python
# Define maze arms as regions
env.regions.add('north_arm', polygon=north_polygon)
env.regions.add('south_arm', polygon=south_polygon)
env.regions.add('east_arm', polygon=east_polygon)
env.regions.add('west_arm', polygon=west_polygon)

# Get region membership matrix
membership = env.region_membership()  # shape: (n_bins, 4)

region_names = list(env.regions.keys())
print("Region membership:")
for j, name in enumerate(region_names):
    n_bins = membership[:, j].sum()
    print(f"  {name}: {n_bins} bins")

# Check for bins in multiple regions
multi_region_bins = (membership.sum(axis=1) > 1)
print(f"Bins in multiple regions: {multi_region_bins.sum()}")

# Compute region-specific statistics
for j, name in enumerate(region_names):
    region_mask = membership[:, j]
    if region_mask.any():
        mean_occ = occupancy[region_mask].mean()
        mean_rate = firing_rate[region_mask].mean()
        print(f"{name}: occupancy={mean_occ:.2f}s, rate={mean_rate:.2f}Hz")

# Find bins exclusive to one region
exclusive_membership = membership & (membership.sum(axis=1, keepdims=True) == 1)
for j, name in enumerate(region_names):
    exclusive_bins = exclusive_membership[:, j].sum()
    print(f"{name} exclusive: {exclusive_bins} bins")
```

**Advanced Usage:**

```python
# Test membership for custom regions (without modifying env.regions)
custom_regions = Regions()
custom_regions.add('area1', polygon=poly1)
custom_regions.add('area2', polygon=poly2)
membership_custom = env.region_membership(regions=custom_regions)

# Strict interior membership (exclude boundaries)
membership_strict = env.region_membership(include_boundary=False)

# Convert to 1D region index (for non-overlapping regions)
# Assigns first matching region, -1 for no membership
region_indices = np.full(env.n_bins, -1, dtype=np.int32)
for j in range(membership.shape[1]):
    region_indices[membership[:, j] & (region_indices == -1)] = j
```

**Use Cases:**

- Multi-region spatial analysis
- Region-specific firing rate computations
- Compartment comparisons (maze arms, contexts)
- Handling overlapping regions
- Spatial segmentation and labeling

---

### Copy

Create a copy of the environment (deep or shallow).

```python
def copy(self, *, deep: bool = True) -> Environment:
    """
    Create copy of environment.

    Parameters
    ----------
    deep : bool, optional
        If True, create a deep copy (independent arrays and objects).
        If False, create a shallow copy (shares array references).
        Default is True.

    Returns
    -------
    env_copy : Environment
        Copy of environment.
        - Deep copy: Modifications to arrays don't affect original
        - Shallow copy: Shares references, modifications propagate
    """
```

**Example:**

```python
# Create baseline environment
env_baseline = Environment.from_samples(positions, bin_size=3.0, name='Session1')

# Deep copy (default) - fully independent
env_deep = env_baseline.copy(deep=True)
env_deep.regions.add('goal', point=np.array([75.0, 75.0]))

# Original unchanged
assert 'goal' not in env_baseline.regions
assert 'goal' in env_deep.regions

# Shallow copy - shares references
env_shallow = env_baseline.copy(deep=False)
# Modifications to shared objects may affect both

# Verify independence (deep copy)
env_deep_2 = env_baseline.copy()
env_deep_2.bin_centers[0] = [999.0, 999.0]
assert not np.allclose(env_baseline.bin_centers[0], [999.0, 999.0])
```

**Use Cases:**

- Safe experimentation with deep copy
- Creating environment variants
- Before/after comparisons
- Preserving original state
- Performance optimization with shallow copy (read-only use)

---

## Best Practices

### Choosing Parameters

**bin_size:**

- Start with 2-5 cm for rodent experiments
- Ensure at least 10-20 samples per bin on average
- Smaller bins = better resolution but more noise
- Balance spatial precision vs statistical power

**Smoothing (sigma):**

- Express in spatial units (cm), then convert to bins: `sigma_bins = sigma_cm / bin_size`
- Typical range: 5-10 cm smoothing for place fields
- More smoothing for noisy data, less for high-quality data
- Visualize raw and smoothed to verify appropriateness

**Occupancy thresholds:**

- Typical minimum: 0.1-1.0 seconds
- Too low: noisy estimates in under-sampled bins
- Too high: exclude potentially important regions
- Report threshold used for reproducibility

### Performance Considerations

**Large environments:**

```python
# Use sparse matrices for transitions
T = env.transitions(positions, normalize=False)  # Sparse CSR format
# DON'T convert to dense unless necessary
# T_dense = T.toarray()  # Only if env.n_bins is small (<1000)

# Subset environment for region-specific analysis
region_bins = get_region_bins(env, region_name)
env_region = env.subset(region_bins)
# Now all operations faster on smaller env_region
```

**Vectorization:**

```python
# GOOD: Vectorized operations
distances = env.distance_to(target_bin)

# AVOID: Loops when vectorization possible
# distances = np.array([env.distance_between_bins(i, target_bin)
#                       for i in range(env.n_bins)])
```

### Common Pitfalls

❌ **Forgetting to filter low occupancy:**

```python
# BAD: Division by tiny occupancy values
firing_rate = spike_counts / occupancy  # Unstable!

# GOOD: Filter first
valid = occupancy >= 0.5
firing_rate = np.full(env.n_bins, np.nan)
firing_rate[valid] = spike_counts[valid] / occupancy[valid]
```

❌ **Ignoring NaN handling:**

```python
# BAD: Operations propagate NaN unexpectedly
mean_rate = firing_rate.mean()  # NaN if any NaN values

# GOOD: Explicit NaN handling
mean_rate = np.nanmean(firing_rate)
# Or: mean_rate = firing_rate[~np.isnan(firing_rate)].mean()
```

❌ **Confusing geodesic and Euclidean distance:**

```python
# These can differ greatly in complex environments!
dist_geodesic = env.distance_to(goal, metric='geodesic')  # Through maze
dist_euclidean = env.distance_to(goal, metric='euclidean')  # Straight line

# Always specify which you're using
```

### Documentation Standards

When reporting analyses, include:

1. **Environment parameters:**
   - bin_size and units
   - Total bins and active bins
   - Spatial extent

2. **Filtering criteria:**
   - Minimum occupancy threshold
   - Invalid position handling
   - Any subset restrictions

3. **Processing steps:**
   - Smoothing parameters (sigma in spatial units AND bins)
   - Interpolation method and range
   - Normalization approach

4. **Example:**

> "Firing rate maps were computed on a 2.5 cm spatial grid (total bins: 256, active: 198).
> Bins with <0.5 s occupancy were excluded. Firing rates were smoothed with a Gaussian
> kernel (σ = 5 cm, 2.0 bins) using the `smooth()` method with `respect_nans=True`."

---

## See Also

- **[Environments Guide](environments.md)**: Core environment concepts and creation
- **[Complete Workflows](workflows.md)**: End-to-end analysis examples
- **[Spike Field Basics](../examples/08_spike_field_basics.ipynb)**: Practical demonstrations
- **[API Reference](../api/index.md)**: Complete API documentation

---

## Next Steps

- **Try it yourself**: Run the [Spike Field Basics notebook](../examples/08_spike_field_basics.ipynb)
- **Explore**: See how operations combine in [workflows guide](workflows.md)
- **Deep dive**: Check [API reference](../api/index.md) for all parameters and details
