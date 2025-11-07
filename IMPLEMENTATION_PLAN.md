# Implementation Plan: Spatial Primitives & Metrics

## Executive Summary

This plan outlines the implementation of **core spatial primitives** and **foundational metrics** for neurospatial v0.3.0, based on comprehensive investigation of capabilities, existing packages, and neuroscience requirements.

**Timeline**: 13 weeks (~3 months)
**Priority**: HIGH - Foundational infrastructure for spatial analysis
**Breaking Changes**: None (no current users - can rename directly)
**Authority**: Algorithms validated against opexebo (Moser lab, Nobel Prize 2014), neurocode (AyA Lab, Cornell), and ecology literature

### What This Release Enables

- **Differential operators** - gradient, divergence, Laplacian on irregular graphs (RL/replay analyses)
- **Signal processing primitives** - neighbor operations, custom convolutions
- **Place field metrics** - standard neuroscience analyses (detection, information, sparsity)
- **Population metrics** - coverage, density, overlap
- **Boundary cell metrics** - border score (wall-preferring cells)
- **Trajectory metrics** - turn angles, step lengths, home range, MSD (from ecology)
- **Behavioral segmentation** - automatic detection of runs, laps, trials

### Deferred to Future Releases

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

**Rationale**: Spatial autocorrelation is complex (FFT for regular grids, graph-based for irregular), grid cells require it, circular stats are specialized. Deferring these reduces scope by ~30% while delivering core functionality.

---

## Phase 1: Core Differential Operators (Weeks 1-3)

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

## Phase 2: Basic Signal Processing Primitives (Weeks 4-6)

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


## Phase 3: Core Metrics Module (Weeks 7-9)

### Goal
Provide standard neuroscience and trajectory analysis metrics as convenience wrappers.

### Module Structure

```
src/neurospatial/metrics/
    __init__.py
    place_fields.py      # Individual place field properties
    population.py        # Population-level metrics
    boundary_cells.py    # Border score, head direction
    trajectory.py        # Trajectory metrics from ecology
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

#### 3.4 Trajectory Metrics (Week 9)

**File**: `src/neurospatial/metrics/trajectory.py`

**Motivation**: Animal movement ecology packages (Traja, yupi, adehabitatHR) provide trajectory characterization metrics that are broadly applicable to neuroscience.

**Functions** (detailed implementations already in plan):
- `compute_turn_angles()` - Path tortuosity
- `compute_step_lengths()` - Graph distances
- `compute_home_range()` - Bins containing X% of time
- `mean_square_displacement()` - Diffusion classification

**Effort**: 3 days
**Risk**: Low
**Blockers**: None

---

#### 3.5 Documentation (Week 9)

**New user guide**: `docs/user-guide/neuroscience-metrics.md`

**Example notebooks**:
- `examples/10_place_field_analysis.ipynb` - Place field detection and metrics
- `examples/11_boundary_cell_analysis.ipynb` - Border score
- `examples/12_trajectory_analysis.ipynb` - Turn angles, MSD, home range

**Effort**: 2 days

---

## Phase 4: Behavioral Segmentation (Weeks 10-11)

### Goal
Implement automatic detection of behavioral epochs from continuous trajectories.

### Motivation
Most packages require manual trial/epoch segmentation. neurospatial can provide spatial primitives for automatic detection of runs, laps, and trials based on spatial regions and trajectory patterns.

### Module Structure

```
src/neurospatial/segmentation/
    __init__.py
    regions.py       # Region-based segmentation
    laps.py          # Lap detection
    trials.py        # Trial segmentation
    similarity.py    # Trajectory similarity
```

### Components

#### 4.1 Region-Based Segmentation (Week 10, Days 1-3)

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

#### 4.2 Lap Detection (Week 10, Days 4-5)

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

#### 4.3 Trial Segmentation (Week 10, Day 6)

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

#### 4.4 Trajectory Similarity (Week 11, Days 1-2)

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

#### 4.5 Tests & Documentation (Week 11, Days 3-5)

**Tests**:
- Test region crossing detection on synthetic trajectories
- Test lap detection on circular tracks (clockwise/counter-clockwise)
- Test trial segmentation on T-maze data
- Test trajectory similarity methods (Jaccard, DTW, etc.)

**Documentation**:
- User guide: `docs/user-guide/behavioral-segmentation.md`
- Example notebook: `examples/13_behavioral_segmentation.ipynb`

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
## Phase 5: Polish & Release (Weeks 12-13)

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

**UPDATED after scope reduction (deferred grid cells, circular stats, RatInABox)**:

| Phase | Duration | Person-Weeks | Risk Level |
|-------|----------|--------------|------------|
| 1. Differential Operators | 3 weeks | 3 | Low |
| 2. Basic Signal Processing | 3 weeks | 3 | Low |
| 3. Core Metrics Module | 3 weeks | 3 | Low |
| 4. Behavioral Segmentation | 2 weeks | 2 | Low |
| 5. Polish & Release | 2 weeks | 2 | Low |
| **Total** | **13 weeks** | **13** | **Low overall** |

**Phase breakdown**:
- **Phase 1** (3 weeks): Differential operator, gradient, divergence, docs
- **Phase 2** (3 weeks): neighbor_reduce, convolve
- **Phase 3** (3 weeks): Place fields, population, boundary cells, trajectory metrics, docs
- **Phase 4** (2 weeks): Region-based segmentation, laps, trials, similarity, docs
- **Phase 5** (2 weeks): Validation (2 days), performance (2 days), docs (3 days), release (2 days)

**Assumptions**:
- One full-time developer
- No major blockers
- Deferred complex features to v0.4.0 (spatial_autocorrelation, grid cells, circular stats)

**Changes from v17 plan**:
- **Timeline**: 17 weeks → 13 weeks (**24% reduction**)
- **Deferred to v0.4.0**: spatial_autocorrelation, grid cells, coherence, circular statistics
- **Deferred examples**: RatInABox integration
- **Risk**: MEDIUM → LOW (removed most complex feature)

**Optimistic**: 12 weeks (if all implementations straightforward)
**Pessimistic**: 15 weeks (if behavioral segmentation needs iteration)

---

## Dependencies & Blockers

```
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
