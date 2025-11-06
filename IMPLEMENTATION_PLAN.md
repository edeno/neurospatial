# Implementation Plan: Spatial Primitives & Metrics

## Executive Summary

This plan outlines the implementation of **missing spatial primitives** and **convenience metrics** for neurospatial, based on comprehensive investigation of capabilities, existing packages, and neuroscience requirements.

**Timeline**: 14 weeks (3.5 months) - UPDATED after opexebo analysis
**Priority**: HIGH - Enables analyses not possible in any other package
**Breaking Changes**: Minimal (one function rename required)
**Authority**: Algorithms validated against opexebo (Moser lab, Nobel Prize 2014)

### What This Enables

- **Grid cell analysis** (Nobel Prize 2014) - currently impossible without spatial autocorrelation
- **RL/replay analyses** - value iteration, successor representation, Bellman backups
- **Standard place field metrics** - convenience wrappers for common analyses
- **Differential operators** - gradient, divergence, Laplacian on irregular graphs

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

**BREAKING CHANGE**: Current `divergence()` is KL/JS divergence, needs renaming!

**Migration strategy**:
```python
# Rename existing function
# In src/neurospatial/field_ops.py
def kl_divergence(p, q, *, kind='kl', eps=1e-12):
    """
    Compute divergence between probability distributions.

    .. deprecated:: 0.3.0
        Use `kl_divergence` instead. The name `divergence` is reserved
        for the vector field divergence operator.
    """
    # Existing implementation

# Add alias for backward compatibility
divergence = kl_divergence  # Deprecated alias
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

**Effort**: 2 days (including migration plan)
**Risk**: Medium (breaking change, need migration guide)
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

## Phase 2: Spatial Signal Processing Primitives (Weeks 4-9)

### Goal
Implement primitives for neuroscience-specific spatial analyses.

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

#### 2.2 spatial_autocorrelation (Weeks 5-8)

**THIS IS THE CRITICAL PRIMITIVE** - Enables grid cell analysis.

**UPDATED AFTER OPEXEBO ANALYSIS**: Risk reduced from HIGH → MEDIUM

**Key insight from opexebo** (Moser lab, Nobel Prize 2014):
- They use normalized 2D cross-correlation via FFT (fast, validated)
- Assumes regular rectangular grids
- Crops 20% of edges to reduce boundary artifacts
- This is the field-standard approach

**Proposed approach** (UPDATED):

**Step 1: Adopt opexebo's FFT approach** (Week 5-6) - **RISK: LOW**
- For regular grids: Use FFT-based 2D cross-correlation (opexebo method)
- Fast, validated, matches field standard
- Authority: Nobel Prize-winning lab

**Step 2: Extend to irregular graphs** (Week 7-8) - **RISK: MEDIUM** (optional)

**Implementation** (UPDATED - adopts opexebo approach):
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

**Decision**: Start with FFT method (opexebo approach), add graph-based if users need it.

**Step 3: Validation** (Week 8)
- Test on synthetic hexagonal grid data
- **Compare with opexebo outputs** (should match within 1%)
- Validate rotation sensitivity

**Tests** (UPDATED):
```python
def test_autocorr_matches_opexebo():
    """Autocorrelation should match opexebo for regular grids"""
    # Create regular grid environment
    env = Environment.from_samples(positions, bin_size=2.5)
    firing_rate = create_hexagonal_pattern()

    # Compute with neurospatial
    autocorr_ns = spatial_autocorrelation(firing_rate, env)

    # Compute with opexebo (if installed)
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

**Effort**: 16-20 days (4 weeks) - UNCHANGED
**Risk**: MEDIUM (reduced from HIGH) - opexebo provides validated algorithm
**Blockers**: None (can implement independently)
**Mitigation**: Adopt opexebo's FFT approach (low risk), defer graph-based (optional)

---

#### 2.3 convolve (Week 9)

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

## Phase 3: Path-Based Operations (Week 10)

### Goal
Implement trajectory and RL primitives.

### Components

#### 3.1 accumulate_along_path

**Add to**: `src/neurospatial/primitives.py`

**Implementation from**: `primitives_poc.py`

**Effort**: 3 days
**Risk**: Low (prototype exists)

---

#### 3.2 propagate

**Add to**: `src/neurospatial/primitives.py`

**Implementation from**: `primitives_poc.py`

**Note**: Evaluate if this is redundant with `distance_field` - may defer or remove.

**Effort**: 2 days
**Risk**: Low
**Decision**: Defer pending user feedback

---

## Phase 4: Convenience Metrics Module (Weeks 11-14)

### Goal
Provide standard neuroscience metrics as convenience wrappers.

### Module Structure

```
src/neurospatial/metrics/
    __init__.py
    place_fields.py      # Individual place field properties
    population.py        # Population-level metrics
    remapping.py         # Stability and remapping
    grid_cells.py        # Grid score (needs spatial_autocorrelation)
    boundary_cells.py    # Border score, head direction
```

### Components

#### 4.1 Place Field Metrics (Week 11)

**File**: `src/neurospatial/metrics/place_fields.py`

**Functions**:
```python
def detect_place_fields(
    firing_rate: NDArray,
    env: Environment,
    *,
    threshold: float = 0.2,
    min_size: float | None = None,
) -> list[NDArray[np.int64]]:
    """Detect place fields as connected components above threshold."""

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

#### 4.2 Population Metrics (Week 12)

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

#### 4.3 Grid Cell Metrics (Week 13-14)

**File**: `src/neurospatial/metrics/grid_cells.py`

**UPDATED**: Adopt opexebo's sophisticated algorithm

**Functions**:
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

def grid_orientation(autocorr_map: NDArray, env: Environment) -> float:
    """Estimate grid orientation (degrees)."""

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

**Key updates**:
- ✅ Adopt opexebo's annular rings approach
- ✅ Automatic radius detection
- ✅ Sliding window smoothing (3 radii)
- ✅ Cross-reference opexebo as authority

**Effort**: 3 days (reduced from 5) - well-defined algorithm
**Risk**: Low (reduced from Medium) - opexebo provides exact specification
**Blockers**: Phase 2.2 (spatial_autocorrelation)

---

#### 4.4 Documentation (Week 14)

**New user guide**: `docs/user-guide/neuroscience-metrics.md`

**Example notebook**: `examples/10_place_field_analysis.ipynb`

**Example notebook**: `examples/11_grid_cell_detection.ipynb`

**Effort**: 3 days

---

## Phase 5: Polish & Release (Weeks 13-14)

**UPDATED**: Overlaps with Phase 4 (reduced timeline)

### Components

#### 5.1 Validation Against opexebo (NEW)
- Test grid score matches opexebo within 1%
- Test coherence matches exactly
- Test spatial information matches exactly
- Test autocorrelation matches exactly
- Document any intentional differences

**Effort**: 2 days

#### 5.2 Performance Optimization
- Profile critical paths
- Optimize hot loops
- Add caching where beneficial

**Effort**: 2 days

#### 5.3 Documentation Polish
- API reference generation
- Cross-linking between docs
- Cross-references to opexebo
- Tutorial videos (optional)

**Effort**: 2 days

#### 5.4 Migration Guide
- Breaking changes (divergence rename)
- Upgrade instructions
- Deprecation timeline
- opexebo integration examples

**Effort**: 1 day

#### 5.5 Release
- Version bump to 0.3.0
- Changelog highlighting opexebo compatibility
- Blog post / announcement
- PyPI release

**Effort**: 1 day

---

## Success Criteria

### Phase 1 (Differential Operators)
- [ ] D matrix construction passes all tests
- [ ] gradient(), divergence() work on all layout types
- [ ] div(grad(f)) == Laplacian(f) validated
- [ ] 50x caching speedup confirmed
- [ ] Breaking change migration guide published

### Phase 2 (Signal Processing Primitives)
- [ ] neighbor_reduce() works on all layout types
- [ ] spatial_autocorrelation() produces hexagonal peaks for grid cells
- [ ] **Autocorrelation matches opexebo within 1%** (for regular grids) ← NEW
- [ ] Grid score matches published values (±0.05)
- [ ] convolve() supports arbitrary kernels

### Phase 3 (Path Operations)
- [ ] accumulate_along_path() supports discount factors
- [ ] Value iteration converges (validated in POC)

### Phase 4 (Metrics Module)
- [ ] Place field metrics match manual calculations
- [ ] **Grid score matches opexebo within 1%** (for regular grids) ← NEW
- [ ] **Coherence matches opexebo exactly** (for regular grids) ← NEW
- [ ] **Spatial information matches opexebo exactly** ← NEW
- [ ] Grid score detects known grid cells
- [ ] Coherence correlates with place field quality
- [ ] All metrics have examples and citations
- [ ] All metrics cross-reference opexebo in documentation

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

**2. Breaking change (divergence rename)**
- **Mitigation**: Deprecation warnings, clear migration guide
- **Fallback**: Keep alias for 1 version, remove in 0.4.0
- **User communication**: Announce early, provide examples
- **Documentation**: Show opexebo integration patterns

**3. Performance regressions**
- **Mitigation**: Benchmark suite in CI/CD
- **Monitoring**: Track key operations (smooth, distance_field, gradient)
- **Target**: No operation >10% slower than baseline
- **Baseline**: opexebo performance for regular grids

### Low-Risk Items

**4. API design conflicts**
- **Status**: LOW RISK - opexebo provides reference APIs
- **Mitigation**: Match opexebo signatures where possible
- **Validation**: User feedback on proposed extensions

**5. Grid score validation**
- **Status**: LOW RISK - opexebo provides gold standard
- **Mitigation**: Test against opexebo outputs (should match within 1%)
- **Resources**: opexebo test cases provide validation data
- **Authority**: Nobel Prize-winning lab implementation

**6. Algorithm correctness** (NEW)
- **Status**: LOW RISK - adopting validated algorithms
- **Mitigation**: Cross-reference opexebo for all overlapping metrics
- **Testing**: Validate outputs match opexebo exactly for regular grids
- **Documentation**: Document intentional differences (irregular graph support)

---

## Effort Estimation

**UPDATED after opexebo analysis**:

| Phase | Duration | Person-Weeks | Risk Level |
|-------|----------|--------------|------------|
| 1. Differential Operators | 3 weeks | 3 | Low |
| 2. Signal Processing | 6 weeks | 6 | Medium (was HIGH) |
| 3. Path Operations | 1 week | 1 | Low |
| 4. Metrics Module | 2 weeks (was 4) | 2 | Low (was Medium) |
| 5. Polish & Release | 2 weeks | 2 | Low |
| **Total** | **14 weeks** | **14** | **Medium overall** |

**Assumptions**:
- One full-time developer
- No major blockers
- spatial_autocorrelation uses opexebo's FFT approach (validated algorithm)

**Changes from original plan**:
- **Timeline reduced**: 16 weeks → 14 weeks
- **Risk reduced**: spatial_autocorrelation HIGH → MEDIUM (adopt opexebo's approach)
- **Metrics reduced**: 4 weeks → 2 weeks (well-defined algorithms from opexebo)

**Optimistic**: 12 weeks (if FFT implementation straightforward)
**Pessimistic**: 16 weeks (if graph-based autocorrelation needed for irregular grids)

---

## Dependencies & Blockers

```
Phase 1: Differential Operators
├── 1.1 D matrix (no blockers) ───┐
├── 1.2 gradient (needs 1.1) ─────┤
├── 1.3 divergence (needs 1.1) ───┤
└── 1.4 docs (needs 1.2, 1.3) ────┘

Phase 2: Signal Processing
├── 2.1 neighbor_reduce (no blockers, LOW RISK)
├── 2.2 spatial_autocorrelation (no blockers, MEDIUM RISK - opexebo FFT approach)
└── 2.3 convolve (no blockers, LOW RISK)

Phase 3: Path Operations
└── 3.1 accumulate (no blockers)

Phase 4: Metrics Module
├── 4.1 place_fields (no blockers)
├── 4.2 population (no blockers)
├── 4.3 grid_cells (needs 2.2 spatial_autocorrelation) ← BLOCKER
└── 4.4 docs (needs 4.1, 4.2, 4.3)

Phase 5: Release
└── needs all above
```

**Critical path**: spatial_autocorrelation (2.2) → grid_score (4.3)

**Parallelization opportunity**: Can implement Phase 1, 3, 4.1, 4.2 in parallel with Phase 2.2

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

### Validation Tests (UPDATED)
- Compare with NetworkX (Laplacian)
- Compare with PyGSP (gradient, divergence)
- **Compare with opexebo** (autocorrelation, grid score, coherence, spatial info) ← NEW
- Validate grid scores match opexebo within 1%
- Validate on synthetic data with known properties (hexagonal patterns)

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

This implementation plan delivers **critical missing functionality** that:

1. **Enables Nobel Prize-winning analyses** (grid cells) not possible in any other package
2. **Provides fundamental spatial primitives** for RL/replay/behavioral analysis
3. **Reduces code duplication** across labs with standard metrics
4. **Maintains backward compatibility** except one well-managed breaking change

**Timeline**: 12-16 weeks
**Risk**: Manageable (one high-risk component with mitigation strategy)
**Impact**: HIGH - Positions neurospatial as THE package for spatial neuroscience

**Critical decision needed**: Approve breaking change (divergence rename) and migration strategy.

**Next step**: Review with maintainers and get approval to proceed with Phase 1.
