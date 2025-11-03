# Environment Operations - Implementation Plan

**Version**: 2.0
**Last Updated**: 2025-11-03
**Status**: Ready for Implementation

## Overview

This document specifies a comprehensive set of spatial analysis operations for the `Environment` class in neurospatial. The plan is organized into four priority phases (P0→P3), covering occupancy analysis, trajectory processing, field smoothing, and utility functions.

**Goals**:

- Make Environment operations feature-complete for neuroscience spatial analysis workflows
- Maintain backward compatibility - all changes are additive
- Follow existing codebase patterns and conventions
- Provide consistent APIs across all layout types where applicable

**Design Principles**:

1. **Immutability**: Environments remain immutable after creation; operations return new data or new Environment instances
2. **Fitted-state enforcement**: All methods use `@check_fitted` decorator to ensure environment is properly initialized
3. **Input validation**: Comprehensive validation with clear error messages including diagnostic information
4. **Layout compatibility**: Operations work on all layout types where feasible; grid-specific operations clearly documented
5. **Caching**: Object identity-based caching for expensive operations (kernels, KDTree)
6. **Deterministic behavior**: Default parameters ensure reproducible results

---

## Table of Contents

- [P0 - Core Analysis Operations](#p0---core-analysis-operations)
- [P1 - Smoothing, Resampling, Masking](#p1---smoothing-resampling-masking)
- [P2 - Field & Interpolation Utilities](#p2---field--interpolation-utilities)
- [P3 - QoL & Robustness](#p3---qol--robustness)
- [Cross-Cutting Concerns](#cross-cutting-concerns)
- [File Organization](#file-organization)
- [Testing Strategy](#testing-strategy)
- [Implementation Roadmap](#implementation-roadmap)
- [Layout Compatibility Matrix](#layout-compatibility-matrix)
- [Acceptance Criteria](#acceptance-criteria)

---

## P0 - Core Analysis Operations

**Goal**: Ship these first - they cover 80% of neuroscience spatial analysis workflows.

### 1. Occupancy / Dwell Time

**Goal**: Compute time-in-bin from `(time, position)` samples with optional speed filtering and gap handling.

**API** (in `environment.py`):

```python
@check_fitted
def occupancy(
    self,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = 0.5,
    kernel_bandwidth: float | None = None,  # More explicit than 'smooth'
    outside_value: float = 0.0,
) -> NDArray[np.float64]  # shape (n_bins,)
    """
    Compute occupancy (time spent in each bin).

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates matching environment dimensions.
    speed : NDArray[np.float64], shape (n_samples,), optional
        Instantaneous speed at each sample. If provided with min_speed,
        samples below threshold are excluded.
    min_speed : float, optional
        Minimum speed threshold (requires speed parameter).
    max_gap : float, optional
        Maximum time gap in seconds. Intervals with Δt > max_gap are
        not counted. Default: 0.5 seconds.
    kernel_bandwidth : float, optional
        If provided, apply diffusion kernel smoothing with this bandwidth
        (in physical units). Uses mode='density' to respect bin volumes.
    outside_value : float, default=0.0
        Occupancy value for samples outside environment bounds.

    Returns
    -------
    occupancy : NDArray[np.float64], shape (n_bins,)
        Time in seconds spent in each bin. Guarantees:
        occupancy.sum() ≈ total_valid_time (within numerical precision).

    Examples
    --------
    >>> # Basic occupancy
    >>> occ = env.occupancy(times, positions)
    >>>
    >>> # Filter slow periods and smooth
    >>> occ = env.occupancy(times, positions, speed=speeds,
    ...                      min_speed=2.0, kernel_bandwidth=5.0)

    Notes
    -----
    Time allocation: Each time interval Δt is assigned entirely to the starting
    bin. For more accurate boundary handling on regular grids, see P2.11 linear
    occupancy method.

    Raises
    ------
    ValueError
        If times and positions have different lengths, or if arrays are empty.
    """
```

**Design**:

1. **Input validation**:
   - Convert to numpy arrays with dtype checks
   - Validate `len(times) == len(positions)`
   - Handle empty arrays gracefully (return zeros)
2. Use `spatial.map_points_to_bins(..., tie_break="lowest_index")` to map positions → bin indices
3. Compute Δt between consecutive samples: `dt = np.diff(times)`
4. Filter intervals where `dt > max_gap` (set to zero)
5. If `speed` and `min_speed` provided, mask samples where `speed < min_speed`
6. Accumulate time per bin: `occupancy[bin_idx[i]] += dt[i]` (assign interval to start bin)
7. If `kernel_bandwidth` is not None:
   - Compute kernel: `K = self.compute_kernel(kernel_bandwidth, mode='density')`
   - Return `K @ occupancy`
8. Guarantee: `occupancy.sum()` equals total valid time

**Tests**:

- Synthetic L-shaped path on 2D grid with known segment durations → verify exact occupancy
- Sparse samples with large gaps → verify gaps are excluded from total
- Speed filtering → verify only high-speed periods counted
- Smoothing → verify total time conserved after smoothing
- Edge case: all samples outside environment → all zeros

---

### 2. Trajectory → Bin Sequences / Runs

**Goal**: Convert continuous trajectory to bin index sequences with run-length encoding.

**API** (in `environment.py`):

```python
@check_fitted
def bin_sequence(
    self,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    dedup: bool = True,
    return_runs: bool = False,
    outside_value: int | None = -1,  # Allow None to drop outside samples
) -> NDArray[np.int32] | tuple[NDArray[np.int32], NDArray[np.int64], NDArray[np.int64]]
    """
    Map trajectory to sequence of bin indices.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    dedup : bool, default=True
        If True, collapse consecutive repeats: [A,A,A,B] → [A,B].
    return_runs : bool, default=False
        If True, also return run boundaries (indices into times array).
    outside_value : int, default=-1
        Bin index for samples outside environment. Use None to drop
        these samples entirely.

    Returns
    -------
    bins : NDArray[np.int32], shape (n_sequences,)
        Bin index at each time point (or deduplicated sequence).
    run_start_idx : NDArray[np.int64], shape (n_runs,), optional
        Start index (into times) of each contiguous run.
    run_end_idx : NDArray[np.int64], shape (n_runs,), optional
        End index (inclusive, into times) of each run.

    Notes
    -----
    A "run" is a maximal contiguous subsequence in the same bin.
    If outside_value=-1, runs are split at boundary crossings.

    Examples
    --------
    >>> bins = env.bin_sequence(times, positions, dedup=True)
    >>> bins, starts, ends = env.bin_sequence(times, positions, return_runs=True)
    >>> # Duration of first run:
    >>> duration = times[ends[0]] - times[starts[0]]
    """
```

**Design**:

1. Map positions → bin indices via KDTree
2. If `outside_value` is None, filter out `-1` entries
3. If `dedup=True`, find change points: `np.where(np.diff(bins) != 0)[0]`
4. If `return_runs=True`:
   - Find run boundaries by change points
   - Return `(unique_bins, run_starts, run_ends)`

**Tests**:

- Known toy trajectory: [bin 5 for 10 samples, bin 7 for 5 samples] → verify 2 runs with correct boundaries
- Deduplication: [A,A,B,B,B,C] → [A,B,C]
- Outside handling: trajectory crossing boundary → runs split correctly

---

### 3. Transitions / Adjacency Weights

**Goal**: Empirical Markov transition matrix from observed movements.

**API** (in `environment.py`):

```python
@check_fitted
def transitions(
    self,
    bins: NDArray[np.int32] | None = None,
    *,
    times: NDArray[np.float64] | None = None,
    positions: NDArray[np.float64] | None = None,
    lag: int = 1,
    normalize: bool = True,
    allow_teleports: bool = False,
) -> scipy.sparse.csr_matrix
    """
    Compute empirical transition matrix from trajectory.

    Parameters
    ----------
    bins : NDArray[np.int32], shape (n_samples,), optional
        Precomputed bin sequence. If None, computed from times/positions.
    times : NDArray[np.float64], shape (n_samples,), optional
        Required if bins is None.
    positions : NDArray[np.float64], shape (n_samples, n_dims), optional
        Required if bins is None.
    lag : int, default=1
        Temporal lag for transitions: count bins[t] → bins[t+lag].
    normalize : bool, default=True
        If True, return row-stochastic matrix (rows sum to 1).
    allow_teleports : bool, default=False
        If False, only count transitions between graph-adjacent bins.
        If True, count all transitions (including non-local jumps).

    Returns
    -------
    T : scipy.sparse.csr_matrix, shape (n_bins, n_bins)
        Transition matrix where T[i,j] = P(next_bin=j | current_bin=i).
        If normalize=True, each row sums to 1.
        If normalize=False, T[i,j] = raw count of i→j transitions.

    Notes
    -----
    When allow_teleports=False, non-adjacent transitions are ignored.
    This filters out tracking errors and enforces physical continuity.

    Examples
    --------
    >>> # Empirical transition probabilities
    >>> T = env.transitions(times=times, positions=positions)
    >>> # Probability of moving from bin 10 to its neighbors:
    >>> T[10, :].toarray()
    """
```

**Design**:

1. If `bins` not provided, compute via `self.bin_sequence(times, positions, dedup=False)`
2. Extract pairs: `(bins[:-lag], bins[lag:])`
3. If `allow_teleports=False`:
   - Build adjacency set from `self.connectivity`
   - Filter pairs where `(i, j)` not in adjacency
4. Count transitions in sparse COO format: `counts[i, j] += 1`
5. Convert to CSR, optionally normalize rows

**Tests**:

- 1D track sequence: [0,1,2,3,2,1,0] → verify symmetric transitions
- Grid with teleport: non-adjacent jump → excluded when `allow_teleports=False`
- Normalization: all rows sum to 1 when `normalize=True`

---

### 4. Connected Components / Reachability

**Goal**: Identify traversable subgraphs and compute neighborhoods.

**API** (in `environment.py`):

```python
def components(
    self,
    largest_only: bool = False,
) -> list[NDArray[np.int32]]:
    """
    Find connected components of the environment graph.

    Parameters
    ----------
    largest_only : bool, default=False
        If True, return only the largest component.

    Returns
    -------
    components : list[NDArray[np.int32]]
        List of bin index arrays, one per component.
        Sorted by component size (largest first).

    Examples
    --------
    >>> comps = env.components()
    >>> print(f"Found {len(comps)} components")
    >>> largest = env.components(largest_only=True)[0]
    """

def reachable_from(
    self,
    source_bin: int,
    *,
    radius: int | float | None = None,
    metric: Literal["hops", "geodesic"] = "hops",
) -> NDArray[np.bool_]:
    """
    Find all bins reachable from source within optional radius.

    Parameters
    ----------
    source_bin : int
        Starting bin index.
    radius : int or float, optional
        Maximum distance/hops. If None, find all reachable bins.
    metric : {'hops', 'geodesic'}, default='hops'
        - 'hops': radius in graph edges (BFS).
        - 'geodesic': radius in physical units (Dijkstra on edge distances).

    Returns
    -------
    reachable : NDArray[np.bool_], shape (n_bins,)
        Boolean mask where True indicates reachable bins.

    Examples
    --------
    >>> # All bins within 3 edges of bin 10
    >>> mask = env.reachable_from(10, radius=3, metric='hops')
    >>>
    >>> # All bins within 50cm geodesic distance
    >>> mask = env.reachable_from(goal_bin, radius=50.0, metric='geodesic')
    """
```

**Design**:

- `components()`: Use `nx.connected_components(self.connectivity)`
- `reachable_from()`:
  - If `metric='hops'`: BFS to depth `radius`
  - If `metric='geodesic'`: Single-source Dijkstra with cutoff

**Tests**:

- Graph with isolated island (2 components) → verify separation
- BFS with radius=2 on grid → verify correct neighborhood size
- Geodesic with radius=10.0 → verify physical distance constraint

---

### 5. Diffusion Kernel Infrastructure

**Goal**: Foundational smoothing mechanism via heat kernel on graph.

**API** (new file `kernels.py`):

```python
def compute_diffusion_kernels(
    graph: nx.Graph,
    bandwidth_sigma: float,
    bin_sizes: Optional[NDArray] = None,
    mode: Literal["transition", "density"] = "transition",
) -> NDArray[np.float64]:
    """
    Compute diffusion-based kernel via matrix-exponential of graph Laplacian.

    The kernel K[i,j] represents the smoothed influence of a unit mass at bin j
    on bin i after diffusion time t = bandwidth_sigma^2 / 2.

    Parameters
    ----------
    graph : nx.Graph
        Nodes represent bins. Each edge must have a 'distance' attribute.
    bandwidth_sigma : float
        Gaussian bandwidth (σ) in physical units. Controls smoothing scale.
    bin_sizes : NDArray[np.float64], shape (n_bins,), optional
        Physical area/volume of each bin. Required for mode='density'.
    mode : {'transition', 'density'}, default='transition'
        Normalization mode:

        - 'transition': Each column sums to 1 (discrete probability).
        - 'density': Each column integrates to 1 over bin volumes (continuous density).

    Returns
    -------
    kernel : ndarray, shape (n_bins, n_bins)
        Diffusion kernel matrix. Column j = smoothed delta function at bin j.

    Raises
    ------
    KeyError
        If any edge lacks a 'distance' attribute.
    ValueError
        If mode='density' but bin_sizes is None, or bin_sizes has wrong shape.

    Notes
    -----
    The kernel is computed as:

    .. math::
        K = \\exp(-t L)

    where :math:`L` is the graph Laplacian and :math:`t = \\sigma^2 / 2`.

    For volume-corrected diffusion (mode='density'), we use :math:`L = M^{-1}(D-W)`
    where :math:`M = \\text{diag}(\\text{bin_sizes})`.

    Examples
    --------
    >>> # Smooth a spike rate map
    >>> kernel = compute_diffusion_kernels(env.connectivity, bandwidth_sigma=5.0,
    ...                                     bin_sizes=env.layout.bin_sizes(),
    ...                                     mode='density')
    >>> smoothed_rates = kernel @ raw_rates
    """
```

**API** (in `environment.py`):

```python
def compute_kernel(
    self,
    bandwidth: float,
    *,
    mode: Literal["transition", "density"] = "density",
    cache: bool = True,
) -> NDArray[np.float64]:
    """
    Convenience wrapper for compute_diffusion_kernels.

    Automatically uses self.connectivity and self.layout.bin_sizes().

    Parameters
    ----------
    bandwidth : float
        Smoothing bandwidth in physical units.
    mode : {'transition', 'density'}, default='density'
        See compute_diffusion_kernels.
    cache : bool, default=True
        Cache result keyed by (graph_version, bandwidth, mode).

    Returns
    -------
    kernel : NDArray[np.float64], shape (n_bins, n_bins)
    """
```

**Design**:

1. Assign Gaussian edge weights: `w_uv = exp(-d_uv^2 / (2*sigma^2))`
2. Build Laplacian: `L = D - W`
3. If `bin_sizes` provided, compute volume-corrected: `L = M^-1 @ L`
4. Matrix exponential: `kernel = expm(-t * L)` where `t = sigma^2 / 2`
5. Normalize:
   - `mode='transition'`: column sums = 1
   - `mode='density'`: weighted column sums = 1 (integrate over volumes)
6. Cache by `(env._version, bandwidth, mode)`

**Tests**:

- Symmetry: For uniform regular grid, `kernel ≈ kernel.T`
- Mass conservation: `(kernel @ field).sum() ≈ field.sum()`
- Normalization (transition): `kernel.sum(axis=0) ≈ 1.0`
- Normalization (density): `(kernel[:, j] * bin_sizes).sum() ≈ 1.0`
- Cache invalidation: Modify graph → cache miss

---

## P1 - Smoothing, Resampling, Masking

### 6. Graph/Grid Smoothing

**Goal**: Apply diffusion smoothing to arbitrary fields (unified interface).

**API** (in `environment.py`):

```python
def smooth(
    self,
    field: NDArray[np.float64],
    bandwidth: float,
    *,
    mode: Literal["transition", "density"] = "density",
) -> NDArray[np.float64]:
    """
    Apply diffusion kernel smoothing to field.

    Parameters
    ----------
    field : NDArray[np.float64], shape (n_bins,)
        Values per bin to smooth.
    bandwidth : float
        Smoothing bandwidth in physical units (σ).
    mode : {'transition', 'density'}, default='density'
        See compute_kernel documentation.

    Returns
    -------
    smoothed : NDArray[np.float64], shape (n_bins,)
        Smoothed field values.

    Examples
    --------
    >>> # Smooth spike counts
    >>> smoothed_spikes = env.smooth(spike_counts, bandwidth=5.0, mode='density')
    >>>
    >>> # Smooth probability distribution
    >>> smoothed_prob = env.smooth(posterior, bandwidth=3.0, mode='transition')
    """
```

**Design**:

1. Compute kernel: `K = self.compute_kernel(bandwidth, mode=mode)`
2. Apply: `smoothed = K @ field`
3. Works uniformly across all layout types (grids, graphs, meshes)

**Tests**:

- Impulse at one bin → spreads to neighbors, conserves mass
- Constant field → remains constant after smoothing
- Edge preservation: smoothing doesn't leak beyond disconnected components

---

### 7. Rebin / Resample (Grid-only)

**Goal**: Conservative coarsening for regular grids.

**API** (in `environment.py`):

```python
def rebin(
    self,
    factor: int | tuple[int, ...],
    *,
    method: Literal["sum", "mean"] = "sum",
) -> "Environment":
    """
    Coarsen regular grid by integer factor (grid-only operation).

    Parameters
    ----------
    factor : int or tuple of int
        Coarsening factor per dimension. If int, applied uniformly.
    method : {'sum', 'mean'}, default='sum'
        Aggregation method:
        - 'sum': Mass-preserving (for counts, occupancy).
        - 'mean': Average values (for rates, probabilities).

    Returns
    -------
    coarse_env : Environment
        New environment with reduced resolution.

    Raises
    ------
    TypeError
        If environment is not a regular grid layout.

    Examples
    --------
    >>> # 100x100 grid → 50x50 grid (2x coarser)
    >>> coarse = env.rebin(factor=2, method='sum')
    """
```

**Design**:

1. Check layout type: only `RegularGridLayout` supported
2. Compute new edges: `new_edges = old_edges[::factor]`
3. Reshape fields to grid, apply `block_reduce` (sum or mean)
4. Build new connectivity from coarse grid
5. New bin centers = mean of grouped sub-centers

**Tests**:

- Constant field with `method='sum'` → total mass preserved
- 2D grid 10x10 → 5x5 with factor=2 → verify bin positions
- Non-grid layout → raises TypeError

---

### 8. Subset / Crop / Mask

**Goal**: Extract subregion as new environment.

**API** (in `environment.py`):

```python
def subset(
    self,
    *,
    bins: NDArray[np.bool_] | None = None,
    region_names: Sequence[str] | None = None,
    polygon: "shapely.Polygon" | None = None,
    invert: bool = False,
) -> "Environment":
    """
    Create new environment containing subset of bins.

    Parameters
    ----------
    bins : NDArray[np.bool_], shape (n_bins,), optional
        Boolean mask of bins to keep.
    region_names : Sequence[str], optional
        Keep bins inside these named regions.
    polygon : shapely.Polygon, optional
        Keep bins whose centers lie inside polygon.
    invert : bool, default=False
        If True, invert the selection mask.

    Returns
    -------
    sub_env : Environment
        New environment with selected bins renumbered to [0, n'-1].
        Connectivity is induced subgraph. Regions are dropped.

    Notes
    -----
    Exactly one of {bins, region_names, polygon} must be provided.

    Examples
    --------
    >>> # Extract bins inside 'goal' region
    >>> goal_env = env.subset(region_names=['goal'])
    >>>
    >>> # Crop to polygon
    >>> from shapely.geometry import box
    >>> cropped = env.subset(polygon=box(0, 0, 50, 50))
    """
```

**Design**:

1. Build mask from one of: bins, region_names, polygon
2. Extract subgraph: `G_sub = self.connectivity.subgraph(selected_nodes)`
3. Renumber nodes to contiguous [0..n'-1]
4. Create new layout with subset of bin_centers and new connectivity
5. Drop all regions (user can re-add)

**Tests**:

- Crop 10x10 grid to left half → 5x10 bins
- Polygon selection → verify bin centers inside
- Connectivity preserved: neighbors in original remain neighbors in subset

---

## P2 - Field & Interpolation Utilities

### 9. Per-bin ↔ Continuous Interpolation

**Goal**: Evaluate bin-valued fields at arbitrary points.

**API** (in `environment.py`):

```python
def interpolate(
    self,
    field: NDArray[np.float64],
    points: NDArray[np.float64],
    *,
    mode: Literal["nearest", "linear"] = "nearest",
) -> NDArray[np.float64]:
    """
    Interpolate field values at arbitrary points.

    Parameters
    ----------
    field : NDArray[np.float64], shape (n_bins,)
        Values per bin.
    points : NDArray[np.float64], shape (n_points, n_dims)
        Query points in environment coordinates.
    mode : {'nearest', 'linear'}, default='nearest'
        Interpolation mode:
        - 'nearest': Use value of nearest bin.
        - 'linear': Bilinear (2D) or trilinear (3D) for regular grids only.

    Returns
    -------
    values : NDArray[np.float64], shape (n_points,)
        Interpolated field values. Points outside environment → NaN.

    Raises
    ------
    NotImplementedError
        If mode='linear' requested for non-grid layout.

    Examples
    --------
    >>> # Evaluate rate map at continuous positions
    >>> rates_at_pos = env.interpolate(rate_map, trajectory_points, mode='linear')
    """
```

**Design**:

- `mode='nearest'`: Use KDTree on bin_centers
- `mode='linear'`:
  - Check layout is `RegularGridLayout`
  - Use `scipy.interpolate.RegularGridInterpolator`
  - Reshape field to grid, evaluate at points

**Tests**:

- Known plane field `f(x,y) = x + 2*y` on grid → bilinear matches exactly
- Nearest mode on irregular graph → picks closest bin center
- Points outside → NaN values

---

### 10. Field Math Helpers

**Goal**: Common operations on bin-valued fields.

**API** (new file `field_ops.py`):

```python
def normalize_field(
    field: NDArray[np.float64],
    *,
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """Normalize field to sum to 1 (probability distribution)."""

def clamp(
    field: NDArray[np.float64],
    *,
    lo: float = 0.0,
    hi: float = np.inf,
) -> NDArray[np.float64]:
    """Clamp field values to [lo, hi] range."""

def combine_fields(
    fields: Sequence[NDArray[np.float64]],
    weights: Sequence[float] | None = None,
    mode: Literal["mean", "max", "min"] = "mean",
) -> NDArray[np.float64]:
    """
    Combine multiple fields.

    Parameters
    ----------
    fields : Sequence[NDArray], each shape (n_bins,)
    weights : Sequence[float], optional
        Weights for mode='mean'. Must sum to 1.
    mode : {'mean', 'max', 'min'}

    Returns
    -------
    combined : NDArray[np.float64], shape (n_bins,)
    """

def divergence(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    *,
    kind: Literal["kl", "js", "cosine"] = "kl",
    eps: float = 1e-12,
) -> float:
    """
    Compute divergence between two probability distributions.

    Parameters
    ----------
    p, q : NDArray[np.float64], shape (n_bins,)
        Probability distributions (should sum to 1).
    kind : {'kl', 'js', 'cosine'}
        - 'kl': Kullback-Leibler divergence D_KL(p || q).
        - 'js': Jensen-Shannon divergence (symmetric).
        - 'cosine': Cosine distance (1 - cosine_similarity).
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    distance : float

    Notes
    -----
    KL divergence is not symmetric: D_KL(p||q) ≠ D_KL(q||p).
    JS divergence is symmetric and bounded: 0 ≤ D_JS ≤ 1.
    """
```

**Design**:

- Pure NumPy implementations
- Shape validation: all fields must be `(n_bins,)`
- Numerical stability: use `eps` to avoid division by zero

**Tests**:

- `divergence(p, p, kind='kl')` → 0.0
- `divergence(p, q, kind='js')` → symmetric
- `cosine` distance → bounded [0, 2]
- Normalization preserves field shape

---

### 11. Occupancy with Linear Time Interpolation (Advanced)

**Goal**: More accurate occupancy by splitting time across bins during boundary crossings.

**API** (add to `occupancy()` in P0.1):

```python
def occupancy(
    self,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    method: Literal["nearest", "linear"] = "nearest",  # NEW
    ...
) -> NDArray[np.float64]:
    """
    ...
    method : {'nearest', 'linear'}, default='nearest'
        - 'nearest': Assign entire Δt to starting bin.
        - 'linear': Split Δt proportionally when crossing bin boundaries.
          Only supported for RegularGridLayout.
    ...
    """
```

**Design** (grid-only):

1. For each segment `(t[i], pos[i])` → `(t[i+1], pos[i+1])`:
2. Find bin_start and bin_end
3. If same bin: assign all Δt to that bin
4. If different:
   - Compute ray-grid intersection (2D: line-segment/grid-edge crossings)
   - Split Δt proportionally by segment lengths
   - Accumulate fractions to each traversed bin

**Tests**:

- Diagonal trajectory across 4 grid cells → verify proportional split
- Compare `method='linear'` vs `'nearest'` → linear gives smoother maps
- Non-grid layout with `method='linear'` → raises NotImplementedError

---

## P3 - QoL & Robustness

### 12. Per-bin Region Membership

**Goal**: Vectorized region containment checks.

**API** (in `environment.py`):

```python
def region_membership(
    self,
    regions: Regions | None = None,
    *,
    include_boundary: bool = True,
) -> NDArray[np.bool_]:
    """
    Check which bins belong to which regions.

    Parameters
    ----------
    regions : Regions, optional
        Regions to test. If None, uses self.regions.
    include_boundary : bool, default=True
        If True, bins on region boundary count as inside (shapely.covers).
        If False, strict interior only (shapely.contains).

    Returns
    -------
    membership : NDArray[np.bool_], shape (n_bins, n_regions)
        membership[i, j] = True if bin i is in region j.

    Examples
    --------
    >>> membership = env.region_membership()
    >>> goal_bins = np.where(membership[:, env.regions.index('goal')])[0]
    """
```

**Design**:

- Extract bin centers
- For each region polygon: vectorized `shapely.covers(polygon, points)`
- Cache by `(env._version, hash(regions))`

**Tests**:

- Square region aligned to grid → exact membership pattern
- Point on boundary → included if `include_boundary=True`

---

### 13. Distance Transforms & Rings

**Goal**: Convenience wrappers around existing distance_field.

**API** (in `environment.py`):

```python
def distance_to(
    self,
    targets: Sequence[int] | str,
    *,
    metric: Literal["euclidean", "geodesic"] = "geodesic",
) -> NDArray[np.float64]:
    """
    Compute distance from each bin to target set.

    Parameters
    ----------
    targets : Sequence[int] or str
        Target bin indices, or region name.
    metric : {'euclidean', 'geodesic'}, default='geodesic'
        - 'euclidean': Straight-line distance.
        - 'geodesic': Graph distance respecting connectivity.

    Returns
    -------
    distances : NDArray[np.float64], shape (n_bins,)
        Distance from each bin to nearest target.

    Examples
    --------
    >>> # Distance to goal region
    >>> dist = env.distance_to('goal', metric='geodesic')
    >>>
    >>> # Distance to specific bins
    >>> dist = env.distance_to([10, 20, 30])
    """

def rings(
    self,
    center_bin: int,
    *,
    hops: int,
) -> list[NDArray[np.int32]]:
    """
    Compute k-hop neighborhoods (BFS layers).

    Parameters
    ----------
    center_bin : int
        Starting bin.
    hops : int
        Number of hop layers to compute.

    Returns
    -------
    rings : list[NDArray[np.int32]], length hops+1
        rings[k] = array of bins exactly k hops from center.
        rings[0] = [center_bin].

    Examples
    --------
    >>> rings = env.rings(center_bin=50, hops=3)
    >>> # Bins 2 hops away:
    >>> two_hop_neighbors = rings[2]
    """
```

**Design**:

- `distance_to()`:
  - If targets is str, map region → bin indices via membership
  - Call existing `distance_field(self.connectivity, sources=targets)`
- `rings()`: BFS by layers, collect nodes at each depth

**Tests**:

- 1D line graph: distances = index differences
- Rings on grid: symmetric patterns
- Region-based distance: correct multi-source behavior

---

### 14. Copy / Clone

**Goal**: Safe copying with cache invalidation.

**API** (in `environment.py`):

```python
def copy(
    self,
    *,
    deep: bool = True,
) -> "Environment":
    """
    Create copy of environment.

    Parameters
    ----------
    deep : bool, default=True
        If True, deep copy arrays and graph.
        If False, shallow copy (shares underlying data).

    Returns
    -------
    env_copy : Environment
        New environment instance. Transient caches (KDTree) are cleared.

    Examples
    --------
    >>> env2 = env.copy()
    >>> env2.regions.add('test', point=[10, 10])  # Doesn't affect env
    """
```

**Design**:

- Deep: `copy.deepcopy()` on `bin_centers`, `connectivity`, `regions`
- Shallow: share references
- Always reset `_kdtree_cache`, `_kernel_cache` to None
- Increment `_version` on copy

**Tests**:

- Modify copy → original unchanged (deep)
- Modify copy → original changed (shallow)
- Cache miss after copy

---

### 15. Deterministic KDTree & Distance Thresholds

**Goal**: Reproducible boundary decisions.

**API** (in `spatial.py`):

```python
def map_points_to_bins(
    points: NDArray[np.float64],
    environment: "Environment",
    *,
    tie_break: Literal["random", "lowest_index"] = "lowest_index",
    max_distance: float | None = None,  # NEW
    max_distance_factor: float | None = None,  # NEW
) -> NDArray[np.int32]:
    """
    ...
    max_distance : float, optional
        Absolute distance threshold. Points farther than this from nearest
        bin center → -1 (outside).
    max_distance_factor : float, optional
        Relative threshold as multiple of typical bin size.
        E.g., max_distance_factor=1.5 means points >1.5× bin_size away → outside.
    ...
    """
```

**Design**:

- Compute nearest bin via KDTree
- If `max_distance` or `max_distance_factor` set:
  - Query distances: `dist, idx = tree.query(points)`
  - Compute threshold: `thresh = max_distance or (max_distance_factor * median_bin_size)`
  - Mask: `idx[dist > thresh] = -1`
- Replace random subsampling with deterministic quantiles for threshold estimation

**Tests**:

- Repeated calls produce identical results
- Points just outside threshold → -1
- Points just inside threshold → valid bin index

---

### 16. Geodesic Cache Optimization

**Goal**: Speed up repeated distance queries (optional optimization of existing `distance_field`).

**API** (in `distance.py`):

```python
class GeodesicCache:
    """
    Cache for multi-source shortest path results.

    Auto-invalidates when environment version changes.
    """
    def __init__(self, environment: "Environment"):
        self.env = environment
        self._cache = {}
        self._version = environment._version

    def distance_from(
        self,
        sources: Sequence[int],
    ) -> NDArray[np.float64]:
        """
        Compute distances from sources, using cache if available.

        Returns
        -------
        distances : NDArray[np.float64], shape (n_bins,)
        """
```

**Design**:

- Key: `frozenset(sources)`
- Store: precomputed distance array
- On access: check `self._version == env._version`, invalidate if stale

**Tests**:

- Cache hit → faster than recompute
- Modify graph → cache invalidates
- Different source sets → independent cache entries

---

## Cross-Cutting Concerns

### Version-Based Cache Invalidation

**Design**:

```python
class Environment:
    def __init__(self, ...):
        self._version = 0
        self._kdtree_cache = None
        self._kernel_cache = {}  # keyed by (bandwidth, mode)
        self._membership_cache = {}

    def _invalidate_caches(self):
        """Call whenever bin_centers or connectivity changes."""
        self._version += 1
        self._kdtree_cache = None
        self._kernel_cache.clear()
        self._membership_cache.clear()
```

- Increment `_version` when:
  - `bin_centers` modified
  - `connectivity` modified
  - `subset()` creates new environment
- All cache lookups check version first

### Validation Enhancements

**Extend `validate_environment()`**:

```python
def validate_environment(
    env: Environment,
    *,
    strict: bool = False,
    check_graph_metadata: bool = True,
) -> None:
    """
    ...
    check_graph_metadata : bool, default=True
        Verify edge attributes are consistent:
        - vector[u,v] ≈ pos[v] - pos[u]
        - distance[u,v] ≈ ||vector[u,v]||
        - angle_2d matches vector (for 2D layouts)
    ...
    """
```

### Documentation

**New guide**: `docs/guides/spatial-analysis.md`

Sections:

1. Computing occupancy and rate maps
2. Trajectory analysis (sequences, transitions)
3. Distance fields and navigation
4. Field smoothing and visualization
5. On-edge semantics (covers vs contains)
6. Unit conventions and bin volumes

**Docstring additions**:

- Each new method includes "Supported layouts" section
- Explicit statement of on-edge behavior for geometric operations
- Examples showing typical neuroscience workflows

---

## File Organization

**Existing files**:

- `environment.py`: Public methods (thin wrappers)
- `spatial.py`: KDTree operations, occupancy, bin_sequence
- `regions/ops.py`: Region geometric operations

**New files**:

```
src/neurospatial/
├── kernels.py              # Diffusion kernel computation
│   ├── compute_diffusion_kernels()
│   ├── DiffusionKernelCache
│   └── suggest_bandwidth()
├── field_ops.py            # Field math utilities
│   ├── normalize_field()
│   ├── clamp()
│   ├── combine_fields()
│   └── divergence()
└── distance.py             # Enhanced distance operations
    ├── GeodesicCache
    └── distance field helpers
```

**Module organization**:

- `environment.py`: User-facing API, delegates to specialized modules
- `kernels.py`: All diffusion-related computation
- `spatial.py`: Point-to-bin mapping, occupancy, sequences
- `field_ops.py`: Pure NumPy field operations (no Environment dependency)
- `distance.py`: Graph distance algorithms and caching

---

## Acceptance Criteria

To declare Environment operations **feature-complete**, all of the following must be satisfied:

### P0 (Required for v0.2.0)

- ✅ `occupancy()` implemented with tests (nearest + optional kernel smoothing)
- ✅ `bin_sequence()` with run-length encoding
- ✅ `transitions()` with adjacency filtering
- ✅ `components()` and `reachable_from()`
- ✅ `compute_diffusion_kernels()` in `kernels.py` with full test suite
- ✅ `Environment.compute_kernel()` and `smooth()` wrappers
- ✅ All P0 functions work on all layout types (or raise clear `NotImplementedError`)
- ✅ At least one example script: occupancy → rate map → smoothing → transitions

### P1 (Target for v0.3.0)

- ✅ `rebin()` for grid coarsening
- ✅ `subset()` for cropping/masking
- ✅ All operations have "Supported layouts" documented

### P2 (Target for v0.4.0)

- ✅ `interpolate()` for continuous field evaluation
- ✅ Field math helpers (`field_ops.py` module)
- ✅ Linear occupancy method for regular grids

### P3 (Polish - ongoing)

- ✅ `region_membership()` vectorized
- ✅ `distance_to()` and `rings()` convenience wrappers
- ✅ `copy()` with cache invalidation
- ✅ Deterministic KDTree with distance thresholds
- ✅ Geodesic caching infrastructure

### Documentation & Testing

- ✅ New guide: `docs/guides/spatial-analysis.md`
- ✅ Each method has NumPy-style docstring with Examples section
- ✅ On-edge semantics explicitly stated (covers vs contains)
- ✅ Unit conventions documented (physical units, bin volumes)
- ✅ Edge cases tested: empty environment, single bin, disconnected graph
- ✅ Performance target: occupancy on 1M samples < 1 second

### Reproducibility & Code Quality

- ✅ KDTree operations deterministic by default (`tie_break="lowest_index"`)
- ✅ All methods use `@check_fitted` decorator
- ✅ Comprehensive input validation with diagnostic error messages
- ✅ Caches keyed by object identity (`id(self)`)
- ✅ All public methods have complete NumPy docstrings

---

## Testing Strategy

### Test Organization

All tests follow the existing structure in `tests/`:

- `tests/test_environment.py` - Core Environment operation tests
- `tests/test_kernels.py` - Diffusion kernel tests (new)
- `tests/test_field_ops.py` - Field math utilities (new)
- `tests/conftest.py` - Shared fixtures

### Fixtures and Parametrization

Use pytest parametrization to test across layout types:

```python
@pytest.mark.parametrize("layout_kind", [
    "RegularGrid",
    "Hexagonal",
    "Graph",
    "MaskedGrid",
    "ShapelyPolygon",
])
def test_occupancy_basic(layout_kind, sample_trajectory):
    """Test occupancy computation on all layout types."""
    env = create_test_environment(layout_kind)
    occ = env.occupancy(sample_trajectory.times, sample_trajectory.positions)
    assert occ.shape == (env.n_bins,)
    assert np.all(occ >= 0)
```

### Property-Based Tests

Use property tests for mathematical invariants:

```python
def test_occupancy_mass_conservation(env, trajectory):
    """Occupancy sum equals total valid time."""
    occ = env.occupancy(trajectory.times, trajectory.positions)
    expected_time = trajectory.times[-1] - trajectory.times[0]
    assert np.isclose(occ.sum(), expected_time, rtol=1e-6)

def test_kernel_normalization(env, bandwidth):
    """Kernel columns sum/integrate to 1."""
    kernel = env.compute_kernel(bandwidth, mode='transition')
    assert np.allclose(kernel.sum(axis=0), 1.0)
```

### Edge Cases

Every operation must handle:

- Empty inputs (zero-length arrays)
- Single sample
- All samples outside environment
- Disconnected graph components
- Mismatched array lengths

### Performance Benchmarks

Add benchmarks for acceptance criteria:

```python
def test_occupancy_performance(benchmark, large_trajectory):
    """Occupancy on 1M samples completes in <1s."""
    env = create_large_environment(n_bins=1000)
    result = benchmark(env.occupancy,
                       large_trajectory.times,
                       large_trajectory.positions)
    assert benchmark.stats['mean'] < 1.0  # seconds
```

---

## Implementation Roadmap

### Phase 1: Kernel Infrastructure (Week 1)

**Goal**: Establish diffusion kernel foundation for all smoothing operations.

**Tasks**:

1. Create `src/neurospatial/kernels.py`
   - Implement `compute_diffusion_kernels()` with NumPy docstrings
   - Add `_assign_gaussian_weights_from_distance()` helper
   - Implement object identity-based caching
2. Add `Environment.compute_kernel()` method in `environment.py`
3. Create `tests/test_kernels.py` with:
   - Symmetry tests
   - Normalization tests (transition vs density modes)
   - Mass conservation tests
   - Cache behavior tests
4. Add performance warnings to docstrings (O(n³) complexity)

**Deliverable**: Kernel infrastructure ready for use in occupancy/smoothing.

---

### Phase 2: Core Analysis Operations (Weeks 2-3)

**Goal**: Implement P0 operations (occupancy, sequences, transitions, connectivity).

**Tasks**:

1. **Occupancy** (`environment.py`)
   - Add `@check_fitted` decorator
   - Implement input validation
   - Map points to bins using `spatial.map_points_to_bins`
   - Time accumulation logic
   - Optional kernel smoothing
   - Tests: L-shaped path, speed filtering, smoothing

2. **Bin Sequence** (`environment.py`)
   - Implement with run-length encoding
   - Handle `outside_value=None` for dropping samples
   - Tests: known trajectories, deduplication, runs

3. **Transitions** (`environment.py`)
   - Adjacency filtering with `allow_teleports` parameter
   - Sparse CSR matrix output
   - Normalization option
   - Tests: 1D track, grid navigation, teleport filtering

4. **Connectivity** (`environment.py`)
   - `components()` using NetworkX
   - `reachable_from()` with BFS/Dijkstra
   - Tests: disconnected graphs, neighborhoods

**Deliverable**: P0 operations complete with full test coverage.

---

### Phase 3: Smoothing & Masking (Week 4)

**Goal**: Implement P1 operations (smooth, rebin, subset).

**Tasks**:

1. **Smooth** (`environment.py`)
   - Thin wrapper around `compute_kernel`
   - Field shape validation
   - Tests: impulse spreading, mass conservation

2. **Rebin** (`environment.py`)
   - Grid-only operation with clear error for non-grids
   - Conservative aggregation (sum/mean)
   - Tests: constant field preservation, bin positions

3. **Subset** (`environment.py`)
   - Bins/regions/polygon selection
   - Subgraph induction with node renumbering
   - Tests: cropping, polygon masks, connectivity preservation

**Deliverable**: P1 operations with layout compatibility checks.

---

### Phase 4: Interpolation & Field Utilities (Week 5)

**Goal**: Implement P2 operations (interpolation, field math).

**Tasks**:

1. **Interpolate** (`environment.py`)
   - Nearest neighbor (all layouts)
   - Linear mode (grids only)
   - Tests: known plane fields, out-of-bounds

2. **Field Operations** (`field_ops.py` - new module)
   - `normalize_field`, `clamp`, `combine_fields`, `divergence`
   - Pure NumPy implementations
   - Tests: KL/JS identities, normalization

3. **Linear Occupancy** (enhancement to P0.1)
   - Ray-grid intersection for grids
   - Proportional time splitting
   - Tests: diagonal trajectories

**Deliverable**: P2 operations and field utilities.

---

### Phase 5: Utilities & Documentation (Week 6)

**Goal**: Complete P3 operations and documentation.

**Tasks**:

1. **P3 Operations** (`environment.py`)
   - `region_membership()` - vectorized Shapely operations
   - `distance_to()` - wrapper around `distance_field`
   - `rings()` - BFS layers
   - `copy()` - deep/shallow with cache reset

2. **Documentation**
   - Create `docs/guides/spatial-analysis.md`
   - Add example scripts to `examples/` directory
   - Performance benchmarks

3. **Polish**
   - Deterministic KDTree improvements in `spatial.py`
   - Public API exports in `__init__.py`
   - Final integration tests

**Deliverable**: Feature-complete environment operations with documentation.

---

## Best Practices & Guidelines

### API Design Patterns

1. **Fitted State Enforcement**
   - All methods use `@check_fitted` decorator
   - Prevents confusing errors from calling methods on uninitialized environments
   - Consistent with existing `bin_at()`, `contains()`, etc.

2. **Input Validation**
   - Convert inputs to numpy arrays with explicit dtype
   - Validate array shapes and lengths match
   - Handle empty arrays gracefully (return appropriate zero-filled results)
   - Provide diagnostic error messages showing actual vs expected values

3. **Keyword-Only Parameters**
   - Use `*` to enforce keyword arguments for optional parameters
   - Improves code clarity: `env.occupancy(times, positions, kernel_bandwidth=5.0)`
   - Prevents positional argument mistakes

4. **Type Hints**
   - Use `NDArray[np.float64]` for typed arrays
   - Use `Literal` for string enums: `Literal["transition", "density"]`
   - Allow `| None` for optional parameters

5. **NumPy Docstring Format**
   - Use `Parameters` section (not `Args`)
   - Include array shapes in type annotations: `shape (n_samples, n_dims)`
   - Provide `Examples` section with doctests
   - Add `Notes` for implementation details
   - Add `Raises` for validation errors

### Design Rationale

1. **Diffusion Kernels as Foundation**
   - More principled than ad-hoc Gaussian or iterative methods
   - Mathematically grounded in heat equation
   - Works uniformly across all layout types (grids, graphs, meshes)
   - Handles volume correction elegantly via `mode` parameter

2. **Object Identity Caching**
   - Cache keyed by `id(self)` instead of version tracking
   - Simpler implementation, aligns with environment immutability
   - Caches automatically scoped to environment lifetime
   - Example: `cache_key = (id(self), bandwidth, mode)`

3. **Grid-Only Operations Clearly Marked**
   - Operations like `rebin()` and `interpolate(linear)` only work for rectangular grids
   - Raise `NotImplementedError` with clear message and suggestion for alternatives
   - Docstrings include "Supported layouts" section

4. **Conservative Defaults**
   - `allow_teleports=False` - Filters non-adjacent transitions to catch tracking errors
   - `tie_break="lowest_index"` - Deterministic, no randomness
   - `max_gap=0.5` - Reasonable default for neuroscience data at typical sampling rates
   - `mode="density"` - Respects bin volumes for continuous fields

5. **Time Allocation Strategy**
   - Intervals assigned to starting bin by default (simple, fast)
   - Linear mode available in P2 for higher accuracy on grids
   - Explicit in documentation to avoid confusion

6. **Regions Handling in subset()**
   - Regions dropped when creating subsets
   - Simplifies implementation (no coordinate transformation logic)
   - Users can easily re-add regions to cropped environment with updated coordinates

### Future Extensions

Beyond feature-complete scope (P0-P3):

- **Adaptive bandwidth selection** - Automatically choose sigma based on data density
- **Temporal binning** - Convert spike trains to time-binned representations
- **Cross-environment alignment** - Use transition matrices to align environments
- **GPU acceleration** - CUDA kernels for large matrix exponentials
- **Sparse approximations** - Truncated kernels for huge environments (>10k bins)
- **Batch operations** - Process multiple sessions/trials efficiently

---

## Layout Compatibility Matrix

This table shows which operations are supported on which layout types. ✓ = fully supported, ⚠️ = partial support, ✗ = not supported.

| Operation | RegularGrid | Hexagonal | Graph (1D) | Triangular | MaskedGrid | Polygon | ImageMask |
|-----------|-------------|-----------|------------|------------|------------|---------|-----------|
| **P0: Core Analysis** | | | | | | | |
| `occupancy()` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `bin_sequence()` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `transitions()` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `components()` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `reachable_from()` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `compute_kernel()` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **P1: Smoothing/Masking** | | | | | | | |
| `smooth()` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `rebin()` | ✓ | ✗ | ✗ | ✗ | ⚠️ | ⚠️ | ⚠️ |
| `subset()` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **P2: Interpolation** | | | | | | | |
| `interpolate(nearest)` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `interpolate(linear)` | ✓ | ✗ | ✗ | ⚠️ | ✓ | ✓ | ✓ |
| `occupancy(linear)` | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| **P3: Utilities** | | | | | | | |
| `region_membership()` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `distance_to()` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `rings()` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `copy()` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**Notes**:

- **RegularGrid**: Full support for all operations
- **Hexagonal/Triangular**: No rectangular grid operations (rebin, linear interpolation)
- **Graph (1D)**: Track-based; no 2D grid operations but has `to_linear()` method
- **MaskedGrid/Polygon/ImageMask**: Support grid operations where underlying grid is regular; rebin ⚠️ requires rectangular active region
- **All layouts**: Support graph-based operations (kernel smoothing, connectivity, distances)

**Error Handling**:

- Unsupported operations raise `NotImplementedError` with clear message
- Partial support (⚠️) may raise errors for certain configurations
- All errors include suggestion for alternative approaches when available

---

**End of Plan**
