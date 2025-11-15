# Implementation Plan for Code Review Recommendations

**Created**: 2025-11-15
**Based on**: COMPREHENSIVE_CODE_REVIEW.md
**Target Version**: 0.3.0

---

## Overview

This plan addresses the actionable recommendations from the comprehensive code review, organized into phases with clear dependencies and test coverage requirements.

**Key Principles**:

1. **Test Coverage First**: Before replacing custom implementations with scipy/networkx, ensure comprehensive test coverage
2. **Incremental Changes**: Each phase is independently testable
3. **UX Focus**: Simplify API for neuroscientists with intermediate Python skills

---

## Phase 1: Test Coverage Audit & Enhancement

**Goal**: Ensure sufficient test coverage before refactoring critical algorithms

**Priority**: 🔴 CRITICAL - Must complete before Phase 2

### Tasks

#### 1.1 Audit Test Coverage for Graph Algorithms

- [ ] Run coverage report for `distance.py`

  ```bash
  uv run pytest tests/ --cov=src/neurospatial/distance.py --cov-report=html
  ```

- [ ] Verify coverage for:
  - [ ] `geodesic_distance_matrix()` - all code paths
  - [ ] `euclidean_distance_matrix()` - edge cases (0 points, 1 point, many points)
  - [ ] `distance_field()` - both metrics (geodesic, euclidean), with/without cutoff
  - [ ] `pairwise_distances()` - disconnected graphs, empty graphs
  - [ ] `neighbors_within()` - both metrics, include_center variants

**Acceptance Criteria**: ≥95% line coverage on all functions in `distance.py`

#### 1.2 Audit Test Coverage for Differential Operators

- [ ] Run coverage report for `differential.py`

  ```bash
  uv run pytest tests/ --cov=src/neurospatial/differential.py --cov-report=html
  ```

- [ ] Verify coverage for:
  - [ ] `compute_differential_operator()` - various graph topologies
  - [ ] `gradient()` - 1D, 2D, 3D environments
  - [ ] `divergence()` - all dimensions

**Acceptance Criteria**: ≥95% line coverage on `differential.py`

#### 1.3 Audit Test Coverage for Kernels

- [ ] Run coverage report for `kernels.py`

  ```bash
  uv run pytest tests/ --cov=src/neurospatial/kernels.py --cov-report=html
  ```

- [ ] Verify coverage for:
  - [ ] `compute_diffusion_kernels()` - transition vs density modes
  - [ ] Kernel caching mechanism
  - [ ] Edge cases (bandwidth=0, very large bandwidth)

**Acceptance Criteria**: ≥95% line coverage on `kernels.py`

#### 1.4 Audit Test Coverage for Place Field Detection

- [ ] Run coverage report for `metrics/place_fields.py`

  ```bash
  uv run pytest tests/metrics/test_place_fields.py --cov=src/neurospatial/metrics/place_fields.py --cov-report=html
  ```

- [ ] Verify coverage for:
  - [ ] `detect_place_fields()` - with/without subfields, various thresholds
  - [ ] Connected component detection logic
  - [ ] Edge cases (no fields, interneuron exclusion, single-bin fields)

**Acceptance Criteria**: ≥95% line coverage on `detect_place_fields()`

#### 1.5 Add Missing Tests (if needed)

- [ ] Create regression tests with pinned expected values for:
  - [ ] Distance matrix on known small graphs (5-10 nodes)
  - [ ] Laplacian eigenvalues on regular grids (analytical solution)
  - [ ] Place field detection on synthetic Gaussian fields
- [ ] Add property-based tests (hypothesis) for:
  - [ ] Distance matrix symmetry
  - [ ] Laplacian properties (row sums, negative eigenvalues)

**Deliverable**: Test suite with ≥95% coverage on all functions to be refactored

**Estimated Effort**: 1-2 days

---

## Phase 2: Replace Custom Implementations with scipy/networkx

**Goal**: Leverage optimized standard library implementations for significant performance gains

**Priority**: 🔴 HIGH - Significant performance impact (10-100× speedup)

**Dependencies**: Phase 1 complete (test coverage verified)

### Tasks

#### 2.1 Replace geodesic_distance_matrix with scipy.sparse.csgraph

- [ ] **File**: `src/neurospatial/distance.py` (lines 32-62)
- [ ] **Implementation**:

  ```python
  from scipy.sparse.csgraph import shortest_path

  def geodesic_distance_matrix(
      G: nx.Graph,
      n_states: int,
      weight: str = "distance",
  ) -> NDArray[np.float64]:
      """Compute geodesic (shortest-path) distance matrix on a graph.

      Uses scipy.sparse.csgraph.shortest_path for optimized performance.
      """
      if G.number_of_nodes() == 0:
          return np.empty((0, 0), dtype=np.float64)

      # Convert to scipy sparse adjacency matrix
      adj = nx.to_scipy_sparse_array(G, weight=weight, format='csr')

      # Compute all-pairs shortest paths using Dijkstra
      dist_matrix = shortest_path(
          adj,
          directed=False,
          method='D',  # Dijkstra's algorithm
          return_predecessors=False
      )

      return np.asarray(dist_matrix, dtype=np.float64)
  ```

- [ ] **Testing**:
  - [ ] Run existing test suite: `uv run pytest tests/test_distance.py -v`
  - [ ] Verify numerical equivalence on known graphs
  - [ ] Benchmark performance improvement (expect 10-100× speedup)
- [ ] **Documentation**:
  - [ ] Update docstring to mention scipy implementation
  - [ ] Add "Performance" note about optimization

**Acceptance Criteria**:

- All tests pass with identical numerical results (within `np.allclose` tolerance)
- Performance benchmark shows ≥10× improvement on 1000+ node graphs

#### 2.2 Replace custom Laplacian with scipy.sparse.csgraph.laplacian

- [ ] **File**: `src/neurospatial/differential.py`
- [ ] **Investigation**:
  - [ ] Compare current `compute_differential_operator()` output with `scipy.sparse.csgraph.laplacian()`
  - [ ] Verify normalization mode matches (symmetric normalized Laplacian)
  - [ ] Check if additional processing needed (sign convention, scaling)
- [ ] **Implementation** (if compatible):

  ```python
  from scipy.sparse.csgraph import laplacian

  def compute_differential_operator(
      G: nx.Graph,
      *,
      weight: str = "distance",
      normalized: bool = True,
  ) -> sparse.csr_matrix:
      """Compute graph Laplacian using scipy.sparse.csgraph.

      Parameters
      ----------
      G : nx.Graph
          NetworkX graph representing spatial connectivity.
      weight : str, default="distance"
          Edge attribute to use as weight.
      normalized : bool, default=True
          If True, return normalized Laplacian (L_sym = D^(-1/2) L D^(-1/2)).
          If False, return unnormalized Laplacian (L = D - A).

      Returns
      -------
      L : scipy.sparse.csr_matrix
          Graph Laplacian matrix.
      """
      adj = nx.to_scipy_sparse_array(G, weight=weight, format='csr')
      L = laplacian(adj, normed=normalized, return_diag=False)
      return L
  ```

- [ ] **Testing**:
  - [ ] Verify eigenvalue properties (all non-positive, 0 eigenvalue for connected graph)
  - [ ] Check gradient/divergence operators still work
  - [ ] Run full test suite: `uv run pytest tests/test_differential.py -v`

**Acceptance Criteria**:

- All differential operator tests pass
- Eigenvalue properties preserved
- Numerical equivalence with current implementation

#### 2.3 Replace connected components with scipy.ndimage.label

- [ ] **File**: `src/neurospatial/metrics/place_fields.py`
- [ ] **Investigation**:
  - [ ] Determine when `scipy.ndimage.label` is applicable (grid-based environments only)
  - [ ] Identify fallback strategy for non-grid graphs
- [ ] **Implementation Strategy**:

  ```python
  def detect_place_fields(
      firing_rate: NDArray[np.float64],
      env: Environment,
      *,
      threshold: float = 0.2,
      min_size: int | None = None,
      max_mean_rate: float = 10.0,
      detect_subfields: bool = True,
  ) -> list[NDArray[np.int64]]:
      # ... existing validation ...

      # Check if environment is grid-based (has grid_shape attribute)
      if hasattr(env.layout, 'grid_shape') and env.layout.grid_shape is not None:
          # Fast path: use scipy.ndimage.label for grid environments
          fields = _detect_fields_grid(firing_rate, env, threshold, min_size, detect_subfields)
      else:
          # Fallback: use graph-based flood-fill for irregular graphs
          fields = _detect_fields_graph(firing_rate, env, threshold, min_size, detect_subfields)

      return fields

  def _detect_fields_grid(
      firing_rate: NDArray[np.float64],
      env: Environment,
      threshold: float,
      min_size: int,
      detect_subfields: bool,
  ) -> list[NDArray[np.int64]]:
      """Fast grid-based field detection using scipy.ndimage.label."""
      from scipy.ndimage import label

      # Reshape to grid
      grid_shape = env.layout.grid_shape
      rate_grid = np.full(grid_shape, np.nan, dtype=np.float64)
      # ... populate grid from firing_rate using active_mask ...

      # Threshold and label connected components
      binary = rate_grid > (threshold * np.nanmax(rate_grid))
      labeled, n_components = label(binary)

      # Extract fields and convert back to flat bin indices
      fields = []
      for component_id in range(1, n_components + 1):
          field_bins = np.where(labeled.ravel() == component_id)[0]
          # Filter by size, etc.
          fields.append(field_bins)

      return fields
  ```

- [ ] **Testing**:
  - [ ] Test on regular grids (should use scipy.ndimage path)
  - [ ] Test on irregular graphs (should use fallback path)
  - [ ] Verify identical results for grid environments
  - [ ] Benchmark performance improvement on grids

**Acceptance Criteria**:

- All place field tests pass
- Grid environments show ≥5× speedup
- Irregular graphs still work (fallback path)

**Estimated Effort**: 3-4 days

---

## Phase 3: API Simplification

**Goal**: Reduce confusion for intermediate users by consolidating APIs

**Priority**: 🔴 HIGH - Major UX improvement

**Dependencies**: None (independent of Phase 1-2)

### Tasks

#### 3.1 Consolidate bin_at() and map_points_to_bins()

- [ ] **Current State**:
  - `Environment.bin_at()` - delegates to layout engine, single point
  - `map_points_to_bins()` - KDTree-based, batch, tie-breaking, caching
- [ ] **Decision**: Make `bin_at()` handle everything (user's preference)
- [ ] **Implementation Plan**:

  **Option A: Enhance bin_at() to handle batch operations**

  ```python
  # In src/neurospatial/environment/queries.py

  @check_fitted
  def bin_at(
      self: "Environment",
      points: NDArray[np.float64],
      *,
      tie_break: TieBreakStrategy | Literal["lowest_index", "closest_center"] = TieBreakStrategy.LOWEST_INDEX,
      return_dist: bool = False,
      max_distance: float | None = None,
      max_distance_factor: float | None = None,
  ) -> NDArray[np.int64] | tuple[NDArray[np.int64], NDArray[np.float64]]:
      """Map point(s) to bin index/indices.

      Handles both single points and batches efficiently with automatic
      KDTree caching for repeated queries.

      Parameters
      ----------
      points : NDArray[np.float64], shape (n_dims,) or (n_points, n_dims)
          Point coordinates. Can be:
          - Single point: shape (n_dims,)
          - Multiple points: shape (n_points, n_dims)
      tie_break : TieBreakStrategy or {"lowest_index", "closest_center"}, default="lowest_index"
          Strategy for resolving ties when point is equidistant from multiple bins.
      return_dist : bool, default=False
          If True, also return distance(s) to assigned bin center(s).
      max_distance : float, optional
          Absolute distance threshold. Points farther than this are marked as outside (-1).
      max_distance_factor : float, optional
          Relative distance threshold as multiple of typical bin spacing.

      Returns
      -------
      bin_index : int or NDArray[np.int64]
          Bin index for single point (int) or indices for multiple points (array).
          Value of -1 indicates point is outside environment.
      distance : float or NDArray[np.float64], optional
          Distance to assigned bin center (only if return_dist=True).

      Examples
      --------
      Single point:
      >>> bin_idx = env.bin_at([10.0, 20.0])
      >>> print(bin_idx)
      42

      Multiple points (batch):
      >>> points = np.array([[10.0, 20.0], [30.0, 40.0]])
      >>> bin_indices = env.bin_at(points)
      >>> print(bin_indices)
      [42 89]

      With distances:
      >>> bin_idx, dist = env.bin_at([10.0, 20.0], return_dist=True)
      >>> print(f"Bin {bin_idx}, distance {dist:.2f}")
      Bin 42, distance 0.23
      """
      # Normalize input shape
      points_arr = np.atleast_2d(points)
      is_single_point = points.ndim == 1

      # Delegate to map_points_to_bins for efficiency
      from neurospatial.spatial import map_points_to_bins

      result = map_points_to_bins(
          points_arr,
          self,
          tie_break=tie_break,
          return_dist=return_dist,
          max_distance=max_distance,
          max_distance_factor=max_distance_factor,
      )

      # Convert back to scalar for single point
      if is_single_point:
          if return_dist:
              bin_idx, dist = result
              return (int(bin_idx[0]), float(dist[0]))
          else:
              return int(result[0])

      return result
  ```

- [ ] **Deprecation Strategy**:
  - [ ] Keep `map_points_to_bins()` as public API with deprecation warning:

    ```python
    def map_points_to_bins(...):
        """DEPRECATED: Use Environment.bin_at() instead.

        This function will be removed in version 0.4.0.
        Use env.bin_at(points, ...) for the same functionality.
        """
        import warnings
        warnings.warn(
            "map_points_to_bins() is deprecated and will be removed in v0.4.0. "
            "Use Environment.bin_at() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Delegate to implementation
        ...
    ```

  - [ ] Update all internal uses to call `env.bin_at()` instead
  - [ ] Update documentation and examples

- [ ] **Testing**:
  - [ ] Test single point input (backward compatibility)
  - [ ] Test batch input (new functionality)
  - [ ] Test tie-breaking strategies
  - [ ] Test return_dist parameter
  - [ ] Test max_distance parameters
  - [ ] Verify deprecation warning fires for `map_points_to_bins()`

- [ ] **Documentation Updates**:
  - [ ] Update `bin_at()` docstring with batch examples
  - [ ] Add migration guide in release notes
  - [ ] Update all examples in docstrings to use `env.bin_at()`

**Acceptance Criteria**:

- `bin_at()` handles both single and batch inputs efficiently
- All existing functionality preserved
- Deprecation warning clear and actionable
- Documentation comprehensive

**Estimated Effort**: 2-3 days

---

## Phase 4: Modern Python Features & UX Improvements

**Goal**: Modernize codebase and improve discoverability

**Priority**: 🟡 MEDIUM - Quality of life improvements

**Dependencies**: None (independent)

### Tasks

#### 4.1 Add pathlib.Path Support to I/O Functions

- [ ] **Files**: `src/neurospatial/io.py`, `src/neurospatial/environment/serialization.py`
- [ ] **Implementation**:

  ```python
  from pathlib import Path
  from typing import Union

  PathLike = Union[str, Path]

  def to_file(
      env: Environment,
      path: PathLike,
      *,
      overwrite: bool = False,
  ) -> None:
      """Save environment to files.

      Parameters
      ----------
      path : str or pathlib.Path
          Base path for output files (without extension).
          Creates two files: {path}.json and {path}.npz
      """
      path = Path(path)  # Convert str to Path

      json_path = path.with_suffix('.json')
      npz_path = path.with_suffix('.npz')

      # ... rest of implementation ...
  ```

- [ ] **Files to Update**:
  - [ ] `to_file()`, `from_file()` in `io.py`
  - [ ] `Environment.to_file()`, `Environment.from_file()` in `serialization.py`
  - [ ] Any region I/O functions
- [ ] **Testing**:
  - [ ] Test with `str` paths (backward compatibility)
  - [ ] Test with `pathlib.Path` objects
  - [ ] Test with relative and absolute paths
- [ ] **Type Hints**:
  - [ ] Update type hints to `str | Path` (Python 3.10+)
  - [ ] Or use `Union[str, Path]` for Python 3.9 compatibility

**Acceptance Criteria**: All I/O functions accept both `str` and `Path`

#### 4.2 Add Custom EnvironmentNotFittedError Exception

- [ ] **File**: `src/neurospatial/environment/decorators.py`
- [ ] **Implementation**:

  ```python
  class EnvironmentNotFittedError(RuntimeError):
      """Raised when an operation requires a fitted Environment.

      Environment instances must be created using factory methods
      (from_samples, from_polygon, from_graph, etc.) rather than
      direct instantiation.

      Examples
      --------
      Correct usage:
      >>> env = Environment.from_samples(positions, bin_size=2.0)  # ✓
      >>> env.bin_at([10.0, 20.0])  # Works

      Incorrect usage:
      >>> env = Environment()  # ✗ Not fitted
      >>> env.bin_at([10.0, 20.0])  # Raises EnvironmentNotFittedError
      """

      def __init__(self, method_name: str | None = None):
          if method_name:
              message = (
                  f"Cannot call {method_name}() on unfitted Environment. "
                  f"Use a factory method to create the environment:\n"
                  f"  - Environment.from_samples(positions, bin_size=...)\n"
                  f"  - Environment.from_polygon(polygon, bin_size=...)\n"
                  f"  - Environment.from_graph(graph, edge_spacing=...)\n"
                  f"See documentation for all factory methods."
              )
          else:
              message = (
                  "Environment must be fitted before use. "
                  "Use Environment.from_samples() or another factory method."
              )
          super().__init__(message)

  # Update check_fitted decorator
  def check_fitted(method):
      @functools.wraps(method)
      def wrapper(self, *args, **kwargs):
          if not self._is_fitted:
              raise EnvironmentNotFittedError(method.__name__)
          return method(self, *args, **kwargs)
      return wrapper
  ```

- [ ] **Export**: Add to `__init__.py`
- [ ] **Testing**: Test error message clarity

**Acceptance Criteria**: Clear, actionable error messages for unfitted environments

#### 4.3 Add Environment.info() Method

- [ ] **File**: `src/neurospatial/environment/core.py`
- [ ] **Implementation**:

  ```python
  @check_fitted
  def info(self) -> str:
      """Return human-readable environment summary.

      Displays key properties including bin count, dimensions, spatial extent,
      layout type, regions, and cache status. Useful for debugging and
      understanding environment configuration.

      Returns
      -------
      str
          Multi-line formatted summary of environment properties.

      Examples
      --------
      >>> env = Environment.from_samples(positions, bin_size=2.0, units='cm')
      >>> env.regions.add('goal', point=[50, 50])
      >>> print(env.info())
      Environment: my_maze
        Bins: 1,024
        Dimensions: 2D
        Extent: [0.0, 100.0] × [0.0, 100.0] cm
        Bin size: ~2.0 cm (mean)
        Layout: RegularGridLayout
        Connectivity: 4-connected (avg degree: 3.74)
        Regions: 1 defined (goal)
        Cache: KDTree (built), kernels (2 cached)
      """
      # Compute average bin size
      bin_sizes = self.bin_sizes()
      mean_bin_size = np.mean(bin_sizes)

      # Compute extent
      extent_str = " × ".join(
          f"[{self.dimension_ranges[i][0]:.1f}, {self.dimension_ranges[i][1]:.1f}]"
          for i in range(self.n_dims)
      )
      if self.units:
          extent_str += f" {self.units}"

      # Connectivity info
      avg_degree = 2 * self.connectivity.number_of_edges() / self.n_bins if self.n_bins > 0 else 0

      # Determine connectivity type
      if hasattr(self.layout, '_build_params_used'):
          params = self.layout._build_params_used
          connect_diag = params.get('connect_diagonal_neighbors', False)
          if self.n_dims == 2:
              conn_type = "8-connected" if connect_diag else "4-connected"
          else:
              conn_type = f"{int(avg_degree)}-connected"
      else:
          conn_type = f"{avg_degree:.1f} avg degree"

      # Cache status
      cache_parts = []
      if hasattr(self, '_kdtree_cache') and self._kdtree_cache is not None:
          cache_parts.append("KDTree (built)")
      if hasattr(self, '_kernel_cache') and len(self._kernel_cache) > 0:
          cache_parts.append(f"kernels ({len(self._kernel_cache)} cached)")
      cache_str = ", ".join(cache_parts) if cache_parts else "empty"

      # Build summary
      lines = [
          f"Environment: {self.name or 'unnamed'}",
          f"  Bins: {self.n_bins:,}",
          f"  Dimensions: {self.n_dims}D",
          f"  Extent: {extent_str}",
          f"  Bin size: ~{mean_bin_size:.1f} {self.units or '(no units)'} (mean)",
          f"  Layout: {self.layout.__class__.__name__}",
          f"  Connectivity: {conn_type} (avg degree: {avg_degree:.2f})",
      ]

      if len(self.regions) > 0:
          region_names = ", ".join(list(self.regions.keys())[:3])
          if len(self.regions) > 3:
              region_names += f", ... ({len(self.regions) - 3} more)"
          lines.append(f"  Regions: {len(self.regions)} defined ({region_names})")
      else:
          lines.append(f"  Regions: none defined")

      lines.append(f"  Cache: {cache_str}")

      return "\n".join(lines)
  ```

- [ ] **Testing**:
  - [ ] Test with different layout types
  - [ ] Test with/without regions
  - [ ] Test with/without cache
  - [ ] Test with/without units
- [ ] **Documentation**: Add examples to docstring

**Acceptance Criteria**: `env.info()` provides clear, formatted summary

#### 4.4 Add explain_connectivity() Method

- [ ] **File**: `src/neurospatial/environment/metrics.py` (or new `inspection.py` mixin)
- [ ] **Implementation**:

  ```python
  @check_fitted
  def explain_connectivity(self) -> str:
      """Explain how the connectivity graph is constructed.

      Provides human-readable description of the graph structure,
      neighbor relationships, edge weights, and topology.

      Returns
      -------
      str
          Multi-line explanation of connectivity graph.

      Examples
      --------
      >>> env = Environment.from_samples(positions, bin_size=2.0)
      >>> print(env.explain_connectivity())
      Connectivity Graph Structure:
        Type: Undirected graph
        Nodes: 1,024 bins
        Edges: 3,824 connections
        Average degree: 3.74 neighbors per bin

      Edge Weights:
        Attribute: 'distance'
        Type: Euclidean distance between bin centers
        Range: [2.0, 2.83] cm (4-connected grid)

      Topology:
        Grid type: 4-connected (up, down, left, right)
        Diagonal neighbors: No
        Boundary handling: Open boundaries (edge bins have fewer neighbors)

      Example:
        Bin 42 connects to bins: [41, 43, 10, 74]
        Edge weights: [2.0, 2.0, 2.0, 2.0] cm
      """
      G = self.connectivity

      # Basic stats
      n_nodes = G.number_of_nodes()
      n_edges = G.number_of_edges()
      avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0

      # Edge weight info
      if n_edges > 0:
          edge_weights = [G.edges[u, v].get('distance', 0) for u, v in G.edges()]
          weight_min = min(edge_weights)
          weight_max = max(edge_weights)
          weight_mean = sum(edge_weights) / len(edge_weights)

      # Determine topology type
      if hasattr(self.layout, '_layout_type_tag'):
          layout_type = self.layout._layout_type_tag
      else:
          layout_type = self.layout.__class__.__name__

      # Get build parameters if available
      if hasattr(self.layout, '_build_params_used'):
          params = self.layout._build_params_used
          connect_diag = params.get('connect_diagonal_neighbors', False)
      else:
          connect_diag = None

      # Determine connectivity pattern
      if self.n_dims == 2 and connect_diag is not None:
          if connect_diag:
              conn_pattern = "8-connected (up, down, left, right, and diagonals)"
          else:
              conn_pattern = "4-connected (up, down, left, right only)"
      else:
          conn_pattern = f"Average {avg_degree:.1f} neighbors per bin"

      # Example bin (middle of environment)
      example_bin = self.n_bins // 2 if self.n_bins > 0 else 0
      example_neighbors = list(G.neighbors(example_bin))[:5]

      # Build explanation
      lines = [
          "Connectivity Graph Structure:",
          f"  Type: {'Directed' if G.is_directed() else 'Undirected'} graph",
          f"  Nodes: {n_nodes:,} bins",
          f"  Edges: {n_edges:,} connections",
          f"  Average degree: {avg_degree:.2f} neighbors per bin",
          "",
      ]

      if n_edges > 0:
          lines.extend([
              "Edge Weights:",
              f"  Attribute: 'distance'",
              f"  Type: Euclidean distance between bin centers",
              f"  Range: [{weight_min:.2f}, {weight_max:.2f}] {self.units or '(no units)'}",
              f"  Mean: {weight_mean:.2f} {self.units or '(no units)'}",
              "",
          ])

      lines.extend([
          "Topology:",
          f"  Layout: {layout_type}",
          f"  Pattern: {conn_pattern}",
      ])

      if connect_diag is not None:
          lines.append(f"  Diagonal neighbors: {'Yes' if connect_diag else 'No'}")

      lines.append(f"  Boundary handling: Open boundaries (edge bins have fewer neighbors)")

      if example_neighbors:
          example_dists = [G.edges[example_bin, n]['distance'] for n in example_neighbors]
          lines.extend([
              "",
              f"Example (Bin {example_bin}):",
              f"  Neighbors: {example_neighbors[:5]}",
              f"  Distances: {[f'{d:.2f}' for d in example_dists]} {self.units or '(no units)'}",
          ])

      return "\n".join(lines)
  ```

- [ ] **Testing**: Test with various layout types

**Acceptance Criteria**: Clear explanation of graph connectivity

#### 4.5 Add **slots** to Environment Dataclass (Python 3.10+)

- [ ] **File**: `src/neurospatial/environment/core.py`
- [ ] **Check Python Version Requirements**:
  - Current minimum: Python 3.10 (check `pyproject.toml`)
  - `dataclass(slots=True)` requires Python 3.10+
- [ ] **Implementation**:

  ```python
  @dataclass(slots=True)  # Python 3.10+
  class Environment(
      EnvironmentFactories,
      # ... other mixins ...
  ):
      """Environment with slots for memory efficiency."""
      # All fields here
  ```

- [ ] **Testing**:
  - [ ] Verify memory footprint reduction (use memory_profiler)
  - [ ] Ensure all attributes still accessible
  - [ ] Check for attribute assignment errors (slots prevent dynamic attributes)
- [ ] **Benchmark**: Measure memory improvement (expect ~40% reduction)

**Acceptance Criteria**: Memory footprint reduced without breaking functionality

#### 4.6 Add Metric Quick-Reference Guide (Documentation)

- [ ] **File**: `src/neurospatial/metrics/README.md` or in module docstring
- [ ] **Implementation**: Add guide to `__init__.py`:

  ```python
  """Neuroscience metrics for spatial analysis.

  Quick Reference Guide
  ---------------------

  **Place Cells** (Hippocampus, spatial coding):

  - `skaggs_information()` - Spatial information content (bits/spike)
  - `detect_place_fields()` - Identify place field regions
  - `field_size()`, `field_centroid()` - Field geometric properties
  - `field_stability()` - Across-session place field stability
  - `sparsity()`, `selectivity()` - Firing rate distribution metrics

  **Grid Cells** (Entorhinal cortex, hexagonal firing):

  - `grid_score()` - Hexagonal periodicity (Sargolini et al. 2006)
  - `spatial_autocorrelation()` - Visualize grid pattern
  - `periodicity_score()` - FFT-based periodicity detection

  **Boundary Cells** (Border/wall-coding):

  - `border_score()` - Wall proximity preference (Solstad et al. 2008)
  - `compute_region_coverage()` - Spatial coverage metrics

  **Population Analysis** (Multi-unit):

  - `count_place_cells()` - Detect significant place cells
  - `population_coverage()` - Ensemble spatial coverage
  - `field_overlap()` - Spatial overlap between cells
  - `population_vector_correlation()` - Population coding similarity

  **Trajectory Metrics** (Behavior):

  - `compute_turn_angles()`, `compute_step_lengths()` - Kinematic features
  - `compute_home_range()` - Utilization distribution
  - `mean_square_displacement()` - Diffusion analysis

  See Also
  --------
  neurospatial.compute_place_field : Compute firing rate from spikes
  neurospatial.simulation : Generate synthetic data for testing

  References
  ----------
  All metrics validated against field-standard packages (opexebo, neurocode, buzcode).
  """
  ```

- [ ] **Documentation**: Also add to online docs (if using Sphinx/ReadTheDocs)

**Acceptance Criteria**: Users can quickly find relevant metrics for their cell type

**Estimated Effort**: 4-5 days

---

## Phase 5: Code Modernization (Optional)

**Goal**: Apply modern Python idioms for cleaner code

**Priority**: 🟢 LOW - Nice-to-have improvements

**Dependencies**: None

### Tasks

#### 5.1 Add Positional-Only Parameters Where Appropriate

- [ ] **Pattern**: Functions where positional args should never be passed as keywords
- [ ] **Example**:

  ```python
  # Before
  def normalize_field(field: NDArray, *, eps: float = 1e-12):
      ...

  # After
  def normalize_field(field: NDArray, /, *, eps: float = 1e-12):
      # field can only be passed positionally, prevents accidental field=...
      ...
  ```

- [ ] **Files to Update**:
  - [ ] `field_ops.py`: `normalize_field`, `clamp`, `combine_fields`
  - [ ] `distance.py`: First parameter(s) of public functions
  - [ ] Others as appropriate
- [ ] **Testing**: Ensure keyword usage raises TypeError

**Acceptance Criteria**: Better API safety, prevents accidental misuse

#### 5.2 Use Match Statements for Multi-Branch Logic (Python 3.10+)

- [ ] **Files**: `field_ops.py`, `spatial.py`, others with if-elif chains
- [ ] **Example**:

  ```python
  # Before
  if kind == "kl":
      # KL divergence
  elif kind == "js":
      # JS divergence
  elif kind == "cosine":
      # Cosine distance
  else:
      raise ValueError(f"Unknown kind '{kind}'")

  # After (Python 3.10+)
  match kind:
      case "kl":
          # KL divergence
      case "js":
          # JS divergence
      case "cosine":
          # Cosine distance
      case _:
          raise ValueError(f"Unknown kind '{kind}'")
  ```

- [ ] **Testing**: Ensure identical behavior

**Acceptance Criteria**: More readable branching logic

#### 5.3 Add Context Manager for Temporary Cache Clearing

- [ ] **File**: `src/neurospatial/environment/core.py`
- [ ] **Implementation**:

  ```python
  from contextlib import contextmanager

  @contextmanager
  def temporary_cache_clear(self: "Environment"):
      """Temporarily clear caches, restore on exit.

      Useful for testing or when modifying environment attributes.

      Examples
      --------
      >>> with env.temporary_cache_clear():
      ...     # Caches cleared here
      ...     env.bin_centers[0] = new_value  # Don't do this normally!
      ...     # Work with modified environment
      # Caches restored/cleared on exit
      """
      # Save cache state
      old_kdtree = getattr(self, '_kdtree_cache', None)
      old_kernels = getattr(self, '_kernel_cache', {}).copy()

      try:
          # Clear caches
          self.clear_cache()
          yield self
      finally:
          # Restore or re-clear (safer to re-clear)
          self.clear_cache()
  ```

- [ ] **Testing**: Test cache cleared in context, behavior after exit

**Acceptance Criteria**: Safe temporary cache manipulation

**Estimated Effort**: 2-3 days (optional)

---

## Phase 6: Documentation & Release

**Goal**: Update documentation, prepare release

**Priority**: 🔴 HIGH - Required before release

**Dependencies**: Phases 1-4 complete

### Tasks

#### 6.1 Update CHANGELOG.md

- [ ] **File**: `CHANGELOG.md`
- [ ] **Sections**:

  ```markdown
  ## [0.3.0] - 2025-XX-XX

  ### Added
  - `Environment.info()` - Human-readable environment summary
  - `Environment.explain_connectivity()` - Graph connectivity explanation
  - `EnvironmentNotFittedError` - Custom exception with helpful messages
  - `pathlib.Path` support in all I/O functions
  - Metric quick-reference guide in `metrics/__init__.py`

  ### Changed
  - **BREAKING**: `Environment.bin_at()` now handles both single and batch inputs
  - **PERFORMANCE**: Replaced custom distance matrix with scipy.sparse.csgraph (~10-100× speedup)
  - **PERFORMANCE**: Use scipy.sparse.csgraph.laplacian for differential operators
  - **PERFORMANCE**: Use scipy.ndimage.label for place field detection on grids (~5× speedup)
  - Added `__slots__` to Environment dataclass for ~40% memory reduction

  ### Deprecated
  - `map_points_to_bins()` - Use `Environment.bin_at()` instead (will be removed in v0.4.0)

  ### Fixed
  - Improved error messages for unfitted environments
  - Better numerical stability in distance computations

  ### Migration Guide

  **map_points_to_bins() → Environment.bin_at()**

  Before:
  ```python
  from neurospatial import map_points_to_bins
  bins = map_points_to_bins(points, env, tie_break="lowest_index")
  ```

  After:

  ```python
  bins = env.bin_at(points, tie_break="lowest_index")
  ```

  **Single vs. Batch Points**

  bin_at() now automatically handles both:

  ```python
  # Single point (returns int)
  bin_idx = env.bin_at([10.0, 20.0])

  # Multiple points (returns array)
  bin_indices = env.bin_at(np.array([[10, 20], [30, 40]]))
  ```

  ```

#### 6.2 Update Documentation

- [ ] **Update all examples** using `map_points_to_bins()` to use `bin_at()`
- [ ] **Update CLAUDE.md**:
  - [ ] Add `env.info()` to quick reference
  - [ ] Document deprecation of `map_points_to_bins()`
  - [ ] Update performance notes with scipy optimizations
- [ ] **Update module docstrings** with new features
- [ ] **Update type stubs** (if exists)

#### 6.3 Run Full Test Suite

- [ ] Run all tests: `uv run pytest -v`
- [ ] Run with coverage: `uv run pytest --cov=src/neurospatial --cov-report=html`
- [ ] Verify ≥95% coverage maintained
- [ ] Run doctests: `uv run pytest --doctest-modules src/neurospatial/`
- [ ] Run slow tests (benchmarks): `uv run pytest -m slow -v`

#### 6.4 Performance Benchmarks

- [ ] **Create benchmarks** (if not exists):
  - [ ] `tests/benchmarks/test_distance_performance.py`
  - [ ] `tests/benchmarks/test_place_field_performance.py`
- [ ] **Run benchmarks** and document improvements in CHANGELOG

#### 6.5 Pre-release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG complete
- [ ] Version bumped in `pyproject.toml`
- [ ] No deprecation warnings in own test suite
- [ ] Type checking passes: `uv run mypy src/neurospatial/`
- [ ] Linting passes: `uv run ruff check .`
- [ ] Format check: `uv run ruff format --check .`

**Estimated Effort**: 2-3 days

---

## Summary Timeline

**Total Estimated Effort**: 14-20 days

| Phase | Priority | Effort | Dependencies |
|-------|----------|--------|--------------|
| Phase 1: Test Coverage | 🔴 Critical | 1-2 days | None |
| Phase 2: scipy Integration | 🔴 High | 3-4 days | Phase 1 |
| Phase 3: API Simplification | 🔴 High | 2-3 days | None |
| Phase 4: UX Improvements | 🟡 Medium | 4-5 days | None |
| Phase 5: Modernization | 🟢 Low | 2-3 days | None |
| Phase 6: Documentation & Release | 🔴 High | 2-3 days | Phases 1-4 |

**Recommended Approach**:

1. Complete Phase 1 (test coverage) first
2. Run Phases 2, 3, 4 in parallel (different files)
3. Complete Phase 6 (documentation)
4. Phase 5 optional for future release

---

## Success Criteria

### Performance

- [ ] ≥10× speedup for distance matrix on 1000+ node graphs
- [ ] ≥5× speedup for place field detection on grid environments
- [ ] ~40% memory reduction with `__slots__`

### Code Quality

- [ ] ≥95% test coverage maintained
- [ ] All mypy checks pass
- [ ] All ruff checks pass
- [ ] Zero deprecation warnings in own tests

### UX

- [ ] Single unified API for point-to-bin mapping (`bin_at`)
- [ ] Clear error messages with actionable suggestions
- [ ] `info()` method provides useful debugging information
- [ ] Metric quick-reference helps users find relevant functions

### Documentation

- [ ] All examples updated to use new API
- [ ] Migration guide clear and complete
- [ ] CHANGELOG comprehensive

---

## Out of Scope (Future Work)

The following items from the code review are explicitly **excluded** from this plan:

1. ❌ Visualization comparison utilities
2. ❌ Temporal dynamics support
3. ❌ Statistical significance tests (shuffling-based)
4. ❌ Quick-start Jupyter notebook
5. ❌ "Common Pitfalls" documentation section
6. ❌ Multi-scale analysis (wavelets/pyramids)
7. ❌ Directed graph support

These may be considered for future releases (0.4.0+).

---

## Risk Mitigation

### Risk: scipy implementation behaves differently

- **Mitigation**: Comprehensive test coverage (Phase 1), regression tests with pinned values
- **Contingency**: Revert to custom implementation, file scipy bug report

### Risk: Breaking changes affect downstream users

- **Mitigation**: Deprecation warnings, clear migration guide, semantic versioning
- **Contingency**: Extend deprecation period to v0.5.0

### Risk: Performance improvements don't materialize

- **Mitigation**: Benchmark before and after, document actual gains
- **Contingency**: Keep optimization, even if gains are modest (<10×)

### Risk: **slots** breaks mixin pattern

- **Mitigation**: Test thoroughly, check Python 3.10+ dataclass slots behavior
- **Contingency**: Make slots optional or skip this optimization

---

## Post-Release Actions

After v0.3.0 release:

1. **Monitor User Feedback**
   - Watch GitHub issues for migration problems
   - Update FAQ based on common questions

2. **Performance Tracking**
   - Collect benchmarks from users on real data
   - Document typical speedups in different scenarios

3. **Deprecation Timeline**
   - v0.3.0: `map_points_to_bins()` deprecated (warning)
   - v0.4.0: `map_points_to_bins()` removed

4. **Future Planning**
   - Evaluate out-of-scope items for v0.4.0
   - Consider numba integration for further speedups

---

**Plan Status**: Draft
**Next Review**: After Phase 1 completion
**Plan Owner**: Development team
