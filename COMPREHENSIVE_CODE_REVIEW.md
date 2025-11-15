# Comprehensive Code Review: neurospatial

**Review Date**: 2025-11-15
**Version**: 0.2.0
**Reviewer**: Claude Code

---

## Executive Summary

The **neurospatial** package is a well-architected scientific Python library with strong design patterns, comprehensive documentation, and appropriate use of modern Python features. The codebase demonstrates professional software engineering practices adapted for scientific computing. However, there are opportunities for improvement in leveraging standard libraries, reducing custom implementations, and improving UX for neuroscientists with intermediate Python skills.

**Overall Assessment**: ⭐⭐⭐⭐ (4/5 stars)

**Strengths**:

- Excellent architecture (Protocol-based design, mixin pattern)
- Comprehensive testing (95 test files)
- Well-documented (NumPy docstrings throughout)
- Type-annotated with mypy integration
- Domain-validated metrics (against opexebo, neurocode)

**Areas for Improvement**:

- Some custom implementations could use standard libraries
- API complexity may confuse intermediate users
- Opportunities to leverage scipy.sparse more extensively
- Minor redundancies in distance computations

---

## 1. Architecture Assessment

### ✅ Strengths

1. **Protocol-Based Design** (Excellent)
   - Layout engines use PEP 544 Protocols instead of inheritance
   - Enables duck typing and easier testing
   - Clean separation of concerns

2. **Mixin Pattern** (Very Good)
   - Environment class modularized into 9 focused mixins
   - Single dataclass, plain class mixins (prevents field conflicts)
   - Type safety via `EnvironmentProtocol`

3. **Factory Pattern** (Excellent)
   - Six well-named factory methods (`from_samples`, `from_polygon`, etc.)
   - Encapsulates complexity, ensures fitted state
   - Clear decision tree in documentation

4. **Immutable Value Objects** (Excellent)
   - `Region`, `AffineND`, frozen dataclasses
   - Thread-safe, predictable behavior

### ⚠️ Concerns

1. **Mixin Complexity**
   - 9 mixins might be overwhelming for contributors
   - Type annotations require `TYPE_CHECKING` guards and `EnvironmentProtocol`
   - **Recommendation**: Consider providing a "mixin contributor guide" in CONTRIBUTING.md

2. **Graph Metadata Requirements**
   - Mandatory node/edge attributes create tight coupling
   - Difficult to use with externally-created NetworkX graphs
   - **Recommendation**: Add utility to upgrade/validate external graphs

---

## 2. Gaps in Functionality

### Missing Standard Features

1. **No Built-in Visualization Comparison**
   - Metrics exist but no standard plotting for multi-session comparisons
   - **Suggestion**: Add `plot_place_field_comparison(session1, session2)` helper

2. **Limited 3D Support**
   - Most layout engines are 2D-focused
   - Hexagonal, triangular layouts don't extend to 3D naturally
   - **Suggestion**: Document 3D limitations explicitly; add `VolumetricGridLayout` for cubic grids

3. **No Temporal Dynamics**
   - No built-in support for time-varying fields (e.g., remapping over epochs)
   - **Suggestion**: Add `TemporalEnvironment` wrapper with time axis

4. **Limited Graph Types**
   - No support for directed graphs (e.g., one-way passages)
   - No support for weighted connectivity beyond distance
   - **Suggestion**: Add `DirectedEnvironment` subclass or `allow_directed` parameter

5. **No Multi-Scale Analysis**
   - Resampling exists but no wavelet/multi-resolution pyramid
   - **Suggestion**: Add `create_scale_space(env, scales=[1, 2, 4, 8])` utility

6. **Missing Statistical Tests**
   - Metrics compute values but no significance tests
   - **Suggestion**: Add `test_place_cell_significance(firing_rate, env, n_shuffles=1000)`

---

## 3. Custom Implementations vs. Standard Libraries

### ✅ Already Using Standard Libraries Well

1. **scipy.spatial.cKDTree** - Used for point-to-bin mapping (excellent)
2. **networkx** - Graph algorithms (shortest paths, Dijkstra)
3. **scipy.spatial.distance.pdist/squareform** - Pairwise distances
4. **numpy.linalg** - Linear algebra operations
5. **shapely** - Geometric operations (polygons, containment)

### ⚠️ Custom Implementations That Could Use Standard Libraries

#### HIGH PRIORITY

1. **Graph Laplacian and Diffusion Kernels** (`kernels.py`, `differential.py`)
   - **Current**: Custom sparse matrix construction for Laplacian
   - **Alternative**: Use `scipy.sparse.csgraph.laplacian()` directly
   - **Benefit**: Faster, more robust, standard normalization options

   ```python
   # Current (custom)
   from neurospatial.differential import compute_differential_operator
   L = compute_differential_operator(env.connectivity)

   # Could use scipy (proposed)
   from scipy.sparse.csgraph import laplacian
   adj = nx.to_scipy_sparse_array(env.connectivity, weight='distance')
   L = laplacian(adj, normed=True)  # Returns sparse matrix
   ```

2. **Pairwise Distance Matrix** (`distance.py:32-62`)
   - **Current**: Custom `geodesic_distance_matrix()` iterates with NetworkX
   - **Alternative**: Use `scipy.sparse.csgraph.shortest_path()`
   - **Benefit**: C implementation, much faster for large graphs

   ```python
   # Current (custom, 15 lines)
   def geodesic_distance_matrix(G, n_states, weight='distance'):
       dist_matrix = np.full((n_states, n_states), np.inf)
       for src, lengths in nx.shortest_path_length(G, weight=weight):
           for dst, L in lengths.items():
               dist_matrix[src, dst] = L
       return dist_matrix

   # Could use scipy (proposed, 3 lines)
   from scipy.sparse.csgraph import shortest_path
   def geodesic_distance_matrix(G, n_states, weight='distance'):
       adj = nx.to_scipy_sparse_array(G, weight=weight)
       return shortest_path(adj, directed=False, method='D')  # Dijkstra, all-pairs
   ```

3. **Spatial Autocorrelation** (`metrics/grid_cells.py`)
   - **Current**: Custom 2D correlation implementation
   - **Alternative**: Use `scipy.signal.correlate2d()` or `scipy.ndimage.correlate()`
   - **Benefit**: Optimized, handles edge cases better

4. **Connected Components** (`metrics/place_fields.py`)
   - **Current**: Custom flood-fill for field detection
   - **Alternative**: Use `scipy.ndimage.label()` for grid-based environments
   - **Benefit**: Faster, more robust, handles N-D

#### MEDIUM PRIORITY

5. **Convolution** (`primitives.py:192-372`)
   - **Current**: Manual O(n²) loop for graph convolution
   - **Alternative**: Could use sparse matrix multiplication more efficiently
   - **Benefit**: Performance improvement for large environments

   ```python
   # Current approach computes distances in nested loop
   # Could precompute distance matrix once and reuse
   ```

6. **Interpolation** (`environment/fields.py`)
   - **Current**: Custom KDTree-based interpolation
   - **Alternative**: Check if `scipy.interpolate.griddata()` or `scipy.interpolate.RBFInterpolator` applicable
   - **Benefit**: More interpolation methods (cubic, quintic)

7. **Histogram Binning** (`environment/trajectory.py`)
   - **Current**: Custom occupancy computation
   - **Alternative**: Could use `numpy.histogram()` or `numpy.histogramdd()`
   - **Benefit**: Simpler, well-tested edge case handling

#### LOW PRIORITY

8. **KL/JS Divergence** (`field_ops.py:267-478`)
   - **Current**: Custom implementation with eps handling
   - **Alternative**: Use `scipy.stats.entropy()` for KL, or `scipy.spatial.distance.jensenshannon()`
   - **Benefit**: Standardized, validated implementation
   - **Note**: Current implementation is good quality with proper validation; low priority

9. **Box Kernel** (`primitives.py`)
   - **Current**: Custom callable kernel
   - **Alternative**: Could reference `scipy.signal.windows` for standard window functions
   - **Benefit**: Consistency with signal processing literature

### ❌ Good Custom Implementations (Keep)

These are appropriately custom for the domain:

1. **Layout engines** - Domain-specific, no direct scipy equivalent
2. **Place field metrics** - Neuroscience-specific, validated against field standards
3. **Trajectory segmentation** - Domain logic, no standard library
4. **Simulation models** - Scientific models, appropriately custom
5. **Region operations** - Well-integrated with Shapely, appropriate level

---

## 4. Redundancies

### 🔄 Identified Redundancies

1. **Distance Computations** (Minor redundancy)
   - `distance_between()` method on Environment
   - `geodesic_distance_between_points()` function in `distance.py`
   - `distance_field()` for single source
   - **Recommendation**: Consolidate, make `distance_between` call lower-level primitives

2. **Point-to-Bin Mapping** (Not redundant, but API overlap)
   - `Environment.bin_at()` - delegates to layout engine
   - `map_points_to_bins()` - KDTree-based batch mapping
   - **Current status**: Appropriate specialization, but naming could clarify
   - **Recommendation**: Rename `bin_at()` to `find_bin()` or document difference clearly

3. **Normalization** (Minor)
   - `normalize_field()` in `field_ops.py`
   - Normalization logic in `kernels.py` for kernel matrices
   - **Recommendation**: Have kernel normalization call `normalize_field()` internally

4. **File I/O** (Acceptable duplication)
   - `Environment.to_file()` / `from_file()` methods
   - Top-level `to_file()` / `from_file()` functions
   - **Current status**: Convenience, both APIs useful
   - **Recommendation**: Keep both, ensure delegation

5. **Shortest Path** (Appropriate specialization)
   - `Environment.shortest_path()` - returns path
   - `distance_field()` - returns distances
   - **Current status**: Different return types, appropriate

---

## 5. Modern Python Best Practices

### ✅ Excellent Modern Practices

1. **Type Hints** (Excellent)
   - Comprehensive type annotations with `numpy.typing.NDArray`
   - Literal types for string enums
   - Protocol-based structural typing
   - **Gold standard** for scientific Python

2. **Dataclasses** (Excellent)
   - Frozen dataclasses for immutability (`Region`)
   - Proper use of `field()` with factories
   - Clean, readable definitions

3. **Context Managers** (Could improve)
   - No custom context managers currently
   - **Suggestion**: Add context manager for temporary cache clearing:

   ```python
   with env.temporary_cache_clear():
       # Work with modified environment
       pass
   # Cache restored
   ```

4. **Enums** (Good)
   - `TieBreakStrategy`, `LayoutType` as proper Enums
   - **Suggestion**: Use `StrEnum` (Python 3.11+) or `auto()` for values

5. **f-strings** (Excellent)
   - Consistent use throughout
   - Clear error messages with diagnostics

6. **Pathlib** (Missing)
   - File I/O uses string paths, not `pathlib.Path`
   - **Recommendation**: Accept `str | Path` in I/O functions

   ```python
   from pathlib import Path
   def to_file(env: Environment, path: str | Path) -> None:
       path = Path(path)
       # ...
   ```

7. **Logging** (Good)
   - Uses `logging` module appropriately
   - Structured logging with `_logging.py`
   - **Suggestion**: Add log level configuration in Environment creation

8. **Warnings** (Good)
   - Uses `warnings.warn()` for non-critical issues
   - Proper `stacklevel` for user-facing warnings

### ⚠️ Areas to Modernize

1. **Type Hints for Callable** (Minor)
   - Uses `Callable[[NDArray], NDArray]` - could use `collections.abc.Callable`
   - Python 3.9+ allows `from collections.abc import Callable` (preferred over `typing.Callable`)

2. **Match Statements** (Optional, Python 3.10+)
   - Multiple if-elif chains could use `match` (e.g., `field_ops.py:420-476`)
   - **Example**:

   ```python
   # Current
   if kind == "kl":
       # ...
   elif kind == "js":
       # ...
   elif kind == "cosine":
       # ...

   # Could use match (Python 3.10+)
   match kind:
       case "kl":
           # ...
       case "js":
           # ...
       case "cosine":
           # ...
   ```

3. **Slots** (Optimization)
   - Dataclasses don't use `__slots__` for memory efficiency
   - **Recommendation**: Add `__slots__` to `Environment` (Python 3.10+)

   ```python
   @dataclass(slots=True)  # Python 3.10+
   class Environment(...):
       # Reduces memory footprint by ~40%
   ```

4. **Generic Types** (Minor)
   - Could use `TypeVar` for generic field operations
   - Low priority, current approach works well

5. **Positional-Only Parameters** (Missing)
   - Many functions have required positional args that could use `/`
   - **Example**: `normalize_field(field, /, *, eps=1e-12)` prevents accidental `field=...`

---

## 6. UX for Neuroscientists (Domain Experts, Intermediate Python)

### 🎯 Strong UX Elements

1. **Factory Method Names** (Excellent)
   - `from_samples()`, `from_polygon()` - intuitive, clear purpose
   - Decision tree in documentation helps users choose

2. **Default Parameters** (Good)
   - Sensible defaults (e.g., `bin_size` required but other params optional)
   - Clear when to override

3. **Error Messages** (Excellent)
   - Diagnostic values included (e.g., "got bin_size=-2.0")
   - Suggestions for fixes (e.g., "reduce bin_size to 1.0")
   - **Gold standard**

4. **Examples in Docstrings** (Excellent)
   - NumPy-style examples throughout
   - Runnable, realistic use cases

5. **Simulation Examples** (`simulation/examples.py`)
   - Pre-configured sessions: `open_field_session()`, `linear_track_session()`
   - **Extremely helpful** for learning and testing

### ⚠️ UX Confusions for Intermediate Users

#### HIGH PRIORITY CONFUSIONS

1. **Too Many Import Paths** (Confusing)
   - **Problem**: Multiple ways to import the same thing

   ```python
   # All equivalent, but which is "right"?
   from neurospatial import Environment
   from neurospatial.environment import Environment
   from neurospatial.environment.core import Environment
   ```

   - **Recommendation**: Document canonical import paths in `__init__.py` docstring
   - **Best practice**: Recommend `from neurospatial import X` always

2. **Mixin Methods Not Discoverable** (Medium confusion)
   - **Problem**: IDE autocomplete shows all 60+ methods unsorted
   - **Example**: User types `env.` and sees 60 methods in random order
   - **Recommendation**: Add `@property` decorators or group in `__dir__()`:

   ```python
   def __dir__(self):
       # Group methods by category for better IDE experience
       groups = ['Queries', 'Trajectory', 'Metrics', ...]
       # Return sorted by group
   ```

3. **`bin_at()` vs `map_points_to_bins()`** (Confusing)
   - **Problem**: Both map points to bins, unclear when to use which
   - **Explanation** (hidden in docs): `bin_at` is single-point, no caching; `map_points_to_bins` is batched with KDTree
   - **Recommendation**:
     - Rename `bin_at()` to `find_bin_for_point()` (clearer)
     - Or deprecate `bin_at()` in favor of `map_points_to_bins()` always

4. **`check_fitted` Decorator** (Confusing error)
   - **Problem**: Error message is generic Python exception, not domain-specific
   - **Current**: `RuntimeError: Environment must be fitted before calling this method`
   - **Better**: `EnvironmentNotFittedError: Environment not initialized. Use Environment.from_samples() or another factory method instead of Environment().`
   - **Recommendation**: Custom exception class with helpful message

5. **Too Many Metrics** (Overwhelming)
   - **Problem**: 26 neuroscience metrics in `metrics/` - which to use?
   - **Recommendation**: Add "quick start" guide grouping by cell type:

   ```markdown
   ## Quick Metric Guide

   **Place Cells**:
   - `skaggs_information()` - spatial information content
   - `detect_place_fields()` - find fields
   - `field_size()`, `field_centroid()` - field properties

   **Grid Cells**:
   - `grid_score()` - hexagonal periodicity
   - `spatial_autocorrelation()` - visualize grid pattern

   **Boundary Cells**:
   - `border_score()` - wall proximity preference
   ```

#### MEDIUM PRIORITY CONFUSIONS

6. **Kernel Caching** (Hidden magic)
   - **Problem**: Caching happens automatically, not visible to user
   - **Confusion**: "Why is the second call faster?" - user doesn't know about cache
   - **Recommendation**: Add `info()` method showing cache status:

   ```python
   >>> env.info()
   Environment: my_maze
     Bins: 1,024
     Dimensions: 2D (100 × 100 cm)
     Layout: RegularGrid
     Regions: 3 (start, goal, reward)
     Cache: KDTree (built), kernels (2 cached)
   ```

7. **`tie_break` Parameter** (Advanced concept exposed early)
   - **Problem**: Users must choose tie-break strategy immediately
   - **Most users**: Don't care, want deterministic results
   - **Recommendation**: Keep `LOWEST_INDEX` default, but don't expose in quick-start examples

8. **Units Management** (Optional but inconsistent)
   - **Problem**: `units` and `frame` are optional metadata
   - **Confusion**: "Do I need to set units?" - unclear if it affects computation
   - **Recommendation**: Add warning if units not set in `from_samples()`

9. **Region Types** (Point vs Polygon confusion)
   - **Problem**: `Region` can be point or polygon, but operations differ
   - **Example**: `regions_to_mask()` warns that point regions return empty masks
   - **Recommendation**: Separate classes `PointRegion` and `PolygonRegion` for type safety

10. **Graph Connectivity** (Black box)
    - **Problem**: Users don't understand how `connectivity` graph is built
    - **Recommendation**: Add `explain_connectivity()` method:

    ```python
    >>> env.explain_connectivity()
    Connectivity: 4-connected grid (up, down, left, right)
    Diagonal neighbors: No
    Edge weights: Euclidean distance between bin centers
    Total edges: 3,824
    Average degree: 3.74
    ```

#### LOW PRIORITY

11. **Composite Environment** (Advanced feature, under-documented)
    - **Problem**: Bridge inference is "automatic" but opaque
    - **Recommendation**: Add visualization of bridges

12. **Active Bins** (Terminology barrier)
    - **Problem**: "Active bins" is jargon, not intuitive
    - **Recommendation**: Rename to "visited bins" or "occupied bins" in user-facing docs

---

## 7. Performance Opportunities

### 🚀 Quick Wins

1. **Use scipy.sparse.csgraph for All Graph Algorithms** (High impact)
   - Replace custom geodesic distance matrix
   - Potential 10-100× speedup for large graphs

2. **Vectorize Connected Component Detection** (Medium impact)
   - Use `scipy.ndimage.label()` instead of flood-fill
   - Faster for grid-based place field detection

3. **Cache More Aggressively** (Medium impact)
   - Cache `boundary_bins` (currently cached_property ✓)
   - Cache `bin_sizes` (currently cached_property ✓)
   - Add cache for `pairwise_distances` with LRU

4. **Sparse Matrix Operations** (High impact for large envs)
   - Transition matrix in `trajectory.py` - already uses sparse ✓
   - Kernel matrices could be sparse (currently dense)
   - **Recommendation**: Add `sparse` parameter to `compute_kernel()`

5. **Numba JIT** (High impact, but adds dependency)
   - Hot loops in `neighbor_reduce()`, `convolve()` could use `@njit`
   - **Trade-off**: Adds numba dependency
   - **Recommendation**: Optional dependency for power users

### 📊 Benchmarking Recommendations

Add performance benchmarks for:

- Point-to-bin mapping (current tests marked `@pytest.mark.slow`)
- Distance field computation
- Kernel application
- Place field detection

---

## 8. Scientific Correctness

### ✅ Excellent Validation

1. **Metrics Validated Against Field Standards** (Excellent)
   - Place field metrics: opexebo, neurocode, buzcode
   - Grid scores: Sargolini et al. 2006 reference
   - Border scores: Solstad et al. 2008

2. **Simulation Validation** (Good)
   - Ornstein-Uhlenbeck trajectories (realistic diffusion)
   - Poisson spike generation with refractory period
   - **Suggestion**: Add unit tests comparing to analytical expectations

3. **Edge Case Handling** (Excellent)
   - Empty graphs, single-node graphs handled
   - NaN propagation documented
   - Isolated nodes return NaN (appropriate)

### ⚠️ Potential Issues

1. **Boundary Effects in Smoothing** (Minor)
   - Graph-based smoothing handles boundaries correctly ✓
   - Euclidean methods might have edge artifacts
   - **Recommendation**: Document boundary behavior explicitly

2. **Statistical Power** (Missing)
   - No sample size calculations or power analysis
   - **Suggestion**: Add `estimate_required_samples()` for place field detection

3. **Multiple Comparison Correction** (Missing)
   - No Bonferroni or FDR correction for multiple cells
   - **Suggestion**: Add `correct_multiple_comparisons()` utility

---

## 9. Documentation Quality

### ✅ Excellent

1. **NumPy Docstring Format** (Gold standard)
2. **Examples in Every Function** (Excellent)
3. **CLAUDE.md** (Outstanding) - Comprehensive guide for AI assistants
4. **Error Messages with Diagnostics** (Best-in-class)

### 📝 Recommendations

1. **Add Quickstart Notebook** (High value)
   - Jupyter notebook: "10 minutes to neurospatial"
   - Cover: create environment, compute place field, plot results

2. **Add API Reference Grouping** (Medium value)
   - Group functions by category in generated docs (Sphinx)
   - Currently flat alphabetical

3. **Add "Common Pitfalls" Section** (Medium value)
   - Based on user issues, document common mistakes

4. **Add Conceptual Diagrams** (High value for neuroscientists)
   - Diagram showing Environment architecture
   - Flowchart for choosing factory method
   - Visual explanation of active bins

---

## 10. Testing

### ✅ Strong Test Coverage

1. **95 Test Files** - Comprehensive
2. **Property-Based Testing** - hypothesis integration ✓
3. **Benchmark Tests** - `@pytest.mark.slow` for performance
4. **Error Path Testing** - dedicated `test_environment_error_paths.py`

### 📝 Testing Recommendations

1. **Add Integration Tests for Common Workflows** (High value)
   - End-to-end: load data → create environment → compute place fields → plot
   - Currently tests are unit-focused

2. **Add Regression Tests for Metrics** (High value)
   - Pin expected values for synthetic data
   - Ensure updates don't change metric definitions

3. **Add Serialization Round-Trip Tests** (Medium value)
   - Ensure `to_file()` → `from_file()` is lossless
   - Test with all layout types

---

## Priority Recommendations Summary

### 🔴 High Priority (Do Soon)

1. **Replace custom distance matrix with `scipy.sparse.csgraph.shortest_path()`**
   - File: `distance.py:32-62`
   - Benefit: 10-100× speedup, standard implementation

2. **Clarify `bin_at()` vs `map_points_to_bins()` API**
   - Rename or deprecate to reduce confusion
   - Document performance difference

3. **Add "Quick Start" guide with notebook**
   - Target: neuroscientist with basic Python
   - 10-minute end-to-end example

4. **Use `scipy.ndimage.label()` for place field detection**
   - File: `metrics/place_fields.py`
   - Benefit: Faster, more robust

5. **Add `env.info()` method for inspection**
   - Show: bins, dimensions, cache status, regions
   - Improves discoverability

### 🟡 Medium Priority (Consider for Next Release)

6. **Use scipy.sparse.csgraph.laplacian() for graph Laplacian**
   - Files: `differential.py`, `kernels.py`
   - Benefit: Standard, faster

7. **Support `pathlib.Path` in I/O functions**
   - More Pythonic, better type hints

8. **Add custom `EnvironmentNotFittedError` exception**
   - Better error messages for beginners

9. **Add metric quick-reference guide**
   - Group by cell type (place, grid, boundary)

10. **Add slots to Environment dataclass** (Python 3.10+)
    - Memory optimization

### 🟢 Low Priority (Nice to Have)

11. **Use match statements** (Python 3.10+)
    - Modernize if-elif chains

12. **Add context manager for cache clearing**
    - Better control over caching behavior

13. **Add visualization comparison utilities**
    - Multi-session place field comparison plots

14. **Add temporal dynamics support**
    - Time-varying fields, remapping detection

15. **Add statistical significance tests**
    - Shuffling-based place cell detection

---

## Conclusion

The **neurospatial** package is a high-quality scientific Python library with excellent architecture, comprehensive testing, and strong documentation. The codebase follows modern Python best practices and is well-suited for neuroscience research.

**Key Strengths**:

- Professional software engineering (Protocol, mixin, factory patterns)
- Comprehensive neuroscience metric validation
- Excellent documentation and error messages
- Strong type safety with mypy

**Main Opportunities**:

- Leverage scipy.sparse.csgraph for graph algorithms (significant performance gains)
- Simplify API for intermediate Python users (reduce confusion)
- Add quick-start materials for neuroscientists
- Minor modernizations (pathlib, slots, custom exceptions)

**Verdict**: This package is production-ready and suitable for publication. Implementing the high-priority recommendations would significantly improve performance and user experience, but the current state is already well above the standard for research software.

---

## Appendix: Detailed File-by-File Notes

### scipy Integration Opportunities

**distance.py** (Line 32-62):

```python
# BEFORE (custom implementation)
def geodesic_distance_matrix(G, n_states, weight='distance'):
    dist_matrix = np.full((n_states, n_states), np.inf)
    np.fill_diagonal(dist_matrix, 0.0)
    for src, lengths in nx.shortest_path_length(G, weight=weight):
        for dst, L in lengths.items():
            dist_matrix[src, dst] = float(L)
    return dist_matrix

# AFTER (scipy implementation)
from scipy.sparse.csgraph import shortest_path
def geodesic_distance_matrix(G, n_states, weight='distance'):
    adj = nx.to_scipy_sparse_array(G, weight=weight)
    return shortest_path(adj, directed=False, method='D')
```

**differential.py** (Laplacian construction):

```python
# Consider using scipy.sparse.csgraph.laplacian
from scipy.sparse.csgraph import laplacian
adj = nx.to_scipy_sparse_array(G, weight='distance')
L = laplacian(adj, normed=True)  # Normalized Laplacian
```

**metrics/place_fields.py** (Connected components):

```python
# Consider scipy.ndimage.label for grid environments
from scipy.ndimage import label
labeled, n_components = label(firing_rate > threshold)
```

### API Simplification Recommendations

**spatial.py** - Rename for clarity:

```python
# Current (confusing name overlap with bin_at)
map_points_to_bins(points, env, tie_break="lowest_index")

# Suggested (clearer distinction)
batch_find_bins(points, env, tie_break="lowest_index")  # Emphasizes batch operation
# OR deprecate Environment.bin_at() entirely
```

**environment/core.py** - Add info method:

```python
def info(self) -> str:
    """Return human-readable environment summary."""
    return f"""
    Environment: {self.name or 'unnamed'}
      Bins: {self.n_bins:,}
      Dimensions: {self.n_dims}D
      Extent: {self.extent}
      Layout: {self.layout._layout_type_tag}
      Regions: {len(self.regions)}
      Cache: KDTree {'built' if hasattr(self, '_kdtree_cache') else 'not built'}
    """.strip()
```

---

**Review completed**: 2025-11-15
**Recommendations**: 15 prioritized suggestions
**Overall assessment**: Production-ready with opportunities for enhancement
