# Environment Operations - Implementation Tasks

**Status**: Ready to Begin
**Timeline**: 6 weeks (5 phases)
**Last Updated**: 2025-11-03

---

## Phase 1: Kernel Infrastructure (Week 1)

**Goal**: Establish diffusion kernel foundation for all smoothing operations.

### Tasks

- [x] Create `src/neurospatial/kernels.py`
  - [x] Implement `compute_diffusion_kernels()` with NumPy docstrings
  - [x] Implement `_assign_gaussian_weights_from_distance()` helper
  - [x] Add object identity-based caching
  - [x] Add performance warnings (O(n³) complexity)
- [x] Add `Environment.compute_kernel()` method in `environment.py`
  - [x] Add `@check_fitted` decorator
  - [x] Implement caching logic with `id(self)`
  - [x] Support both `mode='transition'` and `mode='density'`
- [x] Create `tests/test_kernels.py`
  - [x] Test kernel symmetry (for uniform grids)
  - [x] Test normalization (transition: column sums = 1)
  - [x] Test normalization (density: weighted sums = 1)
  - [x] Test mass conservation
  - [x] Test cache hit/miss behavior
  - [x] Test across multiple layout types

**Deliverable**: ✅ Kernel infrastructure ready for use in occupancy/smoothing

---

## Phase 2: Core Analysis Operations (Weeks 2-3)

**Goal**: Implement P0 operations (occupancy, sequences, transitions, connectivity).

### P0.1: Occupancy / Dwell Time ✅ COMPLETE

- [x] Implement `Environment.occupancy()` in `environment.py`
  - [x] Add `@check_fitted` decorator
  - [x] Input validation (array shapes, lengths, dtypes, monotonic timestamps)
  - [x] Handle empty arrays gracefully
  - [x] Map points to bins using `spatial.map_points_to_bins`
  - [x] Implement time accumulation logic (using np.bincount)
  - [x] Add speed filtering (`min_speed` parameter)
  - [x] Add gap handling (`max_gap` parameter, default 0.5s)
  - [x] Add optional kernel smoothing (`kernel_bandwidth` parameter, mode='transition')
  - [x] Ensure mass conservation (occupancy.sum() = total_time)
- [x] Tests for `occupancy()`
  - [x] Synthetic L-shaped path with known durations
  - [x] Sparse samples with large gaps
  - [x] Speed filtering behavior
  - [x] Kernel smoothing (verify mass conservation)
  - [x] Edge case: all samples outside environment
  - [x] Edge case: empty input arrays
  - [x] Performance: 100k samples < 1 second (19 tests passing)

### P0.2: Bin Sequence / Runs ✅ COMPLETE

- [x] Implement `Environment.bin_sequence()` in `environment.py`
  - [x] Add `@check_fitted` decorator
  - [x] Input validation (consistent with occupancy())
  - [x] Map trajectory to bin indices (using bin_at())
  - [x] Implement deduplication (`dedup=True`)
  - [x] Implement run-length encoding (`return_runs=True`)
  - [x] Handle `outside_value=None` (drop samples)
- [x] Tests for `bin_sequence()` (24 tests passing)
  - [x] Known trajectory with expected bin sequence
  - [x] Deduplication: [A,A,B,B,C] → [A,B,C]
  - [x] Run boundaries (start/end indices)
  - [x] Outside samples handling
  - [x] Edge case: trajectory crossing boundary

### P0.3: Transitions / Adjacency Matrix ✅ COMPLETE

- [x] Implement `Environment.transitions()` in `environment.py`
  - [x] Add `@check_fitted` decorator
  - [x] Input validation
  - [x] Call `bin_sequence()` if bins not provided
  - [x] Extract transition pairs with lag
  - [x] Build adjacency set from graph
  - [x] Filter non-adjacent transitions (`allow_teleports=False`)
  - [x] Count transitions in sparse COO format
  - [x] Convert to CSR matrix
  - [x] Optional row normalization
- [x] Tests for `transitions()` (26 tests passing)
  - [x] 1D track: symmetric transitions
  - [x] Grid navigation patterns
  - [x] Teleport filtering (non-adjacent transitions)
  - [x] Normalization: rows sum to 1
  - [x] Lag parameter behavior

### P0.4: Connected Components / Reachability ✅ COMPLETE

- [x] Implement `Environment.components()` in `environment.py`
  - [x] Add `@check_fitted` decorator
  - [x] Use `nx.connected_components()`
  - [x] Sort by component size
  - [x] Handle `largest_only=True`
- [x] Implement `Environment.reachable_from()` in `environment.py`
  - [x] Add `@check_fitted` decorator
  - [x] BFS for `metric='hops'`
  - [x] Dijkstra for `metric='geodesic'`
  - [x] Optional radius cutoff
  - [x] Return boolean mask
- [x] Tests for connectivity (21 tests passing)
  - [x] Graph with isolated island (2 components)
  - [x] BFS with radius (hop neighborhoods)
  - [x] Geodesic distance constraint
  - [x] Edge case: disconnected graph

**Deliverable**: P0 operations complete with full test coverage

---

## Phase 3: Smoothing & Masking (Week 4)

**Goal**: Implement P1 operations (smooth, rebin, subset).

### P1.6: Field Smoothing ✅ COMPLETE

- [x] Implement `Environment.smooth()` in `environment.py`
  - [x] Add `@check_fitted` decorator
  - [x] Validate field shape matches `n_bins`
  - [x] Validate field for NaN/Inf values
  - [x] Call `compute_kernel(bandwidth, mode)`
  - [x] Return kernel @ field
- [x] Tests for `smooth()`
  - [x] Impulse spreading to neighbors
  - [x] Mass conservation after smoothing
  - [x] Edge preservation (no cross-component leakage)
  - [x] Invalid field shape raises ValueError
  - [x] NaN/Inf values raise ValueError (23 tests passing)

### P1.7: Rebin (Grid Coarsening) ✅

- [x] Implement `Environment.rebin()` in `environment.py`
  - [x] Add `@check_fitted` decorator
  - [x] Check layout is `RegularGridLayout`
  - [x] Raise `NotImplementedError` for non-grids
  - [x] Parse `factor` (int or tuple)
  - [x] Compute new grid edges
  - [x] Build new connectivity
  - [x] Return new Environment
  - [x] Geometry-only operation (field aggregation done separately)
- [x] Tests for `rebin()`
  - [x] Grid shape reduction (factor=2, factor=3)
  - [x] Bin position correctness
  - [x] Non-grid layout raises error
  - [x] Factor validation (positive, not too large)
  - [x] Non-divisible warning and truncation
  - [x] Connectivity preservation
  - [x] Units/frame preservation (14 tests passing, 3 skipped)

### P1.8: Subset / Crop ✅ COMPLETE

- [x] Implement `Environment.subset()` in `environment.py`
  - [x] Add `@check_fitted` decorator
  - [x] Validate exactly one of {bins, region_names, polygon}
  - [x] Build boolean mask from input
  - [x] Extract subgraph
  - [x] Renumber nodes to [0, n'-1]
  - [x] Create new layout with subset
  - [x] Return new Environment (regions dropped)
- [x] Tests for `subset()` (24 tests passing)
  - [x] Crop 10x10 grid to left half → 5x10
  - [x] Polygon selection (bin centers inside)
  - [x] Connectivity preserved
  - [x] Node renumbering correctness
  - [x] Point-type region validation
  - [x] Invert parameter
  - [x] Metadata handling

**Deliverable**: P1 operations with layout compatibility checks

---

## Phase 4: Interpolation & Field Utilities (Week 5)

**Goal**: Implement P2 operations (interpolation, field math).

### P2.9: Field Interpolation ✅

- [x] Implement `Environment.interpolate()` in `environment.py`
  - [x] Add `@check_fitted` decorator
  - [x] Input validation (field, points, mode, NaN/Inf checks)
  - [x] Nearest neighbor mode (KDTree, all layouts)
  - [x] Linear mode (grids only)
    - [x] Check layout is `RegularGridLayout`
    - [x] Use `scipy.interpolate.RegularGridInterpolator`
  - [x] Handle out-of-bounds → NaN
- [x] Tests for `interpolate()` (24 passing, 1 skipped)
  - [x] Known plane field f(x,y) = 2x + 3y (linear mode - exact recovery)
  - [x] Nearest mode on polygon layout
  - [x] Out-of-bounds points → NaN
  - [x] Linear mode requires RegularGridLayout
  - [x] Comprehensive validation tests (9 tests)
  - [x] Edge cases (3 tests)
  - [x] Multiple layouts (2 tests)
  - [x] Determinism test

### P2.10: Field Math Utilities ✅ COMPLETE

- [x] Create `src/neurospatial/field_ops.py`
  - [x] Implement `normalize_field(field, eps=1e-12)`
  - [x] Implement `clamp(field, lo=0.0, hi=inf)`
  - [x] Implement `combine_fields(fields, weights, mode='mean')`
  - [x] Implement `divergence(p, q, kind='kl', eps=1e-12)`
    - [x] KL divergence
    - [x] JS divergence
    - [x] Cosine distance
  - [x] All functions: shape validation, NumPy docstrings
- [x] Create `tests/test_field_ops.py`
  - [x] Test KL/JS identities (self vs self = 0)
  - [x] Test JS symmetry
  - [x] Test cosine bounds [0, 2]
  - [x] Test normalization preserves shape
  - [x] Test combine_fields with weights (51 tests passing)

### P2.11: Linear Occupancy (Grid Enhancement) ✅ COMPLETE

- [x] Add `time_allocation` parameter to `occupancy()`
  - [x] Support `time_allocation='start'` (default)
  - [x] Support `time_allocation='linear'` (grids only)
  - [x] Ray-grid intersection logic (DDA-like algorithm)
  - [x] Proportional time splitting across bins
  - [x] Helper methods: `_allocate_time_linear()`, `_compute_ray_grid_intersections()`, `_position_to_flat_index()`
- [x] Tests for linear occupancy (17 tests, all passing)
  - [x] Diagonal trajectory across 4 bins
  - [x] Proportional time allocation (horizontal line)
  - [x] Same-bin equivalence with 'start' mode
  - [x] Default behavior unchanged
  - [x] Mass conservation (complex trajectory, with gaps)
  - [x] Layout compatibility (GraphLayout/PolygonLayout raise errors)
  - [x] Input validation (invalid values, type checking)
  - [x] Edge cases (single sample, empty arrays, outside environment)
  - [x] Integration (speed filtering, kernel smoothing)
  - [x] Accuracy (45° diagonal, vertical line)

**Deliverable**: P2 operations and field utilities

---

## Phase 5: Utilities & Documentation (Week 6)

**Goal**: Complete P3 operations and documentation.

### P3.12: Region Membership

- [x] Implement `Environment.region_membership()` in `environment.py` ✅
  - [x] Add `@check_fitted` decorator ✅
  - [x] Vectorized `shapely.covers()` operations ✅
  - [x] Support `include_boundary` parameter ✅
  - [x] Return (n_bins, n_regions) boolean array ✅
  - [ ] Optional caching by (env_id, regions_hash) (Deferred - performance is acceptable without caching)
- [x] Tests for `region_membership()` ✅
  - [x] Square region aligned to grid ✅
  - [x] Boundary inclusion behavior ✅

### P3.13: Distance Utilities ✅ COMPLETE

- [x] Implement `Environment.distance_to()` in `environment.py` ✅
  - [x] Add `@check_fitted` decorator ✅
  - [x] Handle region names (map to bins) ✅
  - [x] Euclidean vs geodesic metric ✅
  - [x] Vectorized Euclidean distance computation (broadcasting) ✅
  - [x] Wrapper around existing `distance_field()` ✅
  - [x] Units documentation in docstring ✅
  - [x] Working doctests ✅
- [x] Implement `Environment.rings()` in `environment.py` ✅
  - [x] Add `@check_fitted` decorator ✅
  - [x] BFS by layers using NetworkX ✅
  - [x] Return list of bin arrays per hop ✅
  - [x] Working doctests ✅
- [x] Tests for distance utilities (28 passing, 2 skipped) ✅
  - [x] Distance to single/multiple bins ✅
  - [x] Region-based distance (multi-source) ✅
  - [x] Euclidean vs geodesic metrics ✅
  - [x] Rings on grid: disjoint layers ✅
  - [x] Comprehensive validation tests ✅
  - [x] Edge cases (single bin, large hops) ✅
  - [x] Integration with distance_field ✅

### P3.14: Copy / Clone ✅ COMPLETE

- [x] Implement `Environment.copy()` in `environment.py`
  - [x] Support `deep=True` (deepcopy)
  - [x] Support `deep=False` (shallow)
  - [x] Clear all caches (KDTree, kernels)
- [x] Tests for `copy()`
  - [x] Deep copy: modify copy doesn't affect original
  - [x] Shallow copy: shared references
  - [x] Cache cleared after copy

### P3.15: Deterministic KDTree

- [ ] Update `spatial.map_points_to_bins()`
  - [ ] Add `max_distance` parameter
  - [ ] Add `max_distance_factor` parameter
  - [ ] Replace random subsampling with deterministic quantiles
- [ ] Tests for deterministic behavior
  - [ ] Repeated calls → identical results
  - [ ] Distance threshold behavior

### Documentation

- [ ] Create `docs/guides/spatial-analysis.md`
  - [ ] Computing occupancy and rate maps
  - [ ] Trajectory analysis (sequences, transitions)
  - [ ] Distance fields and navigation
  - [ ] Field smoothing and visualization
  - [ ] On-edge semantics (covers vs contains)
  - [ ] Unit conventions and bin volumes
- [ ] Create example scripts in `examples/`
  - [ ] Example: occupancy → rate map
  - [ ] Example: transition matrix analysis
  - [ ] Example: distance fields for navigation
- [ ] Update `__init__.py` with public API exports
  - [ ] Add `compute_diffusion_kernels`
  - [ ] Add field_ops functions
  - [ ] Update `__all__`

### Performance & Polish

- [ ] Performance benchmarks
  - [ ] Occupancy on 1M samples
  - [ ] Kernel computation for various n_bins
  - [ ] Transition matrix on long trajectories
- [ ] Final integration tests
  - [ ] End-to-end workflow: data → occupancy → smoothing → analysis
  - [ ] Multi-layout compatibility

**Deliverable**: Feature-complete environment operations with documentation

---

## Acceptance Criteria

### P0 (Required for v0.2.0)

- [ ] `occupancy()` implemented with full tests
- [ ] `bin_sequence()` implemented with full tests
- [ ] `transitions()` implemented with full tests
- [ ] `components()` and `reachable_from()` implemented with full tests
- [ ] `compute_diffusion_kernels()` in `kernels.py` with full tests
- [ ] `Environment.compute_kernel()` and `smooth()` wrappers
- [ ] All P0 functions work on all layout types or raise clear `NotImplementedError`
- [ ] At least one example script demonstrating core workflow

### P1 (Target for v0.3.0)

- [ ] `rebin()` implemented and tested
- [ ] `subset()` implemented and tested
- [ ] Layout compatibility documented for all operations

### P2 (Target for v0.4.0)

- [ ] `interpolate()` implemented and tested
- [ ] `field_ops.py` module complete with tests
- [ ] Linear occupancy method implemented

### P3 (Polish)

- [ ] `region_membership()` implemented and tested
- [ ] `distance_to()` and `rings()` implemented and tested
- [ ] `copy()` implemented and tested
- [ ] Deterministic KDTree improvements complete

### Documentation & Testing

- [ ] `docs/guides/spatial-analysis.md` created
- [ ] Each method has NumPy-style docstring with Examples
- [ ] On-edge semantics explicitly documented
- [ ] Unit conventions documented
- [ ] Edge cases tested: empty environment, single bin, disconnected graph
- [ ] Performance target achieved: occupancy on 1M samples < 1s

### Code Quality

- [ ] All methods use `@check_fitted` decorator
- [ ] Comprehensive input validation with diagnostic errors
- [ ] All caches use object identity keys
- [ ] KDTree operations deterministic by default
- [ ] All public methods have complete NumPy docstrings

---

## Progress Tracking

**Phase 1**: ⬜ Not Started (0/6 tasks)
**Phase 2**: ⬜ Not Started (0/4 sections)
**Phase 3**: ⬜ Not Started (0/3 sections)
**Phase 4**: ⬜ Not Started (0/3 sections)
**Phase 5**: ⬜ Not Started (0/5 sections)

**Overall**: 0% Complete

---

## Quick Reference

**Files to Create**:

- `src/neurospatial/kernels.py`
- `src/neurospatial/field_ops.py`
- `tests/test_kernels.py`
- `tests/test_field_ops.py`
- `docs/guides/spatial-analysis.md`
- Example scripts in `examples/`

**Files to Modify**:

- `src/neurospatial/environment.py` (add ~15 new methods)
- `src/neurospatial/spatial.py` (update `map_points_to_bins`)
- `src/neurospatial/__init__.py` (public API exports)
- `tests/test_environment.py` (add operation tests)

**Key Dependencies**:

- Phase 2 requires Phase 1 (kernel infrastructure)
- P2.11 (linear occupancy) requires P0.1 (basic occupancy)
- Documentation can proceed in parallel with implementation

---

**End of Tasks**
