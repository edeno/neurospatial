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

### P0.1: Occupancy / Dwell Time

- [ ] Implement `Environment.occupancy()` in `environment.py`
  - [ ] Add `@check_fitted` decorator
  - [ ] Input validation (array shapes, lengths, dtypes)
  - [ ] Handle empty arrays gracefully
  - [ ] Map points to bins using `spatial.map_points_to_bins`
  - [ ] Implement time accumulation logic
  - [ ] Add speed filtering (`min_speed` parameter)
  - [ ] Add gap handling (`max_gap` parameter)
  - [ ] Add optional kernel smoothing (`kernel_bandwidth` parameter)
  - [ ] Ensure mass conservation (occupancy.sum() = total_time)
- [ ] Tests for `occupancy()`
  - [ ] Synthetic L-shaped path with known durations
  - [ ] Sparse samples with large gaps
  - [ ] Speed filtering behavior
  - [ ] Kernel smoothing (verify mass conservation)
  - [ ] Edge case: all samples outside environment
  - [ ] Edge case: empty input arrays
  - [ ] Performance: 1M samples < 1 second

### P0.2: Bin Sequence / Runs

- [ ] Implement `Environment.bin_sequence()` in `environment.py`
  - [ ] Add `@check_fitted` decorator
  - [ ] Input validation
  - [ ] Map trajectory to bin indices
  - [ ] Implement deduplication (`dedup=True`)
  - [ ] Implement run-length encoding (`return_runs=True`)
  - [ ] Handle `outside_value=None` (drop samples)
- [ ] Tests for `bin_sequence()`
  - [ ] Known trajectory with expected bin sequence
  - [ ] Deduplication: [A,A,B,B,C] → [A,B,C]
  - [ ] Run boundaries (start/end indices)
  - [ ] Outside samples handling
  - [ ] Edge case: trajectory crossing boundary

### P0.3: Transitions / Adjacency Matrix

- [ ] Implement `Environment.transitions()` in `environment.py`
  - [ ] Add `@check_fitted` decorator
  - [ ] Input validation
  - [ ] Call `bin_sequence()` if bins not provided
  - [ ] Extract transition pairs with lag
  - [ ] Build adjacency set from graph
  - [ ] Filter non-adjacent transitions (`allow_teleports=False`)
  - [ ] Count transitions in sparse COO format
  - [ ] Convert to CSR matrix
  - [ ] Optional row normalization
- [ ] Tests for `transitions()`
  - [ ] 1D track: symmetric transitions
  - [ ] Grid navigation patterns
  - [ ] Teleport filtering (non-adjacent transitions)
  - [ ] Normalization: rows sum to 1
  - [ ] Lag parameter behavior

### P0.4: Connected Components / Reachability

- [ ] Implement `Environment.components()` in `environment.py`
  - [ ] Add `@check_fitted` decorator
  - [ ] Use `nx.connected_components()`
  - [ ] Sort by component size
  - [ ] Handle `largest_only=True`
- [ ] Implement `Environment.reachable_from()` in `environment.py`
  - [ ] Add `@check_fitted` decorator
  - [ ] BFS for `metric='hops'`
  - [ ] Dijkstra for `metric='geodesic'`
  - [ ] Optional radius cutoff
  - [ ] Return boolean mask
- [ ] Tests for connectivity
  - [ ] Graph with isolated island (2 components)
  - [ ] BFS with radius (hop neighborhoods)
  - [ ] Geodesic distance constraint
  - [ ] Edge case: disconnected graph

**Deliverable**: P0 operations complete with full test coverage

---

## Phase 3: Smoothing & Masking (Week 4)

**Goal**: Implement P1 operations (smooth, rebin, subset).

### P1.6: Field Smoothing

- [ ] Implement `Environment.smooth()` in `environment.py`
  - [ ] Add `@check_fitted` decorator
  - [ ] Validate field shape matches `n_bins`
  - [ ] Call `compute_kernel(bandwidth, mode)`
  - [ ] Return kernel @ field
- [ ] Tests for `smooth()`
  - [ ] Impulse spreading to neighbors
  - [ ] Mass conservation after smoothing
  - [ ] Edge preservation (no cross-component leakage)
  - [ ] Invalid field shape raises ValueError

### P1.7: Rebin (Grid Coarsening)

- [ ] Implement `Environment.rebin()` in `environment.py`
  - [ ] Add `@check_fitted` decorator
  - [ ] Check layout is `RegularGridLayout`
  - [ ] Raise `NotImplementedError` for non-grids
  - [ ] Parse `factor` (int or tuple)
  - [ ] Compute new grid edges
  - [ ] Aggregate bins (sum/mean)
  - [ ] Build new connectivity
  - [ ] Return new Environment
- [ ] Tests for `rebin()`
  - [ ] Constant field preservation (method='sum')
  - [ ] 10x10 → 5x5 grid (factor=2)
  - [ ] Bin position correctness
  - [ ] Non-grid layout raises error

### P1.8: Subset / Crop

- [ ] Implement `Environment.subset()` in `environment.py`
  - [ ] Add `@check_fitted` decorator
  - [ ] Validate exactly one of {bins, region_names, polygon}
  - [ ] Build boolean mask from input
  - [ ] Extract subgraph
  - [ ] Renumber nodes to [0, n'-1]
  - [ ] Create new layout with subset
  - [ ] Return new Environment (regions dropped)
- [ ] Tests for `subset()`
  - [ ] Crop 10x10 grid to left half → 5x10
  - [ ] Polygon selection (bin centers inside)
  - [ ] Connectivity preserved
  - [ ] Node renumbering correctness

**Deliverable**: P1 operations with layout compatibility checks

---

## Phase 4: Interpolation & Field Utilities (Week 5)

**Goal**: Implement P2 operations (interpolation, field math).

### P2.9: Field Interpolation

- [ ] Implement `Environment.interpolate()` in `environment.py`
  - [ ] Add `@check_fitted` decorator
  - [ ] Input validation
  - [ ] Nearest neighbor mode (KDTree, all layouts)
  - [ ] Linear mode (grids only)
    - [ ] Check layout is `RegularGridLayout`
    - [ ] Use `scipy.interpolate.RegularGridInterpolator`
  - [ ] Handle out-of-bounds → NaN
- [ ] Tests for `interpolate()`
  - [ ] Known plane field f(x,y) = x + 2*y (linear mode)
  - [ ] Nearest mode on irregular graph
  - [ ] Out-of-bounds points → NaN

### P2.10: Field Math Utilities

- [ ] Create `src/neurospatial/field_ops.py`
  - [ ] Implement `normalize_field(field, eps=1e-12)`
  - [ ] Implement `clamp(field, lo=0.0, hi=inf)`
  - [ ] Implement `combine_fields(fields, weights, mode='mean')`
  - [ ] Implement `divergence(p, q, kind='kl', eps=1e-12)`
    - [ ] KL divergence
    - [ ] JS divergence
    - [ ] Cosine distance
  - [ ] All functions: shape validation, NumPy docstrings
- [ ] Create `tests/test_field_ops.py`
  - [ ] Test KL/JS identities (self vs self = 0)
  - [ ] Test JS symmetry
  - [ ] Test cosine bounds [0, 2]
  - [ ] Test normalization preserves shape
  - [ ] Test combine_fields with weights

### P2.11: Linear Occupancy (Grid Enhancement)

- [ ] Add `time_allocation` parameter to `occupancy()`
  - [ ] Support `time_allocation='start'` (default)
  - [ ] Support `time_allocation='linear'` (grids only)
  - [ ] Ray-grid intersection logic
  - [ ] Proportional time splitting across bins
- [ ] Tests for linear occupancy
  - [ ] Diagonal trajectory across 4 cells
  - [ ] Compare linear vs start allocation
  - [ ] Non-grid layout raises error

**Deliverable**: P2 operations and field utilities

---

## Phase 5: Utilities & Documentation (Week 6)

**Goal**: Complete P3 operations and documentation.

### P3.12: Region Membership

- [ ] Implement `Environment.region_membership()` in `environment.py`
  - [ ] Add `@check_fitted` decorator
  - [ ] Vectorized `shapely.covers()` operations
  - [ ] Support `include_boundary` parameter
  - [ ] Return (n_bins, n_regions) boolean array
  - [ ] Optional caching by (env_id, regions_hash)
- [ ] Tests for `region_membership()`
  - [ ] Square region aligned to grid
  - [ ] Boundary inclusion behavior

### P3.13: Distance Utilities

- [ ] Implement `Environment.distance_to()` in `environment.py`
  - [ ] Add `@check_fitted` decorator
  - [ ] Handle region names (map to bins)
  - [ ] Euclidean vs geodesic metric
  - [ ] Wrapper around existing `distance_field()`
- [ ] Implement `Environment.rings()` in `environment.py`
  - [ ] Add `@check_fitted` decorator
  - [ ] BFS by layers
  - [ ] Return list of bin arrays per hop
- [ ] Tests for distance utilities
  - [ ] 1D line: distances = index differences
  - [ ] Rings on grid: symmetric patterns
  - [ ] Region-based distance (multi-source)

### P3.14: Copy / Clone

- [ ] Implement `Environment.copy()` in `environment.py`
  - [ ] Support `deep=True` (deepcopy)
  - [ ] Support `deep=False` (shallow)
  - [ ] Clear all caches (KDTree, kernels)
- [ ] Tests for `copy()`
  - [ ] Deep copy: modify copy doesn't affect original
  - [ ] Shallow copy: shared references
  - [ ] Cache cleared after copy

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
