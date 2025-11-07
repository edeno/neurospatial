# neurospatial v0.3.0 Implementation Tasks

**Timeline**: 14 weeks
**Authority**: opexebo, neurocode, ecology packages
**Success Criteria**: All tests pass, mypy zero errors, >95% coverage

---

## Milestone 0: Core Spike/Field Primitives (Weeks 1-2)

**Goal**: Implement foundational primitives for spike→field conversion and reward fields

**Note**: Batch operations (`smooth_batch()`, `normalize_fields()`) deferred to v0.3.1 - following "make it work, make it right, make it fast" principle

### 0.1 Spike → Field Conversion + Convenience Function (Week 1, Days 1-4)

**Prerequisites**:

- [x] Add `return_seconds` parameter to `env.occupancy()` method
  - [x] Location: `src/neurospatial/environment/trajectory.py`
  - [x] Add boolean parameter `return_seconds: bool = True` (default True for backward compatibility)
  - [x] If True, weight bins by time differences (`np.diff(times)`)
  - [x] If False, return sample counts (unweighted, each interval = 1)
  - [x] Update docstring with new parameter
  - [x] Add tests: 5 comprehensive tests in `tests/test_occupancy.py::TestOccupancyReturnSeconds`
  - [x] Update `_protocols.py` to include parameter in Protocol definition
  - [x] Verify mypy passes (zero errors)

**Implementation**:

- [x] Create `src/neurospatial/spike_field.py` module
- [x] Implement `spikes_to_field(env, spike_times, times, positions, *, min_occupancy_seconds=0.0)`
  - [x] Step 0: Validate inputs (times/positions same length, handle empty spikes)
  - [x] Step 1: Filter spikes to valid time range with warning
  - [x] Step 2: Compute occupancy using `env.occupancy(times, positions, return_seconds=True)`
  - [x] Step 3: Interpolate spike positions (handle 1D correctly!)
    - [x] Check `positions.ndim == 1` or `positions.shape[1] == 1`
    - [x] For 1D: `np.interp(spike_times, times, positions)[:, None]`
    - [x] For multi-D: `np.column_stack([np.interp(...) for dim in ...])`
  - [x] Step 4: Assign spikes to bins using `env.bin_at(spike_positions)`
  - [x] Step 5: Filter out-of-bounds spikes with warning
  - [x] Step 6: Count spikes per bin with `np.bincount()`
  - [x] Step 7: Normalize by occupancy where valid
  - [x] Step 8: Set low-occupancy bins to NaN (optional, default min_occupancy_seconds=0.0)
  - [x] Add comprehensive NumPy-style docstring with examples
  - [x] Add type hints: `NDArray[np.float64]`
  - [x] Handle edge cases: empty spikes, all spikes out-of-bounds, zero occupancy
- [x] Implement `compute_place_field()` convenience function
  - [x] Combine `spikes_to_field()` + `env.smooth()`
  - [x] Parameters: same as `spikes_to_field` + `smoothing_bandwidth: float | None`
  - [x] If `smoothing_bandwidth` is None, skip smoothing
  - [x] Add docstring showing equivalence to two-step workflow
- [x] Export in public API: `src/neurospatial/__init__.py`
  - [x] Add: `from neurospatial.spike_field import spikes_to_field, compute_place_field`
  - [x] Update `__all__` list

**Testing**:

- [x] Create `tests/test_spike_field.py`
- [x] Test: `test_spikes_to_field_synthetic()` - known spike rate → expected field
- [x] Test: `test_spikes_to_field_min_occupancy()` - low occupancy bins → NaN
- [x] Test: `test_spikes_to_field_empty_spikes()` - empty spike train → all zeros
- [x] Test: `test_spikes_to_field_out_of_bounds_time()` - spikes outside time range → warning + filtered
- [x] Test: `test_spikes_to_field_out_of_bounds_space()` - spikes outside environment → warning + filtered
- [x] Test: `test_spikes_to_field_1d_trajectory()` - handles 1D positions correctly
- [x] Test: `test_spikes_to_field_1d_no_column_dimension()` - handles bare 1D arrays (n,)
- [x] Test: `test_spikes_to_field_nan_occupancy()` - all zero occupancy → all NaN with warning
- [x] Test: `test_spikes_to_field_matches_manual()` - compare with manual computation
- [x] Test: `test_spikes_to_field_parameter_order()` - verify env comes first
- [x] Test: `test_spikes_to_field_validation()` - negative min_occupancy, mismatched lengths
- [x] Test: `test_compute_place_field_with_smoothing()` - matches spikes_to_field + smooth
- [x] Test: `test_compute_place_field_no_smoothing()` - matches spikes_to_field when bandwidth=None
- [x] Test: `test_compute_place_field_parameter_order()` - verify parameter consistency
- [x] Run: `uv run pytest tests/test_spike_field.py -v` (14 tests passed)

**Type Checking**:

- [x] Add `TYPE_CHECKING` guard for Environment import
- [x] Verify mypy passes: `uv run mypy src/neurospatial/spike_field.py`
- [x] No `type: ignore` comments allowed

**Effort**: 4 days (+1 day for prerequisite, fixes, and convenience function)

---

### 0.2 Reward Field Primitives (Week 1, Days 5-6 + Week 2, Day 1)

**Implementation**:

- [x] Create `src/neurospatial/reward.py` module
- [x] Implement `region_reward_field(env, region_name, *, reward_value=1.0, decay="constant", bandwidth=None)`
  - [x] Validate region exists in `env.regions`
  - [x] Get region mask: `region_mask = regions_to_mask(env, [region_name])`
  - [x] If `decay == "constant"`: binary reward (reward_value inside, 0 outside)
  - [x] If `decay == "linear"`: linear decay from boundary using `distance_field()`
  - [x] If `decay == "gaussian"`:
    - [x] Validate `bandwidth` is provided (raise ValueError if None)
    - [x] Create indicator field: `indicator = np.where(region_mask, 1.0, 0.0)`
    - [x] Smooth: `smoothed = env.smooth(indicator, bandwidth)`
    - [x] CRITICAL FIX: Scale by max IN REGION: `max_in_region = smoothed[region_mask].max()`
    - [x] Return: `smoothed / max_in_region * reward_value`
  - [x] Add comprehensive NumPy-style docstring with RL references
  - [x] Type hints: `Literal["constant", "linear", "gaussian"]` for decay
- [x] Implement `goal_reward_field(env, goal_bins, *, decay="exponential", scale=1.0, max_distance=None)`
  - [x] Validate goal_bins are valid indices
  - [x] Convert scalar to 1D array: `goal_bins = np.asarray(goal_bins); if goal_bins.ndim == 0: goal_bins = goal_bins[None]`
  - [x] Compute distances: `distances = distance_field(env.connectivity, sources=goal_bins.tolist())`
  - [x] If `decay == "linear"`: `reward = scale * np.maximum(0, 1 - distances / max_distance)`
  - [x] If `decay == "exponential"`: validate `scale > 0`, then `reward = scale * np.exp(-distances / scale)`
  - [x] If `decay == "inverse"`: `reward = scale / (1 + distances)`
  - [x] Add comprehensive NumPy-style docstring with RL references
  - [x] Type hints: `Literal["linear", "exponential", "inverse"]` for decay
- [x] Export in public API: `src/neurospatial/__init__.py`
  - [x] Add: `from neurospatial.reward import region_reward_field, goal_reward_field`
  - [x] Update `__all__` list

**Testing**:

- [x] Create `tests/test_reward.py`
- [x] Test: `test_region_reward_field_constant()` - binary reward in region
- [x] Test: `test_region_reward_field_linear()` - linear decay from boundary
- [x] Test: `test_region_reward_field_gaussian()` - smooth falloff, peak maintains value IN REGION (critical fix)
- [x] Test: `test_region_reward_field_validation()` - bandwidth required for Gaussian, region must exist
- [x] Test: `test_region_reward_field_parameter_naming()` - verify `decay` parameter (not `falloff`)
- [x] Test: `test_goal_reward_field_exponential()` - exponential decay from goal
- [x] Test: `test_goal_reward_field_linear()` - reaches zero at max distance
- [x] Test: `test_goal_reward_field_inverse()` - inverse distance formula
- [x] Test: `test_goal_reward_field_multiple_goals()` - nearest goal dominates
- [x] Test: `test_goal_reward_field_validation()` - invalid goal bins raise error, scale positive for exponential
- [x] Test: `test_goal_reward_field_parameter_naming()` - verify `decay` parameter (not `kind`)
- [x] Run: `uv run pytest tests/test_reward.py -v` (15/15 tests pass)

**Type Checking**:

- [x] Add `TYPE_CHECKING` guard for Environment import
- [x] Use `Literal` types for decay parameters
- [x] Verify mypy passes: `uv run mypy src/neurospatial/reward.py` (zero errors)
- [x] No `type: ignore` comments allowed

**Effort**: 3 days

---

### 0.3 Documentation (Week 2, Days 2-3)

**Documentation**:

- [x] Create `docs/user-guide/spike-field-primitives.md`
  - [x] Section: Converting spike trains to spatial fields
  - [x] Section: Why occupancy normalization matters (place field analysis standard)
  - [x] Section: Parameter order (env first, matches existing API)
  - [x] Section: `compute_place_field()` convenience function for one-liner workflows
  - [x] Section: Min occupancy threshold (best practices: 0.5 seconds typical)
  - [x] Section: Edge case handling (empty spikes, out-of-bounds, NaN)
  - [x] Include code examples and visualizations
  - [x] Note: Batch operations deferred to v0.3.1
- [x] Create or update `docs/user-guide/rl-primitives.md`
  - [x] Section: Reward field generation from regions
  - [x] Section: Reward shaping strategies (potential-based, distance-based)
  - [x] Section: Decay parameter naming (consistent across functions)
  - [x] Section: Distance-based rewards (exponential, linear, inverse)
  - [x] Section: Gaussian falloff rescaling (uses max IN REGION)
  - [x] Section: Cautions about reward shaping (Ng et al. 1999 reference)
  - [x] Include RL-specific examples (goal-directed navigation)

**Example Notebook**:

- [ ] Create `examples/00_spike_field_basics.ipynb`
  - [ ] Example 1: Convert spike train → firing rate map
    - [ ] Generate synthetic data (trajectory + spike times)
    - [ ] Create environment
    - [ ] Compute firing rate with `spikes_to_field(env, spike_times, times, positions)` (correct order!)
    - [ ] Show `compute_place_field()` convenience function
    - [ ] Visualize: occupancy, spike counts, firing rate
    - [ ] Demonstrate min occupancy threshold effect
  - [ ] Example 2: Create reward field for RL
    - [ ] Define goal region
    - [ ] Create constant reward field (binary)
    - [ ] Create linear decay reward (reaches zero at boundaries)
    - [ ] Create Gaussian falloff reward (smooth potential field)
    - [ ] Create exponential decay from goal bins
    - [ ] Visualize all variations side-by-side
    - [ ] Demonstrate consistent parameter naming (`decay` across all functions)
  - [ ] Add explanatory markdown cells throughout
  - [ ] Add best practices notes
  - [ ] Add cautions about reward shaping
  - [ ] Note: Batch operations examples will be added in v0.3.1

**Testing**:

- [ ] Run all Phase 0 tests with updated signatures
- [ ] Verify coverage: >95% for new code
- [ ] Run: `uv run pytest tests/test_spike_field.py tests/test_reward.py --cov=src/neurospatial --cov-report=term-missing`
- [ ] Target: >95% coverage for new code
- [ ] Run example notebook: `uv run jupyter nbconvert --execute examples/00_spike_field_basics.ipynb`
- [ ] Verify all cells execute without errors
- [ ] Verify visualizations render correctly

**Effort**: 2 days (same as original plan)

---

### Milestone 0 Success Criteria

- [ ] `spikes_to_field()` uses correct parameter order (env first, times/positions match existing API)
- [ ] `compute_place_field()` convenience function works for one-liner workflows
- [ ] Min occupancy threshold correctly filters unreliable bins (→ NaN)
- [ ] Spike interpolation handles 1D and multi-dimensional trajectories correctly
- [ ] Input validation comprehensive (empty spikes, out-of-range, NaN handling)
- [ ] `region_reward_field()` supports all three decay types with correct parameter name (`decay`)
- [ ] `goal_reward_field()` supports all three decay functions with consistent naming (`decay`)
- [ ] Gaussian falloff rescaling uses max IN REGION (not global max)
- [ ] All tests pass with >95% coverage for new code
- [ ] Zero mypy errors (no `type: ignore` comments)
- [ ] Example notebook demonstrates all primitives with clear visualizations
- [ ] Documentation complete and cross-referenced

**Deferred to v0.3.1**: Batch operations (`smooth_batch()`, `normalize_fields()`)

---

## Milestone 1: Core Differential Operators (Weeks 3-5)

**Goal**: Implement weighted differential operator infrastructure for graph signal processing

### 1.1 Differential Operator Matrix (Week 1)

**Implementation**:

- [ ] Create `src/neurospatial/differential.py` module
- [ ] Implement `compute_differential_operator(env)` function
  - [ ] Extract edge data from `env.connectivity` graph
  - [ ] Compute sqrt of edge weights (distances)
  - [ ] Build sparse CSC matrix (n_bins × n_edges)
  - [ ] Add NumPy-style docstring with PyGSP reference
  - [ ] Add type hints: `-> sparse.csc_matrix`
- [ ] Add `differential_operator` cached property to Environment
  - [ ] Location: `src/neurospatial/environment/core.py`
  - [ ] Use `@cached_property` decorator
  - [ ] Import from `neurospatial.differential`
  - [ ] Add NumPy-style docstring

**Testing**:

- [ ] Create `tests/test_differential.py`
- [ ] Test: `test_differential_operator_shape()` - verify (n_bins, n_edges)
- [ ] Test: `test_laplacian_from_differential()` - D @ D.T == nx.laplacian_matrix()
- [ ] Test: `test_differential_operator_caching()` - repeated calls return same object
- [ ] Test: Edge cases (single node, disconnected graph)
- [ ] Run: `uv run pytest tests/test_differential.py -v`

**Type Checking**:

- [ ] Add `TYPE_CHECKING` guard for Environment import
- [ ] Verify mypy passes: `uv run mypy src/neurospatial/differential.py`
- [ ] No `type: ignore` comments allowed

**Effort**: 3 days

---

### 1.2 Gradient Operator (Week 2)

**Implementation**:

- [ ] Add `gradient(field, env)` function to `differential.py`
  - [ ] Implement: `return env.differential_operator.T @ field`
  - [ ] Add validation: check `field.shape == (env.n_bins,)`
  - [ ] Add NumPy-style docstring with examples
  - [ ] Add type hints: `NDArray[np.float64]`
- [ ] Export in public API: `src/neurospatial/__init__.py`
  - [ ] Add: `from neurospatial.differential import gradient`
  - [ ] Update `__all__` list

**Testing**:

- [ ] Test: `test_gradient_shape()` - output shape (n_edges,)
- [ ] Test: `test_gradient_constant_field()` - gradient of constant = 0
- [ ] Test: `test_gradient_linear_field_regular_grid()` - constant gradient for linear field
- [ ] Test: `test_gradient_validation()` - wrong shape raises ValueError
- [ ] Run: `uv run pytest tests/test_differential.py::test_gradient -v`

**Documentation**:

- [ ] Add example in docstring: distance field gradient
- [ ] Cross-reference to `divergence()` function

**Effort**: 2 days

---

### 1.3 Divergence Operator (Week 2)

**Implementation**:

- [ ] Rename existing `divergence()` to `kl_divergence()` in `field_ops.py`
  - [ ] Update function name and docstring
  - [ ] Add note: "Renamed from divergence() in v0.3.0"
  - [ ] Find all internal uses: `uv run grep -r "divergence(" src/`
  - [ ] Update all internal references
- [ ] Add new `divergence(edge_field, env)` to `differential.py`
  - [ ] Implement: `return env.differential_operator @ edge_field`
  - [ ] Add validation: check `edge_field.shape == (n_edges,)`
  - [ ] Add NumPy-style docstring with examples
- [ ] Export in public API: `src/neurospatial/__init__.py`
  - [ ] Add: `from neurospatial.differential import divergence`
  - [ ] Add: `from neurospatial.field_ops import kl_divergence`

**Testing**:

- [ ] Test: `test_divergence_gradient_is_laplacian()` - div(grad(f)) == Laplacian(f)
- [ ] Test: `test_divergence_shape()` - output shape (n_bins,)
- [ ] Test: `test_kl_divergence_renamed()` - old function still works
- [ ] Run: `uv run pytest tests/test_differential.py -v`
- [ ] Run: `uv run pytest tests/test_field_ops.py -v` (ensure kl_divergence tests pass)

**Breaking Changes**:

- [ ] Update CHANGELOG.md: note divergence → kl_divergence rename
- [ ] No migration guide needed (no current users)

**Effort**: 2 days

---

### 1.4 Documentation & Examples (Week 3)

**Documentation**:

- [ ] Create `docs/user-guide/differential-operators.md`
  - [ ] Section: What are differential operators?
  - [ ] Section: Gradient (scalar field → edge field)
  - [ ] Section: Divergence (edge field → scalar field)
  - [ ] Section: Laplacian (composition: div ∘ grad)
  - [ ] Section: Mathematical background (graph signal processing)
  - [ ] Section: When to use (RL value gradients, flow fields)
  - [ ] Include formulas with LaTeX notation

**Example Notebook**:

- [ ] Create `examples/09_differential_operators.ipynb`
  - [ ] Example 1: Gradient of distance field (goal-directed navigation)
  - [ ] Example 2: Divergence of flow field (source/sink detection)
  - [ ] Example 3: Laplacian smoothing (compare to env.smooth())
  - [ ] Example 4: RL successor representation (replay analysis)
  - [ ] Add visualizations with matplotlib
  - [ ] Add explanatory markdown cells

**Testing**:

- [ ] Run notebook: `uv run jupyter nbconvert --execute examples/09_differential_operators.ipynb`
- [ ] Verify all cells execute without errors

**Effort**: 5 days

---

## Milestone 2: Basic Signal Processing Primitives (Weeks 6-8)

**Goal**: Implement foundational spatial signal processing operations

### 2.1 neighbor_reduce (Week 4)

**Implementation**:

- [ ] Create `src/neurospatial/primitives.py` module
- [ ] Implement `neighbor_reduce(field, env, *, op='mean', weights=None, include_self=False)`
  - [ ] Support ops: 'sum', 'mean', 'max', 'min', 'std'
  - [ ] Implement weighted aggregation if weights provided
  - [ ] Add `include_self` logic
  - [ ] Optimize with vectorization where possible
  - [ ] Add NumPy-style docstring with examples
  - [ ] Add type hints with Literal for op parameter
- [ ] Export in public API: `src/neurospatial/__init__.py`

**Testing**:

- [ ] Create `tests/test_primitives.py`
- [ ] Test: `test_neighbor_reduce_mean_regular_grid()` - verify neighbor averaging
- [ ] Test: `test_neighbor_reduce_include_self()` - verify self-inclusion changes result
- [ ] Test: `test_neighbor_reduce_weights()` - verify distance-weighted aggregation
- [ ] Test: `test_neighbor_reduce_operations()` - test all ops (sum, mean, max, min, std)
- [ ] Test: Edge cases (isolated nodes, boundary bins)
- [ ] Run: `uv run pytest tests/test_primitives.py::test_neighbor_reduce -v`

**Type Checking**:

- [ ] Use `Literal['sum', 'mean', 'max', 'min', 'std']` for op parameter
- [ ] Verify mypy: `uv run mypy src/neurospatial/primitives.py`

**Effort**: 3 days

---

### 2.2 convolve (Week 5-6)

**Implementation**:

- [ ] Add `convolve(field, kernel, env, *, normalize=True)` to `primitives.py`
  - [ ] Support callable kernel: `distance -> weight`
  - [ ] Support precomputed kernel matrix (n_bins × n_bins)
  - [ ] Implement normalization (weights sum to 1)
  - [ ] Handle NaN values in field
  - [ ] Add NumPy-style docstring with examples
  - [ ] Add type hints: `Callable[[NDArray], float] | NDArray`

**Testing**:

- [ ] Test: `test_convolve_box_kernel()` - uniform kernel within radius
- [ ] Test: `test_convolve_mexican_hat()` - difference of Gaussians
- [ ] Test: `test_convolve_precomputed_kernel()` - pass kernel matrix directly
- [ ] Test: `test_convolve_normalize()` - verify normalization
- [ ] Test: `test_convolve_nan_handling()` - NaN values don't propagate
- [ ] Test: Compare with env.smooth() for Gaussian kernel
- [ ] Run: `uv run pytest tests/test_primitives.py::test_convolve -v`

**Documentation**:

- [ ] Add examples in docstring: box kernel, Mexican hat, custom kernels
- [ ] Cross-reference to `env.smooth()` and `env.compute_kernel()`

**Effort**: 3 days

---

### 2.3 Documentation (Week 6)

**Documentation**:

- [ ] Create `docs/user-guide/signal-processing-primitives.md`
  - [ ] Section: neighbor_reduce for local aggregation
  - [ ] Section: convolve for custom filtering
  - [ ] Section: Comparison with env.smooth()
  - [ ] Section: Use cases (coherence, custom kernels)

**Example Notebook**:

- [ ] Add examples to `examples/09_differential_operators.ipynb` or create new notebook
  - [ ] Example: Compute coherence using neighbor_reduce
  - [ ] Example: Box filter for occupancy thresholding
  - [ ] Example: Mexican hat edge detection

**Testing**:

- [ ] Run all primitives tests: `uv run pytest tests/test_primitives.py -v`
- [ ] Verify coverage: `uv run pytest tests/test_primitives.py --cov=src/neurospatial/primitives.py`

**Effort**: 2 days

---

## Milestone 3: Core Metrics Module (Weeks 8.5-10)

**Goal**: Provide standard neuroscience metrics as convenience wrappers

### 3.1 Place Field Metrics (Week 7)

**Implementation**:

- [ ] Create `src/neurospatial/metrics/` package
- [ ] Create `src/neurospatial/metrics/__init__.py`
- [ ] Create `src/neurospatial/metrics/place_fields.py`
- [ ] Implement `detect_place_fields(firing_rate, env, *, threshold=0.2, min_size=None, max_mean_rate=10.0, detect_subfields=True)`
  - [ ] Iterative peak-based detection (neurocode approach)
  - [ ] Interneuron exclusion (10 Hz threshold, vandermeerlab)
  - [ ] Subfield discrimination (recursive threshold)
  - [ ] Return list of NDArray[np.int64] (bin indices per field)
  - [ ] Add comprehensive NumPy-style docstring with references
- [ ] Implement `field_size(field_bins, env)` - area in physical units
- [ ] Implement `field_centroid(firing_rate, field_bins, env)` - center of mass
- [ ] Implement `skaggs_information(firing_rate, occupancy, *, base=2.0)` - bits/spike
- [ ] Implement `sparsity(firing_rate, occupancy)` - Skaggs et al. 1996
- [ ] Implement `field_stability(rate_map_1, rate_map_2, *, method='pearson')`

**Testing**:

- [ ] Create `tests/metrics/test_place_fields.py`
- [ ] Test: `test_detect_place_fields_synthetic()` - known field positions
- [ ] Test: `test_detect_place_fields_subfields()` - coalescent fields
- [ ] Test: `test_detect_place_fields_interneuron_exclusion()` - high rate excluded
- [ ] Test: `test_field_size()` - verify area calculation
- [ ] Test: `test_field_centroid()` - weighted center of mass
- [ ] Test: `test_skaggs_information()` - verify formula
- [ ] Test: `test_sparsity()` - verify formula, range [0, 1]
- [ ] Test: `test_field_stability()` - Pearson and Spearman
- [ ] Run: `uv run pytest tests/metrics/test_place_fields.py -v`

**Validation**:

- [ ] Compare with neurocode FindPlaceFields.m output (if available)
- [ ] Verify spatial information matches opexebo/buzcode

**Effort**: 3 days

---

### 3.2 Population Metrics (Week 7)

**Implementation**:

- [ ] Create `src/neurospatial/metrics/population.py`
- [ ] Implement `population_coverage(all_place_fields, n_bins)` - fraction covered
- [ ] Implement `field_density_map(all_place_fields, n_bins)` - overlapping fields
- [ ] Implement `count_place_cells(spatial_information, threshold=0.5)` - count exceeding threshold
- [ ] Implement `field_overlap(field_bins_i, field_bins_j)` - Jaccard coefficient
- [ ] Implement `population_vector_correlation(population_matrix)` - correlation matrix

**Testing**:

- [ ] Create `tests/metrics/test_population.py`
- [ ] Test: `test_population_coverage()` - verify fraction calculation
- [ ] Test: `test_field_density_map()` - count overlaps correctly
- [ ] Test: `test_count_place_cells()` - threshold filtering
- [ ] Test: `test_field_overlap()` - Jaccard index
- [ ] Test: `test_population_vector_correlation()` - correlation matrix shape
- [ ] Run: `uv run pytest tests/metrics/test_population.py -v`

**Effort**: 2 days

---

### 3.3 Boundary Cell Metrics (Week 8)

**Implementation**:

- [ ] Create `src/neurospatial/metrics/boundary_cells.py`
- [ ] Implement `border_score(firing_rate, env, *, threshold=0.3, min_area=200)`
  - [ ] Segment field at 30% of peak (Solstad et al. 2008)
  - [ ] Compute wall contact ratio for each wall
  - [ ] Compute max contact ratio (cM)
  - [ ] Compute firing-rate-weighted distance to walls (d)
  - [ ] Border score: (cM - d) / (cM + d)
  - [ ] Add comprehensive NumPy docstring with TSToolbox_Utils reference
- [ ] Implement `boundary_vector_tuning(firing_rate, env, positions)` (optional)
  - [ ] Preferred distance to boundary
  - [ ] Preferred allocentric direction to boundary

**Testing**:

- [ ] Create `tests/metrics/test_boundary_cells.py`
- [ ] Test: `test_border_score_synthetic()` - known border cell
- [ ] Test: `test_border_score_non_border()` - central field returns low score
- [ ] Test: `test_border_score_match_opexebo()` - compare with opexebo if available
- [ ] Test: `test_border_score_match_tstoolbox()` - compare with MATLAB if possible
- [ ] Run: `uv run pytest tests/metrics/test_boundary_cells.py -v`

**Validation**:

- [ ] Match TSToolbox_Utils Compute_BorderScore.m output
- [ ] Match opexebo.analysis.border_score() output

**Effort**: 2 days

---

### 3.4 Documentation (Week 8)

**Documentation**:

- [ ] Create `docs/user-guide/neuroscience-metrics.md`
  - [ ] Section: Place field detection and metrics
  - [ ] Section: Population-level analyses
  - [ ] Section: Boundary cell metrics
  - [ ] Include formulas and references
  - [ ] Cross-reference to opexebo, neurocode

**Example Notebooks**:

- [ ] Create `examples/10_place_field_analysis.ipynb`
  - [ ] Load example data (generate synthetic or use real)
  - [ ] Compute firing rate map with occupancy
  - [ ] Detect place fields
  - [ ] Compute Skaggs information, sparsity
  - [ ] Visualize fields on environment
- [ ] Create `examples/11_boundary_cell_analysis.ipynb`
  - [ ] Generate synthetic border cell
  - [ ] Compute border score
  - [ ] Visualize wall contact ratios

**Testing**:

- [ ] Run all metrics tests: `uv run pytest tests/metrics/ -v`
- [ ] Verify coverage: `uv run pytest tests/metrics/ --cov=src/neurospatial/metrics/`
- [ ] Run notebooks: verify all cells execute

**Effort**: 2 days

---

## Milestone 4: Trajectory Metrics & Behavioral Segmentation (Weeks 10.5-13)

**Goal**: Implement trajectory characterization and automatic behavioral epoch detection

### 4.1 Trajectory Metrics (Week 9)

**Implementation**:

- [ ] Create `src/neurospatial/metrics/trajectory.py`
- [ ] Implement `compute_turn_angles(trajectory_bins, env)`
  - [ ] Compute angles between consecutive movement vectors
  - [ ] Use env.bin_centers for position lookup
  - [ ] Handle stationary periods (skip if no movement)
  - [ ] Return angles in radians [-π, π]
  - [ ] Add NumPy docstring with Traja reference
- [ ] Implement `compute_step_lengths(trajectory_bins, env)`
  - [ ] Use env.distance_between() for graph distances
  - [ ] Handle consecutive duplicates (distance = 0)
  - [ ] Return array of step lengths
- [ ] Implement `compute_home_range(trajectory_bins, *, percentile=95.0)`
  - [ ] Compute occupancy from trajectory
  - [ ] Sort bins by occupancy
  - [ ] Select bins containing X% of time
  - [ ] Return bin indices in home range
- [ ] Implement `mean_square_displacement(trajectory_bins, times, env, *, max_tau=None)`
  - [ ] Compute MSD(τ) for lag times τ
  - [ ] Use env.distance_between() for graph distances
  - [ ] Return (tau_values, msd_values)
  - [ ] Add docstring explaining MSD ~ τ^α classification

**Testing**:

- [ ] Create `tests/metrics/test_trajectory.py`
- [ ] Test: `test_turn_angles_straight_line()` - angles ~ 0
- [ ] Test: `test_turn_angles_circle()` - constant turning
- [ ] Test: `test_step_lengths()` - verify graph distances
- [ ] Test: `test_home_range()` - verify percentile calculation
- [ ] Test: `test_msd_diffusion()` - verify MSD ~ τ for random walk
- [ ] Run: `uv run pytest tests/metrics/test_trajectory.py -v`

**Validation**:

- [ ] Compare turn angles with Traja output on synthetic trajectory
- [ ] Compare MSD with yupi output on random walk

**Effort**: 3 days

---

### 4.2 Region-Based Segmentation (Week 10, Days 1-3)

**Implementation**:

- [ ] Create `src/neurospatial/segmentation/` package
- [ ] Create `src/neurospatial/segmentation/__init__.py`
- [ ] Create `src/neurospatial/segmentation/regions.py`
- [ ] Implement `detect_region_crossings(trajectory_bins, times, region, env, direction='both')`
  - [ ] Check which bins are in region
  - [ ] Detect entry/exit transitions
  - [ ] Return list of Crossing objects (time, direction)
- [ ] Implement `detect_runs_between_regions(trajectory_positions, times, env, *, source, target, min_duration=0.5, max_duration=10.0, velocity_threshold=None)`
  - [ ] Detect source region exits
  - [ ] Track trajectory until target entry or timeout
  - [ ] Filter by duration and velocity
  - [ ] Return list of Run objects (start_time, end_time, bins, success)
- [ ] Implement `segment_by_velocity(trajectory_positions, times, threshold, *, min_duration=0.5, hysteresis=2.0, smooth_window=0.2)`
  - [ ] Compute velocity from positions
  - [ ] Apply hysteresis thresholding
  - [ ] Filter by minimum duration
  - [ ] Return IntervalSet (or list of tuples if pynapple unavailable)

**Testing**:

- [ ] Create `tests/segmentation/test_regions.py`
- [ ] Test: `test_detect_region_crossings()` - synthetic trajectory with known crossings
- [ ] Test: `test_detect_runs_between_regions_success()` - successful runs
- [ ] Test: `test_detect_runs_between_regions_timeout()` - failed runs
- [ ] Test: `test_segment_by_velocity()` - movement vs rest
- [ ] Run: `uv run pytest tests/segmentation/test_regions.py -v`

**Effort**: 3 days

---

### 4.3 Lap Detection (Week 10, Days 4-5)

**Implementation**:

- [ ] Create `src/neurospatial/segmentation/laps.py`
- [ ] Implement `detect_laps(trajectory_bins, times, env, *, method='auto', min_overlap=0.8, direction='both')`
  - [ ] Method 'auto': detect template from first 10% of trajectory
  - [ ] Method 'reference': user provides reference lap
  - [ ] Method 'region': detect crossings of start region
  - [ ] Compute overlap with template (Jaccard index)
  - [ ] Detect direction (clockwise/counter-clockwise)
  - [ ] Return list of Lap objects (start_time, end_time, direction, overlap_score)

**Testing**:

- [ ] Create `tests/segmentation/test_laps.py`
- [ ] Test: `test_detect_laps_circular_track()` - synthetic circular trajectory
- [ ] Test: `test_detect_laps_direction()` - clockwise vs counter-clockwise
- [ ] Test: `test_detect_laps_auto_template()` - template detection
- [ ] Test: `test_detect_laps_overlap_threshold()` - min_overlap filtering
- [ ] Run: `uv run pytest tests/segmentation/test_laps.py -v`

**Validation**:

- [ ] Compare with neurocode NSMAFindGoodLaps.m if available

**Effort**: 2 days

---

### 4.4 Trial Segmentation (Week 10, Day 6)

**Implementation**:

- [ ] Create `src/neurospatial/segmentation/trials.py`
- [ ] Implement `segment_trials(trajectory_bins, times, env, *, trial_type, start_region, end_regions, min_duration=1.0, max_duration=15.0)`
  - [ ] Detect start region entries
  - [ ] Track trajectory to end regions
  - [ ] Determine outcome (which end region reached)
  - [ ] Filter by duration
  - [ ] Return list of Trial objects (start_time, end_time, outcome, success)

**Testing**:

- [ ] Create `tests/segmentation/test_trials.py`
- [ ] Test: `test_segment_trials_tmaze()` - T-maze left/right trials
- [ ] Test: `test_segment_trials_duration_filter()` - min/max duration
- [ ] Run: `uv run pytest tests/segmentation/test_trials.py -v`

**Effort**: 1 day

---

### 4.5 Trajectory Similarity (Week 11, Days 1-2)

**Implementation**:

- [ ] Create `src/neurospatial/segmentation/similarity.py`
- [ ] Implement `trajectory_similarity(trajectory1_bins, trajectory2_bins, env, *, method='jaccard')`
  - [ ] Method 'jaccard': spatial overlap (set intersection / union)
  - [ ] Method 'correlation': sequential correlation
  - [ ] Method 'hausdorff': maximum deviation
  - [ ] Method 'dtw': dynamic time warping
  - [ ] Return similarity score [0, 1]
- [ ] Implement `detect_goal_directed_runs(trajectory_bins, times, env, *, goal_region, directedness_threshold=0.7, min_progress=20.0)`
  - [ ] Compute distance to goal at start and end
  - [ ] Compute path length
  - [ ] Directedness = (d_start - d_end) / path_length
  - [ ] Filter by threshold and minimum progress
  - [ ] Return list of Run objects

**Testing**:

- [ ] Create `tests/segmentation/test_similarity.py`
- [ ] Test: `test_trajectory_similarity_identical()` - same trajectory = 1.0
- [ ] Test: `test_trajectory_similarity_disjoint()` - no overlap = 0.0
- [ ] Test: `test_trajectory_similarity_methods()` - all methods
- [ ] Test: `test_detect_goal_directed_runs()` - straight path high directedness
- [ ] Run: `uv run pytest tests/segmentation/test_similarity.py -v`

**Effort**: 2 days

---

### 4.6 Tests & Documentation (Week 11, Days 3-5)

**Testing**:

- [ ] Run all segmentation tests: `uv run pytest tests/segmentation/ -v`
- [ ] Verify coverage: `uv run pytest tests/segmentation/ --cov=src/neurospatial/segmentation/`
- [ ] Integration test: full workflow (trajectory → runs → laps → trials)

**Documentation**:

- [ ] Create `docs/user-guide/trajectory-and-behavioral-analysis.md`
  - [ ] Section: Trajectory characterization metrics
  - [ ] Section: Region-based segmentation
  - [ ] Section: Lap detection strategies
  - [ ] Section: Trial segmentation for tasks
  - [ ] Section: Use cases (goal-directed replay, learning dynamics)

**Example Notebooks**:

- [ ] Create `examples/12_trajectory_analysis.ipynb`
  - [ ] Compute turn angles, step lengths
  - [ ] Compute home range (95%)
  - [ ] Compute MSD and classify diffusion
  - [ ] Visualize trajectory properties
- [ ] Create `examples/13_behavioral_segmentation.ipynb`
  - [ ] Detect runs between goal regions
  - [ ] Detect laps on circular track
  - [ ] Segment T-maze trials
  - [ ] Compute trajectory similarity
  - [ ] Use cases: lap-by-lap learning, trial-type selectivity

**pynapple Integration**:

- [ ] Verify IntervalSet return when pynapple installed
- [ ] Add fallback to list of tuples when unavailable

**Effort**: 3 days

---

## Milestone 5: Polish & Release (Weeks 13.5-15)

**Goal**: Validate, optimize, document, and release v0.3.0

### 5.1 Validation Against Authority Packages (Week 12, Days 1-2)

**opexebo Validation**:

- [ ] Create `tests/validation/test_opexebo_comparison.py`
- [ ] Test: Place field detection matches neurocode subfield approach
- [ ] Test: Spatial information matches opexebo/neurocode/buzcode (if available)
- [ ] Test: Sparsity calculation matches opexebo
- [ ] Test: Border score matches TSToolbox_Utils/opexebo
- [ ] Run: `uv run pytest tests/validation/ -v --run-validation` (optional marker)

**Ecology Validation**:

- [ ] Test: Turn angles match Traja on synthetic trajectory
- [ ] Test: Step lengths correct on known path
- [ ] Test: Home range matches adehabitatHR concept (95% KDE)
- [ ] Test: MSD exponent correct for random walk (α ≈ 1)

**Document Differences**:

- [ ] Create `docs/validation-notes.md`
  - [ ] Intentional differences (irregular graph support)
  - [ ] Extensions beyond reference packages
  - [ ] Validation results and discrepancies

**Effort**: 2 days

---

### 5.2 Performance Optimization (Week 12, Days 3-4)

**Profiling**:

- [ ] Create `benchmarks/bench_differential.py`
  - [ ] Benchmark: differential_operator construction
  - [ ] Benchmark: gradient computation
  - [ ] Benchmark: cached vs uncached (verify 50x speedup)
- [ ] Create `benchmarks/bench_primitives.py`
  - [ ] Benchmark: neighbor_reduce operations
  - [ ] Benchmark: convolve with different kernel sizes
- [ ] Create `benchmarks/bench_metrics.py`
  - [ ] Benchmark: detect_place_fields
  - [ ] Benchmark: border_score

**Optimization**:

- [ ] Profile critical paths: `uv run python -m cProfile script.py`
- [ ] Optimize hot loops (vectorize where possible)
- [ ] Add caching where beneficial (beyond differential_operator)
- [ ] Target: No operation >10% slower than baseline

**Testing**:

- [ ] Add performance regression tests to CI/CD
- [ ] Verify no slowdowns: `uv run pytest benchmarks/ -v`

**Effort**: 2 days

---

### 5.3 Documentation Polish (Week 12-13, Days 5-9)

**API Documentation**:

- [ ] Verify all functions have NumPy-style docstrings
- [ ] Verify all type hints present
- [ ] Verify all examples in docstrings work
- [ ] Add cross-references between related functions
- [ ] Generate API docs (if using Sphinx/mkdocs)

**User Guides**:

- [ ] Review `differential-operators.md` for completeness
- [ ] Review `signal-processing-primitives.md` (if created)
- [ ] Review `neuroscience-metrics.md` for accuracy
- [ ] Review `trajectory-and-behavioral-analysis.md` for clarity
- [ ] Add cross-links between guides
- [ ] Add "See Also" sections

**Example Notebooks**:

- [ ] Review all notebooks for clarity
- [ ] Add explanatory markdown cells
- [ ] Add visualizations where helpful
- [ ] Test all notebooks execute: `uv run jupyter nbconvert --execute examples/*.ipynb`

**README Updates**:

- [ ] Update README.md with v0.3.0 features
- [ ] Add installation instructions
- [ ] Add quick start example
- [ ] Add links to documentation

**Effort**: 3 days

---

### 5.4 Release (Week 13, Days 10-13)

**Pre-Release Checks**:

- [ ] Run full test suite: `uv run pytest --cov=src/neurospatial`
- [ ] Verify coverage >95%
- [ ] Run mypy on all modules: `uv run mypy src/neurospatial/`
- [ ] Verify zero mypy errors (no `type: ignore` comments)
- [ ] Run ruff linter: `uv run ruff check .`
- [ ] Run ruff formatter: `uv run ruff format .`
- [ ] Fix any linting issues

**Version Bump**:

- [ ] Update version in `pyproject.toml` to `0.3.0`
- [ ] Update version in `src/neurospatial/__init__.py` (if present)
- [ ] Update `__version__` string

**Changelog**:

- [ ] Create/update `CHANGELOG.md` for v0.3.0
  - [ ] Section: Breaking Changes (divergence → kl_divergence)
  - [ ] Section: New Features
    - [ ] Differential operators (gradient, divergence, Laplacian)
    - [ ] Signal processing primitives (neighbor_reduce, convolve)
    - [ ] Place field metrics (detection, Skaggs info, sparsity)
    - [ ] Population metrics (coverage, density, overlap)
    - [ ] Boundary cell metrics (border score)
    - [ ] Trajectory metrics (turn angles, step lengths, home range, MSD)
    - [ ] Behavioral segmentation (runs, laps, trials, similarity)
  - [ ] Section: Deferred to v0.4.0
    - [ ] Grid cell analysis (spatial_autocorrelation, grid_score)
    - [ ] Circular statistics (von Mises, Rayleigh test)
  - [ ] Section: Migration Guide (minimal - just divergence rename)

**Release Artifacts**:

- [ ] Tag release: `git tag -a v0.3.0 -m "Release v0.3.0"`
- [ ] Push tag: `git push origin v0.3.0`
- [ ] Build package: `uv build`
- [ ] Test install: `uv pip install dist/neurospatial-0.3.0-*.whl`
- [ ] Publish to PyPI: `uv publish` (or `twine upload dist/*`)

**Announcement**:

- [ ] Write blog post / release notes highlighting:
  - [ ] Core spatial primitives now available
  - [ ] Validated against opexebo, neurocode
  - [ ] Extensions for irregular graphs
  - [ ] Deferred features (grid cells in v0.4.0)
- [ ] Post to GitHub Releases
- [ ] Announce on relevant channels (if any)

**Post-Release**:

- [ ] Verify PyPI package installs correctly
- [ ] Update documentation site (if hosted)
- [ ] Close milestone in project tracker
- [ ] Plan v0.4.0 (grid cells, circular stats)

**Effort**: 2 days

---

## Success Criteria (Final Checklist)

### Phase 1: Differential Operators

- [ ] D matrix construction passes all tests
- [ ] gradient(), divergence() work on all layout types (regular, hex, irregular)
- [ ] div(grad(f)) == Laplacian(f) validated
- [ ] 50x caching speedup confirmed
- [ ] divergence() renamed to kl_divergence()

### Phase 2: Signal Processing Primitives

- [ ] neighbor_reduce() works on all layout types
- [ ] convolve() supports arbitrary kernels (callable and matrix)
- [ ] All tests pass for all layouts

### Phase 3: Core Metrics Module

- [ ] Place field detection matches neurocode's subfield discrimination
- [ ] Spatial information matches opexebo/neurocode/buzcode
- [ ] Sparsity calculation matches opexebo
- [ ] Border score matches TSToolbox_Utils/opexebo
- [ ] All metrics have NumPy docstrings, examples, and citations

### Phase 4: Trajectory & Behavioral Segmentation

- [ ] Trajectory metrics validated on synthetic data (turn angles, MSD)
- [ ] Region crossing detection works on synthetic trajectories
- [ ] Lap detection handles clockwise/counter-clockwise
- [ ] Trial segmentation works for T-maze, Y-maze
- [ ] Trajectory similarity methods validated
- [ ] pynapple IntervalSet integration works when available

### Phase 5: Release

- [ ] All tests pass: `uv run pytest` (>95% coverage)
- [ ] Zero mypy errors: `uv run mypy src/neurospatial/`
- [ ] Documentation complete (user guides + examples)
- [ ] Performance benchmarks meet targets
- [ ] Version 0.3.0 released to PyPI

---

## Development Workflow Reminders

**Always Use uv**:

- [ ] All commands prefixed with `uv run`: `uv run pytest`, `uv run mypy`, etc.
- [ ] Never use bare `python`, `pip`, or `pytest`

**Type Checking**:

- [ ] Mypy is mandatory - zero errors required
- [ ] No `type: ignore` comments allowed
- [ ] Use `TYPE_CHECKING` guards for Environment imports in mixins
- [ ] Use `self: "Environment"` annotations in mixin methods

**Documentation**:

- [ ] All docstrings use NumPy format (not Google or RST)
- [ ] All functions have examples in docstrings
- [ ] All examples use `>>>` prompt and show expected output

**Testing**:

- [ ] Tests mirror source structure (tests/metrics/, tests/segmentation/)
- [ ] Use fixtures from conftest.py for common environments
- [ ] Test edge cases (empty, single node, disconnected graphs)
- [ ] Target >95% coverage

**Commit Messages**:

- [ ] Use Conventional Commits format
- [ ] Examples: `feat(metrics): add place field detection`, `fix(differential): correct gradient sign`
- [ ] Reference issues if applicable

---

## Timeline Summary

| Milestone | Duration | Weeks |
|-----------|----------|-------|
| M0: Spike/Field Primitives | 2 weeks | 1-2 |
| M1: Differential Operators | 2.5 weeks | 3-5 |
| M2: Signal Processing | 2 weeks | 5-7 |
| M3: Core Metrics | 1.5 weeks | 7.5-9 |
| M4: Trajectory & Behavioral | 2.5 weeks | 9.5-12 |
| M5: Polish & Release | 1.5 weeks | 12.5-14 |
| **Total** | **14 weeks** | **~3.25 months** |

---

## Risk Mitigation

**Medium Risk**: Performance regressions

- **Mitigation**: Benchmark suite in CI/CD, monitor key operations
- **Fallback**: Optimize hot paths, add optional Numba compilation

**Low Risk**: API design conflicts

- **Mitigation**: Match opexebo signatures where possible
- **Fallback**: User feedback before finalizing

**Low Risk**: Validation mismatches

- **Mitigation**: Test against opexebo/neurocode outputs
- **Fallback**: Document intentional differences (irregular graph support)

---

## Notes

- Tasks marked `(optional)` can be deferred if timeline is tight
- Integration tests should be added throughout, not just at the end
- User feedback should be solicited during Milestones 3-4 (metrics and segmentation)
- Documentation should be written concurrently with implementation, not deferred
