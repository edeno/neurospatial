# Encoding Module Refactor - Task Breakdown

This document breaks down the encoding module refactor into actionable tasks. Each task is designed to be completable in a focused work session.

**Reference**: See [PLAN.md](PLAN.md) for full design specifications.

---

## Milestone 0: Core Infrastructure

**Goal**: Establish shared foundations before implementing any encoding type.

**Success Criteria**: All subsequent milestones can build on these foundations without modification.

### Tasks

- [x] **0.1** Create `encoding/_base.py` with shared protocols and helpers
  - [x] Implement `_to_numpy(arr)` helper for JAX → NumPy conversion
  - [x] Implement `_get_array_module(arr)` for backend dispatch
  - [x] Define `HasOccupancy` protocol
  - [x] Define `HasEnvironment` protocol
  - [x] Implement `SpatialResultMixin` with `peak_locations()` and `peak_firing_rates()`

- [x] **0.2** Create `encoding/_spikes.py` with spike format normalization
  - [x] Implement `normalize_spike_times()` function
  - [x] Handle 1D array (single neuron) input
  - [x] Handle 2D array (n_neurons, max_spikes) with NaN padding
  - [x] Handle list/tuple of 1D arrays (canonical format)
  - [x] Reject ragged object arrays with clear error message
  - [x] Add unit tests for all input formats

- [x] **0.3** Create `encoding/_backend.py` with backend selection
  - [x] Implement `get_backend(name: str)` function
  - [x] Add platform detection for JAX availability (Linux/macOS only)
  - [x] Define `"auto"` behavior: use JAX if available, NumPy otherwise
  - [x] Force NumPy silently on Windows when `"auto"`

- [x] **0.4** Create `encoding/_core_numpy.py` with stubs
  - [x] Define function signatures for core array operations
  - [x] Add placeholder implementations that raise NotImplementedError

- [x] **0.5** Create `encoding/_core_jax.py` with stubs
  - [x] Define function signatures matching `_core_numpy.py`
  - [x] Add placeholder implementations (to be filled in Phase 6)

---

## Milestone 1: Shared Computation Layer

**Goal**: Implement shared utilities used across all encoding types.

**Dependencies**: Milestone 0 complete

**Success Criteria**: `_metrics.py` and `_smoothing.py` pass unit tests and can be imported by encoding modules.

**Note**: `_metrics.py` and `_smoothing.py` are NumPy-only in this milestone. JAX implementations are deferred to Milestone 6. The `_get_array_module()` helper is defined in `_base.py` but backend dispatch in `_metrics.py` will be added when JAX support is implemented.

### Tasks

- [x] **1.1** Create `encoding/_metrics.py` with shared metric implementations
  - [x] Implement `spatial_information(firing_rate, occupancy)` for single neuron
  - [x] Implement `batch_spatial_information(firing_rates, occupancy)` for population
  - [x] Implement `sparsity(firing_rate, occupancy)` for single neuron
  - [x] Implement `batch_sparsity(firing_rates, occupancy)` for population
  - [x] Add unit tests

- [x] **1.2** Create `encoding/_smoothing.py` with shared smoothing code
  - [x] Extract common smoothing logic from existing `place.py`
  - [x] Implement `diffusion_kde` smoothing
  - [x] Implement `gaussian_kde` smoothing
  - [x] Implement `binned` (no smoothing) mode
  - [x] Add unit tests

- [x] **1.3** Implement batch grid score computation in `_metrics.py`
  - [x] Implement `batch_grid_scores(env, firing_rates)` → `(n_neurons,)`
  - [x] Verify delegation to `grid.py` utilities
  - [x] Add unit tests

- [x] **1.4** Implement batch border score computation in `_metrics.py`
  - [x] Implement `batch_border_scores(env, firing_rates, ...)` → `(n_neurons,)`
  - [x] Verify delegation to `border.py` utilities
  - [x] Add unit tests

---

## Milestone 2: Spatial Rate (Place/Grid/Border Cells)

**Goal**: Implement the spatial encoding API with full result classes and metrics.

**Dependencies**: Milestones 0-1 complete

**Success Criteria**: `compute_spatial_rate()` and `compute_spatial_rates()` produce correct results for test data, all result class methods work, existing tests continue to pass.

### Tasks

- [x] **2.1** Create `encoding/spatial.py` with result class definitions
  - [x] Define `SpatialRateResult` dataclass (frozen=True)
  - [x] Define `SpatialRatesResult` dataclass (frozen=True)
  - [x] Inherit from `SpatialResultMixin` for shared methods

- [x] **2.2** Implement `SpatialRateResult` convenience methods
  - [x] Implement `plot(ax, **kwargs)` → delegates to `env.plot_field()`
  - [x] Implement `peak_location()` alias → calls `peak_locations()` from mixin
  - [x] Implement `spatial_information()` → delegates to `_metrics.py`
  - [x] Implement `sparsity()` → delegates to `_metrics.py`

- [x] **2.3** Implement `SpatialRateResult` cell type metrics
  - [x] Implement `grid_score()` → delegates to `grid.py`
  - [x] Implement `grid_properties()` → delegates to `grid.py`
  - [x] Implement `border_score(threshold, min_area, distance_metric)` → delegates to `border.py`
  - [x] Implement `region_coverage(threshold, regions)` → delegates to `border.py`

- [x] **2.4** Implement `SpatialRatesResult` batch methods
  - [x] Implement `__len__`, `__getitem__`, `__iter__` for iteration
  - [x] Implement `plot(idx, ax, **kwargs)`
  - [x] Implement `spatial_information()` → returns `(n_neurons,)`
  - [x] Implement `sparsity()` → returns `(n_neurons,)`

- [x] **2.5** Implement `SpatialRatesResult` batch metrics
  - [x] Implement `grid_scores()` → delegates to `batch_grid_scores()`
  - [x] Implement `border_scores(threshold, distance_metric)` → delegates to `batch_border_scores()`
  - [x] Implement `classify(min_spatial_info, min_grid_score, min_border_score)` → returns `(n_neurons,)` string labels

- [x] **2.6** Implement `SpatialRatesResult.to_dataframe()`
  - [x] Include neuron_id, peak_x, peak_y, peak_rate columns
  - [x] Include spatial_info, sparsity columns
  - [x] Include grid_score, border_score columns
  - [x] Optionally include cell_type classification

- [x] **2.7** Implement binning layer for spatial encoding
  - [x] Create helper to convert (env, spike_times, times, positions) → (spike_counts, occupancy)
  - [x] Spike counts shape: `(n_neurons, n_bins)`
  - [x] Occupancy shape: `(n_bins,)`
  - [x] Parallelize over neurons with joblib

- [x] **2.8** Implement `compute_spatial_rate()` function
  - [x] Accept single spike_times array
  - [x] Apply smoothing via `_smoothing.py`
  - [x] Return `SpatialRateResult`
  - [x] Add `backend` parameter (default "numpy")

- [x] **2.9** Implement `compute_spatial_rates()` function
  - [x] Accept list of spike_times arrays
  - [x] Normalize input via `normalize_spike_times()`
  - [x] Precompute shared quantities (occupancy, position bins)
  - [x] Apply smoothing via `_smoothing.py`
  - [x] Return `SpatialRatesResult`
  - [x] Add `n_jobs` parameter for parallelization
  - [x] Add `backend` parameter (default "numpy")

- [x] **2.10** Write comprehensive tests for spatial encoding
  - [x] Test single neuron computation
  - [x] Test batch computation
  - [x] Test all result class methods
  - [x] Test edge cases (empty spike trains, single spike, etc.)
  - [x] Test `to_dataframe()` output format

---

## Milestone 3: Directional Rate (Head Direction Cells)

**Goal**: Implement the directional encoding API with lazy metric computation.

**Dependencies**: Milestone 0 complete

**Success Criteria**: `compute_directional_rate()` and `compute_directional_rates()` produce correct results, metrics match existing `HeadDirectionMetrics` output.

### Tasks

- [x] **3.1** Create `encoding/directional.py` with result class definitions
  - [x] Define `DirectionalRateResult` dataclass (frozen=True)
  - [x] Define `DirectionalRatesResult` dataclass (frozen=True)
  - [x] Note: No Environment dependency, only bin_centers

- [x] **3.2** Implement `DirectionalRateResult` convenience methods
  - [x] Implement `plot(ax, polar, **kwargs)` → polar or Cartesian tuning curve
  - [x] Implement `preferred_direction()` → circular mean in radians
  - [x] Implement `preferred_direction_deg()` → converts to degrees
  - [x] Implement `peak_firing_rate()` → max of firing_rate

- [x] **3.3** Implement `DirectionalRateResult` tuning metrics
  - [x] Implement `mean_vector_length()` → Rayleigh MVL via `stats.circular`
  - [x] Implement `tuning_width()` → half-width at half-maximum (radians)
  - [x] Implement `tuning_width_deg()` → converts to degrees
  - [x] Implement `rayleigh_pvalue()` → Rayleigh test via `stats.circular`

- [x] **3.4** Implement `DirectionalRateResult` classification
  - [x] Implement `is_hd_cell(min_mvl, alpha)` → boolean based on MVL and p-value
  - [x] Implement `interpretation(min_mvl)` → human-readable string

- [x] **3.5** Implement `DirectionalRatesResult` batch methods
  - [x] Implement `__len__`, `__getitem__`, `__iter__` for iteration
  - [x] Implement `plot(idx, ax, polar, **kwargs)`
  - [x] Implement `preferred_directions()` → returns `(n_neurons,)`
  - [x] Implement `mean_vector_lengths()` → returns `(n_neurons,)`
  - [x] Implement `tuning_widths()` → returns `(n_neurons,)`
  - [x] Implement `detect_hd_cells(min_mvl, alpha)` → returns `(n_neurons,)` bool
  - [x] Implement `peak_firing_rates()` → returns `(n_neurons,)`

- [x] **3.6** Implement `DirectionalRatesResult.to_dataframe()`
  - [x] Include neuron_id, preferred_direction, preferred_direction_deg columns
  - [x] Include mean_vector_length, tuning_width, tuning_width_deg columns
  - [x] Include peak_rate, is_hd_cell columns

- [x] **3.7** Implement binning layer for directional encoding
  - [x] Create helper to convert (spike_times, times, headings, bin_size) → (spike_counts, occupancy)
  - [x] Handle circular binning (0 to 2π)
  - [x] Support `angle_unit` parameter for input conversion

- [x] **3.8** Implement `compute_directional_rate()` function
  - [x] Accept single spike_times array
  - [x] Support `angle_unit` parameter ("rad" or "deg")
  - [x] Optional Gaussian smoothing via `smoothing_sigma`
  - [x] Return `DirectionalRateResult`

- [x] **3.9** Implement `compute_directional_rates()` function
  - [x] Accept list of spike_times arrays
  - [x] Normalize input via `normalize_spike_times()`
  - [x] Precompute shared quantities (occupancy, heading bins)
  - [x] Add `n_jobs` parameter for parallelization
  - [x] Return `DirectionalRatesResult`

- [x] **3.11** Write comprehensive tests for directional encoding
  - [x] Test single neuron computation
  - [x] Test batch computation
  - [x] Test all result class methods
  - [x] Test `angle_unit` conversion
  - [x] Verify metrics match existing `HeadDirectionMetrics` class

---

## Milestone 4: View Rate (Spatial View Cells)

**Goal**: Implement the view encoding API for spatial view cells.

**Dependencies**: Milestones 0-1 complete

**Success Criteria**: `compute_view_rate()` and `compute_view_rates()` produce correct view fields, metrics enable view cell classification.

### Tasks

- [x] **4.1** Create `encoding/view.py` with result class definitions
  - [x] Define `ViewRateResult` dataclass (frozen=True)
  - [x] Define `ViewRatesResult` dataclass (frozen=True)

- [x] **4.2** Implement `ViewRateResult` convenience methods
  - [x] Implement `plot(ax, **kwargs)` → delegates to `env.plot_field()`
  - [x] Implement `peak_view_location()` → location of peak view response
  - [x] Implement `view_spatial_information()` → spatial info based on view occupancy

- [x] **4.3** Implement `ViewRateResult` classification
  - [x] Implement `is_view_cell(min_info)` → boolean based on view spatial info

- [x] **4.4** Implement `ViewRatesResult` batch methods
  - [x] Implement `__len__`, `__getitem__`, `__iter__` for iteration
  - [x] Implement `plot(idx, ax, **kwargs)`
  - [x] Implement `peak_view_locations()` → returns `(n_neurons, n_dims)`
  - [x] Implement `view_spatial_information()` → returns `(n_neurons,)`
  - [x] Implement `detect_view_cells(min_info)` → returns `(n_neurons,)` bool

- [x] **4.5** Implement `ViewRatesResult.to_dataframe()`
  - [x] Include neuron_id, peak_view_x, peak_view_y columns
  - [x] Include peak_rate, view_spatial_info columns
  - [x] Include is_view_cell column

- [x] **4.6** Implement binning layer for view encoding
  - [x] Create helper to compute view occupancy (time viewing each spatial bin)
  - [x] Support gaze models: "fixed_distance", "ray_cast", "boundary"
  - [x] Convert (spike_times, times, positions, headings, gaze_model) → (spike_counts, view_occupancy)

- [x] **4.7** Implement `compute_view_rate()` function
  - [x] Accept single spike_times array
  - [x] Support `gaze_model` parameter
  - [x] Support `view_distance` parameter
  - [x] Apply smoothing via `_smoothing.py`
  - [x] Return `ViewRateResult`

- [x] **4.8** Implement `compute_view_rates()` function
  - [x] Accept list of spike_times arrays
  - [x] Normalize input via `normalize_spike_times()`
  - [x] Precompute shared quantities (view_occupancy, viewed_bins)
  - [x] Add `n_jobs` parameter for parallelization
  - [x] Return `ViewRatesResult`

- [x] **4.9** Write comprehensive tests for view encoding
  - [x] Test single neuron computation
  - [x] Test batch computation
  - [x] Test all result class methods
  - [x] Test different gaze models
  - [x] Test `to_dataframe()` output format

---

## Milestone 5: Egocentric Rate (Object Vector Cells)

**Goal**: Implement the egocentric encoding API for object vector cells.

**Dependencies**: Milestone 0 complete

**Success Criteria**: `compute_egocentric_rate()` and `compute_egocentric_rates()` produce correct egocentric polar fields.

### Tasks

- [x] **5.1** Create `encoding/egocentric.py` with result class definitions
  - [x] Define `EgocentricRateResult` dataclass (frozen=True)
  - [x] Define `EgocentricRatesResult` dataclass (frozen=True)
  - [x] Note: Uses `ego_env` (polar Environment), not spatial Environment

- [x] **5.2** Implement `EgocentricRateResult` convenience methods
  - [x] Implement `plot(ax, **kwargs)` → delegates to `ego_env.plot_field()`
  - [x] Implement `preferred_distance()` → distance component of peak bin
  - [x] Implement `preferred_direction()` → direction component of peak bin (0=ahead)

- [x] **5.3** Implement `EgocentricRateResult` classification
  - [x] Implement `is_ovc(min_info)` → boolean based on egocentric spatial info

- [x] **5.4** Implement `EgocentricRatesResult` batch methods
  - [x] Implement `__len__`, `__getitem__`, `__iter__` for iteration
  - [x] Implement `plot(idx, ax, **kwargs)`
  - [x] Implement `preferred_distances()` → returns `(n_neurons,)`
  - [x] Implement `preferred_directions()` → returns `(n_neurons,)`
  - [x] Implement `detect_ovcs(min_info)` → returns `(n_neurons,)` bool

- [x] **5.5** Implement `EgocentricRatesResult.to_dataframe()`
  - [x] Include neuron_id, preferred_distance, preferred_direction columns
  - [x] Include preferred_direction_deg, peak_rate columns
  - [x] Include is_ovc column

- [x] **5.6** Implement binning layer for egocentric encoding
  - [x] Create helper to compute egocentric occupancy
  - [x] Compute egocentric bearing and distance for all position-object pairs
  - [x] Convert to polar bin indices
  - [x] Support `distance_metric` parameter ("euclidean" or "geodesic")
  - [x] When "geodesic", require `env` parameter

- [x] **5.7** Implement `compute_egocentric_rate()` function
  - [x] Accept single spike_times array
  - [x] Accept `object_positions` array
  - [x] Support `distance_range` and bin count parameters
  - [x] Support `distance_metric` parameter
  - [x] Optional `env` parameter (required for geodesic)
  - [x] Apply smoothing via `_smoothing.py`
  - [x] Return `EgocentricRateResult`

- [x] **5.8** Implement `compute_egocentric_rates()` function
  - [x] Accept list of spike_times arrays
  - [x] Normalize input via `normalize_spike_times()`
  - [x] Precompute shared quantities (egocentric occupancy, polar bins)
  - [x] Add `n_jobs` parameter for parallelization
  - [x] Return `EgocentricRatesResult`

- [x] **5.9** Write comprehensive tests for egocentric encoding
  - [x] Test single neuron computation
  - [x] Test batch computation
  - [x] Test all result class methods
  - [x] Test euclidean vs geodesic distance
  - [x] Test error when geodesic requested without env
  - [x] Test `to_dataframe()` output format

---

## Milestone 6: Optional JAX Backend

**Goal**: Add JAX acceleration for batch computations.

**Dependencies**: Milestones 2-5 complete (all NumPy implementations working)

**Success Criteria**: With JAX installed, `backend="jax"` produces identical results to `backend="numpy"` with better performance on large populations.

### Tasks

- [x] **6.1** Implement JAX smoothing operations in `_core_jax.py`
  - [x] Port `diffusion_kde` smoothing to JAX
  - [x] Port `gaussian_kde` smoothing to JAX
  - [x] Verify numerical equivalence with NumPy

- [x] **6.2** Implement JAX metric computations in `_core_jax.py`
  - [x] Port `spatial_information` to JAX
  - [x] Port `sparsity` to JAX
  - [x] Use `vmap` for batch operations
  - [x] Verify numerical equivalence with NumPy

- [ ] **6.3** Implement JAX grid/border score computations
  - [ ] Port grid score inner loop to JAX
  - [ ] Port border score inner loop to JAX
  - [ ] Use `vmap` for batch operations

- [ ] **6.4** Add backend dispatch to compute functions
  - [ ] Add `backend` parameter to `compute_spatial_rate(s)`
  - [ ] Add `backend` parameter to `compute_directional_rate(s)`
  - [ ] Add `backend` parameter to `compute_view_rate(s)`
  - [ ] Add `backend` parameter to `compute_egocentric_rate(s)`
  - [ ] Route through `_backend.py` selection logic

- [ ] **6.5** Update result class methods for backend awareness
  - [ ] Ensure `_to_numpy()` handles JAX arrays correctly
  - [ ] Ensure `_get_array_module()` detects JAX arrays
  - [ ] Test plotting methods with JAX arrays

- [ ] **6.6** Write JAX-specific tests
  - [ ] Test JAX backend produces same results as NumPy
  - [ ] Test `"auto"` behavior with JAX installed
  - [ ] Test `"auto"` behavior on Windows (forces NumPy)
  - [ ] Test `"jax"` raises clear error when JAX unavailable

- [ ] **6.7** Add performance benchmarks
  - [ ] Create benchmark script comparing NumPy vs JAX
  - [ ] Test with varying population sizes (10, 100, 1000 neurons)
  - [ ] Document performance characteristics

---

## Milestone 7: Cleanup and Documentation

**Goal**: Remove deprecated code, update exports, and document the new API.

**Dependencies**: All previous milestones complete

**Success Criteria**: Clean imports from `neurospatial.encoding`, all documentation updated, no regressions in existing tests.

### Tasks

- [ ] **7.1** Update `encoding/__init__.py` exports
  - [ ] Export all new functions and result classes
  - [ ] Export deprecated shims with deprecation marker
  - [ ] Define `__all__` list

- [ ] **7.2** Remove old files (after deprecation period)
  - [ ] Remove `encoding/place.py` (merged into `spatial.py`)
  - [ ] Remove `encoding/head_direction.py` (replaced by `directional.py`)
  - [ ] Remove `encoding/spatial_view.py` (replaced by `view.py`)
  - [ ] Remove `encoding/object_vector.py` (replaced by `egocentric.py`)
  - [ ] Note: Keep `border.py`, `grid.py`, `population.py`, `phase_precession.py`

- [ ] **7.3** Update example notebooks using old encoding API
  - [ ] Update `08_spike_field_basics.ipynb` to use new API
  - [ ] Update `10_signal_processing_primitives.ipynb` to use new API
  - [ ] Update `11_place_field_analysis.ipynb` to use `compute_spatial_rate(s)`
  - [ ] Update `19_real_data_bandit_task.ipynb` to use new API
  - [ ] Update `20_bayesian_decoding.ipynb` to use `compute_spatial_rate(s)`
  - [ ] Update `21_directional_place_fields.ipynb` to use `compute_directional_rate(s)`
  - [ ] Update `22_spatial_view_cells.ipynb` to use `compute_view_rate(s)`
  - [ ] Sync `examples/` and `docs/examples/` directories after updates

- [ ] **7.4** Update CLAUDE.md documentation
  - [ ] Update "Most Common Patterns" section with new API
  - [ ] Update API reference for encoding module
  - [ ] Add backend parameter documentation
  - [ ] Document result class methods

- [ ] **7.5** Update QUICKSTART.md documentation
  - [ ] Update neural encoding section
  - [ ] Add batch processing examples
  - [ ] Add `to_dataframe()` usage examples

- [ ] **7.6** Run full test suite and fix regressions
  - [ ] Run `uv run pytest` to verify all tests pass
  - [ ] Run `uv run mypy src/neurospatial/encoding/` for type checking
  - [ ] Run `uv run ruff check . && uv run ruff format .` for linting

- [ ] **7.7** Create migration guide
  - [ ] Document old function → new function mapping
  - [ ] Document result class field name changes (`.field` → `.firing_rate`)
  - [ ] Provide code snippets for common migrations

---

## Task Tracking Legend

- [ ] Not started
- [x] Complete
- [~] In progress (use sparingly)

## Notes

- Tasks within a milestone can often be parallelized
- Each task should be testable independently
- When blocked, note the blocking dependency in the task
- Update this file as work progresses
