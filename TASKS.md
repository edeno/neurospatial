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

### Tasks

- [ ] **1.1** Create `encoding/_metrics.py` with shared metric implementations
  - [ ] Implement `spatial_information(firing_rate, occupancy)` for single neuron
  - [ ] Implement `batch_spatial_information(firing_rates, occupancy)` for population
  - [ ] Implement `sparsity(firing_rate, occupancy)` for single neuron
  - [ ] Implement `batch_sparsity(firing_rates, occupancy)` for population
  - [ ] Add unit tests

- [ ] **1.2** Create `encoding/_smoothing.py` with shared smoothing code
  - [ ] Extract common smoothing logic from existing `place.py`
  - [ ] Implement `diffusion_kde` smoothing
  - [ ] Implement `gaussian_kde` smoothing
  - [ ] Implement `binned` (no smoothing) mode
  - [ ] Add unit tests

- [ ] **1.3** Implement batch grid score computation in `_metrics.py`
  - [ ] Implement `batch_grid_scores(env, firing_rates)` → `(n_neurons,)`
  - [ ] Verify delegation to `grid.py` utilities
  - [ ] Add unit tests

- [ ] **1.4** Implement batch border score computation in `_metrics.py`
  - [ ] Implement `batch_border_scores(env, firing_rates, ...)` → `(n_neurons,)`
  - [ ] Verify delegation to `border.py` utilities
  - [ ] Add unit tests

---

## Milestone 2: Spatial Rate (Place/Grid/Border Cells)

**Goal**: Implement the spatial encoding API with full result classes and metrics.

**Dependencies**: Milestones 0-1 complete

**Success Criteria**: `compute_spatial_rate()` and `compute_spatial_rates()` produce correct results for test data, all result class methods work, existing tests continue to pass.

### Tasks

- [ ] **2.1** Create `encoding/spatial.py` with result class definitions
  - [ ] Define `SpatialRateResult` dataclass (frozen=True)
  - [ ] Define `SpatialRatesResult` dataclass (frozen=True)
  - [ ] Inherit from `SpatialResultMixin` for shared methods

- [ ] **2.2** Implement `SpatialRateResult` convenience methods
  - [ ] Implement `plot(ax, **kwargs)` → delegates to `env.plot_field()`
  - [ ] Implement `peak_location()` alias → calls `peak_locations()` from mixin
  - [ ] Implement `spatial_information()` → delegates to `_metrics.py`
  - [ ] Implement `sparsity()` → delegates to `_metrics.py`

- [ ] **2.3** Implement `SpatialRateResult` cell type metrics
  - [ ] Implement `grid_score()` → delegates to `grid.py`
  - [ ] Implement `grid_properties()` → delegates to `grid.py`
  - [ ] Implement `border_score(threshold, distance_metric)` → delegates to `border.py`
  - [ ] Implement `region_coverage(threshold, regions)` → delegates to `border.py`

- [ ] **2.4** Implement `SpatialRatesResult` batch methods
  - [ ] Implement `__len__`, `__getitem__`, `__iter__` for iteration
  - [ ] Implement `plot(idx, ax, **kwargs)`
  - [ ] Implement `spatial_information()` → returns `(n_neurons,)`
  - [ ] Implement `sparsity()` → returns `(n_neurons,)`

- [ ] **2.5** Implement `SpatialRatesResult` batch metrics
  - [ ] Implement `grid_scores()` → delegates to `batch_grid_scores()`
  - [ ] Implement `border_scores(threshold, distance_metric)` → delegates to `batch_border_scores()`
  - [ ] Implement `classify(min_spatial_info, min_grid_score, min_border_score)` → returns `(n_neurons,)` string labels

- [ ] **2.6** Implement `SpatialRatesResult.to_dataframe()`
  - [ ] Include neuron_id, peak_x, peak_y, peak_rate columns
  - [ ] Include spatial_info, sparsity columns
  - [ ] Include grid_score, border_score columns
  - [ ] Optionally include cell_type classification

- [ ] **2.7** Implement binning layer for spatial encoding
  - [ ] Create helper to convert (env, spike_times, times, positions) → (spike_counts, occupancy)
  - [ ] Spike counts shape: `(n_neurons, n_bins)`
  - [ ] Occupancy shape: `(n_bins,)`
  - [ ] Parallelize over neurons with joblib

- [ ] **2.8** Implement `compute_spatial_rate()` function
  - [ ] Accept single spike_times array
  - [ ] Apply smoothing via `_smoothing.py`
  - [ ] Return `SpatialRateResult`
  - [ ] Add `backend` parameter (default "numpy")

- [ ] **2.9** Implement `compute_spatial_rates()` function
  - [ ] Accept list of spike_times arrays
  - [ ] Normalize input via `normalize_spike_times()`
  - [ ] Precompute shared quantities (occupancy, position bins)
  - [ ] Apply smoothing via `_smoothing.py`
  - [ ] Return `SpatialRatesResult`
  - [ ] Add `n_jobs` parameter for parallelization
  - [ ] Add `backend` parameter (default "numpy")

- [ ] **2.10** Write comprehensive tests for spatial encoding
  - [ ] Test single neuron computation
  - [ ] Test batch computation
  - [ ] Test all result class methods
  - [ ] Test edge cases (empty spike trains, single spike, etc.)
  - [ ] Test `to_dataframe()` output format

---

## Milestone 3: Directional Rate (Head Direction Cells)

**Goal**: Implement the directional encoding API with lazy metric computation.

**Dependencies**: Milestone 0 complete

**Success Criteria**: `compute_directional_rate()` and `compute_directional_rates()` produce correct results, metrics match existing `HeadDirectionMetrics` output.

### Tasks

- [ ] **3.1** Create `encoding/directional.py` with result class definitions
  - [ ] Define `DirectionalRateResult` dataclass (frozen=True)
  - [ ] Define `DirectionalRatesResult` dataclass (frozen=True)
  - [ ] Note: No Environment dependency, only bin_centers

- [ ] **3.2** Implement `DirectionalRateResult` convenience methods
  - [ ] Implement `plot(ax, polar, **kwargs)` → polar or Cartesian tuning curve
  - [ ] Implement `preferred_direction()` → circular mean in radians
  - [ ] Implement `preferred_direction_deg()` → converts to degrees
  - [ ] Implement `peak_firing_rate()` → max of firing_rate

- [ ] **3.3** Implement `DirectionalRateResult` tuning metrics
  - [ ] Implement `mean_vector_length()` → Rayleigh MVL via `stats.circular`
  - [ ] Implement `tuning_width()` → half-width at half-maximum (radians)
  - [ ] Implement `tuning_width_deg()` → converts to degrees
  - [ ] Implement `rayleigh_pvalue()` → Rayleigh test via `stats.circular`

- [ ] **3.4** Implement `DirectionalRateResult` classification
  - [ ] Implement `is_hd_cell(min_mvl, alpha)` → boolean based on MVL and p-value
  - [ ] Implement `interpretation(min_mvl)` → human-readable string

- [ ] **3.5** Implement `DirectionalRatesResult` batch methods
  - [ ] Implement `__len__`, `__getitem__`, `__iter__` for iteration
  - [ ] Implement `plot(idx, ax, polar, **kwargs)`
  - [ ] Implement `preferred_directions()` → returns `(n_neurons,)`
  - [ ] Implement `mean_vector_lengths()` → returns `(n_neurons,)`
  - [ ] Implement `tuning_widths()` → returns `(n_neurons,)`
  - [ ] Implement `detect_hd_cells(min_mvl, alpha)` → returns `(n_neurons,)` bool

- [ ] **3.6** Implement `DirectionalRatesResult.to_dataframe()`
  - [ ] Include neuron_id, preferred_direction, preferred_direction_deg columns
  - [ ] Include mean_vector_length, tuning_width, tuning_width_deg columns
  - [ ] Include peak_rate, is_hd_cell columns

- [ ] **3.7** Implement binning layer for directional encoding
  - [ ] Create helper to convert (spike_times, times, headings, bin_size) → (spike_counts, occupancy)
  - [ ] Handle circular binning (0 to 2π)
  - [ ] Support `angle_unit` parameter for input conversion

- [ ] **3.8** Implement `compute_directional_rate()` function
  - [ ] Accept single spike_times array
  - [ ] Support `angle_unit` parameter ("rad" or "deg")
  - [ ] Optional Gaussian smoothing via `smoothing_sigma`
  - [ ] Return `DirectionalRateResult`

- [ ] **3.9** Implement `compute_directional_rates()` function
  - [ ] Accept list of spike_times arrays
  - [ ] Normalize input via `normalize_spike_times()`
  - [ ] Precompute shared quantities (occupancy, heading bins)
  - [ ] Add `n_jobs` parameter for parallelization
  - [ ] Return `DirectionalRatesResult`

- [ ] **3.10** Add deprecated shim for backwards compatibility
  - [ ] Implement `compute_head_direction_tuning_curve()` in `directional.py`
  - [ ] Emit DeprecationWarning pointing to new function
  - [ ] Return legacy `(bin_centers, firing_rate)` tuple
  - [ ] Default to `angle_unit="deg"` for legacy behavior

- [ ] **3.11** Write comprehensive tests for directional encoding
  - [ ] Test single neuron computation
  - [ ] Test batch computation
  - [ ] Test all result class methods
  - [ ] Test `angle_unit` conversion
  - [ ] Test deprecated shim warning and output
  - [ ] Verify metrics match existing `HeadDirectionMetrics` class

---

## Milestone 4: View Rate (Spatial View Cells)

**Goal**: Implement the view encoding API for spatial view cells.

**Dependencies**: Milestones 0-1 complete

**Success Criteria**: `compute_view_rate()` and `compute_view_rates()` produce correct view fields, metrics enable view cell classification.

### Tasks

- [ ] **4.1** Create `encoding/view.py` with result class definitions
  - [ ] Define `ViewRateResult` dataclass (frozen=True)
  - [ ] Define `ViewRatesResult` dataclass (frozen=True)

- [ ] **4.2** Implement `ViewRateResult` convenience methods
  - [ ] Implement `plot(ax, **kwargs)` → delegates to `env.plot_field()`
  - [ ] Implement `peak_view_location()` → location of peak view response
  - [ ] Implement `view_spatial_information()` → spatial info based on view occupancy

- [ ] **4.3** Implement `ViewRateResult` classification
  - [ ] Implement `is_view_cell(min_info)` → boolean based on view spatial info

- [ ] **4.4** Implement `ViewRatesResult` batch methods
  - [ ] Implement `__len__`, `__getitem__`, `__iter__` for iteration
  - [ ] Implement `plot(idx, ax, **kwargs)`
  - [ ] Implement `peak_view_locations()` → returns `(n_neurons, n_dims)`
  - [ ] Implement `view_spatial_information()` → returns `(n_neurons,)`
  - [ ] Implement `detect_view_cells(min_info)` → returns `(n_neurons,)` bool

- [ ] **4.5** Implement `ViewRatesResult.to_dataframe()`
  - [ ] Include neuron_id, peak_view_x, peak_view_y columns
  - [ ] Include peak_rate, view_spatial_info columns
  - [ ] Include is_view_cell column

- [ ] **4.6** Implement binning layer for view encoding
  - [ ] Create helper to compute view occupancy (time viewing each spatial bin)
  - [ ] Support gaze models: "fixed_distance", "ray_cast", "boundary"
  - [ ] Convert (spike_times, times, positions, headings, gaze_model) → (spike_counts, view_occupancy)

- [ ] **4.7** Implement `compute_view_rate()` function
  - [ ] Accept single spike_times array
  - [ ] Support `gaze_model` parameter
  - [ ] Support `view_distance` parameter
  - [ ] Apply smoothing via `_smoothing.py`
  - [ ] Return `ViewRateResult`

- [ ] **4.8** Implement `compute_view_rates()` function
  - [ ] Accept list of spike_times arrays
  - [ ] Normalize input via `normalize_spike_times()`
  - [ ] Precompute shared quantities (view_occupancy, viewed_bins)
  - [ ] Add `n_jobs` parameter for parallelization
  - [ ] Return `ViewRatesResult`

- [ ] **4.9** Write comprehensive tests for view encoding
  - [ ] Test single neuron computation
  - [ ] Test batch computation
  - [ ] Test all result class methods
  - [ ] Test different gaze models
  - [ ] Test `to_dataframe()` output format

---

## Milestone 5: Egocentric Rate (Object Vector Cells)

**Goal**: Implement the egocentric encoding API for object vector cells.

**Dependencies**: Milestone 0 complete

**Success Criteria**: `compute_egocentric_rate()` and `compute_egocentric_rates()` produce correct egocentric polar fields.

### Tasks

- [ ] **5.1** Create `encoding/egocentric.py` with result class definitions
  - [ ] Define `EgocentricRateResult` dataclass (frozen=True)
  - [ ] Define `EgocentricRatesResult` dataclass (frozen=True)
  - [ ] Note: Uses `ego_env` (polar Environment), not spatial Environment

- [ ] **5.2** Implement `EgocentricRateResult` convenience methods
  - [ ] Implement `plot(ax, **kwargs)` → delegates to `ego_env.plot_field()`
  - [ ] Implement `preferred_distance()` → distance component of peak bin
  - [ ] Implement `preferred_direction()` → direction component of peak bin (0=ahead)

- [ ] **5.3** Implement `EgocentricRateResult` classification
  - [ ] Implement `is_ovc(min_info)` → boolean based on egocentric spatial info

- [ ] **5.4** Implement `EgocentricRatesResult` batch methods
  - [ ] Implement `__len__`, `__getitem__`, `__iter__` for iteration
  - [ ] Implement `plot(idx, ax, **kwargs)`
  - [ ] Implement `preferred_distances()` → returns `(n_neurons,)`
  - [ ] Implement `preferred_directions()` → returns `(n_neurons,)`
  - [ ] Implement `detect_ovcs(min_info)` → returns `(n_neurons,)` bool

- [ ] **5.5** Implement `EgocentricRatesResult.to_dataframe()`
  - [ ] Include neuron_id, preferred_distance, preferred_direction columns
  - [ ] Include preferred_direction_deg, peak_rate columns
  - [ ] Include is_ovc column

- [ ] **5.6** Implement binning layer for egocentric encoding
  - [ ] Create helper to compute egocentric occupancy
  - [ ] Compute egocentric bearing and distance for all position-object pairs
  - [ ] Convert to polar bin indices
  - [ ] Support `distance_metric` parameter ("euclidean" or "geodesic")
  - [ ] When "geodesic", require `env` parameter

- [ ] **5.7** Implement `compute_egocentric_rate()` function
  - [ ] Accept single spike_times array
  - [ ] Accept `object_positions` array
  - [ ] Support `distance_range` and bin count parameters
  - [ ] Support `distance_metric` parameter
  - [ ] Optional `env` parameter (required for geodesic)
  - [ ] Apply smoothing via `_smoothing.py`
  - [ ] Return `EgocentricRateResult`

- [ ] **5.8** Implement `compute_egocentric_rates()` function
  - [ ] Accept list of spike_times arrays
  - [ ] Normalize input via `normalize_spike_times()`
  - [ ] Precompute shared quantities (egocentric occupancy, polar bins)
  - [ ] Add `n_jobs` parameter for parallelization
  - [ ] Return `EgocentricRatesResult`

- [ ] **5.9** Write comprehensive tests for egocentric encoding
  - [ ] Test single neuron computation
  - [ ] Test batch computation
  - [ ] Test all result class methods
  - [ ] Test euclidean vs geodesic distance
  - [ ] Test error when geodesic requested without env
  - [ ] Test `to_dataframe()` output format

---

## Milestone 6: Optional JAX Backend

**Goal**: Add JAX acceleration for batch computations.

**Dependencies**: Milestones 2-5 complete (all NumPy implementations working)

**Success Criteria**: With JAX installed, `backend="jax"` produces identical results to `backend="numpy"` with better performance on large populations.

### Tasks

- [ ] **6.1** Implement JAX smoothing operations in `_core_jax.py`
  - [ ] Port `diffusion_kde` smoothing to JAX
  - [ ] Port `gaussian_kde` smoothing to JAX
  - [ ] Verify numerical equivalence with NumPy

- [ ] **6.2** Implement JAX metric computations in `_core_jax.py`
  - [ ] Port `spatial_information` to JAX
  - [ ] Port `sparsity` to JAX
  - [ ] Use `vmap` for batch operations
  - [ ] Verify numerical equivalence with NumPy

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
