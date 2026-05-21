# Changelog

## [0.4.0] - 2026-05-21

This release is the v0.4 UX cleanup: a wide-ranging consolidation of
public-API names, argument orders, return types, error semantics, and
example coverage. There are **no deprecations**; every change is a
clean delete-and-replace. Pin to `<0.4.0` if you need the old surface.

### Breaking changes

#### Parameter renames

- **`distance_metric` / `distance_type` / `use_geodesic` → `metric`** across
  all physical-distance APIs. Legal values are `{"euclidean", "geodesic"}`.
  Affects `Environment.distance_to`, `compute_egocentric_rate(s)`,
  `compute_egocentric_distance`, `compute_spatial_rate(s)`,
  `ObjectVectorCellModel`, `PlaceCellModel`, `BoundaryCellModel`, and the
  boundary / border modules.
- **`smoothing_sigma` / `kernel_bandwidth` → `bandwidth`** across smoothing
  APIs (`compute_spatial_rate(s)`, `compute_view_rate(s)`,
  `compute_egocentric_rate(s)`, `Environment.smooth`, KDE helpers).
- **`velocity_threshold` / `speed_threshold` / `threshold` → `min_speed`**
  across velocity-based behaviour segmentation
  (`segment_by_velocity`, `heading_from_velocity`, etc.).
- **Overlay `data=` → semantic name.** `PositionOverlay(data=...)` →
  `PositionOverlay(positions=...)`; `HeadDirectionOverlay(data=...)` →
  `HeadDirectionOverlay(headings=...)`.

#### Result-class field renames

- **`EgocentricRateResult.ego_env` → `env`**;
  **`ViewRateResult.view_occupancy` → `occupancy`**.
- **`PeriEventResult.firing_rate`** is now a cached attribute, not a method.
  Replace `result.firing_rate()` with `result.firing_rate`.
- **`DecodingResult.uncertainty` → `posterior_entropy`** (matches the free
  function in `decoding/estimates.py`).
- **Singular vs plural method/attribute normalization** on result classes
  (single-neuron results use singular methods; batch results use plural).
  Renames: `SpatialRateResult.peak_firing_rates()` → `peak_firing_rate()`,
  `SpatialRateResult.peak_locations()` → `peak_location()`,
  `ViewRateResult.peak_view_locations()` → `peak_view_location()`.
- **`is_X_cell` method names** normalized to match the free-function names
  (`is_object_vector_cell`, `is_spatial_view_cell`,
  `is_head_direction_cell`).

#### Argument-order canonicalization

- **Encoding functions are now `env`-first**, with the canonical order
  `(env, spike_times, times, positions, headings?, object_positions?, *, ...)`.
  Affects `compute_spatial_rate(s)`, `compute_egocentric_rate(s)`,
  `compute_view_rate(s)`, `detect_place_fields`, `is_spatial_view_cell`,
  `is_object_vector_cell`, `is_border_cell`, and friends.

  ```python
  # v0.3
  compute_spatial_rate(spike_times, times, positions, env)
  compute_egocentric_rate(spike_times, times, positions, headings,
                          object_positions, env=env)
  # v0.4
  compute_spatial_rate(env, spike_times, times, positions)
  compute_egocentric_rate(env, spike_times, times, positions,
                          headings, object_positions)
  ```

- **`compute_directional_rate` / `is_head_direction_cell`** keep the
  heading-domain-native `(spike_times, times, headings, *, ...)`
  signature — this is the documented exception to the env-first rule
  (heading is a circular angular variable, not a spatial position).
  See the function docstrings and `CLAUDE.md` "Canonical Argument
  Order".
- **Egocentric ops** `allocentric_to_egocentric` /
  `egocentric_to_allocentric` reorder to `(positions, headings, targets)`.
- **Behavioural segmentation** functions reordered to
  `(position_bins, times, env, *, ...)`.
- **`distance_to_reward`** in `events.regressors` reordered to
  `(env, times, positions, reward_times, ...)`.
- **`fit_isotonic_trajectory` / `fit_linear_trajectory`** reordered to
  `(env, posterior, times, *, ...)` with a standardized `method`
  keyword.
- **`*` keyword-only separator** added consistently across the public
  API. Numerical parameters and verbose flags become keyword-only.

#### Coordinate / convention changes

- **`Environment.is_1d` → `Environment.is_linearized_track`.** Same
  semantics (a 1-D graph track embedded in 2-D world coordinates); the
  new name resolves the historical "is this n_dims==1 or a 2-D track?"
  ambiguity. Serialized environment metadata uses the new key — pre-v0.4
  saved environments will need to be re-saved.
- **`GridProperties.peak_coords` is now `(x_offset, y_offset)`** instead
  of `(row_offset, col_offset)`. Swap `peak_coords[:, 0]` (was row) for
  `peak_coords[:, 1]` (now y) when reading the second component.
- **`simulate_trajectory_ou(speed_units=...)`** is now required (was
  defaulted). Speed defaults switch from m/s to cm/s. Mismatch between
  `speed_units` and `env.units` raises rather than silently rescaling.

#### Removed (no aliases, no deprecation)

- **`Environment.save` / `Environment.load`.** The pickle path is gone.
  Use `Environment.to_file` / `Environment.from_file` (JSON metadata
  plus npz arrays).
- **`Environment.mask_for_region`.** Use `Environment.region_mask`.
- **`from_image` / `from_mask` factory aliases.** Replaced by
  `from_pixel_mask(image_mask, pixel_size, ...)` and
  `from_grid_mask(active_mask, grid_edges, ...)`.
- **`path_efficiency` (float-returning).** Use `compute_path_efficiency`
  which returns a `PathEfficiencyResult`.
- **Cross-domain re-exports.** Each public symbol now has exactly one
  canonical import path; the top-level `neurospatial` namespace no
  longer re-exports symbols from `encoding`, `decoding`, etc.

### Added

- **`PlaceFieldsResult` dataclass.** `detect_place_fields` returns a
  frozen dataclass with `fields`, `excluded_reason`, and `n_excluded`
  fields. Still iterable / sized / indexable, so existing `for f in
  detect_place_fields(...)` and `len(...)` patterns keep working.
  Closes the "silent drop when mean rate too high" failure mode.
- **`BinSequenceWithRuns` dataclass + new method.**
  `Environment.bin_sequence` always returns an `ndarray`;
  `Environment.bin_sequence_with_runs` returns a dataclass with `bins`,
  `run_starts`, `run_lengths`.
- **`MSDResult` and friends.** Misc result-type cleanup in trajectory
  analysis: `MSDResult`, `SpatialAutocorrelationResult`,
  `PathEfficiencyResult`.
- **`Environment.is_polar` property and `coordinate_kind` attribute.**
  `from_polar_egocentric` sets `coordinate_kind="polar"`. Methods that
  assume Cartesian (`distance_to`, `distance_between`,
  `Environment.contains`, `apply_transform`, `bin_at` on `(x, y)`
  input) raise on polar environments with a clear error.
  `plot_field` switches axis labels and skips the equal-aspect call so
  egocentric polar firing fields still render correctly.
- **Custom exception classes.** `EnvironmentNotFittedError` (already
  existed) now has a free-function variant; added `RegionNotFoundError`,
  `RegionAlreadyExistsError`, and three more in `_exceptions.py`.
- **`Environment.from_pixel_mask` and `Environment.from_grid_mask`
  factories** (replacing `from_image` / `from_mask`).
- **`Environment._state_version` invalidation token.** Cached
  properties verify the version on access; subset / transform / rebin
  bump it, so stale caches are surfaced loudly instead of returning
  silently-wrong results.
- **`Environment.__str__` returns `info()`** for quick inspection.
- **Glossary page** at [docs/glossary.md](docs/glossary.md) defining 14
  core terms. Linked from `docs/getting-started/core-concepts.md` and
  the README.
- **`docs/api/index.md` expansion.** Structured sections for
  `encoding`, `decoding`, `behavior`, `events`, `ops.egocentric`,
  `ops.visibility`, `ops.basis`, `stats`, `animation`, `io.nwb`.
- **`docs/examples/index.md` rewrite.** Goal → notebook table plus
  full per-notebook entries with Time + Prerequisites.
- **Notebooks 24–27.** Object-vector cells, head-direction tuning,
  peri-event PSTH, and NWB loading round-trip.
- **README "Your First Place Field" front-door example.** Canonical
  pattern using `simulate_trajectory_ou`, `PlaceCellModel`,
  `generate_population_spikes`, and `compute_spatial_rate`.
- **CI doc-snippet test.** `scripts/test_doc_snippets.py` plus
  `.github/workflows/test_docs.yml` re-executes a curated manifest of
  doc snippets on every PR.
- **CI notebook regen test.** `.github/workflows/test_notebooks.yml`
  re-executes `11_place_field_analysis.ipynb` per PR to catch silent
  regressions in the example surface.
- **Shared example styling.** `examples/_style.py` Wong / Okabe-Ito
  palette and fixed figure sizes. Wired into the four new advanced
  notebooks (24-27); legacy notebooks 01-22 keep their inline rcParams
  blocks for now.

### Changed

- **Silent failures replaced with loud failures.**
  - `subset()` round-trip now returns a `MaskedGrid` instead of a one-off
    `subset` layout kind, so the result is fully serializable.
  - `bin_at` vs `map_points_to_bins` standardize on `-1` for
    out-of-environment samples in trajectory contexts.
  - `detect_place_fields` returns a `PlaceFieldsResult` with
    `excluded_reason` set instead of silently returning `[]`.
  - `batch_grid_scores` / `batch_border_scores` use NaN as the explicit
    failure marker and warn once per batch.
  - Fitted-state checks at entry of `compute_spatial_rate(s)`,
    `compute_egocentric_rate(s)`, `compute_view_rate(s)`,
    `decode_position` raise immediately instead of failing deep in the
    call stack.
  - `spike_times` validation rejects unsorted / negative / non-finite
    values with diagnostic messages.
  - `decode_position(validate=True)` is the default; rejects negative
    spike counts and posteriors that don't sum to 1.
- **Canonical exception types** throughout. Manual "not fitted" checks
  migrated to `EnvironmentNotFittedError`; warning-and-overwrite paths
  in `Regions.__setitem__` now raise.
- **Errors carry units and stack context.** Length-mismatch errors
  from `_binning` include a `context` arg so messages say "in
  compute_spatial_rate: ..."; magnitude errors include the offending
  unit.
- **Warning hygiene.** `UserWarning` for data-quality, `RuntimeWarning`
  for numerical fallbacks, `stacklevel=2` everywhere.
- **Production `print()` calls** replaced with module-level
  `logger.info` / `logger.debug`.
- **`Environment.bin_attributes`, `edge_attributes`,
  `differential_operator`** converted from `@cached_property` to
  methods (`get_bin_attributes()`, etc.) so the cost is visible.
- **`Environment.units`** validated against a small registry (`{"cm",
  "m", "mm", "px", None}`) with a `UserWarning` for unknown values.
  Documented as advisory.
- **Heading convention** documented explicitly in every function that
  takes a `headings` argument (allocentric world-frame: 0 = East,
  +π/2 = North; egocentric for OVC tuning: 0 = ahead, +π/2 = left).
- **`events.__init__`** is now eager (was lazy).
- **Bandit-task notebook** prints the download URL and exits cleanly
  when `data/` is missing; CI no longer fails on the example.

### Fixed

- **`repr(env)` `name=None` bug** for empty-string names. Now uses
  `repr(self.name)` so empty strings are visible as `''`.
- **`Environment._state_version` cache invalidation** prevents
  stale-cache reads after mutating operations.
- **Polar environment misuse** is now an error instead of producing
  silently-wrong distances or transforms.

### Removed

- **`Environment.save` / `Environment.load`** (pickle). Replaced by
  `to_file` / `from_file`.
- **`Environment.mask_for_region`.** Use `region_mask`.
- **`from_image` / `from_mask` factory aliases.** Replaced by
  `from_pixel_mask` / `from_grid_mask`.
- **`path_efficiency` float-returning function.** Use
  `compute_path_efficiency`.
- **All cross-domain re-exports** from top-level `neurospatial`.

### Major feature additions (v0.3.x development cycle)

The following features were developed during the v0.3.x development
line and ship as part of v0.4.0. The names below reflect the final
v0.4 surface — many of these symbols were introduced under earlier
names that were renamed during the M2 consolidation pass (see
**Breaking changes** above for the v0.3 → v0.4 mapping).

#### Added (features)

- **Spatial View Cells**: Firing-rate fields indexed by gaze location
  - `compute_view_rate()` / `compute_view_rates()` — single / batch
  - `ViewRateResult` frozen dataclass (`firing_rate`, `occupancy`,
    `env`, plus `view_spatial_information()`, `is_spatial_view_cell()`,
    `sparsity()`, `selectivity()` methods)
  - `is_spatial_view_cell()` free-function classifier
  - `SpatialViewCellModel` simulation model with three gaze models
    (`fixed_distance`, `ray_cast`, `boundary`)

- **Visibility and Gaze**: Ray-casting visibility for view cells
  - `FieldOfView` frozen dataclass with species presets
    (`FieldOfView.rat()`, `FieldOfView.primate()`)
  - `ViewshedResult` frozen dataclass (visible bins + visibility
    fraction)
  - `compute_viewed_location()` — gaze-directed location projection
  - `compute_viewshed()` / `compute_viewshed_trajectory()` /
    `compute_view_field()` — observer-position visibility analysis
  - `visible_cues()` — line-of-sight check to cue / landmark positions
  - `visibility_occupancy()` — time each bin was visible

- **Object-Vector Cells**: Firing fields in egocentric polar coordinates
  - `compute_egocentric_rate()` / `compute_egocentric_rates()`
  - `EgocentricRateResult` / `EgocentricRatesResult` frozen dataclasses
    (`firing_rate`, `occupancy`, `env`, plus `preferred_distance()`,
    `preferred_direction()`, `egocentric_spatial_information()`,
    `is_object_vector_cell()` methods)
  - `object_vector_score()` — distance × direction selectivity metric
  - `is_object_vector_cell()` free-function classifier
  - `plot_object_vector_tuning()` — polar heatmap visualization
  - `ObjectVectorCellModel` simulation model with von Mises directional
    tuning
  - `ObjectVectorOverlay` for animation with object–animal vectors

- **Egocentric Reference Frames**: Foundation for object-vector and view cells
  - `EgocentricFrame` dataclass with `to_egocentric()` / `to_allocentric()`
  - `allocentric_to_egocentric()` / `egocentric_to_allocentric()` —
    batch frame transforms
  - `compute_egocentric_bearing()` — angle to targets relative to
    heading (egocentric convention: 0 = ahead, +π/2 = left)
  - `compute_egocentric_distance()` — Euclidean and geodesic distance
  - `heading_from_velocity()` / `heading_from_body_orientation()` —
    derive heading from tracking data
  - `Environment.from_polar_egocentric()` — egocentric polar coordinate
    environment with optional circular connectivity

- **3D Transform Support**: Full N-dimensional affine transformation capabilities
  - New `AffineND` class for N-dimensional affine transforms using (N+1)×(N+1) homogeneous matrices
  - `Affine3D` type alias for convenience (equivalent to `AffineND` with n_dims=3)
  - 3D factory functions: `translate_3d()`, `scale_3d()`, `from_rotation_matrix()`
  - Integration with `scipy.spatial.transform.Rotation` for 3D rotations
  - `estimate_transform()` now auto-detects dimensionality (2D or 3D) from input points
  - `apply_transform_to_environment()` supports N-dimensional environments with validation
  - 45 new tests for 3D transforms including scipy integration
  - Backward compatible: `Affine2D` unchanged for existing 2D workflows

- **N-D Probability Mapping**: `alignment.py` now accepts N×N rotation matrices
  - `map_probabilities_to_nearest_target_bin()` supports 3D environments
  - Updated `_transform_source_bin_centers()` validates rotation matrix dimensionality
  - For 2D: Use `get_2d_rotation_matrix(angle_degrees)`
  - For 3D: Use `scipy.spatial.transform.Rotation.as_matrix()`

#### Changed (internal)

- **Internal**: Refactored `environment.py` (5,335 lines) into modular package structure for improved maintainability
  - Split into 11 focused modules: `core.py` (1,023 lines), `factories.py` (630 lines), `queries.py` (897 lines), `trajectory.py` (1,222 lines), `transforms.py` (634 lines), `fields.py` (564 lines), `metrics.py` (469 lines), `regions.py` (398 lines), `serialization.py` (315 lines), `visualization.py` (211 lines), `decorators.py` (77 lines)
  - Implemented mixin pattern: Environment inherits from all functionality mixins
  - No breaking changes - `from neurospatial import Environment` continues to work
  - All 1,076 tests passing (100% success rate)
  - Improved code organization for easier contribution and maintenance
  - Largest module is trajectory.py at 1,222 lines (down from original analysis.py at 2,104 lines)

#### Documentation

- **3D Support**: Updated dimensionality support documentation to reflect 3D transforms availability
  - Updated `docs/dimensionality_support.md` with 3D transform examples and feature matrix
  - Updated `docs/user-guide/alignment.md` with comprehensive 3D transformation examples
  - Added complete 3D alignment workflow example
  - Updated compatibility matrix: ~75% of neurospatial now works in 3D (up from 70%)

## [v0.2.0] - 2025-11-04

## What's Changed

### Features

- feat(P3.15): implement deterministic KDTree with distance thresholds (7ef1109)
- feat(P3.14): implement Environment.copy() method (17a1d0e)
- feat(P3.13): implement distance utilities (distance_to and rings) (2e07dce)
- feat(P3.12): implement Environment.region_membership() for vectorized region containment (a02d926)
- feat(P2.11): implement linear time allocation for occupancy (d1e3690)
- feat(P2.10): implement field math utility functions (9f8e53b)
- feat(P2.9): implement Environment.interpolate() for field evaluation at points (160da81)
- feat(P1.8): implement Environment.subset() for bin selection and cropping (7269e86)
- feat(P1.7): implement Environment.rebin() for grid coarsening (d841d3d)
- feat(P1.6): implement Environment.smooth() for field smoothing (dc4ce39)
- feat(P0.4): implement connected components and reachability methods (7a1d75f)
- feat(P0.3): implement Environment.transitions() for empirical transition matrices (a9f3beb)
- feat(P0.2): implement Environment.bin_sequence() for trajectory-to-bin conversion (f0e008b)
- feat(P0.1): implement Environment.occupancy() for time-in-bin computation (28ccae0)
- feat(kernels): implement diffusion kernel infrastructure (Phase 1) (ad5f23f)
- feat(ci): add manual workflow dispatch to publish workflow (703ed86)

### Bug Fixes

- fix(lint): resolve ruff errors in example notebook (6b69b2e)
- fix(GraphLayout): support 1D graph layouts (26b8abc)
- fix(P2.11): combine nested if statements, apply ruff format (5bddcdf)
- fix(P0.3): add parameter validation for transitions() method (dad0f96)

### Documentation

- docs: mark all Environment Operations tasks complete (46e65ab)
- docs: add jupytext pairing and track all example notebooks (a341e36)
- docs: mark P3.15 (Deterministic KDTree) as complete (eda4f9b)
- docs: update CHANGELOG.md for v0.1.0 (84e8a46)

### Other Changes

- chore: remove completed project management files (66666e7)
- test: fix disconnected graph tests using systematic debugging (8dc6de6)
- refactor(test): remove untestable unfitted Environment check_fitted test (03b3722)
- refactor(test): remove untestable 1D graph region_membership test (a20bbbd)
- test: implement hexagonal layout interpolation test (233f501)
- chore: mark public API export task complete in TASKS.md (9fbb8eb)
- chore: sync notebooks with formatting changes from pre-commit (f8b7528)
- refactor(P0.3): add unified transitions() interface with model-based methods (d27a1e9)

**Full Changelog**: <https://github.com/edeno/neurospatial/compare/v0.1.0...v0.2.0>

All notable changes to the neurospatial project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-11-04

### Added

#### Environment Operations (Complete Feature Set)

**Core Analysis Operations (P0)**

- `Environment.occupancy()` - Compute time-in-bin from trajectory data with speed filtering, gap handling, and optional kernel smoothing
- `Environment.bin_sequence()` - Convert trajectories to bin sequences with run-length encoding
- `Environment.transitions()` - Compute empirical transition matrices with adjacency filtering and normalization
- `Environment.components()` - Find connected components in environment graph
- `Environment.reachable_from()` - Compute reachable bins via BFS or geodesic distance

**Smoothing & Resampling (P1)**

- `Environment.smooth()` - Apply diffusion kernel smoothing to arbitrary fields
- `Environment.rebin()` - Conservative grid coarsening with mass/mean aggregation (grid-only)
- `Environment.subset()` - Extract subregions by bins, regions, or polygons

**Interpolation & Field Utilities (P2)**

- `Environment.interpolate()` - Evaluate bin-valued fields at continuous points (nearest/linear modes)
- `Environment.occupancy()` linear mode - Ray-grid intersection for accurate boundary handling (grid-only)
- `field_ops.py` module:
  - `normalize_field()` - Normalize to probability distribution
  - `clamp()` - Bound field values
  - `combine_fields()` - Weighted combination (mean/max/min)
  - `divergence()` - KL/JS divergence and cosine distance

**Utilities & Polish (P3)**

- `Environment.region_membership()` - Vectorized bin-to-region containment checks
- `Environment.distance_to()` - Compute distances to target bins or regions (Euclidean/geodesic)
- `Environment.rings()` - K-hop neighborhoods via BFS layers
- `Environment.copy()` - Deep/shallow copying with cache invalidation
- `spatial.map_points_to_bins()` - Enhanced with `max_distance` and `max_distance_factor` thresholds for deterministic boundary decisions

**Diffusion Kernel Infrastructure**

- `kernels.py` module:
  - `compute_diffusion_kernels()` - Matrix-exponential heat kernel on graphs with volume correction
  - `Environment.compute_kernel()` - Convenience wrapper with caching
  - Support for both transition and density normalization modes

**Documentation**

- `docs/user-guide/spatial-analysis.md` - Comprehensive 1,400+ line guide covering all operations with scientific context
- `docs/examples/08_complete_workflow.ipynb` - Enhanced workflow notebook with movement/navigation analysis
- All methods have NumPy-style docstrings with working examples
- "Why This Matters" sections explaining scientific motivation for key operations

### Changed

- **GraphLayout**: Now supports 1D layouts correctly (conditional `angle_2d`, dynamic `dimension_ranges`)
- **KDTree operations**: Now deterministic by default using `tie_break="lowest_index"`
- **All environment operations**: Use `@check_fitted` decorator for consistent state enforcement
- **Input validation**: Comprehensive validation with diagnostic error messages across all operations
- **Caching**: Object identity-based caching for kernels and spatial queries

### Fixed

- GraphLayout `angle_2d` computation for 1D graphs (was unconditionally assuming 2D)
- GraphLayout `dimension_ranges` now correctly handles 1D case
- Disconnected graph handling in connectivity tests
- Hexagonal layout interpolation edge cases

### Testing

- **1067 tests passing** (up from 614 in v0.1.0)
- **0 skipped tests** (eliminated all 12 previous skips)
- Performance benchmarks: occupancy on 100k samples, large transition matrices, kernel computation
- Integration tests: end-to-end workflows, multi-layout compatibility
- Edge case coverage: empty environments, single bins, disconnected graphs

### Internal

- Systematic debugging skill used to eliminate all test skips
- Test-driven development for all features
- Code review and UX review completed
- Pre-commit hooks for code quality

## [0.1.0] - 2025-11-03

### Added

- **CompositeEnvironment API parity**: Added `bins_in_region()`, `mask_for_region()`, `shortest_path()`, `info()`, `save()`, and `load()` methods to CompositeEnvironment for full API compatibility with Environment class
- **KDTree-optimized spatial queries**: CompositeEnvironment.bin_at() now uses KDTree for O(M log N) performance instead of O(N×M) sequential queries (enabled by default via `use_kdtree_query=True`)
- **Structured logging infrastructure**: New `_logging.py` module with NullHandler by default, enabling optional logging for debugging and workflow tracing
- **Centralized numerical constants**: New `_constants.py` module consolidating all magic numbers (tolerances, KDTree parameters, epsilon values) for consistent behavior
- **Comprehensive type validation**: CompositeEnvironment constructor now validates input types with actionable error messages
- **Graph metadata validation**: Added `validate_connectivity_graph()` to enforce required node/edge attributes from layout engines
- **Dimensionality support documentation**: New `docs/dimensionality_support.md` clarifying 1D/2D/3D feature support with compatibility matrix

### Changed

- **Updated alignment module**: Now uses centralized constants (`IDW_MIN_DISTANCE`, `KDTREE_LEAF_SIZE`)
- **Updated regions module**: Uses `POINT_TOLERANCE` constant for consistent geometric comparisons
- **Enhanced error messages**: CompositeEnvironment now provides detailed diagnostics for dimension mismatches and type errors
- **Clarified 2D-only transforms**: Updated `transforms.py` docstring to explicitly state 2D-only status and suggest scipy for 3D

### Fixed

- Removed unused `type: ignore` comment in `regular_grid.py`
- Fixed potential `KeyError` in logging by renaming `name` parameter to `env_name` (avoids conflict with LogRecord reserved field)

### Documentation

- Added comprehensive dimensionality support guide (1D/2D/3D feature matrix)
- Updated CLAUDE.md with latest patterns and requirements
- Added 18 new tests for CompositeEnvironment type validation
- Added 23 new tests for Environment error path coverage
- Added 28 new tests for graph validation

### Internal

- Consolidated duplicate dimension inference code
- All 614 tests passing
- Ruff and mypy checks passing
- Test coverage: 78%
