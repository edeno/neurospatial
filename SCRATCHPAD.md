# SCRATCHPAD.md

**Current Focus**: Milestone 2 - Object-Vector Cells

## Session: 2025-12-05

### Completed: M1.1 Core Reference Frame Module

**Files Created:**

- `src/neurospatial/reference_frames.py` - Core module with all functions
- `tests/test_reference_frames.py` - 35 comprehensive tests

**Implemented:**

- [x] `EgocentricFrame` dataclass with `to_egocentric()` / `to_allocentric()`
- [x] `allocentric_to_egocentric()` - batch transform (n_time, n_points, 2)
- [x] `egocentric_to_allocentric()` - inverse batch transform
- [x] `compute_egocentric_bearing()` - angle to targets relative to heading
- [x] `compute_egocentric_distance()` - Euclidean and geodesic metrics
- [x] `heading_from_velocity()` - compute heading from position timeseries
- [x] `heading_from_body_orientation()` - heading from pose keypoints
- [x] `_wrap_angle()` - wrap angles to (-π, π]
- [x] `_interpolate_heading_circular()` - circular interpolation for NaN gaps

**Key Design Decisions:**

- Allocentric: 0=East, π/2=North (standard mathematical convention)
- Egocentric: 0=ahead, π/2=left, -π/2=right, ±π=behind
- Used `np.einsum('tij,tpj->tpi', rot, centered)` for vectorized rotation
- Circular interpolation via unit vectors for heading gaps (avoids ±π discontinuity)
- Geodesic distance uses `neurospatial.distance.distance_field()`

**Tests:**

- 35/35 passing
- Module structure tests
- Dataclass tests
- Batch transform tests
- Bearing computation tests
- Distance computation tests (Euclidean and geodesic)
- Heading computation tests with edge cases

**Code Quality:**

- ruff: All checks pass
- mypy: No issues found
- Code review: Approved after type safety fixes

### Completed: M1.3 Egocentric Polar Environment

**Files Modified/Created:**

- `src/neurospatial/environment/factories.py` - Added `from_polar_egocentric()` factory method
- `tests/test_polar_egocentric.py` - 21 comprehensive tests

**Implemented:**

- [x] `from_polar_egocentric()` classmethod with parameters:
  - `distance_range`, `angle_range` - define the polar space bounds
  - `distance_bin_size`, `angle_bin_size` - control bin resolution
  - `circular_angle=True` - enables periodic connectivity for full-circle angles
- [x] `_add_circular_connectivity()` helper - connects first/last angle bins at each distance ring
- [x] Comprehensive parameter validation with clear error messages
- [x] NumPy-style docstring with coordinate convention and examples

**Key Design Decisions:**

- Reuses existing `from_mask()` infrastructure (MaskedGridLayout)
- Coordinate convention matches `reference_frames`: angle 0=ahead, π/2=left, -π/2=right
- Circular edges include proper `edge_id` attribute (critical for graph consistency)
- `bin_centers[:, 0]` = distances, `bin_centers[:, 1]` = angles

**Tests:**

- 21/21 passing
- Basic creation tests (n_bins, dimensions, value ranges)
- Circular connectivity tests (wrapping, edge attributes)
- Parameter validation tests (bin sizes, ranges)
- Edge case tests (single angle bin, equal range bounds)

**Code Quality:**

- ruff: All checks pass
- mypy: No issues found
- Code review: Approved after edge_id fix

### Completed: M1.5 Integration and Documentation

**Files Modified:**

- `.claude/QUICKSTART.md` - Added "Egocentric Reference Frames" section with:
  - Heading computation examples (from velocity and pose keypoints)
  - Allocentric to egocentric transform examples
  - Bearing and distance computation examples
  - Coordinate convention documentation
  - Egocentric polar environment creation example
- `.claude/API_REFERENCE.md` - Added "Egocentric Reference Frames (v0.17.0+)" section with:
  - All exports from `neurospatial` top-level
  - Full `neurospatial.reference_frames` module API
  - `Environment.from_polar_egocentric()` factory example

**Milestone 1 Complete!** All tasks in M1 (Egocentric Reference Frames) are now done.

### Completed: M2.1 Object-Vector Cell Model (Simulation)

**Files Created:**

- `src/neurospatial/simulation/models/object_vector_cells.py` - Main implementation
- `tests/simulation/models/test_object_vector_cells.py` - 32 comprehensive tests

**Implemented:**

- [x] `ObjectVectorCellModel` class with parameters:
  - `env`, `object_positions` - required spatial setup
  - `preferred_distance`, `distance_width` - Gaussian distance tuning
  - `preferred_direction`, `direction_kappa` - optional von Mises direction tuning
  - `object_selectivity` - "any", "nearest", or "specific" modes
  - `max_rate`, `baseline_rate` - firing rate parameters
  - `distance_metric` - "euclidean" or "geodesic"
- [x] Parameter validation in `__init__` with clear error messages
- [x] Warning for objects outside environment bounds
- [x] Geodesic distance field precomputation
- [x] `firing_rate()` method with:
  - Gaussian distance tuning: `exp(-0.5 * ((d - pref) / width)^2)`
  - von Mises direction tuning: `exp(κ * cos(θ - θ_pref)) / exp(κ)`
  - Uses `compute_egocentric_bearing()` from M1
  - Three aggregation modes: any/nearest/specific
- [x] `ground_truth` property returning all model parameters

**Key Design Decisions:**

- Class-based (not dataclass) following `PlaceCellModel` pattern
- Direction tuning requires explicit `headings` parameter (not from positions)
- Von Mises normalization: divide by `exp(κ)` so peak response = 1
- Geodesic mode precomputes distance field per object for O(1) lookup
- Euclidean is default (faster, sufficient for open fields)

**Tests:**

- 32/32 passing
- Module structure tests (imports, docstring)
- Parameter validation tests (10 tests for error messages)
- Distance tuning tests (Gaussian shape verified)
- Direction tuning tests (von Mises decreasing away from preferred)
- Object selectivity tests (any/nearest/specific modes)
- Distance metric tests (euclidean/geodesic)
- Ground truth tests (completeness, immutability)
- Protocol compliance (implements NeuralModel)

**Code Quality:**

- ruff: All checks pass
- mypy: No issues found
- Code review: Approved - high quality scientific code
- 38% documentation ratio (522 LOC, ~200 lines of docstrings)
- Complete type annotations

**Module Updated:**

- `src/neurospatial/simulation/models/__init__.py` - Added `ObjectVectorCellModel` export

### Completed: M2.2 Object-Vector Metrics

**Files Created:**

- `src/neurospatial/metrics/object_vector_cells.py` - Main implementation (520 LOC)
- `tests/metrics/test_object_vector_cells.py` - 53 comprehensive tests

**Implemented:**

- [x] `ObjectVectorMetrics` frozen dataclass with fields:
  - `preferred_distance`, `preferred_direction` - peak tuning location
  - `distance_selectivity`, `direction_selectivity` - individual selectivity metrics
  - `object_vector_score` - combined selectivity score [0, 1]
  - `peak_rate`, `mean_rate` - firing rate statistics
  - `tuning_curve`, `distance_bins`, `direction_bins` - full 2D tuning data
  - `interpretation()` method returning human-readable string
  - `__str__()` returns interpretation
- [x] `compute_object_vector_tuning()` function:
  - Bins spikes by egocentric (distance, direction) to nearest object
  - Uses `compute_egocentric_bearing()` from M1
  - Computes occupancy-normalized firing rates
  - Applies `min_occupancy_seconds` threshold (default 0.1s)
  - Returns `ObjectVectorMetrics` dataclass
- [x] `object_vector_score()` function:
  - Distance selectivity: `s_d = peak / mean`
  - Direction selectivity: uses `_mean_resultant_length` from `metrics.circular`
  - Combined: `s_OV = ((s_d - 1) / (s_d* - 1)) * s_θ`
  - Validates `max_distance_selectivity > 1`
- [x] `is_object_vector_cell()` classifier:
  - Checks `score >= threshold` and `peak_rate >= min_rate`
  - Default thresholds: score=0.3, min_peak=5.0 Hz
- [x] `plot_object_vector_tuning()` visualization:
  - Polar heatmap (distance=radial, direction=angular)
  - Optional peak location marker
  - Optional colorbar
  - Configurable colormap

**Key Design Decisions:**

- API pattern mirrors `head_direction.py` (dataclass + compute + classifier + plot)
- Uses frozen dataclass for immutable metrics
- Egocentric coordinate convention: 0=ahead, π/2=left, -π/2=right
- Nearest object used by default (consistent with biological OVCs)
- Combined score formula from Hoydal et al. (2019)

**Tests:**

- 53/53 passing
- Module structure tests
- Dataclass tests (fields, frozen, interpretation)
- Tuning computation tests (binning, occupancy, normalization)
- Score computation tests (distance, direction, combined)
- Classifier tests (thresholds, edge cases)
- Visualization tests (polar projection, peak marking)
- Ground truth recovery tests (simulated OVC → metrics → classification)

**Code Quality:**

- ruff: All checks pass
- mypy: No issues found
- Code review: Approved - excellent scientific code
- NumPy docstrings with full examples
- Complete type annotations

**Module Updated:**

- `src/neurospatial/metrics/__init__.py` - Added all exports:
  - `ObjectVectorMetrics`, `compute_object_vector_tuning`
  - `object_vector_score`, `is_object_vector_cell`, `plot_object_vector_tuning`

### Completed: M2.3 Object-Vector Field Computation

**Files Created:**

- `src/neurospatial/object_vector_field.py` - Main implementation (~430 LOC)
- `tests/test_object_vector_field.py` - 27 comprehensive tests

**Implemented:**

- [x] `ObjectVectorFieldResult` frozen dataclass with fields:
  - `field` - firing rate (Hz) in each egocentric polar bin
  - `ego_env` - egocentric polar coordinate environment
  - `occupancy` - time spent (seconds) in each bin
- [x] `compute_object_vector_field()` function:
  - Creates egocentric polar environment via `from_polar_egocentric()`
  - Computes distance and bearing to nearest object at each timepoint
  - Supports geodesic distance when `allocentric_env` provided
  - Computes occupancy in egocentric polar space (vectorized)
  - Bins spikes by egocentric position at spike time (vectorized)
  - Normalizes by occupancy, applies `min_occupancy_seconds` threshold
  - Supports methods: "binned", "diffusion_kde"
  - Returns `ObjectVectorFieldResult`

**Key Design Decisions:**

- Uses `compute_egocentric_bearing()` from M1 for direction computation
- Nearest object selection (consistent with OVC biology)
- Vectorized bin assignment for performance (np.add.at)
- diffusion_kde uses kernel matrix from ego_env (respects circular boundaries)
- Literal types for method and distance_metric parameters

**Tests:**

- 27/27 passing
- Module structure tests (imports, docstring, exports)
- Dataclass tests (fields, frozen)
- Field computation tests (shape, polar coordinates, occupancy)
- Smoothing method tests (binned, diffusion_kde)
- Validation tests (empty spikes, mismatched lengths, invalid method)
- Geodesic distance tests (with allocentric_env)
- Ground truth recovery tests (simulated OVC → field peaks at preferred location)

**Code Quality:**

- ruff: All checks pass
- mypy: No issues found
- Code review: Approved after critical fixes
- NumPy docstrings with coordinate convention docs
- Complete type annotations with Literal types

**Module Updated:**

- `src/neurospatial/__init__.py` - Added exports:
  - `ObjectVectorFieldResult`, `compute_object_vector_field`

### Completed: M2.4 Object-Vector Overlay (Animation)

**Files Modified:**

- `src/neurospatial/animation/overlays.py` - Added `ObjectVectorOverlay` class and `ObjectVectorData` container
- `src/neurospatial/animation/__init__.py` - Added exports
- `tests/animation/test_object_vector_overlay.py` - 25 comprehensive tests

**Implemented:**

- [x] `ObjectVectorOverlay` dataclass with parameters:
  - `object_positions`, `animal_positions` - required spatial setup
  - `firing_rates`, `times` - optional modulation and timing
  - `color`, `linewidth`, `show_objects`, `object_marker`, `object_size` - styling
  - `interp` ("linear", "nearest") - interpolation method
- [x] `ObjectVectorData` internal container:
  - `object_positions` - static object positions (n_objects, n_dims)
  - `animal_positions` - aligned to frames (n_frames, n_dims)
  - `firing_rates` - optional, aligned to frames (n_frames,)
  - Styling fields passed through
- [x] `convert_to_data()` method:
  - Validates object_positions and animal_positions (finite values, shape)
  - Validates firing_rates (finite values, length match)
  - Aligns animal_positions and firing_rates to frame times
  - Warns for positions outside environment bounds
- [x] Integration with OverlayData pipeline:
  - Updated OverlayProtocol return type
  - Added object_vectors list to OverlayData container
  - Updated _convert_overlays_to_data() dispatch

**Key Design Decisions:**

- Added to existing `overlays.py` (not a separate file) to match codebase pattern
- Follows exact pattern of PositionOverlay, HeadDirectionOverlay
- Static object_positions (not interpolated) vs dynamic animal_positions
- Optional firing_rates for modulating line appearance
- Uses existing validation helpers (_validate_finite_values, _validate_shape, _validate_bounds)
- WHAT/WHY/HOW error messages for all validation errors

**Tests:**

- 25/25 passing
- Module structure tests (imports, protocol compliance)
- Creation tests (basic, with times, with firing rates, custom styling)
- Conversion tests (basic, with firing rates, no times, preserves objects)
- Validation tests (shape, finite values, length match, bounds warning)
- Docstring tests

**Code Quality:**

- ruff: All checks pass
- mypy: No issues found
- Code review: Approved after adding finite value validation for firing_rates
- NumPy-style docstrings with complete parameter documentation
- Full type annotations with Literal types

### Completed: M2.6 Integration and Documentation

**Files Modified:**

- `src/neurospatial/simulation/__init__.py` - Added `ObjectVectorCellModel` export
- `.claude/QUICKSTART.md` - Added "Object-Vector Cells" section with:
  - `compute_object_vector_field()` usage example
  - Metrics and classification examples
  - Simulation example with `ObjectVectorCellModel`
  - Animation with `ObjectVectorOverlay` example
- `.claude/API_REFERENCE.md` - Added "Object-Vector Cells (v0.18.0+)" section with:
  - All exports from `neurospatial` top-level
  - `neurospatial.metrics` exports
  - `neurospatial.simulation` exports
  - `neurospatial.animation` exports

**Milestone 2 Complete!** All tasks in M2 (Object-Vector Cells) are now done.

**All Tests:** 137/137 passing for object-vector modules

### Completed: M3.1 Visibility/Gaze Computation

**Files Created:**

- `src/neurospatial/visibility.py` - Main implementation (~1050 LOC)
- `tests/test_visibility.py` - 45 comprehensive tests

**Implemented:**

- [x] `FieldOfView` frozen dataclass with:
  - `left_angle`, `right_angle`, `binocular_half_angle`, `blind_spot_behind`
  - Factory methods: `symmetric()`, `rat()`, `mouse()`, `primate()`
  - Properties: `total_angle`, `total_angle_degrees`
  - Methods: `contains_angle()`, `is_binocular()`
  - Validation in `__post_init__`
- [x] `ViewshedResult` frozen dataclass with:
  - Fields: `visible_bins`, `visible_cues`, `cue_distances`, `cue_bearings`, `occlusion_map`
  - Properties: `n_visible_bins`, `visibility_fraction`, `n_visible_cues`
  - Methods: `filter_cues()`, `visible_bin_centers()`
- [x] `compute_viewed_location()` - compute gaze-directed locations
  - Methods: "fixed_distance", "ray_cast", "boundary"
  - Support for `gaze_offsets` parameter
- [x] `compute_viewshed()` - ray-casting visibility analysis
  - Support for `fov` parameter (FieldOfView, float, or None)
  - Returns `ViewshedResult` with occlusion map
- [x] `compute_view_field()` - binary visibility mask
- [x] `visible_cues()` - check visibility of cue positions
- [x] `compute_viewshed_trajectory()` - viewshed along trajectory
- [x] `visibility_occupancy()` - time each bin was visible
- [x] `_ray_cast_to_boundary()` - iterative stepping with binary search refinement
- [x] `_line_of_sight_clear()` - line-of-sight check helper

**Key Design Decisions:**

- Species-specific FOV presets based on literature (rat ~320°, primate ~180°)
- Uses ray casting with binary search refinement for boundary detection
- FOV supports blind spots behind and binocular regions
- Removed `visible_boundary_segments` field (not implemented, deferred to future)
- Egocentric coordinate convention: 0=ahead, π/2=left, -π/2=right

**Tests:**

- 45/45 passing
- Module structure tests (imports, docstring, exports)
- FieldOfView tests (dataclass, factories, properties, methods)
- ViewshedResult tests (properties, methods)
- compute_viewed_location tests (methods, validation)
- compute_viewshed tests (basic, FOV restriction, cue visibility)
- visible_cues tests (visibility, occlusion)
- Trajectory tests (compute_viewshed_trajectory, visibility_occupancy)
- Line-of-sight helper tests

**Code Quality:**

- ruff: All checks pass
- mypy: No issues found
- Code review: Issues addressed
- NumPy docstrings with coordinate convention docs
- Complete type annotations

**Module Added to Exports:**

- `src/neurospatial/__init__.py` - Added all visibility exports:
  - `FieldOfView`, `ViewshedResult`
  - `compute_view_field`, `compute_viewed_location`, `compute_viewshed`
  - `compute_viewshed_trajectory`, `visibility_occupancy`, `visible_cues`

### Completed: M3.2 Spatial View Cell Model (Simulation)

**Files Created:**

- `src/neurospatial/simulation/models/spatial_view_cells.py` - Main implementation (~450 LOC)
- `tests/simulation/models/test_spatial_view_cells.py` - 30 comprehensive tests

**Implemented:**

- [x] `SpatialViewCellModel` class with parameters:
  - `env`, `preferred_view_location` - required spatial setup
  - `view_field_width` - Gaussian tuning width (default 10.0)
  - `view_distance` - view distance for fixed_distance model (default 20.0)
  - `gaze_model` - "fixed_distance", "ray_cast", "boundary"
  - `max_rate`, `baseline_rate` - firing rate parameters
  - `require_visibility` - optional line-of-sight check
  - `fov` - optional FieldOfView constraint
- [x] Parameter validation in `__init__` with clear error messages
- [x] Warning for preferred_view_location outside environment bounds
- [x] `firing_rate()` method with:
  - Computes viewed location via `compute_viewed_location()`
  - Gaussian distance tuning: `exp(-0.5 * ((d) / width)^2)`
  - Handles NaN viewed locations (returns baseline)
  - Optional FOV and visibility checks
- [x] `ground_truth` property returning all model parameters

**Key Design Decisions:**

- Class-based (not dataclass) following `ObjectVectorCellModel` pattern
- Uses `compute_viewed_location()` from visibility module for gaze computation
- Gaussian tuning around preferred viewed location (not animal position)
- FOV restriction applies to both visibility check and general viewing
- NaN handling: returns baseline rate when viewing outside environment

**Tests:**

- 30/30 passing
- Module structure tests (imports, docstring)
- Parameter validation tests (8 tests for error messages)
- Firing rate computation tests (shape, non-negative, peaks, Gaussian)
- Gaze model tests (fixed_distance, ray_cast)
- Visibility requirement tests
- FOV integration tests
- Ground truth tests (completeness, immutability)
- Protocol compliance (implements NeuralModel)
- Edge cases (NaN viewing, single position)

**Code Quality:**

- ruff: All checks pass
- mypy: No issues found
- Code review: Approved - high quality scientific code
- NumPy docstrings with full examples and scientific references
- Complete type annotations with Literal types

**Module Updated:**

- `src/neurospatial/simulation/models/__init__.py` - Added `SpatialViewCellModel` export

### Completed: M3.3 Spatial View Field Analysis

**Files Created:**

- `src/neurospatial/spatial_view_field.py` - Main implementation (~400 LOC)
- `tests/test_spatial_view_field.py` - 30 comprehensive tests

**Implemented:**

- [x] `SpatialViewFieldResult` frozen dataclass with fields:
  - `field` - firing rate (Hz) in each spatial bin, binned by *viewed location*
  - `env` - the spatial environment
  - `view_occupancy` - time spent viewing each bin (not position occupancy)
- [x] `compute_spatial_view_field()` function:
  - Computes viewed location at each timepoint via `compute_viewed_location()`
  - Filters NaN viewed locations (viewing outside environment)
  - Computes view occupancy (time viewing each bin)
  - Interpolates spike times to frames, gets viewed bin at spike time
  - Bins spikes by viewed location (not animal position!)
  - Normalizes by view occupancy
  - Applies `min_occupancy_seconds` threshold
  - Supports methods: "binned", "diffusion_kde", "gaussian_kde"
  - Supports gaze models: "fixed_distance", "ray_cast", "boundary"
  - Supports `gaze_offsets` for eye tracking integration

**Key Design Decisions:**

- Uses `compute_viewed_location()` from visibility module for gaze computation
- View occupancy is fundamentally different from position occupancy
- Spikes are binned by where the animal was *looking*, not where it was
- diffusion_kde smooths both spike counts and view occupancy before normalization
- gaussian_kde uses Euclidean distance from viewed locations to bin centers
- gaze_offsets parameter enables integration with eye tracking data

**Tests:**

- 30/30 passing
- Module structure tests (imports, docstring, exports)
- Dataclass tests (fields, frozen, immutability)
- Field computation tests (shape, view occupancy, NaN handling)
- Smoothing method tests (binned, diffusion_kde, gaussian_kde)
- Gaze model tests (fixed_distance, ray_cast)
- Validation tests (empty spikes, mismatched lengths, invalid method/gaze_model)
- gaze_offsets parameter test
- Ground truth recovery tests:
  - Spatial view cell: view field peaks at preferred view location
  - Spatial view cell: view field differs from place field (key scientific test!)

**Code Quality:**

- ruff: All checks pass
- mypy: No issues found
- Code review: Approved after fixes:
  - Fixed doctest skip directive
  - Added gaze_offsets validation
  - Added gaze_offsets test
- NumPy docstrings with coordinate conventions and scientific references
- Complete type annotations with Literal types
- Excellent module docstring with "Which Function Should I Use?" guidance

**Module Updated:**

- `src/neurospatial/__init__.py` - Added exports:
  - `SpatialViewFieldResult`, `compute_spatial_view_field`

### Completed: M3.4 Spatial View Metrics

**Files Created:**

- `src/neurospatial/metrics/spatial_view_cells.py` - Main implementation (~520 LOC)
- `tests/metrics/test_spatial_view_cells.py` - 23 comprehensive tests

**Implemented:**

- [x] `SpatialViewMetrics` frozen dataclass with fields:
  - `view_field_skaggs_info` - Skaggs spatial information for view field
  - `place_field_skaggs_info` - Skaggs spatial information for place field
  - `view_place_correlation` - Pearson correlation between view and place fields
  - `view_field_sparsity` - sparsity of view field
  - `view_field_coherence` - spatial coherence of view field
  - `is_spatial_view_cell` - classification result
  - `interpretation()` method - human-readable string with classification reasoning
- [x] `spatial_view_cell_metrics()` function:
  - Computes both place field and view field from same spike data
  - Uses existing `compute_place_field()` and `compute_spatial_view_field()`
  - Computes Skaggs information for both fields using `skaggs_information()`
  - Computes sparsity and coherence using existing metrics functions
  - Computes correlation between view and place fields (handles NaN)
  - Classification based on info ratio and correlation thresholds
- [x] `is_spatial_view_cell()` classifier:
  - Quick boolean function for screening many neurons
  - Configurable `info_ratio` (default 1.5) and `max_correlation` (default 0.7)
  - Returns False on ValueError (graceful error handling)

**Key Design Decisions:**

- API pattern mirrors `head_direction.py` (dataclass + compute + classifier)
- Uses frozen dataclass for immutable metrics
- Reuses existing Skaggs information, sparsity, and coherence functions from `place_fields.py`
- Classification criteria: view_info > ratio * place_info AND correlation < max_corr
- Default thresholds based on principle that SVCs have view-selective, not position-selective, firing
- Returns Python bool (not numpy bool) for type safety

**Tests:**

- 23/23 passing
- Module structure tests (imports, docstring, exports)
- Dataclass tests (creation, frozen, interpretation for SVC and non-SVC)
- Computation tests (all 5 metrics computed, correct ranges)
- Validation tests (empty spikes, mismatched lengths)
- Parameter tests (view_distance, info_ratio, max_correlation)
- Ground truth recovery tests:
  - Simulated SVC has higher view field info than place field info
  - Place cell not classified as SVC (using PlaceCellModel from simulation)

**Code Quality:**

- ruff: All checks pass
- mypy: No issues found
- Code review: Approved - exemplary implementation:
  - Consistent with project patterns (mirrors head_direction.py)
  - Outstanding documentation with scientific references
  - Robust error handling (NaN, zero variance, edge cases)
  - Comprehensive test coverage with edge cases
- NumPy docstrings with "Which Function Should I Use?" guidance
- Complete type annotations

**Module Updated:**

- `src/neurospatial/metrics/__init__.py` - Added exports:
  - `SpatialViewMetrics`, `spatial_view_cell_metrics`, `is_spatial_view_cell`
- `src/neurospatial/simulation/__init__.py` - Added `SpatialViewCellModel` export

### Next Task

- **M3.5**: Tests for Spatial View Cells
  - Create/update visibility tests
  - Create/update spatial view cell model tests
  - Create/update spatial view field tests

### Blockers

None currently.

---
