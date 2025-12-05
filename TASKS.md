# Implementation Tasks: Object-Vector Cells, Egocentric Frames, and Spatial View Cells

**Based on**: [PLAN.md](PLAN.md)
**Created**: 2025-12-05
**Status**: Ready for implementation

---

## Overview

This task list breaks down the implementation plan into actionable tasks organized by milestone. Each milestone builds on previous work.

**Dependency chain**: M1 (Foundation) → M2 (Object-Vector Cells) → M3 (Spatial View Cells)

---

## Milestone 1: Egocentric Reference Frames (Foundation)

**Goal**: Enable transformations between allocentric (world-centered) and egocentric (animal-centered) coordinate systems.

**Priority**: HIGH - Foundation for M2 and M3

### M1.1: Core Reference Frame Module

**File**: `src/neurospatial/reference_frames.py`

- [x] Create `reference_frames.py` module with module-level docstring
- [x] Implement `EgocentricFrame` dataclass
  - [x] `position` and `heading` attributes
  - [x] `to_egocentric()` method (2x2 rotation matrix)
  - [x] `to_allocentric()` method (inverse transform)
  - [x] Docstring with coordinate convention (0=East, π/2=North, egocentric 0=ahead)
- [x] Implement `allocentric_to_egocentric()` batch function
  - [x] Handle shape normalization: `(n_points, 2)` → `(n_time, n_points, 2)`
  - [x] Use `np.einsum('tij,tpj->tpi', rot, centered)` for vectorized rotation
  - [x] Document broadcasting behavior
- [x] Implement `egocentric_to_allocentric()` inverse batch function
- [x] Implement `compute_egocentric_bearing()` function
  - [x] Use `allocentric_to_egocentric()` internally
  - [x] Extract bearing via `np.arctan2(ego[..., 1], ego[..., 0])`
  - [x] Wrap angles to `(-π, π]` using `(θ + π) % 2π - π`
- [x] Implement distance functions:
  - [x] `compute_egocentric_distance_euclidean()` - pure geometry
  - [x] `compute_egocentric_distance_geodesic()` - uses `distance_field()`
  - [x] `compute_egocentric_distance()` - dispatcher function
- [x] Define `__all__` exports

**Success criteria**:

- Round-trip test: `allocentric → egocentric → allocentric` preserves coordinates
- Heading=0: egocentric x-axis aligns with allocentric x-axis
- Heading=π/2: egocentric x-axis aligns with allocentric y-axis

### M1.2: Heading Computation Utilities

**File**: `src/neurospatial/reference_frames.py` (continued)

- [x] Implement `heading_from_velocity()` function
  - [x] Compute velocity via finite differences
  - [x] Gaussian smoothing with configurable window
  - [x] Compute `heading = atan2(vy, vx)`
  - [x] Mask low-speed periods (`speed < min_speed`)
  - [x] Circular interpolation via unit vectors for masked periods
  - [x] Warning when all speeds below threshold
  - [x] Handle edge case: `n_time < 2` → raise ValueError
- [x] Implement `heading_from_body_orientation()` function
  - [x] Compute body vector: `nose - tail`
  - [x] Extract heading via `atan2`
  - [x] Handle NaN keypoints with circular interpolation
  - [x] Raise ValueError when all keypoints NaN

**Success criteria**:

- Stationary periods produce smooth interpolated headings (no jumps)
- NaN keypoints are filled without discontinuities at ±π boundary

### M1.3: Egocentric Polar Environment

**File**: `src/neurospatial/environment/factories.py`

- [x] Add `from_polar_egocentric()` classmethod to Environment
  - [x] Parameters: `distance_range`, `angle_range`, `distance_bin_size`, `angle_bin_size`
  - [x] Parameter: `circular_angle=True` for periodic connectivity
  - [x] Create polar environment using existing MaskedGrid machinery
  - [x] Document: "This environment lives in egocentric polar coordinates"
  - [x] Return Environment with bin_centers[:, 0] = distances, [:, 1] = angles

**Success criteria**:

- `ego_env.n_bins` matches expected `n_distance * n_angle`
- Circular connectivity wraps first↔last angle bins when `circular_angle=True`

### M1.4: Tests for Reference Frames

**File**: `tests/test_reference_frames.py`

- [x] `TestModuleSetup` class
  - [x] Test imports work
  - [x] Test `__all__` contains expected exports
  - [x] Test module docstring exists
- [x] `TestEgocentricFrame` class
  - [x] Test dataclass creation
  - [x] Test `to_egocentric()` for heading=0 (identity rotation)
  - [x] Test `to_egocentric()` for heading=π/2 (90° rotation)
  - [x] Test `to_allocentric()` inverse
  - [x] Test round-trip preserves coordinates
- [x] `TestAllocentricToEgocentric` class
  - [x] Test batch transform with multiple timepoints
  - [x] Test broadcasting: 2D points → 3D output
  - [x] Test shape validation error messages
- [x] `TestComputeEgocentricBearing` class
  - [x] Test bearing=0 when target is ahead
  - [x] Test bearing=π/2 when target is left
  - [x] Test bearing=-π/2 when target is right
  - [x] Test angle wrapping near ±π
- [x] `TestComputeEgocentricDistance` class
  - [x] Test Euclidean distances
  - [x] Test geodesic distances with Environment
  - [x] Test dispatcher with invalid metric
- [x] `TestHeadingFromVelocity` class
  - [x] Test smooth trajectory
  - [x] Test stationary periods get interpolated
  - [x] Test warning and NaN return when all speeds low
  - [x] Test `n_time < 2` raises ValueError
- [x] `TestHeadingFromBodyOrientation` class
  - [x] Test with valid keypoints
  - [x] Test NaN interpolation
  - [x] Test all-NaN raises ValueError

### M1.5: Integration and Documentation

- [x] Add exports to `src/neurospatial/__init__.py`
  - [x] Export: `EgocentricFrame`, `allocentric_to_egocentric`, `egocentric_to_allocentric`
  - [x] Export: `compute_egocentric_bearing`, `compute_egocentric_distance`
  - [x] Export: `heading_from_velocity`, `heading_from_body_orientation`
- [x] Update `.claude/QUICKSTART.md` with egocentric transform examples
- [x] Update `.claude/API_REFERENCE.md` with `reference_frames` imports

---

## Milestone 2: Object-Vector Cells

**Goal**: Implement object-vector cell models, metrics, and visualization.

**Priority**: HIGH

**Dependencies**: M1 complete

### M2.1: Object-Vector Cell Model (Simulation)

**File**: `src/neurospatial/simulation/models/object_vector_cells.py`

- [x] Create module with docstring and references (Hoydal et al. 2019)
- [x] Implement `ObjectVectorCellModel` dataclass
  - [x] Required: `env`, `object_positions`
  - [x] Tuning parameters: `preferred_distance`, `distance_width`
  - [x] Direction parameters: `preferred_direction` (optional), `direction_kappa`
  - [x] Selectivity: `object_selectivity` ("any", "nearest", "specific")
  - [x] Rate parameters: `max_rate`, `baseline_rate`
  - [x] Distance: `distance_metric` ("euclidean", "geodesic")
- [x] Implement `__post_init__` validation
  - [x] Validate positive rates, widths
  - [x] Ensure object_positions is 2D
  - [x] Warn if objects outside environment bounds
  - [x] Precompute geodesic distance fields if metric="geodesic"
- [x] Implement `firing_rate()` method
  - [x] Compute distances to all objects (Euclidean or geodesic)
  - [x] Distance tuning: Gaussian `exp(-0.5 * ((d - pref) / width)^2)`
  - [x] Direction tuning: von Mises `exp(κ * cos(θ - pref_θ)) / exp(κ)`
  - [x] Vectorized bearing computation via `compute_egocentric_bearing()`
  - [x] Aggregate by selectivity mode (any/nearest/specific)
  - [x] Return: `baseline + (max - baseline) * response`
- [x] Implement `ground_truth` property

**Success criteria**:

- Firing rate peaks at `(preferred_distance, preferred_direction)` from object
- `direction_kappa=4` produces ~30° half-width tuning

### M2.2: Object-Vector Metrics

**File**: `src/neurospatial/metrics/object_vector_cells.py`

- [x] Create module with docstring and references
- [x] Implement `ObjectVectorMetrics` frozen dataclass
  - [x] Fields: `preferred_distance`, `preferred_direction`
  - [x] Fields: `distance_selectivity`, `direction_selectivity`, `object_vector_score`
  - [x] Fields: `peak_rate`, `mean_rate`
  - [x] Fields: `tuning_curve`, `distance_bins`, `direction_bins`
  - [x] Method: `interpretation()` → human-readable string
- [x] Implement `compute_object_vector_tuning()` function
  - [x] Bin spikes by egocentric (distance, direction) to nearest object
  - [x] Compute occupancy in each bin
  - [x] Normalize by occupancy
  - [x] Apply `min_occupancy_seconds` threshold (default 0.1s)
  - [x] Return `ObjectVectorMetrics`
- [x] Implement `object_vector_score()` function
  - [x] Distance selectivity: `s_d = peak / mean`
  - [x] Direction selectivity: reuse `_mean_resultant_length` from `metrics.circular`
  - [x] Combined: `s_OV = ((s_d - 1) / (s_d* - 1)) * s_θ`
  - [x] Document `max_distance_selectivity` parameter (default 10.0)
  - [x] Validate `max_distance_selectivity > 1`
- [x] Implement `is_object_vector_cell()` classifier
  - [x] Check `score >= threshold` and `peak_rate >= min_rate`
- [x] Implement `plot_object_vector_tuning()` visualization
  - [x] Polar heatmap: distance on radial axis, angle on angular axis
  - [x] Optional: mark peak location

### M2.3: Object-Vector Field Computation

**File**: `src/neurospatial/object_vector_field.py`

- [x] Create module with docstring
- [x] Implement `ObjectVectorFieldResult` frozen dataclass
  - [x] Fields: `field`, `ego_env`, `occupancy`
- [x] Implement `compute_object_vector_field()` function
  - [x] Create egocentric polar environment via `from_polar_egocentric()`
  - [x] Compute distance and bearing to nearest object at each timepoint
  - [x] Support geodesic distance when `allocentric_env` provided
  - [x] Compute occupancy in egocentric polar space (time per bin)
  - [x] Bin spikes by egocentric position at spike time (interpolate)
  - [x] Normalize by occupancy, apply `min_occupancy_seconds` threshold
  - [x] Support methods: "binned", "diffusion_kde"
  - [x] Return `ObjectVectorFieldResult`

**Success criteria**:

- [x] Field peaks at `(preferred_distance, preferred_direction)` for simulated cells

### M2.4: Object-Vector Overlay (Animation)

**File**: `src/neurospatial/animation/overlays.py` (added to existing file)

- [x] Implement `ObjectVectorOverlay` dataclass
  - [x] Required: `object_positions`, `animal_positions`
  - [x] Optional: `firing_rates`, `times`
  - [x] Styling: `color`, `linewidth`, `show_objects`, `object_marker`, `object_size`
  - [x] Interpolation: `interp` ("linear", "nearest")
- [x] Implement `ObjectVectorData` internal data container
- [x] Implement `convert_to_data()` method for frame alignment
- [x] Register overlay in `animation/__init__.py` exports
- [x] Update OverlayProtocol return type
- [x] Update OverlayData container
- [x] Update `_convert_overlays_to_data()` dispatch

### M2.5: Tests for Object-Vector Cells

**File**: `tests/simulation/models/test_object_vector_cells.py`

- [x] Test model parameter validation
- [x] Test firing rate with distance tuning only (no direction)
- [x] Test firing rate with direction tuning
- [x] Test object selectivity modes: any, nearest, specific
- [x] Test geodesic vs Euclidean distances
- [x] Test warning for objects outside environment
- [x] Test `ground_truth` property matches parameters

**File**: `tests/metrics/test_object_vector_cells.py`

- [x] Test `compute_object_vector_tuning()` binning
- [x] Test `object_vector_score()` formulas
- [x] Test `is_object_vector_cell()` classification
- [x] Test recovery of ground truth from simulated cell
- [x] Test `min_occupancy_seconds` filtering

**File**: `tests/test_object_vector_field.py`

- [x] Test field computation with binned method
- [x] Test field computation with diffusion_kde method
- [x] Test ego_env returned correctly
- [x] Test geodesic distance mode

### M2.6: Integration and Documentation

- [x] Add exports to `src/neurospatial/__init__.py`
  - [x] Export: `compute_object_vector_field`
- [x] Add exports to `src/neurospatial/simulation/__init__.py`
  - [x] Export: `ObjectVectorCellModel`
- [x] Add exports to `src/neurospatial/metrics/__init__.py`
  - [x] Export: `ObjectVectorMetrics`, `compute_object_vector_tuning`
  - [x] Export: `object_vector_score`, `is_object_vector_cell`
  - [x] Export: `plot_object_vector_tuning`
- [x] Add exports to `src/neurospatial/animation/__init__.py`
  - [x] Export: `ObjectVectorOverlay`, `ObjectVectorData`
- [x] Update `.claude/QUICKSTART.md` with object-vector examples
- [x] Update `.claude/API_REFERENCE.md` with object-vector imports

---

## Milestone 3: Spatial View Cells

**Goal**: Implement spatial view cell models, metrics, and visibility analysis.

**Priority**: MEDIUM

**Dependencies**: M1 complete, M2 patterns helpful

### M3.1: Visibility/Gaze Computation

**File**: `src/neurospatial/visibility.py`

- [x] Create module with docstring
- [x] Implement `compute_viewed_location()` function
  - [x] Method: "fixed_distance" - point at fixed distance in gaze direction
  - [x] Method: "ray_cast" - intersection with environment boundary
  - [x] Method: "boundary" - nearest boundary point in gaze direction
  - [x] Support optional `gaze_offsets` relative to heading
  - [x] Mark as NaN when viewing outside environment
- [x] Implement `_ray_cast_to_boundary()` helper
  - [x] Iterative stepping with binary search refinement
- [x] Implement `FieldOfView` frozen dataclass
  - [x] Attributes: `left_angle`, `right_angle`, `binocular_half_angle`, `blind_spot_behind`
  - [x] Class methods: `symmetric()`, `rat()`, `mouse()`, `primate()`
  - [x] Properties: `total_angle`, `total_angle_degrees`
  - [x] Method: `contains_angle()` - check if bearing in FOV
  - [x] Method: `is_binocular()` - check if in binocular region
  - [x] Validation in `__post_init__`
- [x] Implement `ViewshedResult` frozen dataclass
  - [x] Fields: `visible_bins`, `visible_cues`, `cue_distances`, `cue_bearings`
  - [x] Field: `occlusion_map` (per-bin score [0,1]: fraction of rays unobstructed)
  - [x] Properties: `n_visible_bins`, `visibility_fraction`, `n_visible_cues`
  - [x] Method: `filter_cues(cue_ids)` → visible IDs, distances, bearings
  - [x] Method: `visible_bin_centers(env)` → allocentric positions of visible bins
- [x] Implement `compute_viewshed()` function
  - [x] Cast n_rays from observer position
  - [x] Find intersection with boundary for each ray
  - [x] Mark bins between observer and intersection as visible
  - [x] Support `fov` parameter (FieldOfView, float, or None for 360°)
  - [x] Check cue visibility
  - [x] Return `ViewshedResult`
- [x] Implement `compute_view_field()` function
  - [x] Binary mask of visible bins from position/heading
  - [x] Uses `compute_viewshed()` internally
- [x] Implement `visible_cues()` function
  - [x] Check line-of-sight for each cue
  - [x] Return: visible mask, distances, bearings
- [x] Implement `_line_of_sight_clear()` helper
- [x] Implement `compute_viewshed_trajectory()` function
  - [x] Computation along trajectory
- [x] Implement `visibility_occupancy()` function
  - [x] Time each bin was visible during trajectory

**Success criteria**:

- [x] `FieldOfView.rat().total_angle_degrees` ≈ 320° (within 290-340)
- [x] `FieldOfView.primate().total_angle_degrees` = 180°
- [x] Blind spot behind correctly excludes rear regions

### M3.2: Spatial View Cell Model (Simulation)

**File**: `src/neurospatial/simulation/models/spatial_view_cells.py`

- [x] Create module with docstring and references (Rolls et al. 1997)
- [x] Implement `SpatialViewCellModel` dataclass
  - [x] Required: `env`, `preferred_view_location`
  - [x] Tuning: `view_field_width`
  - [x] Gaze: `view_distance`, `gaze_model`
  - [x] Rate: `max_rate`, `baseline_rate`
  - [x] Visibility: `require_visibility`, `fov`
- [x] Implement `firing_rate()` method
  - [x] Compute viewed location at each timepoint
  - [x] Distance from viewed location to preferred
  - [x] Gaussian tuning
  - [x] Handle NaN (viewing outside environment)
  - [x] Optional visibility check
- [x] Implement `ground_truth` property

**Success criteria**:

- [x] Firing rate peaks when viewing `preferred_view_location`

### M3.3: Spatial View Field Analysis

**File**: `src/neurospatial/spatial_view_field.py`

- [x] Create module with docstring
- [x] Implement `compute_spatial_view_field()` function
  - [x] Compute viewed location at each timepoint
  - [x] Filter NaN (viewing outside environment)
  - [x] Compute view occupancy
  - [x] Interpolate viewed location at spike times
  - [x] Bin spikes by viewed location
  - [x] Normalize by view occupancy
  - [x] Apply `min_occupancy_seconds` threshold
  - [x] Optional smoothing (diffusion_kde, gaussian_kde)
  - [x] Return field with same shape as place field

**Success criteria**:

- [x] View field differs from place field for spatial view cells
- [x] View field similar to place field for place cells

### M3.4: Spatial View Metrics

**File**: `src/neurospatial/metrics/spatial_view_cells.py`

- [x] Create module with docstring
- [x] Implement `SpatialViewMetrics` frozen dataclass
  - [x] Fields: `view_field_skaggs_info`, `place_field_skaggs_info`
  - [x] Fields: `view_place_correlation`
  - [x] Fields: `view_field_sparsity`, `view_field_coherence`
  - [x] Field: `is_spatial_view_cell`
  - [x] Method: `interpretation()`
- [x] Implement `spatial_view_cell_metrics()` function
  - [x] Compute place field and view field
  - [x] Compute Skaggs info for both (reuse from `place_fields.py`)
  - [x] Compute sparsity, coherence (reuse existing)
  - [x] Compute view-place correlation (Z-score, mask NaN)
  - [x] Return `SpatialViewMetrics`
- [x] Implement `is_spatial_view_cell()` classifier
  - [x] `view_info > ratio * place_info` AND `correlation < max_corr`

### M3.5: Tests for Spatial View Cells

**File**: `tests/test_visibility.py`

- [ ] Test `compute_viewed_location()` with fixed_distance method
- [ ] Test `compute_viewed_location()` with ray_cast method
- [ ] Test `FieldOfView` dataclass and presets
- [ ] Test `FieldOfView.contains_angle()` with various bearings
- [ ] Test `compute_viewshed()` returns correct visible bins
- [ ] Test `visible_cues()` occlusion detection
- [ ] Test `visibility_occupancy()` integration

**File**: `tests/simulation/models/test_spatial_view_cells.py`

- [ ] Test model creation and validation
- [ ] Test firing rate computation
- [ ] Test with different gaze models
- [ ] Test `require_visibility` flag
- [ ] Test `ground_truth` property

**File**: `tests/metrics/test_spatial_view_cells.py`

- [ ] Test `spatial_view_cell_metrics()` computation
- [ ] Test classification: spatial view cell vs place cell
- [ ] Test recovery of ground truth from simulation

**File**: `tests/test_spatial_view_field.py`

- [ ] Test field computation
- [ ] Test NaN handling
- [ ] Test occupancy normalization
- [ ] Test smoothing methods

### M3.6: Integration and Documentation

- [ ] Add exports to `src/neurospatial/__init__.py`
  - [ ] Export: `compute_spatial_view_field`
- [ ] Add exports to `src/neurospatial/visibility.py` `__all__`
  - [ ] Export: `compute_viewed_location`, `compute_viewshed`, `compute_view_field`
  - [ ] Export: `visible_boundaries`, `visible_cues`, `visibility_occupancy`
  - [ ] Export: `FieldOfView`, `ViewshedResult`
- [ ] Add exports to `src/neurospatial/simulation/__init__.py`
  - [ ] Export: `SpatialViewCellModel`
- [ ] Add exports to `src/neurospatial/metrics/__init__.py`
  - [ ] Export: `SpatialViewMetrics`, `spatial_view_cell_metrics`
  - [ ] Export: `is_spatial_view_cell`
- [ ] Update `.claude/QUICKSTART.md` with spatial view examples
- [ ] Update `.claude/API_REFERENCE.md` with spatial view imports
- [ ] Update `.claude/ADVANCED.md` with gaze analysis section
- [ ] Create example: `examples/spatial_view_cells.py`

---

## Final Integration

### Documentation Updates

- [ ] Update CLAUDE.md with new module summaries
- [ ] Add examples to notebooks directory (optional)
- [ ] Update CHANGELOG with new features

### Quality Checks

- [ ] Run full test suite: `uv run pytest`
- [ ] Run type checker: `uv run mypy src/neurospatial/`
- [ ] Run linter: `uv run ruff check .`
- [ ] Run formatter: `uv run ruff format .`
- [ ] Verify all docstrings follow NumPy format
- [ ] Verify all new functions have examples in docstrings

---

## Summary Statistics

| Milestone | New Files | Estimated LOC | Dependencies |
|-----------|-----------|---------------|--------------|
| M1 | 2 | ~1,000 | None |
| M2 | 6 | ~2,250 | M1 |
| M3 | 7 | ~2,700 | M1, (M2 patterns) |
| **Total** | **15** | **~5,950** | |

---

## Quick Reference: Implementation Order

1. **M1.1** → Core reference frames (foundation)
2. **M1.2** → Heading utilities
3. **M1.3** → Egocentric grid factory
4. **M1.4** → Tests for M1
5. **M1.5** → M1 integration/docs
6. **M2.1** → Object-vector model
7. **M2.2** → Object-vector metrics
8. **M2.3** → Object-vector field
9. **M2.4** → Object-vector overlay
10. **M2.5** → Tests for M2
11. **M2.6** → M2 integration/docs
12. **M3.1** → Visibility computation
13. **M3.2** → Spatial view model
14. **M3.3** → Spatial view field
15. **M3.4** → Spatial view metrics
16. **M3.5** → Tests for M3
17. **M3.6** → M3 integration/docs
18. **Final** → Documentation and quality checks
