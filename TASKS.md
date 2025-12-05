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

- [ ] Create `reference_frames.py` module with module-level docstring
- [ ] Implement `EgocentricFrame` dataclass
  - [ ] `position` and `heading` attributes
  - [ ] `to_egocentric()` method (2x2 rotation matrix)
  - [ ] `to_allocentric()` method (inverse transform)
  - [ ] Docstring with coordinate convention (0=East, π/2=North, egocentric 0=ahead)
- [ ] Implement `allocentric_to_egocentric()` batch function
  - [ ] Handle shape normalization: `(n_points, 2)` → `(n_time, n_points, 2)`
  - [ ] Use `np.einsum('tij,tpj->tpi', rot, centered)` for vectorized rotation
  - [ ] Document broadcasting behavior
- [ ] Implement `egocentric_to_allocentric()` inverse batch function
- [ ] Implement `compute_egocentric_bearing()` function
  - [ ] Use `allocentric_to_egocentric()` internally
  - [ ] Extract bearing via `np.arctan2(ego[..., 1], ego[..., 0])`
  - [ ] Wrap angles to `(-π, π]` using `(θ + π) % 2π - π`
- [ ] Implement distance functions:
  - [ ] `compute_egocentric_distance_euclidean()` - pure geometry
  - [ ] `compute_egocentric_distance_geodesic()` - uses `distance_field()`
  - [ ] `compute_egocentric_distance()` - dispatcher function
- [ ] Define `__all__` exports

**Success criteria**:
- Round-trip test: `allocentric → egocentric → allocentric` preserves coordinates
- Heading=0: egocentric x-axis aligns with allocentric x-axis
- Heading=π/2: egocentric x-axis aligns with allocentric y-axis

### M1.2: Heading Computation Utilities

**File**: `src/neurospatial/reference_frames.py` (continued)

- [ ] Implement `heading_from_velocity()` function
  - [ ] Compute velocity via finite differences
  - [ ] Gaussian smoothing with configurable window
  - [ ] Compute `heading = atan2(vy, vx)`
  - [ ] Mask low-speed periods (`speed < min_speed`)
  - [ ] Circular interpolation via unit vectors for masked periods
  - [ ] Warning when all speeds below threshold
  - [ ] Handle edge case: `n_time < 2` → raise ValueError
- [ ] Implement `heading_from_body_orientation()` function
  - [ ] Compute body vector: `nose - tail`
  - [ ] Extract heading via `atan2`
  - [ ] Handle NaN keypoints with circular interpolation
  - [ ] Raise ValueError when all keypoints NaN

**Success criteria**:
- Stationary periods produce smooth interpolated headings (no jumps)
- NaN keypoints are filled without discontinuities at ±π boundary

### M1.3: Egocentric Environment Grid

**File**: `src/neurospatial/environment/factories.py`

- [ ] Add `from_egocentric_grid()` classmethod to Environment
  - [ ] Parameters: `distance_range`, `angle_range`, `distance_bin_size`, `angle_bin_size`
  - [ ] Parameter: `circular_angle=True` for periodic connectivity
  - [ ] Create polar grid using existing RegularGrid machinery
  - [ ] Document: "This environment lives in egocentric polar coordinates"
  - [ ] Return Environment with bin_centers[:, 0] = distances, [:, 1] = angles

**Success criteria**:
- `ego_env.n_bins` matches expected `n_distance * n_angle`
- Circular connectivity wraps first↔last angle bins when `circular_angle=True`

### M1.4: Tests for Reference Frames

**File**: `tests/test_reference_frames.py`

- [ ] `TestModuleSetup` class
  - [ ] Test imports work
  - [ ] Test `__all__` contains expected exports
  - [ ] Test module docstring exists
- [ ] `TestEgocentricFrame` class
  - [ ] Test dataclass creation
  - [ ] Test `to_egocentric()` for heading=0 (identity rotation)
  - [ ] Test `to_egocentric()` for heading=π/2 (90° rotation)
  - [ ] Test `to_allocentric()` inverse
  - [ ] Test round-trip preserves coordinates
- [ ] `TestAllocentricToEgocentric` class
  - [ ] Test batch transform with multiple timepoints
  - [ ] Test broadcasting: 2D points → 3D output
  - [ ] Test shape validation error messages
- [ ] `TestComputeEgocentricBearing` class
  - [ ] Test bearing=0 when target is ahead
  - [ ] Test bearing=π/2 when target is left
  - [ ] Test bearing=-π/2 when target is right
  - [ ] Test angle wrapping near ±π
- [ ] `TestComputeEgocentricDistance` class
  - [ ] Test Euclidean distances
  - [ ] Test geodesic distances with Environment
  - [ ] Test dispatcher with invalid metric
- [ ] `TestHeadingFromVelocity` class
  - [ ] Test smooth trajectory
  - [ ] Test stationary periods get interpolated
  - [ ] Test warning when all speeds low
  - [ ] Test `n_time < 2` raises ValueError
- [ ] `TestHeadingFromBodyOrientation` class
  - [ ] Test with valid keypoints
  - [ ] Test NaN interpolation
  - [ ] Test all-NaN raises ValueError

### M1.5: Integration and Documentation

- [ ] Add exports to `src/neurospatial/__init__.py`
  - [ ] Export: `EgocentricFrame`, `allocentric_to_egocentric`, `egocentric_to_allocentric`
  - [ ] Export: `compute_egocentric_bearing`, `compute_egocentric_distance`
  - [ ] Export: `heading_from_velocity`, `heading_from_body_orientation`
- [ ] Update `.claude/QUICKSTART.md` with egocentric transform examples
- [ ] Update `.claude/API_REFERENCE.md` with `reference_frames` imports

---

## Milestone 2: Object-Vector Cells

**Goal**: Implement object-vector cell models, metrics, and visualization.

**Priority**: HIGH

**Dependencies**: M1 complete

### M2.1: Object-Vector Cell Model (Simulation)

**File**: `src/neurospatial/simulation/models/object_vector_cells.py`

- [ ] Create module with docstring and references (Hoydal et al. 2019)
- [ ] Implement `ObjectVectorCellModel` dataclass
  - [ ] Required: `env`, `object_positions`
  - [ ] Tuning parameters: `preferred_distance`, `distance_width`
  - [ ] Direction parameters: `preferred_direction` (optional), `direction_kappa`
  - [ ] Selectivity: `object_selectivity` ("any", "nearest", "specific")
  - [ ] Rate parameters: `max_rate`, `baseline_rate`
  - [ ] Distance: `distance_metric` ("euclidean", "geodesic")
- [ ] Implement `__post_init__` validation
  - [ ] Validate positive rates, widths
  - [ ] Ensure object_positions is 2D
  - [ ] Warn if objects outside environment bounds
  - [ ] Precompute geodesic distance fields if metric="geodesic"
- [ ] Implement `firing_rate()` method
  - [ ] Compute distances to all objects (Euclidean or geodesic)
  - [ ] Distance tuning: Gaussian `exp(-0.5 * ((d - pref) / width)^2)`
  - [ ] Direction tuning: von Mises `exp(κ * cos(θ - pref_θ)) / exp(κ)`
  - [ ] Vectorized bearing computation via `allocentric_to_egocentric()`
  - [ ] Aggregate by selectivity mode (any/nearest/specific)
  - [ ] Return: `baseline + (max - baseline) * response`
- [ ] Implement `ground_truth` property

**Success criteria**:
- Firing rate peaks at `(preferred_distance, preferred_direction)` from object
- `direction_kappa=4` produces ~30° half-width tuning

### M2.2: Object-Vector Metrics

**File**: `src/neurospatial/metrics/object_vector_cells.py`

- [ ] Create module with docstring and references
- [ ] Implement `ObjectVectorMetrics` frozen dataclass
  - [ ] Fields: `preferred_distance`, `preferred_direction`
  - [ ] Fields: `distance_selectivity`, `direction_selectivity`, `object_vector_score`
  - [ ] Fields: `peak_rate`, `mean_rate`
  - [ ] Fields: `tuning_curve`, `distance_bins`, `direction_bins`
  - [ ] Method: `interpretation()` → human-readable string
- [ ] Implement `compute_object_vector_tuning()` function
  - [ ] Bin spikes by egocentric (distance, direction) to nearest object
  - [ ] Compute occupancy in each bin
  - [ ] Normalize by occupancy
  - [ ] Apply `min_occupancy_seconds` threshold (default 0.1s)
  - [ ] Return `ObjectVectorMetrics`
- [ ] Implement `object_vector_score()` function
  - [ ] Distance selectivity: `s_d = peak / mean`
  - [ ] Direction selectivity: reuse `_mean_resultant_length` from `circular.py`
  - [ ] Combined: `s_OV = ((s_d - 1) / (s_d* - 1)) * s_θ`
  - [ ] Document `max_distance_selectivity` parameter (default 10.0)
  - [ ] Validate `max_distance_selectivity > 1`
- [ ] Implement `is_object_vector_cell()` classifier
  - [ ] Check `score >= threshold` and `peak_rate >= min_rate`
- [ ] Implement `plot_object_vector_tuning()` visualization
  - [ ] Polar heatmap: distance on radial axis, angle on angular axis
  - [ ] Optional: mark peak location

### M2.3: Object-Vector Field Computation

**File**: `src/neurospatial/object_vector_field.py`

- [ ] Create module with docstring
- [ ] Implement `ObjectVectorFieldResult` frozen dataclass
  - [ ] Fields: `field`, `ego_env`, `occupancy`
- [ ] Implement `compute_object_vector_field()` function
  - [ ] Create egocentric polar environment via `from_egocentric_grid()`
  - [ ] Compute distance and bearing to nearest object at each timepoint
  - [ ] Support geodesic distance when `allocentric_env` provided
  - [ ] Map to egocentric bins
  - [ ] Support methods: "binned", "diffusion_kde"
  - [ ] Apply `min_occupancy_seconds` threshold
  - [ ] Return `ObjectVectorFieldResult`

**Success criteria**:
- Field peaks at `(preferred_distance, preferred_direction)` for simulated cells

### M2.4: Object-Vector Overlay (Animation)

**File**: `src/neurospatial/animation/overlays/object_vector.py`

- [ ] Create module with docstring
- [ ] Implement `ObjectVectorOverlay` dataclass
  - [ ] Required: `object_positions`, `animal_positions`
  - [ ] Optional: `firing_rates`, `times`
  - [ ] Styling: `color`, `linewidth`, `show_objects`, `object_marker`, `object_size`
  - [ ] Interpolation: `interp` ("linear", "nearest")
- [ ] Implement `convert_to_data()` method for frame alignment
- [ ] Register overlay in `animation/overlays/__init__.py`

### M2.5: Tests for Object-Vector Cells

**File**: `tests/simulation/models/test_object_vector_cells.py`

- [ ] Test model parameter validation
- [ ] Test firing rate with distance tuning only (no direction)
- [ ] Test firing rate with direction tuning
- [ ] Test object selectivity modes: any, nearest, specific
- [ ] Test geodesic vs Euclidean distances
- [ ] Test warning for objects outside environment
- [ ] Test `ground_truth` property matches parameters

**File**: `tests/metrics/test_object_vector_cells.py`

- [ ] Test `compute_object_vector_tuning()` binning
- [ ] Test `object_vector_score()` formulas
- [ ] Test `is_object_vector_cell()` classification
- [ ] Test recovery of ground truth from simulated cell
- [ ] Test `min_occupancy_seconds` filtering

**File**: `tests/test_object_vector_field.py`

- [ ] Test field computation with binned method
- [ ] Test field computation with diffusion_kde method
- [ ] Test ego_env returned correctly
- [ ] Test geodesic distance mode

### M2.6: Integration and Documentation

- [ ] Add exports to `src/neurospatial/__init__.py`
  - [ ] Export: `compute_object_vector_field`
- [ ] Add exports to `src/neurospatial/simulation/__init__.py`
  - [ ] Export: `ObjectVectorCellModel`
- [ ] Add exports to `src/neurospatial/metrics/__init__.py`
  - [ ] Export: `ObjectVectorMetrics`, `compute_object_vector_tuning`
  - [ ] Export: `object_vector_score`, `is_object_vector_cell`
- [ ] Add exports to `src/neurospatial/animation/overlays/__init__.py`
  - [ ] Export: `ObjectVectorOverlay`
- [ ] Update `.claude/QUICKSTART.md` with object-vector examples
- [ ] Update `.claude/API_REFERENCE.md` with object-vector imports

---

## Milestone 3: Spatial View Cells

**Goal**: Implement spatial view cell models, metrics, and visibility analysis.

**Priority**: MEDIUM

**Dependencies**: M1 complete, M2 patterns helpful

### M3.1: Visibility/Gaze Computation

**File**: `src/neurospatial/visibility.py`

- [ ] Create module with docstring
- [ ] Implement `compute_viewed_location()` function
  - [ ] Method: "fixed_distance" - point at fixed distance in gaze direction
  - [ ] Method: "ray_cast" - intersection with environment boundary
  - [ ] Method: "boundary" - nearest boundary point in gaze direction
  - [ ] Support optional `gaze_offsets` relative to heading
  - [ ] Mark as NaN when viewing outside environment
- [ ] Implement `_ray_cast_to_boundary()` helper
  - [ ] Use Shapely for boundary intersection
  - [ ] Iterative stepping with binary search refinement
- [ ] Implement `FieldOfView` frozen dataclass
  - [ ] Attributes: `left_angle`, `right_angle`, `binocular_half_angle`, `blind_spot_behind`
  - [ ] Class methods: `symmetric()`, `rat()`, `mouse()`, `primate()`, `bat()`
  - [ ] Properties: `total_angle`, `total_angle_degrees`
  - [ ] Method: `contains_angle()` - check if bearing in FOV
  - [ ] Method: `is_binocular()` - check if in binocular region
  - [ ] Validation in `__post_init__`
- [ ] Implement `ViewshedResult` frozen dataclass
  - [ ] Fields: `visible_bins`, `visible_boundary_segments`
  - [ ] Fields: `visible_cues`, `cue_distances`, `cue_bearings`
  - [ ] Field: `occlusion_map`
  - [ ] Properties: `n_visible_bins`, `visibility_fraction`, `n_visible_cues`
- [ ] Implement `compute_viewshed()` function
  - [ ] Cast n_rays from observer position
  - [ ] Find intersection with boundary for each ray
  - [ ] Mark bins between observer and intersection as visible
  - [ ] Support `fov` parameter (FieldOfView, float, or None for 360°)
  - [ ] Check cue visibility
  - [ ] Return `ViewshedResult`
- [ ] Implement `compute_view_field()` function
  - [ ] Binary mask of visible bins from position/heading
  - [ ] Uses `compute_viewshed()` internally
- [ ] Implement `visible_boundaries()` function
  - [ ] Return visible boundary segments as list of arrays
- [ ] Implement `visible_cues()` function
  - [ ] Check line-of-sight for each cue
  - [ ] Return: visible mask, distances, bearings
- [ ] Implement `_line_of_sight_clear()` helper using Shapely
- [ ] Implement `compute_viewshed_trajectory()` function
  - [ ] Vectorized computation along trajectory
- [ ] Implement `visibility_occupancy()` function
  - [ ] Time each bin was visible during trajectory

**Success criteria**:
- `FieldOfView.rat().total_angle_degrees` ≈ 300°
- `FieldOfView.primate().total_angle_degrees` ≈ 180°
- Blind spot behind correctly excludes rear regions

### M3.2: Spatial View Cell Model (Simulation)

**File**: `src/neurospatial/simulation/models/spatial_view_cells.py`

- [ ] Create module with docstring and references (Rolls et al. 1997)
- [ ] Implement `SpatialViewCellModel` dataclass
  - [ ] Required: `env`, `preferred_view_location`
  - [ ] Tuning: `view_field_width`
  - [ ] Gaze: `view_distance`, `gaze_model`
  - [ ] Rate: `max_rate`, `baseline_rate`
  - [ ] Visibility: `require_visibility`, `fov`
- [ ] Implement `firing_rate()` method
  - [ ] Compute viewed location at each timepoint
  - [ ] Distance from viewed location to preferred
  - [ ] Gaussian tuning
  - [ ] Handle NaN (viewing outside environment)
  - [ ] Optional visibility check
- [ ] Implement `ground_truth` property

**Success criteria**:
- Firing rate peaks when viewing `preferred_view_location`

### M3.3: Spatial View Field Analysis

**File**: `src/neurospatial/spatial_view_field.py`

- [ ] Create module with docstring
- [ ] Implement `compute_spatial_view_field()` function
  - [ ] Compute viewed location at each timepoint
  - [ ] Filter NaN (viewing outside environment)
  - [ ] Compute view occupancy
  - [ ] Interpolate viewed location at spike times
  - [ ] Bin spikes by viewed location
  - [ ] Normalize by view occupancy
  - [ ] Apply `min_occupancy_seconds` threshold
  - [ ] Optional smoothing (diffusion_kde, gaussian_kde)
  - [ ] Return field with same shape as place field

**Success criteria**:
- View field differs from place field for spatial view cells
- View field similar to place field for place cells

### M3.4: Spatial View Metrics

**File**: `src/neurospatial/metrics/spatial_view_cells.py`

- [ ] Create module with docstring
- [ ] Implement `SpatialViewMetrics` frozen dataclass
  - [ ] Fields: `view_field_skaggs_info`, `place_field_skaggs_info`
  - [ ] Fields: `view_place_correlation`
  - [ ] Fields: `view_field_sparsity`, `view_field_coherence`
  - [ ] Field: `is_spatial_view_cell`
  - [ ] Method: `interpretation()`
- [ ] Implement `spatial_view_cell_metrics()` function
  - [ ] Compute place field and view field
  - [ ] Compute Skaggs info for both (reuse from `place_fields.py`)
  - [ ] Compute sparsity, coherence (reuse existing)
  - [ ] Compute view-place correlation (Z-score, mask NaN)
  - [ ] Return `SpatialViewMetrics`
- [ ] Implement `is_spatial_view_cell()` classifier
  - [ ] `view_info > ratio * place_info` AND `correlation < max_corr`

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
