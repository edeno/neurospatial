# Changelog

## [Unreleased]

### Added

- **Spatial View Cells**: Complete spatial view cell analysis infrastructure
  - `compute_spatial_view_field()` - Compute firing fields indexed by viewed location (not position)
  - `SpatialViewFieldResult` frozen dataclass with field, view_occupancy
  - `SpatialViewMetrics` frozen dataclass with Skaggs info, sparsity, coherence, classification
  - `spatial_view_cell_metrics()` - Compare view fields vs place fields
  - `is_spatial_view_cell()` - Classify based on view info > place info
  - `SpatialViewCellModel` simulation model with gaze models (fixed_distance, ray_cast, boundary)
  - 83 new tests for spatial view cell modules

- **Visibility and Gaze Analysis**: Ray-casting visibility for spatial view cells
  - `FieldOfView` frozen dataclass with species presets (rat ~320°, primate ~180°)
  - `ViewshedResult` frozen dataclass with visible bins, cues, occlusion map
  - `compute_viewed_location()` - Gaze-directed location computation
  - `compute_viewshed()` - Ray-casting visibility analysis from observer position
  - `compute_view_field()` - Binary visibility mask
  - `visible_cues()` - Check line-of-sight to cue/landmark positions
  - `compute_viewshed_trajectory()` - Viewshed analysis along trajectory
  - `visibility_occupancy()` - Time each bin was visible during trajectory
  - 45 new tests for visibility module

- **Object-Vector Cells**: Complete object-vector cell analysis infrastructure
  - `compute_object_vector_field()` - Compute firing fields in egocentric polar coordinates
  - `ObjectVectorFieldResult` frozen dataclass with field, ego_env, occupancy
  - `ObjectVectorMetrics` frozen dataclass with tuning curve, selectivity scores
  - `compute_object_vector_tuning()` - Bin spikes by egocentric distance/direction to objects
  - `object_vector_score()` - Combined distance and direction selectivity metric
  - `is_object_vector_cell()` - Classify based on score threshold
  - `plot_object_vector_tuning()` - Polar heatmap visualization
  - `ObjectVectorCellModel` simulation model with distance/direction tuning
  - `ObjectVectorOverlay` for animation with object-animal vectors
  - 84 new tests for object-vector cell modules

- **Egocentric Reference Frames**: Foundation for object-vector and spatial view cells
  - `EgocentricFrame` dataclass with `to_egocentric()` / `to_allocentric()` transforms
  - `allocentric_to_egocentric()` - Batch transform world→animal coordinates
  - `egocentric_to_allocentric()` - Batch inverse transform
  - `compute_egocentric_bearing()` - Angle to targets relative to heading (0=ahead, π/2=left)
  - `compute_egocentric_distance()` - Euclidean and geodesic distance metrics
  - `heading_from_velocity()` - Compute heading from position timeseries with smoothing
  - `heading_from_body_orientation()` - Compute heading from pose keypoints (nose-tail)
  - `Environment.from_polar_egocentric()` - Create egocentric polar coordinate environment
  - Circular connectivity option for full-circle polar environments
  - 56 new tests for reference frame and polar environment modules

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

### Changed

- **Internal**: Refactored `environment.py` (5,335 lines) into modular package structure for improved maintainability
  - Split into 11 focused modules: `core.py` (1,023 lines), `factories.py` (630 lines), `queries.py` (897 lines), `trajectory.py` (1,222 lines), `transforms.py` (634 lines), `fields.py` (564 lines), `metrics.py` (469 lines), `regions.py` (398 lines), `serialization.py` (315 lines), `visualization.py` (211 lines), `decorators.py` (77 lines)
  - Implemented mixin pattern: Environment inherits from all functionality mixins
  - No breaking changes - `from neurospatial import Environment` continues to work
  - All 1,076 tests passing (100% success rate)
  - Improved code organization for easier contribution and maintenance
  - Largest module is trajectory.py at 1,222 lines (down from original analysis.py at 2,104 lines)

### Documentation

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
