# Changelog

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
