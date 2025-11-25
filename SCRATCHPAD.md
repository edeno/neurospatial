# SCRATCHPAD: Position-Based Boundary Seeding

## Session 2025-11-24

### Milestone 1: COMPLETE

All 7 tasks in Milestone 1 completed:
- [x] **M1.1**: Created `BoundaryConfig` frozen dataclass
- [x] **M1.2**: Implemented `boundary_from_positions()` with config/override pattern
- [x] **M1.3**: Implemented `_convex_hull_boundary()` using scipy ConvexHull
- [x] **M1.4**: Implemented `_alpha_shape_boundary()` with lazy alphashape import + MultiPolygon warning
- [x] **M1.5**: Implemented `_kde_boundary()` with lazy scikit-image import + max_bins cap
- [x] **M1.6**: Added input validation (shape, min points, unique points)
- [x] **M1.7**: Applied buffer and simplification as fraction of bbox diagonal

### Milestone 2: COMPLETE

All 5 tasks in Milestone 2 completed:
- [x] **M2.1**: Implemented `add_initial_boundary_to_shapes()` function
- [x] **M2.2**: Handle coordinate transform via `calibration.transform_cm_to_px()`
- [x] **M2.3**: Convert to napari (row, col) order
- [x] **M2.4**: Preserve existing features and prepend boundary to front
- [x] **M2.5**: Sync face colors from features

**Test Results**: 22 passed, 3 skipped
**Quality**: ruff and mypy pass (also fixed 2 pre-existing unused type-ignore comments)

### Milestone 3: COMPLETE

All 5 tasks in Milestone 3 completed:
- [x] **M3.1**: Added `initial_boundary`, `boundary_config`, `show_positions` params to `annotate_video()`
- [x] **M3.2**: Detect NDArray vs Polygon and dispatch (uses `isinstance` check, calls `boundary_from_positions()` for NDArray)
- [x] **M3.3**: Implemented `_filter_environment_regions()` with warning when conflict detected
- [x] **M3.4**: Implemented `_add_positions_layer()` helper with subsampling for large arrays
- [x] **M3.5**: Correct call order: initial_regions → initial_boundary → positions layer

**Test Results**: 162 passed, 3 skipped
**Quality**: ruff and mypy pass

### Next Task
- **Milestone 4**: Public API Export (BoundaryConfig and boundary_from_positions already exported in M1)

### Notes
- `BoundaryConfig` and `boundary_from_positions` exported via `neurospatial.annotation.__init__.py`
- Using `Literal["convex_hull", "alpha_shape", "kde"]` for method type
- KDE tests use gaussian blob data (uniform data has flat density - no contours)
- alphashape/scikit-image are lazy imports with clear error messages

### Blockers
None
