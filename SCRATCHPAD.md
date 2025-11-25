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

### Next Task
- **Milestone 3**: `annotate_video()` Integration

### Notes
- `BoundaryConfig` and `boundary_from_positions` exported via `neurospatial.annotation.__init__.py`
- Using `Literal["convex_hull", "alpha_shape", "kde"]` for method type
- KDE tests use gaussian blob data (uniform data has flat density - no contours)
- alphashape/scikit-image are lazy imports with clear error messages

### Blockers
None
