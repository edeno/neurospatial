# TASKS: Position-Based Boundary Seeding for Annotation Module

**Reference**: [PLAN.md](PLAN.md) for full design details, code examples, and rationale.

## Milestone 1: Core Boundary Inference Module

Create the boundary inference algorithms and configuration.

**File**: `src/neurospatial/annotation/_boundary_inference.py` (CREATE)

- [x] Create `BoundaryConfig` frozen dataclass with parameters: `method`, `buffer_fraction`, `simplify_fraction`, `alpha`, `kde_threshold`, `kde_sigma`, `kde_max_bins`
- [x] Implement `boundary_from_positions()` function with config/override pattern
- [x] Implement `_convex_hull_boundary()` using scipy ConvexHull
- [x] Implement `_alpha_shape_boundary()` with lazy alphashape import and MultiPolygon warning
- [x] Implement `_kde_boundary()` with lazy scikit-image import and max_bins cap
- [x] Add input validation (shape, minimum points, unique points, collinearity)
- [x] Apply buffer and simplification as fraction of bbox diagonal

## Milestone 2: Napari Integration

Add helper to seed shapes layer with pre-drawn boundary.

**File**: `src/neurospatial/annotation/_napari_widget.py` (MODIFY)

- [x] Implement `add_initial_boundary_to_shapes()` function
- [x] Handle coordinate transform via `calibration.transform_cm_to_px()` (do NOT double-flip Y)
- [x] Convert to napari (row, col) order
- [x] Preserve existing features and prepend boundary to front
- [x] Sync face colors from features

## Milestone 3: annotate_video() Integration

Wire boundary seeding into the main annotation function.

**File**: `src/neurospatial/annotation/core.py` (MODIFY)

- [ ] Add parameters: `initial_boundary`, `boundary_config`, `show_positions`
- [ ] Detect NDArray vs Polygon for `initial_boundary` and dispatch appropriately
- [ ] Handle conflict: warn and filter if both `initial_boundary` and environment region in `initial_regions`
- [ ] Implement `_add_positions_layer()` helper for trajectory reference display
- [ ] Ensure correct call order: `_add_initial_regions()` → `add_initial_boundary_to_shapes()` → positions layer

## Milestone 4: Public API Export

Expose new functionality in public API.

**File**: `src/neurospatial/annotation/__init__.py` (MODIFY)

- [ ] Export `BoundaryConfig` from `_boundary_inference`
- [ ] Export `boundary_from_positions` from `_boundary_inference`
- [ ] Add to `__all__` list

## Milestone 5: Tests

Comprehensive test coverage for boundary inference.

**File**: `tests/annotation/test_boundary_inference.py` (CREATE)

- [ ] `TestBoundaryConfig`: default values, frozen immutability
- [ ] `TestConvexHull`: basic hull, square points, minimum points validation
- [ ] `TestBuffer`: area increase, scaling with bbox diagonal
- [ ] `TestSimplify`: vertex reduction
- [ ] `TestKDE`: import error handling, basic validity, max_bins cap, threshold effect on area
- [ ] `TestAlphaShape`: import error handling, MultiPolygon warning
- [ ] `TestWithConfig`: config overrides defaults, kwargs override config

## Milestone 6: Dependencies

Update optional dependencies for annotation extras.

**File**: `pyproject.toml` (MODIFY)

- [ ] Add `annotation-kde = ["scikit-image>=0.19.0"]` optional dependency
- [ ] Add `annotation-alpha = ["alphashape>=1.3.0"]` optional dependency
- [ ] Add `annotation = [...]` combined optional dependency

## Milestone 7: Documentation

Update CLAUDE.md with usage examples.

**File**: `CLAUDE.md` (MODIFY)

- [ ] Add boundary seeding examples to Quick Reference section
- [ ] Document `annotate_video()` with `initial_boundary` parameter
- [ ] Document `BoundaryConfig` usage for fine-tuning
- [ ] Document `show_positions=True` for trajectory reference
- [ ] Document `boundary_from_positions()` for composable usage
