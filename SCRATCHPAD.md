# Animation Overlays v0.4.0 - Development Scratchpad

**Started:** 2025-11-20
**Current Milestone:** Milestone 2 COMPLETE ✅ (Protocol & Core Dispatcher)
**Status:** Ready for Milestone 3: Napari Backend Updates

---

## Current Task

**Task:** Update Napari backend for overlay rendering (`src/neurospatial/animation/backends/napari_backend.py`)

**Approach:**
- Following TDD: Write tests first, then implementation
- Using NumPy docstring format for all documentation
- Ensuring mypy type checking passes
- Protocol-based design (no inheritance)

---

## Progress Notes

### 2025-11-20

**Starting Point:**
- Read ANIMATION_IMPLEMENTATION_PLAN.md - comprehensive design for overlay feature
- Read TASKS.md - detailed checklist with 9 milestones
- Created SCRATCHPAD.md to track progress
- First task: Create overlay dataclasses (PositionOverlay, BodypartOverlay, HeadDirectionOverlay)

**Status:** ✅ **MILESTONE 1 COMPLETE** (All sub-milestones: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6)

**Completed:**
1. ✅ Created comprehensive test file with 19 tests (Milestone 1.1 & 1.2)
2. ✅ Verified tests fail (RED phase) - ModuleNotFoundError
3. ✅ Implemented all dataclasses (GREEN phase) - all 19 tests pass
4. ✅ Added comprehensive NumPy docstrings with examples
5. ✅ Applied code review and fixed all issues
6. ✅ Exported dataclasses in main __init__.py
7. ✅ Implemented timeline & interpolation helpers (Milestone 1.3)
   - _build_frame_times() with monotonicity validation
   - _interp_linear() vectorized linear interpolation
   - _interp_nearest() vectorized nearest neighbor interpolation
   - 22 comprehensive tests (all passing)
   - Full NumPy docstrings with examples
   - Mypy and ruff pass
8. ✅ Implemented validation functions (Milestone 1.4)
   - _validate_monotonic_time() - detects non-monotonic timestamps
   - _validate_finite_values() - detects NaN/Inf with counts
   - _validate_shape() - validates coordinate dimensions
   - _validate_temporal_alignment() - checks overlay/frame overlap
   - _validate_bounds() - warns for out-of-bounds points
   - _validate_skeleton_consistency() - validates skeleton names with fuzzy suggestions
   - _validate_pickle_ability() - ensures parallel rendering compatibility
   - 38 comprehensive tests (all passing) including 2 added from code review
   - All error messages follow WHAT/WHY/HOW format
   - Code review rating: APPROVE
   - Mypy and ruff pass
9. ✅ Implemented conversion funnel (Milestone 1.5 & 1.6)
   - _convert_overlays_to_data() function (249 lines)
   - Converts all overlay types (Position, Bodypart, HeadDirection) to internal data
   - Per-keypoint interpolation for BodypartOverlay
   - Temporal alignment with linear interpolation
   - NaN extrapolation outside source time range
   - Handles overlays with and without timestamps
   - Comprehensive validation during conversion
   - 16 additional tests (73 total tests, all passing)
   - Full NumPy docstring with examples
   - Code review rating: APPROVE (no critical issues)
   - Mypy and ruff pass
   - Test summary: 73 passed, 1 warning (expected temporal overlap warning)
10. ✅ Updated TASKS.md checkboxes (Milestone 1 complete - 11% overall progress)

11. ✅ Updated EnvironmentProtocol and implementation (Milestone 2.1)
   - Added `overlays` parameter to `animate_fields()` signature
   - Added `frame_times`, `show_regions`, and `region_alpha` parameters
   - Updated `src/neurospatial/environment/_protocols.py` protocol
   - Updated `src/neurospatial/environment/visualization.py` implementation
   - Added comprehensive NumPy docstrings for all new parameters
   - Imports overlay types with TYPE_CHECKING guard
   - Mypy and ruff pass
   - All 73 overlay tests still pass

12. ✅ Updated Core Dispatcher (Milestone 2.2 & 2.3 COMPLETE)
   - Updated `src/neurospatial/animation/core.py` dispatcher signature
   - Added overlay type imports with TYPE_CHECKING guard
   - Implemented frame_times building/verification using _build_frame_times()
   - Added conversion funnel call when overlays provided
   - Updated all 4 backend routing calls (napari, video, html, widget) to pass:
     - overlay_data parameter
     - show_regions parameter
     - region_alpha parameter
   - Added comprehensive dispatcher docstring with NumPy format
   - Added 7 new integration tests in TestDispatcherOverlayIntegration class
   - All 35 tests passing (28 existing + 7 new)
   - Mypy and ruff pass
   - Code review rating: APPROVE (no critical issues, minor docstring update addressed)
   - Test summary: 35 passed in test_core.py

**Next Steps:**
- Commit Milestone 2.2 completion with conventional commit message
- Continue with Milestone 3: Napari Backend Updates

---

## Decisions & Design Notes

### Overlay Dataclasses Design
- Three public dataclasses: PositionOverlay, BodypartOverlay, HeadDirectionOverlay
- All support optional timestamps for temporal alignment
- Immutable where appropriate (consider frozen=True)
- NumPy docstrings with Examples section
- Comprehensive Attributes sections added to all data containers
- See Also cross-references between public and internal containers

**Code Review Fixes Applied:**
- ✅ Fixed doctest failure in PositionOverlay (undefined variables)
- ✅ Updated OverlayData.__post_init__ docstring (clarified placeholder status)
- ✅ Added Attributes sections to all internal data containers
- ✅ Added Notes section to OverlayData explaining usage
- ✅ Added See Also cross-references to internal containers

### Validation Strategy
- WHAT/WHY/HOW format for all error messages
- Actionable guidance in every error
- Warnings vs errors: errors block rendering, warnings inform user
- Intelligent suggestions using difflib.get_close_matches() for typo detection

### Conversion Funnel Design (Milestone 1.5)
- Single function `_convert_overlays_to_data()` handles all overlay types
- Three-section structure: PositionOverlay → BodypartOverlay → HeadDirectionOverlay
- Consistent pattern for each overlay type:
  1. Validate finite values
  2. Validate shape (dimensions)
  3. Validate/align times (if provided)
  4. Interpolate or length-check
  5. Validate bounds (warning only)
  6. Create internal data container
- Per-keypoint interpolation preserves independent temporal dynamics
- NaN extrapolation for scientifically correct handling of missing data
- Pickle-safe OverlayData output for parallel rendering
- Code review rating: APPROVE (249 lines justified for conversion pipeline)

---

## Blockers & Questions

*None currently*

---

## Testing Notes

- All tests must pass before moving to next task
- Use `uv run pytest` for all test execution
- Performance tests marked with `@pytest.mark.slow`
- Visual regression tests using pytest-mpl

---

## Useful Commands

```bash
# Run tests for current work
uv run pytest tests/animation/test_overlays.py -v

# Run with coverage
uv run pytest tests/animation/ --cov=src/neurospatial/animation/overlays.py

# Type check
uv run mypy src/neurospatial/animation/overlays.py

# Lint and format
uv run ruff check src/neurospatial/animation/ && uv run ruff format src/neurospatial/animation/

# Run all tests
uv run pytest

# Commit with conventional format
git commit -m "feat(animation): add overlay dataclasses"
```
