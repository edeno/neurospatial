# Animation Overlays v0.4.0 - Development Scratchpad

**Started:** 2025-11-20
**Current Milestone:** 1.1 Public API Dataclasses
**Status:** Starting implementation

---

## Current Task

**Task:** Create `src/neurospatial/animation/overlays.py` module with public API dataclasses

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

**Status:** ✅ Milestones 1.1, 1.2, and 1.3 COMPLETE

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
8. ✅ Updated TASKS.md checkboxes

**Next Steps:**
- Continue with Milestone 1.4: Validation Functions (WHAT/WHY/HOW)
- Or commit current progress first

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
