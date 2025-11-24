# SCRATCHPAD - v0.8.0 Development

**Current Milestone**: M1.1 - Public API Fixes
**Date**: 2025-11-24
**Status**: In Progress

---

## Current Session Notes

### M1.1.1 - Verify function names ✅ COMPLETE

Verified all three functions exist in segmentation module:
- `detect_goal_directed_runs` → `segmentation/similarity.py:251`
- `detect_runs_between_regions` → `segmentation/regions.py:181`
- `segment_by_velocity` → `segmentation/regions.py:390`

All are already exported in `segmentation/__init__.py` but **NOT** in main `neurospatial/__init__.py`.

### M1.1 - Public API Exports ✅ COMPLETE

**TDD Workflow Followed:**
1. ✅ Wrote import tests FIRST (tests/test_segmentation.py)
2. ✅ Ran tests → FAIL (4/4 failed with ImportError)
3. ✅ Updated src/neurospatial/__init__.py with missing imports
4. ✅ Ran tests → PASS (4/4 passed)
5. ✅ Code quality checks (ruff, mypy) → all passed

**Changes Made:**
- Added 3 missing functions to import statement (lines 237-243)
- Added 3 missing functions to __all__ list (lines 295-300)
- Functions: detect_goal_directed_runs, detect_runs_between_regions, segment_by_velocity

**Tests Created:**
- test_detect_goal_directed_runs_exported()
- test_detect_runs_between_regions_exported()
- test_segment_by_velocity_exported()
- test_all_segmentation_functions_in_all()

---

## Decisions & Blockers

None.

---

## Next Steps

After M1.1 complete:
- Move to M2.1: Create behavioral.py module
- Implement trials_to_region_arrays() helper (with tests first)
