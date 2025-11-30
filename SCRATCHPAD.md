# SCRATCHPAD - Animation API Refactoring

**Started**: 2025-11-30
**Current Task**: 1.1 Add playback constants to `animation/core.py`

## Status

Working on Milestone 1: Core Infrastructure

## Notes

### Task 1.1 - Add Playback Constants
- Need to add: `MAX_PLAYBACK_FPS = 60`, `MIN_PLAYBACK_FPS = 1`, `DEFAULT_SPEED = 1.0`
- Location: Top of `src/neurospatial/animation/core.py` after imports
- Following TDD: write tests first, then implement

### Existing Structure
- `animation/core.py` has `animate_fields()`, `subsample_frames()`, `estimate_colormap_range_from_subset()`, `large_session_napari_config()`
- Test file: `tests/animation/test_core.py` exists

## Blockers

None currently.

## Decisions

- Constants will be module-level for easy importing
