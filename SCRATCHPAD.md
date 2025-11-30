# SCRATCHPAD - Animation API Refactoring

**Started**: 2025-11-30
**Last Updated**: 2025-11-30

## Status

**Milestones 1 and 2 COMPLETE**

- Milestone 1: Core Infrastructure ✅
- Milestone 2: Overlay Validation Refactor ✅
- Milestone 3: Core API Change (next)

## Completed Work

### Milestone 1: Core Infrastructure ✅

**Task 1.1 - Add Playback Constants** ✅
- Added `MAX_PLAYBACK_FPS = 60`, `MIN_PLAYBACK_FPS = 1`, `DEFAULT_SPEED = 1.0`
- Location: `src/neurospatial/animation/core.py` (lines 21-24)
- Tests: `tests/animation/test_core.py::TestPlaybackConstants` (10 tests)
- Commit: `d986f2e`

**Task 1.2 - Implement `_compute_playback_fps()` helper** ✅
- Function signature: `(frame_times, speed, max_fps) -> tuple[int, float]`
- Handles edge cases: single frame, zero duration, min/max clamping
- Location: `src/neurospatial/animation/core.py` (lines 27-81)
- Tests: `tests/animation/test_core.py::TestComputePlaybackFps` (10 tests)
- Commit: `d986f2e`

### Milestone 2: Overlay Validation Refactor ✅

**Task 2.1 - Rename `_build_frame_times()` → `_validate_frame_times()`** ✅
- New function only validates, does NOT synthesize from fps
- Signature: `(frame_times, n_frames) -> NDArray`
- Location: `src/neurospatial/animation/overlays.py` (lines 2932-2999)
- Tests: `tests/animation/test_timeline_helpers.py::TestValidateFrameTimes` (9 tests)
- Commit: `8af7335`

**Task 2.2 - Update call sites** ✅
- Kept `_build_frame_times()` as backwards compatibility alias
- Alias delegates to `_validate_frame_times()` when `frame_times` is provided
- Full migration to `_validate_frame_times()` will happen in Milestone 3
- Commit: `8af7335`

## Next Steps

**Milestone 3: Core API Change** (Tasks 3.1-3.3)
- Make `frame_times` required parameter (no default)
- Add `speed: float = 1.0` parameter
- Add `max_playback_fps: int = 60` parameter
- Emit warning when speed is capped

## Blockers

None currently.

## Decisions Made

1. **Constants as module-level** - Easy importing: `from neurospatial.animation.core import MAX_PLAYBACK_FPS`
2. **Backwards compatibility alias** - `_build_frame_times` kept to avoid breaking existing code
3. **TDD workflow** - All features implemented with tests first
4. **Strict monotonicity** - `_validate_frame_times` requires strictly increasing timestamps (no duplicates)

## Test Coverage

All tests pass (102 tests in animation/test_core.py and test_timeline_helpers.py):
- TestPlaybackConstants: 10 tests
- TestComputePlaybackFps: 10 tests
- TestValidateFrameTimes: 9 tests
- Other existing tests: 73 tests
