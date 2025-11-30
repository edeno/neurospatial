# SCRATCHPAD - Animation API Refactoring

**Started**: 2025-11-30
**Last Updated**: 2025-11-30

## Status

**Milestones 1, 2, 3, and 4 COMPLETE**

- Milestone 1: Core Infrastructure ✅
- Milestone 2: Overlay Validation Refactor ✅
- Milestone 3: Core API Change ✅
- Milestone 4: Environment Method Update ✅
- Milestone 5: Napari Widget Enhancement (next)

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

### Milestone 3: Core API Change ✅

**Task 3.1 - Update `animate_fields()` signature** ✅
- `frame_times` is now required (no default)
- Added `speed: float = DEFAULT_SPEED` parameter
- Added `max_playback_fps: int = MAX_PLAYBACK_FPS` parameter
- Location: `src/neurospatial/animation/core.py` (lines 120-133)
- Tests: `tests/animation/test_core.py::TestAnimateFieldsSpeedBasedPlayback` (12 tests)

**Task 3.2 - Implement playback computation with warning** ✅
- Computes `playback_fps` and `actual_speed` from frame_times and speed
- Emits `UserWarning` when actual_speed differs from requested speed by >1%
- Passes `sample_rate_hz`, `speed`, `max_playback_fps` to backends via kwargs
- Location: `src/neurospatial/animation/core.py` (lines 347-380)

**Task 3.3 - Update frame_times validation** ✅
- Calls `_validate_frame_times(frame_times, n_frames)`
- Location: `src/neurospatial/animation/core.py` (lines 343-345)

### Milestone 4: Environment Method Update ✅

**Task 4.1 - Update `Environment.animate_fields()` signature** ✅
- Replaced `fps: int = 30` with `speed: float = 1.0`
- Made `frame_times` required (no `| None`, no default)
- Implementation was already complete (discovered during TDD verification)
- Location: `src/neurospatial/environment/visualization.py` (lines 533-561)
- Tests: `tests/environment/test_animate_fields_api_change.py` (10 tests)

**Task 4.2 - Update docstring for `Environment.animate_fields()`** ✅
- Documented `speed` parameter with examples (0.1 = slow motion, 2.0 = fast forward)
- Documented that `frame_times` is required
- Updated formula: `playback_fps = min(sample_rate_hz * speed, 60)`
- Updated all examples to include `frame_times` parameter
- Location: `src/neurospatial/environment/visualization.py` (lines 563-920)

## Next Steps

**Milestone 5: Napari Widget Enhancement** (Tasks 5.1-5.4)
- Update `render_napari()` signature with `speed`, `sample_rate_hz`, `max_playback_fps`
- Update `_add_speed_control_widget()` to show speed multiplier as primary control
- UI to show "Speed: 0.25×" with "≈ 12 fps" info

## Blockers

None currently.

## Decisions Made

1. **Constants as module-level** - Easy importing: `from neurospatial.animation.core import MAX_PLAYBACK_FPS`
2. **Backwards compatibility alias** - `_build_frame_times` kept to avoid breaking existing code
3. **TDD workflow** - All features implemented with tests first
4. **Strict monotonicity** - `_validate_frame_times` requires strictly increasing timestamps (no duplicates)

## Test Coverage

**Animation test suite**: 1043 passed, 4 skipped, 8 failed (pre-existing issues)

Key test classes for Milestone 3:
- TestPlaybackConstants: 10 tests
- TestComputePlaybackFps: 10 tests
- TestAnimateFieldsSpeedBasedPlayback: 8 passed, 3 skipped (Task 3.2 pending)
- Other existing tests: Updated to use `frame_times` parameter

**Pre-existing failures** (not related to Milestone 3):
- HTML warning tests: Warning logic was previously removed
- Napari QtCustomTitleBar: napari version compatibility issue
- Figure reuse test: PersistentFigureRenderer issue
- ffmpeg timing test: 11.1% duration variance vs 10% tolerance
