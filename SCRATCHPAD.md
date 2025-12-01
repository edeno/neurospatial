# SCRATCHPAD - Animation API Refactoring

**Started**: 2025-11-30
**Last Updated**: 2025-11-30

## Status

**ALL MILESTONES COMPLETE** ✅

- Milestone 1: Core Infrastructure ✅
- Milestone 2: Overlay Validation Refactor ✅
- Milestone 3: Core API Change ✅
- Milestone 4: Environment Method Update ✅
- Milestone 5: Napari Widget Enhancement ✅
- Milestone 6: Test Updates ✅
- Milestone 7: Demo Script Updates ✅
- Milestone 8: Documentation Updates ✅

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

### Milestone 5: Napari Widget Enhancement ✅

**Task 5.1 - Update `render_napari()` signature** ✅
- Added `speed: float = 1.0`, `sample_rate_hz: float | None = None`, `max_playback_fps: int = 60`
- Location: `src/neurospatial/animation/backends/napari_backend.py`
- Tests: `tests/animation/test_napari_backend.py::TestSpeedBasedAPI` (6 tests)

**Task 5.2 - Update `_add_speed_control_widget()` signature** ✅
- Added `initial_speed`, `sample_rate_hz`, `max_playback_fps` parameters
- Kept `initial_fps` for backwards compatibility (deprecated)
- Location: `src/neurospatial/animation/backends/napari_backend.py`
- Tests: `tests/animation/test_napari_backend.py::TestSpeedControlWidgetSignature` (4 tests)

**Task 5.3 - Implement speed-based slider** ✅
- Replaced fps slider with FloatSlider for speed (0.01-4.0)
- Added speed_info label showing "X.XX× (≈Y fps)"
- Widget computes fps from `min(sample_rate_hz * speed, max_playback_fps)`
- Location: `src/neurospatial/animation/backends/napari_backend.py`
- Tests: `tests/animation/test_napari_backend.py::TestSpeedSliderImplementation` (4 tests)

**Task 5.4 - Update call site in `render_napari()`** ✅
- Updated call to `_add_speed_control_widget()` to pass new parameters
- Updated `_render_multi_field_napari()` signature and call sites
- Renamed test `test_speed_control_widget_high_fps` → `test_speed_control_widget_high_sample_rate`
- Location: `src/neurospatial/animation/backends/napari_backend.py`

### Milestone 6: Test Updates ✅

Tests already completed as part of Milestones 1-5. Verified 1060 tests pass (8 pre-existing failures unrelated to this work).

### Milestone 7: Demo Script Updates ✅

**Task 7.1 - Update `data/demo_spike_overlay_napari.py`** ✅

- Replaced `--fps` CLI argument with `--speed` (default 1.0)
- Changed `main()` parameter from `fps: int = 20` to `speed: float = 1.0`
- Now uses data timestamps (`times_window`) directly for `frame_times`
- Updated `trail_length` to use computed `sample_rate_hz`
- Updated print statements: "data at X Hz, playback speed: Yx"
- Fixed variable name collision: renamed local `speed` (movement) to `animal_speed`
- Location: `data/demo_spike_overlay_napari.py`

**Task 7.2 - Update other demo scripts** ✅

- Updated `examples/16_field_animation.py` to use new API
- Created `frame_times` for trial-based data (1 second per trial)
- Created `large_session_frame_times` for 250 Hz session data
- Updated all `animate_fields()` calls to use `frame_times` and `speed`
- Synced `examples/16_field_animation.ipynb` via jupytext
- Updated common patterns code comments

### Milestone 8: Documentation Updates ✅

**Task 8.1 - Update CLAUDE.md Quick Reference** ✅
- Updated all animate_fields examples to use `frame_times` (required)
- Added examples for different use cases (30 Hz position, 500 Hz replay)
- Showed speed=0.1 for slow motion, speed=2.0 for fast forward
- Updated troubleshooting section examples
- Fixed tests missing `frame_times`: scale_bar tests, layout_integration test

**Task 8.2 - Update CLAUDE.md animation section** ✅
- Added "Animation Playback Control (v0.15.0+)" section
- Documented speed parameter semantics with use case table
- Explained playback_fps = sample_rate_hz * speed formula
- Documented 60 fps cap and max_playback_fps override

**Task 8.3 - Add migration notes** ✅
- Added Gotcha #15: "Animation API migration (v0.15.0+)"
- Documented breaking changes in table format
- Showed before/after code examples
- Added migration steps and rationale

## Next Steps

All milestones complete! Animation API refactoring is done.

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
