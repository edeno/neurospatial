# TASKS.md - Animation API Refactoring

**Goal**: Separate data sample rate from playback speed in the animation API

**Summary**: Replace the ambiguous `fps` parameter with explicit `frame_times` (required) and `speed` multiplier. This enables proper temporal semantics for different analysis types (replay decoding at 500 Hz vs position tracking at 30 Hz).

---

## Milestone 1: Core Infrastructure

**Objective**: Add foundational constants and helper functions to `animation/core.py`

### Tasks

- [x] **1.1 Add playback constants to `animation/core.py`**
  - Add `MAX_PLAYBACK_FPS: int = 60` (display refresh rate limit)
  - Add `MIN_PLAYBACK_FPS: int = 1` (minimum usable playback)
  - Add `DEFAULT_SPEED: float = 1.0` (real-time by default)
  - Location: Top of file after imports
  - Success: Constants importable from module

- [x] **1.2 Implement `_compute_playback_fps()` helper in `animation/core.py`**
  - Signature: `(frame_times, speed, max_fps) -> tuple[int, float]`
  - Compute sample_rate_hz from frame_times duration
  - Return `(playback_fps, actual_speed)` tuple
  - Handle edge cases: single frame, zero duration
  - Success: Function passes unit tests for normal and edge cases

---

## Milestone 2: Overlay Validation Refactor

**Objective**: Simplify frame time handling by removing synthesis logic

### Tasks

- [x] **2.1 Rename `_build_frame_times()` → `_validate_frame_times()` in `animation/overlays.py`**
  - Remove `fps` parameter from signature
  - Remove `np.linspace` synthesis branch
  - Keep validation logic (length check, monotonicity check)
  - New signature: `(frame_times, n_frames) -> NDArray`
  - Success: Function only validates, never synthesizes

- [x] **2.2 Update all call sites of `_build_frame_times()`**
  - Search for usages in `animation/core.py`
  - Update calls to use new name and signature
  - Success: All tests pass after rename

---

## Milestone 3: Core API Change

**Objective**: Update `animate_fields()` to use new speed-based API

**Dependencies**: Milestones 1, 2

### Tasks

- [x] **3.1 Update `animate_fields()` signature in `animation/core.py`**
  - Make `frame_times: NDArray[np.float64]` required (no default)
  - Add `speed: float = 1.0` parameter
  - Add `max_playback_fps: int = 60` parameter
  - Remove `fps` from `**kwargs` extraction if present
  - Success: Signature matches plan specification

- [x] **3.2 Implement playback computation in `animate_fields()`**
  - Call `_compute_playback_fps(frame_times, speed, max_playback_fps)`
  - Emit `UserWarning` if `actual_speed` differs from `speed` by >1%
  - Compute `sample_rate_hz` for napari backend
  - Pass `fps`, `sample_rate_hz`, `speed`, `max_playback_fps` to backends via kwargs
  - Success: Warning emitted for 500 Hz data at speed=1.0

- [x] **3.3 Update `frame_times` validation in `animate_fields()`**
  - Call `_validate_frame_times(frame_times, n_frames)`
  - Raise `ValueError` with helpful message if length mismatch
  - Success: Clear error message guides users to fix timestamp issues

---

## Milestone 4: Environment Method Update

**Objective**: Update the user-facing `Environment.animate_fields()` method

**Dependencies**: Milestone 3

### Tasks

- [x] **4.1 Update `Environment.animate_fields()` in `environment/visualization.py`**
  - Replace `fps: int = 30` with `speed: float = 1.0`
  - Make `frame_times` required parameter (remove `Optional`)
  - Pass through to core `animate_fields()`
  - Success: Method signature matches core function

- [x] **4.2 Update docstring for `Environment.animate_fields()`**
  - Document `speed` parameter with examples (0.1 = slow motion, 2.0 = fast forward)
  - Document that `frame_times` is required
  - Explain playback fps is computed as `sample_rate_hz * speed`
  - Note 60 fps cap for display compatibility
  - Success: Docstring follows NumPy format with clear examples

---

## Milestone 5: Napari Widget Enhancement

**Objective**: Update napari backend to show speed multiplier as primary control

**Dependencies**: Milestone 3

### Tasks

- [x] **5.1 Update `render_napari()` signature in `napari_backend.py`**
  - Add `speed: float = 1.0` parameter
  - Add `sample_rate_hz: float | None = None` parameter
  - Add `max_playback_fps: int = 60` parameter
  - Remove direct `fps` parameter usage (receive via kwargs)
  - Success: Function accepts new parameters

- [x] **5.2 Update `_add_speed_control_widget()` signature**
  - Change from `initial_fps: int` to:
    - `initial_speed: float = 1.0`
    - `sample_rate_hz: float = 30.0`
    - `max_playback_fps: int = 60`
  - Keep `frame_labels` parameter
  - Success: New signature matches plan

- [x] **5.3 Implement speed-based slider in `_add_speed_control_widget()`**
  - Replace fps slider with speed multiplier slider (0.01 to 4.0)
  - Add secondary info label showing "≈ X fps"
  - Internally compute fps = min(sample_rate_hz * speed, max_playback_fps)
  - Update napari settings with computed fps
  - Success: UI shows "Speed: 0.25×" with "≈ 12 fps" info

- [x] **5.4 Update call site in `render_napari()`**
  - Pass `initial_speed`, `sample_rate_hz`, `max_playback_fps` to widget
  - Remove `initial_fps` usage
  - Success: Widget initializes with correct speed

---

## Milestone 6: Test Updates

**Objective**: Update existing tests and add new test coverage

**Dependencies**: Milestones 1-5

### Tasks

- [x] **6.1 Add unit tests for `_compute_playback_fps()`**
  - Test normal case: 30 Hz data at speed=1.0 → 30 fps
  - Test capping: 500 Hz data at speed=1.0 → 60 fps, actual_speed≈0.12
  - Test slow motion: 30 Hz at speed=0.1 → 3 fps
  - Test minimum clamping: 10 Hz at speed=0.01 → 1 fps
  - Test edge case: single frame → returns max_fps
  - Test edge case: zero duration → returns max_fps
  - Location: `tests/animation/test_core.py::TestComputePlaybackFps`
  - Success: All edge cases covered

- [x] **6.2 Update existing `animate_fields` tests**
  - Replace `fps=X` with `speed=X` where applicable
  - Add `frame_times` parameter to all test calls
  - Update mock assertions to check for speed-related kwargs
  - Verified: 26 test calls use frame_times=, none use fps= with animate_fields
  - Success: All existing tests pass

- [x] **6.3 Add warning emission test**
  - Test that warning is emitted when speed is capped
  - Use `pytest.warns(UserWarning, match="Capped to 60 fps")`
  - Test with 500 Hz data at speed=1.0
  - Location: `tests/animation/test_core.py::test_warning_emitted_when_speed_capped`
  - Success: Warning message matches expected format

- [x] **6.4 Add frame_times validation tests**
  - Test ValueError on length mismatch
  - Test ValueError on non-monotonic timestamps
  - Test that valid timestamps pass through
  - Location: `tests/animation/test_timeline_helpers.py::TestValidateFrameTimes`
  - Success: Clear error messages for invalid input

---

## Milestone 7: Demo Script Updates

**Objective**: Update demo scripts to use new API

**Dependencies**: Milestones 1-4

### Tasks

- [x] **7.1 Update `data/demo_spike_overlay_napari.py`**
  - Remove `--fps` CLI argument
  - Add `--speed` CLI argument (default=1.0)
  - Use data timestamps for `frame_times` instead of synthesizing
  - Update `trail_length` calculation to use sample_rate
  - Update print statements to show sample rate and speed
  - Fixed variable name collision: `speed` (playback) vs `animal_speed` (movement)
  - Success: Demo runs with new API, shows "data at X Hz, playback speed: Yx"

- [x] **7.2 Search and update other demo scripts**
  - Run: `grep -r "animate_fields.*fps" data/ scripts/ examples/`
  - Updated `examples/16_field_animation.py` (and synced .ipynb via jupytext)
  - Created appropriate `frame_times` for trial-based and large-session examples
  - Success: No scripts use old `fps` parameter

---

## Milestone 8: Documentation Updates

**Objective**: Update documentation to reflect API changes

**Dependencies**: Milestones 1-7

### Tasks

- [ ] **8.1 Update CLAUDE.md Quick Reference**
  - Update animate_fields examples to use `speed` and `frame_times`
  - Add examples for different use cases (replay, theta, place fields)
  - Show speed=0.1 for slow motion, speed=2.0 for fast forward
  - Success: Examples are copy-paste runnable

- [ ] **8.2 Update CLAUDE.md animation section**
  - Document speed parameter semantics
  - Explain relationship: playback_fps = sample_rate_hz * speed
  - Note 60 fps cap and how to override with `max_playback_fps`
  - Success: Full documentation of new API

- [ ] **8.3 Add migration notes to CLAUDE.md (if needed)**
  - Document breaking changes (fps removed, frame_times required)
  - Show before/after code examples
  - Success: Users can migrate existing code

---

## Verification Checklist

After all milestones complete:

- [ ] `uv run pytest` passes all tests
- [ ] `uv run ruff check .` shows no linting errors
- [ ] `uv run mypy src/neurospatial/` passes type checking
- [ ] Demo script `data/demo_spike_overlay_napari.py` runs successfully
- [ ] Napari widget shows speed multiplier slider
- [ ] Warning emitted for high sample rate data at speed=1.0
- [ ] CLAUDE.md examples are accurate and runnable

---

## Implementation Order

Recommended sequence (respecting dependencies):

1. Milestone 1 (constants, helper function)
2. Milestone 2 (overlay validation refactor)
3. Milestone 3 (core API change)
4. Milestone 4 (Environment method)
5. Milestone 5 (napari widget)
6. Milestone 6 (tests) - can run in parallel with 5
7. Milestone 7 (demo scripts)
8. Milestone 8 (documentation)
9. Verification checklist

---

## Notes

- **No backwards compatibility**: The old `fps` parameter is removed entirely
- **frame_times required**: Forces users to think about temporal structure
- **speed is intuitive**: 0.1× = slow motion, 1.0× = real-time, 2.0× = fast forward
- **60 fps cap is configurable**: Use `max_playback_fps` for high-refresh displays
