# TASKS.md - Napari Playback Interface Enhancement

**Goal**: Replace FPS slider with speed dropdown and add keyboard shortcuts for neural data exploration.

**Branch**: `feat/napari-speed-dropdown`

**Source**: [PLAN.md](PLAN.md)

---

## Milestone 1: Setup & Dependencies

### Task 0: Add pytest-qt Dependency

- [ ] **Add dependency to pyproject.toml**
  - File: `pyproject.toml`
  - Location: `[project.optional-dependencies].dev` section (~line 49)
  - Add: `"pytest-qt>=4.2.0",`

- [ ] **Verify installation**
  - Run: `uv sync --dev && uv run python -c "import pytestqt; print('pytest-qt available')"`
  - Success: Output shows "pytest-qt available"

---

## Milestone 2: Core Implementation

### Task 1: Add Speed Preset Constants

- [ ] **Add SPEED_PRESETS tuple**
  - File: `src/neurospatial/animation/backends/napari_backend.py`
  - Location: After `FPS_SLIDER_DEFAULT_MAX` (~line 110)
  - Values: `(0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)`
  - Include docstring explaining multiplier semantics and FPS capping behavior

- [ ] **Add SPEED_PRESET_LABELS dict**
  - Maps float speed values to human-readable labels
  - Example: `{0.0625: "1/16x", 0.125: "1/8x", ...}`

- [ ] **Verify constants import**
  - Run: `uv run python -c "from neurospatial.animation.backends.napari_backend import SPEED_PRESETS, SPEED_PRESET_LABELS; print(SPEED_PRESETS)"`
  - Success: Prints the tuple of speed values

---

### Task 2: Create Helper Functions

- [ ] **Implement `_compute_fps_from_speed()`**
  - File: `src/neurospatial/animation/backends/napari_backend.py`
  - Location: Before `_add_speed_control_widget` (~line 2300)
  - Parameters: `speed`, `sample_rate_hz`, `max_playback_fps`
  - Returns: Clamped FPS in range [1, max_playback_fps]
  - Must raise ValueError for non-positive inputs

- [ ] **Implement `_format_speed_label()`**
  - Parameters: `speed`, `fps`
  - Returns: Formatted string like "1/4x (≈8 fps)"
  - Uses SPEED_PRESET_LABELS for known presets, decimal notation otherwise

- [ ] **Verify helper functions**
  - Run verification script from PLAN.md
  - Success: `_compute_fps_from_speed(1.0, 30.0)` returns 30
  - Success: `_format_speed_label(0.25, 8)` returns "1/4x (≈8 fps)"

---

### Task 3: Rewrite Speed Control Widget

**Dependencies**: Tasks 1, 2 must be complete

- [ ] **Replace `_add_speed_control_widget` function**
  - File: `src/neurospatial/animation/backends/napari_backend.py`
  - Replace lines ~2302-2491 with new implementation
  - New parameters: `initial_speed`, `sample_rate_hz`, `max_playback_fps`

- [ ] **Implement speed dropdown using magicgui ComboBox**
  - Build choices dynamically from SPEED_PRESETS
  - Include FPS info in labels: "1/4x (≈8 fps)"
  - Set minimum width to 200px for readability

- [ ] **Implement playback state tracking**
  - Track: `is_playing`, `speed_index`, `last_frame`
  - Sync speed_index when dropdown changes

- [ ] **Implement Play/Pause toggle**
  - Button text toggles between "▶ Play" and "⏸ Pause"
  - Calls `viewer.window._toggle_play()`

- [ ] **Implement frame info updates**
  - Display: "Frame: 15 / 30" format
  - Include frame labels if provided: "Frame: 15 / 30 (Trial 15)"
  - Throttle updates during playback

- [ ] **Implement keyboard shortcuts**
  - Space: Toggle play/pause
  - Left/Right: Step one frame
  - [ / ]: Decrease/increase speed (update dropdown)
  - Home/End: Jump to first/last frame
  - Show status message at speed boundaries

- [ ] **Add dock widget to viewer**
  - Name: "Playback Controls"
  - Area: "left"

- [ ] **Verify with existing napari tests**
  - Run: `uv run pytest tests/test_animation.py -k "napari" -v --timeout=30`
  - Success: All tests pass

---

### Task 4: Update Docstrings

- [ ] **Update `render_napari` docstring**
  - File: `src/neurospatial/animation/backends/napari_backend.py`
  - Location: "Enhanced Playback Controls" section (~line 2902)
  - Replace keyboard shortcuts documentation with new shortcuts

- [ ] **Verify docstring update**
  - Run: `uv run python -c "from neurospatial.animation.backends.napari_backend import render_napari; help(render_napari)" | head -100`
  - Success: New keyboard shortcuts appear in help output

---

## Milestone 3: Testing

### Task 5: Add Unit Tests for Speed Helpers

**Dependencies**: Tasks 1, 2 must be complete

- [ ] **Create test file**
  - File: `tests/test_napari_speed_controls.py` (new)

- [ ] **Implement `TestSpeedPresets` class**
  - `test_presets_ordered_ascending`: Verify ascending order
  - `test_presets_all_positive`: Verify all positive values
  - `test_presets_include_realtime`: Verify 1.0 exists
  - `test_labels_match_presets`: Verify all presets have labels

- [ ] **Implement `TestComputeFpsFromSpeed` class**
  - `test_realtime_30hz`: 1.0 speed, 30Hz -> 30 fps
  - `test_half_speed_30hz`: 0.5 speed, 30Hz -> 15 fps
  - `test_quarter_speed_30hz`: 0.25 speed, 30Hz -> 8 fps (rounded)
  - `test_double_speed_capped`: 2.0 speed, 100Hz -> 60 fps (capped)
  - `test_high_rate_data_not_capped`: 0.0625 speed, 500Hz -> 31 fps
  - `test_high_rate_realtime_capped`: 1.0 speed, 500Hz -> 60 fps (capped)
  - `test_minimum_fps_is_one`: Very slow speed -> 1 fps
  - `test_custom_max_fps`: Custom max_playback_fps respected
  - `test_zero_speed_raises`: ValueError for speed=0
  - `test_negative_speed_raises`: ValueError for negative speed
  - `test_zero_sample_rate_raises`: ValueError for sample_rate=0
  - `test_negative_sample_rate_raises`: ValueError for negative sample_rate

- [ ] **Implement `TestFormatSpeedLabel` class**
  - `test_known_preset_format`: Known presets use fractional notation
  - `test_unknown_speed_format`: Unknown speeds use decimal notation

- [ ] **Verify unit tests pass**
  - Run: `uv run pytest tests/test_napari_speed_controls.py::TestSpeedPresets tests/test_napari_speed_controls.py::TestComputeFpsFromSpeed tests/test_napari_speed_controls.py::TestFormatSpeedLabel -v`
  - Success: All unit tests pass

---

### Task 6: Add Integration Tests

**Dependencies**: Tasks 0-4 must be complete

- [ ] **Add `sample_env_and_fields` fixture**
  - Creates Environment with random positions
  - Creates random fields array (20 frames)

- [ ] **Implement `TestSpeedControlIntegration` class** (marked `@pytest.mark.slow`)
  - `test_speed_dropdown_changes_fps`: Verify fps setting updates
  - `test_keyboard_shortcuts_bound`: Verify all shortcuts registered
  - `test_arrow_keys_respect_boundaries`: Verify boundary checks
  - `test_home_end_jump_to_boundaries`: Verify Home/End navigation

- [ ] **Verify integration tests pass**
  - Run: `uv run pytest tests/test_napari_speed_controls.py -v --timeout=60`
  - Success: All tests pass (may skip on headless CI)

---

## Milestone 4: Verification & Quality

### Task 7: Run Full Test Suite

- [ ] **Unit tests pass**
  - Run: `uv run pytest tests/test_napari_speed_controls.py -v`

- [ ] **Existing napari tests pass**
  - Run: `uv run pytest tests/test_animation.py -k "napari" -v --timeout=60`

- [ ] **Type checking passes**
  - Run: `uv run mypy src/neurospatial/animation/backends/napari_backend.py`
  - Success: No type errors

- [ ] **Linting passes**
  - Run: `uv run ruff check src/neurospatial/animation/backends/napari_backend.py && uv run ruff format --check src/neurospatial/animation/backends/napari_backend.py`
  - Success: No linting errors

---

### Task 8: Manual Verification (Optional - Interactive)

- [ ] **Standard 30Hz data test**
  - Create test environment with 100 frames at 30Hz
  - Verify: Speed dropdown appears with 1/16x to 4x options
  - Verify: Dropdown label shows "Speed ([ ] keys)"
  - Verify: Changing dropdown updates playback speed
  - Verify: Space toggles play/pause
  - Verify: Left/Right arrows step frames
  - Verify: [ and ] change speed (dropdown updates)
  - Verify: Boundary status messages at 1/16x and 4x
  - Verify: Home/End jump to start/end
  - Verify: Playback stops at last frame (no loop)
  - Verify: Pause at frame 50, Space -> resumes at 51

- [ ] **High sample rate test (500 Hz)**
  - Create test data: 1000 frames at 500Hz (2 seconds)
  - Verify: 1/16x shows "1/16x (≈31 fps)"
  - Verify: 1x shows "1x (≈60 fps)" (capped)
  - Verify: Playback is smooth (no Qt event loop stalls)
  - Verify: Speed changes via dropdown and keyboard work smoothly

---

## Summary

| Milestone | Tasks | Description |
|-----------|-------|-------------|
| 1 | 0 | Setup pytest-qt dependency |
| 2 | 1-4 | Core implementation (constants, helpers, widget, docstrings) |
| 3 | 5-6 | Unit and integration tests |
| 4 | 7-8 | Verification and quality checks |

**Estimated Total**: 8 tasks across 4 milestones

**Rollback**: If issues arise, run `git checkout HEAD~1 -- src/neurospatial/animation/backends/napari_backend.py`
