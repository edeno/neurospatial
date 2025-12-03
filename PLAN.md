# PLAN3: Napari Playback Interface Enhancement

**Goal**: Replace the FPS slider with an intuitive speed dropdown and add keyboard shortcuts for efficient neural data exploration, particularly for millisecond-scale replay analysis.

**Branch**: `feat/napari-speed-dropdown`

---

## Context

Users investigating hippocampal replay need to examine neural data at multiple timescales:

- **Replay events**: 2-4ms bins requiring slow playback (1/16x to 1/4x speed)
- **Behavior**: Real-time or faster playback (1x to 4x speed)

The current FPS slider is unintuitive—users think in terms of speed multipliers relative to real-time, not raw frame rates.

---

## Tasks

### Task 0: Add pytest-qt Dependency

**File**: `pyproject.toml`

**Location**: In `[project.optional-dependencies].dev` section (around line 49)

**Add** after the existing test dependencies:

```toml
    "pytest-qt>=4.2.0",
```

**Verify**: `uv sync --dev && uv run python -c "import pytestqt; print('pytest-qt available')"`

---

### Task 1: Add Speed Preset Constants

**File**: `src/neurospatial/animation/backends/napari_backend.py`

**Location**: After line ~110 (after `FPS_SLIDER_DEFAULT_MAX`)

**Add**:

```python
# Speed multiplier presets for dropdown (ordered slow to fast)
SPEED_PRESETS: tuple[float, ...] = (
    0.0625,  # 1/16x - frame-by-frame for high-rate data
    0.125,   # 1/8x
    0.25,    # 1/4x
    0.5,     # 1/2x
    1.0,     # 1x - real-time
    2.0,     # 2x
    4.0,     # 4x
)
"""Speed multiplier presets for playback dropdown.

Values represent multipliers relative to real-time playback:
- < 1.0: Slow motion (e.g., 0.25 = 1/4 speed)
- 1.0: Real-time (1 second of data = 1 second viewing)
- > 1.0: Fast forward (e.g., 2.0 = 2x speed)

Note: For very high sample rates (>1000 Hz), multiple slow presets may produce
identical playback FPS due to max_playback_fps capping (60 fps). For example,
at 1000 Hz both 1/16x (62.5→60) and 1/8x (125→60) would cap at 60 fps.
"""

SPEED_PRESET_LABELS: dict[float, str] = {
    0.0625: "1/16x",
    0.125: "1/8x",
    0.25: "1/4x",
    0.5: "1/2x",
    1.0: "1x",
    2.0: "2x",
    4.0: "4x",
}
"""Human-readable labels for speed presets."""
```

**Verify**: `uv run python -c "from neurospatial.animation.backends.napari_backend import SPEED_PRESETS, SPEED_PRESET_LABELS; print(SPEED_PRESETS)"`

---

### Task 2: Create Helper Functions

**File**: `src/neurospatial/animation/backends/napari_backend.py`

**Location**: Before `_add_speed_control_widget` function (around line 2300)

**Add**:

```python
def _compute_fps_from_speed(
    speed: float,
    sample_rate_hz: float,
    max_playback_fps: int = MAX_PLAYBACK_FPS,
) -> int:
    """Compute playback FPS from speed multiplier and sample rate.

    Parameters
    ----------
    speed : float
        Speed multiplier (e.g., 0.5 for half-speed, 2.0 for 2x).
        Must be positive.
    sample_rate_hz : float
        Data sample rate in Hz. Must be positive.
    max_playback_fps : int, default=MAX_PLAYBACK_FPS
        Maximum allowed playback FPS.

    Returns
    -------
    int
        Clamped playback FPS in range [1, max_playback_fps].

    Raises
    ------
    ValueError
        If speed or sample_rate_hz is not positive.

    Examples
    --------
    >>> _compute_fps_from_speed(1.0, 30.0)  # Real-time 30Hz data
    30  # Not capped (30 < 60)
    >>> _compute_fps_from_speed(0.5, 30.0)  # Half-speed 30Hz data
    15
    >>> _compute_fps_from_speed(1.0, 100.0)  # Real-time 100Hz data
    60  # Capped at MAX_PLAYBACK_FPS
    >>> _compute_fps_from_speed(0.0625, 500.0)  # 1/16x speed 500Hz data
    31  # 500 * 0.0625 = 31.25 -> rounds to 31
    """
    if speed <= 0:
        msg = f"speed must be positive, got {speed}"
        raise ValueError(msg)
    if sample_rate_hz <= 0:
        msg = f"sample_rate_hz must be positive, got {sample_rate_hz}"
        raise ValueError(msg)

    computed = sample_rate_hz * speed
    return max(1, min(round(computed), max_playback_fps))


def _format_speed_label(speed: float, fps: int) -> str:
    """Format speed dropdown label with FPS info.

    Parameters
    ----------
    speed : float
        Speed multiplier.
    fps : int
        Computed playback FPS.

    Returns
    -------
    str
        Formatted label like "1/4x (≈8 fps)" or "1x (≈25 fps)".
    """
    speed_str = SPEED_PRESET_LABELS.get(speed, f"{speed}x")
    return f"{speed_str} (≈{fps} fps)"
```

**Verify**:

```bash
uv run python -c "
from neurospatial.animation.backends.napari_backend import _compute_fps_from_speed, _format_speed_label
print(_compute_fps_from_speed(1.0, 30.0))  # Should print 30
print(_compute_fps_from_speed(0.5, 30.0))  # Should print 15
print(_format_speed_label(0.25, 8))  # Should print '1/4x (≈8 fps)'
"
```

---

### Task 3: Rewrite `_add_speed_control_widget` with Dropdown

**File**: `src/neurospatial/animation/backends/napari_backend.py`

**Replace**: The entire `_add_speed_control_widget` function (lines ~2302-2491)

**New implementation**:

```python
def _add_speed_control_widget(
    viewer: napari.Viewer,
    frame_labels: list[str] | None = None,
    *,
    initial_speed: float = 1.0,
    sample_rate_hz: float = 30.0,
    max_playback_fps: int = MAX_PLAYBACK_FPS,
) -> None:
    """Add enhanced playback control widget with speed dropdown to napari viewer.

    Creates a comprehensive docked widget with:
    - Play/Pause button (large, prominent)
    - Speed dropdown (1/16x to 4x range, shows speed multiplier and fps)
    - Frame counter ("Frame: 15 / 30")
    - Frame label (if provided: "Trial 15")

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance to add widget to.
    frame_labels : list of str, optional
        Labels for each frame (e.g., ["Trial 1", "Trial 2", ...]).
    initial_speed : float, default=1.0
        Initial playback speed multiplier (1.0 = real-time).
    sample_rate_hz : float, default=30.0
        Data sample rate in Hz. For high sample rates (>400 Hz), multiple
        slow presets may produce identical FPS due to max_playback_fps capping.
    max_playback_fps : int, default=MAX_PLAYBACK_FPS
        Maximum playback FPS (speed capped accordingly).

    Notes
    -----
    Keyboard shortcuts are bound to the viewer:

    - Space: Play/Pause
    - Left/Right arrows: Step 1 frame backward/forward
    - [ / ]: Decrease/Increase speed
    - Home/End: Jump to first/last frame
    """
    try:
        from magicgui import magicgui
        from napari.settings import get_settings
    except ImportError:
        return

    # Build speed choices with fps info
    speed_choices: list[tuple[str, float]] = []
    for speed in SPEED_PRESETS:
        fps = _compute_fps_from_speed(speed, sample_rate_hz, max_playback_fps)
        label = _format_speed_label(speed, fps)
        speed_choices.append((label, speed))

    # Find initial speed index (default to 1x if not in presets)
    initial_index = 4  # Default to 1x (index 4 in SPEED_PRESETS)
    for i, speed in enumerate(SPEED_PRESETS):
        if speed == initial_speed:
            initial_index = i
            break

    # Track playback state
    playback_state = {
        "is_playing": False,
        "speed_index": initial_index,
        "last_frame": -1,
    }

    # Compute initial fps
    initial_computed_fps = _compute_fps_from_speed(
        initial_speed, sample_rate_hz, max_playback_fps
    )

    # Throttle widget updates during playback
    update_interval = max(1, initial_computed_fps // WIDGET_UPDATE_TARGET_HZ)

    @magicgui(
        auto_call=True,
        play={"widget_type": "PushButton", "text": "▶ Play"},
        speed={
            "widget_type": "ComboBox",
            "choices": speed_choices,
            "value": SPEED_PRESETS[initial_index],
            "label": "Speed ([ ] keys)",
        },
        frame_info={"widget_type": "Label", "label": ""},
    )
    def playback_widget(
        play: bool = False,
        speed: float = 1.0,
        frame_info: str = "",
    ) -> None:
        """Enhanced playback control widget with speed dropdown."""
        # Update FPS setting when speed changes
        settings = get_settings()
        new_fps = _compute_fps_from_speed(speed, sample_rate_hz, max_playback_fps)
        settings.application.playback_fps = new_fps

        # Update throttle interval
        nonlocal update_interval
        update_interval = max(1, new_fps // WIDGET_UPDATE_TARGET_HZ)

    # Make dropdown wider for readability
    with contextlib.suppress(Exception):
        playback_widget.speed.native.setMinimumWidth(200)

    # --- Sync speed_index when dropdown changes ---
    def on_speed_changed(new_speed: float) -> None:
        """Sync speed_index state when dropdown changes."""
        for i, speed in enumerate(SPEED_PRESETS):
            if speed == new_speed:
                playback_state["speed_index"] = i
                break

    playback_widget.speed.changed.connect(on_speed_changed)

    # --- Play/Pause toggle ---
    def toggle_playback(event: Any | None = None) -> None:
        """Toggle animation playback."""
        playback_state["is_playing"] = not playback_state["is_playing"]
        if playback_state["is_playing"]:
            playback_widget.play.text = "⏸ Pause"
        else:
            playback_widget.play.text = "▶ Play"
        viewer.window._toggle_play()

    playback_widget.play.changed.connect(toggle_playback)

    # --- Frame info updates ---
    def update_frame_info(event: Any | None = None) -> None:
        """Update frame counter (throttled during playback)."""
        try:
            current_frame = viewer.dims.current_step[0] if viewer.dims.ndim > 0 else 0

            # Throttle during playback only
            if (
                playback_state["is_playing"]
                and current_frame % update_interval != 0
                and current_frame != playback_state["last_frame"]
            ):
                playback_state["last_frame"] = current_frame
                return

            playback_state["last_frame"] = current_frame

            total_frames = (
                viewer.dims.range[0][2]
                if viewer.dims.ndim > 0 and viewer.dims.range
                else 0
            )

            frame_text = f"Frame: {current_frame + 1} / {int(total_frames)}"
            if frame_labels and 0 <= current_frame < len(frame_labels):
                frame_text += f" ({frame_labels[current_frame]})"

            playback_widget.frame_info.value = frame_text
        except Exception:
            playback_widget.frame_info.value = "Frame: -- / --"

    viewer.dims.events.current_step.connect(update_frame_info)
    update_frame_info()

    # --- Keyboard shortcuts ---
    @viewer.bind_key("Space")
    def _spacebar_toggle(viewer_instance: napari.Viewer) -> None:
        """Toggle playback with spacebar."""
        toggle_playback()

    @viewer.bind_key("Left")
    def _step_backward(viewer_instance: napari.Viewer) -> None:
        """Step one frame backward."""
        current = viewer_instance.dims.current_step[0]
        if current > 0:
            viewer_instance.dims.set_current_step(0, current - 1)

    @viewer.bind_key("Right")
    def _step_forward(viewer_instance: napari.Viewer) -> None:
        """Step one frame forward."""
        current = viewer_instance.dims.current_step[0]
        max_frame = int(viewer_instance.dims.range[0][2]) - 1
        if current < max_frame:
            viewer_instance.dims.set_current_step(0, current + 1)

    @viewer.bind_key("[")
    def _speed_slower(viewer_instance: napari.Viewer) -> None:
        """Decrease playback speed."""
        idx = playback_state["speed_index"]
        if idx > 0:
            playback_state["speed_index"] = idx - 1
            playback_widget.speed.value = SPEED_PRESETS[idx - 1]
        else:
            viewer_instance.status = "Already at slowest speed (1/16x)"

    @viewer.bind_key("]")
    def _speed_faster(viewer_instance: napari.Viewer) -> None:
        """Increase playback speed."""
        idx = playback_state["speed_index"]
        if idx < len(SPEED_PRESETS) - 1:
            playback_state["speed_index"] = idx + 1
            playback_widget.speed.value = SPEED_PRESETS[idx + 1]
        else:
            viewer_instance.status = "Already at fastest speed (4x)"

    @viewer.bind_key("Home")
    def _jump_start(viewer_instance: napari.Viewer) -> None:
        """Jump to first frame."""
        viewer_instance.dims.set_current_step(0, 0)

    @viewer.bind_key("End")
    def _jump_end(viewer_instance: napari.Viewer) -> None:
        """Jump to last frame."""
        max_frame = int(viewer_instance.dims.range[0][2]) - 1
        viewer_instance.dims.set_current_step(0, max_frame)

    # Add dock widget
    with contextlib.suppress(Exception):
        viewer.window.add_dock_widget(
            playback_widget,
            name="Playback Controls",
            area="left",
        )
```

**Verify**: Run existing napari animation test

```bash
uv run pytest tests/test_animation.py -k "napari" -v --timeout=30
```

---

### Task 4: Update `render_napari` Docstring

**File**: `src/neurospatial/animation/backends/napari_backend.py`

**Location**: In `render_napari` docstring, find the "Enhanced Playback Controls" section (around line 2902)

**Replace** the keyboard shortcuts section:

```python
    - **Keyboard shortcuts**:
        - **Spacebar** - Play/pause animation (toggle)
        - **← →** Arrow keys - Step forward/backward through frames
```

**With**:

```python
    - **Keyboard shortcuts**:
        - **Space** - Play/pause animation
        - **← / →** - Step one frame backward/forward
        - **[ / ]** - Decrease/increase playback speed
        - **Home / End** - Jump to first/last frame
```

**Verify**: `uv run python -c "from neurospatial.animation.backends.napari_backend import render_napari; help(render_napari)" | head -100`

---

### Task 5: Add Unit Tests for Speed Helpers

**File**: `tests/test_napari_speed_controls.py` (new file)

**Content**:

```python
"""Tests for napari speed control helpers."""

import pytest

from neurospatial.animation.backends.napari_backend import (
    MAX_PLAYBACK_FPS,
    SPEED_PRESET_LABELS,
    SPEED_PRESETS,
    _compute_fps_from_speed,
    _format_speed_label,
)


class TestSpeedPresets:
    """Tests for speed preset constants."""

    def test_presets_ordered_ascending(self):
        """Speed presets should be in ascending order."""
        assert list(SPEED_PRESETS) == sorted(SPEED_PRESETS)

    def test_presets_all_positive(self):
        """All speed presets should be positive."""
        assert all(s > 0 for s in SPEED_PRESETS)

    def test_presets_include_realtime(self):
        """Speed presets should include 1.0 (real-time)."""
        assert 1.0 in SPEED_PRESETS

    def test_labels_match_presets(self):
        """All presets should have labels."""
        for speed in SPEED_PRESETS:
            assert speed in SPEED_PRESET_LABELS


class TestComputeFpsFromSpeed:
    """Tests for _compute_fps_from_speed function."""

    def test_realtime_30hz(self):
        """Real-time 30Hz data should give 30 fps (not capped)."""
        assert _compute_fps_from_speed(1.0, 30.0) == 30

    def test_half_speed_30hz(self):
        """Half-speed 30Hz data should give 15 fps."""
        assert _compute_fps_from_speed(0.5, 30.0) == 15

    def test_quarter_speed_30hz(self):
        """Quarter-speed 30Hz data should give 8 fps (rounded)."""
        # 30 * 0.25 = 7.5 -> rounds to 8
        assert _compute_fps_from_speed(0.25, 30.0) == 8

    def test_double_speed_capped(self):
        """2x speed at 100Hz should be capped at max_playback_fps (60)."""
        # 100 * 2.0 = 200 -> capped to 60
        assert _compute_fps_from_speed(2.0, 100.0) == MAX_PLAYBACK_FPS

    def test_high_rate_data_not_capped(self):
        """500Hz data at 1/16x should give 31 fps (not capped)."""
        # 500 * 0.0625 = 31.25 -> rounds to 31 (< 60, not capped)
        assert _compute_fps_from_speed(0.0625, 500.0) == 31

    def test_high_rate_realtime_capped(self):
        """500Hz data at 1x should be capped at 60 fps."""
        # 500 * 1.0 = 500 -> capped to 60
        assert _compute_fps_from_speed(1.0, 500.0) == MAX_PLAYBACK_FPS

    def test_minimum_fps_is_one(self):
        """Very slow speed should give minimum 1 fps."""
        assert _compute_fps_from_speed(0.001, 10.0) == 1

    def test_custom_max_fps(self):
        """Custom max_playback_fps should be respected."""
        assert _compute_fps_from_speed(1.0, 100.0, max_playback_fps=50) == 50

    def test_zero_speed_raises(self):
        """Zero speed should raise ValueError."""
        with pytest.raises(ValueError, match="speed must be positive"):
            _compute_fps_from_speed(0.0, 30.0)

    def test_negative_speed_raises(self):
        """Negative speed should raise ValueError."""
        with pytest.raises(ValueError, match="speed must be positive"):
            _compute_fps_from_speed(-1.0, 30.0)

    def test_zero_sample_rate_raises(self):
        """Zero sample rate should raise ValueError."""
        with pytest.raises(ValueError, match="sample_rate_hz must be positive"):
            _compute_fps_from_speed(1.0, 0.0)

    def test_negative_sample_rate_raises(self):
        """Negative sample rate should raise ValueError."""
        with pytest.raises(ValueError, match="sample_rate_hz must be positive"):
            _compute_fps_from_speed(1.0, -30.0)


class TestFormatSpeedLabel:
    """Tests for _format_speed_label function."""

    def test_known_preset_format(self):
        """Known presets should use fractional notation."""
        assert _format_speed_label(0.25, 8) == "1/4x (≈8 fps)"
        assert _format_speed_label(1.0, 25) == "1x (≈25 fps)"
        assert _format_speed_label(2.0, 25) == "2x (≈25 fps)"

    def test_unknown_speed_format(self):
        """Unknown speeds should use decimal notation."""
        assert _format_speed_label(0.33, 10) == "0.33x (≈10 fps)"
```

**Verify**: `uv run pytest tests/test_napari_speed_controls.py -v`

---

### Task 6: Add Integration Tests

**File**: `tests/test_napari_speed_controls.py` (append to file)

**Append**:

```python
@pytest.mark.slow
class TestSpeedControlIntegration:
    """Integration tests requiring napari viewer (marked slow)."""

    @pytest.fixture
    def sample_env_and_fields(self):
        """Create sample environment and fields for testing."""
        import numpy as np
        from neurospatial import Environment

        positions = np.random.rand(50, 2) * 100
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = np.random.rand(20, env.n_bins)
        return env, fields

    def test_speed_dropdown_changes_fps(self, sample_env_and_fields, qtbot):
        """Changing speed dropdown should update napari playback fps."""
        pytest.importorskip("napari")
        from napari.settings import get_settings

        from neurospatial.animation.backends.napari_backend import render_napari

        env, fields = sample_env_and_fields

        viewer = render_napari(
            env,
            fields,
            speed=1.0,
            sample_rate_hz=30.0,
        )
        qtbot.addWidget(viewer.window._qt_window)

        try:
            settings = get_settings()
            initial_fps = settings.application.playback_fps

            # At 1x speed with 30Hz data, expect 30 fps (not capped, 30 < 60)
            assert initial_fps == 30
        finally:
            viewer.close()

    def test_keyboard_shortcuts_bound(self, sample_env_and_fields, qtbot):
        """Keyboard shortcuts should be bound to viewer."""
        pytest.importorskip("napari")

        from neurospatial.animation.backends.napari_backend import render_napari

        env, fields = sample_env_and_fields

        viewer = render_napari(env, fields)
        qtbot.addWidget(viewer.window._qt_window)

        try:
            # Check that our shortcuts are bound
            # Note: viewer.keymap.keys() returns KeyBinding objects, convert to strings
            bound_keys = {str(k) for k in viewer.keymap.keys()}
            expected_keys = {"Space", "Left", "Right", "[", "]", "Home", "End"}
            assert expected_keys.issubset(bound_keys)
        finally:
            viewer.close()

    def test_arrow_keys_respect_boundaries(self, sample_env_and_fields, qtbot):
        """Arrow keys should respect frame boundaries."""
        pytest.importorskip("napari")

        from neurospatial.animation.backends.napari_backend import render_napari

        env, fields = sample_env_and_fields

        viewer = render_napari(env, fields)
        qtbot.addWidget(viewer.window._qt_window)

        try:
            # Start at frame 0
            assert viewer.dims.current_step[0] == 0

            # Left at frame 0 should stay at 0 (boundary check)
            current = viewer.dims.current_step[0]
            if current > 0:
                viewer.dims.set_current_step(0, current - 1)
            assert viewer.dims.current_step[0] == 0

            # Step forward works
            viewer.dims.set_current_step(0, 5)
            assert viewer.dims.current_step[0] == 5
        finally:
            viewer.close()

    def test_home_end_jump_to_boundaries(self, sample_env_and_fields, qtbot):
        """Home/End should jump to first/last frame."""
        pytest.importorskip("napari")

        from neurospatial.animation.backends.napari_backend import render_napari

        env, fields = sample_env_and_fields

        viewer = render_napari(env, fields)
        qtbot.addWidget(viewer.window._qt_window)

        try:
            n_frames = fields.shape[0]

            # Jump to middle
            viewer.dims.set_current_step(0, 10)
            assert viewer.dims.current_step[0] == 10

            # Home jumps to start
            viewer.dims.set_current_step(0, 0)
            assert viewer.dims.current_step[0] == 0

            # End jumps to last frame
            max_frame = int(viewer.dims.range[0][2]) - 1
            viewer.dims.set_current_step(0, max_frame)
            assert viewer.dims.current_step[0] == n_frames - 1
        finally:
            viewer.close()

```

**Verify**: `uv run pytest tests/test_napari_speed_controls.py -v --timeout=60`

---

## Verification Checklist

After all tasks complete:

1. **Unit tests pass**:

   ```bash
   uv run pytest tests/test_napari_speed_controls.py -v
   ```

2. **Existing napari tests pass**:

   ```bash
   uv run pytest tests/test_animation.py -k "napari" -v --timeout=60
   ```

3. **Type checking passes**:

   ```bash
   uv run mypy src/neurospatial/animation/backends/napari_backend.py
   ```

4. **Linting passes**:

   ```bash
   uv run ruff check src/neurospatial/animation/backends/napari_backend.py
   ```

5. **Manual verification** (interactive):

   ```python
   import numpy as np
   from neurospatial import Environment

   # Create test data
   positions = np.random.rand(100, 2) * 100
   env = Environment.from_samples(positions, bin_size=5.0)
   fields = np.random.rand(100, env.n_bins)
   frame_times = np.arange(100) / 30.0  # 30 Hz

   # Launch viewer
   env.animate_fields(fields, frame_times=frame_times, backend="napari")

   # Verify:
   # - [ ] Speed dropdown appears with options 1/16x to 4x
   # - [ ] Dropdown label shows "Speed ([ ] keys)" for discoverability
   # - [ ] Changing dropdown updates playback speed
   # - [ ] Space toggles play/pause
   # - [ ] Left/Right arrows step frames
   # - [ ] [ and ] change speed (dropdown updates)
   # - [ ] Pressing [ at 1/16x shows status "Already at slowest speed"
   # - [ ] Pressing ] at 4x shows status "Already at fastest speed"
   # - [ ] Home/End jump to start/end
   # - [ ] Playback stops at last frame (no loop)
   # - [ ] Pause at frame 50, press Space -> resumes at 51
   ```

6. **High sample rate verification** (500 Hz data):

   ```python
   import numpy as np
   from neurospatial import Environment

   # Simulate 500 Hz replay data
   positions = np.random.rand(100, 2) * 100
   env = Environment.from_samples(positions, bin_size=5.0)
   fields = np.random.rand(1000, env.n_bins)  # 2 seconds at 500 Hz
   frame_times = np.arange(1000) / 500.0

   env.animate_fields(fields, frame_times=frame_times, backend="napari")

   # Verify:
   # - [ ] 1/16x shows "1/16x (≈31 fps)" - 500 * 0.0625 = 31.25
   # - [ ] 1x shows "1x (≈60 fps)" - capped at MAX_PLAYBACK_FPS
   # - [ ] Playback is smooth (no Qt event loop stalls from ComboBox)
   # - [ ] Changing speed via dropdown or [ ] keys works smoothly
   ```

   **Important**: The previous FPS slider was replaced due to Qt event loop
   issues with FloatSlider at high sample rates. The ComboBox (dropdown)
   with discrete values should not have this problem, but verify playback
   doesn't stall when changing speeds.

---

## Rollback

If issues arise, revert to the FPS slider implementation:

```bash
git checkout HEAD~1 -- src/neurospatial/animation/backends/napari_backend.py
```
