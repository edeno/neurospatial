# Plan: Separate Data Sample Rate from Playback Speed

## Problem Statement

Currently, the animation API conflates two distinct concepts:

1. **Data sample rate**: How frequently the data was recorded (e.g., 500 Hz for replay decoding, 30 Hz for position tracking)
2. **Playback speed**: How fast the viewer displays frames

The current API uses a single `fps` parameter that callers often misuse to control both data resolution AND playback speed (as seen in the demo script).

### Use Cases

| Analysis Type | Data Sample Rate | Desired Viewing | Notes |
|---------------|------------------|-----------------|-------|
| Replay decoding | 500 Hz | Slow motion (see trajectory unfold) | All frames preserved |
| Theta sequences | ~30 Hz | Real-time | Natural dynamics |
| Place fields | ~30 Hz | Real-time or 2x | Quick preview |

### Key Insight

The **playback fps should be derived from the data's temporal structure**, not specified independently. If you have 500 Hz data and want to view at 10% speed, the system should compute `playback_fps = 500 * 0.1 = 50 fps`. If that exceeds display limits, cap it.

---

## Proposed API Change

### Current API (to be replaced)

```python
def animate_fields(
    fields,
    fps: int = 30,                    # Ambiguous: data rate or playback?
    frame_times: NDArray | None = None,  # Optional
    ...
)
```

### New API

```python
def animate_fields(
    fields,
    frame_times: NDArray[np.float64],  # REQUIRED - defines temporal structure
    speed: float = 1.0,                 # 1.0 = real-time, 0.1 = 10% speed
    ...
)
```

### Semantics

```python
# Infer sample rate from timestamps
duration = frame_times[-1] - frame_times[0]
sample_rate_hz = (len(frame_times) - 1) / duration

# Compute playback fps, capped to reasonable display rate
MAX_PLAYBACK_FPS = 60
requested_fps = sample_rate_hz * speed
playback_fps = min(requested_fps, MAX_PLAYBACK_FPS)

# Warn if capping affects requested speed
if playback_fps < requested_fps:
    actual_speed = playback_fps / sample_rate_hz
    warnings.warn(
        f"Requested speed={speed}x would require {requested_fps:.0f} fps. "
        f"Capped to {MAX_PLAYBACK_FPS} fps (effective speed={actual_speed:.2f}x)."
    )
```

### Usage Examples

```python
# Replay: 500 Hz data, view at 10% speed
# playback_fps = 500 * 0.1 = 50 fps (within limit)
env.animate_fields(fields, frame_times=decode_times, speed=0.1)

# Theta: 30 Hz data, view at real-time
# playback_fps = 30 * 1.0 = 30 fps
env.animate_fields(fields, frame_times=position_times, speed=1.0)

# Replay at real-time (not practical, but requested)
# playback_fps = min(500 * 1.0, 60) = 60 fps
# Warning: "effective speed=0.12x"
env.animate_fields(fields, frame_times=decode_times, speed=1.0)

# Quick preview: 2x speed
# playback_fps = min(30 * 2.0, 60) = 60 fps
env.animate_fields(fields, frame_times=position_times, speed=2.0)
```

---

## Implementation Plan

### Phase 1: Core Module Changes

#### 1.1 Add constants to `animation/core.py`

```python
# Playback speed limits
MAX_PLAYBACK_FPS: int = 60  # Display refresh rate limit
MIN_PLAYBACK_FPS: int = 1   # Minimum usable playback
DEFAULT_SPEED: float = 1.0  # Real-time by default
```

#### 1.2 Add helper function to `animation/core.py`

```python
def _compute_playback_fps(
    frame_times: NDArray[np.float64],
    speed: float,
    max_fps: int = MAX_PLAYBACK_FPS,
) -> tuple[int, float]:
    """Compute playback fps from frame timestamps and speed multiplier.

    Parameters
    ----------
    frame_times : ndarray of shape (n_frames,)
        Timestamps for each frame in seconds.
    speed : float
        Playback speed multiplier (1.0 = real-time).
    max_fps : int
        Maximum allowed playback fps.

    Returns
    -------
    playback_fps : int
        Computed playback fps (capped to max_fps).
    actual_speed : float
        Actual speed after capping (may differ from requested).
    """
    if len(frame_times) < 2:
        return max_fps, speed

    duration = frame_times[-1] - frame_times[0]
    if duration <= 0:
        return max_fps, speed

    sample_rate_hz = (len(frame_times) - 1) / duration
    requested_fps = sample_rate_hz * speed
    playback_fps = int(min(max(requested_fps, MIN_PLAYBACK_FPS), max_fps))
    actual_speed = playback_fps / sample_rate_hz

    return playback_fps, actual_speed
```

#### 1.3 Update `animate_fields()` in `animation/core.py`

**Changes:**

- Remove `fps` from `**kwargs` extraction (line 268)
- Add `speed: float = 1.0` parameter
- Add `max_playback_fps: int = 60` parameter (advanced knob)
- Make `frame_times` **strictly required** (ValueError if None)
- Compute `playback_fps` using helper
- Emit warning if speed was capped
- Pass computed `playback_fps` to backends via kwargs

```python
def animate_fields(
    env: EnvironmentProtocol,
    fields: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    *,
    backend: Literal["auto", "napari", "video", "html", "widget"] = "auto",
    save_path: str | None = None,
    overlays: list[OverlayProtocol] | None = None,
    frame_times: NDArray[np.float64],  # REQUIRED - no default
    speed: float = 1.0,  # NEW: replaces fps in public API
    max_playback_fps: int = 60,  # Advanced knob with strong default
    show_regions: bool | list[str] = False,
    region_alpha: float = 0.3,
    scale_bar: bool | Any = False,
    **kwargs: Any,
) -> Any:
```

**Implementation logic:**

```python
# frame_times is required - no fallback
# Validation happens via type system (no default) and length check
if len(frame_times) != n_frames:
    raise ValueError(
        f"frame_times length ({len(frame_times)}) must match number of fields ({n_frames}). "
        f"Provide timestamps from your data source."
    )

# Compute playback fps
playback_fps, actual_speed = _compute_playback_fps(
    frame_times, speed, max_fps=max_playback_fps
)

if abs(actual_speed - speed) > 0.01:  # Significant difference
    sample_rate = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
    warnings.warn(
        f"Requested speed={speed:.2f}x would require {sample_rate * speed:.0f} fps. "
        f"Capped to {max_playback_fps} fps (effective speed={actual_speed:.2f}x).",
        UserWarning,
        stacklevel=2,
    )

# Compute sample_rate_hz for backends that need it (napari widget)
duration = frame_times[-1] - frame_times[0]
if duration > 0 and len(frame_times) > 1:
    sample_rate_hz = (len(frame_times) - 1) / duration
else:
    sample_rate_hz = 30.0  # Fallback for edge cases

# Pass to backends
kwargs['fps'] = playback_fps  # Backends still use fps internally
kwargs['sample_rate_hz'] = sample_rate_hz  # For napari interactive widget
kwargs['speed'] = speed  # For napari interactive widget
kwargs['max_playback_fps'] = max_playback_fps  # For napari interactive widget
```

#### 1.4 Update `_build_frame_times()` in `animation/overlays.py`

**Current behavior:** Builds frame_times from `fps` if not provided, or validates if provided.

**New behavior:** Simply validate the provided `frame_times`. No synthesis from fps.

**Changes:**

- Remove `fps` parameter entirely
- Remove synthesis logic (the `np.linspace` branch)
- Keep validation logic (length check, monotonicity check)
- Rename to `_validate_frame_times()` for clarity

```python
def _validate_frame_times(
    frame_times: NDArray[np.float64],
    n_frames: int,
) -> NDArray[np.float64]:
    """Validate frame times array for animation.

    Parameters
    ----------
    frame_times : NDArray[np.float64]
        Frame times array with shape (n_frames,). Must be monotonically increasing.
    n_frames : int
        Expected number of frames.

    Returns
    -------
    NDArray[np.float64]
        Validated frame times array.

    Raises
    ------
    ValueError
        If frame_times length does not match n_frames.
    ValueError
        If frame_times is not monotonically increasing.
    """
    if len(frame_times) != n_frames:
        raise ValueError(
            f"frame_times length ({len(frame_times)}) must match n_frames ({n_frames})"
        )

    # Check monotonicity
    if len(frame_times) > 1:
        diffs = np.diff(frame_times)
        if not np.all(diffs > 0):
            raise ValueError(
                "frame_times must be strictly monotonically increasing. "
                f"Found {np.sum(diffs <= 0)} non-increasing intervals."
            )

    return frame_times
```

**Update call sites in `animation/core.py`:**

```python
# Old (line 269-271):
frame_times = _build_frame_times(
    n_frames=n_frames, fps=fps, frame_times=frame_times
)

# New:
frame_times = _validate_frame_times(frame_times, n_frames)
```

### Phase 2: Environment Method Update

#### 2.1 Update `Environment.animate_fields()` in `environment/visualization.py`

**Changes:**

- Replace `fps: int = 30` with `speed: float = 1.0`
- Update docstring
- Pass through to core `animate_fields()`

```python
def animate_fields(
    self: SelfEnv,
    fields: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    *,
    backend: Literal["auto", "napari", "video", "html", "widget"] = "auto",
    save_path: str | None = None,
    speed: float = 1.0,  # CHANGED from fps
    cmap: str = "viridis",
    ...
    frame_times: NDArray[np.float64],  # CHANGED: now required (was Optional)
    ...
) -> Any:
    """...

    Parameters
    ----------
    ...
    speed : float, default=1.0
        Playback speed relative to real-time:
        - 1.0: Real-time playback (1 second of data = 1 second viewing)
        - 0.1: 10% speed (slow motion, good for replay analysis)
        - 2.0: 2x speed (fast forward)

        The actual playback fps is computed as:
        ``playback_fps = sample_rate_hz * speed``

        where sample_rate_hz is inferred from frame_times.
        Playback is capped at 60 fps for display compatibility.
    frame_times : NDArray[np.float64], shape (n_frames,)
        Timestamps for each frame in seconds. **Required** - provides the
        temporal structure of your data. Use timestamps from your data source
        (e.g., position timestamps, decoding time bins).
    ...
    """
```

### Phase 3: Backend Updates (Internal Only)

Most backends (`render_video`, `render_html`, `render_widget`) continue to accept `fps` as an internal parameter. **No changes needed to their signatures** - they just receive the computed `fps` value from the dispatcher.

**Exception: Napari backend** requires additional changes for the interactive speed control widget (see Phase 4). The napari widget needs `sample_rate_hz` to convert between speed multiplier and fps in real-time as the user adjusts the slider.

The backends are internal implementation details; the public API change is only at the `animate_fields()` level.

### Phase 4: Napari Widget Enhancement

#### 4.1 Update speed control widget in `napari_backend.py`

**Current:** Shows "Speed (FPS)" slider with raw fps values.

**New:** Show speed multiplier as primary control, fps as secondary info.

**UI Design:**

- Primary label: "Speed: 0.25×" (matches scientist mental model)
- Secondary info: "≈ 12 fps" (in smaller text or tooltip for debugging)
- Slider controls speed multiplier, not raw fps
- Internally still drives napari with fps

**Changes to `_add_speed_control_widget()`:**

```python
# New parameters needed from caller
initial_speed: float = 1.0
sample_rate_hz: float = 30.0  # Inferred from frame_times

# Widget shows speed, computes fps internally
@magicgui(
    auto_call=True,
    play={"widget_type": "PushButton", "text": "▶ Play"},
    speed={
        "widget_type": "FloatSlider",
        "min": 0.01,
        "max": 4.0,
        "step": 0.01,
        "value": initial_speed,
        "label": "Speed",
    },
    speed_info={"widget_type": "Label", "label": ""},
)
def playback_widget(
    play: bool = False,
    speed: float = initial_speed,
    speed_info: str = "",
) -> None:
    # Compute fps from speed
    fps = int(min(sample_rate_hz * speed, max_playback_fps))
    actual_speed = fps / sample_rate_hz

    # Update napari playback
    settings = get_settings()
    settings.application.playback_fps = fps

    # Update info label
    playback_widget.speed_info.value = f"{actual_speed:.2f}× (≈{fps} fps)"
```

**Signature change for `render_napari()`:**

```python
def render_napari(
    env: Environment,
    fields: ...,
    *,
    speed: float = 1.0,  # NEW: replaces fps
    sample_rate_hz: float | None = None,  # NEW: inferred from frame_times if None
    max_playback_fps: int = 60,  # NEW: advanced knob
    # ... rest unchanged
)
```

**Note:** The napari backend needs `sample_rate_hz` to convert speed ↔ fps. This is computed in `animate_fields()` and passed via kwargs.

#### 4.2 Update `_add_speed_control_widget()` signature

**Current signature:**

```python
def _add_speed_control_widget(
    viewer: napari.Viewer,
    initial_fps: int = DEFAULT_FPS,
    frame_labels: list[str] | None = None,
) -> None:
```

**New signature:**

```python
def _add_speed_control_widget(
    viewer: napari.Viewer,
    initial_speed: float = 1.0,
    sample_rate_hz: float = 30.0,
    max_playback_fps: int = 60,
    frame_labels: list[str] | None = None,
) -> None:
```

#### 4.3 Update call site in `render_napari()`

**Current call (around line 2050):**

```python
_add_speed_control_widget(viewer, initial_fps=fps, frame_labels=frame_labels)
```

**New call:**

```python
_add_speed_control_widget(
    viewer,
    initial_speed=speed,
    sample_rate_hz=sample_rate_hz,
    max_playback_fps=max_playback_fps,
    frame_labels=frame_labels,
)
```

### Phase 5: Demo Script Updates

#### 5.1 Update `data/demo_spike_overlay_napari.py`

The demo currently generates synthetic `frame_times` from `fps`:

```python
# CURRENT (wrong - fps controls both data resolution AND playback)
n_frames = int(duration_s * fps)
frame_times = np.linspace(start_time, end_time, n_frames)
fields = np.tile(combined_field, (n_frames, 1))
```

**New approach:** Build `frame_times` from the underlying data's timestamps.

```python
# NEW: frame_times come from data, not from arbitrary fps
frame_times = times_window  # Actual position timestamps from data
n_frames = len(frame_times)

# Fields: one per data timestamp (static field repeated)
fields = np.tile(combined_field, (n_frames, 1))
```

**CLI argument changes:**

```python
# REMOVE:
parser.add_argument(
    "--fps",
    type=int,
    default=33,
    help="Frames per second (default: 33)",
)

# ADD:
parser.add_argument(
    "--speed",
    type=float,
    default=1.0,
    help="Playback speed relative to real-time (default: 1.0 = real-time)",
)
```

**Update animate_fields call:**

```python
# BEFORE:
env.animate_fields(
    fields,
    backend="napari",
    fps=fps,
    frame_times=frame_times,
    ...
)

# AFTER:
env.animate_fields(
    fields,
    backend="napari",
    speed=args.speed,  # 1.0 = real-time, 0.5 = half speed
    frame_times=frame_times,  # From data timestamps
    ...
)
```

**Update trail_length calculation:**

```python
# BEFORE: trail based on fps (wrong - conflates concepts)
trail_length=int(fps * 0.5)

# AFTER: trail based on data sample rate
sample_rate = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
trail_length=int(sample_rate * 0.5)  # 0.5 seconds of trail
```

**Update print statements:**

```python
# BEFORE:
print(f"\nAnimation: {n_frames} frames at {fps} fps")

# AFTER:
sample_rate = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
print(f"\nAnimation: {n_frames} frames (data at {sample_rate:.1f} Hz)")
print(f"  Playback speed: {args.speed}x")
```

#### 5.2 Update other demo scripts

Search for other scripts using `animate_fields` with `fps` parameter and update them similarly:

```bash
grep -r "animate_fields.*fps" data/ scripts/ examples/
```

### Phase 6: Test Updates

#### 6.1 Update existing tests

- Replace `fps=X` with `speed=X` where appropriate
- Add `frame_times` to test calls
- Update mocks/assertions

#### 6.2 Add new tests

```python
def test_speed_computation():
    """Test playback fps computation from speed."""
    # 30 Hz data, real-time
    frame_times = np.linspace(0, 10, 301)  # 30 Hz, 10 seconds
    fps, actual = _compute_playback_fps(frame_times, speed=1.0)
    assert fps == 30
    assert actual == pytest.approx(1.0)

def test_speed_capping():
    """Test that high speed is capped."""
    # 500 Hz data, real-time would be 500 fps
    frame_times = np.linspace(0, 1, 501)  # 500 Hz, 1 second
    fps, actual = _compute_playback_fps(frame_times, speed=1.0)
    assert fps == 60  # Capped
    assert actual == pytest.approx(0.12, rel=0.01)  # 60/500

def test_speed_warning(recwarn):
    """Test warning emitted when speed is capped."""
    frame_times = np.linspace(0, 1, 501)  # 500 Hz
    env.animate_fields(fields, frame_times=frame_times, speed=1.0)
    assert len(recwarn) == 1
    assert "Capped to 60 fps" in str(recwarn[0].message)

def test_single_frame_edge_case():
    """Test edge case: single frame returns defaults."""
    frame_times = np.array([0.0])  # Single frame
    fps, actual = _compute_playback_fps(frame_times, speed=1.0)
    assert fps == 60  # Returns max_fps
    assert actual == 1.0  # Returns requested speed

def test_zero_duration_edge_case():
    """Test edge case: zero duration (all same timestamp)."""
    frame_times = np.array([5.0, 5.0, 5.0])  # Zero duration
    fps, actual = _compute_playback_fps(frame_times, speed=1.0)
    assert fps == 60  # Returns max_fps
    assert actual == 1.0  # Returns requested speed

def test_slow_motion():
    """Test slow motion playback."""
    # 30 Hz data, 10% speed
    frame_times = np.linspace(0, 10, 301)  # 30 Hz, 10 seconds
    fps, actual = _compute_playback_fps(frame_times, speed=0.1)
    assert fps == 3  # 30 * 0.1 = 3 fps
    assert actual == pytest.approx(0.1)

def test_minimum_fps_clamping():
    """Test that fps is clamped to minimum (1 fps)."""
    # 10 Hz data, 1% speed would be 0.1 fps
    frame_times = np.linspace(0, 10, 101)  # 10 Hz
    fps, actual = _compute_playback_fps(frame_times, speed=0.01)
    assert fps == 1  # Clamped to MIN_PLAYBACK_FPS
    assert actual == pytest.approx(0.1)  # 1/10 = 0.1
```

### Phase 7: Documentation Updates

#### 7.1 Update CLAUDE.md

- Update Quick Reference examples
- Update animate_fields documentation
- Add note about speed vs fps
- Update demo script examples

#### 7.2 Update docstrings

All updated in Phase 2 (Environment method) and Phase 1 (core function).

---

## Migration Notes

### Breaking Changes

1. **`fps` parameter removed from public API**
   - Old: `env.animate_fields(fields, fps=30)`
   - New: `env.animate_fields(fields, frame_times=times, speed=1.0)`

2. **`frame_times` is now required (no default)**
   - Old: Optional, would synthesize from fps if missing
   - New: Required parameter, ValueError if not provided
   - Rationale: Time comes from data, not arbitrary fps

3. **`speed` replaces `fps` for playback control**
   - Old: `fps=30` (raw frames per second)
   - New: `speed=1.0` (multiplier relative to real-time)
   - `speed=0.1` means 10% of real-time (slow motion)
   - `speed=2.0` means 2x real-time (fast forward)

### Backwards Compatibility Strategy

We are **not** maintaining backwards compatibility (per user request). The old `fps` parameter will be removed entirely.

If `fps` is passed in `**kwargs`, it will be ignored (backends receive computed fps, not user-provided).

---

## File Change Summary

| File | Changes |
|------|---------|
| `src/neurospatial/animation/core.py` | Add constants, `_compute_playback_fps()` helper, update `animate_fields()` signature (require `frame_times`, add `speed`), pass `sample_rate_hz`/`speed`/`max_playback_fps` to backends via kwargs |
| `src/neurospatial/animation/overlays.py` | Rename `_build_frame_times()` → `_validate_frame_times()`, remove `fps` parameter, remove synthesis logic |
| `src/neurospatial/animation/backends/napari_backend.py` | Update `render_napari()` signature (add `speed`, `sample_rate_hz`, `max_playback_fps`), update `_add_speed_control_widget()` signature and implementation, update call site |
| `src/neurospatial/environment/visualization.py` | Replace `fps` with `speed`, make `frame_times` required, update docstring |
| `data/demo_spike_overlay_napari.py` | Remove `--fps`, add `--speed`, use data timestamps for `frame_times`, update trail_length calculation |
| `tests/animation/test_*.py` | Update existing tests (add `frame_times`, replace `fps` with `speed`), add `_compute_playback_fps()` unit tests including edge cases |
| `CLAUDE.md` | Update Quick Reference, animate_fields docs, demo examples |

---

## Design Decisions (Resolved)

1. **`frame_times` is strictly required**

   For correct temporal semantics and overlay alignment, `frame_times` is the canonical way to express time. No fallback or warning-with-default.

   - Rationale: The whole point of this change is that time comes from data, not from arbitrary fps.
   - Error message guides users: "Provide timestamps from your data source."

2. **`max_playback_fps` is configurable with strong default**

   Exposed as `max_playback_fps: int = 60` parameter for advanced users.

   - Default 60 fps works for most displays
   - Advanced users can increase for high-refresh displays (120/144 Hz)
   - Or decrease for slower systems

3. **Napari widget shows speed multiplier as primary control**

   Speed multiplier is the right mental model: "0.1×", "1.0×", "2.0×" matches how scientists think about time scaling.

   - Primary label: "Speed: 0.25×"
   - Secondary info: "≈ 12 fps" (for debugging)
   - Slider controls speed, internally converts to fps for napari

---

## Implementation Order

1. **Phase 1.1-1.2**: Add constants and `_compute_playback_fps()` helper to `core.py`
2. **Phase 1.4**: Update `_build_frame_times()` → `_validate_frame_times()` in `overlays.py`
3. **Phase 1.3**: Update `animate_fields()` in `core.py` (requires steps 1-2)
4. **Phase 2**: Update `Environment.animate_fields()` in `visualization.py`
5. **Phase 4.1-4.3**: Update napari backend:
   - 4.1: Update `_add_speed_control_widget()` implementation (speed slider)
   - 4.2: Update `_add_speed_control_widget()` signature
   - 4.3: Update `render_napari()` signature and call site
6. **Phase 6**: Update tests (existing tests + new edge case tests)
7. **Phase 5**: Update demo scripts
8. **Phase 7**: Update CLAUDE.md
