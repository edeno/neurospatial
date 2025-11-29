# EventOverlay Implementation Tasks

> **Goal**: Add `EventOverlay` for visualizing discrete timestamped events (spikes, licks, rewards, etc.) at specified spatial positions during animations. Supports both events at animal position (interpolated) and events at explicit locations.

## Milestones Overview

| Milestone | Description | Status |
|-----------|-------------|--------|
| M1 | Core Data Structures | ✅ Complete |
| M2 | Napari Backend | ✅ Complete |
| M3 | Video/Matplotlib Backend | ✅ Complete |
| M4 | HTML Backend | Not Started |
| M5 | Public API & Documentation | Not Started |
| M6 | Testing | Not Started |

---

## Milestone 1: Core Data Structures

**Goal**: Define `EventOverlay`, `EventData`, and conversion logic in `overlays.py`

**Dependencies**: None (foundation for all other milestones)

### Tasks

- [x] **1.1** Add colormap import to `overlays.py`
  - File: `src/neurospatial/animation/overlays.py`
  - Add: `from matplotlib import colormaps`
  - Location: Import section at top of file

- [x] **1.2** Define `EventOverlay` dataclass
  - File: `src/neurospatial/animation/overlays.py`
  - Location: After `HeadDirectionOverlay` (around line 673)
  - Fields:
    - `event_times`: `NDArray[np.float64] | dict[str, NDArray[np.float64]]`
    - **Position Mode A - Explicit positions (for rewards, stimuli, zone events):**
      - `event_positions`: `NDArray[np.float64] | dict[str, NDArray[np.float64]] | None`
        - Shape: (n_events, n_dims) per event type, or single (1, n_dims) broadcast to all
        - If provided, positions used directly (no interpolation)
    - **Position Mode B - Interpolate from trajectory (for spikes, licks):**
      - `positions`: `NDArray[np.float64] | None` (n_samples, n_dims) - animal trajectory
      - `position_times`: `NDArray[np.float64] | None` (n_samples,) - trajectory timestamps
      - `interp`: `Literal["linear", "nearest"]` (default: "linear")
    - **Mutual exclusion**: Must provide EITHER `event_positions` OR (`positions` + `position_times`)
    - **Appearance:**
      - `colors`: `str | dict[str, str] | None` (default: None → auto tab10)
      - `size`: `float` (default: 8.0)
      - `decay_frames`: `int | None` (default: None → instant)
      - `markers`: `str | dict[str, str] | None` (default: None → 'o')
      - `border_color`: `str` (default: "white")
      - `border_width`: `float` (default: 0.5)
  - Follow NumPy docstring format from existing overlays

- [x] **1.3** Implement `EventOverlay.__post_init__()` for validation
  - **Position mode validation:**
    - If `event_positions` provided: validate shape matches event_times
    - If `positions` + `position_times` provided: validate shapes, monotonicity
    - Error if both modes provided (mutual exclusion)
    - Error if neither mode provided
  - **Broadcast handling for explicit positions:**
    - Single position (1, n_dims) → broadcast to all events of that type
  - Normalize `event_times` to dict format (if array → {"event": array})
  - Normalize `event_positions` to dict format (if array → {"event": array})
  - Validate all event time arrays are 1D and contain finite values

- [x] **1.4** Add `SpikeOverlay` alias
  - Location: Immediately after `EventOverlay` class
  - Code: `SpikeOverlay = EventOverlay`
  - Add docstring: `"""Convenience alias for EventOverlay for neural spike visualization."""`

- [x] **1.5** Define `EventData` dataclass
  - File: `src/neurospatial/animation/overlays.py`
  - Location: After `HeadDirectionData` (around line 1129)
  - Fields:
    - `event_positions`: `dict[str, NDArray[np.float64]]` (name → positions)
    - `event_frame_indices`: `dict[str, NDArray[np.int_]]` (name → frame indices)
    - `colors`: `dict[str, str]` (resolved colors per event type)
    - `markers`: `dict[str, str]` (resolved markers per event type)
    - `size`: `float`
    - `decay_frames`: `int` (0 = instant)
    - `border_color`: `str`
    - `border_width`: `float`

- [x] **1.6** Update `OverlayData` to include events field
  - Add: `events: list[EventData] = field(default_factory=list)`
  - Location: After `videos` field in `OverlayData` dataclass

- [x] **1.7** Implement `EventOverlay.convert_to_data()` method
  - Input: `frame_times: NDArray[np.float64]` (animation frame timestamps)
  - Logic:
    1. Resolve colors: None → tab10 auto-assign, str → all same, dict → validate
    2. Resolve markers: None → 'o', str → all same, dict → validate
    3. For each event type:
       - Filter events within `frame_times` range
       - **If explicit positions mode**: use `event_positions` directly (broadcast if single)
       - **If trajectory mode**: interpolate positions at event times using `np.interp()`
       - Find nearest frame index using `np.searchsorted()`
    4. Return `EventData` with all resolved values
  - Handle edge case: empty events → empty arrays

- [x] **1.8** Add `_validate_event_times()` helper function
  - Check each event times array is 1D
  - Check all values are finite (not NaN/Inf)
  - Warn if any events outside frame_times range (trajectory mode: also check position_times)
  - Return filtered event times within valid range

- [x] **1.9** Update `_convert_overlays_to_data()` to handle EventOverlay
  - Add dispatch for `EventOverlay` type
  - Call `overlay.convert_to_data(frame_times)`
  - Append result to `overlay_data.events`

### Success Criteria

- [x] `EventOverlay` can be instantiated with single event array
- [x] `EventOverlay` can be instantiated with dict of event arrays
- [x] **Explicit positions mode**: `event_positions` used directly
- [x] **Trajectory mode**: positions interpolated from `positions` + `position_times`
- [x] **Broadcast**: single position (1, n_dims) broadcasts to all events
- [x] Mutual exclusion enforced (error if both modes or neither)
- [x] `convert_to_data()` returns valid `EventData`
- [x] Colors auto-assigned from tab10 when None
- [x] All validations raise appropriate errors

---

## Milestone 2: Napari Backend

**Goal**: Render events in napari viewer with instant and decay modes

**Dependencies**: M1 (Core Data Structures)

### Tasks

- [x] **2.1** Add `_render_event_overlay()` function
  - File: `src/neurospatial/animation/backends/napari_backend.py`
  - Signature: `(viewer, event_data, env, name_suffix="") -> list[Layer]`
  - Study existing overlay renderers first to match patterns

- [x] **2.2** Implement instant mode (decay_frames=0)
  - Use Points layer with native time dimension `(t, y, x)`
  - Transform coordinates using existing `_transform_coords_for_napari()`
  - Set face_color, border_color, border_width from EventData

- [x] **2.3** Implement decay mode (decay_frames > 0)
  - **Option A**: Tracks layer with `tail_length` (preferred for performance) ✓
  - Also adds Points layer for prominent current-frame marker
  - Match pattern used by existing overlays (check `_render_position_overlay`)

- [x] **2.4** Handle multiple event types
  - Create separate layer for each event type
  - Name layers as `"Events {event_name}{name_suffix}"`
  - Apply per-type colors from EventData.colors

- [x] **2.5** Update `render_napari()` to call `_render_event_overlay()`
  - Add call after other overlay rendering
  - Pass: viewer, event_data, env

- [x] **2.6** Update `_render_multi_field_napari()` to call `_render_event_overlay()`
  - Same pattern as `render_napari()`
  - Note: `render_napari_non_blocking()` doesn't exist as separate function

### Success Criteria

- [x] Events display at correct positions in napari
- [x] Events appear on correct frames
- [x] Instant mode: events visible only on their frame
- [x] Decay mode: events fade over specified frames
- [x] Multiple event types have distinct colors
- [x] Performance acceptable with 10K+ events (uses efficient Tracks layer for decay)

---

## Milestone 3: Video/Matplotlib Backend

**Goal**: Render events in video export with matplotlib

**Dependencies**: M1 (Core Data Structures)

### Tasks

- [x] **3.1** Add `_render_event_overlay_matplotlib()` function
  - File: `src/neurospatial/animation/_parallel.py`
  - Signature: `(ax, event_data, frame_idx) -> None`

- [x] **3.2** Implement instant mode rendering
  - Filter events where `frame_indices == frame_idx`
  - Use `ax.scatter()` with marker, color, size from EventData

- [x] **3.3** Implement decay mode rendering
  - Filter events in window `[frame_idx - decay_frames, frame_idx]`
  - Compute alpha based on age: `alpha = 1.0 - (age / (decay_frames + 1))`
  - Use `matplotlib.colors.to_rgba()` to apply per-event alpha

- [x] **3.4** Handle multiple event types
  - Iterate over all event types in EventData
  - Apply per-type colors and markers

- [x] **3.5** Update `_render_single_frame()` to call event renderer
  - Add call after existing overlay rendering
  - Check if `overlay_data.events` is non-empty

- [x] **3.6** Set appropriate zorder
  - Events should render above position overlay
  - Use `zorder=104` (above head direction at 103)

### Success Criteria

- [x] Events display correctly in exported videos
- [x] Instant and decay modes work
- [x] Different markers per event type render correctly
- [x] Alpha decay produces smooth fade effect
- [x] Events don't obscure main field visualization

---

## Milestone 4: HTML Backend

**Goal**: Basic event support in HTML standalone player

**Dependencies**: M1 (Core Data Structures)

### Tasks

- [ ] **4.1** Add event rendering to HTML backend
  - File: `src/neurospatial/animation/backends/html_backend.py`
  - Support instant mode only (no per-frame alpha in HTML)

- [ ] **4.2** Warn if decay_frames > 0
  - Log warning: "HTML backend does not support event decay. Events will display instantly only."
  - Fall back to instant mode behavior

- [ ] **4.3** Render events as SVG circles
  - Create SVG circle elements for events on each frame
  - Apply colors from EventData

- [ ] **4.4** Handle frame visibility
  - Only show events on their assigned frame
  - Match pattern used by position overlay in HTML

### Success Criteria

- [ ] Events appear in HTML export
- [ ] Warning displayed for decay_frames > 0
- [ ] Events positioned correctly
- [ ] Colors applied correctly

---

## Milestone 5: Public API & Documentation

**Goal**: Export new classes and document usage

**Dependencies**: M1-M4 (all implementation complete)

### Tasks

- [x] **5.1** Update `src/neurospatial/animation/__init__.py`
  - Add exports: `EventOverlay`, `SpikeOverlay`, `EventData`
  - Add to `__all__` list

- [ ] **5.2** Update `src/neurospatial/__init__.py`
  - Add top-level exports: `EventOverlay`, `SpikeOverlay`
  - Users should be able to: `from neurospatial import EventOverlay`

- [ ] **5.3** Update CLAUDE.md Quick Reference
  - Add EventOverlay examples after existing overlay examples
  - Include:
    - Single neuron spike visualization (using SpikeOverlay)
    - Multiple neurons with auto-colors
    - Behavioral events with different markers
    - Decay mode example

- [ ] **5.4** Add EventOverlay to overlay coordinate conventions table
  - Clarify that event positions use position trajectory coordinates
  - Note that events are plotted at interpolated animal position

- [ ] **5.5** Add EventOverlay to backend capability matrix
  - Document: Napari ✓, Video ✓, HTML ⚠️ (instant only)

### Success Criteria

- [ ] `from neurospatial import EventOverlay` works
- [ ] `from neurospatial import SpikeOverlay` works
- [ ] CLAUDE.md has complete usage examples
- [ ] Backend limitations documented

---

## Milestone 6: Testing

**Goal**: Comprehensive test coverage for EventOverlay

**Dependencies**: M1-M5 (all implementation and docs complete)

### Tasks

- [x] **6.1** Create test file `tests/animation/test_event_overlay.py`
  - Follow existing test patterns in `tests/animation/`

- [ ] **6.2** Test explicit positions mode (Mode A)
  - Create EventOverlay with `event_positions` array
  - Verify positions used directly (no interpolation)
  - Verify frame_indices correct

- [ ] **6.3** Test trajectory interpolation mode (Mode B)
  - Create EventOverlay with `positions` + `position_times`
  - Verify positions interpolated at event times
  - Verify frame_indices correct

- [ ] **6.4** Test position broadcast (single position)
  - Provide single position (1, 2) with multiple events
  - Verify position broadcast to all events

- [ ] **6.5** Test mutual exclusion validation
  - Error if both `event_positions` AND `positions`/`position_times` provided
  - Error if neither provided
  - Verify clear error messages

- [ ] **6.6** Test decay mode
  - decay_frames=5
  - Verify EventData.decay_frames set correctly

- [ ] **6.7** Test multiple event types, auto-colors
  - Dict of 3+ event types
  - Verify colors auto-assigned from tab10
  - Verify colors are distinct

- [ ] **6.8** Test multiple event types, custom colors
  - Provide colors dict
  - Verify colors match input

- [ ] **6.9** Test multiple event types, custom markers
  - Provide markers dict
  - Verify markers match input

- [ ] **6.10** Test events outside time range
  - Some events before/after frame_times (or position_times in Mode B)
  - Verify warning emitted
  - Verify out-of-range events excluded

- [ ] **6.11** Test empty event times
  - Empty array input
  - Verify returns empty EventData (no crash)

- [ ] **6.12** Test high event rate (performance)
  - 10,000+ events
  - Verify convert_to_data() completes in reasonable time (<1s)

- [x] **6.13** Test napari backend integration
  - Mock napari.Viewer for unit tests
  - Verify layers created correctly (instant mode, decay mode)
  - Tests in `TestNapariEventOverlay*` classes

- [ ] **6.14** Test video backend integration
  - Create small animation with events
  - Verify no errors during rendering

- [ ] **6.15** Test temporal alignment
  - Events at different rate than frames
  - Verify events assigned to correct frames

- [ ] **6.16** Test SpikeOverlay alias
  - Verify `SpikeOverlay is EventOverlay`
  - Verify SpikeOverlay instantiation works

- [ ] **6.17** Test position interpolation modes (Mode B only)
  - Test interp="linear" (default)
  - Test interp="nearest"
  - Verify different results

### Success Criteria

- [ ] All tests pass with `uv run pytest tests/animation/test_event_overlay.py`
- [ ] Test coverage >90% for new code
- [ ] Edge cases handled gracefully (no crashes)

---

## Implementation Order

**Recommended sequence**:

```text
M1 (Core) → M2 (Napari) → M3 (Video) → M4 (HTML) → M5 (API/Docs) → M6 (Tests)
```

**Parallel opportunities**:

- M2, M3, M4 can be developed in parallel after M1 is complete
- M6 tests can be written alongside implementation (TDD approach)

**Critical path**:

- M1.7 (convert_to_data) blocks all backend work
- M5 should wait until backends are tested manually

---

## Notes for Implementation

### Two Position Modes

**Mode A - Explicit positions** (rewards, stimuli, zone events):

```python
# Each event has its own position
events = EventOverlay(
    event_times=reward_times,           # When events occurred
    event_positions=reward_locations,   # Where events occurred (n_events, 2)
)

# Fixed location (broadcast to all events)
events = EventOverlay(
    event_times=door_open_times,
    event_positions=np.array([[50.0, 25.0]]),  # Single position → all events here
)
```

**Mode B - Trajectory interpolation** (spikes, licks, animal-centric events):

```python
# Position interpolated from animal trajectory at event time
events = EventOverlay(
    event_times=spike_times,
    positions=trajectory,        # Animal trajectory (n_samples, 2)
    position_times=timestamps,   # Trajectory timestamps (n_samples,)
)
```

### Coordinate Convention Reminder

- All overlay coordinates use **environment space** (x, y)
- Animation system transforms to napari pixel space internally
- Don't manually swap axes or invert Y before passing to overlay

### Color Auto-Assignment

```python
from matplotlib import colormaps
from matplotlib.colors import to_hex

tab10 = colormaps["tab10"]
# Normalize RGBA tuples to hex strings for type consistency
colors = {name: to_hex(tab10(i % 10)) for i, name in enumerate(event_names)}
# Result: {"cell_001": "#1f77b4", "cell_002": "#ff7f0e", ...}
```

**Type consistency**: Always normalize colors to hex strings internally. This ensures:

- `EventData.colors` is always `dict[str, str]` as documented
- Backends receive consistent string format
- User-provided colors (str) and auto-assigned colors (from RGBA) are both strings

### Position Interpolation (Mode B only)

```python
# Interpolate position at event times from trajectory
event_positions = np.column_stack([
    np.interp(event_times, position_times, positions[:, d])
    for d in range(n_dims)
])
```

### Position Broadcast (Mode A with single position)

```python
# Broadcast single position to all events
if event_positions.shape[0] == 1:
    event_positions = np.broadcast_to(event_positions, (n_events, n_dims))
```

### Frame Index Assignment

```python
# Find nearest frame for each event
frame_indices = np.searchsorted(frame_times, event_times)
frame_indices = np.clip(frame_indices, 0, len(frame_times) - 1)
```

---

## Files to Modify

| File | Milestone | Changes |
|------|-----------|---------|
| `src/neurospatial/animation/overlays.py` | M1 | Add EventOverlay, EventData, update OverlayData |
| `src/neurospatial/animation/backends/napari_backend.py` | M2 | Add `_render_event_overlay()` |
| `src/neurospatial/animation/_parallel.py` | M3 | Add `_render_event_overlay_matplotlib()` |
| `src/neurospatial/animation/backends/html_backend.py` | M4 | Add event support |
| `src/neurospatial/animation/__init__.py` | M5 | Export new classes |
| `src/neurospatial/__init__.py` | M5 | Top-level exports |
| `CLAUDE.md` | M5 | Documentation |
| `tests/animation/test_event_overlay.py` | M6 | New test file |
