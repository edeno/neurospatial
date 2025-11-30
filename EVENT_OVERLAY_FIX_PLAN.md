# EventOverlay Performance Fix: Points + Dynamic Visibility

## Problem Statement

Current EventOverlay implementation for napari creates one Track per event. For long recordings with high-firing neurons, this creates hundreds of thousands of Tracks, overwhelming napari's rendering pipeline.

**Example:** 1 hour recording, 3 neurons @ 50 Hz = 540,000 tracks → napari freezes

## Goal

Enable performant visualization of millions of events with two modes:

1. **Cumulative mode**: Events appear and stay permanently (spikes accumulate)
2. **Decay mode**: Events visible for N frames then disappear

Target: 1 hour recording @ 500 Hz display rate with 100+ neurons should render smoothly.

## Solution Overview

Replace Tracks layer with Points layer + dynamic `shown` mask updated via callback on frame change.

### Why This Works

| Approach | Objects | Frame Update Cost |
|----------|---------|-------------------|
| Current (Tracks) | 540K Track objects | Re-render all tracks |
| Proposed (Points) | 1 Points layer | Boolean mask update (~10ms) |

napari's Points layer efficiently handles millions of points. The `shown` property is a boolean mask that controls visibility without recreating geometry.

## Architecture

### Current Flow (napari backend)

```
EventOverlay → _convert_overlays_to_data() → EventData → _render_event_overlay()
                                                              ↓
                                                        Creates Tracks layer (1 per event)
                                                        OR Points layer (decay=0 only)
```

### New Flow

```
EventOverlay → _convert_overlays_to_data() → EventData → _render_event_overlay()
                                                              ↓
                                                        Creates ONE Points layer (all events)
                                                        Registers frame-change callback
                                                        Callback updates layer.shown mask
```

## API Changes

### EventOverlay Parameters

```python
EventOverlay(
    event_times=...,
    positions=...,
    position_times=...,

    # CHANGED: decay_frames semantics
    decay_frames=None,  # None = cumulative (events persist forever) [NEW DEFAULT]
    decay_frames=0,     # Instant (visible only on exact frame)
    decay_frames=40,    # Visible for 40 frames then hidden

    # Existing parameters unchanged
    colors=...,
    size=...,
    border_color=...,
    border_width=...,
)
```

**Key change:** `decay_frames=None` now means cumulative mode (was previously invalid/defaulted to 0).

### Behavior Matrix

| `decay_frames` | Behavior | Implementation |
|----------------|----------|----------------|
| `None` | Cumulative - events stay forever | `shown = frame_idx <= current_frame` |
| `0` | Instant - visible only on exact frame | `shown = frame_idx == current_frame` |
| `N > 0` | Decay - visible for N frames | `shown = (current - N <= frame_idx) & (frame_idx <= current)` |

## Implementation Plan

### Phase 1: Refactor `_render_event_overlay()` in napari_backend.py

**File:** `src/neurospatial/animation/backends/napari_backend.py`

#### Step 1.1: Remove Tracks layer code path

Current code has two branches:

- `if use_decay:` → Creates Tracks layer
- `else:` → Creates Points layer

Remove the Tracks branch entirely.

#### Step 1.2: Create unified Points layer

```python
def _render_event_overlay(
    viewer: napari.Viewer,
    event_data: EventData,
    env: Environment,
    name_suffix: str = "",
) -> list[Layer]:
    """Render event overlay using Points layer with dynamic visibility."""
    layers: list[Layer] = []

    # Collect ALL events across all event types into single arrays
    all_positions = []
    all_frame_indices = []
    all_colors = []
    all_event_types = []  # For legend/layer naming

    for event_type, positions in event_data.event_positions.items():
        frame_indices = event_data.event_frame_indices[event_type]
        color = event_data.colors.get(event_type, "#ffffff")

        if len(positions) == 0:
            continue

        # Transform coordinates to napari (y, x) with Y-axis inversion
        transformed = _transform_coords_for_napari(positions, env)

        all_positions.append(transformed)
        all_frame_indices.append(frame_indices)
        all_colors.extend([color] * len(positions))
        all_event_types.extend([event_type] * len(positions))

    if not all_positions:
        return layers

    # Concatenate into single arrays
    positions_array = np.vstack(all_positions)  # (N_total, 2)
    frame_indices_array = np.concatenate(all_frame_indices)  # (N_total,)
    colors_array = np.array(all_colors)  # (N_total,) strings or RGBA

    # Sort by frame index for efficient cumulative updates
    sort_idx = np.argsort(frame_indices_array)
    positions_array = positions_array[sort_idx]
    frame_indices_array = frame_indices_array[sort_idx]
    colors_array = colors_array[sort_idx]

    # Create Points layer with all events, initially hidden
    n_events = len(positions_array)
    initial_shown = np.zeros(n_events, dtype=bool)

    # Add points WITHOUT time dimension - we control visibility via shown mask
    points_layer = viewer.add_points(
        positions_array,  # (N, 2) - just y, x
        name=f"Events{name_suffix}",
        size=event_data.size,
        face_color=colors_array,
        border_color=event_data.border_color,
        border_width=event_data.border_width,
        shown=initial_shown,
    )

    # Store metadata for callback
    points_layer.metadata["frame_indices"] = frame_indices_array
    points_layer.metadata["decay_frames"] = event_data.decay_frames
    points_layer.metadata["n_events"] = n_events

    # Register frame-change callback
    _register_event_visibility_callback(viewer, points_layer)

    layers.append(points_layer)
    return layers
```

#### Step 1.3: Implement visibility callback

```python
def _register_event_visibility_callback(
    viewer: napari.Viewer,
    points_layer: napari.layers.Points,
) -> None:
    """Register callback to update event visibility on frame change."""

    def on_frame_change(event):
        # Get current frame from dims slider
        current_frame = viewer.dims.current_step[0]

        # Retrieve metadata
        frame_indices = points_layer.metadata["frame_indices"]
        decay_frames = points_layer.metadata["decay_frames"]
        n_events = points_layer.metadata["n_events"]

        # Compute visibility mask based on mode
        if decay_frames is None:
            # Cumulative mode: all events up to current frame
            # Use searchsorted for O(log N) instead of O(N) comparison
            cutoff_idx = np.searchsorted(frame_indices, current_frame, side='right')
            shown = np.zeros(n_events, dtype=bool)
            shown[:cutoff_idx] = True
        elif decay_frames == 0:
            # Instant mode: only events on exact frame
            shown = frame_indices == current_frame
        else:
            # Decay mode: events within window
            start_frame = current_frame - decay_frames
            shown = (frame_indices >= start_frame) & (frame_indices <= current_frame)

        # Update layer visibility
        points_layer.shown = shown

    # Connect to dims change event
    viewer.dims.events.current_step.connect(on_frame_change)

    # Trigger initial update
    on_frame_change(None)
```

### Phase 2: Update EventData and conversion

**File:** `src/neurospatial/animation/overlays.py`

#### Step 2.1: Update EventData dataclass

```python
@dataclass
class EventData:
    """Internal representation of event overlay data."""
    event_positions: dict[str, NDArray[np.float64]]
    event_frame_indices: dict[str, NDArray[np.int_]]
    colors: dict[str, str]
    markers: dict[str, str]
    size: float
    decay_frames: int | None  # CHANGED: Allow None for cumulative
    border_color: str
    border_width: float
```

#### Step 2.2: Update EventOverlay.to_data()

Handle `decay_frames=None` as cumulative mode:

```python
def to_data(self, env, frame_times, n_frames) -> EventData:
    # ... existing code ...

    # Resolve decay_frames (None stays None for cumulative)
    decay = self.decay_frames  # Don't convert None to 0

    return EventData(
        # ... existing fields ...
        decay_frames=decay,
    )
```

### Phase 3: Update EventOverlay default

**File:** `src/neurospatial/animation/overlays.py`

```python
@dataclass
class EventOverlay:
    """Overlay for discrete events (spikes, rewards, zone crossings)."""

    event_times: ...
    positions: ...
    # ...

    decay_frames: int | None = None  # CHANGED: Default to cumulative
```

### Phase 4: Clean up Tracks-related code

Remove from `_render_event_overlay()`:

- Tracks layer creation code
- Colormap workaround for Tracks
- `head_length` / `tail_length` parameters
- Related imports

### Phase 5: Update other backends

#### Video backend (`_parallel.py`)

The video backend renders frame-by-frame, so the current approach is fine. Just ensure it handles `decay_frames=None`:

```python
def _render_events_on_frame(ax, event_data, frame_idx, ...):
    decay = event_data.decay_frames

    if decay is None:
        # Cumulative: show all events up to current frame
        mask = frame_indices <= frame_idx
    elif decay == 0:
        mask = frame_indices == frame_idx
    else:
        mask = (frame_indices >= frame_idx - decay) & (frame_indices <= frame_idx)

    # Render visible events
    ...
```

#### HTML backend

Similar update to handle `decay_frames=None`.

### Phase 6: Remove warning about track count

**File:** `src/neurospatial/animation/backends/napari_backend.py`

Remove or update the MAX_RECOMMENDED_TRACKS warning since Points layer handles any count.

### Phase 7: Update demo script

**File:** `data/demo_spike_overlay_napari.py`

- Remove the "too many spikes" warning
- Update documentation to explain new decay_frames semantics
- Add examples of cumulative vs decay modes

## Testing Strategy

### Unit Tests

1. **Test cumulative mode**: Events accumulate as frames advance
2. **Test instant mode**: Events only visible on exact frame
3. **Test decay mode**: Events visible for N frames then hidden
4. **Test coordinate transform**: Events appear at correct positions
5. **Test multiple event types**: Colors preserved, all events visible
6. **Test empty events**: No crash with zero events
7. **Test callback cleanup**: No memory leak on viewer close

### Performance Tests

1. **1M events benchmark**: Create EventOverlay with 1M events, verify:
   - Initialization < 5 seconds
   - Frame update < 50ms
   - Memory < 500 MB

2. **Full recording test**: 1 hour @ 500 Hz with 100 neurons:
   - Should not freeze
   - Smooth playback at 30+ fps

### Integration Tests

1. **With PositionOverlay**: Events + position trail work together
2. **With video backend**: Same events render correctly to video
3. **With frame_times alignment**: Events align to correct frames

## Migration Notes

### Breaking Changes

1. `decay_frames` default changes from `5` to `None` (cumulative)
   - Users who relied on default decay behavior need to explicitly set `decay_frames=5`

2. Visual change: No more "tail" effect with decay > 0
   - Events now binary visible/hidden instead of fading trail
   - This is more semantically correct for discrete events

### Deprecations

None - we're replacing internals, not adding new API.

## File Changes Summary

| File | Changes |
|------|---------|
| `src/neurospatial/animation/backends/napari_backend.py` | Replace Tracks with Points + callback |
| `src/neurospatial/animation/overlays.py` | Update EventData, EventOverlay default |
| `src/neurospatial/animation/_parallel.py` | Handle decay_frames=None |
| `src/neurospatial/animation/backends/html_backend.py` | Handle decay_frames=None |
| `data/demo_spike_overlay_napari.py` | Update examples, remove warning |
| `tests/animation/test_overlays.py` | Add cumulative mode tests |
| `tests/animation/test_napari_backend.py` | Update event rendering tests |

## Estimated Effort

- Phase 1-2: Core implementation - 2-3 hours
- Phase 3-4: Cleanup - 30 min
- Phase 5: Other backends - 1 hour
- Phase 6-7: Demo/docs - 30 min
- Testing: 2 hours

**Total: ~6-8 hours**

## Open Questions

1. **Per-event-type layers vs single layer?**
   - Current plan: Single layer with all events (simpler callback)
   - Alternative: One layer per event type (easier to toggle visibility in napari)
   - Recommendation: Single layer, use `features` for filtering if needed

2. **Alpha fade for decay mode?**
   - Could modulate `face_color` alpha based on age
   - More expensive (update colors each frame vs just shown mask)
   - Recommendation: Skip for v1, add later if requested

3. **Callback cleanup on viewer close?**
   - napari should handle this automatically when viewer is garbage collected
   - May need explicit disconnect if memory leaks observed
   - Recommendation: Test and add cleanup if needed
