# EventOverlay Implementation Plan

## Overview

Add a new overlay type `EventOverlay` for visualizing discrete timestamped events at the animal's position when each event occurred. This is a general-purpose overlay that supports:

- **Neural events**: Spike times, ripple events, sharp-wave events
- **Behavioral events**: Licks, lever presses, nose pokes, reward delivery
- **State transitions**: Trial starts, zone entries/exits
- **Any timestamped event**: Stimuli, optogenetic pulses, annotations

A convenience alias `SpikeOverlay` is provided for the common neural use case.

## Design Goals

1. **Simple API**: Easy to use for single event type, scales to multiple types
2. **Visualization Best Practices**:
   - Perceptually distinct colors for multiple event types (tab10 colormap)
   - Small markers that don't obscure the spatial field
   - Optional temporal decay for recent event visibility
   - Handles high event rates gracefully
   - Different markers per event type (optional)
3. **Consistent Architecture**: Follow existing overlay patterns (PositionOverlay, BodypartOverlay)
4. **All Backends**: Support napari, video, and HTML backends

## API Design

### User-Facing Dataclass: `EventOverlay`

```python
@dataclass
class EventOverlay:
    """Overlay for discrete timestamped events at specified spatial positions.

    Displays events as markers at positions that can be either:
    - Explicitly provided per event (for rewards, stimuli, zone events)
    - Interpolated from animal trajectory (for spikes, licks, animal-centric events)

    Supports multiple event types with distinct colors and optional
    temporal persistence to show recent event history.

    Parameters
    ----------
    event_times : NDArray[np.float64] | dict[str, NDArray[np.float64]]
        Event timestamps in seconds. Either:
        - Single event type: 1D array of event times
        - Multiple event types: dict mapping event names to time arrays

    Position Mode A - Explicit positions (for rewards, stimuli, zone events):
    event_positions : NDArray[np.float64] | dict[str, NDArray[np.float64]] | None
        Explicit event positions in environment coordinates.
        - Shape: (n_events, n_dims) for per-event positions
        - Shape: (1, n_dims) to broadcast single position to all events
        - dict: Mapping event names to position arrays
        If provided, positions are used directly (no interpolation).
        Mutually exclusive with positions/position_times.

    Position Mode B - Trajectory interpolation (for spikes, licks):
    positions : NDArray[np.float64] | None
        Animal position trajectory with shape (n_samples, n_dims) in environment
        (x, y) coordinates. Used to interpolate position at each event time.
        Mutually exclusive with event_positions.
    position_times : NDArray[np.float64] | None
        Timestamps for position samples in seconds. Must be monotonically
        increasing. Required when using positions parameter.
    interp : {"linear", "nearest"}, optional
        Interpolation for position lookup at event times. Default is "linear".
        Only used with positions/position_times mode.

    Appearance:
    colors : str | dict[str, str] | None, optional
        Colors for event markers:
        - str: Single color for all event types (e.g., "red")
        - dict: Mapping event names to colors
        - None: Auto-assign from perceptually distinct colormap (tab10)
        Default is None.
    size : float, optional
        Marker size in points. Default is 8.0.
    decay_frames : int | None, optional
        Number of frames over which event markers persist and fade.
        - None: Events appear only on their exact frame (instant)
        - 0: Same as None (instant)
        - >0: Events persist for N frames with decaying opacity
        Default is None (instant, no decay).
    markers : str | dict[str, str] | None, optional
        Marker style(s) for matplotlib/video backend ('o', 's', '^', 'v', 'd').
        - str: Single marker for all event types
        - dict: Mapping event names to markers
        - None: Use 'o' (circle) for all
        Napari uses circles regardless. Default is None.
    border_color : str, optional
        Border color for markers. Default is "white".
    border_width : float, optional
        Border width in pixels. Default is 0.5.

    Examples
    --------
    Reward delivery at fixed feeder location (explicit positions)::

        events = EventOverlay(
            event_times=reward_times,
            event_positions=np.array([[50.0, 25.0]]),  # Feeder location (broadcast)
            colors="gold",
            markers="s",
        )

    Zone entry events at zone boundaries (explicit positions per event)::

        events = EventOverlay(
            event_times=zone_entry_times,
            event_positions=zone_entry_locations,  # (n_events, 2)
            colors="cyan",
        )

    Neural spikes at animal position (trajectory interpolation)::

        events = EventOverlay(
            event_times=spike_times,      # Shape: (n_spikes,)
            positions=trajectory,          # Shape: (n_samples, 2)
            position_times=timestamps,     # Shape: (n_samples,)
            colors="red",
            size=10.0,
        )
        env.animate_fields(fields, overlays=[events])

    Multiple neurons with auto-colors::

        events = EventOverlay(
            event_times={
                "cell_001": spikes_1,
                "cell_002": spikes_2,
                "cell_003": spikes_3,
            },
            positions=trajectory,
            position_times=timestamps,
            # colors=None → auto-assign from tab10
        )

    Behavioral events with different markers::

        events = EventOverlay(
            event_times={
                "lick": lick_times,
                "reward": reward_times,
                "lever_press": press_times,
            },
            positions=trajectory,
            position_times=timestamps,
            colors={"lick": "cyan", "reward": "gold", "lever_press": "magenta"},
            markers={"lick": "o", "reward": "s", "lever_press": "^"},
        )

    With temporal decay (events visible for 5 frames)::

        events = EventOverlay(
            event_times=event_times,
            positions=trajectory,
            position_times=timestamps,
            decay_frames=5,  # Recent events fade over 5 frames
        )

    See Also
    --------
    SpikeOverlay : Convenience alias for EventOverlay for neural spike visualization.
    """


# Convenience alias for neural use case
SpikeOverlay = EventOverlay
```

### Internal Data Container: `EventData`

```python
@dataclass
class EventData:
    """Internal container for event overlay data aligned to animation frames.

    Parameters
    ----------
    event_positions : dict[str, NDArray[np.float64]]
        Mapping from event type name to event positions.
        Each array has shape (n_events, n_dims) in environment coordinates.
    event_frame_indices : dict[str, NDArray[np.int_]]
        Mapping from event type name to frame indices when events occur.
        Each array has shape (n_events,) with values in [0, n_frames-1].
    colors : dict[str, str]
        Mapping from event type name to color string (resolved from user input).
    markers : dict[str, str]
        Mapping from event type name to marker style (resolved from user input).
    size : float
        Marker size in points.
    decay_frames : int
        Number of frames for decay (0 = instant).
    border_color : str
        Marker border color.
    border_width : float
        Marker border width.
    """
```

## Implementation Tasks

### Task 1: Add EventOverlay and EventData to overlays.py

**File**: `src/neurospatial/animation/overlays.py`

1. Add imports for colormap handling:

   ```python
   from matplotlib import colormaps
   ```

2. Add `EventOverlay` dataclass after `HeadDirectionOverlay` (around line 673):
   - Implement `__post_init__` for validation
   - Implement `convert_to_data()` method

3. Add `SpikeOverlay` as alias:

   ```python
   # Convenience alias for neural use case
   SpikeOverlay = EventOverlay
   ```

4. Add `EventData` dataclass after `HeadDirectionData` (around line 1129):
   - Simple dataclass with fields listed above

5. Update `OverlayData` to include events:

   ```python
   @dataclass
   class OverlayData:
       positions: list[PositionData] = field(default_factory=list)
       bodypart_sets: list[BodypartData] = field(default_factory=list)
       head_directions: list[HeadDirectionData] = field(default_factory=list)
       videos: list[VideoData] = field(default_factory=list)
       events: list[EventData] = field(default_factory=list)  # NEW
       regions: dict[int, list[str]] | None = None
   ```

6. Update `_convert_overlays_to_data()` to dispatch EventData

7. Add validation helper `_validate_event_times()`:
   - Check event times are 1D and finite
   - Check all event times are within position_times range (warn if not)

### Task 2: Implement EventOverlay.convert_to_data()

**Logic**:

1. Normalize event_times to dict format (if single array → {"event": array})
2. Validate inputs (finite values, monotonic position_times, shape)
3. Resolve colors:
   - If None: auto-assign from `tab10` colormap
   - If str: use for all event types
   - If dict: validate all event types have colors
4. Resolve markers:
   - If None: use 'o' for all
   - If str: use for all event types
   - If dict: validate all event types have markers
5. For each event type:
   - Find events within animation time range [frame_times[0], frame_times[-1]]
   - Interpolate position at each event time
   - Find frame index for each event (nearest frame)
6. Return EventData with all event types

**Position interpolation**:

```python
# For each event time, interpolate position from trajectory
event_positions = np.column_stack([
    np.interp(event_times_in_range, position_times, positions[:, d])
    for d in range(n_dims)
])
```

**Frame index assignment**:

```python
# Find nearest frame for each event
event_frame_indices = np.searchsorted(frame_times, event_times_in_range)
# Clamp to valid range
event_frame_indices = np.clip(event_frame_indices, 0, n_frames - 1)
```

### Task 3: Napari Backend Rendering

**File**: `src/neurospatial/animation/backends/napari_backend.py`

**Implementation Note**: The patterns below are suggestions. During implementation, reconcile with the existing backend architecture - examine how `_render_position_overlay()`, `_render_bodypart_overlay()`, and `_render_head_direction_overlay()` handle time dimensions, layer types, and updates. Match established patterns for consistency; only deviate if there's a clear performance or correctness benefit.

**Considerations**:

1. **Time dimension**: Check how existing overlays use `(time, y, x)` format for Points layers
2. **Decay visualization**: Evaluate whether Tracks layer (tail_length) or expanded Points is more consistent with existing code
3. **Coloring approach**: Follow the pattern used by PositionOverlay for color handling
4. **Pre-compute vs callbacks**: Match existing overlay rendering approach

Add `_render_event_overlay()` function:

```python
def _render_event_overlay(
    viewer: napari.Viewer,
    event_data: EventData,
    env: Environment,
    n_frames: int,
    name_suffix: str = "",
) -> list[Layer]:
    """Render event overlay using napari best practices.

    For instant mode (decay_frames=0):
        Uses Points layer with native time dimension (t, y, x).
        Napari handles frame slicing automatically - O(1) per frame.

    For decay mode (decay_frames > 0):
        Uses Tracks layer with tail_length for efficient decay visualization.
        Each event becomes a short "track" that napari fades automatically.
        Avoids duplicating points for each decay frame.
    """
    layers = []

    for event_name, positions in event_data.event_positions.items():
        frame_indices = event_data.event_frame_indices[event_name]
        color = event_data.colors[event_name]

        if len(positions) == 0:
            continue

        # Transform to napari coordinates (y, x with Y-axis inversion)
        transformed = _transform_coords_for_napari(positions, env)

        if event_data.decay_frames <= 0:
            # INSTANT MODE: Points layer with native time dimension
            # Format: (time, y, x) - napari slices by time automatically
            points_data = np.column_stack([
                frame_indices,
                transformed[:, 0],  # Y
                transformed[:, 1],  # X
            ])

            layer = viewer.add_points(
                points_data,
                name=f"Events {event_name}{name_suffix}",
                size=event_data.size,
                face_color=color,
                border_color=event_data.border_color,
                border_width=event_data.border_width,
            )
            layers.append(layer)
        else:
            # DECAY MODE: Tracks layer with tail_length
            # Each event is a single-point "track" that persists via tail_length
            # Format: (track_id, time, y, x)
            #
            # Why Tracks layer?
            # - Native tail_length parameter handles decay automatically
            # - No need to duplicate points for each decay frame
            # - Efficient for high event counts (10K+ events)
            # - Built-in opacity decay

            n_events = len(positions)
            track_data = np.column_stack([
                np.arange(n_events),    # track_id (each event is its own track)
                frame_indices,           # time
                transformed[:, 0],       # Y
                transformed[:, 1],       # X
            ])

            # Use feature-based coloring for uniform color
            # (Tracks layer requires colormap approach, not direct color)
            from napari.utils.colormaps import Colormap
            features = {"color": np.zeros(n_events)}
            custom_colormap = Colormap(
                colors=[color, color],
                name=f"event_color_{event_name}{name_suffix}",
            )
            colormaps_dict = {"color": custom_colormap}

            layer = viewer.add_tracks(
                track_data,
                name=f"Events {event_name}{name_suffix}",
                tail_length=event_data.decay_frames,
                tail_width=event_data.size / 2,  # Track width (points/2 for visual match)
                features=features,
                colormaps_dict=colormaps_dict,
            )
            layer.color_by = "color"  # Set after features applied (avoids warning)
            layers.append(layer)

            # Also add current event marker (Points) for prominence
            # The track shows the tail, the point shows the current event
            points_data = np.column_stack([
                frame_indices,
                transformed[:, 0],
                transformed[:, 1],
            ])
            point_layer = viewer.add_points(
                points_data,
                name=f"Event Markers {event_name}{name_suffix}",
                size=event_data.size,
                face_color=color,
                border_color=event_data.border_color,
                border_width=event_data.border_width,
            )
            layers.append(point_layer)

    return layers
```

**Performance Considerations**:

- 10,000 events with decay_frames=10: Only 10,000 track entries (not 110,000 points)
- Napari's Tracks layer handles tail rendering in shader (GPU accelerated)
- Points layer with time dimension: napari filters visible points per frame automatically

Update `render_napari()` and `render_napari_non_blocking()` to call this function.

### Task 4: Video/Matplotlib Backend Rendering

**File**: `src/neurospatial/animation/_parallel.py`

Add `_render_event_overlay_matplotlib()`:

```python
def _render_event_overlay_matplotlib(
    ax: Any, event_data: EventData, frame_idx: int
) -> None:
    """Render event markers for current frame on matplotlib axes."""
    for event_name, positions in event_data.event_positions.items():
        frame_indices = event_data.event_frame_indices[event_name]
        color = event_data.colors[event_name]
        marker = event_data.markers[event_name]

        if event_data.decay_frames <= 0:
            # Instant mode: only show events on their exact frame
            mask = frame_indices == frame_idx
        else:
            # Decay mode: show events within decay window
            min_frame = frame_idx - event_data.decay_frames
            mask = (frame_indices >= max(0, min_frame)) & (frame_indices <= frame_idx)

        active_positions = positions[mask]
        active_frames = frame_indices[mask]

        if len(active_positions) == 0:
            continue

        # Compute per-event alpha based on recency
        if event_data.decay_frames > 0:
            # Alpha = 1.0 for current frame, decays to 0 for oldest
            ages = frame_idx - active_frames  # 0 = newest, decay_frames = oldest
            alphas = 1.0 - (ages / (event_data.decay_frames + 1))
        else:
            alphas = np.ones(len(active_positions))

        # Render each event (or batch with same alpha)
        from matplotlib.colors import to_rgba
        base_rgba = to_rgba(color)

        for pos, alpha in zip(active_positions, alphas):
            if np.any(np.isnan(pos)):
                continue
            ax.scatter(
                pos[0], pos[1],
                c=[(*base_rgba[:3], alpha)],
                s=event_data.size ** 2,
                marker=marker,
                zorder=103,
                edgecolors=event_data.border_color,
                linewidths=event_data.border_width,
            )
```

Update `_render_single_frame()` to call this function.

### Task 5: HTML Backend Support

**File**: `src/neurospatial/animation/backends/html_backend.py`

Add event rendering support (limited - no decay, just instant):

- Warn if decay_frames > 0 (HTML doesn't support per-frame alpha well)
- Render events as SVG circles on their frame

### Task 6: Update Public API Exports

**File**: `src/neurospatial/animation/__init__.py`

Add exports:

```python
from neurospatial.animation.overlays import (
    EventOverlay,
    SpikeOverlay,  # Alias for neural use case
    EventData,     # For advanced users
    # ... existing exports
)
```

**File**: `src/neurospatial/__init__.py`

Add to public API:

```python
from neurospatial.animation import EventOverlay, SpikeOverlay
```

### Task 7: Update CLAUDE.md Documentation

Add EventOverlay to Quick Reference section with examples for:

- Neural spikes (using SpikeOverlay alias)
- Behavioral events (licks, rewards)
- Multiple event types with different markers

### Task 8: Tests

**File**: `tests/animation/test_event_overlay.py`

Test cases:

1. Single event type, no decay
2. Single event type, with decay
3. Multiple event types, auto-colors
4. Multiple event types, custom colors
5. Multiple event types, custom markers
6. Events outside position time range (should warn)
7. Empty event times (edge case)
8. High event rate (performance)
9. Napari backend rendering
10. Video backend rendering
11. Temporal alignment (events at different rate than frames)
12. SpikeOverlay alias works identically

## Visualization Best Practices Applied

1. **Perceptually distinct colors**: Default to `tab10` colormap which has 10 maximally distinct colors
2. **Small markers**: Default size 8.0 (smaller than position marker's 10.0)
3. **White border**: Ensures visibility against any background
4. **Optional decay**: Users can choose instant (cleaner) or persistent (shows history)
5. **Different markers per event type**: Distinguishes event types beyond just color (accessibility)
6. **Consistent with existing overlays**: Same coordinate handling, validation patterns

## Edge Cases

1. **No events in time range**: Return empty arrays (no rendering)
2. **Events before/after position data**: Warn and exclude those events
3. **High event rate**: May cause visual clutter - suggest using decay_frames=0
4. **Overlapping events**: Multiple event types at same location - render in order (last on top)

## File Changes Summary

| File | Changes |
|------|---------|
| `src/neurospatial/animation/overlays.py` | Add EventOverlay, SpikeOverlay alias, EventData, update OverlayData |
| `src/neurospatial/animation/backends/napari_backend.py` | Add `_render_event_overlay()` |
| `src/neurospatial/animation/_parallel.py` | Add `_render_event_overlay_matplotlib()` |
| `src/neurospatial/animation/backends/html_backend.py` | Add event support (basic) |
| `src/neurospatial/animation/__init__.py` | Export EventOverlay, SpikeOverlay |
| `src/neurospatial/__init__.py` | Export EventOverlay, SpikeOverlay |
| `CLAUDE.md` | Add documentation with examples |
| `tests/animation/test_event_overlay.py` | New test file |
