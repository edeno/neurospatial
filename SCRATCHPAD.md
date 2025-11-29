# EventOverlay Implementation Scratchpad

## Current Status

**Date**: 2025-11-29
**Working on**: Milestone 4 Complete - Moving to M5

## Progress

- [x] M1: Core Data Structures - COMPLETE
- [x] M2: Napari Backend - COMPLETE
- [x] M3: Video/Matplotlib Backend - COMPLETE
- [x] M4: HTML Backend - COMPLETE
- [ ] M5: Public API & Documentation
- [ ] M6: Testing

## Notes

### Milestone 4 Completed (2025-11-29)

**Implemented event overlay support in HTML backend:**

1. **Instant mode** (decay_frames=0):
   - Events visible only on their exact frame
   - Uses Canvas 2D context for rendering (not SVG)
   - Renders filled circles with border

2. **Decay warning** (decay_frames > 0):
   - Emits UserWarning: "HTML backend does not support event decay"
   - Falls back to instant mode behavior
   - Suggests video or napari backends for decay support

3. **Event serialization to JSON**:
   - Added `EventOverlayJSON` TypedDict for type safety
   - Events serialized via `_serialize_overlay_data()` function
   - Includes event_positions, frame_indices, colors, size, border settings

4. **JavaScript rendering in `renderOverlays()`:**
   - Iterates over event types, checks frame_indices matching current frame
   - Renders circles with fillStyle and strokeStyle
   - Handles NaN positions gracefully

**Key implementation details:**

- Added event size estimation to `_estimate_overlay_json_size()`
- Updated module docstring with events in JSON schema
- Both embedded and non-embedded HTML modes support events
- Tests: 16 HTML event overlay tests added and passing
- All 955 animation tests pass

---

### Milestone 3 Completed (2025-11-29)

**Implemented `_render_event_overlay_matplotlib()` in _parallel.py:**

1. **Instant mode** (decay_frames=0):
   - Filters events where `frame_indices == frame_idx`
   - Renders with `ax.scatter()` using marker, color, size from EventData
   - Events visible only on their exact frame

2. **Decay mode** (decay_frames > 0):
   - Filters events in window `[frame_idx - decay_frames, frame_idx]`
   - Computes per-event alpha based on age: older events fade
   - Uses `matplotlib.colors.to_rgba()` to apply alpha to each event
   - +1 in denominator prevents alpha from reaching exactly 0

3. **Multiple event types**:
   - Iterates over all event types in EventData
   - Applies per-type colors and markers via separate scatter calls

**Key implementation details:**

- Uses zorder=104 (above head direction at 103)
- Handles empty event arrays gracefully
- Skips NaN positions
- Updated `_render_all_overlays()` to call event renderer

**Tests added:**

- 17 matplotlib backend tests in `TestMatplotlibEventOverlay*` classes
- Tests instant mode, decay mode, multiple types, integration, edge cases
- All 71 event overlay tests pass (M1 + M2 + M3)

---

### Milestone 2 Completed (2025-11-29)

**Implemented `_render_event_overlay()` in napari backend:**

1. **Instant mode** (decay_frames=0):
   - Uses Points layer with `(time, y, x)` format
   - Events visible only on their exact frame

2. **Decay mode** (decay_frames > 0):
   - Uses Tracks layer with `tail_length` for efficient GPU rendering
   - Also adds Points layer for prominent current-frame marker
   - Events persist and fade over specified frames

3. **Multiple event types**:
   - Creates separate layer for each event type
   - Naming: `"Events {event_name}{suffix}"`
   - Per-type colors from EventData.colors

**Key implementation details:**

- Coordinates transformed via `_transform_coords_for_napari()`
- Uses napari's Colormap for uniform track colors
- `markers` parameter documented as not used (napari renders all as circles)
- Both `render_napari()` and `_render_multi_field_napari()` updated

**Tests added:**

- 12 napari backend tests in `TestNapariEventOverlay*` classes
- Tests instant mode, decay mode, multiple types, coordinates, edge cases
- All 54 event overlay tests pass

### Key Design Decisions from PLAN.md

**Two position modes:**
- Mode A (Explicit): `event_positions` for rewards, stimuli at fixed locations
- Mode B (Interpolated): `positions` + `position_times` for spikes, licks at animal position

**Mutual exclusion**: Must provide either Mode A or Mode B, not both, not neither.

**Color auto-assignment**: Uses tab10 colormap when colors=None

## Blockers

None currently.

## Questions

None currently.

## Next Steps

1. Milestone 5: Public API & Documentation
   - Update top-level exports in `__init__.py`
   - Update CLAUDE.md with EventOverlay examples
   - Add to overlay coordinate conventions table
   - Add to backend capability matrix
