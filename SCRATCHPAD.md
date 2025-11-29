# EventOverlay Implementation Scratchpad

## Current Status

**Date**: 2025-11-29
**Working on**: Milestone 2 Complete - Moving to M3

## Progress

- [x] M1: Core Data Structures - COMPLETE
- [x] M2: Napari Backend - COMPLETE
- [ ] M3: Video/Matplotlib Backend
- [ ] M4: HTML Backend
- [ ] M5: Public API & Documentation
- [ ] M6: Testing

## Notes

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

1. Milestone 3: Video/Matplotlib Backend
   - Add `_render_event_overlay_matplotlib()` function
   - Support instant and decay modes with alpha fading
   - Update `_render_single_frame()` to call it
