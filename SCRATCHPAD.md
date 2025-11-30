# TimeSeriesOverlay Implementation - Scratchpad

## Current Status
**Date**: 2025-11-30
**Phase**: 2 - Napari Backend (COMPLETE)
**Task**: Phase 2 COMPLETE - all napari backend tasks done. Ready for Phase 3 (Video Backend)

## Progress Log

### 2025-11-30 (Session 3)

- **Phase 2.3, 2.4, 2.5 COMPLETE** - Napari dock widget integration
- Implemented `_add_timeseries_dock()` in napari_backend.py:
  - Creates FigureCanvasQTAgg with matplotlib figure
  - Wraps in QWidget with QVBoxLayout
  - Adds dock widget to viewer's right area ("Time Series")
  - Connects to `viewer.dims.events.current_step` for frame updates
  - Update throttling at 40 Hz (`TIMESERIES_MAX_UPDATE_HZ`)
  - Initial render at frame 0
- Updated `render_napari()` and `_render_multi_field_napari()` to call `_add_timeseries_dock()`
- Added `frame_times` field to `OverlayData` for time series synchronization
- Added 4 integration tests (skip in headless CI - require Qt):
  - `test_timeseries_dock_widget_added` - dock widget created
  - `test_timeseries_dock_updates_on_frame_change` - callback works
  - `test_render_napari_with_timeseries` - end-to-end integration
  - `test_no_timeseries_dock_when_empty` - no dock when no timeseries
- All tests pass (37 passed, 4 skipped in headless CI)
- Ruff and mypy: all passing

### 2025-11-30 (Session 2)
- **Phase 2.1 and 2.2 COMPLETE**
- Created `src/neurospatial/animation/_timeseries.py`:
  - `_group_timeseries()` - Groups overlays by their `group` parameter
  - `_get_group_index()` - Returns group index for a TimeSeriesData
  - `TimeSeriesArtistManager` dataclass with `create()` and `update()` methods
- Key implementation decisions:
  - Used identity check (`is`) instead of `in` for numpy array membership
  - Cached groups in manager to avoid recomputing on each `update()` call
  - Cursor is `None` if all overlays in group have `show_cursor=False`
  - Used walrus operator for safe label type checking (mypy fix)
- Added 13 new tests (all passing, 37 total timeseries tests)
- Code review: addressed 4 of 7 issues:
  - Removed dead fallback code in `update()`
  - Used `enumerate()` instead of `axes.index(ax)` for O(n) improvement
  - Respect `show_cursor=False` setting per group
  - Cache groups in manager (O(1) instead of O(n) per frame)
- Ruff and mypy: all passing

### 2025-11-30 (Session 1)
- **Phase 1 COMPLETE**
- Created `TimeSeriesOverlay` dataclass with full validation:
  - 1D data/times validation
  - Same length validation
  - Minimum 1 sample validation (added per code review)
  - Monotonically increasing times validation
  - Finite times validation (no NaN/Inf)
  - Positive window_seconds validation
  - Alpha in [0, 1] validation
  - Positive linewidth validation (added per code review)
  - No Inf in data validation (NaN allowed for gaps)
- Created `TimeSeriesData` internal container with:
  - O(1) window extraction via `get_window_slice()`
  - Cursor value interpolation (linear/nearest) via `get_cursor_value()`
  - Precomputed start/end indices for all frames
- Updated `OverlayData` with `timeseries` field
- Updated `_convert_overlays_to_data()` to dispatch `TimeSeriesData`
- Added exports to `__init__.py`
- Created 24 tests, all passing
- Code review passed after addressing feedback

## Decisions Made

1. **NaN handling**: NaN values in `data` are allowed (creates gaps in line). Inf values are rejected.
2. **Normalization with constant data**: Returns all zeros (min == max case)
3. **Empty data**: Rejected with validation error (must have at least 1 sample)
4. **Linewidth**: Must be positive (validation added per code review)

## Blockers

(None)

## Notes

- Feature adds continuous variable visualization (speed, LFP, etc.) to animations
- Time series displayed in right column with scrolling window
- Follows existing overlay pattern (PositionOverlay, BodypartOverlay, etc.)
- Ready for Phase 2: Napari Backend integration
