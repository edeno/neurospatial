# TimeSeriesOverlay Implementation - Scratchpad

## Current Status
**Date**: 2025-11-30
**Phase**: 5.5 - Final Integration Tests (IN PROGRESS)
**Task**: Phase 5.1-5.4 COMPLETE. Working on final integration tests.

## Progress Log

### 2025-11-30 (Session 6)

- **Phase 5.1 COMPLETE** - HTML backend time series warning
- Design Decision: Implemented warning approach instead of full HTML support
  - TimeSeriesOverlay requires separate panel with scrolling time series plots
  - HTML backend embeds frames as static base64 images
  - Dynamic chart updates and multi-panel layout not feasible in HTML
  - Consistent with VideoOverlay warning pattern
- Added time series overlay check in `render_html()`:
  - Checks `overlay_data.timeseries` length
  - Emits detailed warning with WHAT/WHY/HOW format
  - Suggests video or napari backends as alternatives
  - Overlay is skipped but other overlays still render
- Added 4 new tests in `TestHTMLBackendTimeSeriesWarning` class:
  - `test_html_backend_warns_on_timeseries_overlay`
  - `test_html_backend_still_renders_with_timeseries_present`
  - `test_html_backend_renders_other_overlays_with_timeseries_present`
  - `test_html_backend_no_warning_without_timeseries`
- Ruff and mypy: all passing
- Total time series tests: 56

### 2025-11-30 (Session 5)

- **Phase 4 COMPLETE** - Widget backend time series integration
- Updated `render_field_to_png_bytes_with_overlays()`:
  - Added `frame_times` parameter for time series synchronization
  - Uses GridSpec layout when time series present via `_setup_video_figure_with_timeseries()`
  - Calls `ts_manager.update()` for time series rendering per frame
- Updated `PersistentFigureRenderer`:
  - Added `frame_times` parameter to constructor
  - Lazy figure creation: detects time series on first render
  - Uses GridSpec layout when time series present
  - Updates time series manager on first and subsequent renders
- Updated `render_widget()`:
  - Added `frame_times` parameter with docstring documentation
  - Passes `frame_times` to pre-rendering loop
  - Passes `frame_times` to `PersistentFigureRenderer` constructor
- Fixed mypy errors:
  - Added explicit type narrowing for `overlay_data is not None` checks
  - Added type narrowing for `frame_times is not None` checks
  - Removed unused `type: ignore` comment
- Added 6 new tests (all passing, 52 total timeseries tests):
  - `test_render_field_to_png_bytes_with_timeseries`
  - `test_persistent_figure_renderer_with_timeseries`
  - `test_persistent_figure_renderer_timeseries_updates`
  - `test_persistent_figure_renderer_multiple_timeseries`
  - `test_render_widget_with_timeseries`
  - `test_widget_backend_no_timeseries_still_works`
- Ruff and mypy: all passing
- Code review: addressed all critical issues identified by code-reviewer agent

### 2025-11-30 (Session 4)

- **Phase 3 COMPLETE** - Video backend time series integration
- Implemented `_setup_video_figure_with_timeseries()` in `_timeseries.py`:
  - Uses `GridSpec` with width_ratios=[3, 1] (spatial field 75%, time series 25%)
  - Spatial field spans all rows on left
  - Time series axes stacked vertically on right
  - Returns `(fig, ax_field, ts_manager)` tuple
- Added `create_from_axes()` classmethod to `TimeSeriesArtistManager`:
  - Accepts pre-created axes for GridSpec layout integration
  - Reuses same artist creation logic as `create()`
  - Supports light theme (video backend uses white background)
- Updated `_render_worker_frames()` in `_parallel.py`:
  - Detects time series data in `overlay_data.timeseries`
  - Uses GridSpec layout when time series present
  - Updates time series manager per frame alongside overlay manager
  - Uses wider figure (12x6) to accommodate time series column
- Updated `parallel_render_frames()` to accept `frame_times` parameter
- Updated `render_video()` to accept and pass `frame_times` parameter
- Added 9 new tests (all passing):
  - `TestTimeSeriesVideoBackendLayout` (3 tests)
  - `TestTimeSeriesArtistManagerFromAxes` (2 tests)
  - `TestTimeSeriesVideoRender` (2 tests)
  - `TestTimeSeriesArtistManagerPickleSafe` (2 tests)
- Pickle safety verified: `TimeSeriesData` and `OverlayData.timeseries` are pickle-safe
- Ruff and mypy: all passing

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
- Phase 1-4 complete: Core infrastructure, Napari, Video, and Widget backends all working
- Ready for Phase 5: Polish & Performance (HTML backend, edge cases, documentation)
