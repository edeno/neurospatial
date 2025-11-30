# TimeSeriesOverlay Implementation Tasks

**Feature**: Continuous Variable Animation Overlay (speed, head direction, LFP, etc.)
**Plan**: See [PLAN.md](PLAN.md) for full design specification

---

## Phase 1: Core Infrastructure

**Goal**: Define data structures and conversion pipeline for time series overlays.

**Files to modify**:
- `src/neurospatial/animation/overlays.py`
- `src/neurospatial/animation/__init__.py`

### 1.1 TimeSeriesOverlay Dataclass

- [x] Add `TimeSeriesOverlay` dataclass to `overlays.py` (after other overlay classes ~line 280)
  - Fields: `data`, `times`, `label`, `color`, `window_seconds`, `linewidth`, `alpha`, `group`, `normalize`, `show_cursor`, `cursor_color`, `vmin`, `vmax`, `interp`
  - Include comprehensive NumPy-style docstring with examples
  - Add `__post_init__` validation:
    - `data` and `times` must be 1D arrays of same length
    - `times` must be monotonically increasing
    - `window_seconds` must be positive
    - `alpha` must be in [0, 1]
    - `linewidth` must be positive (added per code review)
    - `data` must have at least 1 sample (added per code review)

### 1.2 TimeSeriesData Internal Container

- [x] Add `TimeSeriesData` dataclass to `overlays.py` (after `TimeSeriesOverlay`)
  - Fields: `data`, `times`, `start_indices`, `end_indices`, `label`, `color`, `window_seconds`, `linewidth`, `alpha`, `group`, `normalize`, `show_cursor`, `cursor_color`, `global_vmin`, `global_vmax`, `use_global_limits`, `interp`
  - Include `get_window_slice(frame_idx)` method for O(1) extraction
  - Include `get_cursor_value(current_time)` method with linear/nearest interpolation

### 1.3 convert_to_data Implementation

- [x] Implement `convert_to_data()` method on `TimeSeriesOverlay`
  - Validate `times` are finite (NaN not allowed in times)
  - Validate no Inf values in `data` (NaN allowed for gaps)
  - Precompute `start_indices` and `end_indices` using `np.searchsorted`
  - Compute `global_vmin`, `global_vmax` from finite values
  - Apply normalization if `normalize=True`
  - Return `TimeSeriesData` instance

### 1.4 OverlayData Extension

- [x] Add `timeseries: list[TimeSeriesData] = field(default_factory=list)` to `OverlayData`

### 1.5 Update _convert_overlays_to_data

- [x] Add `timeseries_data_list: list[TimeSeriesData] = []` initialization
- [x] Add dispatch branch: `elif isinstance(internal_data, TimeSeriesData): timeseries_data_list.append(internal_data)`
- [x] Update error message to include `TimeSeriesData`
- [x] Add `timeseries=timeseries_data_list` to `OverlayData` constructor

### 1.6 Export

- [x] Add `TimeSeriesOverlay` and `TimeSeriesData` to `src/neurospatial/animation/__init__.py` exports

### 1.7 Tests for Phase 1

- [x] Create `tests/animation/test_timeseries_overlay.py` (24 tests total)
- [x] Test: `test_timeseries_overlay_creation` - valid construction with all parameters
- [x] Test: `test_timeseries_overlay_validation` - parameter validation errors
- [x] Test: `test_timeseries_convert_to_data` - conversion produces correct indices
- [x] Test: `test_timeseries_window_extraction` - `get_window_slice()` returns correct data
- [x] Test: `test_timeseries_cursor_value_linear` - linear interpolation works
- [x] Test: `test_timeseries_cursor_value_nearest` - nearest interpolation works
- [x] Test: `test_timeseries_nan_handling` - NaN in data creates gaps
- [x] Test: `test_timeseries_normalization` - normalize scales to [0, 1]

**Success criteria for Phase 1**:
- [x] `TimeSeriesOverlay` can be instantiated and validated
- [x] `convert_to_data()` produces `TimeSeriesData` with correct precomputed indices
- [x] All Phase 1 tests pass (24 tests passing)

---

## Phase 2: Napari Backend

**Goal**: Display time series in a dock widget that updates with frame slider.

**Files to modify/create**:
- `src/neurospatial/animation/_timeseries.py` (NEW)
- `src/neurospatial/animation/backends/napari_backend.py`

### 2.1 TimeSeriesArtistManager

- [x] Create `src/neurospatial/animation/_timeseries.py`
- [x] Implement `TimeSeriesArtistManager` dataclass:
  - Fields: `axes`, `lines`, `cursors`, `value_texts`, `frame_times`, `group_window_seconds`
  - `create()` classmethod that builds matplotlib figure and artists once
  - `update(frame_idx, timeseries_data)` method for efficient per-frame updates
- [x] Implement `_group_timeseries()` helper to organize overlays by group
- [x] Implement `_get_group_index()` helper
- [x] Style for napari dark theme (background #262930, white text)

### 2.2 Group Conflict Detection

- [x] Add warning when same group has different `window_seconds`
- [x] Add warning when same group has mixed `normalize` settings
- [x] Use "first wins" strategy for conflicting parameters

### 2.3 Napari Dock Widget

- [x] Implement `_add_timeseries_dock()` in `napari_backend.py`
  - Create `FigureCanvasQTAgg` with matplotlib figure
  - Wrap in `QWidget` with `QVBoxLayout`
  - Add dock widget to viewer's right area
  - Connect to `viewer.dims.events.current_step`
- [x] Add update throttling (max 40 Hz) to prevent matplotlib bottleneck
- [x] Initial render on frame 0

### 2.4 Integration with render_napari

- [x] Update `render_napari()` to check for `overlay_data.timeseries`
- [x] Call `_add_timeseries_dock()` if time series overlays present
- [x] Ensure proper cleanup on viewer close (handled by napari via QWidget lifecycle)

### 2.5 Tests for Phase 2

- [x] Test: `test_timeseries_artist_manager_create` - manager creation with mock figure
- [x] Test: `test_timeseries_artist_manager_update` - update changes line data
- [x] Test: `test_timeseries_grouping_stacked` - ungrouped overlays create separate rows
- [x] Test: `test_timeseries_grouping_overlaid` - same-group overlays share axes
- [x] Test: `test_timeseries_group_conflict_warning` - warnings emitted for conflicts
- [x] Test: `test_timeseries_napari_dock` (integration) - dock widget created in viewer

**Success criteria for Phase 2**:
- [x] `TimeSeriesOverlay` displays in napari dock widget
- [x] Plot updates when scrubbing through frames
- [x] Multiple time series can be stacked or overlaid
- [x] Cursor line shows current time position
- [x] Cursor value text updates with interpolated value

---

## Phase 3: Video Backend

**Goal**: Include time series column in exported video files.

**Files to modify**:
- `src/neurospatial/animation/backends/video_backend.py`
- `src/neurospatial/animation/_timeseries.py`
- `src/neurospatial/animation/_parallel.py`

### 3.1 GridSpec Layout

- [x] Implement `_setup_video_figure_with_timeseries()` function
  - Use `GridSpec` with width_ratios=[3, 1] (spatial:timeseries)
  - Spatial field spans all rows on left
  - Time series axes stacked on right
  - Return `(fig, ax_field, manager)`

### 3.2 create_from_axes Method

- [x] Add `create_from_axes()` classmethod to `TimeSeriesArtistManager`
  - Accept existing axes instead of creating new figure
  - Reuse same artist creation logic
  - Support light theme (video exports)

### 3.3 Video Render Integration

- [x] Update `_render_worker_frames()` to call `manager.update()` when time series present
- [x] Update figure creation to use GridSpec layout when time series present
- [x] Ensure aspect ratio accounts for added column (wider figsize: 12x6)
- [x] Pass `frame_times` parameter through rendering pipeline

### 3.4 Parallel Rendering Support

- [x] Ensure `TimeSeriesData` is pickle-safe (dataclass with numpy arrays)
- [x] Ensure `OverlayData.timeseries` is pickle-safe
- [x] Test with `n_workers > 1`

### 3.5 Tests for Phase 3

- [x] Test: `test_setup_video_figure_with_timeseries_creates_gridspec` - GridSpec creates correct layout
- [x] Test: `test_setup_video_figure_with_multiple_timeseries` - multiple time series rows
- [x] Test: `test_setup_video_figure_light_theme` - light theme for video backend
- [x] Test: `test_create_from_axes_basic` - create_from_axes with pre-created axes
- [x] Test: `test_create_from_axes_multiple_groups` - multiple groups with pre-created axes
- [x] Test: `test_video_render_includes_timeseries_column` - video export works
- [x] Test: `test_video_render_parallel_with_timeseries` - parallel rendering works
- [x] Test: `test_timeseries_data_is_pickle_safe` - TimeSeriesData is pickle-safe
- [x] Test: `test_overlay_data_with_timeseries_is_pickle_safe` - OverlayData is pickle-safe

**Success criteria for Phase 3**:
- [x] Video export includes time series column on right
- [x] Time series updates correctly per frame
- [x] Parallel rendering works without errors

---

## Phase 4: Widget Backend

**Goal**: Display time series in Jupyter notebook widget.

**Files to modify**:
- `src/neurospatial/animation/backends/widget_backend.py`

### 4.1 Widget Layout

- [x] Update widget figure creation to include time series subplot
- [x] Use similar GridSpec approach as video backend

### 4.2 Widget Render Integration

- [x] Update rendering loop to call `manager.update()`
- [x] Ensure smooth playback with time series
- [x] Add `frame_times` parameter to `render_widget()`
- [x] Add `frame_times` parameter to `render_field_to_png_bytes_with_overlays()`
- [x] Add `frame_times` parameter to `PersistentFigureRenderer`

### 4.3 Tests for Phase 4

- [x] Test: `test_render_field_to_png_bytes_with_timeseries` - function includes time series
- [x] Test: `test_persistent_figure_renderer_with_timeseries` - persistent renderer handles time series
- [x] Test: `test_persistent_figure_renderer_timeseries_updates` - updates work across frames
- [x] Test: `test_persistent_figure_renderer_multiple_timeseries` - multiple stacked time series
- [x] Test: `test_render_widget_with_timeseries` - widget integration test
- [x] Test: `test_widget_backend_no_timeseries_still_works` - backward compatibility

**Success criteria for Phase 4**:
- [x] Widget displays time series alongside spatial field
- [x] Animation playback updates both spatial field and time series

---

## Phase 5: Polish & Performance

**Goal**: Edge cases, performance optimization, documentation.

**Files to modify**:
- `src/neurospatial/animation/backends/html_backend.py`
- `CLAUDE.md`
- Various test files

### 5.1 HTML Backend (Optional)

- [x] ~~Add `TimeSeriesDataJSON` TypedDict to HTML backend~~ (not needed - using warning approach)
- [x] ~~Include precomputed indices in serialized data~~ (not needed - using warning approach)
- [x] ~~Implement JavaScript time series renderer (if feasible)~~ (not needed - using warning approach)
- [x] Or: Add warning that HTML backend doesn't support TimeSeriesOverlay
  - Implemented warning similar to VideoOverlay pattern
  - Tests: `TestHTMLBackendTimeSeriesWarning` class (4 tests)

### 5.2 Performance Testing

- [x] Test: `test_timeseries_high_rate_data` - 1 kHz data over 1 hour
  - Tests conversion from TimeSeriesOverlay to TimeSeriesData with 3.6M samples
- [x] Profile rendering loop for bottlenecks
  - `test_timeseries_artist_manager_update_performance`: < 10ms per frame verified
- [x] Verify O(1) window extraction with precomputed indices
  - `test_timeseries_window_extraction_is_o1`: Large data < 10x small data time
  - `test_timeseries_data_precomputed_indices`: Index access is O(1)

### 5.3 Edge Case Tests

- [x] Test: `test_timeseries_empty_window` - frame time outside data range
  - `test_timeseries_empty_window_frame_outside_data`
- [x] Test: `test_timeseries_partial_window` - window partially outside data
  - `test_timeseries_partial_window_start`, `test_timeseries_partial_window_end`
- [x] Test: `test_timeseries_single_point` - single data point
  - `test_timeseries_single_data_point`, `test_timeseries_two_data_points`
- [x] Test: `test_timeseries_all_nan` - all NaN data
  - `test_timeseries_all_nan`, `test_timeseries_with_nan_values`
- [x] Test: `test_timeseries_mismatched_rates` - field and timeseries at different rates
  - `test_timeseries_mismatched_rates_high_rate_ts`, `test_timeseries_mismatched_rates_low_rate_ts`
- [x] Additional edge case tests:
  - `test_timeseries_irregular_times` - irregular time spacing
  - `test_timeseries_conversion_with_edge_frame_times` - boundary frame times
  - `test_timeseries_conversion_empty_overlay_list` - empty overlay handling

### 5.4 Documentation

- [x] Update `CLAUDE.md` Quick Reference with `TimeSeriesOverlay` examples
  - Added after EventOverlay section (v0.14.0+)
- [x] Add usage examples for single variable, stacked rows, overlaid groups
  - Single: speed overlay with window and cursor
  - Stacked: multiple ungrouped overlays in separate rows
  - Overlaid: same group parameter for shared axes
  - Normalized: scales to [0, 1] for comparison
  - Fixed limits: vmin/vmax for manual control
- [x] Document coordinate conventions (if any)
  - N/A: TimeSeriesOverlay uses data values, not spatial coordinates
- [x] Add entry to "Backend capability matrix"
  - Updated both in Quick Reference and Common Gotchas sections
  - Napari/Video/Widget: TimeSeriesOverlay ✓
  - HTML: TimeSeriesOverlay skipped with warning ⚠️
- [x] Add troubleshooting section for common issues
  - HTML backend not supporting TimeSeriesOverlay
  - frame_times parameter required
  - Data and times must be same length

### 5.5 Final Integration Tests

- [x] Test: `test_timeseries_with_position_overlay` - multiple overlay types together
- [x] Test: `test_timeseries_with_event_overlay` - time series + event overlay
- [x] Test: `test_timeseries_full_workflow` - end-to-end with real-ish data
  - `test_timeseries_full_workflow_video_backend`
  - `test_timeseries_full_workflow_multiple_stacked`
  - `test_timeseries_full_workflow_grouped`
  - `test_timeseries_full_workflow_normalized`

**Success criteria for Phase 5**:
- [x] Performance acceptable for 1 kHz data over 1-hour sessions
- [x] All edge cases handled gracefully
- [x] Documentation complete
- [x] All tests pass

---

## Success Criteria Summary

Copy from PLAN.md for tracking:

- [x] `TimeSeriesOverlay` works with napari backend
- [x] Multiple time series can be stacked as rows
- [x] Multiple time series can be overlaid with `group` parameter
- [x] Scrolling window centered on current frame
- [x] No downsampling - full resolution preserved
- [x] Video export includes time series column
- [x] Widget backend includes time series
- [x] Performance acceptable for 1 kHz data over 1-hour sessions
- [x] HTML backend emits warning (TimeSeriesOverlay not supported)
- [x] All edge cases handled gracefully
- [x] Documentation complete in CLAUDE.md

---

## Dependencies & Notes

### External Dependencies
- `matplotlib` (already a dependency) - for time series rendering
- `qtpy` (napari dependency) - for dock widget

### Code Location References
- Existing overlay classes: `src/neurospatial/animation/overlays.py:1-280`
- `_convert_overlays_to_data()`: `src/neurospatial/animation/overlays.py:2900-2970`
- `OverlayData`: `src/neurospatial/animation/overlays.py:~2770`
- Napari backend: `src/neurospatial/animation/backends/napari_backend.py`
- Video backend: `src/neurospatial/animation/backends/video_backend.py`
- Widget backend: `src/neurospatial/animation/backends/widget_backend.py`

### Design Decisions (from PLAN.md)
- **NaN handling**: Creates gaps in line (matplotlib default behavior)
- **Group conflicts**: Warning + first wins (not strict errors)
- **Y-axis limits**: Global by default for stable scales
- **Update throttling**: Max 40 Hz to prevent matplotlib bottleneck
- **Frame alignment**: Precompute indices in `convert_to_data()`, O(1) per-frame access
