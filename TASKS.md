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

- [ ] Create `src/neurospatial/animation/_timeseries.py`
- [ ] Implement `TimeSeriesArtistManager` dataclass:
  - Fields: `axes`, `lines`, `cursors`, `value_texts`, `frame_times`, `group_window_seconds`
  - `create()` classmethod that builds matplotlib figure and artists once
  - `update(frame_idx, timeseries_data)` method for efficient per-frame updates
- [ ] Implement `_group_timeseries()` helper to organize overlays by group
- [ ] Implement `_get_group_index()` helper
- [ ] Style for napari dark theme (background #262930, white text)

### 2.2 Group Conflict Detection

- [ ] Add warning when same group has different `window_seconds`
- [ ] Add warning when same group has mixed `normalize` settings
- [ ] Use "first wins" strategy for conflicting parameters

### 2.3 Napari Dock Widget

- [ ] Implement `_add_timeseries_dock()` in `napari_backend.py`
  - Create `FigureCanvasQTAgg` with matplotlib figure
  - Wrap in `QWidget` with `QVBoxLayout`
  - Add dock widget to viewer's right area
  - Connect to `viewer.dims.events.current_step`
- [ ] Add update throttling (max 40 Hz) to prevent matplotlib bottleneck
- [ ] Initial render on frame 0

### 2.4 Integration with render_napari

- [ ] Update `render_napari()` to check for `overlay_data.timeseries`
- [ ] Call `_add_timeseries_dock()` if time series overlays present
- [ ] Ensure proper cleanup on viewer close

### 2.5 Tests for Phase 2

- [ ] Test: `test_timeseries_artist_manager_create` - manager creation with mock figure
- [ ] Test: `test_timeseries_artist_manager_update` - update changes line data
- [ ] Test: `test_timeseries_grouping_stacked` - ungrouped overlays create separate rows
- [ ] Test: `test_timeseries_grouping_overlaid` - same-group overlays share axes
- [ ] Test: `test_timeseries_group_conflict_warning` - warnings emitted for conflicts
- [ ] Test: `test_timeseries_napari_dock` (integration) - dock widget created in viewer

**Success criteria for Phase 2**:
- [ ] `TimeSeriesOverlay` displays in napari dock widget
- [ ] Plot updates when scrubbing through frames
- [ ] Multiple time series can be stacked or overlaid
- [ ] Cursor line shows current time position
- [ ] Cursor value text updates with interpolated value

---

## Phase 3: Video Backend

**Goal**: Include time series column in exported video files.

**Files to modify**:
- `src/neurospatial/animation/backends/video_backend.py`
- `src/neurospatial/animation/_timeseries.py`

### 3.1 GridSpec Layout

- [ ] Implement `_setup_video_figure_with_timeseries()` function
  - Use `GridSpec` with width_ratios=[3, 1] (spatial:timeseries)
  - Spatial field spans all rows on left
  - Time series axes stacked on right
  - Return `(fig, ax_field, manager)`

### 3.2 create_from_axes Method

- [ ] Add `create_from_axes()` classmethod to `TimeSeriesArtistManager`
  - Accept existing axes instead of creating new figure
  - Reuse same artist creation logic
  - Support light theme (video exports)

### 3.3 Video Render Integration

- [ ] Update `_render_frame()` to call `manager.update()` when time series present
- [ ] Update `_setup_figure()` to handle time series layout
- [ ] Ensure aspect ratio accounts for added column

### 3.4 Parallel Rendering Support

- [ ] Ensure `TimeSeriesArtistManager` is pickle-safe
- [ ] Test with `n_workers > 1`

### 3.5 Tests for Phase 3

- [ ] Test: `test_timeseries_video_layout` - GridSpec creates correct layout
- [ ] Test: `test_timeseries_video_render` - video export includes time series
- [ ] Test: `test_timeseries_video_parallel` - parallel rendering works

**Success criteria for Phase 3**:
- [ ] Video export includes time series column on right
- [ ] Time series updates correctly per frame
- [ ] Parallel rendering works without errors

---

## Phase 4: Widget Backend

**Goal**: Display time series in Jupyter notebook widget.

**Files to modify**:
- `src/neurospatial/animation/backends/widget_backend.py`

### 4.1 Widget Layout

- [ ] Update widget figure creation to include time series subplot
- [ ] Use similar GridSpec approach as video backend

### 4.2 Widget Render Integration

- [ ] Update rendering loop to call `manager.update()`
- [ ] Ensure smooth playback with time series

### 4.3 Tests for Phase 4

- [ ] Test: `test_timeseries_widget_render` - widget includes time series

**Success criteria for Phase 4**:
- [ ] Widget displays time series alongside spatial field
- [ ] Animation playback updates both spatial field and time series

---

## Phase 5: Polish & Performance

**Goal**: Edge cases, performance optimization, documentation.

**Files to modify**:
- `src/neurospatial/animation/backends/html_backend.py`
- `CLAUDE.md`
- Various test files

### 5.1 HTML Backend (Optional)

- [ ] Add `TimeSeriesDataJSON` TypedDict to HTML backend
- [ ] Include precomputed indices in serialized data
- [ ] Implement JavaScript time series renderer (if feasible)
- [ ] Or: Add warning that HTML backend doesn't support TimeSeriesOverlay

### 5.2 Performance Testing

- [ ] Test: `test_timeseries_high_rate_data` - 1 kHz data over 1 hour
- [ ] Profile rendering loop for bottlenecks
- [ ] Verify O(1) window extraction with precomputed indices

### 5.3 Edge Case Tests

- [ ] Test: `test_timeseries_empty_window` - frame time outside data range
- [ ] Test: `test_timeseries_partial_window` - window partially outside data
- [ ] Test: `test_timeseries_single_point` - single data point
- [ ] Test: `test_timeseries_all_nan` - all NaN data
- [ ] Test: `test_timeseries_mismatched_rates` - field and timeseries at different rates

### 5.4 Documentation

- [ ] Update `CLAUDE.md` Quick Reference with `TimeSeriesOverlay` examples
- [ ] Add usage examples for single variable, stacked rows, overlaid groups
- [ ] Document coordinate conventions (if any)
- [ ] Add entry to "Backend capability matrix"
- [ ] Add troubleshooting section for common issues

### 5.5 Final Integration Tests

- [ ] Test: `test_timeseries_with_position_overlay` - multiple overlay types together
- [ ] Test: `test_timeseries_with_video_overlay` - time series + video overlay
- [ ] Test: `test_timeseries_full_workflow` - end-to-end with real-ish data

**Success criteria for Phase 5**:
- [ ] Performance acceptable for 1 kHz data over 1-hour sessions
- [ ] All edge cases handled gracefully
- [ ] Documentation complete
- [ ] All tests pass

---

## Success Criteria Summary

Copy from PLAN.md for tracking:

- [ ] `TimeSeriesOverlay` works with napari backend
- [ ] Multiple time series can be stacked as rows
- [ ] Multiple time series can be overlaid with `group` parameter
- [ ] Scrolling window centered on current frame
- [ ] No downsampling - full resolution preserved
- [ ] Video export includes time series column
- [ ] Widget backend includes time series
- [ ] Performance acceptable for 1 kHz data over 1-hour sessions

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
