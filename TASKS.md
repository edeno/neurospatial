# TASKS.md - Napari Playback Performance Optimization

**Goal**: Smooth napari playback with all overlays (position, bodyparts+skeleton, head direction, events/spikes, time series, video)

**Reference**: [PLAN.md](./PLAN.md) for detailed implementation notes and code snippets

---

## Phase 0: Establish Performance Baseline

**Purpose**: Measure current performance to identify bottlenecks and track improvements

### 0.1: Set Up Performance Monitoring

- [x] Create `scripts/benchmark_napari_playback.py`
  - Generate synthetic test data (environment, fields, positions)
  - Support command-line args for overlay selection
  - Print timing metrics to stdout
- [x] Create `scripts/perfmon_config.json` for detailed tracing
  - Enable `trace_qt_events: true`
  - Trace video callback and time series dock functions
- [ ] Run baseline measurements:
  - [ ] Position overlay only → record frame time
  - [ ] Bodyparts + skeleton → record frame time
  - [ ] Head direction → record frame time
  - [ ] Events (instant vs cumulative) → record frame time
  - [ ] Video overlay → record frame time
  - [ ] Time series dock → record frame time
  - [ ] **All overlays combined** → record frame time
- [ ] Document baseline in `docs/performance_baseline.md`
  - Target: <10ms per frame render, <15-20ms total with Qt paint

**Success Criteria**: Documented baseline with per-overlay timing breakdown

---

## Phase 1: Introduce PlaybackController

**Purpose**: Centralize playback control to enable frame skipping and coordinated updates

**Depends on**: Phase 0 (for baseline comparison)

### 1.1: Create PlaybackController Class

- [ ] Create `PlaybackController` in `src/neurospatial/animation/backends/napari_backend.py`
  - Properties: `viewer`, `n_frames`, `fps`, `frame_times`, `_current_frame`
  - Timer: Own QTimer instead of using napari's built-in
- [ ] Implement core methods:
  - [ ] `go_to_frame(frame_idx)` - jump to specific frame, update viewer dims
  - [ ] `step()` - advance to next frame with elapsed-time-based skipping
  - [ ] `play()` - start playback, record start time
  - [ ] `pause()` - stop playback
  - [ ] `register_callback(fn)` - register frame change callbacks
- [ ] Add frame skip calculation in `step()`:
  - Calculate target frame from elapsed time: `start_frame + int(elapsed * fps)`
  - Skip directly to target if behind schedule

### 1.2: Integrate into render_napari()

- [ ] Create `PlaybackController` after viewer setup in `render_napari()`
- [ ] Store controller in `viewer.metadata["playback_controller"]`
- [ ] Wire existing play/pause widget to controller (instead of napari built-in)
- [ ] Keep existing `dims.events.current_step` callbacks (migration in Phase 5)

**Success Criteria**: Controller created and wired; existing functionality unchanged

---

## Phase 2: Optimize Video Overlay

**Purpose**: Eliminate per-frame `layer.data = frame` overhead for in-memory video

**Depends on**: Phase 1 (for callback migration path)

### 2.1: Use Time-Indexed Image Layer for In-Memory Video

- [ ] In `_add_video_layer()`, detect if source is in-memory ndarray
- [ ] If in-memory `(n_frames, H, W, 3)`:
  - Create Image layer with full array and time dimension
  - Let napari handle frame selection natively (no callback needed)
- [ ] If file-based:
  - Keep current callback approach (optimize in 2.2)

### 2.2: Add Ring Buffer for File-Based Video

- [ ] Create `VideoFrameCache` class
  - LRU OrderedDict with configurable cache size (default 100 frames)
  - `get_frame(idx)` method with cache hit/miss logic
- [ ] Use cache in video frame callback
- [ ] Consider async prefetching for sequential playback (optional enhancement)

**Success Criteria**: In-memory video uses native time dimension; file-based video cached

---

## Phase 3: Optimize Time Series Dock

**Purpose**: Reduce matplotlib drawing overhead during playback

**Depends on**: Phase 1 (for PlaybackController integration)

### 3.1: Add Update Mode Option

- [ ] Add `update_mode` parameter to `TimeSeriesOverlay`
  - `"live"` (default): Update every N frames (20 Hz max, already throttled)
  - `"on_pause"`: Only update when playback pauses
  - `"manual"`: Only update via explicit API call
- [ ] Implement mode handling in `_render_timeseries_dock()`:
  - Skip updates during playback if `mode == "on_pause"`
- [ ] Wire to PlaybackController play/pause events

### 3.2: Reduce Matplotlib Draw Calls

- [ ] Increase throttle frequency during playback (consider 10 Hz instead of 20 Hz)
- [ ] Cache last window bounds; only call `ax.set_xlim()` when window changes
- [ ] Profile matplotlib blitting overhead to identify further optimizations

**Success Criteria**: Time series updates configurable; overhead reduced during playback

---

## Phase 4: Playback Scheduling and Frame Skipping

**Purpose**: Enable automatic frame dropping when behind schedule

**Depends on**: Phase 1 (PlaybackController must exist)

### 4.1: Integrate Frame Skipping

- [ ] Add metrics tracking to PlaybackController:
  - `_frames_rendered` counter
  - `_frames_skipped` counter
- [ ] Add `allow_frame_skip: bool = True` parameter to `__init__()`
- [ ] Expose metrics via properties: `frames_rendered`, `frames_skipped`

### 4.2: Handle Rapid Scrubbing

- [ ] Add debounce timer to `go_to_frame()` (16ms = ~60 Hz max)
- [ ] Store `_pending_frame` instead of immediate update
- [ ] Apply pending frame when debounce timer fires
- [ ] Ensure manual seek still feels responsive

**Success Criteria**: Frame skipping works; scrubbing is responsive without stutters

---

## Phase 5: Clean Up Event Wiring

**Purpose**: Remove redundant callbacks and centralize through PlaybackController

**Depends on**: Phases 2, 3, 4 (all optimizations in place)

### 5.1: Audit and Migrate Callbacks

- [ ] Search for all `viewer.dims.events.current_step.connect(...)` calls
- [ ] Audit each callback:

  | Location | Action |
  |----------|--------|
  | `_make_video_frame_callback` | Remove for in-memory; keep for file-based |
  | `_render_event_overlay` (cumulative/decay) | Keep (already efficient) |
  | `_render_timeseries_dock` | Migrate to controller for `on_pause` mode |

- [ ] Migrate appropriate callbacks to `PlaybackController.register_callback()`

### 5.2: Remove Deprecated layer.data Assignments

- [ ] Audit all overlays for `layer.data = large_array` in callbacks
- [ ] Ensure only efficient updates remain:
  - Video: Native time dimension or cached read
  - Events (cumulative): `layer.shown = mask` (efficient, keep)
  - Others: Native time dimension (no reassignment)

**Success Criteria**: All callbacks audited; redundant ones removed

---

## Phase 6: Verification and Profiling

**Purpose**: Confirm optimizations meet performance targets

**Depends on**: All previous phases complete

### 6.1: Create Automated Benchmark Suite

- [ ] Create `tests/benchmarks/test_napari_playback.py`
- [ ] Add pytest-benchmark fixtures:
  - Each overlay type individually
  - All overlays combined
  - Different field sizes (100x100, 500x500, 1000x1000)
  - Different frame counts (100, 1000, 10000)
- [ ] Run benchmarks and compare to Phase 0 baseline

### 6.2: Verify Performance Targets

| Metric | Target | Acceptable |
|--------|--------|------------|
| Per-frame callback total | <5ms | <10ms |
| Canvas paint time | <10ms | <15ms |
| Total frame latency | <16ms (60 fps) | <40ms (25 fps) |
| Frame skip rate at 25 fps | <5% | <15% |

- [ ] All metrics within "Acceptable" range
- [ ] Most metrics within "Target" range

### 6.3: Manual Testing Checklist

- [ ] Playback smooth at 25 fps with all overlays enabled
- [ ] Scrubbing responsive (no perceptible lag when dragging slider)
- [ ] No memory growth during 5+ minute extended playback
- [ ] Frame counter updates correctly (accounting for skipped frames)
- [ ] Time series plot updates smoothly (or pauses correctly during playback)

**Success Criteria**: All targets met; manual tests pass

---

## Implementation Order Summary

Recommended sequence (total: ~10-15 hours):

1. **Phase 0.1**: Baseline measurement
2. **Phase 1.1-1.2**: PlaybackController
3. **Phase 4.1-4.2**: Frame skipping
4. **Phase 2.1-2.2**: Video optimization
5. **Phase 3.1-3.2**: Time series optimization
6. **Phase 5.1-5.2**: Cleanup
7. **Phase 6.1-6.3**: Verification

---

## Notes

### Do Not Modify (Already Optimized)

- Skeleton precomputation (`_build_skeleton_vectors`)
- Event instant mode (native 3D Points format)
- Position/Head direction overlays (Tracks with time dimension)
- Field caching (LRU with lazy loading)

### Out of Scope

- Replace matplotlib with pyqtgraph (major change)
- WebGL-based playback
- Multi-GPU rendering
