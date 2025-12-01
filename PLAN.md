# Napari Playback Performance Optimization Plan

**Goal**: Get napari playback smooth while using all overlays (position, bodyparts+skeleton, head direction, events/spikes, time series, video).

**Last Updated**: 2025-12-01

---

## Pre-Work: Current State Analysis

### Already Optimized (No Changes Needed)

Based on codebase analysis, these components are already optimized for napari's native rendering:

| Component | Current Implementation | Status |
|-----------|----------------------|--------|
| **Position overlay** | Tracks layer with time axis | Native napari handling |
| **Bodypart overlay** | Points layer with time axis | Native napari handling |
| **Skeleton overlay** | Pre-computed Vectors layer (`_build_skeleton_vectors()`) | No per-frame callbacks |
| **Head direction** | Tracks layer (2-point lines per frame) | Native napari handling |
| **Events (instant)** | 3D Points layer `(time, y, x)` format | No callbacks |
| **Events (cumulative/decay)** | Boolean `shown` mask with `O(log N)` searchsorted | Efficient updates |
| **Field rendering** | LRU cache with lazy loading | On-demand rendering |
| **Widget updates** | Throttled to 10 Hz | Reduced Qt overhead |

### Needs Optimization

| Component | Current Issue | Impact |
|-----------|--------------|--------|
| **Video overlay** | Per-frame `layer.data = frame` assignment | ~2-3ms per frame |
| **Time series dock** | Matplotlib drawing even with blitting | ~5-15ms per frame |
| **No central playback controller** | Scattered `dims.events.current_step` callbacks | Hard to implement frame skipping |
| **No frame skipping** | Must render every frame even if behind schedule | Stutters at high fps |

---

## Phase 0: Establish Performance Baseline

### 0.1: Set Up Performance Monitoring

Before any changes, measure baseline performance using napari's perfmon.

**Tasks**

- [ ] Create benchmark script `scripts/benchmark_napari_playback.py`
- [ ] Test with each overlay individually:
  - Position only
  - Bodyparts + skeleton
  - Head direction
  - Events (instant vs cumulative)
  - Video
  - Time series
- [ ] Test with all overlays combined
- [ ] Record metrics for each test:
  - Frame render time (target: <10ms)
  - Qt paint time (target: <15-20ms)
  - Callback overhead per frame

**Implementation**

```bash
# Enable perfmon before running
NAPARI_PERFMON=1 uv run python scripts/benchmark_napari_playback.py

# Or with config file for detailed tracing
NAPARI_PERFMON=/path/to/perfmon_config.json uv run python scripts/benchmark_napari_playback.py
```

**perfmon_config.json**:
```json
{
    "trace_qt_events": true,
    "trace_file_on_start": "/tmp/napari_trace_baseline.json",
    "trace_callables": ["napari_callbacks"],
    "callable_lists": {
        "napari_callbacks": [
            "neurospatial.animation.backends.napari_backend._make_video_frame_callback",
            "neurospatial.animation.backends.napari_backend._render_timeseries_dock"
        ]
    }
}
```

**Viewing Results**:
- Open Chrome: `chrome://tracing`
- Load `/tmp/napari_trace_baseline.json`
- Look for:
  - `MetaCall:QObject` (Qt event processing)
  - `Paint:CanvasBackendDesktop` (GPU rendering)
  - Any custom events from trace_callables

---

## Phase 1: Introduce PlaybackController (Central Control Point)

### 1.1: Create PlaybackController Class

Currently `render_napari` passes `fps` to napari's built-in playback, and overlays attach their own `dims.events.current_step` callbacks. This makes frame skipping impossible.

**Goal**: Centralize playback into one controller that:
- Owns the `QTimer` and playback rate
- Owns the mapping `frame_idx -> frame_time`
- Is the *only* place that calls `viewer.dims.set_current_step(...)`
- Can skip frames if falling behind

**Tasks**

- [ ] Create `PlaybackController` class in `napari_backend.py`:
  ```python
  class PlaybackController:
      """Central playback controller for frame-skipping-aware animation."""

      def __init__(
          self,
          viewer: napari.Viewer,
          n_frames: int,
          fps: float,
          frame_times: NDArray[np.float64] | None = None,
      ):
          self.viewer = viewer
          self.n_frames = n_frames
          self.fps = fps
          self.frame_times = frame_times
          self._current_frame = 0
          self._playing = False
          self._start_time: float | None = None
          self._start_frame: int = 0
          self._timer: QTimer | None = None
          self._callbacks: list[Callable[[int], None]] = []

      def go_to_frame(self, frame_idx: int) -> None:
          """Jump to specific frame, updating viewer dims."""
          self._current_frame = max(0, min(frame_idx, self.n_frames - 1))
          self.viewer.dims.set_current_step(0, self._current_frame)
          # Notify registered callbacks
          for callback in self._callbacks:
              callback(self._current_frame)

      def step(self) -> None:
          """Advance to next frame with skipping if behind schedule."""
          if not self._playing or self._start_time is None:
              return

          # Calculate target frame based on elapsed time
          elapsed = time.perf_counter() - self._start_time
          target_frame = self._start_frame + int(elapsed * self.fps)

          # Skip directly to target (frame dropping)
          if target_frame > self._current_frame:
              self.go_to_frame(target_frame)

          # Check for end
          if self._current_frame >= self.n_frames - 1:
              self.pause()

      def play(self) -> None:
          """Start playback."""
          self._playing = True
          self._start_time = time.perf_counter()
          self._start_frame = self._current_frame
          # Start timer at target fps
          if self._timer is None:
              from qtpy.QtCore import QTimer
              self._timer = QTimer()
              self._timer.timeout.connect(self.step)
          self._timer.start(int(1000 / self.fps))

      def pause(self) -> None:
          """Pause playback."""
          self._playing = False
          if self._timer:
              self._timer.stop()

      def register_callback(self, callback: Callable[[int], None]) -> None:
          """Register callback to be notified on frame changes."""
          self._callbacks.append(callback)
  ```

- [ ] Integrate `PlaybackController` into `render_napari()`:
  - Create controller after viewer setup
  - Store in `viewer.metadata["playback_controller"]`
  - Wire custom play/pause widget to controller instead of napari's built-in

- [ ] Keep existing `dims.events.current_step` callbacks for now (Phase 2 will migrate them)

**Why This Helps**:
- Frame skipping happens in `step()` when falling behind
- All frame changes go through `go_to_frame()` which can throttle/batch updates
- Callbacks registered here can be managed/throttled centrally

---

## Phase 2: Optimize Video Overlay

### 2.1: Use Time-Indexed Image Layer (If In-Memory)

Currently video updates via `layer.data = frame` on every frame change. For in-memory arrays, napari can handle time dimension natively.

**Tasks**

- [ ] In `_add_video_layer()`, detect if source is in-memory ndarray
- [ ] If in-memory with shape `(n_video_frames, H, W, 3)`:
  - Create Image layer with full array and time dimension
  - Set `viewer.dims.ndim` to include video time axis
  - Let napari handle frame selection natively (no callback)
- [ ] If file-based or streaming:
  - Keep current callback approach but add frame caching ring buffer

**Implementation for in-memory**:
```python
if isinstance(video_data.reader, np.ndarray):
    # Full array available - use napari's native time handling
    video_array = video_data.reader  # (n_frames, H, W, 3)
    layer = viewer.add_image(
        video_array,
        name=name,
        rgb=True,
        opacity=video_data.alpha,
        affine=affine,
        blending="translucent",
    )
    # No callback needed - napari handles dims[0] time axis
```

### 2.2: Add Ring Buffer for File-Based Video

**Tasks**

- [ ] Create `VideoFrameCache` class with LRU ring buffer:
  ```python
  class VideoFrameCache:
      def __init__(self, reader, cache_size: int = 100):
          self.reader = reader
          self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
          self.cache_size = cache_size

      def get_frame(self, idx: int) -> np.ndarray:
          if idx in self.cache:
              self.cache.move_to_end(idx)
              return self.cache[idx]

          frame = self.reader[idx]
          if len(self.cache) >= self.cache_size:
              self.cache.popitem(last=False)  # Remove oldest
          self.cache[idx] = frame
          return frame
  ```

- [ ] Use cache in video callback to avoid redundant seeks

---

## Phase 3: Optimize Time Series Dock

### 3.1: Add Update Mode Option

Currently time series updates on every frame with blitting. Even blitting costs ~5-15ms.

**Tasks**

- [ ] Add `update_mode` parameter to `TimeSeriesOverlay`:
  - `"live"` (default): Update every N frames (already throttled to 20 Hz max)
  - `"on_pause"`: Only update when playback pauses
  - `"manual"`: Only update via explicit call

- [ ] In `_render_timeseries_dock()`, respect update mode:
  ```python
  def on_frame_change(event):
      if update_mode == "on_pause" and controller.is_playing:
          return  # Skip during playback
      # ... existing update logic
  ```

- [ ] Wire to `PlaybackController` play/pause events for `on_pause` mode

### 3.2: Reduce Matplotlib Draw Calls

**Tasks**

- [ ] Increase throttle frequency during playback (currently 20 Hz, consider 10 Hz)
- [ ] Use `ax.set_xlim()` only when window actually changes (cache last window bounds)
- [ ] Consider using pyqtgraph instead of matplotlib for <1ms updates (optional, major change)

---

## Phase 4: Playback Scheduling and Frame Skipping

### 4.1: Integrate Frame Skipping

With `PlaybackController` from Phase 1, enable automatic frame skipping.

**Tasks**

- [ ] `PlaybackController.step()` already calculates `target_frame` based on elapsed time
- [ ] Add metrics tracking:
  ```python
  self._frames_rendered = 0
  self._frames_skipped = 0
  ```
- [ ] Add option to disable skipping for testing:
  ```python
  def __init__(..., allow_frame_skip: bool = True):
      self.allow_frame_skip = allow_frame_skip
  ```

### 4.2: Handle Rapid Scrubbing

When user drags slider quickly, process only the latest requested frame.

**Tasks**

- [ ] In `go_to_frame()`, if called rapidly (within 16ms), debounce:
  ```python
  def go_to_frame(self, frame_idx: int) -> None:
      self._pending_frame = frame_idx
      if not self._debounce_timer.isActive():
          self._debounce_timer.start(16)  # ~60 Hz max update rate

  def _apply_pending_frame(self) -> None:
      if self._pending_frame is not None:
          self._current_frame = self._pending_frame
          self.viewer.dims.set_current_step(0, self._current_frame)
          self._pending_frame = None
  ```

---

## Phase 5: Clean Up Event Wiring

### 5.1: Audit and Remove Redundant Callbacks

**Tasks**

- [ ] Search for all `viewer.dims.events.current_step.connect(...)` calls
- [ ] For each callback, determine if it can be:
  - Removed (already using native time dimension)
  - Migrated to `PlaybackController.register_callback()`
  - Throttled via central controller

**Current callbacks to audit**:
| Location | Callback | Action |
|----------|----------|--------|
| `_make_video_frame_callback` | `update_video_frames` | Keep for file-based, remove for in-memory |
| `_render_event_overlay` | `on_frame_change` (cumulative/decay) | Keep (already efficient) |
| `_render_timeseries_dock` | `on_frame_change` | Migrate to controller for `on_pause` mode |

### 5.2: Remove Deprecated layer.data Assignments

**Tasks**

- [ ] Ensure no overlay does `layer.data = large_array` in callbacks
- [ ] Current audit:
  - Video: `layer.data = frame` (optimize in Phase 2)
  - Events (cumulative): `layer.shown = mask` (efficient, keep)
  - Others: Use native time dimension (no data reassignment)

---

## Phase 6: Verification and Profiling

### 6.1: Create Automated Benchmark Suite

**Tasks**

- [ ] Create `tests/benchmarks/test_napari_playback.py` with pytest-benchmark:
  ```python
  @pytest.mark.benchmark
  def test_playback_position_only(benchmark, env, fields, positions):
      """Benchmark playback with position overlay."""
      overlay = PositionOverlay(data=positions, color="red")
      viewer = env.animate_fields(
          fields, frame_times=frame_times,
          backend="napari", overlays=[overlay]
      )
      benchmark(lambda: advance_n_frames(viewer, 100))
  ```

- [ ] Test configurations:
  - Each overlay type individually
  - All overlays combined
  - Different field sizes (100x100, 500x500, 1000x1000)
  - Different frame counts (100, 1000, 10000)

### 6.2: Performance Targets

| Metric | Target | Acceptable |
|--------|--------|------------|
| Per-frame callback total | <5ms | <10ms |
| Canvas paint time | <10ms | <15ms |
| Total frame latency | <16ms (60 fps) | <40ms (25 fps) |
| Frame skip rate at 25 fps | <5% | <15% |

### 6.3: Manual Testing Checklist

- [ ] Playback smooth at 25 fps with all overlays
- [ ] Scrubbing responsive (no lag when dragging slider)
- [ ] No memory growth during extended playback
- [ ] Frame counter updates correctly (not missing frames)
- [ ] Time series plot updates smoothly (or correctly pauses during playback)

---

## Implementation Order

**Recommended sequence**:

1. **Phase 0.1**: Baseline measurement (1-2 hours)
2. **Phase 1.1**: PlaybackController (2-3 hours)
3. **Phase 4.1-4.2**: Frame skipping (1-2 hours)
4. **Phase 2.1-2.2**: Video optimization (2-3 hours)
5. **Phase 3.1-3.2**: Time series optimization (1-2 hours)
6. **Phase 5.1-5.2**: Cleanup (1 hour)
7. **Phase 6.1-6.3**: Verification (1-2 hours)

**Total estimated effort**: 10-15 hours

---

## Appendix: perfmon Usage Reference

### Enable perfmon

```bash
# Simple enable
NAPARI_PERFMON=1 uv run python script.py

# With config file
NAPARI_PERFMON=perfmon_config.json uv run python script.py
```

### Create perfmon config

```json
{
    "trace_qt_events": true,
    "trace_file_on_start": "/tmp/trace.json",
    "trace_callables": ["my_functions"],
    "callable_lists": {
        "my_functions": [
            "module.Class.method"
        ]
    }
}
```

### View traces

1. Open Chrome
2. Navigate to `chrome://tracing`
3. Click "Load" and select trace JSON file
4. Use WASD keys to navigate timeline

### Key metrics to watch

- `MetaCall:QObject`: Qt event processing overhead
- `Paint:CanvasBackendDesktop`: GPU rendering time
- Custom traced functions: Callback execution time

### Add custom timing in code

```python
from napari.utils.perf import perf_timer, add_instant_event

# Time a code block
with perf_timer("my_operation"):
    do_something()

# Mark an event
add_instant_event("frame_rendered")
```

---

## Notes

### What NOT to Change

The following are already optimized and should not be modified:

1. **Skeleton precomputation** (`_build_skeleton_vectors`): Already builds full Vectors layer at init
2. **Event instant mode**: Already uses native 3D Points format
3. **Position/Head direction overlays**: Already use Tracks with time dimension
4. **Field caching**: LRU cache with lazy loading is appropriate for large datasets

### Future Considerations (Out of Scope)

- Replace matplotlib with pyqtgraph for time series (major change, minimal benefit vs Phase 3)
- WebGL-based playback for web deployment
- Multi-GPU rendering for very large environments
