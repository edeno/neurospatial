# SCRATCHPAD.md - Napari Performance Optimization

**Started**: 2025-12-01
**Current Phase**: Phase 2.1 Complete - Time-Indexed Video Layer for In-Memory Video

---

## Completed Tasks

### Task: Create `scripts/benchmark_napari_playback.py`
**Status**: COMPLETED (2025-12-01)

**What was implemented**:
- Created benchmark script with synthetic test data generation
- Added command-line arguments for overlay selection:
  - `--position` - Position overlay with trail
  - `--bodyparts` - Bodypart overlay with skeleton
  - `--head-direction` - Head direction overlay
  - `--events` - Event overlay (spike-like events)
  - `--timeseries` - Time series dock widget
  - `--all-overlays` - Enable all overlays
- Added execution modes:
  - `--headless` - For CI testing
  - `--no-playback` - Skip playback timing
- Prints timing metrics to stdout:
  - Setup time
  - Per-frame timing statistics (mean, median, p95, min, max)
  - Performance assessment vs 30 fps target

**Files created/modified**:
- `scripts/benchmark_napari_playback.py` (new) - Main benchmark script
- `tests/animation/test_benchmark_napari_playback.py` (new) - 24 unit tests

**Usage**:
```bash
# Quick test with position overlay
uv run python scripts/benchmark_napari_playback.py --frames 50 --position --headless

# Full benchmark with all overlays
uv run python scripts/benchmark_napari_playback.py --all-overlays --frames 500 --playback-frames 100

# With napari perfmon tracing
NAPARI_PERFMON=/tmp/trace.json uv run python scripts/benchmark_napari_playback.py --all-overlays
```

### Task: Update `scripts/perfmon_config.json` with video/timeseries tracing
**Status**: COMPLETED (2025-12-01)

**What was implemented**:
- Added missing callables to `scripts/perfmon_config.json`:
  - `_make_video_frame_callback` - Video overlay per-frame callback
  - `_add_video_layer` - Video layer setup
  - `_add_timeseries_dock` - Time series dock widget setup
  - `_render_event_overlay` - Event overlay rendering
- Created comprehensive test suite (`tests/animation/test_perfmon_config.py`):
  - 13 tests validating config structure and contents
  - Tests for required fields (trace_qt_events, trace_file_on_start, etc.)
  - Tests ensuring critical functions are traced (video, timeseries, rendering, overlays)
  - Consistency tests (fully qualified names, non-empty lists, etc.)
- Fixed documentation error in `.claude/PROFILING.md`:
  - Corrected `_render_timeseries_dock` to `_add_timeseries_dock`

**Files created/modified**:
- `scripts/perfmon_config.json` (modified) - Added video/timeseries/event callables
- `tests/animation/test_perfmon_config.py` (new) - 13 validation tests
- `.claude/PROFILING.md` (modified) - Fixed function name in example

**Usage**:
```bash
# Run with detailed tracing using config file
NAPARI_PERFMON=scripts/perfmon_config.json uv run python scripts/benchmark_napari_playback.py --all-overlays

# View trace in Chrome
# Open chrome://tracing and load /tmp/napari_neurospatial_trace.json
```

### Task: Run baseline measurements
**Status**: COMPLETED (2025-12-01)

**What was implemented**:
- Added `--video` flag to benchmark script for video overlay testing
- Added `--auto-close` flag for automated benchmarking (closes viewer after timing)
- Ran benchmarks for all 6 overlay types individually and combined
- Created comprehensive baseline documentation in `docs/performance_baseline.md`

**Key Results**:

| Configuration | Mean (ms) | Achievable FPS | Target Met? |
|---------------|-----------|----------------|-------------|
| Position only | 21.87 | ~46 fps | Yes |
| Bodyparts + skeleton | 26.30 | ~38 fps | Yes |
| Head direction | 18.44 | ~54 fps | Yes |
| Events (decay) | 19.20 | ~52 fps | Yes |
| Time series dock | 18.49 | ~54 fps | Yes |
| Video overlay | 18.39 | ~54 fps | Yes |
| **All 6 overlays** | **47.38** | **~21 fps** | **No** |

**Key Finding**: Individual overlays all meet 30 fps target, but combined performance drops to ~21 fps. Optimization needed for multi-overlay use cases.

**Files created/modified**:
- `scripts/benchmark_napari_playback.py` (modified) - Added `--video` and `--auto-close` flags
- `docs/performance_baseline.md` (new) - Comprehensive baseline documentation

### Task: Phase 1.1 - Create PlaybackController class
**Status**: COMPLETED (2025-12-01)

**What was implemented**:
- Created `PlaybackController` class in `napari_backend.py` with:
  - `go_to_frame(frame_idx)` - Jump to frame with clamping and callback notification
  - `step()` - Advance with elapsed-time-based frame skipping
  - `play()` / `pause()` - Playback control
  - `register_callback(fn)` - Callback registration
  - `allow_frame_skip` parameter for testing/video export use cases
  - Metrics tracking: `frames_rendered`, `frames_skipped`
- Comprehensive test suite: 24 tests covering all functionality
- Follows TDD: tests written first, then implementation

**Files created/modified**:
- `src/neurospatial/animation/backends/napari_backend.py` (modified) - Added PlaybackController class
- `tests/animation/test_playback_controller.py` (new) - 24 unit tests

### Task: Phase 1.2 - Integrate PlaybackController into render_napari()
**Status**: COMPLETED (2025-12-01)

**What was implemented**:
- Integrated PlaybackController into `render_napari()`:
  - Creates controller after viewer setup and before overlay rendering
  - Stores controller as `viewer.playback_controller` using `object.__setattr__` to bypass napari's pydantic validation
  - Extracts `frame_times` from `overlay_data` if available
  - Controller is accessible but widget still uses napari's built-in playback (widget wiring comes in Phase 4)
- Added controller to multi-field path (`_render_multi_field_napari()`):
  - Same integration pattern as single-field
  - Computes `n_frames` from first sequence
- Added proper type hints:
  - `Callable[[int], None]` for callback parameter and `_callbacks` list
  - Added `Callable` import from `collections.abc`
- Comprehensive test suite: 14 integration tests covering:
  - Controller creation for single-field and multi-field
  - Correct `n_frames`, `fps`, `frame_times` attributes
  - Controller initial state
  - Widget integration (controller accessible via `viewer.playback_controller`)

**Files created/modified**:
- `src/neurospatial/animation/backends/napari_backend.py` (modified) - Added controller integration
- `tests/animation/test_playback_controller_integration.py` (new) - 14 integration tests

### Task: Phase 2.1 - Time-Indexed Image Layer for In-Memory Video
**Status**: COMPLETED (2025-12-01)

**What was implemented**:
- Modified `_add_video_layer()` to detect in-memory vs file-based video sources
- For in-memory np.ndarray:
  - Creates 4D Image layer with shape `(n_animation_frames, H, W, 3)`
  - Pre-orders frames according to `frame_indices` mapping
  - Uses vectorized NumPy assignment for performance
  - napari handles frame selection natively via dims[0] (no callback)
- For file-based VideoReaderProtocol:
  - Keeps existing callback approach (layer.data = frame on each change)
- Changed `_add_video_layer()` return type to `tuple[Layer, bool]` (layer, uses_native_time)
- Updated both calling sites (single-field and multi-field paths) to only register callbacks for file-based videos
- Comprehensive test suite: 12 tests covering all scenarios

**Files created/modified**:
- `src/neurospatial/animation/backends/napari_backend.py` (modified) - Added time-indexed optimization
- `tests/animation/test_video_time_indexed.py` (new) - 12 unit tests

**Performance benefit**:
- Eliminates per-frame `layer.data = frame` overhead (~2-3ms per frame)
- For 30 fps playback, this saves ~60-90ms/second of playback time
- One-time setup cost for frame reordering is amortized over playback duration

---

## Next Task

**Task**: Phase 2.2 - Add Ring Buffer for File-Based Video
**Purpose**: Add frame caching for file-based video to reduce seek overhead

---

## Notes

### 2025-12-01

- Explored existing animation backend structure
- Found existing `benchmark_datasets` module with `BenchmarkConfig` and data generators
- Used TDD approach: wrote tests first, then implemented script
- Code review identified several issues that were fixed:
  - Added type hint for `timer()` context manager
  - Added safety limit for boundary reflection algorithm
  - Used `DEFAULT_FPS` constant instead of hardcoded 30.0
  - Improved docstrings with Notes sections
  - Fixed return type annotation for `create_selected_overlays()`
  - Added edge case tests for coverage

---

## Blockers

None currently.

---
