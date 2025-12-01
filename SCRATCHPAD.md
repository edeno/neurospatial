# SCRATCHPAD.md - Napari Performance Optimization

**Started**: 2025-12-01
**Current Phase**: Phase 0 - Establish Performance Baseline

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

---

## Next Task

**Task**: Run baseline measurements
**Purpose**: Record frame times for each overlay type individually and combined

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
