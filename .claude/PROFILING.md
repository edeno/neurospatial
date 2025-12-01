# Performance Profiling Guide

Tools and techniques for profiling napari animation performance.

**IMPORTANT: All commands MUST be prefixed with `uv run`. Run from project root.**

---

## Quick Start

```bash
# Run benchmark with all overlays (headless mode for quick results)
uv run python scripts/benchmark_napari_playback.py --all-overlays --headless

# Run benchmark with specific overlays
uv run python scripts/benchmark_napari_playback.py --position --events --frames 500

# Run with napari's built-in perfmon tracing
NAPARI_PERFMON=/tmp/trace.json uv run python scripts/benchmark_napari_playback.py --all-overlays
```

---

## Benchmark Script

The `scripts/benchmark_napari_playback.py` script measures napari playback performance with synthetic test data.

### Command-Line Options

```bash
# Frame configuration
--frames N              # Number of animation frames (default: 500)
--playback-frames N     # Frames to step through for timing (default: 100)
--grid-size N           # Environment grid size (default: 50)
--seed N                # Random seed for reproducibility (default: 42)

# Overlay selection (mix and match)
--position              # Position overlay with trail
--bodyparts             # Bodypart overlay with skeleton
--head-direction        # Head direction overlay
--events                # Event overlay (spike-like)
--timeseries            # Time series dock widget
--all-overlays          # Enable all overlays

# Execution modes
--headless              # Close viewer after setup (for CI/scripting)
--no-playback           # Skip playback timing (measure setup only)
```

### Example Output

```
============================================================
NAPARI PLAYBACK BENCHMARK RESULTS
============================================================

Overlays enabled: position, bodyparts, head_direction, events, timeseries

Setup time: 2.864s

Frame timing (100 frames):
  Mean:   15.32 ms
  Median: 14.89 ms
  P95:    22.45 ms
  Min:    12.10 ms
  Max:    35.67 ms

Performance assessment (target: 30 fps = 33.3 ms/frame):
  ✓ Target met: Mean frame time allows ~65.3 fps
  Frames exceeding target: 0/100 (0.0%)
============================================================
```

---

## Napari Perfmon

Napari includes a built-in performance monitoring system that generates Chrome Trace format JSON files.

### Enable Perfmon

```bash
# Simple enable (writes to specified path)
NAPARI_PERFMON=/tmp/napari_trace.json uv run python scripts/benchmark_napari_playback.py

# With config file for detailed tracing
NAPARI_PERFMON=scripts/perfmon_config.json uv run python scripts/benchmark_napari_playback.py
```

### Perfmon Config File

Create `scripts/perfmon_config.json` for detailed tracing:

```json
{
    "trace_qt_events": true,
    "trace_file_on_start": "/tmp/napari_trace.json",
    "trace_callables": ["napari_callbacks"],
    "callable_lists": {
        "napari_callbacks": [
            "neurospatial.animation.backends.napari_backend._make_video_frame_callback",
            "neurospatial.animation.backends.napari_backend._render_timeseries_dock"
        ]
    }
}
```

### View Traces

**Chrome DevTools:**

1. Open Chrome/Chromium
2. Navigate to `chrome://tracing`
3. Click "Load" and select the JSON file
4. Use WASD keys to navigate timeline

**Speedscope (flame graphs):**

1. Go to <https://www.speedscope.app/>
2. Drag and drop the JSON file
3. View as flame graph, sandwich, or timeline

### Key Metrics in Traces

| Event Name | Description | Target |
|------------|-------------|--------|
| `MetaCall:QObject` | Qt event processing overhead | <5ms |
| `Paint:CanvasBackendDesktop` | GPU rendering time | <10ms |
| Custom traced functions | Callback execution time | <5ms |

---

## Built-in Timing Instrumentation

neurospatial includes a timing system that can be enabled via environment variable.

### Enable Timing

```bash
NEUROSPATIAL_TIMING=1 uv run python scripts/benchmark_napari_playback.py
```

### Output

```
[TIMING] _render_position_overlay: 2.34 ms
[TIMING] _render_event_overlay: 1.56 ms
[TIMING] _add_timeseries_dock: 45.23 ms
```

### Add Custom Timing

```python
from neurospatial.animation._timing import timing, timed

# Context manager for code blocks
with timing("my_operation"):
    do_something()

# Decorator for functions
@timed
def my_function():
    pass
```

---

## Performance Targets

| Metric | Target | Acceptable |
|--------|--------|------------|
| Per-frame callback total | <5ms | <10ms |
| Canvas paint time | <10ms | <15ms |
| Total frame latency | <16ms (60 fps) | <40ms (25 fps) |
| Frame skip rate at 25 fps | <5% | <15% |

---

## Profiling Workflow

### 1. Establish Baseline

```bash
# Run benchmark with default settings
uv run python scripts/benchmark_napari_playback.py --all-overlays --headless

# Record the results (setup time, mean frame time, p95)
```

### 2. Identify Bottlenecks

```bash
# Enable detailed tracing
NAPARI_PERFMON=/tmp/trace.json uv run python scripts/benchmark_napari_playback.py --all-overlays

# Open in chrome://tracing and look for long operations
```

### 3. Test Individual Overlays

```bash
# Test each overlay type separately to isolate slow components
uv run python scripts/benchmark_napari_playback.py --position --headless
uv run python scripts/benchmark_napari_playback.py --bodyparts --headless
uv run python scripts/benchmark_napari_playback.py --head-direction --headless
uv run python scripts/benchmark_napari_playback.py --events --headless
uv run python scripts/benchmark_napari_playback.py --timeseries --headless
```

### 4. Profile with Python Tools

```bash
# cProfile for function-level profiling
uv run python -m cProfile -o profile.prof scripts/benchmark_napari_playback.py --headless
uv run snakeviz profile.prof  # Visual profiler

# py-spy for sampling profiler (system calls visible)
py-spy record -o profile.svg -- uv run python scripts/benchmark_napari_playback.py --headless

# Memory profiling
uv run python -m tracemalloc scripts/benchmark_napari_playback.py --headless
```

---

## Common Performance Issues

### Slow Frame Updates

**Symptom:** High per-frame time (>30ms)

**Causes:**

- Layer data reassignment (`layer.data = array`) on each frame
- Matplotlib drawing in time series dock
- Video frame loading from disk

**Solutions:**

- Use napari's native time dimension for layers
- Throttle time series updates during playback
- Cache video frames in memory

### High Setup Time

**Symptom:** Setup time >5s for 1000 frames

**Causes:**

- Field pre-rendering
- Large overlay data conversion
- Multiple file reads

**Solutions:**

- Use lazy field rendering (already implemented)
- Minimize overlay data copies
- Use memory-mapped arrays for large datasets

### Memory Growth

**Symptom:** Memory increases during playback

**Causes:**

- Unbounded caching
- Event handler accumulation
- Matplotlib figure accumulation

**Solutions:**

- Use LRU cache with size limits (already implemented)
- Disconnect event handlers on cleanup
- Reuse matplotlib figures with blitting

---

## Advanced: Custom Perfmon Events

Add custom events to napari traces:

```python
from napari.utils.perf import add_instant_event, perf_timer

# Mark an instant event
add_instant_event("frame_rendered")

# Time a code block (appears in trace)
with perf_timer("my_operation"):
    do_something()
```

---

## Test Files

| File | Purpose |
|------|---------|
| `scripts/benchmark_napari_playback.py` | Main benchmark script |
| `scripts/perfmon_config.json` | Napari perfmon configuration |
| `tests/animation/test_benchmark_napari_playback.py` | Benchmark script tests |
| `tests/animation/test_benchmarks.py` | Performance regression tests |

---

## Related Documentation

- [QUICKSTART.md - Animation](.claude/QUICKSTART.md#visualization--animation) - Basic animation usage
- [ADVANCED.md - Video Overlay](.claude/ADVANCED.md#video-overlay-v050) - Video overlay setup
- [napari perfmon](https://napari.org/dev/howtos/perfmon.html) - Official napari docs
