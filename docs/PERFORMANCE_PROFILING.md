# Performance Profiling for Napari Animation

This document describes approaches to diagnose and optimize performance of the napari animation backend, particularly for large datasets like the bandit task example (~41k frames).

## Quick Start

```bash
# Run the profiling script
cd /path/to/neurospatial

# Basic interactive run
uv run python scripts/profile_napari_animation.py

# With napari perfmon trace
NAPARI_PERFMON=/tmp/napari_trace.json uv run python scripts/profile_napari_animation.py

# Quick test with fewer frames
uv run python scripts/profile_napari_animation.py --frames=5000
```

## Profiling Approaches

### 1. Napari Perfmon (Recommended First Step)

Napari has built-in performance monitoring that outputs Chrome Trace format files.

**Usage:**
```bash
# Output to specific file
NAPARI_PERFMON=/tmp/napari_trace.json uv run python scripts/profile_napari_animation.py

# Enable performance logging (no file)
NAPARI_PERFMON=1 uv run python scripts/profile_napari_animation.py
```

**Viewing trace files:**
- **Chrome**: Open `chrome://tracing`, load the JSON file
- **Speedscope**: Upload to https://www.speedscope.app/ for flame graphs

**What to look for:**
- Long synchronous blocks in the main thread
- Qt event processing time (`Qt events` category)
- Layer update durations
- Memory allocation patterns

### 2. py-spy (Sampling Profiler)

Best for seeing the whole call stack including C extensions and system calls.

**Installation:**
```bash
pip install py-spy
```

**Usage:**
```bash
# Generate SVG flame graph
py-spy record -o profile.svg -- uv run python scripts/profile_napari_animation.py --mode=headless

# Real-time top-like view
py-spy top -- uv run python scripts/profile_napari_animation.py --mode=headless
```

**Advantages:**
- Low overhead (sampling-based)
- Shows native/C code
- No code modification needed

### 3. cProfile (Deterministic Profiler)

Best for detailed function-level timing.

**Usage:**
```bash
uv run python scripts/profile_napari_animation.py --mode=cprofile
```

**View results:**
```bash
# With snakeviz (interactive web viewer)
pip install snakeviz
snakeviz /tmp/napari_animation.prof

# With pstats (command line)
python -c "import pstats; pstats.Stats('/tmp/napari_animation.prof').sort_stats('cumulative').print_stats(30)"
```

### 4. Memory Profiling

**Usage:**
```bash
uv run python scripts/profile_napari_animation.py --mode=memory
```

**Alternative: memory_profiler package:**
```bash
pip install memory_profiler
mprof run uv run python scripts/profile_napari_animation.py --mode=headless
mprof plot  # View graph
```

### 5. Line Profiler (Detailed Line-by-Line)

Best for profiling specific hot functions.

**Installation:**
```bash
pip install line_profiler
```

**Usage:** Add `@profile` decorator to functions of interest, then:
```bash
kernprof -l -v scripts/profile_napari_animation.py
```

## Known Performance Hotspots

Based on the architecture, likely performance hotspots include:

### 1. Initial Setup Phase
- **Place field computation**: O(n_spikes) operations
- **Fields array creation**: Memory allocation for `np.tile(field, (n_frames, 1))`
- **Coordinate transformations**: Converting env coords to napari pixel space

### 2. Overlay Creation
- **Position overlay trails**: Creating `(n_frames,)` track data
- **Head direction vectors**: Per-frame loop for vector computation
- **Skeleton building**: `_build_skeleton_vectors()` - vectorized but still O(n_frames * n_edges)

### 3. Playback Phase
- **Frame stepping**: `viewer.dims.set_current_step()`
- **Qt event processing**: `napari.qt.get_app().processEvents()`
- **Layer data updates**: Video overlay callbacks update `layer.data`

### 4. Memory Hotspots
- **Fields array**: `(n_frames, n_bins)` float64 array
- **Overlay data arrays**: Pre-computed for all frames
- **napari Image layer**: Internal texture caching

## Potential Optimizations

### A. Reduce Frame Count
```python
# Subsample high-frequency data before animation
from neurospatial.animation import subsample_frames
fields_30fps = subsample_frames(fields, source_fps=500, target_fps=30)
```

### B. Use Chunked Caching (Already Implemented)
The napari backend uses chunked caching for >10k frames:
```python
# Tune cache parameters if needed
viewer = env.animate_fields(
    fields,
    backend="napari",
    cache_size=1000,      # Per-frame cache size
    chunk_size=10,        # Frames per chunk
    max_chunks=100,       # Max cached chunks
)
```

### C. Lazy Overlay Loading
Instead of pre-computing all overlay data, compute on-demand:
```python
# Future improvement: lazy overlay evaluation
# Currently all overlays are pre-computed at startup
```

### D. Reduce Overlay Complexity
```python
# Use shorter trail lengths
position_overlay = PositionOverlay(
    data=positions,
    trail_length=5,  # Reduce from 15
)

# Skip head direction overlay if not needed
overlays = [position_overlay]  # Don't add head_direction
```

### E. Memory-Mapped Arrays
For very large datasets, use memory-mapped arrays:
```python
import numpy as np
fields_mmap = np.memmap('/tmp/fields.dat', dtype='float64', mode='r', shape=(n_frames, n_bins))
env.animate_fields(fields_mmap, backend="napari")
```

### F. Video Export Instead of Interactive
For very large datasets, export to video instead of interactive:
```python
env.clear_cache()  # Required for parallel rendering
env.animate_fields(
    fields,
    backend="video",
    save_path="output.mp4",
    n_workers=4,
    fps=30,
)
```

## Profiling Specific Components

### Profile Only Overlay Rendering
```python
import time
from neurospatial.animation.backends.napari_backend import _render_head_direction_overlay

start = time.perf_counter()
_render_head_direction_overlay(viewer, head_dir_data, env)
print(f"Head direction rendering: {time.perf_counter() - start:.3f}s")
```

### Profile Frame Updates
```python
# In the profiling script, measure frame stepping time
import time

times = []
for i in range(100):
    start = time.perf_counter()
    viewer.dims.set_current_step(0, i)
    napari.qt.get_app().processEvents()
    times.append(time.perf_counter() - start)

print(f"Mean frame time: {np.mean(times)*1000:.2f}ms")
print(f"Max frame time: {np.max(times)*1000:.2f}ms")
```

## Interpreting Results

### Good Performance Indicators
- Frame step time < 33ms (for 30 fps playback)
- Initial setup < 5 seconds for 10k frames
- Memory usage < 1GB for 10k frames

### Warning Signs
- Frame step time > 100ms (stuttering playback)
- Memory usage growing during playback (memory leak)
- Initial setup > 30 seconds (optimization needed)

## Architecture Notes

The napari animation backend uses several strategies for performance:

1. **Lazy frame loading**: Frames are rendered on-demand, not pre-computed
2. **LRU caching**: Recently viewed frames are cached
3. **Chunked caching**: For large datasets, frames are cached in chunks
4. **Pre-computed overlays**: Skeleton vectors computed once at initialization
5. **Native napari time dimension**: Overlays use napari's built-in time slicing

The main trade-off is between:
- **Memory**: Pre-computing everything uses more RAM but faster playback
- **Latency**: Lazy computation uses less RAM but may have frame delays

## References

- [Napari Perfmon Documentation](https://napari.org/stable/howtos/perfmon.html)
- [py-spy Documentation](https://github.com/benfred/py-spy)
- [Chrome Tracing Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU)
- [Speedscope](https://www.speedscope.app/)
