# Napari Playback Performance Baseline

**Generated**: 2025-12-01
**Test Configuration**:
- Frames: 500
- Playback frames: 100
- Grid size: 50x50 (845 active bins)
- Fields shape: (500, 845)
- Target: 30 fps (33.3 ms/frame)

---

## Summary

| Configuration | Mean (ms) | Median (ms) | P95 (ms) | Max (ms) | Achievable FPS | Target Met? |
|---------------|-----------|-------------|----------|----------|----------------|-------------|
| Position only | 21.87 | 19.13 | 21.26 | 268.78 | ~46 fps | Yes |
| Bodyparts + skeleton | 26.30 | 24.17 | 27.80 | 206.03 | ~38 fps | Yes |
| Head direction | 18.44 | 16.75 | 20.32 | 115.35 | ~54 fps | Yes |
| Events (decay) | 19.20 | 16.53 | 18.71 | 254.80 | ~52 fps | Yes |
| Time series dock | 18.49 | 13.20 | 29.36 | 188.53 | ~54 fps | Yes |
| Video overlay | 18.39 | 16.39 | 18.85 | 181.92 | ~54 fps | Yes |
| All 5 overlays (no video) | 40.75 | 40.70 | 52.50 | 241.51 | ~25 fps | **No** |
| **All 6 overlays** | **47.38** | **49.92** | **56.97** | **250.49** | **~21 fps** | **No** |

---

## Key Findings

### Individual Overlays: All Meet Target

Each overlay type individually achieves 30+ fps:

1. **Head direction** (18.44 ms) - Fastest, minimal overhead
2. **Video overlay** (18.39 ms) - Surprisingly fast for in-memory video
3. **Time series dock** (18.49 ms) - Good performance with matplotlib blitting
4. **Events** (19.20 ms) - Efficient with decay mode
5. **Position** (21.87 ms) - Tracks layer with trail
6. **Bodyparts + skeleton** (26.30 ms) - Most complex single overlay

### Combined Overlays: Below Target

When multiple overlays are combined, frame times accumulate:

- **5 overlays (no video)**: 40.75 ms (~25 fps) - 56% frames exceed target
- **6 overlays (all)**: 47.38 ms (~21 fps) - 73% frames exceed target

### Observations

1. **Overhead is roughly additive**: Each overlay adds ~3-8 ms to frame time
2. **High variance in max times**: All configs show occasional >100ms spikes (GC or system load)
3. **Video overlay is NOT the bottleneck**: Only adds ~6.6 ms when combined with others
4. **Bodyparts + skeleton is heaviest**: Adds ~8 ms to base frame time

---

## Detailed Results

### Position Overlay Only

```
Mean:   21.87 ms
Median: 19.13 ms
P95:    21.26 ms
Min:    17.96 ms
Max:    268.78 ms

✓ Target met: Mean frame time allows ~45.7 fps
Frames exceeding target: 1/100 (1.0%)
```

### Bodyparts + Skeleton

```
Mean:   26.30 ms
Median: 24.17 ms
P95:    27.80 ms
Min:    22.04 ms
Max:    206.03 ms

✓ Target met: Mean frame time allows ~38.0 fps
Frames exceeding target: 1/100 (1.0%)
```

### Head Direction

```
Mean:   18.44 ms
Median: 16.75 ms
P95:    20.32 ms
Min:    14.99 ms
Max:    115.35 ms

✓ Target met: Mean frame time allows ~54.2 fps
Frames exceeding target: 2/100 (2.0%)
```

### Events (Decay Mode)

```
Mean:   19.20 ms
Median: 16.53 ms
P95:    18.71 ms
Min:    15.30 ms
Max:    254.80 ms

✓ Target met: Mean frame time allows ~52.1 fps
Frames exceeding target: 1/100 (1.0%)
```

### Time Series Dock

```
Mean:   18.49 ms
Median: 13.20 ms
P95:    29.36 ms
Min:    11.13 ms
Max:    188.53 ms

✓ Target met: Mean frame time allows ~54.1 fps
Frames exceeding target: 2/100 (2.0%)
```

### Video Overlay

```
Mean:   18.39 ms
Median: 16.39 ms
P95:    18.85 ms
Min:    15.29 ms
Max:    181.92 ms

✓ Target met: Mean frame time allows ~54.4 fps
Frames exceeding target: 1/100 (1.0%)
```

### All 5 Overlays (No Video)

```
Overlays: position, bodyparts, head_direction, events, timeseries

Mean:   40.75 ms
Median: 40.70 ms
P95:    52.50 ms
Min:    28.80 ms
Max:    241.51 ms

✗ Target not met: Mean frame time allows only ~24.5 fps
Frames exceeding target: 56/100 (56.0%)
```

### All 6 Overlays Combined

```
Overlays: position, bodyparts, head_direction, events, timeseries, video

Mean:   47.38 ms
Median: 49.92 ms
P95:    56.97 ms
Min:    30.39 ms
Max:    250.49 ms

✗ Target not met: Mean frame time allows only ~21.1 fps
Frames exceeding target: 73/100 (73.0%)
```

---

## Optimization Targets

Based on baseline measurements, these areas should be prioritized:

### High Priority

1. **PlaybackController for frame skipping** - When at 21 fps, skipping frames gracefully maintains perceived smoothness
2. **Reduce callback overhead** - Combined overlays suggest callback dispatching overhead is significant

### Medium Priority

3. **Time series dock update mode** - Option to update only on pause would save ~5-10ms during playback
4. **Video in-memory optimization** - Use napari's native time dimension instead of per-frame `layer.data =` assignment

### Lower Priority (Already Efficient)

5. Events decay mode - Already using efficient `layer.shown` mask
6. Position/head direction - Already using native Tracks layer

---

## Test Environment

- macOS Darwin 22.6.0
- Python 3.13
- napari (latest)
- neurospatial v0.9.0

---

## Reproduction

```bash
# Run individual overlay benchmark
uv run python scripts/benchmark_napari_playback.py --position --frames 500 --playback-frames 100 --auto-close

# Run all overlays benchmark
uv run python scripts/benchmark_napari_playback.py --all-overlays --frames 500 --playback-frames 100 --auto-close

# With perfmon tracing
NAPARI_PERFMON=scripts/perfmon_config.json uv run python scripts/benchmark_napari_playback.py --all-overlays
```
