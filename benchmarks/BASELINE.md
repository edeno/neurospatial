# Performance Baseline Metrics

**Date:** 2025-11-21
**Machine:** macOS (darwin)
**Python:** 3.13.5

This document records baseline performance metrics for each animation backend
before optimization work begins. Use these numbers to measure improvement.

---

## Benchmark Configurations

| Config | Frames | Grid Size | Overlays |
|--------|--------|-----------|----------|
| small | 100 | 40x40 | Position only |
| medium | 5,000 | 100x100 | Position + Skeleton + Head Direction |
| large | 100,000 | 100x100 | Position + Skeleton (7 bodyparts) + Head Direction |

---

## Napari Backend

Best for large-scale interactive exploration (100K+ frames).

### small (100 frames, position overlay)

| Metric | Time (ms) | Memory (MB) |
|--------|-----------|-------------|
| Data creation | 1,346 | 229 |
| Viewer init | 2,585 | 452 |
| Random seek (avg, n=100) | 4.38 | - |

### medium (5,000 frames, all overlays)

| Metric | Time (ms) | Memory (MB) |
|--------|-----------|-------------|
| Data creation | 1,479 | 249 |
| Viewer init | 2,974 | 517 |
| Random seek (avg, n=100) | 15.80 | - |

### Key Observations

- Viewer initialization is ~2.5-3 seconds regardless of frame count
- Random seek time scales with overlay complexity (4ms small, 16ms medium)
- Memory usage increases with overlays (452MB small, 517MB medium)

---

## Video Backend

Best for exporting animations to MP4 for publications/presentations.

### small (100 frames, position overlay)

| Metric | Time (ms) | Memory (MB) |
|--------|-----------|-------------|
| Data creation | 1,516 | 228 |
| Export serial (100 frames) | 2,995 | 230 |
| Time per frame (serial) | 29.95 | - |
| Export parallel (4 workers) | 2,757 | 233 |
| Time per frame (parallel) | 27.57 | - |
| File size | - | 0.18 MB |

**Parallel speedup:** 1.09x (limited by overhead for small frame counts)

### medium (500 frames, no overlays - truncated)

| Metric | Time (ms) | Memory (MB) |
|--------|-----------|-------------|
| Data creation | 350 | 252 |
| Export serial (500 frames) | 7,885 | 236 |
| Time per frame (serial) | 15.77 | - |
| Export parallel (4 workers) | 3,967 | 239 |
| Time per frame (parallel) | 7.93 | - |

**Parallel speedup:** 1.99x

### large (500 frames, no overlays - truncated)

| Metric | Time (ms) | Memory (MB) |
|--------|-----------|-------------|
| Data creation | 6,584 | 614 |
| Export serial (500 frames) | 8,444 | 221 |
| Time per frame (serial) | 16.89 | - |
| Export parallel (4 workers) | 4,010 | 225 |
| Time per frame (parallel) | 8.02 | - |

**Parallel speedup:** 2.11x

### Key Observations

- Time per frame: ~15-30ms depending on grid size and overlays
- Parallel rendering gives ~2x speedup for 500+ frames
- For small frame counts (<200), parallel overhead negates benefits
- Memory usage dominated by field data

---

## Widget Backend

Best for Jupyter notebook integration and quick exploration.

### small (100 frames, position overlay)

| Metric | Time (ms) | Memory (MB) |
|--------|-----------|-------------|
| Data creation | 1,286 | 229 |
| First render | 15.39 | 257 |
| Average render (n=100) | 8.93 | - |
| Average scrub (n=50) | 8.92 | 257 |
| Scrub P50 | 8.92 | - |
| Scrub P95 | 9.63 | - |

### medium (500 frames, no overlays - truncated)

| Metric | Time (ms) | Memory (MB) |
|--------|-----------|-------------|
| Data creation | 353 | 276 |
| First render | 7.44 | 280 |
| Average render (n=100) | 10.00 | - |
| Average scrub (n=50) | 9.81 | 282 |
| Scrub P50 | 9.81 | - |
| Scrub P95 | 10.20 | - |

### large (500 frames, no overlays - truncated)

| Metric | Time (ms) | Memory (MB) |
|--------|-----------|-------------|
| Data creation | 6,359 | 680 |
| First render | 7.66 | 680 |
| Average render (n=100) | 9.95 | - |
| Average scrub (n=50) | 10.02 | 680 |
| Scrub P50 | 9.79 | - |
| Scrub P95 | 10.94 | - |

### Key Observations

- First render is slower than subsequent renders (figure setup overhead)
- Consistent ~9-10ms per frame regardless of frame count
- P95 scrub times are close to average (consistent performance)
- PersistentFigureRenderer reuse is effective

---

## Summary Table

| Backend | Init Time | Frame Time | Parallel Speedup |
|---------|-----------|------------|------------------|
| Napari (small) | 2,585 ms | 4.38 ms/seek | N/A |
| Napari (medium) | 2,974 ms | 15.80 ms/seek | N/A |
| Video (small) | N/A | 29.95 ms/frame | 1.09x |
| Video (medium) | N/A | 15.77 ms/frame | 1.99x |
| Widget (small) | 15.39 ms | 8.93 ms/frame | N/A |
| Widget (medium) | 7.44 ms | 10.00 ms/frame | N/A |

---

## Notes

1. **Truncation for benchmarks:** Medium and large configs were truncated to 500
   frames for video/widget benchmarks to keep benchmark time reasonable. Overlays
   were excluded when truncating to avoid skeleton alignment issues.

2. **Large napari benchmark not run:** The full 100K frame napari benchmark
   requires significant time for data creation and should be run separately.

3. **Overlay impact:** Benchmarks with overlays (small config) show higher
   per-frame times than those without (medium/large truncated).

---

## Running Benchmarks

```bash
# Individual backends
uv run python benchmarks/bench_napari.py --config small
uv run python benchmarks/bench_video.py --config medium
uv run python benchmarks/bench_widget.py --config large

# All configs for a backend
uv run python benchmarks/bench_napari.py --all
uv run python benchmarks/bench_video.py --all
uv run python benchmarks/bench_widget.py --all
```

---

## Encoding Backend Baseline Metrics

**Date:** 2025-12-19
**Machine:** macOS (darwin)
**Python:** 3.13.5
**JAX available:** Yes (x64 mode enabled)

This section records baseline performance metrics for the encoding module backends
(NumPy vs JAX) for spatial rate computation.

---

### Binned Smoothing (No smoothing)

Fastest mode - raw binned spike counts divided by occupancy.

| n_neurons | Backend | Time (ms) | ms/neuron | Speedup |
|-----------|---------|-----------|-----------|---------|
| 10        | numpy   | 4.6       | 0.46      | 1.00x   |
| 10        | jax     | 3.8       | 0.38      | 1.23x   |
| 100       | numpy   | 13.2      | 0.13      | 1.00x   |
| 100       | jax     | 13.6      | 0.14      | 0.97x   |
| 1000      | numpy   | 112.5     | 0.11      | 1.00x   |
| 1000      | jax     | 115.7     | 0.12      | 0.97x   |

**Key Observations:**
- For binned (no smoothing), NumPy and JAX have similar performance
- Overhead of JAX array conversion negates any benefits for simple operations
- Time scales linearly with population size (~0.11-0.13 ms/neuron)

---

### Diffusion KDE Smoothing

Graph-based boundary-aware KDE smoothing. This is the recommended method
for publication-quality rate maps.

| n_neurons | Backend | Time (ms) | ms/neuron | Speedup |
|-----------|---------|-----------|-----------|---------|
| 10        | numpy   | 3.1       | 0.31      | 1.00x   |
| 10        | jax     | 3.6       | 0.36      | 0.86x   |
| 100       | numpy   | 12.9      | 0.13      | 1.00x   |
| 100       | jax     | 12.0      | 0.12      | 1.08x   |
| 1000      | numpy   | 149.3     | 0.15      | 1.00x   |
| 1000      | jax     | 92.0      | 0.09      | **1.62x** |

**Key Observations:**
- JAX shows meaningful speedup for large populations (1.62x at 1000 neurons)
- Speedup comes from accelerated matrix operations in the smoothing kernel
- For small populations (<100), JAX overhead negates benefits
- Diffusion KDE is slightly faster than binned per-neuron due to kernel reuse

---

### When to Use JAX Backend

| Use Case | Recommendation |
|----------|---------------|
| Small population (<100 neurons) | Use NumPy (default) |
| Large population (1000+ neurons) | Use JAX for 1.5-2x speedup |
| Downstream JAX pipeline | Use JAX to avoid array copies |
| Windows | Use NumPy (JAX not supported) |
| Simple exploration | Use NumPy for simplicity |

**Note:** The binning layer always runs on NumPy (CPU). JAX acceleration
applies only to the smoothing/rate computation step.

---

### Running Encoding Benchmarks

```bash
# Default benchmark (10, 100, 1000 neurons, binned smoothing)
uv run python benchmarks/bench_encoding_backends.py

# Custom population sizes
uv run python benchmarks/bench_encoding_backends.py --n-neurons 10 100 500 1000

# With diffusion KDE smoothing
uv run python benchmarks/bench_encoding_backends.py --smoothing diffusion_kde

# NumPy only (skip JAX)
uv run python benchmarks/bench_encoding_backends.py --no-jax

# Save results to CSV
uv run python benchmarks/bench_encoding_backends.py --csv results.csv
```
