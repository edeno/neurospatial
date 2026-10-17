# Performance and Caching

This guide covers performance characteristics, caching strategies, and optimization techniques for neurospatial.

## One analysis cache per environment

Each environment keeps one analysis cache. Its values depend only on the bins
and their connectivity, which never change, so a cached value is always valid.
`env.clear_cache()` (no arguments) releases everything in it to free memory;
the next analysis recomputes what it needs and returns the same result.

Environments returned by `env.with_metadata(...)` and `env.with_regions(...)`
have the same bins, so they share their parent's cache: renaming an
environment or changing its regions does not throw away the kernels and
eigenbases already computed. Clearing the cache on any of them clears it for
all of them. Environments made by `subset`, `clip`, `rebin`, `apply_transform`,
`with_track_width` or `combine`, and environments loaded from a file, a dict,
NWB or a pickle, start with their own empty cache. Pickles, files and NWB
exports never include cached analyses, so there is no need to clear the cache
before saving an environment or sending it to worker processes.

### What the cache holds

| Value | Built by | Size |
|-------|----------|------|
| KD-tree over the bin centers | `nearest_bin`, `interpolate(mode="nearest")` and other nearest-center lookups | about 50 bytes per bin |
| Face and diffusion geometry | `smooth`, `diffuse`, `compute_kernel`, `get_diffusion_geometry` | sparse, a few numbers per bin and per edge |
| Diffusion eigenbasis | `smooth` and `diffuse` | `n_bins × k` floats; `k` grows with the smallest bandwidth used |
| MRF eigenbasis | `get_mrf_basis` and `method="glm"` rate maps | `n_bins × rank` floats |
| Dense diffusion kernels | `compute_kernel(bandwidth=..., mode=...)` with `cache=True` | `n_bins × n_bins` floats each; the 8 most recent `(bandwidth, mode)` pairs are kept |
| Gaussian weights | `compute_gaussian_kernel(bandwidth=...)` | `n_bins × n_bins` floats each; the 8 most recent bandwidths are kept |
| Gradient and divergence operators, boundary bins, bin and edge attribute tables | their accessors | sparse or one row per bin |

`env.smooth` and `env.diffuse` never build a dense `(n_bins, n_bins)` kernel:
they apply the heat operator through the cached eigenbasis, so their memory
grows with `n_bins × k`, not `n_bins²`. Only `compute_kernel` and
`compute_gaussian_kernel` build dense matrices. A dense kernel of 10,000 bins
takes about 800 MB, and the cache keeps at most 8 of them (about 6.4 GB), so
on large environments pass `cache=False` to `compute_kernel` for a kernel you
use once.

```python
import numpy as np
from neurospatial import Environment

positions = np.random.default_rng(0).uniform(0, 100, size=(5000, 2))  # cm
env = Environment.from_samples(positions, bin_size=2.0)
rate = np.random.default_rng(1).random(env.n_bins)  # Hz

smoothed = env.smooth(rate, bandwidth=5.0)  # builds the eigenbasis once
smoothed_again = env.smooth(rate, bandwidth=5.0)  # reuses it
kernel = env.compute_kernel(bandwidth=5.0, mode="average", cache=False)
bins = env.nearest_bin(positions)  # builds the KD-tree once
env.clear_cache()  # release every cached analysis
```

`env.bin_at(positions)` and `env.contains(positions)` use each layout's own
containment test instead of the nearest center. Prefer them for tracking data
when positions outside the environment should get `-1` / `False`.

### Memory management

To process many environments one after another, clear each one's cache when
you are done with it, or let the environment go out of scope:

```python
for env in environments:
    for bandwidth in (5.0, 10.0, 15.0):
        process_result(env.smooth(field, bandwidth=bandwidth))
    env.clear_cache()
```

---

## Performance Benchmarks

### Spatial Query Performance

Benchmarked on MacBook Pro M1, Python 3.13, numpy 2.3

**`env.nearest_bin()` - Nearest-neighbor point-to-bin mapping**:

| Environment | n_bins | n_points | Uncached | Cached  | Speedup |
|-------------|--------|----------|----------|---------|---------|
| Grid 10×10  | 100    | 1,000    | 1.2 ms   | 0.3 ms  | 4×      |
| Grid 100×100| 10,000 | 1,000    | 8.5 ms   | 0.8 ms  | 10×     |
| Grid 100×100| 10,000 | 10,000   | 15 ms    | 3.5 ms  | 4×      |
| Hex 100×100 | 10,000 | 1,000    | 12 ms    | 1.2 ms  | 10×     |

**`neighbors()` - Get bin neighbors**:

| Environment | n_bins | Time   | Notes                |
|-------------|--------|--------|----------------------|
| Grid 10×10  | 100    | 2 µs   | O(1) graph lookup    |
| Grid 100×100| 10,000 | 2 µs   | O(1) graph lookup    |
| Hex 100×100 | 10,000 | 3 µs   | O(1) graph lookup    |

**`shortest_path()` - Dijkstra's algorithm**:

| Environment | n_bins | Time    | Notes               |
|-------------|--------|---------|---------------------|
| Grid 10×10  | 100    | 50 µs   | Average path        |
| Grid 100×100| 10,000 | 800 µs  | Average path        |
| Track (1D)  | 200    | 30 µs   | Linear topology     |

---

### Field Operation Performance

**`smooth()` - Diffusion kernel smoothing**:

| Environment | n_bins | Bandwidth | Uncached | Cached  | Notes                    |
|-------------|--------|-----------|----------|---------|--------------------------|
| Grid 50×50  | 2,500  | 10 cm     | 45 ms    | 8 ms    | Kernel construction cost |
| Grid 100×100| 10,000 | 10 cm     | 280 ms   | 55 ms   | Dense kernel expensive   |
| Grid 100×100| 10,000 | 5 cm      | 150 ms   | 30 ms   | Smaller bandwidth faster |

**Key insight**: Kernel caching provides 5-10× speedup for repeated smoothing operations.

**`interpolate()` - Evaluate fields at arbitrary points**:

| Method       | n_bins | n_points | Time   | Notes                         |
|--------------|--------|----------|--------|-------------------------------|
| nearest      | 10,000 | 1,000    | 1.5 ms | Uses KDTree cache             |
| bilinear     | 10,000 | 1,000    | 12 ms  | Grid-specific, no tree needed |
| nearest      | 10,000 | 10,000   | 8 ms   | Linear in number of points    |

---

### Trajectory Analysis Performance

**`occupancy()` - Time-in-bin computation**:

| n_bins | Trajectory Length | With Smoothing | Without Smoothing |
|--------|-------------------|----------------|-------------------|
| 2,500  | 10,000 samples    | 65 ms          | 15 ms             |
| 10,000 | 10,000 samples    | 180 ms         | 18 ms             |
| 10,000 | 100,000 samples   | 220 ms         | 35 ms             |

**`bin_sequence()` - Convert trajectory to bin indices**:

| n_bins | Trajectory Length | Time  |
|--------|-------------------|-------|
| 2,500  | 10,000 samples    | 8 ms  |
| 10,000 | 100,000 samples   | 45 ms |

---

## Computational Limits

### Recommended Maximum Sizes

Based on typical hardware (16 GB RAM):

| Operation                    | Max n_bins | Notes                                |
|------------------------------|------------|--------------------------------------|
| General operations           | 100,000    | Connectivity graph is sparse         |
| Diffusion smoothing (cached) | 10,000     | Kernel is dense (n² memory)          |
| Geodesic distances (all pairs) | 5,000     | O(n²log n) computation               |
| Real-time queries            | 50,000     | Cached KDTree very fast              |

### Scaling Strategies for Large Environments

**Problem**: Need to analyze 100,000+ bins

**Solution 1**: Spatial partitioning

```python
# Divide large environment into regions
regions = partition_environment(large_env, n_regions=4)

results = []
for region_env in regions:
    # Process each region independently
    result = analyze(region_env)
    results.append(result)

# Combine results
final_result = combine_results(results)
```

**Solution 2**: Coarsen then refine

```python
# Start with coarse analysis
coarse_env = large_env.rebin(factor=4)  # Reduce from 100k to 6k bins
coarse_result = analyze(coarse_env)

# Identify regions of interest as a boolean mask over large_env's bins
roi_mask = identify_interesting_regions(coarse_result)

# Refine only interesting regions
refined_env = large_env.subset(mask=roi_mask)
refined_result = analyze(refined_env)
```

---

## Thread Safety

Threads can share one environment. Each cached analysis is computed once,
even when several threads ask for it at the same time; the other threads wait
and then use the same result. Smoothing with `env.smooth` or `env.diffuse` and
building an MRF basis take turns while they grow the shared eigenbasis, so
every thread gets the same answer as a single-threaded run.

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(env.smooth, field, bandwidth=10.0) for field in fields]
```

Because of these turns, smoothing calls on one environment do not run in
parallel. To spread smoothing over CPU cores, pass the environment to worker
processes (for example with `multiprocessing`); each process unpickles its own
copy with an empty cache.

---

## Best Practices

### ✅ DO:

- **Pre-compute kernels** for frequently used bandwidths
- **Clear caches** between independent analysis sessions
- **Monitor memory usage** when processing many environments
- **Use caching** for repeated operations (smoothing, queries)
- **Profile your code** to identify bottlenecks

### ❌ DON'T:

- **Share environments** between threads without copying
- **Cache indefinitely** in long-running services (memory leak)
- **Ignore cache growth** when processing many different parameters
- **Manually modify** internal cache data structures (use provided methods)

### Example: Production-Ready Analysis Loop

```python
def process_session(session_data, bin_size=2.0):
    """Process one experimental session with cache management."""
    # Create environment
    env = Environment.from_samples(
        session_data['positions'],
        bin_size=bin_size
    )

    # Pre-compute commonly used kernel
    env.compute_kernel(bandwidth=10.0, cache=True)

    # Process multiple fields
    results = {}
    for field_name, field_data in session_data['fields'].items():
        smoothed = env.smooth(field_data, bandwidth=10.0)
        results[field_name] = analyze_field(smoothed)

    # Clear caches before returning (prevent memory leak)
    env.clear_cache()

    return results

# Process multiple sessions
all_results = []
for session in sessions:
    result = process_session(session)
    all_results.append(result)
```

---

## Debugging Performance Issues

### Profiling Memory Usage

```python
import tracemalloc

tracemalloc.start()

env.compute_kernel(bandwidth=10.0, cache=True)
current, peak = tracemalloc.get_traced_memory()

print(f"Current memory: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")

env.clear_cache()
tracemalloc.stop()
```

### Measuring Operation Times

```python
import time

# Benchmark uncached vs cached
env.clear_cache()

start = time.time()
result1 = env.smooth(field, bandwidth=10.0)
uncached_time = time.time() - start

start = time.time()
result2 = env.smooth(field, bandwidth=10.0)
cached_time = time.time() - start

print(f"Uncached: {uncached_time*1000:.2f} ms")
print(f"Cached: {cached_time*1000:.2f} ms")
print(f"Speedup: {uncached_time/cached_time:.1f}×")
```

---

## See Also

- [User Guide: Spatial Analysis](spatial-analysis.md) - Field smoothing and interpolation
- [User Guide: Environments](environments.md) - Environment creation and management
- [API Reference: Environment](../api/index.md) - Complete API documentation
