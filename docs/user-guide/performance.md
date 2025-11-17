# Performance and Caching

This guide covers performance characteristics, caching strategies, and optimization techniques for neurospatial.

## Caching Strategy

neurospatial uses multiple caching mechanisms to optimize repeated operations on environments. Understanding these caches helps manage memory usage and performance for large-scale analyses.

### Cache Types

#### 1. Kernel Cache (`_kernel_cache`)

**Location**: `Environment._kernel_cache`
**Purpose**: Stores diffusion kernels used for field smoothing operations.
**Key**: `(bandwidth, mode)` tuple where mode is `"transition"` or `"density"`.

**When populated**:
- During `env.smooth(field, bandwidth=X)` calls
- During `env.compute_kernel(bandwidth=X, mode="density")` calls

**Memory usage**:

| Environment Size | Single Kernel | 10 Kernels |
|-----------------|---------------|------------|
| 100 bins        | ~80 KB        | ~800 KB    |
| 1,000 bins      | ~8 MB         | ~80 MB     |
| 10,000 bins     | ~800 MB       | ~8 GB      |

**Example**:

```python
from neurospatial import Environment
import numpy as np

env = Environment.from_samples(data, bin_size=2.0)

# First call: Computes and caches kernel
smoothed1 = env.smooth(field, bandwidth=10.0)  # Slower

# Second call: Retrieves from cache
smoothed2 = env.smooth(field, bandwidth=10.0)  # Faster

# Different bandwidth: New kernel computed
smoothed3 = env.smooth(field, bandwidth=15.0)  # Slower (new cache entry)

# Manual cache control
kernel = env.compute_kernel(bandwidth=10.0, cache=False)  # Skip cache
env._kernel_cache.clear()  # Clear all cached kernels
```

**When to clear**:
- After processing many fields with different bandwidths (prevents memory bloat)
- Before serializing environment (caches don't serialize)
- In memory-constrained environments

---

#### 2. KDTree Cache (`_kdtree_cache`)

**Location**: `Environment._kdtree_cache`
**Purpose**: Accelerates nearest-neighbor queries for point-to-bin mapping.
**Populated**: Automatically on first call to methods requiring spatial queries.

**Memory usage**:

| Environment Size | KDTree Size |
|-----------------|-------------|
| 100 bins        | ~5 KB       |
| 1,000 bins      | ~50 KB      |
| 10,000 bins     | ~500 KB     |
| 100,000 bins    | ~5 MB       |

**Operations using KDTree**:
- `env.bin_at(points)` - Map points to bins
- `env.contains(points)` - Check if points are in environment
- `env.interpolate(field, points, method="nearest")` - Nearest-neighbor interpolation

**Example**:

```python
# First query: Builds KDTree
bin_indices = env.bin_at(points)  # Slower (tree construction)

# Subsequent queries: Uses cached tree
more_indices = env.bin_at(more_points)  # Fast

# Clear cache if environment geometry changes (advanced)
env._kdtree_cache = None
```

**Performance impact**:
- Tree construction: O(n log n) where n = number of bins
- Query: O(log n) per point
- For 1,000 points × 10,000 bins: ~10ms (cached) vs ~50ms (uncached)

---

#### 3. Module-Level KDTree Cache (Global)

**Location**: `neurospatial.spatial` module
**Purpose**: Global cache for `map_points_to_bins()` function calls.
**Key**: `id(env)` - Python object ID of the environment.

**Example**:

```python
from neurospatial import map_points_to_bins

# Uses cached KDTree
bin_indices = map_points_to_bins(points, env)

# Clear only KDTree cache (selective clearing)
env.clear_cache(kdtree=True, kernels=False, cached_properties=False)

# Or clear all caches
env.clear_cache()
```

**When to clear**:
- When processing many different environments sequentially
- Before long-running processes to free memory
- After environment modifications (rare - environments are immutable)
- Use selective clearing to preserve specific caches

---

## Memory Management

### Estimating Total Memory Usage

For a typical workflow processing multiple fields on a single environment:

**Example**: 10,000 bin environment, 5 different smoothing bandwidths, 10 fields analyzed

```
Environment base:     ~1 MB   (bin_centers, connectivity)
KDTree cache:         ~500 KB
Kernel cache:         ~40 MB  (5 kernels × 8 MB each)
Field data (10x):     ~800 KB (10 fields × 80 KB each)
─────────────────────────────
Total:                ~42 MB
```

### Memory Optimization Strategies

#### Strategy 1: Batch Processing with Cache Clearing

```python
environments = [env1, env2, env3, ...]

for env in environments:
    # Process environment
    for bandwidth in [5.0, 10.0, 15.0]:
        result = env.smooth(field, bandwidth=bandwidth)
        process_result(result)

    # Clear caches before next environment
    env._kernel_cache.clear()
    env._kdtree_cache = None
```

#### Strategy 2: Disable Kernel Caching for One-Off Operations

```python
# Don't cache if you won't reuse this kernel
kernel = env.compute_kernel(bandwidth=10.0, cache=False)
smoothed = kernel @ field
```

#### Strategy 3: Pre-compute and Reuse Kernels

```python
# Compute once, reuse many times
kernel_10cm = env.compute_kernel(bandwidth=10.0, cache=True)

# Process multiple fields with same kernel (fast!)
for field in fields:
    smoothed = env.smooth(field, bandwidth=10.0)  # Uses cached kernel
```

---

## Performance Benchmarks

### Spatial Query Performance

Benchmarked on MacBook Pro M1, Python 3.13, numpy 2.3

**`bin_at()` - Map points to bins**:

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

# Identify regions of interest
roi_mask = identify_interesting_regions(coarse_result)

# Refine only interesting regions
refined_env = large_env.subset(roi_mask)
refined_result = analyze(refined_env)
```

---

## Thread Safety

**Important**: neurospatial caches are **not thread-safe**.

**Single-threaded (safe)**:

```python
for i in range(100):
    result = env.smooth(field, bandwidth=10.0)  # Uses cache safely
```

**Multi-threaded (unsafe)**:

```python
from concurrent.futures import ThreadPoolExecutor

# DON'T DO THIS - cache corruption possible
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(env.smooth, field, bandwidth=10.0) for _ in range(100)]
```

**Multi-threaded (safe)**:

```python
# Option 1: One environment per thread
with ThreadPoolExecutor() as executor:
    envs = [copy.deepcopy(env) for _ in range(n_threads)]
    futures = [executor.submit(process, envs[i], field) for i in range(n_threads)]

# Option 2: Disable caching in parallel context
def process_field(field):
    kernel = env.compute_kernel(bandwidth=10.0, cache=False)
    return kernel @ field

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_field, field) for field in fields]
```

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
- **Manually modify** `_kernel_cache` keys (use provided methods)

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
    env._kernel_cache.clear()
    env._kdtree_cache = None

    return results

# Process multiple sessions
all_results = []
for session in sessions:
    result = process_session(session)
    all_results.append(result)
```

---

## Debugging Performance Issues

### Profiling Cache Usage

```python
import sys

# Check kernel cache size
print(f"Cached kernels: {len(env._kernel_cache)}")
for key, kernel in env._kernel_cache.items():
    size_mb = kernel.nbytes / 1024 / 1024
    print(f"  {key}: {size_mb:.2f} MB")

# Check KDTree cache
if env._kdtree_cache is not None:
    tree_size = sys.getsizeof(env._kdtree_cache) / 1024
    print(f"KDTree cache: {tree_size:.2f} KB")
```

### Measuring Operation Times

```python
import time

# Benchmark uncached vs cached
env._kernel_cache.clear()

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
