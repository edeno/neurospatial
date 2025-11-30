# Population Computation Engine Implementation Plan

## Overview

Implement population-level computation for efficient analysis across 50-100 cells, plus targeted performance improvements.

**Goals:**

1. Compute shared data (trajectory bins, occupancy, kernels) once per session
2. Extend `compute_place_field()` to accept precomputed data
3. Provide `PopulationSession` class for batch processing with optional parallelism
4. Fix known performance bottlenecks (deque, vectorized Gaussian KDE)

---

## Architecture

| Module | Purpose |
|--------|---------|
| `src/neurospatial/spike_field.py` (edit) | Add precomputed data parameters to `compute_place_field()` and internal functions |
| `src/neurospatial/population.py` (new) | `PopulationSession` class for batch computation |
| `src/neurospatial/metrics/place_fields.py` (edit) | Fix `deque` in BFS |

---

## Part 1: Extend `compute_place_field()` to Accept Precomputed Data ✅ COMPLETED

### 1.1 New Optional Parameters

Add to `compute_place_field()` signature:

```python
def compute_place_field(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy_seconds: float = 0.0,
    # NEW: Optional precomputed data
    trajectory_bins: NDArray[np.int64] | None = None,
    dt: NDArray[np.float64] | None = None,
    occupancy_density: NDArray[np.float64] | None = None,
    kernel: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
```

### 1.2 Internal Function Changes

Update `_diffusion_kde()` and `_gaussian_kde()` to accept precomputed data:

```python
def _diffusion_kde(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    bandwidth: float,
    *,
    # NEW: Optional precomputed data
    trajectory_bins: NDArray[np.int64] | None = None,
    dt: NDArray[np.float64] | None = None,
    occupancy_density: NDArray[np.float64] | None = None,
    kernel: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    # Use precomputed if provided, else compute
    if trajectory_bins is None:
        trajectory_bins = env.bin_at(positions)
    if dt is None:
        dt = np.diff(times, prepend=times[0])
    if kernel is None:
        kernel = env.compute_kernel(bandwidth, mode="density", cache=True)
    if occupancy_density is None:
        occupancy_counts = np.bincount(trajectory_bins, weights=dt, minlength=env.n_bins)
        occupancy_density = kernel @ occupancy_counts
    # ... rest of computation
```

---

## Part 2: PopulationSession Class

### 2.1 Class Design

**Location:** `src/neurospatial/population.py`

```python
@dataclass
class PopulationSession:
    """Precomputed session data for efficient population analysis.

    Immutable core data with lazy-cached derived data per (bandwidth, method).
    """
    env: Environment
    times: NDArray[np.float64]
    positions: NDArray[np.float64]

    # Immutable core (computed once at creation)
    _trajectory_bins: NDArray[np.int64] = field(init=False)
    _dt: NDArray[np.float64] = field(init=False)

    # Lazy caches per (bandwidth, method)
    _occupancy_cache: dict[tuple, NDArray] = field(default_factory=dict, repr=False)
    _kernel_cache: dict[tuple, NDArray] = field(default_factory=dict, repr=False)
```

### 2.2 Factory Method

```python
@classmethod
def from_trajectory(
    cls,
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
) -> "PopulationSession":
    """Create session from trajectory data.

    Validates inputs and precomputes trajectory bins and time deltas.
    """
    # Validation
    if len(times) != len(positions):
        raise ValueError(f"times ({len(times)}) and positions ({len(positions)}) must have same length")
    if not env._is_fitted:
        raise RuntimeError("Environment must be fitted")

    session = cls(env=env, times=times, positions=positions)
    # Compute immutable core
    object.__setattr__(session, '_trajectory_bins', env.bin_at(positions))
    object.__setattr__(session, '_dt', np.diff(times, prepend=times[0]))
    return session
```

### 2.3 Key Methods

| Method | Description |
|--------|-------------|
| `occupancy(bandwidth, method)` | Returns cached occupancy density, computes if needed |
| `kernel(bandwidth, method)` | Returns cached kernel (delegates to env.compute_kernel) |
| `compute_place_field(spike_times, ...)` | Single cell using cached data |
| `compute_population_fields(spike_times_dict, ...)` | Batch with optional parallelism |
| `compute_summary_statistics(fields, ...)` | Population metrics (wraps existing functions) |

### 2.4 Population Fields with Auto-Parallelism

```python
def compute_population_fields(
    self,
    spike_times: dict[str, NDArray[np.float64]],
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy_seconds: float = 0.0,
    n_jobs: int | Literal["auto"] = "auto",
) -> dict[str, NDArray[np.float64]]:
    """Compute place fields for all cells.

    Parameters
    ----------
    n_jobs : int or "auto"
        Number of parallel workers.
        - "auto": Smart default based on workload (sequential if < 500ms estimated)
        - 1: Sequential (no multiprocessing overhead)
        - > 1: Parallel with multiprocessing.Pool
    """
    # Ensure caches populated BEFORE parallelization
    _ = self.occupancy(bandwidth, method)
    _ = self.kernel(bandwidth, method)

    actual_n_jobs = self._choose_n_jobs(len(spike_times), n_jobs, method)

    if actual_n_jobs == 1:
        return {
            cell_id: self.compute_place_field(spikes, bandwidth=bandwidth, method=method)
            for cell_id, spikes in spike_times.items()
        }
    else:
        # Parallel execution with frozen cache data
        return self._parallel_compute(spike_times, bandwidth, method, actual_n_jobs)

def _choose_n_jobs(self, n_cells: int, n_jobs: int | str, method: str) -> int:
    """Auto-select parallelism based on workload."""
    if n_jobs != "auto":
        return n_jobs

    # Estimate per-cell time
    n_bins = self.env.n_bins
    if method == "diffusion_kde":
        per_cell_ms = n_bins * 0.005  # ~5μs per bin
    elif method == "gaussian_kde":
        per_cell_ms = n_bins * 0.05   # ~50μs per bin
    else:
        per_cell_ms = n_bins * 0.001

    total_ms = n_cells * per_cell_ms

    # Only parallelize if total work > 500ms
    if total_ms < 500:
        return 1

    import os
    max_workers = min(4, os.cpu_count() or 1)
    return min(max_workers, n_cells // 10)
```

### 2.5 Summary Statistics

**Design consideration:** Different brain regions require different metrics:

| Region | Cell Type | Key Metrics |
|--------|-----------|-------------|
| Hippocampus | Place cells | spatial_info, sparsity, selectivity, coherence, field_size, n_fields |
| MEC | Grid cells | gridness, grid_spacing, grid_orientation, grid_field_size |
| MEC | Border cells | border_score, field_width |
| MEC | Head direction | tuning_width, peak_direction, mean_vector_length |

**Approach:** Return a flexible `PopulationSummary` with a `per_cell_metrics` DataFrame. Users select which metrics to compute via a `metrics` parameter. Default metrics are place cell-appropriate.

```python
@dataclass
class PopulationSummary:
    """Population-level summary statistics."""
    n_cells: int
    per_cell_metrics: pd.DataFrame  # Always returned, columns depend on metrics computed

    # Aggregate stats computed from per_cell_metrics
    aggregates: dict[str, float]  # e.g., {"mean_spatial_info": 1.2, "std_spatial_info": 0.3, ...}

def compute_summary_statistics(
    self,
    fields: dict[str, NDArray[np.float64]],
    *,
    metrics: list[str] | None = None,  # None = place cell defaults
    detect_fields: bool = True,
    field_threshold: float = 0.2,
) -> PopulationSummary:
    """Compute summary statistics for population.

    Parameters
    ----------
    metrics : list[str], optional
        Which metrics to compute. Default (None) uses place cell metrics:
        ["spatial_info", "sparsity", "selectivity", "coherence", "field_size", "n_fields"]

        Available metrics:
        - Place cells: spatial_info, sparsity, selectivity, coherence, field_size, n_fields
        - Grid cells: gridness, grid_spacing, grid_orientation (requires 2D fields)
        - Border cells: border_score
        - General: peak_rate, mean_rate, coverage

    Delegates to existing metrics.place_fields and metrics.grid_cells functions.
    """
```

**Example usage:**

```python
# Place cells (default)
stats = session.compute_summary_statistics(fields)

# Grid cells
stats = session.compute_summary_statistics(
    fields,
    metrics=["gridness", "grid_spacing", "grid_orientation", "spatial_info"]
)

# Access results
print(stats.per_cell_metrics[["cell_id", "gridness", "grid_spacing"]])
print(stats.aggregates["mean_gridness"])
```

---

## Part 3: Performance Improvements

### 3.1 Fix `frontier.pop(0)` → `deque` ✅ COMPLETED

**Location:** `src/neurospatial/metrics/place_fields.py:344-347`

```python
# Before (O(n²))
frontier = [seed_idx]
while frontier:
    current = frontier.pop(0)

# After (O(n))
from collections import deque
frontier = deque([seed_idx])
while frontier:
    current = frontier.popleft()
```

### 3.2 Vectorize Gaussian KDE (Chunked)

**Location:** `src/neurospatial/spike_field.py` - replace `_gaussian_kde()` internals

Replace the Python loop at line 380 with chunked vectorization:

```python
def _gaussian_kde(
    env, spike_times, times, positions, bandwidth,
    *, trajectory_bins=None, dt=None, max_memory_mb=500.0
):
    # ... setup code ...

    # Auto-tune chunk size
    n_bins, n_spikes, n_samples = env.n_bins, len(spike_positions), len(positions)
    bytes_per_bin = (n_spikes + n_samples) * n_dims * 8
    chunk_size = max(1, min(n_bins, int(max_memory_mb * 1e6 / bytes_per_bin)))

    spike_density = np.zeros(n_bins)
    occupancy_density = np.zeros(n_bins)
    two_sigma_sq = 2 * bandwidth ** 2

    for start in range(0, n_bins, chunk_size):
        end = min(start + chunk_size, n_bins)
        chunk_centers = env.bin_centers[start:end]  # (chunk, n_dims)

        # Spike density - vectorized over chunk
        if len(spike_positions) > 0:
            diff = chunk_centers[:, np.newaxis, :] - spike_positions[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            spike_density[start:end] = np.sum(np.exp(-dist_sq / two_sigma_sq), axis=1)

        # Occupancy density - vectorized over chunk
        diff_traj = chunk_centers[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_sq_traj = np.sum(diff_traj ** 2, axis=2)
        traj_weights = np.exp(-dist_sq_traj / two_sigma_sq)
        occupancy_density[start:end] = np.sum(traj_weights * dt[np.newaxis, :], axis=1)

    # Normalize
    firing_rate = np.where(occupancy_density > 0, spike_density / occupancy_density, np.nan)
    return firing_rate
```

---

## Part 4: Implementation Order

### Phase 1: Quick Win - Deque Fix ✅ COMPLETED

1. ~~Change `frontier.pop(0)` to `deque.popleft()` in `metrics/place_fields.py:347`~~
2. ~~Add regression test verifying BFS ordering unchanged~~

### Phase 2: Extend compute_place_field() ✅ COMPLETED

1. ~~Add optional precomputed parameters to `compute_place_field()`~~
2. ~~Update `_diffusion_kde()` and `_gaussian_kde()` to use them~~
3. ~~Add tests verifying identical results with/without precomputed data~~

### Phase 3: Vectorize Gaussian KDE

1. Replace Python loop with chunked vectorization
2. Add regression test: `np.allclose(new, old, rtol=1e-10, atol=1e-12)`
3. Add memory budget parameter

### Phase 4: PopulationSession Class

1. Create `src/neurospatial/population.py`
2. Implement `PopulationSession` with factory method
3. Implement `compute_place_field()` using precomputed data
4. Implement `compute_population_fields()` with auto-parallelism
5. Add tests

### Phase 5: Summary Statistics

1. Implement `PopulationSummary` dataclass
2. Implement `compute_summary_statistics()` wrapping existing metrics
3. Export from `__init__.py`
4. Update CLAUDE.md

---

## File Changes Summary

| File | Change |
|------|--------|
| `src/neurospatial/population.py` | **New** - PopulationSession, PopulationSummary |
| `src/neurospatial/spike_field.py` | Edit - add precomputed params, vectorize gaussian_kde |
| `src/neurospatial/metrics/place_fields.py` | Edit - deque fix |
| `src/neurospatial/__init__.py` | Edit - export PopulationSession, PopulationSummary |
| `tests/test_population.py` | **New** - PopulationSession tests |
| `tests/test_spike_field.py` | Edit - regression tests |
| `tests/metrics/test_place_fields.py` | Edit - BFS ordering test |
| `CLAUDE.md` | Edit - document population APIs |

---

## Testing Strategy

| Test Type | Description | Tolerance |
|-----------|-------------|-----------|
| Regression (gaussian_kde) | Vectorized vs original | `rtol=1e-10, atol=1e-12` |
| Regression (deque) | BFS bin indices | Exact match |
| Regression (precomputed) | With vs without precomputed data | `rtol=1e-10, atol=1e-12` |
| Integration | 100-cell synthetic population | Verify speedup ≥ 5× |
| Memory | Chunking respects budget | Monitor peak allocation |

---

## Success Criteria

- [ ] `compute_population_fields()` for 100 cells is ≥5× faster than 100× `compute_place_field()`
- [x] `compute_place_field()` accepts precomputed data without breaking existing API
- [x] `frontier.pop(0)` replaced with `deque.popleft()`
- [ ] Gaussian KDE vectorized with chunking
- [ ] All existing tests pass
- [ ] Documentation updated
