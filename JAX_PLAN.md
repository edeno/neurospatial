# JAX Backend Integration Plan

**Goal**: Add optional JAX acceleration for compute-intensive operations across neurospatial, enabling 10-500x speedups for multi-neuron analyses while maintaining full NumPy compatibility.

**Design Principles**:

- Zero-config for most users (auto-selects best available backend)
- Memory-safe (chunked execution prevents OOM on GPU)
- Cross-platform (NumPy fallback on Windows where JAX unavailable)
- No Python loops in hot paths (vmap everything)

---

## V1 Scope (Recommended)

Based on code review feedback, the v1 implementation focuses on **maximum impact with minimum complexity**:

### V1 Includes

| Component | Rationale |
|-----------|-----------|
| `memory_budget()` context manager | Central memory management for all ops |
| `use_backend()` / `get_backend()` API | Clean backend selection |
| `apply_kernel_batch()` | Heavy numeric op, easy to JAXify |
| `log_poisson_likelihood_batch()` | Core decoding bottleneck |
| `compute_shuffled_likelihoods()` | 1000+ iterations → massive speedup |
| `gaussian_kde_batch()` | O(n_bins × n_spikes) pairwise distances |

### V1 Defers

| Component | Reason | When to Add |
|-----------|--------|-------------|
| JAX-native `bin_at()` | Complex layout dispatch; binning is rarely the bottleneck | V2: add for RegularGrid only |
| `compute_spike_counts_batch()` in JAX | Depends on JAX binning; use NumPy for v1 | V2: after binning works |
| `detect_fields_batch()` | Returns Python lists; hard to JAXify | Keep in NumPy |
| Multi-GPU (`jax.pmap`) | Overkill for typical neuroscience workloads | V3: if users request |

### V1 Strategy

1. **Spike binning stays on CPU**: Use existing `env.bin_at()` in NumPy to produce `(n_neurons, n_bins)` counts
2. **Hand off to JAX for downstream ops**: Kernels, likelihoods, shuffles, KDE
3. **80-90% of speedup with 20% of engineering**: The heavy costs are downstream, not binning

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Memory-Aware Execution](#memory-aware-execution)
3. [Backend Infrastructure](#backend-infrastructure)
4. [Environment Binning Integration (V2)](#environment-binning-integration-v2---deferred)
5. [Acceleration Opportunities](#acceleration-opportunities)
6. [Implementation Phases (V1)](#implementation-phases-v1---simplified)
7. [Future Phases (V2+)](#future-phases-v2)
8. [API Reference](#api-reference)
9. [Testing Strategy](#testing-strategy)
10. [JAX Performance Notes](#jax-performance-notes)
11. [Migration Guide](#migration-guide)

---

## Architecture Overview

### Directory Structure

```
src/neurospatial/
├── _backends/                      # Package-wide backend infrastructure
│   ├── __init__.py                 # get_backend(), available_backends(), use_backend()
│   ├── _base.py                    # MemoryConfig, compute_chunk_size(), context managers
│   ├── _protocol.py                # ComputeBackend Protocol definition
│   ├── _numpy.py                   # NumPy implementations (always available)
│   └── _jax.py                     # JAX implementations (optional)
│
├── spike_field.py                  # compute_place_fields() - batch API
├── kernels.py                      # apply_kernel_batch()
├── decoding/
│   ├── likelihood.py               # log_poisson_likelihood_batch()
│   └── shuffle.py                  # compute_shuffled_likelihoods()
├── metrics/
│   ├── place_fields.py             # detect_place_fields_batch()
│   ├── population.py               # population_coverage()
│   └── trajectory.py               # compute_turn_angles_batch()
└── simulation/
    └── spikes.py                   # generate_population_spikes()
```

### Core Pattern: vmap + JIT + Chunking

All JAX acceleration follows the same pattern:

```python
# 1. Define single-item computation
@jax.jit
def compute_single(item, shared_data):
    ...

# 2. vmap over items (neurons, shuffles, bins)
compute_batch = jax.vmap(compute_single, in_axes=(0, None))

# 3. Chunk to respect memory budget
for chunk in chunks(items, chunk_size):
    result_chunk = compute_batch(chunk, shared_data)
    results.append(np.asarray(result_chunk))  # Move to CPU immediately
```

---

## Memory-Aware Execution

### The Problem

Naive vmap over large batches causes GPU OOM:

| Operation | Naive Shape | Memory |
|-----------|-------------|--------|
| 200 neurons × 5000 bins | (200, 5000) | 8 MB |
| 1000 shuffles × 10000 time × 5000 bins | (1000, 10000, 5000) | **400 GB** |
| KDE: 5000 bins × 100000 spikes | (5000, 100000) | **4 GB** |

### Solution: Automatic Chunking

```python
import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

@dataclass
class MemoryConfig:
    """Configuration for memory-aware batch computation."""

    max_memory_bytes: int = 2 * 1024**3  # 2GB default
    neuron_chunk_size: int | None = None  # Auto-compute
    shuffle_chunk_size: int | None = None
    time_chunk_size: int | None = None


# Thread-safe context variable for memory configuration
_memory_config_ctx: contextvars.ContextVar[MemoryConfig] = contextvars.ContextVar(
    "memory_config", default=MemoryConfig()
)


def get_memory_config() -> MemoryConfig:
    """Get current memory configuration from context."""
    return _memory_config_ctx.get()


@contextmanager
def memory_budget(
    max_gb: float | None = None,
    *,
    neuron_chunk_size: int | None = None,
    shuffle_chunk_size: int | None = None,
    time_chunk_size: int | None = None,
) -> Iterator[None]:
    """Context manager for memory budget configuration."""
    config = MemoryConfig(
        max_memory_bytes=int(max_gb * 1024**3) if max_gb else 2 * 1024**3,
        neuron_chunk_size=neuron_chunk_size,
        shuffle_chunk_size=shuffle_chunk_size,
        time_chunk_size=time_chunk_size,
    )
    token = _memory_config_ctx.set(config)
    try:
        yield
    finally:
        _memory_config_ctx.reset(token)


def compute_chunk_size(
    item_memory_bytes: int,
    n_items: int,
    max_memory_bytes: int | None = None,
    min_chunk_size: int = 1,
    max_chunk_size: int | None = None,
) -> int:
    """
    Compute optimal chunk size given memory constraints.

    Examples
    --------
    >>> # 1000 shuffles, each producing 400MB
    >>> chunk = compute_chunk_size(400 * 1024**2, 1000, max_memory_bytes=2 * 1024**3)
    >>> chunk
    5  # Process 5 shuffles at a time
    """
    if max_memory_bytes is None:
        max_memory_bytes = get_memory_config().max_memory_bytes

    chunk_size = max(min_chunk_size, max_memory_bytes // max(item_memory_bytes, 1))

    if max_chunk_size is not None:
        chunk_size = min(chunk_size, max_chunk_size)

    return min(chunk_size, n_items)
```

### Memory Budget Context Manager

```python
from neurospatial._backends import memory_budget

# Default: 2GB budget, auto-chunking
fields = compute_place_fields(env, spike_times, times, positions)

# Large GPU: use more memory for fewer chunks (faster)
with memory_budget(max_gb=8.0):
    fields = compute_place_fields(env, spike_times, times, positions)

# Small GPU / CPU: aggressive chunking
with memory_budget(max_gb=1.0):
    fields = compute_place_fields(env, spike_times, times, positions)

# Fine-grained control
with memory_budget(max_gb=4.0, shuffle_chunk_size=100):
    p_values = compute_shuffle_significance(counts, rates, n_shuffles=1000)
```

---

## Backend Infrastructure

### Protocol Definition (V1 - Simplified)

The v1 protocol focuses on **numeric-heavy operations** that benefit most from JAX.
Spike binning and field detection stay in backend-agnostic code.

```python
# src/neurospatial/_backends/_protocol.py

from typing import Protocol, runtime_checkable, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import jax

# Type alias for arrays that could be NumPy or JAX
ArrayLike = NDArray[np.float64]  # At runtime, JAX arrays also work


@runtime_checkable
class ComputeBackend(Protocol):
    """
    Protocol defining backend capabilities for v1.

    V1 focuses on downstream numeric operations (kernels, likelihoods, KDE).
    Spike binning uses env.bin_at() on CPU, then hands off to backend.
    """

    name: str

    # === Kernel Operations ===

    def apply_kernel_batch(
        self,
        fields: ArrayLike,   # (n_fields, n_bins)
        kernel: ArrayLike,   # (n_bins, n_bins)
    ) -> ArrayLike:
        """Apply kernel to batch of fields. Returns (n_fields, n_bins)."""
        ...

    # === Likelihood Operations ===

    def log_poisson_likelihood_batch(
        self,
        spike_counts: ArrayLike,   # (n_time, n_neurons)
        expected_rates: ArrayLike, # (n_neurons, n_bins)
    ) -> ArrayLike:
        """Compute log-likelihood. Returns (n_time, n_bins)."""
        ...

    def compute_shuffled_likelihoods(
        self,
        spike_counts: ArrayLike,
        expected_rates: ArrayLike,
        n_shuffles: int,
        seed: int,
    ) -> ArrayLike:
        """Vectorized shuffle significance. Returns (n_shuffles, n_time, n_bins)."""
        ...

    # === KDE Operations ===

    def gaussian_kde_batch(
        self,
        eval_points: ArrayLike,    # (n_eval, n_dims)
        sample_points: ArrayLike,  # (n_samples, n_dims)
        bandwidth: float,
        weights: ArrayLike | None,
    ) -> ArrayLike:
        """Vectorized Gaussian KDE. Returns (n_eval,)."""
        ...


# NOTE: The following are NOT in the v1 Protocol (kept in backend-agnostic code):
#
# - compute_spike_counts_batch: Uses env.bin_at() on CPU, returns NumPy array
# - detect_fields_batch: Returns Python lists, hard to JAXify
# - vmap/jit primitives: Not needed in protocol; backends use internally
```

### Backend Selection

```python
# src/neurospatial/_backends/__init__.py

from functools import lru_cache
from typing import Literal

import contextvars
from contextlib import contextmanager
from typing import Iterator

from ._base import MemoryConfig, memory_budget, get_memory_config, compute_chunk_size
from ._protocol import ComputeBackend

BackendName = Literal["auto", "numpy", "jax"]

# Thread-safe context variable for backend selection
_backend_ctx: contextvars.ContextVar[BackendName] = contextvars.ContextVar(
    "backend", default="auto"
)

__all__ = [
    "get_backend",
    "available_backends",
    "use_backend",
    "get_current_backend",
    "memory_budget",
    "BackendName",
    "ComputeBackend",
]


def get_current_backend() -> BackendName:
    """Get current backend from context (default: 'auto')."""
    return _backend_ctx.get()


@contextmanager
def use_backend(name: BackendName) -> Iterator[None]:
    """
    Context manager to temporarily override backend selection.

    Examples
    --------
    >>> with use_backend("jax"):
    ...     fields = compute_place_fields(env, spike_times, times, positions)

    >>> with use_backend("numpy"):
    ...     # Force NumPy for reproducibility testing
    ...     fields = compute_place_fields(env, spike_times, times, positions)
    """
    token = _backend_ctx.set(name)
    try:
        yield
    finally:
        _backend_ctx.reset(token)


@lru_cache(maxsize=1)
def _jax_available() -> bool:
    """Check if JAX is importable and functional."""
    try:
        import jax
        import jax.numpy as jnp
        _ = jnp.array([1.0, 2.0])
        return True
    except ImportError:
        return False
    except Exception:
        return False


def available_backends() -> dict[str, bool]:
    """Check which backends are available."""
    return {
        "numpy": True,
        "jax": _jax_available(),
    }


def get_backend(name: BackendName | None = None) -> ComputeBackend:
    """
    Get computation backend instance.

    Parameters
    ----------
    name : {"auto", "numpy", "jax"}, optional
        Backend to use. If None, uses current context (default "auto").

    Returns
    -------
    ComputeBackend
        Backend instance with compute methods.

    Raises
    ------
    ImportError
        If "jax" requested but not installed.
    """
    if name is None:
        name = get_current_backend()

    if name == "numpy":
        from ._numpy import NumPyBackend
        return NumPyBackend()

    if name == "jax":
        from ._jax import JAXBackend
        return JAXBackend()

    # "auto" - prefer JAX
    if _jax_available():
        from ._jax import JAXBackend
        return JAXBackend()

    from ._numpy import NumPyBackend
    return NumPyBackend()
```

### NumPy Backend (V1)

```python
# src/neurospatial/_backends/_numpy.py

"""NumPy backend - always available, optimized vectorization."""

import numpy as np
from numpy.typing import NDArray

from ._base import compute_chunk_size


class NumPyBackend:
    """
    Pure NumPy backend with vectorized operations.

    V1 Protocol: apply_kernel_batch, log_poisson_likelihood_batch,
                 compute_shuffled_likelihoods, gaussian_kde_batch
    """

    name = "numpy"

    def apply_kernel_batch(
        self,
        fields: NDArray[np.float64],
        kernel: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Batched kernel: (n_fields, n_bins) @ (n_bins, n_bins).T"""
        return fields @ kernel.T

    def log_poisson_likelihood_batch(
        self,
        spike_counts: NDArray[np.float64],
        expected_rates: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Vectorized Poisson log-likelihood."""
        log_rates = np.log(np.maximum(expected_rates, 1e-15))
        spike_term = spike_counts @ log_rates
        rate_penalty = -np.sum(expected_rates, axis=0)
        return spike_term + rate_penalty

    def compute_shuffled_likelihoods(
        self,
        spike_counts: NDArray[np.float64],
        expected_rates: NDArray[np.float64],
        n_shuffles: int,
        seed: int,
    ) -> NDArray[np.float64]:
        """Chunked shuffle computation."""
        n_time, n_neurons = spike_counts.shape
        n_bins = expected_rates.shape[1]

        bytes_per_shuffle = n_time * n_bins * 8
        chunk_size = compute_chunk_size(
            bytes_per_shuffle, n_shuffles,
            min_chunk_size=10, max_chunk_size=500,
        )

        rng = np.random.default_rng(seed)
        log_rates = np.log(np.maximum(expected_rates, 1e-15))
        rate_penalty = -np.sum(expected_rates, axis=0)

        results = np.empty((n_shuffles, n_time, n_bins), dtype=np.float64)

        for start in range(0, n_shuffles, chunk_size):
            end = min(start + chunk_size, n_shuffles)
            for i in range(start, end):
                perm = rng.permutation(n_time)
                shuffled = spike_counts[perm]
                results[i] = shuffled @ log_rates + rate_penalty

        return results

    def gaussian_kde_batch(
        self,
        eval_points: NDArray[np.float64],
        sample_points: NDArray[np.float64],
        bandwidth: float,
        weights: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Vectorized Gaussian KDE using broadcasting."""
        if weights is None:
            weights = np.ones(len(sample_points))

        # Broadcast pairwise distances: (n_eval, n_samples)
        diff = eval_points[:, None, :] - sample_points[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)

        two_sigma_sq = 2 * bandwidth ** 2
        kernel_vals = np.exp(-dist_sq / two_sigma_sq)

        return (kernel_vals @ weights) / np.sum(weights)
```

### Spike Binning (Backend-Agnostic)

Spike binning stays outside the backend protocol. This function lives in `spike_field.py`
and uses `env.bin_at()` on CPU, producing NumPy arrays that are then passed to the backend.

```python
# In spike_field.py (not _backends/)

def compute_spike_counts_batch(
    env: Environment,
    spike_times: Sequence[NDArray[np.float64]],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute spike counts per bin for all neurons using CPU binning.

    This is backend-agnostic: uses env.bin_at() on CPU, then hands
    the resulting (n_neurons, n_bins) array to the selected backend
    for downstream operations (kernels, likelihoods, etc.).

    Parameters
    ----------
    env : Environment
        Fitted environment with bin_at() method.
    spike_times : sequence of arrays
        Spike times for each neuron.
    times : ndarray, shape (n_timepoints,)
        Timestamps for position data.
    positions : ndarray, shape (n_timepoints, n_dims)
        Position data aligned with times.

    Returns
    -------
    spike_counts : ndarray, shape (n_neurons, n_bins)
        Spike counts per spatial bin for each neuron.
    """
    n_neurons = len(spike_times)
    n_bins = env.n_bins
    time_min, time_max = times[0], times[-1]
    n_dims = positions.shape[1]

    # Flatten all spikes with neuron IDs
    all_spikes = []
    all_neuron_ids = []

    for neuron_id, spikes in enumerate(spike_times):
        if len(spikes) == 0:
            continue
        valid = (spikes >= time_min) & (spikes <= time_max)
        valid_spikes = spikes[valid]
        all_spikes.append(valid_spikes)
        all_neuron_ids.append(np.full(len(valid_spikes), neuron_id, dtype=np.int64))

    if not all_spikes:
        return np.zeros((n_neurons, n_bins), dtype=np.float64)

    all_spikes_arr = np.concatenate(all_spikes)
    all_neuron_ids_arr = np.concatenate(all_neuron_ids)

    # Interpolate positions for ALL spikes at once
    spike_positions = np.column_stack([
        np.interp(all_spikes_arr, times, positions[:, d])
        for d in range(n_dims)
    ])

    # Bin ALL spikes at once using env.bin_at (CPU)
    spike_bins = env.bin_at(spike_positions)
    valid = spike_bins >= 0
    spike_bins = spike_bins[valid]
    all_neuron_ids_arr = all_neuron_ids_arr[valid]

    # Scatter into (n_neurons, n_bins) via linear indexing
    linear_idx = all_neuron_ids_arr * n_bins + spike_bins
    counts_flat = np.bincount(linear_idx, minlength=n_neurons * n_bins)

    return counts_flat.reshape(n_neurons, n_bins).astype(np.float64)
```

### JAX Backend (V1)

```python
# src/neurospatial/_backends/_jax.py

"""JAX backend - JIT compilation, vmap, GPU acceleration."""

from __future__ import annotations

import sys

import numpy as np
from numpy.typing import NDArray

from ._base import compute_chunk_size, get_memory_config


class JAXBackend:
    """
    JAX backend with JIT compilation and vmap vectorization.

    V1 Protocol: apply_kernel_batch, log_poisson_likelihood_batch,
                 compute_shuffled_likelihoods, gaussian_kde_batch

    Note: All methods use self._jax and self._jnp (lazy-loaded in __init__)
    to avoid module-level import of JAX.
    """

    name = "jax"

    def __init__(self):
        # Check Windows compatibility at instantiation, not import
        if sys.platform == "win32":
            raise ImportError(
                "JAX backend is not available on Windows.\n\n"
                "Use backend='numpy' instead. The NumPy backend is optimized\n"
                "and handles most use cases well."
            )

        # Lazy import JAX only when backend is instantiated
        try:
            import jax
            import jax.numpy as jnp
            self._jax = jax
            self._jnp = jnp
            _ = jnp.array([1.0])  # Verify JAX works
        except ImportError as e:
            raise ImportError(
                "JAX backend requires JAX to be installed.\n\n"
                "Install with:\n"
                "  pip install neurospatial[jax]      # CPU only\n"
                "  pip install neurospatial[jax-cuda] # With CUDA GPU support\n\n"
                "Or use backend='numpy' for pure NumPy computation."
            ) from e

    def apply_kernel_batch(
        self,
        fields: NDArray[np.float64],
        kernel: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Memory-aware batched kernel application."""
        jax = self._jax
        jnp = self._jnp

        n_fields, n_bins = fields.shape

        bytes_per_field = n_bins * 8 * 2
        kernel_bytes = n_bins * n_bins * 8

        config = get_memory_config()
        available = config.max_memory_bytes - kernel_bytes
        chunk_size = compute_chunk_size(
            bytes_per_field, n_fields,
            max_memory_bytes=max(available, bytes_per_field * 10),
            min_chunk_size=10,
        )

        fields_jax = jnp.array(fields)
        kernel_jax = jnp.array(kernel)

        @jax.jit
        def apply_chunk(chunk):
            return chunk @ kernel_jax.T

        if chunk_size >= n_fields:
            return np.asarray(apply_chunk(fields_jax))

        results = []
        for start in range(0, n_fields, chunk_size):
            end = min(start + chunk_size, n_fields)
            results.append(np.asarray(apply_chunk(fields_jax[start:end])))

        return np.concatenate(results, axis=0)

    def log_poisson_likelihood_batch(
        self,
        spike_counts: NDArray[np.float64],
        expected_rates: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """JIT-compiled Poisson log-likelihood."""
        jax = self._jax
        jnp = self._jnp

        @jax.jit
        def compute(counts, rates):
            log_rates = jnp.log(jnp.maximum(rates, 1e-15))
            spike_term = counts @ log_rates
            rate_penalty = -jnp.sum(rates, axis=0)
            return spike_term + rate_penalty

        return np.asarray(compute(jnp.array(spike_counts), jnp.array(expected_rates)))

    def compute_shuffled_likelihoods(
        self,
        spike_counts: NDArray[np.float64],
        expected_rates: NDArray[np.float64],
        n_shuffles: int,
        seed: int,
    ) -> NDArray[np.float64]:
        """Memory-aware vectorized shuffle computation."""
        jax = self._jax
        jnp = self._jnp

        n_time, n_neurons = spike_counts.shape
        n_bins = expected_rates.shape[1]

        bytes_per_shuffle = n_time * n_bins * 8
        fixed_memory = n_time * n_neurons * 8 + n_neurons * n_bins * 8

        config = get_memory_config()
        available = config.max_memory_bytes - fixed_memory
        chunk_size = compute_chunk_size(
            bytes_per_shuffle, n_shuffles,
            max_memory_bytes=max(available, bytes_per_shuffle * 10),
            min_chunk_size=10, max_chunk_size=500,
        )

        spike_counts_jax = jnp.array(spike_counts)
        expected_rates_jax = jnp.array(expected_rates)
        log_rates = jnp.log(jnp.maximum(expected_rates_jax, 1e-15))
        rate_penalty = -jnp.sum(expected_rates_jax, axis=0)

        @jax.jit
        def compute_chunk(keys_chunk):
            def single_shuffle(key):
                perm = jax.random.permutation(key, n_time)
                shuffled = spike_counts_jax[perm]
                return shuffled @ log_rates + rate_penalty
            return jax.vmap(single_shuffle)(keys_chunk)

        key = jax.random.PRNGKey(seed)
        all_keys = jax.random.split(key, n_shuffles)

        results = []
        for start in range(0, n_shuffles, chunk_size):
            end = min(start + chunk_size, n_shuffles)
            chunk_result = compute_chunk(all_keys[start:end])
            results.append(np.asarray(chunk_result))  # Move to CPU immediately

        return np.concatenate(results, axis=0)

    def gaussian_kde_batch(
        self,
        eval_points: NDArray[np.float64],
        sample_points: NDArray[np.float64],
        bandwidth: float,
        weights: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Memory-aware Gaussian KDE with chunked evaluation."""
        jax = self._jax
        jnp = self._jnp

        n_eval = len(eval_points)
        n_samples = len(sample_points)

        bytes_per_eval = n_samples * 8 * 2
        chunk_size = compute_chunk_size(bytes_per_eval, n_eval, min_chunk_size=100)

        samples_jax = jnp.array(sample_points)
        weights_jax = jnp.ones(n_samples) if weights is None else jnp.array(weights)
        weights_sum = jnp.sum(weights_jax)
        two_sigma_sq = 2 * bandwidth ** 2

        @jax.jit
        def kde_chunk(eval_chunk):
            diff = eval_chunk[:, None, :] - samples_jax[None, :, :]
            dist_sq = jnp.sum(diff ** 2, axis=2)
            kernel_vals = jnp.exp(-dist_sq / two_sigma_sq)
            return (kernel_vals @ weights_jax) / weights_sum

        eval_jax = jnp.array(eval_points)

        if chunk_size >= n_eval:
            return np.asarray(kde_chunk(eval_jax))

        results = []
        for start in range(0, n_eval, chunk_size):
            end = min(start + chunk_size, n_eval)
            results.append(np.asarray(kde_chunk(eval_jax[start:end])))

        return np.concatenate(results, axis=0)
```

---

## Environment Binning Integration (V2 - Deferred)

> **Note**: This section describes V2 enhancements. For V1, spike binning uses
> `env.bin_at()` on CPU (see "Spike Binning (Backend-Agnostic)" above).
> The V1 approach captures 80-90% of the speedup since the heavy costs are
> downstream (kernels, likelihoods, shuffles), not binning.

### The Challenge

The `bin_at()` method is the core operation that maps positions to spatial bins. JAX requires **static shapes** for JIT compilation, but `bin_at()` has complex dispatch logic:

- **Regular grids**: Direct calculation from bin edges
- **Hexagonal grids**: Axial coordinate transformation
- **Masked/polygon layouts**: KDTree nearest-neighbor lookup
- **Graph layouts**: Node-based lookup

### Strategy: Layout-Aware Binning (V2)

```python
# Add to ComputeBackend Protocol
class ComputeBackend(Protocol):
    def prepare_binning_state(
        self, env: "Environment"
    ) -> tuple[str, Any]:
        """
        Extract JAX-compatible binning information from environment.

        Returns
        -------
        binning_method : {"regular_grid", "precomputed"}
            Strategy to use for binning.
        binning_data : Any
            Method-specific data:
            - regular_grid: (bin_edges, grid_shape, active_mask, flat_to_active)
            - precomputed: None (pass bin indices directly)
        """
        ...
```

### Implementation by Layout Type

| Layout Type | JAX Strategy | Data Passed | Performance |
|-------------|--------------|-------------|-------------|
| `RegularGrid` | Native JAX binning | `bin_edges`, `grid_shape` | Full acceleration |
| `Masked` | Native JAX binning | `bin_edges`, `active_mask` | Full acceleration |
| `Polygon` | Native JAX binning | `bin_edges`, `active_mask` | Full acceleration |
| `Hexagonal` | Pre-bin on CPU | `bin_indices` | Partial (kernel only) |
| `Graph` | Pre-bin on CPU | `bin_indices` | Partial (kernel only) |
| `TriangularMesh` | Pre-bin on CPU | `bin_indices` | Partial (kernel only) |

### JAX-Native Binning for Regular Grids

```python
# In _jax.py
def _regular_grid_bin_at(
    positions: jnp.ndarray,  # (n_points, n_dims)
    bin_edges: tuple[jnp.ndarray, ...],  # Per-dimension edges
    grid_shape: tuple[int, ...],
    active_mask: jnp.ndarray,  # (n_grid_cells,) bool
    flat_to_active: jnp.ndarray,  # Maps flat grid index to active bin index
) -> jnp.ndarray:
    """
    JAX-compatible binning for regular grids.

    Returns bin indices, or -1 for out-of-bounds positions.
    """
    n_dims = len(bin_edges)

    # Digitize each dimension
    dim_indices = []
    for d in range(n_dims):
        # jnp.searchsorted gives bin index
        idx = jnp.searchsorted(bin_edges[d], positions[:, d], side='right') - 1
        # Clip to valid range
        idx = jnp.clip(idx, 0, len(bin_edges[d]) - 2)
        dim_indices.append(idx)

    # Convert to flat grid index
    flat_idx = dim_indices[0]
    for d in range(1, n_dims):
        flat_idx = flat_idx * grid_shape[d] + dim_indices[d]

    # Check bounds and active mask
    in_bounds = jnp.ones(len(positions), dtype=bool)
    for d in range(n_dims):
        in_bounds &= (positions[:, d] >= bin_edges[d][0])
        in_bounds &= (positions[:, d] <= bin_edges[d][-1])

    is_active = active_mask[flat_idx]
    valid = in_bounds & is_active

    # Map to active bin index
    bin_idx = jnp.where(valid, flat_to_active[flat_idx], -1)

    return bin_idx
```

### Pre-Binning Strategy for Non-Grid Layouts

For layouts that don't support JAX-native binning, we pre-bin on CPU:

```python
def compute_place_fields(
    env: Environment,
    spike_times: Sequence[NDArray],
    times: NDArray,
    positions: NDArray,
    *,
    backend: BackendName = "auto",
    ...
) -> NDArray:
    backend_impl = get_backend(backend)

    # Check if JAX-native binning is supported
    binning_method, binning_data = backend_impl.prepare_binning_state(env)

    if binning_method == "precomputed":
        # Pre-bin ALL spike positions on CPU
        all_spike_positions = _interpolate_all_spikes(spike_times, times, positions)
        all_spike_bins = env.bin_at(all_spike_positions)  # CPU binning

        # Pass pre-computed bin indices to JAX
        spike_counts = backend_impl.compute_spike_counts_from_bins(
            all_spike_bins, n_spikes_per_neuron, env.n_bins
        )
    else:
        # Full JAX acceleration with native binning
        spike_counts = backend_impl.compute_spike_counts_batch(
            spike_times_padded, n_spikes_per_neuron,
            times, positions, binning_data, env.n_bins
        )

    # Rest of computation is always accelerated
    spike_density = backend_impl.apply_kernel_batch(spike_counts, kernel)
    ...
```

### Ragged Spike Array Padding

JAX vmap requires fixed-size arrays. Helper to pad ragged spike times:

```python
# In _backends/_base.py
def pad_ragged_spike_times(
    spike_times: Sequence[NDArray[np.float64]],
    pad_value: float = np.nan,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Convert ragged spike time lists to padded array + counts.

    Parameters
    ----------
    spike_times : sequence of arrays
        Variable-length spike times per neuron.
    pad_value : float, default=np.nan
        Value to use for padding (NaN allows easy masking).

    Returns
    -------
    spike_times_padded : NDArray, shape (n_neurons, max_spikes)
        Padded array with pad_value for invalid entries.
    n_spikes_per_neuron : NDArray, shape (n_neurons,)
        Actual spike count per neuron.

    Examples
    --------
    >>> spikes = [np.array([1.0, 2.0]), np.array([3.0]), np.array([4.0, 5.0, 6.0])]
    >>> padded, counts = pad_ragged_spike_times(spikes)
    >>> padded.shape
    (3, 3)
    >>> counts
    array([2, 1, 3])
    """
    if len(spike_times) == 0:
        return np.array([]).reshape(0, 0), np.array([], dtype=np.int64)

    n_neurons = len(spike_times)
    max_spikes = max(len(st) for st in spike_times) if spike_times else 0

    if max_spikes == 0:
        return np.full((n_neurons, 1), pad_value), np.zeros(n_neurons, dtype=np.int64)

    padded = np.full((n_neurons, max_spikes), pad_value, dtype=np.float64)
    counts = np.array([len(st) for st in spike_times], dtype=np.int64)

    for i, st in enumerate(spike_times):
        if len(st) > 0:
            padded[i, :len(st)] = st

    return padded, counts
```

---

## Acceleration Opportunities

### Summary Table

| Module | Function | JAX Benefit | Speedup | Priority |
|--------|----------|-------------|---------|----------|
| `spike_field` | `compute_place_fields` | vmap neurons, jit | 50-200x | **HIGH** |
| `spike_field` | `_gaussian_kde` | vmap bins, jit, GPU | 100-500x | **HIGH** |
| `decoding/likelihood` | `log_poisson_likelihood` | jit, GPU matmul | 10-50x | **HIGH** |
| `decoding/shuffle` | `compute_shuffled_likelihoods` | vmap shuffles | 100-500x | **HIGH** |
| `metrics/place_fields` | `detect_place_fields` | vmap neurons | 50-100x | MEDIUM |
| `metrics/population` | `population_coverage` | vmap neurons | 50-150x | MEDIUM |
| `metrics/trajectory` | `compute_turn_angles` | vmap trajectory | 50-100x | MEDIUM |
| `simulation/spikes` | `generate_population_spikes` | vmap neurons | 100-300x | MEDIUM |
| `kernels` | `apply_kernel` | vmap fields | 50-200x | MEDIUM |
| `metrics/grid_cells` | `spatial_autocorrelation` | vmap bins | 10-50x | LOW |
| `distance` | `distance_field` | vmap sources | 10-50x | LOW |

### Detailed Opportunities

#### 1. Batch Place Field Computation (HIGH)

**Current** (`spike_field.py`):

```python
# User loops over neurons
for neuron_spikes in all_spike_times:
    field = compute_place_field(env, neuron_spikes, ...)
```

**Proposed**:

```python
# Single call, all neurons
fields = compute_place_fields(env, all_spike_times, times, positions)
# Returns: (n_neurons, n_bins)
```

**JAX acceleration**:

- Precompute shared: kernel, occupancy_density (once)
- vmap spike binning over neurons
- Single matrix multiply: `spike_counts @ kernel.T`

#### 2. Gaussian KDE (HIGH)

**Current** (`spike_field.py:653-753`):

```python
for i, bin_center in enumerate(env.bin_centers):  # Loop over n_bins!
    spike_distances_sq = np.sum((spike_positions - bin_center) ** 2, axis=1)
    ...
```

**Proposed**:

```python
# Vectorized over all bins
density = backend.gaussian_kde_batch(bin_centers, spike_positions, bandwidth)
```

**JAX acceleration**:

- Broadcast pairwise distances: `(n_bins, 1, n_dims) - (1, n_spikes, n_dims)`
- Single GPU kernel for all bins

#### 3. Shuffle Significance Testing (HIGH)

**Current** (`decoding/shuffle.py`):

```python
for _ in range(n_shuffles):  # 1000+ iterations
    perm = rng.permutation(n_time)
    shuffled = spike_counts[perm]
    ll = log_poisson_likelihood(shuffled, rates)
```

**Proposed**:

```python
# All shuffles in one call
likelihoods = backend.compute_shuffled_likelihoods(
    spike_counts, rates, n_shuffles=1000, seed=42
)
# Returns: (n_shuffles, n_time, n_bins)
```

**JAX acceleration**:

- Generate all permutation keys upfront
- vmap likelihood computation over shuffles
- Chunk to respect memory budget

#### 4. Population Spike Generation (MEDIUM)

**Current** (`simulation/spikes.py`):

```python
for model in models:  # Loop over neurons
    rates = model.firing_rate(positions)
    spikes = generate_poisson_spikes(rates, times)
```

**Proposed**:

```python
# All neurons at once
all_spikes = generate_population_spikes(models, positions, times)
```

**JAX acceleration**:

- vmap rate computation over neurons
- Vectorized Poisson sampling with `jax.random`

---

## Implementation Phases (V1 - Simplified)

V1 focuses on downstream numeric operations. Spike binning stays on CPU using `env.bin_at()`.

### Phase 1: Backend Infrastructure

**Goal**: Create the `_backends/` module with memory management and backend selection.

**Tasks**:

1. Create `_backends/` directory structure
2. Implement `_base.py` with MemoryConfig, `memory_budget()`, `compute_chunk_size()`
3. Implement `_protocol.py` with simplified ComputeBackend Protocol (4 methods)
4. Implement `_numpy.py` with v1 backend methods
5. Add `[jax]` optional dependency to `pyproject.toml`
6. Add unit tests for backend infrastructure

**Files to create**:

- `src/neurospatial/_backends/__init__.py`
- `src/neurospatial/_backends/_base.py`
- `src/neurospatial/_backends/_protocol.py`
- `src/neurospatial/_backends/_numpy.py`
- `tests/test_backends.py`

**pyproject.toml additions**:

```toml
[project.optional-dependencies]
jax = [
    "jax>=0.4.20; sys_platform != 'win32'",
    "jaxlib>=0.4.20; sys_platform != 'win32'",
]
jax-cuda = [
    "jax[cuda12]>=0.4.20; sys_platform != 'win32'",
]
fast = [
    "neurospatial[jax]",
]
```

### Phase 2: JAX Backend

**Goal**: Implement JAX backend with all v1 protocol methods.

**Tasks**:

1. Implement `_jax.py` with v1 methods (using `self._jax` pattern)
2. Add memory-aware chunking to all methods
3. Add integration tests comparing NumPy vs JAX results
4. Add statistical comparison tests for shuffle (different RNGs)

**Files to create/modify**:

- `src/neurospatial/_backends/_jax.py`
- `tests/test_backends_jax.py`

**Note**: V1 does NOT include JAX-native binning. Spike binning uses `env.bin_at()` on CPU.

### Phase 3: Batch Place Fields

**Goal**: Add `compute_place_fields()` batch API.

**Tasks**:

1. Add `compute_spike_counts_batch()` to `spike_field.py` (CPU binning)
2. Add `compute_place_fields()` function using backend for kernel application
3. Add `pad_ragged_spike_times()` helper
4. Update `__init__.py` exports
5. Add comprehensive tests and benchmarks

**Files to modify**:

- `src/neurospatial/spike_field.py`
- `src/neurospatial/__init__.py`
- `tests/test_spike_field.py`

**New function signature**:

```python
def compute_place_fields(
    env: Environment,
    spike_times: Sequence[NDArray[np.float64]],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    method: Literal["diffusion_kde", "gaussian_kde"] = "diffusion_kde",
    bandwidth: float = 5.0,
    backend: BackendName = "auto",
    show_progress: bool = True,
) -> NDArray[np.float64]:
    """
    Compute place fields for multiple neurons efficiently.

    Spike binning uses env.bin_at() on CPU. Kernel application and
    smoothing use the selected backend (JAX if available).
    """
```

### Phase 4: Likelihood & Shuffle

**Goal**: Add batch likelihood and shuffle functions to decoding.

**Tasks**:

1. Add `log_poisson_likelihood_batch()` wrapper in `decoding/likelihood.py`
2. Add `compute_shuffled_likelihoods()` wrapper in `decoding/shuffle.py`
3. Integrate with existing decoding API
4. Add tests and benchmarks

**Files to modify**:

- `src/neurospatial/decoding/likelihood.py`
- `src/neurospatial/decoding/shuffle.py`
- `tests/test_decoding.py`

### Phase 5: Documentation & Polish

**Goal**: Document the JAX backend for users.

**Tasks**:

1. Add JAX backend section to user guide
2. Add performance benchmarks to docs
3. Add memory tuning guide
4. Update CLAUDE.md with JAX patterns
5. Create migration examples

---

## Future Phases (V2+)

These are deferred from v1:

### V2: JAX-Native Binning (RegularGrid only)

- Implement `_regular_grid_bin_at()` in JAX
- Add `prepare_binning_state()` to protocol
- Move spike binning to GPU for regular grids
- Keep CPU binning for irregular layouts

### V2: Batch Field Detection

- Add `detect_fields_batch()` with NumPy implementation
- Consider sparse representation for large environments

### V3: Multi-GPU Support

- Add `jax.pmap` for data parallelism
- Only if users request it

---

## API Reference

### Backend Selection

```python
from neurospatial._backends import get_backend, available_backends, use_backend

# Check availability
available_backends()  # {'numpy': True, 'jax': True}

# Get backend (auto-selects)
backend = get_backend()
backend.name  # 'jax' or 'numpy'

# Explicit selection
backend = get_backend("numpy")
backend = get_backend("jax")  # ImportError if unavailable

# Context manager
with use_backend("jax"):
    fields = compute_place_fields(...)
```

### Memory Configuration

```python
from neurospatial._backends import memory_budget

# Set memory limit
with memory_budget(max_gb=4.0):
    fields = compute_place_fields(...)

# Fine-grained control
with memory_budget(max_gb=2.0, shuffle_chunk_size=100):
    p_values = compute_shuffle_significance(...)

# Environment variable
# NEUROSPATIAL_MAX_MEMORY_GB=8.0
```

### Batch Operations

```python
from neurospatial import compute_place_fields
from neurospatial.decoding import compute_shuffled_likelihoods

# Batch place fields
fields = compute_place_fields(
    env, spike_times, times, positions,
    backend="auto",  # Auto-select JAX if available
)

# Batch shuffles
likelihoods = compute_shuffled_likelihoods(
    spike_counts, expected_rates,
    n_shuffles=1000,
    seed=42,
    backend="jax",
)
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_backends.py

import pytest
import numpy as np
from neurospatial._backends import get_backend, available_backends

class TestBackendSelection:
    def test_numpy_always_available(self):
        backends = available_backends()
        assert backends["numpy"] is True

    def test_get_numpy_backend(self):
        backend = get_backend("numpy")
        assert backend.name == "numpy"

    @pytest.mark.skipif(not available_backends()["jax"], reason="JAX not installed")
    def test_get_jax_backend(self):
        backend = get_backend("jax")
        assert backend.name == "jax"


class TestMemoryConfig:
    def test_default_budget(self):
        from neurospatial._backends._base import get_memory_config
        config = get_memory_config()
        assert config.max_memory_bytes == 2 * 1024**3

    def test_compute_chunk_size(self):
        from neurospatial._backends._base import compute_chunk_size
        # 100 items, 100MB each, 2GB budget -> 20 items per chunk
        chunk = compute_chunk_size(100 * 1024**2, 100, max_memory_bytes=2 * 1024**3)
        assert chunk == 20
```

### Integration Tests

```python
# tests/test_backends_integration.py

import pytest
import numpy as np
from numpy.testing import assert_allclose

from neurospatial._backends import get_backend, available_backends

@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    return {
        "spike_counts": rng.random((100, 50)),
        "expected_rates": rng.random((50, 200)) + 0.1,
    }

class TestBackendConsistency:
    """Verify JAX and NumPy backends produce identical results."""

    @pytest.mark.skipif(not available_backends()["jax"], reason="JAX not installed")
    def test_likelihood_consistency(self, sample_data):
        numpy_backend = get_backend("numpy")
        jax_backend = get_backend("jax")

        numpy_result = numpy_backend.log_poisson_likelihood_batch(
            sample_data["spike_counts"],
            sample_data["expected_rates"],
        )
        jax_result = jax_backend.log_poisson_likelihood_batch(
            sample_data["spike_counts"],
            sample_data["expected_rates"],
        )

        assert_allclose(numpy_result, jax_result, rtol=1e-5)

    @pytest.mark.skipif(not available_backends()["jax"], reason="JAX not installed")
    def test_shuffle_consistency(self, sample_data):
        """
        Test that shuffle results have correct shape and statistical properties.

        NOTE: NumPy and JAX use different RNG algorithms (PCG64 vs ThreeFry),
        so identical seeds produce different permutation sequences. This is
        expected and NOT a bug. We verify:
        1. Shapes match
        2. Statistical properties match (mean, variance of results)
        3. Each backend is internally reproducible with same seed
        """
        numpy_backend = get_backend("numpy")
        jax_backend = get_backend("jax")

        numpy_result = numpy_backend.compute_shuffled_likelihoods(
            sample_data["spike_counts"],
            sample_data["expected_rates"],
            n_shuffles=100,  # More shuffles for stable statistics
            seed=42,
        )
        jax_result = jax_backend.compute_shuffled_likelihoods(
            sample_data["spike_counts"],
            sample_data["expected_rates"],
            n_shuffles=100,
            seed=42,
        )

        # Shapes must match exactly
        assert numpy_result.shape == jax_result.shape

        # Statistical properties should be similar (not identical values)
        # Mean and std across shuffles should be close
        assert_allclose(numpy_result.mean(), jax_result.mean(), rtol=0.1)
        assert_allclose(numpy_result.std(), jax_result.std(), rtol=0.2)

    @pytest.mark.skipif(not available_backends()["jax"], reason="JAX not installed")
    def test_shuffle_reproducibility(self, sample_data):
        """Test that each backend is internally reproducible with same seed."""
        jax_backend = get_backend("jax")

        result1 = jax_backend.compute_shuffled_likelihoods(
            sample_data["spike_counts"],
            sample_data["expected_rates"],
            n_shuffles=10,
            seed=42,
        )
        result2 = jax_backend.compute_shuffled_likelihoods(
            sample_data["spike_counts"],
            sample_data["expected_rates"],
            n_shuffles=10,
            seed=42,
        )

        # Same seed → identical results within same backend
        assert_allclose(result1, result2, rtol=1e-10)
```

### Edge Case Tests

```python
# tests/test_backends_edge_cases.py

import pytest
import numpy as np
from numpy.testing import assert_allclose

from neurospatial._backends import get_backend, available_backends
from neurospatial._backends._base import pad_ragged_spike_times


class TestEmptyInputs:
    """Test handling of empty or minimal inputs."""

    def test_empty_spike_times(self):
        """Empty spike list should return zeros."""
        backend = get_backend("numpy")
        padded, counts = pad_ragged_spike_times([])
        assert padded.shape == (0, 0)
        assert counts.shape == (0,)

    def test_neurons_with_no_spikes(self):
        """Neurons with zero spikes should work."""
        backend = get_backend("numpy")
        spike_times = [
            np.array([1.0, 2.0]),
            np.array([]),  # No spikes
            np.array([3.0]),
        ]
        padded, counts = pad_ragged_spike_times(spike_times)
        assert padded.shape == (3, 2)  # max_spikes = 2
        assert list(counts) == [2, 0, 1]

    def test_single_spike(self):
        """Single spike should work."""
        backend = get_backend("numpy")
        spike_times = [np.array([1.5])]
        padded, counts = pad_ragged_spike_times(spike_times)
        assert padded.shape == (1, 1)
        assert counts[0] == 1
        assert padded[0, 0] == 1.5


class TestBoundaryConditions:
    """Test boundary conditions and limits."""

    def test_very_large_n_shuffles(self):
        """Large shuffle counts should chunk correctly without OOM."""
        backend = get_backend("numpy")
        spike_counts = np.random.rand(100, 10)
        expected_rates = np.random.rand(10, 50) + 0.1

        # Should not raise OOM - will chunk internally
        result = backend.compute_shuffled_likelihoods(
            spike_counts, expected_rates,
            n_shuffles=5000,  # Large but should be chunked
            seed=42,
        )
        assert result.shape == (5000, 100, 50)

    def test_single_bin_environment(self):
        """Single-bin environment should work."""
        backend = get_backend("numpy")
        # 1 bin, 5 neurons
        fields = np.random.rand(5, 1)
        kernel = np.array([[1.0]])  # 1x1 identity kernel

        result = backend.apply_kernel_batch(fields, kernel)
        assert result.shape == (5, 1)
        assert_allclose(result, fields)

    def test_single_neuron(self):
        """Single neuron should work."""
        backend = get_backend("numpy")
        spike_counts = np.random.rand(100, 1)  # 100 time, 1 neuron
        expected_rates = np.random.rand(1, 50) + 0.1  # 1 neuron, 50 bins

        result = backend.log_poisson_likelihood_batch(spike_counts, expected_rates)
        assert result.shape == (100, 50)


class TestNumericalStability:
    """Test numerical edge cases."""

    def test_very_small_rates(self):
        """Very small rates should not produce -inf."""
        backend = get_backend("numpy")
        spike_counts = np.array([[1.0, 0.0]])
        expected_rates = np.array([[1e-15, 1e-15]])  # Very small

        result = backend.log_poisson_likelihood_batch(spike_counts, expected_rates)
        assert np.all(np.isfinite(result))

    def test_zero_spike_counts(self):
        """Zero spike counts should work."""
        backend = get_backend("numpy")
        spike_counts = np.zeros((100, 10))
        expected_rates = np.random.rand(10, 50) + 0.1

        result = backend.log_poisson_likelihood_batch(spike_counts, expected_rates)
        assert np.all(np.isfinite(result))


@pytest.mark.skipif(not available_backends()["jax"], reason="JAX not installed")
class TestJAXSpecific:
    """JAX-specific edge cases."""

    def test_jax_handles_empty_inputs(self):
        """JAX should handle empty inputs gracefully."""
        backend = get_backend("jax")
        fields = np.zeros((0, 100))  # 0 neurons
        kernel = np.random.rand(100, 100)

        result = backend.apply_kernel_batch(fields, kernel)
        assert result.shape == (0, 100)

    def test_jax_chunking_boundary(self):
        """Test exact chunk size boundaries."""
        from neurospatial._backends import memory_budget

        backend = get_backend("jax")
        spike_counts = np.random.rand(100, 10)
        expected_rates = np.random.rand(10, 50) + 0.1

        # Force very small chunks
        with memory_budget(max_gb=0.001):
            result = backend.compute_shuffled_likelihoods(
                spike_counts, expected_rates,
                n_shuffles=100,
                seed=42,
            )
        assert result.shape == (100, 100, 50)
```

### Benchmark Tests

```python
# tests/benchmarks/bench_backends.py

import pytest
import numpy as np

from neurospatial._backends import get_backend, available_backends

@pytest.fixture
def large_data():
    rng = np.random.default_rng(42)
    return {
        "spike_counts": rng.random((10000, 200)),  # 10k time bins, 200 neurons
        "expected_rates": rng.random((200, 5000)) + 0.1,  # 200 neurons, 5k bins
    }

class TestBackendPerformance:

    @pytest.mark.benchmark(group="likelihood")
    def test_numpy_likelihood(self, benchmark, large_data):
        backend = get_backend("numpy")
        benchmark(
            backend.log_poisson_likelihood_batch,
            large_data["spike_counts"],
            large_data["expected_rates"],
        )

    @pytest.mark.skipif(not available_backends()["jax"], reason="JAX not installed")
    @pytest.mark.benchmark(group="likelihood")
    def test_jax_likelihood(self, benchmark, large_data):
        backend = get_backend("jax")
        # Warm up JIT
        _ = backend.log_poisson_likelihood_batch(
            large_data["spike_counts"][:10],
            large_data["expected_rates"],
        )
        benchmark(
            backend.log_poisson_likelihood_batch,
            large_data["spike_counts"],
            large_data["expected_rates"],
        )
```

---

## JAX Performance Notes

### JIT Compilation Overhead

JAX functions decorated with `@jax.jit` are compiled on first execution. This introduces a one-time compilation overhead:

```python
# First call: ~500ms-2s compilation + execution
fields = compute_place_fields(env, spike_times, times, positions, backend="jax")

# Subsequent calls: ~10ms execution only (100x+ faster)
fields = compute_place_fields(env, spike_times_2, times, positions, backend="jax")
```

**Best practices**:

1. **Warm up in benchmarks**: Always run one throwaway call before timing
2. **Batch similar operations**: Compile once, run many times
3. **Avoid shape changes**: Recompilation triggers when array shapes change

### Static Shapes Requirement

JAX JIT requires static array shapes. Changing shapes triggers recompilation:

```python
# Recompiles each iteration (slow)
for n_neurons in [10, 50, 100, 200]:
    spike_times = generate_spikes(n_neurons)
    fields = compute_place_fields(env, spike_times, ...)  # Recompiles!

# Better: Pad to consistent shape
max_neurons = 200
spike_times_padded, counts = pad_ragged_spike_times(all_spike_times)
fields = compute_place_fields(env, spike_times_padded, ...)  # Compiles once
```

**Our strategy**: The `pad_ragged_spike_times()` helper converts variable-length spike arrays to fixed-shape arrays with NaN padding. This allows single compilation even with varying neuron counts.

### Memory Transfer Overhead

Moving data between CPU and GPU has overhead. Minimize transfers:

```python
# Bad: Repeated transfers
for i in range(100):
    result = jax_fn(jnp.array(data[i]))  # CPU→GPU each iteration
    results.append(np.asarray(result))    # GPU→CPU each iteration

# Good: Batch transfer
data_gpu = jnp.array(np.stack(data))  # Single CPU→GPU
results_gpu = jax.vmap(jax_fn)(data_gpu)
results = np.asarray(results_gpu)  # Single GPU→CPU
```

**Our strategy**: Chunked execution transfers one chunk to GPU, processes it, then transfers results back to CPU before loading next chunk. This bounds GPU memory usage while minimizing transfer overhead.

### Common Pitfalls

#### 1. Python Control Flow in JIT

```python
# Bad: Python control flow recompiles
@jax.jit
def bad_fn(x, flag):
    if flag:  # Python boolean - triggers recompilation for each value
        return x * 2
    return x

# Good: JAX control flow
@jax.jit
def good_fn(x, flag):
    return jax.lax.cond(flag, lambda: x * 2, lambda: x)
```

#### 2. Dynamic Shapes in Loops

```python
# Bad: Shape changes in loop
@jax.jit
def bad_loop(data):
    result = jnp.array([])
    for x in data:
        result = jnp.concatenate([result, process(x)])  # Shape changes!
    return result

# Good: Pre-allocate or use scan
@jax.jit
def good_loop(data):
    return jax.lax.map(process, data)  # Fixed shapes
```

#### 3. RNG State Management

JAX uses explicit RNG key passing (no global state):

```python
# NumPy style (not in JAX)
np.random.seed(42)
samples = np.random.randn(100)

# JAX style (explicit keys)
key = jax.random.PRNGKey(42)
samples = jax.random.normal(key, shape=(100,))

# Multiple samples need key splitting
key1, key2 = jax.random.split(key)
samples1 = jax.random.normal(key1, shape=(100,))
samples2 = jax.random.normal(key2, shape=(100,))
```

### Debugging Tips

#### Check for Recompilation

```python
# Enable JIT compilation logging
import jax
jax.config.update("jax_log_compiles", True)

# Run your code - watch for repeated compilation messages
fields = compute_place_fields(...)
```

#### Profile GPU Memory

```python
# Check available GPU memory
import jax
devices = jax.devices()
for d in devices:
    print(f"{d}: {d.memory_stats()}")
```

#### Force Synchronization for Timing

JAX operations are asynchronous. Force sync for accurate timing:

```python
import time
import jax

start = time.time()
result = jax_fn(data)
result.block_until_ready()  # Wait for GPU computation to complete
elapsed = time.time() - start
```

---

## Migration Guide

### For Users

**Before (sequential)**:

```python
from neurospatial import Environment, compute_place_field

env = Environment.from_samples(positions, bin_size=5.0)

# Slow: Python loop over neurons
fields = []
for neuron_spikes in all_spike_times:
    field = compute_place_field(env, neuron_spikes, times, positions)
    fields.append(field)
fields = np.stack(fields)
```

**After (batch)**:

```python
from neurospatial import Environment, compute_place_fields

env = Environment.from_samples(positions, bin_size=5.0)

# Fast: single call, auto-selects JAX if available
fields = compute_place_fields(env, all_spike_times, times, positions)
```

### For Contributors

**Adding a new backend method**:

1. Add method signature to `_protocol.py`:

```python
class ComputeBackend(Protocol):
    def new_operation(self, ...) -> NDArray:
        ...
```

2. Implement in `_numpy.py`:

```python
def new_operation(self, ...) -> NDArray:
    # NumPy implementation (must always work)
    ...
```

3. Implement in `_jax.py`:

```python
def new_operation(self, ...) -> NDArray:
    # JAX implementation with memory-aware chunking
    chunk_size = compute_chunk_size(...)

    @jax.jit
    def compute_chunk(...):
        ...

    results = []
    for chunk in chunks(data, chunk_size):
        results.append(np.asarray(compute_chunk(chunk)))
    return np.concatenate(results)
```

4. Add tests verifying NumPy/JAX consistency.

---

## Open Questions

1. **Environment binning integration**: How to efficiently expose bin_at logic to JAX?
   - Pass bin_edges for regular grids?
   - Pre-bin on CPU for irregular layouts?
   - Use host_callback (slower but flexible)?

2. **Gradient support**: Should we define custom VJPs for backprop through place fields?
   - Useful for gradient-based optimization
   - Adds complexity

3. **Multi-GPU**: Support for data parallelism across GPUs?
   - `jax.pmap` for multi-GPU
   - Probably overkill for typical use cases

4. **Sparse operations**: Many fields are sparse - use JAX sparse arrays?
   - Could reduce memory further
   - API complexity

---

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX vmap tutorial](https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html)
- [Memory management in JAX](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html)
- [non_local_detector KDE implementation](https://github.com/LorenFrankLab/non_local_detector/blob/main/src/non_local_detector/likelihoods/sorted_spikes_kde.py)
