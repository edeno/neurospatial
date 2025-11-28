# Animation Optimization Implementation Plan

**Goal**: Enable visualization of 1–2 hour decoding sessions at 250–500 Hz in napari without temporal downsampling.

**Constraints**:

- Keep one field per decode sample (no temporal decimation of fields)
- Maintain interactive napari playback (smooth scrubbing, tolerable stalls)
- Avoid O(n_frames × n_bins) work that scales badly with long sessions
- Keep API backward-compatible; add changes behind explicit opts

---

## Phase 1: Large-Session Configuration Surface

**Goal**: Establish best practices at the script level before library changes.

### Task 1.1: Update Example Scripts with Best Practices

**File**: `data/view_bandit_napari.py`

Add explicit `frame_times`, `vmin`, `vmax`, and napari tuning kwargs:

```python
# IMPORTANT: Use the streaming helper, NOT np.nanpercentile() which is O(n_frames×n_bins)
from neurospatial.animation import estimate_colormap_range_from_subset
vmin, vmax = estimate_colormap_range_from_subset(fields)  # Samples ~10K frames

# Build frame_times for temporal alignment
frame_times = np.arange(n_frames) / sample_rate_hz

viewer = env.animate_fields(
    fields,
    backend="napari",
    fps=30,
    frame_times=frame_times,
    vmin=vmin,
    vmax=vmax,
    chunk_size=50,      # Larger chunks for sequential playback
    max_chunks=200,     # More chunks for 100K+ frame sessions
)
```

**File**: `data/large_session_napari_example.py`

Update to demonstrate the "full-rate decoding session" pattern with explicit configuration.

### Task 1.2: Document Large-Session Recommendations

**File**: `docs/user-guide/animation.md`

Add "Long Decoding Sessions (No Downsampling)" section with:

- When to use explicit `vmin`/`vmax` (pre-computed from subset)
- Recommended `chunk_size`/`max_chunks` for different session lengths
- Example showing memmap-backed napari workflow

**File**: `CLAUDE.md`

Add Quick Reference example for large sessions under the animation section.

**Verification**: Run `data/large_session_napari_example.py` and confirm documentation matches behavior.

---

## Phase 2: Stop Fighting Memmaps in animate_fields

**Goal**: Keep 2D arrays/memmaps as-is for napari backend instead of converting to list.

### Task 2.1: Modify animate_fields to Preserve Array Format for Napari

**File**: `src/neurospatial/animation/core.py`

**Current behavior** (lines 205-210):

```python
if isinstance(fields, np.ndarray):
    if fields.ndim < 2:
        raise ValueError("fields must be at least 2D (n_frames, n_bins)")
    fields = [fields[i] for i in range(len(fields))]  # Always converts to list
```

**New behavior**:

```python
# Determine if fields should remain as array (napari) or convert to list (other backends)
fields_is_array = isinstance(fields, np.ndarray) and fields.ndim >= 2
n_frames = fields.shape[0] if fields_is_array else len(fields)

# Only convert to list for non-napari backends (video/html/widget need per-frame iteration)
if backend != "napari" and fields_is_array:
    fields = [fields[i] for i in range(fields.shape[0])]
elif not fields_is_array:
    fields = list(fields)
```

**Key changes**:

1. Add `fields_is_array` flag to track input format
2. Compute `n_frames` from array shape or list length
3. Only convert to list for non-napari backends
4. Pass array directly to napari backend

### Task 2.2: Update Field Validation for Arrays

**File**: `src/neurospatial/animation/core.py`

Update field shape validation (around line 227-233) to handle both formats:

```python
if not is_multi_field:
    if fields_is_array:
        # Validate shape: (n_frames, n_bins)
        if fields.shape[1] != env.n_bins:
            raise ValueError(
                f"Fields array has shape {fields.shape} but expected ({n_frames}, {env.n_bins})"
            )
    else:
        for i, field in enumerate(fields):
            if len(field) != env.n_bins:
                raise ValueError(
                    f"Field {i} has {len(field)} values but environment has {env.n_bins} bins."
                )
```

**Verification**:

```bash
# Test with memmap
uv run pytest tests/animation/test_core.py -v -k "memmap or array"

# Manual test
uv run python -c "
import numpy as np
from neurospatial import Environment

positions = np.random.randn(100, 2) * 50
env = Environment.from_samples(positions, bin_size=5.0)

# Test array format preserved for napari
fields = np.random.rand(100, env.n_bins).astype(np.float64)
# This should NOT convert to list internally
"
```

### Task 2.3: Multi-Field Mode Interaction

**Important**: The array preservation logic must NOT interfere with existing multi-field mode.

**Multi-field mode** accepts a **list of sequences** (each a 2D array) for side-by-side comparison (napari only). This is detected by checking if the input is a list of arrays rather than a single 2D array.

```python
# Single-field mode (preserve array for napari)
fields = np.random.rand(100, n_bins)  # Shape: (n_frames, n_bins)
# fields_is_array = True → pass through as array to napari

# Multi-field mode (list of arrays for side-by-side)
fields = [np.random.rand(100, n_bins), np.random.rand(100, n_bins)]
# fields_is_array = False (it's a list) → existing multi-field handling applies
```

**Key check**:

```python
# Multi-field detection (existing logic, must remain unchanged)
is_multi_field = (
    isinstance(fields, (list, tuple))
    and len(fields) > 0
    and isinstance(fields[0], (list, tuple, np.ndarray))
    and hasattr(fields[0], '__len__')
    and len(fields[0]) > 0
    and hasattr(fields[0][0], '__len__')  # Nested sequence check
)

# Array detection (new logic, only for single-field mode)
fields_is_array = (
    isinstance(fields, np.ndarray)
    and fields.ndim >= 2
    and not is_multi_field  # Never true for list inputs
)
```

**Success criteria**:

- Multi-field mode still works: `env.animate_fields([fields1, fields2], backend="napari")`
- Single-field array mode works: `env.animate_fields(fields_2d, backend="napari")`
- Tests: `uv run pytest tests/animation/ -v -k "multi_field"`

---

## Phase 3: Make Napari Backend Array/Memmap-Friendly

**Goal**: Update napari backend to accept 2D arrays and use efficient slicing.

**Key insight from napari docs**: Napari natively supports dask arrays for lazy loading. Consider using `dask.array` as an alternative to custom `LazyFieldRenderer` for simpler, more robust lazy loading. See: <https://napari.org/dev/tutorials/processing/dask.html>

**Alternative approach (dask-based)**:

```python
import dask.array as da

# Wrap memmap/array in dask with chunking
fields_dask = da.from_array(fields, chunks=(chunk_size, -1))

# Apply RGB conversion lazily per chunk
def convert_chunk(block, env, cmap_lookup, vmin, vmax):
    return np.stack([field_to_rgb_for_napari(env, f, cmap_lookup, vmin, vmax) for f in block])

rgb_dask = da.map_blocks(convert_chunk, fields_dask, ...)
viewer.add_image(rgb_dask, contrast_limits=(0, 255), multiscale=False)
```

This leverages napari's native dask support rather than custom array-like classes.

### Task 3.5: Implement Dask-Based Alternative

**File**: `src/neurospatial/animation/backends/napari_backend.py`

Add `_create_dask_field_renderer()` as alternative to `_create_lazy_field_renderer()`:

```python
def _create_dask_field_renderer(
    env: Environment,
    fields: NDArray[np.float64],
    cmap_lookup: NDArray[np.uint8],
    vmin: float,
    vmax: float,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> da.Array:
    """Create dask-based lazy renderer leveraging napari's native dask support."""
    import dask.array as da

    fields_dask = da.from_array(fields, chunks=(chunk_size, -1))

    def convert_chunk(block, block_info=None):
        return np.stack([
            field_to_rgb_for_napari(env, field, cmap_lookup, vmin, vmax)
            for field in block
        ])

    sample_rgb = field_to_rgb_for_napari(env, fields[0], cmap_lookup, vmin, vmax)
    height, width = sample_rgb.shape[:2]

    rgb_dask = da.map_blocks(
        convert_chunk,
        fields_dask,
        dtype=np.uint8,
        drop_axis=1,
        new_axis=[1, 2, 3],
        chunks=(chunk_size, height, width, 3),
    )
    return rgb_dask
```

Add `use_dask: bool = False` parameter to `render_napari` to enable dask backend.

### Task 3.6: Benchmark Both Approaches

**File**: `benchmarks/benchmark_lazy_renderers.py`

Create benchmark comparing:

- **Memory overhead**: Creation cost of renderer
- **Frame access time**: Random and sequential access patterns
- **Scrubbing responsiveness**: Simulated rapid frame changes
- **Dataset sizes**: 10K, 100K, 500K frames
- **Input types**: In-memory arrays vs memmaps

Expected trade-offs:

| Aspect | LazyFieldRenderer | Dask |
|--------|-------------------|------|
| No new dependency | ✓ | ✗ (requires dask) |
| Custom caching control | ✓ | ✗ (dask manages) |
| Napari native support | ✗ | ✓ |
| Code complexity | More | Less |
| Tested in production | ✓ | Needs testing |

### Task 3.1: Update render_napari Type Hints and Input Handling

**File**: `src/neurospatial/animation/backends/napari_backend.py`

Update `render_napari` function signature to accept both formats:

```python
def render_napari(
    env: Environment,
    fields: Sequence[NDArray[np.float64]] | NDArray[np.float64],  # Accept both
    *,
    overlay_data: OverlayData | None = None,
    ...
) -> napari.Viewer:
```

Add input handling at the start of `render_napari`:

```python
# Determine input format
fields_is_array = isinstance(fields, np.ndarray) and fields.ndim == 2
n_frames = fields.shape[0] if fields_is_array else len(fields)

# Validate field types are consistent (only for list inputs)
if not fields_is_array:
    _validate_field_types_consistent(fields)
```

### Task 3.2: Update _create_lazy_field_renderer

**File**: `src/neurospatial/animation/backends/napari_backend.py`

Update function signature (around line 2086):

```python
def _create_lazy_field_renderer(
    env: Environment,
    fields: list[NDArray[np.float64]] | NDArray[np.float64],  # Accept both
    cmap_lookup: NDArray[np.uint8],
    vmin: float,
    vmax: float,
    cache_size: int = DEFAULT_CACHE_SIZE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> LazyFieldRenderer | ChunkedLazyFieldRenderer:
```

Update n_frames calculation:

```python
n_frames = fields.shape[0] if isinstance(fields, np.ndarray) else len(fields)
```

### Task 3.3: Update LazyFieldRenderer for Array Input

**File**: `src/neurospatial/animation/backends/napari_backend.py`

Update `LazyFieldRenderer.__init__` (around line 2185):

```python
def __init__(
    self,
    env: Environment,
    fields: list[NDArray[np.float64]] | NDArray[np.float64],  # Accept both
    cmap_lookup: NDArray[np.uint8],
    vmin: float,
    vmax: float,
    cache_size: int = DEFAULT_CACHE_SIZE,
) -> None:
    self.env = env
    self.fields = fields
    self._fields_is_array = isinstance(fields, np.ndarray)
    # ... rest unchanged
```

Update `__len__` method:

```python
def __len__(self) -> int:
    return self.fields.shape[0] if self._fields_is_array else len(self.fields)
```

Update `_get_frame` method (line ~2304):

```python
# Access field via row indexing (works for both list and array)
field = self.fields[idx]  # Array slicing is a view (cheap for memmaps)
```

### Task 3.4: Update ChunkedLazyFieldRenderer for Array Input

**File**: `src/neurospatial/animation/backends/napari_backend.py`

Same pattern as LazyFieldRenderer:

```python
def __init__(
    self,
    env: Environment,
    fields: list[NDArray[np.float64]] | NDArray[np.float64],
    ...
) -> None:
    self.fields = fields
    self._fields_is_array = isinstance(fields, np.ndarray)
    # ...
```

Update `_render_chunk` to use slicing for arrays:

```python
def _render_chunk(self, chunk_idx: int) -> list[NDArray[np.uint8]]:
    start_frame = chunk_idx * self._chunk_size
    end_frame = min(start_frame + self._chunk_size, len(self))

    # Use slice for arrays (cheap view on memmaps)
    if self._fields_is_array:
        chunk_fields = self.fields[start_frame:end_frame]  # View, not copy
    else:
        chunk_fields = self.fields[start_frame:end_frame]  # List slicing

    frames = []
    for idx, field in enumerate(chunk_fields):
        rgb = field_to_rgb_for_napari(self.env, field, self.cmap_lookup, self.vmin, self.vmax)
        frames.append(rgb)
    return frames
```

**Verification**:

```bash
uv run pytest tests/animation/ -v -k "napari"
```

---

## Phase 4: Make Colormap Range Computation Scale

**Goal**: Add streaming/chunked computation for large arrays to avoid loading everything into RAM.

**CRITICAL (from napari docs)**: Omitting `contrast_limits` (vmin/vmax) causes napari to compute min/max over the **entire dataset**, resulting in "extremely long processing times" for large data. Always pass explicit limits for large datasets.

**Also important**: Set `multiscale=False` when adding large Image layers to avoid unnecessary pyramid computation.

### Task 4.1: Add Streaming Path to compute_global_colormap_range

**File**: `src/neurospatial/animation/rendering.py`

Update `compute_global_colormap_range` (line ~29):

```python
def compute_global_colormap_range(
    fields: list[NDArray[np.float64]] | NDArray[np.float64],
    vmin: float | None = None,
    vmax: float | None = None,
    *,
    max_frames_for_exact: int = 50_000,  # NEW: threshold for streaming
    sample_stride: int | None = None,    # NEW: optional stride for sampling
) -> tuple[float, float]:
    """Compute NaN-robust color scale across all fields.

    Parameters
    ----------
    ...
    max_frames_for_exact : int, default=50_000
        For arrays with more frames, use chunked streaming to avoid RAM spike.
    sample_stride : int, optional
        If set, sample every Nth frame for range estimation (faster but approximate).
        Useful for very large datasets (>200K frames). Default is None (exact).
    """
```

Add streaming implementation:

```python
# Check if we should use streaming for large arrays
if isinstance(fields, np.ndarray) and fields.ndim == 2:
    n_frames = fields.shape[0]

    # Use sampling if stride provided
    if sample_stride is not None:
        fields_to_scan = fields[::sample_stride]
        n_frames = fields_to_scan.shape[0]
    else:
        fields_to_scan = fields

    # Streaming path for large arrays
    if n_frames > max_frames_for_exact:
        all_min = float("inf")
        all_max = float("-inf")
        chunk_size = 10_000  # Process 10K frames at a time

        for start in range(0, n_frames, chunk_size):
            end = min(start + chunk_size, n_frames)
            chunk = fields_to_scan[start:end]  # View for memmaps

            finite_mask = np.isfinite(chunk)
            if finite_mask.any():
                all_min = min(all_min, float(chunk[finite_mask].min()))
                all_max = max(all_max, float(chunk[finite_mask].max()))

        vmin = vmin if vmin is not None else all_min
        vmax = vmax if vmax is not None else all_max
        # Continue to degenerate case handling below
    else:
        # Existing vectorized path for smaller arrays
        stacked = fields_to_scan
        # ... existing logic
```

### Task 4.2: Update render_napari to Use Streaming for Large Datasets

**File**: `src/neurospatial/animation/backends/napari_backend.py`

In `render_napari`, update colormap range computation:

```python
# Compute global range (supports streaming for large arrays)
from neurospatial.animation.rendering import compute_global_colormap_range

# For very large datasets, use sampling to estimate range quickly
sample_stride = None
if n_frames > 200_000 and vmin is None and vmax is None:
    sample_stride = max(1, n_frames // 50_000)  # Sample ~50K frames
    warnings.warn(
        f"Large dataset ({n_frames:,} frames): estimating colormap range from "
        f"1/{sample_stride} of frames. For exact range, pass explicit vmin/vmax.",
        UserWarning,
        stacklevel=2,
    )

vmin, vmax = compute_global_colormap_range(
    fields, vmin=vmin, vmax=vmax, sample_stride=sample_stride
)
```

**Verification**:

```bash
# Unit test
uv run pytest tests/animation/test_rendering.py -v -k "colormap_range"

# Manual test with large array
uv run python -c "
import numpy as np
from neurospatial.animation.rendering import compute_global_colormap_range

# Simulate 500K frames
fields = np.random.rand(500_000, 100).astype(np.float64)
vmin, vmax = compute_global_colormap_range(fields, sample_stride=10)
print(f'Range: [{vmin:.4f}, {vmax:.4f}]')
"
```

### Task 4.3: Always Set multiscale=False

**File**: `src/neurospatial/animation/backends/napari_backend.py`

**What to do**:

1. Find where `viewer.add_image()` is called for the field layer
2. Add `multiscale=False` parameter unconditionally
3. This prevents napari from computing image pyramids (not needed for spatial fields)

Note: napari's default is already non-multiscale for single 4D arrays, but setting this explicitly ensures consistent behavior and documents the intent.

**Success criteria**:

- `viewer.add_image(..., multiscale=False)` present in code
- No regressions in existing tests

---

## Phase 5: Overlay Subsampling (DEFERRED)

**Status**: DEFERRED - Not required for core large-session support. May revisit if overlay memory becomes a bottleneck.

**Goal**: Add subsampling options for overlays without affecting fields.

### Task 5.1: Add Overlay Subsampling Parameters

**File**: `src/neurospatial/animation/core.py`

Add new parameters to `animate_fields`:

```python
def animate_fields(
    env: EnvironmentProtocol,
    fields: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    *,
    # ... existing params ...
    overlay_subsample: int | None = None,  # NEW: subsample overlays
    max_overlay_frames: int | None = None,  # NEW: cap overlay frames
    **kwargs: Any,
) -> Any:
    """
    Parameters
    ----------
    overlay_subsample : int, optional
        If set, subsample overlay data by taking every Nth frame. Does not affect
        field data. Useful for reducing memory when overlays are at higher rate
        than needed for visualization. Default is None (no subsampling).
    max_overlay_frames : int, optional
        Maximum number of overlay frames to keep. If overlay has more frames,
        uniformly subsample to this limit. Default is None (no limit).
    """
```

### Task 5.2: Implement Overlay Subsampling in Conversion

**File**: `src/neurospatial/animation/overlays.py`

Add subsampling to `_convert_overlays_to_data`:

```python
def _convert_overlays_to_data(
    overlays: list[OverlayProtocol],
    frame_times: NDArray[np.float64],
    n_frames: int,
    env: Any,
    *,
    overlay_subsample: int | None = None,  # NEW
    max_overlay_frames: int | None = None,  # NEW
) -> list[PositionData | BodypartData | HeadDirectionData | VideoData]:
    """Convert overlays to internal data representation.

    Parameters
    ----------
    overlay_subsample : int, optional
        Subsample overlay data by taking every Nth sample before interpolation.
    max_overlay_frames : int, optional
        Cap overlay frames to this maximum (uniform subsampling).
    """
    # Apply subsampling to frame_times if requested
    effective_frame_times = frame_times
    if max_overlay_frames is not None and n_frames > max_overlay_frames:
        stride = max(1, n_frames // max_overlay_frames)
        indices = np.arange(0, n_frames, stride)[:max_overlay_frames]
        effective_frame_times = frame_times[indices]
    elif overlay_subsample is not None:
        indices = np.arange(0, n_frames, overlay_subsample)
        effective_frame_times = frame_times[indices]

    # Convert each overlay using effective_frame_times
    # ...
```

### Task 5.3: Pass Overlay Parameters Through

**File**: `src/neurospatial/animation/core.py`

Update overlay conversion call:

```python
if overlays is not None:
    overlay_data = _convert_overlays_to_data(
        overlays=overlays,
        frame_times=frame_times,
        n_frames=n_frames,
        env=env,
        overlay_subsample=overlay_subsample,
        max_overlay_frames=max_overlay_frames,
    )
```

**Verification**:

```bash
uv run pytest tests/animation/test_overlays.py -v -k "subsample"
```

---

## Phase 6: Long-Session Helpers and Ergonomics

**Goal**: Add utility functions for large-session workflows.

### Task 6.1: Add estimate_colormap_range_from_subset Helper

**File**: `src/neurospatial/animation/core.py`

```python
def estimate_colormap_range_from_subset(
    fields: NDArray[np.float64] | Sequence[NDArray[np.float64]],
    n_samples: int = 10_000,
    percentile: tuple[float, float] = (1.0, 99.0),
) -> tuple[float, float]:
    """Estimate colormap range from a random subset of frames.

    Useful for pre-computing vmin/vmax for large datasets without scanning
    all frames. Uses percentile-based range to exclude outliers.

    Parameters
    ----------
    fields : ndarray of shape (n_frames, n_bins) or list of arrays
        Field data to estimate range from.
    n_samples : int, default=10_000
        Number of frames to sample for estimation.
    percentile : tuple of float, default=(1.0, 99.0)
        Percentile range to use (excludes outliers).

    Returns
    -------
    vmin : float
        Estimated minimum value.
    vmax : float
        Estimated maximum value.

    Examples
    --------
    >>> fields = np.memmap("large_session.dat", dtype=np.float32, shape=(900_000, 500))
    >>> vmin, vmax = estimate_colormap_range_from_subset(fields)
    >>> env.animate_fields(fields, vmin=vmin, vmax=vmax, backend="napari")
    """
    if isinstance(fields, np.ndarray):
        n_frames = fields.shape[0]
    else:
        n_frames = len(fields)

    # Sample random frames
    rng = np.random.default_rng(42)  # Reproducible
    sample_indices = rng.choice(n_frames, size=min(n_samples, n_frames), replace=False)
    sample_indices.sort()

    # Gather samples
    if isinstance(fields, np.ndarray):
        samples = fields[sample_indices].ravel()
    else:
        samples = np.concatenate([fields[i] for i in sample_indices])

    # Compute percentile-based range
    finite_samples = samples[np.isfinite(samples)]
    if len(finite_samples) == 0:
        return 0.0, 1.0

    vmin, vmax = np.percentile(finite_samples, percentile)
    return float(vmin), float(vmax)
```

### Task 6.2: Add large_session_napari_config Helper

**File**: `src/neurospatial/animation/core.py`

```python
def large_session_napari_config(
    n_frames: int,
    sample_rate_hz: float | None = None,
) -> dict[str, Any]:
    """Get recommended napari configuration for large sessions.

    Returns a dict of kwargs suitable for animate_fields() based on
    the dataset size.

    Parameters
    ----------
    n_frames : int
        Number of frames in the session.
    sample_rate_hz : float, optional
        Original sample rate. If provided, suggests appropriate playback fps.

    Returns
    -------
    config : dict
        Recommended kwargs for animate_fields(backend="napari", **config).
        Keys may include: fps, chunk_size, max_chunks.

    Examples
    --------
    >>> config = large_session_napari_config(n_frames=900_000, sample_rate_hz=250)
    >>> env.animate_fields(fields, backend="napari", **config)
    """
    config: dict[str, Any] = {}

    # FPS recommendation
    if sample_rate_hz is not None:
        # Cap at 60 fps for smooth playback
        config["fps"] = min(60, int(sample_rate_hz / 4))  # 4x slower than real-time
    else:
        config["fps"] = 30

    # Chunk size: larger for bigger datasets (more sequential access benefit)
    if n_frames > 500_000:
        config["chunk_size"] = 100
        config["max_chunks"] = 500
    elif n_frames > 100_000:
        config["chunk_size"] = 50
        config["max_chunks"] = 300
    elif n_frames > 10_000:
        config["chunk_size"] = 20
        config["max_chunks"] = 200

    # Note: overlay_subsample is available but deferred (Phase 5)
    # Uncomment when Phase 5 is implemented:
    # if n_frames > 500_000:
    #     config["overlay_subsample"] = 10
    # elif n_frames > 100_000:
    #     config["overlay_subsample"] = 5

    return config
```

### Task 6.3: Export Helpers from Animation Module

**File**: `src/neurospatial/animation/__init__.py`

```python
from neurospatial.animation.core import (
    animate_fields,
    estimate_colormap_range_from_subset,
    large_session_napari_config,
    subsample_frames,
)
```

**Verification**:

```bash
uv run python -c "
from neurospatial.animation import estimate_colormap_range_from_subset, large_session_napari_config
print(large_session_napari_config(900_000, 250))
"
```

---

## Phase 7: Tests and Documentation

### Task 7.1: Add Tests for Array Input to Napari Backend

**File**: `tests/animation/test_napari_backend.py` (create or extend)

```python
import numpy as np
import pytest

from neurospatial import Environment


class TestNapariArrayInput:
    """Test napari backend with 2D array and memmap inputs."""

    @pytest.fixture
    def env(self):
        positions = np.random.randn(100, 2) * 50
        return Environment.from_samples(positions, bin_size=5.0)

    def test_2d_array_input(self, env):
        """Test that 2D arrays are passed through without list conversion."""
        fields = np.random.rand(50, env.n_bins).astype(np.float64)

        # This should work and not blow up memory
        from neurospatial.animation.backends.napari_backend import _create_lazy_field_renderer
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("viridis")
        cmap_lookup = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

        renderer = _create_lazy_field_renderer(env, fields, cmap_lookup, 0, 1)
        assert len(renderer) == 50

    def test_memmap_input(self, env, tmp_path):
        """Test memmap arrays work correctly with chunked access."""
        mmap_path = tmp_path / "test.dat"
        fields = np.memmap(str(mmap_path), dtype="float64", mode="w+", shape=(100, env.n_bins))
        fields[:] = np.random.rand(100, env.n_bins)
        fields.flush()

        # Should use efficient slicing, not load everything
        from neurospatial.animation.backends.napari_backend import _create_lazy_field_renderer
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("viridis")
        cmap_lookup = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

        renderer = _create_lazy_field_renderer(env, fields, cmap_lookup, 0, 1)

        # Access a few frames
        _ = renderer[0]
        _ = renderer[50]
        _ = renderer[99]
```

### Task 7.2: Add Tests for Streaming Colormap Range

**File**: `tests/animation/test_rendering.py` (extend)

```python
def test_streaming_colormap_range_large_array():
    """Test streaming path for large arrays doesn't blow RAM."""
    # Create array that exceeds max_frames_for_exact threshold (50K)
    # Use 20K×200 to keep test fast and CI-friendly (~32MB)
    fields = np.random.rand(20_000, 200).astype(np.float64)

    # Should use streaming, not stack everything
    from neurospatial.animation.rendering import compute_global_colormap_range
    vmin, vmax = compute_global_colormap_range(fields, max_frames_for_exact=10_000)

    assert np.isfinite(vmin)
    assert np.isfinite(vmax)
    assert vmin < vmax

def test_sampled_colormap_range():
    """Test sampling path with stride."""
    # 20K×200 is enough to test sampling logic (~32MB)
    fields = np.random.rand(20_000, 200).astype(np.float64)

    from neurospatial.animation.rendering import compute_global_colormap_range
    vmin, vmax = compute_global_colormap_range(fields, sample_stride=10)

    # Range should be approximately correct (within a few percent of true range)
    # For uniform random [0, 1), min approaches 0 and max approaches 1
    assert vmin < 0.05  # Should be close to 0
    assert vmax > 0.95  # Should be close to 1
```

### Task 7.3: Add Tests for Overlay Subsampling (DEFERRED)

**Status**: DEFERRED - Depends on Phase 5 which is deferred.

**File**: `tests/animation/test_overlays.py` (extend)

```python
# DEFERRED: Only implement if Phase 5 is prioritized
def test_overlay_subsample():
    """Test overlay subsampling reduces data."""
    from neurospatial.animation.overlays import _convert_overlays_to_data, PositionOverlay

    # Create overlay with 1000 frames
    overlay = PositionOverlay(
        data=np.random.rand(1000, 2),
        color="red",
    )

    frame_times = np.arange(100) / 30.0  # 100 animation frames

    # Without subsampling
    result = _convert_overlays_to_data([overlay], frame_times, 100, mock_env)
    assert result[0].data.shape[0] == 100

    # With subsampling
    result = _convert_overlays_to_data(
        [overlay], frame_times, 100, mock_env, overlay_subsample=2
    )
    assert result[0].data.shape[0] == 50  # Half the frames
```

### Task 7.4: Update Documentation

**File**: `docs/user-guide/animation.md`

Add section:

```markdown
## Large Datasets (100K+ Frames)

For long decoding sessions (60+ minutes at 250 Hz), use these recommended settings:

### Pre-compute Colormap Range

```python
from neurospatial.animation import estimate_colormap_range_from_subset

# Sample 10K frames to estimate range (fast, ~0.1s)
vmin, vmax = estimate_colormap_range_from_subset(fields)

env.animate_fields(fields, vmin=vmin, vmax=vmax, backend="napari")
```

### Use Memory-Mapped Arrays

```python
# Create memmap for 900K frames
fields = np.memmap("session.dat", dtype="float32", mode="r", shape=(900_000, env.n_bins))

# Get recommended config
from neurospatial.animation import large_session_napari_config
config = large_session_napari_config(n_frames=900_000, sample_rate_hz=250)

env.animate_fields(fields, backend="napari", **config)
```

### Recommended Settings by Session Length

| Session | Frames | chunk_size | max_chunks | Notes |
|---------|--------|------------|------------|-------|
| <10 min | <150K  | 20         | 200        | Default |
| 10-30 min | 150K-450K | 50 | 300 | Larger chunks |
| 30-60 min | 450K-900K | 100 | 500 | Pre-compute vmin/vmax |
| >60 min | >900K | 100 | 500 | Use memmaps, pre-compute vmin/vmax |

```

**Verification**:
```bash
# Run all animation tests
uv run pytest tests/animation/ -v

# Build docs to check formatting
# (if using mkdocs)
uv run mkdocs build
```

---

## Implementation Order

1. **Phase 2** (core.py array preservation) - Foundation for all other changes
2. **Phase 3** (napari backend array support) - Enables memmap passthrough
3. **Phase 4** (colormap scaling) - Critical for avoiding OOM on large datasets
4. **Phase 1** (example scripts) - Can be done in parallel, demonstrates patterns
5. **Phase 6** (helpers) - Convenience, can be last
6. **Phase 7** (tests/docs) - Throughout, but finalize at end
7. ~~**Phase 5** (overlay subsampling)~~ - DEFERRED: Not required for core support

## Verification Checklist

- [ ] `uv run pytest tests/animation/` passes
- [ ] `data/large_session_napari_example.py` runs with 100K+ frames without OOM
- [ ] `data/view_bandit_napari.py` still works (backward compatibility)
- [ ] Memmap arrays not converted to list (check with memory profiling)
- [ ] Colormap range computed in streaming fashion for >50K frames
- [ ] Documentation updated with new patterns
