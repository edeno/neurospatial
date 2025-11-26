# Animation Subpackage Refactoring Plan

**Created**: 2025-11-26
**Based on**: Code Review (8.5/10) and UX Review (Needs Polish)
**Scope**: `src/neurospatial/animation/`

## Executive Summary

The animation subpackage is production-ready with excellent architecture and error handling. This plan addresses 3 critical bugs, 5 important issues, and 8 UX improvements identified during review.

---

## Phase 1: Critical Bug Fixes (Must Fix)

### 1.1 Empty Bodyparts Validation in Skeleton Rendering

**File**: `backends/napari_backend.py:285-313`

**Problem**: `_build_skeleton_vectors` checks for empty skeleton but doesn't validate that `bodyparts` dict is non-empty when skeleton edges exist. If `bodyparts` is empty but `skeleton.edges` is non-empty, the function will fail when accessing `napari_coords[part_name]`.

**Fix**:

```python
# After line 288, move the empty bodyparts check BEFORE skeleton edge processing
if not bodyparts:
    empty_vectors = np.empty((0, 2, 3), dtype=dtype)
    empty_features = {"edge_name": np.empty(0, dtype=object)}
    return empty_vectors, empty_features

# Then proceed with skeleton_edges = skeleton.edges
```

**Test**: Add test case with empty bodyparts dict + non-empty skeleton edges.

---

### 1.2 Thread Safety for VideoReader Pickle

**File**: `_video_io.py:156-327`

**Problem**: `__getstate__` and `__setstate__` manipulate `_cached_read_frame` without thread safety. Parallel rendering with `n_workers > 1` could trigger race conditions.

**Fix**:

```python
from threading import Lock

class VideoReader:
    def __init__(self, ...):
        # ... existing code ...
        self._lock = Lock()
        self._setup_cache()

    def __getstate__(self) -> dict[str, Any]:
        with self._lock:
            state = self.__dict__.copy()
            state.pop("_cached_read_frame", None)
            state.pop("_lock", None)  # Don't pickle the lock
            return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = Lock()
        self._setup_cache()
```

**Test**: Add concurrent pickle/unpickle test with ThreadPoolExecutor.

---

### 1.3 Integer Overflow in `_find_nearest_indices`

**File**: `overlays.py:2076`

**Problem**: `searchsorted` returns `intp` (platform-dependent), which could overflow on 32-bit systems with >2^31 frames.

**Fix**:

```python
idx_right = np.searchsorted(t_src, t_valid, side="left").astype(np.int64)
idx_right = np.clip(idx_right, 1, len(t_src) - 1)
```

**Test**: Add test with simulated large frame count (mock the array length check).

---

## Phase 2: Important Code Quality Issues

### 2.1 Vectorize Coordinate Transforms for Pose Tracking

**File**: `backends/napari_backend.py:296-300`

**Problem**: Per-bodypart coordinate transforms in a loop create overhead for 20+ keypoint pose tracking.

**Current**:

```python
for part_name, coords in bodyparts.items():
    napari_coords[part_name] = _transform_coords_for_napari(
        coords, env_scale if env_scale is not None else env
    )
```

**Fix**:

```python
def _batch_transform_bodyparts(
    bodyparts: dict[str, NDArray[np.float64]],
    env_or_scale: Environment | tuple[float, float],
) -> dict[str, NDArray[np.float64]]:
    """Batch transform all bodypart coordinates in single operation."""
    if not bodyparts:
        return {}

    # Concatenate all coordinates
    names = list(bodyparts.keys())
    coords_list = [bodyparts[name] for name in names]
    lengths = [len(c) for c in coords_list]
    all_coords = np.concatenate(coords_list, axis=0)

    # Single transform call
    all_transformed = _transform_coords_for_napari(all_coords, env_or_scale)

    # Split back to dict
    result = {}
    offset = 0
    for name, length in zip(names, lengths):
        result[name] = all_transformed[offset:offset + length]
        offset += length
    return result
```

**Expected speedup**: 5-10x for typical pose tracking (20 bodyparts, 40K frames).

---

### 2.2 Add Upper Bound Validation for `downsample`

**File**: `overlays.py:837-842`

**Fix**:

```python
if not isinstance(self.downsample, int) or self.downsample < 1:
    raise ValueError(
        f"WHAT: downsample must be a positive integer >= 1, got {self.downsample}.\n"
        f"WHY: Downsample factor controls spatial resolution reduction.\n"
        f"HOW: Use downsample=1 (full resolution) or downsample=2 (half resolution)."
    )

if self.downsample > 16:
    warnings.warn(
        f"downsample={self.downsample} is very large and may produce unusable video. "
        f"Typical values are 1-4.",
        UserWarning,
        stacklevel=2,
    )
```

---

### 2.3 Defensive Validation in `field_to_rgb_for_napari`

**File**: `rendering.py:483-485`

**Fix**:

```python
active_indices = env.layout.active_mask.flatten()
expected_active_count = np.sum(active_indices)

if len(rgb) != expected_active_count:
    raise ValueError(
        f"WHAT: RGB array has {len(rgb)} entries but active_mask indicates "
        f"{expected_active_count} active bins.\n"
        f"WHY: Environment structure may have changed after field computation.\n"
        f"HOW: Re-compute fields after any environment modifications."
    )

full_rgb_flat = full_rgb.reshape(-1, 3)
full_rgb_flat[active_indices] = rgb
```

---

### 2.4 Warn When `flip_y` Used with Landmark Calibration

**File**: `calibration.py:241-274`

**Fix**:

```python
elif landmarks_px is not None or landmarks_env is not None:
    # ... existing validation ...

    if flip_y is not True:  # Default is True
        warnings.warn(
            "flip_y parameter is ignored when using landmark calibration. "
            "Y-axis flip is implicit in the landmark correspondences. "
            "If video appears inverted, swap Y coordinates in landmarks_env.",
            UserWarning,
            stacklevel=2,
        )

    transform = calibrate_from_landmarks(...)
```

---

### 2.5 Validate Sorted Timestamps in `_interp_linear`

**File**: `overlays.py:1978-2010`

**Fix**:

```python
def _interp_linear(
    t_src: NDArray[np.float64],
    x_src: NDArray[np.float64],
    t_frame: NDArray[np.float64],
) -> NDArray[np.float64]:
    if len(t_src) > 1 and not np.all(np.diff(t_src) > 0):
        raise ValueError(
            "t_src must be monotonically increasing for interpolation. "
            "This is an internal bug - please report this issue."
        )
    # ... rest of function ...
```

---

## Phase 3: UX Improvements

### 3.1 Add Progress Bar for Video Rendering

**File**: `backends/video_backend.py`

**Dependency**: Add `tqdm` to optional dependencies (or use it if already present).

**Fix**:

```python
from tqdm.auto import tqdm

def render_video(...):
    # ... existing setup ...

    with tqdm(total=len(fields), desc="Rendering frames", unit="frame") as pbar:
        for chunk_result in process_chunks(fields, ...):
            pbar.update(len(chunk_result))
            # ... existing processing ...
```

---

### 3.2 Print Backend Selection to stdout

**File**: `core.py:370-467`

**Fix**: Add print statement in `_select_backend()`:

```python
def _select_backend(...) -> str:
    # ... existing logic ...

    if selected == "napari":
        reason = f"{n_frames:,} frames (GPU-accelerated)"
    elif selected == "widget":
        reason = "Jupyter environment detected"
    elif selected == "video":
        reason = f"save_path={save_path}"
    else:
        reason = "default"

    print(f"Using '{selected}' backend ({reason})")
    logger.info(f"Auto-selected '{selected}' backend: {reason}")

    return selected
```

---

### 3.3 Improve Pickle-ability Error Messages

**File**: `core.py:40-55`

**Fix**:

```python
raise ValueError(
    f"WHAT: Cannot use parallel rendering (n_workers > 1).\n\n"
    f"WHY: Python requires 'pickling' to pass data between worker processes. "
    f"Your environment contains objects that can't be pickled.\n"
    f"  Technical detail: {type(e).__name__}: {e}\n\n"
    f"Common causes:\n"
    f"  - KDTree cache (call env.clear_cache() before rendering)\n"
    f"  - Lambda functions in custom overlays\n"
    f"  - Open file handles\n\n"
    f"HOW:\n"
    f"  1. Clear caches: env.clear_cache()\n"
    f"  2. Or use n_workers=1 (serial rendering, slower but works)"
)
```

---

### 3.4 Add Coordinate Validation Warning

**File**: `overlays.py` - in `_validate_bounds()` or new function

**Fix**: Add heuristic check for pre-converted coordinates:

```python
def _check_coordinate_convention(
    data: NDArray[np.float64],
    env: Environment,
    name: str,
) -> None:
    """Warn if coordinates appear to be pre-converted to napari format."""
    if env.n_dims != 2:
        return

    # Check if data looks swapped (x range matches y bounds and vice versa)
    data_x_range = (np.nanmin(data[:, 0]), np.nanmax(data[:, 0]))
    data_y_range = (np.nanmin(data[:, 1]), np.nanmax(data[:, 1]))

    env_x_range = env.dimension_ranges[0]
    env_y_range = env.dimension_ranges[1]

    # Heuristic: if x matches env_y and y matches env_x, likely swapped
    x_matches_env_y = _ranges_overlap(data_x_range, env_y_range, threshold=0.8)
    y_matches_env_x = _ranges_overlap(data_y_range, env_x_range, threshold=0.8)
    x_matches_env_x = _ranges_overlap(data_x_range, env_x_range, threshold=0.5)

    if x_matches_env_y and y_matches_env_x and not x_matches_env_x:
        warnings.warn(
            f"Overlay '{name}' coordinates may have been manually swapped to napari format.\n"
            f"  Data X range: {data_x_range} matches environment Y range: {env_y_range}\n"
            f"  Data Y range: {data_y_range} matches environment X range: {env_x_range}\n\n"
            f"HOW: Pass coordinates in environment (x, y) format - the animation system "
            f"handles napari conversion automatically. Do NOT pre-swap coordinates.",
            UserWarning,
            stacklevel=3,
        )
```

---

### 3.5 Add Frame-by-Frame NaN Diagnostics

**File**: `overlays.py` - after temporal alignment

**Fix**: When NaN values are produced due to extrapolation, log diagnostic info:

```python
def _diagnose_nan_frames(
    result: NDArray[np.float64],
    frame_times: NDArray[np.float64],
    overlay_times: tuple[float, float],
    name: str,
) -> None:
    """Log diagnostic info about NaN frames from extrapolation."""
    nan_mask = np.any(np.isnan(result), axis=tuple(range(1, result.ndim)))
    if not np.any(nan_mask):
        return

    nan_indices = np.where(nan_mask)[0]
    first_nan = nan_indices[0]
    last_nan = nan_indices[-1]

    logger.info(
        f"Overlay '{name}': {len(nan_indices)} frames have NaN values "
        f"(frames {first_nan}-{last_nan}) - outside overlay time range "
        f"[{overlay_times[0]:.3f}, {overlay_times[1]:.3f}]"
    )
```

---

### 3.6 File Overwrite Confirmation (Optional)

**File**: `backends/video_backend.py`

**Fix**: Add optional confirmation for existing files:

```python
def render_video(
    ...,
    overwrite: bool = False,
) -> Path | None:
    save_path = Path(save_path)

    if save_path.exists() and not overwrite:
        # In interactive context, ask for confirmation
        if sys.stdin.isatty():
            response = input(f"File {save_path} exists. Overwrite? [y/N]: ")
            if response.lower() != 'y':
                print("Cancelled. Use overwrite=True to skip confirmation.")
                return None
        else:
            raise FileExistsError(
                f"File {save_path} already exists. "
                f"Use overwrite=True to replace it."
            )
```

---

## Phase 4: Minor Improvements

### 4.1 Standardize Float Type Annotations

**Files**: Multiple

**Convention**: Use `float` for scalars, `np.float64` for array dtypes:

```python
# Scalars
def compute_range(...) -> tuple[float, float]:

# Arrays
def process_data(...) -> NDArray[np.float64]:
```

---

### 4.2 Document Magic Numbers in Rendering Constants

**File**: `backends/napari_backend.py:69-96`

**Fix**: Add docstrings explaining constant choices:

```python
REGION_CIRCLE_SEGMENTS: int = 9
"""Line segments for circle approximation (8 for octagon + 1 closing).
Provides good visual quality at typical zoom levels with minimal vertices."""
```

---

### 4.3 Move `_build_frame_times` to Shared Module

**Current**: `overlays.py:1776`

**Fix**: Create `src/neurospatial/animation/_timeline.py`:

```python
"""Timeline utilities shared across animation modules."""

def build_frame_times(
    n_frames: int,
    frame_times: NDArray[np.float64] | None,
    fps: float,
) -> NDArray[np.float64]:
    """Build or validate frame times array for animation."""
    # ... move implementation from overlays.py ...
```

Then import from both `overlays.py` and `core.py`.

---

### 4.4 Add Docstring to VideoOverlay.**post_init**

**File**: `overlays.py:818-861`

**Fix**:

```python
def __post_init__(self) -> None:
    """Validate VideoOverlay parameters after initialization.

    Performs the following validations:
    - alpha is in range [0.0, 1.0]
    - downsample is a positive integer >= 1
    - source array has correct shape (n_frames, H, W, 3) if provided
    - source array has dtype uint8 if provided

    Raises
    ------
    ValueError
        If any validation fails, with detailed WHAT/WHY/HOW guidance.
    """
```

---

## Implementation Order

### Sprint 1: Critical Fixes (1-2 days)

1. [ ] 1.1 Empty bodyparts validation
2. [ ] 1.2 VideoReader thread safety
3. [ ] 1.3 Integer overflow fix

### Sprint 2: Important Issues (2-3 days)

4. [ ] 2.1 Vectorize coordinate transforms
5. [ ] 2.2 Downsample validation
6. [ ] 2.3 Defensive RGB validation
7. [ ] 2.4 flip_y warning
8. [ ] 2.5 Sorted timestamp validation

### Sprint 3: UX Improvements (2-3 days)

9. [ ] 3.1 Progress bar for video rendering
10. [ ] 3.2 Print backend selection
11. [ ] 3.3 Better pickle error messages
12. [ ] 3.4 Coordinate validation warning

### Sprint 4: Polish (1-2 days)

13. [ ] 3.5 NaN frame diagnostics
14. [ ] 3.6 File overwrite confirmation
15. [ ] 4.1-4.4 Minor improvements

---

## Testing Requirements

Each fix should include:

1. Unit test for the specific issue
2. Regression test to prevent reintroduction
3. Update to existing tests if behavior changes

**New test files needed**:

- `tests/animation/test_thread_safety.py` - for 1.2
- `tests/animation/test_coordinate_validation.py` - for 3.4

---

## Success Criteria

- [ ] All critical issues fixed and tested
- [ ] Mypy passes without errors
- [ ] All existing tests pass
- [ ] Code review score improves to 9+/10
- [ ] UX review status changes to "Ready"
