# Lessons from movement Package Code

**Date**: 2025-11-06
**Analysis**: What neurospatial can learn from movement's implementation

---

## Overview

After analyzing movement's codebase, here are valuable design patterns, implementation strategies, and architectural decisions that could benefit neurospatial.

---

## 1. Data Structure Design: xarray ⭐⭐⭐

### What They Do

**xarray.DataArray as primary data structure**:

```python
# movement kinematics functions
def compute_velocity(data: xr.DataArray) -> xr.DataArray:
    """
    Compute velocity from position data.

    Parameters
    ----------
    data : xarray.DataArray
        Position data with dimensions (time, space, keypoints, individuals)

    Returns
    -------
    velocity : xarray.DataArray
        Velocity with same dimensions and labeled coordinates
    """
    validate_dims_coords(data, {"time": [], "space": []})
    return compute_time_derivative(data, order=1)
```

**Key benefits**:
- ✅ **Labeled dimensions** - `time`, `space`, `keypoints`, `individuals`
- ✅ **Automatic alignment** - Operations align by dimension labels
- ✅ **Metadata preservation** - Attributes travel with data
- ✅ **Self-documenting** - Data structure encodes meaning

### What Neurospatial Could Learn

**Current approach** (NumPy arrays):
```python
# neurospatial (current)
def occupancy(
    self,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Returns occupancy array, but loses metadata."""
    # Returns bare NumPy array - no labels, no metadata
```

**Potential improvement** (xarray):
```python
# neurospatial (potential)
def occupancy(
    self,
    trajectory: xr.Dataset,  # Has time, position, metadata
) -> xr.DataArray:
    """
    Returns occupancy with labeled dimensions.

    Returns
    -------
    occupancy : xr.DataArray
        Dimensions: (bins,)
        Coordinates: bin_centers (spatial coordinates)
        Attributes: units, frame, dt, etc.
    """
    times = trajectory.coords["time"].values
    positions = trajectory["position"].values

    occ = self._compute_occupancy(times, positions)

    # Return labeled array
    return xr.DataArray(
        occ,
        dims=["bins"],
        coords={"bin_centers": self.bin_centers},
        attrs={
            "units": self.units,
            "frame": self.frame,
            "dt": np.median(np.diff(times)),
        },
    )
```

**Benefits for neurospatial**:
- Occupancy carries spatial coordinates
- Firing rate = spikes / occupancy (auto-aligned)
- Metadata (units, frame) preserved through pipeline
- Self-documenting outputs

**Recommendation**: ⚠️ **Consider optional xarray support**
- Keep NumPy as primary interface (simplicity)
- Add `to_xarray()` / `from_xarray()` methods
- Return xarray when `return_xarray=True` flag set

**Effort**: 2-3 weeks (Phase 5.5)
**Risk**: Low (optional feature, doesn't break existing API)

---

## 2. Validation Architecture: attrs + Custom Validators ⭐⭐⭐

### What They Do

**Use attrs library for data classes**:

```python
from attrs import define, field
from attrs import validators as attrvalidators

@define
class ValidPosesDataset:
    """
    Validator for pose estimation datasets.

    Uses attrs for automatic validation and type checking.
    """
    position_array: np.ndarray = field(validator=_validate_position_array)
    confidence_array: np.ndarray = field(validator=_validate_confidence_array)
    individual_names: list[str] = field(
        converter=_convert_to_list_of_str,
        validator=attrvalidators.instance_of(list),
    )
    keypoint_names: list[str] = field(
        converter=_convert_to_list_of_str,
        validator=attrvalidators.instance_of(list),
    )
    fps: float | None = field(
        default=None,
        converter=_convert_fps_to_none_if_invalid,
    )

    @position_array.validator
    def _validate_position_array(self, attribute, value):
        """Custom validator for position arrays."""
        _validate_type_ndarray(value)
        _validate_array_shape(
            value,
            expected_shape=(None, None, len(self.keypoint_names), len(self.individual_names)),
        )
```

**Validation patterns**:

1. **Converters** - Transform input before validation
   ```python
   fps: float | None = field(converter=_convert_fps_to_none_if_invalid)
   # Converts negative fps to None automatically
   ```

2. **Validators** - Check constraints
   ```python
   @position_array.validator
   def _validate_shape(self, attribute, value):
       # Raises ValueError if shape is wrong
   ```

3. **Composition** - Chain converters and validators
   ```python
   field(
       converter=_convert_to_list_of_str,
       validator=attrvalidators.instance_of(list),
   )
   ```

### What Neurospatial Could Learn

**Current approach** (manual checks):
```python
# neurospatial (current)
def from_samples(
    cls,
    data: NDArray[np.float64],
    bin_size: float,
    **kwargs,
) -> Environment:
    """Manual validation scattered throughout method."""
    if bin_size <= 0:
        raise ValueError(f"bin_size must be positive (got {bin_size})")

    if data.ndim != 2:
        raise ValueError(f"data must be 2D (got {data.ndim}D)")

    # ... more validation throughout
```

**Potential improvement** (attrs):
```python
# neurospatial (potential)
from attrs import define, field, validators as v

@define
class EnvironmentConfig:
    """Configuration for creating Environment (validated automatically)."""
    data: np.ndarray = field(validator=[
        v.instance_of(np.ndarray),
        _validate_2d_or_3d,
    ])
    bin_size: float = field(validator=[
        v.instance_of(float),
        v.gt(0),  # Greater than 0
    ])
    units: str | None = field(
        default=None,
        validator=v.optional(v.in_(["cm", "m", "mm", "pixels"])),
    )
    bin_count_threshold: int = field(
        default=1,
        validator=[v.instance_of(int), v.ge(0)],
    )

# Usage
@classmethod
def from_samples(
    cls,
    data: NDArray[np.float64],
    bin_size: float,
    **kwargs,
) -> Environment:
    # Validation happens automatically in constructor
    config = EnvironmentConfig(
        data=data,
        bin_size=bin_size,
        units=kwargs.get("units"),
        bin_count_threshold=kwargs.get("bin_count_threshold", 1),
    )

    # All validation already done, proceed with confidence
    return cls._from_config(config)
```

**Benefits**:
- Validation logic centralized
- Type checking automatic
- Error messages consistent
- Less boilerplate in factory methods

**Recommendation**: ⚠️ **Consider for major refactor**
- Don't adopt attrs for current dataclass (breaking change)
- Use for **new modules** (segmentation, metrics)
- Evaluate in Phase 2.5 (behavioral segmentation)

**Example** (behavioral segmentation):
```python
@define
class RunConfig:
    """Configuration for detecting runs between regions."""
    trajectory_positions: np.ndarray = field(validator=_validate_2d_array)
    times: np.ndarray = field(validator=_validate_1d_array)
    source: str = field(validator=v.instance_of(str))
    target: str = field(validator=v.instance_of(str))
    min_duration: float = field(default=0.5, validator=v.gt(0))
    max_duration: float = field(default=10.0, validator=v.gt(0))

    @min_duration.validator
    def _validate_duration_order(self, attribute, value):
        if hasattr(self, 'max_duration') and value >= self.max_duration:
            raise ValueError("min_duration must be < max_duration")
```

**Effort**: Minimal (use for new modules only)
**Risk**: Low (doesn't affect existing code)

---

## 3. Logging Architecture: Structured Logging ⭐⭐

### What They Do

**Custom logger with dual output**:

```python
# movement/__init__.py
from movement.utils.logging import logger

# Configure logging to both stderr and file
logger.configure()

# Usage throughout codebase
logger.error(ValueError("Invalid input"))
logger.warning("Missing metadata, using defaults")
```

**Patterns**:

1. **Raise via logger.error()**
   ```python
   # Logs AND raises
   raise logger.error(ValueError("bin_size must be positive"))
   ```

2. **Warnings via logger.warning()**
   ```python
   # Just logs, doesn't raise
   logger.warning("Low confidence detected, consider filtering")
   ```

3. **Validation warnings**
   ```python
   def _warn_about_nan_proportion(data, threshold=0.1):
       """Warn if too much missing data."""
       nan_ratio = np.isnan(data).sum() / data.size
       if nan_ratio > threshold:
           logger.warning(
               f"Data contains {nan_ratio:.1%} NaN values "
               f"(threshold: {threshold:.1%})"
           )
   ```

### What Neurospatial Could Learn

**Current approach** (standard exceptions):
```python
# neurospatial (current)
if bin_size <= 0:
    raise ValueError(f"bin_size must be positive (got {bin_size})")

# No logging, just raises
```

**Potential improvement**:
```python
# neurospatial (potential)
from neurospatial.utils.logging import logger

if bin_size <= 0:
    raise logger.error(
        ValueError(f"bin_size must be positive (got {bin_size})")
    )
    # Logs error THEN raises - helps debugging

# Warning for suspicious but valid inputs
if bin_size > 100:
    logger.warning(
        f"Unusually large bin_size ({bin_size}) - "
        f"this will create very coarse discretization"
    )
```

**Benefits**:
- Error messages logged to file (debugging)
- Warnings visible without raising
- Centralized logging configuration
- Production diagnostics easier

**Recommendation**: ✅ **Add in Phase 5**
- Create `src/neurospatial/utils/logging.py`
- Integrate with existing validation
- Log to both stderr and `~/.neurospatial/logs/`

**Effort**: 2-3 days
**Risk**: Low (backward compatible)

---

## 4. Composition Over Inheritance: Wrapping shapely ⭐⭐

### What They Do

**Wrap shapely.Geometry instead of inheriting**:

```python
from shapely.geometry import Polygon as ShapelyPolygon

class Polygon(BaseRegionOfInterest):
    """
    Polygon ROI wrapping shapely.Polygon.

    We wrap rather than inherit to avoid shapely's __new__() complexity.
    """
    def __init__(self, vertices: list[tuple[float, float]]):
        # Create shapely polygon
        shapely_polygon = ShapelyPolygon(vertices)

        # Store as attribute (composition)
        self._shapely_geometry = shapely_polygon

        super().__init__(...)

    def contains_point(self, point):
        """Delegate to shapely."""
        return self._shapely_geometry.contains(point)
```

**Why composition**:
- shapely classes have complex `__new__()` methods
- Wrapping avoids inheritance complications
- Can easily swap geometry implementations
- Clear separation of concerns

### What Neurospatial Could Learn

**Current approach** (dict storage):
```python
# neurospatial Regions (current)
class Regions:
    """Dict-like container for Region objects."""
    def __init__(self):
        self._regions: dict[str, Region] = {}

    # Dict-like interface
    def __getitem__(self, name: str) -> Region:
        return self._regions[name]
```

**This is already good!** Neurospatial uses composition effectively.

**Similar pattern** (wrapping NetworkX):
```python
# Environment wraps nx.Graph (composition)
class Environment:
    def __init__(self, ...):
        self.connectivity: nx.Graph = nx.Graph()
        # Doesn't inherit from nx.Graph
```

**Recommendation**: ✅ **Keep current approach**
- Composition already used correctly
- No changes needed

---

## 5. Functional Composition: Small, Chainable Functions ⭐⭐⭐

### What They Do

**Small functions that compose**:

```python
# movement kinematics

def compute_time_derivative(data, order=1):
    """Generalized nth-order derivative."""
    # Central differences implementation
    ...

def compute_velocity(data):
    """First derivative of position."""
    return compute_time_derivative(data, order=1)

def compute_acceleration(data):
    """Second derivative of position."""
    return compute_time_derivative(data, order=2)

def compute_speed(data):
    """Magnitude of velocity."""
    velocity = compute_velocity(data)
    return compute_norm(velocity, dims="space")

def compute_path_length(data, start, stop):
    """Sum of displacement magnitudes."""
    displacement = compute_forward_displacement(data)
    displacement_norm = compute_norm(displacement, dims="space")
    return displacement_norm.sel(time=slice(start, stop)).sum()
```

**Pattern**: Each function does ONE thing, composes with others.

**Benefits**:
- Easy to test (single responsibility)
- Easy to understand (small scope)
- Reusable (compose in different ways)
- Debuggable (clear data flow)

### What Neurospatial Could Learn

**Current approach** (some long methods):
```python
# neurospatial (current)
def from_samples(cls, data, bin_size, **kwargs):
    """
    Long method (100+ lines) doing:
    - Validation
    - Layout creation
    - Environment initialization
    - Region setup
    - Return
    """
    # Lots of logic in one method
```

**Potential improvement**:
```python
# neurospatial (potential refactor)

def from_samples(cls, data, bin_size, **kwargs):
    """High-level orchestration."""
    # Validate inputs
    config = _validate_sampling_config(data, bin_size, **kwargs)

    # Create layout
    layout = _create_layout_from_samples(config)

    # Initialize environment
    env = cls._from_layout(layout, config)

    # Setup regions if provided
    if config.regions:
        _setup_regions(env, config.regions)

    return env

def _validate_sampling_config(data, bin_size, **kwargs):
    """Separate validation logic."""
    ...

def _create_layout_from_samples(config):
    """Separate layout creation."""
    ...
```

**Benefits**:
- Each helper is testable independently
- Easier to understand data flow
- Can reuse helpers in other factory methods
- Debugging easier (clear failure point)

**Recommendation**: ⚠️ **Refactor in Phase 5 polish**
- Break up long methods (> 50 lines)
- Extract validation helpers
- Extract layout creation helpers

**Effort**: 1 week (polish phase)
**Risk**: Low (internal refactor, same API)

---

## 6. Validation-First Philosophy: validate_dims_coords ⭐⭐⭐

### What They Do

**Validate inputs BEFORE processing**:

```python
def compute_velocity(data):
    """Compute velocity from position."""
    # VALIDATE FIRST
    validate_dims_coords(data, {"time": [], "space": []})

    # Then process
    return compute_time_derivative(data, order=1)

def validate_dims_coords(data, required_dims):
    """
    Validate xarray has required dimensions.

    Raises
    ------
    ValueError
        If dimensions missing or coordinates invalid
    """
    for dim, coords in required_dims.items():
        if dim not in data.dims:
            raise ValueError(f"Data must have '{dim}' dimension")

        if coords and not all(c in data.coords for c in coords):
            raise ValueError(f"Missing required coordinates for '{dim}'")
```

**Pattern**: Fail fast with clear error messages.

### What Neurospatial Could Learn

**Current approach** (validation during processing):
```python
# neurospatial (current)
def bin_at(self, points):
    """Map points to bins."""
    # Validation mixed with logic
    if not self._is_fitted:
        raise RuntimeError("...")

    if points.ndim != 2:
        raise ValueError("...")

    # ... processing ...
```

**Potential improvement**:
```python
# neurospatial (potential)

def bin_at(self, points):
    """Map points to bins."""
    # Validate FIRST (separate function)
    _validate_points_for_binning(points, self.n_dim)
    self._check_fitted()  # Dedicated check

    # Then process (no more validation checks)
    return self._map_points_to_bins(points)

def _validate_points_for_binning(points, expected_dims):
    """Centralized validation."""
    if not isinstance(points, np.ndarray):
        raise TypeError(f"points must be ndarray, got {type(points)}")

    if points.ndim != 2:
        raise ValueError(f"points must be 2D, got {points.ndim}D")

    if points.shape[1] != expected_dims:
        raise ValueError(
            f"points must have {expected_dims} columns, "
            f"got {points.shape[1]}"
        )
```

**Benefits**:
- Validation logic reusable
- Clear separation: validate vs. process
- Easier to test validation independently
- Cleaner main logic

**Recommendation**: ✅ **Adopt in new modules**
- Use for segmentation module (Phase 2.5)
- Use for metrics module (Phase 4)
- Gradually refactor existing modules (Phase 5)

**Effort**: Minimal (pattern only)
**Risk**: None (improves code quality)

---

## 7. Warning Thresholds: Informative, Not Fatal ⭐⭐

### What They Do

**Warn about suspicious inputs without failing**:

```python
def _warn_about_nan_proportion(
    data,
    threshold=0.1,
    policy="ffill",
):
    """
    Warn if data contains too many NaN values.

    Doesn't raise - just warns user to check data quality.
    """
    nan_ratio = np.isnan(data).sum() / data.size

    if nan_ratio > threshold:
        logger.warning(
            f"Data contains {nan_ratio:.1%} NaN values "
            f"(threshold: {threshold:.1%}). "
            f"Using policy '{policy}' to handle missing data."
        )
```

**Pattern**: Warn about potential issues, let user decide.

### What Neurospatial Could Learn

**Current approach** (silent or fatal):
```python
# neurospatial (current)
def from_samples(cls, data, bin_size, **kwargs):
    """Either raises or proceeds silently."""
    # Check if enough data
    n_samples = len(data)
    if n_samples < 10:
        raise ValueError("Need at least 10 samples")

    # No warning for suspicious but valid inputs
```

**Potential improvement**:
```python
# neurospatial (potential)

def from_samples(cls, data, bin_size, **kwargs):
    """Warn about suspicious inputs."""
    n_samples = len(data)

    # Fatal error
    if n_samples < 2:
        raise ValueError("Need at least 2 samples")

    # Warning (not fatal)
    if n_samples < 100:
        logger.warning(
            f"Only {n_samples} samples provided. "
            f"Consider using more data for robust discretization."
        )

    # Warning for suspicious bin size
    extent = np.ptp(data, axis=0)
    bins_per_dim = extent / bin_size
    if np.any(bins_per_dim < 5):
        logger.warning(
            f"bin_size ({bin_size}) creates very coarse discretization "
            f"({bins_per_dim} bins per dimension). "
            f"Consider using smaller bin_size."
        )

    # Proceed anyway
    ...
```

**When to warn vs. raise**:
- **Raise**: Invalid inputs that will cause errors
- **Warn**: Suspicious inputs that might produce poor results

**Recommendation**: ✅ **Add warnings in validation**
- Warn for small sample sizes
- Warn for extreme bin sizes
- Warn for low occupancy coverage

**Effort**: 1-2 days (add to existing validation)
**Risk**: None (improves UX)

---

## 8. Type Hints with Runtime Validation ⭐⭐

### What They Do

**Type hints + runtime checks**:

```python
def compute_velocity(
    data: xr.DataArray,
) -> xr.DataArray:
    """
    Compute velocity.

    Type hints for documentation, runtime checks for validation.
    """
    # Runtime validation (not just type hints)
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"Expected xr.DataArray, got {type(data)}")

    validate_dims_coords(data, {"time": [], "space": []})
    ...
```

**Pattern**: Type hints for IDE/docs, runtime checks for safety.

### What Neurospatial Already Does

**Current approach** (good!):
```python
# neurospatial (current) ✅
def bin_at(
    self,
    points: NDArray[np.float64],
) -> NDArray[np.int_]:
    """Type hints + runtime checks."""
    if points.ndim != 2:
        raise ValueError(...)  # Runtime check
```

**Recommendation**: ✅ **Keep current approach**
- Type hints for documentation
- Runtime checks for validation
- Both are valuable

---

## 9. Property-Based Testing Hints ⭐

### What They Notice

**NaN handling policies**:
```python
def compute_path_length(data, start, stop, nan_policy="ffill"):
    """
    Compute path length with configurable NaN handling.

    Parameters
    ----------
    nan_policy : {'ffill', 'scale'}
        How to handle missing data:
        - 'ffill': Forward-fill NaN values
        - 'scale': Scale result by proportion of valid segments
    """
```

**Suggests testing strategy**: Properties should hold regardless of NaN handling.

### What Neurospatial Could Learn

**Test with missing data**:
```python
# Test idea for neurospatial

def test_occupancy_with_missing_data():
    """Occupancy should handle NaN positions gracefully."""
    positions = np.random.randn(100, 2)

    # Introduce missing data
    positions[20:25] = np.nan

    env = Environment.from_samples(positions, bin_size=2.0)

    # Should not crash
    occ = env.occupancy(times, positions)

    # Should have reduced total occupancy
    assert occ.sum() < len(positions) - 5  # Missing ~5 samples
```

**Recommendation**: ✅ **Add property-based tests**
- Test with missing data (NaN)
- Test with edge cases (single sample, duplicate positions)
- Test invariants (sum of occupancy = total time)

**Effort**: Ongoing (add to test suite)
**Risk**: None (improves robustness)

---

## 10. File Format Abstraction: Dispatching ⭐⭐

### What They Do

**Single entry point, multiple backends**:

```python
def from_file(
    file_path,
    source_software,  # "DeepLabCut", "SLEAP", etc.
    **kwargs,
):
    """
    Load pose data from any tracking software.

    Dispatches to software-specific loader based on source_software.
    """
    loaders = {
        "DeepLabCut": from_dlc_file,
        "SLEAP": from_sleap_file,
        "LightningPose": from_lp_file,
        "Anipose": from_anipose_file,
    }

    if source_software not in loaders:
        raise ValueError(f"Unknown source_software: {source_software}")

    # Dispatch to appropriate loader
    return loaders[source_software](file_path, **kwargs)
```

**Pattern**: Unified interface, multiple implementations.

### What Neurospatial Could Learn

**Not directly applicable** (neurospatial doesn't load tracking files).

**But could use pattern for**:
```python
# Potential: Load environment from different sources

def load_environment(source, **kwargs):
    """
    Load environment from various sources.

    Parameters
    ----------
    source : str or Path
        Source to load from:
        - Path to .json file
        - Path to .npz file
        - Path to pickle file
        - URL to remote environment
    """
    loaders = {
        ".json": Environment.from_file,
        ".npz": Environment.from_file,  # Handles both
        ".pkl": _load_from_pickle,
        "http": _load_from_url,
    }

    # Detect format
    if isinstance(source, str) and source.startswith("http"):
        fmt = "http"
    else:
        fmt = Path(source).suffix

    if fmt not in loaders:
        raise ValueError(f"Unsupported format: {fmt}")

    return loaders[fmt](source, **kwargs)
```

**Recommendation**: ⚠️ **Low priority**
- Current `from_file()` is fine
- Dispatcher pattern useful if adding more formats

---

## Summary: Top Recommendations for Neurospatial

### High Priority ✅ (Adopt Now)

1. **Validation-first philosophy** - Validate inputs before processing
   - Effort: Minimal (pattern only)
   - Benefit: Cleaner code, better errors

2. **Logging infrastructure** - Add structured logging
   - Effort: 2-3 days (Phase 5)
   - Benefit: Better debugging, production diagnostics

3. **Warning thresholds** - Warn about suspicious inputs
   - Effort: 1-2 days
   - Benefit: Better UX, fewer support issues

4. **Functional composition** - Break long methods into small functions
   - Effort: 1 week (Phase 5 refactor)
   - Benefit: More maintainable, testable

### Medium Priority ⚠️ (Consider)

5. **attrs for new modules** - Use attrs in segmentation, metrics
   - Effort: Minimal (new code only)
   - Benefit: Less boilerplate, automatic validation

6. **Optional xarray support** - Add `to_xarray()` / `return_xarray=` flag
   - Effort: 2-3 weeks (Phase 5.5)
   - Benefit: Better integration with movement, self-documenting outputs

### Low Priority ❌ (Don't Do)

7. **Rewrite existing code with attrs** - Breaking change, not worth it
8. **Adopt xarray as primary** - Too invasive, NumPy is fine

---

## Implementation Timeline

### Phase 5: Polish & Release (Week 16) - EXPANDED

**Week 16 (Days 1-3): Code Quality**
- Add logging infrastructure (`utils/logging.py`)
- Add validation warnings (small sample size, large bin_size)
- Refactor long methods (> 50 lines) into composable functions

**Week 16 (Days 4-5): Documentation**
- Document validation patterns
- Document logging usage
- Add troubleshooting guide (check logs first)

**Total effort**: +3 days (16 weeks → 16.5 weeks)

---

## Code Examples: Before/After

### Example 1: Logging

**Before**:
```python
if bin_size <= 0:
    raise ValueError(f"bin_size must be positive (got {bin_size})")
```

**After**:
```python
from neurospatial.utils.logging import logger

if bin_size <= 0:
    raise logger.error(
        ValueError(f"bin_size must be positive (got {bin_size})")
    )
```

### Example 2: Warnings

**Before**:
```python
# Silent - user doesn't know about potential issue
```

**After**:
```python
if n_samples < 100:
    logger.warning(
        f"Only {n_samples} samples - consider more data for robust discretization"
    )
```

### Example 3: Validation Helper

**Before**:
```python
def bin_at(self, points):
    if points.ndim != 2:
        raise ValueError("...")
    if points.shape[1] != self.n_dim:
        raise ValueError("...")
    # ... processing ...
```

**After**:
```python
def bin_at(self, points):
    _validate_points_for_binning(points, self.n_dim)
    # ... processing ...

def _validate_points_for_binning(points, expected_dims):
    """Centralized validation - reusable."""
    if points.ndim != 2:
        raise ValueError("...")
    if points.shape[1] != expected_dims:
        raise ValueError("...")
```

---

## Conclusion

movement provides excellent examples of:
1. ✅ **Structured validation** (attrs, validators)
2. ✅ **Logging architecture** (dual output, warnings)
3. ✅ **Functional composition** (small, chainable functions)
4. ✅ **Validation-first** (fail fast, clear errors)
5. ✅ **Warning thresholds** (inform, don't block)

**Most valuable for neurospatial**:
- Logging infrastructure (high impact, low effort)
- Validation-first philosophy (improves code quality)
- Warning thresholds (better UX)

**Adopt gradually**:
- Add logging in Phase 5
- Use attrs in new modules (segmentation, metrics)
- Refactor long methods during polish phase

**Total additional effort**: ~1 week (Phase 5 expansion)
