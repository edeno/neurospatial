# TASKS.md - Scale Bar Implementation

## Overview

Add scale bar functionality to neurospatial visualizations for publication-quality figures and accurate spatial interpretation.

**Target Version**: v0.11.0
**Status**: Not Started

---

## Milestone 1: Core Scale Bar Utilities (P0)

Foundation work for scale bar rendering. Must be completed before any integration.

### Task 1.1: Create visualization subpackage structure

**File**: `src/neurospatial/visualization/__init__.py` (NEW)

**Actions**:

1. Create `src/neurospatial/visualization/` directory
2. Create `__init__.py` with exports for `ScaleBarConfig`, `add_scale_bar_to_axes`, `compute_nice_length`, `format_scale_label`

**Success Criteria**:

- `from neurospatial.visualization import ScaleBarConfig` works
- No circular import errors

---

### Task 1.2: Implement ScaleBarConfig dataclass

**File**: `src/neurospatial/visualization/scale_bar.py` (NEW)

**Actions**:

1. Create frozen dataclass `ScaleBarConfig` with these fields:
   - `length: float | None = None` - Auto-calculated if None
   - `position: Literal["lower right", "lower left", "upper right", "upper left"] = "lower right"`
   - `color: str | None = None` - Auto-contrast if None
   - `background: str | None = "white"`
   - `background_alpha: float = 0.7`
   - `font_size: int = 10`
   - `pad: float = 0.5` - Multiple of font size for AnchoredSizeBar
   - `sep: float = 5.0` - Separation between bar and label (points)
   - `label_top: bool = True`
   - `box_style: Literal["round", "square", None] = "round"`
   - `show_label: bool = True`
2. Add NumPy-style docstring documenting matplotlib-only vs all-backend attributes

**Success Criteria**:

- `ScaleBarConfig()` creates default config
- `ScaleBarConfig(length=20, color="white")` creates custom config
- Frozen (immutable)

---

### Task 1.3: Implement compute_nice_length function

**File**: `src/neurospatial/visualization/scale_bar.py`

**Actions**:

1. Implement 1-2-5 rule algorithm:

   ```python
   def compute_nice_length(extent: float, target_fraction: float = 0.2) -> float:
       target = extent * target_fraction
       magnitude = 10 ** np.floor(np.log10(target))
       normalized = target / magnitude
       if normalized < 1.5:
           nice = 1
       elif normalized < 3.5:
           nice = 2
       elif normalized < 7.5:
           nice = 5
       else:
           nice = 10
       return nice * magnitude
   ```

2. Add NumPy-style docstring with examples

**Success Criteria**:

- `compute_nice_length(100) == 20`
- `compute_nice_length(50) == 10`
- `compute_nice_length(0.5) == 0.1`
- Result mantissa always 1, 2, or 5

---

### Task 1.4: Implement format_scale_label function

**File**: `src/neurospatial/visualization/scale_bar.py`

**Actions**:

1. Implement label formatter:

   ```python
   def format_scale_label(length: float, units: str | None) -> str:
       # Integer if whole number
       if length == int(length):
           length_str = str(int(length))
       else:
           length_str = str(length)
       if units:
           return f"{length_str} {units}"
       return length_str
   ```

2. Add NumPy-style docstring

**Success Criteria**:

- `format_scale_label(10, "cm") == "10 cm"`
- `format_scale_label(10.0, "cm") == "10 cm"` (not "10.0 cm")
- `format_scale_label(2.5, "cm") == "2.5 cm"`
- `format_scale_label(10, None) == "10"`

---

### Task 1.5: Implement add_scale_bar_to_axes function

**File**: `src/neurospatial/visualization/scale_bar.py`

**Dependencies**: Tasks 1.2, 1.3, 1.4

**Actions**:

1. Import `AnchoredSizeBar` from `mpl_toolkits.axes_grid1.anchored_artists`
2. Implement function:

   ```python
   def add_scale_bar_to_axes(
       ax: matplotlib.axes.Axes,
       extent: float,
       units: str | None = None,
       config: ScaleBarConfig | None = None,
   ) -> AnchoredOffsetbox:
   ```

3. Handle config defaults (use ScaleBarConfig() if None)
4. Compute length if not specified using `compute_nice_length(extent)`
5. Format label using `format_scale_label()`
6. Handle auto-contrast color (compute from axes background luminance)
7. Map position strings to AnchoredSizeBar loc parameter
8. Create and add AnchoredSizeBar to axes
9. Return the artist

**Success Criteria**:

- Adds scale bar to matplotlib axes
- Auto-sizes correctly when length=None
- Custom config overrides work
- Returns the artist for optional further manipulation

---

### Task 1.6: Implement configure_napari_scale_bar function

**File**: `src/neurospatial/visualization/scale_bar.py`

**Dependencies**: Task 1.2

**Actions**:

1. Implement napari configuration helper:

   ```python
   def configure_napari_scale_bar(
       viewer: napari.Viewer,
       units: str | None = None,
       config: ScaleBarConfig | None = None,
   ) -> None:
   ```

2. Map position names: "lower right" -> "bottom_right", etc.
3. Set `viewer.scale_bar.visible = True`
4. Apply config.position, config.font_size, config.color
5. Set units if provided
6. Document that `config.length` is ignored (napari auto-sizes)

**Success Criteria**:

- `viewer.scale_bar.visible` becomes True
- Position maps correctly
- Font size and color apply

---

## Milestone 2: Static Plot Integration (P1)

Integrate scale bars into matplotlib-based static visualizations.

### Task 2.1: Add scale_bar parameter to plot_field()

**File**: `src/neurospatial/environment/visualization.py`

**Dependencies**: Milestone 1 complete

**Actions**:

1. Add parameter: `scale_bar: bool | ScaleBarConfig = False`
2. Add import (inside function to avoid circular imports):

   ```python
   from neurospatial.visualization.scale_bar import (
       add_scale_bar_to_axes,
       ScaleBarConfig,
   )
   ```

3. At end of function, before return:
   - Check if `scale_bar` is truthy
   - Convert bool to ScaleBarConfig if needed
   - Validate `dimension_ranges` exists and is finite
   - Compute extent from `dimension_ranges[0]`
   - Warn and skip if extent invalid
   - Call `add_scale_bar_to_axes(ax, extent, self.units, config)`
4. Update docstring with scale_bar parameter documentation

**Success Criteria**:

- `env.plot_field(field, scale_bar=True)` adds scale bar
- `env.plot_field(field, scale_bar=ScaleBarConfig(...))` uses custom config
- `env.plot_field(field)` (default) has no scale bar
- Warnings emitted for invalid extents

---

### Task 2.2: Add scale_bar parameter to plot()

**File**: `src/neurospatial/environment/visualization.py`

**Dependencies**: Task 2.1

**Actions**:

1. Add parameter: `scale_bar: bool | ScaleBarConfig = False`
2. Apply same logic as plot_field() at end of function
3. Update docstring

**Success Criteria**:

- `env.plot(scale_bar=True)` adds scale bar
- Works with `show_regions=True`

---

## Milestone 3: Animation System Integration (P1)

Propagate scale bar through animation pipeline.

### Task 3.1: Add scale_bar to animate_fields() method

**File**: `src/neurospatial/environment/visualization.py`

**Dependencies**: Milestone 1 complete

**Actions**:

1. Add parameter: `scale_bar: bool | ScaleBarConfig = False`
2. Pass to `_animate()` call
3. Update docstring

**Success Criteria**:

- `env.animate_fields(fields, scale_bar=True)` accepted

---

### Task 3.2: Update animation core dispatcher

**File**: `src/neurospatial/animation/core.py`

**Dependencies**: Task 3.1

**Actions**:

1. Add `scale_bar: bool | ScaleBarConfig = False` to `animate_fields()` signature
2. Pass to backend-specific functions
3. Add import for ScaleBarConfig type hint

**Success Criteria**:

- Scale bar parameter flows to backend functions

---

### Task 3.3: Update rendering functions

**File**: `src/neurospatial/animation/rendering.py`

**Dependencies**: Task 3.2

**Actions**:

1. Add `scale_bar: bool | ScaleBarConfig = False` to:
   - `render_field_to_rgb()`
   - `render_field_to_image_bytes()`
2. Pass `scale_bar=scale_bar` to `env.plot_field()` calls
3. Add import for ScaleBarConfig type hint

**Success Criteria**:

- Video/HTML/Widget backends get scale bars automatically via plot_field()

---

### Task 3.4: Update parallel rendering

**File**: `src/neurospatial/animation/_parallel.py`

**Dependencies**: Task 3.3

**Actions**:

1. Add `scale_bar` to `_render_frames_worker()` params
2. Add `scale_bar` to `RenderParams` if using dataclass for params
3. Pass through to render functions

**Success Criteria**:

- Parallel video rendering includes scale bars

---

### Task 3.5: Update napari backend

**File**: `src/neurospatial/animation/backends/napari_backend.py`

**Dependencies**: Task 1.6

**Actions**:

1. Add `scale_bar` parameter to `render_napari()` and `_render_multi_field_napari()`
2. Import `configure_napari_scale_bar` from visualization module
3. After viewer setup, call:

   ```python
   if scale_bar:
       from neurospatial.visualization.scale_bar import configure_napari_scale_bar
       config = scale_bar if isinstance(scale_bar, ScaleBarConfig) else None
       configure_napari_scale_bar(viewer, units=env.units, config=config)
   ```

**Success Criteria**:

- `env.animate_fields(fields, scale_bar=True, backend="napari")` shows scale bar
- Napari's native scale bar is enabled and configured

---

## Milestone 4: Public API Export (P2)

### Task 4.1: Export ScaleBarConfig from main package

**File**: `src/neurospatial/__init__.py`

**Dependencies**: Milestone 1 complete

**Actions**:

1. Add import: `from neurospatial.visualization.scale_bar import ScaleBarConfig`
2. Add `"ScaleBarConfig"` to `__all__`

**Success Criteria**:

- `from neurospatial import ScaleBarConfig` works

---

## Milestone 5: Testing (P3)

### Task 5.1: Create test directory structure

**Actions**:

1. Create `tests/visualization/__init__.py`

---

### Task 5.2: Test compute_nice_length

**File**: `tests/visualization/test_scale_bar.py` (NEW)

**Actions**:

1. Create `TestComputeNiceLength` class
2. Test basic extents: 100 -> 20, 50 -> 10
3. Test 1-2-5 rule compliance using mantissa check with `np.isclose()`
4. Test target_fraction parameter
5. Test small extents (0.1 -> 0.02)
6. Test large extents (10000 -> 2000)

---

### Task 5.3: Test format_scale_label

**File**: `tests/visualization/test_scale_bar.py`

**Actions**:

1. Create `TestFormatScaleLabel` class
2. Test with units
3. Test without units
4. Test integer display (10.0 -> "10", not "10.0")
5. Test decimal display

---

### Task 5.4: Test add_scale_bar_to_axes

**File**: `tests/visualization/test_scale_bar.py`

**Actions**:

1. Create `TestAddScaleBarToAxes` class
2. Test artist is added to axes
3. Test custom config applies
4. Test no background option

---

### Task 5.5: Test plot_field with scale_bar

**File**: `tests/visualization/test_scale_bar.py`

**Actions**:

1. Create `TestPlotFieldWithScaleBar` class
2. Use fixtures: `small_2d_env`, `medium_2d_env`, `small_1d_env` from `tests/conftest.py`
3. Test `scale_bar=True`
4. Test `scale_bar=ScaleBarConfig(...)`
5. Test with colorbar
6. Test 1D environment
7. Test default (no scale bar)

---

### Task 5.6: Test plot with scale_bar

**File**: `tests/visualization/test_scale_bar.py`

**Actions**:

1. Create `TestPlotWithScaleBar` class
2. Test `env.plot(scale_bar=True)`

---

### Task 5.7: Test animation backends

**File**: `tests/visualization/test_scale_bar.py`

**Actions**:

1. Create `TestAnimateFieldsWithScaleBar` class
2. Test video backend with scale_bar (skip if ffmpeg unavailable)
3. Test napari scale bar configuration (skip if napari unavailable)

---

## Milestone 6: Documentation (P3)

### Task 6.1: Update CLAUDE.md Quick Reference

**File**: `CLAUDE.md`

**Actions**:

1. Add scale bar examples to Quick Reference section:

   ```python
   # Scale bars on visualizations (v0.11.0+)
   ax = env.plot_field(field, scale_bar=True)  # Auto-sized
   ax = env.plot(scale_bar=True)

   from neurospatial import ScaleBarConfig
   config = ScaleBarConfig(length=20, position="lower left", color="white")
   ax = env.plot_field(field, scale_bar=config)

   # Scale bars in animations
   env.animate_fields(fields, scale_bar=True, backend="napari")
   env.animate_fields(fields, scale_bar=True, save_path="video.mp4")
   ```

2. Add terminology note distinguishing from `calibrate_video(scale_bar=...)`

---

### Task 6.2: Update method docstrings

**Files**: `src/neurospatial/environment/visualization.py`

**Actions**:

1. Add `scale_bar` parameter to `plot_field()` docstring Examples section
2. Add `scale_bar` parameter to `plot()` docstring Examples section
3. Add `scale_bar` parameter to `animate_fields()` docstring Examples section

---

## Implementation Order

Execute milestones in this order:

1. **Milestone 1** (Tasks 1.1-1.6) - Core utilities must exist first
2. **Milestone 2** (Tasks 2.1-2.2) - Static plots depend on M1
3. **Milestone 3** (Tasks 3.1-3.5) - Animation depends on M1
4. **Milestone 4** (Task 4.1) - Export after implementation works
5. **Milestone 5** (Tasks 5.1-5.7) - Tests after implementation
6. **Milestone 6** (Tasks 6.1-6.2) - Documentation last

---

## Edge Cases to Handle

| Scenario | Expected Behavior |
|----------|-------------------|
| `dimension_ranges` is None | Warn, skip scale bar |
| `dimension_ranges` has infinite values | Warn, skip scale bar |
| Extent < 1e-6 | Warn, skip scale bar |
| `env.units` is None | Show numeric value only ("10" not "10 cm") |
| 3D+ environments | Not supported, skip (future work) |
| 1D environments | Use dimension_ranges[0], works normally |

---

## Files Summary

| File | Action | Milestone |
|------|--------|-----------|
| `src/neurospatial/visualization/__init__.py` | CREATE | M1 |
| `src/neurospatial/visualization/scale_bar.py` | CREATE | M1 |
| `src/neurospatial/environment/visualization.py` | MODIFY | M2, M3 |
| `src/neurospatial/animation/core.py` | MODIFY | M3 |
| `src/neurospatial/animation/rendering.py` | MODIFY | M3 |
| `src/neurospatial/animation/_parallel.py` | MODIFY | M3 |
| `src/neurospatial/animation/backends/napari_backend.py` | MODIFY | M3 |
| `src/neurospatial/__init__.py` | MODIFY | M4 |
| `tests/visualization/__init__.py` | CREATE | M5 |
| `tests/visualization/test_scale_bar.py` | CREATE | M5 |
| `CLAUDE.md` | MODIFY | M6 |
