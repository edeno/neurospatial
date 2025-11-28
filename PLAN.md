# Scale Bar Implementation Plan

## Overview

Add scale bar functionality to neurospatial visualizations, allowing users to display physical scale references on spatial plots and animations. Scale bars are essential for publication-quality figures and accurate spatial interpretation.

## Design Goals

1. **Automatic smart sizing** - Calculate aesthetically pleasing "nice" lengths (1, 2, 5, 10, 20, 50 units)
2. **Unit-aware** - Use `env.units` metadata when available (e.g., "10 cm")
3. **Configurable** - Allow customization of position, color, style, and length
4. **Backend-agnostic API** - Same parameter works across matplotlib, napari, video, HTML
5. **Non-intrusive** - Optional parameter that doesn't affect existing behavior
6. **Unified rendering path** - Scale bar added in `plot_field()`, propagates to all backends automatically

## Terminology Note

This feature adds **visualization scale bars** (showing physical scale on plots). This is distinct from the existing **calibration scale bar** feature (`calibrate_from_scale_bar()`, `calibrate_video(scale_bar=...)`) which is used for video-to-environment coordinate transforms. Documentation will clarify:

- `scale_bar=True` on `plot_field()` → adds visual scale indicator to plot
- `calibrate_video(scale_bar=...)` → uses known distance in video for calibration

## API Design

### Simple Usage (boolean toggle)

```python
# Enable scale bar with auto-sizing and positioning
env.plot_field(field, scale_bar=True)
env.plot(scale_bar=True)
env.animate_fields(fields, scale_bar=True)
```

### Advanced Usage (configuration object)

```python
from neurospatial.visualization import ScaleBarConfig

# Fully customized scale bar
config = ScaleBarConfig(
    length=20.0,            # Override auto-length (in environment units)
    position="lower right", # Position: "lower left", "lower right", "upper left", "upper right"
    color="white",          # Bar and text color (auto-contrast if None)
    background="black",     # Background box color (None for transparent)
    background_alpha=0.5,   # Background transparency
    font_size=12,           # Text font size
    bar_height=4,           # Bar thickness in points
    pad=0.5,                # Padding from axes edge (in inches)
    label_top=True,         # Label above bar (False = below)
    box_style="round",      # "round", "square", or None
)

env.plot_field(field, scale_bar=config)
```

### Expected Output

```
                    +----------------------------------+
                    |                                  |
                    |     [Spatial Field Plot]         |
                    |                                  |
                    |                                  |
                    |                        +-------+ |
                    |                        | 10 cm | |
                    |                        | ===== | |
                    |                        +-------+ |
                    +----------------------------------+
```

## Implementation Tasks

### Task 1: Create Scale Bar Utility Module

**File**: `src/neurospatial/visualization/scale_bar.py` (NEW)

```python
@dataclass(frozen=True)
class ScaleBarConfig:
    """Configuration for scale bar rendering.

    Attributes
    ----------
    length : float | None
        Scale bar length in environment units. If None, auto-calculated
        using the 1-2-5 rule to produce aesthetically pleasing values.
        **Matplotlib-only**: Napari auto-sizes based on layer scale.
    position : {"lower right", "lower left", "upper right", "upper left"}
        Scale bar position. Works on all backends.
    color : str | None
        Bar and text color. If None, auto-selects for contrast.
        Works on all backends.
    background : str | None
        Background box color. None for transparent. **Matplotlib-only**.
    background_alpha : float
        Background transparency (0-1). **Matplotlib-only**.
    font_size : int
        Text font size in points. Works on all backends.
    pad : float
        Padding from axes edge. AnchoredSizeBar uses this as a multiple
        of font size, not inches. **Matplotlib-only**.
    sep : float
        Separation between bar and label in points. **Matplotlib-only**.
    label_top : bool
        Label above bar (True) or below (False). **Matplotlib-only**.
    box_style : {"round", "square", None}
        Background box style. None for no box. **Matplotlib-only**.
    show_label : bool
        Whether to show "10 cm" text label. Works on all backends.
    """
    length: float | None = None
    position: Literal["lower right", "lower left", "upper right", "upper left"] = "lower right"
    color: str | None = None
    background: str | None = "white"
    background_alpha: float = 0.7
    font_size: int = 10
    pad: float = 0.5                    # Multiple of font size for AnchoredSizeBar
    sep: float = 5.0                    # Separation between bar and label (points)
    label_top: bool = True
    box_style: Literal["round", "square", None] = "round"
    show_label: bool = True


def compute_nice_length(extent: float, target_fraction: float = 0.2) -> float:
    """Compute aesthetically pleasing scale bar length.

    Uses 1-2-5 rule: lengths are always 1, 2, or 5 x 10^n.
    Target is approximately `target_fraction` of the total extent.

    Examples:
        extent=100 -> 20 (20% of extent)
        extent=73 -> 10 or 20
        extent=0.5 -> 0.1
    """
    ...


def add_scale_bar_to_axes(
    ax: matplotlib.axes.Axes,
    extent_x: float,
    extent_y: float,
    units: str | None = None,
    config: ScaleBarConfig | None = None,
) -> AnchoredOffsetbox:
    """Add a scale bar to matplotlib axes.

    Uses AnchoredOffsetbox for precise positioning that survives
    figure resizing and zoom operations.
    """
    ...


def format_scale_label(length: float, units: str | None) -> str:
    """Format scale bar label (e.g., '10 cm', '0.5 m', '100').

    - Integer if whole number: "10 cm" not "10.0 cm"
    - Decimal if needed: "2.5 cm"
    - No units if None: "10"
    """
    ...
```

### Task 2: Create Napari Scale Bar Helper

**File**: `src/neurospatial/visualization/scale_bar.py` (addition)

```python
def configure_napari_scale_bar(
    viewer: napari.Viewer,
    units: str | None = None,
    config: ScaleBarConfig | None = None,
) -> None:
    """Configure napari's native scale bar.

    Napari has built-in scale bar support with automatic sizing based on
    the current zoom level and layer scale. We configure it here.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    units : str | None
        Unit label (e.g., "cm", "um"). If None, no unit shown.
    config : ScaleBarConfig | None
        Configuration object. Note: `config.length` is ignored because
        napari auto-sizes based on layer scale and zoom level.

    Notes
    -----
    **Napari scale bar auto-sizing**: Unlike matplotlib where we set an
    explicit length, napari calculates the scale bar length dynamically
    based on the current view. This is actually beneficial for interactive
    use but means `ScaleBarConfig.length` has no effect in napari.

    **Layer scale**: For napari's scale bar to show correct units, the
    image layer must have the correct `scale` attribute set. This is
    typically set via `EnvScale` in `animation/transforms.py`. If scale
    is (1, 1), the scale bar shows pixels regardless of `unit` setting.

    Supported config attributes:
    - position: Maps to napari's position names
    - color: Scale bar color
    - font_size: Text size

    Ignored config attributes (matplotlib-only):
    - length, background, background_alpha, pad, sep, label_top, box_style
    """
    if config is None:
        config = ScaleBarConfig()

    # Position mapping: our API -> napari names
    position_map = {
        "lower right": "bottom_right",
        "lower left": "bottom_left",
        "upper right": "top_right",
        "upper left": "top_left",
    }

    viewer.scale_bar.visible = True
    viewer.scale_bar.position = position_map.get(config.position, "bottom_right")
    viewer.scale_bar.font_size = config.font_size

    if units:
        viewer.scale_bar.unit = units

    if config.color:
        viewer.scale_bar.color = config.color
```

**Important**: This is the ONLY napari scale bar function. The napari backend
(`animation/backends/napari_backend.py`) should call this function directly
rather than duplicating the logic.

### Task 3: Integrate into Static Visualization Methods

**File**: `src/neurospatial/environment/visualization.py`

#### 3.1 Update `plot_field()` signature

```python
def plot_field(
    self: SelfEnv,
    field: NDArray[np.float64],
    ax: matplotlib.axes.Axes | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    colorbar_label: str = "",
    nan_color: str | None = "lightgray",
    rasterized: bool = True,
    scale_bar: bool | ScaleBarConfig = False,  # NEW
    **kwargs: Any,
) -> matplotlib.axes.Axes:
```

#### 3.2 Add scale bar rendering logic

```python
# At end of plot_field(), before return:
if scale_bar:
    from neurospatial.visualization.scale_bar import (
        add_scale_bar_to_axes,
        ScaleBarConfig,
    )

    config = scale_bar if isinstance(scale_bar, ScaleBarConfig) else ScaleBarConfig()

    if self.dimension_ranges and len(self.dimension_ranges) >= 2:
        extent_x = self.dimension_ranges[0][1] - self.dimension_ranges[0][0]
        extent_y = self.dimension_ranges[1][1] - self.dimension_ranges[1][0]
        add_scale_bar_to_axes(ax, extent_x, extent_y, self.units, config)
```

#### 3.3 Update `plot()` signature (same pattern)

```python
def plot(
    self: SelfEnv,
    ax: matplotlib.axes.Axes | None = None,
    show_regions: bool = False,
    layout_plot_kwargs: dict[str, Any] | None = None,
    regions_plot_kwargs: dict[str, Any] | None = None,
    scale_bar: bool | ScaleBarConfig = False,  # NEW
    **kwargs: Any,
) -> matplotlib.axes.Axes:
```

### Task 4: Integrate into Animation System

**File**: `src/neurospatial/environment/visualization.py`

#### 4.1 Update `animate_fields()` signature

```python
def animate_fields(
    self: SelfEnv,
    fields: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    *,
    # ... existing params ...
    scale_bar: bool | ScaleBarConfig = False,  # NEW
    **kwargs: Any,
) -> Any:
```

#### 4.2 Pass scale_bar to animation dispatcher

```python
return _animate(
    env=self,
    fields=fields,
    # ... existing params ...
    scale_bar=scale_bar,  # NEW
    **kwargs,
)
```

### Task 5: Update Animation Core Dispatcher

**File**: `src/neurospatial/animation/core.py`

#### 5.1 Update `animate_fields()` function signature

```python
def animate_fields(
    env: Environment,
    fields: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    *,
    # ... existing params ...
    scale_bar: bool | ScaleBarConfig = False,  # NEW
    **kwargs: Any,
) -> Any:
```

#### 5.2 Pass to backend-specific functions

Each backend handles scale_bar differently:
- **napari**: Use native `viewer.scale_bar` API
- **video/html/widget**: Pass to matplotlib frame renderer

### Task 6: Implement Backend-Specific Rendering

**Key insight from architecture review**: The video/html/widget backends all render
frames via `env.plot_field()` (through `render_field_to_rgb()`, `render_field_to_image_bytes()`,
or `_render_frames_worker()`). By adding `scale_bar` to `plot_field()` in Task 3, these
backends get scale bar support **automatically** without additional code.

#### 6.1 Napari Backend

**File**: `src/neurospatial/animation/backends/napari_backend.py`

Call the centralized helper function from Task 2:

```python
# In render_napari() or _render_multi_field_napari(), after viewer setup:
from neurospatial.visualization.scale_bar import configure_napari_scale_bar

if scale_bar:
    config = scale_bar if isinstance(scale_bar, ScaleBarConfig) else None
    configure_napari_scale_bar(viewer, units=env.units, config=config)
```

**No duplication**: The napari backend calls `configure_napari_scale_bar()` from
the visualization module rather than implementing its own logic.

#### 6.2 Video/HTML/Widget Backends - Automatic via plot_field()

**Files**: `animation/rendering.py`, `animation/_parallel.py`

These backends use `env.plot_field()` for frame rendering:
- `render_field_to_rgb()` calls `env.plot_field()`
- `render_field_to_image_bytes()` calls `env.plot_field()`
- `_render_frames_worker()` uses the above functions

**Minimal changes needed**: Just pass `scale_bar` parameter through the call chain:

```python
# In rendering.py:render_field_to_rgb()
def render_field_to_rgb(
    env: Environment,
    field: NDArray[np.float64],
    cmap: str,
    vmin: float,
    vmax: float,
    dpi: int = 100,
    scale_bar: bool | ScaleBarConfig = False,  # NEW
) -> NDArray[np.uint8]:
    ...
    env.plot_field(
        field,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=False,
        scale_bar=scale_bar,  # NEW - propagates to plot_field()
    )
```

Similar changes to `render_field_to_image_bytes()` and the parallel worker.

**Benefits of this approach**:
1. Single point of scale bar logic (in `plot_field()`)
2. No duplication across backends
3. Automatic consistency between static plots and animations
4. Easier maintenance

### Task 7: Update `plot_regions()` Function (Optional)

**File**: `src/neurospatial/regions/plot.py`

```python
def plot_regions(
    regions: Regions,
    ax: matplotlib.axes.Axes | None = None,
    extent: tuple[tuple[float, float], tuple[float, float]] | None = None,  # NEW
    units: str | None = None,  # NEW
    scale_bar: bool | ScaleBarConfig = False,  # NEW
    **kwargs: Any,
) -> matplotlib.axes.Axes:
```

Note: `plot_regions()` needs extent context for scale bars. Options:
- Pass `extent` and `units` parameters explicitly
- Compute from region bounds (less accurate)
- Skip scale bar for standalone region plots

**Recommendation**: Add `extent` and `units` params; infer from region bounds as fallback.

### Task 8: Export in Public API

**File**: `src/neurospatial/__init__.py`

```python
from neurospatial.visualization.scale_bar import ScaleBarConfig

__all__ = [
    # ... existing exports ...
    "ScaleBarConfig",
]
```

**File**: `src/neurospatial/visualization/__init__.py` (NEW)

```python
"""Visualization utilities for neurospatial."""

from neurospatial.visualization.scale_bar import (
    ScaleBarConfig,
    add_scale_bar_to_axes,
    compute_nice_length,
    format_scale_label,
)

__all__ = [
    "ScaleBarConfig",
    "add_scale_bar_to_axes",
    "compute_nice_length",
    "format_scale_label",
]
```

### Task 9: Add Tests

**File**: `tests/visualization/test_scale_bar.py` (NEW)

**Important**: Use existing fixtures from `tests/conftest.py` (not `tests/nwb/conftest.py`):

- `small_2d_env` - Small 10x10 cm grid (25 bins)
- `medium_2d_env` - Medium 50x50 cm grid (625 bins)
- `small_1d_env` - Small 1D linear track (5 bins)

```python
"""Tests for scale bar visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurospatial.visualization.scale_bar import (
    ScaleBarConfig,
    add_scale_bar_to_axes,
    compute_nice_length,
    format_scale_label,
)


class TestComputeNiceLength:
    """Test nice length computation."""

    def test_basic_extents(self):
        assert compute_nice_length(100) == 20  # 20% of 100
        assert compute_nice_length(50) == 10   # 20% of 50

    def test_follows_125_rule(self):
        """Results should be 1, 2, or 5 x 10^n.

        Uses tolerance for floating-point robustness.
        """
        for extent in [7, 13, 27, 73, 156, 0.3, 0.7]:
            length = compute_nice_length(extent)
            # Compute mantissa (normalized to [1, 10))
            mantissa = length / (10 ** np.floor(np.log10(length)))
            # Check with tolerance for floating-point robustness
            valid_mantissas = (1.0, 2.0, 5.0)
            assert any(
                np.isclose(mantissa, v, atol=1e-9) for v in valid_mantissas
            ), f"extent={extent} produced length={length}, mantissa={mantissa}"

    def test_target_fraction(self):
        # Different target fractions
        assert compute_nice_length(100, target_fraction=0.1) == 10
        assert compute_nice_length(100, target_fraction=0.25) == 20

    def test_small_extents(self):
        """Test with small extents (common in normalized data)."""
        assert compute_nice_length(1.0) == 0.2
        assert compute_nice_length(0.1) == 0.02

    def test_large_extents(self):
        """Test with large extents (e.g., meters)."""
        assert compute_nice_length(10000) == 2000


class TestFormatScaleLabel:
    """Test scale label formatting."""

    def test_with_units(self):
        assert format_scale_label(10, "cm") == "10 cm"
        assert format_scale_label(2.5, "m") == "2.5 m"

    def test_without_units(self):
        assert format_scale_label(10, None) == "10"

    def test_integer_display(self):
        # Whole numbers should not show decimal
        assert format_scale_label(10.0, "cm") == "10 cm"  # Not "10.0 cm"
        assert format_scale_label(5.0, "m") == "5 m"

    def test_decimal_display(self):
        # Non-whole numbers should show decimal
        assert format_scale_label(2.5, "cm") == "2.5 cm"
        assert format_scale_label(0.1, "m") == "0.1 m"


class TestAddScaleBarToAxes:
    """Test matplotlib scale bar rendering."""

    def test_adds_artist(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        artist = add_scale_bar_to_axes(ax, extent=100, units="cm")
        # AnchoredSizeBar is added to ax.artists
        assert len(ax.artists) > 0 or artist in ax.get_children()
        plt.close(fig)

    def test_custom_config(self):
        config = ScaleBarConfig(length=25, position="upper left", color="red")
        fig, ax = plt.subplots()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        add_scale_bar_to_axes(ax, extent=100, units="cm", config=config)
        assert len(ax.artists) > 0
        plt.close(fig)

    def test_no_background(self):
        config = ScaleBarConfig(box_style=None)
        fig, ax = plt.subplots()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        add_scale_bar_to_axes(ax, extent=100, units="cm", config=config)
        plt.close(fig)


class TestPlotFieldWithScaleBar:
    """Test scale bar integration in plot_field.

    Uses fixtures from tests/conftest.py:
    - small_2d_env: 10x10 cm grid
    - medium_2d_env: 50x50 cm grid
    - small_1d_env: 10 cm linear track
    """

    def test_scale_bar_bool(self, small_2d_env):
        """Test scale_bar=True adds scale bar."""
        field = np.random.rand(small_2d_env.n_bins)
        ax = small_2d_env.plot_field(field, scale_bar=True)
        # Check that at least one artist was added (scale bar)
        assert len(ax.artists) > 0
        plt.close()

    def test_scale_bar_config(self, small_2d_env):
        """Test scale_bar=ScaleBarConfig works."""
        config = ScaleBarConfig(length=5.0, color="white", position="upper left")
        field = np.random.rand(small_2d_env.n_bins)
        ax = small_2d_env.plot_field(field, scale_bar=config)
        assert len(ax.artists) > 0
        plt.close()

    def test_scale_bar_with_colorbar(self, small_2d_env):
        """Test scale bar works alongside colorbar."""
        field = np.random.rand(small_2d_env.n_bins)
        ax = small_2d_env.plot_field(field, scale_bar=True, colorbar=True)
        assert len(ax.artists) > 0
        plt.close()

    def test_scale_bar_1d_env(self, small_1d_env):
        """Test scale bar works with 1D environments."""
        field = np.random.rand(small_1d_env.n_bins)
        ax = small_1d_env.plot_field(field, scale_bar=True)
        # 1D should still get a scale bar
        # (may be in different location since 1D plots are line plots)
        plt.close()

    def test_no_scale_bar_default(self, small_2d_env):
        """Test scale_bar=False (default) adds no scale bar."""
        field = np.random.rand(small_2d_env.n_bins)
        ax = small_2d_env.plot_field(field, scale_bar=False)
        # No additional artists (beyond the field itself)
        assert len(ax.artists) == 0
        plt.close()


class TestPlotWithScaleBar:
    """Test scale bar in env.plot()."""

    def test_plot_with_scale_bar(self, small_2d_env):
        """Test env.plot(scale_bar=True)."""
        ax = small_2d_env.plot(scale_bar=True)
        assert len(ax.artists) > 0
        plt.close()


class TestAnimateFieldsWithScaleBar:
    """Test scale bar in animation backends."""

    def test_video_scale_bar(self, small_2d_env, tmp_path):
        """Test video backend includes scale bar in frames."""
        pytest.importorskip("ffmpeg")  # Skip if ffmpeg not available
        fields = [np.random.rand(small_2d_env.n_bins) for _ in range(3)]
        output = tmp_path / "test_scalebar.mp4"
        small_2d_env.animate_fields(
            fields,
            save_path=str(output),
            scale_bar=True,
            fps=1,
        )
        assert output.exists()
        assert output.stat().st_size > 0

    def test_napari_scale_bar_config(self):
        """Test configure_napari_scale_bar function."""
        napari = pytest.importorskip("napari")
        from neurospatial.visualization.scale_bar import configure_napari_scale_bar

        # Create a minimal viewer for testing
        viewer = napari.Viewer(show=False)
        try:
            config = ScaleBarConfig(position="upper left", font_size=14)
            configure_napari_scale_bar(viewer, units="cm", config=config)

            assert viewer.scale_bar.visible is True
            assert viewer.scale_bar.unit == "cm"
            assert viewer.scale_bar.position == "top_left"  # Mapped position
            assert viewer.scale_bar.font_size == 14
        finally:
            viewer.close()
```

### Task 10: Update Documentation

#### 10.1 Update CLAUDE.md Quick Reference

Add to Quick Reference section:

```python
# Scale bars on visualizations (v0.11.0+)
ax = env.plot_field(field, scale_bar=True)  # Auto-sized scale bar
ax = env.plot(scale_bar=True)

# Custom scale bar
from neurospatial import ScaleBarConfig
config = ScaleBarConfig(length=20, position="lower left", color="white")
ax = env.plot_field(field, scale_bar=config)

# Scale bars in animations
env.animate_fields(fields, scale_bar=True, backend="napari")
env.animate_fields(fields, scale_bar=True, save_path="video.mp4")
```

#### 10.2 Add docstring examples

Update `plot_field()` docstring with scale bar examples.

## Implementation Order

1. **Core utility** (`scale_bar.py`) - Foundation with `ScaleBarConfig`, `compute_nice_length()`, `add_scale_bar_to_axes()`
2. **Static plots** - `plot_field()`, `plot()` integration
3. **Animation system** - `animate_fields()` and backend implementations
4. **Tests** - Comprehensive test coverage
5. **Documentation** - CLAUDE.md and docstrings

## Technical Considerations

### Matplotlib Implementation Details

Use `AnchoredSizeBar` from `mpl_toolkits.axes_grid1` (included with matplotlib):

```python
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

# Simple usage
fontprops = fm.FontProperties(size=config.font_size)
scalebar = AnchoredSizeBar(
    ax.transData,              # Transform (data coordinates)
    size=length,               # Length in data units
    label=f"{length} {units}", # Label text
    loc="lower right",         # Location
    pad=config.pad,
    color=config.color,
    frameon=config.box_style is not None,
    size_vertical=config.bar_height,
    fontproperties=fontprops,
)
ax.add_artist(scalebar)
```

Location mapping for `AnchoredSizeBar`:
- "lower right" -> `loc='lower right'` or `loc=4`
- "lower left" -> `loc='lower left'` or `loc=3`
- "upper right" -> `loc='upper right'` or `loc=1`
- "upper left" -> `loc='upper left'` or `loc=2`

### Napari Native Scale Bar

Napari has built-in scale bar support:

```python
viewer.scale_bar.visible = True
viewer.scale_bar.unit = "cm"
viewer.scale_bar.position = "bottom_right"  # napari naming convention
viewer.scale_bar.font_size = 12
viewer.scale_bar.colored = True             # Use layer colormap
viewer.scale_bar.ticks = True               # Show tick marks
```

Key difference: Napari uses its own position names (`bottom_right` vs our `lower right`).

**Position mapping**:
| Our API | Napari |
|---------|--------|
| `"lower right"` | `"bottom_right"` |
| `"lower left"` | `"bottom_left"` |
| `"upper right"` | `"top_right"` |
| `"upper left"` | `"top_left"` |

### Extent Handling and Edge Cases

**Extent selection rule**: Scale bar length is based on the X-axis extent by default:
- `extent = dimension_ranges[0][1] - dimension_ranges[0][0]`
- If X and Y extents differ significantly, still use X (scale bar is horizontal)
- Future: Add `ScaleBarConfig.use_y_extent=True` option if needed

**1D Environment Handling**:

For 1D linearized tracks (`layout.is_1d == True`):
- Use `dimension_ranges[0]` (the only dimension)
- Scale bar is meaningful (represents linear distance along track)
- Place in corner of the 1D plot (default: lower right)
- Works automatically since `plot_field()` already handles 1D

**Edge cases**:

| Scenario | Behavior |
|----------|----------|
| `dimension_ranges` is None | Skip scale bar with warning |
| `dimension_ranges` has infinite values | Skip scale bar with warning |
| `dimension_ranges[0]` is very small (<1e-6) | Skip scale bar with warning |
| `env.units` is None | Show numeric value only (e.g., "10" not "10 cm") |
| 3D+ environments | Not supported - skip with warning (future work) |

**Implementation in plot_field()**:

```python
# At end of plot_field(), before return:
if scale_bar:
    from neurospatial.visualization.scale_bar import (
        add_scale_bar_to_axes,
        ScaleBarConfig,
    )

    config = scale_bar if isinstance(scale_bar, ScaleBarConfig) else ScaleBarConfig()

    # Validate extent is available and reasonable
    if not self.dimension_ranges:
        warnings.warn(
            "Cannot add scale bar: dimension_ranges not available",
            UserWarning,
        )
    elif not np.isfinite(self.dimension_ranges[0]).all():
        warnings.warn(
            "Cannot add scale bar: dimension_ranges contains non-finite values",
            UserWarning,
        )
    else:
        extent_x = self.dimension_ranges[0][1] - self.dimension_ranges[0][0]
        if extent_x < 1e-6:
            warnings.warn(
                f"Cannot add scale bar: extent too small ({extent_x})",
                UserWarning,
            )
        else:
            add_scale_bar_to_axes(ax, extent_x, self.units, config)
```

### Color Auto-Contrast

When `color=None`, compute contrasting color based on background:

```python
def _auto_contrast_color(ax: Axes) -> str:
    """Determine scale bar color based on axes background."""
    bg_color = ax.get_facecolor()
    # Convert to grayscale luminance
    luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
    return "white" if luminance < 0.5 else "black"
```

### Nice Length Algorithm

The 1-2-5 rule produces aesthetically pleasing values:

```python
def compute_nice_length(extent: float, target_fraction: float = 0.2) -> float:
    """Compute nice scale bar length using 1-2-5 rule."""
    target = extent * target_fraction

    # Find order of magnitude
    magnitude = 10 ** np.floor(np.log10(target))

    # Normalized value
    normalized = target / magnitude

    # Round to nearest 1, 2, or 5
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

Examples:
- `extent=100, target=20` -> 20
- `extent=73, target=14.6` -> 10 or 20
- `extent=0.5, target=0.1` -> 0.1

## Files to Create/Modify

| File | Action | Priority | Description |
|------|--------|----------|-------------|
| `src/neurospatial/visualization/__init__.py` | CREATE | P0 | New visualization subpackage |
| `src/neurospatial/visualization/scale_bar.py` | CREATE | P0 | Core scale bar utilities + napari helper |
| `src/neurospatial/environment/visualization.py` | MODIFY | P1 | Add scale_bar param to `plot()`, `plot_field()`, `animate_fields()` |
| `src/neurospatial/animation/core.py` | MODIFY | P1 | Pass scale_bar through to backends |
| `src/neurospatial/animation/rendering.py` | MODIFY | P1 | Pass scale_bar to `plot_field()` calls |
| `src/neurospatial/animation/_parallel.py` | MODIFY | P1 | Pass scale_bar through parallel renderer |
| `src/neurospatial/animation/backends/napari_backend.py` | MODIFY | P2 | Call `configure_napari_scale_bar()` |
| `src/neurospatial/__init__.py` | MODIFY | P2 | Export ScaleBarConfig |
| `tests/visualization/__init__.py` | CREATE | P3 | Test package init |
| `tests/visualization/test_scale_bar.py` | CREATE | P3 | Scale bar tests using existing fixtures |
| `CLAUDE.md` | MODIFY | P3 | Add scale bar documentation |

**Note**: Video/HTML/widget backends don't need direct modification - they automatically
get scale bars through `rendering.py` -> `env.plot_field()` call chain.

## Open Questions

1. **Should `plot_regions()` support scale bars?**
   - Pro: Consistent API
   - Con: Standalone region plots may not have extent context
   - **Decision**: Defer to v2. Most users call `env.plot(show_regions=True)` anyway.

2. **What about 3D environments (future)?**
   - Scale bars in 3D are more complex (cube? orthogonal bars?)
   - **Decision**: Skip for now, add in future version. Document as unsupported.

3. **Should we support multiple scale bars (X and Y)?**
   - Some users may want separate X/Y scale bars if units differ
   - **Decision**: Not in v1, can add later via `ScaleBarConfig.axes="xy"`

4. **Integration with `mpl_toolkits.axes_grid1.AnchoredSizeBar`?**
   - This exists in matplotlib's toolkit
   - **Decision**: Use it directly - it's part of matplotlib, handles all complexity

5. **Napari scale bar length control?**
   - Napari auto-sizes based on zoom level and layer scale
   - **Decision**: Document that `ScaleBarConfig.length` is matplotlib-only.
     Napari's auto-sizing is actually better for interactive exploration.

## Resolved Questions (from review)

1. **Unified rendering path**: Yes, use `plot_field()` as the single integration point
2. **Napari helper duplication**: Single `configure_napari_scale_bar()` in visualization module
3. **AnchoredSizeBar units**: `pad` is multiple of font size, not inches (documented in ScaleBarConfig)
4. **Test fixtures**: Use `small_2d_env`, `medium_2d_env`, `small_1d_env` from `tests/conftest.py`
5. **Floating-point tests**: Use `np.isclose()` with tolerance for 1-2-5 rule validation
6. **scale_bar terminology**: Document distinction from `calibrate_from_scale_bar()` in CLAUDE.md

## Version Target

This feature would be part of **v0.11.0** based on current version (v0.9.0) and planned features.
