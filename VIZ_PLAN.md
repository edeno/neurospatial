# Implementation Plan: `Environment.plot_field()`

## Overview
Add a new method `plot_field()` to `EnvironmentVisualization` mixin that visualizes spatial field data (firing rates, probabilities, decoded positions) over the environment using bin-appropriate rendering.

**Goal**: Replace manual scatter plotting with a single method that renders bins with their actual geometric shapes.

---

## Method Signature

```python
@check_fitted
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
    **kwargs: Any
) -> matplotlib.axes.Axes:
    """Plot spatial field data over the environment."""
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field` | `NDArray[np.float64]` | required | 1D array of values per bin, shape `(n_bins,)` |
| `ax` | `Axes \| None` | `None` | Matplotlib axes (creates new if None) |
| `cmap` | `str` | `"viridis"` | Colormap name |
| `vmin` | `float \| None` | `None` | Minimum value for colormap (auto: `nanmin`) |
| `vmax` | `float \| None` | `None` | Maximum value for colormap (auto: `nanmax`) |
| `colorbar` | `bool` | `True` | Whether to add colorbar |
| `colorbar_label` | `str` | `""` | Label for colorbar |
| `nan_color` | `str \| None` | `"lightgray"` | Color for NaN bins (`None` = skip rendering) |
| `rasterized` | `bool` | `True` | Rasterize output for large grids (performance) |
| `**kwargs` | `Any` | - | Pass to underlying plot function |

---

## Layout-Specific Rendering Strategy

### 1. Grid Layouts
**Layouts**: `RegularGrid`, `MaskedGrid`, `ImageMask`, `ShapelyPolygon`

**Detection**:
```python
layout_tag = self.layout._layout_type_tag
is_grid = layout_tag in ("RegularGrid", "MaskedGrid", "ImageMask", "ShapelyPolygon")
```

**Rendering Method**: `pcolormesh` (fast, crisp bin edges)

**Implementation**:
```python
from neurospatial.layout.helpers.utils import map_active_data_to_grid

# Convert 1D active bin data → 2D grid
grid_data = map_active_data_to_grid(
    self.layout.grid_shape,
    self.layout.active_mask,
    field,
    fill_value=np.nan
)

# Setup colormap with NaN handling
cmap_obj = plt.get_cmap(cmap).copy()
if nan_color is not None:
    cmap_obj.set_bad(color=nan_color)

# Plot
mesh = ax.pcolormesh(
    self.layout.grid_edges[0],
    self.layout.grid_edges[1],
    grid_data.T,
    cmap=cmap_obj,
    vmin=vmin,
    vmax=vmax,
    rasterized=rasterized,
    **kwargs
)
```

**Requirements**:
- Ensure `len(self.layout.grid_shape) == 2` (only 2D grids supported)
- Ensure `grid_edges`, `grid_shape`, `active_mask` exist

---

### 2. Hexagonal Layout
**Layouts**: `Hexagonal`

**Detection**:
```python
is_hex = self.layout._layout_type_tag == "Hexagonal"
```

**Rendering Method**: `PatchCollection` with colored hexagon patches

**Implementation**:
```python
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection

# Create hexagon patches
patches = []
colors = []

for i, (center, value) in enumerate(zip(self.bin_centers, field)):
    # Skip NaN bins if nan_color is None
    if np.isnan(value) and nan_color is None:
        continue

    patch = RegularPolygon(
        xy=center,
        numVertices=6,
        radius=self.layout.hex_radius_,
        orientation=self.layout.hex_orientation_
    )
    patches.append(patch)
    colors.append(value)

# Create collection
collection = PatchCollection(
    patches,
    cmap=cmap,
    rasterized=rasterized,
    **kwargs
)
collection.set_array(np.array(colors))
collection.set_clim(vmin, vmax)

# Handle NaN separately if needed
if nan_color is not None:
    # Color NaN hexagons with nan_color
    pass

ax.add_collection(collection)
```

**Requirements**:
- Ensure `self.layout.hex_radius_` exists
- Ensure `self.layout.hex_orientation_` exists

---

### 3. Triangular Mesh Layout
**Layouts**: `TriangularMesh`

**Detection**:
```python
is_trimesh = self.layout._layout_type_tag == "TriangularMesh"
```

**Rendering Method**: `tripcolor` (matplotlib's built-in triangular mesh coloring)

**Implementation**:
```python
# Get triangle vertex indices
triangles = self.layout._full_delaunay_tri.simplices[
    self.layout._active_original_simplex_indices
]

# Plot using tripcolor (colors triangle faces)
mesh = ax.tripcolor(
    self.bin_centers[:, 0],  # x coordinates of centroids
    self.bin_centers[:, 1],  # y coordinates of centroids
    triangles,               # triangle connectivity
    facecolors=field,        # color per triangle (not per vertex!)
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
    rasterized=rasterized,
    **kwargs
)
```

**Alternative (more NaN control)**: `PolyCollection` with triangle patches
```python
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

mesh_points = self.layout._full_delaunay_tri.points
active_triangles = self.layout._full_delaunay_tri.simplices[
    self.layout._active_original_simplex_indices
]

patches = []
colors = []

for i, triangle_vertex_indices in enumerate(active_triangles):
    value = field[i]

    if np.isnan(value) and nan_color is None:
        continue

    vertices = mesh_points[triangle_vertex_indices]
    patch = Polygon(vertices, closed=True)
    patches.append(patch)
    colors.append(value)

collection = PatchCollection(patches, cmap=cmap, rasterized=rasterized, **kwargs)
collection.set_array(np.array(colors))
collection.set_clim(vmin, vmax)

ax.add_collection(collection)
```

**Requirements**:
- Ensure `self.layout._full_delaunay_tri` exists
- Ensure `self.layout._active_original_simplex_indices` exists

**Decision**: Use `tripcolor` for simplicity unless NaN handling requires `PolyCollection`.

---

### 4. Graph Layout (1D Tracks)
**Layouts**: `Graph` with `is_1d == True`

**Detection**:
```python
is_1d_track = self.layout.is_1d
```

**Rendering Method**: 1D line plot with filled area

**Implementation**:
```python
# Filter NaN values
valid_mask = ~np.isnan(field)
valid_bins = np.where(valid_mask)[0]
valid_values = field[valid_mask]

# Use bin indices as x-axis (or linear positions if available)
x_positions = valid_bins

# Plot
line = ax.plot(x_positions, valid_values, linewidth=2, **kwargs)[0]
ax.fill_between(
    x_positions,
    0,
    valid_values,
    alpha=0.3,
    color=line.get_color()
)

ax.set_xlabel("Bin Index")
ax.set_ylabel(colorbar_label if colorbar_label else "Field Value")
ax.set_title("1D Field Plot")
```

**Note**: For 1D, colorbar doesn't apply (use y-axis instead).

---

### 5. Fallback (Unknown Layouts)
**Layouts**: Any layout not matching above categories

**Rendering Method**: Scatter plot with colored markers

**Implementation**:
```python
# Filter NaN values (if nan_color is None)
if nan_color is None:
    valid_mask = ~np.isnan(field)
    plot_centers = self.bin_centers[valid_mask]
    plot_values = field[valid_mask]
else:
    plot_centers = self.bin_centers
    plot_values = field.copy()

# Auto-compute marker size from bin spacing
if self.n_bins > 1:
    # Estimate typical bin spacing
    from scipy.spatial import cKDTree
    tree = cKDTree(self.bin_centers)
    # Find nearest neighbor distance for first 100 bins (sample for speed)
    sample_size = min(100, self.n_bins)
    distances, _ = tree.query(self.bin_centers[:sample_size], k=2)
    typical_spacing = np.median(distances[:, 1])
    marker_size = (typical_spacing * 72 / 2) ** 2  # Convert to points^2
else:
    marker_size = 100

# Plot
scatter = ax.scatter(
    plot_centers[:, 0],
    plot_centers[:, 1],
    c=plot_values,
    s=marker_size,
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
    edgecolors='none',
    rasterized=rasterized,
    **kwargs
)
```

**Requirements**:
- Ensure `self.n_dims == 2` for spatial scatter
- Warn if `n_dims > 2`

---

## Common Operations

### Input Validation
```python
# 1. Check field shape
if field.ndim != 1 or field.shape[0] != self.n_bins:
    raise ValueError(
        f"field must be 1D array with length n_bins={self.n_bins}, "
        f"got shape {field.shape}"
    )

# 2. Check dimensionality (for spatial plots)
if not self.layout.is_1d and self.n_dims > 2:
    raise NotImplementedError(
        f"Cannot plot {self.n_dims}D fields spatially. "
        "Only 1D and 2D environments are supported."
    )

# 3. Check for 2D grid requirements
if is_grid_layout and len(self.layout.grid_shape) != 2:
    raise NotImplementedError(
        f"pcolormesh requires 2D grids, got grid_shape={self.layout.grid_shape}"
    )
```

### Auto vmin/vmax
```python
if vmin is None:
    vmin = np.nanmin(field)
    if np.isnan(vmin) or np.isinf(vmin):  # All NaN or empty
        vmin = 0.0

if vmax is None:
    vmax = np.nanmax(field)
    if np.isnan(vmax) or np.isinf(vmax):  # All NaN or empty
        vmax = 1.0

# Ensure vmin < vmax
if vmin >= vmax:
    vmax = vmin + 1.0
```

### Colorbar
```python
if colorbar and mappable is not None:
    cbar = plt.colorbar(mappable, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
```

### Axes Setup
```python
if ax is None:
    _, ax = plt.subplots(figsize=(8, 7))

# Set aspect ratio for spatial plots (not 1D)
if not self.layout.is_1d:
    ax.set_aspect('equal')

    # Set labels
    unit_label = f" ({self.units})" if self.units else ""
    ax.set_xlabel(f"X Position{unit_label}", fontsize=12)
    ax.set_ylabel(f"Y Position{unit_label}", fontsize=12)

    # Set limits
    if self.dimension_ranges and len(self.dimension_ranges) >= 2:
        ax.set_xlim(self.dimension_ranges[0])
        ax.set_ylim(self.dimension_ranges[1])
```

---

## Implementation Flow

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
    **kwargs: Any
) -> matplotlib.axes.Axes:
    """Plot spatial field data over the environment."""

    # 1. Validate inputs
    if field.ndim != 1 or field.shape[0] != self.n_bins:
        raise ValueError(...)

    if not self.layout.is_1d and self.n_dims > 2:
        raise NotImplementedError(...)

    # 2. Create axes if needed
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    # 3. Auto vmin/vmax
    if vmin is None:
        vmin = np.nanmin(field)
        if np.isnan(vmin):
            vmin = 0.0

    if vmax is None:
        vmax = np.nanmax(field)
        if np.isnan(vmax):
            vmax = 1.0

    if vmin >= vmax:
        vmax = vmin + 1.0

    # 4. Dispatch to layout-specific renderer
    layout_tag = self.layout._layout_type_tag
    mappable = None

    GRID_LAYOUTS = ("RegularGrid", "MaskedGrid", "ImageMask", "ShapelyPolygon")

    if layout_tag in GRID_LAYOUTS:
        mappable = _plot_grid_field(self, field, ax, cmap, vmin, vmax,
                                     nan_color, rasterized, **kwargs)
    elif layout_tag == "Hexagonal":
        mappable = _plot_hex_field(self, field, ax, cmap, vmin, vmax,
                                    nan_color, rasterized, **kwargs)
    elif layout_tag == "TriangularMesh":
        mappable = _plot_trimesh_field(self, field, ax, cmap, vmin, vmax,
                                        nan_color, rasterized, **kwargs)
    elif self.layout.is_1d:
        mappable = _plot_1d_field(self, field, ax, colorbar_label, **kwargs)
    else:
        mappable = _plot_scatter_field(self, field, ax, cmap, vmin, vmax,
                                        nan_color, rasterized, **kwargs)

    # 5. Add colorbar (not for 1D)
    if colorbar and mappable is not None and not self.layout.is_1d:
        cbar = plt.colorbar(mappable, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    # 6. Format axes
    if not self.layout.is_1d:
        ax.set_aspect('equal')
        unit_label = f" ({self.units})" if self.units else ""
        ax.set_xlabel(f"X Position{unit_label}", fontsize=12)
        ax.set_ylabel(f"Y Position{unit_label}", fontsize=12)

        if self.dimension_ranges and len(self.dimension_ranges) >= 2:
            ax.set_xlim(self.dimension_ranges[0])
            ax.set_ylim(self.dimension_ranges[1])

    return ax
```

---

## Helper Functions

Each rendering path should be a separate helper function for clarity:

### `_plot_grid_field()`
```python
def _plot_grid_field(
    env: SelfEnv,
    field: NDArray[np.float64],
    ax: matplotlib.axes.Axes,
    cmap: str,
    vmin: float,
    vmax: float,
    nan_color: str | None,
    rasterized: bool,
    **kwargs: Any
) -> Any:
    """Plot field on grid layout using pcolormesh."""
    from neurospatial.layout.helpers.utils import map_active_data_to_grid

    # Validate 2D grid
    if len(env.layout.grid_shape) != 2:
        raise NotImplementedError(
            f"pcolormesh requires 2D grids, got grid_shape={env.layout.grid_shape}"
        )

    # Convert to grid
    grid_data = map_active_data_to_grid(
        env.layout.grid_shape,
        env.layout.active_mask,
        field,
        fill_value=np.nan
    )

    # Setup colormap with NaN handling
    cmap_obj = plt.get_cmap(cmap).copy()
    if nan_color is not None:
        cmap_obj.set_bad(color=nan_color, alpha=1.0)

    # Plot
    mesh = ax.pcolormesh(
        env.layout.grid_edges[0],
        env.layout.grid_edges[1],
        grid_data.T,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        rasterized=rasterized,
        shading='auto',
        **kwargs
    )

    return mesh
```

### `_plot_hex_field()`, `_plot_trimesh_field()`, `_plot_1d_field()`, `_plot_scatter_field()`
Similar structure for each layout type.

---

## Testing Strategy

### Unit Tests (`tests/test_environment_visualization.py`)

Create comprehensive tests for each layout type:

```python
class TestPlotField:
    """Test suite for Environment.plot_field() method."""

    def test_grid_layout_pcolormesh(self):
        """Test that grid layouts use pcolormesh."""
        positions = np.random.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions, bin_size=5.0)
        field = np.random.rand(env.n_bins)

        fig, ax = plt.subplots()
        result_ax = env.plot_field(field, ax=ax)

        # Check that pcolormesh was used
        assert any(isinstance(c, matplotlib.collections.QuadMesh)
                   for c in ax.collections)

        plt.close(fig)

    def test_hexagonal_layout_patches(self):
        """Test that hexagonal layouts use PatchCollection."""
        # Create hex environment
        env = Environment.from_samples(
            np.random.uniform(0, 100, (1000, 2)),
            bin_size=5.0,
            layout_type="hexagonal"
        )
        field = np.random.rand(env.n_bins)

        fig, ax = plt.subplots()
        result_ax = env.plot_field(field, ax=ax)

        # Check that PatchCollection was added
        assert len(ax.collections) > 0

        plt.close(fig)

    def test_nan_handling_skip(self):
        """Test that NaN bins are skipped when nan_color=None."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)),
            bin_size=10.0
        )
        field = np.random.rand(env.n_bins)
        field[::2] = np.nan  # Set half to NaN

        fig, ax = plt.subplots()
        env.plot_field(field, ax=ax, nan_color=None)

        # Should not raise, bins should be skipped
        plt.close(fig)

    def test_nan_handling_color(self):
        """Test that NaN bins are colored when nan_color is set."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)),
            bin_size=10.0
        )
        field = np.random.rand(env.n_bins)
        field[::2] = np.nan

        fig, ax = plt.subplots()
        env.plot_field(field, ax=ax, nan_color="gray")

        # Should render with gray color for NaN
        plt.close(fig)

    def test_auto_vmin_vmax(self):
        """Test automatic vmin/vmax from data."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)),
            bin_size=10.0
        )
        field = np.random.uniform(5.0, 15.0, env.n_bins)

        fig, ax = plt.subplots()
        env.plot_field(field, ax=ax)

        # Check colorbar limits match data range
        # (would need to inspect QuadMesh.get_clim())

        plt.close(fig)

    def test_custom_vmin_vmax(self):
        """Test explicit vmin/vmax."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)),
            bin_size=10.0
        )
        field = np.random.rand(env.n_bins)

        fig, ax = plt.subplots()
        env.plot_field(field, ax=ax, vmin=0.0, vmax=10.0)

        plt.close(fig)

    def test_colorbar_with_label(self):
        """Test colorbar creation with label."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)),
            bin_size=10.0
        )
        field = np.random.rand(env.n_bins)

        fig, ax = plt.subplots()
        env.plot_field(
            field,
            ax=ax,
            colorbar=True,
            colorbar_label="Firing Rate (Hz)"
        )

        # Check that figure has a colorbar axes
        assert len(fig.axes) == 2  # main ax + colorbar ax

        plt.close(fig)

    def test_invalid_field_shape(self):
        """Test that invalid field shape raises ValueError."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)),
            bin_size=10.0
        )

        # Wrong shape
        bad_field = np.random.rand(env.n_bins + 10)

        with pytest.raises(ValueError, match="field must be 1D array"):
            env.plot_field(bad_field)

    def test_3d_environment_not_supported(self):
        """Test that >2D environments raise NotImplementedError."""
        # Create 3D environment
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 3)),
            bin_size=10.0
        )
        field = np.random.rand(env.n_bins)

        with pytest.raises(NotImplementedError, match="Cannot plot.*3D"):
            env.plot_field(field)

    def test_1d_environment_line_plot(self):
        """Test that 1D environments use line plot."""
        env = Environment.from_samples(
            np.linspace(0, 100, 1000).reshape(-1, 1),
            bin_size=5.0
        )
        field = np.random.rand(env.n_bins)

        fig, ax = plt.subplots()
        env.plot_field(field, ax=ax)

        # Check that line was plotted
        assert len(ax.lines) > 0

        plt.close(fig)
```

### Integration Tests

Test with real neuroscience workflows:

```python
def test_place_field_visualization():
    """Integration test: visualize computed place field."""
    from neurospatial import compute_place_field

    # Generate synthetic data
    positions = np.random.uniform(20, 80, (1000, 2))
    times = np.linspace(0, 100, 1000)
    spike_times = np.random.uniform(0, 100, 50)

    # Create environment
    env = Environment.from_samples(positions, bin_size=5.0)

    # Compute place field
    firing_rate = compute_place_field(
        env, spike_times, times, positions, bandwidth=8.0
    )

    # Visualize
    fig, ax = plt.subplots(figsize=(8, 7))
    env.plot_field(
        firing_rate,
        ax=ax,
        cmap="hot",
        colorbar_label="Firing Rate (Hz)",
        vmin=0
    )

    assert ax is not None
    plt.close(fig)
```

---

## Documentation

### Docstring

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
    **kwargs: Any
) -> matplotlib.axes.Axes:
    """Plot spatial field data over the environment.

    Automatically selects the appropriate visualization method based on the
    layout type to render bins with their actual geometric shapes:

    - **Grid layouts**: Uses ``pcolormesh`` for crisp rectangular bins
    - **Hexagonal**: Colored hexagon patches via ``PatchCollection``
    - **Triangular mesh**: Colored triangle faces via ``tripcolor``
    - **1D tracks**: Line plot with filled area
    - **Other**: Scatter plot with auto-sized colored markers

    Parameters
    ----------
    field : NDArray[np.float64], shape (n_bins,)
        Spatial field values for each bin (e.g., firing rate, probability,
        decoded position density).
    ax : Optional[matplotlib.axes.Axes], optional
        Axes to plot on. If None, creates new figure and axes. Default: None.
    cmap : str, default="viridis"
        Matplotlib colormap name (e.g., "hot", "Blues", "viridis").
    vmin : Optional[float], optional
        Minimum value for colormap normalization. If None, uses
        ``np.nanmin(field)``. Default: None.
    vmax : Optional[float], optional
        Maximum value for colormap normalization. If None, uses
        ``np.nanmax(field)``. Default: None.
    colorbar : bool, default=True
        Whether to add a colorbar. Not applicable for 1D plots.
    colorbar_label : str, default=""
        Label text for the colorbar.
    nan_color : Optional[str], default="lightgray"
        Color for bins with NaN values. If None, NaN bins are not rendered
        (skipped). Use "lightgray", "white", etc.
    rasterized : bool, default=True
        Rasterize output for better performance and smaller file sizes with
        large grids. Recommended for environments with >1000 bins.
    **kwargs : Any
        Additional keyword arguments passed to underlying plot function
        (e.g., ``pcolormesh``, ``PatchCollection.set``).

    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the field was plotted.

    Raises
    ------
    ValueError
        If ``field`` shape does not match ``(n_bins,)``.
    NotImplementedError
        If environment dimensionality is >2D (not supported for spatial plots).
    NotImplementedError
        If attempting to use ``pcolormesh`` on non-2D grid.

    See Also
    --------
    compute_place_field : Compute firing rate maps from spike trains.
    plot : Plot environment structure (bins and connectivity).
    plot_1d : Plot 1D linearized environment.

    Notes
    -----
    **NaN Handling:**

    - ``nan_color=None``: NaN bins are skipped (not rendered)
    - ``nan_color="lightgray"``: NaN bins rendered with specified color

    **Performance:**

    For large environments (>10,000 bins), set ``rasterized=True`` (default)
    to improve rendering performance and reduce file size.

    **Colormaps:**

    Common choices for neuroscience:
    - ``"hot"``: Firing rate maps (black → red → yellow)
    - ``"viridis"``: General purpose, perceptually uniform
    - ``"Blues"``: Probability distributions
    - ``"RdBu_r"``: Diverging data (e.g., correlation)

    Examples
    --------
    **Plot a firing rate map:**

    >>> import numpy as np
    >>> from neurospatial import Environment, compute_place_field
    >>>
    >>> # Generate data
    >>> positions = np.random.uniform(0, 100, (1000, 2))
    >>> times = np.linspace(0, 100, 1000)
    >>> spike_times = np.random.uniform(0, 100, 50)
    >>>
    >>> # Create environment
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>>
    >>> # Compute firing rate
    >>> firing_rate = compute_place_field(
    ...     env, spike_times, times, positions, bandwidth=8.0
    ... )
    >>>
    >>> # Visualize
    >>> ax = env.plot_field(
    ...     firing_rate,
    ...     cmap="hot",
    ...     colorbar_label="Firing Rate (Hz)",
    ...     vmin=0
    ... )  # doctest: +SKIP

    **Plot decoded probability distribution:**

    >>> posterior = decoder.decode(neural_data)  # doctest: +SKIP
    >>> ax = env.plot_field(  # doctest: +SKIP
    ...     posterior,
    ...     cmap="Blues",
    ...     colorbar_label="Probability"
    ... )

    **Plot with custom value range:**

    >>> ax = env.plot_field(  # doctest: +SKIP
    ...     field,
    ...     vmin=-1.0,
    ...     vmax=1.0,
    ...     cmap="RdBu_r"
    ... )

    **Skip rendering NaN bins:**

    >>> field_with_nans = compute_place_field(...)  # doctest: +SKIP
    >>> ax = env.plot_field(field_with_nans, nan_color=None)  # doctest: +SKIP
    """
```

---

## Files to Modify

1. **`src/neurospatial/environment/visualization.py`**:
   - Add `plot_field()` method to `EnvironmentVisualization` class
   - Add helper functions: `_plot_grid_field()`, `_plot_hex_field()`,
     `_plot_trimesh_field()`, `_plot_1d_field()`, `_plot_scatter_field()`

2. **`tests/test_environment_visualization.py`** (create if doesn't exist):
   - Add `TestPlotField` class with comprehensive unit tests
   - Add integration tests with `compute_place_field()`

3. **Update class docstring** in `EnvironmentVisualization`:
   - Add `plot_field()` to Methods section

4. **Update notebook examples** (optional):
   - Replace manual scatter plotting in `11_place_field_analysis.ipynb`
   - Show `plot_field()` usage

---

## Implementation Checklist

- [ ] Add imports to `visualization.py`
- [ ] Implement `plot_field()` main method
- [ ] Implement `_plot_grid_field()` helper
- [ ] Implement `_plot_hex_field()` helper
- [ ] Implement `_plot_trimesh_field()` helper
- [ ] Implement `_plot_1d_field()` helper
- [ ] Implement `_plot_scatter_field()` helper
- [ ] Write unit tests for each layout type
- [ ] Write integration tests with `compute_place_field()`
- [ ] Test NaN handling (skip vs. color)
- [ ] Test auto vmin/vmax
- [ ] Test colorbar creation
- [ ] Test input validation
- [ ] Run mypy type checking
- [ ] Run ruff linting
- [ ] Update docstrings
- [ ] Test on real data
- [ ] Update example notebooks (optional)

---

## Future Enhancements (Out of Scope)

1. **3D Support**: Volume rendering or slicing for 3D environments
2. **Animation**: `plot_field_sequence()` for time series
3. **Overlay Mode**: `overlay=True` to overlay on existing layout plot
4. **Interactive**: Integration with ipywidgets for colormap/range control
5. **Vector Fields**: Arrow plots for gradient/velocity fields
6. **Contour Lines**: Add contour lines on top of field
7. **Multiple Fields**: `plot_fields([field1, field2])` with subplots

---

## Notes

- **Thread Safety**: Not required (matplotlib is not thread-safe)
- **Memory**: For very large grids (>100k bins), rasterization is critical
- **Compatibility**: Requires matplotlib >= 3.5 for best pcolormesh performance
- **Type Checking**: All code must pass `mypy --strict`
- **Testing**: Must achieve >90% coverage for visualization module
