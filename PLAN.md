# Plan: Position-Based Boundary Seeding for Annotation Module

## Overview

Add the ability to seed the environment boundary polygon from position data (e.g., convex hull, alpha shape, KDE contour) so users can adjust pre-drawn boundaries in napari rather than drawing from scratch.

## Final Design: Hybrid Approach (Option C)

Following Raymond Hettinger's "make the simple case simple" and Brandon Rhodes' "separate concerns":

1. **Simple case**: Pass positions directly, get reasonable defaults
2. **Advanced case**: Use `BoundaryConfig` dataclass for tuning
3. **Composable**: `boundary_from_positions()` available as separate function

### API Usage Examples

```python
from neurospatial.annotation import annotate_video, boundary_from_positions, BoundaryConfig

# 90% of users - just pass positions, sensible defaults
result = annotate_video(
    "video.mp4",
    bin_size=2.0,
    initial_boundary=positions,  # NDArray → auto-infer boundary
)
# Uses: convex_hull, 2% buffer, 1% simplify

# 10% who need control - use BoundaryConfig
config = BoundaryConfig(method="kde", buffer_fraction=0.05)
result = annotate_video(
    "video.mp4",
    bin_size=2.0,
    initial_boundary=positions,
    boundary_config=config,
)

# Or pre-compute boundary with full control
boundary = boundary_from_positions(
    positions,
    method="alpha_shape",
    alpha=0.05,
    buffer_fraction=0.03,
    simplify_fraction=0.02,
)
result = annotate_video("video.mp4", bin_size=2.0, initial_boundary=boundary)

# Show positions as reference while editing
result = annotate_video(
    "video.mp4",
    bin_size=2.0,
    initial_boundary=positions,
    show_positions=True,  # Adds Points layer
)
```

## Resolved Design Decisions

| Decision | Resolution |
|----------|------------|
| Show positions in napari | Optional `show_positions=False` parameter |
| Buffer/padding | `buffer_fraction=0.02` (2% of bbox diagonal) |
| Auto-simplify | `simplify_fraction=0.01` (1% of bbox diagonal) |
| Units | Fraction of bounding box diagonal (scale-independent) |
| MultiPolygon from alpha | Take largest polygon + emit warning |
| Parameter naming | `initial_boundary` (accepts Polygon or NDArray) |
| Export location | `from neurospatial.annotation import boundary_from_positions` |

## Implementation Tasks

### Task 1: Create BoundaryConfig dataclass

**File**: `src/neurospatial/annotation/_boundary_inference.py`

```python
"""Boundary inference algorithms for seeding annotation."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPolygon, Polygon

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class BoundaryConfig:
    """
    Configuration for boundary inference from positions.

    Parameters
    ----------
    method : {"convex_hull", "alpha_shape", "kde"}
        Boundary inference algorithm. Default is "convex_hull".
    buffer_fraction : float
        Buffer size as fraction of bounding box diagonal.
        Default 0.02 (2%) adds small padding around boundary.
    simplify_fraction : float
        Simplification tolerance as fraction of bounding box diagonal.
        Default 0.01 (1%) removes jagged edges. Set to 0 to disable.
    alpha : float
        Alpha parameter for alpha_shape method (smaller = tighter fit).
        Only used when method="alpha_shape". Default 0.05.
    kde_threshold : float
        Density threshold for KDE boundary (0-1, fraction of max density).
        Only used when method="kde". Default 0.1.
    kde_sigma : float
        Gaussian smoothing sigma for KDE (in grid bins).
        Only used when method="kde". Default 3.0.
    kde_max_bins : int
        Maximum number of bins per dimension for KDE grid.
        Caps memory usage for large coordinate ranges. Default 512.

    Examples
    --------
    >>> config = BoundaryConfig(method="kde", buffer_fraction=0.05)
    >>> boundary = boundary_from_positions(positions, config=config)
    """

    method: Literal["convex_hull", "alpha_shape", "kde"] = "convex_hull"
    buffer_fraction: float = 0.02
    simplify_fraction: float = 0.01
    alpha: float = 0.05
    kde_threshold: float = 0.1
    kde_sigma: float = 3.0
    kde_max_bins: int = 512


def boundary_from_positions(
    positions: NDArray[np.float64],
    method: Literal["convex_hull", "alpha_shape", "kde"] | None = None,
    *,
    config: BoundaryConfig | None = None,
    # Convenience overrides (applied after config)
    buffer_fraction: float | None = None,
    simplify_fraction: float | None = None,
    **method_kwargs,
) -> Polygon:
    """
    Infer environment boundary from position data.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, 2)
        Animal positions in (x, y) format.
    method : {"convex_hull", "alpha_shape", "kde"}, optional
        Boundary inference algorithm. Overrides config.method if provided.
    config : BoundaryConfig, optional
        Full configuration object. If None, uses BoundaryConfig defaults.
    buffer_fraction : float, optional
        Override config.buffer_fraction.
    simplify_fraction : float, optional
        Override config.simplify_fraction.
    **method_kwargs
        Method-specific overrides (alpha, kde_threshold, kde_sigma).

    Returns
    -------
    Polygon
        Shapely Polygon representing inferred boundary.

    Raises
    ------
    ValueError
        If positions has wrong shape or insufficient points.
    ImportError
        If method="alpha_shape" and alphashape not installed.

    Examples
    --------
    >>> positions = np.random.rand(1000, 2) * 100
    >>> boundary = boundary_from_positions(positions)  # Uses defaults
    >>> boundary.is_valid
    True

    >>> # With custom config
    >>> config = BoundaryConfig(method="kde", buffer_fraction=0.05)
    >>> boundary = boundary_from_positions(positions, config=config)
    """
    # Validate and cast input
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            f"positions must have shape (n, 2), got {positions.shape}"
        )
    if len(positions) < 3:
        raise ValueError(
            f"positions must have at least 3 points, got {len(positions)}"
        )

    # Check for degenerate cases (all points identical or collinear)
    unique_points = np.unique(positions, axis=0)
    if len(unique_points) < 3:
        raise ValueError(
            f"positions must have at least 3 unique points, got {len(unique_points)}"
        )

    # Build effective config
    cfg = config or BoundaryConfig()
    effective_method = method or cfg.method
    effective_buffer = buffer_fraction if buffer_fraction is not None else cfg.buffer_fraction
    effective_simplify = simplify_fraction if simplify_fraction is not None else cfg.simplify_fraction

    # Compute bounding box diagonal for fraction-based parameters
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)

    # Dispatch to method
    if effective_method == "convex_hull":
        boundary = _convex_hull_boundary(positions)
    elif effective_method == "alpha_shape":
        alpha = method_kwargs.get("alpha", cfg.alpha)
        boundary = _alpha_shape_boundary(positions, alpha)
    elif effective_method == "kde":
        threshold = method_kwargs.get("kde_threshold", cfg.kde_threshold)
        sigma = method_kwargs.get("kde_sigma", cfg.kde_sigma)
        max_bins = method_kwargs.get("kde_max_bins", cfg.kde_max_bins)
        boundary = _kde_boundary(positions, threshold, sigma, max_bins)
    else:
        raise ValueError(f"Unknown method: {effective_method}")

    # Apply buffer (fraction of bbox diagonal)
    if effective_buffer > 0:
        buffer_distance = effective_buffer * bbox_diagonal
        boundary = boundary.buffer(buffer_distance, join_style="round")

    # Apply simplification (fraction of bbox diagonal)
    if effective_simplify > 0:
        simplify_tolerance = effective_simplify * bbox_diagonal
        boundary = boundary.simplify(simplify_tolerance, preserve_topology=True)

    return boundary


def _convex_hull_boundary(positions: NDArray[np.float64]) -> Polygon:
    """Compute convex hull boundary."""
    from scipy.spatial import QhullError

    try:
        hull = ConvexHull(positions)
    except QhullError as e:
        raise ValueError(
            "Cannot compute convex hull: points may be collinear or degenerate. "
            f"Original error: {e}"
        ) from e

    vertices = positions[hull.vertices]
    return Polygon(vertices)


def _alpha_shape_boundary(
    positions: NDArray[np.float64],
    alpha: float,
) -> Polygon:
    """Compute alpha shape (concave hull) boundary."""
    try:
        import alphashape
    except ImportError:
        raise ImportError(
            "alphashape package required for alpha_shape method. "
            "Install with: pip install alphashape"
        ) from None

    result = alphashape.alphashape(positions, alpha)

    # Handle MultiPolygon: take largest, warn user
    if isinstance(result, MultiPolygon):
        largest = max(result.geoms, key=lambda g: g.area)
        warnings.warn(
            f"Alpha shape produced {len(result.geoms)} disconnected regions. "
            f"Using largest polygon (area={largest.area:.1f}). "
            "Consider increasing alpha parameter for a single connected region.",
            UserWarning,
            stacklevel=3,
        )
        return largest

    return result


def _kde_boundary(
    positions: NDArray[np.float64],
    threshold: float,
    sigma: float,
    max_bins: int = 512,
) -> Polygon:
    """Compute KDE-based boundary from density contour."""
    try:
        from scipy.ndimage import gaussian_filter
        from skimage.measure import find_contours
    except ImportError as e:
        raise ImportError(
            "scikit-image is required for KDE boundary method. "
            "Install with: pip install scikit-image"
        ) from e

    # Create 2D histogram with capped bin count
    x_range = positions[:, 0].max() - positions[:, 0].min()
    y_range = positions[:, 1].max() - positions[:, 1].min()
    # Target ~2 unit bins but cap to prevent memory issues
    n_bins = min(max_bins, max(50, int(max(x_range, y_range) / 2)))

    hist, x_edges, y_edges = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=n_bins
    )

    # Smooth and normalize
    hist_smooth = gaussian_filter(hist, sigma=sigma)
    hist_norm = hist_smooth / hist_smooth.max()

    # Find contour at threshold
    contours = find_contours(hist_norm.T, level=threshold)
    if not contours:
        raise ValueError(
            f"No contour found at threshold {threshold}. "
            "Try lowering kde_threshold or increasing kde_sigma."
        )

    # Take largest contour
    largest = max(contours, key=len)

    # Convert grid indices to coordinates
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    coords = np.column_stack([
        np.interp(largest[:, 1], np.arange(len(x_centers)), x_centers),
        np.interp(largest[:, 0], np.arange(len(y_centers)), y_centers),
    ])

    return Polygon(coords)
```

### Task 2: Add helper to add initial boundary to napari shapes layer

**File**: `src/neurospatial/annotation/_napari_widget.py` (add function)

**IMPORTANT**: Must follow existing patterns in `_add_initial_regions()`:
- Use `calibration.transform_cm_to_px()` (NOT `inverse_transform`)
- `transform_cm_to_px` already handles Y-flip internally - do NOT double-flip
- Use `rebuild_features()` from `_helpers.py` for consistency
- When no calibration, coords are already in pixels - no Y-flip needed

```python
def add_initial_boundary_to_shapes(
    shapes_layer: "napari.layers.Shapes",
    boundary: "Polygon",
    calibration: "VideoCalibration | None" = None,
) -> None:
    """
    Add pre-drawn boundary polygon to shapes layer for editing.

    Parameters
    ----------
    shapes_layer : napari.layers.Shapes
        The annotation shapes layer (may already contain shapes from initial_regions).
    boundary : Polygon
        Shapely Polygon. Coordinate system depends on calibration:
        - With calibration: environment units (cm), Y-up origin
        - Without calibration: video pixels (x, y), Y-down origin
    calibration : VideoCalibration, optional
        Transform from environment coords (cm) to video pixels.
        If None, boundary coords are assumed to be in video pixels already.

    Notes
    -----
    Mirrors the pattern in _add_initial_regions() for consistency.
    Preserves existing shapes/features and appends the boundary.
    """
    from shapely import get_coordinates

    from neurospatial.annotation._helpers import (
        rebuild_features,
        sync_face_colors_from_features,
    )

    # Preserve existing features before adding new shape
    existing_roles = list(shapes_layer.features.get("role", []))
    existing_names = list(shapes_layer.features.get("name", []))

    # Get polygon exterior vertices
    coords = get_coordinates(boundary.exterior)

    # Transform to pixels if calibration provided
    # NOTE: transform_cm_to_px handles Y-flip internally - don't double-flip!
    if calibration is not None:
        coords = calibration.transform_cm_to_px(coords)

    # Convert to napari (row, col) order
    coords_rc = coords[:, ::-1]

    # Add to shapes layer (appends to existing shapes)
    shapes_layer.add([coords_rc], shape_type="polygon")

    # Extend features with the new boundary shape
    # NOTE: Boundary should be first in the list for proper widget mode handling
    new_roles = ["environment"] + existing_roles
    new_names = ["arena"] + existing_names
    shapes_layer.features = rebuild_features(new_roles, new_names)

    # Reorder data so boundary is first (environment boundary should be drawn first)
    # This ensures the widget's mode logic works correctly
    if len(existing_roles) > 0:
        # Move the last shape (just added) to the front
        data = list(shapes_layer.data)
        data = [data[-1]] + data[:-1]
        shapes_layer.data = data

    # Sync face colors from features
    sync_face_colors_from_features(shapes_layer)
```

**IMPORTANT: Ordering of helpers**

The order of calling `add_initial_boundary_to_shapes` vs `_add_initial_regions` matters:

1. Call `_add_initial_regions()` FIRST (adds non-environment regions)
2. Call `add_initial_boundary_to_shapes()` SECOND (prepends environment boundary)

This ensures the environment boundary appears first in the shapes list, which is important for the widget's mode cycling logic. The `add_initial_boundary_to_shapes` function handles reordering internally.

### Task 3: Modify annotate_video() to accept initial boundary

**File**: `src/neurospatial/annotation/core.py`

Add parameters to `annotate_video()`:

```python
def annotate_video(
    video_path: str | Path,
    *,
    # Existing parameters...
    frame_index: int = 0,
    bin_size: float | None = None,
    calibration: VideoCalibration | None = None,
    # NEW: Boundary seeding parameters
    initial_boundary: Polygon | NDArray[np.float64] | None = None,
    boundary_config: BoundaryConfig | None = None,
    show_positions: bool = False,
    # Rest of existing parameters...
) -> AnnotationResult:
    """
    Interactively annotate a video frame to define environment and regions.

    Parameters
    ----------
    ...existing params...
    initial_boundary : Polygon | NDArray | None
        Pre-drawn boundary for editing. Can be:
        - Shapely Polygon: Used directly
        - NDArray (n, 2): Position data to infer boundary from
        If None, user draws boundary manually.
    boundary_config : BoundaryConfig, optional
        Configuration for boundary inference when initial_boundary is an array.
        If None, uses BoundaryConfig defaults (convex_hull, 2% buffer, 1% simplify).
    show_positions : bool
        If True and initial_boundary is an array, show positions as a
        Points layer for reference while editing. Default False.
    ...
    """
```

Add logic before napari viewer opens:

```python
# Process initial boundary
boundary_polygon = None
positions_for_display = None

if initial_boundary is not None:
    if isinstance(initial_boundary, np.ndarray):
        # Infer boundary from positions
        from neurospatial.annotation._boundary_inference import (
            boundary_from_positions,
        )
        boundary_polygon = boundary_from_positions(
            initial_boundary,
            config=boundary_config,
        )
        if show_positions:
            positions_for_display = initial_boundary
    else:
        # Assume Shapely Polygon
        boundary_polygon = initial_boundary

# Handle conflict: initial_boundary vs environment region in initial_regions
# initial_boundary takes precedence - warn if both provided
if boundary_polygon is not None and initial_regions is not None:
    env_regions = [
        name for name, r in initial_regions.items()
        if r.metadata.get("role") == "environment"
    ]
    if env_regions:
        import warnings
        warnings.warn(
            f"Both initial_boundary and environment regions in initial_regions "
            f"({env_regions}) provided. Using initial_boundary; ignoring "
            f"environment regions from initial_regions.",
            UserWarning,
            stacklevel=2,
        )
        # Filter out environment regions from initial_regions
        # Note: Regions() expects Iterable[Region], not dict
        from neurospatial.regions import Regions
        initial_regions = Regions(
            r for r in initial_regions.values()
            if r.metadata.get("role") != "environment"
        )

# ... create napari viewer and shapes layer ...

# IMPORTANT: Order matters for feature preservation
# 1. First add initial_regions (if any) - these are non-environment regions
if initial_regions is not None:
    _add_initial_regions(shapes_layer, initial_regions, calibration)

# 2. Then add initial_boundary - this prepends to front and reorders
if boundary_polygon is not None:
    from neurospatial.annotation._napari_widget import (
        add_initial_boundary_to_shapes,
    )
    add_initial_boundary_to_shapes(
        shapes_layer,
        boundary_polygon,
        calibration=calibration,
    )

# 3. Finally add positions layer (separate layer, no conflict)
if positions_for_display is not None:
    _add_positions_layer(viewer, positions_for_display, calibration)


def _add_positions_layer(
    viewer: "napari.Viewer",
    positions: NDArray[np.float64],
    calibration: "VideoCalibration | None",
) -> None:
    """Add positions as semi-transparent Points layer for reference.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer.
    positions : NDArray[np.float64]
        Position data. Coordinate system depends on calibration:
        - With calibration: environment units (cm), Y-up origin
        - Without calibration: video pixels (x, y), Y-down origin
    calibration : VideoCalibration, optional
        Transform from environment coords (cm) to video pixels.

    Notes
    -----
    Mirrors the pattern in add_initial_boundary_to_shapes for consistency.
    """
    coords = positions.copy()

    # Transform to pixels if calibration provided
    # NOTE: transform_cm_to_px handles Y-flip internally - don't double-flip!
    if calibration is not None:
        coords = calibration.transform_cm_to_px(coords)

    # Convert to napari (row, col) order
    coords_rc = coords[:, ::-1]

    # Subsample if too many points (for performance)
    if len(coords_rc) > 5000:
        step = len(coords_rc) // 5000
        coords_rc = coords_rc[::step]

    viewer.add_points(
        coords_rc,
        name="Trajectory (reference)",
        size=3,
        face_color="cyan",
        opacity=0.3,
        blending="translucent",
    )
```

### Task 4: Export in public API

**File**: `src/neurospatial/annotation/__init__.py`

```python
from neurospatial.annotation._boundary_inference import (
    BoundaryConfig,
    boundary_from_positions,
)
from neurospatial.annotation.core import annotate_video

__all__ = [
    "annotate_video",
    "BoundaryConfig",
    "boundary_from_positions",
    # ... existing exports
]
```

### Task 5: Add tests

**File**: `tests/annotation/test_boundary_inference.py`

```python
"""Tests for boundary inference algorithms."""
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from neurospatial.annotation import BoundaryConfig, boundary_from_positions


class TestBoundaryConfig:
    """Tests for BoundaryConfig dataclass."""

    def test_default_values(self):
        """Default config has sensible values."""
        config = BoundaryConfig()
        assert config.method == "convex_hull"
        assert config.buffer_fraction == 0.02
        assert config.simplify_fraction == 0.01

    def test_frozen(self):
        """Config is immutable."""
        config = BoundaryConfig()
        with pytest.raises(AttributeError):
            config.method = "kde"


class TestConvexHull:
    """Tests for convex hull boundary inference."""

    def test_basic_hull(self):
        """Convex hull contains all input points."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (1000, 2))

        # Disable buffer/simplify to test raw hull
        boundary = boundary_from_positions(
            positions,
            method="convex_hull",
            buffer_fraction=0,
            simplify_fraction=0,
        )

        assert boundary.is_valid
        # All points should be inside or on boundary
        for pos in positions[:100]:  # Check subset for speed
            assert boundary.contains(Point(pos)) or boundary.exterior.distance(Point(pos)) < 1e-10

    def test_square_points(self):
        """Hull of square corners is square."""
        positions = np.array([[0, 0], [0, 10], [10, 10], [10, 0], [5, 5]])

        boundary = boundary_from_positions(
            positions,
            method="convex_hull",
            buffer_fraction=0,
            simplify_fraction=0,
        )

        assert boundary.is_valid
        assert abs(boundary.area - 100) < 0.01  # 10x10 square

    def test_minimum_points(self):
        """Requires at least 3 points."""
        positions = np.array([[0, 0], [1, 1]])

        with pytest.raises(ValueError, match="at least 3"):
            boundary_from_positions(positions, method="convex_hull")


class TestBuffer:
    """Tests for buffer functionality."""

    def test_buffer_increases_area(self):
        """Buffer adds padding around boundary."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (100, 2))

        no_buffer = boundary_from_positions(
            positions, buffer_fraction=0, simplify_fraction=0
        )
        with_buffer = boundary_from_positions(
            positions, buffer_fraction=0.05, simplify_fraction=0
        )

        assert with_buffer.area > no_buffer.area

    def test_buffer_fraction_scales_with_bbox(self):
        """Buffer is proportional to bounding box diagonal."""
        # Small positions
        small = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
        # Large positions (10x bigger)
        large = small * 10

        small_boundary = boundary_from_positions(
            small, buffer_fraction=0.1, simplify_fraction=0
        )
        large_boundary = boundary_from_positions(
            large, buffer_fraction=0.1, simplify_fraction=0
        )

        # Area should scale by 100x (10^2)
        ratio = large_boundary.area / small_boundary.area
        assert 90 < ratio < 110  # Allow some tolerance


class TestSimplify:
    """Tests for simplification functionality."""

    def test_simplify_reduces_vertices(self):
        """Simplification reduces number of vertices."""
        rng = np.random.default_rng(42)
        # Create jagged positions that will produce many vertices with KDE
        positions = rng.uniform(0, 100, (1000, 2))

        raw = boundary_from_positions(
            positions, method="kde", simplify_fraction=0, buffer_fraction=0
        )
        simplified = boundary_from_positions(
            positions, method="kde", simplify_fraction=0.05, buffer_fraction=0
        )

        n_raw = len(raw.exterior.coords)
        n_simple = len(simplified.exterior.coords)
        assert n_simple < n_raw


class TestKDE:
    """Tests for KDE boundary inference."""

    def test_kde_import_error(self, monkeypatch):
        """Clear error message if scikit-image not installed."""
        import sys

        # Temporarily hide skimage module
        monkeypatch.setitem(sys.modules, "skimage", None)
        monkeypatch.setitem(sys.modules, "skimage.measure", None)

        positions = np.array([[0, 0], [0, 10], [10, 10], [10, 0], [5, 5]])

        with pytest.raises(ImportError, match="scikit-image is required"):
            boundary_from_positions(
                positions, method="kde",
                buffer_fraction=0, simplify_fraction=0
            )

    def test_basic_kde(self):
        """KDE boundary is valid polygon."""
        pytest.importorskip("skimage")

        rng = np.random.default_rng(42)
        positions = rng.uniform(20, 80, (1000, 2))

        boundary = boundary_from_positions(
            positions, method="kde", buffer_fraction=0, simplify_fraction=0
        )

        assert boundary.is_valid
        assert isinstance(boundary, Polygon)

    def test_max_bins_caps_grid_size(self):
        """kde_max_bins prevents unbounded grid allocation."""
        pytest.importorskip("skimage")

        rng = np.random.default_rng(42)
        # Large coordinate range that would create huge grid without cap
        positions = rng.uniform(0, 10000, (1000, 2))

        # Should not raise MemoryError due to max_bins cap
        boundary = boundary_from_positions(
            positions, method="kde", kde_max_bins=100,
            buffer_fraction=0, simplify_fraction=0
        )

        assert boundary.is_valid

    def test_threshold_affects_area(self):
        """Lower threshold = larger boundary."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (1000, 2))

        low_threshold = boundary_from_positions(
            positions, method="kde", kde_threshold=0.05,
            buffer_fraction=0, simplify_fraction=0
        )
        high_threshold = boundary_from_positions(
            positions, method="kde", kde_threshold=0.2,
            buffer_fraction=0, simplify_fraction=0
        )

        assert low_threshold.area > high_threshold.area


class TestAlphaShape:
    """Tests for alpha shape boundary inference."""

    def test_alpha_shape_import_error(self, monkeypatch):
        """Clear error message if alphashape not installed."""
        import sys

        # Temporarily hide alphashape module
        monkeypatch.setitem(sys.modules, "alphashape", None)

        positions = np.array([[0, 0], [0, 10], [10, 10], [10, 0], [5, 5]])

        with pytest.raises(ImportError, match="alphashape package required"):
            boundary_from_positions(
                positions, method="alpha_shape",
                buffer_fraction=0, simplify_fraction=0
            )

    def test_multipolygon_warning(self):
        """Warning emitted when alpha shape produces multiple polygons."""
        pytest.importorskip("alphashape")

        rng = np.random.default_rng(42)
        # Create two clusters that will produce MultiPolygon with high alpha
        cluster1 = rng.uniform(0, 10, (100, 2))
        cluster2 = rng.uniform(90, 100, (100, 2))
        positions = np.vstack([cluster1, cluster2])

        with pytest.warns(UserWarning, match="disconnected regions"):
            boundary = boundary_from_positions(
                positions, method="alpha_shape", alpha=0.5,
                buffer_fraction=0, simplify_fraction=0
            )

        # Should return largest polygon
        assert isinstance(boundary, Polygon)


class TestWithConfig:
    """Tests for BoundaryConfig integration."""

    def test_config_overrides_defaults(self):
        """Config parameters override defaults."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (500, 2))

        config = BoundaryConfig(method="kde", buffer_fraction=0.1)
        boundary = boundary_from_positions(positions, config=config)

        assert boundary.is_valid

    def test_kwargs_override_config(self):
        """Explicit kwargs override config values."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (100, 2))

        config = BoundaryConfig(buffer_fraction=0.1)

        # buffer_fraction=0 should override config's 0.1
        no_buffer = boundary_from_positions(
            positions, config=config, buffer_fraction=0, simplify_fraction=0
        )
        with_buffer = boundary_from_positions(
            positions, config=config, simplify_fraction=0
        )

        assert with_buffer.area > no_buffer.area
```

### Task 6: Update CLAUDE.md documentation

Add to Quick Reference section:

```python
# Seed boundary from position data (v0.9.0+)
from neurospatial.annotation import annotate_video, boundary_from_positions, BoundaryConfig

# Simple: just pass positions (90% of use cases)
result = annotate_video(
    "experiment.mp4",
    bin_size=2.0,
    initial_boundary=positions,  # Auto-infer with sensible defaults
)
# Uses: convex_hull, 2% buffer, 1% simplify (fraction of bbox diagonal)

# With config for fine-tuning
config = BoundaryConfig(method="kde", buffer_fraction=0.05)
result = annotate_video(
    "experiment.mp4",
    bin_size=2.0,
    initial_boundary=positions,
    boundary_config=config,
)

# Show trajectory as reference while editing
result = annotate_video(
    "experiment.mp4",
    bin_size=2.0,
    initial_boundary=positions,
    show_positions=True,  # Adds semi-transparent Points layer
)

# Composable: create boundary explicitly
boundary = boundary_from_positions(
    positions,
    method="alpha_shape",
    alpha=0.05,
    buffer_fraction=0.03,
)
result = annotate_video("experiment.mp4", bin_size=2.0, initial_boundary=boundary)
```

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `src/neurospatial/annotation/_boundary_inference.py` | **CREATE** | BoundaryConfig + inference algorithms |
| `src/neurospatial/annotation/_napari_widget.py` | MODIFY | Add `add_initial_boundary_to_shapes()` |
| `src/neurospatial/annotation/core.py` | MODIFY | Add `initial_boundary`, `boundary_config`, `show_positions` |
| `src/neurospatial/annotation/__init__.py` | MODIFY | Export `BoundaryConfig`, `boundary_from_positions` |
| `tests/annotation/test_boundary_inference.py` | **CREATE** | Comprehensive tests |
| `CLAUDE.md` | MODIFY | Document new feature |

## Dependencies

- **Required**: `scipy` (already present), `shapely` (already present)
- **Lazy imports with friendly errors**:
  - `scikit-image` - Required only for `method="kde"`. ImportError with install instructions if missing.
  - `alphashape` - Required only for `method="alpha_shape"`. ImportError with install instructions if missing.

Update `pyproject.toml`:

```toml
# pyproject.toml
[project.optional-dependencies]
# KDE boundary method requires scikit-image
annotation-kde = ["scikit-image>=0.19.0"]
# Alpha shape boundary method
annotation-alpha = ["alphashape>=1.3.0"]
# All annotation extras
annotation = [
    "scikit-image>=0.19.0",
    "alphashape>=1.3.0",
]
```

**Note**: `convex_hull` method works with only scipy (already a core dependency), making boundary seeding available to all users without extra installs.

## Backward Compatibility

- All new parameters default to `None` / `False`
- Existing `annotate_video()` calls work unchanged
- No breaking changes to public API

## Important Implementation Notes

### Coordinate System Documentation

Document explicitly in docstrings:

| Parameter | With Calibration | Without Calibration |
|-----------|-----------------|---------------------|
| `positions` / `initial_boundary` | Environment units (cm), Y-up origin | Video pixels (x, y), Y-down origin |
| Returned `Environment` | Always in environment units | Pixels treated as environment units |

### Double Simplification

There are two places where simplification can occur:

1. `boundary_from_positions(simplify_fraction=0.01)` - Applied to inferred boundary
2. `annotate_video(simplify_tolerance=...)` / `shapes_to_regions()` - Applied after user edits

**Resolution**: Document that `simplify_fraction` in `BoundaryConfig` is for the initial inference only. Final simplification after user edits is controlled by existing `simplify_tolerance` parameter. Both are valid use cases:
- Initial simplification: Removes jagged edges from KDE/alpha shape before display
- Final simplification: Cleans up user's hand-drawn adjustments

### Conflict Resolution

When both `initial_boundary` and `initial_regions` contain environment boundaries:
- `initial_boundary` takes precedence
- Environment regions from `initial_regions` are filtered out
- Warning emitted to inform user

### Error Handling Hierarchy

```
ValueError: positions shape/count issues
     └── ValueError: degenerate points (< 3 unique)
           └── ValueError: QhullError (collinear points)
                 └── ImportError: missing optional dependency
```
