# Video Annotation Feature Implementation Plan

## Overview

This plan implements an interactive annotation system for defining spatial environments and regions from video frames. The feature enables users to:

1. **Annotate directly in napari**: Draw environment boundaries and named regions on video frames
2. **Import from external tools**: Load annotations from LabelMe JSON or CVAT XML
3. **Create Environments**: Convert annotated boundaries to discretized `Environment` objects
4. **Apply calibration**: Transform pixel coordinates to world coordinates (cm) using `VideoCalibration`

## Architecture

### Module Structure

```
src/neurospatial/annotation/
├── __init__.py          # Public API exports
├── core.py              # annotate_video() main entry point
├── converters.py        # shapes_to_regions, env_from_boundary_region
├── io.py                # regions_from_labelme, regions_from_cvat
└── _napari_widget.py    # Qt dock widget for annotation UX
```

### Dependencies on Existing Code

| Component | Location | Usage |
|-----------|----------|-------|
| `VideoCalibration` | `transforms.py:721-835` | Pixel↔cm transforms via `transform_px_to_cm` |
| `Region`/`Regions` | `regions/core.py:36,161` | Annotation output data structures |
| `load_labelme_json` | `regions/io.py:83` | Already accepts `pixel_to_world: SpatialTransform` |
| `load_cvat_xml` | `regions/io.py:785` | Already accepts `pixel_to_world: SpatialTransform` |
| `VideoReader` | `animation/_video_io.py:99-350` | Load video frames |
| `Environment.from_polygon` | `environment/factories.py:399` | Create env from boundary |
| `SpatialTransform` | `transforms.py:44` | Protocol for coordinate transforms |

### Coordinate Systems

**Video Pixels**: Origin top-left, (x, y) = (column, row)
**Napari Shapes**: Origin top-left, (row, col) order
**Environment**: Origin bottom-left, (x, y) with Y increasing upward

Transform pipeline:
```
napari (row, col) → video (x, y) pixels → calibration → world (x, y) cm
```

### Napari Annotation Best Practices

Based on [napari annotation tutorials](https://napari.org/stable/tutorials/annotation/annotate_points.html) and [segmentation guide](https://napari.org/stable/tutorials/segmentation/annotate_segmentation.html):

| Pattern | Description | Implementation |
|---------|-------------|----------------|
| **Features-based coloring** | Use `features` dict with `face_color='role'` for automatic color cycling | Shapes layer uses `face_color_cycle` to distinguish environment (cyan) vs region (yellow) |
| **Text labels on shapes** | Display metadata directly on shapes via `text` parameter | Shape names shown with `text={'string': '{name}', ...}` |
| **feature_defaults** | Auto-set metadata for newly-drawn shapes | New shapes automatically get role and auto-incremented name |
| **Keyboard shortcuts** | Bind keys for common actions | `E` = environment mode, `R` = region mode |
| **Layer data events** | Respond to data changes | `shapes.events.data.connect()` for reliable shape-added detection |
| **magicgui widgets** | Prefer over raw Qt for napari integration | `ComboBox`, `LineEdit`, `PushButton` from magicgui |
| **Categorical features** | Use `pd.Categorical` for color cycling | Enables proper color assignment per category |

**Key code patterns:**

```python
# Initialize features with pandas DataFrame (enables color cycling)
features = pd.DataFrame({
    "role": pd.Categorical([], categories=["region", "environment"]),
    "name": pd.Series([], dtype=str),
})

# Add shapes layer with features-based coloring and text labels
shapes = viewer.add_shapes(
    features=features,
    face_color="role",                    # Color by role feature
    face_color_cycle=["yellow", "cyan"],  # region=yellow, environment=cyan
    text={"string": "{name}", "size": 10, "color": "white"},
)

# Set defaults for new shapes
shapes.feature_defaults["role"] = "region"
shapes.feature_defaults["name"] = "region_0"

# Keyboard shortcut
@viewer.bind_key("e")
def set_environment_mode(viewer):
    shapes.feature_defaults["role"] = "environment"

# Layer data event for auto-increment (more reliable than mouse callbacks)
_prev_count = [0]
def on_data_changed(event):
    if len(shapes.data) > _prev_count[0]:
        shapes.feature_defaults["name"] = f"region_{len(shapes.data)}"
    _prev_count[0] = len(shapes.data)
shapes.events.data.connect(on_data_changed)
```

---

## Milestone 1: Module Structure and Public API

### Task 1.1: Create `annotation/__init__.py`

**File**: `src/neurospatial/annotation/__init__.py`

```python
"""Video annotation tools for defining environments and regions."""

from neurospatial.annotation.converters import (
    env_from_boundary_region,
    shapes_to_regions,
)
from neurospatial.annotation.core import AnnotationResult, annotate_video
from neurospatial.annotation.io import regions_from_cvat, regions_from_labelme

__all__ = [
    "AnnotationResult",
    "annotate_video",
    "shapes_to_regions",
    "env_from_boundary_region",
    "regions_from_labelme",
    "regions_from_cvat",
]
```

### Task 1.2: Update main package `__init__.py`

**File**: `src/neurospatial/__init__.py`

Add to imports:
```python
from neurospatial.annotation import (
    AnnotationResult,
    annotate_video,
    regions_from_cvat,
    regions_from_labelme,
)
```

Add to `__all__`:
```python
"AnnotationResult",
"annotate_video",
"regions_from_labelme",
"regions_from_cvat",
```

---

## Milestone 2: Converters Module

### Task 2.1: Create `converters.py`

**File**: `src/neurospatial/annotation/converters.py`

```python
"""Convert between napari shapes and neurospatial Regions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shapely.geometry as shp

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from neurospatial import Environment
    from neurospatial.regions import Region, Regions
    from neurospatial.transforms import VideoCalibration


def shapes_to_regions(
    shapes_data: list[NDArray[np.float64]],
    names: list[str],
    roles: list[str],
    calibration: VideoCalibration | None = None,
    simplify_tolerance: float | None = None,
) -> tuple[Regions, Region | None]:
    """
    Convert napari polygon shapes to Regions.

    Parameters
    ----------
    shapes_data : list of NDArray
        List of polygon vertices from napari Shapes layer. Each array has
        shape (n_vertices, 2) in napari (row, col) order.
    names : list of str
        Name for each shape.
    roles : list of str
        Role for each shape: "environment" or "region".
    calibration : VideoCalibration, optional
        If provided, transforms pixel coordinates to world coordinates (cm)
        using ``calibration.transform_px_to_cm``.
    simplify_tolerance : float, optional
        If provided, simplifies polygons using Shapely's Douglas-Peucker
        algorithm. Tolerance is in output coordinate units (cm if calibration
        provided, else pixels). Recommended: 1.0 for cleaner boundaries.

    Returns
    -------
    regions : Regions
        All regions with role="region".
    env_boundary : Region or None
        The region with role="environment", if any.

    Notes
    -----
    Napari shapes use (row, col) order. This function converts to (x, y)
    pixel coordinates before applying calibration.
    """
    from neurospatial.regions import Region, Regions

    regions_list: list[Region] = []
    env_boundary: Region | None = None

    for poly_rc, name, role in zip(shapes_data, names, roles, strict=True):
        # Convert napari (row, col) to video (x, y) pixels
        pts_px = poly_rc[:, ::-1].astype(np.float64)

        # Apply calibration if available
        if calibration is not None:
            pts_world = calibration.transform_px_to_cm(pts_px)
            coord_system = "cm"
        else:
            pts_world = pts_px
            coord_system = "pixels"

        # Skip invalid polygons
        if len(pts_world) < 3:
            continue

        poly = shp.Polygon(pts_world)

        # Optional simplification (Douglas-Peucker algorithm)
        if simplify_tolerance is not None:
            poly = shp.simplify(poly, tolerance=simplify_tolerance, preserve_topology=True)

        metadata = {
            "source": "napari_annotation",
            "coord_system": coord_system,
            "role": role,
        }

        region = Region(
            name=str(name),
            kind="polygon",
            data=poly,
            metadata=metadata,
        )

        if role == "environment":
            env_boundary = region
        else:
            regions_list.append(region)

    return Regions(regions_list), env_boundary


def env_from_boundary_region(
    boundary: Region,
    bin_size: float,
    **from_polygon_kwargs,
) -> Environment:
    """
    Create Environment from an annotated boundary polygon.

    Parameters
    ----------
    boundary : Region
        Region with kind="polygon" defining the environment boundary.
    bin_size : float
        Bin size for discretization (in same units as boundary coordinates).
    **from_polygon_kwargs
        Additional arguments passed to ``Environment.from_polygon()``.

    Returns
    -------
    Environment
        Discretized environment fitted to the boundary polygon.

    Raises
    ------
    ValueError
        If boundary is not a polygon region.

    See Also
    --------
    Environment.from_polygon : Factory method used internally.
    """
    from neurospatial import Environment

    if boundary.kind != "polygon":
        raise ValueError(f"Boundary must be polygon, got {boundary.kind}")

    return Environment.from_polygon(
        polygon=boundary.data,
        bin_size=bin_size,
        **from_polygon_kwargs,
    )
```

### Task 2.2: Verification

```bash
uv run python -c "from neurospatial.annotation.converters import shapes_to_regions, env_from_boundary_region; print('OK')"
```

---

## Milestone 3: External Tool Import Wrappers

### Task 3.1: Create `io.py`

**File**: `src/neurospatial/annotation/io.py`

```python
"""Import annotations from external tools (LabelMe, CVAT)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neurospatial.regions import Regions
    from neurospatial.transforms import VideoCalibration


def regions_from_labelme(
    json_path: str | Path,
    calibration: VideoCalibration | None = None,
    *,
    label_key: str = "label",
    points_key: str = "points",
) -> Regions:
    """
    Load regions from LabelMe JSON with optional calibration.

    Parameters
    ----------
    json_path : str or Path
        Path to LabelMe JSON file.
    calibration : VideoCalibration, optional
        If provided, transforms pixel coordinates to world coordinates (cm).
    label_key : str, default="label"
        Key in JSON for region name.
    points_key : str, default="points"
        Key in JSON for polygon vertices.

    Returns
    -------
    Regions
        Loaded regions with coordinates in cm (if calibrated) or pixels.

    See Also
    --------
    neurospatial.regions.io.load_labelme_json : Underlying implementation.

    Examples
    --------
    >>> from neurospatial.annotation import regions_from_labelme
    >>> from neurospatial.transforms import VideoCalibration, calibrate_from_scale_bar
    >>> # Without calibration (pixel coordinates)
    >>> regions = regions_from_labelme("annotations.json")
    >>> # With calibration (cm coordinates)
    >>> transform = calibrate_from_scale_bar((0, 0), (100, 0), 50.0, (640, 480))
    >>> calib = VideoCalibration(transform, (640, 480))
    >>> regions = regions_from_labelme("annotations.json", calibration=calib)
    """
    from neurospatial.regions.io import load_labelme_json

    pixel_to_world = calibration.transform_px_to_cm if calibration else None
    return load_labelme_json(
        json_path,
        pixel_to_world=pixel_to_world,
        label_key=label_key,
        points_key=points_key,
    )


def regions_from_cvat(
    xml_path: str | Path,
    calibration: VideoCalibration | None = None,
) -> Regions:
    """
    Load regions from CVAT XML with optional calibration.

    Parameters
    ----------
    xml_path : str or Path
        Path to CVAT XML export file.
    calibration : VideoCalibration, optional
        If provided, transforms pixel coordinates to world coordinates (cm).

    Returns
    -------
    Regions
        Loaded regions with coordinates in cm (if calibrated) or pixels.

    See Also
    --------
    neurospatial.regions.io.load_cvat_xml : Underlying implementation.

    Examples
    --------
    >>> from neurospatial.annotation import regions_from_cvat
    >>> regions = regions_from_cvat("cvat_export.xml")
    """
    from neurospatial.regions.io import load_cvat_xml

    pixel_to_world = calibration.transform_px_to_cm if calibration else None
    return load_cvat_xml(xml_path, pixel_to_world=pixel_to_world)
```

### Task 3.2: Verification

```bash
uv run python -c "from neurospatial.annotation.io import regions_from_labelme, regions_from_cvat; print('OK')"
```

---

## Milestone 4: Napari Annotation Widget

> **Note**: This module implements the patterns from [Napari Annotation Best Practices](#napari-annotation-best-practices) in the Architecture section above.

### Task 4.1: Create `_napari_widget.py`

**File**: `src/neurospatial/annotation/_napari_widget.py`

```python
"""Magicgui-based widget for napari annotation workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from magicgui.widgets import ComboBox, Container, Label, LineEdit, PushButton

if TYPE_CHECKING:
    import napari

# Role categories - order determines color cycle mapping
ROLE_CATEGORIES = ["region", "environment"]

# Color scheme for role-based visualization
# Order must match ROLE_CATEGORIES
ROLE_COLORS = {
    "region": "yellow",
    "environment": "cyan",
}
ROLE_COLOR_CYCLE = [ROLE_COLORS[cat] for cat in ROLE_CATEGORIES]  # ["yellow", "cyan"]


def create_annotation_widget(
    viewer: napari.Viewer,
    shapes_layer_name: str = "Annotations",
) -> Container:
    """
    Create annotation control widget using magicgui.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    shapes_layer_name : str, default="Annotations"
        Name of the Shapes layer to manage.

    Returns
    -------
    Container
        Magicgui container widget with annotation controls.
    """
    # Get shapes layer
    def get_shapes():
        try:
            return viewer.layers[shapes_layer_name]
        except KeyError:
            return None

    # --- Widget Components ---
    instructions = Label(
        value=(
            "Draw polygons: E=environment, R=region\n"
            "Edit: 3=move shape, 4=edit vertices, Delete=remove\n"
            "Click shape to select, then Apply to change metadata"
        )
    )

    role_selector = ComboBox(
        choices=["region", "environment"],
        value="region",
        label="New shape role:",
    )

    name_input = LineEdit(value="region_0", label="New shape name:")

    apply_btn = PushButton(text="Apply to Selected")
    save_btn = PushButton(text="Save and Close")

    # --- Event Handlers ---
    @role_selector.changed.connect
    def on_role_changed(role: str):
        """Update feature_defaults when role selector changes."""
        shapes = get_shapes()
        if shapes is not None:
            shapes.feature_defaults["role"] = role
            # Update colors for visual feedback
            shapes.current_face_color = ROLE_COLORS.get(role, "yellow")

    @name_input.changed.connect
    def on_name_changed(name: str):
        """Update feature_defaults when name changes."""
        shapes = get_shapes()
        if shapes is not None:
            shapes.feature_defaults["name"] = name

    @apply_btn.clicked.connect
    def apply_to_selected():
        """Apply current role/name to selected shapes."""
        shapes = get_shapes()
        if shapes is None or len(shapes.selected_data) == 0:
            return

        # Use DataFrame operations for safer feature updates
        features_df = shapes.features.copy()
        for idx in shapes.selected_data:
            if 0 <= idx < len(features_df):
                features_df.loc[idx, "role"] = role_selector.value
                features_df.loc[idx, "name"] = name_input.value

        shapes.features = features_df
        shapes.refresh()

    @save_btn.clicked.connect
    def save_and_close():
        """Close viewer to return control to Python."""
        viewer.close()

    # --- Keyboard Shortcuts ---
    @viewer.bind_key("e")
    def set_environment_mode(viewer):
        """Set mode to draw environment boundary."""
        role_selector.value = "environment"
        shapes = get_shapes()
        if shapes:
            shapes.feature_defaults["role"] = "environment"
            shapes.current_face_color = ROLE_COLORS["environment"]

    @viewer.bind_key("r")
    def set_region_mode(viewer):
        """Set mode to draw named region."""
        role_selector.value = "region"
        shapes = get_shapes()
        if shapes:
            shapes.feature_defaults["role"] = "region"
            shapes.current_face_color = ROLE_COLORS["region"]

    # --- Data Change Callback for Auto-increment ---
    # Use events.data instead of mouse_drag_callbacks for reliable detection
    # of when shapes are actually added (learned from napari-segment-anything)
    _prev_shape_count = [0]  # Mutable container for closure

    def on_data_changed(event):
        """Auto-increment name when a new shape is added."""
        shapes = get_shapes()
        if shapes is None:
            return

        current_count = len(shapes.data)
        if current_count > _prev_shape_count[0]:
            # New shape was added - update name for next shape
            new_name = f"region_{current_count}"
            name_input.value = new_name
            shapes.feature_defaults["name"] = new_name
        _prev_shape_count[0] = current_count

    shapes = get_shapes()
    if shapes is not None:
        shapes.events.data.connect(on_data_changed)
        _prev_shape_count[0] = len(shapes.data)

    # --- Build Container ---
    widget = Container(
        widgets=[
            instructions,
            role_selector,
            name_input,
            apply_btn,
            save_btn,
        ],
        labels=False,
    )

    return widget


def setup_shapes_layer_for_annotation(viewer: napari.Viewer) -> napari.layers.Shapes:
    """
    Create and configure a Shapes layer optimized for annotation.

    Uses napari best practices:
    - Features-based coloring for role distinction
    - Text labels showing shape names
    - Initialized feature_defaults for new shapes

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.

    Returns
    -------
    napari.layers.Shapes
        Configured shapes layer ready for annotation.
    """
    import pandas as pd

    # Initialize with empty features DataFrame (categorical for color cycling)
    features = pd.DataFrame({
        "role": pd.Categorical([], categories=ROLE_CATEGORIES),
        "name": pd.Series([], dtype=str),
    })

    shapes = viewer.add_shapes(
        name="Annotations",
        shape_type="polygon",
        # Features-based coloring
        features=features,
        face_color="role",
        face_color_cycle=ROLE_COLOR_CYCLE,
        edge_color="white",
        edge_width=2,
        # Text labels on shapes
        text={
            "string": "{name}",
            "size": 10,
            "color": "white",
            "anchor": "upper_left",
            "translation": [5, 5],
        },
    )

    # Set defaults for new shapes
    shapes.feature_defaults["role"] = "region"
    shapes.feature_defaults["name"] = "region_0"
    shapes.current_face_color = ROLE_COLORS["region"]

    # Start in polygon drawing mode
    shapes.mode = "add_polygon"

    return shapes


def get_annotation_data(
    shapes_layer: napari.layers.Shapes,
) -> tuple[list, list[str], list[str]]:
    """
    Extract annotation data from shapes layer.

    Parameters
    ----------
    shapes_layer : napari.layers.Shapes
        The shapes layer containing annotations.

    Returns
    -------
    shapes_data : list of NDArray
        List of polygon vertex arrays in napari (row, col) format.
    names : list of str
        Name for each shape.
    roles : list of str
        Role for each shape ("environment" or "region").
    """
    if shapes_layer is None or len(shapes_layer.data) == 0:
        return [], [], []

    data = list(shapes_layer.data)

    # Get from features (preferred) or properties (fallback)
    if hasattr(shapes_layer, "features") and len(shapes_layer.features) > 0:
        features = shapes_layer.features
        names = list(features.get("name", [f"region_{i}" for i in range(len(data))]))
        roles = list(features.get("role", ["region"] * len(data)))
    else:
        props = shapes_layer.properties
        names = list(props.get("name", [f"region_{i}" for i in range(len(data))]))
        roles = list(props.get("role", ["region"] * len(data)))

    # Ensure lists match data length
    while len(names) < len(data):
        names.append(f"region_{len(names)}")
    while len(roles) < len(data):
        roles.append("region")

    return data, [str(n) for n in names], [str(r) for r in roles]
```

### Task 4.2: Verification

```bash
uv run python -c "from neurospatial.annotation._napari_widget import create_annotation_widget, setup_shapes_layer_for_annotation; print('OK')"
```

---

## Milestone 5: Main Entry Point

### Task 5.1: Create `core.py`

**File**: `src/neurospatial/annotation/core.py`

```python
"""Main annotation entry point."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.regions import Regions
    from neurospatial.transforms import VideoCalibration


class AnnotationResult(NamedTuple):
    """
    Result from an annotation session.

    Attributes
    ----------
    environment : Environment or None
        Discretized environment if boundary was annotated, else None.
    regions : Regions
        All annotated regions (excluding environment boundary).
    """

    environment: Environment | None
    regions: Regions


def annotate_video(
    video_path: str | Path,
    *,
    frame_index: int = 0,
    initial_regions: Regions | None = None,
    calibration: VideoCalibration | None = None,
    mode: Literal["environment", "regions", "both"] = "both",
    bin_size: float | None = None,
    simplify_tolerance: float | None = None,
) -> AnnotationResult:
    """
    Launch interactive napari annotation on a video frame.

    Opens a napari viewer with the specified video frame. Users can draw
    polygons to define an environment boundary and/or named regions.
    After closing the viewer, annotations are converted to Regions and
    optionally an Environment.

    Parameters
    ----------
    video_path : str or Path
        Path to video file (any format supported by OpenCV).
    frame_index : int, default=0
        Which frame to display for annotation.
    initial_regions : Regions, optional
        Pre-existing regions to display for editing.
    calibration : VideoCalibration, optional
        Pixel-to-cm transform. If provided, output coordinates are in cm.
        If None, coordinates remain in pixels.
    mode : {"environment", "regions", "both"}, default="both"
        What to annotate:
        - "environment": Only expect environment boundary
        - "regions": Only expect named regions
        - "both": Expect both boundary and regions
    bin_size : float, optional
        Bin size for environment discretization. Required if mode is
        "environment" or "both".
    simplify_tolerance : float, optional
        If provided, simplifies hand-drawn polygons using Douglas-Peucker
        algorithm. Removes jagged edges from freehand drawing. Tolerance
        is in output units (cm if calibration provided, else pixels).
        Recommended: 1.0-2.0 for typical use cases.

    Returns
    -------
    AnnotationResult
        Named tuple containing:
        - environment: Environment or None
        - regions: Regions collection

    Raises
    ------
    ValueError
        If bin_size is not provided when mode requires environment creation.
    ImportError
        If napari is not installed.

    Examples
    --------
    >>> from neurospatial.annotation import annotate_video
    >>> # Simple annotation (pixel coordinates)
    >>> result = annotate_video("experiment.mp4", bin_size=10.0)
    >>> print(result.environment)  # Environment from boundary
    >>> print(result.regions)      # Named regions

    >>> # With calibration (cm coordinates)
    >>> from neurospatial.transforms import VideoCalibration, calibrate_from_scale_bar
    >>> transform = calibrate_from_scale_bar((0, 0), (200, 0), 100.0, (640, 480))
    >>> calib = VideoCalibration(transform, (640, 480))
    >>> result = annotate_video("experiment.mp4", calibration=calib, bin_size=2.0)

    Notes
    -----
    The napari viewer runs in blocking mode. The function returns only after
    the user closes the viewer (via the "Save and Close" button or window close).

    Coordinate Systems
    ^^^^^^^^^^^^^^^^^^
    - Napari shapes: (row, col) with origin at top-left
    - Video pixels: (x, y) with origin at top-left
    - Environment: (x, y) with origin at bottom-left (if calibrated)

    See Also
    --------
    regions_from_labelme : Import from LabelMe JSON
    regions_from_cvat : Import from CVAT XML
    """
    import napari

    from neurospatial.animation._video_io import VideoReader
    from neurospatial.annotation._napari_widget import (
        create_annotation_widget,
        get_annotation_data,
        setup_shapes_layer_for_annotation,
    )
    from neurospatial.annotation.converters import (
        env_from_boundary_region,
        shapes_to_regions,
    )
    from neurospatial.regions import Regions

    # Validate parameters
    if mode in ("environment", "both") and bin_size is None:
        raise ValueError(
            f"bin_size is required when mode={mode!r}. "
            "Provide bin_size for environment discretization."
        )

    # Load video frame
    video_path = Path(video_path)
    reader = VideoReader(str(video_path))
    frame = reader[frame_index]  # (H, W, 3) RGB uint8

    # Create viewer
    viewer = napari.Viewer(title=f"Annotate: {video_path.name}")
    viewer.add_image(frame, name="video_frame", rgb=True)

    # Add shapes layer with annotation-optimized settings
    # (features-based coloring, text labels, keyboard shortcuts)
    shapes = setup_shapes_layer_for_annotation(viewer)

    # Add existing regions if provided
    if initial_regions is not None:
        _add_initial_regions(shapes, initial_regions, calibration)

    # Add annotation control widget (magicgui-based)
    widget = create_annotation_widget(viewer, "Annotations")
    viewer.window.add_dock_widget(
        widget,
        name="Annotation Controls",
        area="right",
    )

    # Run napari (blocking until viewer closes)
    napari.run()

    # Get annotation data from shapes layer
    shapes_data, names, roles = get_annotation_data(shapes)

    # Handle empty annotations
    if not shapes_data:
        return AnnotationResult(environment=None, regions=Regions([]))

    # Convert shapes to regions
    regions, env_boundary = shapes_to_regions(
        shapes_data, names, roles, calibration, simplify_tolerance
    )

    # Build environment if requested and boundary exists
    environment = None
    if mode in ("environment", "both") and env_boundary is not None:
        environment = env_from_boundary_region(env_boundary, bin_size)
        # Attach regions to environment
        for name, region in regions.items():
            environment.regions.add(
                name,
                polygon=region.data,
                metadata=dict(region.metadata),
            )

    return AnnotationResult(environment=environment, regions=regions)


def _add_initial_regions(
    shapes_layer,
    regions: Regions,
    calibration: VideoCalibration | None,
) -> None:
    """
    Add existing regions to shapes layer for editing.

    Parameters
    ----------
    shapes_layer : napari.layers.Shapes
        The shapes layer to add regions to.
    regions : Regions
        Existing regions to display.
    calibration : VideoCalibration or None
        If provided, transforms cm coordinates back to pixels for display.
    """
    import pandas as pd
    from shapely import get_coordinates

    data = []
    names = []
    roles = []

    for name, region in regions.items():
        if region.kind != "polygon":
            continue

        # Get polygon vertices in world coordinates
        coords = get_coordinates(region.data)

        # Transform to pixels if calibration provided
        if calibration is not None:
            coords = calibration.transform_cm_to_px(coords)

        # Convert to napari (row, col) order
        coords_rc = coords[:, ::-1]
        data.append(coords_rc)
        names.append(name)
        roles.append(region.metadata.get("role", "region"))

    if data:
        # Add shapes to layer
        shapes_layer.add(data, shape_type="polygon")

        # Update features DataFrame (consistent with setup_shapes_layer_for_annotation)
        # Import ROLE_CATEGORIES from module level
        from neurospatial.annotation._napari_widget import ROLE_CATEGORIES

        shapes_layer.features = pd.DataFrame({
            "role": pd.Categorical(roles, categories=ROLE_CATEGORIES),
            "name": pd.Series(names, dtype=str),
        })
```

### Task 5.2: Verification

```bash
uv run python -c "from neurospatial.annotation import annotate_video, AnnotationResult; print('OK')"
```

---

## Milestone 6: Tests

### Task 6.1: Create test directory

```bash
mkdir -p tests/annotation
touch tests/annotation/__init__.py
```

**Note:** Tests using napari viewers are marked with `@pytest.mark.gui`. To skip these in headless CI environments, run:

```bash
uv run pytest -m "not gui"
```

Register the marker in `pyproject.toml` to avoid warnings:

```toml
[tool.pytest.ini_options]
markers = [
    "gui: tests requiring display/Qt backend (napari)",
]
```

### Task 6.2: Create `test_converters.py`

**File**: `tests/annotation/test_converters.py`

```python
"""Tests for annotation converters."""

import numpy as np
import pytest
import shapely.geometry as shp

from neurospatial.annotation.converters import (
    env_from_boundary_region,
    shapes_to_regions,
)
from neurospatial.regions import Region
from neurospatial.transforms import Affine2D, VideoCalibration


class TestShapesToRegions:
    """Tests for shapes_to_regions function."""

    def test_basic_conversion(self):
        """Convert napari shapes to regions without calibration."""
        # Napari format: (row, col) order
        shapes_data = [
            np.array([[0, 0], [0, 100], [100, 100], [100, 0]], dtype=float),
        ]
        names = ["test_region"]
        roles = ["region"]

        regions, env_boundary = shapes_to_regions(shapes_data, names, roles)

        assert len(regions) == 1
        assert "test_region" in regions
        assert env_boundary is None
        # Verify coordinates swapped: (row, col) -> (x, y)
        assert regions["test_region"].kind == "polygon"

    def test_environment_boundary_extraction(self):
        """Extract environment boundary from shapes."""
        shapes_data = [
            np.array([[0, 0], [0, 100], [100, 100], [100, 0]], dtype=float),
            np.array([[10, 10], [10, 20], [20, 20], [20, 10]], dtype=float),
        ]
        names = ["arena", "reward_zone"]
        roles = ["environment", "region"]

        regions, env_boundary = shapes_to_regions(shapes_data, names, roles)

        assert len(regions) == 1
        assert "reward_zone" in regions
        assert env_boundary is not None
        assert env_boundary.name == "arena"

    def test_with_calibration(self):
        """Apply calibration transform to coordinates."""
        # Simple 2x scale transform
        scale_matrix = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        transform = Affine2D(scale_matrix)
        calibration = VideoCalibration(transform, frame_size_px=(640, 480))

        # Square in napari (row, col): corners at (0,0), (0,10), (10,10), (10,0)
        # After swap to (x, y): (0,0), (10,0), (10,10), (0,10)
        # After 2x scale: (0,0), (20,0), (20,20), (0,20)
        shapes_data = [
            np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=float),
        ]
        names = ["scaled_region"]
        roles = ["region"]

        regions, _ = shapes_to_regions(shapes_data, names, roles, calibration)

        assert len(regions) == 1
        poly = regions["scaled_region"].data
        bounds = poly.bounds  # (minx, miny, maxx, maxy)
        assert bounds[2] == pytest.approx(20.0)  # maxx
        assert bounds[3] == pytest.approx(20.0)  # maxy

    def test_skip_invalid_polygons(self):
        """Skip shapes with fewer than 3 vertices."""
        shapes_data = [
            np.array([[0, 0], [10, 10]], dtype=float),  # Line, not polygon
            np.array([[0, 0], [0, 100], [100, 100]], dtype=float),  # Valid
        ]
        names = ["line", "triangle"]
        roles = ["region", "region"]

        regions, _ = shapes_to_regions(shapes_data, names, roles)

        assert len(regions) == 1
        assert "triangle" in regions

    def test_metadata_populated(self):
        """Check metadata is properly set."""
        shapes_data = [
            np.array([[0, 0], [0, 100], [100, 100], [100, 0]], dtype=float),
        ]
        names = ["test"]
        roles = ["region"]

        regions, _ = shapes_to_regions(shapes_data, names, roles)

        metadata = regions["test"].metadata
        assert metadata["source"] == "napari_annotation"
        assert metadata["coord_system"] == "pixels"
        assert metadata["role"] == "region"

    def test_simplify_tolerance(self):
        """Simplify polygon with tolerance parameter."""
        # Create a polygon with many redundant vertices on a line
        # Square with extra points along edges
        vertices = np.array([
            [0, 0], [0, 25], [0, 50], [0, 75], [0, 100],  # Left edge
            [25, 100], [50, 100], [75, 100], [100, 100],  # Top edge
            [100, 75], [100, 50], [100, 25], [100, 0],    # Right edge
            [75, 0], [50, 0], [25, 0],                     # Bottom edge
        ], dtype=float)
        shapes_data = [vertices]
        names = ["detailed"]
        roles = ["region"]

        # Without simplification
        regions_full, _ = shapes_to_regions(shapes_data, names, roles)
        poly_full = regions_full["detailed"].data
        n_coords_full = len(poly_full.exterior.coords)

        # With simplification (tolerance=5.0 should remove colinear points)
        regions_simple, _ = shapes_to_regions(
            shapes_data, names, roles, simplify_tolerance=5.0
        )
        poly_simple = regions_simple["detailed"].data
        n_coords_simple = len(poly_simple.exterior.coords)

        # Simplified should have fewer vertices (just 4 corners + closing point)
        assert n_coords_simple < n_coords_full
        assert n_coords_simple == 5  # 4 corners + closing point


class TestEnvFromBoundaryRegion:
    """Tests for env_from_boundary_region function."""

    def test_basic_environment_creation(self):
        """Create environment from polygon boundary."""
        poly = shp.Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        boundary = Region(name="arena", kind="polygon", data=poly)

        env = env_from_boundary_region(boundary, bin_size=10.0)

        assert env._is_fitted
        assert env.n_bins > 0

    def test_rejects_non_polygon(self):
        """Raise error for non-polygon regions."""
        point = Region(name="point", kind="point", data=np.array([50.0, 50.0]))

        with pytest.raises(ValueError, match="must be polygon"):
            env_from_boundary_region(point, bin_size=10.0)

    def test_passes_kwargs(self):
        """Forward kwargs to Environment.from_polygon."""
        poly = shp.Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        boundary = Region(name="arena", kind="polygon", data=poly)

        env = env_from_boundary_region(
            boundary,
            bin_size=10.0,
            infer_active_bins=True,
        )

        assert env._is_fitted
```

### Task 6.3: Create `test_io.py`

**File**: `tests/annotation/test_io.py`

```python
"""Tests for annotation IO functions."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from neurospatial.annotation.io import regions_from_cvat, regions_from_labelme
from neurospatial.transforms import Affine2D, VideoCalibration


class TestRegionsFromLabelme:
    """Tests for regions_from_labelme function."""

    def test_basic_import(self, tmp_path):
        """Import LabelMe JSON without calibration."""
        json_data = {
            "shapes": [
                {
                    "label": "arena",
                    "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
                },
                {
                    "label": "reward",
                    "points": [[10, 10], [20, 10], [20, 20], [10, 20]],
                },
            ]
        }
        json_path = tmp_path / "annotations.json"
        json_path.write_text(json.dumps(json_data))

        regions = regions_from_labelme(json_path)

        assert len(regions) == 2
        assert "arena" in regions
        assert "reward" in regions

    def test_with_calibration(self, tmp_path):
        """Apply calibration during import."""
        # 2x scale calibration
        scale_matrix = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        transform = Affine2D(scale_matrix)
        calibration = VideoCalibration(transform, frame_size_px=(640, 480))

        json_data = {
            "shapes": [
                {
                    "label": "test",
                    "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                },
            ]
        }
        json_path = tmp_path / "annotations.json"
        json_path.write_text(json.dumps(json_data))

        regions = regions_from_labelme(json_path, calibration=calibration)

        poly = regions["test"].data
        bounds = poly.bounds
        assert bounds[2] == pytest.approx(20.0)  # maxx scaled

    def test_custom_keys(self, tmp_path):
        """Use custom label and points keys."""
        json_data = {
            "shapes": [
                {
                    "name": "custom_name",
                    "coords": [[0, 0], [50, 0], [50, 50], [0, 50]],
                },
            ]
        }
        json_path = tmp_path / "custom.json"
        json_path.write_text(json.dumps(json_data))

        regions = regions_from_labelme(
            json_path,
            label_key="name",
            points_key="coords",
        )

        assert "custom_name" in regions


class TestRegionsFromCvat:
    """Tests for regions_from_cvat function."""

    def test_basic_import(self, tmp_path):
        """Import CVAT XML without calibration."""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
        <annotations>
            <image id="0" name="frame_0.png" width="640" height="480">
                <polygon label="arena" points="0,0;100,0;100,100;0,100"/>
                <polygon label="reward" points="10,10;20,10;20,20;10,20"/>
            </image>
        </annotations>
        """
        xml_path = tmp_path / "annotations.xml"
        xml_path.write_text(xml_content)

        regions = regions_from_cvat(xml_path)

        assert len(regions) == 2
        assert "arena" in regions
        assert "reward" in regions

    def test_with_calibration(self, tmp_path):
        """Apply calibration during CVAT import."""
        scale_matrix = np.array([
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 1.0],
        ])
        transform = Affine2D(scale_matrix)
        calibration = VideoCalibration(transform, frame_size_px=(640, 480))

        xml_content = """<?xml version="1.0" encoding="utf-8"?>
        <annotations>
            <image id="0" name="frame.png" width="640" height="480">
                <polygon label="scaled" points="0,0;100,0;100,100;0,100"/>
            </image>
        </annotations>
        """
        xml_path = tmp_path / "annotations.xml"
        xml_path.write_text(xml_content)

        regions = regions_from_cvat(xml_path, calibration=calibration)

        poly = regions["scaled"].data
        bounds = poly.bounds
        assert bounds[2] == pytest.approx(50.0)  # maxx at 0.5x scale
```

### Task 6.4: Create `test_core.py`

**File**: `tests/annotation/test_core.py`

```python
"""Tests for annotation core functions."""

import numpy as np
import pytest

from neurospatial.annotation.core import AnnotationResult, _add_initial_regions
from neurospatial.regions import Region, Regions
from neurospatial.transforms import Affine2D, VideoCalibration
import shapely.geometry as shp


class TestAnnotationResult:
    """Tests for AnnotationResult named tuple."""

    def test_creation(self):
        """Create AnnotationResult with environment and regions."""
        from neurospatial import Environment

        regions = Regions([])
        result = AnnotationResult(environment=None, regions=regions)

        assert result.environment is None
        assert isinstance(result.regions, Regions)

    def test_unpacking(self):
        """Unpack AnnotationResult as tuple."""
        regions = Regions([])
        result = AnnotationResult(environment=None, regions=regions)

        env, regs = result

        assert env is None
        assert regs is regions


@pytest.mark.gui  # Skip in headless CI with: pytest -m "not gui"
class TestAddInitialRegions:
    """Tests for _add_initial_regions helper."""

    def test_adds_polygon_regions(self):
        """Add existing polygon regions to shapes layer."""
        pytest.importorskip("napari")
        import napari

        # Create regions
        poly = shp.Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        region = Region(
            name="test",
            kind="polygon",
            data=poly,
            metadata={"role": "region"},
        )
        regions = Regions([region])

        # Create shapes layer (no initial features needed - _add_initial_regions sets them)
        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(name="Test")

        _add_initial_regions(shapes, regions, calibration=None)

        assert len(shapes.data) == 1
        # Check features DataFrame (not properties)
        assert shapes.features["name"].iloc[0] == "test"
        assert shapes.features["role"].iloc[0] == "region"
        viewer.close()

    def test_skips_point_regions(self):
        """Skip non-polygon regions."""
        pytest.importorskip("napari")
        import napari

        point_region = Region(
            name="point",
            kind="point",
            data=np.array([50.0, 50.0]),
        )
        regions = Regions([point_region])

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(name="Test")

        _add_initial_regions(shapes, regions, calibration=None)

        assert len(shapes.data) == 0
        viewer.close()

    def test_transforms_with_calibration(self):
        """Transform coordinates back to pixels when calibration provided."""
        pytest.importorskip("napari")
        import napari

        # 2x scale calibration (cm -> pixels means inverse)
        scale_matrix = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        transform = Affine2D(scale_matrix)
        calibration = VideoCalibration(transform, frame_size_px=(640, 480))

        # Region in "cm" (world coords after 2x scale)
        poly = shp.Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
        region = Region(name="scaled", kind="polygon", data=poly)
        regions = Regions([region])

        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(name="Test")

        _add_initial_regions(shapes, regions, calibration)

        # Coordinates should be back in pixels (0.5x scale = inverse of 2x)
        coords = shapes.data[0]
        # After cm_to_px (0.5x) and row/col swap, (20, 20) cm -> (10, 10) px -> (10, 10) napari
        assert coords.max() == pytest.approx(10.0)
        viewer.close()
```

### Task 6.5: Run tests

```bash
uv run pytest tests/annotation/ -v
```

---

## Milestone 7: Documentation

### Task 7.1: Update CLAUDE.md Quick Reference

Add to Quick Reference section:

```python
# Annotate video frames interactively (v0.6.0+)
from neurospatial import annotate_video, regions_from_labelme, regions_from_cvat

# Interactive napari annotation
result = annotate_video("experiment.mp4", bin_size=2.0)
env = result.environment  # Environment from boundary polygon
regions = result.regions   # Named regions

# With calibration (pixel → cm)
from neurospatial.transforms import VideoCalibration, calibrate_from_scale_bar
transform = calibrate_from_scale_bar((0, 0), (200, 0), 100.0, (640, 480))
calib = VideoCalibration(transform, (640, 480))
result = annotate_video("experiment.mp4", calibration=calib, bin_size=2.0)

# Import from external tools
regions = regions_from_labelme("labelme_export.json", calibration=calib)
regions = regions_from_cvat("cvat_export.xml", calibration=calib)
```

---

## Implementation Checklist

- [ ] Create `src/neurospatial/annotation/__init__.py`
- [ ] Create `src/neurospatial/annotation/converters.py`
- [ ] Create `src/neurospatial/annotation/io.py`
- [ ] Create `src/neurospatial/annotation/_napari_widget.py`
- [ ] Create `src/neurospatial/annotation/core.py`
- [ ] Update `src/neurospatial/__init__.py` with new exports
- [ ] Create `tests/annotation/__init__.py`
- [ ] Create `tests/annotation/test_converters.py`
- [ ] Create `tests/annotation/test_io.py`
- [ ] Create `tests/annotation/test_core.py`
- [ ] Run all tests: `uv run pytest tests/annotation/ -v`
- [ ] Run full test suite: `uv run pytest`
- [ ] Run linting: `uv run ruff check . && uv run ruff format .`
- [ ] Update CLAUDE.md documentation
