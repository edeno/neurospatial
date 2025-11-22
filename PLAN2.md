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
| `VideoCalibration` | `transforms.py:702-816` | Pixel↔cm transforms via `transform_px_to_cm` |
| `Region`/`Regions` | `regions/core.py` | Annotation output data structures |
| `load_labelme_json` | `regions/io.py:83` | Already accepts `pixel_to_world: SpatialTransform` |
| `load_cvat_xml` | `regions/io.py:785` | Already accepts `pixel_to_world: SpatialTransform` |
| `VideoReader` | `animation/_video_io.py:41-309` | Load video frames |
| `Environment.from_polygon` | `environment/factories.py:399` | Create env from boundary |
| `SpatialTransform` | `transforms.py:44-47` | Protocol for coordinate transforms |

### Coordinate Systems

**Video Pixels**: Origin top-left, (x, y) = (column, row)
**Napari Shapes**: Origin top-left, (row, col) order
**Environment**: Origin bottom-left, (x, y) with Y increasing upward

Transform pipeline:
```
napari (row, col) → video (x, y) pixels → calibration → world (x, y) cm
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

### Task 4.1: Create `_napari_widget.py`

**File**: `src/neurospatial/annotation/_napari_widget.py`

```python
"""Qt dock widget for napari annotation workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari


class AnnotationWidget(QWidget):
    """
    Dock widget for managing annotation shapes.

    Provides UI for:
    - Viewing all drawn shapes in a table
    - Editing shape names
    - Assigning roles (environment vs region)
    - Saving and closing the viewer

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    shapes_layer_name : str, default="Annotations"
        Name of the Shapes layer to manage.
    """

    def __init__(
        self,
        viewer: napari.Viewer,
        shapes_layer_name: str = "Annotations",
    ) -> None:
        super().__init__()
        self._viewer = viewer
        self._shapes_name = shapes_layer_name
        self._setup_ui()
        self._connect_signals()
        self._refresh_table()

    def _setup_ui(self) -> None:
        """Build the widget UI."""
        layout = QVBoxLayout()

        # Instructions
        instructions = QLabel(
            "Draw polygons on the image.\n"
            "Select rows and assign roles.\n"
            "Double-click names to edit."
        )
        layout.addWidget(instructions)

        # Shape table
        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Name", "Role"])
        self._table.setColumnWidth(0, 150)
        self._table.setColumnWidth(1, 100)
        layout.addWidget(self._table)

        # Role assignment buttons
        btn_layout = QHBoxLayout()
        self._env_btn = QPushButton("Set as Environment")
        self._region_btn = QPushButton("Set as Region")
        self._env_btn.setToolTip("Mark selected shape(s) as environment boundary")
        self._region_btn.setToolTip("Mark selected shape(s) as named region")
        btn_layout.addWidget(self._env_btn)
        btn_layout.addWidget(self._region_btn)
        layout.addLayout(btn_layout)

        # Save button
        self._save_btn = QPushButton("Save and Close")
        self._save_btn.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._save_btn)

        self.setLayout(layout)

    def _connect_signals(self) -> None:
        """Connect Qt signals to handlers."""
        self._env_btn.clicked.connect(lambda: self._set_selected_role("environment"))
        self._region_btn.clicked.connect(lambda: self._set_selected_role("region"))
        self._save_btn.clicked.connect(self._on_save)
        self._table.cellChanged.connect(self._on_cell_changed)

        # Listen for shape changes
        shapes = self._get_shapes_layer()
        if shapes is not None:
            shapes.events.data.connect(self._refresh_table)

    def _get_shapes_layer(self):
        """Get the Shapes layer by name."""
        try:
            return self._viewer.layers[self._shapes_name]
        except KeyError:
            return None

    def _refresh_table(self, event=None) -> None:
        """Update table when shapes change."""
        shapes = self._get_shapes_layer()
        if shapes is None:
            return

        # Block signals during update to avoid recursive calls
        self._table.blockSignals(True)

        n_shapes = len(shapes.data)
        self._table.setRowCount(n_shapes)

        # Get or initialize properties
        props = shapes.properties
        names = props.get("name", np.array([], dtype=object))
        roles = props.get("role", np.array([], dtype=object))

        # Extend arrays if needed (new shapes added)
        if len(names) < n_shapes:
            new_names = [f"region_{i}" for i in range(len(names), n_shapes)]
            names = np.concatenate([names, np.array(new_names, dtype=object)])
            shapes.properties["name"] = names

        if len(roles) < n_shapes:
            new_roles = ["region"] * (n_shapes - len(roles))
            roles = np.concatenate([roles, np.array(new_roles, dtype=object)])
            shapes.properties["role"] = roles

        # Populate table
        for i in range(n_shapes):
            name = str(names[i]) if i < len(names) else f"region_{i}"
            role = str(roles[i]) if i < len(roles) else "region"

            name_item = QTableWidgetItem(name)
            role_item = QTableWidgetItem(role)
            role_item.setFlags(role_item.flags() & ~0x2)  # Not editable

            self._table.setItem(i, 0, name_item)
            self._table.setItem(i, 1, role_item)

        self._table.blockSignals(False)

    def _on_cell_changed(self, row: int, col: int) -> None:
        """Handle cell edits (name changes)."""
        if col != 0:  # Only name column is editable
            return

        shapes = self._get_shapes_layer()
        if shapes is None:
            return

        item = self._table.item(row, 0)
        if item is None:
            return

        new_name = item.text()
        names = list(shapes.properties.get("name", []))
        if row < len(names):
            names[row] = new_name
            shapes.properties["name"] = np.array(names, dtype=object)

    def _set_selected_role(self, role: str) -> None:
        """Set role for selected table rows."""
        shapes = self._get_shapes_layer()
        if shapes is None:
            return

        selected_rows = set(item.row() for item in self._table.selectedItems())
        if not selected_rows:
            return

        # Update properties
        roles = list(shapes.properties.get("role", ["region"] * len(shapes.data)))
        for row in selected_rows:
            if row < len(roles):
                roles[row] = role
                role_item = self._table.item(row, 1)
                if role_item:
                    role_item.setText(role)

        shapes.properties["role"] = np.array(roles, dtype=object)

    def _on_save(self) -> None:
        """Finalize properties and close viewer."""
        shapes = self._get_shapes_layer()
        if shapes is not None:
            # Ensure all names are saved from table
            names = []
            roles = []
            for i in range(self._table.rowCount()):
                name_item = self._table.item(i, 0)
                role_item = self._table.item(i, 1)
                names.append(name_item.text() if name_item else f"region_{i}")
                roles.append(role_item.text() if role_item else "region")

            shapes.properties["name"] = np.array(names, dtype=object)
            shapes.properties["role"] = np.array(roles, dtype=object)

        self._viewer.close()

    def get_annotation_data(self) -> tuple[list, list[str], list[str]]:
        """
        Get current annotation data from shapes layer.

        Returns
        -------
        shapes_data : list
            List of polygon vertex arrays.
        names : list of str
            Name for each shape.
        roles : list of str
            Role for each shape.
        """
        shapes = self._get_shapes_layer()
        if shapes is None or len(shapes.data) == 0:
            return [], [], []

        data = list(shapes.data)
        names = list(shapes.properties.get("name", []))
        roles = list(shapes.properties.get("role", []))

        # Ensure lists match data length
        while len(names) < len(data):
            names.append(f"region_{len(names)}")
        while len(roles) < len(data):
            roles.append("region")

        return data, names, roles
```

### Task 4.2: Verification

```bash
uv run python -c "from neurospatial.annotation._napari_widget import AnnotationWidget; print('OK')"
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
    from neurospatial.annotation._napari_widget import AnnotationWidget
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

    # Add shapes layer for annotations
    shapes = viewer.add_shapes(
        name="Annotations",
        shape_type="polygon",
        edge_color="yellow",
        face_color=[1, 1, 0, 0.2],
        edge_width=2,
    )
    shapes.mode = "add_polygon"

    # Initialize properties
    shapes.properties = {
        "name": np.array([], dtype=object),
        "role": np.array([], dtype=object),
    }

    # Add existing regions if provided
    if initial_regions is not None:
        _add_initial_regions(shapes, initial_regions, calibration)

    # Add annotation widget
    widget = AnnotationWidget(viewer, "Annotations")
    viewer.window.add_dock_widget(
        widget,
        name="Annotation Controls",
        area="right",
    )

    # Run napari (blocking until viewer closes)
    napari.run()

    # Get annotation data from widget
    shapes_data, names, roles = widget.get_annotation_data()

    # Handle empty annotations
    if not shapes_data:
        return AnnotationResult(environment=None, regions=Regions([]))

    # Convert shapes to regions
    regions, env_boundary = shapes_to_regions(
        shapes_data, names, roles, calibration
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
        shapes_layer.add(data, shape_type="polygon")
        shapes_layer.properties["name"] = np.array(names, dtype=object)
        shapes_layer.properties["role"] = np.array(roles, dtype=object)
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

        # Create mock shapes layer
        viewer = napari.Viewer(show=False)
        shapes = viewer.add_shapes(name="Test")
        shapes.properties = {"name": np.array([], dtype=object), "role": np.array([], dtype=object)}

        _add_initial_regions(shapes, regions, calibration=None)

        assert len(shapes.data) == 1
        assert shapes.properties["name"][0] == "test"
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
        shapes.properties = {"name": np.array([], dtype=object), "role": np.array([], dtype=object)}

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
        shapes.properties = {"name": np.array([], dtype=object), "role": np.array([], dtype=object)}

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
