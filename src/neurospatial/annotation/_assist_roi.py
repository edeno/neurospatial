"""Optional, isolated napari control for selecting an inference crop.

The optional inference ROI was inspired by CVAT (https://github.com/cvat-ai/cvat).
See docs/user-guide/video-annotation.md#design-credit-and-references for the
pinned upstream workflow and implementation references.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from napari.layers import Shapes
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from neurospatial.annotation._roi import (
    ROIBounds,
    rectangle_to_roi_bounds,
)

if TYPE_CHECKING:
    import napari


class AnnotationROIControl(QWidget):
    """Select an optional crop without changing annotation coordinates.

    The dedicated rectangle uses pixel ``(row, column)`` coordinates. Clicking
    Draw starts a new selection and immediately invalidates previous inference.
    An unfinished or removed active rectangle is an error until the user redraws
    it or explicitly chooses Use full image.
    """

    def __init__(
        self,
        viewer: napari.Viewer,
        frame_shape: Sequence[int],
        *,
        on_change: Callable[[], None],
    ) -> None:
        super().__init__()
        self._viewer = viewer
        self._frame_shape = tuple(frame_shape[:2])
        self._on_change = on_change
        self._active = False
        self._geometry_pending = False
        self._disposed = False
        self.layer = Shapes(
            ndim=2,
            name="Inference region of interest",
            shape_type="rectangle",
            edge_color="yellow",
            face_color=[1.0, 1.0, 0.0, 0.08],
            edge_width=2,
        )
        self.layer.events.data.connect(self._region_changed)
        viewer.layers.events.removed.connect(self._layer_removed)

        layout = QVBoxLayout(self)
        help_text = QLabel(
            "Optional: limit inference to a rectangle around the arena. "
            "Pixels keep their original size. Keep every inside and outside "
            "point within the rectangle."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.draw_button = QPushButton("Draw region of interest")
        self.draw_button.clicked.connect(self._draw_region)
        layout.addWidget(self.draw_button)
        self.full_image_button = QPushButton("Use full image")
        self.full_image_button.clicked.connect(self._use_full_image)
        layout.addWidget(self.full_image_button)
        self.status = QLabel("Using the full image.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    @property
    def bounds(self) -> ROIBounds | None:
        """Return valid clipped half-open bounds, or None for explicit full image."""
        if not self._active:
            return None
        if self._geometry_pending:
            raise ValueError(
                "Finish drawing or editing the region of interest, "
                "or choose Use full image."
            )
        if self.layer not in self._viewer.layers or len(self.layer.data) != 1:
            raise ValueError(
                "Draw one region-of-interest rectangle, or choose Use full image."
            )
        return rectangle_to_roi_bounds(self.layer.data[0], self._frame_shape)

    def _draw_region(self) -> None:
        if self._disposed:
            return
        self._active = True
        self._geometry_pending = True
        # Invalidate before changing layers: a draw may emit several data events.
        self._on_change()
        with self.layer.events.data.blocker():
            self.layer.data = []
        if self.layer not in self._viewer.layers:
            self._viewer.add_layer(self.layer)
        self.layer.visible = True
        self._viewer.layers.selection.active = self.layer
        self.layer.mode = "add_rectangle"
        self.status.setText("Draw a rectangle around the inference region.")

    def _use_full_image(self) -> None:
        if self._disposed:
            return
        self._active = False
        self._geometry_pending = False
        self._on_change()
        with self.layer.events.data.blocker():
            self.layer.data = []
        self.layer.mode = "pan_zoom"
        self.layer.visible = False
        self.status.setText("Using the full image.")

    def _region_changed(self, event: Any = None) -> None:
        if self._disposed or not self._active:
            return
        action = getattr(event, "action", None)
        if action in ("adding", "changing", "removing"):
            self._geometry_pending = True
        elif action in ("added", "changed", "removed"):
            self._geometry_pending = False
        self._on_change()
        try:
            bounds = self.bounds
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.status.setText(f"Inference pixel bounds (x0, y0, x1, y1): {bounds}.")
        # Change tools only as this layer finishes a draw. A later edit must not
        # steal focus from the scientist's currently selected annotation tool.
        if (
            action == "added"
            and self._viewer.layers.selection.active is self.layer
            and self.layer.mode == "add_rectangle"
        ):
            self.layer.mode = "select"

    def _layer_removed(self, event: Any) -> None:
        if getattr(event, "value", None) is self.layer:
            self._region_changed()

    def dispose(self) -> None:
        """Disconnect callbacks and remove this control's temporary layer."""
        if self._disposed:
            return
        self._disposed = True
        with suppress(ValueError, RuntimeError):
            self.layer.events.data.disconnect(self._region_changed)
        with suppress(ValueError, RuntimeError):
            self._viewer.layers.events.removed.disconnect(self._layer_removed)
        if self.layer in self._viewer.layers:
            self._viewer.layers.remove(self.layer)
