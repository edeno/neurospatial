"""Optional napari rectangles for an inference crop and a box prompt.

The optional inference ROI and the box prompt were inspired by CVAT
(https://github.com/cvat-ai/cvat). See
docs/user-guide/video-annotation.md#design-credit-and-references for the
pinned upstream workflow and implementation references.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import numpy as np
from napari.layers import Shapes
from numpy.typing import NDArray
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from neurospatial.annotation._roi import (
    ROIBounds,
    rectangle_to_roi_bounds,
)

if TYPE_CHECKING:
    import napari


class _RectangleControl(QWidget):
    """Draw one optional rectangle without changing annotation coordinates.

    The dedicated rectangle uses pixel ``(row, column)`` coordinates. Clicking
    Draw starts a new rectangle and immediately invalidates previous inference.
    An unfinished or removed active rectangle is an error until the user redraws
    it or explicitly clears it.
    """

    def __init__(
        self,
        viewer: napari.Viewer,
        frame_shape: Sequence[int],
        *,
        on_change: Callable[[], None],
        name: str,
        color: tuple[float, float, float],
        noun: str,
        help_text: str,
        draw_label: str,
        clear_label: str,
        idle_status: str,
        compact: bool = False,
    ) -> None:
        super().__init__()
        self._viewer = viewer
        self._frame_shape = tuple(frame_shape[:2])
        self._on_change = on_change
        self._noun = noun
        self._clear_label = clear_label
        self._idle_status = idle_status
        self._active = False
        self._geometry_pending = False
        self._disposed = False
        self.layer = Shapes(
            ndim=2,
            name=name,
            shape_type="rectangle",
            edge_color=[*color, 1.0],
            face_color=[*color, 0.08],
            edge_width=2,
        )
        self.layer.events.data.connect(self._region_changed)
        viewer.layers.events.removed.connect(self._layer_removed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.draw_button = QPushButton(draw_label)
        self.draw_button.clicked.connect(self._draw_region)
        self.clear_button = QPushButton(clear_label)
        self.clear_button.clicked.connect(self.clear)
        buttons = QHBoxLayout()
        buttons.addWidget(self.draw_button)
        buttons.addWidget(self.clear_button)
        self.status = QLabel(idle_status)
        self.status.setWordWrap(True)
        if compact:
            # A compact control sits in another panel's button row and reports
            # problems through that panel's status line.
            self.draw_button.setToolTip(help_text)
            self.status.hide()
        else:
            help_label = QLabel(help_text)
            help_label.setWordWrap(True)
            layout.addWidget(help_label)
        layout.addLayout(buttons)
        layout.addWidget(self.status)

    @property
    def rectangle(self) -> NDArray[np.float64] | None:
        """Return the drawn ``(row, column)`` corners, or None when cleared."""
        if not self._active:
            return None
        if self._geometry_pending:
            raise ValueError(
                f"Finish drawing or editing the {self._noun}, "
                f"or choose {self._clear_label}."
            )
        if self.layer not in self._viewer.layers or len(self.layer.data) != 1:
            raise ValueError(
                f"Draw one {self._noun} rectangle, or choose {self._clear_label}."
            )
        return np.asarray(self.layer.data[0], dtype=np.float64)

    def _describe(self) -> str:
        """Summarize the valid rectangle for the status line."""
        raise NotImplementedError

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
        self.status.setText(f"Draw the {self._noun} rectangle.")

    def clear(self) -> None:
        """Remove the rectangle and notify the owner."""
        if self._disposed:
            return
        self._active = False
        self._geometry_pending = False
        self._on_change()
        with self.layer.events.data.blocker():
            self.layer.data = []
        self.layer.mode = "pan_zoom"
        self.layer.visible = False
        self.status.setText(self._idle_status)

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
            description = self._describe()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.status.setText(description)
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


class AnnotationROIControl(_RectangleControl):
    """Select an optional inference crop; pixels keep their original size."""

    def __init__(
        self,
        viewer: napari.Viewer,
        frame_shape: Sequence[int],
        *,
        on_change: Callable[[], None],
    ) -> None:
        super().__init__(
            viewer,
            frame_shape,
            on_change=on_change,
            name="Inference region of interest",
            color=(1.0, 1.0, 0.0),
            noun="region of interest",
            help_text=(
                "Optional: limit inference to a rectangle around the arena. "
                "Pixels keep their original size. Keep every prompt within "
                "the rectangle."
            ),
            draw_label="Draw region of interest",
            clear_label="Use full image",
            idle_status="Using the full image.",
        )

    @property
    def bounds(self) -> ROIBounds | None:
        """Return valid clipped half-open bounds, or None for explicit full image."""
        rectangle = self.rectangle
        if rectangle is None:
            return None
        return rectangle_to_roi_bounds(rectangle, self._frame_shape)

    def _describe(self) -> str:
        return f"Inference pixel bounds (x0, y0, x1, y1): {self.bounds}."


class BoxPromptControl(_RectangleControl):
    """Draw an optional box around the object, used as a segmentation prompt."""

    def __init__(
        self,
        viewer: napari.Viewer,
        frame_shape: Sequence[int],
        *,
        on_change: Callable[[], None],
    ) -> None:
        super().__init__(
            viewer,
            frame_shape,
            on_change=on_change,
            name="Box prompt",
            color=(0.0, 1.0, 1.0),
            noun="box",
            help_text=(
                "Draw a box closely around the area to outline. SAM 2 outlines "
                "the object in the box; region growing stays inside it. Add "
                "inside and outside points to refine."
            ),
            draw_label="Draw box",
            clear_label="Clear box",
            idle_status="No box.",
            compact=True,
        )

    @property
    def box(self) -> NDArray[np.float64] | None:
        """Return ``(x_min, y_min, x_max, y_max)`` within the image, or None."""
        rectangle = self.rectangle
        if rectangle is None:
            return None
        height, width = self._frame_shape
        # Integer coordinates are pixel centers: the image covers [-0.5, size - 0.5].
        rows = np.clip(rectangle[:, 0], -0.5, height - 0.5)
        columns = np.clip(rectangle[:, 1], -0.5, width - 0.5)
        box = np.array([columns.min(), rows.min(), columns.max(), rows.max()])
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Draw the box over the image, with positive size.")
        return box

    def _describe(self) -> str:
        box = self.box
        assert box is not None
        x0, y0, x1, y1 = box
        return f"Box (x0, y0, x1, y1): ({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}) px."
