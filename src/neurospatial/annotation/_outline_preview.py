"""Editable masks and the exact polygon paths offered for acceptance.

The interaction design was inspired by CVAT (https://github.com/cvat-ai/cvat).
Pinned upstream references and design differences are recorded in
docs/user-guide/video-annotation.md#design-credit-and-references.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from napari.layers import Labels, Points, Shapes
from napari.utils.colormaps import DirectLabelColormap
from qtpy.QtCore import Qt, QTimer  # type: ignore[attr-defined]
from qtpy.QtWidgets import QLabel, QSlider, QVBoxLayout, QWidget

from neurospatial.annotation._cvat_polygon import (
    CvatContour,
    cvat_contour,
    simplify_cvat_contour,
)
from neurospatial.annotation._polygon_outline import (
    PolygonOutline,
    polygon_outline,
    simplify_polygon_outline,
)
from neurospatial.annotation.segmentation import SegmentationProposal

if TYPE_CHECKING:
    import napari


OutlineStyle = Literal["preserve", "cvat"]


class OutlinePreviewControl(QWidget):
    """Keep painted foreground authoritative while previewing simpler polygons."""

    def __init__(self, viewer: napari.Viewer, *, on_change: Callable[[], None]) -> None:
        super().__init__()
        self._viewer = viewer
        self._on_change = on_change
        self._disposed = False
        self._updating = False
        self._proposal: SegmentationProposal | None = None
        self._exact: PolygonOutline | None = None
        self._cvat_contour: CvatContour | None = None
        self._style: OutlineStyle = "preserve"
        self._detail_values: dict[OutlineStyle, int] = {"preserve": 14, "cvat": 9}
        self._outline: PolygonOutline | None = None
        self.mask_layer: Labels | None = None
        self.polygon_layer: Shapes | None = None
        # The enclosing napari dock may be deleted before dispose().
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._refresh)
        viewer.layers.events.removed.connect(self._layer_removed)
        layout = QVBoxLayout(self)
        self.detail_label = QLabel("Polygon detail")
        layout.addWidget(self.detail_label)
        self.detail_slider = QSlider(Qt.Orientation.Horizontal)
        self.detail_slider.setRange(0, 100)
        # CVAT accuracy 9 maps to 10/7 px; round to our 0.1 px slider step.
        # See the guide's pinned settings-reducer and OpenCV references above.
        self.detail_slider.setValue(14)
        self.detail_slider.setEnabled(False)
        self.detail_slider.setToolTip(
            "Simplification distance in pixels (default: 1.4): "
            "0 keeps the exact mask boundary; "
            "larger values reduce vertices while keeping components and holes."
        )
        self.detail_slider.valueChanged.connect(self._detail_changed)
        layout.addWidget(self.detail_slider)
        self.detail_status = QLabel("Suggest an outline to adjust polygon detail.")
        self.detail_status.setWordWrap(True)
        layout.addWidget(self.detail_status)

    @property
    def outline(self) -> PolygonOutline | None:
        """Return the valid displayed geometry, or None while edits are pending."""
        return self._outline

    def set_style(self, style: OutlineStyle) -> None:
        """Select a conversion policy and restore its separate detail setting."""
        if style not in ("preserve", "cvat"):
            raise ValueError("Outline style must be 'preserve' or 'cvat'.")
        if self._disposed or style == self._style:
            return
        self._detail_values[self._style] = self.detail_slider.value()
        self.invalidate()
        self._style = style
        self.detail_slider.blockSignals(True)
        try:
            self.detail_slider.setRange(0, 13 if style == "cvat" else 100)
            self.detail_slider.setValue(self._detail_values[style])
        finally:
            self.detail_slider.blockSignals(False)
        self.detail_label.setText(
            "Polygon accuracy" if style == "cvat" else "Polygon detail"
        )
        self.detail_slider.setToolTip(
            "CVAT accuracy: 0 gives fewer vertices, 13 retains contour detail. "
            "Default 9 uses about 1.43 px tolerance. Holes and other contours "
            "remain excluded at every accuracy."
            if style == "cvat"
            else "Simplification distance in pixels (default: 1.4): "
            "0 keeps the exact mask boundary; larger values reduce vertices "
            "while keeping components and holes."
        )

    def show(self, proposal: SegmentationProposal | None = None) -> None:
        """Display a new proposal without changing the selected annotation tool."""
        if self._disposed:
            return
        if proposal is None:
            super().show()
            return
        self._timer.stop()
        self._proposal = proposal
        self._exact = None
        self._cvat_contour = None
        self._outline = None
        layers = self._viewer.layers
        selection = set(layers.selection)
        active = layers.selection.active
        active_mode = getattr(active, "mode", None)
        self._updating = True
        try:
            data = np.array(proposal.mask, dtype=np.uint8, copy=True)
            if self.mask_layer is None or self.mask_layer not in layers:
                self.mask_layer = Labels(
                    data,
                    name="Suggested outline",
                    colormap=DirectLabelColormap(
                        color_dict={None: "cyan", 0: "transparent", 1: "cyan"}
                    ),
                )
                self.mask_layer.events.data.connect(self._mask_changed)
                self.mask_layer.events.paint.connect(self._mask_changed)
                self._viewer.add_layer(self.mask_layer)
            else:
                self.mask_layer.data = data
            self.mask_layer.selected_label = 1
            self.mask_layer.contour = 2
            self.mask_layer.opacity = 1.0
            self.mask_layer.mode = "pan_zoom"
            self.mask_layer.visible = False
            if self.polygon_layer is None or self.polygon_layer not in layers:
                self.polygon_layer = Shapes(
                    ndim=2,
                    name="Polygon outline preview",
                    shape_type="path",
                    edge_color="cyan",
                    edge_width=2,
                    face_color="transparent",
                )
                self.polygon_layer.events.editable.connect(self._lock_polygon_editing)
                self._viewer.add_layer(self.polygon_layer)
            self.polygon_layer.opacity = 1.0
            self.polygon_layer.visible = True
            self.polygon_layer.mode = "pan_zoom"
            self.polygon_layer.editable = False
            # Keep both preview layers below prompt markers, with paths above mask.
            other_layers = [
                item
                for item in layers
                if item is not self.mask_layer and item is not self.polygon_layer
            ]
            insertion = next(
                (i for i, item in enumerate(other_layers) if isinstance(item, Points)),
                len(other_layers),
            )
            layers.move(layers.index(self.mask_layer), insertion)
            layers.move(layers.index(self.polygon_layer), insertion + 1)
        finally:
            self._updating = False
            layers.selection.clear()
            layers.selection.update(item for item in selection if item in layers)
            if active is not None and active in layers:
                layers.selection.active = active
                if active_mode is not None and active not in (
                    self.mask_layer,
                    self.polygon_layer,
                ):
                    active.mode = active_mode
        self._refresh()

    def _lock_polygon_editing(self, event: Any = None) -> None:
        # Napari resets Shapes.editable after data and displayed-dimension changes.
        # The source mask is the only editing surface for an assisted outline.
        if self.polygon_layer is not None and self.polygon_layer.editable:
            self.polygon_layer.editable = False

    def _mask_changed(self, event: Any = None) -> None:
        if self._disposed or self._updating or self._proposal is None:
            return
        self._outline = None
        self.detail_status.setText("Updating painted outline…")
        self._timer.start()
        self._on_change()

    def _detail_changed(self, value: int) -> None:
        if not self._disposed and self._proposal is not None:
            self._refresh()

    def _refresh(self) -> None:
        if not self._disposed:
            with suppress(ValueError, ImportError):
                self.snapshot()

    def snapshot(self) -> PolygonOutline:
        """Synchronously vectorize current painted pixels and return shown paths."""
        self._timer.stop()
        if self._disposed:
            raise ValueError("The outline preview is closed.")
        can_adjust_detail = False
        try:
            if (
                self._proposal is None
                or self.mask_layer is None
                or self.mask_layer not in self._viewer.layers
                or self.polygon_layer is None
                or self.polygon_layer not in self._viewer.layers
            ):
                raise ValueError("The preview was removed. Suggest another outline.")
            mask = np.asarray(self.mask_layer.data) != 0
            if mask.shape != self._proposal.mask.shape:
                raise ValueError("Preview dimensions must match the suggested mask.")
            if not np.any(mask):
                raise ValueError(
                    "The outline is empty. Paint foreground before accepting."
                )
            if self._style == "cvat":
                if self._cvat_contour is None or not np.array_equal(
                    mask, self._cvat_contour.proposal.mask
                ):
                    self._cvat_contour = cvat_contour(
                        SegmentationProposal(
                            mask,
                            score=self._proposal.score,
                            metadata=self._proposal.metadata,
                        )
                    )
                can_adjust_detail = True
                outline = simplify_cvat_contour(
                    self._cvat_contour, accuracy=self.detail_slider.value()
                )
                detail = (
                    f"CVAT accuracy {self.detail_slider.value()}/13 "
                    f"({outline.simplify_tolerance_px:.2f} px) · "
                    f"{self._cvat_contour.dropped_contour_count} other contours dropped"
                )
            else:
                if self._exact is None or not np.array_equal(
                    mask, self._exact.proposal.mask
                ):
                    self._exact = polygon_outline(
                        SegmentationProposal(
                            mask,
                            score=self._proposal.score,
                            metadata=self._proposal.metadata,
                        )
                    )
                can_adjust_detail = True
                outline = simplify_polygon_outline(
                    self._exact, self.detail_slider.value() / 10.0
                )
                detail = f"{outline.simplify_tolerance_px:g} px simplification"
            self.polygon_layer.data = [ring[:, ::-1] for ring in outline.rings]
            self.polygon_layer.shape_type = "path"
            self.polygon_layer.opacity = 1.0
            self._outline = outline
            self.detail_slider.setEnabled(True)
            self.detail_status.setText(f"{outline.vertex_count} vertices · {detail}")
        except (ValueError, ImportError) as exc:
            self._outline = None
            if not can_adjust_detail:
                self._exact = None
                self._cvat_contour = None
            self.detail_slider.setEnabled(can_adjust_detail)
            self.detail_status.setText(str(exc))
            if (
                self.polygon_layer is not None
                and self.polygon_layer in self._viewer.layers
            ):
                self.polygon_layer.data = []
            self._on_change()
            raise
        self._on_change()
        return outline

    def edit(self) -> None:
        """Activate the foreground brush, including recovery from an empty mask."""
        if self._disposed or self._proposal is None or self.mask_layer is None:
            return
        if self.mask_layer not in self._viewer.layers:
            return
        self.mask_layer.visible = True
        self.mask_layer.opacity = 0.5
        self.mask_layer.contour = 0
        self._viewer.layers.selection.active = self.mask_layer
        self.mask_layer.mode = "paint"

    def invalidate(self, *, keep_preview: bool = False) -> None:
        """Cancel pending edits and dim or remove geometry that became stale."""
        if self._disposed:
            return
        self._timer.stop()
        self._proposal = None
        self._exact = None
        self._cvat_contour = None
        self._outline = None
        self.detail_slider.setEnabled(False)
        self.detail_status.setText("Suggest an outline to adjust polygon detail.")
        if keep_preview:
            for layer in (self.mask_layer, self.polygon_layer):
                if layer is not None and layer in self._viewer.layers:
                    layer.opacity = 0.3
                    layer.mode = "pan_zoom"
        else:
            self._remove_layers()
        self._on_change()

    def _layer_removed(self, event: Any) -> None:
        if (
            not self._disposed
            and not self._updating
            and getattr(event, "value", None)
            in (
                self.mask_layer,
                self.polygon_layer,
            )
        ):
            self._timer.stop()
            if (
                self.mask_layer is not None
                and getattr(event, "value", None) is self.mask_layer
            ):
                for signal in (
                    self.mask_layer.events.data,
                    self.mask_layer.events.paint,
                ):
                    with suppress(ValueError, RuntimeError):
                        signal.disconnect(self._mask_changed)
                self.mask_layer = None
            else:
                if self.polygon_layer is not None:
                    with suppress(ValueError, RuntimeError):
                        self.polygon_layer.events.editable.disconnect(
                            self._lock_polygon_editing
                        )
                self.polygon_layer = None
            self._outline = None
            self._exact = None
            self._cvat_contour = None
            self.detail_slider.setEnabled(False)
            self.detail_status.setText(
                "The preview was removed. Suggest another outline."
            )
            self._on_change()

    def _remove_layers(self) -> None:
        self._updating = True
        try:
            if self.mask_layer is not None:
                for signal in (
                    self.mask_layer.events.data,
                    self.mask_layer.events.paint,
                ):
                    with suppress(ValueError, RuntimeError):
                        signal.disconnect(self._mask_changed)
            if self.polygon_layer is not None:
                with suppress(ValueError, RuntimeError):
                    self.polygon_layer.events.editable.disconnect(
                        self._lock_polygon_editing
                    )
            for layer in (self.mask_layer, self.polygon_layer):
                if layer is not None and layer in self._viewer.layers:
                    self._viewer.layers.remove(layer)
            self.mask_layer = None
            self.polygon_layer = None
        finally:
            self._updating = False

    def dispose(self) -> None:
        """Release the independent timer and layers after a dock has been deleted."""
        if self._disposed:
            return
        self._disposed = True
        self._timer.stop()
        self._timer.timeout.disconnect(self._refresh)
        with suppress(ValueError, RuntimeError):
            self._viewer.layers.events.removed.disconnect(self._layer_removed)
        self._remove_layers()
        self._proposal = None
        self._outline = None
        self._exact = None
        self._cvat_contour = None
