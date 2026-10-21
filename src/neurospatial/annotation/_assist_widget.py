"""Optional Qt controls for pixel-space calibration and segmentation previews."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import numpy as np
from napari.layers import Points
from napari.qt.threading import thread_worker
from napari.utils.colormaps import DirectLabelColormap
from numpy.typing import NDArray
from qtpy.QtCore import QTimer, Slot  # type: ignore[attr-defined]
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from neurospatial.annotation._mask_cleanup import remove_unprompted_fragments
from neurospatial.annotation.segmentation import (
    RegionGrowProvider,
    SegmentAnythingProvider,
    SegmentationPrompt,
    SegmentationProposal,
    SegmentationProvider,
    _SAMRefinement,
)
from neurospatial.ops.transforms import VideoCalibration, calibrate_from_scale_bar

if TYPE_CHECKING:
    import napari


@thread_worker(ignore_errors=True)
def _predict(
    provider: SegmentationProvider,
    frame: NDArray[np.uint8],
    prompt: SegmentationPrompt,
    previous: _SAMRefinement | None = None,
) -> SegmentationProposal | _SAMRefinement:
    """Run all inference, including lazy model setup, off the GUI thread."""
    if isinstance(provider, SegmentAnythingProvider):
        return provider._predict_with_refinement(frame, prompt, previous)
    return provider.predict(frame, prompt)


class AnnotationAssistWidget(QWidget):
    """Keep temporary pixel prompts separate from committed annotations.

    Callbacks are the only route to session mutations. ``dispose`` must be called
    when the owning annotation session closes, even if inference is still running.
    """

    def __init__(
        self,
        viewer: napari.Viewer,
        frame: NDArray[np.uint8],
        *,
        provider: SegmentationProvider,
        calibration: VideoCalibration | None,
        bin_size: float | None,
        on_calibration: Callable[[VideoCalibration | None], None],
        on_bin_size: Callable[[float], None],
        on_accept: Callable[[SegmentationProposal], None],
        on_change: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._viewer = viewer
        self._frame = np.array(frame, copy=True)
        self._provider = provider
        self._on_calibration = on_calibration
        self._on_bin_size = on_bin_size
        self._on_accept = on_accept
        self._on_change = on_change
        self._disposed = False
        self._generation = 0
        self._job_generation = -1
        self._worker: Any = None
        self._pending_preview = False
        # Own the timer separately: napari deletes the dock before dispose().
        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self._update_preview)
        self._raw_proposal: SegmentationProposal | None = None
        self._refinement: _SAMRefinement | None = None
        self._job_prompt: SegmentationPrompt | None = None
        self._proposal: SegmentationProposal | None = None
        self._preview: Any = None
        self._connections: list[tuple[Any, Any]] = []
        self._scale = Points(
            np.empty((0, 2)),
            name="Calibration scale",
            ndim=2,
            face_color="cyan",
            size=10,
        )
        self._inside = Points(
            np.empty((0, 2)),
            name="Inside prompts",
            ndim=2,
            face_color="lime",
            size=8,
        )
        self._outside = Points(
            np.empty((0, 2)),
            name="Outside prompts",
            ndim=2,
            face_color="red",
            size=8,
        )
        for layer in (self._inside, self._outside):
            self._connect(layer.events.data, self._prompts_changed)

        layout = QVBoxLayout(self)
        calibration_toggle = QPushButton("Calibration and output settings ▸")
        calibration_toggle.setCheckable(True)
        calibration_settings = QWidget()
        calibration_settings.hide()
        calibration_toggle.toggled.connect(calibration_settings.setVisible)
        calibration_toggle.toggled.connect(
            lambda expanded: calibration_toggle.setText(
                "Calibration and output settings " + ("▾" if expanded else "▸")
            )
        )
        form = QFormLayout(calibration_settings)
        calibration_help = QLabel(
            "Optional: set a known distance to export in centimeters. "
            "Outlining works in pixels without calibration."
        )
        calibration_help.setWordWrap(True)
        form.addRow(calibration_help)
        mark_scale = QPushButton("Mark two scale endpoints")
        mark_scale.clicked.connect(lambda: self._activate(self._scale))
        form.addRow(mark_scale)
        self.known_distance = self._spin(10.0)
        self.known_distance.setSuffix(" cm")
        form.addRow("Known distance", self.known_distance)
        apply_scale = QPushButton("Apply calibration")
        apply_scale.clicked.connect(self._apply_calibration)
        form.addRow(apply_scale)
        clear_scale = QPushButton("Clear calibration")
        clear_scale.clicked.connect(self._clear_calibration)
        form.addRow(clear_scale)
        self.bin_size = self._spin(1.0 if bin_size is None else bin_size)
        self.bin_size.setEnabled(bin_size is not None)
        self.bin_size.valueChanged.connect(self._bin_changed)
        form.addRow("Output bin size", self.bin_size)
        self.units_status = QLabel()
        self.units_status.setWordWrap(True)
        form.addRow(self.units_status)
        self.sync_calibration(calibration)

        assistance_group = QGroupBox("Assisted outline")
        assistance = QFormLayout(assistance_group)
        instructions = QLabel(
            "Choose a role and name in Annotation Controls. "
            "Mark points inside the desired area, and outside to exclude areas. "
            "The outline updates as you click. Refine it, then accept."
        )
        instructions.setWordWrap(True)
        assistance.addRow(instructions)
        inside_button = QPushButton("Mark inside points")
        inside_button.clicked.connect(lambda: self._activate(self._inside))
        assistance.addRow(inside_button)
        outside_button = QPushButton("Mark outside points")
        outside_button.clicked.connect(lambda: self._activate(self._outside))
        assistance.addRow(outside_button)
        self.live_preview = QCheckBox("Update outline as I click")
        self.live_preview.setChecked(True)
        self.live_preview.toggled.connect(self._live_preview_changed)
        assistance.addRow(self.live_preview)
        self.remove_fragments = QCheckBox("Remove stray fragments")
        self.remove_fragments.setChecked(True)
        self.remove_fragments.setToolTip(
            "Keep regions containing inside points, preserving their holes. "
            "Turn off to see the original prediction. "
            "Changing this option replaces manual preview edits."
        )
        self.remove_fragments.toggled.connect(self._refresh_preview)
        assistance.addRow(self.remove_fragments)
        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(0.0, 1.0)
        self.tolerance.setDecimals(3)
        self.tolerance.setSingleStep(0.01)
        if isinstance(provider, RegionGrowProvider):
            self.tolerance.setValue(provider.tolerance)
            self.tolerance.valueChanged.connect(self._prompts_changed)
            assistance.addRow("Color tolerance", self.tolerance)
        else:
            self.tolerance.hide()
        self.suggest_button = QPushButton("Update outline")
        self.suggest_button.clicked.connect(self._suggest)
        assistance.addRow(self.suggest_button)
        clear_prompts = QPushButton("Clear prompts")
        clear_prompts.clicked.connect(self._clear_prompts)
        assistance.addRow(clear_prompts)
        self.edit_button = QPushButton("Paint / erase preview")
        self.edit_button.setEnabled(False)
        self.edit_button.clicked.connect(self._edit_preview)
        assistance.addRow(self.edit_button)
        self.accept_button = QPushButton("Accept outline")
        self.accept_button.setEnabled(False)
        self.accept_button.clicked.connect(self._accept)
        assistance.addRow(self.accept_button)
        discard = QPushButton("Discard outline")
        discard.clicked.connect(self._invalidate)
        assistance.addRow(discard)
        self.status = QLabel("Ready for inside points.")
        self.status.setWordWrap(True)
        assistance.addRow(self.status)
        layout.addWidget(assistance_group)
        layout.addWidget(calibration_toggle)
        layout.addWidget(calibration_settings)

    @staticmethod
    def _spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.000001, 1_000_000.0)
        spin.setDecimals(6)
        spin.setValue(value)
        return spin

    def _connect(self, signal: Any, callback: Any) -> None:
        signal.connect(callback)
        self._connections.append((signal, callback))

    def _activate(self, layer: Any) -> None:
        if self._disposed:
            return
        if layer not in self._viewer.layers:
            self._viewer.add_layer(layer)
        self._viewer.layers.selection.active = layer
        layer.mode = "add"

    def sync_calibration(self, calibration: VideoCalibration | None) -> None:
        """Refresh unit labels without changing the explicit numeric bin size."""
        self.bin_size.setSuffix(" pixels" if calibration is None else " cm")
        self.units_status.setText(
            "Output grid size is in pixels."
            if calibration is None
            else "Output grid size is now in cm. Review the bin size above. "
            "Pixel annotations stay in place."
        )

    def sync_bin_size(self, value: float) -> None:
        """Refresh an externally restored grid size without issuing a callback."""
        self.bin_size.blockSignals(True)
        try:
            self.bin_size.setValue(value)
        finally:
            self.bin_size.blockSignals(False)

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()

    def _bin_changed(self, value: float) -> None:
        try:
            self._on_bin_size(value)
            self._changed()
        except Exception as exc:
            self._error("Could not change output bin size", exc)

    def _apply_calibration(self) -> None:
        try:
            points = np.asarray(self._scale.data, dtype=float)
            if points.shape != (2, 2) or not np.all(np.isfinite(points)):
                raise ValueError("Mark exactly two finite scale endpoints first.")
            if np.array_equal(points[0], points[1]):
                raise ValueError("The two scale endpoints must be distinct.")
            # napari points use (row, column); calibration expects (x, y).
            xy = points[:, ::-1]
            height, width = self._frame.shape[:2]
            transform = calibrate_from_scale_bar(
                (float(xy[0, 0]), float(xy[0, 1])),
                (float(xy[1, 0]), float(xy[1, 1])),
                self.known_distance.value(),
                frame_size_px=(width, height),
            )
            calibration = VideoCalibration(transform, frame_size_px=(width, height))
            self._on_calibration(calibration)
            self.sync_calibration(calibration)
            self._changed()
        except Exception as exc:
            self._error("Could not calibrate", exc)

    def _clear_calibration(self) -> None:
        try:
            self._on_calibration(None)
            self.sync_calibration(None)
            self._changed()
        except Exception as exc:
            self._error("Could not clear calibration", exc)

    def _clear_prompts(self) -> None:
        self._inside.data = np.empty((0, 2))
        self._outside.data = np.empty((0, 2))
        self._invalidate()
        self.status.setText("Ready for inside points.")

    def _invalidate(
        self,
        *_args: Any,
        keep_preview: bool = False,
        keep_refinement: bool = False,
    ) -> None:
        self._preview_timer.stop()
        self._pending_preview = False
        self._generation += 1
        self._raw_proposal = None
        if not keep_refinement:
            self._refinement = None
        self._job_prompt = None
        self._proposal = None
        if not self._disposed:
            self.accept_button.setEnabled(False)
            self.edit_button.setEnabled(False)
        if self._preview is not None:
            if self._preview not in self._viewer.layers:
                self._preview = None
            elif keep_preview:
                self._preview.opacity = 0.3
            else:
                self._viewer.layers.remove(self._preview)
                self._preview = None
        if not self._disposed:
            self.status.setText(
                "Outline discarded. Add points or click Update outline."
            )

    def _prompts_changed(self, *_args: Any) -> None:
        if self._disposed:
            return
        if len(self._inside.data) == 0:
            self._invalidate()
            self.status.setText("Mark a point inside the desired area to begin.")
            return
        if self._refinement is not None:
            try:
                if not self._refinement.accepts(self._snapshot_prompt()):
                    self._refinement = None
            except ValueError:
                self._refinement = None
        self._invalidate(keep_preview=True, keep_refinement=True)
        if self.live_preview.isChecked():
            self._pending_preview = True
            self._preview_timer.start()
            self.status.setText("Updating outline…")
        else:
            self.status.setText("Prompts changed. Click Update outline when ready.")

    def _live_preview_changed(self, enabled: bool) -> None:
        if enabled:
            self._prompts_changed()
        else:
            if self._pending_preview:
                self.status.setText("Click Update outline when ready.")
            self._preview_timer.stop()
            self._pending_preview = False

    @Slot()
    def _update_preview(self) -> None:
        # Coalesce edits made during inference into one request for the latest
        # prompts. Never run concurrent predictions on the stateful SAM model.
        if self._pending_preview and self._worker is None:
            self._suggest()

    def _snapshot_prompt(self) -> SegmentationPrompt:
        inside = np.asarray(self._inside.data, dtype=float)[:, ::-1]
        outside = np.asarray(self._outside.data, dtype=float)[:, ::-1]
        return SegmentationPrompt(
            points=np.concatenate((inside, outside)),
            labels=np.concatenate((np.ones(len(inside)), np.zeros(len(outside)))),
        )

    def _suggest(self) -> None:
        if self._disposed or self._worker is not None:
            return
        self._invalidate(keep_preview=True, keep_refinement=True)
        if len(self._inside.data) == 0:
            self.status.setText("Mark a point inside the desired area to begin.")
            return
        try:
            prompt = self._snapshot_prompt()
            provider = self._provider
            if isinstance(provider, RegionGrowProvider):
                provider = RegionGrowProvider(tolerance=self.tolerance.value())
            self._job_generation = self._generation
            self._job_prompt = prompt
            self._worker = _predict(
                provider, self._frame.copy(), prompt, self._refinement
            )
            self._worker.returned.connect(self._show_proposal)
            self._worker.errored.connect(self._inference_error)
            self._worker.finished.connect(self._finished)
            self.suggest_button.setEnabled(False)
            self.status.setText("Updating outline…")
            self._worker.start()
        except Exception as exc:
            self._worker = None
            self._refinement = None
            self.suggest_button.setEnabled(True)
            self._error("Could not suggest an outline", exc)

    @Slot(object)
    def _show_proposal(self, result: SegmentationProposal | _SAMRefinement) -> None:
        if self._disposed or self._job_generation != self._generation:
            return
        try:
            refinement = result if isinstance(result, _SAMRefinement) else None
            proposal = result.proposal if isinstance(result, _SAMRefinement) else result
            if proposal.mask.shape != self._frame.shape[:2]:
                raise ValueError("Provider returned a mask with the wrong frame size.")
            self._refinement = refinement
            self._raw_proposal = proposal
            self._refresh_preview()
        except Exception as exc:
            self._invalidate()
            self._error("Could not display outline", exc)

    def _refresh_preview(self, *_args: Any) -> None:
        if self._disposed or self._raw_proposal is None or self._job_prompt is None:
            return
        try:
            proposal = self._raw_proposal
            cleanup_note = ""
            if self.remove_fragments.isChecked():
                cleaned = remove_unprompted_fragments(proposal, self._job_prompt)
                proposal = cleaned.proposal
                if cleaned.removed_components:
                    noun = (
                        "fragment" if cleaned.removed_components == 1 else "fragments"
                    )
                    cleanup_note = (
                        f"Removed {cleaned.removed_components} stray {noun}. "
                    )
                if cleaned.missed_inside_points:
                    noun = "point" if cleaned.missed_inside_points == 1 else "points"
                    cleanup_note += (
                        f"The prediction misses {cleaned.missed_inside_points} "
                        f"inside {noun}; refine the outline. "
                    )
            data = np.asarray(proposal.mask, dtype=np.uint8).copy()
            layers = self._viewer.layers
            if self._preview is None or self._preview not in layers:
                selection = set(layers.selection)
                try:
                    self._preview = self._viewer.add_labels(
                        data,
                        name="Suggested outline",
                        colormap=DirectLabelColormap(
                            color_dict={None: "cyan", 0: "transparent", 1: "cyan"}
                        ),
                    )
                    # Keep the outline behind the click markers, and preserve
                    # the active tool so the next click is another prompt.
                    prompt_indices = [
                        layers.index(layer)
                        for layer in (self._inside, self._outside)
                        if layer in layers
                    ]
                    if prompt_indices:
                        layers.move(layers.index(self._preview), min(prompt_indices))
                finally:
                    layers.selection.clear()
                    layers.selection.update(selection)
            else:
                self._preview.data = data
            self._preview.selected_label = 1
            self._preview.contour = 2
            self._preview.opacity = 1.0
            self._preview.mode = "pan_zoom"
            self._proposal = proposal
            self.accept_button.setEnabled(True)
            self.edit_button.setEnabled(True)
            self.status.setText(
                cleanup_note
                + "Click to refine the outline, paint / erase it, or accept."
            )
        except Exception as exc:
            self._invalidate()
            self._error("Could not display outline", exc)

    @Slot(object)
    def _inference_error(self, exc: Exception) -> None:
        if not self._disposed and self._job_generation == self._generation:
            self._refinement = None
            self._error("Could not suggest an outline; edit prompts and retry", exc)

    @Slot()
    def _finished(self) -> None:
        self._worker = None
        if not self._disposed:
            self.suggest_button.setEnabled(True)
            if self._pending_preview and not self._preview_timer.isActive():
                self._preview_timer.start()

    def _edit_preview(self) -> None:
        if self._proposal is None or self._preview not in self._viewer.layers:
            return
        self._viewer.layers.selection.active = self._preview
        self._preview.contour = 0
        self._preview.opacity = 0.5
        self._preview.mode = "paint"
        self.status.setText(
            "Paint or erase the preview, then accept. "
            "Changing prompts or fragment cleanup replaces edits."
        )

    def _accept(self) -> None:
        if self._disposed or self._proposal is None or self._preview is None:
            return
        try:
            if self._preview not in self._viewer.layers:
                raise ValueError("The preview was removed. Suggest another outline.")
            mask = np.asarray(self._preview.data) != 0
            if not np.any(mask):
                raise ValueError(
                    "The outline is empty. Paint foreground before accepting."
                )
            proposal = SegmentationProposal(
                mask=mask,
                score=self._proposal.score,
                metadata=self._proposal.metadata,
            )
            self._on_accept(proposal)
        except Exception as exc:
            self._error("Could not accept outline", exc)
            return
        self._clear_prompts()
        self.status.setText("Outline accepted. Mark new points to suggest another.")

    def _error(self, message: str, exc: Exception) -> None:
        from napari.utils.notifications import show_warning

        text = f"{message}: {exc}"
        self.status.setText(text)
        show_warning(text)

    def dispose(self) -> None:
        """Invalidate outstanding inference and detach this dock's own layers."""
        if self._disposed:
            return
        self._disposed = True
        self._invalidate()
        if self._worker is not None:
            for signal, callback in (
                (self._worker.returned, self._show_proposal),
                (self._worker.errored, self._inference_error),
                (self._worker.finished, self._finished),
            ):
                # Qt already disconnects receiver slots when the dock dies.
                with suppress(RuntimeError, TypeError):
                    signal.disconnect(callback)
            self._worker.quit()
            self._worker = None
        for signal, callback in self._connections:
            signal.disconnect(callback)
        self._connections.clear()
        for layer in (self._scale, self._inside, self._outside):
            if layer in self._viewer.layers:
                self._viewer.layers.remove(layer)
