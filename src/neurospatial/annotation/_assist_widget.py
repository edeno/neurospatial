"""Optional Qt controls for pixel-space calibration and segmentation previews."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from napari.layers import Points
from napari.qt.threading import thread_worker
from numpy.typing import NDArray
from qtpy.QtCore import QTimer, Slot  # type: ignore[attr-defined]
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from neurospatial._results import read_only_copy
from neurospatial.annotation._assist_roi import AnnotationROIControl, BoxPromptControl
from neurospatial.annotation._outline_preview import OutlinePreviewControl
from neurospatial.annotation._polygon_outline import PolygonOutline
from neurospatial.annotation._prompt_tools import PromptTools
from neurospatial.annotation._roi import (
    ROIBounds,
    clean_roi_fragments,
    crop_segmentation_input,
    restore_roi_proposal,
)
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


@dataclass(frozen=True)
class _PredictionResult:
    """A full-frame mask and crop-local, nonpersistent model refinement state."""

    proposal: SegmentationProposal
    refinement: _SAMRefinement | None


def _run_prediction(
    provider: SegmentationProvider,
    frame: NDArray[np.uint8],
    prompt: SegmentationPrompt,
    previous: _SAMRefinement | None = None,
    bounds: ROIBounds | None = None,
) -> _PredictionResult:
    image, local_prompt = crop_segmentation_input(frame, prompt, bounds=bounds)
    refinement = None
    if isinstance(provider, SegmentAnythingProvider):
        refinement = provider._predict_with_refinement(image, local_prompt, previous)
        proposal = refinement.proposal
    else:
        proposal = provider.predict(image, local_prompt)
    return _PredictionResult(
        restore_roi_proposal(proposal, frame.shape, bounds=bounds), refinement
    )


@thread_worker(ignore_errors=True)
def _warm_up(provider: SegmentAnythingProvider, frame: NDArray[np.uint8]) -> None:
    """Compute SAM image features before the first prompt, off the GUI thread."""
    provider._prepare(frame)


@thread_worker(ignore_errors=True)
def _predict(
    provider: SegmentationProvider,
    frame: NDArray[np.uint8],
    prompt: SegmentationPrompt,
    previous: _SAMRefinement | None = None,
    bounds: ROIBounds | None = None,
) -> _PredictionResult:
    """Run all inference, including lazy model setup, off the GUI thread."""
    return _run_prediction(provider, frame, prompt, previous, bounds)


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
        on_accept: Callable[[PolygonOutline], bool | None],
        role_and_name: tuple[Any, Any] | None = None,
    ) -> None:
        super().__init__()
        self._viewer = viewer
        # Inference threads share this frame, so it must never change.
        self._frame = read_only_copy(frame)
        self._provider = provider
        self._on_calibration = on_calibration
        self._on_bin_size = on_bin_size
        self._on_accept = on_accept
        self._disposed = False
        self._generation = 0
        self._job_generation = -1
        self._worker: Any = None
        self._worker_slots: list[tuple[Any, Any]] = []
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
        # Inside and outside differ in shape as well as color.
        self._outside = Points(
            np.empty((0, 2)),
            name="Outside prompts",
            ndim=2,
            symbol="x",
            face_color="orange",
            border_color="orange",
            size=10,
        )
        for layer in (self._inside, self._outside):
            self._connect(layer.events.data, self._prompts_changed)

        self.prompt_tools = PromptTools(
            viewer, self._inside, self._outside, self._frame.shape
        )
        self.roi = AnnotationROIControl(
            viewer, self._frame.shape, on_change=self._roi_changed
        )
        self.box = BoxPromptControl(
            viewer, self._frame.shape, on_change=self._prompts_changed
        )
        self.outline_preview = OutlinePreviewControl(
            viewer, on_change=self._outline_changed
        )

        layout = QVBoxLayout(self)

        # Units come first: the output grid depends on them. The toggle keeps a
        # one-line summary visible while the settings are collapsed.
        self._grid_estimate: int | None = None
        self.units_toggle = QPushButton()
        self.units_toggle.setCheckable(True)
        self.units_toggle.setStyleSheet("text-align: left;")
        units_settings = QWidget()
        units_settings.hide()
        self.units_toggle.toggled.connect(units_settings.setVisible)
        self.units_toggle.toggled.connect(self._update_units_summary)
        form = QFormLayout(units_settings)
        calibration_help = QLabel(
            "To export in centimeters, mark the two ends of an object of known "
            "length, enter its length, and apply. Without calibration, outputs "
            "and the bin size are in pixels."
        )
        calibration_help.setWordWrap(True)
        form.addRow(calibration_help)
        mark_scale = QPushButton("Mark two scale endpoints")
        mark_scale.clicked.connect(lambda: self._activate(self._scale))
        form.addRow(mark_scale)
        self.known_distance = self._spin(10.0, decimals=2)
        self.known_distance.setSuffix(" cm")
        form.addRow("Known distance", self.known_distance)
        calibration_actions = QHBoxLayout()
        apply_scale = QPushButton("Apply calibration")
        apply_scale.clicked.connect(self._apply_calibration)
        calibration_actions.addWidget(apply_scale)
        clear_scale = QPushButton("Clear calibration")
        clear_scale.clicked.connect(self._clear_calibration)
        calibration_actions.addWidget(clear_scale)
        form.addRow(calibration_actions)
        self.bin_size = self._spin(1.0 if bin_size is None else bin_size, decimals=3)
        self.bin_size.setEnabled(bin_size is not None)
        if bin_size is None:
            self.bin_size.setToolTip(
                "Only environments are binned; this session has none."
            )
        self.bin_size.valueChanged.connect(self._bin_changed)
        form.addRow("Output bin size", self.bin_size)
        self.units_status = QLabel()
        self.units_status.setWordWrap(True)
        form.addRow(self.units_status)
        layout.addWidget(self.units_toggle)
        layout.addWidget(units_settings)
        self.sync_calibration(calibration)

        assistance_group = QGroupBox("Assisted outline")
        assistance = QVBoxLayout(assistance_group)
        instructions = QLabel(
            "Prompt with a box or clicks, check the cyan preview, then accept it."
        )
        instructions.setWordWrap(True)
        assistance.addWidget(instructions)
        self.accept_as_role: QComboBox | None = None
        self.accept_as_name: QLineEdit | None = None
        if role_and_name is not None:
            assistance.addLayout(self._mirror_role_and_name(*role_and_name))
        prompt_row = QHBoxLayout()
        self.mark_points_button = QPushButton("Mark points")
        self.mark_points_button.clicked.connect(lambda: self._activate(self._inside))
        prompt_row.addWidget(self.mark_points_button)
        prompt_row.addWidget(self.box)
        assistance.addLayout(prompt_row)
        click_row = QHBoxLayout()
        click_row.addWidget(QLabel("Left-click adds:"))
        self.left_inside = QRadioButton("Inside (dot)")
        self.left_outside = QRadioButton("Outside (x)")
        self.left_inside.setChecked(True)
        self.left_outside.toggled.connect(
            lambda outside: setattr(self.prompt_tools, "left_click_outside", outside)
        )
        click_row.addWidget(self.left_inside)
        click_row.addWidget(self.left_outside)
        click_row.addStretch()
        assistance.addLayout(click_row)
        prompt_help = QLabel("Right-click adds the other kind • Alt-click removes")
        prompt_help.setWordWrap(True)
        assistance.addWidget(prompt_help)
        point_actions = QHBoxLayout()
        self.undo_point_button = QPushButton("Undo last point")
        self.undo_point_button.clicked.connect(self.prompt_tools.undo_last)
        point_actions.addWidget(self.undo_point_button)
        self.delete_points_button = QPushButton("Delete selected")
        self.delete_points_button.setToolTip(
            "Select points with napari's selection tool, then delete them. "
            "Click Mark points to resume assisted clicking."
        )
        self.delete_points_button.clicked.connect(self.prompt_tools.remove_selected)
        point_actions.addWidget(self.delete_points_button)
        clear_prompts = QPushButton("Clear prompts")
        clear_prompts.clicked.connect(self._clear_prompts)
        point_actions.addWidget(clear_prompts)
        assistance.addLayout(point_actions)
        update_row = QHBoxLayout()
        self.live_preview = QCheckBox("Update as I prompt")
        self.live_preview.setChecked(True)
        self.live_preview.toggled.connect(self._live_preview_changed)
        update_row.addWidget(self.live_preview)
        self.suggest_button = QPushButton("Update outline")
        self.suggest_button.clicked.connect(self._suggest)
        update_row.addWidget(self.suggest_button)
        assistance.addLayout(update_row)
        self.remove_fragments = QCheckBox("Remove stray fragments")
        self.remove_fragments.setChecked(True)
        self.remove_fragments.setToolTip(
            "Keep regions containing inside points, preserving their holes. "
            "Turn off to see the original prediction. "
            "Changing this option replaces manual preview edits."
        )
        self.remove_fragments.toggled.connect(self._refresh_preview)
        assistance.addWidget(self.remove_fragments)
        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(0.0, 1.0)
        self.tolerance.setDecimals(3)
        self.tolerance.setSingleStep(0.01)
        self.tolerance.setToolTip(
            "Largest color difference from the clicked pixels, from 0 (exact) "
            "to 1 (any color). Raise it if only part of the floor is selected; "
            "lower it if shadows or walls are included."
        )
        if isinstance(provider, RegionGrowProvider):
            self.tolerance.setValue(provider.tolerance)
            self.tolerance.valueChanged.connect(self._prompts_changed)
            tolerance_row = QHBoxLayout()
            tolerance_row.addWidget(QLabel("Color tolerance"))
            tolerance_row.addWidget(self.tolerance)
            assistance.addLayout(tolerance_row)
        else:
            self.tolerance.hide()
        assistance.addWidget(self.outline_preview)
        outline_actions = QHBoxLayout()
        self.edit_button = QPushButton("Paint / erase")
        self.edit_button.setEnabled(False)
        self.edit_button.clicked.connect(self._edit_preview)
        outline_actions.addWidget(self.edit_button)
        self.accept_button = QPushButton("Accept outline")
        self.accept_button.setEnabled(False)
        self.accept_button.clicked.connect(self._accept)
        outline_actions.addWidget(self.accept_button)
        discard = QPushButton("Discard")
        discard.clicked.connect(self._invalidate)
        outline_actions.addWidget(discard)
        assistance.addLayout(outline_actions)
        self.status = QLabel(self._prompt_hint())
        self.status.setWordWrap(True)
        assistance.addWidget(self.status)
        layout.addWidget(assistance_group)
        roi_toggle = QPushButton("Region of interest ▸")
        roi_toggle.setCheckable(True)
        roi_toggle.setStyleSheet("text-align: left;")
        self.roi.hide()
        roi_toggle.toggled.connect(self.roi.setVisible)
        roi_toggle.toggled.connect(
            lambda expanded: roi_toggle.setText(
                "Region of interest " + ("▾" if expanded else "▸")
            )
        )
        layout.addWidget(roi_toggle)
        layout.addWidget(self.roi)
        if isinstance(provider, SegmentAnythingProvider):
            self._start_warm_up()

    def _start_warm_up(self) -> None:
        # The first prompt would otherwise wait for the image features. Prompts
        # made meanwhile wait in the single worker slot and run afterwards.
        self.suggest_button.setEnabled(False)
        self.status.setText("Preparing SAM 2 for this frame…")
        self._start_worker(
            _warm_up(self._provider, self._frame),
            self._warm_up_done,
            self._warm_up_error,
        )

    def _start_worker(self, worker: Any, returned: Any, errored: Any) -> None:
        """Run one background job; only one ever uses the provider at a time."""
        self._worker = worker
        self._worker_slots = [
            (worker.returned, returned),
            (worker.errored, errored),
            (worker.finished, self._finished),
        ]
        for signal, callback in self._worker_slots:
            signal.connect(callback)
        worker.start()

    @Slot(object)
    def _warm_up_done(self, _result: object) -> None:
        if not self._disposed and not self._pending_preview:
            self.status.setText("SAM 2 is ready. " + self._prompt_hint())

    @Slot(object)
    def _warm_up_error(self, exc: Exception) -> None:
        if not self._disposed:
            self._error("Could not prepare SAM 2; prompts will retry", exc)

    @staticmethod
    def _spin(value: float, *, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(10.0**-decimals, 1_000_000.0)
        spin.setValue(value)
        return spin

    def _mirror_role_and_name(self, role_widget: Any, name_widget: Any) -> QHBoxLayout:
        """Show and edit the next annotation's role and name here as well."""
        row = QHBoxLayout()
        row.addWidget(QLabel("Accept as"))
        role = QComboBox()
        # Assisted outlines are areas; walls and landmarks are drawn directly.
        role.addItems(["environment", "hole", "region"])
        role.setPlaceholderText("choose a role")
        name = QLineEdit()
        name.setPlaceholderText("auto-named")

        def from_controls(*_args: Any) -> None:
            role.blockSignals(True)
            role.setCurrentIndex(role.findText(str(role_widget.value)))
            role.blockSignals(False)
            name.blockSignals(True)
            name.setText(str(name_widget.value))
            name.blockSignals(False)

        def role_to_controls(text: str) -> None:
            if text:
                role_widget.value = text

        def name_to_controls(text: str) -> None:
            name_widget.value = text

        self._connect(role_widget.changed, from_controls)
        self._connect(name_widget.changed, from_controls)
        role.currentTextChanged.connect(role_to_controls)
        name.textEdited.connect(name_to_controls)
        from_controls()
        row.addWidget(role)
        row.addWidget(name)
        self.accept_as_role = role
        self.accept_as_name = name
        return row

    def set_grid_estimate(self, bins: int | None) -> None:
        """Show roughly how many bins the drawn environment will have."""
        self._grid_estimate = bins
        self._update_units_summary()

    def _update_units_summary(self, *_args: Any) -> None:
        arrow = "▾" if self.units_toggle.isChecked() else "▸"
        unit = self.bin_size.suffix().strip()
        summary = (
            f"{self.bin_size.value():g} {unit} bins"
            if self.bin_size.isEnabled()
            else f"outputs in {unit}"
        )
        if self._grid_estimate is not None:
            summary += f", ≈ {self._grid_estimate:,} bins"
        self.units_toggle.setText(f"Units and bin size {arrow}   {summary}")

    def _connect(self, signal: Any, callback: Any) -> None:
        signal.connect(callback)
        self._connections.append((signal, callback))

    def _activate(self, layer: Any) -> None:
        if self._disposed:
            return
        if layer in (self._inside, self._outside):
            self.prompt_tools.activate(layer)
            return
        self.prompt_tools.deactivate()
        if layer not in self._viewer.layers:
            self._viewer.add_layer(layer)
        self._viewer.layers.selection.active = layer
        layer.mode = "add"

    def sync_calibration(self, calibration: VideoCalibration | None) -> None:
        """Refresh unit labels without changing the explicit numeric bin size."""
        self.bin_size.setSuffix(" px" if calibration is None else " cm")
        self.units_status.setText(
            "Output grid size is in pixels."
            if calibration is None
            else "Output grid size is in cm. Changing calibration converts the "
            "bin size, so the grid keeps its size in the image. "
            "Pixel annotations stay in place."
        )
        self._update_units_summary()

    def sync_bin_size(self, value: float) -> None:
        """Refresh an externally restored grid size without issuing a callback."""
        self.bin_size.blockSignals(True)
        try:
            self.bin_size.setValue(value)
        finally:
            self.bin_size.blockSignals(False)
        self._update_units_summary()

    def _bin_changed(self, value: float) -> None:
        self._update_units_summary()
        try:
            self._on_bin_size(value)
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
        except Exception as exc:
            self._error("Could not calibrate", exc)

    def _clear_calibration(self) -> None:
        try:
            self._on_calibration(None)
        except Exception as exc:
            self._error("Could not clear calibration", exc)

    def _clear_prompts(self) -> None:
        self._inside.data = np.empty((0, 2))
        self._outside.data = np.empty((0, 2))
        self.box.clear()
        self._invalidate()
        self.status.setText(self._prompt_hint())

    def _prompt_hint(self) -> str:
        if isinstance(self._provider, SegmentAnythingProvider):
            return "Draw a box or mark a point inside the desired area to begin."
        return "Mark a point inside the desired area to begin."

    def _missing_prompt(self) -> str | None:
        """Explain why prediction cannot start yet, or return None when ready."""
        try:
            box = self.box.box
        except ValueError as exc:
            return str(exc)
        if len(self._inside.data):
            return None
        # SAM 2 can outline from a box alone; region growing needs a seed.
        if box is not None and isinstance(self._provider, SegmentAnythingProvider):
            return None
        return self._prompt_hint()

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
            self.outline_preview.invalidate(keep_preview=keep_preview)
            self.status.setText(
                "Outline discarded. Add points or click Update outline."
            )

    @property
    def _preview(self) -> Any:
        return self.outline_preview.mask_layer

    def _outline_changed(self) -> None:
        if self._disposed:
            return
        self.accept_button.setEnabled(
            self._proposal is not None and self.outline_preview.outline is not None
        )
        self.edit_button.setEnabled(
            self._proposal is not None
            and self._preview in self._viewer.layers
            and self.outline_preview.polygon_layer in self._viewer.layers
        )

    def _roi_changed(self) -> None:
        if self._disposed:
            return
        self._invalidate(keep_preview=True)
        try:
            _ = self.roi.bounds
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self._prompts_changed()

    def _local_prompt(self, prompt: SegmentationPrompt) -> SegmentationPrompt:
        return crop_segmentation_input(self._frame, prompt, bounds=self.roi.bounds)[1]

    def _prompts_changed(self, *_args: Any) -> None:
        if self._disposed:
            return
        missing = self._missing_prompt()
        if missing is not None:
            self._invalidate()
            self.status.setText(missing)
            return
        if self._refinement is not None:
            # Any ROI change has already discarded the refinement.
            try:
                if not self._refinement.accepts(
                    self._local_prompt(self._snapshot_prompt())
                ):
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
            box=self.box.box,
        )

    def _suggest(self) -> None:
        if self._disposed or self._worker is not None:
            return
        self._invalidate(keep_preview=True, keep_refinement=True)
        missing = self._missing_prompt()
        if missing is not None:
            self.status.setText(missing)
            return
        try:
            prompt = self._snapshot_prompt()
            self._local_prompt(prompt)  # Validate before launching inference.
            bounds = self.roi.bounds
            provider = self._provider
            if isinstance(provider, RegionGrowProvider):
                provider = replace(provider, tolerance=self.tolerance.value())
            self._job_generation = self._generation
            self._job_prompt = prompt
            self.suggest_button.setEnabled(False)
            self.status.setText("Updating outline…")
            self._start_worker(
                _predict(provider, self._frame, prompt, self._refinement, bounds),
                self._show_proposal,
                self._inference_error,
            )
        except Exception as exc:
            self._worker = None
            self._refinement = None
            self.suggest_button.setEnabled(True)
            self._error("Could not suggest an outline", exc)

    @Slot(object)
    def _show_proposal(self, result: _PredictionResult) -> None:
        if self._disposed or self._job_generation != self._generation:
            return
        try:
            self._refinement = result.refinement
            self._raw_proposal = result.proposal
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
                cleaned = clean_roi_fragments(
                    proposal, self._job_prompt, bounds=self.roi.bounds
                )
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
            self._proposal = proposal
            # Adding preview layers temporarily selects them. Restore the custom
            # prompt interaction as well as napari's selected layer/tool.
            prompt_active = self.prompt_tools.active
            active_layer = self._viewer.layers.selection.active
            try:
                self.outline_preview.display(proposal)
            finally:
                if prompt_active:
                    self.prompt_tools.activate(active_layer)
            self._outline_changed()
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
        self._worker_slots = []
        if not self._disposed:
            self.suggest_button.setEnabled(True)
            if self._pending_preview and not self._preview_timer.isActive():
                self._preview_timer.start()

    def _edit_preview(self) -> None:
        if self._proposal is None or self._preview not in self._viewer.layers:
            return
        self.outline_preview.edit()
        self.status.setText(
            "Paint or erase the preview, then accept. "
            "Changing prompts or fragment cleanup replaces edits."
        )

    def _accept(self) -> None:
        if self._disposed or self._proposal is None or self._preview is None:
            return
        try:
            accepted = self._on_accept(self.outline_preview.snapshot())
        except Exception as exc:
            self._error("Could not accept outline", exc)
            return
        if accepted is False:
            self.status.setText("Not accepted; the outline is kept for you to adjust.")
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
        self.prompt_tools.dispose()
        self.roi.dispose()
        self.box.dispose()
        self.outline_preview.dispose()
        self._invalidate()
        if self._worker is not None:
            for signal, callback in self._worker_slots:
                # Qt already disconnects receiver slots when the dock dies.
                with suppress(RuntimeError, TypeError):
                    signal.disconnect(callback)
            self._worker.quit()
            self._worker = None
            self._worker_slots = []
        for signal, callback in self._connections:
            signal.disconnect(callback)
        self._connections.clear()
        for layer in (self._scale, self._inside, self._outside):
            if layer in self._viewer.layers:
                self._viewer.layers.remove(layer)
