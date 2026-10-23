"""Coordinate editable projects, history, and optional assistance in napari."""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from qtpy.QtCore import QTimer  # type: ignore[attr-defined]
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QShortcut,
    QVBoxLayout,
    QWidget,
)

from neurospatial._results import read_only_copy
from neurospatial.annotation._assist_widget import AnnotationAssistWidget
from neurospatial.annotation._draft import add_polygons, annotation_roles
from neurospatial.annotation._polygon_outline import PolygonOutline
from neurospatial.annotation._project_layer import capture_shapes, restore_shapes
from neurospatial.annotation._state import AnnotationModeState, make_unique_name
from neurospatial.annotation._types import AnnotationMode, MultipleBoundaryStrategy
from neurospatial.annotation.project import AnnotationProject, AnnotationShape
from neurospatial.annotation.segmentation import (
    RegionGrowProvider,
    SegmentationProvider,
)
from neurospatial.ops.transforms import VideoCalibration
from neurospatial.regions.core import _thaw_metadata

if TYPE_CHECKING:
    from collections.abc import Callable

    import napari
    from numpy.typing import NDArray

    from neurospatial.annotation._napari_widget import _AnnotationWidget


@dataclass(frozen=True)
class _EditingState:
    """History excludes the reference image and all temporary prompt layers."""

    shapes: tuple[AnnotationShape, ...]
    calibration: VideoCalibration | None
    bin_size: float | None


def _metadata_key(shape: AnnotationShape) -> str:
    # NaN never equals itself, so compare the serialized form instead.
    return json.dumps(_thaw_metadata(shape.metadata), sort_keys=True)


def _same(left: _EditingState, right: _EditingState) -> bool:
    if left.bin_size != right.bin_size or len(left.shapes) != len(right.shapes):
        return False
    if (left.calibration is None) != (right.calibration is None):
        return False
    if (
        left.calibration is not None
        and right.calibration is not None
        and left.calibration.to_dict() != right.calibration.to_dict()
    ):
        return False
    return all(
        a.identifier == b.identifier
        and a.parent_id == b.parent_id
        and a.name == b.name
        and a.role == b.role
        and a.shape_type == b.shape_type
        and _metadata_key(a) == _metadata_key(b)
        and np.array_equal(a.vertices, b.vertices)
        for a, b in zip(left.shapes, right.shapes, strict=True)
    )


class AnnotationProjectSession:
    """Keep persistence and assistance behind the existing annotation contract."""

    def __init__(
        self,
        viewer: napari.Viewer,
        shapes_layer: Any,
        controls: _AnnotationWidget,
        *,
        frame: NDArray[np.uint8],
        video_path: str | Path | None,
        frame_index: int,
        calibration: VideoCalibration | None,
        mode: AnnotationMode,
        bin_size: float | None,
        simplify_tolerance: float | None,
        multiple_boundaries: MultipleBoundaryStrategy,
        project_path: str | Path | None,
        project_saved: bool,
        autosave_interval: float | None,
        segmentation_provider: SegmentationProvider | None,
    ) -> None:
        self.viewer = viewer
        self.layer = shapes_layer
        self.controls = controls
        self.frame = read_only_copy(frame)
        self.video_path = video_path
        self.frame_index = frame_index
        self.calibration = calibration
        self.mode = mode
        self.bin_size = bin_size
        self.simplify_tolerance = simplify_tolerance
        self.multiple_boundaries = multiple_boundaries
        self.project_path = None if project_path is None else Path(project_path)
        self._disposed = False
        self._restoring = False
        self._checkpointing = False
        self._undo: list[_EditingState] = []
        self._redo: list[_EditingState] = []
        self._current = self._capture()
        # A project resumed from its own file is already saved; autosave
        # rewrites it only after an edit.
        self._last_saved: _EditingState | None = (
            self._current if project_saved else None
        )
        # Timers are owned here rather than by the dock, which napari deletes
        # before the enclosing session's finally block runs.
        self._checkpoint_timer = QTimer()
        self._checkpoint_timer.setSingleShot(True)
        self._checkpoint_timer.setInterval(150)
        self._checkpoint_timer.timeout.connect(self._checkpoint_safely)
        self._autosave_timer = QTimer()
        self._autosave_timer.timeout.connect(self._autosave)
        if autosave_interval is not None:
            self._autosave_timer.setInterval(
                max(1, round(min(autosave_interval, 2_147_483.647) * 1000))
            )
            self._autosave_timer.start()

        panel = QWidget()
        layout = QVBoxLayout(panel)
        save_row = QHBoxLayout()
        save = QPushButton("Save Project")
        save.clicked.connect(lambda: self._run(self.save_project))
        save_row.addWidget(save)
        save_as = QPushButton("Save As…")
        save_as.clicked.connect(lambda: self._run(self._save_as))
        save_row.addWidget(save_as)
        layout.addLayout(save_row)
        history_row = QHBoxLayout()
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(lambda: self._run(self.undo))
        history_row.addWidget(self.undo_button)
        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(lambda: self._run(self.redo))
        history_row.addWidget(self.redo_button)
        layout.addLayout(history_row)
        self.status = QLabel("Save a project to resume editing later.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.assist = AnnotationAssistWidget(
            viewer,
            self.frame,
            provider=(
                RegionGrowProvider()
                if segmentation_provider is None
                else segmentation_provider
            ),
            calibration=calibration,
            bin_size=bin_size,
            on_calibration=self.set_calibration,
            on_bin_size=self.set_bin_size,
            on_accept=self.accept_outline,
        )
        layout.addWidget(self.assist)
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        viewer.window.add_dock_widget(
            scroll, name="Project and Assistance", area="right", tabify=True
        )
        # Napari raises the newly added assistance tab. Show the controls dock
        # explicitly before raising it: Qt may not have shown that dock yet.
        controls_dock = controls.native.parentWidget()
        if controls_dock is not None:
            controls_dock.show()
            controls_dock.raise_()
        self._shortcuts = []
        for key, callback in (
            (QKeySequence.StandardKey.Save, self.save_project),
            (QKeySequence.StandardKey.Undo, self.undo),
            (QKeySequence.StandardKey.Redo, self.redo),
        ):
            shortcut = QShortcut(QKeySequence(key), viewer.window._qt_window)
            shortcut.activated.connect(lambda cb=callback: self._run(cb))
            self._shortcuts.append(shortcut)
        self.layer.events.data.connect(self._changed)
        self.layer.events.features.connect(self._changed)
        self._update_history_buttons()

    def _capture(self) -> _EditingState:
        self.controls.flush_pending()
        calibration = (
            None
            if self.calibration is None
            else VideoCalibration.from_dict(self.calibration.to_dict())
        )
        return _EditingState(capture_shapes(self.layer), calibration, self.bin_size)

    def _changed(self, event: Any = None) -> None:
        if not self._disposed and not self._restoring and not self._checkpointing:
            self._checkpoint_timer.start()

    def checkpoint(self) -> None:
        """Flush pending shape edits and record one coherent history step."""
        self._checkpoint_timer.stop()
        self._checkpointing = True
        try:
            state = self._capture()
            if not _same(state, self._current):
                self._undo.append(self._current)
                del self._undo[:-50]
                self._current = state
                self._redo.clear()
                self.status.setText("Unsaved changes")
            self._update_history_buttons()
        finally:
            self._checkpointing = False

    def _checkpoint_safely(self) -> None:
        self._run(self.checkpoint)

    def _update_history_buttons(self) -> None:
        self.undo_button.setEnabled(bool(self._undo))
        self.redo_button.setEnabled(bool(self._redo))

    def _restore(self, state: _EditingState) -> None:
        self._restoring = True
        self._checkpoint_timer.stop()
        try:
            restore_shapes(state.shapes, self.layer)
            self.calibration = state.calibration
            self.bin_size = state.bin_size
            self.controls.sync_from_layer()
            self.assist.sync_calibration(self.calibration)
            if self.bin_size is not None:
                self.assist.sync_bin_size(self.bin_size)
            self.viewer.layers.selection.active = self.layer
            self._current = state
            self.status.setText("Unsaved changes")
            self._update_history_buttons()
        finally:
            self._restoring = False

    def undo(self) -> None:
        """Undo committed drawing edits, calibration, or an accepted proposal."""
        try:
            self.checkpoint()
        except ValueError:
            # A malformed external layer edit must not prevent recovery.
            self._restore(self._current)
            return
        if self._undo:
            self._redo.append(self._current)
            self._restore(self._undo.pop())

    def redo(self) -> None:
        """Restore an undone edit unless new work has replaced its history."""
        self.checkpoint()
        if self._redo:
            self._undo.append(self._current)
            self._restore(self._redo.pop())

    def set_calibration(self, calibration: VideoCalibration | None) -> None:
        """Change export coordinates without moving pixel-space annotations.

        The output bin size is converted to the new units, so the grid keeps
        its size in the image.
        """
        self.checkpoint()
        if self.bin_size is not None:
            bin_size_px = (
                self.bin_size
                if self.calibration is None
                else self.bin_size / self.calibration.cm_per_px
            )
            self.bin_size = (
                bin_size_px
                if calibration is None
                else bin_size_px * calibration.cm_per_px
            )
        self.calibration = calibration
        self.checkpoint()
        self.assist.sync_calibration(calibration)
        if self.bin_size is not None:
            self.assist.sync_bin_size(self.bin_size)

    def set_bin_size(self, value: float) -> None:
        """Record an explicit positive output grid size."""
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Output bin size must be positive and finite.")
        self.checkpoint()
        self.bin_size = value
        self.checkpoint()
        self.assist.sync_bin_size(value)

    def accept_outline(self, outline: PolygonOutline) -> None:
        """Insert the previewed polygons as one undoable domain edit."""
        self.checkpoint()
        proposal = outline.proposal
        if proposal.mask.shape != self.frame.shape[:2]:
            raise ValueError("Outline dimensions must match the reference frame.")
        role, requested_name = self.controls.next_annotation()
        if role == "wall":
            raise ValueError(
                "For walls, draw a path instead of accepting a filled area."
            )
        if role == "landmark":
            raise ValueError(
                "Draw landmarks directly. Assisted outlines define environment, "
                "region, or hole areas."
            )
        polygons = outline.polygons
        metadata = proposal.metadata
        outline_metadata: dict[str, Any] = {
            "simplify_tolerance_px": outline.simplify_tolerance_px
        }
        if "polygon_outline" in metadata:
            outline_metadata["proposal_metadata"] = metadata["polygon_outline"]
        metadata["polygon_outline"] = outline_metadata
        if role == "environment" and len(polygons) != 1:
            raise ValueError(
                "An environment needs one connected outline. Refine the preview "
                "or accept separate components as regions."
            )
        before = self._current
        existing = list(before.shapes)
        if role == "environment" and any(s.role == "environment" for s in existing):
            answer = QMessageBox.question(
                self.viewer.window._qt_window,
                "Replace environment?",
                "Replace the existing environment boundary with this outline?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                raise ValueError(
                    "Environment replacement cancelled; the preview is retained."
                )
            existing = [s for s in existing if s.role != "environment"]
        names = [s.name for s in existing]
        self._restoring = True
        try:
            if len(existing) != len(before.shapes):
                restore_shapes(existing, self.layer)
            # Name unnamed outlines as the annotation controls name drawings.
            state = AnnotationModeState(region_type=role)
            state.sync_counts_from_region_types(annotation_roles(self.layer.features))
            batch = []
            for polygon in polygons:
                name = (
                    make_unique_name(requested_name, names)
                    if requested_name
                    else state.generate_auto_name(names)
                )
                state.record_shape_added(role)
                names.append(name)
                batch.append((polygon, name, role, metadata))
            with self.layer.events.blocker_all():
                add_polygons(self.layer, batch, None)
            self.controls.sync_from_layer()
        except Exception:
            self._restore(before)
            raise
        finally:
            self._restoring = False
        self.viewer.layers.selection.active = self.layer
        self.checkpoint()

    def save_project(self, path: str | Path | None = None) -> None:
        """Save an editable draft without closing or requiring valid polygons."""
        target = Path(path) if path is not None else self.project_path
        if target is None:
            self._save_as()
            return
        self.checkpoint()
        self._write(target)

    def _write(self, target: Path) -> None:
        """Save the last checkpoint, which the caller has just taken.

        The session's own file is replaced: the entry point checked the
        destination, and Save As confirms a replacement in its dialog.
        """
        project = AnnotationProject(
            self.frame,
            shapes=self._current.shapes,
            video_path=self.video_path,
            frame_index=self.frame_index,
            calibration=self.calibration,
            mode=self.mode,
            bin_size=self.bin_size,
            simplify_tolerance=self.simplify_tolerance,
            multiple_boundaries=self.multiple_boundaries,
        )
        project.save(target, overwrite=True)
        self.project_path = target
        self._last_saved = self._current
        self.status.setText(f"Saved: {target.name}")

    def _save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self.viewer.window._qt_window,
            "Save annotation project",
            str(self.project_path or "annotations.nsp"),
            "Annotation project (*.nsp);;All files (*)",
        )
        if path:
            self.save_project(path)

    def save_if_path(self) -> None:
        """Persist the validated final edit when a project destination is set."""
        if self.project_path is not None:
            self.save_project()

    def _autosave(self) -> None:
        if self._disposed or self.project_path is None:
            return
        self._run(self._save_changed)

    def _save_changed(self) -> None:
        self.checkpoint()
        if self.project_path is not None and (
            self._last_saved is None or not _same(self._current, self._last_saved)
        ):
            self._write(self.project_path)

    def _run(self, callback: Callable[[], None]) -> None:
        if self._disposed:
            return
        try:
            callback()
        except Exception as exc:
            from napari.utils.notifications import show_warning

            self.status.setText(str(exc))
            show_warning(f"Annotation project: {exc}")

    def dispose(self) -> None:
        """Release timers and handlers even after napari destroys its docks."""
        if self._disposed:
            return
        self._disposed = True
        self._checkpoint_timer.stop()
        self._autosave_timer.stop()
        self._checkpoint_timer.timeout.disconnect(self._checkpoint_safely)
        self._autosave_timer.timeout.disconnect(self._autosave)
        self.layer.events.data.disconnect(self._changed)
        self.layer.events.features.disconnect(self._changed)
        self.assist.dispose()
        for shortcut in self._shortcuts:
            with suppress(RuntimeError):
                shortcut.setEnabled(False)
                shortcut.deleteLater()
