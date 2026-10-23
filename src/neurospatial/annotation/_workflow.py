"""Coordinate editable projects, history, and optional assistance in napari."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import QTimer  # type: ignore[attr-defined]
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QShortcut,
    QVBoxLayout,
    QWidget,
)
from shapely.geometry import Polygon

from neurospatial._validation import validate_finite_scalar
from neurospatial.annotation._assist_widget import AnnotationAssistWidget
from neurospatial.annotation._draft import add_polygons, annotation_roles
from neurospatial.annotation._napari_widget import confirm_environment_replacement
from neurospatial.annotation._polygon_outline import PolygonOutline
from neurospatial.annotation._project_layer import capture_shapes, restore_shapes
from neurospatial.annotation._state import AnnotationModeState, make_unique_name
from neurospatial.annotation.project import (
    AnnotationProject,
    AnnotationShape,
    _copy_calibration,
)
from neurospatial.annotation.segmentation import (
    RegionGrowProvider,
    SegmentationProvider,
)
from neurospatial.ops.transforms import VideoCalibration

if TYPE_CHECKING:
    from collections.abc import Callable

    import napari

    from neurospatial.annotation._napari_widget import _AnnotationWidget


@dataclass(frozen=True)
class _EditingState:
    """History excludes the reference image and all temporary prompt layers."""

    shapes: tuple[AnnotationShape, ...]
    calibration: VideoCalibration | None
    bin_size: float | None
    simplify_tolerance: float | None


def _convert_length(
    value: float | None,
    old: VideoCalibration | None,
    new: VideoCalibration | None,
) -> float | None:
    """Express an output length after a calibration change.

    Only a change of units converts: pixels to centimeters or back keeps the
    length's size in the image. Replacing one calibration with another keeps
    the centimeter value, the physical length the user chose.
    """
    if value is not None and old is None and new is not None:
        return value * new.cm_per_px
    if value is not None and old is not None and new is None:
        return value / old.cm_per_px
    return value


def _same(left: _EditingState, right: _EditingState) -> bool:
    # Unchanged shapes are shared between states, so most comparisons are
    # identity checks.
    return (
        left.bin_size == right.bin_size
        and left.simplify_tolerance == right.simplify_tolerance
        and (None if left.calibration is None else left.calibration.to_dict())
        == (None if right.calibration is None else right.calibration.to_dict())
        and left.shapes == right.shapes
    )


class AnnotationProjectSession:
    """Keep persistence and assistance behind the existing annotation contract."""

    def __init__(
        self,
        viewer: napari.Viewer,
        shapes_layer: Any,
        controls: _AnnotationWidget,
        project: AnnotationProject,
        *,
        project_path: str | Path | None,
        project_saved: bool,
        autosave_interval: float | None,
        segmentation_provider: SegmentationProvider | None,
    ) -> None:
        self.viewer = viewer
        self.layer = shapes_layer
        self.controls = controls
        # The project supplies the frame and the settings this session does
        # not edit; saves replace its shapes, calibration and bin size.
        self._project = project
        self.frame = project.frame
        self.calibration = project.calibration
        self.bin_size = project.bin_size
        self.simplify_tolerance = project.simplify_tolerance
        self.project_path = None if project_path is None else Path(project_path)
        self._disposed = False
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
            calibration=self.calibration,
            bin_size=self.bin_size,
            on_calibration=self.set_calibration,
            on_bin_size=self.set_bin_size,
            on_accept=self.accept_outline,
            role_and_name=(controls.role_selector, controls.name_input),
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
            (QKeySequence.StandardKey.Undo, self._undo_key),
            (QKeySequence.StandardKey.Redo, self._redo_key),
        ):
            shortcut = QShortcut(QKeySequence(key), viewer.window._qt_window)
            shortcut.activated.connect(lambda cb=callback: self._run(cb))
            self._shortcuts.append(shortcut)
        self.layer.events.data.connect(self._changed)
        self.layer.events.features.connect(self._changed)
        self._update_history_buttons()
        self._update_grid_estimate()

    def _capture(self, previous: tuple[AnnotationShape, ...] = ()) -> _EditingState:
        self.controls.flush_pending()
        height, width = self.frame.shape[:2]
        calibration = (
            None
            if self.calibration is None
            else _copy_calibration(self.calibration, (width, height))
        )
        return _EditingState(
            capture_shapes(self.layer, previous),
            calibration,
            self.bin_size,
            self.simplify_tolerance,
        )

    def _changed(self, event: Any = None) -> None:
        # A change made by the session itself only leads to a checkpoint that
        # finds nothing new. Edits still in progress announce their end later.
        if self._disposed or getattr(event, "action", None) in ("adding", "changing"):
            return
        self._checkpoint_timer.start()

    def _edit_in_progress(self) -> bool:
        """Whether a shape is being drawn or awaits the controls' processing.

        Such a shape has no identity yet, for example while napari draws a
        polygon or while a dialog asks about a new boundary.
        """
        return bool(getattr(self.layer, "_is_creating", False)) or not bool(
            self.layer.features["_polygon_id"].astype(bool).all()
        )

    def checkpoint(self) -> None:
        """Flush pending shape edits and record one coherent history step."""
        self._checkpoint_timer.stop()
        state = self._capture(self._current.shapes)
        if not _same(state, self._current):
            self._undo.append(self._current)
            del self._undo[:-50]
            self._current = state
            self._redo.clear()
            self.status.setText("Unsaved changes")
        self._update_history_buttons()
        self._update_grid_estimate()

    def _checkpoint_safely(self) -> None:
        if self._edit_in_progress():
            self._checkpoint_timer.start()
            return
        self._run(self.checkpoint)

    def _update_grid_estimate(self) -> None:
        """Estimate the environment's bin count from its drawn area."""
        shapes = self._current.shapes
        environments = {
            s.identifier: s
            for s in shapes
            if s.role == "environment" and not s.parent_id and len(s.vertices) >= 3
        }
        if not environments or self.bin_size is None:
            self.assist.set_grid_estimate(None)
            return
        area = sum(Polygon(s.vertices).area for s in environments.values())
        # Holes are environment interior rings or separately drawn hole shapes.
        area -= sum(
            Polygon(s.vertices).area
            for s in shapes
            if len(s.vertices) >= 3
            and (s.parent_id in environments or (s.role == "hole" and not s.parent_id))
        )
        bin_px = (
            self.bin_size
            if self.calibration is None
            else self.bin_size / self.calibration.cm_per_px
        )
        self.assist.set_grid_estimate(max(round(area / bin_px**2), 0))

    def _update_history_buttons(self) -> None:
        self.undo_button.setEnabled(bool(self._undo))
        self.redo_button.setEnabled(bool(self._redo))

    def _restore(self, state: _EditingState) -> None:
        self._checkpoint_timer.stop()
        restore_shapes(state.shapes, self.layer)
        self.calibration = state.calibration
        self.bin_size = state.bin_size
        self.simplify_tolerance = state.simplify_tolerance
        self.controls.sync_from_layer()
        self.assist.sync_calibration(self.calibration)
        if self.bin_size is not None:
            self.assist.sync_bin_size(self.bin_size)
        self.viewer.layers.selection.active = self.layer
        self._current = state
        self.status.setText("Unsaved changes")
        self._update_history_buttons()
        self._update_grid_estimate()

    def _painting(self) -> bool:
        # Brush strokes have napari's own history, apart from committed edits.
        mask = self.assist.outline_preview.mask_layer
        return mask is not None and self.viewer.layers.selection.active is mask

    def _undo_key(self) -> None:
        if self._painting():
            self.assist.outline_preview.undo_paint()
        else:
            self.undo()

    def _redo_key(self) -> None:
        if self._painting():
            self.assist.outline_preview.undo_paint(redo=True)
        else:
            self.redo()

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

        Calibrating pixels converts the bin size and export simplification to
        centimeters, keeping their size in the image; clearing converts back.
        Replacing a calibration keeps their centimeter values.
        """
        self.checkpoint()
        self.bin_size = _convert_length(self.bin_size, self.calibration, calibration)
        self.simplify_tolerance = _convert_length(
            self.simplify_tolerance, self.calibration, calibration
        )
        self.calibration = calibration
        self.checkpoint()
        self.assist.sync_calibration(calibration)
        if self.bin_size is not None:
            self.assist.sync_bin_size(self.bin_size)

    def set_bin_size(self, value: float) -> None:
        """Record an explicit positive output grid size."""
        validate_finite_scalar(value, name="Output bin size", positive=True)
        self.checkpoint()
        self.bin_size = value
        self.checkpoint()
        self.assist.sync_bin_size(value)

    def accept_outline(self, outline: PolygonOutline) -> bool:
        """Insert the previewed polygons as one undoable domain edit.

        Returns False when the user declines to replace the environment.
        """
        self.checkpoint()
        proposal = outline.proposal
        if proposal.mask.shape != self.frame.shape[:2]:
            raise ValueError("Outline dimensions must match the reference frame.")
        role, requested_name = self.controls.next_annotation()
        if role in ("wall", "landmark"):
            raise ValueError(
                f"An outline is an area, not a {role}. Choose environment, hole "
                f"or region under 'Accept as'; draw {role}s directly with "
                "Annotation Controls."
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
            if not confirm_environment_replacement(self.viewer.window._qt_window):
                return False
            existing = [s for s in existing if s.role != "environment"]
        names = [s.name for s in existing]
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
            # The blocked data event would have redrawn the canvas.
            self.layer.refresh()
            self.controls.sync_from_layer()
        except Exception:
            self._restore(before)
            raise
        self.viewer.layers.selection.active = self.layer
        self.checkpoint()
        return True

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
        self._project._replace(
            shapes=self._current.shapes,
            calibration=self.calibration,
            bin_size=self.bin_size,
            simplify_tolerance=self.simplify_tolerance,
        ).save(target, overwrite=True)
        self.project_path = target
        self._last_saved = self._current
        self.status.setText(f"Saved: {target.name}")

    def _save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self.viewer.window._qt_window,
            "Save annotation project",
            str(self.project_path or "arena.annotation.npz"),
            "Annotation project (*.annotation.npz);;All files (*)",
        )
        if path:
            self.save_project(path)

    def save_if_path(self) -> None:
        """Persist the validated final edit when a project destination is set."""
        if self.project_path is not None:
            self.save_project()

    def _autosave(self) -> None:
        # The next tick saves an edit that is still being drawn.
        if self._disposed or self.project_path is None or self._edit_in_progress():
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
