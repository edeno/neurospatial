"""Native Qt checks for project saving and shared annotation history."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial.annotation.project import AnnotationProject
from neurospatial.annotation.segmentation import SegmentationProposal
from neurospatial.ops.transforms import VideoCalibration, calibrate_from_scale_bar

pytestmark = pytest.mark.gui


@pytest.fixture
def session_factory():
    napari = pytest.importorskip("napari")
    from neurospatial.annotation._napari_widget import (
        create_annotation_widget,
        setup_shapes_layer_for_annotation,
    )
    from neurospatial.annotation._workflow import AnnotationProjectSession

    sessions = []

    def create(*, initial_mode="region", project_path=None, autosave_interval=None):
        viewer = napari.Viewer(show=False)
        shapes = setup_shapes_layer_for_annotation(viewer, initial_mode)
        controls = create_annotation_widget(viewer, initial_mode=initial_mode)
        viewer.window.add_dock_widget(controls, name="Annotations")
        session = AnnotationProjectSession(
            viewer,
            shapes,
            controls,
            frame=np.zeros((16, 24, 3), dtype=np.uint8),
            video_path="original-movie.mp4",
            frame_index=12,
            calibration=None,
            mode="both",
            bin_size=2.0,
            simplify_tolerance=None,
            multiple_boundaries="last",
            project_path=project_path,
            autosave_interval=autosave_interval,
            segmentation_provider=None,
        )
        state = (session, viewer, shapes, controls)
        sessions.append(state)
        return state

    yield create
    for session, viewer, _shapes, _controls in reversed(sessions):
        session.dispose()
        viewer.close()


def _draw_region(shapes, controls, vertices):
    shapes.add(np.asarray(vertices, dtype=float), shape_type="polygon")
    controls.flush_pending()


def test_save_project_preserves_invalid_draft_without_export_or_close(
    session_factory, tmp_path
):
    session, viewer, shapes, controls = session_factory()
    bowtie = np.array([[1, 2], [10, 12], [1, 12], [10, 2]], dtype=float)
    _draw_region(shapes, controls, bowtie)
    session.checkpoint()
    path = tmp_path / "unfinished.npz"
    session.save_project(path)
    project = AnnotationProject.load(path)
    assert len(project.shapes) == 1
    assert_array_equal(project.shapes[0].vertices, bowtie[:, ::-1])
    assert project.shapes[0].role == "region"
    assert project.video_path == "original-movie.mp4"
    assert project.frame_index == 12
    assert project.frame.shape == (16, 24, 3)
    assert shapes in viewer.layers
    assert viewer.window._qt_window.isVisible() is False
    # Saving a draft must not pretend it has become scientifically valid.
    from neurospatial.annotation.core import _process_annotation_results

    with pytest.raises(ValueError, match="invalid"):
        _process_annotation_results(shapes, "regions", None, None, None, "last")


def test_autosave_flushes_edits_without_closing_or_exporting(session_factory, tmp_path):
    path = tmp_path / "autosave.npz"
    session, viewer, shapes, controls = session_factory(
        project_path=path, autosave_interval=1.0
    )
    _draw_region(shapes, controls, [[1, 1], [1, 10], [10, 10], [10, 1]])
    session._autosave_timer.timeout.emit()
    project = AnnotationProject.load(path)
    assert len(project.shapes) == 1
    assert shapes in viewer.layers
    assert len(shapes.data) == 1
    assert session._autosave_timer.isActive()
    # Another autosave captures edits after the first write to the same file.
    shapes.data = [shapes.data[0] + 1.0]
    session._autosave_timer.timeout.emit()
    updated = AnnotationProject.load(path)
    assert_array_equal(updated.shapes[0].vertices, project.shapes[0].vertices + 1)


def test_accepted_components_and_hole_are_one_undoable_annotation_edit(session_factory):
    session, viewer, shapes, _controls = session_factory()
    mask = np.zeros((16, 24), dtype=bool)
    mask[1:10, 1:10] = True
    mask[4:6, 4:6] = False
    mask[11:14, 16:20] = True
    session.accept_proposal(SegmentationProposal(mask, metadata={"model": "test"}))
    # Two exteriors and one interior all survive as editable linked rings.
    assert len(shapes.data) == 3
    roles = list(shapes.features["role"])
    assert roles.count("region") == 3
    parents = list(shapes.features["_parent_id"])
    identifiers = list(shapes.features["_polygon_id"])
    assert sum(bool(parent) for parent in parents) == 1
    assert next(parent for parent in parents if parent) in identifiers
    from shapely.geometry import Point
    from shapely.ops import unary_union

    from neurospatial.annotation.core import _process_annotation_results

    result = _process_annotation_results(shapes, "regions", None, None, None, "last")
    region_union = unary_union([result.regions[name].data for name in result.regions])
    assert region_union.area == pytest.approx(float(mask.sum()))
    assert not region_union.covers(Point(4, 4))
    assert region_union.covers(Point(17, 12))
    saved_geometry = [ring.copy() for ring in shapes.data]
    saved_features = shapes.features.copy()
    assert viewer.layers.selection.active is shapes
    session.undo()
    assert len(shapes.data) == 0
    session.redo()
    assert len(shapes.data) == 3
    for actual, expected in zip(shapes.data, saved_geometry, strict=True):
        assert_array_equal(actual, expected)
    assert shapes.features.equals(saved_features)


def test_calibration_and_explicit_bin_size_share_history_without_moving_shapes(
    session_factory,
):
    session, _viewer, shapes, controls = session_factory()
    _draw_region(shapes, controls, [[1, 1], [1, 10], [10, 10], [10, 1]])
    session.checkpoint()
    original = shapes.data[0].copy()
    calibration = VideoCalibration(
        calibrate_from_scale_bar((0, 0), (10, 0), 5.0, (24, 16)), (24, 16)
    )
    session.set_calibration(calibration)
    assert session.calibration.cm_per_px == pytest.approx(0.5)
    assert session.bin_size == 2.0
    assert session.assist.bin_size.suffix() == " cm"
    session.set_bin_size(3.0)
    session.undo()
    assert session.bin_size == 2.0
    assert session.assist.bin_size.value() == 2.0
    session.undo()
    assert session.calibration is None
    assert session.assist.bin_size.suffix() == " pixels"
    assert_array_equal(shapes.data[0], original)
    session.redo()
    assert session.calibration.cm_per_px == pytest.approx(0.5)
    assert_array_equal(shapes.data[0], original)


@pytest.mark.parametrize("role", ["wall", "landmark", "environment"])
def test_unsupported_assisted_outline_does_not_partially_mutate_history(
    session_factory, role
):
    session, _viewer, shapes, controls = session_factory(initial_mode=role)
    assert controls.next_annotation()[0] == role
    mask = np.zeros((16, 24), dtype=bool)
    mask[1:4, 1:4] = True
    mask[10:14, 16:20] = True
    with pytest.raises(ValueError):
        session.accept_proposal(SegmentationProposal(mask))
    assert len(shapes.data) == 0
    session.undo()
    assert len(shapes.data) == 0


@pytest.mark.parametrize("autosave_interval", [1.0, 1e308])
def test_dispose_stops_autosave_and_can_follow_viewer_close(
    session_factory, tmp_path, autosave_interval
):
    session, viewer, _shapes, _controls = session_factory(
        project_path=tmp_path / "draft.npz", autosave_interval=autosave_interval
    )
    timer = session._autosave_timer
    assert timer.isActive()
    assert 1 <= timer.interval() <= 2_147_483_647
    viewer.close()
    session.dispose()
    assert not timer.isActive()
    session.dispose()


def test_failed_save_preserves_previous_file_and_unsaved_editor(
    session_factory, tmp_path, monkeypatch
):
    from pathlib import Path

    session, viewer, shapes, controls = session_factory()
    _draw_region(shapes, controls, [[1, 1], [1, 10], [10, 10], [10, 1]])
    path = tmp_path / "draft.nsp"
    session.save_project(path)
    previous_file = path.read_bytes()
    previous_geometry = shapes.data[0].copy()
    shapes.data = [previous_geometry + 1.0]

    def fail_replace(_source, _target):
        raise OSError("Disk write interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_replace)
        with pytest.raises(OSError, match="Disk write interrupted"):
            session.save_project()
    assert path.read_bytes() == previous_file
    assert list(tmp_path.iterdir()) == [path]
    assert shapes in viewer.layers
    assert_array_equal(shapes.data[0], previous_geometry + 1.0)
    assert session.project_path == path
    # A failed save leaves both editing and retry available.
    session.save_project()
    assert_array_equal(
        AnnotationProject.load(path).shapes[0].vertices,
        (previous_geometry + 1.0)[:, ::-1],
    )
    session.undo()
    assert_array_equal(shapes.data[0], previous_geometry)


def test_environment_replacement_cancel_then_confirm_preserves_holes_on_undo(
    session_factory, monkeypatch
):
    from qtpy.QtWidgets import QMessageBox
    from shapely.geometry import Polygon

    from neurospatial.annotation._draft import add_polygon
    from neurospatial.annotation.core import _process_annotation_results

    session, _viewer, shapes, controls = session_factory(initial_mode="environment")
    original_boundary = Polygon(
        [(1, 1), (12, 1), (12, 12), (1, 12)],
        [[(4, 4), (7, 4), (7, 7), (4, 7)]],
    )
    roi = Polygon([(8, 8), (10, 8), (10, 10), (8, 10)])
    with shapes.events.blocker_all():
        add_polygon(shapes, original_boundary, "original", "environment", None)
        add_polygon(shapes, roi, "reward", "region", None, metadata={"task": "reward"})
    controls.sync_from_layer()
    session.checkpoint()
    original_geometry = [ring.copy() for ring in shapes.data]
    original_features = shapes.features.copy()
    mask = np.zeros((16, 24), dtype=bool)
    mask[1:14, 1:15] = True
    proposal = SegmentationProposal(mask)
    answers = [QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes]
    monkeypatch.setattr(QMessageBox, "question", lambda *args: answers.pop(0))
    with pytest.raises(ValueError, match="cancelled"):
        session.accept_proposal(proposal)
    assert shapes.features.equals(original_features)
    for actual, expected in zip(shapes.data, original_geometry, strict=True):
        assert_array_equal(actual, expected)
    session.accept_proposal(proposal)
    assert len(shapes.data) == 2
    assert list(shapes.features["role"]).count("environment") == 1
    assert "reward" in list(shapes.features["name"])
    session.undo()
    assert shapes.features.equals(original_features)
    for actual, expected in zip(shapes.data, original_geometry, strict=True):
        assert_array_equal(actual, expected)
    result = _process_annotation_results(shapes, "both", 1.0, None, None, "last")
    assert not result.environment.contains(np.array([[5.0, 5.0]]))[0]
    assert result.environment.contains(np.array([[9.0, 9.0]]))[0]
    assert result.regions["reward"].data.equals(roi)
    assert result.regions["reward"].metadata["task"] == "reward"
    session.redo()
    assert len(shapes.data) == 2
    assert sum(bool(parent) for parent in shapes.features["_parent_id"]) == 0
    assert answers == []
