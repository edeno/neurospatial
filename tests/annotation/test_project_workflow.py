"""Native Qt checks for project saving and shared annotation history."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial.annotation._polygon_outline import polygon_outline
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

    def create(
        *,
        initial_mode="region",
        project_path=None,
        project_saved=False,
        autosave_interval=None,
        populate=None,
    ):
        viewer = napari.Viewer(show=False)
        shapes = setup_shapes_layer_for_annotation(viewer, initial_mode)
        if populate is not None:
            populate(shapes)
        controls = create_annotation_widget(viewer, initial_mode=initial_mode)
        viewer.window.add_dock_widget(controls, name="Annotations")
        session = AnnotationProjectSession(
            viewer,
            shapes,
            controls,
            AnnotationProject(
                np.zeros((16, 24, 3), dtype=np.uint8),
                video_path="original-movie.mp4",
                frame_index=12,
                mode="both",
                bin_size=2.0,
            ),
            project_path=project_path,
            project_saved=project_saved,
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


def _accept(session, proposal):
    session.accept_outline(polygon_outline(proposal))


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


def test_initial_region_with_missing_metadata_value_starts_saves_and_has_no_history(
    session_factory, tmp_path
):
    from shapely.geometry import box

    from neurospatial.annotation._draft import add_polygons

    def populate(shapes):
        add_polygons(
            shapes, [(box(1, 1, 8, 8), "zone", "region", {"depth": np.nan})], None
        )

    session, _viewer, _shapes, _controls = session_factory(populate=populate)
    # NaN never equals itself; unchanged metadata must not look like an edit.
    session.checkpoint()
    session.checkpoint()
    assert not session._undo
    path = tmp_path / "draft.nsp"
    session.save_project(path)
    assert np.isnan(AnnotationProject.load(path).shapes[0].metadata["depth"])


def test_pasted_polygon_with_hole_gets_its_own_identities_and_saves(
    session_factory, tmp_path
):
    from shapely.geometry import Polygon

    from neurospatial.annotation._draft import add_polygons

    def populate(shapes):
        ring = Polygon(
            [(1, 1), (10, 1), (10, 10), (1, 10)], [[(4, 4), (6, 4), (6, 6), (4, 6)]]
        )
        add_polygons(shapes, [(ring, "zone", "region", {"source": "import"})], None)

    session, _viewer, shapes, _controls = session_factory(populate=populate)
    shapes.selected_data = {0, 1}
    shapes._copy_data()
    shapes._paste_data()
    assert len(shapes.data) == 4
    path = tmp_path / "draft.nsp"
    session.save_project(path)
    saved = AnnotationProject.load(path).shapes
    assert len({shape.identifier for shape in saved}) == 4
    original, hole, copy, copied_hole = saved
    assert hole.parent_id == original.identifier
    assert copied_hole.parent_id == copy.identifier
    assert (original.name, copy.name) == ("zone", "zone_2")
    assert dict(copy.metadata) == {"source": "import"}
    # Export sees distinct polygons, each with its own hole.
    from neurospatial.annotation.core import _process_annotation_results

    result = _process_annotation_results(shapes, "regions", None, None, None, "last")
    assert len(result.regions) == 2
    assert all(len(region.data.interiors) == 1 for region in result.regions.values())


def test_unnamed_outlines_continue_the_region_numbering(session_factory):
    session, _viewer, shapes, controls = session_factory()
    mask = np.zeros((16, 24), dtype=bool)
    mask[1:6, 1:6] = True
    mask[3:4, 3:4] = False
    mask[8:12, 1:6] = True
    _accept(session, SegmentationProposal(mask))
    mask = np.zeros((16, 24), dtype=bool)
    mask[1:6, 10:14] = True
    _accept(session, SegmentationProposal(mask))
    _draw_region(shapes, controls, [[13, 16], [13, 20], [15, 20], [15, 16]])
    top_level = [
        name
        for name, parent in zip(
            shapes.features["name"], shapes.features["_parent_id"], strict=True
        )
        if not parent
    ]
    assert top_level[:3] == ["region_1", "region_2", "region_3"]


def test_resumed_project_is_not_rewritten_until_edited(
    session_factory, tmp_path, monkeypatch
):
    path = tmp_path / "draft.nsp"
    path.write_bytes(b"resumed project")
    saves = []
    monkeypatch.setattr(
        AnnotationProject, "save", lambda self, target, **kwargs: saves.append(1)
    )
    session, _viewer, shapes, controls = session_factory(
        project_path=path, project_saved=True, autosave_interval=1.0
    )
    session._autosave_timer.timeout.emit()
    assert not saves
    _draw_region(shapes, controls, [[1, 1], [1, 10], [10, 10], [10, 1]])
    session._autosave_timer.timeout.emit()
    assert saves == [1]


def test_accepted_components_and_hole_are_one_undoable_annotation_edit(session_factory):
    session, viewer, shapes, _controls = session_factory()
    mask = np.zeros((16, 24), dtype=bool)
    mask[1:10, 1:10] = True
    mask[4:6, 4:6] = False
    mask[11:14, 16:20] = True
    _accept(session, SegmentationProposal(mask, metadata={"model": "test"}))
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


def test_redo_survives_the_delayed_checkpoint_after_undo(session_factory):
    from qtpy.QtTest import QTest

    session, _viewer, shapes, controls = session_factory()
    _draw_region(shapes, controls, [[1, 1], [1, 10], [10, 10], [10, 1]])
    session.checkpoint()
    session.undo()
    assert len(shapes.data) == 0
    # Any change the session made while restoring triggers only a checkpoint
    # that finds nothing new; it must not discard the redo step.
    QTest.qWait(300)
    assert session.redo_button.isEnabled()
    session.redo()
    assert len(shapes.data) == 1


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
    # A 2 pixel bin at 0.5 cm per pixel stays the same grid: 1 cm.
    session.set_calibration(calibration)
    assert session.calibration.cm_per_px == pytest.approx(0.5)
    assert session.bin_size == pytest.approx(1.0)
    assert session.assist.bin_size.value() == pytest.approx(1.0)
    assert session.assist.bin_size.suffix() == " cm"
    session.set_bin_size(3.0)
    session.undo()
    assert session.bin_size == pytest.approx(1.0)
    assert session.assist.bin_size.value() == pytest.approx(1.0)
    session.undo()
    assert session.calibration is None
    assert session.bin_size == 2.0
    assert session.assist.bin_size.value() == 2.0
    assert session.assist.bin_size.suffix() == " pixels"
    assert_array_equal(shapes.data[0], original)
    session.redo()
    assert session.calibration.cm_per_px == pytest.approx(0.5)
    assert session.bin_size == pytest.approx(1.0)
    assert_array_equal(shapes.data[0], original)
    # Recalibrating from cm to cm keeps the pixel grid: 1 cm at 0.5 cm/px is
    # 2 px, which is 4 cm at 2 cm/px. Clearing returns the 2 px grid.
    session.set_calibration(
        VideoCalibration(
            calibrate_from_scale_bar((0, 0), (10, 0), 20.0, (24, 16)), (24, 16)
        )
    )
    assert session.bin_size == pytest.approx(4.0)
    session.set_calibration(None)
    assert session.bin_size == pytest.approx(2.0)
    assert session.assist.bin_size.suffix() == " pixels"
    assert_array_equal(shapes.data[0], original)


def test_simplified_preview_is_accepted_exactly_and_survives_history_and_save(
    session_factory, tmp_path
):
    from shapely.geometry import MultiPolygon

    from neurospatial.annotation._polygon_outline import (
        polygon_outline,
        simplify_polygon_outline,
    )
    from neurospatial.annotation._project_layer import capture_shapes, restore_shapes
    from neurospatial.annotation.core import _process_annotation_results
    from neurospatial.regions.core import _thaw_metadata

    session, _viewer, shapes, controls = session_factory()
    mask = np.zeros((16, 24), dtype=bool)
    for row in range(1, 13):
        mask[row, 1 : 9 + row // 3] = True
    mask[5:7, 4:6] = False
    mask[10:14, 17:21] = True
    provider_metadata = {
        "provider": "test",
        "settings": {"thresholds": [0.2, 0.8]},
        "polygon_outline": {"source": ["provider-owned", 3]},
    }
    proposal = SegmentationProposal(mask, metadata=provider_metadata)
    exact = polygon_outline(proposal)
    preview = simplify_polygon_outline(exact, 1.0)
    assert preview.vertex_count < exact.vertex_count
    session.accept_outline(preview)
    accepted = capture_shapes(shapes)

    def assert_preview_preserved():
        result = _process_annotation_results(
            shapes, "regions", None, None, None, "last"
        )
        actual = MultiPolygon([result.regions[name].data for name in result.regions])
        assert actual.equals_exact(MultiPolygon(preview.polygons), 0.0)
        assert sum(len(p.interiors) for p in actual.geoms) == 1
        for name in result.regions:
            metadata = _thaw_metadata(result.regions[name].metadata)
            assert metadata["provider"] == "test"
            assert metadata["settings"] == {"thresholds": [0.2, 0.8]}
            assert metadata["polygon_outline"] == {
                "simplify_tolerance_px": 1.0,
                "proposal_metadata": {"source": ["provider-owned", 3]},
            }

    assert_preview_preserved()
    assert proposal.metadata == provider_metadata
    session.undo()
    assert len(shapes.data) == 0
    session.redo()
    assert_preview_preserved()
    path = tmp_path / "simplified.nsp"
    session.save_project(path)
    loaded = AnnotationProject.load(path)
    assert len(loaded.shapes) == len(accepted) == 3
    for actual, expected in zip(loaded.shapes, accepted, strict=True):
        assert_array_equal(actual.vertices, expected.vertices)
        assert actual.metadata == expected.metadata
    with shapes.events.blocker_all():
        restore_shapes(loaded.shapes, shapes)
    controls.sync_from_layer()
    assert_preview_preserved()


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
        _accept(session, SegmentationProposal(mask))
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

    from neurospatial.annotation._draft import add_polygons
    from neurospatial.annotation.core import _process_annotation_results

    session, _viewer, shapes, controls = session_factory(initial_mode="environment")
    original_boundary = Polygon(
        [(1, 1), (12, 1), (12, 12), (1, 12)],
        [[(4, 4), (7, 4), (7, 7), (4, 7)]],
    )
    roi = Polygon([(8, 8), (10, 8), (10, 10), (8, 10)])
    with shapes.events.blocker_all():
        add_polygons(
            shapes,
            [
                (original_boundary, "original", "environment", None),
                (roi, "reward", "region", {"task": "reward"}),
            ],
            None,
        )
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
        _accept(session, proposal)
    assert shapes.features.equals(original_features)
    for actual, expected in zip(shapes.data, original_geometry, strict=True):
        assert_array_equal(actual, expected)
    _accept(session, proposal)
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
