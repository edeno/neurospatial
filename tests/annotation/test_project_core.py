"""Entry points resume embedded drafts and export using live session settings."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from neurospatial.annotation import core
from neurospatial.annotation.project import AnnotationProject, AnnotationShape
from neurospatial.ops.transforms import Affine2D, VideoCalibration


@pytest.mark.parametrize("interval", [0, -1, np.inf, np.nan, True, "30"])
@pytest.mark.parametrize("entry", ["video", "project"])
def test_invalid_autosave_settings_fail_before_loading_or_viewer(
    monkeypatch, interval, entry
):
    def forbidden(*args, **kwargs):
        pytest.fail(
            "Invalid timer settings must be rejected before loading or opening GUI"
        )

    monkeypatch.setattr(core, "_setup_annotation_viewer", forbidden)
    monkeypatch.setattr(core, "_load_video_frame", forbidden)
    monkeypatch.setattr(AnnotationProject, "load", forbidden)
    with pytest.raises(ValueError, match="autosave_interval"):
        if entry == "video":
            core.annotate_video("unused.mp4", bin_size=1.0, autosave_interval=interval)
        else:
            core.annotate_project("unused.project", autosave_interval=interval)


def test_fresh_session_cannot_overwrite_existing_project(tmp_path, monkeypatch):
    path = tmp_path / "existing.project"
    path.write_bytes(b"saved scientific draft")
    monkeypatch.setattr(
        core,
        "_load_video_frame",
        lambda *args: pytest.fail(
            "Must reject existing destination before opening source"
        ),
    )
    with pytest.raises(FileExistsError, match="annotate_project"):
        core.annotate_video("unused.mp4", bin_size=1.0, project_path=path)
    assert path.read_bytes() == b"saved scientific draft"


@pytest.mark.parametrize("from_file", [False, True])
def test_resume_uses_embedded_frame_and_current_session_calibration(
    tmp_path, monkeypatch, from_file
):
    from neurospatial.annotation import _session

    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    draft = AnnotationProject(
        frame, mode="regions", video_path="deleted.mp4", frame_index=13
    )
    path = tmp_path / "draft.project"
    draft.save(path)
    source = path if from_file else draft
    live_calibration = VideoCalibration(Affine2D(np.diag([0.25, -0.25, 1.0])), (20, 10))
    order = []
    captured = {}
    viewer = SimpleNamespace(close=lambda: order.append("close"))
    shapes, controls = object(), object()

    def setup(title, image, *args, draft=None):
        captured.update(title=title, frame=image, draft=draft)
        return viewer, shapes, lambda: order.append("flush"), controls

    class Session:
        def __init__(self, actual_viewer, actual_shapes, actual_controls, **kwargs):
            assert (actual_viewer, actual_shapes, actual_controls) == (
                viewer,
                shapes,
                controls,
            )
            captured.update(settings=kwargs)
            self.calibration = live_calibration
            self.bin_size = 2.5

        def save_if_path(self):
            order.append("save")

        def dispose(self):
            order.append("dispose")

    def process(actual_shapes, mode, bin_size, calibration, tolerance, multiple):
        order.append("export")
        assert actual_shapes is shapes
        assert mode == "regions"
        assert bin_size == 2.5
        assert calibration is live_calibration
        return "validated result"

    monkeypatch.setitem(sys.modules, "napari", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "neurospatial.annotation._workflow",
        SimpleNamespace(AnnotationProjectSession=Session),
    )
    monkeypatch.setattr(core, "_setup_annotation_viewer", setup)
    monkeypatch.setattr(
        core,
        "_load_video_frame",
        lambda *args: pytest.fail("Resume must not open the source video"),
    )
    monkeypatch.setattr(core, "_process_annotation_results", process)
    monkeypatch.setattr(
        _session, "run_annotation_session", lambda actual_viewer, prepare: prepare()
    )
    assert core.annotate_project(source, autosave_interval=None) == "validated result"
    np.testing.assert_array_equal(captured["frame"], frame)
    assert captured["draft"].frame_index == 13
    assert captured["settings"]["video_path"] == Path("deleted.mp4")
    assert captured["settings"]["project_path"] == (path if from_file else None)
    assert captured["settings"]["autosave_interval"] is None
    assert order == ["flush", "export", "save", "dispose"]


def test_failed_export_does_not_trigger_final_save_and_disposes_session(monkeypatch):
    from neurospatial.annotation import _session

    calls = []
    draft = AnnotationProject(np.zeros((4, 4, 3), dtype=np.uint8), mode="regions")

    class Session:
        calibration = None
        bin_size = None

        def __init__(self, *args, **kwargs):
            pass

        def save_if_path(self):
            calls.append("save")

        def dispose(self):
            calls.append("dispose")

    def fail(*args):
        raise ValueError("Repair invalid boundary")

    monkeypatch.setitem(sys.modules, "napari", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "neurospatial.annotation._workflow",
        SimpleNamespace(AnnotationProjectSession=Session),
    )
    monkeypatch.setattr(
        core,
        "_setup_annotation_viewer",
        lambda *args, **kwargs: (object(), object(), lambda: None, object()),
    )
    monkeypatch.setattr(core, "_process_annotation_results", fail)
    monkeypatch.setattr(
        _session, "run_annotation_session", lambda viewer, prepare: prepare()
    )
    with pytest.raises(ValueError, match="Repair invalid boundary"):
        core.annotate_project(draft)
    assert calls == ["dispose"]


def test_resume_requires_bin_size_when_environment_export_is_requested(monkeypatch):
    draft = AnnotationProject(np.zeros((4, 4, 3), dtype=np.uint8))
    monkeypatch.setattr(
        core,
        "_run_annotation_editor",
        lambda *args, **kwargs: pytest.fail("Validate before opening GUI"),
    )
    with pytest.raises(ValueError, match="bin_size is required when mode='both'"):
        core.annotate_project(draft)


@pytest.mark.gui
def test_draft_geometry_is_restored_before_annotation_widget_creation(monkeypatch):
    from magicgui.widgets import Label, Select

    from neurospatial.annotation import _napari_widget

    draft = AnnotationProject(
        np.zeros((20, 20, 3), dtype=np.uint8),
        mode="regions",
        shapes=[
            AnnotationShape(
                np.array([[2, 1], [10, 1], [10, 10]]),
                name="saved label",
                identifier="stable",
            )
        ],
    )
    create = _napari_widget.create_annotation_widget
    initialized = []

    def capture(viewer, layer_name, **kwargs):
        layer = viewer.layers[layer_name]
        initialized.append(
            (list(layer.features["name"]), list(layer.features["_polygon_id"]))
        )
        return create(viewer, layer_name, **kwargs)

    monkeypatch.setattr(_napari_widget, "create_annotation_widget", capture)
    viewer, shapes, flush, controls = core._setup_annotation_viewer(
        Path("draft"), draft.frame, "regions", None, None, None, None, draft=draft
    )
    try:
        assert initialized == [(["saved label"], ["stable"])]
        flush()
        assert list(shapes.features["name"]) == ["saved label"]
        assert controls.flush_pending == flush
        choices = next(widget for widget in controls if isinstance(widget, Select))
        assert choices.choices == (0,)
        viewer.add_points(np.array([[5.0, 5.0]]), name="Inside prompts")
        assert choices.choices == (0,)
        assert any(
            "1 regions" in widget.value
            for widget in controls
            if isinstance(widget, Label)
        )
        np.testing.assert_array_equal(shapes.data[0], draft.shapes[0].vertices[:, ::-1])
    finally:
        viewer.close()


@pytest.mark.gui
def test_public_resume_keeps_invalid_close_editable_and_saves_live_calibration(
    tmp_path, monkeypatch
):
    import napari

    from neurospatial.annotation import _workflow

    vertices = np.array([[1, 1], [11, 1], [11, 11], [1, 11]], dtype=float)
    draft = AnnotationProject(
        np.zeros((20, 20, 3), dtype=np.uint8),
        shapes=[
            AnnotationShape(
                vertices, name="arena", role="environment", identifier="arena-id"
            )
        ],
        mode="both",
        bin_size=1.0,
    )
    path = tmp_path / "saved.project"
    draft.save(path)
    original = path.read_bytes()
    sessions = []
    errors = []
    original_session = _workflow.AnnotationProjectSession

    class CaptureSession(original_session):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            sessions.append(self)

    def edit_and_close():
        session = sessions[0]
        session.layer.data = [np.array([[1, 1], [11, 11], [1, 11], [11, 1]])[:, ::-1]]
        assert not session.viewer.window._qt_window.close()
        assert errors and "invalid" in errors[-1]
        assert path.read_bytes() == original
        session.layer.data = [vertices[:, ::-1]]
        session.set_calibration(
            VideoCalibration(
                Affine2D(np.array([[0.5, 0, 0], [0, -0.5, 10], [0, 0, 1]])), (20, 20)
            )
        )
        session.set_bin_size(0.5)
        assert session.viewer.window._qt_window.close()

    monkeypatch.setattr(_workflow, "AnnotationProjectSession", CaptureSession)
    monkeypatch.setattr("napari.utils.notifications.show_error", errors.append)
    monkeypatch.setattr(napari, "run", edit_and_close)
    monkeypatch.setattr(
        core,
        "_load_video_frame",
        lambda *args: pytest.fail("Resume cannot need the original video"),
    )
    result = core.annotate_project(path, autosave_interval=None)
    assert result.environment.units == "cm"
    assert result.environment.n_bins == 100
    assert result.environment.contains(np.array([[3, 7]]))[0]
    saved = AnnotationProject.load(path)
    assert saved.bin_size == 0.5
    assert saved.calibration.cm_per_px == 0.5
    assert_array = np.testing.assert_array_equal
    assert_array(saved.shapes[0].vertices, vertices)
    assert saved.shapes[0].identifier == "arena-id"
    assert sessions[0]._disposed


def test_failed_session_start_closes_created_viewer(monkeypatch):
    closed = []
    viewer = SimpleNamespace(close=lambda: closed.append(True))
    draft = AnnotationProject(np.zeros((4, 4, 3), dtype=np.uint8), mode="regions")

    def fail(*args, **kwargs):
        raise RuntimeError("Cannot initialize assistance")

    monkeypatch.setitem(sys.modules, "napari", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "neurospatial.annotation._workflow",
        SimpleNamespace(AnnotationProjectSession=fail),
    )
    monkeypatch.setattr(
        core,
        "_setup_annotation_viewer",
        lambda *args, **kwargs: (viewer, object(), lambda: None, object()),
    )
    with pytest.raises(RuntimeError, match="Cannot initialize assistance"):
        core.annotate_project(draft)
    assert closed == [True]


@pytest.mark.parametrize(
    "shape_type,count",
    [
        ("polygon", 0),
        ("polygon", 1),
        ("path", 1),
        ("line", 1),
        ("line", 3),
        ("rectangle", 3),
        ("ellipse", 3),
    ],
)
@pytest.mark.parametrize("from_file", [False, True])
def test_unrenderable_draft_fails_before_viewer_and_preserves_saved_geometry(
    tmp_path,
    monkeypatch,
    shape_type,
    count,
    from_file,
):
    vertices = np.arange(count * 2, dtype=float).reshape(count, 2)
    draft = AnnotationProject(
        np.zeros((10, 10, 3), dtype=np.uint8),
        mode="regions",
        shapes=[AnnotationShape(vertices, shape_type=shape_type, name="unfinished")],
    )
    path = tmp_path / "incomplete.project"
    draft.save(path)
    original = path.read_bytes()
    monkeypatch.setattr(
        core,
        "_run_annotation_editor",
        lambda *args, **kwargs: pytest.fail("Native viewer must not be created"),
    )
    with pytest.raises(
        ValueError, match=r"Cannot display.*unfinished.*napari requires"
    ):
        core.annotate_project(path if from_file else draft)
    assert path.read_bytes() == original
    loaded = AnnotationProject.load(path)
    np.testing.assert_array_equal(loaded.shapes[0].vertices, vertices)
