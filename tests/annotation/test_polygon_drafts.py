"""Editable polygon interiors remain associated with their own exterior."""

import numpy as np
import pytest
from shapely.geometry import Polygon

from neurospatial.annotation.core import (
    _add_initial_regions,
    _process_annotation_results,
)
from neurospatial.ops.transforms import Affine2D, VideoCalibration
from neurospatial.regions import Region, Regions


@pytest.mark.gui
@pytest.mark.parametrize("calibrated", [False, True])
def test_donut_boundary_and_roi_roundtrip(calibrated):
    napari = pytest.importorskip("napari")

    from neurospatial.annotation._napari_widget import (
        add_initial_boundary_to_shapes,
        setup_shapes_layer_for_annotation,
    )

    calibration = (
        VideoCalibration(Affine2D(np.diag([0.5, 0.5, 1.0])), frame_size_px=(100, 100))
        if calibrated
        else None
    )
    boundary = Polygon(
        [(0, 0), (20, 0), (20, 20), (0, 20)], [[(3, 3), (7, 3), (7, 7), (3, 7)]]
    )
    roi = Polygon(
        [(10, 10), (18, 10), (18, 18), (10, 18)],
        [[(12, 12), (16, 12), (16, 16), (12, 16)]],
    )
    viewer = napari.Viewer(show=False)
    try:
        shapes = setup_shapes_layer_for_annotation(viewer)
        _add_initial_regions(
            shapes,
            Regions([Region("donut", "polygon", roi, metadata={"meaning": "reward"})]),
            calibration,
        )
        add_initial_boundary_to_shapes(shapes, boundary, calibration)
        result = _process_annotation_results(
            shapes, "both", 1.0, calibration, None, "last"
        )
        assert result.regions["donut"].data.equals(roi)
        assert result.regions["donut"].metadata["meaning"] == "reward"
        assert not result.environment.contains(np.array([[5.0, 5.0]]))[0]
        assert result.environment.contains(np.array([[14.0, 14.0]]))[0]
        assert result.environment.units == ("cm" if calibrated else "pixels")
        # Rename the exterior and move only the ROI's interior vertices.
        shapes.features.loc[2, "name"] = "renamed"
        shapes.data[3] = shapes.data[3] + 0.25
        result = _process_annotation_results(
            shapes, "both", 1.0, calibration, None, "last"
        )
        assert result.regions["renamed"].data.is_valid
        assert len(result.regions["renamed"].data.interiors) == 1
        # Deleting an interior fills only its own polygon.
        shapes.selected_data = {3}
        shapes.remove_selected()
        result = _process_annotation_results(
            shapes, "both", 1.0, calibration, None, "last"
        )
        assert len(result.regions["renamed"].data.interiors) == 0
        assert not result.environment.contains(np.array([[5.0, 5.0]]))[0]
    finally:
        viewer.close()


def test_invalid_polygons_rejected_without_implicit_repair():
    from neurospatial.annotation.converters import shapes_to_regions

    for simplify in (None, 1.0):
        with pytest.raises(ValueError, match=r"invalid.*Edit its vertices"):
            shapes_to_regions(
                [np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0], [10.0, 0.0]])],
                ["bowtie"],
                ["region"],
                simplify_tolerance=simplify,
                validate=False,
            )


@pytest.mark.gui
@pytest.mark.parametrize("close_route", ["save", "escape", "window"])
@pytest.mark.parametrize("invalid_role", ["environment", "landmark"])
def test_public_annotation_rejects_invalid_close_and_allows_retry(
    monkeypatch, close_route, invalid_role
):
    napari = pytest.importorskip("napari")
    from qtpy.QtWidgets import QPushButton

    from neurospatial.annotation import _session, core

    viewers = []
    setup = core._setup_annotation_viewer
    errors = []
    builds = []
    process = core._process_annotation_results

    def capture_setup(*args, **kwargs):
        viewer, layer, flush_pending = setup(*args, **kwargs)
        viewers.append((viewer, layer))
        return viewer, layer, flush_pending

    def capture_process(*args, **kwargs):
        builds.append(1)
        return process(*args, **kwargs)

    def run_session():
        viewer, shapes = viewers[0]
        valid = shapes.data[0].copy()
        shapes.data = [np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0], [10.0, 0.0]])]
        shapes.features.loc[0, "role"] = invalid_role
        window = viewer.window._qt_window

        def close():
            if close_route == "save":
                button = next(
                    button
                    for button in window.findChildren(QPushButton)
                    if button.text() == "Save and Close"
                )
                button.click()
            elif close_route == "escape":
                next(
                    callback
                    for key, callback in viewer.keymap.items()
                    if str(key).lower() == "escape"
                )(viewer)
            else:
                window.close()

        close()
        assert errors and "invalid" in errors[-1]
        assert len(viewer.layers) == 2
        assert len(shapes.data) == 1
        shapes.data = [valid]
        shapes.features.loc[0, "role"] = "environment"
        close()

    monkeypatch.setattr(
        core, "_load_video_frame", lambda *args: np.zeros((30, 30, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(core, "_setup_annotation_viewer", capture_setup)
    monkeypatch.setattr(core, "_process_annotation_results", capture_process)
    monkeypatch.setattr(
        _session, "_keep_editing", lambda parent, error: not errors.append(str(error))
    )
    monkeypatch.setattr(napari, "run", run_session)
    result = core.annotate_video(
        "unused.mp4",
        initial_boundary=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        bin_size=1.0,
    )
    assert result.environment.n_bins == 100
    assert len(builds) == 2


@pytest.mark.gui
def test_polygon_links_survive_append_and_controller_delete():
    napari = pytest.importorskip("napari")

    from neurospatial.annotation._controller import ShapesLayerController
    from neurospatial.annotation._napari_widget import (
        setup_shapes_layer_for_annotation,
    )
    from neurospatial.annotation._state import AnnotationModeState

    viewer = napari.Viewer(show=False)
    try:
        shapes = setup_shapes_layer_for_annotation(viewer)
        donut = Polygon(
            [(0, 0), (20, 0), (20, 20), (0, 20)], [[(3, 3), (7, 3), (7, 7), (3, 7)]]
        )
        _add_initial_regions(shapes, Regions([Region("donut", "polygon", donut)]), None)
        controller = ShapesLayerController(
            shapes, AnnotationModeState(region_type="region")
        )
        identity = shapes.features.loc[0, "_polygon_id"]
        shapes.add(
            np.array([[30.0, 30.0], [40.0, 30.0], [40.0, 40.0], [30.0, 40.0]]),
            shape_type="polygon",
        )
        controller.update_features_for_new_shapes(2, name_override="new")
        assert shapes.features.loc[1, "_parent_id"] == identity
        assert shapes.features.loc[2, "_polygon_id"] == ""
        assert controller.delete_shapes_by_indices({0}) == 2
        assert list(shapes.features["name"]) == ["new"]
    finally:
        viewer.close()


@pytest.mark.gui
@pytest.mark.parametrize("paste", [False, True])
def test_selected_interior_and_native_paste_keep_polygon_associations(paste):
    napari = pytest.importorskip("napari")

    from neurospatial.annotation._controller import ShapesLayerController
    from neurospatial.annotation._napari_widget import (
        setup_shapes_layer_for_annotation,
    )
    from neurospatial.annotation._state import AnnotationModeState

    viewer = napari.Viewer(show=False)
    try:
        shapes = setup_shapes_layer_for_annotation(viewer)
        donut = Polygon(
            [(0, 0), (20, 0), (20, 20), (0, 20)], [[(3, 3), (7, 3), (7, 7), (3, 7)]]
        )
        _add_initial_regions(shapes, Regions([Region("donut", "polygon", donut)]), None)
        shapes.mode = "select"
        shapes.selected_data = {1}
        if paste:
            shapes._copy_data()
            shapes._paste_data()
        assert shapes.feature_defaults["_polygon_id"].iloc[0] == ""
        assert shapes.feature_defaults["_parent_id"].iloc[0] == ""
        shapes.mode = "add_polygon"
        shapes.add(
            np.array([[30.0, 30.0], [40.0, 30.0], [40.0, 40.0], [30.0, 40.0]]),
            shape_type="polygon",
        )
        controller = ShapesLayerController(
            shapes, AnnotationModeState(region_type="region")
        )
        controller.update_features_for_new_shapes(2, name_override="new")
        if paste:
            with pytest.raises(ValueError, match="duplicate identities"):
                _process_annotation_results(shapes, "regions", None, None, None, "last")
        else:
            result = _process_annotation_results(
                shapes, "regions", None, None, None, "last"
            )
            assert result.regions["donut"].data.equals(donut)
            assert result.regions["new"].data.area == 100.0
    finally:
        viewer.close()


@pytest.mark.parametrize("role", ["landmark", "wall", "region", "environment", "hole"])
@pytest.mark.parametrize(
    "vertices",
    [
        np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0], [10.0, 0.0]]),
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
    ],
)
def test_interactive_shapes_require_valid_areas(role, vertices):
    from neurospatial.annotation.converters import _validate_drawn_shapes

    with pytest.raises(ValueError, match=r"Cannot save.*drawn"):
        _validate_drawn_shapes([vertices], ["drawn"], [role], ["polygon"], None)


@pytest.mark.gui
@pytest.mark.parametrize("delete_route", ["button", "native"])
def test_widget_delete_then_draw_assigns_fresh_features(delete_route):
    napari = pytest.importorskip("napari")

    from qtpy.QtWidgets import QApplication, QPushButton

    from neurospatial.annotation._napari_widget import (
        create_annotation_widget,
        setup_shapes_layer_for_annotation,
    )

    viewer = napari.Viewer(show=False)
    try:
        shapes = setup_shapes_layer_for_annotation(viewer)
        _add_initial_regions(
            shapes,
            Regions(
                [
                    Region(
                        "a",
                        "polygon",
                        Polygon(
                            [(0, 0), (2, 0), (2, 2), (0, 2)],
                            [[(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)]],
                        ),
                    ),
                    Region("b", "polygon", Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])),
                ]
            ),
            None,
        )
        widget = create_annotation_widget(viewer, initial_mode="region")
        viewer.window.add_dock_widget(widget)
        shapes.selected_data = {0}
        QApplication.processEvents()
        if delete_route == "button":
            button = next(
                button
                for button in viewer.window._qt_window.findChildren(QPushButton)
                if button.text() == "Delete Selected"
            )
            assert button.isEnabled()
            button.click()
        else:
            shapes.remove_selected()
        # Draw before processing queued timers: deletions must already be tracked.
        shapes.add(
            np.array([[6.0, 6.0], [8.0, 6.0], [8.0, 8.0], [6.0, 8.0]]),
            shape_type="polygon",
        )
        QApplication.processEvents()
        result = _process_annotation_results(shapes, "regions", 1.0, None, None, "last")
        assert len(result.regions) == 2
        assert len(set(shapes.features["name"])) == 2
        assert list(shapes.features["role"]) == ["region", "region"]
        assert shapes.features["_polygon_id"].iloc[1] == ""
        assert shapes.features["_parent_id"].iloc[1] == ""
    finally:
        viewer.close()


@pytest.mark.gui
def test_delete_targets_survive_pending_boundary_replacement(monkeypatch):
    napari = pytest.importorskip("napari")

    from qtpy.QtWidgets import QMessageBox, QPushButton

    from neurospatial.annotation._napari_widget import (
        add_initial_boundary_to_shapes,
        create_annotation_widget,
        setup_shapes_layer_for_annotation,
    )

    viewer = napari.Viewer(show=False)
    try:
        shapes = setup_shapes_layer_for_annotation(viewer)
        _add_initial_regions(
            shapes,
            Regions(
                [Region("roi", "polygon", Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]))]
            ),
            None,
        )
        add_initial_boundary_to_shapes(
            shapes, Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), None
        )
        widget = create_annotation_widget(viewer)
        viewer.window.add_dock_widget(widget)
        monkeypatch.setattr(
            QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes
        )
        shapes.add(
            np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]]),
            shape_type="polygon",
        )
        shapes.selected_data = {2}
        shapes.events.highlight()
        button = next(
            button
            for button in widget.native.findChildren(QPushButton)
            if button.text() == "Delete Selected"
        )
        assert button.isEnabled()
        button.click()
        widget.flush_pending()
        assert list(shapes.features["name"]) == ["roi"]
        assert not any(
            str(column).startswith("_delete_target_") for column in shapes.features
        )
        result = _process_annotation_results(shapes, "regions", 1.0, None, None, "last")
        assert list(result.regions) == ["roi"]
    finally:
        viewer.close()


@pytest.mark.gui
def test_close_flush_finishes_rejected_boundary_before_export(monkeypatch):
    napari = pytest.importorskip("napari")
    from qtpy.QtWidgets import QMessageBox

    from neurospatial.annotation import core

    captured = {}
    setup = core._setup_annotation_viewer

    def capture(*args, **kwargs):
        viewer, shapes, flush = setup(*args, **kwargs)
        captured.update(viewer=viewer, shapes=shapes)
        return viewer, shapes, flush

    def run():
        shapes = captured["shapes"]
        shapes.add(
            np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]]),
            shape_type="polygon",
        )
        # Closing immediately must resolve and apply the rejection, with no Qt
        # timer cycle between the new drawing and result construction.
        assert captured["viewer"].window._qt_window.close()

    monkeypatch.setattr(
        core, "_load_video_frame", lambda *args: np.zeros((30, 30, 3), np.uint8)
    )
    monkeypatch.setattr(core, "_setup_annotation_viewer", capture)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No
    )
    monkeypatch.setattr(napari, "run", run)
    result = core.annotate_video(
        "unused.mp4",
        initial_boundary=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        bin_size=1.0,
    )
    assert result.environment.n_bins == 100


@pytest.mark.gui
def test_native_delete_does_not_hide_pending_addition():
    napari = pytest.importorskip("napari")

    from neurospatial.annotation._napari_widget import (
        create_annotation_widget,
        setup_shapes_layer_for_annotation,
    )

    viewer = napari.Viewer(show=False)
    try:
        shapes = setup_shapes_layer_for_annotation(viewer, "region")
        _add_initial_regions(
            shapes,
            Regions(
                [Region("a", "polygon", Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]))]
            ),
            None,
        )
        widget = create_annotation_widget(viewer, initial_mode="region")
        shapes.feature_defaults["name"] = "a"
        shapes.add(
            np.array([[3.0, 3.0], [5.0, 3.0], [5.0, 5.0], [3.0, 5.0]]),
            shape_type="polygon",
        )
        shapes.selected_data = {0}
        shapes.remove_selected()
        widget.flush_pending()
        assert list(shapes.features["role"]) == ["region"]
        assert list(shapes.features["name"]) == ["region_1"]
        result = _process_annotation_results(shapes, "regions", 1.0, None, None, "last")
        assert list(result.regions) == ["region_1"]
    finally:
        viewer.close()


@pytest.mark.gui
def test_rejected_boundary_then_pending_region_keeps_fresh_features(monkeypatch):
    napari = pytest.importorskip("napari")

    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QMessageBox

    from neurospatial.annotation._napari_widget import (
        add_initial_boundary_to_shapes,
        create_annotation_widget,
        setup_shapes_layer_for_annotation,
    )

    viewer = napari.Viewer(show=False)
    try:
        shapes = setup_shapes_layer_for_annotation(viewer)
        add_initial_boundary_to_shapes(
            shapes, Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), None
        )
        widget = create_annotation_widget(viewer)
        queued = []
        monkeypatch.setattr(
            QTimer, "singleShot", lambda interval, callback: queued.append(callback)
        )
        monkeypatch.setattr(
            QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No
        )
        shapes.add(
            np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]]),
            shape_type="polygon",
        )
        # The first scheduled flush resolves the dialog and its deferred
        # rejection atomically, before a later queued edit can start.
        queued.pop(0)()
        assert len(shapes.data) == 1
        next(
            callback
            for key, callback in viewer.keymap.items()
            if str(key).lower() == "r"
        )(viewer)
        shapes.feature_defaults["name"] = "environment"
        assert shapes.feature_defaults["_polygon_id"].iloc[0] == ""
        shapes.add(
            np.array([[3.0, 3.0], [5.0, 3.0], [5.0, 5.0], [3.0, 5.0]]),
            shape_type="polygon",
        )
        widget.flush_pending()
        assert list(shapes.features["role"]) == ["environment", "region"]
        assert shapes.features["name"].iloc[1] == "region_1"
        # Stale timer callbacks must not consume or remove the later drawing.
        for callback in queued:
            callback()
        assert len(shapes.data) == 2
        result = _process_annotation_results(shapes, "both", 1.0, None, None, "last")
        assert result.environment.n_bins == 100
        assert list(result.regions) == ["region_1"]
    finally:
        viewer.close()
