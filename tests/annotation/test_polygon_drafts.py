"""Editable polygon interiors remain associated with their own exterior."""

import numpy as np
import pytest
from shapely.geometry import Polygon

from neurospatial.annotation._napari_widget import (
    add_initial_boundary_to_shapes,
    setup_shapes_layer_for_annotation,
)
from neurospatial.annotation.core import (
    _add_initial_regions,
    _process_annotation_results,
)
from neurospatial.ops.transforms import Affine2D, VideoCalibration
from neurospatial.regions import Region, Regions


@pytest.mark.gui
@pytest.mark.parametrize("calibrated", [False, True])
def test_donut_boundary_and_roi_roundtrip(calibrated):
    import napari

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
def test_public_annotation_rejects_invalid_close_and_allows_retry(
    monkeypatch, close_route
):
    import napari
    from qtpy.QtWidgets import QPushButton

    from neurospatial.annotation import core

    viewers = []
    setup = core._setup_annotation_viewer
    errors = []
    builds = []
    process = core._process_annotation_results

    def capture_setup(*args, **kwargs):
        viewer, layer = setup(*args, **kwargs)
        viewers.append((viewer, layer))
        return viewer, layer

    def capture_process(*args, **kwargs):
        builds.append(1)
        return process(*args, **kwargs)

    def run_session():
        viewer, shapes = viewers[0]
        valid = shapes.data[0].copy()
        shapes.data = [np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0], [10.0, 0.0]])]
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
        close()

    monkeypatch.setattr(
        core, "_load_video_frame", lambda *args: np.zeros((30, 30, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(core, "_setup_annotation_viewer", capture_setup)
    monkeypatch.setattr(core, "_process_annotation_results", capture_process)
    monkeypatch.setattr("napari.utils.notifications.show_error", errors.append)
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
    import napari

    from neurospatial.annotation._controller import ShapesLayerController
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
    import napari

    from neurospatial.annotation._controller import ShapesLayerController
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
