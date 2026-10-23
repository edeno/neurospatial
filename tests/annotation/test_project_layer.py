"""Editing and exporting resumed projects preserves shape and ring meaning."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from shapely.geometry import Polygon

from neurospatial.annotation._project_layer import (
    capture_project,
    capture_shapes,
    restore_project_shapes,
    restore_shapes,
)
from neurospatial.annotation.project import AnnotationProject, AnnotationShape


@pytest.mark.gui
def test_resumed_donut_wall_and_landmark_export_preserves_scientific_geometry(tmp_path):
    import napari

    from neurospatial.annotation._draft import add_polygons
    from neurospatial.annotation._napari_widget import setup_shapes_layer_for_annotation
    from neurospatial.annotation.core import _process_annotation_results
    from neurospatial.ops.transforms import Affine2D, VideoCalibration

    viewer = napari.Viewer(show=False)
    resumed = napari.Viewer(show=False)
    try:
        layer = setup_shapes_layer_for_annotation(viewer)
        calibration = VideoCalibration(Affine2D(np.diag([0.5, -0.5, 1.0])), (60, 50))
        polygon = Polygon(
            [(1, -1), (20, -1), (20, -20), (1, -20)],
            [[(3, -3), (7, -3), (7, -7), (3, -7)]],
        )
        add_polygons(
            layer,
            [
                (polygon, "donut", "region", {"cue": {"trials": [1, 2]}}),
                (
                    Polygon([(0, 0), (25, 0), (25, -25), (0, -25)]),
                    "arena",
                    "environment",
                    None,
                ),
            ],
            calibration,
        )
        initial = list(capture_shapes(layer))
        initial.extend(
            [
                AnnotationShape(
                    np.array([[20, 2], [20, 40]]),
                    shape_type="line",
                    name="divider",
                    role="wall",
                ),
                AnnotationShape(
                    np.array([[30, 30], [34, 34]]),
                    shape_type="ellipse",
                    name="cue",
                    role="landmark",
                ),
            ]
        )
        restore_shapes(initial, layer)
        expected = _process_annotation_results(
            layer, "both", 1.0, calibration, None, "last"
        )
        project = capture_project(
            np.zeros((50, 60, 3), dtype=np.uint8),
            layer,
            video_path="removed.mp4",
            frame_index=3,
            calibration=calibration,
            mode="both",
            bin_size=1.0,
            simplify_tolerance=None,
            multiple_boundaries="last",
        )
        path = tmp_path / "session.project"
        project.save(path)
        loaded = AnnotationProject.load(path)
        restored = setup_shapes_layer_for_annotation(resumed)
        restore_project_shapes(loaded, restored)
        actual = _process_annotation_results(
            restored, "both", 1.0, loaded.calibration, None, "last"
        )
        assert actual.regions["donut"].data.equals(polygon)
        assert actual.regions["donut"].metadata["cue"]["trials"] == (1, 2)
        assert actual.walls["divider"].equals(expected.walls["divider"])
        assert_array_equal(actual.landmarks["cue"].data, expected.landmarks["cue"].data)
        assert_array_equal(
            actual.environment.bin_centers, expected.environment.bin_centers
        )
        assert list(restored.shape_type)[-2:] == ["line", "ellipse"]
        # Editing the exterior label and its own interior is still associated.
        restored.features.loc[0, "name"] = "edited"
        vertices = [array.copy() for array in restored.data]
        vertices[1] += 0.5
        restored.data = list(zip(vertices, restored.shape_type, strict=True))
        edited = _process_annotation_results(
            restored, "both", 1.0, loaded.calibration, None, "last"
        )
        assert len(edited.regions["edited"].data.interiors) == 1
        assert edited.regions["edited"].metadata["cue"]["trials"] == (1, 2)
    finally:
        viewer.close()
        resumed.close()


@pytest.mark.gui
def test_snapshot_assigns_stable_ids_without_events_and_restores_runtime_guards():
    import napari

    from neurospatial.annotation._napari_widget import setup_shapes_layer_for_annotation

    viewer = napari.Viewer(show=False)
    try:
        layer = setup_shapes_layer_for_annotation(viewer)
        layer.add(np.array([[1, 2], [1, 10], [10, 10]]), shape_type="polygon")
        layer.features.loc[0, "name"] = "drawn"
        layer.features["_deleted_temporary"] = [True]
        events = []
        layer.events.data.connect(lambda event: events.append("data"))
        layer.events.features.connect(lambda event: events.append("features"))
        first = capture_shapes(layer)
        assert not events
        second = capture_shapes(layer)
        assert first[0].identifier == second[0].identifier
        assert_array_equal(first[0].vertices, [[2, 1], [10, 1], [10, 10]])
        restore_shapes(first, layer)
        assert not events
        assert "_deleted_temporary" not in layer.features
        layer.mode = "select"
        layer.selected_data = {0}
        assert layer.feature_defaults["_polygon_id"].iloc[0] == ""
        assert layer.feature_defaults["_parent_id"].iloc[0] == ""
        # Original callbacks remain connected after history replacement.
        layer.add(np.array([[20, 20], [20, 25], [25, 25]]), shape_type="polygon")
        assert events
        assert layer.features["_polygon_id"].iloc[-1] == ""
        assert capture_shapes(layer)[-1].identifier != first[0].identifier
    finally:
        viewer.close()


@pytest.mark.gui
def test_invalid_polygon_survives_save_resume_until_repaired(tmp_path):
    import napari

    from neurospatial.annotation._napari_widget import setup_shapes_layer_for_annotation
    from neurospatial.annotation.core import _process_annotation_results

    viewer = napari.Viewer(show=False)
    try:
        layer = setup_shapes_layer_for_annotation(viewer)
        bowtie = np.array([[1, 1], [10, 10], [1, 10], [10, 1]], dtype=float)
        draft = AnnotationProject(
            np.zeros((20, 20, 3), dtype=np.uint8),
            shapes=[AnnotationShape(bowtie, name="unfinished")],
            mode="regions",
        )
        path = tmp_path / "draft"
        draft.save(path)
        restored = AnnotationProject.load(path)
        restore_project_shapes(restored, layer)
        assert_array_equal(layer.data[0], bowtie[:, ::-1])
        with pytest.raises(ValueError, match="invalid"):
            _process_annotation_results(layer, "regions", None, None, None, "last")
        layer.data = [np.array([[1, 1], [1, 10], [10, 10], [10, 1]])]
        result = _process_annotation_results(layer, "regions", None, None, None, "last")
        assert result.regions["unfinished"].data.area == 81
    finally:
        viewer.close()
