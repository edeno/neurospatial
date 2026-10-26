"""Portable annotation drafts preserve coordinate meaning and unfinished work."""

import json
import pickle
from copy import deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial.annotation.project import AnnotationProject, AnnotationShape
from neurospatial.ops.transforms import AffineND, VideoCalibration


def test_roundtrip_is_self_contained_and_preserves_pixel_geometry(tmp_path):
    frame = np.arange(24 * 30 * 3, dtype=np.uint8).reshape(24, 30, 3)
    matrix = np.array([[0.5, 0.1, -2.0], [0.0, -0.5, 12.0], [0.0, 0.0, 1.0]])
    calibration = VideoCalibration(AffineND(matrix), (30, 24))
    exterior = AnnotationShape(
        np.array([[1, 2], [20, 2], [20, 22], [1, 22]]),
        identifier="outer",
        name="arena",
        role="environment",
        metadata={"animal": {"sessions": [3, 4], "tag": None}},
    )
    interior = AnnotationShape(
        np.array([[5, 5], [8, 5], [8, 8]]),
        identifier="inner",
        parent_id="outer",
        role="environment",
    )
    wall = AnnotationShape(
        np.array([[10, 2], [10, 7], [12, 9]]),
        shape_type="path",
        role="wall",
    )
    cue = AnnotationShape(
        np.array([[4, 3], [7, 9]]),
        shape_type="ellipse",
        role="landmark",
    )
    project = AnnotationProject(
        frame,
        shapes=(exterior, interior, wall, cue),
        video_path=tmp_path / "removed-source.mp4",
        frame_index=17,
        calibration=calibration,
        mode="both",
        bin_size=0.5,
        simplify_tolerance=0.2,
        multiple_boundaries="error",
    )
    path = tmp_path / "session.annotation"
    project.save(path)
    assert path.exists()
    assert not path.with_suffix(".annotation.npz").exists()
    loaded = AnnotationProject.load(path)
    assert_array_equal(loaded.frame, frame)
    assert not loaded.frame.flags.writeable
    assert loaded.video_path == str(tmp_path / "removed-source.mp4")
    assert loaded.frame_index == 17
    assert loaded.bin_size == 0.5
    assert loaded.simplify_tolerance == 0.2
    assert loaded.multiple_boundaries == "error"
    for actual, expected in zip(loaded.shapes, project.shapes, strict=True):
        assert_array_equal(actual.vertices, expected.vertices)
        assert actual.shape_type == expected.shape_type
        assert actual.role == expected.role
        assert actual.identifier == expected.identifier
        assert actual.parent_id == expected.parent_id
        assert actual.metadata == expected.metadata
        assert not actual.vertices.flags.writeable
    points = np.array([[1.0, 2.0], [20.0, 22.0]])
    np.testing.assert_allclose(
        loaded.calibration.transform_px_to_cm(points),
        calibration.transform_px_to_cm(points),
    )
    np.testing.assert_allclose(
        loaded.calibration.transform_cm_to_px(calibration.transform_px_to_cm(points)),
        points,
    )


def test_snapshot_owns_arrays_nested_metadata_and_calibration():
    vertices = np.array([[1.0, 2.0], [3.0, 4.0]])
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    metadata = {"labels": [{"value": "original"}]}
    matrix = np.diag([0.5, -0.5, 1.0])
    calibration = VideoCalibration(AffineND(matrix), (20, 10))
    shape = AnnotationShape(vertices, metadata=metadata)
    project = AnnotationProject(frame, shapes=[shape], calibration=calibration)
    vertices[:] = 100
    frame[:] = 100
    metadata["labels"][0]["value"] = "changed"
    matrix[0, 0] = 50
    with pytest.raises(FrozenInstanceError):
        calibration.frame_size_px = (1, 1)
    assert_array_equal(shape.vertices, [[1, 2], [3, 4]])
    assert project.frame.max() == 0
    assert shape.metadata["labels"][0]["value"] == "original"
    assert project.calibration.frame_size_px == (20, 10)
    assert project.calibration.transform_px_to_cm.A[0, 0] == 0.5
    with pytest.raises(TypeError):
        shape.metadata["labels"][0]["value"] = "changed"
    with pytest.raises(FrozenInstanceError):
        project.frame_index = 5


@pytest.mark.parametrize(
    "vertices",
    [
        np.empty((0, 2)),
        np.array([[1, 2]]),
        np.array([[0, 0], [10, 10], [0, 10], [10, 0]]),
    ],
)
def test_unfinished_or_invalid_geometry_remains_editable(tmp_path, vertices):
    shape = AnnotationShape(vertices, identifier="draft", name="unfinished")
    path = tmp_path / "draft"
    AnnotationProject(np.zeros((12, 12, 3), dtype=np.uint8), shapes=[shape]).save(path)
    loaded = AnnotationProject.load(path)
    assert_array_equal(loaded.shapes[0].vertices, vertices)
    assert loaded.shapes[0].identifier == "draft"


@pytest.mark.parametrize(
    "shapes,match",
    [
        (
            [
                AnnotationShape(np.zeros((1, 2)), identifier="same"),
                AnnotationShape(np.zeros((2, 2)), identifier="same"),
            ],
            "unique",
        ),
        (
            [AnnotationShape(np.zeros((1, 2)), identifier="child", parent_id="absent")],
            "missing parent",
        ),
        (
            [AnnotationShape(np.zeros((1, 2)), identifier="self", parent_id="self")],
            "top-level",
        ),
    ],
)
def test_broken_ring_associations_are_rejected(shapes, match):
    with pytest.raises(ValueError, match=match):
        AnnotationProject(np.zeros((2, 2, 3), dtype=np.uint8), shapes=shapes)


def test_shapes_compare_by_value_including_missing_metadata():
    vertices = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0]])
    shape = AnnotationShape(vertices, identifier="ring", metadata={"depth": np.nan})
    same = AnnotationShape(
        vertices.copy(), identifier="ring", metadata={"depth": np.nan}
    )
    assert shape == same
    assert hash(shape) == hash(same)
    assert shape != AnnotationShape(vertices + 1, identifier="ring")
    assert shape != AnnotationShape(vertices, identifier="ring", name="renamed")
    assert shape == pickle.loads(pickle.dumps(shape))


@pytest.mark.parametrize("metadata", [{"callable": lambda: 1}, {3: "coerced"}])
def test_non_json_metadata_cannot_enter_a_snapshot(metadata):
    with pytest.raises(ValueError, match="Metadata"):
        AnnotationShape(np.zeros((3, 2)), metadata=metadata)


def test_non_finite_metadata_survives_save_and_load(tmp_path):
    # Region metadata allows NaN for a missing value; a draft must keep it.
    shape = AnnotationShape(
        np.zeros((3, 2)),
        metadata={"missing": np.nan, "limits": [-np.inf, np.inf]},
    )
    path = tmp_path / "draft.nsp"
    AnnotationProject(np.zeros((4, 4, 3), dtype=np.uint8), shapes=[shape]).save(path)
    metadata = AnnotationProject.load(path).shapes[0].metadata
    assert np.isnan(metadata["missing"])
    assert tuple(metadata["limits"]) == (-np.inf, np.inf)


@pytest.mark.parametrize(
    "matrix,size,match",
    [
        (np.eye(3), (9, 8), "frame_size_px"),
        (np.zeros((3, 3)), (10, 8), "affine"),
        (np.diag([0, 1, 1]), (10, 8), "invertible"),
        (np.array([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]]), (10, 8), "finite"),
        (np.eye(3), (10.0, 8), "frame_size_px"),
    ],
)
def test_calibration_matches_image_and_invertible_affine(matrix, size, match):
    with pytest.raises(ValueError, match=match):
        AnnotationProject(
            np.zeros((8, 10, 3), dtype=np.uint8),
            calibration=VideoCalibration(AffineND(matrix), size),
        )


def _rewrite_archive(path, change):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    document = json.loads(arrays["metadata"].tobytes())
    change(document, arrays)
    arrays["metadata"] = np.frombuffer(json.dumps(document).encode(), dtype=np.uint8)
    with path.open("wb") as stream:
        np.savez(stream, **arrays)


@pytest.mark.parametrize("version", [0, 2, True, "1", None])
def test_unknown_schema_version_is_explicit(tmp_path, version):
    path = tmp_path / "draft"
    AnnotationProject(np.zeros((2, 2, 3), dtype=np.uint8)).save(path)
    _rewrite_archive(path, lambda document, arrays: document.update(version=version))
    with pytest.raises(ValueError, match="Unsupported annotation project version"):
        AnnotationProject.load(path)


@pytest.mark.parametrize(
    "corruption",
    [
        "truncated",
        "missing_frame",
        "bad_json",
        "object_array",
        "bad_shape",
        "extra_field",
        "bad_calibration",
    ],
)
def test_corrupt_archives_fail_with_actionable_context(tmp_path, corruption):
    path = tmp_path / "draft"
    AnnotationProject(
        np.zeros((2, 2, 3), dtype=np.uint8), shapes=[AnnotationShape(np.zeros((3, 2)))]
    ).save(path)
    if corruption == "truncated":
        path.write_bytes(b"broken archive")
    else:

        def change(document, arrays):
            if corruption == "missing_frame":
                del arrays["frame"]
            elif corruption == "object_array":
                arrays["shape_0"] = np.array([object()], dtype=object)
            elif corruption == "bad_shape":
                arrays["shape_0"] = np.ones((2, 3))
            elif corruption == "extra_field":
                document["unrecognized"] = True
            elif corruption == "bad_calibration":
                document["calibration"] = {
                    "frame_size_px": [2, 2],
                    "transform_px_to_cm": [[1]],
                }

        _rewrite_archive(path, change)
        if corruption == "bad_json":
            with path.open("wb") as stream:
                np.savez(stream, metadata=np.frombuffer(b"{", dtype=np.uint8))
    with pytest.raises(ValueError, match="Corrupt annotation project"):
        AnnotationProject.load(path)


def test_failed_save_keeps_existing_project_and_removes_temporary_file(
    tmp_path, monkeypatch
):
    path = tmp_path / "draft"
    project = AnnotationProject(np.zeros((2, 2, 3), dtype=np.uint8))
    project.save(path)
    original = path.read_bytes()

    def fail(stream, **arrays):
        stream.write(b"partial archive")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="disk full"):
        project.save(path, overwrite=True)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_save_refuses_an_existing_file_unless_overwrite(tmp_path):
    path = tmp_path / "draft.nsp"
    path.write_bytes(b"another session's work")
    project = AnnotationProject(np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(FileExistsError, match="overwrite=True"):
        project.save(path)
    assert path.read_bytes() == b"another session's work"
    project.save(path, overwrite=True)
    assert AnnotationProject.load(path).frame.shape == (2, 2, 3)


@pytest.mark.parametrize(
    "clone", [deepcopy, lambda value: pickle.loads(pickle.dumps(value))]
)
def test_copied_project_keeps_owned_immutable_data(clone):
    original = AnnotationProject(
        np.zeros((4, 5, 3), dtype=np.uint8),
        shapes=[
            AnnotationShape(
                np.array([[1, 2], [3, 4]]), metadata={"nested": [{"trial": 3}]}
            )
        ],
        calibration=VideoCalibration(AffineND(np.eye(3)), (5, 4)),
    )
    copied = clone(original)
    assert copied is not original
    assert not copied.frame.flags.writeable
    assert not copied.shapes[0].vertices.flags.writeable
    assert not np.shares_memory(copied.frame, original.frame)
    assert not np.shares_memory(copied.shapes[0].vertices, original.shapes[0].vertices)
    assert copied.shapes[0].metadata["nested"][0]["trial"] == 3
    with pytest.raises(TypeError):
        copied.shapes[0].metadata["nested"][0]["trial"] = 4


def test_display_preflight_allows_native_incomplete_and_complete_cardinalities():
    from neurospatial.annotation._project_layer import _validate_shapes_for_display

    for shape_type, counts in {
        "polygon": (2, 3, 5),
        "path": (2, 3, 5),
        "line": (2,),
        "rectangle": (2, 4),
        "ellipse": (2, 4),
    }.items():
        for count in counts:
            _validate_shapes_for_display(
                [AnnotationShape(np.zeros((count, 2)), shape_type=shape_type)]
            )
