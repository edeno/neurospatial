"""Compare the optional CVAT boundary path to the upstream OpenCV algorithm."""

import math
import pickle
import subprocess
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from shapely.geometry import Polygon

from neurospatial.annotation._cvat_polygon import (
    CVAT_UPSTREAM_REF,
    CvatContour,
    cvat_contour,
    cvat_tolerance,
    simplify_cvat_contour,
)
from neurospatial.annotation._polygon_outline import polygon_outline
from neurospatial.annotation.segmentation import SegmentationProposal


@pytest.fixture
def require_opencv():
    """Keep the optional comparison tests usable without the video extra."""
    return pytest.importorskip("cv2")


def _oracle(mask, accuracy):
    """Literal CVAT OpenCV sequence, independent of production helpers.

    cvat-core/src/opencv/opencv-interface.ts at CVAT_UPSTREAM_REF:
    contours.findContours, contours.approxPoly and utils.thresholdFromAccuracy.
    """
    cv2 = pytest.importorskip("cv2")
    expanded = cv2.copyMakeBorder(
        mask.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
    )
    expanded = cv2.dilate(
        expanded,
        np.ones((2, 2), dtype=np.uint8),
        anchor=(-1, -1),
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
    )
    contours, _ = cv2.findContours(expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    converted = []
    for contour in contours:
        points = contour.reshape(-1, 2) - 1
        previous_x, previous_y = points[0]
        length = 0.0
        for x, y in points:
            length += math.hypot(x - previous_x, y - previous_y)
            previous_x, previous_y = x, y
        converted.append((length, points))
    converted.sort(key=lambda item: item[0], reverse=True)
    distance = 13 - accuracy
    epsilon = (
        0.0
        if distance == 0
        else (2.75 * distance - 1) / 7
        if distance <= 8
        else 2 ** (distance - 7)
    )
    points = converted[0][1].astype(np.float32)
    approximated = cv2.approxPolyDP(points, epsilon, True).reshape(-1, 2)
    return points, approximated, len(contours)


@pytest.mark.parametrize(
    ("accuracy", "epsilon"),
    [
        (0, 64),
        (1, 32),
        (2, 16),
        (3, 8),
        (4, 4),
        (5, 3),
        (6, 73 / 28),
        (7, 31 / 14),
        (8, 51 / 28),
        (9, 10 / 7),
        (10, 29 / 28),
        (11, 9 / 14),
        (12, 0.25),
        (13, 0),
    ],
)
def test_exact_cvat_epsilon_mapping(accuracy, epsilon):
    assert cvat_tolerance(accuracy) == epsilon
    assert cvat_tolerance(np.int64(accuracy)) == epsilon


@pytest.mark.parametrize(
    "accuracy", [-1, 14, 9.0, True, np.bool_(False), "9", None, np.nan]
)
def test_invalid_accuracy_is_actionable(accuracy):
    with pytest.raises(ValueError, match="integer between 0 and 13"):
        cvat_tolerance(accuracy)


@pytest.mark.parametrize("accuracy", [5, 9, 12, 13])
def test_full_pipeline_matches_small_cvat_oracle(accuracy):
    mask = np.zeros((80, 100), dtype=bool)
    for row in range(3, 70):
        mask[row, 2 : 58 + row // 4] = True
    mask[20:40, 20:30] = False
    mask[74:78, 90:96] = True
    proposal = SegmentationProposal(mask)
    expected_contour, expected_polygon, count = _oracle(mask, accuracy)
    contour = cvat_contour(proposal)
    outline = simplify_cvat_contour(contour, accuracy=accuracy)
    assert_array_equal(contour.points, expected_contour)
    assert_array_equal(
        np.array(outline.polygons[0].exterior.coords)[:-1], expected_polygon
    )
    assert contour.contour_count == count
    assert outline.simplify_tolerance_px == cvat_tolerance(accuracy)


@pytest.mark.usefixtures("require_opencv")
def test_largest_perimeter_wins_even_when_area_is_smaller():
    mask = np.zeros((35, 60), dtype=bool)
    mask[2:12, 2:12] = True  # larger area, shorter contour
    mask[25:26, 25:55] = True
    contour = cvat_contour(SegmentationProposal(mask))
    outline = simplify_cvat_contour(contour, accuracy=13)
    assert outline.polygons[0].bounds == (25, 25, 55, 26)
    assert contour.source_component_count == 2
    assert contour.contour_count == 2
    assert contour.dropped_contour_count == 1


def test_equal_perimeter_keeps_opencv_order():
    mask = np.zeros((30, 30), dtype=bool)
    mask[2:9, 2:9] = True
    mask[20:27, 20:27] = True
    expected, _, _ = _oracle(mask, 13)
    assert_array_equal(cvat_contour(SegmentationProposal(mask)).points, expected)


@pytest.mark.usefixtures("require_opencv")
def test_integer_boundary_and_unclipped_border_differ_from_pixel_cells():
    proposal = SegmentationProposal(np.ones((3, 4), dtype=bool))
    outline = simplify_cvat_contour(cvat_contour(proposal), accuracy=13)
    assert outline.polygons[0].bounds == (0, 0, 4, 3)
    assert polygon_outline(proposal).polygons[0].bounds == (-0.5, -0.5, 3.5, 2.5)
    assert_array_equal(outline.rings[0], np.floor(outline.rings[0]))


@pytest.mark.usefixtures("require_opencv")
def test_holes_and_disconnected_fragments_are_discarded_only_in_polygon():
    mask = np.zeros((40, 50), dtype=bool)
    mask[2:30, 2:30] = True
    mask[10:20, 10:20] = False
    mask[35:38, 40:45] = True
    proposal = SegmentationProposal(mask)
    contour = cvat_contour(proposal)
    outline = simplify_cvat_contour(contour, accuracy=13)
    assert contour.source_component_count == 2
    assert contour.source_hole_count == 1
    assert len(outline.polygons) == 1
    assert len(outline.polygons[0].interiors) == 0
    assert_array_equal(outline.proposal.mask, mask)
    assert not outline.proposal.mask[15, 15]
    assert outline.proposal.mask[36, 42]


@pytest.mark.usefixtures("require_opencv")
def test_dilation_can_merge_fragments_without_extra_bridging():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 1:3] = True
    mask[2:5, 4:6] = True
    contour = cvat_contour(SegmentationProposal(mask))
    assert contour.source_component_count == 2
    assert contour.contour_count == 1
    assert contour.dropped_contour_count == 0


@pytest.mark.usefixtures("require_opencv")
def test_slider_reuses_original_contour_without_boundary_processing(monkeypatch):
    mask = np.zeros((60, 60), dtype=bool)
    for row in range(3, 55):
        mask[row, 3 : 25 + row // 4] = True
    contour = cvat_contour(SegmentationProposal(mask))
    original = contour.points.copy()
    cv2 = pytest.importorskip("cv2")

    def fail(*args, **kwargs):
        pytest.fail("Changing accuracy must not repeat dilation or contour extraction")

    monkeypatch.setattr(cv2, "dilate", fail)
    monkeypatch.setattr(cv2, "findContours", fail)
    precise = simplify_cvat_contour(contour, accuracy=13)
    rough = simplify_cvat_contour(contour, accuracy=5)
    repeated = simplify_cvat_contour(contour, accuracy=13)
    assert rough.vertex_count < precise.vertex_count
    assert repeated.polygons[0].equals_exact(precise.polygons[0], 0)
    assert_array_equal(contour.points, original)


@pytest.mark.usefixtures("require_opencv")
def test_owned_inputs_provider_provenance_and_conversion_metadata():
    mask = np.ones((20, 20), dtype=bool)
    metadata = {"provider": "fake_sam", "nested": {"selected": [2]}}
    proposal = SegmentationProposal(mask, score=0.85, metadata=metadata)
    contour = cvat_contour(proposal)
    outline = simplify_cvat_contour(contour)
    mask[:] = False
    metadata["nested"]["selected"].append(3)
    assert contour.proposal is proposal
    assert proposal.metadata == {"provider": "fake_sam", "nested": {"selected": [2]}}
    result = outline.proposal.metadata
    assert result["provider"] == "fake_sam"
    assert result["nested"] == {"selected": [2]}
    assert result["annotation_style"] == "cvat"
    assert result["cvat_polygon"]["accuracy"] == 9
    assert result["cvat_polygon"]["epsilon_px"] == 10 / 7
    assert result["cvat_polygon"]["upstream_ref"] == CVAT_UPSTREAM_REF
    assert outline.proposal.score == 0.85
    assert outline.proposal.mask.all()
    result["cvat_polygon"]["accuracy"] = 1
    assert outline.proposal.metadata["cvat_polygon"]["accuracy"] == 9
    for owned in (contour, deepcopy(contour), pickle.loads(pickle.dumps(contour))):
        with pytest.raises(ValueError, match="read-only"):
            owned.points[0, 0] = 100
        with pytest.raises(FrozenInstanceError):
            owned.contour_count = 100


@pytest.mark.parametrize(
    "metadata",
    [{"annotation_style": "provider_style"}, {"cvat_polygon": {"provider": True}}],
)
@pytest.mark.usefixtures("require_opencv")
def test_conflicting_owned_keys_cannot_silently_overwrite_provider_metadata(metadata):
    proposal = SegmentationProposal(np.ones((20, 20), dtype=bool), metadata=metadata)
    with pytest.raises(ValueError, match=r"conflicting.*provenance"):
        simplify_cvat_contour(cvat_contour(proposal))
    assert proposal.metadata == metadata


@pytest.mark.usefixtures("require_opencv")
def test_tiny_object_can_collapse_and_higher_accuracy_recovers_exact_cvat_boundary():
    contour = cvat_contour(SegmentationProposal(np.ones((1, 1), dtype=bool)))
    with pytest.raises(ValueError, match=r"Adjust accuracy.*topology-preserving"):
        simplify_cvat_contour(contour)
    assert simplify_cvat_contour(contour, accuracy=13).polygons[0].area == 1


@pytest.mark.usefixtures("require_opencv")
def test_invalid_contour_raises_without_repair_or_fallback():
    points = np.array([(0, 0), (4, 4), (0, 4), (4, 0)], dtype=np.float32)
    contour = CvatContour(SegmentationProposal(np.ones((5, 5), bool)), points, 1, 1, 0)
    assert not Polygon(points).is_valid
    with pytest.raises(ValueError, match=r"invalid geometry.*Adjust accuracy"):
        simplify_cvat_contour(contour, accuracy=13)
    assert_array_equal(contour.points, points)


def test_missing_opencv_error_is_optional_and_actionable(monkeypatch):
    import neurospatial.annotation._cvat_polygon as module

    def unavailable(name):
        assert name == "cv2"
        raise ModuleNotFoundError("No cv2")

    monkeypatch.setattr(module.importlib, "import_module", unavailable)
    with pytest.raises(ImportError, match=r"requires OpenCV.*neurospatial\[video\]"):
        cvat_contour(SegmentationProposal(np.ones((3, 3), dtype=bool)))
    assert cvat_tolerance(9) == 10 / 7


def test_module_import_does_not_load_optional_opencv():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import neurospatial.annotation._cvat_polygon; "
            "assert 'cv2' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_cached_points_are_owned_and_keep_javascript_number_precision():
    proposal = SegmentationProposal(np.ones((3, 3), dtype=bool))
    offset = 2**24
    points = np.array(
        [(offset, 0), (offset + 1, 0), (offset + 1, 1), (offset, 1)],
        dtype=np.float64,
    )
    contour = CvatContour(proposal, points, 1, 1, 0)
    points[:] = 0
    assert contour.points[1, 0] - contour.points[0, 0] == 1
    assert contour.points.dtype == np.float64
