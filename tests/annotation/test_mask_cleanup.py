"""Preview cleanup preserves explicitly selected regions and real obstacles."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial.annotation._mask_cleanup import remove_unprompted_fragments
from neurospatial.annotation.segmentation import (
    SegmentationPrompt,
    SegmentationProposal,
    mask_to_polygons,
)


def test_cleanup_removes_strays_but_keeps_holes_and_every_clicked_component():
    mask = np.zeros((12, 25), dtype=bool)
    mask[1:9, 12:20] = True
    mask[3:6, 14:17] = False  # A real obstacle must remain a hole.
    mask[9:11, 22:24] = True  # Explicitly selected second region.
    mask[1:8, 1:8] = True  # Larger than the second region, but not selected.
    mask[11, 0] = True
    original = mask.copy()
    proposal = SegmentationProposal(mask, score=0.8, metadata={"model": "sam2"})
    prompt = SegmentationPrompt(
        np.array([[13, 2], [22, 10], [2, 2]]), np.array([1, 1, 0])
    )
    result = remove_unprompted_fragments(proposal, prompt)
    expected = original.copy()
    expected[:, :9] = False
    assert_array_equal(result.proposal.mask, expected)
    assert_array_equal(proposal.mask, original)
    assert_array_equal(mask, original)
    polygons = mask_to_polygons(result.proposal.mask)
    assert len(polygons) == 2
    assert sum(len(p.interiors) for p in polygons) == 1
    assert result.removed_components == 2
    assert result.missed_inside_points == 0
    assert result.proposal.score == 0.8
    assert result.proposal.metadata["annotation_cleanup"] == {
        "method": "inside_points",
        "connectivity": 4,
        "removed_components": 2,
        "removed_pixels": 50,
    }
    assert proposal.metadata == {"model": "sam2"}
    assert not result.proposal.mask.flags.writeable


def test_cleanup_respects_pixel_cell_connectivity_without_bridging_diagonal_gaps():
    mask = np.eye(3, dtype=bool)
    result = remove_unprompted_fragments(
        SegmentationProposal(mask),
        SegmentationPrompt(np.array([[1, 1]]), np.array([1])),
    )
    expected = np.zeros((3, 3), dtype=bool)
    expected[1, 1] = True
    assert_array_equal(result.proposal.mask, expected)
    assert result.removed_components == 2


def test_cleanup_uses_nearest_pixel_center_and_clamps_rounding_at_image_edge():
    mask = np.zeros((4, 8), dtype=bool)
    mask[1, 3] = mask[3, 7] = mask[0, 0] = True
    result = remove_unprompted_fragments(
        SegmentationProposal(mask),
        SegmentationPrompt(np.array([[2.7, 1.2], [7.8, 3.9]]), np.ones(2)),
    )
    expected = mask.copy()
    expected[0, 0] = False
    assert_array_equal(result.proposal.mask, expected)
    assert result.removed_components == 1


@pytest.mark.parametrize("labels", [[1], [0]])
def test_cleanup_leaves_original_when_no_foreground_component_is_prompted(labels):
    mask = np.eye(4, dtype=bool)
    proposal = SegmentationProposal(mask)
    result = remove_unprompted_fragments(
        proposal, SegmentationPrompt(np.array([[3, 0]]), np.array(labels))
    )
    assert result.proposal is proposal
    assert result.removed_components == 0
    assert result.missed_inside_points == labels[0]


def test_cleanup_reports_missed_points_without_removing_any_clicked_region():
    mask = np.eye(4, dtype=bool)
    result = remove_unprompted_fragments(
        SegmentationProposal(mask),
        SegmentationPrompt(np.array([[0, 0], [1, 1], [3, 0]]), np.ones(3)),
    )
    expected = np.zeros((4, 4), dtype=bool)
    expected[0, 0] = expected[1, 1] = True
    assert_array_equal(result.proposal.mask, expected)
    assert result.missed_inside_points == 1


@pytest.mark.parametrize("point", [[-1, 0], [8, 0], [0, 4]])
def test_cleanup_rejects_out_of_bounds_inside_points(point):
    with pytest.raises(ValueError, match="within the preview"):
        remove_unprompted_fragments(
            SegmentationProposal(np.ones((4, 8), dtype=bool)),
            SegmentationPrompt(np.array([point]), np.ones(1)),
        )
