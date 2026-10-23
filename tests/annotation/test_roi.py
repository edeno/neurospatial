"""Inference crops preserve original pixels, coordinates and provenance."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial.annotation._roi import (
    clean_roi_fragments,
    crop_segmentation_input,
    rectangle_to_roi_bounds,
    restore_roi_proposal,
    validate_roi_bounds,
)
from neurospatial.annotation.segmentation import (
    RegionGrowProvider,
    SegmentationPrompt,
    SegmentationProposal,
)


def test_crop_preserves_rgb_and_exact_subpixel_offsets():
    image = np.arange(7 * 13 * 3, dtype=np.uint8).reshape(7, 13, 3)
    prompt = SegmentationPrompt(
        np.array([[4.25, 2.5], [9.25, 5.4]]),
        np.array([1, 0]),
        box=np.array([4.0, 2.0, 9.0, 5.0]),
    )
    crop, local = crop_segmentation_input(image, prompt, bounds=(4, 2, 10, 6))
    assert crop.shape == (4, 6, 3)
    assert np.shares_memory(crop, image)
    assert_array_equal(crop, image[2:6, 4:10])
    assert_array_equal(local.points, prompt.points - [4, 2])
    assert_array_equal(local.labels, [1, 0])
    assert_array_equal(local.box, [0, 0, 5, 3])
    assert_array_equal(prompt.points, [[4.25, 2.5], [9.25, 5.4]])


# Integer coordinates are pixel centers: bounds (4, 2, 10, 6) cover
# x in [3.5, 9.5) and y in [1.5, 5.5).
@pytest.mark.parametrize("point", [[3.4, 3], [9.5, 3], [5, 1.4], [5, 5.5]])
@pytest.mark.parametrize("label", [0, 1])
def test_crop_rejects_every_outside_prompt_actionably(point, label):
    prompt = SegmentationPrompt(np.array([point]), np.array([label]))
    with pytest.raises(ValueError, match=r"outside the region.*Move or remove"):
        crop_segmentation_input(
            np.zeros((7, 13, 3), dtype=np.uint8),
            prompt,
            bounds=(4, 2, 10, 6),
        )


def test_crop_rejects_prompt_box_outside_roi():
    prompt = SegmentationPrompt(
        np.empty((0, 2)), np.empty(0), box=np.array([3, 2, 10, 6])
    )
    with pytest.raises(ValueError, match="prompt box extends outside"):
        crop_segmentation_input(
            np.zeros((7, 13, 3), dtype=np.uint8),
            prompt,
            bounds=(4, 2, 10, 6),
        )


def test_restore_mask_preserves_components_holes_score_and_metadata():
    mask = np.array([[1, 0, 1], [1, 1, 0]], dtype=bool)
    proposal = SegmentationProposal(mask, score=0.81, metadata={"source": ["sam"]})
    restored = restore_roi_proposal(proposal, (7, 13), bounds=(5, 3, 8, 5))
    expected = np.zeros((7, 13), dtype=bool)
    expected[3:5, 5:8] = mask
    assert_array_equal(restored.mask, expected)
    assert restored.score == 0.81
    assert restored.metadata == {"source": ["sam"], "roi_bounds": [5, 3, 8, 5]}
    assert proposal.metadata == {"source": ["sam"]}
    assert not restored.mask.flags.writeable


@pytest.mark.parametrize("point", [[3.5, 1.5], [9.49, 5.49]])
def test_crop_accepts_prompts_on_the_outer_half_of_edge_pixels(point):
    prompt = SegmentationPrompt(np.array([point]), np.array([1]))
    _, local = crop_segmentation_input(
        np.zeros((7, 13, 3), dtype=np.uint8), prompt, bounds=(4, 2, 10, 6)
    )
    assert_array_equal(local.points, [np.subtract(point, [4, 2])])


def test_fractional_click_near_crop_upper_edges_selects_last_crop_pixel():
    image = np.zeros((7, 13, 3), dtype=np.uint8)
    image[4, 7] = 255
    prompt = SegmentationPrompt(np.array([[7.4, 4.4]]), np.array([1]))
    bounds = (4, 2, 8, 5)
    crop, local = crop_segmentation_input(image, prompt, bounds=bounds)
    proposal = RegionGrowProvider(tolerance=0).predict(crop, local)
    restored = restore_roi_proposal(proposal, image.shape, bounds=bounds)
    expected = np.zeros(image.shape[:2], dtype=bool)
    expected[4, 7] = True
    assert_array_equal(restored.mask, expected)


def test_roi_cleanup_retains_both_components_at_fractional_crop_edges():
    image = np.zeros((7, 13, 3), dtype=np.uint8)
    image[2, 4] = 255
    image[4, 7] = 255
    prompt = SegmentationPrompt(np.array([[4, 2], [7.4, 4.4]]), np.array([1, 1]))
    bounds = (4, 2, 8, 5)
    crop, local = crop_segmentation_input(image, prompt, bounds=bounds)
    restored = restore_roi_proposal(
        RegionGrowProvider(tolerance=0).predict(crop, local), image.shape, bounds=bounds
    )
    cleaned = clean_roi_fragments(restored, prompt, bounds=bounds)
    assert_array_equal(cleaned.proposal.mask, restored.mask)
    assert cleaned.removed_components == 0
    assert cleaned.missed_inside_points == 0
    assert cleaned.proposal.metadata == restored.metadata


def test_roi_cleanup_preserves_counters_score_and_provenance():
    mask = np.zeros((7, 13), dtype=bool)
    mask[2, 4] = True
    mask[4, 7] = True
    proposal = SegmentationProposal(mask, score=0.9, metadata={"provider": "fake"})
    prompt = SegmentationPrompt(np.array([[7.4, 4.4], [5, 3]]), np.array([1, 1]))
    cleaned = clean_roi_fragments(proposal, prompt, bounds=(4, 2, 8, 5))
    expected = np.zeros_like(mask)
    expected[4, 7] = True
    assert_array_equal(cleaned.proposal.mask, expected)
    assert cleaned.removed_components == 1
    assert cleaned.missed_inside_points == 1
    assert cleaned.proposal.score == 0.9
    assert cleaned.proposal.metadata == {
        "provider": "fake",
        "roi_bounds": [4, 2, 8, 5],
        "annotation_cleanup": {
            "method": "inside_points",
            "connectivity": 4,
            "removed_components": 1,
            "removed_pixels": 1,
        },
    }


def test_full_image_cleanup_ignores_roi_metadata_as_geometry():
    mask = np.zeros((7, 13), dtype=bool)
    mask[2, 4] = True
    proposal = SegmentationProposal(mask, metadata={"roi_bounds": [0, 0, 1, 1]})
    prompt = SegmentationPrompt(np.array([[4, 2]]), np.array([1]))
    cleaned = clean_roi_fragments(proposal, prompt, bounds=None)
    assert cleaned.proposal is proposal
    assert cleaned.removed_components == 0
    assert cleaned.missed_inside_points == 0


def test_restored_polygon_boundaries_follow_original_pixel_centers():
    from neurospatial.annotation.segmentation import mask_to_polygons

    proposal = SegmentationProposal(np.ones((2, 3), dtype=bool))
    restored = restore_roi_proposal(proposal, (7, 13), bounds=(5, 3, 8, 5))
    polygon = mask_to_polygons(restored.mask)[0]
    # Integers mark pixel centers. Foreground cells extend half a pixel either
    # side; ROI provenance records array-slice limits rather than polygon edges.
    assert polygon.bounds == (4.5, 2.5, 7.5, 4.5)
    assert polygon.area == 6
    assert restored.metadata["roi_bounds"] == [5, 3, 8, 5]


@pytest.mark.parametrize("bounds", [None, (0, 0, 13, 7)])
def test_full_frame_edges_keep_coordinates(bounds):
    image = np.zeros((7, 13, 3), dtype=np.uint8)
    prompt = SegmentationPrompt(
        np.array([[-0.5, -0.5], [12.49, 6.49]]), np.array([1, 0])
    )
    crop, local = crop_segmentation_input(image, prompt, bounds=bounds)
    assert_array_equal(crop, image)
    assert_array_equal(local.points, prompt.points)
    proposal = SegmentationProposal(np.ones((7, 13), dtype=bool))
    restored = restore_roi_proposal(proposal, image.shape, bounds=bounds)
    assert_array_equal(restored.mask, proposal.mask)


@pytest.mark.parametrize("bounds", [None, (5, 3, 8, 5)])
def test_mismatched_provider_mask_has_actionable_error(bounds):
    with pytest.raises(ValueError, match="one mask pixel per input pixel"):
        restore_roi_proposal(
            SegmentationProposal(np.ones((3, 2), dtype=bool)),
            (7, 13),
            bounds=bounds,
        )


@pytest.mark.parametrize(
    "bounds", [(0, 0, 0, 2), (-1, 0, 2, 2), (0, 0, 14, 7), (0, 0, 13, 8)]
)
def test_invalid_or_empty_bounds(bounds):
    with pytest.raises(ValueError, match="nonempty and inside"):
        validate_roi_bounds(bounds, (7, 13))


def test_fractional_integer_bounds_are_not_silently_truncated():
    with pytest.raises(ValueError, match="integer"):
        validate_roi_bounds((1.5, 0, 3, 4), (7, 13))


def test_rectangle_includes_partially_covered_edge_pixels():
    # Pixel 21 covers [20.5, 21.5], so a rectangle ending at 20.6 overlaps it;
    # pixel 9 covers [8.5, 9.5] and lies outside a rectangle starting at 9.6.
    vertices = np.array([[9.6, 9.6], [9.6, 20.6], [20.6, 20.6], [20.6, 9.6]])
    bounds = rectangle_to_roi_bounds(vertices, (30, 30))
    assert bounds == (10, 10, 22, 22)
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    image[15, 21] = 255
    prompt = SegmentationPrompt(np.array([[20.55, 15]]), np.array([1]))
    crop, local = crop_segmentation_input(image, prompt, bounds=bounds)
    proposal = RegionGrowProvider(tolerance=0).predict(crop, local)
    restored = restore_roi_proposal(proposal, image.shape, bounds=bounds)
    assert np.argwhere(restored.mask).tolist() == [[15, 21]]


def test_rectangle_corners_at_pixel_centers_cover_both_edge_pixels():
    vertices = np.array([[1, 2], [1, 9], [6, 9], [6, 2]])
    assert rectangle_to_roi_bounds(vertices, (7, 13)) == (2, 1, 10, 7)


def test_napari_rectangle_uses_row_column_and_clips_edges():
    vertices = np.array([[-2.2, 4.1], [-2.2, 15], [5.4, 15], [5.4, 4.1]])
    assert rectangle_to_roi_bounds(vertices, (7, 13)) == (4, 0, 13, 6)


@pytest.mark.parametrize(
    "vertices",
    [
        np.empty((0, 2)),
        np.array([[1, 1], [1, 2], [1, 2], [1, 1]]),
        np.array([[1.2, 1], [1.2, 2], [1.2, 2], [1.2, 1]]),
        np.array([[8, 1], [8, 4], [9, 4], [9, 1]]),
    ],
)
def test_incomplete_empty_or_outside_rectangle_is_invalid(vertices):
    with pytest.raises(ValueError, match="Use full image"):
        rectangle_to_roi_bounds(vertices, (7, 13))
