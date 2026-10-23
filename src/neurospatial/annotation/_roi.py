"""Pixel-preserving inference crops for assisted annotation.

Bounds are integer ``(x0, y0, x1, y1)`` with exclusive upper limits.
Only inference inputs move into crop coordinates; annotations stay in the
original frame. No resizing or interpolation is performed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from neurospatial.annotation._mask_cleanup import (
    _CleanedOutline,
    remove_unprompted_fragments,
)
from neurospatial.annotation.segmentation import (
    SegmentationPrompt,
    SegmentationProposal,
    _inside_pixels,
    _validate_inputs,
)

ROIBounds = tuple[int, int, int, int]


def _frame_size(frame_shape: Sequence[int]) -> tuple[int, int]:
    if (
        len(frame_shape) < 2
        or any(isinstance(value, (bool, np.bool_)) for value in frame_shape[:2])
        or any(not isinstance(value, (int, np.integer)) for value in frame_shape[:2])
        or min(frame_shape[:2]) <= 0
    ):
        raise ValueError("Frame shape must provide positive integer height and width.")
    return int(frame_shape[0]), int(frame_shape[1])


def validate_roi_bounds(bounds: ROIBounds, frame_shape: Sequence[int]) -> ROIBounds:
    """Validate a nonempty half-open crop entirely inside the original frame."""
    height, width = _frame_size(frame_shape)
    if len(bounds) != 4 or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in bounds
    ):
        raise ValueError(
            "Region of interest needs integer (x0, y0, x1, y1) pixel bounds."
        )
    x0, y0, x1, y1 = (int(value) for value in bounds)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(
            f"Region of interest must be nonempty and inside the {width} by "
            f"{height} image; redraw it or choose Use full image."
        )
    return x0, y0, x1, y1


def rectangle_to_roi_bounds(
    vertices_rc: NDArray[np.float64], frame_shape: Sequence[int]
) -> ROIBounds:
    """Enclose a napari rectangle in clipped integer array-slice bounds.

    Napari supplies four ``(row, column)`` corners. Integer coordinates are
    pixel centers, so pixel ``i`` covers ``[i - 0.5, i + 0.5]``; the bounds
    include every pixel that the rectangle overlaps with positive area.
    Rectangles entirely outside the image and incomplete draws are errors.
    """
    height, width = _frame_size(frame_shape)
    vertices = np.asarray(vertices_rc, dtype=np.float64)
    if vertices.shape != (4, 2) or not np.isfinite(vertices).all():
        raise ValueError(
            "Finish drawing one region-of-interest rectangle, or choose Use full image."
        )
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    if (maximum <= minimum).any():
        raise ValueError(
            "Region of interest must have positive width and height; "
            "redraw it or choose Use full image."
        )
    lower = np.floor(minimum + 0.5)
    upper = np.ceil(maximum + 0.5)
    bounds = (
        int(np.clip(lower[1], 0, width)),
        int(np.clip(lower[0], 0, height)),
        int(np.clip(upper[1], 0, width)),
        int(np.clip(upper[0], 0, height)),
    )
    return validate_roi_bounds(bounds, frame_shape)


def roi_context_key(
    frame_shape: Sequence[int], bounds: ROIBounds | None
) -> tuple[int, int, int, int, int, int]:
    """Identify inference geometry, including its original-frame offset.

    Equal-sized crops at different offsets must not share SAM refinement state.
    Callers also invalidate state when the image content or ROI selection changes.
    """
    height, width = _frame_size(frame_shape)
    x0, y0, x1, y1 = (
        (0, 0, width, height)
        if bounds is None
        else validate_roi_bounds(bounds, frame_shape)
    )
    return height, width, x0, y0, x1, y1


def crop_segmentation_input(
    image: NDArray[np.uint8],
    prompt: SegmentationPrompt,
    *,
    bounds: ROIBounds | None,
) -> tuple[NDArray[np.uint8], SegmentationPrompt]:
    """Crop original RGB pixels and translate every prompt without scaling.

    Foreground and background clicks must all lie inside the active crop. A
    prompt box must also be contained; no prompt information is silently lost.
    Returned image pixels are a view of the original input when possible.
    """
    image = _validate_inputs(image, prompt)
    if bounds is None:
        return image, prompt
    validated = validate_roi_bounds(bounds, image.shape)
    local_prompt = _local_roi_prompt(prompt, validated)
    x0, y0, x1, y1 = validated
    return image[y0:y1, x0:x1], local_prompt


def _local_roi_prompt(
    prompt: SegmentationPrompt, bounds: ROIBounds
) -> SegmentationPrompt:
    """Validate containment and subtract a previously validated crop offset."""
    x0, y0, x1, y1 = bounds
    points = prompt.points
    if not _inside_pixels(points, bounds):
        raise ValueError(
            "An inside or outside prompt lies outside the region of interest. "
            "Move or remove that point, enlarge the region, or choose Use full image."
        )
    local_box = None
    if prompt.box is not None:
        bx0, by0, bx1, by1 = prompt.box
        if bx0 < x0 - 0.5 or by0 < y0 - 0.5 or bx1 > x1 - 0.5 or by1 > y1 - 0.5:
            raise ValueError(
                "The prompt box extends outside the region of interest. "
                "Enlarge the region or choose Use full image."
            )
        local_box = prompt.box - np.array([x0, y0, x0, y0])
    return SegmentationPrompt(points - np.array([x0, y0]), prompt.labels, box=local_box)


def restore_roi_proposal(
    proposal: SegmentationProposal,
    frame_shape: Sequence[int],
    *,
    bounds: ROIBounds | None,
) -> SegmentationProposal:
    """Restore a crop proposal at its exact offset, with zero foreground outside."""
    height, width = _frame_size(frame_shape)
    x0, y0, x1, y1 = (
        (0, 0, width, height)
        if bounds is None
        else validate_roi_bounds(bounds, frame_shape)
    )
    expected = (y1 - y0, x1 - x0)
    if proposal.mask.shape != expected:
        raise ValueError(
            f"Provider returned mask shape {proposal.mask.shape}; the inference "
            f"image requires {expected}. Return one mask pixel per input pixel."
        )
    if bounds is None:
        return proposal
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = proposal.mask
    metadata = proposal.metadata
    metadata["roi_bounds"] = [x0, y0, x1, y1]
    return SegmentationProposal(mask, score=proposal.score, metadata=metadata)


def clean_roi_fragments(
    proposal: SegmentationProposal,
    prompt: SegmentationPrompt,
    *,
    bounds: ROIBounds | None,
) -> _CleanedOutline:
    """Clean a full-frame proposal using only components inside the crop.

    Inside points must lie in the crop, as for inference. Bounds are explicit;
    provenance metadata never determines geometry.
    """
    if bounds is None:
        return remove_unprompted_fragments(proposal, prompt)
    validated = validate_roi_bounds(bounds, proposal.mask.shape)
    x0, y0, x1, y1 = validated
    local_prompt = _local_roi_prompt(prompt, validated)
    local_proposal = SegmentationProposal(
        proposal.mask[y0:y1, x0:x1],
        score=proposal.score,
        metadata=proposal.metadata,
    )
    cleaned = remove_unprompted_fragments(local_proposal, local_prompt)
    return _CleanedOutline(
        restore_roi_proposal(cleaned.proposal, proposal.mask.shape, bounds=validated),
        cleaned.removed_components,
        cleaned.missed_inside_points,
    )
