"""Select prompted foreground components without changing their geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from neurospatial.annotation.segmentation import (
    SegmentationPrompt,
    SegmentationProposal,
)


@dataclass(frozen=True)
class _CleanedOutline:
    proposal: SegmentationProposal
    removed_components: int
    missed_inside_points: int


def remove_unprompted_fragments(
    proposal: SegmentationProposal, prompt: SegmentationPrompt
) -> _CleanedOutline:
    """Keep every component containing an inside click, including its holes.

    Four-neighbor connectivity matches pixel-cell polygon conversion: pixels
    meeting only at a corner belong to separate polygons. This operation never
    fills holes, bridges gaps, or smooths retained boundaries. If the prediction
    misses every inside point, leave it intact for correction rather than
    guessing which component the user meant.
    """
    points = prompt.points[prompt.labels == 1]
    if not len(points):
        return _CleanedOutline(proposal, 0, 0)
    height, width = proposal.mask.shape
    if (
        (points < 0).any()
        or (points[:, 0] >= width).any()
        or (points[:, 1] >= height).any()
    ):
        raise ValueError("Inside points must lie within the preview image.")
    # Prompts use (x, y), with integers at pixel centers. Clamp rounding at
    # the image edge, consistent with the providers' [0, width) input bounds.
    pixels = np.minimum(np.floor(points + 0.5).astype(int), [width - 1, height - 1])
    components, count = ndimage.label(proposal.mask)
    clicked = components[pixels[:, 1], pixels[:, 0]]
    missed = int(np.count_nonzero(clicked == 0))
    keep = np.unique(clicked[clicked != 0])
    if not len(keep) or len(keep) == count:
        return _CleanedOutline(proposal, 0, missed)

    mask = np.isin(components, keep)
    removed = int(count - len(keep))
    metadata = proposal.metadata
    metadata["annotation_cleanup"] = {
        "method": "inside_points",
        "connectivity": 4,
        "removed_components": removed,
        "removed_pixels": int(np.count_nonzero(proposal.mask & ~mask)),
    }
    return _CleanedOutline(
        SegmentationProposal(mask, score=proposal.score, metadata=metadata),
        removed,
        missed,
    )
