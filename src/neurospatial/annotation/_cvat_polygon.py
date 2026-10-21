"""CVAT-compatible mask boundaries for the optional comparison workflow.

The algorithm follows CVAT's OpenCV interface at ``CVAT_UPSTREAM_REF``.
It deliberately discards holes and every contour except the longest one.
Coordinates describe the dilated mask's pixel centers, unlike the half-pixel
cell boundaries used by the topology-preserving annotation workflow.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from shapely.geometry import Polygon
from shapely.validation import explain_validity

from neurospatial._results import ReadOnlyArraysMixin
from neurospatial.annotation._polygon_outline import PolygonOutline
from neurospatial.annotation.segmentation import SegmentationProposal

CVAT_UPSTREAM_REF = "172ea51a2ffc11adb08b1c9fb757282d142cd810"


def _opencv() -> Any:
    """Import the optional dependency only when conversion is requested."""
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:
        raise ImportError(
            "CVAT-style polygon conversion requires OpenCV. Install "
            "neurospatial[video] or opencv-python, or choose the "
            "topology-preserving polygon mode."
        ) from exc


def cvat_tolerance(accuracy: int) -> float:
    """Convert CVAT's integer accuracy slider (0..13) to pixels of epsilon.

    Higher accuracy keeps more detail. CVAT's default accuracy, 9, uses
    ``10 / 7`` pixels; accuracy 13 uses zero epsilon.
    """
    if (
        isinstance(accuracy, bool | np.bool_)
        or not isinstance(accuracy, int | np.integer)
        or not 0 <= accuracy <= 13
    ):
        raise ValueError("CVAT accuracy must be an integer between 0 and 13.")
    distance = 13 - int(accuracy)
    if distance == 0:
        return 0.0
    if distance <= 8:
        return (2.75 * distance - 1) / 7
    return float(2 ** (distance - 7))


@dataclass(frozen=True)
class CvatContour(ReadOnlyArraysMixin):
    """One cached, unsimplified contour and counts describing discarded support.

    ``points`` are unclosed pixel (x, y) coordinates. Counts of original mask
    components and holes use eight-connected foreground and four-connected
    background, matching OpenCV's external-contour interpretation. Dilation
    may merge foreground components before ``contour_count`` is measured.
    The original proposal remains intact, including its holes and fragments.
    """

    proposal: SegmentationProposal
    points: NDArray[np.float64]
    contour_count: int
    source_component_count: int
    source_hole_count: int

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not len(points)
            or not np.isfinite(points).all()
        ):
            raise ValueError("A cached CVAT contour needs finite pixel (x, y) points.")
        for name in ("contour_count", "source_component_count", "source_hole_count"):
            value = getattr(self, name)
            if (
                isinstance(value, bool | np.bool_)
                or not isinstance(value, int | np.integer)
                or value < (0 if name == "source_hole_count" else 1)
            ):
                raise ValueError("CVAT contour counts must be nonnegative integers.")
            object.__setattr__(self, name, int(value))
        object.__setattr__(self, "points", points)
        self._own_arrays()

    @property
    def dropped_contour_count(self) -> int:
        """Number of external contours discarded after dilation."""
        return self.contour_count - 1


def cvat_contour(proposal: SegmentationProposal) -> CvatContour:
    """Cache CVAT's longest external contour after one 2×2 dilation.

    The one-pixel zero border and default even-kernel anchor are intentional:
    boundaries can extend to the image width/height and are never clipped.
    CVAT ranks contours by consecutive-point distance *without* the closing
    segment. Equal lengths retain OpenCV's original contour order.
    """
    cv2 = _opencv()
    padded = np.pad(proposal.mask.astype(np.uint8), 1)
    expanded = cv2.dilate(
        padded,
        np.ones((2, 2), dtype=np.uint8),
        anchor=(-1, -1),
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
    )
    contours, _ = cv2.findContours(expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError(
            "CVAT conversion found no external contour; adjust the mask or "
            "choose the topology-preserving polygon mode."
        )
    points = [contour.reshape(-1, 2).astype(np.float64) - 1 for contour in contours]
    # Python's sort is stable, as is Array.sort in the upstream JavaScript.
    points.sort(
        key=lambda contour: sum(
            math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in pairwise(contour)
        ),
        reverse=True,
    )
    components = ndimage.label(proposal.mask, structure=np.ones((3, 3)))[1]
    holes = ndimage.binary_fill_holes(proposal.mask) & ~proposal.mask
    hole_count = ndimage.label(holes)[1]
    return CvatContour(proposal, points[0], len(points), components, hole_count)


def simplify_cvat_contour(contour: CvatContour, *, accuracy: int = 9) -> PolygonOutline:
    """Approximate the cached contour using CVAT's closed Douglas–Peucker path.

    Invalid or collapsed geometry raises an actionable error. No repair,
    fallback, clipping or additional connection of fragments is performed.
    The result records the conversion alongside the original provider metadata.
    """
    epsilon = cvat_tolerance(accuracy)
    cv2 = _opencv()
    points = cv2.approxPolyDP(
        np.array(contour.points, dtype=np.float32, copy=True), epsilon, True
    ).reshape(-1, 2)
    failure = (
        "CVAT polygon simplification produced degenerate or invalid geometry. "
        "Adjust accuracy, refine prompts, or choose the topology-preserving polygon mode."
    )
    if len(points) < 3:
        raise ValueError(failure)
    polygon = Polygon(points)
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ValueError(f"{failure} Geometry: {explain_validity(polygon)}")
    metadata = contour.proposal.metadata
    owned_metadata = {
        "annotation_style": "cvat",
        "cvat_polygon": {
            "accuracy": int(accuracy),
            "epsilon_px": epsilon,
            "dilation": {"kernel": [2, 2], "anchor": [-1, -1], "iterations": 1},
            "padding_px": 1,
            "contour_count": contour.contour_count,
            "dropped_contour_count": contour.dropped_contour_count,
            "source_component_count": contour.source_component_count,
            "source_hole_count": contour.source_hole_count,
            "source_foreground_connectivity": 8,
            "source_background_connectivity": 4,
            "upstream_ref": CVAT_UPSTREAM_REF,
        },
    }
    for key, value in owned_metadata.items():
        if key in metadata and metadata[key] != value:
            raise ValueError(
                f"Proposal metadata already contains conflicting {key!r} provenance. "
                "Use the original provider proposal for CVAT conversion."
            )
    metadata.update(owned_metadata)
    proposal = SegmentationProposal(
        contour.proposal.mask, score=contour.proposal.score, metadata=metadata
    )
    return PolygonOutline(proposal, (polygon,), epsilon)
