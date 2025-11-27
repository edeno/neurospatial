"""Type definitions for annotation module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# RegionType: Type of annotation shape
# - "environment": Primary boundary polygon defining the spatial environment
# - "hole": Excluded areas within environment boundary (subtracted)
# - "region": Named regions of interest (ROIs)
RegionType = Literal["environment", "hole", "region"]

# Strategy for handling multiple environment boundaries
# - "last": Use the last drawn boundary (default, current behavior)
# - "first": Use the first drawn boundary
# - "error": Raise an error if multiple boundaries are drawn
MultipleBoundaryStrategy = Literal["last", "first", "error"]


@dataclass(frozen=True)
class AnnotationConfig:
    """
    Configuration for annotation UI settings.

    Groups UI-related parameters for `annotate_video()` to reduce function
    signature complexity. All fields have sensible defaults, so the simplest
    usage is ``AnnotationConfig()``.

    Parameters
    ----------
    frame_index : int
        Which video frame to display for annotation. Default is 0 (first frame).
    simplify_tolerance : float or None
        Tolerance for polygon simplification using Douglas-Peucker algorithm.
        Removes vertices that deviate less than this distance from the
        simplified line. Units depend on calibration (cm with calibration,
        pixels without). Default is None (no simplification).

        Recommended values:

        - For cm: 1.0-2.0 (removes hand-drawn jitter)
        - For pixels: 2.0-5.0
    multiple_boundaries : {"last", "first", "error"}
        How to handle multiple environment boundaries:

        - "last": Use the last drawn boundary (default). A warning is emitted.
        - "first": Use the first drawn boundary. A warning is emitted.
        - "error": Raise ValueError if multiple boundaries are drawn.
    show_positions : bool
        If True and initial_boundary is an array of positions, show the
        positions as a semi-transparent Points layer for reference while
        editing. Default is False.

    Examples
    --------
    >>> config = AnnotationConfig(frame_index=100, simplify_tolerance=1.5)
    >>> config.frame_index
    100

    >>> # Use with annotate_video
    >>> result = annotate_video(
    ...     "experiment.mp4",
    ...     bin_size=2.0,
    ...     config=AnnotationConfig(simplify_tolerance=1.0),
    ... )  # doctest: +SKIP

    See Also
    --------
    annotate_video : Main annotation function that accepts this config.
    BoundaryConfig : Configuration for boundary inference from positions.
    """

    frame_index: int = 0
    simplify_tolerance: float | None = None
    multiple_boundaries: MultipleBoundaryStrategy = "last"
    show_positions: bool = False
