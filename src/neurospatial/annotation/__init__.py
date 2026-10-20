"""Video annotation tools for defining environments and regions."""

from neurospatial.annotation._boundary_inference import (
    BoundaryConfig,
    boundary_from_positions,
)
from neurospatial.annotation._types import (
    AnnotationConfig,
    MultipleBoundaryStrategy,
    RegionType,
)
from neurospatial.annotation.core import (
    AnnotationResult,
    annotate_project,
    annotate_video,
)
from neurospatial.annotation.io import (
    regions_from_cvat,
    regions_from_labelme,
)
from neurospatial.annotation.project import AnnotationProject, AnnotationShape
from neurospatial.annotation.segmentation import (
    RegionGrowProvider,
    SegmentAnythingProvider,
    SegmentationPrompt,
    SegmentationProposal,
    SegmentationProvider,
    mask_to_polygons,
)
from neurospatial.annotation.track_graph import (
    TrackGraphResult,
    annotate_track_graph,
)
from neurospatial.annotation.validation import validate_annotations

__all__ = [
    "AnnotationConfig",
    "AnnotationProject",
    "AnnotationResult",
    "AnnotationShape",
    "BoundaryConfig",
    "MultipleBoundaryStrategy",
    "RegionGrowProvider",
    "RegionType",
    "SegmentAnythingProvider",
    "SegmentationPrompt",
    "SegmentationProposal",
    "SegmentationProvider",
    "TrackGraphResult",
    "annotate_project",
    "annotate_track_graph",
    "annotate_video",
    "boundary_from_positions",
    "mask_to_polygons",
    "regions_from_cvat",
    "regions_from_labelme",
    "validate_annotations",
]
