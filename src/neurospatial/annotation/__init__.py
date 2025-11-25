"""Video annotation tools for defining environments and regions."""

from neurospatial.annotation._boundary_inference import (
    BoundaryConfig,
    boundary_from_positions,
)
from neurospatial.annotation._types import MultipleBoundaryStrategy, Role
from neurospatial.annotation.core import AnnotationResult, annotate_video
from neurospatial.annotation.io import (
    regions_from_cvat,
    regions_from_labelme,
)
from neurospatial.annotation.validation import validate_annotations

__all__ = [
    "AnnotationResult",
    "BoundaryConfig",
    "MultipleBoundaryStrategy",
    "Role",
    "annotate_video",
    "boundary_from_positions",
    "regions_from_cvat",
    "regions_from_labelme",
    "validate_annotations",
]
