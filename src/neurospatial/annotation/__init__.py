"""Video annotation tools for defining environments and regions."""

from neurospatial.annotation.core import AnnotationResult, annotate_video
from neurospatial.annotation.io import (
    regions_from_cvat,
    regions_from_labelme,
)

__all__ = [
    "AnnotationResult",
    "annotate_video",
    "regions_from_cvat",
    "regions_from_labelme",
]
