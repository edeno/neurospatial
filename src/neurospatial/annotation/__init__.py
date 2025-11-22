"""Video annotation tools for defining environments and regions."""

from neurospatial.annotation.converters import (
    env_from_boundary_region,
    shapes_to_regions,
)
from neurospatial.annotation.core import AnnotationResult, annotate_video
from neurospatial.annotation.io import (
    regions_from_cvat,
    regions_from_labelme,
)

__all__ = [
    "AnnotationResult",
    "annotate_video",
    "env_from_boundary_region",
    "regions_from_cvat",
    "regions_from_labelme",
    "shapes_to_regions",
]
