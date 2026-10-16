"""Compatibility aliases for owned geometry; implementations live in geometry.

Kept for existing imports and trusted builder-free pickle records.
"""

from neurospatial.geometry._face_normals import (
    FaceNormals as FaceNormals,
)
from neurospatial.geometry._face_normals import (
    _unit_rows as _unit_rows,
)
from neurospatial.geometry._face_normals import (
    geometric_normals as geometric_normals,
)
from neurospatial.geometry._face_normals import (
    validate_normals as validate_normals,
)
