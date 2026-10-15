"""Compatibility aliases for owned geometry; implementations live in geometry.

Kept for existing imports and trusted builder-free pickle records.
"""

from neurospatial.geometry._face_transform import (
    ORTHOGONALITY_TOLERANCE as ORTHOGONALITY_TOLERANCE,
)
from neurospatial.geometry._face_transform import (
    _center_vectors as _center_vectors,
)
from neurospatial.geometry._face_transform import (
    _combined_vectors as _combined_vectors,
)
from neurospatial.geometry._face_transform import (
    certify_transformed_faces as certify_transformed_faces,
)
from neurospatial.geometry._face_transform import (
    transform_normals as transform_normals,
)
