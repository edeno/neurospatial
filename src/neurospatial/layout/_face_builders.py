"""Compatibility aliases for owned geometry; implementations live in geometry.

Kept for existing imports and trusted builder-free pickle records.
"""

from neurospatial.geometry._face_builders import (
    _MESH_SKEW_ANGLE_DEG as _MESH_SKEW_ANGLE_DEG,
)
from neurospatial.geometry._face_builders import (
    _MESH_SKEW_FRACTION as _MESH_SKEW_FRACTION,
)
from neurospatial.geometry._face_builders import (
    grid_geometry as grid_geometry,
)
from neurospatial.geometry._face_builders import (
    hex_geometry as hex_geometry,
)
from neurospatial.geometry._face_builders import (
    mesh_geometry as mesh_geometry,
)
from neurospatial.geometry._face_builders import (
    polar_geometry as polar_geometry,
)
from neurospatial.geometry._face_builders import (
    track_geometry as track_geometry,
)
from neurospatial.geometry._face_builders import (
    track_junctions as track_junctions,
)
