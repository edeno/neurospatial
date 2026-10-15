"""Compatibility aliases for owned geometry; implementations live in geometry.

Kept for existing imports and trusted builder-free pickle records.
"""

from neurospatial.geometry._diffusion import (
    DiffusionGeometry as DiffusionGeometry,
)
from neurospatial.geometry._diffusion import (
    FaceGeometry as FaceGeometry,
)
from neurospatial.geometry._diffusion import (
    FiniteVolumeGeometryProvider as FiniteVolumeGeometryProvider,
)
from neurospatial.geometry._diffusion import (
    _validate_volumes as _validate_volumes,
)
