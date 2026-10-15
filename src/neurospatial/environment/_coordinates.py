"""Compatibility aliases for owned geometry; implementations live in geometry.

Kept for existing imports and trusted builder-free pickle records.
"""

from neurospatial.geometry._coordinates import (
    CartesianSystem as CartesianSystem,
)
from neurospatial.geometry._coordinates import (
    CoordinateSystem as CoordinateSystem,
)
from neurospatial.geometry._coordinates import (
    EgocentricPolarSystem as EgocentricPolarSystem,
)
from neurospatial.geometry._coordinates import (
    polar_circularity as polar_circularity,
)
