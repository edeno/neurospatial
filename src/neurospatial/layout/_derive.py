"""Compatibility aliases for owned geometry; implementations live in geometry.

Kept for existing imports and trusted builder-free pickle records.
"""

from neurospatial.geometry._derive import (
    _classification as _classification,
)
from neurospatial.geometry._derive import (
    _mapped as _mapped,
)
from neurospatial.geometry._derive import (
    _matrix as _matrix,
)
from neurospatial.geometry._derive import (
    _refusal as _refusal,
)
from neurospatial.geometry._derive import (
    _restricted_mask as _restricted_mask,
)
from neurospatial.geometry._derive import (
    _track_transform as _track_transform,
)
from neurospatial.geometry._derive import (
    _transform_binning as _transform_binning,
)
from neurospatial.geometry._derive import (
    subset_binning as subset_binning,
)
from neurospatial.geometry._derive import (
    transform_binning as transform_binning,
)
