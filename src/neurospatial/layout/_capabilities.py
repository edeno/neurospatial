"""Compatibility aliases for owned geometry; implementations live in geometry.

Kept for existing imports and trusted builder-free pickle records.
"""

from neurospatial.geometry._capabilities import (
    RectilinearGrid as RectilinearGrid,
)
from neurospatial.geometry._capabilities import (
    RectilinearGridProvider as RectilinearGridProvider,
)
from neurospatial.geometry._capabilities import (
    TrackLinearizationProvider as TrackLinearizationProvider,
)
from neurospatial.geometry._capabilities import (
    require_rectilinear as require_rectilinear,
)
