"""Compatibility aliases for owned geometry; implementations live in geometry.

Kept for existing imports and trusted builder-free pickle records.
"""

from neurospatial.geometry._grid_frame import (
    FramedGrid as FramedGrid,
)
from neurospatial.geometry._grid_frame import (
    frame_from_locator as frame_from_locator,
)
from neurospatial.geometry._grid_frame import (
    require_grid_context as require_grid_context,
)
