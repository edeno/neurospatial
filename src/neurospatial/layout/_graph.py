"""Compatibility aliases for owned geometry; implementations live in geometry.

Kept for existing imports and trusted builder-free pickle records.
"""

from neurospatial.geometry._graph import (
    _graph_from_binning as _graph_from_binning,
)
from neurospatial.geometry._graph import (
    _source_grid as _source_grid,
)
