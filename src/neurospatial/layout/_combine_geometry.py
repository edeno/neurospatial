"""Compatibility aliases for owned geometry; implementations live in geometry.

Kept for existing imports and trusted builder-free pickle records.
"""

from neurospatial.geometry._combine_geometry import (
    _arena_pieces as _arena_pieces,
)
from neurospatial.geometry._combine_geometry import (
    _arena_segment_interior as _arena_segment_interior,
)
from neurospatial.geometry._combine_geometry import (
    _ball_segment_interval as _ball_segment_interval,
)
from neurospatial.geometry._combine_geometry import (
    _boxes as _boxes,
)
from neurospatial.geometry._combine_geometry import (
    _boxes_overlap as _boxes_overlap,
)
from neurospatial.geometry._combine_geometry import (
    _grid_segment_intervals as _grid_segment_intervals,
)
from neurospatial.geometry._combine_geometry import (
    _nearest_partition as _nearest_partition,
)
from neurospatial.geometry._combine_geometry import (
    _nearest_segment_intervals as _nearest_segment_intervals,
)
from neurospatial.geometry._combine_geometry import (
    _overlap as _overlap,
)
from neurospatial.geometry._combine_geometry import (
    _partitioned as _partitioned,
)
from neurospatial.geometry._combine_geometry import (
    _Piece as _Piece,
)
from neurospatial.geometry._combine_geometry import (
    _pieces as _pieces,
)
from neurospatial.geometry._combine_geometry import (
    certify_components as certify_components,
)
from neurospatial.geometry._combine_geometry import (
    segment_enters_component as segment_enters_component,
)
