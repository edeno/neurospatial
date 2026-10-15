"""Compatibility aliases for owned geometry; implementations live in geometry.

Kept for existing imports and trusted builder-free pickle records.
"""

from neurospatial.geometry._validation import (
    _active_indices as _active_indices,
)
from neurospatial.geometry._validation import (
    _computed_equal as _computed_equal,
)
from neurospatial.geometry._validation import (
    _current_clip_mapping as _current_clip_mapping,
)
from neurospatial.geometry._validation import (
    _equal as _equal,
)
from neurospatial.geometry._validation import (
    _gap as _gap,
)
from neurospatial.geometry._validation import (
    _grid_width_errors as _grid_width_errors,
)
from neurospatial.geometry._validation import (
    _locator_geometry as _locator_geometry,
)
from neurospatial.geometry._validation import (
    _LocatorGeometry as _LocatorGeometry,
)
from neurospatial.geometry._validation import (
    _mapped_geometry as _mapped_geometry,
)
from neurospatial.geometry._validation import (
    _polygon_area_allowance as _polygon_area_allowance,
)
from neurospatial.geometry._validation import (
    _polygon_coordinate_errors as _polygon_coordinate_errors,
)
from neurospatial.geometry._validation import (
    _primitive_geometry as _primitive_geometry,
)
from neurospatial.geometry._validation import (
    _rectilinear_geometry as _rectilinear_geometry,
)
from neurospatial.geometry._validation import (
    _same_ring as _same_ring,
)
from neurospatial.geometry._validation import (
    _stored_cells as _stored_cells,
)
from neurospatial.geometry._validation import (
    _uniform_grid_widths as _uniform_grid_widths,
)
from neurospatial.geometry._validation import (
    _validate_clip_mapping as _validate_clip_mapping,
)
from neurospatial.geometry._validation import (
    _validate_components as _validate_components,
)
from neurospatial.geometry._validation import (
    _validate_graph as _validate_graph,
)
from neurospatial.geometry._validation import (
    _validate_rectilinear_faces as _validate_rectilinear_faces,
)
