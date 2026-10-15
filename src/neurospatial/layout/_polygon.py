"""Compatibility aliases for owned geometry; implementations live in geometry.

Kept for existing imports and trusted builder-free pickle records.
"""

from neurospatial.geometry._polygon import (
    ClipMapping as ClipMapping,
)
from neurospatial.geometry._polygon import (
    ClippedLocator as ClippedLocator,
)
from neurospatial.geometry._polygon import (
    PolygonCells as PolygonCells,
)
from neurospatial.geometry._polygon import (
    _boundary_cell_indices as _boundary_cell_indices,
)
from neurospatial.geometry._polygon import (
    _check_disjoint as _check_disjoint,
)
from neurospatial.geometry._polygon import (
    _error as _error,
)
from neurospatial.geometry._polygon import (
    _key as _key,
)
from neurospatial.geometry._polygon import (
    _opening as _opening,
)
from neurospatial.geometry._polygon import (
    _overlay as _overlay,
)
from neurospatial.geometry._polygon import (
    _parts as _parts,
)
from neurospatial.geometry._polygon import (
    _representative as _representative,
)
from neurospatial.geometry._polygon import (
    _validate_polygon as _validate_polygon,
)
from neurospatial.geometry._polygon import (
    cells_from_locator as cells_from_locator,
)
from neurospatial.geometry._polygon import (
    certify_faces as certify_faces,
)
from neurospatial.geometry._polygon import (
    clip_binning as clip_binning,
)
from neurospatial.geometry._polygon import (
    materialize_cells as materialize_cells,
)
from neurospatial.geometry._polygon import (
    subset_polygon_cells as subset_polygon_cells,
)
from neurospatial.geometry._polygon import (
    transform_polygon_cells as transform_polygon_cells,
)
