"""regions
=======

Immutable *continuous* ROI objects plus optional helpers.

This sub-package offers:

* :class:`Region`   - an immutable ROI (point or polygon), compared by value
  and unhashable; key collections by its name
* :class:`Regions`  - a small `MutableMapping[str, Region]` builder
* Lightweight import / export helpers
* A single convenience plot function (Matplotlib imported lazily)

"""

from __future__ import annotations

# --- core value objects ---------------------------------
from .core import Region, Regions
from .core import _freeze_metadata as _freeze_metadata
from .core import _snapshot_regions as _snapshot_regions
from .core import _thaw_metadata as _thaw_metadata

# --- thin adapters --------------------------------------
from .io import (
    load_cvat_xml,
    load_labelme_json,
    mask_to_region,
    regions_from_json,
    regions_to_json,
)
from .ops import points_in_any_region, regions_containing_points
from .plot import plot_regions

# The ``_`` names imported here are internal helpers shared with other
# neurospatial packages; they are not public API.
__all__ = [
    "Region",
    "Regions",
    "load_cvat_xml",
    "load_labelme_json",
    "mask_to_region",
    "plot_regions",
    "points_in_any_region",
    "regions_containing_points",
    "regions_from_json",
    "regions_to_json",
]
