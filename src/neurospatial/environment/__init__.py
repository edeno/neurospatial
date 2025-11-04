"""Environment module for spatial discretization.

This module provides the Environment class, which discretizes continuous
N-dimensional spatial environments into bins/nodes with connectivity graphs.

The module is organized using a mixin pattern for better maintainability:

- core: Main Environment class with dataclass fields and core methods
- factories: Factory classmethods for creating environments
- queries: Spatial query methods (bin_at, contains, neighbors, etc.)
- serialization: Save/load methods (to_file, from_file, to_dict, from_dict)
- regions: Region-related operations (bins_in_region, mask_for_region)
- visualization: Plotting methods (plot, plot_1d)
- analysis: Analysis methods (boundary_bins, bin_attributes, etc.)
- decorators: Utility decorators (check_fitted)

Import Examples
---------------
Primary import path (recommended):
    >>> from neurospatial import Environment

Direct import path (also supported):
    >>> from neurospatial.environment import Environment

Import decorator:
    >>> from neurospatial.environment import check_fitted

Notes
-----
The Environment class is assembled from multiple mixin classes, but users
interact with it as a single unified class. All methods are available
directly on Environment instances.
"""

from neurospatial.environment.core import Environment
from neurospatial.environment.decorators import check_fitted
from neurospatial.environment.regions import _HAS_SHAPELY

__all__ = ["_HAS_SHAPELY", "Environment", "check_fitted"]
