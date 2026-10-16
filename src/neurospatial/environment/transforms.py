"""Compatibility aliases for private rebin and subset implementation paths.

Grid construction lives in ``_rebin`` and bin selection in ``_subset``. These
aliases retain older imports and trusted pickle references without duplicating
the implementations.
"""

from neurospatial.environment._rebin import (
    _rebin_factor as _rebin_factor,
)
from neurospatial.environment._rebin import (
    _require_native_grid_navigation as _require_native_grid_navigation,
)
from neurospatial.environment._subset import (
    _subset as _subset,
)
from neurospatial.environment._subset import (
    _subset_mask as _subset_mask,
)

_rebin = _rebin_factor
