"""Region membership tables for Environment.

Which bins a region covers is decided by
:func:`neurospatial.geometry._region_bins.region_bin_mask`; this module only
arranges that rule's answers for the region methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurospatial.geometry._binning import Binning
from neurospatial.geometry._region_bins import region_bin_mask
from neurospatial.regions import Regions

if TYPE_CHECKING:
    from neurospatial.geometry._arena import Arena


def _region_membership(
    regions: Regions,
    *,
    binning: Binning,
    arena: Arena,
    include_boundary: bool = True,
) -> NDArray[np.bool_]:
    """Membership of every bin in every region, ``(n_bins, n_regions)``."""
    if not isinstance(regions, Regions):
        raise TypeError(
            f"regions must be a Regions instance or None, got {type(regions).__name__}"
        )
    if not isinstance(include_boundary, bool):
        raise TypeError(
            f"include_boundary must be a bool, got {type(include_boundary).__name__}"
        )
    membership = np.zeros((binning.bins.n_bins, len(regions)), dtype=np.bool_)
    for column, region in enumerate(regions.values()):
        membership[:, column] = region_bin_mask(
            region, binning=binning, arena=arena, include_boundary=include_boundary
        )
    return membership
