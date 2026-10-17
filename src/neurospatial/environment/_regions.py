"""Region membership tables for Environment.

Which bins a region covers is decided by
:func:`neurospatial.geometry._region_bins.region_bin_mask`; this module only
arranges that rule's answers for the region methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial.geometry._binning import Binning
from neurospatial.geometry._region_bins import region_bin_mask
from neurospatial.regions import Regions

if TYPE_CHECKING:
    from neurospatial.environment._cache import AnalysisCache
    from neurospatial.geometry._arena import Arena


def _region_membership(
    regions: object,
    *,
    binning: Binning,
    arena: Arena,
    cache: AnalysisCache,
    include_boundary: bool = True,
) -> NDArray[np.bool_]:
    """Membership of every bin in every region, ``(n_bins, n_regions)``."""
    if not isinstance(regions, Regions):
        names = list(regions) if isinstance(regions, (list, tuple)) else ["goal"]
        raise TypeError(
            _format_error(
                f"region_membership() takes a Regions collection or None, got "
                f"{type(regions).__name__}.",
                why="Why: it returns one column per region in the collection; "
                "a list of names says which regions but not their shapes.",
                fix="select the regions from env.regions: from neurospatial "
                "import Regions; env.region_membership(Regions(env.regions[name] "
                f"for name in {names!r})), or env.region_mask({names!r}) for one "
                "mask over all of them",
            )
        )
    if not isinstance(include_boundary, bool):
        raise TypeError(
            f"include_boundary must be a bool, got {type(include_boundary).__name__}"
        )
    membership = np.zeros((binning.bins.n_bins, len(regions)), dtype=np.bool_)
    for column, region in enumerate(regions.values()):
        membership[:, column] = region_bin_mask(
            region,
            binning=binning,
            arena=arena,
            include_boundary=include_boundary,
            indexes=cache,
        )
    return membership
