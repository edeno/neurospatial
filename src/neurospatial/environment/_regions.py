"""Region membership tables for Environment.

Which bins a region covers is decided by
:func:`neurospatial.geometry.region_bin_mask`; this module only
arranges that rule's answers for the region methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import RegionNotFoundError, _format_error
from neurospatial.geometry import Binning, region_bin_mask, region_bin_masks
from neurospatial.regions import Regions, _snapshot_regions

if TYPE_CHECKING:
    from neurospatial.environment._cache import AnalysisCache
    from neurospatial.geometry import Arena


def _owned_regions(regions: object, n_dims: int) -> Regions:
    """Return an immutable snapshot of ``regions`` for an ``n_dims`` environment.

    Raises
    ------
    TypeError
        If ``regions`` is not a Regions collection.
    ValueError
        If a region's coordinates have a different dimension.
    """
    if not isinstance(regions, Regions):
        raise TypeError("regions must be a Regions instance.")
    if any(region.n_dims != n_dims for region in regions.values()):
        raise ValueError("Region coordinate dimensions must match the environment.")
    return _snapshot_regions(regions)


def _bins_in_region(
    region_name: str,
    regions: Regions,
    *,
    binning: Binning,
    arena: Arena,
    cache: AnalysisCache,
    include_boundary: bool,
) -> NDArray[np.intp]:
    """Indices of the bins the named region covers, in increasing order."""
    if region_name not in regions:
        raise RegionNotFoundError(
            region_name, available=list(regions), argument="region_name"
        )
    mask = region_bin_mask(
        regions[region_name],
        binning=binning,
        arena=arena,
        include_boundary=include_boundary,
        indexes=cache,
    )
    return np.flatnonzero(mask).astype(np.intp, copy=False)


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
    return region_bin_masks(
        list(regions.values()),
        binning=binning,
        arena=arena,
        include_boundary=include_boundary,
        indexes=cache,
    )
