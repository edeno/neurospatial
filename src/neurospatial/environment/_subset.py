"""Select owned bins while preserving their existing navigation geometry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import shapely
import shapely.geometry.base
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial.environment._fields import with_cached_faces
from neurospatial.environment._publication import _Derivation
from neurospatial.geometry import (
    Binning,
    polygon_bin_mask,
    select_bins,
)
from neurospatial.ops import _regions_mask
from neurospatial.regions import Region

if TYPE_CHECKING:
    from neurospatial.environment._cache import AnalysisCache
    from neurospatial.geometry import Arena


def _subset_mask(
    binning: Binning,
    arena: Arena,
    named_regions: Mapping[str, Region],
    *,
    mask: NDArray[np.bool_] | None = None,
    regions: Any = None,
    polygon: shapely.Polygon | None = None,
    invert: bool = False,
) -> NDArray[np.bool_]:
    """Validate a spatial selection and build its active-bin mask.

    ``named_regions`` are the environment's regions, which names in
    ``regions`` refer to.
    """
    n_bins = binning.bins.n_bins
    # --- Input Validation ---
    # Exactly one selection parameter must be provided
    n_params = sum([mask is not None, regions is not None, polygon is not None])
    if n_params == 0:
        msg = "Exactly one of {mask, regions, polygon} must be provided."
        raise ValueError(msg)
    if n_params > 1:
        msg = (
            "Exactly one of {mask, regions, polygon} must be provided. "
            f"Got {n_params} parameters."
        )
        raise ValueError(msg)

    # --- Build Selection Mask ---

    if mask is not None:
        mask = np.asarray(mask)
        if mask.dtype != bool:
            raise ValueError(
                _format_error(
                    f"subset() got mask= with dtype {mask.dtype}; mask= is a "
                    "boolean array with one entry per bin.",
                    why="Why: integers would be read as bin indices, not as keep flags.",
                    fix="for bin indices, build a mask: keep = np.zeros(env.n_bins, "
                    "dtype=bool); keep[indices] = True; env.subset(mask=keep)",
                )
            )
        if mask.shape != (n_bins,):
            msg = (
                f"mask must have shape (n_bins,) = ({n_bins},), got shape {mask.shape}"
            )
            raise ValueError(msg)

    elif regions is not None:
        # The same selection as env.region_mask(regions): a tuple or an
        # array of names is the list it takes.
        if isinstance(regions, (tuple, np.ndarray)):
            regions = list(regions)
        if isinstance(regions, (list, Mapping)) and len(regions) == 0:
            raise ValueError(
                _format_error(
                    "subset() got regions= that is empty.",
                    why="Why: with no region there are no bins to keep.",
                    fix="name at least one region, such as regions='goal' or "
                    "regions=['left', 'right']",
                )
            )
        mask = _regions_mask(
            regions, named_regions=named_regions, binning=binning, arena=arena
        )

    elif polygon is not None:
        # Validate polygon parameter
        try:
            # Type check
            if not isinstance(polygon, shapely.geometry.base.BaseGeometry):
                msg = f"polygon must be a Shapely geometry object, got {type(polygon)}"
                raise TypeError(msg)

            mask = polygon_bin_mask(polygon, binning.bins.centers)

        except (AttributeError, TypeError, ValueError) as e:
            msg = f"Invalid polygon: {e}"
            raise ValueError(msg) from e

    else:
        # Should never reach here due to earlier validation
        msg = "No selection method specified (should be unreachable)"
        raise RuntimeError(msg)

    # Apply invert if requested
    if invert:
        mask = ~mask

    # Check that selection is not empty
    if not np.any(mask):
        msg = f"No bins selected. Selection resulted in empty mask. (invert={invert})"
        raise ValueError(msg)

    return mask


def _subset(
    binning: Binning,
    arena: Arena,
    named_regions: Mapping[str, Region],
    cache: AnalysisCache,
    *,
    mask: NDArray[np.bool_] | None = None,
    regions: Any = None,
    polygon: shapely.Polygon | None = None,
    invert: bool = False,
) -> _Derivation:
    """Derive the selected bins, with the bounds of what is kept."""
    mask = _subset_mask(
        binning,
        arena,
        named_regions,
        mask=mask,
        regions=regions,
        polygon=polygon,
        invert=invert,
    )
    return _Derivation(select_bins(with_cached_faces(cache, binning), mask), "subset")
