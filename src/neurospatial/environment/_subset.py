"""Select owned bins while preserving their existing navigation geometry."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import shapely
import shapely.geometry.base
from numpy.typing import NDArray

from neurospatial._exceptions import RegionNotFoundError, _format_error
from neurospatial.environment._fields import with_cached_faces
from neurospatial.environment._publication import _Derivation
from neurospatial.geometry import (
    Binning,
    polygon_bin_mask,
    region_bin_masks,
    select_bins,
)
from neurospatial.regions import Regions

if TYPE_CHECKING:
    from neurospatial.environment._cache import AnalysisCache
    from neurospatial.geometry import Arena


def _subset_mask(
    binning: Binning,
    arena: Arena,
    regions: Regions,
    cache: AnalysisCache,
    *,
    mask: NDArray[np.bool_] | None = None,
    region_names: Sequence[str] | None = None,
    polygon: shapely.Polygon | None = None,
    include_boundary: bool = True,
    invert: bool = False,
) -> NDArray[np.bool_]:
    """Validate a spatial selection and build its active-bin mask."""
    n_bins = binning.bins.n_bins
    # --- Input Validation ---
    if not isinstance(include_boundary, bool):
        raise TypeError(
            f"include_boundary must be a bool, got {type(include_boundary).__name__}."
        )

    # Exactly one selection parameter must be provided
    n_params = sum([mask is not None, region_names is not None, polygon is not None])
    if n_params == 0:
        raise ValueError(
            "Exactly one of {mask, region_names, polygon} must be provided."
        )
    if n_params > 1:
        raise ValueError(
            "Exactly one of {mask, region_names, polygon} must be provided. "
            f"Got {n_params} parameters."
        )

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
            raise ValueError(
                f"mask must have shape (n_bins,) = ({n_bins},), got shape {mask.shape}"
            )
        if not include_boundary:
            raise ValueError(
                _format_error(
                    "subset() got include_boundary= with mask=.",
                    why="Why: include_boundary decides polygon edges for "
                    "region_names= and polygon=; a mask already lists the bins.",
                    fix="drop include_boundary=, or build the mask with "
                    "env.region_mask(..., include_boundary=False)",
                )
            )

    elif region_names is not None:
        # Validate region_names parameter
        if not isinstance(region_names, (list, tuple)):
            raise ValueError(
                f"region_names must be a list or tuple, got {type(region_names)}"
            )

        if len(region_names) == 0:
            raise ValueError("region_names cannot be empty")

        # Check all regions exist
        for name in region_names:
            if name not in regions:
                raise RegionNotFoundError(
                    name, available=list(regions), argument="region_names", in_list=True
                )

        covered = region_bin_masks(
            [regions[name] for name in region_names],
            binning=binning,
            arena=arena,
            include_boundary=include_boundary,
            indexes=cache,
        )
        mask = np.asarray(covered.any(axis=1), dtype=np.bool_)

    elif polygon is not None:
        # Validate polygon parameter
        try:
            # Type check
            if not isinstance(polygon, shapely.geometry.base.BaseGeometry):
                raise TypeError(
                    f"polygon must be a Shapely geometry object, got {type(polygon)}"
                )

            mask = polygon_bin_mask(
                polygon, binning.bins.centers, include_boundary=include_boundary
            )

        except (AttributeError, TypeError, ValueError) as e:
            raise ValueError(f"Invalid polygon: {e}") from e

    else:
        # Should never reach here due to earlier validation
        raise RuntimeError("No selection method specified (should be unreachable)")

    # Apply invert if requested
    if invert:
        mask = ~mask

    # Check that selection is not empty
    if not np.any(mask):
        raise ValueError(
            f"No bins selected. Selection resulted in empty mask. (invert={invert})"
        )

    return mask


def _subset(
    binning: Binning,
    arena: Arena,
    regions: Regions,
    cache: AnalysisCache,
    *,
    mask: NDArray[np.bool_] | None = None,
    region_names: Sequence[str] | None = None,
    polygon: shapely.Polygon | None = None,
    include_boundary: bool = True,
    invert: bool = False,
) -> _Derivation:
    """Derive the selected bins, with the bounds of what is kept."""
    mask = _subset_mask(
        binning,
        arena,
        regions,
        cache,
        mask=mask,
        region_names=region_names,
        polygon=polygon,
        include_boundary=include_boundary,
        invert=invert,
    )
    return _Derivation(select_bins(with_cached_faces(cache, binning), mask), "subset")
