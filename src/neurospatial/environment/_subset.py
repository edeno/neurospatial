"""Select owned bins while preserving their existing navigation geometry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from neurospatial.regions import Region

if TYPE_CHECKING:
    from neurospatial.environment._cache import AnalysisCache
    from neurospatial.geometry import Arena


def _subset_mask(
    binning: Binning,
    arena: Arena,
    regions: Mapping[str, Region],
    cache: AnalysisCache,
    *,
    mask: NDArray[np.bool_] | None = None,
    region_names: Sequence[str] | None = None,
    polygon: shapely.Polygon | None = None,
    invert: bool = False,
) -> NDArray[np.bool_]:
    """Validate a spatial selection and build its active-bin mask."""
    n_bins = binning.bins.n_bins
    # --- Input Validation ---
    # Exactly one selection parameter must be provided
    n_params = sum([mask is not None, region_names is not None, polygon is not None])
    if n_params == 0:
        msg = "Exactly one of {mask, region_names, polygon} must be provided."
        raise ValueError(msg)
    if n_params > 1:
        msg = (
            "Exactly one of {mask, region_names, polygon} must be provided. "
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

    elif region_names is not None:
        # Validate region_names parameter
        if not isinstance(region_names, (list, tuple)):
            msg = f"region_names must be a list or tuple, got {type(region_names)}"
            raise ValueError(msg)

        if len(region_names) == 0:
            msg = "region_names cannot be empty"
            raise ValueError(msg)

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
            indexes=cache,
        )
        mask = np.asarray(covered.any(axis=1), dtype=np.bool_)

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
    regions: Mapping[str, Region],
    cache: AnalysisCache,
    *,
    mask: NDArray[np.bool_] | None = None,
    region_names: Sequence[str] | None = None,
    polygon: shapely.Polygon | None = None,
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
        invert=invert,
    )
    return _Derivation(select_bins(with_cached_faces(cache, binning), mask), "subset")
