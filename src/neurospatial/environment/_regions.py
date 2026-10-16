"""Region-related operations for Environment.

This module provides methods for querying bins and masks within named regions.
Regions can be points or polygons (2D only) defined in the Environment's regions
container.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely import contains, covers
from shapely import points as shapely_points

from neurospatial.environment._cartesian import require_cartesian
from neurospatial.environment._lookup import locate
from neurospatial.geometry._binning import Binning
from neurospatial.geometry._coordinates import CartesianSystem, EgocentricPolarSystem
from neurospatial.ops.binning import _polygon_regions_mask
from neurospatial.regions import Region, Regions

if TYPE_CHECKING:
    from neurospatial.geometry._arena import Arena


def _bins_in_region(
    region: Region,
    *,
    binning: Binning,
    arena: Arena,
    coordinates: CartesianSystem | EgocentricPolarSystem,
) -> NDArray[np.intp]:
    """Active bins that contain a point region or lie inside a polygon region."""
    n_dims = binning.bins.n_dims
    if region.kind == "point":
        point_nd = np.asarray(region.data).reshape(1, -1)
        if point_nd.shape[1] != n_dims:
            raise ValueError(
                f"Region point dimension {point_nd.shape[1]} "
                f"does not match environment dimension {n_dims}.",
            )
        require_cartesian(coordinates, "bins_in_region() for a point region")
        bin_idx = locate(binning, arena, point_nd)
        return np.asarray(bin_idx[bin_idx != -1], dtype=np.intp)

    if region.kind == "polygon":
        if n_dims != 2:  # pragma: no cover
            raise ValueError(
                "Polygon regions are only supported for 2D environments.",
            )

        centers = binning.bins.centers
        contained_mask = shapely.contains_xy(region.data, centers[:, 0], centers[:, 1])

        return np.flatnonzero(contained_mask)

    # pragma: no cover
    raise ValueError(f"Unsupported region kind: {region.kind}")


def _region_membership(
    bin_centers: NDArray[np.float64],
    regions: Regions,
    *,
    include_boundary: bool = True,
) -> NDArray[np.bool_]:
    """Boolean membership of every bin center in every region, ``(n_bins, n_regions)``."""
    # Input validation
    if not isinstance(regions, Regions):
        raise TypeError(
            f"regions must be a Regions instance or None, got {type(regions).__name__}"
        )

    if not isinstance(include_boundary, bool):
        raise TypeError(
            f"include_boundary must be a bool, got {type(include_boundary).__name__}"
        )

    n_bins = len(bin_centers)

    # Handle empty regions case
    if len(regions) == 0:
        return np.zeros((n_bins, 0), dtype=bool)

    # Initialize membership array
    n_regions = len(regions)
    membership = np.zeros((n_bins, n_regions), dtype=bool)

    # Check if any polygon regions exist and validate dimensionality
    has_polygon_regions = any(r.kind == "polygon" for r in regions.values())
    if has_polygon_regions:
        # Only supports 2D for now
        if bin_centers.shape[1] != 2:
            raise NotImplementedError(
                f"region_membership currently only supports 2D environments "
                f"for polygon regions. Environment has {bin_centers.shape[1]} dimensions."
            )
        # Create shapely Points array ONCE before loop (major optimization)
        points = shapely_points(bin_centers[:, 0], bin_centers[:, 1])

    # Iterate over regions and check containment
    for region_idx, (_region_name, region) in enumerate(regions.items()):
        # Handle point regions - points have no area, so no bins can be inside
        if region.kind == "point":
            # Leave column as all False (no bin can be "inside" a point)
            continue

        # Handle polygon regions
        if region.kind == "polygon":
            # Use pre-created points array (created once before loop)
            # Vectorized containment check
            if include_boundary:
                # covers: True if point is inside or on boundary
                mask = covers(region.data, points)
            else:
                # contains: True only if strictly inside
                mask = contains(region.data, points)

            membership[:, region_idx] = mask

    return membership


def _region_mask(
    regions: str | list[str] | Region | Regions,
    *,
    named_regions: Regions,
    binning: Binning,
    arena: Arena,
    coordinates: CartesianSystem | EgocentricPolarSystem,
    include_boundary: bool = True,
) -> NDArray[np.bool_]:
    """Union mask of the bins selected by regions or names in ``named_regions``."""
    # Resolve `regions` into the list of Region objects we'll union over.
    # Strings (single or list) refer to entries in the environment's regions.
    resolved: list[Region]
    if isinstance(regions, str):
        resolved = [named_regions[regions]]
    elif isinstance(regions, list):
        resolved = [named_regions[r] if isinstance(r, str) else r for r in regions]
    elif isinstance(regions, Region):
        resolved = [regions]
    elif isinstance(regions, Regions):
        resolved = list(regions.values())
    else:
        raise TypeError(
            "regions must be str, list[str], Region, or Regions; "
            f"got {type(regions).__name__}."
        )

    centers = binning.bins.centers
    mask = np.zeros(centers.shape[0], dtype=bool)

    # Point regions: use bin lookup (the polygon containment predicate
    # would return empty because points have no area).
    polygon_regions = Regions()
    for region in resolved:
        if region.kind == "point":
            require_cartesian(coordinates, "region_mask() for a point region")
            bin_idx = locate(binning, arena, np.asarray(region.data).reshape(1, -1))
            if bin_idx[0] != -1:
                mask[bin_idx[0]] = True
        else:
            polygon_regions[region.name] = region

    # Polygon (and other area-bearing) regions: delegate to the
    # canonical containment predicate.
    if len(polygon_regions) > 0:
        mask |= _polygon_regions_mask(
            centers, polygon_regions, include_boundary=include_boundary
        )

    return mask
