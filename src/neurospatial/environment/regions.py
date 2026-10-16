"""Region-related operations for Environment.

This module provides methods for querying bins and masks within named regions.
Regions can be points or polygons (2D only) defined in the Environment's regions
container.

"""

from __future__ import annotations

import numpy as np
import shapely
from numpy.typing import NDArray

from neurospatial.environment._protocols import RegionConsumer
from neurospatial.regions import Region, Regions


def _bins_in_region(env: RegionConsumer, region_name: str) -> NDArray[np.intp]:
    """Implement Environment.bins_in_region using explicit regions inputs."""
    region = env.regions[region_name]

    if region.kind == "point":
        point_nd = np.asarray(region.data).reshape(1, -1)
        if point_nd.shape[1] != env.n_dims:
            raise ValueError(
                f"Region point dimension {point_nd.shape[1]} "
                f"does not match environment dimension {env.n_dims}.",
            )
        bin_idx = env._require_cartesian("bins_in_region() for a point region").bin_at(
            point_nd
        )
        return np.asarray(bin_idx[bin_idx != -1], dtype=np.intp)

    if region.kind == "polygon":
        if env.n_dims != 2:  # pragma: no cover
            raise ValueError(
                "Polygon regions are only supported for 2D environments.",
            )

        polygon = region.data
        contained_mask = shapely.contains_xy(
            polygon,
            env.bin_centers[:, 0],
            env.bin_centers[:, 1],
        )

        return np.flatnonzero(contained_mask)

    # pragma: no cover
    raise ValueError(f"Unsupported region kind: {region.kind}")


def _region_membership(
    env: RegionConsumer,
    regions: Regions | None = None,
    *,
    include_boundary: bool = True,
) -> NDArray[np.bool_]:
    """Implement Environment.region_membership using explicit regions inputs."""
    from neurospatial.regions import Regions

    # Input validation
    if regions is None:
        regions = env.regions
    elif not isinstance(regions, Regions):
        raise TypeError(
            f"regions must be a Regions instance or None, got {type(regions).__name__}"
        )

    if not isinstance(include_boundary, bool):
        raise TypeError(
            f"include_boundary must be a bool, got {type(include_boundary).__name__}"
        )

    # Handle empty regions case
    if len(regions) == 0:
        return np.zeros((env.n_bins, 0), dtype=bool)

    # Get bin centers as points
    bin_centers = env.bin_centers  # shape (n_bins, n_dims)

    # Initialize membership array
    n_regions = len(regions)
    membership = np.zeros((env.n_bins, n_regions), dtype=bool)

    # Import shapely functions for vectorized operations
    from shapely import contains, covers
    from shapely import points as shapely_points

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
    env: RegionConsumer,
    regions: str | list[str] | Region | Regions,
    *,
    include_boundary: bool = True,
) -> NDArray[np.bool_]:
    """Implement Environment.region_mask using explicit regions inputs."""
    from typing import cast

    from neurospatial.environment.core import Environment
    from neurospatial.ops.binning import regions_to_mask
    from neurospatial.regions import Region as _Region
    from neurospatial.regions import Regions as _Regions

    # Resolve `regions` into the list of Region objects we'll union over.
    # Strings (single or list) refer to entries in self.regions.
    resolved: list[_Region]
    if isinstance(regions, str):
        resolved = [env.regions[regions]]
    elif isinstance(regions, list):
        resolved = [env.regions[r] if isinstance(r, str) else r for r in regions]
    elif isinstance(regions, _Region):
        resolved = [regions]
    elif isinstance(regions, _Regions):
        resolved = list(regions.values())
    else:
        raise TypeError(
            "regions must be str, list[str], Region, or Regions; "
            f"got {type(regions).__name__}."
        )

    mask = np.zeros(env.bin_centers.shape[0], dtype=bool)

    # Point regions: use bin_at (the free-function regions_to_mask
    # would return empty because points have no area).
    polygon_regions = _Regions()
    for region in resolved:
        if region.kind == "point":
            bin_idx = env._require_cartesian("region_mask() for a point region").bin_at(
                np.asarray(region.data).reshape(1, -1)
            )
            if bin_idx[0] != -1:
                mask[bin_idx[0]] = True
        else:
            polygon_regions[region.name] = region

    # Polygon (and other area-bearing) regions: delegate to the
    # canonical containment predicate.
    if len(polygon_regions) > 0:
        poly_mask = regions_to_mask(
            cast("Environment", env),
            polygon_regions,
            include_boundary=include_boundary,
        )
        mask |= poly_mask

    return mask
