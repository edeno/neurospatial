"""One representative bin for a region, shared by trial and decision analyses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from neurospatial.environment import Environment


def region_representative_bin(env: Environment, region_name: str) -> int | None:
    """Return the bin that stands for a (possibly multi-bin) region.

    The representative is the region bin whose center is closest to the
    centroid of the region's bin centers, both measured where distance is
    physical: on an egocentric polar environment, in the animal's plane, so
    a region across the angular seam averages its angles correctly. It
    always lies inside the region,
    also for non-convex regions whose geometric centroid falls outside. When
    several region bins are equally close to the centroid, as in a symmetric
    box on a grid, the lowest bin index among them is chosen.

    Parameters
    ----------
    env : Environment
        Spatial environment, for bin centers and the region lookup.
    region_name : str
        Name of a region in ``env.regions``.

    Returns
    -------
    int or None
        Index of the representative bin, or None when the region covers no
        bin.
    """
    region_bins = env.bins_in_region(region_name)
    if len(region_bins) == 0:
        return None
    # (n_region_bins, n_dims), in a frame where distance is physical
    places = env.coordinate_system.cartesian_embedding(env.bin_centers[region_bins])
    centroid = places.mean(axis=0)  # (n_dims,)
    nearest = np.argmin(np.linalg.norm(places - centroid, axis=1))
    return int(region_bins[nearest])
