"""The bin that stands for a region is central in physical space."""

from __future__ import annotations

import numpy as np
from shapely.geometry import box

from neurospatial import Environment, Regions
from neurospatial.behavior._region_bins import region_representative_bin


def test_polar_region_representative_is_nearest_the_physical_centroid():
    """A wide wedge ahead of the animal is centered closer than its mid-distance.

    Averaging ``(distance, angle)`` rows puts a 10-40 cm, ±90° wedge's center
    at 25 cm; in the animal's plane its bins average to about 16 cm ahead.
    """
    env = Environment.from_polar_egocentric(
        distance_range=(0.0, 40.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=10.0,
        angle_bin_size=np.pi / 8,
    )
    regions = Regions()
    regions.add("ahead", polygon=box(10.0, -np.pi / 2, 40.0, np.pi / 2))
    env = env.with_regions(regions)

    bins = env.bins_in_region("ahead")
    distance, angle = env.bin_centers[bins].T
    places = np.c_[distance * np.cos(angle), distance * np.sin(angle)]
    gaps = np.linalg.norm(places - places.mean(axis=0), axis=1)

    representative = region_representative_bin(env, "ahead")

    assert representative == bins[np.argmin(gaps)]
    assert env.bin_centers[representative, 0] == 15.0
