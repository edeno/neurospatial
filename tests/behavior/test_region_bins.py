"""The bin that stands for a region is central in physical space."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import CartesianOnlyError, Environment, Regions
from neurospatial.behavior._region_bins import region_representative_bin


def test_polar_polygon_region_is_refused():
    """A polygon's vertices are (x, y); a polar environment refuses to read them."""
    env = Environment.from_polar_egocentric(
        distance_range=(0.0, 40.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=10.0,
        angle_bin_size=np.pi / 8,
    )
    regions = Regions()
    regions.add("ahead", polygon=box(10.0, -np.pi / 2, 40.0, np.pi / 2))
    env = env.with_regions(regions)

    with pytest.raises(CartesianOnlyError, match="polygon region"):
        region_representative_bin(env, "ahead")
