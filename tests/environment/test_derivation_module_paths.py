"""Derivations share one publication path and preserve annotations."""

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, box

from neurospatial import Environment, Regions
from neurospatial.arena import OpenArena


@pytest.mark.parametrize("mode", ["factor", "resolution"])
def test_rebin_modes_preserve_annotations_walls_and_fresh_caches(mode):
    regions = Regions()
    regions.add("goal", point=[5.5, 5.5], metadata={"reward": {"value": 1}})
    regions.add("zone", polygon=box(0, 0, 2, 2))
    parent = Environment.from_grid_mask(
        np.ones((6, 6), dtype=bool),
        grid_edges=(np.arange(7.0),) * 2,
        name="arena",
        units="cm",
        frame="camera",
        arena=OpenArena(box(0, 0, 6, 6), walls=(LineString([(2, 0), (2, 6)]),)),
    ).with_regions(regions)
    expected_regions = {name: region.to_dict() for name, region in regions.items()}
    parent_kernel = parent.compute_kernel(0.5)
    parent.nearest_bin(np.array([[1.0, 1.0]]))
    parent.get_divergence_operator()
    assert parent._kernel_cache and parent._kdtree_cache is not None

    child = parent.rebin(2) if mode == "factor" else parent.rebin(bin_size=2.0)

    assert child.name == "arena"
    assert (child.units, child.frame) == ("cm", "camera")
    assert child.n_bins == 9
    assert child._kernel_cache == {} and child._kdtree_cache is None
    assert nx.number_connected_components(child.connectivity) == 2
    sides = child.bin_centers[:, 0] < 2
    assert all(sides[u] == sides[v] for u, v in child.edge_pairs)
    np.testing.assert_array_equal(child.arena.contains([[1.0, 1.0], [5.0, 5.0]]), True)
    assert {
        name: region.to_dict() for name, region in child.regions.items()
    } == expected_regions
    assert child.regions is not parent.regions
    exported_point = child.regions["goal"].data
    assert exported_point is not parent.regions["goal"].data
    exported_point.shape = (1, 2)
    np.testing.assert_array_equal(child.regions["goal"].data, [5.5, 5.5])
    np.testing.assert_array_equal(parent.regions["goal"].data, [5.5, 5.5])
    with pytest.raises(TypeError):
        child.regions["goal"].metadata["reward"]["value"] = 0
    regions.update_region("goal", point=[0.5, 0.5])
    np.testing.assert_array_equal(child.regions["goal"].data, [5.5, 5.5])
    np.testing.assert_array_equal(parent.compute_kernel(0.5), parent_kernel)
