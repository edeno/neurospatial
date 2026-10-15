"""Coarsening retains continuous domains and representable physical walls."""

import copy

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from neurospatial import Environment
from neurospatial.arena import MaskArena, OpenArena, UnknownArena
from neurospatial.io import from_dict, to_dict


def _regular(*, arena=None, holes=False, n_dims=2):
    params = {
        "bin_size": 1.0,
        "dimension_ranges": ((0.0, 6.0),) * n_dims,
        "infer_active_bins": holes,
        "dilate": False,
        "fill_holes": False,
        "close_gaps": False,
        "connect_diagonal_neighbors": False,
    }
    if holes:
        centers = np.stack(
            np.meshgrid(*(np.arange(6.0) + 0.5,) * n_dims, indexing="ij"), axis=-1
        ).reshape(-1, n_dims)
        keep = ~np.all((centers > 1) & (centers < 3), axis=1)
        params["positions"] = centers[keep]
    return Environment.from_layout("RegularGrid", params, arena=arena)


def _walled():
    return _regular(
        arena=OpenArena(box(0, 0, 6, 6), walls=(LineString([(2, 0), (2, 6)]),))
    )


@pytest.mark.parametrize("factor", [1, 2])
def test_rebin_preserves_walls_in_the_layout_facade(factor):
    parent = _walled()
    parent.units, parent.frame, parent.name = "cm", "camera", "arena"
    child = parent.rebin(factor)
    assert child.arena is parent.arena
    assert child.units == "cm" and child.frame == "camera"
    assert child.name == "arena_rebinned"
    assert nx.number_connected_components(child.connectivity) == 2
    sides = child.bin_centers[:, 0] < 2
    assert all(sides[u] == sides[v] for u, v in child.connectivity.edges)
    np.testing.assert_allclose(child.bin_sizes, factor**2)
    if factor == 1:
        np.testing.assert_array_equal(child.bin_centers, parent.bin_centers)
        np.testing.assert_array_equal(
            child._binning.bins.pairs, parent._binning.bins.pairs
        )


def test_rebin_refuses_a_physical_wall_inside_a_coarse_cell():
    with pytest.raises(ValueError, match="wall cuts through bin"):
        _walled().rebin(3)


@pytest.mark.parametrize("physical", [False, True])
def test_rebin_selects_strict_polygon_centers_and_preserves_holes(physical):
    domain = Polygon(
        box(0, 0, 6, 6).exterior.coords, holes=[box(1, 1, 3, 3).exterior.coords]
    )
    arena = OpenArena(domain) if physical else None
    parent = _regular(arena=arena, holes=True)
    assert parent.layout._layout_type_tag == "RegularGrid"
    child = parent.rebin(2)
    assert child.arena is parent.arena
    assert child.arena.physical_footprint_known is physical
    np.testing.assert_array_equal(
        child.bin_centers, [[1, 5], [3, 5], [5, 1], [5, 3], [5, 5]]
    )
    assert not child.contains(np.array([[2.0, 2.0], [1.0, 1.0]])).any()
    np.testing.assert_allclose(child.bin_sizes, 4.0)


def test_rebin_preserves_unknown_arena_without_inventing_a_domain():
    arena = UnknownArena(2)
    child = _regular(arena=arena).rebin(2)
    assert child.arena is arena
    assert not child.arena.domain_known
    assert child.n_bins == 9


def test_rebin_nonpolygon_arena_uses_its_containment_contract():
    parent = _regular(holes=True, n_dims=1)
    assert isinstance(parent.arena, MaskArena)
    assert parent.arena.polygonal_domain() is None
    child = parent.rebin(2)
    assert child.arena is parent.arena
    # The nonpolygon closed-cell union includes the retained boundary at1.
    np.testing.assert_array_equal(child.bin_centers, [[1.0], [3.0], [5.0]])
    assert not child.contains(np.array([[2.0]]))[0]


def test_rebin_refuses_when_no_coarse_center_is_in_the_arena():
    parent = Environment.from_layout(
        "RegularGrid",
        {
            "bin_size": 1.0,
            "dimension_ranges": ((0.0, 6.0),) * 2,
            "positions": np.array([[0.5, 0.5]]),
            "dilate": False,
            "close_gaps": False,
            "fill_holes": False,
        },
    )
    with pytest.raises(ValueError, match=r"coarse.*center.*arena"):
        parent.rebin(2)


def test_sparse_rebin_result_remains_in_the_regular_grid_family():
    parent = _regular(holes=True)
    first = parent.rebin(2)
    assert first.layout._layout_type_tag == "RegularGrid"
    assert first._binning.kind == "RegularGrid"
    with pytest.warns(UserWarning, match="truncated"):
        direct = parent.rebin((2, 4))
    for variant in (first, copy.deepcopy(first), from_dict(to_dict(first))):
        repeated = variant.rebin(1)
        np.testing.assert_array_equal(repeated.bin_centers, first.bin_centers)
        with pytest.warns(UserWarning, match="truncated"):
            sequential = variant.rebin((1, 2))
        np.testing.assert_array_equal(sequential.bin_centers, [[5.0, 2.0]])
        np.testing.assert_array_equal(sequential.bin_centers, direct.bin_centers)
        np.testing.assert_array_equal(sequential.bin_sizes, direct.bin_sizes)
        assert sequential.arena is variant.arena


def test_rebin_arena_survives_subset_copy_and_dict_roundtrip():
    child = _walled().rebin(2)
    child = child.subset(bins=child.bin_centers[:, 1] < 4)
    restored = from_dict(to_dict(child))
    for variant in (child, copy.deepcopy(child), restored):
        assert variant.arena.domain_known and variant.arena.physical_footprint_known
        assert variant.arena.polygonal_domain().equals(box(0, 0, 6, 6))
        assert variant.arena.walls[0].equals(LineString([(2, 0), (2, 6)]))
        np.testing.assert_array_equal(
            variant._binning.bins.pairs, child._binning.bins.pairs
        )
        assert nx.number_connected_components(variant.connectivity) == 2
