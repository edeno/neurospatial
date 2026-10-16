"""Coarsening retains continuous domains and representable physical walls."""

import copy
import re
import warnings

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon, box

from neurospatial import Environment
from neurospatial.arena import MaskArena, OpenArena, UnknownArena
from neurospatial.geometry._arena import encode_arena
from neurospatial.io import from_dict, to_dict
from neurospatial.layout import create_layout


def _arena_record(arena):
    """Compare domain values independently of detached public export identity."""
    metadata, arrays = encode_arena(arena)
    return metadata, {name: array.tolist() for name, array in arrays.items()}


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
    return Environment.from_layout(create_layout("RegularGrid", **params), arena=arena)


def _walled():
    return _regular(
        arena=OpenArena(box(0, 0, 6, 6), walls=(LineString([(2, 0), (2, 6)]),))
    )


@pytest.mark.parametrize("factor", [1, 2])
def test_rebin_preserves_walls_in_the_layout_facade(factor):
    parent = _walled()
    parent = parent.with_metadata(units="cm", frame="camera", name="arena")
    child = parent.rebin(factor)
    assert _arena_record(child.arena) == _arena_record(parent.arena)
    assert child.units == "cm" and child.frame == "camera"
    assert child.name == "arena"
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
def test_rebin_selects_closed_domain_centers_and_preserves_holes(physical):
    domain = Polygon(
        box(0, 0, 6, 6).exterior.coords, holes=[box(1, 1, 3, 3).exterior.coords]
    )
    arena = OpenArena(domain) if physical else None
    parent = _regular(arena=arena, holes=True)
    assert parent.layout_type == "RegularGrid"
    child = parent.rebin(2)
    assert _arena_record(child.arena) == _arena_record(parent.arena)
    assert child.arena.physical_footprint_known is physical
    # Coarse centers on the hole's corners still hold three active fine bins.
    centers = np.arange(1.0, 6.0, 2.0)
    np.testing.assert_array_equal(
        child.bin_centers,
        np.stack(np.meshgrid(centers, centers, indexing="ij"), -1).reshape(-1, 2),
    )
    assert np.all(child.bin_at(parent.bin_centers) >= 0)
    hole = np.array([[2.0, 2.0], [1.5, 1.5], [2.5, 2.5], [1.5, 2.5]])
    assert not child.contains(hole).any()
    np.testing.assert_array_equal(child.bin_at(hole), -1)
    np.testing.assert_allclose(child.bin_sizes, 4.0)


def _corner_missing_environment(kind):
    """Fine unit grids over a 4-wide square (cube) without its first corner cell."""
    if kind == "polygon":
        footprint = box(0, 0, 4, 4).difference(box(0, 0, 1, 1))
        return Environment.from_polygon(
            footprint, bin_size=1.0, connect_diagonal_neighbors=False
        )
    n_dims = 3 if kind == "mask-3d" else 2
    mask = np.ones((4,) * n_dims, dtype=bool)
    mask[(0,) * n_dims] = False
    return Environment.from_grid_mask(
        mask, (np.arange(5.0),) * n_dims, connect_diagonal_neighbors=False
    )


@pytest.mark.parametrize("kind", ["mask-2d", "mask-3d", "polygon"])
def test_factor_two_rebin_merges_every_fine_bin_beside_a_domain_corner(kind):
    # The coarse cell around the missing corner is centered on the arena
    # boundary. It still holds active fine bins, so it must be retained.
    fine = _corner_missing_environment(kind)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing is lost, so nothing to report
        coarse = fine.rebin(2)
    assert coarse.n_bins == 2**fine.n_dims
    assert np.all(coarse.bin_at(fine.bin_centers) >= 0)

    # Occupancy time is conserved: each sample's coarse bin is the union of
    # its fine bin, so no recorded time disappears from coarse totals.
    rng = np.random.default_rng(0)
    positions = rng.uniform(-0.5, 4.5, size=(20_000, fine.n_dims))
    fine_bins = fine.bin_at(positions)
    positions = positions[fine_bins >= 0]
    coarse_bins = coarse.bin_at(positions)
    assert np.all(coarse_bins >= 0)
    np.testing.assert_array_equal(
        coarse_bins, coarse.bin_at(fine.bin_centers)[fine_bins[fine_bins >= 0]]
    )
    dt = 0.02
    fine_time = np.bincount(fine_bins[fine_bins >= 0], minlength=fine.n_bins) * dt
    coarse_time = np.bincount(coarse_bins, minlength=coarse.n_bins) * dt
    assert coarse_time.sum() == pytest.approx(fine_time.sum())


def _reported_loss(record):
    """Return the bin count and percentage quoted by the coverage warning."""
    (message,) = [
        str(item.message) for item in record if "without a new bin" in str(item.message)
    ]
    count, total, percent = re.search(
        r"(\d+) of (\d+) original bins \((\d+\.\d)%", message
    ).groups()
    return int(count), int(total), float(percent), message


def test_factor_rebin_reports_fine_bins_left_outside_every_coarse_cell():
    # Perimeter coarse cells of a circle have centers outside it, so some
    # recorded fine bins have no coarse cell. The warning quantifies the loss.
    circle = Point(0, 0).buffer(10.0, quad_segs=64)
    fine = Environment.from_polygon(circle, bin_size=1.0)
    with pytest.warns(UserWarning, match="without a new bin") as record:
        coarse = fine.rebin(2)
    lost = coarse.bin_at(fine.bin_centers) < 0
    assert lost.any()
    count, total, percent, message = _reported_loss(record)
    assert (count, total) == (np.count_nonzero(lost), fine.n_bins)
    assert percent == pytest.approx(
        100 * fine.bin_sizes[lost].sum() / fine.bin_sizes.sum(), abs=0.05
    )
    assert "-1" in message and "clip" in message

    # The suggested route covers the whole arena: no fine bin is lost and
    # the exact cells add up to the arena's area.
    covering = Environment.from_polygon(box(*circle.bounds), bin_size=2.0)
    exact = Environment.from_bins(covering.clip(circle).binning, arena=fine.arena)
    assert np.all(exact.bin_at(fine.bin_centers) >= 0)
    assert exact.bin_sizes.sum() == pytest.approx(circle.area)


def test_factor_rebin_reports_truncation_and_coverage_separately():
    # Truncation drops the last row and column with its own warning; the
    # coverage warning counts only bins inside the truncated coarse extent.
    mask = np.ones((5, 5), dtype=bool)
    mask[0, 0] = mask[0, 1] = mask[1, 0] = False
    fine = Environment.from_grid_mask(
        mask, (np.arange(6.0),) * 2, connect_diagonal_neighbors=False
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        coarse = fine.rebin(3)
    assert any("truncated" in str(item.message) for item in record)
    # Coarse cell [0, 3]^2 has center (1.5, 1.5) in the mask; nothing inside the
    # 3-by-3 extent is lost, so no coverage warning is issued.
    assert coarse.n_bins == 1
    assert not any("without a new bin" in str(item.message) for item in record)


def test_rebin_preserves_unknown_arena_without_inventing_a_domain():
    arena = UnknownArena(2)
    child = _regular(arena=arena).rebin(2)
    assert _arena_record(child.arena) == _arena_record(arena)
    assert not child.arena.domain_known
    assert child.n_bins == 9


def test_rebin_nonpolygon_arena_uses_its_containment_contract():
    parent = _regular(holes=True, n_dims=1)
    assert isinstance(parent.arena, MaskArena)
    assert parent.arena.polygonal_domain() is None
    child = parent.rebin(2)
    assert _arena_record(child.arena) == _arena_record(parent.arena)
    # The nonpolygon closed-cell union includes the retained boundary at1.
    np.testing.assert_array_equal(child.bin_centers, [[1.0], [3.0], [5.0]])
    assert not child.contains(np.array([[2.0]]))[0]


def test_rebin_refuses_when_no_coarse_center_is_in_the_arena():
    parent = Environment.from_layout(
        create_layout(
            "RegularGrid",
            bin_size=1.0,
            dimension_ranges=((0.0, 6.0),) * 2,
            positions=np.array([[0.5, 0.5]]),
            dilate=False,
            close_gaps=False,
            fill_holes=False,
        )
    )
    # The only coarse center near the observed cell is (1.5, 1.5), outside it.
    with pytest.raises(ValueError, match=r"coarse.*center.*arena"):
        parent.rebin(3)


def test_sparse_rebin_result_remains_in_the_regular_grid_family():
    parent = _regular(holes=True)
    first = parent.rebin(2)
    assert first.layout_type == "RegularGrid"
    assert first._binning.kind == "RegularGrid"
    with pytest.warns(UserWarning, match="truncated"):
        direct = parent.rebin((2, 4))
    for variant in (first, copy.deepcopy(first), from_dict(to_dict(first))):
        repeated = variant.rebin(1)
        np.testing.assert_array_equal(repeated.bin_centers, first.bin_centers)
        with pytest.warns(UserWarning, match="truncated"):
            sequential = variant.rebin((1, 2))
        np.testing.assert_array_equal(
            sequential.bin_centers, [[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]]
        )
        np.testing.assert_array_equal(sequential.bin_centers, direct.bin_centers)
        np.testing.assert_array_equal(sequential.bin_sizes, direct.bin_sizes)
        assert _arena_record(sequential.arena) == _arena_record(variant.arena)


@pytest.mark.parametrize("holes", [False, True])
def test_rebin_publishes_only_the_final_environment(monkeypatch, holes):
    parent = _regular(holes=holes)
    from environment_helpers import record_publications

    published = record_publications(monkeypatch)
    child = parent.rebin(2)
    assert published == [child]
    assert child._geometry is not parent._geometry
    assert len(child._cache) == 0


def test_rebin_arena_survives_subset_copy_and_dict_roundtrip():
    child = _walled().rebin(2)
    child = child.subset(mask=child.bin_centers[:, 1] < 4)
    restored = from_dict(to_dict(child))
    for variant in (child, copy.deepcopy(child), restored):
        assert variant.arena.domain_known and variant.arena.physical_footprint_known
        assert variant.arena.polygonal_domain().equals(box(0, 0, 6, 6))
        assert variant.arena.walls[0].equals(LineString([(2, 0), (2, 6)]))
        np.testing.assert_array_equal(
            variant._binning.bins.pairs, child._binning.bins.pairs
        )
        assert nx.number_connected_components(variant.connectivity) == 2
