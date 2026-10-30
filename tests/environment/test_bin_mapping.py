"""Exact bin correspondence between an environment and the one it came from.

``bin_mapping_from`` pairs bins only where geometry proves the pair: the same
cell (subset, an uncut clip cell), or a whole fine cell inside one coarse cell
(rebin on the same lattice). Transfers keep missing data missing, pool rates
from counts and occupancy, and agree with estimating on the derived
environment from the same trajectory.
"""

from __future__ import annotations

import copy
import pickle
import warnings

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box

from neurospatial import Environment, IncompatibleEnvironmentError
from neurospatial.encoding import compare_spatial_rates, compute_spatial_rates
from neurospatial.environment import BinMapping
from neurospatial.io import from_file, to_file
from neurospatial.ops.transforms import AffineND

_RNG = np.random.default_rng(7)
# 20 minutes at 30 Hz over a 40 x 40 cm box: every 4 cm bin is visited.
_TIMES = np.arange(36_000) / 30.0
_POSITIONS = _RNG.uniform(0.0, 40.0, size=(len(_TIMES), 2))
_SPIKE_RNG = np.random.default_rng(11)
_SPIKE_TIMES = [
    np.sort(_SPIKE_RNG.choice(_TIMES, size=n, replace=False)) for n in (500, 2000)
]


def _grid(shape: tuple[int, int] = (10, 10), width: float = 4.0) -> Environment:
    """A grid of ``width`` cm cells from the origin, every cell active."""
    return Environment.from_grid_mask(
        np.ones(shape, dtype=bool),
        grid_edges=tuple(np.arange(n + 1) * width for n in shape),
        units="cm",
    )


def _quiet(call):
    """Run a rebin without its truncation and lost-bin warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return call()


def _binned(env: Environment, **options):
    return compute_spatial_rates(
        env,
        _SPIKE_TIMES,
        _TIMES,
        _POSITIONS,
        method="binned",
        bandwidth=0.0,
        unit_ids=[3, 8],
        **options,
    )


# ---------------------------------------------------------------------------
# Identical cells
# ---------------------------------------------------------------------------


def test_subset_pairs_are_the_kept_bins_in_order():
    env = _grid()
    mask = env.bin_centers[:, 0] + env.bin_centers[:, 1] < 30
    part = env.subset(mask=mask)

    mapping = part.bin_mapping_from(env)

    expected = np.column_stack([np.flatnonzero(mask), np.arange(part.n_bins)])
    np.testing.assert_array_equal(mapping.pairs, expected)
    assert mapping.pairs.dtype == np.intp
    assert mapping.kind == "one_to_one"
    np.testing.assert_array_equal(mapping.unmapped_parent_bins, np.flatnonzero(~mask))
    assert mapping.split_parent_bins.size == 0
    assert mapping.uncovered_child_bins.size == 0
    np.testing.assert_array_equal(mapping.child_coverage, 1.0)


def test_subset_pairs_feed_compare_spatial_rates_like_a_manual_mapping():
    env = _grid()
    mask = env.bin_centers[:, 0] < 24
    part = env.subset(mask=mask)
    full_rates, part_rates = _binned(env), _binned(part)
    manual = np.column_stack([np.flatnonzero(mask), np.arange(part.n_bins)])

    by_mapping = compare_spatial_rates(
        full_rates,
        part_rates,
        bin_mapping=part.bin_mapping_from(env).pairs,
        topology="may_differ",
    )
    by_hand = compare_spatial_rates(
        full_rates, part_rates, bin_mapping=manual, topology="may_differ"
    )

    np.testing.assert_array_equal(by_mapping.correlation, by_hand.correlation)
    assert by_mapping.status.tolist() == ["scored", "scored"]
    np.testing.assert_array_equal(by_mapping.bins_a, np.flatnonzero(mask))
    np.testing.assert_array_equal(by_mapping.bins_b, np.arange(part.n_bins))


def test_subset_of_hexagons_and_identical_rebuild_pair_one_to_one():
    hexagons = Environment.from_samples(_POSITIONS, bin_size=5.0, layout="Hexagonal")
    keep = np.arange(hexagons.n_bins) % 3 != 0
    mapping = hexagons.subset(mask=keep).bin_mapping_from(hexagons)
    np.testing.assert_array_equal(mapping.pairs[:, 0], np.flatnonzero(keep))

    rebuilt = _grid()
    identity = rebuilt.bin_mapping_from(_grid())
    np.testing.assert_array_equal(identity.pairs[:, 0], identity.pairs[:, 1])
    assert identity.kind == "one_to_one"
    assert len(identity.pairs) == rebuilt.n_bins


def test_subset_of_a_track_pairs_one_to_one():
    track = Environment.linear_track(
        node_positions=[(0.0, 0.0), (100.0, 0.0)], bin_size=2.0, track_width=8.0
    )
    keep = np.arange(track.n_bins) >= 10
    mapping = track.subset(mask=keep).bin_mapping_from(track)
    np.testing.assert_array_equal(mapping.pairs[:, 0], np.flatnonzero(keep))


@pytest.mark.parametrize("n_dims", [1, 3])
def test_rebin_factor_nests_in_one_and_three_dimensions(n_dims):
    env = Environment.from_grid_mask(
        np.ones((4,) * n_dims, dtype=bool), grid_edges=(np.arange(5.0),) * n_dims
    )
    mapping = env.rebin(2).bin_mapping_from(env)
    np.testing.assert_array_equal(np.bincount(mapping.pairs[:, 1]), 2**n_dims)
    np.testing.assert_allclose(mapping.sum(env.bin_sizes), env.rebin(2).bin_sizes)


def test_clip_pairs_uncut_cells_and_reports_cut_ones_as_split():
    env = _grid()
    clip = Polygon([(0, 0), (22, 0), (30, 40), (0, 40)])
    clipped = env.clip(clip)
    cells = shapely.box(
        env.bin_centers[:, 0] - 2,
        env.bin_centers[:, 1] - 2,
        env.bin_centers[:, 0] + 2,
        env.bin_centers[:, 1] + 2,
    )
    whole = shapely.covers(clip, cells)
    cut = shapely.intersects(clip, cells) & ~whole
    cut &= shapely.area(shapely.intersection(clip, cells)) > 0

    mapping = clipped.bin_mapping_from(env)

    assert mapping.kind == "one_to_one"
    np.testing.assert_array_equal(mapping.pairs[:, 0], np.flatnonzero(whole))
    np.testing.assert_array_equal(mapping.split_parent_bins, np.flatnonzero(cut))
    assert not np.isin(mapping.split_parent_bins, mapping.pairs[:, 0]).any()
    np.testing.assert_array_equal(
        mapping.unmapped_parent_bins, np.flatnonzero(~whole & ~cut)
    )
    # The pieces of cut cells contain no whole parent cell.
    pieces = np.setdiff1d(np.arange(clipped.n_bins), mapping.pairs[:, 1])
    np.testing.assert_array_equal(mapping.uncovered_child_bins, pieces)
    np.testing.assert_array_equal(
        clipped.clip_mapping.parent_bin_indices[mapping.pairs[:, 1]],
        mapping.pairs[:, 0],
    )
    np.testing.assert_array_equal(mapping.child_coverage[pieces], 0.0)
    occupancy = env.occupancy(_TIMES, _POSITIONS)
    np.testing.assert_array_equal(
        np.isnan(mapping.sum(occupancy)), mapping.child_coverage == 0
    )


def test_split_pieces_of_a_cell_pair_with_themselves_after_a_subset():
    env = _grid((2, 2))
    clipped = env.clip(box(0, 0, 6, 8))  # cuts the right column in half
    part = clipped.subset(mask=np.arange(clipped.n_bins) != 0)

    mapping = part.bin_mapping_from(clipped)

    np.testing.assert_array_equal(
        mapping.pairs, np.column_stack([np.arange(1, clipped.n_bins), np.arange(3)])
    )
    assert mapping.split_parent_bins.size == 0


# ---------------------------------------------------------------------------
# Nested grids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(10, 10), (9, 7)], ids=["even", "truncated"])
def test_rebin_factor_maps_each_block_of_fine_bins_to_its_coarse_bin(shape):
    env = _grid(shape)
    coarse = _quiet(lambda: env.rebin(2))

    mapping = coarse.bin_mapping_from(env)

    assert mapping.kind == "many_to_one"
    rows, columns = np.unravel_index(np.arange(env.n_bins), shape)
    kept = (rows < shape[0] // 2 * 2) & (columns < shape[1] // 2 * 2)
    block = (rows // 2) * (shape[1] // 2) + columns // 2
    order = np.argsort(mapping.pairs[:, 0])
    np.testing.assert_array_equal(mapping.pairs[order, 0], np.flatnonzero(kept))
    np.testing.assert_array_equal(mapping.pairs[order, 1], block[kept])
    np.testing.assert_array_equal(np.bincount(mapping.pairs[:, 1]), 4)
    np.testing.assert_array_equal(mapping.unmapped_parent_bins, np.flatnonzero(~kept))
    np.testing.assert_array_equal(mapping.child_coverage, 1.0)
    # Sorted by child bin, then parent bin.
    np.testing.assert_array_equal(
        np.lexsort((mapping.pairs[:, 0], mapping.pairs[:, 1])),
        np.arange(len(mapping.pairs)),
    )


@pytest.mark.parametrize("shape", [(10, 10), (9, 7)], ids=["even", "truncated"])
def test_summed_fine_occupancy_is_the_coarse_occupancy(shape):
    """Totals agree on retained support.

    They agree when positions in fine bins the truncation dropped are excluded
    on both sides. Occupancy counts an interval only when both of its samples
    are in a bin, so an interval ending in a dropped fine bin counts on the
    fine grid (in the bin it starts in) but not on the coarse grid.
    """
    env = _grid(shape)
    coarse = _quiet(lambda: env.rebin(2))
    mapping = coarse.bin_mapping_from(env)
    positions = _POSITIONS.copy()
    positions[np.isin(env.bin_at(positions), mapping.unmapped_parent_bins)] = np.nan

    summed = mapping.sum(env.occupancy(_TIMES, positions))

    np.testing.assert_allclose(
        summed, coarse.occupancy(_TIMES, positions), rtol=1e-12, atol=0
    )
    if mapping.unmapped_parent_bins.size:
        # Without excluding them, intervals into dropped bins count only on
        # the fine grid.
        kept = mapping.sum(env.occupancy(_TIMES, _POSITIONS))
        assert np.all(kept > coarse.occupancy(_TIMES, _POSITIONS))


def test_pooled_rates_are_the_coarse_binned_rates():
    env = _grid()
    coarse = env.rebin(2)
    fine = _binned(env)
    assert np.all(fine.occupancy > 0)
    counts = fine.firing_rates * fine.occupancy

    pooled = coarse.bin_mapping_from(env).rates(counts, fine.occupancy)

    np.testing.assert_allclose(pooled, _binned(coarse).firing_rates, rtol=1e-12)
    # Averaging rates instead would be wrong.
    averaged = coarse.bin_mapping_from(env).sum(fine.firing_rates) / 4
    assert not np.allclose(averaged, pooled, rtol=1e-6)


def test_pooled_rates_with_unvisited_fine_bins_use_zero_counts_there():
    """Unvisited bins have NaN rates but no spikes, so their count is 0."""
    env = _grid((12, 12))  # 48 cm: the 40 cm trajectory leaves bins unvisited
    coarse = env.rebin(4)
    fine = _binned(env)
    assert np.any(fine.occupancy == 0) and np.any(np.isnan(fine.firing_rates))
    counts = np.where(fine.occupancy > 0, fine.firing_rates * fine.occupancy, 0.0)

    pooled = coarse.bin_mapping_from(env).rates(counts, fine.occupancy)

    direct = _binned(coarse).firing_rates
    np.testing.assert_array_equal(np.isnan(pooled), np.isnan(direct))
    np.testing.assert_allclose(pooled, direct, rtol=1e-12)


def test_nested_coarsening_composes():
    env = _grid((8, 8))
    middle = env.rebin(2)
    coarse = middle.rebin(2)
    occupancy = env.occupancy(_TIMES, _POSITIONS)

    in_steps = coarse.bin_mapping_from(middle).sum(
        middle.bin_mapping_from(env).sum(occupancy)
    )
    at_once = coarse.bin_mapping_from(env).sum(occupancy)

    np.testing.assert_allclose(in_steps, at_once, rtol=1e-12)
    np.testing.assert_allclose(at_once.sum(), occupancy.sum(), rtol=1e-12)


def test_rebin_bin_size_on_the_same_lattice_nests():
    env = _grid((6, 6), width=1.0)
    coarse = env.rebin(bin_size=2.0)

    mapping = coarse.bin_mapping_from(env)

    assert mapping.kind == "many_to_one"
    np.testing.assert_array_equal(np.bincount(mapping.pairs[:, 1]), 4)
    assert mapping.unmapped_parent_bins.size == 0
    fine_centers = env.bin_centers[mapping.pairs[:, 0]]
    coarse_centers = coarse.bin_centers[mapping.pairs[:, 1]]
    assert np.all(np.abs(fine_centers - coarse_centers) < 1.0)


def test_coarse_cell_beyond_the_parent_has_partial_coverage():
    """A 4 cm lattice over a 6 cm grid: the outer coarse cells are half empty."""
    env = _grid((3, 3), width=2.0)
    env = env.subset(mask=np.ones(env.n_bins, dtype=bool))
    coarse = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool),
        grid_edges=(np.array([0.0, 4.0, 8.0]),) * 2,
        units="cm",
    )

    mapping = coarse.bin_mapping_from(env)

    # 4 fine cells fill the first coarse cell; 2, 2 and 1 cover the others.
    np.testing.assert_allclose(mapping.child_coverage, [1.0, 0.5, 0.5, 0.25])
    np.testing.assert_array_equal(np.bincount(mapping.pairs[:, 1]), [4, 2, 2, 1])
    occupancy = np.ones(env.n_bins)
    np.testing.assert_array_equal(mapping.sum(occupancy), [4.0, 2.0, 2.0, 1.0])


def test_offset_lattice_does_not_nest():
    env = _grid((6, 6), width=1.0)
    offset = Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool),
        grid_edges=(np.arange(4) * 2.0 + 0.5,) * 2,
        units="cm",
    )

    with pytest.raises(ValueError, match="do not fall on parent's cell edges"):
        offset.bin_mapping_from(env)


def test_rebin_by_a_whole_factor_nests_and_off_the_lattice_does_not():
    """from_samples cells are exactly 10 cm: 20 cm cells nest, 15 cm cut them."""
    env = Environment.from_samples(_POSITIONS * 2.5, bin_size=10.0, units="cm")
    coarse = _quiet(lambda: env.rebin(bin_size=20.0))
    assert coarse.bin_mapping_from(env).kind == "many_to_one"

    off_lattice = _quiet(lambda: env.rebin(bin_size=15.0))
    with pytest.raises(ValueError, match="edges cut parent cells"):
        off_lattice.bin_mapping_from(env)


def test_refinement_is_refused_and_the_reverse_direction_works():
    env = _grid((6, 6), width=2.0)
    finer = env.rebin(bin_size=1.0)

    with pytest.raises(ValueError, match="refinement") as caught:
        finer.bin_mapping_from(env)
    assert "parent.bin_mapping_from(env)" in str(caught.value)

    assert env.bin_mapping_from(finer).kind == "many_to_one"


def test_rotated_lattices_nest_in_their_shared_frame():
    rotation = np.array(
        [[np.cos(0.3), -np.sin(0.3), 1.0], [np.sin(0.3), np.cos(0.3), 2.0], [0, 0, 1]]
    )
    env = _grid((6, 6), width=1.0).apply_transform(AffineND(rotation))
    coarse = _quiet(lambda: env.rebin(bin_size=2.0))

    mapping = coarse.bin_mapping_from(env)

    assert mapping.kind == "many_to_one"
    assert np.bincount(mapping.pairs[:, 1]).max() == 4


def test_rotated_grid_against_an_axis_aligned_grid_has_no_pair():
    rotation = np.array(
        [[np.cos(0.3), -np.sin(0.3), 0.0], [np.sin(0.3), np.cos(0.3), 0.0], [0, 0, 1]]
    )
    env = _grid((6, 6), width=1.0)
    rotated = env.apply_transform(AffineND(rotation))

    with pytest.raises(ValueError, match="different lattice frames") as caught:
        rotated.bin_mapping_from(env)
    message = str(caught.value)
    assert "transforms keep bin IDs" in message
    assert "bin i of the transformed environment is bin i of the original" in message
    # What the message says is true: transforms keep bin order.
    np.testing.assert_allclose(
        rotated.bin_centers, env.bin_centers @ rotation[:2, :2].T, atol=1e-12
    )


def test_polar_grids_nest():
    fine = Environment.from_polar_egocentric(
        distance_range=(0.0, 40.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=10.0,
        angle_bin_size=np.pi / 4,
    )
    coarse = Environment.from_polar_egocentric(
        distance_range=(0.0, 40.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=20.0,
        angle_bin_size=np.pi / 2,
    )

    mapping = coarse.bin_mapping_from(fine)

    np.testing.assert_array_equal(np.bincount(mapping.pairs[:, 1]), 4)
    np.testing.assert_allclose(
        mapping.sum(fine.bin_sizes), coarse.bin_sizes, rtol=1e-12
    )


# ---------------------------------------------------------------------------
# Transfers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def strip() -> BinMapping:
    """Six 1 cm bins in a row merged in pairs, with the last pair dropped."""
    fine = Environment.from_grid_mask(
        np.ones((1, 6), dtype=bool), grid_edges=(np.arange(2.0), np.arange(7.0))
    )
    coarse = fine.subset(mask=np.arange(6) < 4).rebin((1, 2))
    return coarse.bin_mapping_from(fine)


def test_sum_propagates_nan_and_never_carries_unmapped_bins(strip):
    np.testing.assert_array_equal(strip.unmapped_parent_bins, [4, 5])
    values = np.array([1.0, np.nan, 2.0, 3.0, 100.0, 100.0])
    np.testing.assert_array_equal(strip.sum(values), [np.nan, 5.0])
    rows = strip.sum(np.array([values, np.arange(6.0)]))
    np.testing.assert_array_equal(rows, [[np.nan, 5.0], [1.0, 5.0]])
    assert strip.sum(np.arange(6)).dtype == np.float64


def test_uncovered_child_bins_are_nan():
    env = _grid((2, 2))
    clipped = env.clip(box(0, 0, 6, 8))
    mapping = clipped.bin_mapping_from(env)
    assert mapping.uncovered_child_bins.tolist() == [2, 3]

    summed = mapping.sum(np.ones(env.n_bins))
    np.testing.assert_array_equal(summed, [1.0, 1.0, np.nan, np.nan])
    rates = mapping.rates(np.ones(env.n_bins), np.ones(env.n_bins))
    np.testing.assert_array_equal(np.isnan(rates), [False, False, True, True])


def test_rates_pool_counts_and_occupancy(strip):
    counts = np.array([0.0, 100.0, 5.0, 0.0, 9.0, 9.0])
    occupancy = np.array([100.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(strip.rates(counts, occupancy), [100 / 101, np.nan])
    # NaN in any mapped input is NaN out.
    occupancy[0] = np.nan
    assert np.isnan(strip.rates(counts, occupancy)[0])
    # Per-unit counts share one occupancy row.
    per_unit = strip.rates(np.array([counts, 2 * counts]), np.ones(6))
    np.testing.assert_allclose(per_unit, [[50.0, 2.5], [100.0, 5.0]])


def test_category_keeps_only_unanimous_labels(strip):
    names = np.array(["arm", "arm", "arm", "goal", "x", "x"])
    assert strip.category(names, missing="mixed").tolist() == ["arm", "mixed"]
    numbers = strip.category(np.array([1, 1, 2, 2, 0, 0]), missing=-1)
    assert numbers.tolist() == [1, 2]
    assert numbers.dtype.kind == "i"
    flags = strip.category(
        np.array([True, True, True, False, True, True]), missing=None
    )
    assert flags.tolist() == [True, None]
    per_unit = strip.category(
        np.array([[1, 1, 2, 3, 0, 0], [4, 5, 6, 6, 0, 0]]), missing=-1
    )
    assert per_unit.tolist() == [[1, -1], [-1, 6]]


def test_category_gives_missing_to_uncovered_bins():
    env = _grid((2, 2))
    mapping = env.clip(box(0, 0, 6, 8)).bin_mapping_from(env)
    labels = mapping.category(np.array(["a", "a", "b", "b"]), missing="none")
    assert labels.tolist() == ["a", "a", "none", "none"]


def test_category_refuses_a_missing_value_that_is_a_label(strip):
    with pytest.raises(ValueError, match="missing=0, which is one of the labels"):
        strip.category(np.array([0, 0, 1, 1, 2, 2]), missing=0)
    with pytest.raises(ValueError, match="one of the labels"):
        strip.category(np.array(["a"] * 6), missing="a")


@pytest.mark.parametrize(
    ("values", "fix"),
    [
        (np.ones(2), "values on the child's 2 bins"),
        (np.ones((6, 3)), "pass values.T"),
        (np.ones(5), "one value per parent bin, shape (6,)"),
        (np.ones((2, 2, 6)), "one value per parent bin"),
    ],
)
def test_wrong_shapes_are_refused_with_the_fix(strip, values, fix):
    with pytest.raises(ValueError, match="takes values on the parent") as caught:
        strip.sum(values)
    assert fix in str(caught.value)


def test_rates_refuse_occupancy_of_another_shape(strip):
    with pytest.raises(ValueError, match="occupancy with shape"):
        strip.rates(np.ones((2, 6)), np.ones((3, 6)))


def test_sum_refuses_labels(strip):
    with pytest.raises(ValueError, match=r"BinMapping\.category"):
        strip.sum(np.array(["a"] * 6))


# ---------------------------------------------------------------------------
# Saved environments, refusals, ownership
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("derive", ["subset", "clip", "rebin"])
def test_mapping_is_the_same_after_saving_both_environments(tmp_path, derive):
    env = _grid((6, 6))
    child = {
        "subset": lambda: env.subset(mask=env.bin_centers[:, 1] < 12),
        "clip": lambda: env.clip(box(0, 0, 13, 24)),
        "rebin": lambda: env.rebin(2),
    }[derive]()
    to_file(env, tmp_path / "parent")
    to_file(child, tmp_path / "child")

    loaded = from_file(tmp_path / "child").bin_mapping_from(
        from_file(tmp_path / "parent")
    )
    fresh = child.bin_mapping_from(env)

    np.testing.assert_array_equal(loaded.pairs, fresh.pairs)
    np.testing.assert_array_equal(loaded.split_parent_bins, fresh.split_parent_bins)
    np.testing.assert_array_equal(loaded.child_coverage, fresh.child_coverage)


def test_different_units_are_refused():
    env = _grid()
    in_meters = Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool),
        grid_edges=(np.arange(11) * 4.0,) * 2,
        units="m",
    )
    with pytest.raises(IncompatibleEnvironmentError, match=r"units 'cm'.*units 'm'"):
        in_meters.bin_mapping_from(env)


def test_different_frame_after_a_transform_is_refused_with_the_bin_id_rule():
    env = _grid()
    moved = env.apply_transform(AffineND(np.eye(3)), frame="session2")
    with pytest.raises(IncompatibleEnvironmentError) as caught:
        moved.bin_mapping_from(env)
    message = str(caught.value)
    assert "frame 'session2'" in message
    assert "transforms keep bin IDs" in message


def test_independent_environments_have_no_exact_pair():
    env = _grid()
    hexagons = Environment.from_samples(
        _POSITIONS, bin_size=4.0, layout="Hexagonal", units="cm"
    )
    with pytest.raises(
        ValueError, match="no bin of this environment is the same cell"
    ) as caught:
        hexagons.bin_mapping_from(env)
    message = str(caught.value)
    assert "nearest center" in message
    assert "transforms keep bin IDs" not in message


def test_parent_must_be_an_environment():
    with pytest.raises(TypeError, match="parent of type ndarray"):
        _grid().bin_mapping_from(np.arange(3))


def test_arrays_are_read_only_and_owned(strip):
    names = (
        "pairs",
        "unmapped_parent_bins",
        "split_parent_bins",
        "uncovered_child_bins",
        "child_coverage",
    )
    for restored in (strip, pickle.loads(pickle.dumps(strip)), copy.deepcopy(strip)):
        for name in names:
            array = getattr(restored, name)
            assert not array.flags.writeable, name
            with pytest.raises(ValueError, match="read-only"):
                array[...] = 0
    with pytest.raises(AttributeError):
        strip.kind = "one_to_one"


def test_summary_and_repr_count_every_bin(strip):
    summary = strip.summary()
    assert summary == {
        "kind": "many_to_one",
        "n_parent_bins": 6,
        "n_child_bins": 2,
        "n_mapped_parent_bins": 4,
        "n_unmapped_parent_bins": 2,
        "n_split_parent_bins": 0,
        "n_fully_covered_child_bins": 2,
        "n_partly_covered_child_bins": 0,
        "n_uncovered_child_bins": 0,
    }
    assert all(np.isscalar(value) for value in summary.values())
    assert repr(strip) == (
        "BinMapping(kind='many_to_one', 6 parent bins -> 2 child bins; parent "
        "bins: 4 mapped, 2 unmapped, 0 split; child bins: 2 fully covered, 0 "
        "partly covered, 0 uncovered)"
    )


def _explicit_points(centers) -> Environment:
    """Three point bins in a line, with unit measures and no cells."""
    from neurospatial.binning import (
        BinConnections,
        Binning,
        MissingFaces,
        NearestBinLocator,
        SpatialBins,
    )

    centers = np.asarray(centers, dtype=float)
    return Environment.from_bins(
        Binning(
            "explicit",
            SpatialBins.from_arrays(
                centers, np.ones(len(centers)), np.ones(len(centers), dtype=np.intp)
            ),
            BinConnections.from_arrays(np.array([[0, 1], [1, 2]]), np.ones(2)),
            NearestBinLocator(centers, max_distance=0.5),
            MissingFaces("explicit points"),
        ),
        units="cm",
    )


def test_a_boundary_point_given_to_another_coarse_cell_splits_its_fine_cell():
    # Fine bin 1 owns x = 2 because its right neighbor is empty; the coarse
    # grid gives x = 2 to coarse bin 1. Carrying fine bin 1 into coarse bin 0
    # would put a sample at x = 2 in the wrong coarse bin.
    fine = Environment.from_grid_mask(
        np.array([True, True, False, True, True, True]), (np.arange(7.0),)
    )
    coarse = fine.rebin(2)
    assert fine.bin_at(np.array([[2.0]])) == [1]
    assert coarse.bin_at(np.array([[2.0]])) == [1]

    mapping = coarse.bin_mapping_from(fine)

    np.testing.assert_array_equal(mapping.split_parent_bins, [1])
    assert 1 not in mapping.pairs[:, 0]
    np.testing.assert_array_equal(mapping.uncovered_child_bins, [0, 1])
    times, positions = np.array([0.0, 0.1, 0.2]), np.full((3, 1), 2.0)
    carried = mapping.sum(fine.occupancy(times, positions))
    direct = coarse.occupancy(times, positions)
    assert np.isnan(carried[:2]).all()  # unknown, never the wrong bin
    np.testing.assert_array_equal(carried[2:], direct[2:])
    assert mapping.summary()["n_uncovered_child_bins"] == 2
    assert mapping.summary()["n_partly_covered_child_bins"] == 0


def test_carried_occupancy_matches_the_coarse_grid_for_samples_on_cell_edges():
    # Worst case for ownership: samples on every corner and edge midpoint of
    # every fine cell of an irregular arena, where empty neighbors give
    # boundary points to the cells beside them.
    rng = np.random.default_rng(0)
    trajectory = (
        50
        + 40
        * np.column_stack(
            [np.sin(np.linspace(0, 40, 6000)), np.cos(np.linspace(0, 26, 6000))]
        )
        + rng.normal(0, 2, (6000, 2))
    )
    env = Environment.from_samples(trajectory, bin_size=5.0)
    edges = env.binning.rectilinear.edges
    ix, iy = np.nonzero(env.binning.rectilinear.active_mask)
    xs = np.stack(
        [edges[0][ix], (edges[0][ix] + edges[0][ix + 1]) / 2, edges[0][ix + 1]], 1
    )
    ys = np.stack(
        [edges[1][iy], (edges[1][iy] + edges[1][iy + 1]) / 2, edges[1][iy + 1]], 1
    )
    positions = np.column_stack(
        [np.repeat(xs, 3, axis=1).ravel(), np.tile(ys, (1, 3)).ravel()]
    )
    times = np.arange(len(positions)) * 0.01
    coarse = env.rebin(2)
    mapping = coarse.bin_mapping_from(env)
    assert len(mapping.split_parent_bins) > 0  # the arena has such boundaries
    # Positions in fine bins the coarse grid dropped are left out of both.
    positions[np.isin(env.bin_at(positions), mapping.unmapped_parent_bins)] = np.nan

    carried = mapping.sum(_quiet(lambda: env.occupancy(times, positions)))
    direct = _quiet(lambda: coarse.occupancy(times, positions))

    known = np.isfinite(carried)
    assert known.sum() == coarse.n_bins - len(mapping.uncovered_child_bins)
    np.testing.assert_allclose(carried[known], direct[known])


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ([[0], [1], [2]], [[0, 0], [1, 1], [2, 2]]),
        ([[0, 0], [1, 1], [2, 2]], [[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
    ],
    ids=["1d-vs-2d", "2d-vs-3d"],
)
def test_coordinates_of_different_widths_are_never_the_same_cells(first, second):
    from neurospatial.encoding import SpatialRatesResult

    env_a, env_b = _explicit_points(first), _explicit_points(second)

    def rates(env):
        return SpatialRatesResult(
            firing_rates=np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]]),
            occupancy=np.ones(3),
            env=env,
            method="binned",
            bandwidth=0.0,
            unit_ids=["a", "b"],
        )

    with pytest.raises(IncompatibleEnvironmentError, match="-D coordinates"):
        env_b.bin_mapping_from(env_a)
    for mapping in (None, np.column_stack([np.arange(3), np.arange(3)])):
        with pytest.raises(IncompatibleEnvironmentError, match="bin coordinates"):
            compare_spatial_rates(
                rates(env_a),
                rates(env_b),
                bin_mapping=mapping,
                topology="may_differ",
                min_common_bins=3,
            )


def test_a_grid_whose_every_bin_is_split_is_refused_naming_the_splits():
    # Edges 1e-11 apart nest within the edge tolerance (a relative 1e-9),
    # but lie farther apart than the rounding position lookup allows on a
    # boundary, so each environment's lookup gives the shared boundaries to
    # different cells and no pair is exact.
    rng = np.random.default_rng(0)
    samples = rng.uniform(0, 20, (500, 2))
    parent = Environment.from_samples(samples, bin_size=2.0)
    child = Environment.from_samples(samples + 1e-11, bin_size=2.0)
    with pytest.raises(ValueError, match="own boundary points"):
        child.bin_mapping_from(parent)
