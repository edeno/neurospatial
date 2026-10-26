"""Comparing rate maps between conditions: unit identity, bins and support.

A comparison pairs units by label and bins by place. Unsupported bins (no
sufficient occupancy, or a NaN rate) never enter a score, and an undefined
score is NaN with a reason, never a fabricated 0 or ±1.
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest
from environment_helpers import (
    order_preserving_permutation,
    rates_result,
    renumbered,
    renumbered_cells,
    wall_split_arena,
)
from scipy import stats
from shapely.geometry import LineString, Polygon, box

from neurospatial import Environment, IncompatibleEnvironmentError
from neurospatial.arena import OpenArena
from neurospatial.binning import FaceGeometry, KnownFaces
from neurospatial.encoding import (
    RateComparisonResult,
    compare_spatial_rates,
    compute_spatial_rates,
)
from neurospatial.encoding.spatial import SpatialRatesResult
from neurospatial.environment import _environment_differences
from neurospatial.ops.transforms import AffineND

F = [1.0, 2.0, 3.0, 4.0, 5.0]
G = [5.0, 1.0, 4.0, 2.0, 3.0]
OCC = np.ones(5)


@pytest.fixture(scope="module")
def env() -> Environment:
    env = Environment.from_samples(
        np.linspace(0, 40, 41)[:, None], bin_size=10.0, units="cm"
    )
    assert env.n_bins == 5
    return env


@pytest.fixture
def make(env):
    def make(
        rates,
        occupancy=OCC,
        ids=(1,),
        settings=None,
        *,
        on=None,
        bandwidth=5.0,
        min_occupancy=0.0,
    ) -> SpatialRatesResult:
        return rates_result(
            env if on is None else on,
            rates,
            occupancy,
            unit_ids=ids,
            bandwidth=bandwidth,
            min_occupancy=min_occupancy,
            analysis_settings=settings,
        )

    return make


# ---------------------------------------------------------------------------
# Unit identity
# ---------------------------------------------------------------------------


def test_units_are_matched_by_id_not_row_position(make) -> None:
    rates_a = make([F, G], ids=[1, 2])
    rates_b = make([G, F], ids=[2, 1])

    result = compare_spatial_rates(rates_a, rates_b)

    assert isinstance(result, RateComparisonResult)
    assert result.unit_ids.tolist() == [1, 2]
    np.testing.assert_allclose(result.correlation, [1.0, 1.0], atol=1e-12)
    # Positional pairing would have compared f with g instead.
    assert stats.pearsonr(F, G).statistic < 0.5


def test_permuting_one_input_leaves_per_unit_scores_unchanged(make) -> None:
    rng = np.random.default_rng(3)
    ids = np.array([7, 3, 5])
    rates_a = rng.uniform(0, 10, (3, 5))
    rates_b = rng.uniform(0, 10, (3, 5))
    rates_b[1, 2] = np.nan
    occupancy_b = np.array([1.0, 1.0, 1.0, 0.0, 1.0])
    first = compare_spatial_rates(
        make(rates_a, ids=ids), make(rates_b, occupancy_b, ids=ids)
    )
    for order in ([2, 0, 1], [1, 2, 0], [2, 1, 0]):
        permuted = make(rates_b[order], occupancy_b, ids=ids[order])
        result = compare_spatial_rates(make(rates_a, ids=ids), permuted)
        np.testing.assert_array_equal(result.unit_ids, ids)
        np.testing.assert_array_equal(result.correlation, first.correlation)
        np.testing.assert_array_equal(result.n_common_bins, first.n_common_bins)
        np.testing.assert_array_equal(result.status, first.status)


def test_units_recorded_in_one_condition_are_reported_per_side(make) -> None:
    rates_a = make([F, G, F], ids=[1, 2, 3])
    rates_b = make([G, F, G], ids=[2, 3, 4])

    result = compare_spatial_rates(rates_a, rates_b)

    assert result.unit_ids.tolist() == [2, 3]
    assert result.unit_ids_only_a.tolist() == [1]
    assert result.unit_ids_only_b.tolist() == [4]
    with pytest.raises(ValueError, match="unit_ids"):
        compare_spatial_rates(rates_a, rates_b, unit_selection="exact")
    with pytest.raises(ValueError, match="no unit_ids in common"):
        compare_spatial_rates(make([F], ids=[1]), make([F], ids=[9]))


def test_exact_unit_selection_accepts_the_same_units_in_any_order(make) -> None:
    result = compare_spatial_rates(
        make([F, G], ids=[1, 2]), make([G, F], ids=[2, 1]), unit_selection="exact"
    )
    assert result.unit_ids.tolist() == [1, 2]
    assert result.unit_ids_only_a.size == result.unit_ids_only_b.size == 0


@pytest.mark.parametrize("side", ["a", "b"])
def test_generated_unit_ids_are_refused(make, side) -> None:
    generated, labelled = make([F], ids=None), make([F], ids=[0])
    pair = (generated, labelled) if side == "a" else (labelled, generated)
    with pytest.raises(ValueError, match="unit_ids="):
        compare_spatial_rates(*pair)


# ---------------------------------------------------------------------------
# Common observed support
# ---------------------------------------------------------------------------


def test_finite_rate_in_an_unoccupied_bin_is_not_support(make) -> None:
    rates_a = make([1, 2, 3, 4, 100], [1, 1, 1, 1, 0])
    rates_b = make([1, 2, 3, 4, -50])

    result = compare_spatial_rates(rates_a, rates_b)

    assert result.n_common_bins.tolist() == [4]
    np.testing.assert_allclose(result.correlation, [1.0], atol=1e-12)
    assert result.common_support[0, 4] is np.False_
    assert np.isnan(result.rate_difference[0, 4])
    assert result.occupied_a.tolist() == [True, True, True, True, False]


def test_min_occupancy_is_read_from_each_result(make) -> None:
    rates_a = make(F, [1, 0.2, 1, 1, 1], min_occupancy=0.5)
    with pytest.warns(UserWarning, match="min_occupancy"):
        result = compare_spatial_rates(rates_a, make(F))
    assert result.n_common_bins.tolist() == [4]
    assert not result.common_support[0, 1]


def test_one_units_missing_bin_does_not_remove_it_from_another(make) -> None:
    rates_a = make([[np.nan, 2, 3, 4, 5], F], ids=[1, 2])
    result = compare_spatial_rates(rates_a, make([F, F], ids=[1, 2]))
    assert result.n_common_bins.tolist() == [4, 5]
    assert result.status.tolist() == ["scored", "scored"]


def test_too_few_common_bins_is_nan_with_a_reason(make) -> None:
    result = compare_spatial_rates(make(F), make(G, [1, 1, 0, 0, 0]))
    assert np.isnan(result.correlation[0])
    assert result.status.tolist() == ["insufficient_common_bins"]
    assert result.n_common_bins.tolist() == [2]


def test_raising_min_common_bins_marks_more_units_insufficient(make) -> None:
    result = compare_spatial_rates(make(F), make(G), min_common_bins=6)
    assert result.status.tolist() == ["insufficient_common_bins"]
    assert np.isnan(result.correlation[0])
    assert result.analysis_settings["min_common_bins"] == 6


@pytest.mark.parametrize(
    ("rates_a", "rates_b", "status"),
    [
        (F, np.zeros(5), "constant_rates_b"),
        (np.full(5, 2.0), G, "constant_rates_a"),
        (np.full(5, 2.0), np.zeros(5), "constant_rates_both"),
    ],
)
@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_constant_map_is_nan_with_a_reason(
    make, rates_a, rates_b, status, method
) -> None:
    result = compare_spatial_rates(make(rates_a), make(rates_b), method=method)
    assert np.isnan(result.correlation[0])
    assert result.status.tolist() == [status]


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_map_flat_up_to_rounding_is_constant(make, method) -> None:
    # 1 + 1e-15 differs from 1 only by floating-point rounding (5 ulps);
    # correlating that noise with a real field would report r = 0.71.
    flat = [1.0, 1.0, 1.0, 1.0, 1.0 + 1e-15]
    assert flat[-1] != flat[0]
    result = compare_spatial_rates(make(F), make(flat), method=method)
    assert np.isnan(result.correlation[0])
    assert result.status.tolist() == ["constant_rates_b"]


@pytest.mark.parametrize("estimator", ["diffusion_kde", "gaussian_kde", "binned"])
def test_flat_map_from_an_estimator_is_constant(estimator) -> None:
    # One spike at the start of every occupancy interval fires at exactly
    # 1 / dt = 30 Hz everywhere the animal went: smoothed counts over smoothed
    # occupancy are flat analytically, and differ only by rounding.
    times = np.arange(3000) / 30.0
    positions = 50 + 40 * np.column_stack(
        [np.sin(2 * np.pi * times / 20), np.cos(2 * np.pi * times / 13)]
    )
    env = Environment.from_samples(positions, bin_size=10.0, units="cm")
    in_field = np.linalg.norm(positions - [70.0, 70.0], axis=1) < 15
    field_spikes = times[in_field][::2]

    def estimate(spike_times):
        return compute_spatial_rates(
            env, [spike_times], times, positions, method=estimator, unit_ids=[1]
        )

    flat = estimate(times[:-1])
    rates = flat.firing_rates[0][flat.spatial_support]
    np.testing.assert_allclose(rates, 30.0, rtol=1e-12)
    assert rates.max() > rates.min()  # rounding left a spread

    result = compare_spatial_rates(flat, estimate(field_spikes))

    assert result.status.tolist() == ["constant_rates_a"]
    assert np.isnan(result.correlation[0])


def test_small_real_variation_is_still_scored(make) -> None:
    # A field a millionth of its rate deep is 1e6 times the flat tolerance.
    result = compare_spatial_rates(make(F), make(1.0 + 1e-6 * np.asarray(F)))
    assert result.status.tolist() == ["scored"]
    np.testing.assert_allclose(result.correlation, [1.0], atol=1e-6)


def test_constant_check_uses_only_common_bins(make) -> None:
    # B varies only in the bin A did not visit, so over common support it is flat.
    rates_b = make([3, 3, 3, 3, 9])
    result = compare_spatial_rates(make(F, [1, 1, 1, 1, 0]), rates_b)
    assert result.status.tolist() == ["constant_rates_b"]


@pytest.mark.parametrize("value", [2, 0, 3.0, True])
def test_min_common_bins_must_be_an_integer_of_at_least_three(make, value) -> None:
    with pytest.raises((ValueError, TypeError), match="min_common_bins"):
        compare_spatial_rates(make(F), make(G), min_common_bins=value)


@pytest.mark.parametrize(
    ("argument", "value"),
    [("method", "kendall"), ("unit_selection", "union"), ("topology", "any")],
)
def test_choices_are_validated(make, argument, value) -> None:
    with pytest.raises(ValueError, match=argument):
        compare_spatial_rates(make(F), make(G), **{argument: value})


# ---------------------------------------------------------------------------
# Spatial correspondence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topology", ["same", "may_differ"])
@pytest.mark.parametrize("mismatch", ["units", "frame", "coordinates"])
def test_equal_shapes_cannot_bypass_environment_validation(
    env, make, mismatch, topology
) -> None:
    if mismatch == "coordinates":
        other = Environment.from_samples(
            np.linspace(1000, 1040, 41)[:, None], bin_size=10.0, units="cm"
        )
    else:
        other = env.with_metadata(**{mismatch: "mm" if mismatch == "units" else "x"})
    assert other.n_bins == env.n_bins
    rates_a = make(F)

    # A bin difference refuses whatever topology= says: may_differ only
    # accepts different connectivity between the same places.
    with pytest.raises(IncompatibleEnvironmentError, match="differ in"):
        compare_spatial_rates(rates_a, make(G, on=other), topology=topology)
    assert rates_a.env.units == "cm"


def test_different_cell_faces_between_the_same_bins_need_may_differ(env, make) -> None:
    # Same bins, neighbors and arena; halving each face's size halves how
    # much of a field smoothing passes between neighbors.
    binning = renumbered(env, np.arange(env.n_bins)).binning
    faces = binning.build_faces()
    narrow = Environment.from_bins(
        replace(
            binning,
            faces=KnownFaces(
                FaceGeometry(faces.edges, faces.A / 2, faces.distance, faces.volumes)
            ),
        ),
        units=env.units,
        arena=env.arena,
    )
    rates_a, rates_b = make(F), make(G, on=narrow)
    with pytest.raises(IncompatibleEnvironmentError, match="face sizes"):
        compare_spatial_rates(rates_a, rates_b)
    result = compare_spatial_rates(rates_a, rates_b, topology="may_differ")
    assert result.analysis_settings["topology_differs"] is True


@pytest.fixture
def door_pair():
    """Same 2x2 bins, with and without diagonal connections."""

    def grid(diagonal: bool) -> Environment:
        grid_env = Environment.from_grid_mask(
            np.ones((2, 2), dtype=bool),
            grid_edges=(np.arange(3.0) * 10,) * 2,
            connect_diagonal_neighbors=diagonal,
        )
        return grid_env.with_metadata(units="cm", frame="world")

    def rates(on: Environment, values) -> SpatialRatesResult:
        return SpatialRatesResult(
            firing_rates=np.asarray([values], dtype=float),
            occupancy=np.ones(4),
            env=on,
            method="binned",
            bandwidth=5.0,
            unit_ids=[1],
        )

    return rates(grid(True), [1, 2, 3, 5]), rates(grid(False), [1, 3, 2, 6])


def test_topology_difference_must_be_declared(door_pair) -> None:
    rates_a, rates_b = door_pair
    with pytest.raises(IncompatibleEnvironmentError, match="weighted connectivity"):
        compare_spatial_rates(rates_a, rates_b)

    result = compare_spatial_rates(rates_a, rates_b, topology="may_differ")

    np.testing.assert_allclose(
        result.correlation, [stats.pearsonr([1, 2, 3, 5], [1, 3, 2, 6]).statistic]
    )
    assert result.analysis_settings["topology"] == "may_differ"
    assert result.analysis_settings["topology_differs"] is True


def test_same_environment_records_no_topology_difference(make) -> None:
    result = compare_spatial_rates(make(F), make(G), topology="may_differ")
    assert result.analysis_settings["topology_differs"] is False


def test_one_to_one_subset_mapping_equals_selecting_the_same_bins(env, make) -> None:
    rng = np.random.default_rng(5)
    mask = np.array([True, True, False, True, True])
    rates_a = make(rng.uniform(0, 10, (2, 5)), ids=[1, 2])
    rates_b_full = rng.uniform(0, 10, (2, 5))
    occupancy_b = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
    subset = env.subset(mask=mask)
    rates_b = make(rates_b_full[:, mask], occupancy_b[mask], ids=[1, 2], on=subset)
    mapping = np.column_stack([np.flatnonzero(mask), np.arange(4)])

    # Bins 1 and 3 neighbor bin 2 in env but not in the subset, so their
    # smoothing used different neighbors: the caller has to accept that.
    with pytest.raises(IncompatibleEnvironmentError, match="outside the mapping"):
        compare_spatial_rates(rates_a, rates_b, bin_mapping=mapping)
    mapped = compare_spatial_rates(
        rates_a, rates_b, bin_mapping=mapping, topology="may_differ"
    )
    identity = compare_spatial_rates(
        rates_a,
        make(rates_b_full, occupancy_b, ids=[1, 2]),
        bins=np.flatnonzero(mask),
    )

    np.testing.assert_array_equal(mapped.correlation, identity.correlation)
    np.testing.assert_array_equal(mapped.n_common_bins, identity.n_common_bins)
    np.testing.assert_array_equal(mapped.bins_a, identity.bins_a)
    assert mapped.bins_b.tolist() == [0, 1, 2, 3]
    assert mapped.analysis_settings["bin_mapping"] == "explicit"
    assert mapped.analysis_settings["topology_differs"] is True
    assert identity.analysis_settings["bin_mapping"] is None
    assert identity.analysis_settings["bins_selected"] == 4
    assert identity.analysis_settings["n_compared_bins"] == 4


def test_bins_selection_restricts_a_mapping(env, make) -> None:
    mask = np.array([True, True, False, True, True])
    rates_b = make(np.asarray(G)[mask], OCC[mask], on=env.subset(mask=mask))
    mapping = np.column_stack([np.flatnonzero(mask), np.arange(4)])
    result = compare_spatial_rates(
        make(F), rates_b, bin_mapping=mapping, bins=[0, 2, 3, 4], topology="may_differ"
    )
    assert result.bins_a.tolist() == [0, 3, 4]
    assert result.bins_b.tolist() == [0, 2, 3]
    # Bin 2 was selected but has no partner in the subset, so it is dropped.
    assert result.analysis_settings["bins_selected"] == 4
    assert result.analysis_settings["n_compared_bins"] == 3


def test_renumbered_environment_has_the_same_topology_through_the_mapping(
    env, make
) -> None:
    order = np.array([3, 0, 4, 1, 2])
    copy = renumbered(env, order)
    rng = np.random.default_rng(8)
    rates_a = make(rng.uniform(0, 10, (2, 5)), ids=[1, 2])
    rates_b = rng.uniform(0, 10, (2, 5))
    rates_b_renumbered = make(rates_b[:, order], ids=[1, 2], on=copy)
    with pytest.raises(IncompatibleEnvironmentError, match="bin coordinates"):
        compare_spatial_rates(rates_a, rates_b_renumbered)

    mapped = compare_spatial_rates(
        rates_a, rates_b_renumbered, bin_mapping=np.column_stack([order, np.arange(5)])
    )

    identity = compare_spatial_rates(rates_a, make(rates_b, ids=[1, 2]))
    assert mapped.analysis_settings["topology_differs"] is False
    np.testing.assert_allclose(mapped.correlation, identity.correlation, atol=1e-12)
    assert mapped.bins_a.tolist() == order.tolist()


@pytest.mark.parametrize("copy", ["same", "with_metadata"])
def test_partial_mapping_between_identical_environments_equals_bin_selection(
    env, make, copy
) -> None:
    rng = np.random.default_rng(9)
    rates_a = make(rng.uniform(0, 10, (2, 5)), ids=[1, 2])
    other = env if copy == "same" else env.with_metadata(name="copy")
    rates_b = make(rng.uniform(0, 10, (2, 5)), ids=[1, 2], on=other)
    kept = np.array([0, 1, 3, 4])

    mapped = compare_spatial_rates(
        rates_a, rates_b, bin_mapping=np.column_stack([kept, kept])
    )
    selected = compare_spatial_rates(rates_a, rates_b, bins=kept)

    assert mapped.analysis_settings["topology_differs"] is False
    assert selected.analysis_settings["topology_differs"] is False
    np.testing.assert_array_equal(mapped.correlation, selected.correlation)
    np.testing.assert_array_equal(mapped.n_common_bins, selected.n_common_bins)
    np.testing.assert_array_equal(mapped.bins_a, selected.bins_a)


def test_mapping_a_whole_connected_component_has_the_same_topology(make) -> None:
    # Two 3-bin corridors with a gap; the second piece has no edges leaving it.
    two_pieces = Environment.from_grid_mask(
        np.array([True, True, True, False, True, True, True])[:, None],
        grid_edges=(np.arange(8.0) * 10, np.arange(2.0) * 10),
    ).with_metadata(units="cm")
    kept = np.array([False, False, False, True, True, True])
    piece = two_pieces.subset(mask=kept)
    rates_a = make([9.0, 9.0, 9.0, 1.0, 2.0, 3.0], np.ones(6), on=two_pieces)
    rates_b = make([2.0, 4.0, 6.0], np.ones(3), on=piece)

    result = compare_spatial_rates(
        rates_a,
        rates_b,
        bin_mapping=np.column_stack([np.flatnonzero(kept), np.arange(3)]),
    )

    assert result.analysis_settings["topology_differs"] is False
    np.testing.assert_allclose(result.correlation, [1.0], atol=1e-12)


@pytest.fixture
def corridor_pair():
    """A 5-bin corridor, and a subset of it with a barrier inside the subset."""

    def corridor(walls=()) -> Environment:
        return Environment.from_grid_mask(
            np.ones((5, 1), dtype=bool),
            grid_edges=(np.arange(6.0) * 10, np.arange(2.0) * 10),
            arena=OpenArena(box(0, 0, 50, 10), walls=walls),
        ).with_metadata(units="cm")

    open_corridor = corridor()
    barrier = corridor(walls=(LineString([(20, 0), (20, 10)]),))
    kept = np.array([True, True, True, False, False])
    subset = barrier.subset(mask=kept)
    # Bins 1 and 2 are neighbors in the open corridor; the barrier separates them.
    assert open_corridor.connectivity.has_edge(1, 2)
    assert not subset.connectivity.has_edge(1, 2)
    return open_corridor, subset, np.column_stack([np.flatnonzero(kept), np.arange(3)])


def test_barrier_inside_a_mapped_subset_is_a_topology_difference(
    make, corridor_pair
) -> None:
    open_corridor, subset, mapping = corridor_pair
    rates_a = make([1.0, 2.0, 4.0, 3.0, 5.0], on=open_corridor)
    rates_b = make([2.0, 1.0, 5.0], OCC[:3], on=subset)

    with pytest.raises(IncompatibleEnvironmentError, match="weighted connectivity"):
        compare_spatial_rates(rates_a, rates_b, bin_mapping=mapping)
    result = compare_spatial_rates(
        rates_a, rates_b, bin_mapping=mapping, topology="may_differ"
    )

    assert result.analysis_settings["topology_differs"] is True
    np.testing.assert_allclose(
        result.correlation, [stats.pearsonr([1, 2, 4], [2, 1, 5]).statistic]
    )


def test_merging_bins_is_refused_because_rates_cannot_be_averaged(env, make) -> None:
    # Two fine bins, 0 Hz over 100 s and 100 Hz over 1 s, hold 100 spikes in
    # 101 s: the merged rate is 0.99 Hz, not the 50 Hz mean of the two rates.
    assert pytest.approx(0.990, abs=1e-3) == (0 * 100 + 100 * 1) / 101
    assert np.mean([0, 100]) == 50
    fine = env.subset(mask=np.array([True, True, False, False, False]))
    coarse = env.subset(mask=np.array([True, False, False, False, False]))
    rates_a = make([[0.0, 100.0]], [100.0, 1.0], on=fine)
    rates_b = make([[0.99]], [101.0], on=coarse)

    with pytest.raises(ValueError, match=r"many-to-one(.|\n)*0\.99 Hz"):
        compare_spatial_rates(rates_a, rates_b, bin_mapping=[[0, 0], [1, 0]])


def test_splitting_a_bin_is_refused(make) -> None:
    with pytest.raises(ValueError, match="one-to-one"):
        compare_spatial_rates(make(F), make(G), bin_mapping=[[0, 0], [0, 1]])


def test_mapping_between_different_places_is_refused(make) -> None:
    mapping = [[0, 1], [1, 0], [2, 2], [3, 3], [4, 4]]
    with pytest.raises(IncompatibleEnvironmentError, match="bin coordinates"):
        compare_spatial_rates(make(F), make(G), bin_mapping=mapping)


def test_mapping_requires_the_same_units(env, make) -> None:
    with pytest.raises(IncompatibleEnvironmentError, match="units"):
        compare_spatial_rates(
            make(F),
            make(G, on=env.with_metadata(units="mm")),
            bin_mapping=np.column_stack([np.arange(5), np.arange(5)]),
        )


_ROOM = OpenArena(box(-20, -20, 30, 30))


def _two_by_two(*, offset: float = 0.0) -> Environment:
    return Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(offset + np.arange(3.0) * 10,) * 2
    )


def _polar_and_cartesian() -> tuple[Environment, Environment]:
    polar = Environment.from_polar_egocentric(
        distance_range=(0.0, 20.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=10.0,
        angle_bin_size=np.pi / 2,
    )
    grid = Environment.from_grid_mask(
        np.ones((2, 4), dtype=bool), grid_edges=(np.arange(3.0), np.arange(5.0))
    )
    return polar, grid


def _clipped_cell(clip) -> Environment:
    parent = Environment.from_grid_mask(
        np.ones((1, 1), dtype=bool),
        grid_edges=(np.arange(2.0),) * 2,
        arena=OpenArena(box(0, 0, 1, 1)),
    )
    return parent.clip(clip)


def _clipped_strip(x_edges: list[float]) -> Environment:
    """Cells [0, 1] x [0, 1] and [1, 2] x [0, 1], cut from a grid on ``x_edges``."""
    parent = Environment.from_grid_mask(
        np.ones((len(x_edges) - 1, 1), dtype=bool),
        grid_edges=(np.asarray(x_edges), np.array([0.0, 1.0])),
        arena=_ROOM,
    )
    return parent.clip(box(0, 0, 2, 1))


def _one_cell(x_edges: list[float], y_edges: list[float]) -> Environment:
    return Environment.from_grid_mask(
        np.ones((1, 1), dtype=bool),
        grid_edges=(np.array(x_edges), np.array(y_edges)),
        arena=_ROOM,
    )


_DIAMOND = [(0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)]

# Every bin difference _environment_differences reports, as (environments,
# reason without a mapping, reason through a mapping): a mapping pairs bins
# explicitly, so it cannot differ in their order.
_BIN_DIFFERENCES = [
    pytest.param(
        _polar_and_cartesian,
        "coordinate systems",
        "coordinate systems",
        id="coordinate systems",
    ),
    pytest.param(
        lambda: (
            _two_by_two().with_metadata(units="cm"),
            _two_by_two().with_metadata(units="m"),
        ),
        "units or coordinate frames",
        "units or coordinate frames",
        id="units",
    ),
    pytest.param(
        lambda: (_two_by_two(), _two_by_two(offset=1000.0)),
        "bin coordinates or ordering",
        "bin coordinates",
        id="bin coordinates",
    ),
    pytest.param(
        lambda: (
            _clipped_cell(box(0.25, 0.25, 0.75, 0.75)),
            _clipped_cell(box(0.1, 0.1, 0.9, 0.9)),
        ),
        "bin measures",
        "bin measures",
        id="bin measures",
    ),
    pytest.param(
        lambda: (
            _clipped_cell(box(0, 0.25, 1, 0.75)),
            _clipped_cell(Polygon(_DIAMOND)),
        ),
        "polygon cell geometry",
        "polygon cell geometry",
        id="polygon cell geometry",
    ),
    pytest.param(
        lambda: (_clipped_strip([0.0, 1.0, 2.0]), _clipped_strip([-1.0, 0, 1, 2])),
        "bin-boundary ownership",
        "bin-boundary ownership",
        id="bin-boundary ownership",
    ),
    pytest.param(
        lambda: (_one_cell([0, 10], [0, 10]), _one_cell([-5, 15], [2.5, 7.5])),
        "cell shapes",
        "cell shapes",
        id="cell shapes",
    ),
]


@pytest.mark.parametrize(("pair", "reason", "mapped_reason"), _BIN_DIFFERENCES)
def test_an_identity_mapping_is_refused_like_no_mapping(
    make, pair, reason, mapped_reason
) -> None:
    env_a, env_b = pair()
    assert env_a.n_bins == env_b.n_bins
    assert _environment_differences(env_a, env_b).bins == reason
    rates_a, rates_b = (
        make(np.arange(1.0, env.n_bins + 1), np.ones(env.n_bins), on=env)
        for env in (env_a, env_b)
    )
    identity = np.column_stack([np.arange(env_a.n_bins)] * 2)

    with pytest.raises(
        IncompatibleEnvironmentError, match=f"differ in {re.escape(reason)}"
    ):
        compare_spatial_rates(rates_a, rates_b, topology="may_differ")
    with pytest.raises(
        IncompatibleEnvironmentError, match=f"differ in {re.escape(mapped_reason)}"
    ):
        compare_spatial_rates(
            rates_a, rates_b, bin_mapping=identity, topology="may_differ"
        )


# The fix each bin difference needs. A mapping cannot bridge coordinate
# labels or different cells; it is the fix only for bins in another order.
_RELABEL = "a bin_mapping cannot pair bins in different"
_ONE_ENVIRONMENT = "re-estimate both conditions on one common environment"
_FIXES = {
    "coordinate systems": (_RELABEL, _RELABEL),
    "units or coordinate frames": (_RELABEL, _RELABEL),
    "bin coordinates or ordering": ("pass a one-to-one bin_mapping", None),
    "bin coordinates": (None, "pair only bins with identical centers"),
    "bin measures": (_ONE_ENVIRONMENT, _ONE_ENVIRONMENT),
    "polygon cell geometry": (_ONE_ENVIRONMENT, _ONE_ENVIRONMENT),
    "bin-boundary ownership": (_ONE_ENVIRONMENT, _ONE_ENVIRONMENT),
    "cell shapes": (_ONE_ENVIRONMENT, _ONE_ENVIRONMENT),
}


@pytest.mark.parametrize(("pair", "reason", "mapped_reason"), _BIN_DIFFERENCES)
def test_a_bin_refusal_suggests_the_fix_for_its_reason(
    make, pair, reason, mapped_reason
) -> None:
    env_a, env_b = pair()
    rates_a, rates_b = (
        make(np.arange(1.0, env.n_bins + 1), np.ones(env.n_bins), on=env)
        for env in (env_a, env_b)
    )
    identity = np.column_stack([np.arange(env_a.n_bins)] * 2)
    for mapping, fix in (
        (None, _FIXES[reason][0]),
        (identity, _FIXES[mapped_reason][1]),
    ):
        with pytest.raises(IncompatibleEnvironmentError) as refused:
            compare_spatial_rates(
                rates_a, rates_b, bin_mapping=mapping, topology="may_differ"
            )
        fix_line = str(refused.value).splitlines()[-1]
        assert fix_line.startswith("Fix: ")
        assert fix in fix_line
        # Only bins in another order are fixed by a mapping.
        suggests_mapping = "pass a one-to-one bin_mapping" in fix_line
        assert suggests_mapping == (
            reason == "bin coordinates or ordering" and mapping is None
        )


@pytest.mark.parametrize("mapped", [False, True])
def test_a_label_refusal_names_both_environments_labels(env, make, mapped) -> None:
    other = env.with_metadata(units="m", frame="room")
    mapping = np.column_stack([np.arange(5)] * 2) if mapped else None
    with pytest.raises(
        IncompatibleEnvironmentError,
        match=re.escape("units or coordinate frames ('cm', None vs 'm', 'room')"),
    ):
        compare_spatial_rates(make(F), make(G, on=other), bin_mapping=mapping)

    polar, grid = _polar_and_cartesian()
    with pytest.raises(
        IncompatibleEnvironmentError,
        match=re.escape(
            "coordinate systems (EgocentricPolarSystem(circular=True) vs "
            "CartesianSystem())"
        ),
    ):
        compare_spatial_rates(
            make(np.arange(1.0, 9), np.ones(8), on=polar),
            make(np.arange(1.0, 9), np.ones(8), on=grid),
            bin_mapping=np.column_stack([np.arange(8)] * 2) if mapped else None,
        )


@pytest.mark.parametrize("mapped", [False, True])
def test_polygon_cells_compare_with_bins_without_cells_by_center(make, mapped) -> None:
    # The copy records no cells, so the bins are compared by center and
    # measure only; they agree, and so do the connections and faces.
    clipped = Environment.from_grid_mask(
        np.ones((4, 4), dtype=bool), grid_edges=(np.arange(5.0),) * 2
    ).clip(box(0, 0, 3.5, 3.5))
    copy = renumbered(clipped, np.arange(clipped.n_bins))
    assert copy.binning.shapes is None
    rates = np.random.default_rng(3).uniform(0, 10, clipped.n_bins)
    identity = np.column_stack([np.arange(clipped.n_bins)] * 2)

    result = compare_spatial_rates(
        make(rates, np.ones(clipped.n_bins), on=clipped),
        make(rates, np.ones(clipped.n_bins), on=copy),
        bin_mapping=identity if mapped else None,
    )

    assert result.analysis_settings["topology_differs"] is False
    np.testing.assert_allclose(result.correlation, [1.0])


def test_a_subset_of_polygon_cells_maps_onto_its_parent(make) -> None:
    # A subset keeps its parent's cells and boundary ownership for the bins
    # it keeps, so only the connections to the dropped bins differ.
    parent = Environment.from_grid_mask(
        np.ones((4, 4), dtype=bool), grid_edges=(np.arange(5.0),) * 2
    ).clip(box(0, 0, 3.5, 3.5))
    kept = np.arange(parent.n_bins) < parent.n_bins - 3
    part = parent.subset(mask=kept)
    mapping = np.column_stack([np.flatnonzero(kept), np.arange(part.n_bins)])
    rates = np.random.default_rng(2).uniform(0, 10, parent.n_bins)
    rates_a = make(rates, np.ones(parent.n_bins), on=parent)
    rates_b = make(rates[kept], np.ones(part.n_bins), on=part)

    result = compare_spatial_rates(
        rates_a, rates_b, bin_mapping=mapping, topology="may_differ"
    )

    assert result.analysis_settings["topology_differs"] is True
    np.testing.assert_allclose(result.correlation, [1.0])


def test_a_renumbered_wall_split_arena_maps_onto_the_original(make) -> None:
    # Each cut cell's pieces keep their relative order, so every bin owns the
    # same boundary points in both environments.
    env = wall_split_arena()
    order = order_preserving_permutation(env.n_bins)
    copy = renumbered_cells(env, order)
    rates = np.random.default_rng(4).uniform(0, 10, env.n_bins)

    result = compare_spatial_rates(
        make(rates, np.ones(env.n_bins), on=env),
        make(rates[order], np.ones(env.n_bins), on=copy),
        bin_mapping=np.column_stack([order, np.arange(env.n_bins)]),
    )

    assert result.analysis_settings["topology_differs"] is False
    np.testing.assert_allclose(result.correlation, [1.0])


def test_swapping_two_touching_pieces_of_a_cut_cell_is_refused(make) -> None:
    # A point on the line between pieces 12 and 13 belongs to the earlier
    # piece, so swapping their numbers moves it to the other piece.
    env = wall_split_arena()
    order = np.arange(env.n_bins)
    order[[12, 13]] = [13, 12]
    copy = renumbered_cells(env, order)
    rates = np.random.default_rng(4).uniform(0, 10, env.n_bins)

    with pytest.raises(IncompatibleEnvironmentError, match="bin-boundary ownership"):
        compare_spatial_rates(
            make(rates, np.ones(env.n_bins), on=env),
            make(rates[order], np.ones(env.n_bins), on=copy),
            bin_mapping=np.column_stack([order, np.arange(env.n_bins)]),
        )


def _clipped_session():
    """Binned maps of one session on a sampled grid and on a clip of it.

    The grid's edges are not round numbers, so the centroids of the cells
    the clip kept whole differ from the grid's centers by rounding.
    """
    rng = np.random.default_rng(5)
    times = np.arange(9000) / 30.0
    positions = rng.uniform(0.0, 40.0, size=(len(times), 2))
    spike_rng = np.random.default_rng(6)
    spike_times = [
        np.sort(spike_rng.choice(times, n, replace=False)) for n in (800, 300)
    ]
    env = Environment.from_samples(positions, bin_size=4.0, units="cm")
    clipped = env.clip(box(0.0, 0.0, 21.0, 40.0))
    maps = [
        compute_spatial_rates(
            where,
            spike_times,
            times,
            positions,
            method="binned",
            bandwidth=0.0,
            unit_ids=[1, 2],
        )
        for where in (env, clipped)
    ]
    return env, clipped, *maps


def test_cells_a_clip_kept_whole_map_onto_their_parent() -> None:
    env, clipped, full_rates, clipped_rates = _clipped_session()
    pairs = clipped.bin_mapping_from(env).pairs
    parent_bins, child_bins = pairs.T
    # Rounding-level centroid differences are accepted as the same cell.
    assert np.any(env.bin_centers[parent_bins] != clipped.bin_centers[child_bins])

    result = compare_spatial_rates(
        full_rates, clipped_rates, bin_mapping=pairs, topology="may_differ"
    )

    np.testing.assert_array_equal(result.bins_a, parent_bins)
    np.testing.assert_array_equal(result.bins_b, child_bins)
    # The same score as correlating the uncut cells' rates directly.
    for row in range(2):
        a = full_rates.firing_rates[row, parent_bins]
        b = clipped_rates.firing_rates[row, child_bins]
        support = (
            np.isfinite(a)
            & np.isfinite(b)
            & (full_rates.occupancy[parent_bins] > 0)
            & (clipped_rates.occupancy[child_bins] > 0)
        )
        assert result.n_common_bins[row] == support.sum()
        np.testing.assert_allclose(
            result.correlation[row],
            stats.pearsonr(a[support], b[support]).statistic,
            rtol=1e-12,
        )


def test_a_cell_moved_by_more_than_rounding_is_refused_by_both() -> None:
    env, clipped, full_rates, _ = _clipped_session()
    pairs = clipped.bin_mapping_from(env).pairs
    # A shift of 1e-7 cm is far above rounding at 40 cm (about 1e-12 cm).
    moved = clipped.apply_transform(
        AffineND(np.array([[1.0, 0.0, 1e-7], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    )
    moved_rates = SpatialRatesResult(
        firing_rates=np.ones((2, moved.n_bins)),
        occupancy=np.ones(moved.n_bins),
        env=moved,
        method="binned",
        bandwidth=0.0,
        unit_ids=[1, 2],
    )

    with pytest.raises(ValueError, match="no bin of this environment"):
        moved.bin_mapping_from(env)
    with pytest.raises(IncompatibleEnvironmentError, match="bin coordinates"):
        compare_spatial_rates(
            full_rates, moved_rates, bin_mapping=pairs, topology="may_differ"
        )


@pytest.mark.parametrize(
    ("mapping", "match"),
    [
        ([0, 1], "shape"),
        ([[0.0, 1.0]], "integer"),
        ([[0, 5]], "out of range"),
        ([[-1, 0]], "out of range"),
    ],
)
def test_malformed_mapping_is_refused(make, mapping, match) -> None:
    with pytest.raises(ValueError, match=match):
        compare_spatial_rates(make(F), make(G), bin_mapping=mapping)


def test_bin_selection_is_validated(make) -> None:
    with pytest.raises(ValueError, match="out of range"):
        compare_spatial_rates(make(F), make(G), bins=[0, 9])


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def test_only_spatial_rate_results_are_compared(rate_family_results, make) -> None:
    directional = rate_family_results["directional"]
    view = rate_family_results["view"]
    rates = make(F)
    for other in (directional, view, directional[0], view[0], np.ones((1, 5))):
        with pytest.raises(TypeError, match="SpatialRatesResult"):
            compare_spatial_rates(other, other)
        with pytest.raises(TypeError, match="SpatialRatesResult"):
            compare_spatial_rates(rates, other)


def test_mixing_single_and_population_results_is_refused(make) -> None:
    rates = make(F)
    with pytest.raises(TypeError, match=r"sel\(unit_id="):
        compare_spatial_rates(rates, rates[0])


@pytest.fixture(scope="module")
def two_sessions():
    times = np.arange(3000) / 30.0
    positions = 50 + 40 * np.column_stack(
        [np.sin(2 * np.pi * times / 20), np.cos(2 * np.pi * times / 13)]
    )
    env = Environment.from_samples(positions, bin_size=10.0, units="cm")
    rng = np.random.default_rng(11)
    spikes = [np.sort(rng.uniform(0, times[-1], n)) for n in (300, 200, 400)]
    first = compute_spatial_rates(
        env, spikes, times, positions, epochs=[[0.0, 50.0]], unit_ids=[4, 5, 6]
    )
    second = compute_spatial_rates(
        env, spikes, times, positions, epochs=[[50.0, 100.0]], unit_ids=[4, 5, 6]
    )
    return first, second


def test_single_units_compare_like_the_population_row(two_sessions) -> None:
    first, second = two_sessions
    population = compare_spatial_rates(first, second)
    single = compare_spatial_rates(first[0], second[0])

    assert len(single) == 1
    assert single.unit_ids.tolist() == [4]
    assert single.unit_ids_only_a.size == single.unit_ids_only_b.size == 0
    assert single.correlation[0] == population.correlation[0]
    assert single.status[0] == population.status[0] == "scored"


def test_single_unit_labels_must_agree(two_sessions) -> None:
    first, second = two_sessions
    with pytest.raises(ValueError, match="unit_id"):
        compare_spatial_rates(first[0], second[1])


def test_single_unit_label_comes_from_whichever_side_has_one(env) -> None:
    from neurospatial.encoding.spatial import SpatialRateResult

    def single(unit_id):
        return SpatialRateResult(
            firing_rate=np.asarray(F),
            occupancy=OCC,
            env=env,
            method="binned",
            bandwidth=5.0,
            unit_id=unit_id,
        )

    assert compare_spatial_rates(single(None), single(8)).unit_ids.tolist() == [8]
    assert compare_spatial_rates(single(None), single(None)).unit_ids.tolist() == [None]


def test_different_estimator_settings_warn(make) -> None:
    with pytest.warns(UserWarning, match="bandwidth"):
        result = compare_spatial_rates(make(F), make(G, bandwidth=10.0))
    settings = result.analysis_settings
    assert settings["estimator_a"]["bandwidth"] == 5.0
    assert settings["estimator_b"]["bandwidth"] == 10.0


def test_same_estimator_settings_do_not_warn(make, recwarn) -> None:
    compare_spatial_rates(make(F), make(G))
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_no_occupancy_threshold_is_the_same_whether_none_or_zero(
    two_sessions, recwarn
) -> None:
    # compute_spatial_rates records min_occupancy=0.0; a hand-built result
    # without settings has none. Both mean "no threshold".
    first, second = two_sessions
    assert first.analysis_settings["min_occupancy"] == 0.0
    manual = SpatialRatesResult(
        firing_rates=second.firing_rates,
        occupancy=second.occupancy,
        env=second.env,
        method=second.method,
        bandwidth=second.bandwidth,
        unit_ids=second.unit_ids,
    )

    result = compare_spatial_rates(first, manual)

    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]
    assert result.analysis_settings["estimator_a"]["min_occupancy"] == 0.0
    assert result.analysis_settings["estimator_b"]["min_occupancy"] == 0.0
    np.testing.assert_array_equal(
        result.correlation, compare_spatial_rates(first, second).correlation
    )


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "reference"),
    [("pearson", stats.pearsonr), ("spearman", stats.spearmanr)],
)
def test_vectorized_scores_match_scipy_per_unit(make, method, reference) -> None:
    rng = np.random.default_rng(2)
    rates_a = rng.uniform(0, 10, (8, 5))
    rates_b = rng.uniform(0, 10, (8, 5))
    rates_a[:, 0] = rates_a[:, 1]  # ties for the rank method
    rates_a[rng.random((8, 5)) < 0.2] = np.nan
    rates_b[rng.random((8, 5)) < 0.2] = np.nan
    ids = list(range(8))

    result = compare_spatial_rates(
        make(rates_a, ids=ids), make(rates_b, ids=ids), method=method
    )

    common = np.isfinite(rates_a) & np.isfinite(rates_b)
    np.testing.assert_array_equal(result.common_support, common)
    np.testing.assert_array_equal(result.n_common_bins, common.sum(axis=1))
    scored = result.status == "scored"
    assert scored.sum() >= 4
    expected = [
        reference(rates_a[u, common[u]], rates_b[u, common[u]]).statistic
        for u in np.flatnonzero(scored)
    ]
    np.testing.assert_allclose(result.correlation[scored], expected, atol=1e-12)
    assert np.all(np.isnan(result.correlation[~scored]))
    assert result.method == method


def test_rate_difference_is_b_minus_a_on_common_support(make) -> None:
    result = compare_spatial_rates(make(F, [1, 1, 1, 1, 0]), make(G))
    expected = np.array(G) - np.array(F)
    expected[4] = np.nan
    np.testing.assert_array_equal(result.rate_difference[0], expected)
    assert not hasattr(result, "firing_rates")
    assert not hasattr(result, "firing_rate")


def test_settings_record_the_comparison(make) -> None:
    settings = compare_spatial_rates(make(F), make(G)).analysis_settings
    assert settings == {
        "method": "pearson",
        "unit_selection": "intersection",
        "min_common_bins": 3,
        "topology": "same",
        "topology_differs": False,
        "bin_mapping": None,
        "n_compared_bins": 5,
        "bins_selected": None,
        "estimator_a": {"method": "binned", "bandwidth": 5.0, "min_occupancy": 0.0},
        "estimator_b": {"method": "binned", "bandwidth": 5.0, "min_occupancy": 0.0},
        "units": "cm",
    }


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


_RESULT_ARRAYS = (
    "unit_ids",
    "correlation",
    "status",
    "n_common_bins",
    "bins_a",
    "bins_b",
    "firing_rates_a_on_bins",
    "firing_rates_b_on_bins",
    "occupancy_a",
    "occupancy_b",
    "occupied_a",
    "occupied_b",
    "common_support",
    "unit_ids_only_a",
    "unit_ids_only_b",
)


def test_result_owns_read_only_arrays(make) -> None:
    rates_a = make([F, G, F], ids=[1, 2, 3])
    rates_b = make([G, F, G], ids=[1, 2, 4])
    result = compare_spatial_rates(rates_a, rates_b)
    inputs = [
        np.asarray(value)
        for rates in (rates_a, rates_b)
        for value in (rates.firing_rates, rates.occupancy, rates.unit_ids)
    ]

    for name in _RESULT_ARRAYS:
        array = getattr(result, name)
        assert not array.flags.writeable, name
        assert not any(np.shares_memory(array, source) for source in inputs), name
        with pytest.raises(ValueError, match="read-only"):
            array[...] = array
    difference = result.rate_difference
    with pytest.raises(ValueError, match="read-only"):
        difference[0, 0] = 0.0
    # A one-unit selection owns its arrays too.
    child = result[0]
    for name in _RESULT_ARRAYS:
        array = getattr(child, name)
        assert not array.flags.writeable, name
        assert not any(np.shares_memory(array, source) for source in inputs), name


def test_indexing_returns_one_unit_and_sel_selects_by_label(make) -> None:
    result = compare_spatial_rates(
        make([F, G, F], ids=[1, 2, 3]), make([F, F, G], ids=[1, 2, 3])
    )

    second = result[1]

    assert isinstance(second, RateComparisonResult)
    assert len(result) == 3 and len(second) == 1
    assert second.unit_ids.tolist() == [2]
    assert second.correlation[0] == result.correlation[1]
    assert second.firing_rates_a_on_bins.shape == (1, 5)
    np.testing.assert_array_equal(second.bins_a, result.bins_a)
    assert second.analysis_settings == result.analysis_settings
    assert result.sel(unit_id=3).correlation[0] == result.correlation[2]
    with pytest.raises(TypeError, match="sel"):
        result["a"]
    with pytest.raises(IndexError):
        result[3]


def test_construction_checks_shapes(env) -> None:
    with pytest.raises(ValueError, match="correlation"):
        RateComparisonResult(
            unit_ids=np.array([1, 2]),
            correlation=np.ones(3),
            status=np.array(["scored", "scored"]),
            n_common_bins=np.array([5, 5]),
            bins_a=np.arange(5),
            bins_b=np.arange(5),
            firing_rates_a_on_bins=np.ones((2, 5)),
            firing_rates_b_on_bins=np.ones((2, 5)),
            occupancy_a=np.ones(5),
            occupancy_b=np.ones(5),
            occupied_a=np.ones(5, dtype=bool),
            occupied_b=np.ones(5, dtype=bool),
            common_support=np.ones((2, 5), dtype=bool),
            env=env,
            method="pearson",
            unit_ids_only_a=np.array([]),
            unit_ids_only_b=np.array([]),
        )
