"""Comparing rate maps between conditions: unit identity, bins and support.

A comparison pairs units by label and bins by place. Unsupported bins (no
sufficient occupancy, or a NaN rate) never enter a score, and an undefined
score is NaN with a reason, never a fabricated 0 or ±1.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from neurospatial import Environment, IncompatibleEnvironmentError
from neurospatial.encoding import (
    RateComparisonResult,
    compare_spatial_rates,
    compute_spatial_rates,
)
from neurospatial.encoding.spatial import SpatialRatesResult

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
    ) -> SpatialRatesResult:
        return SpatialRatesResult(
            firing_rates=np.atleast_2d(np.asarray(rates, dtype=float)),
            occupancy=np.asarray(occupancy, dtype=float),
            env=env if on is None else on,
            method="binned",
            bandwidth=bandwidth,
            unit_ids=None if ids is None else list(ids),
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


def test_min_occupancy_is_read_from_each_results_settings(make) -> None:
    rates_a = make(F, [1, 0.2, 1, 1, 1], settings={"min_occupancy": 0.5})
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


@pytest.mark.parametrize("mismatch", ["units", "frame", "coordinates"])
def test_equal_shapes_cannot_bypass_environment_validation(env, make, mismatch) -> None:
    if mismatch == "coordinates":
        other = Environment.from_samples(
            np.linspace(1000, 1040, 41)[:, None], bin_size=10.0, units="cm"
        )
    else:
        other = env.with_metadata(**{mismatch: "mm" if mismatch == "units" else "x"})
    assert other.n_bins == env.n_bins
    rates_a = make(F)

    with pytest.raises(IncompatibleEnvironmentError, match="differ in"):
        compare_spatial_rates(rates_a, make(G, on=other))
    assert rates_a.env.units == "cm"


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
        "estimator_a": {"method": "binned", "bandwidth": 5.0, "min_occupancy": None},
        "estimator_b": {"method": "binned", "bandwidth": 5.0, "min_occupancy": None},
        "units": "cm",
    }


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


def test_result_owns_read_only_arrays(make) -> None:
    rates = np.array([F, G])
    occupancy = OCC.copy()
    rates_a = make(rates, occupancy, ids=[1, 2])
    result = compare_spatial_rates(rates_a, make([G, F], ids=[1, 2]))
    before = result.firing_rates_a_on_bins.copy()

    rates[:] = 0
    occupancy[:] = 0

    np.testing.assert_array_equal(result.firing_rates_a_on_bins, before)
    assert not np.shares_memory(result.firing_rates_a_on_bins, rates_a.firing_rates)
    with pytest.raises(ValueError, match="read-only"):
        result.correlation[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        result.rate_difference[0, 0] = 0.0


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
