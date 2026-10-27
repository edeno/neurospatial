"""Population-vector correlation of two rate results: units by label, bins by place.

Entry ``c`` of a population vector must be the same cell in both conditions,
and both vectors must be read at the same place. With two ``SpatialRatesResult``
the pairing comes from ``unit_ids`` and the environments, never from row
order, and only bins observed in both conditions are compared (missing is not
zero).
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment, IncompatibleEnvironmentError
from neurospatial.encoding import (
    compute_spatial_rates,
    population_vector_correlation,
)
from neurospatial.encoding.spatial import SpatialRatesResult

N_BINS = 6


@pytest.fixture(scope="module")
def env() -> Environment:
    env = Environment.from_samples(
        np.linspace(0, 50, 51)[:, None], bin_size=10.0, units="cm"
    )
    assert env.n_bins == N_BINS
    return env


@pytest.fixture
def make(env):
    def make(rates, ids, occupancy=None, *, on=None, min_occupancy=0.0):
        on = env if on is None else on
        return SpatialRatesResult(
            firing_rates=np.asarray(rates, dtype=float),
            occupancy=np.ones(on.n_bins) if occupancy is None else occupancy,
            env=on,
            method="binned",
            bandwidth=5.0,
            unit_ids=None if ids is None else list(ids),
            min_occupancy=min_occupancy,
        )

    return make


@pytest.fixture(scope="module")
def maps() -> tuple[np.ndarray, np.ndarray]:
    """Five units' maps in two conditions; condition 2 is a noisy copy."""
    rng = np.random.default_rng(11)
    first = rng.uniform(0.0, 10.0, (5, N_BINS))
    second = first + rng.normal(0.0, 1.0, first.shape)
    return first, second


IDS = np.array([7, 3, 5, 9, 1])


def test_permuting_one_condition_gives_the_same_answer_with_labels(make, maps):
    first, second = maps
    expected = population_vector_correlation(make(first, IDS), make(second, IDS))
    assert np.isfinite(expected).all()
    assert expected.min() > 0.5  # the conditions share their code

    for order in ([4, 3, 2, 1, 0], [1, 2, 3, 4, 0], [2, 0, 4, 1, 3]):
        permuted = make(second[order], IDS[order])
        np.testing.assert_array_equal(
            population_vector_correlation(make(first, IDS), permuted), expected
        )


def test_bare_arrays_pair_rows_by_position_so_permuting_gives_a_wrong_answer(maps):
    """Why identity matters: positional pairing correlates different cells."""
    first, second = maps
    paired = population_vector_correlation(first, second)
    shuffled = population_vector_correlation(first, second[[4, 3, 2, 1, 0]])
    assert np.median(paired) > 0.5
    assert np.median(shuffled) < 0.5
    assert not np.allclose(paired, shuffled)


def test_results_match_the_array_form_when_rows_already_line_up(make, maps):
    first, second = maps
    second = second.copy()
    second[2, 4] = np.nan
    occupancy = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    rates_1, rates_2 = make(first, IDS), make(second, IDS, occupancy)

    from_results = population_vector_correlation(rates_1, rates_2)
    from_arrays = population_vector_correlation(
        rates_1.firing_rates,
        rates_2.firing_rates,
        spatial_support=rates_1.spatial_support & rates_2.spatial_support,
    )

    np.testing.assert_array_equal(from_results, from_arrays)
    assert np.isnan(from_results[[2, 4]]).all()


def test_intersection_uses_shared_units_in_label_order(make, maps):
    first, second = maps
    rates_1 = make(first[:4], [1, 2, 3, 4])
    rates_2 = make(second[[3, 2, 1, 4]], [4, 3, 2, 5])

    result = population_vector_correlation(rates_1, rates_2)

    shared = population_vector_correlation(first[1:4], second[1:4])
    np.testing.assert_array_equal(result, shared)


def test_exact_selection_refuses_unshared_units_naming_both_sides(make, maps):
    first, second = maps
    rates_1 = make(first[:4], [1, 2, 3, 4])
    rates_2 = make(second[[3, 2, 1, 4]], [4, 3, 2, 5])

    with pytest.raises(
        ValueError,
        match=re.escape("only in firing_rates_1: [1], only in firing_rates_2: [5]"),
    ) as caught:
        population_vector_correlation(rates_1, rates_2, unit_selection="exact")
    assert "population_vector_correlation()" in str(caught.value)
    assert "unit_selection='intersection'" in str(caught.value)


def test_exact_selection_accepts_the_same_units_in_any_order(make, maps):
    first, second = maps
    order = [2, 0, 4, 1, 3]
    exact = population_vector_correlation(
        make(first, IDS), make(second[order], IDS[order]), unit_selection="exact"
    )
    np.testing.assert_array_equal(exact, population_vector_correlation(first, second))


def test_unvisited_bins_are_nan_never_zero(make, maps):
    """An unsupported bin is missing whatever finite rate the map holds there."""
    first, second = maps
    occupancy = np.array([0.0, 1.0, 1.0, 1.0, 0.02, 1.0])
    kept = second.copy()
    kept[:, [0, 4]] = np.nan  # NaN-kept map
    filled = second.copy()
    filled[:, 0] = [0.0, 0.0, 3.0, 0.0, 0.0]  # unvisited: filled or extrapolated
    filled[:, 4] = [1.0, 0.0, 0.0, 2.0, 0.0]  # visited for less than min_occupancy

    pv_kept = population_vector_correlation(
        make(first, IDS),
        make(kept, IDS, occupancy, min_occupancy=0.05),
    )
    pv_filled = population_vector_correlation(
        make(first, IDS),
        make(filled, IDS, occupancy, min_occupancy=0.05),
    )

    np.testing.assert_array_equal(pv_kept, pv_filled)
    assert np.isnan(pv_filled[[0, 4]]).all()
    assert np.isfinite(pv_filled[[1, 2, 3, 5]]).all()
    # Read as observed rates, the locations would have entered the comparison.
    as_zero = population_vector_correlation(first, filled)
    assert np.isfinite(as_zero[[0, 4]]).all()


def test_a_unit_left_out_by_intersection_does_not_remove_bins(make, maps):
    first, second = maps
    with_extra = np.vstack([first, np.full((1, N_BINS), np.nan)])
    rates_1 = make(with_extra, [*IDS, 99])
    assert not rates_1.spatial_support.any()  # the extra unit is NaN everywhere

    pv = population_vector_correlation(rates_1, make(second, IDS))

    np.testing.assert_array_equal(pv, population_vector_correlation(first, second))


def test_generated_unit_ids_are_refused(make, maps):
    first, second = maps
    for rates_1, rates_2, name in (
        (make(first, None), make(second, IDS), "firing_rates_1"),
        (make(first, IDS), make(second, None), "firing_rates_2"),
    ):
        with pytest.raises(ValueError, match=f"{name} has no unit_ids") as caught:
            population_vector_correlation(rates_1, rates_2)
        assert "unit_ids= to compute_spatial_rates()" in str(caught.value)


def test_mixing_a_result_with_an_array_is_refused(make, maps):
    first, second = maps
    result = make(first, IDS)
    with pytest.raises(
        TypeError, match="SpatialRatesResult for firing_rates_1 and an array for"
    ) as caught:
        population_vector_correlation(result, second)
    assert "rates.firing_rates" in str(caught.value)
    with pytest.raises(
        TypeError, match="SpatialRatesResult for firing_rates_2 and an array for"
    ):
        population_vector_correlation(first, result)


def test_single_unit_result_is_refused(make, maps):
    first, second = maps
    rates = make(first, IDS)
    with pytest.raises(TypeError, match="single-unit SpatialRateResult"):
        population_vector_correlation(rates[0], make(second, IDS)[0])


def test_fewer_than_two_shared_units_is_refused(make, maps):
    first, second = maps
    with pytest.raises(ValueError, match="only unit \\[2\\] is recorded in both"):
        population_vector_correlation(make(first[:2], [1, 2]), make(second[:2], [2, 3]))


def test_two_shared_units_are_refused(make, maps):
    """Two units always correlate at +-1 at every location: undefined."""
    first, second = maps
    with pytest.raises(ValueError, match="only units \\[2, 3\\] are recorded in both"):
        population_vector_correlation(
            make(first[:3], [1, 2, 3]), make(second[:3], [2, 3, 4])
        )


def test_spatial_support_with_results_is_refused(make, maps):
    first, second = maps
    rates_1, rates_2 = make(first, IDS), make(second, IDS)
    with pytest.raises(ValueError, match="spatial_support= with two") as caught:
        population_vector_correlation(
            rates_1,
            rates_2,
            spatial_support=rates_1.spatial_support & rates_2.spatial_support,
        )
    assert "drop spatial_support=" in str(caught.value)


def test_different_environments_are_refused(make, maps, env):
    first, second = maps
    shifted = Environment.from_samples(
        np.linspace(5, 55, 51)[:, None], bin_size=10.0, units="cm"
    )
    assert shifted.n_bins == env.n_bins  # same count, different places
    with pytest.raises(IncompatibleEnvironmentError, match="bin j of one map"):
        population_vector_correlation(make(first, IDS), make(second, IDS, on=shifted))


def test_bin_mapping_pairs_a_subset_by_place(make, maps, env):
    first, second = maps
    keep = np.array([False, True, True, True, True, False])
    part = env.subset(mask=keep)
    mapping = np.column_stack([np.flatnonzero(keep), np.arange(part.n_bins)])
    on_part = make(second[:, keep], IDS, on=part)

    with pytest.raises(IncompatibleEnvironmentError, match="topology"):
        population_vector_correlation(make(first, IDS), on_part, bin_mapping=mapping)
    pv = population_vector_correlation(
        make(first, IDS), on_part, bin_mapping=mapping, topology="may_differ"
    )

    assert pv.shape == (env.n_bins,)
    assert np.isnan(pv[~keep]).all()
    np.testing.assert_array_equal(
        pv[keep], population_vector_correlation(first[:, keep], second[:, keep])
    )


def test_bin_mapping_pairs_the_cells_a_clip_kept_whole(make):
    """Clipped cells kept whole are the same cells, whatever their centroids."""
    positions = np.random.default_rng(4).uniform(0.0, 40.0, size=(3000, 2))
    grid = Environment.from_samples(positions, bin_size=4.0, units="cm")
    clipped = grid.clip(box(0.0, 0.0, 21.0, 40.0))
    pairs = clipped.bin_mapping_from(grid).pairs
    parents, children = pairs.T
    assert np.any(grid.bin_centers[parents] != clipped.bin_centers[children])
    rng = np.random.default_rng(12)
    first = rng.uniform(0.0, 10.0, (5, grid.n_bins))
    second = rng.uniform(0.0, 10.0, (5, clipped.n_bins))

    pv = population_vector_correlation(
        make(first, IDS, on=grid),
        make(second, IDS, on=clipped),
        bin_mapping=pairs,
        topology="may_differ",
    )

    unpaired = np.setdiff1d(np.arange(grid.n_bins), parents)
    assert np.isnan(pv[unpaired]).all()
    expected = [
        np.corrcoef(first[:, a], second[:, b])[0, 1]
        for a, b in zip(parents, children, strict=True)
    ]
    np.testing.assert_allclose(pv[parents], expected, rtol=1e-12)


@pytest.mark.parametrize(
    ("options", "named"),
    [
        ({"unit_selection": "exact"}, "unit_selection='exact'"),
        ({"bin_mapping": np.array([[0, 0]])}, "bin_mapping="),
        ({"topology": "may_differ"}, "topology='may_differ'"),
    ],
)
def test_results_only_options_are_refused_with_arrays(maps, options, named):
    first, second = maps
    with pytest.raises(ValueError, match=re.escape(f"got {named} with two arrays")):
        population_vector_correlation(first, second, **options)


def test_invalid_option_values_are_refused(make, maps):
    first, second = maps
    with pytest.raises(ValueError, match="unit_selection='all'"):
        population_vector_correlation(first, second, unit_selection="all")
    with pytest.raises(ValueError, match="topology='different'"):
        population_vector_correlation(
            make(first, IDS), make(second, IDS), topology="different"
        )


def test_reordered_sorter_output_gives_the_same_correlation():
    """End to end: listing the units in another order in one session half."""
    times = np.arange(3000) / 30.0
    positions = 50 + 40 * np.column_stack(
        [np.sin(2 * np.pi * times / 20), np.cos(2 * np.pi * times / 13)]
    )
    env = Environment.from_samples(positions, bin_size=10.0, units="cm")
    rng = np.random.default_rng(5)
    spike_times = [np.sort(rng.uniform(0, 100, n)) for n in (300, 200, 400, 250)]
    labels = ["t1c1", "t1c2", "t2c1", "t3c4"]

    def rates(trains, unit_ids, window):
        return compute_spatial_rates(
            env, trains, times, positions, epochs=[window], unit_ids=unit_ids
        )

    first = rates(spike_times, labels, [0.0, 50.0])
    second = rates(spike_times, labels, [50.0, 100.0])
    order = [2, 0, 3, 1]
    reordered = rates(
        [spike_times[i] for i in order], [labels[i] for i in order], [50.0, 100.0]
    )

    expected = population_vector_correlation(first, second)
    np.testing.assert_array_equal(
        population_vector_correlation(first, reordered), expected
    )
    shared = first.spatial_support & second.spatial_support
    assert np.isnan(expected[~shared]).all()
    assert np.isfinite(expected[shared]).any()


def test_different_estimator_settings_warn_as_in_compare_spatial_rates(env, maps):
    """Smoothing differences change the maps, so the comparison says so."""
    first, second = maps

    def estimated(rates, bandwidth):
        return SpatialRatesResult(
            firing_rates=rates,
            occupancy=np.ones(env.n_bins),
            env=env,
            method="binned",
            bandwidth=bandwidth,
            unit_ids=list(IDS),
        )

    with pytest.warns(UserWarning) as caught:
        pv = population_vector_correlation(
            estimated(first, 5.0), estimated(second, 10.0)
        )
    assert len(caught) == 1
    assert str(caught[0].message).startswith(
        "population_vector_correlation(): firing_rates_1 and firing_rates_2 were "
        "estimated with different bandwidth (bandwidth: 5.0 vs 10.0)."
    )
    assert caught[0].filename == __file__  # points at the caller's line
    np.testing.assert_array_equal(pv, population_vector_correlation(first, second))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        population_vector_correlation(estimated(first, 5.0), estimated(second, 5.0))


def test_rates_are_missing_outside_each_session_support():
    """Smoothed rates stop at the visited bins, so bare arrays match results."""
    times = np.arange(3000) / 30.0
    positions = 50 + 40 * np.column_stack(
        [np.sin(2 * np.pi * times / 20), np.cos(2 * np.pi * times / 13)]
    )
    env = Environment.from_samples(positions, bin_size=8.0, units="cm")
    rng = np.random.default_rng(5)
    spike_times = [np.sort(rng.uniform(0, 100, n)) for n in (300, 200, 400, 250)]
    labels = ["t1c1", "t1c2", "t2c1", "t3c4"]
    first, second = (
        compute_spatial_rates(
            env, spike_times, times, positions, epochs=[window], unit_ids=labels
        )
        for window in ([0.0, 50.0], [50.0, 100.0])
    )
    shared = first.spatial_support & second.spatial_support
    assert not shared.all()  # each half of the session misses some bins
    for rates in (first, second):
        visited = np.asarray(rates.occupancy) > 0
        np.testing.assert_array_equal(
            np.isfinite(np.asarray(rates.firing_rates)).all(axis=0), visited
        )

    from_results = population_vector_correlation(first, second)
    assert np.isnan(from_results[~shared]).all()
    np.testing.assert_array_equal(
        from_results,
        population_vector_correlation(first.firing_rates, second.firing_rates),
    )
