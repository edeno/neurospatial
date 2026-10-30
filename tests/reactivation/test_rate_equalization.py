"""Equalizing each unit's firing rate before and after the task (Giri 2019)."""

from __future__ import annotations

import dataclasses
import zlib

import numpy as np
import pytest

from neurospatial.decoding import bin_spikes_in_time
from neurospatial.reactivation import (
    GIRI_2019,
    KUDRIMOTI_1999,
    PEYRACHE_2010,
    ReactivationMethod,
    ReactivationWindows,
    compare_reactivation_methods,
    equalize_firing_rates,
    explained_variance_reactivation,
    pairwise_correlations,
    simulated_reactivation_session,
)

# ---------------------------------------------------------------------------
# equalize_firing_rates
# ---------------------------------------------------------------------------


def _counts(n_bins: int, rates: list[float], seed: int) -> np.ndarray:
    return (
        np.random.default_rng(seed).poisson(rates, (n_bins, len(rates))).astype(float)
    )


def test_equal_lengths_give_each_unit_the_smaller_count_in_both_periods() -> None:
    before = _counts(400, [0.5, 2.0, 1.0, 3.0], seed=0)
    after = _counts(400, [1.5, 1.0, 1.0, 0.2], seed=1)
    new_before, new_after = equalize_firing_rates(before, after, rng=0)

    smaller = np.minimum(before.sum(axis=0), after.sum(axis=0))
    np.testing.assert_array_equal(new_before.sum(axis=0), smaller)
    np.testing.assert_array_equal(new_after.sum(axis=0), smaller)
    # Spikes are only removed, bin by bin, and only from the period that
    # fired more; the other period is unchanged.
    assert np.all(new_before <= before) and np.all(new_after <= after)
    before_higher = before.sum(axis=0) > after.sum(axis=0)
    np.testing.assert_array_equal(new_after[:, before_higher], after[:, before_higher])
    np.testing.assert_array_equal(
        new_before[:, ~before_higher], before[:, ~before_higher]
    )
    assert before_higher.any() and (~before_higher).any()


def test_different_lengths_equalize_the_rate_per_bin() -> None:
    before = _counts(500, [1.0, 2.0, 0.5], seed=2)
    after = _counts(800, [2.0, 1.0, 0.5], seed=3)
    new_before, new_after = equalize_firing_rates(before, after, rng=1)

    rate = np.minimum(before.sum(axis=0) / 500, after.sum(axis=0) / 800)
    np.testing.assert_array_equal(new_before.sum(axis=0), np.round(rate * 500))
    np.testing.assert_array_equal(new_after.sum(axis=0), np.round(rate * 800))
    # Equal rates up to rounding to a whole spike.
    gap = np.abs(new_before.mean(axis=0) - new_after.mean(axis=0))
    assert np.all(gap <= 0.5 / 500 + 1e-12)


def test_kept_spikes_are_a_uniform_sample_of_the_units_spikes() -> None:
    """Keeping 5 of 10 spikes uniformly keeps each bin's spikes with
    probability 1/2: on average 3, 1.5, 0.5 and 0 of counts 6, 3, 1, 0."""
    after = np.array([[6.0], [3.0], [1.0], [0.0]])
    before = np.array([[5.0], [0.0], [0.0], [0.0]])
    generator = np.random.default_rng(4)
    kept = np.array(
        [
            equalize_firing_rates(before, after, rng=generator)[1][:, 0]
            for _ in range(4000)
        ]
    )
    assert np.all(kept.sum(axis=1) == 5)
    assert np.all(kept <= after[:, 0])
    # Hypergeometric standard deviation of the first bin: sqrt(5 * 0.6 * 0.4
    # * 5 / 9) = 0.82, so the mean of 4000 draws is within 0.05 of 3.
    np.testing.assert_allclose(kept.mean(axis=0), [3.0, 1.5, 0.5, 0.0], atol=0.05)
    np.testing.assert_allclose(
        kept[:, 0].std(), np.sqrt(5 * 0.6 * 0.4 * 5 / 9), rtol=0.05
    )


def test_subsamples_are_seeded() -> None:
    before = _counts(300, [1.0, 2.0], seed=5)
    after = _counts(300, [2.0, 1.0], seed=6)
    first = equalize_firing_rates(before, after, rng=7)
    again = equalize_firing_rates(before, after, rng=7)
    other = equalize_firing_rates(before, after, rng=8)
    for a, b in zip(first, again, strict=True):
        np.testing.assert_array_equal(a, b)
    assert not all(np.array_equal(a, b) for a, b in zip(first, other, strict=True))


@pytest.mark.parametrize(
    ("before", "after", "message"),
    [
        (np.full((5, 2), 0.5), np.ones((5, 2)), "fractional"),
        (np.ones((5, 2)), np.ones((5, 3)), "same units"),
        (-np.ones((5, 2)), np.ones((5, 2)), "negative"),
        (np.ones((0, 2)), np.ones((5, 2)), "at least one time bin"),
    ],
)
def test_refuses_counts_it_cannot_subsample(before, after, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        equalize_firing_rates(before, after)


# ---------------------------------------------------------------------------
# ReactivationMethod(n_rate_subsamples=...)
# ---------------------------------------------------------------------------

GROUPS = {"tetrode": np.arange(30) // 3}
KUDRIMOTI_ONE_WINDOW = dataclasses.replace(
    KUDRIMOTI_1999, match_windows=ReactivationWindows(length=600.0, count=1)
)
EQUALIZED = dataclasses.replace(
    KUDRIMOTI_ONE_WINDOW, name="kudrimoti_equalized", n_rate_subsamples=10
)


def _values(session, seed: int):
    table = compare_reactivation_methods(
        session.spike_times,
        session.template_epochs,
        session.match_epochs,
        session.control_epochs,
        reactivation_methods=[KUDRIMOTI_ONE_WINDOW, EQUALIZED],
        unit_groups=GROUPS,
        rng=seed,
    )
    return table.set_index("method")


@pytest.fixture(scope="module")
def tables():
    sessions = {
        "rate_change": {"reactivation": 0.0, "rate_change": 1.0},
        "no_reactivation": {"reactivation": 0.0},
        "reactivated": {"rate_change": 1.0},
    }
    return {
        name: [
            _values(
                simulated_reactivation_session(duration=600.0, rng=seed, **options),
                seed,
            )
            for seed in range(3)
        ]
        for name, options in sessions.items()
    }


def test_equal_rates_bring_a_rate_change_to_the_null_level(tables) -> None:
    """Members fire twice as much after the task without firing together:
    EV with every spike calls it reactivation; with equal rates it falls to
    the EV of the same simulation without the rate change."""
    for changed, null in zip(
        tables["rate_change"], tables["no_reactivation"], strict=True
    ):
        assert changed.loc["kudrimoti_1999", "value"] > 0.05
        assert changed.loc["kudrimoti_1999", "is_reactivated"]
        equalized = changed.loc["kudrimoti_equalized", "value"]
        assert equalized < 0.02
        assert abs(equalized - null.loc["kudrimoti_equalized", "value"]) < 0.01
        assert not changed.loc["kudrimoti_equalized", "is_reactivated"]


def test_equal_rates_keep_real_reactivation(tables) -> None:
    """With assembly events after the task and the rate change, EV stays far
    above the null level after equalizing (lower, since the subsample also
    removes spikes of the co-firing)."""
    for table in tables["reactivated"]:
        equalized = table.loc["kudrimoti_equalized", "value"]
        assert 0.5 < equalized < table.loc["kudrimoti_1999", "value"]
        assert table.loc["kudrimoti_equalized", "is_reactivated"]


def test_values_are_medians_over_subsamples_of_ev_averaged_over_controls() -> None:
    """Recompute two control windows' rate-equalized EV with the public
    functions and the method's documented random stream."""
    session = simulated_reactivation_session(duration=600.0, rng=3, rate_change=0.5)
    method = ReactivationMethod(
        name="two_controls",
        reference="test",
        measure="explained_variance",
        bin_size=0.1,
        control_windows=ReactivationWindows(length=300.0),
        n_rate_subsamples=5,
    )
    table = compare_reactivation_methods(
        session.spike_times,
        session.template_epochs,
        session.match_epochs,
        session.control_epochs,
        reactivation_methods=[method],
        rng=11,
    )

    def counts(epochs):
        return bin_spikes_in_time(session.spike_times, dt=0.1, epochs=epochs)[0]

    template = pairwise_correlations(counts(session.template_epochs))
    match = counts(session.match_epochs).astype(float)
    controls = [counts([[0.0, 300.0]]), counts([[300.0, 600.0]])]
    generator = np.random.default_rng(
        np.random.SeedSequence(
            table.attrs["rng_entropy"],
            spawn_key=(zlib.crc32(b"two_controls"), 0, 1),
        )
    )
    explained, reversed_ = [], []
    for _ in range(5):
        draws = []
        for control in controls:
            new_control, new_match = equalize_firing_rates(
                control.astype(float), match, rng=generator
            )
            draws.append(
                explained_variance_reactivation(
                    template,
                    pairwise_correlations(new_match),
                    control_correlations=pairwise_correlations(new_control),
                )
            )
        explained.append(np.mean([d.explained_variance for d in draws]))
        reversed_.append(np.mean([d.reversed_explained_variance for d in draws]))
    row = table.iloc[0]
    assert row["n_control_windows"] == 2
    np.testing.assert_allclose(row["explained_variance"], np.median(explained))
    np.testing.assert_allclose(row["reversed_explained_variance"], np.median(reversed_))


def test_rate_subsampled_tables_are_seeded() -> None:
    session = simulated_reactivation_session(duration=600.0, rng=0, rate_change=1.0)

    def table(seed):
        return compare_reactivation_methods(
            session.spike_times,
            session.template_epochs,
            session.match_epochs,
            session.control_epochs,
            reactivation_methods=[EQUALIZED],
            unit_groups=GROUPS,
            rng=seed,
        )

    first, again, other = table(5), table(5), table(6)
    assert first["value"].iloc[0] == again["value"].iloc[0]
    assert first["value"].iloc[0] != other["value"].iloc[0]


@pytest.mark.parametrize(
    ("base", "value", "message"),
    [
        (GIRI_2019, 0, "positive integer"),
        (GIRI_2019, 2.5, "positive integer"),
        (GIRI_2019, True, "positive integer"),
        (PEYRACHE_2010, 10, "only to EV methods"),
        (
            dataclasses.replace(KUDRIMOTI_1999, test="pair_bootstrap"),
            10,
            "pair bootstrap",
        ),
    ],
)
def test_recipe_refuses_rate_subsamples_it_cannot_apply(
    base, value, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(base, n_rate_subsamples=value)
