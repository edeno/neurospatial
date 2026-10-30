"""Equalizing each unit's firing rate before and after the task (Giri 2019)."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.reactivation import equalize_firing_rates

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
