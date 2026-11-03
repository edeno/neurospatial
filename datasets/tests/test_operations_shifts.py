"""The circular-shift null shifts spikes within the time the rate maps count."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from operations import MIN_SHIFT_S, N_SHIFTS, information_vs_shifts, shifted_trains

INTERVALS = np.array([[10.0, 40.0], [100.0, 130.0], [500.0, 560.0]])


def _concatenated(train, intervals):
    """Each spike's place on the concatenated counted time, NaN outside it."""
    out = np.full(train.shape, np.nan)
    offset = 0.0
    for start, stop in intervals:
        inside = (train >= start) & (train < stop)
        out[inside] = train[inside] - start + offset
        offset += stop - start
    return out


def test_shifted_spikes_stay_inside_the_counted_time():
    rng = np.random.default_rng(0)
    inside = np.sort(np.concatenate([rng.uniform(a, b, 50) for a, b in INTERVALS]))
    outside = np.array([0.5, 70.0, 300.0, 900.0])
    train = np.sort(np.concatenate([inside, outside]))
    (moved,) = shifted_trains([train], INTERVALS, seed=3)
    assert moved.size == train.size
    assert np.all(np.diff(moved) >= 0)
    counted = np.isfinite(_concatenated(moved, INTERVALS))
    assert counted.sum() == inside.size
    np.testing.assert_array_equal(np.sort(moved[~counted]), outside)


def test_one_shift_per_unit_on_the_concatenated_time():
    # Every spike of a unit moves by the same amount, modulo the counted
    # length, so its spike-to-spike intervals are kept except across a join.
    total = float(np.sum(np.diff(INTERVALS, axis=1)))
    rng = np.random.default_rng(1)
    train = np.sort(np.concatenate([rng.uniform(a, b, 40) for a, b in INTERVALS]))
    before = _concatenated(train, INTERVALS)
    for seed in range(5):
        (moved,) = shifted_trains([train], INTERVALS, seed=seed)
        after = np.sort(_concatenated(moved, INTERVALS))
        shifts = (after[:, None] - before[None, :]) % total
        # The one shift explains every spike: some offset matches all of them.
        candidates = shifts[0]
        explained = [
            np.isclose(((before + c) % total)[:, None], after[None, :], atol=1e-9)
            .any(axis=1)
            .all()
            for c in candidates
        ]
        assert any(explained)
        shift = candidates[int(np.argmax(explained))]
        assert MIN_SHIFT_S <= shift <= total - MIN_SHIFT_S


def test_shifts_are_seeded():
    train = np.linspace(11.0, 555.0, 30)
    a = shifted_trains([train, train], INTERVALS, seed=7)
    b = shifted_trains([train, train], INTERVALS, seed=7)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    c = shifted_trains([train], INTERVALS, seed=8)
    assert not np.array_equal(a[0], c[0])


def test_counted_time_too_short_for_the_shift_raises():
    with pytest.raises(ValueError, match="counted time"):
        shifted_trains([np.array([1.0])], np.array([[0.0, 30.0]]), seed=0)


@pytest.fixture(scope="module")
def untuned_in_short_epochs(simulated_recording):
    # Untuned Poisson units that fire only in the counted epochs, a fifth of
    # the recording (as spikes kept only during a bat's flights): shifted
    # over the whole span, most shifted spikes would land outside the
    # epochs, so the null would have fewer spikes and more bits per spike.
    rec = simulated_recording
    rng = np.random.default_rng(11)
    start = rec.times[0]
    epochs = np.array([[start + 60.0, start + 120.0], [start + 300.0, start + 360.0]])
    trains = tuple(
        np.sort(
            np.concatenate([rng.uniform(a, b, rng.poisson(120.0)) for a, b in epochs])
        )
        for _ in range(100)
    )
    params = {**rec.params, "analysis_epochs": epochs}
    return dataclasses.replace(
        rec,
        spike_times=trains,
        unit_ids=np.arange(len(trains)),
        params=params,
    )


def test_untuned_units_beat_every_shift_at_the_chance_rate(untuned_in_short_epochs):
    out = information_vs_shifts(untuned_in_short_epochs, {})
    summary = out.summary
    assert summary["n_units"] == 100
    # Exchangeable real and shifted maps: P(real beats all N) = 1 / (N + 1).
    assert summary["frac_above_every_shift"] == pytest.approx(
        1 / (N_SHIFTS + 1), abs=0.12
    )
    assert summary["median_shifted_bits_per_spike"] == pytest.approx(
        summary["median_bits_per_spike"], rel=0.25
    )
