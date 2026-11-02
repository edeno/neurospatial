from __future__ import annotations

from dataclasses import replace

import numpy as np
from operations import _transient_onsets, imaging_rates, imaging_rates_workaround

from conftest import IMAGING_FIELD


def _peak(cache: dict) -> np.ndarray:
    return cache["env"].bin_centers[np.nanargmax(cache["imaging_maps"][0])]


def test_binned_rate_workaround_recovers_field(imaging_recording):
    cache: dict = {}
    outcome = imaging_rates_workaround(imaging_recording, cache)
    assert outcome.status == "workaround"
    assert outcome.gap == "G-029"
    assert np.all(np.abs(_peak(cache) - IMAGING_FIELD) <= 5.0)  # within one bin


def test_binned_rate_workaround_on_another_clock_with_gaps(imaging_recording):
    # Frames half a sample after tracking, and a stretch of lost tracking.
    positions = imaging_recording.positions.copy()
    positions[5000:6000] = np.nan
    rec = replace(
        imaging_recording,
        positions=positions,
        activity_times=imaging_recording.times + 1 / 60,
    )
    cache: dict = {}
    imaging_rates_workaround(rec, cache)
    assert np.all(np.abs(_peak(cache) - IMAGING_FIELD) <= 5.0)


def _map_vs_truth(rec, dff) -> tuple[float, dict]:
    cache: dict = {}
    outcome = imaging_rates_workaround(
        replace(rec, activity=dff[:, None], activity_kind="dff"), cache
    )
    env, rates = cache["env"], cache["imaging_maps"][0]
    distance2 = np.sum((env.bin_centers - IMAGING_FIELD) ** 2, axis=1)
    truth = np.exp(-distance2 / (2 * 8.0**2))
    finite = np.isfinite(rates)
    return float(np.corrcoef(rates[finite], truth[finite])[0, 1]), {
        "outcome": outcome,
        **cache,
    }


def test_dff_events_map_recovers_field(imaging_recording):
    rng = np.random.default_rng(0)
    # Transients at Poisson events, up to about 2 events/s in the field with a
    # 0.5 s decay. At much higher rates transients overlap, the trace never
    # falls back, and an onset rule counts a whole field visit as one event.
    rate = imaging_recording.activity[:, 0] * (2.0 / 15.2)
    events = rng.random(rate.size) < rate / 30.0
    kernel = np.exp(-np.arange(45) / 15.0)
    dff = np.convolve(events.astype(float), kernel)[: rate.size] + rng.normal(
        0, 0.05, rate.size
    )
    corr, cache = _map_vs_truth(imaging_recording, dff)
    assert "onset above median + 3 robust SD (1.4826 x MAD)" in cache["outcome"].note
    assert corr > 0.5
    assert 0 < len(cache["imaging_events"][0]) <= events.sum()
    # Control: the same trace shifted in time keeps no relation to position.
    shifted_corr, _ = _map_vs_truth(imaging_recording, np.roll(dff, 9000))
    assert abs(shifted_corr) < 0.2


def test_noise_recrossing_counts_once():
    # Baseline +-1 (median 0, robust SD 1.48): onset above 4.4, end below 1.5.
    # The transient dips under the onset level twice without ending.
    times = np.arange(12, dtype=float)
    trace = np.array([1, -1, 1, -1, 10, 3, 5, 3, 5, -1, 1, -1], dtype=float)
    assert _transient_onsets(trace, times).tolist() == [4.0]


def test_imaging_rates_is_missing(imaging_recording):
    outcome = imaging_rates(imaging_recording, {})
    assert outcome.status == "missing"
    assert outcome.gap == "G-029"
