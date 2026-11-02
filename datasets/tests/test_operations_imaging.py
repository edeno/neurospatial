from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from _record import Recording
from operations import imaging_rates, imaging_rates_workaround

FIELD = np.array([30.0, 70.0])  # cm


@pytest.fixture(scope="module")
def imaging_recording() -> Recording:
    """A 100 cm box covered by a seeded random walk; one cell with a field at FIELD."""
    rng = np.random.default_rng(4)
    times = np.arange(30_000) / 30.0
    steps = rng.normal(0, 1.5, (times.size, 2))
    positions = np.abs(
        (np.cumsum(steps, axis=0) + 50.0) % 200.0 - 100.0
    )  # reflect into [0, 100]
    distance2 = np.sum((positions - FIELD) ** 2, axis=1)
    rate = 0.2 + 15.0 * np.exp(-distance2 / (2 * 8.0**2))
    return Recording(
        dataset="imaging_sim",
        session="0",
        source="simulated",
        license="n/a",
        times=times,
        positions=positions,
        position_units="cm",
        activity=rate[:, None],
        activity_times=times.copy(),
        activity_kind="binned_rate",
        params={"bin_size": 5.0},
    )


def test_binned_rate_workaround_recovers_field(imaging_recording):
    cache: dict = {}
    outcome = imaging_rates_workaround(imaging_recording, cache)
    assert outcome.status == "workaround"
    assert outcome.gap == "G-029"
    env = cache["env"]
    peak = env.bin_centers[np.nanargmax(cache["imaging_maps"][0])]
    assert np.all(np.abs(peak - FIELD) <= 5.0)  # within one bin


def test_event_threshold_rule_is_stated(imaging_recording):
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 0.05, imaging_recording.times.size)
    dff = replace(
        imaging_recording,
        activity=(imaging_recording.activity[:, 0] / 15.0 + noise)[:, None],
        activity_kind="dff",
    )
    outcome = imaging_rates_workaround(dff, {})
    assert outcome.status == "workaround"
    assert "median + 3 robust SD (1.4826 x MAD)" in outcome.note


def test_imaging_rates_is_missing(imaging_recording):
    outcome = imaging_rates(imaging_recording, {})
    assert outcome.status == "missing"
    assert outcome.gap == "G-029"
