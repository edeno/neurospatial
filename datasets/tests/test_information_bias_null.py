from __future__ import annotations

import numpy as np
import pytest
from operations import _env
from studies.information_bias_null import bursty_trains, poisson_trains

from neurospatial.encoding import compute_spatial_rates


@pytest.fixture(scope="module")
def dense(simulated_recording):
    # 600 s in a 100 cm box: most 2.5 cm bins are visited for seconds.
    return simulated_recording, _env(simulated_recording, {})


def _info_and_bias(rec, env, trains):
    result = compute_spatial_rates(env, trains, rec.times, rec.positions)
    info = np.asarray(result.spatial_information(), dtype=float)
    bias = np.asarray(result.spatial_information_bias(), dtype=float)
    return info, bias


def test_poisson_ratio_near_one_on_dense_occupancy(dense):
    rec, env = dense
    info, bias = _info_and_bias(
        rec, env, poisson_trains(rec.times, 2.0, 200, np.random.default_rng(0))
    )
    assert 0.8 < info.mean() / bias.mean() < 1.25


def test_bursts_raise_information_above_poisson(dense):
    rec, env = dense
    poisson, _ = _info_and_bias(
        rec, env, poisson_trains(rec.times, 2.0, 100, np.random.default_rng(1))
    )
    bursty, _ = _info_and_bias(
        rec, env, bursty_trains(rec.times, 2.0, 100, np.random.default_rng(1))
    )
    assert bursty.mean() > 2 * poisson.mean()


def test_simulated_trains_have_the_requested_rate(dense):
    rec, _ = dense
    duration = rec.times[-1] - rec.times[0]
    rng = np.random.default_rng(2)
    for make in (poisson_trains, bursty_trains):
        trains = make(rec.times, 4.0, 50, rng)
        assert np.mean([len(t) for t in trains]) / duration == pytest.approx(
            4.0, rel=0.05
        )
