"""Population silence is measured only within continuously tracked periods."""

import warnings

import numpy as np
import pytest

from neurospatial.encoding import compute_spatial_rates


def test_population_spike_window_restores_rate_and_suppresses_silence_warning(
    continuous_recording,
):
    r = continuous_recording
    trains = [np.arange(100.1, 200, 0.2) + 0.04 * u for u in range(5)]
    with pytest.warns(
        UserWarning, match=r"All 5 units are silent from 0\.0 s to 100\.\d s"
    ):
        assumed = compute_spatial_rates(
            r.env, trains, r.times, r.positions, method="binned", bandwidth=0
        )
    pooled = (
        np.nansum(assumed.firing_rates * assumed.occupancy, axis=1)
        / assumed.occupancy.sum()
    )
    np.testing.assert_allclose(pooled, 2.5, rtol=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        explicit = compute_spatial_rates(
            r.env,
            trains,
            r.times,
            r.positions,
            method="binned",
            bandwidth=0,
            spike_window=(100, 200),
        )
    pooled = (
        np.nansum(explicit.firing_rates * explicit.occupancy, axis=1)
        / explicit.occupancy.sum()
    )
    np.testing.assert_allclose(pooled, 5.0, rtol=0.05)


@pytest.mark.parametrize(
    "n_units,silent_seconds,expected", [(5, 0, 0), (4, 100, 0), (5, 59, 0), (5, 61, 1)]
)
def test_population_silence_thresholds(
    continuous_recording, n_units, silent_seconds, expected
):
    r = continuous_recording
    trains = []
    for unit in range(n_units):
        spikes = r.spike_times + 0.04 * unit
        spikes = spikes[(spikes < 40) | (spikes >= 40 + silent_seconds)]
        trains.append(spikes)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_spatial_rates(r.env, trains, r.times, r.positions, method="binned")
    silence = [w for w in caught if str(w.message).startswith("All ")]
    assert len(silence) == expected


def test_population_silence_does_not_bridge_untracked_pause(two_epoch_recording):
    r = two_epoch_recording
    trains = []
    for unit in range(5):
        spikes = r.spike_times + 0.04 * unit
        trains.append(spikes[(spikes < 70) | (spikes >= 1131)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_spatial_rates(r.env, trains, r.times, r.positions, method="binned")
    assert not any(str(w.message).startswith("All ") for w in caught)


def test_population_trailing_silence_warns_at_caller(continuous_recording):
    r = continuous_recording
    trains = []
    for unit in range(5):
        spikes = np.arange(0.1, 100, 0.2) + 0.04 * unit
        trains.append(spikes[spikes < 100])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_spatial_rates(r.env, trains, r.times, r.positions, method="binned")
    silence = [w for w in caught if str(w.message).startswith("All ")]
    assert len(silence) == 1
    assert "to 200.0 s" in str(silence[0].message)
    assert silence[0].filename == __file__


@pytest.mark.parametrize("stop,expected", [(59.0, 0), (60.0, 1)])
def test_population_silence_respects_analysis_epochs(
    continuous_recording, stop, expected
):
    r = continuous_recording
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_spatial_rates(
            r.env,
            [np.array([])] * 5,
            r.times,
            r.positions,
            method="binned",
            epochs=(0, stop),
        )
    silence = [w for w in caught if str(w.message).startswith("All ")]
    assert len(silence) == expected


def test_population_silence_during_rest(continuous_recording):
    r = continuous_recording
    with pytest.warns(UserWarning, match="All 5 units are silent"):
        compute_spatial_rates(
            r.env,
            [np.array([])] * 5,
            r.times,
            r.positions,
            method="binned",
            speed=np.zeros_like(r.times),
            min_speed=1,
            warn_on_drop=False,
        )
