"""Decoding tiles observed runs without inventing posterior rows in pauses."""

import numpy as np
import pytest

from neurospatial.decoding import (
    BayesianDecoder,
    decode_session,
    decode_session_summary,
)
from neurospatial.encoding import compute_spatial_rates


@pytest.fixture
def gap_population(two_epoch_recording):
    r = two_epoch_recording
    return r, [r.spike_times + 0.04 * u for u in range(5)]


def test_decode_session_has_no_bins_in_pause(gap_population):
    r, trains = gap_population
    result = decode_session(
        r.env, trains, r.times, r.positions, method="binned", dt=0.025
    )
    assert result.times.size == 7998
    assert not np.any((result.times > 99.98) & (result.times < 1100))
    assert result.posterior.shape == (7998, r.env.n_bins)
    np.testing.assert_allclose(result.posterior.sum(axis=1), 1, rtol=1e-12, atol=0)


def test_summary_matches_full_decode(gap_population):
    r, trains = gap_population
    full = decode_session(r.env, trains, r.times, r.positions, method="binned")
    summary = decode_session_summary(
        r.env, trains, r.times, r.positions, method="binned", time_chunk=1000
    )
    assert len(summary.times) == 7998
    np.testing.assert_array_equal(summary.times, full.times)
    np.testing.assert_array_equal(summary.map_bin, full.posterior.argmax(axis=1))
    np.testing.assert_allclose(
        summary.mean_position, full.mean_position, rtol=1e-12, atol=0
    )


def test_spikes_in_pause_are_ignored(gap_population):
    r, trains = gap_population
    added = [np.sort(np.r_[s, np.linspace(200, 1000, 500)]) for s in trains]
    options = {"method": "binned", "warn_on_drop": False}
    a = decode_session(r.env, trains, r.times, r.positions, **options)
    b = decode_session(r.env, added, r.times, r.positions, **options)
    np.testing.assert_array_equal(a.posterior, b.posterior)
    a = compute_spatial_rates(r.env, trains, r.times, r.positions, **options)
    b = compute_spatial_rates(r.env, added, r.times, r.positions, **options)
    np.testing.assert_array_equal(a.firing_rates, b.firing_rates)


@pytest.mark.parametrize(
    "options", [{"epochs": [(0.0, 100.0)]}, {"spike_window": (1100.0, 1200.0)}]
)
def test_epochs_restrict_decode_bins(gap_population, options):
    r, trains = gap_population
    result = decode_session(
        r.env, trains, r.times, r.positions, method="binned", **options
    )
    assert len(result.times) == 3999
    if "epochs" in options:
        assert np.all(result.times < 100)
    else:
        assert np.all(result.times >= 1100)


def test_no_bin_fits_error(gap_population):
    r, trains = gap_population
    with pytest.raises(ValueError, match="No decode time bin fits") as caught:
        decode_session(r.env, trains, r.times, r.positions, method="binned", dt=200.0)
    assert any(line.startswith("Fix:") for line in str(caught.value).splitlines())


def test_immobility_still_decoded(gap_population):
    r, trains = gap_population
    speed = np.r_[np.full(5000, 20.0), np.zeros(5000)]
    result = decode_session(
        r.env,
        trains,
        r.times,
        r.positions,
        method="binned",
        speed=speed,
        min_speed=10.0,
    )
    assert np.count_nonzero(result.times >= 1100) == 3999


def test_decode_bounds_do_not_restrict_observed_runs(gap_population):
    r, trains = gap_population
    model = compute_spatial_rates(
        r.env, trains, r.times, r.positions, method="binned"
    ).firing_rates
    result = BayesianDecoder.from_models(r.env, model).predict(trains, r.times)
    assert len(result.times) == 7998


@pytest.fixture
def dropout_session():
    """30 Hz tracking with a 1.5 s dropout after every 0.7 s of observation."""
    from neurospatial import Environment

    run = np.arange(22) / 30.0  # 0.7 s observed
    times = np.concatenate([run + 2.2 * k for k in range(40)])
    x = 50.0 + 45.0 * np.sin(2 * np.pi * times / 20.0)
    positions = np.column_stack([x, np.zeros_like(x)])
    env = Environment.from_samples(positions, bin_size=10.0)
    rng = np.random.default_rng(3)
    trains = [np.sort(rng.uniform(times[0], times[-1], 300)) for _ in range(4)]
    return env, trains, times, positions


def test_decoding_warns_when_run_remainders_drop_much_observed_time(
    dropout_session,
):
    env, trains, times, positions = dropout_session
    with pytest.warns(UserWarning, match=r"29% is not decoded.*smaller dt") as record:
        decode_session(env, trains, times, positions, method="binned", dt=0.5)
    assert sum("not decoded" in str(w.message) for w in record) == 1
    decoder = BayesianDecoder(env, dt=0.5, method="binned").fit(
        trains, times, positions
    )
    with pytest.warns(UserWarning, match="29% is not decoded"):
        decoder.predict(trains, times)


def test_decoding_is_silent_when_bins_fit_the_runs(dropout_session, recwarn):
    env, trains, times, positions = dropout_session
    decode_session(env, trains, times, positions, method="binned", dt=0.1)
    assert not [w for w in recwarn.list if "not decoded" in str(w.message)]
    decode_session(
        env, trains, times, positions, method="binned", dt=0.5, warn_on_drop=False
    )
    assert not [w for w in recwarn.list if "not decoded" in str(w.message)]
