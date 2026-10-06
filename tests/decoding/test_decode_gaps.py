"""Decoding tiles observed runs without inventing posterior rows in pauses."""

import numpy as np
import pytest

from neurospatial.decoding import decode_session, decode_session_summary
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
    a = compute_spatial_rates(
        r.env, trains, r.times, r.positions, fill_value=0.0, **options
    )
    b = compute_spatial_rates(
        r.env, added, r.times, r.positions, fill_value=0.0, **options
    )
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
        r.env, trains, r.times, r.positions, method="binned", fill_value=0.0
    ).firing_rates
    result = decode_session(
        r.env, trains, r.times, np.full_like(r.positions, 1e6), encoding_models=model
    )
    assert len(result.times) == 7998
