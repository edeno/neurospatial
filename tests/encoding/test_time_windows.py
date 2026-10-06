"""Spatial spikes and occupancy use matching analysis and acquisition windows."""

import numpy as np
import pytest

from neurospatial.encoding import compute_spatial_rate, compute_spatial_rates


@pytest.mark.parametrize("method", ["binned", "diffusion_kde"])
@pytest.mark.parametrize("plural", [False, True])
def test_spatial_epochs_equal_slicing(continuous_recording, method, plural):
    r = continuous_recording
    function = compute_spatial_rates if plural else compute_spatial_rate
    spikes = [r.spike_times, r.spike_times + 0.04] if plural else r.spike_times
    sliced_spikes = [s[s < 100] for s in spikes] if plural else spikes[spikes < 100]
    actual = function(
        r.env, spikes, r.times, r.positions, method=method, epochs=[(0.0, 100.0)]
    )
    keep = r.times <= 100
    expected = function(
        r.env, sliced_spikes, r.times[keep], r.positions[keep], method=method
    )
    rates = "firing_rates" if plural else "firing_rate"
    np.testing.assert_allclose(
        getattr(actual, rates),
        getattr(expected, rates),
        rtol=1e-12,
        atol=0,
        equal_nan=True,
    )
    np.testing.assert_allclose(actual.occupancy, expected.occupancy, rtol=1e-12, atol=0)


@pytest.mark.parametrize("plural", [False, True])
def test_spatial_spike_window_restores_true_rate(continuous_recording, plural):
    r = continuous_recording
    function = compute_spatial_rates if plural else compute_spatial_rate
    spikes = np.arange(100.1, 200, 0.2)
    actual = function(
        r.env,
        [spikes] if plural else spikes,
        r.times,
        r.positions,
        method="binned",
        spike_window=(100, 200),
    )
    rates = np.asarray(actual.firing_rates if plural else actual.firing_rate)
    assert np.nansum(
        rates * actual.occupancy
    ) / actual.occupancy.sum() == pytest.approx(5.0, rel=0.05)
    assert actual.occupancy.sum() == pytest.approx(99.98, abs=1e-9)


def test_epochs_and_spike_window_errors(continuous_recording):
    r = continuous_recording
    with pytest.raises(ValueError) as exc:
        compute_spatial_rate(
            r.env,
            r.spike_times,
            r.times,
            r.positions,
            epochs=[[5, 1]],
            spike_window="bad",
        )
    assert "epochs" in str(exc.value)
    assert "spike_window" in str(exc.value)
    assert "\nFix:" in str(exc.value)


def test_all_excluded_warning_names_epochs(continuous_recording):
    r = continuous_recording
    with pytest.warns(UserWarning, match="ALL trajectory intervals.*epochs"):
        result = compute_spatial_rate(
            r.env,
            r.spike_times,
            r.times,
            r.positions,
            method="binned",
            epochs=[(5000, 6000)],
        )
    assert result.occupancy.sum() == 0
    assert np.isnan(result.firing_rate).all()


@pytest.mark.parametrize("method", ["binned", "glm"])
@pytest.mark.parametrize("n_units", [0, 2])
@pytest.mark.parametrize("warn_on_drop", [False, True])
def test_spatial_analysis_mask_is_shared_once(
    continuous_recording, monkeypatch, method, n_units, warn_on_drop
):
    import neurospatial.environment.trajectory as trajectory

    r = continuous_recording
    original = trajectory.interval_valid_mask
    calls = []

    def track_mask(*args, **kwargs):
        mask = original(*args, **kwargs)
        calls.append(mask)
        return mask

    monkeypatch.setattr(trajectory, "interval_valid_mask", track_mask)
    result = compute_spatial_rates(
        r.env,
        [r.spike_times] * n_units,
        r.times,
        r.positions,
        method=method,
        spike_window=(0, 100),
        warn_on_drop=warn_on_drop,
    )
    assert len(calls) == 1
    assert calls[0].sum() == 5000
    assert result.occupancy.sum() == pytest.approx(100.0, abs=1e-9)
    assert result.firing_rates.shape == (n_units, r.env.n_bins)


def test_single_spatial_mask_is_shared_once(continuous_recording, monkeypatch):
    import neurospatial.environment.trajectory as trajectory

    r = continuous_recording
    original = trajectory.interval_valid_mask
    calls = []

    def track_mask(*args, **kwargs):
        mask = original(*args, **kwargs)
        calls.append(mask)
        return mask

    monkeypatch.setattr(trajectory, "interval_valid_mask", track_mask)
    result = compute_spatial_rate(
        r.env,
        r.spike_times,
        r.times,
        r.positions,
        method="binned",
        epochs=(0, 100),
        warn_on_drop=False,
    )
    assert len(calls) == 1
    assert result.occupancy.sum() == pytest.approx(100.0, abs=1e-9)
