"""Spatial spikes and occupancy use matching analysis and acquisition windows."""

from dataclasses import replace

import numpy as np
import pytest

from neurospatial.encoding import compute_spatial_rate, compute_spatial_rates


@pytest.fixture
def direct_rate_result_factory(continuous_recording):
    from neurospatial import Environment
    from neurospatial.encoding.directional import (
        DirectionalRateResult,
        DirectionalRatesResult,
    )
    from neurospatial.encoding.egocentric import (
        EgocentricRateResult,
        EgocentricRatesResult,
    )
    from neurospatial.encoding.spatial import SpatialRateResult, SpatialRatesResult
    from neurospatial.encoding.view import ViewRateResult, ViewRatesResult

    classes = {
        "spatial": (SpatialRateResult, SpatialRatesResult),
        "directional": (DirectionalRateResult, DirectionalRatesResult),
        "view": (ViewRateResult, ViewRatesResult),
        "egocentric": (EgocentricRateResult, EgocentricRatesResult),
    }

    def make_result(family, plural):
        env = continuous_recording.env
        kwargs = {}
        if family == "egocentric":
            env = Environment.from_polar_egocentric(
                distance_range=(0, 50),
                angle_range=(-np.pi, np.pi),
                distance_bin_size=25.0,
                angle_bin_size=np.pi / 2,
            )
            kwargs.update(
                env=env, distance_range=(0, 50), n_distance_bins=2, n_direction_bins=4
            )
        elif family == "directional":
            kwargs.update(
                bin_centers=np.linspace(-np.pi, np.pi, env.n_bins, endpoint=False),
                bin_size=2 * np.pi / env.n_bins,
                bandwidth=0.3,
            )
        else:
            kwargs.update(env=env, method="binned", bandwidth=5.0)
            if family == "view":
                kwargs.update(gaze_model="fixed_distance", view_distance=10.0)
        kwargs["firing_rates" if plural else "firing_rate"] = np.ones(
            (2, env.n_bins) if plural else env.n_bins
        )
        kwargs["occupancy"] = np.ones(env.n_bins)
        return classes[family][int(plural)](**kwargs)

    return make_result


@pytest.mark.parametrize("family", ["spatial", "directional", "view", "egocentric"])
@pytest.mark.parametrize("plural", [False, True])
def test_direct_rate_results_make_assumption_visible(
    direct_rate_result_factory, family, plural
):
    result = direct_rate_result_factory(family, plural)
    assert result.spike_window is None
    assert result.spike_window_assumed is True
    assert result.summary()["spike_window"] is None
    assert result.summary()["spike_window_assumed"] is True
    explicit = replace(result, spike_window=np.array([[0.0, 100.0]]))
    assert explicit.spike_window_assumed is False
    assert explicit.summary()["spike_window"] == [[0.0, 100.0]]


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


@pytest.mark.parametrize("method", ["binned", "glm"])
@pytest.mark.parametrize("plural", [False, True])
@pytest.mark.parametrize("window", [None, (100.0, 200.0)])
def test_spatial_results_record_spike_window(
    continuous_recording, method, plural, window
):
    r = continuous_recording
    function = compute_spatial_rates if plural else compute_spatial_rate
    result = function(
        r.env,
        [r.spike_times] if plural else r.spike_times,
        r.times,
        r.positions,
        method=method,
        spike_window=window,
    )
    assert result.spike_window_assumed is (window is None)
    assert result.summary()["spike_window_assumed"] is (window is None)
    if window is None:
        assert result.spike_window is None
        assert result.summary()["spike_window"] is None
    else:
        np.testing.assert_array_equal(result.spike_window, [[100.0, 200.0]])
        assert result.summary()["spike_window"] == [[100.0, 200.0]]
    if plural:
        child = result[0]
        assert child.spike_window_assumed is (window is None)
        if window is not None:
            np.testing.assert_array_equal(child.spike_window, result.spike_window)


@pytest.mark.parametrize("method", ["binned", "glm"])
def test_empty_population_records_spike_window(continuous_recording, method):
    r = continuous_recording
    result = compute_spatial_rates(
        r.env, [], r.times, r.positions, method=method, spike_window=(100, 200)
    )
    np.testing.assert_array_equal(result.spike_window, [[100, 200]])
    assert result.summary()["spike_window_assumed"] is False
