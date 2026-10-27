"""Scientific meaning survives common user choices and operation handoffs."""

import json

import numpy as np
import pandas as pd
import pytest

from neurospatial import Environment, compute_spatial_rate, compute_spatial_rates
from neurospatial.behavior import in_epochs, restrict
from neurospatial.behavior.decisions import compute_pre_decision_metrics
from neurospatial.behavior.trajectory import compute_step_lengths
from neurospatial.decoding import (
    BayesianDecoder,
    decode_position,
    decode_position_summary,
    decode_session,
    decode_session_summary,
)
from neurospatial.encoding import SpikeTrains


@pytest.fixture
def partial_recording():
    env = Environment.from_grid_mask(
        np.ones(10, dtype=bool), grid_edges=(np.arange(11.0) * 10,)
    )
    env = env.with_metadata(units="cm")
    times = np.arange(101) / 10
    positions = np.full((len(times), 1), 5.0)
    spikes = [np.arange(0.05, 10.0, 0.1)]
    return env, spikes, times, positions


def test_unsupported_space_never_becomes_evidence_of_silence(partial_recording):
    env, spikes, times, positions = partial_recording
    rates = compute_spatial_rates(env, spikes, times, positions, min_occupancy=0.1)
    expected_support = np.arange(env.n_bins) == 0
    np.testing.assert_array_equal(rates.spatial_support, expected_support)
    silent = [np.array([])]
    fitted = BayesianDecoder(env, dt=0.1, min_occupancy=0.1).fit(
        spikes, times, positions
    )
    for decoder in (fitted, BayesianDecoder.from_rates(rates, dt=0.1)):
        result = decoder.predict(silent, times)
        np.testing.assert_array_equal(result.posterior[:, 0], 1.0)
        np.testing.assert_array_equal(result.posterior[:, 1:], 0.0)
        summary = decoder.predict_summary(silent, times, time_chunk=7)
        np.testing.assert_array_equal(summary.map_position, result.map_position)
        np.testing.assert_allclose(summary.posterior_entropy, 0.0, atol=1e-12)
    for chunk in (None, 7):
        result = decode_position(
            env, np.zeros((10, 1)), rates, dt=0.1, time_chunk=chunk
        )
        np.testing.assert_array_equal(result.posterior[:, 1:], 0.0)
    summary = decode_position_summary(env, np.zeros((10, 1)), rates, dt=0.1)
    np.testing.assert_array_equal(summary.map_position, 5.0)


def test_session_support_intersects_static_and_time_varying_priors(partial_recording):
    env, spikes, times, positions = partial_recording
    result = decode_session(env, spikes, times, positions, dt=0.1, min_occupancy=0.1)
    for prior in (np.ones(env.n_bins), np.ones_like(result.posterior)):
        decoded = decode_session(
            env, spikes, times, positions, dt=0.1, min_occupancy=0.1, prior=prior
        )
        summary = decode_session_summary(
            env,
            spikes,
            times,
            positions,
            dt=0.1,
            min_occupancy=0.1,
            prior=prior,
            time_chunk=7,
        )
        np.testing.assert_array_equal(decoded.posterior[:, 1:], 0.0)
        np.testing.assert_array_equal(summary.map_position, decoded.map_position)
    with pytest.raises(ValueError, match="no probability to supported"):
        decode_session(
            env,
            spikes,
            times,
            positions,
            dt=0.1,
            min_occupancy=0.1,
            prior=np.r_[0.0, np.ones(env.n_bins - 1)],
        )


def test_low_occupancy_bins_stay_missing_through_decoding(partial_recording):
    env, spikes, times, positions = partial_recording
    positions = positions.copy()
    positions[-2:] = 15.0
    rates = compute_spatial_rates(env, spikes, times, positions, min_occupancy=0.5)
    assert 0 < rates.occupancy[1] < 0.5
    assert np.isnan(rates.firing_rates[0, 1])
    assert not rates.spatial_support[1]
    decoded = decode_position(env, np.zeros((3, 1)), rates, dt=0.1)
    np.testing.assert_array_equal(decoded.posterior[:, 1:], 0.0)
    unsupported = compute_spatial_rates(
        env, spikes, times, positions, min_occupancy=100
    )
    with pytest.raises(ValueError, match="supported training data"):
        BayesianDecoder.from_rates(unsupported)


def test_editing_settings_cannot_change_supported_training_bins(partial_recording):
    env, spikes, times, positions = partial_recording
    positions = positions.copy()
    positions[-2:] = 15.0
    rates = compute_spatial_rates(env, spikes, times, positions, min_occupancy=0.5)
    before = decode_position(env, np.zeros((1, 1)), rates, dt=0.1)
    settings = rates.analysis_settings
    settings["min_occupancy"] = 0.0
    settings["observed_intervals"][0][0] = -10.0
    after = decode_position(env, np.zeros((1, 1)), rates, dt=0.1)

    assert rates.analysis_settings["min_occupancy"] == 0.5
    assert not rates.spatial_support[1]
    np.testing.assert_array_equal(after.posterior, before.posterior)
    np.testing.assert_array_equal(after.posterior[:, 1:], 0.0)


@pytest.mark.parametrize("missing", [100.0, np.nan])
def test_geodesic_steps_preserve_missing_positions(missing):
    env = Environment.from_grid_mask(
        np.ones(4, dtype=bool), grid_edges=(np.arange(5.0),)
    )
    positions = np.array([[0.5], [1.5], [missing], [missing], [2.5]])
    steps = compute_step_lengths(positions, metric="geodesic", env=env)
    assert steps[0] == 1.0
    assert np.isnan(steps[1:]).all()


@pytest.mark.parametrize("entry", [0.0, 5.0, 10.0, 10.1])
def test_absent_pre_decision_evidence_cannot_label_deliberation(entry):
    times = np.array([0.0, 0.1, 10.0, 10.1])
    result = compute_pre_decision_metrics(
        times,
        np.c_[times, np.zeros(len(times))],
        entry,
        window_duration=1.0,
        min_speed=5.0,
    )
    assert not result.is_valid
    assert not result.suggests_deliberation()
    assert np.isnan(result.mean_speed)
    assert np.isnan(result.heading_circular_variance)


def test_stationary_tracking_is_not_head_scanning():
    times = np.arange(21) / 10
    result = compute_pre_decision_metrics(
        times, np.ones((21, 2)), 2.0, window_duration=1.0, min_speed=5.0
    )
    assert result.mean_speed == 0.0
    assert not result.is_valid
    assert not result.suggests_deliberation()


@pytest.mark.parametrize("method", ["binned", "diffusion_kde", "gaussian_kde"])
def test_default_smoothing_is_invariant_to_physical_units(method):
    bins = np.tile(np.arange(20), 20)
    times = np.arange(len(bins)) / 10
    positions = (bins * 5.0 + 2.5)[:, None]
    spikes = times[bins == 5] + 0.01
    reference = None
    for scale, units in [(1.0, "cm"), (0.01, "m"), (10.0, "mm")]:
        env = Environment.from_grid_mask(
            np.ones(20, dtype=bool), grid_edges=(np.arange(21.0) * 5 * scale,)
        )
        env = env.with_metadata(units=units)
        result = compute_spatial_rate(
            env, spikes, times, positions * scale, method=method
        )
        assert result.bandwidth == pytest.approx(5.0 * scale)
        assert result.spatial_information() > 1.0
        if reference is None:
            reference = result.firing_rate
        else:
            np.testing.assert_allclose(result.firing_rate, reference, atol=1e-10)
    with pytest.warns(UserWarning, match="erase spatial tuning"):
        compute_spatial_rate(env, spikes, times, positions * scale, bandwidth=1000)


def test_epoch_workflow_preserves_excluded_time():
    times = np.arange(11) / 10
    positions = np.full((len(times), 1), 0.5)
    env = Environment.from_grid_mask(
        np.ones(1, dtype=bool), grid_edges=(np.array([0.0, 1.0]),)
    )
    epochs = np.array([[0.0, 0.3], [0.4, 0.8]])
    with pytest.warns(UserWarning, match="pass the original"):
        kept_times, kept_positions = restrict(times, positions, epochs=epochs)
    full = env.occupancy(times, positions, epochs=epochs)
    kept = env.occupancy(kept_times, kept_positions, epochs=epochs)
    # The original arrays with epochs= are exact: [0, 0.3) and [0.4, 0.8).
    np.testing.assert_allclose(full, [0.7])
    # Slicing drops the samples on each stop (0.3 s, 0.8 s), so each window's
    # last 0.1 s interval is lost, which is why restrict() warns.
    np.testing.assert_allclose(kept, [0.5])
    train = in_epochs(times, (0.0, 0.3), closed="left")
    test = in_epochs(times, (0.3, 0.8), closed="left")
    assert not np.any(train & test)


@pytest.mark.parametrize("identity", ["index", "column"])
def test_metadata_filtering_and_encoding_preserve_identity(partial_recording, identity):
    env, spikes, times, positions = partial_recording
    table = pd.DataFrame(
        {"region": ["CA3", "CA1"]}, index=pd.Index([20, 10], name="unit_id")
    )
    if identity == "column":
        table = table.reset_index()
    original = table.copy()
    trains = SpikeTrains(
        [spikes[0], spikes[0][::2]], unit_ids=[10, 20], unit_table=table
    )
    assert trains.filter("region == 'CA1'").unit_ids.tolist() == [10]
    if identity == "index":
        assert trains.filter("region == 'CA1'").unit_table.index.tolist() == [10]
    rates = compute_spatial_rates(env, trains, times, positions)
    assert rates.unit_table.iloc[0]["region"] == "CA1"
    pd.testing.assert_frame_equal(table, original)
    with pytest.raises(ValueError, match="labels do not match"):
        SpikeTrains([spikes[0], spikes[0]], unit_ids=[1, 2], unit_table=table)


def test_analysis_settings_survive_indexing_and_exports(partial_recording, tmp_path):
    env, spikes, times, positions = partial_recording
    result = compute_spatial_rates(
        env,
        spikes,
        times,
        positions,
        epochs=(0, 5),
        spike_window=(0, 10),
        max_gap=0.3,
        min_occupancy=0.2,
        bandwidth=2.0,
        speed=np.full(len(times), 10.0),
        min_speed=5.0,
    )
    settings = result.analysis_settings
    assert settings["epochs"] == [[0, 5]]
    assert settings["observed_intervals"] == [[0, 5]]
    assert settings["max_gap"] == 0.3
    assert settings["speed_source"] == "provided"
    assert settings["min_speed"] == 5.0
    assert settings["min_occupancy"] == 0.2
    assert result[0].analysis_settings == settings
    assert result.summary_table().attrs["analysis_settings"] == settings
    frame = result.to_dataframe()
    assert frame.attrs["analysis_settings"] == settings
    assert frame.attrs["units"]["bin_center_x"] == "cm"
    frame.attrs["analysis_settings"]["max_gap"] = 99
    assert result.analysis_settings["max_gap"] == 0.3
    xr = pytest.importorskip("xarray")
    dataset = result.to_xarray()
    assert json.loads(dataset.attrs["analysis_settings"]) == settings
    path = tmp_path / "rates.nc"
    dataset.to_netcdf(path)
    with xr.open_dataset(path) as restored:
        assert json.loads(restored.attrs["analysis_settings"]) == settings
        np.testing.assert_array_equal(restored.spatial_support, result.spatial_support)
