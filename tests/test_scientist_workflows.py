"""Scientific meaning survives common user choices and operation handoffs."""

import numpy as np
import pandas as pd
import pytest

from neurospatial import Environment, compute_spatial_rates
from neurospatial.behavior import in_epochs, restrict
from neurospatial.behavior.decisions import compute_pre_decision_metrics
from neurospatial.behavior.trajectory import compute_step_lengths
from neurospatial.encoding import SpikeTrains


@pytest.fixture
def partial_recording():
    env = Environment.from_grid_mask(
        np.ones(10, dtype=bool), grid_edges=(np.arange(11.0) * 10,)
    )
    env.units = "cm"
    times = np.arange(101) / 10
    positions = np.full((len(times), 1), 5.0)
    spikes = [np.arange(0.05, 10.0, 0.1)]
    return env, spikes, times, positions


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
        times, np.c_[times, np.zeros(len(times))], entry, 1.0
    )
    assert not result.is_valid
    assert not result.suggests_deliberation()
    assert np.isnan(result.mean_speed)
    assert np.isnan(result.heading_circular_variance)


def test_stationary_tracking_is_not_head_scanning():
    times = np.arange(21) / 10
    result = compute_pre_decision_metrics(times, np.ones((21, 2)), 2.0, 1.0)
    assert result.mean_speed == 0.0
    assert not result.is_valid
    assert not result.suggests_deliberation()


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
    np.testing.assert_allclose(full, [0.7])
    np.testing.assert_allclose(kept, full)
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
