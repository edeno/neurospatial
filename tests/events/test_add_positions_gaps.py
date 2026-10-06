"""Event positions use observed runs and their closed sample boundaries."""

import numpy as np
import pandas as pd
import pytest

from neurospatial.events import add_positions


@pytest.fixture(scope="module")
def pause_events(two_epoch_recording):
    r = two_epoch_recording
    return (
        r.times,
        r.positions,
        pd.DataFrame({"timestamp": [50.0, 600.0, 1150.0, 5000.0]}),
    )


def test_event_in_pause_gets_nan(pause_events):
    times, positions, events = pause_events
    result = add_positions(events, times=times, positions=positions)
    xy = result[["x", "y"]].to_numpy()
    assert np.isfinite(xy[[0, 2]]).all()
    assert np.isnan(xy[[1, 3]]).all()
    np.testing.assert_array_equal(xy[0], positions[2500])
    np.testing.assert_array_equal(xy[2], positions[7500])


def test_event_on_run_edges(pause_events):
    times, positions, _ = pause_events
    events = pd.DataFrame({"timestamp": [times[0], times[4999], 99.99, times[-1]]})
    result = add_positions(events, times=times, positions=positions)
    xy = result[["x", "y"]].to_numpy()
    np.testing.assert_array_equal(xy[[0, 1, 3]], positions[[0, 4999, -1]])
    assert np.isnan(xy[2]).all()


def test_event_on_isolated_sample_gets_nan():
    times = np.array([0.0, 0.1, 5.0, 10.0, 10.1])
    positions = np.column_stack([times, times * 2])
    events = pd.DataFrame({"timestamp": [0.05, 5.0, 10.05]})
    result = add_positions(events, times=times, positions=positions)
    assert np.isnan(result.loc[1, ["x", "y"]].to_numpy(dtype=float)).all()
    np.testing.assert_allclose(result.loc[[0, 2], "x"], [0.05, 10.05])


def test_event_with_no_observed_runs_gets_nan():
    result = add_positions(
        pd.DataFrame({"timestamp": [0.0, 0.5, 1.0, np.nan]}),
        times=np.array([0.0, 1.0]),
        positions=np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    assert np.isnan(result[["x", "y"]].to_numpy()).all()


def test_sorted_trajectory_and_epochs_keep_alignment(pause_events):
    times, positions, events = pause_events
    result = add_positions(
        events, times=times[::-1], positions=positions[::-1], epochs=(1100, 1200)
    )
    xy = result[["x", "y"]].to_numpy()
    assert np.isnan(xy[[0, 1, 3]]).all()
    np.testing.assert_array_equal(xy[2], positions[7500])


def test_max_gap_none_still_prevents_extrapolation():
    result = add_positions(
        pd.DataFrame({"timestamp": [-1.0, 0.5, 2.0]}),
        times=np.array([0.0, 1.0]),
        positions=np.array([[1.0], [3.0]]),
        max_gap=None,
    )
    assert result.loc[1, "x"] == 2
    assert np.isnan(result.loc[[0, 2], "x"]).all()
