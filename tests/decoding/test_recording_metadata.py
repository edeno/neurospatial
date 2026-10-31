"""Decoder acquisition metadata preserves frozen-array ownership."""

import dataclasses

import numpy as np
import pytest

from neurospatial.decoding import DecodingResult, decode_session, decode_session_summary


@pytest.mark.parametrize("summary", [False, True])
@pytest.mark.parametrize("window", [None, (100.0, 200.0)])
def test_results_record_spike_window(continuous_recording, summary, window):
    r = continuous_recording
    function = decode_session_summary if summary else decode_session
    result = function(
        r.env,
        [r.spike_times],
        r.times,
        r.positions,
        method="binned",
        spike_window=window,
    )
    assert result.spike_window_assumed is (window is None)
    assert result.summary()["spike_window_assumed"] is (window is None)
    assert result.summary()["spike_window"] == (
        None if window is None else [[100.0, 200.0]]
    )
    if window is not None:
        np.testing.assert_array_equal(result.spike_window, [[100.0, 200.0]])
        assert not result.spike_window.flags.writeable


def test_decode_session_allocates_one_posterior(continuous_recording, monkeypatch):
    r = continuous_recording
    original = DecodingResult._from_owned_posterior
    seen = []

    def track(cls, posterior, **fields):
        seen.append(posterior)
        return original(posterior, **fields)

    monkeypatch.setattr(DecodingResult, "_from_owned_posterior", classmethod(track))
    result = decode_session(
        r.env,
        [r.spike_times],
        r.times,
        r.positions,
        method="binned",
        spike_window=(100, 200),
    )
    assert result.posterior is seen[0]
    assert all(np.shares_memory(result.posterior, array) for array in seen)
    changed = dataclasses.replace(result, spike_window=None)
    assert not np.shares_memory(changed.posterior, result.posterior)


@pytest.mark.parametrize("summary", [False, True])
def test_session_decode_records_the_decoded_epochs(continuous_recording, summary):
    """The gap between decode epochs is not decoded time."""
    r = continuous_recording
    function = decode_session_summary if summary else decode_session
    result = function(
        r.env,
        [r.spike_times],
        r.times,
        r.positions,
        method="binned",
        dt=0.25,
        epochs=[[0.0, 50.0], [70.0, 120.0]],
    )
    # 0.25 s tiles each 50 s epoch exactly, so the decoded bins cover the
    # epochs and nothing between them.
    np.testing.assert_array_equal(result.time_support, [[0.0, 50.0], [70.0, 120.0]])
    assert not result.time_support.flags.writeable
    assert result.summary()["decoded_duration"] == 100.0
    # Each bin [t - dt/2, t + dt/2) lies inside one decoded epoch.
    left, right = result.times - 0.125, result.times + 0.125
    row = np.searchsorted(result.time_support[:, 0], left, side="right") - 1
    assert np.all(row >= 0)
    assert np.all(right <= result.time_support[row, 1] + 1e-12)
    assert np.count_nonzero(row == 0) == np.count_nonzero(row == 1) == 200


def test_session_decode_time_support_excludes_a_dropped_remainder(
    continuous_recording,
):
    r = continuous_recording
    result = decode_session(
        r.env,
        [r.spike_times],
        r.times,
        r.positions,
        method="binned",
        dt=0.3,
        epochs=(0.0, 10.0),
    )
    # 33 whole 0.3 s bins fit in 10 s; the last 0.1 s is not decoded.
    np.testing.assert_allclose(result.time_support, [[0.0, 9.9]], rtol=0, atol=1e-12)
    assert result.n_time_bins == 33


@pytest.mark.parametrize("summary", [False, True])
def test_array_decode_records_bins_around_times(summary):
    from neurospatial import Environment
    from neurospatial.decoding import (
        bin_spikes_in_time,
        decode_jointly,
        decode_position,
        decode_position_summary,
    )

    rng = np.random.default_rng(1)
    positions = rng.uniform(0.0, 50.0, (500, 2))
    env = Environment.from_samples(positions, bin_size=10.0)
    spikes = [np.sort(rng.uniform(0.0, 120.0, 300)) for _ in range(3)]
    windows = [[0.0, 50.0], [70.0, 120.0]]
    counts, centers = bin_spikes_in_time(spikes, dt=0.25, epochs=windows)
    rates = rng.uniform(1.0, 5.0, (3, env.n_bins))
    function = decode_position_summary if summary else decode_position
    result = function(env, counts, rates, dt=0.25, times=centers)

    np.testing.assert_allclose(result.time_support, windows, rtol=0, atol=1e-12)
    if not summary:
        joint = decode_jointly(
            env, counts, {"a": rates, "b": rates}, dt=0.25, times=centers
        )
        for conditional in joint.conditional.values():
            np.testing.assert_array_equal(conditional.time_support, result.time_support)
    without_times = function(env, counts, rates, dt=0.25)
    assert without_times.time_support is None
    assert np.isnan(without_times.summary()["decoded_duration"])


def test_decoding_result_time_support_must_hold_every_time():
    from neurospatial import Environment

    env = Environment.from_samples(np.array([[0.0, 0.0], [10.0, 10.0]]), bin_size=5.0)
    posterior = np.full((3, env.n_bins), 1.0 / env.n_bins)
    times = np.array([0.5, 1.5, 2.5])
    with pytest.raises(ValueError, match=r"1 of 3 times lie outside time_support"):
        DecodingResult(posterior, env, times, time_support=[[0.0, 2.0]])
    with pytest.raises(ValueError, match="time_support was given without times"):
        DecodingResult(posterior, env, time_support=[[0.0, 3.0]])

    result = DecodingResult(posterior, env, times, time_support=[[2.0, 3.0], [0, 2]])
    # Rows are sorted and touching rows merge, as for every epochs= argument.
    np.testing.assert_array_equal(result.time_support, [[0.0, 3.0]])


def test_time_support_survives_pickle_and_xarray(continuous_recording):
    import pickle

    r = continuous_recording
    result = decode_session(
        r.env,
        [r.spike_times],
        r.times,
        r.positions,
        method="binned",
        dt=0.25,
        epochs=[[0.0, 50.0], [70.0, 120.0]],
    )
    restored = pickle.loads(pickle.dumps(result))
    np.testing.assert_array_equal(restored.time_support, result.time_support)
    assert not restored.time_support.flags.writeable
    xr = pytest.importorskip("xarray")
    assert xr is not None
    np.testing.assert_array_equal(
        result.to_xarray().attrs["time_support"], [0.0, 50.0, 70.0, 120.0]
    )
