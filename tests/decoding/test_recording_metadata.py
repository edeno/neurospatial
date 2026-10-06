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


def test_evolve_rejects_posterior_and_keeps_fields(continuous_recording):
    env = continuous_recording.env
    r = DecodingResult(np.full((3, env.n_bins), 1 / env.n_bins), env, np.arange(3.0))
    with pytest.raises(ValueError, match="never replaces"):
        r._evolve(posterior=r.posterior)
    window = np.array([[0.0, 3.0]])
    evolved = r._evolve(spike_window=window)
    assert evolved.posterior is r.posterior
    assert evolved.env is r.env
    np.testing.assert_array_equal(evolved.times, r.times)
    assert not np.shares_memory(evolved.spike_window, window)
    assert not evolved.spike_window.flags.writeable
    window[0, 0] = 1
    np.testing.assert_array_equal(evolved.spike_window, [[0.0, 3.0]])
