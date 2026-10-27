"""Declared recording breaks (``break_before``) in encoding-to-decoding workflows.

A break before sample ``k`` must decode exactly like a pause longer than
``max_gap`` between samples ``k - 1`` and ``k``: the interval is dropped from
the encoding models, no decode bin spans it, and ground truth is never
interpolated across it.
"""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import (
    BayesianDecoder,
    DecodingResult,
    decode_session,
    decode_session_summary,
)

OFFSET = 1000.0
DT = 0.1


@pytest.fixture(scope="module")
def session():
    """Two concatenated 15 s recordings on one clock, and the paused twin."""
    rng = np.random.default_rng(21)
    n = 900
    times = np.cumsum(np.full(n, 1 / 30) + rng.uniform(-0.004, 0.004, n))
    phase = 2 * np.pi * times / 8.0
    positions = np.column_stack(
        [50 + 35 * np.sin(phase), 50 + 30 * np.cos(0.7 * phase)]
    )
    k = 450
    # The second recording starts elsewhere in the arena.
    positions[k:] = positions[k:][::-1]
    env = Environment.from_samples(positions, bin_size=5.0, units="cm")
    spike_rng = np.random.default_rng(22)
    trains = [np.sort(spike_rng.uniform(times[0], times[-1], 500)) for _ in range(6)]
    breaks = np.zeros(n, dtype=bool)
    breaks[k] = True
    shifted_times = np.where(np.arange(n) >= k, times + OFFSET, times)
    shifted_trains = [np.where(t >= times[k], t + OFFSET, t) for t in trains]
    return SimpleNamespace(
        env=env,
        times=times,
        positions=positions,
        trains=trains,
        k=k,
        breaks=breaks,
        shifted_times=shifted_times,
        shifted_trains=shifted_trains,
    )


def _unshift(s, decode_times):
    """Map decode times of the paused twin back onto the original clock."""
    return np.where(
        decode_times >= s.times[s.k] + OFFSET, decode_times - OFFSET, decode_times
    )


@pytest.mark.parametrize("summary", [False, True], ids=["full", "summary"])
def test_session_decode_equals_a_pause_longer_than_max_gap(session, summary):
    s = session
    decode = decode_session_summary if summary else decode_session
    declared = decode(
        s.env, s.trains, s.times, s.positions, dt=DT, break_before=s.breaks
    )
    paused = decode(s.env, s.shifted_trains, s.shifted_times, s.positions, dt=DT)
    np.testing.assert_allclose(declared.times, _unshift(s, paused.times))
    # No decode bin spans the break.
    left = declared.times - DT / 2
    assert not np.any((left < s.times[s.k]) & (left + DT > s.times[s.k - 1]))
    if summary:
        np.testing.assert_allclose(declared.map_position, paused.map_position)
        np.testing.assert_allclose(declared.mean_position, paused.mean_position)
    else:
        np.testing.assert_allclose(declared.posterior, paused.posterior)
        np.testing.assert_allclose(
            declared.error_against(s.times, s.positions, break_before=s.breaks),
            paused.error_against(s.shifted_times, s.positions),
            equal_nan=True,
        )


def test_decoder_methods_equal_a_pause_longer_than_max_gap(session):
    s = session
    declared = BayesianDecoder(s.env, dt=DT).fit(
        s.trains, s.times, s.positions, break_before=s.breaks
    )
    paused = BayesianDecoder(s.env, dt=DT).fit(
        s.shifted_trains, s.shifted_times, s.positions
    )
    np.testing.assert_allclose(declared.encoding_models, paused.encoding_models)
    np.testing.assert_array_equal(declared.spatial_support, paused.spatial_support)
    assert declared.analysis_settings["break_times"] == [s.times[s.k]]

    full = declared.predict(s.trains, s.times, break_before=s.breaks)
    np.testing.assert_allclose(
        full.posterior, paused.predict(s.shifted_trains, s.shifted_times).posterior
    )
    streamed = declared.predict_summary(s.trains, s.times, break_before=s.breaks)
    np.testing.assert_allclose(
        streamed.map_position,
        paused.predict_summary(s.shifted_trains, s.shifted_times).map_position,
    )
    assert declared.score(
        s.trains, s.times, s.positions, break_before=s.breaks
    ) == pytest.approx(paused.score(s.shifted_trains, s.shifted_times, s.positions))


def test_training_break_times_survive_from_rates(session):
    from neurospatial.encoding import compute_spatial_rates

    s = session
    rates = compute_spatial_rates(
        s.env, s.trains, s.times, s.positions, break_before=s.breaks
    )
    decoder = BayesianDecoder.from_rates(rates)
    assert decoder.analysis_settings["break_times"] == [s.times[s.k]]
    decoder.analysis_settings["break_times"].append(-1.0)
    assert decoder.analysis_settings["break_times"] == [s.times[s.k]]


def test_no_breaks_leaves_decoding_unchanged(session):
    s = session
    reference = decode_session(s.env, s.trains, s.times, s.positions, dt=DT)
    truth = reference.error_against(s.times, s.positions)
    for value in (None, np.zeros(len(s.times), dtype=bool)):
        result = decode_session(
            s.env, s.trains, s.times, s.positions, dt=DT, break_before=value
        )
        np.testing.assert_array_equal(result.posterior, reference.posterior)
        np.testing.assert_array_equal(result.times, reference.times)
        np.testing.assert_array_equal(
            result.error_against(s.times, s.positions, break_before=value), truth
        )


@pytest.fixture
def uniform_decode():
    env = Environment.from_samples(
        np.c_[np.arange(0.0, 41.0), np.zeros(41)], bin_size=2.0
    )
    posterior = np.full((9, env.n_bins), 1.0 / env.n_bins)
    return DecodingResult(posterior=posterior, env=env, times=np.arange(9) + 0.5)


def test_ground_truth_is_never_interpolated_across_a_break(uniform_decode):
    result = uniform_decode
    true_times = np.arange(10.0)
    true_positions = np.c_[np.arange(10.0) * 4, np.zeros(10)]
    breaks = np.zeros(10, dtype=bool)
    breaks[5] = True  # new recording at t = 5
    continuous = result.error_against(true_times, true_positions, max_gap=None)
    declared = result.error_against(
        true_times, true_positions, max_gap=None, break_before=breaks
    )
    # Only the decode time at 4.5 s lies between the two recordings.
    np.testing.assert_array_equal(np.isnan(declared), result.times == 4.5)
    keep = result.times != 4.5
    np.testing.assert_array_equal(declared[keep], continuous[keep])
    assert np.all(np.isfinite(continuous))

    axes = {}
    for label, value in (("continuous", None), ("declared", breaks)):
        axes[label] = result.plot(
            true_times=true_times,
            true_positions=true_positions,
            max_gap=None,
            break_before=value,
        )
    lines = {
        label: next(
            line for line in ax.get_lines() if line.get_label() == "Actual position"
        )
        for label, ax in axes.items()
    }
    assert np.all(np.isfinite(lines["continuous"].get_ydata()))
    assert np.isnan(lines["declared"].get_ydata()).any()
    for ax in axes.values():
        plt.close(ax.figure)


def test_plotted_truth_is_split_at_a_break_between_decode_samples(uniform_decode):
    """Decode times exactly on the samples either side of a break are not joined."""
    env = uniform_decode.env
    result = DecodingResult(
        posterior=np.full((10, env.n_bins), 1.0 / env.n_bins),
        env=env,
        times=np.arange(10.0),
    )
    true_times = np.arange(10.0)
    true_positions = np.c_[np.arange(10.0) * 4, np.zeros(10)]
    breaks = np.zeros(10, dtype=bool)
    breaks[5] = True
    ax = result.plot(
        true_times=true_times,
        true_positions=true_positions,
        max_gap=None,
        break_before=breaks,
    )
    line = next(
        line for line in ax.get_lines() if line.get_label() == "Actual position"
    )
    x = np.asarray(line.get_xdata(), dtype=float)
    y = np.asarray(line.get_ydata(), dtype=float)
    # Every sample is plotted, with one NaN separating the two recordings.
    np.testing.assert_array_equal(x[np.isfinite(y)], true_times)
    split = int(np.flatnonzero(np.isnan(y))[0])
    assert np.isnan(y).sum() == 1
    assert x[split - 1] == 4.0 and x[split + 1] == 5.0
    plt.close(ax.figure)


def _decoding_calls(s, bad):
    decoder = BayesianDecoder(s.env, dt=DT)
    fitted = decoder.fit(s.trains, s.times, s.positions)
    result = decode_session(s.env, s.trains, s.times, s.positions, dt=DT)
    return {
        "decode_session": lambda: decode_session(
            s.env, s.trains, s.times, s.positions, dt=DT, break_before=bad
        ),
        "decode_session_summary": lambda: decode_session_summary(
            s.env, s.trains, s.times, s.positions, dt=DT, break_before=bad
        ),
        "BayesianDecoder.fit": lambda: decoder.fit(
            s.trains, s.times, s.positions, break_before=bad
        ),
        "BayesianDecoder.predict": lambda: fitted.predict(
            s.trains, s.times, break_before=bad
        ),
        "BayesianDecoder.predict_summary": lambda: fitted.predict_summary(
            s.trains, s.times, break_before=bad
        ),
        "BayesianDecoder.score": lambda: fitted.score(
            s.trains, s.times, s.positions, break_before=bad
        ),
        "DecodingResult.error_against": lambda: result.error_against(
            s.times, s.positions, break_before=bad
        ),
        "DecodingResult.plot": lambda: result.plot(
            true_times=s.times, true_positions=s.positions, break_before=bad
        ),
    }


@pytest.mark.parametrize("problem", ["length", "dtype"])
def test_invalid_breaks_raise_naming_the_call(session, problem):
    s = session
    bad = s.breaks[:-1] if problem == "length" else s.breaks.astype(int)
    for name, call in _decoding_calls(s, bad).items():
        with pytest.raises(ValueError, match=rf"^{name}\(\) needs break_before"):
            call()
    plt.close("all")
