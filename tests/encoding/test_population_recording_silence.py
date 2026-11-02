"""Population silence is measured only within continuously tracked periods."""

import warnings

import numpy as np
import pytest

import neurospatial.encoding._spatial_counts as spatial_counts
from neurospatial.encoding import compute_spatial_rates
from neurospatial.encoding._spatial_counts import (
    _SILENCE_MIN_SECONDS,
    _has_empty_observed_bin,
    _warn_if_population_silent,
)


def test_population_spike_window_restores_rate_and_suppresses_silence_warning(
    continuous_recording,
):
    r = continuous_recording
    trains = [np.arange(100.1, 200, 0.2) + 0.04 * u for u in range(5)]
    with pytest.warns(
        UserWarning, match=r"All 5 units are silent from 0\.0 s to 100\.\d s"
    ):
        assumed = compute_spatial_rates(
            r.env, trains, r.times, r.positions, method="binned", bandwidth=0
        )
    pooled = (
        np.nansum(assumed.firing_rates * assumed.occupancy, axis=1)
        / assumed.occupancy.sum()
    )
    np.testing.assert_allclose(pooled, 2.5, rtol=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        explicit = compute_spatial_rates(
            r.env,
            trains,
            r.times,
            r.positions,
            method="binned",
            bandwidth=0,
            spike_window=(100, 200),
        )
    pooled = (
        np.nansum(explicit.firing_rates * explicit.occupancy, axis=1)
        / explicit.occupancy.sum()
    )
    np.testing.assert_allclose(pooled, 5.0, rtol=0.05)


@pytest.mark.parametrize(
    ("n_units", "silent_seconds", "expected"),
    [(5, 0, 0), (4, 100, 0), (5, 59, 0), (5, 61, 1)],
)
def test_population_silence_thresholds(
    continuous_recording, n_units, silent_seconds, expected
):
    r = continuous_recording
    trains = []
    for unit in range(n_units):
        spikes = r.spike_times + 0.04 * unit
        spikes = spikes[(spikes < 40) | (spikes >= 40 + silent_seconds)]
        trains.append(spikes)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_spatial_rates(r.env, trains, r.times, r.positions, method="binned")
    silence = [w for w in caught if str(w.message).startswith("All ")]
    assert len(silence) == expected


def test_population_silence_does_not_bridge_untracked_pause(two_epoch_recording):
    r = two_epoch_recording
    trains = []
    for unit in range(5):
        spikes = r.spike_times + 0.04 * unit
        trains.append(spikes[(spikes < 70) | (spikes >= 1131)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_spatial_rates(r.env, trains, r.times, r.positions, method="binned")
    assert not any(str(w.message).startswith("All ") for w in caught)


def test_population_trailing_silence_warns_at_caller(continuous_recording):
    r = continuous_recording
    trains = []
    for unit in range(5):
        spikes = np.arange(0.1, 100, 0.2) + 0.04 * unit
        trains.append(spikes[spikes < 100])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_spatial_rates(r.env, trains, r.times, r.positions, method="binned")
    silence = [w for w in caught if str(w.message).startswith("All ")]
    assert len(silence) == 1
    assert "to 200.0 s" in str(silence[0].message)
    assert silence[0].filename == __file__


@pytest.mark.parametrize(("stop", "expected"), [(59.0, 0), (60.0, 1)])
def test_population_silence_respects_analysis_epochs(
    continuous_recording, stop, expected
):
    r = continuous_recording
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_spatial_rates(
            r.env,
            [np.array([])] * 5,
            r.times,
            r.positions,
            method="binned",
            epochs=(0, stop),
        )
    silence = [w for w in caught if str(w.message).startswith("All ")]
    assert len(silence) == expected


def test_population_silence_during_rest(continuous_recording):
    r = continuous_recording
    # At rest for the first half, running in the second: silent throughout.
    speed = np.where(r.times < r.times[-1] / 2, 0.0, 10.0)
    with pytest.warns(UserWarning, match="All 5 units are silent"):
        compute_spatial_rates(
            r.env,
            [np.array([])] * 5,
            r.times,
            r.positions,
            method="binned",
            speed=speed,
            min_speed=1,
            warn_on_drop=False,
        )


def test_boundary_vector_population_silence_warns_at_caller(continuous_recording):
    from neurospatial.encoding import compute_boundary_vector_rates

    r = continuous_recording
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_boundary_vector_rates(
            r.env,
            [np.array([])] * 5,
            r.times,
            r.positions,
            direction_frame="allocentric",
            boundary_source="analysis",
        )
    silence = [w for w in caught if str(w.message).startswith("All ")]
    assert len(silence) == 1
    assert "All 5 units are silent" in str(silence[0].message)
    assert silence[0].filename == __file__


def test_population_silence_skips_lost_tracking_like_deleted_rows(
    continuous_recording,
):
    """Silence during 79 s of NaN positions is not silence while tracked."""
    r = continuous_recording
    trains = []
    for unit in range(5):
        spikes = r.spike_times + 0.04 * unit
        trains.append(spikes[(spikes < 40) | (spikes >= 121)])
    lost = (r.times > 41) & (r.times < 120)
    positions = r.positions.copy()
    positions[lost] = np.nan
    for times, rows in ((r.times, positions), (r.times[~lost], r.positions[~lost])):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_spatial_rates(r.env, trains, times, rows, method="binned")
        assert not any(str(w.message).startswith("All ") for w in caught)


def _longest_silence(spikes: np.ndarray, runs: np.ndarray) -> float:
    """Longest stretch between run edges and spikes within one run, by loop."""
    longest = 0.0
    for start, stop in runs:
        inside = np.sort(spikes[(spikes >= start) & (spikes < stop)])
        longest = max(longest, float(np.diff(np.r_[start, inside, stop]).max()))
    return longest


def _silence_cases(rng: np.random.Generator):
    """Runs and spikes with silences near the threshold, many on bin edges."""
    width = _SILENCE_MIN_SECONDS / 3
    for _ in range(400):
        origin = rng.choice([0.0, 12.345, 1.7e9])
        n_runs = int(rng.integers(1, 4))
        lengths = rng.uniform(30, 400, n_runs)
        gaps = rng.uniform(1, 50, n_runs)
        starts = origin + np.r_[0.0, np.cumsum(lengths + gaps)[:-1]]
        runs = np.column_stack([starts, starts + lengths])
        spikes = np.concatenate([rng.uniform(a, b, int(b - a)) for a, b in runs])
        spikes = np.r_[spikes, rng.uniform(origin - 100, runs[-1, 1] + 100, 20)]
        # Clear a silence of length about the threshold, starting on a bin
        # edge, just after one, or at a run start, with spikes at its ends.
        start, stop = runs[rng.integers(n_runs)]
        length = _SILENCE_MIN_SECONDS + rng.choice([-0.5, 0.0, 1e-9, 0.5, 15.0])
        anchor = rng.choice(["edge", "after_edge", "run_start", "random"])
        if anchor == "run_start":
            a = start
        else:
            k = np.ceil(
                (rng.uniform(start, max(start, stop - length)) - origin) / width
            )
            a = origin + k * width
            if anchor == "after_edge":
                a += rng.choice([1e-9, width / 2, width - 1e-9])
            elif anchor == "random":
                a = rng.uniform(start, stop)
        b = a + length
        spikes = spikes[(spikes <= a) | (spikes >= b)]
        if a > start:
            spikes = np.r_[spikes, a]
        if b < stop:
            spikes = np.r_[spikes, b]
        yield rng.permutation(spikes), runs


def test_silence_prefilter_never_misses_a_long_silence():
    """An empty whole bin exists inside a run whenever a silence reaches 60 s."""
    width = _SILENCE_MIN_SECONDS / 3
    outcomes = set()
    for spikes, runs in _silence_cases(np.random.default_rng(11)):
        found = _has_empty_observed_bin(spikes, runs, width)
        outcomes.add((_longest_silence(spikes, runs) >= _SILENCE_MIN_SECONDS, found))
        if _longest_silence(spikes, runs) >= _SILENCE_MIN_SECONDS:
            assert found, (spikes, runs)
    # Both long and short silences were drawn, and short ones are filtered.
    assert {(True, True), (False, False)} <= outcomes


def test_silence_prefilter_does_not_change_the_warning(monkeypatch):
    """The warning is the same with and without the prefilter."""
    rng = np.random.default_rng(5)
    times = np.sort(rng.uniform(0, 2000, 40_000))
    times[20_000:] += 5.0  # a recording gap longer than max_gap
    tracked = np.ones(times.size, dtype=bool)
    tracked[5_000:5_400] = False

    def messages(trains):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_population_silent(
                trains,
                times,
                max_gap=1.0,
                epochs=None,
                spike_window=None,
                tracked=tracked,
            )
        return [str(w.message) for w in caught]

    seen = set()
    for silent in (0.0, 45.0, 59.9, 60.0, 61.0, 150.0):
        for _ in range(10):
            a = rng.uniform(0, 2000 - silent)
            trains = []
            for _unit in range(5):
                spikes = np.sort(rng.uniform(0, times[-1], 400))
                trains.append(spikes[(spikes < a) | (spikes >= a + silent)])
            with_prefilter = messages(trains)
            with monkeypatch.context() as patch:
                patch.setattr(
                    spatial_counts, "_has_empty_observed_bin", lambda *_: True
                )
                without = messages(trains)
            assert with_prefilter == without
            seen.add(bool(with_prefilter))
    assert seen == {True, False}
