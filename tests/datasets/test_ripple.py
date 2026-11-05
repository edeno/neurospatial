"""The ripple_detection adapter: spikes and speed on the LFP's grid, events found."""

from __future__ import annotations

import builtins
import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest

from conftest import _identity, unit_table
from neurospatial import Environment
from neurospatial._exceptions import InputError
from neurospatial.datasets import Conversion, LFPSource, Recording
from neurospatial.datasets._lfp import LFPFile
from neurospatial.datasets._ripple import (
    RIPPLE_DETECTION_COMMIT,
    event_rows,
    ripple_recording,
    spike_counts,
)

RATE = 1500.0
ROOT = Path(__file__).parents[2]


def _lfp_source(rate: float = RATE) -> LFPSource:
    basis = Conversion("as stored", "stated", "the simulation")
    return LFPSource(
        dataset="synthetic_2026",
        files=(LFPFile(path="a.bin", sha256="a" * 64, size_bytes=1, url="u"),),
        sampling_rate=rate,
        channels={"pyramidal": "the ripple channel"},
        unit="uV",
        unit_basis=basis,
        times_basis=basis,
        reader=lambda sources, channels: (np.zeros((2, 1)), np.arange(2.0) / rate),
    )


def _session(spike_times, times, positions, *, units="cm") -> Recording:
    """A session of these spikes and positions, with an LFP source at ``RATE``."""
    env = Environment.from_samples(
        positions,
        bin_size=0.05 if units == "m" else 5.0,
        units="pixels" if units == "pixel" else units,
    )
    return Recording(
        **_identity("ripple"),
        times=times,
        spike_times=tuple(spike_times),
        units=unit_table(len(spike_times)),
        env=env,
        env_kind="inferred",
        env_source="the samples (simulation)",
        positions=positions,
        position_units=units,
        tracking="simulated head position",
        lfp=_lfp_source(),
    )


def _slow_path(duration: float, *, speed: float = 1.0):
    """Tracking at 30 Hz along a line, at a constant ``speed`` per second."""
    times = np.arange(0.0, duration, 1 / 30)
    return times, np.column_stack([speed * times, np.zeros_like(times)])


def test_a_count_belongs_to_the_sample_at_or_before_the_spike():
    """Half-open cells: a spike exactly on a sample is that sample's."""
    grid = np.arange(4) / 10.0
    counts = spike_counts(
        [np.array([0.0, 0.05, 0.1, 0.25, 0.35])], grid, sampling_rate=10.0
    )
    assert counts[:, 0].tolist() == [2, 1, 1, 1]


def test_spikes_off_the_grid_are_not_counted():
    """Before the first sample, in a gap, and past the last period count nowhere."""
    grid = np.array([1.0, 1.1, 1.2, 5.0, 5.1])
    train = np.array([0.5, 1.05, 1.25, 1.35, 4.9, 5.05, 5.15, 5.25])
    counts = spike_counts([train], grid, sampling_rate=10.0)
    assert counts[:, 0].tolist() == [1, 0, 1, 1, 1]


def test_counts_match_a_histogram_on_a_regular_grid():
    """Checked against np.histogram, which shares no code with the adapter."""
    rng = np.random.default_rng(0)
    grid = np.arange(2000) / 1000.0
    train = np.sort(rng.uniform(grid[0], grid[-1] + 1e-3, 5000))
    expected = np.histogram(train, np.r_[grid, grid[-1] + 1e-3])[0]
    assert np.array_equal(
        spike_counts([train], grid, sampling_rate=1000.0)[:, 0], expected
    )


def test_counts_take_one_byte_unless_a_sample_needs_more():
    grid = np.arange(4.0)
    assert spike_counts([np.array([0.5])], grid, sampling_rate=1.0).dtype == np.uint8
    burst = np.full(300, 1.5)
    counts = spike_counts([burst], grid, sampling_rate=1.0)
    assert counts.dtype == np.uint16
    assert counts[1, 0] == 300


def test_a_unit_without_spikes_and_a_session_without_units():
    grid = np.arange(3.0)
    assert spike_counts([np.array([])], grid, sampling_rate=1.0).tolist() == [
        [0],
        [0],
        [0],
    ]
    assert spike_counts([], grid, sampling_rate=1.0).shape == (3, 0)


@pytest.fixture
def ld():
    """``ripple_detection.literature_methods``; absent in the floors' environment."""
    return pytest.importorskip("ripple_detection.literature_methods")


def _grid(duration: float) -> np.ndarray:
    return np.arange(int(duration * RATE)) / RATE


@pytest.mark.parametrize(("units", "speed"), [("cm", 7.0), ("m", 0.07)])
def test_speed_reaches_ripple_detection_in_cm_per_s(ld, units, speed):
    """7 cm/s stays 7 cm/s from positions in cm and in m."""
    times, positions = _slow_path(20.0, speed=speed)
    rec = _session([np.array([1.0])], times, positions, units=units)
    grid = _grid(10.0)[int(RATE) :]
    signals = ripple_recording(rec, np.zeros((grid.size, 1)), grid)
    assert np.nanmedian(signals.speed) == pytest.approx(7.0, rel=1e-3)


def test_speed_is_unknown_outside_tracking_and_across_gaps(ld):
    """Missing stays NaN, not zero: outside the span and over a long gap."""
    times, positions = _slow_path(20.0)
    kept = (times < 8.0) | (times > 12.0)
    rec = _session([np.array([1.0])], times[kept], positions[kept])
    assert rec._max_gap == pytest.approx(1.0)
    grid = _grid(25.0)
    speed = ripple_recording(rec, np.zeros((grid.size, 1)), grid).speed
    assert np.all(np.isnan(speed[(grid > 8.1) & (grid < 11.9)]))
    assert np.all(np.isnan(speed[grid > 20.0]))
    assert np.all(np.isfinite(speed[(grid > 2.0) & (grid < 7.0)]))


def test_pixel_positions_give_no_speed(ld):
    """A method needing speed then refuses, rather than reading pixels as cm."""
    times, positions = _slow_path(10.0, speed=5.0)
    rec = _session([np.array([1.0])], times, positions, units="pixel")
    grid = _grid(2.0)
    signals = ripple_recording(rec, np.zeros((grid.size, 1)), grid)
    with pytest.raises(ValueError, match="speed"):
        ld.run_method("igata_2021", signals)


def test_selections_are_by_unit_id(ld):
    times, positions = _slow_path(10.0)
    rec = _session([np.array([1.0])] * 3, times, positions)
    grid = _grid(2.0)
    signals = ripple_recording(
        rec, np.zeros((grid.size, 1)), grid, place_cells=[2, 0], pyramidal=[1]
    )
    assert signals.place_cells.tolist() == [True, False, True]
    assert signals.pyramidal.tolist() == [False, True, False]
    with pytest.raises(InputError, match=r"\[7\] are not units"):
        ripple_recording(rec, np.zeros((grid.size, 1)), grid, place_cells=[7])


def test_no_selection_is_invented(ld):
    """Nothing the caller did not pass reaches the recording."""
    times, positions = _slow_path(10.0)
    rec = _session([np.array([1.0])] * 2, times, positions)
    grid = _grid(2.0)
    signals = ripple_recording(rec, np.zeros((grid.size, 1)), grid)
    assert not signals.place_cells.any()
    assert not signals.pyramidal.any()
    assert signals.sleep_intervals is None
    assert signals.baseline_intervals is None
    assert signals.example_ripples is None
    assert signals.external_ripples is None


def test_a_session_without_lfp_is_refused(ld):
    times, positions = _slow_path(10.0)
    rec = dataclasses.replace(_session([np.array([1.0])], times, positions), lfp=None)
    with pytest.raises(InputError, match="has no LFP source"):
        ripple_recording(rec, np.zeros((10, 1)), np.arange(10) / RATE)


def test_without_ripple_detection_it_says_how_to_install(monkeypatch):
    real_import = builtins.__import__

    def absent(name, *args, **kwargs):
        if name.partition(".")[0] == "ripple_detection":
            msg = "No module named 'ripple_detection'"
            raise ModuleNotFoundError(msg, name="ripple_detection")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", absent)
    times, positions = _slow_path(10.0)
    rec = _session([np.array([1.0])], times, positions)
    with pytest.raises(ImportError, match=RIPPLE_DETECTION_COMMIT):
        ripple_recording(rec, np.zeros((10, 1)), np.arange(10) / RATE)


def test_the_pinned_commit_is_the_one_in_pyproject():
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert re.search(
        rf'^ripple-detection = .*rev = "{RIPPLE_DETECTION_COMMIT}"',
        pyproject,
        re.MULTILINE,
    )


class TestKnownEventsAreFound:
    """A simulated session goes through the adapter and its ripples come back."""

    RIPPLES = (6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0)

    def _found(self, events) -> list[bool]:
        """Whether an event overlaps each simulated ripple's 100 ms."""
        starts = events["start_time"].to_numpy()
        ends = events["end_time"].to_numpy()
        return [
            bool(np.any((starts <= ripple + 0.05) & (ends >= ripple - 0.05)))
            for ripple in self.RIPPLES
        ]

    @pytest.fixture(scope="class")
    def simulated(self):
        simulate = pytest.importorskip("ripple_detection.simulate")
        return simulate.simulate_session(
            simulate.simulate_time(int(48 * RATE), RATE),
            self.RIPPLES,
            n_units=100,
            ripple_rate_gain=12.0,
            ripple_duration=0.1,
            rng=0,
            sampling_frequency=RATE,
        )

    @pytest.fixture
    def signals(self, simulated):
        """The simulated counts as spike times, through the adapter."""
        spike_times = [
            simulated.time[np.repeat(np.arange(column.size), column.astype(int))]
            for column in simulated.multiunit.T
        ]
        times, positions = _slow_path(48.0, speed=0.5)
        rec = _session(spike_times, times, positions)
        return ripple_recording(
            rec,
            simulated.lfps[:, :1],
            simulated.time,
            example_ripples=simulated.ripple_windows,
        )

    def test_counts_arrive_unchanged(self, simulated, signals):
        assert np.array_equal(signals.multiunit, simulated.multiunit)

    def test_slow_animal_is_slow(self, signals):
        assert np.nanmax(signals.speed) == pytest.approx(0.5, rel=1e-3)

    def test_every_simulated_ripple_is_found_from_spikes(self, ld, signals):
        """Igata's population rule finds each simulated burst."""
        assert all(self._found(ld.run_method("igata_2021", signals)))

    def test_carey_finds_what_it_finds_from_the_arrays_themselves(
        self, ld, simulated, signals
    ):
        """The adapter changes nothing: the same events as the simulator's arrays.

        Carey's joint score misses one of the seven on this seed either way, so
        the check is the equality, with the six it finds pinned.
        """
        direct = ld.Recording.from_arrays(
            simulated.time,
            RATE,
            lfps=simulated.lfps[:, :1],
            multiunit=simulated.multiunit,
            speed=simulated.speed,
            example_ripples=simulated.ripple_windows,
        )
        expected = ld.run_method("carey_2019", direct)
        events = ld.run_method("carey_2019", signals)
        assert sum(self._found(events)) == 6
        np.testing.assert_allclose(
            events[["start_time", "end_time"]], expected[["start_time", "end_time"]]
        )

    def test_spikes_moved_away_lose_the_events(self, ld, simulated):
        """The population method finds them through the spikes, not by accident."""
        times, positions = _slow_path(48.0, speed=0.5)
        shifted = [
            np.sort(
                (
                    simulated.time[
                        np.repeat(np.arange(column.size), column.astype(int))
                    ]
                    + 3.0
                )
                % 48.0
            )
            for column in simulated.multiunit.T
        ]
        rec = _session(shifted, times, positions)
        events = ld.run_method(
            "igata_2021",
            ripple_recording(rec, simulated.lfps[:, :1], simulated.time),
        )
        assert not any(self._found(events))


def _in_a_row(train, start, stop, *, time_bin, step):
    """Whether each spike of ``train`` is in a row, counted one row at a time."""
    counts, _ = event_rows([train], start, stop, time_bin=time_bin, step=step)
    n_rows = counts.shape[0]
    left = start + step * np.arange(n_rows)
    right = np.minimum(left + time_bin, stop)
    held = (train[:, None] >= left) & (train[:, None] < right)
    return held.any(axis=1)


@pytest.mark.parametrize(
    ("time_bin", "step"), [(0.02, 0.01), (0.02, 0.005), (0.02, 0.02)]
)
@pytest.mark.parametrize("stop", [0.1, 0.105, 0.1051, 0.1099, 0.113])
def test_event_rows_hold_every_spike_up_to_the_stop(time_bin, step, stop):
    """The last spike is in a row for any width, and nothing past the stop is."""
    train = np.sort(np.random.default_rng(0).uniform(0.0, 0.2, 200))
    inside = (train >= 0.0) & (train < stop)
    assert np.array_equal(
        _in_a_row(train, 0.0, stop, time_bin=time_bin, step=step), inside
    )
    last = np.array([stop - 1e-6])
    assert event_rows([last], 0.0, stop, time_bin=time_bin, step=step)[0].sum() >= 1
    assert event_rows([np.array([stop])], 0.0, stop, time_bin=time_bin, step=step)[
        0
    ].sum() == (0)


def test_event_rows_equal_hand_summed_steps_over_whole_steps():
    """Over whole steps the rows are the sum of tiled bins, row for row and on time."""
    from neurospatial.decoding import bin_spikes_in_time

    rng = np.random.default_rng(1)
    trains = [np.sort(rng.uniform(1.0, 1.2, 40)) for _ in range(3)]
    counts, centers = event_rows(trains, 1.0, 1.14, time_bin=0.02, step=0.005)
    tiled, _ = bin_spikes_in_time(trains, dt=0.005, t_start=1.0, t_stop=1.14)
    running = np.vstack([np.zeros((1, 3), np.int64), np.cumsum(tiled, axis=0)])
    np.testing.assert_array_equal(counts, running[4:] - running[:-4])
    np.testing.assert_allclose(centers, 1.01 + 0.005 * np.arange(counts.shape[0]))


def test_event_rows_add_one_cut_row_for_a_partial_step():
    """A partial step adds one row, no narrower than a bin less a step."""
    train = np.array([0.0449])
    full, _ = event_rows([train], 0.0, 0.04, time_bin=0.02, step=0.01)
    partial, _ = event_rows([train], 0.0, 0.045, time_bin=0.02, step=0.01)
    assert (full.shape[0], partial.shape[0]) == (3, 4)
    assert (full.sum(), partial.sum()) == (0, 1)


def test_event_rows_give_one_row_to_an_event_shorter_than_a_bin():
    counts, centers = event_rows(
        [np.array([0.001, 0.008]), np.array([0.02])],
        0.0,
        0.01,
        time_bin=0.02,
        step=0.01,
    )
    np.testing.assert_array_equal(counts, [[2, 0]])
    np.testing.assert_allclose(centers, [0.01])
