"""Lap sessions honor duration without discarding traversals or pauses."""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.simulation import (
    linear_track_session,
    simulate_session,
    tmaze_alternation_session,
)


@pytest.fixture
def short_track():
    return Environment.from_samples(
        np.linspace(0.0, 1.0, 11)[:, None], bin_size=0.1, units="cm"
    )


def assert_recording_clock(sim, duration, frequency):
    assert sim.metadata["duration"] == duration
    assert sim.times[0] == 0.0
    assert 0 < duration - sim.times[-1] <= 1.0 / frequency + 1e-12
    np.testing.assert_allclose(np.diff(sim.times), 1.0 / frequency, atol=1e-12)
    assert sim.positions.shape == (len(sim.times), sim.env.n_dims)
    assert np.all(np.isfinite(sim.positions))
    assert all(
        np.all((spikes >= 0.0) & (spikes < duration)) for spikes in sim.spike_trains
    )


@pytest.mark.parametrize("duration", [1.0, 2.0, 1.001])
def test_linear_track_requested_duration_controls_clock(duration):
    sim = linear_track_session(
        duration=duration,
        track_length=1.0,
        bin_size=0.2,
        n_place_cells=2,
        n_laps=2,
        seed=42,
    )
    assert_recording_clock(sim, duration, 500.0)


def test_tmaze_requested_duration_controls_clock():
    sim = tmaze_alternation_session(duration=1.0, n_trials=1, n_place_cells=1, seed=42)
    assert_recording_clock(sim, 1.0, 500.0)


@pytest.mark.parametrize("duration", [1.25, 2.25])
def test_lap_session_keeps_every_requested_traversal(short_track, duration):
    sim = simulate_session(
        short_track,
        duration=duration,
        trajectory_method="laps",
        n_laps=4,
        pause_duration=0.1,
        sampling_frequency=40.0,
        n_cells=2,
        max_rate=200.0,
        seed=42,
        show_progress=False,
    )
    assert_recording_clock(sim, duration, 40.0)
    moving = np.sign(np.diff(sim.positions[:, 0]))
    moving = moving[moving != 0]
    directions = moving[np.r_[True, np.diff(moving) != 0]]
    np.testing.assert_array_equal(directions, [1, -1, 1, -1])
    assert sum(len(spikes) for spikes in sim.spike_trains) > 0


def test_minimum_sample_budget_keeps_pause_and_both_endpoints(short_track):
    sim = simulate_session(
        short_track,
        duration=1.008,
        trajectory_method="laps",
        n_laps=2,
        pause_duration=1.0,
        sampling_frequency=500.0,
        n_cells=1,
        seed=42,
        show_progress=False,
    )
    assert_recording_clock(sim, 1.008, 500.0)
    start = short_track.bin_centers[:, 0].min()
    end = short_track.bin_centers[:, 0].max()
    assert sim.positions[0, 0] == sim.positions[-1, 0] == start
    np.testing.assert_array_equal(sim.positions[1:-1, 0], end)
    assert (
        len(sim.times) == 4 + 500
    )  # Two endpoints per traversal plus the fixed pause.


def test_duration_too_short_for_pauses_and_traversals_raises():
    with pytest.raises(ValueError, match=r"duration.*n_laps") as raised:
        linear_track_session(
            duration=0.5,
            track_length=1.0,
            bin_size=0.2,
            n_place_cells=1,
            n_laps=2,
            seed=42,
        )
    assert "Why:" in str(raised.value) and "Fix:" in str(raised.value)


@pytest.mark.parametrize("duration", [np.nan, np.inf])
def test_lap_duration_requires_finite_seconds(short_track, duration):
    with pytest.raises(ValueError, match="duration") as raised:
        simulate_session(
            short_track,
            duration=duration,
            trajectory_method="laps",
            n_laps=1,
            n_cells=1,
            show_progress=False,
        )
    assert "Fix:" in str(raised.value)


def test_duration_controlled_laps_are_seeded(short_track):
    kwargs = {
        "duration": 1.25,
        "trajectory_method": "laps",
        "n_laps": 3,
        "pause_duration": 0.1,
        "sampling_frequency": 40.0,
        "n_cells": 2,
        "max_rate": 200.0,
        "seed": 42,
        "show_progress": False,
    }
    first = simulate_session(short_track, **kwargs)
    second = simulate_session(short_track, **kwargs)
    np.testing.assert_array_equal(first.positions, second.positions)
    np.testing.assert_array_equal(first.times, second.times)
    for left, right in zip(first.spike_trains, second.spike_trains, strict=True):
        np.testing.assert_array_equal(left, right)
