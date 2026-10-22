"""Lap trajectories look like a running animal: realistic speed, continuous path."""

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.simulation import (
    linear_track_session,
    simulate_trajectory_laps,
    tmaze_alternation_session,
)

DEFAULT_SPEED_CM_PER_S = 20.0
MAX_SPEED_CM_PER_S = 1.5 * DEFAULT_SPEED_CM_PER_S
SAMPLING_FREQUENCY = 500.0
PAUSE_SAMPLES = int(0.5 * SAMPLING_FREQUENCY)  # default pause_duration=0.5 s


@pytest.fixture(scope="module")
def box_env():
    """A 100 x 20 cm box with 2 cm bins."""
    return Environment.from_polygon(box(0, 0, 100, 20), bin_size=2, units="cm")


@pytest.fixture(scope="module")
def box_laps(box_env):
    """Three traversals of the box with the default speed."""
    return simulate_trajectory_laps(box_env, n_laps=3, seed=1, return_metadata=True)


def _pause_mask(metadata):
    """True for the pause samples that end every lap but the last."""
    lap_ids = metadata["lap_ids"]
    paused = np.zeros(lap_ids.size, dtype=bool)
    for lap in np.unique(lap_ids)[:-1]:
        paused[np.flatnonzero(lap_ids == lap)[-PAUSE_SAMPLES:]] = True
    return paused


def _moving_steps(metadata):
    """True for steps whose two samples are outside pauses and not a lap start."""
    paused = _pause_mask(metadata)
    lap_start = np.zeros(paused.size, dtype=bool)
    lap_start[metadata["lap_boundaries"]] = True
    return ~paused[:-1] & ~paused[1:] & ~lap_start[1:]


def _steps(positions):
    return np.linalg.norm(np.diff(positions, axis=0), axis=1)


def test_duration_is_plausible(box_laps):
    _positions, times, _metadata = box_laps
    assert 10.0 < times[-1] < 20.0
    assert len(times) < 20_000


def test_running_speed_is_realistic(box_laps):
    positions, times, metadata = box_laps
    moving = _moving_steps(metadata)
    speed = _steps(positions)[moving] / np.diff(times)[moving]
    assert 15.0 <= np.median(speed) <= 30.0
    assert speed.max() < MAX_SPEED_CM_PER_S


def test_path_is_smooth(box_laps):
    positions, times, metadata = box_laps
    moving_steps = _steps(positions)[_moving_steps(metadata)]
    assert np.all(moving_steps > 0)
    assert moving_steps.max() <= 1.01 * MAX_SPEED_CM_PER_S / SAMPLING_FREQUENCY
    assert len(np.unique(positions, axis=0)) > 0.9 * len(times)


def test_pauses_only_at_lap_ends(box_laps):
    positions, _times, metadata = box_laps
    zero_steps = np.flatnonzero(_steps(positions) == 0)
    # Step i joins samples i and i + 1. The pause before the lap starting at
    # boundary b fills samples [b - PAUSE_SAMPLES, b) at the previous lap's end,
    # and the next lap starts there, so its zero steps are [b - PAUSE_SAMPLES - 1, b).
    boundaries = metadata["lap_boundaries"][1:]
    at_lap_end = np.any(
        (zero_steps[:, None] >= boundaries - PAUSE_SAMPLES - 1)
        & (zero_steps[:, None] < boundaries),
        axis=1,
    )
    assert np.all(at_lap_end)
    # Every lap end but the last holds still for the whole pause.
    for boundary in boundaries:
        pause_steps = _steps(positions)[boundary - PAUSE_SAMPLES - 1 : boundary]
        assert np.all(pause_steps == 0)


def test_laps_stay_inside_environment(box_env, box_laps):
    positions, _times, _metadata = box_laps
    assert np.all(box_env.bin_at(positions) >= 0)


@pytest.mark.parametrize(
    "make_session",
    [
        lambda: linear_track_session(duration=60, n_place_cells=3, n_laps=6, seed=42),
        lambda: tmaze_alternation_session(
            duration=60, n_trials=4, n_place_cells=3, seed=1
        ),
    ],
    ids=["linear_track", "tmaze"],
)
@pytest.mark.filterwarnings("ignore:Requested n_cells=3:UserWarning")
def test_session_laps_stay_inside_environment(make_session):
    session = make_session()
    assert np.all(session.env.bin_at(session.positions) >= 0)


def test_session_paths_are_smooth():
    session = linear_track_session(duration=60, n_laps=6, seed=42)
    pause_samples = int(
        session.metadata["pause_duration"] * session.metadata["sampling_frequency"]
    )
    zero = _steps(session.positions) == 0
    # A pause is a run of at least pause_samples zero steps; nothing else is zero.
    edges = np.flatnonzero(np.diff(np.concatenate([[0], zero.astype(int), [0]])))
    run_lengths = edges[1::2] - edges[::2]
    assert np.all(run_lengths >= pause_samples)
    assert len(run_lengths) == 6 - 1


def test_default_speed_follows_environment_units(box_laps):
    positions_cm, times_cm, _metadata = box_laps
    env_m = Environment.from_polygon(box(0, 0, 1, 0.2), bin_size=0.02, units="m")
    positions_m, times_m = simulate_trajectory_laps(env_m, n_laps=3, seed=1)
    np.testing.assert_allclose(times_m, times_cm, rtol=1e-12)
    np.testing.assert_allclose(positions_m, positions_cm / 100, rtol=1e-12)


def test_default_speed_needs_metric_units():
    env = Environment.from_polygon(box(0, 0, 100, 20), bin_size=2)
    with pytest.raises(ValueError, match="speed_mean"):
        simulate_trajectory_laps(env, n_laps=3, seed=1)
    positions, _times = simulate_trajectory_laps(env, n_laps=3, seed=1, speed_mean=20.0)
    assert np.all(env.bin_at(positions) >= 0)


def test_custom_path_must_be_chain_of_neighbors(box_env):
    assert not box_env.connectivity.has_edge(0, 2)
    with pytest.raises(ValueError, match="neighboring"):
        simulate_trajectory_laps(box_env, n_laps=2, outbound_path=[0, 2])
