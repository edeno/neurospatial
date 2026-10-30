"""Tests for simulate_replay_event: simulated candidate events with known paths."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import bin_spikes_in_time, decode_position
from neurospatial.replay import (
    detect_trajectory_radon,
    replay_significance,
    weighted_correlation,
)
from neurospatial.simulation import ReplayEventSimulation, simulate_replay_event

DT = 0.02


@pytest.fixture(scope="module")
def place_code():
    """40 place cells on a 200 cm track in 4 cm bins."""
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [200.0, 0.0]], bin_size=4.0, units="cm"
    )
    x = env.bin_centers[:, 0]
    rates = 0.5 + 25.0 * np.exp(
        -0.5 * ((x - np.linspace(0.0, 200.0, 40)[:, None]) / 10.0) ** 2
    )
    return env, rates


def _decode(env, rates, event):
    counts, grid = bin_spikes_in_time(
        list(event.spike_times), dt=DT, t_start=0.0, t_stop=event.duration
    )
    return decode_position(env, counts, rates, dt=DT, times=grid)


class TestPaths:
    def test_sweep_moves_at_its_speed(self, place_code):
        env, rates = place_code
        event = simulate_replay_event(env, rates, speed=600.0, seed=0)
        slope = np.polyfit(event.times, event.track_positions, 1)[0]
        assert slope == pytest.approx(600.0)
        assert event.track_positions[0] == pytest.approx(0.0)
        assert event.speed == 600.0

    def test_backward_sweep_starts_at_the_track_end(self, place_code):
        env, rates = place_code
        event = simulate_replay_event(env, rates, speed=-600.0, seed=0)
        assert event.track_positions[0] == pytest.approx(200.0, abs=1e-6)
        assert np.all(np.diff(event.track_positions) < 0)

    def test_stationary_event_stays_put(self, place_code):
        env, rates = place_code
        event = simulate_replay_event(env, rates, kind="stationary", seed=0)
        assert np.all(event.track_positions == 100.0)
        assert event.speed == 0.0

    def test_jump_has_two_fragments_jump_distance_apart(self, place_code):
        env, rates = place_code
        event = simulate_replay_event(
            env, rates, kind="jump", speed=400.0, jump_distance=60.0, seed=0
        )
        steps = np.diff(event.track_positions)
        jump = np.argmax(np.abs(steps))
        assert steps[jump] == pytest.approx(60.0 + 400.0 * 0.001)
        assert np.sum(np.abs(steps) > 1.0) == 1
        default = simulate_replay_event(env, rates, kind="jump", speed=600.0, seed=0)
        assert np.max(np.diff(default.track_positions)) == pytest.approx(
            50.0 + 0.6, abs=0.01
        )  # a quarter of the track, at any speed

    def test_path_skips_the_gaps_between_track_segments(self):
        """A linearized track with edge_spacing: no sample falls in a gap."""
        import networkx as nx

        graph = nx.Graph()
        graph.add_node(0, pos=(0.0, 0.0))
        graph.add_node(1, pos=(100.0, 0.0))
        graph.add_node(2, pos=(100.0, 100.0))
        graph.add_edge(0, 1, distance=100.0)
        graph.add_edge(1, 2, distance=100.0)
        env = Environment.from_graph(
            graph,
            edge_order=[(0, 1), (1, 2)],
            edge_spacing=30.0,
            bin_size=10.0,
            units="cm",
        )
        rates = np.ones((5, env.n_bins))
        for kind, speed in (("sweep", 900.0), ("random", None), ("stationary", None)):
            event = simulate_replay_event(env, rates, kind=kind, speed=speed, seed=1)
            assert np.isfinite(event.positions).all(), kind
        sweep = simulate_replay_event(env, rates, speed=900.0, seed=1)
        on_segment = np.isclose(sweep.positions[:, 1], 0.0) | np.isclose(
            sweep.positions[:, 0], 100.0
        )
        assert on_segment.all()

    def test_random_event_draws_a_position_per_interval(self, place_code):
        env, rates = place_code
        event = simulate_replay_event(env, rates, kind="random", seed=0)
        assert np.unique(event.track_positions).size == 10  # 0.2 s / 20 ms
        assert np.isnan(event.speed)

    def test_positions_are_environment_coordinates(self, place_code):
        env, rates = place_code
        event = simulate_replay_event(env, rates, speed=600.0, seed=0)
        np.testing.assert_allclose(event.positions[:, 0], event.track_positions)
        np.testing.assert_allclose(event.positions[:, 1], 0.0)


class TestSpikes:
    def test_cells_fire_where_the_path_is(self, place_code):
        env, rates = place_code
        event = simulate_replay_event(env, rates, speed=900.0, rate_gain=4.0, seed=1)
        fields = np.linspace(0.0, 200.0, 40)
        first_spikes = np.array(
            [t.min() if t.size else np.nan for t in event.spike_times]
        )
        active = np.isfinite(first_spikes) & (fields < 170.0)
        rho = np.corrcoef(fields[active], first_spikes[active])[0, 1]
        assert rho > 0.8

    def test_mean_spike_count_matches_the_rate_integral(self, place_code):
        """Expected count: sum of g r / (1 + g r tau) dt, with a 2 ms refractory."""
        env, rates = place_code
        tau = 0.002
        for gain in (1.0, 4.0):
            events = [
                simulate_replay_event(
                    env, rates, kind="stationary", rate_gain=gain, seed=s
                )
                for s in range(200)
            ]
            rate = gain * rates[:, np.argmin(np.abs(env.bin_centers[:, 0] - 100.0))]
            expected = np.sum(rate / (1 + rate * tau)) * events[0].duration
            observed = np.mean([e.summary()["n_spikes"] for e in events])
            assert observed == pytest.approx(expected, rel=0.05)

    def test_seed_repeats_the_event(self, place_code):
        env, rates = place_code
        first = simulate_replay_event(env, rates, kind="random", seed=5)
        second = simulate_replay_event(env, rates, kind="random", seed=5)
        np.testing.assert_array_equal(first.track_positions, second.track_positions)
        for a, b in zip(first.spike_times, second.spike_times, strict=True):
            np.testing.assert_array_equal(a, b)


class TestWithReplayMethods:
    def test_planted_sweep_is_detected_with_its_speed(self, place_code):
        env, rates = place_code
        event = simulate_replay_event(env, rates, speed=900.0, rate_gain=4.0, seed=2)
        decoded = _decode(env, rates, event)
        line = detect_trajectory_radon(env, decoded.posterior, decoded.times)
        assert line.speed == pytest.approx(900.0, rel=0.2)
        result = replay_significance(
            env,
            decoded.posterior,
            decoded.times,
            score=weighted_correlation,
            n_shuffles=199,
            rng=0,
        )
        assert result.p_value < 0.01

    def test_random_events_are_rarely_significant(self, place_code):
        """Tirole et al.'s three nulls together keep alpha on random events."""
        env, rates = place_code
        shuffle_rng = np.random.default_rng(3)
        p_values = []
        for seed in range(80):
            event = simulate_replay_event(
                env, rates, kind="random", rate_gain=4.0, seed=100 + seed
            )
            counts, grid = bin_spikes_in_time(
                list(event.spike_times), dt=DT, t_start=0.0, t_stop=event.duration
            )
            posterior = decode_position(env, counts, rates, dt=DT, times=grid).posterior
            p_values.append(
                replay_significance(
                    env,
                    posterior,
                    grid,
                    score=weighted_correlation,
                    nulls=[
                        "spike_train_circular",
                        "place_field_circular",
                        "column_cycle",
                    ],
                    spike_counts=counts,
                    encoding_models=rates,
                    dt=DT,
                    n_shuffles=39,
                    rng=shuffle_rng,
                ).p_value
            )
        # 8 of 80 expected at p <= 0.1 (sd 2.7).
        assert np.sum(np.asarray(p_values) <= 0.1) <= 16

    def test_stationary_event_fools_the_line_fit_but_not_the_correlation(
        self, place_code
    ):
        """van der Meer et al. (2020): a stationary event scores high on a line fit."""
        env, rates = place_code
        event = simulate_replay_event(
            env, rates, kind="stationary", rate_gain=4.0, seed=4
        )
        decoded = _decode(env, rates, event)
        line = detect_trajectory_radon(env, decoded.posterior, decoded.times)
        correlation = weighted_correlation(env, decoded.posterior, decoded.times)
        assert line.score > 0.5
        assert abs(line.speed) < 200.0
        assert abs(correlation.score) < 0.5


class TestRefusals:
    def test_sweep_needs_a_speed(self, place_code):
        env, rates = place_code
        with pytest.raises(ValueError, match="needs a nonzero speed"):
            simulate_replay_event(env, rates)

    def test_sweep_must_stay_on_the_track(self, place_code):
        env, rates = place_code
        with pytest.raises(ValueError, match="leaves the track"):
            simulate_replay_event(env, rates, speed=2000.0)

    def test_rates_need_every_bin(self, place_code):
        env, rates = place_code
        bad = rates.copy()
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite and non-negative"):
            simulate_replay_event(env, bad, kind="stationary")

    def test_unknown_kind(self, place_code):
        env, rates = place_code
        with pytest.raises(ValueError, match="kind must be one of"):
            simulate_replay_event(env, rates, kind="theta")  # type: ignore[arg-type]

    def test_open_arena_is_refused(self):
        arena = Environment.from_samples(
            np.random.default_rng(0).uniform(0, 50, (500, 2)), bin_size=10.0
        )
        with pytest.raises(ValueError, match="linearized track"):
            simulate_replay_event(arena, np.ones((3, arena.n_bins)), kind="stationary")


def test_result_is_read_only_and_pickles(place_code):
    env, rates = place_code
    event = simulate_replay_event(env, rates, speed=600.0, seed=0)
    assert isinstance(event, ReplayEventSimulation)
    with pytest.raises(ValueError, match="read-only"):
        event.track_positions[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        event.spike_times[0][:1] = 1.0
    for clone in (pickle.loads(pickle.dumps(event)), copy.deepcopy(event)):
        np.testing.assert_array_equal(clone.track_positions, event.track_positions)
        assert not clone.track_positions.flags.writeable
        assert not any(train.flags.writeable for train in clone.spike_times)
