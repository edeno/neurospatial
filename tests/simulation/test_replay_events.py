"""Tests for simulate_replay_event: simulated candidate events with known paths."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from scipy import stats

from neurospatial import Environment
from neurospatial.decoding import bin_spikes_in_time, decode_position
from neurospatial.replay import (
    detect_trajectory_radon,
    rank_order_correlation,
    rank_order_significance,
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


def _multiset_difference(larger, smaller):
    """Values of ``larger`` left after removing each value of ``smaller`` once."""
    values, counts = np.unique(larger, return_counts=True)
    removed = np.searchsorted(values, smaller)
    assert np.array_equal(values[removed], smaller), "smaller is not contained"
    np.subtract.at(counts, removed, 1)
    assert np.all(counts >= 0), "smaller is not contained"
    return np.repeat(values, counts)


class TestDegradation:
    """Removed rate-map spikes and added noise spikes (Tingley & Peyrache 2020)."""

    def test_no_degradation_leaves_the_event_unchanged(self, place_code):
        env, rates = place_code
        default = simulate_replay_event(env, rates, speed=600.0, seed=3)
        explicit = simulate_replay_event(
            env, rates, speed=600.0, drop_fraction=0.0, n_noise_spikes=0, seed=3
        )
        for a, b in zip(default.spike_times, explicit.spike_times, strict=True):
            np.testing.assert_array_equal(a, b)
        assert default.n_dropped_spikes == 0
        assert default.n_noise_spikes == 0

    @pytest.mark.parametrize("fraction", [0.25, 0.5, 1.0])
    def test_removes_that_fraction_of_the_rate_map_spikes(self, place_code, fraction):
        env, rates = place_code
        clean = simulate_replay_event(env, rates, speed=600.0, rate_gain=3.0, seed=4)
        dropped = simulate_replay_event(
            env, rates, speed=600.0, rate_gain=3.0, drop_fraction=fraction, seed=4
        )
        n_clean = sum(train.size for train in clean.spike_times)
        n_left = sum(train.size for train in dropped.spike_times)
        # Nearest whole spike, half rounding up.
        expected = int(np.floor(fraction * n_clean + 0.5))
        assert dropped.n_dropped_spikes == expected
        assert n_clean - n_left == expected
        # Every remaining spike is one of the cell's own rate-map spikes.
        for full, kept in zip(clean.spike_times, dropped.spike_times, strict=True):
            _multiset_difference(full, kept)

    def test_removal_draws_from_all_cells_together(self, place_code):
        """Each spike is equally likely to go, so cells lose spikes in proportion."""
        env, rates = place_code
        lost = np.zeros(len(rates))
        total = np.zeros(len(rates))
        for seed in range(40):
            clean = simulate_replay_event(
                env, rates, kind="random", rate_gain=4.0, seed=seed
            )
            half = simulate_replay_event(
                env, rates, kind="random", rate_gain=4.0, drop_fraction=0.5, seed=seed
            )
            sizes = np.array([t.size for t in clean.spike_times])
            lost += sizes - np.array([t.size for t in half.spike_times])
            total += sizes
        assert lost.sum() / total.sum() == pytest.approx(0.5, abs=0.01)
        # Cells with many spikes lose about half of them, not a fixed number.
        busy = total >= 60
        assert busy.sum() >= 10
        np.testing.assert_allclose(lost[busy] / total[busy], 0.5, atol=0.2)

    def test_adds_noise_spikes_uniformly_over_cells_and_time(self, place_code):
        env, rates = place_code
        clean = simulate_replay_event(env, rates, kind="stationary", seed=5)
        n_noise = 20_000
        noisy = simulate_replay_event(
            env, rates, kind="stationary", n_noise_spikes=n_noise, seed=5
        )
        assert noisy.n_noise_spikes == n_noise
        added = [
            _multiset_difference(full, base)
            for full, base in zip(noisy.spike_times, clean.spike_times, strict=True)
        ]
        per_cell = np.array([a.size for a in added])
        assert per_cell.sum() == n_noise
        # Uniform over the 40 cells (chi-square) and over the event (KS).
        assert stats.chisquare(per_cell).pvalue > 1e-3
        times = np.concatenate(added)
        assert times.min() >= 0.0
        assert times.max() < noisy.duration
        assert stats.kstest(times / noisy.duration, "uniform").pvalue > 1e-3
        for train in noisy.spike_times:
            assert np.all(np.diff(train) >= 0)

    def test_noise_spikes_do_not_depend_on_the_removal(self, place_code):
        """Removal acts on the rate-map spikes; the noise spikes are kept whole."""
        env, rates = place_code
        common = {"speed": 600.0, "rate_gain": 3.0, "seed": 6}
        clean = simulate_replay_event(env, rates, **common)
        noise_only = simulate_replay_event(env, rates, n_noise_spikes=50, **common)
        dropped = simulate_replay_event(env, rates, drop_fraction=0.6, **common)
        both = simulate_replay_event(
            env, rates, n_noise_spikes=50, drop_fraction=0.6, **common
        )
        n_clean = sum(t.size for t in clean.spike_times)
        assert both.n_dropped_spikes == int(np.floor(0.6 * n_clean + 0.5))
        for a, b, c, d in zip(
            noise_only.spike_times,
            clean.spike_times,
            both.spike_times,
            dropped.spike_times,
            strict=True,
        ):
            noise = _multiset_difference(a, b)
            np.testing.assert_array_equal(_multiset_difference(c, d), noise)

    def test_seed_repeats_the_degraded_event(self, place_code):
        env, rates = place_code
        kwargs = {"speed": 600.0, "drop_fraction": 0.3, "n_noise_spikes": 30}
        first = simulate_replay_event(env, rates, seed=8, **kwargs)
        second = simulate_replay_event(env, rates, seed=8, **kwargs)
        other = simulate_replay_event(env, rates, seed=9, **kwargs)
        for a, b in zip(first.spike_times, second.spike_times, strict=True):
            np.testing.assert_array_equal(a, b)
        assert any(
            a.size != b.size or not np.array_equal(a, b)
            for a, b in zip(first.spike_times, other.spike_times, strict=True)
        )

    def test_summary_reports_the_degradation(self, place_code):
        env, rates = place_code
        event = simulate_replay_event(
            env, rates, speed=600.0, drop_fraction=0.5, n_noise_spikes=7, seed=0
        )
        summary = event.summary()
        assert summary["drop_fraction"] == 0.5
        assert summary["n_noise_spikes"] == 7
        assert summary["n_dropped_spikes"] == event.n_dropped_spikes
        assert summary["n_spikes"] == sum(t.size for t in event.spike_times)

    def test_rank_order_detection_falls_as_noise_rises(self, place_code):
        """A sensitivity curve in the manner of Tingley & Peyrache's Fig. 4."""
        env, rates = place_code

        def detection_rate(**kwargs):
            hits = 0
            for seed in range(20):
                event = simulate_replay_event(
                    env, rates, speed=800.0, rate_gain=2.0, seed=seed, **kwargs
                )
                spikes, interval = list(event.spike_times), (0.0, event.duration)
                # Too few active cells to correlate is a missed event.
                if np.isnan(rank_order_correlation(env, spikes, interval, rates).score):
                    continue
                test = rank_order_significance(
                    env, spikes, interval, rates, n_shuffles=200, rng=seed
                )
                hits += test.p_value < 0.05
            return hits / 20

        by_noise = [detection_rate(n_noise_spikes=n) for n in (0, 100, 400)]
        assert by_noise[0] >= 0.9
        assert by_noise[0] > by_noise[1] > by_noise[2]
        assert by_noise[2] <= 0.5
        by_removal = [detection_rate(drop_fraction=f) for f in (0.0, 0.9)]
        assert by_removal[0] - by_removal[1] >= 0.4

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"drop_fraction": -0.1}, "drop_fraction must be a fraction"),
            ({"drop_fraction": 1.5}, "drop_fraction must be a fraction"),
            ({"drop_fraction": np.nan}, "drop_fraction must be a fraction"),
            ({"drop_fraction": True}, "drop_fraction must be a fraction"),
            ({"drop_fraction": "0.5"}, "drop_fraction must be a fraction"),
            ({"n_noise_spikes": -1}, "n_noise_spikes must be a non-negative"),
            ({"n_noise_spikes": 2.5}, "n_noise_spikes must be a non-negative"),
            ({"n_noise_spikes": True}, "n_noise_spikes must be a non-negative"),
        ],
    )
    def test_settings_out_of_range_are_refused(self, place_code, kwargs, match):
        env, rates = place_code
        with pytest.raises(ValueError, match=match):
            simulate_replay_event(env, rates, kind="stationary", **kwargs)


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
