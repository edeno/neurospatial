"""Published replay recipes on simulated events: replay is found, chance is not.

Each recipe runs with fewer shuffles than the paper's, so the tests take
seconds; the recipes' published values are checked field by field in
``test_recipe_specs.py``.
"""

from __future__ import annotations

import dataclasses

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import bin_spikes_in_time, decode_jointly, decode_position
from neurospatial.replay import (
    detect_replay,
    detect_trajectory_radon,
    replay_significance,
)
from neurospatial.replay.methods import (
    BUSH_2022,
    FAROOQ_2019,
    HARVEY_2023,
    JADHAV_2016,
    LIU_2023,
    MUESSIG_2019,
    OLAFSDOTTIR_2015,
    OLAFSDOTTIR_2016,
    SHIN_2019,
    TANG_2017,
    WIKENHEISER_REDISH_2013,
    YAMAMOTO_TONEGAWA_2017,
)
from neurospatial.simulation import simulate_replay_event

DT = 0.02
N_CELLS = 40


@pytest.fixture(scope="module")
def place_code():
    """A 200 cm track in 4 cm bins with 40 place cells tiling it."""
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [200.0, 0.0]], bin_size=4.0, units="cm"
    )
    x = env.bin_centers[:, 0]
    rates = 0.5 + 25.0 * np.exp(
        -0.5 * ((x - np.linspace(0.0, 200.0, N_CELLS)[:, None]) / 10.0) ** 2
    )
    return env, rates


def _event(env, rates, kind, seed, **kwargs):
    """Spike counts of a simulated event in 20 ms bins and their times."""
    event = simulate_replay_event(
        env, rates, kind=kind, rate_gain=4.0, seed=seed, **kwargs
    )
    return bin_spikes_in_time(
        list(event.spike_times), dt=DT, t_start=0.0, t_stop=event.duration
    )


def _decoded(env, rates, kind, seed, **kwargs):
    counts, times = _event(env, rates, kind, seed, **kwargs)
    return decode_position(env, counts, rates, dt=DT, times=times), counts


def _sweep(env, rates, seed=1):
    return _decoded(env, rates, "sweep", seed=seed, speed=900.0)


def _random(env, rates, seed):
    return _decoded(env, rates, "random", seed=seed)


def _zigzag(rates):
    """Rate maps whose fields alternate between the track's ends in cell order.

    A sweep through the cells in order is no sequence on these maps, in
    either direction.
    """
    half = len(rates) // 2
    order = np.ravel(
        np.column_stack([np.arange(half), np.arange(len(rates) - 1, half - 1, -1)])
    )
    return rates[order]


def _detect(env, decoded, counts, method, rates, rng=0):
    return detect_replay(
        env,
        decoded.posterior,
        decoded.times,
        replay_method=method,
        spike_counts=counts,
        encoding_models=rates,
        dt=DT,
        rng=rng,
    )


def _one_hot(env, bins):
    return np.eye(env.n_bins)[list(bins)]


# --------------------------------------------------------------------------
# Line searches on a linear track
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    [OLAFSDOTTIR_2015, YAMAMOTO_TONEGAWA_2017, LIU_2023, BUSH_2022],
    ids=lambda m: m.name,
)
def test_line_searches_detect_a_sweep_and_reject_random_events(place_code, method):
    env, rates = place_code
    method = dataclasses.replace(method, n_shuffles=99)
    sweep, counts = _sweep(env, rates)
    result = _detect(env, sweep, counts, method, rates)
    assert result.is_replay, result.summary()
    assert result.p_value == pytest.approx(1 / 100)
    verdicts = [
        _detect(env, *_random(env, rates, seed=300 + seed), method, rates, rng=seed)
        for seed in range(4)
    ]
    assert sum(v.is_replay for v in verdicts) <= 1


@pytest.mark.parametrize(
    ("n_active", "expected"), [(6, False), (7, True)], ids=["6 cells", "7 cells"]
)
def test_olafsdottir_2015_needs_7_active_cells(place_code, n_active, expected):
    env, _ = place_code
    counts = np.zeros((20, N_CELLS), dtype=int)
    counts[np.arange(20), np.arange(20) % n_active] = 1
    result = detect_replay(
        env,
        _one_hot(env, range(20)),
        np.arange(20) * DT,
        replay_method=dataclasses.replace(OLAFSDOTTIR_2015, nulls=()),
        spike_counts=counts,
    )
    assert result.criteria["n_active_cells >= 7"] is expected


@pytest.mark.parametrize(
    ("n_active", "n_cells", "expected"),
    [(4, 20, False), (5, 40, False), (6, 40, True), (5, 20, True)],
)
def test_bush_needs_5_cells_and_15_percent(place_code, n_active, n_cells, expected):
    """At least 5 active cells or 15% of the cells, whichever is larger."""
    env, _ = place_code
    counts = np.zeros((20, n_cells), dtype=int)
    counts[np.arange(20), np.arange(20) % n_active] = 1
    result = detect_replay(
        env,
        _one_hot(env, range(20)),
        np.arange(20) * DT,
        replay_method=dataclasses.replace(BUSH_2022, nulls=(), null_options={}),
        spike_counts=counts,
    )
    assert all(result.criteria.values()) is expected


def test_bush_shuffles_only_the_active_cells(place_code):
    """The recipe's cell-identity shuffle permutes the event's active cells."""
    env, rates = place_code
    sweep, counts = _sweep(env, rates)
    assert counts.sum(axis=1).all(), "no empty time bins to drop"
    method = dataclasses.replace(BUSH_2022, n_shuffles=49)
    inputs = {"spike_counts": counts, "encoding_models": rates, "dt": DT}

    def tested(active_only):
        return replay_significance(
            env,
            sweep.posterior,
            sweep.times,
            score=detect_trajectory_radon,
            score_options=dict(BUSH_2022.score_options),
            nulls="unit_identity",
            null_options={"unit_identity": {"active_only": active_only}},
            n_shuffles=49,
            rng=0,
            **inputs,
        )

    result = detect_replay(
        env, sweep.posterior, sweep.times, replay_method=method, rng=0, **inputs
    )
    active, every_cell = tested(True), tested(False)
    assert result.p_value == active.p_value
    assert not np.array_equal(
        active.shuffles["unit_identity"].null_scores,
        every_cell.shuffles["unit_identity"].null_scores,
    )


# --------------------------------------------------------------------------
# Templates: the direction or trajectory with the smallest p
# --------------------------------------------------------------------------


def test_olafsdottir_2016_takes_the_direction_with_the_lower_p(place_code):
    env, rates = place_code
    maps = {"outbound": rates, "inbound": _zigzag(rates)}
    counts, times = _event(env, rates, "sweep", seed=1, speed=900.0)
    posteriors = {
        name: decode_position(env, counts, maps[name], dt=DT, times=times).posterior
        for name in maps
    }
    result = detect_replay(
        env,
        posteriors,
        times,
        replay_method=OLAFSDOTTIR_2016,
        spike_counts=counts,
        encoding_models=maps,
        dt=DT,
        rng=0,
    )
    assert result.template == "outbound"
    assert result.is_replay
    assert result.p_value == pytest.approx(1 / 101)


@pytest.mark.parametrize(
    "method", [JADHAV_2016, TANG_2017, SHIN_2019], ids=lambda m: m.name
)
def test_w_track_recipes_take_the_trajectory_with_the_smallest_p(place_code, method):
    """One of four trajectory templates holds the sweep; it is the one tested."""
    env, rates = place_code
    order = np.random.default_rng(3).permutation(N_CELLS)
    maps = {
        "center_to_left": _zigzag(rates),
        "left_to_center": rates[order],
        "center_to_right": rates,
        "right_to_center": _zigzag(rates[::-1]),
    }
    counts, times = _event(env, rates, "sweep", seed=2, speed=900.0)
    joint = decode_jointly(env, counts, maps, dt=DT, times=times)
    fast = dataclasses.replace(
        method,
        n_shuffles=99,
        score_options={"method": "sample", "n_samples_per_time_bin": 100},
    )
    result = detect_replay(
        env, joint, times, replay_method=fast, spike_counts=counts, rng=0
    )
    assert result.template == "center_to_right"
    assert result.is_replay
    assert result.p_value == pytest.approx(1 / 100)


# --------------------------------------------------------------------------
# Weighted correlations
# --------------------------------------------------------------------------


def test_farooq_tests_the_signed_correlation_in_both_tails(place_code):
    env, rates = place_code
    method = dataclasses.replace(FAROOQ_2019, n_shuffles=199)
    sweep, counts = _sweep(env, rates)
    forward = _detect(env, sweep, counts, method, rates)
    reverse = detect_replay(
        env,
        sweep.posterior[::-1],
        sweep.times,
        replay_method=method,
        spike_counts=counts[::-1],
        rng=0,
    )
    assert forward.score > 0.5 and reverse.score < -0.5
    assert forward.is_replay and reverse.is_replay
    # Two-sided: the smallest p-value is 2 / (n + 1).
    assert forward.p_value == pytest.approx(2 / 200)
    verdicts = [
        _detect(env, *_random(env, rates, seed=400 + seed), method, rates, rng=seed)
        for seed in range(4)
    ]
    assert sum(v.is_replay for v in verdicts) <= 1


def test_harvey_detects_forward_and_reverse_sweeps(place_code):
    env, rates = place_code
    method = dataclasses.replace(HARVEY_2023, n_shuffles=99)
    sweep, counts = _sweep(env, rates)
    forward = _detect(env, sweep, counts, method, rates)
    reverse = detect_replay(
        env,
        sweep.posterior[::-1],
        sweep.times,
        replay_method=method,
        spike_counts=counts[::-1],
        rng=0,
    )
    assert forward.is_replay and reverse.is_replay
    assert reverse.score < 0


@pytest.mark.parametrize(("n_empty", "expected"), [(5, True), (6, False)])
def test_harvey_keeps_events_with_at_most_half_their_bins_empty(
    place_code, n_empty, expected
):
    env, _ = place_code
    counts = np.zeros((10, N_CELLS), dtype=int)
    counts[np.arange(10), np.arange(10) % 5] = 1
    counts[:n_empty] = 0
    result = detect_replay(
        env,
        _one_hot(env, range(10)),
        np.arange(10) * DT,
        replay_method=dataclasses.replace(HARVEY_2023, nulls=()),
        spike_counts=counts,
    )
    assert result.criteria["empty_fraction <= 0.5"] is expected


# --------------------------------------------------------------------------
# Closed loops
# --------------------------------------------------------------------------


def _loop_rates(env, n_cells, width):
    """Place fields evenly around a loop, in the bins' order along it."""
    edges = env.binning.linearization.grid_edges[0]
    along = (edges[:-1] + edges[1:]) / 2
    length = float(edges[-1] - edges[0])
    centers = np.linspace(0.0, length, n_cells, endpoint=False)
    distance = np.abs(along[None, :] - centers[:, None])
    distance = np.minimum(distance, length - distance)
    return 0.5 + 25.0 * np.exp(-0.5 * (distance / width) ** 2)


def _seam_sweep_counts(n_cells, n_time, first_cell, dt, rng):
    """Two to three spikes per time bin from cells in loop order, across the seam."""
    counts = np.zeros((n_time, n_cells), dtype=np.int64)
    for t in range(n_time):
        cell = (first_cell + t) % n_cells
        counts[t, cell] = 2
        counts[t, (cell + 1) % n_cells] = rng.integers(0, 2)
    return counts, dt * (np.arange(n_time) + 0.5)


@pytest.fixture(scope="module")
def circular_track():
    """An 80 cm diameter circular track in 64 bins (Wikenheiser & Redish)."""
    n_nodes, radius = 16, 40.0
    graph = nx.cycle_graph(n_nodes)
    for node in graph.nodes:
        angle = 2 * np.pi * node / n_nodes
        graph.nodes[node]["pos"] = (radius * np.cos(angle), radius * np.sin(angle))
    env = Environment.from_graph(
        graph,
        edge_order=[(i, (i + 1) % n_nodes) for i in range(n_nodes)],
        bin_size=4.0,
        units="cm",
    )
    assert env.n_bins == 64
    return env, _loop_rates(env, n_cells=32, width=8.0)


class TestWikenheiserRedish:
    DT = 0.01

    def _detect(self, env, rates, counts, times, n_shuffles=199, rng=0):
        decoded = decode_position(env, counts, rates, dt=self.DT, times=times)
        return detect_replay(
            env,
            decoded.posterior,
            times,
            replay_method=dataclasses.replace(
                WIKENHEISER_REDISH_2013, n_shuffles=n_shuffles
            ),
            spike_counts=counts,
            rng=rng,
        )

    def test_a_sweep_across_the_seam_is_replay(self, circular_track):
        env, rates = circular_track
        counts, times = _seam_sweep_counts(
            32, 14, first_cell=24, dt=self.DT, rng=np.random.default_rng(0)
        )
        counts[5] = 0  # an empty time bin is dropped
        result = self._detect(env, rates, counts, times)
        assert result.is_replay, result.summary()
        assert result.p_value == pytest.approx(1 / 200)

    def test_random_events_are_not_replay(self, circular_track):
        env, rates = circular_track
        verdicts = []
        for seed in range(4):
            generator = np.random.default_rng(100 + seed)
            counts = np.zeros((14, 32), dtype=np.int64)
            counts[np.arange(14), generator.integers(0, 32, size=14)] = 2
            times = self.DT * (np.arange(14) + 0.5)
            verdicts.append(self._detect(env, rates, counts, times, rng=seed))
        assert sum(v.is_replay for v in verdicts) <= 1

    def test_needs_5_spikes_from_3_cells(self, circular_track):
        env, rates = circular_track
        counts = np.zeros((4, 32), dtype=np.int64)
        counts[[0, 1, 2, 3], [0, 1, 2, 2]] = 1  # 3 cells, 4 spikes
        result = self._detect(env, rates, counts, self.DT * np.arange(4), n_shuffles=9)
        assert result.criteria["n_active_cells >= 3"]
        assert not result.criteria["n_spikes >= 5"]
        assert not result.is_replay


@pytest.fixture(scope="module")
def square_track():
    """A square track of 1 m sides as a closed loop, 2.5 cm bins (Muessig)."""
    graph = nx.Graph()
    for node, position in enumerate([(0, 0), (100, 0), (100, 100), (0, 100)]):
        graph.add_node(node, pos=position)
    edge_order = [(0, 1), (1, 2), (2, 3), (3, 0)]
    graph.add_edges_from(edge_order)
    env = Environment.from_graph(graph, edge_order, bin_size=2.5, units="cm")
    clockwise = _loop_rates(env, n_cells=N_CELLS, width=10.0)
    return env, {"cw": clockwise, "ccw": _zigzag(clockwise)}


class TestMuessig:
    def test_the_sweeping_direction_is_replay_across_the_seam(self, square_track):
        env, maps = square_track
        # Cells 34 to 47 (mod 40): the sweep runs through the seam at 400 cm.
        counts, times = _seam_sweep_counts(
            N_CELLS, 14, first_cell=34, dt=DT, rng=np.random.default_rng(1)
        )
        posteriors = {
            name: decode_position(env, counts, rates, dt=DT, times=times).posterior
            for name, rates in maps.items()
        }
        method = dataclasses.replace(MUESSIG_2019, n_shuffles=99)
        inputs = {"spike_counts": counts, "encoding_models": maps, "dt": DT}
        result = detect_replay(
            env, posteriors, times, replay_method=method, rng=0, **inputs
        )
        assert result.template == "cw"
        assert result.is_replay
        assert result.p_value == pytest.approx(1 / 100)
        # Read as an open track, the sweep breaks where it crosses the seam.
        options = {**MUESSIG_2019.score_options, "wrap": False}
        open_track = dataclasses.replace(method, score_options=options)
        broken = detect_replay(
            env, posteriors, times, replay_method=open_track, rng=0, **inputs
        )
        assert broken.score < result.score - 0.1

    def test_stationary_lines_are_searched(self, square_track):
        """Events at one place fit a line of speed 0 (stationary events)."""
        env, maps = square_track
        times = np.arange(10) * DT
        counts = np.zeros((10, N_CELLS), dtype=np.int64)
        counts[:, [7, 8]] = 2
        posterior = decode_position(
            env, counts, maps["cw"], dt=DT, times=times
        ).posterior
        fit = detect_trajectory_radon(
            env, posterior, times, **MUESSIG_2019.score_options
        )
        assert abs(fit.speed) < 50.0
        slower = detect_trajectory_radon(
            env, posterior, times, **{**MUESSIG_2019.score_options, "min_speed": 500.0}
        )
        assert slower.score < fit.score
