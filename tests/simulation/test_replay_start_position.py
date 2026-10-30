"""simulate_replay_event: start positions on a gapped track, and refusals."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.simulation import simulate_replay_event


@pytest.fixture(scope="module")
def gapped_track():
    """Two 100 cm segments, (0, 0)-(100, 0) and (100, 0)-(100, 100).

    Linearized with a 30 cm gap: the first segment covers track positions
    0-100, the second 130-230.
    """
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
    return env, np.ones((5, env.n_bins))


def test_a_start_on_the_second_segment_is_where_it_says(gapped_track):
    env, rates = gapped_track
    event = simulate_replay_event(
        env, rates, kind="stationary", start_position=150.0, seed=0
    )
    np.testing.assert_allclose(event.track_positions, 150.0)
    # 20 cm along the second segment, which runs from (100, 0) to (100, 100).
    np.testing.assert_allclose(event.positions, np.tile([100.0, 20.0], (200, 1)))


def test_a_start_in_the_gap_is_refused(gapped_track):
    env, rates = gapped_track
    with pytest.raises(ValueError, match="not inside any bin"):
        simulate_replay_event(env, rates, kind="stationary", start_position=115.0)


def test_a_backward_sweep_crosses_the_gap_without_entering_it(gapped_track):
    """From 150 back at 900 cm/s for 50 ms: 45 cm along the track.

    20 cm reach the second segment's start at 130; the remaining 25 cm
    continue from the first segment's end at 100, down to about 75.
    """
    env, rates = gapped_track
    event = simulate_replay_event(
        env, rates, speed=-900.0, start_position=150.0, duration=0.05, seed=0
    )
    positions = event.track_positions
    assert positions[0] == pytest.approx(150.0)
    assert not np.any((positions >= 100.0) & (positions < 130.0))
    assert positions.min() == pytest.approx(150.0 - 30.0 - 900.0 * 0.049)
    assert np.all(np.diff(positions) < 0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"duration": 0.0}, "duration must be positive"),
        ({"duration": -0.1}, "duration must be positive"),
        ({"rate_gain": 0.0}, "rate_gain must be positive"),
        ({"random_interval": 0.0}, "random_interval must be positive"),
        ({"kind": "jump", "jump_distance": 0.0}, "jump_distance must be positive"),
        ({"kind": "jump", "jump_distance": -5.0}, "jump_distance must be positive"),
    ],
)
def test_refusals(gapped_track, kwargs, match):
    env, rates = gapped_track
    options = {"speed": 100.0, **kwargs}
    with pytest.raises(ValueError, match=match):
        simulate_replay_event(env, rates, **options)


def test_negative_rates_are_refused(gapped_track):
    env, rates = gapped_track
    bad = rates.copy()
    bad[2, 3] = -1.0
    with pytest.raises(ValueError, match="finite and non-negative"):
        simulate_replay_event(env, bad, kind="stationary")
