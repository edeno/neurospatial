from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
from _record import Recording, alternating_blocks
from operations import env_from_graph, replay_compare_methods, turn_sequence


@pytest.fixture(scope="module")
def linear_track() -> Recording:
    """240 s of 40 place cells running laps on a 200 cm track (seeded)."""
    from neurospatial.simulation import linear_track_session

    session = linear_track_session(
        duration=240.0, track_length=200.0, bin_size=4.0, n_place_cells=40, rng=0
    )
    train, test = alternating_blocks(session.times[0], session.times[-1], 30.0)
    graph = nx.Graph()
    graph.add_node("start", pos=(0.0,))
    graph.add_node("end", pos=(200.0,))
    graph.add_edge("start", "end", distance=200.0)
    return Recording(
        dataset="simulated_track",
        session="rng0",
        source="neurospatial.simulation.linear_track_session",
        license="n/a",
        times=session.times,
        positions=session.positions,
        position_units="cm",
        spike_times=tuple(np.sort(train) for train in session.spike_times),
        unit_ids=np.arange(len(session.spike_times)),
        geometry={"graph": graph, "graph_root": "start"},
        params={
            "bin_size": 4.0,
            "dt": 0.25,
            "train_epochs": train,
            "test_epochs": test,
        },
    )


def test_graph_env_on_simulated_track(linear_track):
    outcome = env_from_graph(linear_track, {})
    assert outcome.status == "ok"
    assert outcome.summary["n_bins"] == 50  # 200 cm / 4 cm


def test_ops_na_without_graph_or_ripples(linear_track):
    no_graph = replace(linear_track, geometry={})
    assert env_from_graph(no_graph, {}).note == "geometry has no graph"
    assert turn_sequence(no_graph, {}).status == "n/a"
    outcome = replay_compare_methods(linear_track, {})
    assert outcome.status == "n/a"
    assert "ripples" in outcome.note


def _with_events(track: Recording, kinds: list[str]) -> Recording:
    """``track`` plus simulated replay events of ``kinds`` after the run, as ripples."""
    from neurospatial import Environment
    from neurospatial.encoding import compute_spatial_rates
    from neurospatial.simulation import simulate_replay_event

    env = Environment.from_samples(track.positions, bin_size=4.0, units="cm")
    rates = compute_spatial_rates(
        env, list(track.spike_times), track.times, track.positions
    )
    rng = np.random.default_rng(1)
    trains: list[list[np.ndarray]] = [[] for _ in track.spike_times]
    events = []
    for index, kind in enumerate(kinds):
        extra = (
            {"speed": float(rng.choice([-1, 1]) * rng.uniform(400, 800))}
            if kind == "sweep"
            else {}
        )
        event = simulate_replay_event(
            env, rates, kind=kind, duration=0.15, rate_gain=4.0, rng=index, **extra
        )
        start = 300.0 + 2.0 * index
        for cell, spikes in enumerate(event.spike_times):
            trains[cell].append(np.asarray(spikes) + start)
        events.append((start, start + event.duration))
    spikes = tuple(
        np.sort(np.concatenate([original, *extra_spikes]))
        for original, extra_spikes in zip(track.spike_times, trains, strict=True)
    )
    return replace(
        track,
        spike_times=spikes,
        events={"ripples": np.asarray(events)},
        params={**track.params, "replay_dt": 0.01, "replay_n_shuffles": 100},
    )


def test_replay_op_detects_simulated_sweeps(linear_track):
    # Sweeps are replay by construction. Stationary events can be replay too
    # (DAVIDSON_2009 includes zero-speed lines), so they are no negative control.
    outcome = replay_compare_methods(_with_events(linear_track, ["sweep"] * 12), {})
    assert outcome.status == "ok"
    assert outcome.summary["n_events"] == 12
    for method in ("davidson_2009", "grosmark_2016"):
        assert outcome.summary[f"{method}_rate"] > 0.5, method
