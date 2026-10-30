"""Fernández-Ruiz-lab replay recipes on simulated events.

Sweeps are replay and random events are not. Each recipe runs with fewer
shuffles than the paper's; the published values are checked field by field
in ``test_recipe_specs.py``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import bin_spikes_in_time, decode_jointly, decode_position
from neurospatial.replay import compare_replay_methods, regression_line_score
from neurospatial.replay.methods import CHANG_2025, HARVEY_2023_LINE, SOULA_2023
from neurospatial.simulation import simulate_replay_event

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


def _counts(env, rates, kind, seed, dt, **kwargs):
    event = simulate_replay_event(
        env, rates, kind=kind, rate_gain=4.0, seed=seed, **kwargs
    )
    return bin_spikes_in_time(
        list(event.spike_times), dt=dt, t_start=0.0, t_stop=event.duration
    )


def _zigzag(rates):
    """Fields that alternate between the track's ends in cell order."""
    half = len(rates) // 2
    order = np.ravel(
        np.column_stack([np.arange(half), np.arange(len(rates) - 1, half - 1, -1)])
    )
    return rates[order]


KINDS = ["sweep", "sweep", "reverse", "random", "random", "random", "random"]


def _events(env, rates, dt, decode):
    events, counts = [], []
    for index, kind in enumerate(KINDS):
        simulated = "random" if kind == "random" else "sweep"
        speed = {"sweep": 900.0, "reverse": -900.0}.get(kind)
        event_counts, times = _counts(
            env,
            rates,
            simulated,
            seed=10 + index,
            dt=dt,
            **({} if speed is None else {"speed": speed}),
        )
        events.append(decode(event_counts, times))
        counts.append(event_counts)
    return events, counts


@pytest.mark.parametrize("method", [SOULA_2023, HARVEY_2023_LINE], ids=lambda m: m.name)
def test_regression_line_recipes_find_sweeps_and_reject_random_events(
    place_code, method
):
    env, rates = place_code
    dt = 0.02
    events, counts = _events(
        env,
        rates,
        dt,
        lambda c, t: decode_position(env, c, rates, dt=dt, times=t),
    )
    table = compare_replay_methods(
        env,
        events,
        replay_methods=[dataclasses.replace(method, n_shuffles=199)],
        spike_counts=counts,
        rng=0,
    )
    assert (table["error"] == "").all()
    verdicts = table.set_index("event")["is_replay"]
    kinds = np.array(KINDS)
    # Forward and reverse sweeps both hold their mass near one line.
    assert verdicts[kinds != "random"].all()
    assert verdicts[kinds == "random"].sum() <= 1
    sweep = events[0]
    expected = regression_line_score(
        env,
        np.where(counts[0].sum(axis=1)[:, None] > 0, sweep.posterior, 0.0),
        sweep.times,
        band=10.5,
    ).score
    assert table.loc[0, "score"] == pytest.approx(expected)
    assert table.loc[0, "p_value"] == pytest.approx(1 / 200)


def test_chang_tests_both_tails_and_keeps_the_trajectory_with_the_smallest_p(
    place_code,
):
    """The sweep runs on the left trajectory's fields; the right holds none."""
    env, rates = place_code
    dt = 0.01  # the paper's 10 ms bins
    maps = {"left": rates, "right": _zigzag(rates)}
    events, counts = _events(
        env,
        rates,
        dt,
        lambda c, t: decode_jointly(env, c, maps, dt=dt, times=t),
    )
    table = compare_replay_methods(
        env,
        events,
        replay_methods=[dataclasses.replace(CHANG_2025, n_shuffles=199)],
        spike_counts=counts,
        encoding_models=maps,
        dt=dt,
        rng=0,
    )
    assert (table["error"] == "").all()
    rows = table.set_index("event")
    kinds = np.array(KINDS)
    assert rows["is_replay"][kinds != "random"].all()
    assert (rows["template"][kinds != "random"] == "left").all()
    # A signed correlation: forward sweeps above every shuffle, reverse below.
    assert (rows["score"][kinds == "sweep"] > 0.5).all()
    assert (rows["score"][kinds == "reverse"] < -0.5).all()
    assert np.allclose(rows["p_value"][kinds != "random"], 2 / 200)
    assert rows["is_replay"][kinds == "random"].sum() <= 1


def test_chang_weighs_every_time_bin_the_same_as_the_code(place_code):
    """The code normalizes each time bin of a trajectory's slice of the joint
    posterior, which with a uniform prior is that trajectory's own decode;
    row_normalization="none" weighs each time bin by the trajectory's share."""
    env, rates = place_code
    dt = 0.01
    maps = {"left": rates, "right": _zigzag(rates)}
    counts, times = _counts(env, rates, "sweep", seed=3, dt=dt, speed=900.0)
    joint = decode_jointly(env, counts, maps, dt=dt, times=times)
    separate = {
        name: decode_position(env, counts, maps[name], dt=dt, times=times)
        for name in maps
    }
    recipe = dataclasses.replace(CHANG_2025, n_shuffles=49)
    by_share = dataclasses.replace(recipe, score_options={"row_normalization": "none"})
    common = {"spike_counts": [counts], "encoding_models": maps, "dt": dt, "rng": 0}
    on_joint = compare_replay_methods(env, [joint], replay_methods=[recipe], **common)
    on_separate = compare_replay_methods(
        env, [separate], replay_methods=[recipe], **common
    )
    shares = compare_replay_methods(env, [joint], replay_methods=[by_share], **common)
    assert on_joint.loc[0, "score"] == pytest.approx(
        on_separate.loc[0, "score"], abs=1e-9
    )
    assert shares.loc[0, "score"] != pytest.approx(on_joint.loc[0, "score"], abs=1e-6)
    # Independently: the weighted correlation of the chosen trajectory's
    # rows of the joint posterior, each time bin normalized to sum to 1.
    rows = joint.joint(on_joint.loc[0, "template"])
    spiking = counts.sum(axis=1) > 0
    weights = rows[spiking] / rows[spiking].sum(axis=1, keepdims=True)
    t = np.broadcast_to(times[spiking, None], weights.shape)
    x = np.broadcast_to(env.bin_centers[:, 0], weights.shape)
    mean_t, mean_x = np.average(t, weights=weights), np.average(x, weights=weights)
    cov = np.average((t - mean_t) * (x - mean_x), weights=weights)
    var_t = np.average((t - mean_t) ** 2, weights=weights)
    var_x = np.average((x - mean_x) ** 2, weights=weights)
    assert on_joint.loc[0, "score"] == pytest.approx(
        cov / np.sqrt(var_t * var_x), abs=1e-9
    )
