"""Joint-decode shares, seeds and environments of the events a method tests."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import decode_jointly, decode_position
from neurospatial.replay import (
    ReplayMethod,
    compare_replay_methods,
    detect_replay,
    fit_linear_trajectory,
)
from neurospatial.replay.methods import (
    OLAFSDOTTIR_2017,
    PFEIFFER_FOSTER_2013,
    WU_2017,
    ReplayCriterion,
)

DT = 0.02
N_CELLS = 20


@pytest.fixture(scope="module")
def track():
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=5.0, units="cm"
    )
    x = env.bin_centers[:, 0]
    centers = np.linspace(2.5, 97.5, N_CELLS)[:, None]
    rates = 0.5 + 20.0 * np.exp(-0.5 * ((x - centers) / 6.0) ** 2)
    return env, rates


def _sweep_counts(n_time=10):
    counts = np.zeros((n_time, N_CELLS), dtype=np.int64)
    for t in range(n_time):
        cell = int(t * N_CELLS / n_time)
        counts[t, cell] = 2
        counts[t, min(cell + 1, N_CELLS - 1)] = 1
    return counts


@pytest.fixture(scope="module")
def joint_sweep(track):
    """A sweep decoded jointly on two directions with the same maps."""
    env, rates = track
    counts = _sweep_counts()
    times = DT * (np.arange(len(counts)) + 0.5)
    maps = {"out": rates, "in": rates}
    joint = decode_jointly(env, counts, maps, dt=DT, times=times)
    return env, maps, counts, times, joint


class TestTemplateShares:
    """Scores that need rows summing to 1 see the posterior given the template."""

    def test_share_rows_do_not_sum_to_one(self, joint_sweep):
        *_, joint = joint_sweep
        np.testing.assert_allclose(joint.joint("out").sum(axis=1), 0.5)

    def test_a_regression_scores_the_conditional_posterior(self, joint_sweep):
        env, _maps, counts, times, joint = joint_sweep
        method = dataclasses.replace(WU_2017, n_shuffles=50)
        share = detect_replay(
            env,
            joint,
            times,
            replay_method=method,
            spike_counts=counts,
            template="out",
            rng=0,
        )
        alone = detect_replay(
            env,
            joint.conditional["out"].posterior,
            times,
            replay_method=method,
            spike_counts=counts,
            rng=0,
        )
        assert share.score == pytest.approx(alone.score)
        assert share.is_replay

    def test_a_null_that_decodes_again_conditions_its_shuffles(self, joint_sweep):
        env, maps, counts, times, joint = joint_sweep
        result = detect_replay(
            env,
            joint,
            times,
            replay_method=dataclasses.replace(OLAFSDOTTIR_2017, n_shuffles=30),
            spike_counts=counts,
            encoding_models=maps,
            dt=DT,
            template="out",
            rng=0,
        )
        assert np.isfinite(result.p_value)


def test_a_sampled_score_without_shuffles_follows_the_seed(track):
    env, rates = track
    counts = _sweep_counts()
    times = DT * (np.arange(len(counts)) + 0.5)
    event = decode_position(env, counts, rates, dt=DT, times=times)
    method = ReplayMethod(
        name="sampled_fit",
        reference="test",
        score=fit_linear_trajectory,
        score_options={"method": "sample", "n_samples": 200},
        criteria=(ReplayCriterion("score", ">", 0.5),),
    )
    scores = [
        detect_replay(env, event.posterior, times, replay_method=method, rng=seed).score
        for seed in (123, 123, 7)
    ]
    assert scores[0] == scores[1]
    assert scores[0] != scores[2]
    first = compare_replay_methods(env, [event], replay_methods=[method], rng=None)
    again = compare_replay_methods(
        env, [event], replay_methods=[method], rng=first.attrs["rng_entropy"]
    )
    assert first["score"].iloc[0] == again["score"].iloc[0]


def test_compare_refuses_an_event_decoded_on_another_environment(track):
    env, rates = track
    small = Environment.linear_track(
        endpoints=[[0.0, 0.0], [10.0, 0.0]], bin_size=0.5, units="cm"
    )
    assert small.n_bins == env.n_bins
    counts = _sweep_counts()
    times = DT * (np.arange(len(counts)) + 0.5)
    event = decode_position(env, counts, rates, dt=DT, times=times)
    with pytest.raises(ValueError, match="decoded on an environment that differs"):
        compare_replay_methods(
            small, [event], replay_methods=[PFEIFFER_FOSTER_2013], rng=0
        )
