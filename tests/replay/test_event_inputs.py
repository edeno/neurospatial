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
    trajectory_metrics,
)
from neurospatial.replay.methods import (
    OLAFSDOTTIR_2017,
    PFEIFFER_FOSTER_2013,
    TIROLE_2022,
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


@pytest.fixture(scope="module")
def joint_with_a_vanishing_template(track):
    """A sweep whose "in" template is impossible in one time bin.

    Cell 0 is silent in the "in" maps and fires 40 spikes in bin 4, so the
    "in" template's probability there underflows to exactly 0: its share has
    no mass, while its conditional posterior is still a probability mass.
    """
    env, rates = track
    counts = _sweep_counts()
    counts[4, 0] = 40
    times = DT * (np.arange(len(counts)) + 0.5)
    silent = rates.copy()
    silent[0] = 0.0
    maps = {"out": rates, "in": silent}
    joint = decode_jointly(env, counts, maps, dt=DT, times=times)
    return env, maps, counts, times, joint


class TestVanishingTemplate:
    """A share that underflows to 0 is scored on the exact conditional posterior."""

    def test_the_share_row_is_empty_and_the_conditional_is_not(
        self, joint_with_a_vanishing_template
    ):
        *_, joint = joint_with_a_vanishing_template
        assert joint.joint("in")[4].sum() == 0.0
        np.testing.assert_allclose(joint.conditional["in"].posterior.sum(axis=1), 1.0)

    def test_posterior_nulls_score_the_conditional(
        self, joint_with_a_vanishing_template
    ):
        env, _maps, counts, times, joint = joint_with_a_vanishing_template
        method = dataclasses.replace(WU_2017, n_shuffles=50)
        share = detect_replay(
            env,
            joint,
            times,
            replay_method=method,
            spike_counts=counts,
            template="in",
            rng=0,
        )
        alone = detect_replay(
            env,
            joint.conditional["in"].posterior,
            times,
            replay_method=method,
            spike_counts=counts,
            rng=0,
        )
        assert share.score == pytest.approx(alone.score)
        assert share.p_value == alone.p_value

    def test_a_null_that_decodes_again_scores_the_conditional(
        self, joint_with_a_vanishing_template
    ):
        env, maps, counts, times, joint = joint_with_a_vanishing_template
        result = detect_replay(
            env,
            joint,
            times,
            replay_method=dataclasses.replace(OLAFSDOTTIR_2017, n_shuffles=30),
            spike_counts=counts,
            encoding_models=maps,
            dt=DT,
            template="in",
            rng=0,
        )
        assert np.isfinite(result.score)
        assert np.isfinite(result.p_value)

    def test_a_score_without_shuffles_scores_the_conditional(
        self, joint_with_a_vanishing_template
    ):
        env, _maps, _counts, times, joint = joint_with_a_vanishing_template
        method = ReplayMethod(
            name="fit_only",
            reference="test",
            score=fit_linear_trajectory,
            score_options={"method": "map"},
            criteria=(ReplayCriterion("score", ">", 0.5),),
        )
        share = detect_replay(env, joint, times, replay_method=method, template="in")
        alone = detect_replay(
            env, joint.conditional["in"].posterior, times, replay_method=method
        )
        assert share.score == pytest.approx(alone.score)

    def test_trajectory_quantities_read_positions_given_the_template(
        self, joint_with_a_vanishing_template
    ):
        env, maps, counts, times, joint = joint_with_a_vanishing_template
        method = dataclasses.replace(
            TIROLE_2022,
            n_shuffles=30,
            criteria=(ReplayCriterion("map_path_length", ">", 0.0),),
        )
        result = detect_replay(
            env,
            joint,
            times,
            replay_method=method,
            spike_counts=counts,
            encoding_models=maps,
            dt=DT,
            template="in",
            rng=0,
        )
        expected = trajectory_metrics(env, joint.conditional["in"].posterior, times)
        assert result.values["map_path_length > 0"] == pytest.approx(
            expected.map_path_length
        )


class TestTemplateWithoutMass:
    """A share that underflows to 0 in every time bin has no weighted correlation."""

    def test_the_event_gets_no_verdict(self, track):
        env, rates = track
        counts = _sweep_counts()
        counts[:, 0] = 40  # every bin: impossible on the "in" maps
        times = DT * (np.arange(len(counts)) + 0.5)
        silent = rates.copy()
        silent[0] = 0.0
        maps = {"out": rates, "in": silent}
        joint = decode_jointly(env, counts, maps, dt=DT, times=times)
        assert not joint.joint("in").any()
        result = detect_replay(
            env,
            joint,
            times,
            replay_method=dataclasses.replace(TIROLE_2022, n_shuffles=30),
            spike_counts=counts,
            encoding_models=maps,
            dt=DT,
            template="in",
            rng=0,
        )
        assert np.isnan(result.score)
        assert not result.is_replay

    def test_a_shuffle_without_mass_scores_nan(self, joint_sweep):
        from neurospatial.replay.significance import _WeightedCorrelationScorer

        env, _maps, _counts, times, joint = joint_sweep
        share = joint.joint("out")
        scorer = _WeightedCorrelationScorer(env, share, times, row_normalization="none")
        scores = scorer.scores([share, np.zeros_like(share), share], 3)
        assert np.isnan(scores[1])
        assert scores[0] == scores[2] == pytest.approx(scorer.observe().score)


def test_a_sampled_score_without_shuffles_follows_the_seed(track, monkeypatch):
    env, rates = track
    counts = _sweep_counts()
    times = DT * (np.arange(len(counts)) + 0.5)
    event = decode_position(env, counts, rates, dt=DT, times=times)
    method = ReplayMethod(
        name="sampled_fit",
        reference="test",
        score=fit_linear_trajectory,
        score_options={"method": "sample", "n_samples_per_time_bin": 20},
        criteria=(ReplayCriterion("score", ">", 0.5),),
    )
    scores = [
        detect_replay(env, event.posterior, times, replay_method=method, rng=seed).score
        for seed in (123, 123, 7)
    ]
    assert scores[0] == scores[1]
    assert scores[0] != scores[2]
    from neurospatial import _random

    first = compare_replay_methods(env, [event], replay_methods=[method], rng=None)
    # The recorded entropy, fed to the step every stream derives from.
    monkeypatch.setattr(_random, "_entropy", lambda rng: first.attrs["rng_entropy"])
    again = compare_replay_methods(env, [event], replay_methods=[method], rng=None)
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
