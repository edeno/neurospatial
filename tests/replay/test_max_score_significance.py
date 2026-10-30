"""The largest score over separately decoded templates (Muessig et al. 2019)."""

from __future__ import annotations

import numpy as np
import pytest
from replay_helpers import template_stream, wrapped_bump

from neurospatial import Environment
from neurospatial.replay import nulls as replay_nulls
from neurospatial.replay import replay_significance, weighted_correlation

DT = 0.02
TIMES = DT * np.arange(15)
COLUMN_CYCLE = {"score": weighted_correlation, "nulls": "column_cycle"}


@pytest.fixture(scope="module")
def env():
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=5.0, units="cm"
    )


@pytest.fixture(scope="module")
def posteriors(env):
    """Two templates' posteriors whose rows are centered at random."""
    generator = np.random.default_rng(0)
    return {
        name: wrapped_bump(env, generator.uniform(0, 100, len(TIMES)), sigma=8.0)
        for name in ("cw", "ccw")
    }


def test_the_event_and_each_shuffle_take_the_largest_template(
    env, posteriors, monkeypatch
):
    drawn = []
    original = replay_nulls.shuffle_posterior_circular

    def spy(*args, **kwargs):
        shuffles = [np.array(shuffled) for shuffled in original(*args, **kwargs)]
        drawn.append(shuffles)
        yield from shuffles

    monkeypatch.setattr(replay_nulls, "shuffle_posterior_circular", spy)
    result = replay_significance(
        env, posteriors, TIMES, n_shuffles=50, rng=1, **COLUMN_CYCLE
    )
    assert len(drawn) == 2  # one set of shuffles per template
    per_template = np.array(
        [
            [abs(weighted_correlation(env, rows, TIMES).score) for rows in shuffles]
            for shuffles in drawn
        ]
    )
    np.testing.assert_allclose(
        result.shuffles["column_cycle"].null_scores, per_template.max(axis=0)
    )
    observed = {
        name: abs(weighted_correlation(env, rows, TIMES).score)
        for name, rows in posteriors.items()
    }
    best = max(observed, key=observed.__getitem__)
    assert result.template == best
    assert result.summary()["template"] == best
    assert result.observed.score == pytest.approx(
        weighted_correlation(env, posteriors[best], TIMES).score
    )
    assert result.shuffles["column_cycle"].observed_score == pytest.approx(
        observed[best]
    )


def test_identical_templates_are_shuffled_independently(env, posteriors):
    """Not yoked: the same posterior in two templates gets other shuffles."""
    rows = posteriors["cw"]
    both = replay_significance(
        env, {"a": rows, "b": rows}, TIMES, n_shuffles=50, rng=2, **COLUMN_CYCLE
    )
    one = replay_significance(
        env, {"a": rows}, TIMES, n_shuffles=50, rng=2, **COLUMN_CYCLE
    )
    maxed = both.shuffles["column_cycle"].null_scores
    single = one.shuffles["column_cycle"].null_scores
    assert np.all(maxed >= single) and np.any(maxed > single)


def test_each_template_draws_its_own_stream(env, posteriors):
    result = replay_significance(
        env, posteriors, TIMES, n_shuffles=50, rng=9, **COLUMN_CYCLE
    )
    alone = [
        replay_significance(
            env,
            posteriors[name],
            TIMES,
            n_shuffles=50,
            rng=template_stream(9, name),
            **COLUMN_CYCLE,
        )
        .shuffles["column_cycle"]
        .null_scores
        for name in ("cw", "ccw")
    ]
    np.testing.assert_array_equal(
        result.shuffles["column_cycle"].null_scores, np.maximum(*alone)
    )
    reordered = replay_significance(
        env,
        {"ccw": posteriors["ccw"], "cw": posteriors["cw"]},
        TIMES,
        n_shuffles=50,
        rng=9,
        **COLUMN_CYCLE,
    )
    assert reordered.p_value == result.p_value


def test_max_score_is_calibrated_on_random_events(env):
    """The event's largest score against the largest of each shuffle.

    Each template's posterior has rows centered at random, so every
    column-cycle shuffle is exchangeable with the event and the test's
    false-positive rate is P(k <= 8 of 99 shuffles) = 0.09 at alpha = 0.1.
    Comparing the event's largest score with the best template's own
    shuffles instead calls about 1 - 0.91**2 = 0.17 of these events replay.
    """
    alpha, n_events = 0.1, 300
    times = DT * np.arange(10)
    generator = np.random.default_rng(12)
    maxed, best_alone = [], []
    for event in range(n_events):
        event_posteriors = {
            name: wrapped_bump(env, generator.uniform(0, 100, 10), sigma=6.0)
            for name in ("cw", "ccw")
        }
        options = {**COLUMN_CYCLE, "n_shuffles": 99, "rng": event}
        maxed.append(
            replay_significance(env, event_posteriors, times, **options).p_value
        )
        alone = [
            replay_significance(env, rows, times, **options)
            for rows in event_posteriors.values()
        ]
        best = max(alone, key=lambda result: abs(result.observed.score))
        best_alone.append(best.p_value)
    maxed_rate = np.mean(np.asarray(maxed) < alpha)
    best_alone_rate = np.mean(np.asarray(best_alone) < alpha)
    assert abs(maxed_rate - 0.09) < 0.04
    assert best_alone_rate > 0.14


def test_refusals(env, posteriors):
    with pytest.raises(ValueError, match="alternative='greater'"):
        replay_significance(env, posteriors, TIMES, alternative="less", **COLUMN_CYCLE)
    with pytest.raises(ValueError, match="pseudo_event_pool"):
        replay_significance(
            env,
            posteriors,
            TIMES,
            pseudo_event_pool=posteriors["cw"],
            **COLUMN_CYCLE,
        )
    with pytest.raises(ValueError, match="template"):
        replay_significance(env, posteriors, TIMES, template="cw", **COLUMN_CYCLE)
    with pytest.raises(ValueError, match="env has keys"):
        replay_significance(env={"cw": env}, posterior=posteriors, times=TIMES)
    with pytest.raises(ValueError, match="different numbers of time bins"):
        replay_significance(
            env,
            {"cw": posteriors["cw"], "ccw": posteriors["ccw"][1:]},
            TIMES,
            **COLUMN_CYCLE,
        )
