"""Random streams of compare_replay_methods: one per (event, method) pair."""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

import neurospatial.replay.methods as methods_module
from neurospatial import Environment
from neurospatial.decoding import DecodingResult
from neurospatial.replay import compare_replay_methods
from neurospatial.replay.methods import SILVA_2015

DT = 0.02


@pytest.fixture(scope="module")
def two_identical_events():
    """Two copies of one noisy forward sweep: only their streams differ."""
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=10.0, units="cm"
    )
    rng = np.random.default_rng(0)
    posterior = np.eye(env.n_bins)[np.arange(env.n_bins)] + rng.uniform(
        0.0, 0.3, (env.n_bins, env.n_bins)
    )
    posterior /= posterior.sum(axis=1, keepdims=True)
    times = np.arange(env.n_bins) * DT
    event = DecodingResult(posterior=posterior, env=env, times=times)
    return env, [event, copy.deepcopy(event)]


def _cycle(name):
    return dataclasses.replace(
        SILVA_2015, name=name, nulls=("column_cycle",), n_shuffles=19
    )


def _draws(monkeypatch, env, events, methods, rng):
    """The first draws of the stream each (event, method) pair is given."""
    draws = {}
    detect = methods_module.detect_replay

    def spy(*args, **kwargs):
        stream = copy.deepcopy(kwargs["rng"])
        draws[len(draws)] = tuple(stream.random(4))
        return detect(*args, **kwargs)

    monkeypatch.setattr(methods_module, "detect_replay", spy)
    table = compare_replay_methods(env, events, replay_methods=methods, rng=rng)
    monkeypatch.undo()
    keys = list(zip(table["event"], table["method"], strict=True))
    return {key: draws[i] for i, key in enumerate(keys)}, table


def test_every_event_and_method_pair_draws_its_own_stream(
    monkeypatch, two_identical_events
):
    env, events = two_identical_events
    draws, _ = _draws(monkeypatch, env, events, [_cycle("a"), _cycle("b")], rng=0)
    assert len(draws) == 4
    # Same event, two methods; two events, same method; and across both.
    assert len(set(draws.values())) == 4


def test_a_generator_seeds_the_streams(monkeypatch, two_identical_events):
    env, events = two_identical_events
    methods = [_cycle("a"), _cycle("b")]
    first, table = _draws(monkeypatch, env, events, methods, np.random.default_rng(0))
    again, table_again = _draws(
        monkeypatch, env, events, methods, np.random.default_rng(0)
    )
    other, _ = _draws(monkeypatch, env, events, methods, np.random.default_rng(1))
    assert first == again
    assert table.equals(table_again)
    assert all(first[key] != other[key] for key in first)


def test_adding_a_method_keeps_the_other_rows(monkeypatch, two_identical_events):
    env, events = two_identical_events
    alone, alone_table = _draws(monkeypatch, env, events, [_cycle("a")], rng=0)
    both, both_table = _draws(
        monkeypatch, env, events, [_cycle("b"), _cycle("a")], rng=0
    )
    assert all(both[key] == alone[key] for key in alone)
    kept = both_table[both_table["method"] == "a"].reset_index(drop=True)
    assert kept[alone_table.columns].equals(alone_table)
