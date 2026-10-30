"""Replay functions refuse an environment that differs from the decode's."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.decoding import DecodingResult, JointDecodingResult
from neurospatial.replay import detect_replay, event_properties
from neurospatial.replay.methods import SILVA_2015

DT = 0.02


@pytest.fixture(scope="module")
def track():
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=10.0, units="cm"
    )


@pytest.fixture(scope="module")
def shifted_track():
    """The same number of bins, 5 cm further along."""
    return Environment.linear_track(
        endpoints=[[5.0, 0.0], [105.0, 0.0]], bin_size=10.0, units="cm"
    )


@pytest.fixture(scope="module")
def grid():
    """A 2-D arena with as many bins as the track."""
    return Environment.from_polygon(box(0, 0, 50, 20), bin_size=10.0, units="cm")


def _sweep(env, n_time=8):
    rows = np.linspace(0, env.n_bins - 1, n_time).round().astype(int)
    posterior = np.full((n_time, env.n_bins), 0.01)
    posterior[np.arange(n_time), rows] = 1.0
    posterior /= posterior.sum(axis=1, keepdims=True)
    return DecodingResult(posterior=posterior, env=env, times=np.arange(n_time) * DT)


def _joint(env_a, env_b):
    a, b = _sweep(env_a), _sweep(env_b)
    return JointDecodingResult(
        template_names=("a", "b"),
        template_probability=np.full((len(a.times), 2), 0.5),
        conditional={"a": a, "b": b},
    )


def test_event_properties_refuses_another_environment(track, shifted_track):
    with pytest.raises(ValueError, match="was decoded on"):
        event_properties(shifted_track, [_sweep(track)])
    event_properties(track, [_sweep(track)])


def test_event_properties_checks_a_template_environment(track, shifted_track):
    joint = _joint(track, track)
    with pytest.raises(ValueError, match="was decoded on"):
        event_properties({"a": shifted_track, "b": track}, [joint], template="a")


def test_event_properties_checks_the_summed_environment(track, shifted_track):
    with pytest.raises(ValueError, match="was decoded on"):
        event_properties(shifted_track, [_joint(track, track)])


def test_a_joint_decode_on_another_kind_of_environment_is_named(track, grid):
    assert grid.n_bins == track.n_bins
    joint = _joint(track, track)
    with pytest.raises(ValueError, match="differs from the environment the joint"):
        detect_replay(
            grid,
            joint,
            np.asarray(joint.conditional["a"].times),
            replay_method=SILVA_2015,
            template="a",
        )
