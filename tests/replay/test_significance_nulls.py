"""What each null of replay_significance draws, and how the nulls combine."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import decode_jointly, decode_position
from neurospatial.replay import replay_significance

DT = 0.02
ALL_NULLS = (
    "column_cycle",
    "unit_identity",
    "pseudo_event",
    "spike_train_circular",
    "place_field_circular",
    "time_bin_permutation",
    "spike_time_uniform",
    "column_cycle_weighted",
)


@pytest.fixture(scope="module")
def place_code():
    """12 place cells on a 100 cm track in 10 cm bins."""
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=10.0, units="cm"
    )
    x = env.bin_centers[:, 0]
    rates = 0.5 + 30.0 * np.exp(
        -0.5 * ((x - np.linspace(0.0, 100.0, 12)[:, None]) / 10.0) ** 2
    )
    return env, rates


@pytest.fixture(scope="module")
def sweep(place_code):
    """Spike counts of a forward sweep, with some time bins without spikes."""
    _, rates = place_code
    rng = np.random.default_rng(5)
    n_time = 10
    counts = np.zeros((n_time, len(rates)), dtype=np.int64)
    for row in range(n_time):
        if row in (2, 6):
            continue
        cell = round(row * (len(rates) - 1) / (n_time - 1))
        counts[row, cell] = 1 + rng.poisson(1.0)
    return counts, np.arange(n_time) * DT


class _Recorder:
    """A score that keeps every posterior it is given."""

    def __init__(self):
        self.posteriors = []

    def __call__(self, env, posterior, times):
        posterior = np.asarray(posterior, dtype=np.float64)
        self.posteriors.append(posterior.copy())
        along = env.bin_centers[:, 0]
        return SimpleNamespace(score=float(np.sum(posterior @ along * times)))


@pytest.mark.parametrize("prior_kind", ["default", "non_uniform"])
def test_place_field_shuffles_keep_unsupported_bins_empty(
    place_code, sweep, prior_kind
):
    """Shifted maps decode with the event's support and prior.

    Bins 0-2 have no training data; the prior also excludes bins 5-6 in the
    non-uniform case. Shuffles must give neither any mass, also in time bins
    without spikes, where the likelihood is flat over the zero-filled maps.
    """
    env, rates = place_code
    counts, times = sweep
    partial = rates.copy()
    partial[:, :3] = np.nan
    if prior_kind == "default":
        prior, excluded = None, [0, 1, 2]
    else:
        prior = np.linspace(1.0, 3.0, env.n_bins)
        prior[[5, 6]] = 0.0
        excluded = [0, 1, 2, 5, 6]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        posterior = decode_position(env, counts, partial, dt=DT, prior=prior).posterior
        recorder = _Recorder()
        replay_significance(
            env,
            posterior,
            times,
            score=recorder,
            nulls="place_field_circular",
            spike_counts=counts,
            encoding_models=partial,
            dt=DT,
            prior=prior,
            n_shuffles=20,
            rng=0,
        )
    observed, *shuffled = recorder.posteriors
    assert len(shuffled) == 20
    np.testing.assert_array_equal(observed[:, excluded], 0.0)
    for decoded in shuffled:
        np.testing.assert_allclose(decoded.sum(axis=1), 1.0)
        np.testing.assert_array_equal(decoded[:, excluded], 0.0)
    # The shuffles do move the decoded positions.
    assert not all(np.allclose(decoded, observed) for decoded in shuffled)


def test_the_event_p_value_is_the_largest_over_the_nulls(place_code):
    """A stationary event: permuting its time bins changes nothing.

    Every time-bin permutation ties with the event (p = 1), while the
    column cycle scatters the posterior and scores lower (small p).
    """
    env, _ = place_code
    posterior = np.zeros((10, env.n_bins))
    posterior[:, 4] = 1.0
    times = np.arange(10) * DT
    both = replay_significance(
        env,
        posterior,
        times,
        nulls=["column_cycle", "time_bin_permutation"],
        n_shuffles=49,
        rng=0,
    )
    assert both.shuffles["time_bin_permutation"].p_value == 1.0
    assert both.shuffles["column_cycle"].p_value < 0.1
    assert both.p_value == 1.0


@pytest.fixture(scope="module")
def every_null_input(place_code, sweep):
    env, rates = place_code
    counts, times = sweep
    posterior = decode_position(env, counts, rates, dt=DT).posterior
    pool = np.random.default_rng(1).dirichlet(np.ones(env.n_bins), size=30)
    return (
        env,
        posterior,
        times,
        {
            "spike_counts": counts,
            "encoding_models": rates,
            "dt": DT,
            "pseudo_event_pool": pool,
        },
    )


def _null_scores(inputs, nulls, rng):
    env, posterior, times, kwargs = inputs
    result = replay_significance(
        env, posterior, times, nulls=nulls, n_shuffles=15, rng=rng, **kwargs
    )
    return {name: test.null_scores for name, test in result.shuffles.items()}


@pytest.mark.parametrize("null", ALL_NULLS)
def test_a_seed_reproduces_each_null(every_null_input, null):
    first = _null_scores(every_null_input, [null], rng=3)[null]
    again = _null_scores(every_null_input, [null], rng=3)[null]
    other = _null_scores(every_null_input, [null], rng=4)[null]
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_the_order_of_the_nulls_does_not_change_their_shuffles(every_null_input):
    forward = _null_scores(every_null_input, list(ALL_NULLS), rng=3)
    backward = _null_scores(every_null_input, list(ALL_NULLS)[::-1], rng=3)
    assert set(forward) == set(ALL_NULLS)
    for name in ALL_NULLS:
        np.testing.assert_array_equal(forward[name], backward[name])


def test_pseudo_events_draw_pool_rows_with_replacement(place_code):
    env, _ = place_code
    # Three distinct pool rows, none of them a row of the event.
    pool = np.eye(env.n_bins)[[7, 8, 9]]
    posterior = np.eye(env.n_bins)[np.arange(6)]
    recorder = _Recorder()
    replay_significance(
        env,
        posterior,
        np.arange(6) * DT,
        score=recorder,
        nulls="pseudo_event",
        pseudo_event_pool=pool,
        n_shuffles=30,
        rng=0,
    )
    _, *shuffled = recorder.posteriors
    assert len(shuffled) == 30
    drawn = np.concatenate([decoded.argmax(axis=1) for decoded in shuffled])
    for decoded in shuffled:
        assert decoded.shape == posterior.shape
        assert set(decoded.argmax(axis=1)) <= {7, 8, 9}
        np.testing.assert_array_equal(decoded.max(axis=1), 1.0)
    # Six rows from a pool of three repeat rows, and every pool row is drawn.
    assert set(drawn) == {7, 8, 9}


def test_joint_place_field_shuffles_shift_each_template_independently(
    place_code, sweep, monkeypatch
):
    """With the same maps in both templates, a shared shift would keep them equal."""
    import neurospatial.decoding as decoding

    env, rates = place_code
    counts, times = sweep
    models = {"a": rates, "b": rates.copy()}
    joint = decode_jointly(env, counts, models, dt=DT)
    seen = []
    original = decoding.decode_jointly

    def spy(env, counts, models, **kwargs):
        seen.append({name: np.asarray(maps) for name, maps in models.items()})
        return original(env, counts, models, **kwargs)

    monkeypatch.setattr(decoding, "decode_jointly", spy)
    replay_significance(
        env,
        joint.joint("a"),
        times,
        score=_Recorder(),
        nulls="place_field_circular",
        spike_counts=counts,
        encoding_models=models,
        dt=DT,
        template="a",
        n_shuffles=10,
        rng=0,
    )
    shuffled = [maps for maps in seen if maps["a"] is not rates]
    assert len(shuffled) >= 10
    shuffled = shuffled[-10:]
    for maps in shuffled:
        assert not np.allclose(maps["a"], rates)
    assert sum(not np.allclose(maps["a"], maps["b"]) for maps in shuffled) == 10
