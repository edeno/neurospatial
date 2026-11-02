"""The random lines of the Radon search follow ``rng``."""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.replay import detect_trajectory_radon


@pytest.fixture(scope="module")
def event():
    track = Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=2.0, units="cm"
    )
    posterior = np.random.default_rng(1).random((12, track.n_bins))
    posterior /= posterior.sum(axis=1, keepdims=True)
    return track, posterior, np.arange(12) * 0.02


def _fit(event, **options):
    track, posterior, times = event
    result = detect_trajectory_radon(
        track, posterior, times, band=10.0, max_speed=400.0, **options
    )
    return result.score, result.speed, result.start_position


def test_detect_trajectory_radon_default_is_deterministic(event):
    assert _fit(event, n_lines=200) == _fit(event, n_lines=200)
    assert _fit(event, n_lines=200) == _fit(event, n_lines=200, rng=0)


def test_an_int_rng_gives_the_lines_the_same_int_line_seed_gave_before(event):
    """Values recorded with ``line_seed`` before it became ``rng``."""
    np.testing.assert_allclose(
        _fit(event, n_lines=200, rng=3),
        (0.229486827665, 85.4711097581329, 61.47019966175609),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        _fit(event, n_lines=200),
        (0.226963148392, 71.942934125131, 64.63603994982017),
        rtol=1e-9,
    )


def test_a_generator_draws_the_lines_of_the_int_it_was_seeded_with(event):
    assert _fit(event, n_lines=200, rng=np.random.default_rng(3)) == _fit(
        event, n_lines=200, rng=3
    )


def test_rng_none_draws_new_lines_and_a_negative_seed_is_refused(event):
    assert _fit(event, n_lines=200, rng=None) != _fit(event, n_lines=200, rng=None)
    with pytest.raises(ValueError, match="rng"):
        _fit(event, n_lines=200, rng=-1)


def test_a_numpy_integer_rng_is_an_int_rng(event):
    """``line_seed=np.int64(3)`` worked before; ``rng`` keeps it working."""
    pinned = (0.229486827665, 85.4711097581329, 61.47019966175609)
    np.testing.assert_allclose(
        _fit(event, n_lines=200, rng=np.int64(3)), pinned, rtol=1e-9
    )


def test_a_wrapped_radon_score_keeps_its_own_fixed_lines():
    """``replay_significance`` derives ``rng`` only for scores that sample."""
    import functools

    from neurospatial.replay.significance import _takes_rng

    def sampled(env, posterior, times, *, rng=None):
        return None

    assert not _takes_rng(detect_trajectory_radon)
    assert not _takes_rng(functools.partial(detect_trajectory_radon, n_lines=200))
    assert _takes_rng(sampled)
    assert _takes_rng(functools.partial(sampled))
