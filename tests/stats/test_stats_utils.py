"""Random-stream helpers shared by seeded analyses."""

import numpy as np

from neurospatial.stats._utils import _entropy, _name_key, _stream


def test_seed_gives_its_own_entropy_and_reproducible_streams():
    assert _entropy(7) == np.random.SeedSequence(7).entropy
    first = _stream(_entropy(7), 1).random(3)
    assert np.array_equal(first, _stream(_entropy(7), 1).random(3))
    assert not np.array_equal(first, _stream(_entropy(7), 2).random(3))


def test_generator_contributes_exactly_one_draw():
    generator = np.random.default_rng(3)
    reference = np.random.default_rng(3)
    assert _entropy(generator) == int(reference.integers(2**63))
    assert generator.random() == reference.random()


def test_name_keys_are_stable_and_distinct():
    assert _name_key("place") == _name_key("place")
    assert _name_key("place") != _name_key("grid")


def test_encoding_shares_the_stats_entropy():
    """Encoding's shuffle seeds derive through the same helper."""
    from neurospatial.encoding import _significance

    assert _significance._entropy is _entropy
