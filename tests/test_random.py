"""Random-stream helpers shared by seeded analyses."""

import numpy as np

from neurospatial._random import _name_key, _recorded_seed, _root, _stream


def test_seed_and_its_fresh_generator_give_one_root():
    """``rng=7`` and a fresh ``default_rng(7)`` root the same streams."""
    seed_root = _root(7)
    generator_root = _root(np.random.default_rng(7))
    expected = np.random.SeedSequence(7).spawn(1)[0]
    for root in (seed_root, generator_root):
        assert root.entropy == expected.entropy
        assert root.spawn_key == expected.spawn_key
    assert np.array_equal(
        _stream(seed_root, 1).random(3), _stream(generator_root, 1).random(3)
    )


def test_streams_are_reproducible_and_keyed():
    first = _stream(_root(7), 1).random(3)
    assert np.array_equal(first, _stream(_root(7), 1).random(3))
    assert not np.array_equal(first, _stream(_root(7), 2).random(3))


def test_generator_spawns_a_new_root_on_each_call_without_drawing():
    generator = np.random.default_rng(3)
    first, second = _root(generator), _root(generator)
    assert first.spawn_key != second.spawn_key
    assert not np.array_equal(_stream(first, 1).random(3), _stream(second, 1).random(3))
    # Spawning leaves the generator's own draws where they were.
    assert generator.random() == np.random.default_rng(3).random()


def test_recorded_seed_reproduces_the_root():
    for rng in (7, None):
        root = _root(rng)
        again = _root(_recorded_seed(rng, root))
        assert (again.entropy, again.spawn_key) == (root.entropy, root.spawn_key)
    generator = np.random.default_rng(7)
    assert _recorded_seed(generator, _root(generator)) is None


def test_name_keys_are_stable_and_distinct():
    assert _name_key("place") == _name_key("place")
    assert _name_key("place") != _name_key("grid")


def test_encoding_shares_the_package_root():
    """Encoding's shuffle seeds derive through the same helper."""
    from neurospatial.encoding import _significance

    assert _significance._root is _root
