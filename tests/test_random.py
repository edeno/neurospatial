"""Random-stream helpers shared by seeded analyses."""

import numpy as np

from neurospatial._random import _name_key, _recorded_seed, _root, _stream


def _first_draw(rng):
    return int(np.random.default_rng(rng).integers(2**63))


def test_seed_and_its_fresh_generator_give_one_root():
    """``rng=7`` and a fresh ``default_rng(7)`` root the same streams."""
    seed_root = _root(7)
    generator_root = _root(np.random.default_rng(7))
    for root in (seed_root, generator_root):
        assert root.entropy == _first_draw(7)
        assert root.spawn_key == ()
    assert np.array_equal(
        _stream(seed_root, 1).random(3), _stream(generator_root, 1).random(3)
    )


def test_streams_are_reproducible_and_keyed():
    first = _stream(_root(7), 1).random(3)
    assert np.array_equal(first, _stream(_root(7), 1).random(3))
    assert not np.array_equal(first, _stream(_root(7), 2).random(3))
    assert not np.array_equal(first, _stream(_root(7), 1, 0).random(3))


def test_generator_state_sets_the_root_and_one_draw_advances_it():
    generator = np.random.default_rng(3)
    first, second = _root(generator), _root(generator)
    assert first.entropy != second.entropy
    assert not np.array_equal(_stream(first, 1).random(3), _stream(second, 1).random(3))
    # The root took exactly one draw from the generator each time.
    reference = np.random.default_rng(3)
    reference.integers(2**63, size=2)
    assert generator.random() == reference.random()


def test_generators_in_one_state_give_one_root():
    """The state counts, not the seed the generator was built from."""
    restored = np.random.default_rng(123)
    restored.bit_generator.state = np.random.default_rng(0).bit_generator.state
    assert _root(restored).entropy == _root(0).entropy
    jumped = np.random.Generator(np.random.PCG64(0).jumped())
    twin = np.random.default_rng(5)
    twin.bit_generator.state = jumped.bit_generator.state
    assert _root(jumped).entropy == _root(twin).entropy != _root(0).entropy
    advanced = np.random.default_rng(0)
    advanced.random(1000)
    assert _root(advanced).entropy != _root(0).entropy


def test_recorded_seed_is_the_root_entropy():
    """The record is an int for every input and rebuilds every stream."""
    for rng in (7, None, np.random.default_rng(7)):
        root = _root(rng)
        recorded = _recorded_seed(root)
        assert isinstance(recorded, int)
        rebuilt = np.random.default_rng(
            np.random.SeedSequence(recorded, spawn_key=(4, 2))
        )
        assert np.array_equal(rebuilt.random(3), _stream(root, 4, 2).random(3))


def test_name_keys_are_stable_and_distinct():
    assert _name_key("place") == _name_key("place")
    assert _name_key("place") != _name_key("grid")


def test_encoding_shares_the_package_root():
    """Encoding's shuffle seeds derive through the same helper."""
    from neurospatial.encoding import _significance

    assert _significance._root is _root
