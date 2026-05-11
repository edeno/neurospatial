"""Tests for Environment._state_version and versioned_cached_property (M5.1)."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.environment.decorators import versioned_cached_property


@pytest.fixture
def env() -> Environment:
    rng = np.random.default_rng(0)
    return Environment.from_samples(rng.uniform(0, 100, (200, 2)), bin_size=5.0)


class TestStateVersionField:
    """Environment._state_version must increment on documented mutation paths."""


class TestVersionedCachedProperty:
    """Decorator behavior: caches per (instance, _state_version)."""

    def test_caches_within_same_version(self) -> None:
        calls = {"n": 0}

        class Stub:
            _state_version = 0

            @versioned_cached_property
            def value(self) -> int:
                calls["n"] += 1
                return 42

        stub = Stub()
        assert stub.value == 42
        assert stub.value == 42
        assert calls["n"] == 1

    def test_recomputes_after_version_bump(self) -> None:
        calls = {"n": 0}

        class Stub:
            _state_version = 0

            @versioned_cached_property
            def value(self) -> int:
                calls["n"] += 1
                return calls["n"]

        stub = Stub()
        assert stub.value == 1
        assert stub.value == 1  # still cached
        stub._state_version += 1
        assert stub.value == 2  # recomputed
        assert stub.value == 2  # cached at the new version

    def test_separate_instances_have_separate_caches(self) -> None:
        class Stub:
            _state_version = 0

            def __init__(self, v: int) -> None:
                self._v = v

            @versioned_cached_property
            def value(self) -> int:
                return self._v

        a = Stub(1)
        b = Stub(2)
        assert a.value == 1
        assert b.value == 2
        a._v = 99
        # No version bump → still cached at the old value.
        assert a.value == 1
        a._state_version = 1
        assert a.value == 99

    def test_double_assignment_in_class_raises(self) -> None:
        # Ensure the same descriptor cannot be assigned to two different
        # attribute names on the same class — matches the cached_property
        # invariant that names are stable.
        descriptor = versioned_cached_property(lambda self: 0)

        with pytest.raises(TypeError, match="two different names"):

            class Stub:
                _state_version = 0
                first = descriptor
                second = descriptor

            _ = Stub  # silence unused warning
