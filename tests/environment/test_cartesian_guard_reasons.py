"""Bound and unbound calls explain the same Cartesian-only capability."""

import inspect

import numpy as np
import pytest
from environment_helpers import call_unbound

from neurospatial import Environment
from neurospatial._exceptions import CartesianOnlyError
from neurospatial.environment.core import (
    CARTESIAN_GRID_OPERATIONS,
    CARTESIAN_OPERATIONS,
)

_SHARED = ("bin_centers", "n_bins", "neighbors", "path_between", "smooth")


def _polar(cls=Environment):
    return cls.from_polar_egocentric((0, 2), (-np.pi, np.pi), 1, np.pi / 2)


def _cartesian(cls=Environment):
    return cls.from_samples(np.zeros((4, 2)), bin_size=1.0)


@pytest.mark.parametrize("name", sorted(CARTESIAN_OPERATIONS))
def test_polar_bins_refuse_bound_and_unbound_calls_alike(name):
    env = _polar()
    assert not hasattr(env, name)
    with pytest.raises(CartesianOnlyError) as bound:
        getattr(env, name)
    with pytest.raises(CartesianOnlyError) as unbound:
        call_unbound(env, name)
    reason = (
        "Cartesian-grid operation"
        if name in CARTESIAN_GRID_OPERATIONS
        else "Cartesian-coordinate operation"
    )
    assert str(unbound.value) == str(bound.value)
    assert str(bound.value).startswith(f"Environment.{name}() is not available")
    assert f"({reason})" in str(bound.value)
    assert isinstance(bound.value, NotImplementedError)


def test_dir_hides_cartesian_only_members_on_polar_bins_only():
    polar_names = set(dir(_polar()))
    cartesian_names = set(dir(_cartesian()))
    assert not polar_names & CARTESIAN_OPERATIONS
    assert cartesian_names >= CARTESIAN_OPERATIONS
    assert cartesian_names - polar_names == CARTESIAN_OPERATIONS
    assert polar_names >= set(_SHARED)
    assert set(dir(Environment)) >= CARTESIAN_OPERATIONS


def test_cartesian_bins_keep_every_cartesian_only_member():
    env = _cartesian()
    assert all(hasattr(env, name) for name in CARTESIAN_OPERATIONS)
    np.testing.assert_array_equal(env.bin_at(env.bin_centers), np.arange(env.n_bins))
    np.testing.assert_array_equal(
        Environment.bin_at(env, env.bin_centers), np.arange(env.n_bins)
    )
    assert env.component_keys == ()


@pytest.mark.parametrize("name", sorted(CARTESIAN_OPERATIONS))
def test_guard_keeps_the_member_name_signature_and_docstring(name):
    member = inspect.getattr_static(Environment, name)
    function = member.fget if isinstance(member, property) else member
    assert function.__name__ == name
    assert function.__qualname__ == f"Environment.{name}"
    assert list(inspect.signature(function, follow_wrapped=False).parameters) == [
        "self",
        "args",
        "kwargs",
    ]
    assert function.__doc__ == function.__wrapped__.__doc__
    assert function.__doc__


def test_refusal_names_the_runtime_subclass():
    class RecordingRoom(Environment):
        pass

    env = _polar(RecordingRoom)
    with pytest.raises(CartesianOnlyError) as bound:
        env.bin_at(np.zeros((1, 2)))
    with pytest.raises(CartesianOnlyError) as unbound:
        Environment.bin_at(env, np.zeros((1, 2)))
    assert str(bound.value).startswith("RecordingRoom.bin_at() is not available")
    assert str(unbound.value) == str(bound.value)
    assert bound.value.environment_type == "RecordingRoom"
    assert not set(dir(env)) & CARTESIAN_OPERATIONS
    assert hasattr(_cartesian(RecordingRoom), "bin_at")
