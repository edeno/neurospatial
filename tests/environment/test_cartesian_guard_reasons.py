"""Cartesian-only members stay visible on polar bins and refuse when called."""

import inspect

import numpy as np
import pytest
from environment_helpers import call_unbound

from neurospatial import CartesianOnlyError, Environment
from neurospatial.environment.core import CARTESIAN_OPERATIONS

_SHARED = ("bin_centers", "n_bins", "neighbors", "path_between", "smooth")


def _polar(cls=Environment):
    return cls.from_polar_egocentric(
        distance_range=(0, 2),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=1,
        angle_bin_size=np.pi / 2,
    )


def _cartesian(cls=Environment):
    return cls.from_samples(np.zeros((4, 2)), bin_size=1.0)


@pytest.mark.parametrize("name", sorted(CARTESIAN_OPERATIONS))
def test_polar_bins_refuse_each_call_with_what_why_and_fix(name):
    env = _polar()
    assert hasattr(env, name)
    assert callable(getattr(env, name))
    with pytest.raises(CartesianOnlyError) as refused:
        call_unbound(env, name)
    message = str(refused.value)
    what, why, fix = message.split("\n")
    assert what == f"Environment.{name}() does not work on a polar environment."
    assert why.startswith("Why: ")
    assert "(distance, angle)" in why
    assert fix.startswith("Fix: ") and len(fix) > len("Fix: ")
    assert refused.value.operation == f"Environment.{name}()"
    assert isinstance(refused.value, TypeError)
    assert not isinstance(refused.value, (AttributeError, NotImplementedError))


def test_each_member_states_its_own_reason():
    env = _polar()
    reasons = {}
    for name in CARTESIAN_OPERATIONS:
        with pytest.raises(CartesianOnlyError) as refused:
            call_unbound(env, name)
        reasons[name] = str(refused.value).split("\n")[1:]
    # Members that refuse for the same reason share it; the rest differ.
    shared = {"bin_sequence", "bin_sequence_with_runs"}
    distinct = {tuple(reasons[name]) for name in CARTESIAN_OPERATIONS - shared}
    assert len(distinct) == len(CARTESIAN_OPERATIONS - shared)


def test_dir_lists_the_same_members_on_both_coordinate_kinds():
    polar_names = set(dir(_polar()))
    cartesian_names = set(dir(_cartesian()))
    assert polar_names >= CARTESIAN_OPERATIONS
    assert polar_names == cartesian_names
    assert polar_names >= set(_SHARED)


def test_cartesian_bins_keep_every_cartesian_only_member():
    env = _cartesian()
    assert all(hasattr(env, name) for name in CARTESIAN_OPERATIONS)
    np.testing.assert_array_equal(env.bin_at(env.bin_centers), np.arange(env.n_bins))
    np.testing.assert_array_equal(
        Environment.bin_at(env, env.bin_centers), np.arange(env.n_bins)
    )


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
    assert str(bound.value).startswith(
        "RecordingRoom.bin_at() does not work on a polar environment."
    )
    assert str(unbound.value) == str(bound.value)
    assert bound.value.operation == "RecordingRoom.bin_at()"
    assert hasattr(env, "bin_at")
