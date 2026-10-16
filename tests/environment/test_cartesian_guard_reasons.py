"""Bound and unbound calls explain the same Cartesian-only capability."""

import inspect

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial._exceptions import CartesianOnlyError
from neurospatial.environment.core import (
    CARTESIAN_GRID_OPERATIONS,
    CARTESIAN_OPERATIONS,
)


def _polar(cls=Environment):
    return cls.from_polar_egocentric((0, 2), (-np.pi, np.pi), 1, np.pi / 2)


def _cartesian(cls=Environment):
    return cls.from_samples(np.zeros((4, 2)), bin_size=1.0)


def _call_unbound(env, name):
    """Call a member through the class, passing None for required arguments."""
    member = inspect.getattr_static(Environment, name)
    if isinstance(member, property):
        return member.fget(env)
    args, kwargs = [env], {}
    for parameter in list(inspect.signature(member).parameters.values())[1:]:
        if parameter.default is not inspect.Parameter.empty:
            continue
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[parameter.name] = None
        else:
            args.append(None)
    return member(*args, **kwargs)


@pytest.mark.parametrize("operation", ["to_linear", "linear_to_nd", "rebin", "subset"])
def test_grid_operation_reason_matches_bound_and_unbound_calls(operation):
    env = _polar()
    arguments = {
        "to_linear": ((np.array([[1.0, 0.0]]),), {}),
        "linear_to_nd": ((np.array([1.0]),), {}),
        "rebin": ((1,), {}),
        "subset": ((), {"bins": np.ones(env.n_bins, dtype=bool)}),
    }
    assert not hasattr(env, operation)
    with pytest.raises(CartesianOnlyError) as bound:
        getattr(env, operation)
    args, kwargs = arguments[operation]
    with pytest.raises(CartesianOnlyError) as unbound:
        getattr(Environment, operation)(env, *args, **kwargs)
    assert "(Cartesian-grid operation)" in str(unbound.value)
    assert str(unbound.value) == str(bound.value)


@pytest.mark.parametrize("name", sorted(CARTESIAN_OPERATIONS))
def test_unbound_refusal_repeats_the_bound_message(name):
    env = _polar()
    with pytest.raises(CartesianOnlyError) as bound:
        getattr(env, name)
    with pytest.raises(CartesianOnlyError) as unbound:
        _call_unbound(env, name)
    reason = (
        "Cartesian-grid operation"
        if name in CARTESIAN_GRID_OPERATIONS
        else "Cartesian-coordinate operation"
    )
    assert str(unbound.value) == str(bound.value)
    assert str(bound.value).startswith(f"Environment.{name}() is not available")
    assert f"({reason})" in str(bound.value)


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
    assert inspect.signature(function) == inspect.signature(function.__wrapped__)
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
    assert hasattr(_cartesian(RecordingRoom), "bin_at")
