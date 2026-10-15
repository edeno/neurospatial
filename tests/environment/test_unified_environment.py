"""One public value preserves coordinate-specific refusals and physical math."""

from __future__ import annotations

import copy
import inspect
import pickle

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial._exceptions import CartesianOnlyError
from neurospatial.environment.cartesian import CARTESIAN_OPERATIONS
from neurospatial.geometry._coordinates import CartesianSystem, EgocentricPolarSystem


def _polar():
    return Environment.from_polar_egocentric(
        (0.0, 3.0), (-np.pi, np.pi), 1.0, np.pi / 2, units="cm"
    )


def test_one_runtime_class_with_owned_coordinate_meanings():
    cartesian = Environment.from_grid_mask(
        np.ones((2, 2), bool), grid_edges=(np.arange(3.0),) * 2
    )
    polar = _polar()
    assert type(polar) is type(cartesian) is Environment
    assert isinstance(cartesian._coordinate_system, CartesianSystem)
    assert isinstance(polar._coordinate_system, EgocentricPolarSystem)
    assert polar._coordinate_system.axis_labels(
        "cm"
    ) != cartesian._coordinate_system.axis_labels("cm")
    assert not np.array_equal(
        polar._coordinate_system.squared_distances(polar.bin_centers),
        cartesian._coordinate_system.squared_distances(polar.bin_centers),
    )


@pytest.mark.parametrize("name", sorted(CARTESIAN_OPERATIONS))
def test_bound_and_unbound_cartesian_operations_refuse_polar(name):
    polar = _polar()
    assert not hasattr(polar, name)
    with pytest.raises(CartesianOnlyError):
        getattr(polar, name)
    member = inspect.getattr_static(Environment, name)
    if isinstance(member, property):
        with pytest.raises(CartesianOnlyError):
            member.fget(polar)
        return
    signature = inspect.signature(member)
    args = [polar]
    kwargs = {}
    for parameter in list(signature.parameters.values())[1:]:
        if parameter.default is not inspect.Parameter.empty:
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            args.append(None)
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[parameter.name] = None
    with pytest.raises(CartesianOnlyError):
        member(*args, **kwargs)


@pytest.mark.parametrize(
    "restore",
    [
        copy.deepcopy,
        lambda x: pickle.loads(pickle.dumps(x)),
        lambda x: Environment.from_dict(x.to_dict()),
    ],
)
def test_common_class_restoration_keeps_polar_units_and_kernels(restore):
    env = _polar()
    result = restore(env)
    assert type(result) is Environment
    assert result.units == "cm"
    assert result._coordinate_system == env._coordinate_system
    assert not hasattr(result, "bin_at")
    np.testing.assert_array_equal(result.bin_sizes, env.bin_sizes)
    np.testing.assert_array_equal(result.compute_kernel(0.8), env.compute_kernel(0.8))


def test_old_polar_class_is_not_a_public_alias():
    import neurospatial.environment as public
    import neurospatial.environment.polar as adapter

    assert not hasattr(public, "EgocentricPolarEnvironment")
    assert not hasattr(adapter, "EgocentricPolarEnvironment")
