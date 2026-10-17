"""Coordinates and lazy numerical geometry belong to the published value."""

from __future__ import annotations

import copy
import pickle
from dataclasses import fields, is_dataclass

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.geometry._coordinates import EgocentricPolarSystem
from neurospatial.geometry._native_faces import NativeFaces
from neurospatial.layout.base import LayoutEngine


@pytest.mark.parametrize("n_angle", [1, 2, 8])
@pytest.mark.parametrize("circular", [False, True])
@pytest.mark.parametrize(
    "restore",
    [
        copy.deepcopy,
        lambda x: pickle.loads(pickle.dumps(x)),
        lambda x: Environment.from_dict(x.to_dict()),
    ],
)
def test_polar_coordinate_state_is_owned(n_angle, circular, restore):
    env = Environment.from_polar_egocentric(
        distance_range=(0.0, 3.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=1.0,
        angle_bin_size=2 * np.pi / n_angle,
        circular_angle=circular,
    )
    expected = env._coordinate_system
    assert isinstance(expected, EgocentricPolarSystem)
    assert expected.circular == circular
    assert env._coordinate_system is expected
    assert env._angular_is_circular == circular
    restored = restore(env)
    assert restored._coordinate_system == expected
    np.testing.assert_array_equal(
        restored.compute_kernel(bandwidth=0.8), env.compute_kernel(bandwidth=0.8)
    )


def test_native_face_recipe_contains_only_owned_data(built_environment):
    source = built_environment._binning.faces
    assert isinstance(source, NativeFaces)
    seen = set()

    def check(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        assert not isinstance(value, LayoutEngine)
        assert not callable(value)
        if is_dataclass(value) and not isinstance(value, type):
            for item in fields(value):
                check(getattr(value, item.name))
        elif isinstance(value, (tuple, list)):
            for item in value:
                check(item)

    check(source)
    assert not hasattr(source, "layout")
    for restored in (copy.deepcopy(source), pickle.loads(pickle.dumps(source))):
        check(restored)
        for actual, expected in zip(restored.build(), source.build(), strict=True):
            np.testing.assert_array_equal(actual, expected)
            assert not actual.flags.writeable


@pytest.mark.parametrize(
    "restore",
    [
        copy.copy,
        copy.deepcopy,
        lambda x: pickle.loads(pickle.dumps(x)),
        lambda x: Environment.from_dict(x.to_dict()),
    ],
)
def test_environment_state_contains_no_builder_or_delegated_callback(
    built_environment, restore
):
    from collections.abc import Mapping

    import networkx as nx

    env = restore(built_environment)
    assert not hasattr(env, "layout")
    seen = set()

    def check(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        assert not isinstance(value, LayoutEngine)
        assert not callable(value)
        if isinstance(value, Environment):
            check(vars(value))
        elif is_dataclass(value) and not isinstance(value, type):
            for item in fields(value):
                check(getattr(value, item.name))
        elif isinstance(value, Mapping):
            for item in value.values():
                check(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                check(item)
        elif isinstance(value, nx.Graph):
            check(value.graph)
            check(dict(value.nodes(data=True)))
            check(list(value.edges(data=True)))

    check(env)


def test_direct_layout_constructor_is_removed():
    with pytest.raises(ValueError, match="E1006"):
        Environment()
    with pytest.raises(TypeError, match="layout"):
        Environment(layout=object())
