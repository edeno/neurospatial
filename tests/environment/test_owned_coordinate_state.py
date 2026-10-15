"""Coordinates and lazy numerical geometry belong to the published value."""

from __future__ import annotations

import copy
import pickle
from dataclasses import fields, is_dataclass

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.environment._coordinates import EgocentricPolarSystem
from neurospatial.layout._native_faces import NativeFaces
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
        (0.0, 3.0), (-np.pi, np.pi), 1.0, 2 * np.pi / n_angle, circular_angle=circular
    )
    expected = env._coordinate_system
    assert isinstance(expected, EgocentricPolarSystem)
    # One angular bin preserves the option; three+ use the actual graph seams.
    assert expected.circular == circular
    env._layout_params_used["circular_angle"] = not circular
    assert env._coordinate_system is expected
    assert env._angular_is_circular == circular
    # Serialization writes normalized circularity rather than edited provenance.
    env._layout_params_used["circular_angle"] = circular
    restored = restore(env)
    assert restored._coordinate_system == expected
    np.testing.assert_array_equal(restored.compute_kernel(0.8), env.compute_kernel(0.8))


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
