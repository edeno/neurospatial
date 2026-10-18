"""One public value preserves coordinate-specific refusals and physical math."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from neurospatial import CartesianOnlyError, Environment
from neurospatial.geometry._coordinates import CartesianSystem, EgocentricPolarSystem


def _polar():
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 3.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=1.0,
        angle_bin_size=np.pi / 2,
        units="cm",
    )


def test_one_runtime_class_with_owned_coordinate_meanings():
    cartesian = Environment.from_grid_mask(
        np.ones((2, 2), bool), grid_edges=(np.arange(3.0),) * 2
    )
    polar = _polar()
    assert type(polar) is type(cartesian) is Environment
    assert isinstance(cartesian.coordinate_system, CartesianSystem)
    assert isinstance(polar.coordinate_system, EgocentricPolarSystem)
    assert polar.coordinate_system.axis_labels(
        "cm"
    ) != cartesian.coordinate_system.axis_labels("cm")
    assert not np.array_equal(
        polar.coordinate_system.squared_distances(polar.bin_centers),
        cartesian.coordinate_system.squared_distances(polar.bin_centers),
    )


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
    assert result.coordinate_system == env.coordinate_system
    with pytest.raises(CartesianOnlyError):
        result.interpolate(np.ones(result.n_bins), result.bin_centers)
    np.testing.assert_array_equal(
        result.bin_at(result.bin_centers), np.arange(env.n_bins)
    )
    np.testing.assert_array_equal(result.bin_sizes, env.bin_sizes)
    np.testing.assert_array_equal(
        result.compute_kernel(bandwidth=0.8, mode="density"),
        env.compute_kernel(bandwidth=0.8, mode="density"),
    )


def test_old_polar_class_is_not_a_public_alias():
    import neurospatial.environment as public
    import neurospatial.environment._polar as adapter

    assert not hasattr(public, "EgocentricPolarEnvironment")
    assert not hasattr(adapter, "EgocentricPolarEnvironment")
