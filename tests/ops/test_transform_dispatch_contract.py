"""Affine entry points retain their public outputs and error contracts."""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial._exceptions import CartesianOnlyError
from neurospatial.ops.transforms import (
    AffineND,
    apply_transform_to_environment,
    translate,
    translate_3d,
)


@pytest.mark.parametrize("n_dims", [2, 3])
def test_free_and_bound_transforms_preserve_the_same_geometry(n_dims):
    env = Environment.from_grid_mask(
        np.ones((2,) * n_dims, dtype=bool),
        grid_edges=(np.arange(3.0),) * n_dims,
        name="source",
        units="cm",
        frame="lab",
    )
    matrix = np.eye(n_dims + 1)
    matrix[:-1, -1] = np.arange(1.0, n_dims + 1)
    transform = AffineND(matrix)
    free = apply_transform_to_environment(
        env, transform, name="aligned", frame="session2"
    )
    bound = env.apply_transform(transform, name="aligned", frame="session2")

    for actual in (free, bound):
        np.testing.assert_array_equal(
            actual.bin_centers, env.bin_centers + matrix[:-1, -1]
        )
        np.testing.assert_array_equal(actual.edge_pairs, env.edge_pairs)
        np.testing.assert_array_equal(actual.bin_sizes, env.bin_sizes)
        np.testing.assert_array_equal(
            actual.bin_at(actual.bin_centers), np.arange(env.n_bins)
        )
        assert (actual.name, actual.units, actual.frame) == (
            "aligned",
            "cm",
            "session2",
        )
    np.testing.assert_array_equal(free.compute_kernel(0.5), bound.compute_kernel(0.5))


@pytest.mark.parametrize("entry", ["free", "bound"])
@pytest.mark.parametrize("invalid", ["dimension", "none", "singular"])
def test_transform_entry_points_preserve_invalid_transform_errors(entry, invalid):
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    if invalid == "dimension":
        transform = translate_3d(1, 2, 3)
        error_type, message = (
            ValueError,
            "Transform dimensionality (3D) does not match environment dimensionality (2D).",
        )
    elif invalid == "none":
        transform = None
        error_type, message = (
            AttributeError,
            "'NoneType' object has no attribute 'inverse'",
        )
    else:
        transform = AffineND(np.diag([0.0, 1.0, 1.0]))
        error_type, message = (
            ValueError,
            (
                "Environment transform must be invertible.\n"
                "Why: a singular transform collapses bins and cannot preserve containment.\n"
                "Fix: use nonzero scale factors and an invertible affine matrix."
            ),
        )
    with pytest.raises(error_type) as error:
        if entry == "free":
            apply_transform_to_environment(env, transform)
        else:
            env.apply_transform(transform)
    assert str(error.value) == message


def test_polar_entry_points_raise_the_same_refusal():
    env = Environment.from_polar_egocentric(
        distance_range=(0, 2),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=1,
        angle_bin_size=np.pi / 2,
    )
    transform = translate(1, 2)
    with pytest.raises(CartesianOnlyError, match=r"Environment.apply_transform\(\)"):
        apply_transform_to_environment(env, transform)
    with pytest.raises(CartesianOnlyError, match=r"Environment.apply_transform\(\)"):
        env.apply_transform(transform)
    with pytest.raises(CartesianOnlyError, match=r"Environment.apply_transform\(\)"):
        Environment.apply_transform(env, transform)
