"""Environment.apply_transform keeps its geometry and error contracts."""

import numpy as np
import pytest

from neurospatial import CartesianOnlyError, Environment
from neurospatial.ops.transforms import AffineND, translate, translate_3d


@pytest.mark.parametrize("n_dims", [2, 3])
def test_bound_and_unbound_transforms_preserve_the_same_geometry(n_dims):
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
    bound = env.apply_transform(transform, name="aligned", frame="session2")
    unbound = Environment.apply_transform(
        env, transform, name="aligned", frame="session2"
    )

    for actual in (bound, unbound):
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
    np.testing.assert_array_equal(
        bound.compute_kernel(bandwidth=0.5), unbound.compute_kernel(bandwidth=0.5)
    )


@pytest.mark.parametrize("invalid", ["dimension", "none", "singular"])
def test_invalid_transforms_raise_actionable_errors(invalid):
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
            TypeError,
            "apply_transform() takes an AffineND or Affine2D transform, got NoneType.",
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
        env.apply_transform(transform)
    # The type refusal goes on to say why and how to wrap a matrix.
    shown = str(error.value).splitlines()[0] if invalid == "none" else str(error.value)
    assert shown == message


def test_polar_bound_and_unbound_calls_raise_the_same_refusal():
    env = Environment.from_polar_egocentric(
        distance_range=(0, 2),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=1,
        angle_bin_size=np.pi / 2,
    )
    transform = translate(1, 2)
    with pytest.raises(CartesianOnlyError) as bound:
        env.apply_transform(transform)
    with pytest.raises(CartesianOnlyError) as unbound:
        Environment.apply_transform(env, transform)
    assert str(bound.value) == str(unbound.value)
    assert bound.value.operation == "Environment.apply_transform()"
