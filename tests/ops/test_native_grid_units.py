"""Physical Cartesian face geometry is independent of coordinate units."""

import numpy as np
import pytest

from neurospatial import Environment


@pytest.mark.parametrize("scale", [1e-12, 1e-10, 1e-8, 1.0])
def test_native_grid_faces_and_diffusion_scale_with_coordinate_units(scale):
    reference = Environment.from_grid_mask(
        np.ones((2, 1), dtype=bool), grid_edges=(np.arange(3.0), np.arange(2.0))
    )
    scaled = Environment.from_grid_mask(
        np.ones((2, 1), dtype=bool),
        grid_edges=(np.arange(3.0) * scale, np.arange(2.0) * scale),
    )
    faces = scaled._face_geometry
    np.testing.assert_allclose(faces.volumes, scale**2, rtol=1e-14, atol=0)
    assert faces.A[0] == pytest.approx(scale, rel=1e-14, abs=0)
    assert faces.distance[0] == pytest.approx(scale, rel=1e-14, abs=0)
    np.testing.assert_allclose(
        scaled.compute_kernel(0.5 * scale, mode="average"),
        reference.compute_kernel(0.5, mode="average"),
        rtol=1e-13,
        atol=1e-14,
    )


def test_tiny_axis_does_not_turn_diagonal_contacts_into_physical_faces():
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool),
        grid_edges=(np.arange(3.0) * 1e-12, np.arange(3.0)),
        connect_diagonal_neighbors=True,
    )
    faces = env._face_geometry
    areas = {
        frozenset(pair): area for pair, area in zip(faces.edges, faces.A, strict=True)
    }
    assert areas[frozenset((0, 1))] == 1e-12  # y-neighbor: face length in x
    assert areas[frozenset((0, 2))] == 1.0  # x-neighbor: face length in y
    assert areas[frozenset((0, 3))] == 0.0  # a corner has no physical interface
