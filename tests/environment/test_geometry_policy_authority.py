"""Owned geometry, rather than descriptive labels, determines behavior."""

from dataclasses import replace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.binning import KnownFaces


@pytest.mark.parametrize("n_angle", [1, 2])
@pytest.mark.parametrize("circular", [False, True])
def test_stored_polar_arena_owns_small_grid_circularity(n_angle, circular):
    env = Environment.from_polar_egocentric(
        distance_range=(0.0, 3.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=1.0,
        angle_bin_size=2 * np.pi / n_angle,
        circular_angle=circular,
    )
    record = env.to_dict()
    record["layout_parameters"]["circular_angle"] = not circular
    restored = Environment.from_dict(record)
    assert restored._angular_is_circular is circular
    np.testing.assert_array_equal(restored.boundary_bins, env.boundary_bins)
    np.testing.assert_array_equal(restored.compute_kernel(0.7), env.compute_kernel(0.7))


def test_factor_rebin_depends_on_grid_geometry_instead_of_kind_label():
    source = Environment.from_samples(
        np.array([[0.0, 0.0], [4.0, 4.0]]),
        bin_size=1.0,
        infer_active_bins=False,
    )
    renamed = Environment.from_bins(
        replace(
            source._binning,
            kind="RecordedGrid",
            faces=KnownFaces(source._face_geometry),
        ),
        arena=source.arena,
    )
    expected = source.rebin(2)
    actual = renamed.rebin(2)
    np.testing.assert_array_equal(actual.bin_centers, expected.bin_centers)
    np.testing.assert_array_equal(actual.edge_pairs, expected.edge_pairs)
    np.testing.assert_array_equal(actual.bin_sizes, expected.bin_sizes)


@pytest.mark.parametrize("kind", ["Hexagonal", "Graph", "RegularGrid"])
def test_boundary_detection_uses_owned_context_not_descriptive_kind(kind):
    source = Environment.from_grid_mask(
        np.array([[True, True, True], [True, False, True], [True, True, True]]),
        grid_edges=(np.arange(4.0),) * 2,
    )
    renamed = Environment.from_bins(
        replace(source._binning, kind=kind, faces=KnownFaces(source._face_geometry)),
        arena=source.arena,
    )
    np.testing.assert_array_equal(renamed.boundary_bins, source.boundary_bins)
