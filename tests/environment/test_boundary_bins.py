"""Boundary bins follow cell adjacency through subsets and affine maps."""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.layout import create_layout
from neurospatial.ops.transforms import AffineND


def _rotation(angle: float) -> AffineND:
    c, s = np.cos(angle), np.sin(angle)
    return AffineND(np.array([[c, -s, 3.0], [s, c, -1.0], [0.0, 0.0, 1.0]]))


def _with_hole(layout: str) -> tuple[Environment, int, np.ndarray]:
    rng = np.random.default_rng(0)
    env = Environment.from_samples(
        rng.uniform(0, 30, (5000, 2)), bin_size=3.0, layout=layout
    )
    center = int(np.argmin(np.linalg.norm(env.bin_centers - 15.0, axis=1)))
    keep = np.ones(env.n_bins, dtype=bool)
    keep[center] = False
    return env, center, keep


def test_removing_an_interior_grid_bin_exposes_its_orthogonal_neighbors():
    env, center, keep = _with_hole("RegularGrid")
    assert center not in env.boundary_bins
    child_of = np.cumsum(keep) - 1
    offsets = np.abs(
        env.bin_centers[env.neighbors(bins=center)] - env.bin_centers[center]
    )
    orthogonal = np.array(env.neighbors(bins=center))[np.min(offsets, axis=1) < 1e-9]
    assert len(orthogonal) == 4
    expected = set(child_of[np.setdiff1d(env.boundary_bins, center)]) | set(
        child_of[orthogonal]
    )
    assert set(env.subset(mask=keep).boundary_bins.tolist()) == expected


@pytest.mark.parametrize("layout", ["RegularGrid", "Hexagonal"])
def test_boundary_bins_do_not_depend_on_an_affine_map(layout):
    env, _, keep = _with_hole(layout)
    subset = env.subset(mask=keep).boundary_bins
    assert len(subset) > len(env.boundary_bins) - 1
    np.testing.assert_array_equal(
        env.apply_transform(_rotation(0.7)).subset(mask=keep).boundary_bins, subset
    )
    np.testing.assert_array_equal(
        env.subset(mask=keep).apply_transform(_rotation(-0.3)).boundary_bins, subset
    )


@pytest.mark.parametrize("derive", ["subset", "native", "transformed"])
def test_removing_an_interior_hexagon_exposes_all_six_neighbors(derive):
    env, center, keep = _with_hole("Hexagonal")
    neighbors = np.array(env.neighbors(bins=center))
    assert len(neighbors) == 6
    assert center not in env.boundary_bins
    child_of = np.cumsum(keep) - 1
    expected = set(child_of[np.setdiff1d(env.boundary_bins, center)]) | set(
        child_of[neighbors]
    )
    if derive == "native":
        # A hexagonal layout built directly with the same hole.
        child = Environment.from_layout(
            create_layout(
                "Hexagonal",
                **{
                    **env.layout_parameters,
                    "positions": env.bin_centers[keep],
                    "infer_active_bins": True,
                    "bin_count_threshold": 0,
                },
            )
        )
        np.testing.assert_array_equal(child.bin_centers, env.bin_centers[keep])
    else:
        child = env.subset(mask=keep)
        if derive == "transformed":
            child = child.apply_transform(_rotation(0.7))
    assert set(child.boundary_bins.tolist()) == expected
