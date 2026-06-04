"""Shared fixtures for environment subsystem tests."""

import numpy as np
import pytest

from neurospatial import Environment


@pytest.fixture
def holed_grid_env():
    """A 2D RegularGrid with an interior hole (inactive bins).

    Positions densely cover a 20x20 cm square EXCEPT a 6x6 cm hole in the
    middle, so ``from_samples`` infers inactive bins there. With bin_size=2.0
    this yields a 10x10 full grid (100 cells) but fewer active bins, so
    active-bin ids != full-grid flat indices -- the exact condition that
    exposes the occupancy/interpolate/rebin index bugs.
    """
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, 20.0, size=(4000, 2))
    # Punch a hole: drop samples in [7, 13] x [7, 13].
    hole = (
        (pts[:, 0] >= 7.0)
        & (pts[:, 0] <= 13.0)
        & (pts[:, 1] >= 7.0)
        & (pts[:, 1] <= 13.0)
    )
    pts = pts[~hole]
    env = Environment.from_samples(pts, bin_size=2.0, bin_count_threshold=1)
    assert env.n_bins < np.prod(env.layout.grid_shape), (
        "fixture must have inactive bins to exercise the index mapping"
    )
    return env


def _corner_holed_pts(seed):
    """Sample positions over a 20x20 square with a large corner block removed.

    Removing the [5, 20] x [5, 20] quadrant leaves fewer active bins than the
    full-grid flat index of the geometric center, so the old center-node
    degree probe (which indexed the active-keyed graph with a full-grid index)
    misses entirely and falls through to its diagonal-connectivity default --
    exactly the masked-grid failure the connectivity inference must avoid.
    """
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 20.0, size=(8000, 2))
    hole = (
        (pts[:, 0] >= 5.0)
        & (pts[:, 0] <= 20.0)
        & (pts[:, 1] >= 5.0)
        & (pts[:, 1] <= 20.0)
    )
    return pts[~hole]


@pytest.fixture
def holed_grid_env_8conn():
    """A holed 2D RegularGrid built with diagonal (8-connected) connectivity."""
    env = Environment.from_samples(
        _corner_holed_pts(1),
        bin_size=2.0,
        bin_count_threshold=1,
        connect_diagonal_neighbors=True,
    )
    assert env.n_bins < np.prod(env.layout.grid_shape)
    return env


@pytest.fixture
def holed_grid_env_4conn():
    """A holed 2D RegularGrid built without diagonal (4-connected) connectivity."""
    env = Environment.from_samples(
        _corner_holed_pts(2),
        bin_size=2.0,
        bin_count_threshold=1,
        connect_diagonal_neighbors=False,
    )
    assert env.n_bins < np.prod(env.layout.grid_shape)
    return env


@pytest.fixture
def full_grid_env():
    """A gap-free 2D RegularGrid (all bins active)."""
    rng = np.random.default_rng(3)
    pts = rng.uniform(0.0, 20.0, size=(4000, 2))
    return Environment.from_samples(pts, bin_size=2.0, bin_count_threshold=1)
