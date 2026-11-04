"""A scene's spatial views frame positions tracked outside the environment."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import Animation, PositionOverlay
from neurospatial.animation._prepared import PreparedScene

TIMES = np.linspace(0.0, 1.0, 11)


def _scene(n_dims: int, perch: np.ndarray) -> Animation:
    """A box from 0 to 10 per axis; the animal is lost, then sits at ``perch``."""
    env = Environment.from_grid_mask(
        np.ones((5,) * n_dims, dtype=bool), (np.arange(6.0) * 2,) * n_dims
    )
    positions = np.tile(perch, (TIMES.size, 1))
    positions[:3] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # positions outside the environment
        return Animation(
            env,
            np.ones((2, env.n_bins)),
            frame_times=TIMES[[0, -1]],
            overlays=[PositionOverlay(positions, TIMES)],
        )


@pytest.mark.parametrize("n_dims", [2, 3])
def test_extent_covers_positions_outside_the_environment(n_dims):
    perch = np.r_[5.0, -30.0, 5.0][:n_dims]
    extent = PreparedScene(_scene(n_dims, perch)).spatial_extent()
    assert extent.shape == (n_dims, 2)
    assert extent[1, 0] == -30.0
    assert extent[1, 1] == 10.0
    np.testing.assert_array_equal(extent[0], [0.0, 10.0])


def test_extent_is_the_environment_when_positions_are_inside():
    # Negative control: nothing tracked outside, nothing widened.
    scene = _scene(2, np.r_[5.0, 5.0])
    np.testing.assert_array_equal(
        PreparedScene(scene).spatial_extent(), np.asarray(scene.env.dimension_ranges)
    )


def test_extent_counts_only_samples_on_the_scene_clock():
    env = Environment.from_grid_mask(
        np.ones((5, 5), dtype=bool), (np.arange(6.0) * 2,) * 2
    )
    times = np.linspace(0.0, 2.0, 21)
    positions = np.full((21, 2), 5.0)
    positions[-1] = [5.0, -30.0]  # after the scene's last frame
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scene = Animation(
            env,
            np.ones((2, env.n_bins)),
            frame_times=[0.0, 1.0],
            overlays=[PositionOverlay(positions, times)],
        )
    assert PreparedScene(scene).spatial_extent()[1, 0] == 0.0


@pytest.mark.parametrize("n_dims", [2, 3])
def test_matplotlib_view_covers_the_perch(n_dims):
    perch = np.r_[5.0, -30.0, 5.0][:n_dims]
    scene = _scene(n_dims, perch)
    try:
        scene.render_frame(0.5, dpi=30)
        ax = scene._renderer.figure.axes[0]
        limits = [ax.get_xlim(), ax.get_ylim()]
        if n_dims == 3:
            limits.append(ax.get_zlim())
        for value, (low, high) in zip(perch, limits, strict=True):
            assert min(low, high) <= value <= max(low, high)
    finally:
        scene.close()
