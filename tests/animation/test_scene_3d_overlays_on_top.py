"""In 3-D, the animal's overlays stay visible over the field's voxels."""

from __future__ import annotations

import warnings

import numpy as np

from neurospatial import Environment
from neurospatial.animation import Animation, PositionOverlay

TIMES = np.linspace(0.0, 1.0, 11)


def test_a_marker_inside_the_voxels_shows_its_color():
    # A solid 5x5x5 block of opaque cells; the animal sits at its center,
    # behind the voxels from every view.
    env = Environment.from_grid_mask(
        np.ones((5, 5, 5), dtype=bool), (np.arange(6.0) * 2,) * 3
    )
    positions = np.tile([5.0, 5.0, 5.0], (TIMES.size, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scene = Animation(
            env,
            np.ones((2, env.n_bins)),
            frame_times=TIMES[[0, -1]],
            overlays=[PositionOverlay(positions, TIMES, color="red", size=30)],
        )
    try:
        frame = scene.render_frame(0.5, dpi=60)
        ax = scene._renderer.figure.axes[0]
        from mpl_toolkits.mplot3d import proj3d

        x, y, _ = proj3d.proj_transform(5.0, 5.0, 5.0, ax.get_proj())
        column, row = ax.transData.transform((x, y))
        height = frame.shape[0]
        pixel = frame[round(height - row), round(column)] / 255.0
        np.testing.assert_allclose(pixel, [1.0, 0.0, 0.0], atol=0.1)
        assert not ax.computed_zorder
    finally:
        scene.close()
