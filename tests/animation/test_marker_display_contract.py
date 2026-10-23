"""Screen-sized overlays stay independent of units, zoom and export DPI."""

import numpy as np
import pytest
from matplotlib.collections import PathCollection

from neurospatial import Environment
from neurospatial.animation.overlays import PositionOverlay
from neurospatial.animation.scene import Animation


@pytest.mark.parametrize("n_dims", [2, 3])
@pytest.mark.parametrize("coordinate_scale", [0.01, 1000.0])
@pytest.mark.parametrize("dpi", [50, 150])
def test_scene_position_diameter_is_pixels(n_dims, coordinate_scale, dpi):
    env = Environment.from_grid_mask(
        np.ones((2,) * n_dims, dtype=bool),
        tuple(np.arange(3) * coordinate_scale for _ in range(n_dims)),
    )
    with Animation(
        env,
        np.ones((1, env.n_bins)),
        frame_times=[0.0],
        overlays=[PositionOverlay(np.full((1, n_dims), coordinate_scale), size=14.0)],
    ) as scene:
        scene.render_frame(0.0, dpi=dpi)
        ax = scene._renderer.figure.axes[0]
        marker = next(
            item for item in ax.collections if isinstance(item, PathCollection)
        )
        diameter_pixels = np.sqrt(marker.get_sizes()) * dpi / 72.0
        np.testing.assert_allclose(diameter_pixels, [14.0])
