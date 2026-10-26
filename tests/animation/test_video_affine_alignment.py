"""Rendered video pixels coincide with their calibrated spatial coordinates."""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import Animation, AnimationField
from neurospatial.animation.overlays import VideoData
from neurospatial.ops.transforms import AffineND

TIMES = np.array([0.0, 0.1])


class CalibratedVideo:
    """Two black frames with a red marker, mapped to the arena by ``matrix``."""

    times = TIMES
    interp = "nearest"

    def __init__(self, matrix, markers):
        self.transform = AffineND(np.array(matrix))
        self.frames = np.zeros((2, 100, 120, 3), dtype=np.uint8)
        for index, (column, row) in enumerate(markers):
            self.frames[index, row - 2 : row + 3, column - 2 : column + 3, 0] = 255

    def convert_to_data(self, frame_times, n_frames, env):
        return VideoData(
            np.arange(2),
            self.frames,
            self.transform,
            (0.0, 150.0, 0.0, 150.0),
            1.0,
            "above",
            times=TIMES,
        )


@pytest.fixture
def env():
    edges = np.linspace(0.0, 150.0, 16)
    return Environment.from_grid_mask(np.ones((15, 15), dtype=bool), (edges, edges))


def _scene(env, overlays):
    field = AnimationField(
        np.zeros((2, env.n_bins)), TIMES, cmap="Greys", vmin=0.0, vmax=1.0
    )
    return Animation(env, {"field": field}, overlays=overlays)


@pytest.mark.parametrize(
    "matrix",
    [
        [[1.0, 0.0, -20.0], [0.0, -1.0, 110.0], [0.0, 0.0, 1.0]],
        [[0.8, 0.25, 5.0], [0.2, -0.9, 100.0], [0.0, 0.0, 1.0]],
    ],
    ids=["flip-and-translate", "rotate-and-shear"],
)
def test_raw_video_markers_land_at_full_affine_world_coordinates(env, matrix):
    markers = np.array([[45, 30], [65, 55]])
    video = CalibratedVideo(matrix, markers)
    with _scene(env, []) as plain:
        plain.render_frame(0.0, dpi=100)
        limits = (plain._renderer.figure.axes[0].get_xlim(),)
        limits += (plain._renderer.figure.axes[0].get_ylim(),)
    with _scene(env, [video]) as scene:
        for time, marker in zip(TIMES, markers, strict=True):
            pixels = scene.render_frame(time, dpi=100).astype(int)
            ax = scene._renderer.figure.axes[0]
            # The calibrated image keeps the axes in arena units.
            np.testing.assert_allclose((ax.get_xlim(), ax.get_ylim()), limits)
            red = (pixels[..., 0] > 200) & (pixels[..., 1] < 60) & (pixels[..., 2] < 60)
            rows, columns = np.nonzero(red)
            assert rows.size > 10
            world = video.transform(marker[None, :])[0]
            expected_x, expected_y = ax.transData.transform(world)
            np.testing.assert_allclose(
                [columns.mean(), rows.mean()],
                [expected_x, pixels.shape[0] - 1 - expected_y],
                atol=1.5,
            )
