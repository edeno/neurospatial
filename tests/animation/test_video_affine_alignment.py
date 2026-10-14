"""Rendered video pixels must coincide with calibrated spatial coordinates."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurospatial.animation._parallel import (
    VideoFrameRenderer,
    _render_video_background,
)
from neurospatial.animation.overlays import VideoData
from neurospatial.ops.transforms import Affine2D


@pytest.mark.parametrize("reuse", [False, True], ids=["export", "widget"])
@pytest.mark.parametrize(
    "matrix",
    [
        [[1.0, 0.0, -20.0], [0.0, -1.0, 110.0], [0.0, 0.0, 1.0]],
        [[0.8, 0.25, 5.0], [0.2, -0.9, 100.0], [0.0, 0.0, 1.0]],
    ],
    ids=["flip-and-translate", "rotate-and-shear"],
)
def test_raw_video_markers_land_at_full_affine_world_coordinates(matrix, reuse):
    markers = np.array([[45, 30], [65, 55]])
    frames = np.zeros((2, 100, 120, 3), dtype=np.uint8)
    for index, (column, row) in enumerate(markers):
        frames[index, row - 2 : row + 3, column - 2 : column + 3, 0] = 255
    transform = Affine2D(np.array(matrix))
    video = VideoData(
        frame_indices=np.arange(2),
        reader=frames,
        transform_to_env=transform,
        env_bounds=(-20.0, 130.0, -10.0, 140.0),
        alpha=1.0,
        z_order="above",
    )
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    renderer = VideoFrameRenderer(ax, video, None) if reuse else None
    try:
        artist = None
        for index, marker in enumerate(markers):
            if renderer is not None:
                renderer.render(ax, index)
                if artist is not None:
                    assert ax.images[0] is artist
                artist = ax.images[0]
            else:
                ax.clear()
                _render_video_background(ax, video, index)
            ax.set_xlim(-20, 130)
            ax.set_ylim(-10, 140)
            fig.canvas.draw()
            pixels = np.asarray(fig.canvas.buffer_rgba())
            red = (pixels[..., 0] > 220) & (pixels[..., 1] < 40) & (pixels[..., 2] < 40)
            rows, columns = np.nonzero(red)
            assert rows.size > 10
            world = transform(marker[None, :])[0]
            expected_x, expected_y = ax.transData.transform(world)
            np.testing.assert_allclose(
                [columns.mean(), rows.mean()],
                [expected_x, pixels.shape[0] - 1 - expected_y],
                atol=1.5,
            )
    finally:
        plt.close(fig)


@pytest.mark.parametrize("existing_field", [False, True])
def test_calibrated_video_keeps_axes_and_data_limits_in_world_units(existing_field):
    frame = np.zeros((1, 20, 30, 3), dtype=np.uint8)
    transform = Affine2D(
        np.array([[2.0, 0.0, 200.0], [0.0, -2.0, 500.0], [0.0, 0.0, 1.0]])
    )
    video = VideoData(
        frame_indices=np.arange(1),
        reader=frame,
        transform_to_env=transform,
        env_bounds=(190.0, 270.0, 450.0, 510.0),
        alpha=1.0,
        z_order="above",
    )
    fig, ax = plt.subplots()
    try:
        if existing_field:
            ax.plot([190, 270], [450, 510])
            previous = (ax.get_xlim(), ax.get_ylim())
        _render_video_background(ax, video, 0)
        assert ax.dataLim.xmin > 100
        assert ax.dataLim.ymin > 400
        if existing_field:
            np.testing.assert_array_equal((ax.get_xlim(), ax.get_ylim()), previous)
        else:
            np.testing.assert_allclose(ax.get_xlim(), [199, 259])
            np.testing.assert_allclose(ax.get_ylim(), [461, 501])
    finally:
        plt.close(fig)
