"""Views never draw an animal where it was not observed.

Trails do not connect samples across a recording break, a missing keypoint
hides its marker and skeleton edges, and a heading arrow with no observed
anchor is not drawn.
"""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import Animation, AnimationField
from neurospatial.animation.overlays import (
    BodypartOverlay,
    HeadDirectionOverlay,
    PositionOverlay,
)
from neurospatial.animation.skeleton import Skeleton

TIMES = np.arange(6) * 0.1


@pytest.fixture
def env():
    return Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool), (np.arange(11.0), np.arange(11.0))
    )


@pytest.fixture
def napari_view(env):
    pytest.importorskip("napari")
    views = []

    def open_(overlays):
        view = env.animate_fields(
            np.ones((len(TIMES), env.n_bins)),
            frame_times=TIMES,
            overlays=overlays,
            backend="napari",
            show=False,
        )
        views.append(view)
        return view

    yield open_
    for view in views:
        view.close()


@pytest.mark.napari
@pytest.mark.xdist_group(name="napari_gui")
def test_napari_trail_breaks_at_missing_samples_and_recording_breaks(napari_view):
    path = np.column_stack([np.arange(6.0) + 2, np.full(6, 5.0)])
    path[2] = np.nan  # tracking lost
    breaks = np.zeros(6, dtype=bool)
    breaks[4] = True  # a recording break between samples 3 and 4
    view = napari_view(
        [PositionOverlay(path, TIMES, trail_length=6, break_before=breaks)]
    )
    view.seek(TIMES[-1])
    trail = view.native.layers["Position trail 1"].data
    expected = [path[[0, 1]], path[[4, 5]]]
    assert len(trail) == len(expected)
    for segment, pair in zip(trail, expected, strict=True):
        np.testing.assert_allclose(segment, pair[:, ::-1])
    view.seek(TIMES[2])
    assert len(view.native.layers["Position 1"].data) == 0


@pytest.mark.napari
@pytest.mark.xdist_group(name="napari_gui")
def test_napari_pose_hides_missing_keypoints_and_their_edges(napari_view):
    nose = np.column_stack([np.arange(6.0) + 2, np.full(6, 6.0)])
    nose[1] = np.nan
    tail = nose.copy()
    tail[:, 1] = 4.0
    tail[1] = [3.0, 4.0]
    skeleton = Skeleton(name="rat", nodes=("nose", "tail"), edges=(("nose", "tail"),))
    view = napari_view(
        [BodypartOverlay({"nose": nose, "tail": tail}, TIMES, skeleton=skeleton)]
    )
    layers = view.native.layers
    view.seek(TIMES[1])
    assert len(layers["Body 1: nose"].data) == 0
    np.testing.assert_allclose(layers["Body 1: tail"].data, [[4.0, 3.0]])
    assert len(layers["Skeleton 1"].data) == 0
    view.seek(TIMES[2])
    np.testing.assert_allclose(layers["Body 1: nose"].data, [[6.0, 4.0]])
    assert len(layers["Skeleton 1"].data) == 1


@pytest.mark.napari
@pytest.mark.xdist_group(name="napari_gui")
def test_napari_heading_is_hidden_while_its_animal_is_missing(napari_view):
    path = np.full((6, 2), 5.0)
    path[1] = np.nan
    view = napari_view(
        [PositionOverlay(path, TIMES), HeadDirectionOverlay(np.zeros(6), TIMES)]
    )
    layer = view.native.layers["Head direction 1"]
    view.seek(TIMES[1])
    assert len(layer.data) == 0
    view.seek(TIMES[2])
    np.testing.assert_allclose(layer.data[0, 0], [5.0, 5.0])


def _red_pixels(frame):
    frame = frame.astype(int)
    red, green, blue = frame[..., 0], frame[..., 1], frame[..., 2]
    return np.argwhere((red > 200) & (green < 60) & (blue < 60))


def test_rendered_heading_is_hidden_while_its_animal_is_missing(env):
    """Exported frames draw no heading arrow without an observed anchor."""
    path = np.full((6, 2), 5.0)
    path[1] = np.nan
    heading = HeadDirectionOverlay(np.zeros(6), TIMES, length=3.0, color="red")
    # A white field, so red pixels can only come from the arrow.
    field = AnimationField(
        np.zeros((6, env.n_bins)), TIMES, cmap="Greys", vmin=0.0, vmax=1.0
    )
    # Below two frames, so the dropout is a tracking loss, not bridged.
    position = PositionOverlay(path, TIMES, color="blue", max_gap=0.15)
    with Animation(env, {"field": field}, overlays=[position, heading]) as scene:
        assert len(_red_pixels(scene.render_frame(TIMES[1], dpi=40))) == 0
        assert len(_red_pixels(scene.render_frame(TIMES[2], dpi=40))) > 0
    # With no position overlay the arrow is a compass at the arena center.
    with Animation(env, {"field": field}, overlays=[heading]) as scene:
        assert len(_red_pixels(scene.render_frame(TIMES[1], dpi=40))) > 0
