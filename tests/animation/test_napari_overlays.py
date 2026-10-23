"""animate_fields(backend="napari") draws overlays where the animal was.

Overlay data are environment ``(x, y)`` coordinates; napari stores ``(y, x)``.
The arena is wider in y than in x so a swapped or flipped axis shows up.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation.overlays import (
    BodypartOverlay,
    HeadDirectionOverlay,
    PositionOverlay,
    VideoData,
)
from neurospatial.animation.skeleton import Skeleton
from neurospatial.regions import Regions

pytestmark = [pytest.mark.napari, pytest.mark.xdist_group(name="napari_gui")]

TIMES = np.array([0.0, 0.5, 1.0])
# Nose and tail sit this (x, y) offset ahead of and behind the animal.
OFFSET = np.array([1.0, 2.0])


@pytest.fixture
def env():
    from shapely.geometry import Polygon

    env = Environment.from_grid_mask(
        np.ones((4, 6), dtype=bool),
        (np.linspace(0.0, 40.0, 5), np.linspace(0.0, 60.0, 7)),
        units="cm",
    )
    regions = Regions()
    regions.add("reward", point=(35.0, 55.0))
    regions.add("start", polygon=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]))
    return env.with_regions(regions)


@pytest.fixture
def open_view(env):
    pytest.importorskip("napari")
    views = []

    def open_(overlays=(), **options):
        view = env.animate_fields(
            np.ones((len(TIMES), env.n_bins)),
            frame_times=TIMES,
            overlays=list(overlays),
            backend="napari",
            show=False,
            vmin=0,
            vmax=1,
            **options,
        )
        views.append(view)
        return view

    yield open_
    for view in views:
        view.close()


def test_position_pose_and_heading_land_at_environment_coordinates(open_view):
    path = np.array([[5.0, 50.0], [20.0, 30.0], [35.0, 10.0]])
    skeleton = Skeleton(name="rat", nodes=("nose", "tail"), edges=(("nose", "tail"),))
    view = open_view(
        [
            PositionOverlay(path, TIMES, color="red"),
            BodypartOverlay(
                {"nose": path + OFFSET, "tail": path - OFFSET},
                TIMES,
                skeleton=skeleton,
                colors={"nose": "yellow", "tail": "cyan"},
            ),
            # East, then North: (x, y) displacements (2, 0) and (0, 2).
            HeadDirectionOverlay(
                np.array([0.0, np.pi / 2, np.pi / 2]), TIMES, length=2.0, color="red"
            ),
        ]
    )
    layers = view.native.layers
    for k, time in enumerate(TIMES[:2]):
        view.seek(time)
        x, y = path[k]
        np.testing.assert_allclose(layers["Position 1"].data, [[y, x]])
        np.testing.assert_allclose(layers["Body 1: nose"].data, [[y + 2, x + 1]])
        np.testing.assert_allclose(layers["Body 1: tail"].data, [[y - 2, x - 1]])
        (edge,) = layers["Skeleton 1"].data
        np.testing.assert_allclose(edge, [[y + 2, x + 1], [y - 2, x - 1]])
        # The arrow starts at the animal, the only position overlay.
        direction = [[0.0, 2.0], [2.0, 0.0]][k]
        np.testing.assert_allclose(
            layers["Head direction 1"].data, [[[y, x], direction]], atol=1e-12
        )
    # Each bodypart keeps its own color.
    np.testing.assert_allclose(layers["Body 1: nose"].face_color[0], [1, 1, 0, 1])
    np.testing.assert_allclose(layers["Body 1: tail"].face_color[0], [0, 1, 1, 1])


def test_heading_without_an_animal_is_a_compass_at_the_center(open_view, env):
    view = open_view([HeadDirectionOverlay(np.zeros(3), TIMES, length=2.0)])
    view.seek(0.5)
    center = env.bin_centers.mean(axis=0)
    np.testing.assert_allclose(
        view.native.layers["Head direction 1"].data,
        [[center[::-1], [0.0, 2.0]]],
        atol=1e-12,
    )


def test_heading_with_several_animals_is_refused(open_view):
    """No single position anchors the arrow, so the call says how to choose."""
    overlays = [PositionOverlay(np.full((3, 2), 5.0 + 10 * i), TIMES) for i in range(2)]
    with pytest.raises(
        ValueError, match=r"cannot tell which of the 2 PositionOverlays"
    ):
        open_view([*overlays, HeadDirectionOverlay(np.zeros(3), TIMES)])


def test_regions_are_drawn_only_when_requested(open_view):
    def regions(view):
        return {
            layer.name: layer
            for layer in view.native.layers
            if layer.name.startswith("Region")
        }

    assert regions(open_view()) == {}
    shown = regions(open_view(show_regions=["reward"], region_alpha=0.4))
    assert list(shown) == ["Region: reward"]
    np.testing.assert_allclose(shown["Region: reward"].data, [[55.0, 35.0]])
    assert shown["Region: reward"].opacity == pytest.approx(0.4)
    shown = regions(open_view(show_regions=True))
    assert set(shown) == {"Region: reward", "Region: start"}
    (square,) = shown["Region: start"].data
    np.testing.assert_allclose(square.min(axis=0), [0.0, 0.0])
    np.testing.assert_allclose(square.max(axis=0), [10.0, 10.0])


@pytest.mark.parametrize("z_order", ["above", "below"])
def test_video_is_stacked_against_the_field_with_its_opacity(open_view, z_order):
    class Video:
        times = None
        interp = "nearest"

        def convert_to_data(self, frame_times, n_frames, env):
            frames = np.zeros((n_frames, 2, 2, 3), dtype=np.uint8)
            return VideoData(
                np.arange(n_frames),
                frames,
                None,
                (0, 40, 0, 60),
                0.4,
                z_order,
                times=frame_times,
            )

    view = open_view([Video()])
    names = [layer.name for layer in view.native.layers]
    assert names.index("Video 1") == (1 if z_order == "above" else 0)
    assert names[1 - names.index("Video 1")] == "field"
    assert view.native.layers["Video 1"].opacity == pytest.approx(0.4)
