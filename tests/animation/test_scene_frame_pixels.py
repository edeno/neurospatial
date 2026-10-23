"""Rendered scene frames draw fields and overlays where they belong.

Each test reads pixels of ``Animation.render_frame`` at environment
coordinates, so it checks what a viewer or an exported file shows rather than
which Matplotlib artists were created.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib.colors import to_rgb

from neurospatial import Environment, Regions
from neurospatial.animation import Animation, AnimationField
from neurospatial.animation.overlays import (
    BodypartOverlay,
    EventOverlay,
    HeadDirectionOverlay,
    PositionOverlay,
)
from neurospatial.animation.skeleton import Skeleton

DPI = 60
TIMES = np.arange(4) * 0.1  # seconds


@pytest.fixture
def env():
    """A 100 x 100 arena of 10-unit cells."""
    edges = np.arange(0.0, 110.0, 10.0)
    return Environment.from_grid_mask(np.ones((10, 10), dtype=bool), (edges, edges))


def dark(env, times=TIMES):
    """A zero field drawn black, so overlay colors show unblended."""
    values = np.zeros((len(times), env.n_bins))
    return {"field": AnimationField(values, times, cmap="gray", vmin=0, vmax=1)}


def render(scene, time):
    """Render ``time``; return the image and an environment-point pixel reader."""
    image = scene.render_frame(time, dpi=DPI)
    ax = scene._renderer.figure.axes[0]

    def at(x, y, radius=0):
        col, row = ax.transData.transform((x, y))
        r, c = int(image.shape[0] - row), int(col)
        patch = image[r - radius : r + radius + 1, c - radius : c + radius + 1]
        return patch.reshape(-1, 3).astype(int)

    return image, at


def rgb(color):
    return np.asarray(to_rgb(color)) * 255


def shows(pixels, color, atol=12):
    """Whether any pixel matches ``color``."""
    return bool(np.any(np.all(np.abs(pixels - rgb(color)) <= atol, axis=1)))


@pytest.mark.parametrize(
    ("cmap", "vmax", "hot_level"),
    [("viridis", 10.0, 1.0), ("viridis", 20.0, 0.5), ("magma", 10.0, 1.0)],
)
def test_field_values_take_their_colormap_color_at_their_bin(
    env, cmap, vmax, hot_level
):
    values = np.zeros(env.n_bins)
    hot, missing = env.bin_at(np.array([[85.0, 15.0], [15.0, 85.0]]))
    values[hot], values[missing] = 10.0, np.nan
    field = AnimationField(np.tile(values, (2, 1)), [0.0, 0.1], cmap=cmap, vmax=vmax)
    with Animation(env, {"rate": field}) as scene:
        _, at = render(scene, 0.0)
        assert shows(at(85, 15), colormaps[cmap](hot_level), atol=3)
        assert shows(at(55, 55), colormaps[cmap](0.0), atol=3)
        # The hot bin is not mirrored to another corner.
        assert not shows(at(85, 85), colormaps[cmap](hot_level), atol=20)
        # A missing bin is drawn as missing, not as zero activity.
        assert not shows(at(15, 85), colormaps[cmap](0.0), atol=20)


@pytest.fixture
def recording():
    positions = np.array([[25.0, 25.0], [40.0, 30.0], [55.0, 60.0], [70.0, 70.0]])
    return positions


def test_position_marker_follows_the_recording_and_hides_when_missing(env, recording):
    positions = recording.copy()
    positions[2] = np.nan
    overlay = PositionOverlay(
        positions, TIMES, color="red", size=12.0, interp="nearest"
    )
    with Animation(env, dark(env), overlays=[overlay]) as scene:
        _, at = render(scene, 0.1)
        assert shows(at(40, 30), "red", atol=3)
        assert not shows(at(25, 25, radius=3), "red", atol=60)
        _, at = render(scene, 0.2)
        assert not any(
            shows(at(x, y, radius=3), "red", atol=60) for x, y in recording[[0, 1, 3]]
        )


def test_trail_fades_with_age(env):
    # Older segments fade, as in fastplotlib, so the direction of travel shows.
    x = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
    positions = np.column_stack([x, np.full(5, 50.0)])
    times = np.arange(5) * 0.1
    overlay = PositionOverlay(positions, times, color="red", size=6.0, trail_length=5)
    with Animation(env, dark(env, times), overlays=[overlay]) as scene:
        _, at = render(scene, 0.4)
        redness = [at(mid, 50.0, radius=2)[:, 0].max() for mid in (20, 40, 60, 80)]
    assert min(redness) > 0
    assert np.all(np.diff(redness) > 0), redness


def test_pose_draws_observed_parts_and_only_complete_skeleton_edges(env, recording):
    nose = recording + np.array([-5.0, 0.0])
    tail = recording + np.array([0.0, -8.0])
    tail[1] = np.nan
    skeleton = Skeleton(name="test", nodes=("nose", "tail"), edges=(("nose", "tail"),))
    overlay = BodypartOverlay(
        {"nose": nose, "tail": tail},
        TIMES,
        skeleton=skeleton,
        colors={"nose": "magenta", "tail": "cyan"},
    )
    with Animation(env, dark(env), overlays=[overlay]) as scene:
        _, at = render(scene, 0.0)
        assert shows(at(*nose[0], radius=1), "magenta")
        assert shows(at(*tail[0], radius=1), "cyan")
        midpoint = (nose[0] + tail[0]) / 2
        assert shows(at(*midpoint, radius=2), "white", atol=60)
        _, at = render(scene, 0.1)
        assert shows(at(*nose[1], radius=1), "magenta")
        # No tail, so no edge drawn toward a guessed location.
        assert not shows(at(*(nose[1] + [0.0, -4.0]), radius=2), "white", atol=60)


def test_skeleton_is_drawn_in_its_own_style(env, recording):
    # The skeleton's edge color and width, as napari and fastplotlib draw it.
    nose = recording + np.array([-15.0, 0.0])
    tail = recording + np.array([15.0, 0.0])
    skeleton = Skeleton(
        name="test",
        nodes=("nose", "tail"),
        edges=(("nose", "tail"),),
        edge_color="lime",
        edge_width=6.0,
    )
    overlay = BodypartOverlay({"nose": nose, "tail": tail}, TIMES, skeleton=skeleton)
    with Animation(env, dark(env), overlays=[overlay]) as scene:
        _, at = render(scene, 0.0)
        assert shows(at(*(nose[0] + tail[0]) / 2), "lime", atol=3)
        # Six points are five pixels at 60 dpi, across the horizontal edge.
        column = at(*(nose[0] + tail[0]) / 2, radius=6)[6::13]
        lime = np.all(np.abs(column - rgb("lime")) <= 30, axis=1)
        assert lime.sum() >= 4


@pytest.mark.parametrize("color", ["hsv", "twilight", "yellow"])
@pytest.mark.parametrize("angle", [0.0, np.pi / 2, np.pi])
def test_heading_arrow_points_along_the_heading_in_its_color(env, color, angle):
    overlay = HeadDirectionOverlay(
        np.full(4, angle), TIMES, color=color, length=30.0, width=3.0
    )
    expected = (
        colormaps[color]((angle / (2 * np.pi)) % 1.0) if color in colormaps else color
    )
    direction = np.array([np.cos(angle), np.sin(angle)])
    # Without a position overlay the arrow starts at the arena center.
    center = env.bin_centers.mean(axis=0)
    with Animation(env, dark(env), overlays=[overlay]) as scene:
        _, at = render(scene, 0.0)
        assert shows(at(*(center + 12 * direction)), expected, atol=3)
        assert not shows(at(*(center - 12 * direction), radius=2), expected, atol=40)


def test_events_draw_at_their_location_in_their_type_color(env):
    overlay = EventOverlay.from_fixed_positions(
        {"reward": np.array([0.1]), "lick": np.array([0.1])},
        {"reward": np.array([[30.0, 70.0]]), "lick": np.array([[70.0, 30.0]])},
        colors={"reward": "cyan", "lick": "orange"},
        markers={"reward": "s", "lick": "o"},
        size=14.0,
        opacity=1.0,
        decay_frames=0,
    )
    with Animation(env, dark(env), overlays=[overlay]) as scene:
        _, at = render(scene, 0.1)
        assert shows(at(30, 70), "cyan", atol=3)
        assert shows(at(70, 30), "orange", atol=3)
        assert not shows(at(70, 30, radius=3), "cyan", atol=40)
        # decay_frames=0 shows an event only on its own frame.
        _, at = render(scene, 0.2)
        assert not shows(at(30, 70, radius=3), "cyan", atol=60)


def test_event_without_a_tracked_location_is_not_drawn(env):
    track_times = np.array([0.0, 0.1, 0.2])
    track = np.array([[20.0, 20.0], [50.0, 50.0], [80.0, 80.0]])

    def events(times):
        return EventOverlay.from_trajectory(
            times, track, track_times, colors="cyan", size=14.0, opacity=1.0
        )

    # The event at 0.3 s falls after tracking ends, so its location is unknown.
    with (
        Animation(env, dark(env), overlays=[events(np.array([0.1, 0.3]))]) as scene,
        Animation(env, dark(env), overlays=[events(np.array([0.1]))]) as located,
    ):
        image, at = render(scene, 0.3)
        assert shows(at(50, 50), "cyan", atol=3)
        np.testing.assert_array_equal(image, located.render_frame(0.3, dpi=DPI))


def test_shown_regions_are_drawn_at_their_location(env):
    regions = Regions()
    regions.add("goal", point=np.array([75.0, 25.0]))
    env = env.with_regions(regions)
    with (
        Animation(env, dark(env), show_regions=["goal"], region_alpha=0.5) as shown,
        Animation(env, dark(env)) as hidden,
    ):
        _, at = render(shown, 0.0)
        _, at_hidden = render(hidden, 0.0)
        assert at(75, 25).sum() > at_hidden(75, 25).sum() + 100
        np.testing.assert_array_equal(at(25, 75, radius=3), at_hidden(25, 75, radius=3))
