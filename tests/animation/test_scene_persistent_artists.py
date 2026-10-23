"""A reused scene renderer draws each time exactly as a new renderer would."""

import numpy as np
import pytest
from matplotlib.collections import LineCollection, QuadMesh

from neurospatial import Environment
from neurospatial.animation._scene_rendering import SceneRenderer
from neurospatial.animation.overlays import (
    BodypartOverlay,
    EventOverlay,
    HeadDirectionOverlay,
    PositionOverlay,
    SpikeRasterOverlay,
    TimeSeriesOverlay,
    VideoOverlay,
)
from neurospatial.animation.scene import Animation
from neurospatial.animation.skeleton import Skeleton

CLOCK = np.arange(12) * 0.1


@pytest.fixture
def env():
    points = np.random.default_rng(3).uniform(0, 10, (300, 2))
    return Environment.from_samples(points, bin_size=2.0)


@pytest.fixture
def scene(env):
    rng = np.random.default_rng(4)
    fields = rng.random((len(CLOCK), env.n_bins))
    fields[5:7] = np.nan  # a recording gap
    walk = 2 + 6 * rng.random((len(CLOCK), 2))
    walk[4] = np.nan  # a tracking dropout
    video = VideoOverlay(
        rng.integers(0, 255, (6, 4, 4, 3), dtype=np.uint8),
        times=CLOCK[3:9],
        z_order="below",
        alpha=1.0,
    )
    overlays = [
        video,
        PositionOverlay(walk, CLOCK, trail_length=4),
        BodypartOverlay(
            {"nose": walk, "tail": walk + 0.5},
            CLOCK,
            skeleton=Skeleton(
                name="m", nodes=("nose", "tail"), edges=(("nose", "tail"),)
            ),
        ),
        HeadDirectionOverlay(np.linspace(0, 6, len(CLOCK)), CLOCK, length=2.0),
        EventOverlay.from_fixed_positions(
            np.array([0.15, 0.55]), np.array([[3.0, 3.0], [7.0, 7.0]]), decay_frames=2
        ),
        TimeSeriesOverlay(np.sin(CLOCK * 5), CLOCK, label="LFP"),
        SpikeRasterOverlay([np.array([0.1, 0.9]), np.array([0.2])]),
    ]
    with pytest.warns(UserWarning, match="no calibration"):
        animation = Animation(env, fields, frame_times=CLOCK, overlays=overlays)
    yield animation
    animation.close()


def render_new(scene, time):
    renderer = SceneRenderer(scene, dpi=25)
    try:
        return renderer.render(time)
    finally:
        renderer.close()


def test_seeking_in_any_order_draws_what_a_new_renderer_draws(scene):
    # Times inside the video, the tracking dropout and the field gap, and
    # outside the video, so every artist is hidden and shown again.
    times = [1.1, 0.0, 0.45, 0.55, 0.15, 0.95, 0.4, 0.65]
    expected = {time: render_new(scene, time) for time in times}
    renderer = SceneRenderer(scene, dpi=25)
    for time in [*times, *times[::-1]]:
        np.testing.assert_array_equal(renderer.render(time), expected[time])
    renderer.close()


def test_missing_samples_hide_artists_drawn_on_the_previous_frame(scene):
    renderer = SceneRenderer(scene, dpi=25)
    renderer.render(0.3)
    ax = renderer.figure.axes[0]

    def mesh():
        return next(item for item in ax.collections if isinstance(item, QuadMesh))

    trail = next(item for item in ax.collections if isinstance(item, LineCollection))
    marker = next(item for item in ax.collections if item.get_zorder() == 101)
    assert np.isfinite(mesh().get_array()).any()
    assert len(trail.get_segments()) == 2 and marker.get_visible()
    # Position dropout: no marker and no trail bridging the gap.
    renderer.render(0.4)
    assert not marker.get_visible()
    assert trail.get_segments() == []
    # Field gap: every bin is missing, not the previous frame's values.
    renderer.render(0.55)
    assert np.ma.getmaskarray(mesh().get_array()).all()
    # Overlay artists are updated rather than replaced.
    assert trail in ax.collections and marker in ax.collections
    renderer.close()


def test_raster_rows_stay_fixed_while_units_fall_silent(scene):
    renderer = SceneRenderer(scene, dpi=25)
    renderer.render(0.2)
    raster = next(ax for ax in renderer.figure.axes if ax.get_ylabel() == "Unit")
    limits = raster.get_ylim()
    assert limits[0] <= -0.4 and limits[1] >= 1.4  # every row is in view
    renderer.render(1.1)  # the second unit has no spike in this window
    assert raster.get_ylim() == limits
    renderer.close()
