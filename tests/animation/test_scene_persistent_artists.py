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
    walk[4] = np.nan  # a tracking dropout, longer than max_gap below
    video = VideoOverlay(
        rng.integers(0, 255, (6, 4, 4, 3), dtype=np.uint8),
        times=CLOCK[3:9],
        z_order="below",
        alpha=1.0,
    )
    overlays = [
        video,
        PositionOverlay(walk, CLOCK, trail_length=4, max_gap=0.15),
        BodypartOverlay(
            {"nose": walk, "tail": walk + 0.5},
            CLOCK,
            max_gap=0.15,
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


def test_1d_field_axes_stay_fixed_and_keep_markers_through_missing_rows():
    # A missing field row must not move the axes or hide the animal, and the
    # value axis keeps one scale so changes in magnitude stay visible.
    from neurospatial import Environment
    from neurospatial.animation import Animation, AnimationField, PositionOverlay

    env = Environment.from_samples(np.linspace(0, 20, 200)[:, None], bin_size=2.0)
    times = np.arange(4) * 0.1
    values = np.vstack([np.linspace(1, 2, env.n_bins), np.full(env.n_bins, np.nan)] * 2)
    field = AnimationField(values, times, vmin=0.5, vmax=3.0)
    position = PositionOverlay(np.full((4, 1), 12.0), times=times)
    with Animation(env, field, overlays=[position]) as scene:
        scene.render_frame(0.0, dpi=30)
        ax = scene._renderer.figure.axes[0]
        limits = ax.get_xlim(), ax.get_ylim()
        scene.render_frame(0.1, dpi=30)
        assert (ax.get_xlim(), ax.get_ylim()) == limits
        xmin, xmax = limits[0]
        ymin, ymax = limits[1]
        assert xmin <= env.dimension_ranges[0][0] and xmax >= env.dimension_ranges[0][1]
        assert ymin <= 0 and ymax >= 3.0
        markers = [
            artist.get_offsets()
            for artist in ax.collections
            if artist.get_visible() and len(artist.get_offsets())
        ]
        assert any(np.allclose(offsets, [[12.0, 0.0]]) for offsets in markers)


def test_trail_spans_scene_frames_whatever_the_tracking_rate():
    # trail_length counts animation frames, as in fastplotlib: 60 Hz tracking
    # under a 10 Hz field still trails 0.3 s for trail_length=3, every sample.
    from matplotlib.collections import LineCollection

    from neurospatial import Environment
    from neurospatial.animation import Animation, PositionOverlay

    env = Environment.from_grid_mask(
        np.ones((4, 4), dtype=bool), (np.arange(5.0), np.arange(5.0))
    )
    frames = np.arange(11) * 0.1
    tracking = np.arange(61) / 60
    path = np.column_stack([np.linspace(0.5, 3.5, 61), np.full(61, 2.0)])
    position = PositionOverlay(path, times=tracking, trail_length=3)
    with Animation(
        env, np.ones((11, env.n_bins)), frame_times=frames, overlays=[position]
    ) as scene:
        # Off the sample grid, so the 0.3 s window edge is not a roundoff tie.
        scene.render_frame(0.805, dpi=30)
        rows = scene._renderer.prepared.trail_rows(
            scene._overlay_data.positions[0], 0.805
        )
        assert tracking[rows.start] == pytest.approx(31 / 60)
        assert tracking[rows.stop - 1] == pytest.approx(48 / 60)
        (trail,) = [
            artist
            for artist in scene._renderer.figure.axes[0].collections
            if isinstance(artist, LineCollection) and len(artist.get_segments())
        ]
        assert len(trail.get_segments()) == len(rows) - 1 == 17
