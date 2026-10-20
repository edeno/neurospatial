"""Prepared streams share support and panel data across complete-scene consumers."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from unittest.mock import Mock

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import (
    Animation,
    AnimationField,
    HeadDirectionOverlay,
    ObjectVectorOverlay,
    PositionOverlay,
    SpikeRasterOverlay,
    TimeSeriesOverlay,
)
from neurospatial.animation._capabilities import require_scene_view
from neurospatial.animation._prepared import PreparedScene
from neurospatial.animation._sampling import sample_at_times
from neurospatial.animation._scene_rendering import SceneRenderer
from neurospatial.animation.overlays import VideoData


@pytest.fixture
def env():
    return Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )


class MappedVideo:
    """A custom source whose native samples reorder and omit reader frames."""

    times = np.array([0.0, 0.1, 0.2])
    interp = "nearest"

    def convert_to_data(self, frame_times, n_frames, env):
        frames = np.stack(
            [np.full((2, 2, 3), value, dtype=np.uint8) for value in [20, 100, 200]]
        )
        return VideoData(
            np.array([2, -1, 0]),
            frames,
            None,
            (0, 2, 0, 2),
            1.0,
            "above",
            times=self.times,
        )


def test_scene_video_uses_prepared_native_mapping_for_every_read(env):
    scene = Animation(
        env, np.ones((2, env.n_bins)), frame_times=[0.0, 0.2], overlays=[MappedVideo()]
    )
    prepared = PreparedScene(scene)
    video = scene._overlay_data.videos[0]
    np.testing.assert_array_equal(video.frame_indices, [2, -1, 0])
    assert prepared.clock(video) is video.times
    assert prepared.video_frame(video, -0.001) is None
    assert prepared.video_frame(video, 0.201) is None
    try:
        for time, expected in [(0.0, 200), (0.05, 200), (0.1, None), (0.2, 20)]:
            scene.render_frame(time, dpi=25)
            images = scene._renderer.figure.axes[0].images
            if expected is None:
                assert not images
            else:
                assert images[0].get_array()[0, 0, 0] == expected
    finally:
        scene.close()


def test_1d_headings_refused_consistently_before_render_or_export(
    tmp_path, monkeypatch
):
    env = Environment.from_grid_mask(np.ones(3, dtype=bool), (np.arange(4.0),))
    scene = Animation(
        env,
        np.ones((2, env.n_bins)),
        frame_times=[0.0, 0.1],
        overlays=[
            HeadDirectionOverlay(np.array([0.0, 1.0]), times=np.array([0.0, 0.1]))
        ],
    )
    writer = Mock(
        side_effect=AssertionError("No writer may open for an unsupported scene")
    )
    monkeypatch.setattr("imageio.v2.get_writer", writer)
    for action in (
        lambda: require_scene_view(scene, "widget", {}),
        lambda: scene.show(backend="widget"),
        lambda: scene.render_frame(0.05),
        lambda: scene.export(tmp_path / "unsupported.gif", fps=10),
    ):
        with pytest.raises(
            ValueError, match=r"1D.*HeadDirectionOverlay.*TimeSeriesOverlay"
        ):
            action()
    assert scene._renderer is None
    writer.assert_not_called()
    assert not (tmp_path / "unsupported.gif").exists()
    scene.close()


def test_1d_positions_and_object_vectors_have_defined_spatial_baseline():
    env = Environment.from_grid_mask(np.ones(3, dtype=bool), (np.arange(4.0),))
    times = np.array([0.0, 0.1])
    scene = Animation(
        env,
        np.ones((2, env.n_bins)),
        frame_times=times,
        overlays=[
            PositionOverlay(
                np.array([[0.0], [1.0]]), times=times, color="cyan", trail_length=2
            ),
            ObjectVectorOverlay(
                np.array([[2.0]]),
                np.array([[0.0], [1.0]]),
                times=times,
                color="magenta",
            ),
        ],
    )
    require_scene_view(scene, "widget", {})
    try:
        scene.render_frame(0.05, dpi=25)
        ax = scene._renderer.figure.axes[0]
        line = next(line for line in ax.lines if line.get_color() == "magenta")
        np.testing.assert_allclose(line.get_xdata(), [0.5, 2.0])
        np.testing.assert_array_equal(line.get_ydata(), [0, 0])
        points = np.concatenate([artist.get_offsets() for artist in ax.collections])
        assert any(np.allclose(point, [2.0, 0.0]) for point in points)
        assert any(np.allclose(point, [0.5, 0.0]) for point in points)
    finally:
        scene.close()


def test_prepared_panels_keep_native_support_and_half_open_field_endpoints():
    env = Environment.from_grid_mask(np.ones(2, dtype=bool), (np.arange(3.0),))
    field = AnimationField(
        np.array([[1, np.nan], [2, 3], [100, 100], [4, 5], [6, 7]], dtype=float),
        [0.0, 0.05, 0.1, 0.2, 0.25],
        epochs=[(0, 0.1), (0.2, 0.3)],
        max_gap=None,
    )
    scene = Animation(
        env,
        {"posterior": field},
        frame_times=[0.0, 0.25],
        window_seconds=0.5,
        overlays=[
            TimeSeriesOverlay(
                np.array([0.0, 2.0, 1.0]), np.array([0.0, 0.1, 0.2]), normalize=True
            ),
            SpikeRasterOverlay([np.array([0.01, 0.2])], unit_ids=["unit"]),
        ],
    )
    prepared = PreparedScene(scene)
    assert not hasattr(prepared, "figure")
    assert prepared.scene.fields["posterior"].values is field.values
    window = prepared.track_window(scene.fields["posterior"], 0.1)
    assert not np.any(
        (window.polygons[:, :, 0].min(axis=1) < 0.2)
        & (window.polygons[:, :, 0].max(axis=1) > 0.1)
    )
    assert not np.any(window.values == 100)
    np.testing.assert_array_equal(scene.sample_fields(0.075)["posterior"], [2, 3])
    assert np.isnan(scene.sample_fields(0.1)["posterior"]).all()
    signal = scene._overlay_data.timeseries[0]
    times, values = prepared.signal_window(signal, 0.1)
    np.testing.assert_array_equal(times, signal.times)
    np.testing.assert_array_equal(values, [0, 1, 0.5])
    try:
        renderer = SceneRenderer(scene, dpi=25)
        renderer.render(0.1)
        collection = renderer.figure.axes[1].collections[0]
        np.testing.assert_array_equal(collection.get_array(), window.values)
        signal_ax = renderer.figure.axes[2]
        np.testing.assert_array_equal(signal_ax.lines[0].get_ydata(), values)
        raster_ax = renderer.figure.axes[3]
        actual = np.array(
            [segment[0, 0] for segment in raster_ax.collections[0].get_segments()]
        )
        np.testing.assert_array_equal(
            actual, prepared.raster_window(scene._overlay_data.spike_rasters[0], 0, 0.1)
        )
        renderer.close()
    finally:
        scene.close()


def test_sampling_checks_changed_values_only_at_requested_rows(monkeypatch):
    times = np.arange(100_000, dtype=float)
    values = np.ones((len(times), 2))
    values[11, 0] = np.nan  # A caller can change values after support preparation.
    isfinite = np.isfinite
    sizes = []

    def bounded_isfinite(array, *args, **kwargs):
        sizes.append(np.asarray(array).size)
        return isfinite(array, *args, **kwargs)

    monkeypatch.setattr(np, "isfinite", bounded_isfinite)
    result = sample_at_times(
        times, values, [10.5], sample_valid=np.ones(len(times), dtype=bool)
    )
    assert np.isnan(result).all()
    assert max(sizes) <= 4


def test_fresh_napari_preparation_imports_no_canvas_or_renderer():
    script = textwrap.dedent("""
        import sys
        import numpy as np
        from neurospatial import Environment
        from neurospatial.animation import Animation, AnimationField
        from neurospatial.animation.backends import _napari_scene
        env = Environment.from_grid_mask(np.ones((2, 2), dtype=bool), (np.arange(3.), np.arange(3.)))
        track = Environment.linear_track(endpoints=[[0, .5], [2, .5]], bin_size=1)
        scene = Animation(env, AnimationField(np.ones((2, env.n_bins)), [0, .1]), track=track)
        prepared = _napari_scene.PreparedScene(scene)
        assert not hasattr(prepared, 'figure')
        assert not hasattr(_napari_scene, 'SceneRenderer')
        assert 'matplotlib.backends.backend_agg' not in sys.modules
        assert 'matplotlib.pyplot' not in sys.modules
        assert len(prepared.track_window(scene.fields['field'], .05).values) == 2
        scene.close()
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
