"""plot, plot_field and animate_fields with backend="fastplotlib", rendered off-screen.

Skipped without fastplotlib (``pip install "neurospatial[fastplotlib]"``) or a
GPU adapter.
"""

from __future__ import annotations

import os
import warnings

# Render into memory: no window, and the same in a terminal, on CI or over SSH.
os.environ.setdefault("RENDERCANVAS_FORCE_OFFSCREEN", "1")

import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib.colors import to_rgba
from shapely.geometry import LineString, box

fastplotlib = pytest.importorskip("fastplotlib")
wgpu = pytest.importorskip("wgpu")
try:
    _ADAPTER = wgpu.gpu.request_adapter_sync()
except Exception:
    _ADAPTER = None
if _ADAPTER is None:
    pytest.skip("no GPU adapter for fastplotlib", allow_module_level=True)

from neurospatial import Environment  # noqa: E402
from neurospatial.animation import (  # noqa: E402
    EventOverlay,
    FastplotlibAnimation,
    PositionOverlay,
    TimeSeriesOverlay,
    VideoOverlay,
)
from neurospatial.arena import OpenArena  # noqa: E402
from neurospatial.landmarks import Landmark, Landmarks  # noqa: E402
from neurospatial.ops.transforms import Affine2D, VideoCalibration  # noqa: E402

EDGES = np.arange(0.0, 41.0, 4.0)


@pytest.fixture(scope="module")
def apparatus():
    arena = OpenArena(
        box(0, 0, 40, 40), walls={"divider": LineString([(20, 0), (20, 28)])}
    )
    env = Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool),
        grid_edges=(EDGES, EDGES),
        units="cm",
        arena=arena,
    )
    return env.with_landmarks(Landmarks([Landmark("cue", "point", [20.0, 44.0])]))


def _render(animation):
    """Draw every graphic at the slider's time now.

    Without a running event loop, the widget's updates are only queued.
    """
    from fastplotlib.widgets.nd_widget._async import run_sync

    for graphic in animation.widget.ndgraphics:
        run_sync(graphic._set_indices_())


def _pixel(figure, subplot, x, y):
    """The rendered color at environment coordinates (x, y) of ``subplot``."""
    image = np.asarray(figure.canvas.draw())
    xy = subplot.map_world_to_screen((x, y, 0))
    ratio = image.shape[1] / figure.canvas.get_logical_size()[0]
    column, row = (np.asarray(xy[:2]) * ratio).astype(int)
    return image[row, column, :3] / 255.0


def test_plot_and_plot_field_return_figures(apparatus):
    figure = apparatus.plot(backend="fastplotlib")
    assert isinstance(figure, fastplotlib.Figure)
    figure.show()
    texts = [g.text for g in figure[0, 0].graphics if hasattr(g, "text")]
    assert {"divider", "cue"} <= set(texts)  # walls and landmarks are named
    field = np.arange(apparatus.n_bins, dtype=float)
    assert isinstance(
        apparatus.plot_field(field, backend="fastplotlib"), fastplotlib.Figure
    )


def test_each_bin_shows_its_value_and_missing_bins_their_own_color():
    env = Environment.from_grid_mask(
        np.ones((2, 1), dtype=bool),
        grid_edges=(np.array([0.0, 10, 20]), np.array([0.0, 10])),
    )
    figure = env.plot_field(
        np.array([0.0, np.nan]), backend="fastplotlib", vmin=0.0, vmax=1.0
    )
    figure.show()
    plot = figure[0, 0]
    low = _pixel(figure, plot, 5.0, 5.0)
    missing = _pixel(figure, plot, 15.0, 5.0)
    np.testing.assert_allclose(low, colormaps["viridis"](0.0)[:3], atol=2 / 255)
    np.testing.assert_allclose(missing, to_rgba("lightgray")[:3], atol=2 / 255)


def test_missing_bins_keep_their_color_whatever_the_table_length(monkeypatch):
    # pygfx resamples an array colormap that is not 256 rows long; passed as a
    # nearest-sampled texture, the missing row is never blended.
    import neurospatial.environment._fastplotlib as drawing

    monkeypatch.setattr(drawing, "_N_COLORS", 100)
    env = Environment.from_grid_mask(
        np.ones((2, 1), dtype=bool),
        grid_edges=(np.array([0.0, 10, 20]), np.array([0.0, 10])),
    )
    figure = env.plot_field(
        np.array([0.0, np.nan]), backend="fastplotlib", vmin=0.0, vmax=1.0
    )
    figure.show()
    np.testing.assert_allclose(
        _pixel(figure, figure[0, 0], 15.0, 5.0), to_rgba("lightgray")[:3], atol=2 / 255
    )
    np.testing.assert_allclose(
        _pixel(figure, figure[0, 0], 5.0, 5.0),
        colormaps["viridis"](0.0)[:3],
        atol=2 / 255,
    )


def test_three_dimensional_fields_draw_points_whose_opacity_follows_value():
    env = Environment.from_samples(
        np.random.default_rng(0).uniform(0, 10, (3000, 3)), bin_size=2.0
    )
    field = np.linspace(0.0, 1.0, env.n_bins)
    figure = env.plot_field(field, backend="fastplotlib", colorbar=False)
    (points,) = [
        g for g in figure[0, 0].graphics if type(g).__name__ == "ScatterGraphic"
    ]
    alpha = points.colors[:][:, 3]
    assert alpha[np.argmax(field)] > 0.99 > 0.1 > alpha[np.argmin(field)]


def test_three_dimensional_landmarks_and_regions_are_drawn():
    from neurospatial.regions import Regions

    env = Environment.from_samples(
        np.random.default_rng(0).uniform(0, 10, (3000, 3)), bin_size=2.0
    )
    regions = Regions()
    regions.add("nest", point=[2.0, 2.0, 2.0])
    env = env.with_regions(regions).with_landmarks(
        Landmarks([Landmark("feeder", "point", [8.0, 8.0, 9.0])])
    )
    figure = env.plot(backend="fastplotlib")
    texts = [g.text for g in figure[0, 0].graphics if hasattr(g, "text")]
    assert {"nest", "feeder"} <= set(texts)
    points = [
        g.data[:].tolist()
        for g in figure[0, 0].graphics
        if type(g).__name__ == "ScatterGraphic" and len(g.data[:]) == 1
    ]
    assert [8.0, 8.0, 9.0] in [p[0] for p in points]  # z kept


def test_matplotlib_options_are_refused_and_unknown_backends_named(apparatus):
    with pytest.raises(TypeError, match="Matplotlib-only options: scale_bar"):
        apparatus.plot(backend="fastplotlib", scale_bar=True)
    with pytest.raises(TypeError, match="ax"):
        apparatus.plot_field(
            np.zeros(apparatus.n_bins), ax=object(), backend="fastplotlib"
        )
    # Options at a non-default value count, even when falsy.
    with pytest.raises(TypeError, match="alpha, rasterized"):
        apparatus.plot_field(
            np.zeros(apparatus.n_bins), backend="fastplotlib", rasterized=False, alpha=0
        )
    with pytest.raises(ValueError, match="Unknown backend 'plotly'"):
        apparatus.plot(backend="plotly")


@pytest.fixture
def walk(apparatus):
    n = 120
    t = np.arange(n) / 30.0
    path = np.c_[20 + 10 * np.cos(t), 20 + 10 * np.sin(t)]
    fields = np.exp(
        -np.sum((apparatus.bin_centers[None] - path[:, None]) ** 2, axis=2) / 30
    )
    return t, path, fields


def test_animation_moves_the_overlays_with_the_frame(apparatus, walk):
    t, path, fields = walk
    events = EventOverlay(
        event_times=t[[10, 50]], event_positions=path[[10, 50]], decay_frames=5
    )
    animation = apparatus.animate_fields(
        fields,
        frame_times=t,
        backend="fastplotlib",
        overlays=[PositionOverlay(positions=path, times=t, trail_length=10), events],
    )
    assert isinstance(animation, FastplotlibAnimation)
    assert animation._fields is fields  # an array is read a frame at a time
    animation.show()
    position, event_layer = animation._layers
    ((_, _, _, event_graphic),) = event_layer.kinds
    for frame, shown in ((10, True), (14, True), (16, False), (50, True)):
        animation.frame = frame
        _render(animation)
        np.testing.assert_allclose(position.marker.data[0, :2], path[frame], rtol=1e-6)
        assert (event_graphic.colors[:][:, 3] > 0).any() == shown
    assert animation._plot.title.text.endswith(f"t = {t[50]:.2f} s")


def test_a_frame_is_shown_until_the_next_one_starts(apparatus, walk):
    t, _, fields = walk
    animation = apparatus.animate_fields(
        fields, frame_times=t, backend="fastplotlib", speed=1.0
    )
    for time, frame in ((t[5], 5), (t[5] + 0.9 * (t[6] - t[5]), 5), (-1.0, 0)):
        animation.widget.indices = {"time": time}
        assert animation.frame == frame
    animation.frame = 10**6
    assert animation.frame == animation.n_frames - 1
    # Playing steps one frame per 1 / fps seconds of the recording's clock.
    assert animation.widget.ui_sliders._fps["time"] == 30  # 30 Hz frames at speed 1
    assert animation.widget.ranges["time"].step == pytest.approx(t[1] - t[0])


def test_a_shown_animation_draws_without_warnings(apparatus, walk):
    t, _, fields = walk
    animation = apparatus.animate_fields(fields, frame_times=t, backend="fastplotlib")
    animation.show()
    # Recorded, not raised: the canvas catches errors raised while drawing.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")  # fastplotlib calls _advance(figure)
        for _ in range(3):
            animation.figure.canvas.draw()
    assert [str(w.message) for w in caught] == []


def test_an_event_marker_without_a_gpu_shape_warns(apparatus, walk):
    t, path, fields = walk
    events = EventOverlay(event_times=t[[10]], event_positions=path[[10]], markers="p")
    with pytest.warns(UserWarning, match="no marker 'p'"):
        apparatus.animate_fields(
            fields, frame_times=t, backend="fastplotlib", overlays=[events]
        )


def test_object_vectors_are_skipped_with_a_warning(apparatus, walk):
    from neurospatial.animation import ObjectVectorOverlay

    t, path, fields = walk
    vectors = ObjectVectorOverlay(
        object_positions=np.array([[10.0, 10.0]]), animal_positions=path, times=t
    )
    with pytest.warns(UserWarning, match="does not draw object-vector overlays"):
        apparatus.animate_fields(
            fields, frame_times=t, backend="fastplotlib", overlays=[vectors]
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        apparatus.animate_fields(fields, frame_times=t, backend="fastplotlib")


def test_time_series_are_drawn_beside_the_field_on_its_clock(apparatus, walk):
    t, _, fields = walk
    series_times = np.arange(0.0, 4.0, 0.01)  # 100 Hz, not the frames' 30 Hz
    speed = TimeSeriesOverlay(
        data=np.sin(series_times),
        times=series_times,
        label="speed",
        window_seconds=1.0,
    )
    accel = TimeSeriesOverlay(
        data=np.cos(series_times),
        times=series_times,
        label="accel",
        window_seconds=1.0,
        group="kinematics",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # time series coverage notes
        animation = apparatus.animate_fields(
            fields, frame_times=t, backend="fastplotlib", overlays=[speed, accel]
        )
    animation.show()
    animation.figure.canvas.draw()
    assert animation.frame == 0  # fitting the series' view does not move the time
    assert {s.title.text for s in animation.figure if s.name != "field"} == {
        "speed",
        "accel",
    }
    animation.frame = 50
    _render(animation)
    (subplot,) = [s for s in animation.figure if s.title.text == "speed"]
    low, high = animation.widget[subplot]["speed"].display_range
    assert low < t[50] < high
    assert high - low == pytest.approx(1.0, abs=0.02)
    low, high = subplot.y_range  # the series' whole range, with a margin
    assert low < np.sin(series_times).min() and high > np.sin(series_times).max()


@pytest.fixture
def video(walk):
    """Frames whose top rows are red, and a calibration that flips y."""
    t, _, _ = walk
    frames = np.zeros((len(t), 20, 40, 3), dtype=np.uint8)
    frames[:, :10, :, 0] = 255
    # x = column, y = 40 - 2 * row: the camera's rows run down the arena.
    pixels_to_cm = Affine2D(np.array([[1.0, 0, 0], [0, -2.0, 40.0], [0, 0, 1]]))
    return frames, VideoCalibration(pixels_to_cm, frame_size_px=(40, 20))


@pytest.mark.parametrize(("z_order", "above"), [("above", True), ("below", False)])
def test_video_is_placed_through_its_calibration_under_the_walls(
    apparatus, walk, video, z_order, above
):
    t, _, _ = walk
    frames, calibration = video
    overlay = VideoOverlay(
        source=frames, times=t, calibration=calibration, alpha=1.0, z_order=z_order
    )
    animation = apparatus.animate_fields(
        np.zeros((len(t), apparatus.n_bins)),
        frame_times=t,
        backend="fastplotlib",
        vmin=0.0,
        vmax=1.0,
        overlays=[overlay],
    )
    animation.show()
    plot = animation._plot
    field = np.asarray(colormaps["viridis"](0.0)[:3])
    red, black = np.array([1.0, 0, 0]), np.zeros(3)
    top, bottom = (
        _pixel(animation.figure, plot, 10, 35),
        _pixel(animation.figure, plot, 10, 5),
    )
    np.testing.assert_allclose(top, red if above else field, atol=2 / 255)
    np.testing.assert_allclose(bottom, black if above else field, atol=2 / 255)
    wall = _pixel(animation.figure, plot, 20, 10)  # the divider, red too
    np.testing.assert_allclose(wall, to_rgba("firebrick")[:3], atol=3 / 255)
    assert plot.camera.local.scale_y > 0  # y still points up


def test_dry_run_reports_without_drawing(apparatus, walk, capsys):
    t, _, fields = walk
    result = apparatus.animate_fields(
        fields, frame_times=t, backend="fastplotlib", dry_run=True
    )
    assert result is None
    assert f"{len(t):,} frames of {apparatus.n_bins:,} bins" in capsys.readouterr().err
