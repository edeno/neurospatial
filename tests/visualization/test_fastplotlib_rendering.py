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
    Animation,
    AnimationField,
    AnimationView,
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
    view = apparatus.animate_fields(
        fields,
        frame_times=t,
        backend="fastplotlib",
        overlays=[PositionOverlay(positions=path, times=t, trail_length=10), events],
    )
    assert isinstance(view, AnimationView)
    animation = view.native
    assert isinstance(animation, FastplotlibAnimation)
    position, event_layer = animation._layers
    ((_, _, (event_graphic,)),) = event_layer.kinds.values()
    for frame, shown in ((10, True), (14, True), (16, False), (50, True)):
        animation.frame = frame
        _render(animation)
        np.testing.assert_allclose(
            position.markers[0].data[0, :2], path[frame], rtol=1e-6
        )
        assert (event_graphic.colors[:][:, 3] > 0).any() == shown
    assert animation.status.endswith(f"t = {t[50]:.2f} s")
    assert view.time == pytest.approx(t[50])  # the view reads the native clock
    view.seek(t[10])
    _render(animation)
    np.testing.assert_allclose(position.markers[0].data[0, :2], path[10], rtol=1e-6)


def test_a_frame_is_shown_until_the_next_one_starts(apparatus, walk):
    t, _, fields = walk
    animation = apparatus.animate_fields(
        fields, frame_times=t, backend="fastplotlib", speed=1.0
    ).native
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
    animation = apparatus.animate_fields(
        fields, frame_times=t, backend="fastplotlib"
    ).native
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


def test_object_vectors_point_at_the_objects_and_fade_with_the_rate(apparatus, walk):
    from neurospatial.animation import ObjectVectorOverlay

    t, path, fields = walk
    rates = np.linspace(0.0, 10.0, len(t))
    objects = np.array([[10.0, 10.0], [30.0, 35.0]])
    vectors = ObjectVectorOverlay(
        object_positions=objects, animal_positions=path, times=t, firing_rates=rates
    )
    animation = apparatus.animate_fields(
        fields, frame_times=t, backend="fastplotlib", overlays=[vectors]
    ).native
    (layer,) = animation._layers
    alphas = {}
    for frame in (0, len(t) - 1):
        animation.frame = frame
        _render(animation)
        segments = layer.lines[0].data[:]
        np.testing.assert_allclose(segments[0::3, :2], path[[frame, frame]], rtol=1e-6)
        np.testing.assert_allclose(segments[1::3, :2], objects)
        alphas[frame] = layer.lines[0].colors.a
    assert alphas[len(t) - 1] == pytest.approx(1.0)  # the highest rate
    assert 0.0 < alphas[0] < 0.2  # zero rate: faint, still drawn


def test_the_field_recolors_and_is_read_one_frame_at_a_time(apparatus, walk):
    t, _, fields = walk

    reads = []

    class CountingFrames(np.ndarray):
        """An array that records the frames read from it."""

        def __getitem__(self, key):
            reads.append(key)
            return super().__getitem__(key)

    counted = fields.view(CountingFrames)
    animation = apparatus.animate_fields(
        counted, frame_times=t, backend="fastplotlib"
    ).native
    reads.clear()
    before = animation._field_graphics[0].graphic.mapcoords[:].copy()
    animation.frame = 60
    _render(animation)
    assert reads == [60]
    assert not np.array_equal(animation._field_graphics[0].graphic.mapcoords[:], before)
    np.testing.assert_array_equal(animation._field_graphics[0].values, fields[60])


def test_the_right_click_settings_pause_the_field(apparatus, walk):
    t, path, fields = walk
    animation = apparatus.animate_fields(
        fields,
        frame_times=t,
        backend="fastplotlib",
        overlays=[PositionOverlay(positions=path, times=t)],
    ).native
    (frames,) = animation.widget[animation._plot].nd_graphics
    assert (
        frames.graphic is animation._field_graphics[0].graphic
    )  # the menu draws its settings
    frames.pause = True
    animation.frame = 40
    for graphic in animation.widget.ndgraphics:
        animation.widget.indices._schedule_fetch(graphic)  # skipped while paused
    np.testing.assert_allclose(
        animation._layers[0].markers[0].data[0, :2], path[0], rtol=1e-6
    )


def test_hovering_a_bin_shows_its_position_and_value(apparatus):
    field = np.arange(apparatus.n_bins, dtype=float)
    field[3] = np.nan
    figure = apparatus.plot_field(field, backend="fastplotlib")
    (cells,) = [g for g in figure[0, 0].graphics if type(g).__name__ == "MeshGraphic"]
    triangles = cells.indices[:]
    owner = 7
    (face,) = np.flatnonzero(
        np.all(
            np.isin(triangles, np.flatnonzero(cells_vertex_bin(apparatus) == owner)), 1
        )
    )[:1]
    text = cells.tooltip_format({"face_index": face})
    x, y = apparatus.bin_centers[owner]
    assert text.splitlines() == [
        "bin 7",
        f"X Position (cm): {x:.4g}",
        f"Y Position (cm): {y:.4g}",
        "value: 7",
    ]
    (face,) = np.flatnonzero(
        np.all(np.isin(triangles, np.flatnonzero(cells_vertex_bin(apparatus) == 3)), 1)
    )[:1]
    assert cells.tooltip_format({"face_index": face}).endswith("value: missing")


def cells_vertex_bin(env):
    from neurospatial.environment._fastplotlib import cell_mesh

    return cell_mesh(env).vertex_bin


def _name(subplot):
    """A time panel's title without the window it shows."""
    return subplot.title.text.split("  \u00b7  ")[0]


def test_time_series_are_drawn_beside_the_field_on_its_clock(apparatus, walk):
    t, path, fields = walk
    series_times = np.arange(0.0, 4.0, 0.0005)  # 2 kHz, not the frames' 30 Hz
    speed = TimeSeriesOverlay(
        data=np.sin(series_times),
        times=series_times,
        label="speed",
        cursor_color="orange",
    )
    accel = TimeSeriesOverlay(
        data=np.cos(series_times),
        times=series_times,
        label="accel",
        group="kinematics",
    )
    rewards = EventOverlay(
        event_times={"reward": t[[20, 70]]},
        event_positions={"reward": path[[20, 70]]},
        colors={"reward": "gold"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # time series coverage notes
        animation = apparatus.animate_fields(
            fields,
            frame_times=t,
            backend="fastplotlib",
            window_seconds=1.0,  # every time panel's
            overlays=[speed, accel, rewards],
        ).native
    animation.show()
    animation.figure.canvas.draw()
    assert animation.frame == 0  # fitting the series' view does not move the time
    panels = {_name(s): s for s in animation.figure if s.name != "field"}
    assert set(panels) == {"speed", "accel"}
    assert panels["speed"].title.text.endswith("1 s window")
    animation.frame = 50
    _render(animation)
    subplot = panels["speed"]
    low, high = subplot.x_range
    assert low < t[50] < high
    assert high - low == pytest.approx(1.0, abs=0.02)
    # Every sample in the window is drawn: 2,000.
    (trace,) = [series for series in animation._series if series.subplot is subplot]
    (line,) = trace.lines
    x = line.data[:][:, 0]
    x = x[np.isfinite(x)]
    assert len(np.unique(x)) >= 1999
    low, high = subplot.y_range  # the series' whole range, with a margin
    assert low < np.sin(series_times).min() and high > np.sin(series_times).max()
    # Its ends are labeled, and 0 between: a short panel shows its scale.
    lowest, highest = np.sin(series_times).min(), np.sin(series_times).max()
    ticks = subplot.axes.y.ticks
    np.testing.assert_allclose(list(ticks), [lowest, 0.0, highest])
    assert list(ticks.values()) == [f"{lowest:.3g}", "0", f"{highest:.3g}"]
    (marks,) = [g for g in subplot.graphics if g.name == "reward times"]
    np.testing.assert_allclose(np.sort(marks.data[:]), t[[20, 70]], rtol=1e-5)
    (cursor,) = subplot.selectors
    np.testing.assert_allclose(cursor.edge_color, to_rgba("orange"))


def test_session_timestamps_keep_their_precision(apparatus, walk):
    t, _, fields = walk
    session = 1.7e9 + t  # seconds since 1970, as some acquisition systems save
    series_times = 1.7e9 + np.arange(0.0, 4.0, 0.01)
    speed = TimeSeriesOverlay(
        data=np.sin(series_times - 1.7e9),
        times=series_times,
        label="speed",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        animation = apparatus.animate_fields(
            fields, frame_times=session, backend="fastplotlib", overlays=[speed]
        ).native
    animation.frame = 60
    _render(animation)
    assert animation.widget.indices["time"] == pytest.approx(t[60])  # from frame 0
    (series,) = animation._series
    (line,) = series.lines
    x = line.data[:][:, 0]
    assert len(np.unique(x[np.isfinite(x)])) >= 99  # not collapsed by float32
    assert animation.status == f"t = {t[60]:.2f} s from 1700000000.00 s"


def test_a_series_without_values_at_the_start_is_drawn_from_its_first_value(
    apparatus, walk
):
    t, _, fields = walk
    series_times = np.arange(0.0, 4.0, 0.01)
    speed = np.sin(series_times)
    speed[series_times < 1.5] = np.nan  # tracking starts late
    overlays = [
        TimeSeriesOverlay(data=speed, times=series_times, label="speed"),
        TimeSeriesOverlay(data=np.full(5, np.nan), times=np.arange(5.0), label="empty"),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        animation = apparatus.animate_fields(
            fields, frame_times=t, backend="fastplotlib", overlays=overlays
        ).native
    assert animation.frame == 0
    titles = {_name(s) for s in animation.figure if s.name != "field"}
    # A signal without values draws an empty panel, as in every view.
    assert titles == {"speed", "empty"}
    animation.frame = 60  # t = 2 s: speed is tracked
    _render(animation)
    speed = next(
        series for series in animation._series if series.group[0].label == "speed"
    )
    x = speed.lines[0].data[:][:, 0]
    assert np.nanmin(x) == pytest.approx(1.0, abs=0.01)  # the window's left


def test_uneven_frames_are_each_reached_and_gaps_warned(apparatus, walk):
    _, _, fields = walk
    # Two bursts of 2 ms decoding bins, 5 s apart.
    times = np.concatenate([np.arange(60) * 0.002, 5.0 + np.arange(60) * 0.002])
    with pytest.warns(UserWarning, match="unevenly spaced"):
        animation = apparatus.animate_fields(
            fields, frame_times=times, backend="fastplotlib"
        ).native
    step = animation.widget.ranges["time"].step
    assert step <= np.diff(times).min()  # stepping never passes over a frame
    jittered = np.cumsum(np.random.default_rng(0).uniform(0.9, 1.1, len(times)) / 30.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # ordinary jitter is not warned about
        animation = apparatus.animate_fields(
            fields, frame_times=jittered, backend="fastplotlib"
        ).native
    step = animation.widget.ranges["time"].step
    reached = {
        animation._frame_at(k * step) for k in range(int(1 + jittered[-1] / step))
    }
    assert reached == set(range(len(jittered)))


@pytest.fixture(scope="module")
def track():
    """Two 30 cm segments, 10 cm apart along the track: bins 0-2, then 3-5."""
    return Environment.linear_track(
        node_positions=[[0.0, 0.0], [30.0, 0.0], [30.0, 30.0]],
        bin_size=10.0,
        edge_spacing=10.0,
        units="cm",
    )


@pytest.fixture
def decoded(track):
    """One bin holds all of each frame's field, as a sharp posterior would.

    Frames come at 10 Hz in two runs, 3 s apart.
    """
    times = np.concatenate([np.arange(20) * 0.1, 5.0 + np.arange(20) * 0.1])
    peak = np.arange(len(times)) % track.n_bins
    fields = np.zeros((len(times), track.n_bins))
    fields[np.arange(len(times)), peak] = 1.0
    return times, peak, fields


def test_a_track_shows_its_fields_against_position_and_time(track, decoded):
    times, peak, fields = decoded
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the 3 s gap is warned about
        animation = track.animate_fields(
            fields,
            frame_times=times,
            backend="fastplotlib",
            vmin=0.0,
            vmax=1.0,
            window_seconds=1.0,
            overlays=[PositionOverlay(positions=track.bin_centers[peak], times=times)],
        ).native
    animation.show()
    panel = animation.figure["track"]
    high, low = colormaps["viridis"](1.0)[:3], colormaps["viridis"](0.0)[:3]
    animation.frame = 15  # its field is all in bin 3, 40-50 cm along the track
    _render(animation)
    middle = times[15] + 0.05
    np.testing.assert_allclose(
        _pixel(animation.figure, panel, middle, 45.0), high, atol=2 / 255
    )
    np.testing.assert_allclose(
        _pixel(animation.figure, panel, middle, 15.0), low, atol=2 / 255
    )
    # Between the segments there is no bin.
    np.testing.assert_allclose(
        _pixel(animation.figure, panel, middle, 35.0), 0.0, atol=0.05
    )
    panel_layer = animation._tracks[0]
    (column,) = np.flatnonzero(np.isclose(panel_layer.edges[:, 0], times[15]))
    points = panel_layer.paths.data[:][:, :2]
    at = np.flatnonzero(np.isclose(points[:, 0], times[15]))
    np.testing.assert_allclose(points[at], [[times[15], 45.0]] * len(at), rtol=1e-6)
    face = 2 * (column * track.n_bins + 3)
    assert animation._tracks[0].mesh.tooltip_format(
        {"face_index": face}
    ).splitlines() == [
        f"t = {times[15]:.3f} s",
        "bin 3, position along track (cm): 45",
        "value: 1",
    ]
    # A row is held until the next one, but not across the 3 s gap, which is
    # longer than max_gap: the last row before it is not drawn into the gap.
    animation.frame = 19
    _render(animation)
    np.testing.assert_allclose(  # frame 18, bin 0 (0-10 cm), held 0.1 s
        _pixel(animation.figure, panel, times[18] + 0.05, 5.0), high, atol=2 / 255
    )
    for after in (0.05, 0.3):  # frame 19, bin 1 (10-20 cm)
        np.testing.assert_allclose(
            _pixel(animation.figure, panel, times[19] + after, 15.0), 0.0, atol=0.05
        )
    animation._tracks[0].view.cursor.selection = times[12]  # dragging the cursor
    assert animation.widget.indices["time"] == pytest.approx(times[12])


def test_a_raster_draws_each_spike_in_its_units_row_on_the_clock(apparatus, walk):
    from neurospatial.animation import SpikeRasterOverlay

    t, path, fields = walk
    session = 100.0 + t  # the clock counts from the first frame
    spikes = [100.0 + np.array([0.5, 1.0]), 100.0 + np.array([2.0]), np.array([])]
    raster = SpikeRasterOverlay(
        spikes,
        order=[1, 2, 0],
        unit_ids=["CA1-a", "CA1-b", "CA1-c"],
    )
    rewards = EventOverlay(
        event_times={"reward": session[[20]]}, event_positions={"reward": path[[20]]}
    )
    animation = apparatus.animate_fields(
        fields,
        frame_times=session,
        backend="fastplotlib",
        window_seconds=1.0,
        overlays=[raster, rewards],
    ).native
    panel = animation.figure["raster 0"]
    (layer,) = animation._rasters
    animation.frame = 30  # t = 1 s: spikes from 0.5 to 1.5 s are in the window
    _render(animation)
    # Unit 0 in the top row, unit 1 (at 2 s) in the bottom one: a tick a spike.
    ticks = layer.ticks.data[:][:, :2]
    np.testing.assert_allclose(ticks[0:6:3], [[0.5, 1.6], [1.0, 1.6]], rtol=1e-6)
    np.testing.assert_allclose(ticks[1:6:3], [[0.5, 2.4], [1.0, 2.4]], rtol=1e-6)
    assert np.isnan(ticks[6:]).all()  # nothing else in the window
    assert layer.describe({"vertex_index": 1}).splitlines() == [
        "unit CA1-a",
        "t = 0.5000 s",  # from the first frame, at 100 s
    ]
    low, high = panel.y_range
    assert high == pytest.approx(2.5) and low < -0.5  # room below for the time axis
    (marks,) = [g for g in panel.graphics if g.name == "reward times"]
    np.testing.assert_allclose(marks.data[:], t[[20]], rtol=1e-5)
    animation.frame = 60
    _render(animation)
    assert panel.x_range == pytest.approx((t[60] - 0.5, t[60] + 0.5))
    np.testing.assert_allclose(
        layer.ticks.data[:2, :2], [[2.0, -0.4], [2.0, 0.4]], rtol=1e-6
    )
    layer.view.cursor.selection = t[30]  # dragging the cursor
    assert animation.widget.indices["time"] == pytest.approx(t[30])


def _click(figure, subplot, x, y):
    """Click at world (x, y) of ``subplot``, as a person would."""
    import time

    column, row = subplot.map_world_to_screen((x, y, 0))[:2]
    for kind, buttons in (("pointer_down", (1,)), ("pointer_up", ())):
        figure.canvas.submit_event(
            {
                "event_type": kind,
                "x": float(column),
                "y": float(row),
                "button": 1,
                "buttons": buttons,
                "modifiers": (),
                "ntouches": 0,
                "touches": {},
                "time_stamp": time.perf_counter(),
            }
        )
        figure.canvas._process_events()  # what the event loop would do


def test_clicking_a_units_row_shows_its_rate_map(apparatus, walk, track, decoded):
    from neurospatial.animation import SpikeRasterOverlay

    t, _, fields = walk
    spikes = [np.array([0.5, 1.0]), np.array([2.0]), np.array([1.2])]
    rates = np.arange(3 * apparatus.n_bins, dtype=float).reshape(3, -1)
    rates[1, :5] = np.nan  # bins without an estimate
    raster = SpikeRasterOverlay(spikes, order=[1, 2, 0], rates=rates)
    animation = apparatus.animate_fields(
        fields, frame_times=t, backend="fastplotlib", overlays=[raster]
    ).native
    animation.show()
    animation.frame = 30  # t = 1 s
    _render(animation)
    animation.figure.canvas.draw()
    (layer,) = animation._rasters
    unit_panel = animation.figure["unit"]
    assert "click" in unit_panel.title.text
    # Between the ticks of the bottom row, unit 1's.
    _click(animation.figure, animation.figure["raster 0"], t[30] + 0.3, -0.3)
    assert (
        unit_panel.title.text == f"unit 1: rate map, 0 to {np.nanmax(rates[1]):.3g} Hz"
    )
    np.testing.assert_array_equal(layer.rates.field.values, rates[1])
    colors = layer.ticks.colors[:]
    np.testing.assert_allclose(colors[0:3], np.tile(to_rgba("orange"), (3, 1)))
    np.testing.assert_allclose(colors[3:9], np.tile(to_rgba("white"), (6, 1)))
    _click(animation.figure, animation._plot, 20.0, 20.0)  # the field: no change
    assert unit_panel.title.text.startswith("unit 1:")
    # The chosen row is striped, not only recolored.
    alpha = layer.bands.colors[:][:, 3].reshape(3, 4)[:, 0]
    assert alpha[0] > 100 * alpha[1] and alpha[1] == alpha[2] > 0
    # Hovering anywhere in a row names its unit: the band is picked.
    raster_panel = animation.figure["raster 0"]
    x, y = raster_panel.map_world_to_screen((t[30] + 0.4, 2.0, 0))[:2]
    animation.figure.canvas.draw()
    info = animation.figure.renderer.get_pick_info((x, y))
    assert info["world_object"] == layer.bands.world_object  # a weak proxy
    assert layer.describe_row(info) == "unit 0"  # row 2: unit 0
    # The keyboard chooses units too.
    _press(animation.figure, "ArrowUp")
    assert unit_panel.title.text.startswith("unit 2:")  # row 1
    _press(animation.figure, "ArrowDown")
    _press(animation.figure, "ArrowDown")  # stays on the bottom row
    assert unit_panel.title.text.startswith("unit 1:")
    _press(animation.figure, "Escape")
    assert "click a unit's row" in unit_panel.title.text
    assert (layer.bands.colors[:][:, 3] < 1e-3).all()

    # On a track, the rate map is drawn against position along the track.
    times, _, fields = decoded
    rates = np.tile(np.arange(track.n_bins, dtype=float), (2, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the gap between runs
        animation = track.animate_fields(
            fields,
            frame_times=times,
            backend="fastplotlib",
            overlays=[SpikeRasterOverlay([times[:3], times[3:5]], rates=rates)],
        ).native
    (layer,) = animation._rasters
    layer.rates.show(layer, 0)
    np.testing.assert_allclose(
        layer.rates.line.data[:][:, :2],
        [[5, 0], [15, 1], [25, 2], [45, 3], [55, 4], [65, 5]],
    )


def _press(figure, key, modifiers=()):
    """Press ``key`` with the figure focused."""
    import time

    figure.canvas.submit_event(
        {
            "event_type": "key_down",
            "key": key,
            "modifiers": modifiers,
            "time_stamp": time.perf_counter(),
        }
    )
    figure.canvas._process_events()  # what the event loop would do


def test_epochs_are_marked_named_and_jumped_between(track, decoded):
    from neurospatial.animation import EpochOverlay, SpikeRasterOverlay

    times, _, fields = decoded  # 10 Hz in [0, 1.9] and [5, 6.9]
    ripples = EpochOverlay([[0.45, 0.65], [5.2, 5.4]], label="ripple", color="cyan")
    reward = EventOverlay(
        event_times=np.array([1.2]), event_positions=np.array([[5.0, 0.0]])
    )
    series = TimeSeriesOverlay(data=np.sin(times), times=times, label="lfp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the gap between runs
        animation = track.animate_fields(
            fields,
            frame_times=times,
            backend="fastplotlib",
            overlays=[
                ripples,
                reward,
                series,
                SpikeRasterOverlay([times[::3]]),
            ],
        ).native
    animation.show()
    figure = animation.figure
    # Over the track's fields, a strip above them rather than shading.
    (strip,) = [g for g in figure["track"].graphics if g.name == "ripple epochs"]
    corners = strip.positions[:]
    np.testing.assert_allclose(
        [corners[:4, 0].min(), corners[:4, 0].max()], [0.45, 0.65], atol=1e-6
    )
    assert corners[:, 1].min() > animation._tracks[0].upper.max()
    # ...named at the panel's edge.
    assert "ripple" in [label.text for label, _ in animation._tracks[0].labels]
    for panel in ("raster 0", "series 0"):
        (bands,) = [g for g in figure[panel].graphics if g.name == "ripple epochs"]
        x = bands.positions[:][:, 0]
        np.testing.assert_allclose([x[:4].min(), x[:4].max()], [0.45, 0.65], atol=1e-6)
    assert "N/Shift+N" in animation.key_hint
    marks = []
    for _ in range(4):
        _press(figure, "n")
        marks.append(animation.widget.indices["time"])
        _render(animation)
        if marks[-1] == pytest.approx(0.45):
            assert animation.status.endswith("ripple 1 of 2")
    np.testing.assert_allclose(marks, [0.45, 1.2, 5.2, 5.2])  # stays at the last
    _press(figure, "N", ("Shift",))
    assert animation.widget.indices["time"] == pytest.approx(1.2)
    _render(animation)
    assert "ripple" not in animation.status


def test_keys_step_frames_play_and_resize_the_time_windows(track, decoded):
    times, _, fields = decoded
    series = TimeSeriesOverlay(data=np.sin(times), times=times, label="lfp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        animation = track.animate_fields(
            fields,
            frame_times=times,
            backend="fastplotlib",
            window_seconds=1.0,
            vmin=0.0,
            vmax=1.0,
            overlays=[series],
        ).native
    animation.show()
    figure = animation.figure
    # The keys are named above the panels; N only where there is somewhere to go.
    assert "Space play" in animation.key_hint and "N/Shift+N" not in animation.key_hint
    animation.frame = 19
    _press(figure, "ArrowRight")  # across the 3 s gap: the next frame
    assert animation.frame == 20
    _press(figure, "ArrowLeft")
    assert animation.frame == 19
    _press(figure, " ")
    assert animation.widget.ui_sliders._playing["time"]
    _press(figure, " ")
    assert not animation.widget.ui_sliders._playing["time"]

    animation.frame = 30  # 6.0 s, bin 0 holds the field
    _press(figure, "]")
    assert animation.window_seconds == pytest.approx(2.0)
    # The wider window draws frames a second away, in every time panel.
    _render(animation)
    low, high = figure["series 0"].x_range
    assert high - low == pytest.approx(2.0)
    far = times[21] + 0.05  # 0.85 s back, past a 1 s window; bin 3 holds it
    np.testing.assert_allclose(
        _pixel(figure, figure["track"], far, 45.0),
        colormaps["viridis"](1.0)[:3],
        atol=2 / 255,
    )
    np.testing.assert_allclose(  # and on the right: frame 38, bin 2
        _pixel(figure, figure["track"], times[38] + 0.05, 25.0),
        colormaps["viridis"](1.0)[:3],
        atol=2 / 255,
    )
    _press(figure, "[")
    _press(figure, "[")
    assert animation.window_seconds == pytest.approx(0.5)


def test_panels_drawn_against_time_share_one_window(track, decoded):
    from neurospatial.animation import SpikeRasterOverlay

    times, _, fields = decoded
    series = TimeSeriesOverlay(data=np.sin(times), times=times, label="lfp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        animation = track.animate_fields(
            fields,
            frame_times=times,
            backend="fastplotlib",
            window_seconds=1.5,
            overlays=[series, SpikeRasterOverlay([times[::2]])],
        ).native
    animation.show()
    names = ["track", "raster 0", "series 0"]
    for window in (1.5, 3.0):
        animation.frame = 30
        _render(animation)
        for name in names:
            low, high = animation.figure[name].x_range
            assert (low, high) == pytest.approx(
                (times[30] - window / 2, times[30] + window / 2), abs=1e-4
            ), name
            assert animation.figure[name].title.text.endswith(f"{window:.3g} s window")
        _press(animation.figure, "]")


def test_the_data_stays_above_the_time_axis(track, decoded):
    from neurospatial.animation import SpikeRasterOverlay

    times, _, fields = decoded
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        animation = track.animate_fields(
            fields,
            frame_times=times,
            backend="fastplotlib",
            overlays=[
                TimeSeriesOverlay(data=np.sin(times), times=times, label="lfp"),
                SpikeRasterOverlay(
                    [times[::3]], rates=np.arange(track.n_bins, dtype=float)[None]
                ),
            ],
        ).native
    animation.show()
    animation._unit.show(animation._rasters[0], 0)
    for _ in range(2):  # tick labels are measured as they are first drawn
        animation.figure.canvas.draw()
    for subplot, lowest in (
        (animation.figure["series 0"], -1.0),
        (animation.figure["unit"], 0.0),
    ):
        _, top, _, height = subplot.viewport.rect
        axis = top + height - subplot.axes._get_label_margins()[0] - 4
        assert subplot.map_world_to_screen((1.0, lowest, 0))[1] <= axis + 1


def test_a_tracks_segments_are_bounded_and_named(track, decoded):
    times, _, fields = decoded
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        animation = track.animate_fields(
            fields[:20], frame_times=times[:20], backend="fastplotlib"
        ).native
    panel = animation.figure["track"]
    # 10 cm of spacing separates the segments already: no line between them.
    lines = [g for g in panel.graphics if type(g).__name__ == "InfLineGraphic"]
    assert [g.data[:].tolist() for g in lines] == [[0.0]]  # the cursor's edge
    names = [(label.text, middle) for label, middle in animation._tracks[0].labels]
    assert names == [("0\u20131", 15.0), ("1\u20132", 55.0)]
    animation.frame = 10
    _render(animation)
    (label, _), _ = animation._tracks[0].labels
    assert label.offset[0] == pytest.approx(times[10] + 1.0)  # the window's right

    # Segments that meet get a line, with a dark edge.
    joined = Environment.linear_track(
        node_positions=[[0.0, 0.0], [30.0, 0.0], [30.0, 30.0]], bin_size=10.0
    )
    animation = joined.animate_fields(
        np.zeros((5, joined.n_bins)), frame_times=np.arange(5.0), backend="fastplotlib"
    ).native
    lines = [
        g
        for g in animation.figure["track"].graphics
        if type(g).__name__ == "InfLineGraphic" and g.data[:].tolist() == [30.0]
    ]
    assert len(lines) == 2  # the edge and the line


def test_several_fields_are_shown_side_by_side_on_one_scale(
    apparatus, walk, track, decoded
):
    t, path, fields = walk
    doubled = 2 * fields
    animation = apparatus.animate_fields(
        [list(fields), list(doubled)],
        frame_times=t,
        backend="fastplotlib",
        layer_names=["posterior", "likelihood"],
        overlays=[PositionOverlay(positions=path, times=t)],
    ).native
    first, second = animation._field_graphics
    assert [p.name for p in animation._plots] == ["field 0", "field 1"]
    animation.frame = 40
    _render(animation)
    np.testing.assert_allclose(first.values, fields[40])
    np.testing.assert_allclose(second.values, doubled[40])
    # One scale, so they compare: it spans the doubled field.
    assert (first.scale.vmin, first.scale.vmax) == (
        second.scale.vmin,
        second.scale.vmax,
    )
    assert first.scale.vmax > fields.max()
    assert animation._plot.title.text == "posterior"
    assert animation.status.startswith("t = ")
    assert animation.figure["field 1"].title.text == "likelihood"
    # The animal is drawn over both.
    (position,) = animation._layers
    assert len(position.markers) == 2
    for marker in position.markers:
        np.testing.assert_allclose(marker.data[0, :2], path[40], rtol=1e-6)

    times, _, fields = decoded
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        animation = track.animate_fields(
            [list(fields), list(fields[::-1])],
            frame_times=times,
            backend="fastplotlib",
            layout="vertical",
        ).native
    assert len(animation._tracks) == 2
    assert animation.figure["track 1"].title.text.startswith("field 2: ")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the gap between runs
        with pytest.raises(ValueError, match="one name per field"):
            track.animate_fields(
                [list(fields), list(fields)],
                frame_times=times,
                backend="fastplotlib",
                layer_names=["only one"],
            )
        with pytest.raises(ValueError, match="layout must be"):
            track.animate_fields(
                [list(fields), list(fields)],
                frame_times=times,
                backend="fastplotlib",
                layout="diagonal",
            )


def test_a_1d_environment_is_drawn_as_position_against_time():
    from neurospatial.animation import SpikeRasterOverlay

    env = Environment.from_samples(np.linspace(0, 100, 200)[:, None], bin_size=5.0)
    times = np.arange(40) * 0.1
    fields = np.random.default_rng(0).random((40, env.n_bins))
    animation = env.animate_fields(
        fields,
        frame_times=times,
        backend="fastplotlib",
        overlays=[
            TimeSeriesOverlay(data=np.sin(times), times=times, label="lfp"),
            SpikeRasterOverlay([times[::4]], rates=np.ones((1, env.n_bins))),
        ],
    ).native
    names = [subplot.name for subplot in animation.figure]
    assert names == ["track", "raster 0", "series 0", "unit"]  # no field panel
    assert not animation._field_graphics
    track = animation.figure["track"]
    assert animation.status.startswith("t = 0.00 s")
    assert track.title.text.startswith("position along track")
    assert track.docks["right"].size > 0  # the colorbar
    # One time axis, at the bottom, and no toolbars on the time panels.
    time_panels = [animation.figure[name] for name in names[:3]]
    assert [p.axes.x.visible for p in time_panels] == [False, False, True]
    assert not any(p.toolbar for p in time_panels)
    assert len({p.docks["right"].size for p in time_panels}) == 1  # aligned
    animation.frame = 20
    _render(animation)
    assert np.isclose(animation._tracks[0].edges[:, 0], times[20]).any()


def _colorbar_triangles(plot):
    """The clipped-end markers of a colorbar: bottom and top triangles."""
    return [
        g
        for g in plot.docks["right"].graphics
        if type(g).__name__ == "MeshGraphic" and len(g.positions) == 3
    ]


def test_side_by_side_fields_are_drawn_the_same_size(apparatus, walk):
    t, _, fields = walk
    animation = apparatus.animate_fields(
        [list(fields), list(fields)], frame_times=t, backend="fastplotlib"
    ).native
    first, second = animation._plots
    assert first.docks["right"].size == second.docks["right"].size > 0
    # The shared limits put the top at the 99th percentile of the sampled
    # frames' maxima, so some sampled values pass it: a triangle says so.
    # Every value is above the bottom, 0, so that end is not marked.
    assert fields.min() > 0
    for plot in (first, second):
        triangles = _colorbar_triangles(plot)
        assert len(triangles) == 1
        assert triangles[0].positions.value[:, 1].max() > 1.0  # the top
    explicit = apparatus.animate_fields(
        fields, frame_times=t, backend="fastplotlib", vmin=0.0, vmax=1.0
    ).native
    assert not _colorbar_triangles(explicit._plot)  # 0 to 1 holds every value


@pytest.mark.parametrize(
    ("limits", "bottom", "top"),
    [((0.2, 0.5), True, True), ((0.0, 0.5), False, True), ((0.2, 1.0), True, False)],
)
def test_explicit_limits_mark_the_ends_that_clip_sampled_values(
    apparatus, walk, limits, bottom, top
):
    t, _, fields = walk
    assert fields.min() < 0.2 and fields.max() > 0.5
    field = AnimationField(fields, t, vmin=limits[0], vmax=limits[1])
    with (
        Animation(apparatus, field) as scene,
        scene.show(backend="fastplotlib") as view,
    ):
        tips = [
            float(triangle.positions.value[:, 1].max())
            for triangle in _colorbar_triangles(view.native._plot)
        ]
        assert (any(tip <= 0.0 for tip in tips), any(tip > 1.0 for tip in tips)) == (
            bottom,
            top,
        )


def test_track_lines_break_between_segments_and_across_gaps(track):
    # Out along segment 0-1, then on 1-2 past the spacing; tracking stops for
    # a second in between, or not.
    along = np.concatenate([np.linspace(5, 25, 10), np.linspace(45, 65, 10)])
    for times in (
        np.concatenate([np.arange(0.0, 1.0, 0.1), np.arange(2.0, 3.0, 0.1)]),
        np.arange(0.0, 2.0, 0.1),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            animation = track.animate_fields(
                np.zeros((4, track.n_bins)),
                frame_times=np.arange(4.0),
                backend="fastplotlib",
                window_seconds=4.0,
                overlays=[
                    PositionOverlay(positions=track.linear_to_nd(along), times=times)
                ],
            ).native
        animation.widget.indices = {"time": 1.5}
        _render(animation)
        vertices = animation._tracks[0].paths.data[:].reshape(-1, 3, 3)
        segments = vertices[np.isfinite(vertices[:, 0, 0]), :2, :2]
        # Each run of nine segments, and none from one segment to the other.
        assert len(segments) == 18
        assert not np.any((segments[:, 0, 1] < 30) & (segments[:, 1, 1] > 40))
        np.testing.assert_allclose(segments[[0, 8], [0, 1], 1], [5, 25], atol=0.01)


def test_opacity_looks_as_in_matplotlib_and_the_cursor_shows_on_bright_bins(
    track, decoded
):
    from neurospatial.animation import EpochOverlay, SpikeRasterOverlay

    times, _, _ = decoded
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        animation = track.animate_fields(
            np.ones((len(times), track.n_bins)),  # every bin the top color
            frame_times=times,
            backend="fastplotlib",
            vmin=0.0,
            vmax=1.0,
            overlays=[
                EpochOverlay([[1.0, 1.4]], color="white", alpha=0.25),
                SpikeRasterOverlay([np.array([5.0])]),
            ],
        ).native
    animation.show()
    animation.widget.indices = {"time": 1.2}
    _render(animation)
    figure = animation.figure
    # White at 0.25 over black, blended as Matplotlib does: a quarter gray.
    band = _pixel(figure, figure["raster 0"], 1.1, 0.0)
    np.testing.assert_allclose(band, 0.25, atol=0.04)
    # The cursor's dark edge sets it apart from the brightest bins.
    track_panel = figure["track"]
    column, row = track_panel.map_world_to_screen((1.2, 50.0, 0))[:2]
    image = np.asarray(figure.canvas.draw())[..., :3] / 255
    around = image[int(row), int(column) - 4 : int(column) + 5]
    yellow = colormaps["viridis"](1.0)[:3]
    assert (np.abs(around - yellow).sum(axis=1) > 0.5).sum() >= 3


def test_a_2d_posterior_is_drawn_along_a_track_of_the_maze(track):
    # A 2-D grid around the L-shaped track; each frame's probability is all
    # in one 2-D bin.
    rng = np.random.default_rng(0)
    along = track.linear_to_nd(rng.uniform(0, 70, 3000))
    plane = Environment.from_samples(
        along + rng.normal(0, 1, along.shape),
        bin_size=5.0,
        units=track.units,
        frame=track.frame,
    )
    times = np.arange(20) * 0.1
    hot = rng.integers(0, plane.n_bins, len(times))
    fields = np.zeros((len(times), plane.n_bins))
    fields[np.arange(len(times)), hot] = 1.0
    animation = plane.animate_fields(
        fields,
        frame_times=times,
        backend="fastplotlib",
        window_seconds=4.0,
        track=track,
        overlays=[PositionOverlay(positions=along[:20], times=times)],
    ).native
    assert animation._plots and animation._tracks  # both views
    _render(animation)
    panel = animation._tracks[0]
    _, nearest = track.bin_center_tree().query(plane.bin_centers[hot])
    # Every held frame is a column; the last, held for no time, is not.
    np.testing.assert_allclose(panel.edges[:, 0], times[:-1])
    for frame, row in enumerate(panel.values):
        assert row[nearest[frame]] == 1.0  # the mass moves to the nearest bin
        assert np.nansum(row) == pytest.approx(1.0)  # and is kept
    assert animation.figure["track"].docks["right"].size > 0  # its own colorbar

    with pytest.raises(ValueError, match="linearized track"):
        plane.animate_fields(
            fields, frame_times=times, backend="fastplotlib", track=plane
        )
    # A track finer than the 2-D bins leaves bins without probability.
    fine = Environment.linear_track(
        node_positions=[[0.0, 0.0], [30.0, 0.0], [30.0, 30.0]],
        bin_size=1.0,
        units=track.units,
        frame=track.frame,
    )
    with pytest.warns(UserWarning, match="bin_size of at least the 2-D bins'"):
        plane.animate_fields(
            fields, frame_times=times, backend="fastplotlib", track=fine
        )

    # A track bin no 2-D bin maps to, or only missing ones, is missing.
    from scipy.sparse import csr_matrix

    from neurospatial.animation._tracks import _project

    to_track = csr_matrix(([1.0, 1.0, 1.0], ([0, 1, 2], [0, 0, 1])), shape=(3, 3))
    projected = _project(np.array([[0.2, 0.3, np.nan]]), to_track)
    np.testing.assert_array_equal(projected, [[0.5, np.nan, np.nan]])


def test_see_through_fields_keep_the_displayed_peak_opaque():
    from neurospatial.environment._fastplotlib import ColorScale

    # The typical peak below the colors' floor: every shown value is opaque.
    alpha = ColorScale("viridis", 0.2, 1.0, "lightgray", opaque_from=0.1).lut[1:, 3]
    np.testing.assert_allclose(alpha, 1.0)
    # Above the colors' top: the top color is still opaque, the floor clear.
    alpha = ColorScale("viridis", 0.0, 0.5, "lightgray", opaque_from=0.8).lut[1:, 3]
    assert alpha[0] == 0.0 and alpha[-1] == pytest.approx(1.0)


def test_side_by_side_posteriors_share_the_tracks_color_limits(track):
    rng = np.random.default_rng(0)
    along = track.linear_to_nd(rng.uniform(0, 70, 3000))
    plane = Environment.from_samples(
        along + rng.normal(0, 1, along.shape),
        bin_size=10.0,
        units=track.units,
        frame=track.frame,
    )
    times = np.arange(30) * 0.1
    diffuse = np.full((30, plane.n_bins), 1.0 / plane.n_bins)
    sharp = np.zeros((30, plane.n_bins))
    sharp[np.arange(30), rng.integers(0, plane.n_bins, 30)] = 1.0
    tops = []
    for fields in ([list(diffuse), list(sharp)], [list(sharp), list(diffuse)]):
        animation = plane.animate_fields(
            fields, frame_times=times, backend="fastplotlib", track=track
        ).native
        tops.append([panel.scale.vmax for panel in animation._tracks])
    # Each posterior's track has its own limits, whatever the panel order,
    # and the sharp posterior's peak is on its scale.
    assert tops[0] == pytest.approx(tops[1][::-1])
    assert tops[0][1] == pytest.approx(1.0)
    assert tops[0][0] < 1.0


def test_a_tracks_window_reuses_rows_read_for_the_last_one(track):
    from neurospatial.animation._tracks import _project

    rng = np.random.default_rng(1)
    along = track.linear_to_nd(rng.uniform(0, 70, 3000))
    plane = Environment.from_samples(
        along + rng.normal(0, 1, along.shape),
        bin_size=10.0,
        units=track.units,
        frame=track.frame,
    )
    fields = rng.random((300, plane.n_bins))
    animation = plane.animate_fields(
        fields,
        frame_times=np.arange(300) * 0.01,
        backend="fastplotlib",
        track=track,
        window_seconds=0.5,
    ).native
    frame_times = np.arange(300) * 0.01
    panel = animation._tracks[0]
    # Forward, back, a jump and a wider window: as read afresh each time.
    for time in (0.3, 0.32, 0.6, 0.31, 2.9, 0.0, "]", 1.0, 1.4):
        if time == "]":
            animation._scale_windows(2.0)
            continue
        animation.widget.indices = {"time": time}
        _render(animation)
        left, right = (
            time - animation.window_seconds / 2,
            time + animation.window_seconds / 2,
        )
        held = np.searchsorted(frame_times, panel.edges[:, 0], side="right") - 1
        # Each column is a frame held in the window, projected onto the track.
        assert held[0] == max(np.searchsorted(frame_times, left, side="right") - 1, 0)
        assert frame_times[held[-1] + 1] >= min(right, frame_times[-1])
        np.testing.assert_allclose(
            panel.values, _project(fields[held], animation._prepared.projection)
        )


def test_only_tracks_get_a_position_by_time_panel(apparatus, walk, track, decoded):
    t, _, fields = walk
    animation = apparatus.animate_fields(
        fields, frame_times=t, backend="fastplotlib"
    ).native
    assert [subplot.name for subplot in animation.figure] == ["field"]
    times, _, fields = decoded
    with pytest.raises(ValueError, match="window_seconds must be positive"):
        track.animate_fields(
            fields[:20], frame_times=times[:20], backend="fastplotlib", window_seconds=0
        )


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
    # The field is 1 on the right of the arena and 0 on the left.
    field = (apparatus.bin_centers[:, 0] > 20).astype(float)
    animation = apparatus.animate_fields(
        np.tile(field, (len(t), 1)),
        frame_times=t,
        backend="fastplotlib",
        vmin=0.0,
        vmax=1.0,
        overlays=[overlay],
    ).native
    animation.show()
    plot = animation._plot
    high = np.asarray(colormaps["viridis"](1.0)[:3])
    red, black = np.array([1.0, 0, 0]), np.zeros(3)
    figure = animation.figure
    # Above, the video covers the field. Below, it shows through where the
    # field is low (clear at 0), and the field is opaque at its peak.
    np.testing.assert_allclose(_pixel(figure, plot, 10, 35), red, atol=2 / 255)
    np.testing.assert_allclose(_pixel(figure, plot, 10, 5), black, atol=2 / 255)
    np.testing.assert_allclose(
        _pixel(figure, plot, 30, 35), red if above else high, atol=2 / 255
    )
    wall = _pixel(animation.figure, plot, 20, 10)  # the divider, red too
    np.testing.assert_allclose(wall, to_rgba("firebrick")[:3], atol=3 / 255)
    assert plot.camera.local.scale_y > 0  # y still points up


def test_video_follows_the_slider_and_stands_upright_without_calibration(
    apparatus, walk
):
    t, _, _ = walk
    frames = np.zeros((len(t), 20, 40, 3), dtype=np.uint8)
    frames[:, :10, :, 0] = 255  # top rows red
    frames[60:, :, :, 2] = 255  # later frames also blue
    with pytest.warns(UserWarning, match="no calibration"):
        overlay = VideoOverlay(source=frames, times=t, alpha=1.0)
        animation = apparatus.animate_fields(
            np.zeros((len(t), apparatus.n_bins)),
            frame_times=t,
            backend="fastplotlib",
            overlays=[overlay],
        ).native
    animation.show()
    plot = animation._plot
    top = _pixel(animation.figure, plot, 10, 35)
    np.testing.assert_allclose(top, [1.0, 0, 0], atol=2 / 255)  # first rows at top
    animation.frame = 80
    _render(animation)
    np.testing.assert_allclose(
        _pixel(animation.figure, plot, 10, 35), [1.0, 0, 1.0], atol=2 / 255
    )


def test_overlays_move_at_their_own_rate_between_slow_frames(apparatus, track):
    # One rate map a second; tracking and video at 30 Hz.
    frame_times = np.arange(5.0)
    times = np.arange(0.5, 4.0, 1 / 30)  # tracking starts at 0.5 s
    path = np.c_[5 + 30 * times / 4, np.full(len(times), 20.0)]
    frames = np.zeros((len(times), 20, 40, 3), dtype=np.uint8)
    frames[:, 0, :, 1] = (np.arange(len(times)) % 256)[:, None]  # its own green
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # no calibration; partial overlap
        animation = apparatus.animate_fields(
            np.zeros((5, apparatus.n_bins)),
            frame_times=frame_times,
            backend="fastplotlib",
            overlays=[
                PositionOverlay(positions=path, times=times, trail_length=2),
                VideoOverlay(source=frames, times=times),
            ],
        ).native
    position = animation._layers[0]
    (video,) = animation._videos
    for time in (0.2, 1.0, 1.51, 2.99):
        animation.widget.indices = {"time": time}
        _render(animation)
        sample = int(np.searchsorted(times, time, side="right")) - 1
        assert animation.frame == int(time)  # the field holds its frame
        if sample < 0:  # before tracking starts: nothing
            assert np.isnan(position.markers[0].data[0, 0])
            assert not video.images[0].visible
            continue
        expected = [np.interp(time, times, path[:, axis]) for axis in range(2)]
        np.testing.assert_allclose(position.markers[0].data[0, :2], expected, rtol=1e-6)
        nearest = int(np.argmin(np.abs(times - time)))
        assert video.images[0].data[0, 0, 1] == nearest % 256
        # The trail covers the last two frames' time: 2 s of samples, to
        # within one sample at its edge.
        trail = position.trails[0].data[:][:, 0]
        trail = np.unique(trail[np.isfinite(trail)])
        assert abs(len(trail) - min(sample + 1, 60)) <= 1
    animation.widget.indices = {"time": 4.5}  # tracking ended at 4 s
    _render(animation)
    assert np.isnan(position.markers[0].data[0, 0])

    # On a track, the position line has every sample in the window.
    times = np.arange(0.0, 4.0, 1 / 30)
    # Linear 30-40 cm is the gap between segments, off the track.
    on_track = track.linear_to_nd(np.linspace(0.0, 30.0, len(times)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        animation = track.animate_fields(
            np.zeros((5, track.n_bins)),
            frame_times=frame_times,
            backend="fastplotlib",
            window_seconds=1.0,
            overlays=[PositionOverlay(positions=on_track, times=times)],
        ).native
    animation.widget.indices = {"time": 2.0}
    _render(animation)
    drawn = animation._tracks[0].paths.data[:][:, :2]
    drawn = np.unique(drawn[np.isfinite(drawn[:, 0])], axis=0)
    inside = np.abs(times - 2.0) <= 0.5
    np.testing.assert_allclose(drawn[:, 0], times[inside], atol=1e-5)
    np.testing.assert_allclose(
        drawn[:, 1], track.to_linear(on_track)[inside], atol=1e-3
    )


def test_playback_reaches_overlay_samples_between_slow_frames(apparatus):
    # One rate map a second; tracking at 20 Hz.
    times = np.arange(0.0, 5.0, 0.05)
    path = np.c_[5 + 30 * times / 5, np.full(len(times), 20.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        animation = apparatus.animate_fields(
            np.zeros((5, apparatus.n_bins)),
            frame_times=np.arange(5.0),
            backend="fastplotlib",
            overlays=[PositionOverlay(positions=path, times=times)],
        ).native
    step = animation.widget.ranges["time"].step
    assert step <= 0.05 + 1e-9  # every tracking sample is played
    # ...in real time: 20 steps a second.
    assert animation.widget.ui_sliders._fps["time"] == 20

    # Faster than playback steps (60 a second by default): real time still,
    # in 1/60 s steps.
    times = np.arange(0.0, 5.0, 0.002)
    path = np.c_[5 + 30 * times / 5, np.full(len(times), 20.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        animation = apparatus.animate_fields(
            np.zeros((5, apparatus.n_bins)),
            frame_times=np.arange(5.0),
            backend="fastplotlib",
            overlays=[PositionOverlay(positions=path, times=times)],
        ).native
    assert animation.widget.ranges["time"].step == pytest.approx(1 / 60)
    assert animation.widget.ui_sliders._fps["time"] == 60


def test_timestamp_jitter_is_not_a_gap(apparatus):
    times = np.array([0.0, 0.1, 0.21, 0.31, 2.0, 2.1])  # jitter, then a real gap
    path = np.c_[np.linspace(5, 30, 6), np.full(6, 20.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        animation = apparatus.animate_fields(
            np.zeros((3, apparatus.n_bins)),
            frame_times=np.array([0.0, 1.0, 2.0]),
            backend="fastplotlib",
            overlays=[PositionOverlay(positions=path, times=times)],
        ).native
    (position,) = animation._layers
    for time, tracked in ((0.205, True), (0.30, True), (1.0, False)):
        # Drawn through 10 ms of jitter, but not 1.7 s without tracking.
        animation.widget.indices = {"time": time}
        _render(animation)
        assert np.isfinite(position.markers[0].data[0, :2]).all() == tracked


def test_every_field_with_a_video_stands_upright(apparatus, walk, video):
    t, _, fields = walk
    frames, calibration = video
    animation = apparatus.animate_fields(
        [list(fields), list(fields)],
        frame_times=t,
        backend="fastplotlib",
        overlays=[VideoOverlay(source=frames, times=t, calibration=calibration)],
    ).native
    animation.show()
    assert [p.camera.local.scale_y > 0 for p in animation._plots] == [True, True]


def test_recording_gaps_do_not_slow_playback(apparatus):
    # Two bursts of 100 Hz frames, 9 s apart.
    times = np.concatenate([np.arange(100) * 0.01, 10.0 + np.arange(100) * 0.01])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the gap is warned about
        animation = apparatus.animate_fields(
            np.zeros((200, apparatus.n_bins)),
            frame_times=times,
            backend="fastplotlib",
            speed=1.0,
            max_playback_fps=100,
        ).native
    steps = animation.widget.ui_sliders._fps["time"]
    assert steps * animation.widget.ranges["time"].step == pytest.approx(1.0)
    # 100 frames a second cannot each be shown in 60 steps: slower, warned.
    with pytest.warns(UserWarning, match="plays at speed 0.6"):
        animation = apparatus.animate_fields(
            np.zeros((200, apparatus.n_bins)), frame_times=times, backend="fastplotlib"
        ).native
    steps = animation.widget.ui_sliders._fps["time"]
    assert steps * animation.widget.ranges["time"].step == pytest.approx(0.6)


def test_an_event_shows_from_its_own_time(apparatus):
    reward = EventOverlay(
        event_times=np.array([0.5]), event_positions=np.array([[20.0, 20.0]])
    )
    animation = apparatus.animate_fields(
        np.zeros((5, apparatus.n_bins)),
        frame_times=np.arange(5.0),
        backend="fastplotlib",
        overlays=[reward],
    ).native
    (layer,) = [layer for layer in animation._layers if hasattr(layer, "kinds")]
    ((_, _, (graphic,)),) = layer.kinds.values()
    for time, shown in ((0.4, False), (0.5, True), (0.9, True)):
        animation.widget.indices = {"time": time}
        _render(animation)
        assert (graphic.colors[:][:, 3] > 0).any() == shown, time


def test_frame_labels_name_each_frame_and_must_match_the_frames(apparatus):
    fields = np.zeros((3, apparatus.n_bins))
    with pytest.raises(ValueError, match="frame_labels has 2 labels for 3 frames"):
        apparatus.animate_fields(
            fields,
            frame_times=np.arange(3.0) / 10,
            backend="fastplotlib",
            frame_labels=["trial 1", "trial 2"],
        )
    animation = apparatus.animate_fields(
        fields,
        frame_times=np.arange(3.0) / 10,
        backend="fastplotlib",
        frame_labels=["trial 1", "trial 2", "trial 3"],
    ).native
    animation.widget.indices = {"time": 0.1}
    _render(animation)
    assert animation.status.startswith("trial 2")


def test_the_status_shows_the_sliders_time(apparatus):
    from neurospatial.animation import EpochOverlay

    animation = apparatus.animate_fields(
        np.zeros((3, apparatus.n_bins)),
        frame_times=np.arange(3.0),
        backend="fastplotlib",
        overlays=[EpochOverlay([[0.5, 0.6]], label="ripple")],
    ).native
    animation.widget.indices = {"time": 0.55}
    _render(animation)
    assert animation.status == "t = 0.55 s  |  ripple 1 of 1"
