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
    )
    (layer,) = animation._layers
    alphas = {}
    for frame in (0, len(t) - 1):
        animation.frame = frame
        _render(animation)
        segments = layer.lines.data[:]
        np.testing.assert_allclose(segments[0::3, :2], path[[frame, frame]], rtol=1e-6)
        np.testing.assert_allclose(segments[1::3, :2], objects)
        alphas[frame] = layer.lines.colors.a
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
    animation = apparatus.animate_fields(counted, frame_times=t, backend="fastplotlib")
    reads.clear()
    before = animation._field.graphic.mapcoords[:].copy()
    animation.frame = 60
    _render(animation)
    assert reads == [60]
    assert not np.array_equal(animation._field.graphic.mapcoords[:], before)
    np.testing.assert_array_equal(animation._field.values, fields[60])


def test_the_right_click_settings_pause_the_field(apparatus, walk):
    t, path, fields = walk
    animation = apparatus.animate_fields(
        fields,
        frame_times=t,
        backend="fastplotlib",
        overlays=[PositionOverlay(positions=path, times=t)],
    )
    (frames,) = animation.widget[animation._plot].nd_graphics
    assert frames.graphic is animation._field.graphic  # the menu draws its settings
    frames.pause = True
    animation.frame = 40
    for graphic in animation.widget.ndgraphics:
        animation.widget.indices._schedule_fetch(graphic)  # skipped while paused
    np.testing.assert_allclose(
        animation._layers[0].marker.data[0, :2], path[0], rtol=1e-6
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


def test_time_series_are_drawn_beside_the_field_on_its_clock(apparatus, walk):
    t, path, fields = walk
    series_times = np.arange(0.0, 4.0, 0.0005)  # 2 kHz, not the frames' 30 Hz
    speed = TimeSeriesOverlay(
        data=np.sin(series_times),
        times=series_times,
        label="speed",
        window_seconds=1.0,
        cursor_color="orange",
    )
    accel = TimeSeriesOverlay(
        data=np.cos(series_times),
        times=series_times,
        label="accel",
        window_seconds=1.0,
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
            overlays=[speed, accel, rewards],
        )
    animation.show()
    animation.figure.canvas.draw()
    assert animation.frame == 0  # fitting the series' view does not move the time
    panels = {s.title.text: s for s in animation.figure if s.name != "field"}
    assert set(panels) == {"speed", "accel"}
    animation.frame = 50
    _render(animation)
    subplot = panels["speed"]
    trace = animation.widget[subplot]["speed"]
    low, high = trace.display_range
    assert low < t[50] < high
    assert high - low == pytest.approx(1.0, abs=0.02)
    # Every sample in the window is drawn: 2,000, past fastplotlib's 1,000.
    (line,) = trace.graphic.graphics
    x = line.data[:][:, 0]
    x = x[np.isfinite(x)]
    assert len(np.unique(x)) >= 1999
    low, high = subplot.y_range  # the series' whole range, with a margin
    assert low < np.sin(series_times).min() and high > np.sin(series_times).max()
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
        window_seconds=1.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        animation = apparatus.animate_fields(
            fields, frame_times=session, backend="fastplotlib", overlays=[speed]
        )
    animation.frame = 60
    _render(animation)
    assert animation.widget.indices["time"] == pytest.approx(t[60])  # from frame 0
    (subplot,) = [s for s in animation.figure if s.name != "field"]
    (line,) = animation.widget[subplot]["speed"].graphic.graphics
    x = line.data[:][:, 0]
    assert len(np.unique(x[np.isfinite(x)])) >= 99  # not collapsed by float32
    assert animation._plot.title.text.endswith(f"t = {session[60]:.2f} s")


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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        animation = apparatus.animate_fields(
            fields, frame_times=t, backend="fastplotlib", overlays=overlays
        )
    assert any("'empty' has no values" in str(w.message) for w in caught)
    assert animation.frame == 0
    titles = {s.title.text for s in animation.figure if s.name != "field"}
    assert "speed" in titles


def test_uneven_frames_are_each_reached_and_gaps_warned(apparatus, walk):
    _, _, fields = walk
    # Two bursts of 2 ms decoding bins, 5 s apart.
    times = np.concatenate([np.arange(60) * 0.002, 5.0 + np.arange(60) * 0.002])
    with pytest.warns(UserWarning, match="unevenly spaced"):
        animation = apparatus.animate_fields(
            fields, frame_times=times, backend="fastplotlib"
        )
    step = animation.widget.ranges["time"].step
    assert step <= np.diff(times).min()  # stepping never passes over a frame
    jittered = np.cumsum(np.random.default_rng(0).uniform(0.9, 1.1, len(times)) / 30.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # ordinary jitter is not warned about
        animation = apparatus.animate_fields(
            fields, frame_times=jittered, backend="fastplotlib"
        )
    step = animation.widget.ranges["time"].step
    reached = {
        animation._frame_of(k * step) for k in range(int(1 + jittered[-1] / step))
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
        )
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
    (line,) = animation._track.lines
    column = 15 - animation._track.first
    np.testing.assert_allclose(line.data[column, :2], [times[15], 45.0], rtol=1e-6)
    face = 2 * (column * track.n_bins + 3)
    assert animation._track.mesh.tooltip_format({"face_index": face}).splitlines() == [
        f"t = {times[15]:.3f} s",
        "bin 3, position along track (cm): 45",
        "value: 1",
    ]
    # The last frame before the gap lasts one frame interval, not the gap.
    animation.frame = 19  # bin 1, 10-20 cm
    _render(animation)
    np.testing.assert_allclose(
        _pixel(animation.figure, panel, times[19] + 0.05, 15.0), high, atol=2 / 255
    )
    np.testing.assert_allclose(
        _pixel(animation.figure, panel, times[19] + 0.3, 15.0), 0.0, atol=0.05
    )
    animation._track.view.cursor.selection = times[12]  # dragging the cursor
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
        window_seconds=1.0,
    )
    rewards = EventOverlay(
        event_times={"reward": session[[20]]}, event_positions={"reward": path[[20]]}
    )
    animation = apparatus.animate_fields(
        fields, frame_times=session, backend="fastplotlib", overlays=[raster, rewards]
    )
    panel = animation.figure["raster 0"]
    (layer,) = animation._rasters
    # Unit 1 in the bottom row, unit 0 in the top one: a tick a spike.
    ticks = layer.ticks.data[:][:, :2]
    np.testing.assert_allclose(
        ticks[0::3], [[2.0, -0.4], [0.5, 1.6], [1.0, 1.6]], rtol=1e-6
    )
    np.testing.assert_allclose(
        ticks[1::3], [[2.0, 0.4], [0.5, 2.4], [1.0, 2.4]], rtol=1e-6
    )
    assert layer.describe({"vertex_index": 4}).splitlines() == [
        "unit CA1-a",
        "t = 100.5000 s",
    ]
    assert panel.y_range == pytest.approx((-0.5, 2.5))
    (marks,) = [g for g in panel.graphics if g.name == "reward times"]
    np.testing.assert_allclose(marks.data[:], t[[20]], rtol=1e-5)
    animation.frame = 60
    _render(animation)
    assert panel.x_range == pytest.approx((t[60] - 0.5, t[60] + 0.5))
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
    )
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
        unit_panel.title.text == f"unit 1: rate map, peak {np.nanmax(rates[1]):.3g} Hz"
    )
    np.testing.assert_array_equal(layer.rates.field.values, rates[1])
    colors = layer.ticks.colors[:]
    np.testing.assert_allclose(colors[0:3], np.tile(to_rgba("orange"), (3, 1)))
    np.testing.assert_allclose(colors[3:9], np.tile(to_rgba("white"), (6, 1)))
    _click(animation.figure, animation._plot, 20.0, 20.0)  # the field: no change
    assert unit_panel.title.text.startswith("unit 1:")

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
        )
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
                SpikeRasterOverlay([times[::3]], window_seconds=1.0),
            ],
        )
    animation.show()
    figure = animation.figure
    (bounds,) = [g for g in figure["track"].graphics if g.name == "ripple bounds"]
    np.testing.assert_allclose(np.sort(bounds.data[:]), [0.45, 0.65, 5.2, 5.4])
    for panel in ("raster 0", "series 0"):
        (bands,) = [g for g in figure[panel].graphics if g.name == "ripple epochs"]
        x = bands.positions[:][:, 0]
        np.testing.assert_allclose([x[:4].min(), x[:4].max()], [0.45, 0.65], atol=1e-6)
    marks = []
    for _ in range(4):
        _press(figure, "n")
        marks.append(animation.widget.indices["time"])
        _render(animation)
        if marks[-1] == pytest.approx(0.45):
            assert figure["field"].title.text.endswith("ripple 1 of 2")
    np.testing.assert_allclose(marks, [0.45, 1.2, 5.2, 5.2])  # stays at the last
    _press(figure, "N", ("Shift",))
    assert animation.widget.indices["time"] == pytest.approx(1.2)
    _render(animation)
    assert "ripple" not in figure["field"].title.text


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
        )
    animation.show()
    figure = animation.figure
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
    assert animation._track.window == pytest.approx(2.0)
    ((lfp, _),) = animation._series
    assert lfp.display_window == pytest.approx(4.0)  # each its own window, doubled
    # The wider window draws frames a second away.
    _render(animation)
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
    assert animation._track.window == pytest.approx(0.5)


def test_a_tracks_segments_are_bounded_and_named(track, decoded):
    times, _, fields = decoded
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        animation = track.animate_fields(
            fields[:20], frame_times=times[:20], backend="fastplotlib"
        )
    panel = animation.figure["track"]
    (bounds,) = [g for g in panel.graphics if type(g).__name__ == "InfLineGraphic"]
    # Segment 0-1 stops at 30 cm, then 10 cm of spacing to 1-2.
    np.testing.assert_allclose(np.sort(bounds.data[:]), [30.0, 40.0])
    names = [(label.text, middle) for label, middle in animation._track.labels]
    assert names == [("0\u20131", 15.0), ("1\u20132", 55.0)]
    animation.frame = 10
    _render(animation)
    (label, _), _ = animation._track.labels
    assert label.offset[0] == pytest.approx(times[10] - 1.0)  # the window's left


def test_only_tracks_get_a_position_by_time_panel(apparatus, walk, track, decoded):
    t, _, fields = walk
    animation = apparatus.animate_fields(fields, frame_times=t, backend="fastplotlib")
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
        )
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
        )
    position = animation._layers[0]
    (video,) = animation._videos
    for time in (0.2, 1.0, 1.51, 2.99):
        animation.widget.indices = {"time": time}
        _render(animation)
        sample = int(np.searchsorted(times, time, side="right")) - 1
        assert animation.frame == int(time)  # the field holds its frame
        if sample < 0:  # before tracking starts: nothing
            assert np.isnan(position.marker.data[0, 0])
            assert not video.image.visible
            continue
        np.testing.assert_allclose(position.marker.data[0, :2], path[sample], rtol=1e-6)
        assert video.image.data[0, 0, 1] == sample % 256
        # The trail covers the last two frames' time: 2 s of samples, to
        # within one sample at its edge.
        trail = position.trail.data[:][:, 0]
        trail = trail[np.isfinite(trail)]
        assert abs(len(trail) - min(sample + 1, 60)) <= 1
    animation.widget.indices = {"time": 4.5}  # tracking ended at 4 s
    _render(animation)
    assert np.isnan(position.marker.data[0, 0])

    # On a track, the position line has every sample in the window.
    times = np.arange(0.0, 4.0, 1 / 30)
    on_track = track.linear_to_nd(np.linspace(0.0, 70.0, len(times)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        animation = track.animate_fields(
            np.zeros((5, track.n_bins)),
            frame_times=frame_times,
            backend="fastplotlib",
            window_seconds=1.0,
            overlays=[PositionOverlay(positions=on_track, times=times)],
        )
    animation.widget.indices = {"time": 2.0}
    _render(animation)
    (line,) = animation._track.lines
    drawn = line.data[:][:, :2]
    drawn = drawn[np.isfinite(drawn[:, 0])]
    inside = np.abs(times - 2.0) <= 0.5
    np.testing.assert_allclose(drawn[:, 0], times[inside], atol=1e-5)
    np.testing.assert_allclose(
        drawn[:, 1], track.to_linear(on_track)[inside], atol=1e-3
    )


def test_dry_run_reports_without_drawing(apparatus, walk, capsys):
    t, _, fields = walk
    result = apparatus.animate_fields(
        fields, frame_times=t, backend="fastplotlib", dry_run=True
    )
    assert result is None
    assert f"{len(t):,} frames of {apparatus.n_bins:,} bins" in capsys.readouterr().err
