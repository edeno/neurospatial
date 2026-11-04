"""The overview figure and the animation scene, read back from their artists."""

from __future__ import annotations

import dataclasses

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from neurospatial._exceptions import InputError
from neurospatial.animation import (
    EpochOverlay,
    EventOverlay,
    HeadDirectionOverlay,
    PositionOverlay,
    SpikeRasterOverlay,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _titled(fig, text):
    (ax,) = [a for a in fig.axes if a.get_title().startswith(text)]
    return ax


def _texts(ax):
    return [t.get_text() for t in ax.texts]


def test_occupancy_panel_is_the_data(square_recording):
    rec = square_recording
    fig = rec.plot_overview()
    assert len(fig.axes) > 6  # six panels, plus colorbar and insets
    ax = _titled(fig, "Occupancy")
    (mesh,) = ax.collections
    array = mesh.get_array()
    corners = mesh.get_coordinates()  # (ny + 1, nx + 1, 2)
    centers = (
        corners[:-1, :-1] + corners[1:, :-1] + corners[:-1, 1:] + corners[1:, 1:]
    ) / 4
    # Read the cell each bin center falls in back from the mesh.
    distance = np.linalg.norm(
        rec.env.bin_centers[:, None, None, :] - centers[None], axis=-1
    )
    nearest = distance.reshape(len(rec.env.bin_centers), -1).argmin(axis=1)
    shown = np.ma.getdata(array).ravel()[nearest]
    masked = np.ma.getmaskarray(array).ravel()[nearest]
    occupancy = rec.env.occupancy(rec.times, rec.positions)
    visited = occupancy > 0
    assert visited.any()
    assert (~visited).any()
    np.testing.assert_allclose(shown[visited], occupancy[visited])
    # Never-visited bins are masked, so they draw in the NaN color.
    assert masked[~visited].all()
    assert not masked[visited].any()
    assert not _texts(ax)
    # Every sample is a visited bin: the occupancy sums to the tracked time.
    assert occupancy.sum() == pytest.approx(rec.times[-1] - rec.times[0], rel=0.02)


def test_heading_only_recording_marks_missing_panels(heading_recording):
    fig = heading_recording.plot_overview()
    for title in ("Environment and path", "Occupancy", "Speed", "First 60 s"):
        ax = _titled(fig, title)
        assert any("not in this dataset" in t for t in _texts(ax)), title
    heading = _titled(fig, "Head direction")
    assert not any("not in this dataset" in t for t in _texts(heading))
    assert heading.patches  # the polar bars
    units = _titled(fig, "Units per region")
    assert [p.get_height() for p in units.patches][:1] == [heading_recording.n_units]


def test_square_recording_without_headings_says_so(square_recording):
    rec = dataclasses.replace(square_recording, headings=None, heading_source="")
    ax = _titled(rec.plot_overview(), "Head direction")
    assert any("not in this dataset" in t for t in _texts(ax))


def test_speed_panel_marks_min_speed(square_recording):
    rec = dataclasses.replace(square_recording, params={"min_speed": 5.0})
    ax = _titled(rec.plot_overview(), "Speed")
    (line,) = ax.lines
    assert list(line.get_xdata()) == [5.0, 5.0]
    assert sum(p.get_height() for p in ax.patches) > 0


def test_speed_panel_range_is_not_set_by_tracking_jumps(
    jumpy_box_recording, long_box_recording
):
    rec = jumpy_box_recording
    _median, _p95, p999, fastest = rec._speed_quantiles
    assert fastest > 4 * p999
    ax = _titled(rec.plot_overview(), "Speed")
    assert ax.get_xlim()[1] < 1.1 * p999
    beyond = int(np.sum(rec._speed > p999))
    (note,) = [t for t in _texts(ax) if "beyond" in t]
    assert note.startswith(f"{beyond} samples beyond the 99.9th")
    assert f"fastest {fastest:.3g} cm/s" in note
    # Negative control: without jumps the axis ends near the fastest step.
    clean = _titled(long_box_recording.plot_overview(), "Speed")
    assert clean.get_xlim()[1] > 0.9 * long_box_recording._speed_quantiles[3]


@pytest.fixture
def stem_recording(square_recording):
    """The box session's x position on a 1-D 0-100 cm stem track."""
    env, positions, _ = _stem_env(np.asarray(square_recording.positions))
    return dataclasses.replace(
        square_recording,
        env=env,
        positions=positions,
        headings=None,
        heading_source="",
    )


def test_a_1d_environment_panel_shows_samples_along_the_track(stem_recording):
    rec = stem_recording
    ax = _titled(rec.plot_overview(), "Environment and path")
    assert not ax.yaxis.get_visible()
    assert not any("not drawn" in t for t in _texts(ax))
    (strip,) = ax.child_axes
    edges = np.arange(0.0, 101.0, 2.0)
    expected, _ = np.histogram(rec.positions[:, 0], bins=edges)
    np.testing.assert_array_equal(_bar_heights(strip), expected)
    assert "along the track" in strip.get_xlabel()


@pytest.fixture
def room_grid_recording(square_recording):
    """The box session lifted into a 1 m x 1 m x 40 cm room, 10 cm cubes."""
    from neurospatial import Environment

    xy = np.asarray(square_recording.positions)
    z = 20.0 + 15.0 * np.sin(square_recording.times / 3.0)
    edges = (
        np.linspace(0.0, 100.0, 11),
        np.linspace(0.0, 100.0, 11),
        np.linspace(0.0, 40.0, 5),
    )
    mask = np.ones((10, 10, 4), dtype=bool)
    mask[0, 0, :] = False  # a column with no bin
    env = Environment.from_grid_mask(mask, grid_edges=edges, units="cm")
    return dataclasses.replace(
        square_recording,
        env=env,
        positions=np.column_stack([xy, z]),
        headings=None,
        heading_source="",
    )


def test_3d_occupancy_panel_is_the_occupancy_summed_over_z(room_grid_recording):
    from neurospatial.datasets._overview import top_view_occupancy

    rec = room_grid_recording
    # Independently: seconds per 3-D bin from the samples, summed per column.
    occupancy = rec.env.occupancy(rec.times, rec.positions)
    columns = np.floor(rec.env.bin_centers[:, :2] / 10).astype(int)
    expected = np.zeros((10, 10))
    np.add.at(expected, (columns[:, 0], columns[:, 1]), occupancy)
    visited = expected > 0
    edges, seconds = top_view_occupancy(rec)
    np.testing.assert_array_equal(edges[0], np.linspace(0.0, 100.0, 11))
    np.testing.assert_allclose(seconds[visited], expected[visited])
    np.testing.assert_array_equal(np.isnan(seconds), ~visited)
    assert np.isnan(seconds[0, 0])  # the column with no bin is missing, not zero
    np.testing.assert_allclose(np.nansum(seconds), occupancy.sum())
    # The panel draws exactly these columns.
    ax = _titled(rec.plot_overview(), "Occupancy")
    assert "top view (x\u2013y, summed over z)" in ax.get_title()
    (mesh,) = ax.collections
    drawn = mesh.get_array()
    np.testing.assert_allclose(
        np.sort(np.ma.compressed(drawn)), np.sort(seconds[visited])
    )


def test_3d_path_panel_is_the_xy_path(room_grid_recording):
    rec = room_grid_recording
    ax = _titled(rec.plot_overview(), "Environment and path")
    assert "top view (x\u2013y, summed over z)" in ax.get_title()
    (path,) = [c for c in ax.collections if len(c.get_offsets()) == len(rec.times)]
    np.testing.assert_array_equal(path.get_offsets(), rec.positions[:, :2])


def test_time_panel_draws_the_first_minute(square_recording):
    rec = square_recording
    ax = _titled(rec.plot_overview(), "First 60 s")
    top, bottom = ax.child_axes
    (x_line, _y_line) = top.lines
    # The first spike comes just after the first sample: the window starts there.
    start = min(train[0] for train in rec.spike_times if train.size)
    assert rec.times[0] < start < rec.times[0] + 0.1
    stop = start + 60.0
    window = (rec.times >= start) & (rec.times < stop)
    np.testing.assert_array_equal(x_line.get_ydata(), rec.positions[window, 0])
    n_ticks = sum(
        len(seg) for coll in bottom.collections for seg in [coll.get_segments()]
    )
    expected = sum(int(np.sum((s >= start) & (s < stop))) for s in rec.spike_times)
    assert n_ticks == expected


def _late_spikes(rec, start):
    """``rec`` with every spike before ``start`` s removed."""
    return dataclasses.replace(
        rec, spike_times=tuple(train[train >= start] for train in rec.spike_times)
    )


def test_time_panel_and_scene_start_where_spikes_start(square_recording):
    # The neural recording starts 25 s after the tracking, as the bat's does.
    rec = _late_spikes(square_recording, 25.0)
    first_spike = min(train[0] for train in rec.spike_times if train.size)
    assert first_spike > rec.times[0] + 20
    ax = _titled(rec.plot_overview(), "First 60 s")
    _top, bottom = ax.child_axes
    assert bottom.get_xlim() == (first_spike, first_spike + 60.0)
    n_ticks = sum(len(coll.get_segments()) for coll in bottom.collections)
    assert n_ticks == sum(int(np.sum(t < first_spike + 60.0)) for t in rec.spike_times)
    assert n_ticks > 0
    scene = rec.animation()
    assert first_spike <= scene.frame_times[0] < first_spike + 0.02


def test_window_opens_on_the_first_analysed_epoch(square_recording):
    # As the bat's flights: spikes from the start, the analysed behaviour later.
    rec = dataclasses.replace(
        square_recording, params={"analysis_epochs": np.array([[30.0, 40.0]])}
    )
    ax = _titled(rec.plot_overview(), "First 60 s")
    assert ax.child_axes[1].get_xlim() == (30.0, 90.0)
    assert rec.animation().frame_times[0] == 30.0
    # An analysed epoch that ended before the spikes start moves nothing.
    late = _late_spikes(
        dataclasses.replace(
            square_recording, params={"analysis_epochs": np.array([[1.0, 5.0]])}
        ),
        25.0,
    )
    first_spike = min(train[0] for train in late.spike_times if train.size)
    assert _titled(late.plot_overview(), "First 60 s").child_axes[1].get_xlim()[0] == (
        first_spike
    )


def test_time_panel_starts_at_tracking_when_spikes_start_first(square_recording):
    # Negative control: spikes from the start, tracking lost for the first 10 s.
    positions = np.array(square_recording.positions)
    positions[square_recording.times < 10.0] = np.nan
    rec = dataclasses.replace(square_recording, positions=positions)
    first_tracked = rec.times[rec.times >= 10.0][0]
    assert min(train[0] for train in rec.spike_times if train.size) < first_tracked
    ax = _titled(rec.plot_overview(), "First 60 s")
    assert ax.child_axes[1].get_xlim() == (first_tracked, first_tracked + 60.0)
    assert rec.animation().frame_times[0] == first_tracked


def test_fig_argument_is_used(square_recording):
    fig = plt.figure()
    assert square_recording.plot_overview(fig=fig) is fig


def test_animation_scene_holds_the_data(square_recording):
    rec = dataclasses.replace(
        square_recording,
        events={"reward": np.array([5.0, 10.0, 20.0])},
        epochs={"run": np.array([[0.0, 30.0]]), "rest": np.array([[40.0, 50.0]])},
    )
    scene = rec.animation(trail_length=10)
    occupancy = rec.env.occupancy(rec.times, rec.positions)
    shown = scene.fields["occupancy"][0]
    np.testing.assert_allclose(shown[occupancy > 0], occupancy[occupancy > 0])
    assert np.isnan(shown[occupancy == 0]).all()
    # Held over the whole window, not only at the first frame.
    middle = scene.sample_fields(float(scene.frame_times[len(scene.frame_times) // 2]))
    np.testing.assert_array_equal(middle["occupancy"], shown)

    kinds = {type(o): o for o in scene.overlays}
    position = kinds[PositionOverlay]
    np.testing.assert_array_equal(position.positions, rec.positions)
    np.testing.assert_array_equal(position.times, rec.times)
    assert position.trail_length == 10
    assert position.units == "cm"
    np.testing.assert_array_equal(kinds[HeadDirectionOverlay].headings, rec.headings)
    raster = kinds[SpikeRasterOverlay]
    assert len(raster.spike_times) == rec.n_units
    for got, want in zip(raster.spike_times, rec.spike_times, strict=True):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(
        kinds[EventOverlay].event_times["reward"], rec.events["reward"]
    )
    epochs = {o.label: o for o in scene.overlays if isinstance(o, EpochOverlay)}
    assert set(epochs) == {"run", "rest"}
    np.testing.assert_array_equal(epochs["rest"].epochs, rec.epochs["rest"])
    # The default window is the first 60 s with tracking and spikes (here the
    # whole session), from the first sample at or after the first spike.
    first_spike = min(train[0] for train in rec.spike_times if train.size)
    assert scene.frame_times[0] == rec.times[rec.times >= first_spike][0]
    # 100 Hz tracking is drawn at 30 frames a second, a rate viewers can play.
    np.testing.assert_allclose(np.diff(scene.frame_times), 1 / 30)
    assert scene.frame_times[-1] < first_spike + 60.0


def test_animation_time_range(square_recording):
    scene = square_recording.animation(time_range=(10.0, 20.0))
    assert scene.frame_times[0] >= 10.0
    assert scene.frame_times[-1] < 20.0
    with pytest.raises(InputError, match="time_range"):
        square_recording.animation(time_range=(1000.0, 1010.0))


def test_animation_regions_follow_the_environment(square_recording):
    assert square_recording.animation().show_regions is False


def test_animation_needs_positions(heading_recording):
    with pytest.raises(InputError, match="no positions") as raised:
        heading_recording.animation()
    assert "headings" in str(raised.value)
    assert "spike_times" in str(raised.value)


def test_animation_export_frames(square_recording, tmp_path):
    scene = square_recording.animation(time_range=(0.0, 2.0))
    frames = tmp_path / "frames"
    scene.export(
        tmp_path / "scene.html", fps=10, dpi=30, embed=False, frames_dir=frames
    )
    assert len(list(frames.glob("frame_*.png"))) == 20


@pytest.fixture
def room_recording(square_recording):
    """The box session lifted into 3-D with a slow vertical wander."""
    from neurospatial import Environment

    rec = square_recording
    z = 20.0 + 10.0 * np.sin(rec.times / 5.0)
    positions = np.column_stack([rec.positions, z])
    env = Environment.from_samples(positions, bin_size=10.0, units="cm")
    return dataclasses.replace(
        rec, env=env, positions=positions, headings=None, heading_source=""
    )


def test_3d_recording_overview_and_scene(room_recording):
    fig = room_recording.plot_overview()
    for title in ("Environment", "Occupancy"):
        ax = _titled(fig, title)
        assert ax.get_title().count("top view") == 1
        assert not any("open rec.animation()" in t for t in _texts(ax))
    scene = room_recording.animation(time_range=(0.0, 5.0))
    assert scene.env.n_dims == 3
    assert scene.fields["occupancy"][0].shape == (room_recording.env.n_bins,)


def test_track_recording_overview_and_scene(track_recording):
    fig = track_recording.plot_overview()
    missing = [
        ax.get_title()
        for ax in fig.axes
        if any("not in this dataset" in t for t in _texts(ax))
    ]
    assert missing == ["Head direction"]  # the track session has no headings
    assert track_recording.animation(time_range=(0.0, 5.0)).env is track_recording.env


def _stem_env(positions):
    """A straight track whose nodes hold 1-D positions (``grosmark_2016``'s stem)."""
    import networkx as nx

    from neurospatial import Environment

    graph = nx.Graph()
    graph.add_node("stem_start", pos=(0.0,))
    graph.add_node("stem_end", pos=(100.0,))
    graph.add_edge("stem_start", "stem_end", distance=100.0)
    env = Environment.from_graph(
        graph, [("stem_start", "stem_end")], bin_size=2.0, units="cm"
    )
    return env, positions[:, :1], "cm"


def _linear_env(positions):
    """A 2-D straight track with a width (``grosmark_2016`` linear, ``valero_2022``)."""
    import networkx as nx

    from neurospatial import Environment

    graph = nx.Graph()
    graph.add_node("start", pos=(0.0, 40.0))
    graph.add_node("end", pos=(100.0, 40.0))
    graph.add_edge("start", "end", distance=100.0)
    env = Environment.from_graph(
        graph,
        [("start", "end")],
        bin_size=2.0,
        edge_spacing=0.0,
        units="cm",
        track_width=80.0,
    )
    return env, positions, "cm"


def _loop_env(positions):
    """A closed 2-D track with a width (``grosmark_2016``'s circle)."""
    import networkx as nx

    from neurospatial import Environment

    angle = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    points = 50.0 + 40.0 * np.column_stack([np.cos(angle), np.sin(angle)])
    graph = nx.Graph()
    names = [f"p{k}" for k in range(len(points))]
    for name, point in zip(names, points, strict=True):
        graph.add_node(name, pos=(float(point[0]), float(point[1])))
    order = list(zip(names, names[1:] + names[:1], strict=True))
    for k, (a, b) in enumerate(order):
        distance = float(np.linalg.norm(points[(k + 1) % len(points)] - points[k]))
        graph.add_edge(a, b, distance=distance)
    env = Environment.from_graph(
        graph, order, bin_size=2.0, units="cm", track_width=20.0
    )
    return env, positions, "cm"


def _box_env(positions):
    """A square arena from a polygon (``sargolini_2006``)."""
    import shapely

    from neurospatial import Environment

    env = Environment.from_polygon(shapely.box(0.0, 0.0, 100.0, 100.0), bin_size=2.5)
    return env, positions, "cm"


def _room_env(positions):
    """A 3-D room in metres, every voxel active (``forli_2025``)."""
    from neurospatial import Environment

    z = 1.0 + 0.5 * np.sin(np.arange(len(positions)) / 500.0)
    metres = np.column_stack([positions / 100.0, z])
    room = [(0.0, 1.0), (0.0, 1.0), (0.0, 2.0)]
    env = Environment.from_samples(
        metres,
        bin_size=(0.2, 0.2, 0.2),
        dimension_ranges=room,
        infer_active_bins=False,
        units="m",
    )
    return env, metres, "m"


@pytest.mark.parametrize(
    "build", [_stem_env, _linear_env, _loop_env, _box_env, _room_env]
)
def test_overview_draws_every_loader_environment_shape(build, square_recording):
    """Each shape a shipped loader builds draws its overview without failing."""
    env, positions, units = build(square_recording.positions)
    rec = dataclasses.replace(
        square_recording,
        env=env,
        positions=positions,
        position_units=units,
        headings=None,
        heading_source="",
    )
    fig = rec.plot_overview()
    environment = _titled(fig, "Environment")
    occupancy = _titled(fig, "Occupancy")
    for ax in (environment, occupancy):
        assert not any("not in this dataset" in t for t in _texts(ax))
    if env.n_dims < 3:
        assert environment.collections or environment.lines
        assert occupancy.collections or occupancy.lines


@pytest.mark.gui
@pytest.mark.parametrize("dims", [2, 3])
def test_animation_show_fastplotlib(dims, square_recording, room_recording):
    pytest.importorskip("fastplotlib")
    rec = square_recording if dims == 2 else room_recording
    view = rec.animation().show(backend="fastplotlib")
    view.close()


def _normalised(values, edges):
    counts, _ = np.histogram(values[np.isfinite(values)], bins=edges)
    return counts / counts.sum()


def _bar_heights(ax):
    return np.array([p.get_height() for p in ax.patches])


@pytest.fixture
def two_region_recording(square_recording):
    """Six units: four in CA1, two in DG, so the counts differ."""
    regions = ["CA1"] * 4 + ["DG"] * 2
    units = pd.DataFrame(
        {"brain_region": regions, "label": ["pyramidal"] * 6},
        index=square_recording.units.index,
    )
    return dataclasses.replace(square_recording, units=units, params={"min_speed": 5.0})


def test_environment_panel_path_is_the_positions(square_recording):
    ax = _titled(square_recording.plot_overview(), "Environment and path")
    path = ax.collections[-1]
    np.testing.assert_array_equal(path.get_offsets(), square_recording.positions)


def test_speed_panel_is_the_speed_histogram(two_region_recording):
    from neurospatial.behavior import compute_speed

    rec = two_region_recording
    ax = _titled(rec.plot_overview(), "Speed")
    speed = compute_speed(rec.times, rec.positions)
    speed = speed[np.isfinite(speed)]
    # The histogram stops at the 99.9th percentile; the panel counts the rest.
    p999 = np.percentile(speed, 99.9)
    expected, _ = np.histogram(speed, bins=60, range=(speed.min(), p999))
    np.testing.assert_array_equal(_bar_heights(ax), expected)
    beyond = int(np.sum(speed > p999))
    assert beyond == len(speed) - expected.sum()
    (note,) = [t for t in _texts(ax) if "beyond" in t]
    assert note.startswith(f"{beyond} samples beyond the 99.9th")
    assert f"fastest {speed.max():.3g} cm/s" in note
    # The speed itself, from successive positions, agrees up to the smoothing.
    step = np.linalg.norm(np.diff(rec.positions, axis=0), axis=1) / np.diff(rec.times)
    assert abs(np.median(speed) - np.median(step)) < 0.15 * np.median(step)


def test_heading_panels_are_the_normalised_histograms(two_region_recording):
    from neurospatial.behavior import heading_from_velocity

    rec = two_region_recording
    ax = _titled(rec.plot_overview(), "Head direction")
    edges = np.linspace(-np.pi, np.pi, 37)
    np.testing.assert_allclose(_bar_heights(ax), _normalised(rec.headings, edges))
    (inset,) = ax.child_axes
    travel = heading_from_velocity(rec.times, rec.positions, min_speed=5.0)
    offset = (rec.headings - travel + np.pi) % (2 * np.pi) - np.pi
    np.testing.assert_allclose(_bar_heights(inset), _normalised(offset, edges))
    assert "while running" in inset.get_title()


def test_all_nan_headings_are_missing(square_recording):
    rec = dataclasses.replace(
        square_recording, headings=np.full(len(square_recording.times), np.nan)
    )
    ax = _titled(rec.plot_overview(), "Head direction")
    assert any("not in this dataset" in t for t in _texts(ax))
    assert not ax.patches


def test_unit_bars_count_the_regions(two_region_recording):
    ax = _titled(two_region_recording.plot_overview(), "Units per region")
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["CA1", "DG"]
    np.testing.assert_array_equal(_bar_heights(ax), [4, 2])
    (inset,) = ax.child_axes
    assert _bar_heights(inset).sum() == two_region_recording.n_units


def test_time_panel_skips_leading_nan_positions(square_recording):
    positions = np.array(square_recording.positions)
    positions[:300] = np.nan  # three seconds without tracking
    rec = dataclasses.replace(square_recording, positions=positions)
    ax = _titled(rec.plot_overview(), "First 60 s")
    top, bottom = ax.child_axes
    start = rec.times[300]
    assert bottom.get_xlim() == (start, start + 60.0)
    inside = (rec.times >= start) & (rec.times < start + 60.0)
    (x_line, _y_line) = top.lines
    np.testing.assert_array_equal(x_line.get_xdata(), rec.times[inside])
    assert rec.animation().frame_times[0] == pytest.approx(start)


def test_animation_shows_regions_when_the_environment_has_them(square_recording):
    from neurospatial.regions import Region

    env = square_recording.env.with_regions(
        {"well": Region(name="well", kind="point", data=[50.0, 50.0])}
    )
    rec = dataclasses.replace(square_recording, env=env)
    assert rec.animation().show_regions is True


def test_animation_refuses_a_window_of_one_frame(square_recording):
    # Three samples at 100 Hz is 0.02 s: one frame at 30 frames a second.
    with pytest.raises(InputError, match=r"at least 0\.067 s"):
        square_recording.animation(time_range=(10.0, 10.03))


def test_interval_events_do_not_replace_epochs(square_recording):
    rec = dataclasses.replace(
        square_recording,
        epochs={"run": np.array([[0.0, 30.0]])},
        events={"run": np.array([[40.0, 45.0]])},
    )
    labels = {
        o.label: o for o in rec.animation().overlays if isinstance(o, EpochOverlay)
    }
    assert set(labels) == {"run", "event: run"}
    np.testing.assert_array_equal(labels["run"].epochs, rec.epochs["run"])
    np.testing.assert_array_equal(labels["event: run"].epochs, rec.events["run"])


def test_1d_grid_positions_are_shown_along_it(square_recording):
    from neurospatial import Environment

    x = np.asarray(square_recording.positions[:, :1])
    env = Environment.from_samples(x, bin_size=5.0, units="cm")
    rec = dataclasses.replace(
        square_recording, env=env, positions=x, headings=None, heading_source=""
    )
    ax = _titled(rec.plot_overview(), "Environment and path")
    (strip,) = ax.child_axes
    # One bar per bin: every tracked sample is counted.
    assert len(strip.patches) == env.n_bins
    assert _bar_heights(strip).sum() == len(x)


def test_epoch_panel_shows_the_window_epochs(square_recording):
    """An epoch the scene's window does not meet gets no row in the epoch panel."""
    rec = dataclasses.replace(
        square_recording,
        epochs={"run": np.array([[0.0, 30.0]]), "later": np.array([[40.0, 50.0]])},
    )
    labels = {
        o.label
        for o in rec.animation(time_range=(0.0, 20.0)).overlays
        if isinstance(o, EpochOverlay)
    }
    assert labels == {"run"}


def test_speed_panel_says_when_no_speed_can_be_computed(square_recording):
    """Tracked samples more than 1 s apart give no speed, as in the summary."""
    rec = square_recording
    sparse = np.full_like(rec.positions, np.nan)
    keep = np.searchsorted(rec.times, np.arange(rec.times[0], rec.times[-1], 1.5))
    sparse[keep] = rec.positions[keep]
    rec = dataclasses.replace(rec, positions=sparse)
    assert np.isnan(rec.median_speed)
    # The occupancy panel refuses such a session, so draw the speed panel alone.
    import matplotlib.pyplot as plt

    from neurospatial.datasets._overview import _panel_speed

    fig, ax = plt.subplots()
    _panel_speed(rec, ax)
    assert any("no speed" in t for t in _texts(ax))
    plt.close(fig)


#: Planted place-field centers (cm along the stem); None is a silent unit.
FIELD_CENTERS = (70.0, None, 20.0, 50.0, None, 90.0, 35.0)


@pytest.fixture
def place_cell_stem_recording(square_recording):
    """300 s of laps on a 0-100 cm stem, one unit per planted field center."""
    times = np.arange(15_000) / 50.0
    lap = np.abs((times / 10.0) % 2.0 - 1.0)  # 10 s each way
    positions = (1.0 + 98.0 * lap)[:, None]
    env, _, _ = _stem_env(positions)
    rng = np.random.default_rng(7)  # spikes only; the path is deterministic
    trains = []
    for center in FIELD_CENTERS:
        if center is None:
            trains.append(np.empty(0))
            continue
        rate = 15.0 * np.exp(-0.5 * ((positions[:, 0] - center) / 4.0) ** 2)
        trains.append(times[rng.random(times.size) < rate / 50.0])
    units = pd.DataFrame(
        {"brain_region": ["CA1"] * len(trains), "label": ["pyramidal"] * len(trains)},
        index=pd.Index(np.arange(len(trains)), name="unit_id"),
    )
    return dataclasses.replace(
        square_recording,
        times=times,
        positions=positions,
        env=env,
        spike_times=tuple(trains),
        units=units,
        epochs={},
        headings=None,
        heading_source="",
        params={"min_speed": 2.0},
    )


def _expected_field_order():
    firing = [i for i, c in enumerate(FIELD_CENTERS) if c is not None]
    silent = [i for i, c in enumerate(FIELD_CENTERS) if c is None]
    return [*sorted(firing, key=lambda i: FIELD_CENTERS[i]), *silent]


def test_hippocampal_track_raster_is_sorted_by_place_field_peak(
    place_cell_stem_recording,
):
    rec = place_cell_stem_recording
    expected = _expected_field_order()
    np.testing.assert_array_equal(rec._raster_order, expected)
    # The overview's raster draws unit expected[k] in row k.
    ax = _titled(rec.plot_overview(), "First 60 s")
    raster = ax.child_axes[1]
    assert raster.get_ylabel() == "units, sorted by place-field peak"
    start, stop = raster.get_xlim()
    for row, unit in enumerate(expected):
        train = rec.spike_times[unit]
        n = int(np.sum((train >= start) & (train < stop)))
        assert len(raster.collections[row].get_segments()) == n
        if n:
            np.testing.assert_allclose(
                raster.collections[row].get_segments()[0][:, 1], [row - 0.4, row + 0.4]
            )
    # The scene's raster uses the same order.
    (overlay,) = [
        o for o in rec.animation().overlays if isinstance(o, SpikeRasterOverlay)
    ]
    np.testing.assert_array_equal(overlay.order, expected)


@pytest.mark.parametrize("region", ["MEC LII", "thalamus"])
def test_non_hippocampal_raster_keeps_the_file_order(place_cell_stem_recording, region):
    # Negative control: the same planted fields, recorded outside the hippocampus.
    rec = dataclasses.replace(
        place_cell_stem_recording,
        brain_regions=(region,),
        units=place_cell_stem_recording.units.assign(brain_region=region),
    )
    assert rec._raster_order is None
    ax = _titled(rec.plot_overview(), "First 60 s")
    assert ax.child_axes[1].get_ylabel() == "unit"


def test_2d_hippocampal_raster_keeps_the_file_order(square_recording):
    # Negative control: CA1 units in a 2-D box have no order along a track.
    assert square_recording._raster_order is None


@pytest.mark.parametrize(
    ("region", "hippocampal"),
    [
        ("lCA1", True),
        ("dorsal hippocampus", True),
        ("DG", True),
        ("CA3", True),
        ("edge cortex", False),
        ("MEC LII", False),
    ],
)
def test_hippocampal_regions_are_matched_by_name(square_recording, region, hippocampal):
    from neurospatial.datasets._overview import is_hippocampal

    rec = dataclasses.replace(
        square_recording,
        brain_regions=(region,),
        units=square_recording.units.assign(brain_region=region),
    )
    assert is_hippocampal(rec) is hippocampal


def test_speed_counts_go_log_when_one_bar_holds_most_samples(long_box_recording):
    # Still for 90% of the session (a bat on its perch), then the smooth path.
    positions = np.array(long_box_recording.positions)
    positions[: int(0.9 * len(positions))] = positions[0]
    still = dataclasses.replace(long_box_recording, positions=positions)
    ax = _titled(still.plot_overview(), "Speed")
    assert ax.get_yscale() == "log"
    assert ax.get_ylabel() == "samples (log scale)"
    # Negative control: a session spent moving keeps linear counts.
    moving = _titled(long_box_recording.plot_overview(), "Speed")
    assert moving.get_yscale() == "linear"


def test_raster_shows_every_units_row(square_recording):
    # The last unit is silent in the window; its row is still on the axis.
    trains = list(square_recording.spike_times)
    trains[-1] = np.empty(0)
    rec = dataclasses.replace(square_recording, spike_times=tuple(trains))
    raster = _titled(rec.plot_overview(), "First 60 s").child_axes[1]
    assert raster.get_ylim() == (-0.5, rec.n_units - 0.5)


@pytest.mark.parametrize("window", [(10.0, 14.0), (10.004, 14.009), (0.0, 60.0)])
def test_scene_exports_its_own_window(square_recording, tmp_path, window):
    from neurospatial.animation._export import prepare_export

    scene = square_recording.animation(time_range=window)
    a, b = window
    # The clock spans the window to within one frame at each end.
    assert a <= scene.frame_times[0] < a + 1 / 30
    assert b - 1 / 30 - 1e-9 <= scene.frame_times[-1] < b
    plan = prepare_export(
        tmp_path / "scene.gif", scene.frame_times, time_range=window, fps=10
    )
    assert scene.frame_times[0] <= plan.times[0]
    assert plan.times[-1] < b
