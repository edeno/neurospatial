"""The overview figure and the animation scene, read back from their artists."""

from __future__ import annotations

import dataclasses

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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
    assert visited.any() and (~visited).any()
    np.testing.assert_allclose(shown[visited], occupancy[visited])
    # Never-visited bins are masked, so they draw in the NaN color.
    assert masked[~visited].all() and not masked[visited].any()
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


def test_time_panel_draws_the_first_minute(square_recording):
    rec = square_recording
    ax = _titled(rec.plot_overview(), "First 60 s")
    top, bottom = ax.child_axes
    (x_line, y_line) = top.lines
    stop = rec.times[0] + 60.0
    n_window = int(np.sum((rec.times >= rec.times[0]) & (rec.times < stop)))
    np.testing.assert_array_equal(x_line.get_ydata(), rec.positions[:n_window, 0])
    n_ticks = sum(
        len(seg) for coll in bottom.collections for seg in [coll.get_segments()]
    )
    expected = sum(int(np.sum(s < stop)) for s in rec.spike_times)
    assert n_ticks == expected


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
    assert position.trail_length == 10 and position.units == "cm"
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
    np.testing.assert_array_equal(epochs["rest"]._intervals, rec.epochs["rest"])
    # The default window is the first 60 s of tracking (here the whole session).
    assert scene.frame_times[0] == rec.times[0]
    assert scene.frame_times[-1] < rec.times[0] + 60.0


def test_animation_time_range(square_recording):
    scene = square_recording.animation(time_range=(10.0, 20.0))
    assert scene.frame_times[0] >= 10.0 and scene.frame_times[-1] < 20.0
    with pytest.raises(InputError, match="time_range"):
        square_recording.animation(time_range=(1000.0, 1010.0))


def test_animation_regions_follow_the_environment(square_recording):
    assert square_recording.animation().show_regions is False


def test_animation_needs_positions(heading_recording):
    with pytest.raises(InputError, match="no positions") as raised:
        heading_recording.animation()
    assert "headings" in str(raised.value) and "spike_times" in str(raised.value)


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
        assert any("3-D" in t for t in _texts(_titled(fig, title)))
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


@pytest.mark.gui
@pytest.mark.parametrize("dims", [2, 3])
def test_animation_show_fastplotlib(dims, square_recording, room_recording):
    rec = square_recording if dims == 2 else room_recording
    view = rec.animation().show(backend="fastplotlib")
    view.close()
