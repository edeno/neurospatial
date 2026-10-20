"""Native GPU overlay sampling and lifecycle tests independent of GPU access."""

from __future__ import annotations

from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from neurospatial.animation.backends import fastplotlib_backend as backend
from neurospatial.animation.overlays import (
    BodypartData,
    HeadDirectionData,
    ObjectVectorData,
    PositionData,
)


class Plot:
    """Capture backend graphic updates without a graphics adapter."""

    def add_line(self, data, **kwargs):
        return SimpleNamespace(data=data, **kwargs)

    add_scatter = add_line


def clock(times):
    return backend._Samples(np.asarray(times, dtype=float), lambda time: int(time), 1.0)


def position(interp="linear", *, break_interval=False):
    return PositionData(
        data=np.array([[0.0, 2.0], [10.0, 4.0], [20.0, 6.0]]),
        color="red",
        size=5,
        trail_length=3,
        times=np.array([0.0, 1.0, 2.0]),
        interp=interp,
        sample_valid=np.ones(3, dtype=bool),
        interval_valid=np.array([not break_interval, True]),
    )


@pytest.mark.parametrize(
    "interp, expected", [("linear", [2.5, 2.5]), ("nearest", [0.0, 2.0])]
)
def test_position_uses_actual_time_on_observation_clock(interp, expected):
    data = position(interp)
    drawn = backend._Position(Plot(), data, clock(data.times))
    drawn.update(0.25)
    np.testing.assert_allclose(drawn.marker.data[0, :2], expected)
    drawn.update(2.01)
    assert np.isnan(drawn.marker.data[0, :2]).all()


def test_position_trail_never_bridges_explicit_tracking_break():
    data = position(break_interval=True)
    drawn = backend._Position(Plot(), data, clock(data.times))
    drawn.update(0.25)
    assert np.isnan(drawn.marker.data[0, :2]).all()
    drawn.update(2.0)
    trail = drawn.trail.data[:, :2]
    # Consecutive finite vertices may join 10->20, never the unsupported 0->10.
    for left, right in pairwise(trail):
        if np.isfinite([left, right]).all():
            np.testing.assert_allclose([left, right], [[10, 4], [20, 6]])
    assert np.isfinite(trail).any()


def test_heading_wraps_at_pi_and_hides_when_position_anchor_is_missing():
    data = position()
    anchor = backend._Position(Plot(), data, clock(data.times))
    headings = HeadDirectionData(
        data=np.deg2rad([170.0, -170.0]),
        color="hsv",
        length=1.0,
        times=np.array([0.0, 1.0]),
        interval_valid=np.array([True]),
    )
    drawn = backend._HeadDirection(
        Plot(), headings, clock(headings.times), anchor, np.zeros(2)
    )
    drawn.update(0.5)
    np.testing.assert_allclose(drawn.line.data[:, :2], [[5, 3], [4, 3]], atol=1e-7)
    data.interval_valid[0] = False
    drawn.update(0.5)
    assert np.isnan(drawn.line.data[:, :2]).all()


def test_bodyparts_and_object_vectors_sample_each_observed_time():
    times = np.array([0.0, 1.0])
    parts = BodypartData(
        bodyparts={"nose": np.array([[0.0, 0.0], [4.0, 2.0]])},
        skeleton=None,
        colors=None,
        times=times,
        interval_valid={"nose": np.array([True])},
    )
    drawn = backend._Bodyparts(Plot(), parts, clock(times))
    drawn.update(0.5)
    np.testing.assert_allclose(drawn.points.data[0, :2], [2.0, 1.0])
    parts.interval_valid["nose"][0] = False
    drawn.update(0.5)
    assert np.isnan(drawn.points.data[0, :2]).all()
    vectors = ObjectVectorData(
        object_positions=np.array([[8.0, 8.0]]),
        animal_positions=np.array([[0.0, 0.0], [4.0, 2.0]]),
        firing_rates=np.array([0.0, 10.0]),
        color="red",
        linewidth=1.0,
        show_objects=False,
        object_marker="o",
        object_size=5.0,
        times=times,
        interval_valid=np.array([True]),
    )
    drawn = backend._ObjectVectors(Plot(), vectors, clock(times))
    drawn.update(0.5)
    np.testing.assert_allclose(drawn.lines.data[:2, :2], [[2.0, 1.0], [8.0, 8.0]])
    np.testing.assert_allclose(drawn.lines.colors[3], 0.55)


def test_named_field_sources_keep_lazy_identity_without_reading():
    class Lazy:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            raise AssertionError("Field data must not be read during organization")

    sources = {"posterior": Lazy(), "rate": Lazy()}
    stacks, names = backend._field_stacks(sources, None)
    assert names == list(sources)
    assert stacks[0] is sources["posterior"]
    assert stacks[1] is sources["rate"]


def test_native_read_uses_absolute_time_callback_and_seek_uses_recording_time():
    animation = object.__new__(backend.FastplotlibAnimation)
    animation._closed = False
    animation._clock = np.array([0.0, 1.0])
    animation._start = 100.0
    animation.widget = SimpleNamespace(indices={"time": 0.0})
    animation._videos, animation._tracks = [], []
    animation._field_at_time = Mock(return_value=[np.array([0.75])])
    frame = animation._read(0.75)
    animation._field_at_time.assert_called_once_with(100.75)
    np.testing.assert_array_equal(frame.values[0], [0.75])
    animation.seek(100.75)
    assert animation.widget.indices == {"time": 0.75}
    with pytest.raises(ValueError, match="finite"):
        animation.seek(np.nan)


def test_native_close_releases_graphics_canvas_and_owned_readers_once():
    animation = object.__new__(backend.FastplotlibAnimation)
    animation._closed = False
    animation._frames = Mock()
    other = SimpleNamespace(slicer=Mock())
    animation.widget = SimpleNamespace(
        ndgraphics=[animation._frames, other], close=Mock()
    )
    animation.figure = SimpleNamespace(renderer=Mock())
    canvas = animation._canvas = Mock()
    reader = Mock()
    animation._owned_video_readers = [reader, reader]
    animation.close()
    animation.close()
    animation._frames.close.assert_called_once()
    other.slicer.close.assert_called_once()
    animation.widget.close.assert_called_once()
    canvas.close.assert_called_once()
    reader.close.assert_called_once()


def test_native_video_uses_nearest_frame_without_extrapolating():
    samples = clock([0.0, 0.2])
    assert samples.nearest(0.16) == 1
    assert samples.nearest(0.10) == 0  # midpoint ties use the earlier sample
    assert samples.nearest(0.20) == 1
    assert samples.nearest(0.201) == -1
    assert samples.nearest(-0.01) == -1


def test_custom_native_graphic_closes_both_worker_executors():
    pytest.importorskip("fastplotlib.widgets.nd_widget")
    graphic = backend._frames_graphic(
        SimpleNamespace(),
        {"time": 0.0},
        np.zeros((2, 3)),
        lambda time: int(time),
        lambda time: np.zeros(3),
        lambda time, data: None,
        SimpleNamespace(),
    )
    reader = graphic._reader
    slicer_executor = graphic.slicer._executor
    graphic.close()
    graphic.close()
    for executor in (reader, slicer_executor):
        with pytest.raises(RuntimeError, match="shutdown"):
            executor.submit(lambda: None)
