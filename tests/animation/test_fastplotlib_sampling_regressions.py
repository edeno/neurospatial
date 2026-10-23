"""Native GPU overlay sampling and lifecycle tests independent of GPU access."""

from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError
from itertools import pairwise
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from neurospatial.animation._backend_view import _CloseNotification
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


def test_native_read_uses_absolute_time_callback_and_seek_uses_recording_time():
    animation = object.__new__(backend.FastplotlibAnimation)
    from threading import RLock

    animation._source_lock = RLock()
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


def test_native_close_releases_graphics_and_canvas_once():
    animation = object.__new__(backend.FastplotlibAnimation)
    from threading import RLock

    animation._source_lock = RLock()
    animation._closed = False
    animation._close_notification = _CloseNotification()
    animation._frames = Mock()
    other = SimpleNamespace(slicer=Mock())
    animation.widget = SimpleNamespace(
        ndgraphics=[animation._frames, other], close=Mock()
    )
    animation.figure = SimpleNamespace(renderer=Mock())
    canvas = animation._canvas = Mock()
    animation.close()
    animation.close()
    animation._frames.close.assert_called_once()
    other.slicer.close.assert_called_once()
    animation.widget.close.assert_called_once()
    canvas.close.assert_called_once()


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
        (2, 3),
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


def test_capture_synchronization_discards_an_in_flight_worker_read():
    pytest.importorskip("fastplotlib.widgets.nd_widget")
    started, release = Event(), Event()
    show = Mock()

    def read(time):
        started.set()
        assert release.wait(timeout=5)
        return time

    graphic = backend._frames_graphic(
        SimpleNamespace(),
        {"time": 0.0},
        (2, 3),
        lambda time: int(time),
        read,
        show,
        SimpleNamespace(),
    )

    async def verify():
        pending = asyncio.create_task(graphic._set_indices_({"time": 0.5}))
        assert await asyncio.to_thread(started.wait, 5)
        synchronizing = asyncio.create_task(asyncio.to_thread(graphic.synchronize))
        # Synchronization invalidates writes before waiting for the worker.
        while graphic._generation == 0:
            await asyncio.sleep(0)
        release.set()
        await synchronizing
        await pending
        show.assert_not_called()
        await graphic._set_indices_({"time": 0.75})
        show.assert_called_once_with(0.75, 0.75)

    try:
        asyncio.run(asyncio.wait_for(verify(), timeout=10))
    finally:
        release.set()
        graphic.close()


def test_stale_time_series_read_cannot_overwrite_restored_viewer():
    update = Mock()
    series = SimpleNamespace(indices={"time": 0.25}, _update_graphic=update)
    backend._guard_series_updates(series)
    with pytest.raises(CancelledError):
        series._update_graphic({"data": "old window"}, {"time": 0.75})
    update.assert_not_called()
    series._update_graphic({"data": "restored window"}, {"time": 0.25})
    update.assert_called_once_with({"data": "restored window"}, {"time": 0.25})


def _capture_view():
    class Indices(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._fetch_rev = {}

    animation = object.__new__(backend.FastplotlibAnimation)
    from threading import RLock

    animation._source_lock = RLock()
    animation._closed = False
    animation.show = Mock()
    animation._frames = Mock(pause=False)
    other = Mock(pause=True)
    animation.widget = SimpleNamespace(
        indices=Indices(time=0.25),
        ui_sliders=SimpleNamespace(_playing={"time": True}),
        ndgraphics=[animation._frames, other],
    )
    camera = Mock()
    camera.get_state.return_value = {"width": 12.0}
    animation.figure = MagicMock()
    animation.figure.__iter__.return_value = [SimpleNamespace(camera=camera)]
    return animation, camera, other


def test_initial_capture_synchronization_failure_restores_controls():
    pytest.importorskip("fastplotlib.widgets.nd_widget")
    animation, camera, other = _capture_view()
    animation._frames.synchronize.side_effect = RuntimeError("worker failed")
    with (
        pytest.raises(RuntimeError, match="worker failed"),
        animation.capture_session(),
    ):
        pytest.fail("A failed setup must never yield the capture callable")
    assert animation.widget.indices["time"] == 0.25
    assert animation.widget.ui_sliders._playing["time"]
    assert animation._frames.pause is False
    assert other.pause is True
    camera.set_state.assert_called_once_with({"width": 12.0})


def test_failed_time_restore_still_restores_cameras_and_playback():
    pytest.importorskip("fastplotlib.widgets.nd_widget")
    animation, camera, other = _capture_view()
    animation._read = Mock(side_effect=RuntimeError("read failed"))
    with pytest.raises(RuntimeError, match="read failed"), animation.capture_session():
        animation.widget.indices["time"] = 0.8
    assert animation.widget.indices["time"] == 0.25
    assert animation.widget.ui_sliders._playing["time"]
    assert animation._frames.pause is False
    assert other.pause is True
    camera.set_state.assert_called_once_with({"width": 12.0})


def test_worker_payload_owns_each_source_row_and_video_frame():
    """Sources may reuse the same buffer across fields and later worker reads."""
    from threading import RLock

    scratch = [np.empty(2), np.empty(2)]
    pixels = np.empty((2, 2, 3), dtype=np.uint8)

    def sample_fields(time):
        # Like a lazy reader, each field reuses its row buffer on the next read.
        for buffer, base in zip(scratch, (1, 10), strict=True):
            buffer[:] = base + time
        return scratch

    def read_video(index):
        pixels[:] = index + 20
        return pixels

    animation = object.__new__(backend.FastplotlibAnimation)
    animation._source_lock = RLock()
    animation._clock = np.array([0.0, 1.0])
    animation._start = 0.0
    animation._field_at_time = sample_fields
    animation._tracks = []
    animation._videos = [
        SimpleNamespace(
            data=SimpleNamespace(get_frame=read_video),
            clock=SimpleNamespace(nearest=lambda time: int(time)),
        )
    ]
    first = animation._read(0.0)
    second = animation._read(1.0)
    np.testing.assert_array_equal(first.values[0], [1, 1])
    np.testing.assert_array_equal(first.values[1], [10, 10])
    np.testing.assert_array_equal(second.values[0], [2, 2])
    np.testing.assert_array_equal(second.values[1], [11, 11])
    np.testing.assert_array_equal(first.images[0], 20)
    np.testing.assert_array_equal(second.images[0], 21)
