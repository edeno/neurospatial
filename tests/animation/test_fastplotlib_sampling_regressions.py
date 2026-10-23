"""Native GPU overlay sampling and lifecycle tests independent of GPU access."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from itertools import pairwise
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import (
    Animation,
    BodypartOverlay,
    HeadDirectionOverlay,
    ObjectVectorOverlay,
    PositionOverlay,
)
from neurospatial.animation._backend_view import _CloseNotification
from neurospatial.animation._prepared import PreparedScene
from neurospatial.animation.backends import fastplotlib_backend as backend
from neurospatial.animation.overlays import VideoData
from neurospatial.environment._fastplotlib import perceived_alpha


class Plot:
    """Capture backend graphic updates without a graphics adapter."""

    def add_line(self, data, **kwargs):
        return SimpleNamespace(data=data, world_object=SimpleNamespace(), **kwargs)

    add_scatter = add_line


@pytest.fixture
def env():
    return Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool), (np.arange(0.0, 31, 10), np.arange(0.0, 31, 10))
    )


@contextmanager
def drawn(env, layer, overlays, kind):
    """``layer`` drawing the scene's first ``kind`` overlay, and its update."""
    times = np.array([0.0, 1.0, 2.0])
    with Animation(
        env, np.ones((3, env.n_bins)), frame_times=times, overlays=overlays
    ) as scene:
        prepared = PreparedScene(scene)
        graphic = layer([Plot()], prepared, getattr(scene._overlay_data, kind)[0])

        def update(time):
            graphic.apply(time, graphic.prepare(time))

        yield graphic, update


def position(interp="linear", *, break_interval=False):
    return PositionOverlay(
        np.array([[0.0, 2.0], [10.0, 4.0], [20.0, 6.0]]),
        times=np.array([0.0, 1.0, 2.0]),
        trail_length=3,
        interp=interp,
        max_gap=None,
        break_before=np.array([False, break_interval, False]),
    )


@pytest.mark.parametrize(
    "interp, expected", [("linear", [2.5, 2.5]), ("nearest", [0.0, 2.0])]
)
def test_position_uses_actual_time_on_observation_clock(env, interp, expected):
    with drawn(env, backend._Position, [position(interp)], "positions") as (
        layer,
        update,
    ):
        update(0.25)
        np.testing.assert_allclose(layer.markers[0].data[0, :2], expected)


def test_position_trail_never_bridges_explicit_tracking_break(env):
    with drawn(
        env, backend._Position, [position(break_interval=True)], "positions"
    ) as (layer, update):
        update(0.25)
        assert np.isnan(layer.markers[0].data[0, :2]).all()
        update(2.0)
        trail = layer.trails[0].data[:, :2]
        # Consecutive finite vertices may join 10->20, never the unsupported 0->10.
        for left, right in pairwise(trail):
            if np.isfinite([left, right]).all():
                np.testing.assert_allclose([left, right], [[10, 4], [20, 6]])
        assert np.isfinite(trail).any()


def test_trail_segments_fade_with_their_age(env):
    # Each segment's opacity is the shared rule's, by the age of its newer
    # sample, not by its place in the line.
    with drawn(env, backend._Position, [position()], "positions") as (layer, update):
        update(2.0)
        _, alphas = layer.prepared.trail(layer.data, 2.0)
        colors = layer.trails[0].colors
        np.testing.assert_allclose(
            colors[: 3 * len(alphas), 3], np.repeat(perceived_alpha(alphas), 3)
        )
        assert alphas[0] < alphas[-1]


def test_heading_wraps_at_pi_and_hides_when_position_anchor_is_missing(env):
    headings = HeadDirectionOverlay(
        np.deg2rad([170.0, -170.0, -170.0]),
        times=np.array([0.0, 1.0, 2.0]),
        length=1.0,
        max_gap=None,
    )
    with drawn(
        env, backend._HeadDirection, [position(), headings], "head_directions"
    ) as (
        layer,
        update,
    ):
        update(0.5)
        np.testing.assert_allclose(
            layer.lines[0].data[:, :2], [[5, 3], [4, 3]], atol=1e-7
        )
    with drawn(
        env,
        backend._HeadDirection,
        [position(break_interval=True), headings],
        "head_directions",
    ) as (layer, update):
        update(0.5)
        assert np.isnan(layer.lines[0].data[:, :2]).all()


def test_bodyparts_and_object_vectors_sample_each_observed_time(env):
    times = np.array([0.0, 1.0, 2.0])
    nose = np.array([[0.0, 0.0], [4.0, 2.0], [4.0, 2.0]])
    for breaks, expected in ((False, [2.0, 1.0]), (True, [np.nan, np.nan])):
        parts = BodypartOverlay(
            {"nose": nose},
            times=times,
            max_gap=None,
            break_before=np.array([False, breaks, False]),
        )
        with drawn(env, backend._Bodyparts, [parts], "bodypart_sets") as (
            layer,
            update,
        ):
            update(0.5)
            np.testing.assert_allclose(layer.points[0].data[0, :2], expected)
    vectors = ObjectVectorOverlay(
        np.array([[8.0, 8.0]]),
        nose,
        times=times,
        firing_rates=np.array([0.0, 10.0, 10.0]),
        show_objects=False,
        max_gap=None,
    )
    with drawn(env, backend._ObjectVectors, [vectors], "object_vectors") as (
        layer,
        update,
    ):
        update(0.5)
        np.testing.assert_allclose(
            layer.lines[0].data[:2, :2], [[2.0, 1.0], [8.0, 8.0]]
        )
        np.testing.assert_allclose(layer.lines[0].colors[3], 0.55)


def test_native_read_uses_recording_time_and_seek_uses_recording_time():
    from threading import RLock

    animation = object.__new__(backend.FastplotlibAnimation)
    animation._source_lock = RLock()
    animation._closed = False
    animation._clock = np.array([0.0, 1.0])
    animation._start = 100.0
    animation.widget = SimpleNamespace(indices={"time": 0.0})
    layer = Mock()
    animation._updates = [layer]
    frame = animation._read(0.75)
    layer.prepare.assert_called_once_with(100.75)
    assert frame.time == 100.75
    animation.seek(100.75)
    assert animation.widget.indices == {"time": 0.75}
    with pytest.raises(ValueError, match="finite"):
        animation.seek(np.nan)


def test_native_close_releases_graphics_and_canvas_once():
    animation = object.__new__(backend.FastplotlibAnimation)
    animation._closed = False
    animation._key_registered = False
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


def _capture_view():
    class Indices(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._fetch_rev = {}

    animation = object.__new__(backend.FastplotlibAnimation)
    from threading import RLock

    animation._source_lock = RLock()
    animation._closed = False
    animation._start = 0.0
    animation._prepared = Mock(scene=SimpleNamespace(frame_times=np.array([0.0, 1.0])))
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
    animation._prepare = Mock(side_effect=RuntimeError("read failed"))
    with pytest.raises(RuntimeError, match="read failed"), animation.capture_session():
        animation.widget.indices["time"] = 0.8
    assert animation.widget.indices["time"] == 0.25
    assert animation.widget.ui_sliders._playing["time"]
    assert animation._frames.pause is False
    assert other.pause is True
    camera.set_state.assert_called_once_with({"width": 12.0})


def test_worker_payload_owns_each_source_row_and_video_frame(env):
    """Sources may reuse the same buffer across fields and later worker reads."""
    scratch = np.empty(env.n_bins)

    class Rows:
        # Like a lazy reader, each read reuses one row buffer.
        shape = (2, env.n_bins)

        def __len__(self):
            return 2

        def __getitem__(self, index):
            scratch[:] = index + 1
            return scratch

    pixels = np.empty((2, 2, 3), dtype=np.uint8)

    class Reader:
        # Like a decoder, each read reuses one frame buffer.
        frame_size_px = (2, 2)

        def __getitem__(self, index):
            pixels[:] = index + 20
            return pixels

    class Video:
        def convert_to_data(self, frame_times, n_frames, env):
            return VideoData(
                np.arange(2),
                Reader(),
                None,
                (0, 2, 0, 2),
                0.5,
                "above",
                times=frame_times,
            )

    with Animation(env, Rows(), frame_times=[0.0, 1.0], overlays=[Video()]) as scene:
        prepared = PreparedScene(scene)
        field = backend._Field(None, prepared, scene.fields["field"])
        video = object.__new__(backend._Video)
        video.prepared, video.data = prepared, scene._overlay_data.videos[0]
        first = field.prepare(0.0), video.prepare(0.0)
        second = field.prepare(1.0), video.prepare(1.0)
    np.testing.assert_array_equal(first[0], 1)
    np.testing.assert_array_equal(second[0], 2)
    np.testing.assert_array_equal(first[1], 20)
    np.testing.assert_array_equal(second[1], 21)
