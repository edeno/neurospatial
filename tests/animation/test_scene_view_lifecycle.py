"""Scene shutdown stops every consumer before releasing shared readers."""

from __future__ import annotations

import gc
import warnings
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import Animation, AnimationView
from neurospatial.animation.overlays import VideoOverlay


def test_close_waits_for_an_active_public_source_read(env):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from neurospatial.animation import AnimationField

    reading, release, closing, reader_closed = Event(), Event(), Event(), Event()

    class Rows:
        shape = (2, env.n_bins)

        def __len__(self):
            return 2

        def __getitem__(self, index):
            reading.set()
            assert release.wait(5)
            assert not reader_closed.is_set()
            return np.ones(env.n_bins)

    scene = Animation(env, AnimationField(Rows(), [0.0, 0.1], vmin=0, vmax=1))
    scene._owned_readers = SimpleNamespace(close=reader_closed.set)

    def close():
        closing.set()
        scene.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        read = pool.submit(scene.sample_fields, 0.0)
        try:
            assert reading.wait(5)
            closed = pool.submit(close)
            assert closing.wait(5)
            assert not reader_closed.is_set()
        finally:
            release.set()
        read.result(timeout=5)
        closed.result(timeout=5)
    assert reader_closed.is_set()
    with pytest.raises(RuntimeError, match="closed"):
        scene.sample_fields(0.0)


@pytest.fixture
def env():
    return Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )


class Reader:
    n_frames = 3
    original_size_px = frame_size_px = (2, 2)

    def __init__(self, events, *, fail=False):
        self.events = events
        self.fail = fail
        self.closed = False

    def close(self):
        self.events.append("reader")
        self.closed = True
        if self.fail:
            raise RuntimeError("reader close failed")


def animation(env, reader=None):
    """A scene; with ``reader``, its video overlay opens and owns that reader."""
    if reader is None:
        return Animation(env, np.ones((3, env.n_bins)), frame_times=[0, 0.1, 0.2])
    with (
        patch("neurospatial._video_io._VideoReader", lambda *args, **kwargs: reader),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", UserWarning)  # No calibration.
        return Animation(
            env,
            np.ones((3, env.n_bins)),
            frame_times=[0, 0.1, 0.2],
            overlays=[VideoOverlay("video.mp4")],
        )


def test_scene_closes_all_views_before_reader_and_allows_pending_source_reads(env):
    events = []
    reader = Reader(events)
    scene = animation(env, reader)

    def stop(name):
        assert scene._closing and not scene._closed
        assert not reader.closed
        np.testing.assert_array_equal(scene.sample_fields(0.1)["field"], 1)
        with pytest.raises(RuntimeError, match="closing"):
            scene.show()
        events.append(name)

    first = AnimationView._from_callbacks(
        scene, object(), seek=Mock(), close=lambda: stop("first")
    )
    second = AnimationView._from_callbacks(
        scene, object(), seek=Mock(), close=lambda: stop("second")
    )
    scene.close()
    assert set(events[:-1]) == {"first", "second"}
    assert events[-1] == "reader"
    assert first._closed and second._closed and scene._closed
    assert not scene._views
    scene.close()
    first.close()
    second.close()
    assert len(events) == 3
    for operation in (lambda: first.seek(0.1), lambda: second.select_time(0, 0.1)):
        with pytest.raises(RuntimeError, match="closed"):
            operation()


def test_explicit_view_close_unregisters_and_preserves_other_consumers(env):
    scene = animation(env)
    closes = [Mock(), Mock()]
    first = AnimationView._from_callbacks(scene, object(), seek=Mock(), close=closes[0])
    second = AnimationView._from_callbacks(
        scene, object(), seek=Mock(), close=closes[1]
    )
    first.close()
    assert first not in scene._views and second in scene._views
    second.seek(0.1)
    scene.close()
    closes[0].assert_called_once_with()
    closes[1].assert_called_once_with()


def test_scene_keeps_wrapper_alive_when_caller_retains_only_native(env):
    scene = animation(env)
    native = SimpleNamespace(stopped=False)

    def stop():
        native.stopped = True

    view = AnimationView._from_callbacks(scene, native, seek=Mock(), close=stop)
    reference = weakref.ref(view)
    del view
    gc.collect()
    assert reference() is not None
    scene.close()
    assert native.stopped
    assert not scene._views


def test_time_control_already_disposed_does_not_prevent_cleanup(env):
    scene = animation(env)
    close = Mock()
    read_time = Mock(side_effect=RuntimeError("native controls already disposed"))
    view = AnimationView._from_callbacks(
        scene, object(), seek=Mock(), close=close, read_time=read_time
    )
    scene.close()
    close.assert_called_once_with()
    assert view._closed


def test_failed_close_attempts_other_views_renderer_and_every_reader(env):
    events = []
    reader = Reader(events, fail=True)
    scene = animation(env, reader)

    def failed_view_close():
        events.append("failed view")
        raise ValueError("view close failed")

    first = AnimationView._from_callbacks(
        scene, object(), seek=Mock(), close=failed_view_close
    )
    second = AnimationView._from_callbacks(
        scene, object(), seek=Mock(), close=lambda: events.append("other view")
    )

    def failed_renderer_close():
        events.append("renderer")
        raise RuntimeError("renderer close failed")

    scene._renderer = SimpleNamespace(close=failed_renderer_close)
    from neurospatial.animation._resources import _cleanup_error

    with pytest.raises(
        type(_cleanup_error("", [RuntimeError()])), match="Animation cleanup failed"
    ) as captured:
        scene.close()
    assert set(events[:2]) == {"failed view", "other view"}
    assert events[2:] == ["renderer", "reader"]
    assert len(captured.value.exceptions) == 3
    assert first._closed and second._closed and scene._closed
    assert not scene._views
    scene.close()
    assert len(events) == 4


def test_constructor_rejects_closed_parent_and_releases_new_backend(env):
    scene = animation(env)
    scene.close()
    close = Mock()
    with pytest.raises(RuntimeError, match="closed"):
        AnimationView._from_callbacks(scene, object(), seek=Mock(), close=close)
    close.assert_called_once_with()
    assert not scene._views


def test_parent_closing_during_backend_show_cleans_unregistered_native(
    env, monkeypatch
):
    from neurospatial.animation.backends import fastplotlib_backend

    scene = animation(env)
    native = SimpleNamespace(
        time=0.0, show=scene.close, close=Mock(), seek=Mock(), capture_session=Mock()
    )
    monkeypatch.setattr(
        fastplotlib_backend, "FastplotlibAnimation", Mock(return_value=native)
    )
    with pytest.raises(RuntimeError, match="closed"):
        scene.show(backend="fastplotlib")
    native.close.assert_called_once_with()
    assert not scene._views


@pytest.mark.parametrize("compat_cleanup", [False, True])
def test_context_preserves_body_failure_and_reports_cleanup_failure(
    env, monkeypatch, compat_cleanup
):
    if compat_cleanup:
        from neurospatial.animation import _resources

        # Exercise Python 3.10's grouped cleanup errors on every interpreter.
        monkeypatch.setattr(_resources, "builtins", SimpleNamespace())
    scene = animation(env)
    AnimationView._from_callbacks(
        scene,
        object(),
        seek=Mock(),
        close=Mock(side_effect=ValueError("cleanup failure")),
    )
    with pytest.raises(RuntimeError, match="body failure") as captured, scene:
        raise RuntimeError("body failure")
    assert any("cleanup failure" in note for note in captured.value.__notes__)
    assert scene._closed


def test_widget_parent_context_stops_playback_and_disconnects_controls(env):
    pytest.importorskip("ipywidgets")
    with animation(env) as scene:
        view = scene.show(dpi=25)
        native = view.native
        play, slider = native.children[1].children
        play.playing = True
        assert play.playing
    assert view._closed and not play.playing
    assert native.comm is None
    # A queued frontend update after shutdown must no longer request rendering.
    slider.value = 0.1
    play.value = 1
    assert not scene._views


def test_widget_failed_initial_render_closes_and_unregisters_view(env, monkeypatch):
    pytest.importorskip("ipywidgets")
    scene = animation(env)
    monkeypatch.setattr(
        scene, "render_frame", Mock(side_effect=RuntimeError("render failed"))
    )
    with pytest.raises(RuntimeError, match="render failed"):
        scene.show()
    assert not scene._views
    scene.close()


def test_view_close_callbacks_all_run_even_when_native_cleanup_fails(env):
    scene = animation(env)
    calls = []
    view = AnimationView._from_callbacks(
        scene,
        object(),
        seek=Mock(),
        close=Mock(side_effect=RuntimeError("native failed")),
    )

    def failed_callback():
        calls.append("failed callback")
        raise ValueError("callback failed")

    view._add_close_callback(failed_callback)
    view._add_close_callback(lambda: calls.append("other callback"))
    with pytest.raises(
        BaseException, match="Animation view cleanup failed"
    ) as captured:
        view.close()
    assert len(captured.value.exceptions) == 2
    assert calls == ["failed callback", "other callback"]
    assert not scene._views
    view.close()
    scene.close()


def test_widget_capture_does_not_restart_playback_after_parent_closes(env):
    pytest.importorskip("ipywidgets")
    scene = animation(env)
    view = scene.show(dpi=25)
    play, _ = view.native.children[1].children
    play.playing = True
    with view._backend.capture_session() as capture:
        capture(0.1)
        scene.close()
    assert view._closed
    assert not play.playing


def test_scene_close_disconnects_event_inspector_controls(env):
    pytest.importorskip("ipywidgets")
    import pandas as pd

    scene = animation(env)
    inspector = scene.inspect_events(
        pd.DataFrame(
            {
                "event_id": ["first", "second"],
                "start_time": [0, 0.1],
                "stop_time": [0.05, 0.2],
            }
        )
    )
    container = inspector.show(dpi=25)
    controls = list(inspector._controls)
    selector = controls[0]
    scene.close()
    assert inspector.view is None
    assert not inspector._observers
    assert not inspector._controls
    assert all(control.comm is None for control in controls)
    inspector.select = Mock()
    selector.value = "second"
    inspector.select.assert_not_called()
    container.close()
    inspector.close()


def test_failed_registration_preserves_primary_error_if_backend_cleanup_also_fails(env):
    scene = animation(env)
    scene.close()
    with pytest.raises(RuntimeError, match="Animation is closed") as captured:
        AnimationView._from_callbacks(
            scene,
            object(),
            seek=Mock(),
            close=Mock(side_effect=ValueError("cleanup failed")),
        )
    assert any("cleanup failed" in note for note in captured.value.__notes__)
    assert not scene._views


@pytest.mark.napari
@pytest.mark.parametrize("supplied_viewer", [False, True])
def test_live_napari_scene_close_stops_timer_before_readers_and_preserves_supplied_viewer(
    env, supplied_viewer
):
    import napari
    from qtpy.QtWidgets import QApplication

    events = []

    class VideoReader(Reader):
        def __getitem__(self, index):
            assert not self.closed
            events.append("read")
            return np.zeros((2, 2, 3), dtype=np.uint8)

    reader = VideoReader(events)
    scene = animation(env, reader)
    viewer = napari.Viewer(show=False) if supplied_viewer else None
    if viewer is not None:
        viewer.add_points([[0.5, 0.5]], name="Caller annotation")
    try:
        options = {"show": False}
        if viewer is not None:
            options["viewer"] = viewer
        view = scene.show(backend="napari", **options)
        controller = view._backend._controller
        native = view.native
        timer_deleted = []
        controller.timer.destroyed.connect(lambda *_: timer_deleted.append(True))
        controller.timer.start(5)
        assert controller.timer.isActive()
        scene.close()
        QApplication.processEvents()
        assert view._closed and controller._closed
        if not timer_deleted:
            assert not controller.timer.isActive()
        assert reader.closed and events[-1] == "reader"
        assert not scene._views
        with pytest.raises(RuntimeError, match="closed"):
            view.seek(0.1)
        if supplied_viewer:
            assert "Caller annotation" in native.layers
            native.add_points([[1.0, 1.0]], name="Still usable")
            assert {layer.name for layer in native.layers} == {
                "Caller annotation",
                "Still usable",
            }
        scene.close()
        view.close()
    finally:
        scene.close()
        if viewer is not None:
            viewer.close()
