"""Closing native windows completes their public view lifecycle."""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import Animation, AnimationView
from neurospatial.animation._backend_view import _CloseNotification, _NativeBackendView
from neurospatial.animation.backends.fastplotlib_backend import FastplotlibAnimation


@pytest.fixture
def scene():
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )
    with Animation(env, np.ones((2, env.n_bins)), frame_times=[0.0, 0.1]) as scene:
        yield scene


def native_controller(*, fail_cleanup=False):
    """Exercise the actual native canvas close handler without a GPU."""
    native = FastplotlibAnimation.__new__(FastplotlibAnimation)
    native._closed = False
    native._key_registered = False
    native._close_notification = _CloseNotification()
    native._start = 0.0
    native._clock = np.array([0.0, 0.1])
    native._frames = Mock()
    native.widget = SimpleNamespace(
        indices={"time": 0.1},
        ndgraphics=[native._frames],
        close=Mock(
            side_effect=RuntimeError("native widget cleanup failed")
            if fail_cleanup
            else None
        ),
    )
    native.figure = SimpleNamespace(renderer=Mock())
    native._canvas = Mock()
    return native


@pytest.mark.parametrize("fail_cleanup", [False, True])
def test_fastplotlib_native_close_unregisters_and_notifies_once(scene, fail_cleanup):
    native = native_controller(fail_cleanup=fail_cleanup)
    view = AnimationView._from_backend(scene, _NativeBackendView(native, native))
    called = Mock()
    view._add_close_callback(called)
    reference = weakref.ref(view)

    if fail_cleanup:
        with pytest.raises(Exception, match="native animation resources"):
            native._canvas_closed(None)
    else:
        native._canvas_closed(None)
    assert view._closed and native._closed
    assert view.time == 0.1
    assert not scene._views
    assert not scene._closed
    called.assert_called_once_with()
    native._frames.close.assert_called_once_with()
    native.widget.close.assert_called_once_with()
    # Native notification cannot reenter cleanup or close the canvas twice.
    native._canvas_closed(None)
    view.close()
    called.assert_called_once_with()
    native.widget.close.assert_called_once_with()
    del view
    gc.collect()
    assert reference() is None
    assert not scene._closed


def test_native_close_runs_remaining_callbacks_after_callback_failure(scene):
    native = native_controller()
    view = AnimationView._from_backend(scene, _NativeBackendView(native, native))
    failed = Mock(side_effect=ValueError("callback failed"))
    called = Mock()
    view._add_close_callback(failed)
    view._add_close_callback(called)
    with pytest.raises(Exception, match="native animation resources"):
        native._canvas_closed(None)
    assert view._closed and not scene._views
    failed.assert_called_once_with()
    called.assert_called_once_with()
    native._canvas_closed(None)
    view.close()
    called.assert_called_once_with()


def test_native_closed_before_registration_does_not_leave_a_registered_view(scene):
    native = native_controller()
    native.close()
    view = AnimationView._from_backend(scene, _NativeBackendView(native, native))
    assert view._closed
    assert not scene._views
    called = Mock()
    view._add_close_callback(called)
    called.assert_called_once_with()
    view.close()
    native.widget.close.assert_called_once_with()


@pytest.mark.napari
@pytest.mark.parametrize("supplied_viewer", [False, True])
def test_napari_native_close_unregisters_notifies_and_releases_graphics(
    scene, supplied_viewer
):
    import napari
    from qtpy.QtCore import QCoreApplication, QEvent
    from qtpy.QtWidgets import QApplication

    viewer = napari.Viewer(show=True) if supplied_viewer else None
    options = {"show": True}
    if viewer is not None:
        options["viewer"] = viewer
    view = scene.show(backend="napari", **options)
    native = view.native
    controller = view._backend._controller
    references = [
        weakref.ref(view),
        weakref.ref(controller),
        weakref.ref(next(iter(controller.field_layers.values()))),
    ]
    called = Mock()
    view._add_close_callback(called)
    controller.timer.start()
    try:
        native.close()
        for _ in range(3):
            QApplication.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert controller._closed and view._closed
        assert not controller._layers and not controller.field_layers
        assert not scene._views
        called.assert_called_once_with()
        view.close()
        called.assert_called_once_with()
        assert not scene._closed
        np.testing.assert_array_equal(next(iter(scene.sample_fields(0.1).values())), 1)
        with pytest.raises(RuntimeError, match="view is closed"):
            view.seek(0.1)
        del view, controller
        for _ in range(3):
            QApplication.processEvents()
            gc.collect()
        assert all(reference() is None for reference in references)
    finally:
        native.close()
