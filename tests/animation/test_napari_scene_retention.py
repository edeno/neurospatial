"""Caller-owned Qt viewers cannot retain a closed animation scene."""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import Animation, TimeSeriesOverlay
from neurospatial.animation.backends._napari_scene import NapariScene

pytestmark = pytest.mark.napari


def test_supplied_viewer_releases_closed_scene_controls_and_callbacks():
    import napari
    from qtpy.QtCore import QCoreApplication, QEvent
    from qtpy.QtWidgets import QApplication, QWidget

    viewer = napari.Viewer(show=True)
    caller_controls = QWidget()
    caller_dock = viewer.window.add_dock_widget(caller_controls, name="Caller controls")
    caller_layer = viewer.add_points(np.array([[0.0, 0.0]]), name="Caller layer")
    try:
        env = Environment.from_grid_mask(
            np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
        )
        times = np.array([0.0, 0.1, 0.2])
        scene = Animation(
            env,
            np.ones((3, env.n_bins)),
            frame_times=times,
            overlays=[TimeSeriesOverlay(np.array([0.0, 1.0, 0.0]), times)],
        )
        view = scene.show(backend="napari", viewer=viewer, show=True)
        handle = view._backend._controller
        references = [weakref.ref(value) for value in (scene, view, handle)]
        handle.timer.start()
        scene.close()
        assert not handle.timer.isActive()
        assert list(viewer.layers) == [caller_layer]
        assert viewer.window.dock_widgets["Caller controls"] is caller_controls
        assert viewer.window._qt_window.isVisible()
        del scene, view, handle
        for _ in range(3):
            QApplication.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            gc.collect()
        assert all(reference() is None for reference in references)
        assert caller_dock.isVisible()
        caller_layer.size = 7
        viewer.add_points(np.array([[1.0, 1.0]]), name="Next analysis")
        assert len(viewer.layers) == 2
    finally:
        viewer.close()


def test_napari_construction_preserves_original_failure_if_cleanup_also_fails(
    monkeypatch,
):
    import napari

    viewer = napari.Viewer(show=False)
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )
    scene = Animation(env, np.ones((2, env.n_bins)), frame_times=[0.0, 0.1])
    original_close = NapariScene.close

    def fail_source(self):
        raise ValueError("Original overlay preparation failure")

    def failed_cleanup(self):
        original_close(self)
        raise RuntimeError("Cleanup diagnostic")

    monkeypatch.setattr(NapariScene, "_build_overlays", fail_source)
    monkeypatch.setattr(NapariScene, "close", failed_cleanup)
    try:
        with pytest.raises(
            ValueError, match="Original overlay preparation failure"
        ) as captured:
            scene.show(backend="napari", viewer=viewer, show=False)
        assert "Cleanup diagnostic" in " ".join(captured.value.__notes__)
        assert not viewer.layers
    finally:
        scene.close()
        viewer.close()
