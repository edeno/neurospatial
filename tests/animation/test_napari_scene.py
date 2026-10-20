"""Native scalar scene adapters retain clocks and user-managed graphics."""

from types import SimpleNamespace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation._scene_rendering import SceneRenderer
from neurospatial.animation.backends._napari_scene import (
    NapariScene,
    _image,
    _native_grid,
)
from neurospatial.animation.overlays import (
    BodypartOverlay,
    EventOverlay,
    HeadDirectionOverlay,
    PositionOverlay,
)
from neurospatial.animation.scene import Animation, AnimationField
from neurospatial.animation.skeleton import Skeleton


class ModelViewer:
    """Use real napari layers without requiring a Qt viewer or a display."""

    def __init__(self):
        self.layers = []
        self.text_overlay = SimpleNamespace(visible=False, text="")

    def __getattr__(self, name):
        if not name.startswith("add_"):
            raise AttributeError(name)
        from napari import layers

        classes = {
            "image": layers.Image,
            "points": layers.Points,
            "shapes": layers.Shapes,
            "vectors": layers.Vectors,
        }

        def add(data, **kwargs):
            layer = classes[name[4:]](data, **kwargs)
            self.layers.append(layer)
            return layer

        return add


def model_handle(scene):
    handle = NapariScene.__new__(NapariScene)
    handle.scene = scene
    handle.viewer = ModelViewer()
    handle._closed = False
    handle._time = float(scene.frame_times[0])
    handle._helper = SceneRenderer(scene)
    handle._panels = None
    handle._layers, handle._updates, handle.field_layers = [], [], {}
    handle.time_control = SimpleNamespace(
        blockSignals=lambda _: None, setValue=lambda _: None
    )
    handle._build_fields()
    handle._build_overlays()
    return handle


def test_native_fields_sample_independent_clocks_and_keep_native_edits():
    mask = np.ones((2, 3), dtype=bool)
    mask[0, 1] = False
    env = Environment.from_grid_mask(
        mask, (np.array([10.0, 12.0, 14.0]), np.array([20.0, 23.0, 26.0, 29.0]))
    )
    first = AnimationField(
        np.array([np.zeros(env.n_bins), np.ones(env.n_bins), np.full(env.n_bins, 2.0)]),
        [100.0, 100.2, 101.0],
        vmin=0,
        vmax=2,
        cmap="magma",
        label="Rate (Hz)",
    )
    second = AnimationField(
        np.array([np.full(env.n_bins, 10.0), np.full(env.n_bins, 20.0)]),
        [100.1, 100.3],
        vmin=10,
        vmax=20,
        cmap="viridis",
        break_before=[False, True],
    )
    scene = Animation(env, {"first": first, "second": second})
    handle = model_handle(scene)
    layer = handle.field_layers["first"]
    layer.opacity = 0.37
    handle.seek(100.15)
    assert np.all(layer.data[np.isfinite(layer.data)] == 0)
    assert np.isnan(layer.data[1, 0])  # inactive cell remains missing
    assert np.isnan(handle.field_layers["second"].data).all()  # declared break
    handle.seek(100.25)
    assert np.isnan(layer.data).all()  # long unsupported sampling interval
    handle.seek(101.0)
    assert np.all(layer.data[np.isfinite(layer.data)] == 2)
    assert handle.field_layers["first"] is layer and layer.opacity == 0.37
    np.testing.assert_allclose(layer.contrast_limits, [0, 2])
    assert layer.colormap.name == "magma"
    assert layer.name == "first (Rate (Hz))"
    assert handle.viewer.text_overlay.text == "t = 101.000 s"
    grid, affine = _native_grid(env)
    np.testing.assert_allclose(affine @ [0.0, 0.0, 1.0], [21.5, 11.0, 1.0])
    np.testing.assert_allclose(
        _image(np.arange(env.n_bins), grid).T[mask], np.arange(env.n_bins)
    )
    scene.close()


def test_native_3d_overlays_keep_z_and_persistent_layers():
    env = Environment.from_grid_mask(
        np.ones((2, 2, 2), dtype=bool),
        tuple(np.array([0.0, 5.0, 10.0]) for _ in range(3)),
    )
    times = np.array([5.0, 5.2, 5.4])
    positions = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
    scene = Animation(
        env,
        np.ones((3, env.n_bins)),
        frame_times=times,
        overlays=[
            PositionOverlay(positions, times, trail_length=3),
            BodypartOverlay(
                {"nose": positions, "tail": positions - 0.1},
                times,
                skeleton=Skeleton(
                    name="mouse", nodes=("nose", "tail"), edges=(("nose", "tail"),)
                ),
            ),
            HeadDirectionOverlay(np.array([[0.0, 0.0, 1.0]] * 3), times, length=2),
            EventOverlay.from_fixed_positions(
                np.array([5.2]), np.array([[1.0, 2.0, 7.0]])
            ),
        ],
    )
    handle = model_handle(scene)
    layers = list(handle.viewer.layers)
    handle.seek(5.2)
    names = {layer.name: layer for layer in layers}
    np.testing.assert_allclose(names["Position 1"].data, [[6.0, 4.0, 2.0]])
    np.testing.assert_allclose(names["Skeleton 1"].data[0][:, 0], [6.0, 5.9])
    np.testing.assert_allclose(
        names["Head direction 1"].data, [[[6.0, 4.0, 2.0], [2.0, 0.0, 0.0]]]
    )
    event = next(layer for layer in layers if layer.name.startswith("Events"))
    np.testing.assert_allclose(event.data, [[7.0, 2.0, 1.0]])
    handle.seek(5.3)
    assert handle.viewer.layers == layers
    np.testing.assert_allclose(names["Position 1"].data, [[7.5, 5.0, 2.5]])
    scene.close()


@pytest.mark.napari
def test_native_qt_capture_restores_time_camera_and_custom_graphics():
    from neurospatial.animation.backends._napari_scene import build_napari_scene

    env = Environment.from_grid_mask(
        np.ones((2, 2, 2), dtype=bool),
        tuple(np.array([0.0, 5.0, 10.0]) for _ in range(3)),
    )
    scene = Animation(
        env, np.arange(3 * env.n_bins).reshape(3, -1), frame_times=[10.0, 10.2, 10.4]
    )
    handle = build_napari_scene(scene, show=True)
    handle.viewer.add_points(
        [[8.0, 5.0, 2.0]], name="User annotation", face_color="red", size=4
    )
    handle.seek(10.1)
    handle.viewer.camera.angles = (20.0, 30.0, 40.0)
    angles = tuple(handle.viewer.camera.angles)
    try:
        with handle.capture_session() as capture:
            frame = capture(10.2)
            assert frame.dtype == np.uint8 and frame.shape[2] == 3
        assert handle.time == 10.1
        assert tuple(handle.viewer.camera.angles) == angles
        assert "User annotation" in handle.viewer.layers
    finally:
        handle.close()
        scene.close()


def test_irregular_cells_use_physical_bin_centers_and_keep_missing_values():
    env = Environment.from_samples(
        np.random.default_rng(3).uniform(0, 10, (100, 2)),
        bin_size=3,
        layout="hexagonal",
    )
    values = np.arange(2 * env.n_bins, dtype=float).reshape(2, -1)
    values[0, 1] = np.nan
    scene = Animation(env, AnimationField(values, [0.0, 0.2], vmin=0, vmax=7))
    handle = model_handle(scene)
    layer = handle.field_layers["field"]
    handle.seek(0.0)
    assert "bin centers" in layer.name
    np.testing.assert_allclose(layer.data, env.bin_centers[:, ::-1])
    np.testing.assert_array_equal(layer.shown, np.isfinite(values[0]))
    assert np.isnan(layer.features["value"][1])
    layer.face_contrast_limits = [1.0, 9.0]
    handle.seek(0.2)
    np.testing.assert_array_equal(layer.shown, True)
    np.testing.assert_allclose(layer.features["value"], values[1])
    np.testing.assert_allclose(layer.face_contrast_limits, [1.0, 9.0])
    scene.close()


@pytest.mark.napari
def test_scientific_dock_capture_preserves_annotations_and_panel_artists():
    from neurospatial.animation.backends._napari_scene import build_napari_scene
    from neurospatial.animation.overlays import (
        EpochOverlay,
        SpikeRasterOverlay,
        TimeSeriesOverlay,
    )

    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )
    times = np.array([10.0, 10.2, 10.4])
    scene = Animation(
        env,
        np.ones((3, env.n_bins)),
        frame_times=times,
        overlays=[
            TimeSeriesOverlay(
                np.array([0.0, 2.0, 1.0]), times, label="Speed", color="cyan"
            ),
            SpikeRasterOverlay([np.array([10.1, 10.3])], unit_ids=[77]),
            EpochOverlay([(10.0, 10.25)], label="Run"),
        ],
    )
    handle = build_napari_scene(scene, show=True)
    try:
        panels = handle._panels
        panels.axes[0].text(10.2, 1.0, "Scientist note", color="white")
        lines = list(panels.axes[0].lines)
        handle.seek(10.1)
        with handle.capture_session() as capture:
            image = capture(10.2)
            canvas = handle.viewer.screenshot(canvas_only=True)
        assert image.shape[0] > canvas.shape[0]
        assert handle.time == 10.1
        assert list(panels.axes[0].lines) == lines
        assert panels.axes[0].texts[0].get_text() == "Scientist note"
        np.testing.assert_allclose(panels.axes[0].lines[0].get_xdata(), times)
    finally:
        handle.close()
        scene.close()


@pytest.mark.napari
def test_native_window_close_stops_playback_and_releases_helper():
    from qtpy.QtWidgets import QApplication

    from neurospatial.animation.backends._napari_scene import build_napari_scene

    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )
    scene = Animation(env, np.ones((2, env.n_bins)), frame_times=[0.0, 0.2])
    handle = build_napari_scene(scene, show=True)
    timer_deleted = []
    handle.timer.destroyed.connect(lambda: timer_deleted.append(True))
    handle._helper.figure.text(0.5, 0.5, "Owned renderer")
    handle.timer.start()
    handle.viewer.close()
    QApplication.processEvents()
    assert handle._closed
    assert timer_deleted or not handle.timer.isActive()
    assert not handle._helper.figure.texts
    assert not scene._closed
    scene.close()
