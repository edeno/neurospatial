"""Native scalar scene adapters retain clocks and user-managed graphics."""

from types import SimpleNamespace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation._prepared import PreparedScene
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

pytestmark = pytest.mark.napari


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
    handle._helper = PreparedScene(scene)
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
    from neurospatial.animation.backends._napari_compat import scene_camera

    camera = scene_camera(handle.viewer)
    camera.angles = (20.0, 30.0, 40.0)
    angles = tuple(camera.angles)
    try:
        with handle.capture_session() as capture:
            frame = capture(10.2)
            assert frame.dtype == np.uint8 and frame.shape[2] == 3
        assert handle.time == 10.1
        assert tuple(camera.angles) == angles
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
    assert not hasattr(handle._helper, "figure")
    handle.timer.start()
    handle.viewer.close()
    QApplication.processEvents()
    assert handle._closed
    assert timer_deleted or not handle.timer.isActive()
    assert not hasattr(handle._helper, "figure")
    assert not scene._closed
    scene.close()


@pytest.mark.parametrize("mapping", [[2, -1, 0], [-1, 2, 0], [-1, -1, -1]])
def test_native_video_uses_custom_mapping_and_hides_missing_rows(mapping):
    from neurospatial.animation.overlays import VideoData

    class MappedVideo:
        times = np.array([0.0, 0.1, 0.2])
        interp = "nearest"

        def convert_to_data(self, frame_times, n_frames, env):
            frames = np.stack(
                [np.full((2, 2, 3), value, dtype=np.uint8) for value in [20, 100, 200]]
            )
            return VideoData(
                np.array(mapping),
                frames,
                None,
                (0, 2, 0, 2),
                1,
                "above",
                times=self.times,
            )

    class Layers(list):
        def move(self, old, new):
            self.insert(new, self.pop(old))

    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )
    scene = Animation(
        env, np.ones((2, env.n_bins)), frame_times=[0.0, 0.2], overlays=[MappedVideo()]
    )
    handle = NapariScene.__new__(NapariScene)
    handle.scene = scene
    handle.viewer = ModelViewer()
    handle.viewer.layers = Layers()
    handle._closed = False
    handle._time = 0.0
    handle._helper = PreparedScene(scene)
    handle._panels = None
    handle._layers, handle._updates, handle.field_layers = [], [], {}
    handle.time_control = SimpleNamespace(
        blockSignals=lambda _: None, setValue=lambda _: None
    )
    handle._build_fields()
    handle._build_overlays()
    video = next(layer for layer in handle.viewer.layers if layer.name == "Video 1")
    for time, mapped in zip(
        [0.0, 0.1, 0.2, 0.1, 0.0],
        [mapping[0], mapping[1], mapping[2], mapping[1], mapping[0]],
        strict=True,
    ):
        handle.seek(time)
        assert video.visible == (mapped >= 0)
        if mapped >= 0:
            assert video.data[0, 0, 0] == [20, 100, 200][mapped]
    scene.close()


def test_screen_markers_have_fixed_canvas_diameter_in_3d():
    from neurospatial.animation.backends._napari_compat import scene_camera

    env = Environment.from_grid_mask(
        np.ones((2, 2, 2), dtype=bool),
        tuple(np.array([0.0, 0.05, 0.1]) for _ in range(3)),
    )
    times = np.array([0.0, 0.1])
    scene = Animation(
        env,
        np.ones((2, env.n_bins)),
        frame_times=times,
        overlays=[PositionOverlay(np.full((2, 3), 0.05), times, size=12)],
    )
    handle = model_handle(scene)
    handle.seek(0.05)
    point = next(layer for layer in handle.viewer.layers if layer.name == "Position 1")
    np.testing.assert_equal(point.canvas_size_limits, (12, 12))
    # Modern accessors must not evaluate deprecated legacy properties.
    modern = SimpleNamespace(scene=SimpleNamespace(camera="modern"))
    assert scene_camera(modern) == "modern"
    scene.close()


def test_held_native_field_does_not_reread_or_mutate_layer():
    class Rows:
        shape = (3, 4)

        def __init__(self):
            self.reads = []

        def __len__(self):
            return 3

        def __getitem__(self, index):
            self.reads.append(index)
            return np.full(4, index, dtype=float)

    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )
    rows = Rows()
    scene = Animation(env, AnimationField(rows, [0.0, 0.2, 0.4], vmin=0, vmax=2))
    handle = model_handle(scene)
    rows.reads.clear()
    handle.seek(0.01)
    layer = handle.field_layers["field"]
    image = layer.data
    handle.seek(0.05)
    handle.seek(0.19)
    assert rows.reads == []  # first row was retained during construction
    assert layer.data is image
    handle.seek(0.2)
    assert rows.reads == [1]
    scene.close()


def test_async_scrubs_coalesce_and_playback_does_not_starve(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, get_ident

    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )
    scene = Animation(env, np.arange(12).reshape(3, 4), frame_times=[0.0, 0.2, 0.4])
    handle = model_handle(scene)
    handle._generation, handle._inflight, handle._pending = 0, None, None
    handle._capturing = False
    handle._source_lock = handle._helper.source_lock
    handle._bridge = SimpleNamespace(ready=SimpleNamespace(emit=lambda _: None))
    handle.timer = SimpleNamespace(isActive=lambda: False)
    entered, release = Event(), Event()
    calls, applies = [], []
    main_thread = get_ident()
    original = handle._prepare

    def slow_prepare(time):
        calls.append((time, get_ident()))
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        return original(time)

    original_apply = handle._apply

    def apply(time, samples):
        applies.append((time, get_ident()))
        original_apply(time, samples)

    monkeypatch.setattr(handle, "_prepare", slow_prepare)
    monkeypatch.setattr(handle, "_apply", apply)
    with ThreadPoolExecutor(max_workers=1) as handle._executor:
        handle._request_seek(0.05)
        assert entered.wait(5)
        first = handle._inflight
        handle._request_seek(0.1)
        handle._request_seek(0.15)
        handle._request_seek(0.2)
        assert len(calls) == 1 and handle._pending[1] == 0.2
        release.set()
        first.result(timeout=5)
        handle._prepared_ready((1, 0.05, first))
        assert not applies  # obsolete scrub result must not flash onscreen
        second = handle._inflight
        second.result(timeout=5)
        handle._prepared_ready((handle._generation, 0.2, second))
        assert handle.time == 0.2 and len(calls) == 2
        handle._request_seek(0.25, playback=True)
        third = handle._inflight
        handle._request_seek(0.3, playback=True)
        handle._request_seek(0.35, playback=True)
        third.result(timeout=5)
        handle._prepared_ready((handle._generation, 0.25, third))
        assert handle.time == 0.25  # slow playback still makes visible progress
        fourth = handle._inflight
        fourth.result(timeout=5)
        # A public exact seek wins over an already completed, queued job.
        handle.seek(0.4)
        handle._prepared_ready((handle._generation - 1, 0.35, fourth))
        assert handle.time == 0.4
    assert all(thread != main_thread for _, thread in calls[:-1])
    assert all(thread == main_thread for _, thread in applies)
    scene.close()


def test_playback_clock_tracks_elapsed_seconds(monkeypatch):
    import neurospatial.animation.backends._napari_scene as module

    handle = NapariScene.__new__(NapariScene)
    handle.scene = SimpleNamespace(frame_times=np.array([100.0, 110.0]))
    handle._time = 100.0
    handle._play_origin, handle._play_started, handle._speed = 100.0, 50.0, 2.0
    requested = []
    handle._request_seek = lambda time, **_: requested.append(time)
    handle.timer = SimpleNamespace(stop=lambda: None)
    handle.play_button = SimpleNamespace(setText=lambda _: None)
    monkeypatch.setattr(module, "monotonic", lambda: 52.5)
    handle._tick()
    assert requested == [105.0]


@pytest.mark.napari
def test_native_async_reads_keep_qt_responsive_and_capture_discards_pending(
    monkeypatch,
):
    from threading import Event, Timer, get_ident
    from time import monotonic, sleep

    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication

    from neurospatial.animation.backends._napari_scene import build_napari_scene

    class Rows:
        shape = (6, 4)

        def __init__(self):
            self.entered, self.release = Event(), Event()
            self.block = False
            self.threads = []

        def __len__(self):
            return 6

        def __getitem__(self, index):
            if self.block:
                self.threads.append(get_ident())
                self.entered.set()
                assert self.release.wait(5)
            return np.full(4, index, dtype=float)

    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )
    rows = Rows()
    scene = Animation(env, AnimationField(rows, np.arange(6) * 0.2, vmin=0, vmax=5))
    handle = build_napari_scene(scene, show=True)
    try:
        rows.block = True
        main_thread = get_ident()
        handle.time_control.setValue(0.2)
        assert rows.entered.wait(5)
        handle.time_control.setValue(0.4)
        handle.time_control.setValue(0.6)
        # A lazy read waits on the worker while the native event loop is live.
        responsive = []
        QTimer.singleShot(0, lambda: responsive.append(True))
        QApplication.processEvents()
        assert responsive and handle.time == 0.0
        assert handle._pending[1] == 0.6
        rows.release.set()
        deadline = monotonic() + 5
        while handle.time != 0.6 and monotonic() < deadline:
            QApplication.processEvents()
            sleep(0.001)
        assert handle.time == 0.6
        assert all(thread != main_thread for thread in rows.threads)

        # An export owns exact synchronous time even when old work is queued.
        rows.release.clear()
        rows.entered.clear()
        handle.time_control.setValue(0.8)
        assert rows.entered.wait(5)
        Timer(0.02, rows.release.set).start()
        with handle.capture_session() as capture:
            image = capture(1.0)
            assert handle.time == 1.0 and image.ndim == 3
        QApplication.processEvents()
        assert handle.time == 0.6

        # Native worker errors become inspectable and never escape Qt slots.
        def fail(time):
            raise OSError("decoder read failed")

        monkeypatch.setattr(handle, "_prepare", fail)
        handle.time_control.setValue(0.2)
        deadline = monotonic() + 5
        while handle.last_error is None and monotonic() < deadline:
            QApplication.processEvents()
            sleep(0.001)
        assert isinstance(handle.last_error, OSError)
        assert not handle.timer.isActive()
    finally:
        rows.release.set()
        handle.close()
        scene.close()


def test_native_field_names_cannot_collide_with_prepared_metadata():
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )
    values = np.arange(8).reshape(2, 4)
    scene = Animation(
        env,
        {
            "__videos__": AnimationField(values, [0.0, 0.1]),
            "__panels__": AnimationField(values + 10, [0.0, 0.1]),
        },
    )
    handle = model_handle(scene)
    handle.seek(0.1)
    np.testing.assert_array_equal(
        handle.field_layers["__videos__"].data.T.ravel(), values[1]
    )
    np.testing.assert_array_equal(
        handle.field_layers["__panels__"].data.T.ravel(), values[1] + 10
    )
    scene.close()


@pytest.mark.napari
@pytest.mark.parametrize("dimensions", [2, 3])
def test_native_marker_capture_keeps_diameter_when_camera_zooms(dimensions):
    from neurospatial.animation.backends._napari_compat import scene_camera
    from neurospatial.animation.backends._napari_scene import build_napari_scene

    env = Environment.from_grid_mask(
        np.ones((2,) * dimensions, dtype=bool),
        tuple(np.array([0.0, 0.05, 0.1]) for _ in range(dimensions)),
    )
    scene = Animation(
        env,
        np.ones((2, env.n_bins)),
        frame_times=[0.0, 0.1],
        overlays=[
            PositionOverlay(
                np.full((2, dimensions), 0.05),
                np.array([0.0, 0.1]),
                size=12,
                color="red",
            ),
        ],
    )
    handle = build_napari_scene(scene, show=True)
    try:
        handle.field_layers["field"].visible = False
        camera = scene_camera(handle.viewer)
        zoom = camera.zoom
        diameters = []
        for factor in [0.5, 2.0]:
            camera.zoom = zoom * factor
            image = handle.capture_frame(0.0)
            red = (image[..., 0] > 180) & (image[..., 1] < 80) & (image[..., 2] < 80)
            y, x = np.nonzero(red)
            assert len(x), "the physical position must remain visible"
            diameters.append(max(np.ptp(x), np.ptp(y)) + 1)
        assert max(diameters) <= 32  # catches enormous world-sized 3D markers
        assert max(diameters) / min(diameters) < 1.5
    finally:
        handle.close()
        scene.close()


@pytest.mark.napari
def test_native_track_panel_keeps_collections_across_seeks_and_clears_cache():
    from neurospatial.animation.backends._napari_scene import build_napari_scene

    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )
    track = Environment.linear_track(endpoints=[[0.0, 0.5], [2.0, 0.5]], bin_size=1)
    scene = Animation(
        env, np.arange(12).reshape(3, 4), frame_times=[0.0, 0.1, 0.2], track=track
    )
    handle = build_napari_scene(scene, show=True)
    try:
        panels = handle._panels
        artists = list(panels.axes[0].collections)
        note = panels.axes[0].text(0.1, 0.5, "Track annotation")
        handle.seek(0.05)
        handle.seek(0.15)
        assert list(panels.axes[0].collections) == artists
        assert note in panels.axes[0].texts
        assert handle._helper._rows
        handle.close()
        assert not handle._helper._rows
        assert not handle._helper._held
        assert not handle._helper._video_held
        assert handle._panels is None
        assert not scene._closed
    finally:
        handle.close()
        scene.close()


def test_legacy_spatial_units_leave_frame_axes_dimensionless():
    from napari.layers import Image

    from neurospatial.animation.backends._napari_compat import set_spatial_units

    layer = Image(np.ones((2, 3, 4)))
    set_spatial_units(layer, "cm", 2)
    assert str(layer.units[0]) == "dimensionless"
    assert tuple(str(unit) for unit in layer.units[1:]) == ("centimeter", "centimeter")


def test_freeform_environment_unit_labels_do_not_break_native_layers():
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool),
        (np.arange(3.0), np.arange(3.0)),
        units="arena units",
    )
    scene = Animation(env, np.ones((2, 4)), frame_times=[0.0, 0.1])
    handle = model_handle(scene)
    handle.seek(0.05)
    assert env.units == "arena units"
    assert np.isfinite(handle.field_layers["field"].data).all()
    scene.close()


def _layout_scene():
    from shapely.geometry import Polygon

    from neurospatial.regions import Regions

    env = Environment.from_grid_mask(
        np.ones((4, 6), dtype=bool),
        (np.linspace(0.0, 40.0, 5), np.linspace(0.0, 60.0, 7)),
        units="cm",
    )
    regions = Regions()
    regions.add("reward", point=(35.0, 55.0))
    regions.add("start", polygon=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]))
    env = env.with_regions(regions)
    times = np.array([0.0, 0.5, 1.0, 1.5])
    positions = np.array([[5.0, 5.0], [20.0, 30.0], [np.nan, np.nan], [30.0, 50.0]])
    ones = np.ones((4, env.n_bins))
    return Animation(
        env,
        {
            "cellA": AnimationField(ones, times, vmin=0, vmax=1, cmap="Reds"),
            "cellB": AnimationField(ones, times, vmin=0, vmax=1, cmap="Greens"),
        },
        overlays=[
            PositionOverlay(positions, times, trail_length=2, color="blue", size=14),
            EventOverlay.from_fixed_positions(
                np.array([0.5]), np.array([[30.0, 10.0]])
            ),
        ],
        show_regions=True,
    )


@pytest.mark.napari
def test_layout_draws_overlays_and_regions_in_every_field_panel():
    from neurospatial.animation.backends._napari_compat import canvas_grid
    from neurospatial.animation.backends._napari_scene import build_napari_scene

    scene = _layout_scene()
    # The single-panel view defines the environment-to-napari transform.
    reference = model_handle(scene)
    overlay_names = [
        layer.name
        for layer in reference.viewer.layers
        if layer not in reference.field_layers.values()
    ]
    handle = build_napari_scene(scene, layout="horizontal", show=True)
    try:
        layers = list(handle.viewer.layers)
        grid = canvas_grid(handle.viewer)
        stride = len(overlay_names) + 1
        assert grid.enabled and grid.shape == (1, 2) and grid.stride == stride
        assert len(layers) == 2 * stride
        panels = {}
        for index, name in enumerate(scene.fields):
            # Napari places each run of ``stride`` layers in one panel.
            panel = {layer.name: layer for layer in layers[index * stride :][:stride]}
            assert layers[index * stride] is handle.field_layers[name]
            assert set(panel) - {handle.field_layers[name].name} == {
                f"{name}: {overlay}" for overlay in overlay_names
            }
            panels[name] = {
                overlay: panel[f"{name}: {overlay}"] for overlay in overlay_names
            }

        def assert_panels_match_reference(time):
            handle.seek(time)
            reference.seek(time)
            expected = {layer.name: layer.data for layer in reference.viewer.layers}
            for panel in panels.values():
                for overlay in overlay_names:
                    actual = panel[overlay].data
                    if isinstance(actual, list):  # shapes: one array per shape
                        assert len(actual) == len(expected[overlay])
                        for a, e in zip(actual, expected[overlay], strict=True):
                            np.testing.assert_allclose(a, e)
                    else:
                        np.testing.assert_allclose(actual, expected[overlay])

        assert_panels_match_reference(0.5)
        for panel in panels.values():
            # Environment (x, y) is napari (row, column) = (y, x).
            np.testing.assert_allclose(panel["Position 1"].data, [[30.0, 20.0]])
            np.testing.assert_allclose(panel["Events 1: event"].data, [[10.0, 30.0]])
            np.testing.assert_allclose(panel["Region: reward"].data, [[55.0, 35.0]])
            assert len(panel["Position trail 1"].data) == 1
        assert_panels_match_reference(1.0)
        for panel in panels.values():
            assert len(panel["Position 1"].data) == 0  # missing stays missing
            assert len(panel["Position trail 1"].data) == 0
        assert_panels_match_reference(1.5)
        for panel in panels.values():
            np.testing.assert_allclose(panel["Position 1"].data, [[50.0, 30.0]])

        with handle.capture_session() as capture:
            frame = capture(0.5).astype(int)
        assert handle.time == 1.5
        red, green, blue = frame[..., 0], frame[..., 1], frame[..., 2]
        reds = (red > 50) & (red > 3 * green) & (red > 3 * blue)
        greens = (green > 30) & (green > 2 * red) & (green > blue)
        markers = (blue > 150) & (red < 80) & (green < 80)
        half = frame.shape[1] // 2
        assert reds[:, :half].any() and not reds[:, half:].any()
        assert greens[:, half:].any() and not greens[:, :half].any()
        assert markers[:, :half].any() and markers[:, half:].any()
    finally:
        handle.close()
        scene.close()


@pytest.mark.napari
def test_3d_layout_draws_overlays_in_every_volume_panel():
    from neurospatial.animation.backends._napari_compat import canvas_grid
    from neurospatial.animation.backends._napari_scene import build_napari_scene

    env = Environment.from_grid_mask(
        np.ones((2, 2, 2), dtype=bool),
        tuple(np.array([0.0, 5.0, 10.0]) for _ in range(3)),
    )
    times = np.array([0.0, 1.0])
    scene = Animation(
        env,
        {"a": np.ones((2, env.n_bins)), "b": np.zeros((2, env.n_bins))},
        frame_times=times,
        overlays=[PositionOverlay(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), times)],
    )
    handle = build_napari_scene(scene, layout="horizontal", show=True)
    try:
        assert handle.viewer.dims.ndisplay == 3
        assert canvas_grid(handle.viewer).stride == 2
        handle.seek(1.0)
        for name in ("a", "b"):
            np.testing.assert_allclose(
                handle.viewer.layers[f"{name}: Position 1"].data, [[5.0, 4.0, 3.0]]
            )
        with handle.capture_session() as capture:
            assert capture(0.0).shape[2] == 3
    finally:
        handle.close()
        scene.close()


@pytest.mark.napari
def test_vertical_layout_keeps_video_below_each_panel_field():
    from neurospatial.animation.backends._napari_compat import canvas_grid
    from neurospatial.animation.backends._napari_scene import build_napari_scene
    from neurospatial.animation.overlays import VideoData

    class Video:
        times = None
        interp = "nearest"

        def convert_to_data(self, frame_times, n_frames, env):
            frames = np.zeros((n_frames, 2, 2, 3), dtype=np.uint8)
            return VideoData(
                np.arange(n_frames), frames, None, (0, 40, 0, 60), 1, "below"
            )

    env = Environment.from_grid_mask(
        np.ones((4, 6), dtype=bool),
        (np.linspace(0.0, 40.0, 5), np.linspace(0.0, 60.0, 7)),
    )
    times = np.array([0.0, 0.5])
    scene = Animation(
        env,
        {"first": np.ones((2, env.n_bins)), "second": np.zeros((2, env.n_bins))},
        frame_times=times,
        overlays=[Video(), PositionOverlay(np.full((2, 2), 20.0), times)],
    )
    handle = build_napari_scene(scene, layout="vertical", show=True)
    try:
        grid = canvas_grid(handle.viewer)
        assert grid.shape == (2, 1) and grid.stride == 3
        assert [layer.name for layer in handle.viewer.layers] == [
            "first: Video 1",
            "first",
            "first: Position 1",
            "second: Video 1",
            "second",
            "second: Position 1",
        ]
    finally:
        handle.close()
        scene.close()
