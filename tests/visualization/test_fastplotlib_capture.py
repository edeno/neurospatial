"""Native GPU capture preserves the viewer and independent scientific units."""

from __future__ import annotations

import os

os.environ.setdefault("RENDERCANVAS_FORCE_OFFSCREEN", "1")

import numpy as np
import pytest
from matplotlib import colormaps

pytest.importorskip("fastplotlib")
wgpu = pytest.importorskip("wgpu")
try:
    _ADAPTER = wgpu.gpu.request_adapter_sync()
except Exception:
    _ADAPTER = None
if _ADAPTER is None:
    pytest.skip("no GPU adapter for fastplotlib", allow_module_level=True)

from neurospatial import Environment  # noqa: E402
from neurospatial.animation import (  # noqa: E402
    Animation,
    AnimationField,
    SpikeRasterOverlay,
    TimeSeriesOverlay,
)
from neurospatial.animation.backends.fastplotlib_backend import (  # noqa: E402
    FastplotlibAnimation,
)


@pytest.fixture
def env():
    return Environment.from_grid_mask(
        np.ones((2, 1), dtype=bool),
        grid_edges=(np.array([0.0, 10, 20]), np.array([0.0, 10])),
    )


def pixel(animation, image, plot, position):
    xy = plot.map_world_to_screen((*position, 0))
    ratio = image.shape[1] / animation.figure.canvas.get_logical_size()[0]
    column, row = (np.asarray(xy[:2]) * ratio).astype(int)
    return image[row, column, :3] / 255.0


def labels(plot):
    return {
        graphic.text
        for graphic in plot.docks["right"].graphics
        if hasattr(graphic, "text")
    }


def test_fields_keep_independent_scales_labels_and_rendered_colors(env):
    animation = FastplotlibAnimation(
        env,
        {
            "posterior": np.full((2, env.n_bins), 0.5),
            "rate": np.full((2, env.n_bins), 10.0),
        },
        frame_times=np.array([100.0, 101.0]),
        _field_styles=[
            {"cmap": "viridis", "vmin": 0.0, "vmax": 1.0, "label": "probability"},
            {"cmap": "magma", "vmin": 0.0, "vmax": 20.0, "label": "Hz"},
        ],
    )
    try:
        image = animation.capture_frame(100.5)
        for i, (colormap, label) in enumerate(
            [(colormaps["viridis"], "probability"), (colormaps["magma"], "Hz")]
        ):
            plot = animation.figure[f"field {i}"]
            assert label in labels(plot)
            np.testing.assert_allclose(
                pixel(animation, image, plot, (5, 5)), colormap(0.5)[:3], atol=2 / 255
            )
        assert [scale.vmax for scale in animation._field_scales] == [1.0, 20.0]
    finally:
        animation.close()


def test_projected_track_limits_are_derived_for_each_field(env):
    track = Environment.linear_track(
        node_positions=[[0.0, 5.0], [20.0, 5.0]], bin_size=10.0
    )
    animation = FastplotlibAnimation(
        env,
        {
            "posterior": np.array([[0.25, 0.5], [0.5, 0.25]]),
            "rate": np.array([[25.0, 50.0], [50.0, 25.0]]),
        },
        frame_times=np.array([0.0, 1.0]),
        track=track,
        _field_styles=[
            {"cmap": "viridis", "vmin": 0.0, "vmax": 1.0, "label": "probability"},
            {"cmap": "magma", "vmin": 0.0, "vmax": 100.0, "label": "Hz"},
        ],
    )
    try:
        animation.capture_frame(0.5)
        first, second = animation._track_scales
        assert second.vmax == pytest.approx(100 * first.vmax)
        assert second.vmax != animation._field_scales[1].vmax
        assert animation._tracks[0].scale is first
        assert animation._tracks[1].scale is second
        assert "probability" in labels(animation.figure["track 0"])
        assert "Hz" in labels(animation.figure["track 1"])
    finally:
        animation.close()


def test_one_dimensional_track_panels_each_label_their_own_scale():
    env = Environment.from_grid_mask(
        np.ones(2, dtype=bool), grid_edges=(np.array([0.0, 10.0, 20.0]),)
    )
    animation = FastplotlibAnimation(
        env,
        {"posterior": np.ones((2, env.n_bins)), "rate": np.full((2, env.n_bins), 20.0)},
        frame_times=np.array([0.0, 1.0]),
        _field_styles=[
            {"cmap": "viridis", "vmin": 0.0, "vmax": 1.0, "label": "probability"},
            {"cmap": "magma", "vmin": 0.0, "vmax": 20.0, "label": "Hz"},
        ],
    )
    try:
        animation.capture_frame(0.5)
        assert not animation._plots
        assert [panel.scale.vmax for panel in animation._tracks] == [1.0, 20.0]
        assert "probability" in labels(animation.figure["track 0"])
        assert "Hz" in labels(animation.figure["track 1"])
    finally:
        animation.close()


def test_native_capture_uses_exact_time_and_preserves_camera_custom_graphics(env):
    animation = FastplotlibAnimation(
        env,
        np.zeros((2, env.n_bins)),
        frame_times=np.array([100.0, 101.0]),
        vmin=0.0,
        vmax=1.0,
        _field_at_time=lambda time: [np.full(env.n_bins, time - 100.0)],
    )
    try:
        animation.show()
        plot = animation.figure["field"]
        custom = plot.add_scatter(
            np.array([[15.0, 5.0, 1.0]], dtype=np.float32), colors="cyan", sizes=35
        )
        plot.camera.width *= 0.9
        camera = plot.camera.get_state()
        animation.widget.indices = {"time": 0.125}
        animation.widget.ui_sliders._playing["time"] = True
        animation.figure.canvas.set_logical_size(640, 480)
        with animation.capture_session() as capture:
            early = capture(100.25)
            late = capture(100.75)
            assert animation.widget.indices["time"] == pytest.approx(0.75)
            assert not animation.widget.ui_sliders._playing["time"]
            assert early.shape == late.shape == (480, 640, 3)
            assert early.dtype == np.uint8
            np.testing.assert_allclose(
                pixel(animation, late, plot, (5, 5)),
                colormaps["viridis"](0.75)[:3],
                atol=3 / 255,
            )
            np.testing.assert_allclose(
                pixel(animation, late, plot, (15, 5)), [0, 1, 1], atol=2 / 255
            )
            assert not np.array_equal(early, late)
        assert animation.widget.indices["time"] == pytest.approx(0.125)
        assert animation.widget.ui_sliders._playing["time"]
        for name, value in camera.items():
            np.testing.assert_array_equal(plot.camera.get_state()[name], value)
        assert custom in plot.graphics
        np.testing.assert_allclose(animation._field_graphics[0].values, 0.125)
    finally:
        animation.close()


def test_capture_failure_restores_time_playback_and_selected_unit(env):
    fields = np.zeros((2, env.n_bins))
    rates = np.arange(2 * env.n_bins, dtype=float).reshape(2, -1)
    animation = env.animate_fields(
        fields,
        frame_times=np.array([100.0, 101.0]),
        backend="fastplotlib",
        overlays=[
            SpikeRasterOverlay([np.array([100.2]), np.array([100.6])], rates=rates),
            TimeSeriesOverlay(
                np.array([1.0, 2.0]), times=np.array([100.0, 101.0]), label="speed"
            ),
        ],
    )
    try:
        animation.show()
        animation._unit.show(animation._rasters[0], 1)
        animation.widget.indices = {"time": 0.2}
        animation.widget.ui_sliders._playing["time"] = True
        with (
            pytest.raises(RuntimeError, match="writer failed"),
            animation.capture_session() as capture,
        ):
            capture(100.8)
            np.testing.assert_allclose(animation._unit.field.values, rates[1])
            series = animation._series[0][0]
            assert series.indices_displayed["time"] == pytest.approx(0.8)
            raise RuntimeError("writer failed")
        assert animation.widget.indices["time"] == pytest.approx(0.2)
        assert animation.widget.ui_sliders._playing["time"]
        assert all(not graphic.pause for graphic in animation.widget.ndgraphics)
        np.testing.assert_allclose(animation._unit.field.values, rates[1])
    finally:
        animation.close()


def test_scene_view_exports_native_3d_camera_and_independent_streams(tmp_path):
    import imageio.v2 as imageio

    env = Environment.from_grid_mask(
        np.ones((2, 2, 2), dtype=bool),
        grid_edges=tuple(np.array([0.0, 2.0, 4.0]) for _ in range(3)),
        units="cm",
    )
    with (
        Animation(
            env,
            {
                "rate": AnimationField(
                    np.repeat([[5.0], [10.0], [15.0]], env.n_bins, axis=1),
                    [100.0, 100.2, 100.4],
                    cmap="magma",
                    vmin=0.0,
                    vmax=20.0,
                    label="Hz",
                ),
                "posterior": AnimationField(
                    np.repeat([[0.1], [0.4], [0.8], [0.2]], env.n_bins, axis=1),
                    [100.0, 100.1, 100.3, 100.4],
                    cmap="viridis",
                    vmin=0.0,
                    vmax=1.0,
                    label="probability",
                ),
            },
        ) as scene,
        scene.show(backend="fastplotlib", field_rendering="points") as view,
    ):
        native = view.native
        native.figure.canvas.set_logical_size(720, 480)
        view.seek(100.15)
        baseline = native.capture_frame(100.1)
        plot = native.figure["field 0"]
        plot.camera.local.position = plot.camera.local.position + np.array(
            [1.0, 2.0, 0.0]
        )
        custom = plot.add_scatter(
            np.array([[2.0, 2.0, 2.0]], dtype=np.float32),
            colors="cyan",
            sizes=40,
        )
        camera = plot.camera.get_state()
        expected = native.capture_frame(100.1)
        assert np.mean(np.abs(expected.astype(float) - baseline)) > 1.0

        def custom_pixels(image):
            # Translucent 3D field markers blend with the cyan custom marker.
            first_panel = image[:, : image.shape[1] // 2]
            return np.count_nonzero(
                (first_panel[..., 0] < 80)
                & (first_panel[..., 1] > 170)
                & (first_panel[..., 2] > 170)
            )

        assert custom_pixels(expected) > 10
        native.widget.ui_sliders._playing["time"] = True
        view.select_time(100.1, 100.3)
        path = view.export(tmp_path / "native-3d.gif", fps=10)
        decoded = imageio.mimread(path)
        assert len(decoded) == 2
        assert decoded[0].shape == expected.shape == (480, 720, 3)
        # GIF palette reduction may round colors; the exported image must
        # still retain the edited camera and the added native graphic.
        assert np.mean(np.abs(decoded[0].astype(float) - expected)) < 1.0
        assert custom_pixels(decoded[0]) > 10
        assert not np.array_equal(decoded[0], decoded[1])
        assert view.time == pytest.approx(100.15)
        assert native.widget.ui_sliders._playing["time"]
        for name, value in camera.items():
            np.testing.assert_array_equal(plot.camera.get_state()[name], value)
        assert custom in plot.graphics
        np.testing.assert_allclose(native._field_graphics[0].values, 5.0)
        np.testing.assert_allclose(native._field_graphics[1].values, 0.4)
        assert "Hz" in labels(native.figure["field 0"])
        assert "probability" in labels(native.figure["field 1"])


def test_native_canvas_close_drains_workers_and_releases_owner(env):
    from unittest.mock import Mock

    from neurospatial.animation._resources import _OwnedReaders

    reader = Mock()
    native = FastplotlibAnimation(
        env,
        np.ones((2, env.n_bins)),
        frame_times=np.array([0.0, 0.1]),
        _reader_owner=_OwnedReaders([reader]),
    )
    native.show()
    native.figure.canvas.close()
    assert native._closed
    assert not any(native.widget.ui_sliders._playing.values())
    reader.close.assert_called_once()
    with pytest.raises(RuntimeError, match="shutdown"):
        native._frames._reader.submit(lambda: None)
    native.close()
    reader.close.assert_called_once()


def test_native_constructor_failure_drains_allocated_workers_before_owner(
    env, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from unittest.mock import Mock

    from neurospatial.animation._resources import _OwnedReaders

    allocated = []
    cleanup_order = []

    def fail_initial_show(self, time, frame):
        allocated.append(self)
        original_close = self._frames.close

        def drain():
            original_close()
            cleanup_order.append("workers")

        self._frames.close = drain
        original_canvas_close = self.figure.canvas.close

        def close_canvas():
            original_canvas_close()
            cleanup_order.append("canvas")

        monkeypatch.setattr(self.figure.canvas, "close", close_canvas)
        # Start a real read executor so the assertion verifies shutdown.
        self._frames._reader.submit(lambda: None).result()
        raise RuntimeError("Initial source read failed")

    monkeypatch.setattr(FastplotlibAnimation, "_show", fail_initial_show)
    reader = Mock()
    reader.close.side_effect = lambda: cleanup_order.append("reader")
    with pytest.raises(RuntimeError, match="Initial source read failed"):
        FastplotlibAnimation(
            env,
            np.ones((2, env.n_bins)),
            frame_times=np.array([0.0, 0.1]),
            _reader_owner=_OwnedReaders([reader]),
        )
    native = allocated[0]
    assert native._closed
    assert cleanup_order == ["workers", "canvas", "reader"]
    assert native.figure.canvas.get_closed()
    assert isinstance(native._frames._reader, ThreadPoolExecutor)
    with pytest.raises(RuntimeError, match="shutdown"):
        native._frames._reader.submit(lambda: None)
    with pytest.raises(RuntimeError, match="shutdown"):
        native._frames.slicer._executor.submit(lambda: None)
    reader.close.assert_called_once()
