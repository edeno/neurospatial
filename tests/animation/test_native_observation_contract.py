"""Native and complete-scene consumers honor inherited support and rate scales."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import Animation, ObjectVectorOverlay, PositionOverlay
from neurospatial.animation._prepared import PreparedScene
from neurospatial.animation._style import _finite_rate_peak, _rate_opacity
from neurospatial.animation.backends import fastplotlib_backend as gpu
from neurospatial.animation.overlays import PositionData, _convert_overlays_to_data


class Plot:
    def add_line(self, data, **kwargs):
        return SimpleNamespace(data=data, **kwargs)

    add_scatter = add_line


@pytest.fixture
def env():
    return Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )


def samples(item, times):
    return gpu._Samples(
        item.times,
        lambda time: max(int(np.searchsorted(times, time, side="right")) - 1, 0),
        0.1,
        frame_times=times,
    )


@pytest.mark.parametrize("own_samples", [False, True])
@pytest.mark.parametrize(
    "interp, expected", [("linear", [0.25, 0.25]), ("nearest", [0, 0])]
)
def test_inherited_native_clock_honors_interpolation_and_recording_gap(
    env, own_samples, interp, expected
):
    times = np.array([0.0, 0.1, 1.0, 1.1])
    positions = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)
    data = _convert_overlays_to_data(
        [PositionOverlay(positions, interp=interp)],
        times,
        len(times),
        env,
        own_samples=own_samples,
    ).positions[0]
    drawn = gpu._Position(Plot(), data, samples(data, times))
    drawn.update(0.025)
    np.testing.assert_allclose(drawn.marker.data[0, :2], expected)
    drawn.update(0.5)
    assert np.isnan(drawn.marker.data[0, :2]).all()
    drawn.update(0.1)
    np.testing.assert_array_equal(drawn.marker.data[0, :2], positions[1])


def test_custom_inherited_mask_matches_scene_and_native_renderers(env):
    times = np.array([0.0, 0.1, 0.2, 0.3])

    class Masked:
        times = None
        interp = "linear"

        def convert_to_data(self, frame_times, n_frames, env):
            return PositionData(
                np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float),
                "red",
                5,
                None,
                sample_valid=np.array([True, False, True, True]),
                interval_valid=np.array([False, False, True]),
            )

    with Animation(
        env, np.ones((4, env.n_bins)), frame_times=times, overlays=[Masked()]
    ) as scene:
        item = scene.overlay_data(own_samples=True).positions[0]
        prepared = PreparedScene(scene)
        drawn = gpu._Position(Plot(), item, samples(item, times))
        for time in (0.1, 0.15, 0.25):
            drawn.update(time)
            expected = prepared.sample(item, item.data, time)
            np.testing.assert_allclose(
                drawn.marker.data[0, :2], expected, equal_nan=True
            )
        assert np.isnan(prepared.sample(item, item.data, 0.1)).all()


@pytest.mark.parametrize(
    "rates, expected_peak",
    [(None, 1), ([np.nan, np.inf], 1), ([0, 0], 1), ([0, 5, np.nan, 10], 10)],
)
def test_rate_summary_ignores_missing_and_nonfinite_observations(rates, expected_peak):
    assert (
        _finite_rate_peak(None if rates is None else np.array(rates)) == expected_peak
    )
    assert _rate_opacity(5, 10) == pytest.approx(0.55)
    assert _rate_opacity(np.nan, 10) == 0
    assert _rate_opacity(0, 10) == pytest.approx(0.1)
    assert 0 <= _rate_opacity(-10, 10) <= 1
    assert _rate_opacity(100, 10) == 1


def test_object_vector_consumers_share_full_recording_finite_peak(env, monkeypatch):
    from neurospatial.animation import _style

    calls = []
    original = _style._finite_rate_peak

    def summarize(rates):
        calls.append(True)
        return original(rates)

    monkeypatch.setattr(_style, "_finite_rate_peak", summarize)
    times = np.array([0.0, 0.1, 0.2, 0.3])
    vector = ObjectVectorOverlay(
        np.array([[1.0, 1.0]]),
        np.zeros((4, 2)),
        times,
        firing_rates=np.array([0.0, 5.0, np.nan, 10.0]),
    )
    with Animation(
        env, np.ones((4, env.n_bins)), frame_times=times, overlays=[vector]
    ) as scene:
        data = scene.overlay_data(own_samples=True).object_vectors[0]
        assert data._rate_peak == 10
        from neurospatial.animation.backends._napari_scene import NapariScene

        native_napari = NapariScene.__new__(NapariScene)
        native_napari.scene = scene
        native_napari._helper = PreparedScene(scene)
        native_napari._updates, native_napari._layers = [], []
        native_napari.viewer = SimpleNamespace(
            add_vectors=lambda points, **kwargs: SimpleNamespace(data=points, **kwargs),
            add_points=lambda points, **kwargs: SimpleNamespace(data=points, **kwargs),
        )
        native_napari._build_overlays()
        for update in native_napari._updates:
            update(0.1, {})
        assert native_napari._layers[0].opacity == pytest.approx(0.55)
        for update in native_napari._updates:
            update(0.2, {})
        assert native_napari._layers[0].opacity == 0
        native = gpu._ObjectVectors(Plot(), data, samples(data, times))
        native.update(0.1)
        assert native.top_rate == 10
        assert native.lines.colors[3] == pytest.approx(0.55)
        native.update(0.2)
        assert native.lines.colors[3] == 0
        subset = scene.overlay_data(np.array([0.0, 0.1])).object_vectors[0]
        assert subset._rate_peak == 10
        subset_native = gpu._ObjectVectors(
            Plot(), subset, samples(subset, np.array([0.0, 0.1]))
        )
        subset_native.update(0.1)
        assert subset_native.lines.colors[3] == pytest.approx(0.55)
        scene.render_frame(0.1, dpi=25)
        line = next(
            line
            for line in scene._renderer.figure.axes[0].lines
            if line.get_zorder() == 102
        )
        assert line.get_alpha() == pytest.approx(0.55)
        scene.render_frame(0.2, dpi=25)
        line = next(
            line
            for line in scene._renderer.figure.axes[0].lines
            if line.get_zorder() == 102
        )
        assert line.get_alpha() == 0
    assert len(calls) == 1  # Seeking and sampling windows do not rescan the recording.


def test_native_close_drains_workers_and_owner_after_window_failure():
    from unittest.mock import Mock

    from neurospatial.animation._resources import _OwnedReaders

    calls = []
    animation = gpu.FastplotlibAnimation.__new__(gpu.FastplotlibAnimation)
    animation._closed = False
    animation._frames = Mock()
    animation._frames.close.side_effect = lambda: calls.append("field worker")
    other = SimpleNamespace(slicer=Mock())
    other.slicer.close.side_effect = lambda: calls.append("signal worker")
    animation.widget = SimpleNamespace(
        ndgraphics=[animation._frames, other],
        close=Mock(side_effect=RuntimeError("window already destroyed")),
    )
    animation.figure = SimpleNamespace(renderer=Mock())
    animation._canvas = Mock()
    reader = Mock()
    reader.close.side_effect = lambda: calls.append("reader")
    animation._reader_owner = _OwnedReaders([reader])
    with pytest.raises(Exception) as error:
        animation.close()
    assert "window already destroyed" in str(error.value.exceptions[0])
    assert calls == ["field worker", "signal worker", "reader"]
    animation.close()
    reader.close.assert_called_once()


def test_native_capture_does_not_restore_or_resume_a_closed_controller():
    from unittest.mock import MagicMock, Mock

    pytest.importorskip("fastplotlib.widgets.nd_widget")
    animation = gpu.FastplotlibAnimation.__new__(gpu.FastplotlibAnimation)
    animation._closed = False
    animation.show = Mock()
    animation._frames = Mock(pause=False)
    other = Mock(pause=False)
    animation.widget = SimpleNamespace(
        indices={"time": 0.1},
        ui_sliders=SimpleNamespace(_playing={"time": True}),
        ndgraphics=[animation._frames, other],
        close=Mock(),
    )
    animation.figure = MagicMock()
    camera = Mock()
    animation.figure.__iter__.return_value = [SimpleNamespace(camera=camera)]
    animation._canvas = Mock()
    animation._reader_owner = Mock()
    animation._read = Mock(side_effect=AssertionError("Closed scenes cannot be read"))
    with animation.capture_session():
        animation.close()
    assert not animation.widget.ui_sliders._playing["time"]
    assert animation._frames.pause and other.pause
    animation._read.assert_not_called()
    camera.set_state.assert_not_called()


def test_napari_capture_does_not_restart_after_close_or_parent_shutdown():
    from unittest.mock import Mock

    from neurospatial.animation.backends._napari_scene import NapariScene

    native = NapariScene.__new__(NapariScene)
    native._closed = False
    native.scene = SimpleNamespace(_closed=False, _closing=False)
    native.timer = Mock()
    native.play_button = Mock()
    native._time = 0.1
    native.seek = Mock()
    with native.capture_session():
        native._closed = True
    native.seek.assert_not_called()
    native.timer.start.assert_not_called()
    native._closed = False
    native.scene._closing = True
    native.restore_state({"time": 0.1, "playing": True})
    native.seek.assert_not_called()
    native.timer.start.assert_not_called()


def test_prepared_signal_values_are_not_renormalized_by_display_metadata(env):
    from dataclasses import replace

    from neurospatial.animation import TimeSeriesOverlay

    times = np.array([0.0, 0.1, 0.2])
    signal = TimeSeriesOverlay(np.array([5.0, np.nan, 5.0]), times, normalize=True)
    with Animation(
        env, np.ones((3, env.n_bins)), frame_times=times, overlays=[signal]
    ) as scene:
        item = scene._overlay_data.timeseries[0]
        prepared = PreparedScene(scene)
        clock, values = prepared.signal_window(item, 0.1)
        np.testing.assert_array_equal(clock, times)
        np.testing.assert_array_equal(values, [0, np.nan, 0])
        custom = replace(item, global_vmin=5.0, global_vmax=10.0)
        np.testing.assert_array_equal(prepared.signal_window(custom, 0.1)[1], values)
        scene.render_frame(0.1, dpi=25)
        np.testing.assert_array_equal(
            scene._renderer.figure.axes[-1].lines[0].get_ydata(), values
        )
