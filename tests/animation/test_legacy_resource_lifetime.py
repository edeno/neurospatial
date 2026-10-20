"""Legacy dispatch owns decoders transactionally, then hands them to viewers."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from neurospatial import Environment
from neurospatial.animation._resources import _bind_cleanup, _OwnedReaders
from neurospatial.animation.core import animate_fields
from neurospatial.animation.overlays import (
    PositionData,
    PositionOverlay,
    TimeSeriesOverlay,
    VideoData,
    _convert_overlays_to_data,
)


@pytest.fixture
def env():
    return Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )


class Reader:
    n_frames = 4

    def __init__(self, *, fail=False):
        self.close_count = 0
        self.fail = fail

    def __getitem__(self, index):
        assert self.close_count == 0
        return np.full((2, 2, 3), 100, dtype=np.uint8)

    def close(self):
        self.close_count += 1
        if self.fail:
            raise RuntimeError("decoder close failed")


class VideoAdapter:
    times = None
    interp = "nearest"

    def __init__(self, reader, *, owns=True):
        self.reader, self.owns = reader, owns
        self.calls = 0

    def convert_to_data(self, frame_times, n_frames, env):
        self.calls += 1
        return VideoData(
            np.arange(n_frames),
            self.reader,
            None,
            (0, 2, 0, 2),
            1.0,
            "above",
            _owns_reader=self.owns,
        )


def run(env, adapter, backend="widget", **kwargs):
    return animate_fields(
        env,
        np.ones((4, env.n_bins)),
        frame_times=np.arange(4.0) * 0.1,
        overlays=[adapter],
        backend=backend,
        **kwargs,
    )


@pytest.mark.parametrize(
    "kwargs", [{"backend": "unknown"}, {"bad_option": True}, {"backend": "video"}]
)
def test_pure_dispatch_errors_do_not_acquire_overlay_readers(env, kwargs, monkeypatch):
    monkeypatch.setattr(
        "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
        lambda: True,
    )
    adapter = VideoAdapter(Reader())
    with pytest.raises((TypeError, ValueError)):
        run(env, adapter, **kwargs)
    assert adapter.calls == 0
    assert adapter.reader.close_count == 0


@pytest.mark.parametrize("owns", [False, True])
@pytest.mark.parametrize(
    "backend", ["widget", "napari", "fastplotlib", "video", "html"]
)
def test_dispatch_failure_closes_only_owned_readers(
    env, owns, backend, monkeypatch, tmp_path
):
    from neurospatial.animation.backends import (
        fastplotlib_backend,
        html_backend,
        napari_backend,
        video_backend,
        widget_backend,
    )

    modules = {
        "widget": widget_backend,
        "napari": napari_backend,
        "fastplotlib": fastplotlib_backend,
        "video": video_backend,
        "html": html_backend,
    }

    def fail(*args, **kwargs):
        raise ValueError("dispatch failed")

    monkeypatch.setattr(modules[backend], f"render_{backend}", fail)
    monkeypatch.setattr(video_backend, "check_ffmpeg_available", lambda: True)
    reader = Reader()
    adapter = VideoAdapter(reader, owns=owns)
    kwargs = {"save_path": str(tmp_path / "out.mp4")} if backend == "video" else {}
    with pytest.raises(ValueError, match="dispatch failed"):
        run(env, adapter, backend=backend, **kwargs)
    assert adapter.calls == 1
    assert reader.close_count == int(owns)


@pytest.mark.parametrize(
    "backend", ["video", "html", "widget", "napari", "fastplotlib"]
)
def test_completed_export_and_dry_runs_close_owned_readers(
    env, backend, monkeypatch, tmp_path
):
    from neurospatial.animation.backends import (
        fastplotlib_backend,
        html_backend,
        napari_backend,
        video_backend,
        widget_backend,
    )

    module = {
        "widget": widget_backend,
        "napari": napari_backend,
        "fastplotlib": fastplotlib_backend,
        "video": video_backend,
        "html": html_backend,
    }[backend]
    monkeypatch.setattr(module, f"render_{backend}", lambda *args, **kwargs: None)
    monkeypatch.setattr(video_backend, "check_ffmpeg_available", lambda: True)
    reader = Reader()
    kwargs = {"dry_run": True}
    if backend == "video":
        kwargs["save_path"] = str(tmp_path / "out.mp4")
    run(env, VideoAdapter(reader), backend=backend, **kwargs)
    assert reader.close_count == 1


@pytest.mark.parametrize("backend", ["video", "html"])
def test_successful_serialized_exports_release_owned_readers(
    env, backend, monkeypatch, tmp_path
):
    from neurospatial.animation.backends import html_backend, video_backend

    module = video_backend if backend == "video" else html_backend
    monkeypatch.setattr(
        module, f"render_{backend}", lambda *args, **kwargs: "completed"
    )
    monkeypatch.setattr(video_backend, "check_ffmpeg_available", lambda: True)
    reader = Reader()
    kwargs = {"save_path": str(tmp_path / "out.mp4")} if backend == "video" else {}
    assert run(env, VideoAdapter(reader), backend=backend, **kwargs) == "completed"
    assert reader.close_count == 1


def test_post_conversion_pickle_failure_releases_owned_reader(
    env, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "neurospatial.animation.core._validate_env_pickleable", lambda env: None
    )

    def fail(*args, **kwargs):
        raise ValueError("pickle validation failed")

    monkeypatch.setattr(
        "neurospatial.animation.overlays._validate_pickle_ability", fail
    )
    reader = Reader()
    with pytest.raises(ValueError, match="pickle validation failed"):
        run(
            env,
            VideoAdapter(reader),
            backend="video",
            save_path=str(tmp_path / "out.mp4"),
            n_workers=2,
        )
    assert reader.close_count == 1


@pytest.mark.parametrize("return_container", [False, True])
def test_widget_owns_reader_until_actual_container_closes(
    env, monkeypatch, return_container
):
    from neurospatial.animation.backends import widget_backend

    displayed = []
    monkeypatch.setattr(widget_backend, "display", displayed.append)
    reader = Reader()
    result = run(
        env,
        VideoAdapter(reader),
        return_container=return_container,
        initial_cache_size=1,
        dpi=20,
    )
    container = displayed[0]
    assert result is container if return_container else result is None
    assert reader.close_count == 0
    container.children[0].children[1].value = 3
    play = container.children[0].children[0]
    play.playing = True
    container.close()
    assert not play.playing
    container.close()
    assert reader.close_count == 1


def test_owner_deduplicates_and_completes_cleanup_after_one_reader_fails():
    first, second = Reader(fail=True), Reader()
    owner = _OwnedReaders([first, first, second])
    with pytest.raises(BaseException) as caught:
        owner.close()
    assert len(caught.value.exceptions) == 1
    assert first.close_count == second.close_count == 1
    owner.close()
    assert first.close_count == second.close_count == 1


def test_native_cleanup_wrapper_releases_resources_when_native_close_fails():
    reader = Reader()

    def fail():
        raise ValueError("native close failed")

    handle = SimpleNamespace(close=fail)
    _bind_cleanup(handle, _OwnedReaders([reader]).close)
    with pytest.raises(BaseException) as caught:
        handle.close()
    assert "native close failed" in str(caught.value.exceptions[0])
    assert reader.close_count == 1
    handle.close()
    assert reader.close_count == 1


@pytest.mark.napari
def test_legacy_napari_native_window_close_releases_owned_readers():
    import napari

    from neurospatial.animation.backends.napari_backend import _bind_napari_cleanup

    reader = Reader()
    viewer = napari.Viewer(show=False)
    try:
        _bind_napari_cleanup(viewer, _OwnedReaders([reader]).close)
        viewer.window._qt_window.close()
        assert reader.close_count == 1
        viewer.close()
        assert reader.close_count == 1
    finally:
        viewer.close()


def test_constant_normalization_preserves_missing_observations_in_consumers(env):
    from neurospatial.animation._prepared import PreparedScene
    from neurospatial.animation.scene import Animation

    times = np.array([0.0, 0.1, 0.2])
    overlay = TimeSeriesOverlay(np.array([5.0, np.nan, 5.0]), times, normalize=True)
    data = overlay.convert_to_data(times, 3, env)
    assert_allclose(data.data, [0, np.nan, 0], equal_nan=True)
    assert np.isnan(data.get_cursor_value(0.1))
    with Animation(
        env, np.ones((3, env.n_bins)), frame_times=times, overlays=[overlay]
    ) as scene:
        signal = scene.overlay_data(own_samples=True).timeseries[0]
        clock, values = PreparedScene(scene).signal_window(signal, 0.1)
        assert_array_equal(clock, times)
        assert_allclose(values, [0, np.nan, 0], equal_nan=True)
        scene.render_frame(0.1, dpi=20)
        line = scene._renderer.figure.axes[-1].lines[0]
        assert_allclose(line.get_ydata(), [0, np.nan, 0], equal_nan=True)


@pytest.mark.parametrize("custom", [False, True])
def test_own_sample_conversion_fills_inherited_clock_without_losing_support(
    env, custom
):
    times = np.array([0.0, 0.1, 1.0, 1.1])
    positions = np.ones((4, 2))

    class Adapter:
        times = None
        interp = "linear"

        def convert_to_data(self, frame_times, n_frames, env):
            return PositionData(
                positions,
                "red",
                5,
                None,
                sample_valid=np.array([True, False, True, True]),
                interval_valid=np.array([False, False, True]),
            )

    overlay = Adapter() if custom else PositionOverlay(positions)
    prepared = _convert_overlays_to_data([overlay], times, 4, env, own_samples=True)
    item = prepared.positions[0]
    assert_array_equal(item.times, times)
    assert_array_equal(
        item.interval_valid, [False, False, True] if custom else [True, False, True]
    )
    if custom:
        assert_array_equal(item.sample_valid, [True, False, True, True])


def test_dispatch_error_survives_cleanup_failure(env, monkeypatch):
    from neurospatial.animation.backends import widget_backend

    def fail(*args, **kwargs):
        raise ValueError("original dispatch failure")

    monkeypatch.setattr(widget_backend, "render_widget", fail)
    reader = Reader(fail=True)
    with pytest.raises(ValueError, match="original dispatch failure") as caught:
        run(env, VideoAdapter(reader))
    assert reader.close_count == 1
    assert "Resource cleanup also failed" in " ".join(caught.value.__notes__)


def test_widget_initial_frame_failure_drains_consumers_before_reader(env, monkeypatch):
    from neurospatial.animation.backends import widget_backend

    order = []

    class Renderer:
        def __init__(self, **kwargs):
            pass

        def render(self, *args, **kwargs):
            raise ValueError("initial frame failed")

        def close(self):
            order.append("renderer")

    class OwnedReader(Reader):
        def close(self):
            assert order == ["renderer"]
            order.append("reader")
            super().close()

    monkeypatch.setattr(widget_backend, "PersistentFigureRenderer", Renderer)
    reader = OwnedReader()
    with pytest.raises(ValueError, match="initial frame failed"):
        run(env, VideoAdapter(reader), initial_cache_size=0)
    assert order == ["renderer", "reader"]
    assert reader.close_count == 1


def test_partial_napari_setup_closes_native_viewer_before_reader(env, monkeypatch):
    from neurospatial.animation.backends import napari_backend

    order = []

    class Viewer:
        layers = (
            SimpleNamespace(
                metadata={
                    "windowed_manager": SimpleNamespace(
                        disconnect=lambda: order.append("manager")
                    )
                }
            ),
        )

        def close(self):
            order.append("viewer")

        def add_image(self, *args, **kwargs):
            raise ValueError("native setup failed")

    class OwnedReader(Reader):
        def close(self):
            assert order == ["manager", "viewer"]
            order.append("reader")
            super().close()

    monkeypatch.setattr(napari_backend.napari, "Viewer", lambda **kwargs: Viewer())
    monkeypatch.setattr(napari_backend, "NAPARI_AVAILABLE", True)
    reader = OwnedReader()
    with pytest.raises(ValueError, match="native setup failed"):
        run(env, VideoAdapter(reader), backend="napari")
    assert order == ["manager", "viewer", "reader"]
    assert reader.close_count == 1
