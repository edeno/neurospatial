"""animate_fields owns the scene it builds, and the readers the scene opens.

Exports release them when the file is written or the export fails; views
release them when the view closes. Readers a custom adapter returns are
borrowed and stay open.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from neurospatial import Environment
from neurospatial.animation._resources import _OwnedReaders
from neurospatial.animation.core import animate_fields
from neurospatial.animation.overlays import (
    PositionData,
    PositionOverlay,
    TimeSeriesOverlay,
    VideoData,
    VideoOverlay,
    _convert_overlays_to_data,
)


@pytest.fixture
def env():
    return Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )


class Reader:
    n_frames = 4
    original_size_px = frame_size_px = (2, 2)

    def __init__(self, *, fail=False, fail_read=False):
        self.close_count = 0
        self.fail = fail
        self.fail_read = fail_read

    def __getitem__(self, index):
        assert self.close_count == 0
        if self.fail_read:
            raise ValueError("decode failed")
        return np.full((2, 2, 3), 100, dtype=np.uint8)

    def close(self):
        self.close_count += 1
        if self.fail:
            raise RuntimeError("decoder close failed")


class VideoAdapter:
    """A custom adapter: the reader it returns stays its own."""

    def __init__(self, reader):
        self.reader = reader
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
            times=frame_times,
        )


@pytest.fixture
def file_video(monkeypatch):
    """A VideoOverlay of a file, whose reader the scene opens and owns."""
    opened = []

    def make(**options):
        def open_reader(*args, **kwargs):
            opened.append(Reader(**options))
            return opened[-1]

        monkeypatch.setattr("neurospatial._video_io._VideoReader", open_reader)
        return VideoOverlay("video.mp4"), opened

    warnings.filterwarnings("ignore", "VideoOverlay has no calibration")
    return make


def run(env, overlay, backend="widget", **kwargs):
    if backend != "napari":
        kwargs.setdefault("dpi", 20)
    return animate_fields(
        env,
        np.ones((4, env.n_bins)),
        frame_times=np.arange(4.0) * 0.1,
        overlays=[overlay],
        backend=backend,
        **kwargs,
    )


def _widget():
    pytest.importorskip("ipywidgets")


@pytest.mark.parametrize(
    "kwargs", [{"backend": "unknown"}, {"bad_option": True}, {"backend": "video"}]
)
def test_dispatch_errors_do_not_acquire_overlay_readers(env, kwargs):
    adapter = VideoAdapter(Reader())
    with pytest.raises((TypeError, ValueError)):
        run(env, adapter, **kwargs)
    assert adapter.calls == 0
    assert adapter.reader.close_count == 0


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("backend", ["widget", "html", "video"])
def test_failure_closes_only_owned_readers(env, file_video, owned, backend, tmp_path):
    if backend == "widget":
        _widget()
    if owned:
        overlay, opened = file_video(fail_read=True)
    else:
        overlay = VideoAdapter(Reader(fail_read=True))
        opened = [overlay.reader]
    save_path = {"html": tmp_path / "out.html", "video": tmp_path / "out.gif"}
    with pytest.raises(ValueError, match="decode failed"):
        run(env, overlay, backend=backend, save_path=save_path.get(backend))
    assert len(opened) == 1
    assert opened[0].close_count == int(owned)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("backend", ["widget", "html"])
def test_failure_survives_cleanup_failure(env, file_video, backend, tmp_path):
    """The rendering failure is raised; the failed cleanup is noted on it."""
    if backend == "widget":
        _widget()
    overlay, opened = file_video(fail=True, fail_read=True)
    save_path = tmp_path / "out.html" if backend == "html" else None
    with pytest.raises(ValueError, match="decode failed") as caught:
        run(env, overlay, backend=backend, save_path=save_path)
    assert opened[0].close_count == 1
    assert "Animation cleanup also failed" in " ".join(caught.value.__notes__)


@pytest.mark.parametrize("backend", ["html", "video"])
def test_completed_export_releases_owned_readers(env, file_video, backend, tmp_path):
    overlay, opened = file_video()
    path = tmp_path / ("out.html" if backend == "html" else "out.gif")
    assert run(env, overlay, backend=backend, save_path=path) == path
    assert path.exists()
    assert opened[0].close_count == 1


def test_worker_pickle_failure_releases_owned_reader(
    env, file_video, tmp_path, monkeypatch
):
    """A scene that cannot go to workers is refused, and its readers released."""
    import neurospatial._video_io as video_io

    overlay, opened = file_video()
    open_reader = video_io._VideoReader

    def unpicklable_reader(*args, **kwargs):
        reader = open_reader(*args, **kwargs)
        reader.callback = lambda: None  # workers would receive this reader
        return reader

    monkeypatch.setattr(video_io, "_VideoReader", unpicklable_reader)
    with pytest.raises(ValueError, match="Video overlay 0 cannot be pickled"):
        run(env, overlay, backend="html", save_path=tmp_path / "out.html", n_workers=2)
    assert opened[0].close_count == 1
    assert list(tmp_path.iterdir()) == []


def test_widget_owns_reader_until_the_view_closes(env, file_video):
    _widget()
    overlay, opened = file_video()
    view = run(env, overlay)
    reader = opened[0]
    assert reader.close_count == 0
    view.seek(0.3)
    view.native.children[1].children[0].playing = True
    view.close()
    assert not view.native.children[1].children[0].playing
    assert reader.close_count == 1
    view.close()
    assert reader.close_count == 1


@pytest.mark.napari
def test_closing_the_napari_window_releases_owned_readers(env, file_video):
    pytest.importorskip("napari")
    from qtpy.QtWidgets import QApplication

    overlay, opened = file_video()
    view = run(env, overlay, backend="napari", show=False)
    reader = opened[0]
    try:
        assert reader.close_count == 0
        view.native.window._qt_window.close()
        QApplication.processEvents()
        assert reader.close_count == 1
    finally:
        view.close()
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


def test_constant_normalization_preserves_missing_observations_in_consumers(env):
    from neurospatial.animation._prepared import PreparedScene
    from neurospatial.animation.scene import Animation

    times = np.array([0.0, 0.1, 0.2])
    overlay = TimeSeriesOverlay(np.array([5.0, np.nan, 5.0]), times, normalize=True)
    data = overlay.convert_to_data(times, 3, env)
    assert_allclose(data.data, [0, np.nan, 0], equal_nan=True)
    with Animation(
        env, np.ones((3, env.n_bins)), frame_times=times, overlays=[overlay]
    ) as scene:
        signal = scene.overlay_data().timeseries[0]
        clock, values = PreparedScene(scene).signal_window(signal, 0.1)
        assert_array_equal(clock, times)
        assert_allclose(values, [0, np.nan, 0], equal_nan=True)
        scene.render_frame(0.1, dpi=20)
        line = scene._renderer.figure.axes[-1].lines[0]
        assert_allclose(line.get_ydata(), [0, np.nan, 0], equal_nan=True)


@pytest.mark.parametrize("custom", [False, True])
def test_conversion_keeps_frame_clock_samples_and_their_support(env, custom):
    times = np.array([0.0, 0.1, 2.0, 2.1])
    positions = np.ones((4, 2))

    class Adapter:
        def convert_to_data(self, frame_times, n_frames, env):
            return PositionData(
                positions,
                "red",
                5,
                None,
                times=frame_times,
                sample_valid=np.array([True, False, True, True]),
                interval_valid=np.array([False, False, True]),
            )

    overlay = Adapter() if custom else PositionOverlay(positions)
    prepared, _ = _convert_overlays_to_data([overlay], times, 4, env)
    item = prepared.positions[0]
    assert_array_equal(item.times, times)
    # The 1.9 s interval is longer than max_gap; a rejected sample takes its
    # intervals with it.
    assert_array_equal(
        item.interval_valid, [False, False, True] if custom else [True, False, True]
    )
    if custom:
        assert_array_equal(item.sample_valid, [True, False, True, True])
