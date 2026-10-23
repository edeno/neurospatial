"""Opening another view serializes lazy source reads without locking GUI work."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event, Lock, RLock, current_thread
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import Animation, AnimationField
from neurospatial.animation._prepared import PreparedScene
from neurospatial.animation.overlays import VideoData


class _Rows:
    """A scratch-buffer reader whose first worker read can be held open."""

    def __init__(self, n_bins):
        self.shape = (1000, n_bins)
        self.buffer = np.empty(n_bins)
        self.started, self.release = Event(), Event()
        self.mutex = Lock()
        self.active = 0
        self.overlaps = []
        self.reads = []

    def __len__(self):
        return self.shape[0]

    def __array__(self, *args, **kwargs):
        raise AssertionError("View preparation must keep sources lazy")

    def __getitem__(self, index):
        with self.mutex:
            self.active += 1
            self.reads.append(index)
            if self.active > 1:
                self.overlaps.append(current_thread().name)
                self.release.set()
        try:
            if current_thread().name.startswith("existing-view"):
                self.started.set()
                assert self.release.wait(5)
            self.buffer[:] = index + 1
            return self.buffer
        finally:
            with self.mutex:
                self.active -= 1


class _ContendedLock:
    """Release the existing reader when a second consumer tries its lock."""

    def __init__(self, rows):
        self.lock = RLock()
        self.rows = rows

    def __enter__(self):
        if self.rows.started.is_set():
            self.rows.release.set()
        return self.lock.__enter__()

    def __exit__(self, *args):
        return self.lock.__exit__(*args)


@contextmanager
def _existing_read(scene, rows):
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="existing-view") as pool:
        read = pool.submit(scene.sample_fields, 0.0)
        try:
            assert rows.started.wait(5)
            yield
        finally:
            rows.release.set()
        read.result(timeout=5)


@pytest.fixture
def env():
    return Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )


def _scene(env, *, explicit_limits, overlays=()):
    rows = _Rows(env.n_bins)
    field = AnimationField(
        rows,
        np.arange(1000) * 0.1,
        **({"vmin": 0, "vmax": 1000} if explicit_limits else {}),
    )
    scene = Animation(env, field, overlays=overlays)
    scene._source_lock = _ContendedLock(rows)
    return scene, field, rows


def _read_during_native_work(scene):
    # A native callback may process events or wait on an existing view worker.
    # This must finish while the new view is constructing its GUI.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(scene.sample_fields, 0.2).result(timeout=5)


@pytest.mark.parametrize("explicit_limits", [False, True])
def test_fastplotlib_view_statistics_share_lock_with_existing_reader(
    env, monkeypatch, explicit_limits
):
    from neurospatial.animation.backends import fastplotlib_backend

    scene, field, rows = _scene(env, explicit_limits=explicit_limits)
    native = SimpleNamespace(
        time=0.0, show=Mock(), close=Mock(), seek=Mock(), _add_close_callback=Mock()
    )

    def render(*args, **kwargs):
        assert kwargs["_source_lock"] is scene._source_lock
        assert kwargs["_field_styles"][0].opaque_from > 0
        _read_during_native_work(scene)
        return native

    monkeypatch.setattr(fastplotlib_backend, "render_fastplotlib", render)
    try:
        assert not field._extremes
        with _existing_read(scene, rows):
            view = scene.show(backend="fastplotlib")
        assert not rows.overlaps
        # Scaling samples at most 100 rows, plus the two individual consumers.
        assert len(rows.reads) == 102
        view.close()
    finally:
        scene.close()


class _BelowVideo:
    times = np.array([0.0, 99.9])
    interp = "nearest"

    def convert_to_data(self, frame_times, n_frames, env):
        return VideoData(
            np.array([0, 1]),
            np.zeros((2, 2, 2, 3), dtype=np.uint8),
            None,
            (0, 2, 0, 2),
            1,
            "below",
            times=self.times,
        )


def test_napari_below_video_statistics_share_lock_with_existing_reader(env):
    pytest.importorskip("napari.utils.colormaps")
    from neurospatial.animation.backends._napari_scene import NapariScene

    scene, field, rows = _scene(env, explicit_limits=True, overlays=[_BelowVideo()])
    handle = NapariScene.__new__(NapariScene)
    handle.scene = scene
    handle._time = 0.0
    handle._updates, handle.field_layers = [], {}
    initial = []

    def add(kind, data, **kwargs):
        initial.append(np.array(data, copy=True))
        _read_during_native_work(scene)
        # A different read reuses the source scratch buffer during GUI work.
        # Initial image data must already own its values.
        np.testing.assert_equal(data, initial[-1])
        assert kwargs["contrast_limits"] == (0, 1000)
        return SimpleNamespace()

    handle._add = add
    try:
        assert not field._extremes  # Nothing is sampled before the build.
        with _existing_read(scene, rows):
            # The prepared scene reads the scale statistics.
            handle._helper = PreparedScene(scene)
            handle._build_fields()
        assert not rows.overlaps
        assert len(rows.reads) == 103  # 100 scale rows and three consumers.
        assert initial and np.all(initial[0] == 1)
    finally:
        scene.close()


def test_scene_close_waits_for_view_statistics_snapshot(env, monkeypatch):
    from neurospatial.animation.backends import fastplotlib_backend

    scene, _, _rows = _scene(env, explicit_limits=True)
    reading, release, closing, closed_reader = Event(), Event(), Event(), Event()
    original_read = _Rows.__getitem__

    def read(self, index):
        reading.set()
        assert release.wait(5)
        assert not closed_reader.is_set()
        return original_read(self, index)

    monkeypatch.setattr(_Rows, "__getitem__", read)
    scene._owned_readers = SimpleNamespace(close=closed_reader.set)
    native = SimpleNamespace(
        time=0.0, show=Mock(), close=Mock(), seek=Mock(), _add_close_callback=Mock()
    )
    monkeypatch.setattr(
        fastplotlib_backend, "render_fastplotlib", Mock(return_value=native)
    )

    def close():
        closing.set()
        scene.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        opened = pool.submit(scene.show, backend="fastplotlib")
        try:
            assert reading.wait(5)
            closed = pool.submit(close)
            assert closing.wait(5)
            assert not closed_reader.is_set()
        finally:
            release.set()
        # Registration may lose the race with scene shutdown, which closes
        # the already-built native view instead of reading a released source.
        try:
            view = opened.result(timeout=5)
        except RuntimeError as error:
            assert "clos" in str(error)
        else:
            view.close()
        closed.result(timeout=5)
    assert closed_reader.is_set()
    assert native.close.called


class _ProbedRows:
    """Record whether another consumer could read while each row is read."""

    def __init__(self, n_bins):
        self.shape = (1000, n_bins)
        self.scene = None
        self.unguarded = []

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        lock = self.scene._source_lock

        def probe():
            if lock.acquire(blocking=False):
                lock.release()
                self.unguarded.append(index)

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(probe).result(timeout=5)
        return np.full(self.shape[1], index + 1.0)


def test_scene_export_reads_sources_under_scene_lock(env, tmp_path):
    # A view's worker or Animation.close() may run during an export. Each
    # field read must exclude them, including reads for every encoded frame.
    pytest.importorskip("imageio")
    rows = _ProbedRows(env.n_bins)
    scene = Animation(env, AnimationField(rows, np.arange(1000) * 0.1))
    rows.scene = scene
    try:
        scene.export(tmp_path / "scene.gif", time_range=(0.0, 0.3), fps=10)
    finally:
        scene.close()
    assert not rows.unguarded
