"""Scientific invariants for backend coordinates, missing data, and controls."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import unquote

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation._timeseries import TimeSeriesArtistManager
from neurospatial.animation.backends import html_backend, napari_backend, widget_backend
from neurospatial.animation.overlays import TimeSeriesOverlay, VideoData
from neurospatial.animation.rendering import field_to_rgb_for_napari
from neurospatial.ops.transforms import Affine2D


@pytest.fixture
def physical_grid():
    """An offset grid with unequal, non-unit physical pixel sizes."""
    return Environment.from_grid_mask(
        np.ones((3, 4), dtype=bool),
        grid_edges=(np.arange(4) * 2.0 + 100, np.arange(5) * 5.0 - 30),
    )


@pytest.mark.parametrize("file_reader", [False, True], ids=["array", "file"])
@pytest.mark.parametrize("calibrated", [False, True])
def test_video_markers_align_with_physically_scaled_field(
    physical_grid, file_reader, calibrated
):
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    reader = SimpleNamespace(frame_size_px=(30, 20))
    if file_reader:

        class Reader:
            frame_size_px = (30, 20)

            def __getitem__(self, index):
                return frame

        reader = Reader()
    else:
        reader = frame[None, ...]
    matrix = np.array([[0.2, 0.03, 100.0], [0.04, -0.5, -10.0], [0, 0, 1]])
    video = VideoData(
        reader=reader,
        frame_indices=np.array([0]),
        transform_to_env=Affine2D(matrix) if calibrated else None,
        env_bounds=(100, 106, -30, -10),
        alpha=1.0,
        z_order="above",
    )
    viewer = SimpleNamespace(add_image=Mock(return_value=SimpleNamespace(metadata={})))
    _, native = napari_backend._add_video_layer(viewer, video, physical_grid, 1)
    assert native is not file_reader
    affine = viewer.add_image.call_args.kwargs["affine"]
    # Video pixel (row=8, column=12), calibrated in (column,row) order.
    pixel = np.array([8, 12, 1.0])
    if calibrated:
        x, y, _ = matrix @ pixel[[1, 0, 2]]
    else:
        x, y = 100 + 12 * 6 / 30, -10 - 8 * 20 / 20
    # Field pixel centres occur at world row=i*5 and column=j*2.
    expected = [(-10 - y) - 2.5, (x - 100) - 1.0]
    np.testing.assert_allclose((affine @ pixel)[:2], expected)


@pytest.mark.parametrize("limits", [(0.0, 1.0), (0.0, 0.0)])
def test_napari_missing_observation_differs_from_zero(physical_grid, limits):
    lookup = np.zeros((256, 3), dtype=np.uint8)
    lookup[:, 0] = np.arange(256, dtype=np.uint8)
    field = np.zeros(physical_grid.n_bins)
    field[1] = np.nan
    rgb = field_to_rgb_for_napari(physical_grid, field, lookup, *limits)
    assert rgb.shape == (4, 3, 3)
    assert rgb.dtype == np.uint8
    # x-major source bins are transposed and y is flipped in the raster.
    np.testing.assert_array_equal(rgb[2, 0], [128, 128, 128])
    assert not np.array_equal(rgb[3, 0], rgb[2, 0])


@pytest.mark.parametrize("external_axes", [False, True])
def test_timeseries_duplicate_labels_keep_independent_traces_and_union_limits(
    external_axes,
):
    times = np.arange(3.0)
    traces = [
        TimeSeriesOverlay(
            data=np.array(values, dtype=float), times=times, label="Speed", group=group
        ).convert_to_data(frame_times=times, n_frames=3, env=None)
        for values, group in [
            ([0, 1, 2], "a"),
            ([100, 101, 102], "a"),
            ([-2, -1, 0], "b"),
        ]
    ]
    fig = plt.figure()
    try:
        if external_axes:
            axes = list(fig.subplots(2, 1))
            manager = TimeSeriesArtistManager.create_from_axes(axes, traces, times)
        else:
            manager = TimeSeriesArtistManager.create(fig, traces, times)
        manager.update(1, traces)
        assert len(manager.lines) == 3
        assert len(manager.value_texts) == 3
        for trace in traces:
            key = manager._ts_data_keys[id(trace)]
            np.testing.assert_array_equal(
                manager.lines[key].get_ydata(), trace.get_window_slice(1)[0]
            )
        assert manager.axes[0].get_ylim()[0] < 0
        assert manager.axes[0].get_ylim()[1] > 102
        assert manager.axes[1].get_ylim()[1] < 1
    finally:
        plt.close(fig)


def test_widget_throttle_flushes_latest_request_and_cancels_on_close(monkeypatch):
    """Drive an actual asyncio scheduler with a deterministic clock."""
    loop = asyncio.new_event_loop()
    now = [0.0]
    monkeypatch.setattr(loop, "time", lambda: now[0])
    monkeypatch.setattr(widget_backend.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(widget_backend.asyncio, "get_running_loop", lambda: loop)
    rendered = []
    throttle = widget_backend._FrameUpdateThrottle(rendered.append)
    try:
        throttle.request(1)
        now[0] = 0.01
        throttle.request(2)
        throttle.request(3)
        assert rendered == [1]
        now[0] = 0.04
        loop.run_until_complete(asyncio.sleep(0))
        assert rendered == [1, 3]
        now[0] = 0.05
        throttle.request(4)
        throttle.close()
        now[0] = 0.1
        loop.run_until_complete(asyncio.sleep(0))
        throttle.request(5)
        assert rendered == [1, 3]
    finally:
        loop.close()


@pytest.mark.parametrize("initial_cache_size", [0, 2])
def test_widget_without_overlays_honors_jpeg_and_returns_closeable_container(
    physical_grid, monkeypatch, initial_cache_size
):
    pytest.importorskip("ipywidgets")
    shown = []
    monkeypatch.setattr(widget_backend, "display", shown.append)
    container = widget_backend.render_widget(
        physical_grid,
        [np.zeros(physical_grid.n_bins), np.ones(physical_grid.n_bins)],
        dpi=20,
        image_format="jpeg",
        initial_cache_size=initial_cache_size,
        return_container=True,
    )
    try:
        assert shown == [container]
        image = container.children[2]
        assert image.format == "jpeg"
        assert bytes(image.value).startswith(b"\xff\xd8\xff")
        container.children[0].children[1].value = 1
        assert bytes(image.value).startswith(b"\xff\xd8\xff")
    finally:
        container.close()
        container.close()  # Cleanup is idempotent.
    assert container.comm is None
    assert container.children[0].children[1].comm is None


@pytest.mark.parametrize("directory", ["nested/frame images", "../outside #files'<>/"])
def test_external_html_urls_resolve_relative_to_html(tmp_path, directory):
    output = tmp_path / "report" / "player.html"
    frames = output.parent / directory
    html = html_backend._generate_non_embedded_html_player(
        frames, 2, ["a", "b"], 30, "test", "png", output_path=output
    )
    url = json.loads(re.search(r"const framesDir = (.*);", html).group(1))
    assert " " not in url and "#" not in url and "<" not in url and "'" not in url
    assert (output.parent / Path(unquote(url))).resolve() == frames.resolve()


def test_widget_close_cancels_pending_render_and_releases_renderer_and_link(
    physical_grid, monkeypatch
):
    pytest.importorskip("ipywidgets")
    loop = asyncio.new_event_loop()
    now = [0.0]
    monkeypatch.setattr(loop, "time", lambda: now[0])
    monkeypatch.setattr(widget_backend.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(widget_backend.asyncio, "get_running_loop", lambda: loop)
    renderer = Mock()
    renderer.render.return_value = b"frame"
    monkeypatch.setattr(
        widget_backend, "PersistentFigureRenderer", Mock(return_value=renderer)
    )
    monkeypatch.setattr(widget_backend, "display", lambda container: None)
    container = widget_backend.render_widget(
        physical_grid,
        [np.zeros(physical_grid.n_bins)] * 3,
        overlay_data=SimpleNamespace(),
        initial_cache_size=0,
        return_container=True,
    )
    link = container._links[0]
    slider = container.children[0].children[1]
    try:
        slider.value = 1
        now[0] = 0.01
        slider.value = 2  # deferred trailing request
        assert renderer.render.call_count == 2
        container.close()
        now[0] = 0.05
        loop.run_until_complete(asyncio.sleep(0))
        assert renderer.render.call_count == 2
        renderer.close.assert_called_once()
        assert link.comm is None
    finally:
        container.close()
        loop.close()


def test_external_html_export_passes_output_location_to_url_builder(
    physical_grid, tmp_path, monkeypatch
):
    from neurospatial.animation import _parallel

    monkeypatch.setattr(_parallel, "parallel_render_frames", Mock())
    report = tmp_path / "report"
    report.mkdir()
    output = report / "animation.html"
    frames = tmp_path / "assets" / "camera #1"
    html_backend.render_html(
        physical_grid,
        [np.zeros(physical_grid.n_bins)],
        save_path=output,
        embed=False,
        frames_dir=frames,
    )
    url = json.loads(re.search(r"const framesDir = (.*);", output.read_text()).group(1))
    assert (output.parent / unquote(url)).resolve() == frames.resolve()
