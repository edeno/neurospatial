"""A selected movie retains the edited view and restores interactive state."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import imageio.v2 as imageio
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import Animation, AnimationField, AnimationView


@pytest.fixture
def scene():
    env = Environment.from_samples(
        np.array([[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]]), bin_size=1
    )
    times = np.array([100.0, 100.1, 100.3])
    with Animation(
        env,
        {
            "rate": AnimationField(
                np.arange(3 * env.n_bins).reshape(3, -1),
                times,
                cmap="magma",
                vmin=0,
                vmax=100,
                label="Rate (Hz)",
            ),
            "posterior": AnimationField(
                np.ones((3, env.n_bins)) / env.n_bins,
                times,
                cmap="viridis",
                vmin=0,
                vmax=1,
                label="Probability",
            ),
        },
    ) as animation:
        yield animation


@pytest.fixture
def encoded_frames(monkeypatch):
    frames = []

    class Writer:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def append_data(self, frame):
            frames.append(frame.copy())

    monkeypatch.setattr(imageio, "get_writer", lambda *args, **kw: Writer())
    return frames


def _view(scene, *, fail=False):
    native = SimpleNamespace(time=100.25, playing=True, color=73, captures=[])

    def seek(time):
        native.time = time

    @contextmanager
    def capture_session():
        state = native.time, native.playing
        native.playing = False

        def capture(time):
            seek(time)
            native.captures.append(time)
            if fail:
                raise RuntimeError("capture failed")
            image = np.full((24, 32, 3), round((time - 100) * 100), dtype=np.uint8)
            image[0, 0] = native.color  # A user-added graphic/style.
            return image

        try:
            yield capture
        finally:
            native.time, native.playing = state

    return AnimationView._from_callbacks(
        scene,
        native,
        seek=seek,
        close=lambda: None,
        read_time=lambda: native.time,
        capture_session=capture_session,
    )


def test_native_export_preserves_edits_and_restores_clock(
    scene, encoded_frames, tmp_path
):
    with _view(scene) as view:
        view.select_time(100.1, 100.3)
        path = view.export(tmp_path / "view.gif", fps=10)
        assert path.exists()
        np.testing.assert_allclose(view.native.captures, [100.1, 100.2])
        assert view.time == 100.25
        assert view.native.playing
        assert len(encoded_frames) == 2
        assert encoded_frames[0].shape == (24, 32, 3)
        assert np.all(encoded_frames[0][0, 0] == 73)
        assert encoded_frames[0][1, 1, 0] == 10
        assert encoded_frames[1][1, 1, 0] == 20


def test_native_capture_writes_an_html_player(scene, tmp_path):
    import base64
    import io
    import json
    import re

    from PIL import Image

    with _view(scene) as view:
        view.select_time(100.1, 100.3)
        with pytest.raises(ValueError, match="n_workers=1"):
            view.export(tmp_path / "view.html", fps=10, n_workers=2)
        assert not view.native.captures
        path = view.export(tmp_path / "view.html", fps=10)
        assert view.time == 100.25
    text = path.read_text(encoding="utf-8")
    frames = json.loads(re.search(r"const frames = (\[.*?\]);", text).group(1))
    images = [np.asarray(Image.open(io.BytesIO(base64.b64decode(f)))) for f in frames]
    assert [image[1, 1, 0] for image in images] == [10, 20]
    assert np.all(images[0][0, 0] == 73)  # The user's edit is in the page.
    assert '"t = 100.100 s", "t = 100.200 s"' in text


def test_failed_capture_restores_view_and_preserves_existing_export(scene, tmp_path):
    path = tmp_path / "existing.gif"
    path.write_bytes(b"original")
    with _view(scene, fail=True) as view:
        with pytest.raises(RuntimeError, match="capture failed"):
            view.export(path, fps=10, overwrite=True)
        assert view.time == 100.25
        assert view.native.playing
        assert path.read_bytes() == b"original"
        assert not list(tmp_path.glob(".animation-*"))


def test_export_preflight_does_not_move_view(scene, tmp_path):
    with _view(scene) as view:
        path = tmp_path / "existing.gif"
        path.write_bytes(b"original")
        with pytest.raises(FileExistsError):
            view.export(path)
        with pytest.raises(ValueError, match="inside"):
            view.export(tmp_path / "bad.gif", time_range=(99, 100))
        with pytest.raises(TypeError, match="current viewer size"):
            view.export(tmp_path / "sized.gif", dpi=50)
        assert view.time == 100.25
        assert view.native.playing
        assert not view.native.captures


def test_scene_export_is_an_explicit_alternative(scene, monkeypatch):
    export = Mock(return_value="scene.gif")
    monkeypatch.setattr(scene, "export", export)
    with _view(scene) as view:
        view.select_time(100.1, 100.3)
        view.export("scene.gif", render="scene", dpi=60, figsize=(8, 4))
        export.assert_called_once_with(
            "scene.gif", time_range=(100.1, 100.3), dpi=60, figsize=(8, 4)
        )
        assert not view.native.captures


def test_encoding_failure_restores_time_and_removes_partial_file(
    scene, monkeypatch, tmp_path
):
    class BrokenWriter:
        def __enter__(self):
            raise RuntimeError("encoder unavailable")

        def __exit__(self, *exc):
            pass

    monkeypatch.setattr(imageio, "get_writer", lambda *args, **kw: BrokenWriter())
    with _view(scene) as view:
        with pytest.raises(RuntimeError, match="encoder unavailable"):
            view.export(tmp_path / "failed.mp4")
        assert view.time == 100.25
        assert view.native.playing
        assert not list(tmp_path.iterdir())


def test_widget_export_retains_display_resolution_and_restores_playback(
    scene, encoded_frames, tmp_path
):
    pytest.importorskip("ipywidgets")
    import io

    from PIL import Image

    with scene.show(dpi=30, figsize=(6, 4)) as view:
        view.seek(100.1)
        expected = np.asarray(Image.open(io.BytesIO(view.native.children[0].value)))
        play, _ = view.native.children[1].children
        play.playing = True
        view.select_time(100.1, 100.3)
        view.export(tmp_path / "widget.gif", fps=10)
        np.testing.assert_array_equal(encoded_frames[0], expected)
        assert view.time == 100.1
        assert play.playing


def test_scene_bridge_passes_independent_color_scales(scene, monkeypatch):
    from neurospatial.animation.backends import fastplotlib_backend

    native = SimpleNamespace(
        time=100.0,
        widget=SimpleNamespace(indices={"time": 0}),
        show=Mock(),
        seek=Mock(),
        close=Mock(),
        capture_session=Mock(),
        _add_close_callback=Mock(),
    )
    render = Mock(return_value=native)
    monkeypatch.setattr(fastplotlib_backend, "render_fastplotlib", render)
    with scene.show(backend="fastplotlib"):
        native.show.assert_called_once_with()
        styles = render.call_args.kwargs["_field_styles"]
        assert [
            (item.cmap, item.vmin, item.vmax, item.label, item.extend)
            for item in styles
        ] == [
            ("magma", 0, 100, "Rate (Hz)", (False, False)),
            ("viridis", 0, 1, "Probability", (False, False)),
        ]
    # Shared limits that clip a field's sampled values mark the clipped ends.
    rates = np.asarray(scene.fields["rate"].values)
    posterior = np.asarray(scene.fields["posterior"].values)
    assert rates.min() < 0.05 and rates.max() > 5
    assert posterior.min() >= 0.05 and posterior.max() <= 5
    with scene.show(backend="fastplotlib", vmin=0.05, vmax=5):
        styles = render.call_args.kwargs["_field_styles"]
        assert [item.extend for item in styles] == [(True, True), (False, False)]


def test_native_capture_rejects_window_resize_and_restores_state(
    scene, monkeypatch, tmp_path
):
    from neurospatial.animation._export import prepare_export

    calls = 0

    def render(time):
        nonlocal calls
        calls += 1
        return np.zeros((20 + calls, 32, 3), dtype=np.uint8)

    plan = prepare_export(tmp_path / "resized.gif", scene.frame_times, fps=10)
    with pytest.raises(ValueError, match="size changed"):
        plan.write(render)
    assert not list(tmp_path.iterdir())


def test_export_clock_preserves_exact_observations_and_half_open_stop(tmp_path):
    from neurospatial.animation._export import prepare_export

    # The explicit scene clock need not include this stream's observations.
    source = np.array([100.1, 100.3])
    field = AnimationField(
        np.array([[1.0], [2.0], [3.0]]),
        [100.1, 100.2, 100.3],
        break_before=[False, True, True],
    )
    # The whole recording ends on its last sample; a selection excludes stop.
    plan = prepare_export(
        tmp_path / "clock.gif", source, fps=10, sample_times=[field.times]
    )
    np.testing.assert_array_equal(plan.times, [100.1, 100.2, 100.3])
    assert [field.sample(t)[0] for t in plan.times] == [1.0, 2.0, 3.0]
    plan = prepare_export(
        tmp_path / "clock.gif",
        source,
        time_range=(100.1, 100.3),
        fps=10,
        sample_times=[field.times],
    )
    np.testing.assert_array_equal(plan.times, [100.1, 100.2])

    # A time measurably before a sample must stay before it. Snapping is
    # limited to floating-point arithmetic noise, not a time-bin tolerance.
    plan = prepare_export(
        tmp_path / "between.gif",
        source,
        time_range=(100.2 - 1e-8, 100.3),
        sample_times=[field.times],
    )
    assert plan.times[0] == 100.2 - 1e-8
    assert np.isnan(field.sample(plan.times[0])).all()
    assert np.all(plan.times < 100.3)
