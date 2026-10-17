"""Every animation backend uses the options it is given, or refuses them.

A dry run writes nothing on any backend, an option the chosen backend would
ignore raises instead of changing nothing, and the suggested fix runs.
"""

import inspect
import shutil
from pathlib import Path

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation.core import _BACKEND_OPTIONS

BACKENDS = ["napari", "video", "html", "widget"]

# Values the dispatcher supplies itself rather than taking from the caller.
_SUPPLIED = {
    "napari": {
        "fps",
        "sample_rate_hz",
        "speed",
        "max_playback_fps",
        "frame_times",
        "overlay_data",
    },
    "video": {"fps", "frame_times", "overlay_data"},
    "html": {"fps", "overlay_data"},
    "widget": {"fps", "frame_times", "overlay_data"},
}


def _renderer(backend):
    module = __import__(
        f"neurospatial.animation.backends.{backend}_backend",
        fromlist=[f"render_{backend}"],
    )
    return getattr(module, f"render_{backend}")


def _available(backend):
    if backend == "video" and shutil.which("ffmpeg") is None:
        pytest.skip("the video backend needs ffmpeg")
    if backend == "napari":
        pytest.importorskip("napari")
    if backend == "widget":
        pytest.importorskip("ipywidgets")


@pytest.fixture
def animation(tmp_path, monkeypatch):
    """A 5-frame animation on a small arena, run from an empty directory."""
    monkeypatch.chdir(tmp_path)
    x, y = np.meshgrid(np.arange(0.0, 20.0, 4.0), np.arange(0.0, 20.0, 4.0))
    env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=4.0)
    fields = np.random.default_rng(0).random((5, env.n_bins))
    return env, fields, np.arange(5) / 30.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_option_table_lists_exactly_what_each_renderer_takes(backend):
    parameters = inspect.signature(_renderer(backend)).parameters
    assert not any(p.kind is p.VAR_KEYWORD for p in parameters.values())
    taken = set(parameters) - {"env", "fields"} - _SUPPLIED[backend]
    assert taken == _BACKEND_OPTIONS[backend]


@pytest.mark.parametrize("backend", BACKENDS)
def test_dry_run_writes_nothing_and_returns_none(animation, tmp_path, backend):
    _available(backend)
    env, fields, times = animation
    save_path = {"video": "out.mp4", "html": "out.html"}.get(backend)
    result = env.animate_fields(
        fields,
        frame_times=times,
        backend=backend,
        save_path=save_path,
        dry_run=True,
    )
    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_html_dry_run_without_save_path_does_not_write_the_default_file(animation):
    env, fields, times = animation
    result = env.animate_fields(fields, frame_times=times, backend="html", dry_run=True)
    assert result is None
    assert not Path("animation.html").exists()


@pytest.mark.parametrize("backend", BACKENDS)
def test_misspelled_option_names_the_closest_one_and_the_fix_runs(
    animation, tmp_path, backend
):
    _available(backend)
    env, fields, times = animation
    save_path = {"video": "out.mp4", "html": "out.html"}.get(backend)
    labels = [f"{t:.2f} s" for t in times]
    with pytest.raises(TypeError, match="unknown option 'frame_label'") as error:
        env.animate_fields(
            fields,
            frame_times=times,
            backend=backend,
            save_path=save_path,
            frame_label=labels,
        )
    assert "pass frame_labels= instead of frame_label=" in str(error.value)
    assert list(tmp_path.iterdir()) == []
    # The suggested name is accepted.
    result = env.animate_fields(
        fields,
        frame_times=times,
        backend=backend,
        save_path=save_path,
        frame_labels=labels,
        dry_run=True,
    )
    assert result is None


@pytest.mark.parametrize(
    ("backend", "option", "owner"),
    [
        ("html", {"codec": "vp9"}, "video"),
        ("napari", {"image_format": "jpeg"}, "html"),
        ("widget", {"save_path": "out.html"}, "video"),
        ("widget", {"scale_bar": True}, "napari"),
        ("video", {"embed": False}, "html"),
        ("video", {"title": "Session 1"}, "napari"),
    ],
)
def test_another_backends_option_is_refused_and_the_fix_runs(
    animation, tmp_path, backend, option, owner
):
    env, fields, times = animation
    (name,) = option
    save_path = {"video": "out.mp4", "html": "out.html"}.get(backend)
    if name == "save_path":
        save_path = option.pop(name)
    with pytest.raises(ValueError, match=f"does not use the {name}= option") as error:
        env.animate_fields(
            fields,
            frame_times=times,
            backend=backend,
            save_path=save_path,
            **option,
        )
    message = str(error.value)
    assert f"pass backend={owner!r}" in message
    assert list(tmp_path.iterdir()) == []
    # The suggested backend uses the option.
    _available(owner)
    owner_path = {"video": "out.mp4", "html": "out.html"}.get(owner)
    if name == "save_path":
        option = {}
        owner_path = "out.mp4"
    result = env.animate_fields(
        fields,
        frame_times=times,
        backend=owner,
        save_path=owner_path,
        dry_run=True,
        **option,
    )
    assert result is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_defaults_of_other_backends_options_are_accepted(animation, backend):
    """Environment.animate_fields forwards its defaults to every backend."""
    _available(backend)
    env, fields, times = animation
    save_path = {"video": "out.mp4", "html": "out.html"}.get(backend)
    result = env.animate_fields(
        fields,
        frame_times=times,
        backend=backend,
        save_path=save_path,
        codec="h264",
        dpi=100,
        image_format="png",
        max_html_frames=500,
        dry_run=True,
    )
    assert result is None


@pytest.mark.parametrize("backend", ["video", "widget"])
def test_frame_times_reach_backends_that_align_time_series(
    animation, monkeypatch, backend
):
    """Time-series overlays are drawn against the frame timestamps."""
    received = {}

    def renderer(env, fields, *args, **kwargs):
        received.update(kwargs)

    monkeypatch.setattr(
        f"neurospatial.animation.backends.{backend}_backend.render_{backend}",
        renderer,
    )
    monkeypatch.setattr(
        "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
        lambda: True,
    )
    env, fields, times = animation
    save_path = "out.mp4" if backend == "video" else None
    env.animate_fields(fields, frame_times=times, backend=backend, save_path=save_path)
    np.testing.assert_array_equal(received["frame_times"], times)


def test_html_draws_the_scale_bar(animation, tmp_path):
    env, fields, times = animation
    pages = {}
    for scale_bar in (False, True):
        path = tmp_path / f"scale_bar_{scale_bar}.html"
        env.animate_fields(
            fields[:1],
            frame_times=times[:1],
            backend="html",
            save_path=str(path),
            dpi=30,
            scale_bar=scale_bar,
        )
        pages[scale_bar] = path.read_text()
    assert pages[True] != pages[False]
