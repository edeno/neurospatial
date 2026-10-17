"""Every animation backend uses the options it is given, or refuses them.

A dry run writes nothing on any backend, a misspelled option raises with the
closest name, and an option only another backend uses is ignored with one
warning whose suggested call works.
"""

import base64
import inspect
import re
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation.core import _BACKEND_CALLS, _BACKEND_OPTIONS

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


def _ignore_warnings(record):
    return [w for w in record if "backend ignores it" in str(w.message)]


@pytest.mark.parametrize(
    ("backend", "option", "owner"),
    [
        ("html", {"crf": 15}, "video"),
        ("video", {"embed": False}, "html"),
        ("napari", {"image_format": "jpeg"}, "html"),
        ("widget", {"scale_bar": True}, "napari"),
        ("video", {"title": "Session 1"}, "napari"),
    ],
)
def test_another_backends_option_is_ignored_with_one_warning(
    animation, tmp_path, backend, option, owner
):
    _available(backend)
    env, fields, times = animation
    (name,) = option
    save_path = {"video": "out.mp4", "html": "out.html"}.get(backend)
    with pytest.warns(UserWarning) as record:
        result = env.animate_fields(
            fields,
            frame_times=times,
            backend=backend,
            save_path=save_path,
            dry_run=True,
            **option,
        )
    assert result is None
    (warning,) = _ignore_warnings(record)
    message = str(warning.message)
    assert f"{name} is used by the {owner}" in message
    assert f"the {backend} backend ignores it" in message
    assert "auto" not in message  # the caller named the backend
    assert list(tmp_path.iterdir()) == []
    # The suggested call uses the option, so it does not warn.
    _available(owner)
    call = _BACKEND_CALLS[owner]
    assert f"pass {call}" in message
    suggested = eval(f"dict({call})")
    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter("always")
        env.animate_fields(
            fields, frame_times=times, dry_run=True, **suggested, **option
        )
    assert _ignore_warnings(again) == []


def test_several_ignored_options_share_one_warning(animation):
    env, fields, times = animation
    with pytest.warns(UserWarning) as record:
        env.animate_fields(
            fields,
            frame_times=times,
            backend="html",
            crf=15,
            preset="slow",
            dry_run=True,
        )
    (warning,) = _ignore_warnings(record)
    assert "crf is used by the video backend" in str(warning.message)
    assert "preset is used by the video backend" in str(warning.message)


@pytest.mark.parametrize(
    ("n_frames", "reason"),
    [(5, "there are 5 frames (10,000 or fewer)"), (10_001, "more than 10,000")],
)
def test_auto_says_which_backend_it_chose_and_why(
    animation, monkeypatch, n_frames, reason
):
    """Outside Jupyter and without a file, auto picks napari for both counts."""
    received = []
    monkeypatch.setattr(
        "neurospatial.animation.backends.napari_backend.NAPARI_AVAILABLE", True
    )
    monkeypatch.setattr(
        "neurospatial.animation.backends.napari_backend.render_napari",
        lambda env, fields, **kwargs: received.append(kwargs),
    )
    env, _, _ = animation
    fields = np.zeros((n_frames, env.n_bins))
    times = np.arange(n_frames) / 30.0
    with pytest.warns(UserWarning, match="embed is used by the html") as record:
        env.animate_fields(fields, frame_times=times, embed=False)
    message = str(_ignore_warnings(record)[0].message)
    assert "backend='auto' chose 'napari' because" in message
    assert reason in message
    assert "pass backend='html', save_path='animation.html'" in message
    assert "embed" not in received[0]


def test_options_left_out_raise_no_warning(animation):
    env, fields, times = animation
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        env.animate_fields(fields, frame_times=times, backend="html", dry_run=True)
    assert _ignore_warnings(record) == []


def test_video_crf_reaches_the_encoder_without_a_bitrate(animation, monkeypatch):
    """Constant-quality encoding is not replaced by a default bitrate."""
    received = {}
    monkeypatch.setattr(
        "neurospatial.animation.backends.video_backend.render_video",
        lambda env, fields, save_path, **kwargs: received.update(kwargs),
    )
    monkeypatch.setattr(
        "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
        lambda: True,
    )
    env, fields, times = animation
    env.animate_fields(
        fields, frame_times=times, backend="video", save_path="out.mp4", crf=15
    )
    assert received["crf"] == 15
    assert "bitrate" not in received


def test_video_warns_that_bitrate_overrides_crf(animation, monkeypatch):
    """With both, the encoder uses the bitrate; the caller is told crf is lost."""
    received = {}
    monkeypatch.setattr(
        "neurospatial.animation.backends.video_backend.render_video",
        lambda env, fields, save_path, **kwargs: received.update(kwargs),
    )
    monkeypatch.setattr(
        "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
        lambda: True,
    )
    env, fields, times = animation
    with pytest.warns(UserWarning, match="crf is ignored") as record:
        env.animate_fields(
            fields,
            frame_times=times,
            backend="video",
            save_path="out.mp4",
            crf=15,
            bitrate=800,
        )
    assert len([w for w in record if "crf is ignored" in str(w.message)]) == 1
    assert received["bitrate"] == 800
    with warnings.catch_warnings(record=True) as alone:
        warnings.simplefilter("always")
        env.animate_fields(
            fields, frame_times=times, backend="video", save_path="out.mp4", crf=15
        )
    assert not [w for w in alone if "crf is ignored" in str(w.message)]


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


def _png_width(png: bytes) -> int:
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    return int.from_bytes(png[16:20], "big")


def test_html_frames_are_the_scale_bar_renders(animation, tmp_path):
    """The embedded frame is the shared renderer's frame with a scale bar."""
    from neurospatial.animation.rendering import (
        compute_global_colormap_range,
        render_field_to_image_bytes,
    )

    env, fields, times = animation
    vmin, vmax = compute_global_colormap_range([fields[0]], None, None)
    path = tmp_path / "scale_bar.html"
    env.animate_fields(
        fields[:1],
        frame_times=times[:1],
        backend="html",
        save_path=str(path),
        dpi=30,
        scale_bar=True,
    )
    (frame,) = re.findall(r"iVBORw0KGgo[A-Za-z0-9+/=]+", path.read_text())
    embedded = base64.b64decode(frame)
    with_bar, without_bar = (
        render_field_to_image_bytes(
            env, fields[0], "viridis", vmin, vmax, 30, scale_bar=scale_bar
        )
        for scale_bar in (True, False)
    )
    assert embedded == with_bar
    assert embedded != without_bar


def _time_series_overlay():
    from neurospatial.animation.overlays import TimeSeriesOverlay

    samples = np.linspace(0.0, 0.2, 50)
    return TimeSeriesOverlay(data=np.sin(samples * 30.0), times=samples)


def _video_width(path: Path) -> int:
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is required to read the video size")
    output = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(output.stdout.strip())


@pytest.mark.parametrize("backend", ["video", "widget"])
def test_time_series_overlay_adds_a_panel_column(
    animation, tmp_path, monkeypatch, backend
):
    """Frames with a time-series overlay are wider: the panel is drawn."""
    _available(backend)
    env, fields, times = animation
    shown = []
    monkeypatch.setattr(
        "neurospatial.animation.backends.widget_backend.display", shown.append
    )
    widths = {}
    for with_overlay in (False, True):
        overlays = [_time_series_overlay()] if with_overlay else None
        path = tmp_path / f"panel_{with_overlay}.mp4"
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            env.animate_fields(
                fields,
                frame_times=times,
                backend=backend,
                save_path=str(path) if backend == "video" else None,
                overlays=overlays,
                dpi=50,
                **({"n_workers": 1} if backend == "video" else {}),
            )
        assert not any("tight_layout" in str(w.message) for w in record)
        if backend == "video":
            widths[with_overlay] = _video_width(path)
        else:
            image = shown[-1].children[2]
            widths[with_overlay] = _png_width(image.value)
    assert widths[True] > widths[False]


def test_napari_dry_run_checks_the_inputs_a_real_run_checks(animation):
    """A multi-field call without a layout fails the same way in a dry run."""
    _available("napari")
    env, fields, times = animation
    sequences = [list(fields), list(fields)]
    with pytest.raises(ValueError, match="requires 'layout'"):
        env.animate_fields(sequences, frame_times=times, backend="napari", dry_run=True)
    result = env.animate_fields(
        sequences,
        frame_times=times,
        backend="napari",
        layout="horizontal",
        dry_run=True,
    )
    assert result is None
