"""Every animation backend uses the options it is given, or refuses them.

A misspelled option raises with the closest name, and an option only another
backend uses is ignored with one warning whose suggested call works.
"""

import inspect
import io
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from neurospatial import Environment
from neurospatial.animation.core import (
    _BACKEND_CALLS,
    _BACKEND_OPTIONS,
    _SCENE_OPTIONS,
)
from neurospatial.animation.scene import Animation

# Backends whose calls need no display: two exports and the notebook widget.
BACKENDS = ["video", "html", "widget"]


def _available(backend):
    if backend == "video":
        pytest.importorskip("imageio_ffmpeg")
    if backend == "widget":
        pytest.importorskip("ipywidgets")


@pytest.fixture
def animation(tmp_path, monkeypatch):
    """A 3-frame animation on a small arena, run from an empty directory."""
    monkeypatch.chdir(tmp_path)
    x, y = np.meshgrid(np.arange(0.0, 20.0, 4.0), np.arange(0.0, 20.0, 4.0))
    env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=4.0)
    fields = np.random.default_rng(0).random((3, env.n_bins))
    return env, fields, np.arange(3) / 30.0


def _run(animation, backend, **options):
    """Call animate_fields, close any view, and return the written files."""
    env, fields, times = animation
    save_path = {"video": "out.mp4", "html": "out.html"}.get(backend)
    result = env.animate_fields(
        fields, frame_times=times, backend=backend, save_path=save_path, **options
    )
    if backend == "widget":
        result.close()
    return result


@pytest.mark.parametrize("backend", ["video", "html"])
def test_export_options_are_export_parameters(backend):
    """Each option the table routes to an export reaches Animation.export."""
    parameters = set(inspect.signature(Animation.export).parameters)
    routed = _BACKEND_OPTIONS[backend] - {"save_path"}
    assert routed <= parameters


def test_scene_options_are_scene_or_field_settings():
    parameters = set(inspect.signature(Animation).parameters)
    field_settings = {"cmap", "vmin", "vmax", "colorbar_label", "layer_names"}
    assert _SCENE_OPTIONS - field_settings <= parameters


@pytest.mark.parametrize("backend", BACKENDS)
def test_misspelled_option_names_the_closest_one_and_the_fix_runs(
    animation, tmp_path, backend
):
    _available(backend)
    with pytest.raises(TypeError, match="unknown option 'colorbar_lable'") as error:
        _run(animation, backend, colorbar_lable="Rate (Hz)")
    assert "pass colorbar_label= instead of colorbar_lable=" in str(error.value)
    assert list(tmp_path.iterdir()) == []
    # The suggested name is accepted.
    _run(animation, backend, colorbar_label="Rate (Hz)", dpi=20)


def _ignore_warnings(record):
    return [w for w in record if "backend ignores it" in str(w.message)]


@pytest.mark.parametrize(
    ("backend", "option", "owner"),
    [
        ("html", {"crf": 15}, "video"),
        ("video", {"embed": False}, "html"),
        ("widget", {"image_format": "jpeg"}, "html"),
        ("widget", {"ndisplay": 2}, "napari"),
        ("video", {"field_rendering": "points"}, "fastplotlib"),
    ],
)
def test_another_backends_option_is_ignored_with_one_warning(
    animation, backend, option, owner
):
    _available(backend)
    (name,) = option
    with pytest.warns(UserWarning) as record:
        _run(animation, backend, dpi=20, **option)
    (warning,) = _ignore_warnings(record)
    message = str(warning.message)
    assert f"{name} is used by the {owner}" in message
    assert f"the {backend} backend ignores it" in message
    assert "auto" not in message  # the caller named the backend
    assert f"pass {_BACKEND_CALLS[owner]}" in message


@pytest.mark.parametrize(
    ("option", "owner"), [({"crf": 15}, "video"), ({"embed": False}, "html")]
)
def test_the_suggested_call_uses_the_option(animation, tmp_path, option, owner):
    """Following the warning's advice writes the file with the option."""
    _available(owner)
    suggested = eval(f"dict({_BACKEND_CALLS[owner]})")
    env, fields, times = animation
    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter("always")
        path = env.animate_fields(
            fields, frame_times=times, dpi=20, **suggested, **option
        )
    assert _ignore_warnings(again) == []
    assert path.exists()
    if owner == "html":
        # embed=False writes the frames beside the page, one per row.
        assert sorted(item.name for item in (tmp_path / "animation").iterdir()) == [
            f"frame_{k:05d}.png" for k in range(len(times))
        ]


def test_several_ignored_options_share_one_warning(animation):
    with pytest.warns(UserWarning) as record:
        _run(animation, "html", crf=15, preset="slow", dpi=20)
    (warning,) = _ignore_warnings(record)
    assert "crf is used by the video backend" in str(warning.message)
    assert "preset is used by the video backend" in str(warning.message)


def test_auto_says_which_backend_it_chose_and_why(animation):
    """In Jupyter auto picks the widget and says so in the warning."""
    pytest.importorskip("ipywidgets")
    env, fields, times = animation
    with (
        patch("IPython.get_ipython", return_value=MagicMock()),
        pytest.warns(UserWarning, match="embed is used by the html") as record,
    ):
        view = env.animate_fields(fields, frame_times=times, embed=False, dpi=20)
    view.close()
    message = str(_ignore_warnings(record)[0].message)
    assert "backend='auto' chose 'widget' because this is a Jupyter session" in message
    assert "pass backend='html', save_path='animation.html'" in message


def test_options_left_out_raise_no_warning(animation):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _run(animation, "html", dpi=20)
    assert _ignore_warnings(record) == []


def test_video_crf_with_bitrate_warns_once_at_the_call_site(animation):
    """With both, the encoder uses the bitrate; the caller is told crf is lost."""
    _available("video")
    with pytest.warns(UserWarning, match="crf is ignored") as record:
        _run(animation, "video", crf=15, bitrate=800, dpi=20)
    (warning,) = [w for w in record if "crf is ignored" in str(w.message)]
    assert warning.filename == __file__
    with warnings.catch_warnings(record=True) as alone:
        warnings.simplefilter("always")
        _run(animation, "video", crf=15, dpi=20, overwrite=True)
    assert not [w for w in alone if "crf is ignored" in str(w.message)]


def test_widget_dpi_and_figsize_set_the_displayed_image(animation):
    """The notebook view draws at the requested size and resolution."""
    _available("widget")
    env, fields, times = animation
    view = env.animate_fields(
        fields, frame_times=times, backend="widget", dpi=20, figsize=(4, 3)
    )
    try:
        image = Image.open(io.BytesIO(view.native.children[0].value))
        assert image.size == (80, 60)
    finally:
        view.close()


def test_time_series_overlay_adds_a_panel(animation):
    """Frames with a time-series overlay are taller: the panel is drawn."""
    _available("widget")
    from neurospatial.animation.overlays import TimeSeriesOverlay

    env, fields, times = animation
    samples = np.linspace(0.0, times[-1], 50)
    heights = {}
    for with_overlay in (False, True):
        overlays = (
            [TimeSeriesOverlay(data=np.sin(samples * 30.0), times=samples)]
            if with_overlay
            else None
        )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            view = env.animate_fields(
                fields, frame_times=times, backend="widget", overlays=overlays, dpi=20
            )
        try:
            assert not any("tight_layout" in str(w.message) for w in record)
            heights[with_overlay] = Image.open(
                io.BytesIO(view.native.children[0].value)
            ).size[1]
        finally:
            view.close()
    assert heights[True] > heights[False]
