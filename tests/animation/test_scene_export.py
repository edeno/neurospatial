"""Scene exports: containers, encoder options, HTML players and worker rendering."""

import base64
import io
import json
import pickle
import re
import sys
from pathlib import Path
from unittest.mock import patch
from urllib.parse import unquote

import numpy as np
import pytest
from PIL import Image

from neurospatial import Environment
from neurospatial.animation import Animation, AnimationField
from neurospatial.animation._export import prepare_export
from neurospatial.animation.overlays import PositionOverlay, VideoOverlay

TIMES = np.arange(4) / 10
# Exporting the whole recording samples [0, 0.3] at 10 fps, last row included.
N_FRAMES = 4


@pytest.fixture
def scene():
    rng = np.random.default_rng(3)
    env = Environment.from_samples(rng.uniform(0, 10, (100, 2)), bin_size=2)
    with Animation(
        env,
        {"rate": AnimationField(rng.random((4, env.n_bins)), TIMES, vmin=0, vmax=1)},
        overlays=[PositionOverlay(rng.uniform(0, 10, (4, 2)), TIMES, trail_length=2)],
        title="Session 3",
    ) as animation:
        yield animation


def _embedded(path: Path) -> list[np.ndarray]:
    text = path.read_text(encoding="utf-8")
    frames = json.loads(re.search(r"const frames = (\[.*?\]);", text).group(1))
    return [np.asarray(Image.open(io.BytesIO(base64.b64decode(f)))) for f in frames]


def _labels(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return json.loads(re.search(r"const labels = (\[.*?\]);", text).group(1))


@pytest.mark.parametrize(
    ("suffix", "codec"),
    [(".mp4", "h264"), (".webm", "vp9"), (".mov", "h264"), (".avi", "h264")],
)
def test_video_containers_hold_every_frame_at_rendered_size(
    scene, tmp_path, suffix, codec
):
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    path = scene.export(tmp_path / f"movie{suffix}", fps=10, dpi=30)
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    frames = list(reader)
    expected = scene.render_frame(0.0, dpi=30)
    assert meta["codec"] == codec
    assert meta["pix_fmt"].startswith("yuv420p")
    assert meta["size"] == (expected.shape[1], expected.shape[0])
    assert len(frames) == N_FRAMES
    assert [item.name for item in tmp_path.iterdir()] == [path.name]


def test_gif_holds_every_frame(scene, tmp_path):
    import imageio.v3 as iio

    path = scene.export(tmp_path / "movie.gif", fps=10, dpi=30)
    frames = iio.imread(path, index=None)
    assert frames.shape[:3] == (N_FRAMES, *scene.render_frame(0.0, dpi=30).shape[:2])


@pytest.mark.parametrize(
    ("suffix", "options", "expected"),
    [
        (".mp4", {}, ("-preset", "medium", "-crf", "18", "-movflags", "+faststart")),
        (
            ".mp4",
            {"codec": "h265", "crf": 23, "preset": "slow"},
            ("-preset", "slow", "-crf", "23", "-x265-params", "log-level=error"),
        ),
        (".mov", {"bitrate": 2500}, ("-preset", "medium", "-b:v", "2500k")),
        (".webm", {"crf": 31}, ("-crf", "31", "-b:v", "0")),
        (".avi", {"codec": "mpeg4", "bitrate": 800}, ("-b:v", "800k")),
    ],
)
def test_encoder_options_reach_ffmpeg(tmp_path, suffix, options, expected):
    plan = prepare_export(tmp_path / f"a{suffix}", TIMES, **options)
    assert plan.output_params[: len(expected)] == expected
    assert plan.codec == {
        "h265": "libx265",
        "mpeg4": "mpeg4",
    }.get(options.get("codec"), "libvpx-vp9" if suffix == ".webm" else "libx264")


def test_lower_crf_keeps_more_detail(scene, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    fine = scene.export(tmp_path / "fine.mp4", fps=10, dpi=30, crf=0)
    coarse = scene.export(tmp_path / "coarse.mp4", fps=10, dpi=30, crf=45)
    assert fine.stat().st_size > 2 * coarse.stat().st_size


def test_crf_with_bitrate_warns_at_the_call_site(tmp_path):
    with pytest.warns(UserWarning, match="crf is ignored") as record:
        plan = prepare_export(tmp_path / "a.mp4", TIMES, crf=18, bitrate=1000)
    assert record[0].filename == __file__
    assert "-crf" not in plan.output_params
    assert "1000k" in plan.output_params


@pytest.mark.parametrize(
    ("name", "options", "message"),
    [
        ("a.gif", {"crf": 20}, r"crf does not apply to \.gif"),
        ("a.gif", {"codec": "h264"}, r"codec does not apply to \.gif"),
        ("a.html", {"bitrate": 900}, r"bitrate does not apply to \.html"),
        ("a.mp4", {"embed": False}, r"embed does not apply to \.mp4"),
        ("a.webm", {"image_format": "jpeg"}, r"image_format does not apply to \.webm"),
        ("a.gif", {"frames_dir": "frames"}, r"frames_dir does not apply to \.gif"),
        ("a.mov", {"max_html_frames": 9}, r"max_html_frames does not apply to \.mov"),
        ("a.html", {"frames_dir": "frames"}, "embed=False"),
        ("a.html", {"embed": False, "max_html_frames": 9}, "embedded players"),
        ("a.avi", {"codec": "mpeg4", "crf": 20}, "mpeg4"),
        ("a.webm", {"preset": "fast"}, "h264 and h265"),
    ],
)
def test_options_that_do_not_apply_are_refused(scene, tmp_path, name, options, message):
    with pytest.raises(TypeError, match=message):
        scene.export(tmp_path / name, **options)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("name", "options", "message"),
    [
        ("a.webm", {"codec": "h264"}, "WebM"),
        ("a.mp4", {"crf": 52}, "crf must be"),
        ("a.mp4", {"preset": "quick"}, "preset must be"),
        ("a.mp4", {"bitrate": 0}, "bitrate must be"),
        ("a.mkv", {}, r"writes \.mp4, \.mov"),
        ("a.html", {"image_format": "tiff"}, "image_format"),
        ("a.html", {"max_html_frames": 0}, "max_html_frames"),
        ("a.gif", {"n_workers": 0}, "n_workers"),
        ("a.gif", {"n_workers": True}, "n_workers"),
    ],
)
def test_invalid_option_values_are_actionable(tmp_path, name, options, message):
    with pytest.raises(ValueError, match=message):
        prepare_export(tmp_path / name, TIMES, **options)


@pytest.mark.parametrize("image_format", ["png", "jpeg"])
def test_html_player_embeds_rendered_frames_with_physical_times(
    scene, tmp_path, image_format
):
    path = scene.export(
        tmp_path / "player.html",
        time_range=(0.1, 0.3),
        fps=10,
        dpi=30,
        image_format=image_format,
    )
    text = path.read_text(encoding="utf-8")
    assert f"data:image/{image_format};base64," in text
    assert "<title>Session 3</title>" in text
    assert _labels(path) == ["t = 0.100 s", "t = 0.200 s"]
    frames = _embedded(path)
    assert len(frames) == 2
    expected = scene.render_frame(0.2, dpi=30)
    if image_format == "png":
        np.testing.assert_array_equal(frames[1], expected)
    else:
        assert frames[1].shape == expected.shape
        assert np.abs(frames[1].astype(int) - expected).mean() < 5
    assert [item.name for item in tmp_path.iterdir()] == ["player.html"]


@pytest.mark.parametrize("frames_dir", [None, "nested/frame images"])
def test_html_player_with_external_frames(scene, tmp_path, frames_dir):
    path = tmp_path / "player.html"
    scene.export(
        path,
        fps=10,
        dpi=30,
        embed=False,
        frames_dir=None if frames_dir is None else tmp_path / frames_dir,
    )
    directory = tmp_path / (frames_dir or "player")
    text = path.read_text(encoding="utf-8")
    url = json.loads(re.search(r"const framesDir = (.*);", text).group(1))
    assert (path.parent / unquote(url)).resolve() == directory.resolve()
    names = sorted(item.name for item in directory.iterdir())
    assert names == [f"frame_{index:05d}.png" for index in range(N_FRAMES)]
    np.testing.assert_array_equal(
        np.asarray(Image.open(directory / names[2])), scene.render_frame(0.2, dpi=30)
    )
    assert _labels(path) == [f"t = {time:.3f} s" for time in TIMES]


def test_html_frame_limit_is_checked_before_rendering(scene, tmp_path):
    from neurospatial.animation import _scene_rendering

    with (
        patch.object(_scene_rendering, "SceneRenderer") as renderer,
        pytest.raises(ValueError, match=r"time_range=.*fps=.*embed=False"),
    ):
        scene.export(tmp_path / "long.html", fps=100, max_html_frames=10)
    renderer.assert_not_called()
    assert not list(tmp_path.iterdir())
    # External frames load on demand, so the page itself stays small.
    scene.export(tmp_path / "long.html", fps=100, dpi=20, embed=False)
    assert len(list((tmp_path / "long").iterdir())) == 31  # [0, 0.3] at 100 fps


def test_html_overwrite_replaces_page_and_frames_only(scene, tmp_path):
    path = tmp_path / "player.html"
    frames = tmp_path / "frames"
    path.write_text("keep")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        scene.export(path)
    assert path.read_text() == "keep"
    path.unlink()

    frames.mkdir()
    (frames / "notes.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        scene.export(path, embed=False, frames_dir=frames)
    assert not path.exists()
    assert sorted(item.name for item in frames.iterdir()) == ["notes.txt"]

    scene.export(path, fps=20, dpi=20, embed=False, frames_dir=frames, overwrite=True)
    assert len(list(frames.glob("frame_*.png"))) == 7  # [0, 0.3] at 20 fps
    scene.export(path, fps=10, dpi=20, embed=False, frames_dir=frames, overwrite=True)
    # Frames of the earlier, longer export do not linger beside the new page.
    assert sorted(item.name for item in frames.iterdir()) == [
        *(f"frame_{index:05d}.png" for index in range(N_FRAMES)),
        "notes.txt",
    ]
    assert len(_labels(path)) == N_FRAMES
    assert not list(tmp_path.glob(".animation-*"))


def test_empty_frames_directory_is_writable(scene, tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    scene.export(tmp_path / "a.html", fps=10, dpi=20, embed=False, frames_dir=frames)
    assert len(list(frames.iterdir())) == N_FRAMES


def test_workers_render_identical_frames_and_reopen_sources(
    tmp_path, sample_video, sample_calibration
):
    rng = np.random.default_rng(4)
    env = Environment.from_samples(rng.uniform(0, 10, (100, 2)), bin_size=2)
    times = np.arange(12) / 10
    values = np.memmap(
        tmp_path / "rates.dat", mode="w+", dtype=float, shape=(24, env.n_bins)
    )
    values[:] = rng.random(values.shape)
    values.flush()
    scene = Animation(
        env,
        # A strided view of the recording, as from subsampling a long session.
        {"rate": AnimationField(values[::2], times)},
        overlays=[
            PositionOverlay(rng.uniform(0, 10, (12, 2)), times, trail_length=3),
            # Each worker opens its own reader on the video file.
            VideoOverlay(sample_video, sample_calibration, times=np.arange(10) / 10),
        ],
        window_seconds=0.5,
    )
    import matplotlib

    # Workers draw with the caller's Matplotlib style, not the defaults.
    with scene, matplotlib.rc_context({"figure.facecolor": "black"}):
        serial = scene.export(tmp_path / "serial.html", fps=10, dpi=30)
        parallel = scene.export(tmp_path / "parallel.html", fps=10, dpi=30, n_workers=3)
    expected, frames = _embedded(serial), _embedded(parallel)
    assert len(frames) == len(expected) == 12
    for frame, reference in zip(frames, expected, strict=True):
        np.testing.assert_array_equal(frame, reference)


def test_memmap_is_sent_to_workers_by_reference(tmp_path):
    from neurospatial.animation._export_workers import _dumps

    values = np.memmap(tmp_path / "big.dat", mode="w+", dtype=float, shape=(500, 64))
    values[:] = np.arange(values.size).reshape(values.shape)
    view = values[100:400:3, 5:40]
    data = _dumps(view)
    assert len(data) < 1000
    np.testing.assert_array_equal(pickle.loads(data), view)
    # Copy-on-write maps may differ from their file, so they are sent by value.
    private = np.memmap(tmp_path / "big.dat", mode="c", dtype=float, shape=(500, 64))
    private[0] = -1
    np.testing.assert_array_equal(pickle.loads(_dumps(private))[0], -1)


def test_unpicklable_field_names_itself_before_rendering(scene, tmp_path):
    class Rows:
        shape = (4, scene.env.n_bins)

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return np.ones(scene.env.n_bins)

    with (
        Animation(scene.env, {"lazy": AnimationField(Rows(), TIMES)}) as lazy,
        pytest.raises(ValueError, match=r"field 'lazy' cannot be pickled.*n_workers=1"),
    ):
        lazy.export(tmp_path / "a.gif", n_workers=2)
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
@pytest.mark.parametrize("name", ["movie.gif", "player.html"])
def test_published_files_get_the_usual_permissions(scene, tmp_path, name):
    import os

    umask = os.umask(0o022)
    try:
        path = scene.export(tmp_path / name, fps=10, dpi=20)
    finally:
        os.umask(umask)
    # Readable by others, e.g. a web server, like any file the user writes.
    assert path.stat().st_mode & 0o777 == 0o644
