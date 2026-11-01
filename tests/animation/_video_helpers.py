"""Shared helper for tests that read exported video files."""

from __future__ import annotations

import re
import subprocess

import pytest

# The first video stream line of `ffmpeg -i`, e.g.
# "Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 32x24, ..., 20 fps"
_VIDEO_STREAM = re.compile(
    r"Video: (?P<codec>\w+)[^,]*, (?P<pix_fmt>\w+(?:\([^)]*\))?)[^,]*, "
    r"(?P<width>\d+)x(?P<height>\d+)[^\n]*?(?P<fps>[\d.]+) fps"
)


def read_video(path) -> tuple[dict, list[bytes]]:
    """Read a video file's stream metadata and every frame as raw RGB bytes.

    Runs the ffmpeg that imageio-ffmpeg bundles. ``imageio_ffmpeg.read_frames``
    is not used: it leaves ffmpeg's pipes open when ffmpeg exits before the
    reader finishes, which a short file always does.
    """
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True
    )
    stream = _VIDEO_STREAM.search(probe.stderr)
    assert stream is not None, probe.stderr
    width, height = int(stream["width"]), int(stream["height"])
    meta = {
        "codec": stream["codec"],
        "pix_fmt": stream["pix_fmt"],
        "size": (width, height),
        "fps": float(stream["fps"]),
    }
    raw = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    frame_bytes = width * height * 3
    return meta, [raw[i : i + frame_bytes] for i in range(0, len(raw), frame_bytes)]
