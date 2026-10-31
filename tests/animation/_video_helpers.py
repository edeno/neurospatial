"""Shared helper for tests that read exported video files."""

from __future__ import annotations

import pytest


def read_video(path) -> tuple[dict, list[bytes]]:
    """Read a video file's ffmpeg metadata and every frame's raw bytes.

    ``imageio_ffmpeg.read_frames`` leaves the ffmpeg process's stdin and stdout
    pipes open when ffmpeg exits before the generator finishes, which a small
    file always does (a ``ResourceWarning`` at garbage collection). The pipes are
    closed here, through the process the generator holds, so reading a file
    leaves nothing open.
    """
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    process = reader.gi_frame.f_locals["process"]
    try:
        frames = list(reader)
    finally:
        reader.close()
        process.stdin.close()
        process.stdout.close()
    return meta, frames
