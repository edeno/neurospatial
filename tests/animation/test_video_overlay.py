"""Exported animations composite a calibrated video on the recording clock."""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from neurospatial import Environment
from neurospatial.animation.overlays import VideoOverlay

if TYPE_CHECKING:
    from neurospatial.ops.transforms import VideoCalibration

N_ROWS = 10
FRAME_TIMES = np.arange(N_ROWS) / 10


@pytest.fixture
def video_test_env() -> Environment:
    """A 16 x 16 cm environment matching the 16 x 16 pixel test video."""
    edges = np.linspace(0.0, 16.0, 5)
    return Environment.from_grid_mask(np.ones((4, 4), dtype=bool), (edges, edges))


def _embedded(path: Path) -> list[np.ndarray]:
    text = path.read_text(encoding="utf-8")
    frames = json.loads(re.search(r"const frames = (\[.*?\]);", text).group(1))
    return [
        np.asarray(Image.open(io.BytesIO(base64.b64decode(item))), dtype=int)
        for item in frames
    ]


def test_video_is_composited_into_every_frame_and_follows_the_clock(
    tmp_path: Path,
    video_test_env: Environment,
    sample_video_array: np.ndarray,
    sample_calibration: VideoCalibration,
):
    """Video frame i has brightness 25 * i, so later frames are brighter."""
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    overlay = VideoOverlay(
        source=sample_video_array,
        calibration=sample_calibration,
        alpha=0.5,
        z_order="above",
    )
    path = video_test_env.animate_fields(
        np.zeros((N_ROWS, video_test_env.n_bins)),
        overlays=[overlay],
        save_path=tmp_path / "with_video.mp4",
        frame_times=FRAME_TIMES,
        vmin=0.0,
        vmax=1.0,
        dpi=50,
    )
    reader = imageio_ffmpeg.read_frames(str(path))
    next(reader)
    frames = [np.frombuffer(frame, dtype=np.uint8).astype(float) for frame in reader]
    # One output frame per row in the half-open [first, last) time range.
    assert len(frames) == N_ROWS - 1
    brightness = [frame.mean() for frame in frames]
    assert np.all(np.diff(brightness) > 0)
    assert [item.name for item in tmp_path.iterdir()] == ["with_video.mp4"]


def test_video_z_order_puts_the_video_above_or_below_the_field(
    tmp_path: Path,
    video_test_env: Environment,
    sample_video_array: np.ndarray,
    sample_calibration: VideoCalibration,
):
    fields = np.random.default_rng(42).random((N_ROWS, video_test_env.n_bins))

    def first_frame(z_order: str) -> np.ndarray:
        overlay = VideoOverlay(
            source=sample_video_array,
            calibration=sample_calibration,
            alpha=1.0,
            z_order=z_order,
        )
        path = video_test_env.animate_fields(
            fields,
            overlays=[overlay],
            save_path=tmp_path / f"{z_order}.html",
            frame_times=FRAME_TIMES,
            vmin=0.0,
            vmax=1.0,
            dpi=40,
        )
        return _embedded(path)[5]

    above, below = first_frame("above"), first_frame("below")
    # An opaque video above the field hides the field: its pixels are the gray
    # of video frame 5 (brightness 125).
    gray = (np.abs(above - 125) <= 2).all(axis=-1)
    assert gray.sum() > 100
    # Below the field, the field's colors show where the video lay above it.
    still_gray = (np.abs(below[gray] - 125) <= 2).all(axis=-1).mean()
    assert still_gray < 0.5
