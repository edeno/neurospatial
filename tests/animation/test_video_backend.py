"""animate_fields video files play on the recording's physical clock."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment

# 21 rows sampled at 20 Hz span [0, 1] s.
FRAME_TIMES = np.arange(21) / 20


@pytest.fixture
def env():
    positions = np.random.default_rng(42).standard_normal((100, 2)) * 50
    return Environment.from_samples(positions, bin_size=10.0)


@pytest.fixture
def fields(env):
    return np.random.default_rng(0).random((len(FRAME_TIMES), env.n_bins))


def _metadata(path):
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    return meta, sum(1 for _ in reader)


@pytest.mark.parametrize(("speed", "fps"), [(1.0, 20), (0.5, 10), (2.0, 40)])
def test_file_rate_is_the_sample_rate_times_speed(env, fields, tmp_path, speed, fps):
    pytest.importorskip("imageio_ffmpeg")
    path = env.animate_fields(
        fields,
        frame_times=FRAME_TIMES,
        save_path=tmp_path / "replay.mp4",
        speed=speed,
        dpi=30,
    )
    meta, n_frames = _metadata(path)
    assert meta["fps"] == pytest.approx(fps)
    # Every row is shown once, through the last: the export covers [0, 1] s.
    assert n_frames == len(FRAME_TIMES)
    assert [item.name for item in tmp_path.iterdir()] == ["replay.mp4"]


def test_capped_rate_keeps_speed_and_skips_rows(env, fields, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    with pytest.warns(UserWarning, match="max_playback_fps=20.*skips rows"):
        path = env.animate_fields(
            fields,
            frame_times=FRAME_TIMES,
            save_path=tmp_path / "fast.mp4",
            speed=4.0,
            max_playback_fps=20,
            dpi=30,
        )
    meta, n_frames = _metadata(path)
    assert meta["fps"] == pytest.approx(20)
    # A frame every 0.2 s of recording (20 fps at 4x), not one per 0.05 s row:
    # 0, 0.2, ..., 1.0 s.
    assert n_frames == 6


def test_encoder_options_reach_the_file(env, fields, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    path = env.animate_fields(
        fields,
        frame_times=FRAME_TIMES,
        save_path=tmp_path / "replay.avi",
        codec="mpeg4",
        bitrate=800,
        dpi=30,
    )
    meta, _ = _metadata(path)
    assert meta["codec"] == "mpeg4"
