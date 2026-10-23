"""animate_fields HTML players: playback clock, controls and a safe page."""

import json
import re

import numpy as np
import pytest

from neurospatial import Environment

# 11 rows sampled at 20 Hz span [0, 0.5] s.
FRAME_TIMES = np.arange(11) / 20


@pytest.fixture
def env():
    positions = np.random.default_rng(42).standard_normal((100, 2)) * 50
    return Environment.from_samples(positions, bin_size=10.0)


@pytest.fixture
def fields(env):
    return np.random.default_rng(0).random((len(FRAME_TIMES), env.n_bins))


def _labels(text: str) -> list[str]:
    return json.loads(re.search(r"const labels = (\[.*?\]);", text).group(1))


def test_player_plays_rows_at_their_rate_times_speed(env, fields, tmp_path):
    path = env.animate_fields(
        fields,
        frame_times=FRAME_TIMES,
        save_path=tmp_path / "replay.html",
        speed=0.5,
        dpi=20,
    )
    text = path.read_text(encoding="utf-8")
    assert re.search(r"const baseFPS = (\S+);", text).group(1) == "10.0"
    # One labelled frame per row in the half-open [0, 0.5) s.
    assert _labels(text) == [f"t = {k / 20:.3f} s" for k in range(10)]
    # Play, pause, stepping, a slider, a speed menu and arrow-key stepping.
    for control in ('id="play"', 'id="pause"', 'id="prev"', 'id="next"'):
        assert control in text
    assert 'type="range" id="slider"' in text
    assert 'id="speed"' in text
    assert "ArrowLeft" in text and "ArrowRight" in text


def test_title_is_escaped(env, fields, tmp_path):
    path = env.animate_fields(
        fields,
        frame_times=FRAME_TIMES,
        save_path=tmp_path / "title.html",
        title="Test <script>alert('xss')</script> & Title",
        dpi=20,
    )
    text = path.read_text(encoding="utf-8")
    assert "&lt;script&gt;" in text
    assert "&amp; Title" in text
    assert "<script>alert" not in text
