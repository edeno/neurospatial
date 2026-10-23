"""The documented fixed-scale HTML export renders an actual timestamped player."""

import base64
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

from .test_executable_docs import ROOT, _fixture_namespace, collect_blocks


def test_fixed_scale_html_player_renders_timestamped_frames(monkeypatch, tmp_path):
    path = ROOT / "docs/user-guide/animation.md"
    block = next(
        block
        for block in collect_blocks(path.read_text(encoding="utf-8"), str(path))
        if "scaled_rates.html" in block.code
    )
    monkeypatch.chdir(tmp_path)
    try:
        exec(compile(block.code, str(path), "exec"), _fixture_namespace())
        html = Path("scaled_rates.html").read_text(encoding="utf-8")
        frames = json.loads(re.search(r"const frames = (\[.*?\]);", html, re.S)[1])
        labels = json.loads(re.search(r"const labels = (\[.*?\]);", html, re.S)[1])
        # Five rows 0.1 s apart cover the half-open [10.05, 10.45) s.
        assert labels == [f"t = {10.05 + 0.1 * k:.3f} s" for k in range(4)]
        assert len(frames) == 4
        assert all(
            base64.b64decode(frame).startswith(b"\x89PNG\r\n\x1a\n") for frame in frames
        )
    finally:
        plt.close("all")
