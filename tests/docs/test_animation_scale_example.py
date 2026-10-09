"""The published HTML scale recovery renders an actual labeled player."""

import base64
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

from .test_executable_docs import ROOT, collect_blocks


def test_static_scale_and_html_timestamps_render(monkeypatch, tmp_path):
    path = ROOT / "docs/user-guide/animation.md"
    block = next(
        block
        for block in collect_blocks(path.read_text(encoding="utf-8"), str(path))
        if "scaled_rates.html" in block.code
    )
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__main__"}
    try:
        exec(compile(block.code, str(path), "exec"), namespace)
        ax = namespace["ax"]
        assert ax.collections[0].norm.vmin == 0.0
        assert ax.collections[0].norm.vmax == 10.0
        assert ax.figure.axes[1].get_ylabel() == "Firing rate (Hz)"
        html = Path("scaled_rates.html").read_text(encoding="utf-8")
        match = re.search(r"const frames = (\[.*?\]);", html, re.S)
        assert match is not None
        frames = json.loads(match[1])
        assert len(frames) == 5
        assert all(
            base64.b64decode(frame).startswith(b"\x89PNG\r\n\x1a\n") for frame in frames
        )
        assert "10.05 s" in html
        assert "10.45 s" in html
    finally:
        plt.close("all")
