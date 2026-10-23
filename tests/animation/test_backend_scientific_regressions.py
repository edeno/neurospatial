"""Scientific invariants of scene panels and exported players."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation.overlays import TimeSeriesOverlay
from neurospatial.animation.scene import Animation


@pytest.fixture
def physical_grid():
    """An offset grid with unequal, non-unit physical pixel sizes."""
    return Environment.from_grid_mask(
        np.ones((3, 4), dtype=bool),
        grid_edges=(np.arange(4) * 2.0 + 100, np.arange(5) * 5.0 - 30),
    )


def test_time_series_with_one_label_keep_independent_traces_and_union_limits(
    physical_grid,
):
    """Two signals named "Speed" in one group are two traces on one scale."""
    times = np.arange(3.0)
    values = {
        "a": [np.array([0.0, 1, 2]), np.array([100.0, 101, 102])],
        "b": [np.array([-2.0, -1, 0])],
    }
    overlays = [
        TimeSeriesOverlay(data=trace, times=times, label="Speed", group=group)
        for group, traces in values.items()
        for trace in traces
    ]
    with Animation(
        physical_grid,
        np.zeros((3, physical_grid.n_bins)),
        frame_times=times,
        overlays=overlays,
    ) as scene:
        scene.render_frame(1.0, dpi=20)
        panels = [
            ax
            for ax in scene._renderer.figure.axes
            if ax.get_xlabel() == "Time (s)" and ax.lines
        ]
        assert len(panels) == 2
        for ax, traces in zip(panels, values.values(), strict=True):
            drawn = [ln.get_ydata() for ln in ax.lines if ln.get_label() == "Speed"]
            assert len(drawn) == len(traces)
            for line, trace in zip(drawn, traces, strict=True):
                np.testing.assert_array_equal(line, trace)
            low, high = ax.get_ylim()
            assert low <= min(t.min() for t in traces)
            assert high >= max(t.max() for t in traces)
        # Group "b" is not stretched to group "a"'s scale.
        assert panels[1].get_ylim()[1] < 1


@pytest.mark.parametrize("directory", ["nested/frame images", "../outside #files'<>/"])
def test_external_html_urls_resolve_relative_to_the_page(
    physical_grid, tmp_path, directory
):
    output = tmp_path / "report" / "player.html"
    output.parent.mkdir()
    frames = output.parent / directory
    physical_grid.animate_fields(
        np.zeros((3, physical_grid.n_bins)),
        frame_times=np.arange(3) / 10,
        save_path=output,
        embed=False,
        frames_dir=frames,
        dpi=20,
    )
    url = json.loads(
        re.search(r"const framesDir = (.*);", output.read_text(encoding="utf-8")).group(
            1
        )
    )
    assert " " not in url and "#" not in url and "<" not in url and "'" not in url
    assert (output.parent / Path(unquote(url))).resolve() == frames.resolve()
    assert sorted(item.name for item in frames.iterdir()) == [
        "frame_00000.png",
        "frame_00001.png",
    ]
