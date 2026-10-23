"""Exported animation frames draw overlays and regions where they belong.

``animate_fields`` exports through the scene renderer. These tests decode the
frames of a lossless HTML export and check where overlays land, not only that
an export completes.
"""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import box

from neurospatial import Environment, Regions
from neurospatial.animation.overlays import PositionOverlay

TIMES = np.arange(5) / 10


@pytest.fixture
def arena():
    """A 100 x 100 cm open field with a goal on the left and a start on the right."""
    edges = np.linspace(0.0, 100.0, 11)
    env = Environment.from_grid_mask(np.ones((10, 10), dtype=bool), (edges, edges))
    regions = Regions()
    regions.add("goal", polygon=box(5.0, 40.0, 30.0, 60.0))
    regions.add("start", polygon=box(70.0, 40.0, 95.0, 60.0))
    return env.with_regions(regions)


def _frames(path: Path) -> list[np.ndarray]:
    text = path.read_text(encoding="utf-8")
    frames = json.loads(re.search(r"const frames = (\[.*?\]);", text).group(1))
    return [
        np.asarray(Image.open(io.BytesIO(base64.b64decode(item))), dtype=int)
        for item in frames
    ]


def _export(env, path, **kwargs):
    """A constant zero field, so overlays are the only thing that changes."""
    return env.animate_fields(
        np.zeros((len(TIMES), env.n_bins)),
        frame_times=TIMES,
        save_path=path,
        vmin=0.0,
        vmax=1.0,
        dpi=40,
        **kwargs,
    )


def _red(frame: np.ndarray) -> np.ndarray:
    return (frame[..., 0] > 200) & (frame[..., 1] < 80) & (frame[..., 2] < 80)


def test_position_marker_lands_at_the_animals_position_and_hides_when_missing(
    arena, tmp_path
):
    positions = np.array(
        [[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [np.nan, np.nan], [50.0, 50.0]]
    )
    path = _export(
        arena,
        tmp_path / "positions.html",
        overlays=[PositionOverlay(positions, TIMES, color="red", size=12)],
    )
    # Export holds the half-open [0, 0.4): one frame per row but the last.
    frames = _frames(path)
    assert len(frames) == 4
    rows, cols = [], []
    for frame in frames[:3]:
        r, c = np.nonzero(_red(frame))
        assert r.size > 0
        rows.append(r.mean())
        cols.append(c.mean())
    # Moving +x moves the marker right on screen; moving +y moves it up.
    assert cols[1] - cols[0] > 10
    assert abs(rows[1] - rows[0]) < 2
    assert rows[1] - rows[2] > 10
    assert abs(cols[2] - cols[1]) < 2
    # Equal displacements in x and y cover equal distances on an equal-aspect axis.
    assert cols[1] - cols[0] == pytest.approx(rows[1] - rows[2], rel=0.05)
    # A missing tracking sample draws no marker rather than a stale or zero one.
    assert not _red(frames[3]).any()


def test_regions_are_drawn_only_when_requested_and_where_they_are(arena, tmp_path):
    def frame(show_regions):
        name = f"{show_regions}.html".replace("'", "")
        path = _export(arena, tmp_path / name, show_regions=show_regions)
        return _frames(path)[0]

    plain = frame(False)
    goal = np.any(frame(["goal"]) != plain, axis=-1)
    both = np.any(frame(True) != plain, axis=-1)
    middle = plain.shape[1] / 2
    goal_cols = np.nonzero(goal)[1]
    assert goal_cols.size > 0
    # Only the named region is drawn, at the left where it lies.
    assert goal_cols.max() < middle
    right = np.nonzero(both)[1] > middle
    assert right.any()
    assert np.nonzero(goal)[1].size < np.nonzero(both)[1].size


def test_unpicklable_overlay_is_named_before_parallel_export(arena, tmp_path):
    class LocalPositions(PositionOverlay):
        """Defined in a function, so worker processes cannot unpickle it."""

    overlay = LocalPositions(np.full((len(TIMES), 2), 50.0), TIMES)
    with pytest.raises(
        ValueError, match=r"overlay 0 \(LocalPositions\) cannot be pickled.*n_workers=1"
    ):
        _export(arena, tmp_path / "parallel.html", overlays=[overlay], n_workers=2)
    assert not list(tmp_path.iterdir())
    # Rendering in this process needs no pickling.
    serial = _export(arena, tmp_path / "serial.html", overlays=[overlay], n_workers=1)
    assert len(_frames(serial)) == 4
