"""End-to-end animation of environments with different layouts.

Exports write one frame per output time, and napari places each bin's value
at that bin's position, whatever the grid's extent or orientation.
"""

from __future__ import annotations

import base64
import io
import json
import re

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.layout import create_layout
from neurospatial.ops.transforms import AffineND

pytestmark = pytest.mark.integration

# Ten rows at 9 Hz: the half-open export interval [0, 1) holds nine frames.
N_ROWS = 10
FRAME_TIMES = np.linspace(0.0, 1.0, N_ROWS)


def test_hexagonal_layout_exports_every_frame_at_the_sampling_rate(tmp_path):
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    env = Environment.from_layout(
        create_layout(
            "hexagonal",
            hexagon_width=10.0,
            dimension_ranges=((-50.0, 50.0), (-50.0, 50.0)),
        ),
        name="hexagonal_test",
    )
    # The field brightens over time, so successive frames must differ.
    fields = np.linspace(0.0, 1.0, N_ROWS)[:, None] * np.ones(env.n_bins)
    path = env.animate_fields(
        fields,
        frame_times=FRAME_TIMES,
        save_path=tmp_path / "hexagonal.mp4",
        vmin=0.0,
        vmax=1.0,
        dpi=40,
    )
    assert path == tmp_path / "hexagonal.mp4"
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    frames = [np.frombuffer(frame, dtype=np.uint8) for frame in reader]
    assert meta["fps"] == pytest.approx(9.0, rel=0.01)
    assert len(frames) == N_ROWS - 1
    assert not np.array_equal(frames[0], frames[-1])
    assert [item.name for item in tmp_path.iterdir()] == ["hexagonal.mp4"]


def test_1d_track_exports_an_html_player(tmp_path):
    pytest.importorskip("track_linearization")
    import networkx as nx
    from PIL import Image

    graph = nx.Graph()
    for i in range(5):
        graph.add_node(i, pos=(float(i * 10), 0.0))
    for i in range(4):
        graph.add_edge(i, i + 1, distance=10.0)
    env = Environment.from_graph(
        graph,
        edge_order=[(i, i + 1) for i in range(4)],
        edge_spacing=10.0,
        bin_size=2.0,
        name="linear_track_test",
    )
    fields = np.random.default_rng(42).random((N_ROWS, env.n_bins))
    path = env.animate_fields(
        fields, frame_times=FRAME_TIMES, save_path=tmp_path / "track.html", dpi=40
    )
    text = path.read_text(encoding="utf-8")
    frames = json.loads(re.search(r"const frames = (\[.*?\]);", text).group(1))
    assert len(frames) == N_ROWS - 1
    image = Image.open(io.BytesIO(base64.b64decode(frames[0])))
    assert image.size[0] > 0 and image.size[1] > 0


def _grid_subset():
    # 5 x 4 cells of 2 x 3 units; keep only the left two columns.
    grid = Environment.from_grid_mask(
        np.ones((5, 4), dtype=bool),
        grid_edges=(np.arange(0.0, 11.0, 2.0), np.arange(0.0, 13.0, 3.0)),
    )
    return grid.subset(mask=grid.bin_centers[:, 0] < 4.0)


def _rotated_grid():
    grid = Environment.from_grid_mask(
        np.ones((3, 4), dtype=bool),
        grid_edges=(np.arange(0.0, 7.0, 2.0), np.arange(0.0, 13.0, 3.0)),
    )
    angle = np.deg2rad(30.0)
    matrix = np.eye(3)
    matrix[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    matrix[:2, 2] = [100.0, -50.0]
    return grid.apply_transform(AffineND(matrix))


@pytest.mark.napari
@pytest.mark.xdist_group(name="napari_gui")
@pytest.mark.parametrize(
    "make_env", [_grid_subset, _rotated_grid], ids=["subset", "rotated"]
)
def test_napari_shows_each_bin_value_at_its_bin_center(make_env):
    pytest.importorskip("napari")
    env = make_env()
    values = np.arange(env.n_bins, dtype=float)
    view = env.animate_fields(
        np.array([values, values]),
        frame_times=np.array([0.0, 0.1]),
        backend="napari",
        show=False,
    )
    try:
        layer = view.native.layers["field"]
        # napari world coordinates are (y, x); data indices are pixels.
        pixels = np.array(
            [layer.world_to_data(center[::-1]) for center in env.bin_centers]
        )
        indices = np.rint(pixels).astype(int)
        np.testing.assert_allclose(pixels, indices, atol=1e-9)  # pixel centers
        np.testing.assert_array_equal(layer.data[tuple(indices.T)], values)
    finally:
        view.close()
