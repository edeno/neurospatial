"""Worked notebook cells preserve spike-window and pixel-center conventions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.transforms import Affine2D as MplAffine2D

from neurospatial import Environment
from neurospatial.events import population_peri_event_histogram
from neurospatial.ops.transforms import Affine2D

jupytext = pytest.importorskip("jupytext")
ROOT = Path(__file__).resolve().parents[2]
VERSIONS = [
    (folder, suffix)
    for folder in ("examples", "docs/examples")
    for suffix in ("py", "ipynb")
]


def _cell(stem, version, marker):
    folder, suffix = version
    path = ROOT / folder / f"{stem}.{suffix}"
    cells = jupytext.read(path).cells
    return next(
        cell.source
        for cell in cells
        if cell.cell_type == "code" and marker in cell.source
    )


@pytest.mark.parametrize("version", VERSIONS)
def test_event_raster_and_histogram_count_the_same_half_open_windows(version):
    events = np.array([0.2, 1.0, 4.0, 8.0, 9.8])
    # Exactly two in-window spikes per retained event. The +1 s spikes at 2/9 s
    # belong to the recording, but are outside the first/last event's window.
    spikes = np.array([0.5, 1.0, 2.0, 3.5, 4.0, 7.5, 8.0, 9.0])
    namespace = {
        "np": np,
        "plt": plt,
        "event_times": events,
        "event_names": np.array([f"cue_{i}" for i in range(len(events))]),
        "spike_trains": [spikes, spikes.copy()],
        "unit_ids": np.array([7, 42]),
        "epochs": np.array([[0.0, 10.0]]),
        "spike_window": np.array([[0.0, 5.0], [7.0, 10.0]]),
        "population_peri_event_histogram": population_peri_event_histogram,
    }
    source = _cell(
        "07_peri_event_analysis",
        version,
        "population = population_peri_event_histogram(",
    )
    try:
        exec(compile(source, "peri_event_cohort_cell", "exec"), namespace)
        plot_source = _cell(
            "07_peri_event_analysis", version, "plot_peri_event_response("
        )
        exec(compile(plot_source, "peri_event_plot_cell", "exec"), namespace)
        result = namespace["result"]
        np.testing.assert_array_equal(result.event_indices, [1, 2, 3])
        rows = namespace["raster_ax"].collections
        assert [len(row.get_segments()) for row in rows] == [2, 2, 2]
        for row in rows:
            x = np.array([segment[0, 0] for segment in row.get_segments()])
            assert np.all((x >= -0.5) & (x < 1.0))
        assert result.histogram.sum() == pytest.approx(
            sum(len(row.get_segments()) for row in rows) / result.n_events
        )
    finally:
        plt.close("all")


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize(
    "matrix",
    [
        [[1.0, 0.0, -20.0], [0.0, -1.0, 110.0], [0.0, 0.0, 1.0]],
        [[0.8, 0.25, 5.0], [0.2, -0.9, 100.0], [0.0, 0.0, 1.0]],
    ],
    ids=["axis-aligned", "rotated-and-sheared"],
)
def test_video_preview_maps_pixel_centers_to_their_calibrated_positions(
    version, matrix
):
    height, width = 100, 120
    env = Environment.from_grid_mask(
        np.ones((15, 15), dtype=bool), grid_edges=(np.arange(16.0) * 10,) * 2
    )
    transform = Affine2D(np.array(matrix))
    landmarks = np.array([[20, 10], [110, 10], [110, 90], [20, 90]])
    marker = np.array([45.0, 30.0])
    namespace = {
        "plt": plt,
        "np": np,
        "MplAffine2D": MplAffine2D,
        "width": width,
        "height": height,
        "env": env,
        "transform": transform,
        "landmarks_px": landmarks,
        "video_frames": np.zeros((4, height, width, 3), dtype=np.uint8),
        "fields": np.zeros((4, env.n_bins)),
        "trajectory": np.repeat(transform(marker[None, :]), 4, axis=0),
        "frame_times": np.arange(4.0),
    }
    source = _cell("08_visualization", version, 'title="Raw synthetic frame"')
    try:
        exec(compile(source, "video_preview_cell", "exec"), namespace)
        ax = namespace["axes"][1]
        image = ax.images[0]
        left, right, bottom, top = image.get_extent()
        column, row = marker
        image_coordinates = [
            left + (column + 0.5) * (right - left) / width,
            top - (row + 0.5) * (top - bottom) / height,
        ]
        displayed_world = ax.transData.inverted().transform(
            image.get_transform().transform(image_coordinates)
        )
        np.testing.assert_allclose(
            displayed_world, transform(marker[None, :])[0], atol=1e-10
        )
    finally:
        plt.close("all")
