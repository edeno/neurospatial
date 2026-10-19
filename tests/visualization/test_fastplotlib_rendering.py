"""plot and plot_field with backend="fastplotlib", rendered off-screen.

Skipped without fastplotlib (``pip install "neurospatial[fastplotlib]"``) or a
GPU adapter.
"""

from __future__ import annotations

import os

# Render into memory: no window, and the same in a terminal, on CI or over SSH.
os.environ.setdefault("RENDERCANVAS_FORCE_OFFSCREEN", "1")

import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib.colors import to_rgba
from shapely.geometry import LineString, box

fastplotlib = pytest.importorskip("fastplotlib")
wgpu = pytest.importorskip("wgpu")
try:
    _ADAPTER = wgpu.gpu.request_adapter_sync()
except Exception:
    _ADAPTER = None
if _ADAPTER is None:
    pytest.skip("no GPU adapter for fastplotlib", allow_module_level=True)

from neurospatial import Environment  # noqa: E402
from neurospatial.arena import OpenArena  # noqa: E402
from neurospatial.landmarks import Landmark, Landmarks  # noqa: E402

EDGES = np.arange(0.0, 41.0, 4.0)


@pytest.fixture(scope="module")
def apparatus():
    arena = OpenArena(
        box(0, 0, 40, 40), walls={"divider": LineString([(20, 0), (20, 28)])}
    )
    env = Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool),
        grid_edges=(EDGES, EDGES),
        units="cm",
        arena=arena,
    )
    return env.with_landmarks(Landmarks([Landmark("cue", "point", [20.0, 44.0])]))


def _pixel(figure, subplot, x, y):
    """The rendered color at environment coordinates (x, y) of ``subplot``."""
    image = np.asarray(figure.canvas.draw())
    xy = subplot.map_world_to_screen((x, y, 0))
    ratio = image.shape[1] / figure.canvas.get_logical_size()[0]
    column, row = (np.asarray(xy[:2]) * ratio).astype(int)
    return image[row, column, :3] / 255.0


def test_plot_and_plot_field_return_figures(apparatus):
    figure = apparatus.plot(backend="fastplotlib")
    assert isinstance(figure, fastplotlib.Figure)
    figure.show()
    texts = [g.text for g in figure[0, 0].graphics if hasattr(g, "text")]
    assert {"divider", "cue"} <= set(texts)  # walls and landmarks are named
    field = np.arange(apparatus.n_bins, dtype=float)
    assert isinstance(
        apparatus.plot_field(field, backend="fastplotlib"), fastplotlib.Figure
    )


def test_each_bin_shows_its_value_and_missing_bins_their_own_color():
    env = Environment.from_grid_mask(
        np.ones((2, 1), dtype=bool),
        grid_edges=(np.array([0.0, 10, 20]), np.array([0.0, 10])),
    )
    figure = env.plot_field(
        np.array([0.0, np.nan]), backend="fastplotlib", vmin=0.0, vmax=1.0
    )
    figure.show()
    plot = figure[0]
    low = _pixel(figure, plot, 5.0, 5.0)
    missing = _pixel(figure, plot, 15.0, 5.0)
    np.testing.assert_allclose(low, colormaps["viridis"](0.0)[:3], atol=2 / 255)
    np.testing.assert_allclose(missing, to_rgba("lightgray")[:3], atol=2 / 255)


def test_three_dimensional_fields_draw_points_whose_opacity_follows_value():
    env = Environment.from_samples(
        np.random.default_rng(0).uniform(0, 10, (3000, 3)), bin_size=2.0
    )
    field = np.linspace(0.0, 1.0, env.n_bins)
    figure = env.plot_field(field, backend="fastplotlib", colorbar=False)
    (points,) = [
        g for g in figure[0, 0].graphics if type(g).__name__ == "ScatterGraphic"
    ]
    alpha = points.colors[:][:, 3]
    assert alpha[np.argmax(field)] > 0.99 > 0.1 > alpha[np.argmin(field)]


def test_matplotlib_options_are_refused_and_unknown_backends_named(apparatus):
    with pytest.raises(TypeError, match="Matplotlib-only options: scale_bar"):
        apparatus.plot(backend="fastplotlib", scale_bar=True)
    with pytest.raises(TypeError, match="ax"):
        apparatus.plot_field(
            np.zeros(apparatus.n_bins), ax=object(), backend="fastplotlib"
        )
    with pytest.raises(ValueError, match="Unknown backend 'plotly'"):
        apparatus.plot(backend="plotly")
