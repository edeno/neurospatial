"""The environment overview: bins and the connections between them."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection, QuadMesh
from shapely.geometry import Polygon

from neurospatial import Environment

EDGES = np.arange(0.0, 41.0, 4.0)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_connections_are_one_collection_whatever_the_bin_count():
    env = Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool), grid_edges=(EDGES, EDGES), units="cm"
    )
    ax = env.plot()
    connections = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert [len(c.get_segments()) for c in connections] == [len(env.edge_pairs)]
    assert len(ax.lines) == 0  # no line artist per connection


def test_active_cells_look_the_same_in_full_and_partial_grids():
    full = Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool), grid_edges=(EDGES, EDGES), units="cm"
    )
    mask = np.ones((10, 10), dtype=bool)
    mask[:3, :3] = False
    partial = Environment.from_grid_mask(mask, grid_edges=(EDGES, EDGES), units="cm")

    def cell_colors(env):
        (mesh,) = [c for c in env.plot().collections if isinstance(c, QuadMesh)]
        mesh.figure.canvas.draw()
        colors = mesh.get_facecolors()
        return colors, np.ma.getmaskarray(mesh.get_array()).ravel()

    full_colors, _ = cell_colors(full)
    partial_colors, hidden = cell_colors(partial)
    np.testing.assert_allclose(
        np.unique(full_colors, axis=0), partial_colors[~hidden][:1]
    )
    assert hidden.sum() == 9  # the inactive cells are not filled


def test_polygon_grid_outlines_its_polygon():
    env = Environment.from_polygon(
        Polygon([(0, 0), (40, 0), (40, 30), (20, 40), (0, 30)]), bin_size=4.0
    )
    ax = env.plot()
    (outline,) = ax.patches
    assert outline.get_facecolor()[3] == 0.0  # drawn, not filled over the bins
    assert outline.get_edgecolor()[:3] == (0.0, 0.0, 0.0)


@pytest.mark.wallclock
def test_a_hundred_thousand_bins_draw_in_seconds():
    env = Environment.from_samples(
        np.random.default_rng(0).uniform(0, 100, (200_000, 2)), bin_size=0.28
    )
    assert env.n_bins > 100_000
    start = time.perf_counter()
    env.plot().figure.canvas.draw()
    assert time.perf_counter() - start < 5.0  # 31 s with a line per connection
