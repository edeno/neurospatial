"""The environment overview: bins, connections, walls, landmarks and regions."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection, QuadMesh
from shapely.geometry import LineString, Polygon, box

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.landmarks import Landmark, Landmarks
from neurospatial.regions import Regions

EDGES = np.arange(0.0, 41.0, 4.0)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def apparatus():
    # A box with a divider, a cue beyond the north wall, a monitor and a goal.
    arena = OpenArena(
        box(0, 0, 40, 40), walls={"divider": LineString([(20, 0), (20, 28)])}
    )
    env = Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool),
        grid_edges=(EDGES, EDGES),
        units="cm",
        arena=arena,
    )
    regions = Regions()
    regions.add("goal", point=[30.0, 30.0])
    return env.with_regions(regions).with_landmarks(
        Landmarks(
            [
                Landmark("cue", "point", [20.0, 55.0]),
                Landmark("monitor", "line", [(5.0, 40.0), (35.0, 40.0)]),
                Landmark("object", "polygon", box(28, 8, 34, 14)),
            ]
        )
    )


def _texts(ax):
    return {text.get_text() for text in ax.texts}


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


def test_walls_landmarks_and_regions_are_drawn_by_default(apparatus):
    ax = apparatus.plot()
    assert {"divider", "cue", "monitor", "object"} <= _texts(ax)
    assert ax.get_legend_handles_labels()[1] == ["goal"]
    walls = [c for c in ax.collections if c.get_linewidths()[0] == 3]
    assert [segments.tolist() for segments in walls[0].get_segments()] == [
        [[20.0, 0.0], [20.0, 28.0]]
    ]
    assert ax.get_ylim()[1] > 55.0  # the cue beyond the arena is in view


def test_each_layer_can_be_hidden(apparatus):
    ax = apparatus.plot(show_walls=False, show_landmarks=False, show_regions=False)
    assert _texts(ax) == set()
    assert ax.get_legend() is None
    assert ax.get_ylim()[1] == pytest.approx(40.0)


def test_polygon_grid_outlines_its_polygon():
    env = Environment.from_polygon(
        Polygon([(0, 0), (40, 0), (40, 30), (20, 40), (0, 30)]), bin_size=4.0
    )
    ax = env.plot()
    (outline,) = ax.patches
    assert outline.get_facecolor()[3] == 0.0  # drawn, not filled over the bins
    assert outline.get_edgecolor()[:3] == (0.0, 0.0, 0.0)


@pytest.mark.slow
@pytest.mark.wallclock
def test_a_hundred_thousand_bins_draw_in_seconds():
    env = Environment.from_samples(
        np.random.default_rng(0).uniform(0, 100, (200_000, 2)), bin_size=0.28
    )
    assert env.n_bins > 100_000
    start = time.perf_counter()
    env.plot().figure.canvas.draw()
    assert time.perf_counter() - start < 5.0  # 31 s with a line per connection
