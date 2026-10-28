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
    # The connections are drawn under the cells; the cell edges over them.
    connections = [
        c
        for c in ax.collections
        if isinstance(c, LineCollection) and c.get_zorder() < 0
    ]
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


def test_a_region_past_the_bins_is_in_view(apparatus):
    regions = Regions(apparatus.regions.values())
    regions.add("start_box", polygon=box(40.0, 0.0, 60.0, 10.0))
    ax = apparatus.with_regions(regions).plot(show_landmarks=False)
    assert ax.get_xlim()[1] > 60.0
    assert ax.get_ylim()[1] == pytest.approx(40.0)  # never shrinks or overgrows


def test_only_drawn_regions_widen_the_view(apparatus):
    regions = Regions(apparatus.regions.values())
    regions.add("far", polygon=box(1000.0, 0.0, 1010.0, 10.0))
    env = apparatus.with_regions(regions)
    ax = env.plot(show_landmarks=False, regions_plot_kwargs={"region_names": ["goal"]})
    assert ax.get_xlim()[1] < 100.0


def test_each_layer_can_be_hidden(apparatus):
    ax = apparatus.plot(show_walls=False, show_landmarks=False, show_regions=False)
    assert _texts(ax) == set()
    assert ax.get_legend() is None
    assert ax.get_ylim()[1] == pytest.approx(40.0)


@pytest.mark.parametrize("n", [4, 10])
def test_every_cell_edge_is_drawn(n):
    # Minor ticks on major ticks are dropped by Matplotlib; the cell edges
    # must not depend on them (with edges 0..4 every edge is a major tick).
    edges = np.arange(n + 1.0)
    env = Environment.from_grid_mask(
        np.ones((n, n), dtype=bool), grid_edges=(edges, edges)
    )
    ax = env.plot(show_connectivity=False)
    ax.figure.canvas.draw()
    (lines,) = [c for c in ax.collections if isinstance(c, LineCollection)]
    segments = np.array(lines.get_segments())
    vertical = segments[segments[:, 0, 0] == segments[:, 1, 0]]
    horizontal = segments[segments[:, 0, 1] == segments[:, 1, 1]]
    np.testing.assert_array_equal(np.sort(vertical[:, 0, 0]), edges)
    np.testing.assert_array_equal(np.sort(horizontal[:, 0, 1]), edges)
    np.testing.assert_array_equal(vertical[:, :, 1], [[0.0, n]] * (n + 1))


def test_a_ring_shaped_landmark_keeps_its_hole():
    env = Environment.from_grid_mask(
        np.ones((4, 4), dtype=bool), grid_edges=(np.arange(5.0), np.arange(5.0))
    )
    ring = box(0.5, 0.5, 3.5, 3.5).difference(box(1.5, 1.5, 2.5, 2.5))
    with_ring = env.with_landmarks(Landmarks([Landmark("ring", "polygon", ring)]))

    def color_at(environment, x, y):
        ax = environment.plot()
        ax.figure.canvas.draw()
        image = np.asarray(ax.figure.canvas.buffer_rgba())[..., :3]
        column, row = ax.transData.transform((x, y))
        return image[image.shape[0] - int(row), int(column)].astype(int)

    # The hole shows the cells beneath, as without the landmark; the ring does not.
    np.testing.assert_array_equal(
        color_at(with_ring, 2.0, 2.0), color_at(env, 2.0, 2.0)
    )
    assert not np.array_equal(color_at(with_ring, 1.0, 2.0), color_at(env, 1.0, 2.0))


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
