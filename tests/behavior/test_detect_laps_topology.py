"""detect_laps needs a loop; back-and-forth runs are trials."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import shapely

from neurospatial import Environment, InputError
from neurospatial.behavior.segmentation import detect_laps
from neurospatial.simulation.mazes import make_plus_maze, make_w_maze, make_y_maze


def test_detect_laps_refuses_linear_track(linear_track, continuous_lap_track):
    env, times, positions = linear_track.env, linear_track.times, linear_track.positions
    assert nx.cycle_basis(env.connectivity) == []
    with pytest.raises(ValueError, match=r"has no loop[\s\S]*detect_trials"):
        detect_laps(env, times, positions)

    # A template round trip is the user's own definition of a lap, so it
    # still segments the runs: out to the far end and back to the start.
    x = positions[:, 0]
    far = int(np.argmax(x > x.max() - 5.0))
    back = far + int(np.argmax(x[far:] < x[0] + 5.0))
    first = env.bin_at(positions[: back + 1])
    trips = detect_laps(env, times, positions, method="reference", reference_lap=first)
    assert len(trips) > 1
    assert np.all(trips.overlap_score >= 0.8)

    # On a loop, laps are still detected, one per 40 s circuit, with a direction.
    loop = continuous_lap_track
    assert nx.cycle_basis(loop.env.connectivity) != []
    laps = detect_laps(loop.env, loop.times, loop.positions)
    assert len(laps) >= 4
    assert np.all(np.isin(laps.direction, ["clockwise", "counter-clockwise"]))


# The loop test reads the topology of the space, not the bin graph: grid
# neighbours and the bins at a junction always form short cycles.


def _circle(center, radius, n_samples, turns=3.0):
    theta = np.linspace(0, turns * 2 * np.pi, n_samples)
    return np.c_[center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)]


def test_open_box_has_no_loop():
    """Circling in an open box returns no lap: the box has no hole."""
    x = np.linspace(0, 100, 101)
    xx, yy = np.meshgrid(x, x)
    env = Environment.from_samples(np.c_[xx.ravel(), yy.ravel()], bin_size=2.5)
    assert nx.cycle_basis(env.connectivity) != []  # grid cycles are not loops
    positions = _circle((50, 50), 30.0, 900)
    with pytest.raises(InputError, match="has no hole to run around"):
        detect_laps(env, np.arange(900) / 30.0, positions)


@pytest.mark.parametrize("build", ["polygon", "samples"])
def test_annulus_has_a_loop(build):
    ring = shapely.Point(50, 50).buffer(40).difference(shapely.Point(50, 50).buffer(25))
    positions = _circle((50, 50), 32.0, 900)
    if build == "polygon":
        env = Environment.from_polygon(ring, bin_size=3.0)
    else:
        env = Environment.from_samples(positions, bin_size=3.0)
    laps = detect_laps(env, np.arange(900) / 30.0, positions)
    assert len(laps) >= 2
    assert np.all(np.isin(laps.direction, ["clockwise", "counter-clockwise"]))


@pytest.mark.parametrize("direction", ["increasing", "decreasing"])
def test_loop_refuses_linear_directions(direction):
    """Laps round a loop are clockwise or counter-clockwise, never increasing."""
    ring = shapely.Point(50, 50).buffer(40).difference(shapely.Point(50, 50).buffer(25))
    env = Environment.from_polygon(ring, bin_size=3.0)
    # Three counter-clockwise turns (theta increases).
    positions = _circle((50, 50), 32.0, 900)
    times = np.arange(900) / 30.0
    laps = detect_laps(env, times, positions, direction="counter-clockwise")
    assert len(laps) >= 2
    with pytest.raises(
        ValueError, match=r"'clockwise' or 'counter-clockwise'[\s\S]*has a loop"
    ):
        detect_laps(env, times, positions, direction=direction)


@pytest.mark.parametrize("direction", ["clockwise", "counter-clockwise"])
def test_linear_track_refuses_loop_directions(linear_track, direction):
    """Round trips on a linear track are increasing or decreasing, never clockwise."""
    env, times, positions = linear_track.env, linear_track.times, linear_track.positions
    x = positions[:, 0]
    far = int(np.argmax(x > x.max() - 5.0))
    back = far + int(np.argmax(x[far:] < x[0] + 5.0))
    first = env.bin_at(positions[: back + 1])
    trips = detect_laps(env, times, positions, method="reference", reference_lap=first)
    # The template starts at the low end, so every round trip leaves toward
    # larger linear coordinates.
    increasing = detect_laps(
        env,
        times,
        positions,
        method="reference",
        reference_lap=first,
        direction="increasing",
    )
    assert len(increasing) == np.sum(trips.direction == "increasing") > 0
    with pytest.raises(ValueError, match=r"'increasing' or 'decreasing'[\s\S]*no loop"):
        detect_laps(
            env,
            times,
            positions,
            method="reference",
            reference_lap=first,
            direction=direction,
        )


def test_figure_eight_track_has_loops():
    nodes = {
        0: (0, 0),
        1: (20, 20),
        2: (40, 0),
        3: (20, -20),
        4: (-20, 20),
        5: (-40, 0),
        6: (-20, -20),
    }
    order = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5), (5, 6), (6, 0)]
    graph = nx.Graph()
    for node, pos in nodes.items():
        graph.add_node(node, pos=pos)
    graph.add_edges_from(order)
    env = Environment.from_graph(
        graph, edge_order=order, edge_spacing=0.0, bin_size=2.0, track_width=4.0
    )
    circuit = np.vstack(
        [np.linspace(nodes[a], nodes[b], 40, endpoint=False) for a, b in order]
    )
    positions = np.vstack([circuit] * 3)
    # The start is the crossing, so the first circuit is one lobe and the
    # trips round the other lobe are in no lap, which the warning reports.
    with pytest.warns(UserWarning, match=r"3 trips .* against 3 laps"):
        laps = detect_laps(env, np.arange(len(positions)) / 30.0, positions)
    assert len(laps) >= 2


@pytest.mark.parametrize("make", [make_plus_maze, make_w_maze, make_y_maze])
def test_tree_track_has_no_loop(make):
    """A junction's clique of bins is a short cycle, not a loop."""
    env = make().env_track
    assert nx.cycle_basis(env.connectivity) != []
    from neurospatial.behavior.segmentation import _space_has_loop

    assert _space_has_loop(env) is False
    assert _space_has_loop(make().env_2d) is False


def test_three_dimensional_environment_is_refused():
    rng = np.random.default_rng(0)
    env = Environment.from_samples(rng.uniform(0, 50, (2000, 3)), bin_size=10.0)
    positions = rng.uniform(0, 50, (50, 3))
    with pytest.raises(InputError, match="cannot tell whether this 3-D"):
        detect_laps(env, np.arange(50) / 30.0, positions)


@pytest.mark.parametrize(
    ("hole", "ring"), [(h, w) for h in (1, 2, 3, 4) for w in (1, 3)]
)
def test_grid_raster_check_matches_cell_polygons(hole, ring):
    """The fast raster check on grids gives the polygon answer.

    A square hole of ``hole`` bins has an inscribed radius of ``hole / 2``
    bins, a loop only above one bin.
    """
    from neurospatial.behavior.segmentation import _cells_have_loop, _space_has_loop

    n = hole + 2 * ring
    mask = np.ones((n, n), dtype=bool)
    mask[ring : ring + hole, ring : ring + hole] = False
    env = Environment.from_grid_mask(mask, (np.arange(n + 1.0), np.arange(n + 1.0)))
    assert _space_has_loop(env) is _cells_have_loop(env) is (hole > 2)


@pytest.mark.parametrize(
    ("outer", "width", "bin_size", "loop"),
    [(30, 10, 10, True), (15, 5, 5, True), (15, 5, 8, False), (20, 8, 10, False)],
)
def test_annulus_raster_check_matches_cell_polygons(outer, width, bin_size, loop):
    from neurospatial.behavior.segmentation import _cells_have_loop, _space_has_loop

    ring = (
        shapely.Point(0, 0)
        .buffer(outer)
        .difference(shapely.Point(0, 0).buffer(outer - width))
    )
    env = Environment.from_polygon(ring, bin_size=bin_size)
    assert _space_has_loop(env) is _cells_have_loop(env) is loop
