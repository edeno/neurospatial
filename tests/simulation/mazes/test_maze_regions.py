"""Every named region of a built-in maze covers at least one bin.

A region covering no bins makes trial, visit and run analyses refuse it, so a
built-in maze must place its wells, arm ends and goals where its bins are, in
both its 2-D and its linearized track environment.
"""

import pytest


@pytest.fixture
def maze(default_maze):
    return default_maze


def test_2d_regions_cover_bins(maze):
    env = maze.env_2d
    assert len(env.regions) > 0
    empty = [name for name in env.regions if len(env.bins_in_region(name)) == 0]
    assert empty == []


def test_track_carries_the_2d_regions_and_they_cover_bins(maze):
    if maze.env_track is None:
        pytest.skip("open-field maze without a track")
    track = maze.env_track
    assert list(track.regions) == list(maze.env_2d.regions)
    empty = [name for name in track.regions if len(track.bins_in_region(name)) == 0]
    assert empty == []


def test_track_point_regions_sit_at_the_nearest_centerline_point(maze):
    """A 2-D point region moves onto the track by the shortest distance.

    The nearest centreline point is computed independently with shapely, so a
    region placed on the wrong segment (for example at another node sharing
    its linear position) fails, however far away it lands.
    """
    if maze.env_track is None:
        pytest.skip("open-field maze without a track")
    import numpy as np
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union

    graph = maze.env_track._binning.linearization.graph
    centerline = unary_union(
        [
            LineString([graph.nodes[u]["pos"], graph.nodes[v]["pos"]])
            for u, v in graph.edges
        ]
    )
    misplaced = []
    for name, region in maze.env_2d.regions.items():
        if region.kind != "point":
            continue
        point = np.asarray(region.data, dtype=float)
        on_track = np.asarray(maze.env_track.regions[name].data, dtype=float)
        moved = float(np.linalg.norm(on_track - point))
        nearest = centerline.distance(Point(point))
        if moved > nearest + 1e-6:
            misplaced.append((name, round(moved, 2), round(nearest, 2)))
    assert misplaced == []
