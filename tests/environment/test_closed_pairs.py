"""Walls close neighbor pairs without changing the bins or their pair order."""

from __future__ import annotations

import pickle
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, MultiPolygon, box

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.binning import BinConnections, KnownFaces
from neurospatial.geometry import decode, encode
from neurospatial.ops.transforms import translate

ROOMS = box(0, 0, 4, 3)
# Two 2 x 3 rooms side by side, split at x = 2.
WITH_DOOR = {
    "lower": LineString([(2, 0), (2, 1)]),
    "upper": LineString([(2, 2), (2, 3)]),
}
CLOSED = {"divider": LineString([(2, 0), (2, 3)])}


def _rooms(walls):
    return Environment.from_grid_mask(
        np.ones((4, 3), dtype=bool),
        grid_edges=(np.arange(5.0), np.arange(4.0)),
        arena=OpenArena(ROOMS, walls=walls),
        connect_diagonal_neighbors=False,
    )


def _across(env, u, v):
    x = env.bin_centers[[u, v], 0]
    return (x[0] < 2) != (x[1] < 2)


@pytest.fixture(scope="module")
def configurations():
    return _rooms({}), _rooms(WITH_DOOR), _rooms(CLOSED)


def test_configurations_share_bins_and_pair_order(configurations):
    no_wall, door, closed = configurations
    positions = np.random.default_rng(0).uniform([0, 0], [4, 3], (200, 2))
    for env in (door, closed):
        np.testing.assert_array_equal(env.bin_centers, no_wall.bin_centers)
        np.testing.assert_array_equal(env.bin_sizes, no_wall.bin_sizes)
        np.testing.assert_array_equal(env.bin_at(positions), no_wall.bin_at(positions))
        links = env.binning.connections
        np.testing.assert_array_equal(
            links.all_pairs, no_wall.binning.connections.all_pairs
        )
        np.testing.assert_array_equal(env.edge_pairs, links.all_pairs[links.is_open])


def test_only_pairs_across_a_wall_are_closed(configurations):
    no_wall, door, closed = configurations
    pairs = no_wall.binning.connections.all_pairs
    across = np.array([_across(no_wall, u, v) for u, v in pairs])
    row = no_wall.bin_centers[pairs[:, 0], 1]
    in_door = across & (row > 1) & (row < 2)
    np.testing.assert_array_equal(~closed.binning.connections.is_open, across)
    np.testing.assert_array_equal(~door.binning.connections.is_open, across & ~in_door)
    assert no_wall.binning.connections.is_open.all()


def test_closing_the_door_changes_reachability_not_assignment(configurations):
    no_wall, door, closed = configurations
    west, east = no_wall.bin_at(np.array([[0.5, 0.5], [3.5, 0.5]]))
    straight = nx.shortest_path_length(no_wall.connectivity, west, east, "distance")
    through_door = nx.shortest_path_length(door.connectivity, west, east, "distance")
    assert straight == pytest.approx(3.0)
    assert through_door == pytest.approx(5.0)  # up to the door row and back down
    assert not nx.has_path(closed.connectivity, west, east)


def test_blocked_edges_name_the_wall_that_closes_each_pair(configurations):
    no_wall, door, closed = configurations
    assert no_wall.get_blocked_edges().empty
    table = door.get_blocked_edges()
    assert list(table.columns) == ["source", "target", "distance", "wall"]
    lower = table["wall"] == "lower"
    rows = door.bin_centers[table["source"], 1]
    assert set(table["wall"]) == {"lower", "upper"}
    np.testing.assert_array_equal(rows[lower] < 1, True)
    np.testing.assert_array_equal(rows[~lower] > 2, True)
    np.testing.assert_allclose(table["distance"], 1.0)
    assert set(closed.get_blocked_edges()["wall"]) == {"divider"}


def test_closed_pairs_survive_derivation_storage_and_pickling(configurations, tmp_path):
    _, door, _ = configurations
    expected = door.binning.connections
    keep = np.ones(door.n_bins, dtype=bool)
    keep[0] = False  # drop the bottom-left bin, which has no closed pair
    variants = {
        "transform": door.apply_transform(translate(10, -5)),
        "dict": Environment.from_dict(door.to_dict()),
        "pickle": pickle.loads(pickle.dumps(door)),
    }
    door.to_file(tmp_path / "door")
    variants["file"] = Environment.from_file(tmp_path / "door")
    for name, env in variants.items():
        links = env.binning.connections
        np.testing.assert_array_equal(links.all_pairs, expected.all_pairs, err_msg=name)
        np.testing.assert_array_equal(links.is_open, expected.is_open, err_msg=name)
        assert list(env.get_blocked_edges()["wall"]) == list(
            door.get_blocked_edges()["wall"]
        ), name
    subset = door.subset(mask=keep).binning.connections
    assert np.count_nonzero(~subset.is_open) == np.count_nonzero(~expected.is_open)


def test_split_bins_close_the_pieces_a_wall_separates():
    env = Environment.from_grid_mask(
        np.ones((4, 4), dtype=bool),
        grid_edges=(np.arange(5.0), np.arange(5.0)),
        arena=OpenArena(
            box(0, 0, 4, 4), walls={"divider": LineString([(1.5, 0), (1.5, 4)])}
        ),
        wall_fit="split",
    )
    table = env.get_blocked_edges()
    assert set(table["wall"]) == {"divider"}
    x = env.bin_centers[:, 0]
    assert np.all((x[table["source"]] < 1.5) != (x[table["target"]] < 1.5))
    # Each of the four split cells' halves, at least, meet across the wall.
    assert len(table) >= 4
    open_pairs = env.edge_pairs
    assert not np.any((x[open_pairs[:, 0]] < 1.5) != (x[open_pairs[:, 1]] < 1.5))


def test_split_pieces_meeting_at_a_corner_on_a_wall_record_that_wall():
    # The wall bends at the grid corner (2, 2), so split pieces of diagonal
    # cells meet only there, and the line between their uneven centers can
    # pass beside the corner. The corner touching the wall closes the pair.
    env = Environment.from_grid_mask(
        np.ones((4, 4), dtype=bool),
        grid_edges=(np.arange(5.0), np.arange(5.0)),
        arena=OpenArena(
            box(0, 0, 4, 4), walls={"bend": LineString([(0, 1), (2, 2), (4, 3.6)])}
        ),
        wall_fit="split",
        connect_diagonal_neighbors=True,
    )
    links = env.binning.connections
    closed = ~links.is_open
    centers = env.bin_centers[links.all_pairs[closed]]
    beside = [not LineString(pair).intersects(env.arena.walls[0]) for pair in centers]
    assert any(beside)
    assert set(env.get_blocked_edges()["wall"]) == {"bend"}
    assert np.all(links.closed_by[closed] == 0)


def test_from_bins_keeps_supplied_closed_pairs_without_a_wall():
    source = _rooms(CLOSED).binning
    source = replace(source, faces=KnownFaces(source.faces.build()))
    env = Environment.from_bins(source, arena=OpenArena(ROOMS))
    np.testing.assert_array_equal(
        env.binning.connections.is_open, source.connections.is_open
    )
    assert env.get_blocked_edges()["wall"].isna().all()


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_binnings_saved_by_earlier_versions_are_refused(version):
    meta, arrays = encode(_rooms({}).binning, "binning")
    meta["codec_version"] = version
    with pytest.raises(
        ValueError, match=r"Binning codec_version .*supported version is 6"
    ):
        decode(meta, arrays, "binning")


def test_connections_validate_openness_and_close_only_open_pairs():
    with pytest.raises(ValueError, match="is_open must be a boolean array of shape"):
        BinConnections.from_arrays([[0, 1]], [1.0], is_open=[1])
    with pytest.raises(ValueError, match="is_open must be a boolean array of shape"):
        BinConnections.from_arrays([[0, 1]], [1.0], is_open=[True, False])
    links = BinConnections.from_arrays(
        [[0, 1], [1, 2], [2, 3]], [1.0, 2.0, 3.0], is_open=[True, False, True]
    )
    np.testing.assert_array_equal(links.pairs, [[0, 1], [2, 3]])
    np.testing.assert_array_equal(links.pair_distances, [1.0, 3.0])
    closed = links.closed(np.array([False, True]))
    np.testing.assert_array_equal(closed.is_open, [False, False, True])
    kept, open_kept = links.select(np.array([-1, 0, 1, 2]))
    np.testing.assert_array_equal(kept.all_pairs, [[0, 1], [1, 2]])
    np.testing.assert_array_equal(kept.is_open, [False, True])
    np.testing.assert_array_equal(open_kept, [False, True])
    for array in (links.all_pairs, links.is_open, links.pairs, links.pair_distances):
        assert not array.flags.writeable


def test_clipping_unchanged_cells_keeps_every_closed_pair():
    # Diagonal neighbors across the wall meet only at a corner on it.
    env = Environment.from_grid_mask(
        np.ones((4, 3), dtype=bool),
        grid_edges=(np.arange(5.0), np.arange(4.0)),
        arena=OpenArena(ROOMS, walls=CLOSED),
    )
    clipped = env.clip(ROOMS)
    assert clipped.n_bins == env.n_bins
    before, after = env.get_blocked_edges(), clipped.get_blocked_edges()
    assert len(before) == 7  # three edge neighbors and four diagonals

    def located(environment, table):
        # Each closed pair as the unordered pair of its bin centers.
        centers = environment.bin_centers
        return {
            frozenset((tuple(centers[u]), tuple(centers[v])))
            for u, v in zip(table["source"], table["target"], strict=True)
        }

    assert located(clipped, after) == located(env, before)
    assert set(after["wall"]) == {"divider"}


def _declared_closed_distance():
    env = _rooms(CLOSED)
    binning = env.binning
    links = binning.connections
    distances = links.all_distances.copy()
    closed = np.flatnonzero(~links.is_open)[0]
    distances[closed] = 7.0  # a measured detour around the divider
    declared = replace(
        binning,
        connections=BinConnections(
            links.all_pairs, distances, links.is_open, links.closed_by
        ),
        faces=KnownFaces(binning.faces.build()),
    )
    return Environment.from_bins(declared, arena=env.arena), links.all_pairs[closed]


def test_clipping_keeps_a_declared_distance_on_a_closed_pair():
    env, pair = _declared_closed_distance()
    clipped = env.clip(ROOMS).binning.connections
    row = np.flatnonzero((clipped.all_pairs == pair).all(axis=1))
    assert clipped.all_distances[row].tolist() == [7.0]
    assert not clipped.is_open[row].any()


def test_clipping_refuses_to_move_an_end_of_a_declared_closed_path():
    env, _ = _declared_closed_distance()
    # Cutting the column west of the divider moves the closed pair's west end.
    with pytest.raises(ValueError, match="moved an endpoint of declared navigation"):
        env.clip(box(1.5, 0, 4, 3))


def test_pieces_touching_only_at_a_clip_corner_are_not_closed_pairs():
    # No wall separates them: they are simply not neighbors.
    cell = Environment.from_grid_mask(
        np.ones((1, 1), dtype=bool), grid_edges=(np.array([0.0, 2.0]),) * 2
    )
    pieces = cell.clip(MultiPolygon([box(0, 0, 1, 1), box(1, 1, 2, 2)]))
    assert pieces.n_bins == 2
    assert len(pieces.binning.connections.all_pairs) == 0
    assert pieces.get_blocked_edges().empty


def test_each_closed_pair_records_the_wall_that_closes_it(configurations):
    _, door, _ = configurations
    links = door.binning.connections
    closed = ~links.is_open
    assert np.all(links.closed_by[~closed] == -1)
    names = np.array(door.arena.wall_names)[links.closed_by[closed]]
    rows = door.bin_centers[links.all_pairs[closed, 0], 1]
    assert set(names[rows < 1]) == {"lower"} and set(names[rows > 2]) == {"upper"}
    restored = Environment.from_dict(door.to_dict()).binning.connections
    np.testing.assert_array_equal(restored.closed_by, links.closed_by)


def test_combined_components_keep_their_own_walls():
    left = _rooms({"divider": LineString([(2, 0), (2, 3)])})
    right = Environment.from_grid_mask(
        np.ones((4, 3), dtype=bool),
        grid_edges=(np.arange(5.0) + 10, np.arange(4.0)),
        arena=OpenArena(
            box(10, 0, 14, 3), walls={"shelf": LineString([(10, 1), (14, 1)])}
        ),
        connect_diagonal_neighbors=False,
    )
    env = Environment.combine({"left": left, "right": right}, auto_bridge=False)
    table = env.get_blocked_edges()
    east = env.bin_centers[table["source"], 0] >= 10
    assert set(table["wall"][~east]) == {"divider"}
    assert set(table["wall"][east]) == {"shelf"}
    assert env.arena.wall_names == ("divider", "shelf")


def test_closing_walls_must_be_recorded_for_closed_pairs_only():
    with pytest.raises(ValueError, match="-1 for every open pair"):
        BinConnections.from_arrays([[0, 1]], [1.0], is_open=[True], closed_by=[0])
    with pytest.raises(ValueError, match="one wall index >= -1 per pair"):
        BinConnections.from_arrays([[0, 1]], [1.0], is_open=[False], closed_by=[-2])
