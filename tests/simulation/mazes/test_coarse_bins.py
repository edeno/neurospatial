"""Bins too coarse for a maze's corridors or regions are reported, not silent.

Bins wider than about half a corridor can leave a stretch of corridor with
no bin centre, splitting the 2-D maze into pieces; a region smaller than a
bin can contain no bin centre. Either breaks paths, distances or visits
without an error, so the maze factories warn and name a bin size that works.
"""

from __future__ import annotations

import re
import warnings

import pytest

from neurospatial.simulation import mazes

MAKERS = sorted(
    name
    for name in dir(mazes)
    if name.startswith("make_")
    and (name.endswith("maze") or name == "make_linear_track")
)


def _build(name, bin_size):
    """Build a maze, returning it and the coarse-bin warnings it raised."""
    kwargs = {"bin_size": bin_size}
    if name not in {
        "make_barnes_maze",
        "make_cheeseboard_maze",
        "make_square_maze",
        "make_watermaze",
    }:
        kwargs["include_track"] = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        maze = getattr(mazes, name)(**kwargs)
    messages = [
        str(w.message)
        for w in caught
        if "unconnected pieces" in str(w.message) or "cover" in str(w.message)
    ]
    return maze, messages


def _suggested_bin_size(message: str) -> float:
    return float(re.search(r"bin_size at most ([0-9.]+) cm", message).group(1))


@pytest.mark.parametrize("name", MAKERS)
def test_no_maze_warns_at_its_default_bin_size(name):
    if name == "make_rat_hexmaze":
        pytest.skip("covered at bin_size=4 below; the 2 cm grid is large")
    maze, messages = _build(name, 2.0)
    assert messages == []
    assert len(maze.env_2d.components()) == 1


def test_rat_hexmaze_does_not_warn_at_4_cm():
    _, messages = _build("make_rat_hexmaze", 4.0)
    assert messages == []


@pytest.mark.parametrize(
    ("name", "label", "bin_size", "n_pieces", "corridor"),
    [
        ("make_crossword_maze", "crossword maze", 5.0, 2, "5.00"),
        ("make_sungod_maze", "Sungod maze", 5.0, 3, "4.46"),
        ("make_small_hex_maze", "small hex maze", 10.0, 2, "7.00"),
    ],
)
def test_coarse_bins_that_split_a_maze_warn_with_a_bin_size_that_works(
    name, label, bin_size, n_pieces, corridor
):
    maze, messages = _build(name, bin_size)
    assert len(maze.env_2d.components()) == n_pieces
    split = [m for m in messages if "unconnected pieces" in m]
    assert len(split) == 1
    assert f"splits the {label} into {n_pieces} unconnected pieces" in split[0]
    assert f"narrowest corridor is {corridor} cm wide" in split[0]
    fixed, after = _build(name, _suggested_bin_size(split[0]))
    assert len(fixed.env_2d.components()) == 1
    assert [m for m in after if "unconnected pieces" in m] == []


@pytest.mark.parametrize(
    ("name", "bin_size", "region"),
    [
        ("make_barnes_maze", 8.0, "escape_hole"),
        ("make_radial_arm_maze", 8.0, "food_cup_0"),
        ("make_w_maze", 8.0, "well_2"),
        ("make_t_maze", 10.0, "start"),
    ],
)
def test_regions_that_cover_no_bin_warn_with_a_bin_size_that_works(
    name, bin_size, region
):
    maze, messages = _build(name, bin_size)
    assert len(maze.env_2d.bins_in_region(region)) == 0
    empty = [m for m in messages if "cover" in m]
    assert len(empty) == 1
    assert region in empty[0]
    fixed, after = _build(name, _suggested_bin_size(empty[0]))
    env = fixed.env_2d
    assert all(len(env.bins_in_region(r)) > 0 for r in env.regions)
    assert [m for m in after if "cover" in m] == []
