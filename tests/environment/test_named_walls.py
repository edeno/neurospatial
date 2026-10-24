"""Walls carry names through fitting, derivation, errors and storage."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from shapely.geometry import LineString, box

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.binning import Binning, KnownFaces
from neurospatial.ops.transforms import translate

DIVIDER = LineString([(2, 0), (2, 3)])
SHELF = LineString([(0, 3), (2, 3)])


def _grid(arena, **options):
    return Environment.from_grid_mask(
        np.ones((4, 4), bool),
        grid_edges=(np.arange(5.0), np.arange(5.0)),
        arena=arena,
        **options,
    )


def test_a_mapping_names_walls_and_a_sequence_gets_default_names():
    named = OpenArena(box(0, 0, 4, 4), walls={"divider": DIVIDER, "shelf": SHELF})
    assert named.wall_names == ("divider", "shelf")
    assert named.walls[0].equals(DIVIDER)
    unnamed = OpenArena(box(0, 0, 4, 4), walls=(DIVIDER, SHELF))
    assert unnamed.wall_names == ("wall_0", "wall_1")
    segments = OpenArena(
        box(0, 0, 4, 4), walls=(DIVIDER, SHELF), wall_names=("barrier", "barrier")
    )
    assert segments.wall_names == ("barrier", "barrier")


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"walls": {"divider": DIVIDER}, "wall_names": ("x",)}, "mapping of walls"),
        ({"walls": (DIVIDER,), "wall_names": ("a", "b")}, "2 names for 1 walls"),
        ({"walls": {"": DIVIDER}}, "nonempty strings"),
        ({"walls": (DIVIDER,), "wall_names": (3,)}, "nonempty strings"),
        ({"walls": {"outside": LineString([(5, 0), (6, 0)])}}, "'outside' lies"),
    ],
)
def test_invalid_wall_names_are_refused(options, match):
    with pytest.raises(ValueError, match=match):
        OpenArena(box(0, 0, 4, 4), **options)


def test_snapping_keeps_each_segment_named_after_its_wall():
    arena = OpenArena(
        box(0, 0, 4, 4),
        walls={"divider": LineString([(1.5, 0), (1.5, 4)]), "shelf": SHELF},
    )
    with pytest.warns(UserWarning, match=r"up to 0\.5 .*most for wall 'divider'"):
        env = _grid(arena)

    assert set(env.arena.wall_names) == {"divider", "shelf"}
    assert len(env.arena.wall_names) == len(env.arena.walls)
    for wall, name in zip(env.arena.walls, env.arena.wall_names, strict=True):
        # The divider moved to x = 1 or x = 2; the shelf was already on edges.
        if name == "divider":
            assert {x for x, _ in wall.coords} <= {1.0, 2.0}
        else:
            assert wall.equals(SHELF)


def test_a_wall_snapped_to_a_single_bin_edge_is_kept():
    # Dividing one bin, the wall snaps to exactly one edge.
    arena = OpenArena(box(0, 0, 4, 4), walls={"stub": LineString([(1.8, 0), (1.8, 1)])})
    with pytest.warns(UserWarning, match="moved the walls onto bin edges"):
        env = _grid(arena)
    assert env.arena.wall_names == ("stub",)
    assert env.arena.walls[0].equals(LineString([(2, 0), (2, 1)]))


def test_snapped_walls_stay_inside_a_slanted_arena():
    # Cells whose centers are inside a slanted outline can have edges outside
    # it; a wall snapped onto those edges keeps only its part in the arena.
    from shapely.geometry import Polygon

    from neurospatial.geometry._arena_fit import clip_wall

    angle = 0.0445
    c, s = np.cos(angle), np.sin(angle)
    corners = np.array([(0, 0), (10, 0), (10, 10), (0, 10)]) @ [[c, s], [-s, c]]
    outline = Polygon(corners)
    (wall,) = clip_wall(LineString([(-4.769, 9.436), (8.827, 8.0)]), outline)
    arena = OpenArena(outline, walls={"w": wall})
    with pytest.warns(UserWarning, match="moved the walls onto bin edges"):
        env = Environment.from_polygon(outline, bin_size=1.0, arena=arena)
    assert set(env.arena.wall_names) == {"w"}
    assert all(outline.covers(segment) for segment in env.arena.walls)
    assert set(env.get_blocked_edges()["wall"]) == {"w"}


def test_a_wall_through_a_saved_binning_is_named_in_the_error():
    source = _grid(None).binning
    source = replace(source, faces=KnownFaces(source.build_faces()))
    arena = OpenArena(
        box(0, 0, 4, 4), walls={"divider": LineString([(1.5, 0), (1.5, 4)])}
    )
    with pytest.raises(ValueError, match=r"wall 'divider' cuts through bin"):
        Environment.from_bins(source, arena=arena)


def test_names_survive_derivations_and_storage(tmp_path):
    env = _grid(OpenArena(box(0, 0, 4, 4), walls={"divider": DIVIDER}))
    moved = env.apply_transform(translate(10, 0))
    assert moved.arena.wall_names == ("divider",)
    assert moved.subset(mask=np.arange(env.n_bins) < 8).arena.wall_names == ("divider",)
    assert Environment.from_dict(env.to_dict()).arena.wall_names == ("divider",)
    env.to_file(tmp_path / "walled")
    assert Environment.from_file(tmp_path / "walled").arena.wall_names == ("divider",)
    combined = Environment.combine({"left": env, "right": moved}, auto_bridge=False)
    assert combined.arena.wall_names == ("divider", "divider")


def test_binning_import_is_unchanged_for_wall_free_arenas():
    # Arenas without walls report no names, so code that zips walls with
    # names needs no special case.
    env = _grid(None)
    assert isinstance(env.binning, Binning)
    assert env.arena.walls == ()
    assert env.arena.wall_names == ()
