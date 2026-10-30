"""Every maze's dimensions refuse values that cannot build the apparatus.

A length must be positive and finite, a count large enough to make the
maze, and the parts must stay apart and in place. Refusing at construction
says which values conflict, instead of building a silently wrong maze or
failing inside the geometry code.
"""

from __future__ import annotations

import dataclasses
import math
from itertools import pairwise

import numpy as np
import pytest
import shapely

from neurospatial.simulation import mazes

DIMS_CLASSES = sorted(
    (
        getattr(mazes, name)
        for name in mazes.__all__
        if name.endswith("Dims") and name != "MazeDims"
    ),
    key=lambda cls: cls.__name__,
)
# Gaps that may close to zero.
ZERO_ALLOWED = {("HoneycombDims", "gap")}


def _numeric_fields(cls):
    for field in dataclasses.fields(cls):
        if isinstance(field.default, bool):
            continue
        if isinstance(field.default, (int, float)):
            yield field


def test_every_dims_class_is_checked():
    assert len(DIMS_CLASSES) == 19


@pytest.mark.parametrize("cls", DIMS_CLASSES, ids=lambda cls: cls.__name__)
def test_defaults_build_and_are_frozen(cls):
    dims = cls()
    field = dataclasses.fields(cls)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(dims, field, getattr(dims, field))


@pytest.mark.parametrize("cls", DIMS_CLASSES, ids=lambda cls: cls.__name__)
def test_every_length_refuses_negative_and_nan(cls):
    for field in _numeric_fields(cls):
        if isinstance(field.default, int):
            continue
        bad = [math.nan, math.inf, -1.0]
        if (cls.__name__, field.name) not in ZERO_ALLOWED:
            bad.append(0.0)
        for value in bad:
            with pytest.raises(ValueError, match=field.name):
                cls(**{field.name: value})


@pytest.mark.parametrize("cls", DIMS_CLASSES, ids=lambda cls: cls.__name__)
def test_every_count_refuses_zero_and_fractions(cls):
    for field in _numeric_fields(cls):
        if not isinstance(field.default, int):
            continue
        for value in (0, -1, 2.5):
            with pytest.raises(ValueError, match=field.name):
                cls(**{field.name: value})


@pytest.mark.parametrize(
    ("cls", "kwargs", "match"),
    [
        # Silently wrong before: a negative side built a 30 cm box.
        ("SquareMazeDims", {"side_length": -30.0}, "is not a length"),
        # The wells swapped sides.
        (
            "TMazeDims",
            {"arm_length": 3.0, "corridor_width": 10.0},
            "no longer than it is wide",
        ),
        ("TMazeDims", {"stem_length": math.nan}, "is not a length"),
        ("LinearTrackDims", {"length": -150.0}, "is not a length"),
        ("LinearTrackDims", {"length": 8.0}, "wells"),
        ("YMazeDims", {"arm_length": 5.0}, "no longer than it is wide"),
        ("PlusMazeDims", {"arm_length": 5.0}, "no longer than it is wide"),
        # Hole 0 moved to 180 degrees.
        ("BarnesDims", {"hole_inset": 70.0}, "overlap"),
        # A hole spilled past the rim.
        ("BarnesDims", {"hole_inset": 3.0}, "run off the platform"),
        ("BarnesDims", {"n_holes": 1}, "at least one other hole"),
        ("BarnesDims", {"n_holes": 40}, "overlap"),
        # Wells off the board, and a ZeroDivisionError.
        ("CheeseboardDims", {"rim_margin": -5.0}, "rim_margin"),
        ("CheeseboardDims", {"rim_margin": 1.0}, "run off the board"),
        ("CheeseboardDims", {"grid_spacing": 0.0}, "is not a length"),
        ("CheeseboardDims", {"grid_spacing": 2.0}, "overlap"),
        ("CheeseboardDims", {"rim_margin": 60.0}, "no room for wells"),
        ("WMazeDims", {"n_wells": 1}, "at least two arms"),
        ("WMazeDims", {"n_wells": 0}, "at least two arms"),
        # Arms merged.
        (
            "WMazeDims",
            {"outer_arm_spacing": 5.0, "corridor_width": 7.0},
            "merge",
        ),
        ("WMazeDims", {"n_wells": 11}, "merge"),
        ("RadialArmDims", {"n_arms": 1}, "at least two arms"),
        ("RadialArmDims", {"n_arms": 0}, "at least two arms"),
        ("RadialArmDims", {"n_arms": 16}, "overlap where they leave"),
        ("RadialArmDims", {"food_cup_width": 12.0}, "wider than the arm"),
        ("RadialArmDims", {"food_cup_length": 80.0}, "longer than the arm"),
        ("HoneycombDims", {"gap": -5.0}, "is not a length"),
        ("HoneycombDims", {"n_rings": 0}, "at least one ring"),
        ("SungodDims", {"scale": -1.0}, "not a scale factor"),
        ("SungodDims", {"scale": math.nan}, "not a scale factor"),
        # A MultiPolygon TypeError from from_polygon.
        ("HamletDims", {"split_radius": 10.0}, "in order"),
        ("HamletDims", {"split_radius": 25.0}, "radial alley"),
        ("HamletDims", {"n_arms": 2}, "at least three sides"),
        ("HamletDims", {"agora_wall_thickness": 20.0}, "no walled agora"),
        ("RatHexmazeDims", {"bridge_length": -200.0}, "is not a length"),
        ("RatHexmazeDims", {"bridge_length": 8.0}, "small bridge"),
        ("RatHexmazeDims", {"n_modules": 5}, "HexMaze layout"),
        ("RatHexmazeDims", {"node_spacing": 9.0}, "module corridor"),
        ("HamptonCourtDims", {"frame_width": 150.0}, "galleries"),
        ("RepeatedTDims", {"long_arm_length": 40.0}, "blind alley 6"),
        ("RepeatedTDims", {"wall_thickness": 35.0}, "blind alley 4"),
        ("RepeatedTDims", {"food_box_depth": 41.0}, "food box"),
        ("RepeatedYDims", {"n_junctions": 0}, "at least one Y-junction"),
        ("RepeatedYDims", {"fork_length": 50.0}, "forks"),
        ("RepeatedYDims", {"fork_length": 8.0}, "fork arm"),
        ("CrosswordDims", {"corridor_width": 19.0}, "parallel tracks touch"),
        ("CrosswordDims", {"box_size": 34.0}, "start box reach"),
    ],
)
def test_refuses_dimensions_that_cannot_build_the_maze(cls, kwargs, match):
    dims_class = getattr(mazes, cls)
    with pytest.raises(ValueError, match=match) as error:
        dims_class(**kwargs)
    # The message names the class and every value it refuses.
    assert cls in str(error.value)
    for name in kwargs:
        assert name in str(error.value)


def test_a_length_error_says_what_to_pass():
    with pytest.raises(ValueError) as error:
        mazes.TMazeDims(stem_length=-1.0)
    assert str(error.value) == (
        "TMazeDims(stem_length=-1.0) is not a length: stem_length must be a "
        "finite number of cm, positive. Pass a value such as the default, "
        "stem_length=100.0."
    )


def test_radial_arms_just_narrower_than_the_limit_stay_apart():
    """The limit is where neighbouring arms meet at the platform's edge."""
    dims = mazes.RadialArmDims()
    limit = 2 * dims.center_radius * np.sin(np.pi / 16)
    narrow = mazes.RadialArmDims(
        n_arms=16, corridor_width=0.98 * limit, food_cup_width=5.0
    )
    floor = mazes.make_radial_arm_maze(narrow, include_track=False).env_2d
    outside = floor.arena.polygon.difference(
        shapely.Point(0.0, 0.0).buffer(dims.center_radius + 0.01)
    )
    assert len(outside.geoms) == 16
    with pytest.raises(ValueError, match="overlap"):
        mazes.RadialArmDims(n_arms=16, corridor_width=1.02 * limit)


def test_repeated_y_forks_just_shorter_than_the_limit_stay_apart():
    dims = mazes.RepeatedYDims()
    limit = 2 * dims.arm_length - dims.corridor_width / np.sqrt(3)
    maze = mazes.make_repeated_y_maze(
        mazes.RepeatedYDims(fork_length=0.98 * limit), include_track=False
    )
    # A chain of Y-junctions is a tree: forks that touched would close a loop.
    assert len(maze.env_2d.arena.polygon.interiors) == 0
    with pytest.raises(ValueError, match="forks"):
        mazes.RepeatedYDims(fork_length=1.02 * limit)
    # With two junctions no forks face each other.
    mazes.RepeatedYDims(n_junctions=2, fork_length=1.02 * limit)


def test_barnes_holes_just_apart_build():
    """33 holes of 9.5 cm overlap on the 49 cm ring; 32 do not."""
    maze = mazes.make_barnes_maze(mazes.BarnesDims(n_holes=32))
    holes = [
        maze.env_2d.regions[name].data
        for name in maze.env_2d.regions
        if "hole" in name and name != "start_box"
    ]
    assert not any(a.intersects(b) for a, b in pairwise([*holes, holes[0]]))
