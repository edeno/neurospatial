"""Boundary kinds: walls block movement and gaze, clear walls and drops only movement.

An opaque wall stops the animal and its gaze. A transparent wall and the
drop edge of an elevated platform stop the animal but not its gaze. Analyses
of where the animal is relative to the edges it cannot cross (border scores,
boundary vectors) use every physical boundary; analyses of what it sees use
only opaque walls. An outline inferred from positions is no boundary at all.
"""

import pickle

import numpy as np
import pytest
from shapely.geometry import LineString, box

from neurospatial import Environment
from neurospatial.arena import MaskArena, OpenArena
from neurospatial.encoding import border_score, compute_boundary_vector_rate
from neurospatial.geometry import boundary_geometry, line_of_sight_clear
from neurospatial.ops import compute_boundary_distances
from neurospatial.ops.visibility import compute_viewed_location
from neurospatial.simulation import (
    BorderCellModel,
    BoundaryVectorCellModel,
    generate_poisson_spikes,
)

SQUARE = box(0, 0, 100, 100)


def _env(arena, bin_size=5.0):
    return Environment.from_polygon(SQUARE, bin_size=bin_size, units="cm", arena=arena)


@pytest.fixture(scope="module")
def platform_with_walls():
    """An elevated platform holding a clear wall at x=30 and an opaque one at x=70."""
    return _env(
        OpenArena(
            SQUARE,
            outline="drop",
            walls={
                "glass": LineString([(30, 20), (30, 80)]),
                "board": LineString([(70, 20), (70, 80)]),
            },
            wall_kinds={"glass": "transparent_wall"},
        ),
        bin_size=2.0,
    )


def test_gaze_passes_clear_walls_and_drops_and_stops_at_opaque_walls(
    platform_with_walls,
):
    env = platform_with_walls
    observer = np.array([[10.0, 50.0], [10.0, 50.0]])
    east, west = 0.0, np.pi
    viewed = compute_viewed_location(
        observer, np.array([east, west]), method="ray_cast", env=env
    )
    # East: through the glass at x=30 onto the board at x=70.
    np.testing.assert_allclose(viewed[0], [70.0, 50.0])
    # West: past the drop edge at x=0, onto nothing in the environment.
    assert np.isnan(viewed[1]).all()
    target_behind_glass = np.array([50.0, 50.0])
    target_behind_board = np.array([90.0, 50.0])
    assert line_of_sight_clear(env.binning, env.arena, observer[0], target_behind_glass)
    assert not line_of_sight_clear(
        env.binning, env.arena, observer[0], target_behind_board
    )


def test_movement_stops_at_every_boundary(platform_with_walls):
    distances, directions = compute_boundary_distances(
        platform_with_walls,
        np.array([[10.0, 50.0]]),
        direction_frame="allocentric",
        n_directions=4,
    )
    reach = dict(zip(np.round(np.degrees(directions)), distances[0], strict=True))
    # Toward +x (northeast and southeast rays) the glass at x=30 is the first
    # boundary; toward -x the drop edge at x=0 is.
    assert reach[45.0] == pytest.approx(20 * np.sqrt(2))
    assert reach[135.0] == pytest.approx(10 * np.sqrt(2))


def test_a_clear_stretch_of_a_walled_outline_lets_the_gaze_out():
    """A named wall on the outline gives that stretch its own kind."""
    env = _env(
        OpenArena(
            SQUARE,
            walls={"window": LineString([(100, 20), (100, 80)])},
            wall_kinds={"window": "transparent_wall"},
        )
    )
    viewed = compute_viewed_location(
        np.array([[50.0, 50.0], [50.0, 90.0]]),
        np.zeros(2),
        method="ray_cast",
        env=env,
    )
    assert np.isnan(viewed[0]).all()
    np.testing.assert_allclose(viewed[1], [100.0, 90.0])
    gaze = boundary_geometry(env.binning, env.arena, "physical", blocking="gaze")
    movement = boundary_geometry(
        env.binning, env.arena, "physical", blocking="movement"
    )
    assert movement.length == pytest.approx(400.0)
    assert gaze.length == pytest.approx(340.0)


def test_border_score_of_an_elevated_platform_equals_the_walled_box():
    walled = _env(None)
    elevated = _env(OpenArena(SQUARE, outline="drop"))
    x = walled.bin_centers[:, 0]
    for rates in [np.where(x < 10, 10.0, 0.2), np.exp(-((x - 50) ** 2) / 200)]:
        assert border_score(elevated, rates) == border_score(walled, rates)


def test_boundary_vector_map_on_an_elevated_platform_peaks_at_the_drop_edge():
    """A cell tuned to an edge 10 cm east keeps that tuning on a drop edge."""
    elevated = _env(OpenArena(SQUARE, outline="drop"), bin_size=2.0)
    walled = _env(None, bin_size=2.0)
    cell = BoundaryVectorCellModel(
        elevated,
        preferred_distance=10.0,
        preferred_direction=0.0,
        distance_tolerance=4.0,
        direction_tolerance=np.pi / 6,
        max_rate=20.0,
        baseline_rate=0.5,
    )
    times = np.arange(0.0, 600.0, 1 / 30)
    positions = np.random.default_rng(1).uniform(1.0, 99.0, (len(times), 2))
    rates = cell.firing_rate(times, positions)
    assert np.isfinite(rates).all()
    spike_times = generate_poisson_spikes(rates, times, seed=2)

    def rate_map(env):
        return compute_boundary_vector_rate(
            env,
            spike_times,
            times,
            positions,
            direction_frame="allocentric",
            distance_range=(0.0, 40.0),
            distance_bin_size=2.5,
            max_gap=1.0,
        )

    on_drop = rate_map(elevated)
    supported = np.isfinite(on_drop.firing_rate)
    assert supported.mean() > 0.9
    assert on_drop.preferred_distance() == pytest.approx(10.0, abs=2.5)
    assert np.cos(on_drop.preferred_direction()) > np.cos(np.pi / 6)
    # Same geometry, same map: the drop edge is measured exactly like a wall.
    np.testing.assert_array_equal(
        on_drop.firing_rate, rate_map(walled).firing_rate, strict=True
    )


def test_an_inferred_outline_is_no_physical_boundary():
    positions = np.random.default_rng(0).uniform(0, 100, (5000, 2))
    env = Environment.from_samples(positions, bin_size=5.0, units="cm")
    assert env.arena.outline is None
    rates = np.ones(env.n_bins)
    with pytest.raises(NotImplementedError, match="Physical boundaries are unknown"):
        border_score(env, rates)
    with pytest.raises(NotImplementedError, match="outline='drop'"):
        compute_boundary_vector_rate(
            env,
            np.array([1.0]),
            np.arange(0.0, 10.0, 0.1),
            positions[:100],
            direction_frame="allocentric",
            max_gap=1.0,
        )
    with pytest.raises(NotImplementedError, match="Physical boundaries are unknown"):
        BorderCellModel(env, distance_tolerance=5.0, max_rate=10.0, baseline_rate=0.1)
    # A mask declared as a drop-edged platform is a physical boundary again.
    mask_env = Environment.from_grid_mask(
        np.ones((4, 4), dtype=bool),
        grid_edges=(np.arange(0.0, 50.0, 10.0),) * 2,
        arena=MaskArena(
            (np.arange(0.0, 50.0, 10.0),) * 2, np.ones((4, 4), bool), outline="drop"
        ),
    )
    assert np.isfinite(border_score(mask_env, np.ones(mask_env.n_bins)))


def test_pickle_keeps_boundary_kinds(platform_with_walls):
    restored = pickle.loads(pickle.dumps(platform_with_walls))
    assert restored.arena.outline == "drop"
    assert restored.arena.wall_names == ("glass", "board")
    assert restored.arena.wall_kinds == ("transparent_wall", "wall")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"outline": "glass"}, "not a boundary kind"),
        (
            {"walls": {"a": LineString([(1, 1), (2, 2)])}, "wall_kinds": "ledge"},
            "not a boundary kind",
        ),
        (
            {"walls": {"a": LineString([(1, 1), (2, 2)])}, "wall_kinds": {"b": "drop"}},
            "do not exist",
        ),
        (
            {
                "walls": (LineString([(1, 1), (2, 2)]), LineString([(3, 3), (4, 4)])),
                "wall_names": ("a", "a"),
                "wall_kinds": ("wall", "drop"),
            },
            "different kinds",
        ),
        (
            {
                "walls": {"a": LineString([(1, 1), (2, 2)])},
                "wall_kinds": ("wall", "drop"),
            },
            "2 kinds for 1 walls",
        ),
    ],
)
def test_invalid_boundary_kinds_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        OpenArena(SQUARE, **kwargs)
