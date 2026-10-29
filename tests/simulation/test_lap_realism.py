"""Lap trajectories look like a running animal: realistic speed, continuous path."""

import re
import warnings

import networkx as nx
import numpy as np
import pytest
import shapely
from environment_helpers import rotation_matrix
from shapely.geometry import LineString, Polygon, box

from neurospatial import Environment
from neurospatial.geometry import OpenArena
from neurospatial.layout import create_layout
from neurospatial.ops import AffineND
from neurospatial.simulation import (
    linear_track_session,
    simulate_session,
    simulate_trajectory_laps,
    tmaze_alternation_session,
)
from neurospatial.simulation.mazes import make_y_maze

DEFAULT_SPEED_CM_PER_S = 20.0
MAX_SPEED_CM_PER_S = 1.5 * DEFAULT_SPEED_CM_PER_S
SAMPLING_FREQUENCY = 500.0
PAUSE_SAMPLES = int(0.5 * SAMPLING_FREQUENCY)  # default pause_duration=0.5 s


@pytest.fixture(scope="module")
def box_env():
    """A 100 x 20 cm box with 2 cm bins."""
    return Environment.from_polygon(box(0, 0, 100, 20), bin_size=2, units="cm")


@pytest.fixture(scope="module")
def box_laps(box_env):
    """Three traversals of the box with the default speed."""
    return simulate_trajectory_laps(box_env, n_laps=3, seed=1, return_metadata=True)


def _pause_mask(metadata):
    """True for the pause samples that end every lap but the last."""
    lap_ids = metadata["lap_ids"]
    paused = np.zeros(lap_ids.size, dtype=bool)
    for lap in np.unique(lap_ids)[:-1]:
        paused[np.flatnonzero(lap_ids == lap)[-PAUSE_SAMPLES:]] = True
    return paused


def _moving_steps(metadata):
    """True for steps whose two samples are outside pauses and not a lap start."""
    paused = _pause_mask(metadata)
    lap_start = np.zeros(paused.size, dtype=bool)
    lap_start[metadata["lap_boundaries"]] = True
    return ~paused[:-1] & ~paused[1:] & ~lap_start[1:]


def _steps(positions):
    return np.linalg.norm(np.diff(positions, axis=0), axis=1)


def test_duration_is_plausible(box_laps):
    _positions, times, _metadata = box_laps
    assert 10.0 < times[-1] < 20.0
    assert len(times) < 20_000


def test_running_speed_is_realistic(box_laps):
    positions, times, metadata = box_laps
    moving = _moving_steps(metadata)
    speed = _steps(positions)[moving] / np.diff(times)[moving]
    assert 15.0 <= np.median(speed) <= 30.0
    assert speed.max() < MAX_SPEED_CM_PER_S


def test_path_is_smooth(box_laps):
    positions, times, metadata = box_laps
    moving_steps = _steps(positions)[_moving_steps(metadata)]
    assert np.all(moving_steps > 0)
    assert moving_steps.max() <= 1.01 * MAX_SPEED_CM_PER_S / SAMPLING_FREQUENCY
    assert len(np.unique(positions, axis=0)) > 0.9 * len(times)


def test_pauses_only_at_lap_ends(box_laps):
    positions, _times, metadata = box_laps
    zero_steps = np.flatnonzero(_steps(positions) == 0)
    # Step i joins samples i and i + 1. The pause before the lap starting at
    # boundary b fills samples [b - PAUSE_SAMPLES, b) at the previous lap's end,
    # and the next lap starts there, so its zero steps are [b - PAUSE_SAMPLES - 1, b).
    boundaries = metadata["lap_boundaries"][1:]
    at_lap_end = np.any(
        (zero_steps[:, None] >= boundaries - PAUSE_SAMPLES - 1)
        & (zero_steps[:, None] < boundaries),
        axis=1,
    )
    assert np.all(at_lap_end)
    # Every lap end but the last holds still for the whole pause.
    for boundary in boundaries:
        pause_steps = _steps(positions)[boundary - PAUSE_SAMPLES - 1 : boundary]
        assert np.all(pause_steps == 0)


def test_laps_stay_inside_environment(box_env, box_laps):
    positions, _times, _metadata = box_laps
    assert np.all(box_env.bin_at(positions) >= 0)


# Built-in sessions set the speed from duration and path length, not from the
# 20 cm/s default, so their bound is looser. The snapped path jumped a whole
# 2 cm bin within one 2 ms sample (1000 cm/s).
SESSION_MAX_SPEED_CM_PER_S = 2 * DEFAULT_SPEED_CM_PER_S


LAP_SESSIONS = {
    "linear_track": lambda: linear_track_session(
        duration=60, n_place_cells=3, n_laps=6, seed=42
    ),
    "tmaze": lambda: tmaze_alternation_session(
        duration=60, n_trials=4, n_place_cells=3, seed=1
    ),
}


@pytest.fixture(scope="module", params=list(LAP_SESSIONS), ids=list(LAP_SESSIONS))
def lap_session(request):
    """A built-in lap session."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Requested n_cells=3", UserWarning)
        return LAP_SESSIONS[request.param]()


def test_session_laps_stay_inside_environment(lap_session):
    assert np.all(lap_session.env.bin_at(lap_session.positions) >= 0)


def test_session_laps_move_smoothly(lap_session):
    sampling_frequency = lap_session.metadata["sampling_frequency"]
    largest_step = _steps(lap_session.positions).max()
    assert largest_step < SESSION_MAX_SPEED_CM_PER_S / sampling_frequency


@pytest.mark.parametrize("lap_session", ["linear_track"], indirect=True)
def test_session_paths_are_smooth(lap_session):
    session = lap_session
    pause_samples = int(
        session.metadata["pause_duration"] * session.metadata["sampling_frequency"]
    )
    zero = _steps(session.positions) == 0
    # A pause is a run of at least pause_samples zero steps; nothing else is zero.
    edges = np.flatnonzero(np.diff(np.concatenate([[0], zero.astype(int), [0]])))
    run_lengths = edges[1::2] - edges[::2]
    assert np.all(run_lengths >= pause_samples)
    assert len(run_lengths) == 6 - 1


def test_default_speed_follows_environment_units(box_laps):
    positions_cm, times_cm, _metadata = box_laps
    env_m = Environment.from_polygon(box(0, 0, 1, 0.2), bin_size=0.02, units="m")
    positions_m, times_m = simulate_trajectory_laps(env_m, n_laps=3, seed=1)
    np.testing.assert_allclose(times_m, times_cm, rtol=1e-12)
    np.testing.assert_allclose(positions_m, positions_cm / 100, rtol=1e-12)


def test_default_speed_needs_metric_units():
    env = Environment.from_polygon(box(0, 0, 100, 20), bin_size=2)
    with pytest.raises(ValueError, match="speed_mean") as error:
        simulate_trajectory_laps(env, n_laps=3, seed=1)
    assert "not a metric length unit" in str(error.value)
    positions, _times = simulate_trajectory_laps(env, n_laps=3, seed=1, speed_mean=20.0)
    assert np.all(env.bin_at(positions) >= 0)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("speed_mean", np.nan),
        ("speed_mean", 0.0),
        ("speed_std", -1.0),
        ("speed_std", np.inf),
        ("sampling_frequency", np.inf),
        ("sampling_frequency", 0.0),
        ("pause_duration", np.nan),
        ("pause_duration", -0.5),
    ],
)
def test_invalid_lap_timing_is_refused(box_env, option, value):
    with pytest.raises(ValueError, match=f"{option} must be finite"):
        simulate_trajectory_laps(box_env, n_laps=2, **{option: value})


@pytest.mark.parametrize("n_laps", [2.5, True, 0, -1, np.float64(2.0)])
def test_n_laps_must_be_a_positive_integer(box_env, n_laps):
    with pytest.raises(ValueError, match="n_laps must be a positive integer"):
        simulate_trajectory_laps(box_env, n_laps=n_laps)


def test_custom_path_must_be_chain_of_neighbors(box_env):
    assert not box_env.connectivity.has_edge(0, 2)
    with pytest.raises(ValueError, match="neighboring"):
        simulate_trajectory_laps(box_env, n_laps=2, outbound_path=[0, 2])


def test_custom_path_may_repeat_a_bin(box_env):
    """A repeated bin is a zero-length step: the path is still connected."""
    assert box_env.connectivity.has_edge(0, 1)
    positions, _times = simulate_trajectory_laps(
        box_env, n_laps=2, outbound_path=[0, 0, 1], seed=1
    )
    np.testing.assert_allclose(positions[0], box_env.bin_centers[0])
    np.testing.assert_allclose(positions[-1], box_env.bin_centers[0])


@pytest.fixture(scope="module")
def two_rooms_env():
    """Two 10 x 10 cm rooms 40 cm apart, with no bins between them."""
    grid = np.stack(np.meshgrid(np.arange(1, 10, 2.0), np.arange(1, 10, 2.0)), -1)
    room = grid.reshape(-1, 2)
    samples = np.concatenate([room, room + np.array([50.0, 0.0])])
    return Environment.from_samples(samples, bin_size=2.0, units="cm")


def test_path_errors_name_the_function_called(two_rooms_env, box_env):
    with pytest.raises(ValueError, match=r"simulate_trajectory_laps\(\) found no"):
        simulate_trajectory_laps(two_rooms_env, n_laps=2)
    with pytest.raises(ValueError, match=r"simulate_trajectory_laps\(\): outbound"):
        simulate_trajectory_laps(box_env, n_laps=2, outbound_path=[0, 2])
    session_call = re.escape('simulate_session(trajectory_method="laps")')
    with pytest.raises(ValueError, match=session_call + " found no"):
        simulate_session(
            two_rooms_env,
            duration=10.0,
            n_cells=1,
            trajectory_method="laps",
            show_progress=False,
            seed=1,
        )
    with pytest.raises(ValueError, match=session_call + ": outbound"):
        simulate_session(
            box_env,
            duration=10.0,
            n_cells=1,
            trajectory_method="laps",
            outbound_path=[0, 2],
            show_progress=False,
            seed=1,
        )


# Laps follow the environment's geometry: a step between two neighboring bins
# never cuts across space outside the environment or through a wall.


@pytest.fixture(scope="module")
def l_track():
    """A 2 cm wide L-shaped track: two 10 cm bins that meet at a right angle."""
    return Environment.linear_track(
        node_positions=[(-5, 0), (5, 0), (5, 10)],
        bin_size=10,
        track_width=2,
        units="cm",
    )


def test_track_lap_turns_at_the_junction(l_track):
    positions, _times = simulate_trajectory_laps(l_track, n_laps=1, seed=1)
    assert np.all(l_track.bin_at(positions) >= 0)
    # Every sample lies on the centerline: projecting onto it changes nothing.
    np.testing.assert_allclose(
        l_track.linear_to_nd(l_track.to_linear(positions)), positions, atol=1e-9
    )
    # The outbound lap runs steadily along the track, through the corner.
    assert np.all(np.diff(l_track.to_linear(positions)) > 0)


def test_track_lap_time_counts_toward_occupancy(l_track):
    """Every second of the lap lands in a place-field bin."""
    positions, times = simulate_trajectory_laps(l_track, n_laps=1, seed=1)
    np.testing.assert_allclose(
        l_track.occupancy(times, positions).sum(), times[-1] - times[0]
    )


def _t_maze_track():
    """A T-maze whose right arm is not next to the stem in linear position."""
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(0.0, 40.0))
    graph.add_node(2, pos=(-30.0, 40.0))
    graph.add_node(3, pos=(30.0, 40.0))
    graph.add_edges_from([(0, 1), (1, 2), (1, 3)])
    return Environment.from_graph(
        graph,
        edge_order=[(0, 1), (1, 2), (1, 3)],
        edge_spacing=10.0,
        bin_size=7.0,
        track_width=4.0,
        units="cm",
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: Environment.linear_track(
            node_positions=[(0, 0), (50, 0), (50, 50), (0, 60)],
            bin_size=7.0,
            track_width=3.0,
            units="cm",
        ),
        _t_maze_track,
        lambda: Environment.linear_track(
            node_positions=[(-5, 0), (5, 0), (5, 10)],
            bin_size=10,
            track_width=2,
            units="cm",
        ).apply_transform(AffineND(rotation_matrix(0.4, offset=(100.0, 50.0)))),
    ],
    ids=["zigzag", "t_maze", "rotated"],
)
def test_track_laps_stay_on_the_centerline(build):
    env = build()
    walk = _walk_over_every_neighbor_pair(env)
    positions, _times = simulate_trajectory_laps(
        env, n_laps=2, outbound_path=walk, seed=1
    )
    assert np.all(env.bin_at(positions) >= 0)
    np.testing.assert_allclose(
        env.linear_to_nd(env.to_linear(positions)), positions, atol=1e-9
    )


def _walk_over_every_neighbor_pair(env):
    """A chain of neighboring bins that steps across every open pair."""
    circuit = list(nx.eulerian_circuit(nx.eulerize(env.connectivity)))
    return [int(u) for u, _v in circuit] + [int(circuit[-1][1])]


def _corner_walk_grid():
    """Bins visited by a staircase walk: neighbors that meet only at corners."""
    steps = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 0.0], [6.0, 2.0], [8.0, 4.0]])
    return Environment.from_samples(steps + 1.0, bin_size=2.0, units="cm")


def _checkerboard():
    mask = (np.add.outer(np.arange(4), np.arange(4)) % 2) == 0
    edges = np.arange(5.0) * 2
    return Environment.from_grid_mask(mask, grid_edges=(edges, edges), units="cm")


L_SHAPE = Polygon([(0, 0), (40, 0), (40, 10), (10, 10), (10, 40), (0, 40)])


def _star(n_points=5, outer=40.0, inner=12.0):
    angle = np.linspace(0, 2 * np.pi, 2 * n_points, endpoint=False)
    radius = np.where(np.arange(2 * n_points) % 2 == 0, outer, inner)
    return Polygon(np.c_[radius * np.cos(angle), radius * np.sin(angle)])


def _l_mask_grid():
    mask = np.ones((6, 6), bool)
    mask[3:, 3:] = False
    edges = np.arange(7.0) * 2
    return Environment.from_grid_mask(mask, grid_edges=(edges, edges), units="cm")


def _walled_box(wall, **options):
    return Environment.from_grid_mask(
        np.ones((4, 4), bool),
        grid_edges=(np.arange(5.0) * 5, np.arange(5.0) * 5),
        arena=OpenArena(box(0, 0, 20, 20), walls=(wall,)),
        units="cm",
        **options,
    )


def _samples_in(polygon, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    low_x, low_y, high_x, high_y = polygon.bounds
    points = rng.uniform((low_x, low_y), (high_x, high_y), (n, 2))
    return points[shapely.contains_xy(polygon, points[:, 0], points[:, 1])]


LAYOUTS = {
    "grid_concave_corner": _l_mask_grid,
    "grid_rotated": lambda: Environment.from_polygon(
        L_SHAPE, bin_size=3, units="cm"
    ).apply_transform(AffineND(rotation_matrix(0.5, offset=(3.0, -2.0)))),
    "grid_rotated_corners_only": lambda: _corner_walk_grid().apply_transform(
        AffineND(rotation_matrix(0.3))
    ),
    "checkerboard_rotated_far": lambda: _checkerboard().apply_transform(
        AffineND(rotation_matrix(0.7, offset=(1e4, -3e3)))
    ),
    "hexagonal": lambda: Environment.from_samples(
        _samples_in(L_SHAPE), bin_size=3, layout="hexagonal", units="cm"
    ),
    "triangular_mesh": lambda: Environment.from_layout(
        create_layout(
            "TriangularMesh",
            boundary_polygon=Polygon(
                [(0, 0), (30, 0), (30, 30), (0, 30)],
                [[(10, 10), (20, 10), (20, 20), (10, 20)]],
            ),
            point_spacing=4.0,
        )
    ).with_metadata(units="cm"),
    "polygon_arena": lambda: Environment.from_polygon(_star(), bin_size=4, units="cm"),
    "clipped_cells": lambda: Environment.from_polygon(
        _star(), bin_size=5, units="cm"
    ).clip(_star()),
    "wall_split": lambda: _walled_box(
        LineString([(1.5, 1.0), (18.5, 14.5)]), wall_fit="split"
    ),
    "wall_split_bent": lambda: _walled_box(
        LineString([(1.0, 2.5), (7.5, 12.5), (1.0, 18.0)]), wall_fit="split"
    ),
    "combined": lambda: Environment.combine(
        {
            "box": Environment.from_polygon(box(0, 0, 20, 20), bin_size=2, units="cm"),
            "corridor": Environment.from_polygon(
                box(20, 8, 40, 12), bin_size=2, units="cm"
            ),
        }
    ),
    "y_maze": lambda: make_y_maze().env_2d,
}


# The layouts whose arenas have walls.
WALLED_LAYOUTS = ["wall_split", "wall_split_bent"]


@pytest.fixture(scope="module", params=list(LAYOUTS), ids=list(LAYOUTS))
def layout_laps(request):
    """An environment and laps that step across every one of its neighbor pairs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        env = LAYOUTS[request.param]()
    walk = _walk_over_every_neighbor_pair(env)
    positions, _times = simulate_trajectory_laps(
        env, n_laps=2, outbound_path=walk, seed=1, sampling_frequency=200.0
    )
    return env, positions


def test_laps_stay_inside_every_layout(layout_laps):
    env, positions = layout_laps
    assert np.all(env.bin_at(positions) >= 0)


@pytest.mark.parametrize("layout_laps", WALLED_LAYOUTS, indirect=True)
def test_laps_do_not_cross_walls(layout_laps):
    env, positions = layout_laps
    walls = env.arena.walls
    assert walls
    steps = shapely.linestrings(np.stack([positions[:-1], positions[1:]], axis=1))
    longest_step = shapely.length(steps).max()
    for wall in walls:
        # The straight line between two samples may clip the very end of a
        # wall the lap runs around, but never cuts through the wall itself.
        crossings = shapely.intersection(steps[shapely.crosses(steps, wall)], wall)
        ends = shapely.multipoints(shapely.get_coordinates(wall)[[0, -1]])
        assert np.all(shapely.distance(crossings, ends) <= longest_step)


@pytest.fixture(scope="module")
def two_rooms_bridged():
    """Two rooms 10 cm apart, joined by bridges with no bins between them."""
    return Environment.combine(
        {
            "left": Environment.from_polygon(box(0, 0, 20, 20), bin_size=2, units="cm"),
            "right": Environment.from_polygon(
                box(30, 0, 50, 20), bin_size=2, units="cm"
            ),
        }
    )


def test_bridge_across_a_gap_is_refused(two_rooms_bridged):
    """No lap across the gap can stay inside the environment."""
    with pytest.raises(ValueError, match="leave the environment"):
        simulate_trajectory_laps(two_rooms_bridged, n_laps=2, seed=1)


def test_bridge_refusal_names_the_path_to_fix(two_rooms_bridged):
    env = two_rooms_bridged
    pairs = env.edge_pairs
    gap = np.linalg.norm(np.diff(env.bin_centers[pairs], axis=1)[:, 0], axis=1)
    left, right = (int(b) for b in pairs[np.argmax(gap)])
    with pytest.raises(ValueError, match=r"inbound_path steps") as error:
        simulate_trajectory_laps(
            env, n_laps=2, outbound_path=[left], inbound_path=[right, left]
        )
    assert "pass inbound_path= as a chain" in str(error.value)
