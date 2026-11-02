"""Every goal measure reads ``goal`` the same way, per trial when given trials.

A goal is a region name, a point, or one per trial. Each trial is measured
from its own samples, as if it were the whole recording. A region is
approached at its nearest bin.
"""

import numpy as np
import pandas as pd
import pytest
from shapely import box

from neurospatial import Environment, Intervals, Regions
from neurospatial.arena import OpenArena
from neurospatial.behavior import (
    angular_efficiency,
    approach_rate,
    compute_goal_directed_metrics,
    compute_path_efficiency,
    distance_to_goal,
    goal_bias,
    graph_turn_sequence,
    instantaneous_goal_alignment,
)

FS = 30.0  # Hz


@pytest.fixture(scope="module")
def arena():
    """A walled 100 x 100 cm arena with two corner wells and an east arm."""
    xs, ys = np.meshgrid(np.linspace(0, 100, 101), np.linspace(0, 100, 101))
    env = Environment.from_samples(
        np.c_[xs.ravel(), ys.ravel()],
        bin_size=2.0,
        units="cm",
        arena=OpenArena(box(0, 0, 100, 100)),
    )
    regions = Regions()
    regions.add("west_well", point=(10.0, 50.0), radius=5.0)
    regions.add("east_well", point=(90.0, 50.0), radius=5.0)
    regions.add("north_arm", polygon=box(60, 80, 100, 90))
    return env.with_regions(regions)


@pytest.fixture(scope="module")
def session():
    """Three 4 s runs between the wells (east, west, east) with 1 s pauses."""
    leg = np.linspace(0, 1, int(4 * FS), endpoint=False)
    pause = np.zeros(int(1 * FS))
    wiggle = 3.0 * np.sin(np.linspace(0, 6 * np.pi, leg.size))
    east = np.c_[15 + 70 * leg, 50 + wiggle]
    west = np.c_[85 - 70 * leg, 50 - wiggle]
    positions = np.vstack(
        [
            east,
            np.c_[85 + pause, 50 + pause],
            west,
            np.c_[15 + pause, 50 + pause],
            east,
        ]
    )
    times = np.arange(len(positions)) / FS
    n_leg = leg.size
    starts = np.array([0, n_leg + pause.size, 2 * (n_leg + pause.size)]) / FS
    trials = Intervals.from_arrays(
        starts,
        starts + n_leg / FS,
        end_region=np.array(["east_well", "west_well", "east_well"], dtype=object),
    )
    return times, positions, trials


def _trial_samples(times, trials, row):
    return (times >= trials.start[row]) & (times < trials.stop[row])


SUMMARIES = [
    (angular_efficiency, {}),
    (goal_bias, {"min_speed": 2.0}),
]


@pytest.mark.parametrize(("function", "options"), SUMMARIES)
def test_summary_per_trial_equals_each_trial_alone(arena, session, function, options):
    times, positions, trials = session
    per_trial = function(
        arena, times, positions, trials, goal=trials.end_region, **options
    )
    assert isinstance(per_trial, pd.Series)
    assert per_trial.index.name == "interval_id"
    assert per_trial.index.tolist() == trials.interval_ids.tolist()
    for row, goal in enumerate(trials.end_region):
        inside = _trial_samples(times, trials, row)
        alone = function(arena, times[inside], positions[inside], goal=goal, **options)
        assert per_trial.iloc[row] == pytest.approx(alone, rel=1e-12)
    # Each run heads straight at its own well.
    assert (per_trial > 0.8).all()


@pytest.mark.parametrize(
    ("function", "columns"),
    [
        (
            compute_path_efficiency,
            ("traveled_length", "shortest_length", "efficiency", "angular_efficiency"),
        ),
        (
            compute_goal_directed_metrics,
            (
                "goal_bias",
                "mean_approach_rate",
                "goal_distance_at_start",
                "min_distance_to_goal",
                "time_to_reach_goal",
            ),
        ),
    ],
)
def test_result_table_per_trial_equals_each_trial_alone(
    arena, session, function, columns
):
    times, positions, trials = session
    table = function(arena, times, positions, trials, goal=trials.end_region)
    assert isinstance(table, pd.DataFrame)
    assert table.index.name == "interval_id"
    assert table["goal"].tolist() == list(trials.end_region)
    for row, goal in enumerate(trials.end_region):
        inside = _trial_samples(times, trials, row)
        alone = function(arena, times[inside], positions[inside], goal=goal)
        for column in columns:
            assert table[column].iloc[row] == pytest.approx(
                getattr(alone, column), rel=1e-12, nan_ok=True
            ), column


@pytest.mark.parametrize(
    ("function", "options"),
    [(approach_rate, {}), (instantaneous_goal_alignment, {"min_speed": 2.0})],
)
def test_per_sample_measures_cut_at_trial_edges(arena, session, function, options):
    times, positions, trials = session
    values = function(
        arena, times, positions, trials, goal=trials.end_region, **options
    )
    in_any = np.zeros(times.size, dtype=bool)
    for row, goal in enumerate(trials.end_region):
        inside = _trial_samples(times, trials, row)
        in_any |= inside
        alone = function(arena, times[inside], positions[inside], goal=goal, **options)
        np.testing.assert_allclose(values[inside], alone, rtol=1e-12, equal_nan=True)
        # Nothing reaches back across the trial's start.
        assert np.isnan(values[np.flatnonzero(inside)[0]]) or function is (
            instantaneous_goal_alignment
        )
    assert np.isnan(values[~in_any]).all()


def test_unknown_goal_gives_nan_for_that_trial(arena, session):
    times, positions, trials = session
    goals = list(trials.end_region)
    goals[1] = None
    bias = goal_bias(arena, times, positions, trials, goal=goals, min_speed=2.0)
    assert np.isnan(bias.iloc[1])
    assert np.isfinite(bias.iloc[[0, 2]]).all()
    table = compute_path_efficiency(arena, times, positions, trials, goal=goals)
    assert np.isnan(table["efficiency"].iloc[1])
    assert table["goal"].iloc[1] is None


def test_overlapping_trials_are_refused_only_for_per_sample_results(arena, session):
    times, positions, _ = session
    overlapping = Intervals.from_arrays([0.0, 2.0], [4.0, 6.0])
    with pytest.raises(ValueError, match="overlap"):
        approach_rate(arena, times, positions, overlapping, goal="east_well")
    bias = goal_bias(
        arena, times, positions, overlapping, goal="east_well", min_speed=2.0
    )
    assert len(bias) == 2


def test_round_well_is_approached_at_its_center(arena):
    """For a round well the nearest-bin direction points at the well's center."""
    times = np.arange(60) / FS
    # A run heading straight at the east well's center, from the south-west.
    start, center = np.array([40.0, 20.0]), np.array([90.0, 50.0])
    path = start + np.linspace(0, 0.8, 60)[:, None] * (center - start)
    region = instantaneous_goal_alignment(
        arena, times, path, goal="east_well", min_speed=2.0
    )
    point = instantaneous_goal_alignment(arena, times, path, goal=center, min_speed=2.0)
    moving = np.isfinite(point)
    np.testing.assert_allclose(point[moving], 1.0, atol=1e-9)  # straight at it
    # The nearest goal bin's center lies within half a bin diagonal of the
    # line to the center, so the goal direction turns by at most
    # asin(half diagonal / distance to the well's edge).
    half_diagonal = np.sqrt(2.0) * arena.bin_spacing / 2
    to_edge = np.linalg.norm(path - center, axis=1) - 5.0
    bound = np.cos(np.arcsin(np.minimum(half_diagonal / to_edge, 1.0)))
    assert np.all(region[moving] >= bound[moving] - 1e-12)
    assert np.nanmin(region) > 0.98


def test_arm_is_approached_at_its_nearest_bin_not_its_centroid(arena):
    """Running straight at the nearest part of an arm is fully aligned."""
    times = np.arange(60) / FS
    # North along the bin column at x = 61 cm, below the arm's west end
    # (the arm spans x = 60..100, y = 80..90).
    path = np.c_[np.full(60, 61.0), np.linspace(20, 60, 60)]
    to_arm = instantaneous_goal_alignment(
        arena, times, path, goal="north_arm", min_speed=2.0
    )
    # The arm's centroid, (80, 85), lies to the north-east of the path; it is
    # closest to straight ahead at the start of the run, (61, 20), where it
    # lies at (+19, +65), so aiming at it scores at most 65 / hypot(19, 65).
    to_centroid = instantaneous_goal_alignment(
        arena, times, path, goal=[80.0, 85.0], min_speed=2.0
    )
    np.testing.assert_allclose(to_arm[np.isfinite(to_arm)], 1.0, atol=1e-9)
    assert np.nanmax(to_centroid) <= 65 / np.hypot(19, 65) + 1e-6


def test_region_arrival_is_entering_the_region(arena):
    times = np.arange(90) / FS
    path = np.c_[np.linspace(40, 95, 90), np.full(90, 50.0)]
    metrics = compute_goal_directed_metrics(arena, times, path, goal="east_well")
    # The well's bins have centers within 5 cm of (90, 50); x reaches 85 cm
    # (the first bin center inside, at 85 or 86 cm) by sample index ~73.
    entered = np.flatnonzero(
        np.isin(arena.bin_at(path), arena.bins_in_region("east_well"))
    )
    assert metrics.time_to_reach_goal == pytest.approx(times[entered[0]] - times[0])
    assert metrics.goal == "east_well"
    with pytest.raises(ValueError, match="region goal"):
        compute_goal_directed_metrics(
            arena, times, path, goal="east_well", goal_radius=5.0
        )


def test_point_goal_is_its_exact_coordinates(arena):
    positions = np.array([[10.0, 10.0], [30.0, 40.0]])
    goal = np.array([13.3, 14.4])  # off any bin center
    distances = distance_to_goal(
        arena, np.arange(2.0), positions, goal=goal, metric="euclidean"
    )
    np.testing.assert_allclose(distances, np.linalg.norm(positions - goal, axis=1))


def test_costs_need_the_geodesic_metric(arena):
    with pytest.raises(ValueError, match="metric='geodesic'"):
        distance_to_goal(
            arena,
            np.arange(2.0),
            np.array([[10.0, 10.0], [20.0, 20.0]]),
            goal="east_well",
            metric="euclidean",
            cost_map=np.zeros(arena.n_bins),
        )


def test_default_metric_is_geodesic(arena, session):
    times, positions, _ = session
    default = approach_rate(arena, times, positions, goal="east_well")
    geodesic = approach_rate(
        arena, times, positions, goal="east_well", metric="geodesic"
    )
    np.testing.assert_array_equal(default, geodesic)


def test_positions_outside_the_environment_are_missing(arena):
    times = np.arange(4.0)
    positions = np.array([[10.0, 50.0], [120.0, 50.0], [30.0, 50.0], [40.0, 50.0]])
    distances = distance_to_goal(
        arena, times, positions, goal="east_well", metric="euclidean"
    )
    assert np.isnan(distances[1])
    assert np.isfinite(distances[[0, 2, 3]]).all()


def test_turn_sequence_per_trial(arena):
    north = np.c_[np.full(20, 40.0), np.linspace(5, 40, 20)]
    west = np.c_[np.linspace(38, 5, 20), np.full(20, 40.0)]
    east = np.c_[np.linspace(42, 75, 20), np.full(20, 40.0)]
    positions = np.r_[north, west, north, east]
    times = np.arange(len(positions)) / 10
    trials = Intervals.from_arrays([0.0, 4.0], [4.0, 8.0])
    turns = graph_turn_sequence(arena, times, positions, trials)
    assert turns.tolist() == ["left", "right"]
    assert turns.index.name == "interval_id"
