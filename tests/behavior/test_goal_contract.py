"""Every goal measure reads ``goal`` the same way, per trial when given trials.

A goal is a region name, a point, or one per trial. Each trial is measured
from its own samples, as if it were the whole recording. A region is
reached at its edge: the distance to it is 0 inside and the distance to its
polygon outside.
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
    table = function(
        arena, times, positions, trials, goal=trials.end_region
    ).summary_table()
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
    result = compute_path_efficiency(arena, times, positions, trials, goal=goals)
    assert np.isnan(result.efficiency[1])
    assert result.goal[1] is None
    # Tracking, not the goal, decides fully_observed.
    alone = compute_path_efficiency(
        arena, times, positions, trials, goal=list(trials.end_region)
    )
    assert result.fully_observed[1] == alone.fully_observed[1]
    table = result.summary_table()
    assert np.isnan(table["efficiency"].iloc[1])
    assert table["goal"].isna().tolist() == [False, True, False]


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
    """A round well's nearest edge point lies on the line to its center."""
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
    # The well is a polygon disk: its nearest point is on the line to its
    # center, up to the polygon's facets (well under a degree here).
    assert np.nanmin(region[moving]) > 0.9999


def test_arm_is_approached_at_its_nearest_edge_not_its_centroid(arena):
    """Running straight at the nearest part of an arm is fully aligned."""
    times = np.arange(60) / FS
    # North along x = 61 cm, below the arm's west end (the arm spans
    # x = 60..100, y = 80..90), whose nearest edge is straight ahead.
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
    # Arrival is the first sample inside the well's polygon (its edge at
    # x = 85 cm on this line), not the first in one of its bins.
    import shapely

    entered = np.flatnonzero(
        shapely.covers(arena.regions["east_well"].data, shapely.points(path))
    )
    assert metrics.time_to_reach_goal == pytest.approx(times[entered[0]] - times[0])
    assert metrics.goal == "east_well"
    with pytest.raises(ValueError, match="region goal"):
        compute_goal_directed_metrics(
            arena, times, path, goal="east_well", goal_radius=5.0
        )


def test_region_goal_distance_is_zero_inside_and_polygon_distance_outside(arena):
    """Checked against shapely's distance to the region polygon."""
    import shapely

    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 100, (500, 2))
    positions[:20] = rng.uniform([87.0, 47.0], [93.0, 53.0], (20, 2))  # in the well
    times = np.arange(len(positions), dtype=float)
    polygon = arena.regions["east_well"].data
    expected = shapely.distance(polygon, shapely.points(positions))
    assert (expected[:20] == 0).sum() > 10
    for metric in ("euclidean", "geodesic"):
        distances = distance_to_goal(
            arena, times, positions, goal="east_well", metric=metric
        )
        # The arena is convex: the shortest path inside it is the straight line.
        np.testing.assert_allclose(distances, expected, rtol=1e-9, atol=1e-9)
        assert np.all(distances[expected == 0] == 0.0)


def test_region_goal_behind_a_corner_goes_round_it():
    """Around the inner corner of an L, checked against the exact path.

    The shortest path to a well in the other arm is the straight line when it
    stays inside the L, else the line to the inner corner and on to the
    well's edge. The geodesic engine routes through bin centers, so it may
    exceed the exact path by up to about one bin per corner, never fall short
    of it; its first leg heads for the corner.
    """
    import shapely
    from shapely.geometry import LineString, Point

    from neurospatial.behavior._goals import resolve_goal
    from neurospatial.ops import _environment_paths

    arena = box(0, 0, 100, 100).difference(box(50, 50, 100, 100))
    env = Environment.from_polygon(arena, bin_size=2.0, units="cm")
    regions = Regions()
    regions.add("well", point=(25.0, 85.0), radius=5.0)
    env = env.with_regions(regions)
    well = env.regions["well"].data
    corner = np.array([50.0, 50.0])
    positions = np.array([[90.0, 25.0], [80.0, 10.0], [95.0, 45.0], [60.0, 30.0]])

    exact, toward = [], []
    for p in positions:
        nearest = np.asarray(shapely.shortest_line(Point(p), well).coords[1])
        if arena.covers(LineString([p, nearest])):
            exact.append(np.linalg.norm(nearest - p))
            toward.append(nearest - p)
        else:
            exact.append(np.linalg.norm(corner - p) + well.distance(Point(corner)))
            toward.append(corner - p)
    exact, toward = np.array(exact), np.array(toward)
    hidden = [0, 1, 2]  # the last position sees the well past the corner
    assert not any(
        arena.covers(LineString([positions[i], np.asarray(well.centroid.coords[0])]))
        for i in hidden
    )

    distances = distance_to_goal(env, np.arange(4.0), positions, goal="well")
    assert np.all(distances >= exact - 1e-9)
    assert np.all(distances <= exact + env.bin_spacing)
    np.testing.assert_allclose(distances[3], exact[3], rtol=1e-12)

    paths = _environment_paths(env, call="test")
    goal = resolve_goal(env, "well", paths=paths, call="test")
    unit = goal.directions(env, paths, positions)
    cosine = np.sum(unit * toward, axis=1) / np.linalg.norm(toward, axis=1)
    assert np.all(cosine > 0.99)


def test_track_region_goal_is_reached_where_the_track_enters_it():
    """On a track the edge is where the centerline crosses the region."""
    from neurospatial.behavior import shortest_path_length

    track = Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=4.0, units="cm"
    ).with_track_width(4.0)
    regions = Regions()
    regions.add("well", point=(90.0, 0.0), radius=5.0)  # on the track: 85 to 95
    track = track.with_regions(regions)
    positions = np.array([[10.0, 0.0], [50.0, 1.0], [88.0, 0.0], [97.0, 0.0]])
    distances = distance_to_goal(track, np.arange(4.0), positions, goal="well")
    # 97 cm is past the well: its nearest edge is the far one, at 95 cm.
    np.testing.assert_allclose(distances, [75.0, 35.0, 0.0, 2.0], atol=1e-9)
    # The same as the geodesic to the entry point itself.
    entry = np.array([85.0, 0.0])
    assert distances[0] == pytest.approx(
        shortest_path_length(track, positions[0], entry)
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


@pytest.mark.parametrize(
    ("function", "options"),
    [
        (distance_to_goal, {"goal": "east_well"}),
        (approach_rate, {"goal": "east_well"}),
        (goal_bias, {"goal": "east_well"}),
        (compute_path_efficiency, {"goal": "east_well"}),
        (graph_turn_sequence, {}),
    ],
)
def test_empty_input_says_there_are_no_samples(arena, function, options):
    with pytest.raises(ValueError, match=r"no samples|empty"):
        function(arena, np.array([]), np.empty((0, 2)), **options)
