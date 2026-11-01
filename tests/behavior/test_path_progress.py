"""Harmonic path progress: guarantees, caveats and inputs, checked independently.

Route distances come from ``env.distance_to`` (bin-graph distances, exact
along a track), random walks are simulated directly from the conductances,
and the documented caveats are pinned at their measured sizes.
"""

from __future__ import annotations

import warnings
from functools import cache

import networkx as nx
import numpy as np
import pytest
from scipy.sparse import csgraph
from shapely.geometry import box

from neurospatial import Environment, Intervals, Regions
from neurospatial.behavior import Trial, distance_to_goal, path_progress
from neurospatial.simulation import mazes

# Start and goal of each maze, joined by a single path on the tree-shaped
# tracks and by several on the looped ones.
TREE_MAZES = {
    "make_t_maze": ("start", "well_left"),
    "make_y_maze": ("arm1_end", "arm2_end"),
    "make_plus_maze": ("south_end", "north_end"),
    "make_w_maze": ("well_1", "well_3"),
    "make_radial_arm_maze": ("food_cup_0", "food_cup_4"),
    "make_sungod_maze": ("home_port", "arm_port_1"),
    "make_repeated_t_maze": ("start_box", "food_box"),
    "make_small_hex_maze": ("port_A", "port_B"),
}
LOOP_MAZES = {
    "make_crossword_maze": ("start_box_top_left", "start_box_bottom_right"),
    "make_hamlet_maze": ("agora", "drink_house"),
    "make_honeycomb_maze": ("platform_0", "platform_36"),
    "make_hampton_court_maze": ("entrance", "central_enclosure"),
    "make_rat_hexmaze": ("node_A0", "node_D23"),
}
OPEN_MAZES = {
    "make_barnes_maze": ("hole_1", "escape_hole"),
    "make_cheeseboard_maze": ("well_0_5", "well_10_10"),
    "make_square_maze": ("SW", "NE"),
    "make_watermaze": ("SW", "platform"),
}


@cache
def _maze(name: str):
    with warnings.catch_warnings():
        # The large 2-D mazes announce their grid size; that is not under test.
        warnings.filterwarnings("ignore", message="Creating large grid")
        return getattr(mazes, name)()


def _at_bins(env: Environment, start, goal, **kwargs) -> np.ndarray:
    """Progress at every bin centre, with one start and goal for the session."""
    centers = env.bin_centers
    return path_progress(
        env,
        np.arange(env.n_bins, dtype=float),
        centers,
        start=start,
        goal=goal,
        **kwargs,
    )


def _fixed_bins(env: Environment, *names, radius=None) -> np.ndarray:
    bins = []
    for name in names:
        region = env.regions[name]
        if region.kind == "point" and radius is not None:
            distance = np.linalg.norm(env.bin_centers - np.asarray(region.data), axis=1)
            bins.append(np.flatnonzero(distance <= radius))
        else:
            bins.append(env.bins_in_region(name))
    return np.concatenate(bins)


def _route_distances(track: Environment, start: str, goal: str):
    """Bin-graph distances from the start and to the goal, and the route length."""
    d_start = track.distance_to(region=start)
    d_goal = track.distance_to(region=goal)
    route = float(np.min(d_start[track.bins_in_region(goal)]))
    return d_start, d_goal, route


def _assert_no_false_extrema(env: Environment, progress, fixed) -> None:
    """No bin outside the start and goal bins is above or below all neighbours."""
    finite = np.isfinite(progress)
    coo = env.get_diffusion_geometry().W.tocoo()
    keep = (coo.data > 0) & finite[coo.row] & finite[coo.col]
    rows, cols = coo.row[keep], coo.col[keep]
    neighbour_max = np.full(env.n_bins, -np.inf)
    neighbour_min = np.full(env.n_bins, np.inf)
    np.maximum.at(neighbour_max, rows, progress[cols])
    np.minimum.at(neighbour_min, rows, progress[cols])
    free = finite & ~fixed & np.isfinite(neighbour_max)
    assert not np.any(progress[free] > neighbour_max[free] + 1e-12)
    assert not np.any(progress[free] < neighbour_min[free] - 1e-12)


ALL_FORMS = [
    *((name, "env_track", pair) for name, pair in {**TREE_MAZES, **LOOP_MAZES}.items()),
    *(
        (name, "env_2d", pair)
        for name, pair in {**TREE_MAZES, **LOOP_MAZES, **OPEN_MAZES}.items()
    ),
]


@pytest.mark.parametrize(
    ("name", "form", "pair"), ALL_FORMS, ids=[f"{n}-{f}" for n, f, _ in ALL_FORMS]
)
def test_maze_invariants(name, form, pair):
    env = getattr(_maze(name), form)
    radius = 5.0 if form == "env_2d" else None
    progress = _at_bins(env, *pair, point_radius=radius)
    fixed = np.zeros(env.n_bins, dtype=bool)
    fixed[_fixed_bins(env, *pair, radius=radius)] = True

    finite = np.isfinite(progress)
    assert np.all((progress[finite] >= 0.0) & (progress[finite] <= 1.0))
    # NaN exactly on components (by positive conductance) without both ends.
    conductances = env.get_diffusion_geometry().W
    _, component = csgraph.connected_components(conductances, directed=False)
    start_bins, goal_bins = (
        _fixed_bins(env, pair[0], radius=radius),
        _fixed_bins(env, pair[1], radius=radius),
    )
    both = np.intersect1d(component[start_bins], component[goal_bins])
    np.testing.assert_array_equal(finite, np.isin(component, both))
    _assert_no_false_extrema(env, progress, fixed)


@pytest.mark.parametrize("name", list(TREE_MAZES))
def test_route_fraction_on_tree_tracks(name):
    track = _maze(name).env_track
    start, goal = TREE_MAZES[name]
    progress = _at_bins(track, start, goal)
    d_start, d_goal, route = _route_distances(track, start, goal)
    on_route = np.isclose(d_start + d_goal, route, rtol=0, atol=1e-9)
    on_route[_fixed_bins(track, start, goal)] = False
    assert on_route.sum() > 5
    np.testing.assert_allclose(
        progress[on_route], d_start[on_route] / route, rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("name", list(TREE_MAZES))
def test_dead_ends_flat_on_tracks(name):
    """A bin off the route has the value of the junction where its arm leaves.

    On a tree, a bin ``x`` along an arm off the route at junction ``j`` has
    ``d_start - d_goal`` equal to that at ``j``, so the junction's value is
    ``(d_start - d_goal + L) / (2 L)``.
    """
    track = _maze(name).env_track
    start, goal = TREE_MAZES[name]
    progress = _at_bins(track, start, goal)
    d_start, d_goal, route = _route_distances(track, start, goal)
    off_route = d_start + d_goal > route + 1e-9
    off_route[_fixed_bins(track, start, goal)] = False
    assert off_route.sum() > 0
    junction_value = (d_start - d_goal + route) / (2 * route)
    np.testing.assert_allclose(
        progress[off_route], junction_value[off_route], rtol=0, atol=1e-12
    )


def test_hampton_court_blinds_flat_except_the_loop():
    track = _maze("make_hampton_court_maze").env_track
    progress = _at_bins(track, *LOOP_MAZES["make_hampton_court_maze"])
    conductances = track.get_diffusion_geometry().W.tocsr()
    for blind in (f"blind_{k}" for k in range(1, 8)):
        inside = track.bins_in_region(blind)
        values = progress[inside]
        if blind == "blind_4":
            # Two openings make it a loop, not a dead end.
            assert np.ptp(values) > 1e-3
            continue
        assert np.ptp(values) < 1e-12, blind
        # The blind has the value of the corridor bins it opens onto.
        neighbours = np.setdiff1d(conductances[inside].indices, inside)
        np.testing.assert_allclose(progress[neighbours], values[0], atol=1e-12)


def test_dead_ends_bounded_in_2d():
    env = _maze("make_t_maze").env_2d
    progress = _at_bins(env, "start", "well_left", point_radius=5.0)
    junction = np.asarray(env.regions["junction"].data)
    right = np.asarray(env.regions["well_right"].data)
    x, y = env.bin_centers.T
    arm = (x > junction[0] + 5.0) & (np.abs(y - junction[1]) < 10.0)
    entrance = arm & (x <= x[arm].min())
    assert np.all(progress[arm] <= progress[entrance].max() + 1e-12)
    # The spread across the arm shrinks with distance from the junction.
    near = arm & (x < junction[0] + 12.0)
    far = arm & (x > right[0] - 12.0)
    assert np.ptp(progress[far]) < np.ptp(progress[near])


def _random_walk_hits(
    conductances, start_bins, goal_bins, sources, *, n_walks: int, seed: int
):
    """Fraction of walks from each source that reach the goal before the start."""
    rng = np.random.default_rng(seed)
    csr = conductances.tocsr()
    indptr, indices = csr.indptr, csr.indices
    # A step from bin i picks neighbour j with probability W_ij / sum_j W_ij:
    # one search in the cumulative weights of all rows.
    cumulative = np.cumsum(csr.data)
    before_row = np.concatenate([[0.0], cumulative])[indptr[:-1]]
    row_total = np.asarray(csr.sum(axis=1)).ravel()
    absorbing = np.zeros(csr.shape[0], dtype=np.int8)
    absorbing[start_bins] = -1
    absorbing[goal_bins] = 1
    fractions = []
    for source in sources:
        position = np.full(n_walks, source)
        outcome = absorbing[position].astype(int)
        while np.any(outcome == 0):
            active = np.flatnonzero(outcome == 0)
            here = position[active]
            target = before_row[here] + rng.random(active.size) * row_total[here]
            step = np.searchsorted(cumulative, target, side="right")
            step = np.clip(step, indptr[here], indptr[here + 1] - 1)
            position[active] = indices[step]
            outcome[active] = absorbing[position[active]]
        fractions.append(np.mean(outcome == 1))
    return np.array(fractions)


def test_matches_random_walks():
    track = _maze("make_t_maze").env_track
    progress = _at_bins(track, "start", "well_left")
    start_bins = track.bins_in_region("start")
    goal_bins = track.bins_in_region("well_left")
    free = np.setdiff1d(
        np.flatnonzero(np.isfinite(progress)), [*start_bins, *goal_bins]
    )
    sources = np.random.default_rng(11).choice(free, size=4, replace=False)
    n_walks = 4_000
    hits = _random_walk_hits(
        track.get_diffusion_geometry().W,
        start_bins,
        goal_bins,
        sources,
        n_walks=n_walks,
        seed=12,
    )
    expected = progress[sources]
    standard_error = np.sqrt(expected * (1 - expected) / n_walks)
    assert np.all(np.abs(hits - expected) <= 4 * standard_error + 1e-12)


def test_corridor_width_constant():
    xs, ys = np.meshgrid(np.arange(0.0, 100.0, 0.5), np.arange(0.0, 10.0, 0.5))
    samples = np.column_stack([xs.ravel(), ys.ravel()])
    env = Environment.from_samples(samples, bin_size=2.0)
    rows = np.unique(env.bin_centers[:, 1])
    assert rows.size >= 5  # several bins across the corridor
    regions = Regions()
    regions.add("start", polygon=box(-1.0, -1.0, 4.0, 11.0))
    regions.add("goal", polygon=box(96.0, -1.0, 101.0, 11.0))
    env = env.with_regions(regions)
    progress = _at_bins(env, "start", "goal")

    x = env.bin_centers[:, 0]
    interior = (x > 10.0) & (x < 90.0)
    for column in np.unique(x[interior]):
        assert np.ptp(progress[x == column]) < 1e-9
    start_edge = env.bin_centers[env.bins_in_region("start"), 0].max()
    goal_edge = env.bin_centers[env.bins_in_region("goal"), 0].min()
    np.testing.assert_allclose(
        progress[interior],
        (x[interior] - start_edge) / (goal_edge - start_edge),
        atol=1e-9,
    )


def test_wall_respected_3d():
    grid = np.arange(1.0, 20.0, 2.0)
    xs, ys, zs = np.meshgrid(grid, grid, grid, indexing="ij")
    points = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    # A wall at x = 9 with a gap in one corner (y, z > 14).
    wall = (points[:, 0] == 9.0) & ~((points[:, 1] > 14.0) & (points[:, 2] > 14.0))
    env = Environment.from_samples(points[~wall], bin_size=2.0)
    start, goal = [1.0, 1.0, 1.0], [19.0, 1.0, 1.0]
    progress = _at_bins(env, start, goal, point_radius=3.0)

    finite = progress[np.isfinite(progress)]
    assert np.all((finite >= 0) & (finite <= 1))
    centers = env.bin_centers
    fixed = (np.linalg.norm(centers - start, axis=1) <= 3.0) | (
        np.linalg.norm(centers - goal, axis=1) <= 3.0
    )
    _assert_no_false_extrema(env, progress, fixed)
    before = env.bin_at(np.array([[7.0, 1.0, 1.0]]))[0]
    after = env.bin_at(np.array([[11.0, 1.0, 1.0]]))[0]
    assert progress[after] - progress[before] > 0.5


@pytest.mark.slow
def test_sized_targets_converge():
    probes = np.array([[0.0, 0.0], [0.0, 40.0], [-30.0, 50.0], [30.0, 50.0]])
    values = []
    for bin_size in (1.0, 0.5, 0.25):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = mazes.make_t_maze(bin_size=bin_size, include_track=False).env_2d
        values.append(
            path_progress(
                env,
                np.arange(len(probes), dtype=float),
                probes,
                start="start",
                goal="well_left",
                point_radius=5.0,
            )
        )
    assert np.max(np.abs(values[2] - values[1])) < 0.005


def test_point_target_needs_radius():
    env = _maze("make_t_maze").env_2d
    with pytest.raises(ValueError, match="point_radius was not given"):
        _at_bins(env, "start", "well_left")
    with pytest.raises(ValueError, match="point_radius was not given"):
        _at_bins(env, [0.0, -40.0], "well_left")
    with pytest.raises(ValueError, match=r"covers [01] bin\(s\)"):
        _at_bins(env, "start", "well_left", point_radius=0.5)
    track = _maze("make_t_maze").env_track
    assert np.isfinite(_at_bins(track, "start", [-45.0, 50.0])).any()


def _lollipop(*, with_long_route: bool) -> Environment:
    """A stem to a split, then a short route and (optionally) a long one."""
    graph = nx.Graph()
    nodes = {
        "s": (0.0, 0.0),
        "j": (0.0, 20.0),
        "g": (0.0, 40.0),
        "a": (20.0, 20.0),
        "b": (20.0, 40.0),
    }
    for node, xy in nodes.items():
        graph.add_node(node, pos=xy)
    edges = [("s", "j"), ("j", "g")]
    if with_long_route:
        edges += [("j", "a"), ("a", "b"), ("b", "g")]
    for u, v in edges:
        graph.add_edge(u, v, distance=float(np.hypot(*np.subtract(nodes[u], nodes[v]))))
    env = Environment.from_graph(graph, edges, bin_size=1.0, track_width=4.0)
    regions = Regions()
    regions.add("start", point=nodes["s"])
    regions.add("goal", point=nodes["g"])
    return env.with_regions(regions)


def test_alternative_routes_depend_on_topology():
    loop = _lollipop(with_long_route=True)
    probe = np.array([[0.0, 20.0], [0.0, 30.0]])  # the split, the short route
    with_loop = path_progress(loop, np.arange(2.0), probe, start="start", goal="goal")
    # Neither route's length fraction at the split: 20 / 40 or 20 / 80.
    assert abs(with_loop[0] - 0.5) > 0.04
    assert abs(with_loop[0] - 0.25) > 0.04
    stem_only = _lollipop(with_long_route=False)
    without = path_progress(
        stem_only, np.arange(2.0), probe, start="start", goal="goal"
    )
    np.testing.assert_allclose(without, [0.5, 0.75], atol=0.03)
    assert abs(with_loop[1] - without[1]) > 0.01


def test_coverage_effect_documented():
    xs, ys = np.meshgrid(np.arange(0.0, 100.0, 0.5), np.arange(0.0, 100.0, 0.5))
    full = np.column_stack([xs.ravel(), ys.ravel()])
    patch = (np.abs(full[:, 0] - 50.0) < 15.0) & (np.abs(full[:, 1] - 50.0) < 15.0)
    values = []
    for samples in (full, full[~patch]):
        env = Environment.from_samples(samples, bin_size=2.0)
        values.append((env, _at_bins(env, [5.0, 5.0], [95.0, 95.0], point_radius=6.0)))
    (env_full, with_patch_visited), (env_holed, with_hole) = values
    common = env_full.bin_at(env_holed.bin_centers)
    distance = np.max(np.abs(env_holed.bin_centers - 50.0), axis=1)
    far = distance > 45.0  # more than 30 cm from the patch's edge
    change = np.abs(with_patch_visited[common] - with_hole)
    assert np.all(change[far] < 0.1)
    assert np.max(change) > np.max(change[far])


@pytest.fixture
def t_track_trials():
    """Three trials on the T-maze track: left, unknown, right."""
    track = _maze("make_t_maze").env_track
    times = np.arange(30.0)
    stem = np.column_stack([np.zeros(10), np.linspace(-40.0, 45.0, 10)])
    positions = np.vstack([stem, stem, stem])
    trials = Intervals(
        [
            Trial(0.0, 10.0, "start", "well_left", "reached"),
            Trial(10.0, 20.0, "start", None, "timeout"),
            Trial(20.0, 30.0, "start", "well_right", "reached"),
        ]
    )
    return track, times, positions, trials


def test_per_interval_targets_and_unknown(t_track_trials, monkeypatch):
    from neurospatial.behavior import navigation

    track, times, positions, trials = t_track_trials
    solves = []
    real = navigation._harmonic_progress
    monkeypatch.setattr(
        navigation,
        "_harmonic_progress",
        lambda *args: solves.append(1) or real(*args),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        progress = path_progress(
            track, times, positions, trials, start="start", goal=trials.end_region
        )
    assert len(solves) == 2  # one per distinct (start, goal) pair
    assert np.all(np.isnan(progress[10:20]))  # unknown destination
    left = path_progress(track, times, positions, start="start", goal="well_left")
    right = path_progress(track, times, positions, start="start", goal="well_right")
    np.testing.assert_array_equal(progress[:10], left[:10])
    np.testing.assert_array_equal(progress[20:], right[20:])
    filled = path_progress(
        track, times, positions, trials, start="start", goal="well_left"
    )
    np.testing.assert_array_equal(filled[10:20], left[10:20])


def test_overlapping_intervals_refused(t_track_trials):
    track, times, positions, _ = t_track_trials
    overlapping = Intervals.from_arrays([0.0, 5.0], [10.0, 15.0])
    with pytest.raises(ValueError, match="overlap"):
        path_progress(
            track, times, positions, overlapping, start="start", goal="well_left"
        )
    with pytest.raises(ValueError, match="overlap"):
        distance_to_goal(track, times, positions, overlapping, goal="well_left")


def test_mostly_nan_warns():
    left = np.column_stack([np.arange(0.0, 20.0, 0.5), np.zeros(40)])
    right = left + np.array([100.0, 0.0])
    env = Environment.from_samples(np.vstack([left, right]), bin_size=2.0)
    positions = np.vstack([left[:5], right])  # mostly in the other component
    with pytest.warns(
        UserWarning,
        match=r"40 of 45 analyzed samples \(89%\) are NaN.*does not connect",
    ) as record:
        progress = path_progress(
            env,
            np.arange(len(positions), dtype=float),
            positions,
            start=[1.0, 0.0],
            goal=[18.0, 0.0],
            point_radius=2.5,
        )
    assert len(record) == 1
    assert np.all(np.isnan(progress[5:]))


def test_start_goal_same_bin(t_track_trials):
    track, times, positions, _ = t_track_trials
    with pytest.raises(ValueError, match="share 1 bin"):
        path_progress(track, times, positions, start="well_left", goal="well_left")
    trials = Intervals.from_arrays(
        [0.0, 25.0], [25.0, 30.0], goal=["well_left", "start"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        progress = path_progress(
            track, times, positions, trials, start="start", goal=trials.goal
        )
    assert np.all(np.isnan(progress[25:]))
    assert np.isfinite(progress[:25]).all()


def test_target_forms(t_track_trials):
    """Per-interval targets as lists, tuples, points, None and Series by id."""
    import pandas as pd

    track, times, positions, trials = t_track_trials
    left = np.asarray(track.regions["well_left"].data, dtype=float)
    right = np.asarray(track.regions["well_right"].data, dtype=float)
    by_name = path_progress(
        track,
        times,
        positions,
        trials,
        start="start",
        goal=["well_left", None, "well_right"],
    )
    as_tuple = path_progress(
        track,
        times,
        positions,
        trials,
        start="start",
        goal=("well_left", None, "well_right"),
    )
    as_points = path_progress(
        track, times, positions, trials, start="start", goal=[left, None, right]
    )
    series = pd.Series(
        ["well_right", "well_left", None], index=trials.interval_ids[[2, 0, 1]]
    )
    by_id = path_progress(track, times, positions, trials, start="start", goal=series)
    for other in (as_tuple, as_points, by_id):
        np.testing.assert_array_equal(other, by_name)
    assert np.isnan(by_name[10:20]).all()

    # No trial reached an end region: every sample is NaN, which is expected.
    none = Intervals.from_arrays([0.0, 10.0], [10.0, 20.0], goal=[None, None])
    with pytest.warns(UserWarning, match="the start or goal is unknown"):
        unknown = path_progress(
            track, times, positions, none, start="start", goal=none.goal
        )
    assert np.isnan(unknown).all()
    with pytest.raises(ValueError, match="has 2 entries"):
        path_progress(
            track, times, positions, trials, start="start", goal=["well_left"] * 2
        )
    with pytest.raises(TypeError, match="region names, points or None"):
        path_progress(
            track,
            times,
            positions,
            trials,
            start="start",
            goal=[{"x": 1}, None, "well_left"],
        )
    # A list of numbers is one point, checked against the environment.
    with pytest.raises(ValueError, match="has 3 coordinates"):
        path_progress(track, times, positions, trials, start="start", goal=[1, 2, 3])


def test_one_dimensional_points_need_no_radius():
    line = np.arange(0.0, 100.0, 0.5)[:, None]
    env = Environment.from_samples(line, bin_size=2.0)
    progress = path_progress(
        env, [0.0, 1.0, 2.0], [[1.0], [50.0], [99.0]], start=1.0, goal=[99.0]
    )
    np.testing.assert_allclose(progress, [0.0, 0.5, 1.0], atol=0.02)


def test_single_shared_target_raises_with_intervals(t_track_trials):
    track, times, positions, trials = t_track_trials
    with pytest.raises(ValueError, match="share 1 bin"):
        path_progress(
            track, times, positions, trials, start="well_left", goal="well_left"
        )
