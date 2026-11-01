"""Known-answer tests for the bin-based navigation covariates.

Every expected value here is derived from the geometry of the environment,
not from the code under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import BinIndexOutOfRangeError, Environment, Intervals
from neurospatial.behavior import (
    cost_to_goal,
    distance_to_reward,
    goal_pair_direction_labels,
    graph_turn_sequence,
    heading_direction_labels,
    headings_to_direction_labels,
    path_progress,
    time_to_goal,
)
from neurospatial.behavior.segmentation import Trial


@pytest.fixture(scope="module")
def corridor() -> Environment:
    """A 1 cm wide, 100 cm long corridor of 1 cm bins centred at 0.5 ... 99.5."""
    x = np.arange(0.5, 100.0, 1.0)
    return Environment.from_samples(np.c_[x, np.zeros_like(x)], bin_size=1.0)


def _bin(env: Environment, x: float) -> int:
    return int(env.bin_at(np.array([[x, 0.0]]))[0])


class TestPathProgress:
    def test_halfway_along_a_corridor_is_one_half(self, corridor):
        """On a uniform 1-bin-wide corridor progress is linear between the zones.

        Start is the bin centred at 10.5 and goal the one at 90.5, so 30.5 is a
        quarter and 50.5 half of the way; the stretch behind the start is a dead
        end that joins only the start, so it reads 0.
        """
        from shapely.geometry import box

        from neurospatial import Regions

        regions = Regions(corridor.regions.values())
        regions.add("start", polygon=box(10, -1, 11, 1))
        regions.add("goal", polygon=box(90, -1, 91, 1))
        env = corridor.with_regions(regions)
        positions = np.c_[[10.5, 30.5, 50.5, 90.5, 5.5], np.zeros(5)]
        progress = path_progress(
            env, np.arange(5.0), positions, start="start", goal="goal"
        )
        np.testing.assert_allclose(progress, [0.0, 0.25, 0.5, 1.0, 0.0], atol=1e-12)


@pytest.mark.parametrize(
    "call",
    [
        lambda bins, env: cost_to_goal(bins, env, goal_bins=5),
        lambda bins, env: cost_to_goal(
            bins, env, goal_bins=5, cost_map=np.zeros(env.n_bins)
        ),
    ],
    ids=["cost_to_goal", "cost_to_goal_weighted"],
)
class TestPositionBinsValidation:
    def test_minus_two_is_refused(self, corridor, call):
        with pytest.raises(BinIndexOutOfRangeError):
            call(np.array([0, -2, 3]), corridor)

    def test_past_the_last_bin_is_refused(self, corridor, call):
        with pytest.raises(BinIndexOutOfRangeError):
            call(np.array([0, corridor.n_bins]), corridor)

    def test_floats_are_refused(self, corridor, call):
        with pytest.raises(TypeError, match=r"position_bins"):
            call(np.array([0.7, 3.2]), corridor)

    def test_minus_one_is_missing(self, corridor, call):
        result = call(np.array([0, -1, 5]), corridor)
        assert np.isnan(result[1])
        np.testing.assert_allclose(result[[0, 2]], [5.0, 0.0])


def _corridor(bin_size: float) -> Environment:
    """A 100 cm corridor of ``bin_size`` cm bins centred at b/2, 3b/2, ..."""
    x = np.arange(bin_size / 2, 100.0, bin_size)
    return Environment.from_samples(np.c_[x, np.zeros_like(x)], bin_size=bin_size)


class TestCostToGoal:
    """Cost is a per-unit-length rate integrated along the path.

    The start bin centre is at 20 - b/2 and the goal bin centre at 100 - b/2,
    80 cm apart for every bin size b.
    """

    @pytest.mark.parametrize("bin_size", [1.0, 2.0, 5.0])
    def test_uniform_cost_scales_the_length_independent_of_bin_size(self, bin_size):
        env = _corridor(bin_size)
        start = _bin(env, 20 - bin_size / 2)
        goal = _bin(env, 100 - bin_size / 2)
        bins = np.array([start, goal])
        ones = np.ones(env.n_bins)

        np.testing.assert_allclose(cost_to_goal(bins, env, goal_bins=goal), [80, 0])
        np.testing.assert_allclose(
            cost_to_goal(bins, env, goal_bins=goal, cost_map=np.zeros(env.n_bins)),
            [80, 0],
        )
        np.testing.assert_allclose(
            cost_to_goal(bins, env, goal_bins=goal, cost_map=ones), [160, 0]
        )
        np.testing.assert_allclose(
            cost_to_goal(bins, env, goal_bins=goal, terrain_difficulty=2 * ones),
            [160, 0],
        )
        np.testing.assert_allclose(
            cost_to_goal(
                bins, env, goal_bins=goal, cost_map=ones, terrain_difficulty=2 * ones
            ),
            [320, 0],
        )

    @pytest.mark.parametrize("bin_size", [1.0, 2.0, 5.0])
    def test_a_punished_stretch_adds_its_length_times_the_cost(self, bin_size):
        """Bins centred in [40, 60) cost 9 extra per cm: 80 + 9 * 20 = 260."""
        env = _corridor(bin_size)
        start = _bin(env, 20 - bin_size / 2)
        goal = _bin(env, 100 - bin_size / 2)
        centers = env.bin_centers[:, 0]
        cost_map = np.where((centers >= 40) & (centers < 60), 9.0, 0.0)

        result = cost_to_goal(np.array([start]), env, goal_bins=goal, cost_map=cost_map)

        np.testing.assert_allclose(result, [260.0])

    @pytest.mark.parametrize(
        ("name", "kind", "match"),
        [
            ("cost_map", "negative", r"cost_map.*negative"),
            ("cost_map", "nan", r"cost_map.*finite"),
            ("cost_map", "short", r"cost_map.*shape"),
            ("terrain_difficulty", "zero", r"terrain_difficulty.*positive"),
            ("terrain_difficulty", "nan", r"terrain_difficulty.*finite"),
            ("terrain_difficulty", "short", r"terrain_difficulty.*shape"),
        ],
    )
    def test_invalid_costs_are_refused(self, corridor, name, kind, match):
        """A negative weight hung Dijkstra; NaN silently gave inf everywhere."""
        values = {
            "negative": np.r_[-1.0, np.zeros(corridor.n_bins - 1)],
            "zero": np.r_[0.0, np.ones(corridor.n_bins - 1)],
            "nan": np.r_[np.nan, np.ones(corridor.n_bins - 1)],
            "short": np.ones(corridor.n_bins - 1),
        }[kind]
        with pytest.raises(ValueError, match=match):
            cost_to_goal(np.array([0, 1]), corridor, goal_bins=5, **{name: values})

    def test_samples_outside_the_environment_are_nan(self, corridor):
        result = cost_to_goal(
            np.array([0, -1]),
            corridor,
            goal_bins=5,
            cost_map=np.ones(corridor.n_bins),
        )
        np.testing.assert_array_equal(result, [10.0, np.nan])


@pytest.fixture(scope="module")
def t_maze() -> Environment:
    """A T-maze track: a 50 cm stem heading North and two 50 cm arms, 5 cm bins."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, -50.0))
    graph.add_node(1, pos=(0.0, 0.0))
    graph.add_node(2, pos=(-50.0, 0.0))
    graph.add_node(3, pos=(50.0, 0.0))
    for u, v in [(0, 1), (1, 2), (1, 3)]:
        graph.add_edge(u, v, distance=50.0)
    return Environment.from_graph(
        graph,
        edge_order=[(0, 1), (1, 2), (1, 3)],
        bin_size=5.0,
        track_width=5.0,
    )


def _t_maze_run(env: Environment, arm_x: float, rate_hz: float) -> np.ndarray:
    """Bins of a 20 cm/s run up the stem and out to the arm end at ``arm_x``."""
    n = round(50 / 20 * rate_hz)
    stem = np.c_[np.zeros(n), np.linspace(-49.0, -1.0, n)]
    arm = np.c_[np.linspace(0.0, arm_x, n), np.zeros(n)]
    return env.bin_at(np.vstack([stem, arm]))


class TestGraphTurnSequence:
    @pytest.mark.parametrize("rate_hz", [30.0, 500.0])
    @pytest.mark.parametrize(("arm_x", "expected"), [(-49.0, "left"), (49.0, "right")])
    def test_one_turn_at_a_t_junction_at_any_frame_rate(
        self, t_maze, rate_hz, arm_x, expected
    ):
        """Heading North, the West arm is a left turn, the East arm a right."""
        bins = _t_maze_run(t_maze, arm_x, rate_hz)
        assert graph_turn_sequence(t_maze, bins) == expected

    def test_a_straight_oblique_run_has_no_turns(self):
        """A lattice staircase along a straight line is not a sequence of turns."""
        x, y = np.meshgrid(np.arange(0.0, 50.0), np.arange(0.0, 50.0))
        field = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        t = np.linspace(0, 1, 2000)
        run = np.c_[5 + 30 * t, 5 + 20 * t]
        assert graph_turn_sequence(field, field.bin_at(run)) == ""

    def test_jitter_at_a_bin_border_is_not_a_turn(self):
        """Flicking between two bins, then running straight East: no turn."""
        x, y = np.meshgrid(np.arange(0.0, 50.0), np.arange(0.0, 50.0))
        field = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        rng = np.random.default_rng(0)
        dwell = np.c_[np.full(200, 10.0), 20.0 + rng.normal(0, 0.3, 200)]
        run = np.c_[np.linspace(10.0, 40.0, 300), 20.0 + rng.normal(0, 0.3, 300)]
        bins = field.bin_at(np.vstack([dwell, run]))
        assert len(np.unique(bins[:200])) == 2  # the dwell does flicker
        assert graph_turn_sequence(field, bins) == ""

    def test_noisy_corner_in_an_open_field_is_one_left_turn(self):
        """East 30 cm, then North 30 cm, with 0.3 cm tracking noise: one left."""
        x, y = np.meshgrid(np.arange(0.0, 50.0), np.arange(0.0, 50.0))
        field = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        rng = np.random.default_rng(1)
        east = np.c_[np.linspace(5.0, 35.0, 300), np.full(300, 5.0)]
        north = np.c_[np.full(300, 35.0), np.linspace(5.0, 35.0, 300)]
        run = np.vstack([east, north]) + rng.normal(0, 0.3, (600, 2))
        assert graph_turn_sequence(field, field.bin_at(run)) == "left"

    def test_samples_outside_the_environment_are_skipped(self, t_maze):
        """-1 must not read as the last bin and invent a turn there."""
        bins = _t_maze_run(t_maze, -49.0, 30.0)
        bins[::7] = -1
        assert graph_turn_sequence(t_maze, bins) == "left"

    def test_float_bins_are_refused(self, t_maze):
        with pytest.raises(TypeError, match=r"position_bins"):
            graph_turn_sequence(t_maze, np.array([0.0, 1.0, 2.0]))

    def test_out_of_range_bins_are_refused(self, t_maze):
        with pytest.raises(BinIndexOutOfRangeError):
            graph_turn_sequence(t_maze, np.array([0, 1, t_maze.n_bins]))


@pytest.fixture(scope="module")
def field_1cm() -> Environment:
    """A 10 x 10 cm field of 1 cm bins."""
    x, y = np.meshgrid(np.arange(0.5, 10.0), np.arange(0.5, 10.0))
    return Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=1.0)


class TestDistanceToReward:
    def test_euclidean_is_the_exact_distance_not_bin_centre_to_bin_centre(
        self, field_1cm
    ):
        times = np.arange(5) * 0.1
        positions = np.c_[np.array([0.2, 1.3, 2.9, 3.4, 6.1]), np.full(5, 5.2)]
        reward_xy = np.array([[3.2, 5.2]])
        result = distance_to_reward(
            field_1cm,
            times,
            positions,
            np.array([0.2]),
            reward_xy,
            metric="euclidean",
        )
        np.testing.assert_allclose(result, np.abs(positions[:, 0] - 3.2))

    def test_reward_next_to_a_missing_sample_is_interpolated(self, field_1cm):
        """A missing sample is bridged: a reward on it is located between its
        observed neighbours, here at x = 4.5."""
        times = np.arange(8) * 0.1
        positions = np.c_[np.arange(8.0) + 0.5, np.full(8, 5.5)]
        positions[4] = np.nan
        result = distance_to_reward(
            field_1cm, times, positions, np.array([0.4]), metric="euclidean"
        )
        expected = np.abs(positions[:, 0] - 4.5)
        np.testing.assert_allclose(result, expected)

    def test_explicit_rewards_on_another_clock_warn(self, field_1cm):
        """Reward times in ms against times in s used to give plausible distances."""
        times = np.arange(10) * 0.1
        positions = np.c_[np.arange(10.0) + 0.5, np.full(10, 5.5)]
        with pytest.warns(UserWarning, match=r"recorded span"):
            distance_to_reward(
                field_1cm,
                times,
                positions,
                np.array([300.0, 700.0]),
                np.array([[3.5, 5.5], [7.5, 5.5]]),
                metric="euclidean",
            )

    def test_explicit_reward_outside_the_environment_warns_and_is_nan(self, field_1cm):
        times = np.arange(4) * 0.1
        positions = np.c_[np.arange(4.0) + 0.5, np.full(4, 5.5)]
        with pytest.warns(UserWarning, match=r"outside the environment"):
            result = distance_to_reward(
                field_1cm,
                times,
                positions,
                np.array([0.1]),
                np.array([[50.0, 50.0]]),
                metric="euclidean",
            )
        assert np.all(np.isnan(result))


class TestHeadingDirectionLabels:
    def test_jittered_eastward_run_is_one_label(self):
        """Sectors are centred on the cardinal directions, so East is not split.

        With sector edges at 0 and +-pi a noisy eastward run used to split
        about half and half between two labels.
        """
        rng = np.random.default_rng(0)
        t = np.arange(0, 10, 1 / 30)
        x = np.r_[np.linspace(0, 100, 150), np.linspace(100, 0, 150)]
        y = rng.normal(0, 0.3, len(x))
        labels = heading_direction_labels(t, np.c_[x, y], min_speed=5.0)
        assert set(labels[5:145]) == {"0°"}
        assert set(labels[155:295]) == {"180°"}

    @pytest.mark.parametrize(
        ("heading", "n_directions", "expected"),
        [
            (0.0, 8, "0°"),
            (np.pi / 2, 8, "90°"),
            (-np.pi / 2, 8, "-90°"),
            (np.pi, 8, "180°"),
            (-np.pi, 8, "180°"),
            (np.pi / 4, 8, "45°"),
            (-3 * np.pi / 4, 8, "-135°"),
            # A sector is half-open [centre - w/2, centre + w/2): its
            # clockwise edge belongs to it.
            (np.pi / 8, 8, "45°"),
            (np.pi / 8 - 1e-9, 8, "0°"),
            (np.pi / 3, 4, "90°"),
            (2.0, 3, "120°"),
            (0.3, 16, "22.5°"),
        ],
    )
    def test_sector_labels_are_the_centre_direction(
        self, heading, n_directions, expected
    ):
        labels = headings_to_direction_labels(
            np.array([heading]),
            np.array([10.0]),
            n_directions=n_directions,
            min_speed=5.0,
        )
        assert labels.tolist() == [expected]

    @pytest.fixture(params=["heading_direction_labels", "headings_to_direction_labels"])
    def label(self, request):
        """Each labeler on three samples, with the options given."""
        if request.param == "heading_direction_labels":
            times = np.arange(3) / 10
            positions = np.c_[np.arange(3.0), np.zeros(3)]
            return lambda **kw: heading_direction_labels(times, positions, **kw)
        return lambda **kw: headings_to_direction_labels(np.zeros(3), np.ones(3), **kw)

    @pytest.mark.parametrize("min_speed", [np.nan, -1.0, np.inf])
    def test_min_speed_must_be_finite_and_non_negative(self, label, min_speed):
        with pytest.raises(ValueError, match=r"min_speed"):
            label(min_speed=min_speed)

    @pytest.mark.parametrize("n_directions", [0, -4, 2.5, True])
    def test_n_directions_must_be_a_positive_integer(self, label, n_directions):
        with pytest.raises(ValueError, match=r"n_directions"):
            label(n_directions=n_directions, min_speed=0.5)

    def test_headings_in_degrees_warn(self):
        headings = np.array([10.0, 200.0, 350.0, 90.0])
        with pytest.warns(UserWarning, match=r"degrees"):
            headings_to_direction_labels(headings, np.full(4, 10.0), min_speed=5.0)

    @pytest.mark.parametrize(
        ("headings", "speeds", "match"),
        [
            (np.zeros((3, 1)), np.ones(3), r"1-D headings.*\(3, 1\)"),
            (np.zeros(3), np.ones((1, 3)), r"1-D speeds.*\(1, 3\)"),
            (np.zeros(()), np.ones(()), r"1-D headings"),
            (
                np.array([0.0, np.inf, 0.0]),
                np.ones(3),
                r"infinite value\(s\) in headings, first at index 1",
            ),
            (np.zeros(3), np.array([1.0, -0.5, 1.0]), r"speeds\[1\] = -0.5"),
            (np.zeros(3), np.array([1.0, 1.0, np.inf]), r"speeds\[2\] = inf"),
        ],
    )
    def test_precomputed_inputs_are_validated(self, headings, speeds, match):
        with pytest.raises(ValueError, match=match) as caught:
            headings_to_direction_labels(headings, speeds, min_speed=1.0)
        message = str(caught.value)
        assert message.startswith("headings_to_direction_labels()")
        assert "Fix:" in message

    def test_swapped_headings_and_speeds_name_the_order(self):
        """Wrapped headings passed as speeds are negative, so the swap raises."""
        headings = np.array([0.5, -2.0, 1.0])
        speeds = np.array([10.0, 12.0, 8.0])
        with pytest.raises(
            ValueError, match=r"headings_to_direction_labels\(headings, speeds"
        ):
            headings_to_direction_labels(speeds, headings, min_speed=5.0)

    def test_empty_precomputed_input_has_no_labels(self):
        labels = headings_to_direction_labels(np.array([]), np.array([]), min_speed=1.0)
        assert labels.dtype == object
        assert labels.shape == (0,)


class TestTrialHelpersOnAnotherClock:
    """Trials on a different clock than ``times`` warn instead of silently
    returning all NaN / -1 / None."""

    @pytest.fixture
    def env_with_regions(self, corridor):
        from shapely.geometry import box

        from neurospatial import Regions

        regions = Regions(corridor.regions.values())
        regions.add("home", polygon=box(0, -1, 5, 1))
        regions.add("goal", polygon=box(95, -1, 100, 1))
        return corridor.with_regions(regions)

    @staticmethod
    def _positions(times: np.ndarray) -> np.ndarray:
        """A 10 cm/s run along the corridor, inside it at every sample."""
        return np.c_[10.0 * times + 0.5, np.zeros(len(times))]

    @staticmethod
    def _trials(scale: float = 1.0) -> Intervals[Trial]:
        return Intervals(
            [
                Trial(
                    start_time=1.0 * scale,
                    stop_time=4.0 * scale,
                    start_region="home",
                    end_region="goal",
                    outcome="reached",
                )
            ]
        )

    def test_time_to_goal_warns(self):
        times = np.arange(10.0)
        with pytest.warns(UserWarning, match=r"same clock"):
            result = time_to_goal(times, self._trials(scale=1000.0))
        assert np.all(np.isnan(result))

    def test_goal_pair_direction_labels_warns(self):
        times = np.arange(10.0)
        with pytest.warns(UserWarning, match=r"same clock"):
            labels = goal_pair_direction_labels(times, self._trials(scale=1000.0))
        assert labels.tolist() == [None] * 10

    def test_path_progress_warns(self, env_with_regions):
        times = np.arange(10.0)
        with pytest.warns(UserWarning, match=r"same clock"):
            progress = path_progress(
                env_with_regions,
                times,
                self._positions(times),
                self._trials(scale=1000.0),
                start="home",
                goal="goal",
            )
        assert np.all(np.isnan(progress))

    @pytest.mark.parametrize(
        ("times", "match"),
        [
            (np.array([0.0, 2.0, 1.0, 3.0]), "decreases"),
            (np.array([0.0, 1.0, 1.0, 3.0]), "strictly increasing"),
            (np.array([0.0, np.nan, 2.0, 3.0]), "non-finite"),
        ],
    )
    def test_trial_arrays_need_strictly_increasing_times(
        self, env_with_regions, times, match
    ):
        """A trial's samples are a slice of the session clock, so it must be sorted."""
        with pytest.raises(ValueError, match=match):
            time_to_goal(times, self._trials())
        with pytest.raises(ValueError, match=match):
            path_progress(
                env_with_regions,
                times,
                self._positions(np.arange(4.0)),
                self._trials(),
                start="home",
                goal="goal",
            )

    def test_overlapping_trials_do_not_warn(self, env_with_regions):
        import warnings

        times = np.arange(10.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ttg = time_to_goal(times, self._trials())
            goal_pair_direction_labels(times, self._trials())
            path_progress(
                env_with_regions,
                times,
                self._positions(times),
                self._trials(),
                start="home",
                goal="goal",
            )
        np.testing.assert_array_equal(ttg[1:5], [3.0, 2.0, 1.0, np.nan])

    def test_no_trials_do_not_warn(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert np.all(
                np.isnan(time_to_goal(np.arange(5.0), Intervals([], record_type=Trial)))
            )


class TestDistanceToGoal:
    """distance_to_goal against independent known answers."""

    @staticmethod
    def _box(*, wall: bool = False) -> Environment:
        from shapely.geometry import box as shapely_box

        from neurospatial import Regions

        xs, ys = np.meshgrid(np.arange(0.0, 50.0, 0.5), np.arange(0.0, 50.0, 0.5))
        samples = np.column_stack([xs.ravel(), ys.ravel()])
        if wall:
            # A wall, two bins thick, around x = 25 from y = 0 to 40, open above.
            samples = samples[
                ~((np.abs(samples[:, 0] - 25.0) < 3.0) & (samples[:, 1] < 40.0))
            ]
        env = Environment.from_samples(samples, bin_size=2.0)
        regions = Regions()
        regions.add("reward", polygon=shapely_box(40.0, 0.0, 50.0, 10.0))
        return env.with_regions(regions)

    def test_distance_to_goal_needs_no_start(self):
        from neurospatial import Intervals
        from neurospatial.behavior import distance_to_goal

        env = self._box()
        times = np.arange(6.0)
        positions = np.array(
            [
                [100.0, 100.0],
                [5.0, 5.0],
                [20.0, 30.0],
                [41.0, 1.0],
                [5.0, 45.0],
                [30.0, 5.0],
            ]
        )
        trials = Intervals.from_arrays([0.0, 3.0], [3.0, 5.0])
        geodesic = distance_to_goal(env, times, positions, trials, goal="reward")
        # The first trial starts outside the environment; the rest is defined.
        assert np.isnan(geodesic[0])
        assert np.isnan(geodesic[5])  # outside every trial
        goal_centers = env.bin_centers[env.bins_in_region("reward")]
        straight = np.min(
            np.linalg.norm(positions[:, None, :] - goal_centers[None], axis=2), axis=1
        )
        # A convex box: the shortest path inside is the straight line.
        np.testing.assert_allclose(geodesic[1:5], straight[1:5], rtol=1e-9)
        euclidean = distance_to_goal(
            env, times, positions, trials, goal="reward", metric="euclidean"
        )
        np.testing.assert_allclose(euclidean[1:5], straight[1:5], rtol=1e-12)
        at_goal = distance_to_goal(env, [0.0], goal_centers[:1], goal="reward")
        assert at_goal[0] == 0.0

    def test_distance_goes_around_a_wall(self):
        from neurospatial.behavior import distance_to_goal

        env = self._box(wall=True)
        assert env.n_bins < self._box().n_bins  # the wall removed bins
        position = np.array([[10.0, 5.0]])
        around = distance_to_goal(env, [0.0], position, goal="reward")[0]
        straight = distance_to_goal(
            env, [0.0], position, goal="reward", metric="euclidean"
        )[0]
        assert around > straight + 20.0

    def test_distance_along_a_track(self):
        from neurospatial.behavior import distance_to_goal
        from neurospatial.simulation.mazes import make_linear_track

        track = make_linear_track().env_track
        goal = np.asarray(track.regions["well_right"].data, dtype=float)
        start = np.asarray(track.regions["well_left"].data, dtype=float)
        positions = start + np.linspace(0.0, 1.0, 11)[:, None] * (goal - start)
        distances = distance_to_goal(track, np.arange(11.0), positions, goal=goal)
        goal_center = track.bin_centers[track.bin_at(goal[None])[0]]
        np.testing.assert_allclose(
            distances, np.linalg.norm(positions - goal_center, axis=1), atol=1e-9
        )
