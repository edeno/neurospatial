"""Regression tests for behavior-layer binning and dt correctness.

Covers three families of silent-correctness defects:

1. Out-of-bounds bin indices (``-1`` / ``>= n_bins``) silently wrapping into a
   real region / Voronoi label when fancy-indexing a ``(n_bins,)`` array.
2. Velocity / rate / lag computations dividing by ``dt`` without guarding
   zero / negative / duplicate timestamps.
3. Path-efficiency helpers reporting an unreachable goal as ``0.0`` and mixing
   Euclidean with geodesic optimal distances.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from shapely.geometry import Point

from neurospatial import Environment, Regions
from neurospatial.behavior.decisions import (
    compute_decision_analysis,
    detect_boundary_crossings,
    geodesic_voronoi_labels,
)
from neurospatial.behavior.navigation import (
    approach_rate,
    compute_path_efficiency,
    time_efficiency,
)
from neurospatial.behavior.segmentation import (
    detect_region_crossings,
    segment_by_velocity,
)
from neurospatial.behavior.trajectory import mean_square_displacement
from neurospatial.behavior.trials import detect_trials

from .conftest import duplicate_timestamps, nonfinite_timestamps

# =============================================================================
# Family 1 — out-of-bounds bins must not wrap into real regions / labels
# =============================================================================


class TestSafeGatherRegions:
    """Out-of-env (-1) bins must read as 'outside', not wrap to the last bin."""

    def test_detect_region_crossings_ignores_out_of_env_sample(
        self, grid_env_with_last_bin_region
    ):
        env, region_name, _last_bin = grid_env_with_last_bin_region

        # Pick a bin that is genuinely *outside* the region.
        outside_bin = 0
        from neurospatial.ops.mapping import regions_to_mask

        assert not regions_to_mask(env, [region_name])[outside_bin]

        # Surrounding samples are outside the region; the middle sample is -1
        # (out of env). A wrapped -1 -> last_bin would be *inside* the region,
        # creating a spurious entry+exit crossing.
        positions = env.bin_centers[[outside_bin] * 5]
        positions[2] = np.nan  # lost tracking: bin -1
        times = np.linspace(0.0, 1.0, len(positions))

        crossings = detect_region_crossings(
            env, times, positions, region_name=region_name
        )

        assert crossings == []

    @staticmethod
    def _grid_with_corner_regions():
        x = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, x)
        env = Environment.from_samples(
            np.column_stack([xx.ravel(), yy.ravel()]), bin_size=5.0
        )
        regions = Regions(env.regions.values())
        regions.add("start", polygon=Point(*env.bin_centers[0]).buffer(6.0))
        # The last bin: a wrapped -1 would read as inside the goal.
        regions.add("goal", polygon=Point(*env.bin_centers[-1]).buffer(6.0))
        return env.with_regions(regions)

    def test_detect_trials_out_of_env_sample_does_not_reach_the_goal(self):
        env = self._grid_with_corner_regions()
        neutral_bin = 100  # in neither region
        from neurospatial.ops.mapping import regions_to_mask

        assert not regions_to_mask(env, ["start", "goal"])[neutral_bin]
        # Leave the start, then one sample outside the environment.
        positions = env.bin_centers[[0, 0, neutral_bin, neutral_bin, neutral_bin]]
        positions[-1] = [500.0, 500.0]
        times = np.linspace(0.0, 1.0, len(positions))

        trials = detect_trials(
            env, times, positions, regions=["start", "goal"], max_gap=None
        )

        assert trials.outcome.tolist() == ["censored"]
        assert trials.stop.tolist() == [times[-2]]


class TestSafeGatherVoronoiLabels:
    """Out-of-env (-1) bins must read label -1 (unreachable), not wrap."""

    @pytest.fixture
    def voronoi_env(self):
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        env = Environment.from_samples(
            np.column_stack([xx.ravel(), yy.ravel()]), bin_size=5.0
        )
        left_bin = int(env.bin_at(np.array([[10.0, 50.0]]))[0])
        right_bin = int(env.bin_at(np.array([[90.0, 50.0]]))[0])
        labels = geodesic_voronoi_labels(env, [left_bin, right_bin])
        return env, labels

    def test_detect_boundary_crossings_minus_one_bin_is_unreachable(self, voronoi_env):
        env, labels = voronoi_env

        left_side = int(env.bin_at(np.array([[20.0, 50.0]]))[0])
        right_side = int(env.bin_at(np.array([[80.0, 50.0]]))[0])
        assert labels[left_side] == 0
        assert labels[right_side] == 1

        # All on the left side, with an out-of-env (-1) sample. A wrapped -1 ->
        # last-bin label could differ from 0 and create a phantom crossing.
        position_bins = np.array(
            [left_side, left_side, -1, left_side, left_side],
            dtype=np.int64,
        )
        times = np.linspace(0.0, 1.0, len(position_bins))

        crossing_times, crossing_dirs = detect_boundary_crossings(
            position_bins, labels, times
        )

        assert crossing_times.size == 0
        assert crossing_dirs.shape == (0, 2)

    def test_compute_decision_analysis_goal_labels_minus_one_for_out_of_env(
        self,
    ):
        # T-maze-like env with start, center decision region, and two goals.
        stem_x = np.linspace(45, 55, 5)
        stem_y = np.linspace(0, 50, 20)
        sxx, syy = np.meshgrid(stem_x, stem_y)
        bar_x = np.linspace(0, 100, 40)
        bar_y = np.linspace(50, 60, 5)
        bxx, byy = np.meshgrid(bar_x, bar_y)
        positions = np.vstack(
            [
                np.column_stack([sxx.ravel(), syy.ravel()]),
                np.column_stack([bxx.ravel(), byy.ravel()]),
            ]
        )
        env = Environment.from_samples(positions, bin_size=5.0)
        region_builder = Regions(env.regions.values())
        region_builder.add("center", point=(50.0, 55.0))
        region_builder.add("left", point=(10.0, 55.0))
        region_builder.add("right", point=(90.0, 55.0))
        env = env.with_regions(region_builder)

        # Trajectory: up the stem, into the center, with one sample that is
        # genuinely outside the environment (maps to -1).
        traj = np.array(
            [
                [50.0, 5.0],
                [50.0, 25.0],
                [50.0, 45.0],
                [50.0, 55.0],  # center
                [1000.0, 1000.0],  # out of env -> bin -1
                [60.0, 55.0],
            ]
        )
        # Document why the -1 is there.
        assert int(env.bin_at(np.array([[1000.0, 1000.0]]))[0]) == -1

        times = np.linspace(0.0, 5.0, len(traj))

        result = compute_decision_analysis(
            env,
            times,
            traj,
            decision_region="center",
            goal_regions=["left", "right"],
            max_gap=None,
            min_speed=5.0,
        )

        assert result.boundary is not None
        goal_labels = result.boundary.goal_labels
        # The out-of-env sample (index 4) must read -1, not the last goal label.
        assert goal_labels[4] == -1


# =============================================================================
# Family 2 — dt guards
# =============================================================================


class TestDtGuards:
    """Duplicate / non-finite timestamps must raise, not produce inf/nan."""

    def test_segment_by_velocity_duplicate_timestamp_raises(self):
        positions = np.column_stack([np.linspace(0, 100, 11), np.zeros(11)])
        times = duplicate_timestamps(n=11, total=5.0)

        with pytest.raises(ValueError, match="times"):
            segment_by_velocity(times, positions, min_speed=2.0)

    def test_detect_trials_duplicate_timestamp_raises(self):
        env = TestSafeGatherRegions._grid_with_corner_regions()
        positions = env.bin_centers[[0, 0, 100, 100, 100]]
        times = np.array([0.0, 1.0, 2.0, 2.0, 3.0])  # a repeated frame time

        with pytest.raises(ValueError, match="times"):
            detect_trials(env, times, positions, regions=["start", "goal"])

    def test_approach_rate_duplicate_timestamp_raises(self):
        positions = np.column_stack([np.linspace(0, 50, 11), np.zeros(11)])
        times = duplicate_timestamps(n=11, total=5.0)
        goal = np.array([100.0, 0.0])

        with pytest.raises(ValueError, match="times"):
            approach_rate(times, positions, goal)

    def test_approach_rate_nonfinite_times_raises(self):
        positions = np.column_stack([np.linspace(0, 50, 11), np.zeros(11)])
        times = nonfinite_timestamps(n=11, total=5.0)
        goal = np.array([100.0, 0.0])

        with pytest.raises(ValueError, match="times"):
            approach_rate(times, positions, goal)

    def test_mean_square_displacement_all_duplicate_times_raises(self):
        positions = np.column_stack([np.linspace(0, 50, 11), np.zeros(11)])
        # All timestamps identical -> median dt == 0.
        times = np.zeros(11)

        with pytest.raises(ValueError, match="increasing"):
            mean_square_displacement(times, positions, metric="euclidean")

    def test_mean_square_displacement_repeated_timestamp_raises(self):
        positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
        times = np.linspace(0.0, 10.0, 21)
        times[5] = times[4]

        with pytest.raises(ValueError, match="strictly increasing"):
            mean_square_displacement(times, positions, metric="euclidean", max_tau=5.0)


# =============================================================================
# Family 3 — path efficiency: unreachable goal + metric consistency
# =============================================================================


class TestPathEfficiencyCorrectness:
    """Unreachable goals report inf; time efficiency follows the chosen metric."""

    @pytest.fixture
    def disconnected_env(self):
        """Environment with two disconnected components (a gap between them)."""
        left = np.column_stack(
            [
                np.repeat(np.linspace(0, 20, 5), 5),
                np.tile(np.linspace(0, 20, 5), 5),
            ]
        )
        right = np.column_stack(
            [
                np.repeat(np.linspace(80, 100, 5), 5),
                np.tile(np.linspace(0, 20, 5), 5),
            ]
        )
        positions = np.vstack([left, right])
        env = Environment.from_samples(positions, bin_size=5.0)
        return env, left, right

    def test_compute_path_efficiency_unreachable_goal_reports_inf(
        self, disconnected_env
    ):
        env, left, right = disconnected_env

        # Travel only within the left component; goal is in the disconnected
        # right component -> geodesic shortest length is inf.
        traj = left[:5]
        times = np.linspace(0.0, 2.0, len(traj))
        goal = right[0]

        result = compute_path_efficiency(env, times, traj, goal, metric="geodesic")

        assert np.isinf(result.shortest_length)
        assert np.isnan(result.efficiency)
        # summary()/str() must not raise on inf/nan.
        assert isinstance(result.summary(), dict)
        assert isinstance(str(result), str)

    def test_compute_path_efficiency_time_efficiency_is_geodesic(self):
        # L-shaped environment: geodesic optimal distance >> Euclidean straight
        # line because a straight path cuts across the empty corner.
        arm1 = np.column_stack(
            [
                np.repeat(np.linspace(0, 60, 13), 3),
                np.tile(np.linspace(0, 10, 3), 13),
            ]
        )
        arm2 = np.column_stack(
            [
                np.tile(np.linspace(50, 60, 3), 13),
                np.repeat(np.linspace(0, 60, 13), 3),
            ]
        )
        positions = np.vstack([arm1, arm2])
        env = Environment.from_samples(positions, bin_size=5.0)

        start = np.array([5.0, 5.0])
        goal = np.array([55.0, 55.0])

        from neurospatial.behavior.navigation import shortest_path_length

        geodesic_dist = shortest_path_length(env, start, goal, metric="geodesic")
        euclidean_dist = float(np.linalg.norm(goal - start))
        # Confirm the fixture: geodesic detour is meaningfully longer.
        assert geodesic_dist > euclidean_dist * 1.2

        # Build a trajectory along the L and a matching time base.
        traj = np.vstack(
            [
                np.column_stack([np.linspace(5, 55, 25), np.full(25, 5.0)]),
                np.column_stack([np.full(25, 55.0), np.linspace(5, 55, 25)]),
            ]
        )
        times = np.linspace(0.0, 10.0, len(traj))
        reference_speed = 20.0

        result = compute_path_efficiency(
            env,
            times,
            traj,
            goal,
            metric="geodesic",
            reference_speed=reference_speed,
        )

        # The reported time efficiency must match the geodesic optimal distance,
        # not the (shorter) Euclidean straight line.
        expected_geodesic = time_efficiency(
            times,
            traj,
            reference_speed=reference_speed,
            optimal_distance=result.shortest_length,
        )
        euclidean_value = time_efficiency(
            times,
            traj,
            reference_speed=reference_speed,
            optimal_distance=euclidean_dist,
        )

        assert_allclose(result.time_efficiency, expected_geodesic)
        assert not np.isclose(result.time_efficiency, euclidean_value)

    def test_time_efficiency_infinite_optimal_distance_is_nan(self):
        positions = np.column_stack([np.linspace(0, 50, 11), np.zeros(11)])
        times = np.linspace(0.0, 5.0, 11)

        eff = time_efficiency(
            times,
            positions,
            reference_speed=10.0,
            optimal_distance=np.inf,
        )

        assert np.isnan(eff)
