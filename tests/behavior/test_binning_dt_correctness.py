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

from neurospatial import Environment
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
    detect_runs_between_regions,
    segment_by_velocity,
    segment_trials,
)
from neurospatial.behavior.trajectory import mean_square_displacement

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
        from neurospatial.ops.binning import regions_to_mask

        assert not regions_to_mask(env, [region_name])[outside_bin]

        # Surrounding samples are outside the region; the middle sample is -1
        # (out of env). A wrapped -1 -> last_bin would be *inside* the region,
        # creating a spurious entry+exit crossing.
        position_bins = np.array(
            [outside_bin, outside_bin, -1, outside_bin, outside_bin],
            dtype=np.int64,
        )
        times = np.linspace(0.0, 1.0, len(position_bins))

        crossings = detect_region_crossings(
            position_bins, times, env, region_name=region_name
        )

        assert crossings == []

    def test_detect_runs_between_regions_excludes_out_of_env_from_target(
        self,
    ):
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        env = Environment.from_samples(
            np.column_stack([xx.ravel(), yy.ravel()]), bin_size=5.0
        )
        last_bin = env.n_bins - 1
        env.regions.add("source", polygon=Point(*env.bin_centers[0]).buffer(6.0))
        env.regions.add("target", polygon=Point(*env.bin_centers[last_bin]).buffer(6.0))

        # Start inside source, exit, then a single out-of-env (-1) sample.
        # A wrapped -1 -> last_bin would be inside 'target', wrongly marking
        # the run success=True.
        source_bin = 0
        neutral_bin = 100  # not in source, not in target
        from neurospatial.ops.binning import regions_to_mask

        assert not regions_to_mask(env, ["target"])[neutral_bin]
        assert not regions_to_mask(env, ["source"])[neutral_bin]

        position_bins = np.array(
            [source_bin, source_bin, neutral_bin, neutral_bin, -1],
            dtype=np.int64,
        )
        times = np.linspace(0.0, 1.0, len(position_bins))

        runs = detect_runs_between_regions(
            position_bins,
            times,
            env,
            source="source",
            target="target",
            min_duration=0.0,
            max_duration=100.0,
        )

        # The out-of-env sample must not count as reaching the target.
        assert all(not run.success for run in runs)

    def test_detect_runs_speed_filter_ignores_out_of_env_bin(self):
        """min_speed filter must not treat a -1 sample as bin_centers[-1].

        A run whose valid samples sit on a single bin has zero true speed and
        should be dropped by ``min_speed > 0``. With the old code the off-env
        (-1) sample wrapped to ``bin_centers[-1]`` (a far corner), injecting a
        large displacement that inflated the mean speed above the threshold and
        wrongly kept the run.
        """
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        env = Environment.from_samples(
            np.column_stack([xx.ravel(), yy.ravel()]), bin_size=5.0
        )
        last_bin = env.n_bins - 1
        env.regions.add("source", polygon=Point(*env.bin_centers[0]).buffer(6.0))
        env.regions.add("target", polygon=Point(*env.bin_centers[last_bin]).buffer(6.0))

        # Source bin, then a single neutral bin held constant (zero true speed),
        # with one off-env (-1) sample. The wrapped -1 -> last_bin sits near the
        # opposite corner, so the buggy speed estimate is large.
        neutral_bin = 100
        from neurospatial.ops.binning import regions_to_mask

        assert not regions_to_mask(env, ["target"])[neutral_bin]
        assert not regions_to_mask(env, ["source"])[neutral_bin]

        position_bins = np.array(
            [0, neutral_bin, neutral_bin, -1, neutral_bin],
            dtype=np.int64,
        )
        # Small dt so a wrapped -1 displacement would read as a high speed.
        times = np.linspace(0.0, 0.4, len(position_bins))

        # True speed over valid samples is 0 (all neutral_bin). With min_speed
        # above 0, the run must be filtered out -- and the off-env sample must
        # not raise or inflate the speed.
        runs = detect_runs_between_regions(
            position_bins,
            times,
            env,
            source="source",
            target="target",
            min_duration=0.0,
            max_duration=100.0,
            min_speed=10.0,
        )

        assert runs == []

    def test_detect_runs_speed_filter_drops_run_with_no_usable_speed(self):
        """min_speed set + no usable consecutive valid pair => run is DROPPED.

        Repro of the reported bug: a run slice like ``[source, -1, neutral]``
        has no consecutive on-environment pair from which to estimate speed.
        The run must not silently pass a speed gate it never satisfied -- it is
        dropped, not kept.
        """
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        env = Environment.from_samples(
            np.column_stack([xx.ravel(), yy.ravel()]), bin_size=5.0
        )
        last_bin = env.n_bins - 1
        env.regions.add("source", polygon=Point(*env.bin_centers[0]).buffer(6.0))
        env.regions.add("target", polygon=Point(*env.bin_centers[last_bin]).buffer(6.0))

        neutral_bin = 100
        from neurospatial.ops.binning import regions_to_mask

        assert not regions_to_mask(env, ["target"])[neutral_bin]
        assert not regions_to_mask(env, ["source"])[neutral_bin]

        # Exit source, then an off-env (-1) sample between the only two
        # on-env samples => no consecutive valid pair, no usable speed.
        position_bins = np.array(
            [0, neutral_bin, -1, neutral_bin],
            dtype=np.int64,
        )
        times = np.linspace(0.0, 0.4, len(position_bins))

        runs = detect_runs_between_regions(
            position_bins,
            times,
            env,
            source="source",
            target="target",
            min_duration=0.0,
            max_duration=100.0,
            min_speed=10.0,
        )

        assert runs == []

    def test_detect_runs_speed_filter_fully_valid_runs(self):
        """A fully-valid fast run passes; a fully-valid slow run is filtered."""
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        env = Environment.from_samples(
            np.column_stack([xx.ravel(), yy.ravel()]), bin_size=5.0
        )
        last_bin = env.n_bins - 1
        env.regions.add("source", polygon=Point(*env.bin_centers[0]).buffer(6.0))
        env.regions.add("target", polygon=Point(*env.bin_centers[last_bin]).buffer(6.0))

        # Exit source at the first step, then traverse distinct neutral bins
        # (a steady move each frame) so the run has a well-defined mean speed.
        from neurospatial.ops.binning import regions_to_mask

        neutral_bins = [200, 201, 202]
        for b in neutral_bins:
            assert not regions_to_mask(env, ["source"])[b]
            assert not regions_to_mask(env, ["target"])[b]

        position_bins = np.array([0, *neutral_bins], dtype=np.int64)
        # The run begins at the source-exit sample, so its path is the neutral
        # bins only (index 0 is the last in-source sample, excluded). Mean speed
        # is taken over the consecutive neutral->neutral pairs.
        run_path_bins = np.array(neutral_bins, dtype=np.int64)
        step_dists = np.linalg.norm(
            np.diff(env.bin_centers[run_path_bins], axis=0), axis=1
        )

        # dt of 0.1 s per step; mean speed = mean(step_dists / 0.1).
        times = np.array([0.0, 0.1, 0.2, 0.3])
        mean_speed = float(np.mean(step_dists / 0.1))
        slow_threshold = mean_speed * 0.5
        high_threshold = mean_speed * 2.0

        fast_runs = detect_runs_between_regions(
            position_bins,
            times,
            env,
            source="source",
            target="target",
            min_duration=0.0,
            max_duration=100.0,
            min_speed=slow_threshold,
        )
        assert len(fast_runs) == 1  # mean speed above threshold => kept

        slow_runs = detect_runs_between_regions(
            position_bins,
            times,
            env,
            source="source",
            target="target",
            min_duration=0.0,
            max_duration=100.0,
            min_speed=high_threshold,
        )
        assert slow_runs == []  # mean speed below threshold => filtered

    def test_segment_trials_out_of_env_not_counted_in_end_region(self):
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        env = Environment.from_samples(
            np.column_stack([xx.ravel(), yy.ravel()]), bin_size=5.0
        )
        last_bin = env.n_bins - 1
        env.regions.add("start", polygon=Point(*env.bin_centers[0]).buffer(6.0))
        env.regions.add("goal", polygon=Point(*env.bin_centers[last_bin]).buffer(6.0))

        start_bin = 0
        neutral_bin = 100

        # Trajectory with an out-of-env (-1) sample in the middle. The same
        # trajectory with the -1 sample removed must produce identical trials.
        bins_with_oob = np.array(
            [start_bin, neutral_bin, -1, neutral_bin, start_bin, neutral_bin],
            dtype=np.int64,
        )
        times_with_oob = np.linspace(0.0, 5.0, len(bins_with_oob))

        keep = bins_with_oob != -1
        bins_without_oob = bins_with_oob[keep]
        times_without_oob = times_with_oob[keep]

        trials_with = segment_trials(
            bins_with_oob,
            times_with_oob,
            env,
            start_region="start",
            end_regions=["goal"],
            min_duration=0.1,
            max_duration=100.0,
        )
        trials_without = segment_trials(
            bins_without_oob,
            times_without_oob,
            env,
            start_region="start",
            end_regions=["goal"],
            min_duration=0.1,
            max_duration=100.0,
        )

        # No trial should reach the goal because of the wrapped -1 sample.
        assert all(t.end_region != "goal" for t in trials_with)
        # Same end-region outcomes with and without the out-of-env sample.
        assert [t.end_region for t in trials_with] == [
            t.end_region for t in trials_without
        ]


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

        assert crossing_times == []
        assert crossing_dirs == []

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
        env.regions.add("center", point=(50.0, 55.0))
        env.regions.add("left", point=(10.0, 55.0))
        env.regions.add("right", point=(90.0, 55.0))

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
            traj,
            times,
            decision_region="center",
            goal_regions=["left", "right"],
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
            segment_by_velocity(positions, times, min_speed=2.0)

    def test_detect_runs_between_regions_duplicate_timestamp_raises(self):
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        env = Environment.from_samples(
            np.column_stack([xx.ravel(), yy.ravel()]), bin_size=5.0
        )
        env.regions.add("source", polygon=Point(*env.bin_centers[0]).buffer(6.0))
        env.regions.add(
            "target",
            polygon=Point(*env.bin_centers[env.n_bins - 1]).buffer(6.0),
        )

        source_bin = 0
        mid_bin = 100
        position_bins = np.array(
            [source_bin, source_bin, mid_bin, mid_bin, mid_bin],
            dtype=np.int64,
        )
        # Duplicate frame time inside the run slice.
        times = np.array([0.0, 1.0, 2.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="times"):
            detect_runs_between_regions(
                position_bins,
                times,
                env,
                source="source",
                target="target",
                min_duration=0.0,
                max_duration=100.0,
                min_speed=1.0,
            )

    def test_approach_rate_duplicate_timestamp_raises(self):
        positions = np.column_stack([np.linspace(0, 50, 11), np.zeros(11)])
        times = duplicate_timestamps(n=11, total=5.0)
        goal = np.array([100.0, 0.0])

        with pytest.raises(ValueError, match="times"):
            approach_rate(positions, times, goal)

    def test_approach_rate_nonfinite_times_raises(self):
        positions = np.column_stack([np.linspace(0, 50, 11), np.zeros(11)])
        times = nonfinite_timestamps(n=11, total=5.0)
        goal = np.array([100.0, 0.0])

        with pytest.raises(ValueError, match="times"):
            approach_rate(positions, times, goal)

    def test_mean_square_displacement_all_duplicate_times_raises(self):
        positions = np.column_stack([np.linspace(0, 50, 11), np.zeros(11)])
        # All timestamps identical -> median dt == 0.
        times = np.zeros(11)

        with pytest.raises(ValueError, match="increasing"):
            mean_square_displacement(positions, times, metric="euclidean")

    def test_mean_square_displacement_tolerates_single_duplicate(self):
        positions = np.column_stack([np.linspace(0, 100, 21), np.zeros(21)])
        times = np.linspace(0.0, 10.0, 21)
        # One duplicated timestamp; median step stays positive.
        times[5] = times[4]

        result = mean_square_displacement(
            positions, times, metric="euclidean", max_tau=5.0
        )

        assert len(result.lags) > 0
        assert np.all(np.isfinite(result.msd))
        # MSD increases with lag for ballistic-like motion.
        assert result.msd[-1] > result.msd[0]


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

        result = compute_path_efficiency(env, traj, times, goal, metric="geodesic")

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
            traj,
            times,
            goal,
            metric="geodesic",
            reference_speed=reference_speed,
        )

        # The reported time efficiency must match the geodesic optimal distance,
        # not the (shorter) Euclidean straight line.
        expected_geodesic = time_efficiency(
            traj,
            times,
            reference_speed=reference_speed,
            optimal_distance=result.shortest_length,
        )
        euclidean_value = time_efficiency(
            traj,
            times,
            reference_speed=reference_speed,
            optimal_distance=euclidean_dist,
        )

        assert_allclose(result.time_efficiency, expected_geodesic)
        assert not np.isclose(result.time_efficiency, euclidean_value)

    def test_time_efficiency_infinite_optimal_distance_is_nan(self):
        positions = np.column_stack([np.linspace(0, 50, 11), np.zeros(11)])
        times = np.linspace(0.0, 5.0, 11)

        eff = time_efficiency(
            positions,
            times,
            reference_speed=10.0,
            optimal_distance=np.inf,
        )

        assert np.isnan(eff)
