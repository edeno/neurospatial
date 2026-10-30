"""Segmenting a decoded candidate event into smooth trajectories."""

from __future__ import annotations

import pickle

import networkx as nx
import numpy as np
import pytest
from environment_helpers import renumbered, strips_1d

from neurospatial import Environment
from neurospatial.replay import (
    ReplaySegmentsResult,
    replay_segments,
    trajectory_metrics,
)

DT = 0.02


@pytest.fixture(scope="module")
def track():
    """A 100 cm straight track in ten 10 cm bins, centers 5, 15, ..., 95."""
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=10.0, units="cm"
    )


@pytest.fixture(scope="module")
def l_track():
    """An L-shaped track: 50 cm east, then 50 cm north, in 10 cm bins.

    Bin 3 (center (35, 0)) and bin 6 (center (50, 15)) are 30 cm apart
    along the track and sqrt(450) = 21.2 cm apart in a straight line.
    """
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(50.0, 0.0))
    graph.add_node(2, pos=(50.0, 50.0))
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    return Environment.from_graph(
        graph, edge_order=[(0, 1), (1, 2)], edge_spacing=0.0, bin_size=10.0, units="cm"
    )


@pytest.fixture(scope="module")
def arena():
    """A 50 x 50 cm arena of 10 cm square bins."""
    x, y = np.meshgrid(np.arange(0.5, 50.0), np.arange(0.5, 50.0))
    return Environment.from_samples(
        np.c_[x.ravel(), y.ravel()], bin_size=10.0, units="cm"
    )


def _times(n, dt=DT, start=0.0):
    return start + np.arange(n) * dt


def _sweep(env, bins):
    """One posterior row per time bin with all mass in ``bins[t]``."""
    return np.eye(env.n_bins)[np.asarray(bins)]


def _segments(result):
    return result.segments.tolist()


class TestJumpRules:
    """MAP bins 1, 2, 3, 4, 5, then a 40 cm jump to bin 9, then 8 and 7."""

    BINS = (1, 2, 3, 4, 5, 9, 8, 7)

    def test_between_splits_without_losing_a_bin(self, track):
        result = replay_segments(
            track, _sweep(track, self.BINS), _times(8), max_jump=20.0
        )
        assert _segments(result) == [[0, 5], [5, 8]]
        assert result.trajectory_time_bins.all()

    def test_drop_earlier_rejects_the_bin_before_the_jump(self, track):
        result = replay_segments(
            track,
            _sweep(track, self.BINS),
            _times(8),
            max_jump=20.0,
            jump_rule="drop_earlier",
        )
        assert _segments(result) == [[0, 4], [5, 8]]
        np.testing.assert_array_equal(np.flatnonzero(~result.trajectory_time_bins), [4])

    def test_a_lone_outlier_costs_its_earlier_neighbor(self, track):
        # Jumps 10, 10, 60, 50, 10: bin 2 carries the jump into the outlier
        # and the outlier carries the jump out of it.
        posterior = _sweep(track, [1, 2, 3, 9, 4, 5])
        between = replay_segments(track, posterior, _times(6), max_jump=20.0)
        earlier = replay_segments(
            track, posterior, _times(6), max_jump=20.0, jump_rule="drop_earlier"
        )
        assert _segments(between) == [[0, 3], [3, 4], [4, 6]]
        assert _segments(earlier) == [[0, 2], [4, 6]]

    def test_the_last_bin_is_not_rejected_by_a_jump(self, track):
        result = replay_segments(
            track,
            _sweep(track, [1, 2, 3, 9]),
            _times(4),
            max_jump=20.0,
            jump_rule="drop_earlier",
        )
        assert _segments(result) == [[0, 2], [3, 4]]

    def test_a_jump_equal_to_max_jump_splits(self, track):
        posterior = _sweep(track, [1, 2, 4])  # jumps 10 and 20
        at = replay_segments(track, posterior, _times(3), max_jump=20.0)
        rounded = replay_segments(track, posterior, _times(3), max_jump=20.0 + 1e-12)
        above = replay_segments(track, posterior, _times(3), max_jump=20.5)
        assert _segments(at) == _segments(rounded) == [[0, 2], [2, 3]]
        assert _segments(above) == [[0, 3]]

    def test_posterior_mean_jumps(self, track):
        # Means 15, 25, 35 then 47 (0.6 in bin 3, 0.4 in bin 6): a 12 cm
        # mean jump although the MAP stays in bin 3.
        posterior = _sweep(track, [1, 2, 3, 3]).astype(float)
        posterior[3] = 0.0
        posterior[3, [3, 6]] = [0.6, 0.4]
        by_map = replay_segments(track, posterior, _times(4), max_jump=11.0)
        by_mean = replay_segments(
            track, posterior, _times(4), max_jump=11.0, method="mean"
        )
        assert _segments(by_map) == [[0, 4]]
        assert _segments(by_mean) == [[0, 3], [3, 4]]


class TestGaps:
    def test_one_gap_bin_bridges_an_outlier(self, track):
        posterior = _sweep(track, [1, 2, 9, 3, 4])
        no_gap = replay_segments(track, posterior, _times(5), max_jump=20.0)
        one_gap = replay_segments(
            track, posterior, _times(5), max_jump=20.0, max_gap_bins=1
        )
        assert _segments(no_gap) == [[0, 2], [2, 3], [3, 5]]
        assert _segments(one_gap) == [[0, 5]]
        np.testing.assert_array_equal(
            one_gap.trajectory_time_bins, [True, True, False, True, True]
        )
        # 15 -> 25, across the gap to 35, then 45.
        np.testing.assert_allclose(one_gap.path_lengths, [30.0])
        np.testing.assert_allclose(one_gap.start_to_end_distances, [30.0])

    def test_two_outliers_need_two_gap_bins(self, track):
        posterior = _sweep(track, [1, 2, 9, 9, 3, 4])
        one_gap = replay_segments(
            track, posterior, _times(6), max_jump=20.0, max_gap_bins=1
        )
        two_gaps = replay_segments(
            track, posterior, _times(6), max_jump=20.0, max_gap_bins=2
        )
        assert _segments(one_gap) == [[0, 2], [2, 4], [4, 6]]
        assert _segments(two_gaps) == [[0, 6]]
        np.testing.assert_array_equal(
            np.flatnonzero(~two_gaps.trajectory_time_bins), [2, 3]
        )

    def test_the_jump_across_the_gap_must_pass(self, track):
        posterior = _sweep(track, [1, 2, 9, 4, 5])  # 25 -> 45 across the gap
        result = replay_segments(
            track, posterior, _times(5), max_jump=20.0, max_gap_bins=1
        )
        assert _segments(result) == [[0, 2], [2, 3], [3, 5]]

    def test_a_gap_can_skip_a_rejected_bin(self, track):
        posterior = _sweep(track, [1, 2, 3, 4, 5]).astype(float)
        posterior[2] = 1.0 / track.n_bins  # flat: peak 0.1
        result = replay_segments(
            track,
            posterior,
            _times(5),
            max_jump=25.0,
            max_gap_bins=1,
            min_peak_probability=0.5,
        )
        assert _segments(result) == [[0, 5]]
        assert not result.trajectory_time_bins[2]


class TestTimeBinMasks:
    @pytest.fixture(scope="class")
    def blurred(self, track):
        """Bins 1 to 5, with the middle row 0.6 in bin 3 and 0.4 in bin 6.

        That row's MAP is bin 3, its peak 0.6, its spread
        sqrt(0.6 * 0.4) * 30 = 14.7 cm.
        """
        posterior = _sweep(track, [1, 2, 3, 4, 5]).astype(float)
        posterior[2] = 0.0
        posterior[2, [3, 6]] = [0.6, 0.4]
        return posterior

    def test_max_spread_rejects_a_blurred_bin(self, track, blurred):
        kept = replay_segments(track, blurred, _times(5), max_jump=20.0)
        masked = replay_segments(
            track, blurred, _times(5), max_jump=20.0, max_spread=10.0
        )
        assert _segments(kept) == [[0, 5]]
        assert _segments(masked) == [[0, 2], [3, 5]]
        assert not masked.trajectory_time_bins[2]

    def test_min_peak_probability_is_strict(self, track, blurred):
        at = replay_segments(
            track, blurred, _times(5), max_jump=20.0, min_peak_probability=0.6
        )
        below = replay_segments(
            track, blurred, _times(5), max_jump=20.0, min_peak_probability=0.59
        )
        assert _segments(at) == [[0, 2], [3, 5]]
        assert _segments(below) == [[0, 5]]

    def test_joint_share_uses_the_mass_as_given(self, track):
        # One template's share of a joint decode: half of each row.
        share = 0.5 * _sweep(track, [1, 2, 3, 4])
        full = replay_segments(track, share * 2, _times(4), max_jump=20.0)
        kept = replay_segments(
            track, share, _times(4), max_jump=20.0, min_peak_probability=0.4
        )
        rejected = replay_segments(
            track, share, _times(4), max_jump=20.0, min_peak_probability=0.5
        )
        assert _segments(kept) == _segments(full) == [[0, 4]]
        np.testing.assert_allclose(kept.dispersions, full.dispersions)
        assert rejected.n_segments == 0


class TestMerge:
    @pytest.fixture(scope="class")
    def gapped(self, track):
        """Bins 1, 2, 3, a flat row, then 4, 5, 6.

        With the flat row rejected, the pieces are rows 0-2 and 4-6; their
        facing bins are 2 time bins (40 ms) and 10 cm apart.
        """
        posterior = _sweep(track, [1, 2, 3, 3, 4, 5, 6]).astype(float)
        posterior[3] = 1.0 / track.n_bins
        return posterior

    def _merge(self, track, posterior, time_gap, distance):
        return replay_segments(
            track,
            posterior,
            _times(len(posterior)),
            max_jump=20.0,
            min_peak_probability=0.5,
            merge_max_time_gap=time_gap,
            merge_max_distance=distance,
        )

    def test_merges_when_both_gaps_are_small(self, track, gapped):
        result = self._merge(track, gapped, 0.05, 20.0)
        assert _segments(result) == [[0, 7]]
        assert result.time_bin_counts.tolist() == [7]
        np.testing.assert_array_equal(np.flatnonzero(~result.trajectory_time_bins), [3])
        np.testing.assert_allclose(result.path_lengths, [50.0])

    def test_no_merge_when_only_the_time_gap_is_small(self, track, gapped):
        result = self._merge(track, gapped, 0.05, 10.0)
        assert _segments(result) == [[0, 3], [4, 7]]

    def test_no_merge_when_only_the_distance_is_small(self, track, gapped):
        result = self._merge(track, gapped, 0.04, 20.0)
        assert _segments(result) == [[0, 3], [4, 7]]

    def test_merges_a_chain_of_pieces(self, track):
        posterior = _sweep(track, [1, 2, 2, 3, 4, 4, 5, 6]).astype(float)
        posterior[[2, 5]] = 1.0 / track.n_bins
        result = self._merge(track, posterior, 0.05, 20.0)
        assert _segments(result) == [[0, 8]]

    def test_drop_earlier_and_merge_bridge_an_outlier(self, track):
        """Widloski & Foster (2022) by hand, posterior means, 5 ms steps.

        Means 15, 25, 35, 95, 45, 55, 65. Jumps 10, 10, 60, 50, 10, 10:
        rows 2 and 3 are rejected, leaving rows 0-1 and 4-6, 15 ms and
        20 cm apart (25 -> 45).
        """
        posterior = _sweep(track, [1, 2, 3, 9, 4, 5, 6])
        result = replay_segments(
            track,
            posterior,
            _times(7, dt=0.005),
            max_jump=20.0,
            method="mean",
            jump_rule="drop_earlier",
            merge_max_time_gap=0.05,
            merge_max_distance=25.0,
        )
        assert _segments(result) == [[0, 7]]
        np.testing.assert_array_equal(
            np.flatnonzero(~result.trajectory_time_bins), [2, 3]
        )
        # Dispersion over the trajectory bins only: 15, 25, 45, 55, 65.
        positions = np.array([15.0, 25.0, 45.0, 55.0, 65.0])
        expected = np.sqrt(np.mean((positions - positions.mean()) ** 2))
        np.testing.assert_allclose(result.dispersions, [expected])
        np.testing.assert_allclose(result.path_lengths, [50.0])
        assert result.durations[0] == pytest.approx(0.03)


class TestFilters:
    def test_min_duration_is_inclusive_despite_rounding(self, track):
        times = _times(12, dt=0.005, start=1234.567)
        # Six 5 ms steps after t0 = 1234.567 round to just under 30 ms.
        assert times[6] - times[0] < 0.03
        posterior = _sweep(track, [1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6])
        result = replay_segments(
            track, posterior, times, max_jump=20.0, min_duration=0.03
        )
        # Rows 0-6 last 30 ms; rows 7-11 last 20 ms.
        assert _segments(result) == [[0, 7]]
        assert not result.trajectory_time_bins[7:].any()

    def test_min_time_bins(self, track):
        posterior = _sweep(track, [1, 2, 3, 4, 5, 9, 8, 7])
        three = replay_segments(
            track, posterior, _times(8), max_jump=20.0, min_time_bins=3
        )
        four = replay_segments(
            track, posterior, _times(8), max_jump=20.0, min_time_bins=4
        )
        assert _segments(three) == [[0, 5], [5, 8]]
        assert _segments(four) == [[0, 5]]

    def test_min_dispersion(self, track):
        # Positions 15..55 (dispersion sqrt(200) = 14.1) and 95, 85, 75
        # (dispersion sqrt(200 / 3) = 8.2).
        posterior = _sweep(track, [1, 2, 3, 4, 5, 9, 8, 7])
        result = replay_segments(
            track, posterior, _times(8), max_jump=20.0, min_dispersion=12.0
        )
        unfiltered = replay_segments(track, posterior, _times(8), max_jump=20.0)
        np.testing.assert_allclose(
            unfiltered.dispersions, [np.sqrt(200.0), np.sqrt(200.0 / 3)]
        )
        assert _segments(result) == [[0, 5]]
        assert not result.trajectory_time_bins[5:].any()

    def test_filters_apply_before_choosing_the_longest(self, track):
        # The longer piece (bins 4, 4, 4, 4) never moves.
        posterior = _sweep(track, [1, 2, 3, 9, 9, 9, 9])
        result = replay_segments(
            track, posterior, _times(7), max_jump=20.0, min_dispersion=5.0
        )
        assert _segments(result) == [[0, 3]]
        assert result.longest == 0


class TestLongest:
    def test_longest_spans_the_most_time_bins(self, track):
        result = replay_segments(
            track, _sweep(track, [1, 2, 9, 8, 7, 6]), _times(6), max_jump=20.0
        )
        assert result.longest == 1
        assert result.rows() == slice(2, 6)
        assert result.rows(0) == slice(0, 2)
        assert result.rows(-1) == slice(2, 6)

    def test_a_tie_goes_to_the_earliest(self, track):
        result = replay_segments(
            track, _sweep(track, [1, 2, 3, 9, 8, 7]), _times(6), max_jump=20.0
        )
        assert result.time_bin_counts.tolist() == [3, 3]
        assert result.longest == 0

    def test_rows_select_the_segment_for_trajectory_metrics(self, track):
        posterior = _sweep(track, [1, 2, 9, 8, 7, 6])
        times = _times(6)
        result = replay_segments(track, posterior, times, max_jump=20.0)
        rows = result.rows()
        metrics = trajectory_metrics(track, posterior[rows], times[rows])
        assert metrics.map_path_length == pytest.approx(result.path_lengths[1])
        assert metrics.duration == pytest.approx(result.durations[1])
        assert metrics.dispersion == pytest.approx(result.dispersions[1])

    def test_no_segment(self, track):
        result = replay_segments(
            track, _sweep(track, [1, 2, 3]), _times(3), max_jump=20.0, min_time_bins=4
        )
        assert result.segments.shape == (0, 2)
        assert result.longest is None
        assert not result.trajectory_time_bins.any()
        summary = result.summary()
        assert summary["n_segments"] == 0
        assert np.isnan(summary["longest_duration"])
        with pytest.raises(ValueError, match="no segment"):
            result.rows()

    def test_rows_refuses_a_missing_segment(self, track):
        result = replay_segments(
            track, _sweep(track, [1, 2, 3]), _times(3), max_jump=20.0
        )
        with pytest.raises(IndexError, match="does not exist"):
            result.rows(1)


class TestMalloryByHand:
    def test_longest_smooth_segment(self, track):
        """``filter_candidate_events.m`` on one template's share, by hand.

        Rows hold 0.8 of each time bin's mass (the other template has the
        rest). Chance over both templates is 1 / 20, so bins need a peak
        above 5 / 20 = 0.25; the jump threshold is 40% of the 100 cm track.

        - Row 3 is flat (peak 0.08): rejected.
        - Rows 7 -> 8 jump 60 cm: split, no bin lost.
        - Pieces 0-2, 4-7, 8-10. Pieces 0-2 and 4-7 face each other 10 ms
          and 20 cm apart (35 -> 55): merged. 4-7 and 8-10 are 60 cm apart.
        - Rows 0-7 last 35 ms (kept); rows 8-10 last 10 ms (< 30 ms, dropped).
        """
        bins = [1, 2, 3, 3, 5, 6, 7, 8, 2, 3, 4]
        share = 0.8 * _sweep(track, bins)
        share[3] = 0.8 / track.n_bins
        track_length = 100.0
        result = replay_segments(
            track,
            share,
            _times(11, dt=0.005),
            max_jump=0.4 * track_length,
            method="map",
            min_peak_probability=5 / (2 * track.n_bins),
            merge_max_time_gap=0.05,
            merge_max_distance=0.4 * track_length,
            min_duration=0.03,
        )
        assert _segments(result) == [[0, 8]]
        assert result.longest == 0
        np.testing.assert_array_equal(
            result.trajectory_time_bins,
            [True, True, True, False, True, True, True, True, False, False, False],
        )
        assert result.durations[0] == pytest.approx(0.035)
        # The code's halved merge threshold (20 cm) would not merge 35 -> 55,
        # and both pieces would then be shorter than 30 ms.
        halved = replay_segments(
            track,
            share,
            _times(11, dt=0.005),
            max_jump=0.4 * track_length,
            min_peak_probability=5 / (2 * track.n_bins),
            merge_max_time_gap=0.05,
            merge_max_distance=0.2 * track_length,
            min_duration=0.03,
        )
        assert halved.n_segments == 0


class TestGeometry:
    def test_arena_uses_straight_line_jumps(self, arena):
        # (5, 5) -> (15, 5) -> (25, 5) -> (25, 15) -> (35, 25), then a jump
        # to (5, 45) and on to (15, 45). The diagonal step is 14.1 cm.
        points = np.array(
            [[5, 5], [15, 5], [25, 5], [25, 15], [35, 25], [5, 45], [15, 45]],
            dtype=float,
        )
        posterior = np.eye(arena.n_bins)[arena.bin_at(points)]
        result = replay_segments(arena, posterior, _times(7), max_jump=20.0)
        assert result.metric == "euclidean"
        assert _segments(result) == [[0, 5], [5, 7]]
        np.testing.assert_allclose(
            result.start_to_end_distances, [np.hypot(30.0, 20.0), 10.0]
        )
        np.testing.assert_allclose(result.path_lengths, [30.0 + np.sqrt(200.0), 10.0])

    def test_arena_posterior_means(self, arena):
        points = np.array([[5, 5], [15, 5], [25, 5], [45, 45]], dtype=float)
        posterior = np.eye(arena.n_bins)[arena.bin_at(points)]
        result = replay_segments(
            arena, posterior, _times(4), max_jump=20.0, method="mean"
        )
        assert _segments(result) == [[0, 3], [3, 4]]

    def test_track_jumps_follow_the_track(self, l_track):
        posterior = _sweep(l_track, [1, 2, 3, 6, 7])
        along = replay_segments(l_track, posterior, _times(5), max_jump=25.0)
        straight = replay_segments(
            l_track, posterior, _times(5), max_jump=25.0, metric="euclidean"
        )
        assert along.metric == "geodesic"
        assert _segments(along) == [[0, 3], [3, 5]]
        assert _segments(straight) == [[0, 5]]

    def test_renumbered_bins_give_the_same_segments(self):
        env = strips_1d(np.r_[0.0, np.cumsum(np.full(20, 5.0))])
        order = np.random.default_rng(3).permutation(env.n_bins)
        shuffled = renumbered(env, order)
        posterior = _sweep(env, [2, 3, 4, 5, 15, 16, 17, 9, 10])
        options = {"max_jump": 12.0, "max_gap_bins": 1}
        original = replay_segments(env, posterior, _times(9), **options)
        moved = replay_segments(shuffled, posterior[:, order], _times(9), **options)
        assert _segments(moved) == _segments(original)
        np.testing.assert_allclose(moved.path_lengths, original.path_lengths)
        assert moved.summary() == pytest.approx(original.summary())


class TestResult:
    @pytest.fixture(scope="class")
    def result(self, track):
        return replay_segments(
            track, _sweep(track, [1, 2, 3, 4, 5, 9, 8, 7]), _times(8), max_jump=20.0
        )

    def test_per_segment_values(self, result):
        np.testing.assert_allclose(result.start_times, [0.0, 5 * DT])
        np.testing.assert_allclose(result.end_times, [4 * DT, 7 * DT])
        np.testing.assert_allclose(result.durations, [4 * DT, 2 * DT])
        assert result.time_bin_counts.tolist() == [5, 3]
        np.testing.assert_allclose(result.start_to_end_distances, [40.0, 20.0])
        assert result.n_segments == 2
        assert result.n_time_bins == 8
        assert result.units == "cm"
        assert (result.method, result.jump_rule) == ("map", "between")

    def test_arrays_are_read_only(self, result):
        for name in ("segments", "durations", "trajectory_time_bins"):
            with pytest.raises(ValueError, match="read-only"):
                getattr(result, name)[0] = 0

    def test_pickle_keeps_arrays_read_only(self, result):
        copy = pickle.loads(pickle.dumps(result))
        assert not copy.segments.flags.writeable
        assert copy.summary() == pytest.approx(result.summary())

    def test_summary_holds_only_scalars(self, result):
        summary = result.summary()
        assert all(not isinstance(value, np.ndarray) for value in summary.values())
        assert summary["n_segments"] == 2
        assert summary["longest_time_bin_count"] == 5
        assert summary["longest_start_to_end_distance"] == pytest.approx(40.0)
        assert isinstance(result, ReplaySegmentsResult)
        assert "ReplaySegmentsResult" in repr(result)


class TestRefusals:
    @pytest.mark.parametrize(
        ("options", "match"),
        [
            ({"method": "median"}, "method must be one of"),
            ({"jump_rule": "drop_later"}, "jump_rule must be one of"),
            ({"max_jump": 0.0}, "max_jump must be a finite number > 0"),
            ({"max_jump": np.inf}, "max_jump must be a finite number > 0"),
            ({"max_jump": True}, "max_jump must be a number"),
            ({"max_gap_bins": -1}, "max_gap_bins must be an integer >= 0"),
            ({"max_gap_bins": 1.0}, "max_gap_bins must be an integer >= 0"),
            ({"max_spread": 0.0}, "max_spread must be a finite number > 0"),
            ({"min_peak_probability": -0.1}, "must be a finite number >= 0"),
            ({"min_peak_probability": 1.0}, "min_peak_probability must be below 1"),
            ({"merge_max_time_gap": 0.05}, "must be given together"),
            ({"merge_max_distance": 20.0}, "must be given together"),
            ({"min_duration": np.nan}, "min_duration must be a finite number"),
            ({"min_time_bins": 0}, "min_time_bins must be an integer >= 1"),
            ({"min_dispersion": -1.0}, "min_dispersion must be a finite number"),
            ({"metric": "manhattan"}, "metric must be one of"),
        ],
    )
    def test_bad_options(self, track, options, match):
        options = {"max_jump": 20.0, **options}
        with pytest.raises(ValueError, match=match):
            replay_segments(track, _sweep(track, [1, 2]), _times(2), **options)

    def test_max_jump_is_required(self, track):
        with pytest.raises(TypeError, match="max_jump"):
            replay_segments(track, _sweep(track, [1, 2]), _times(2))

    def test_errors_name_this_function(self, track):
        with pytest.raises(ValueError, match=r"replay_segments\(\).*one column"):
            replay_segments(track, np.eye(4), _times(4), max_jump=20.0)

    def test_rows_that_are_not_probabilities(self, track):
        with pytest.raises(ValueError, match="sum to more than 1"):
            replay_segments(track, 2 * _sweep(track, [1, 2]), _times(2), max_jump=20.0)

    def test_times_out_of_order(self, track):
        with pytest.raises(ValueError, match=r"replay_segments\(\): times must"):
            replay_segments(
                track, _sweep(track, [1, 2, 3]), [0.0, 0.04, 0.02], max_jump=20.0
            )

    def test_polar_environment(self):
        from neurospatial import CartesianOnlyError

        env = Environment.from_polar_egocentric(
            distance_range=(0.0, 30.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=15.0,
            angle_bin_size=np.pi / 4,
        )
        with pytest.raises(CartesianOnlyError, match="allocentric"):
            replay_segments(env, np.eye(env.n_bins)[:2], _times(2), max_jump=20.0)
