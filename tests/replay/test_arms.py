"""Tests for single-arm and joint-arm replay on a maze (Wu & Foster 2014)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import DecodingResult
from neurospatial.replay import (
    ArmReplayMethod,
    compare_replay_methods,
    detect_arm_replay,
)
from neurospatial.replay.methods import BERNERS_LEE_2021, WU_FOSTER_2014

DT = 0.01


@pytest.fixture(scope="module")
def y_maze():
    """Three 40 cm arms in 4 cm bins: center 0-9, left 10-19, right 20-29."""
    graph = nx.Graph()
    for node, xy in {0: (0, 0), 1: (0, -40), 2: (-28, 28), 3: (28, 28)}.items():
        graph.add_node(node, pos=xy)
    graph.add_edges_from([(0, 1), (0, 2), (0, 3)])
    env = Environment.from_graph(
        graph,
        edge_order=[(1, 0), (0, 2), (0, 3)],
        edge_spacing=0.0,
        bin_size=4.0,
        units="cm",
    )
    assert env.n_bins == 30
    arms = {
        "center": np.arange(10)[::-1],
        "left": np.arange(10, 20),
        "right": np.arange(20, 30),
    }
    return env, arms


def _detect(maze, posterior, method=WU_FOSTER_2014):
    env, arms = maze
    return detect_arm_replay(
        env,
        posterior,
        np.arange(len(posterior)) * DT,
        arms=arms,
        arm_replay_method=method,
    )


class TestWuFoster:
    def test_a_path_through_the_junction_is_a_joint_replay(self, y_maze):
        env, arms = y_maze
        path = np.r_[arms["center"][::-1], arms["left"]]
        result = _detect(y_maze, np.eye(env.n_bins)[path])
        assert result.is_replay and result.template == "center+left"
        assert result.score == pytest.approx(1.0)
        assert result.criteria["center+left: same sign as joint"]

    def test_one_arm_alone_is_a_single_arm_replay(self, y_maze):
        env, arms = y_maze
        result = _detect(y_maze, np.eye(env.n_bins)[arms["left"]])
        assert result.is_replay and result.template == "left"
        assert not result.criteria["center: duration >= 0.05"]

    def test_arms_replayed_the_wrong_way_round_are_not_joint(self, y_maze):
        """Center outward then left outward: two single-arm replays."""
        env, arms = y_maze
        path = np.r_[arms["center"], arms["left"]]
        result = _detect(y_maze, np.eye(env.n_bins)[path])
        assert not result.criteria["center+left: same sign as joint"]
        assert result.is_replay and result.template == "center,left"

    def test_two_joint_replays_cancel(self, y_maze):
        """Into the junction from the center, then out along left and right."""
        env, arms = y_maze
        posterior = np.zeros((20, env.n_bins))
        posterior[np.arange(10), arms["center"][::-1]] = 1.0
        posterior[np.arange(10, 20), arms["left"]] = 0.5
        posterior[np.arange(10, 20), arms["right"]] = 0.5
        result = _detect(y_maze, posterior)
        assert result.criteria["center+left: |wc| > 0.5"]
        assert result.criteria["center+right: |wc| > 0.5"]
        assert not result.is_replay and result.template is None

    def test_coverage_must_exceed_half_the_arm(self, y_maze):
        env, arms = y_maze
        half = _detect(y_maze, np.eye(env.n_bins)[arms["left"][:5]].repeat(2, axis=0))
        assert half.values["left: coverage > 0.5"] == pytest.approx(0.5)
        assert not half.is_replay

    @pytest.mark.parametrize(("n_rows", "expected"), [(2, False), (3, True)])
    def test_a_subregion_needs_five_time_bins(self, y_maze, n_rows, expected):
        """Smoothing spreads a short sweep by one time bin at each end."""
        env, arms = y_maze
        bins = arms["left"][np.linspace(0, 9, n_rows).round().astype(int)]
        posterior = np.full((n_rows + 6, env.n_bins), 1.0 / env.n_bins)
        posterior[3 : 3 + n_rows] = np.eye(env.n_bins)[bins]
        result = _detect(y_maze, posterior)
        assert result.criteria["left: duration >= 0.05"] is expected

    def test_inputs_are_checked(self, y_maze):
        env, arms = y_maze
        posterior = np.eye(env.n_bins)[arms["left"]]
        times = np.arange(10) * DT
        with pytest.raises(ValueError, match="each bin belongs to one arm"):
            detect_arm_replay(
                env,
                posterior,
                times,
                arms={"a": [0, 1, 2], "b": [2, 3]},
                arm_replay_method=WU_FOSTER_2014,
            )
        with pytest.raises(ValueError, match="evenly spaced"):
            detect_arm_replay(
                env,
                posterior,
                times**2,
                arms=arms,
                arm_replay_method=WU_FOSTER_2014,
            )
        with pytest.raises(ValueError, match="one column per bin"):
            detect_arm_replay(
                env,
                posterior[:, :20],
                times,
                arms=arms,
                arm_replay_method=WU_FOSTER_2014,
            )

    def test_compare_classifies_each_event(self, y_maze):
        env, arms = y_maze
        events = [
            DecodingResult(
                posterior=np.eye(env.n_bins)[bins],
                env=env,
                times=np.arange(len(bins)) * DT,
            )
            for bins in (np.r_[arms["center"][::-1], arms["right"]], arms["left"])
        ]
        table = compare_replay_methods(
            env, events, replay_methods=[WU_FOSTER_2014], arms=arms
        )
        assert table["template"].tolist() == ["center+right", "left"]
        with pytest.raises(ValueError, match="need arms="):
            compare_replay_methods(env, events, replay_methods=[WU_FOSTER_2014])

    def test_recipe_matches_the_paper(self):
        method = WU_FOSTER_2014
        assert (method.peak_threshold, method.smoothing_sigma) == (5.0, 0.01)
        assert method.min_duration == 0.05
        assert (method.min_coverage, method.min_abs_correlation) == (0.5, 0.5)
        assert method.joint_min_abs_correlation == 0.5

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"min_coverage": 1.0}, "min_coverage"),
            ({"min_duration": -0.05}, "min_duration"),
            ({"smoothing_sigma": -0.01}, "smoothing_sigma"),
        ],
    )
    def test_invalid_methods_are_refused(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ArmReplayMethod(name="x", reference="y", **kwargs)


class TestArmReviewCases:
    def test_constituents_must_share_the_joint_sign(self, y_maze):
        """Left outward first, then center inward: each arm runs along the
        path, but the joint subregion is a sawtooth with a negative sign."""
        env, arms = y_maze
        path = np.r_[arms["left"], arms["center"][::-1]]
        result = _detect(y_maze, np.eye(env.n_bins)[path])
        assert result.criteria["center+left: |wc| > 0.5"]
        assert not result.criteria["center+left: same sign as joint"]
        assert result.template != "center+left"

    def test_a_joint_replay_suppresses_single_arm_replays(self, y_maze):
        """A center-to-left path, then a right-arm sweep toward the junction.

        The right sweep is a single-arm replay, but runs against the path
        through the junction from either other arm, so only center+left is
        a joint replay.
        """
        env, arms = y_maze
        rows = np.r_[arms["center"][::-1], arms["left"], [0] * 10, arms["right"][::-1]]
        posterior = np.eye(env.n_bins)[rows]
        posterior[20:30] = 1.0 / env.n_bins
        result = _detect(y_maze, posterior)
        assert result.criteria["right: |wc| > 0.5"]
        assert result.template == "center+left"

    def test_several_single_arm_replays_report_the_largest_correlation(self, y_maze):
        env, arms = y_maze
        path = np.r_[arms["center"], arms["left"]]
        posterior = np.eye(env.n_bins)[path]
        result = _detect(y_maze, posterior)
        assert result.template == "center,left"
        assert result.score == pytest.approx(
            max(result.values["center: |wc| > 0.5"], result.values["left: |wc| > 0.5"])
        )

    def test_the_largest_peak_is_kept_over_an_earlier_weaker_run(self, y_maze):
        env, arms = y_maze
        posterior = np.full((30, env.n_bins), 1.0 / env.n_bins)
        posterior[2:6, :] = 0.0
        posterior[2:6, arms["left"][:4]] = np.eye(4) * 0.3
        posterior[2:6] += (1.0 - posterior[2:6].sum(axis=1, keepdims=True)) / env.n_bins
        posterior[15:25] = np.eye(env.n_bins)[arms["left"]]
        result = _detect(y_maze, posterior)
        assert result.values["left: coverage > 0.5"] == pytest.approx(1.0)
        assert result.template == "left"

    def test_the_length_rule_is_in_seconds(self, y_maze):
        """Three 20 ms time bins last 60 ms; five time bins would be needed at 10 ms."""
        env, arms = y_maze
        posterior = np.full((9, env.n_bins), 1.0 / env.n_bins)
        posterior[3:6] = np.eye(env.n_bins)[arms["left"][[0, 5, 9]]]
        result = detect_arm_replay(
            env,
            posterior,
            np.arange(9) * 0.02,
            arms=arms,
            arm_replay_method=WU_FOSTER_2014,
        )
        assert result.values["left: duration >= 0.05"] == pytest.approx(0.06)
        assert result.criteria["left: duration >= 0.05"]

    def test_arms_must_run_outward_along_neighboring_bins(self, y_maze):
        env, arms = y_maze
        posterior = np.eye(env.n_bins)[arms["left"]]
        times = np.arange(10) * DT
        inward = {**arms, "center": arms["center"][::-1]}
        with pytest.raises(ValueError, match="runs toward the junction"):
            detect_arm_replay(
                env, posterior, times, arms=inward, arm_replay_method=WU_FOSTER_2014
            )
        gappy = {**arms, "left": np.r_[10, 12, 13]}
        with pytest.raises(ValueError, match="not neighbors"):
            detect_arm_replay(
                env, posterior, times, arms=gappy, arm_replay_method=WU_FOSTER_2014
            )

    @pytest.mark.parametrize("names", [(1, "1", "r"), ("a+b", "c", "d")])
    def test_arm_names_must_be_distinct_and_plain(self, y_maze, names):
        env, arms = y_maze
        renamed = dict(zip(names, arms.values(), strict=True))
        with pytest.raises(ValueError, match="arm name"):
            detect_arm_replay(
                env,
                np.eye(env.n_bins)[arms["left"]],
                np.arange(10) * DT,
                arms=renamed,
                arm_replay_method=WU_FOSTER_2014,
            )

    def test_a_posterior_with_nan_is_refused(self, y_maze):
        env, arms = y_maze
        posterior = np.eye(env.n_bins)[arms["left"]]
        posterior[3] = np.nan
        with pytest.raises(ValueError, match="NaN, Inf or negative"):
            _detect(y_maze, posterior)

    def test_compare_gives_a_short_event_an_error_row(self, y_maze):
        env, arms = y_maze
        events = [
            DecodingResult(
                posterior=np.eye(env.n_bins)[arms["left"][:1]],
                env=env,
                times=np.zeros(1),
            )
        ]
        table = compare_replay_methods(
            env, events, replay_methods=[WU_FOSTER_2014], arms=arms
        )
        assert not table["is_replay"].iloc[0]
        assert table["error"].iloc[0] == "fewer than two time bins to score"


STEP = 0.005


def _sweep(arm_bins, n_rows, reverse=False):
    """Rows visiting an arm's bins in order (or reversed) over n_rows."""
    order = np.linspace(0, len(arm_bins) - 1, n_rows).round().astype(int)
    bins = np.asarray(arm_bins)[order]
    return bins[::-1] if reverse else bins


def _bl(maze, posterior, method=BERNERS_LEE_2021):
    env, arms = maze
    return detect_arm_replay(
        env,
        posterior,
        np.arange(len(posterior)) * STEP,
        arms=arms,
        arm_replay_method=method,
    )


class TestBernersLee:
    def test_a_path_through_the_junction_is_a_joint_replay(self, y_maze):
        env, arms = y_maze
        rows = np.r_[_sweep(arms["center"], 15, reverse=True), _sweep(arms["left"], 15)]
        result = _bl(y_maze, np.eye(env.n_bins)[rows])
        assert result.is_replay and result.template == "center+left"

    def test_the_joint_sign_is_not_checked(self, y_maze):
        """Left outward, then center inward: a sawtooth along the path.

        Berners-Lee et al. report the pair; with Wu & Foster's same-sign
        rule switched on, it is not a joint replay.
        """
        import dataclasses

        env, arms = y_maze
        rows = np.r_[_sweep(arms["left"], 15), _sweep(arms["center"], 15, reverse=True)]
        posterior = np.eye(env.n_bins)[rows]
        bl = _bl(y_maze, posterior)
        assert bl.template == "left+center"
        signed = _bl(
            y_maze,
            posterior,
            dataclasses.replace(BERNERS_LEE_2021, name="signed", joint_same_sign=True),
        )
        assert not signed.criteria["left+center: same sign as joint"]
        assert signed.template in ("left", "center")

    def test_the_strongest_of_several_pairs_is_reported(self, y_maze):
        """Center inward, then left and right outward together."""
        env, arms = y_maze
        posterior = np.zeros((30, env.n_bins))
        posterior[np.arange(15), _sweep(arms["center"], 15, reverse=True)] = 1.0
        posterior[np.arange(15, 30), _sweep(arms["left"], 15)] = 0.7
        posterior[np.arange(15, 30), _sweep(arms["right"], 15)[::-1]] = 0.3
        result = _bl(y_maze, posterior)
        assert result.template == "center+left"
        assert result.score == pytest.approx(result.values["center+left: |wc| > 0.3"])

    def test_the_strongest_single_arm_replay_is_reported(self, y_maze):
        """Clean right and noisier left sweeps outward at the same time.

        Running outward from the junction together, they form no joint
        replay (|wc| about 0.08 along the path), so the stronger arm wins.
        """
        env, arms = y_maze
        posterior = np.zeros((20, env.n_bins))
        posterior[np.arange(20), _sweep(arms["right"], 20)] = 0.5
        posterior[np.arange(20), _sweep(arms["left"], 20)] = 0.35
        posterior[:, arms["left"][5]] += 0.15
        result = _bl(y_maze, posterior)
        assert result.criteria["left: |wc| > 0.3"]
        assert not result.criteria["left+right: |wc| > 0.3"]
        assert result.template == "right"

    def test_a_chain_of_three_runs_merges_whole(self, y_maze):
        """Two short dropouts split a left sweep in three; all merge."""
        env, arms = y_maze
        posterior = np.full((45, env.n_bins), 1.0 / env.n_bins)
        posterior[:12] = np.eye(env.n_bins)[_sweep(arms["left"][:4], 12)]
        posterior[16:28] = np.eye(env.n_bins)[_sweep(arms["left"][4:7], 12)]
        posterior[32:44] = np.eye(env.n_bins)[_sweep(arms["left"][7:], 12)]
        result = _bl(y_maze, posterior)
        assert result.values["left: coverage > 0.5"] == pytest.approx(0.9)
        assert result.values["left: duration >= 0.055"] > 0.2

    def test_a_fallback_arm_replay_is_flagged(self, y_maze):
        env, arms = y_maze
        result = _bl(y_maze, np.eye(env.n_bins)[_sweep(arms["left"][:4], 15)])
        assert result.is_replay and not result.criteria["sequence"]
        sweep = _bl(y_maze, np.eye(env.n_bins)[_sweep(arms["right"], 20)])
        assert sweep.criteria["sequence"]

    def test_only_pair_tests_each_arms_strongest_run(self, y_maze):
        """A hold on the left arm, then a clean sweep along it."""
        import dataclasses

        env, arms = y_maze
        posterior = np.full((40, env.n_bins), 1.0 / env.n_bins)
        posterior[:12] = np.eye(env.n_bins)[arms["left"][3]]
        posterior[25:] = np.eye(env.n_bins)[_sweep(arms["left"], 15)]
        method = dataclasses.replace(
            BERNERS_LEE_2021,
            name="runs_only_pair",
            joint="only_pair",
            arm_fallback=False,
        )
        result = _bl(y_maze, posterior, method)
        assert result.template == "left"

    def test_runs_merge_across_short_gaps(self, y_maze):
        """A sweep along the left arm with a 25 ms dropout in the middle."""
        import dataclasses

        env, arms = y_maze
        posterior = np.full((29, env.n_bins), 1.0 / env.n_bins)
        posterior[:12] = np.eye(env.n_bins)[_sweep(arms["left"][:6], 12)]
        posterior[17:] = np.eye(env.n_bins)[_sweep(arms["left"][6:], 12)]
        merged = _bl(y_maze, posterior)
        apart = _bl(
            y_maze,
            posterior,
            dataclasses.replace(BERNERS_LEE_2021, name="apart", merge_max_gap=None),
        )
        assert merged.values["left: coverage > 0.5"] == pytest.approx(0.9)
        assert merged.criteria["left: coverage > 0.5"]
        assert apart.values["left: coverage > 0.5"] <= 0.5

    def test_an_arm_above_threshold_is_still_an_arm_replay(self, y_maze):
        """The posterior sits on four bins of the left arm for 75 ms."""
        env, arms = y_maze
        rows = _sweep(arms["left"][:4], 15)
        result = _bl(y_maze, np.eye(env.n_bins)[rows])
        assert not result.criteria["left: coverage > 0.5"]
        assert result.is_replay and result.template == "left"

    def test_equally_strong_fallback_runs_give_no_replay(self, y_maze):
        """The same short sweep on the left and right arms at once."""
        env, arms = y_maze
        posterior = np.zeros((15, env.n_bins))
        posterior[np.arange(15), _sweep(arms["left"][:4], 15)] = 0.5
        posterior[np.arange(15), _sweep(arms["right"][:4], 15)] = 0.5
        result = _bl(y_maze, posterior)
        assert result.values["left: |wc| > 0.3"] == pytest.approx(
            result.values["right: |wc| > 0.3"]
        )
        assert result.values["left: |wc| > 0.3"] > 0.3
        assert not result.is_replay

    def test_merging_needs_runs(self):
        with pytest.raises(ValueError, match="merge_max_gap"):
            ArmReplayMethod(name="x", reference="y", merge_max_gap=0.05)

    def test_recipe_matches_the_code(self):
        method = BERNERS_LEE_2021
        assert (method.peak_threshold, method.smoothing_sigma) == (4.0, 0.005)
        assert (method.min_duration, method.merge_max_gap) == (0.055, 0.05)
        assert (method.min_abs_correlation, method.joint_min_abs_correlation) == (
            0.3,
            0.3,
        )
        assert (method.subregions, method.coverage, method.joint) == (
            "runs",
            "mean_range",
            "strongest_pair",
        )
        assert not method.joint_same_sign and method.arm_fallback
