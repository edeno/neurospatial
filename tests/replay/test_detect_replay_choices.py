"""detect_replay: the alpha boundary and which segment a method tests."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.replay import (
    ReplayCriterion,
    ReplayMethod,
    ReplaySegmentation,
    detect_replay,
    weighted_correlation,
)

DT = 0.005


@pytest.fixture(scope="module")
def track():
    """50 bins of 4 cm."""
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [200.0, 0.0]], bin_size=4.0, units="cm"
    )


def test_a_p_value_equal_to_alpha_is_not_replay(track):
    """A clean sweep beats all 19 column-cycle shuffles: p = 1 / 20 = alpha."""
    method = ReplayMethod(
        name="cycle_19",
        reference="test",
        score=weighted_correlation,
        nulls=("column_cycle",),
        n_shuffles=19,
        alpha=0.05,
    )
    posterior = np.eye(track.n_bins)[np.arange(0, 50, 2)]
    times = np.arange(len(posterior)) * DT
    result = detect_replay(track, posterior, times, replay_method=method, rng=0)
    assert result.score == pytest.approx(1.0)
    assert result.p_value == 1 / 20
    assert not result.is_replay
    above = dataclasses.replace(method, name="alpha_above", alpha=0.051)
    assert detect_replay(track, posterior, times, replay_method=above, rng=0).is_replay


class TestChooseAny:
    """Carey et al. (2019); Widloski & Foster (2022): any passing segment."""

    @pytest.fixture(scope="class")
    def two_segments(self, track):
        """25 time bins up the track, a 100 cm jump, then 40 bins back down."""
        bins = np.r_[np.arange(0, 25), np.arange(49, 9, -1)]
        posterior = np.eye(track.n_bins)[bins]
        return posterior, np.arange(len(bins)) * DT

    @staticmethod
    def _method(minimum):
        return ReplayMethod(
            name=f"any_{minimum}",
            reference="test",
            score=None,
            segmentation=ReplaySegmentation({"max_jump": 10.0}, choose="any"),
            criteria=(ReplayCriterion("segment_time_bin_count", ">=", minimum),),
        )

    def test_the_longest_passing_segment_is_tested(self, track, two_segments):
        posterior, times = two_segments
        result = detect_replay(track, posterior, times, replay_method=self._method(20))
        np.testing.assert_array_equal(result.segments.segments, [[0, 25], [25, 65]])
        assert result.segment == 1
        assert result.values["segment_time_bin_count >= 20"] == 40
        assert result.is_replay

    def test_only_a_passing_segment_is_tested(self, track, two_segments):
        """With a rule only the shorter segment meets, it is the tested one."""
        posterior, times = two_segments
        method = dataclasses.replace(
            self._method(20),
            name="any_short",
            criteria=(ReplayCriterion("segment_time_bin_count", "<", 30),),
        )
        result = detect_replay(track, posterior, times, replay_method=method)
        assert result.segment == 0 and result.is_replay

    def test_without_a_passing_segment_the_longest_is_reported(
        self, track, two_segments
    ):
        posterior, times = two_segments
        result = detect_replay(track, posterior, times, replay_method=self._method(50))
        assert result.segment == 1
        assert not result.is_replay
