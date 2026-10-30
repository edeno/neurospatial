"""Column-cycle shuffles without the zero shift.

``null_options={"column_cycle": {"exclude_zero_shift": True}}`` and
``shuffle_posterior_circular(..., exclude_zero_shift=True)`` draw shifts of
1 to n_bins - 1 bins, as nelpy does (Maboudi et al. 2018).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial import Environment
from neurospatial.replay import (
    detect_trajectory_radon,
    replay_significance,
    shuffle_posterior_circular,
)

DT = 0.02


@pytest.fixture(scope="module")
def short_track():
    """A 20 cm track in 2 cm bins."""
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [20.0, 0.0]], bin_size=2.0, units="cm"
    )


class TestColumnCycleWithoutZeroShift:
    def test_no_row_stays_in_place(self, short_track):
        posterior = np.eye(short_track.n_bins)[:6]
        shuffles = np.array(
            list(
                shuffle_posterior_circular(
                    short_track,
                    posterior,
                    n_shuffles=300,
                    rng=0,
                    exclude_zero_shift=True,
                )
            )
        )
        in_place = (shuffles == posterior).all(axis=2)
        assert not in_place.any()
        default = np.array(
            list(
                shuffle_posterior_circular(
                    short_track, posterior, n_shuffles=300, rng=0
                )
            )
        )
        assert (default == posterior).all(axis=2).any()

    def test_default_draws_every_shift_from_the_generator(self, short_track):
        posterior = np.eye(short_track.n_bins)[:6]
        shuffled = next(iter(shuffle_posterior_circular(short_track, posterior, rng=3)))
        shifts = np.random.default_rng(3).integers(0, short_track.n_bins, size=6)
        assert_array_equal(np.argmax(shuffled, axis=1), (np.arange(6) + shifts) % 10)

    def test_replay_significance_passes_the_option(self, short_track):
        posterior = np.eye(short_track.n_bins)[:6]
        times = np.arange(6) * DT
        common = {
            "score_options": {"band": 2.0, "max_speed": 600.0},
            "nulls": "column_cycle",
            "n_shuffles": 30,
            "rng": 2,
        }
        excluded = replay_significance(
            short_track,
            posterior,
            times,
            null_options={"column_cycle": {"exclude_zero_shift": True}},
            **common,
        )
        shuffles = shuffle_posterior_circular(
            short_track, posterior, n_shuffles=30, rng=2, exclude_zero_shift=True
        )
        expected = [
            detect_trajectory_radon(
                short_track, s, times, band=2.0, max_speed=600.0
            ).score
            for s in shuffles
        ]
        assert_array_equal(excluded.shuffles["column_cycle"].null_scores, expected)

    def test_non_boolean_option_is_refused(self, short_track):
        with pytest.raises(ValueError, match="exclude_zero_shift of column_cycle"):
            replay_significance(
                short_track,
                np.eye(short_track.n_bins)[:4],
                np.arange(4) * DT,
                nulls="column_cycle",
                null_options={"column_cycle": {"exclude_zero_shift": 1}},
                n_shuffles=5,
            )
