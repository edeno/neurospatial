"""Behavioral tests for the public ``object_vector_score`` function.

``object_vector_score(tuning_curve)`` takes a 2D egocentric polar tuning curve
of shape ``(n_dist, n_dir)`` and returns a scalar in ``[0, 1]`` combining
distance selectivity and direction selectivity. These tests drive the function
directly on hand-built tuning curves rather than constructing results, so the
assertions exercise the score math itself.
"""

import numpy as np
import pytest

from neurospatial.encoding.egocentric import object_vector_score

N_DIST = 10
N_DIR = 12


def _gaussian_bump(
    width: float,
    *,
    dist_center: int = 5,
    dir_center: int = 6,
    peak: float = 20.0,
    baseline: float = 0.1,
) -> np.ndarray:
    """Build an (N_DIST, N_DIR) tuning curve with a 2D Gaussian bump."""
    d = np.arange(N_DIST)[:, None]
    theta = np.arange(N_DIR)[None, :]
    bump = peak * np.exp(
        -((d - dist_center) ** 2 + (theta - dir_center) ** 2) / (2.0 * width**2)
    )
    return bump + baseline


class TestObjectVectorScore:
    """Score is high for concentrated tuning, low for diffuse tuning."""

    def test_concentrated_tuning_scores_high(self):
        # Sharp bin over a small uniform baseline. The baseline forms a pedestal
        # in the direction marginal that limits the MRL, so the score lands well
        # above the uniform case but below 1. This matches the documented
        # example in object_vector_score's docstring (asserts > 0.5).
        tc = np.full((N_DIST, N_DIR), 0.1)
        tc[5, 6] = 20.0
        score = object_vector_score(tc)
        assert score > 0.5

    def test_uniform_tuning_low_score(self):
        # Constant tuning: distance selectivity s_d = 1 -> normalized term = 0.
        tc = np.full((N_DIST, N_DIR), 5.0)
        score = object_vector_score(tc)
        assert score < 0.1

    def test_score_monotonic_in_concentration(self):
        # Widening the bump lowers both distance and direction selectivity.
        score_narrow = object_vector_score(_gaussian_bump(width=1.0))
        score_medium = object_vector_score(_gaussian_bump(width=3.0))
        score_wide = object_vector_score(_gaussian_bump(width=10.0))
        assert score_narrow > score_medium > score_wide

    def test_score_matches_hand_computed_factorization(self):
        # The score factorizes as normalized_dist_sel * direction_selectivity.
        # (a) All mass in ONE distance bin but UNIFORM across direction:
        #     distance selectivity is maximal, but direction MRL = 0, so the
        #     product must be exactly 0. Pins that direction_selectivity gates
        #     the score even when distance selectivity is saturated.
        tc_uniform_direction = np.zeros((N_DIST, N_DIR))
        tc_uniform_direction[3, :] = 7.0
        # MRL of a uniform marginal is 0 up to floating-point (bin-center
        # exponentials don't cancel to exactly 0), so the gated score is ~1e-16.
        assert object_vector_score(tc_uniform_direction) == pytest.approx(
            0.0, abs=1e-10
        )

        # (b) A single nonzero bin: distance selectivity s_d = N_DIST*N_DIR = 120,
        #     which clips normalized_dist_sel to 1.0; the direction marginal has
        #     all mass at one angle so MRL = 1.0. Product = 1.0 * 1.0 = 1.0.
        tc_delta = np.zeros((N_DIST, N_DIR))
        tc_delta[5, 6] = 1.0
        assert object_vector_score(tc_delta) == pytest.approx(1.0, abs=1e-10)

    def test_all_zero_tuning_curve_returns_zero(self):
        # Documented behavior (egocentric.py): mean_rate == 0 -> returns 0.0.
        tc = np.zeros((N_DIST, N_DIR))
        assert object_vector_score(tc) == 0.0

    def test_all_nan_tuning_curve_returns_nan(self):
        # Documented behavior: no finite values -> returns NaN.
        tc = np.full((N_DIST, N_DIR), np.nan)
        assert np.isnan(object_vector_score(tc))

    def test_invalid_max_distance_selectivity_raises(self):
        tc = np.full((N_DIST, N_DIR), 0.1)
        tc[5, 6] = 20.0
        with pytest.raises(ValueError, match="max_distance_selectivity"):
            object_vector_score(tc, max_distance_selectivity=1.0)
