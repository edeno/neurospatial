"""Behavioral tests for the public ``object_vector_score`` function.

``object_vector_score`` takes an object-vector result, or its rate map shaped
``(n_distance_bins, n_direction_bins)``, and returns a scalar in ``[0, 1]``:
the normalized sparsity of the distance profile times the mean resultant
length of the direction profile, both over the rings sampled in every
direction. Most tests drive it on hand-built maps so the assertions exercise
the score math itself.
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
        # One sharp bin over a low baseline: both profiles peak sharply.
        tc = np.full((N_DIST, N_DIR), 0.01)
        tc[5, 6] = 20.0
        assert object_vector_score(tc) > 0.8

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

        # (b) A single nonzero bin: the distance profile has all its rate in
        #     one of K rings, sparsity 1/K, so (1 - 1/K) / (1 - 1/K) = 1; the
        #     direction profile has all mass at one angle, MRL = 1.
        tc_delta = np.zeros((N_DIST, N_DIR))
        tc_delta[5, 6] = 1.0
        assert object_vector_score(tc_delta) == pytest.approx(1.0, abs=1e-10)

    def test_silent_map_is_nan(self):
        # The sparsity is 0 / 0 for a silent cell: undefined, not untuned.
        tc = np.zeros((N_DIST, N_DIR))
        assert np.isnan(object_vector_score(tc))

    def test_all_nan_tuning_curve_returns_nan(self):
        # Documented behavior: no finite values -> returns NaN.
        tc = np.full((N_DIST, N_DIR), np.nan)
        assert np.isnan(object_vector_score(tc))

    @pytest.mark.parametrize("n_rings", [5, 10, 40])
    def test_one_bin_scores_one_on_any_grid(self, n_rings):
        # The distance factor's bound is derived from the ring count, not a
        # tuning constant, so all firing in one bin is a perfect score.
        tc = np.zeros((n_rings, N_DIR))
        tc[n_rings // 2, 6] = 3.0
        assert object_vector_score(tc) == pytest.approx(1.0, abs=1e-10)

    def test_fewer_than_two_complete_rings_is_nan(self):
        # The distance factor compares rings; one ring has nothing to compare.
        tc = np.full((N_DIST, N_DIR), np.nan)
        tc[0] = 5.0
        tc[1:, 6] = 20.0
        assert np.isnan(object_vector_score(tc))


def _von_mises(n: int, center: int, kappa: float = 4.0) -> np.ndarray:
    angles = 2 * np.pi * (np.arange(n) - center) / n
    return np.exp(kappa * (np.cos(angles) - 1))


class TestObjectVectorScoreProfiles:
    """Each factor reads its own profile of the map."""

    def test_direction_only_tuning_has_no_distance_selectivity(self):
        # Rate depends on direction alone: the distance profile is flat, so
        # this is a head-direction-like map, not an object-vector one.
        tc = np.tile(0.1 + 10 * _von_mises(N_DIR, 6), (N_DIST, 1))
        assert object_vector_score(tc) == pytest.approx(0.0, abs=1e-12)

    def test_distance_only_tuning_has_no_direction_selectivity(self):
        tc = np.tile((0.1 + 10 * _von_mises(N_DIST, 4))[:, None], (1, N_DIR))
        assert object_vector_score(tc) == pytest.approx(0.0, abs=1e-10)

    def test_unvisited_bins_are_not_counted_as_silent(self):
        # Ring tuning with no direction preference, but the preferred ring was
        # never visited behind the animal. Summing over rings would make the
        # rear directions look quiet, i.e. invent a forward preference.
        tc = np.tile((0.1 + 10 * _von_mises(N_DIST, 4))[:, None], (1, N_DIR))
        tc[3:6, :3] = np.nan
        assert object_vector_score(tc) == pytest.approx(0.0, abs=1e-10)

    def test_rings_sampled_on_one_side_add_no_distance_tuning(self):
        # Rate depends on direction alone, but the far rings were reached only
        # in the preferred directions (the object near a wall). Their ring
        # means are high, which a profile over every ring would read as
        # distance tuning; over the complete rings the profile stays flat.
        tc = np.tile(0.1 + 10 * _von_mises(N_DIR, 6), (N_DIST, 1))
        tc[6:, :5] = np.nan
        tc[6:, 8:] = np.nan
        assert object_vector_score(tc) == pytest.approx(0.0, abs=1e-12)

    def test_partial_direction_coverage_is_nan(self):
        # Only a quarter of the directions observed (an object in a corner):
        # an arc of flat rates would look directional.
        tc = np.full((N_DIST, N_DIR), np.nan)
        tc[:, :3] = 5.0
        tc[3, 1] = 6.0
        assert np.isnan(object_vector_score(tc))

    def test_flat_rate_map_input_raises_with_reshape_guidance(self):
        with pytest.raises(ValueError, match="reshape"):
            object_vector_score(np.ones(N_DIST * N_DIR))

    def test_baseline_subtracted_map_raises(self):
        # Subtracting the mean leaves negative rates, and the score weights
        # directions by rate: random noise used to score 1.0.
        rates = np.random.default_rng(0).normal(size=(N_DIST, N_DIR))
        n_negative = int((rates < 0).sum())
        with pytest.raises(ValueError) as excinfo:
            object_vector_score(rates)
        message = str(excinfo.value)
        assert f"{n_negative} negative" in message
        assert f"{rates.min():.3g}" in message
        assert "Why:" in message
        assert "Fix:" in message
        assert "z-scored" in message

    def test_infinite_rate_raises(self):
        rates = _gaussian_bump(width=1.0)
        rates[2, 3] = np.inf
        with pytest.raises(ValueError, match="1 infinite"):
            object_vector_score(rates)

    def test_nan_marks_unvisited_bins_not_an_error(self):
        rates = _gaussian_bump(width=1.0)
        rates[0, :4] = np.nan
        assert np.isfinite(object_vector_score(rates))


def test_result_input_matches_its_reshaped_rate_map():
    """A result is scored on its own grid, distance-major."""
    from neurospatial.encoding import compute_object_vector_rate

    rng = np.random.default_rng(0)
    times = np.arange(0, 300, 1 / 30)
    positions = rng.uniform(5, 95, (len(times), 2))
    obj = np.array([[50.0, 50.0]])
    delta = positions - obj
    distance = np.hypot(*delta.T)
    direction = np.arctan2(delta[:, 1], delta[:, 0]) + np.pi  # animal to object
    rate = 0.2 + 15 * np.exp(-0.5 * ((distance - 20) / 4) ** 2) * np.exp(
        3 * (np.cos(direction) - 1)
    )
    from neurospatial.simulation import generate_poisson_spikes

    spikes = generate_poisson_spikes(rate, times, seed=11)
    untuned = generate_poisson_spikes(np.full(len(times), rate.mean()), times, seed=12)
    settings = {"distance_range": (0.0, 50.0), "distance_bin_size": 5.0}
    result = compute_object_vector_rate(None, spikes, times, positions, obj, **settings)
    grid = np.asarray(result.firing_rate).reshape(
        result.n_distance_bins, result.n_direction_bins
    )
    assert object_vector_score(result) == pytest.approx(object_vector_score(grid))
    control = compute_object_vector_rate(
        None, untuned, times, positions, obj, **settings
    )
    assert object_vector_score(result) > 5 * object_vector_score(control)


def _open_field_cell(rate_fn, seed: int):
    """20 minutes of uniform positions in a 100 cm box, object in the centre."""
    from neurospatial import Environment
    from neurospatial.simulation import generate_poisson_spikes

    rng = np.random.default_rng(0)
    times = np.arange(0, 1200, 1 / 30)
    positions = rng.uniform(0, 100, (len(times), 2))
    env = Environment.from_samples(positions, bin_size=4.0, units="cm")
    obj = np.array([[50.0, 50.0]])
    delta = obj - positions  # animal to object
    distance = np.hypot(*delta.T)
    direction = np.arctan2(delta[:, 1], delta[:, 0])
    spikes = generate_poisson_spikes(rate_fn(distance, direction), times, seed=seed)
    return env, spikes, times, positions, obj


def test_object_vector_cell_outscores_a_direction_only_cell_on_the_arena_grid():
    """A bearing-only cell has no distance tuning, even where far rings are partial.

    The default grid covers the box diagonal, so rings past 50 cm are sampled
    only toward the corners.
    """
    from neurospatial.encoding import compute_object_vector_rate

    def ovc(distance, direction):
        tuned = np.exp(-0.5 * ((distance - 20) / 5) ** 2)
        return 0.2 + 15 * tuned * np.exp(4 * (np.cos(direction - 0.5) - 1))

    def bearing(distance, direction):
        return 0.2 + 8 * np.exp(2 * (np.cos(direction - np.pi / 4) - 1))

    scores = {}
    for name, rate_fn, seed in (("ovc", ovc, 1), ("bearing", bearing, 2)):
        env, spikes, times, positions, obj = _open_field_cell(rate_fn, seed)
        result = compute_object_vector_rate(env, spikes, times, positions, obj)
        scores[name] = object_vector_score(result)
    assert scores["ovc"] > 0.3
    assert scores["bearing"] < 0.05
