"""Tests for probability mapping between environments."""

import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

from neurospatial.environment import CartesianSystem
from neurospatial.ops.alignment import IDW_MIN_DISTANCE, map_probabilities
from neurospatial.ops.transforms import estimate_transform, rotate_2d, scale_2d


# Minimal mock Environment class for testing
class MockEnvironment:
    """Mock Environment class for testing alignment functions."""

    coordinate_system = CartesianSystem()

    def __init__(self, bin_centers, n_dims):
        self.bin_centers = bin_centers
        self.n_dims = n_dims
        self.n_bins = len(bin_centers)

    def bin_center_tree(self):
        return cKDTree(self.bin_centers)

    def bin_at(self, positions):
        """Point bins: a position is in a bin only on its center."""
        distance, nearest = self.bin_center_tree().query(np.asarray(positions))
        return np.where(distance == 0, nearest, -1)

    @property
    def bin_spacing(self):
        """Median nearest-neighbor distance, as Environment.bin_spacing."""
        if self.n_bins < 2:
            return np.inf
        distances, _ = cKDTree(self.bin_centers).query(self.bin_centers, k=2)
        return float(np.median(distances[:, 1]))


class TestMapProbabilities:
    """Tests for map_probabilities function."""

    def test_basic_mapping(self):
        """Test basic probability mapping with identical environments."""
        src_bins = np.array([[0, 0], [1, 0], [0, 1]])
        tgt_bins = np.array([[0, 0], [1, 0], [0, 1]])
        src_probs = np.array([0.2, 0.5, 0.3])
        src_env = MockEnvironment(src_bins, 2)
        tgt_env = MockEnvironment(tgt_bins, 2)
        tgt_probs = map_probabilities(src_env, tgt_env, src_probs)
        np.testing.assert_allclose(tgt_probs, src_probs)

    def test_with_rotation(self):
        """Test mapping with rotation transform."""
        src_bins = np.array([[1, 0], [0, 1]])
        tgt_bins = np.array([[0, 1], [-1, 0]])
        src_probs = np.array([0.7, 0.3])
        src_env = MockEnvironment(src_bins, 2)
        tgt_env = MockEnvironment(tgt_bins, 2)
        tgt_probs = map_probabilities(
            src_env, tgt_env, src_probs, source_transform=rotate_2d(np.pi / 2)
        )
        np.testing.assert_allclose(tgt_probs, [0.7, 0.3])

    def test_fitted_transform_plugs_in(self):
        """estimate_transform(source landmarks, target landmarks) is the argument."""
        src_bins = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        known = rotate_2d(np.pi / 3, center=(0.5, 0.5)) @ scale_2d(2.0)
        tgt_env = MockEnvironment(known(src_bins), 2)
        fitted = estimate_transform(src_bins, known(src_bins), kind="similarity")
        mapped = map_probabilities(
            MockEnvironment(src_bins, 2),
            tgt_env,
            np.array([0.2, 0.3, 0.5]),
            source_transform=fitted,
        )
        np.testing.assert_allclose(mapped, [0.2, 0.3, 0.5])

    def test_source_transform_must_be_an_affine_of_the_same_dimension(self):
        src_env = MockEnvironment(np.array([[0.0, 0.0], [1.0, 0.0]]), 2)
        probs = np.array([0.5, 0.5])
        with pytest.raises(TypeError, match="AffineND or None"):
            map_probabilities(src_env, src_env, probs, source_transform=np.eye(3))
        with pytest.raises(ValueError, match="3-D but the environments are 2-D"):
            map_probabilities(
                src_env,
                src_env,
                probs,
                source_transform=estimate_transform(
                    np.eye(4)[:, :3], np.eye(4)[:, :3], kind="rigid"
                ),
            )

    def test_duplicate_mapping(self):
        """Test multiple source bins mapping to same target bin."""
        src_bins = np.array([[0, 0], [0.1, 0]])
        tgt_bins = np.array([[0, 0]])
        src_probs = np.array([0.4, 0.6])
        src_env = MockEnvironment(src_bins, 2)
        tgt_env = MockEnvironment(tgt_bins, 2)
        tgt_probs = map_probabilities(src_env, tgt_env, src_probs)
        np.testing.assert_allclose(tgt_probs, [1.0])

    def test_empty_source(self):
        """Test with empty source environment."""
        src_bins = np.empty((0, 2))
        tgt_bins = np.array([[0, 0], [1, 1]])
        src_probs = np.array([])
        src_env = MockEnvironment(src_bins, 2)
        tgt_env = MockEnvironment(tgt_bins, 2)
        tgt_probs = map_probabilities(src_env, tgt_env, src_probs)
        np.testing.assert_allclose(tgt_probs, [0, 0])

    def test_empty_target(self):
        """Test with empty target environment."""
        src_bins = np.array([[0, 0]])
        tgt_bins = np.empty((0, 2))
        src_probs = np.array([1.0])
        src_env = MockEnvironment(src_bins, 2)
        tgt_env = MockEnvironment(tgt_bins, 2)
        tgt_probs = map_probabilities(src_env, tgt_env, src_probs)
        assert tgt_probs.size == 0

    def test_shape_mismatch(self):
        """Test error when probability shape doesn't match."""
        src_bins = np.array([[0, 0], [1, 1]])
        tgt_bins = np.array([[0, 0], [1, 1]])
        src_probs = np.array([1.0])  # Wrong size
        src_env = MockEnvironment(src_bins, 2)
        tgt_env = MockEnvironment(tgt_bins, 2)
        with pytest.raises(ValueError):
            map_probabilities(src_env, tgt_env, src_probs)

    def test_dimension_mismatch(self):
        """Test error when environments have different dimensions."""
        src_bins = np.array([[0, 0]])
        tgt_bins = np.array([[0, 0, 0]])
        src_probs = np.array([1.0])
        src_env = MockEnvironment(src_bins, 2)
        tgt_env = MockEnvironment(tgt_bins, 3)
        with pytest.raises(ValueError):
            map_probabilities(src_env, tgt_env, src_probs)

    def test_with_scale(self):
        """Test mapping with source scaling."""
        src_bins = np.array([[1, 0], [0, 1]])
        tgt_bins = np.array([[2, 0], [0, 2]])
        src_probs = np.array([0.6, 0.4])
        src_env = MockEnvironment(src_bins, 2)
        tgt_env = MockEnvironment(tgt_bins, 2)
        tgt_probs = map_probabilities(
            src_env, tgt_env, src_probs, source_transform=scale_2d(2.0)
        )
        np.testing.assert_allclose(tgt_probs, [0.6, 0.4])

    def test_map_probabilities_nan_centers_raises(self):
        """A target env with NaN bin centers raises instead of returning zeros.

        Previously a cKDTree construction failure on malformed centers was
        swallowed into an all-zero probability map; a genuine error must now
        propagate.
        """
        src_bins = np.array([[0.0, 0.0], [1.0, 0.0]])
        tgt_bins = np.array([[0.0, 0.0], [np.nan, 0.0]])
        src_probs = np.array([0.6, 0.4])
        src_env = MockEnvironment(src_bins, 2)
        tgt_env = MockEnvironment(tgt_bins, 2)

        with pytest.raises(ValueError):
            map_probabilities(src_env, tgt_env, src_probs)

    def test_map_probabilities_empty_env_still_returns_zeros(self):
        """The legitimate empty-target path still returns zeros with a warning."""
        src_bins = np.array([[0.0, 0.0]])
        tgt_bins = np.empty((0, 2))
        src_probs = np.array([1.0])
        src_env = MockEnvironment(src_bins, 2)
        tgt_env = MockEnvironment(tgt_bins, 2)

        with pytest.warns(UserWarning, match="zero bins"):
            tgt_probs = map_probabilities(src_env, tgt_env, src_probs)

        assert tgt_probs.size == 0


class TestMapProbabilitiesValidation:
    """Invalid inputs are refused before any mass is moved."""

    @staticmethod
    def _envs():
        bins = np.array([[0.0, 0.0], [1.0, 0.0]])
        return MockEnvironment(bins, 2), MockEnvironment(bins, 2)

    def test_invalid_mode(self):
        src_env, tgt_env = self._envs()
        with pytest.raises(ValueError, match="Unrecognized mode"):
            map_probabilities(src_env, tgt_env, np.array([0.5, 0.5]), mode="invalid")

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_refuses_non_finite_probabilities(self, bad):
        """NaN passes a ``probs < 0`` check; it must still be refused."""
        src_env, tgt_env = self._envs()
        with pytest.raises(ValueError, match="NaN or infinite values in 1 bin"):
            map_probabilities(src_env, tgt_env, np.array([0.5, bad]))

    def test_refuses_negative_probabilities(self):
        src_env, tgt_env = self._envs()
        with pytest.raises(ValueError, match="nonnegative"):
            map_probabilities(src_env, tgt_env, np.array([1.5, -0.5]))

    @pytest.mark.parametrize("eps", [0.0, -1e-3, np.nan])
    def test_refuses_eps_that_is_not_positive(self, eps):
        """eps = 0 gives NaN weights at a target center; eps < 0 wrong ones."""
        src_env, tgt_env = self._envs()
        with pytest.raises(ValueError, match="eps must be a positive"):
            map_probabilities(
                src_env,
                tgt_env,
                np.array([0.5, 0.5]),
                mode="inverse-distance-weighted",
                n_neighbors=2,
                eps=eps,
            )

    def test_accepts_numpy_integer_neighbor_count(self):
        src_env, tgt_env = self._envs()
        probs = np.array([0.5, 0.5])
        options = {"mode": "inverse-distance-weighted"}
        np.testing.assert_array_equal(
            map_probabilities(
                src_env, tgt_env, probs, n_neighbors=np.int64(2), **options
            ),
            map_probabilities(src_env, tgt_env, probs, n_neighbors=2, **options),
        )

    @pytest.mark.parametrize("n_neighbors", [0, 1.5, True])
    def test_refuses_invalid_neighbor_count(self, n_neighbors):
        src_env, tgt_env = self._envs()
        with pytest.raises(ValueError, match="n_neighbors must be an integer"):
            map_probabilities(
                src_env,
                tgt_env,
                np.array([0.5, 0.5]),
                mode="inverse-distance-weighted",
                n_neighbors=n_neighbors,
            )


class TestMapProbabilitiesInverseDistanceWeighted:
    """Positive-behavior matrix for the IDW mapping mode."""

    def test_two_neighbor_split_is_hand_computed(self):
        # Source mass entirely at (0,0). Targets at distances 1 and 3.
        src = MockEnvironment(np.array([[0.0, 0.0]]), n_dims=2)
        tgt = MockEnvironment(np.array([[1.0, 0.0], [3.0, 0.0]]), n_dims=2)
        out = map_probabilities(
            src,
            tgt,
            np.array([1.0]),
            mode="inverse-distance-weighted",
            n_neighbors=2,
        )
        eps = IDW_MIN_DISTANCE
        w = np.array([1.0 / (1.0 + eps), 1.0 / (3.0 + eps)])
        expected = w / w.sum()  # ~[0.75, 0.25]
        np.testing.assert_allclose(out, expected, atol=1e-9)

    def test_mass_is_conserved(self):
        src = MockEnvironment(
            np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]), n_dims=2
        )
        tgt = MockEnvironment(
            np.array([[1.0, 1.0], [9.0, 1.0], [1.0, 9.0], [5.0, 5.0]]), n_dims=2
        )
        src_probs = np.array([0.5, 0.3, 0.2])
        out = map_probabilities(
            src,
            tgt,
            src_probs,
            mode="inverse-distance-weighted",
            n_neighbors=3,
        )
        # IDW weights per source row are normalized, so total mass is preserved.
        assert out.sum() == pytest.approx(src_probs.sum())
        assert out.shape == (tgt.n_bins,)

    def test_n_neighbors_clamped_to_target_count(self):
        # Requesting more neighbors than target bins must clamp to n_tgt,
        # giving the same result as requesting exactly n_tgt.
        src = MockEnvironment(np.array([[0.0, 0.0]]), n_dims=2)
        tgt = MockEnvironment(np.array([[1.0, 0.0], [3.0, 0.0]]), n_dims=2)
        sp = np.array([1.0])
        clamped = map_probabilities(
            src, tgt, sp, mode="inverse-distance-weighted", n_neighbors=5
        )
        exact = map_probabilities(
            src, tgt, sp, mode="inverse-distance-weighted", n_neighbors=2
        )
        np.testing.assert_allclose(clamped, exact, atol=1e-12)

    def test_k_eff_one_is_nearest_only(self):
        # k_eff == 1: all mass goes to the single nearest target (no split).
        src = MockEnvironment(np.array([[0.0, 0.0]]), n_dims=2)
        tgt = MockEnvironment(np.array([[1.0, 0.0], [3.0, 0.0]]), n_dims=2)
        out = map_probabilities(
            src,
            tgt,
            np.array([1.0]),
            mode="inverse-distance-weighted",
            n_neighbors=1,
        )
        np.testing.assert_allclose(out, [1.0, 0.0], atol=1e-12)


class TestMapProbabilitiesMaxDistance:
    """Mass is never moved farther than max_distance to a target bin."""

    @pytest.fixture(scope="class")
    def square(self):
        from neurospatial import Environment

        rng = np.random.default_rng(0)
        samples = rng.uniform(0, 100, (2000, 2))
        return (
            Environment.from_samples(samples, bin_size=5.0),
            Environment.from_samples(samples, bin_size=10.0),
        )

    def test_coarser_bins_of_the_same_arena_are_accepted(self, square):
        """Every fine bin lies inside a coarse bin, so nothing is refused."""
        fine, coarse = square
        probs = np.full(fine.n_bins, 1.0 / fine.n_bins)
        mapped = map_probabilities(fine, coarse, probs)
        assert mapped.sum() == pytest.approx(1.0)

    def test_elongated_coarser_bins_of_the_same_arena_are_accepted(self):
        """A 10x1 bin holds points 4.5 from its center though centers are 1 apart.

        Each source center lies inside a target bin, so the mapping is valid
        and every target bin collects its ten source bins.
        """
        from neurospatial import Environment

        y_edges = np.arange(0.0, 11.0)
        fine = Environment.from_grid_mask(
            np.ones((20, 10), dtype=bool), grid_edges=(np.arange(0.0, 21.0), y_edges)
        )
        elongated = Environment.from_grid_mask(
            np.ones((2, 10), dtype=bool),
            grid_edges=(np.array([0.0, 10.0, 20.0]), y_edges),
        )
        probs = np.full(fine.n_bins, 1.0 / fine.n_bins)

        mapped = map_probabilities(fine, elongated, probs)

        np.testing.assert_allclose(mapped, 10 / fine.n_bins)
        owner = elongated.bin_at(fine.bin_centers)
        np.testing.assert_allclose(
            mapped, np.bincount(owner, probs, minlength=elongated.n_bins)
        )

    def test_misaligned_mass_is_refused_with_the_amount(self, square):
        """Rotating about the origin moves the arena off the target bins.

        Nearest mapping used to pile that mass onto the target's edge bins.
        """
        fine, coarse = square
        probs = np.full(fine.n_bins, 1.0 / fine.n_bins)
        turn = rotate_2d(np.pi / 4) @ scale_2d(0.9)
        moved = turn(fine.bin_centers)
        distance, _ = coarse.bin_center_tree().query(moved)
        expected = f"{np.mean(distance > 10.0):.1%} of the source probability"
        with pytest.raises(ValueError, match=expected) as error:
            map_probabilities(fine, coarse, probs, source_transform=turn)
        assert "one target bin spacing (10)" in str(error.value)
        assert "math.inf" in str(error.value)

        anywhere = map_probabilities(
            fine,
            coarse,
            probs,
            source_transform=turn,
            max_distance=math.inf,
        )
        assert anywhere.sum() == pytest.approx(1.0)

    def test_only_bins_that_hold_mass_count(self):
        """A far source bin with zero probability moves nothing and passes."""
        src_env = MockEnvironment(np.array([[0.0, 0.0], [100.0, 0.0]]), 2)
        tgt_env = MockEnvironment(np.array([[0.0, 0.0], [1.0, 0.0]]), 2)
        np.testing.assert_allclose(
            map_probabilities(src_env, tgt_env, np.array([1.0, 0.0])), [1.0, 0.0]
        )
        with pytest.raises(ValueError, match=r"max_distance=50"):
            map_probabilities(src_env, tgt_env, np.array([0.5, 0.5]), max_distance=50.0)

    def test_a_negligible_share_beyond_the_limit_is_mapped(self):
        """A posterior's far tail is not refused; it goes to its nearest bin.

        Decoded posteriors are positive on every bin, so a tail of 1e-9 beyond
        the target would otherwise refuse every cross-session map.
        """
        src_env = MockEnvironment(np.array([[0.0, 0.0], [100.0, 0.0]]), 2)
        tgt_env = MockEnvironment(np.array([[0.0, 0.0], [1.0, 0.0]]), 2)
        tail = 1e-9
        mapped = map_probabilities(
            src_env, tgt_env, np.array([1.0 - tail, tail]), max_distance=50.0
        )
        np.testing.assert_allclose(mapped, [1.0 - tail, tail], rtol=0, atol=1e-15)

        with pytest.raises(ValueError, match=r"0\.001% of the source probability"):
            map_probabilities(
                src_env, tgt_env, np.array([1.0 - 1e-5, 1e-5]), max_distance=50.0
            )

    def test_default_limit_is_one_target_bin_spacing(self):
        """Centers within one target spacing of a target center are kept."""
        tgt_env = MockEnvironment(np.array([[0.0, 0.0], [1.0, 0.0]]), 2)
        near = MockEnvironment(np.array([[0.0, 0.0], [1.8, 0.0]]), 2)
        far = MockEnvironment(np.array([[0.0, 0.0], [2.5, 0.0]]), 2)
        probs = np.array([0.5, 0.5])
        np.testing.assert_allclose(map_probabilities(near, tgt_env, probs), [0.5, 0.5])
        with pytest.raises(ValueError, match=r"one target bin spacing \(1\)"):
            map_probabilities(far, tgt_env, probs)

    def test_inside_a_target_bin_is_never_outside(self):
        """A center inside a coarse bin is at distance 0, whatever max_distance."""
        from neurospatial import Environment

        samples = np.random.default_rng(0).uniform(0, 100, (2000, 2))
        fine = Environment.from_samples(samples, bin_size=5.0)
        coarse = Environment.from_samples(samples, bin_size=10.0)
        probs = np.full(fine.n_bins, 1.0 / fine.n_bins)
        assert np.all(coarse.bin_at(fine.bin_centers) >= 0)
        mapped = map_probabilities(fine, coarse, probs, max_distance=0.0)
        assert mapped.sum() == pytest.approx(1.0)

    def test_max_outside_fraction_sets_the_verdict(self):
        """The same 1e-5 outside share is refused, then accepted and conserved."""
        src_env = MockEnvironment(np.array([[0.0, 0.0], [100.0, 0.0]]), 2)
        tgt_env = MockEnvironment(np.array([[0.0, 0.0], [1.0, 0.0]]), 2)
        probs = np.array([1.0 - 1e-5, 1e-5])
        with pytest.raises(ValueError, match="max_outside_fraction=1e-06 allows"):
            map_probabilities(src_env, tgt_env, probs)
        mapped = map_probabilities(src_env, tgt_env, probs, max_outside_fraction=1e-4)
        np.testing.assert_allclose(mapped, [1.0 - 1e-5, 1e-5], rtol=0, atol=1e-15)
        assert mapped.sum() == pytest.approx(1.0)

    def test_zero_fraction_refuses_any_mass_outside(self):
        src_env = MockEnvironment(np.array([[0.0, 0.0], [100.0, 0.0]]), 2)
        tgt_env = MockEnvironment(np.array([[0.0, 0.0], [1.0, 0.0]]), 2)
        with pytest.raises(ValueError, match="lies outside the target"):
            map_probabilities(
                src_env, tgt_env, np.array([1.0 - 1e-12, 1e-12]), max_outside_fraction=0
            )

    @pytest.mark.parametrize(
        ("fraction", "error"),
        [
            (-0.1, ValueError),
            (1.5, ValueError),
            (np.nan, ValueError),
            ("a", TypeError),
            (True, TypeError),
        ],
    )
    def test_refuses_invalid_max_outside_fraction(self, fraction, error):
        src_env = MockEnvironment(np.array([[0.0, 0.0]]), 2)
        with pytest.raises(error, match="max_outside_fraction"):
            map_probabilities(
                src_env, src_env, np.array([1.0]), max_outside_fraction=fraction
            )

    @pytest.mark.parametrize(
        ("max_distance", "error"),
        [
            (-1.0, ValueError),
            (np.nan, ValueError),
            ("far", TypeError),
            (True, TypeError),
        ],
    )
    def test_refuses_invalid_max_distance(self, max_distance, error):
        src_env = MockEnvironment(np.array([[0.0, 0.0]]), 2)
        with pytest.raises(error, match="max_distance"):
            map_probabilities(
                src_env, src_env, np.array([1.0]), max_distance=max_distance
            )
