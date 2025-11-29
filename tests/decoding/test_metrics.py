"""Tests for decoding quality metrics in neurospatial.decoding.metrics.

These tests verify that decoding error metrics correctly compute distances
between decoded and actual positions, handle various edge cases, and
support both Euclidean and graph-based distance metrics.
"""

import numpy as np
import pytest


class TestDecodingError:
    """Test decoding_error function."""

    def test_decoding_error_shape(self, small_2d_env):
        """decoding_error should return (n_time_bins,) array of distances."""
        from neurospatial.decoding.metrics import decoding_error

        n_time_bins = 10
        n_dims = small_2d_env.n_dims
        rng = np.random.default_rng(42)

        decoded = rng.uniform(0, 10, (n_time_bins, n_dims))
        actual = rng.uniform(0, 10, (n_time_bins, n_dims))

        result = decoding_error(decoded, actual)

        assert result.shape == (n_time_bins,)
        assert result.dtype == np.float64

    def test_decoding_error_identical_positions_is_zero(self, small_2d_env):
        """decoding_error should be zero when decoded equals actual."""
        from neurospatial.decoding.metrics import decoding_error

        n_time_bins = 5
        n_dims = small_2d_env.n_dims
        rng = np.random.default_rng(42)

        positions = rng.uniform(0, 10, (n_time_bins, n_dims))

        result = decoding_error(positions, positions)

        np.testing.assert_array_almost_equal(result, np.zeros(n_time_bins))

    def test_decoding_error_known_distances_1d(self):
        """decoding_error should compute correct 1D Euclidean distances."""
        from neurospatial.decoding.metrics import decoding_error

        decoded = np.array([[0.0], [1.0], [2.0], [3.0]])
        actual = np.array([[1.0], [1.0], [0.0], [6.0]])
        # Expected: |0-1|=1, |1-1|=0, |2-0|=2, |3-6|=3
        expected = np.array([1.0, 0.0, 2.0, 3.0])

        result = decoding_error(decoded, actual)

        np.testing.assert_array_almost_equal(result, expected)

    def test_decoding_error_known_distances_2d(self):
        """decoding_error should compute correct 2D Euclidean distances."""
        from neurospatial.decoding.metrics import decoding_error

        decoded = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
        actual = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        # Expected: 0, 3, 4 (3-4-5 triangle components)
        expected = np.array([0.0, 3.0, 4.0])

        result = decoding_error(decoded, actual)

        np.testing.assert_array_almost_equal(result, expected)

    def test_decoding_error_known_distances_3d(self):
        """decoding_error should compute correct 3D Euclidean distances."""
        from neurospatial.decoding.metrics import decoding_error

        # 3D: sqrt(3^2 + 4^2 + 0^2) = 5
        decoded = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
        actual = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        # Expected: 5.0, sqrt(3)
        expected = np.array([5.0, np.sqrt(3.0)])

        result = decoding_error(decoded, actual)

        np.testing.assert_array_almost_equal(result, expected)

    def test_decoding_error_non_negative(self, small_2d_env):
        """decoding_error should always return non-negative values."""
        from neurospatial.decoding.metrics import decoding_error

        n_time_bins = 20
        n_dims = small_2d_env.n_dims
        rng = np.random.default_rng(42)

        decoded = rng.uniform(-10, 10, (n_time_bins, n_dims))
        actual = rng.uniform(-10, 10, (n_time_bins, n_dims))

        result = decoding_error(decoded, actual)

        assert np.all(result >= 0)

    def test_decoding_error_symmetry(self, small_2d_env):
        """decoding_error(a, b) should equal decoding_error(b, a)."""
        from neurospatial.decoding.metrics import decoding_error

        n_time_bins = 10
        n_dims = small_2d_env.n_dims
        rng = np.random.default_rng(42)

        positions_a = rng.uniform(0, 10, (n_time_bins, n_dims))
        positions_b = rng.uniform(0, 10, (n_time_bins, n_dims))

        error_ab = decoding_error(positions_a, positions_b)
        error_ba = decoding_error(positions_b, positions_a)

        np.testing.assert_array_almost_equal(error_ab, error_ba)


class TestDecodingErrorNaNHandling:
    """Test NaN handling in decoding_error."""

    def test_decoding_error_nan_in_decoded_propagates(self):
        """NaN in decoded positions should propagate to output."""
        from neurospatial.decoding.metrics import decoding_error

        decoded = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
        actual = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

        result = decoding_error(decoded, actual)

        assert np.isfinite(result[0])
        assert np.isnan(result[1])  # NaN propagates
        assert np.isfinite(result[2])

    def test_decoding_error_nan_in_actual_propagates(self):
        """NaN in actual positions should propagate to output."""
        from neurospatial.decoding.metrics import decoding_error

        decoded = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        actual = np.array([[0.0, 0.0], [np.nan, 0.0], [0.0, 0.0]])

        result = decoding_error(decoded, actual)

        assert np.isfinite(result[0])
        assert np.isnan(result[1])  # NaN propagates
        assert np.isfinite(result[2])

    def test_decoding_error_all_nan(self):
        """All-NaN inputs should return all-NaN output."""
        from neurospatial.decoding.metrics import decoding_error

        decoded = np.full((3, 2), np.nan)
        actual = np.full((3, 2), np.nan)

        result = decoding_error(decoded, actual)

        assert np.all(np.isnan(result))


class TestDecodingErrorGraphMetric:
    """Test graph-based distance metric in decoding_error."""

    def test_decoding_error_graph_metric_nan_propagates(self, small_2d_env):
        """Graph metric should propagate NaN values from inputs."""
        from neurospatial.decoding.metrics import decoding_error

        bin_centers = small_2d_env.bin_centers
        decoded = np.array([bin_centers[0], [np.nan, np.nan], bin_centers[1]])
        actual = np.array([bin_centers[1], bin_centers[0], bin_centers[0]])

        result = decoding_error(decoded, actual, metric="graph", env=small_2d_env)

        assert np.isfinite(result[0])
        assert np.isnan(result[1])  # NaN propagates even with graph metric
        assert np.isfinite(result[2])

    def test_decoding_error_graph_metric_requires_env(self):
        """Graph metric should raise ValueError if env is None."""
        from neurospatial.decoding.metrics import decoding_error

        decoded = np.array([[1.0, 2.0], [3.0, 4.0]])
        actual = np.array([[0.0, 0.0], [0.0, 0.0]])

        with pytest.raises(ValueError, match=r"env.*required.*graph"):
            decoding_error(decoded, actual, metric="graph")

    def test_decoding_error_graph_metric_uses_env(self, small_2d_env):
        """Graph metric should use environment's distance_between method."""
        from neurospatial.decoding.metrics import decoding_error

        # Use bin centers to ensure positions are valid
        n_time_bins = 5
        bin_centers = small_2d_env.bin_centers
        rng = np.random.default_rng(42)

        # Pick random bin centers for decoded and actual
        decoded_idx = rng.choice(len(bin_centers), n_time_bins)
        actual_idx = rng.choice(len(bin_centers), n_time_bins)

        decoded = bin_centers[decoded_idx]
        actual = bin_centers[actual_idx]

        result = decoding_error(decoded, actual, metric="graph", env=small_2d_env)

        assert result.shape == (n_time_bins,)
        assert np.all(result >= 0)

    def test_decoding_error_graph_vs_euclidean_on_straight_path(self, small_2d_env):
        """For straight paths, graph and Euclidean distances may differ."""
        from neurospatial.decoding.metrics import decoding_error

        # Use two specific bin centers
        bin_centers = small_2d_env.bin_centers
        if len(bin_centers) < 2:
            pytest.skip("Need at least 2 bins")

        decoded = bin_centers[[0]]
        actual = bin_centers[[len(bin_centers) - 1]]

        euclidean = decoding_error(decoded, actual, metric="euclidean")
        graph = decoding_error(decoded, actual, metric="graph", env=small_2d_env)

        # Graph distance should be >= Euclidean (triangle inequality)
        assert graph[0] >= euclidean[0] - 1e-10  # Small tolerance

    def test_decoding_error_invalid_metric_raises(self):
        """Invalid metric should raise ValueError."""
        from neurospatial.decoding.metrics import decoding_error

        decoded = np.array([[1.0, 2.0]])
        actual = np.array([[0.0, 0.0]])

        with pytest.raises(ValueError, match=r"metric.*euclidean.*graph"):
            decoding_error(decoded, actual, metric="manhattan")


class TestMedianDecodingError:
    """Test median_decoding_error function."""

    def test_median_decoding_error_returns_float(self):
        """median_decoding_error should return a single float."""
        from neurospatial.decoding.metrics import median_decoding_error

        decoded = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        actual = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

        result = median_decoding_error(decoded, actual)

        assert isinstance(result, (float, np.floating))

    def test_median_decoding_error_identical_is_zero(self):
        """median_decoding_error should be 0.0 for identical positions."""
        from neurospatial.decoding.metrics import median_decoding_error

        positions = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = median_decoding_error(positions, positions)

        assert result == 0.0

    def test_median_decoding_error_known_median(self):
        """median_decoding_error should compute correct median."""
        from neurospatial.decoding.metrics import median_decoding_error

        # Errors will be [1, 2, 3, 4, 5] -> median = 3
        decoded = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])
        actual = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])

        result = median_decoding_error(decoded, actual)

        assert result == pytest.approx(3.0)

    def test_median_decoding_error_even_count(self):
        """median_decoding_error should handle even number of time bins."""
        from neurospatial.decoding.metrics import median_decoding_error

        # Errors will be [1, 2, 3, 4] -> median = (2+3)/2 = 2.5
        decoded = np.array([[0.0], [0.0], [0.0], [0.0]])
        actual = np.array([[1.0], [2.0], [3.0], [4.0]])

        result = median_decoding_error(decoded, actual)

        assert result == pytest.approx(2.5)


class TestMedianDecodingErrorNaNHandling:
    """Test NaN handling in median_decoding_error."""

    def test_median_decoding_error_ignores_nan(self):
        """median_decoding_error should ignore NaN values (nanmedian)."""
        from neurospatial.decoding.metrics import median_decoding_error

        # Errors: [1, NaN, 3] -> median of [1, 3] = 2
        decoded = np.array([[0.0], [np.nan], [0.0]])
        actual = np.array([[1.0], [0.0], [3.0]])

        result = median_decoding_error(decoded, actual)

        assert result == pytest.approx(2.0)

    def test_median_decoding_error_all_nan_returns_nan(self):
        """median_decoding_error with all NaN should return NaN."""
        from neurospatial.decoding.metrics import median_decoding_error

        decoded = np.full((3, 2), np.nan)
        actual = np.full((3, 2), np.nan)

        result = median_decoding_error(decoded, actual)

        assert np.isnan(result)


class TestSuccessCriteria:
    """Test success criteria from TASKS.md."""

    def test_success_criteria_shapes_and_types(self, small_2d_env):
        """Verify success criteria from TASKS.md for Milestone 2.1."""
        from neurospatial.decoding.metrics import decoding_error, median_decoding_error

        n_time_bins = 10
        n_dims = small_2d_env.n_dims
        rng = np.random.default_rng(42)

        decoded = rng.uniform(0, 10, (n_time_bins, n_dims))
        actual = rng.uniform(0, 10, (n_time_bins, n_dims))

        # Success criteria from TASKS.md
        errors = decoding_error(decoded, actual)
        assert errors.shape == (n_time_bins,)

        median = median_decoding_error(decoded, actual)
        assert isinstance(median, (float, np.floating))


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_time_bin(self):
        """Functions should work with a single time bin."""
        from neurospatial.decoding.metrics import decoding_error, median_decoding_error

        decoded = np.array([[1.0, 2.0]])
        actual = np.array([[3.0, 4.0]])

        errors = decoding_error(decoded, actual)
        assert errors.shape == (1,)

        median = median_decoding_error(decoded, actual)
        assert isinstance(median, (float, np.floating))
        # For single time bin, median equals the only value
        np.testing.assert_almost_equal(median, errors[0])

    def test_large_errors(self):
        """Functions should handle large distances correctly."""
        from neurospatial.decoding.metrics import decoding_error, median_decoding_error

        decoded = np.array([[0.0, 0.0], [1e6, 1e6]])
        actual = np.array([[1e6, 1e6], [0.0, 0.0]])

        errors = decoding_error(decoded, actual)

        # Both should be sqrt(2) * 1e6
        expected = np.sqrt(2) * 1e6
        np.testing.assert_array_almost_equal(errors, [expected, expected])

        median = median_decoding_error(decoded, actual)
        assert median == pytest.approx(expected)

    def test_high_dimensional_positions(self):
        """Functions should work with high-dimensional positions."""
        from neurospatial.decoding.metrics import decoding_error, median_decoding_error

        n_dims = 5
        n_time_bins = 3
        rng = np.random.default_rng(42)

        decoded = rng.uniform(0, 1, (n_time_bins, n_dims))
        actual = rng.uniform(0, 1, (n_time_bins, n_dims))

        errors = decoding_error(decoded, actual)
        assert errors.shape == (n_time_bins,)
        assert np.all(errors >= 0)

        median = median_decoding_error(decoded, actual)
        assert np.isfinite(median)
        assert median >= 0

    def test_shape_mismatch_raises(self):
        """Mismatched shapes should raise ValueError."""
        from neurospatial.decoding.metrics import decoding_error

        decoded = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        actual = np.array([[1.0, 2.0, 3.0]])  # (1, 3) - wrong shape

        with pytest.raises((ValueError, IndexError)):
            decoding_error(decoded, actual)
