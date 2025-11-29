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


class TestConfusionMatrix:
    """Test confusion_matrix function."""

    def test_confusion_matrix_shape(self, small_2d_env):
        """confusion_matrix should return (n_bins, n_bins) array."""
        from neurospatial.decoding.metrics import confusion_matrix

        n_bins = small_2d_env.n_bins
        n_time_bins = 50
        rng = np.random.default_rng(42)

        # Create random posterior (normalized per row)
        posterior = rng.random((n_time_bins, n_bins))
        posterior /= posterior.sum(axis=1, keepdims=True)

        # Random actual bins
        actual_bins = rng.integers(0, n_bins, n_time_bins)

        result = confusion_matrix(small_2d_env, posterior, actual_bins)

        assert result.shape == (n_bins, n_bins)
        assert result.dtype == np.float64

    def test_confusion_matrix_map_method_sum_equals_n_time_bins(self, small_2d_env):
        """For method='map', confusion matrix should sum to n_time_bins."""
        from neurospatial.decoding.metrics import confusion_matrix

        n_bins = small_2d_env.n_bins
        n_time_bins = 100
        rng = np.random.default_rng(42)

        posterior = rng.random((n_time_bins, n_bins))
        posterior /= posterior.sum(axis=1, keepdims=True)
        actual_bins = rng.integers(0, n_bins, n_time_bins)

        result = confusion_matrix(small_2d_env, posterior, actual_bins, method="map")

        # For MAP method, total counts should equal n_time_bins
        assert result.sum() == pytest.approx(n_time_bins)

    def test_confusion_matrix_expected_method_row_sums(self, small_2d_env):
        """For method='expected', each row should sum to count of actual bin occurrences."""
        from neurospatial.decoding.metrics import confusion_matrix

        n_bins = small_2d_env.n_bins
        n_time_bins = 100
        rng = np.random.default_rng(42)

        posterior = rng.random((n_time_bins, n_bins))
        posterior /= posterior.sum(axis=1, keepdims=True)
        actual_bins = rng.integers(0, n_bins, n_time_bins)

        result = confusion_matrix(
            small_2d_env, posterior, actual_bins, method="expected"
        )

        # Each row should sum to the count of times that bin occurred
        for bin_idx in range(n_bins):
            expected_count = np.sum(actual_bins == bin_idx)
            assert result[bin_idx].sum() == pytest.approx(expected_count)

    def test_confusion_matrix_perfect_decoding_map(self, small_2d_env):
        """Perfect decoding should produce diagonal confusion matrix (MAP)."""
        from neurospatial.decoding.metrics import confusion_matrix

        n_bins = small_2d_env.n_bins
        n_time_bins = n_bins * 2  # Each bin occurs twice
        rng = np.random.default_rng(42)

        # Create actual bins - each bin appears twice
        actual_bins = np.repeat(np.arange(n_bins), 2)
        rng.shuffle(actual_bins)

        # Create posterior that perfectly decodes (delta at actual position)
        posterior = np.zeros((n_time_bins, n_bins))
        posterior[np.arange(n_time_bins), actual_bins] = 1.0

        result = confusion_matrix(small_2d_env, posterior, actual_bins, method="map")

        # Should be diagonal with 2 in each diagonal entry
        expected = np.diag(np.full(n_bins, 2.0))
        np.testing.assert_array_almost_equal(result, expected)

    def test_confusion_matrix_perfect_decoding_expected(self, small_2d_env):
        """Perfect decoding should produce diagonal matrix (expected)."""
        from neurospatial.decoding.metrics import confusion_matrix

        n_bins = small_2d_env.n_bins
        n_time_bins = n_bins * 2
        rng = np.random.default_rng(42)

        actual_bins = np.repeat(np.arange(n_bins), 2)
        rng.shuffle(actual_bins)

        posterior = np.zeros((n_time_bins, n_bins))
        posterior[np.arange(n_time_bins), actual_bins] = 1.0

        result = confusion_matrix(
            small_2d_env, posterior, actual_bins, method="expected"
        )

        expected = np.diag(np.full(n_bins, 2.0))
        np.testing.assert_array_almost_equal(result, expected)

    def test_confusion_matrix_uniform_posterior_expected(self, small_2d_env):
        """Uniform posterior should spread mass equally across columns."""
        from neurospatial.decoding.metrics import confusion_matrix

        n_bins = small_2d_env.n_bins
        n_time_bins = n_bins  # One occurrence per bin

        # Uniform posterior
        posterior = np.ones((n_time_bins, n_bins)) / n_bins
        actual_bins = np.arange(n_bins)  # Each bin occurs once

        result = confusion_matrix(
            small_2d_env, posterior, actual_bins, method="expected"
        )

        # Each row should sum to 1 (one occurrence per actual bin)
        # Each entry in a row should be 1/n_bins (uniform spread)
        expected = np.ones((n_bins, n_bins)) / n_bins
        np.testing.assert_array_almost_equal(result, expected)

    def test_confusion_matrix_known_values_map(self, small_2d_env):
        """Test with known posterior and actual bins (MAP)."""
        from neurospatial.decoding.metrics import confusion_matrix

        n_bins = small_2d_env.n_bins
        if n_bins < 3:
            pytest.skip("Need at least 3 bins for this test")

        # Create simple 3-time-bin scenario
        posterior = np.zeros((3, n_bins))
        # Time 0: max at bin 0 (actual: bin 0 -> correct)
        # Time 1: max at bin 1 (actual: bin 0 -> wrong)
        # Time 2: max at bin 2 (actual: bin 1 -> wrong)
        posterior[0, 0] = 0.8
        posterior[0, 1] = 0.2
        posterior[1, 1] = 0.9
        posterior[1, 0] = 0.1
        posterior[2, 2] = 0.7
        posterior[2, 1] = 0.3

        actual_bins = np.array([0, 0, 1])

        result = confusion_matrix(small_2d_env, posterior, actual_bins, method="map")

        # Row 0 (actual=0): decoded=0 once, decoded=1 once
        assert result[0, 0] == pytest.approx(1.0)
        assert result[0, 1] == pytest.approx(1.0)
        # Row 1 (actual=1): decoded=2 once
        assert result[1, 2] == pytest.approx(1.0)

    def test_confusion_matrix_known_values_expected(self, small_2d_env):
        """Test with known posterior and actual bins (expected)."""
        from neurospatial.decoding.metrics import confusion_matrix

        n_bins = small_2d_env.n_bins
        if n_bins < 3:
            pytest.skip("Need at least 3 bins for this test")

        # Create simple scenario
        posterior = np.zeros((2, n_bins))
        # Time 0: 0.8 at bin 0, 0.2 at bin 1 (actual: bin 0)
        # Time 1: 0.5 at bin 0, 0.5 at bin 1 (actual: bin 0)
        posterior[0, 0] = 0.8
        posterior[0, 1] = 0.2
        posterior[1, 0] = 0.5
        posterior[1, 1] = 0.5

        actual_bins = np.array([0, 0])

        result = confusion_matrix(
            small_2d_env, posterior, actual_bins, method="expected"
        )

        # Row 0 (actual=0): accumulate posteriors for both time bins
        # Col 0: 0.8 + 0.5 = 1.3
        # Col 1: 0.2 + 0.5 = 0.7
        assert result[0, 0] == pytest.approx(1.3)
        assert result[0, 1] == pytest.approx(0.7)
        # Row 1 should be all zeros (no time bins with actual=1)
        np.testing.assert_array_almost_equal(result[1], np.zeros(n_bins))

    def test_confusion_matrix_non_negative(self, small_2d_env):
        """Confusion matrix should have non-negative entries."""
        from neurospatial.decoding.metrics import confusion_matrix

        n_bins = small_2d_env.n_bins
        n_time_bins = 50
        rng = np.random.default_rng(42)

        posterior = rng.random((n_time_bins, n_bins))
        posterior /= posterior.sum(axis=1, keepdims=True)
        actual_bins = rng.integers(0, n_bins, n_time_bins)

        for method in ["map", "expected"]:
            result = confusion_matrix(
                small_2d_env, posterior, actual_bins, method=method
            )
            assert np.all(result >= 0), f"Negative values for method={method}"

    def test_confusion_matrix_invalid_method_raises(self, small_2d_env):
        """Invalid method should raise ValueError."""
        from neurospatial.decoding.metrics import confusion_matrix

        n_bins = small_2d_env.n_bins
        posterior = np.ones((5, n_bins)) / n_bins
        actual_bins = np.zeros(5, dtype=np.int64)

        with pytest.raises(ValueError, match=r"method.*map.*expected"):
            confusion_matrix(small_2d_env, posterior, actual_bins, method="invalid")

    def test_confusion_matrix_out_of_range_bins_raises(self, small_2d_env):
        """Actual bins outside valid range should raise ValueError."""
        from neurospatial.decoding.metrics import confusion_matrix

        n_bins = small_2d_env.n_bins
        posterior = np.ones((5, n_bins)) / n_bins
        # Bin index out of range
        actual_bins = np.array([0, 1, n_bins, 0, 0])  # n_bins is out of range

        with pytest.raises(ValueError, match=r"actual_bins.*range"):
            confusion_matrix(small_2d_env, posterior, actual_bins)

    def test_confusion_matrix_negative_bins_raises(self, small_2d_env):
        """Negative bin indices should raise ValueError."""
        from neurospatial.decoding.metrics import confusion_matrix

        n_bins = small_2d_env.n_bins
        posterior = np.ones((5, n_bins)) / n_bins
        actual_bins = np.array([0, 1, -1, 0, 0])

        with pytest.raises(ValueError, match=r"actual_bins.*range"):
            confusion_matrix(small_2d_env, posterior, actual_bins)

    def test_confusion_matrix_shape_mismatch_raises(self, small_2d_env):
        """Mismatched time bins between posterior and actual_bins should raise."""
        from neurospatial.decoding.metrics import confusion_matrix

        n_bins = small_2d_env.n_bins
        posterior = np.ones((10, n_bins)) / n_bins
        actual_bins = np.zeros(5, dtype=np.int64)  # Different length

        with pytest.raises(ValueError, match=r"mismatch|length"):
            confusion_matrix(small_2d_env, posterior, actual_bins)

    def test_confusion_matrix_posterior_bins_mismatch_raises(self, small_2d_env):
        """Posterior with wrong number of bins should raise."""
        from neurospatial.decoding.metrics import confusion_matrix

        n_bins = small_2d_env.n_bins
        wrong_n_bins = n_bins + 5
        posterior = np.ones((10, wrong_n_bins)) / wrong_n_bins
        actual_bins = np.zeros(10, dtype=np.int64)

        with pytest.raises(ValueError, match=r"bins|posterior"):
            confusion_matrix(small_2d_env, posterior, actual_bins)

    def test_confusion_matrix_empty_bins_have_zero_rows(self, small_2d_env):
        """Bins that never occur should have all-zero rows."""
        from neurospatial.decoding.metrics import confusion_matrix

        n_bins = small_2d_env.n_bins
        if n_bins < 2:
            pytest.skip("Need at least 2 bins")

        # Only use bin 0, never bin 1 or higher
        posterior = np.zeros((5, n_bins))
        posterior[:, 0] = 1.0  # Always decode to bin 0
        actual_bins = np.zeros(5, dtype=np.int64)  # Always at bin 0

        result = confusion_matrix(small_2d_env, posterior, actual_bins, method="map")

        # Row 0 should have all mass, rows 1+ should be zero
        assert result[0, 0] == pytest.approx(5.0)
        for i in range(1, n_bins):
            np.testing.assert_array_almost_equal(result[i], np.zeros(n_bins))


class TestConfusionMatrixSuccessCriteria:
    """Test success criteria from TASKS.md for confusion_matrix."""

    def test_success_criteria_shapes_and_sums(self, small_2d_env):
        """Verify success criteria from TASKS.md for Milestone 2.2."""
        from neurospatial.decoding.metrics import confusion_matrix

        n_bins = small_2d_env.n_bins
        n_time_bins = 100
        rng = np.random.default_rng(42)

        posterior = rng.random((n_time_bins, n_bins))
        posterior /= posterior.sum(axis=1, keepdims=True)
        actual_bins = rng.integers(0, n_bins, n_time_bins)

        # Success criteria from TASKS.md
        cm = confusion_matrix(small_2d_env, posterior, actual_bins)
        assert cm.shape == (n_bins, n_bins)
        assert cm.sum() == pytest.approx(n_time_bins)  # for method="map"
