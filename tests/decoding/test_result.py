"""Tests for DecodingResult container.

These tests verify that DecodingResult correctly stores decoding results
and computes derived properties (MAP, mean, entropy) via cached properties.
"""

import numpy as np
import pytest


class TestDecodingResultBasic:
    """Test basic DecodingResult instantiation and properties."""

    def test_decoding_result_stores_posterior(self, small_2d_env):
        """DecodingResult should store the posterior array."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        n_bins = small_2d_env.n_bins
        posterior = np.ones((n_time_bins, n_bins)) / n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.posterior is posterior
        assert result.posterior.shape == (n_time_bins, n_bins)

    def test_decoding_result_stores_env(self, small_2d_env):
        """DecodingResult should store the environment reference."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.env is small_2d_env

    def test_decoding_result_stores_times(self, small_2d_env):
        """DecodingResult should store optional times array."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins
        times = np.arange(n_time_bins) * 0.025

        result = DecodingResult(posterior=posterior, env=small_2d_env, times=times)

        assert result.times is times
        np.testing.assert_array_equal(result.times, times)

    def test_decoding_result_times_defaults_to_none(self, small_2d_env):
        """DecodingResult.times should default to None."""
        from neurospatial.decoding import DecodingResult

        posterior = np.ones((5, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.times is None

    def test_n_time_bins_property(self, small_2d_env):
        """n_time_bins property should return correct count."""
        from neurospatial.decoding import DecodingResult

        for n_time_bins in [1, 5, 10, 100]:
            posterior = (
                np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins
            )
            result = DecodingResult(posterior=posterior, env=small_2d_env)
            assert result.n_time_bins == n_time_bins


class TestMapEstimate:
    """Test MAP (maximum a posteriori) estimate computation."""

    def test_map_estimate_shape(self, small_2d_env):
        """map_estimate should return (n_time_bins,) array of bin indices."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.map_estimate.shape == (n_time_bins,)
        assert result.map_estimate.dtype in (np.int64, np.intp)

    def test_map_estimate_delta_posterior(self, small_2d_env):
        """map_estimate should return correct bin for delta (one-hot) posterior."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        # Create delta posteriors: all mass on a single bin per time step
        posterior = np.zeros((n_time_bins, n_bins))
        expected_bins = [0, 3, n_bins - 1, 5, 2]
        for t, bin_idx in enumerate(expected_bins):
            posterior[t, bin_idx] = 1.0

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        np.testing.assert_array_equal(result.map_estimate, expected_bins)

    def test_map_estimate_identity_posterior(self, small_2d_env):
        """map_estimate on identity posterior should return diagonal indices."""
        from neurospatial.decoding import DecodingResult

        # Use identity matrix (10x10) if env has enough bins
        n_time_bins = min(10, small_2d_env.n_bins)
        posterior = np.eye(n_time_bins, small_2d_env.n_bins)

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        expected = np.arange(n_time_bins)
        np.testing.assert_array_equal(result.map_estimate, expected)


class TestMapPosition:
    """Test MAP position in environment coordinates."""

    def test_map_position_shape(self, small_2d_env):
        """map_position should return (n_time_bins, n_dims) array."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.map_position.shape == (n_time_bins, small_2d_env.n_dims)
        assert result.map_position.dtype == np.float64

    def test_map_position_matches_bin_centers(self, small_2d_env):
        """map_position should equal bin_centers[map_estimate]."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        # Create delta posteriors at known bins
        posterior = np.zeros((n_time_bins, n_bins))
        known_bins = [0, 1, 2, 3, 4]
        for t, bin_idx in enumerate(known_bins):
            posterior[t, bin_idx] = 1.0

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        expected_positions = small_2d_env.bin_centers[known_bins]
        np.testing.assert_array_almost_equal(result.map_position, expected_positions)


class TestMeanPosition:
    """Test posterior mean position computation."""

    def test_mean_position_shape(self, small_2d_env):
        """mean_position should return (n_time_bins, n_dims) array."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.mean_position.shape == (n_time_bins, small_2d_env.n_dims)
        assert result.mean_position.dtype == np.float64

    def test_mean_position_delta_equals_map(self, small_2d_env):
        """For delta posteriors, mean_position should equal map_position."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        # Create delta posteriors
        posterior = np.zeros((n_time_bins, n_bins))
        for t in range(n_time_bins):
            posterior[t, t % n_bins] = 1.0

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        np.testing.assert_array_almost_equal(result.mean_position, result.map_position)

    def test_mean_position_uniform_is_centroid(self, small_2d_env):
        """For uniform posterior, mean_position should be centroid of all bins."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 3
        n_bins = small_2d_env.n_bins
        posterior = np.ones((n_time_bins, n_bins)) / n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        # Expected: centroid of all bin centers
        expected_centroid = small_2d_env.bin_centers.mean(axis=0)
        for t in range(n_time_bins):
            np.testing.assert_array_almost_equal(
                result.mean_position[t], expected_centroid
            )

    def test_mean_position_two_bin_weighted(self, small_2d_env):
        """Mean position with two bins should be weighted average."""
        from neurospatial.decoding import DecodingResult

        n_bins = small_2d_env.n_bins
        if n_bins < 2:
            pytest.skip("Need at least 2 bins for this test")

        # Put 75% mass on bin 0, 25% on bin 1
        posterior = np.zeros((1, n_bins))
        posterior[0, 0] = 0.75
        posterior[0, 1] = 0.25

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        # Expected: 0.75 * bin_centers[0] + 0.25 * bin_centers[1]
        expected = (
            0.75 * small_2d_env.bin_centers[0] + 0.25 * small_2d_env.bin_centers[1]
        )
        np.testing.assert_array_almost_equal(result.mean_position[0], expected)


class TestUncertainty:
    """Test posterior entropy (uncertainty) computation."""

    def test_uncertainty_shape(self, small_2d_env):
        """uncertainty should return (n_time_bins,) array."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.uncertainty.shape == (n_time_bins,)
        assert result.uncertainty.dtype == np.float64

    def test_uncertainty_delta_is_zero(self, small_2d_env):
        """Delta (one-hot) posterior should have zero entropy."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        posterior = np.zeros((n_time_bins, n_bins))
        for t in range(n_time_bins):
            posterior[t, t % n_bins] = 1.0

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        np.testing.assert_array_almost_equal(result.uncertainty, np.zeros(n_time_bins))

    def test_uncertainty_uniform_is_maximum(self, small_2d_env):
        """Uniform posterior should have maximum entropy = log2(n_bins)."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 3
        n_bins = small_2d_env.n_bins
        posterior = np.ones((n_time_bins, n_bins)) / n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        max_entropy = np.log2(n_bins)
        np.testing.assert_array_almost_equal(
            result.uncertainty, np.full(n_time_bins, max_entropy)
        )

    def test_uncertainty_bounds(self, small_2d_env):
        """Uncertainty should be bounded: 0 <= entropy <= log2(n_bins)."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        n_bins = small_2d_env.n_bins

        # Random posterior (normalized)
        rng = np.random.default_rng(42)
        posterior = rng.random((n_time_bins, n_bins))
        posterior = posterior / posterior.sum(axis=1, keepdims=True)

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert np.all(result.uncertainty >= 0)
        assert np.all(result.uncertainty <= np.log2(n_bins) + 1e-10)

    def test_uncertainty_handles_exact_zeros(self, small_2d_env):
        """Uncertainty should handle exact zeros without NaN (mask-based)."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 3
        n_bins = small_2d_env.n_bins
        # Posterior with many exact zeros
        posterior = np.zeros((n_time_bins, n_bins))
        posterior[:, 0] = 0.5
        posterior[:, 1] = 0.5

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        # Should not be NaN
        assert np.all(np.isfinite(result.uncertainty))
        # Entropy of 50-50 distribution = 1 bit
        np.testing.assert_array_almost_equal(result.uncertainty, np.ones(n_time_bins))


class TestCachedPropertyBehavior:
    """Test that cached properties work correctly."""

    def test_map_estimate_is_cached(self, small_2d_env):
        """map_estimate should be computed only once (cached)."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        # Access twice
        first_call = result.map_estimate
        second_call = result.map_estimate

        # Should be the same object (cached)
        assert first_call is second_call

    def test_map_position_is_cached(self, small_2d_env):
        """map_position should be computed only once (cached)."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        first_call = result.map_position
        second_call = result.map_position

        assert first_call is second_call

    def test_mean_position_is_cached(self, small_2d_env):
        """mean_position should be computed only once (cached)."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        first_call = result.mean_position
        second_call = result.mean_position

        assert first_call is second_call

    def test_uncertainty_is_cached(self, small_2d_env):
        """uncertainty should be computed only once (cached)."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        first_call = result.uncertainty
        second_call = result.uncertainty

        assert first_call is second_call


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_time_bin(self, small_2d_env):
        """DecodingResult should work with a single time bin."""
        from neurospatial.decoding import DecodingResult

        posterior = np.ones((1, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.n_time_bins == 1
        assert result.map_estimate.shape == (1,)
        assert result.map_position.shape == (1, small_2d_env.n_dims)
        assert result.mean_position.shape == (1, small_2d_env.n_dims)
        assert result.uncertainty.shape == (1,)

    def test_single_bin_environment(self):
        """DecodingResult should work with single-bin environment."""
        from neurospatial import Environment
        from neurospatial.decoding import DecodingResult

        # Create a minimal single-bin environment
        env = Environment.from_samples(
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            bin_size=10.0,  # Large bin size to get single bin
        )

        if env.n_bins == 1:
            n_time_bins = 5
            posterior = np.ones((n_time_bins, 1))  # Only one bin, always prob=1

            result = DecodingResult(posterior=posterior, env=env)

            assert result.n_time_bins == n_time_bins
            np.testing.assert_array_equal(result.map_estimate, np.zeros(n_time_bins))
            # Entropy of single bin is 0 (log2(1) = 0)
            np.testing.assert_array_almost_equal(
                result.uncertainty, np.zeros(n_time_bins)
            )
        else:
            pytest.skip("Could not create single-bin environment")

    def test_identity_posterior_success_criteria(self, small_2d_env):
        """Verify success criteria from TASKS.md with identity posterior."""
        from neurospatial.decoding import DecodingResult

        # Use identity matrix as posterior per TASKS.md success criteria
        n_time_bins = min(10, small_2d_env.n_bins)
        posterior = np.eye(n_time_bins, small_2d_env.n_bins)
        times = np.arange(n_time_bins) * 0.025

        result = DecodingResult(posterior=posterior, env=small_2d_env, times=times)

        # Success criteria from TASKS.md
        assert result.map_estimate.shape == (n_time_bins,)
        assert result.uncertainty.shape == (n_time_bins,)


class TestPlotMethod:
    """Test DecodingResult.plot() method."""

    def test_plot_returns_axes(self, small_2d_env):
        """plot() should return a matplotlib Axes object."""
        import matplotlib.pyplot as plt

        from neurospatial.decoding import DecodingResult

        posterior = np.ones((5, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        ax = result.plot()

        assert ax is not None
        # Check it's a matplotlib Axes object
        assert hasattr(ax, "imshow")
        assert hasattr(ax, "set_xlabel")
        plt.close("all")

    def test_plot_creates_figure_when_no_ax_provided(self, small_2d_env):
        """plot() should create a new figure when ax is None."""
        import matplotlib.pyplot as plt

        from neurospatial.decoding import DecodingResult

        posterior = np.ones((5, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        ax = result.plot()

        # Should have created a figure
        assert ax.figure is not None
        plt.close("all")

    def test_plot_uses_provided_axes(self, small_2d_env):
        """plot() should use provided axes when given."""
        import matplotlib.pyplot as plt

        from neurospatial.decoding import DecodingResult

        posterior = np.ones((5, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        _, provided_ax = plt.subplots()
        returned_ax = result.plot(ax=provided_ax)

        assert returned_ax is provided_ax
        plt.close("all")

    def test_plot_kwargs_passed_to_imshow(self, small_2d_env):
        """plot() should pass kwargs to imshow."""
        import matplotlib.pyplot as plt

        from neurospatial.decoding import DecodingResult

        posterior = np.ones((5, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        # Test with a custom colormap
        ax = result.plot(cmap="viridis")

        # The plot should have been created without error
        assert ax is not None
        plt.close("all")

    def test_plot_with_times_uses_time_extent(self, small_2d_env):
        """plot() should use times for x-axis extent when provided."""
        import matplotlib.pyplot as plt

        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins
        times = np.linspace(0.0, 1.0, n_time_bins)
        result = DecodingResult(posterior=posterior, env=small_2d_env, times=times)

        ax = result.plot()

        # Axis should have been labeled appropriately
        assert ax is not None
        plt.close("all")

    def test_plot_with_colorbar(self, small_2d_env):
        """plot() should support adding a colorbar."""
        import matplotlib.pyplot as plt

        from neurospatial.decoding import DecodingResult

        posterior = np.ones((5, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        ax = result.plot(colorbar=True)

        # Should have created colorbar without error
        assert ax is not None
        plt.close("all")

    def test_plot_shows_map_overlay(self, small_2d_env):
        """plot() should optionally show MAP estimate overlay."""
        import matplotlib.pyplot as plt

        from neurospatial.decoding import DecodingResult

        # Create posterior with clear MAP trajectory
        n_time_bins = 10
        n_bins = small_2d_env.n_bins
        posterior = np.zeros((n_time_bins, n_bins))
        for t in range(n_time_bins):
            posterior[t, t % n_bins] = 1.0

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        ax = result.plot(show_map=True)

        assert ax is not None
        plt.close("all")


class TestToDataFrameMethod:
    """Test DecodingResult.to_dataframe() method."""

    def test_to_dataframe_returns_dataframe(self, small_2d_env):
        """to_dataframe() should return a pandas DataFrame."""
        import pandas as pd

        from neurospatial.decoding import DecodingResult

        posterior = np.ones((5, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        df = result.to_dataframe()

        assert isinstance(df, pd.DataFrame)

    def test_to_dataframe_has_correct_row_count(self, small_2d_env):
        """to_dataframe() should have one row per time bin."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        df = result.to_dataframe()

        assert len(df) == n_time_bins

    def test_to_dataframe_has_time_column_when_times_provided(self, small_2d_env):
        """to_dataframe() should include time column when times are provided."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins
        times = np.arange(n_time_bins) * 0.025
        result = DecodingResult(posterior=posterior, env=small_2d_env, times=times)

        df = result.to_dataframe()

        assert "time" in df.columns
        np.testing.assert_array_almost_equal(df["time"].values, times)

    def test_to_dataframe_no_time_column_when_times_none(self, small_2d_env):
        """to_dataframe() should not have time column when times is None."""
        from neurospatial.decoding import DecodingResult

        posterior = np.ones((5, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        df = result.to_dataframe()

        assert "time" not in df.columns

    def test_to_dataframe_has_map_bin_column(self, small_2d_env):
        """to_dataframe() should include map_bin column."""
        from neurospatial.decoding import DecodingResult

        # Create delta posteriors with known MAP bins
        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        posterior = np.zeros((n_time_bins, n_bins))
        expected_bins = [0, 1, 2, 3, 4]
        for t, b in enumerate(expected_bins):
            posterior[t, b] = 1.0

        result = DecodingResult(posterior=posterior, env=small_2d_env)
        df = result.to_dataframe()

        assert "map_bin" in df.columns
        np.testing.assert_array_equal(df["map_bin"].values, expected_bins)

    def test_to_dataframe_has_map_position_columns_2d(self, small_2d_env):
        """to_dataframe() should include map_x and map_y for 2D environments."""
        from neurospatial.decoding import DecodingResult

        posterior = np.ones((5, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        df = result.to_dataframe()

        assert "map_x" in df.columns
        assert "map_y" in df.columns

    def test_to_dataframe_has_mean_position_columns_2d(self, small_2d_env):
        """to_dataframe() should include mean_x and mean_y for 2D environments."""
        from neurospatial.decoding import DecodingResult

        posterior = np.ones((5, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        df = result.to_dataframe()

        assert "mean_x" in df.columns
        assert "mean_y" in df.columns

    def test_to_dataframe_has_uncertainty_column(self, small_2d_env):
        """to_dataframe() should include uncertainty column."""
        from neurospatial.decoding import DecodingResult

        posterior = np.ones((5, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        df = result.to_dataframe()

        assert "uncertainty" in df.columns
        # Uniform posterior has max entropy
        expected_entropy = np.log2(small_2d_env.n_bins)
        np.testing.assert_array_almost_equal(
            df["uncertainty"].values, np.full(5, expected_entropy)
        )

    def test_to_dataframe_map_positions_match_property(self, small_2d_env):
        """to_dataframe() map positions should match map_position property."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        posterior = np.zeros((n_time_bins, n_bins))
        for t in range(n_time_bins):
            posterior[t, t % n_bins] = 1.0

        result = DecodingResult(posterior=posterior, env=small_2d_env)
        df = result.to_dataframe()

        np.testing.assert_array_almost_equal(
            df["map_x"].values, result.map_position[:, 0]
        )
        np.testing.assert_array_almost_equal(
            df["map_y"].values, result.map_position[:, 1]
        )

    def test_to_dataframe_mean_positions_match_property(self, small_2d_env):
        """to_dataframe() mean positions should match mean_position property."""
        from neurospatial.decoding import DecodingResult

        posterior = np.ones((5, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)
        df = result.to_dataframe()

        np.testing.assert_array_almost_equal(
            df["mean_x"].values, result.mean_position[:, 0]
        )
        np.testing.assert_array_almost_equal(
            df["mean_y"].values, result.mean_position[:, 1]
        )

    def test_to_dataframe_1d_environment(self):
        """to_dataframe() should use 'x' for 1D environments."""
        from neurospatial import Environment
        from neurospatial.decoding import DecodingResult

        # Create 1D environment
        positions = np.linspace(0, 100, 100).reshape(-1, 1)
        env = Environment.from_samples(positions, bin_size=5.0)

        posterior = np.ones((5, env.n_bins)) / env.n_bins
        result = DecodingResult(posterior=posterior, env=env)
        df = result.to_dataframe()

        assert "map_x" in df.columns
        assert "mean_x" in df.columns
        assert "map_y" not in df.columns
        assert "mean_y" not in df.columns

    def test_to_dataframe_high_dimensional_environment(self):
        """to_dataframe() should use dim_0, dim_1, etc. for >3D environments."""
        from neurospatial import Environment
        from neurospatial.decoding import DecodingResult

        # Create 4D environment
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 10, (100, 4))
        env = Environment.from_samples(positions, bin_size=5.0)

        posterior = np.ones((5, env.n_bins)) / env.n_bins
        result = DecodingResult(posterior=posterior, env=env)
        df = result.to_dataframe()

        for i in range(4):
            assert f"map_dim_{i}" in df.columns
            assert f"mean_dim_{i}" in df.columns
        assert "map_x" not in df.columns

    def test_to_dataframe_success_criteria(self, small_2d_env):
        """Verify success criteria from TASKS.md."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins
        times = np.arange(n_time_bins) * 0.025
        result = DecodingResult(posterior=posterior, env=small_2d_env, times=times)

        df = result.to_dataframe()

        # Success criteria from TASKS.md
        assert "time" in df.columns or result.times is None
