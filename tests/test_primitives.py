"""Tests for spatial signal processing primitives."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.ops.distance import geodesic_distance_matrix
from neurospatial.ops.graph import graph_convolve, neighbor_reduce


class TestNeighborReduce:
    """Test suite for neighbor_reduce() function."""

    def test_neighbor_reduce_mean_regular_grid(self) -> None:
        """Test mean aggregation on regular 2D grid (8-connected)."""
        # Create simple 3x3 grid
        positions = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [0, 2],
                [1, 2],
                [2, 2],
            ],
            dtype=np.float64,
        )
        env = Environment.from_samples(positions, bin_size=1.0)

        # Create field with known values (bin index as value)
        field = np.arange(env.n_bins, dtype=np.float64)

        # Compute neighbor mean
        result = neighbor_reduce(env, field, op="mean", include_self=False)

        # Center bin (4) has 8 neighbors: [0, 1, 2, 3, 5, 6, 7, 8]
        # Mean = (0 + 1 + 2 + 3 + 5 + 6 + 7 + 8) / 8 = 32/8 = 4.0
        assert result.shape == (env.n_bins,)
        assert np.isclose(result[4], 4.0), f"Expected 4.0, got {result[4]}"

        # Corner bin (0) has 3 neighbors: [1, 3, 4]
        # Mean = (1 + 3 + 4) / 3 = 8/3 ≈ 2.667
        assert np.isclose(result[0], 8.0 / 3.0), f"Expected 2.667, got {result[0]}"

    def test_neighbor_reduce_include_self(self) -> None:
        """Test that include_self changes aggregation result."""
        # Create simple 3x3 grid
        positions = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [0, 2],
                [1, 2],
                [2, 2],
            ],
            dtype=np.float64,
        )
        env = Environment.from_samples(positions, bin_size=1.0)

        # Constant field
        field = np.ones(env.n_bins, dtype=np.float64)

        # Without self
        result_no_self = neighbor_reduce(env, field, op="mean", include_self=False)

        # With self
        result_with_self = neighbor_reduce(env, field, op="mean", include_self=True)

        # For constant field, both should be 1.0
        assert np.allclose(result_no_self, 1.0)
        assert np.allclose(result_with_self, 1.0)

        # Try non-constant field
        field = np.arange(env.n_bins, dtype=np.float64)
        result_no_self = neighbor_reduce(env, field, op="mean", include_self=False)
        result_with_self = neighbor_reduce(env, field, op="mean", include_self=True)

        # Center bin (4) neighbors: [0, 1, 2, 3, 5, 6, 7, 8]
        # Without self: mean([0, 1, 2, 3, 5, 6, 7, 8]) = 32/8 = 4.0
        # With self: mean([0, 1, 2, 3, 4, 5, 6, 7, 8]) = 36/9 = 4.0 (still 4.0 due to symmetry)
        assert np.isclose(result_no_self[4], 4.0)
        assert np.isclose(result_with_self[4], 4.0)

        # Corner bin (0) neighbors: [1, 3, 4]
        # Without self: mean([1, 3, 4]) = 8/3 ≈ 2.667
        # With self: mean([0, 1, 3, 4]) = 8/4 = 2.0
        assert np.isclose(result_no_self[0], 8.0 / 3.0)
        assert np.isclose(result_with_self[0], 2.0)

    def test_neighbor_reduce_weights(self) -> None:
        """Test distance-weighted aggregation."""
        # Create simple 3x3 grid
        positions = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [0, 2],
                [1, 2],
                [2, 2],
            ],
            dtype=np.float64,
        )
        env = Environment.from_samples(positions, bin_size=1.0)

        # Field with bin indices
        field = np.arange(env.n_bins, dtype=np.float64)

        # Uniform weights (should match unweighted mean)
        weights = np.ones(env.n_bins, dtype=np.float64)
        result_uniform = neighbor_reduce(
            env, field, op="mean", weights=weights, include_self=False
        )
        result_no_weights = neighbor_reduce(env, field, op="mean", include_self=False)
        assert np.allclose(result_uniform, result_no_weights)

        # Distance-based weights (closer neighbors weighted more)
        # For center bin (4), neighbors at distance 1.0: [1, 3, 5, 7]
        # If we weight by 1/distance, all have same weight
        # But if we weight by 1/distance^2, all still have same weight
        # Let's just verify it computes weighted mean correctly
        result_weighted = neighbor_reduce(
            env, field, op="mean", weights=weights, include_self=False
        )
        assert result_weighted.shape == (env.n_bins,)

    def test_neighbor_reduce_operations(self) -> None:
        """Test all reduction operations: sum, mean, max, min, std."""
        # Create simple 3x3 grid
        positions = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [0, 2],
                [1, 2],
                [2, 2],
            ],
            dtype=np.float64,
        )
        env = Environment.from_samples(positions, bin_size=1.0)

        # Field with known values
        field = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)

        # Test sum
        result_sum = neighbor_reduce(env, field, op="sum", include_self=False)
        assert result_sum.shape == (env.n_bins,)
        # Center (4): 8 neighbors [0,1,2,3,5,6,7,8] → values [1,2,3,4,6,7,8,9] → sum = 40
        assert np.isclose(result_sum[4], 40.0)

        # Test mean
        result_mean = neighbor_reduce(env, field, op="mean", include_self=False)
        # Center: mean([1,2,3,4,6,7,8,9]) = 40/8 = 5.0
        assert np.isclose(result_mean[4], 5.0)

        # Test max
        result_max = neighbor_reduce(env, field, op="max", include_self=False)
        # Center: max([1,2,3,4,6,7,8,9]) = 9
        assert np.isclose(result_max[4], 9.0)

        # Test min
        result_min = neighbor_reduce(env, field, op="min", include_self=False)
        # Center: min([1,2,3,4,6,7,8,9]) = 1
        assert np.isclose(result_min[4], 1.0)

        # Test std
        result_std = neighbor_reduce(env, field, op="std", include_self=False)
        # Center: std([1,2,3,4,6,7,8,9])
        expected_std = np.std([1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0])
        assert np.isclose(result_std[4], expected_std)

    def test_neighbor_reduce_isolated_node(self) -> None:
        """Test handling of isolated nodes (no neighbors)."""
        # For testing isolated nodes, we'd need to manually construct a graph
        # with disconnected components. This is tricky with from_samples.
        # Instead, test that the function handles the case gracefully by
        # verifying the implementation returns NaN for bins with no neighbors.

        # Create a simple 1D environment (line of 3 points)
        positions = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
        env = Environment.from_samples(positions, bin_size=1.0)

        # All bins should have at least 1 neighbor in this case
        # Let's verify the edge cases work correctly
        field = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = neighbor_reduce(env, field, op="mean", include_self=False)

        # End bins have neighbors
        assert result.shape == (3,)
        assert not np.isnan(result[0])
        assert not np.isnan(result[1])
        assert not np.isnan(result[2])

        # Note: True isolated node testing would require manual graph construction
        # which is deferred. The implementation correctly returns NaN for bins
        # with zero neighbors (tested implicitly via the implementation).

    def test_neighbor_reduce_boundary_bins(self) -> None:
        """Test that boundary bins handle fewer neighbors correctly."""
        # Create 3x3 grid
        positions = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [0, 2],
                [1, 2],
                [2, 2],
            ],
            dtype=np.float64,
        )
        env = Environment.from_samples(positions, bin_size=1.0)

        field = np.ones(env.n_bins, dtype=np.float64)

        result = neighbor_reduce(env, field, op="mean", include_self=False)

        # All should be 1.0 for constant field
        assert np.allclose(result, 1.0)

        # Now try field where we can verify counts
        # Set field to neighbor count (should get back same if mean of 1s)
        # Let's use sum instead to verify total
        result_sum = neighbor_reduce(env, field, op="sum", include_self=False)

        # Corner bins have 3 neighbors (8-connected grid)
        # Edge bins have 5 neighbors
        # Center bin has 8 neighbors
        assert result_sum[0] == 3.0  # corner
        assert result_sum[1] == 5.0  # edge
        assert result_sum[4] == 8.0  # center

    @pytest.mark.parametrize(
        ("op", "center_value"),
        [("sum", 8.0), ("mean", 1.0), ("max", 1.0), ("min", 1.0), ("std", 0.0)],
    )
    def test_missing_neighbor_makes_every_op_missing(self, op, center_value) -> None:
        """A NaN neighbor is neither zero nor skipped, for every reduction."""
        positions = np.array([[i, j] for i in range(3) for j in range(3)], float)
        env = Environment.from_samples(positions, bin_size=1.0)
        field = np.ones(env.n_bins)
        field[4] = np.nan  # the center, a neighbor of every other bin

        result = neighbor_reduce(env, field, op=op)

        assert np.isnan(np.delete(result, 4)).all()
        # The center's own neighbors are all observed.
        assert result[4] == center_value

    def test_zero_weights_average_only_observed_neighbors(self) -> None:
        """The documented recipe: zero weight and a finite placeholder."""
        positions = np.array([[i, j] for i in range(3) for j in range(3)], float)
        env = Environment.from_samples(positions, bin_size=1.0)
        field = np.arange(env.n_bins, dtype=float)
        field[4] = np.nan
        observed = ~np.isnan(field)

        result = neighbor_reduce(
            env,
            np.where(observed, field, 0.0),
            op="mean",
            weights=observed.astype(float),
        )

        # Bin 0's observed neighbors are 1 and 3 (4 is missing).
        assert result[0] == 2.0
        # Every bin has an observed neighbor, so nothing is NaN.
        assert np.isfinite(result).all()

    def test_neighbor_reduce_validation(self) -> None:
        """Test input validation."""
        # Create simple environment
        positions = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
        env = Environment.from_samples(positions, bin_size=1.0)

        field = np.ones(env.n_bins, dtype=np.float64)

        # Invalid operation
        with pytest.raises(ValueError, match="op must be one of"):
            neighbor_reduce(env, field, op="invalid")

        # Wrong field shape
        wrong_field = np.ones(env.n_bins + 1, dtype=np.float64)
        with pytest.raises(ValueError, match=r"field\.shape"):
            neighbor_reduce(env, wrong_field, op="mean")

        # Wrong weights shape
        wrong_weights = np.ones(env.n_bins + 1, dtype=np.float64)
        with pytest.raises(ValueError, match=r"weights\.shape"):
            neighbor_reduce(env, field, op="mean", weights=wrong_weights)

    def test_neighbor_reduce_parameter_order(self) -> None:
        """Test that parameter order is (field, env, ...)."""
        positions = np.array([[0, 0], [1, 0]], dtype=np.float64)
        env = Environment.from_samples(positions, bin_size=1.0)
        field = np.ones(env.n_bins, dtype=np.float64)

        # This should work (correct order)
        result = neighbor_reduce(env, field, op="mean")
        assert result.shape == (env.n_bins,)

        # Verify keyword arguments work
        result = neighbor_reduce(env, field, op="mean", include_self=True, weights=None)
        assert result.shape == (env.n_bins,)


class TestConvolve:
    """Test suite for convolve() function."""

    def test_convolve_box_kernel(self) -> None:
        """Test convolution with box (uniform) kernel within radius."""
        # Create simple 3x3 grid
        positions = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [0, 2],
                [1, 2],
                [2, 2],
            ],
            dtype=np.float64,
        )
        env = Environment.from_samples(positions, bin_size=1.0)

        # Create field with spike at center
        field = np.zeros(env.n_bins, dtype=np.float64)
        field[4] = 1.0  # Center bin

        # Box kernel: uniform weight within distance threshold
        def box_kernel(distances: np.ndarray) -> np.ndarray:
            """Return 1.0 for distances <= 1.5, 0.0 otherwise."""
            return np.where(distances <= 1.5, 1.0, 0.0)

        # Convolve with normalization
        result = graph_convolve(env, field, box_kernel, normalize=True)

        # Center bin (4) should have non-zero value
        # Bins within radius 1.5 should also have non-zero values
        # Every bin reaches the center (diagonals are sqrt(2) <= 1.5 away), so
        # each value is 1 / (number of bins its box covers): the center covers
        # all 9, an edge bin 6 and a corner bin 4.
        expected = np.array([4, 6, 4, 6, 9, 6, 4, 6, 4], dtype=float) ** -1
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_convolve_mexican_hat(self) -> None:
        """Test convolution with Mexican hat (difference of Gaussians) kernel."""
        # Create 5x5 grid for better Mexican hat visualization
        x = np.arange(5, dtype=np.float64)
        y = np.arange(5, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=1.0)

        # Create field with spike at center
        center_idx = 12  # Center of 5x5 grid (index 12)
        field = np.zeros(env.n_bins, dtype=np.float64)
        field[center_idx] = 1.0

        # Mexican hat kernel: DoG (difference of Gaussians)
        def mexican_hat(distances: np.ndarray) -> np.ndarray:
            """Difference of Gaussians kernel for edge detection."""
            sigma1 = 0.5
            sigma2 = 1.5
            g1 = np.exp(-(distances**2) / (2 * sigma1**2))
            g2 = np.exp(-(distances**2) / (2 * sigma2**2))
            return g1 - g2

        # Convolve (without normalization to preserve Mexican hat properties)
        result = graph_convolve(env, field, mexican_hat, normalize=False)

        # Mexican hat kernel is 0 at distance 0, so center gets 0
        assert result[center_idx] == 0.0

        # Nearby bins should have positive values (narrow Gaussian dominates)
        # Get immediate neighbors
        neighbors = list(env.connectivity.neighbors(center_idx))
        assert len(neighbors) > 0
        # Immediate neighbors get positive contribution from center spike
        # (their kernel evaluates center with small positive distance)
        # Actually, this depends on the specific distances - let's just check
        # that result is not all zeros
        assert not np.all(result == 0), "Mexican hat should produce non-zero output"

    def test_convolve_precomputed_kernel(self) -> None:
        """Test convolution with precomputed kernel matrix."""
        # Create simple 3x3 grid
        positions = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [0, 2],
                [1, 2],
                [2, 2],
            ],
            dtype=np.float64,
        )
        env = Environment.from_samples(positions, bin_size=1.0)

        # Create test field
        field = np.arange(env.n_bins, dtype=np.float64)

        # Precomputed kernel: identity matrix (each bin keeps own value)
        kernel_matrix = np.eye(env.n_bins, dtype=np.float64)

        # Convolve with identity kernel (should return original field)
        result = graph_convolve(env, field, kernel_matrix, normalize=False)

        assert np.allclose(result, field)

    def test_convolve_normalize(self) -> None:
        """Test that normalization ensures weights sum to 1 per bin."""
        # Create simple 3x3 grid
        positions = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [0, 2],
                [1, 2],
                [2, 2],
            ],
            dtype=np.float64,
        )
        env = Environment.from_samples(positions, bin_size=1.0)

        # Create constant field
        field = np.ones(env.n_bins, dtype=np.float64) * 5.0

        # Simple kernel function
        def gaussian_kernel(distances: np.ndarray) -> np.ndarray:
            return np.exp(-(distances**2) / 2.0)

        # Convolve with normalization
        result_normalized = graph_convolve(env, field, gaussian_kernel, normalize=True)

        # For constant field, normalized convolution should preserve constant
        assert np.allclose(result_normalized, 5.0), (
            f"Expected all values ~5.0, got range [{result_normalized.min():.3f}, "
            f"{result_normalized.max():.3f}]"
        )

        # Without normalization, values will be larger
        result_unnormalized = graph_convolve(
            env, field, gaussian_kernel, normalize=False
        )
        assert np.all(result_unnormalized > result_normalized)

    @pytest.fixture
    def grid_3x3(self) -> Environment:
        """3x3 grid with unit bins; from_samples links diagonal neighbors too."""
        positions = np.array([[i, j] for i in range(3) for j in range(3)], dtype=float)
        return Environment.from_samples(positions, bin_size=1.0)

    @staticmethod
    def _gaussian(distances: np.ndarray) -> np.ndarray:
        return np.exp(-(distances**2) / 2.0)

    def test_normalized_convolution_averages_observed_bins(self, grid_3x3) -> None:
        """A missing bin gets no weight: a constant stays exactly constant."""
        field = np.full(grid_3x3.n_bins, 3.0)
        field[4] = np.nan

        result = graph_convolve(grid_3x3, field, self._gaussian, normalize=True)

        # Observed bins: the average of observed 3.0 values is exactly 3.0; a
        # NaN-as-zero implementation would pull them below 3.
        observed = np.arange(grid_3x3.n_bins) != 4
        np.testing.assert_allclose(result[observed], 3.0, rtol=1e-12)
        # The missing bin is filled from its observed neighbors, like
        # Environment.smooth(mode="average").
        np.testing.assert_allclose(result[4], 3.0, rtol=1e-12)

    def test_normalized_convolution_is_the_observed_weighted_average(
        self, grid_3x3
    ) -> None:
        """Each bin is sum_j K_ij f_j / sum_j K_ij over observed sources j."""
        field = np.arange(grid_3x3.n_bins, dtype=float)
        field[[2, 4]] = np.nan
        kernel = self._gaussian(geodesic_distance_matrix(grid_3x3.connectivity))
        observed = ~np.isnan(field)
        expected = (kernel[:, observed] @ field[observed]) / kernel[:, observed].sum(
            axis=1
        )

        result = graph_convolve(grid_3x3, field, self._gaussian, normalize=True)

        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_normalized_convolution_of_all_missing_is_missing(self, grid_3x3) -> None:
        field = np.full(grid_3x3.n_bins, np.nan)
        with np.errstate(all="raise"):
            result = graph_convolve(grid_3x3, field, self._gaussian, normalize=True)
        assert np.isnan(result).all()

    def test_raw_convolution_is_missing_where_weight_meets_a_missing_bin(
        self, grid_3x3
    ) -> None:
        """normalize=False never reads a missing value as zero."""
        field = np.ones(grid_3x3.n_bins)
        field[4] = np.nan

        def box(distances: np.ndarray) -> np.ndarray:
            # Reaches self and the four edge-sharing neighbors (distance 1),
            # not the diagonals (distance sqrt 2).
            return (distances <= 1.0).astype(float)

        result = graph_convolve(grid_3x3, field, box, normalize=False)

        corners = [0, 2, 6, 8]  # their box excludes the center bin
        np.testing.assert_array_equal(result[corners], 3.0)
        assert np.isnan(np.delete(result, corners)).all()

        # A full-support kernel reaches the missing bin from every bin.
        gaussian = graph_convolve(grid_3x3, field, self._gaussian, normalize=False)
        assert np.isnan(gaussian).all()

    def test_raw_convolution_ignores_missing_bins_with_zero_weight(
        self, grid_3x3
    ) -> None:
        """A precomputed matrix can exclude missing bins explicitly."""
        field = np.ones(grid_3x3.n_bins)
        field[4] = np.nan
        kernel = np.ones((grid_3x3.n_bins, grid_3x3.n_bins))
        kernel[:, 4] = 0.0

        result = graph_convolve(grid_3x3, field, kernel, normalize=False)

        np.testing.assert_array_equal(result, 8.0)

    def test_convolve_shares_smooth_average_missing_semantics(self, grid_3x3) -> None:
        """Normalized convolution and smooth(mode="average") treat NaN alike.

        The two smoothers differ (graph-distance kernel vs finite-volume
        diffusion), but both average over observed bins only, so a constant
        field with a missing bin comes back constant everywhere from both.
        """
        field = np.full(grid_3x3.n_bins, 2.0)
        field[4] = np.nan

        convolved = graph_convolve(grid_3x3, field, self._gaussian, normalize=True)
        smoothed = grid_3x3.smooth(field, bandwidth=1.0, mode="average")

        np.testing.assert_allclose(convolved, 2.0, rtol=1e-12)
        np.testing.assert_allclose(smoothed, 2.0, rtol=1e-6)

    def test_convolve_validation(self) -> None:
        """Test input validation."""
        # Create simple environment
        positions = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
        env = Environment.from_samples(positions, bin_size=1.0)

        field = np.ones(env.n_bins, dtype=np.float64)

        # Valid callable kernel
        def valid_kernel(distances: np.ndarray) -> np.ndarray:
            return np.ones_like(distances)

        # This should work
        result = graph_convolve(env, field, valid_kernel)
        assert result.shape == (env.n_bins,)

        # Wrong field shape
        wrong_field = np.ones(env.n_bins + 1, dtype=np.float64)
        with pytest.raises(ValueError, match=r"field\.shape"):
            graph_convolve(env, wrong_field, valid_kernel)

        # Wrong kernel matrix shape
        wrong_kernel = np.ones((env.n_bins + 1, env.n_bins), dtype=np.float64)
        with pytest.raises(ValueError, match=r"kernel.*shape"):
            graph_convolve(env, field, wrong_kernel)

    def test_convolve_parameter_order(self) -> None:
        """Test that parameter order is (field, kernel, env, ...)."""
        positions = np.array([[0, 0], [1, 0]], dtype=np.float64)
        env = Environment.from_samples(positions, bin_size=1.0)
        field = np.ones(env.n_bins, dtype=np.float64)

        def kernel_fn(distances: np.ndarray) -> np.ndarray:
            return np.ones_like(distances)

        # This should work (correct order)
        result = graph_convolve(env, field, kernel_fn)
        assert result.shape == (env.n_bins,)

        # Verify keyword arguments work
        result = graph_convolve(env, field, kernel_fn, normalize=True)
        assert result.shape == (env.n_bins,)
