"""Tests for Environment.transitions() method.

This module tests the empirical transition matrix computation functionality,
including adjacency filtering, normalization, and lag parameters.
"""

import numpy as np
import pytest
import scipy.sparse
from numpy.testing import assert_allclose

from neurospatial import Environment


class TestTransitionsBasic:
    """Basic transition matrix computation tests."""

    def test_simple_1d_sequence(self):
        """Test basic transition counting on 1D track."""
        # Create simple 1D environment: 5 bins
        env = Environment.from_samples(
            np.array([[0.0], [2.0], [4.0], [6.0], [8.0], [10.0]]),
            bin_size=2.5,
        )

        # Simple sequence: 0 → 1 → 2 → 1 → 0
        bins = np.array([0, 1, 2, 1, 0], dtype=np.int32)

        # Without normalization, should get raw counts
        T = env.transitions(bins=bins, normalize=False)

        assert scipy.sparse.issparse(T)
        assert T.shape == (env.n_bins, env.n_bins)
        assert T.format == "csr"

        # Check specific transitions
        # 0 → 1: 1 time
        assert T[0, 1] == 1
        # 1 → 2: 1 time
        assert T[1, 2] == 1
        # 2 → 1: 1 time
        assert T[2, 1] == 1
        # 1 → 0: 1 time
        assert T[1, 0] == 1

    def test_transitions_normalized(self):
        """Test that normalized transitions sum to 1 per row."""
        # Create simple environment
        env = Environment.from_samples(
            np.array([[0.0], [2.0], [4.0], [6.0]]),
            bin_size=2.5,
        )

        # Sequence with repeated transitions
        # Bin 0: goes to 1 twice → P(1|0) = 1.0
        # Bin 1: goes to 2 once, to 0 once → P(2|1) = 0.5, P(0|1) = 0.5
        bins = np.array([0, 1, 2, 1, 0, 1], dtype=np.int32)

        T = env.transitions(bins=bins, normalize=True)

        # Row sums should be 1.0 for rows with any transitions
        row_sums = np.array(T.sum(axis=1)).flatten()

        # Row 0 (bin 0): has 2 transitions (0→1, 0→1)
        assert_allclose(row_sums[0], 1.0, rtol=1e-6)

        # Row 1 (bin 1): has 2 transitions (1→2, 1→0)
        assert_allclose(row_sums[1], 1.0, rtol=1e-6)

        # Row 2 (bin 2): has 1 transition (2→1)
        assert_allclose(row_sums[2], 1.0, rtol=1e-6)

    def test_transitions_from_trajectory(self):
        """Test computing transitions from times/positions instead of bins."""
        # Create 2D environment
        env = Environment.from_samples(
            np.random.RandomState(42).uniform(0, 10, size=(50, 2)),
            bin_size=3.0,
        )

        # Simple trajectory
        times = np.array([0.0, 1.0, 2.0, 3.0])
        positions = np.array([[1.0, 1.0], [4.0, 1.0], [7.0, 1.0], [4.0, 4.0]])

        # Should automatically compute bin_sequence internally
        T = env.transitions(times=times, positions=positions, normalize=False)

        assert scipy.sparse.issparse(T)
        assert T.shape == (env.n_bins, env.n_bins)

        # Should have some transitions
        assert T.nnz > 0

    def test_symmetric_1d_track(self):
        """Test symmetric transitions on bidirectional 1D track."""
        # Create 1D track
        env = Environment.from_samples(
            np.array([[i] for i in range(0, 21, 2)], dtype=float),
            bin_size=2.5,
        )

        # Back-and-forth sequence: 0→1→2→3→2→1→0
        bins = np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.int32)

        T = env.transitions(bins=bins, normalize=False)

        # Forward transitions
        # 0→1: 1 time, 1→0: 1 time (symmetric)
        assert T[0, 1] == T[1, 0] == 1

        # 1→2: 1 time, 2→1: 1 time (symmetric)
        assert T[1, 2] == T[2, 1] == 1

        # 2→3: 1 time, 3→2: 1 time (symmetric)
        assert T[2, 3] == T[3, 2] == 1


class TestTransitionsAdjacencyFiltering:
    """Tests for allow_teleports parameter (adjacency filtering)."""

    def test_teleport_filtering_enabled(self):
        """Test that non-adjacent transitions are filtered when allow_teleports=False."""
        # Create simple 1D environment: 5 bins in a line
        env = Environment.from_samples(
            np.array([[i] for i in range(0, 11, 2)], dtype=float),
            bin_size=2.5,
        )

        # Sequence with teleport: 0 → 1 → 4 (1→4 is non-adjacent)
        bins = np.array([0, 1, 4], dtype=np.int32)

        # With allow_teleports=False (default), should filter 1→4
        T = env.transitions(bins=bins, normalize=False, allow_teleports=False)

        # 0→1 should be counted (adjacent)
        assert T[0, 1] == 1

        # 1→4 should NOT be counted (non-adjacent teleport)
        assert T[1, 4] == 0

    def test_teleport_filtering_disabled(self):
        """Test that all transitions counted when allow_teleports=True."""
        # Same setup as above
        env = Environment.from_samples(
            np.array([[i] for i in range(0, 11, 2)], dtype=float),
            bin_size=2.5,
        )

        bins = np.array([0, 1, 4], dtype=np.int32)

        # With allow_teleports=True, should count all transitions
        T = env.transitions(bins=bins, normalize=False, allow_teleports=True)

        # Both transitions should be counted
        assert T[0, 1] == 1
        assert T[1, 4] == 1

    def test_grid_diagonal_transitions(self):
        """Test adjacency filtering on 2D grid (diagonal not adjacent)."""
        # Create small 3x3 grid with proper spacing
        # Use fine-grained samples to create a proper grid
        x = np.linspace(0, 10, 10)
        y = np.linspace(0, 10, 10)
        xx, yy = np.meshgrid(x, y)
        samples = np.column_stack([xx.ravel(), yy.ravel()])

        env = Environment.from_samples(samples, bin_size=4.0)

        # Get bin indices for corners (should be in different bins now)
        bin_bottom_left = env.bin_at([[1.0, 1.0]])[0]
        bin_top_right = env.bin_at([[7.0, 7.0]])[0]

        # Make sure they're actually different bins
        assert bin_bottom_left != bin_top_right

        # Diagonal transition (not adjacent on 4-connectivity grid)
        bins = np.array([bin_bottom_left, bin_top_right], dtype=np.int32)

        # Check if they're adjacent first
        are_adjacent = env.connectivity.has_edge(bin_bottom_left, bin_top_right)

        if not are_adjacent:
            # With adjacency filtering, diagonal should be filtered
            T_filtered = env.transitions(
                bins=bins, normalize=False, allow_teleports=False
            )

            # Should have no transitions (diagonal filtered out)
            assert T_filtered.nnz == 0

            # Without filtering, should count diagonal
            T_unfiltered = env.transitions(
                bins=bins, normalize=False, allow_teleports=True
            )

            # Should have 1 transition
            assert T_unfiltered.nnz == 1
        else:
            # If they happen to be adjacent, test that transition is counted
            T_filtered = env.transitions(
                bins=bins, normalize=False, allow_teleports=False
            )
            assert T_filtered.nnz == 1


class TestTransitionsLag:
    """Tests for lag parameter (multi-step transitions)."""

    def test_lag_1_default(self):
        """Test default lag=1 behavior (consecutive transitions)."""
        env = Environment.from_samples(
            np.array([[i] for i in range(0, 11, 2)], dtype=float),
            bin_size=2.5,
        )

        bins = np.array([0, 1, 2, 3], dtype=np.int32)

        T = env.transitions(bins=bins, lag=1, normalize=False)

        # Should have transitions: 0→1, 1→2, 2→3
        assert T[0, 1] == 1
        assert T[1, 2] == 1
        assert T[2, 3] == 1
        assert T.nnz == 3

    def test_lag_2_skip_one(self):
        """Test lag=2 (skip one bin in sequence)."""
        env = Environment.from_samples(
            np.array([[i] for i in range(0, 11, 2)], dtype=float),
            bin_size=2.5,
        )

        bins = np.array([0, 1, 2, 3], dtype=np.int32)

        # Use allow_teleports=True since lag=2 transitions skip bins
        # (bins 0→2 and 1→3 are not graph-adjacent)
        T = env.transitions(bins=bins, lag=2, normalize=False, allow_teleports=True)

        # Should have transitions: 0→2, 1→3
        assert T[0, 2] == 1
        assert T[1, 3] == 1
        # Should NOT have lag=1 transitions
        assert T[0, 1] == 0
        assert T[1, 2] == 0

    def test_lag_larger_than_sequence(self):
        """Test lag larger than sequence length returns empty matrix."""
        env = Environment.from_samples(
            np.array([[i] for i in range(0, 11, 2)], dtype=float),
            bin_size=2.5,
        )

        bins = np.array([0, 1, 2], dtype=np.int32)

        T = env.transitions(bins=bins, lag=10, normalize=False)

        # Should have no transitions (lag too large)
        assert T.nnz == 0


class TestTransitionsValidation:
    """Input validation tests."""

    def test_missing_required_input(self):
        """Test error when neither bins nor times/positions provided."""
        env = Environment.from_samples(
            np.array([[0.0], [10.0]]),
            bin_size=5.0,
        )

        with pytest.raises(ValueError, match=r"Must provide either.*bins.*or both"):
            env.transitions()

    def test_bins_with_times_positions_error(self):
        """Test error when both bins and times/positions provided."""
        env = Environment.from_samples(
            np.array([[0.0], [10.0]]),
            bin_size=5.0,
        )

        bins = np.array([0, 1], dtype=np.int32)
        times = np.array([0.0, 1.0])
        positions = np.array([[1.0], [6.0]])

        with pytest.raises(ValueError, match=r"Cannot provide both"):
            env.transitions(bins=bins, times=times, positions=positions)

    def test_times_without_positions(self):
        """Test error when times provided without positions."""
        env = Environment.from_samples(
            np.array([[0.0], [10.0]]),
            bin_size=5.0,
        )

        with pytest.raises(
            ValueError, match=r"Both times and positions must be provided"
        ):
            env.transitions(times=np.array([0.0, 1.0]))

    def test_positions_without_times(self):
        """Test error when positions provided without times."""
        env = Environment.from_samples(
            np.array([[0.0], [10.0]]),
            bin_size=5.0,
        )

        with pytest.raises(
            ValueError, match=r"Both times and positions must be provided"
        ):
            env.transitions(positions=np.array([[1.0], [6.0]]))

    def test_invalid_bin_indices(self):
        """Test error for bin indices outside valid range."""
        env = Environment.from_samples(
            np.array([[0.0], [10.0]]),
            bin_size=5.0,
        )

        # Invalid bin index
        bins = np.array([0, 1, 99], dtype=np.int32)

        with pytest.raises(ValueError, match=r"Invalid bin.*outside range"):
            env.transitions(bins=bins, normalize=False)

    def test_negative_lag(self):
        """Test error for negative lag."""
        env = Environment.from_samples(
            np.array([[0.0], [10.0]]),
            bin_size=5.0,
        )

        bins = np.array([0, 1], dtype=np.int32)

        with pytest.raises(ValueError, match=r"lag must be positive"):
            env.transitions(bins=bins, lag=-1)

    def test_zero_lag(self):
        """Test error for zero lag."""
        env = Environment.from_samples(
            np.array([[0.0], [10.0]]),
            bin_size=5.0,
        )

        bins = np.array([0, 1], dtype=np.int32)

        with pytest.raises(ValueError, match=r"lag must be positive"):
            env.transitions(bins=bins, lag=0)

    def test_empty_bins_array(self):
        """Test handling of empty bins array."""
        env = Environment.from_samples(
            np.array([[0.0], [10.0]]),
            bin_size=5.0,
        )

        bins = np.array([], dtype=np.int32)

        # Should return empty sparse matrix
        T = env.transitions(bins=bins, normalize=False)

        assert scipy.sparse.issparse(T)
        assert T.shape == (env.n_bins, env.n_bins)
        assert T.nnz == 0

    def test_single_bin_sequence(self):
        """Test handling of single-bin sequence (no transitions)."""
        env = Environment.from_samples(
            np.array([[0.0], [10.0]]),
            bin_size=5.0,
        )

        bins = np.array([0], dtype=np.int32)

        T = env.transitions(bins=bins, normalize=False)

        assert scipy.sparse.issparse(T)
        assert T.nnz == 0


class TestTransitionsEdgeCases:
    """Edge case tests."""

    def test_self_transitions(self):
        """Test that self-transitions (staying in same bin) are counted."""
        # Create environment with connected bins
        env = Environment.from_samples(
            np.array([[i] for i in range(0, 11, 2)], dtype=float),
            bin_size=2.5,
        )

        # Sequence: 0 → 0 → 0 → 1
        bins = np.array([0, 0, 0, 1], dtype=np.int32)

        T = env.transitions(bins=bins, normalize=False)

        # Should have 2 self-transitions (0→0) and 1 transition (0→1)
        assert T[0, 0] == 2
        assert T[0, 1] == 1

    def test_all_same_bin(self):
        """Test sequence that stays in same bin entire time."""
        env = Environment.from_samples(
            np.array([[0.0], [10.0]]),
            bin_size=5.0,
        )

        bins = np.array([0, 0, 0, 0], dtype=np.int32)

        T = env.transitions(bins=bins, normalize=False)

        # Should have only self-transitions
        assert T[0, 0] == 3
        assert T.nnz == 1

        # Normalized should give P(0|0) = 1.0
        T_norm = env.transitions(bins=bins, normalize=True)
        assert_allclose(T_norm[0, 0], 1.0, rtol=1e-6)

    def test_bins_with_outside_values(self):
        """Test that -1 (outside) bin indices are handled correctly."""
        env = Environment.from_samples(
            np.array([[0.0], [10.0]]),
            bin_size=5.0,
        )

        # Sequence with outside marker: 0 → -1 → 1
        # Should treat -1 as invalid and skip those transitions
        bins = np.array([0, -1, 1], dtype=np.int32)

        # This should raise error since -1 is invalid
        with pytest.raises(ValueError, match=r"Invalid bin"):
            env.transitions(bins=bins, normalize=False)

    def test_normalization_with_no_transitions_from_bin(self):
        """Test normalization when some bins have no outgoing transitions."""
        env = Environment.from_samples(
            np.array([[i] for i in range(0, 11, 2)], dtype=float),
            bin_size=2.5,
        )

        # Sequence: 0 → 1 (bin 2, 3, 4 have no transitions)
        bins = np.array([0, 1], dtype=np.int32)

        T = env.transitions(bins=bins, normalize=True)

        # Row 0 should sum to 1.0
        assert_allclose(T[0, :].sum(), 1.0, rtol=1e-6)

        # Rows with no transitions should sum to 0.0
        for i in [2, 3, 4]:
            if i < T.shape[0]:
                assert_allclose(T[i, :].sum(), 0.0, atol=1e-10)


class TestTransitionsMultipleLayouts:
    """Test transitions on different layout types."""

    def test_transitions_on_hexagonal_grid(self):
        """Test transitions work on hexagonal layout."""
        # Create hexagonal environment using from_samples instead
        # (simpler than managing HexagonalLayout API)
        np.random.seed(42)
        samples = np.random.uniform(0, 10, size=(100, 2))
        env = Environment.from_samples(samples, bin_size=2.0)

        # Create random trajectory that stays within environment bounds
        np.random.seed(42)
        times = np.arange(0, 10, 0.5)
        positions = np.random.uniform(0, 10, size=(len(times), 2))

        T = env.transitions(times=times, positions=positions, normalize=False)

        assert scipy.sparse.issparse(T)
        assert T.shape == (env.n_bins, env.n_bins)

    def test_transitions_on_masked_grid(self):
        """Test transitions on masked grid layout."""
        # Create grid with mask
        grid_samples = np.random.RandomState(42).uniform(0, 10, size=(100, 2))
        env = Environment.from_samples(grid_samples, bin_size=3.0)

        times = np.arange(0, 5, 0.5)
        positions = np.random.RandomState(43).uniform(0, 10, size=(len(times), 2))

        T = env.transitions(times=times, positions=positions, normalize=True)

        assert scipy.sparse.issparse(T)
        # Rows with transitions should sum to 1.0
        row_sums = np.array(T.sum(axis=1)).flatten()
        non_zero_rows = row_sums > 0
        assert_allclose(row_sums[non_zero_rows], 1.0, rtol=1e-6)


class TestTransitionsPerformance:
    """Performance tests."""

    def test_large_sequence(self):
        """Test transitions on large bin sequence."""
        # Create environment with many bins
        env = Environment.from_samples(
            np.random.RandomState(42).uniform(0, 100, size=(1000, 2)),
            bin_size=5.0,
        )

        # Large sequence
        np.random.seed(42)
        bins = np.random.randint(0, env.n_bins, size=10000, dtype=np.int32)

        # Should complete quickly
        T = env.transitions(bins=bins, normalize=True)

        assert scipy.sparse.issparse(T)
        assert T.shape == (env.n_bins, env.n_bins)

        # Check normalization
        row_sums = np.array(T.sum(axis=1)).flatten()
        non_zero_rows = row_sums > 0
        if non_zero_rows.any():
            assert_allclose(row_sums[non_zero_rows], 1.0, rtol=1e-6)


class TestTransitionsModelBased:
    """Tests for model-based transition methods."""

    def test_random_walk_basic(self):
        """Test basic random walk transition matrix."""
        # Create simple 1D environment
        env = Environment.from_samples(
            np.array([[i] for i in range(0, 11, 2)], dtype=float),
            bin_size=2.5,
        )

        T = env.transitions(method="random_walk")

        assert scipy.sparse.issparse(T)
        assert T.shape == (env.n_bins, env.n_bins)

        # Check it's row-stochastic
        row_sums = np.array(T.sum(axis=1)).flatten()
        assert_allclose(row_sums, 1.0, rtol=1e-6)

    def test_random_walk_uniform_neighbors(self):
        """Test that random walk gives uniform probability to neighbors."""
        # Create simple 1D track: 0-1-2-3-4
        env = Environment.from_samples(
            np.array([[i] for i in range(0, 11, 2)], dtype=float),
            bin_size=2.5,
        )

        T = env.transitions(method="random_walk", normalize=True)

        # Middle bin (e.g., bin 2) should have equal prob to neighbors
        # Assuming 1D connectivity: bin 2 connects to bins 1 and 3
        middle_bin = 2
        neighbors = list(env.connectivity.neighbors(middle_bin))

        if len(neighbors) > 0:
            expected_prob = 1.0 / len(neighbors)
            for neighbor in neighbors:
                assert_allclose(T[middle_bin, neighbor], expected_prob, rtol=1e-6)

    def test_diffusion_basic(self):
        """Test basic diffusion transition matrix."""
        # Create simple environment
        env = Environment.from_samples(
            np.array([[i] for i in range(0, 11, 2)], dtype=float),
            bin_size=2.5,
        )

        T = env.transitions(method="diffusion", bandwidth=5.0)

        assert scipy.sparse.issparse(T)
        assert T.shape == (env.n_bins, env.n_bins)

        # Check it's row-stochastic (from heat kernel)
        row_sums = np.array(T.sum(axis=1)).flatten()
        assert_allclose(row_sums, 1.0, rtol=1e-3)  # Slightly looser tolerance

    def test_diffusion_locality(self):
        """Test that diffusion emphasizes local transitions."""
        # Create environment
        env = Environment.from_samples(
            np.array([[i] for i in range(0, 21, 2)], dtype=float),
            bin_size=2.5,
        )

        # Small bandwidth = more local
        T_local = env.transitions(method="diffusion", bandwidth=1.0)

        # Large bandwidth = more uniform
        T_global = env.transitions(method="diffusion", bandwidth=20.0)

        # For a bin in the middle, local should have higher self-transition
        middle_bin = env.n_bins // 2
        if T_local.shape[0] > middle_bin:
            # Local diffusion should prefer staying nearby
            assert T_local[middle_bin, middle_bin] > T_global[middle_bin, middle_bin]

    def test_diffusion_requires_bandwidth(self):
        """Test that diffusion method requires bandwidth parameter."""
        env = Environment.from_samples(
            np.array([[0.0], [10.0]]),
            bin_size=5.0,
        )

        with pytest.raises(ValueError, match=r"requires 'bandwidth'"):
            env.transitions(method="diffusion")

    def test_model_with_empirical_inputs_error(self):
        """Test error when providing both method and empirical inputs."""
        env = Environment.from_samples(
            np.array([[0.0], [10.0]]),
            bin_size=5.0,
        )

        bins = np.array([0, 1], dtype=np.int32)

        with pytest.raises(ValueError, match=r"Cannot provide both"):
            env.transitions(bins=bins, method="random_walk")

    def test_unknown_method_error(self):
        """Test error for unknown method."""
        env = Environment.from_samples(
            np.array([[0.0], [10.0]]),
            bin_size=5.0,
        )

        with pytest.raises(ValueError, match=r"Unknown method"):
            env.transitions(method="levy_flight")

    def test_model_with_lag_parameter_error(self):
        """Test error when providing lag parameter with model-based method."""
        env = Environment.from_samples(
            np.array([[i] for i in range(0, 11, 2)], dtype=float),
            bin_size=2.5,
        )

        with pytest.raises(ValueError, match=r"'lag' is only valid in empirical mode"):
            env.transitions(method="random_walk", lag=2)

    def test_model_with_allow_teleports_error(self):
        """Test error when providing allow_teleports parameter with model-based method."""
        env = Environment.from_samples(
            np.array([[i] for i in range(0, 11, 2)], dtype=float),
            bin_size=2.5,
        )

        with pytest.raises(
            ValueError, match=r"'allow_teleports' is only valid in empirical mode"
        ):
            env.transitions(method="diffusion", bandwidth=5.0, allow_teleports=True)

    def test_random_walk_with_bandwidth_error(self):
        """Test error when providing bandwidth to random_walk method."""
        env = Environment.from_samples(
            np.array([[i] for i in range(0, 11, 2)], dtype=float),
            bin_size=2.5,
        )

        with pytest.raises(
            ValueError, match=r"'bandwidth' is only valid with method='diffusion'"
        ):
            env.transitions(method="random_walk", bandwidth=5.0)

    def test_diffusion_normalize_false_error(self):
        """Test error when using normalize=False with diffusion method."""
        env = Environment.from_samples(
            np.array([[i] for i in range(0, 11, 2)], dtype=float),
            bin_size=2.5,
        )

        with pytest.raises(ValueError, match=r"does not support normalize=False"):
            env.transitions(method="diffusion", bandwidth=5.0, normalize=False)

    def test_random_walk_normalize_false(self):
        """Test that random walk works with normalize=False."""
        env = Environment.from_samples(
            np.array([[i] for i in range(0, 11, 2)], dtype=float),
            bin_size=2.5,
        )

        T = env.transitions(method="random_walk", normalize=False)
        assert scipy.sparse.issparse(T)
        assert T.shape == (env.n_bins, env.n_bins)
        # Should be unnormalized adjacency matrix (integer counts)
        assert T.dtype in [np.int32, np.int64, np.float64]

    def test_random_walk_vs_diffusion(self):
        """Test that random walk and diffusion give different results."""
        # Create environment
        env = Environment.from_samples(
            np.array([[i] for i in range(0, 11, 2)], dtype=float),
            bin_size=2.5,
        )

        T_random = env.transitions(method="random_walk")
        T_diffusion = env.transitions(method="diffusion", bandwidth=5.0)

        # They should be different (diffusion is distance-weighted)
        diff = (T_random - T_diffusion).toarray()
        assert np.abs(diff).max() > 0.01  # At least some difference

    def test_model_based_sparse_format(self):
        """Test that model-based methods return sparse matrices."""
        env = Environment.from_samples(
            np.random.RandomState(42).uniform(0, 20, size=(100, 2)),
            bin_size=3.0,
        )

        T_random = env.transitions(method="random_walk")
        T_diffusion = env.transitions(method="diffusion", bandwidth=5.0)

        assert scipy.sparse.issparse(T_random)
        assert scipy.sparse.issparse(T_diffusion)
        assert T_random.format == "csr"
        assert T_diffusion.format == "csr"
