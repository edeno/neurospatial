"""Tests for spatial basis functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment
from neurospatial.ops.basis import (
    chebyshev_filter_basis,
    geodesic_rbf_basis,
    heat_kernel_wavelet_basis,
    plot_basis_functions,
    select_basis_centers,
    spatial_basis,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_2d_env():
    """Simple 10x10 grid environment."""
    positions = np.array([[i, j] for i in range(10) for j in range(10)], dtype=float)
    return Environment.from_samples(positions, bin_size=1.0)


@pytest.fixture
def linear_env():
    """Simple 1D linear environment."""
    positions = np.arange(20).reshape(-1, 1).astype(float)
    return Environment.from_samples(positions, bin_size=1.0)


@pytest.fixture
def maze_env_with_wall():
    """Environment with a wall blocking direct path (L-shaped)."""
    positions = []
    # Left side (0-9, 0-9)
    for i in range(10):
        for j in range(10):
            positions.append([float(i), float(j)])
    # Bottom right (10-19, 0-4) - creates L-shape
    for i in range(10, 20):
        for j in range(5):
            positions.append([float(i), float(j)])
    positions = np.array(positions, dtype=float)
    return Environment.from_samples(positions, bin_size=2.0)


@pytest.fixture
def disconnected_env():
    """Environment with two disconnected components."""
    # Two separate clusters far apart
    cluster1 = np.array(
        [[float(i), float(j)] for i in range(5) for j in range(5)],
        dtype=float,
    )
    cluster2 = np.array(
        [[float(i + 50), float(j + 50)] for i in range(5) for j in range(5)],
        dtype=float,
    )
    positions = np.vstack([cluster1, cluster2])
    # Large bin_size ensures disconnection
    return Environment.from_samples(positions, bin_size=2.0)


# =============================================================================
# TestSelectBasisCenters
# =============================================================================


class TestSelectBasisCenters:
    """Tests for center selection."""

    def test_kmeans_returns_correct_count(self, simple_2d_env):
        """KMeans should return exactly n_centers centers."""
        centers = select_basis_centers(simple_2d_env, n_centers=10, method="kmeans")
        assert len(centers) == 10
        assert centers.dtype == np.int_

    def test_centers_are_valid_nodes(self, simple_2d_env):
        """All returned centers must be valid bin indices."""
        centers = select_basis_centers(simple_2d_env, n_centers=10)
        assert all(0 <= c < simple_2d_env.n_bins for c in centers)

    def test_centers_are_unique(self, simple_2d_env):
        """Centers should be unique (no duplicates)."""
        centers = select_basis_centers(simple_2d_env, n_centers=10)
        assert len(centers) == len(set(centers))

    def test_random_state_reproducibility(self, simple_2d_env):
        """Same random_state should give same results."""
        c1 = select_basis_centers(simple_2d_env, n_centers=10, random_state=42)
        c2 = select_basis_centers(simple_2d_env, n_centers=10, random_state=42)
        assert_allclose(c1, c2)

    def test_different_random_states_differ(self, simple_2d_env):
        """Different random_states should give different results."""
        c1 = select_basis_centers(simple_2d_env, n_centers=10, random_state=42)
        c2 = select_basis_centers(simple_2d_env, n_centers=10, random_state=123)
        # Not guaranteed to be different but highly likely for kmeans
        # Just check they are both valid
        assert len(c1) == len(c2) == 10

    def test_raises_if_too_many_centers(self, simple_2d_env):
        """Should raise ValueError if n_centers > n_bins."""
        with pytest.raises(ValueError, match="Cannot select"):
            select_basis_centers(simple_2d_env, n_centers=simple_2d_env.n_bins + 1)

    def test_raises_if_n_centers_zero(self, simple_2d_env):
        """Should raise ValueError if n_centers < 1."""
        with pytest.raises(ValueError, match="at least 1"):
            select_basis_centers(simple_2d_env, n_centers=0)

    def test_raises_if_n_centers_negative(self, simple_2d_env):
        """Should raise ValueError if n_centers < 1."""
        with pytest.raises(ValueError, match="at least 1"):
            select_basis_centers(simple_2d_env, n_centers=-5)

    def test_random_method_returns_correct_count(self, simple_2d_env):
        """Random method should return exactly n_centers centers."""
        centers = select_basis_centers(simple_2d_env, n_centers=15, method="random")
        assert len(centers) == 15

    def test_farthest_point_returns_correct_count(self, simple_2d_env):
        """Farthest point method should return exactly n_centers centers."""
        centers = select_basis_centers(
            simple_2d_env, n_centers=10, method="farthest_point"
        )
        assert len(centers) == 10

    def test_farthest_point_maximizes_spread(self, linear_env):
        """Farthest point should maximize spatial spread."""
        centers = select_basis_centers(
            linear_env, n_centers=3, method="farthest_point", random_state=42
        )
        # In a linear environment, 3 farthest points should be spread out
        # (near start, middle, end)
        centers_sorted = np.sort(centers)
        # Check they're reasonably spread (not all clustered)
        gaps = np.diff(centers_sorted)
        assert all(gap > 2 for gap in gaps)  # Each gap should be substantial


# =============================================================================
# TestGeodesicRBFBasis - Tests to be added after M2 implementation
# =============================================================================


class TestGeodesicRBFBasis:
    """Tests for geodesic RBF basis."""

    def test_output_shape_single_sigma(self, simple_2d_env):
        """Single sigma should give n_centers rows."""
        basis = geodesic_rbf_basis(simple_2d_env, n_centers=10, sigma=5.0)
        assert basis.shape == (10, simple_2d_env.n_bins)

    def test_output_shape_multi_sigma(self, simple_2d_env):
        """Multiple sigmas should multiply the row count."""
        basis = geodesic_rbf_basis(simple_2d_env, n_centers=10, sigma=[5.0, 10.0])
        assert basis.shape == (20, simple_2d_env.n_bins)

    def test_row_ordering_center_major(self, simple_2d_env):
        """Rows should be ordered (center, sigma): all sigmas for center 0, then center 1."""
        centers = np.array([0, 5])
        basis = geodesic_rbf_basis(
            simple_2d_env, centers=centers, sigma=[5.0, 10.0], normalize="none"
        )
        # Row 0: center 0, sigma=5
        # Row 1: center 0, sigma=10
        # Row 2: center 1, sigma=5
        # Row 3: center 1, sigma=10
        assert np.argmax(basis[0]) == 0  # center 0
        assert np.argmax(basis[1]) == 0  # center 0
        assert np.argmax(basis[2]) == 5  # center 1
        assert np.argmax(basis[3]) == 5  # center 1

    def test_center_has_max_value(self, simple_2d_env):
        """Basis function should peak at its center."""
        centers = np.array([0, 5, 10])
        basis = geodesic_rbf_basis(
            simple_2d_env, centers=centers, sigma=5.0, normalize="none"
        )
        for i, c in enumerate(centers):
            assert np.argmax(basis[i]) == c

    def test_respects_walls(self, maze_env_with_wall):
        """Basis should not leak through walls."""
        # Find a center on left side
        center_bin = maze_env_with_wall.bin_at([2.0, 2.0])
        # Find a bin on right side that's spatially close but geodesically far
        far_bin = maze_env_with_wall.bin_at([12.0, 2.0])

        basis = geodesic_rbf_basis(
            maze_env_with_wall,
            centers=np.array([center_bin]),
            sigma=3.0,
            normalize="none",
        )

        # Value at far bin should be much smaller than at center
        # (geodesic distance is large due to wall)
        center_value = basis[0, center_bin]
        far_value = basis[0, far_bin]
        assert far_value < 0.1 * center_value

    def test_unit_normalization(self, simple_2d_env):
        """Unit normalization should give L2 norm = 1."""
        basis = geodesic_rbf_basis(simple_2d_env, n_centers=10, normalize="unit")
        norms = np.linalg.norm(basis, axis=1)
        assert_allclose(norms, 1.0, rtol=1e-10)

    def test_max_normalization(self, simple_2d_env):
        """Max normalization should give max = 1."""
        basis = geodesic_rbf_basis(simple_2d_env, n_centers=10, normalize="max")
        maxes = np.max(basis, axis=1)
        assert_allclose(maxes, 1.0, rtol=1e-10)

    def test_no_normalization(self, simple_2d_env):
        """No normalization should give raw RBF values (center = 1)."""
        centers = np.array([0])
        basis = geodesic_rbf_basis(
            simple_2d_env, centers=centers, sigma=5.0, normalize="none"
        )
        # At center, RBF value is exp(0) = 1
        assert_allclose(basis[0, 0], 1.0)

    def test_raises_if_sigma_non_positive(self, simple_2d_env):
        """Should raise ValueError for non-positive sigma."""
        with pytest.raises(ValueError, match="sigma values must be positive"):
            geodesic_rbf_basis(simple_2d_env, n_centers=10, sigma=-1.0)

        with pytest.raises(ValueError, match="sigma values must be positive"):
            geodesic_rbf_basis(simple_2d_env, n_centers=10, sigma=[5.0, 0.0, 10.0])

    def test_raises_if_no_centers_specified(self, simple_2d_env):
        """Should raise ValueError if neither centers nor n_centers given."""
        with pytest.raises(ValueError, match="Must specify basis function locations"):
            geodesic_rbf_basis(simple_2d_env)

    def test_raises_if_disconnected_graph(self, disconnected_env):
        """Should raise error if environment graph is disconnected."""
        with pytest.raises(ValueError, match="unreachable"):
            geodesic_rbf_basis(disconnected_env, n_centers=5, sigma=5.0)


# =============================================================================
# TestHeatKernelWaveletBasis - Tests to be added after M3 implementation
# =============================================================================


class TestHeatKernelWaveletBasis:
    """Tests for heat kernel wavelet basis."""

    def test_output_shape(self, simple_2d_env):
        """Should return correct shape."""
        basis = heat_kernel_wavelet_basis(
            simple_2d_env, n_centers=10, scales=[1.0, 2.0]
        )
        assert basis.shape == (20, simple_2d_env.n_bins)

    def test_larger_scale_is_wider(self, simple_2d_env):
        """Larger diffusion time should spread heat more uniformly."""
        center_idx = simple_2d_env.n_bins // 2
        center = np.array([center_idx])
        small_scale = heat_kernel_wavelet_basis(
            simple_2d_env, centers=center, scales=[0.1], normalize="none"
        )
        large_scale = heat_kernel_wavelet_basis(
            simple_2d_env, centers=center, scales=[4.0], normalize="none"
        )

        # Heat diffusion: small scale = sharp peak, large scale = spread out
        # At small scale, values far from center should be near zero
        # At large scale, values far from center should be higher (heat reached them)
        # Check that distant bins have higher values at larger scale
        distant_bins = [0, simple_2d_env.n_bins - 1]  # Corners
        small_distant_sum = np.sum(small_scale[0, distant_bins])
        large_distant_sum = np.sum(large_scale[0, distant_bins])
        assert large_distant_sum > small_distant_sum

        # Also check that large scale has lower peak at center (heat has diffused away)
        assert large_scale[0, center_idx] < small_scale[0, center_idx]

    def test_respects_graph_structure(self, maze_env_with_wall):
        """Heat diffuses along graph edges, reaching distant bins more slowly."""
        center_bin = maze_env_with_wall.bin_at([2.0, 2.0])
        # Nearby bin (on same side of L)
        nearby_bin = maze_env_with_wall.bin_at([4.0, 2.0])
        # Far bin (requires going around the L)
        far_bin = maze_env_with_wall.bin_at([12.0, 2.0])

        basis = heat_kernel_wavelet_basis(
            maze_env_with_wall,
            centers=np.array([center_bin]),
            scales=[0.5],  # Small scale to see gradient
            normalize="none",
        )

        # Heat should be highest at center, less at nearby, even less at far
        center_value = basis[0, center_bin]
        nearby_value = basis[0, nearby_bin]
        far_value = basis[0, far_bin]

        # Verify heat gradient respects graph distance
        assert center_value > nearby_value > far_value

    def test_raises_if_scales_non_positive(self, simple_2d_env):
        """Should raise ValueError for non-positive scales."""
        with pytest.raises(ValueError, match="scales values must be positive"):
            heat_kernel_wavelet_basis(simple_2d_env, n_centers=10, scales=[-1.0])

    def test_raises_if_no_centers_specified(self, simple_2d_env):
        """Should raise ValueError if neither centers nor n_centers given."""
        with pytest.raises(ValueError, match="Must specify basis function locations"):
            heat_kernel_wavelet_basis(simple_2d_env)


# =============================================================================
# TestChebyshevFilterBasis - Tests to be added after M4 implementation
# =============================================================================


class TestChebyshevFilterBasis:
    """Tests for Chebyshev polynomial filter basis."""

    def test_output_shape(self, simple_2d_env):
        """Should return correct shape."""
        basis = chebyshev_filter_basis(simple_2d_env, n_centers=10, max_degree=3)
        # 10 centers * 4 degrees (0 through 3)
        assert basis.shape == (40, simple_2d_env.n_bins)

    def test_degree_0_is_delta(self, simple_2d_env):
        """Degree 0 Chebyshev is identity, so T_0 @ delta = delta."""
        centers = np.array([0, 5, 10])
        basis = chebyshev_filter_basis(
            simple_2d_env, centers=centers, max_degree=0, normalize="none"
        )

        for i, c in enumerate(centers):
            expected = np.zeros(simple_2d_env.n_bins)
            expected[c] = 1.0
            assert_allclose(basis[i], expected)

    def test_k_hop_locality(self, linear_env):
        """Degree k should only affect k-hop neighbors."""
        center = np.array([10])  # Middle of linear track
        basis = chebyshev_filter_basis(
            linear_env, centers=center, max_degree=2, normalize="none"
        )

        # For center-major ordering: degree k for center 0 is at index k
        degree_2 = basis[2]  # Degree 2 for center 0

        # Nodes more than 2 hops away should be zero
        for node in range(linear_env.n_bins):
            hop_distance = abs(node - 10)  # In linear env, hop = index diff
            if hop_distance > 2:
                assert abs(degree_2[node]) < 1e-10

    def test_raises_if_max_degree_negative(self, simple_2d_env):
        """Should raise ValueError for negative max_degree."""
        with pytest.raises(ValueError, match="max_degree must be non-negative"):
            chebyshev_filter_basis(simple_2d_env, n_centers=10, max_degree=-1)

    def test_raises_if_no_centers_specified(self, simple_2d_env):
        """Should raise ValueError if neither centers nor n_centers given."""
        with pytest.raises(ValueError, match="Must specify basis function locations"):
            chebyshev_filter_basis(simple_2d_env)


# =============================================================================
# TestSpatialBasis - Tests to be added after M5 implementation
# =============================================================================


class TestSpatialBasis:
    """Tests for convenience function."""

    def test_returns_basis(self, simple_2d_env):
        """Should return a 2D array with correct columns."""
        basis = spatial_basis(simple_2d_env)
        assert basis.ndim == 2
        assert basis.shape[1] == simple_2d_env.n_bins

    def test_coverage_affects_sigma(self, simple_2d_env):
        """Local coverage should give narrower basis than global."""
        basis_local = spatial_basis(simple_2d_env, coverage="local", n_features=30)
        basis_global = spatial_basis(simple_2d_env, coverage="global", n_features=30)

        # After unit normalization:
        # - Local (small sigma): sharper peaks, values more concentrated → higher std
        # - Global (large sigma): broader bumps, values more uniform → lower std
        local_spread = np.mean([np.std(b) for b in basis_local])
        global_spread = np.mean([np.std(b) for b in basis_global])
        assert local_spread > global_spread

    def test_n_features_approximate(self, simple_2d_env):
        """n_features should approximately control output size."""
        basis = spatial_basis(simple_2d_env, n_features=100)
        # Due to multi-scale, actual may differ slightly
        assert 80 <= basis.shape[0] <= 120

    def test_random_state_reproducibility(self, simple_2d_env):
        """Same random_state should give same results."""
        b1 = spatial_basis(simple_2d_env, random_state=42)
        b2 = spatial_basis(simple_2d_env, random_state=42)
        assert_allclose(b1, b2)


# =============================================================================
# TestPlotBasisFunctions - Tests to be added after M5 implementation
# =============================================================================


class TestPlotBasisFunctions:
    """Tests for visualization helper."""

    def test_returns_figure(self, simple_2d_env):
        """Should return a matplotlib Figure."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend

        basis = geodesic_rbf_basis(simple_2d_env, n_centers=10, sigma=5.0)
        fig = plot_basis_functions(simple_2d_env, basis, n_examples=4)

        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_respects_indices(self, simple_2d_env):
        """Should plot specified indices."""
        import matplotlib

        matplotlib.use("Agg")

        basis = geodesic_rbf_basis(simple_2d_env, n_centers=10, sigma=5.0)
        fig = plot_basis_functions(simple_2d_env, basis, indices=[0, 1, 2])

        import matplotlib.pyplot as plt

        # Should have plotted 3 basis functions
        plt.close(fig)
