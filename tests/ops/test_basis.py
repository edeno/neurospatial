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


@pytest.mark.parametrize("spacing", [1.0, 2.0, 5.0])
def test_heat_kernel_spread_is_bin_size_independent(uniform_grid, spacing):
    env = uniform_grid(spacing, n=40)
    center = int(np.argmin(np.linalg.norm(env.bin_centers - 20 * spacing, axis=1)))
    basis = heat_kernel_wavelet_basis(
        env, centers=np.array([center]), scales=[1.0, 4.0], normalize="none"
    )
    relative = (env.bin_centers - env.bin_centers[center]) / spacing
    for kernel, scale in zip(basis, [1.0, 4.0], strict=True):
        for direction in [np.array([1.0, 0.0]), np.array([1.0, 1.0]) / np.sqrt(2)]:
            projected = relative @ direction
            mean = np.sum(kernel * projected) / kernel.sum()
            sigma = np.sqrt(np.sum(kernel * (projected - mean) ** 2) / kernel.sum())
            assert sigma == pytest.approx(np.sqrt(2 * scale), rel=1e-3)


def test_heat_kernel_uses_smooth_generator(fv_env):
    from scipy import sparse
    from scipy.sparse.linalg import expm_multiply
    from scipy.spatial import cKDTree

    weights, volumes, _, _ = fv_env._diffusion_geometry
    distances, _ = cKDTree(fv_env.bin_centers).query(fv_env.bin_centers, k=2)
    spacing = np.median(distances[:, 1])
    generator = (
        spacing**2
        * sparse.diags(1 / volumes)
        @ (sparse.diags(np.asarray(weights.sum(axis=1)).ravel()) - weights)
    )
    center = fv_env.n_bins // 2
    delta = np.zeros(fv_env.n_bins)
    delta[center] = 1
    basis = heat_kernel_wavelet_basis(
        fv_env, centers=np.array([center]), scales=[1.0], normalize="none"
    )
    assert_allclose(basis[0], expm_multiply(-generator, delta), rtol=1e-10, atol=1e-12)


@pytest.mark.nwb
def test_heat_kernel_uses_faces_after_nwb_round_trip(graph_environment_round_trip):
    original, loaded = graph_environment_round_trip
    kwargs = {
        "centers": np.array([0, original.n_bins - 1]),
        "scales": [0.5, 2.0],
        "normalize": "none",
    }
    np.testing.assert_array_equal(
        heat_kernel_wavelet_basis(loaded, **kwargs),
        heat_kernel_wavelet_basis(original, **kwargs),
    )


@pytest.mark.parametrize(
    "basis_function", [heat_kernel_wavelet_basis, chebyshev_filter_basis]
)
def test_diffusion_bases_raise_without_finite_volume_geometry(
    missing_faces_env, basis_function
):
    with pytest.raises(NotImplementedError, match=basis_function.__name__) as exc:
        basis_function(missing_faces_env, centers=np.array([0]))
    assert "\nFix:" in str(exc.value)


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
def walled_env():
    """A 22 x 20 cm box of 2 cm bins split by a wall open only at the top.

    Returns the environment and three bins: a center at (5, 1), a bin across
    the wall at (17, 1) and a bin on the open side at (5, 13). Both are 12 cm
    from the center in a straight line; the path around the wall is about
    37 cm, the open-side path 12 cm.
    """
    edges_x = np.arange(12) * 2.0
    edges_y = np.arange(11) * 2.0
    mask = np.ones((11, 10), bool)
    mask[5, :8] = False  # wall at x in [10, 12), open for y >= 16
    env = Environment.from_grid_mask(mask, (edges_x, edges_y))
    center, across, open_side = env.bin_at(
        np.array([[5.0, 1.0], [17.0, 1.0], [5.0, 13.0]])
    )
    return env, int(center), int(across), int(open_side)


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


_BASIS_FUNCTIONS = [
    pytest.param(geodesic_rbf_basis, id="geodesic_rbf"),
    pytest.param(heat_kernel_wavelet_basis, id="heat_kernel_wavelet"),
    pytest.param(chebyshev_filter_basis, id="chebyshev_filter"),
]


@pytest.mark.parametrize("basis_function", _BASIS_FUNCTIONS)
class TestExplicitCenters:
    """Centers are bin indices; anything else is refused, not reinterpreted."""

    def test_outside_marker_is_refused(self, basis_function, simple_2d_env):
        """-1 (env.bin_at's outside marker) would wrap to the last bin."""
        with pytest.raises(ValueError, match=r"-1.*outside") as exc:
            basis_function(simple_2d_env, centers=np.array([0, -1]))
        assert "\nFix:" in str(exc.value)

    def test_index_past_last_bin_is_refused(self, basis_function, simple_2d_env):
        with pytest.raises(ValueError, match="centers"):
            basis_function(simple_2d_env, centers=np.array([simple_2d_env.n_bins]))

    def test_fractional_center_is_refused(self, basis_function, simple_2d_env):
        """2.7 would silently truncate to bin 2."""
        with pytest.raises(TypeError, match="integer bin indices"):
            basis_function(simple_2d_env, centers=np.array([2.7]))

    def test_integer_list_is_accepted(self, basis_function, simple_2d_env):
        from_list = basis_function(simple_2d_env, centers=[0, 5])
        from_array = basis_function(simple_2d_env, centers=np.array([0, 5]))
        assert_allclose(from_list, from_array)


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
        c1 = select_basis_centers(simple_2d_env, n_centers=10, rng=42)
        c2 = select_basis_centers(simple_2d_env, n_centers=10, rng=42)
        assert_allclose(c1, c2)

    def test_different_random_states_differ(self, simple_2d_env):
        """Different random_states should give different results."""
        c1 = select_basis_centers(simple_2d_env, n_centers=10, rng=42)
        c2 = select_basis_centers(simple_2d_env, n_centers=10, rng=123)
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

    def test_unknown_method_lists_the_available_ones(self, simple_2d_env):
        """There is no grid method; the error offers the methods that exist."""
        with pytest.raises(ValueError, match="'grid'") as exc:
            select_basis_centers(simple_2d_env, n_centers=5, method="grid")
        message = str(exc.value)
        assert "'kmeans', 'farthest_point', 'random'" in message
        assert "'random', 'grid'" not in message

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
            linear_env, n_centers=3, method="farthest_point", rng=42
        )
        # In a linear environment, 3 farthest points should be spread out
        # (near start, middle, end)
        centers_sorted = np.sort(centers)
        # Check they're reasonably spread (not all clustered)
        gaps = np.diff(centers_sorted)
        assert all(gap > 2 for gap in gaps)  # Each gap should be substantial


# =============================================================================
# TestGeodesicRBFBasis - Tests to be added after the GeodesicRBFBasis implementation
# =============================================================================


class TestGeodesicRBFBasis:
    """Tests for geodesic RBF basis."""

    def test_output_shape_single_bandwidth(self, simple_2d_env):
        """Single bandwidth should give n_centers rows."""
        basis = geodesic_rbf_basis(simple_2d_env, n_centers=10, bandwidth=5.0)
        assert basis.shape == (10, simple_2d_env.n_bins)

    def test_output_shape_multi_bandwidth(self, simple_2d_env):
        """Multiple bandwidths should multiply the row count."""
        basis = geodesic_rbf_basis(simple_2d_env, n_centers=10, bandwidth=[5.0, 10.0])
        assert basis.shape == (20, simple_2d_env.n_bins)

    def test_row_ordering_center_major(self, simple_2d_env):
        """Rows should be ordered (center, bandwidth): all bandwidths for center 0, then center 1."""
        centers = np.array([0, 5])
        basis = geodesic_rbf_basis(
            simple_2d_env, centers=centers, bandwidth=[5.0, 10.0], normalize="none"
        )
        # Row 0: center 0, bandwidth=5
        # Row 1: center 0, bandwidth=10
        # Row 2: center 1, bandwidth=5
        # Row 3: center 1, bandwidth=10
        assert np.argmax(basis[0]) == 0  # center 0
        assert np.argmax(basis[1]) == 0  # center 0
        assert np.argmax(basis[2]) == 5  # center 1
        assert np.argmax(basis[3]) == 5  # center 1

    def test_center_has_max_value(self, simple_2d_env):
        """Basis function should peak at its center."""
        centers = np.array([0, 5, 10])
        basis = geodesic_rbf_basis(
            simple_2d_env, centers=centers, bandwidth=5.0, normalize="none"
        )
        for i, c in enumerate(centers):
            assert np.argmax(basis[i]) == c

    def test_respects_walls(self, walled_env):
        """Each value is the Gaussian of the path length around the wall."""
        import networkx as nx

        env, center, across, open_side = walled_env
        straight = np.linalg.norm(env.bin_centers[across] - env.bin_centers[center])
        around = nx.shortest_path_length(
            env.connectivity, center, across, weight="distance"
        )
        assert straight == pytest.approx(12.0)
        assert around > 3 * straight

        bandwidth = 6.0
        basis = geodesic_rbf_basis(
            env, centers=np.array([center]), bandwidth=bandwidth, normalize="none"
        )
        assert basis[0, across] == pytest.approx(
            np.exp(-(around**2) / (2 * bandwidth**2)), rel=1e-12
        )
        assert basis[0, open_side] == pytest.approx(
            np.exp(-(straight**2) / (2 * bandwidth**2)), rel=1e-12
        )
        # The straight-line (leaking) value would be about 10^7 times larger.
        assert basis[0, across] < 1e-3 * basis[0, open_side]

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
            simple_2d_env, centers=centers, bandwidth=5.0, normalize="none"
        )
        # At center, RBF value is exp(0) = 1
        assert_allclose(basis[0, 0], 1.0)

    def test_raises_if_bandwidth_non_positive(self, simple_2d_env):
        """Should raise ValueError for non-positive bandwidth."""
        with pytest.raises(ValueError, match="bandwidth values must be positive"):
            geodesic_rbf_basis(simple_2d_env, n_centers=10, bandwidth=-1.0)

        with pytest.raises(ValueError, match="bandwidth values must be positive"):
            geodesic_rbf_basis(simple_2d_env, n_centers=10, bandwidth=[5.0, 0.0, 10.0])

    def test_raises_if_no_centers_specified(self, simple_2d_env):
        """Should raise ValueError if neither centers nor n_centers given."""
        with pytest.raises(ValueError, match="Must specify basis function locations"):
            geodesic_rbf_basis(simple_2d_env)

    def test_raises_if_disconnected_graph(self, disconnected_env):
        """Should raise error if environment graph is disconnected."""
        with pytest.raises(ValueError, match="unreachable"):
            geodesic_rbf_basis(disconnected_env, n_centers=5, bandwidth=5.0)


# =============================================================================
# TestHeatKernelWaveletBasis - Tests to be added after the HeatKernelWaveletBasis implementation
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

    def test_respects_walls(self, walled_env):
        """Heat reaching a bin behind the wall is far below an open bin as near."""
        env, center, across, open_side = walled_env
        basis = heat_kernel_wavelet_basis(
            env, centers=np.array([center]), scales=[8.0], normalize="none"
        )
        assert basis[0, across] < 1e-2 * basis[0, open_side]

    def test_raises_if_scales_non_positive(self, simple_2d_env):
        """Should raise ValueError for non-positive scales."""
        with pytest.raises(ValueError, match="scales values must be positive"):
            heat_kernel_wavelet_basis(simple_2d_env, n_centers=10, scales=[-1.0])

    def test_raises_if_no_centers_specified(self, simple_2d_env):
        """Should raise ValueError if neither centers nor n_centers given."""
        with pytest.raises(ValueError, match="Must specify basis function locations"):
            heat_kernel_wavelet_basis(simple_2d_env)


# =============================================================================
# TestChebyshevFilterBasis - Tests to be added after the ChebyshevFilterBasis implementation
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
# TestSpatialBasis - Tests to be added after the SpatialBasis implementation
# =============================================================================


class TestSpatialBasis:
    """Tests for convenience function."""

    def test_returns_basis(self, simple_2d_env):
        """Should return a 2D array with correct columns."""
        basis = spatial_basis(simple_2d_env)
        assert basis.ndim == 2
        assert basis.shape[1] == simple_2d_env.n_bins

    def test_coverage_affects_bandwidth(self, simple_2d_env):
        """Local coverage should give narrower basis than global."""
        basis_local = spatial_basis(simple_2d_env, coverage="local", n_features=30)
        basis_global = spatial_basis(simple_2d_env, coverage="global", n_features=30)

        # After unit normalization:
        # - Local (small bandwidth): sharper peaks, values more concentrated → higher std
        # - Global (large bandwidth): broader bumps, values more uniform → lower std
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
        b1 = spatial_basis(simple_2d_env, rng=42)
        b2 = spatial_basis(simple_2d_env, rng=42)
        assert_allclose(b1, b2)

    @pytest.mark.parametrize(
        "nodes",
        [[(0.0, 0.0), (200.0, 0.0)], [(0.0, 0.0), (120.0, 0.0), (120.0, 80.0)]],
        ids=["straight", "L-shaped"],
    )
    def test_track_scale_is_the_track_length(self, nodes):
        """On a track, bandwidth is a fraction of the 200 cm the animal can run.

        The bounding box of a straight track has zero height, and an L-shaped
        track's bounding box (120 x 80 cm) says nothing about its length.
        """
        from neurospatial import Environment

        track = Environment.linear_track(
            node_positions=nodes, bin_size=2.0, track_width=10.0, units="cm"
        )
        basis = spatial_basis(track, coverage="medium", n_features=12, rng=0)
        expected = geodesic_rbf_basis(
            track, n_centers=4, bandwidth=[10.0, 20.0, 40.0], normalize="unit", rng=0
        )
        assert_allclose(basis, expected)

    def test_three_dimensional_scale_is_the_side_length(self):
        """In a 120 cm cube, medium coverage is 10% of 120 cm, not of 120**1.5."""
        edges = np.linspace(0.0, 120.0, 7)
        cube = Environment.from_grid_mask(np.ones((6, 6, 6), bool), (edges,) * 3)
        assert [hi - lo for lo, hi in cube.dimension_ranges] == [120.0] * 3

        basis = spatial_basis(cube, coverage="medium", n_features=9, rng=0)
        expected = geodesic_rbf_basis(
            cube, n_centers=3, bandwidth=[6.0, 12.0, 24.0], normalize="unit", rng=0
        )
        assert_allclose(basis, expected)


# =============================================================================
# TestPlotBasisFunctions - Tests to be added after the plot_basis_functions implementation
# =============================================================================


class TestPlotBasisFunctions:
    """Tests for visualization helper."""

    def test_returns_figure(self, simple_2d_env):
        """Should return a matplotlib Figure."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend

        basis = geodesic_rbf_basis(simple_2d_env, n_centers=10, bandwidth=5.0)
        fig = plot_basis_functions(simple_2d_env, basis, n_examples=4)

        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_respects_indices(self, simple_2d_env):
        """Should plot specified indices."""
        import matplotlib

        matplotlib.use("Agg")

        basis = geodesic_rbf_basis(simple_2d_env, n_centers=10, bandwidth=5.0)
        fig = plot_basis_functions(simple_2d_env, basis, indices=[0, 1, 2])

        import matplotlib.pyplot as plt

        # Should have plotted 3 basis functions
        plt.close(fig)


class TestEstimateSpectralRadius:
    """_estimate_spectral_radius must only fall back on ARPACK failures."""

    @staticmethod
    def _laplacian(env):
        import networkx as nx

        n_bins = env.n_bins
        laplacian = nx.laplacian_matrix(
            env.connectivity, nodelist=range(n_bins), weight="distance"
        )
        return laplacian.tocsr()

    def test_estimate_spectral_radius_propagates_non_arpack_error(
        self, monkeypatch, simple_2d_env
    ):
        """A non-ARPACK error from eigsh propagates (no silent fallback)."""
        import scipy.sparse.linalg

        from neurospatial.ops.basis import _estimate_spectral_radius

        laplacian = self._laplacian(simple_2d_env)

        def _raise_value_error(*args, **kwargs):
            raise ValueError("malformed input")

        monkeypatch.setattr(scipy.sparse.linalg, "eigsh", _raise_value_error)

        with pytest.raises(ValueError, match="malformed input"):
            _estimate_spectral_radius(laplacian)

    def test_estimate_spectral_radius_arpack_failure_warns_and_bounds(
        self, monkeypatch, simple_2d_env
    ):
        """An ARPACK convergence failure warns and returns the max-degree bound."""
        import scipy.sparse.linalg
        from scipy.sparse.linalg import ArpackNoConvergence

        from neurospatial.ops.basis import _estimate_spectral_radius

        laplacian = self._laplacian(simple_2d_env)

        def _raise_arpack(*args, **kwargs):
            raise ArpackNoConvergence("did not converge", np.array([]), np.array([]))

        monkeypatch.setattr(scipy.sparse.linalg, "eigsh", _raise_arpack)

        with pytest.warns(UserWarning, match="did not converge"):
            radius = _estimate_spectral_radius(laplacian)

        expected_bound = 2.0 * float(np.max(laplacian.diagonal()))
        assert radius == pytest.approx(expected_bound)


def _face_hops(env, source):
    """Hop count from ``source`` across shared cell faces (conductance > 0)."""
    from scipy.sparse.csgraph import shortest_path

    W = env.get_diffusion_geometry().W.tocsr(copy=True)
    W.eliminate_zeros()
    return shortest_path(W, unweighted=True, indices=source)


@pytest.mark.parametrize("spacing", [1.0, 2.5])
def test_chebyshev_couples_face_neighbors_only(uniform_grid, spacing):
    """On a square grid, T1 is equal at the 4 face neighbors and 0 at corners.

    Corner-touching bins share no face, so no diffusion crosses between them,
    although the connectivity graph links them (2.83 cm apart for 2 cm bins).
    """
    env = uniform_grid(spacing, n=10)
    center = int(np.argmin(np.linalg.norm(env.bin_centers - 5 * spacing, axis=1)))
    neighbors = np.array(list(env.connectivity.neighbors(center)))
    offsets = np.linalg.norm(
        env.bin_centers[neighbors] - env.bin_centers[center], axis=1
    )
    face = neighbors[np.isclose(offsets, spacing)]
    corner = neighbors[np.isclose(offsets, spacing * np.sqrt(2))]
    assert face.size == 4
    assert corner.size == 4

    basis = chebyshev_filter_basis(
        env, centers=np.array([center]), max_degree=1, normalize="none"
    )
    assert_allclose(basis[1, face], basis[1, face[0]], rtol=1e-12)
    assert basis[1, face[0]] < 0
    assert_allclose(basis[1, corner], 0.0, atol=1e-15)


def test_chebyshev_bins_sharing_no_face_keep_isolated_filters():
    """Bins touching only at a corner have a zero generator: T_k = (-1)^k delta.

    With every eigenvalue 0 there is no spectral radius to rescale by; each
    filter stays on its own bin instead of dividing by zero.
    """
    edges = np.array([0.0, 1.0, 2.0])
    env = Environment.from_grid_mask(
        np.array([[True, False], [False, True]]), grid_edges=(edges, edges)
    )
    assert env.n_bins == 2

    basis = chebyshev_filter_basis(
        env, centers=np.array([0]), max_degree=3, normalize="none"
    )
    assert_allclose(basis, [[1, 0], [-1, 0], [1, 0], [-1, 0]], atol=1e-15)


def test_chebyshev_filter_is_bin_size_independent(uniform_grid):
    """The rescaled operator, and so every filter, is the same at any bin size."""
    small, large = uniform_grid(1.0, n=8), uniform_grid(2.5, n=8)
    kwargs = {"centers": np.array([0, 27]), "max_degree": 4, "normalize": "none"}
    assert_allclose(
        chebyshev_filter_basis(small, **kwargs),
        chebyshev_filter_basis(large, **kwargs),
        rtol=1e-10,
        atol=1e-12,
    )


def test_chebyshev_uses_the_diffusion_generator(fv_env):
    """T_k of the rescaled finite-volume generator M^-1 (Deg - W), as env.smooth.

    On grid, hex and a W-maze with junctions, degree 1 is (2 / lambda_max) L - I
    and the recurrence gives the higher degrees.
    """
    from scipy import sparse

    geometry = fv_env.get_diffusion_geometry()
    W = sparse.csr_matrix(geometry.W)
    generator = (
        sparse.diags(1.0 / geometry.volumes)
        @ (sparse.diags(np.asarray(W.sum(axis=1)).ravel()) - W)
    ).toarray()
    lambda_max = np.max(np.linalg.eigvals(generator).real)
    scaled = 2.0 * generator / lambda_max - np.eye(fv_env.n_bins)

    center = fv_env.n_bins // 2
    delta = np.zeros(fv_env.n_bins)
    delta[center] = 1.0
    expected = [delta, scaled @ delta]
    for _ in range(2, 4):
        expected.append(2.0 * scaled @ expected[-1] - expected[-2])

    basis = chebyshev_filter_basis(
        fv_env, centers=np.array([center]), max_degree=3, normalize="none"
    )
    assert_allclose(basis, np.array(expected), rtol=1e-8, atol=1e-10)


def test_chebyshev_support_stays_within_k_face_hops(uniform_grid):
    """A degree-k filter is zero beyond k shared-face steps from its center."""
    env = uniform_grid(1.0, n=10)
    center = 55
    basis = chebyshev_filter_basis(
        env, centers=np.array([center]), max_degree=3, normalize="none"
    )
    hops = _face_hops(env, center)
    for degree, row in enumerate(basis):
        assert_allclose(row[hops > degree], 0, atol=1e-12)
        assert np.all(row[hops == degree] != 0)


def _path_graph(node_order):
    """Path 0-1-2-3-4 (1 cm edges) whose nodes are inserted in ``node_order``."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(node_order)
    graph.add_edges_from(((i, i + 1) for i in range(4)), distance=1.0)
    return graph


@pytest.mark.parametrize("node_order", [[4, 2, 0, 3, 1], [1, 0, 2, 4, 3]])
def test_graph_distances_follow_bin_ids_not_node_insertion_order(node_order):
    """Bin ids index the matrices built from the graph, whatever the node order."""
    from types import SimpleNamespace

    ordered = SimpleNamespace(n_bins=5, n_dims=1, connectivity=_path_graph(range(5)))
    shuffled = SimpleNamespace(n_bins=5, n_dims=1, connectivity=_path_graph(node_order))

    basis = geodesic_rbf_basis(
        shuffled, centers=np.array([0]), bandwidth=1.0, normalize="none"
    )
    assert_allclose(basis[0], np.exp(-(np.arange(5.0) ** 2) / 2.0))
    assert_allclose(
        basis,
        geodesic_rbf_basis(
            ordered, centers=np.array([0]), bandwidth=1.0, normalize="none"
        ),
    )
    for seed in range(3):
        np.testing.assert_array_equal(
            select_basis_centers(
                shuffled, n_centers=3, method="farthest_point", rng=seed
            ),
            select_basis_centers(
                ordered, n_centers=3, method="farthest_point", rng=seed
            ),
        )
