"""Tests to verify ops.basis exports work correctly after move.

These tests verify that basis functions are correctly importable from
their new location at neurospatial.ops.basis.
"""

import numpy as np
import pytest

from neurospatial import Environment

# Also verify they're available from the ops package namespace
from neurospatial.ops import (
    chebyshev_filter_basis as cheb_from_ops,
)
from neurospatial.ops import (
    geodesic_rbf_basis as rbf_from_ops,
)
from neurospatial.ops import (
    heat_kernel_wavelet_basis as heat_from_ops,
)
from neurospatial.ops import (
    plot_basis_functions as plot_from_ops,
)
from neurospatial.ops import (
    select_basis_centers as centers_from_ops,
)
from neurospatial.ops import (
    spatial_basis as spatial_from_ops,
)

# These imports must work after the move
from neurospatial.ops.basis import (
    chebyshev_filter_basis,
    geodesic_rbf_basis,
    heat_kernel_wavelet_basis,
    plot_basis_functions,
    select_basis_centers,
    spatial_basis,
)


@pytest.fixture
def simple_2d_env():
    """Simple 10x10 grid environment for testing."""
    positions = np.array([[i, j] for i in range(10) for j in range(10)], dtype=float)
    return Environment.from_samples(positions, bin_size=1.0)


class TestOpsImports:
    """Test that all basis functions are importable from ops.basis."""

    def test_select_basis_centers_import(self):
        """select_basis_centers should be importable from ops.basis."""
        assert select_basis_centers is not None
        assert callable(select_basis_centers)
        assert select_basis_centers is centers_from_ops

    def test_geodesic_rbf_basis_import(self):
        """geodesic_rbf_basis should be importable from ops.basis."""
        assert geodesic_rbf_basis is not None
        assert callable(geodesic_rbf_basis)
        assert geodesic_rbf_basis is rbf_from_ops

    def test_heat_kernel_wavelet_basis_import(self):
        """heat_kernel_wavelet_basis should be importable from ops.basis."""
        assert heat_kernel_wavelet_basis is not None
        assert callable(heat_kernel_wavelet_basis)
        assert heat_kernel_wavelet_basis is heat_from_ops

    def test_chebyshev_filter_basis_import(self):
        """chebyshev_filter_basis should be importable from ops.basis."""
        assert chebyshev_filter_basis is not None
        assert callable(chebyshev_filter_basis)
        assert chebyshev_filter_basis is cheb_from_ops

    def test_spatial_basis_import(self):
        """spatial_basis should be importable from ops.basis."""
        assert spatial_basis is not None
        assert callable(spatial_basis)
        assert spatial_basis is spatial_from_ops

    def test_plot_basis_functions_import(self):
        """plot_basis_functions should be importable from ops.basis."""
        assert plot_basis_functions is not None
        assert callable(plot_basis_functions)
        assert plot_basis_functions is plot_from_ops


class TestBasicFunctionality:
    """Verify core functionality still works after move."""

    def test_select_basis_centers_basic(self, simple_2d_env):
        """select_basis_centers should work with basic inputs."""
        centers = select_basis_centers(simple_2d_env, n_centers=10, random_state=42)
        assert len(centers) == 10
        assert centers.dtype == np.int_
        assert all(0 <= c < simple_2d_env.n_bins for c in centers)

    def test_geodesic_rbf_basis_basic(self, simple_2d_env):
        """geodesic_rbf_basis should produce correct output shape."""
        basis = geodesic_rbf_basis(simple_2d_env, n_centers=5, sigma=3.0)
        assert basis.shape == (5, simple_2d_env.n_bins)
        assert not np.any(np.isnan(basis))

    def test_heat_kernel_wavelet_basis_basic(self, simple_2d_env):
        """heat_kernel_wavelet_basis should produce correct output shape."""
        basis = heat_kernel_wavelet_basis(simple_2d_env, n_centers=5, scales=[0.5, 1.0])
        assert basis.shape == (10, simple_2d_env.n_bins)  # 5 centers * 2 scales
        assert not np.any(np.isnan(basis))

    def test_chebyshev_filter_basis_basic(self, simple_2d_env):
        """chebyshev_filter_basis should produce correct output shape."""
        basis = chebyshev_filter_basis(simple_2d_env, n_centers=5, max_degree=2)
        assert basis.shape == (15, simple_2d_env.n_bins)  # 5 centers * 3 degrees
        assert not np.any(np.isnan(basis))

    def test_spatial_basis_basic(self, simple_2d_env):
        """spatial_basis should return a valid basis matrix."""
        basis = spatial_basis(simple_2d_env, n_features=30, random_state=42)
        assert basis.ndim == 2
        assert basis.shape[1] == simple_2d_env.n_bins
        assert not np.any(np.isnan(basis))

    def test_plot_basis_functions_basic(self, simple_2d_env):
        """plot_basis_functions should return a matplotlib Figure."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        basis = geodesic_rbf_basis(simple_2d_env, n_centers=5, sigma=3.0)
        fig = plot_basis_functions(simple_2d_env, basis, n_examples=2)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
