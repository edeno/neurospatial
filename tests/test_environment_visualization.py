"""Tests for Environment visualization methods (plot, plot_1d, plot_field)."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PatchCollection, QuadMesh

from neurospatial import Environment


class TestPlotField:
    """Test suite for Environment.plot_field() method."""

    def test_grid_layout_pcolormesh(self):
        """Test that grid layouts use pcolormesh (QuadMesh)."""
        positions = np.random.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions, bin_size=5.0)
        field = np.random.rand(env.n_bins)

        fig, ax = plt.subplots()
        result_ax = env.plot_field(field, ax=ax, colorbar=False)

        # Check that pcolormesh was used (creates QuadMesh)
        assert any(isinstance(c, QuadMesh) for c in ax.collections)
        assert result_ax is ax

        plt.close(fig)

    def test_hexagonal_layout_patches(self):
        """Test that hexagonal layouts use PatchCollection."""
        from neurospatial.layout import create_layout

        # Create hex environment using correct parameters
        data = np.random.uniform(0, 100, (100, 2))
        layout = create_layout(
            kind="hexagonal",
            hexagon_width=10.0,
            data_samples=data,
            infer_active_bins=True,
        )
        env = Environment(layout=layout, layout_type_used="hexagonal")
        field = np.random.rand(env.n_bins)

        fig, ax = plt.subplots()
        result_ax = env.plot_field(field, ax=ax, colorbar=False)

        # Check that PatchCollection was added
        assert any(isinstance(c, PatchCollection) for c in ax.collections)
        assert result_ax is ax

        plt.close(fig)

    def test_triangular_mesh_layout(self):
        """Test that triangular mesh layouts render correctly."""
        from shapely.geometry import Polygon

        from neurospatial.layout.engines.triangular_mesh import TriangularMeshLayout

        # Create triangular mesh using correct parameters
        polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        layout = TriangularMeshLayout()
        layout.build(polygon, point_spacing=20.0)

        env = Environment(layout=layout, layout_type_used="triangular_mesh")
        field = np.random.rand(env.n_bins)

        fig, ax = plt.subplots()
        result_ax = env.plot_field(field, ax=ax, colorbar=False)

        # Check that something was plotted
        assert len(ax.collections) > 0 or len(ax.patches) > 0
        assert result_ax is ax

        plt.close(fig)

    def test_scatter_fallback(self):
        """Test scatter plot fallback for unknown layouts."""
        positions = np.random.uniform(0, 100, (500, 2))
        env = Environment.from_samples(positions, bin_size=10.0)
        field = np.random.rand(env.n_bins)

        fig, ax = plt.subplots()
        # Scatter is used as fallback for regular grid
        result_ax = env.plot_field(field, ax=ax, colorbar=False)

        assert result_ax is ax
        plt.close(fig)

    def test_nan_handling_skip(self):
        """Test that NaN bins are skipped when nan_color=None."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)), bin_size=10.0
        )
        field = np.random.rand(env.n_bins)
        field[::2] = np.nan  # Set half to NaN

        fig, ax = plt.subplots()
        env.plot_field(field, ax=ax, nan_color=None, colorbar=False)

        # Should not raise, bins should be skipped
        plt.close(fig)

    def test_nan_handling_color(self):
        """Test that NaN bins are colored when nan_color is set."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)), bin_size=10.0
        )
        field = np.random.rand(env.n_bins)
        field[::2] = np.nan

        fig, ax = plt.subplots()
        env.plot_field(field, ax=ax, nan_color="gray", colorbar=False)

        # Should render with gray color for NaN
        plt.close(fig)

    def test_all_nan_field_with_skip(self):
        """Test behavior when all field values are NaN and nan_color=None."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)), bin_size=10.0
        )
        field = np.full(env.n_bins, np.nan)

        fig, ax = plt.subplots()
        result_ax = env.plot_field(field, ax=ax, nan_color=None, colorbar=False)

        # Should produce empty plot without error
        assert result_ax is ax
        plt.close(fig)

    def test_auto_vmin_vmax(self):
        """Test automatic vmin/vmax from data."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)), bin_size=10.0
        )
        field = np.random.uniform(5.0, 15.0, env.n_bins)

        fig, ax = plt.subplots()
        env.plot_field(field, ax=ax, colorbar=False)

        # Check that something was plotted
        assert len(ax.collections) > 0 or len(ax.patches) > 0

        plt.close(fig)

    def test_custom_vmin_vmax(self):
        """Test explicit vmin/vmax."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)), bin_size=10.0
        )
        field = np.random.rand(env.n_bins)

        fig, ax = plt.subplots()
        env.plot_field(field, ax=ax, vmin=0.0, vmax=10.0, colorbar=False)

        # Check that plot was created
        assert len(ax.collections) > 0 or len(ax.patches) > 0

        plt.close(fig)

    def test_colorbar_creation(self):
        """Test colorbar creation."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)), bin_size=10.0
        )
        field = np.random.rand(env.n_bins)

        fig, ax = plt.subplots()
        env.plot_field(field, ax=ax, colorbar=True, colorbar_label="Test Label")

        # Check that figure has a colorbar axes
        assert len(fig.axes) == 2  # main ax + colorbar ax

        plt.close(fig)

    def test_no_colorbar(self):
        """Test disabling colorbar."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)), bin_size=10.0
        )
        field = np.random.rand(env.n_bins)

        fig, ax = plt.subplots()
        env.plot_field(field, ax=ax, colorbar=False)

        # Check that only main axes exists
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_invalid_field_shape(self):
        """Test that invalid field shape raises ValueError."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)), bin_size=10.0
        )

        # Wrong shape
        bad_field = np.random.rand(env.n_bins + 10)

        with pytest.raises(ValueError, match="field must be 1D array"):
            env.plot_field(bad_field)

    def test_invalid_field_2d(self):
        """Test that 2D field raises ValueError."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)), bin_size=10.0
        )

        # 2D array instead of 1D
        bad_field = np.random.rand(env.n_bins, 2)

        with pytest.raises(ValueError, match="field must be 1D array"):
            env.plot_field(bad_field)

    def test_3d_environment_not_supported(self):
        """Test that >2D environments raise NotImplementedError."""
        # Create 3D environment
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 3)), bin_size=10.0
        )
        field = np.random.rand(env.n_bins)

        with pytest.raises(NotImplementedError, match=r"Cannot plot.*3D"):
            env.plot_field(field)

    def test_custom_colormap(self):
        """Test custom colormap."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)), bin_size=10.0
        )
        field = np.random.rand(env.n_bins)

        fig, ax = plt.subplots()
        env.plot_field(field, ax=ax, cmap="hot", colorbar=False)

        # Should not raise
        plt.close(fig)

    def test_rasterized_option(self):
        """Test rasterized option."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)), bin_size=10.0
        )
        field = np.random.rand(env.n_bins)

        fig, ax = plt.subplots()
        env.plot_field(field, ax=ax, rasterized=True, colorbar=False)
        env.plot_field(field, ax=ax, rasterized=False, colorbar=False)

        # Should not raise
        plt.close(fig)

    def test_axes_labels_2d(self):
        """Test that axes labels are set for 2D plots."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)), bin_size=10.0
        )
        env.units = "cm"
        field = np.random.rand(env.n_bins)

        fig, ax = plt.subplots()
        env.plot_field(field, ax=ax, colorbar=False)

        # Check labels include units
        assert "cm" in ax.get_xlabel()
        assert "cm" in ax.get_ylabel()

        plt.close(fig)

    def test_axes_aspect_equal(self):
        """Test that axes aspect ratio is set to equal for 2D plots."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)), bin_size=10.0
        )
        field = np.random.rand(env.n_bins)

        fig, ax = plt.subplots()
        env.plot_field(field, ax=ax, colorbar=False)

        # Check aspect ratio (matplotlib converts "equal" to 1.0)
        assert ax.get_aspect() == 1.0

        plt.close(fig)

    def test_creates_axes_if_none(self):
        """Test that axes are created if not provided."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)), bin_size=10.0
        )
        field = np.random.rand(env.n_bins)

        result_ax = env.plot_field(field, ax=None, colorbar=False)

        # Check that axes were created
        assert isinstance(result_ax, matplotlib.axes.Axes)

        plt.close(result_ax.figure)

    def test_vmin_equals_vmax_handling(self):
        """Test handling when vmin >= vmax."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)), bin_size=10.0
        )
        # All same value
        field = np.full(env.n_bins, 5.0)

        fig, ax = plt.subplots()
        # Should not raise, should adjust vmax
        env.plot_field(field, ax=ax, colorbar=False)

        plt.close(fig)

    def test_integration_with_compute_place_field(self):
        """Integration test: visualize computed place field."""
        from neurospatial import compute_place_field

        # Generate synthetic data
        positions = np.random.uniform(20, 80, (1000, 2))
        times = np.linspace(0, 100, 1000)
        spike_times = np.random.uniform(0, 100, 50)

        # Create environment
        env = Environment.from_samples(positions, bin_size=5.0)

        # Compute place field
        firing_rate = compute_place_field(
            env, spike_times, times, positions, bandwidth=8.0
        )

        # Visualize
        fig, ax = plt.subplots(figsize=(8, 7))
        result_ax = env.plot_field(
            firing_rate, ax=ax, cmap="hot", colorbar_label="Firing Rate (Hz)", vmin=0
        )

        assert result_ax is not None
        assert len(fig.axes) == 2  # main + colorbar

        plt.close(fig)


class TestPlotFieldEdgeCases:
    """Test edge cases and error handling for plot_field()."""

    def test_empty_environment(self):
        """Test plotting on environment with minimal bins."""
        # Create tiny environment
        positions = np.array([[50, 50], [51, 51]])
        env = Environment.from_samples(positions, bin_size=10.0)
        field = np.random.rand(env.n_bins)

        fig, ax = plt.subplots()
        env.plot_field(field, ax=ax, colorbar=False)

        plt.close(fig)

    def test_inf_values_in_field(self):
        """Test handling of inf values in field."""
        env = Environment.from_samples(
            np.random.uniform(0, 100, (500, 2)), bin_size=10.0
        )
        field = np.random.rand(env.n_bins)
        field[0] = np.inf
        field[1] = -np.inf

        fig, ax = plt.subplots()
        # Should handle inf gracefully
        env.plot_field(field, ax=ax, colorbar=False)

        plt.close(fig)

    def test_masked_grid_layout(self):
        """Test plot_field with masked grid layout."""
        from neurospatial.layout import create_layout

        # Create masked grid using correct parameters
        mask = np.ones((20, 20), dtype=bool)
        mask[5:15, 5:15] = False  # Hollow square
        edges = (
            np.linspace(0, 100, 21),  # 20 bins + 1 edge
            np.linspace(0, 100, 21),
        )

        layout = create_layout(
            kind="masked_grid",
            active_mask=mask,
            grid_edges=edges,
        )
        env = Environment(layout=layout, layout_type_used="masked_grid")
        field = np.random.rand(env.n_bins)

        fig, ax = plt.subplots()
        env.plot_field(field, ax=ax, colorbar=False)

        plt.close(fig)

    def test_polygon_layout(self):
        """Test plot_field with polygon layout."""
        from shapely.geometry import Polygon

        from neurospatial.layout import create_layout

        # Create polygon layout using correct parameters
        polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])

        layout = create_layout(
            kind="shapely_polygon",
            polygon=polygon,
            bin_size=10.0,
        )
        env = Environment(layout=layout, layout_type_used="shapely_polygon")
        field = np.random.rand(env.n_bins)

        fig, ax = plt.subplots()
        env.plot_field(field, ax=ax, colorbar=False)

        plt.close(fig)
