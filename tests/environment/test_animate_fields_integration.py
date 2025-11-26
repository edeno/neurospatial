"""Integration tests for Environment.animate_fields() method.

This module tests that the animate_fields() method is properly integrated into
the Environment class and correctly delegates to the animation.core module.
Tests cover different layout types to ensure animation works across all layouts.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from neurospatial import Environment

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestAnimateFieldsIntegration:
    """Test suite for Environment.animate_fields() method integration."""

    def test_method_exists(self):
        """Test that animate_fields method exists on Environment."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (100, 2))
        env = Environment.from_samples(positions, bin_size=10.0)

        assert hasattr(env, "animate_fields")
        assert callable(env.animate_fields)

    @patch("neurospatial.animation.core.animate_fields")
    def test_delegates_to_core_dispatcher(self, mock_animate):
        """Test that method delegates to animation.core.animate_fields()."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (100, 2))
        env = Environment.from_samples(positions, bin_size=10.0)

        # Create simple field data
        fields = [rng.random(env.n_bins) for _ in range(5)]

        # Call method
        env.animate_fields(fields, backend="html", save_path="test.html")

        # Verify delegation
        mock_animate.assert_called_once()
        call_args = mock_animate.call_args

        # Check that env was passed
        assert call_args.kwargs["env"] is env
        # Check that fields were passed
        assert len(call_args.kwargs["fields"]) == 5
        # Check that backend was passed
        assert call_args.kwargs["backend"] == "html"
        # Check that save_path was passed
        assert call_args.kwargs["save_path"] == "test.html"

    @patch("neurospatial.animation.core.animate_fields")
    def test_forwards_all_parameters(self, mock_animate):
        """Test that all parameters are forwarded correctly."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (100, 2))
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins) for _ in range(3)]

        # Call with many parameters
        env.animate_fields(
            fields,
            backend="video",
            save_path="test.mp4",
            fps=30,
            cmap="hot",
            vmin=0.0,
            vmax=1.0,
            frame_labels=["A", "B", "C"],
            dpi=150,
            codec="h265",
            n_workers=4,
        )

        # Verify all parameters forwarded
        call_kwargs = mock_animate.call_args.kwargs
        assert call_kwargs["backend"] == "video"
        assert call_kwargs["save_path"] == "test.mp4"
        assert call_kwargs["fps"] == 30
        assert call_kwargs["cmap"] == "hot"
        assert call_kwargs["vmin"] == 0.0
        assert call_kwargs["vmax"] == 1.0
        assert call_kwargs["frame_labels"] == ["A", "B", "C"]
        assert call_kwargs["dpi"] == 150
        assert call_kwargs["codec"] == "h265"
        assert call_kwargs["n_workers"] == 4

    @patch("neurospatial.animation.core.animate_fields")
    def test_returns_dispatcher_result(self, mock_animate):
        """Test that method returns the result from core dispatcher."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (100, 2))
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins)]

        # Mock return value
        mock_return = Mock()
        mock_animate.return_value = mock_return

        result = env.animate_fields(fields)

        assert result is mock_return

    @patch("neurospatial.animation.core.animate_fields")
    def test_works_with_grid_layout(self, mock_animate):
        """Test that method works with regular grid layouts."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (200, 2))
        env = Environment.from_samples(positions, bin_size=5.0)
        fields = [rng.random(env.n_bins) for _ in range(3)]

        env.animate_fields(fields, backend="html")

        # Should have been called successfully
        mock_animate.assert_called_once()
        assert (
            mock_animate.call_args.kwargs["env"].layout._layout_type_tag
            == "RegularGrid"
        )

    @patch("neurospatial.animation.core.animate_fields")
    def test_works_with_hexagonal_layout(self, mock_animate):
        """Test that method works with hexagonal layouts."""
        from neurospatial.layout import create_layout

        rng = np.random.default_rng(42)
        data = rng.uniform(0, 100, (100, 2))
        layout = create_layout(
            kind="hexagonal",
            hexagon_width=10.0,
            positions=data,
            infer_active_bins=True,
        )
        env = Environment(layout=layout, layout_type_used="hexagonal")
        fields = [rng.random(env.n_bins) for _ in range(3)]

        env.animate_fields(fields, backend="html")

        # Should have been called successfully
        mock_animate.assert_called_once()
        assert (
            mock_animate.call_args.kwargs["env"].layout._layout_type_tag == "Hexagonal"
        )

    @patch("neurospatial.animation.core.animate_fields")
    def test_works_with_1d_layout(self, mock_animate):
        """Test that method works with 1D graph layouts."""
        pytest.importorskip("track_linearization")

        # Create simple 1D track graph manually
        import networkx as nx

        rng = np.random.default_rng(42)

        # Create simple linear track graph
        track_graph = nx.Graph()
        n_nodes = 10
        for i in range(n_nodes):
            track_graph.add_node(i, pos=(float(i), 0.0))
        for i in range(n_nodes - 1):
            # Add edges with required 'distance' attribute
            track_graph.add_edge(i, i + 1, distance=1.0)

        # Use from_graph with correct parameters
        edge_order = [(i, i + 1) for i in range(n_nodes - 1)]
        env = Environment.from_graph(
            track_graph, edge_order=edge_order, edge_spacing=1.0, bin_size=1.0
        )

        # 1D environment should work
        assert env.is_1d

        fields = [rng.random(env.n_bins) for _ in range(3)]
        env.animate_fields(fields, backend="html")

        # Should have been called successfully
        mock_animate.assert_called_once()
        assert mock_animate.call_args.kwargs["env"].is_1d

    @patch("neurospatial.animation.core.animate_fields")
    def test_works_with_masked_grid(self, mock_animate):
        """Test that method works with masked grid layouts."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(
            positions,
            bin_size=5.0,
            infer_active_bins=True,
            bin_count_threshold=2,
        )
        fields = [rng.random(env.n_bins) for _ in range(3)]

        env.animate_fields(fields, backend="html")

        # Should have been called successfully
        mock_animate.assert_called_once()

    def test_requires_fitted_environment(self):
        """Test that method requires environment to be fitted.

        Note: The @check_fitted decorator is tested on the core dispatcher,
        which checks env._is_fitted. Testing this at the Environment method
        level is tricky because creating an Environment always calls
        _setup_from_layout(), which requires a fitted layout. Instead, we
        test that the decorator is present and that the core validation works.
        """
        # The @check_fitted decorator is applied to animate_fields()
        # Verify it exists in the method's qualname
        assert hasattr(Environment.animate_fields, "__wrapped__")

        # The actual fitted state check is done in the core dispatcher
        # (tested in test_core.py), so we just verify the method has the decorator

    @patch("neurospatial.animation.core.animate_fields")
    def test_accepts_ndarray_input(self, mock_animate):
        """Test that method accepts ndarray input (not just list)."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (100, 2))
        env = Environment.from_samples(positions, bin_size=10.0)

        # Pass ndarray instead of list
        fields = rng.random((5, env.n_bins))

        env.animate_fields(fields, backend="html")

        # Should have been called successfully
        mock_animate.assert_called_once()

        # Verify fields were passed (core handles conversion to list)
        passed_fields = mock_animate.call_args.kwargs["fields"]
        assert isinstance(passed_fields, np.ndarray)
        assert passed_fields.shape == (5, env.n_bins)

    @patch("neurospatial.animation.core.animate_fields")
    def test_default_backend_auto(self, mock_animate):
        """Test that default backend is 'auto'."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (100, 2))
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins)]

        # Call without specifying backend
        env.animate_fields(fields)

        # Should default to 'auto'
        assert mock_animate.call_args.kwargs["backend"] == "auto"

    @patch("neurospatial.animation.core.animate_fields")
    def test_overlay_trajectory_parameter(self, mock_animate):
        """Test that overlay_trajectory parameter is forwarded."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (100, 2))
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [rng.random(env.n_bins)]

        # Create trajectory
        trajectory = rng.uniform(0, 100, (50, 2))

        env.animate_fields(fields, overlay_trajectory=trajectory)

        # Verify trajectory forwarded
        assert np.array_equal(
            mock_animate.call_args.kwargs["overlay_trajectory"], trajectory
        )
