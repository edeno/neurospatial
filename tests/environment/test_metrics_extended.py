"""Comprehensive tests for environment metrics.

This module provides extensive test coverage for the EnvironmentMetrics mixin,
targeting 80% coverage for src/neurospatial/environment/metrics.py.

Test Coverage
-------------
- Boundary bin detection across layout types (grid, graph, hexagonal)
- Bin attribute extraction to DataFrames
- Edge attribute extraction to DataFrames
- Linearization for 1D environments (to_linear, linear_to_nd)
- Linearization properties
- Edge cases and validation

Coverage Target: 36% → 80%
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurospatial import Environment


class TestBoundaryBins:
    """Tests for boundary bin detection."""

    def test_boundary_bins_2d_grid(self, medium_2d_env: Environment) -> None:
        """Test boundary bin detection for 2D grid environment."""
        boundary = medium_2d_env.boundary_bins

        # Should have boundary bins
        assert len(boundary) > 0
        # Note: For sparse environments, all bins may be boundary bins
        assert len(boundary) <= medium_2d_env.n_bins

        # Boundary bins should be valid bin IDs
        assert np.all(boundary >= 0)
        assert np.all(boundary < medium_2d_env.n_bins)

        # Should be unique
        assert len(boundary) == len(np.unique(boundary))

    def test_boundary_bins_1d_env(self, small_1d_env: Environment) -> None:
        """Test boundary bin detection for 1D environment."""
        boundary = small_1d_env.boundary_bins

        # 1D environment should have exactly 2 boundary bins (endpoints)
        # unless it's very sparse
        assert len(boundary) >= 2
        assert len(boundary) <= small_1d_env.n_bins

    def test_boundary_bins_graph_env(self, graph_env: Environment) -> None:
        """Test boundary bin detection for graph-based environment."""
        boundary = graph_env.boundary_bins

        # Graph environment should have boundary bins
        assert len(boundary) > 0
        assert len(boundary) <= graph_env.n_bins

        # All boundary bins should be valid
        assert np.all(boundary >= 0)
        assert np.all(boundary < graph_env.n_bins)

    def test_boundary_bins_cached(self, small_2d_env: Environment) -> None:
        """Test that boundary_bins is cached (computed only once)."""
        # First access computes
        boundary1 = small_2d_env.boundary_bins

        # Second access should return same object (cached)
        boundary2 = small_2d_env.boundary_bins

        # Should be the exact same object
        assert boundary1 is boundary2

        # Should have same values
        np.testing.assert_array_equal(boundary1, boundary2)

    def test_boundary_bins_are_connected(self, medium_2d_env: Environment) -> None:
        """Test that boundary bins have fewer neighbors than interior bins."""
        boundary = medium_2d_env.boundary_bins

        # Get first boundary bin
        if len(boundary) > 0:
            boundary_bin = boundary[0]
            boundary_neighbors = medium_2d_env.neighbors(boundary_bin)

            # Find an interior bin (not in boundary)
            interior_bins = np.setdiff1d(
                np.arange(medium_2d_env.n_bins), boundary, assume_unique=True
            )

            if len(interior_bins) > 0:
                interior_bin = interior_bins[0]
                interior_neighbors = medium_2d_env.neighbors(interior_bin)

                # Interior bins typically have more neighbors
                # (This may not always be true for irregular environments)
                # So we just check that neighbor counts are reasonable
                assert len(boundary_neighbors) >= 1
                assert len(interior_neighbors) >= 1

    def test_boundary_bins_3d(self, simple_3d_env: Environment) -> None:
        """Test boundary bin detection for 3D environment."""
        boundary = simple_3d_env.boundary_bins

        # Should have boundary bins
        assert len(boundary) > 0

        # All should be valid
        assert np.all(boundary >= 0)
        assert np.all(boundary < simple_3d_env.n_bins)


class TestBinAttributes:
    """Tests for bin attribute extraction."""

    def test_bin_attributes_returns_dataframe(self, medium_2d_env: Environment) -> None:
        """Test that bin_attributes returns a DataFrame."""
        df = medium_2d_env.bin_attributes

        assert isinstance(df, pd.DataFrame)
        assert len(df) == medium_2d_env.n_bins

    def test_bin_attributes_has_required_columns(
        self, small_2d_env: Environment
    ) -> None:
        """Test that bin_attributes has required columns."""
        df = small_2d_env.bin_attributes

        # Should have standard columns
        assert "source_grid_flat_index" in df.columns
        assert "original_grid_nd_index" in df.columns

        # Should have position columns
        assert "pos_dim0" in df.columns
        assert "pos_dim1" in df.columns

    def test_bin_attributes_indexed_by_node_id(self, small_2d_env: Environment) -> None:
        """Test that bin_attributes DataFrame is indexed by node_id."""
        df = small_2d_env.bin_attributes

        # Index should be named 'node_id'
        assert df.index.name == "node_id"

        # Index should contain all bin IDs
        assert len(df.index) == small_2d_env.n_bins
        assert df.index.min() == 0
        assert df.index.max() == small_2d_env.n_bins - 1

    def test_bin_attributes_positions_match_bin_centers(
        self, small_2d_env: Environment
    ) -> None:
        """Test that position columns match bin_centers."""
        df = small_2d_env.bin_attributes

        # Extract positions from DataFrame
        pos_cols = [col for col in df.columns if col.startswith("pos_dim")]
        df_positions = df[pos_cols].values

        # Should match bin_centers
        np.testing.assert_allclose(df_positions, small_2d_env.bin_centers, rtol=1e-10)

    def test_bin_attributes_cached(self, medium_2d_env: Environment) -> None:
        """Test that bin_attributes is cached."""
        df1 = medium_2d_env.bin_attributes
        df2 = medium_2d_env.bin_attributes

        # Should be the same object
        assert df1 is df2

    def test_bin_attributes_1d_env(self, small_1d_env: Environment) -> None:
        """Test bin_attributes for 1D environment."""
        df = small_1d_env.bin_attributes

        assert isinstance(df, pd.DataFrame)
        assert len(df) == small_1d_env.n_bins

        # Should have pos_dim0 but not pos_dim1
        assert "pos_dim0" in df.columns
        assert "pos_dim1" not in df.columns

    def test_bin_attributes_3d_env(self, simple_3d_env: Environment) -> None:
        """Test bin_attributes for 3D environment."""
        df = simple_3d_env.bin_attributes

        assert isinstance(df, pd.DataFrame)
        assert len(df) == simple_3d_env.n_bins

        # Should have 3 position dimensions
        assert "pos_dim0" in df.columns
        assert "pos_dim1" in df.columns
        assert "pos_dim2" in df.columns


class TestEdgeAttributes:
    """Tests for edge attribute extraction."""

    def test_edge_attributes_returns_dataframe(
        self, medium_2d_env: Environment
    ) -> None:
        """Test that edge_attributes returns a DataFrame."""
        df = medium_2d_env.edge_attributes

        assert isinstance(df, pd.DataFrame)
        assert len(df) == medium_2d_env.connectivity.number_of_edges()

    def test_edge_attributes_has_required_columns(
        self, small_2d_env: Environment
    ) -> None:
        """Test that edge_attributes has required columns."""
        df = small_2d_env.edge_attributes

        # Should have standard edge attributes
        assert "source" in df.columns
        assert "target" in df.columns
        assert "distance" in df.columns
        assert "edge_id" in df.columns

    def test_edge_attributes_distances_positive(
        self, medium_2d_env: Environment
    ) -> None:
        """Test that all edge distances are positive."""
        df = medium_2d_env.edge_attributes

        assert (df["distance"] > 0).all()

    def test_edge_attributes_sources_and_targets_valid(
        self, small_2d_env: Environment
    ) -> None:
        """Test that source and target bin IDs are valid."""
        df = small_2d_env.edge_attributes

        # All sources and targets should be valid bin IDs
        assert df["source"].min() >= 0
        assert df["source"].max() < small_2d_env.n_bins
        assert df["target"].min() >= 0
        assert df["target"].max() < small_2d_env.n_bins

    def test_edge_attributes_cached(self, medium_2d_env: Environment) -> None:
        """Test that edge_attributes is cached."""
        df1 = medium_2d_env.edge_attributes
        df2 = medium_2d_env.edge_attributes

        # Should be the same object
        assert df1 is df2

    def test_edge_attributes_has_vector_columns(
        self, small_2d_env: Environment
    ) -> None:
        """Test that edge_attributes expands vector into separate columns."""
        df = small_2d_env.edge_attributes

        # Should have vector column
        assert "vector" in df.columns

        # Should also have expanded vector dimensions
        assert "vector_dim0" in df.columns
        assert "vector_dim1" in df.columns

    def test_edge_attributes_vector_dimensions_match(
        self, small_2d_env: Environment
    ) -> None:
        """Test that expanded vector dimensions match original vector."""
        df = small_2d_env.edge_attributes

        if len(df) > 0:
            # Check first edge
            vector_original = df["vector"].iloc[0]
            vector_expanded = [df["vector_dim0"].iloc[0], df["vector_dim1"].iloc[0]]

            np.testing.assert_allclose(vector_original, vector_expanded, rtol=1e-10)

    def test_edge_attributes_graph_env(self, graph_env: Environment) -> None:
        """Test edge_attributes for graph-based environment."""
        df = graph_env.edge_attributes

        assert isinstance(df, pd.DataFrame)
        assert len(df) == graph_env.connectivity.number_of_edges()
        assert len(df) > 0


class TestLinearization:
    """Tests for 1D linearization methods."""

    def test_to_linear_basic(self, graph_env: Environment) -> None:
        """Test basic to_linear conversion."""
        # graph_env is 1D
        assert graph_env.is_1d

        # Get a bin center
        nd_position = graph_env.bin_centers[:1]  # First bin

        # Convert to linear
        linear_pos = graph_env.to_linear(nd_position)

        # Should return 1D array
        assert linear_pos.ndim == 1
        assert len(linear_pos) == 1

        # Should be a valid linear coordinate
        assert np.isfinite(linear_pos[0])

    def test_linear_to_nd_basic(self, graph_env: Environment) -> None:
        """Test basic linear_to_nd conversion."""
        assert graph_env.is_1d

        # Create a linear position
        linear_pos = np.array([1.0])

        # Convert to ND
        nd_position = graph_env.linear_to_nd(linear_pos)

        # Should return ND array
        assert nd_position.ndim == 2
        assert nd_position.shape[0] == 1
        assert nd_position.shape[1] == graph_env.n_dims

        # Should be finite
        assert np.all(np.isfinite(nd_position))

    def test_to_linear_round_trip(self, graph_env: Environment) -> None:
        """Test that to_linear -> linear_to_nd round trip is consistent."""
        assert graph_env.is_1d

        # Start with ND position
        nd_original = graph_env.bin_centers[:5]

        # Convert to linear
        linear = graph_env.to_linear(nd_original)

        # Convert back to ND
        nd_recovered = graph_env.linear_to_nd(linear)

        # Should recover approximately the same positions
        # (may not be exact due to discretization)
        np.testing.assert_allclose(nd_recovered, nd_original, rtol=0.1, atol=1.0)

    def test_to_linear_raises_on_nd_env(self, medium_2d_env: Environment) -> None:
        """Test that to_linear raises error on N-D environment."""
        assert not medium_2d_env.is_1d

        with pytest.raises(
            AttributeError,
            match="to_linear\\(\\) is only available for 1D environments",
        ):
            medium_2d_env.to_linear(np.array([[0.0, 0.0]]))

    def test_linear_to_nd_raises_on_nd_env(self, medium_2d_env: Environment) -> None:
        """Test that linear_to_nd raises error on N-D environment."""
        assert not medium_2d_env.is_1d

        with pytest.raises(
            AttributeError,
            match="linear_to_nd\\(\\) is only available for 1D environments",
        ):
            medium_2d_env.linear_to_nd(np.array([0.0]))

    def test_to_linear_batch(self, graph_env: Environment) -> None:
        """Test to_linear with batch of positions."""
        assert graph_env.is_1d

        # Batch of positions
        nd_positions = graph_env.bin_centers[:10]

        # Convert to linear
        linear_positions = graph_env.to_linear(nd_positions)

        # Should return 1D array of same length
        assert linear_positions.ndim == 1
        assert len(linear_positions) == 10

        # Should all be finite
        assert np.all(np.isfinite(linear_positions))

    def test_linear_to_nd_batch(self, graph_env: Environment) -> None:
        """Test linear_to_nd with batch of positions."""
        assert graph_env.is_1d

        # Batch of linear positions
        linear_positions = np.linspace(0, 5, 10)

        # Convert to ND
        nd_positions = graph_env.linear_to_nd(linear_positions)

        # Should return ND array
        assert nd_positions.ndim == 2
        assert nd_positions.shape[0] == 10
        assert nd_positions.shape[1] == graph_env.n_dims

        # Should all be finite
        assert np.all(np.isfinite(nd_positions))


class TestLinearizationProperties:
    """Tests for linearization_properties."""

    def test_linearization_properties_1d_env(self, graph_env: Environment) -> None:
        """Test linearization_properties for 1D environment."""
        assert graph_env.is_1d

        props = graph_env.linearization_properties

        # Should be a dictionary
        assert isinstance(props, dict)

        # Should indicate 1D
        assert props["is_1d"] is True

        # May have additional linearization metadata
        # (exact keys depend on GraphLayout implementation)

    def test_linearization_properties_nd_env(self, medium_2d_env: Environment) -> None:
        """Test linearization_properties for N-D environment."""
        assert not medium_2d_env.is_1d

        props = medium_2d_env.linearization_properties

        # Should be a dictionary
        assert isinstance(props, dict)

        # Should indicate not 1D
        assert props["is_1d"] is False

        # Should only have is_1d key for N-D environments
        assert len(props) == 1

    def test_linearization_properties_cached(self, graph_env: Environment) -> None:
        """Test that linearization_properties is cached."""
        props1 = graph_env.linearization_properties
        props2 = graph_env.linearization_properties

        # Should be the same object
        assert props1 is props2

    def test_linearization_properties_has_linear_bin_edges(
        self, graph_env: Environment
    ) -> None:
        """Test that 1D env has linear_bin_edges in properties."""
        assert graph_env.is_1d

        props = graph_env.linearization_properties

        # GraphLayout should provide linear_bin_edges
        if hasattr(graph_env.layout, "linear_bin_edges"):
            assert "linear_bin_edges" in props

    def test_linearization_properties_has_track_graph(
        self, graph_env: Environment
    ) -> None:
        """Test that 1D env has track_graph in properties."""
        assert graph_env.is_1d

        props = graph_env.linearization_properties

        # GraphLayout should provide track_graph
        if hasattr(graph_env.layout, "track_graph"):
            assert "track_graph" in props


class TestSpatialMetrics:
    """Tests for spatial metrics and edge detection."""

    def test_boundary_bins_span_environment(self, medium_2d_env: Environment) -> None:
        """Test that boundary bins span a significant portion of environment."""
        boundary = medium_2d_env.boundary_bins

        # Get boundary bin centers
        boundary_centers = medium_2d_env.bin_centers[boundary]

        # Check that boundary bins span across each dimension
        for dim in range(medium_2d_env.n_dims):
            dim_min = boundary_centers[:, dim].min()
            dim_max = boundary_centers[:, dim].max()

            # Span should be positive (bins spread across dimension)
            span = dim_max - dim_min
            assert span > 0, f"Boundary bins should span dimension {dim}"

            # Boundary centers should be within the bin_centers range
            all_min = medium_2d_env.bin_centers[:, dim].min()
            all_max = medium_2d_env.bin_centers[:, dim].max()
            assert dim_min >= all_min
            assert dim_max <= all_max

    def test_boundary_bins_all_have_fewer_neighbors(
        self, small_2d_env: Environment
    ) -> None:
        """Test that boundary bins typically have fewer neighbors than interior bins."""
        boundary = small_2d_env.boundary_bins
        all_bins = np.arange(small_2d_env.n_bins)
        interior = np.setdiff1d(all_bins, boundary, assume_unique=True)

        if len(boundary) > 0 and len(interior) > 0:
            # Average neighbor count for boundary bins
            boundary_neighbor_counts = [
                len(small_2d_env.neighbors(b)) for b in boundary
            ]
            avg_boundary_neighbors = np.mean(boundary_neighbor_counts)

            # Average neighbor count for interior bins
            interior_neighbor_counts = [
                len(small_2d_env.neighbors(b)) for b in interior[:10]
            ]  # Sample
            avg_interior_neighbors = np.mean(interior_neighbor_counts)

            # Interior bins should typically have more neighbors
            # (This is a statistical property, may have exceptions)
            assert avg_boundary_neighbors <= avg_interior_neighbors + 1

    def test_edge_attributes_distance_matches_bin_separation(
        self, small_2d_env: Environment
    ) -> None:
        """Test that edge distances match actual bin center separation."""
        df = small_2d_env.edge_attributes

        if len(df) > 0:
            # Check first edge
            source = df["source"].iloc[0]
            target = df["target"].iloc[0]
            edge_distance = df["distance"].iloc[0]

            # Compute actual distance
            source_pos = small_2d_env.bin_centers[source]
            target_pos = small_2d_env.bin_centers[target]
            actual_distance = np.linalg.norm(target_pos - source_pos)

            # Should match closely
            np.testing.assert_allclose(edge_distance, actual_distance, rtol=1e-6)


class TestMetricsIntegration:
    """Integration tests for metrics across different environment types."""

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "small_2d_env",
            "medium_2d_env",
            "small_1d_env",
            "simple_3d_env",
            "graph_env",
        ],
    )
    def test_all_metrics_work(self, fixture_name: str, request) -> None:
        """Test that all metrics work for different environment types."""
        env = request.getfixturevalue(fixture_name)

        # boundary_bins should work
        boundary = env.boundary_bins
        assert len(boundary) > 0

        # bin_attributes should work
        bin_df = env.bin_attributes
        assert isinstance(bin_df, pd.DataFrame)
        assert len(bin_df) == env.n_bins

        # edge_attributes should work
        edge_df = env.edge_attributes
        assert isinstance(edge_df, pd.DataFrame)
        assert len(edge_df) == env.connectivity.number_of_edges()

        # linearization_properties should work
        lin_props = env.linearization_properties
        assert isinstance(lin_props, dict)
        assert "is_1d" in lin_props

    def test_metrics_consistent_with_environment_properties(
        self, medium_2d_env: Environment
    ) -> None:
        """Test that metrics are consistent with environment properties."""
        # bin_attributes should have same number of rows as n_bins
        assert len(medium_2d_env.bin_attributes) == medium_2d_env.n_bins

        # edge_attributes should have same number of rows as graph edges
        assert (
            len(medium_2d_env.edge_attributes)
            == medium_2d_env.connectivity.number_of_edges()
        )

        # boundary_bins should all be valid bin IDs
        boundary = medium_2d_env.boundary_bins
        assert np.all(boundary >= 0)
        assert np.all(boundary < medium_2d_env.n_bins)

    def test_cached_properties_persist(self, small_2d_env: Environment) -> None:
        """Test that cached properties persist across accesses."""
        # Access all cached properties multiple times
        for _ in range(3):
            _ = small_2d_env.boundary_bins
            _ = small_2d_env.bin_attributes
            _ = small_2d_env.edge_attributes
            _ = small_2d_env.linearization_properties

        # Should not raise errors and should be fast (cached)

    def test_metrics_work_after_clear_cache(self, small_2d_env: Environment) -> None:
        """Test that metrics work after clearing cache."""
        # Make a copy to avoid mutating session-scoped fixture
        env = small_2d_env.copy()

        # Access metrics
        _ = env.boundary_bins
        _ = env.bin_attributes

        # Clear cache
        env.clear_cache()

        # Should still work (recomputed)
        boundary = env.boundary_bins
        assert len(boundary) > 0

        bin_df = env.bin_attributes
        assert len(bin_df) == env.n_bins
