"""Tests for graph validation utilities."""

import networkx as nx
import numpy as np
import pytest

from neurospatial.layout.validation import (
    GraphValidationError,
    validate_connectivity_graph,
)


class TestGraphValidation:
    """Test suite for validate_connectivity_graph function."""

    def test_validate_valid_2d_graph(self):
        """Test validation passes for valid 2D graph."""
        G = nx.Graph()
        G.add_node(
            0, pos=(10.0, 20.0), source_grid_flat_index=0, original_grid_nd_index=(0, 0)
        )
        G.add_node(
            1, pos=(12.0, 20.0), source_grid_flat_index=1, original_grid_nd_index=(1, 0)
        )
        G.add_edge(0, 1, distance=2.0, vector=(2.0, 0.0), edge_id=0)

        # Should not raise
        validate_connectivity_graph(G, n_dims=2)

    def test_validate_valid_3d_graph(self):
        """Test validation passes for valid 3D graph."""
        G = nx.Graph()
        G.add_node(
            0,
            pos=(10.0, 20.0, 5.0),
            source_grid_flat_index=0,
            original_grid_nd_index=(0, 0, 0),
        )
        G.add_node(
            1,
            pos=(12.0, 20.0, 5.0),
            source_grid_flat_index=1,
            original_grid_nd_index=(1, 0, 0),
        )
        G.add_edge(0, 1, distance=2.0, vector=(2.0, 0.0, 0.0), edge_id=0)

        # Should not raise
        validate_connectivity_graph(G, n_dims=3)

    def test_validate_invalid_graph_type(self):
        """Test error for non-Graph object."""
        with pytest.raises(GraphValidationError, match=r"Expected networkx\.Graph"):
            validate_connectivity_graph("not a graph", n_dims=2)

        with pytest.raises(GraphValidationError, match=r"Expected networkx\.Graph"):
            validate_connectivity_graph([], n_dims=2)

    def test_validate_empty_graph(self):
        """Test validation allows empty graph (edge case: all bins masked out)."""
        G = nx.Graph()

        # Should not raise - empty graphs are allowed
        validate_connectivity_graph(G, n_dims=2)

    def test_validate_invalid_n_dims(self):
        """Test error for invalid n_dims."""
        G = nx.Graph()
        G.add_node(
            0, pos=(10.0, 20.0), source_grid_flat_index=0, original_grid_nd_index=(0, 0)
        )

        with pytest.raises(GraphValidationError, match="n_dims must be positive"):
            validate_connectivity_graph(G, n_dims=0)

        with pytest.raises(GraphValidationError, match="n_dims must be positive"):
            validate_connectivity_graph(G, n_dims=-1)

    def test_validate_missing_node_attribute(self):
        """Test error for missing required node attribute."""
        G = nx.Graph()
        # Missing source_grid_flat_index and original_grid_nd_index
        G.add_node(0, pos=(10.0, 20.0))

        with pytest.raises(GraphValidationError, match="missing required attributes"):
            validate_connectivity_graph(G, n_dims=2)

    def test_validate_missing_all_node_attributes(self):
        """Test error when node has no attributes."""
        G = nx.Graph()
        G.add_node(0)  # No attributes at all

        with pytest.raises(
            GraphValidationError, match="Node 0 missing required attributes"
        ):
            validate_connectivity_graph(G, n_dims=2)

    def test_validate_invalid_pos_type(self):
        """Test error for non-sequence pos attribute."""
        G = nx.Graph()
        G.add_node(
            0, pos="invalid", source_grid_flat_index=0, original_grid_nd_index=(0, 0)
        )

        with pytest.raises(
            GraphValidationError, match="'pos' must be tuple/list/array"
        ):
            validate_connectivity_graph(G, n_dims=2)

    def test_validate_wrong_pos_dimensionality(self):
        """Test error for pos with wrong number of dimensions."""
        G = nx.Graph()
        # 1D pos for 2D graph
        G.add_node(
            0, pos=(10.0,), source_grid_flat_index=0, original_grid_nd_index=(0,)
        )

        with pytest.raises(GraphValidationError, match="has 1 dimensions, expected 2"):
            validate_connectivity_graph(G, n_dims=2)

    def test_validate_pos_with_non_numeric_values(self):
        """Test error for pos containing non-numeric values."""
        G = nx.Graph()
        G.add_node(
            0,
            pos=("a", "b"),
            source_grid_flat_index=0,
            original_grid_nd_index=(0, 0),
        )

        with pytest.raises(GraphValidationError, match="non-numeric values"):
            validate_connectivity_graph(G, n_dims=2)

    def test_validate_invalid_flat_index_type(self):
        """Test error for non-integer source_grid_flat_index."""
        G = nx.Graph()
        G.add_node(
            0,
            pos=(10.0, 20.0),
            source_grid_flat_index="invalid",
            original_grid_nd_index=(0, 0),
        )

        with pytest.raises(
            GraphValidationError, match="'source_grid_flat_index' must be int"
        ):
            validate_connectivity_graph(G, n_dims=2)

    def test_validate_invalid_nd_index_type(self):
        """Test error for non-sequence original_grid_nd_index."""
        G = nx.Graph()
        G.add_node(
            0, pos=(10.0, 20.0), source_grid_flat_index=0, original_grid_nd_index=42
        )

        with pytest.raises(
            GraphValidationError, match="'original_grid_nd_index' must be tuple/list"
        ):
            validate_connectivity_graph(G, n_dims=2)

    def test_validate_missing_edge_attribute(self):
        """Test error for missing required edge attribute."""
        G = nx.Graph()
        G.add_node(
            0, pos=(10.0, 20.0), source_grid_flat_index=0, original_grid_nd_index=(0, 0)
        )
        G.add_node(
            1, pos=(12.0, 20.0), source_grid_flat_index=1, original_grid_nd_index=(1, 0)
        )
        # Missing vector and edge_id
        G.add_edge(0, 1, distance=2.0)

        with pytest.raises(GraphValidationError, match="missing required attributes"):
            validate_connectivity_graph(G, n_dims=2)

    def test_validate_invalid_distance_type(self):
        """Test error for non-numeric distance."""
        G = nx.Graph()
        G.add_node(
            0, pos=(10.0, 20.0), source_grid_flat_index=0, original_grid_nd_index=(0, 0)
        )
        G.add_node(
            1, pos=(12.0, 20.0), source_grid_flat_index=1, original_grid_nd_index=(1, 0)
        )
        G.add_edge(0, 1, distance="invalid", vector=(2.0, 0.0), edge_id=0)

        with pytest.raises(GraphValidationError, match="'distance' must be numeric"):
            validate_connectivity_graph(G, n_dims=2)

    def test_validate_negative_distance(self):
        """Test error for negative distance."""
        G = nx.Graph()
        G.add_node(
            0, pos=(10.0, 20.0), source_grid_flat_index=0, original_grid_nd_index=(0, 0)
        )
        G.add_node(
            1, pos=(12.0, 20.0), source_grid_flat_index=1, original_grid_nd_index=(1, 0)
        )
        G.add_edge(0, 1, distance=-2.0, vector=(2.0, 0.0), edge_id=0)

        with pytest.raises(GraphValidationError, match="must be non-negative"):
            validate_connectivity_graph(G, n_dims=2)

    def test_validate_infinite_distance(self):
        """Test error for infinite distance."""
        G = nx.Graph()
        G.add_node(
            0, pos=(10.0, 20.0), source_grid_flat_index=0, original_grid_nd_index=(0, 0)
        )
        G.add_node(
            1, pos=(12.0, 20.0), source_grid_flat_index=1, original_grid_nd_index=(1, 0)
        )
        G.add_edge(0, 1, distance=np.inf, vector=(2.0, 0.0), edge_id=0)

        with pytest.raises(GraphValidationError, match="must be finite"):
            validate_connectivity_graph(G, n_dims=2)

    def test_validate_invalid_vector_type(self):
        """Test error for non-sequence vector."""
        G = nx.Graph()
        G.add_node(
            0, pos=(10.0, 20.0), source_grid_flat_index=0, original_grid_nd_index=(0, 0)
        )
        G.add_node(
            1, pos=(12.0, 20.0), source_grid_flat_index=1, original_grid_nd_index=(1, 0)
        )
        G.add_edge(0, 1, distance=2.0, vector="invalid", edge_id=0)

        with pytest.raises(
            GraphValidationError, match="'vector' must be tuple/list/array"
        ):
            validate_connectivity_graph(G, n_dims=2)

    def test_validate_wrong_vector_dimensionality(self):
        """Test error for vector with wrong number of dimensions."""
        G = nx.Graph()
        G.add_node(
            0, pos=(10.0, 20.0), source_grid_flat_index=0, original_grid_nd_index=(0, 0)
        )
        G.add_node(
            1, pos=(12.0, 20.0), source_grid_flat_index=1, original_grid_nd_index=(1, 0)
        )
        # 1D vector for 2D graph
        G.add_edge(0, 1, distance=2.0, vector=(2.0,), edge_id=0)

        with pytest.raises(GraphValidationError, match="has 1 dimensions, expected 2"):
            validate_connectivity_graph(G, n_dims=2)

    def test_validate_vector_with_non_numeric_values(self):
        """Test error for vector containing non-numeric values."""
        G = nx.Graph()
        G.add_node(
            0, pos=(10.0, 20.0), source_grid_flat_index=0, original_grid_nd_index=(0, 0)
        )
        G.add_node(
            1, pos=(12.0, 20.0), source_grid_flat_index=1, original_grid_nd_index=(1, 0)
        )
        G.add_edge(0, 1, distance=2.0, vector=("a", "b"), edge_id=0)

        with pytest.raises(GraphValidationError, match="non-numeric values"):
            validate_connectivity_graph(G, n_dims=2)

    def test_validate_invalid_edge_id_type(self):
        """Test error for non-integer edge_id."""
        G = nx.Graph()
        G.add_node(
            0, pos=(10.0, 20.0), source_grid_flat_index=0, original_grid_nd_index=(0, 0)
        )
        G.add_node(
            1, pos=(12.0, 20.0), source_grid_flat_index=1, original_grid_nd_index=(1, 0)
        )
        G.add_edge(0, 1, distance=2.0, vector=(2.0, 0.0), edge_id="invalid")

        with pytest.raises(GraphValidationError, match="'edge_id' must be int"):
            validate_connectivity_graph(G, n_dims=2)

    def test_validate_skip_node_attrs(self):
        """Test validation can skip node attribute checking."""
        G = nx.Graph()
        G.add_node(0)  # No attributes

        # Should not raise when check_node_attrs=False
        validate_connectivity_graph(G, n_dims=2, check_node_attrs=False)

    def test_validate_skip_edge_attrs(self):
        """Test validation can skip edge attribute checking."""
        G = nx.Graph()
        G.add_node(
            0, pos=(10.0, 20.0), source_grid_flat_index=0, original_grid_nd_index=(0, 0)
        )
        G.add_node(
            1, pos=(12.0, 20.0), source_grid_flat_index=1, original_grid_nd_index=(1, 0)
        )
        G.add_edge(0, 1)  # No attributes

        # Should not raise when check_edge_attrs=False
        validate_connectivity_graph(G, n_dims=2, check_edge_attrs=False)

    def test_validate_graph_with_no_edges(self):
        """Test validation passes for graph with nodes but no edges."""
        G = nx.Graph()
        G.add_node(
            0, pos=(10.0, 20.0), source_grid_flat_index=0, original_grid_nd_index=(0, 0)
        )
        G.add_node(
            1, pos=(12.0, 20.0), source_grid_flat_index=1, original_grid_nd_index=(1, 0)
        )

        # Should not raise - edges are optional
        validate_connectivity_graph(G, n_dims=2)

    def test_validate_with_numpy_types(self):
        """Test validation works with numpy types."""
        G = nx.Graph()
        G.add_node(
            0,
            pos=np.array([10.0, 20.0]),
            source_grid_flat_index=np.int64(0),
            original_grid_nd_index=np.array([0, 0]),
        )
        G.add_node(
            1,
            pos=np.array([12.0, 20.0]),
            source_grid_flat_index=np.int64(1),
            original_grid_nd_index=np.array([1, 0]),
        )
        G.add_edge(
            0,
            1,
            distance=np.float64(2.0),
            vector=np.array([2.0, 0.0]),
            edge_id=np.int64(0),
        )

        # Should not raise - numpy types are acceptable
        validate_connectivity_graph(G, n_dims=2)

    def test_validate_with_optional_angle_2d(self):
        """Test validation allows optional angle_2d attribute."""
        G = nx.Graph()
        G.add_node(
            0, pos=(10.0, 20.0), source_grid_flat_index=0, original_grid_nd_index=(0, 0)
        )
        G.add_node(
            1, pos=(12.0, 20.0), source_grid_flat_index=1, original_grid_nd_index=(1, 0)
        )
        G.add_edge(
            0,
            1,
            distance=2.0,
            vector=(2.0, 0.0),
            edge_id=0,
            angle_2d=0.0,  # Optional
        )

        # Should not raise - angle_2d is optional
        validate_connectivity_graph(G, n_dims=2)

    def test_error_message_includes_node_id(self):
        """Test error messages include specific node/edge IDs for debugging."""
        G = nx.Graph()
        G.add_node(
            0, pos=(10.0, 20.0), source_grid_flat_index=0, original_grid_nd_index=(0, 0)
        )
        G.add_node(5, pos=(10.0, 20.0))  # Node 5 missing attrs

        with pytest.raises(GraphValidationError, match="Node 5"):
            validate_connectivity_graph(G, n_dims=2)

    def test_error_message_includes_edge_ids(self):
        """Test error messages include specific edge IDs for debugging."""
        G = nx.Graph()
        G.add_node(
            0, pos=(10.0, 20.0), source_grid_flat_index=0, original_grid_nd_index=(0, 0)
        )
        G.add_node(
            1, pos=(12.0, 20.0), source_grid_flat_index=1, original_grid_nd_index=(1, 0)
        )
        G.add_edge(0, 1, distance=2.0)  # Missing vector and edge_id

        with pytest.raises(GraphValidationError, match=r"Edge \(0, 1\)"):
            validate_connectivity_graph(G, n_dims=2)

    def test_error_message_suggests_claude_md(self):
        """Test error messages reference CLAUDE.md documentation."""
        G = nx.Graph()
        G.add_node(0, pos=(10.0, 20.0))  # Missing attrs

        with pytest.raises(GraphValidationError, match=r"CLAUDE\.md"):
            validate_connectivity_graph(G, n_dims=2)
