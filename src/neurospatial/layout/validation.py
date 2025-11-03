"""Graph and environment validation utilities for neurospatial.

This module provides validation functions to ensure connectivity graphs and
environments have the required structure and metadata attributes as documented
in CLAUDE.md.

All layout engines must produce graphs with mandatory node and edge attributes.
This validator enforces those requirements and provides clear error messages
when violations are detected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from neurospatial.environment import Environment

# Required attributes per CLAUDE.md
REQUIRED_NODE_ATTRS = {"pos", "source_grid_flat_index", "original_grid_nd_index"}
REQUIRED_EDGE_ATTRS = {"distance", "vector", "edge_id"}
OPTIONAL_EDGE_ATTRS = {"angle_2d"}


class GraphValidationError(ValueError):
    """Raised when connectivity graph has invalid structure or metadata.

    This error indicates a bug in the layout engine that produced the graph,
    not a user error. All layout engines must produce graphs that pass
    validation.

    See Also
    --------
    validate_connectivity_graph : Main validation function
    """

    pass


def validate_connectivity_graph(
    graph: nx.Graph,
    n_dims: int,
    check_node_attrs: bool = True,
    check_edge_attrs: bool = True,
) -> None:
    """Validate that connectivity graph has required structure and metadata.

    This function enforces the mandatory graph metadata requirements documented
    in CLAUDE.md. All layout engines must produce graphs that pass this
    validation.

    Parameters
    ----------
    graph : nx.Graph
        The connectivity graph to validate
    n_dims : int
        Expected number of spatial dimensions
    check_node_attrs : bool, default=True
        Verify all required node attributes are present with correct types
    check_edge_attrs : bool, default=True
        Verify all required edge attributes are present with correct types

    Raises
    ------
    GraphValidationError
        If graph structure or metadata is invalid. Error message includes
        details about what is missing and which node/edge failed validation.

    Notes
    -----
    Required node attributes (all nodes must have these):
        - 'pos' : tuple/list/array of length n_dims
        - 'source_grid_flat_index' : int
        - 'original_grid_nd_index' : tuple of ints

    Required edge attributes (all edges must have these):
        - 'distance' : float >= 0
        - 'vector' : tuple/list/array of length n_dims
        - 'edge_id' : int

    Optional edge attributes:
        - 'angle_2d' : float (for 2D layouts only)

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G.add_node(
    ...     0, pos=(10.0, 20.0), source_grid_flat_index=0, original_grid_nd_index=(0, 0)
    ... )
    >>> G.add_node(
    ...     1, pos=(12.0, 20.0), source_grid_flat_index=1, original_grid_nd_index=(1, 0)
    ... )
    >>> G.add_edge(0, 1, distance=2.0, vector=(2.0, 0.0), edge_id=0)
    >>> validate_connectivity_graph(G, n_dims=2)  # Passes validation

    >>> # Missing required attributes will raise error
    >>> G2 = nx.Graph()
    >>> G2.add_node(0, pos=(10.0, 20.0))  # Missing other attrs
    >>> validate_connectivity_graph(G2, n_dims=2)  # doctest: +SKIP
    GraphValidationError: Node 0 missing required attributes: {...}

    See Also
    --------
    neurospatial.environment.Environment._setup_from_layout : Calls this validator
    """
    # Validate graph type
    if not isinstance(graph, nx.Graph):
        raise GraphValidationError(
            f"Expected networkx.Graph, got {type(graph).__name__}. "
            f"Layout engines must produce NetworkX Graph objects."
        )

    # Allow empty graphs (edge case: all bins masked out)
    # If empty, skip validation
    if len(graph.nodes) == 0:
        return

    # Validate dimensionality is positive
    if n_dims <= 0:
        raise GraphValidationError(
            f"n_dims must be positive, got {n_dims}. "
            f"This indicates a bug in the layout engine."
        )

    # Validate node attributes
    if check_node_attrs:
        for node_id in graph.nodes:
            node_data = graph.nodes[node_id]
            missing = REQUIRED_NODE_ATTRS - set(node_data.keys())

            if missing:
                raise GraphValidationError(
                    f"Node {node_id} missing required attributes: {missing}.\n\n"
                    f"All nodes must have: {REQUIRED_NODE_ATTRS}.\n"
                    f"Node {node_id} has: {set(node_data.keys())}\n\n"
                    f"This is a layout engine bug. Please report this issue.\n"
                    f"See CLAUDE.md section 'Graph Metadata Requirements' for details."
                )

            # Validate pos attribute
            pos = node_data["pos"]
            if not isinstance(pos, (tuple, list, np.ndarray)):
                raise GraphValidationError(
                    f"Node {node_id} 'pos' must be tuple/list/array, "
                    f"got {type(pos).__name__}. "
                    f"This is a layout engine bug."
                )

            # Validate pos dimensionality
            try:
                pos_len = len(pos)
            except TypeError as e:
                raise GraphValidationError(
                    f"Node {node_id} 'pos' is not a sequence: {pos}. "
                    f"Must be tuple/list/array of coordinates."
                ) from e

            if pos_len != n_dims:
                raise GraphValidationError(
                    f"Node {node_id} 'pos' has {pos_len} dimensions, "
                    f"expected {n_dims}.\n"
                    f"pos = {pos}\n"
                    f"This is a layout engine bug."
                )

            # Validate pos contains numbers
            try:
                _ = [float(x) for x in pos]
            except (TypeError, ValueError) as e:
                raise GraphValidationError(
                    f"Node {node_id} 'pos' contains non-numeric values: {pos}. "
                    f"Error: {e}"
                ) from e

            # Validate source_grid_flat_index is integer
            flat_idx = node_data["source_grid_flat_index"]
            if not isinstance(flat_idx, (int, np.integer)):
                raise GraphValidationError(
                    f"Node {node_id} 'source_grid_flat_index' must be int, "
                    f"got {type(flat_idx).__name__}."
                )

            # Validate original_grid_nd_index
            nd_idx = node_data["original_grid_nd_index"]
            if not isinstance(nd_idx, (tuple, list, np.ndarray)):
                raise GraphValidationError(
                    f"Node {node_id} 'original_grid_nd_index' must be tuple/list/array, "
                    f"got {type(nd_idx).__name__}."
                )

    # Validate edge attributes
    if check_edge_attrs and len(graph.edges) > 0:
        for u, v in graph.edges:
            edge_data = graph.edges[u, v]
            missing = REQUIRED_EDGE_ATTRS - set(edge_data.keys())

            if missing:
                raise GraphValidationError(
                    f"Edge ({u}, {v}) missing required attributes: {missing}.\n\n"
                    f"All edges must have: {REQUIRED_EDGE_ATTRS}.\n"
                    f"Edge ({u}, {v}) has: {set(edge_data.keys())}\n\n"
                    f"This is a layout engine bug. Please report this issue.\n"
                    f"See CLAUDE.md section 'Graph Metadata Requirements' for details."
                )

            # Validate distance
            distance = edge_data["distance"]
            try:
                distance_float = float(distance)
            except (TypeError, ValueError) as e:
                raise GraphValidationError(
                    f"Edge ({u}, {v}) 'distance' must be numeric, "
                    f"got {type(distance).__name__}: {distance}. "
                    f"Error: {e}"
                ) from e

            if distance_float < 0:
                raise GraphValidationError(
                    f"Edge ({u}, {v}) 'distance' must be non-negative, "
                    f"got {distance_float}. "
                    f"This is a layout engine bug."
                )

            if not np.isfinite(distance_float):
                raise GraphValidationError(
                    f"Edge ({u}, {v}) 'distance' must be finite, "
                    f"got {distance_float}. "
                    f"This is a layout engine bug."
                )

            # Validate vector
            vector = edge_data["vector"]
            if not isinstance(vector, (tuple, list, np.ndarray)):
                raise GraphValidationError(
                    f"Edge ({u}, {v}) 'vector' must be tuple/list/array, "
                    f"got {type(vector).__name__}."
                )

            # Validate vector dimensionality
            try:
                vec_len = len(vector)
            except TypeError as e:
                raise GraphValidationError(
                    f"Edge ({u}, {v}) 'vector' is not a sequence: {vector}. "
                    f"Must be tuple/list/array of displacement."
                ) from e

            if vec_len != n_dims:
                raise GraphValidationError(
                    f"Edge ({u}, {v}) 'vector' has {vec_len} dimensions, "
                    f"expected {n_dims}.\n"
                    f"vector = {vector}\n"
                    f"This is a layout engine bug."
                )

            # Validate vector contains numbers
            try:
                _ = [float(x) for x in vector]
            except (TypeError, ValueError) as e:
                raise GraphValidationError(
                    f"Edge ({u}, {v}) 'vector' contains non-numeric values: {vector}. "
                    f"Error: {e}"
                ) from e

            # Validate edge_id is integer
            edge_id = edge_data["edge_id"]
            if not isinstance(edge_id, (int, np.integer)):
                raise GraphValidationError(
                    f"Edge ({u}, {v}) 'edge_id' must be int, "
                    f"got {type(edge_id).__name__}."
                )


def validate_environment(env: Environment, *, strict: bool = True) -> None:
    """Validate that an Environment satisfies all structural invariants.

    This function provides a single entry point for validating Environment
    objects. It checks:
    - Required node/edge attributes on connectivity graph
    - Bin geometry consistency (bin_centers matches graph nodes)
    - Connectivity structure (no duplicate edges, consistent node IDs)
    - Unit presence (if strict=True, warns about missing units/frame)

    Downstream packages can use this to verify invariants before processing.

    Parameters
    ----------
    env : Environment
        Environment instance to validate.
    strict : bool, default=True
        If True, performs additional checks like warning about missing units.
        If False, only validates critical structural requirements.

    Raises
    ------
    GraphValidationError
        If connectivity graph is invalid (missing attributes, wrong dimensions).
    ValueError
        If bin_centers and connectivity graph are inconsistent.
    RuntimeError
        If environment is not fitted (was not created with factory method).

    Examples
    --------
    >>> from neurospatial import Environment
    >>> from neurospatial.layout.validation import validate_environment
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>> validate_environment(env)  # Passes if environment is valid

    >>> # Catch validation errors
    >>> try:
    ...     validate_environment(potentially_invalid_env)
    ... except (GraphValidationError, ValueError) as e:
    ...     print(f"Environment is invalid: {e}")

    See Also
    --------
    validate_connectivity_graph : Lower-level graph validation
    Environment._setup_from_layout : Calls validation during creation

    Notes
    -----
    This validator is fail-fast with standardized error messages. It is
    designed to catch layout engine bugs and data corruption early.

    Downstream packages that rely on neurospatial environments should call
    this at their entry points to ensure invariants hold, avoiding the need
    for duplicate defensive checks.

    """
    import warnings

    # Check fitted status
    if not getattr(env, "_is_fitted", False):
        raise RuntimeError(
            f"Environment '{env.name}' is not fitted. "
            f"Environments must be created with factory methods "
            f"(e.g., Environment.from_samples()) before validation."
        )

    # Validate connectivity graph
    n_dims = env.n_dims
    validate_connectivity_graph(
        env.connectivity, n_dims=n_dims, check_node_attrs=True, check_edge_attrs=True
    )

    # Validate bin_centers consistency with graph
    n_bins_from_centers = env.bin_centers.shape[0]
    n_nodes_from_graph = len(env.connectivity.nodes)

    if n_bins_from_centers != n_nodes_from_graph:
        raise ValueError(
            f"bin_centers and connectivity graph are inconsistent.\n"
            f"  bin_centers has {n_bins_from_centers} rows\n"
            f"  connectivity graph has {n_nodes_from_graph} nodes\n"
            f"These must match. This indicates a layout engine bug."
        )

    # Validate node IDs are sequential from 0 to n_bins-1
    node_ids = sorted(env.connectivity.nodes)
    expected_ids = list(range(n_bins_from_centers))
    if node_ids != expected_ids:
        raise ValueError(
            f"connectivity graph node IDs are not sequential.\n"
            f"  Expected: [0, 1, ..., {n_bins_from_centers - 1}]\n"
            f"  Got: {node_ids[:10]}{'...' if len(node_ids) > 10 else ''}\n"
            f"This indicates a layout engine bug."
        )

    # Validate bin_centers has correct shape
    if env.bin_centers.ndim != 2:
        raise ValueError(
            f"bin_centers must be 2D array (n_bins, n_dims), "
            f"got shape {env.bin_centers.shape}"
        )

    if env.bin_centers.shape[1] != n_dims:
        raise ValueError(
            f"bin_centers has {env.bin_centers.shape[1]} columns, "
            f"expected {n_dims} dimensions"
        )

    # Check for duplicate edges (undirected graph should have only one edge per pair)
    edges_set = set()
    for u, v in env.connectivity.edges:
        edge_tuple = tuple(sorted([u, v]))
        if edge_tuple in edges_set:
            raise ValueError(
                f"Duplicate edge found: {edge_tuple}. "
                f"This indicates a layout engine bug."
            )
        edges_set.add(edge_tuple)

    # Strict mode checks
    if strict:
        # Warn about missing units
        if not hasattr(env, "units") or env.units is None:
            warnings.warn(
                f"Environment '{env.name}' has no units specified. "
                f"Consider setting env.units (e.g., 'cm', 'px', 'm') "
                f"to prevent unit confusion in downstream analysis.",
                stacklevel=2,
            )

        # Warn about missing coordinate frame
        if not hasattr(env, "frame") or env.frame is None:
            warnings.warn(
                f"Environment '{env.name}' has no coordinate frame specified. "
                f"Consider setting env.frame (e.g., 'world', 'camera_1') "
                f"to prevent confusion when aligning multiple sessions.",
                stacklevel=2,
            )
