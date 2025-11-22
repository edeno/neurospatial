"""Skeleton data structure for pose tracking.

This module provides the Skeleton dataclass for defining anatomical relationships
between body parts, enabling skeleton visualization in animations. Designed for
easy conversion from common pose tracking formats (ndx-pose, movement).

Public API
----------
Skeleton : dataclass
    Immutable skeleton definition with nodes (bodyparts) and edges (connections).

Common presets:
- MOUSE_SKELETON: Standard mouse tracking skeleton
- RAT_SKELETON: Standard rat tracking skeleton
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# =============================================================================
# Helper functions
# =============================================================================


def _canonicalize_edge(edge: tuple[str, str]) -> tuple[str, str]:
    """Convert edge to canonical form: (min(src, dst), max(src, dst)).

    This ensures that ("a", "b") and ("b", "a") are represented the same way,
    since skeleton edges are undirected for visualization purposes.

    Parameters
    ----------
    edge : tuple[str, str]
        Edge as (source_node, target_node) pair.

    Returns
    -------
    tuple[str, str]
        Edge in canonical form with nodes sorted lexicographically.
    """
    src, dst = edge
    return (src, dst) if src <= dst else (dst, src)


def _normalize_edges(edges: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    """Normalize edges: canonicalize direction and remove duplicates.

    Parameters
    ----------
    edges : tuple[tuple[str, str], ...]
        Original edges, possibly with reversed pairs or duplicates.

    Returns
    -------
    tuple[tuple[str, str], ...]
        Normalized edges in canonical form, deduplicated, preserving order
        of first occurrence.
    """
    seen: set[tuple[str, str]] = set()
    normalized: list[tuple[str, str]] = []

    for edge in edges:
        canonical = _canonicalize_edge(edge)
        if canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)

    return tuple(normalized)


def _build_adjacency(
    nodes: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> dict[str, list[str]]:
    """Build adjacency list representation from nodes and edges.

    Parameters
    ----------
    nodes : tuple[str, ...]
        All node names in the skeleton.
    edges : tuple[tuple[str, str], ...]
        Edges as (node1, node2) pairs.

    Returns
    -------
    dict[str, list[str]]
        Mapping from each node to a sorted list of its neighbors.
        All nodes are included, even isolated ones (empty lists).
    """
    # Initialize with all nodes (including isolated ones)
    adjacency: dict[str, list[str]] = {node: [] for node in nodes}

    # Add edges (undirected - add both directions)
    for src, dst in edges:
        if src == dst:
            # Self-loop: add only once
            adjacency[src].append(dst)
        else:
            adjacency[src].append(dst)
            adjacency[dst].append(src)

    # Sort neighbor lists for deterministic output
    for node in adjacency:
        adjacency[node].sort()

    return adjacency


# =============================================================================
# Skeleton dataclass
# =============================================================================


@dataclass(frozen=True, slots=True)
class Skeleton:
    """Immutable definition of anatomical structure for pose tracking.

    A skeleton defines the relationship between tracked body parts (nodes) and
    their connections (edges). This separates the anatomical structure from
    temporal pose data, enabling reuse across sessions and animals.

    Parameters
    ----------
    name : str
        Unique identifier for this skeleton definition.
    nodes : tuple[str, ...]
        Ordered tuple of body part names. Order may be significant for
        certain visualizations or analyses.
    edges : tuple[tuple[str, str], ...]
        Tuple of (source_node, target_node) pairs defining connections.
        All node names must exist in `nodes`. Edges are undirected for
        visualization purposes.

    Attributes
    ----------
    name : str
        Skeleton identifier.
    nodes : tuple[str, ...]
        Body part names.
    edges : tuple[tuple[str, str], ...]
        Connection definitions.
    node_colors : dict[str, str] | None
        Optional per-node colors (matplotlib color strings).
    edge_color : str
        Default edge color for skeleton lines.
    edge_width : float
        Default edge width in points.
    metadata : Mapping[str, Any]
        Additional metadata (source, species, etc.).

    See Also
    --------
    BodypartOverlay : Overlay that uses Skeleton for pose visualization

    Notes
    -----
    Skeleton is immutable (frozen dataclass) to ensure consistency once defined.
    Use factory methods to create from external formats:

    - `from_ndx_pose()`: Convert from NWB ndx-pose extension
    - `from_movement()`: Extract from movement xarray Dataset
    - `from_edge_list()`: Infer nodes from edge list

    Examples
    --------
    Create a simple skeleton:

    >>> skeleton = Skeleton(
    ...     name="simple",
    ...     nodes=("head", "body", "tail"),
    ...     edges=(("head", "body"), ("body", "tail")),
    ... )
    >>> skeleton.name
    'simple'
    >>> len(skeleton.nodes)
    3
    >>> skeleton.edges
    (('head', 'body'), ('body', 'tail'))

    With custom colors:

    >>> skeleton = Skeleton(
    ...     name="colored",
    ...     nodes=("nose", "neck", "tail"),
    ...     edges=(("nose", "neck"), ("neck", "tail")),
    ...     node_colors={"nose": "red", "neck": "blue", "tail": "green"},
    ...     edge_color="white",
    ... )
    >>> skeleton.node_colors["nose"]
    'red'
    """

    name: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    node_colors: dict[str, str] | None = None
    edge_color: str = "white"
    edge_width: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)
    _adjacency: dict[str, list[str]] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate skeleton structure after initialization.

        Also normalizes edges:
        - Canonicalizes edge direction: (src, dst) → (min(src, dst), max(src, dst))
        - Deduplicates edges (including reversed duplicates)
        - Preserves order based on first occurrence
        """
        # Deep copy mutable fields to prevent aliasing
        if self.node_colors is not None:
            object.__setattr__(self, "node_colors", dict(self.node_colors))
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))

        # Normalize edges: canonicalize direction and deduplicate
        # Edges are undirected, so ("a", "b") and ("b", "a") are the same
        normalized_edges = _normalize_edges(self.edges)
        object.__setattr__(self, "edges", normalized_edges)

        # Validate nodes
        if len(self.nodes) == 0:
            raise ValueError(
                "WHAT: Skeleton must have at least one node.\n"
                "WHY: A skeleton with no nodes cannot represent any body parts.\n"
                "HOW: Provide at least one node name in the 'nodes' parameter."
            )

        # Check for duplicate nodes
        if len(self.nodes) != len(set(self.nodes)):
            duplicates = [n for n in self.nodes if self.nodes.count(n) > 1]
            raise ValueError(
                f"WHAT: Duplicate node names found: {sorted(set(duplicates))}\n"
                "WHY: Each node name must be unique to identify body parts.\n"
                "HOW: Remove duplicate names or rename body parts."
            )

        # Validate edges reference existing nodes
        node_set = set(self.nodes)
        for edge in self.edges:
            if len(edge) != 2:
                raise ValueError(
                    f"WHAT: Invalid edge format: {edge}\n"
                    "WHY: Edges must be (source, target) pairs.\n"
                    "HOW: Provide edges as tuple of 2-tuples: ((a, b), (b, c), ...)"
                )
            src, dst = edge
            missing = []
            if src not in node_set:
                missing.append(src)
            if dst not in node_set:
                missing.append(dst)
            if missing:
                raise ValueError(
                    f"WHAT: Edge {edge} references unknown node(s): {missing}\n"
                    f"WHY: All edge endpoints must be defined in 'nodes'.\n"
                    f"HOW: Add missing nodes to 'nodes' parameter or fix edge names.\n"
                    f"     Available nodes: {sorted(node_set)}"
                )

        # Validate node_colors keys match nodes
        if self.node_colors is not None:
            unknown_colors = set(self.node_colors.keys()) - node_set
            if unknown_colors:
                raise ValueError(
                    f"WHAT: node_colors references unknown nodes: {sorted(unknown_colors)}\n"
                    f"WHY: Colors can only be assigned to existing nodes.\n"
                    f"HOW: Remove unknown keys or add corresponding nodes.\n"
                    f"     Available nodes: {sorted(node_set)}"
                )

        # Precompute adjacency for graph traversal
        adjacency = _build_adjacency(self.nodes, self.edges)
        object.__setattr__(self, "_adjacency", adjacency)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        """Number of nodes (body parts) in the skeleton."""
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        """Number of edges (connections) in the skeleton."""
        return len(self.edges)

    @property
    def adjacency(self) -> dict[str, list[str]]:
        """Adjacency list representation of the skeleton graph.

        Maps each node to a sorted list of its neighbors. Since skeleton
        edges are undirected, if A-B is an edge, both A's list contains B
        and B's list contains A.

        Returns
        -------
        dict[str, list[str]]
            Mapping from each node name to a sorted list of neighbor names.
            All nodes are included, even isolated ones (which have empty lists).

        Examples
        --------
        >>> skeleton = Skeleton(
        ...     name="chain",
        ...     nodes=("head", "body", "tail"),
        ...     edges=(("head", "body"), ("body", "tail")),
        ... )
        >>> skeleton.adjacency["head"]
        ['body']
        >>> sorted(skeleton.adjacency["body"])
        ['head', 'tail']
        >>> skeleton.adjacency["tail"]
        ['body']

        Notes
        -----
        Precomputed at initialization for O(1) access. Useful for:
        - Graph traversal algorithms
        - Topology-based styling
        - Finding connected components
        """
        assert self._adjacency is not None  # Guaranteed by __post_init__
        return self._adjacency

    # -------------------------------------------------------------------------
    # Factory methods for format conversion
    # -------------------------------------------------------------------------

    @classmethod
    def from_edge_list(
        cls,
        edges: list[tuple[str, str]],
        name: str = "skeleton",
        *,
        node_order: list[str] | None = None,
        **kwargs: Any,
    ) -> Skeleton:
        """Create skeleton by inferring nodes from edge list.

        This is useful when you have only edge definitions and want to
        automatically extract the node names.

        Parameters
        ----------
        edges : list[tuple[str, str]]
            List of (source, target) node name pairs.
        name : str, optional
            Name for the skeleton. Default is "skeleton".
        node_order : list[str] | None, optional
            Explicit ordering of nodes. If None, nodes are sorted alphabetically.
            Must contain all nodes referenced in edges.
        **kwargs : Any
            Additional arguments passed to Skeleton (node_colors, edge_color, etc.).

        Returns
        -------
        Skeleton
            New Skeleton instance with nodes inferred from edges.

        Examples
        --------
        >>> edges = [("head", "body"), ("body", "tail"), ("body", "left_paw")]
        >>> skeleton = Skeleton.from_edge_list(edges, name="mouse")
        >>> skeleton.nodes
        ('body', 'head', 'left_paw', 'tail')

        With explicit node order:

        >>> skeleton = Skeleton.from_edge_list(
        ...     edges,
        ...     name="mouse",
        ...     node_order=["head", "body", "left_paw", "tail"],
        ... )
        >>> skeleton.nodes
        ('head', 'body', 'left_paw', 'tail')
        """
        # Extract all unique nodes from edges
        inferred_nodes = set()
        for src, dst in edges:
            inferred_nodes.add(src)
            inferred_nodes.add(dst)

        if node_order is not None:
            # Validate node_order contains all inferred nodes
            order_set = set(node_order)
            missing = inferred_nodes - order_set
            if missing:
                raise ValueError(
                    f"WHAT: node_order is missing nodes from edges: {sorted(missing)}\n"
                    "WHY: All nodes in edges must appear in node_order.\n"
                    "HOW: Add missing nodes to node_order or remove edges referencing them."
                )
            nodes = tuple(node_order)
        else:
            # Default: alphabetical order
            nodes = tuple(sorted(inferred_nodes))

        return cls(name=name, nodes=nodes, edges=tuple(edges), **kwargs)

    @classmethod
    def from_ndx_pose(
        cls,
        ndx_skeleton: Any,
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> Skeleton:
        """Create Skeleton from ndx-pose Skeleton object.

        Converts from the NWB (Neurodata Without Borders) ndx-pose extension
        format, which stores pose estimation data with skeletons defined by
        node arrays and edge index arrays.

        Parameters
        ----------
        ndx_skeleton : ndx_pose.Skeleton
            ndx-pose Skeleton object with `name`, `nodes`, and `edges` attributes.
        name : str | None, optional
            Override name. If None, uses ndx_skeleton.name.
        **kwargs : Any
            Additional arguments passed to Skeleton (node_colors, edge_color, etc.).

        Returns
        -------
        Skeleton
            New Skeleton instance converted from ndx-pose format.

        Notes
        -----
        ndx-pose stores edges as integer index pairs into the nodes array.
        This method converts those indices to node name pairs.

        Examples
        --------
        >>> # Assuming ndx_pose is installed and skeleton is loaded from NWB
        >>> # skeleton = nwb.processing["behavior"]["PoseEstimation"].skeleton
        >>> # ns_skeleton = Skeleton.from_ndx_pose(skeleton)

        See Also
        --------
        from_movement : Convert from movement xarray format
        """
        # Get nodes as list of strings
        nodes_array = ndx_skeleton.nodes[:]  # Convert to numpy/list
        nodes = tuple(str(n) for n in nodes_array)

        # Get edges as index pairs and convert to name pairs
        edges_array = ndx_skeleton.edges[:]  # Shape: (n_edges, 2)
        edges = tuple((nodes[int(i)], nodes[int(j)]) for i, j in edges_array)

        skeleton_name = name if name is not None else ndx_skeleton.name

        return cls(name=skeleton_name, nodes=nodes, edges=edges, **kwargs)

    @classmethod
    def from_movement(
        cls,
        dataset: Any,
        *,
        name: str = "skeleton",
        edges_attr: str = "skeleton_edges",
        **kwargs: Any,
    ) -> Skeleton:
        """Create Skeleton from movement xarray Dataset.

        Extracts skeleton information from a movement-style xarray Dataset,
        which stores keypoint names in coordinates and skeleton edges in
        attributes.

        Parameters
        ----------
        dataset : xarray.Dataset
            movement Dataset with 'keypoints' coordinate and optional
            skeleton edges in attributes.
        name : str, optional
            Name for the skeleton. Default is "skeleton".
        edges_attr : str, optional
            Attribute name for skeleton edges in dataset.attrs.
            Default is "skeleton_edges".
        **kwargs : Any
            Additional arguments passed to Skeleton (node_colors, edge_color, etc.).

        Returns
        -------
        Skeleton
            New Skeleton instance extracted from movement Dataset.

        Notes
        -----
        The movement library uses xarray with dimensions:
        (time, space, keypoints, individuals)

        Skeleton edges may be stored in dataset.attrs under various names
        depending on the source format (DeepLabCut, SLEAP, etc.).

        Examples
        --------
        >>> # Assuming movement is installed and dataset is loaded
        >>> # import movement.io as mio
        >>> # ds = mio.load_poses.from_dlc_file("predictions.h5")
        >>> # skeleton = Skeleton.from_movement(ds)

        See Also
        --------
        from_ndx_pose : Convert from NWB ndx-pose format
        """
        # Get keypoint names from coordinates
        if "keypoints" not in dataset.coords:
            raise ValueError(
                "WHAT: Dataset missing 'keypoints' coordinate.\n"
                "WHY: Cannot extract skeleton nodes without keypoint names.\n"
                "HOW: Ensure dataset has 'keypoints' coordinate from movement.io loaders."
            )

        nodes = tuple(str(k) for k in dataset.coords["keypoints"].values)

        # Get edges from attributes (may not exist)
        edges_data = dataset.attrs.get(edges_attr, [])
        edges = tuple(tuple(e) for e in edges_data) if edges_data else ()

        return cls(name=name, nodes=nodes, edges=edges, **kwargs)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Export Skeleton to a JSON-serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the Skeleton with keys:
            'name', 'nodes', 'edges', 'node_colors', 'edge_color',
            'edge_width', 'metadata'.

        Examples
        --------
        >>> skeleton = Skeleton(
        ...     name="simple",
        ...     nodes=("a", "b"),
        ...     edges=(("a", "b"),),
        ... )
        >>> d = skeleton.to_dict()
        >>> d["name"]
        'simple'
        >>> d["nodes"]
        ['a', 'b']
        """
        return {
            "name": self.name,
            "nodes": list(self.nodes),
            "edges": [list(e) for e in self.edges],
            "node_colors": self.node_colors,
            "edge_color": self.edge_color,
            "edge_width": self.edge_width,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Skeleton:
        """Create Skeleton from dictionary representation.

        Parameters
        ----------
        data : Mapping[str, Any]
            Dictionary with keys: 'name', 'nodes', 'edges', and optionally
            'node_colors', 'edge_color', 'edge_width', 'metadata'.

        Returns
        -------
        Skeleton
            Reconstructed Skeleton instance.

        Examples
        --------
        >>> d = {"name": "test", "nodes": ["a", "b"], "edges": [["a", "b"]]}
        >>> skeleton = Skeleton.from_dict(d)
        >>> skeleton.name
        'test'
        """
        return cls(
            name=data["name"],
            nodes=tuple(data["nodes"]),
            edges=tuple(tuple(e) for e in data["edges"]),
            node_colors=data.get("node_colors"),
            edge_color=data.get("edge_color", "white"),
            edge_width=data.get("edge_width", 1.0),
            metadata=data.get("metadata", {}),
        )

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    def with_colors(
        self,
        node_colors: dict[str, str] | None = None,
        edge_color: str | None = None,
        edge_width: float | None = None,
    ) -> Skeleton:
        """Create a copy with updated visual styling.

        Since Skeleton is immutable, this returns a new instance with
        modified color/width settings.

        Parameters
        ----------
        node_colors : dict[str, str] | None, optional
            New per-node colors. If None, keeps existing.
        edge_color : str | None, optional
            New edge color. If None, keeps existing.
        edge_width : float | None, optional
            New edge width. If None, keeps existing.

        Returns
        -------
        Skeleton
            New Skeleton with updated styling.

        Examples
        --------
        >>> skeleton = Skeleton(name="s", nodes=("a", "b"), edges=(("a", "b"),))
        >>> styled = skeleton.with_colors(edge_color="red", edge_width=2.0)
        >>> styled.edge_color
        'red'
        >>> skeleton.edge_color  # Original unchanged
        'white'
        """
        return Skeleton(
            name=self.name,
            nodes=self.nodes,
            edges=self.edges,
            node_colors=node_colors if node_colors is not None else self.node_colors,
            edge_color=edge_color if edge_color is not None else self.edge_color,
            edge_width=edge_width if edge_width is not None else self.edge_width,
            metadata=self.metadata,
        )

    def __str__(self) -> str:
        """Return string representation."""
        return f"Skeleton({self.name!r}, {self.n_nodes} nodes, {self.n_edges} edges)"


# =============================================================================
# Common skeleton presets
# =============================================================================

#: Standard mouse tracking skeleton (DeepLabCut-style)
MOUSE_SKELETON = Skeleton(
    name="mouse",
    nodes=(
        "nose",
        "left_ear",
        "right_ear",
        "neck",
        "body_center",
        "tail_base",
        "tail_mid",
        "tail_tip",
    ),
    edges=(
        ("nose", "neck"),
        ("left_ear", "neck"),
        ("right_ear", "neck"),
        ("neck", "body_center"),
        ("body_center", "tail_base"),
        ("tail_base", "tail_mid"),
        ("tail_mid", "tail_tip"),
    ),
    metadata={"species": "mouse", "source": "neurospatial"},
)

#: Standard rat tracking skeleton
RAT_SKELETON = Skeleton(
    name="rat",
    nodes=(
        "nose",
        "left_ear",
        "right_ear",
        "neck",
        "body_center",
        "tail_base",
        "tail_mid",
        "tail_tip",
    ),
    edges=(
        ("nose", "neck"),
        ("left_ear", "neck"),
        ("right_ear", "neck"),
        ("neck", "body_center"),
        ("body_center", "tail_base"),
        ("tail_base", "tail_mid"),
        ("tail_mid", "tail_tip"),
    ),
    metadata={"species": "rat", "source": "neurospatial"},
)

#: Simple 3-point skeleton (nose-body-tail)
SIMPLE_SKELETON = Skeleton(
    name="simple",
    nodes=("nose", "body", "tail"),
    edges=(("nose", "body"), ("body", "tail")),
    metadata={"source": "neurospatial"},
)
