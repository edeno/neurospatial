"""Track graphs for standard mazes, built from ordered node positions."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np

from neurospatial._exceptions import _format_error


def add_edge_with_distance(graph: nx.Graph, u: Any, v: Any) -> None:
    """Add an edge to `graph` with `distance` computed from node positions.

    ``GraphLayout.build`` numbers ``edge_id`` by ``graph.edges()`` order; any
    ``edge_id`` on the input graph is ignored.
    """
    p1 = np.asarray(graph.nodes[u]["pos"], dtype=float)
    p2 = np.asarray(graph.nodes[v]["pos"], dtype=float)
    graph.add_edge(u, v, distance=float(np.linalg.norm(p2 - p1)))


def maze_track_graph(
    kind: str,
    node_positions: dict[Any, Sequence[float]],
) -> tuple[nx.Graph, list[tuple[Any, Any]]]:
    """Assemble a standard W / plus / T maze graph from ordered node positions.

    Topology is derived from the DOCUMENTED ORDER of ``node_positions`` (a
    coordinate-system-independent contract), never from x/y heuristics. The
    required ordering per `kind` is:

    - ``"plus"`` (5 nodes): ``[center, arm1, arm2, arm3, arm4]`` -> edges
      center--each arm.
    - ``"t"`` (4 nodes): ``[stem_end, junction, arm_left, arm_right]`` -> edges
      stem_end--junction, junction--arm_left, junction--arm_right.
    - ``"w"`` (6 nodes): ``[base_left, base_mid, base_right, arm_left, arm_mid,
      arm_right]`` -> connector edges base_left--base_mid--base_right plus arm
      edges base_left--arm_left, base_mid--arm_mid, base_right--arm_right.

    Because the topology comes from order alone, the SAME graph is produced for
    any consistent coordinate system (e.g. y increasing downward, or arms in
    -y); coordinates are stored only as ``pos`` attributes / edge distances.

    Parameters
    ----------
    kind : {"w", "plus", "t"}
        Normalized (lower-case) maze kind.
    node_positions : dict
        Ordered mapping of node label -> (x, y) coordinate. The insertion order
        defines the topology (see above). Python dicts preserve insertion order.

    Returns
    -------
    graph : networkx.Graph
        Track graph with `pos` node attributes and `distance` edge weights.
    edge_order : list of (node, node)
        Ordered edge list for linearization.

    Raises
    ------
    ValueError
        If `node_positions` does not contain the expected number of nodes for
        `kind`, or if `kind` is unknown.
    """
    labels = list(node_positions.keys())
    graph = nx.Graph()
    for label, pos in node_positions.items():
        graph.add_node(label, pos=tuple(float(c) for c in pos))

    edge_order: list[tuple[Any, Any]] = []

    if kind == "plus":
        # Order contract: [center, arm1, arm2, arm3, arm4].
        # Star topology: center connects to each of the four arm tips.
        if len(labels) != 5:
            raise ValueError(
                "A plus maze needs exactly 5 nodes in `node_positions`, ordered "
                "[center, arm1, arm2, arm3, arm4]: one center followed by four "
                f"arm tips. Got {len(labels)}."
            )
        center = labels[0]
        for arm in labels[1:]:
            add_edge_with_distance(graph, center, arm)
            edge_order.append((center, arm))

    elif kind == "t":
        # Order contract: [stem_end, junction, arm_left, arm_right].
        # Stem (stem_end->junction) + crossbar (junction->each arm tip).
        if len(labels) != 4:
            raise ValueError(
                "A T maze needs exactly 4 nodes in `node_positions`, ordered "
                "[stem_end, junction, arm_left, arm_right]. "
                f"Got {len(labels)}."
            )
        stem_end, junction, arm_left, arm_right = labels
        add_edge_with_distance(graph, stem_end, junction)
        edge_order.append((stem_end, junction))
        for arm in (arm_left, arm_right):
            add_edge_with_distance(graph, junction, arm)
            edge_order.append((junction, arm))

    elif kind == "w":
        # Order contract:
        #   [base_left, base_mid, base_right, arm_left, arm_mid, arm_right].
        # Horizontal connector along the first three base nodes, then one
        # vertical arm edge joining each base node to the arm tip at the
        # matching position in the documented order.
        if len(labels) != 6:
            raise ValueError(
                "A W maze needs exactly 6 nodes in `node_positions`, ordered "
                "[base_left, base_mid, base_right, arm_left, arm_mid, "
                "arm_right]: three base/junction nodes followed by three arm "
                f"tips (5 base+arm edges). Got {len(labels)}."
            )
        base = labels[:3]
        arms = labels[3:]
        # Horizontal connector along the base (base_left -> base_mid -> base_right).
        for left, right in itertools.pairwise(base):
            add_edge_with_distance(graph, left, right)
            edge_order.append((left, right))
        # Vertical arms: each base node up to the arm at the matching position.
        for base_node, arm_node in zip(base, arms, strict=True):
            add_edge_with_distance(graph, base_node, arm_node)
            edge_order.append((base_node, arm_node))

    else:
        raise ValueError(
            _format_error(
                f"Unknown maze kind {kind!r}",
                fix="pass kind='w', kind='plus', or kind='t' with the documented node_positions order",
                why="Why: maze topology depends on a supported maze kind.",
            )
        )

    return graph, edge_order
