"""Resolve annotation edge order and gaps together in graph coordinate units."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import networkx as nx
import numpy as np
from numpy.typing import NDArray


def validate_edge_coverage(
    edges: Sequence[tuple[int, int]],
    expected_edges: Sequence[tuple[int, int]],
    *,
    name: str,
) -> None:
    """Require every undirected edge exactly once, retaining its orientation."""
    actual = [frozenset(edge) for edge in edges]
    expected = {frozenset(edge) for edge in expected_edges}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(
            f"{name} must contain every track graph edge exactly once "
            "(no missing, duplicate, or unknown edges). "
            "Recompute the edge order after changing the graph."
        )


def validate_edge_spacing(spacing: object, n_edges: int) -> None:
    """Validate scalar or per-transition gaps in output coordinate units."""
    values = np.asarray(spacing, dtype=np.float64)
    if values.ndim > 1 or (values.ndim == 1 and len(values) != max(0, n_edges - 1)):
        raise ValueError(
            f"edge_spacing must have {max(0, n_edges - 1)} values or be a scalar."
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("edge_spacing must contain finite, non-negative values.")


def resolve_track_layout(
    graph: nx.Graph,
    *,
    start_node: int | None = None,
    edge_order: Sequence[tuple[int, int]] | None = None,
    edge_spacing: float | Sequence[float] | NDArray[np.float64] | None = None,
) -> tuple[list[tuple[int, int]], NDArray[np.float64]]:
    """Resolve complete oriented edges and their gaps in graph coordinate units.

    Automatic traversal visits the start component first, then every remaining
    component. Automatic gaps are 15 output units wherever the preceding edge's
    endpoint differs from the following edge's start. Manual gaps already use
    output units and are never scaled by calibration.
    """
    if start_node is not None and start_node not in graph:
        raise ValueError("start_node must identify a node in the track graph.")
    if edge_order is None:
        order: list[tuple[int, int]] = []
        visited: set[int] = set()
        roots = ([start_node] if start_node is not None else []) + list(graph.nodes)
        for root in roots:
            if root in visited:
                continue
            order.extend(nx.edge_bfs(graph, source=root))
            visited.update(nx.node_connected_component(graph, root))
    else:
        order = list(edge_order)
    validate_edge_coverage(order, list(graph.edges), name="edge_order")
    if edge_spacing is None:
        spacing = np.asarray(
            [0.0 if left[1] == right[0] else 15.0 for left, right in pairwise(order)],
            dtype=np.float64,
        )
    else:
        validate_edge_spacing(edge_spacing, len(order))
        values = np.asarray(edge_spacing, dtype=np.float64)
        spacing = (
            np.full(max(0, len(order) - 1), float(values), dtype=np.float64)
            if values.ndim == 0
            else values.copy()
        )
    return order, spacing
