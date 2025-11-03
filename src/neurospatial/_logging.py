"""Structured logging infrastructure for neurospatial.

This module provides a centralized logging system for the neurospatial package.
By default, logging is disabled (NullHandler), but users can enable it by
configuring the root logger.

Examples
--------
Enable logging to console::

    import logging

    logging.basicConfig(level=logging.INFO)

    from neurospatial import Environment

    env = Environment.from_samples(data, bin_size=2.0)
    # INFO:neurospatial:Building layout: regular_grid
    # INFO:neurospatial:Environment created: 245 bins, 2 dims

Enable logging with more detail::

    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

Filter to only neurospatial logs::

    import logging

    neurospatial_logger = logging.getLogger("neurospatial")
    neurospatial_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    neurospatial_logger.addHandler(handler)
"""

import logging
from typing import Any

# Create package logger
logger = logging.getLogger("neurospatial")

# Add NullHandler to prevent "No handler found" warnings
# Users must explicitly enable logging
logger.addHandler(logging.NullHandler())


def log_layout_build(layout_type: str, params: dict[str, Any]) -> None:
    """Log layout engine build operation.

    Parameters
    ----------
    layout_type : str
        Type of layout being built (e.g., 'regular_grid', 'hexagonal').
    params : dict
        Parameters used to build the layout.
    """
    logger.info(
        f"Building layout: {layout_type}",
        extra={"layout_type": layout_type, "params": params},
    )


def log_graph_validation(n_nodes: int, n_edges: int, n_dims: int) -> None:
    """Log graph validation operation.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the connectivity graph.
    n_edges : int
        Number of edges in the connectivity graph.
    n_dims : int
        Number of spatial dimensions.
    """
    logger.debug(
        f"Validating connectivity graph: {n_nodes} nodes, {n_edges} edges, {n_dims}D",
        extra={"n_nodes": n_nodes, "n_edges": n_edges, "n_dims": n_dims},
    )


def log_environment_created(
    env_type: str, n_bins: int, n_dims: int, env_name: str | None = None
) -> None:
    """Log environment creation.

    Parameters
    ----------
    env_type : str
        Type of environment (layout type tag).
    n_bins : int
        Number of bins in the environment.
    n_dims : int
        Number of spatial dimensions.
    env_name : str or None
        Optional name of the environment.
    """
    name_str = f" '{env_name}'" if env_name else ""
    logger.info(
        f"Environment created{name_str}: {n_bins} bins, {n_dims}D",
        extra={
            "type": env_type,
            "n_bins": n_bins,
            "n_dims": n_dims,
            "env_name": env_name,
        },
    )


def log_composite_build(n_subenvs: int, total_bins: int, n_bridges: int) -> None:
    """Log composite environment construction.

    Parameters
    ----------
    n_subenvs : int
        Number of sub-environments merged.
    total_bins : int
        Total number of bins in composite.
    n_bridges : int
        Number of bridge edges created.
    """
    logger.info(
        f"CompositeEnvironment created: {n_subenvs} sub-envs, "
        f"{total_bins} total bins, {n_bridges} bridges",
        extra={
            "n_subenvs": n_subenvs,
            "total_bins": total_bins,
            "n_bridges": n_bridges,
        },
    )


def log_region_added(
    region_name: str, region_kind: str, env_name: str | None = None
) -> None:
    """Log region addition to environment.

    Parameters
    ----------
    region_name : str
        Name of the region added.
    region_kind : str
        Type of region ('point' or 'polygon').
    env_name : str or None
        Optional name of the environment.
    """
    env_str = f" to '{env_name}'" if env_name else ""
    logger.debug(
        f"Region '{region_name}' ({region_kind}) added{env_str}",
        extra={
            "region_name": region_name,
            "region_kind": region_kind,
            "env_name": env_name,
        },
    )


def log_spatial_query(query_type: str, n_points: int, n_results: int) -> None:
    """Log spatial query operation.

    Parameters
    ----------
    query_type : str
        Type of query ('bin_at', 'contains', 'neighbors', etc.).
    n_points : int
        Number of points queried.
    n_results : int
        Number of results returned.
    """
    logger.debug(
        f"Spatial query '{query_type}': {n_points} points -> {n_results} results",
        extra={"query_type": query_type, "n_points": n_points, "n_results": n_results},
    )


def log_performance_warning(
    operation: str, duration_ms: float, threshold_ms: float
) -> None:
    """Log performance warning when operation exceeds threshold.

    Parameters
    ----------
    operation : str
        Name of the operation.
    duration_ms : float
        Actual duration in milliseconds.
    threshold_ms : float
        Expected threshold in milliseconds.
    """
    logger.warning(
        f"Performance warning: {operation} took {duration_ms:.1f}ms "
        f"(expected <{threshold_ms:.1f}ms)",
        extra={
            "operation": operation,
            "duration_ms": duration_ms,
            "threshold_ms": threshold_ms,
        },
    )
