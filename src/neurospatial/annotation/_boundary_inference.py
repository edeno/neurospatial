"""Boundary inference algorithms for seeding annotation.

This module provides tools for inferring environment boundaries from
position data, allowing users to adjust pre-drawn boundaries in napari
rather than drawing from scratch.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

# Module logger for debug output
logger = logging.getLogger(__name__)

# Threshold for showing progress message (number of position samples)
LARGE_DATASET_THRESHOLD = 10_000

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from shapely.geometry import Polygon


@dataclass(frozen=True)
class BoundaryConfig:
    """Configuration for boundary inference from positions.

    Parameters
    ----------
    method : {"alpha_shape", "convex_hull"}
        Boundary inference algorithm. Default is "alpha_shape".

        - "alpha_shape": Captures concave boundaries (L-shapes, mazes).
          Provides tighter fit to actual trajectory. Falls back to
          convex_hull if alphashape package not installed.
        - "convex_hull": Fast, robust, always produces single polygon.
          Best when you want guaranteed simple results.
    buffer_fraction : float
        Buffer size as fraction of bounding box diagonal. For example,
        if your trajectory spans 100 cm, buffer_fraction=0.02 adds
        ~2 cm padding around the boundary. Default 0.02 (2%).
    simplify_fraction : float
        Simplification tolerance as fraction of bounding box diagonal.
        Default 0.01 (1%) removes jagged edges. Set to 0 to disable.
    alpha : float
        Alpha parameter for alpha_shape method (smaller = tighter fit).
        Only used when method="alpha_shape". Default 0.05.

    Examples
    --------
    >>> config = BoundaryConfig(method="convex_hull", buffer_fraction=0.05)
    >>> config.method
    'convex_hull'
    >>> config.buffer_fraction
    0.05

    >>> # Default config uses alpha_shape with 2% buffer
    >>> default_config = BoundaryConfig()
    >>> default_config.method
    'alpha_shape'

    """

    method: Literal["alpha_shape", "convex_hull"] = "alpha_shape"
    buffer_fraction: float = 0.02
    simplify_fraction: float = 0.01
    alpha: float = 0.05


def boundary_from_positions(
    positions: NDArray[np.float64],
    method: Literal["alpha_shape", "convex_hull"] | None = None,
    *,
    config: BoundaryConfig | None = None,
    buffer_fraction: float | None = None,
    simplify_fraction: float | None = None,
    **method_kwargs,
) -> Polygon:
    """Infer environment boundary from position data.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, 2)
        Animal positions in (x, y) format. NaN/Inf values are automatically
        filtered with a warning (common in tracking data for lost frames).
    method : {"alpha_shape", "convex_hull"}, optional
        Boundary inference algorithm. Overrides config.method if provided.
    config : BoundaryConfig, optional
        Full configuration object. If None, uses BoundaryConfig defaults.
    buffer_fraction : float, optional
        Override config.buffer_fraction.
    simplify_fraction : float, optional
        Override config.simplify_fraction.
    **method_kwargs
        Method-specific overrides (alpha for alpha_shape).

    Returns
    -------
    Polygon
        Shapely Polygon representing inferred boundary.

    Raises
    ------
    ValueError
        If positions has wrong shape or insufficient points.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> positions = rng.uniform(0, 100, (100, 2))
    >>> boundary = boundary_from_positions(positions)  # Uses defaults
    >>> boundary.is_valid
    True

    >>> # With custom config
    >>> config = BoundaryConfig(method="convex_hull", buffer_fraction=0.05)
    >>> boundary = boundary_from_positions(positions, config=config)

    """
    # Validate and cast input
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must have shape (n, 2), got {positions.shape}")

    # Filter out NaN/Inf values (common in tracking data for lost frames)
    valid_mask = np.all(np.isfinite(positions), axis=1)
    n_invalid = (~valid_mask).sum()
    positions = positions[valid_mask]

    if n_invalid > 0:
        import warnings

        warnings.warn(
            f"Filtered {n_invalid} positions with NaN/Inf values "
            f"({n_invalid / (len(positions) + n_invalid) * 100:.1f}% of data).",
            UserWarning,
            stacklevel=2,
        )

    if len(positions) < 3:
        raise ValueError(
            f"positions must have at least 3 valid points, got {len(positions)} "
            f"(after filtering {n_invalid} NaN/Inf values)",
        )

    # Check for degenerate cases (all points identical or collinear)
    unique_points = np.unique(positions, axis=0)
    if len(unique_points) < 3:
        raise ValueError(
            f"positions must have at least 3 unique points, got {len(unique_points)}",
        )

    # Build effective config
    cfg = config or BoundaryConfig()
    effective_method = method or cfg.method
    effective_buffer = (
        buffer_fraction if buffer_fraction is not None else cfg.buffer_fraction
    )
    effective_simplify = (
        simplify_fraction if simplify_fraction is not None else cfg.simplify_fraction
    )

    # Compute bounding box diagonal for fraction-based parameters
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)

    # Dispatch to method
    if effective_method == "convex_hull":
        boundary = _convex_hull_boundary(positions)
    elif effective_method == "alpha_shape":
        alpha = method_kwargs.get("alpha", cfg.alpha)
        try:
            boundary = _alpha_shape_boundary(positions, alpha)
        except ImportError:
            import warnings

            warnings.warn(
                "alphashape package not installed. Falling back to convex_hull. "
                "Install with: pip install alphashape",
                UserWarning,
                stacklevel=2,
            )
            boundary = _convex_hull_boundary(positions)
    else:
        raise ValueError(f"Unknown method: {effective_method}")

    # Apply buffer (fraction of bbox diagonal)
    if effective_buffer > 0:
        buffer_distance = effective_buffer * bbox_diagonal
        boundary = boundary.buffer(buffer_distance, join_style="round")

    # Apply simplification (fraction of bbox diagonal)
    if effective_simplify > 0:
        simplify_tolerance = effective_simplify * bbox_diagonal
        boundary = boundary.simplify(simplify_tolerance, preserve_topology=True)

    return boundary


def _convex_hull_boundary(positions: NDArray[np.float64]) -> Polygon:
    """Compute convex hull boundary.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, 2)
        Animal positions in (x, y) format.

    Returns
    -------
    Polygon
        Shapely Polygon representing convex hull.

    Raises
    ------
    ValueError
        If points are collinear or degenerate.

    """
    from scipy.spatial import ConvexHull, QhullError
    from shapely.geometry import Polygon

    try:
        hull = ConvexHull(positions)
    except QhullError as e:
        raise ValueError(
            "Cannot compute convex hull: points may be collinear or degenerate. "
            f"Original error: {e}",
        ) from e

    vertices = positions[hull.vertices]
    return Polygon(vertices)


def _alpha_shape_boundary(
    positions: NDArray[np.float64],
    alpha: float,
) -> Polygon:
    """Compute alpha shape (concave hull) boundary.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, 2)
        Animal positions in (x, y) format.
    alpha : float
        Alpha parameter (smaller = tighter fit).

    Returns
    -------
    Polygon
        Shapely Polygon representing alpha shape.

    Raises
    ------
    ImportError
        If alphashape package not installed.

    """
    import warnings

    try:
        import alphashape
    except ImportError:
        raise ImportError(
            "alphashape package required for alpha_shape method. "
            "Install with: pip install alphashape",
        ) from None

    from shapely.geometry import MultiPolygon

    # Progress feedback for large datasets
    n_points = len(positions)
    if n_points >= LARGE_DATASET_THRESHOLD:
        logger.info(
            "Computing alpha shape for %d points (this may take a moment)...",
            n_points,
        )
        # Also print to stdout for users who don't have logging configured
        print(
            f"Computing alpha shape boundary for {n_points:,} positions...",
            file=sys.stderr,
            flush=True,
        )

    result = alphashape.alphashape(positions, alpha)

    if n_points >= LARGE_DATASET_THRESHOLD:
        logger.info("Alpha shape computation complete")

    # Handle MultiPolygon: take largest, warn user
    if isinstance(result, MultiPolygon):
        largest = max(result.geoms, key=lambda g: g.area)
        warnings.warn(
            f"Alpha shape produced {len(result.geoms)} disconnected regions "
            f"from your position data. Using largest polygon (area={largest.area:.1f}). "
            f"This usually means your trajectory has spatial gaps or separate clusters.\n\n"
            f"To fix:\n"
            f"  1. Increase alpha from {alpha:.3f} to ~{alpha * 2:.3f} for a looser boundary\n"
            f"  2. Use method='convex_hull' for guaranteed single polygon\n"
            f"  3. Fill gaps in your position data before boundary inference",
            UserWarning,
            stacklevel=3,
        )
        return largest

    return result
