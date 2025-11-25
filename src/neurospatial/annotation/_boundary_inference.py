"""Boundary inference algorithms for seeding annotation.

This module provides tools for inferring environment boundaries from
position data, allowing users to adjust pre-drawn boundaries in napari
rather than drawing from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from shapely.geometry import Polygon


@dataclass(frozen=True)
class BoundaryConfig:
    """
    Configuration for boundary inference from positions.

    Parameters
    ----------
    method : {"convex_hull", "alpha_shape", "kde"}
        Boundary inference algorithm. Default is "convex_hull".

        - "convex_hull": Fast, robust, always produces single polygon.
          Best for most cases where trajectory covers the environment.
        - "alpha_shape": For concave environments (L-shapes, mazes).
          Requires good spatial coverage. Install: pip install alphashape
        - "kde": Density-based contour for irregular coverage or sparse
          trajectories. Install: pip install scikit-image
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
    kde_threshold : float
        Density threshold for KDE boundary (0-1, fraction of max density).
        Only used when method="kde". Default 0.1.
    kde_sigma : float
        Gaussian smoothing sigma for KDE (in grid bins).
        Only used when method="kde". Default 3.0.
    kde_max_bins : int
        Maximum number of bins per dimension for KDE grid.
        Caps memory usage for large coordinate ranges. Default 512.

    Examples
    --------
    >>> config = BoundaryConfig(method="kde", buffer_fraction=0.05)
    >>> config.method
    'kde'
    >>> config.buffer_fraction
    0.05

    >>> # Default config uses convex_hull with 2% buffer
    >>> default_config = BoundaryConfig()
    >>> default_config.method
    'convex_hull'
    """

    method: Literal["convex_hull", "alpha_shape", "kde"] = "convex_hull"
    buffer_fraction: float = 0.02
    simplify_fraction: float = 0.01
    alpha: float = 0.05
    kde_threshold: float = 0.1
    kde_sigma: float = 3.0
    kde_max_bins: int = 512


def boundary_from_positions(
    positions: NDArray[np.float64],
    method: Literal["convex_hull", "alpha_shape", "kde"] | None = None,
    *,
    config: BoundaryConfig | None = None,
    buffer_fraction: float | None = None,
    simplify_fraction: float | None = None,
    **method_kwargs,
) -> Polygon:
    """
    Infer environment boundary from position data.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, 2)
        Animal positions in (x, y) format. NaN/Inf values are automatically
        filtered with a warning (common in tracking data for lost frames).
    method : {"convex_hull", "alpha_shape", "kde"}, optional
        Boundary inference algorithm. Overrides config.method if provided.
    config : BoundaryConfig, optional
        Full configuration object. If None, uses BoundaryConfig defaults.
    buffer_fraction : float, optional
        Override config.buffer_fraction.
    simplify_fraction : float, optional
        Override config.simplify_fraction.
    **method_kwargs
        Method-specific overrides (alpha, kde_threshold, kde_sigma, kde_max_bins).

    Returns
    -------
    Polygon
        Shapely Polygon representing inferred boundary.

    Raises
    ------
    ValueError
        If positions has wrong shape or insufficient points.
    ImportError
        If method="alpha_shape" and alphashape not installed.

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
            f"(after filtering {n_invalid} NaN/Inf values)"
        )

    # Check for degenerate cases (all points identical or collinear)
    unique_points = np.unique(positions, axis=0)
    if len(unique_points) < 3:
        raise ValueError(
            f"positions must have at least 3 unique points, got {len(unique_points)}"
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
        boundary = _alpha_shape_boundary(positions, alpha)
    elif effective_method == "kde":
        threshold = method_kwargs.get("kde_threshold", cfg.kde_threshold)
        sigma = method_kwargs.get("kde_sigma", cfg.kde_sigma)
        max_bins = method_kwargs.get("kde_max_bins", cfg.kde_max_bins)
        boundary = _kde_boundary(positions, threshold, sigma, max_bins)
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
            f"Original error: {e}"
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
            "Install with: pip install alphashape"
        ) from None

    from shapely.geometry import MultiPolygon

    result = alphashape.alphashape(positions, alpha)

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


def _kde_boundary(
    positions: NDArray[np.float64],
    threshold: float,
    sigma: float,
    max_bins: int = 512,
) -> Polygon:
    """Compute KDE-based boundary from density contour.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, 2)
        Animal positions in (x, y) format.
    threshold : float
        Density threshold (0-1, fraction of max density).
        Lower values capture more area; higher values are more selective.
    sigma : float
        Gaussian smoothing sigma in grid bins.
        Higher values produce smoother boundaries.
    max_bins : int, default=512
        Maximum bins per dimension for the density grid.
        Caps memory usage for large coordinate ranges while maintaining
        reasonable resolution. Actual bin count is min(max_bins, data_range/2).

    Returns
    -------
    Polygon
        Shapely Polygon representing density contour.

    Raises
    ------
    ImportError
        If scikit-image not installed.
    ValueError
        If no contour found at threshold. Includes diagnostics and
        suggested fixes.
    """
    try:
        from scipy.ndimage import gaussian_filter
        from skimage.measure import find_contours
    except ImportError as e:
        raise ImportError(
            "scikit-image is required for KDE boundary method. "
            "Install with: pip install scikit-image"
        ) from e

    from shapely.geometry import Polygon

    # Create 2D histogram with capped bin count
    x_range = positions[:, 0].max() - positions[:, 0].min()
    y_range = positions[:, 1].max() - positions[:, 1].min()
    # Target ~2 unit bins but cap to prevent memory issues
    n_bins = min(max_bins, max(50, int(max(x_range, y_range) / 2)))

    hist, x_edges, y_edges = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=n_bins
    )

    # Smooth and normalize
    hist_smooth = gaussian_filter(hist, sigma=sigma)
    hist_norm = hist_smooth / hist_smooth.max()

    # Find contour at threshold
    contours = find_contours(hist_norm.T, level=threshold)
    if not contours:
        max_density = hist_norm.max()
        raise ValueError(
            f"No density contour found at threshold {threshold:.2f} "
            f"(fraction of max density).\n"
            f"Your position data may be too sparse or evenly distributed.\n\n"
            f"To fix (try in order):\n"
            f"  1. Lower kde_threshold from {threshold:.2f} to 0.05 (captures more area)\n"
            f"  2. Increase kde_sigma from {sigma:.1f} to {sigma * 2:.1f} (smoother density)\n"
            f"  3. Use method='convex_hull' instead (simpler, no tuning needed)\n\n"
            f"Diagnostics: max_density={max_density:.3f}, threshold={threshold:.3f}, "
            f"grid_size={n_bins}x{n_bins}"
        )

    # Take largest contour
    largest = max(contours, key=len)

    # Convert grid indices to coordinates
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    coords = np.column_stack(
        [
            np.interp(largest[:, 1], np.arange(len(x_centers)), x_centers),
            np.interp(largest[:, 0], np.arange(len(y_centers)), y_centers),
        ]
    )

    return Polygon(coords)
