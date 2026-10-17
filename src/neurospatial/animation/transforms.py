"""Coordinate transformation utilities for animation backends.

This module provides coordinate transformation functions to convert between
environment (x, y) coordinates and napari pixel (row, col) coordinates.
These transforms handle axis swapping, Y-axis inversion, and scaling.

The napari coordinate system has:
- Row 0 at the top of the image
- Columns increase left-to-right
- Y-axis is inverted relative to typical environment coordinates

Environment coordinate system (typical):
- Y increases upward
- X increases rightward
- Origin at bottom-left

Notes
-----
After flipping the RGB image vertically for napari display, row 0 contains
data from max Y. These transforms map environment coordinates to pixel/row
indices that match the flipped image.

Examples
--------
>>> import numpy as np
>>> from neurospatial import Environment
>>> from neurospatial.animation.transforms import transform_coords_for_napari
>>>
>>> positions = np.random.randn(100, 2) * 10
>>> env = Environment.from_samples(positions, bin_size=1.0)
>>> coords = np.array([[5.0, 5.0]])  # Center of env
>>> napari_coords = transform_coords_for_napari(coords, env)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "EnvScale",
    "build_env_to_napari_matrix",
    "make_env_scale",
    "transform_coords_for_napari",
    "transform_direction_for_napari",
]

# Module-level flag for once-per-session fallback warning
_TRANSFORM_FALLBACK_WARNED: bool = False
"""Flag to ensure transform fallback warning is shown only once per session."""


class EnvScale:
    """The napari field image's pixel grid, in environment coordinates.

    Napari draws a field as a raster over the environment's full rectangular
    grid: one pixel per grid cell (``grid_shape``), inactive cells included,
    spanning ``grid_edges``. This records that grid, so environment
    coordinates can be mapped to the image's pixels. It is independent of
    ``dimension_ranges``, which is the extent of the active cells only.

    Attributes
    ----------
    x_min, x_max : float
        First and last grid edges along x (environment units).
    y_min, y_max : float
        First and last grid edges along y.
    n_x, n_y : int
        Grid cells along x (image columns) and y (image rows).
    x_scale, y_scale : float
        Pixels per environment unit along x and y.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.animation.transforms import EnvScale
    >>> env = Environment.from_grid_mask(
    ...     np.ones((4, 2), dtype=bool), grid_edges=(np.arange(5.0), np.arange(3.0))
    ... )
    >>> scale = EnvScale.from_env(env)
    >>> (scale.n_x, scale.n_y, scale.x_scale)
    (4, 2, 1.0)
    """

    __slots__ = ("n_x", "n_y", "x_max", "x_min", "x_scale", "y_max", "y_min", "y_scale")

    def __init__(self, env: Any) -> None:
        """Initialize the pixel grid from the environment's rectangular grid.

        Parameters
        ----------
        env : Environment
            Environment with two ``grid_edges`` axes and ``grid_shape``.
        """
        if getattr(getattr(env, "_binning", None), "grid_frame", None) is not None:
            raise NotImplementedError(
                "Napari raster coordinates do not support framed grids; use Matplotlib, video or HTML rendering."
            )
        x_edges, y_edges = env.grid_edges
        self.x_min, self.x_max = float(x_edges[0]), float(x_edges[-1])
        self.y_min, self.y_max = float(y_edges[0]), float(y_edges[-1])
        self.n_x, self.n_y = env.grid_shape

        # Pixels per environment unit (avoid division by zero)
        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min
        self.x_scale = self.n_x / x_range if x_range > 0 else 1.0
        self.y_scale = self.n_y / y_range if y_range > 0 else 1.0

    @property
    def pixel_size(self) -> tuple[float, float]:
        """Size of one image pixel as ``(height, width)`` in environment units."""
        return 1.0 / self.y_scale, 1.0 / self.x_scale

    @classmethod
    def from_env(cls, env: Any) -> EnvScale | None:
        """Create EnvScale from environment, or None if not applicable.

        Returns None when the environment has no 2-D rectangular grid, which
        is the only layout napari draws as a raster image (hexagonal, mesh
        and track environments have none), allowing fallback to simple axis
        swap.

        Parameters
        ----------
        env : Environment or None
            Environment to extract scale factors from.

        Returns
        -------
        EnvScale or None
            Scale factors, or None if environment lacks a 2-D grid.

        Examples
        --------
        >>> scale = EnvScale.from_env(env)  # doctest: +SKIP
        >>> if scale is None:  # doctest: +SKIP
        ...     print("Using fallback transform")  # doctest: +SKIP
        """
        if getattr(getattr(env, "_binning", None), "grid_frame", None) is not None:
            return cls(env)  # refuses framed grids with the fix
        if (
            env is None
            or len(getattr(env, "grid_edges", ())) != 2
            or len(getattr(env, "grid_shape", ())) != 2
        ):
            return None
        return cls(env)

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return (
            f"EnvScale(x=[{self.x_min:.2f}, {self.x_max:.2f}], "
            f"y=[{self.y_min:.2f}, {self.y_max:.2f}], "
            f"grid=({self.n_x}, {self.n_y}))"
        )


def make_env_scale(env: Any) -> EnvScale | None:
    """Create cached scale factors from environment.

    Convenience function wrapping EnvScale.from_env().

    Parameters
    ----------
    env : Environment
        Environment instance.

    Returns
    -------
    scale : EnvScale or None
        Cached scale factors, or None if env lacks required attributes.
    """
    return EnvScale.from_env(env)


def build_env_to_napari_matrix(scale: EnvScale) -> NDArray[np.float64]:
    """Build 3x3 homogeneous matrix for env_cm → napari_px transform.

    This matrix encodes the same transformation as transform_coords_for_napari()
    but in matrix form for use with napari's affine parameter.

    The transformation maps environment coordinates (x, y) to napari pixel
    coordinates (row, col), with pixel ``i`` centered on grid cell ``i``:
    - col = (x - x_min) * x_scale - 0.5
    - row = (n_y - 0.5) - (y - y_min) * y_scale

    Parameters
    ----------
    scale : EnvScale
        Pre-computed scale factors from environment.

    Returns
    -------
    T : ndarray of shape (3, 3)
        Homogeneous transformation matrix. Apply as:
        ``napari_coords = (T @ [x, y, 1].T).T[:2]``

    Notes
    -----
    The matrix form is:

    .. code-block:: text

        [row]   [0,       -y_scale, (n_y-0.5) + y_min*y_scale] [x]
        [col] = [x_scale,  0,       -x_min*x_scale - 0.5     ] [y]
        [1  ]   [0,        0,        1                       ] [1]

    This is derived from the transformation equations:
    - row = (n_y - 0.5) - (y - y_min) * y_scale
          = -y_scale * y + ((n_y - 0.5) + y_min*y_scale)
    - col = (x - x_min) * x_scale - 0.5
          = x_scale * x - (x_min*x_scale + 0.5)

    Examples
    --------
    >>> from neurospatial.animation.transforms import (  # doctest: +SKIP
    ...     EnvScale,
    ...     build_env_to_napari_matrix,
    ... )
    >>> scale = EnvScale.from_env(env)  # doctest: +SKIP
    >>> matrix = build_env_to_napari_matrix(scale)  # doctest: +SKIP
    >>> # Transform point (x=5.0, y=5.0)
    >>> point = np.array([5.0, 5.0, 1.0])  # doctest: +SKIP
    >>> napari_coords = matrix @ point  # [row, col, 1]  # doctest: +SKIP

    See Also
    --------
    transform_coords_for_napari : Function-based coordinate transformation
    EnvScale : Cached scale factors for transformation
    """
    return np.array(
        [
            [0.0, -scale.y_scale, (scale.n_y - 0.5) + scale.y_min * scale.y_scale],
            [scale.x_scale, 0.0, -scale.x_min * scale.x_scale - 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _warn_fallback(suppress: bool = False) -> None:
    """Emit warning about fallback transform behavior (once per session).

    Parameters
    ----------
    suppress : bool, optional
        If True, suppress the warning even if it would normally be shown.
        Used by napari backend for per-viewer warning tracking.
        Default is False (allow warning to be emitted).
    """
    global _TRANSFORM_FALLBACK_WARNED
    if suppress:
        return
    if not _TRANSFORM_FALLBACK_WARNED:
        warnings.warn(
            "WHAT: Napari coordinate transform falling back to simple axis swap.\n"
            "  This may cause misalignment between overlays and the field image.\n\n"
            "WHY: Napari draws a field as a raster over a 2-D rectangular grid\n"
            "  (grid_edges and grid_shape), and this environment has none:\n"
            "  hexagonal, mesh and track layouts are not rasters.\n\n"
            "Fix: Use a grid environment (Environment.from_samples, from_polygon,\n"
            "  from_grid_mask or from_pixel_mask) for napari, or render this\n"
            "  layout with the video or HTML backend",
            UserWarning,
            stacklevel=4,
        )
        _TRANSFORM_FALLBACK_WARNED = True


def transform_coords_for_napari(
    coords: NDArray[np.float64],
    env_or_scale: Any | EnvScale | None = None,
    *,
    suppress_warning: bool = False,
) -> NDArray[np.float64]:
    """Transform coordinates from environment (x, y) to napari pixel (row, col).

    Napari displays images with row 0 at top. After flipping the RGB image vertically,
    row 0 contains data from max Y. This function maps environment coordinates to
    pixel/row indices that match the flipped image.

    Transformation, with pixel ``i`` centered on grid cell ``i`` of the full
    grid (``grid_edges``, inactive cells included):
    1. Map X to column index: col = (x - x_min) / (x_max - x_min) * n_x - 0.5
    2. Map Y to flipped row index: row = (y_max - y) / (y_max - y_min) * n_y - 0.5

    Parameters
    ----------
    coords : ndarray
        Coordinates with shape (..., n_dims) where last dimension is spatial.
        For 2D data, expects (..., 2) with (x, y) ordering in environment space.
    env_or_scale : Environment or EnvScale, optional
        Environment instance or pre-computed EnvScale for coordinate transformation.
        Required for 2D coords to avoid misalignment.
    suppress_warning : bool, optional
        If True, suppress the fallback warning even if it would normally be shown.
        Used by napari backend for per-viewer warning tracking. Default is False.

    Returns
    -------
    transformed : ndarray
        Coordinates in napari pixel space. For 2D: (x, y) -> (row, col).
        Higher dimensions returned unchanged.

    Examples
    --------
    >>> coords = np.array([[5.0, 5.0]])  # Center of environment
    >>> napari_coords = transform_coords_for_napari(coords, env)  # doctest: +SKIP
    >>> row, col = napari_coords[0]  # doctest: +SKIP
    """
    if coords.shape[-1] == 2:
        # Get or create scale factors
        scale: EnvScale | None
        if isinstance(env_or_scale, EnvScale):
            scale = env_or_scale
        else:
            scale = make_env_scale(env_or_scale)

        if scale is None:
            # Fallback: just swap x and y (may cause alignment issues)
            _warn_fallback(suppress=suppress_warning)
            return coords[..., ::-1]

        x_coords = coords[..., 0]
        y_coords = coords[..., 1]

        # Map to pixel indices using cached scale factors
        # X -> column (no flip)
        col = (x_coords - scale.x_min) * scale.x_scale - 0.5

        # Y -> row (with flip: high Y -> low row)
        row = (scale.n_y - 0.5) - (y_coords - scale.y_min) * scale.y_scale

        # Return in napari (row, col) order. Allocate float64 explicitly:
        # np.empty_like would inherit an integer dtype from integer-typed input
        # (e.g. pixel-mask or DLC coords) and silently truncate the fractional
        # pixel position.
        result = np.empty(coords.shape, dtype=np.float64)
        result[..., 0] = row
        result[..., 1] = col
        return result

    # For other dimensions, return unchanged
    return coords


def transform_direction_for_napari(
    direction: NDArray[np.float64],
    env_or_scale: Any | EnvScale | None = None,
    *,
    suppress_warning: bool = False,
) -> NDArray[np.float64]:
    """Transform direction vectors from environment (dx, dy) to napari (dr, dc).

    Unlike positions, direction vectors are displacements and should NOT be
    translated. They need only axis swapping, Y-axis inversion, and scaling.

    Parameters
    ----------
    direction : ndarray
        Direction vectors with shape (..., n_dims) where last dimension is spatial.
        For 2D data, expects (..., 2) with (dx, dy) ordering in environment space.
    env_or_scale : Environment or EnvScale, optional
        Environment instance or pre-computed EnvScale for scaling.
        If None, only swaps axes and inverts Y.
    suppress_warning : bool, optional
        If True, suppress the fallback warning even if it would normally be shown.
        Used by napari backend for per-viewer warning tracking. Default is False.

    Returns
    -------
    transformed : ndarray
        Direction vectors in napari space. For 2D: (dx, dy) -> (dr, dc).

    Examples
    --------
    >>> direction = np.array([[1.0, 0.0]])  # Pointing right in env
    >>> napari_dir = transform_direction_for_napari(direction, env)  # doctest: +SKIP
    >>> # In napari: positive X -> positive col direction
    """
    if direction.shape[-1] == 2:
        dx = direction[..., 0]
        dy = direction[..., 1]

        # Get or create scale factors
        scale: EnvScale | None
        if isinstance(env_or_scale, EnvScale):
            scale = env_or_scale
        else:
            scale = make_env_scale(env_or_scale)

        if scale is None:
            # Fallback: just swap axes and invert Y (may cause alignment issues)
            _warn_fallback(suppress=suppress_warning)
            result = np.empty(direction.shape, dtype=np.float64)
            result[..., 0] = -dy  # Y inverted (environment Y up, napari row down)
            result[..., 1] = dx
            return result

        # Scale direction vectors using cached scale factors (no translation!)
        # X -> column (scale only)
        dc = dx * scale.x_scale
        # Y -> row (scale and invert: positive dy -> negative dr)
        dr = -dy * scale.y_scale

        result = np.empty(direction.shape, dtype=np.float64)
        result[..., 0] = dr
        result[..., 1] = dc
        return result

    # For other dimensions, return unchanged
    return direction


def reset_transform_warning() -> None:
    """Reset the fallback warning flag (for testing purposes).

    This allows the fallback warning to be shown again. Useful in test
    cleanup or when deliberately testing warning behavior.
    """
    global _TRANSFORM_FALLBACK_WARNED
    _TRANSFORM_FALLBACK_WARNED = False
