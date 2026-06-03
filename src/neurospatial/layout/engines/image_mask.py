import warnings
from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.layout.base import capture_build_params
from neurospatial.layout.helpers.regular_grid import (
    _create_regular_grid_connectivity_graph,
)
from neurospatial.layout.helpers.utils import check_grid_size_safety
from neurospatial.layout.mixins import _GridMixin
from neurospatial.layout.validation import validate_connectivity_graph


class ImageMaskLayout(_GridMixin):
    """2D layout derived from a boolean image mask.

    Each `True` pixel in the input `image_mask` corresponds to an active bin
    in the environment. The spatial scale of these pixel-bins is determined
    by `pixel_size`. Inherits grid functionalities from `_GridMixin`.

    The layout uses the (x, y) coordinate convention: dimension 0 is x
    (image columns) and dimension 1 is y (image rows). `grid_shape`,
    `grid_edges`, `bin_centers`, and `active_mask` are all expressed in
    (x, y) order, matching `dimension_ranges` and every other grid engine.
    """

    bin_centers: NDArray[np.float64]
    connectivity: nx.Graph | None = None
    dimension_ranges: Sequence[tuple[float, float]] | None = None

    grid_edges: tuple[NDArray[np.float64], ...] | None = None
    grid_shape: tuple[int, ...] | None = None
    active_mask: NDArray[np.bool_] | None = None

    _layout_type_tag: str
    _build_params_used: dict[str, Any]

    def __init__(self) -> None:
        """Initialize an ImageMaskLayout engine."""
        self._layout_type_tag = "ImageMask"
        self._build_params_used = {}
        self.bin_centers = np.empty((0, 2), dtype=np.float64)
        self.connectivity = None
        self.dimension_ranges = None
        self.grid_edges = None
        self.grid_shape = None
        self.active_mask = None

    @property
    def layout_type(self) -> str:
        """Return standardized category for this layout type."""
        return "mask"

    @property
    def is_grid_compatible(self) -> bool:
        """Return True - image mask layouts can be rendered as 2D images."""
        return True

    @capture_build_params
    def build(
        self,
        *,
        image_mask: NDArray[
            np.bool_
        ],  # Defines candidate pixels, shape (n_rows, n_cols)
        pixel_size: float | tuple[float, float] | None = None,
        connect_diagonal_neighbors: bool = True,
        bin_size: float | tuple[float, float] | None = None,  # deprecated alias
    ) -> None:
        """Build the layout from a 2D image mask.

        The layout follows the (x, y) coordinate convention: dimension 0 is x
        (image columns), dimension 1 is y (image rows). `grid_shape`,
        `grid_edges`, `bin_centers`, and `active_mask` are all expressed in
        (x, y) order so that `point_to_bin_index` digitizes ``points[:, 0]``
        against the x edges and ``points[:, 1]`` against the y edges.

        Parameters
        ----------
        image_mask : NDArray[np.bool_], shape (n_rows, n_cols)
            A 2D boolean array where `True` pixels define active bins. Indexed
            as ``[row, col]`` == ``[y, x]``.
        pixel_size : float or tuple of (float, float) or None, optional
            The spatial size of each pixel, in units per pixel.
            If float: pixels are square (size x size).
            If tuple (width, height): specifies pixel width (x) and pixel
            height (y). If None and `bin_size` is also None, defaults to 1.0
            (one unit per pixel).
        connect_diagonal_neighbors : bool, default=True
            If True, connect diagonally adjacent active pixel-bins.
        bin_size : float or tuple of (float, float) or None, optional
            Deprecated alias for `pixel_size`, kept for backward compatibility.
            Emits a `DeprecationWarning` when used. Cannot be combined with
            `pixel_size`.

        Raises
        ------
        TypeError
            If `image_mask` is not a NumPy array.
        ValueError
            If `image_mask` is not 2D, not boolean, contains no True values,
            if `pixel_size` is invalid (wrong type/shape or non-positive), or
            if both `pixel_size` and the deprecated `bin_size` alias are given.

        """
        # Resolve the pixel-size argument. ``pixel_size`` is the public name;
        # ``bin_size`` is accepted as a deprecated alias for backward
        # compatibility with callers that forwarded the legacy key.
        if pixel_size is not None and bin_size is not None:
            raise ValueError(
                "Pass either 'pixel_size' or the deprecated 'bin_size' alias, not both."
            )
        if pixel_size is None:
            if bin_size is None:
                pixel_size = 1.0  # one unit per pixel
            else:
                warnings.warn(
                    "'bin_size' is deprecated for ImageMaskLayout; use "
                    "'pixel_size' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                pixel_size = bin_size

        if not isinstance(image_mask, np.ndarray):
            raise TypeError("image_mask must be a numpy array.")
        if image_mask.ndim != 2:
            raise ValueError("image_mask must be a 2D array.")
        if not np.issubdtype(image_mask.dtype, np.bool_):
            raise ValueError("image_mask must be a boolean array.")

        # Validate pixel_size. Use an array-safe comparison so that both scalar
        # and per-axis pixel sizes -- including those deserialized from JSON as
        # an ndarray (e.g. an anisotropic (3.0, 1.0) round-tripped through
        # to_dict/from_dict) -- are checked without raising the "truth value of
        # an array is ambiguous" error.
        if np.any(np.asarray(pixel_size, dtype=float) <= 0):
            raise ValueError("pixel_size must be positive.")
        if not np.any(image_mask):
            raise ValueError("image_mask must contain at least one True value.")
        if not np.all(np.isfinite(image_mask)):
            raise ValueError("image_mask must not contain NaN or Inf values.")

        # Determine pixel sizes for x and y (units per pixel)
        pixel_size_x: float
        pixel_size_y: float
        if isinstance(pixel_size, (float, int, np.number)):
            pixel_size_x = float(pixel_size)
            pixel_size_y = float(pixel_size)
        elif isinstance(pixel_size, (list, tuple, np.ndarray)) and len(pixel_size) == 2:
            pixel_size_x = float(pixel_size[0])  # width of a pixel (x)
            pixel_size_y = float(pixel_size[1])  # height of a pixel (y)
        else:
            raise ValueError(
                "pixel_size for ImageMaskLayout must be a float or a "
                "2-element sequence (width, height).",
            )
        if pixel_size_x <= 0 or pixel_size_y <= 0:
            raise ValueError("pixel_size components must be positive.")

        n_rows, n_cols = image_mask.shape  # rows = y, cols = x

        # Coordinate convention: dimension 0 is x (cols), dimension 1 is y (rows).
        # grid_shape, grid_edges, bin_centers, and active_mask are all in (x, y)
        # order so that point_to_bin_index digitizes points[:, 0] against x_edges
        # and points[:, 1] against y_edges (matching dimension_ranges and every
        # other grid engine).
        self.grid_shape = (n_cols, n_rows)

        # Safety check: warn or error if grid is very large
        n_dims = 2  # ImageMask is always 2D
        check_grid_size_safety(self.grid_shape, n_dims)

        x_edges = np.arange(n_cols + 1) * pixel_size_x
        y_edges = np.arange(n_rows + 1) * pixel_size_y
        self.grid_edges = (x_edges, y_edges)
        self.dimension_ranges = (
            (x_edges[0], x_edges[-1]),
            (y_edges[0], y_edges[-1]),
        )

        x_centers = (np.arange(n_cols) + 0.5) * pixel_size_x
        y_centers = (np.arange(n_rows) + 0.5) * pixel_size_y
        # indexing="ij" over (x_centers, y_centers): outer loop x, inner loop y,
        # giving the same x-major ravel order as grid_shape == (n_cols, n_rows).
        xv, yv = np.meshgrid(x_centers, y_centers, indexing="ij")
        full_grid_bin_centers = np.stack((xv.ravel(), yv.ravel()), axis=1)

        # image_mask is (rows, cols) == (y, x); transpose to (x, y) so its ravel
        # aligns row-for-row with full_grid_bin_centers above.
        self.active_mask = np.ascontiguousarray(image_mask.T)
        self.bin_centers = full_grid_bin_centers[self.active_mask.ravel()]
        self.connectivity = _create_regular_grid_connectivity_graph(
            full_grid_bin_centers=full_grid_bin_centers,
            active_mask_nd=self.active_mask,
            grid_shape=self.grid_shape,
            connect_diagonal=connect_diagonal_neighbors,
        )

        # Validate connectivity graph has required attributes
        validate_connectivity_graph(self.connectivity, n_dims=2)
