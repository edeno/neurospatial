from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from neurospatial.layout._labels import LayoutType
from neurospatial.layout.base import capture_build_params
from neurospatial.layout.helpers.utils import check_grid_size_safety
from neurospatial.layout.mixins import _GridMixin
from neurospatial.layout.validation import validate_cell_size


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
    dimension_ranges: Sequence[tuple[float, float]] | None = None

    grid_edges: tuple[NDArray[np.float64], ...] | None = None
    grid_shape: tuple[int, ...] | None = None
    active_mask: NDArray[np.bool_] | None = None

    layout_type: str = LayoutType.IMAGE_MASK.value
    size_parameter: ClassVar[str | None] = "pixel_size"
    _build_params_used: dict[str, Any]

    def __init__(self) -> None:
        """Initialize an ImageMaskLayout engine."""
        self._build_params_used = {}
        self.bin_centers = np.empty((0, 2), dtype=np.float64)
        self.dimension_ranges = None
        self.grid_edges = None
        self.grid_shape = None
        self.active_mask = None

    @capture_build_params(data=("image_mask",))
    def build(
        self,
        *,
        image_mask: NDArray[
            np.bool_
        ],  # Defines candidate pixels, shape (n_rows, n_cols)
        pixel_size: float | tuple[float, float] | None = None,
        connect_diagonal_neighbors: bool = True,
    ) -> None:
        """Build the layout from a 2D image mask.

        The layout follows the (x, y) coordinate convention: dimension 0 is x
        (image columns), dimension 1 is y (image rows). `grid_shape`,
        `grid_edges`, `bin_centers`, and `active_mask` are all expressed in
        (x, y) order so that the grid locator digitizes ``points[:, 0]``
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
            height (y). If None, defaults to 1.0
            (one unit per pixel).
        connect_diagonal_neighbors : bool, default=True
            If True, connect diagonally adjacent active pixel-bins.

        Raises
        ------
        TypeError
            If `image_mask` is not a NumPy array.
        ValueError
            If `image_mask` is not 2D, not boolean, contains no True values,
            or if `pixel_size` is invalid (wrong type/shape or non-positive).

        """
        if pixel_size is None:
            pixel_size = 1.0  # one unit per pixel

        if not isinstance(image_mask, np.ndarray):
            raise TypeError("image_mask must be a numpy array.")
        if image_mask.ndim != 2:
            raise ValueError("image_mask must be a 2D array.")
        if not np.issubdtype(image_mask.dtype, np.bool_):
            raise ValueError("image_mask must be a boolean array.")

        # One size, or (width, height) of a pixel, also as an ndarray read
        # back from to_dict.
        pixel_size_x, pixel_size_y = validate_cell_size(
            pixel_size, name="pixel_size", n_dims=2
        ).tolist()
        if not np.any(image_mask):
            raise ValueError("image_mask must contain at least one True value.")

        n_rows, n_cols = image_mask.shape  # rows = y, cols = x

        # Coordinate convention: dimension 0 is x (cols), dimension 1 is y (rows).
        # grid_shape, grid_edges, bin_centers, and active_mask are all in (x, y)
        # order so that the grid locator digitizes points[:, 0] against x_edges
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
        self._assign_cells(
            full_grid_bin_centers, connect_diagonal=connect_diagonal_neighbors
        )
