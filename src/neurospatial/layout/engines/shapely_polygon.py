from collections.abc import Sequence
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import Polygon

from neurospatial.geometry import Binning, Outline
from neurospatial.layout._labels import LayoutType
from neurospatial.layout.base import capture_build_params
from neurospatial.layout.helpers.regular_grid import _create_regular_grid
from neurospatial.layout.helpers.utils import check_grid_size_safety
from neurospatial.layout.mixins import _GridMixin


class ShapelyPolygonLayout(_GridMixin):
    """2D grid layout masked by a Shapely Polygon.

    Creates a regular grid based on the polygon's bounds and specified
    `bin_size`. Only grid cells whose centers are contained within the
    polygon are considered active. Inherits grid functionalities from
    `_GridMixin`.
    """

    bin_centers: NDArray[np.float64]
    dimension_ranges: Sequence[tuple[float, float]] | None = None

    grid_edges: tuple[NDArray[np.float64], ...] | None = None
    grid_shape: tuple[int, ...] | None = None
    active_mask: NDArray[np.bool_] | None = None

    layout_type: str = LayoutType.SHAPELY_POLYGON.value
    size_parameter: ClassVar[str | None] = "bin_size"
    _build_params_used: dict[str, Any]

    # Layout Specific
    polygon_definition_: Polygon | None = None

    def __init__(self) -> None:
        """Initialize a ShapelyPolygonLayout engine."""
        self._build_params_used = {}
        self.bin_centers = np.empty((0, 2), dtype=np.float64)  # 2D Layout
        self.dimension_ranges = None
        self.grid_edges = None
        self.grid_shape = None
        self.active_mask = None
        self.polygon_definition_ = None

    @capture_build_params(data=("polygon",))
    def build(
        self,
        *,
        polygon: Polygon,
        bin_size: float | Sequence[float],
        connect_diagonal_neighbors: bool = True,
    ) -> None:
        """Build the Shapely Polygon masked grid layout.

        Parameters
        ----------
        polygon : shapely.geometry.Polygon
            The Shapely Polygon object that defines the boundary of the
            active area.
        bin_size : Union[float, Sequence[float]]
            The side length(s) of the grid cells. If float, cells are
            square (or cubic). If sequence (length 2 for 2D), specifies
            (width, height).
        connect_diagonal_neighbors : bool, default=True
            If True, connect diagonally adjacent active grid cells in the
            `connectivity`.

        Raises
        ------
        RuntimeError
            If the 'shapely' package is not installed (should be caught by
            SHAPELY_AVAILABLE check at class definition).
        TypeError
            If `polygon` is not a Shapely Polygon.

        """
        self._begin_build()
        if not isinstance(polygon, Polygon):
            raise TypeError("polygon must be a Shapely Polygon object.")

        self.polygon_definition_ = polygon
        minx, miny, maxx, maxy = polygon.bounds
        self.dimension_ranges = [(minx, maxx), (miny, maxy)]

        (
            self.grid_edges,
            full_grid_bin_centers,
            self.grid_shape,
        ) = _create_regular_grid(
            positions=None,
            bin_size=bin_size,
            dimension_range=self.dimension_ranges,
            add_boundary_bins=False,
        )

        # Safety check: warn or error if grid is very large
        n_dims = 2  # ShapelyPolygon is always 2D
        check_grid_size_safety(self.grid_shape, n_dims)

        # 1. Intrinsic mask from Shapely
        pts_to_check = (
            full_grid_bin_centers[:, :2]
            if full_grid_bin_centers.shape[0] > 0
            else np.empty((0, 2))
        )
        shapely_mask_flat = (
            shapely.contains(polygon, shapely.points(pts_to_check))
            if pts_to_check.shape[0] > 0
            else np.array([], dtype=bool)
        )
        self.active_mask = shapely_mask_flat.reshape(self.grid_shape)

        self._assign_cells(
            full_grid_bin_centers, connect_diagonal=connect_diagonal_neighbors
        )

    def binning(self) -> Binning:
        """Return the grid's owned binning, with the polygon as its outline."""
        return replace(super().binning(), outline=self.outline())

    def outline(self) -> Outline:
        """Return the defining polygon, including any interior rings."""
        if self.polygon_definition_ is None:
            raise RuntimeError(
                "Polygon definition is missing; ensure build() was called successfully."
            )
        return Outline(self.polygon_definition_)
