from collections.abc import Sequence
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import Polygon

from neurospatial._exceptions import LayoutNotBuiltError
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

    layout_type: ClassVar[str] = LayoutType.SHAPELY_POLYGON.value
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
            The side length(s) of the grid cells, in the units of the polygon
            coordinates. A float gives square cells; a length-2 sequence
            gives (width, height).
        connect_diagonal_neighbors : bool, default=True
            If True, connect diagonally adjacent active grid cells in the
            `connectivity`.

        Raises
        ------
        TypeError
            If `polygon` is not a Shapely Polygon, or `bin_size` is not
            numeric.
        ValueError
            If `bin_size` is not positive and finite or has the wrong number
            of entries, if `polygon` is empty (its bounds are not finite), or
            if no cell center lies inside the polygon.

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

        # Warn if the grid is very large.
        check_grid_size_safety(self.grid_shape)

        # Cells whose centers lie strictly inside the polygon are active.
        self.active_mask = shapely.contains_xy(
            polygon, full_grid_bin_centers[:, 0], full_grid_bin_centers[:, 1]
        ).reshape(self.grid_shape)

        self._assign_cells(
            full_grid_bin_centers, connect_diagonal=connect_diagonal_neighbors
        )

    def binning(self) -> Binning:
        """Return the grid's owned binning, with the polygon as its outline."""
        return replace(super().binning(), outline=self.outline())

    def outline(self) -> Outline:
        """Return the defining polygon, including any interior rings.

        Raises
        ------
        LayoutNotBuiltError
            If the layout has not been built.
        """
        if self.polygon_definition_ is None:
            raise LayoutNotBuiltError(type(self).__name__, "outline")
        return Outline(self.polygon_definition_)
