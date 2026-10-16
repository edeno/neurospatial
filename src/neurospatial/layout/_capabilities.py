"""Native builder interfaces and compatibility exports for owned grid data."""

from typing import Protocol, runtime_checkable

from neurospatial.geometry._capabilities import (
    RectilinearGrid as RectilinearGrid,
)
from neurospatial.geometry._capabilities import (
    require_rectilinear as require_rectilinear,
)


@runtime_checkable
class RectilinearGridProvider(Protocol):
    """Layout capability behind linear interpolation and linear time allocation."""

    def rectilinear_grid(self) -> RectilinearGrid:
        """Return this layout's cells: per-axis edges, shape and active mask."""
        ...
