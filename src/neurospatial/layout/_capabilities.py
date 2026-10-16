"""Native builder interfaces for owned grid data."""

from typing import Protocol, runtime_checkable

from neurospatial.geometry._capabilities import RectilinearGrid


@runtime_checkable
class RectilinearGridProvider(Protocol):
    """Layout capability behind linear interpolation and linear time allocation."""

    def rectilinear_grid(self) -> RectilinearGrid:
        """Return this layout's cells: per-axis edges, shape and active mask."""
        ...
