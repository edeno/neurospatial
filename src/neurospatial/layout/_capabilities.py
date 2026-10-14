"""Layout capabilities behind grid-only and track-only environment operations.

A layout has a capability only when it defines the method, so ``isinstance``
against the protocol is a truthful check, as for
:class:`~neurospatial.layout._diffusion.FiniteVolumeGeometryProvider`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error

if TYPE_CHECKING:
    from neurospatial.layout._binning import Binning


@dataclass(frozen=True, eq=False)
class RectilinearGrid:
    """Axis-aligned rectangular cells, in the layout's own coordinates.

    Attributes
    ----------
    edges : tuple of ndarray
        Cell edges along each axis; ``edges[d]`` has shape ``(shape[d] + 1,)``.
        Spacing may vary along an axis.
    shape : tuple of int
        Number of cells along each axis, active or not.
    active_mask : ndarray of bool, shape ``shape``
        Which cells are environment bins; active bins are numbered in
        row-major order of this mask.
    """

    edges: tuple[NDArray[np.float64], ...]
    shape: tuple[int, ...]
    active_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        edges = tuple(
            np.array(edge, dtype=np.float64, copy=True) for edge in self.edges
        )
        active_mask = np.array(self.active_mask, dtype=np.bool_, copy=True)
        for array in (*edges, active_mask):
            array.setflags(write=False)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "active_mask", active_mask)

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore write protection after pickle or deepcopy."""
        for name, value in state.items():
            object.__setattr__(self, name, value)
        for array in (*self.edges, self.active_mask):
            array.setflags(write=False)


@runtime_checkable
class RectilinearGridProvider(Protocol):
    """Layout capability behind linear interpolation and linear time allocation."""

    def rectilinear_grid(self) -> RectilinearGrid:
        """Return this layout's cells: per-axis edges, shape and active mask."""
        ...


@runtime_checkable
class TrackLinearizationProvider(Protocol):
    """Layout capability behind ``to_linear`` and ``linear_to_nd``."""

    def to_linear(self, data_points: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def linear_to_nd(
        self, linear_coordinates: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...


def require_rectilinear(
    binning: Binning, operation: str, *, fix: str
) -> RectilinearGrid:
    """Return the published rectangular cells, or raise naming ``operation``.

    Raises
    ------
    NotImplementedError
        If the binning has no axis-aligned rectangular cells.
    """
    if binning.rectilinear is not None:
        return binning.rectilinear
    raise NotImplementedError(
        _format_error(
            f"{operation} needs a grid of axis-aligned rectangular cells; this "
            f"environment's {binning.kind} layout does not have one.",
            why=(
                "Why: it reads the cell edges along each axis, which grids built "
                "by from_samples, from_grid_mask, from_pixel_mask and "
                "from_polygon provide."
            ),
            fix=fix,
        )
    )
