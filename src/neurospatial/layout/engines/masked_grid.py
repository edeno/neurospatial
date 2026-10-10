from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.layout.base import capture_build_params
from neurospatial.layout.helpers.regular_grid import (
    _create_regular_grid_connectivity_graph,
)
from neurospatial.layout.helpers.utils import check_grid_size_safety, get_centers
from neurospatial.layout.mixins import _GridMixin
from neurospatial.layout.validation import validate_connectivity_graph


class MaskedGridLayout(_GridMixin):
    """Layout from a pre-defined N-D boolean mask and explicit grid edges.

    Allows for precise specification of active bins in an N-dimensional grid
    by providing the complete grid structure (`grid_edges`) and a mask
    (`active_mask`) that designates which cells of that grid are active.
    Inherits grid functionalities from `_GridMixin`.
    """

    bin_centers: NDArray[np.float64]
    connectivity: nx.Graph | None = None
    dimension_ranges: Sequence[tuple[float, float]] | None = None

    grid_edges: tuple[NDArray[np.float64], ...] | None = None
    grid_shape: tuple[int, ...] | None = None
    active_mask: NDArray[np.bool_] | None = None

    _layout_type_tag: str
    _build_params_used: dict[str, Any]
    bin_size_: NDArray[np.float64] | None = None

    def __init__(self) -> None:
        """Initialize a MaskedGridLayout engine."""
        self._layout_type_tag = "MaskedGrid"
        self._build_params_used = {}
        self.bin_centers = np.empty((0, 2), dtype=np.float64)
        self.connectivity = None
        self.dimension_ranges = None
        self.grid_edges = None
        self.grid_shape = None
        self.active_mask = None
        self.bin_size_ = None

    @property
    def layout_type(self) -> str:
        """Return standardized category for this layout type."""
        return "mask"

    @property
    def is_grid_compatible(self) -> bool:
        """Return True - masked grids can be rendered as 2D images."""
        return True

    @capture_build_params
    def build(
        self,
        *,
        active_mask: NDArray[np.bool_],  # User's N-D definition mask
        grid_edges: tuple[NDArray[np.float64], ...],
        connect_diagonal_neighbors: bool = True,
    ) -> None:
        """Build the layout from a mask and grid edges.

        Parameters
        ----------
        active_mask : NDArray[np.bool_]
            N-dimensional boolean array where `True` indicates an active bin.
            Its shape must correspond to the number of bins defined by `grid_edges`
            (i.e., `tuple(len(e)-1 for e in grid_edges)`).
        grid_edges : Tuple[NDArray[np.float64], ...]
            A tuple where each element is a 1D NumPy array of bin edge
            positions for that dimension, defining the full grid structure.
            Edges must be finite, strictly increasing, and uniformly spaced
            along each axis; axes may differ.
        connect_diagonal_neighbors : bool, default=True
            If True, connect diagonally adjacent active grid cells.

        Raises
        ------
        TypeError
            If `active_mask` is not a NumPy array.
        ValueError
            If `active_mask` does not have boolean dtype, if its shape does not
            match the `grid_edges` definition, or if `grid_edges` are invalid.

        """
        if not isinstance(active_mask, np.ndarray):
            raise TypeError(
                f"active_mask must be a NumPy array, got {type(active_mask).__name__}."
            )
        if active_mask.dtype != np.bool_:
            raise ValueError(
                f"active_mask must have boolean dtype (np.bool_), got "
                f"{active_mask.dtype}. Convert with `mask.astype(bool)` — but be "
                f"sure the values are genuine True/False flags, not bin data."
            )
        if len(grid_edges) != active_mask.ndim or len(grid_edges) == 0:
            raise ValueError(
                f"grid_edges has {len(grid_edges)} edge arrays but active_mask is "
                f"{active_mask.ndim}-D; one edge array per mask axis is required.\n"
                "Fix: pass grid_edges=(edges_axis0, edges_axis1, ...) matching "
                "active_mask.ndim."
            )
        grid_edges = tuple(np.asarray(e, dtype=np.float64) for e in grid_edges)
        for axis, edges in enumerate(grid_edges):
            if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
                raise ValueError(
                    f"grid_edges[{axis}] must be a finite 1-D array with >= 2 edges, "
                    f"got shape {edges.shape}.\n"
                    "Fix: pass e.g. np.linspace(start, stop, n_bins + 1)."
                )
            widths = np.diff(edges)
            if np.any(widths <= 0):
                raise ValueError(
                    f"grid_edges[{axis}] must be strictly increasing; smallest width "
                    f"is {widths.min():.6g}.\n"
                    "Fix: sort the edges and drop duplicates."
                )
            w0 = widths[0]
            # Float spacing at the largest coordinate bounds each edge's
            # representation error.
            largest = float(np.max(np.abs(edges)))
            ulp = np.spacing(largest)
            if 4 * ulp > 1e-4 * w0:
                raise ValueError(
                    f"grid_edges[{axis}] reaches {largest:.6g}, where float64 "
                    f"resolves only {ulp:.3g}; a bin width of {w0:.6g} cannot be "
                    "represented to 1 part in 10^4 there, so bin widths and volumes "
                    "would be wrong.\n"
                    "Fix: subtract an origin offset before building the environment, "
                    "e.g. positions - positions.min(axis=0)."
                )
            deviation = float(np.max(np.abs(widths - w0)))
            allowance = 1e-7 * w0 + 4 * ulp
            if deviation > allowance:
                raise ValueError(
                    f"grid_edges[{axis}] must be uniformly spaced: widths differ from "
                    f"the first ({w0:.10g}) by up to {deviation:.3g}, more than the "
                    f"{allowance:.3g} rounding allowance; cell volumes and diffusion "
                    "face measures assume one width per axis.\n"
                    "Fix: use np.linspace(start, stop, n_bins + 1) for each axis."
                )

        self.active_mask = active_mask
        self.grid_edges = grid_edges
        self.grid_shape = tuple(len(edge) - 1 for edge in grid_edges)

        if self.active_mask.shape != self.grid_shape:
            raise ValueError(
                f"active_mask shape {self.active_mask.shape} must match "
                f"the shape implied by grid_edges {self.grid_shape}.",
            )

        # Safety check: warn or error if grid is very large
        n_dims = len(self.grid_shape)
        check_grid_size_safety(self.grid_shape, n_dims)

        # Create full_grid_bin_centers as (N_total_bins, N_dims) array
        centers_per_dim = [get_centers(edge_dim) for edge_dim in self.grid_edges]
        mesh_centers_list = np.meshgrid(*centers_per_dim, indexing="ij", sparse=False)
        full_grid_bin_centers = np.stack(
            [c.ravel() for c in mesh_centers_list],
            axis=-1,
        )

        self.bin_size_ = np.array(
            [np.diff(edge_dim)[0] for edge_dim in self.grid_edges],
            dtype=np.float64,
        )

        self.dimension_ranges = tuple(
            (edge_dim[0], edge_dim[-1]) for edge_dim in self.grid_edges
        )
        self.bin_centers = full_grid_bin_centers[self.active_mask.ravel()]

        self.connectivity = _create_regular_grid_connectivity_graph(
            full_grid_bin_centers=full_grid_bin_centers,
            active_mask_nd=self.active_mask,
            grid_shape=self.grid_shape,
            connect_diagonal=connect_diagonal_neighbors,
        )

        # Validate connectivity graph has required attributes
        validate_connectivity_graph(
            self.connectivity, n_dims=len(self.dimension_ranges)
        )
