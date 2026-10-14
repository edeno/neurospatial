"""Owned bin geometry and the capabilities used by environment operations.

Layout-backed adapters retain point lookup and finite-volume geometry behavior
while exposing their capabilities together with the published bin arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, cast

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial.layout._capabilities import (
    RectilinearGrid,
    RectilinearGridProvider,
    TrackLinearizationProvider,
)
from neurospatial.layout._diffusion import (
    FaceGeometry,
    FiniteVolumeGeometryProvider,
)
from neurospatial.layout.base import LayoutEngine


@dataclass(frozen=True, eq=False)
class SpatialBins:
    """Discrete bins with owned, read-only geometry arrays.

    Attributes
    ----------
    centers : ndarray of float64, shape (n_bins, n_dims)
        Finite bin centers in the environment's coordinates.
    sizes : ndarray of float64, shape (n_bins,)
        Finite, strictly positive bin measures in ``units ** measure_dim``.
    measure_dim : int
        Dimension of the bin measures: one for track lengths, otherwise the
        coordinate dimension for areas or volumes.
    pairs : ndarray of intp, shape (n_pairs, 2)
        Neighbor pairs in connectivity edge order and orientation.
    pair_distances : ndarray of float64, shape (n_pairs,)
        Finite, non-negative navigation distances, distinct from the flux
        distances used by finite-volume faces at track junctions.
    """

    centers: NDArray[np.float64]
    sizes: NDArray[np.float64]
    measure_dim: int
    pairs: NDArray[np.intp]
    pair_distances: NDArray[np.float64]

    def __post_init__(self) -> None:
        centers = np.array(self.centers, dtype=np.float64, copy=True)
        sizes = np.array(self.sizes, dtype=np.float64, copy=True)
        raw_pairs = np.asarray(self.pairs)
        distances = np.array(self.pair_distances, dtype=np.float64, copy=True)

        if centers.ndim != 2:
            raise ValueError(f"centers must be 2D, got shape {centers.shape}.")
        index: int | tuple[int, ...]
        invalid = ~np.isfinite(centers)
        if np.any(invalid):
            index = tuple(int(i) for i in np.argwhere(invalid)[0])
            raise ValueError(f"centers{index} must be finite.")
        n_bins, n_dims = centers.shape
        if sizes.shape != (n_bins,):
            raise ValueError(f"sizes must have shape ({n_bins},), got {sizes.shape}.")
        invalid = ~np.isfinite(sizes) | (sizes <= 0.0)
        if np.any(invalid):
            index = int(np.flatnonzero(invalid)[0])
            raise ValueError(f"sizes[{index}] must be finite and strictly positive.")
        if (
            isinstance(self.measure_dim, (bool, np.bool_))
            or not isinstance(self.measure_dim, (int, np.integer))
            or not 1 <= self.measure_dim <= n_dims
        ):
            raise ValueError(
                f"measure_dim must be an integer between 1 and {n_dims}, "
                f"got {self.measure_dim!r}."
            )
        if raw_pairs.ndim != 2 or raw_pairs.shape[1] != 2:
            raise ValueError(
                f"pairs must have shape (n_pairs, 2), got {raw_pairs.shape}."
            )
        if raw_pairs.size and raw_pairs.dtype.kind not in "iu":
            if raw_pairs.dtype.kind == "f":
                invalid = ~np.isfinite(raw_pairs) | (raw_pairs != np.floor(raw_pairs))
                if np.any(invalid):
                    index = tuple(int(i) for i in np.argwhere(invalid)[0])
                    raise ValueError(f"pairs{index} must contain integer bin indices.")
            else:
                raise ValueError("pairs(0, 0) must contain integer bin indices.")
        invalid = (raw_pairs < 0) | (raw_pairs >= n_bins)
        if np.any(invalid):
            index = tuple(int(i) for i in np.argwhere(invalid)[0])
            raise ValueError(f"pairs{index} must be in [0, {n_bins}).")
        pairs = np.array(raw_pairs, dtype=np.intp, copy=True)
        self_pairs = pairs[:, 0] == pairs[:, 1]
        if np.any(self_pairs):
            index = int(np.flatnonzero(self_pairs)[0])
            raise ValueError(f"pairs[{index}] is a self pair.")
        _, first_indices = np.unique(np.sort(pairs, axis=1), axis=0, return_index=True)
        duplicates = np.ones(len(pairs), dtype=bool)
        duplicates[first_indices] = False
        if np.any(duplicates):
            index = int(np.flatnonzero(duplicates)[0])
            raise ValueError(f"pairs[{index}] duplicates an earlier unordered pair.")
        if distances.shape != (len(pairs),):
            raise ValueError(
                f"pair_distances must have shape ({len(pairs)},), "
                f"got {distances.shape}."
            )
        invalid = ~np.isfinite(distances) | (distances < 0.0)
        if np.any(invalid):
            index = int(np.flatnonzero(invalid)[0])
            raise ValueError(
                f"pair_distances[{index}] must be finite and non-negative."
            )

        for name, array in (
            ("centers", centers),
            ("sizes", sizes),
            ("pairs", pairs),
            ("pair_distances", distances),
        ):
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore array write protection after pickle or deepcopy."""
        for name, value in state.items():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
            object.__setattr__(self, name, value)

    @classmethod
    def from_arrays(
        cls,
        centers: ArrayLike,
        sizes: ArrayLike,
        measure_dim: int,
        pairs: ArrayLike,
        pair_distances: ArrayLike,
    ) -> SpatialBins:
        """Validate and own bin centers, measures and ordered neighbor pairs.

        Parameters
        ----------
        centers : array-like, shape (n_bins, n_dims)
            Finite bin coordinates.
        sizes : array-like, shape (n_bins,)
            Finite, strictly positive bin measures.
        measure_dim : int
            Measure dimension in ``[1, n_dims]``.
        pairs : array-like of int, shape (n_pairs, 2)
            Distinct unordered neighbor pairs, kept in their input order.
        pair_distances : array-like, shape (n_pairs,)
            Finite, non-negative navigation distances.

        Returns
        -------
        SpatialBins
            Validated independent snapshot with read-only arrays.

        Raises
        ------
        ValueError
            If shapes, dimensions or values are invalid. Elementwise errors
            identify the offending array index.
        """
        return cls(
            np.asarray(centers),
            np.asarray(sizes),
            measure_dim,
            np.asarray(pairs),
            np.asarray(pair_distances),
        )

    @property
    def n_bins(self) -> int:
        """Number of active bins."""
        return int(self.centers.shape[0])

    @property
    def n_dims(self) -> int:
        """Number of coordinate columns."""
        return int(self.centers.shape[1])


class Locator(Protocol):
    """Find the bin that contains each query point."""

    def locate(self, points: NDArray[np.float64]) -> NDArray[np.intp]:
        """Return bin indices for ``(n_points, n_dims)`` coordinates, or -1.

        Non-finite rows return -1; a wrong number of columns raises ValueError.
        The input is not modified.
        """
        ...


@dataclass(frozen=True, eq=False)
class LayoutLocator:
    """Point lookup delegated to a built layout."""

    layout: LayoutEngine

    def locate(self, points: NDArray[np.float64]) -> NDArray[np.intp]:
        """Return the layout's bin indices with native integer dtype."""
        return self.layout.point_to_bin_index(points).astype(np.intp, copy=False)


class FacesSource(Protocol):
    """Source of an environment's finite-volume cell faces."""

    @property
    def available(self) -> bool:
        """Whether validated finite-volume faces can be built."""
        ...

    def build(self) -> FaceGeometry:
        """Build validated faces in connectivity pair order, without caching."""
        ...

    def missing_error(self, operation: str) -> Exception:
        """Return an actionable error when the named operation needs faces."""
        ...


def _missing_geometry_error(layout: object, operation: str) -> NotImplementedError:
    """Explain which operation needs cell geometry that ``layout`` cannot supply.

    Parameters
    ----------
    layout : object
        The environment's layout engine, which lacks ``finite_volume_geometry``.
    operation : str
        Name of the analysis the scientist called, used to start the message.

    Returns
    -------
    NotImplementedError
        Error with What / Why / Fix sections, ready to raise.
    """
    from neurospatial._exceptions import _format_error

    explain = getattr(layout, "_explain_missing_geometry", None)
    if explain is not None:
        return NotImplementedError(explain(operation))
    return NotImplementedError(
        _format_error(
            f"{operation} needs finite-volume cell geometry, which layout "
            f"{type(layout).__name__!r} does not provide.",
            why=(
                "Why: layouts built by Environment factories supply cell faces "
                f"and measures; {type(layout).__name__!r} does not."
            ),
            fix=(
                "build the environment with a factory method, e.g. "
                "Environment.from_samples(positions, bin_size=...)"
            ),
        )
    )


@dataclass(frozen=True, eq=False)
class LayoutFaces:
    """Finite-volume faces delegated to a layout, checked against connectivity."""

    layout: LayoutEngine
    connectivity: nx.Graph
    _available: bool = field(init=False, repr=False)
    _n_bins: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_available", isinstance(self.layout, FiniteVolumeGeometryProvider)
        )
        object.__setattr__(self, "_n_bins", self.layout.bin_centers.shape[0])

    @property
    def available(self) -> bool:
        """Whether the layout supplied finite-volume geometry at publication."""
        return self._available

    def build(self) -> FaceGeometry:
        """Build validated cell faces in the published connectivity order."""
        layout = cast("FiniteVolumeGeometryProvider", self.layout)
        graph, volumes = layout.finite_volume_geometry()
        # Gradient and divergence index edge fields by env.connectivity's edge
        # order, so the layout's graph must be that graph, edge for edge.
        n_bins = self._n_bins
        if graph.number_of_nodes() != n_bins or list(graph.edges()) != list(
            self.connectivity.edges()
        ):
            raise ValueError(
                f"{type(layout).__name__}.finite_volume_geometry() returned a graph "
                f"with {graph.number_of_nodes()} bins and {graph.number_of_edges()} "
                f"edges that does not match the environment's {n_bins} bins "
                f"and {self.connectivity.number_of_edges()} connectivity edges in "
                "order; the layout's graph is out of sync with env.connectivity."
            )
        return FaceGeometry.from_graph(graph, volumes)

    def missing_error(self, operation: str) -> NotImplementedError:
        """Explain why the named operation cannot obtain cell geometry."""
        return _missing_geometry_error(self.layout, operation)


@dataclass(frozen=True, eq=False)
class Binning:
    """Published bins, point lookup, faces and optional structural components."""

    kind: str
    bins: SpatialBins
    locator: Locator
    faces: FacesSource
    rectilinear: RectilinearGrid | None = None
    linearization: TrackLinearizationProvider | None = None
    shapes: None = None
    outline: None = None


def binning_from_layout(layout: LayoutEngine, connectivity: nx.Graph) -> Binning:
    """Snapshot a built layout's bins and validated connectivity capabilities.

    Parameters
    ----------
    layout : LayoutEngine
        Built layout supplying bin coordinates, measures and point lookup.
    connectivity : networkx.Graph
        Validated environment connectivity in its published edge order.

    Returns
    -------
    Binning
        Owned bin arrays and adapters retaining the layout's capabilities.
    """
    edges = list(connectivity.edges(data=True))
    pairs = np.array([(u, v) for u, v, _ in edges], dtype=np.intp).reshape(-1, 2)
    distances = np.array([data["distance"] for _, _, data in edges], dtype=np.float64)
    centers = np.asarray(layout.bin_centers, dtype=np.float64)
    bins = SpatialBins.from_arrays(
        centers=centers,
        sizes=layout.bin_sizes(),
        measure_dim=1 if layout.is_linearized_track else centers.shape[1],
        pairs=pairs,
        pair_distances=distances,
    )
    return Binning(
        kind=getattr(layout, "_layout_type_tag", type(layout).__name__),
        bins=bins,
        locator=LayoutLocator(layout),
        faces=LayoutFaces(layout, connectivity),
        rectilinear=(
            layout.rectilinear_grid()
            if isinstance(layout, RectilinearGridProvider)
            else None
        ),
        linearization=(
            layout if isinstance(layout, TrackLinearizationProvider) else None
        ),
    )
