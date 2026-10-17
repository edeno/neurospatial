"""The environment members that public ops functions read at their boundary.

A public ops function that takes an environment reads the components it
needs once, through these members, and computes from those components; it
does not call back into environment analysis methods. Each base protocol
names one to three members; the composite protocols only combine bases.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import networkx as nx
    import numpy as np
    from matplotlib.axes import Axes
    from numpy.typing import NDArray
    from scipy import sparse
    from scipy.spatial import cKDTree

    from neurospatial.geometry import (
        Arena,
        Binning,
        CartesianSystem,
        DiffusionGeometry,
        EgocentricPolarSystem,
    )
    from neurospatial.ops._types import KernelMode
    from neurospatial.ops.mapping import TieBreakStrategy
    from neurospatial.regions import Regions


# ---------------------------------------------------------------------------
# Base protocols: one to three members each.
# ---------------------------------------------------------------------------
class CoordinateConsumer(Protocol):
    """The coordinate descriptor, for functions with a Cartesian-only mode."""

    @property
    def coordinate_system(self) -> CartesianSystem | EgocentricPolarSystem: ...


class CartesianConsumer(CoordinateConsumer, Protocol):
    """An environment whose bin centers are Cartesian positions."""

    @property
    def coordinate_system(self) -> CartesianSystem: ...


class BinCentersConsumer(Protocol):
    """Bin centers, in the environment's coordinates."""

    @property
    def bin_centers(self) -> NDArray[np.float64]: ...


class BinsConsumer(BinCentersConsumer, Protocol):
    """Bin count, dimensionality and centers."""

    @property
    def n_bins(self) -> int: ...

    @property
    def n_dims(self) -> int: ...


class GraphConsumer(Protocol):
    """Bin count and navigation graph."""

    @property
    def n_bins(self) -> int: ...

    @property
    def connectivity(self) -> nx.Graph: ...


class BinningConsumer(Protocol):
    """The bins with their point lookup, and the continuous domain."""

    @property
    def binning(self) -> Binning: ...

    @property
    def arena(self) -> Arena: ...


class DiffusionGeometryConsumer(Protocol):
    """The environment's cached conductances and cell measures."""

    def get_diffusion_geometry(self) -> DiffusionGeometry: ...


class GradientConsumer(Protocol):
    """Bin count and cached gradient operator for applying a scalar field."""

    @property
    def n_bins(self) -> int: ...

    def get_differential_operator(self) -> sparse.csc_matrix: ...


class DivergenceConsumer(Protocol):
    """Edge count and cached divergence operator for applying an edge field."""

    @property
    def connectivity(self) -> nx.Graph: ...

    def get_divergence_operator(self) -> sparse.csc_matrix: ...


class FieldPlotConsumer(Protocol):
    """Draw one value per bin on an Axes, through the environment's renderer."""

    def plot_field(
        self,
        field: NDArray[np.float64],
        ax: Axes | None = ...,
        *,
        cmap: str = ...,
        colorbar: bool = ...,
    ) -> Axes: ...


# ---------------------------------------------------------------------------
# Composite protocols: named combinations of the bases.
# ---------------------------------------------------------------------------
class CartesianBinsConsumer(CartesianConsumer, BinsConsumer, Protocol):
    """Bin count, dimensionality and Cartesian centers."""


class RegionMaskConsumer(BinningConsumer, Protocol):
    """Bins, their domain and the named regions for region membership masks."""

    @property
    def regions(self) -> Regions: ...


class GraphLookupConsumer(BinningConsumer, CoordinateConsumer, Protocol):
    """Point lookup, coordinate meaning and graph for geodesic distances."""

    @property
    def connectivity(self) -> nx.Graph: ...


class BasisCentersConsumer(GraphConsumer, BinsConsumer, CoordinateConsumer, Protocol):
    """Centers, graph and coordinate meaning for choosing basis centers.

    ``bin_center_tree`` is the environment's cached KD-tree over its centers.
    """

    def bin_center_tree(self) -> cKDTree: ...


class SpatialBasisConsumer(CartesianConsumer, BasisCentersConsumer, Protocol):
    """Cartesian bins with extents for sizing a default spatial basis."""

    @property
    def dimension_ranges(self) -> Sequence[tuple[float, float]] | None: ...

    @property
    def is_linearized_track(self) -> bool: ...

    @property
    def bin_sizes(self) -> NDArray[np.float64]: ...


class WaveletBasisConsumer(BasisCentersConsumer, DiffusionGeometryConsumer, Protocol):
    """Diffusion geometry, cell faces and graph for heat-kernel wavelets."""

    @property
    def binning(self) -> Binning: ...


class ProbabilityTargetConsumer(CartesianBinsConsumer, Protocol):
    """Cartesian bins that receive probabilities from their nearest centers.

    ``bin_center_tree`` is read once at the public boundary of
    ``map_probabilities``: the target environment caches the KD-tree.
    """

    def bin_center_tree(self) -> cKDTree: ...


class ResampleSourceConsumer(CartesianBinsConsumer, Protocol):
    """Cartesian bins whose field is pulled back by nearest bin center.

    ``nearest_bin`` is read once at the public boundary of ``resample_field``:
    the source environment caches the KD-tree it searches.
    """

    def nearest_bin(
        self, positions: NDArray[np.float64], *, tie_break: TieBreakStrategy = ...
    ) -> NDArray[np.intp] | tuple[NDArray[np.intp], NDArray[np.float64]]: ...


class ResampleTargetConsumer(
    CartesianConsumer, BinCentersConsumer, DiffusionGeometryConsumer, Protocol
):
    """Cartesian bins that receive a resampled field and may diffuse it.

    ``diffuse`` is read once at the public boundary of ``resample_field``: the
    destination environment caches the eigenbasis it smooths with.
    """

    def diffuse(
        self, fields: Any, *, bandwidth: float, mode: KernelMode = ...
    ) -> NDArray[np.float64]: ...
