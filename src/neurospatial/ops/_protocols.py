"""Small environment capabilities that ops functions read, named member by member."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    import networkx as nx
    import numpy as np
    from matplotlib.axes import Axes
    from numpy.typing import NDArray
    from scipy import sparse

    from neurospatial.geometry._arena import Arena
    from neurospatial.geometry._binning import Binning
    from neurospatial.geometry._coordinates import (
        CartesianSystem,
        EgocentricPolarSystem,
    )
    from neurospatial.geometry._diffusion import DiffusionGeometry
    from neurospatial.ops._types import KernelMode
    from neurospatial.ops.binning import TieBreakStrategy
    from neurospatial.regions import Regions


class CartesianConsumer(Protocol):
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


class CartesianBinsConsumer(BinsConsumer, CartesianConsumer, Protocol):
    """Bin count, dimensionality and Cartesian centers."""


class GraphConsumer(Protocol):
    """Bin count and navigation graph."""

    @property
    def n_bins(self) -> int: ...

    @property
    def connectivity(self) -> nx.Graph: ...


class BasisCentersConsumer(GraphConsumer, Protocol):
    """Centers and graph for choosing basis-function centers."""

    @property
    def coordinate_system(self) -> CartesianSystem | EgocentricPolarSystem: ...

    @property
    def n_dims(self) -> int: ...

    @property
    def bin_centers(self) -> NDArray[np.float64]: ...


class SpatialBasisConsumer(BasisCentersConsumer, Protocol):
    """Cartesian bins with extents for sizing a default spatial basis."""

    @property
    def coordinate_system(self) -> CartesianSystem: ...

    @property
    def dimension_ranges(self) -> Sequence[tuple[float, float]] | None: ...


class FieldPlotConsumer(Protocol):
    """Draw one value per bin on an Axes."""

    def plot_field(
        self,
        field: NDArray[np.float64],
        ax: Axes | None = ...,
        cmap: str = ...,
        *,
        colorbar: bool = ...,
    ) -> Axes: ...


class BinSpacingConsumer(BinCentersConsumer, Protocol):
    """Bin centers and measures for the typical spacing between bins."""

    @property
    def bin_sizes(self) -> NDArray[np.float64]: ...

    @property
    def is_linearized_track(self) -> bool: ...


class RegionMaskConsumer(Protocol):
    """Bins, their domain and named regions for region membership masks."""

    @property
    def binning(self) -> Binning: ...

    @property
    def arena(self) -> Arena: ...

    @property
    def regions(self) -> Regions: ...


class BinLookupConsumer(Protocol):
    """Point-to-bin containment and the navigation graph."""

    @property
    def connectivity(self) -> nx.Graph: ...

    def bin_at(self, points_nd: NDArray[np.float64]) -> NDArray[np.intp]: ...


class NearestBinConsumer(Protocol):
    """Nearest-bin assignment with environment-owned query caching."""

    def nearest_bin(
        self,
        points: NDArray[np.float64],
        *,
        tie_break: TieBreakStrategy | Literal["lowest_index", "closest_center"] = ...,
        return_dist: bool = False,
        max_distance: float | None = None,
        max_distance_factor: float | None = None,
    ) -> NDArray[np.int64] | tuple[NDArray[np.int64], NDArray[np.float64]]: ...


class CartesianNearestBinConsumer(NearestBinConsumer, CartesianConsumer, Protocol):
    """Nearest-bin assignment for points in Cartesian coordinates."""


class ResampleSourceConsumer(CartesianNearestBinConsumer, Protocol):
    """Cartesian bins whose field values are carried to another environment."""

    @property
    def n_bins(self) -> int: ...

    @property
    def bin_centers(self) -> NDArray[np.float64]: ...


class DiffusionGeometryConsumer(Protocol):
    """Access to an environment's cached conductances and cell measures."""

    def get_diffusion_geometry(self) -> DiffusionGeometry: ...


class ResampleTargetConsumer(DiffusionGeometryConsumer, CartesianConsumer, Protocol):
    """Cartesian bins that receive a resampled field and may diffuse it."""

    @property
    def bin_centers(self) -> NDArray[np.float64]: ...

    def diffuse(
        self, fields: Any, bandwidth: float, *, mode: KernelMode = ...
    ) -> NDArray[np.float64]: ...


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


class WaveletGeometryConsumer(DiffusionGeometryConsumer, Protocol):
    """Cell geometry and coordinates for a physical diffusion-wavelet generator."""

    @property
    def bin_centers(self) -> NDArray[np.float64]: ...

    @property
    def binning(self) -> Binning: ...


class BoundaryConsumer(Protocol):
    """Continuous domain and explicit cells for physical/analysis boundaries."""

    @property
    def arena(self) -> Arena: ...

    @property
    def binning(self) -> Binning: ...


class ObserverConsumer(BoundaryConsumer, Protocol):
    """Boundary geometry plus assignment for explicit analysis-mode observers."""

    def bin_at(self, points_nd: NDArray[np.float64]) -> NDArray[np.intp]: ...


class ViewshedConsumer(ObserverConsumer, Protocol):
    """Observer geometry plus the bins a viewshed reports."""

    @property
    def n_bins(self) -> int: ...

    @property
    def bin_centers(self) -> NDArray[np.float64]: ...


class WaveletBasisConsumer(WaveletGeometryConsumer, BasisCentersConsumer, Protocol):
    """Diffusion geometry and graph for heat-kernel wavelets and their centers."""
