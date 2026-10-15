"""Small environment capabilities consumed by finite-volume operators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    import networkx as nx
    import numpy as np
    from numpy.typing import NDArray
    from scipy import sparse

    from neurospatial.environment import Environment
    from neurospatial.geometry._arena import Arena
    from neurospatial.geometry._binning import Binning
    from neurospatial.geometry._diffusion import DiffusionGeometry, FaceGeometry
    from neurospatial.ops._types import DiffusionEigenbasisCache
    from neurospatial.ops.binning import TieBreakStrategy
    from neurospatial.ops.transforms import Affine2D, AffineND


class AffineTransformConsumer(Protocol):
    """Fitted status and the environment-owned affine construction adapter."""

    @property
    def _is_fitted(self) -> bool: ...

    def apply_transform(
        self, transform: AffineND | Affine2D, *, name: str | None = None
    ) -> Environment: ...


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


class CacheClearConsumer(Protocol):
    """Public control over which environment-owned analyses are discarded."""

    def clear_cache(
        self,
        *,
        kdtree: bool = True,
        kernels: bool = True,
        cached_properties: bool = True,
    ) -> None: ...


class DiffusionGeometryConsumer(Protocol):
    """Access to an environment's cached conductances and cell measures."""

    @property
    def _diffusion_geometry(self) -> DiffusionGeometry: ...


class DiffusionAccuracyConsumer(Protocol):
    """Bin count and resolved eigenbasis ranks for reporting apply accuracy."""

    @property
    def n_bins(self) -> int: ...

    @property
    def _diffusion_eigenbasis(self) -> DiffusionEigenbasisCache: ...


class FaceGeometryConsumer(Protocol):
    """Public access to an environment-owned differential operator."""

    @property
    def n_bins(self) -> int: ...

    def get_differential_operator(self) -> sparse.csc_matrix: ...


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

    def _require_face_geometry(self, operation: str) -> FaceGeometry: ...


class BoundaryConsumer(Protocol):
    """Continuous domain and explicit cells for physical/analysis boundaries."""

    @property
    def arena(self) -> Arena: ...

    @property
    def binning(self) -> Binning: ...


class ObserverConsumer(BoundaryConsumer, Protocol):
    """Boundary geometry plus assignment for explicit analysis-mode observers."""

    def bin_at(self, points_nd: NDArray[np.float64]) -> NDArray[np.intp]: ...
