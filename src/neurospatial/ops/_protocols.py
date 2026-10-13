"""Small environment capabilities consumed by finite-volume operators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import networkx as nx
    import numpy as np
    from numpy.typing import NDArray
    from scipy import sparse

    from neurospatial.layout._diffusion import DiffusionGeometry, FaceGeometry
    from neurospatial.ops._types import DiffusionEigenbasisCache


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
    """Cell faces and the cached gradient operator for finite-volume calculus."""

    @property
    def n_bins(self) -> int: ...

    def _require_face_geometry(self, operation: str) -> FaceGeometry: ...

    def get_differential_operator(self) -> sparse.csc_matrix: ...


class DivergenceConsumer(Protocol):
    """Edge count and cached divergence operator for applying an edge field."""

    @property
    def connectivity(self) -> nx.Graph: ...

    @property
    def _divergence_operator_cached(self) -> sparse.csc_matrix: ...


class WaveletGeometryConsumer(DiffusionGeometryConsumer, Protocol):
    """Cell geometry and coordinates for a physical diffusion-wavelet generator."""

    @property
    def bin_centers(self) -> NDArray[np.float64]: ...

    def _require_face_geometry(self, operation: str) -> FaceGeometry: ...
