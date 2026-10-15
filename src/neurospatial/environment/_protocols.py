"""Narrow structural contracts for domain functions consuming environment data.

Index-dtype convention: point/region *index* arrays (``bin_at``,
``bins_in_region``, ``boundary_bins``, ``point_to_bin_index``,
``trajectory._allocate_time_linear``) are ``np.intp`` -- pointer-sized (64-bit on every
platform, including Windows where ``np.int_`` is 32-bit) and the dtype NumPy
fancy-indexing returns. Trajectory *sequence* arrays (``bin_sequence``,
``BinSequenceWithRuns.bins``) stay compact ``np.int32`` on purpose, and the
``run_starts`` / ``run_lengths`` offsets are ``np.int64``. The split is
intentional -- portability for indices, memory for long sequences -- so do not
collapse them to one dtype.

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.geometry._coordinates import CellLengthConsumer as CellLengthConsumer

if TYPE_CHECKING:
    import pandas as pd
    from scipy.spatial import cKDTree

    from neurospatial.geometry._arena import Arena
    from neurospatial.geometry._binning import Binning
    from neurospatial.geometry._coordinates import CoordinateSystem
    from neurospatial.geometry._diffusion import DiffusionGeometry, FaceGeometry
    from neurospatial.ops._types import (
        DiffusionEigenbasisCache,
        KernelMode,
        MRFEigenbasisCache,
    )
    from neurospatial.regions import Region, Regions


class CartesianLookupConsumer(Protocol):
    """Cartesian lookup/trajectory operations exposed only after a guard."""

    def bin_at(self, points_nd: NDArray[np.float64]) -> NDArray[np.intp]:
        """Find bin indices for given points (-1 for points outside)."""
        ...

    def bin_sequence(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        dedup: bool = True,
        outside_value: int | None = -1,
        max_gap: float | None = 0.5,
        epochs: Any = None,
    ) -> NDArray[np.int32]:
        """Convert a trajectory to a sequence of bin indices."""
        ...


class TrajectoryBinningConsumer(Protocol):
    """Point lookup, bin counts and layout for binning a trajectory."""

    @property
    def _binning(self) -> Binning: ...

    @property
    def arena(self) -> Arena: ...

    @property
    def n_bins(self) -> int: ...

    @property
    def n_dims(self) -> int: ...

    @property
    def layout_type(self) -> str | None: ...

    @property
    def dimension_ranges(self) -> Sequence[tuple[float, float]] | None: ...

    def bin_at(self, points_nd: NDArray[np.float64]) -> NDArray[np.intp]: ...

    def compute_kernel(
        self, bandwidth: float, *, mode: KernelMode = "density", cache: bool = True
    ) -> NDArray[np.float64]: ...


class TransitionsConsumer(Protocol):
    """Graph, kernel and access to Cartesian trajectory binning for transitions."""

    @property
    def n_bins(self) -> int: ...

    @property
    def connectivity(self) -> nx.Graph: ...

    def compute_kernel(
        self, bandwidth: float, *, mode: KernelMode = "density", cache: bool = True
    ) -> NDArray[np.float64]: ...

    def _require_cartesian(self, operation: str) -> CartesianLookupConsumer:
        """This environment typed as Cartesian, or raise naming ``operation``."""
        ...


class RebinConsumer(Protocol):
    """Layout, arena and metadata for re-binning rectangular grids."""

    @property
    def _binning(self) -> Binning: ...

    @property
    def connectivity(self) -> nx.Graph: ...

    @property
    def bin_centers(self) -> NDArray[np.float64]: ...

    @property
    def arena(self) -> Arena: ...

    @property
    def name(self) -> str: ...

    @property
    def frame(self) -> str | None: ...

    @property
    def units(self) -> str | None: ...

    @property
    def regions(self) -> Regions: ...


class SubsetConsumer(Protocol):
    """Published geometry, regions and metadata for selecting spatial bins."""

    @property
    def _binning(self) -> Binning: ...

    @property
    def arena(self) -> Arena: ...

    @property
    def active_mask(self) -> NDArray[np.bool_] | None: ...

    @property
    def bin_centers(self) -> NDArray[np.float64]: ...

    @property
    def connectivity(self) -> nx.Graph: ...

    @property
    def grid_edges(self) -> tuple[NDArray[np.float64], ...] | None: ...

    @property
    def grid_shape(self) -> tuple[int, ...] | None: ...

    @property
    def is_linearized_track(self) -> bool: ...

    @property
    def n_bins(self) -> int: ...

    @property
    def regions(self) -> Regions: ...

    @property
    def frame(self) -> str | None: ...

    @property
    def units(self) -> str | None: ...

    @property
    def _layout_params_used(self) -> dict[str, Any]: ...


class InterpolationConsumer(Protocol):
    """Bin geometry and cached point lookup for interpolating a scalar field."""

    @property
    def _binning(self) -> Binning: ...

    def _locate(self, positions: NDArray[np.float64]) -> NDArray[np.intp]: ...

    @property
    def bin_centers(self) -> NDArray[np.float64]: ...

    @property
    def n_bins(self) -> int: ...

    @property
    def layout_type(self) -> str | None: ...

    _kdtree_cache: cKDTree | None


class QueryConsumer(Protocol):
    """Only the environment members read by queries operations."""

    @property
    def _bin_sizes_cached(self) -> NDArray[np.float64]: ...

    @property
    def _binning(self) -> Binning: ...

    def _require_cartesian(self, operation: str) -> CartesianLookupConsumer:
        """This environment typed as Cartesian, or raise naming ``operation``."""
        ...

    @property
    def bin_centers(self) -> NDArray[np.float64]: ...

    @property
    def connectivity(self) -> nx.Graph: ...

    @property
    def n_bins(self) -> int:
        """Number of spatial bins in the environment."""
        ...

    @property
    def n_dims(self) -> int:
        """Number of spatial dimensions (1, 2, or 3)."""
        ...

    def region_membership(
        self,
        regions: Regions | None = None,
        *,
        include_boundary: bool = True,
    ) -> NDArray[np.bool_]:
        """Boolean membership mask for all regions, shape ``(n_bins, n_regions)``."""
        ...

    @property
    def regions(self) -> Regions: ...


class RegionConsumer(Protocol):
    """Only the environment members read by regions operations."""

    def _require_cartesian(self, operation: str) -> CartesianLookupConsumer:
        """This environment typed as Cartesian, or raise naming ``operation``."""
        ...

    @property
    def bin_centers(self) -> NDArray[np.float64]: ...

    @property
    def n_bins(self) -> int:
        """Number of spatial bins in the environment."""
        ...

    @property
    def n_dims(self) -> int:
        """Number of spatial dimensions (1, 2, or 3)."""
        ...

    @property
    def regions(self) -> Regions: ...


class VisualizationConsumer(Protocol):
    """Only the environment members read by visualization operations."""

    @property
    def _is_fitted(self) -> bool: ...

    @property
    def _binning(self) -> Binning: ...

    @property
    def _coordinate_system(self) -> CoordinateSystem: ...

    _layout_type_used: str | None

    @property
    def active_mask(self) -> NDArray[np.bool_] | None: ...

    @property
    def bin_centers(self) -> NDArray[np.float64]: ...

    @property
    def connectivity(self) -> nx.Graph: ...

    @property
    def dimension_ranges(self) -> Sequence[tuple[float, float]] | None: ...

    @property
    def grid_edges(self) -> tuple[NDArray[np.float64], ...] | None: ...

    @property
    def is_linearized_track(self) -> bool:
        """Whether this is a 1D (linearized) environment."""
        ...

    @property
    def layout_type(self) -> str | None:
        """Type of layout engine used to create this environment, or None."""
        ...

    @property
    def n_bins(self) -> int:
        """Number of spatial bins in the environment."""
        ...

    @property
    def n_dims(self) -> int:
        """Number of spatial dimensions (1, 2, or 3)."""
        ...

    @property
    def name(self) -> str: ...

    @property
    def regions(self) -> Regions: ...

    @property
    def units(self) -> str | None: ...


class MetricsConsumer(Protocol):
    """Only the environment members read by metrics operations."""

    @property
    def _bin_attributes_cached(self) -> pd.DataFrame: ...

    @property
    def _binning(self) -> Binning: ...

    @property
    def _boundary_bins_cached(self) -> NDArray[np.intp]: ...

    @property
    def _edge_attributes_cached(self) -> pd.DataFrame: ...

    _is_linearized_track_env: bool

    _layout_type_used: str | None

    @property
    def _linearization_properties_cached(self) -> dict[str, Any]: ...

    @property
    def active_mask(self) -> NDArray[np.bool_] | None: ...

    @property
    def bin_centers(self) -> NDArray[np.float64]: ...

    @property
    def connectivity(self) -> nx.Graph: ...

    @property
    def grid_shape(self) -> tuple[int, ...] | None: ...


class FieldConsumer(Protocol):
    """Only the environment members read by fields operations."""

    @property
    def _binning(self) -> Binning: ...

    @property
    def _diffusion_eigenbasis(self) -> DiffusionEigenbasisCache: ...

    @property
    def _diffusion_geometry(self) -> DiffusionGeometry: ...

    @property
    def _face_geometry(self) -> FaceGeometry: ...

    _kernel_cache: dict[tuple[float, KernelMode], NDArray[np.float64]]

    @property
    def _mrf_eigenbasis(self) -> MRFEigenbasisCache: ...

    def _require_face_geometry(self, operation: str) -> FaceGeometry:
        """Cached cell faces, or an error naming ``operation`` if unsupported."""
        ...

    def diffuse(
        self,
        # ``Any`` is load-bearing: on ``backend="jax"`` diffuse accepts (and
        # returns) a JAX ``Array``, and jax is an optional dependency whose array
        # type is not importable here -- so ``NDArray | jax.Array`` cannot be
        # spelled. ``Any`` keeps the backend-polymorphic surface without breaking
        # jax-absent type checks.
        fields: Any,
        bandwidth: float,
        *,
        mode: KernelMode = "density",
        backend: Literal["numpy", "jax"] = "numpy",
    ) -> NDArray[np.float64]:
        """Matrix-free diffusion smoothing (no dense ``(n_bins, n_bins)`` kernel)."""
        ...

    @property
    def n_bins(self) -> int:
        """Number of spatial bins in the environment."""
        ...


class AnimationConsumer(Protocol):
    """Environment members read by animation/core.py."""

    @property
    def _is_fitted(self) -> bool: ...

    @property
    def n_bins(self) -> int:
        """Number of spatial bins in the environment."""
        ...


class FieldMetricsConsumer(Protocol):
    """Environment members read by encoding/_field_metrics.py."""

    @property
    def bin_centers(self) -> NDArray[np.float64]: ...

    @property
    def bin_sizes(self) -> NDArray[np.float64]:
        """Per-bin cell volume; the canonical mass ``M``, shape ``(n_bins,)``."""
        ...

    @property
    def connectivity(self) -> nx.Graph: ...

    @property
    def n_bins(self) -> int:
        """Number of spatial bins in the environment."""
        ...

    @property
    def n_dims(self) -> int:
        """Number of spatial dimensions (1, 2, or 3)."""
        ...


class BorderConsumer(FieldMetricsConsumer, Protocol):
    """Environment members read by encoding/border.py, encoding/_physical_border.py."""

    @property
    def binning(self) -> Binning: ...

    @property
    def arena(self) -> Arena: ...

    @property
    def bin_centers(self) -> NDArray[np.float64]: ...

    @property
    def boundary_bins(self) -> NDArray[np.intp]:
        """Integer indices of bins at the environment boundary."""
        ...

    @property
    def connectivity(self) -> nx.Graph: ...

    @property
    def n_bins(self) -> int:
        """Number of spatial bins in the environment."""
        ...

    def region_mask(
        self,
        regions: str | list[str] | Region | Regions,
        *,
        include_boundary: bool = True,
    ) -> NDArray[np.bool_]:
        """Boolean mask (shape ``(n_bins,)``) for one or more regions."""
        ...

    @property
    def regions(self) -> Regions: ...


class SmoothingConsumer(Protocol):
    """Environment members read by encoding/_smoothing.py."""

    @property
    def _diffusion_eigenbasis(self) -> DiffusionEigenbasisCache: ...

    @property
    def _diffusion_geometry(self) -> DiffusionGeometry: ...

    @property
    def _coordinate_system(self) -> CoordinateSystem: ...

    @property
    def _gaussian_kernels(self) -> dict[float, NDArray[np.float64]]: ...

    @property
    def bin_centers(self) -> NDArray[np.float64]: ...

    @property
    def connectivity(self) -> nx.Graph: ...

    def diffuse(
        self,
        # ``Any`` is load-bearing: on ``backend="jax"`` diffuse accepts (and
        # returns) a JAX ``Array``, and jax is an optional dependency whose array
        # type is not importable here -- so ``NDArray | jax.Array`` cannot be
        # spelled. ``Any`` keeps the backend-polymorphic surface without breaking
        # jax-absent type checks.
        fields: Any,
        bandwidth: float,
        *,
        mode: KernelMode = "density",
        backend: Literal["numpy", "jax"] = "numpy",
    ) -> NDArray[np.float64]:
        """Matrix-free diffusion smoothing (no dense ``(n_bins, n_bins)`` kernel)."""
        ...

    @property
    def dimension_ranges(self) -> Sequence[tuple[float, float]] | None: ...

    @property
    def n_bins(self) -> int:
        """Number of spatial bins in the environment."""
        ...

    @property
    def units(self) -> str | None: ...


class ClipConsumer(Protocol):
    """Environment members read by environment/_clipping.py."""

    @property
    def _binning(self) -> Binning: ...

    @property
    def arena(self) -> Arena: ...

    @property
    def frame(self) -> str | None: ...

    @property
    def n_bins(self) -> int:
        """Number of spatial bins in the environment."""
        ...

    @property
    def n_dims(self) -> int:
        """Number of spatial dimensions (1, 2, or 3)."""
        ...

    @property
    def name(self) -> str: ...

    @property
    def units(self) -> str | None: ...


class SummaryConsumer(Protocol):
    """Published metadata and geometry read only for text and HTML summaries."""

    @property
    def name(self) -> str: ...

    @property
    def _is_fitted(self) -> bool: ...

    @property
    def _layout_type_used(self) -> str | None: ...

    @property
    def layout_type(self) -> str | None: ...

    @property
    def n_dims(self) -> int: ...

    @property
    def n_bins(self) -> int: ...

    @property
    def bin_centers(self) -> NDArray[np.float64]: ...

    @property
    def dimension_ranges(self) -> Sequence[tuple[float, float]] | None: ...

    @property
    def bin_sizes(self) -> NDArray[np.float64]: ...

    @property
    def grid_edges(self) -> tuple[NDArray[np.float64], ...] | None: ...

    @property
    def regions(self) -> Regions: ...

    @property
    def is_linearized_track(self) -> bool: ...

    def info(self) -> str: ...
