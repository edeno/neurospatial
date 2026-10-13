"""Protocol definitions for Environment mixins.

This module defines Protocol classes that specify the interface mixins expect
from the Environment class. Using Protocols lets mypy understand the mixin
pattern without ``type: ignore`` comments or disabled error codes.

Docstrings here are intentionally one-line summaries. The full parameter and
return documentation lives on the concrete ``Environment`` (and sibling)
methods that users actually read; keeping this typing-only file to bare
signatures avoids the docstring drift that duplication invites.

Index-dtype convention: point/region *index* arrays (``bin_at``,
``bins_in_region``, ``boundary_bins``, ``point_to_bin_index``,
``trajectory._allocate_time_linear``) are ``np.intp`` -- pointer-sized (64-bit on every
platform, including Windows where ``np.int_`` is 32-bit) and the dtype NumPy
fancy-indexing returns. Trajectory *sequence* arrays (``bin_sequence``,
``BinSequenceWithRuns.bins``) stay compact ``np.int32`` on purpose, and the
``run_starts`` / ``run_lengths`` offsets are ``np.int64``. The split is
intentional -- portability for indices, memory for long sequences -- so do not
collapse them to one dtype.

See: https://mypy.readthedocs.io/en/latest/more_types.html#mixin-classes
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, TypeVar

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

if TYPE_CHECKING:
    from scipy.spatial import cKDTree

    from neurospatial.animation.config import ScaleBarConfig
    from neurospatial.animation.overlays import OverlayProtocol
    from neurospatial.environment.trajectory import BinSequenceWithRuns
    from neurospatial.layout._diffusion import DiffusionGeometry, FaceGeometry
    from neurospatial.layout.base import LayoutEngine
    from neurospatial.ops._types import (
        DiffusionEigenbasisCache,
        KernelMode,
        MRFEigenbasisCache,
    )
    from neurospatial.ops.diffusion import MRFBasis
    from neurospatial.regions import Region, Regions


class EnvironmentProtocol(Protocol):
    """Interface that Cartesian and polar environments provide to mixins.

    Mixins annotate ``self`` as ``EnvironmentProtocol`` (or the ``SelfEnv``
    TypeVar) instead of ``Environment`` to avoid "erased type" errors in mypy.
    ``tests/typing/environment_protocol.py`` checks both concrete types.
    Read-only members are properties so cached descriptors match the protocol.
    """

    # Core attributes
    name: str
    layout: LayoutEngine
    regions: Regions
    units: str | None
    frame: str | None
    # Per-class marker: ``False`` for Cartesian environments, ``True`` for
    # egocentric polar environments (EgocentricPolarEnvironment). Mixins that
    # adapt presentation to polar geometry (e.g. plot_field axis labels) read
    # this; it is a class constant, not a runtime-mutable flag.
    _POLAR: ClassVar[bool]

    # Internal/cache attributes that mixins access
    _kernel_cache: dict[tuple[float, KernelMode], NDArray[np.float64]]

    # Finite-volume geometry + growable truncated-eigenbasis caches for
    # env.diffuse (both versioned_cached_property, dropped on _state_version).
    @property
    def _diffusion_geometry(self) -> DiffusionGeometry: ...

    @property
    def _face_geometry(self) -> FaceGeometry: ...

    @property
    def _diffusion_eigenbasis(self) -> DiffusionEigenbasisCache: ...
    # Growable global eigenbasis holder for the MRF penalty-basis resolver
    # (versioned_cached_property; geometry-keyed, dropped on _state_version).
    @property
    def _mrf_eigenbasis(self) -> MRFEigenbasisCache: ...

    _layout_type_used: str | None
    _layout_params_used: dict[str, Any]
    _is_linearized_track_env: bool
    _kdtree_cache: cKDTree | None

    def _require_face_geometry(self, operation: str) -> FaceGeometry:
        """Cached cell faces, or an error naming ``operation`` if unsupported."""
        ...

    # Computed properties
    @property
    def n_bins(self) -> int:
        """Number of spatial bins in the environment."""
        ...

    @property
    def n_dims(self) -> int:
        """Number of spatial dimensions (1, 2, or 3)."""
        ...

    @property
    def is_linearized_track(self) -> bool:
        """Whether this is a 1D (linearized) environment."""
        ...

    @property
    def bin_sizes(self) -> NDArray[np.float64]:
        """Per-bin cell volume; the canonical mass ``M``, shape ``(n_bins,)``."""
        ...

    @property
    def layout_type(self) -> str | None:
        """Type of layout engine used to create this environment, or None."""
        ...

    @property
    def layout_parameters(self) -> dict[str, Any] | None:
        """Parameters used to create the layout, or None."""
        ...

    def get_differential_operator(self) -> sparse.csc_matrix:
        """Build (or fetch a cached) edge-oriented differential operator ``D``."""
        ...

    def get_divergence_operator(self) -> sparse.csc_matrix:
        """Fetch the cached finite-volume divergence operator."""
        ...

    @property
    def boundary_bins(self) -> NDArray[np.intp]:
        """Integer indices of bins at the environment boundary."""
        ...

    # Methods that mixins call
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

    def bin_sequence_with_runs(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        outside_value: int | None = -1,
        max_gap: float | None = 0.5,
        epochs: Any = None,
    ) -> BinSequenceWithRuns:
        """Convert a trajectory to a bin sequence plus per-run boundaries."""
        ...

    def bins_in_region(self, region_name: str) -> NDArray[np.intp]:
        """Get integer bin indices within a named region."""
        ...

    def region_mask(
        self,
        regions: str | list[str] | Region | Regions,
        *,
        include_boundary: bool = True,
    ) -> NDArray[np.bool_]:
        """Boolean mask (shape ``(n_bins,)``) for one or more regions."""
        ...

    def region_membership(
        self,
        regions: Regions | None = None,
        *,
        include_boundary: bool = True,
    ) -> NDArray[np.bool_]:
        """Boolean membership mask for all regions, shape ``(n_bins, n_regions)``."""
        ...

    def compute_kernel(
        self,
        bandwidth: float,
        *,
        mode: KernelMode = "density",
        cache: bool = True,
    ) -> NDArray[np.float64]:
        """Compute a smoothing kernel matrix, shape ``(n_bins, n_bins)``."""
        ...

    def smooth(
        self,
        field: NDArray[np.float64],
        bandwidth: float,
        *,
        mode: KernelMode = "density",
    ) -> NDArray[np.float64]:
        """Apply graph-based smoothing to a spatial field."""
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

    def _mrf_basis(
        self,
        occupancy: NDArray[np.float64],
        *,
        rank: int | None,
    ) -> MRFBasis:
        """Reduced-rank MRF penalty basis for the penalized-Poisson GAM."""
        ...

    def occupancy(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        speed: NDArray[np.float64] | None = None,
        min_speed: float | None = None,
        max_gap: float | None = 0.5,
        bandwidth: float | None = None,
        time_allocation: Literal["start", "linear"] = "start",
        return_seconds: bool = True,
    ) -> NDArray[np.float64]:
        """Compute spatial occupancy (time per bin) from position data."""
        ...

    def distance_between(
        self,
        point1: NDArray[np.float64],
        point2: NDArray[np.float64],
        edge_weight: str = "distance",
    ) -> float:
        """Compute graph (shortest-path) distance between two points."""
        ...

    # Published geometry: read-only views, replaced whole by _setup_from_layout.
    @property
    def bin_centers(self) -> NDArray[np.float64]: ...

    @property
    def connectivity(self) -> nx.Graph: ...

    @property
    def dimension_ranges(self) -> Sequence[tuple[float, float]] | None: ...

    @property
    def active_mask(self) -> NDArray[np.bool_] | None: ...

    @property
    def grid_shape(self) -> tuple[int, ...] | None: ...

    @property
    def grid_edges(self) -> tuple[NDArray[np.float64], ...] | None: ...

    # Linearization methods (1D environments)
    def to_linear(self, nd_position: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert N-D positions to 1D linear positions (1D environments only)."""
        ...

    def linear_to_nd(self, linear_position: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert 1D linear positions to N-D coordinates (1D environments only)."""
        ...

    # Visualization and animation methods
    def animate_fields(
        self,
        fields: Sequence[NDArray[np.float64]] | NDArray[np.float64],
        *,
        frame_times: NDArray[np.float64],
        backend: Literal["auto", "napari", "video", "html", "widget"] = "auto",
        save_path: str | None = None,
        speed: float = 1.0,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        frame_labels: Sequence[str] | None = None,
        overlay_trajectory: NDArray[np.float64] | None = None,
        title: str = "Spatial Field Animation",
        dpi: int = 100,
        codec: str = "h264",
        bitrate: int = 5000,
        n_workers: int | None = None,
        dry_run: bool = False,
        image_format: Literal["png", "jpeg"] = "png",
        max_html_frames: int = 500,
        contrast_limits: tuple[float, float] | None = None,
        show_colorbar: bool = False,
        colorbar_label: str = "",
        overlays: list[OverlayProtocol] | None = None,
        show_regions: bool | list[str] = False,
        region_alpha: float = 0.3,
        scale_bar: bool | ScaleBarConfig = False,
        **kwargs: Any,
    ) -> Any:
        """Animate spatial fields over time (backend-specific return value)."""
        ...


class TrajectoryBinningConsumer(Protocol):
    """Point lookup, bin counts and layout for binning a trajectory."""

    @property
    def n_bins(self) -> int: ...

    @property
    def n_dims(self) -> int: ...

    @property
    def layout(self) -> LayoutEngine: ...

    @property
    def dimension_ranges(self) -> Sequence[tuple[float, float]] | None: ...

    def bin_at(self, points_nd: NDArray[np.float64]) -> NDArray[np.intp]: ...

    def compute_kernel(
        self, bandwidth: float, *, mode: KernelMode = "density", cache: bool = True
    ) -> NDArray[np.float64]: ...


class TransitionsConsumer(Protocol):
    """Graph, bin count, kernel and trajectory binning for transition matrices."""

    @property
    def n_bins(self) -> int: ...

    @property
    def connectivity(self) -> nx.Graph: ...

    def compute_kernel(
        self, bandwidth: float, *, mode: KernelMode = "density", cache: bool = True
    ) -> NDArray[np.float64]: ...

    def bin_sequence(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        dedup: bool = True,
        outside_value: int | None = -1,
        max_gap: float | None = 0.5,
        epochs: Any = None,
    ) -> NDArray[np.int32]: ...


class RebinConsumer(Protocol):
    """Layout, connectivity and metadata for coarsening a regular grid."""

    @property
    def connectivity(self) -> nx.Graph: ...

    @property
    def layout(self) -> LayoutEngine: ...

    @property
    def name(self) -> str: ...

    @property
    def frame(self) -> str | None: ...

    @property
    def units(self) -> str | None: ...


class SubsetConsumer(Protocol):
    """Published geometry, regions and metadata for selecting spatial bins."""

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
    def bin_centers(self) -> NDArray[np.float64]: ...

    @property
    def n_bins(self) -> int: ...

    @property
    def layout(self) -> LayoutEngine: ...

    _kdtree_cache: cKDTree | None


# TypeVar bound to EnvironmentProtocol for use in mixin ``self`` annotations.
# Lets mixins access Protocol attributes while still accepting concrete
# Environment instances without "Invalid self argument" errors.
SelfEnv = TypeVar("SelfEnv", bound=EnvironmentProtocol)
