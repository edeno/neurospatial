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
``_allocate_time_linear``) are ``np.intp`` -- pointer-sized (64-bit on every
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
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple, Protocol, TypeVar

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

if TYPE_CHECKING:
    from scipy.spatial import cKDTree

    from neurospatial.animation.config import ScaleBarConfig
    from neurospatial.animation.overlays import OverlayProtocol
    from neurospatial.environment.trajectory import BinSequenceWithRuns
    from neurospatial.layout.base import LayoutEngine
    from neurospatial.ops._types import KernelMode
    from neurospatial.regions import Region, Regions


class DiffusionGeometry(NamedTuple):
    """Finite-volume geometry cached by ``_diffusion_geometry``.

    Built once and dropped wholesale on any ``_state_version`` bump; shared by
    ``diffuse``'s eigenbasis build and the smoothing consumers' W-component
    support gates.
    """

    W: sparse.csr_matrix
    volumes: NDArray[np.float64]
    n_components: int
    labels: NDArray[np.int_]


class EnvironmentProtocol(Protocol):
    """Protocol defining the interface that Environment provides to mixins.

    Mixins annotate ``self`` as ``EnvironmentProtocol`` (or the ``SelfEnv``
    TypeVar) instead of ``Environment`` to avoid "erased type" errors in mypy.
    """

    # Core attributes
    name: str
    layout: LayoutEngine
    bin_centers: NDArray[np.float64]
    connectivity: nx.Graph
    dimension_ranges: tuple[tuple[float, float], ...]
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
    _diffusion_geometry: DiffusionGeometry
    _diffusion_eigenbasis: dict[str, Any]
    _layout_type_used: str | None
    _layout_params_used: dict[str, Any]
    _is_linearized_track_env: bool
    _kdtree_cache: cKDTree | None

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
    ) -> NDArray[np.int32]:
        """Convert a trajectory to a sequence of bin indices."""
        ...

    def bin_sequence_with_runs(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        outside_value: int | None = -1,
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

    # Layout-specific attributes and methods (may be None if not applicable)
    active_mask: NDArray[np.bool_] | None
    grid_shape: tuple[int, ...] | None
    grid_edges: tuple[NDArray[np.float64], ...] | None
    linear_bin_centers_1d: NDArray[np.float64] | None
    linear_bin_edges: NDArray[np.float64] | None
    track_graph: nx.Graph | None

    # Linearization methods (1D environments)
    def to_linear(self, nd_position: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert N-D positions to 1D linear positions (1D environments only)."""
        ...

    def linear_to_nd(self, linear_position: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert 1D linear positions to N-D coordinates (1D environments only)."""
        ...

    # Private methods that mixins call (trajectory analysis)
    def _allocate_time_linear(
        self,
        positions: NDArray[np.float64],
        dt: NDArray[np.float64],
        valid_mask: NDArray[np.bool_],
        bin_indices: NDArray[np.intp],
        return_seconds: bool,
    ) -> NDArray[np.float64]:
        """Allocate time linearly across trajectory segments."""
        ...

    def _empirical_transitions(
        self,
        bins: NDArray[np.int32] | None = None,
        *,
        times: NDArray[np.float64] | None = None,
        positions: NDArray[np.float64] | None = None,
        lag: int = 1,
        normalize: bool = True,
        allow_teleports: bool = False,
    ) -> sparse.csr_matrix:
        """Compute empirical transition matrix from trajectory data."""
        ...

    def _random_walk_transitions(
        self,
        *,
        normalize: bool = True,
    ) -> sparse.csr_matrix:
        """Compute random walk transition matrix from connectivity."""
        ...

    def _diffusion_transitions(
        self,
        bandwidth: float,
        *,
        normalize: bool = True,
    ) -> sparse.csr_matrix:
        """Compute diffusion-based transition matrix."""
        ...

    def _compute_ray_grid_intersections(
        self,
        start_pos: NDArray[np.float64],
        end_pos: NDArray[np.float64],
        grid_edges: Sequence[NDArray[np.float64]],
        grid_shape: tuple[int, ...],
        total_time: float,
    ) -> list[tuple[int, float]]:
        """Compute grid cell intersections along a ray."""
        ...

    def _position_to_flat_index(
        self,
        pos: NDArray[np.float64],
        grid_edges: Sequence[NDArray[np.float64]],
        grid_shape: tuple[int, ...],
    ) -> int:
        """Convert a position to a flat grid index."""
        ...

    # Private methods that mixins call (field interpolation)
    def _interpolate_nearest(
        self,
        field: NDArray[np.float64],
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Interpolate field values using nearest-neighbor."""
        ...

    def _interpolate_linear(
        self,
        field: NDArray[np.float64],
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Interpolate field values using linear interpolation."""
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


# TypeVar bound to EnvironmentProtocol for use in mixin ``self`` annotations.
# Lets mixins access Protocol attributes while still accepting concrete
# Environment instances without "Invalid self argument" errors.
SelfEnv = TypeVar("SelfEnv", bound=EnvironmentProtocol)
