"""Protocol definitions for Environment mixins.

This module defines Protocol classes that specify the interface mixins expect from
the Environment class. Using Protocols allows mypy to understand the mixin pattern
without requiring type: ignore comments or disabling error codes.

See: https://mypy.readthedocs.io/en/latest/more_types.html#mixin-classes
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

if TYPE_CHECKING:
    from neurospatial.layout.base import LayoutEngine
    from neurospatial.regions import Regions


class EnvironmentProtocol(Protocol):
    """Protocol defining the interface that Environment provides to mixins.

    This protocol specifies all attributes and methods that mixins may access
    from the Environment class. Mixins use `self: EnvironmentProtocol` instead
    of `self: Environment` to avoid "erased type" errors in mypy.
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

    # Internal/cache attributes that mixins access
    _kernel_cache: dict[
        tuple[float, Literal["transition", "density"]], NDArray[np.float64]
    ]
    _layout_type_used: str | None
    _is_1d_env: bool
    _kdtree_cache: Any  # scipy.spatial.cKDTree, but avoiding import

    # Computed properties
    @property
    def n_bins(self) -> int: ...

    @property
    def n_dims(self) -> int: ...

    @property
    def is_1d(self) -> bool: ...

    @property
    def bin_sizes(self) -> NDArray[np.float64]: ...

    @property
    def layout_type(self) -> str | None: ...

    @property
    def layout_parameters(self) -> dict[str, Any] | None: ...

    @property
    def differential_operator(self) -> sparse.csc_matrix: ...

    @property
    def boundary_bins(self) -> NDArray[np.int_]: ...

    # Methods that mixins call
    def bin_at(self, points_nd: NDArray[np.float64]) -> NDArray[np.int_]: ...

    def bin_sequence(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        dedup: bool = True,
        return_runs: bool = False,
        outside_value: int | None = -1,
    ) -> Any: ...  # Returns different types based on return_runs

    def bins_in_region(self, region_name: str) -> NDArray[np.int_]: ...

    def region_membership(
        self,
        regions: Regions | None = None,
        *,
        include_boundary: bool = True,
    ) -> NDArray[np.bool_]: ...

    def compute_kernel(
        self,
        bandwidth: float,
        *,
        mode: Literal["transition", "density"] = "density",
        cache: bool = True,
    ) -> NDArray[np.float64]: ...

    def smooth(
        self,
        field: NDArray[np.float64],
        bandwidth: float,
        *,
        mode: Literal["transition", "density"] = "density",
    ) -> NDArray[np.float64]: ...

    def occupancy(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        speed: NDArray[np.float64] | None = None,
        min_speed: float | None = None,
        max_gap: float | None = None,
        kernel_bandwidth: float | None = None,
        time_allocation: Literal["start", "linear"] = "start",
        return_seconds: bool = True,
    ) -> NDArray[np.float64]: ...

    def distance_between(
        self,
        point1: NDArray[np.float64],
        point2: NDArray[np.float64],
        edge_weight: str = "distance",
    ) -> float: ...

    # Layout-specific attributes and methods (may be None if not applicable)
    active_mask: NDArray[np.bool_] | None
    grid_shape: tuple[int, ...] | None
    grid_edges: tuple[NDArray[np.float64], ...] | None
    linear_bin_centers_1d: NDArray[np.float64] | None
    linear_bin_edges: NDArray[np.float64] | None
    track_graph: nx.Graph | None

    # Layout methods
    def point_to_bin_index(self, points: NDArray[np.float64]) -> NDArray[np.int_]: ...

    # Linearization methods (1D environments)
    def to_linear(self, nd_position: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def linear_to_nd(
        self, linear_position: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    # Private methods that mixins call (trajectory analysis)
    def _allocate_time_linear(
        self,
        positions: NDArray[np.float64],
        dt: NDArray[np.float64],
        valid_mask: NDArray[np.bool_],
        bin_indices: NDArray[np.int64],
        return_seconds: bool,
    ) -> NDArray[np.float64]: ...

    def _empirical_transitions(
        self,
        bins: NDArray[np.int32] | None = None,
        *,
        times: NDArray[np.float64] | None = None,
        positions: NDArray[np.float64] | None = None,
        lag: int = 1,
        normalize: bool = True,
        allow_teleports: bool = False,
    ) -> Any: ...  # scipy.sparse.csr_matrix

    def _random_walk_transitions(
        self,
        *,
        normalize: bool = True,
    ) -> Any: ...  # scipy.sparse.csr_matrix

    def _diffusion_transitions(
        self,
        bandwidth: float,
        *,
        normalize: bool = True,
    ) -> Any: ...  # scipy.sparse.csr_matrix

    def _compute_ray_grid_intersections(
        self,
        start_pos: NDArray[np.float64],
        end_pos: NDArray[np.float64],
        grid_edges: list[NDArray[np.float64]],
        grid_shape: tuple[int, ...],
        total_time: float,
    ) -> list[tuple[int, float]]: ...

    def _position_to_flat_index(
        self,
        pos: NDArray[np.float64],
        grid_edges: list[NDArray[np.float64]],
        grid_shape: tuple[int, ...],
    ) -> int: ...

    # Private methods that mixins call (field interpolation)
    def _interpolate_nearest(
        self,
        field: NDArray[np.float64],
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

    def _interpolate_linear(
        self,
        field: NDArray[np.float64],
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...
