"""Protocol definitions for Environment mixins.

This module defines Protocol classes that specify the interface mixins expect from
the Environment class. Using Protocols allows mypy to understand the mixin pattern
without requiring type: ignore comments or disabling error codes.

See: https://mypy.readthedocs.io/en/latest/more_types.html#mixin-classes
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

if TYPE_CHECKING:
    from neurospatial.animation.overlays import OverlayProtocol
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
    def n_bins(self) -> int:
        """
        Number of spatial bins in the environment.

        Returns
        -------
        int
            Total count of discrete spatial bins.
        """
        ...

    @property
    def n_dims(self) -> int:
        """
        Number of spatial dimensions.

        Returns
        -------
        int
            Dimensionality of the environment (1, 2, or 3).
        """
        ...

    @property
    def is_1d(self) -> bool:
        """
        Whether this is a 1D (linearized) environment.

        Returns
        -------
        bool
            True if environment supports linearization methods.
        """
        ...

    @property
    def bin_sizes(self) -> NDArray[np.float64]:
        """
        Size of bins along each dimension.

        Returns
        -------
        ndarray of shape (n_dims,)
            Bin width in each spatial dimension.
        """
        ...

    @property
    def layout_type(self) -> str | None:
        """
        Type of layout engine used to create this environment.

        Returns
        -------
        str or None
            Layout type name (e.g., 'RegularGrid', 'Hexagonal'), or None
            if not set.
        """
        ...

    @property
    def layout_parameters(self) -> dict[str, Any] | None:
        """
        Parameters used to create the layout.

        Returns
        -------
        dict or None
            Dictionary of layout configuration parameters, or None if not set.
        """
        ...

    @property
    def differential_operator(self) -> sparse.csc_matrix:
        """
        Graph Laplacian matrix for diffusion operations.

        Returns
        -------
        scipy.sparse.csc_matrix of shape (n_bins, n_bins)
            Sparse differential operator matrix.
        """
        ...

    @property
    def boundary_bins(self) -> NDArray[np.int_]:
        """
        Indices of bins at the environment boundary.

        Returns
        -------
        ndarray of shape (n_boundary,)
            Integer indices of boundary bins.
        """
        ...

    # Methods that mixins call
    def bin_at(self, points_nd: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Find bin indices for given points.

        Parameters
        ----------
        points_nd : ndarray of shape (n_points, n_dims) or (n_dims,)
            Spatial coordinates to query.

        Returns
        -------
        ndarray of shape (n_points,) or int
            Bin index for each point. Returns -1 for points outside
            the environment.
        """
        ...

    def bin_sequence(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        dedup: bool = True,
        return_runs: bool = False,
        outside_value: int | None = -1,
    ) -> Any:
        """
        Convert a trajectory to a sequence of bin indices.

        Parameters
        ----------
        times : ndarray of shape (n_samples,)
            Timestamps for each position sample.
        positions : ndarray of shape (n_samples, n_dims)
            Spatial coordinates at each time point.
        dedup : bool, default=True
            If True, remove consecutive duplicate bin indices.
        return_runs : bool, default=False
            If True, return run-length encoded sequence.
        outside_value : int or None, default=-1
            Value to use for positions outside environment.

        Returns
        -------
        ndarray or tuple
            Bin indices, or (bins, run_lengths) if return_runs=True.
        """
        ...

    def bins_in_region(self, region_name: str) -> NDArray[np.int_]:
        """
        Get bin indices within a named region.

        Parameters
        ----------
        region_name : str
            Name of the region to query.

        Returns
        -------
        ndarray of shape (n_bins_in_region,)
            Integer indices of bins within the region.
        """
        ...

    def mask_for_region(self, region_name: str) -> NDArray[np.bool_]:
        """
        Get boolean mask for bins within a named region.

        Parameters
        ----------
        region_name : str
            Name of the region to query.

        Returns
        -------
        ndarray of shape (n_bins,)
            Boolean mask where True indicates bin is in region.
        """
        ...

    def region_membership(
        self,
        regions: Regions | None = None,
        *,
        include_boundary: bool = True,
    ) -> NDArray[np.bool_]:
        """
        Get membership mask for all regions.

        Parameters
        ----------
        regions : Regions, optional
            Regions container to use. If None, uses self.regions.
        include_boundary : bool, default=True
            Whether to include boundary bins in region membership.

        Returns
        -------
        ndarray of shape (n_bins, n_regions)
            Boolean mask where entry [i, j] is True if bin i belongs
            to region j.
        """
        ...

    def compute_kernel(
        self,
        bandwidth: float,
        *,
        mode: Literal["transition", "density"] = "density",
        cache: bool = True,
    ) -> NDArray[np.float64]:
        """
        Compute a smoothing kernel matrix.

        Parameters
        ----------
        bandwidth : float
            Smoothing bandwidth in spatial units.
        mode : {'transition', 'density'}, default='density'
            Kernel normalization mode.
        cache : bool, default=True
            Whether to cache the computed kernel.

        Returns
        -------
        ndarray of shape (n_bins, n_bins)
            Kernel matrix for smoothing operations.
        """
        ...

    def smooth(
        self,
        field: NDArray[np.float64],
        bandwidth: float,
        *,
        mode: Literal["transition", "density"] = "density",
    ) -> NDArray[np.float64]:
        """
        Apply graph-based smoothing to a spatial field.

        Parameters
        ----------
        field : ndarray of shape (n_bins,)
            Spatial field values to smooth.
        bandwidth : float
            Smoothing bandwidth in spatial units.
        mode : {'transition', 'density'}, default='density'
            Kernel normalization mode.

        Returns
        -------
        ndarray of shape (n_bins,)
            Smoothed field values.
        """
        ...

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
    ) -> NDArray[np.float64]:
        """
        Compute spatial occupancy from position data.

        Parameters
        ----------
        times : ndarray of shape (n_samples,)
            Timestamps for each position sample.
        positions : ndarray of shape (n_samples, n_dims)
            Spatial coordinates at each time point.
        speed : ndarray of shape (n_samples,), optional
            Speed at each time point for filtering.
        min_speed : float, optional
            Minimum speed threshold for inclusion.
        max_gap : float, optional
            Maximum time gap before splitting trajectory.
        kernel_bandwidth : float, optional
            Bandwidth for smoothing occupancy.
        time_allocation : {'start', 'linear'}, default='start'
            How to allocate time between samples.
        return_seconds : bool, default=True
            If True, return occupancy in seconds; else in samples.

        Returns
        -------
        ndarray of shape (n_bins,)
            Time spent in each bin.
        """
        ...

    def distance_between(
        self,
        point1: NDArray[np.float64],
        point2: NDArray[np.float64],
        edge_weight: str = "distance",
    ) -> float:
        """
        Compute graph distance between two points.

        Parameters
        ----------
        point1 : ndarray of shape (n_dims,)
            First spatial coordinate.
        point2 : ndarray of shape (n_dims,)
            Second spatial coordinate.
        edge_weight : str, default='distance'
            Edge attribute to use for distance calculation.

        Returns
        -------
        float
            Shortest path distance between points.
        """
        ...

    # Layout-specific attributes and methods (may be None if not applicable)
    active_mask: NDArray[np.bool_] | None
    grid_shape: tuple[int, ...] | None
    grid_edges: tuple[NDArray[np.float64], ...] | None
    linear_bin_centers_1d: NDArray[np.float64] | None
    linear_bin_edges: NDArray[np.float64] | None
    track_graph: nx.Graph | None

    # Layout methods
    def point_to_bin_index(self, points: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Convert N-D points to flat bin indices.

        Parameters
        ----------
        points : ndarray of shape (n_points, n_dims) or (n_dims,)
            Spatial coordinates to convert.

        Returns
        -------
        ndarray of shape (n_points,) or int
            Flat bin index for each point.
        """
        ...

    # Linearization methods (1D environments)
    def to_linear(self, nd_position: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert N-D positions to 1D linear positions.

        Only available for 1D environments (is_1d=True).

        Parameters
        ----------
        nd_position : ndarray of shape (n_points, n_dims) or (n_dims,)
            N-dimensional spatial coordinates.

        Returns
        -------
        ndarray of shape (n_points,) or float
            Linear position along the track.
        """
        ...

    def linear_to_nd(self, linear_position: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Convert 1D linear positions to N-D coordinates.

        Only available for 1D environments (is_1d=True).

        Parameters
        ----------
        linear_position : ndarray of shape (n_points,) or float
            Linear position along the track.

        Returns
        -------
        ndarray of shape (n_points, n_dims) or (n_dims,)
            N-dimensional spatial coordinates.
        """
        ...

    # Private methods that mixins call (trajectory analysis)
    def _allocate_time_linear(
        self,
        positions: NDArray[np.float64],
        dt: NDArray[np.float64],
        valid_mask: NDArray[np.bool_],
        bin_indices: NDArray[np.int64],
        return_seconds: bool,
    ) -> NDArray[np.float64]:
        """
        Allocate time linearly across trajectory segments.

        Parameters
        ----------
        positions : ndarray of shape (n_samples, n_dims)
            Spatial coordinates at each time point.
        dt : ndarray of shape (n_samples - 1,)
            Time differences between consecutive samples.
        valid_mask : ndarray of shape (n_samples,)
            Boolean mask indicating valid samples.
        bin_indices : ndarray of shape (n_samples,)
            Bin index for each sample.
        return_seconds : bool
            If True, return time in seconds; else in samples.

        Returns
        -------
        ndarray of shape (n_bins,)
            Allocated time per bin.
        """
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
    ) -> Any:
        """
        Compute empirical transition matrix from trajectory data.

        Parameters
        ----------
        bins : ndarray of shape (n_samples,), optional
            Pre-computed bin indices.
        times : ndarray of shape (n_samples,), optional
            Timestamps for position data.
        positions : ndarray of shape (n_samples, n_dims), optional
            Position data to bin.
        lag : int, default=1
            Number of steps for transition computation.
        normalize : bool, default=True
            Whether to row-normalize the matrix.
        allow_teleports : bool, default=False
            Whether to include non-adjacent transitions.

        Returns
        -------
        scipy.sparse.csr_matrix of shape (n_bins, n_bins)
            Empirical transition probability matrix.
        """
        ...

    def _random_walk_transitions(
        self,
        *,
        normalize: bool = True,
    ) -> Any:
        """
        Compute random walk transition matrix from connectivity.

        Parameters
        ----------
        normalize : bool, default=True
            Whether to row-normalize the matrix.

        Returns
        -------
        scipy.sparse.csr_matrix of shape (n_bins, n_bins)
            Random walk transition probability matrix.
        """
        ...

    def _diffusion_transitions(
        self,
        bandwidth: float,
        *,
        normalize: bool = True,
    ) -> Any:
        """
        Compute diffusion-based transition matrix.

        Parameters
        ----------
        bandwidth : float
            Diffusion bandwidth in spatial units.
        normalize : bool, default=True
            Whether to row-normalize the matrix.

        Returns
        -------
        scipy.sparse.csr_matrix of shape (n_bins, n_bins)
            Diffusion transition probability matrix.
        """
        ...

    def _compute_ray_grid_intersections(
        self,
        start_pos: NDArray[np.float64],
        end_pos: NDArray[np.float64],
        grid_edges: list[NDArray[np.float64]],
        grid_shape: tuple[int, ...],
        total_time: float,
    ) -> list[tuple[int, float]]:
        """
        Compute grid cell intersections along a ray.

        Parameters
        ----------
        start_pos : ndarray of shape (n_dims,)
            Ray starting position.
        end_pos : ndarray of shape (n_dims,)
            Ray ending position.
        grid_edges : list of ndarray
            Bin edges for each dimension.
        grid_shape : tuple of int
            Shape of the grid.
        total_time : float
            Total time for the ray segment.

        Returns
        -------
        list of tuple (int, float)
            List of (bin_index, time_in_bin) pairs.
        """
        ...

    def _position_to_flat_index(
        self,
        pos: NDArray[np.float64],
        grid_edges: list[NDArray[np.float64]],
        grid_shape: tuple[int, ...],
    ) -> int:
        """
        Convert position to flat grid index.

        Parameters
        ----------
        pos : ndarray of shape (n_dims,)
            Spatial coordinate.
        grid_edges : list of ndarray
            Bin edges for each dimension.
        grid_shape : tuple of int
            Shape of the grid.

        Returns
        -------
        int
            Flat index into the grid.
        """
        ...

    # Private methods that mixins call (field interpolation)
    def _interpolate_nearest(
        self,
        field: NDArray[np.float64],
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Interpolate field values using nearest-neighbor.

        Parameters
        ----------
        field : ndarray of shape (n_bins,)
            Field values at bin centers.
        points : ndarray of shape (n_points, n_dims)
            Query points for interpolation.

        Returns
        -------
        ndarray of shape (n_points,)
            Interpolated field values.
        """
        ...

    def _interpolate_linear(
        self,
        field: NDArray[np.float64],
        points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Interpolate field values using linear interpolation.

        Parameters
        ----------
        field : ndarray of shape (n_bins,)
            Field values at bin centers.
        points : ndarray of shape (n_points, n_dims)
            Query points for interpolation.

        Returns
        -------
        ndarray of shape (n_points,)
            Interpolated field values.
        """
        ...

    # Visualization and animation methods
    def animate_fields(
        self,
        fields: Any,  # Sequence[NDArray[np.float64]] | NDArray[np.float64]
        *,
        backend: Literal["auto", "napari", "video", "html", "widget"] = "auto",
        save_path: str | None = None,
        fps: int = 30,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        frame_labels: Any = None,  # Sequence[str] | None
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
        frame_times: NDArray[np.float64] | None = None,
        show_regions: bool | list[str] = False,
        region_alpha: float = 0.3,
        scale_bar: bool | Any = False,  # bool | ScaleBarConfig
        **kwargs: Any,
    ) -> Any:
        """
        Animate spatial fields over time.

        Parameters
        ----------
        fields : ndarray of shape (n_frames, n_bins) or sequence
            Spatial field values for each frame.
        backend : {'auto', 'napari', 'video', 'html', 'widget'}, default='auto'
            Animation backend to use.
        save_path : str, optional
            Path to save video file.
        fps : int, default=30
            Frames per second for video output.
        cmap : str, default='viridis'
            Colormap name.
        vmin : float, optional
            Minimum value for color scaling.
        vmax : float, optional
            Maximum value for color scaling.
        frame_labels : sequence of str, optional
            Labels for each frame.
        overlay_trajectory : ndarray of shape (n_frames, n_dims), optional
            Trajectory to overlay on animation.
        title : str, default='Spatial Field Animation'
            Animation title.
        dpi : int, default=100
            Resolution for video output.
        codec : str, default='h264'
            Video codec.
        bitrate : int, default=5000
            Video bitrate in kbps.
        n_workers : int, optional
            Number of parallel workers for rendering.
        dry_run : bool, default=False
            If True, validate inputs without rendering.
        image_format : {'png', 'jpeg'}, default='png'
            Format for frame images.
        max_html_frames : int, default=500
            Maximum frames for HTML output.
        contrast_limits : tuple of (float, float), optional
            Min/max values for contrast adjustment.
        show_colorbar : bool, default=False
            Whether to display colorbar.
        colorbar_label : str, default=''
            Label for colorbar.
        overlays : list of OverlayProtocol, optional
            Additional overlay objects.
        frame_times : ndarray of shape (n_frames,), optional
            Timestamps for each frame.
        show_regions : bool or list of str, default=False
            Whether to show region boundaries.
        region_alpha : float, default=0.3
            Transparency for region overlays.
        scale_bar : bool or ScaleBarConfig, default=False
            Whether to show scale bar.
        **kwargs : dict
            Additional backend-specific arguments.

        Returns
        -------
        Any
            Backend-specific return value (viewer, path, or HTML).
        """
        ...


# TypeVar bound to EnvironmentProtocol for use in mixin self annotations
# This allows mixins to access Protocol attributes while still accepting
# concrete Environment instances without "Invalid self argument" errors.
SelfEnv = TypeVar("SelfEnv", bound=EnvironmentProtocol)
