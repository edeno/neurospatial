"""Visualization methods for Environment.

This module contains the EnvironmentVisualization mixin class that provides
plotting capabilities for Environment instances.

Classes
-------
EnvironmentVisualization
    Mixin class providing plot() and plot_1d() methods.

Notes
-----
This module uses TYPE_CHECKING guards to avoid circular imports with the
Environment class. Type hints use string annotations ("Environment") for
forward references.

"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon
from numpy.typing import NDArray

from neurospatial.environment._protocols import SelfEnv
from neurospatial.environment.decorators import check_fitted

if TYPE_CHECKING:
    from neurospatial.animation.overlays import (
        BodypartOverlay,
        HeadDirectionOverlay,
        PositionOverlay,
    )


class EnvironmentVisualization:
    """Mixin class providing visualization methods for Environment.

    This mixin provides plotting functionality for Environment instances,
    including both N-dimensional and 1D linearized visualizations.

    Methods
    -------
    plot(ax=None, show_regions=False, layout_plot_kwargs=None, regions_plot_kwargs=None, **kwargs)
        Plot the environment's layout and optionally defined regions.
    plot_1d(ax=None, layout_plot_kwargs=None, **kwargs)
        Plot a 1D representation of the environment, if applicable.

    Notes
    -----
    This is a mixin class designed to be inherited by the Environment class.
    It should NOT be decorated with @dataclass. Only Environment itself
    should be a dataclass.

    The methods assume the presence of the following attributes on self:
    - layout : LayoutEngine instance
    - name : str
    - regions : Regions instance (optional)

    See Also
    --------
    neurospatial.environment.core.Environment : Main Environment class.

    """

    @check_fitted
    def plot(
        self: SelfEnv,
        ax: matplotlib.axes.Axes | None = None,
        show_regions: bool = False,
        layout_plot_kwargs: dict[str, Any] | None = None,
        regions_plot_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """Plot the environment's layout and optionally defined regions.

        This method delegates plotting of the base layout to the `plot` method
        of the underlying `LayoutEngine`. If `show_regions` is True, it then
        overlays any defined spatial regions managed by `self.regions`.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            The Matplotlib axes to plot on. If None, a new figure and axes
            are created. Defaults to None.
        show_regions : bool, optional
            If True, plot defined spatial regions on top of the layout.
            Defaults to False.
        layout_plot_kwargs : Optional[Dict[str, Any]], optional
            Keyword arguments to pass to the `layout.plot()` method.
            Defaults to None.
        regions_plot_kwargs : Optional[Dict[str, Any]], optional
            Keyword arguments to pass to the `regions.plot_regions()` method.
            Defaults to None.
        **kwargs : Any
            Additional keyword arguments that are passed to `layout.plot()`.
            These can be overridden by `layout_plot_kwargs`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the environment was plotted.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> ax = env.plot()  # doctest: +SKIP

        Plot with regions:

        >>> env.regions.add("goal", point=[5.0, 5.0])
        >>> ax = env.plot(show_regions=True)  # doctest: +SKIP

        """
        l_kwargs = layout_plot_kwargs if layout_plot_kwargs is not None else {}
        l_kwargs.update(kwargs)  # Allow direct kwargs to override for layout.plot

        ax = self.layout.plot(ax=ax, **l_kwargs)
        # layout.plot() always returns an Axes object, never None
        assert ax is not None

        if show_regions and hasattr(self, "regions") and self.regions is not None:
            from neurospatial.regions.plot import plot_regions

            r_kwargs = regions_plot_kwargs if regions_plot_kwargs is not None else {}
            plot_regions(self.regions, ax=ax, **r_kwargs)

        plot_title = self.name
        if (
            self.layout
            and hasattr(self.layout, "_layout_type_tag")
            and self.layout._layout_type_tag
        ):
            plot_title += f" ({self.layout._layout_type_tag})"

        # Only set title if layout.plot didn't set one or user didn't pass one via kwargs to layout.plot
        if ax.get_title() == "":
            ax.set_title(plot_title)

        return ax

    def plot_1d(
        self: SelfEnv,
        ax: matplotlib.axes.Axes | None = None,
        layout_plot_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes | None:
        """Plot a 1D representation of the environment, if applicable.

        This method is primarily for environments where `is_1d` is True
        (e.g., using `GraphLayout`). It calls the `plot_linear_layout`
        method of the underlying layout if it exists and the layout is 1D.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            The Matplotlib axes to plot on. If None, a new figure and axes
            are created. Defaults to None.
        layout_plot_kwargs : Optional[Dict[str, Any]], optional
            Keyword arguments to pass to the layout's 1D plotting method.
        **kwargs : Any
            Additional keyword arguments passed to the layout's 1D plotting method.

        Returns
        -------
        matplotlib.axes.Axes | None
            The axes on which the 1D layout was plotted, or the original `ax`
            (which may be None) if plotting was not applicable.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        AttributeError
            If `self.layout.is_1d` is True but the layout does not have a
            `plot_linear_layout` method.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> # Create a 1D track environment
        >>> from track_linearization import make_track_graph
        >>> position = np.random.rand(100, 2) * 10
        >>> track_graph = make_track_graph(position, use_HMM=False)
        >>> env = Environment.from_graph(track_graph, track_graph_name="test_track")
        >>> if env.is_1d:
        ...     ax = env.plot_1d()  # doctest: +SKIP

        """
        l_kwargs = layout_plot_kwargs if layout_plot_kwargs is not None else {}
        l_kwargs.update(kwargs)  # Allow direct kwargs to override for layout.plot
        if self.layout.is_1d:
            if hasattr(self.layout, "plot_linear_layout"):
                ax = self.layout.plot_linear_layout(ax=ax, **l_kwargs)
            else:
                warnings.warn(
                    f"Layout '{self._layout_type_used}' is 1D but does not "
                    "have a 'plot_linear_layout' method. Skipping 1D plot.",
                    UserWarning,
                )
        else:
            warnings.warn(
                "Environment is not 1D. Skipping 1D plot. Use regular plot() method.",
                UserWarning,
            )

        return ax

    @check_fitted
    def plot_field(
        self: SelfEnv,
        field: NDArray[np.float64],
        ax: matplotlib.axes.Axes | None = None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        colorbar: bool = True,
        colorbar_label: str = "",
        nan_color: str | None = "lightgray",
        rasterized: bool = True,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """Plot spatial field data over the environment.

        Automatically selects the appropriate visualization method based on the
        layout type to render bins with their actual geometric shapes:

        - **Grid layouts**: Uses ``pcolormesh`` for crisp rectangular bins
        - **Hexagonal**: Colored hexagon patches via ``PatchCollection``
        - **Triangular mesh**: Colored triangle faces via ``tripcolor``
        - **1D tracks**: Line plot with filled area
        - **Other**: Scatter plot with auto-sized colored markers

        Parameters
        ----------
        field : NDArray[np.float64], shape (n_bins,)
            Spatial field values for each bin (e.g., firing rate, probability,
            decoded position density).
        ax : Optional[matplotlib.axes.Axes], optional
            Axes to plot on. If None, creates new figure and axes. Default: None.
        cmap : str, default="viridis"
            Matplotlib colormap name (e.g., "hot", "Blues", "viridis").
        vmin : Optional[float], optional
            Minimum value for colormap normalization. If None, uses
            ``np.nanmin(field)``. Default: None.
        vmax : Optional[float], optional
            Maximum value for colormap normalization. If None, uses
            ``np.nanmax(field)``. Default: None.
        colorbar : bool, default=True
            Whether to add a colorbar. Not applicable for 1D plots.
        colorbar_label : str, default=""
            Label text for the colorbar.
        nan_color : Optional[str], default="lightgray"
            Color for bins with NaN values. If None, NaN bins are not rendered
            (skipped). Use "lightgray", "white", etc.
        rasterized : bool, default=True
            Rasterize output for better performance and smaller file sizes with
            large grids. Recommended for environments with >1000 bins.
        **kwargs : Any
            Additional keyword arguments passed to underlying plot function
            (e.g., ``pcolormesh``, ``PatchCollection.set``).

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the field was plotted.

        Raises
        ------
        ValueError
            If ``field`` shape does not match ``(n_bins,)``.
        NotImplementedError
            If environment dimensionality is >2D (not supported for spatial plots).
        NotImplementedError
            If attempting to use ``pcolormesh`` on non-2D grid.

        See Also
        --------
        compute_place_field : Compute firing rate maps from spike trains.
        plot : Plot environment structure (bins and connectivity).
        plot_1d : Plot 1D linearized environment.

        Notes
        -----
        **NaN Handling:**

        - ``nan_color=None``: NaN bins are skipped (not rendered)
        - ``nan_color="lightgray"``: NaN bins rendered with specified color
        - If all field values are NaN and ``nan_color=None``, an empty plot is
          produced (no colorbar will be added).

        **Performance:**

        For large environments (>10,000 bins), set ``rasterized=True`` (default)
        to improve rendering performance and reduce file size.

        **Colormaps:**

        Common choices for neuroscience:

        - ``"hot"``: Firing rate maps (black → red → yellow)
        - ``"viridis"``: General purpose, perceptually uniform
        - ``"Blues"``: Probability distributions
        - ``"RdBu_r"``: Diverging data (e.g., correlation)

        Examples
        --------
        Plot a firing rate map:

        >>> import numpy as np
        >>> from neurospatial import Environment, compute_place_field
        >>> positions = np.random.uniform(0, 100, (1000, 2))
        >>> times = np.linspace(0, 100, 1000)
        >>> spike_times = np.random.uniform(0, 100, 50)
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> firing_rate = compute_place_field(
        ...     env, spike_times, times, positions, bandwidth=8.0
        ... )
        >>> ax = env.plot_field(
        ...     firing_rate, cmap="hot", colorbar_label="Firing Rate (Hz)", vmin=0
        ... )  # doctest: +SKIP

        Plot decoded probability distribution:

        >>> posterior = np.random.rand(env.n_bins)  # Example posterior
        >>> ax = env.plot_field(
        ...     posterior, cmap="Blues", colorbar_label="Probability"
        ... )  # doctest: +SKIP

        """
        # Validate field shape
        if field.ndim != 1 or field.shape[0] != self.n_bins:
            raise ValueError(
                f"field must be 1D array with length n_bins={self.n_bins}, "
                f"got shape {field.shape}"
            )

        # Check dimensionality for spatial plots
        if not self.layout.is_1d and self.n_dims > 2:
            raise NotImplementedError(
                f"Cannot plot {self.n_dims}D fields spatially. "
                "Only 1D and 2D environments are supported. "
                "Consider using marginal plots, slicing, or 3D scatter plots instead."
            )

        # Create axes if needed
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 7))

        # Auto vmin/vmax
        if vmin is None:
            vmin = np.nanmin(field)
            if np.isnan(vmin) or np.isinf(vmin):
                vmin = 0.0

        if vmax is None:
            vmax = np.nanmax(field)
            if np.isnan(vmax) or np.isinf(vmax):
                vmax = 1.0

        # Ensure vmin < vmax
        if vmin >= vmax:
            vmax = vmin + 1.0

        # Dispatch to layout-specific renderer
        layout_tag = self.layout._layout_type_tag
        mappable = None

        grid_layouts = ("RegularGrid", "MaskedGrid", "ImageMask", "ShapelyPolygon")

        # Validate layout-specific requirements before dispatch
        if layout_tag in grid_layouts:
            if not (
                hasattr(self.layout, "grid_shape")
                and hasattr(self.layout, "grid_edges")
                and hasattr(self.layout, "active_mask")
            ):
                raise RuntimeError(
                    f"Layout '{layout_tag}' missing required grid attributes "
                    "(grid_shape, grid_edges, active_mask). "
                    "This indicates a malformed layout engine."
                )
            mappable = _plot_grid_field(
                self, field, ax, cmap, vmin, vmax, nan_color, rasterized, **kwargs
            )
        elif layout_tag == "Hexagonal":
            if not (
                hasattr(self.layout, "hex_radius_")
                and hasattr(self.layout, "hex_orientation_")
            ):
                raise RuntimeError(
                    "Hexagonal layout missing required attributes "
                    "(hex_radius_, hex_orientation_). "
                    "This indicates a malformed layout engine."
                )
            mappable = _plot_hex_field(
                self, field, ax, cmap, vmin, vmax, nan_color, rasterized, **kwargs
            )
        elif layout_tag == "TriangularMesh":
            if not (
                hasattr(self.layout, "_full_delaunay_tri")
                and hasattr(self.layout, "_active_original_simplex_indices")
            ):
                raise RuntimeError(
                    "TriangularMesh layout missing required attributes "
                    "(_full_delaunay_tri, _active_original_simplex_indices). "
                    "This indicates a malformed layout engine."
                )
            mappable = _plot_trimesh_field(
                self, field, ax, cmap, vmin, vmax, nan_color, rasterized, **kwargs
            )
        elif self.layout.is_1d:
            mappable = _plot_1d_field(self, field, ax, colorbar_label, **kwargs)
        else:
            mappable = _plot_scatter_field(
                self, field, ax, cmap, vmin, vmax, nan_color, rasterized, **kwargs
            )

        # Add colorbar (not for 1D)
        # Note: mappable can be None if all field values are NaN and nan_color=None
        if colorbar and mappable is not None and not self.layout.is_1d:
            cbar = plt.colorbar(mappable, ax=ax)
            if colorbar_label:
                cbar.set_label(colorbar_label, fontsize=12)
            cbar.ax.tick_params(labelsize=10)

        # Format axes
        if not self.layout.is_1d:
            ax.set_aspect("equal")
            unit_label = f" ({self.units})" if self.units else ""
            ax.set_xlabel(f"X Position{unit_label}", fontsize=12)
            ax.set_ylabel(f"Y Position{unit_label}", fontsize=12)

            if self.dimension_ranges and len(self.dimension_ranges) >= 2:
                ax.set_xlim(self.dimension_ranges[0])
                ax.set_ylim(self.dimension_ranges[1])

        return ax

    @check_fitted
    def animate_fields(
        self: SelfEnv,
        fields: Sequence[NDArray[np.float64]] | NDArray[np.float64],
        *,
        backend: Literal["auto", "napari", "video", "html", "widget"] = "auto",
        save_path: str | None = None,
        fps: int = 30,
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
        overlays: list[PositionOverlay | BodypartOverlay | HeadDirectionOverlay]
        | None = None,
        frame_times: NDArray[np.float64] | None = None,
        show_regions: bool | list[str] = False,
        region_alpha: float = 0.3,
        **kwargs: Any,
    ) -> Any:
        """Animate spatial fields over time with multiple backend options.

        Creates animations of spatial field data across different time points,
        with support for four different backends optimized for different use cases:
        Napari (GPU-accelerated interactive viewing), Video (MP4/WebM export),
        HTML (standalone shareable player), and Jupyter Widget (notebook integration).

        Parameters
        ----------
        fields : Sequence[NDArray[np.float64]] or NDArray[np.float64]
            Spatial field data to animate. Can be:
            - List of arrays: Each array shape (n_bins,) represents one frame
            - Single ndarray: Shape (n_frames, n_bins) with first dimension as time
            Each field must match the number of bins in this environment.
        backend : {"auto", "napari", "video", "html", "widget"}, default="auto"
            Animation backend to use:
            - "auto": Automatically selects based on context and data size
            - "napari": GPU-accelerated interactive viewer (best for large datasets)
            - "video": Export MP4/WebM with parallel rendering (best for publications)
            - "html": Standalone HTML player with instant scrubbing (best for sharing)
            - "widget": Jupyter widget with slider control (best for notebooks)
        save_path : str, optional
            Output file path. If provided, file extension determines format:
            - .mp4, .webm, .avi, .mov: video export (requires ffmpeg)
            - .html: standalone HTML player (no dependencies)
            - None: display interactively (napari or widget depending on context)
        fps : int, default=30
            Playback frame rate in frames per second
        cmap : str, default="viridis"
            Matplotlib colormap name (e.g., "hot", "Blues", "viridis", "RdBu_r")
        vmin : float, optional
            Minimum value for colormap normalization. If None, computed from all frames.
        vmax : float, optional
            Maximum value for colormap normalization. If None, computed from all frames.
        frame_labels : Sequence[str], optional
            Labels for each frame (e.g., ["Trial 1", "Trial 2", ...]). If None,
            auto-generated as "Frame 1", "Frame 2", etc.
        overlay_trajectory : NDArray[np.float64], optional
            Trajectory to overlay on animation. Shape: (n_timepoints, n_dims).
            For 2D environments: rendered as track line. For higher dimensions:
            rendered as point cloud. Only supported by napari backend.
        title : str, default="Spatial Field Animation"
            Animation title shown in player controls or window title
        dpi : int, default=100
            Resolution for rendering frames (video and HTML backends). Higher values
            produce sharper images but larger file sizes.
        codec : str, default="h264"
            Video codec for MP4 export. Options: "h264", "h265", "vp9", "mpeg4".
            Only used with video backend.
        bitrate : int, default=5000
            Video bitrate in kbps. Higher values produce better quality but larger
            files. Only used with video backend.
        n_workers : int, optional
            Number of parallel workers for video rendering. If None, uses
            CPU count / 2. Set to 1 for serial rendering. Requires environment
            to be pickle-able (call `env.clear_cache()` if pickle errors occur).
            Only used with video backend.
        dry_run : bool, default=False
            If True, estimate rendering time and file size without actually rendering.
            Only used with video backend.
        image_format : {"png", "jpeg"}, default="png"
            Image format for frame embedding in HTML. JPEG produces smaller files
            but with lossy compression. Only used with HTML backend.
        max_html_frames : int, default=500
            Maximum number of frames allowed for HTML export. Prevents creating
            huge files that crash browsers. Only used with HTML backend.
        contrast_limits : tuple[float, float], optional
            Napari contrast limits (min, max). If provided, overrides vmin/vmax.
            Only used with napari backend.
        show_colorbar : bool, default=False
            Whether to include colorbar in rendered frames (not yet implemented)
        colorbar_label : str, default=""
            Label for colorbar axis (not yet implemented)
        overlays : list of overlay objects, optional
            List of overlay objects to render on top of the spatial field animation.
            Supported types: PositionOverlay, BodypartOverlay, HeadDirectionOverlay.
            Multiple overlays of the same type can be provided for multi-animal tracking.
            See neurospatial.animation.overlays for details. Default: None (no overlays).
        frame_times : NDArray[np.float64], shape (n_frames,), optional
            Explicit timestamps for each frame in seconds. If provided, overlays with
            timestamps will be aligned via interpolation. If None, frames are assumed
            to be evenly spaced at the specified fps. Default: None.
        show_regions : bool or list of str, default=False
            Whether to render region overlays. If True, all regions defined in
            env.regions are rendered. If a list of region names, only those regions
            are rendered. Regions are rendered as semi-transparent polygons.
            Default: False (no region overlays).
        region_alpha : float, default=0.3
            Alpha transparency for region overlays. Range: [0.0, 1.0] where
            0.0 is fully transparent and 1.0 is fully opaque. Only used when
            show_regions is True or a list. Default: 0.3.
        **kwargs : dict
            Backend-specific parameters. Common backend-specific options:

            **Napari backend:**
            - ``layout`` : {"horizontal", "vertical", "grid"}, optional
              Layout for multi-field viewing (required when fields is a list of lists)
            - ``layer_names`` : list of str, optional
              Custom names for each layer in multi-field mode
            - ``cache_size`` : int, default=1000
              Maximum frames to cache (per-frame caching)
            - ``chunk_size`` : int, default=10
              Frames per chunk (chunked caching for >10K frames)
            - ``max_chunks`` : int, default=100
              Maximum chunks to cache

            **Video backend:**
            - ``crf`` : int, default=18
              Constant rate factor for quality (0=lossless, 51=worst)
            - ``preset`` : str, default="medium"
              Encoding speed preset (ultrafast, fast, medium, slow, veryslow)

            **Widget backend:**
            - ``initial_cache_size`` : int, optional
              Number of frames to pre-render
            - ``cache_limit`` : int, default=1000
              Maximum cache size

        Returns
        -------
        viewer or path or widget
            Return value depends on backend:
            - napari: napari.Viewer instance (blocking - shows window)
            - video: Path to saved video file
            - html: Path to saved HTML file
            - widget: ipywidgets.interact instance (displays automatically in notebook)

        Raises
        ------
        RuntimeError
            If environment is not fitted (use factory methods like
            Environment.from_samples())
        ValueError
            If field shapes don't match environment n_bins
        ImportError
            If required backend dependencies are not installed
        RuntimeError
            If ffmpeg is not available for video backend

        See Also
        --------
        plot_field : Plot static spatial field
        plot : Plot environment structure
        neurospatial.animation.subsample_frames : Subsample frames for large datasets

        Notes
        -----
        **Backend Selection (auto mode):**
        When backend="auto", the selection logic is:
        - If save_path has video extension (.mp4, .webm, etc.) → video backend
        - If save_path is .html → HTML backend
        - If >10,000 frames → napari backend (GPU acceleration needed)
        - If in Jupyter notebook → widget backend
        - Otherwise → napari backend (if available) or error

        **Layout Support:**
        All layout types are supported:
        - Grid layouts (RegularGrid, MaskedGrid, etc.): Direct rendering
        - Hexagonal/Triangular: Layout-aware rendering with proper shapes
        - Graph layouts (1D): Rendered as 1D line plots
        The animation automatically uses the appropriate renderer based on layout type.

        **Performance Tips:**
        - For large datasets (>10K frames), use memory-mapped arrays to avoid loading
          all data into RAM
        - Use napari backend for interactive exploration of large datasets
        - For video export, increase n_workers for faster parallel rendering
        - For HTML, use image_format="jpeg" to reduce file size (with quality loss)
        - Use subsample_frames() utility to downsample high-frequency recordings

        **Memory Considerations:**
        - Napari backend: Frames loaded on-demand (LRU cache), handles 900K+ frames
        - Video backend: Renders in chunks, temporary files cleaned up automatically
        - HTML backend: All frames embedded in single file (limit: 500 frames default)
        - Widget backend: Pre-renders first 500 frames, on-demand for rest

        Examples
        --------
        Create environment and simulate field evolution:

        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> positions = np.random.uniform(0, 100, (1000, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> # Simulate place field formation over 20 trials
        >>> fields = []
        >>> center_bin = env.n_bins // 2  # Use middle bin
        >>> for trial in range(20):
        ...     distances = env.distance_to([center_bin])
        ...     field = np.exp(-distances / (10 + trial))
        ...     fields.append(field)

        Interactive exploration with Napari:

        >>> viewer = env.animate_fields(fields, backend="napari")  # doctest: +SKIP

        Video export for publication:

        >>> env.animate_fields(
        ...     fields,
        ...     save_path="place_field_learning.mp4",
        ...     fps=5,
        ...     frame_labels=[f"Trial {i + 1}" for i in range(20)],
        ...     cmap="hot",
        ... )  # doctest: +SKIP

        Shareable HTML with instant scrubbing:

        >>> env.animate_fields(
        ...     fields, save_path="exploration.html", fps=10
        ... )  # doctest: +SKIP

        Quick notebook check with widget:

        >>> env.animate_fields(fields, backend="widget")  # doctest: +SKIP

        Large-scale session (memory-mapped data):

        >>> # For hour-long recording (900K frames at 250 Hz)
        >>> from neurospatial.animation import subsample_frames
        >>> # Subsample to 30 fps for video export
        >>> fields_sub = subsample_frames(fields_mmap, target_fps=30, source_fps=250)
        >>> env.animate_fields(
        ...     fields_sub, save_path="session_summary.mp4", n_workers=8
        ... )  # doctest: +SKIP

        **Overlay Examples (v0.4.0+):**

        Position overlay with trail:

        >>> from neurospatial.animation import PositionOverlay
        >>> trajectory = np.random.uniform(0, 100, (100, 2))
        >>> position_overlay = PositionOverlay(
        ...     data=trajectory,
        ...     color="red",
        ...     size=12.0,
        ...     trail_length=10,  # Show last 10 frames as trail
        ... )
        >>> env.animate_fields(
        ...     fields, overlays=[position_overlay], backend="napari"
        ... )  # doctest: +SKIP

        Pose tracking with skeleton:

        >>> from neurospatial.animation import BodypartOverlay
        >>> pose_data = {
        ...     "nose": np.random.uniform(0, 100, (100, 2)),
        ...     "ear_left": np.random.uniform(0, 100, (100, 2)),
        ...     "ear_right": np.random.uniform(0, 100, (100, 2)),
        ...     "tail_base": np.random.uniform(0, 100, (100, 2)),
        ... }
        >>> bodypart_overlay = BodypartOverlay(
        ...     data=pose_data,
        ...     skeleton=[
        ...         ("nose", "ear_left"),
        ...         ("nose", "ear_right"),
        ...         ("nose", "tail_base"),
        ...     ],
        ...     colors={
        ...         "nose": "red",
        ...         "ear_left": "blue",
        ...         "ear_right": "blue",
        ...         "tail_base": "green",
        ...     },
        ...     skeleton_color="white",
        ... )
        >>> env.animate_fields(
        ...     fields,
        ...     overlays=[bodypart_overlay],
        ...     backend="video",
        ...     save_path="pose.mp4",
        ... )  # doctest: +SKIP

        Head direction visualization:

        >>> from neurospatial.animation import HeadDirectionOverlay
        >>> angles = np.linspace(0, 2 * np.pi, 100)
        >>> direction_overlay = HeadDirectionOverlay(
        ...     data=angles,
        ...     color="yellow",
        ...     length=15.0,  # Arrow length in environment units
        ... )
        >>> env.animate_fields(
        ...     fields, overlays=[direction_overlay], backend="napari"
        ... )  # doctest: +SKIP

        Multi-animal tracking (multiple overlays):

        >>> animal1_pos = PositionOverlay(
        ...     data=trajectory1, color="red", size=12.0, trail_length=10
        ... )
        >>> animal2_pos = PositionOverlay(
        ...     data=trajectory2, color="blue", size=12.0, trail_length=10
        ... )
        >>> env.animate_fields(
        ...     fields, overlays=[animal1_pos, animal2_pos], backend="napari"
        ... )  # doctest: +SKIP

        All overlay types combined with regions:

        >>> from neurospatial.regions import Region
        >>> env.regions.add("goal", point=np.array([80.0, 80.0]))
        >>> env.animate_fields(
        ...     fields,
        ...     overlays=[position_overlay, bodypart_overlay, direction_overlay],
        ...     show_regions=True,
        ...     region_alpha=0.4,
        ...     backend="video",
        ...     save_path="comprehensive.mp4",
        ... )  # doctest: +SKIP

        Temporal alignment with mixed sampling rates:

        >>> # Trajectory sampled at 250 Hz, fields at 30 fps
        >>> trajectory_250hz = np.random.uniform(0, 100, (2500, 2))
        >>> timestamps = np.linspace(0, 10, 2500)  # 10 seconds at 250 Hz
        >>> position_overlay = PositionOverlay(
        ...     data=trajectory_250hz,
        ...     times=timestamps,  # Temporal alignment
        ...     color="red",
        ...     trail_length=15,
        ... )
        >>> frame_times = np.linspace(0, 10, 100)  # 100 frames over 10 seconds
        >>> env.animate_fields(
        ...     fields,
        ...     overlays=[position_overlay],
        ...     frame_times=frame_times,  # Interpolation aligns overlay to frames
        ...     backend="video",
        ...     save_path="aligned.mp4",
        ... )  # doctest: +SKIP

        """
        from neurospatial.animation.core import animate_fields as _animate

        return _animate(
            env=self,
            fields=fields,
            backend=backend,
            save_path=save_path,
            fps=fps,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            frame_labels=frame_labels,
            overlay_trajectory=overlay_trajectory,
            title=title,
            dpi=dpi,
            codec=codec,
            bitrate=bitrate,
            n_workers=n_workers,
            dry_run=dry_run,
            image_format=image_format,
            max_html_frames=max_html_frames,
            contrast_limits=contrast_limits,
            show_colorbar=show_colorbar,
            colorbar_label=colorbar_label,
            overlays=overlays,
            frame_times=frame_times,
            show_regions=show_regions,
            region_alpha=region_alpha,
            **kwargs,  # Forward backend-specific parameters (e.g., layout, layer_names for napari)
        )


# Helper functions for layout-specific rendering


def _plot_grid_field(
    env: SelfEnv,
    field: NDArray[np.float64],
    ax: matplotlib.axes.Axes,
    cmap: str,
    vmin: float,
    vmax: float,
    nan_color: str | None,
    rasterized: bool,
    **kwargs: Any,
) -> Any:
    """Plot field on grid layout using pcolormesh."""
    from neurospatial.layout.helpers.utils import map_active_data_to_grid

    # Validate grid attributes exist
    if env.layout.grid_shape is None or env.layout.grid_edges is None:
        raise RuntimeError("Grid layout missing grid_shape or grid_edges")

    if env.layout.active_mask is None:
        raise RuntimeError("Grid layout missing active_mask")

    # Validate 2D grid
    if len(env.layout.grid_shape) != 2:
        raise NotImplementedError(
            f"pcolormesh requires 2D grids, got grid_shape={env.layout.grid_shape}"
        )

    # Convert to grid
    grid_data = map_active_data_to_grid(
        env.layout.grid_shape, env.layout.active_mask, field, fill_value=np.nan
    )

    # Setup colormap with NaN handling
    cmap_obj = plt.get_cmap(cmap).copy()
    if nan_color is not None:
        cmap_obj.set_bad(color=nan_color, alpha=1.0)

    # Plot
    mesh = ax.pcolormesh(
        env.layout.grid_edges[0],
        env.layout.grid_edges[1],
        grid_data.T,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        rasterized=rasterized,
        shading="flat",
        **kwargs,
    )

    return mesh


def _plot_hex_field(
    env: SelfEnv,
    field: NDArray[np.float64],
    ax: matplotlib.axes.Axes,
    cmap: str,
    vmin: float,
    vmax: float,
    nan_color: str | None,
    rasterized: bool,
    **kwargs: Any,
) -> Any:
    """Plot field on hexagonal layout using PatchCollection."""
    # Validate hexagonal layout attributes exist
    if not hasattr(env.layout, "hex_radius_") or not hasattr(
        env.layout, "hex_orientation_"
    ):
        raise RuntimeError(
            "Hexagonal layout missing hex_radius_ or hex_orientation_ attributes"
        )

    # Create hexagon patches
    # Note: Unlike grid layouts which use colormap's built-in NaN handling,
    # PatchCollection requires manual filtering of NaN values to avoid creating
    # empty patches that matplotlib doesn't handle well.
    patches = []
    colors = []

    for center, value in zip(env.bin_centers, field, strict=True):
        # Skip NaN bins if nan_color is None
        if np.isnan(value) and nan_color is None:
            continue

        patch = RegularPolygon(
            xy=center,
            numVertices=6,
            radius=env.layout.hex_radius_,
            orientation=env.layout.hex_orientation_,
        )
        patches.append(patch)
        colors.append(value)

    if not patches:
        return None

    # Create collection
    collection = PatchCollection(patches, cmap=cmap, rasterized=rasterized, **kwargs)
    collection.set_array(np.array(colors))
    collection.set_clim(vmin, vmax)
    collection.set_edgecolor("none")

    ax.add_collection(collection)
    return collection


def _plot_trimesh_field(
    env: SelfEnv,
    field: NDArray[np.float64],
    ax: matplotlib.axes.Axes,
    cmap: str,
    vmin: float,
    vmax: float,
    nan_color: str | None,
    rasterized: bool,
    **kwargs: Any,
) -> Any:
    """Plot field on triangular mesh layout using tripcolor."""
    # Validate triangular mesh layout attributes exist
    if not hasattr(env.layout, "_full_delaunay_tri") or not hasattr(
        env.layout, "_active_original_simplex_indices"
    ):
        raise RuntimeError(
            "TriangularMesh layout missing _full_delaunay_tri or "
            "_active_original_simplex_indices attributes"
        )

    # Get triangle vertex indices
    triangles = env.layout._full_delaunay_tri.simplices[
        env.layout._active_original_simplex_indices
    ]

    # Get mesh points
    mesh_points = env.layout._full_delaunay_tri.points

    # Setup colormap with NaN handling
    cmap_obj = plt.get_cmap(cmap).copy()
    if nan_color is not None:
        cmap_obj.set_bad(color=nan_color, alpha=1.0)

    # Plot using tripcolor
    mesh = ax.tripcolor(
        mesh_points[:, 0],
        mesh_points[:, 1],
        triangles,
        facecolors=field,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        rasterized=rasterized,
        **kwargs,
    )

    return mesh


def _plot_1d_field(
    env: SelfEnv,
    field: NDArray[np.float64],
    ax: matplotlib.axes.Axes,
    ylabel: str,
    **kwargs: Any,
) -> Any:
    """Plot field on 1D layout using line plot."""
    # Filter NaN values
    valid_mask = ~np.isnan(field)
    valid_bins = np.where(valid_mask)[0]
    valid_values = field[valid_mask]

    if len(valid_bins) == 0:
        return None

    # Use bin indices as x-axis
    x_positions = valid_bins

    # Plot
    (line,) = ax.plot(x_positions, valid_values, linewidth=2, **kwargs)
    ax.fill_between(x_positions, 0, valid_values, alpha=0.3, color=line.get_color())

    ax.set_xlabel("Bin Index", fontsize=12)
    ax.set_ylabel(ylabel if ylabel else "Field Value", fontsize=12)
    ax.set_title("1D Field Plot", fontsize=14)

    return line


def _plot_scatter_field(
    env: SelfEnv,
    field: NDArray[np.float64],
    ax: matplotlib.axes.Axes,
    cmap: str,
    vmin: float,
    vmax: float,
    nan_color: str | None,
    rasterized: bool,
    **kwargs: Any,
) -> Any:
    """Plot field using scatter plot (fallback for unknown layouts)."""
    # Filter NaN values if nan_color is None
    if nan_color is None:
        valid_mask = ~np.isnan(field)
        plot_centers = env.bin_centers[valid_mask]
        plot_values = field[valid_mask]
    else:
        plot_centers = env.bin_centers
        plot_values = field.copy()

    if len(plot_centers) == 0:
        return None

    # Auto-compute marker size from bin spacing
    if env.n_bins > 1:
        from scipy.spatial import cKDTree

        tree = cKDTree(env.bin_centers)
        # Find nearest neighbor distance for sample
        sample_size = min(100, env.n_bins)
        distances, _ = tree.query(env.bin_centers[:sample_size], k=2)
        typical_spacing = np.median(distances[:, 1])
        marker_size = (typical_spacing * 72 / 2) ** 2 if typical_spacing > 0 else 100
    else:
        marker_size = 100

    # Plot
    scatter = ax.scatter(
        plot_centers[:, 0],
        plot_centers[:, 1],
        c=plot_values,
        s=marker_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
        rasterized=rasterized,
        **kwargs,
    )

    return scatter
