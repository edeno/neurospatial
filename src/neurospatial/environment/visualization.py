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
from typing import Any

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon
from numpy.typing import NDArray

from neurospatial.environment._protocols import SelfEnv
from neurospatial.environment.decorators import check_fitted


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

        if layout_tag in grid_layouts:
            mappable = _plot_grid_field(
                self, field, ax, cmap, vmin, vmax, nan_color, rasterized, **kwargs
            )
        elif layout_tag == "Hexagonal":
            mappable = _plot_hex_field(
                self, field, ax, cmap, vmin, vmax, nan_color, rasterized, **kwargs
            )
        elif layout_tag == "TriangularMesh":
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
        shading="auto",
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
