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
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.axes

from neurospatial.environment.decorators import check_fitted

if TYPE_CHECKING:
    from neurospatial.environment._protocols import EnvironmentProtocol


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
        self: EnvironmentProtocol,
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
        self: EnvironmentProtocol,
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
