"""DecodingResult container for Bayesian decoding results.

This module provides the DecodingResult dataclass which stores posterior
distributions from neural decoding and computes derived properties lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from neurospatial.environment import Environment


@dataclass
class DecodingResult:
    """Container for Bayesian decoding results.

    Stores the posterior distribution over positions for each time bin,
    along with a reference to the environment used for decoding.
    Derived quantities (MAP estimate, mean position, entropy) are computed
    lazily via cached properties on first access.

    Parameters
    ----------
    posterior : NDArray[np.float64]
        Posterior probability distribution over positions, shape
        (n_time_bins, n_bins). Each row sums to 1.0.
    env : Environment
        Reference to the environment used for decoding. Provides
        bin_centers for coordinate transforms.
    times : NDArray[np.float64] | None, optional
        Time bin centers in seconds. If provided, used for plotting
        and DataFrame export. Default is None.

    Attributes
    ----------
    posterior : NDArray[np.float64]
        Primary data: posterior probability distribution.
    env : Environment
        Reference to environment for coordinate transforms.
    times : NDArray[np.float64] | None
        Optional time bin centers (seconds).

    Examples
    --------
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding import DecodingResult
    >>> import numpy as np
    >>>
    >>> # Create environment
    >>> positions = np.random.uniform(0, 50, (1000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>>
    >>> # Create uniform posterior (for demonstration)
    >>> n_time_bins = 100
    >>> posterior = np.ones((n_time_bins, env.n_bins)) / env.n_bins
    >>>
    >>> # Create result
    >>> result = DecodingResult(posterior=posterior, env=env)
    >>> print(f"MAP estimate shape: {result.map_estimate.shape}")
    MAP estimate shape: (100,)
    >>> print(f"Mean uncertainty: {result.uncertainty.mean():.2f} bits")

    Notes
    -----
    The class uses ``@dataclass`` (not frozen) to allow ``@cached_property``.
    The class is effectively immutable since modifying ``posterior`` or ``env``
    after construction would invalidate cached properties without clearing them.

    Memory usage is dominated by the posterior array:
    ``n_time_bins * n_bins * 8 bytes`` (float64).

    See Also
    --------
    neurospatial.decoding.decode_position : Create DecodingResult from spikes
    """

    posterior: NDArray[np.float64]
    env: Environment
    times: NDArray[np.float64] | None = None

    @property
    def n_time_bins(self) -> int:
        """Number of time bins in the posterior.

        Returns
        -------
        int
            Number of rows in the posterior array.
        """
        return int(self.posterior.shape[0])

    @cached_property
    def map_estimate(self) -> NDArray[np.int64]:
        """Maximum a posteriori bin index for each time bin.

        Returns the index of the bin with highest posterior probability
        at each time step.

        Returns
        -------
        NDArray[np.int64]
            Bin indices of shape (n_time_bins,).

        Notes
        -----
        Uses ``np.argmax(axis=1)`` which returns the first maximum
        in case of ties.

        See Also
        --------
        map_position : MAP position in environment coordinates
        """
        result: NDArray[np.int64] = np.argmax(self.posterior, axis=1).astype(np.int64)
        return result

    @cached_property
    def map_position(self) -> NDArray[np.float64]:
        """MAP position in environment coordinates.

        Returns the coordinates of the bin with highest posterior probability
        at each time step.

        Returns
        -------
        NDArray[np.float64]
            Positions of shape (n_time_bins, n_dims).

        See Also
        --------
        map_estimate : MAP bin index
        mean_position : Posterior mean position
        """
        return self.env.bin_centers[self.map_estimate]

    @cached_property
    def mean_position(self) -> NDArray[np.float64]:
        """Posterior mean position (expected value).

        Computes the probability-weighted average of bin center coordinates.

        Returns
        -------
        NDArray[np.float64]
            Mean positions of shape (n_time_bins, n_dims).

        Notes
        -----
        Computed as: ``posterior @ bin_centers``

        For unimodal posteriors, this is similar to the MAP position.
        For multimodal posteriors, this may fall between modes.

        See Also
        --------
        map_position : MAP position (mode of posterior)
        """
        return self.posterior @ self.env.bin_centers

    @cached_property
    def uncertainty(self) -> NDArray[np.float64]:
        """Posterior entropy in bits.

        Measures the uncertainty in the position estimate. Higher values
        indicate more spread-out (uncertain) posteriors.

        Returns
        -------
        NDArray[np.float64]
            Entropy values of shape (n_time_bins,).
            Range: [0, log2(n_bins)].

        Notes
        -----
        Uses mask-based computation to avoid bias from exact zeros:

        .. math::

            H = -\\sum_{i: p_i > 0} p_i \\log_2(p_i)

        This is more accurate than global clipping to ``[1e-10, 1]`` which
        can slightly bias entropy upward when many exact zeros occur.

        Maximum entropy (uniform distribution) is ``log2(n_bins)``.
        Minimum entropy (delta distribution) is 0.

        See Also
        --------
        map_estimate : Point estimate with zero uncertainty consideration
        """
        p = np.clip(self.posterior, 0.0, 1.0)
        # Vectorized mask-based entropy: avoid log(0) by using np.where
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = np.where(p > 0, np.log2(p), 0.0)
        entropy: NDArray[np.float64] = cast(
            "NDArray[np.float64]", -np.sum(p * log_p, axis=1)
        )
        return entropy

    def plot(
        self,
        ax: Axes | None = None,
        *,
        colorbar: bool = False,
        show_map: bool = False,
        cmap: str = "viridis",
        **kwargs: Any,
    ) -> Axes:
        """Plot posterior probability over time as heatmap.

        Visualizes the posterior probability distribution as a 2D heatmap
        with time on the x-axis and spatial bins on the y-axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes | None, optional
            Axes to plot on. If None, creates new figure and axes.
        colorbar : bool, default=False
            If True, add a colorbar showing probability scale.
        show_map : bool, default=False
            If True, overlay the MAP (maximum a posteriori) trajectory
            as a line plot on top of the heatmap.
        cmap : str, default="viridis"
            Colormap name for the heatmap.
        **kwargs
            Additional keyword arguments passed to ``imshow``.
            Common options: ``vmin``, ``vmax``, ``interpolation``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.

        Examples
        --------
        >>> result = decode_position(env, spike_counts, encoding_models, dt)
        >>> ax = result.plot()  # Basic heatmap

        >>> # With colorbar and MAP overlay
        >>> ax = result.plot(colorbar=True, show_map=True)

        >>> # With custom colormap
        >>> ax = result.plot(cmap="hot", vmin=0, vmax=0.5)

        Notes
        -----
        This method shows a "spectrogram-style" view with time on the x-axis
        and spatial bin indices on the y-axis. This is useful for quick
        diagnostics and 1D linearized tracks.

        For 2D environments, use ``env.animate_fields(result.posterior)``
        instead to visualize the posterior in actual spatial coordinates::

            # 2D spatial visualization (recommended for 2D environments)
            env.animate_fields(result.posterior, backend="napari")

            # With position overlay
            from neurospatial.animation.overlays import PositionOverlay

            overlay = PositionOverlay(data=positions, times=times)
            env.animate_fields(result.posterior, overlays=[overlay])

        When ``times`` is provided, the x-axis shows time in seconds with
        proper extent. Otherwise, the x-axis shows time bin indices.

        The MAP trajectory (``show_map=True``) shows the bin with highest
        posterior probability at each time step as a white line.

        See Also
        --------
        to_dataframe : Export results to pandas DataFrame
        neurospatial.Environment.animate_fields : Animate posterior in 2D space
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        # Compute extent for proper axis labeling
        # extent = [left, right, bottom, top]
        if self.times is not None:
            # Use actual time values
            t_min = float(self.times[0])
            t_max = float(self.times[-1])
            extent = [t_min, t_max, -0.5, self.posterior.shape[1] - 0.5]
            x_label = "Time (s)"
        else:
            # Use bin indices
            extent = [
                -0.5,
                self.posterior.shape[0] - 0.5,
                -0.5,
                self.posterior.shape[1] - 0.5,
            ]
            x_label = "Time bin"

        # Build imshow kwargs
        im_kwargs: dict[str, Any] = {
            "aspect": "auto",
            "origin": "lower",
            "interpolation": "nearest",
            "cmap": cmap,
            "extent": extent,
        }
        im_kwargs.update(kwargs)

        # Plot the heatmap
        im = ax.imshow(self.posterior.T, **im_kwargs)

        # Add colorbar if requested
        if colorbar:
            plt.colorbar(im, ax=ax, label="Probability")

        # Add MAP trajectory overlay if requested
        if show_map:
            # Get time coordinates for plotting
            if self.times is not None:
                x_coords: NDArray[np.float64] = self.times
            else:
                x_coords = np.arange(self.n_time_bins, dtype=np.float64)

            # Plot MAP estimate as a line
            ax.plot(
                x_coords,
                self.map_estimate,
                color="white",
                linewidth=1.5,
                linestyle="-",
                alpha=0.8,
                label="MAP",
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Spatial bin")
        ax.set_title("Posterior probability")

        return ax

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with times and position estimates.

        Creates a DataFrame with one row per time bin, containing the
        decoded position estimates and uncertainty measures.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:

            - ``time`` : Time bin center in seconds (only if times provided)
            - ``map_bin`` : Bin index of maximum posterior probability
            - ``map_x``, ``map_y``, ... : MAP position coordinates
            - ``mean_x``, ``mean_y``, ... : Mean position coordinates
            - ``uncertainty`` : Posterior entropy in bits

        Notes
        -----
        Coordinate column naming:

        - For 1D environments: uses ``x``
        - For 2D environments: uses ``x``, ``y``
        - For 3D environments: uses ``x``, ``y``, ``z``
        - For higher dimensions: uses ``dim_0``, ``dim_1``, etc.

        Examples
        --------
        >>> result = decode_position(
        ...     env, spike_counts, encoding_models, dt, times=times
        ... )
        >>> df = result.to_dataframe()
        >>> df.head()
           time  map_bin   map_x   map_y  mean_x  mean_y  uncertainty
        0  0.000       12  25.5    32.5   24.8    31.2         2.45
        1  0.025       15  32.5    40.5   33.1    39.8         1.89

        See Also
        --------
        plot : Visualize posterior as heatmap
        """
        data: dict[str, Any] = {}

        # Add times if available
        if self.times is not None:
            data["time"] = self.times

        # Add MAP bin index
        data["map_bin"] = self.map_estimate

        # Add MAP positions
        n_dims = self.env.n_dims
        dim_names: list[str] = (
            ["x", "y", "z"][:n_dims]
            if n_dims <= 3
            else [f"dim_{i}" for i in range(n_dims)]
        )
        for i, name in enumerate(dim_names):
            data[f"map_{name}"] = self.map_position[:, i]

        # Add mean positions
        for i, name in enumerate(dim_names):
            data[f"mean_{name}"] = self.mean_position[:, i]

        # Add uncertainty
        data["uncertainty"] = self.uncertainty

        return pd.DataFrame(data)
