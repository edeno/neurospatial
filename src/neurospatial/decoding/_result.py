"""DecodingResult container for Bayesian decoding results.

This module provides the DecodingResult dataclass which stores posterior
distributions from neural decoding and computes derived properties lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from neurospatial._results import ResultMixin

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from neurospatial.environment import Environment


@dataclass
class DecodingResult(ResultMixin):
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
    >>> print(
    ...     f"Mean entropy: {result.posterior_entropy.mean():.2f} bits"
    ... )  # doctest: +SKIP

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

        A row whose posterior is entirely non-finite (all-NaN / all-Inf)
        is undecodable; ``np.argmax`` returns bin 0 for such a row, so the
        MAP index there is not meaningful. Callers that care about decode
        failures (e.g. :meth:`error_against`) detect all-non-finite rows
        separately and treat them as undefined (NaN).

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
    def posterior_entropy(self) -> NDArray[np.float64]:
        """Posterior entropy in bits.

        Measures the entropy in the position estimate. Higher values
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
        map_estimate : Point estimate with zero entropy consideration
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
        >>> result = decode_position(
        ...     env, spike_counts, encoding_models, dt
        ... )  # doctest: +SKIP
        >>> ax = result.plot()  # doctest: +SKIP
        >>> ax = result.plot(colorbar=True, show_map=True)  # doctest: +SKIP
        >>> ax = result.plot(cmap="hot", vmin=0, vmax=0.5)  # doctest: +SKIP

        Notes
        -----
        This method shows a "spectrogram-style" view with time on the x-axis
        and spatial bin indices on the y-axis. This is useful for quick
        diagnostics and 1D linearized tracks.

        For 2D environments, use ``env.animate_fields`` with one timestamp
        per posterior row to visualize the posterior in actual spatial
        coordinates::

            # 2D spatial visualization (recommended for 2D environments)
            env.animate_fields(
                result.posterior,
                frame_times=result.times,
                backend="napari",
            )

            # With position overlay
            from neurospatial.animation.overlays import PositionOverlay

            overlay = PositionOverlay(positions=positions, times=times)
            env.animate_fields(
                result.posterior,
                frame_times=result.times,
                overlays=[overlay],
            )

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

    def summary(self) -> dict[str, Any]:
        """Scalar headline metrics for the decoded posterior.

        Returns
        -------
        dict
            Mapping with keys ``n_time_bins`` (int), ``n_bins`` (int, spatial
            bins), ``mean_entropy`` (float, bits), and ``max_entropy`` (float,
            bits) -- the latter being ``log2(n_bins)``, the entropy of a
            uniform posterior.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.decoding import DecodingResult
        >>> positions = np.random.default_rng(0).uniform(0, 50, (200, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> posterior = np.ones((10, env.n_bins)) / env.n_bins
        >>> result = DecodingResult(posterior=posterior, env=env)
        >>> sorted(result.summary())
        ['max_entropy', 'mean_entropy', 'n_bins', 'n_time_bins']
        """
        n_bins = int(self.posterior.shape[1])
        return {
            "n_time_bins": self.n_time_bins,
            "n_bins": n_bins,
            "mean_entropy": float(np.mean(self.posterior_entropy)),
            "max_entropy": float(np.log2(n_bins)) if n_bins > 0 else 0.0,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with times and position estimates.

        Creates a DataFrame with one row per time bin, containing the
        decoded position estimates and entropy measures.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:

            - ``time`` : Time bin center in seconds (only if times provided)
            - ``map_bin`` : Bin index of maximum posterior probability
            - ``map_x``, ``map_y``, ... : MAP position coordinates
            - ``mean_x``, ``mean_y``, ... : Mean position coordinates
            - ``posterior_entropy`` : Posterior entropy in bits

        Notes
        -----
        Coordinate column naming:

        - For 1D environments: uses ``x``
        - For 2D environments: uses ``x``, ``y``
        - For 3D environments: uses ``x``, ``y``, ``z``
        - For higher dimensions: uses ``dim_0``, ``dim_1``, etc.

        Examples
        --------
        >>> result = decode_position(  # doctest: +SKIP
        ...     env, spike_counts, encoding_models, dt, times=times
        ... )
        >>> df = result.to_dataframe()  # doctest: +SKIP
        >>> df.head()  # doctest: +SKIP

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

        # Add posterior_entropy
        data["posterior_entropy"] = self.posterior_entropy

        return pd.DataFrame(data)

    def to_xarray(self) -> Any:
        """Convert the posterior to a labeled :class:`xarray.Dataset`.

        Wraps the ``(n_time_bins, n_bins)`` posterior in a labeled
        :class:`xarray.Dataset` with dims ``("time", "bin")`` -- a posterior
        over space per decode time bin (no ``unit_id`` axis). The ``time``
        index coordinate is taken from :attr:`times` when available (see Notes
        for the fallback); the ``bin`` dimension carries non-index
        ``bin_center_x`` / ``bin_center_y`` (and ``bin_center_z`` for 3-D, or
        ``bin_center_distance`` / ``bin_center_angle`` for polar)
        coordinates.

        Returns
        -------
        xarray.Dataset
            Dataset with:

            - data var ``posterior``, dims ``("time", "bin")`` (equals
              :attr:`posterior`).
            - data var ``map_position`` per dimension is *not* added; the MAP
              bin index is exposed via the ``map_bin`` data var on ``time``.
            - index coord ``time`` (= :attr:`times`, or the integer-index
              fallback, see Notes).
            - non-index ``bin_center_*`` coords on ``bin``.
            - ``attrs``: ``units``, ``env`` fingerprint, and
              ``software_version``.

        Raises
        ------
        ImportError
            If ``xarray`` is not installed. xarray is an optional dependency;
            install it with ``pip install neurospatial[xarray]`` or
            ``pip install xarray``.

        Notes
        -----
        ``xarray`` is imported lazily inside this method, so it never becomes
        an import-time dependency of ``neurospatial``.

        **Fallback when ``times`` is ``None``.** :attr:`times` defaults to
        ``None`` (the decode produces no timestamps unless the caller supplies
        them). In that case the ``time`` coordinate is the positional integer
        index ``np.arange(n_time_bins)`` rather than ``None`` (passing ``None``
        as a coordinate would raise). The ``time`` coordinate is therefore a
        bin index, not seconds, whenever :attr:`times` is unset.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.decoding import DecodingResult
        >>> positions = np.random.default_rng(0).uniform(0, 50, (200, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> posterior = np.ones((10, env.n_bins)) / env.n_bins
        >>> times = np.linspace(0.0, 1.0, 10)
        >>> result = DecodingResult(posterior=posterior, env=env, times=times)
        >>> ds = result.to_xarray()  # doctest: +SKIP
        >>> ds["posterior"].dims  # doctest: +SKIP
        ('time', 'bin')

        See Also
        --------
        to_dataframe : Export point estimates and entropy to a DataFrame.
        """
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError(
                "to_xarray() requires the optional 'xarray' dependency, which "
                "is not installed. Install it with "
                "'pip install neurospatial[xarray]' or 'pip install xarray'."
            ) from exc

        from neurospatial._results import (
            _bin_center_coords,
            env_fingerprint,
            software_version,
            units_attr,
        )

        n_time = self.posterior.shape[0]
        n_bins = self.posterior.shape[1]

        # times defaults to None; fall back to a positional integer index
        # rather than passing None into the coord (which would raise).
        if self.times is not None:
            time_coord: NDArray[Any] = np.asarray(self.times)
            if time_coord.ndim != 1:
                raise ValueError(
                    "DecodingResult.to_xarray(): times must be 1-D, but got "
                    f"shape {time_coord.shape}.\n"
                    "  WHY: the xarray 'time' coordinate labels one posterior "
                    "row per decoded time bin.\n"
                    "  HOW: pass a 1-D times array with length "
                    "posterior.shape[0], or leave times=None to use integer "
                    "time-bin indices."
                )
            if time_coord.shape[0] != n_time:
                raise ValueError(
                    "DecodingResult.to_xarray(): times length mismatch: got "
                    f"{time_coord.shape[0]} time value(s), but posterior has "
                    f"{n_time} time bin(s).\n"
                    "  WHY: the xarray 'time' coordinate must align one-to-one "
                    "with posterior rows.\n"
                    "  HOW: pass times with length posterior.shape[0], or "
                    "leave times=None to use integer time-bin indices."
                )
        else:
            time_coord = np.arange(n_time)

        coords: dict[str, Any] = {"time": time_coord, "bin": np.arange(n_bins)}
        coords.update(_bin_center_coords(self.env, n_bins))

        data_vars: dict[str, Any] = {
            "posterior": (("time", "bin"), np.asarray(self.posterior)),
            # map_bin is cheaply available (cached); expose the MAP bin index.
            "map_bin": (("time",), np.asarray(self.map_estimate)),
        }

        attrs: dict[str, Any] = {
            **units_attr(self.env),
            "env": env_fingerprint(self.env),
            "software_version": software_version(),
        }

        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    def error_against(
        self,
        true_times: NDArray[np.float64],
        true_positions: NDArray[np.float64],
        *,
        metric: Literal["euclidean", "geodesic"] = "euclidean",
    ) -> NDArray[np.float64]:
        """Per-time-bin decode error against externally sampled ground truth.

        Aligns this decode's time grid to a separately sampled ground-truth
        position track and returns the position error at each decode time bin.
        This removes the hand-rolled ``searchsorted`` alignment that callers
        otherwise write between decode times and behavioral position samples.

        Parameters
        ----------
        true_times : NDArray[np.float64], shape (n_true,)
            Timestamps (seconds) of the ground-truth positions. Must be sorted
            ascending.
        true_positions : NDArray[np.float64], shape (n_true, n_dims)
            Ground-truth positions sampled at ``true_times``. ``n_dims`` must
            match the environment dimensionality.
        metric : {"euclidean", "geodesic"}, default="euclidean"
            Distance metric passed through to
            :func:`neurospatial.decoding.metrics.decoding_error`. ``"geodesic"``
            uses this result's :attr:`env` for graph distances.

        Returns
        -------
        errors : NDArray[np.float64], shape (n_time_bins,)
            Distance between the decoded (MAP) position and the time-aligned
            ground-truth position at each decode time bin. Units match the
            environment (e.g. cm). Decode times whose posterior row is entirely
            non-finite (all-NaN / all-Inf) are undecodable and reported as
            ``nan`` rather than a spurious finite error from bin 0.

        Raises
        ------
        ValueError
            If :attr:`times` is ``None`` (decode times are required to align),
            if ``true_times`` is not 1D, if ``true_times`` is not strictly
            increasing (duplicates are rejected because :func:`numpy.interp`
            resolves duplicate x-values arbitrarily), if ``true_positions``
            does not have shape ``(len(true_times), n_dims)``, or if :attr:`times`,
            ``true_times``, or ``true_positions`` contain any NaN or Inf value
            (which would otherwise yield silent NaN errors through
            :func:`numpy.interp`).

        Notes
        -----
        Alignment uses linear interpolation of each ground-truth coordinate
        onto the decode times via :func:`numpy.interp`. Decode times outside
        the span of ``true_times`` are clamped to the nearest endpoint (the
        ``numpy.interp`` default), so do not extrapolate beyond the tracked
        interval. The decoded position is the MAP estimate
        (:attr:`map_position`); the aligned ground truth is compared against it
        with :func:`neurospatial.decoding.metrics.decoding_error`.

        Linear interpolation of position assumes the ground-truth samples are
        dense enough that no linearization-segment boundary / track junction
        (or circular wrap) is crossed between consecutive ``true_times``;
        otherwise an interpolated midpoint may not correspond to a real
        position.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.decoding import DecodingResult
        >>> positions = np.random.default_rng(0).uniform(0, 50, (200, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> posterior = np.ones((5, env.n_bins)) / env.n_bins
        >>> decode_times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        >>> result = DecodingResult(posterior=posterior, env=env, times=decode_times)
        >>> true_times = np.array([0.0, 1.0])
        >>> true_positions = np.array([[0.0, 0.0], [10.0, 10.0]])
        >>> errors = result.error_against(true_times, true_positions)
        >>> errors.shape
        (5,)

        See Also
        --------
        neurospatial.decoding.metrics.decoding_error : Underlying error core.
        map_position : Decoded MAP position compared against ground truth.
        """
        from neurospatial._validation import validate_finite
        from neurospatial.decoding.metrics import decoding_error

        if self.times is None:
            raise ValueError(
                "error_against() requires decode times, but this "
                "DecodingResult has times=None. Provide `times` when "
                "constructing the result so the decode grid can be aligned "
                "to the ground-truth track."
            )

        # Reject non-finite values up front. A NaN/Inf in any of these arrays
        # would otherwise propagate silently through np.interp and yield NaN
        # errors that look like (but are not) decode failures.
        validate_finite(self.times, name="times")
        true_times = validate_finite(true_times, name="true_times")
        true_positions = validate_finite(true_positions, name="true_positions")

        if true_times.ndim != 1:
            raise ValueError(f"true_times must be 1D, got shape {true_times.shape}")
        if np.any(np.diff(true_times) <= 0):
            raise ValueError(
                "true_times must be strictly increasing (no duplicates); "
                "error_against() aligns ground truth onto decode times with "
                "linear interpolation, which silently produces wrong results "
                "for unsorted or duplicated true_times (np.interp resolves "
                "duplicate x-values arbitrarily). Deduplicate and sort "
                "true_times (and reorder true_positions to match) before "
                "calling."
            )
        n_dims = self.env.n_dims
        if true_positions.shape != (true_times.shape[0], n_dims):
            raise ValueError(
                f"true_positions must have shape ({true_times.shape[0]}, "
                f"{n_dims}) to match true_times and the environment "
                f"dimensionality, got {true_positions.shape}"
            )

        # Align ground truth onto decode times by interpolating each
        # coordinate independently. np.interp clamps out-of-range times to
        # the nearest endpoint.
        decode_times = np.asarray(self.times, dtype=np.float64)
        aligned = np.empty((decode_times.shape[0], n_dims), dtype=np.float64)
        for d in range(n_dims):
            aligned[:, d] = np.interp(decode_times, true_times, true_positions[:, d])

        errors = decoding_error(
            self.map_position,
            aligned,
            env=self.env,
            metric=metric,
        )

        # Undecodable rows: a posterior row that is entirely non-finite
        # (all-NaN / all-Inf) carries no position information. np.argmax would
        # otherwise pick bin 0 and produce a finite, wrong error. Mark these
        # rows NaN so they are clearly flagged as decode failures.
        undecodable = ~np.any(np.isfinite(self.posterior), axis=1)
        if np.any(undecodable):
            errors = np.asarray(errors, dtype=np.float64).copy()
            errors[undecodable] = np.nan
        return errors


@dataclass(frozen=True)
class DecodingSummary(ResultMixin):
    """Per-time decode reductions, without the full posterior.

    Memory-safe sibling of :class:`DecodingResult`. Produced by
    :func:`~neurospatial.decoding.decode_position_summary` (and
    :func:`~neurospatial.decoding.decode_session_summary`), which stream over
    time and never materialize the full ``(n_time_bins, n_bins)`` posterior.
    This container therefore holds only the ``(n_time_bins, ...)`` reductions.

    The accessor *names* deliberately match :class:`DecodingResult`
    (``map_position``, ``mean_position``, ``posterior_entropy``, ``map_bin``)
    so user code ports between the full and summary decoders with no renaming.
    The key difference is that here they are **fields** holding the streamed
    result rather than cached properties computed from a posterior.

    Parameters
    ----------
    times : NDArray[np.float64] | None
        Time-bin centers in seconds, or ``None`` if not supplied.
    map_position : NDArray[np.float64], shape (n_time_bins, n_dims)
        MAP (maximum a posteriori) position per time bin.
    mean_position : NDArray[np.float64], shape (n_time_bins, n_dims)
        Posterior-mean position per time bin.
    posterior_entropy : NDArray[np.float64], shape (n_time_bins,)
        Posterior entropy per time bin, in bits.
    peak_prob : NDArray[np.float64], shape (n_time_bins,)
        Maximum posterior probability per time bin.
    env : Environment
        Reference to the environment used for decoding.
    map_bin : NDArray[np.int64], shape (n_time_bins,)
        MAP bin index per time bin.

    See Also
    --------
    DecodingResult : Full-posterior decode result.
    neurospatial.decoding.decode_position_summary : Produces this container.
    """

    times: NDArray[np.float64] | None
    map_position: NDArray[np.float64]
    mean_position: NDArray[np.float64]
    posterior_entropy: NDArray[np.float64]
    peak_prob: NDArray[np.float64]
    env: Environment
    map_bin: NDArray[np.int64]

    @property
    def n_time_bins(self) -> int:
        """Number of decoded time bins.

        Returns
        -------
        int
            Length of the per-time reduction arrays.
        """
        return int(self.map_bin.shape[0])

    def _dim_names(self) -> list[str]:
        """Coordinate column names matching ``DecodingResult.to_dataframe``."""
        n_dims = self.env.n_dims
        if n_dims <= 3:
            return ["x", "y", "z"][:n_dims]
        return [f"dim_{i}" for i in range(n_dims)]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a tidy DataFrame, one row per time bin.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns:

            - ``time`` : time-bin center in seconds (only if ``times`` set)
            - ``map_bin`` : MAP bin index
            - ``map_x``, ``map_y``, ... : MAP position coordinates
            - ``mean_x``, ``mean_y``, ... : mean position coordinates
            - ``posterior_entropy`` : posterior entropy (bits)
            - ``peak_prob`` : max posterior probability

        Notes
        -----
        Coordinate column naming matches
        :meth:`DecodingResult.to_dataframe` exactly (``x``/``y``/``z`` for
        <=3-D, ``dim_i`` otherwise).
        """
        data: dict[str, Any] = {}

        if self.times is not None:
            data["time"] = self.times

        data["map_bin"] = self.map_bin

        dim_names = self._dim_names()
        for i, name in enumerate(dim_names):
            data[f"map_{name}"] = self.map_position[:, i]
        for i, name in enumerate(dim_names):
            data[f"mean_{name}"] = self.mean_position[:, i]

        data["posterior_entropy"] = self.posterior_entropy
        data["peak_prob"] = self.peak_prob

        return pd.DataFrame(data)

    def summary(self) -> dict[str, Any]:
        """Scalar headline metrics for the streamed decode.

        Returns
        -------
        dict
            Mapping with keys ``n_time_bins`` (int), ``n_bins`` (int, the
            environment's spatial bin count), ``mean_entropy`` (float, bits),
            ``max_entropy`` (float, bits = ``log2(n_bins)``), and
            ``mean_peak_prob`` (float).

        Notes
        -----
        ``n_bins`` is read from ``env.n_bins`` here (there is no posterior to
        read a shape from). The shared keys (``n_time_bins``, ``n_bins``,
        ``mean_entropy``, ``max_entropy``) match
        :meth:`DecodingResult.summary`.
        """
        n_bins = int(self.env.n_bins)
        return {
            "n_time_bins": self.n_time_bins,
            "n_bins": n_bins,
            "mean_entropy": float(np.mean(self.posterior_entropy))
            if self.posterior_entropy.size
            else 0.0,
            "max_entropy": float(np.log2(n_bins)) if n_bins > 0 else 0.0,
            "mean_peak_prob": float(np.mean(self.peak_prob))
            if self.peak_prob.size
            else 0.0,
        }

    def plot(
        self,
        ax: Axes | None = None,
        *,
        quantity: Literal["entropy", "map"] = "entropy",
        **kwargs: Any,
    ) -> Axes:
        """Plot a per-time summary over time.

        Since there is no full posterior to display as a heatmap, this plots a
        per-time scalar over time: either the posterior entropy (default) or
        the MAP position coordinate(s).

        Parameters
        ----------
        ax : matplotlib.axes.Axes | None, optional
            Axes to plot on. If None, a new figure and axes are created.
        quantity : {"entropy", "map"}, default="entropy"
            What to plot: ``"entropy"`` plots ``posterior_entropy`` vs time;
            ``"map"`` plots each MAP position coordinate vs time.
        **kwargs
            Additional keyword arguments forwarded to ``ax.plot``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        if self.times is not None:
            x = np.asarray(self.times, dtype=np.float64)
            x_label = "Time (s)"
        else:
            x = np.arange(self.n_time_bins, dtype=np.float64)
            x_label = "Time bin"

        if quantity == "entropy":
            ax.plot(x, self.posterior_entropy, **kwargs)
            ax.set_ylabel("Posterior entropy (bits)")
            ax.set_title("Posterior entropy over time")
        elif quantity == "map":
            for i, name in enumerate(self._dim_names()):
                ax.plot(x, self.map_position[:, i], label=f"map_{name}", **kwargs)
            ax.set_ylabel("MAP position")
            ax.set_title("MAP position over time")
            ax.legend()
        else:
            raise ValueError(f"quantity must be 'entropy' or 'map', got {quantity!r}.")

        ax.set_xlabel(x_label)
        return ax

    def to_xarray(self) -> Any:
        """Convert the per-time reductions to a labeled :class:`xarray.Dataset`.

        Produces a "track" Dataset with a single ``time`` dimension and **no**
        ``bin`` posterior axis (the full posterior was never materialized).

        Returns
        -------
        xarray.Dataset
            Dataset with data vars on ``("time",)``: ``map_bin``,
            ``map_x``/``map_y``/... , ``mean_x``/..., ``posterior_entropy``,
            ``peak_prob``. The ``time`` index coordinate is :attr:`times` when
            set, else the integer index ``np.arange(n_time_bins)``. ``attrs``
            carry ``units`` (when set), an ``env`` fingerprint, and
            ``software_version``.

        Raises
        ------
        ImportError
            If ``xarray`` is not installed (optional dependency).
        """
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError(
                "to_xarray() requires the optional 'xarray' dependency, which "
                "is not installed. Install it with "
                "'pip install neurospatial[xarray]' or 'pip install xarray'."
            ) from exc

        from neurospatial._results import (
            env_fingerprint,
            software_version,
            units_attr,
        )

        n_time = self.n_time_bins

        if self.times is not None:
            time_coord: NDArray[Any] = np.asarray(self.times)
            if time_coord.ndim != 1 or time_coord.shape[0] != n_time:
                raise ValueError(
                    "DecodingSummary.to_xarray(): times must be 1-D with "
                    f"length n_time_bins ({n_time}), got shape "
                    f"{time_coord.shape}."
                )
        else:
            time_coord = np.arange(n_time)

        dim_names = self._dim_names()
        data_vars: dict[str, Any] = {
            "map_bin": (("time",), np.asarray(self.map_bin)),
            "posterior_entropy": (
                ("time",),
                np.asarray(self.posterior_entropy),
            ),
            "peak_prob": (("time",), np.asarray(self.peak_prob)),
        }
        for i, name in enumerate(dim_names):
            data_vars[f"map_{name}"] = (("time",), self.map_position[:, i])
        for i, name in enumerate(dim_names):
            data_vars[f"mean_{name}"] = (("time",), self.mean_position[:, i])

        attrs: dict[str, Any] = {
            **units_attr(self.env),
            "env": env_fingerprint(self.env),
            "software_version": software_version(),
        }

        return xr.Dataset(
            data_vars=data_vars,
            coords={"time": time_coord},
            attrs=attrs,
        )
