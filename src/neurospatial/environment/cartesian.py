"""Operations that read bin centers as Cartesian ``(x, y[, z])`` coordinates.

``Environment`` composes ``CartesianCoordinates``. An environment whose bins
are not Cartesian, such as ``EgocentricPolarEnvironment``, does not inherit
these methods; accessing one raises ``CartesianOnlyError``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial.environment._coordinates import CartesianSystem
from neurospatial.environment._protocols import SelfCartesianEnv
from neurospatial.environment.decorators import check_fitted
from neurospatial.environment.fields import _interpolate
from neurospatial.environment.trajectory import (
    BinSequenceWithRuns,
    _bin_sequence,
    _occupancy,
)
from neurospatial.environment.transforms import (
    _rebin,
    _subset_mask,
)

if TYPE_CHECKING:
    import shapely

    from neurospatial import Environment
    from neurospatial.ops.transforms import Affine2D, AffineND


_CARTESIAN = CartesianSystem()


class CartesianCoordinates:
    """Entry points for environments whose bin centers are Cartesian coordinates.

    The host interface is ``CartesianEnvironmentProtocol``. Trajectory,
    transform and interpolation implementations live in module functions.
    """

    @property
    def _coordinate_system(self) -> CartesianSystem:
        """Cartesian distance and axis semantics shared by these environments."""
        return _CARTESIAN

    def _require_cartesian(self: SelfCartesianEnv, operation: str) -> SelfCartesianEnv:
        """Return this environment: its bin centers are Cartesian."""
        return self

    @check_fitted
    def bin_at(
        self: SelfCartesianEnv, points_nd: NDArray[np.float64]
    ) -> NDArray[np.intp]:
        """Map N-dimensional points to bins using geometric containment.

        This method uses the layout-specific geometric logic to determine which
        bin contains each point. This is different from nearest-neighbor mapping
        and respects the actual geometry of the bins (grid cells, hexagons, etc.).

        For nearest-neighbor mapping with caching and tie-breaking, see
        `neurospatial.ops.map_points_to_bins()`.

        A track without a declared width retains nearest-track projection:
        a finite point can map to a bin even when far from the centerline.
        Use ``contains`` with a declared ``track_width`` to check its footprint.

        Parameters
        ----------
        points_nd : NDArray[np.float64], shape (n_points, n_dims)
            An array of N-dimensional points to map.

        Returns
        -------
        NDArray[np.intp], shape (n_points,)
            An array of active bin indices (0 to `n_active_bins - 1`).
            A value of -1 indicates that the corresponding point did not map
            to any active bin (e.g., it's outside the environment).

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        ValueError
            If ``points_nd`` has a different dimensionality than the
            environment (e.g. 3-D points for a 2-D environment).

        See Also
        --------
        neurospatial.ops.map_points_to_bins : Nearest-neighbor mapping with KDTree caching

        Notes
        -----
        **Geometric Containment vs Nearest-Neighbor:**

        - `bin_at()` (this method): Uses layout-specific geometric containment
          to determine which bin actually contains the point. For grids, this
          checks which grid cell the point falls into. For hexagons, which
          hexagon contains it. This is exact but may be slower for large batches.

        - `map_points_to_bins()`: Uses KDTree to find the bin whose center is
          closest to the point. This is fast (O(log N)) with caching, supports
          tie-breaking on boundaries, and allows distance thresholds. Best for
          large batches, trajectory processing, and when approximate assignment
          is acceptable.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> x, y = np.meshgrid([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        >>> points = np.array([[2.0, 2.0], [15.0, 15.0]])
        >>> indices = env.bin_at(points)
        >>> indices.tolist()  # -1 denotes the point outside the environment
        [4, -1]

        """
        # Normalize to the portable ``np.intp`` index dtype at the Environment
        # boundary, regardless of what an arbitrary ``LayoutEngine`` returns.
        return self._locate(points_nd)

    @check_fitted
    def contains(
        self: SelfCartesianEnv, points_nd: NDArray[np.float64]
    ) -> NDArray[np.bool_]:
        """Check if N-dimensional continuous points fall within any active bin.

        Parameters
        ----------
        points_nd : NDArray[np.float64], shape (n_points, n_dims)
            An array of N-dimensional points to check.

        Returns
        -------
        NDArray[np.bool_], shape (n_points,)
            A boolean array where `True` indicates the corresponding point
            maps to an active bin, and `False` indicates it does not.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        ValueError
            If ``points_nd`` has a different dimensionality than the
            environment (e.g. 3-D points for a 2-D environment).
        NotImplementedError
            If a linearized track has no declared physical footprint.
            Supply ``track_width`` when constructing the track, or use
            ``bin_at`` to project points without asserting containment.

        Notes
        -----
        Tracks require a declared width: a centerline and topology alone do
        not establish which off-track points are inside. For other layouts,
        containment reuses the bin lookup and checks for the -1 sentinel.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> points = np.array([[5.0, 5.0], [15.0, 15.0]])
        >>> mask = env.contains(points)
        >>> print(mask)  # doctest: +SKIP
        [ True False]  # First point in environment, second outside

        """
        if self.is_linearized_track and not self.arena.domain_known:
            raise NotImplementedError(
                _format_error(
                    "contains() requires a known track footprint.",
                    why="A centerline and bin size do not determine the physical track width.",
                    fix="supply track_width when constructing the track, or use bin_at() for nearest-track projection",
                )
            )
        indices = self._locate(points_nd)
        return np.asarray(indices != -1, dtype=np.bool_)

    @check_fitted
    def distance_between(
        self: SelfCartesianEnv,
        point1: NDArray[np.float64],
        point2: NDArray[np.float64],
        edge_weight: str = "distance",
    ) -> float:
        """Calculate the geodesic distance between two points in the environment.

        Points are first mapped to their nearest active bins using `self.bin_at()`.
        The geodesic distance (distance along the shortest path through the space)
        is then the shortest path length in the `connectivity` graph between these
        bins, using the specified `edge_weight`.

        Parameters
        ----------
        point1 : NDArray[np.float64], shape (n_dims,) or (1, n_dims)
            The first N-dimensional point.
        point2 : NDArray[np.float64], shape (n_dims,) or (1, n_dims)
            The second N-dimensional point.
        edge_weight : str, optional
            The edge attribute to use as weight for path calculation,
            by default "distance". If None, the graph is treated as unweighted.

        Returns
        -------
        float
            The geodesic distance. Returns `np.inf` if points do not map to
            valid active bins, if bins are disconnected, or if the connectivity
            graph is not available.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> p1 = np.array([1.0, 1.0])
        >>> p2 = np.array([9.0, 9.0])
        >>> dist = env.distance_between(p1, p2)
        >>> print(dist > 0)
        True

        """
        source_bin = self.bin_at(np.atleast_2d(point1))[0]
        target_bin = self.bin_at(np.atleast_2d(point2))[0]

        if source_bin == -1 or target_bin == -1:
            # One or both points didn't map to a valid active bin. Returning
            # np.inf silently can mask a coordinate-frame or units mistake, so
            # warn when neither point lands in the environment.
            if source_bin == -1 and target_bin == -1:
                warnings.warn(
                    "distance_between() returning np.inf: neither point mapped "
                    "to an active bin (both bin_at() lookups returned -1). "
                    "Check that the points are inside the environment and in "
                    "the same coordinate frame/units as bin_centers.",
                    category=UserWarning,
                    stacklevel=2,
                )
            return np.inf

        try:
            return float(
                nx.shortest_path_length(
                    self.connectivity,
                    source=source_bin,
                    target=target_bin,
                    weight=edge_weight,
                )
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return np.inf

    @check_fitted
    def to_linear(
        self: SelfCartesianEnv,
        nd_position: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Convert N-D positions to 1D linear coordinates.

        For 1D linearized environments (created with `from_graph()`), this
        converts N-D positions (e.g., 2D x,y coordinates) to their corresponding
        1D linear coordinates along the linearized track.

        Parameters
        ----------
        nd_position : NDArray[np.floating], shape (n_samples, n_dims)
            N-D position array in the original coordinate space.
            For 2D environments, shape is (n_samples, 2).

        Returns
        -------
        NDArray[np.floating], shape (n_samples,)
            1D linear coordinates corresponding to the input positions.

        Raises
        ------
        AttributeError
            If this environment is not 1D (does not support linearization).
            Check `env.is_linearized_track` before calling this method.

        Notes
        -----
        - **Only works for 1D environments** created with `Environment.from_graph()`
        - For N-D environments, use `bin_at()` to map positions to bins instead
        - Linear coordinates start at 0 and increase monotonically along the track

        See Also
        --------
        linear_to_nd : Convert 1D linear coordinates back to N-D positions
        Environment.from_graph : Create 1D linearized environment
        is_linearized_track : Check if environment supports linearization

        Examples
        --------
        >>> import numpy as np
        >>> import networkx as nx
        >>> from neurospatial import Environment
        >>>
        >>> # Create a simple 1D track
        >>> G = nx.Graph()  # doctest: +SKIP
        >>> G.add_node(0, pos=(0.0, 0.0))  # doctest: +SKIP
        >>> G.add_node(1, pos=(10.0, 0.0))  # doctest: +SKIP
        >>> G.add_edge(0, 1, edge_id=0, distance=10.0)  # doctest: +SKIP
        >>> env = Environment.from_graph(
        ...     G, edge_order=[(0, 1)], edge_spacing=0.0, bin_size=2.0
        ... )  # doctest: +SKIP
        >>>
        >>> # Convert 2D position to 1D linear coordinate
        >>> position_2d = np.array(
        ...     [[5.0, 0.0]]
        ... )  # Halfway along track  # doctest: +SKIP
        >>> position_1d = env.to_linear(position_2d)  # doctest: +SKIP
        >>> print(f"Linear position: {position_1d[0]:.1f}")  # doctest: +SKIP
        Linear position: 5.0

        """
        if not self._is_linearized_track_env:
            msg = (
                "to_linear() is only available for 1D environments (GraphLayout). "
                f"This environment is {self.n_dims}D. "
                "Use bin_at() to map positions to bins for N-D environments."
            )
            raise AttributeError(msg)

        linearization = self._binning.linearization
        if linearization is None:
            raise AttributeError(
                _format_error(
                    "to_linear() needs the track graph that defines the linearization, and "
                    f"this environment's {self._binning.kind} layout does not keep it.",
                    why="Why: track environments read from NWB restore bins and connectivity, but not the track graph.",
                    fix="rebuild the environment with Environment.from_graph or Environment.linear_track from the same track",
                )
            )
        return linearization.to_linear(nd_position)

    @check_fitted
    def linear_to_nd(
        self: SelfCartesianEnv,
        linear_position: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Convert 1D linear coordinates to N-D positions.

        For 1D linearized environments (created with `from_graph()`), this
        converts 1D linear coordinates back to the original N-D coordinate
        space (e.g., 2D x,y coordinates).

        Parameters
        ----------
        linear_position : NDArray[np.floating], shape (n_samples,)
            1D linear coordinate array.

        Returns
        -------
        NDArray[np.floating], shape (n_samples, n_dims)
            N-D positions corresponding to the input linear coordinates.
            For 2D environments, shape is (n_samples, 2).

        Raises
        ------
        AttributeError
            If this environment is not 1D (does not support linearization).
            Check `env.is_linearized_track` before calling this method.

        Notes
        -----
        - **Only works for 1D environments** created with `Environment.from_graph()`
        - Inverse operation of `to_linear()`
        - Useful for plotting 1D data in the original 2D space

        See Also
        --------
        to_linear : Convert N-D positions to 1D linear coordinates
        Environment.from_graph : Create 1D linearized environment
        is_linearized_track : Check if environment supports linearization

        Examples
        --------
        >>> import numpy as np
        >>> import networkx as nx
        >>> from neurospatial import Environment
        >>>
        >>> # Create a simple 1D track
        >>> G = nx.Graph()  # doctest: +SKIP
        >>> G.add_node(0, pos=(0.0, 0.0))  # doctest: +SKIP
        >>> G.add_node(1, pos=(10.0, 0.0))  # doctest: +SKIP
        >>> G.add_edge(0, 1, edge_id=0, distance=10.0)  # doctest: +SKIP
        >>> env = Environment.from_graph(
        ...     G, edge_order=[(0, 1)], edge_spacing=0.0, bin_size=2.0
        ... )  # doctest: +SKIP
        >>>
        >>> # Convert 1D linear coordinate to 2D position
        >>> linear_pos = np.array([5.0])  # Halfway along track  # doctest: +SKIP
        >>> position_2d = env.linear_to_nd(linear_pos)  # doctest: +SKIP
        >>> print(f"2D position: {position_2d[0]}")  # doctest: +SKIP
        2D position: [5. 0.]

        """
        if not self._is_linearized_track_env:
            msg = (
                "linear_to_nd() is only available for 1D environments (GraphLayout). "
                f"This environment is {self.n_dims}D. "
                "Use bin_center_of() to get bin centers for N-D environments."
            )
            raise AttributeError(msg)

        linearization = self._binning.linearization
        if linearization is None:
            raise AttributeError(
                _format_error(
                    "linear_to_nd() needs the track graph that defines the linearization, and "
                    f"this environment's {self._binning.kind} layout does not keep it.",
                    why="Why: track environments read from NWB restore bins and connectivity, but not the track graph.",
                    fix="rebuild the environment with Environment.from_graph or Environment.linear_track from the same track",
                )
            )
        return linearization.linear_to_nd(linear_position)

    @check_fitted
    def occupancy(
        self: SelfCartesianEnv,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        speed: NDArray[np.float64] | None = None,
        min_speed: float | None = None,
        max_gap: float | None = 0.5,
        epochs: Any = None,
        bandwidth: float | None = None,
        time_allocation: Literal["start", "linear"] = "start",
        return_seconds: bool = True,
    ) -> NDArray[np.float64]:
        """Compute occupancy (time spent in each bin).

        Accumulates time spent in each bin from continuous trajectory samples.
        Supports optional speed filtering, gap handling, and kernel smoothing.

        Parameters
        ----------
        times : NDArray[np.float64], shape (n_samples,)
            Timestamps in seconds. Must be monotonically increasing.
        positions : NDArray[np.float64], shape (n_samples, n_dims)
            Position coordinates matching environment dimensions.
        speed : NDArray[np.float64], shape (n_samples,), optional
            Instantaneous speed at each sample. If provided with min_speed,
            samples below threshold are excluded from occupancy calculation.
        min_speed : float, optional
            Minimum speed threshold in physical units per second. Requires
            speed parameter. Samples with speed < min_speed are excluded.
        max_gap : float or None, default=0.5
            Longest sampling interval (seconds) treated as continuous recording.
            Longer intervals (dropped frames, pauses between sessions) are excluded
            from occupancy and their spikes are not counted. ``None`` disables the
            gap check.
        epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
            Restrict the analysis to these half-open [start, stop) windows (seconds,
            same clock as ``times``). An interval counts only if it lies entirely
            inside one window. ``None`` (default) means unrestricted.
        bandwidth : float, optional
            If provided, apply diffusion kernel smoothing with this bandwidth
            (in physical units). Uses mode='transition' to preserve total mass.
            Smoothing preserves total occupancy time.
        time_allocation : {'start', 'linear'}, default='start'
            Method for allocating time intervals across bins:

            - 'start': Assign entire Δt to starting bin (fast, works on all layouts).
            - 'linear': Split Δt proportionally across bins traversed by
              straight-line path (grid layouts with rectangular cells).
        return_seconds : bool, default=True
            If True, return time in seconds (time-weighted occupancy).
            If False, return sample counts (unweighted, counts number of
            intervals starting in each bin). Use True for rate calculations
            (e.g., firing rates), False for presence/absence analyses.

        Returns
        -------
        occupancy : NDArray[np.float64], shape (n_bins,)
            If return_seconds=True: Time in seconds spent in each bin.
            If return_seconds=False: Count of intervals starting in each bin.
            The sum equals total valid time (when return_seconds=True) or
            total number of intervals (when return_seconds=False), excluding
            filtered periods and large gaps.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        ValueError
            If times and positions have different lengths, if arrays are
            inconsistent, or if min_speed is provided without speed.
        ValueError
            If positions have wrong number of dimensions.
        ValueError
            If time_allocation is not 'start' or 'linear'.
        ValueError
            If ``times`` contains non-finite values (NaN or Inf). The error
            names ``times`` and the first offending index.
        NotImplementedError
            If linear allocation lacks exact supported cell geometry.

        See Also
        --------
        compute_kernel : Compute diffusion kernel for smoothing.
        bin_at : Map single N-dimensional point to bin index.

        Notes
        -----
        **Time allocation methods**:

        - time_allocation='start' (default): Each time interval Δt is assigned
          entirely to the bin at the starting position. Fast and works on all
          layout types, but may underestimate occupancy in bins the animal
          passed through.

        - time_allocation='linear': Splits Δt proportionally across all bins
          traversed by the straight-line path between consecutive samples.
          Supported for rectangular grids and exact planar polygon cells.
          Polygon allocation retains entry and re-entry intervals, excludes
          holes and outside portions without renormalizing their duration,
          and refuses proper crossings of a known physical wall. Observation
          validity is independent of whether the starting point is in a bin.

        **Mass conservation**: The sum of the returned occupancy array equals
        the total valid time:

        .. math::
            \\sum_i \\text{occupancy}[i] = \\sum_{\\text{valid } k} (t_{k+1} - t_k)

        where valid intervals satisfy:
        - Δt ≤ max_gap (if max_gap is not None)
        - speed[k] ≥ min_speed (if min_speed is not None)
        - positions[k] is inside environment

        Linear allocation works on grids built with ``from_samples``,
        ``from_grid_mask``, ``from_pixel_mask``, ``from_polygon``, and grids
        read from NWB. Inactive cells receive no time.

        For ``time_allocation='linear'`` on a masked grid (one with inactive
        bins), any time the straight-line path spends crossing **inactive**
        bins is dropped rather than reassigned, since the animal cannot occupy
        a bin that does not exist. As a result the linear total may be slightly
        below the ``time_allocation='start'`` total by exactly that
        inactive-bin crossing time.

        **Kernel smoothing**: When bandwidth is provided, smoothing
        is applied after accumulation using mode='transition' normalization
        (kernel columns sum to 1), which preserves the total occupancy mass.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> x, y = np.meshgrid(np.arange(0.0, 22.0, 2.0), np.arange(0.0, 22.0, 2.0))
        >>> env = Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=2.0)
        >>> times = np.array([0.0, 1.0, 2.0, 3.0])
        >>> positions = np.array([[5.0, 5.0], [5.0, 5.0], [10.0, 10.0], [10.0, 10.0]])
        >>> # These 1 Hz demonstration samples are intentionally continuous.
        >>> occ = env.occupancy(times, positions, max_gap=None)
        >>> float(occ.sum())
        3.0
        >>> speeds = np.array([5.0, 5.0, 0.5, 5.0])
        >>> occ_filtered = env.occupancy(
        ...     times,
        ...     positions,
        ...     speed=speeds,
        ...     min_speed=2.0,
        ...     bandwidth=3.0,
        ...     max_gap=None,
        ... )

        """
        return _occupancy(
            self,
            times,
            positions,
            speed=speed,
            min_speed=min_speed,
            max_gap=max_gap,
            epochs=epochs,
            bandwidth=bandwidth,
            time_allocation=time_allocation,
            return_seconds=return_seconds,
        )

    def bin_sequence(
        self: SelfCartesianEnv,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        dedup: bool = True,
        outside_value: int | None = -1,
        max_gap: float | None = 0.5,
        epochs: Any = None,
    ) -> NDArray[np.int32]:
        """Map trajectory to sequence of bin indices.

        Converts a continuous trajectory (times and positions) into a discrete
        sequence of bin indices, with optional deduplication of consecutive
        repeats. To also obtain per-run start indices and lengths, call
        :meth:`bin_sequence_with_runs` instead.

        Parameters
        ----------
        times : NDArray[np.float64], shape (n_samples,)
            Timestamps in seconds. Should be monotonically increasing.
        positions : NDArray[np.float64], shape (n_samples, n_dims)
            Position coordinates matching environment dimensions.
        dedup : bool, default=True
            If True, collapse consecutive repeats: [A,A,A,B] → [A,B].
            If False, return a bin index for every retained sample.
        outside_value : int or None, default=-1
            Bin index for samples outside environment bounds.
            - If -1 (default), outside samples are marked with -1.
            - If None, outside samples are dropped from the sequence entirely.
        max_gap : float or None, default=0.5
            Longest sampling interval (seconds) treated as continuous recording.
            Longer intervals (dropped frames, pauses between sessions) are excluded
            from the analysis. ``None`` disables the gap check.
        epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
            Restrict the analysis to these half-open [start, stop) windows (seconds,
            same clock as ``times``). An interval counts only if it lies entirely
            inside one window. ``None`` (default) means unrestricted.

        Returns
        -------
        bins : NDArray[np.int32], shape (n_sequences,)
            Bin index at each retained time point (or deduplicated sequence).
            Values are in range [0, n_bins-1] for valid bins, or -1 for
            outside samples (when outside_value=-1).

        Raises
        ------
        ValueError
            If times and positions have different lengths, if positions
            have wrong number of dimensions, or if timestamps are not
            monotonically increasing (non-decreasing).

        See Also
        --------
        bin_sequence_with_runs : Same mapping plus run-length boundaries.
        occupancy : Compute time spent in each bin.
        transitions : Build empirical transition matrix from trajectory.

        Notes
        -----
        Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``)
        is analyzed as a separate recording; no segment spans a pause.

        Samples that touch no valid interval are dropped, including singleton
        inputs. Deduplication never merges samples across an invalid interval.
        With dedup=False, one bin is returned per retained sample.

        Timestamps must be monotonically increasing (non-decreasing).
        Sort your data by time before calling this method if needed.

        Examples
        --------
        >>> bins = env.bin_sequence(times, positions)  # doctest: +SKIP
        >>> bins = env.bin_sequence(times, positions, dedup=False)  # doctest: +SKIP
        >>> bins = env.bin_sequence(
        ...     times, positions, outside_value=None
        ... )  # doctest: +SKIP

        """
        result = _bin_sequence(
            self,
            times,
            positions,
            dedup=dedup,
            outside_value=outside_value,
            gap_splits_runs=False,
            max_gap=max_gap,
            epochs=epochs,
        )
        return result.bins

    def bin_sequence_with_runs(
        self: SelfCartesianEnv,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        outside_value: int | None = -1,
        max_gap: float | None = 0.5,
        epochs: Any = None,
    ) -> BinSequenceWithRuns:
        """Map trajectory to bin sequence plus per-run boundaries.

        Returns the run-length encoding of the trajectory so callers
        can recover per-run durations from the original ``times`` array.
        ``bins``, ``run_starts``, and ``run_lengths`` all share shape
        ``(n_runs,)``: the per-run view is intrinsically deduplicated.
        For per-sample bins use :meth:`bin_sequence` with ``dedup=False``.

        Parameters
        ----------
        times : NDArray[np.float64], shape (n_samples,)
            Timestamps in seconds. Should be monotonically increasing.
        positions : NDArray[np.float64], shape (n_samples, n_dims)
            Position coordinates matching environment dimensions.
        outside_value : int or None, default=-1
            Bin index for samples outside environment bounds (see
            :meth:`bin_sequence`). With ``outside_value=None``, outside
            gaps split runs even when consecutive in-env samples land
            in the same bin (so ``run_lengths.sum()`` equals the
            post-filter count of in-env samples).
        max_gap : float or None, default=0.5
            Longest sampling interval (seconds) treated as continuous recording.
            Longer intervals (dropped frames, pauses between sessions) are excluded
            from the analysis. ``None`` disables the gap check.
        epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
            Restrict the analysis to these half-open [start, stop) windows (seconds,
            same clock as ``times``). An interval counts only if it lies entirely
            inside one window. ``None`` (default) means unrestricted.

        Returns
        -------
        BinSequenceWithRuns
            Frozen dataclass with ``bins``, ``run_starts``, and
            ``run_lengths`` fields, all shape ``(n_runs,)``. See
            :class:`BinSequenceWithRuns` for the precise contract.

        See Also
        --------
        bin_sequence : Returns just the bin sequence (supports ``dedup``).

        Notes
        -----
        Each run of samples with gaps no longer than ``max_gap`` (inside ``epochs``)
        is analyzed as a separate recording; no segment spans a pause.

        Samples that touch no valid interval are dropped, including singleton
        inputs. Invalid intervals split runs even if the bins match. Run
        starts and lengths refer to the original input indices, so durations
        computed from them exclude pauses.

        Examples
        --------
        >>> result = env.bin_sequence_with_runs(times, positions)  # doctest: +SKIP
        >>> # Duration of run i in seconds:
        >>> # times[result.run_starts[i] + result.run_lengths[i] - 1]
        >>> #   - times[result.run_starts[i]]
        """
        return _bin_sequence(
            self,
            times,
            positions,
            dedup=True,
            outside_value=outside_value,
            gap_splits_runs=True,
            max_gap=max_gap,
            epochs=epochs,
        )

    def interpolate(
        self: SelfCartesianEnv,
        field: NDArray[np.float64],
        points: NDArray[np.float64],
        *,
        mode: Literal["nearest", "linear"] = "nearest",
    ) -> NDArray[np.float64]:
        """Interpolate field values at arbitrary points.

        Evaluates bin-valued fields at continuous query points using either
        nearest-neighbor or linear interpolation. Nearest mode works on all
        layout types; linear mode requires grid layouts with rectangular cells.

        Parameters
        ----------
        field : NDArray[np.float64], shape (n_bins,)
            Field values per bin. Must be a 1-D array with length equal to n_bins.
            Must not contain NaN or Inf values.
        points : NDArray[np.float64], shape (n_points, n_dims)
            Query points in environment coordinates. Must be a 2-D array where
            each row is a point with dimensionality matching the environment.
        mode : {'nearest', 'linear'}, default='nearest'
            Interpolation mode:

            - 'nearest': Use value of nearest bin center (all layouts).
              Points outside environment bounds return NaN.
            - 'linear': Bilinear (2D) or trilinear (3D) interpolation for
              grids with rectangular cells: ``from_samples``, ``from_grid_mask``,
              ``from_pixel_mask``, ``from_polygon``, and grids read from NWB.
              Inactive cells return NaN.
              Points outside grid bounds return NaN.

        Returns
        -------
        values : NDArray[np.float64], shape (n_points,)
            Interpolated field values. Points outside environment → NaN.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        ValueError
            If field has wrong shape, wrong dimensionality, contains NaN/Inf,
            points have wrong dimensionality, mode is invalid, or dimensions
            don't match.
        NotImplementedError
            If mode='linear' is requested for non-grid layout.

        See Also
        --------
        smooth : Apply diffusion kernel smoothing to fields.
        occupancy : Compute occupancy with optional smoothing.

        Notes
        -----
        **Nearest-neighbor mode**: Uses KDTree to find closest bin center.
        Deterministic and works on all layout types. Points farther than a
        reasonable threshold from any bin center are marked as outside (NaN).

        **Linear mode**: Uses scipy.interpolate.RegularGridInterpolator for
        smooth interpolation on rectangular grids. For linear functions
        f(x,y) = ax + by + c, interpolation is exact up to numerical precision.

        **Outside handling**: Points outside the environment bounds return NaN
        in both modes. This prevents extrapolation errors.

        Examples
        --------
        >>> # Nearest-neighbor interpolation (all layouts)
        >>> field = np.random.rand(env.n_bins)  # doctest: +SKIP
        >>> query_points = np.array([[5.0, 5.0], [7.5, 3.2]])  # doctest: +SKIP
        >>> values = env.interpolate(
        ...     field, query_points, mode="nearest"
        ... )  # doctest: +SKIP

        >>> # Linear interpolation (grids only)
        >>> # For plane f(x,y) = 2x + 3y, interpolation is exact
        >>> plane_field = (
        ...     2 * env.bin_centers[:, 0] + 3 * env.bin_centers[:, 1]
        ... )  # doctest: +SKIP
        >>> values = env.interpolate(
        ...     plane_field, query_points, mode="linear"
        ... )  # doctest: +SKIP

        >>> # Evaluate rate map at trajectory positions
        >>> rates_at_trajectory = env.interpolate(
        ...     rate_map, positions, mode="linear"
        ... )  # doctest: +SKIP

        """
        return _interpolate(self, field, points, mode=mode)

    @check_fitted
    def rebin(
        self: SelfCartesianEnv,
        factor: int | tuple[int, ...],
    ) -> Environment:
        """Coarsen regular grid by integer factor (geometry-only operation).

        Creates a new environment with coarser spatial resolution by reducing
        the number of bins. This method only modifies the grid geometry and
        connectivity; it does not aggregate any field values.

        Only supported for RegularGridLayout environments.

        Parameters
        ----------
        factor : int or tuple of int
            Coarsening factor per dimension. If int, applied uniformly to all
            dimensions. If tuple, must match the number of dimensions.
            Each factor must be a positive integer.

        Returns
        -------
        coarse_env : Environment
            New environment with reduced resolution. The new grid shape is
            ``original_shape // factor`` in each dimension. Keep coarse bins
            whose centers lie inside the original arena.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        NotImplementedError
            If environment layout is not RegularGridLayout.
        ValueError
            If factor is not positive or if factor is too large for grid shape.
            Also raised when coarse cells cannot represent an existing wall.

        See Also
        --------
        smooth : Apply diffusion kernel smoothing to fields.
        subset : Extract spatial subset of environment.
        bin_at : Assign retained original bin centers to coarsened bins.

        Notes
        -----
        **Geometry only**: This method only coarsens the grid structure. To
        aggregate field values (occupancy, spike counts, etc.) from the original
        grid to the coarsened grid, map the original bin centers and aggregate:

            >>> coarse = env.rebin(factor=2)  # doctest: +SKIP
            >>> coarse_indices = coarse.bin_at(env.bin_centers)  # doctest: +SKIP
            >>> retained = coarse_indices >= 0  # doctest: +SKIP
            >>> coarse_field = np.bincount(  # doctest: +SKIP
            ...     coarse_indices[retained],
            ...     weights=field[retained],
            ...     minlength=coarse.n_bins,
            ... )

        **Grid-only operation**: This method only works for environments with
        RegularGridLayout. Other layout types will raise NotImplementedError.

        **Non-divisible dimensions**: If the grid shape is not evenly divisible
        by the factor in any dimension, the grid is truncated to the largest
        multiple of the factor. A warning is issued in this case.

        **Connectivity**: The connectivity graph is rebuilt for the coarsened
        grid with the same connectivity pattern as the original (e.g., if
        original had diagonal connections, coarsened grid will too). Physical
        walls still block connections and must align with complete coarse faces.

        **Bin centers**: New bin centers are computed from the coarsened grid
        edges as midpoints between edge positions.

        **Active bins**: Keep centers strictly inside a polygonal arena, or
        contained by another known arena. Unknown domains retain the full
        coarse grid. Rebinning preserves the arena rather than filling its holes.

        **Metadata preservation**: The arena, units, frame and named regions
        are preserved. Region coordinates and metadata remain intact even when
        truncation removes the bins near a region.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> # Create 10x10 grid (deterministic)
        >>> rng = np.random.RandomState(42)
        >>> data = rng.rand(1000, 2) * 100
        >>> env = Environment.from_samples(data, bin_size=10.0)
        >>> env.layout.grid_shape  # May vary with random data
        (11, 11)
        >>>
        >>> # Coarsen by factor 2 → ~5x5 grid (11 // 2 = 5)
        >>> coarse = env.rebin(factor=2)
        >>> coarse.layout.grid_shape
        (5, 5)
        >>>
        >>> # Anisotropic coarsening with tuple (11 // 2 = 5, 11 // 5 = 2)
        >>> coarse_aniso = env.rebin(factor=(2, 5))
        >>> coarse_aniso.layout.grid_shape
        (5, 2)
        >>>
        >>> # Aggregate a field to the coarsened grid
        >>> occupancy = np.random.rand(env.n_bins) * 100
        >>> coarse_indices = coarse.bin_at(env.bin_centers)
        >>> retained = coarse_indices >= 0
        >>> coarse_occupancy = np.bincount(
        ...     coarse_indices[retained],
        ...     weights=occupancy[retained],
        ...     minlength=coarse.n_bins,
        ... )
        >>> # The total over retained source-bin centers is preserved
        >>> bool(np.isclose(occupancy[retained].sum(), coarse_occupancy.sum()))
        True

        """
        return _rebin(self, factor)

    @check_fitted
    def subset(
        self: SelfCartesianEnv,
        *,
        bins: NDArray[np.bool_] | None = None,
        region_names: Sequence[str] | None = None,
        polygon: shapely.Polygon | None = None,
        invert: bool = False,
    ) -> Environment:
        """Create new environment containing subset of bins.

        Extracts a subgraph from the environment containing only the selected
        bins. Node indices are renumbered to be contiguous [0, n'-1]. This
        operation drops all regions; users can re-add regions to the subset
        environment if needed. Bin sizes, retained faces, cell shapes and track
        coordinates are preserved. Points in discarded bins are outside the
        subset and map to ``-1``.

        Parameters
        ----------
        bins : NDArray[np.bool_], shape (n_bins,), optional
            Boolean mask of bins to keep. True = keep, False = discard.
        region_names : Sequence[str], optional
            Keep bins whose centers lie inside these named regions.
            Regions must exist in self.regions. Only polygon-type regions
            are supported (point-type regions will raise ValueError).
        polygon : shapely.Polygon, optional
            Keep bins whose centers lie inside this polygon. Only works
            for 2D environments.
        invert : bool, default=False
            If True, invert the selection mask (select complement).

        Returns
        -------
        sub_env : Environment
            New environment with selected bins renumbered to [0, n'-1].
            Connectivity is the induced subgraph. All regions are dropped.
            Metadata (units, frame) is preserved.

        Raises
        ------
        ValueError
            If none or multiple selection parameters provided, if mask has
            wrong shape/dtype, if region names don't exist, if selection is empty.

        Notes
        -----
        Exactly one of {bins, region_names, polygon} must be provided.

        The connectivity graph is the induced subgraph: only edges where both
        endpoints are in the selection are kept. This may create disconnected
        components if the selection is not contiguous.

        Node attributes ('pos', 'source_grid_flat_index', 'original_grid_nd_index')
        describe the selected bins, and edge attributes ('distance', 'vector',
        'angle_2d') are preserved from the original graph. Grid subsets number
        'edge_id' from 0; other subsets keep the original ids. A grid subset
        keeps the parent's edges even where they differ from plain cell
        adjacency, such as a 4-connected grid read from NWB.

        Subsetting selects whole bins without clipping their geometry. The child
        keeps smoothing, plotting, interpolation and serialization wherever its
        retained components support them. Track linear coordinates remain
        absolute in the original track; dropped bins leave gaps in its mask.

        See Also
        --------
        rebin : Coarsen grid resolution (grid-only).
        components : Find connected components.

        Examples
        --------
        >>> # Extract bins inside 'goal' region
        >>> goal_env = env.subset(region_names=["goal"])  # doctest: +SKIP
        >>>
        >>> # Crop to polygon
        >>> from shapely.geometry import box  # doctest: +SKIP
        >>> cropped = env.subset(polygon=box(0, 0, 50, 50))  # doctest: +SKIP
        >>>
        >>> # Select bins by boolean mask
        >>> mask = env.bin_centers[:, 0] < 50  # Left half  # doctest: +SKIP
        >>> left_env = env.subset(bins=mask)  # doctest: +SKIP
        >>>
        >>> # Inverted selection (everything except region)
        >>> outside = env.subset(
        ...     region_names=["obstacle"], invert=True
        ... )  # doctest: +SKIP

        """
        mask = _subset_mask(
            self, bins=bins, region_names=region_names, polygon=polygon, invert=invert
        )
        from neurospatial.layout._binning_layout import BinningLayout
        from neurospatial.layout._components import HexagonShapes, TriangleShapes
        from neurospatial.layout._derive import subset_binning

        binning = subset_binning(self._binning, mask)
        selected = np.flatnonzero(mask)
        child_of = {int(old): new for new, old in enumerate(selected)}
        graph = nx.Graph()
        graph.graph.update(self.connectivity.graph)
        # Retain the parent's node iteration order as well as edge order and
        # orientation; edge fields index that order even on loaded graphs.
        for old, data in self.connectivity.nodes(data=True):
            if old in child_of:
                graph.add_node(child_of[old], **data)
        for index, (u, v) in enumerate(binning.bins.pairs):
            old_u, old_v = selected[u], selected[v]
            data = dict(self.connectivity.edges[old_u, old_v])
            if binning.rectilinear is not None:
                data["edge_id"] = index
            graph.add_edge(int(u), int(v), **data)
        params = {"source": "subset", "original_n_bins": self.n_bins}
        tag = "MaskedGrid" if binning.rectilinear is not None else "subset"
        if binning.rectilinear is not None:
            ranges = [
                (float(edge[0]), float(edge[-1])) for edge in binning.rectilinear.edges
            ]
        else:
            shapes = binning.shapes
            vertices = binning.bins.centers
            if isinstance(shapes, HexagonShapes):
                angles = shapes.orientation + np.pi / 2 + np.arange(6) * np.pi / 3
                offsets = shapes.radius * np.column_stack(
                    (np.cos(angles), np.sin(angles))
                )
                vertices = (shapes.centers[:, None, :] + offsets).reshape(-1, 2)
            elif isinstance(shapes, TriangleShapes):
                vertices = shapes.points[shapes.triangles].reshape(-1, 2)
            else:
                from neurospatial.layout._polygon import PolygonCells

                if isinstance(shapes, PolygonCells):
                    import shapely

                    vertices = np.concatenate(
                        [shapely.get_coordinates(cell) for cell in shapes.polygons]
                    )
            ranges = []
            for axis in vertices.T:
                low, high = float(axis.min()), float(axis.max())
                margin = 0.05 * (high - low) if shapes is not None else 0.0
                ranges.append((low - margin, high + margin))
        layout = BinningLayout(
            binning,
            connectivity=graph,
            dimension_ranges=ranges,
            layout_type_tag=tag,
            layout_type="mask" if binning.rectilinear is not None else "subset",
            build_params=params,
            is_linearized_track=self.is_linearized_track,
        )
        env_cls = cast("type[Environment]", self.__class__)
        result = env_cls(
            "",
            layout,
            tag,
            params,
            arena=self.arena,
            units=self.units,
            frame=self.frame,
        )
        return result

    @check_fitted
    def clip(
        self: SelfCartesianEnv,
        polygon: shapely.Polygon | shapely.MultiPolygon,
        *,
        name: str | None = None,
    ) -> Environment:
        """Clip planar cells to a polygon, returning a new environment.

        Parameters
        ----------
        polygon : Polygon or MultiPolygon
            Retained Cartesian analysis area, including any holes.
        name : str, optional
            Output name; defaults to the parent's name.

        Returns
        -------
        Environment
            Exact positive-area cells. Disconnected pieces get separate bins;
            ``clip_mapping.parent_bin_indices`` identifies their input bins.
            Rates, counts and fitted models are not transferred. Numerical
            operations require certified finite-volume geometry. The analysis
            crop preserves the parent arena and does not assert new walls.

        Raises
        ------
        ValueError
            If geometry is invalid, unavailable, unsupported, or has no
            positive-area intersection. Track, polar and 3-D cells are not
            Cartesian planar area cells.
        """
        from neurospatial.environment._clipping import _clip

        return _clip(self, polygon, name=name)

    @check_fitted
    def apply_transform(
        self: SelfCartesianEnv,
        transform: AffineND | Affine2D,
        *,
        name: str | None = None,
    ) -> Environment:
        """Apply affine transformation to environment, returning a new instance.

        Transforms the environment's bin centers, connectivity graph, and regions
        using an affine transformation. Supports both 2D (Affine2D) and N-D
        (AffineND) transforms. The transformation must match the environment's
        dimensionality.

        Translation and similarity transforms retain finite-volume smoothing
        and differential operators. Positive diagonal scaling also retains grid
        faces and rectilinear interpolation, with updated edges. Reflections
        preserve bin identity through inverse lookup and drop rectilinear
        interpolation. General affine maps keep containment and scale cell
        measures; smoothing refuses when the map loses the right angles needed
        by finite-volume faces. Triangle plots retain their exact mapped vertices
        under every invertible affine map. Tracks accept translation and
        similarity only.

        Parameters
        ----------
        transform : AffineND or Affine2D
            Affine transformation to apply. Must match environment dimensionality:

            - For 2D environments: use Affine2D or AffineND with n_dims=2
            - For 3D environments: use AffineND with n_dims=3
            - For N-D environments: use AffineND with n_dims=N

            Create transforms using factory functions like `translate()`,
            `scale_2d()`, `from_rotation_matrix()`, or compose using `@` operator.
        name : str, optional
            Name for the transformed environment. If None, appends "_transformed"
            to the original name.

        Returns
        -------
        transformed_env : Environment
            New Environment instance with transformed coordinates. The transformation
            is applied to:

            - `bin_centers`: All bin positions are transformed
            - `connectivity`: Node 'pos' attributes updated, edge distances/vectors recomputed
            - `regions`: Point and polygon regions are transformed
            - `metadata`: Units preserved, frame updated with "_transformed" suffix

            All other properties are copied from the source environment.

        Raises
        ------
        RuntimeError
            If environment is not fitted (use factory methods like
            `Environment.from_samples()`).
        ValueError
            If transform dimensionality doesn't match environment dimensionality,
            the transform is not invertible, or a track receives a non-similarity
            transform. Rebuild such a track from transformed nodes with
            ``Environment.from_graph``.

        See Also
        --------
        estimate_transform : Estimate transformation from point correspondences
        neurospatial.ops.transforms.Affine2D : 2D affine transformation class
        neurospatial.ops.transforms.AffineND : N-D affine transformation class
        neurospatial.ops.transforms.translate : Create translation transform
        neurospatial.ops.transforms.scale_2d : Create scaling transform
        neurospatial.ops.transforms.from_rotation_matrix : Create rotation transform

        Notes
        -----
        **Pure function**: This method does not modify the source environment;
        it returns a new Environment instance.

        **Transformation order**: When composing transforms, use the `@` operator.
        `T1 @ T2` applies T2 first, then T1:

            >>> composed = (
            ...     rotation @ translation
            ... )  # translation first, then rotation  # doctest: +SKIP
            >>> transformed = env.apply_transform(composed)  # doctest: +SKIP

        **Edge attributes**: After transformation:

        - `distance`: Recomputed from transformed positions
        - `vector`: Recomputed as displacement between transformed positions
        - `angle_2d`: Recomputed for 2D environments only

        **Dimension ranges**: Bounding box is recomputed by transforming all
        corner points of the original bounding box.

        Examples
        --------
        Translation (2D):

        >>> from neurospatial import Environment
        >>> from neurospatial.ops.transforms import translate
        >>> import numpy as np
        >>> # Create 2D environment
        >>> data = np.random.rand(200, 2) * 100
        >>> env = Environment.from_samples(data, bin_size=5.0, name="session1")
        >>> # Translate by (10, 20) cm
        >>> transform = translate(10, 20)
        >>> env_shifted = env.apply_transform(transform, name="session1_aligned")
        >>> # Bin centers are translated
        >>> np.allclose(env_shifted.bin_centers, env.bin_centers + [10, 20])
        True

        Rotation (2D):

        >>> from neurospatial.ops.transforms import Affine2D
        >>> # 45-degree rotation
        >>> angle = np.pi / 4
        >>> R = np.array(
        ...     [
        ...         [np.cos(angle), -np.sin(angle), 0],
        ...         [np.sin(angle), np.cos(angle), 0],
        ...         [0, 0, 1],
        ...     ]
        ... )
        >>> transform = Affine2D(R)
        >>> env_rotated = env.apply_transform(transform)
        >>> # Distances preserved under rotation
        >>> assert env_rotated.n_bins == env.n_bins

        Composed transforms (scale → rotate → translate):

        >>> from neurospatial.ops.transforms import (
        ...     scale_2d,
        ...     translate,
        ... )  # doctest: +SKIP
        >>> # Build transformation pipeline
        >>> T = translate(50, 50) @ Affine2D(R) @ scale_2d(1.2)  # doctest: +SKIP
        >>> env_aligned = env.apply_transform(T, name="aligned")  # doctest: +SKIP

        Cross-session alignment using landmarks:

        >>> from neurospatial.ops.transforms import estimate_transform  # doctest: +SKIP
        >>> # Session 1 landmarks (e.g., arena corners)
        >>> landmarks_s1 = np.array(
        ...     [[0, 0], [100, 0], [100, 100], [0, 100]]
        ... )  # doctest: +SKIP
        >>> # Session 2 landmarks (same physical locations, different coordinates)
        >>> landmarks_s2 = np.array(
        ...     [[5, 10], [95, 15], [90, 105], [0, 100]]
        ... )  # doctest: +SKIP
        >>> # Estimate rigid transform (rotation + translation)
        >>> T = estimate_transform(
        ...     landmarks_s1, landmarks_s2, kind="rigid"
        ... )  # doctest: +SKIP
        >>> # Transform session 1 environment to session 2 coordinates
        >>> env_s1_aligned = env_s1.apply_transform(
        ...     T, name="session1_in_s2_coords"
        ... )  # doctest: +SKIP

        3D transformation:

        >>> from scipy.spatial.transform import Rotation  # doctest: +SKIP
        >>> from neurospatial.ops.transforms import (
        ...     from_rotation_matrix,
        ...     translate_3d,
        ... )  # doctest: +SKIP
        >>> # Create 3D environment
        >>> data_3d = np.random.randn(500, 3) * 20  # doctest: +SKIP
        >>> env_3d = Environment.from_samples(data_3d, bin_size=3.0)  # doctest: +SKIP
        >>> # Rotate 45 degrees around z-axis and translate
        >>> R_3d = Rotation.from_euler(
        ...     "z", 45, degrees=True
        ... ).as_matrix()  # doctest: +SKIP
        >>> rotation = from_rotation_matrix(R_3d)  # doctest: +SKIP
        >>> translation = translate_3d(10, 20, 30)  # doctest: +SKIP
        >>> T_3d = translation @ rotation  # doctest: +SKIP
        >>> env_3d_transformed = env_3d.apply_transform(T_3d)  # doctest: +SKIP

        With regions (regions are automatically transformed):

        >>> from neurospatial.regions import Regions
        >>> regions = Regions(env.regions.values())
        >>> regions.add("goal", point=[80, 90])  # doctest: +SKIP
        >>> env = env.with_regions(regions)
        >>> env_transformed = env.apply_transform(translate(10, 10))  # doctest: +SKIP
        >>> # Region is transformed along with environment
        >>> goal_region = env_transformed.regions["goal"]  # doctest: +SKIP
        >>> np.allclose(goal_region.data, [90, 100])  # doctest: +SKIP
        True

        """
        # Import here to avoid circular dependency
        # (transforms.py imports Environment, which imports this mixin)
        # Delegate to free function
        # Cast self to Environment for type checking
        from neurospatial.ops.transforms import (
            apply_transform_to_environment as _apply_transform_impl,
        )

        env = cast("Environment", self)

        return _apply_transform_impl(env, transform, name=name)


CARTESIAN_OPERATIONS: frozenset[str] = frozenset(
    name for name in vars(CartesianCoordinates) if not name.startswith("_")
)
"""Public methods that only Cartesian environments define."""

CARTESIAN_GRID_OPERATIONS: frozenset[str] = frozenset(
    {"to_linear", "linear_to_nd", "rebin", "subset"}
)
"""Cartesian operations that also assume a rectangular grid or a track."""
