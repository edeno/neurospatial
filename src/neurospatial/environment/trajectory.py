"""Trajectory analysis methods for Environment class.

This module provides methods for analyzing trajectory data, including
occupancy computation, bin sequence extraction, and transition matrix calculation.

Key Features
------------
- Occupancy (time-in-bin) computation with speed filtering
- Bin sequence extraction with run-length encoding
- Transition matrix calculation (empirical, random walk, diffusion)
- Linear time allocation for grid environments

Notes
-----
This is a mixin class designed to be used with Environment. It should NOT
be decorated with @dataclass. Only the main Environment class in core.py
should be a dataclass.

TYPE_CHECKING Pattern
---------------------
To avoid circular imports, we import Environment only for type checking.

"""

from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING, Literal, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import scipy.sparse

    from neurospatial import Environment
    from neurospatial.environment._protocols import EnvironmentProtocol


class EnvironmentTrajectory:
    """Trajectory analysis methods mixin.

    Provides methods for analyzing trajectories through environments.
    """

    def occupancy(
        self: EnvironmentProtocol,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        speed: NDArray[np.float64] | None = None,
        min_speed: float | None = None,
        max_gap: float | None = 0.5,
        kernel_bandwidth: float | None = None,
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
        max_gap : float, optional
            Maximum time gap in seconds. Intervals with Δt > max_gap are
            not counted toward occupancy. Default: 0.5 seconds. Set to None
            to count all intervals regardless of gap size.
        kernel_bandwidth : float, optional
            If provided, apply diffusion kernel smoothing with this bandwidth
            (in physical units). Uses mode='transition' to preserve total mass.
            Smoothing preserves total occupancy time.
        time_allocation : {'start', 'linear'}, default='start'
            Method for allocating time intervals across bins:

            - 'start': Assign entire Δt to starting bin (fast, works on all layouts).
            - 'linear': Split Δt proportionally across bins traversed by
              straight-line path (more accurate, RegularGridLayout only).
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
        NotImplementedError
            If time_allocation='linear' is used on non-RegularGridLayout.

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
          More accurate for trajectories that cross multiple bins, but only
          supported on RegularGridLayout. Requires ray-grid intersection
          calculations.

        **Mass conservation**: The sum of the returned occupancy array equals
        the total valid time:

        .. math::
            \\sum_i \\text{occupancy}[i] = \\sum_{\\text{valid } k} (t_{k+1} - t_k)

        where valid intervals satisfy:
        - Δt ≤ max_gap (if max_gap is not None)
        - speed[k] ≥ min_speed (if min_speed is not None)
        - positions[k] is inside environment

        **Kernel smoothing**: When kernel_bandwidth is provided, smoothing
        is applied after accumulation using mode='transition' normalization
        (kernel columns sum to 1), which preserves the total occupancy mass.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> # Create environment
        >>> data = np.array([[0, 0], [20, 20]])
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>>
        >>> # Basic occupancy
        >>> times = np.array([0.0, 1.0, 2.0, 3.0])
        >>> positions = np.array([[5, 5], [5, 5], [10, 10], [10, 10]])
        >>> occ = env.occupancy(times, positions)
        >>> occ.sum()  # Total time = 3.0 seconds
        3.0
        >>>
        >>> # Filter slow periods and smooth
        >>> speeds = np.array([5.0, 5.0, 0.5, 5.0])
        >>> occ_filtered = env.occupancy(
        ...     times, positions, speed=speeds, min_speed=2.0, kernel_bandwidth=3.0
        ... )

        """
        from neurospatial.spatial import map_points_to_bins

        # Input validation
        times = np.asarray(times, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64)

        # Validate monotonicity of timestamps
        if len(times) > 1 and not np.all(np.diff(times) >= 0):
            decreasing_indices = np.where(np.diff(times) < 0)[0]
            raise ValueError(
                "times must be monotonically increasing (non-decreasing). "
                f"Found {len(decreasing_indices)} decreasing interval(s) at "
                f"indices: {decreasing_indices.tolist()[:5]}"  # Show first 5
                + (" ..." if len(decreasing_indices) > 5 else "")
            )

        # Check array shapes
        if times.ndim != 1:
            raise ValueError(
                f"times must be 1-dimensional array, got shape {times.shape}"
            )

        if positions.ndim != 2:
            raise ValueError(
                f"positions must be 2-dimensional array (n_samples, n_dims), "
                f"got shape {positions.shape}"
            )

        if len(times) != len(positions):
            raise ValueError(
                f"times and positions must have same length. "
                f"Got times: {len(times)}, positions: {len(positions)}"
            )

        # Validate positions dimensionality
        if self.dimension_ranges is not None:
            expected_dims = len(self.dimension_ranges)
            if positions.shape[1] != expected_dims:
                raise ValueError(
                    f"positions must have {expected_dims} dimensions to match environment. "
                    f"Got {positions.shape[1]} dimensions."
                )

        # Validate speed parameters
        if min_speed is not None and speed is None:
            raise ValueError(
                "min_speed parameter requires speed array to be provided. "
                "Pass speed=<array> along with min_speed=<threshold>."
            )

        if speed is not None:
            speed = np.asarray(speed, dtype=np.float64)
            if len(speed) != len(times):
                raise ValueError(
                    f"speed and times must have same length. "
                    f"Got speed: {len(speed)}, times: {len(times)}"
                )

        # Validate time_allocation parameter
        if time_allocation not in ("start", "linear"):
            raise ValueError(
                f"time_allocation must be 'start' or 'linear' (got '{time_allocation}'). "
                "Use 'start' for simple allocation (all layouts) or 'linear' for "
                "ray-grid intersection (RegularGridLayout only)."
            )

        # Check layout compatibility for linear allocation
        if (
            time_allocation == "linear"
            and type(self.layout).__name__ != "RegularGridLayout"
        ):
            raise NotImplementedError(
                "time_allocation='linear' is only supported for RegularGridLayout. "
                f"Current layout type: {type(self.layout).__name__}. "
                "Use time_allocation='start' for other layout types."
            )

        # Handle empty arrays
        if len(times) == 0:
            return np.zeros(self.n_bins, dtype=np.float64)

        # Handle single sample (no intervals to accumulate)
        if len(times) == 1:
            return np.zeros(self.n_bins, dtype=np.float64)

        # Map positions to bin indices
        bin_indices = cast(
            "NDArray[np.int64]",
            map_points_to_bins(
                positions, cast("Environment", self), tie_break="lowest_index"
            ),
        )

        # Compute time intervals
        dt = np.diff(times)

        # Build mask for valid intervals
        valid_mask = np.ones(len(dt), dtype=bool)

        # Filter by max_gap
        if max_gap is not None:
            valid_mask &= dt <= max_gap

        # Filter by min_speed (applied to starting position of each interval)
        if min_speed is not None and speed is not None:
            valid_mask &= speed[:-1] >= min_speed

        # Filter out intervals starting outside environment bounds
        # (map_points_to_bins returns -1 for points that don't map to any bin)
        valid_mask &= bin_indices[:-1] >= 0

        # Initialize occupancy array
        occupancy = np.zeros(self.n_bins, dtype=np.float64)

        # Dispatch to appropriate time allocation method
        if time_allocation == "start":
            # Simple allocation: entire interval goes to starting bin
            valid_bins = bin_indices[:-1][valid_mask]
            valid_dt = dt[valid_mask]

            # Use np.bincount for efficient accumulation
            if len(valid_bins) > 0:
                # Choose weights based on return_seconds parameter
                weights = valid_dt if return_seconds else np.ones_like(valid_dt)

                counts = np.bincount(valid_bins, weights=weights, minlength=self.n_bins)
                occupancy[:] = counts[: self.n_bins]

        elif time_allocation == "linear":
            # Linear allocation: split time across bins traversed by ray
            occupancy = self._allocate_time_linear(
                positions, dt, valid_mask, bin_indices, return_seconds
            )

        # Apply kernel smoothing if requested
        if kernel_bandwidth is not None:
            # Use mode='transition' for occupancy (counts), not 'density'
            # This ensures mass conservation: kernel columns sum to 1
            kernel = self.compute_kernel(
                bandwidth=kernel_bandwidth, mode="transition", cache=True
            )
            occupancy = kernel @ occupancy

        return occupancy

    def bin_sequence(
        self: EnvironmentProtocol,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        dedup: bool = True,
        return_runs: bool = False,
        outside_value: int | None = -1,
    ) -> (
        NDArray[np.int32]
        | tuple[NDArray[np.int32], NDArray[np.int64], NDArray[np.int64]]
    ):
        """Map trajectory to sequence of bin indices.

        Converts a continuous trajectory (times and positions) into a discrete
        sequence of bin indices, with optional deduplication of consecutive
        repeats and run-length encoding.

        Parameters
        ----------
        times : NDArray[np.float64], shape (n_samples,)
            Timestamps in seconds. Should be monotonically increasing.
        positions : NDArray[np.float64], shape (n_samples, n_dims)
            Position coordinates matching environment dimensions.
        dedup : bool, default=True
            If True, collapse consecutive repeats: [A,A,A,B] → [A,B].
            If False, return bin index for every sample.
        return_runs : bool, default=False
            If True, also return run boundaries (indices into times array).
            A "run" is a maximal contiguous subsequence in the same bin.
        outside_value : int or None, default=-1
            Bin index for samples outside environment bounds.
            - If -1 (default), outside samples are marked with -1.
            - If None, outside samples are dropped from the sequence entirely.

        Returns
        -------
        bins : NDArray[np.int32], shape (n_sequences,)
            Bin index at each time point (or deduplicated sequence).
            Values are in range [0, n_bins-1] for valid bins, or -1 for
            outside samples (when outside_value=-1).
        run_start_idx : NDArray[np.int64], shape (n_runs,), optional
            Start index (into original times array) of each contiguous run.
            Only returned if return_runs=True.
        run_end_idx : NDArray[np.int64], shape (n_runs,), optional
            End index (inclusive, into original times array) of each run.
            Only returned if return_runs=True.

        Raises
        ------
        ValueError
            If times and positions have different lengths, if positions
            have wrong number of dimensions, or if timestamps are not
            monotonically increasing (non-decreasing).

        See Also
        --------
        occupancy : Compute time spent in each bin.
        transitions : Build empirical transition matrix from trajectory.

        Notes
        -----
        A "run" is a maximal contiguous subsequence where all samples map to
        the same bin. When outside_value=-1, runs are split at boundary
        crossings (transitions to/from outside).

        When outside_value=None and samples fall outside the environment,
        they are completely removed from the sequence. This affects run
        boundaries if return_runs=True.

        Timestamps must be monotonically increasing (non-decreasing).
        Sort your data by time before calling this method if needed.

        Examples
        --------
        >>> # Basic usage: deduplicated bin sequence
        >>> bins = env.bin_sequence(times, positions)
        >>>
        >>> # Get run boundaries for duration calculations
        >>> bins, starts, ends = env.bin_sequence(times, positions, return_runs=True)
        >>> # Duration of first run:
        >>> duration = times[ends[0]] - times[starts[0]]
        >>>
        >>> # Keep all samples (no deduplication)
        >>> bins = env.bin_sequence(times, positions, dedup=False)
        >>>
        >>> # Drop outside samples entirely
        >>> bins = env.bin_sequence(times, positions, outside_value=None)

        """
        # Input validation
        times = np.asarray(times, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64)

        # Validate positions is 2D (consistent with occupancy())
        if positions.ndim != 2:
            raise ValueError(
                f"positions must be a 2-dimensional array (n_samples, n_dims), "
                f"got shape {positions.shape}"
            )

        # Validate lengths match
        if len(times) != len(positions):
            raise ValueError(
                f"times and positions must have the same length. "
                f"Got times: {len(times)}, positions: {len(positions)}"
            )

        # Validate dimensions match environment
        n_dims = self.n_dims
        if positions.shape[1] != n_dims:
            raise ValueError(
                f"positions must have {n_dims} dimensions to match environment. "
                f"Got positions.shape[1] = {positions.shape[1]}"
            )

        # Check for monotonic timestamps (raise error for consistency with occupancy())
        if len(times) > 1 and not np.all(np.diff(times) >= 0):
            decreasing_indices = np.where(np.diff(times) < 0)[0]
            raise ValueError(
                "times must be monotonically increasing (non-decreasing). "
                f"Found {len(decreasing_indices)} decreasing interval(s) at "
                f"indices: {decreasing_indices.tolist()[:5]}"
                + (" ..." if len(decreasing_indices) > 5 else "")
            )

        # Handle empty input
        if len(times) == 0:
            empty_bins = np.array([], dtype=np.int32)
            if return_runs:
                empty_runs = np.array([], dtype=np.int64)
                return empty_bins, empty_runs, empty_runs
            return empty_bins

        # Map positions to bin indices
        # Use bin_at which returns -1 for points outside environment
        bin_indices = self.bin_at(positions).astype(np.int32)  # Ensure int32 dtype

        # Handle outside_value=None (drop outside samples)
        if outside_value is None:
            # Filter out samples that are outside (bin_indices == -1)
            valid_mask = bin_indices != -1
            bin_indices = bin_indices[valid_mask]

            # Track original indices for run boundaries
            original_indices = np.arange(len(times))[valid_mask]

            if len(bin_indices) == 0:
                # All samples were outside
                empty_bins = np.array([], dtype=np.int32)
                if return_runs:
                    empty_runs = np.array([], dtype=np.int64)
                    return empty_bins, empty_runs, empty_runs
                return empty_bins
        else:
            # Keep original indices (no filtering)
            original_indices = np.arange(len(times))

        # Apply deduplication if requested
        deduplicated_bins: NDArray[np.int32]
        deduplicated_indices: NDArray[np.int_]

        if dedup:
            if len(bin_indices) == 0:
                # Already empty, nothing to deduplicate
                deduplicated_bins = bin_indices
                deduplicated_indices = original_indices
            else:
                # Find change points (where bin index changes)
                # Prepend True to include first element
                change_points = np.concatenate(
                    [[True], bin_indices[1:] != bin_indices[:-1]]
                )
                deduplicated_bins = bin_indices[change_points]
                deduplicated_indices = original_indices[change_points]
        else:
            deduplicated_bins = bin_indices
            deduplicated_indices = original_indices

        # Return just bins if runs not requested
        if not return_runs:
            return deduplicated_bins

        # Compute run boundaries
        if len(deduplicated_bins) == 0:
            # No runs
            empty_runs = np.array([], dtype=np.int64)
            return deduplicated_bins, empty_runs, empty_runs

        # For each run, find start and end indices in the *original* times array
        if dedup:
            # deduplicated_indices already contains the start of each run
            run_starts = deduplicated_indices

            # End of each run is just before the start of the next run
            # (or the last valid index for the final run)
            if outside_value is None:
                # Use the last valid index from original_indices
                run_ends = np.concatenate(
                    [deduplicated_indices[1:] - 1, [original_indices[-1]]]
                )
            else:
                # Use len(times) - 1 for the last run end
                run_ends = np.concatenate(
                    [deduplicated_indices[1:] - 1, [len(times) - 1]]
                )
        else:
            # No dedup: find runs in the un-deduplicated bin_indices
            # Find change points to identify run boundaries
            if len(bin_indices) == 1:
                # Single sample = single run
                run_starts = np.array([original_indices[0]], dtype=np.int64)
                run_ends = np.array([original_indices[0]], dtype=np.int64)
            else:
                # Find where bin index changes
                # A change occurs when bin_indices[i] != bin_indices[i-1]
                is_change = np.concatenate(
                    [[True], bin_indices[1:] != bin_indices[:-1]]
                )
                change_positions = np.where(is_change)[0]

                # Start of each run is at a change position
                run_starts = original_indices[change_positions]

                # End of each run is just before the next change (or last index)
                run_ends = np.concatenate(
                    [original_indices[change_positions[1:] - 1], [original_indices[-1]]]
                )

        return deduplicated_bins, run_starts, run_ends

    def transitions(
        self: EnvironmentProtocol,
        bins: NDArray[np.int32] | None = None,
        *,
        times: NDArray[np.float64] | None = None,
        positions: NDArray[np.float64] | None = None,
        # Empirical parameters
        lag: int = 1,
        allow_teleports: bool = False,
        # Model-based parameters
        method: Literal["diffusion", "random_walk"] | None = None,
        bandwidth: float | None = None,
        # Common parameters
        normalize: bool = True,
    ) -> scipy.sparse.csr_matrix:
        """Compute transition matrix (empirical or model-based).

        Two modes of operation:

        1. **Empirical**: Count observed transitions from trajectory data.
           Requires bins OR (times + positions). Analyzes actual behavior.

        2. **Model-based**: Generate theoretical transitions from graph structure.
           Requires method parameter. Models expected behavior.

        Parameters
        ----------
        bins : NDArray[np.int32], shape (n_samples,), optional
            [Empirical mode] Precomputed bin sequence. If None, computed from
            times/positions. Must contain valid bin indices in range [0, n_bins).
            Outside values (-1) are not allowed.
        times : NDArray[np.float64], shape (n_samples,), optional
            [Empirical mode] Timestamps in seconds. Must be provided together
            with positions.
        positions : NDArray[np.float64], shape (n_samples, n_dims), optional
            [Empirical mode] Position coordinates matching environment dimensions.
            Must be provided together with times.
        lag : int, default=1
            [Empirical mode] Temporal lag for transitions: count bins[t] → bins[t+lag].
            Must be positive. lag=1 counts consecutive transitions, lag=2 skips one bin.
        allow_teleports : bool, default=False
            [Empirical mode] If False, only count transitions between graph-adjacent
            bins. Non-adjacent transitions (e.g., from tracking errors) are excluded.
            Self-transitions (staying in same bin) are always counted.
            If True, count all transitions including non-local jumps.
        method : {'diffusion', 'random_walk'}, optional
            [Model mode] Type of model-based transitions:
            - 'random_walk': Uniform transitions to graph neighbors
            - 'diffusion': Distance-weighted transitions via heat kernel
            If provided, empirical parameters (bins/times/positions/lag/allow_teleports)
            are ignored.
        bandwidth : float, optional
            [Model: diffusion] Diffusion bandwidth in physical units (σ).
            Required when method='diffusion'. Larger values produce more uniform
            transitions; smaller values emphasize local transitions.
        normalize : bool, default=True
            If True, return row-stochastic matrix where each row sums to 1
            (representing transition probabilities).
            If False, return raw counts (empirical) or unnormalized weights (model).

        Returns
        -------
        T : scipy.sparse.csr_matrix, shape (n_bins, n_bins)
            Transition matrix where T[i,j] represents:
            - If normalize=True: P(next_bin=j | current_bin=i)
            - If normalize=False: count/weight of i→j transitions

            For normalized matrices, each row sums to 1.0 (rows with no
            transitions sum to 0.0).

        Raises
        ------
        ValueError
            If method is None and neither bins nor times/positions are provided.
            If method is provided together with empirical inputs (bins/times/positions).
            If method is provided together with empirical parameters (lag != 1 or allow_teleports != False).
            If method='random_walk' but bandwidth is provided.
            If method='diffusion' but bandwidth is not provided.
            If method='diffusion' but normalize=False (not supported).
            If bins contains invalid indices outside [0, n_bins).
            If lag is not positive (empirical mode).

        See Also
        --------
        bin_sequence : Convert trajectory to bin indices.
        occupancy : Compute time spent in each bin.
        compute_kernel : Low-level diffusion kernel computation.

        Notes
        -----
        **Empirical mode**: Counts observed transitions from trajectory data.
        When allow_teleports=False, filters out non-adjacent transitions using
        the connectivity graph. Useful for removing tracking errors.

        **Model mode**: Generates theoretical transition probabilities:
        - 'random_walk': Each bin transitions uniformly to all graph neighbors.
          Equivalent to normalized adjacency matrix.
        - 'diffusion': Transitions weighted by spatial proximity using heat kernel.
          Models continuous-time random walk with Gaussian steps.

        The sparse CSR format is memory-efficient for large environments
        where most bin pairs have no transitions.

        Examples
        --------
        >>> # Empirical transitions from trajectory
        >>> T_empirical = env.transitions(times=times, positions=positions)

        >>> # Empirical from precomputed bins with lag
        >>> T_lag2 = env.transitions(bins=bin_sequence, lag=2, allow_teleports=True)

        >>> # Model: uniform random walk
        >>> T_random = env.transitions(method="random_walk")

        >>> # Model: diffusion with spatial bias
        >>> T_diffusion = env.transitions(method="diffusion", bandwidth=5.0)

        >>> # Compare empirical vs model
        >>> diff = (T_empirical - T_diffusion).toarray()
        >>> # Large differences indicate non-random exploration

        """
        # Dispatch based on mode
        if method is not None:
            # MODEL-BASED MODE
            # Validate that empirical inputs aren't provided
            if bins is not None or times is not None or positions is not None:
                raise ValueError(
                    "Cannot provide both 'method' (model-based) and empirical "
                    "inputs (bins/times/positions). Choose one mode."
                )

            # Validate that empirical parameters aren't silently ignored
            if lag != 1:
                raise ValueError(
                    f"Parameter 'lag' is only valid in empirical mode. "
                    f"Got lag={lag} with method='{method}'. "
                    f"Remove 'lag' parameter or set method=None for empirical mode."
                )
            if allow_teleports is not False:
                raise ValueError(
                    f"Parameter 'allow_teleports' is only valid in empirical mode. "
                    f"Got allow_teleports={allow_teleports} with method='{method}'. "
                    f"Remove 'allow_teleports' parameter or set method=None for empirical mode."
                )

            # Validate bandwidth parameter usage
            if method == "random_walk" and bandwidth is not None:
                raise ValueError(
                    f"Parameter 'bandwidth' is only valid with method='diffusion'. "
                    f"Got bandwidth={bandwidth} with method='random_walk'. "
                    f"Remove 'bandwidth' parameter."
                )

            # Dispatch to model-based method
            if method == "random_walk":
                return self._random_walk_transitions(normalize=normalize)
            elif method == "diffusion":
                if bandwidth is None:
                    raise ValueError(
                        "method='diffusion' requires 'bandwidth' parameter. "
                        "Provide bandwidth in physical units (sigma)."
                    )
                return self._diffusion_transitions(
                    bandwidth=bandwidth, normalize=normalize
                )
            else:
                raise ValueError(
                    f"Unknown method '{method}'. "
                    f"Valid options: 'random_walk', 'diffusion'."
                )
        else:
            # EMPIRICAL MODE
            return self._empirical_transitions(
                bins=bins,
                times=times,
                positions=positions,
                lag=lag,
                normalize=normalize,
                allow_teleports=allow_teleports,
            )

    def _empirical_transitions(
        self: EnvironmentProtocol,
        bins: NDArray[np.int32] | None = None,
        *,
        times: NDArray[np.float64] | None = None,
        positions: NDArray[np.float64] | None = None,
        lag: int = 1,
        normalize: bool = True,
        allow_teleports: bool = False,
    ) -> scipy.sparse.csr_matrix:
        """Compute empirical transition matrix from observed trajectory data.

        Internal helper for transitions() method. Counts observed transitions
        between bins in a trajectory.

        Parameters
        ----------
        bins : NDArray[np.int32], shape (n_samples,), optional
            Precomputed bin sequence. If None, computed from times/positions.
            Cannot be provided together with times/positions.
            Must contain valid bin indices in range [0, n_bins). Outside values
            (-1) are not allowed; use times/positions input to handle outside samples.
        times : NDArray[np.float64], shape (n_samples,), optional
            Timestamps in seconds. Required if bins is None.
            Must be provided together with positions.
        positions : NDArray[np.float64], shape (n_samples, n_dims), optional
            Position coordinates matching environment dimensions.
            Required if bins is None. Must be provided together with times.
        lag : int, default=1
            Temporal lag for transitions: count bins[t] → bins[t+lag].
            Must be positive. lag=1 counts consecutive transitions,
            lag=2 skips one bin, etc.
        normalize : bool, default=True
            If True, return row-stochastic matrix where each row sums to 1
            (representing transition probabilities).
            If False, return raw transition counts.
        allow_teleports : bool, default=False
            If False, only count transitions between graph-adjacent bins.
            Non-adjacent transitions (e.g., from tracking errors) are excluded.
            Self-transitions (staying in same bin) are always counted.
            If True, count all transitions including non-local jumps.

        Returns
        -------
        T : scipy.sparse.csr_matrix, shape (n_bins, n_bins)
            Transition matrix where T[i,j] represents:
            - If normalize=True: P(next_bin=j | current_bin=i)
            - If normalize=False: count of i→j transitions

            For normalized matrices, each row sums to 1.0 (rows with no
            transitions sum to 0.0).

        Raises
        ------
        ValueError
            If neither bins nor times/positions are provided.
            If both bins and times/positions are provided.
            If only one of times or positions is provided.
            If bins contains invalid indices outside [0, n_bins).
            If lag is not positive.

        See Also
        --------
        bin_sequence : Convert trajectory to bin indices.
        occupancy : Compute time spent in each bin.

        Notes
        -----
        When allow_teleports=False, the method filters out non-adjacent
        transitions by checking the environment's connectivity graph. This
        helps remove artifacts from tracking errors or data gaps.

        Self-transitions (staying in the same bin) are always counted.

        The sparse CSR format is memory-efficient for large environments
        where most bin pairs have no observed transitions.

        Examples
        --------
        >>> # Compute transition probabilities from trajectory
        >>> T = env.transitions(times=times, positions=positions)
        >>> # Probability of moving from bin 10 to its neighbors
        >>> T[10, :].toarray()

        >>> # Get raw transition counts with teleport filtering
        >>> T_counts = env.transitions(
        ...     bins=bin_sequence, normalize=False, allow_teleports=False
        ... )

        >>> # Multi-step transitions (lag=2)
        >>> T_2step = env.transitions(bins=bin_sequence, lag=2)

        """
        import scipy.sparse

        # Validation: Ensure exactly one input method is used
        bins_provided = bins is not None
        trajectory_provided = times is not None or positions is not None

        if not bins_provided and not trajectory_provided:
            raise ValueError(
                "Must provide either 'bins' or both 'times' and 'positions'."
            )

        if bins_provided and trajectory_provided:
            raise ValueError(
                "Cannot provide both 'bins' and 'times'/'positions'. "
                "Use one input method only."
            )

        # If times/positions provided, validate both are present
        if trajectory_provided:
            if times is None or positions is None:
                raise ValueError(
                    "Both times and positions must be provided together "
                    "when computing transitions from trajectory."
                )

            # Compute bin sequence from trajectory
            bins = self.bin_sequence(times, positions, dedup=False, outside_value=-1)

        # Convert to numpy array and validate dtype
        bins = np.asarray(bins)
        if not np.issubdtype(bins.dtype, np.integer):
            raise ValueError(
                f"bins must be an integer array, got dtype {bins.dtype}. "
                f"Ensure bin indices are integers before calling transitions()."
            )
        bins = bins.astype(np.int32)

        # Validate lag
        if lag < 1:
            raise ValueError(f"lag must be positive (got {lag}).")

        # Handle empty or single-element sequences
        if len(bins) == 0 or len(bins) <= lag:
            # Return empty sparse matrix
            return scipy.sparse.csr_matrix((self.n_bins, self.n_bins), dtype=float)

        # Validate bin indices (must be in [0, n_bins))
        # Note: -1 is used for outside values, which is invalid for transitions
        if np.any(bins < 0) or np.any(bins >= self.n_bins):
            invalid_mask = (bins < 0) | (bins >= self.n_bins)
            invalid_indices = np.where(invalid_mask)[0]
            invalid_values = bins[invalid_mask]
            raise ValueError(
                f"Invalid bin indices found outside range [0, {self.n_bins}). "
                f"Found {len(invalid_indices)} invalid values at indices "
                f"{invalid_indices[:5].tolist()}{'...' if len(invalid_indices) > 5 else ''}: "
                f"{invalid_values[:5].tolist()}{'...' if len(invalid_values) > 5 else ''}. "
                f"Note: -1 (outside) values are not allowed in transitions."
            )

        # Extract transition pairs with lag
        source_bins = bins[:-lag]
        target_bins = bins[lag:]

        # Filter non-adjacent transitions if requested
        if not allow_teleports:
            # Build adjacency set from connectivity graph
            adjacency_set = set()
            for u, v in self.connectivity.edges():
                adjacency_set.add((u, v))
                adjacency_set.add((v, u))  # Undirected graph

            # Also include self-transitions (always adjacent)
            for node in self.connectivity.nodes():
                adjacency_set.add((node, node))

            # Filter transitions to only adjacent pairs
            is_adjacent = np.array(
                [
                    (src, tgt) in adjacency_set
                    for src, tgt in zip(source_bins, target_bins, strict=True)
                ]
            )

            source_bins = source_bins[is_adjacent]
            target_bins = target_bins[is_adjacent]

        # Count transitions using sparse COO format
        # Use np.ones to count occurrences
        transition_counts = np.ones(len(source_bins), dtype=float)

        # Build sparse matrix in COO format
        transition_matrix = scipy.sparse.coo_matrix(
            (transition_counts, (source_bins, target_bins)),
            shape=(self.n_bins, self.n_bins),
            dtype=float,
        )

        # Convert to CSR for efficient row operations
        transition_matrix = transition_matrix.tocsr()

        # Sum duplicate entries (multiple transitions between same bins)
        transition_matrix.sum_duplicates()

        # Normalize rows if requested
        if normalize:
            # Get row sums
            row_sums = np.array(transition_matrix.sum(axis=1)).flatten()

            # Avoid division by zero: only normalize rows with transitions
            nonzero_rows = row_sums > 0

            # Create diagonal matrix for normalization
            # Use reciprocal of row sums for nonzero rows, 0 otherwise
            inv_row_sums = np.zeros(self.n_bins)
            inv_row_sums[nonzero_rows] = 1.0 / row_sums[nonzero_rows]

            # Normalize: T_normalized = diag(1/row_sums) @ T
            normalizer = scipy.sparse.diags(inv_row_sums, format="csr")
            transition_matrix = normalizer @ transition_matrix

        return transition_matrix

    def _random_walk_transitions(
        self: EnvironmentProtocol,
        *,
        normalize: bool = True,
    ) -> scipy.sparse.csr_matrix:
        """Compute uniform random walk transition matrix from graph structure.

        Internal helper for transitions(method='random_walk'). Creates a
        transition matrix where each bin transitions uniformly to its neighbors.
        """
        import scipy.sparse

        # Get adjacency matrix from connectivity graph
        adjacency = nx.adjacency_matrix(self.connectivity, nodelist=range(self.n_bins))

        # Convert to float and ensure CSR format
        transition_matrix = adjacency.astype(float).tocsr()

        if normalize:
            # Normalize rows: T[i,j] = 1/degree(i) if j is neighbor of i
            row_sums = np.array(transition_matrix.sum(axis=1)).flatten()

            # Avoid division by zero for isolated nodes
            nonzero_rows = row_sums > 0
            inv_row_sums = np.zeros(self.n_bins)
            inv_row_sums[nonzero_rows] = 1.0 / row_sums[nonzero_rows]

            # Normalize
            normalizer = scipy.sparse.diags(inv_row_sums, format="csr")
            transition_matrix = normalizer @ transition_matrix

        return transition_matrix

    def _diffusion_transitions(
        self: EnvironmentProtocol,
        bandwidth: float,
        *,
        normalize: bool = True,
    ) -> scipy.sparse.csr_matrix:
        """Compute diffusion-based transition matrix using heat kernel.

        Internal helper for transitions(method='diffusion'). Uses the heat
        kernel to model continuous-time diffusion on the graph.
        """
        import scipy.sparse

        # Use existing compute_kernel infrastructure
        kernel = self.compute_kernel(bandwidth=bandwidth, mode="transition")

        # kernel is already row-stochastic from compute_kernel
        # Convert to sparse if needed
        if not scipy.sparse.issparse(kernel):
            kernel = scipy.sparse.csr_matrix(kernel)

        if not normalize:
            raise ValueError(
                "method='diffusion' does not support normalize=False. "
                "Heat kernel transitions are inherently normalized (row-stochastic). "
                "Set normalize=True or use method='random_walk'."
            )

        return kernel

    def _allocate_time_linear(
        self: EnvironmentProtocol,
        positions: NDArray[np.float64],
        dt: NDArray[np.float64],
        valid_mask: NDArray[np.bool_],
        bin_indices: NDArray[np.int64],
        return_seconds: bool,
    ) -> NDArray[np.float64]:
        """Allocate time intervals linearly across traversed bins (helper for occupancy).

        This method implements ray-grid intersection to split each time interval
        proportionally across all bins crossed by the straight-line path between
        consecutive position samples.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (n_samples, n_dims)
            Position samples.
        dt : NDArray[np.float64], shape (n_samples-1,)
            Time intervals between consecutive samples.
        valid_mask : NDArray[np.bool_], shape (n_samples-1,)
            Boolean mask indicating which intervals are valid (pass filtering).
        bin_indices : NDArray[np.int64], shape (n_samples,)
            Bin indices for each position (-1 if outside environment).
        return_seconds : bool
            If True, return time in seconds. If False, return sample counts.

        Returns
        -------
        occupancy : NDArray[np.float64], shape (n_bins,)
            If return_seconds=True: Time in seconds allocated to each bin.
            If return_seconds=False: Count allocated to each bin (proportional).

        """
        from neurospatial.layout.engines.regular_grid import RegularGridLayout

        # Ensure we have RegularGridLayout (already validated in occupancy())
        layout = cast("RegularGridLayout", self.layout)

        # Get grid structure
        grid_edges = layout.grid_edges
        grid_shape = layout.grid_shape

        # Assert non-None for mypy (RegularGridLayout always has these)
        assert grid_edges is not None, "RegularGridLayout must have grid_edges"
        assert grid_shape is not None, "RegularGridLayout must have grid_shape"

        # Initialize occupancy array
        occupancy = np.zeros(self.n_bins, dtype=np.float64)

        # Process each valid interval
        for i in np.where(valid_mask)[0]:
            start_pos = positions[i]
            end_pos = positions[i + 1]
            interval_time = dt[i]

            # Choose weight based on return_seconds parameter
            weight = interval_time if return_seconds else 1.0

            # Get starting and ending bin indices
            start_bin = bin_indices[i]
            end_bin = bin_indices[i + 1]

            # If both points are in same bin, simple allocation
            if start_bin == end_bin and start_bin >= 0:
                occupancy[start_bin] += weight
                continue

            # Compute ray-grid intersections
            # Still use interval_time for proportional splitting, then scale by weight
            bin_times = self._compute_ray_grid_intersections(
                start_pos, end_pos, list(grid_edges), grid_shape, interval_time
            )

            # Accumulate to each bin
            # Scale the time allocations proportionally to maintain total weight
            if return_seconds:
                # Already in seconds from ray-grid intersection
                for bin_idx, time_in_bin in bin_times:
                    if 0 <= bin_idx < self.n_bins:
                        occupancy[bin_idx] += time_in_bin
            else:
                # Convert time proportions to count proportions (sum to 1.0)
                total_allocated = sum(time for _, time in bin_times)
                if total_allocated > 0:
                    for bin_idx, time_in_bin in bin_times:
                        if 0 <= bin_idx < self.n_bins:
                            # Proportional allocation that sums to 1.0
                            occupancy[bin_idx] += (
                                time_in_bin / total_allocated
                            ) * weight

        return occupancy

    def _compute_ray_grid_intersections(
        self: EnvironmentProtocol,
        start_pos: NDArray[np.float64],
        end_pos: NDArray[np.float64],
        grid_edges: list[NDArray[np.float64]],
        grid_shape: tuple[int, ...],
        total_time: float,
    ) -> list[tuple[int, float]]:
        """Compute time spent in each bin along a ray (helper for linear allocation).

        Uses DDA-like algorithm to traverse grid and compute intersection distances.

        Parameters
        ----------
        start_pos : NDArray[np.float64], shape (n_dims,)
            Starting position.
        end_pos : NDArray[np.float64], shape (n_dims,)
            Ending position.
        grid_edges : list[NDArray[np.float64]]
            Grid edges per dimension.
        grid_shape : tuple[int, ...]
            Grid shape.
        total_time : float
            Total time interval to split across bins.

        Returns
        -------
        bin_times : list[tuple[int, float]]
            List of (bin_index, time_in_bin) pairs.

        """
        n_dims = len(grid_shape)

        # Compute ray direction and total distance
        ray_dir = end_pos - start_pos
        total_distance = np.linalg.norm(ray_dir)

        # Handle zero-distance case (no movement)
        if total_distance < 1e-12:
            # No movement - allocate all time to starting bin
            start_bin_idx = self._position_to_flat_index(
                start_pos, list(grid_edges), grid_shape
            )
            if start_bin_idx >= 0:
                return [(start_bin_idx, total_time)]
            return []

        # Normalize ray direction
        ray_dir = ray_dir / total_distance

        # Find all grid crossings along each dimension
        crossings: list[tuple[float, int, int]] = []  # (t, dim, grid_index)

        for dim in range(n_dims):
            if abs(ray_dir[dim]) < 1e-12:
                # Ray parallel to this dimension - no crossings
                continue

            edges = grid_edges[dim]
            # Find which edges the ray crosses
            for edge_idx, edge_pos in enumerate(edges):
                # Parametric intersection: start + t * ray_dir = edge_pos
                t = (edge_pos - start_pos[dim]) / ray_dir[dim]
                if 0 < t < total_distance:  # Exclude endpoints
                    crossings.append((t, dim, edge_idx))

        # Sort crossings by distance along ray
        crossings.sort(key=itemgetter(0))

        # Add start and end points
        segments = [0.0] + [t for t, _, _ in crossings] + [total_distance]

        # Compute bin index and time for each segment
        bin_times: list[tuple[int, float]] = []
        for seg_idx in range(len(segments) - 1):
            # Midpoint of segment (to determine which bin we're in)
            t_mid = (segments[seg_idx] + segments[seg_idx + 1]) / 2
            mid_pos = start_pos + t_mid * ray_dir

            # Get bin index at midpoint
            bin_idx = self._position_to_flat_index(
                mid_pos, list(grid_edges), grid_shape
            )

            if bin_idx >= 0:
                # Compute time in this segment
                seg_distance = segments[seg_idx + 1] - segments[seg_idx]
                seg_time = total_time * (seg_distance / total_distance)
                bin_times.append((bin_idx, seg_time))

        return bin_times

    def _position_to_flat_index(
        self: EnvironmentProtocol,
        pos: NDArray[np.float64],
        grid_edges: list[NDArray[np.float64]],
        grid_shape: tuple[int, ...],
    ) -> int:
        """Convert N-D position to flat bin index (helper for ray intersection).

        Parameters
        ----------
        pos : NDArray[np.float64], shape (n_dims,)
            Position coordinates.
        grid_edges : list[NDArray[np.float64]]
            Grid edges per dimension.
        grid_shape : tuple[int, ...]
            Grid shape.

        Returns
        -------
        flat_index : int
            Flat bin index, or -1 if position is outside grid bounds.

        """
        n_dims = len(grid_shape)
        nd_index = []

        for dim in range(n_dims):
            edges = grid_edges[dim]
            coord = pos[dim]

            # Find which bin this coordinate falls into
            # bins are [edges[i], edges[i+1])
            bin_idx = np.searchsorted(edges, coord, side="right") - 1

            # Check bounds
            if bin_idx < 0 or bin_idx >= grid_shape[dim]:
                return -1  # Outside grid

            nd_index.append(bin_idx)

        # Convert N-D index to flat index (row-major order)
        flat_idx = 0
        stride = 1
        for dim in reversed(range(n_dims)):
            flat_idx += nd_index[dim] * stride
            stride *= grid_shape[dim]

        return flat_idx
