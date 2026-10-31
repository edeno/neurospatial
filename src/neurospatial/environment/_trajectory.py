"""Trajectory adapters on owned geometry and explicit sampled data.

Includes time allocation, occupancy, run extraction and the inputs of the
transition estimators that ``neurospatial.ops`` shares (``_count_transitions``,
``_diffusion_transitions``, ``_random_walk_transitions``). Environment
exposes thin entry points; Cartesian coordinate operations are guarded.

Index-dtype convention: every array of bin indices an Environment returns
(``bin_at``, ``nearest_bin``, ``bins_in_region``, ``boundary_bins``,
``bin_sequence``, ``BinSequenceWithRuns.bins``, ``components``, ``rings``) is
``np.intp``, the dtype NumPy uses for indexing, so the result indexes a field
directly on every platform. The ``run_starts`` / ``run_lengths`` sample
offsets are ``np.int64``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx
import numpy as np
import scipy.sparse
from numpy.typing import NDArray

from neurospatial._intervals import (
    DEFAULT_MAX_GAP,
    as_intervals,
    drop_missing_samples,
    fill_missing_positions,
    interval_gates,
    observed_interval_mask,
    start_allocation,
    validate_speed_gate,
    warn_excluded_intervals,
)
from neurospatial._observation import validate_break_before
from neurospatial._validation import validate_times_positions
from neurospatial.environment._cache import AnalysisCache
from neurospatial.environment._fields import _compute_kernel
from neurospatial.geometry import (
    Binning,
    CartesianSystem,
    EgocentricPolarSystem,
    locate,
    require_cartesian,
    require_position_columns,
    require_rectilinear,
)
from neurospatial.ops import (
    _count_transitions,
    _linear_grid_occupancy,
    _polygon_linear_occupancy,
    _random_walk_transitions,
    _uses_polygon_allocation,
)
from neurospatial.ops import (
    _diffusion_transitions as _diffusion_transition_matrix,
)

if TYPE_CHECKING:
    from neurospatial.geometry import Arena


@dataclass(frozen=True)
class BinSequenceWithRuns:
    """Bin sequence with run-length boundaries returned by ``bin_sequence_with_runs``.

    A "run" is a maximal contiguous subsequence of trajectory samples that
    map to the same bin. ``bins[i]`` is the bin for run ``i``;
    ``run_starts[i]`` and ``run_lengths[i]`` describe the run's extent in
    the *original* (un-deduplicated) ``times``/``positions`` arrays so
    callers can recover per-run durations as
    ``times[run_starts[i] + run_lengths[i] - 1] - times[run_starts[i]]``.

    Attributes
    ----------
    bins : NDArray[np.intp], shape (n_runs,)
        Bin index for each run. ``-1`` denotes runs of out-of-environment
        samples (matching ``Environment.bin_at`` semantics).
    run_starts : NDArray[np.int64], shape (n_runs,)
        Start index of each run in the original ``times`` array.
    run_lengths : NDArray[np.int64], shape (n_runs,)
        Inclusive extent of each run in the original ``times`` array (the
        inclusive end index is ``run_starts[i] + run_lengths[i] - 1``),
        counting any missing (NaN) rows bridged inside the run. Always >= 1
        for any emitted run.

        With ``outside_value=None``, samples outside the environment
        are dropped from ``bins`` *and* outside gaps split runs even
        when consecutive in-env samples land in the same bin — so a
        sequence ``[bin0, outside, bin0]`` becomes two runs of length
        1, not one run of length 3. ``run_lengths.sum()`` then counts
        the in-env samples plus the missing rows bridged inside runs.
    """

    bins: NDArray[np.intp]
    run_starts: NDArray[np.int64]
    run_lengths: NDArray[np.int64]


def _finite_rows(positions: NDArray[np.float64]) -> NDArray[np.bool_]:
    """True for samples whose every coordinate is finite, shape (n_samples,)."""
    return np.asarray(
        np.isfinite(positions).all(axis=tuple(range(1, positions.ndim))), dtype=bool
    )


def _occupancy(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    binning: Binning,
    arena: Arena,
    coordinates: CartesianSystem | EgocentricPolarSystem,
    cache: AnalysisCache,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: Any = None,
    bandwidth: float | None = None,
    time_allocation: Literal["start", "linear"] = "start",
    return_seconds: bool = True,
) -> NDArray[np.float64]:
    """Compute occupancy from sampled positions."""
    n_bins, n_dims = binning.bins.n_bins, binning.bins.n_dims
    times, positions = validate_times_positions(
        times, positions, call="Environment.occupancy"
    )
    # Before the empty and single-sample returns, which never look positions up.
    require_position_columns(positions, n_dims, call="Environment.occupancy")

    speed = validate_speed_gate(speed, min_speed, len(times))
    breaks = validate_break_before(
        break_before, len(times), call="Environment.occupancy"
    )

    # Validate time_allocation parameter
    if time_allocation not in ("start", "linear"):
        msg = (
            f"time_allocation must be 'start' or 'linear' (got '{time_allocation}'). "
            "Use 'start' for simple allocation (all layouts) or 'linear' for "
            "segment intersection with rectangular or exact planar polygon cells."
        )
        raise ValueError(msg)

    # Validate the capability before returning for empty input.
    polygon_allocation = False
    grid = None
    if time_allocation == "linear":
        require_cartesian(
            coordinates,
            "Environment.occupancy(time_allocation='linear')",
            why="It splits each interval along the straight Cartesian path "
            "between two samples, which is not straight in (distance, angle).",
            fix="give each interval to the bin its first sample is in, with "
            "time_allocation='start'",
            example="env.occupancy(times, positions, time_allocation='start')",
        )
        polygon_allocation = _uses_polygon_allocation(binning, arena)
        if not polygon_allocation:
            grid = require_rectilinear(
                binning,
                "occupancy(time_allocation='linear')",
                fix="use time_allocation='start', which works on every layout",
            )

    # Handle empty arrays
    if len(times) == 0:
        return np.zeros(n_bins, dtype=np.float64)

    # Handle single sample (no intervals to accumulate)
    if len(times) == 1:
        return np.zeros(n_bins, dtype=np.float64)

    # Map positions to bin indices using the layout's geometric
    # containment, which matches Environment.bin_sequence and returns
    # -1 for samples outside the active mask. The earlier path
    # used nearest-center lookup (KDTree nearest-neighbor with a
    # heuristic 10x-typical-spacing threshold). That heuristic let
    # samples a few centimeters outside the active mask silently
    # bind to the nearest edge bin, so a tracking-error overshoot
    # inflated occupancy at the boundary while bin_sequence
    # correctly flagged the same sample as -1.
    bin_indices = locate(binning, arena, positions, cache, call="Environment.occupancy")

    resolved_epochs = as_intervals(epochs, name="epochs")
    # A missing (NaN) sample is bridged like a deleted row, by the same rule
    # spike binning uses: the gap between the valid samples on either side
    # passes the gates.
    finite = _finite_rows(positions)
    if time_allocation == "start":
        # Shared with spike binning, including the complete-case rule.
        allocation = start_allocation(
            times,
            bin_indices,
            finite,
            speed=speed,
            min_speed=min_speed,
            max_gap=max_gap,
            epochs=resolved_epochs,
            break_before=breaks,
        )
        occupancy = allocation.occupancy(n_bins, return_seconds=return_seconds)
        retained = allocation.gates
    else:
        # Linear allocation keeps observation validity separate from
        # geometric support, so an interval that enters, leaves or re-enters
        # the environment keeps the time its straight path spends inside; an
        # interval whose path never enters it (outside, or with a NaN end) is
        # excluded only by the geometry. Missing rows lie on the straight
        # path between the valid rows around them.
        gates = interval_gates(
            times,
            speed=speed,
            min_speed=min_speed,
            max_gap=max_gap,
            epochs=resolved_epochs,
            break_before=breaks,
            sample_valid=finite,
        )
        valid_mask = gates.mask
        dt = np.diff(times)
        positions = fill_missing_positions(
            times, positions.reshape(len(times), -1), time_allocation="linear"
        )
        bin_indices = locate(
            binning, arena, positions, cache, call="Environment.occupancy"
        )
        if polygon_allocation:
            pieces, occupancy = _polygon_linear_occupancy(
                binning,
                arena,
                times,
                positions,
                valid_mask,
                return_seconds=return_seconds,
                indexes=cache,
            )
            kept = np.zeros_like(valid_mask)
            kept[[piece.sample_index for piece in pieces]] = True
            retained = gates.narrowed(kept)
        else:
            assert grid is not None
            kept, occupancy = _linear_grid_occupancy(
                n_bins,
                grid,
                positions,
                dt,
                valid_mask,
                bin_indices,
                return_seconds=return_seconds,
            )
            retained = gates.narrowed(kept)

    warn_excluded_intervals(retained, result="occupancy is zero in every bin")

    # Apply kernel smoothing if requested
    if bandwidth is not None:
        # Use mode='transition' for occupancy (counts), not 'density'
        # This ensures mass conservation: kernel columns sum to 1
        kernel = _compute_kernel(
            cache, binning, bandwidth, mode="transition", writable=False
        )
        occupancy = kernel @ occupancy

    return occupancy


def _bin_sequence(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    binning: Binning,
    arena: Arena,
    cache: AnalysisCache,
    call: str,
    dedup: bool,
    outside_value: int | None,
    gap_splits_runs: bool,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: Any = None,
) -> BinSequenceWithRuns:
    """Shared implementation for ``bin_sequence`` and
    ``bin_sequence_with_runs``: always computes both bins and runs;
    the public methods choose which fields of the result to expose.
    ``call`` names the public method in error messages.
    """
    times, positions = validate_times_positions(times, positions, call=call)
    # Before the gap warnings, so wrong positions are reported first.
    require_position_columns(positions, binning.bins.n_dims, call=call)
    breaks = validate_break_before(break_before, times.size, call=call)
    # A missing (NaN) sample is skipped as if its row were deleted; run
    # indices are mapped back to the caller's rows at the end.
    rows, breaks = drop_missing_samples(_finite_rows(positions), breaks)
    times, positions = times[rows], positions[rows]

    mask = observed_interval_mask(
        times,
        max_gap=max_gap,
        epochs=epochs,
        break_before=breaks,
        result="the bin sequence is empty",
    )
    sample_valid = np.zeros(times.size, dtype=bool)
    sample_valid[:-1] |= mask
    sample_valid[1:] |= mask
    invalid_before = np.r_[0, np.cumsum(~mask)]

    empty_int64 = np.array([], dtype=np.int64)

    # Handle empty input
    if len(times) == 0:
        return BinSequenceWithRuns(
            bins=np.array([], dtype=np.intp),
            run_starts=empty_int64,
            run_lengths=empty_int64,
        )

    # Map positions to bin indices
    # Use bin_at which returns -1 for points outside environment
    bin_indices = locate(binning, arena, positions, cache, call=call)

    # Samples belong to a recording only if they touch a valid interval.
    if outside_value is None:
        sample_valid &= bin_indices != -1
    original_indices = np.flatnonzero(sample_valid)
    bin_indices = bin_indices[sample_valid]

    # Apply deduplication if requested
    deduplicated_bins: NDArray[np.intp]

    # Recording breaks always split repeats. Outside-sample gaps retain
    # the plain sequence's legacy dedup behavior, and split the run view.
    if len(bin_indices) == 0:
        change_points = np.zeros(0, dtype=bool)
    else:
        bin_change = np.concatenate([[True], bin_indices[1:] != bin_indices[:-1]])
        if gap_splits_runs and outside_value is None and len(original_indices) > 1:
            gap_change = np.concatenate(
                [[False], (original_indices[1:] - original_indices[:-1]) > 1]
            )
        else:
            gap_change = np.zeros(len(bin_indices), dtype=bool)
        recording_change = np.r_[
            False,
            invalid_before[original_indices[1:]]
            != invalid_before[original_indices[:-1]],
        ]
        change_points = bin_change | gap_change | recording_change

    deduplicated_bins = bin_indices[change_points] if dedup else bin_indices

    if len(deduplicated_bins) == 0:
        return BinSequenceWithRuns(
            bins=deduplicated_bins,
            run_starts=empty_int64,
            run_lengths=empty_int64,
        )

    # For each run, find start and end indices in the *original* times
    # array. ``change_positions`` indexes into the filtered
    # ``original_indices`` array; the run starting at change_positions[i]
    # ends just before change_positions[i+1], or at original_indices[-1]
    # for the final run. With outside_value=None the gap-change above
    # already inserted a run boundary at every drop, so the per-run
    # extent from start to end (inclusive) covers only original-array
    # indices that are actually in this run — outside samples do not
    # leak into run_lengths anymore.
    change_positions = np.flatnonzero(change_points)
    run_starts = original_indices[change_positions]
    run_ends = original_indices[
        np.concatenate([change_positions[1:], [len(original_indices)]]) - 1
    ]

    # Convert (start, end) to (start, length) for the public API. ``end``
    # is inclusive, so length = end - start + 1 is always >= 1.
    run_starts, run_ends = rows[run_starts], rows[run_ends]
    run_lengths = (run_ends - run_starts + 1).astype(np.int64)
    return BinSequenceWithRuns(
        bins=deduplicated_bins,
        run_starts=np.asarray(run_starts, dtype=np.int64),
        run_lengths=run_lengths,
    )


def _transitions(
    bins: NDArray[np.intp] | None = None,
    *,
    graph: nx.Graph,
    binning: Binning,
    arena: Arena,
    cache: AnalysisCache,
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
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: Any = None,
) -> scipy.sparse.csr_matrix:
    """Compute empirical or model-based transition matrices."""
    # Dispatch based on mode
    if method is not None:
        # MODEL-BASED MODE
        # Validate that empirical inputs aren't provided
        if bins is not None or times is not None or positions is not None:
            msg = (
                "Cannot provide both 'method' (model-based) and empirical "
                "inputs (bins/times/positions). Choose one mode."
            )
            raise ValueError(msg)

        # Validate that empirical parameters aren't silently ignored
        if lag != 1:
            msg = (
                f"Parameter 'lag' is only valid in empirical mode. "
                f"Got lag={lag} with method='{method}'. "
                f"Remove 'lag' parameter or set method=None for empirical mode."
            )
            raise ValueError(msg)
        if allow_teleports is not False:
            msg = (
                f"Parameter 'allow_teleports' is only valid in empirical mode. "
                f"Got allow_teleports={allow_teleports} with method='{method}'. "
                f"Remove 'allow_teleports' parameter or set method=None for empirical mode."
            )
            raise ValueError(msg)
        if break_before is not None:
            msg = (
                f"Parameter 'break_before' is only valid in empirical mode. "
                f"Got break_before with method='{method}': model transitions "
                f"come from the graph, not from a recording. "
                f"Remove 'break_before' or set method=None for empirical mode."
            )
            raise ValueError(msg)

        # Validate bandwidth parameter usage
        if method == "random_walk" and bandwidth is not None:
            msg = (
                f"Parameter 'bandwidth' is only valid with method='diffusion'. "
                f"Got bandwidth={bandwidth} with method='random_walk'. "
                f"Remove 'bandwidth' parameter."
            )
            raise ValueError(msg)

        # Dispatch to model-based method
        if method == "random_walk":
            return _random_walk_transitions(graph, normalize=normalize)
        if method == "diffusion":
            if bandwidth is None:
                msg = (
                    "method='diffusion' requires 'bandwidth' parameter. "
                    "Provide bandwidth in physical units (sigma)."
                )
                raise ValueError(msg)
            return _diffusion_transitions(
                cache, binning, bandwidth=bandwidth, normalize=normalize
            )
        msg = f"Unknown method '{method}'. Valid options: 'random_walk', 'diffusion'."
        raise ValueError(msg)
    else:
        # EMPIRICAL MODE
        return _empirical_transitions(
            bins=bins,
            binning=binning,
            arena=arena,
            cache=cache,
            times=times,
            positions=positions,
            lag=lag,
            normalize=normalize,
            allow_teleports=allow_teleports,
            max_gap=max_gap,
            epochs=epochs,
            break_before=break_before,
        )


def _empirical_transitions(
    bins: NDArray[np.intp] | None = None,
    *,
    binning: Binning,
    arena: Arena,
    cache: AnalysisCache,
    times: NDArray[np.float64] | None = None,
    positions: NDArray[np.float64] | None = None,
    lag: int = 1,
    normalize: bool = True,
    allow_teleports: bool = False,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: Any = None,
) -> scipy.sparse.csr_matrix:
    """Compute empirical transition matrix from observed trajectory data.

    Internal helper for transitions() method. Counts observed transitions
    between bins in a trajectory.

    Parameters
    ----------
    bins : NDArray[np.intp], shape (n_samples,), optional
        Precomputed bin sequence. If None, computed from times/positions.
        Cannot be provided together with times/positions.
        Must contain valid bin indices in range [0, n_bins). Outside values
        (-1) are not allowed; use times/positions input to handle outside samples.
        From times/positions, missing (NaN) samples are skipped as deleted
        rows, and a pair counts only when both of its samples are
        inside the environment; outside and NaN samples add no transition.
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
    where most bin pairs have no observed transitions. Executed examples
    are in :meth:`Environment.transitions`.

    """
    n_bins = binning.bins.n_bins
    # Validation: Ensure exactly one input method is used
    bins_provided = bins is not None
    trajectory_provided = times is not None or positions is not None

    if not bins_provided and not trajectory_provided:
        msg = "Must provide either 'bins' or both 'times' and 'positions'."
        raise ValueError(msg)

    if bins_provided and trajectory_provided:
        msg = (
            "Cannot provide both 'bins' and 'times'/'positions'. "
            "Use one input method only."
        )
        raise ValueError(msg)

    # If times/positions provided, validate both are present
    if trajectory_provided:
        if times is None or positions is None:
            msg = (
                "Both times and positions must be provided together "
                "when computing transitions from trajectory."
            )
            raise ValueError(msg)

        # Compute bin sequence from trajectory
        bins = _bin_sequence(
            times,
            positions,
            binning=binning,
            arena=arena,
            cache=cache,
            call="Environment.transitions",
            dedup=False,
            outside_value=-1,
            gap_splits_runs=False,
            max_gap=None,
            epochs=None,
        ).bins
        breaks = validate_break_before(
            break_before, np.asarray(times).size, call="Environment.transitions"
        )
        # The bin sequence skips missing (NaN) samples as deleted rows; gate
        # the same rows.
        times = np.asarray(times, dtype=np.float64)
        rows, breaks = drop_missing_samples(
            _finite_rows(np.asarray(positions, dtype=np.float64)), breaks
        )
        mask = observed_interval_mask(
            times[rows],
            max_gap=max_gap,
            epochs=epochs,
            break_before=breaks,
            result="the transition matrix is empty",
        )

    # Convert to numpy array and validate dtype
    bins = np.asarray(bins)
    if not np.issubdtype(bins.dtype, np.integer):
        msg = (
            f"bins must be an integer array, got dtype {bins.dtype}. "
            f"Ensure bin indices are integers before calling transitions()."
        )
        raise ValueError(msg)
    bins = bins.astype(np.intp)
    if bins_provided:
        # A precomputed sequence has no clock, so only declared breaks apply.
        breaks = validate_break_before(
            break_before, bins.size, call="Environment.transitions"
        )
        mask = np.ones(max(bins.size - 1, 0), dtype=bool)
        if breaks is not None:
            mask = ~breaks[1:]

    # Validate lag
    if lag < 1:
        msg = f"lag must be positive (got {lag})."
        raise ValueError(msg)

    # Handle empty or single-element sequences
    if len(bins) == 0 or len(bins) <= lag:
        # Return empty sparse matrix
        return scipy.sparse.csr_matrix((n_bins, n_bins), dtype=float)

    # Validate bin indices (must be in [0, n_bins)). A precomputed sequence
    # must not hold -1; samples from times/positions may be outside or missing.
    if bins_provided and (np.any(bins < 0) or np.any(bins >= n_bins)):
        invalid_mask = (bins < 0) | (bins >= n_bins)
        invalid_indices = np.where(invalid_mask)[0]
        invalid_values = bins[invalid_mask]
        msg = (
            f"Invalid bin indices found outside range [0, {n_bins}). "
            f"Found {len(invalid_indices)} invalid values at indices "
            f"{invalid_indices[:5].tolist()}{'...' if len(invalid_indices) > 5 else ''}: "
            f"{invalid_values[:5].tolist()}{'...' if len(invalid_values) > 5 else ''}. "
            f"Note: -1 (outside) values are not allowed in transitions."
        )
        raise ValueError(msg)

    # Extract transition pairs with lag
    source_bins = bins[:-lag]
    target_bins = bins[lag:]
    # A pair counts when no excluded interval separates its two samples and,
    # from times/positions, both lie inside the environment (bin -1: outside
    # or NaN).
    invalid_before = np.r_[0, np.cumsum(~mask)]
    pair_ok = invalid_before[lag:] == invalid_before[:-lag]
    if trajectory_provided:
        pair_ok &= (source_bins >= 0) & (target_bins >= 0)
    source_bins, target_bins = source_bins[pair_ok], target_bins[pair_ok]

    return _count_transitions(
        source_bins,
        target_bins,
        n_bins,
        neighbor_pairs=None if allow_teleports else binning.connections.pairs,
        normalize=normalize,
    )


def _diffusion_transitions(
    cache: AnalysisCache,
    binning: Binning,
    bandwidth: float,
    *,
    normalize: bool = True,
) -> scipy.sparse.csr_matrix:
    """Return the heat-diffusion transitions from the cached heat kernel."""
    if not normalize:
        msg = (
            "method='diffusion' does not support normalize=False. "
            "Heat kernel transitions are inherently normalized (row-stochastic). "
            "Set normalize=True or use method='random_walk'."
        )
        raise ValueError(msg)

    return _diffusion_transition_matrix(
        _compute_kernel(cache, binning, bandwidth, mode="transition", writable=False)
    )
