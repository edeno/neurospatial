"""Spike generation functions for converting firing rates to spike trains."""

import logging
import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm

from neurospatial._exceptions import _format_error
from neurospatial._intervals import DEFAULT_MAX_GAP
from neurospatial._validation import validate_finite
from neurospatial.simulation.models.base import NeuralModel

logger = logging.getLogger(__name__)


def generate_poisson_spikes(
    firing_rate: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    refractory_period: float = 0.002,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate Poisson spike train from firing rate time series.

    Implements an inhomogeneous Poisson process with absolute refractory period.

    Parameters
    ----------
    firing_rate : NDArray[np.float64], shape (n_time,)
        Non-negative firing rate in Hz, held constant on each following
        sampling interval. The final sample has no following interval and
        contributes no spikes.
    times : NDArray[np.float64], shape (n_time,)
        Time points in seconds. Must be strictly increasing. Non-uniform
        spacing is supported: each interval contributes its own width to
        the expected spike count. Spikes lie in ``[times[0], times[-1])``.
    refractory_period : float, optional
        Absolute refractory period in seconds (default: 2ms = 0.002s).
        Prevents spikes within this window. Biologically realistic value
        for most neurons is 1-3ms.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Times of generated spikes in seconds, sorted in ascending order.

    Raises
    ------
    ValueError
        If ``firing_rate`` and ``times`` do not have the same length.
    ValueError
        If ``firing_rate`` contains non-finite values (NaN or inf) or
        negative values, or if ``times`` contains non-finite values (NaN or
        inf).
    ValueError
        If ``times`` is not strictly increasing (i.e. any interval has
        zero or negative width).

    Notes
    -----
    **Algorithm (absolute refractory period implementation)**:

    1. Generate candidate spikes from inhomogeneous Poisson process:

       - Draw the spike count in each interval from
         ``Poisson(rate[i] * (times[i+1] - times[i]))``.
       - Conditional on that count, place spikes uniformly within the interval.
         Multiple spikes per interval are supported, even with coarse sampling.

    2. Sort candidate spike times (should already be sorted, but ensures correctness)
    3. Apply refractory period filter (single pass, O(n)):

       - Initialize last_spike_time = -inf
       - For each candidate in order:

         - If candidate >= last_spike_time + refractory_period:

           - Keep spike, update last_spike_time = candidate

         - Else: discard spike (too soon after last spike)

    This ensures minimum inter-spike interval (ISI) >= refractory_period,
    matching biological absolute refractory period dynamics.

    With ``refractory_period=0``, this is an exact piecewise-constant Poisson
    process. A positive refractory period ``τ`` removes candidates and
    therefore lowers the realized firing rate: for a constant rate ``r`` it
    is ``r / (1 + r τ)``, as each kept spike is followed by a dead time ``τ``
    and then an exponential wait of mean ``1 / r``.

    The cost is O(n + k log k) for n time points and k candidate spikes.

    Examples
    --------
    Generate spikes from a place cell:

    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import (
    ...     PlaceCellModel,
    ...     simulate_trajectory_ou,
    ...     generate_poisson_spikes,
    ... )
    >>> import numpy as np
    >>>
    >>> # Create environment and place cell at center for reliable firing
    >>> samples = np.random.default_rng(42).uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(samples, bin_size=2.0)
    >>> env = env.with_metadata(units="cm")
    >>> center = env.bin_centers[len(env.bin_centers) // 2]
    >>> pc = PlaceCellModel(env, center=center, width=20.0, max_rate=50.0, seed=42)
    >>>
    >>> # Generate trajectory and firing rates
    >>> positions, times = simulate_trajectory_ou(
    ...     env, duration=60.0, seed=42, speed_units="cm"
    ... )
    >>> rates = pc.firing_rate(positions)
    >>>
    >>> # Generate spikes
    >>> spike_times = generate_poisson_spikes(rates, times, seed=42)
    >>> len(spike_times) > 0  # Should generate some spikes
    True
    >>>
    >>> # Verify refractory period constraint if spikes were generated
    >>> if len(spike_times) > 1:
    ...     isi = np.diff(spike_times)
    ...     assert np.all(isi >= 0.002), "All ISIs should be >= 2ms"

    Reproduce results with same seed:

    >>> spikes1 = generate_poisson_spikes(rates, times, seed=42)
    >>> spikes2 = generate_poisson_spikes(rates, times, seed=42)
    >>> np.array_equal(spikes1, spikes2)  # Identical
    True

    See Also
    --------
    generate_population_spikes : Generate spikes for multiple neurons
    PlaceCellModel.firing_rate : Compute place cell firing rates
    simulate_trajectory_ou : Generate trajectory for spike generation

    References
    ----------
    Dayan & Abbott (2001), Theoretical Neuroscience, Chapter 1
    """
    # Initialize random number generator
    rng = np.random.default_rng(seed)

    times = np.asarray(times, dtype=np.float64)
    firing_rate = np.asarray(firing_rate, dtype=np.float64)

    if times.ndim != 1 or firing_rate.ndim != 1:
        raise ValueError(
            "firing_rate and times must be 1-D arrays, got "
            f"firing_rate.shape={firing_rate.shape}, times.shape={times.shape}."
        )
    if not np.isfinite(refractory_period) or refractory_period < 0:
        raise ValueError("refractory_period must be finite and non-negative.")

    if len(firing_rate) != len(times):
        raise ValueError(
            f"firing_rate and times must have the same length, got "
            f"firing_rate={len(firing_rate)}, times={len(times)}."
        )

    firing_rate = validate_finite(firing_rate, name="firing_rate")
    negative = np.flatnonzero(firing_rate < 0)
    if len(negative):
        first = int(negative[0])
        raise ValueError(
            _format_error(
                f"firing_rate must be non-negative, got {len(negative)} negative "
                f"value(s); the first is {float(firing_rate[first])!r} Hz at index {first}.",
                why="a Poisson rate is an expected number of spikes per second, "
                "which cannot be below zero.",
                fix="check how the rate was computed (for example a baseline "
                "above the peak, or a modulation deeper than the rate); clip it "
                "with np.clip(rate, 0, None) only if the negative values are "
                "rounding noise",
            )
        )

    if not np.all(np.isfinite(times)):
        n_bad = int(np.sum(~np.isfinite(times)))
        raise ValueError(f"times must be finite, got {n_bad} NaN/inf entr(y/ies).")

    if len(times) < 2:
        return np.array([], dtype=np.float64)

    # Per-bin dt: the inhomogeneous Poisson count uses the actual
    # spacing of each interval, so non-uniform sampling is handled correctly.
    # Require strictly increasing times: a zero-width interval has no spike
    # probability and a decreasing interval is meaningless for a forward-time
    # point process.
    bin_dt = np.diff(times)
    if np.any(bin_dt <= 0.0):
        bad_idx = np.where(bin_dt <= 0.0)[0]
        raise ValueError(
            "times must be strictly increasing. Found "
            f"{len(bad_idx)} non-increasing interval(s) at indices: "
            f"{bad_idx.tolist()[:5]}" + (" ..." if len(bad_idx) > 5 else "")
        )

    # Counts, rather than Bernoulli occupancy, retain Poisson multiplicities.
    # Uniform conditional event times give the exact process for the specified
    # piecewise-constant rate, including nonuniform sampling intervals.
    counts = rng.poisson(firing_rate[:-1] * bin_dt)
    interval_ids = np.repeat(np.arange(len(bin_dt)), counts)
    candidate_spikes = (
        times[interval_ids] + rng.random(len(interval_ids)) * bin_dt[interval_ids]
    )
    # Floating-point addition can round a time onto its interval's right edge.
    candidate_spikes = np.minimum(
        candidate_spikes, np.nextafter(times[interval_ids + 1], times[interval_ids])
    )

    # Sort candidate spikes (should already be sorted, but ensure correctness)
    candidate_spikes = np.sort(candidate_spikes)

    # Apply refractory period filter (single pass, O(n) algorithm)
    if len(candidate_spikes) == 0:
        return np.array([], dtype=np.float64)

    # Keep track of accepted spikes
    accepted_spikes = []
    last_spike_time = -np.inf

    for spike_time in candidate_spikes:
        # Check if spike is outside refractory period
        if spike_time >= last_spike_time + refractory_period:
            accepted_spikes.append(spike_time)
            last_spike_time = spike_time
        # Else: discard spike (within refractory period)

    return np.array(accepted_spikes, dtype=np.float64)


def generate_population_spikes(
    models: list[NeuralModel],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    headings: NDArray[np.float64] | None = None,
    break_before: ArrayLike | None = None,
    refractory_period: float = 0.002,
    seed: int | None = None,
    show_progress: bool = True,
) -> list[NDArray[np.float64]]:
    """Generate spike trains for population of neurons.

    Processes multiple neural models in parallel with progress tracking.
    Each neuron gets an independent spike train based on its firing rate model.

    Parameters
    ----------
    models : list[NeuralModel]
        List of neural models (PlaceCellModel, BorderCellModel, GridCellModel, etc.).
        Each model must implement the NeuralModel protocol.
    times : NDArray[np.float64], shape (n_time,)
        Time points in seconds (must be uniformly spaced).
    positions : NDArray[np.float64], shape (n_time, n_dims)
        Position trajectory in continuous coordinates.
    headings : NDArray[np.float64], shape (n_time,), optional
        Head direction in radians. Required for spatial-view models and
        directional object-vector models. For head-direction models, headings
        are used when provided; otherwise they are the direction of the
        smoothed velocity, held at the last direction of movement through
        stops (samples before the first movement take the first direction).
        A trajectory that never moves then raises ``ValueError``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. Used only to derive head-direction models'
        headings, so the velocity is never smoothed across a break; the held
        direction never carries over a break: each recording starts with its
        own first direction of movement. Not allowed together with
        ``headings``.
    refractory_period : float, optional
        Refractory period for each neuron in seconds (default: 2ms).
    seed : int | None, optional
        Random seed for reproducibility. Each neuron gets a derived seed
        (seed + i) to ensure independence while maintaining reproducibility.
    show_progress : bool, optional
        Show progress bar during spike generation (default: True).
        Set to False for quiet operation in scripts or tests.

    Returns
    -------
    spike_times_list : list[NDArray[np.float64]]
        List of spike time arrays, one per model. Each array contains
        sorted spike times for that neuron.

    Raises
    ------
    ValueError
        If a head-direction model has no ``headings`` and the trajectory never
        moves, so no movement direction can be derived.
    ValueError
        If ``break_before`` is not a Boolean array with one entry per sample,
        or is passed together with ``headings``.

    Examples
    --------
    Generate spikes for population of place cells:

    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import (
    ...     PlaceCellModel,
    ...     simulate_trajectory_ou,
    ...     generate_population_spikes,
    ... )
    >>> import numpy as np
    >>>
    >>> # Create environment
    >>> samples = np.random.default_rng(42).uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(samples, bin_size=2.0)
    >>> env = env.with_metadata(units="cm")
    >>>
    >>> # Create population of place cells
    >>> rng = np.random.default_rng(42)
    >>> place_cells = [
    ...     PlaceCellModel(
    ...         env,
    ...         center=c,
    ...         width=8.0,
    ...         max_rate=rng.uniform(15, 30),
    ...         seed=i,
    ...     )
    ...     for i, c in enumerate(env.bin_centers[::5])
    ... ]
    >>> len(place_cells) > 0  # Should create multiple place cells
    True
    >>>
    >>> # Generate trajectory and spikes
    >>> positions, times = simulate_trajectory_ou(
    ...     env, duration=120.0, seed=42, speed_units="cm"
    ... )
    >>> spike_trains = generate_population_spikes(
    ...     place_cells, times, positions, seed=42, show_progress=False
    ... )
    >>>
    >>> # Verify output structure
    >>> len(spike_trains) == len(place_cells)
    True
    >>> all(isinstance(st, np.ndarray) for st in spike_trains)
    True

    Use with place field detection to validate simulation:

    >>> # Example: Validate place field recovery (actual values vary by trajectory)
    >>> from neurospatial.encoding import compute_spatial_rate  # doctest: +SKIP
    >>> rate_map = compute_spatial_rate(
    ...     env, spike_trains[0], times, positions
    ... ).firing_rate  # doctest: +SKIP
    >>> true_center = place_cells[0].ground_truth["center"]  # doctest: +SKIP
    >>> detected_center = env.bin_centers[np.nanargmax(rate_map)]  # doctest: +SKIP

    Generate spikes quietly (no progress bar):

    >>> spike_trains_quiet = generate_population_spikes(
    ...     place_cells, times, positions, seed=42, show_progress=False
    ... )

    A head-direction cell without ``headings`` fires with the direction of
    movement, held through stops. Here the animal runs East, stops for 2 s and
    runs North:

    >>> from neurospatial.simulation import HeadDirectionCellModel
    >>> stop_times = np.arange(400) / 100  # s
    >>> stop_path = np.column_stack(
    ...     [10.0 * np.clip(stop_times, 0, 1), 10.0 * np.clip(stop_times - 3, 0, 1)]
    ... )  # cm
    >>> hd_spikes = generate_population_spikes(
    ...     [HeadDirectionCellModel(preferred_direction=0.0)],
    ...     stop_times,
    ...     stop_path,
    ...     seed=0,
    ...     show_progress=False,
    ... )
    >>> len(hd_spikes[0]) > 0
    True

    See Also
    --------
    generate_poisson_spikes : Generate spikes for single neuron
    PlaceCellModel : Place cell firing rate model
    BorderCellModel : Border cell firing rate model
    GridCellModel : Grid cell firing rate model

    Notes
    -----
    **Progress Bar**: Uses tqdm to show:

    - Current cell being processed
    - Number of spikes generated for current cell
    - Mean firing rate for current cell

    **Reproducibility**: Each model gets a derived seed (seed + i) to ensure:

    - Independent spike trains across neurons
    - Reproducible results with same seed
    - Consistent behavior across runs

    **Performance**: For 50 cells × 6000 time points, runs in ~5 seconds.
    Use ``show_progress=False`` in tight loops or tests to avoid overhead.
    """
    # Initialize random seed handling
    from neurospatial._validation import validate_times_positions

    times, positions = validate_times_positions(
        times, positions, call="generate_population_spikes"
    )
    from neurospatial._exceptions import _format_error
    from neurospatial._observation import validate_break_before

    breaks = validate_break_before(
        break_before, len(times), call="generate_population_spikes"
    )
    if breaks is not None and headings is not None:
        raise ValueError(
            _format_error(
                "generate_population_spikes() got both headings and break_before.",
                why="break_before only keeps the heading derived from the "
                "velocity from crossing a recording break; given headings are "
                "used as they are.",
                fix="pass headings, or break_before to derive them, not both.",
            )
        )
    base_seed = seed

    # Pre-allocate spike trains list
    spike_trains = []

    # Create progress bar
    iterator = tqdm(
        enumerate(models),
        total=len(models),
        desc="Generating spikes",
        disable=not show_progress,
    )

    # Generate spikes for each model
    for i, model in iterator:
        # Compute firing rates for this model
        rates = _compute_model_firing_rate(
            model, positions, times, headings, break_before=breaks
        )

        # Generate spikes with derived seed for reproducibility
        model_seed = None if base_seed is None else base_seed + i
        spikes = generate_poisson_spikes(
            rates, times, refractory_period=refractory_period, seed=model_seed
        )

        spike_trains.append(spikes)

        # Update progress bar with cell-specific stats
        mean_rate = len(spikes) / times[-1] if len(times) > 0 else 0.0
        iterator.set_postfix(
            {
                "n_spikes": len(spikes),
                "rate": f"{mean_rate:.1f} Hz",
            }
        )

    # Print summary after completion
    if show_progress:
        total_spikes = sum(len(st) for st in spike_trains)
        avg_spikes_per_cell = total_spikes / len(models) if len(models) > 0 else 0.0
        duration = times[-1] if len(times) > 0 else 1.0
        mean_rate = total_spikes / (len(models) * duration) if len(models) > 0 else 0.0

        logger.info(
            "Generated %d cells, %d total spikes "
            "(avg %.0f spikes/cell), mean rate %.1f Hz",
            len(models),
            total_spikes,
            avg_spikes_per_cell,
            mean_rate,
        )

    return spike_trains


def _compute_model_firing_rate(
    model: NeuralModel,
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    headings: NDArray[np.float64] | None,
    *,
    break_before: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Compute firing rates for models with different input conventions."""
    from neurospatial.simulation.models.head_direction_cells import (
        HeadDirectionCellModel,
    )
    from neurospatial.simulation.models.object_vector_cells import (
        ObjectVectorCellModel,
    )
    from neurospatial.simulation.models.spatial_view_cells import SpatialViewCellModel

    model_any: Any = model

    if isinstance(model_any, HeadDirectionCellModel):
        if headings is None:
            if len(times) < 2:
                return np.zeros(len(times), dtype=np.float64)
            headings = _held_movement_heading(
                times, positions, break_before=break_before
            )
        rates = model_any.firing_rate(headings, positions=positions, times=times)
    elif isinstance(model_any, SpatialViewCellModel | ObjectVectorCellModel):
        rates = model_any.firing_rate(positions, times, headings=headings)
    else:
        rates = model_any.firing_rate(positions, times)

    rates = np.asarray(rates, dtype=np.float64)
    if len(rates) != len(times):
        raise ValueError(
            f"Model {type(model).__name__}.firing_rate returned {len(rates)} rates "
            f"for {len(times)} time points."
        )
    return rates


def _held_movement_heading(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    break_before: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Movement direction at every sample, held through stops.

    A simulated head-direction cell needs a heading at every sample, including
    where the animal is still and its movement direction is not defined. The
    last defined direction is held forward through each stop, and samples
    before a recording's first movement take that recording's first defined
    direction. Recordings are split only at declared breaks; nothing is held
    across one.

    Parameters
    ----------
    times : ndarray, shape (n_time,)
        Seconds, strictly increasing.
    positions : ndarray, shape (n_time, n_dims)
        Positions in environment units.
    break_before : ndarray of bool, shape (n_time,), optional
        Already validated declared recording breaks; the velocity is not
        smoothed across them and the held direction does not carry over them.

    Returns
    -------
    ndarray, shape (n_time,)
        Heading in radians (0 = East, pi/2 = North), finite everywhere.

    Raises
    ------
    ValueError
        If a recording never has a movement direction: it never moves, or
        every gap between its samples is longer than the default
        ``max_gap`` (1.0 s).

    Examples
    --------
    Two recordings: the first heads East then stops, the second is still and
    then heads North. The second keeps its own first direction, not East.

    >>> import numpy as np
    >>> from neurospatial.simulation.spikes import _held_movement_heading
    >>> times = np.arange(40) / 10
    >>> x = np.r_[np.arange(10), np.full(10, 9.0), np.full(20, 50.0)]
    >>> y = np.r_[np.zeros(30), np.arange(10)]
    >>> held = _held_movement_heading(
    ...     times, np.c_[x, y], break_before=np.arange(40) == 20
    ... )
    >>> print(f"{held[9]:.3f} {held[20]:.3f}")
    0.000 1.571
    """
    from neurospatial._exceptions import _format_error
    from neurospatial.behavior import heading_from_velocity

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Every observed sample's speed", category=UserWarning
        )
        warnings.filterwarnings(
            "ignore", message="Interval filtering excluded ALL", category=UserWarning
        )
        heading = heading_from_velocity(
            times, positions, break_before=break_before, allow_all_nan=True
        )
    defined = np.isfinite(heading)
    n = heading.size
    breaks = np.zeros(n, dtype=bool)
    if break_before is not None:
        breaks[:] = break_before
    breaks[0] = False
    recording = np.cumsum(breaks)
    starts = np.flatnonzero(np.r_[True, breaks[1:]])
    ends = np.r_[starts[1:], n]
    idx = np.arange(n)
    # Last defined sample at or before each sample, valid inside its recording.
    last = np.maximum.accumulate(np.where(defined, idx, -1))
    # First defined sample at or after each sample (the back-fill source).
    following = np.minimum.accumulate(np.where(defined, idx, n)[::-1])[::-1]
    first_in_recording = following[starts]
    for r in np.flatnonzero(first_in_recording >= ends):
        first_t, last_t = times[starts[r]], times[ends[r] - 1]
        raise ValueError(
            _format_error(
                "Cannot derive head direction for HeadDirectionCellModel: no "
                f"movement direction is defined in recording {r + 1} (samples "
                f"{starts[r]} to {ends[r] - 1}, {first_t:g} to {last_t:g} s), "
                "because it never moves or no interval between its samples is "
                "short enough to use (gaps longer than "
                f"max_gap={DEFAULT_MAX_GAP} s are not used).",
                why="without headings=, the simulated heading is the direction "
                "of the velocity, held within each recording.",
                fix="pass headings= with one angle in radians per sample, "
                "e.g. headings=np.zeros(len(times)) for an animal facing East.",
            )
        )
    source = np.where(last >= starts[recording], last, first_in_recording[recording])
    held: NDArray[np.float64] = heading[source]
    return held


def add_modulation(
    spike_times: NDArray[np.float64],
    *,
    modulation_freq: float,
    modulation_depth: float = 0.5,
    modulation_phase: float = 0.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Add rhythmic modulation to spike train (e.g., theta oscillation).

    Implements non-homogeneous thinning to create phase-locked firing patterns.
    This simulates the tendency of neurons to fire at specific phases of ongoing
    oscillations, common in hippocampus (theta), cortex (gamma), and other regions.

    Parameters
    ----------
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Original spike times in seconds.
    modulation_freq : float
        Modulation frequency in Hz (e.g., 8 Hz for hippocampal theta,
        40 Hz for cortical gamma).
    modulation_depth : float, optional
        Modulation strength from 0 (no modulation) to 1 (full modulation).
        Default: 0.5 (moderate modulation).
        - 0.0: No modulation, all spikes kept
        - 0.5: Moderate phase locking
        - 1.0: Strong phase locking, spikes at anti-preferred phases removed
    modulation_phase : float, optional
        Phase offset in radians (default: 0.0). Controls preferred firing phase.
        - 0.0: Spikes preferred at cosine peaks
        - π/2: Spikes preferred at sine peaks
        - π: Spikes preferred at cosine troughs
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    modulated_spike_times : NDArray[np.float64], shape (n_modulated_spikes,)
        Spike times after rhythmic modulation (subset of original spikes).
        Spikes are kept in sorted order.

    Notes
    -----
    **Algorithm (non-homogeneous thinning)**:

    1. Compute phase of each spike:

       phase[i] = 2π * freq * spike_times[i] + phase_offset

    2. Compute acceptance probability based on cosine modulation:

       p_accept[i] = (1 + depth * cos(phase[i])) / 2

       This ensures:
       - p_accept ∈ [0, 1] for all phases
       - At preferred phase (cos = 1): p = (1 + depth) / 2
       - At anti-preferred phase (cos = -1): p = (1 - depth) / 2

    3. Randomly keep or discard each spike based on acceptance probability

    **Biological Motivation**: Many neurons show phase-locked firing to local
    field potential oscillations. Examples:

    - Hippocampal place cells preferentially fire on theta trough (~180°)
    - Interneurons often fire on theta peak (0°)
    - Cortical neurons show gamma phase locking (30-80 Hz)

    **Performance**: O(n) time complexity for n spikes. Typical processing
    time <10ms for 1000 spikes.

    Examples
    --------
    Add theta modulation to place cell spikes:

    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import (
    ...     PlaceCellModel,
    ...     simulate_trajectory_ou,
    ...     generate_poisson_spikes,
    ...     add_modulation,
    ... )
    >>> import numpy as np
    >>>
    >>> # Create environment and place cell
    >>> samples = np.random.default_rng(42).uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(samples, bin_size=2.0)
    >>> env = env.with_metadata(units="cm")
    >>> center = env.bin_centers[len(env.bin_centers) // 2]
    >>> pc = PlaceCellModel(env, center=center, width=20.0, max_rate=50.0, seed=42)
    >>>
    >>> # Generate trajectory and spikes
    >>> positions, times = simulate_trajectory_ou(
    ...     env, duration=60.0, seed=42, speed_units="cm"
    ... )
    >>> rates = pc.firing_rate(positions)
    >>> spike_times = generate_poisson_spikes(rates, times, seed=42)
    >>>
    >>> # Add theta modulation (8 Hz, moderate depth)
    >>> if len(spike_times) > 0:
    ...     theta_modulated = add_modulation(
    ...         spike_times,
    ...         modulation_freq=8.0,
    ...         modulation_depth=0.7,
    ...         seed=42,
    ...     )
    ...     # Modulation reduces spike count
    ...     assert len(theta_modulated) < len(spike_times)
    ...     # Modulated spikes are subset of original
    ...     assert all(t in spike_times for t in theta_modulated)

    Compare modulation depths:

    >>> spike_times = np.linspace(0, 10, 500)  # Uniform spike train
    >>> no_mod = add_modulation(
    ...     spike_times, modulation_freq=8.0, modulation_depth=0.0, seed=42
    ... )
    >>> len(no_mod) == len(spike_times)  # No modulation keeps all spikes
    True
    >>> strong_mod = add_modulation(
    ...     spike_times, modulation_freq=8.0, modulation_depth=0.9, seed=42
    ... )
    >>> len(strong_mod) < len(no_mod)  # Strong modulation removes spikes
    True

    See Also
    --------
    generate_poisson_spikes : Generate initial spike train
    PlaceCellModel.firing_rate : Compute place cell firing rates

    References
    ----------
    O'Keefe & Recce (1993), Hippocampus - Theta phase precession
    Buzsáki (2002), Neuron - Theta oscillations review
    """
    # Handle empty spike train
    if len(spike_times) == 0:
        return np.array([], dtype=np.float64)

    # Validate parameters
    if not 0.0 <= modulation_depth <= 1.0:
        raise ValueError(
            f"modulation_depth must be in range [0, 1], got {modulation_depth}"
        )

    if modulation_freq <= 0.0:
        raise ValueError(
            f"modulation_freq must be positive (Hz), got {modulation_freq}"
        )

    # Special case: zero modulation depth means no modulation (keep all spikes)
    if modulation_depth == 0.0:
        return np.array(spike_times, dtype=np.float64, copy=True)

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Compute phase of each spike time
    # phase = 2π * frequency * time + phase_offset
    phases = 2 * np.pi * modulation_freq * spike_times + modulation_phase

    # Compute acceptance probability using cosine modulation
    # p_accept = (1 + depth * cos(phase)) / 2
    # This ensures p ∈ [0, 1] for all phases
    acceptance_prob = (1.0 + modulation_depth * np.cos(phases)) / 2.0

    # Thin spikes: keep spike if random value < acceptance probability
    random_values = rng.random(len(spike_times))
    keep_mask = random_values < acceptance_prob

    # Return modulated spikes (already sorted since input is sorted)
    modulated_spikes = spike_times[keep_mask]

    return modulated_spikes
