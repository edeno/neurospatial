"""Spike generation functions for converting firing rates to spike trains."""

import logging
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm

from neurospatial._exceptions import _format_error
from neurospatial._intervals import DEFAULT_MAX_GAP
from neurospatial._validation import validate_finite
from neurospatial.simulation._seeds import _stream_seeds
from neurospatial.simulation.models.base import NeuralModel
from neurospatial.simulation.models.boundary_cells import _shared_ray_distances

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
    >>> rates = pc.firing_rate(times, positions)
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

    Each model's rate is computed along the trajectory and turned into an
    independent Poisson spike train, one model after another, with optional
    progress tracking.

    Parameters
    ----------
    models : list[NeuralModel]
        List of neural models (PlaceCellModel, BorderCellModel, GridCellModel, etc.).
        Each model must implement the NeuralModel protocol.
    times : NDArray[np.float64], shape (n_time,)
        Strictly increasing time points in seconds; the spacing may vary.
    positions : NDArray[np.float64], shape (n_time, n_dims)
        Position trajectory in continuous coordinates.
    headings : NDArray[np.float64], shape (n_time,), optional
        Head direction in radians (0 = East, +pi/2 = North), passed to every
        model. When None and a model needs headings (``requires_headings``:
        head-direction and spatial-view cells, egocentric object-vector
        cells), the head is assumed to point where the animal moves: the
        heading is the direction of the smoothed velocity, held at the last
        direction of movement through stops (samples before the first
        movement take the first direction). A trajectory that never moves
        then raises ``ValueError``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. Used only to derive the headings of models
        that need them, so the velocity is never smoothed across a break; the held
        direction never carries over a break: each recording starts with its
        own first direction of movement. Not allowed together with
        ``headings``.
    refractory_period : float, optional
        Refractory period for each neuron in seconds (default: 2ms).
    seed : int | None, optional
        Random seed for reproducibility. Each neuron's spikes come from its
        own child of ``np.random.SeedSequence(seed)``, so the trains are
        independent and no two seeds share a stream.
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
        If a model needs headings, none are given and the trajectory never
        moves, so no movement direction can be derived.
    ValueError
        If a model's rate is not finite at some sample: the message names
        the model and whether positions or headings are missing there.
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

    **Reproducibility**: ``np.random.SeedSequence(seed).spawn(len(models))``
    gives every model its own stream, so trains are independent across
    neurons, the same ``seed`` reproduces them, and runs with different seeds
    share no stream.

    **Headings**: without ``headings``, models that need one see the
    direction of movement, which assumes the animal faces where it goes.
    Pass measured head directions to simulate a head that turns
    independently of the path.
    """
    from neurospatial._observation import validate_break_before
    from neurospatial._validation import validate_times_positions

    times, positions = validate_times_positions(
        times, positions, call="generate_population_spikes"
    )
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
    if len(times) < 2:
        # No sampling interval, so no spike can occur.
        return [np.array([], dtype=np.float64) for _ in models]
    derived = headings is None and any(model.requires_headings for model in models)
    if derived:
        headings = _held_movement_heading(times, positions, break_before=breaks)

    model_seeds = _stream_seeds(seed, len(models))
    duration = float(times[-1] - times[0])
    spike_trains = []
    iterator = tqdm(
        enumerate(models),
        total=len(models),
        desc="Generating spikes",
        disable=not show_progress,
    )
    # Boundary-vector cells on the same walls cast their rays once.
    with _shared_ray_distances():
        for i, model in iterator:
            rates = np.asarray(
                model.firing_rate(times, positions, headings=headings),
                dtype=np.float64,
            )
            _check_model_rates(
                rates, i, model, times, positions, headings, derived_headings=derived
            )
            spikes = generate_poisson_spikes(
                rates,
                times,
                refractory_period=refractory_period,
                seed=model_seeds[i],
            )
            spike_trains.append(spikes)
            iterator.set_postfix(
                {"n_spikes": len(spikes), "rate": f"{len(spikes) / duration:.1f} Hz"}
            )

    if show_progress and models:
        total_spikes = sum(len(st) for st in spike_trains)
        logger.info(
            "Generated %d cells, %d total spikes "
            "(avg %.0f spikes/cell), mean rate %.1f Hz",
            len(models),
            total_spikes,
            total_spikes / len(models),
            total_spikes / (len(models) * duration),
        )

    return spike_trains


def _check_model_rates(
    rates: NDArray[np.float64],
    index: int,
    model: NeuralModel,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64] | None,
    *,
    derived_headings: bool,
) -> None:
    """Refuse a model's rates that are not one finite rate per sample.

    The error names the model and says which input is missing where the rate
    is undefined.
    """
    name = f"Model {index} ({type(model).__name__})"
    if rates.shape != times.shape:
        raise ValueError(
            _format_error(
                f"{name}.firing_rate returned shape {rates.shape} for "
                f"{len(times)} time points.",
                fix="return one rate in Hz per sample, shape (n_time,)",
            )
        )
    undefined = ~np.isfinite(rates)
    if not undefined.any():
        return
    first = int(np.flatnonzero(undefined)[0])
    no_position = undefined & ~np.all(np.isfinite(positions), axis=1)
    no_heading = np.zeros_like(undefined)
    if headings is not None and model.requires_headings:
        no_heading = undefined & ~np.isfinite(headings) & ~no_position
    other = undefined & ~no_position & ~no_heading
    causes = []
    if no_position.any():
        causes.append(f"{int(no_position.sum())} have a missing (NaN) position")
    if no_heading.any():
        source = "derived from the velocity" if derived_headings else "given"
        causes.append(f"{int(no_heading.sum())} have a missing heading ({source})")
    if other.any():
        causes.append(
            f"{int(other.sum())} have a position where the model has no "
            "defined input, such as outside the walls of a wall-driven cell"
        )
    raise ValueError(
        _format_error(
            f"{name} has no defined firing rate at {int(undefined.sum())} of "
            f"{len(times)} samples (first at t = {times[first]:g} s): "
            + "; ".join(causes)
            + ".",
            why="a missing input is not zero activity, so no spikes can be "
            "drawn there without inventing a rate.",
            fix="simulate only samples with a finite position (and heading, "
            "for a model that needs one) inside the environment, for example "
            "by interpolating short tracking gaps or splitting the recording "
            "at long ones",
        )
    )


def _held_movement_heading(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    break_before: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Movement direction at every sample, held through stops.

    A simulated cell that needs headings needs one at every sample, including
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
                "Cannot derive headings for the models that need them: no "
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

    Thins the spike train so that the kept spikes lock to a phase of an
    oscillation, as neurons fire at preferred phases of ongoing rhythms such
    as hippocampal theta or cortical gamma.

    Parameters
    ----------
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Original spike times in seconds, finite and sorted in ascending order.
    modulation_freq : float
        Modulation frequency in Hz (e.g., 8 Hz for hippocampal theta,
        40 Hz for cortical gamma).
    modulation_depth : float, optional
        Modulation strength from 0 (no modulation) to 1 (full modulation).
        Default: 0.5 (moderate modulation).

        - 0.0: No modulation, all spikes kept
        - 0.5: The anti-preferred phase keeps a third as many spikes as the
          preferred phase
        - 1.0: No spikes at the anti-preferred phase
    modulation_phase : float, optional
        Preferred oscillation phase in radians (default: 0.0): the kept
        spikes concentrate where ``2π · modulation_freq · t`` (mod 2π) equals
        it.

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

    Raises
    ------
    ValueError
        If ``modulation_freq`` is not finite and positive,
        ``modulation_depth`` is not in [0, 1], ``modulation_phase`` is not
        finite, or ``spike_times`` is not a 1-D array of finite times in
        ascending order.

    Notes
    -----
    **Algorithm (thinning)**: each spike at time ``t`` is kept with
    probability

        p(t) = (1 + d cos(2π f t - φ)) / (1 + d),

    with ``d = modulation_depth``, ``f = modulation_freq`` and
    ``φ = modulation_phase``. At the preferred phase ``p = 1``, so the rate
    there equals the input rate; at the anti-preferred phase
    ``p = (1 - d) / (1 + d)``. ``p`` changes continuously with ``d`` and is 1
    everywhere at ``d = 0``.

    **Mean rate**: averaged over the oscillation, thinning keeps ``1 / (1 + d)``
    of the spikes, so the mean rate falls by that factor (two thirds of the
    spikes at ``d = 0.5``). To modulate a cell without changing its mean
    rate, modulate the rate itself before drawing spikes,
    ``generate_poisson_spikes(rate * (1 + d * np.cos(2 * np.pi * f * times - φ)), times)``.

    **Biological Motivation**: Many neurons fire phase-locked to local field
    potential oscillations. Hippocampal pyramidal cells fire most near the
    trough of theta recorded in the CA1 pyramidal layer (Buzsáki 2002), and
    cortical neurons lock to gamma (30-80 Hz).

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
    >>> rates = pc.firing_rate(times, positions)
    >>> spike_times = generate_poisson_spikes(rates, times, seed=7)
    >>>
    >>> # Add theta modulation (8 Hz, preferred phase at the theta trough)
    >>> theta_modulated = add_modulation(
    ...     spike_times,
    ...     modulation_freq=8.0,
    ...     modulation_depth=0.7,
    ...     modulation_phase=np.pi,
    ...     seed=42,
    ... )
    >>> bool(np.isin(theta_modulated, spike_times).all())  # a subset
    True
    >>> # About 1 / (1 + 0.7) = 59% of the spikes are kept
    >>> bool(0.5 < len(theta_modulated) / len(spike_times) < 0.7)
    True

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
    Buzsáki, G. (2002). Theta oscillations in the hippocampus. Neuron, 33(3),
        325-340.
    """
    if not (np.isfinite(modulation_freq) and modulation_freq > 0.0):
        raise ValueError(
            _format_error(
                f"modulation_freq must be a finite positive frequency in Hz, got "
                f"{modulation_freq!r}.",
                fix="pass the oscillation frequency, e.g. modulation_freq=8.0 "
                "for theta",
            )
        )
    if not (np.isfinite(modulation_depth) and 0.0 <= modulation_depth <= 1.0):
        raise ValueError(
            _format_error(
                f"modulation_depth must be in [0, 1], got {modulation_depth!r}.",
                fix="pass 0 for no modulation up to 1 for no spikes at the "
                "anti-preferred phase",
            )
        )
    if not np.isfinite(modulation_phase):
        raise ValueError(
            _format_error(
                f"modulation_phase must be a finite angle in radians, got "
                f"{modulation_phase!r}.",
                fix="pass the preferred phase, e.g. modulation_phase=np.pi for "
                "the trough",
            )
        )
    spike_times = np.asarray(spike_times, dtype=np.float64)
    if spike_times.ndim != 1:
        raise ValueError(
            _format_error(
                f"spike_times must be 1-D, got shape {spike_times.shape}.",
                fix="pass one neuron's spike times in seconds",
            )
        )
    spike_times = validate_finite(spike_times, name="spike_times")
    if np.any(np.diff(spike_times) < 0):
        raise ValueError(
            _format_error(
                "spike_times must be sorted in ascending order.",
                fix="pass np.sort(spike_times)",
            )
        )

    rng = np.random.default_rng(seed)
    phases = 2 * np.pi * modulation_freq * spike_times - modulation_phase
    acceptance_prob = (1.0 + modulation_depth * np.cos(phases)) / (
        1.0 + modulation_depth
    )
    keep_mask = rng.random(len(spike_times)) < acceptance_prob
    modulated_spikes: NDArray[np.float64] = spike_times[keep_mask]
    return modulated_spikes
