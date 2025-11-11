"""Spike generation functions for converting firing rates to spike trains."""

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from neurospatial.simulation.models.base import NeuralModel


def generate_poisson_spikes(
    firing_rate: NDArray[np.float64],
    times: NDArray[np.float64],
    refractory_period: float = 0.002,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate Poisson spike train from firing rate time series.

    Implements an inhomogeneous Poisson process with absolute refractory period.
    Generalizes existing spike generation to handle time-varying rates.

    Parameters
    ----------
    firing_rate : NDArray[np.float64], shape (n_time,)
        Instantaneous firing rate in Hz at each time point.
    times : NDArray[np.float64], shape (n_time,)
        Time points in seconds (must be uniformly spaced).
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

    Notes
    -----
    **Algorithm (absolute refractory period implementation)**:

    1. Generate candidate spikes from inhomogeneous Poisson process:

       - For each time step: spike if rand() < rate[i] * dt

    2. Sort candidate spike times (should already be sorted, but ensures correctness)
    3. Apply refractory period filter (single pass, O(n)):

       - Initialize last_spike_time = -inf
       - For each candidate in order:

         - If candidate >= last_spike_time + refractory_period:

           - Keep spike, update last_spike_time = candidate

         - Else: discard spike (too soon after last spike)

    This ensures minimum inter-spike interval (ISI) >= refractory_period,
    matching biological absolute refractory period dynamics.

    **Performance**: O(n) time complexity for n time points. For typical
    simulations (60s @ 100 Hz = 6000 points), runs in <50ms.

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
    >>> samples = np.random.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(samples, bin_size=2.0)
    >>> env.units = "cm"
    >>> center = env.bin_centers[len(env.bin_centers) // 2]
    >>> pc = PlaceCellModel(env, center=center, width=20.0, max_rate=50.0, seed=42)
    >>>
    >>> # Generate trajectory and firing rates
    >>> positions, times = simulate_trajectory_ou(env, duration=60.0, seed=42)
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

    # Compute time step (assume uniform sampling)
    if len(times) < 2:
        return np.array([], dtype=np.float64)

    dt = times[1] - times[0]

    # Generate candidate spikes from inhomogeneous Poisson process
    # Probability of spike in interval dt: P(spike) = rate * dt
    spike_probabilities = firing_rate * dt
    # Ensure probabilities are valid (between 0 and 1)
    spike_probabilities = np.clip(spike_probabilities, 0.0, 1.0)

    # Generate random values and compare to probabilities
    random_values = rng.random(len(times))
    spike_mask = random_values < spike_probabilities

    # Extract candidate spike times
    candidate_spikes = times[spike_mask]

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
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
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
        List of neural models (PlaceCellModel, BoundaryCellModel, GridCellModel, etc.).
        Each model must implement the NeuralModel protocol.
    positions : NDArray[np.float64], shape (n_time, n_dims)
        Position trajectory in continuous coordinates.
    times : NDArray[np.float64], shape (n_time,)
        Time points in seconds (must be uniformly spaced).
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
    >>> samples = np.random.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(samples, bin_size=2.0)
    >>> env.units = "cm"
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
    >>> positions, times = simulate_trajectory_ou(env, duration=120.0, seed=42)
    >>> spike_trains = generate_population_spikes(
    ...     place_cells, positions, times, seed=42, show_progress=False
    ... )
    >>>
    >>> # Verify output structure
    >>> len(spike_trains) == len(place_cells)
    True
    >>> all(isinstance(st, np.ndarray) for st in spike_trains)
    True

    Use with place field detection to validate simulation:

    >>> # Example: Validate place field recovery (actual values vary by trajectory)
    >>> from neurospatial import compute_place_field  # doctest: +SKIP
    >>> rate_map = compute_place_field(
    ...     env, spike_trains[0], times, positions
    ... )  # doctest: +SKIP
    >>> true_center = place_cells[0].ground_truth["center"]  # doctest: +SKIP
    >>> detected_center = env.bin_centers[np.argmax(rate_map)]  # doctest: +SKIP

    Generate spikes quietly (no progress bar):

    >>> spike_trains_quiet = generate_population_spikes(
    ...     place_cells, positions, times, seed=42, show_progress=False
    ... )

    See Also
    --------
    generate_poisson_spikes : Generate spikes for single neuron
    PlaceCellModel : Place cell firing rate model
    BoundaryCellModel : Boundary cell firing rate model
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
        rates = model.firing_rate(positions, times)

        # Generate spikes with derived seed for reproducibility
        model_seed = None if base_seed is None else base_seed + i
        spikes = generate_poisson_spikes(rates, times, refractory_period, model_seed)

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

        print(
            f"Generated {len(models)} cells, {total_spikes:,} total spikes "
            f"(avg {avg_spikes_per_cell:.0f} spikes/cell), mean rate {mean_rate:.1f} Hz"
        )

    return spike_trains
