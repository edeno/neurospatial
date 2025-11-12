"""High-level session simulation and dataclass containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.simulation.models import NeuralModel


@dataclass(frozen=True)
class SimulationSession:
    """Complete simulation session result.

    This dataclass encapsulates all components of a simulated recording session,
    including the spatial environment, trajectory, spike trains, neural models,
    and ground truth parameters. It provides a convenient container for passing
    simulation results between functions and for validation workflows.

    Attributes
    ----------
    env : Environment
        The spatial environment in which the simulation occurred.
    positions : NDArray[np.float64], shape (n_time, n_dims)
        Trajectory positions in continuous coordinates. Each row is a position
        at a specific time point.
    times : NDArray[np.float64], shape (n_time,)
        Time points in seconds corresponding to each position.
    spike_trains : list[NDArray[np.float64]]
        List of spike time arrays, one per neuron. Each array contains the
        spike times (in seconds) for a single neuron.
    models : list[NeuralModel]
        Neural model instances used to generate the spikes. These implement
        the NeuralModel protocol and can be used to regenerate firing rates.
    ground_truth : dict[str, Any]
        True parameters for each cell, indexed by cell identifier. This
        typically contains ground truth from each model's `.ground_truth`
        property, enabling validation of analysis methods.
    metadata : dict[str, Any]
        Session parameters and configuration. This can include simulation
        settings (duration, trajectory method, cell type), experimental
        parameters, or any other relevant information.

    Notes
    -----
    This dataclass is frozen (immutable) to prevent accidental modification
    of simulation results. To create a modified session, use dataclasses.replace():

        >>> from dataclasses import replace
        >>> new_session = replace(session, metadata={...})  # doctest: +SKIP

    The separation of `models` and `ground_truth` allows flexible usage:
    - Use `models` to regenerate firing rates or extend simulations
    - Use `ground_truth` for validation without needing to keep model instances

    Examples
    --------
    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import (
    ...     simulate_trajectory_ou,
    ...     PlaceCellModel,
    ...     generate_population_spikes,
    ...     SimulationSession,
    ... )

    >>> # Create environment
    >>> env = Environment.from_samples(arena_data, bin_size=2.0)
    >>> env.units = "cm"

    >>> # Generate trajectory
    >>> positions, times = simulate_trajectory_ou(env, duration=60.0)

    >>> # Create place cells
    >>> models = [
    ...     PlaceCellModel(env, center=[50.0, 50.0], width=10.0),
    ...     PlaceCellModel(env, center=[70.0, 60.0], width=8.0),
    ... ]

    >>> # Generate spikes
    >>> spike_trains = generate_population_spikes(models, positions, times)

    >>> # Collect ground truth
    >>> ground_truth = {
    ...     f"cell_{i}": model.ground_truth for i, model in enumerate(models)
    ... }

    >>> # Create session
    >>> session = SimulationSession(
    ...     env=env,
    ...     positions=positions,
    ...     times=times,
    ...     spike_trains=spike_trains,
    ...     models=models,
    ...     ground_truth=ground_truth,
    ...     metadata={"duration": 60.0, "cell_type": "place"},
    ... )

    >>> # Access fields with typed attributes
    >>> print(f"Session duration: {session.times[-1]:.1f}s")
    >>> print(f"Number of cells: {len(session.spike_trains)}")
    >>> print(f"Total spikes: {sum(len(st) for st in session.spike_trains)}")

    See Also
    --------
    simulate_session : High-level function to create a complete session
    validate_simulation : Validate session against ground truth
    plot_session_summary : Visualize session data
    """

    env: Environment
    positions: NDArray[np.float64]
    times: NDArray[np.float64]
    spike_trains: list[NDArray[np.float64]]
    models: list[NeuralModel]
    ground_truth: dict[str, Any]
    metadata: dict[str, Any]


def simulate_session(
    env: Environment,
    duration: float,
    n_cells: int = 50,
    cell_type: Literal["place", "boundary", "grid", "mixed"] = "place",
    trajectory_method: Literal["ou", "sinusoidal", "laps"] = "ou",
    coverage: Literal["uniform", "random"] = "uniform",
    show_progress: bool = True,
    seed: int | None = None,
    **kwargs: Any,
) -> SimulationSession:
    """Simulate complete recording session with trajectory and neural activity.

    High-level convenience function that combines trajectory simulation,
    neural model creation, and spike generation in one call.

    Parameters
    ----------
    env : Environment
        The spatial environment.
    duration : float
        Session duration in seconds.
    n_cells : int, optional
        Number of neurons to simulate (default: 50).
    cell_type : {'place', 'boundary', 'grid', 'mixed'}, optional
        Type of neurons to create (default: 'place').
        - 'place': All place cells
        - 'boundary': All boundary cells
        - 'grid': All grid cells (requires 2D environment)
        - 'mixed': 60% place, 20% boundary, 20% grid
    trajectory_method : {'ou', 'sinusoidal', 'laps'}, optional
        Trajectory generation method (default: 'ou').
    coverage : {'uniform', 'random'}, optional
        Spatial distribution of field centers (default: 'uniform').
        - 'uniform': Evenly space field centers across environment using
          env.bin_centers[::step] where step = max(1, n_bins // n_cells).
          Ensures full coverage without clustering.
        - 'random': Randomly sample field centers from env.bin_centers
          using np.random.choice(env.bin_centers, size=n_cells).
          May result in clustering or gaps in coverage.
    show_progress : bool, optional
        Show progress bars during generation (default: True).
    seed : int | None, optional
        Random seed for reproducibility.
    **kwargs
        Additional parameters passed to trajectory and model functions.
        Examples: speed_mean, coherence_time, max_rate, etc.

    Returns
    -------
    session : SimulationSession
        Dataclass containing:
        - env: Environment instance
        - positions: NDArray, shape (n_time, n_dims) - trajectory
        - times: NDArray, shape (n_time,) - time points
        - spike_trains: list[NDArray] - spike times per cell
        - models: list[NeuralModel] - neural model instances
        - ground_truth: dict - true parameters for each cell
        - metadata: dict - session parameters

    Raises
    ------
    ValueError
        If cell_type is not one of: 'place', 'boundary', 'grid', 'mixed'.
    ValueError
        If trajectory_method is not one of: 'ou', 'sinusoidal', 'laps'.
    ValueError
        If coverage is not one of: 'uniform', 'random'.
    ValueError
        If grid cells requested but environment is not 2D.

    Examples
    --------
    >>> # Quick place cell session
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> arena_data = np.random.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(arena_data, bin_size=2.0)
    >>> env.units = "cm"
    >>> session = simulate_session(env, duration=2.0, n_cells=3, show_progress=False)
    >>> positions = session.positions  # Typed access
    >>> spike_trains = session.spike_trains  # IDE autocomplete
    >>> ground_truth = session.ground_truth  # Discoverable

    >>> # Mixed cell types with custom parameters
    >>> session = simulate_session(
    ...     env,
    ...     duration=2.0,
    ...     n_cells=10,
    ...     cell_type="mixed",
    ...     trajectory_method="ou",
    ...     speed_mean=0.1,
    ...     coherence_time=0.8,
    ...     show_progress=False,
    ... )

    >>> # Validate detected vs true place fields
    >>> from neurospatial.simulation import validate_simulation
    >>> report = validate_simulation(session)
    >>> print(report["summary"])

    See Also
    --------
    simulate_trajectory_ou : Low-level trajectory simulation
    PlaceCellModel : Individual place cell model
    generate_population_spikes : Low-level spike generation
    validate_simulation : Validate simulation results
    """
    # Import functions here to avoid circular imports
    from neurospatial.simulation.models import (
        BoundaryCellModel,
        GridCellModel,
        PlaceCellModel,
    )
    from neurospatial.simulation.spikes import generate_population_spikes
    from neurospatial.simulation.trajectory import (
        simulate_trajectory_laps,
        simulate_trajectory_ou,
        simulate_trajectory_sinusoidal,
    )

    # Validate parameters
    valid_cell_types = {"place", "boundary", "grid", "mixed"}
    if cell_type not in valid_cell_types:
        raise ValueError(
            f"cell_type must be one of {valid_cell_types}, got '{cell_type}'"
        )

    valid_trajectory_methods = {"ou", "sinusoidal", "laps"}
    if trajectory_method not in valid_trajectory_methods:
        raise ValueError(
            f"trajectory_method must be one of {valid_trajectory_methods}, "
            f"got '{trajectory_method}'"
        )

    valid_coverage = {"uniform", "random"}
    if coverage not in valid_coverage:
        raise ValueError(f"coverage must be one of {valid_coverage}, got '{coverage}'")

    # Check for grid cells in non-2D environments
    if cell_type in ("grid", "mixed") and env.n_dims != 2:
        raise ValueError(
            f"Grid cells require 2D environment, but environment has {env.n_dims} dimensions"
        )

    # Validate n_cells and duration
    if n_cells <= 0:
        raise ValueError(f"n_cells must be positive, got {n_cells}")

    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")

    # Generate trajectory (use seed directly for reproducibility)
    if trajectory_method == "ou":
        positions, times = simulate_trajectory_ou(
            env, duration=duration, seed=seed, **kwargs
        )
    elif trajectory_method == "sinusoidal":
        positions, times = simulate_trajectory_sinusoidal(
            env, duration=duration, seed=seed, **kwargs
        )
    elif trajectory_method == "laps":
        # Ensure return_metadata is False for consistent return signature
        # Extract n_laps separately to avoid duplicate keyword argument
        n_laps = kwargs.pop("n_laps", 10)
        laps_kwargs = {k: v for k, v in kwargs.items() if k != "return_metadata"}
        result = simulate_trajectory_laps(
            env,
            n_laps=n_laps,
            seed=seed,
            return_metadata=False,
            **laps_kwargs,
        )
        # Type narrowing: return_metadata=False guarantees 2-tuple
        positions, times = result  # type: ignore[misc]
    else:
        raise ValueError(f"Unknown trajectory method: {trajectory_method}")

    # Initialize RNG for field center selection (use seed+1 to avoid collision with trajectory seed)
    rng = np.random.default_rng(seed + 1 if seed is not None else None)

    # Generate field centers based on coverage
    if coverage == "uniform":
        # Evenly space centers across environment
        step = max(1, len(env.bin_centers) // n_cells)
        field_centers = env.bin_centers[::step][:n_cells]
    elif coverage == "random":
        # Randomly sample centers (with replacement if needed)
        indices = rng.choice(len(env.bin_centers), size=n_cells, replace=True)
        field_centers = env.bin_centers[indices]
    else:
        raise ValueError(f"Unknown coverage: {coverage}")

    # Create models based on cell_type
    models: list[NeuralModel] = []
    if cell_type == "place":
        for center in field_centers:
            models.append(PlaceCellModel(env, center=center))
    elif cell_type == "boundary":
        for _ in range(n_cells):
            models.append(BoundaryCellModel(env))
    elif cell_type == "grid":
        for center in field_centers:
            # Use center for phase offset
            models.append(GridCellModel(env, phase_offset=center))
    elif cell_type == "mixed":
        # 60% place, 20% boundary, 20% grid
        n_place = int(0.6 * n_cells)
        n_boundary = int(0.2 * n_cells)
        n_grid = n_cells - n_place - n_boundary  # Remainder goes to grid

        # Create place cells
        for center in field_centers[:n_place]:
            models.append(PlaceCellModel(env, center=center))

        # Create boundary cells
        for _ in range(n_boundary):
            models.append(BoundaryCellModel(env))

        # Create grid cells
        for center in field_centers[n_place : n_place + n_grid]:
            models.append(GridCellModel(env, phase_offset=center))
    else:
        raise ValueError(f"Unknown cell_type: {cell_type}")

    # Generate spikes for all cells
    spike_trains = generate_population_spikes(
        models,
        positions,
        times,
        seed=seed,
        show_progress=show_progress,
    )

    # Collect ground truth from each model
    ground_truth = {f"cell_{i}": model.ground_truth for i, model in enumerate(models)}

    # Create metadata dict
    metadata = {
        "duration": duration,
        "n_cells": n_cells,
        "cell_type": cell_type,
        "trajectory_method": trajectory_method,
        "coverage": coverage,
        "seed": seed,
        **kwargs,  # Include any additional parameters
    }

    # Return SimulationSession
    return SimulationSession(
        env=env,
        positions=positions,
        times=times,
        spike_trains=spike_trains,
        models=models,
        ground_truth=ground_truth,
        metadata=metadata,
    )
