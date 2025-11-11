"""High-level session simulation and dataclass containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
        >>> new_session = replace(session, metadata={...})

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
