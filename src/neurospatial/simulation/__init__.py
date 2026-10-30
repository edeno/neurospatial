"""Simulation subpackage for generating synthetic spatial data.

This subpackage provides tools for:
- Trajectory simulation (random walks, laps, sinusoidal motion)
- Neural model implementations (place, border, boundary-vector, grid,
  head-direction, object-vector and spatial-view cells)
- Spike train generation (Poisson process with refractory periods)
- High-level session simulation with validation helpers
- Replay events with a known path, such as sweeps, stationary and random
  events, to test replay methods (``simulate_replay_event``)
- Pre-built maze environments (via mazes submodule)

Examples
--------
Low-level API for custom simulations:

>>> import numpy as np
>>> from neurospatial import Environment
>>> from neurospatial.simulation import (
...     simulate_trajectory_ou,
...     PlaceCellModel,
...     generate_poisson_spikes,
... )
>>>
>>> # Create sample environment
>>> data = np.random.uniform(0, 100, (1000, 2))
>>> env = Environment.from_samples(data, bin_size=2.0)
>>> env = env.with_metadata(units="cm")
>>>
>>> # Generate trajectory
>>> positions, times = simulate_trajectory_ou(
...     env, duration=2.0, seed=42, speed_units="cm"
... )
>>>
>>> # Create place cell at environment center with high rate
>>> center = env.bin_centers[len(env.bin_centers) // 2]
>>> place_cell = PlaceCellModel(env, center=center, width=20.0, max_rate=50.0, seed=42)
>>> rates = place_cell.firing_rate(times, positions)
>>> spikes = generate_poisson_spikes(rates, times, seed=1)  # not the trajectory seed
>>> isinstance(spikes, np.ndarray)  # Returns array of spike times
True

Notes
-----
This subpackage follows the "flat is better than nested" principle. All public
functions and classes are importable directly from `neurospatial.simulation`
for maximum discoverability and IDE support.
"""

# Expose mazes submodule for maze environment creation
from neurospatial.simulation import mazes

# Pre-configured example simulations
from neurospatial.simulation.examples import (
    border_cell_session,
    grid_cell_session,
    linear_track_session,
    open_field_session,
    tmaze_alternation_session,
)
from neurospatial.simulation.models import (
    BorderCellModel,
    BoundaryVectorCellModel,
    GridCellModel,
    HeadDirectionCellModel,
    NeuralModel,  # Protocol
    ObjectVectorCellModel,
    PlaceCellModel,
    SpatialViewCellModel,
)
from neurospatial.simulation.replay import (
    ReplayEventSimulation,
    simulate_replay_event,
)
from neurospatial.simulation.session import SimulationSession, simulate_session

# Spike generation
from neurospatial.simulation.spikes import (
    add_modulation,
    generate_poisson_spikes,
    generate_population_spikes,
)
from neurospatial.simulation.trajectory import (
    simulate_trajectory_laps,
    simulate_trajectory_ou,
    simulate_trajectory_sinusoidal,
)
from neurospatial.simulation.validation import (
    SimulationValidationResult,
    plot_session_summary,
    validate_simulation,
)

__all__ = [  # noqa: RUF022 - organized by category
    # Submodule
    "mazes",
    # Models
    "BorderCellModel",
    "BoundaryVectorCellModel",
    "GridCellModel",
    "HeadDirectionCellModel",
    "NeuralModel",
    "ObjectVectorCellModel",
    "PlaceCellModel",
    "ReplayEventSimulation",
    "SimulationSession",
    "SimulationValidationResult",
    "SpatialViewCellModel",
    # Spike generation
    "add_modulation",
    "generate_poisson_spikes",
    "generate_population_spikes",
    # Example sessions
    "border_cell_session",
    "grid_cell_session",
    "linear_track_session",
    "open_field_session",
    "tmaze_alternation_session",
    # Session API
    "plot_session_summary",
    "simulate_session",
    "validate_simulation",
    # Trajectory
    "simulate_trajectory_laps",
    "simulate_trajectory_ou",
    "simulate_trajectory_sinusoidal",
    "simulate_replay_event",
]
