"""Simulation subpackage for generating synthetic spatial data.

This subpackage provides tools for:
- Trajectory simulation (random walks, laps, sinusoidal motion)
- Neural model implementations (place cells, boundary cells, grid cells)
- Spike train generation (Poisson process with refractory periods)
- High-level session simulation with validation helpers

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
>>> env.units = "cm"
>>>
>>> # Generate trajectory
>>> positions, times = simulate_trajectory_ou(env, duration=2.0, seed=42)
>>>
>>> # Create place cell at environment center with high rate
>>> center = env.bin_centers[len(env.bin_centers) // 2]
>>> place_cell = PlaceCellModel(env, center=center, width=20.0, max_rate=50.0, seed=42)
>>> rates = place_cell.firing_rate(positions, times)
>>> spikes = generate_poisson_spikes(rates, times, seed=42)
>>> len(spikes) > 0
True

Note: High-level session API (open_field_session, etc.) will be available in v0.2.0

Notes
-----
This subpackage follows the "flat is better than nested" principle. All public
functions and classes are importable directly from `neurospatial.simulation`
for maximum discoverability and IDE support.
"""

# Milestone 1 imports (core trajectory + place cells)
# Neural models
# Milestone 2 imports (boundary cells + extended features)
# Milestone 3 imports (grid cells)
# Pre-configured example simulations (Milestone 3)
from neurospatial.simulation.examples import (
    boundary_cell_session,
    grid_cell_session,
    linear_track_session,
    open_field_session,
    tmaze_alternation_session,
)
from neurospatial.simulation.models import (
    BoundaryCellModel,
    GridCellModel,
    NeuralModel,  # Protocol
    ObjectVectorCellModel,
    PlaceCellModel,
    SpatialViewCellModel,
)

# Milestone 3 imports (grid cells + session API)
# from neurospatial.simulation.models import GridCellModel  # Already imported above
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
    plot_session_summary,
    validate_simulation,
)

# from neurospatial.simulation.examples import (
#     linear_track_session,
#     tmaze_alternation_session,
#     boundary_cell_session,
#     grid_cell_session,
# )

__all__ = [
    "BoundaryCellModel",
    "GridCellModel",
    "NeuralModel",
    "ObjectVectorCellModel",
    "PlaceCellModel",
    "SimulationSession",
    "SpatialViewCellModel",
    "add_modulation",
    "boundary_cell_session",
    "generate_poisson_spikes",
    "generate_population_spikes",
    "grid_cell_session",
    "linear_track_session",
    "open_field_session",
    "plot_session_summary",
    "simulate_session",
    "simulate_trajectory_laps",
    "simulate_trajectory_ou",
    "simulate_trajectory_sinusoidal",
    "tmaze_alternation_session",
    "validate_simulation",
]
