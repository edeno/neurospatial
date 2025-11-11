"""Pre-configured example simulations for testing and demonstration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from neurospatial.simulation.session import SimulationSession


def open_field_session(
    duration: float = 180.0,
    arena_size: float = 100.0,
    bin_size: float = 2.0,
    n_place_cells: int = 50,
    seed: int | None = None,
) -> SimulationSession:
    """Generate standard open field arena simulation.

    Creates a square arena environment with place cells and Ornstein-Uhlenbeck
    random walk trajectory. This is a convenience function providing sensible
    defaults for typical neuroscience experiments.

    Parameters
    ----------
    duration : float, optional
        Session duration in seconds (default: 180.0).
    arena_size : float, optional
        Square arena side length in cm (default: 100.0).
    bin_size : float, optional
        Spatial bin size in cm (default: 2.0).
    n_place_cells : int, optional
        Number of place cells (default: 50).
    seed : int | None, optional
        Random seed for reproducibility (default: None).

    Returns
    -------
    session : SimulationSession
        Complete simulation session containing:
        - env: Square arena Environment with units="cm"
        - positions: Trajectory from OU random walk
        - times: Time points matching trajectory
        - spike_trains: Poisson spikes for each place cell
        - models: PlaceCellModel instances
        - ground_truth: True parameters for each cell
        - metadata: Session configuration

    Raises
    ------
    ValueError
        If duration, arena_size, bin_size, or n_place_cells are non-positive.
    ValueError
        If bin_size >= arena_size (would create too few bins).

    Examples
    --------
    Quick start for testing:

    >>> from neurospatial.simulation import open_field_session
    >>> session = open_field_session(duration=60.0, n_place_cells=20)
    >>> env = session.env
    >>> spike_trains = session.spike_trains
    >>> len(spike_trains)
    20

    Validate neurospatial's place field detection:

    >>> from neurospatial.simulation import open_field_session, validate_simulation
    >>> session = open_field_session(duration=120.0, n_place_cells=30, seed=42)
    >>> report = validate_simulation(session)
    >>> print(f"Validation: {report['passed']}")
    Validation: True

    Visualize session summary:

    >>> from neurospatial.simulation import open_field_session, plot_session_summary
    >>> import matplotlib.pyplot as plt
    >>> session = open_field_session(duration=60.0, n_place_cells=10, seed=42)
    >>> fig, axes = plot_session_summary(session)
    >>> plt.show()  # doctest: +SKIP

    See Also
    --------
    simulate_session : Low-level session simulation with full control
    validate_simulation : Validate detected vs ground truth fields
    plot_session_summary : Visualize session summary

    Notes
    -----
    **Default Parameters**:

    The defaults are chosen to match typical rodent neuroscience experiments:

    - Duration: 180 seconds (3 minutes) - sufficient for spatial coverage
    - Arena size: 100 cm × 100 cm - standard open field chamber
    - Bin size: 2 cm - balances spatial resolution vs smoothing
    - Number of cells: 50 - realistic hippocampal CA1 recording

    **Trajectory**:

    Uses Ornstein-Uhlenbeck (OU) process for trajectory generation, which
    produces smooth, realistic random walk behavior with tunable statistics
    via speed_mean and coherence_time parameters.

    **Coverage**:

    Place field centers are uniformly distributed across the arena using
    coverage='uniform' to ensure representative spatial sampling. This is
    ideal for testing place field detection algorithms.
    """
    # Import here to avoid circular dependencies
    from neurospatial import Environment
    from neurospatial.simulation.session import simulate_session

    # Validate parameters
    if duration <= 0:
        msg = f"duration must be positive, got {duration}"
        raise ValueError(msg)
    if arena_size <= 0:
        msg = f"arena_size must be positive, got {arena_size}"
        raise ValueError(msg)
    if bin_size <= 0:
        msg = f"bin_size must be positive, got {bin_size}"
        raise ValueError(msg)
    if bin_size >= arena_size:
        msg = f"bin_size ({bin_size}) must be smaller than arena_size ({arena_size})"
        raise ValueError(msg)
    if n_place_cells <= 0:
        msg = f"n_place_cells must be positive, got {n_place_cells}"
        raise ValueError(msg)

    # Create square arena environment
    # Generate 2D data that spans the arena with appropriate resolution
    # Use n_points based on bin_size to ensure proper discretization
    n_points_per_dim = max(20, int(arena_size / bin_size) + 1)
    x = np.linspace(0, arena_size, n_points_per_dim)
    y = np.linspace(0, arena_size, n_points_per_dim)
    xx, yy = np.meshgrid(x, y)
    arena_data = np.column_stack([xx.ravel(), yy.ravel()])

    # Create environment from data
    env = Environment.from_samples(arena_data, bin_size=bin_size)
    env.units = "cm"

    # Simulate session with place cells and OU trajectory
    session = simulate_session(
        env,
        duration=duration,
        n_cells=n_place_cells,
        cell_type="place",
        trajectory_method="ou",
        coverage="uniform",
        seed=seed,
        show_progress=False,  # Disable progress bar for convenience function
    )

    return session
