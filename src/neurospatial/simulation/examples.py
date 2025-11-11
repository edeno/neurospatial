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


def linear_track_session(
    duration: float = 240.0,
    track_length: float = 200.0,
    bin_size: float = 1.0,
    n_place_cells: int = 40,
    n_laps: int = 20,
    seed: int | None = None,
) -> SimulationSession:
    """Generate linear track simulation with lap-based running.

    Creates a 1D linear track environment with place cells and lap-based
    trajectory (repeated back-and-forth traversals). This is a convenience
    function for simulating typical hippocampal recordings during linear
    track navigation experiments.

    Parameters
    ----------
    duration : float, optional
        Session duration in seconds (default: 240.0).
    track_length : float, optional
        Track length in cm (default: 200.0).
    bin_size : float, optional
        Spatial bin size in cm (default: 1.0).
    n_place_cells : int, optional
        Number of place cells (default: 40).
    n_laps : int, optional
        Number of back-and-forth laps (default: 20).
    seed : int | None, optional
        Random seed for reproducibility (default: None).

    Returns
    -------
    session : SimulationSession
        Complete simulation session containing:
        - env: 1D linear track Environment with units="cm"
        - positions: Trajectory from lap-based running
        - times: Time points matching trajectory
        - spike_trains: Poisson spikes for each place cell
        - models: PlaceCellModel instances
        - ground_truth: True parameters for each cell
        - metadata: Session configuration

    Raises
    ------
    ValueError
        If duration, track_length, bin_size, n_place_cells, or n_laps are non-positive.
    ValueError
        If bin_size >= track_length (would create too few bins).

    Examples
    --------
    Quick start for testing:

    >>> from neurospatial.simulation import linear_track_session
    >>> session = linear_track_session(duration=60.0, n_place_cells=20, n_laps=10)
    >>> env = session.env
    >>> env.n_dims
    1
    >>> len(session.spike_trains)
    20

    Validate place field detection on 1D track:

    >>> from neurospatial.simulation import linear_track_session, validate_simulation
    >>> session = linear_track_session(
    ...     duration=120.0, n_place_cells=30, n_laps=15, seed=42
    ... )
    >>> report = validate_simulation(session)
    >>> print(f"Validation: {report['passed']}")
    Validation: True

    Visualize session summary:

    >>> from neurospatial.simulation import linear_track_session, plot_session_summary
    >>> import matplotlib.pyplot as plt
    >>> session = linear_track_session(
    ...     duration=60.0, n_place_cells=10, n_laps=8, seed=42
    ... )
    >>> fig, axes = plot_session_summary(session)
    >>> plt.show()  # doctest: +SKIP

    See Also
    --------
    simulate_session : Low-level session simulation with full control
    validate_simulation : Validate detected vs ground truth fields
    plot_session_summary : Visualize session summary
    simulate_trajectory_laps : Low-level lap trajectory generation

    Notes
    -----
    **Default Parameters**:

    The defaults are chosen to match typical rodent hippocampal recordings:

    - Duration: 240 seconds (4 minutes) - sufficient for multiple laps
    - Track length: 200 cm - standard linear track length
    - Bin size: 1 cm - fine spatial resolution for 1D
    - Number of cells: 40 - typical hippocampal CA1 recording
    - Number of laps: 20 - provides good spatial coverage

    **Trajectory**:

    Uses lap-based trajectory generation with automatic path finding. The
    animal runs back and forth along the track, with speed variations and
    brief pauses at endpoints to simulate realistic behavior.

    **Coverage**:

    Place field centers are uniformly distributed along the track using
    coverage='uniform' to ensure representative spatial sampling across
    the entire track length.

    **1D Environment**:

    Creates a true 1D environment (env.n_dims == 1) suitable for linear
    track analysis. The environment spans [0, track_length] in cm.
    """
    # Import here to avoid circular dependencies
    from neurospatial import Environment
    from neurospatial.simulation.session import simulate_session

    # Validate parameters
    if duration <= 0:
        msg = f"duration must be positive, got {duration}"
        raise ValueError(msg)
    if track_length <= 0:
        msg = f"track_length must be positive, got {track_length}"
        raise ValueError(msg)
    if bin_size <= 0:
        msg = f"bin_size must be positive, got {bin_size}"
        raise ValueError(msg)
    if bin_size >= track_length:
        msg = (
            f"bin_size ({bin_size}) must be smaller than track_length ({track_length})"
        )
        raise ValueError(msg)
    if n_place_cells <= 0:
        msg = f"n_place_cells must be positive, got {n_place_cells}"
        raise ValueError(msg)
    if n_laps <= 0:
        msg = f"n_laps must be positive, got {n_laps}"
        raise ValueError(msg)

    # Create 1D linear track environment
    # Generate 1D data that spans the track with appropriate resolution
    n_points = max(20, int(track_length / bin_size) + 1)
    track_data = np.linspace(0, track_length, n_points).reshape(-1, 1)

    # Create environment from 1D data
    env = Environment.from_samples(track_data, bin_size=bin_size)
    env.units = "cm"

    # Simulate session with place cells and lap trajectory
    session = simulate_session(
        env,
        duration=duration,
        n_cells=n_place_cells,
        cell_type="place",
        trajectory_method="laps",
        coverage="uniform",
        n_laps=n_laps,
        seed=seed,
        show_progress=False,  # Disable progress bar for convenience function
    )

    return session


def tmaze_alternation_session(
    duration: float = 300.0,
    n_trials: int = 20,
    n_place_cells: int = 60,
    seed: int | None = None,
) -> SimulationSession:
    """Generate T-maze spatial alternation task simulation.

    Creates a T-maze graph environment with place cells and lap-based
    trajectory that alternates between left and right arms. This is a
    convenience function for simulating typical spatial alternation tasks
    in hippocampal recordings.

    Parameters
    ----------
    duration : float, optional
        Session duration in seconds (default: 300.0).
    n_trials : int, optional
        Number of alternation trials (default: 20).
    n_place_cells : int, optional
        Number of place cells (default: 60).
    seed : int | None, optional
        Random seed for reproducibility (default: None).

    Returns
    -------
    session : SimulationSession
        Complete simulation session containing:
        - env: T-maze graph Environment with units="cm"
        - positions: Trajectory from lap-based running with alternation
        - times: Time points matching trajectory
        - spike_trains: Poisson spikes for each place cell
        - models: PlaceCellModel instances
        - ground_truth: True parameters for each cell
        - metadata: Session configuration including 'trial_choices'

    Raises
    ------
    ValueError
        If duration, n_trials, or n_place_cells are non-positive.

    Examples
    --------
    Quick start for testing:

    >>> from neurospatial.simulation import tmaze_alternation_session
    >>> session = tmaze_alternation_session(
    ...     duration=60.0, n_trials=10, n_place_cells=20
    ... )
    >>> env = session.env
    >>> len(session.spike_trains)
    20
    >>> "trial_choices" in session.metadata
    True

    Check trial alternation pattern:

    >>> from neurospatial.simulation import tmaze_alternation_session
    >>> session = tmaze_alternation_session(n_trials=10, seed=42)
    >>> trial_choices = session.metadata["trial_choices"]
    >>> len(trial_choices)
    10
    >>> set(trial_choices)
    {'left', 'right'}

    Visualize session summary:

    >>> from neurospatial.simulation import (
    ...     tmaze_alternation_session,
    ...     plot_session_summary,
    ... )
    >>> import matplotlib.pyplot as plt
    >>> session = tmaze_alternation_session(
    ...     duration=60.0, n_trials=10, n_place_cells=10, seed=42
    ... )
    >>> fig, axes = plot_session_summary(session)
    >>> plt.show()  # doctest: +SKIP

    See Also
    --------
    simulate_session : Low-level session simulation with full control
    validate_simulation : Validate detected vs ground truth fields
    plot_session_summary : Visualize session summary
    linear_track_session : Linear track with lap-based running

    Notes
    -----
    **Default Parameters**:

    The defaults are chosen to match typical rodent spatial alternation tasks:

    - Duration: 300 seconds (5 minutes) - sufficient for multiple trials
    - Number of trials: 20 - typical experimental session length
    - Number of cells: 60 - realistic hippocampal CA1 recording

    **T-maze Structure**:

    Creates a simplified T-maze graph with:
    - Central stem (100 cm)
    - Left arm (50 cm)
    - Right arm (50 cm)
    Total track length: approximately 150 cm

    **Trajectory**:

    Uses lap-based trajectory generation. The animal alternates between
    left and right arms according to a perfect alternation pattern,
    starting with a random first choice determined by the seed.

    **Trial Metadata**:

    The metadata['trial_choices'] field contains a list of strings
    indicating the choice on each trial: ['left', 'right', 'left', ...].
    This allows analysis of choice-dependent neural activity.

    **Coverage**:

    Place field centers are uniformly distributed across the T-maze
    environment to ensure representative spatial sampling across all
    regions (stem, left arm, right arm).
    """
    # Import here to avoid circular dependencies
    import networkx as nx

    from neurospatial import Environment
    from neurospatial.simulation.session import simulate_session

    # Validate parameters
    if duration <= 0:
        msg = f"duration must be positive, got {duration}"
        raise ValueError(msg)
    if n_trials <= 0:
        msg = f"n_trials must be positive, got {n_trials}"
        raise ValueError(msg)
    if n_place_cells <= 0:
        msg = f"n_place_cells must be positive, got {n_place_cells}"
        raise ValueError(msg)

    # Create T-maze graph structure
    # Nodes: stem_start (0,0), center (0,100), left_end (-50,150), right_end (50,150)
    tmaze_graph = nx.Graph()
    tmaze_graph.add_node("stem_start", pos=(0.0, 0.0))
    tmaze_graph.add_node("center", pos=(0.0, 100.0))
    tmaze_graph.add_node("left_end", pos=(-50.0, 150.0))
    tmaze_graph.add_node("right_end", pos=(50.0, 150.0))

    # Add edges with distance attribute
    # Stem edge
    pos1 = np.array(tmaze_graph.nodes["stem_start"]["pos"])
    pos2 = np.array(tmaze_graph.nodes["center"]["pos"])
    distance = np.linalg.norm(pos2 - pos1)
    tmaze_graph.add_edge("stem_start", "center", distance=distance)

    # Left arm edge
    pos1 = np.array(tmaze_graph.nodes["center"]["pos"])
    pos2 = np.array(tmaze_graph.nodes["left_end"]["pos"])
    distance = np.linalg.norm(pos2 - pos1)
    tmaze_graph.add_edge("center", "left_end", distance=distance)

    # Right arm edge
    pos1 = np.array(tmaze_graph.nodes["center"]["pos"])
    pos2 = np.array(tmaze_graph.nodes["right_end"]["pos"])
    distance = np.linalg.norm(pos2 - pos1)
    tmaze_graph.add_edge("center", "right_end", distance=distance)

    # Define edge order for linearization (stem -> left arm path as primary)
    edge_order = [("stem_start", "center"), ("center", "left_end")]
    edge_spacing = 0.0  # No spacing between connected edges

    # Create environment from graph
    env = Environment.from_graph(
        graph=tmaze_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
        bin_size=2.0,
    )
    env.units = "cm"

    # Generate trial choices (perfect alternation pattern)
    # Use seed to determine first choice, then alternate
    rng = np.random.default_rng(seed)
    first_choice = rng.choice(["left", "right"])

    trial_choices = []
    for i in range(n_trials):
        if i == 0:
            trial_choices.append(first_choice)
        else:
            # Alternate from previous choice
            prev = trial_choices[-1]
            trial_choices.append("right" if prev == "left" else "left")

    # Simulate session with place cells and lap trajectory
    session = simulate_session(
        env,
        duration=duration,
        n_cells=n_place_cells,
        cell_type="place",
        trajectory_method="laps",
        coverage="uniform",
        n_laps=n_trials,
        seed=seed,
        show_progress=False,  # Disable progress bar for convenience function
    )

    # Add trial_choices to metadata
    session.metadata["trial_choices"] = trial_choices

    return session


def boundary_cell_session(
    duration: float = 180.0,
    arena_shape: str = "square",
    arena_size: float = 100.0,
    bin_size: float = 2.0,
    n_boundary_cells: int = 30,
    n_place_cells: int = 20,
    seed: int | None = None,
) -> SimulationSession:
    """Generate session with boundary and place cells.

    Creates an arena environment with a mix of boundary cells (fire near
    walls/edges) and place cells (fire in specific locations). This is a
    convenience function for simulating recordings that include both
    boundary-responsive and place-responsive neurons.

    Parameters
    ----------
    duration : float, optional
        Session duration in seconds (default: 180.0).
    arena_shape : str, optional
        Arena shape - currently only "square" supported (default: "square").
    arena_size : float, optional
        Arena side length in cm for square arena (default: 100.0).
    bin_size : float, optional
        Spatial bin size in cm (default: 2.0).
    n_boundary_cells : int, optional
        Number of boundary cells (default: 30).
    n_place_cells : int, optional
        Number of place cells (default: 20).
    seed : int | None, optional
        Random seed for reproducibility (default: None).

    Returns
    -------
    session : SimulationSession
        Complete simulation session containing:
        - env: Arena Environment with units="cm"
        - positions: Trajectory from OU random walk
        - times: Time points matching trajectory
        - spike_trains: Poisson spikes for each cell (boundary + place)
        - models: BoundaryCellModel and PlaceCellModel instances
        - ground_truth: True parameters for each cell
        - metadata: Session configuration

    Raises
    ------
    ValueError
        If duration, n_boundary_cells, or n_place_cells are non-positive.
    ValueError
        If bin_size >= arena_size or bin_size <= 0.
    ValueError
        If arena_shape is not "square" (other shapes not yet supported).

    Examples
    --------
    Quick start for testing:

    >>> from neurospatial.simulation import boundary_cell_session
    >>> session = boundary_cell_session(
    ...     duration=60.0, n_boundary_cells=10, n_place_cells=10
    ... )
    >>> env = session.env
    >>> len(session.spike_trains)
    20

    Check cell type distribution:

    >>> from neurospatial.simulation import boundary_cell_session
    >>> from neurospatial.simulation.models import BoundaryCellModel, PlaceCellModel
    >>> session = boundary_cell_session(n_boundary_cells=30, n_place_cells=20, seed=42)
    >>> n_boundary = sum(isinstance(m, BoundaryCellModel) for m in session.models)
    >>> n_place = sum(isinstance(m, PlaceCellModel) for m in session.models)
    >>> print(f"Boundary cells: {n_boundary}, Place cells: {n_place}")
    Boundary cells: 30, Place cells: 20

    Visualize session summary:

    >>> from neurospatial.simulation import (
    ...     boundary_cell_session,
    ...     plot_session_summary,
    ... )
    >>> import matplotlib.pyplot as plt
    >>> session = boundary_cell_session(
    ...     duration=60.0, n_boundary_cells=5, n_place_cells=5, seed=42
    ... )
    >>> fig, axes = plot_session_summary(session)
    >>> plt.show()  # doctest: +SKIP

    See Also
    --------
    simulate_session : Low-level session simulation with full control
    validate_simulation : Validate detected vs ground truth fields
    plot_session_summary : Visualize session summary
    open_field_session : Open field with only place cells

    Notes
    -----
    **Default Parameters**:

    The defaults are chosen to match typical rodent recordings with mixed
    cell types:

    - Duration: 180 seconds (3 minutes) - sufficient for spatial coverage
    - Arena size: 100 cm × 100 cm - standard open field chamber
    - Bin size: 2 cm - balances spatial resolution vs smoothing
    - Boundary cells: 30 - realistic proportion (~60% of total)
    - Place cells: 20 - remaining proportion (~40% of total)

    **Mixed Cell Types**:

    This function creates a heterogeneous population with both boundary
    cells (which fire preferentially near arena walls) and place cells
    (which fire in localized spatial regions). Boundary cells are created
    first, followed by place cells. All cells use uniform spatial coverage
    for their field centers/boundary preferences.

    **Arena Shapes**:

    Currently only square arenas are supported. The arena is created using
    Environment.from_samples() with a regular grid of points spanning the
    arena dimensions.

    **Trajectory**:

    Uses Ornstein-Uhlenbeck (OU) process for trajectory generation, which
    produces smooth, realistic random walk behavior. The trajectory covers
    both central regions (place cell firing) and edges (boundary cell
    firing).
    """
    # Import here to avoid circular dependencies
    from neurospatial import Environment
    from neurospatial.simulation.models import BoundaryCellModel, PlaceCellModel
    from neurospatial.simulation.session import SimulationSession
    from neurospatial.simulation.spikes import generate_population_spikes
    from neurospatial.simulation.trajectory import simulate_trajectory_ou

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
    if n_boundary_cells <= 0:
        msg = f"n_boundary_cells must be positive, got {n_boundary_cells}"
        raise ValueError(msg)
    if n_place_cells <= 0:
        msg = f"n_place_cells must be positive, got {n_place_cells}"
        raise ValueError(msg)
    if arena_shape != "square":
        msg = f"Only 'square' arena_shape is currently supported, got {arena_shape!r}"
        raise ValueError(msg)

    # Create square arena environment (same as open_field_session)
    n_points_per_dim = max(20, int(arena_size / bin_size) + 1)
    x = np.linspace(0, arena_size, n_points_per_dim)
    y = np.linspace(0, arena_size, n_points_per_dim)
    xx, yy = np.meshgrid(x, y)
    arena_data = np.column_stack([xx.ravel(), yy.ravel()])

    # Create environment from data
    env = Environment.from_samples(arena_data, bin_size=bin_size)
    env.units = "cm"

    # Generate trajectory using OU process
    positions, times = simulate_trajectory_ou(env, duration=duration, seed=seed)

    # Create models: boundary cells first, then place cells
    # Use uniform coverage for field centers
    total_cells = n_boundary_cells + n_place_cells
    step = max(1, env.n_bins // total_cells)
    field_centers = env.bin_centers[::step][:total_cells]

    models: list = []
    ground_truth = {}

    # Create boundary cells (don't use field_centers for boundary cells)
    for i in range(n_boundary_cells):
        model = BoundaryCellModel(env)
        models.append(model)
        # Store ground truth for boundary cell
        ground_truth[f"cell_{i}"] = {
            "cell_type": "boundary",
            "preferred_distance": float(model.preferred_distance),
            "distance_tolerance": float(model.distance_tolerance),
            "preferred_direction": (
                float(model.preferred_direction)
                if model.preferred_direction is not None
                else None
            ),
            "direction_tolerance": float(model.direction_tolerance),
            "max_rate": float(model.max_rate),
            "baseline_rate": float(model.baseline_rate),
        }

    # Create place cells using field centers
    for i in range(n_place_cells):
        cell_idx = n_boundary_cells + i
        center = (
            field_centers[cell_idx]
            if cell_idx < len(field_centers)
            else field_centers[i % len(field_centers)]
        )
        place_model = PlaceCellModel(env, center=center)
        models.append(place_model)
        # Store ground truth for place cell
        ground_truth[f"cell_{cell_idx}"] = {
            "cell_type": "place",
            "center": center.tolist(),
            "max_rate": float(place_model.max_rate),
            "baseline_rate": float(place_model.baseline_rate),
        }

    # Generate spikes for all cells
    spike_trains = generate_population_spikes(
        models, positions, times, seed=seed, show_progress=False
    )

    # Create metadata
    metadata = {
        "duration": duration,
        "arena_shape": arena_shape,
        "arena_size": arena_size,
        "bin_size": bin_size,
        "n_boundary_cells": n_boundary_cells,
        "n_place_cells": n_place_cells,
        "cell_type": "mixed_boundary_place",
        "trajectory_method": "ou",
        "coverage": "uniform",
        "seed": seed,
    }

    # Create and return session
    session = SimulationSession(
        env=env,
        positions=positions,
        times=times,
        spike_trains=spike_trains,
        models=models,
        ground_truth=ground_truth,
        metadata=metadata,
    )

    return session
