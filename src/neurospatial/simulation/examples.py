"""Pre-configured example simulations for testing and demonstration."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from neurospatial.simulation.models.place_cells import (
    TRACK_PLACE_FIELD_PEAK_HZ,
    TRACK_PLACE_FIELD_WIDTH_CM,
)
from neurospatial.simulation.session import _require_positive

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.simulation.models import NeuralModel
    from neurospatial.simulation.session import SimulationSession


def _square_arena(arena_size: float, bin_size: float) -> Environment:
    """A square arena of side ``arena_size`` cm, binned from a grid of samples."""
    from neurospatial.environment import Environment

    n_points_per_dim = max(20, int(arena_size / bin_size) + 1)
    x = np.linspace(0, arena_size, n_points_per_dim)
    xx, yy = np.meshgrid(x, x)
    samples = np.column_stack([xx.ravel(), yy.ravel()])
    return Environment.from_samples(samples, bin_size=bin_size, units="cm")


def open_field_session(
    *,
    duration: float = 180.0,
    arena_size: float = 100.0,
    bin_size: float = 2.0,
    n_place_cells: int = 50,
    seed: int | None = None,
    **kwargs: Any,
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
    **kwargs : Any
        Additional parameters passed to PlaceCellModel. Common options:
        - max_rate : float - Peak firing rate in Hz (default: 12.4, see
          ``PlaceCellModel``)
        - width : float - Place field width in cm (default: 11.1, see
          ``PlaceCellModel``)
        - baseline_rate : float - Baseline firing rate in Hz (default: 0.01)

    Returns
    -------
    session : SimulationSession
        Complete simulation session containing:
        - env: Square arena Environment with units="cm"
        - positions: Trajectory from OU random walk
        - times: Time points matching trajectory
        - spike_times: Poisson spikes for each place cell
        - models: PlaceCellModel instances
        - ground_truth: True parameters for each cell
        - metadata: Session configuration

    Raises
    ------
    ValueError
        If duration, arena_size or bin_size is not finite and positive, or
        n_place_cells is not a positive integer.
    ValueError
        If bin_size >= arena_size (would create too few bins).

    Examples
    --------
    Quick start for testing:

    >>> from neurospatial.simulation import open_field_session
    >>> session = open_field_session(duration=5.0, n_place_cells=3)
    >>> env = session.env
    >>> spike_times = session.spike_times
    >>> len(spike_times)
    3

    Validate neurospatial's place field detection. Three minutes cross every
    field often enough to recover it (5 cm bins keep the example fast):

    >>> from neurospatial.simulation import open_field_session, validate_simulation
    >>> session = open_field_session(
    ...     duration=180.0, bin_size=5.0, n_place_cells=3, seed=0
    ... )
    >>> report = validate_simulation(session)
    >>> report.unit_status
    ('passed', 'passed', 'passed')

    Visualize session summary:

    >>> from neurospatial.simulation import open_field_session, plot_session_summary
    >>> import matplotlib.pyplot as plt
    >>> session = open_field_session(
    ...     duration=5.0, n_place_cells=3, seed=42
    ... )  # doctest: +SKIP
    >>> fig, axes = plot_session_summary(session)  # doctest: +SKIP
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

    Each place cell has its own field center, spread over the arena with
    coverage='uniform' and kept 20% of the side from every wall, where the
    animal samples well. This is ideal for testing place field detection
    algorithms.
    """
    # Import here to avoid circular dependencies
    from neurospatial.simulation.session import simulate_session

    _require_positive("duration", duration, units=" (seconds)")
    _require_positive("arena_size", arena_size, units=" (cm)")
    _require_positive("bin_size", bin_size, units=" (cm)")
    if bin_size >= arena_size:
        msg = (
            f"bin_size ({bin_size} cm) must be smaller than "
            f"arena_size ({arena_size} cm)"
        )
        raise ValueError(msg)
    _require_positive("n_place_cells", n_place_cells, count=True)

    env = _square_arena(arena_size, bin_size)

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
        **kwargs,  # Pass through model parameters (e.g., max_rate, width)
    )

    return session


def linear_track_session(
    *,
    duration: float = 240.0,
    track_length: float = 200.0,
    bin_size: float = 1.0,
    n_place_cells: int = 40,
    n_laps: int = 20,
    seed: int | None = None,
    **kwargs: Any,
) -> SimulationSession:
    """Generate linear track simulation with lap-based running.

    Creates a 1D linear track environment with place cells and lap-based
    trajectory (repeated back-and-forth traversals). This is a convenience
    function for simulating typical hippocampal recordings during linear
    track navigation experiments.

    Parameters
    ----------
    duration : float, optional
        Session duration in seconds (default: 240.0). Samples at 500 Hz cover
        ``[0, duration)``; the final sample is within 0.002 s of ``duration``.
    track_length : float, optional
        Track length in cm (default: 200.0).
    bin_size : float, optional
        Spatial bin size in cm (default: 1.0).
    n_place_cells : int, optional
        Number of place cells (default: 40).
    n_laps : int, optional
        Number of one-way traversals (default: 20), alternating outbound and
        inbound starting outbound. Two traversals form one out-and-back cycle.
    seed : int | None, optional
        Random seed for reproducibility (default: None).
    **kwargs : Any
        Place cell parameters passed to ``PlaceCellModel``: ``width`` (cm,
        default 17.5) and ``max_rate`` (Hz, default 14.5), the CA1
        linear-track values of Mizuseki et al. (2012) (see
        ``PlaceCellModel``), and ``baseline_rate`` and ``metric`` with
        ``PlaceCellModel``'s defaults.

    Returns
    -------
    session : SimulationSession
        Complete simulation session containing:
        - env: 1D linear track Environment with units="cm"
        - positions: Trajectory from lap-based running
        - times: Time points matching trajectory
        - spike_times: Poisson spikes for each place cell
        - models: PlaceCellModel instances
        - ground_truth: True parameters for each cell
        - metadata: Session configuration

    Raises
    ------
    ValueError
        If duration, track_length or bin_size is not finite and positive, or
        n_place_cells or n_laps is not a positive integer.
    ValueError
        If bin_size >= track_length (would create too few bins).
    ValueError
        If duration is non-finite or too short for the requested traversals:
        at least two samples per traversal and 0.5 s between traversals.

    Examples
    --------
    Quick start for testing:

    >>> from neurospatial.simulation import linear_track_session
    >>> session = linear_track_session(duration=5.0, n_place_cells=3, n_laps=2)
    >>> env = session.env
    >>> env.n_dims
    1
    >>> len(session.spike_times)
    3

    Validate place field detection on a 1D track. Estimating each field's
    center requires repeated visits and enough spikes. This example records
    one minute with six traversals, approximately 20 cm/s on the 200 cm track:

    >>> from neurospatial.simulation import linear_track_session, validate_simulation
    >>> session = linear_track_session(
    ...     duration=60.0, n_place_cells=3, n_laps=6, seed=42
    ... )
    >>> report = validate_simulation(session)
    >>> print(f"Validation: {report.passed}")
    Validation: True

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
    - Number of traversals: 20 - ten out-and-back cycles

    **Trajectory**:

    Uses lap-based trajectory generation with automatic path finding. The
    animal runs back and forth along the track, preserving all ``n_laps``
    traversals and fixed 0.5 s pauses between them. Remaining samples are
    distributed using seeded relative speed variations; absolute speeds are
    determined by ``duration`` and the traversal count, with no imposed speed
    cap. Infeasible sample budgets raise an error rather than dropping laps.
    Metadata records the requested duration and the sampling convention.

    **Coverage**:

    Place field centers are uniformly distributed along the track using
    coverage='uniform' to ensure representative spatial sampling across
    the entire track length.

    **1D Environment**:

    Creates a true 1D environment (env.n_dims == 1) suitable for linear
    track analysis. The environment spans [0, track_length] in cm.
    """
    # Import here to avoid circular dependencies
    from neurospatial.environment import Environment
    from neurospatial.simulation.session import simulate_session

    _require_positive("duration", duration)
    _require_positive("track_length", track_length)
    _require_positive("bin_size", bin_size)
    if bin_size >= track_length:
        msg = (
            f"bin_size ({bin_size}) must be smaller than track_length ({track_length})"
        )
        raise ValueError(msg)
    _require_positive("n_place_cells", n_place_cells, count=True)
    _require_positive("n_laps", n_laps, count=True)

    # Create 1D linear track environment
    # Generate 1D data that spans the track with appropriate resolution
    n_points = max(20, int(track_length / bin_size) + 1)
    track_data = np.linspace(0, track_length, n_points).reshape(-1, 1)

    # Create environment from 1D data
    env = Environment.from_samples(track_data, bin_size=bin_size)
    env = env.with_metadata(units="cm")

    # Simulate session with place cells and lap trajectory, with linear-track
    # place fields unless the caller chose others.
    kwargs.setdefault("width", TRACK_PLACE_FIELD_WIDTH_CM)
    kwargs.setdefault("max_rate", TRACK_PLACE_FIELD_PEAK_HZ)
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
        **kwargs,  # Pass through model parameters (e.g., max_rate, width)
    )

    return session


def tmaze_alternation_session(
    *,
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
        Session duration in seconds (default: 300.0). Samples at 500 Hz cover
        ``[0, duration)``; the final sample is within 0.002 s of ``duration``.
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
        - spike_times: Poisson spikes for each place cell
        - models: PlaceCellModel instances
        - ground_truth: True parameters for each cell
        - metadata: Session configuration including 'trial_choices'

    Raises
    ------
    ValueError
        If duration is not finite and positive, or n_trials or n_place_cells
        is not a positive integer.
    ValueError
        If duration is non-finite or too short for two samples per trial
        and fixed 0.5 s pauses between the ``n_trials`` trials.

    Examples
    --------
    Quick start for testing:

    >>> from neurospatial.simulation import tmaze_alternation_session
    >>> session = tmaze_alternation_session(duration=5.0, n_trials=3, n_place_cells=3)
    >>> env = session.env
    >>> len(session.spike_times)
    3
    >>> "trial_choices" in session.metadata
    True

    Check trial alternation pattern:

    >>> from neurospatial.simulation import tmaze_alternation_session
    >>> session = tmaze_alternation_session(duration=5.0, n_trials=3, seed=42)
    >>> trial_choices = session.metadata["trial_choices"]
    >>> len(trial_choices)
    3
    >>> sorted(set(str(c) for c in trial_choices))
    ['left', 'right']

    Visualize session summary:

    >>> from neurospatial.simulation import (  # doctest: +SKIP
    ...     tmaze_alternation_session,
    ...     plot_session_summary,
    ... )
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> session = tmaze_alternation_session(  # doctest: +SKIP
    ...     duration=5.0, n_trials=3, n_place_cells=3, seed=42
    ... )
    >>> fig, axes = plot_session_summary(session)  # doctest: +SKIP
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
    - Place fields: 17.5 cm standard deviation and 14.5 Hz peak, the CA1
      linear-track values of Mizuseki et al. (2012); see ``PlaceCellModel``

    **T-maze Structure**:

    The track of ``make_t_maze()``: a 100 cm stem from ``start`` to the
    ``junction`` and two perpendicular 50 cm arms ending at ``well_left`` and
    ``well_right``, with 10 cm corridors and 2 cm bins (200 cm of track).
    These sizes are estimates read from a published figure, not a measured
    apparatus (see ``TMazeDims``). ``session.env`` carries those regions.

    **Trajectory**:

    Each trial runs from the end of the stem to the end of the chosen arm and
    back. Choices alternate perfectly, starting with a random first choice
    determined by the seed, so trial ``i`` visits the arm
    ``trial_choices[i]``.

    Session timing preserves ``n_trials`` round trips and fixed 0.5 s pauses
    between them. Absolute speeds follow from the requested duration;
    seeded speed draws determine relative trial durations. Metadata records
    duration and the sampling convention; infeasible sample budgets raise.

    **Trial Metadata**:

    The metadata["trial_choices"] field contains a tuple of strings
    indicating the choice on each trial: ("left", "right", "left", ...).
    This allows analysis of choice-dependent neural activity.

    **Coverage**:

    Place field centers are evenly spaced along the track, so the stem and
    both arms carry fields.
    """
    # Import here to avoid circular dependencies
    import networkx as nx

    from neurospatial.simulation.mazes import TMazeDims, make_t_maze
    from neurospatial.simulation.session import simulate_session

    _require_positive("duration", duration)
    _require_positive("n_trials", n_trials, count=True)
    _require_positive("n_place_cells", n_place_cells, count=True)

    # The default T-maze track (sizes estimated from a figure; see TMazeDims):
    # a 100 cm stem and two 50 cm arms.
    dims = TMazeDims()
    env = make_t_maze(dims, bin_size=2.0).env_track
    assert env is not None
    half_stem = dims.stem_length / 2.0
    stem_end, left_end, right_end = env.bin_at(
        np.array(
            [
                [0.0, -half_stem],
                [-dims.arm_length, half_stem],
                [dims.arm_length, half_stem],
            ]
        )
    )

    # Perfect alternation from a seeded first choice.
    rng = np.random.default_rng(seed)
    first_choice = str(rng.choice(["left", "right"]))
    second_choice = "right" if first_choice == "left" else "left"
    trial_choices = [
        first_choice if i % 2 == 0 else second_choice for i in range(n_trials)
    ]

    # A trial runs from the stem's end to the chosen arm's end and back. The
    # lap engine alternates its two paths, so trial i follows trial_choices[i].
    arm_ends = {"left": int(left_end), "right": int(right_end)}

    def round_trip(choice: str) -> list[int]:
        run: list[int] = nx.shortest_path(
            env.connectivity, int(stem_end), arm_ends[choice]
        )
        return run + run[-2::-1]

    session = simulate_session(
        env,
        duration=duration,
        n_cells=n_place_cells,
        cell_type="place",
        trajectory_method="laps",
        coverage="uniform",
        n_laps=n_trials,
        outbound_path=round_trip(first_choice),
        inbound_path=round_trip(second_choice),
        width=TRACK_PLACE_FIELD_WIDTH_CM,  # linear-track CA1 fields
        max_rate=TRACK_PLACE_FIELD_PEAK_HZ,
        seed=seed,
        show_progress=False,  # Disable progress bar for convenience function
    )

    return replace(
        session, metadata={**session.metadata, "trial_choices": trial_choices}
    )


def border_cell_session(
    *,
    duration: float = 180.0,
    arena_shape: str = "square",
    arena_size: float = 100.0,
    bin_size: float = 2.0,
    n_border_cells: int = 30,
    n_place_cells: int = 20,
    distance_tolerance: float | None = None,
    seed: int | None = None,
) -> SimulationSession:
    """Generate session with border and place cells.

    Creates a walled square arena with a mix of border cells (each fires
    along the walls in its own preferred direction) and place cells (fire in
    specific locations). This is a convenience function for simulating
    recordings that include both wall-responsive and place-responsive
    neurons.

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
    n_border_cells : int, optional
        Number of border cells (default: 30).
    n_place_cells : int, optional
        Number of place cells (default: 20).
    distance_tolerance : float | None, optional
        Standard deviation in cm of every border cell's distance tuning
        (``BoundaryVectorCellModel(distance_tolerance=...)``). None (default)
        uses the model's rule, ``preferred_distance / 12 + 8`` cm.
    seed : int | None, optional
        Random seed for reproducibility (default: None). The trajectory, the
        cells' tuning and the spikes draw from independent streams spawned
        from it.

    Returns
    -------
    session : SimulationSession
        Complete simulation session containing:
        - env: Walled square arena (``Environment.from_polygon``), units="cm"
        - positions: Trajectory from OU random walk
        - times: Time points matching trajectory
        - spike_times: Poisson spikes for each cell (border, then place)
        - models: BoundaryVectorCellModel and PlaceCellModel instances
        - ground_truth: Each model's ``ground_truth``, by unit label
        - metadata: Session configuration

    Raises
    ------
    ValueError
        If duration is not finite and positive, n_border_cells or
        n_place_cells is not a positive integer, or distance_tolerance is not
        positive.
    ValueError
        If bin_size >= arena_size or bin_size <= 0.
    ValueError
        If arena_shape is not "square" (other shapes not yet supported).

    Examples
    --------
    Quick start for testing:

    >>> from neurospatial.simulation import border_cell_session
    >>> session = border_cell_session(duration=5.0, n_border_cells=3, n_place_cells=2)
    >>> len(session.spike_times)
    5

    Each unit's ground truth says what it is:

    >>> session = border_cell_session(
    ...     duration=5.0, n_border_cells=3, n_place_cells=2, seed=42
    ... )
    >>> [truth["cell_type"] for truth in session.ground_truth.values()]
    ['boundary_vector', 'boundary_vector', 'boundary_vector', 'place', 'place']

    Visualize session summary:

    >>> from neurospatial.simulation import (  # doctest: +SKIP
    ...     border_cell_session,
    ...     plot_session_summary,
    ... )
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> session = border_cell_session(  # doctest: +SKIP
    ...     duration=60.0, n_border_cells=3, n_place_cells=3, seed=42
    ... )
    >>> fig, axes = plot_session_summary(session)  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP

    See Also
    --------
    simulate_session : Low-level session simulation with full control
    validate_simulation : Validate detected vs ground truth fields
    plot_session_summary : Visualize session summary
    open_field_session : Open field with only place cells
    BoundaryVectorCellModel : The border cells' model

    Notes
    -----
    **Default Parameters**:

    - Duration: 180 seconds (3 minutes) - sufficient for spatial coverage
    - Arena size: 100 cm × 100 cm - standard open field chamber
    - Bin size: 2 cm - balances spatial resolution vs smoothing
    - Border cells: 30
    - Place cells: 20

    **Border cells** are boundary vector cells with short preferred
    distances, as in ``simulate_session(cell_type='border')``: each fires
    along the walls that lie in its own preferred allocentric direction,
    drawn uniformly from [-pi, pi), at its own preferred distance, drawn
    uniformly from [0, 8] cm (field centers 6.5-10.9 cm from the wall, as
    in mouse border cells; see ``simulate_session`` Notes). A direction
    toward a corner fires along both walls of that corner.

    **Place cells** have the default ``PlaceCellModel`` fields, centered on
    the interior layout of ``simulate_session(coverage='uniform')``. Border
    cells are units ``0 .. n_border_cells - 1``, place cells follow.

    **Trajectory**:

    Uses Ornstein-Uhlenbeck (OU) process for trajectory generation, which
    produces smooth, realistic random walk behavior. The trajectory covers
    both central regions (place cell firing) and edges (border cell
    firing).
    """
    # Import here to avoid circular dependencies
    from shapely.geometry import box

    from neurospatial.environment import Environment
    from neurospatial.simulation._seeds import _stream_seeds
    from neurospatial.simulation.models import PlaceCellModel
    from neurospatial.simulation.session import (
        _border_cells,
        _field_centers,
        _simulated_session,
    )
    from neurospatial.simulation.trajectory import simulate_trajectory_ou

    _require_positive("duration", duration)
    _require_positive("arena_size", arena_size)
    _require_positive("bin_size", bin_size)
    if bin_size >= arena_size:
        msg = f"bin_size ({bin_size}) must be smaller than arena_size ({arena_size})"
        raise ValueError(msg)
    _require_positive("n_border_cells", n_border_cells, count=True)
    _require_positive("n_place_cells", n_place_cells, count=True)
    if arena_shape != "square":
        msg = f"Only 'square' arena_shape is currently supported, got {arena_shape!r}"
        raise ValueError(msg)

    # A walled square arena: its outline is a physical wall.
    env = Environment.from_polygon(
        box(0.0, 0.0, arena_size, arena_size), bin_size=bin_size, units="cm"
    )

    # Independent streams for the trajectory, the cells' tuning and the spikes.
    trajectory_seed, field_seed, spike_seed = _stream_seeds(seed, 3)
    positions, times = simulate_trajectory_ou(
        env, duration=duration, speed_units="cm", seed=trajectory_seed
    )

    rng = np.random.default_rng(field_seed)
    border_params = (
        {} if distance_tolerance is None else {"distance_tolerance": distance_tolerance}
    )
    models = _border_cells(env, n_border_cells, rng, border_params)
    models.extend(
        PlaceCellModel(env, center=center)
        for center in _field_centers(env, n_place_cells, "uniform", rng)
    )

    metadata = {
        "duration": duration,
        "arena_shape": arena_shape,
        "arena_size": arena_size,
        "bin_size": bin_size,
        "distance_tolerance": distance_tolerance,
        "n_border_cells": n_border_cells,
        "n_place_cells": n_place_cells,
        "cell_type": "mixed_border_place",
        "trajectory_method": "ou",
        "coverage": "uniform",
        "seed": seed,
    }
    return _simulated_session(
        env,
        positions,
        times,
        models,
        spike_seed=spike_seed,
        show_progress=False,
        metadata=metadata,
    )


def grid_cell_session(
    *,
    duration: float = 300.0,
    arena_size: float = 150.0,
    grid_spacing: float = 41.0,
    n_grid_cells: int = 40,
    seed: int | None = None,
) -> SimulationSession:
    """Generate session with grid cells (2D only).

    Creates a square arena environment with grid cells exhibiting hexagonal
    firing patterns. Grid cells are created with random phases distributed
    across the arena to create diverse spatial representations. This is a
    convenience function for simulating grid cell recordings in open field
    environments.

    Parameters
    ----------
    duration : float, optional
        Session duration in seconds (default: 300.0).
    arena_size : float, optional
        Square arena side length in cm (default: 150.0).
    grid_spacing : float, optional
        Grid spacing (distance between firing fields) in cm (default: 41.0,
        dorsal-most rat grid cells; see ``GridCellModel``).
    n_grid_cells : int, optional
        Number of grid cells (default: 40).
    seed : int | None, optional
        Random seed for reproducibility (default: None). The trajectory, the
        cells' phases and orientations, and the spikes draw from independent
        streams spawned from it.

    Returns
    -------
    session : SimulationSession
        Complete simulation session containing:
        - env: Square arena Environment with units="cm"
        - positions: Trajectory from OU random walk
        - times: Time points matching trajectory
        - spike_times: Poisson spikes for each grid cell
        - models: GridCellModel instances
        - ground_truth: Each model's ``ground_truth``, by unit label
        - metadata: Session configuration

    Raises
    ------
    ValueError
        If duration, arena_size or grid_spacing is not finite and positive,
        or n_grid_cells is not a positive integer.
    ValueError
        If grid_spacing >= arena_size (would create too few fields).

    Examples
    --------
    Quick start for testing:

    >>> from neurospatial.simulation import grid_cell_session
    >>> session = grid_cell_session(duration=5.0, n_grid_cells=3)
    >>> env = session.env
    >>> len(session.spike_times)
    3

    Use custom grid spacing:

    >>> from neurospatial.simulation import grid_cell_session
    >>> session = grid_cell_session(
    ...     duration=5.0, grid_spacing=40.0, n_grid_cells=3, seed=42
    ... )
    >>> session.metadata["grid_spacing"]
    40.0

    Visualize session summary:

    >>> from neurospatial.simulation import grid_cell_session, plot_session_summary
    >>> import matplotlib.pyplot as plt
    >>> session = grid_cell_session(
    ...     duration=5.0, n_grid_cells=3, seed=42
    ... )  # doctest: +SKIP
    >>> fig, axes = plot_session_summary(session)  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP

    See Also
    --------
    simulate_session : Low-level session simulation with full control
    validate_simulation : Validate detected vs ground truth fields
    plot_session_summary : Visualize session summary
    open_field_session : Open field with place cells

    Notes
    -----
    **Default Parameters**:

    The defaults are chosen to match typical grid cell recordings:

    - Duration: 300 seconds (5 minutes) - sufficient for spatial coverage
    - Arena size: 150 cm × 150 cm - larger arena to show grid structure
    - Grid spacing: 41 cm - dorsal-most rat grid cells (Hafting et al. 2005)
    - Number of cells: 40 - realistic grid cell recording

    **Grid Cell Model**:

    Each grid cell is created with:
    - Random phase offsets (uniformly distributed)
    - Random orientation angle
    - Shared grid spacing (same module)
    - Hexagonal firing pattern in 2D

    **Trajectory**:

    Uses Ornstein-Uhlenbeck (OU) process for trajectory generation, which
    produces smooth, realistic random walk behavior that extensively covers
    the arena to reveal hexagonal grid structure.

    **Phase Distribution**:

    Grid cell phases are uniformly distributed across the arena by sampling
    from environment bin centers. This ensures diverse phase offsets and
    creates a realistic population of grid cells from the same module.
    """
    # Import here to avoid circular dependencies
    from neurospatial.simulation._seeds import _stream_seeds
    from neurospatial.simulation.models import GridCellModel
    from neurospatial.simulation.session import _simulated_session
    from neurospatial.simulation.trajectory import simulate_trajectory_ou

    _require_positive("duration", duration)
    _require_positive("arena_size", arena_size)
    _require_positive("grid_spacing", grid_spacing)
    if grid_spacing >= arena_size:
        msg = (
            f"grid_spacing ({grid_spacing}) must be smaller than "
            f"arena_size ({arena_size})"
        )
        raise ValueError(msg)
    _require_positive("n_grid_cells", n_grid_cells, count=True)

    # Bins of a fifth of the grid spacing resolve the grid structure.
    bin_size = max(1.0, grid_spacing / 5.0)
    env = _square_arena(arena_size, bin_size)

    # Independent streams for the trajectory, the cells' phases and
    # orientations, and the spikes.
    trajectory_seed, field_seed, spike_seed = _stream_seeds(seed, 3)
    positions, times = simulate_trajectory_ou(
        env, duration=duration, speed_units="cm", seed=trajectory_seed
    )

    # Phases from bin centers spread over the arena, in a random order, and
    # random orientations.
    rng = np.random.default_rng(field_seed)
    step = max(1, env.n_bins // n_grid_cells)
    # Copy: shuffling a view would permute the environment's own bin centers.
    phase_centers = env.bin_centers[::step][:n_grid_cells].copy()
    rng.shuffle(phase_centers)
    orientations = rng.uniform(0, np.pi / 3, size=n_grid_cells)
    models: list[NeuralModel] = [
        GridCellModel(
            env,
            grid_spacing=grid_spacing,
            phase_offset=phase_centers[i % len(phase_centers)],
            grid_orientation=orientations[i],
        )
        for i in range(n_grid_cells)
    ]

    metadata = {
        "duration": duration,
        "arena_shape": "square",
        "arena_size": arena_size,
        "bin_size": bin_size,
        "grid_spacing": grid_spacing,
        "n_grid_cells": n_grid_cells,
        "cell_type": "grid",
        "trajectory_method": "ou",
        "coverage": "uniform",
        "seed": seed,
    }
    return _simulated_session(
        env,
        positions,
        times,
        models,
        spike_seed=spike_seed,
        show_progress=False,
        metadata=metadata,
    )
