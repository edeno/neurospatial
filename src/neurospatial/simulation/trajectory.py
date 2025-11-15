"""Trajectory simulation functions for generating movement patterns."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial import Environment


def _get_conversion_factor(from_unit: str, to_unit: str) -> float:
    """Get conversion factor between spatial units.

    Parameters
    ----------
    from_unit : str
        Source unit ('m', 'cm', 'mm', 'pixels').
    to_unit : str
        Target unit ('m', 'cm', 'mm', 'pixels').

    Returns
    -------
    factor : float
        Multiplication factor to convert from_unit to to_unit.

    Raises
    ------
    ValueError
        If units are incompatible or unknown.
    """
    # Define conversion to meters as base unit
    to_meters: dict[str, float | None] = {
        "m": 1.0,
        "cm": 0.01,
        "mm": 0.001,
        "pixels": None,  # Cannot convert without calibration
    }

    if from_unit not in to_meters:
        msg = f"Unknown unit '{from_unit}'. Supported: {list(to_meters.keys())}"
        raise ValueError(msg)

    if to_unit not in to_meters:
        msg = f"Unknown unit '{to_unit}'. Supported: {list(to_meters.keys())}"
        raise ValueError(msg)

    if from_unit == "pixels" or to_unit == "pixels":
        msg = "Cannot convert to/from 'pixels' without calibration"
        raise ValueError(msg)

    # Convert: from_unit → meters → to_unit
    # After the checks above, we know both values are not None
    from_factor = to_meters[from_unit]
    to_factor = to_meters[to_unit]
    assert from_factor is not None and to_factor is not None
    return from_factor / to_factor


def simulate_trajectory_ou(
    env: Environment,
    duration: float,
    dt: float = 0.01,
    start_position: NDArray[np.float64] | None = None,
    speed_mean: float = 0.08,
    speed_std: float = 0.04,
    coherence_time: float = 0.7,
    rotational_velocity_std: float = 120 * (np.pi / 180),  # 120 deg/s in radians
    rotational_velocity_coherence_time: float = 0.08,
    boundary_mode: Literal["reflect", "periodic", "stop"] = "reflect",
    speed_units: str | None = None,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate trajectory using Ornstein-Uhlenbeck process.

    Generates smooth, realistic random exploration in N-D space with
    biologically plausible movement statistics (fitted to Sargolini et al. 2006).

    Parameters
    ----------
    env : Environment
        The environment to simulate within.
    duration : float
        Simulation duration in seconds.
    dt : float, optional
        Time step in seconds (default: 0.01 = 10ms).
    start_position : NDArray[np.float64], shape (n_dims,), optional
        Starting position. If None, randomly chosen from bin centers.
    speed_mean : float, optional
        Mean speed in units/second (default: 0.08 m/s = 8 cm/s).
    speed_std : float, optional
        Speed standard deviation (default: 0.04).
    coherence_time : float, optional
        Time scale for speed autocorrelation in seconds (default: 0.7s).
        Controls speed smoothness (how gradually speed changes).
        For 2D environments, this does NOT affect direction changes
        (see rotational_velocity_coherence_time instead).
    rotational_velocity_std : float, optional
        Standard deviation of rotational velocity in radians/second
        (default: 120°/s = 2.094 rad/s). Only used for 2D environments.
        Controls how quickly direction changes. Typical range: 60-180°/s.
    rotational_velocity_coherence_time : float, optional
        Time scale for rotational velocity autocorrelation in seconds
        (default: 0.08s). Only used for 2D environments. Controls
        direction change smoothness:

        - 0.05-0.1s: Frequent direction changes (realistic exploration)
        - 0.1-0.2s: Moderate persistence
        - >0.2s: Very persistent straight-line movement

        Smaller values = more uniform spatial exploration.
    boundary_mode : {'reflect', 'periodic', 'stop'}, optional
        How to handle environment boundaries (default: 'reflect').

        - 'reflect': Trajectories bounce off walls elastically. Most realistic
          for typical experiments (open field arenas, water mazes).
        - 'periodic': Opposite boundaries wrap around (torus topology). Use
          only for specialized analyses. NOT biologically realistic.
        - 'stop': Trajectory halts at boundaries. Can create unrealistic
          accumulation at walls.
    speed_units : str | None, optional
        Units for speed_mean/speed_std. If None, assumes same units as env.units.
        If specified and different from env.units, will auto-convert.
        Example: speed_units="m" when env.units="cm" converts m/s to cm/s.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    positions : NDArray[np.float64], shape (n_time, n_dims)
        Continuous position coordinates.
    times : NDArray[np.float64], shape (n_time,)
        Time points in seconds.

    Raises
    ------
    ValueError
        If env.units is not set. Call env.units = "cm" before simulation.
    ValueError
        If speed_units are incompatible with env.units.

    Notes
    -----
    The OU process implements velocity-based random walk with different
    approaches for 2D vs N-D environments:

    **For 2D environments (recommended for realistic exploration):**

    Uses separate OU processes for rotational velocity (direction) and
    speed magnitude, following RatInABox (George et al. 2023):

    .. math::

        d\\omega = -\\theta_r \\omega dt + \\sigma_r dW_r

        d\\theta = \\omega dt

        \\mathbf{v}(t+dt) = R(\\theta) \\mathbf{v}(t)

    where ω is rotational velocity, θ is heading angle, R is rotation matrix,
    θ_r = 1/rotational_velocity_coherence_time, and
    σ_r = rotational_velocity_std * sqrt(2*θ_r/dt).

    Speed magnitude is controlled separately to maintain |v| ≈ speed_mean.
    This produces uniform spatial exploration with biologically realistic
    movement statistics.

    **For N-D environments (n_dims != 2):**

    Uses independent OU processes on each Cartesian velocity component:

    .. math::

        dv_i = -\\theta v_i dt + \\sigma dW_i

        dx_i = v_i dt

    where θ = 1/coherence_time and σ = speed_std * sqrt(2*θ/dt).

    Boundary Handling Implementation:

    - 'reflect': Upon boundary crossing, reflect position across boundary and
      reverse velocity component normal to boundary: v_perp → -v_perp
    - 'periodic': Wrap position using modulo on dimension_ranges, keep velocity
    - 'stop': Clamp position to nearest valid bin center, zero velocity in
      crossing direction

    References
    ----------
    George et al. (2023) eLife
    Sargolini et al. (2006) Science

    Examples
    --------
    Generate 60 seconds of trajectory in a 2D arena:

    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import simulate_trajectory_ou
    >>> import numpy as np
    >>>
    >>> # Create environment
    >>> samples = np.random.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(samples, bin_size=2.0)
    >>> env.units = "cm"
    >>>
    >>> # Simulate trajectory
    >>> positions, times = simulate_trajectory_ou(env, duration=60.0)
    >>> print(positions.shape)  # (6000, 2) at 100 Hz
    (6000, 2)
    >>> print(times.shape)
    (6000,)

    Customize movement parameters:

    >>> # Slower, more persistent movement
    >>> positions, times = simulate_trajectory_ou(
    ...     env,
    ...     duration=60.0,
    ...     speed_mean=5.0,  # cm/s
    ...     coherence_time=1.0,  # seconds
    ...     seed=42,  # reproducible
    ... )

    Convert speed units automatically:

    >>> # Specify speed in m/s, env in cm
    >>> positions, times = simulate_trajectory_ou(
    ...     env,
    ...     duration=60.0,
    ...     speed_mean=0.10,  # m/s
    ...     speed_units="m",  # auto-converts to cm/s
    ... )

    See Also
    --------
    simulate_trajectory_laps : Structured back-and-forth movement
    simulate_trajectory_sinusoidal : Simple periodic motion (1D)
    """
    # Validate environment units
    if env.units is None:
        msg = (
            "Environment units must be set before simulation. "
            "Set env.units = 'cm' (or 'm', 'mm', etc.)"
        )
        raise ValueError(msg)

    # Validate periodic boundary requirements
    if boundary_mode == "periodic" and env.dimension_ranges is None:
        msg = (
            "Environment dimension_ranges must be set for periodic boundary mode. "
            "The periodic mode wraps positions across boundaries using dimension_ranges. "
            "Either set env.dimension_ranges or use boundary_mode='reflect'."
        )
        raise ValueError(msg)

    # Convert speed units if needed
    if speed_units is not None and speed_units != env.units:
        conversion_factor = _get_conversion_factor(speed_units, env.units)
        speed_mean *= conversion_factor
        speed_std *= conversion_factor

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Initialize starting position
    if start_position is None:
        # Randomly choose from bin centers
        start_idx = rng.integers(0, env.n_bins)
        position = env.bin_centers[start_idx].copy()
    else:
        position = np.asarray(start_position, dtype=np.float64).copy()

    # Initialize velocity (random direction with magnitude ~speed_mean)
    n_dims = env.n_dims
    velocity = rng.standard_normal(n_dims)
    velocity = velocity / np.linalg.norm(velocity) * speed_mean  # Normalize and scale

    # OU process parameters depend on dimensionality
    if n_dims == 2:
        # For 2D: use rotational velocity OU (RatInABox approach)
        rotational_velocity = 0.0  # Start with zero angular velocity
        theta_rot = 1.0 / rotational_velocity_coherence_time
        # Euler-Maruyama discretization: sigma without /dt factor
        # because noise is scaled by sqrt(dt) when applied
        sigma_rot = rotational_velocity_std * np.sqrt(2 * theta_rot)
    else:
        # For N-D: use Cartesian velocity OU
        theta = 1.0 / coherence_time  # mean reversion rate
        # Euler-Maruyama discretization: sigma without /dt factor
        sigma = speed_std * np.sqrt(2 * theta)  # noise amplitude

    # Time setup
    n_steps = int(duration / dt)
    times = np.arange(n_steps) * dt

    # Pre-allocate trajectory
    positions = np.zeros((n_steps, n_dims), dtype=np.float64)
    positions[0] = position

    # Euler-Maruyama integration
    for i in range(1, n_steps):
        if n_dims == 2:
            # 2D: Rotational velocity OU (direction changes)
            # Update rotational velocity using OU: dω = -θ_r ω dt + σ_r dW
            dw_rot = rng.standard_normal() * np.sqrt(dt)
            rotational_velocity = (
                rotational_velocity
                - theta_rot * rotational_velocity * dt
                + sigma_rot * dw_rot
            )

            # Rotate velocity vector by dtheta = ω * dt
            dtheta = rotational_velocity * dt
            cos_dtheta = np.cos(dtheta)
            sin_dtheta = np.sin(dtheta)
            # Rotation matrix: [[cos, -sin], [sin, cos]]
            velocity = np.array(
                [
                    velocity[0] * cos_dtheta - velocity[1] * sin_dtheta,
                    velocity[0] * sin_dtheta + velocity[1] * cos_dtheta,
                ]
            )

            # Maintain constant speed (simplest approach)
            # Direction changes via rotation, speed stays at speed_mean
            speed = np.linalg.norm(velocity)
            if speed > 0:
                velocity = velocity * (speed_mean / speed)

            # Set max_speed for boundary reflection
            max_speed = 3 * speed_mean
        else:
            # N-D: Cartesian velocity OU (original approach)
            dw = rng.standard_normal(n_dims) * np.sqrt(dt)
            velocity = velocity - theta * velocity * dt + sigma * dw

            # Clip velocity magnitude to maintain realistic speeds
            speed = np.linalg.norm(velocity)
            max_speed = 3 * speed_mean
            if speed > max_speed:
                velocity = velocity * (max_speed / speed)

        # Handle boundaries - check BEFORE updating position
        if boundary_mode == "reflect":
            # Predict next position
            proposed_position = position + velocity * dt

            # Check if proposed position would be outside environment
            if not env.contains(proposed_position):
                # Find nearest valid bin to current position (not proposed)
                try:
                    nearest_bin = int(env.bin_at(position)[0])
                except Exception:
                    # If bin_at fails, use nearest bin center
                    distances = np.linalg.norm(env.bin_centers - position, axis=1)
                    nearest_bin = int(np.argmin(distances))

                # Compute outward normal (from current position toward boundary)
                # Direction from current position to proposed position
                displacement = proposed_position - position
                dist = np.linalg.norm(displacement)
                if dist > 1e-10:
                    normal = displacement / dist
                else:
                    # Fallback: random direction
                    normal = rng.standard_normal(n_dims)
                    normal = normal / np.linalg.norm(normal)

                # Reflect velocity: v' = v - 2(v·n)n
                v_parallel = np.dot(velocity, normal) * normal
                velocity = velocity - 2 * v_parallel

                # Normalize velocity after reflection
                speed_after = np.linalg.norm(velocity)
                if n_dims == 2:
                    # For 2D rotational OU: maintain constant speed
                    if speed_after > 0:
                        velocity = velocity * (speed_mean / speed_after)
                else:
                    # For N-D Cartesian OU: clip to max_speed
                    if speed_after > max_speed:
                        velocity = velocity * (max_speed / speed_after)

        # Update position with (possibly reflected) velocity
        position = position + velocity * dt

        # Ensure position stays within environment after update
        if boundary_mode == "reflect":
            if not env.contains(position):
                # Clamp to nearest valid bin
                try:
                    nearest_bin = int(env.bin_at(position)[0])
                except Exception:
                    distances = np.linalg.norm(env.bin_centers - position, axis=1)
                    nearest_bin = int(np.argmin(distances))
                position = env.bin_centers[nearest_bin].copy()

        elif boundary_mode == "periodic":
            # Wrap position using dimension ranges
            # Note: dimension_ranges validated at function entry
            for dim in range(n_dims):
                range_min, range_max = env.dimension_ranges[dim]  # type: ignore[index]
                range_size = range_max - range_min
                position[dim] = range_min + (position[dim] - range_min) % range_size

        elif boundary_mode == "stop" and not env.contains(position):
            # Clamp to environment bounds - find nearest valid bin
            try:
                nearest_bin = int(env.bin_at(position)[0])
            except Exception:
                distances = np.linalg.norm(env.bin_centers - position, axis=1)
                nearest_bin = int(np.argmin(distances))

            position = env.bin_centers[nearest_bin].copy()

            # Zero velocity in crossing direction (simple: zero all velocity)
            velocity = np.zeros(n_dims)

        # Note: No else block needed - Literal type ensures all cases covered

        positions[i] = position

    return positions, times  # type: ignore[return-value]


def simulate_trajectory_sinusoidal(
    env: Environment,
    duration: float,
    sampling_frequency: float = 500.0,
    speed: float = 10.0,
    period: float | None = None,
    pause_duration: float = 0.0,
    pause_at_peaks: bool = True,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate sinusoidal trajectory along a 1D track.

    Generates simple back-and-forth motion with optional pauses at track ends.
    Useful for testing place field detection on linear tracks.

    Parameters
    ----------
    env : Environment
        The 1D environment (must have env.is_1d == True).
    duration : float
        Total simulation duration in seconds.
    sampling_frequency : float, optional
        Samples per second (default: 500 Hz).
    speed : float, optional
        Running speed in environment units/second (default: 10.0 cm/s).
    period : float | None, optional
        Period of sinusoidal motion in seconds. If None, computed from
        track length and speed: period = 2 * track_length / speed.
    pause_duration : float, optional
        Pause at track ends in seconds (default: 0.0).
    pause_at_peaks : bool, optional
        Whether to pause at track extrema (default: True).
    seed : int | None, optional
        Random seed for reproducibility (currently unused, reserved for future use).

    Returns
    -------
    positions : NDArray[np.float64], shape (n_time, 1)
        1D position coordinates.
    times : NDArray[np.float64], shape (n_time,)
        Time points in seconds.

    Raises
    ------
    ValueError
        If environment is not 1D (env.is_1d == False).

    Examples
    --------
    Requires 1D environment created with Environment.from_graph():

    .. code-block:: python

        # Note: This example requires GraphLayout (1D track) which needs
        # track-linearization package. See Environment.from_graph() documentation.
        from neurospatial import Environment
        from neurospatial.simulation import simulate_trajectory_sinusoidal

        # Create 1D track using from_graph()
        env_1d = Environment.from_graph(...)  # See from_graph() docs
        env_1d.units = "cm"

        # Generate sinusoidal motion
        positions, times = simulate_trajectory_sinusoidal(
            env_1d, duration=120.0, speed=20.0
        )
        print(positions.shape)  # (60000, 1) for 120s at 500Hz

    See Also
    --------
    simulate_trajectory_ou : Realistic random exploration
    simulate_trajectory_laps : Structured lap-based movement
    """
    if not env.is_1d:
        msg = "Sinusoidal trajectory only works for 1D environments (env.is_1d must be True)"
        raise ValueError(msg)

    # Get track length
    if env.dimension_ranges is None:
        msg = "Environment dimension_ranges must be set for sinusoidal trajectory"
        raise ValueError(msg)
    range_min, range_max = env.dimension_ranges[0]
    track_length = range_max - range_min

    # Compute period if not provided
    if period is None:
        period = 2 * track_length / speed

    # Time setup
    dt = 1.0 / sampling_frequency
    n_steps = int(duration / dt)
    times = np.arange(n_steps) * dt

    # Generate sinusoidal position
    # Position oscillates between range_min and range_max
    center = (range_max + range_min) / 2
    amplitude = (range_max - range_min) / 2

    # Sinusoidal motion: x(t) = center + amplitude * sin(2π t / period)
    angular_freq = 2 * np.pi / period
    positions = center + amplitude * np.sin(angular_freq * times)

    # Handle pauses at peaks if requested
    if pause_at_peaks and pause_duration > 0:
        # Find peaks (times when sin(ωt) ≈ ±1)
        phase = angular_freq * times
        # Peaks occur when phase ≈ π/2 + nπ
        at_peak = np.abs(np.abs(np.sin(phase)) - 1.0) < 0.1

        # Create pause mask
        pause_samples = int(pause_duration * sampling_frequency)
        paused = np.zeros(n_steps, dtype=bool)

        # Mark samples near peaks as paused
        peak_indices = np.where(at_peak)[0]
        for idx in peak_indices:
            start = max(0, idx - pause_samples // 2)
            end = min(n_steps, idx + pause_samples // 2)
            paused[start:end] = True

        # Hold position constant during pauses
        last_valid_pos = positions[0]
        for i in range(n_steps):
            if paused[i]:
                positions[i] = last_valid_pos
            else:
                last_valid_pos = positions[i]

    # Reshape to (n_time, 1) for consistency with N-D trajectories
    positions = positions.reshape(-1, 1)

    return positions, times  # type: ignore[return-value]


def simulate_trajectory_laps(
    env: Environment,
    n_laps: int,
    speed_mean: float = 0.1,
    speed_std: float = 0.02,
    outbound_path: list[int] | None = None,
    inbound_path: list[int] | None = None,
    pause_duration: float = 0.5,
    sampling_frequency: float = 500.0,
    seed: int | None = None,
    return_metadata: bool = False,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64]]
    | tuple[NDArray[np.float64], NDArray[np.float64], dict]
):
    """Simulate structured lap-based trajectory.

    Generates back-and-forth movement along specified or auto-computed paths,
    useful for alternation tasks, T-mazes, and linear tracks.

    Parameters
    ----------
    env : Environment
        The spatial environment.
    n_laps : int
        Number of complete laps (outbound + inbound).
    speed_mean : float, optional
        Mean speed in environment units/second (default: 0.1).
    speed_std : float, optional
        Speed variability (default: 0.02).
    outbound_path : list[int] | None, optional
        Sequence of bin indices for outbound trajectory.
        If None, uses shortest path from start to end of environment.
    inbound_path : list[int] | None, optional
        Sequence of bin indices for inbound trajectory.
        If None, reverses outbound_path.
    pause_duration : float, optional
        Pause at lap ends in seconds (default: 0.5s).
    sampling_frequency : float, optional
        Samples per second (default: 500 Hz).
    seed : int | None, optional
        Random seed for reproducibility.
    return_metadata : bool, optional
        If True, returns metadata dict with lap_ids, lap_boundaries, and
        direction arrays (default: False).

    Returns
    -------
    positions : NDArray[np.float64], shape (n_time, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_time,)
        Time points in seconds.
    metadata : dict, optional (if return_metadata=True)
        Dictionary with keys:

        - 'lap_ids' : NDArray[np.int_], shape (n_time,) - Lap number for each time
        - 'lap_boundaries' : NDArray[np.int_] - Indices where laps start
        - 'direction' : NDArray[str] - 'outbound' or 'inbound' for each time

    Raises
    ------
    ValueError
        If n_laps <= 0, speed_mean <= 0, speed_std < 0, pause_duration < 0,
        or sampling_frequency <= 0.
    ValueError
        If custom paths contain invalid bin indices or are empty.

    Examples
    --------
    Generate laps in a 2D environment:

    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import simulate_trajectory_laps
    >>> import numpy as np
    >>>
    >>> # Create environment
    >>> samples = np.random.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(samples, bin_size=2.0)
    >>> env.units = "cm"
    >>>
    >>> # Simulate laps
    >>> positions, times = simulate_trajectory_laps(env, n_laps=5)
    >>> print(positions.shape[0] > 0)  # Should have positions
    True

    With metadata:

    >>> positions, times, metadata = simulate_trajectory_laps(
    ...     env, n_laps=3, return_metadata=True
    ... )
    >>> print("lap_ids" in metadata)
    True
    >>> print("direction" in metadata)
    True

    See Also
    --------
    simulate_trajectory_ou : Realistic random exploration
    simulate_trajectory_sinusoidal : Simple periodic motion (1D)
    """
    # Validate parameters
    if n_laps <= 0:
        msg = f"n_laps must be positive (got {n_laps})"
        raise ValueError(msg)

    if speed_mean <= 0:
        msg = f"speed_mean must be positive (got {speed_mean})"
        raise ValueError(msg)

    if speed_std < 0:
        msg = f"speed_std must be non-negative (got {speed_std})"
        raise ValueError(msg)

    if pause_duration < 0:
        msg = f"pause_duration must be non-negative (got {pause_duration})"
        raise ValueError(msg)

    if sampling_frequency <= 0:
        msg = f"sampling_frequency must be positive (got {sampling_frequency})"
        raise ValueError(msg)

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Compute paths if not provided
    if outbound_path is None:
        # Use shortest path between environment extrema
        # Find bins at minimum and maximum positions in first dimension
        bin_positions = env.bin_centers
        first_dim_positions = bin_positions[:, 0]
        start_bin = int(np.argmin(first_dim_positions))
        end_bin = int(np.argmax(first_dim_positions))

        # Compute shortest path
        import networkx as nx

        try:
            path = nx.shortest_path(env.connectivity, start_bin, end_bin)
            outbound_path = path
        except nx.NetworkXNoPath:
            # If no path, just use start and end
            outbound_path = [start_bin, end_bin]

    if inbound_path is None:
        # Reverse outbound path
        inbound_path = list(reversed(outbound_path))

    # Validate paths
    for path_name, path in [
        ("outbound_path", outbound_path),
        ("inbound_path", inbound_path),
    ]:
        if not path:
            msg = f"{path_name} must contain at least one bin"
            raise ValueError(msg)
        for bin_idx in path:
            if bin_idx < 0 or bin_idx >= env.n_bins:
                msg = (
                    f"{path_name} contains invalid bin index {bin_idx} "
                    f"(must be in [0, {env.n_bins}))"
                )
                raise ValueError(msg)

    # Time step
    dt = 1.0 / sampling_frequency

    # Build trajectory by concatenating laps
    all_positions = []
    all_times = []
    all_lap_ids = []
    all_directions = []
    lap_boundaries_list = [0]  # Start of first lap (will convert to array later)
    current_time = 0.0

    for lap_idx in range(n_laps):
        # Alternate between outbound and inbound
        if lap_idx % 2 == 0:
            path = outbound_path
            direction = "outbound"
        else:
            path = inbound_path
            direction = "inbound"

        # Generate positions along path
        path_positions = env.bin_centers[path]

        # Compute distances between consecutive bins
        distances = np.linalg.norm(np.diff(path_positions, axis=0), axis=1)
        total_distance = np.sum(distances)

        # Compute time to traverse path (with speed variability)
        speed = rng.normal(speed_mean, speed_std)
        # Ensure positive speed with absolute minimum
        speed = max(speed, 0.01)
        path_duration = total_distance / speed

        # Number of samples for this path
        n_samples = max(2, int(path_duration / dt))

        # Interpolate positions along path
        # Create cumulative distances
        cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
        interpolation_points = np.linspace(0, total_distance, n_samples)

        # Interpolate each dimension separately
        lap_positions = np.zeros((n_samples, env.n_dims))
        for dim in range(env.n_dims):
            lap_positions[:, dim] = np.interp(
                interpolation_points, cumulative_distances, path_positions[:, dim]
            )

        # Ensure all positions are within environment bounds by mapping to bins
        # This handles edge cases where interpolation might go slightly outside
        from neurospatial import map_points_to_bins

        bin_indices = map_points_to_bins(lap_positions, env, tie_break="lowest_index")
        # Use bin centers to ensure all positions are valid
        lap_positions = env.bin_centers[bin_indices]

        # Generate times for this lap
        lap_times = current_time + np.arange(n_samples) * dt
        current_time = lap_times[-1] + dt

        # Append to trajectory
        all_positions.append(lap_positions)
        all_times.append(lap_times)
        all_lap_ids.extend([lap_idx] * n_samples)
        all_directions.extend([direction] * n_samples)

        # Add pause at lap end (except after last lap)
        if lap_idx < n_laps - 1 and pause_duration > 0:
            pause_samples = int(pause_duration * sampling_frequency)
            if pause_samples > 0:
                # Hold position constant during pause
                pause_position = lap_positions[-1:].repeat(pause_samples, axis=0)
                pause_times = current_time + np.arange(pause_samples) * dt
                current_time = pause_times[-1] + dt

                all_positions.append(pause_position)
                all_times.append(pause_times)
                all_lap_ids.extend([lap_idx] * pause_samples)
                all_directions.extend([direction] * pause_samples)

        # Record lap boundary (start of next lap)
        if lap_idx < n_laps - 1:
            lap_boundaries_list.append(len(all_lap_ids))

    # Concatenate all segments
    positions = np.vstack(all_positions)
    times = np.concatenate(all_times)
    lap_ids = np.array(all_lap_ids, dtype=np.int_)
    directions = np.array(all_directions)
    lap_boundaries = np.array(lap_boundaries_list, dtype=np.int_)

    if return_metadata:
        metadata = {
            "lap_ids": lap_ids,
            "lap_boundaries": lap_boundaries,
            "direction": directions,
        }
        return positions, times, metadata
    else:
        return positions, times


def simulate_trajectory_coverage(
    env: Environment,
    duration: float,
    *,
    speed: float = 0.08,
    sampling_frequency: float = 100.0,
    pause_at_boundaries: float = 0.0,
    coverage_bias: float = 2.0,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate trajectory ensuring good spatial coverage.

    Uses biased random walk on the environment's connectivity graph,
    preferentially visiting less-explored bins. Guarantees >90% coverage
    for most environments without looking artificial.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    duration : float
        Total simulation time in seconds.
    speed : float, optional
        Movement speed in environment units/second (default: 0.08 m/s).
    sampling_frequency : float, optional
        Samples per second (default: 100 Hz).
    pause_at_boundaries : float, optional
        Pause duration at boundary bins in seconds (default: 0.0).
    coverage_bias : float, optional
        Strength of bias toward unvisited bins (default: 2.0).
        Higher values = more systematic exploration.
        Lower values = more random movement.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    positions : NDArray[np.float64], shape (n_times, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_times,)
        Time points in seconds.

    Notes
    -----
    Strategy:
    1. Track occupancy count for each bin
    2. At each step, get neighbors from connectivity graph
    3. Weight neighbors inversely to their occupancy: w = 1 / (1 + count)^bias
    4. Choose next bin probabilistically based on weights
    5. Interpolate smooth movement along graph edges
    6. Add small jitter within bins for realism

    This produces realistic-looking trajectories while ensuring sufficient
    sampling for place field estimation.

    Examples
    --------
    >>> from neurospatial import Environment
    >>> import numpy as np
    >>>
    >>> # Create environment
    >>> data = np.random.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>>
    >>> # Generate coverage trajectory
    >>> positions, times = simulate_trajectory_coverage(env, duration=180.0)
    >>>
    >>> # Check coverage
    >>> bin_indices = env.bin_at(positions)
    >>> unique_bins = len(np.unique(bin_indices[bin_indices >= 0]))
    >>> coverage_pct = 100 * unique_bins / env.n_bins
    >>> print(f"Visited {coverage_pct:.1f}% of bins")
    >>> # Expected: >90% for most environments

    See Also
    --------
    simulate_trajectory_ou : Realistic random exploration
    simulate_trajectory_laps : Structured back-and-forth movement
    simulate_trajectory_goal_directed : Multi-goal shortest-path navigation
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1.0 / sampling_frequency
    n_samples = int(duration / dt)

    # Initialize
    current_bin = np.random.randint(env.n_bins)
    occupancy_counts = np.zeros(env.n_bins, dtype=np.int32)
    occupancy_counts[current_bin] = 1

    # Pre-compute boundary bins for pause logic
    boundary_bins = set(env.boundary_bins()) if pause_at_boundaries > 0 else set()

    # Build trajectory bin-by-bin
    trajectory_bins = []
    trajectory_bins.append(current_bin)

    samples_generated = 0

    while samples_generated < n_samples:
        # Get neighbors from graph
        neighbors = list(env.connectivity.neighbors(current_bin))

        if not neighbors:
            # Isolated bin - stay here
            trajectory_bins.append(current_bin)
            samples_generated += 1
            continue

        # Compute weights based on occupancy
        # Less-visited bins get higher weight
        neighbor_counts = occupancy_counts[neighbors]
        weights = 1.0 / (1.0 + neighbor_counts) ** coverage_bias
        weights /= weights.sum()

        # Choose next bin
        next_bin = np.random.choice(neighbors, p=weights)

        # Compute movement duration using Euclidean distance from bin centers
        pos_current = env.bin_centers[current_bin]
        pos_next = env.bin_centers[next_bin]
        edge_distance = np.linalg.norm(pos_next - pos_current)

        # Handle invalid distances (shouldn't happen, but be safe)
        if edge_distance <= 0:
            trajectory_bins.append(next_bin)
            samples_generated += 1
            current_bin = next_bin
            continue

        move_duration = edge_distance / speed
        n_move_samples = max(1, int(move_duration * sampling_frequency))

        # Add movement samples
        trajectory_bins.extend([next_bin] * n_move_samples)
        occupancy_counts[next_bin] += n_move_samples
        samples_generated += n_move_samples

        # Pause at boundaries if requested
        if pause_at_boundaries > 0 and next_bin in boundary_bins:
            n_pause_samples = int(pause_at_boundaries * sampling_frequency)
            trajectory_bins.extend([next_bin] * n_pause_samples)
            occupancy_counts[next_bin] += n_pause_samples
            samples_generated += n_pause_samples

        current_bin = next_bin

    # Trim to exact duration
    trajectory_bins = np.array(trajectory_bins[:n_samples], dtype=np.int32)

    # Convert bin indices to continuous positions
    positions = env.bin_centers[trajectory_bins].copy()

    # Add small jitter within bins for realism
    # Jitter std is 20% of mean bin size (keeps points mostly within bins)
    mean_bin_size = np.mean(env.bin_sizes)
    jitter_std = mean_bin_size * 0.2
    jitter = np.random.randn(*positions.shape) * jitter_std
    positions += jitter

    # Generate time vector
    times = np.arange(n_samples) * dt

    return positions, times


def simulate_trajectory_goal_directed(
    env: Environment,
    goals: list[NDArray[np.float64]] | list[list[float]],
    n_trials: int,
    *,
    speed: float = 0.08,
    sampling_frequency: float = 100.0,
    pause_at_goal: float = 1.0,
    trial_order: Literal["sequential", "random", "alternating"] = "sequential",
    add_jitter: bool = True,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int_]]:
    """Generate goal-directed trajectory between multiple reward locations.

    Uses Environment.shortest_path() to find routes between goals.
    Simulates realistic experimental paradigms (linear track, T-maze,
    plus maze, etc.).

    Parameters
    ----------
    env : Environment
        Spatial environment.
    goals : list of array-like, each shape (n_dims,)
        Reward locations in environment coordinates.
        Examples:
        - Linear track: [[0, 0], [100, 0]]
        - T-maze: [[20, 80], [80, 80]] (left and right arms)
        - Plus maze: [[50, 10], [90, 50], [50, 90], [10, 50]]
    n_trials : int
        Number of goal-to-goal runs.
    speed : float, optional
        Movement speed in units/second (default: 0.08 m/s).
    sampling_frequency : float, optional
        Samples per second (default: 100 Hz).
    pause_at_goal : float, optional
        Pause duration at each goal in seconds (default: 1.0).
    trial_order : {'sequential', 'random', 'alternating'}, optional
        How to sequence goals (default: 'sequential').
        - 'sequential': goal[0]→goal[1]→goal[0]→goal[1]...
        - 'random': random goal each trial
        - 'alternating': goal[0]→goal[1]→goal[2]→...→goal[0] (cycles through all)
    add_jitter : bool, optional
        Add small position jitter for realism (default: True).
    seed : int | None, optional
        Random seed.

    Returns
    -------
    positions : NDArray[np.float64], shape (n_times, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_times,)
        Time points in seconds.
    trial_ids : NDArray[np.int_], shape (n_times,)
        Trial number for each time point (0-indexed).

    Notes
    -----
    Movement along paths is smooth with constant speed. The function:
    1. Converts goal positions to bin indices
    2. Generates trial sequence based on trial_order
    3. For each trial:
       - Finds shortest path between current bin and goal bin
       - Interpolates smooth movement along path edges
       - Pauses at goal location
    4. Concatenates all trials into continuous trajectory

    Examples
    --------
    Linear track - back and forth:

    >>> from neurospatial import Environment
    >>> import numpy as np
    >>>
    >>> # Create 1D track environment
    >>> track_positions = np.linspace(0, 170, 1000)[:, np.newaxis]
    >>> env = Environment.from_samples(track_positions, bin_size=2.0)
    >>> env.units = "cm"
    >>>
    >>> # Goals at track ends
    >>> goals = [[0.0], [170.0]]
    >>>
    >>> # Generate 20 back-and-forth runs
    >>> positions, times, trials = simulate_trajectory_goal_directed(
    ...     env, goals, n_trials=20, trial_order='sequential'
    ... )
    >>>
    >>> # Check trial structure
    >>> print(f"Total duration: {times[-1]:.1f} seconds")
    >>> print(f"Unique trials: {len(np.unique(trials))}")

    T-maze alternation:

    >>> # Create T-maze environment (simplified as 2D grid)
    >>> tmaze_data = np.random.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(tmaze_data, bin_size=2.0)
    >>> env.units = "cm"
    >>>
    >>> # Goals: left and right arms
    >>> goals = [
    ...     [20, 80],   # Left reward well
    ...     [80, 80],   # Right reward well
    ... ]
    >>>
    >>> # Alternating trials (L-R-L-R-...)
    >>> positions, times, trials = simulate_trajectory_goal_directed(
    ...     env, goals, n_trials=40, trial_order='sequential',
    ...     pause_at_goal=1.5
    ... )
    >>>
    >>> # Extract left vs right trials
    >>> left_trials = trials[::2]  # Even trials (0, 2, 4, ...)
    >>> right_trials = trials[1::2]  # Odd trials (1, 3, 5, ...)

    Plus maze - random arm visits:

    >>> # Four goals (N, E, S, W arms)
    >>> goals = [
    ...     [50, 10],   # South
    ...     [90, 50],   # East
    ...     [50, 90],   # North
    ...     [10, 50],   # West
    ... ]
    >>>
    >>> # Random order
    >>> positions, times, trials = simulate_trajectory_goal_directed(
    ...     env, goals, n_trials=100, trial_order='random', seed=42
    ... )

    See Also
    --------
    simulate_trajectory_laps : Structured back-and-forth on fixed path
    simulate_trajectory_coverage : Coverage-ensuring exploration
    simulate_trajectory_ou : Realistic random exploration
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1.0 / sampling_frequency

    # Convert goals to numpy arrays
    goals = [np.asarray(g, dtype=np.float64) for g in goals]

    # Convert goals to bin indices
    goal_bins = []
    for goal_pos in goals:
        if goal_pos.ndim == 1:
            goal_pos = goal_pos[np.newaxis, :]  # Add batch dim
        bin_idx = env.bin_at(goal_pos)[0]
        if bin_idx < 0:
            raise ValueError(f"Goal position {goal_pos.squeeze()} is outside environment")
        goal_bins.append(bin_idx)

    # Generate trial sequence
    n_goals = len(goals)

    if trial_order == "sequential":
        # Alternate between goals: 0→1→0→1→...
        if n_goals == 2:
            sequence = np.tile([0, 1], (n_trials // 2) + 1)[:n_trials]
        else:
            # For >2 goals, visit in order then reverse
            forward = list(range(n_goals))
            backward = list(reversed(forward))
            cycle = forward + backward
            sequence = np.tile(cycle, (n_trials // len(cycle)) + 1)[:n_trials]

    elif trial_order == "alternating":
        # Cycle through all goals in order: 0→1→2→...→0→1→2→...
        sequence = np.tile(range(n_goals), (n_trials // n_goals) + 1)[:n_trials]

    elif trial_order == "random":
        # Random goal each trial
        sequence = np.random.choice(n_goals, size=n_trials)

    else:
        raise ValueError(f"Unknown trial_order: {trial_order}. Use 'sequential', 'random', or 'alternating'")

    # Build trajectory trial-by-trial
    all_positions = []
    all_times = []
    all_trial_ids = []

    current_time = 0.0

    # Start at center of environment (not at a goal)
    center_idx = env.n_bins // 2
    current_bin = center_idx

    for trial_idx, goal_idx in enumerate(sequence):
        goal_bin = goal_bins[goal_idx]

        # Skip if already at goal (avoids zero-length trials)
        if current_bin == goal_bin:
            continue

        # Find shortest path
        path = env.shortest_path(current_bin, goal_bin)

        if path is None:
            # No path exists - skip this trial
            print(f"Warning: No path from bin {current_bin} to {goal_bin}, skipping trial {trial_idx}")
            continue

        # Walk along path, edge by edge
        for i in range(len(path) - 1):
            bin_a, bin_b = path[i], path[i + 1]

            # Get edge distance using Euclidean distance from bin centers
            pos_a = env.bin_centers[bin_a]
            pos_b = env.bin_centers[bin_b]
            edge_distance = np.linalg.norm(pos_b - pos_a)

            # Skip if edge is invalid
            if edge_distance <= 0:
                continue

            move_duration = edge_distance / speed
            n_move_samples = max(1, int(move_duration * sampling_frequency))

            # Linear interpolation along edge
            alphas = np.linspace(0, 1, n_move_samples, endpoint=False)
            segment_positions = pos_a + alphas[:, np.newaxis] * (pos_b - pos_a)

            # Time and trial ID vectors
            segment_times = current_time + np.arange(n_move_samples) * dt
            segment_trial_ids = np.full(n_move_samples, trial_idx, dtype=np.int32)

            all_positions.append(segment_positions)
            all_times.append(segment_times)
            all_trial_ids.append(segment_trial_ids)

            current_time = segment_times[-1] + dt

        # Pause at goal
        if pause_at_goal > 0:
            n_pause_samples = int(pause_at_goal * sampling_frequency)
            pause_pos = np.tile(env.bin_centers[goal_bin], (n_pause_samples, 1))
            pause_times = current_time + np.arange(n_pause_samples) * dt
            pause_trial_ids = np.full(n_pause_samples, trial_idx, dtype=np.int32)

            all_positions.append(pause_pos)
            all_times.append(pause_times)
            all_trial_ids.append(pause_trial_ids)

            current_time = pause_times[-1] + dt

        current_bin = goal_bin

    # Concatenate all segments
    positions = np.vstack(all_positions)
    times = np.concatenate(all_times)
    trial_ids = np.concatenate(all_trial_ids)

    # Add jitter for realism
    if add_jitter:
        mean_bin_size = np.mean(env.bin_sizes)
        jitter_std = mean_bin_size * 0.15  # 15% of mean bin size
        jitter = np.random.randn(*positions.shape) * jitter_std
        positions += jitter

    return positions, times, trial_ids
