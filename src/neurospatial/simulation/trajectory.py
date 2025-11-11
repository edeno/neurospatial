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
        Time scale for velocity autocorrelation in seconds (default: 0.7s).
        Controls movement smoothness:

        - 0.3-0.5s: Jittery, erratic movement (stressed/novel environment)
        - 0.6-0.8s: Natural exploration (default, from Sargolini 2006)
        - 0.9-1.2s: Smooth, persistent movement (familiar environment)

        Larger values = slower direction changes, more persistent trajectories.
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
    The OU process implements an N-dimensional velocity-based random walk:

    For each spatial dimension i:

    .. math::

        dv_i = -\\theta v_i dt + \\sigma dW_i

        dx_i = v_i dt

    where:

    - θ = 1/coherence_time controls mean reversion rate (velocity decorrelation)
    - σ = speed_std * sqrt(2*θ) ensures stationary speed distribution
    - dW_i are independent Wiener increments
    - Velocity magnitude is clipped to ensure |v| ≈ speed_mean

    The stationary distribution of velocity has zero mean and variance speed_std².
    This produces realistic movement with controlled speed and smoothness.

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

    # Initialize velocity (random direction, zero mean)
    n_dims = env.n_dims
    velocity = rng.standard_normal(n_dims) * speed_std

    # OU process parameters
    theta = 1.0 / coherence_time  # mean reversion rate
    sigma = speed_std * np.sqrt(2 * theta)  # noise amplitude

    # Time setup
    n_steps = int(duration / dt)
    times = np.arange(n_steps) * dt

    # Pre-allocate trajectory
    positions = np.zeros((n_steps, n_dims), dtype=np.float64)
    positions[0] = position

    # Euler-Maruyama integration
    for i in range(1, n_steps):
        # OU process update: dv = -θ v dt + σ dW
        dw = rng.standard_normal(n_dims) * np.sqrt(dt)
        velocity = velocity - theta * velocity * dt + sigma * dw

        # Clip velocity magnitude to maintain speed_mean (with some tolerance)
        speed = np.linalg.norm(velocity)
        if speed > 2 * speed_mean:  # Clip extreme speeds
            velocity = velocity * (2 * speed_mean / speed)

        # Update position
        position = position + velocity * dt

        # Handle boundaries
        if boundary_mode == "reflect":
            # Check if position is outside environment
            if not env.contains(position):
                # Find nearest valid bin
                try:
                    nearest_bin = int(env.bin_at(position)[0])
                except Exception:
                    # If bin_at fails (far outside), use nearest bin center
                    distances = np.linalg.norm(env.bin_centers - position, axis=1)
                    nearest_bin = int(np.argmin(distances))

                boundary_point = env.bin_centers[nearest_bin]

                # Compute outward normal (from boundary toward invalid position)
                # Simple approximation: direction from boundary to attempted position
                displacement = position - boundary_point
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

                # Place position just inside boundary
                epsilon = 0.1 * np.mean(env.bin_sizes)  # Small offset
                position = boundary_point - epsilon * normal

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
