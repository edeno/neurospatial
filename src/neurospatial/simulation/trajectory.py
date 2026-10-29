"""Trajectory simulation functions for generating movement patterns."""

from itertools import pairwise
from typing import Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._units import default_speed
from neurospatial.environment import Environment
from neurospatial.geometry import require_cartesian
from neurospatial.simulation._lap_routes import LapRouter
from neurospatial.simulation._walls import Walls


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


#: Default exploration speed, in centimeters per second: the RatInABox
#: default (0.08 m/s; George et al. 2023).
_DEFAULT_OU_SPEED_CM_PER_S = 8.0

# Relative rounding allowed in ``duration / dt`` before a sample is dropped.
_SAMPLE_COUNT_RTOL = 1e-9


def _require_finite(
    checks: list[tuple[str, float, bool]], why: str | None = None
) -> None:
    """Refuse a value that is not finite, or not positive (or non-negative).

    ``checks`` holds ``(name, value, positive)``; ``positive`` False allows 0.
    """
    for name, value, positive in checks:
        if not np.isfinite(value) or (value <= 0 if positive else value < 0):
            bound = "positive" if positive else "non-negative"
            raise ValueError(
                _format_error(
                    f"{name} must be finite and {bound}, got {value}.",
                    why=why,
                    fix=f"pass a finite {bound} {name}",
                )
            )


def _sample_count(duration: float, step: float, step_name: str, call: str) -> int:
    """Number of samples ``0, step, 2 step, ...`` before ``duration``.

    ``duration / step`` is rounded up when it is within
    :data:`_SAMPLE_COUNT_RTOL` of an integer, so ``0.3 / 0.1``
    (2.9999999999999996) gives 3 samples.
    """
    n_samples = int(np.floor(duration / step * (1 + _SAMPLE_COUNT_RTOL)))
    if n_samples < 1:
        raise ValueError(
            _format_error(
                f"{call}: duration={duration} s is shorter than one sample "
                f"({step_name}={step} s).",
                why="Why: the trajectory is sampled every "
                f"{step_name} seconds from time 0, so it needs at least one "
                "sample.",
                fix=f"pass a duration of at least {step} s, or a shorter {step_name}",
            )
        )
    return n_samples


def _require_bins_fill_ranges(env: Environment) -> None:
    """Refuse periodic wrapping unless the bins fill ``env.dimension_ranges``."""
    ranges = env.dimension_ranges
    box_measure = (
        None if ranges is None else float(np.prod(np.ptp(np.asarray(ranges), axis=1)))
    )
    if box_measure is None or not np.isclose(
        float(np.sum(env.bin_sizes)), box_measure, rtol=1e-9
    ):
        raise ValueError(
            _format_error(
                "boundary_mode='periodic' needs bins that fill "
                "env.dimension_ranges, and these bins do not fill them.",
                why="Why: periodic mode joins opposite sides of "
                "env.dimension_ranges into a torus; where the bins leave part "
                "of that box empty (a circular arena, an irregular outline), "
                "wrapped positions land outside the environment.",
                fix="use boundary_mode='reflect', or build the environment as "
                "a full rectangular grid such as Environment.from_polygon("
                "box(...))",
            )
        )


def simulate_trajectory_ou(
    env: Environment,
    *,
    duration: float,
    speed_units: str,
    dt: float = 0.01,
    start_position: NDArray[np.float64] | None = None,
    speed_mean: float | None = None,
    speed_std: float | None = None,
    coherence_time: float = 0.7,
    rotational_velocity_std: float = 120 * (np.pi / 180),  # 120 deg/s in radians
    rotational_velocity_coherence_time: float = 0.08,
    boundary_mode: Literal["reflect", "periodic", "stop"] = "reflect",
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate random exploration with smoothly varying speed and heading.

    The animal's speed and turning rate are Ornstein-Uhlenbeck processes, as
    in RatInABox (George et al. 2023). It never crosses the arena's outline or
    an internal wall: it bounces off them like a billiard ball.

    Parameters
    ----------
    env : Environment
        The environment to simulate within.
    duration : float
        Simulation duration in seconds.
    speed_units : str
        Units of ``speed_mean`` and ``speed_std``. Must equal ``env.units``;
        speeds you pass are not converted between units.
    dt : float, optional
        Time step in seconds (default: 0.01 = 10ms).
    start_position : NDArray[np.float64], shape (n_dims,), optional
        Starting position, inside the environment: ``env.contains`` is True
        for it, or it lies on the outline within rounding. If None, randomly
        chosen from bin centers.
    speed_mean : float or None, optional
        Mean running speed in ``speed_units`` per second. None (default) uses
        8 cm/s, the RatInABox default for a foraging rat, expressed in
        ``env.units`` (0.08 for meters); environments without metric units
        need an explicit value.
    speed_std : float or None, optional
        Standard deviation of the running speed in ``speed_units`` per second.
        None (default) uses ``speed_mean / 2`` (4 cm/s with the default mean).
        0 gives a constant speed.
    coherence_time : float, optional
        Autocorrelation time of the speed in seconds (default: 0.7 s, the
        RatInABox default): how gradually the animal speeds up and slows down.
    rotational_velocity_std : float, optional
        Standard deviation of the turning rate in radians/second (default:
        120°/s = 2.094 rad/s, the RatInABox default). Used in 2-D.
    rotational_velocity_coherence_time : float, optional
        Autocorrelation time of the turning rate in seconds (default: 0.08 s,
        the RatInABox default). In 2-D it sets how long a turn persists; in
        three or more dimensions, how long the heading persists.
    boundary_mode : {'reflect', 'periodic', 'stop'}, optional
        What happens when a step would cross a wall (default: 'reflect').

        - 'reflect': the velocity is mirrored about the wall, keeping its
          component along the wall and reversing the component into it, and
          the animal takes the step in the mirrored direction.
        - 'periodic': opposite sides of ``env.dimension_ranges`` are joined
          (a torus) and walls are ignored. Not biologically realistic. The
          bins must fill ``env.dimension_ranges`` (a full rectangular grid),
          so that every wrapped position is inside the environment.
        - 'stop': the animal waits in place until its heading turns away
          from the wall. In 1-D, where the heading turns only at a wall, it
          reverses there as with 'reflect'.
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
        If env.units is not set. Pass units="cm" when constructing the environment.
    ValueError
        If speed_units differs from env.units, or a speed or time constant is
        not finite and positive (``speed_std`` may be 0), or ``duration`` is
        shorter than ``dt``.
    ValueError
        If ``speed_mean`` is None and ``env.units`` is not a metric length.
    ValueError
        If ``boundary_mode`` is not one of the three modes, or is
        ``'periodic'`` and the bins do not fill ``env.dimension_ranges``.
    ValueError
        If ``start_position`` is not one point of ``env.n_dims`` coordinates
        inside the environment.

    Notes
    -----
    **Speed.** At every time the speed follows a gamma distribution with mean
    ``speed_mean`` and standard deviation ``speed_std``: shape
    ``k = (speed_mean / speed_std)**2`` and scale
    ``speed_std**2 / speed_mean``. Its time course is a standard-normal OU
    process ``z`` with time constant ``coherence_time``, mapped through the
    gamma quantile function, ``speed = F⁻¹(Φ(z))`` (a Gaussian copula).
    RatInABox uses the same construction with a Rayleigh distribution; the
    gamma distribution lets the mean and spread be set separately and keeps
    the speed non-negative. The default spread (``speed_std = speed_mean / 2``)
    is close to the Rayleigh distribution's (standard deviation 0.52 times
    the mean).

    **Heading.** In 2-D the turning rate ``ω`` is an OU process with
    standard deviation ``rotational_velocity_std`` and time constant
    ``rotational_velocity_coherence_time``, as in RatInABox, and the heading
    advances each step by the integral of ``ω`` over the step, drawn jointly
    with ``ω`` from their exact Gaussian distribution (Gillespie 1996), so
    the heading decorrelates at the same rate at any ``dt``, including steps
    longer than the coherence time. In 1-D the animal keeps its direction
    until a wall reverses it. In three or more dimensions the heading diffuses
    on the sphere, decorrelating as ``exp(-t / τ)`` with
    ``τ = rotational_velocity_coherence_time``.

    Each OU process ``x`` is advanced exactly,
    ``x(t + dt) = x(t) e^(-dt/τ) + σ sqrt(1 - e^(-2 dt/τ)) ξ``, so its
    standard deviation is ``σ`` at any ``dt``.

    **Walls.** In 2-D the walls are the outline of the arena (or of the bins,
    when the arena has no known outline) and the arena's internal walls, so
    the animal never steps through a divider. A step that would cross a wall
    is mirrored about the first wall it meets, using that wall's normal; if
    the mirrored step crosses a wall too (a corner), it is mirrored again
    about that wall, up to 32 times, which turns it out of corners as sharp
    as about 6 degrees; an animal still blocked waits one step. An animal that
    starts on the outline (to within rounding) leaves it only toward the
    inside; one that starts on an internal wall may leave it to either side
    and does not cross it afterward. Elsewhere the
    walls are the faces of the bins: a step that would leave the bins
    reverses each velocity component whose move alone would leave them.

    References
    ----------
    George, T. M., de Cothi, W., Stachenfeld, K. L., & Barry, C. (2023).
    RatInABox, a toolkit for modelling locomotion and neuronal activity in
    continuous environments. eLife, 12, e85274.

    Gillespie, D. T. (1996). Exact numerical simulation of the
    Ornstein-Uhlenbeck process and its integral. Physical Review E, 54(2),
    2084-2091.

    Examples
    --------
    Generate 60 seconds of trajectory in a 2D arena:

    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import simulate_trajectory_ou
    >>> import numpy as np
    >>> from shapely.geometry import box
    >>>
    >>> env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")
    >>> positions, times = simulate_trajectory_ou(
    ...     env, duration=60.0, speed_units="cm", seed=0
    ... )
    >>> positions.shape  # (6000, 2) at 100 Hz
    (6000, 2)
    >>> bool(np.all(env.bin_at(positions) >= 0))
    True

    The speed varies around its mean:

    >>> speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / 0.01
    >>> bool(2.0 < speed.std() < 6.0)
    True

    Slower, steadier movement:

    >>> positions, times = simulate_trajectory_ou(
    ...     env,
    ...     duration=60.0,
    ...     speed_units="cm",
    ...     speed_mean=5.0,  # cm/s
    ...     speed_std=1.0,  # cm/s
    ...     coherence_time=1.0,  # seconds
    ...     seed=42,
    ... )

    See Also
    --------
    simulate_trajectory_laps : Structured back-and-forth movement
    simulate_trajectory_sinusoidal : Simple periodic motion (1D)
    """
    require_cartesian(
        env.coordinate_system,
        "simulate_trajectory_ou()",
        why="It moves the animal with a velocity in Cartesian coordinates and "
        "keeps it inside the environment's cells.",
        fix="simulate on the allocentric environment the animal moves in",
    )
    if env.units is None:
        msg = (
            "Environment units must be set before simulation. "
            "Pass units='cm' (or 'm', 'mm', etc.) when constructing the environment"
        )
        raise ValueError(msg)
    if speed_units != env.units:
        raise ValueError(
            _format_error(
                f"speed_units {speed_units!r} must match env.units {env.units!r}.",
                why="Why: speeds are not converted between units.",
                fix="convert speed_mean and speed_std into env.units and pass "
                f"speed_units={env.units!r}",
            )
        )
    if speed_mean is None:
        speed_mean = default_speed(
            env.units,
            _DEFAULT_OU_SPEED_CM_PER_S,
            name="speed_mean",
            call="simulate_trajectory_ou()",
            problem="not a metric length unit, so the default 8 cm/s",
            placeholder="running speed",
        )
    if speed_std is None:
        speed_std = speed_mean / 2
    _require_finite(
        [
            ("duration", duration, True),
            ("dt", dt, True),
            ("speed_mean", speed_mean, True),
            ("speed_std", speed_std, False),
            ("coherence_time", coherence_time, True),
            ("rotational_velocity_std", rotational_velocity_std, False),
            (
                "rotational_velocity_coherence_time",
                rotational_velocity_coherence_time,
                True,
            ),
        ]
    )
    modes = ("reflect", "periodic", "stop")
    if boundary_mode not in modes:
        raise ValueError(
            _format_error(
                f"boundary_mode must be one of {modes}, got {boundary_mode!r}.",
                why="Why: each mode is a different rule for what the animal "
                "does at a wall.",
                fix="pass boundary_mode='reflect' to bounce off walls",
            )
        )
    if boundary_mode == "periodic":
        _require_bins_fill_ranges(env)
    n_steps = _sample_count(duration, dt, "dt", "simulate_trajectory_ou()")
    rng = np.random.default_rng(seed)
    n_dims = env.n_dims
    times = np.arange(n_steps, dtype=np.float64) * dt

    if start_position is None:
        position = env.bin_centers[rng.integers(0, env.n_bins)].copy()
    else:
        position = np.asarray(start_position, dtype=np.float64).copy()

    walls = (
        Walls.enclosing(env) if n_dims == 2 and boundary_mode != "periodic" else None
    )
    if start_position is not None and not (
        position.shape == (n_dims,)
        and (
            env.contains(position[None])[0]
            # A start computed onto a slanted wall may round outside it.
            or (walls is not None and walls.distance(position) <= walls.tolerance)
        )
    ):
        raise ValueError(
            _format_error(
                f"start_position {start_position!r} is not a point inside the "
                "environment.",
                why=f"Why: the walk starts there, so it must be one point of "
                f"{n_dims} coordinates that a bin holds; from outside, the walls "
                "would keep the animal outside.",
                fix="pass a point where env.contains(start_position) is True, "
                "such as one of env.bin_centers, or None for a random bin center",
            )
        )

    speeds = _ou_speeds(n_steps, dt, speed_mean, speed_std, coherence_time, rng)
    if n_dims == 2:
        turns = _ou_step_integrals(
            n_steps,
            dt,
            rotational_velocity_std,
            rotational_velocity_coherence_time,
            rng,
        )
        direction = rng.uniform(-np.pi, np.pi)
        unit = np.array([np.cos(direction), np.sin(direction)])
    else:
        unit = rng.standard_normal(n_dims)
        unit /= np.linalg.norm(unit)
        if n_dims > 2:
            # Rotational diffusion on the sphere with coefficient D: heading
            # correlation decays as exp(-(n_dims - 1) D t), so D = 1 /
            # ((n_dims - 1) tau) gives time constant tau; each step adds
            # independent N(0, 2 D dt) to every component.
            heading_noise = rng.standard_normal((n_steps, n_dims)) * np.sqrt(
                2 * dt / ((n_dims - 1) * rotational_velocity_coherence_time)
            )

    positions = np.zeros((n_steps, n_dims), dtype=np.float64)
    positions[0] = position
    for i in range(1, n_steps):
        if n_dims == 2:
            angle = turns[i]
            cos, sin = np.cos(angle), np.sin(angle)
            unit = np.array(
                [cos * unit[0] - sin * unit[1], sin * unit[0] + cos * unit[1]]
            )
        elif n_dims > 2:
            unit = unit + heading_noise[i]
            unit /= np.linalg.norm(unit)
        step = speeds[i] * dt * unit

        if boundary_mode == "periodic":
            position = position + step
            for dim in range(n_dims):
                low, high = env.dimension_ranges[dim]  # type: ignore[index]
                position[dim] = low + (position[dim] - low) % (high - low)
        elif walls is not None:
            position, unit = walls.move(position, step, unit, boundary_mode)
        else:
            position, unit = _move_within_bins(env, position, step, unit, boundary_mode)
        positions[i] = position

    return positions, times


def _ou_series(
    n_steps: int,
    dt: float,
    std: float,
    coherence_time: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """A stationary OU process, shape (n_steps,), advanced exactly each step.

    ``x(t + dt) = a x(t) + std sqrt(1 - a²) ξ`` with ``a = exp(-dt /
    coherence_time)``, so the standard deviation is ``std`` at any ``dt``.
    """
    from scipy.signal import lfilter

    decay = np.exp(-dt / coherence_time)
    noise = rng.standard_normal(n_steps)
    noise[0] *= std
    noise[1:] *= std * np.sqrt(1.0 - decay**2)
    series: NDArray[np.float64] = lfilter([1.0], [1.0, -decay], noise)
    return series


def _ou_step_integrals(
    n_steps: int,
    dt: float,
    std: float,
    coherence_time: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Integral of a stationary OU process over each step, shape (n_steps,).

    Entry ``i`` is the integral of ``x`` from ``(i - 1) dt`` to ``i dt``;
    entry 0 is 0. Each step draws ``x`` and its integral jointly from their
    exact Gaussian distribution (Gillespie 1996): given ``x(t)``, the
    integral has mean ``τ (1 - a) x(t)`` with ``a = exp(-dt / τ)``, and its
    noise is correlated with the noise that advances ``x``.
    """
    from scipy.signal import lfilter

    h = dt / coherence_time
    decay = np.exp(-h)
    one_minus_decay = -np.expm1(-h)  # 1 - a
    one_minus_decay2 = -np.expm1(-2 * h)  # 1 - a²
    innovations = rng.standard_normal(n_steps)
    innovations[0] *= std
    innovations[1:] *= std * np.sqrt(one_minus_decay2)
    values = lfilter([1.0], [1.0, -decay], innovations)
    # Regression of the integral's noise on x's innovation, and what is left.
    slope = coherence_time * one_minus_decay**2 / one_minus_decay2
    residual_variance = (std * coherence_time) ** 2 * max(
        2 * h
        - 4 * one_minus_decay
        + one_minus_decay2
        - one_minus_decay**4 / one_minus_decay2,
        0.0,
    )
    integrals = np.zeros(n_steps)
    integrals[1:] = (
        coherence_time * one_minus_decay * values[:-1]
        + slope * innovations[1:]
        + np.sqrt(residual_variance) * rng.standard_normal(n_steps - 1)
    )
    return integrals


def _ou_speeds(
    n_steps: int,
    dt: float,
    speed_mean: float,
    speed_std: float,
    coherence_time: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Speeds with a gamma(mean, std) distribution and OU time course."""
    from scipy.stats import gamma, norm

    z = _ou_series(n_steps, dt, 1.0, coherence_time, rng)
    if speed_std == 0:
        return np.full(n_steps, speed_mean)
    shape = (speed_mean / speed_std) ** 2
    scale = speed_std**2 / speed_mean
    # norm.sf keeps the far upper tail representable, where norm.cdf rounds
    # to 1 and the quantile would be infinite.
    speeds: NDArray[np.float64] = gamma.isf(norm.sf(z), shape, scale=scale)
    return speeds


def _move_within_bins(
    env: Environment,
    position: NDArray[np.float64],
    step: NDArray[np.float64],
    unit: NDArray[np.float64],
    boundary_mode: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One step kept inside the bins, without wall geometry.

    A step that would leave the bins reverses each component whose move
    alone would leave them: the mirror image off a face of a grid cell.
    """
    proposed = position + step
    if env.contains(proposed[None, :])[0]:
        return proposed, unit
    if boundary_mode == "stop" and len(position) > 1:
        return position, unit
    single = position[None, :] + np.diag(step)
    leaving = ~env.contains(single)
    if not np.any(leaving):
        leaving[:] = True
    unit = np.where(leaving, -unit, unit)
    proposed = position + np.where(leaving, -step, step)
    if env.contains(proposed[None, :])[0]:
        return proposed, unit
    return position, unit


#: Default sinusoidal running speed, in centimeters per second.
_DEFAULT_SINUSOIDAL_SPEED_CM_PER_S = 10.0


def simulate_trajectory_sinusoidal(
    env: Environment,
    *,
    duration: float,
    sampling_frequency: float = 500.0,
    speed: float | None = None,
    period: float | None = None,
    pause_duration: float = 0.0,
    pause_at_peaks: bool = True,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate sinusoidal back-and-forth running along a track.

    Generates simple back-and-forth motion with optional pauses at track ends.
    Useful for testing place field detection on linear tracks.

    Parameters
    ----------
    env : Environment
        The 1D environment (must have env.is_linearized_track == True).
    duration : float
        Total simulation duration in seconds.
    sampling_frequency : float, optional
        Samples per second (default: 500 Hz).
    speed : float or None, optional
        Mean running speed along the track, in environment units per second.
        None (default) uses 10 cm/s expressed in ``env.units``; environments
        without metric units need an explicit value.
    period : float | None, optional
        Period of sinusoidal motion in seconds. If None, computed from
        track length and speed: period = 2 * track_length / speed.
    pause_duration : float, optional
        Pause at track ends in seconds (default: 0.0).
    pause_at_peaks : bool, optional
        Whether to pause at track extrema (default: True).
    seed : int | None, optional
        Accepted for a uniform trajectory API but has no effect: this trajectory
        is fully deterministic (a closed-form sinusoid with deterministic
        pauses), so the output is identical regardless of ``seed``. Provided so
        callers (e.g. ``simulate_session``) can pass ``seed`` uniformly across
        trajectory methods.

    Returns
    -------
    positions : NDArray[np.float64], shape (n_time, n_dims)
        Positions in the environment's coordinates (``env.n_dims`` of them),
        on the track's centerline, oscillating between its two far ends.
    times : NDArray[np.float64], shape (n_time,)
        Time points in seconds.

    Raises
    ------
    ValueError
        If environment is not 1D (env.is_linearized_track == False).
    ValueError
        If ``duration``, ``sampling_frequency``, ``speed`` or ``period`` is
        not finite and positive, ``pause_duration`` is not finite and
        non-negative, or ``duration`` is shorter than one sample.
    ValueError
        If ``speed`` is None and ``env.units`` is not a metric length.

    Notes
    -----
    The animal runs between the two ends of the track that are farthest
    apart along it: the two ends of a linear or bent track, or, on a
    branched track such as a T-maze, the stem's foot and the end of the
    farther arm. It never visits the other arms; use
    :func:`simulate_trajectory_laps` with ``outbound_path`` to choose the
    route.

    Examples
    --------
    Run an L-shaped track of 120 + 80 cm at 20 cm/s for two minutes:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import simulate_trajectory_sinusoidal
    >>> track = Environment.linear_track(
    ...     node_positions=[(0, 0), (120, 0), (120, 80)],
    ...     bin_size=2.0,
    ...     track_width=10.0,
    ...     units="cm",
    ... )
    >>> positions, times = simulate_trajectory_sinusoidal(
    ...     track, duration=120.0, speed=20.0
    ... )
    >>> positions.shape  # (x, y) at 500 Hz
    (60000, 2)
    >>> bool(np.all(track.bin_at(positions) >= 0))
    True
    >>> linear = track.to_linear(positions)
    >>> print(f"{linear.min():.0f} to {linear.max():.0f} cm along the track")
    0 to 200 cm along the track

    See Also
    --------
    simulate_trajectory_ou : Realistic random exploration
    simulate_trajectory_laps : Structured lap-based movement
    """
    if not env.is_linearized_track:
        msg = "Sinusoidal trajectory only works for 1D environments (env.is_linearized_track must be True)"
        raise ValueError(msg)
    call = "simulate_trajectory_sinusoidal()"
    if speed is None:
        speed = default_speed(
            env.units,
            _DEFAULT_SINUSOIDAL_SPEED_CM_PER_S,
            name="speed",
            call=call,
            problem="not a metric length unit, so the default 10 cm/s",
            placeholder="running speed",
        )
    _require_finite(
        [
            ("duration", duration, True),
            ("sampling_frequency", sampling_frequency, True),
            ("speed", speed, True),
            *([] if period is None else [("period", period, True)]),
            ("pause_duration", pause_duration, False),
        ]
    )

    # The motion runs along the track between its two far ends, in the
    # environment's coordinates.
    waypoints = _track_end_to_end(env)
    waypoint_distances = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))]
    )
    track_length = float(waypoint_distances[-1])

    # Compute period if not provided
    if period is None:
        period = 2 * track_length / speed

    # Time setup
    dt = 1.0 / sampling_frequency
    n_steps = _sample_count(duration, dt, "1 / sampling_frequency", call)
    times = np.arange(n_steps) * dt

    # Distance along the track: d(t) = L/2 + L/2 sin(2π t / period)
    angular_freq = 2 * np.pi / period
    distance = track_length / 2 * (1 + np.sin(angular_freq * times))

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
        last_valid = distance[0]
        for i in range(n_steps):
            if paused[i]:
                distance[i] = last_valid
            else:
                last_valid = distance[i]

    positions = np.column_stack(
        [
            np.interp(distance, waypoint_distances, waypoints[:, dim])
            for dim in range(env.n_dims)
        ]
    )
    return positions, times


def _farthest_node(graph: nx.Graph, source: object, weight: object) -> object:
    """The node farthest from ``source`` along ``graph``, first in node order on ties."""
    lengths = nx.single_source_dijkstra_path_length(graph, source, weight=weight)
    farthest = max(lengths.values())
    return next(
        node
        for node in graph.nodes
        if node in lengths and lengths[node] >= farthest * (1 - 1e-9)
    )


def _farthest_pair(
    graph: nx.Graph, source: object, weight: object
) -> tuple[object, object]:
    """Two nodes far apart along ``graph``, by two sweeps of shortest paths.

    The node farthest from ``source``, then the node farthest from that one.
    On a graph without loops (a track) they are the two ends of its longest
    path; elsewhere they are close to it.
    """
    first = _farthest_node(graph, source, weight)
    return first, _farthest_node(graph, first, weight)


def _track_end_to_end(env: Environment) -> NDArray[np.float64]:
    """The track nodes on the longest route along a track, shape (n, n_dims).

    The route starts at the end with the smaller linear position.
    """
    linearization = env.binning.linearization
    assert linearization is not None
    graph = linearization.graph.edge_subgraph(linearization.edge_order)
    node_positions = {
        node: np.asarray(position, dtype=np.float64)
        for node, position in graph.nodes(data="pos")
    }

    def length(u: object, v: object, _data: object) -> float:
        return float(np.linalg.norm(node_positions[u] - node_positions[v]))

    start, end = _farthest_pair(graph, next(iter(graph.nodes)), length)
    route = nx.shortest_path(graph, start, end, weight=length)
    waypoints = np.array([node_positions[node] for node in route])
    if env.to_linear(waypoints[[0]])[0] > env.to_linear(waypoints[[-1]])[0]:
        waypoints = waypoints[::-1]
    return waypoints


#: Default lap running speed, in centimeters per second.
_DEFAULT_LAP_SPEED_CM_PER_S = 20.0


def simulate_trajectory_laps(
    env: Environment,
    *,
    n_laps: int,
    speed_mean: float | None = None,
    speed_std: float | None = None,
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
        Number of one-way traversals, alternating outbound and inbound,
        starting outbound. Two traversals make one out-and-back cycle.
    speed_mean : float or None, optional
        Mean running speed in position units per second. Each traversal runs
        at one speed drawn from a normal distribution with this mean, floored
        at ``0.1 * speed_mean``. None (default) uses 20 cm/s, a typical rodent
        running speed on a track, expressed in ``env.units``; environments
        without metric units need an explicit value.
    speed_std : float or None, optional
        Standard deviation of the per-traversal speed, in position units per
        second. None (default) uses ``0.2 * speed_mean``.
    outbound_path : list[int] | None, optional
        Bin indices for the outbound traversal, a chain of neighboring bins:
        every consecutive pair must be connected in ``env.connectivity`` or
        repeat the same bin (a step of zero length). A single bin is also
        valid. If None, uses the shortest path between the bins with the
        smallest and largest first coordinate.
    inbound_path : list[int] | None, optional
        Bin indices for the inbound traversal, with the same rule as
        ``outbound_path``. If None, reverses outbound_path.
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
        Position coordinates, in ``env.units``. Positions move at constant
        speed from bin center to bin center along the path, and hold still
        during pauses. Every position lies inside the environment (see
        Notes).
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
        If ``speed_mean`` is None and ``env.units`` is not a metric length.
    ValueError
        If custom paths are empty, contain invalid bin indices or are not
        chains of neighboring bins, or if no connected path joins the bins
        with the smallest and largest first coordinate.
    ValueError
        If a path steps between neighboring bins whose cells do not touch
        inside the environment, such as a bridge between two rooms of an
        ``Environment.combine()`` environment with no bins between them:
        every route between them would leave the environment.

    Notes
    -----
    Between two neighboring bins the animal follows the environment, not the
    straight line between the bin centers, which can leave it:

    - On a track (``env.is_linearized_track``), it runs along the centerline
      and turns at the node where two track segments meet, so its distance
      along the track (``env.to_linear``) changes at the running speed.
    - Between planar cells (grids, hexagons, triangles, polygon arenas,
      clipped or wall-split cells), it takes the shortest route between the
      two centers inside the two cells, kept inside the arena outline and
      off its walls. Cells that touch only at a corner outside the arena are
      joined through the cells of the bins neighboring both.
    - In other environments, such as 3-D grids, it runs straight between the
      centers of the neighboring cells.

    Examples
    --------
    Generate laps in a 2D environment:

    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import simulate_trajectory_laps
    >>> import numpy as np
    >>> from shapely.geometry import box
    >>>
    >>> # Create a 100 x 100 cm arena
    >>> env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0)
    >>> env = env.with_metadata(units="cm")
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

    Three traversals of a 1 m track take about as long as for a running rat:

    >>> track = Environment.from_polygon(box(0, 0, 100, 20), bin_size=2, units="cm")
    >>> positions, times, metadata = simulate_trajectory_laps(
    ...     track, n_laps=3, seed=1, return_metadata=True
    ... )
    >>> print(f"{times[-1]:.1f} s")
    14.4 s
    >>> moving = (metadata["lap_ids"][1:] == metadata["lap_ids"][:-1]) & np.any(
    ...     np.diff(positions, axis=0) != 0, axis=1
    ... )
    >>> speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)
    >>> print(f"{np.median(speed[moving]):.0f} cm/s")
    21 cm/s

    On an L-shaped track the lap turns at the corner instead of cutting
    across it, so every position is on the track:

    >>> l_track = Environment.linear_track(
    ...     node_positions=[(-5, 0), (5, 0), (5, 10)],
    ...     bin_size=10,
    ...     track_width=2,
    ...     units="cm",
    ... )
    >>> positions, times = simulate_trajectory_laps(l_track, n_laps=1, seed=0)
    >>> bool(np.all(l_track.bin_at(positions) >= 0))
    True
    >>> print(f"{l_track.to_linear(positions[[0, -1]])} cm along the track")
    [ 5. 15.] cm along the track

    See Also
    --------
    simulate_trajectory_ou : Realistic random exploration
    simulate_trajectory_sinusoidal : Simple periodic motion (1D)
    simulate_session : Fit a requested number of traversals into a session duration
    """
    return _simulate_trajectory_laps(
        env,
        n_laps,
        speed_mean=speed_mean,
        speed_std=speed_std,
        outbound_path=outbound_path,
        inbound_path=inbound_path,
        pause_duration=pause_duration,
        sampling_frequency=sampling_frequency,
        seed=seed,
        return_metadata=return_metadata,
        call="simulate_trajectory_laps()",
    )


def _simulate_trajectory_laps(
    env: Environment,
    n_laps: int,
    *,
    speed_mean: float | None = None,
    speed_std: float | None = None,
    outbound_path: list[int] | None = None,
    inbound_path: list[int] | None = None,
    pause_duration: float = 0.5,
    sampling_frequency: float = 500.0,
    seed: int | None = None,
    return_metadata: bool = False,
    duration: float | None = None,
    call: str,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64]]
    | tuple[NDArray[np.float64], NDArray[np.float64], dict]
):
    """Shared lap engine; sessions allocate samples before generating positions.

    With ``duration``, speed draws only weight the traversal durations, so the
    default mean is a unitless 1.0 instead of 20 cm/s in ``env.units``.
    ``call`` is the public call the scientist made, named in path errors.
    """
    if speed_mean is None:
        if duration is not None:
            speed_mean = 1.0
        else:
            speed_mean = default_speed(
                env.units,
                _DEFAULT_LAP_SPEED_CM_PER_S,
                name="speed_mean",
                call=call,
                problem="not a metric length unit, so the default 20 cm/s",
                placeholder="running speed",
            )
    if speed_std is None:
        speed_std = 0.2 * speed_mean
    if duration is not None:
        for name, value, positive in [
            ("duration", duration, True),
            ("sampling_frequency", sampling_frequency, True),
            ("speed_mean", speed_mean, True),
            ("speed_std", speed_std, False),
            ("pause_duration", pause_duration, False),
        ]:
            if not np.isfinite(value) or (value <= 0 if positive else value < 0):
                bound = "positive" if positive else "non-negative"
                raise ValueError(
                    _format_error(
                        f"{name} must be finite and {bound}, got {value}.",
                        why="Why: Lap session timing requires finite sample budgets and speeds.",
                        fix=f"Set {name} to a finite {bound} value.",
                    )
                )
        if (
            not isinstance(n_laps, (int, np.integer))
            or isinstance(n_laps, bool)
            or n_laps <= 0
        ):
            raise ValueError(
                _format_error(
                    f"n_laps must be a positive integer, got {n_laps}.",
                    why="Why: Each requested one-way traversal needs its own sample allocation.",
                    fix="Set n_laps=2 for one out-and-back cycle.",
                )
            )

    # Validate parameters
    if n_laps <= 0:
        msg = f"n_laps must be positive (got {n_laps})"
        raise ValueError(msg)

    if speed_mean <= 0:
        msg = f"speed_mean must be positive (env units / s, got {speed_mean})"
        raise ValueError(msg)

    if speed_std < 0:
        msg = f"speed_std must be non-negative (env units / s, got {speed_std})"
        raise ValueError(msg)

    if pause_duration < 0:
        msg = f"pause_duration must be non-negative (seconds, got {pause_duration})"
        raise ValueError(msg)

    if sampling_frequency <= 0:
        msg = f"sampling_frequency must be positive (Hz, got {sampling_frequency})"
        raise ValueError(msg)

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Name the paths in errors as the scientist knows them.
    path_names = (
        "outbound_path"
        if outbound_path is not None
        else "the path between the bins at the smallest and largest x",
        "inbound_path" if inbound_path is not None else "the reversed outbound path",
    )

    # Compute paths if not provided
    if outbound_path is None:
        # Use shortest path between environment extrema
        # Find bins at minimum and maximum positions in first dimension
        bin_positions = env.bin_centers
        first_dim_positions = bin_positions[:, 0]
        start_bin = int(np.argmin(first_dim_positions))
        end_bin = int(np.argmax(first_dim_positions))

        try:
            outbound_path = nx.shortest_path(env.connectivity, start_bin, end_bin)
        except nx.NetworkXNoPath:
            raise ValueError(
                _format_error(
                    f"{call} found no connected path between "
                    f"the bins at the smallest and largest x ({start_bin} and "
                    f"{end_bin}).",
                    why="Why: the animal runs along neighboring bins, and these "
                    "bins lie in disconnected parts of the environment.",
                    fix="pass outbound_path= as a chain of neighboring bins",
                )
            ) from None

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
        # A repeated bin is a zero-length step, so it needs no edge.
        for a, b in pairwise(path):
            if a != b and not env.connectivity.has_edge(a, b):
                raise ValueError(
                    _format_error(
                        f"{call}: {path_name} steps from bin "
                        f"{a} to bin {b}, which are not neighboring bins.",
                        why="Why: the animal runs from each bin into the next "
                        "through the space they share, which only neighboring "
                        "bins have.",
                        fix=f"replace the step with "
                        f"nx.shortest_path(env.connectivity, {a}, {b})",
                    )
                )

    # Each step between neighboring bins follows the environment's geometry
    # (along a track's centerline, inside the cells otherwise), not the
    # straight line between their centers, which can leave the environment.
    router = LapRouter(env, call)
    routes = [
        router.route(path, name, argument)
        for path, name, argument in zip(
            (outbound_path, inbound_path),
            path_names,
            ("outbound_path", "inbound_path"),
            strict=True,
        )
    ]
    # Distance run from the start of each route to each of its waypoints.
    route_distances = [
        np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
        )
        for route in routes
    ]

    # Time step
    dt = 1.0 / sampling_frequency

    lap_sample_counts = None
    if duration is not None:
        # Allocate the requested clock first, keeping fixed pauses and at least
        # two samples per traversal so both path endpoints are represented.
        total_samples = int(np.ceil(duration * sampling_frequency))
        if (total_samples - 1) / sampling_frequency >= duration:
            total_samples -= 1
        pause_samples = int(pause_duration * sampling_frequency)
        minimum_samples = 2 * n_laps + pause_samples * (n_laps - 1)
        if total_samples < minimum_samples:
            raise ValueError(
                _format_error(
                    f"duration={duration} s cannot fit n_laps={n_laps} traversals "
                    f"with pause_duration={pause_duration} s at "
                    f"sampling_frequency={sampling_frequency} Hz.",
                    why=f"Why: Fixed pauses and two endpoints per traversal require {minimum_samples} samples.",
                    fix=f"Increase duration to at least {minimum_samples / sampling_frequency:g} seconds, "
                    "or reduce n_laps or pause_duration.",
                )
            )

        # Speed draws shape relative traversal durations. Absolute speeds follow
        # from the available time; no speed-driven trajectory is generated and
        # then rescaled or truncated.
        distances_by_direction = [distances[-1] for distances in route_distances]
        weights = np.array(
            [
                distances_by_direction[lap_idx % 2]
                / max(rng.normal(speed_mean, speed_std), 0.1 * speed_mean)
                for lap_idx in range(n_laps)
            ]
        )
        if not np.any(weights):
            weights = np.ones(n_laps)
        extra_samples = total_samples - minimum_samples
        shares = extra_samples * (weights / weights.sum())
        extras = np.floor(shares).astype(np.int_)
        remaining = extra_samples - int(extras.sum())
        order = np.argsort(-(shares - extras), kind="stable")
        extras[order[:remaining]] += 1
        lap_sample_counts = extras + 2

    # Build trajectory by concatenating laps
    all_positions = []
    all_times = []
    all_lap_ids = []
    all_directions = []
    lap_boundaries_list = [0]  # Start of first lap (will convert to array later)
    current_time = 0.0

    for lap_idx in range(n_laps):
        # Alternate between outbound and inbound
        direction = "outbound" if lap_idx % 2 == 0 else "inbound"

        # Generate positions along the path's route
        path_positions = routes[lap_idx % 2]
        cumulative_distances = route_distances[lap_idx % 2]
        # The last waypoint's own distance, so the lap ends exactly on it.
        total_distance = cumulative_distances[-1]

        if lap_sample_counts is None:
            # Direct calls retain their speed-driven timing and RNG draws.
            speed = rng.normal(speed_mean, speed_std)
            speed = max(speed, 0.1 * speed_mean)
            path_duration = total_distance / speed
            n_samples = max(2, int(path_duration / dt))
        else:
            n_samples = int(lap_sample_counts[lap_idx])

        # Interpolate positions along the route
        interpolation_points = np.linspace(0, total_distance, n_samples)

        # Interpolate each dimension separately
        lap_positions = np.zeros((n_samples, env.n_dims))
        for dim in range(env.n_dims):
            lap_positions[:, dim] = np.interp(
                interpolation_points, cumulative_distances, path_positions[:, dim]
            )

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
    if duration is not None:
        # One clock avoids accumulated rounding at lap/pause boundaries.
        times = np.arange(len(positions), dtype=np.float64) / sampling_frequency
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
