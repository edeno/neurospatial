"""Simulate candidate replay events with a known path along a track.

Replay methods can be compared on simulated events whose content is known.
Tingley & Peyrache (2020) degraded simulated sequences by adding noise
spikes and removing real ones to see how each method's score falls.
:func:`simulate_replay_event` moves a position along a track and draws spikes
from place-cell rate maps at that position, so each method can be scored
against the truth: a sweep along the track should be detected, a stationary
or scattered event should not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin

if TYPE_CHECKING:
    from neurospatial.encoding.spatial import SpatialRatesResult
    from neurospatial.environment import Environment

_KINDS = ("sweep", "stationary", "jump", "random")
_OPERATION = "simulate_replay_event()"
# Resolution of the simulated path and spike draws, in seconds.
_SAMPLE_INTERVAL = 0.001


@dataclass(frozen=True, repr=False, eq=False)
class ReplayEventSimulation(ReadOnlyArraysMixin, ResultMixin):
    """A simulated candidate event and the path it replays.

    Attributes
    ----------
    spike_times : tuple of NDArray[np.float64]
        Spike times of each cell in seconds, from ``0`` to ``duration``, in the
        order of the rate maps' rows.
    times : NDArray[np.float64], shape (n_samples,)
        Times at which the path is sampled, every 1 ms.
    track_positions : NDArray[np.float64], shape (n_samples,)
        Position along the track at each time, in environment units.
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        The same positions in environment coordinates.
    kind : str
        ``"sweep"``, ``"stationary"``, ``"jump"`` or ``"random"``.
    duration : float
        Event duration in seconds.
    speed : float
        Speed of the sweep along the track, in environment units per second
        (negative toward the track start); NaN for events without a constant
        speed.
    drop_fraction : float
        Fraction of the rate-map spikes asked to be removed.
    n_dropped_spikes : int
        Rate-map spikes removed from the event.
    n_noise_spikes : int
        Noise spikes added to the event, included in ``spike_times``.

    See Also
    --------
    simulate_replay_event : Function that produces this result.
    """

    spike_times: tuple[NDArray[np.float64], ...]
    times: NDArray[np.float64]
    track_positions: NDArray[np.float64]
    positions: NDArray[np.float64]
    kind: str
    duration: float
    speed: float
    drop_fraction: float
    n_dropped_spikes: int
    n_noise_spikes: int

    def __post_init__(self) -> None:
        """Own read-only copies of the arrays, including each spike train."""
        trains = []
        for train in self.spike_times:
            owned = np.array(train, dtype=np.float64)
            owned.flags.writeable = False
            trains.append(owned)
        object.__setattr__(self, "spike_times", tuple(trains))
        self._own_arrays()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from pickle or deepcopy, keeping every array read-only."""
        super().__setstate__(state)
        for train in self.spike_times:
            train.flags.writeable = False

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "kind": self.kind,
            "duration": self.duration,
            "speed": self.speed,
            "drop_fraction": self.drop_fraction,
            "n_dropped_spikes": self.n_dropped_spikes,
            "n_noise_spikes": self.n_noise_spikes,
            "n_spikes": int(sum(train.size for train in self.spike_times)),
            "n_active_cells": int(sum(train.size > 0 for train in self.spike_times)),
        }


def simulate_replay_event(
    env: Environment,
    rates: NDArray[np.float64] | SpatialRatesResult,
    *,
    kind: Literal["sweep", "stationary", "jump", "random"] = "sweep",
    duration: float = 0.2,
    speed: float | None = None,
    start_position: float | None = None,
    jump_distance: float | None = None,
    rate_gain: float = 1.0,
    random_interval: float = 0.02,
    drop_fraction: float = 0.0,
    n_noise_spikes: int = 0,
    seed: int | None = None,
) -> ReplayEventSimulation:
    """Simulate a candidate replay event with a known path along a track.

    A position moves along the track and each cell fires as an
    inhomogeneous Poisson process at ``rate_gain`` times its rate map at that
    position (with a 2 ms refractory period). To degrade the event, a
    fraction of these spikes can be removed and noise spikes added, as
    Tingley & Peyrache (2020, Fig. 4) did to compare how replay methods
    hold up as the signal-to-noise ratio falls.

    Parameters
    ----------
    env : Environment
        A 1-D environment or a linearized track.
    rates : array of shape (n_cells, n_bins), or SpatialRatesResult
        Each cell's rate map on ``env``, in Hz, with a finite rate in every
        bin.
    kind : {"sweep", "stationary", "jump", "random"}, default="sweep"
        The event's content:

        - ``"sweep"``: constant ``speed`` along the track from
          ``start_position``, as replay is usually modeled.
        - ``"stationary"``: a fixed position, which Davidson et al.'s line
          search (``detect_trajectory_radon``) scores well with a flat line,
          but the weighted correlation does not (van der Meer et al. 2020),
          nor the R² of ``fit_linear_trajectory`` (NaN for a fixed MAP).
        - ``"jump"``: a sweep at ``speed`` whose second half continues
          ``jump_distance`` further along the track: two fragments with one
          jump between them. Whether it counts as replay depends on the
          method: a line fit can still score it, while a maximum-jump
          criterion rejects it when the jump exceeds its threshold.
        - ``"random"``: a new position drawn uniformly along the track every
          ``random_interval``, with no sequence: a null event.
    duration : float, default=0.2
        Event duration in seconds.
    speed : float or None
        Sweep speed in environment units per second, negative toward the
        track start. Required for ``"sweep"`` and ``"jump"``.
    start_position : float or None
        Position along the track at the start, in the track's own coordinate
        and environment units, inside a bin. Default: the track start for a
        forward sweep, the track end for a backward sweep, and the middle of
        the track's length for a stationary event.
    jump_distance : float or None
        Size of the jump of a ``"jump"`` event, in environment units. Default:
        a quarter of the track's length.
    rate_gain : float, default=1.0
        Factor applied to every rate. Cells fire faster during sharp-wave
        ripples than during running; raise it to model that. The 2 ms
        refractory period caps the realized rate at ``r / (1 + 0.002 r)``.
    random_interval : float, default=0.02
        How long each position of a ``"random"`` event lasts, in seconds.
    drop_fraction : float, default=0.0
        Fraction of the event's rate-map spikes to remove, from 0 to 1. The
        nearest whole number of spikes (half rounds up) is removed, chosen
        uniformly at random without replacement from all cells' spikes
        together, as Tingley & Peyrache (2020) removed "real" spikes.
    n_noise_spikes : int, default=0
        Number of noise spikes to add after the removal. Each goes to a cell
        drawn uniformly from all cells, at a time drawn uniformly over the
        event, regardless of the path, the rate maps or the refractory
        period, as Tingley & Peyrache (2020) inserted "noise" spikes. A
        background rate of ``r`` Hz per cell is about
        ``r * n_cells * duration`` spikes.
    seed : int or None
        Seed for the path, the spikes, the removal and the noise spikes,
        which draw from separate streams: changing ``drop_fraction`` does not
        move the noise spikes, and neither changes the rate-map spikes
        before removal.

    Notes
    -----
    Tingley & Peyrache (2020) degraded a perfect event, 100 cells firing one
    spike each in field order, by removing up to 90 spikes and adding up to
    90 (``compareNoiseDegradation.m``, on a 1 ms grid, where a noise spike
    on an occupied millisecond of a cell is lost). Here noise spikes take
    continuous times, so every one is added. Sweep ``drop_fraction`` and
    ``n_noise_spikes`` over a grid, with many seeds per setting, to draw a
    method's detection rate against both, as in their Fig. 4.

    The path moves along the track's bins in track order, measured without
    the gaps a linearized track leaves between segments, so it never enters a
    gap. On a maze with several arms the track order joins arm ends, so a path
    that reaches the end of one arm continues on the next; keep a sweep within
    one arm, or use a single-arm track, to avoid that jump.

    Returns
    -------
    ReplayEventSimulation
        Spike times of each cell and the path the event replays.

    Raises
    ------
    ValueError
        If ``env`` is neither 1-D nor a linearized track, if ``rates`` does not
        have one finite, non-negative row per cell and one column per bin, if
        ``speed`` is missing for a sweep, if a sweep leaves the track, or if
        ``duration``, ``rate_gain`` or ``random_interval`` is not positive,
        if ``drop_fraction`` is not a number in ``[0, 1]`` (a bool is
        refused), or if ``n_noise_spikes`` is not a non-negative integer.

    See Also
    --------
    neurospatial.simulation.generate_poisson_spikes : The spike generator.
    neurospatial.replay.replay_significance : Test an event's replay score.

    References
    ----------
    Tingley, D., & Peyrache, A. (2020). On the methods for reactivation and
    replay analysis. *Philosophical Transactions of the Royal Society B*,
    375(1799), 20190231.

    van der Meer, M. A. A., Kemere, C., & Diba, K. (2020). Progress and issues
    in second-order analysis of hippocampal replay. *Philosophical
    Transactions of the Royal Society B*, 375(1799), 20190238.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import simulate_replay_event
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [200.0, 0.0]], bin_size=4.0, units="cm"
    ... )
    >>> x = env.bin_centers[:, 0]
    >>> rates = 0.5 + 25.0 * np.exp(
    ...     -0.5 * ((x - np.linspace(0, 200, 30)[:, None]) / 12.0) ** 2
    ... )
    >>> event = simulate_replay_event(env, rates, speed=800.0, rate_gain=3.0, seed=0)
    >>> event.track_positions[[0, -1]].round()
    array([  0., 159.])
    >>> event.summary()["n_active_cells"] > 15
    True

    The same event with half its spikes removed (rounding half up) and 40
    noise spikes added. With the same seed, the remaining rate-map spikes
    are a subset of the event's:

    >>> noisy = simulate_replay_event(
    ...     env,
    ...     rates,
    ...     speed=800.0,
    ...     rate_gain=3.0,
    ...     drop_fraction=0.5,
    ...     n_noise_spikes=40,
    ...     seed=0,
    ... )
    >>> n_spikes = event.summary()["n_spikes"]
    >>> noisy.n_dropped_spikes == (n_spikes + 1) // 2
    True
    >>> noisy.summary()["n_spikes"] == n_spikes - noisy.n_dropped_spikes + 40
    True
    """
    from neurospatial.environment import _require_track_extents
    from neurospatial.simulation.spikes import generate_poisson_spikes

    if kind not in _KINDS:
        raise ValueError(f"{_OPERATION}: kind must be one of {_KINDS}, got {kind!r}.")
    for name, value in (
        ("duration", duration),
        ("rate_gain", rate_gain),
        ("random_interval", random_interval),
    ):
        if not (np.isfinite(value) and value > 0):
            raise ValueError(f"{_OPERATION}: {name} must be positive, got {value!r}.")
    if (
        isinstance(drop_fraction, bool)
        or not isinstance(drop_fraction, (int, float, np.integer, np.floating))
        or not 0.0 <= drop_fraction <= 1.0
    ):
        raise ValueError(
            f"{_OPERATION}: drop_fraction must be a fraction from 0 to 1 of the "
            f"event's spikes, got {drop_fraction!r}."
        )
    if (
        isinstance(n_noise_spikes, bool)
        or not isinstance(n_noise_spikes, (int, np.integer))
        or n_noise_spikes < 0
    ):
        raise ValueError(
            f"{_OPERATION}: n_noise_spikes must be a non-negative integer, got "
            f"{n_noise_spikes!r}."
        )
    starts, stops = _require_track_extents(
        env,
        _OPERATION,
        alternative="simulate a 2-D event by sampling positions yourself",
    )
    track = _Track(np.asarray(starts), np.asarray(stops))
    maps = _validated_rates(rates, env.n_bins)

    seeds = np.random.SeedSequence(seed)
    # Spawned in this order, the path and spike streams do not depend on
    # the degradation settings.
    path_seed, spike_seed, drop_seed, noise_seed = seeds.spawn(4)
    times: NDArray[np.float64] = np.arange(
        0.0, duration, _SAMPLE_INTERVAL, dtype=np.float64
    )
    distance, event_speed = _path(
        kind,
        times,
        duration,
        speed,
        None if start_position is None else track.distance_of(start_position),
        jump_distance,
        track.length,
        random_interval,
        np.random.default_rng(path_seed),
    )
    bins, track_positions = track.locate(distance)
    # The spike generator treats its last time as the end of the last
    # interval, so the grid ends at ``duration`` to cover the whole event.
    spike_grid = np.append(times, duration)
    cell_seeds = spike_seed.generate_state(len(maps), dtype=np.uint32)
    spike_times = tuple(
        generate_poisson_spikes(
            rate_gain * np.append(cell[bins], cell[bins[-1]]),
            spike_grid,
            seed=int(cell_seed),
        )
        for cell, cell_seed in zip(maps, cell_seeds, strict=True)
    )
    spike_times, n_dropped = _drop_spikes(
        spike_times, drop_fraction, np.random.default_rng(drop_seed)
    )
    spike_times = _add_noise_spikes(
        spike_times, int(n_noise_spikes), duration, np.random.default_rng(noise_seed)
    )
    positions = (
        env.linear_to_nd(track_positions)
        if env.is_linearized_track
        else track_positions[:, np.newaxis]
    )
    return ReplayEventSimulation(
        spike_times=spike_times,
        times=times,
        track_positions=track_positions,
        positions=np.asarray(positions, dtype=np.float64),
        kind=kind,
        duration=float(duration),
        speed=event_speed,
        drop_fraction=float(drop_fraction),
        n_dropped_spikes=n_dropped,
        n_noise_spikes=int(n_noise_spikes),
    )


def _drop_spikes(
    spike_times: tuple[NDArray[np.float64], ...],
    fraction: float,
    rng: np.random.Generator,
) -> tuple[tuple[NDArray[np.float64], ...], int]:
    """Remove a fraction of all cells' spikes, chosen without replacement."""
    sizes = [train.size for train in spike_times]
    n_spikes = sum(sizes)
    n_dropped = int(np.floor(fraction * n_spikes + 0.5))
    if n_dropped == 0:
        return spike_times, 0
    keep = np.ones(n_spikes, dtype=bool)
    keep[rng.choice(n_spikes, size=n_dropped, replace=False)] = False
    pieces = np.split(keep, np.cumsum(sizes)[:-1])
    return (
        tuple(train[kept] for train, kept in zip(spike_times, pieces, strict=True)),
        n_dropped,
    )


def _add_noise_spikes(
    spike_times: tuple[NDArray[np.float64], ...],
    n_noise: int,
    duration: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], ...]:
    """Add spikes to uniformly drawn cells at uniform times in the event."""
    if n_noise == 0:
        return spike_times
    cells = rng.integers(len(spike_times), size=n_noise)
    times = rng.uniform(0.0, duration, size=n_noise)
    return tuple(
        np.sort(np.concatenate([train, times[cells == cell]]))
        for cell, train in enumerate(spike_times)
    )


def _validated_rates(rates: Any, n_bins: int) -> NDArray[np.float64]:
    maps = np.asarray(getattr(rates, "firing_rates", rates), dtype=np.float64)
    if maps.ndim != 2 or maps.shape[1] != n_bins:
        raise ValueError(
            f"{_OPERATION}: rates must have shape (n_cells, {n_bins}), one rate "
            f"map per cell on env, got {maps.shape}."
        )
    if not np.isfinite(maps).all() or np.any(maps < 0):
        raise ValueError(
            _format_error(
                f"{_OPERATION}: rates must be finite and non-negative in every bin.",
                why="each simulated spike is drawn from a cell's rate at the "
                "replayed position, which can be any bin of the track",
                fix="use rate maps with full coverage of the track, or build "
                "model rate maps (for example PlaceCellModel) on env",
            )
        )
    return maps


class _Track:
    """A track's bins in track order, without the gaps between segments.

    The path moves along ``distance``, from 0 to the summed bin lengths, so it
    never enters the gap that a linearized track leaves between segments.
    """

    def __init__(self, starts: NDArray[np.float64], stops: NDArray[np.float64]):
        self.order = np.argsort(starts, kind="stable")
        self.starts = starts[self.order]
        lengths = stops[self.order] - self.starts
        self.edges = np.concatenate([[0.0], np.cumsum(lengths)])
        self.length = float(self.edges[-1])

    def locate(
        self, distance: NDArray[np.float64]
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        """Bin and track position of each distance along the track."""
        k = np.clip(
            np.searchsorted(self.edges, distance, side="right") - 1,
            0,
            len(self.starts) - 1,
        )
        return self.order[k], self.starts[k] + (distance - self.edges[k])

    def distance_of(self, position: float) -> float:
        """Distance along the track of a track position inside a bin."""
        k = int(np.searchsorted(self.starts, position, side="right") - 1)
        lengths = np.diff(self.edges)
        if k < 0 or position >= self.starts[k] + lengths[k]:
            raise ValueError(
                _format_error(
                    f"{_OPERATION}: start_position={position:g} is not inside "
                    "any bin of the track.",
                    why="a linearized track can leave gaps between its "
                    "segments, and positions there belong to no bin",
                    fix="choose a start_position inside a segment of the track",
                )
            )
        return float(self.edges[k] + (position - self.starts[k]))


def _path(
    kind: str,
    times: NDArray[np.float64],
    duration: float,
    speed: float | None,
    start: float | None,
    jump_distance: float | None,
    length: float,
    random_interval: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], float]:
    """Distance along the track at each time, and the constant speed (or NaN)."""
    if kind == "random":
        n_positions = int(np.ceil(duration / random_interval))
        draws = rng.uniform(0.0, length, n_positions)
        segment = np.minimum((times / random_interval).astype(int), n_positions - 1)
        return draws[segment], float("nan")
    if kind == "stationary":
        position = length / 2 if start is None else start
        return np.full(times.shape, float(position)), 0.0
    if speed is None or not np.isfinite(speed) or speed == 0:
        raise ValueError(
            _format_error(
                f"{_OPERATION}: kind={kind!r} needs a nonzero speed, got {speed!r}.",
                why="a sweep moves along the track at this speed",
                fix="pass speed= in environment units per second, e.g. "
                "speed=800.0 for 8 m/s on a track in cm",
            )
        )
    if start is None:
        start = 0.0 if speed > 0 else length - 1e-9
    distance = start + speed * times
    if kind == "jump":
        jump = length / 4 if jump_distance is None else float(jump_distance)
        if not (np.isfinite(jump) and jump > 0):
            raise ValueError(
                f"{_OPERATION}: jump_distance must be positive, got {jump_distance!r}."
            )
        distance = np.where(
            times >= duration / 2, distance + np.sign(speed) * jump, distance
        )
    if np.any(distance < 0) or np.any(distance >= length):
        raise ValueError(
            _format_error(
                f"{_OPERATION}: the event leaves the track (it would travel to "
                f"{distance.min():.4g}..{distance.max():.4g} along a track "
                f"{length:.4g} long).",
                why="spikes are drawn from the rate maps at positions on the track",
                fix="shorten duration, lower speed, shorten jump_distance, or "
                "move start_position",
            )
        )
    return distance, float(speed)
