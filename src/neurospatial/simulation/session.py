"""High-level session simulation and dataclass containers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import read_only_copy

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.simulation.models import NeuralModel


def _frozen(value: Any) -> Any:
    """A read-only equivalent of ``value``: proxied mappings, tuples, read-only arrays."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_frozen(item) for item in value)
    if isinstance(value, np.ndarray):
        return read_only_copy(value)
    return value


def _thawed(value: Any) -> Any:
    """Plain dictionaries in place of the read-only mappings of ``_frozen``."""
    if isinstance(value, Mapping):
        return {key: _thawed(item) for key, item in value.items()}
    return value


def _rebuild_session(state: dict[str, Any]) -> SimulationSession:
    """Rebuild a pickled or copied session through its validating constructor."""
    return SimulationSession(**state)


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
        Trajectory positions in continuous coordinates, one row of
        ``env.n_dims`` coordinates per time point (read-only).
    times : NDArray[np.float64], shape (n_time,)
        Strictly increasing time points in seconds, one per position
        (read-only).
    spike_times : tuple[NDArray[np.float64], ...]
        One read-only array of spike times in seconds per unit, each within
        ``[times[0], times[-1]]``. Any sequence of arrays is accepted and
        stored as a tuple.
    models : tuple[NeuralModel, ...]
        Neural model instances used to generate the spikes. These implement
        the NeuralModel protocol and can be used to regenerate firing rates.
        Any sequence is accepted and stored as a tuple.
    unit_ids : NDArray[np.int64], shape (n_units,)
        Unique integer unit labels aligned one-to-one with spike_times and
        models (read-only). Simulators generate ``np.arange(n_units)``.
    ground_truth : Mapping[int, Mapping[str, Any]]
        True parameters for each cell, keyed by integer unit label. The
        simulators store each model's ``.ground_truth``, enabling validation
        of analysis methods. Stored read-only.
    metadata : Mapping[str, Any]
        Session parameters and configuration. This can include simulation
        settings (duration, trajectory method, cell type), experimental
        parameters, or any other relevant information. Stored read-only.

    Raises
    ------
    ValueError
        If spike_times, unit_ids and models do not align one-to-one; if
        unit_ids are not unique integers; if ground_truth is not keyed by the
        unit labels; if positions are not one row of ``env.n_dims``
        coordinates per time; if times are not strictly increasing; or if a
        spike time is not finite or lies outside ``[times[0], times[-1]]``.

    Notes
    -----
    The session is immutable: its fields cannot be reassigned, its arrays are
    read-only copies, ``spike_times`` and ``models`` are tuples, and
    ``ground_truth`` and ``metadata`` are read-only mappings whose lists
    become tuples. Editing an array passed to the constructor does not change
    the session. To create a modified session, use
    ``dataclasses.replace()``, which validates the new fields:

    >>> from dataclasses import replace
    >>> new_session = replace(
    ...     session, metadata={"note": "relabelled"}
    ... )  # doctest: +SKIP

    The separation of `models` and `ground_truth` allows flexible usage:
    - Use `models` to regenerate firing rates or extend simulations
    - Use `ground_truth` for validation without needing to keep model instances

    Examples
    --------
    >>> import numpy as np
    >>> from shapely.geometry import box
    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import (
    ...     PlaceCellModel,
    ...     SimulationSession,
    ...     generate_population_spikes,
    ...     simulate_trajectory_ou,
    ... )
    >>> env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")
    >>> positions, times = simulate_trajectory_ou(
    ...     env, duration=10.0, speed_units="cm", seed=0
    ... )
    >>> models = [
    ...     PlaceCellModel(env, center=[50.0, 50.0], width=10.0),
    ...     PlaceCellModel(env, center=[70.0, 60.0], width=8.0),
    ... ]
    >>> spike_times = generate_population_spikes(
    ...     models, times, positions, seed=1, show_progress=False
    ... )
    >>> session = SimulationSession(
    ...     env=env,
    ...     positions=positions,
    ...     times=times,
    ...     spike_times=spike_times,
    ...     unit_ids=np.arange(len(models)),
    ...     models=models,
    ...     ground_truth={i: model.ground_truth for i, model in enumerate(models)},
    ...     metadata={"duration": 10.0, "cell_type": "place"},
    ... )
    >>> len(session.spike_times)
    2
    >>> session.positions.flags.writeable
    False

    See Also
    --------
    simulate_session : High-level function to create a complete session
    validate_simulation : Validate session against ground truth
    plot_session_summary : Visualize session data
    """

    env: Environment
    spike_times: Sequence[NDArray[np.float64]]
    unit_ids: NDArray[np.int64]
    times: NDArray[np.float64]
    positions: NDArray[np.float64]
    models: Sequence[NeuralModel]
    ground_truth: Mapping[int, Mapping[str, Any]]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        n = len(self.spike_times)
        if not (len(self.unit_ids) == len(self.models) == n):
            raise ValueError(
                f"spike_times has {n} units, unit_ids {len(self.unit_ids)}, models "
                f"{len(self.models)}; these must match one-to-one.\n"
                "Fix: build unit_ids as np.arange(len(spike_times))."
            )
        unit_ids = np.asarray(self.unit_ids)
        if unit_ids.ndim != 1 or (
            unit_ids.size and not np.issubdtype(unit_ids.dtype, np.integer)
        ):
            raise ValueError(
                _format_error(
                    f"unit_ids must be a 1-D array of integer labels, got "
                    f"{unit_ids.tolist()!r} (dtype {unit_ids.dtype}).",
                    why="ground_truth is keyed by integer labels; a label such "
                    "as 0.7 would silently select the ground truth of unit 0.",
                    fix="pass integer labels, e.g. np.arange(len(spike_times))",
                )
            )
        labels = {int(label) for label in unit_ids}
        if len(labels) != len(unit_ids):
            raise ValueError(
                f"unit_ids {unit_ids.tolist()} are not unique; each unit needs "
                "its own label.\n"
                "Fix: build unit_ids as np.arange(len(spike_times)), or give "
                "every unit a distinct label."
            )
        if set(self.ground_truth) != labels:
            raise ValueError(
                f"ground_truth is keyed by {sorted(self.ground_truth)} but "
                f"unit_ids are {sorted(labels)}; they must match one-to-one.\n"
                "Fix: key ground_truth by each unit's label, "
                "{label: model.ground_truth}."
            )

        times = read_only_copy(self.times, np.float64)
        positions = read_only_copy(self.positions, np.float64)
        if times.ndim != 1 or len(times) != len(positions):
            raise ValueError(
                f"times has shape {times.shape} but positions has "
                f"{len(positions)} samples; they must match one-to-one.\n"
                "Fix: pass one position row per timestamp."
            )
        if positions.ndim != 2 or positions.shape[1] != self.env.n_dims:
            raise ValueError(
                _format_error(
                    f"positions has shape {positions.shape}, but the "
                    f"environment has {self.env.n_dims} dimensions.",
                    why="each row is one position in the environment's coordinates.",
                    fix=f"pass positions of shape (n_time, {self.env.n_dims})",
                )
            )
        if len(times) and not (
            np.all(np.isfinite(times)) and np.all(np.diff(times) > 0)
        ):
            raise ValueError(
                _format_error(
                    "times must be finite and strictly increasing.",
                    why="each position sample needs its own time, in order.",
                    fix="sort the samples by time and drop duplicate timestamps",
                )
            )

        trains = tuple(read_only_copy(train, np.float64) for train in self.spike_times)
        start = times[0] if len(times) else np.nan
        stop = times[-1] if len(times) else np.nan
        for label, train in zip(unit_ids, trains, strict=True):
            if train.ndim != 1 or not np.all(
                np.isfinite(train) & (train >= start) & (train <= stop)
            ):
                raise ValueError(
                    _format_error(
                        f"Unit {int(label)} has spike times outside the recording "
                        f"[{start:g}, {stop:g}] s, or not finite, or not a 1-D "
                        "array.",
                        why="spikes can only be observed while the position is "
                        "sampled.",
                        fix="pass one 1-D array of spike times per unit, "
                        "cropped to [times[0], times[-1]]",
                    )
                )

        object.__setattr__(self, "unit_ids", read_only_copy(unit_ids, np.int64))
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "spike_times", trains)
        object.__setattr__(self, "models", tuple(self.models))
        object.__setattr__(
            self,
            "ground_truth",
            _frozen({int(label): truth for label, truth in self.ground_truth.items()}),
        )
        object.__setattr__(self, "metadata", _frozen(self.metadata))

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        """Pickle and copy through the constructor, which re-freezes the mappings."""
        state = {
            field.name: _thawed(getattr(self, field.name)) for field in fields(self)
        }
        return _rebuild_session, (state,)


def _field_centers(
    env: Environment,
    n_cells: int,
    coverage: Literal["uniform", "random"],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Field centers (or grid phases) for ``n_cells`` cells, shape (n_cells, n_dims).

    ``'uniform'`` spaces the centers evenly along a track (a linearized or
    1-D environment), from one end to the other. In an arena it lays a grid
    of ``ceil(n_cells ** (1 / n_dims))`` targets per axis, inset 20% from
    each side of the bounding box, keeps ``n_cells`` of them evenly spaced
    through the whole grid so that every row and column keeps targets, and
    snaps each to the nearest bin center. The inset keeps fields in the
    interior that an exploring animal samples well. ``'random'`` draws bin
    centers with replacement, ``rng.choice(len(env.bin_centers), n_cells)``.
    """
    if coverage == "random":
        indices = rng.choice(len(env.bin_centers), size=n_cells, replace=True)
        random_centers: NDArray[np.float64] = env.bin_centers[indices]
        return random_centers

    ranges = env.dimension_ranges
    if env.binning.linearization is not None or env.n_dims == 1:
        # Tracks: space centers evenly along the track, so every arm gets
        # fields. A bounding-box grid snapped to a thin track would pile the
        # centers onto the few bins nearest its targets.
        linear = (
            env.bin_centers[:, 0]
            if env.binning.linearization is None
            else env.to_linear(env.bin_centers)
        )
        targets = np.linspace(linear.min(), linear.max(), n_cells)
        nearest = np.abs(linear[None, :] - targets[:, None]).argmin(axis=1)
        field_centers: NDArray[np.float64] = env.bin_centers[nearest]
    elif ranges is None:
        # Layouts without an axis-aligned bounding box: stride through the
        # bin centers.
        step = max(1, len(env.bin_centers) // n_cells)
        field_centers = env.bin_centers[::step][:n_cells]
    else:
        n_dims = len(ranges)
        margin = 0.2
        per_axis = int(np.ceil(n_cells ** (1.0 / n_dims)))
        axis_targets = [
            np.linspace(lo + margin * (hi - lo), hi - margin * (hi - lo), per_axis)
            for (lo, hi) in ranges
        ]
        grid = np.array(np.meshgrid(*axis_targets, indexing="ij")).reshape(n_dims, -1).T
        kept = np.round(np.linspace(0, len(grid) - 1, n_cells)).astype(int)
        distances = np.linalg.norm(
            env.bin_centers[None, :, :] - grid[kept, None, :], axis=2
        )
        field_centers = env.bin_centers[np.argmin(distances, axis=1)]

    n_unique = len(np.unique(field_centers, axis=0))
    if n_unique < n_cells:
        import warnings

        warnings.warn(
            f"Requested n_cells={n_cells} but only {n_unique} distinct "
            f"interior bin centers were available for uniform coverage "
            f"(env has {len(env.bin_centers)} bins). Some cells share a "
            f"center. Consider coverage='random' or a finer bin_size.",
            UserWarning,
            stacklevel=3,
        )
    return field_centers


def simulate_session(
    env: Environment,
    *,
    duration: float,
    n_cells: int = 50,
    cell_type: Literal["place", "border", "grid", "mixed"] = "place",
    trajectory_method: Literal["ou", "sinusoidal", "laps"] = "ou",
    coverage: Literal["uniform", "random"] = "uniform",
    show_progress: bool = True,
    seed: int | None = None,
    **kwargs: Any,
) -> SimulationSession:
    """Simulate complete recording session with trajectory and neural activity.

    High-level convenience function that combines trajectory simulation,
    neural model creation, and spike generation in one call.

    Parameters
    ----------
    env : Environment
        The spatial environment.
    duration : float
        Session duration in seconds. For ``trajectory_method='laps'``, samples
        occur at ``k / sampling_frequency`` in ``[0, duration)``; the final
        sample is within one sample period of ``duration``.
    n_cells : int, optional
        Number of neurons to simulate (default: 50).
    cell_type : {'place', 'border', 'grid', 'mixed'}, optional
        Type of neurons to create (default: 'place').
        - 'place': All place cells
        - 'border': All border cells (``BorderCellModel``; needs an
          environment with declared walls)
        - 'grid': All grid cells (requires 2D environment)
        - 'mixed': 60% place, 20% border, 20% grid
    trajectory_method : {'ou', 'sinusoidal', 'laps'}, optional
        Trajectory generation method (default: 'ou').
    coverage : {'uniform', 'random'}, optional
        Spatial distribution of field centers (default: 'uniform').
        - 'uniform': Evenly space field centers across the environment. On a
          track, centers are evenly spaced along the linearized track; in an
          arena, they form a grid inset 20% from each edge, snapped to the
          nearest bin.
        - 'random': Randomly sample field centers from env.bin_centers
          using np.random.choice(env.bin_centers, size=n_cells).
          May result in clustering or gaps in coverage.
    show_progress : bool, optional
        Show progress bars during generation (default: True).
    seed : int | None, optional
        Random seed for reproducibility. Independent sub-streams for the
        trajectory, field-center selection, and spike generation are spawned
        from a single ``np.random.SeedSequence(seed)`` so the three sources of
        randomness are statistically independent (the trajectory and spike
        streams are not correlated). ``None`` leaves each stream
        nondeterministic but still mutually independent.
    **kwargs : Any
        Additional parameters passed to trajectory and model functions.

        **Trajectory parameters** (all trajectory methods):

        - speed_mean : float - Mean speed for OU process (default varies by method)
        - coherence_time : float - Temporal coherence for OU process
        - n_laps : int - Number of one-way traversals for 'laps' (default: 10),
          alternating outbound/inbound starting outbound
        - sampling_frequency : float - Lap samples per second (default: 500)
        - pause_duration : float - Pause between traversals (default: 0.5 s),
          rounded down to whole samples
        - speed_mean, speed_std : float - For laps, speed draws (default mean
          1.0 and standard deviation ``0.2 * speed_mean``, clipped below at
          ``0.1 * speed_mean``) determine relative traversal times, so their
          scale does not matter and no units are needed.
          Absolute speeds are determined by duration, path lengths and n_laps.
          Every traversal retains at least two samples; fixed pauses are
          reserved first, then remaining samples are allocated in proportion
          to path length divided by the sampled speed.

        **Place cell parameters** (cell_type='place' or 'mixed'):

        - max_rate : float - Peak firing rate in Hz (default: 12.4, see
          ``PlaceCellModel``)
        - width : float | NDArray - Field width in environment units (default:
          11.1 cm in ``env.units``, see ``PlaceCellModel``)
        - baseline_rate : float - Baseline firing rate in Hz (default: 0.01)
        - metric : {'euclidean', 'geodesic'} - Distance computation method
        - condition : Callable - Optional conditional firing function

        **Border cell parameters** (cell_type='border' or 'mixed'):

        - border_width : float - Standard deviation of the distance tuning in
          environment units (``distance_tolerance``; default 9.8 cm in
          ``env.units``, see ``BorderCellModel``)
        - distance : float - Preferred distance from the nearest wall
          (``preferred_distance``, default 0: firing along the walls)
        - max_rate : float - Peak firing rate in Hz (default: 15.0)
        - baseline_rate : float - Baseline firing rate in Hz (default: 0.01)

        **Grid cell parameters** (cell_type='grid' or 'mixed'):

        - spacing : float - Grid spacing in environment units (maps to
          grid_spacing; default 41 cm in ``env.units``, see ``GridCellModel``)
        - orientation : float - Field lattice angle in radians from +x, modulo pi/3 (maps to grid_orientation)
        - max_rate : float - Peak firing rate in Hz (default: 20.0)
        - baseline_rate : float - Baseline firing rate in Hz (default: 0.01)

    Returns
    -------
    session : SimulationSession
        Dataclass containing:
        - env: Environment instance
        - positions: NDArray, shape (n_time, n_dims) - trajectory
        - times: NDArray, shape (n_time,) - time points
        - spike_times: list[NDArray] - spike times per unit
        - unit_ids: NDArray[np.int64] - integer unit labels
        - models: list[NeuralModel] - neural model instances
        - ground_truth: dict - true parameters for each cell
        - metadata: dict - session parameters

    Raises
    ------
    ValueError
        If cell_type is not one of: 'place', 'border', 'grid', 'mixed'.
    ValueError
        If trajectory_method is not one of: 'ou', 'sinusoidal', 'laps'.
    ValueError
        If coverage is not one of: 'uniform', 'random'.
    ValueError
        If grid cells requested but environment is not 2D.
    ValueError
        If lap duration is non-finite or too short for fixed pauses and two
        samples per traversal, or lap timing parameters are invalid.

    Examples
    --------
    >>> # Quick place cell session in a walled 1 m box
    >>> import numpy as np
    >>> from shapely.geometry import box
    >>> from neurospatial import Environment
    >>> env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")
    >>> session = simulate_session(env, duration=2.0, n_cells=3, show_progress=False)
    >>> positions = session.positions  # Typed access
    >>> spike_times = session.spike_times  # IDE autocomplete
    >>> ground_truth = session.ground_truth  # Discoverable

    >>> # Mixed cell types with custom trajectory parameters
    >>> session = simulate_session(
    ...     env,
    ...     duration=2.0,
    ...     n_cells=10,
    ...     cell_type="mixed",
    ...     border_width=5.0,  # border cells' distance tuning, cm
    ...     trajectory_method="ou",
    ...     speed_mean=0.1,
    ...     coherence_time=0.8,
    ...     show_progress=False,
    ... )

    >>> # Place cells with increased firing rate (useful for faster tests)
    >>> session = simulate_session(
    ...     env,
    ...     duration=2.0,
    ...     n_cells=5,
    ...     cell_type="place",
    ...     max_rate=50.0,  # peak rate, Hz
    ...     width=8.0,  # Custom field width
    ...     show_progress=False,
    ... )

    >>> # Validate detected vs true place fields
    >>> from neurospatial.simulation import validate_simulation  # doctest: +SKIP
    >>> report = validate_simulation(session)  # doctest: +SKIP
    >>> print(report["summary"])  # doctest: +SKIP

    See Also
    --------
    simulate_trajectory_ou : Low-level trajectory simulation
    PlaceCellModel : Individual place cell model
    generate_population_spikes : Low-level spike generation
    validate_simulation : Validate simulation results
    """
    # Import functions here to avoid circular imports
    from neurospatial.simulation.models import (
        BorderCellModel,
        GridCellModel,
        PlaceCellModel,
    )
    from neurospatial.simulation.spikes import generate_population_spikes
    from neurospatial.simulation.trajectory import (
        _simulate_trajectory_laps,
        simulate_trajectory_ou,
        simulate_trajectory_sinusoidal,
    )

    # Validate parameters
    valid_cell_types = {"place", "border", "grid", "mixed"}
    if cell_type not in valid_cell_types:
        raise ValueError(
            f"cell_type must be one of {valid_cell_types}, got '{cell_type}'"
        )

    valid_trajectory_methods = {"ou", "sinusoidal", "laps"}
    if trajectory_method not in valid_trajectory_methods:
        raise ValueError(
            f"trajectory_method must be one of {valid_trajectory_methods}, "
            f"got '{trajectory_method}'"
        )

    valid_coverage = {"uniform", "random"}
    if coverage not in valid_coverage:
        raise ValueError(f"coverage must be one of {valid_coverage}, got '{coverage}'")

    # Check for grid cells in non-2D environments
    if cell_type in ("grid", "mixed") and env.n_dims != 2:
        raise ValueError(
            f"Grid cells require 2D environment, but environment has {env.n_dims} dimensions"
        )

    # Validate n_cells and duration
    if n_cells <= 0:
        raise ValueError(f"n_cells must be positive, got {n_cells}")

    if duration <= 0:
        raise ValueError(f"duration must be positive (seconds), got {duration}")

    # Extract model-specific parameters from kwargs BEFORE trajectory generation.
    #
    # Shared rate parameters (``max_rate``, ``baseline_rate``, ``metric``) are
    # accepted by more than one cell type, so they must be *read* (not consumed)
    # by every applicable model type before being removed from ``kwargs``. The
    # earlier implementation popped them into ``place_cell_params`` first, which
    # silently starved the border/grid loops of any caller-supplied rate in
    # mixed mode. We peek the shared params, distribute them per cell type
    # (respecting which constructor accepts which), then pop them once at the
    # end so they don't leak into the trajectory kwargs or session metadata.
    #
    # Note: ``metric`` is accepted only by PlaceCellModel.
    place_cell_params: dict[str, Any] = {}
    border_cell_params: dict[str, Any] = {}
    grid_cell_params: dict[str, Any] = {}

    # Shared rate parameters (read without consuming; popped below).
    for param in ["max_rate", "baseline_rate"]:
        if param in kwargs:
            place_cell_params[param] = kwargs[param]
            border_cell_params[param] = kwargs[param]
            grid_cell_params[param] = kwargs[param]
    if "metric" in kwargs:
        place_cell_params["metric"] = kwargs["metric"]

    # Remove shared params now that all applicable cell types have read them.
    for param in ["max_rate", "baseline_rate", "metric"]:
        kwargs.pop(param, None)

    # PlaceCellModel-only parameters.
    for param in ["width", "condition"]:
        if param in kwargs:
            place_cell_params[param] = kwargs.pop(param)

    # BorderCellModel-only parameters - remap convenience names to actual names.
    if "distance" in kwargs:
        border_cell_params["preferred_distance"] = kwargs.pop("distance")
    if "border_width" in kwargs:
        border_cell_params["distance_tolerance"] = kwargs.pop("border_width")

    # GridCellModel-only parameters - remap convenience names to actual names.
    # Remapped parameters: spacing -> grid_spacing, orientation -> grid_orientation
    if "spacing" in kwargs:
        grid_cell_params["grid_spacing"] = kwargs.pop("spacing")
    if "orientation" in kwargs:
        grid_cell_params["grid_orientation"] = kwargs.pop("orientation")

    # Derive INDEPENDENT seeds for the trajectory, field-center, and spike RNG
    # streams from a single SeedSequence. Spawning guarantees the three streams
    # are statistically independent and non-overlapping, so the trajectory and
    # spike draws no longer share (and correlate) an RNG stream. Passing
    # ``seed=None`` keeps each stream nondeterministic but still mutually
    # independent.
    if seed is None:
        trajectory_seed = field_center_seed = spike_seed = None
    else:
        traj_ss, field_ss, spike_ss = np.random.SeedSequence(seed).spawn(3)
        trajectory_seed = int(traj_ss.generate_state(1)[0])
        field_center_seed = int(field_ss.generate_state(1)[0])
        spike_seed = int(spike_ss.generate_state(1)[0])

    # Generate trajectory using its own independent seed.
    if trajectory_method == "ou":
        # speed_units is required by simulate_trajectory_ou;
        # fall back to env.units when the caller didn't pass it.
        ou_kwargs = dict(kwargs)
        ou_kwargs.setdefault("speed_units", env.units)
        positions, times = simulate_trajectory_ou(
            env, duration=duration, seed=trajectory_seed, **ou_kwargs
        )
    elif trajectory_method == "sinusoidal":
        positions, times = simulate_trajectory_sinusoidal(
            env, duration=duration, seed=trajectory_seed, **kwargs
        )
    elif trajectory_method == "laps":
        # Ensure return_metadata is False for consistent return signature
        # Extract n_laps separately to avoid duplicate keyword argument
        n_laps = kwargs.pop("n_laps", 10)
        laps_kwargs = {k: v for k, v in kwargs.items() if k != "return_metadata"}
        result = _simulate_trajectory_laps(
            env,
            n_laps=n_laps,
            duration=duration,
            seed=trajectory_seed,
            return_metadata=False,
            call='simulate_session(trajectory_method="laps")',
            **laps_kwargs,
        )
        # Type narrowing: return_metadata=False guarantees 2-tuple
        positions, times = result  # type: ignore[misc]
    else:
        raise ValueError(f"Unknown trajectory method: {trajectory_method}")

    # Initialize RNG for field center selection using its own independent seed.
    rng = np.random.default_rng(field_center_seed)

    # Cell counts: mixed sessions are 60% place, 20% border, 20% grid.
    if cell_type == "mixed":
        n_place = int(0.6 * n_cells)
        n_border = int(0.2 * n_cells)
        n_grid = n_cells - n_place - n_border
    else:
        n_place = n_cells if cell_type == "place" else 0
        n_border = n_cells if cell_type == "border" else 0
        n_grid = n_cells if cell_type == "grid" else 0

    # Place centers and grid phases share one even layout; in mixed sessions
    # a seeded permutation assigns them to the two types, so neither type
    # covers only part of the environment.
    field_centers = _field_centers(env, n_place + n_grid, coverage, rng)
    if cell_type == "mixed":
        field_centers = field_centers[rng.permutation(len(field_centers))]

    # Create models based on cell_type
    models: list[NeuralModel] = []
    if cell_type == "place":
        for center in field_centers:
            models.append(PlaceCellModel(env, center=center, **place_cell_params))
    elif cell_type == "border":
        for _ in range(n_cells):
            models.append(BorderCellModel(env, **border_cell_params))
    elif cell_type == "grid":
        for center in field_centers:
            # Use center for phase offset
            models.append(GridCellModel(env, phase_offset=center, **grid_cell_params))
    elif cell_type == "mixed":
        # Create place cells
        for center in field_centers[:n_place]:
            models.append(PlaceCellModel(env, center=center, **place_cell_params))

        # Create border cells
        for _ in range(n_border):
            models.append(BorderCellModel(env, **border_cell_params))

        # Create grid cells
        for center in field_centers[n_place : n_place + n_grid]:
            models.append(GridCellModel(env, phase_offset=center, **grid_cell_params))
    else:
        raise ValueError(f"Unknown cell_type: {cell_type}")

    # Generate spikes for all cells using the independent spike seed.
    spike_times = generate_population_spikes(
        models,
        times,
        positions,
        seed=spike_seed,
        show_progress=show_progress,
    )

    # Collect ground truth from each model
    ground_truth = {i: model.ground_truth for i, model in enumerate(models)}

    # Create metadata dict
    metadata = {
        "duration": duration,
        "n_cells": n_cells,
        "cell_type": cell_type,
        "trajectory_method": trajectory_method,
        "coverage": coverage,
        "seed": seed,
        **kwargs,  # Include any additional parameters
    }
    if trajectory_method == "laps":
        metadata.update(
            n_laps=n_laps,
            sampling_frequency=kwargs.get("sampling_frequency", 500.0),
            pause_duration=kwargs.get("pause_duration", 0.5),
            sampling_convention="k / sampling_frequency in [0, duration)",
        )

    # Return SimulationSession
    return SimulationSession(
        env=env,
        positions=positions,
        times=times,
        spike_times=spike_times,
        unit_ids=np.arange(len(spike_times), dtype=np.int64),
        models=models,
        ground_truth=ground_truth,
        metadata=metadata,
    )
