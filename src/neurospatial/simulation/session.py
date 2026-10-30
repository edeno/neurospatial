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
from neurospatial._validation import is_integer_at_least, validate_finite_scalar
from neurospatial.ops.boundary_vectors import nearest_boundary
from neurospatial.simulation._seeds import _stream_seeds

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


@dataclass(frozen=True, eq=False)
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

    Sessions compare and hash by identity, like the models they hold: a
    copy with the same data is a different session.

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


def _require_positive(
    name: str, value: object, *, count: bool = False, units: str = ""
) -> None:
    """Refuse a count that is not a positive integer, or a value not finite and positive.

    ``units`` follows the requirement in the message, e.g. ``" (seconds)"``.
    """
    if count:
        if is_integer_at_least(value, 1):
            return
    else:
        try:
            validate_finite_scalar(value, name=name, positive=True)
            return
        except ValueError:
            pass
    requirement = "an integer" if count else "finite"
    raise ValueError(
        _format_error(
            f"{name} must be positive and {requirement}{units}, got {value!r}.",
            fix=f"pass a positive {'whole' if count else 'finite'} number as {name}",
        )
    )


def _simulated_session(
    env: Environment,
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    models: list[NeuralModel],
    *,
    spike_seed: int,
    show_progress: bool,
    metadata: Mapping[str, Any],
) -> SimulationSession:
    """Draw each model's spikes along the trajectory and hold them in a session.

    Units are labelled ``0 .. len(models) - 1`` in model order, and each
    unit's ground truth is its model's.
    """
    from neurospatial.simulation.spikes import generate_population_spikes

    spike_times = generate_population_spikes(
        models, times, positions, seed=spike_seed, show_progress=show_progress
    )
    return SimulationSession(
        env=env,
        positions=positions,
        times=times,
        spike_times=spike_times,
        unit_ids=np.arange(len(models), dtype=np.int64),
        models=models,
        ground_truth={i: model.ground_truth for i, model in enumerate(models)},
        metadata=metadata,
    )


#: Model parameters ``simulate_session`` passes to each cell type, by the
#: models' own names.
_MODEL_PARAMETERS: dict[str, frozenset[str]] = {
    "place": frozenset({"width", "metric", "condition"}),
    "border": frozenset(
        {
            "preferred_distance",
            "preferred_direction",
            "distance_tolerance",
            "direction_tolerance",
            "boundary_source",
        }
    ),
    "grid": frozenset({"grid_spacing", "grid_orientation"}),
}

#: Rate parameters every cell model takes.
_RATE_PARAMETERS = ("max_rate", "baseline_rate")


#: Largest preferred wall distance, in centimeters, drawn for a simulated
#: border cell. Border cells are boundary vector cells with short preferred
#: distances (see ``simulate_session``). With the default distance tolerance
#: of ``BoundaryVectorCellModel``, the field above 20% of the peak, measured
#: square to the preferred wall, spans 0 to 13.1 cm at a preferred distance of
#: 0 and 0 to 21.8 cm at 8 cm, so its center lies 6.5 to 10.9 cm from the wall
#: (about ``6.5 + 0.55 d`` cm). Preferred distances drawn uniformly from
#: [0, 8] cm therefore give field centers 6.5-10.9 cm from the wall with a
#: mean of 8.7 cm: within the 6-14 cm range, and at the 8.8 cm mean, of the
#: mouse border fields of Høydal et al. (2019).
BORDER_MAX_PREFERRED_DISTANCE_CM = 8.0


def _border_cells(
    env: Environment,
    n_cells: int,
    rng: np.random.Generator,
    params: Mapping[str, Any],
) -> list[NeuralModel]:
    """``n_cells`` border cells, each with its own wall direction and distance.

    Each cell is a ``BoundaryVectorCellModel`` whose preferred allocentric
    direction is drawn uniformly from [-pi, pi) and whose preferred distance
    is drawn uniformly from [0, 8] cm in ``env.units``, unless ``params``
    fixes them; ``params`` holds the other model parameters.
    """
    from neurospatial._units import default_length
    from neurospatial.simulation.models import BoundaryVectorCellModel

    directions = rng.uniform(-np.pi, np.pi, n_cells)
    distances = rng.uniform(0.0, 1.0, n_cells)
    if "preferred_distance" not in params and n_cells:
        distances = distances * default_length(
            env,
            BORDER_MAX_PREFERRED_DISTANCE_CM,
            name="preferred_distance",
            call="simulate_session(cell_type='border')",
        )
    return [
        BoundaryVectorCellModel(
            env,
            **{
                "preferred_distance": float(distance),
                "preferred_direction": float(direction),
                **params,
            },
        )
        for distance, direction in zip(distances, directions, strict=True)
    ]


#: Share of its local half-width that a uniform field center keeps clear of
#: the walls (see ``_interior_bins``). In a square arena of side ``L`` every
#: interior bin's half-width is ``L / 2``, so the centers keep 20% of the side
#: from every wall, where an exploring animal samples well and no field is
#: cut by a wall.
UNIFORM_WALL_INSET = 0.4


def _bin_depths(env: Environment) -> NDArray[np.float64] | None:
    """Each bin's distance to the nearest wall, shape (n_bins,), or None.

    In 2-D it is the exact distance from the bin center to the edge of the
    bins or a wall between them (``nearest_boundary`` with
    ``boundary_source='analysis'``), so a declared internal wall counts. In
    other dimensions it is measured to the nearest boundary bin's center
    plus half that bin's length, the distance from the bin center to the
    boundary bin's outer edge. None when the bins have no known outline
    (point bins without cells) or no boundary.
    """
    from scipy.spatial import cKDTree

    if env.n_dims == 2:
        try:
            distances, _ = nearest_boundary(
                env, env.bin_centers, boundary_source="analysis"
            )
        except NotImplementedError:
            return None
        return distances
    try:
        boundary = env.boundary_bins
    except NotImplementedError:
        return None
    if boundary.size == 0:
        return None
    bins = env.binning.bins
    lengths = bins.sizes ** (1.0 / np.asarray(bins.measure_dims, dtype=float))
    points = env.coordinate_system.cartesian_embedding(env.bin_centers)
    gaps, nearest = cKDTree(points[boundary]).query(points)
    depths: NDArray[np.float64] = gaps + 0.5 * lengths[boundary][nearest]
    return depths


def _interior_bins(env: Environment, depths: NDArray[np.float64]) -> NDArray[np.intp]:
    """Bins at least ``UNIFORM_WALL_INSET`` of their local half-width from the walls.

    A bin's local half-width is the radius of the largest disc (ball in 3-D)
    centered on a ridge bin of ``depths`` that contains it: the corridor's
    half-width in a corridor, the inscribed circle's radius in an open
    arena. Ridge bins are those whose depth grows by less than half a step
    toward every neighbor; on a straight stretch away from a wall it grows
    by a full step. Measuring the inset against the local half-width, not
    the arena's widest point, keeps narrow arms beside a wide platform
    (a radial-arm maze) in the layout. Boundary bins touch a wall, so they
    are neither ridge bins nor in the layout, even a corner bin whose
    neighbors are all boundary bins too.
    """
    from scipy.spatial import cKDTree

    points = env.coordinate_system.cartesian_embedding(env.bin_centers)
    pairs = env.edge_pairs
    u = np.concatenate([pairs[:, 0], pairs[:, 1]])
    v = np.concatenate([pairs[:, 1], pairs[:, 0]])
    steps = np.linalg.norm(points[u] - points[v], axis=1)
    excluded = np.zeros(env.n_bins, dtype=bool)
    excluded[env.boundary_bins] = True
    np.logical_or.at(excluded, u, depths[v] >= depths[u] + 0.5 * steps)
    ridge = np.flatnonzero(~excluded)
    half_widths = np.zeros(env.n_bins)
    tree = cKDTree(points)
    for bin_index, members in zip(
        ridge, tree.query_ball_point(points[ridge], depths[ridge]), strict=True
    ):
        half_widths[members] = np.maximum(half_widths[members], depths[bin_index])
    interior = (half_widths > 0) & (depths >= UNIFORM_WALL_INSET * half_widths)
    interior[env.boundary_bins] = False
    return np.flatnonzero(interior)


def _spread_bins(
    env: Environment,
    candidates: NDArray[np.intp],
    n_points: int,
    depths: NDArray[np.float64],
) -> NDArray[np.intp]:
    """``n_points`` distinct bins, spread by farthest-point sampling.

    Points go to ``candidates`` first and, once every candidate has one, to
    the other bins, still farthest from all points so far. Distances are
    geodesic, along the bins' connections, so fields are spread over the
    space the animal moves through and centers on either side of a wall
    count as far apart. One point goes to the deepest candidate. Otherwise
    the first point is the candidate farthest from the deepest one, and
    each next point the bin farthest from all points so far, ties going to
    the lowest bin index. The layout is deterministic, and for two or more
    points the layout of ``n`` points is the first ``n`` of the layout of
    ``n + 1``: adding a cell adds one center and moves none.
    """
    from scipy.sparse.csgraph import dijkstra

    from neurospatial.behavior._environment_paths import _bin_graph

    deepest = int(candidates[np.argmax(depths[candidates])])
    if n_points == 1:
        return np.array([deepest])
    graph = _bin_graph(env)

    def distances_from(bin_index: int) -> NDArray[np.float64]:
        return np.asarray(dijkstra(graph, directed=False, indices=bin_index))

    pool = candidates
    chosen = [int(pool[np.argmax(distances_from(deepest)[pool])])]
    nearest = distances_from(chosen[0])
    for _ in range(n_points - 1):
        if len(chosen) == len(candidates):
            pool = np.setdiff1d(np.arange(env.n_bins), candidates)
        chosen.append(int(pool[np.argmax(nearest[pool])]))
        nearest = np.minimum(nearest, distances_from(chosen[-1]))
    return np.array(chosen)


def _field_centers(
    env: Environment,
    n_cells: int,
    coverage: Literal["uniform", "random"],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Field centers (or grid phases) for ``n_cells`` cells, shape (n_cells, n_dims).

    ``'uniform'`` gives every cell its own bin center. Along a track (a
    linearized or 1-D environment) the centers are evenly spaced from one
    end to the other. In an arena they are spread over the bins by
    farthest-point sampling (``_spread_bins``) over the bins that keep
    ``UNIFORM_WALL_INSET`` of their local half-width from the walls
    (``_interior_bins``); cells beyond those bins get the bins nearer the
    walls, farthest from the centers already chosen. ``'random'`` draws bin
    centers with replacement,
    ``rng.choice(len(env.bin_centers), n_cells)``.
    """
    if n_cells == 0:
        return np.empty((0, env.n_dims))
    if coverage == "random":
        indices = rng.choice(len(env.bin_centers), size=n_cells, replace=True)
        random_centers: NDArray[np.float64] = env.bin_centers[indices]
        return random_centers

    if n_cells > env.n_bins:
        raise ValueError(
            _format_error(
                f"coverage='uniform' needs a bin for each of the {n_cells} "
                f"cells, but the environment has {env.n_bins} bins.",
                why="uniform coverage gives every cell its own field center.",
                fix=f"simulate at most {env.n_bins} cells, use a smaller "
                "bin_size, or pass coverage='random', which draws centers "
                "with replacement",
            )
        )
    if env.binning.linearization is not None or env.n_dims == 1:
        # Tracks: space centers evenly along the track, so every arm gets
        # fields from one end to the other.
        linear = (
            env.bin_centers[:, 0]
            if env.binning.linearization is None
            else env.to_linear(env.bin_centers)
        )
        targets = np.linspace(linear.min(), linear.max(), n_cells)
        chosen = np.abs(linear[None, :] - targets[:, None]).argmin(axis=1)
        if len(np.unique(chosen)) < n_cells:
            # Targets closer together than the bins around a gap between
            # track segments: take evenly spaced bins in track order instead.
            order = np.argsort(linear, kind="stable")
            chosen = order[
                np.round(np.linspace(0, env.n_bins - 1, n_cells)).astype(int)
            ]
    else:
        depths = _bin_depths(env)
        if depths is None:
            depths = np.zeros(env.n_bins)
            candidates = np.arange(env.n_bins)
        else:
            candidates = _interior_bins(env, depths)
            if len(candidates) == 0:
                candidates = np.arange(env.n_bins)
        chosen = _spread_bins(env, candidates, n_cells, depths)
    field_centers: NDArray[np.float64] = env.bin_centers[chosen]
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
        - 'border': All border cells (needs an environment with declared
          walls; see Notes)
        - 'grid': All grid cells (requires 2D environment)
        - 'mixed': 60% place, 20% border, 20% grid
    trajectory_method : {'ou', 'sinusoidal', 'laps'}, optional
        Trajectory generation method (default: 'ou').
    coverage : {'uniform', 'random'}, optional
        Spatial distribution of place field centers and grid phases
        (default: 'uniform').

        - 'uniform': Every cell gets its own bin center, spread over the
          environment; the layout depends on the environment alone, not on
          ``seed``. On a track (a linearized or 1-D environment), centers
          are evenly spaced from one end to the other. In an arena of any
          shape, they are spread by farthest-point sampling with distances
          along the bins (so centers on either side of a wall count as far
          apart), over the bins that keep 40% of their local half-width
          from the walls: 20% of the side in a square arena, the middle of
          a corridor in a maze. There fields are sampled well and are not
          cut by a wall. When more cells than those bins are simulated,
          the extra cells take the bins nearer the walls, farthest from
          the centers already chosen; adding a cell adds one center and
          moves none. ``n_cells`` larger than ``env.n_bins`` raises.
        - 'random': Bin centers drawn with replacement from the seeded
          field-parameter stream, ``rng.choice(len(env.bin_centers),
          size=n)``. May result in clustering or gaps in coverage.
    show_progress : bool, optional
        Show progress bars during generation (default: True).
    seed : int | None, optional
        Random seed for reproducibility. Independent streams for the
        trajectory, the field parameters (centers, grid phases) and the spikes
        are spawned from one ``np.random.SeedSequence(seed)``, so spike draws
        are not correlated with the movement. ``None`` leaves each stream
        nondeterministic but still mutually independent.
    **kwargs : Any
        Trajectory and cell-model parameters, by the names of the functions
        and models that take them. Each goes only where it applies: a model
        parameter of a cell type the session does not simulate is refused,
        and an unknown name raises ``TypeError`` from the trajectory
        function.

        **Trajectory parameters**, by ``trajectory_method``:

        - 'ou' (``simulate_trajectory_ou``): ``speed_mean`` and
          ``speed_std`` (default 8 and 4 cm/s in ``env.units``),
          ``speed_units`` (default ``env.units``), ``coherence_time``,
          ``dt``, ``start_position``, ``rotational_velocity_std``,
          ``rotational_velocity_coherence_time`` and ``boundary_mode``.
        - 'sinusoidal' (``simulate_trajectory_sinusoidal``): ``speed``
          (default 10 cm/s in ``env.units``), ``period``,
          ``sampling_frequency``, ``pause_duration`` and ``pause_at_peaks``.
        - 'laps' (``simulate_trajectory_laps``): ``n_laps`` (default 10
          one-way traversals, alternating outbound and inbound starting
          outbound), ``sampling_frequency`` (default 500 Hz),
          ``pause_duration`` (default 0.5 s between traversals, rounded down
          to whole samples), ``outbound_path`` and ``inbound_path``, and
          ``speed_mean`` and ``speed_std``. For laps the speed draws only set
          the relative traversal times (default mean 1.0 and standard
          deviation ``0.2 * speed_mean``, clipped below at
          ``0.1 * speed_mean``), so their scale does not matter and no units
          are needed: absolute speeds follow from ``duration``, the path
          lengths and ``n_laps``. Every traversal keeps at least two
          samples; the pauses are reserved first, then the remaining samples
          are shared in proportion to path length divided by the sampled
          speed.

        **Rate parameters** of every cell type: ``max_rate`` (Hz) and
        ``baseline_rate`` (Hz), each model's default when omitted.

        **Place cell parameters** (``PlaceCellModel``; cell_type='place' or
        'mixed'): ``width`` (default 11.1 cm in ``env.units``), ``metric``
        ({'euclidean', 'geodesic'}) and ``condition``.

        **Border cell parameters** (``BoundaryVectorCellModel``;
        cell_type='border' or 'mixed'): ``preferred_distance`` and
        ``preferred_direction`` (drawn per cell when omitted; see Notes),
        ``distance_tolerance`` (default ``preferred_distance / 12 + 8 cm``),
        ``direction_tolerance`` (default pi/6) and ``boundary_source``.

        **Grid cell parameters** (``GridCellModel``; cell_type='grid' or
        'mixed'): ``grid_spacing`` (default 41 cm in ``env.units``) and
        ``grid_orientation`` (radians from +x, modulo pi/3).

    Returns
    -------
    session : SimulationSession
        Dataclass containing:
        - env: Environment instance
        - positions: NDArray, shape (n_time, n_dims) - trajectory
        - times: NDArray, shape (n_time,) - time points
        - spike_times: tuple[NDArray] - spike times per unit
        - unit_ids: NDArray[np.int64] - integer unit labels
        - models: tuple[NeuralModel] - neural model instances
        - ground_truth: Mapping - each model's ``ground_truth``, by unit label
        - metadata: Mapping - session parameters

    Raises
    ------
    ValueError
        If cell_type is not one of: 'place', 'border', 'grid', 'mixed'.
    ValueError
        If trajectory_method is not one of: 'ou', 'sinusoidal', 'laps'.
    ValueError
        If coverage is not one of: 'uniform', 'random', or ``coverage`` is
        'uniform' and ``n_cells`` exceeds the number of bins.
    ValueError
        If grid cells requested but environment is not 2D.
    ValueError
        If ``n_cells`` is not a positive integer or ``duration`` is not
        finite and positive.
    ValueError
        If a model parameter applies only to a cell type the session does
        not simulate (for example ``grid_spacing`` with
        ``cell_type='place'``).
    TypeError
        If a parameter is neither a model parameter nor a parameter of the
        chosen trajectory function.
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

    >>> # Mixed cell types with custom model and trajectory parameters
    >>> session = simulate_session(
    ...     env,
    ...     duration=2.0,
    ...     n_cells=10,
    ...     cell_type="mixed",
    ...     grid_spacing=50.0,  # grid cells, cm
    ...     trajectory_method="ou",
    ...     speed_mean=15.0,  # cm/s
    ...     coherence_time=0.8,
    ...     show_progress=False,
    ... )
    >>> sorted({truth["cell_type"] for truth in session.ground_truth.values()})
    ['boundary_vector', 'grid', 'place']

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

    >>> # A parameter of a cell type the session does not simulate is refused
    >>> simulate_session(
    ...     env, duration=2.0, n_cells=3, grid_spacing=50.0, show_progress=False
    ... )
    Traceback (most recent call last):
        ...
    ValueError: grid_spacing applies only to cell_type='grid' or 'mixed', but this session simulates cell_type='place'.
    Why: grid_spacing is a parameter of the grid cell model, and no cell in this session would use it.
    Fix: remove grid_spacing, or simulate cell_type='grid'

    See Also
    --------
    simulate_trajectory_ou : Low-level trajectory simulation
    PlaceCellModel : Individual place cell model
    BoundaryVectorCellModel : The border cells' model
    generate_population_spikes : Low-level spike generation
    validate_simulation : Validate simulation results

    Notes
    -----
    **Border cells** fire along one or a few walls of the arena (Solstad et
    al. 2008), and are modelled as boundary vector cells with short preferred
    distances (Barry et al. 2006; Lever et al. 2009): each is a
    ``BoundaryVectorCellModel`` that fires where a wall lies close by in its
    preferred allocentric direction. Every cell has its own tuning, drawn
    from the field-parameter stream: a preferred direction uniform on
    [-pi, pi), as no wall direction is favored, and a preferred distance
    uniform on [0, 8] cm, which puts the field centers 6.5-10.9 cm from the
    wall (mean 8.7 cm), matching the 6-14 cm range and 8.8 cm mean of mouse
    border fields (Høydal et al. 2019; derivation at
    ``BORDER_MAX_PREFERRED_DISTANCE_CM``). A direction toward a corner gives
    a cell that fires along both walls of the corner. Each unit's
    ``ground_truth`` is its model's, with ``cell_type='boundary_vector'``.

    References
    ----------
    Barry, C., et al. (2006). The boundary vector cell model of place cell
        firing and spatial memory. Reviews in the Neurosciences, 17, 71-97.
    Høydal, Ø. A., et al. (2019). Object-vector coding in the medial
        entorhinal cortex. Nature, 568, 400-404.
    Lever, C., et al. (2009). Boundary vector cells in the subiculum of the
        hippocampal formation. Journal of Neuroscience, 29, 9771-9777.
    Solstad, T., et al. (2008). Representation of geometric borders in the
        entorhinal cortex. Science, 322, 1865-1868.
    """
    # Import functions here to avoid circular imports
    from neurospatial.simulation.models import GridCellModel, PlaceCellModel
    from neurospatial.simulation.trajectory import (
        _DEFAULT_LAP_PAUSE_DURATION,
        _DEFAULT_LAP_SAMPLING_FREQUENCY,
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

    _require_positive("n_cells", n_cells, count=True)
    _require_positive("duration", duration, units=" (seconds)")

    # Model parameters go to the cell types that take them; the rest are
    # trajectory parameters.
    present = {"place", "border", "grid"} if cell_type == "mixed" else {cell_type}
    model_params: dict[str, dict[str, Any]] = {kind: {} for kind in _MODEL_PARAMETERS}
    for kind, names in _MODEL_PARAMETERS.items():
        for name in sorted(names.intersection(kwargs)):
            if kind not in present:
                raise ValueError(
                    _format_error(
                        f"{name} applies only to cell_type='{kind}' or 'mixed', "
                        f"but this session simulates cell_type='{cell_type}'.",
                        why=f"{name} is a parameter of the {kind} cell model, "
                        "and no cell in this session would use it.",
                        fix=f"remove {name}, or simulate cell_type='{kind}'",
                    )
                )
            model_params[kind][name] = kwargs.pop(name)
    for name in _RATE_PARAMETERS:
        if name in kwargs:
            value = kwargs.pop(name)
            for params in model_params.values():
                params[name] = value

    # Independent streams for the trajectory, the field parameters and the
    # spikes, so that spike draws are not correlated with the movement.
    trajectory_seed, field_center_seed, spike_seed = _stream_seeds(seed, 3)

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
            env, duration=duration, **kwargs
        )
    else:
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

    # Place cells, then border cells, then grid cells; a type the session
    # does not simulate has no cells. Grid cells take the centers after the
    # place cells' as their phase offsets.
    models: list[NeuralModel] = [
        PlaceCellModel(env, center=center, **model_params["place"])
        for center in field_centers[:n_place]
    ]
    models.extend(_border_cells(env, n_border, rng, model_params["border"]))
    models.extend(
        GridCellModel(env, phase_offset=center, **model_params["grid"])
        for center in field_centers[n_place:]
    )

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
            sampling_frequency=kwargs.get(
                "sampling_frequency", _DEFAULT_LAP_SAMPLING_FREQUENCY
            ),
            pause_duration=kwargs.get("pause_duration", _DEFAULT_LAP_PAUSE_DURATION),
            sampling_convention="k / sampling_frequency in [0, duration)",
        )

    # Spikes for all cells from the independent spike seed.
    return _simulated_session(
        env,
        positions,
        times,
        models,
        spike_seed=spike_seed,
        show_progress=show_progress,
        metadata=metadata,
    )
