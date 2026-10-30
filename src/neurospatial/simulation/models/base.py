"""Base protocol and shared input checks for neural models."""

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import read_only_copy
from neurospatial.geometry import require_position_columns

if TYPE_CHECKING:
    from neurospatial.environment import Environment


@runtime_checkable
class NeuralModel(Protocol):
    """Protocol for neural models that generate firing rates.

    Every model in ``neurospatial.simulation`` has the same interface:

    - ``firing_rate(times, positions, *, headings=None)`` returns the rate in
      Hz at each sample, shape ``(n_time,)``.
    - ``requires_headings`` says whether ``firing_rate`` needs ``headings``.
    - ``ground_truth`` returns the model's parameters.

    Head-direction cells ignore ``positions`` apart from checking that there
    is one per time. Models that need headings (head-direction cells,
    spatial view cells, egocentric object-vector cells) raise when
    ``headings`` is None; :func:`generate_population_spikes` then supplies
    the direction of movement.

    ``ground_truth`` always holds ``'cell_type'`` (one of ``'place'``,
    ``'border'``, ``'boundary_vector'``, ``'grid'``, ``'head_direction'``,
    ``'object_vector'`` and ``'spatial_view'``) and every constructor
    parameter needed to rebuild the model, after defaults and random draws
    are resolved. Arrays in it are read-only. It is where to read the values
    a model fires with: the package's models keep their constructor
    arguments as attributes, with None for a parameter left to its default,
    so ``dataclasses.replace`` and pickling rebuild a model exactly as the
    constructor would. The other keys by cell type:

    **place**

        - ``'center'``: NDArray[np.float64], shape (n_dims,) - field center
        - ``'width'``: float | NDArray[np.float64] - Gaussian standard
          deviation, one value or one per dimension
        - ``'max_rate'``, ``'baseline_rate'``: float - rates in Hz
        - ``'metric'``: str - ``'euclidean'`` or ``'geodesic'``
        - ``'condition'``: Callable | None - firing gate

    **border**

        - ``'preferred_distance'``: float - distance from the nearest wall
        - ``'distance_tolerance'``: float - tuning standard deviation
        - ``'max_rate'``, ``'baseline_rate'``: float - rates in Hz
        - ``'boundary_source'``: str - ``'physical'`` or ``'analysis'``

    **boundary_vector**

        - ``'preferred_distance'``: float - preferred distance to a wall
        - ``'preferred_direction'``: float - allocentric direction (radians)
        - ``'distance_tolerance'``: float - distance tuning width
        - ``'direction_tolerance'``: float - angular half-width (radians)
        - ``'max_rate'``: float - rate next to a straight wall at the
          preferred distance and direction, in Hz
        - ``'baseline_rate'``: float - baseline firing rate in Hz
        - ``'boundary_source'``: str - ``'physical'`` or ``'analysis'``

    **grid**

        - ``'grid_spacing'``: float - distance between grid peaks
        - ``'grid_orientation'``: float - field lattice angle in radians,
          from +x counterclockwise, modulo pi/3
        - ``'phase_offset'``: NDArray[np.float64], shape (2,) - a field's
          position
        - ``'max_rate'``, ``'baseline_rate'``: float - rates in Hz

    **head_direction**

        - ``'preferred_direction'``: float - preferred direction in radians
        - ``'preferred_direction_deg'``: float - the same in degrees
        - ``'concentration'``: float - von Mises concentration (κ)
        - ``'tuning_width'``: float - half-width at half of ``max_rate``,
          in radians; NaN when the rate never falls to half
        - ``'tuning_width_deg'``: float - the same in degrees
        - ``'max_rate'``, ``'baseline_rate'``: float - rates in Hz

    **object_vector**

        - ``'object_positions'``: NDArray[np.float64], shape (n_objects, 2)
        - ``'preferred_distance'``, ``'distance_tolerance'``: float
        - ``'preferred_direction'``: float | None - radians
        - ``'direction_frame'``: str - ``'allocentric'`` or ``'egocentric'``
        - ``'direction_concentration'``: float - von Mises κ
        - ``'object_selectivity'``: str - ``'any'``, ``'nearest'`` or
          ``'specific'``
        - ``'specific_object_index'``: int | None
        - ``'metric'``: str - ``'euclidean'`` or ``'geodesic'``
        - ``'max_rate'``, ``'baseline_rate'``: float - rates in Hz

    **spatial_view**

        - ``'preferred_view_location'``: NDArray[np.float64], shape (2,)
        - ``'view_field_width'``: float - Gaussian standard deviation
        - ``'view_distance'``: float | None - fixed gaze distance
        - ``'gaze_model'``: str - ``'ray_cast'`` or ``'fixed_distance'``
        - ``'boundary_source'``: str - ``'physical'`` or ``'analysis'``
        - ``'require_visibility'``: bool - line-of-sight requirement
        - ``'fov'``: FieldOfView | None - field of view
        - ``'max_rate'``, ``'baseline_rate'``: float - rates in Hz

    Examples
    --------
    Creating a custom neural model:

    >>> from neurospatial.simulation.models.base import NeuralModel
    >>> import numpy as np
    >>>
    >>> class ConstantModel:
    ...     '''Fires at the same rate everywhere.'''
    ...
    ...     requires_headings = False
    ...
    ...     def __init__(self, max_rate):
    ...         self.max_rate = max_rate
    ...
    ...     def firing_rate(self, times, positions, *, headings=None):
    ...         return np.full(len(times), self.max_rate)
    ...
    ...     @property
    ...     def ground_truth(self):
    ...         return {"cell_type": "constant", "max_rate": self.max_rate}
    >>>
    >>> isinstance(ConstantModel(max_rate=10.0), NeuralModel)
    True

    See Also
    --------
    PlaceCellModel : Gaussian place field model
    BorderCellModel : Wall-distance tuned model
    GridCellModel : Hexagonal grid pattern model

    Notes
    -----
    This is a Protocol (PEP 544), not a base class: a model needs only the
    methods and properties above. ``isinstance`` checks that they exist, not
    their signatures.
    """

    @property
    def requires_headings(self) -> bool:
        """Whether ``firing_rate`` needs ``headings``."""
        ...

    def firing_rate(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        headings: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Firing rate at each sample.

        Parameters
        ----------
        times : NDArray[np.float64], shape (n_time,)
            Sample times in seconds.
        positions : NDArray[np.float64], shape (n_time, n_dims)
            Positions in the environment's coordinates and units.
        headings : NDArray[np.float64], shape (n_time,), optional
            Head direction in radians (0 = East, +pi/2 = North). Required
            when ``requires_headings`` is True.

        Returns
        -------
        firing_rate : NDArray[np.float64], shape (n_time,)
            Non-negative rate in Hz; NaN where an input the model needs is
            missing, or where the model is undefined (for example outside the
            walls of a wall-driven cell).
        """
        ...

    @property
    def ground_truth(self) -> dict[str, Any]:
        """Return ground truth parameters for validation.

        Returns
        -------
        parameters : dict[str, Any]
            ``'cell_type'`` and the parameters that rebuild the model (see
            the class docstring for the keys of each cell type).

        Examples
        --------
        >>> from neurospatial.simulation import PlaceCellModel
        >>> from neurospatial import Environment
        >>> import numpy as np
        >>>
        >>> samples = np.random.default_rng(0).uniform(0, 100, (1000, 2))
        >>> env = Environment.from_samples(samples, bin_size=2.0, units="cm")
        >>> pc = PlaceCellModel(env, center=[50, 50], width=10.0, max_rate=20.0)
        >>> pc.ground_truth["cell_type"]
        'place'
        >>> pc.ground_truth["center"]
        array([50., 50.])
        >>> pc.ground_truth["width"]
        10.0
        """
        ...


def geodesic_lookup_bins(
    env: "Environment", positions: NDArray[np.float64]
) -> NDArray[np.intp]:
    """Bin whose geodesic distance field stands for each position.

    A position inside the environment uses its containing bin, and one outside
    uses the nearest bin center. A missing (NaN) position gets bin -1.

    Parameters
    ----------
    env : Environment
        Environment the distance fields were computed on.
    positions : NDArray[np.float64], shape (n_time, n_dims)
        Positions in the environment's coordinates.

    Returns
    -------
    NDArray[np.intp], shape (n_time,)
        Bin index per position, or -1 for a missing position.
    """
    bins = np.asarray(env.nearest_bin(positions), dtype=np.intp)
    inside = env.contains(positions)
    if inside.any():
        bins[inside] = env.bin_at(positions[inside])
    return bins


def _draw_rng(model: Any) -> np.random.Generator:
    """The generator a model draws its random parameters from.

    A model built with ``seed=None`` first draws a seed from the operating
    system and keeps it in its ``seed`` field, so ``dataclasses.replace``
    and pickling rebuild the same draw. A seed that was passed is used as
    passed.
    """
    seed = model.seed
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
        object.__setattr__(model, "seed", seed)
    return np.random.default_rng(seed)


def _check_rates(max_rate: float, baseline_rate: float) -> None:
    """Refuse a non-finite or negative rate, or a baseline above the peak."""
    if not (np.isfinite(max_rate) and max_rate > 0):
        raise ValueError(
            _format_error(
                f"max_rate must be a finite positive rate in Hz, got {max_rate!r}.",
                fix="pass the peak firing rate in Hz, e.g. max_rate=15.0",
            )
        )
    if not (np.isfinite(baseline_rate) and baseline_rate >= 0):
        raise ValueError(
            _format_error(
                "baseline_rate must be a finite non-negative rate in Hz, got "
                f"{baseline_rate!r}.",
                fix="pass the rate away from the field in Hz, e.g. baseline_rate=0.01",
            )
        )
    if baseline_rate > max_rate:
        raise ValueError(
            _format_error(
                f"baseline_rate ({baseline_rate} Hz) is above max_rate "
                f"({max_rate} Hz).",
                why="the tuning curve rises from baseline_rate to its peak, max_rate.",
                fix="pass a baseline_rate no larger than max_rate",
            )
        )


def _check_positive(
    name: str,
    value: float,
    *,
    allow_zero: bool = False,
    units: str | None = "the environment's units",
) -> None:
    """Refuse a non-finite value, or one that is not positive."""
    ok = np.isfinite(value) and (value >= 0 if allow_zero else value > 0)
    if not ok:
        bound = "non-negative" if allow_zero else "positive"
        raise ValueError(
            _format_error(
                f"{name} must be finite and {bound}, got {value!r}.",
                fix=f"pass a finite {bound} {name}" + (f" in {units}" if units else ""),
            )
        )


def _check_angle(name: str, value: float) -> None:
    """Refuse an angle that is not a finite number of radians."""
    if not np.isfinite(value):
        raise ValueError(
            _format_error(
                f"{name} must be a finite angle in radians, got {value!r}.",
                fix=f"pass {name} in radians, e.g. {name}=np.pi / 2 for North",
            )
        )


def _check_point(name: str, value: ArrayLike, n_dims: int) -> NDArray[np.float64]:
    """One finite point of ``n_dims`` coordinates, as a read-only array."""
    point = read_only_copy(value, np.float64)
    if point.shape != (n_dims,) or not np.all(np.isfinite(point)):
        raise ValueError(
            _format_error(
                f"{name} must be {n_dims} finite coordinates, shape ({n_dims},), "
                f"got {point.tolist()!r} with shape {point.shape}.",
                fix=f"pass one point in the environment's {n_dims}-D coordinates",
            )
        )
    return point


def _model_inputs(
    times: ArrayLike, positions: ArrayLike, n_dims: int | None, *, call: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Times and positions as arrays, one position row per time.

    Parameters
    ----------
    times : array_like, shape (n_time,)
        Sample times in seconds.
    positions : array_like, shape (n_time, n_dims)
        Positions; with ``n_dims=None`` only their number is checked.
    n_dims : int or None
        Coordinates per position, or None for a model that does not read
        positions.
    call : str
        The method the scientist called, named in errors.

    Returns
    -------
    times : NDArray[np.float64], shape (n_time,)
    positions : NDArray[np.float64]
    """
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError(
            _format_error(
                f"{call} got times of shape {times.shape}.",
                fix="pass one time in seconds per sample, shape (n_time,)",
            )
        )
    if n_dims is not None:
        require_position_columns(positions, n_dims, call=call)
    if len(positions) != len(times):
        raise ValueError(
            _format_error(
                f"{call} got {len(times)} times but {len(positions)} positions.",
                why="each position is the animal's position at one time.",
                fix="pass times and positions from the same samples, one "
                "position row per time",
            )
        )
    return times, positions


def _model_headings(
    headings: ArrayLike | None, n_time: int, *, call: str, alternative: str = ""
) -> NDArray[np.float64]:
    """Headings as an array of one angle per time; refuse a missing input.

    Parameters
    ----------
    headings : array_like or None, shape (n_time,)
        Head direction in radians.
    n_time : int
        Number of samples.
    call : str
        The method the scientist called, named in errors.
    alternative : str, optional
        Another fix for missing headings, appended to the message.

    Returns
    -------
    NDArray[np.float64], shape (n_time,)
    """
    if headings is None:
        raise ValueError(
            _format_error(
                f"{call} needs headings.",
                why="the cell's firing depends on which way the animal faces.",
                fix="pass headings= with one angle in radians per sample (0 = "
                "East, pi/2 = North); generate_population_spikes() derives "
                "them from the direction of movement when you omit them" + alternative,
            )
        )
    headings = np.asarray(headings, dtype=np.float64)
    if headings.shape != (n_time,):
        raise ValueError(
            _format_error(
                f"{call} got headings of shape {headings.shape} for {n_time} samples.",
                why="a heading is one angle per sample, not a direction vector.",
                fix=f"pass one angle in radians per sample, shape ({n_time},); "
                "convert (dx, dy) vectors with np.arctan2(dy, dx)",
            )
        )
    return headings
