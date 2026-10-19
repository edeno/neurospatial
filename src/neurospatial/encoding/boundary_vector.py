"""Boundary-vector rate maps: firing by distance and direction to walls.

A boundary-vector cell fires when a wall lies at a preferred distance in a
preferred allocentric direction, for example 10 cm to the East (Hartley et
al., 2000; Lever et al., 2009). An egocentric boundary cell fires when a wall
lies at a preferred distance and direction relative to the animal's heading,
for example 15 cm to the left (Hinman et al., 2019; Alexander et al., 2020).

The rate map follows Hinman et al. (2019). From each tracking sample, one ray
per direction bin finds the distance to the first wall. Each sampling
interval adds its duration to the (distance, direction) bin of every ray that
reaches a wall within ``distance_range``, and each spike adds one count to
the same bins of the sample it falls after. The rate is spikes divided by that
boundary occupancy, so a bin's rate is the firing rate while a wall was at
that distance in that direction. Walls are the arena's declared walls by
default (``boundary_source="physical"``).

Examples
--------
>>> import numpy as np
>>> from shapely.geometry import box
>>> from neurospatial import Environment
>>> from neurospatial.encoding import compute_boundary_vector_rate
>>> env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=5.0, units="cm")
>>> rng = np.random.default_rng(0)
>>> times = np.arange(0, 300, 1 / 30)
>>> positions = rng.uniform(5, 95, (len(times), 2))
>>> spike_times = np.sort(rng.uniform(0, 300, 600))
>>> result = compute_boundary_vector_rate(
...     env, spike_times, times, positions, direction_frame="allocentric"
... )
>>> result.firing_rate.shape  # 10 distance bins x 12 direction bins
(120,)

References
----------
Hartley, T., et al. (2000). Modeling place fields in terms of the cortical
    inputs to the hippocampus. Hippocampus, 10, 369-379.
Lever, C., et al. (2009). Boundary vector cells in the subiculum of the
    hippocampal formation. Journal of Neuroscience, 29, 9771-9777.
Hinman, J. R., et al. (2019). Neuronal representation of environmental
    boundaries in egocentric coordinates. Nature Communications, 10, 2772.
Alexander, A. S., et al. (2020). Egocentric boundary vector tuning of the
    retrosplenial cortex. Science Advances, 6, eaaz2322.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._results import UnitSelectionMixin, _copy_analysis_settings, label_at
from neurospatial.encoding._base import SpatialResultMixin, _to_numpy
from neurospatial.encoding._polar_results import _PolarRateMixin, polar_summary_frame

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

    from neurospatial._intervals import TimeSupport
    from neurospatial._typing import SpikeTrainsLike
    from neurospatial.environment import EgocentricPolarSystem, Environment

    PolarEnvironment = Environment[EgocentricPolarSystem]

__all__ = [
    "BoundaryVectorRateResult",
    "BoundaryVectorRatesResult",
    "compute_boundary_vector_rate",
    "compute_boundary_vector_rates",
]


def _check_boundary_labels(direction_frame: str, boundary_source: str) -> None:
    """Refuse frame and wall labels that exports would misread."""
    from neurospatial.encoding.egocentric import _check_direction_frame

    _check_direction_frame(direction_frame, context="boundary-vector result")
    if boundary_source not in ("physical", "analysis"):
        raise ValueError(
            f"boundary_source must be 'physical' or 'analysis', got {boundary_source!r}."
        )


def _boundary_summary(
    result: BoundaryVectorRateResult | BoundaryVectorRatesResult,
    *,
    index: Sequence[Any],
) -> pd.DataFrame:
    """The same metric columns for single and population results."""
    from neurospatial.encoding._metrics import batch_spatial_information

    rates = np.atleast_2d(_to_numpy(result._get_rates()))
    information = batch_spatial_information(rates, _to_numpy(result.occupancy))
    df = polar_summary_frame(
        result,
        index=index,
        extra={"spatial_information": np.asarray(information)},
        extra_units={"spatial_information": "bits/spike"},
        thresholds=None,
    )
    df.attrs["boundary_source"] = result.boundary_source
    return df


@_copy_analysis_settings
@dataclass(frozen=True, repr=False)
class BoundaryVectorRateResult(_PolarRateMixin, SpatialResultMixin):
    """Firing rate of one unit by distance and direction to walls.

    Parameters
    ----------
    firing_rate : ArrayLike, shape (n_bins,)
        Rate in Hz in each (distance, direction) bin of ``env``. NaN where
        a wall was never at that distance and direction for at least
        ``min_occupancy`` seconds: not measured, not silent.
    occupancy : ArrayLike, shape (n_bins,)
        Boundary occupancy in seconds: how long a wall lay at that distance
        in that direction. Each ray counts separately, so the total is up to
        ``n_direction_bins`` times the analyzed duration.
    env : Environment
        The polar (distance, direction) environment of the bins.
    distance_range : tuple of float
        (min, max) distance to a wall covered by the bins.
    n_distance_bins, n_direction_bins : int
        Number of distance and direction bins.
    direction_frame : {"allocentric", "egocentric"}
        Frame of the direction to the wall. Allocentric: 0 = East,
        +pi/2 = North. Egocentric: 0 = ahead, +pi/2 = left.
    boundary_source : {"physical", "analysis"}
        Which walls the rays stopped at.
    unit_id : int or str or None
        Identifier for this unit; set by indexing a population result.

    Other Parameters
    ----------------
    analysis_settings : dict or None
        Resolved analysis windows, retained time intervals, smoothing and
        binning settings.

    See Also
    --------
    compute_boundary_vector_rate : Compute this result.
    BoundaryVectorRatesResult : The population version.
    """

    firing_rate: ArrayLike
    occupancy: ArrayLike
    env: PolarEnvironment
    distance_range: tuple[float, float]
    n_distance_bins: int
    n_direction_bins: int
    direction_frame: Literal["allocentric", "egocentric"] = field(kw_only=True)
    boundary_source: Literal["physical", "analysis"] = field(kw_only=True)
    unit_id: int | str | None = None
    spike_window: NDArray[np.float64] | None = field(
        default=None, kw_only=True, compare=False
    )
    analysis_settings: dict[str, Any] | None = field(
        default=None, kw_only=True, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _check_boundary_labels(self.direction_frame, self.boundary_source)
        self._own_arrays()

    def _family_attrs(self) -> dict[str, Any]:
        return {"boundary_source": self.boundary_source}

    def summary_table(self) -> pd.DataFrame:
        """Per-unit metrics with the same columns as the population table.

        Returns
        -------
        pandas.DataFrame
            One row indexed by ``unit_id``: ``peak_rate``,
            ``preferred_distance``, ``preferred_direction_deg``,
            ``preferred_direction`` and ``spatial_information``. The frame
            and wall source are in ``df.attrs``.
        """
        return _boundary_summary(self, index=self._row_unit_ids().tolist())

    def _headline_metrics(self) -> dict[str, float]:
        return {
            "preferred_distance": self.preferred_distance(),
            "preferred_direction": self.preferred_direction(),
        }

    def preferred_distance(self) -> float:
        """Distance to the wall at the peak rate, in the environment's units.

        Returns
        -------
        float
            NaN when no bin has a rate.
        """
        return float(self._peak_coordinates(0)[0])

    def preferred_direction(self) -> float:
        """Direction from the animal to the wall at the peak rate, in radians.

        Returns
        -------
        float
            In ``direction_frame``; NaN when no bin has a rate.
        """
        return float(self._peak_coordinates(1)[0])

    def spatial_information(self) -> float:
        """Skaggs information of the boundary map, in bits per spike.

        Returns
        -------
        float
            Computed with the boundary occupancy as the bin probabilities. It
            is biased upward for few spikes; compare it with a shuffled
            control before calling a cell boundary-tuned.
        """
        from neurospatial.encoding._metrics import spatial_information

        return spatial_information(
            _to_numpy(self.firing_rate), _to_numpy(self.occupancy)
        )

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot the rate by distance (x axis) and direction (y axis).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure when omitted.
        **kwargs
            Passed to the polar environment's ``plot_field``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return self.env.plot_field(_to_numpy(self.firing_rate), ax=ax, **kwargs)


@_copy_analysis_settings
@dataclass(frozen=True, repr=False)
class BoundaryVectorRatesResult(
    _PolarRateMixin, SpatialResultMixin, UnitSelectionMixin
):
    """Firing rates of several units by distance and direction to walls.

    Parameters
    ----------
    firing_rates : ArrayLike, shape (n_units, n_bins)
        Rate in Hz per unit and (distance, direction) bin; NaN where not
        measured.
    occupancy : ArrayLike, shape (n_bins,)
        Boundary occupancy in seconds, shared by every unit.
    env : Environment
        The polar (distance, direction) environment of the bins.
    distance_range : tuple of float
        (min, max) distance to a wall covered by the bins.
    n_distance_bins, n_direction_bins : int
        Number of distance and direction bins.
    direction_frame : {"allocentric", "egocentric"}
        Frame of the direction to the wall.
    boundary_source : {"physical", "analysis"}
        Which walls the rays stopped at.
    unit_ids : array-like, optional
        One label per unit; ``np.arange(n_units)`` by default.
    unit_table : pandas.DataFrame, optional
        Per-unit metadata indexed by ``unit_id``.

    Other Parameters
    ----------------
    analysis_settings : dict or None
        Resolved analysis windows, retained time intervals, smoothing and
        binning settings.

    See Also
    --------
    compute_boundary_vector_rates : Compute this result.
    BoundaryVectorRateResult : The single-unit version.
    """

    firing_rates: ArrayLike
    occupancy: ArrayLike
    env: PolarEnvironment
    distance_range: tuple[float, float]
    n_distance_bins: int
    n_direction_bins: int
    direction_frame: Literal["allocentric", "egocentric"] = field(kw_only=True)
    boundary_source: Literal["physical", "analysis"] = field(kw_only=True)
    unit_ids: NDArray[Any] | Sequence[Any] | None = field(default=None, compare=False)
    unit_table: pd.DataFrame | None = field(default=None, compare=False)
    spike_window: NDArray[np.float64] | None = field(
        default=None, kw_only=True, compare=False
    )
    analysis_settings: dict[str, Any] | None = field(
        default=None, kw_only=True, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        from neurospatial._results import align_unit_table, resolve_unit_ids

        _check_boundary_labels(self.direction_frame, self.boundary_source)
        n_units = int(np.asarray(self.firing_rates).shape[0])
        object.__setattr__(self, "unit_ids", resolve_unit_ids(self.unit_ids, n_units))
        object.__setattr__(
            self,
            "unit_table",
            align_unit_table(
                self.unit_table, self.unit_ids, context="BoundaryVectorRatesResult"
            ),
        )
        self._own_arrays()

    def _family_attrs(self) -> dict[str, Any]:
        return {"boundary_source": self.boundary_source}

    def __len__(self) -> int:
        """Number of units."""
        return int(np.asarray(self.firing_rates).shape[0])

    def __getitem__(self, idx: int) -> BoundaryVectorRateResult:
        """The single-unit result at position ``idx``; ``sel(unit_id=...)`` by label."""
        idx = self._unit_position(idx)
        return BoundaryVectorRateResult(
            firing_rate=self.firing_rates[idx],  # type: ignore[index]
            occupancy=self.occupancy,
            env=self.env,
            distance_range=self.distance_range,
            n_distance_bins=self.n_distance_bins,
            n_direction_bins=self.n_direction_bins,
            direction_frame=self.direction_frame,
            boundary_source=self.boundary_source,
            unit_id=label_at(self.unit_ids, idx),
            spike_window=self.spike_window,
            analysis_settings=self._analysis_attrs()["analysis_settings"],
        )

    def __iter__(self) -> Iterator[BoundaryVectorRateResult]:
        """Each unit's single-unit result, in order."""
        for i in range(len(self)):
            yield self[i]

    def preferred_distances(self) -> NDArray[np.float64]:
        """Distance to the wall at each unit's peak rate, shape (n_units,)."""
        return self._peak_coordinates(0)

    def preferred_directions(self) -> NDArray[np.float64]:
        """Direction to the wall at each unit's peak rate, shape (n_units,).

        In radians in ``direction_frame``; NaN for a unit with no rate.
        """
        return self._peak_coordinates(1)

    def spatial_information(self) -> NDArray[np.float64]:
        """Skaggs information of each unit's map, bits per spike, shape (n_units,)."""
        from neurospatial.encoding._metrics import batch_spatial_information

        return np.asarray(
            batch_spatial_information(
                _to_numpy(self.firing_rates), _to_numpy(self.occupancy)
            )
        )

    def summary_table(
        self, unit_ids: Sequence[str | int] | None = None
    ) -> pd.DataFrame:
        """One row per unit, indexed by ``unit_id``, with the single-unit columns.

        Parameters
        ----------
        unit_ids : sequence of str or int, optional
            Labels for the index; the result's own ``unit_ids`` by default.

        Returns
        -------
        pandas.DataFrame
        """
        from neurospatial._results import resolve_unit_ids

        if unit_ids is None:
            index: list[Any] = list(np.asarray(self.unit_ids))
        else:
            index = list(unit_ids)
            resolve_unit_ids(
                np.asarray(index, dtype=object),
                len(self),
                context="BoundaryVectorRatesResult.summary_table",
            )
        return _boundary_summary(self, index=index)

    def plot(self, idx: int, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot unit ``idx``'s rate by distance and direction.

        Parameters
        ----------
        idx : int
            Unit position.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure when omitted.
        **kwargs
            Passed to the polar environment's ``plot_field``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return self[idx].plot(ax=ax, **kwargs)


def _boundary_frame_bins(
    env: Environment,
    positions: NDArray[np.float64],
    headings: NDArray[np.float64] | None,
    *,
    direction_frame: Literal["allocentric", "egocentric"],
    distance_range: tuple[float, float],
    n_distance_bins: int,
    n_direction_bins: int,
    boundary_source: Literal["physical", "analysis"],
) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
    """Polar bin of the wall along each ray, and which samples were observed.

    Returns
    -------
    frame_bins : NDArray[np.intp], shape (n_samples, n_direction_bins)
        Flat polar bin of each ray's wall; -1 when the wall is outside
        ``distance_range`` or the sample is missing.
    observed : NDArray[np.bool_], shape (n_samples,)
        Samples with a position inside the walls (and a heading, when used).
    """
    from neurospatial.encoding._egocentric_counts import _coords_to_flat_bin_idx
    from neurospatial.ops.boundary_vectors import compute_boundary_distances

    distances, directions = compute_boundary_distances(
        env,
        positions,
        headings,
        direction_frame=direction_frame,
        n_directions=n_direction_bins,
        max_distance=float(distance_range[1]),
        boundary_source=boundary_source,
    )
    observed = np.asarray(~np.isnan(distances).all(axis=1), dtype=np.bool_)
    frame_bins: NDArray[np.intp] = _coords_to_flat_bin_idx(
        distances.ravel(),
        np.tile(directions, len(distances)),
        distance_range,
        n_distance_bins,
        n_direction_bins,
    ).reshape(distances.shape)
    return frame_bins, observed


def _count_boundary_events(
    spike_trains: Sequence[NDArray[np.float64]],
    times: NDArray[np.float64],
    frame_bins: NDArray[np.intp],
    observed: NDArray[np.bool_],
    n_bins: int,
    *,
    max_gap: float | None,
    epochs: NDArray[np.float64] | None,
    spike_window: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], TimeSupport]:
    """Spike counts and boundary occupancy summed over every ray.

    An interval counts when its start sample was observed and it passes the
    gap and window checks; it then adds its duration, and each of its spikes
    one count, to the bin of every ray of its start sample that met a wall in
    range. Spikes and occupancy use the same intervals.

    Returns
    -------
    spike_counts : NDArray[np.float64], shape (n_units, n_bins)
    occupancy : NDArray[np.float64], shape (n_bins,)
    support : TimeSupport
        The retained intervals, for the result's analysis settings.
    """
    from neurospatial._intervals import warn_excluded_intervals
    from neurospatial.encoding._spatial_counts import prepare_frame_binning

    prepared = prepare_frame_binning(
        times,
        np.where(observed, 0, -1).astype(np.intp),
        n_bins,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
    )
    support = prepared.support
    mask = support.mask
    warn_excluded_intervals(
        times,
        mask,
        max_gap=support.max_gap,
        epochs=support.epochs,
        spike_window=support.spike_window,
        start_bin_gate=True,
        result="the rate map is empty",
    )
    starts = frame_bins[:-1][mask]  # (n_kept_intervals, n_rays)
    durations = np.broadcast_to(np.diff(times)[mask][:, None], starts.shape)
    hit = starts >= 0
    occupancy = np.bincount(
        starts[hit], weights=durations[hit], minlength=n_bins
    ).astype(np.float64)

    spike_counts = np.zeros((len(spike_trains), n_bins), dtype=np.float64)
    for unit, spikes in enumerate(spike_trains):
        spikes = spikes[(spikes >= times[0]) & (spikes < times[-1])]
        frames = np.searchsorted(times, spikes, side="right") - 1
        rays = frame_bins[frames[mask[frames]]]
        spike_counts[unit] = np.bincount(rays[rays >= 0], minlength=n_bins)
    return spike_counts, occupancy, support


def compute_boundary_vector_rates(
    env: Environment,
    spike_times: SpikeTrainsLike,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64] | None = None,
    *,
    direction_frame: Literal["allocentric", "egocentric"],
    distance_range: tuple[float, float] = (0.0, 50.0),
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    boundary_source: Literal["physical", "analysis"] = "physical",
    max_gap: float | None = 0.5,
    epochs: Any = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
) -> BoundaryVectorRatesResult:
    """Firing rate of each unit by distance and direction to the walls.

    Parameters
    ----------
    env : Environment
        Allocentric 2-D environment the positions were recorded in. Its
        arena supplies the walls.
    spike_times : sequence of arrays, or a spike group
        Spike times in seconds, one array per unit.
    times : NDArray[np.float64], shape (n_samples,)
        Tracking timestamps in seconds.
    positions : NDArray[np.float64], shape (n_samples, 2)
        Animal positions in the environment's coordinates.
    headings : NDArray[np.float64], shape (n_samples,), optional
        Head direction in radians (0 = East). Required with
        ``direction_frame="egocentric"`` and refused with ``"allocentric"``.
    direction_frame : {"allocentric", "egocentric"}
        ``"allocentric"`` for boundary-vector cells (0 = East, +pi/2 =
        North); ``"egocentric"`` for egocentric boundary cells (0 = ahead,
        +pi/2 = left). There is no default: the two maps answer different
        questions.
    distance_range : tuple of float, default=(0.0, 50.0)
        (min, max) distance to a wall, in the environment's units. A wall
        farther along a ray than the maximum adds nothing for that ray.
    n_distance_bins : int, default=10
        Number of distance bins.
    n_direction_bins : int, default=12
        Number of direction bins; one ray is cast at each bin's center.
    boundary_source : {"physical", "analysis"}, default="physical"
        ``"physical"`` stops rays at the arena's declared outline and walls;
        ``"analysis"`` at the edge of the environment's bins.
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open windows (seconds).
    spike_window : same forms as epochs, or None
        When the electrophysiology was recording.
    method : {"binned", "diffusion_kde", "gaussian_kde"}, default="binned"
        ``"binned"`` divides counts by occupancy per bin. The kernel methods
        smooth over the polar grid, which also mixes neighboring distances.
    bandwidth : float, optional
        Smoothing bandwidth for the kernel methods, in the environment's
        units.
    min_occupancy : float, default=0.0
        Seconds of boundary occupancy a bin needs to get a rate.
    unit_ids : array-like, optional
        One label per unit.

    Returns
    -------
    BoundaryVectorRatesResult

    Raises
    ------
    ValueError
        If ``headings`` does not match ``direction_frame``, or the inputs do
        not have matching lengths.
    NotImplementedError
        If the arena has no known walls and ``boundary_source="physical"``:
        build the environment with ``Environment.from_polygon`` or pass an
        ``OpenArena`` to say where the walls are.
    CartesianOnlyError
        If ``env`` is an egocentric polar environment.

    See Also
    --------
    compute_boundary_vector_rate : The single-unit version.
    neurospatial.ops.compute_boundary_distances : The ray distances themselves.
    compute_object_vector_rate : Firing by distance and direction to objects.

    Notes
    -----
    Each ray counts separately, so a bin's occupancy is the time a wall lay
    at that distance in that direction, and the total occupancy is up to
    ``n_direction_bins`` times the analyzed duration. An interval counts when
    its start sample was observed (a position inside the walls and, for
    egocentric maps, a heading); spikes and occupancy use the same
    intervals. Bins never reached stay NaN, which means not measured, not
    silent.

    Examples
    --------
    >>> import numpy as np
    >>> from shapely.geometry import box
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding import compute_boundary_vector_rates
    >>> env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=5.0, units="cm")
    >>> rng = np.random.default_rng(0)
    >>> times = np.arange(0, 300, 1 / 30)
    >>> positions = rng.uniform(5, 95, (len(times), 2))
    >>> headings = rng.uniform(-np.pi, np.pi, len(times))
    >>> spikes = [np.sort(rng.uniform(0, 300, 600)) for _ in range(3)]
    >>> rates = compute_boundary_vector_rates(
    ...     env, spikes, times, positions, headings, direction_frame="egocentric"
    ... )
    >>> list(rates.summary_table().columns)
    ['peak_rate', 'preferred_distance', 'preferred_direction_deg', 'preferred_direction', 'spatial_information']
    """
    return _boundary_vector_rates(
        env,
        spike_times,
        times,
        positions,
        headings,
        direction_frame=direction_frame,
        distance_range=distance_range,
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        boundary_source=boundary_source,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        unit_ids=unit_ids,
        context="compute_boundary_vector_rates",
    )


def compute_boundary_vector_rate(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64] | None = None,
    *,
    direction_frame: Literal["allocentric", "egocentric"],
    distance_range: tuple[float, float] = (0.0, 50.0),
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    boundary_source: Literal["physical", "analysis"] = "physical",
    max_gap: float | None = 0.5,
    epochs: Any = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
) -> BoundaryVectorRateResult:
    """Firing rate of one unit by distance and direction to the walls.

    Takes the arguments of :func:`compute_boundary_vector_rates` with one
    unit's ``spike_times`` (an array of seconds) and no ``unit_ids``.

    Returns
    -------
    BoundaryVectorRateResult
        Equal, field for field, to unit 0 of the population result.

    See Also
    --------
    compute_boundary_vector_rates : The population version, with details.
    """
    from neurospatial.encoding._validation import validate_spike_times

    validate_spike_times(
        np.asarray(spike_times, dtype=np.float64),
        context="compute_boundary_vector_rate",
    )
    population = _boundary_vector_rates(
        env,
        [np.asarray(spike_times, dtype=np.float64)],
        times,
        positions,
        headings,
        direction_frame=direction_frame,
        distance_range=distance_range,
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        boundary_source=boundary_source,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        unit_ids=None,
        context="compute_boundary_vector_rate",
    )
    unit = population[0]
    return BoundaryVectorRateResult(
        firing_rate=unit.firing_rate,
        occupancy=unit.occupancy,
        env=unit.env,
        distance_range=unit.distance_range,
        n_distance_bins=unit.n_distance_bins,
        n_direction_bins=unit.n_direction_bins,
        direction_frame=unit.direction_frame,
        boundary_source=unit.boundary_source,
        spike_window=unit.spike_window,
        analysis_settings=unit.analysis_settings,
    )


def _boundary_vector_rates(
    env: Environment,
    spike_times: SpikeTrainsLike,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64] | None,
    *,
    direction_frame: Literal["allocentric", "egocentric"],
    distance_range: tuple[float, float],
    n_distance_bins: int,
    n_direction_bins: int,
    boundary_source: Literal["physical", "analysis"],
    max_gap: float | None,
    epochs: Any,
    spike_window: Any,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"],
    bandwidth: float | None,
    min_occupancy: float,
    unit_ids: NDArray[Any] | Sequence[Any] | None,
    context: str,
) -> BoundaryVectorRatesResult:
    """Validate, cast the rays once, count every unit and estimate rates."""
    from neurospatial._intervals import resolve_time_windows
    from neurospatial._spikes import as_labelled_spike_trains
    from neurospatial.encoding._egocentric_counts import _create_egocentric_environment
    from neurospatial.encoding._provenance import rate_analysis_settings
    from neurospatial.encoding._smoothing import (
        _validate_smoothing_parameters,
        _warn_if_fully_masked,
        resolve_spatial_bandwidth,
        smooth_rate_maps_batch,
    )
    from neurospatial.encoding._spatial_counts import _warn_if_population_silent
    from neurospatial.encoding._validation import (
        _validate_env_argument,
        validate_spike_trains,
        validate_trajectory,
    )
    from neurospatial.encoding.egocentric import _raw_polar_rate

    _validate_env_argument(
        env, context=context, arguments="spike_times, times, positions[, headings]"
    )
    _check_boundary_labels(direction_frame, boundary_source)
    _validate_smoothing_parameters(method, 1.0 if bandwidth is None else bandwidth)
    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)

    spike_trains, resolved_unit_ids, _ = as_labelled_spike_trains(
        spike_times, unit_ids, context=context
    )
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    headings = None if headings is None else np.asarray(headings, dtype=np.float64)
    validate_trajectory(
        times, positions=positions, headings=headings, context=context, n_dims=2
    )
    validate_spike_trains(spike_trains, context=context)
    _warn_if_population_silent(
        spike_trains,
        times,
        max_gap=max_gap,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
    )

    frame_bins, observed = _boundary_frame_bins(
        env,
        positions,
        headings,
        direction_frame=direction_frame,
        distance_range=distance_range,
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        boundary_source=boundary_source,
    )
    polar_env = _create_egocentric_environment(
        distance_range, n_distance_bins, n_direction_bins
    ).with_metadata(units=env.units)
    spike_counts, occupancy, support = _count_boundary_events(
        spike_trains,
        times,
        frame_bins,
        observed,
        polar_env.n_bins,
        max_gap=max_gap,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
    )

    if method == "binned":
        _warn_if_fully_masked(occupancy, min_occupancy)
        firing_rates = (
            np.stack(
                [_raw_polar_rate(c, occupancy, min_occupancy) for c in spike_counts]
            )
            if len(spike_trains)
            else np.empty((0, polar_env.n_bins))
        )
        resolved_bandwidth = None
    else:
        resolved_bandwidth = resolve_spatial_bandwidth(polar_env, bandwidth)
        firing_rates = np.asarray(
            smooth_rate_maps_batch(
                polar_env,
                spike_counts,
                occupancy,
                method=method,
                bandwidth=resolved_bandwidth,
                min_occupancy=min_occupancy,
            )
        )
    settings = rate_analysis_settings(
        support,
        method=method,
        bandwidth=resolved_bandwidth,
        min_occupancy=min_occupancy,
        backend="numpy",
        direction_frame=direction_frame,
        boundary_source=boundary_source,
        distance_range=[float(value) for value in distance_range],
        n_distance_bins=int(n_distance_bins),
        n_direction_bins=int(n_direction_bins),
    )
    return BoundaryVectorRatesResult(
        firing_rates=firing_rates,
        occupancy=occupancy,
        env=polar_env,
        distance_range=distance_range,
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        direction_frame=direction_frame,
        boundary_source=boundary_source,
        unit_ids=resolved_unit_ids,
        spike_window=resolved_spike_window,
        analysis_settings=settings,
    )
