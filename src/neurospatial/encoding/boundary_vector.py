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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._observation import validate_break_before
from neurospatial._results import _copy_analysis_settings, _copy_unit_table
from neurospatial.encoding._base import (
    PopulationRateMixin,
    SingleUnitRateMixin,
    _to_numpy,
)
from neurospatial.encoding._polar_results import (
    DEFAULT_DIRECTION_BIN_SIZE,
    _PolarRateMixin,
    _unknown_units,
    polar_summary_frame,
    require_known_bandwidth,
    resolve_polar_grid,
)

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

    from neurospatial._intervals import TimeSupport
    from neurospatial._typing import SpikeTrainsLike
    from neurospatial.environment import EgocentricPolarSystem, Environment
    from neurospatial.geometry._boundaries import BoundarySource

    PolarEnvironment = Environment[EgocentricPolarSystem]

__all__ = [
    "BoundaryVectorRateResult",
    "BoundaryVectorRatesResult",
    "compute_boundary_vector_rate",
    "compute_boundary_vector_rates",
]


# Default wall distances, 0-50 cm: the range of Hinman et al. (2019)'s
# egocentric boundary rate maps (``distanceBins = 0:1:50``, "How far to look
# (cm)", in EgocentricRatemap.m of github.com/hasselmonians/
# EgocentricBoundaryCells). The range is not the arena: along a line through
# the animal, the walls on either side are a chord apart, so a cell tuned to a
# wall at distance d in one direction is equally tuned to the opposite wall at
# (chord - d). In a 100 cm box, a cell firing 15 cm from the East wall peaked
# at the West wall 87.5 cm away on a grid covering the whole box, and at the
# East wall 17.5 cm away on 0-50 cm.
_DEFAULT_WALL_RANGE_CM = 50.0


def _resolve_wall_range(
    units: str | None,
    distance_range: tuple[float, float] | None,
    *,
    context: str,
) -> tuple[float, float]:
    """The caller's wall-distance range, or 0-50 cm converted to ``units``.

    Raises
    ------
    ValueError
        If ``distance_range`` is None and ``units`` is not a known length
        unit: 50 cm has no meaning without a scale.
    """
    from neurospatial._exceptions import _format_error
    from neurospatial._units import units_per_centimeter

    if distance_range is not None:
        return distance_range
    per_cm = units_per_centimeter(units)
    if per_cm is None:
        raise ValueError(
            _format_error(
                f"{context}: distance_range is required because "
                f"{_unknown_units(units)}.",
                why=f"Why: the default 0-{_DEFAULT_WALL_RANGE_CM:g} cm range "
                "cannot be converted to coordinates of unknown scale",
                fix="pass distance_range=(0.0, max_distance) and "
                "distance_bin_size in the units of positions, or an env with "
                "units='cm' (or 'mm', 'm')",
            )
        )
    return (0.0, _DEFAULT_WALL_RANGE_CM * per_cm)


def _check_boundary_labels(direction_frame: str, boundary_source: str) -> None:
    """Refuse frame and wall labels that exports would misread."""
    from neurospatial.encoding.egocentric import _check_direction_frame

    _check_direction_frame(direction_frame, context="boundary-vector result")
    if boundary_source not in ("physical", "analysis"):
        raise ValueError(
            f"boundary_source must be 'physical' or 'analysis', got {boundary_source!r}."
        )


def _distance_information(
    result: BoundaryVectorRateResult | BoundaryVectorRatesResult,
) -> NDArray[np.float64]:
    """Skaggs information about wall distance along each unit's preferred direction.

    Each ray's distance bins partition the time a wall lay within
    ``distance_range`` in that direction, so Skaggs information is defined
    per direction; it is not defined over the whole map, whose occupancy
    counts every sample once per ray.

    Returns
    -------
    NDArray[np.float64], shape (n_units,)
        Bits per spike in the direction bin of each unit's peak rate; NaN
        for a unit with no rate.
    """
    from neurospatial.encoding._metrics import batch_spatial_information

    shape = (result.n_distance_bins, result.n_direction_bins)
    rates = np.atleast_2d(_to_numpy(result._get_rates()))
    occupancy = np.asarray(_to_numpy(result.occupancy), dtype=np.float64).reshape(shape)
    grids = rates.reshape(len(rates), *shape)
    # (n_units, n_direction_bins): information about distance per direction.
    per_direction = np.stack(
        [
            np.asarray(batch_spatial_information(grids[:, :, j], occupancy[:, j]))
            for j in range(shape[1])
        ],
        axis=1,
    )
    information = np.full(len(rates), np.nan)
    observed = np.isfinite(rates).any(axis=1)
    if observed.any():
        peak_bins = np.nanargmax(rates[observed], axis=1)
        columns = peak_bins % shape[1]  # distance-major flat bins
        information[observed] = per_direction[observed, columns]
    return information


def _boundary_summary(
    result: BoundaryVectorRateResult | BoundaryVectorRatesResult,
    *,
    index: Sequence[Any],
) -> pd.DataFrame:
    """The same metric columns for single and population results."""
    df = polar_summary_frame(
        result,
        index=index,
        extra={"distance_information": _distance_information(result)},
        extra_units={"distance_information": "bits/spike"},
        thresholds=None,
    )
    df.attrs["boundary_source"] = result.boundary_source
    return df


@_copy_analysis_settings
@dataclass(frozen=True, repr=False, eq=False)
class BoundaryVectorRateResult(_PolarRateMixin, SingleUnitRateMixin):
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
    direction_frame : {"allocentric", "egocentric"}
        Frame of the direction to the wall. Allocentric: 0 = East,
        +pi/2 = North. Egocentric: 0 = ahead, +pi/2 = left.
    boundary_source : {"physical", "analysis"}
        Which walls the rays stopped at.
    method : {"binned", "diffusion_kde", "gaussian_kde"}, default="binned"
        Estimator used. ``"binned"`` is the raw rate in each polar bin.
    bandwidth : float or None, default=None
        Kernel bandwidth in the environment's length units for the kernel
        methods; ``None`` for ``"binned"``, which does not smooth.
    unit_id : int or str or None
        Identifier for this unit; set by indexing a population result.

    Attributes
    ----------
    distance_range : tuple of float
        (min, max) distance to a wall covered by the bins; read from ``env``.
    n_distance_bins, n_direction_bins : int
        Number of distance and direction bins; read from ``env``.

    Other Parameters
    ----------------
    _unit_ids_generated : bool, default=False
        True when the unit labels were generated (no ``unit_ids`` or
        ``unit_id`` supplied), so pairing with spike inputs stays positional.
        Set by the compute functions and by indexing; not an analysis choice.
    n_spikes_outside_intervals : int or ndarray of int64, shape (n_units,), or None
        Each unit's spikes outside the analyzed time intervals
        (``analysis_settings["observed_intervals"]``), which are in no rate:
        an int for one unit, one count per unit for a population. ``None``
        when unknown, as for a result built by hand. Also a
        ``summary_table()`` column.
    min_occupancy : float, default=0.0
        Seconds a bin must be occupied to be in ``spatial_support``: the
        threshold the rates were estimated with. ``analysis_settings`` only
        records it.
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
    direction_frame: Literal["allocentric", "egocentric"] = field(kw_only=True)
    boundary_source: BoundarySource = field(kw_only=True)
    method: str = field(default="binned", kw_only=True)
    bandwidth: float | None = field(default=None, kw_only=True)
    unit_id: int | str | None = None
    min_occupancy: float = field(default=0.0, kw_only=True)
    n_spikes_outside_intervals: int | None = field(default=None, kw_only=True)
    spike_window: NDArray[np.float64] | None = field(
        default=None, kw_only=True, compare=False
    )
    analysis_settings: dict[str, Any] | None = field(
        default=None, kw_only=True, repr=False, compare=False
    )
    _unit_ids_generated: bool = field(
        default=False, kw_only=True, repr=False, compare=False
    )

    def _check_family(self) -> None:
        _check_boundary_labels(self.direction_frame, self.boundary_source)

    def _family_attrs(self) -> dict[str, Any]:
        return {"boundary_source": self.boundary_source}

    def summary_table(self) -> pd.DataFrame:
        """Per-unit metrics with the same columns as the population table.

        Returns
        -------
        pandas.DataFrame
            One row indexed by ``unit_id``: ``peak_firing_rate``,
            ``preferred_distance``, ``preferred_direction_deg``,
            ``preferred_direction`` and ``distance_information``. The frame
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

    def distance_information(self) -> float:
        """Information about wall distance along the preferred direction.

        Skaggs information, in bits per spike, of the rates along the
        direction bin that holds the peak, with that direction's boundary
        occupancy as the bin probabilities. Within one direction the
        distance bins partition the time a wall was in range, so this is a
        Skaggs information; over the whole map it is not, because every
        sample is counted once per ray.

        Returns
        -------
        float
            Bits per spike; NaN when no bin has a rate. Plug-in information
            is biased upward for few spikes, and choosing the direction by
            its peak adds to that bias, so compare it with shuffled spike
            trains before calling a cell boundary-tuned.
        """
        return float(_distance_information(self)[0])

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


@_copy_unit_table
@_copy_analysis_settings
@dataclass(frozen=True, repr=False, eq=False)
class BoundaryVectorRatesResult(
    _PolarRateMixin, PopulationRateMixin[BoundaryVectorRateResult]
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
    direction_frame : {"allocentric", "egocentric"}
        Frame of the direction to the wall.
    boundary_source : {"physical", "analysis"}
        Which walls the rays stopped at.
    unit_ids : array-like, optional
        One label per unit; ``np.arange(n_units)`` by default.
    unit_table : pandas.DataFrame, optional
        Per-unit metadata indexed by ``unit_id``.

    Attributes
    ----------
    distance_range : tuple of float
        (min, max) distance to a wall covered by the bins; read from ``env``.
    n_distance_bins, n_direction_bins : int
        Number of distance and direction bins; read from ``env``.

    Other Parameters
    ----------------
    _unit_ids_generated : bool, default=False
        True when the unit labels were generated (no ``unit_ids`` or
        ``unit_id`` supplied), so pairing with spike inputs stays positional.
        Set by the compute functions and by indexing; not an analysis choice.
    n_spikes_outside_intervals : int or ndarray of int64, shape (n_units,), or None
        Each unit's spikes outside the analyzed time intervals
        (``analysis_settings["observed_intervals"]``), which are in no rate:
        an int for one unit, one count per unit for a population. ``None``
        when unknown, as for a result built by hand. Also a
        ``summary_table()`` column.
    min_occupancy : float, default=0.0
        Seconds a bin must be occupied to be in ``spatial_support``: the
        threshold the rates were estimated with. ``analysis_settings`` only
        records it.
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
    direction_frame: Literal["allocentric", "egocentric"] = field(kw_only=True)
    boundary_source: BoundarySource = field(kw_only=True)
    method: str = field(default="binned", kw_only=True)
    bandwidth: float | None = field(default=None, kw_only=True)
    unit_ids: NDArray[Any] | Sequence[Any] | None = field(default=None, compare=False)
    unit_table: pd.DataFrame | None = field(default=None, compare=False)
    min_occupancy: float = field(default=0.0, kw_only=True)
    n_spikes_outside_intervals: NDArray[np.int64] | None = field(
        default=None, kw_only=True
    )
    spike_window: NDArray[np.float64] | None = field(
        default=None, kw_only=True, compare=False
    )
    analysis_settings: dict[str, Any] | None = field(
        default=None, kw_only=True, repr=False, compare=False
    )
    _unit_ids_generated: bool = field(
        default=False, kw_only=True, repr=False, compare=False
    )

    def _check_family(self) -> None:
        _check_boundary_labels(self.direction_frame, self.boundary_source)

    def _family_attrs(self) -> dict[str, Any]:
        return {"boundary_source": self.boundary_source}

    def __getitem__(self, idx: int) -> BoundaryVectorRateResult:
        """The single-unit result at position ``idx``; ``sel(unit_id=...)`` by label."""
        return self._unit_result(
            BoundaryVectorRateResult,
            idx,
            env=self.env,
            direction_frame=self.direction_frame,
            boundary_source=self.boundary_source,
            method=self.method,
            bandwidth=self.bandwidth,
        )

    def preferred_distances(self) -> NDArray[np.float64]:
        """Distance to the wall at each unit's peak rate, shape (n_units,)."""
        return self._peak_coordinates(0)

    def preferred_directions(self) -> NDArray[np.float64]:
        """Direction to the wall at each unit's peak rate, shape (n_units,).

        In radians in ``direction_frame``; NaN for a unit with no rate.
        """
        return self._peak_coordinates(1)

    def distance_information(self) -> NDArray[np.float64]:
        """Information about wall distance along each unit's preferred direction.

        See :meth:`BoundaryVectorRateResult.distance_information`.

        Returns
        -------
        NDArray[np.float64], shape (n_units,)
            Bits per spike; NaN for a unit with no rate.
        """
        return _distance_information(self)

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
    boundary_source: BoundarySource,
) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
    """Polar bin of the wall along each ray, and which samples were observed.

    Returns
    -------
    frame_bins : NDArray[np.intp], shape (n_samples, n_direction_bins)
        Flat polar bin of each ray's wall; ``BEYOND_RANGE`` when the wall is
        at or beyond ``distance_range[1]``, and -1 when the ray meets no
        wall, the wall is nearer than ``distance_range[0]`` or the sample is
        missing.
    observed : NDArray[np.bool_], shape (n_samples,)
        Samples with a position inside the walls (and a heading, when used).
    """
    from neurospatial.encoding._egocentric_counts import (
        BEYOND_RANGE,
        _coords_to_flat_bin_idx,
    )
    from neurospatial.ops.boundary_vectors import compute_boundary_distances

    # Search every wall, not only those in range, so the time a wall lay
    # beyond the range is known.
    distances, directions = compute_boundary_distances(
        env,
        positions,
        headings,
        direction_frame=direction_frame,
        n_directions=n_direction_bins,
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
    with np.errstate(invalid="ignore"):
        frame_bins[np.isfinite(distances) & (distances >= distance_range[1])] = (
            BEYOND_RANGE
        )
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
    break_before: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, TimeSupport]:
    """Spike counts and boundary occupancy summed over every ray.

    An unobserved sample is skipped as if its row were deleted: an interval
    counts when the gap between the observed samples around it passes the
    gap, window and declared-break checks; it then adds its duration, and each
    of its spikes one count, to the bin of every ray of the observed sample
    that starts the gap and met a wall in range. Spikes and occupancy use the
    same intervals.

    Returns
    -------
    spike_counts : NDArray[np.float64], shape (n_units, n_bins)
    occupancy : NDArray[np.float64], shape (n_bins,)
    time_beyond : float
        Seconds of the same intervals, summed over rays, in which a ray's
        wall lay at or beyond the range.
    support : TimeSupport
        The retained intervals, for the result's analysis settings.
    """
    from neurospatial._intervals import hold_last_valid, warn_excluded_intervals
    from neurospatial.encoding._egocentric_counts import BEYOND_RANGE
    from neurospatial.encoding._spatial_counts import prepare_frame_binning

    prepared = prepare_frame_binning(
        times,
        np.where(observed, 0, -1).astype(np.intp),
        n_bins,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        spike_window=spike_window,
        sample_valid=observed,
    )
    frame_bins = hold_last_valid(frame_bins, observed, fill=-1)
    support = prepared.support
    mask = support.mask
    warn_excluded_intervals(
        times,
        mask,
        max_gap=support.max_gap,
        epochs=support.epochs,
        spike_window=support.spike_window,
        start_bin_gate=True,
        break_before=break_before,
        result="the rate map is empty",
    )
    starts = frame_bins[:-1][mask]  # (n_kept_intervals, n_rays)
    durations = np.broadcast_to(np.diff(times)[mask][:, None], starts.shape)
    hit = starts >= 0
    occupancy = np.bincount(
        starts[hit], weights=durations[hit], minlength=n_bins
    ).astype(np.float64)
    time_beyond = float(durations[starts == BEYOND_RANGE].sum())

    spike_counts = np.zeros((len(spike_trains), n_bins), dtype=np.float64)
    for unit, spikes in enumerate(spike_trains):
        spikes = spikes[(spikes >= times[0]) & (spikes < times[-1])]
        frames = np.searchsorted(times, spikes, side="right") - 1
        rays = frame_bins[frames[mask[frames]]]
        spike_counts[unit] = np.bincount(rays[rays >= 0], minlength=n_bins)
    return spike_counts, occupancy, time_beyond, support


def compute_boundary_vector_rates(
    env: Environment,
    spike_times: SpikeTrainsLike,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64] | None = None,
    *,
    direction_frame: Literal["allocentric", "egocentric"],
    distance_range: tuple[float, float] | None = None,
    distance_bin_size: float | None = None,
    direction_bin_size: float = DEFAULT_DIRECTION_BIN_SIZE,
    boundary_source: BoundarySource = "physical",
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
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
    distance_range : tuple of float or None, default=None
        (min, max) distance to a wall, in the environment's units. A wall
        farther along a ray than the maximum adds nothing for that ray.
        None uses 0-50 cm converted to ``env.units``, the range of Hinman
        et al. (2019); with units that are not a known length, pass it
        explicitly. The range does not cover the whole arena on purpose:
        the walls on either side of the animal are a chord apart, so firing
        tied to a wall at distance d in one direction is equally tied to the
        opposite wall at (chord - d), and a grid reaching the far walls lets
        the peak jump to them (see Notes).
        ``analysis_settings["time_beyond_distance_range"]`` records the
        seconds, summed over rays like the occupancy, in which a ray's wall
        lay beyond the range, and
        ``analysis_settings["fraction_of_time_in_distance_range"]`` the
        share of that ray time the map covers (see Notes).
    distance_bin_size : float or None, default=None
        Width of each distance ring, in the environment's units. None uses
        5 cm converted to ``env.units``; pass it when the units are unknown.
        An explicit ``distance_range`` is divided into the whole number of
        rings nearest this width.
    direction_bin_size : float, default=pi/6
        Width of each direction bin in radians (30 degrees); the circle is
        divided into ``round(2 pi / direction_bin_size)`` bins and one ray
        is cast at each bin's center. The 5 cm and 30 degree defaults suit
        unsmoothed maps of 15-minute sessions (see Notes).
    boundary_source : {"physical", "analysis"}, default="physical"
        ``"physical"`` stops rays at the arena's declared outline and walls;
        ``"analysis"`` at the edge of the environment's bins.
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open windows (seconds).
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from occupancy and its spikes are not counted, as for a gap
        longer than ``max_gap``. ``analysis_settings["break_times"]`` records
        ``times[k]`` of each break (``None`` when not given).
    spike_window : same forms as epochs, or None
        When the electrophysiology was recording.
    method : {"binned", "diffusion_kde", "gaussian_kde"}, default="binned"
        ``"binned"`` divides counts by occupancy per bin. The kernel methods
        smooth over the polar grid, which also mixes neighboring distances.
    bandwidth : float, optional
        Smoothing bandwidth for the kernel methods, in the environment's
        units. None uses 5 cm converted to ``env.units``; with units that
        are not a known length, a kernel method needs it explicitly.
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

    The default grid was checked on simulated cells in 15-minute sessions in
    a 100 cm square and a 100 cm circle: Hartley-style boundary-vector and
    egocentric boundary cells (preferred distances 5-35 cm) and
    ``BoundaryVectorCellModel`` cells, with unsmoothed maps. With 5 cm rings
    and 30 degree bins over 0-50 cm, the preferred distance was within one
    ring (2.5 cm) of the truth and the preferred direction within 17 degrees
    for 90% of cells (median 9 degrees); 15 degree bins narrowed the
    direction error to 13 degrees, and 2.5 cm rings did not improve the
    distance. The same grid out to the square's diagonal (0-145 cm) put 45%
    of allocentric cells' preferred direction more than 45 degrees off,
    usually on the opposite wall: in a box, firing tied to one wall is
    equally tied to the wall across from it.

    A field can still continue past the range. Unlike object-vector maps, no
    warning flags a peak in the outermost ring: with walls beyond 50 cm in most
    directions, untuned cells peak there by chance (16% of flat, 31% of
    head-direction and 6% of place cells in the simulations above, against none
    of the boundary cells). ``analysis_settings["time_beyond_distance_range"]``
    and ``analysis_settings["fraction_of_time_in_distance_range"]`` record how
    much ray time the range left out; for a cell whose preferred distance is in
    the last ring, widen ``distance_range`` and check that the peak stays on the
    near wall.

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
    ['peak_firing_rate', 'preferred_distance', 'preferred_direction_deg', 'preferred_direction', 'distance_information', 'n_spikes_outside_intervals']
    """
    return _boundary_vector_rates(
        env,
        spike_times,
        times,
        positions,
        headings,
        direction_frame=direction_frame,
        distance_range=distance_range,
        distance_bin_size=distance_bin_size,
        direction_bin_size=direction_bin_size,
        boundary_source=boundary_source,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
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
    distance_range: tuple[float, float] | None = None,
    distance_bin_size: float | None = None,
    direction_bin_size: float = DEFAULT_DIRECTION_BIN_SIZE,
    boundary_source: BoundarySource = "physical",
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
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
        distance_bin_size=distance_bin_size,
        direction_bin_size=direction_bin_size,
        boundary_source=boundary_source,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        spike_window=spike_window,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        unit_ids=None,
        context="compute_boundary_vector_rate",
    )
    unit = population[0]
    return BoundaryVectorRateResult(
        min_occupancy=unit.min_occupancy,
        firing_rate=unit.firing_rate,
        occupancy=unit.occupancy,
        env=unit.env,
        direction_frame=unit.direction_frame,
        boundary_source=unit.boundary_source,
        method=unit.method,
        bandwidth=unit.bandwidth,
        n_spikes_outside_intervals=unit.n_spikes_outside_intervals,
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
    distance_range: tuple[float, float] | None,
    distance_bin_size: float | None,
    direction_bin_size: float,
    boundary_source: BoundarySource,
    max_gap: float | None,
    epochs: Any,
    break_before: ArrayLike | None,
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
    from neurospatial.encoding._provenance import (
        rate_analysis_settings,
        spikes_outside_intervals,
    )
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
    from neurospatial.encoding.egocentric import (
        _raw_polar_rate,
        _require_planar_env,
    )

    _validate_env_argument(
        env, context=context, arguments="spike_times, times, positions[, headings]"
    )
    _require_planar_env(
        env, f"{context}()", measures="distance and direction to the walls"
    )
    _check_boundary_labels(direction_frame, boundary_source)
    _validate_smoothing_parameters(method, 1.0 if bandwidth is None else bandwidth)
    distance_range, n_distance_bins, n_direction_bins = resolve_polar_grid(
        env.units,
        distance_range=_resolve_wall_range(env.units, distance_range, context=context),
        largest_distance=None,
        distance_bin_size=distance_bin_size,
        direction_bin_size=direction_bin_size,
        context=context,
    )
    require_known_bandwidth(env.units, bandwidth, method, context=context)
    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)

    spike_trains, resolved_unit_ids, labels_supplied = as_labelled_spike_trains(
        spike_times, unit_ids, context=context
    )
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    headings = None if headings is None else np.asarray(headings, dtype=np.float64)
    validate_trajectory(
        times, positions=positions, headings=headings, context=context, n_dims=2
    )
    validate_spike_trains(spike_trains, context=context)
    breaks = validate_break_before(break_before, len(times), call=context)
    _warn_if_population_silent(
        spike_trains,
        times,
        max_gap=max_gap,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
        break_before=breaks,
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
    spike_counts, occupancy, time_beyond, support = _count_boundary_events(
        spike_trains,
        times,
        frame_bins,
        observed,
        polar_env.n_bins,
        max_gap=max_gap,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
        break_before=breaks,
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
    time_in_range = float(occupancy.sum())
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
        time_beyond_distance_range=time_beyond,
        fraction_of_time_in_distance_range=(
            time_in_range / (time_in_range + time_beyond)
            if time_in_range + time_beyond > 0
            else float("nan")
        ),
    )
    return BoundaryVectorRatesResult(
        min_occupancy=min_occupancy,
        firing_rates=firing_rates,
        occupancy=occupancy,
        env=polar_env,
        direction_frame=direction_frame,
        boundary_source=boundary_source,
        method=method,
        bandwidth=resolved_bandwidth,
        unit_ids=resolved_unit_ids if labels_supplied else None,
        unit_table=getattr(spike_times, "unit_table", None),
        n_spikes_outside_intervals=spikes_outside_intervals(spike_trains, support),
        spike_window=resolved_spike_window,
        analysis_settings=settings,
    )
