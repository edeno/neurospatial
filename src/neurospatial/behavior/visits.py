"""Observed visits to named regions, from tracked positions.

A visit is time at a region, from the first sample inside it to the first
sample outside. Visits describe tracking observations, without task trials,
rewards, goals or success; :mod:`neurospatial.behavior.trials` builds trials
from them. Recording boundaries remain explicit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from operator import index
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._intervals import (
    DEFAULT_MAX_GAP,
    as_intervals,
    start_allocation,
    times_in_intervals,
    validate_max_gap,
    warn_excluded_intervals,
)
from neurospatial._observation import validate_break_before
from neurospatial._pynapple import reads_pynapple_samples
from neurospatial.behavior._position_bins import sample_bins
from neurospatial.ops.mapping import _require_region_mask

if TYPE_CHECKING:
    from neurospatial._interval_table import Intervals
    from neurospatial.environment import Environment

__all__ = [
    "RegionVisit",
    "detect_region_visits",
]


def _store_plain_scalars(
    record: Any,
    *,
    ints: Sequence[str] = (),
    floats: Sequence[str] = (),
    bools: Sequence[str] = (),
    strings: Sequence[str] = (),
) -> None:
    """Replace NumPy scalars on a frozen record with plain Python scalars.

    Integers are converted with ``operator.index`` so a fractional index
    raises instead of being truncated.
    """
    record_name = type(record).__name__
    for name in ints:
        value = getattr(record, name)
        if isinstance(value, bool | np.bool_):
            msg = f"{record_name}.{name} must be an integer, got {value!r}."
            raise TypeError(msg)
        try:
            object.__setattr__(record, name, index(value))
        except TypeError:
            msg = f"{record_name}.{name} must be an integer, got {value!r}."
            raise TypeError(msg) from None
    for name in floats:
        object.__setattr__(record, name, float(getattr(record, name)))
    for name in bools:
        value = getattr(record, name)
        if not isinstance(value, bool | np.bool_):
            msg = f"{record_name}.{name} must be a bool, got {value!r}."
            raise TypeError(msg)
        object.__setattr__(record, name, bool(value))
    for name in strings:
        value = getattr(record, name)
        if not isinstance(value, str):
            msg = f"{record_name}.{name} must be a str, got {value!r}."
            raise TypeError(msg)
        object.__setattr__(record, name, str(value))


@dataclass(frozen=True)
class RegionVisit:
    """One uninterrupted observation of membership in a named region.

    Attributes
    ----------
    visit_id, run_id : int
        Chronological visit and continuous-observation identifiers, local to
        one detection call. Run identifiers also count runs without visits.
    region : str
        Name of the region in ``env.regions``.
    start_index, stop_index : int
        Bounds of the inside-region samples in the original input arrays:
        ``positions[start_index:stop_index]``. Stop is exclusive.
    start_time, stop_time : float
        Observed bounds in seconds; the visit is the half-open window
        ``[start_time, stop_time)``. Start is the first inside sample. Stop is
        the first outside sample for an observed exit, otherwise the last
        continuous observation boundary. A complete interval ending exactly
        at an epoch stop can supply that boundary without creating an exit.
    entry_observed, exit_observed : bool
        Whether the corresponding crossing was observed within one recording
        and the requested half-open epochs. False denotes censoring, not a
        failed visit. Starting inside a region never invents an entry.
    duration : float
        ``stop_time - start_time`` in seconds, using start-sample allocation.
        This is only observed dwell time when a boundary is censored; a
        singleton observation has zero duration.

    Notes
    -----
    Records contain only plain Python scalars and are immutable; construction
    checks ``start_index < stop_index`` and ``start_time <= stop_time``.
    """

    visit_id: int
    run_id: int
    region: str
    start_index: int
    stop_index: int
    start_time: float
    stop_time: float
    entry_observed: bool
    exit_observed: bool
    duration: float = field(init=False)

    def __post_init__(self) -> None:
        _store_plain_scalars(
            self,
            ints=("visit_id", "run_id", "start_index", "stop_index"),
            floats=("start_time", "stop_time"),
            bools=("entry_observed", "exit_observed"),
            strings=("region",),
        )
        if not self.start_index < self.stop_index:
            msg = (
                "RegionVisit needs start_index < stop_index (stop is exclusive and "
                f"a visit has at least one sample); got {self.start_index} and "
                f"{self.stop_index}."
            )
            raise ValueError(msg)
        if not self.start_time <= self.stop_time:
            msg = (
                "RegionVisit needs start_time <= stop_time; got "
                f"{self.start_time} and {self.stop_time}."
            )
            raise ValueError(msg)
        object.__setattr__(self, "duration", self.stop_time - self.start_time)


@dataclass(frozen=True)
class _ObservedVisits:
    """Visits of one detection call, with the end time of each observed run."""

    visits: list[RegionVisit]
    run_end_times: dict[int, float]
    settings: dict[str, Any]


def _observed_visits(
    env: Environment,
    times: ArrayLike,
    positions: ArrayLike,
    *,
    regions: Sequence[str],
    max_gap: float | None,
    epochs: Any,
    break_before: ArrayLike | None,
    call: str,
) -> _ObservedVisits:
    """Detect visits, keeping each run's end for trials censored by it."""
    from neurospatial._exceptions import _format_error

    clock, bins = sample_bins(env, times, positions, call=call)
    validate_max_gap(max_gap)
    windows = as_intervals(epochs, name="epochs")
    if isinstance(regions, str):
        msg = f"{call}(): regions must be a sequence of names, for example ['left', 'right']."
        raise TypeError(msg)
    names = list(regions)
    if not names:
        raise ValueError(
            _format_error(
                f"{call}(): regions is empty.",
                why="visits are found by the regions the animal enters.",
                fix="pass the names of regions in env.regions, for example "
                "regions=['well_1', 'well_2'].",
            )
        )
    if any(not isinstance(name, str) for name in names):
        msg = f"{call}(): regions must contain strings from env.regions."
        raise TypeError(msg)
    if len(set(names)) != len(names):
        msg = f"{call}(): regions must not contain duplicate names."
        raise ValueError(msg)
    breaks = validate_break_before(break_before, clock.size, call=call)

    # Resolve every bin's exclusive label before looking at this trajectory:
    # unvisited overlap is still ambiguous geometry, not a name-order policy.
    labels = np.full(env.n_bins, -1, dtype=np.intp)
    for label, name in enumerate(names):
        mask = _require_region_mask(
            env, name, operation=f"{call}()", argument="regions", in_list=True
        )
        if env.regions[name].kind == "point":
            raise ValueError(
                _format_error(
                    f"{call}(): region {name!r} is a point, one bin.",
                    why="an animal that stops beside a well, not on its bin, is "
                    "never inside a point region, so visits and the trials "
                    "between them would be missed or end late.",
                    fix=f"give it the size of the place where the animal stops: "
                    f"regions.add({name!r}, point=..., radius=5.0), or a polygon.",
                )
            )
        overlap = mask & (labels >= 0)
        if overlap.any():
            other = sorted({names[index] for index in labels[overlap]})
            msg = (
                f"Selected region {name!r} overlaps {other} in {overlap.sum()} active bin(s). "
                "Use disjoint bin-center memberships; region order does not resolve overlaps."
            )
            raise ValueError(msg)
        labels[mask] = label

    settings: dict[str, Any] = {
        "regions": names,
        "max_gap": None if max_gap is None else float(max_gap),
        "epochs": None if windows is None else windows.tolist(),
        "break_before_given": breaks is not None,
    }
    # A sample at bin -1 is not located; as in env.occupancy, an interval
    # with either sample there is not observed (the complete-case rule).
    valid = bins >= 0
    selected = valid.copy()
    if windows is not None:
        selected &= times_in_intervals(windows, clock)
    if not selected.any():
        return _ObservedVisits([], {}, settings)

    # A terminal epoch boundary can close exposure, but cannot create an event.
    gates = start_allocation(
        clock, bins, None, max_gap=max_gap, epochs=windows, break_before=breaks
    ).gates
    warn_excluded_intervals(gates, result="every visit ends at its first sample")
    continuous = gates.mask
    connected = continuous & selected[:-1] & selected[1:]
    starts = np.flatnonzero(selected & np.r_[True, ~connected])
    stops = np.flatnonzero(selected & np.r_[~connected, True]) + 1
    sample_labels = np.full(bins.shape, -1, dtype=np.intp)
    sample_labels[valid] = labels[bins[valid]]

    visits: list[RegionVisit] = []
    run_end_times: dict[int, float] = {}
    for run_id, (run_start, run_stop) in enumerate(zip(starts, stops, strict=True)):
        run_end_time = clock[run_stop - 1]
        if run_stop < len(clock) and continuous[run_stop - 1]:
            run_end_time = clock[run_stop]
        run_end_times[run_id] = float(run_end_time)
        run_labels = sample_labels[run_start:run_stop]
        changes = np.flatnonzero(np.r_[True, np.diff(run_labels) != 0, True])
        for first, stop in pairwise(changes):
            label = int(run_labels[first])
            if label < 0:
                continue
            start_index, stop_index = int(run_start + first), int(run_start + stop)
            exit_observed = bool(stop_index < run_stop)
            visits.append(
                RegionVisit(
                    visit_id=len(visits),
                    run_id=run_id,
                    region=names[label],
                    start_index=start_index,
                    stop_index=stop_index,
                    start_time=float(clock[start_index]),
                    stop_time=float(
                        clock[stop_index] if exit_observed else run_end_time
                    ),
                    entry_observed=bool(start_index > run_start),
                    exit_observed=exit_observed,
                )
            )
    return _ObservedVisits(visits, run_end_times, settings)


@reads_pynapple_samples
def detect_region_visits(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    regions: Sequence[str],
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> Intervals[RegionVisit]:
    """Detect observed visits to named regions, such as reward wells.

    A visit is the time the animal is inside a region: from the first sample
    inside to the first sample outside. For the runs between visits (trials
    from one well to the next), use :func:`detect_trials`.

    Parameters
    ----------
    env : Environment
        Allocentric environment holding the named regions, each with an
        area (a polygon, or ``regions.add(name, point=..., radius=...)``).
    times : ndarray, shape (n_samples,)
        Finite, strictly increasing timestamps in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Positions in the environment's coordinates. A NaN position (lost
        tracking) or one outside the environment is not located: as in
        ``env.occupancy``, an interval touching it is not observed, so it
        ends the run and censors the visits around it. To bridge short
        dropouts instead, delete their rows; a deleted row is bridged when
        the samples around it are no more than ``max_gap`` apart.
    regions : sequence of str
        Nonempty, unique region names. Bin-center membership follows
        ``regions_to_mask``: regions that only share edges, such as
        quadrants, split the bins on those edges and never share a bin.
        Regions that overlap in area may not share active bins; ambiguous
        membership raises rather than using the order of names as a
        priority rule. Point regions are refused (see Raises).
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), Intervals, or None
        Half-open event windows on the same clock as ``times``. Only fully
        contained sampling intervals contribute duration. An entry at an
        epoch stop is excluded; the interval ending at that stop can still
        contribute observed dwell.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    Intervals of RegionVisit
        One row per visit in time order, ``interval_ids`` the visit ids, with
        the fields ``region``, ``entry_observed``, ``exit_observed``, sample
        indices and ``run_id``. Censored visits at recording boundaries are
        kept; a visit seen in one sample has zero duration (an observation
        with no time, absent from ``windows``). ``analysis_settings`` records
        the regions, ``max_gap`` and ``epochs``.

    Raises
    ------
    ValueError
        For invalid times or positions, gap thresholds, duplicate or empty
        names, overlapping selected-region bins, a point region (one bin: an
        animal stopped beside it is never inside; give it a ``radius``), or a
        ``break_before`` that is not a Boolean array with one entry per
        sample.
    TypeError
        If names are not strings.
    RegionNotFoundError
        If a selected region does not exist.

    Notes
    -----
    An entry is observed only when a preceding continuous sample was outside
    that region. An exit is observed at the first following outside sample.
    These are sampled crossing times, without sub-frame interpolation. A visit
    starting inside a region after a gap or a declared break is left-censored,
    not a new entry. An unlocated sample, like a gap longer than ``max_gap``,
    censors both adjacent runs, so it never fabricates an exit.

    See Also
    --------
    detect_trials : The runs between consecutive visits.
    trials_between_visits : Trials from visits you bring (for example port
        events).
    """
    from neurospatial._interval_table import Intervals

    observed = _observed_visits(
        env,
        times,
        positions,
        regions=regions,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        call="detect_region_visits",
    )
    return Intervals(
        observed.visits,
        record_type=RegionVisit,
        interval_ids=[v.visit_id for v in observed.visits],
        analysis_settings=observed.settings,
    )
