"""Observed visits and entry-to-entry transitions between named regions.

These events describe tracking observations, without assigning task trials,
rewards, intended goals, or success. Recording boundaries remain explicit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._intervals import (
    as_intervals,
    interval_valid_mask,
    times_in_intervals,
    validate_max_gap,
    warn_excluded_intervals,
)
from neurospatial._observation import validate_break_before
from neurospatial.ops.mapping import _require_region_mask

if TYPE_CHECKING:
    from neurospatial.environment import Environment

__all__ = [
    "RegionTransition",
    "RegionVisit",
    "detect_region_visits",
    "region_transitions",
]


@dataclass(frozen=True)
class RegionVisit:
    """One uninterrupted observation of membership in a named region.

    Attributes
    ----------
    visit_id, run_id : int
        Chronological visit and continuous-observation identifiers, local to
        one detection call. Run identifiers also count runs without visits.
    region_name : str
        Original name from ``env.regions``.
    start_index, stop_index : int
        Bounds of the inside-region samples in the original input arrays:
        ``position_bins[start_index:stop_index]``. Stop is exclusive.
    start_time, end_time : float
        Observed bounds in seconds. Start is the first inside sample. End is
        the first outside sample for an observed exit, otherwise the last
        continuous observation boundary. A complete interval ending exactly
        at an epoch stop can supply that boundary without creating an exit.
    entry_observed, exit_observed : bool
        Whether the corresponding crossing was observed within one recording
        and the requested half-open epochs. False denotes censoring, not a
        failed visit. Starting inside a region never invents an entry.
    duration : float
        ``end_time - start_time`` in seconds, using start-sample allocation.
        This is only observed dwell time when a boundary is censored; a
        singleton observation has zero duration.

    Notes
    -----
    Records contain only scalars and are immutable. ``pandas.DataFrame(visits)``
    converts a nonempty list into an event table, including censoring flags.
    """

    visit_id: int
    run_id: int
    region_name: str
    start_index: int
    stop_index: int
    start_time: float
    end_time: float
    entry_observed: bool
    exit_observed: bool
    duration: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration", self.end_time - self.start_time)


@dataclass(frozen=True)
class RegionTransition:
    """Consecutive observed entries into two different named regions.

    Attributes
    ----------
    source_visit_id, target_visit_id : int
        Identifiers of the original visits, retained without renumbering.
    run_id : int
        Shared continuous-observation identifier, local to the detection call.
    source_region, target_region : str
        Names of the source and destination regions.
    start_index, end_index : int
        Original sample indices of the source and target entries. Both samples
        belong to the transition path; use ``slice(start_index, end_index + 1)``
        for plotting arrival. These are not compressed visit-list indices.
    start_time, end_time : float
        Source and target entry timestamps in seconds. Adjacent transitions
        may share an entry timestamp. As analysis epochs, use the half-open
        ``[start_time, end_time)`` interval to avoid double counting that event.
    duration : float
        Entry-to-entry duration in seconds, including source-region dwell.
        This is not departure-to-arrival travel time or a task-trial outcome.
    """

    source_visit_id: int
    target_visit_id: int
    run_id: int
    source_region: str
    target_region: str
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    duration: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration", self.end_time - self.start_time)


def detect_region_visits(
    position_bins: NDArray[np.int64],
    times: NDArray[np.float64],
    env: Environment,
    *,
    region_names: Sequence[str],
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> list[RegionVisit]:
    """Detect observed visits to disjoint named regions.

    Parameters
    ----------
    position_bins : ndarray of integers, shape (n_samples,)
        Sample-aligned bins from ``env.bin_at(positions)``. Invalid bins
        (negative or at least ``env.n_bins``) denote unknown observations and
        break continuity. A valid bin outside every selected region is observed
        travel, not an observation break.
    times : ndarray, shape (n_samples,)
        Finite, strictly increasing timestamps in seconds.
    env : Environment
        Environment containing the named polygon regions.
    region_names : sequence of str
        Nonempty, unique region names. Bin-center membership follows
        ``regions_to_mask`` with boundaries included. Selected regions may
        not share active bins; ambiguous membership raises rather than using
        the order of names as a priority rule.
    max_gap : float or None, default=0.5
        Maximum interval in seconds treated as continuous. Equality passes.
        None disables the gap threshold, but not invalid-bin or epoch breaks.
    epochs : (start, stop), array-like of shape (n, 2), or None
        Half-open event windows on the same clock as ``times``. IntervalSet-like
        inputs are also accepted. Only fully contained sampling intervals
        contribute duration. An entry at an epoch stop is excluded; the
        interval ending at that stop can still contribute observed dwell.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored.

    Returns
    -------
    list[RegionVisit]
        Chronological immutable scalar records, including censored visits at
        recording boundaries. Original sample indices are preserved. Empty
        when no selected region was observed. No raw input arrays are retained.

    Raises
    ------
    ValueError
        For invalid shapes, clocks, gap thresholds, duplicate/empty names,
        overlapping selected-region bins, or a ``break_before`` that is not a
        Boolean array with one entry per sample.
    TypeError
        If ``position_bins`` is not an integer array or names are not strings.
    RegionNotFoundError
        If a selected region does not exist.

    Notes
    -----
    An entry is observed only when a preceding continuous sample was outside
    that region. An exit is observed at the first following outside sample.
    These are sampled crossing times, without sub-frame interpolation. A visit
    starting inside a region after a gap or a declared break is left-censored,
    not a new entry. Invalid observations censor both adjacent runs and never
    fabricate exits.

    Use :func:`region_transitions` before filtering visits to obtain consecutive
    entry-to-entry pairs. Use ``detect_runs_between_regions`` instead when the
    desired interval starts at source exit.
    """
    bins = np.asarray(position_bins)
    clock = np.asarray(times, dtype=np.float64)
    if bins.ndim != 1 or clock.ndim != 1 or bins.shape != clock.shape:
        raise ValueError(
            "position_bins and times must be 1-D arrays of equal length. "
            f"Got {bins.shape} and {clock.shape}."
        )
    if bins.size and bins.dtype.kind not in "iu":
        raise TypeError(
            "position_bins must contain integer bin indices from env.bin_at()."
        )
    if not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0):
        raise ValueError("times must be finite and strictly increasing, in seconds.")
    validate_max_gap(max_gap)
    windows = as_intervals(epochs, name="epochs")
    if isinstance(region_names, str):
        raise TypeError(
            "region_names must be a sequence of names, for example ['left', 'right']."
        )
    names = list(region_names)
    if not names:
        raise ValueError("region_names must contain at least one named region.")
    if any(not isinstance(name, str) for name in names):
        raise TypeError("region_names must contain strings from env.regions.")
    if len(set(names)) != len(names):
        raise ValueError("region_names must not contain duplicate names.")
    breaks = validate_break_before(
        break_before, clock.size, call="detect_region_visits"
    )

    # Resolve every bin's exclusive label before looking at this trajectory:
    # unvisited overlap is still ambiguous geometry, not a name-order policy.
    labels = np.full(env.n_bins, -1, dtype=np.intp)
    for label, name in enumerate(names):
        mask = _require_region_mask(
            env,
            name,
            operation="detect_region_visits()",
            argument="region_names",
            in_list=True,
        )
        overlap = mask & (labels >= 0)
        if overlap.any():
            other = sorted({names[index] for index in labels[overlap]})
            raise ValueError(
                f"Selected region {name!r} overlaps {other} in {overlap.sum()} active bin(s). "
                "Use disjoint bin-center memberships; region order does not resolve overlaps."
            )
        labels[mask] = label

    valid = (bins >= 0) & (bins < env.n_bins)
    selected = valid.copy()
    if windows is not None:
        selected &= times_in_intervals(windows, clock)
    if not selected.any():
        return []

    # A terminal epoch boundary can close exposure, but cannot create an event.
    gates = interval_valid_mask(
        clock, max_gap=max_gap, epochs=windows, break_before=breaks
    )
    warn_excluded_intervals(
        clock,
        gates,
        max_gap=max_gap,
        epochs=windows,
        break_before=breaks,
        result="every visit ends at its first sample",
    )
    continuous = valid[:-1] & valid[1:] & gates
    connected = continuous & selected[:-1] & selected[1:]
    starts = np.flatnonzero(selected & np.r_[True, ~connected])
    stops = np.flatnonzero(selected & np.r_[~connected, True]) + 1
    sample_labels = np.full(bins.shape, -1, dtype=np.intp)
    sample_labels[valid] = labels[bins[valid]]

    visits: list[RegionVisit] = []
    for run_id, (run_start, run_stop) in enumerate(zip(starts, stops, strict=True)):
        run_end_time = clock[run_stop - 1]
        if run_stop < len(clock) and continuous[run_stop - 1]:
            run_end_time = clock[run_stop]
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
                    region_name=names[label],
                    start_index=start_index,
                    stop_index=stop_index,
                    start_time=float(clock[start_index]),
                    end_time=float(
                        clock[stop_index] if exit_observed else run_end_time
                    ),
                    entry_observed=bool(start_index > run_start),
                    exit_observed=exit_observed,
                )
            )
    return visits


def region_transitions(visits: Sequence[RegionVisit]) -> list[RegionTransition]:
    """Pair consecutive observed entries into different regions in one run.

    Parameters
    ----------
    visits : sequence of RegionVisit
        Chronological records from one :func:`detect_region_visits` call.
        Build transitions before filtering visits. Omitted visits are never
        bridged: paired visit identifiers must also be consecutive.

    Returns
    -------
    list[RegionTransition]
        Entry-to-entry pairs with both entries observed, in one continuous run,
        and different source/destination names. A destination's censored exit
        does not invalidate its observed entry. ``A, A, B`` yields only the
        final ``A -> B``, without merging the two A visits. Durations include
        source dwell time. An empty input produces an empty list.

    Raises
    ------
    ValueError
        If records are duplicated or are not in their original chronological
        visit order. Identifiers are local to a call; do not concatenate
        unrelated recordings as if they shared event identity.

    See Also
    --------
    detect_region_visits : Detect visits and censoring from tracking.
    """
    visits = list(visits)
    if any(not isinstance(visit, RegionVisit) for visit in visits):
        raise TypeError(
            "visits must contain RegionVisit records from detect_region_visits()."
        )
    pairs = list(pairwise(visits))
    if any(
        target.visit_id <= source.visit_id or target.start_time < source.start_time
        for source, target in pairs
    ):
        raise ValueError(
            "visits must retain their original chronological visit order without duplicates."
        )
    return [
        RegionTransition(
            source_visit_id=source.visit_id,
            target_visit_id=target.visit_id,
            run_id=source.run_id,
            source_region=source.region_name,
            target_region=target.region_name,
            start_index=source.start_index,
            end_index=target.start_index,
            start_time=source.start_time,
            end_time=target.start_time,
        )
        for source, target in pairs
        if source.run_id == target.run_id
        and target.visit_id == source.visit_id + 1
        and source.region_name != target.region_name
        and source.entry_observed
        and target.entry_observed
    ]
