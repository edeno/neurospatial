"""Trials: the runs between consecutive visits to named regions.

An animal on a task alternates between being at a region (a visit, such as
time at a reward well) and running between regions (a trial). A trial is
the half-open window from leaving one region to arriving at the next,
whichever it is, so tasks with one start (a T-maze) and with several (a
W-track alternation) are the same.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import DEFAULT_MAX_GAP
from neurospatial._pynapple import reads_pynapple_samples
from neurospatial.behavior.segmentation import Trial
from neurospatial.behavior.visits import _observed_visits

if TYPE_CHECKING:
    from neurospatial._interval_table import Intervals
    from neurospatial.environment import Environment

__all__ = ["detect_trials", "trials_between_visits"]


def _trial(
    departure: float, arrival: float, start_region: str, end_region: str
) -> Trial:
    """The trial from leaving ``start_region`` to arriving at ``end_region``."""
    same = start_region == end_region
    return Trial(
        start_time=departure,
        stop_time=arrival,
        start_region=start_region,
        end_region=None if same else end_region,
        outcome="returned" if same else "reached",
    )


def trials_between_visits(
    visits: Intervals[Any], *, region: str = "region"
) -> Intervals[Trial]:
    """Trials between consecutive visits you already have.

    Use this when the visits come from hardware rather than tracking: a table
    of port or well visits (poke in and out, beam break on and off) read with
    ``Intervals.from_dataframe``. Each pair of consecutive visits gives one
    trial, from leaving the first region to arriving at the next.

    Parameters
    ----------
    visits : Intervals
        One row per visit, ``[arrival, departure)`` in seconds, with a field
        naming the region. The names must be the environment's region names
        when the trials are used with it (map port labels first, for example
        ``pokes["port"].map({"Poke_1": "well_1", ...})``).
    region : str, default="region"
        The field holding each visit's region name.

    Returns
    -------
    Intervals of Trial
        One row per consecutive pair of visits, in time order, ids ``0`` to
        ``n - 1``: ``[departure from visit i, arrival at visit i + 1)``,
        ``start_region`` the first visit's region and ``outcome`` either
        ``"reached"`` (a different region, named in ``end_region``) or
        ``"returned"`` (back to the same region; ``end_region`` is None).
        Fewer than two visits give no trials.

    Raises
    ------
    TypeError
        If ``visits`` is not an Intervals, or a region value is not a name.
    ValueError
        If two visits overlap (a trial would end before it starts), or
        ``region`` is not a field.

    See Also
    --------
    detect_trials : Visits and trials from tracked positions.

    Examples
    --------
    >>> import pandas as pd
    >>> from neurospatial import Intervals
    >>> from neurospatial.behavior import trials_between_visits
    >>> pokes = pd.DataFrame(
    ...     {
    ...         "start_time": [0.0, 6.0, 15.0, 22.0],
    ...         "stop_time": [2.0, 9.0, 17.0, 25.0],
    ...         "well": ["well_2", "well_1", "well_2", "well_2"],
    ...     }
    ... )
    >>> trials = trials_between_visits(Intervals.from_dataframe(pokes), region="well")
    >>> trials.start.tolist(), trials.stop.tolist()
    ([2.0, 9.0, 17.0], [6.0, 15.0, 22.0])
    >>> trials.start_region.tolist(), trials.outcome.tolist()
    (['well_2', 'well_1', 'well_2'], ['reached', 'reached', 'returned'])
    """
    from neurospatial._interval_table import Intervals, require_intervals

    call = "trials_between_visits()"
    visits = require_intervals(visits, name="visits", owner=call)
    if region not in visits.fields:
        raise ValueError(
            _format_error(
                f"{call}: visits have no field {region!r}; their fields are "
                f"{list(visits.fields)}.",
                fix="pass region=<the field naming each visit's region>.",
            )
        )
    names = visits._field_array(region)
    ids = visits.interval_ids
    for interval_id, name in zip(ids, names, strict=True):
        if not isinstance(name, str):
            raise TypeError(
                _format_error(
                    f"{call}: visit {interval_id} has region {name!r}, not a name.",
                    why="each trial starts and ends at a named region.",
                    fix="map port or well numbers to the environment's region "
                    "names first.",
                )
            )
    start, stop = visits.start, visits.stop
    late = np.flatnonzero(stop[:-1] > start[1:])
    if late.size:
        i = int(late[0])
        raise ValueError(
            _format_error(
                f"{call}: visits {ids[i]} and {ids[i + 1]} overlap "
                f"([{start[i]:g}, {stop[i]:g}) and [{start[i + 1]:g}, "
                f"{stop[i + 1]:g}) s).",
                why="a trial runs from leaving one visit to arriving at the next, "
                "so the next arrival cannot come before the departure.",
                fix="merge or drop overlapping visits, for example two sensors "
                "of one port, before building trials.",
            )
        )
    trials = [
        _trial(float(stop[i]), float(start[i + 1]), names[i], names[i + 1])
        for i in range(len(visits) - 1)
    ]
    return Intervals(
        trials,
        record_type=Trial,
        analysis_settings={"source": "visits", "region_field": region},
    )


@reads_pynapple_samples
def detect_trials(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    regions: Sequence[str],
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> Intervals[Trial]:
    """Trials from tracked positions: the runs between visits to the regions.

    The animal visits the listed regions (reward wells, ports, start boxes)
    and runs between them. Each trial runs from leaving one region to
    arriving at the next one, whichever it is, so a task with one start
    region (a T-maze) and one where every well is a start (a W-track
    alternation) need the same call. Label the trials with your task's
    meaning (instructed goal, correct or error) using
    :meth:`Intervals.with_fields`.

    Parameters
    ----------
    env : Environment
        Allocentric environment holding the named regions, each with an area:
        a polygon, or ``regions.add(name, point=..., radius=...)`` for a
        well, the size of the place where the animal stops.
    times : ndarray, shape (n_samples,)
        Finite, strictly increasing timestamps in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Positions in the environment's coordinates. A NaN position (lost
        tracking) or one outside the environment is not observed: it ends the
        run, and a trial under way is censored. To bridge short dropouts,
        delete their rows; a deleted row is bridged when the samples around
        it are no more than ``max_gap`` apart.
    regions : sequence of str
        The regions trials start and end in, for example
        ``["well_1", "well_2", "well_3"]``. They must not share bins; point
        regions are refused.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge. A longer
        gap ends observation; ``None`` bridges every gap.
    epochs : (start, stop), array-like of shape (n, 2), Intervals, or None
        Half-open windows on the clock of ``times``; observation outside them
        is not used, and an epoch boundary ends a trial under way.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break (a new recording) between
        samples ``k - 1`` and ``k``.

    Returns
    -------
    Intervals of Trial
        One row per trial in time order, ids ``0`` to ``n - 1``. A trial is
        ``[departure, arrival)``: from the first sample outside its start
        region to the first sample inside the next region, so the arrival
        sample belongs to the next visit. ``outcome`` is:

        - ``"reached"``: arrived at a different region, named in
          ``end_region``;
        - ``"returned"``: came back to the start region (``end_region`` is
          None);
        - ``"censored"``: observation stopped first (a gap longer than
          ``max_gap``, an epoch edge, a declared break or the end of the
          recording); ``stop_time`` is the last observed time and
          ``end_region`` is None. Leave these out of success rates.

        No trial spans a break in observation. Time before the first visit
        and after a departure into a break is in no trial.
        ``analysis_settings`` records the regions, ``max_gap`` and
        ``epochs``.

    Raises
    ------
    ValueError
        For invalid times or positions, unknown, duplicate, overlapping or
        point regions.
    RegionNotFoundError
        If a region does not exist.

    See Also
    --------
    detect_region_visits : The visits themselves (time at each region).
    trials_between_visits : Trials from visits you bring, such as port
        events.

    Examples
    --------
    An animal on a 100 cm track runs between its two end zones, three times
    in 30 s:

    >>> import numpy as np
    >>> from shapely.geometry import box
    >>> from neurospatial import Environment, Regions
    >>> from neurospatial.behavior import detect_trials
    >>> times = np.arange(0.0, 30.0, 0.5)
    >>> x = 50.0 - 48.0 * np.cos(2 * np.pi * times / 20.0)
    >>> positions = np.column_stack([x, np.zeros_like(x)])
    >>> env = Environment.from_samples(positions, bin_size=4.0)
    >>> regions = Regions()
    >>> _ = regions.add("left", polygon=box(0, -5, 10, 5))
    >>> _ = regions.add("right", polygon=box(90, -5, 100, 5))
    >>> env = env.with_regions(regions)
    >>> trials = detect_trials(env, times, positions, regions=["left", "right"])
    >>> trials.start_region.tolist(), trials.end_region.tolist()
    (['left', 'right', 'left'], ['right', 'left', 'right'])
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
        call="detect_trials",
    )
    visits = observed.visits
    trials: list[Trial] = []
    for i, visit in enumerate(visits):
        if not visit.exit_observed:
            continue  # observation ended inside the region: no departure
        following = visits[i + 1] if i + 1 < len(visits) else None
        if following is not None and following.run_id == visit.run_id:
            trials.append(
                _trial(
                    visit.stop_time,
                    following.start_time,
                    visit.region,
                    following.region,
                )
            )
            continue
        run_end = observed.run_end_times[visit.run_id]
        if run_end > visit.stop_time:
            trials.append(
                Trial(
                    start_time=visit.stop_time,
                    stop_time=run_end,
                    start_region=visit.region,
                    end_region=None,
                    outcome="censored",
                )
            )
    return Intervals(
        trials, record_type=Trial, analysis_settings=dict(observed.settings)
    )
