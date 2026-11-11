"""Diba & Buzsáki (2007)'s preplay and replay shares, on a loaded diba_2007 session.

The worked result's pipeline, as ``diba_2007.WORKED_RESULT["criterion"]``
fixes it: per track segment and running direction, place-cell templates from
the laps; candidate events with ``ripple_detection``'s ``diba_2007`` inside the
end platforms; each judged with ``DIBA_BUZSAKI_2007``; an event found for both
directions assigned as the preset's ``template_choice`` does; forward events
counted at the start end of their direction (before the run), reverse ones at
the far end (after it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurospatial import compute_spatial_rates
from neurospatial.datasets._ripple import _speed_on_grid, spike_counts
from neurospatial.datasets.diba_2007 import END_QUANTILES, PLATFORM
from neurospatial.replay import detect_rank_order_replay
from neurospatial.replay.methods import DIBA_BUZSAKI_2007

SEED = 2007
#: The grid the spikes are counted on for detection, Hz.
GRID_HZ = 1000.0
#: A place cell's direction map peaks at this or more, Hz (the paper).
MIN_PEAK_RATE = 5.0
DIRECTIONS = ("increasing", "decreasing")


@dataclass(frozen=True)
class Event:
    """One judged candidate event."""

    segment: str
    start: float
    stop: float
    end: str  # "low" or "high": the platform it lies on
    template: str  # the direction it was assigned to
    rho: float  # against field positions along the track (increasing)
    p_value: float
    significant: bool
    n_active: int

    @property
    def run_order_rho(self) -> float:
        """The correlation with the run's firing order, the paper's template.

        ``rho`` correlates first-spike times with field positions along the
        track; a run of decreasing position fires its cells in decreasing
        position, so its template's order is the reverse.
        """
        return self.rho if self.template == "increasing" else -self.rho

    @property
    def before_run(self) -> bool:
        """At the start end of its template's direction."""
        return (self.end == "low") == (self.template == "increasing")


def _runs(mask: NDArray[np.bool_], times: NDArray[np.float64]) -> NDArray[np.float64]:
    """[first, last] sample times of each run of True, shape (n, 2)."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return np.column_stack([times[starts], times[stops]]).astype(np.float64)


def segment_inputs(rec, segment):
    """Templates, rate maps and reward areas of one track segment.

    Returns
    -------
    dict
        ``units`` (pyramidal unit ids), ``trains``, ``rates`` (direction ->
        (n_units, n_bins) Hz), ``place`` (direction -> bool mask over units),
        ``areas`` ("low"/"high" -> (n, 2) inclusive rows in s), ``span``.
    """
    rows = rec.epochs[segment]
    pyramidal = rec.units.index[rec.units["label"] == "pyramidal"]
    position = {uid: k for k, uid in enumerate(rec.units.index)}
    trains = [rec.spike_times[position[uid]] for uid in pyramidal]
    rates, place = {}, {}
    for direction in DIRECTIONS:
        laps = rec.epochs[f"{segment}_laps_{direction}"]
        result = compute_spatial_rates(
            rec.env, trains, rec.times, rec.positions, epochs=laps
        )
        rates[direction] = np.asarray(result.firing_rates, dtype=np.float64)
        place[direction] = np.nanmax(rates[direction], axis=1) >= MIN_PEAK_RATE
    inside = (rec.times >= rows[0, 0]) & (rec.times <= rows[0, 1])
    times = rec.times[inside]
    along = rec.env.to_linear(rec.positions[inside])
    low, high = np.nanquantile(along, END_QUANTILES)
    areas = {
        "low": _runs(along <= low + PLATFORM, times),
        "high": _runs(along >= high - PLATFORM, times),
    }
    return {
        "units": np.asarray(pyramidal),
        "trains": trains,
        "rates": rates,
        "place": place,
        "areas": areas,
        "span": (float(rows[0, 0]), float(rows[0, 1])),
    }


def detect(rec, inputs):
    """``diba_2007`` candidate events per direction, inside the platforms.

    Returns
    -------
    dict of str to ndarray, shape (n, 2)
        Each direction's [start, end] windows, seconds.
    """
    from ripple_detection.literature_methods import Recording, run_method

    start, stop = inputs["span"]
    grid = start + np.arange(int((stop - start) * GRID_HZ)) / GRID_HZ
    used = inputs["place"]["increasing"] | inputs["place"]["decreasing"]
    columns = np.flatnonzero(used)
    counts = spike_counts(
        [inputs["trains"][k] for k in columns], grid, sampling_rate=GRID_HZ
    )
    areas = np.concatenate([inputs["areas"]["low"], inputs["areas"]["high"]])
    areas = areas[np.argsort(areas[:, 0])]
    found = {}
    for direction in DIRECTIONS:
        cells = np.isin(columns, np.flatnonzero(inputs["place"][direction]))
        signals = Recording.from_arrays(
            grid,
            GRID_HZ,
            multiunit=counts,
            speed=_speed_on_grid(rec, grid),
            place_cells=cells,
        )
        table = run_method("diba_2007", signals, behavior_intervals=areas)
        found[direction] = table[["start_time", "end_time"]].to_numpy(np.float64)
    return found


def judge(rec, segment, inputs, windows, rng):
    """Each window's verdict on its own template, then one per detected event."""
    judged = {}
    for direction in DIRECTIONS:
        for start, stop in windows[direction]:
            result = detect_rank_order_replay(
                rec.env,
                inputs["trains"],
                (start, stop),
                inputs["rates"],
                rank_order_method=DIBA_BUZSAKI_2007,
                template=direction,
                rng=rng,
            )
            active = int(result.values[DIBA_BUZSAKI_2007.criteria[0].label])
            judged[direction, start, stop] = (result, active)
    # An event found for both directions: the preset's template_choice.
    keys = sorted(judged, key=lambda key: key[1])
    groups: list[list[tuple]] = []
    for key in keys:
        if groups and key[1] <= max(k[2] for k in groups[-1]):
            groups[-1].append(key)
        else:
            groups.append([key])
    events = []
    for group in groups:
        detected = [k for k in group if judged[k][0].is_replay]

        def strength(key):
            result, active = judged[key]
            score = abs(result.score) if np.isfinite(result.score) else -1.0
            return active, score

        chosen = max(detected or group, key=strength)
        result, active = judged[chosen]
        direction, start, stop = chosen
        end = "low" if _within(inputs["areas"]["low"], start, stop) else "high"
        events.append(
            Event(
                segment=segment,
                start=float(start),
                stop=float(stop),
                end=end,
                template=direction,
                rho=float(result.score),
                p_value=float(result.p_value),
                significant=bool(result.is_replay),
                n_active=active,
            )
        )
    return events


def _within(rows, start, stop):
    """Whether [start, stop] lies inside one inclusive row."""
    return bool(np.any((rows[:, 0] <= start) & (stop <= rows[:, 1])))


def run(rec, segments=("long_track", "short_track")):
    """Every judged event of the session's segments, seed ``SEED``."""
    rng = np.random.default_rng(SEED)
    events = []
    for segment in segments:
        inputs = segment_inputs(rec, segment)
        windows = detect(rec, inputs)
        events.extend(judge(rec, segment, inputs, windows, rng))
    return events


def shares(events):
    """Forward-before and reverse-after counts and shares of the significant events."""
    forward = [e for e in events if e.significant and e.run_order_rho > 0]
    reverse = [e for e in events if e.significant and e.run_order_rho < 0]
    before = sum(e.before_run for e in forward)
    after = sum(not e.before_run for e in reverse)
    return {
        "n_events": len(events),
        "n_forward": len(forward),
        "n_reverse": len(reverse),
        "forward_before": before,
        "reverse_after": after,
        "forward_before_share": before / len(forward) if forward else float("nan"),
        "reverse_after_share": after / len(reverse) if reverse else float("nan"),
    }
