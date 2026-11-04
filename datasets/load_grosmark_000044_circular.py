"""Grosmark & Buzsaki 2016 rat CA1 on a circular track with sleep before and after (DANDI 000044).

What was recorded: rat Achilles, 11 Jan 2013, 104 units from bilateral CA1
silicon probes over about 10 h: sleep in the home cage (PRE, 5.4 h), running
on a 1 m circular track (MAZE, 45 min) and sleep again (POST, 4.1 h).
Position (one LED) is tracked at 39.06 Hz during the maze epoch only.

The session is read by ``neurospatial.datasets.load("grosmark_2016",
"sub-Achilles_ses-Achilles-11012013", stream=True)``: the 9.2 GB file is
not downloaded, only the datasets it needs are read by HTTP byte range,
after checking the SHA-256 DANDI publishes. The public module
(``neurospatial/datasets/grosmark_2016.py``) and its docs page give the
conversions (positions and the linearized position in cm, epochs and sleep
states), the declared environment (the stated 1 m circle as a closed loop,
centred by a robust circle fit) and the watch-outs. ``_record.from_public``
converts it to the survey's record; ``dandi_session`` here is shared with
``load_grosmark_000044_linear``.

Analysis choices are the public loader's ``params``, the paper's
place-field analysis: ``bin_size`` 2 cm, ``min_speed`` 5 cm/s, ``dt``
0.25 s and alternating 60 s train/test blocks of the tracked span.

Survey-only:

- ``events["candidate_events"]``: the file has no ripple table. Replay
  candidates are population bursts in POST non-REM sleep found by a stated
  rule (``population_bursts``), not ripples; the first ``N_CANDIDATES`` are
  decoded, to bound the run time. ``events["all_bursts"]`` holds every burst.
- ``params["analysis_epochs"]``, the maze epoch. Without it the rate maps
  take the sleep spikes, which have no position, and the clean run's warning
  about them hides the ``spike_times_in_ms`` hazard probe's.
- ``params["replay_dt"]`` 0.02 s; ``params["nwb_source"]`` and
  ``params["readers_used"]``, for the ``io.nwb_readers`` operation.
- ``geometry["lap_start_region"]`` cuts the loop where the file's linearized
  coordinate starts (``lap_start_region``).
- the declared track's facts the environment fidelity study reads
  (``_tracks.public_track_facts``): fitted to the samples, no stated width,
  and the samples' own diameter against the stated 1 m (``samples_radius``).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from _record import (
    Range,
    Recording,
    SurveyOnly,
    check_expectations,
    from_public,
    public_file,
    validate_recording,
)
from _tracks import cut_region, public_track_facts

DATASET = {
    "key": "grosmark_000044_circular",
    "public": "grosmark_2016",
    "session": "sub-Achilles_ses-Achilles-11012013",
}

# Pre-registered from the dataset catalog (104 units, 39.06 Hz, 1 m circular track, PRE/MAZE/POST epochs) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": 104,
    "n_dims": 2,
    "sampling_rate_hz": Range(38.0, 40.0),
    "position_extent_cm": lambda rec: Range(90, 130).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "epoch_names": ["maze", "post", "pre"],
}


N_CANDIDATES = 200
_BURST_SIGMA_S = 0.015
_BURST_SD = 3.0
_BURST_DURATION_S = (0.05, 0.5)


def population_bursts(
    spike_times: tuple[np.ndarray, ...], windows: np.ndarray
) -> np.ndarray:
    """Population bursts inside ``windows``, as (start, stop) rows in s.

    Rule: the multiunit rate (every unit, 1 ms bins) smoothed with a 15 ms
    Gaussian; an event is a stretch whose peak exceeds the mean by 3 SD
    (mean and SD over ``windows``), bounded by the mean crossings around the
    peak, kept if it lasts 50-500 ms.
    """
    from scipy.ndimage import gaussian_filter1d

    step = 0.001
    events = []
    rows = []
    for start, stop in windows:
        edges = np.arange(start, stop + step, step)
        counts = sum(np.histogram(train, edges)[0] for train in spike_times)
        rows.append(
            (edges, gaussian_filter1d(counts.astype(float), _BURST_SIGMA_S / step))
        )
    rate = np.concatenate([r for _, r in rows])
    mean, sd = rate.mean(), rate.std()
    for edges, smooth in rows:
        above = smooth > mean
        change = np.diff(above.astype(int))
        starts = np.flatnonzero(change == 1) + 1
        stops = np.flatnonzero(change == -1) + 1
        if above[0]:
            starts = np.insert(starts, 0, 0)
        if above[-1]:
            stops = np.append(stops, len(above))
        for a, b in zip(starts, stops, strict=True):
            duration = (b - a) * step
            if smooth[a:b].max() > mean + _BURST_SD * sd and (
                _BURST_DURATION_S[0] <= duration <= _BURST_DURATION_S[1]
            ):
                events.append((edges[a], edges[b]))
    return np.asarray(events, dtype=float).reshape(-1, 2)


def dandi_session(
    key: str,
    session: str,
    *,
    geometry: Callable[[Any], Mapping[str, SurveyOnly]],
) -> Recording:
    """A DANDI 000044 session of ``grosmark_2016``, read by byte range, with candidates.

    ``geometry``, given the public recording, returns its survey-only
    geometry entries (the declared track's facts, at least).
    """
    from neurospatial.datasets import load as load_public

    rec = load_public("grosmark_2016", session, stream=True)
    non_rem, post = rec.epochs["non_rem"], rec.epochs["post"]
    post_nrem = non_rem[(non_rem[:, 0] >= post[0, 0]) & (non_rem[:, 1] <= post[0, 1])]
    bursts = population_bursts(rec.spike_times, post_nrem)
    return from_public(
        rec,
        key=key,
        params={
            "replay_dt": SurveyOnly(0.02, "the replay decoding bin"),
            "analysis_epochs": SurveyOnly(
                rec.epochs["maze"],
                "the maze epoch. Without it the rate maps take the PRE and POST"
                " sleep spikes, which have no position, so the clean run warns"
                " that most spikes lie outside the tracking, and that warning"
                " hides the spike_times_in_ms hazard probe's own",
            ),
            "nwb_source": SurveyOnly(
                public_file("grosmark_2016", session, stream=True),
                "the URL the file was read from by byte range",
            ),
            "readers_used": SurveyOnly(
                ("position", "units"), "the io.nwb readers the public loader calls"
            ),
        },
        events={
            "candidate_events": SurveyOnly(
                bursts[:N_CANDIDATES],
                f"first {N_CANDIDATES} of {len(bursts)} population bursts in"
                " POST non-REM (rule in population_bursts); the file has no"
                " ripple table",
            ),
            "all_bursts": SurveyOnly(bursts, "every population burst in POST non-REM"),
        },
        geometry=geometry(rec),
    )


DIAMETER_CM = 100.0
_RADIUS = re.compile(r"whose own radius is ([\d.]+) cm")


def lap_start_region(env: Any, positions: np.ndarray, linearized: np.ndarray) -> Any:
    """A region across the declared loop where the file's linearized coordinate starts.

    The loop's centre is the mean of its vertices (a regular polygon); the
    start is the mean of the samples whose linearized coordinate is in
    (0, 5) cm, projected onto the stated 1 m circle.
    """
    graph = env.binning.linearization.graph
    vertices = np.array([graph.nodes[node]["pos"] for node in graph.nodes])
    centre = vertices.mean(axis=0)
    radius = float(np.mean(np.linalg.norm(vertices - centre, axis=1)))
    on_track = np.isfinite(linearized) & np.isfinite(positions).all(axis=1)
    start = positions[on_track & (linearized > 0) & (linearized < 5.0)].mean(axis=0)
    angle = np.arctan2(*(start - centre)[::-1])
    point = centre + radius * np.array([np.cos(angle), np.sin(angle)])
    return cut_region(
        "linearized_zero",
        point,
        (-np.sin(angle), np.cos(angle)),
        half_length=3.0,
        half_width=15.0,
    )


def load(cache: Path | None = None) -> Recording:
    """Stream the session; positions in cm, spikes, epochs and candidate events."""
    del cache  # read by byte range: nothing is cached

    def circle(rec: Any) -> dict[str, SurveyOnly]:
        region = lap_start_region(rec.env, rec.positions, rec.linearized)
        return {
            **public_track_facts(
                rec,
                declared_from="fitted to the samples",
                physical_width=None,
                speed_thresholds=(rec.params["min_speed"], 30.0),
                stated_lengths={"diameter": (2 * samples_radius(rec), DIAMETER_CM)},
            ),
            "lap_start_region": SurveyOnly(
                region,
                "a region across the declared loop where the file's linearized"
                " coordinate starts, for behavior.laps with method='region'",
            ),
        }

    return dandi_session(DATASET["key"], DATASET["session"], geometry=circle)


def samples_radius(rec: Any) -> float:
    """The radius of the robust circle fit the public loader centred the loop by, cm.

    Read from its placement ``Conversion`` ("whose own radius is ... cm").
    """
    for conversion in rec.conversions:
        if found := _RADIUS.search(conversion.evidence):
            return float(found.group(1))
    msg = "no Conversion states the samples' own radius; the public loader changed"
    raise ValueError(msg)


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    print(
        f"{rec.dataset}: {len(rec.spike_times or ())} units, {rec.times[-1] - rec.times[0]:.0f} s"
    )
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
