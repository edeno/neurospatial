"""Pfeiffer & Foster (2013): CA1 place-cell sequences in a 2 m open arena (one sample session).

What was recorded: up to 40 tetrodes in dorsal CA1 of both hemispheres while a
rat alternated two phases in a 2 x 2 m arena with 36 reward wells: foraging
for reward at an unknown (Random) well, then returning to that day's Home
well. Position came from two head-mounted LEDs and an overhead camera at
60 Hz (the paper's Methods). The session here is one of the paper's eight:
rat 1, day 1, the 212 units the paper states for it.

Where it lives: not in an archive of the paper's own. Maboudi et al. (2018),
whose authors include Pfeiffer and Foster, released this one session as a
"sample recording epoch" with their code (their Data availability), as
``data/openfield.nel`` (55.8 MB; "example session in open field"), added on
2018-05-16 by co-author Etienne Ackermann (commit 962b3870). It is fetched
from eLife's archived copy of that repository,
github.com/elifesciences-publications/UncoveringTemporalStructureHippocampus,
pinned at commit f1b021d and by SHA-256 (the same file as the kemerelab
original), and is never redistributed. No data license is stated: the
repository's MIT license ("Copyright (c) 2018 Kemere Lab") is a software
license for the code. The paper's eight sessions are "available from Brad
Pfeiffer and David Foster on request" (Maboudi et al. 2018, Data
availability). Their paper credits the open-field data to Pfeiffer & Foster
(2013, 2015) and names no rat or day; rat 1 day 1 is inferred from
the 212 units (the paper's "212 ... units active during exploration from rat
1 on experimental days 1") and the 322 candidate windows (``n_SWRs=322`` for
rat 1 day 1 in Krause & Drugowitsch 2022's ``replay_structure/metadata.py``).

Reading it safely: the file is a gzip ``dill`` pickle of ``nelpy`` 0.1.2
objects. Unpickling it normally would run whatever code it names, so it is
read by ``_pickled.read_dill_pickle``: its imports are listed from the
opcodes first and refused outside an allow-list; ``nelpy``'s classes become
inert containers that hold their pickled state; ``dill``'s helpers resolve
only to NumPy's array and scalar reconstructors. Neither ``nelpy`` nor
``dill`` is imported.

Sources read: Pfeiffer & Foster 2013 (Nature 497:74, main text and Methods),
Pfeiffer & Foster 2015 (Science 349:180, main text and Supplementary Materials
and Methods), Maboudi et al. 2018 (eLife 7:e34467), and Krause & Drugowitsch
2022's code at 45d67dff (``replay_structure/ratday_preprocessing.py``,
``config.py`` and ``metadata.py``), which loads Pfeiffer's MATLAB files.

What the loader converts:

- ``spike_times``: each unit's ``SpikeTrainArray`` times, seconds (nelpy's
  unit); unit ids are the file's, 1 to 212. ``units["label"]`` is
  ``"excitatory"`` or ``"inhibitory"`` from the file's ``ExcitatoryNeuronIDs``
  (203) and ``InhibitoryNeuronIDs`` (9), which are those ids: they are
  disjoint and together cover 1 to 212 (Krause's code subtracts 1 from the
  MATLAB lists for its 0-based arrays).
- ``times``, ``positions``: the position signal's times (seconds) and its two
  rows, x and y, as stored, in cm (inferred: the samples span 198.6 x 191.6
  in the paper's 2 m arena).
- ``events["ripples"]``: the 322 ``BradRippleStartEndTimes`` windows, which
  Krause's code reads as Pfeiffer's MATLAB ``RippleTimes``;
  ``events["significant_ripples"]``: the 110 of them that
  ``BradSignificantRippleIDs`` names (0-based indices; Krause's code
  subtracts 1 from MATLAB's ``SignificantRipples``), and
  ``params["significant"]``, the same as a mask over the 322.
- ``epochs["session"]``: the spike trains' support, 2572.0 to 4759.0 s.

Not loaded: the file's multi-unit rate (``mua``, 100 Hz; the pooled spike
rate of all 212 units, smoothed), its 614 ``mua_epochs``, its binned
``PBEs`` and Maboudi's HMM scores; they are Maboudi's analysis, not the
recording. The file has no LFP, no head direction (one tracked point), no
well locations and no trial or reward times.

The environment is the stated 2 x 2 m arena (the Methods), placed on the
samples' minimum rotated rectangle, with the paper's 2 cm bins.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._arenas import PLACEMENT_TRIM, fitted_rectangle
from neurospatial.datasets._catalog import SessionEntry
from neurospatial.datasets._params import alternating_blocks
from neurospatial.datasets._pickled import inert_fields, read_dill_pickle
from neurospatial.datasets._recording import Conversion, Recording, tracked_mask

if TYPE_CHECKING:
    from neurospatial import Environment

__all__ = [
    "DETECTION_CHECK",
    "EXPECTED_WATCH_OUTS",
    "JUDGING_CHECK",
    "REQUIRES",
    "WATCH_OUTS",
    "WORKED_RESULT",
    "load_session",
]

_PAPER = "Pfeiffer & Foster 2013"
_PAPER_2015 = "Pfeiffer & Foster 2015"
_MABOUDI = "Maboudi et al. 2018"
_KRAUSE = (
    "Krause & Drugowitsch 2022's code (DrugowitschLab/HippocampalSWRDynamics at "
    "45d67dff)"
)

#: Modules of the ``datasets`` extra the loader imports: none.
REQUIRES: tuple[str, ...] = ()

#: The arena's side, cm (the Methods: "a 2 m x 2 m black arena").
SIDE = 200.0
#: Position bins, cm; place fields smoothed with a 4 cm Gaussian (both papers).
BIN_SIZE = 2.0
BANDWIDTH = 4.0
#: Place fields use time moving faster than this, cm/s; candidates time slower.
MIN_SPEED = 5.0
#: A unit has a place field if excitatory with a tuning-curve peak above this, Hz.
MIN_PEAK_RATE = 1.0
#: Decoding windows: 250 ms for behaviour; 20 ms advanced 5 ms for events.
DT = 0.25
EVENT_TIME_BIN = 0.02
EVENT_TIME_STEP = 0.005
#: Train and test blocks for decoding the rat's position, s.
BLOCK = 60.0
#: A position interval longer than this many median intervals is a gap: the
#: file's own support of the position signal breaks at every such interval.
GAP_PERIODS = 2.0

#: The ``nelpy`` classes the file pickles, each read as an inert container.
NELPY_CLASSES = frozenset(
    {
        "nelpy.auxiliary._results.ResultsContainer",
        "nelpy.core._analogsignalarray.AnalogSignalArray",
        "nelpy.core._analogsignalarray.EpochSignalSlicer",
        "nelpy.core._epocharray.EpochArray",
        "nelpy.core._spiketrain.BinnedSpikeTrainArray",
        "nelpy.core._spiketrain.EpochUnitSlicer",
        "nelpy.core._spiketrain.ItemGetter_iloc",
        "nelpy.core._spiketrain.ItemGetter_loc",
        "nelpy.core._spiketrain.SpikeTrainArray",
    }
)
#: The file's key for the one session it holds.
_RAT_DAY = ("Rat1", "Day1")

#: Pfeiffer's candidate windows against ``ripple_detection``'s detectors, both
#: ways: a window "overlaps" another when their open spans share any time.
DETECTION_CHECK: dict[str, Any] = {
    "what": (
        "ripple_detection.literature_methods' Pfeiffer detectors at the commit "
        "pyproject.toml pins, against the 322 windows of events['ripples']"
    ),
    "not_run": {
        "pfeiffer_2013_ripples": (
            "needs the LFP: the 150-250 Hz envelope of one electrode per "
            "tetrode, averaged, smoothed 12.5 ms, peaks above 3 SD over time "
            "below 5 cm/s (P&F 2013, Methods). The file holds no LFP, and no "
            "proxy is made from spikes"
        ),
        "pfeiffer_2015": (
            "needs the LFP likewise: the same envelope, bounds at the mean, 50 "
            "ms to 2 s (P&F 2015, Supplementary Materials). The file holds no LFP"
        ),
    },
    "method": "pfeiffer_2013",
    "inputs": (
        "every unit's spikes counted at 1 kHz over the spikes' support, the 203 "
        "excitatory units as the pyramidal cells the histogram counts, and the "
        "recording's speed (neurospatial's session speed, interpolated onto the "
        "grid): the paper does not say how it computed velocity"
    ),
    "detected": 402,
    "his_overlapped": (272, 322),
    "ours_overlapped": (291, 402),
    "significant_overlapped": (103, 110),
    "missed_below_3_sd": 30,
    "missed_reaching_5_cm_s": 26,
    "why": (
        "Pfeiffer's 322 are not population bursts by the 2013 rule: of the 50 "
        "this detector misses, 30 never rise 3 SD above the stopped mean of the "
        "excitatory units' 1 ms histogram smoothed 10 ms (our recomputation, a "
        "diagnostic), and 26 reach 5 cm/s on our speed, where the detector "
        "counts no time. Krause's code reads them as SWRs, and P&F 2015 took "
        "its candidates from the LFP ripple envelope, so they are most likely "
        "ripple windows; neither LFP detector can run on this file. 103 of the "
        "110 significant windows are found"
    ),
}

#: The ``neurospatial.replay`` presets on Pfeiffer's 322 windows against his
#: 110 significant labels. ``both`` counts windows he labels and the preset
#: judges a trajectory, ``his_only`` those he labels alone.
JUDGING_CHECK: dict[str, Any] = {
    "what": (
        "PFEIFFER_FOSTER_2015 and PFEIFFER_FOSTER_2013, unmodified, on each of "
        "the 322 windows, against params['significant']"
    ),
    "pipeline": (
        "the 203 excitatory units, all of which peak above 1 Hz (the papers' "
        "place-field rule); compute_spatial_rates on rec.env (2 cm bins) with "
        "method='binned', bandwidth=4 cm and min_speed=5 cm/s over the whole "
        "session; decode_position (Poisson, uniform prior over the supported "
        "bins) in 20 ms windows advanced 5 ms from each window's start, the "
        "last cut at its end; detect_replay with each preset"
    ),
    "n_units": 203,
    "supported_bins": (6321, 10000),
    "presets": {
        "pfeiffer_foster_2015": {
            "judged": 78,
            "both": 65,
            "ours_only": 13,
            "his_only": 45,
            "neither": 199,
            "agreement": 0.82,
            "kappa": 0.569,
        },
        "pfeiffer_foster_2013": {
            "judged": 94,
            "both": 40,
            "ours_only": 54,
            "his_only": 70,
            "neither": 158,
            "agreement": 0.615,
            "kappa": 0.113,
        },
    },
    "his_only_2015": {
        "short_of_80_cm": 43,
        "short_of_11_bins": 2,
        "mean_distance_median": 55.4,
        "map_distance_median": 74.4,
        "map_reaching_80_cm": 16,
        "peak_posterior_median": 0.059,
        "peak_posterior_median_both": 0.073,
    },
    "rate_then_smooth_2015": (0.823, 0.572),
    "why": (
        "the 45 windows he labels and PFEIFFER_FOSTER_2015 rejects decode here to "
        "short paths: 43 fall short of 80 cm (2 of fewer than 11 time bins), "
        "with a median 55.4 cm between the segment's first and last posterior "
        "means. Their decoding is also less sharp: the median row puts 0.059 of "
        "the posterior in its peak bin, against 0.073 in the 65 windows both "
        "call trajectories. A broad posterior pulls its mean toward the arena's "
        "centre, but that is not the whole cause: over the same segment the MAP "
        "runs a median 74.4 cm, and reaches 80 cm in only 16 of the 45. The "
        "place fields' reading is not the cause either: dividing first and then "
        "smoothing the rate gives 82.3% and kappa 0.572. Neither paper says how "
        "unvisited bins, the speed or the smoothing were handled, and no code "
        "was published, so what made his decoded paths longer is not known "
        "(unsettled). PFEIFFER_FOSTER_2013's lower agreement is expected if the "
        "labels are 2015's: a different rule on the same windows"
    ),
    "presets_against_the_papers": (
        "both presets state their papers' rules as the Methods give them: "
        "PFEIFFER_FOSTER_2013 the MAP, steps under 20 cm, at least 10 steps (11 "
        "time bins), 40 cm start to end; PFEIFFER_FOSTER_2015 the posterior "
        "mean ('weighted mean posterior probability'), steps under 50 cm, at "
        "least 10 steps, 80 cm. The 2013 shuffle tests are not applied (every "
        "trajectory event passed them, the paper says), and the 2015 paper has "
        "none. Neither paper published code"
    ),
}

#: The paper's finding this dataset reproduces, or why there is none yet.
WORKED_RESULT: dict[str, Any] = {
    "status": "not possible yet",
    "level": None,
    "finding": (
        "Pfeiffer's own per-event labels, 110 of his 322 windows, reproduced by "
        "the trajectory-event rule of the paper that most likely made them"
    ),
    "quote": (
        '"each candidate replay event was truncated to the longest sequence of '
        "time frames with a weighted mean posterior probability less than 50 cm "
        "from that of the previous frame. Candidate events with fewer than 10 "
        "steps in the final sequence or a start-to-end distance less than 80 cm "
        'were eliminated from future analysis." (P&F 2015, Supplementary '
        "Materials, Trajectory Event Analysis)"
    ),
    "source": (
        "Pfeiffer & Foster 2015 (Science 349:180), Supplementary Materials and "
        "Methods; the labels are the file's BradSignificantRippleIDs. Maboudi et "
        "al. 2018, who used Pfeiffer's events, report 815 of 2,980 over their 8 "
        "sessions as P&F 2015's trajectory events, the 815 that paper reports"
    ),
    "paper": (
        "110 of 322 labelled in the file (34.2%). P&F 2013 reports 144 to 373 "
        "trajectory events per session (25.3% to 43.9% of candidates), 274 for "
        "R1 D1, but on its own population-burst candidates under its own rule: "
        "274 is more than the 110 and its candidates are not the 322, so neither "
        "the count nor the share can be set against the 110"
    ),
    "ours": None,
    "criterion": None,
    "parameters": None,
    "unmatched": (
        "how the papers handled unvisited bins, computed speed and smoothed "
        "place fields: not stated, no code",
        "no LFP, so the 322 windows cannot be re-detected with the 2015 rule",
    ),
    "attempted": {
        "value": 0.82,
        "criterion": (
            "fixed in writing before any window was decoded: "
            "PFEIFFER_FOSTER_2015's verdicts agree with Pfeiffer's labels on at "
            "least 80% of the 322 windows and Cohen's kappa is at least 0.6. "
            "Agreement is 82.0%, kappa 0.569, so it is not met"
        ),
        "parameters": JUDGING_CHECK["pipeline"],
        "differs": (
            "PFEIFFER_FOSTER_2015 judges 78 trajectories against his 110: 65 of "
            "them his, 13 not, and 45 of his rejected, 43 of those because their "
            "decoded path falls short of 80 cm here (JUDGING_CHECK['why'])"
        ),
    },
    "why": (
        "the rule agrees with his labels on 82% of windows, kappa 0.569, just "
        "under the 0.6 fixed beforehand; the shortfall is his labelled windows "
        "whose decoded path falls short of 80 cm here, and what made his paths "
        "longer is not stated in either paper"
    ),
}

#: Findings computed from the data that each verified session must report, as
#: patterns searched for in its watch_outs.
EXPECTED_WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "rat1_day1": (r"position intervals? (is|are) longer than 2 median intervals",),
}

#: Pitfalls of each session checked by hand, with their evidence.
WATCH_OUTS: dict[str, tuple[str, ...]] = {
    "rat1_day1": (
        "This is one session of the paper's eight (four rats, two days each), "
        "the only one released: Maboudi et al. 2018's repository holds it as an "
        "example. Nothing here generalizes across rats or days.",
        "No data license is stated. Maboudi et al. 2018 (eLife 7:e34467), whose "
        "authors include Pfeiffer and Foster, write under Data availability: "
        "'Data for Figure 6 was previously reported in Pfeiffer and Foster "
        "(2015, Science) and is available from Brad Pfeiffer and David Foster "
        "on request. ... All analysis code and sample recording epochs for "
        "Figures 1-7 are available on https://github.com/kemerelab/"
        "UncoveringTemporalStructureHippocampus (copy archived at https://"
        "github.com/elifesciences-publications/"
        "UncoveringTemporalStructureHippocampus).' The file was added on "
        "2018-05-16 by co-author Etienne Ackermann (commit 962b3870). The "
        "repository's only license is MIT ('Copyright (c) 2018 Kemere Lab'), a "
        "software license for the code. The loader downloads the file from "
        "eLife's archived copy at a pinned commit; it is never redistributed. "
        "The other sessions are available from the authors on request.",
        "Which rat and day this is, is inferred: the file says only Rat1, Day1. "
        "Its 212 units are the paper's for rat 1 day 1, and its 322 candidate "
        "windows are the n_SWRs = 322 Krause & Drugowitsch 2022's code lists "
        "for rat 1 day 1.",
        "Positions are taken as cm: no source read states the unit, and the "
        "samples span 198.6 x 191.6 in the paper's 2 x 2 m arena.",
        "How the 322 windows (events['ripples']) and the 110 significant ones "
        "were made is not documented in the file. Krause's code reads them as "
        "Pfeiffer's RippleTimes and SignificantRipples and calls them SWRs. "
        "Maboudi et al. 2018 report 815 of 2,980 events over their 8 sessions "
        "as 'trajectory events' (Pfeiffer and Foster, 2015), and that paper "
        "reports 815 open-field trajectory events, so the 110 are most likely "
        "P&F 2015's trajectory events (posterior mean steps under 50 cm, 80 cm "
        "start to end). They are not P&F 2013's: that paper counts 274 "
        "trajectory events for R1 D1 (89 home-events and 185 away-events), "
        "from population-burst candidates, more than the 110.",
        "The file is a pickle. It is read without running any of it: only "
        "NumPy's array reconstruction and inert containers for nelpy's classes "
        "are resolved, and anything else it names is refused.",
        "No LFP is in the file, so neither paper's ripple detector can be run "
        "here, and there is no head direction (one tracked point), no well "
        "location, Home well or trial time.",
        "The file also holds Maboudi's multi-unit rate, 614 mua_epochs, binned "
        "PBEs and HMM scores; they are their analysis and are not loaded.",
    ),
}


def _named(fields: dict[str, Any], key: str) -> Any:
    """``fields[key]``, or say which keys the file has."""
    try:
        return fields[key]
    except KeyError:
        msg = f"the file has no {key!r}; it holds {sorted(fields)}"
        raise ValueError(msg) from None


def _session(path: Path) -> dict[str, Any]:
    """The file's one session dict, its objects inert."""
    root = inert_fields(read_dill_pickle(path, NELPY_CLASSES))
    rat, day = _RAT_DAY
    days = _named(root, "data")
    if list(days) != [rat] or list(days[rat]) != [day]:
        msg = f"expected one session {rat}/{day}; the file holds {root['ratdays']}"
        raise ValueError(msg)
    session: dict[str, Any] = days[rat][day]
    return session


def _unit_labels(
    unit_ids: list[int], excitatory: NDArray[Any], inhibitory: NDArray[Any]
) -> list[str]:
    """Each unit's label from the file's two id lists, which must partition the ids.

    Examples
    --------
    >>> _unit_labels([1, 2, 3], np.array([1, 3]), np.array([2]))
    ['excitatory', 'inhibitory', 'excitatory']
    """
    known = set(unit_ids)
    first, second = set(excitatory.tolist()), set(inhibitory.tolist())
    if first & second or first | second != known:
        msg = (
            f"the excitatory ({len(first)}) and inhibitory ({len(second)}) id "
            f"lists do not partition the {len(known)} unit ids: "
            f"{len(first & second)} in both, {len(known - first - second)} in "
            f"neither, {len((first | second) - known)} unknown"
        )
        raise ValueError(msg)
    return ["excitatory" if unit in first else "inhibitory" for unit in unit_ids]


def _significant(n_events: int, ids: NDArray[Any]) -> NDArray[np.bool_]:
    """A mask over ``n_events`` from 0-based indices, each once and in range.

    Examples
    --------
    >>> _significant(4, np.array([0, 2])).tolist()
    [True, False, True, False]
    """
    index = np.asarray(ids, dtype=np.int64)
    if index.size and (
        index.min() < 0
        or index.max() >= n_events
        or np.unique(index).size != index.size
    ):
        msg = (
            f"the significant ids run from {index.min()} to {index.max()} with "
            f"{index.size - np.unique(index).size} repeats; 0-based indices into "
            f"{n_events} windows were expected"
        )
        raise ValueError(msg)
    mask = np.zeros(n_events, dtype=bool)
    mask[index] = True
    return mask


def _declared_square(positions: NDArray[np.float64]) -> tuple[Environment, Conversion]:
    """The stated 2 m square on the samples' minimum rotated rectangle."""
    from shapely import affinity
    from shapely.geometry import box

    from neurospatial import Environment

    centre, angle, lengths = fitted_rectangle(positions)
    half = SIDE / 2
    square = affinity.translate(
        affinity.rotate(box(-half, -half, half, half), angle, use_radians=True),
        *centre,
    )
    env = Environment.from_polygon(
        square, bin_size=BIN_SIZE, units="cm", name="pfeiffer_2013 rat1_day1"
    )
    sides = ", ".join(f"{side:.4g} cm" for side in lengths)
    placement = Conversion(
        f"environment: the stated {SIDE:g} x {SIDE:g} cm arena centred at "
        f"({centre[0]:.4g}, {centre[1]:.4g}) cm, rotated "
        f"{np.degrees(angle):+.2f} deg from the file's axes; the samples' own "
        f"sides: {sides}",
        "inferred",
        "the file states no origin or rotation; the centre and rotation are "
        "those of the minimum rotated rectangle around the tracked samples, "
        f"less the {4 * PLACEMENT_TRIM:.1%} farthest from their median. "
        f"{_MABOUDI}'s Figure 6 notebook bins tuning curves over x 90 to 290 "
        "and y 0 to 200",
    )
    return env, placement


def _gap_watch_out(times: NDArray[np.float64]) -> str | None:
    """How many position intervals exceed ``GAP_PERIODS`` median intervals."""
    steps = np.diff(times)
    limit = GAP_PERIODS * float(np.median(steps))
    long = steps[steps > limit]
    if not long.size:
        return None
    count = (
        "1 position interval is"
        if long.size == 1
        else f"{long.size} position intervals are"
    )
    return (
        f"{count} longer than {GAP_PERIODS:g} median "
        f"intervals ({limit * 1e3:.1f} ms), {long.sum():.1f} s in all, the "
        f"longest {long.max():.3f} s: samples missing from the file, where its "
        "own support of the position signal breaks. Intervals also vary "
        f"between samples ({np.percentile(steps, 1) * 1e3:.1f} to "
        f"{np.percentile(steps, 99) * 1e3:.1f} ms, 1st to 99th percentile), "
        "so speed from neighbouring samples is noisy. They are kept as stored."
    )


def load_session(
    entry: SessionEntry, path_or_stream: str | os.PathLike[str]
) -> Recording:
    """Read the session's pickle into a :class:`Recording`, running none of it.

    Parameters
    ----------
    entry : SessionEntry
        The session's registry entry.
    path_or_stream : str or path
        The cached ``openfield.nel``.

    Returns
    -------
    Recording
        Unchecked; :func:`neurospatial.datasets.load` runs the checks.

    Raises
    ------
    pickle.UnpicklingError
        If the pickle names a global, module, attribute or dtype outside the
        allow-list.
    ValueError
        If the file holds another session layout, the unit id lists do not
        partition the units, or the significant ids are not 0-based indices
        into the windows.
    """
    import pandas as pd

    session = _session(Path(path_or_stream))
    trains_obj = inert_fields(_named(session, "st"))
    position = inert_fields(_named(session, "pos"))
    unit_ids = [int(unit) for unit in trains_obj["_unit_ids"]]
    trains = tuple(
        np.asarray(train, dtype=np.float64).ravel() for train in trains_obj["_time"]
    )
    labels = _unit_labels(
        unit_ids,
        np.asarray(_named(session, "ExcitatoryNeuronIDs")),
        np.asarray(_named(session, "InhibitoryNeuronIDs")),
    )
    table = pd.DataFrame(
        {"brain_region": ["dorsal CA1"] * len(unit_ids), "label": labels},
        index=pd.Index(unit_ids, name="unit_id"),
    )
    times = np.asarray(position["_time"], dtype=np.float64)
    positions = np.asarray(position["_ydata"], dtype=np.float64).T
    windows = np.asarray(
        inert_fields(_named(session, "BradRippleStartEndTimes"))["_time"],
        dtype=np.float64,
    )
    significant = _significant(
        len(windows), np.asarray(_named(session, "BradSignificantRippleIDs"))
    )
    support = np.asarray(
        inert_fields(trains_obj["_support"])["_time"], dtype=np.float64
    ).reshape(-1, 2)
    env, placement = _declared_square(positions)
    span = np.ptp(positions[tracked_mask(positions)], axis=0)
    train, test = alternating_blocks(float(support[0, 0]), float(support[-1, 1]), BLOCK)
    watch_outs = list(WATCH_OUTS.get(entry.session, ()))
    if (gaps := _gap_watch_out(times)) is not None:
        watch_outs.append(gaps)
    return Recording(
        **entry.identity(path_or_stream),
        animal="rat 1 (the paper's R1), day 1 (inferred)",
        apparatus=entry.info.apparatus,
        brain_regions=entry.info.brain_regions,
        clock=(
            "the recording's clock, from an arbitrary start (the spikes' "
            f"support runs from {support[0, 0]:.1f} to {support[-1, 1]:.1f} s)"
        ),
        times=times,
        spike_times=trains,
        units=table,
        env=env,
        env_kind="declared",
        env_source=(
            f"the stated {SIDE:g} x {SIDE:g} cm arena ({_PAPER}, Methods: 'a 2 m x "
            f"2 m black arena with 30-cm-high walls'), {BIN_SIZE:g} cm bins"
        ),
        positions=positions,
        position_units="cm",
        tracking=(
            "the head: two head-mounted LEDs and an overhead camera at 60 Hz "
            f"({_PAPER}); the file holds one point per sample"
        ),
        epochs={"session": support},
        events={"ripples": windows, "significant_ripples": windows[significant]},
        params={
            "bin_size": BIN_SIZE,
            "bandwidth": BANDWIDTH,
            "min_speed": MIN_SPEED,
            "min_peak_rate": MIN_PEAK_RATE,
            "dt": DT,
            "event_time_bin": EVENT_TIME_BIN,
            "event_time_step": EVENT_TIME_STEP,
            "analysis_epochs": support,
            "train_epochs": train,
            "test_epochs": test,
            "significant": significant,
        },
        conversions=(
            Conversion(
                "the file: a gzip dill pickle of nelpy 0.1.2 objects, read by a "
                "restricted unpickler (NumPy's _reconstruct and scalar through "
                "dill's _get_attr, ndarray, the dtypes O8, f8, i8, u1, u2; nelpy "
                "classes as inert containers); nelpy and dill are not imported",
                "stated",
                "the pickle's imports, listed from its opcodes before it is read",
            ),
            Conversion(
                "times, spike_times and event windows: the nelpy objects' _time, "
                "seconds",
                "inferred",
                "nelpy stores times in seconds; the position samples are a "
                f"median {1e3 / entry.info.stated.sampling_rate:.1f} ms apart "
                f"(the paper's camera ran at 60 Hz) and the spikes, the position "
                "and the 322 windows lie within one span, "
                f"{support[0, 0]:.1f} to {support[-1, 1]:.1f} s",
            ),
            Conversion(
                "positions: the position signal's two rows, x and y, as stored, in cm",
                "inferred",
                f"no source read states the unit; the samples span {span[0]:.1f} "
                f"x {span[1]:.1f} in the stated 2 x 2 m arena ({_PAPER})",
            ),
            Conversion(
                "units: ids 1 to 212, the file's _unit_ids; label from "
                "ExcitatoryNeuronIDs and InhibitoryNeuronIDs, read as those ids",
                "inferred",
                "the two lists are disjoint and together hold every id once; "
                f"{_KRAUSE} subtracts 1 from the MATLAB lists for its 0-based "
                f"arrays. {_PAPER} excluded putative inhibitory units 'on the "
                "basis of spike width and mean firing rate'",
            ),
            Conversion(
                "events['significant_ripples'] and params['significant']: "
                "BradSignificantRippleIDs as 0-based indices into the 322 "
                "BradRippleStartEndTimes",
                "inferred",
                f"{_KRAUSE} loads MATLAB's SignificantRipples - 1; the ids run "
                "from 7 to 318, and only offset 0 lines up with Maboudi's HMM "
                "significance per event (coherent shuffle p = 4e-5; offsets "
                "-1, 1 and 2 give p >= 0.2)",
            ),
            placement,
        ),
        watch_outs=tuple(watch_outs),
    )
