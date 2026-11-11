"""tirole_2022 against the authors' own pipeline: candidates, replay judging and Fig. 3A.

The reference is the authors' MATLAB code (bendor-lab/Elife_Tirole_Huelin_Gorriz_2022
at 44ecf42, CC0) run in Octave on the deposit: ``tests/data/tirole_2022/authors_octave.npz``
(its ``provenance`` entry says what each array is). Ours is
``tests/data/tirole_2022/ours.npz``, written by running this module as a script
(``python tests/datasets/test_tirole_2022_replay.py``, several hours on 4 cores):
for each session, the candidate events ``ripple_detection.literature_methods.tirole_2022``
finds on the loaded data with the package's speed and with the file's own speed,
and ``TIROLE_2022``'s verdict on each of the authors' RUN candidates with ripple
power of 3 SD or more, on each track, for the whole event and its two halves.

The fast tests compute the agreement from the two tables; the ``realdata`` tests
recompute part of ours from the loaded data, so the table cannot drift from the
code, and run the Fig. 3A chain.

Helpers below port the authors' code where the comparison needs its exact inputs:
``calculate_place_fields.m`` (place cells and the 10 cm rate maps decoded with),
``spike_count.m`` (20 ms bins from the event's onset, ``ceil`` of its duration),
the rate floor of ``bayesian_decoding.m``, ``replay_decoding_split_events.m``
(halves), ``number_of_significant_replays.m`` (verdicts and the choice between
tracks), ``rate_remapping_TRACK_PAIRS.m`` and ``get_stable_cells.m`` (Fig. 3A).
They read the file's own speed (``position.v_cm``), which the authors' code uses.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path

import numpy as np
import pytest

from neurospatial.datasets import tirole_2022

DATA = Path(__file__).parents[1] / "data" / "tirole_2022"
SESSIONS = sorted(tirole_2022.SESSIONS)
DEFAULT = "rat3_session2"
#: Ripple power, in SD, a RUN candidate must reach to be judged (the code's default).
RIPPLE_GATE = 3.0
REPLAY_BIN = 0.02


@functools.cache
def authors() -> dict:
    with np.load(DATA / "authors_octave.npz") as file:
        return {key: file[key] for key in file.files}


@functools.cache
def ours() -> dict:
    with np.load(DATA / "ours.npz") as file:
        return {key: file[key] for key in file.files}


# --------------------------------------------------------- the authors' code, ported


def gausswin(n: int, alpha: float = 2.5) -> np.ndarray:
    """MATLAB ``gausswin``."""
    k = np.arange(n) - (n - 1) / 2
    return np.exp(-0.5 * (alpha * k / ((n - 1) / 2)) ** 2)


def nearest(t: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """MATLAB ``interp1(t, values, query, 'nearest')``: NaN outside ``t``."""
    i = np.clip(np.searchsorted(t, query), 1, len(t) - 1)
    pick = np.where(query - t[i - 1] < t[i] - query, i - 1, i)
    out = values[pick].astype(float)
    out[(query < t[0]) | (query > t[-1])] = np.nan
    return out


def skaggs(raw: np.ndarray, dwell: np.ndarray) -> float:
    """``skaggs_information.m``: over the bins with a nonzero rate only."""
    keep = raw != 0
    if not keep.any():
        return 0.0
    p = dwell[keep] / np.nansum(dwell[keep])
    norm = raw[keep] / np.nansum(raw[keep] * p)
    return float(np.nansum(p * norm * np.log2(norm)))


def place_fields(
    position: dict, clusters: dict, bin_width: float, windows=None
) -> dict:
    """``calculate_place_fields.m`` without waveforms (``_epochs.m`` with ``windows``)."""
    from scipy.signal import filtfilt

    t = position["t"]
    tracks = position["linear"]
    ids = clusters["spike_id"].astype(int)
    st = clusters["spike_times"]
    n_units = int(clusters["id_conversion"][:, 0].max())
    spans = []
    for track in tracks:
        on = ~np.isnan(track["linear"])
        spans.append((t[on].min(), t[on].max()))
    all_spikes = np.zeros((len(tracks), n_units))
    for k, (a, b) in enumerate(spans):
        inside = (st > a) & (st < b)
        all_spikes[k] = np.bincount(ids[inside], minlength=n_units + 1)[1:]
    mean_rate = all_spikes.sum(axis=0) / sum(b - a for a, b in spans)
    dt = t[1] - t[0]
    smoothing = np.array([1.0, 1.0]) if bin_width == 10 else gausswin(10)
    smoothing /= smoothing.sum()
    out: dict = {"mean_rate": mean_rate, "track": []}
    for k, track in enumerate(tracks):
        a, b = windows[k] if windows is not None and k < len(windows) else spans[k]
        linear = track["linear"]
        speed = np.abs(position["v_cm"]).astype(float)
        speed[np.isnan(linear)] = np.nan
        at_spike = nearest(t, linear, st)
        speed_at_spike = nearest(t, speed, st)
        edges = np.arange(0, 100 * track["length"] + bin_width / 2, bin_width)
        moving = (t > a) & (t < b) & (speed > 5) & (speed < 50)
        dwell = dt * np.histogram(linear[moving], edges)[0]
        counted = (st > a) & (st < b) & (speed_at_spike > 5) & (speed_at_spike < 50)
        spikes = np.array(
            [
                np.histogram(at_spike[counted & (ids == j)], edges)[0]
                for j in range(1, n_units + 1)
            ]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = spikes / dwell
        raw[np.isnan(raw) | (dwell == 0)[None, :]] = 0.0
        smooth = filtfilt(
            smoothing, [1.0], raw, axis=1, padlen=3 * (len(smoothing) - 1)
        )
        peak, raw_peak = smooth.max(axis=1), raw.max(axis=1)
        info = np.array([skaggs(row, dwell) for row in raw])
        good = (
            np.flatnonzero(
                (peak >= 0.5) & (raw_peak >= 1) & (mean_rate <= 5) & (info > 0)
            )
            + 1
        )
        out["track"].append(
            {
                "raw": raw,
                "dwell": dwell,
                "peak": peak,
                "raw_peak": raw_peak,
                "good_cells": good,
                "window": (a, b),
            }
        )
    out["good_place_cells"] = np.unique(
        np.concatenate([tr["good_cells"] for tr in out["track"]])
    )
    return out


@functools.cache
def deposit(session: str) -> tuple[dict, dict]:
    """The session's position and clusters structs, from the cached files."""
    from scipy.io import loadmat

    from neurospatial.datasets._load import _session_entry, fetch_session

    folder = fetch_session(_session_entry("tirole_2022", session))
    stem = tirole_2022.SESSIONS[session].spikes_file
    position = loadmat(folder / f"{stem}_extracted_position.mat", simplify_cells=True)[
        "position"
    ]
    clusters = loadmat(folder / f"{stem}_extracted_clusters.mat", simplify_cells=True)[
        "clusters"
    ]
    return position, clusters


@functools.cache
def decoding_inputs(session: str, authors_floor: bool = False):
    """Good place cells' spike trains and each track's 10 cm raw rate maps.

    By default the maps follow the package: a bin with no time above 5 cm/s is
    NaN (missing, not zero), and the decoder handles zero rates. With
    ``authors_floor`` they are the code's input instead, as a reference only:
    ``bayesian_decoding.m`` sets those bins to 0 and raises every rate below
    10^(log2(N) - log2(400)) to it.
    """
    position, clusters = deposit(session)
    fields = place_fields(position, clusters, 10)
    good = fields["good_place_cells"]
    ids = clusters["spike_id"].astype(int)
    trains = [np.sort(clusters["spike_times"][ids == g]) for g in good]
    names = [f"track_{k + 1}" for k in range(len(fields["track"]))]
    floor = 10 ** (np.log2(len(good)) - np.log2(400))
    rates = {}
    for k, name in enumerate(names):
        raw = fields["track"][k]["raw"][good - 1]
        if authors_floor:
            rates[name] = np.maximum(raw, floor)
        else:
            rates[name] = np.where(fields["track"][k]["dwell"] == 0, np.nan, raw)
    return trains, names, rates, fields


def gated(session: str) -> np.ndarray:
    """The authors' RUN candidates with ripple power of ``RIPPLE_GATE`` SD or more."""
    return np.flatnonzero(authors()[f"{session}/run_events"][:, 3] >= RIPPLE_GATE)


def judge_events(
    session: str, events, n_shuffles: int = 1000, authors_floor: bool = False
) -> dict[str, np.ndarray]:
    """``TIROLE_2022`` on each event, each track, whole and halves.

    Returns score, p and verdict of shape (n_tracks, n_events, 3) and each track's
    summed posterior (the code's "Bayesian bias" sums), the same shape.
    """
    from neurospatial.decoding import decode_jointly
    from neurospatial.replay import detect_replay
    from neurospatial.replay.methods import TIROLE_2022

    trains, names, rates, _ = decoding_inputs(session, authors_floor)
    envs = {name: tirole_2022.track_environment() for name in names}
    run = authors()[f"{session}/run_events"]
    whole = dataclasses.replace(TIROLE_2022, n_shuffles=n_shuffles)
    half = dataclasses.replace(whole, alpha=whole.alpha / 2)
    shape = (len(names), len(events), 3)
    out = {key: np.full(shape, np.nan) for key in ("score", "p", "verdict", "sums")}
    out["verdict"][:] = 0.0
    for column, event in enumerate(events):
        rng = np.random.default_rng([SESSIONS.index(session), int(event)])
        onset, offset, midpoint = run[event, :3]
        n_bins = int(np.ceil((offset - onset) / REPLAY_BIN))
        edges = onset + REPLAY_BIN * np.arange(n_bins + 1)
        centres = edges[:-1] + REPLAY_BIN / 2
        counts = np.array([np.histogram(train, edges)[0] for train in trains]).T
        middle = int(np.argmin(np.abs(centres - midpoint)))
        parts = (slice(None), slice(0, middle + 1), slice(middle, None))
        for part, rows in enumerate(parts):
            c, t = counts[rows], centres[rows]
            if len(t) < 2 or c.sum() == 0:
                continue
            joint = decode_jointly(envs, c, rates, dt=REPLAY_BIN, times=t)
            out["sums"][:, column, part] = joint.template_probability.sum(axis=0)
            for k, name in enumerate(names):
                result = detect_replay(
                    envs,
                    joint,
                    t,
                    replay_method=whole if part == 0 else half,
                    spike_counts=c,
                    encoding_models=rates,
                    dt=REPLAY_BIN,
                    template=name,
                    rng=rng,
                )
                out["score"][k, column, part] = result.score
                out["p"][k, column, part] = result.p_value
                out["verdict"][k, column, part] = float(result.is_replay)
    return out


def authors_segments(p: np.ndarray) -> np.ndarray:
    """The code's verdict per segment from (track, event, part, shuffle) p-values.

    MATLAB's ``max`` skips NaN; a half's NaN p-value counts as 1.
    """
    with np.errstate(invalid="ignore"):
        whole = (
            np.nanmax(
                np.where(
                    np.isnan(p[:, :, 0]).all(-1, keepdims=True), np.inf, p[:, :, 0]
                ),
                axis=-1,
            )
            < 0.05
        )
    halves = np.where(np.isnan(p[:, :, 1:]), 1.0, p[:, :, 1:]).max(axis=-1)
    return np.stack([whole, halves[..., 0] < 0.025, halves[..., 1] < 0.025], axis=-1)


def choose_tracks(segments: np.ndarray, scores: np.ndarray, sums: np.ndarray):
    """``number_of_significant_replays.m`` (no re-exposure option): one track per event.

    Returns the final verdict (track, event) and each verdict's best segment (1 whole,
    2 first half, 3 second half; 0 not significant).
    """
    significant = segments.any(axis=-1)
    best = np.where(
        significant, np.argmax(segments * np.nan_to_num(np.abs(scores)), axis=-1) + 1, 0
    )
    final = np.zeros_like(significant)
    for e in range(significant.shape[1]):
        tracks = np.flatnonzero(significant[:, e])
        if len(tracks) <= 1 or (
            not np.any(best[:, e] == 1)
            and np.sum(best[:, e] == 2) == 1
            and np.sum(best[:, e] == 3) == 1
        ):
            final[tracks, e] = True
            continue
        kinds = best[tracks, e]
        part = (
            2
            if kinds.min() == 3
            else 1
            if (kinds.min() == 2 and kinds.max() == 2)
            else 0
        )
        bias = sums[tracks, e, part] / sums[tracks, e, part].sum()
        if bias.max() >= 1.2 / len(tracks):
            final[tracks[np.argmax(bias)], e] = True
    return final, best


def fig3a_rows(session: str, final: np.ndarray, best: np.ndarray) -> np.ndarray:
    """``rate_remapping_TRACK_PAIRS.m``, RUN epoch: (new ID, x, y) per cell active in both.

    x is the difference of the 10 cm raw peak rates (track 1 - track 2); y the
    difference of each cell's mean peak instantaneous rate over the local RUN
    replay events it fires in, the spikes binned at 1 ms and filtered with the
    code's unnormalized 100-point ``gausswin``.
    """
    trains, _, _, fields = decoding_inputs(session)
    good = fields["good_place_cells"]
    common = np.intersect1d(
        fields["track"][0]["good_cells"], fields["track"][1]["good_cells"]
    )
    run = authors()[f"{session}/run_events"]
    onset, offset, midpoint = run[:, 0], run[:, 1], run[:, 2]
    times = (onset + offset) / 2
    kernel = gausswin(100)
    peaks, counts = [], []
    for k in (0, 1):
        a, b = fields["track"][k]["window"]
        chosen = np.flatnonzero(final[k] & (times >= a) & (times < b))
        peak = np.zeros((len(chosen), len(common)))
        count = np.zeros_like(peak)
        for row, e in enumerate(chosen):
            n_bins = int(np.ceil((offset[e] - onset[e]) / REPLAY_BIN))
            centres = onset[e] + REPLAY_BIN * (np.arange(n_bins) + 0.5)
            middle = int(np.argmin(np.abs(centres - midpoint[e]))) + 1
            lo, hi, n = {
                1: (onset[e], offset[e], n_bins),
                2: (onset[e], midpoint[e], middle),
                3: (midpoint[e], offset[e], n_bins - middle + 1),
            }[best[k, e]]
            spikes = {
                c: tr[(tr >= lo) & (tr <= hi)]
                for c, tr in zip(good, trains, strict=True)
            }
            everyone = np.concatenate(list(spikes.values()))
            start = min(times[e] - n * REPLAY_BIN / 2, everyone.min()) - 0.1
            stop = max(times[e] + n * REPLAY_BIN / 2, everyone.max()) + 0.1
            edges = start + 0.001 * np.arange(
                int(np.floor((stop - start) / 0.001 + 1e-9)) + 1
            )
            for column, cell in enumerate(common):
                binned = np.histogram(spikes[cell], edges)[0].astype(float)
                peak[row, column] = np.convolve(binned, kernel)[: binned.size].max()
                count[row, column] = len(spikes[cell])
        peaks.append(peak)
        counts.append(count)
    if not (len(peaks[0]) and len(peaks[1])):
        return np.empty((0, 3))
    active = (counts[0].mean(axis=0) != 0) & (counts[1].mean(axis=0) != 0)
    # The mean over the events a cell fires in; 0 where it fires in none (the code).
    means = [
        np.divide(
            p.sum(axis=0),
            (p != 0).sum(axis=0),
            out=np.zeros(p.shape[1]),
            where=(p != 0).sum(axis=0) > 0,
        )
        for p in peaks
    ]
    x = (fields["track"][0]["raw_peak"] - fields["track"][1]["raw_peak"])[common - 1]
    folder = list(authors()["sessions"]).index(session) + 1
    return np.column_stack([1000 * folder + common, x, means[0] - means[1]])[active]


def stable_cells(session: str) -> np.ndarray:
    """``get_stable_cells.m``: peak above 1 Hz in both halves of the laps on both tracks."""
    position, clusters = deposit(session)
    halves = authors()[f"{session}/lap_halves"]
    first = place_fields(position, clusters, 10, windows=[(h[0], h[1]) for h in halves])
    second = place_fields(
        position, clusters, 10, windows=[(h[2], h[3]) for h in halves]
    )
    both = [
        np.flatnonzero(
            (first["track"][k]["peak"] > 1) & (second["track"][k]["peak"] > 1)
        )
        + 1
        for k in (0, 1)
    ]
    fields = decoding_inputs(session)[3]
    common = np.intersect1d(
        fields["track"][0]["good_cells"], fields["track"][1]["good_cells"]
    )
    folder = list(authors()["sessions"]).index(session) + 1
    return 1000 * folder + np.intersect1d(np.intersect1d(*both), common)


def regression(rows: np.ndarray) -> dict[str, float]:
    """Ordinary least squares of y on x, as fitlm: slope, R^2, F and p."""
    from scipy import stats

    fit = stats.linregress(rows[:, 1], rows[:, 2])
    n, r2 = len(rows), fit.rvalue**2
    return {
        "n": n,
        "B": fit.slope,
        "R2": r2,
        "F": r2 / (1 - r2) * (n - 2),
        "p": fit.pvalue,
    }


def detect(session: str, speed: str) -> np.ndarray:
    """``ripple_detection``'s ``tirole_2022`` on the loaded session, (n, 2) bounds.

    Place cells are the code's 2 cm good place cells; the spikes of the other
    units are pooled in one column (the population rate is the same, and a pooled
    column is never a place cell), which keeps memory near 12 GB. ``speed`` is
    ``"package"`` (the recording's speed) or ``"file"`` (``position.v_cm``, nearest
    sample, as ``extract_replay_events.m``).
    """
    from ripple_detection.literature_methods import Recording as Signals
    from ripple_detection.literature_methods import run_method

    from neurospatial.datasets import load
    from neurospatial.datasets._ripple import _speed_on_grid, spike_counts

    rec = load("tirole_2022", session)
    position, clusters = deposit(session)
    good = place_fields(position, clusters, 2)["good_place_cells"]
    samples, times = rec.lfp.fetch(rec.lfp.ripple_channels)
    counts = spike_counts(rec.spike_times, times, sampling_rate=rec.lfp.sampling_rate)
    place = rec.units.index.isin(good)
    multiunit = np.column_stack(
        [counts[:, place], counts[:, ~place].sum(axis=1)]
    ).astype(np.uint16)
    del counts
    if speed == "package":
        on_grid = _speed_on_grid(rec, times)
    else:
        on_grid = nearest(position["t"], position["v_cm"], times)
    signals = Signals.from_arrays(
        times,
        rec.lfp.sampling_rate,
        lfps=samples,
        multiunit=multiunit,
        speed=on_grid,
        place_cells=np.r_[np.ones(place.sum(), bool), False],
    )
    return run_method("tirole_2022", signals)[["start_time", "end_time"]].to_numpy()


def overlapping(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Whether each row of ``a`` overlaps a row of ``b`` (closed intervals)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), bool)
    b = b[np.argsort(b[:, 0])]
    latest_end = np.maximum.accumulate(b[:, 1])
    before = np.searchsorted(b[:, 0], a[:, 1], side="right") - 1
    return (before >= 0) & (latest_end[np.maximum(before, 0)] >= a[:, 0])


# ---------------------------------------------------------------- the checks


def test_ported_segment_rule_is_the_codes_on_its_own_p_values():
    """The ported verdicts reproduce the authors' final significant events exactly."""
    ref = authors()
    for session in SESSIONS:
        p = ref[f"{session}/p_values"]
        final, _ = choose_tracks(
            authors_segments(p), ref[f"{session}/scores"][..., :], np.ones(p.shape[:3])
        )
        gate = ref[f"{session}/run_events"][:, 3] >= RIPPLE_GATE
        theirs = ref[f"{session}/significant"]
        # The code chooses between tracks by the summed posterior, which only
        # events significant on two tracks need: compare the others exactly.
        single = authors_segments(p).any(axis=-1).sum(axis=0) <= 1
        np.testing.assert_array_equal(final[:, gate & single], theirs[:, gate & single])
        assert not theirs[:, ~gate].any()


def detection_agreement(speed: str) -> dict[str, tuple[int, int, int, int]]:
    """Per session and in all: (ours, authors', authors' that ours overlap, ours they overlap).

    The authors' candidates are those of ripple power 3 SD or more, the gate
    ``number_of_significant_replays.m`` applies; the detector applies it itself.
    """
    table: dict[str, tuple[int, int, int, int]] = {}
    for session in SESSIONS:
        mine = ours()[f"{session}/detected_{speed}"]
        candidates = authors()[f"{session}/candidates"]
        theirs = candidates[candidates[:, 2] >= RIPPLE_GATE, :2]
        table[session] = (
            len(mine),
            len(theirs),
            int(overlapping(theirs, mine).sum()),
            int(overlapping(mine, theirs).sum()),
        )
    table["all"] = tuple(int(sum(row[i] for row in table.values())) for i in range(4))
    return table


def judged_pairs(authors_floor: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Each (track, gated event) verdict, the authors' and ours, over the sessions."""
    prefix = "authors_floor/" if authors_floor else ""
    theirs, mine = [], []
    for session in SESSIONS:
        key = f"{prefix}{session}/verdict"
        if key not in ours():
            continue
        events = ours()[f"{session}/judged_events"]
        p = authors()[f"{session}/p_values"][:, events]
        theirs.append(authors_segments(p).any(axis=-1).ravel())
        mine.append(ours()[key].astype(bool).any(axis=-1).ravel())
    return np.concatenate(theirs), np.concatenate(mine)


def shuffle_noise_agreement(n_draws: int = 200, seed: int = 2026) -> np.ndarray:
    """The share of one run's significant pairs that a second run also finds, from noise alone.

    Two runs whose shuffles differ but whose events and decoding are the same:
    each of the 9 p-values of a pair (3 nulls x whole and halves) is redrawn as
    Binomial(1000, p) / 1000 from the authors' p-value, twice, and the two
    redraws' verdicts are compared. One value per pair of draws.
    """
    rng = np.random.default_rng(seed)
    draws = []
    for session in SESSIONS:
        events = gated(session)
        p = authors()[f"{session}/p_values"][:, events]
        missing = np.isnan(p)
        session_draws = []
        for _ in range(n_draws):
            redrawn = rng.binomial(1000, np.nan_to_num(p)) / 1000
            session_draws.append(
                authors_segments(np.where(missing, np.nan, redrawn))
                .any(axis=-1)
                .ravel()
            )
        draws.append(np.array(session_draws))
    verdicts = np.concatenate(draws, axis=1)
    first, second = verdicts[0::2], verdicts[1::2]
    return (first & second).sum(axis=1) / first.sum(axis=1)


def authors_decode(counts: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """``reconstruct`` of ``bayesian_decoding.m``, unnormalized: (n_bins, n_time).

    A reference for the score check only. Rates below 10^(log2(N) - log2(400))
    are raised to it, and so is each rate raised to its spike count.
    """
    floor = 10 ** (np.log2(rates.shape[0]) - np.log2(400))
    fields = np.maximum(rates, floor)
    out = np.empty((fields.shape[1], counts.shape[1]))
    for j in range(counts.shape[1]):
        powered = np.maximum(fields ** counts[:, j][:, None], floor)
        out[:, j] = powered.prod(axis=0) * np.exp(-REPLAY_BIN * fields.sum(axis=0))
    return out


def authors_weighted_correlation(posterior: np.ndarray) -> float:
    """``weighted_correlation.m`` on bin numbers: abs(r), (n_bins, n_time)."""
    position = np.arange(1, posterior.shape[0] + 1)[:, None]
    time = np.arange(1, posterior.shape[1] + 1)[None, :]
    total = posterior.sum()
    mean_t = (posterior * time).sum() / total
    mean_x = (posterior * position).sum() / total
    cov = (posterior * (time - mean_t) * (position - mean_x)).sum() / total
    var_t = (posterior * (time - mean_t) ** 2).sum() / total
    var_x = (posterior * (position - mean_x) ** 2).sum() / total
    return float(abs(cov / np.sqrt(var_t * var_x)))


def whole_event_scores(session: str, route: str) -> tuple[np.ndarray, np.ndarray]:
    """Each (track, gated event) whole-event score, ours and the authors' stored one.

    ``route`` is ``"package"`` (``decode_jointly`` on the package's maps, time
    bins without spikes zeroed by their own counts, ``weighted_correlation``), or
    ``"authors"`` (:func:`authors_decode`, and the columns zeroed where no spike
    falls inside the event's [onset, offset], as ``replay_scoring.m`` does with
    the spikes ``replay_decoding.m`` stored).
    """
    from neurospatial.decoding import decode_jointly
    from neurospatial.replay import weighted_correlation

    trains, names, rates, _ = decoding_inputs(session)
    envs = {name: tirole_2022.track_environment() for name in names}
    run = authors()[f"{session}/run_events"]
    mine, theirs = [], []
    for event in gated(session):
        onset, offset = run[event, :2]
        edges = onset + REPLAY_BIN * np.arange(
            int(np.ceil((offset - onset) / REPLAY_BIN)) + 1
        )
        counts = np.array([np.histogram(train, edges)[0] for train in trains])
        if counts.sum() == 0:
            continue
        if route == "package":
            joint = decode_jointly(
                envs, counts.T, rates, dt=REPLAY_BIN, times=edges[:-1]
            )
            shares = {name: joint.joint(name).T for name in names}
            spiking = counts.sum(axis=0) > 0
        else:
            raw = {
                name: authors_decode(counts, np.nan_to_num(rates[name]))
                for name in names
            }
            total = sum(post.sum(axis=0) for post in raw.values())
            shares = {name: post / total for name, post in raw.items()}
            inside = np.concatenate([t[(t >= onset) & (t <= offset)] for t in trains])
            spiking = np.histogram(inside, edges)[0] > 0
        for k, name in enumerate(names):
            posterior = shares[name] * spiking[None, :]
            if posterior.sum() == 0 or len(edges) - 1 < 5:
                continue
            if route == "package":
                score = weighted_correlation(
                    envs[name], posterior.T, edges[:-1], row_normalization="none"
                )
                mine.append(abs(score.score))
            else:
                mine.append(authors_weighted_correlation(posterior))
            theirs.append(authors()[f"{session}/scores"][k, event, 0])
    return np.array(mine, float), np.array(theirs, float)


#: Per speed, summed over the nine sessions: (ours, authors' with ripple >= 3 SD,
#: authors' overlapped by ours, ours overlapped by the authors').
DETECTION = {
    "package": (16636, 19558, 15731, 15733),
    "file": (19660, 19558, 19550, 19554),
}
#: (pairs, authors' significant, ours, both) over the nine sessions.
JUDGING = (7002, 1658, 1567, 1490)
#: The same on the default session, on the package's maps and on the floored ones.
JUDGING_DEFAULT = {
    "package": (806, 204, 199, 191),
    "authors_floor": (806, 204, 206, 197),
}


def test_detection_agrees_with_the_authors_candidates_both_ways():
    """With the file's speed, every authors' candidate is found; with the package's, fewer.

    The detector is the paper's rule on the same single units as the Octave run
    (no multi-unit clusters are deposited). The speed decides most of the gap:
    the code's speed is position.v_cm, smoother than the recording's and below
    5 cm/s in 85% of the default session's samples against 58%, so with the
    recording's speed more events fail the median-speed gate of 5 cm/s.
    With the file's speed the gaps sit at the rules' bounds. The events the
    authors lack are mostly exactly 100 ms long (the code computes the duration
    in floating point and drops some at the bound), or have ripple power the
    detector recomputes from the raw channel just above 3 SD where the authors'
    stored z-score is just below. Of the 8 authors' events missed, 5 have
    ripple power of 3.00 to 3.02 SD, 1 a multi-unit peak of 3.00 SD and 2
    exactly 5 active place cells.
    """
    for speed, expected in DETECTION.items():
        table = detection_agreement(speed)
        assert table["all"] == expected, (speed, table)


def test_judging_agrees_with_the_authors_within_shuffle_noise():
    """``TIROLE_2022`` on the package's maps against the authors' verdicts, both ways.

    A pair is a gated RUN candidate on one track; it is replay when the whole
    event or a half passes. Of the authors' 1,658 significant pairs we find
    1,490 (89.9%), and 95.1% of our 1,567 are theirs. Two runs that differ only
    in their shuffles agree on 96.7% to 98.0% of significant pairs
    (``shuffle_noise_agreement``), so shuffles explain most of the gap on our
    side and about a quarter of it on theirs. The rest is the decoding: neurospatial's
    decoder has none of the code's rate floors, which make an unlikely bin less
    unlikely, so the package calls fewer events replay; its maps leave bins the
    rat never crossed out; and it zeroes a time bin by its own spikes, not by
    the spikes inside the event's bounds. After one track is chosen per event,
    the code's rule ported, 1,296 of the authors' 1,409 significant events are
    ours (92.0%) and 94.0% of our 1,378 are theirs.
    """
    theirs, mine = judged_pairs()
    assert (
        theirs.size,
        int(theirs.sum()),
        int(mine.sum()),
        int((theirs & mine).sum()),
    ) == JUDGING
    noise = shuffle_noise_agreement()
    low, high = np.percentile(noise, [2.5, 97.5])
    assert (round(low, 3), round(high, 3)) == (0.967, 0.98)
    assert (theirs & mine).sum() / theirs.sum() < low
    assert final_verdicts() == FINAL


def final_verdicts() -> tuple[int, int, int, int]:
    """(pairs, authors' final significant, ours, both) after one track per event."""
    ref = authors()
    total = np.zeros(4, int)
    for session in SESSIONS:
        events = ours()[f"{session}/judged_events"]
        p = ref[f"{session}/p_values"]
        n_tracks, n_events = p.shape[:2]
        mine = np.zeros((n_tracks, n_events, 3), bool)
        scores = np.full((n_tracks, n_events, 3), np.nan)
        sums = np.full((n_tracks, n_events, 3), np.nan)
        mine[:, events] = ours()[f"{session}/verdict"].astype(bool)
        scores[:, events] = ours()[f"{session}/score"]
        sums[:, events] = ours()[f"{session}/sums"]
        final, _ = choose_tracks(mine, scores, sums)
        theirs = ref[f"{session}/significant"][:, events].ravel()
        chosen = final[:, events].ravel()
        total += [theirs.size, theirs.sum(), chosen.sum(), (theirs & chosen).sum()]
    return tuple(int(value) for value in total)


#: (pairs, authors' significant, ours, both) after one track is chosen per event.
FINAL = (7002, 1409, 1378, 1296)


def test_the_codes_floor_changes_few_verdicts_on_the_default_session():
    """The default session judged on the package's maps and on the code's floored maps."""
    ref = authors()
    events = ours()[f"{DEFAULT}/judged_events"]
    theirs = (
        authors_segments(ref[f"{DEFAULT}/p_values"][:, events]).any(axis=-1).ravel()
    )
    for route, expected in JUDGING_DEFAULT.items():
        prefix = "authors_floor/" if route == "authors_floor" else ""
        mine = ours()[f"{prefix}{DEFAULT}/verdict"].astype(bool).any(axis=-1).ravel()
        assert (
            theirs.size,
            int(theirs.sum()),
            int(mine.sum()),
            int((theirs & mine).sum()),
        ) == expected


@pytest.mark.realdata
@pytest.mark.parametrize("session", SESSIONS)
def test_place_cells_and_stable_cells_are_the_codes(session):
    """The port of calculate_place_fields.m and get_stable_cells.m gives the code's cells."""
    position, clusters = deposit(session)
    for width in (2, 10):
        good = place_fields(position, clusters, width)["good_place_cells"]
        np.testing.assert_array_equal(
            good, authors()[f"{session}/good_place_cells_{width}cm"]
        )
    stable = authors()["stable_cells_laps"]
    folder = list(authors()["sessions"]).index(session) + 1
    np.testing.assert_array_equal(
        stable_cells(session), stable[stable // 1000 == folder]
    )


@pytest.mark.realdata
def test_scores_match_the_codes_and_the_package_differs_by_its_decoding():
    """The code's decoding reproduces its scores; the package's differs as documented.

    On the default session's gated candidates, the reference route (the code's
    floors, time bins zeroed by the spikes inside the event) gives the authors'
    stored weighted correlation within 1e-6 for 99.75% of (track, event) pairs
    and within 0.0021 for all. The package's own route differs by the floors and
    by the time bins past the event's end, which the code's 20 ms bins reach.
    """
    mine, theirs = whole_event_scores(DEFAULT, "authors")
    difference = np.abs(mine - theirs)
    assert len(difference) == 806
    assert round(float(np.mean(difference < 1e-6)), 4) == 0.9975
    assert difference.max() < 0.0021
    mine, theirs = whole_event_scores(DEFAULT, "package")
    # One posterior on a single position has no spread, so no correlation.
    defined = np.isfinite(mine)
    assert int((~defined).sum()) == 1
    mine, theirs = mine[defined], theirs[defined]
    difference = np.abs(mine - theirs)
    assert round(float(np.mean(difference < 0.01)), 4) == 0.7255
    assert round(float(np.corrcoef(mine, theirs)[0, 1]), 4) == 0.9668


@pytest.mark.realdata
def test_detection_on_the_default_session_is_the_committed_one():
    pytest.importorskip("ripple_detection.literature_methods")
    for speed in ("package", "file"):
        np.testing.assert_allclose(
            detect(DEFAULT, speed), ours()[f"{DEFAULT}/detected_{speed}"], atol=1e-9
        )


@pytest.mark.realdata
def test_judging_on_the_default_session_is_the_committed_one():
    """The first ten gated candidates, with their seeds, give the committed verdicts."""
    events = ours()[f"{DEFAULT}/judged_events"][:10]
    for floor, prefix in ((False, ""), (True, "authors_floor/")):
        result = judge_events(DEFAULT, events, authors_floor=floor)
        for key in ("score", "p", "verdict"):
            np.testing.assert_allclose(
                result[key], ours()[f"{prefix}{DEFAULT}/{key}"][:, :10], atol=1e-12
            )


@pytest.mark.realdata
def test_fig3a_run_regression():
    """Fig. 3A: the Python chain on the authors' verdicts, then on ours.

    On the authors' own significant events the chain gives the Octave run's
    regression exactly (227 stable cells, B = 0.0502, R^2 = 0.430, F = 169.6), so
    the port is the code's. On ours it gives ``WORKED_RESULT["ours"]``.
    """
    ref = authors()
    rows_theirs, rows_ours = [], []
    for session in SESSIONS:
        p = ref[f"{session}/p_values"]
        segments = authors_segments(p)
        _, best = choose_tracks(
            segments, ref[f"{session}/scores"], np.ones(p.shape[:3])
        )
        rows_theirs.append(fig3a_rows(session, ref[f"{session}/significant"], best))
        events = ours()[f"{session}/judged_events"]
        n_tracks, n_events = p.shape[:2]
        mine = np.zeros((n_tracks, n_events, 3), bool)
        scores = np.full((n_tracks, n_events, 3), np.nan)
        sums = np.full((n_tracks, n_events, 3), np.nan)
        mine[:, events] = ours()[f"{session}/verdict"].astype(bool)
        scores[:, events] = ours()[f"{session}/score"]
        sums[:, events] = ours()[f"{session}/sums"]
        final, best = choose_tracks(mine, scores, sums)
        rows_ours.append(fig3a_rows(session, final, best))
    stable = ref["stable_cells_laps"]
    theirs = np.vstack(rows_theirs)
    theirs = theirs[np.isin(theirs[:, 0], stable)]
    fit = regression(theirs)
    assert (fit["n"], round(fit["B"], 4), round(fit["R2"], 3), round(fit["F"], 1)) == (
        227,
        0.0502,
        0.43,
        169.6,
    )
    reference = ref["fig3a_run_stable"]
    np.testing.assert_array_equal(np.sort(theirs[:, 0]), np.sort(reference[:, 0]))
    mine = np.vstack(rows_ours)
    mine = mine[np.isin(mine[:, 0], stable)]
    fit = regression(mine)
    record = tirole_2022.WORKED_RESULT
    assert (fit["n"], round(fit["B"], 4), round(fit["R2"], 3), round(fit["F"], 1)) == (
        225,
        0.0537,
        0.461,
        190.8,
    )
    assert round(fit["B"], 4) == record["ours"]
    assert abs(fit["B"] - 0.05) <= 0.01
    assert abs(fit["R2"] - 0.41) <= 0.1
    assert fit["p"] < 0.001
    assert record["attempted"] is None


def _judge_session(session: str, authors_floor: bool = False) -> dict[str, np.ndarray]:
    """Every gated event of one session, eight events per process."""
    from concurrent.futures import ProcessPoolExecutor

    events = gated(session)
    chunks = np.array_split(events, max(1, len(events) // 8))
    n = len(chunks)
    with ProcessPoolExecutor(4) as pool:
        parts = list(
            pool.map(
                judge_events, [session] * n, chunks, [1000] * n, [authors_floor] * n
            )
        )
    return {key: np.concatenate([p[key] for p in parts], axis=1) for key in parts[0]}


def write_ours(path: Path = DATA / "ours.npz") -> None:
    """Recompute ``ours.npz``: detection with both speeds, then judging, per session."""
    table: dict[str, np.ndarray] = {}
    for session in SESSIONS:
        for speed in ("package", "file"):
            table[f"{session}/detected_{speed}"] = detect(session, speed)
        for key, values in _judge_session(session).items():
            table[f"{session}/{key}"] = values
        table[f"{session}/judged_events"] = gated(session)
    for key, values in _judge_session(DEFAULT, authors_floor=True).items():
        table[f"authors_floor/{DEFAULT}/{key}"] = values
    table["provenance"] = np.array(OURS_PROVENANCE)
    np.savez_compressed(path, **table)


OURS_PROVENANCE = (
    "Per session: detected_package and detected_file, the [start, end] of each "
    "candidate ripple_detection.literature_methods.tirole_2022 (ripple_detection "
    "0eddae64) finds on the loaded session with the recording's speed and with "
    "the file's position.v_cm (detect()); judged_events, the indices of the "
    "authors' RUN candidates with ripple power >= 3 SD; score, p, verdict and sums "
    "(track, event, whole/first/second) from judge_events() with TIROLE_2022, "
    "1000 shuffles, rng seeded [session index, event index], on the package's "
    "rate maps (unvisited bins NaN); authors_floor/rat3_session2/*, the same on "
    "the code's floored maps, a reference for the floor's effect. Computed on "
    "2026-11-11 by a script running the same steps in parallel; write_ours() in "
    "tests/datasets/test_tirole_2022_replay.py regenerates it, and the realdata "
    "tests recompute the default session's detection and first judged events."
)


if __name__ == "__main__":
    write_ours()
